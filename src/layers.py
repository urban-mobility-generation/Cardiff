from typing import Callable

from einops import rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from src.helpers import default, exists


class Attention(nn.Module):
    """
    Multi-headed attention
    """

    def __init__(
            self,
            dim: int,
            *,
            dim_head: int = 64,
            heads: int = 8,
            context_dim: int = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor = None, mask: torch.tensor = None,
                attn_bias: torch.tensor = None) -> torch.tensor:

        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Block(nn.Module):
    """
    Sub-block for :class:`.ResnetBlock`. GroupNorm/Identity, SiLU, and Conv1D in sequence
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            groups: int = 8,
            norm: bool = True
    ):
        """
        :param dim: Input number of channels.
        :param dim_out: Output number of channels.
        :param groups: Number of GroupNorm groups.
        :param norm: Whether to use GroupNorm or Identity.
        """
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(dim, dim_out, 3, padding=1)

    def forward(self, x: torch.tensor, scale_shift: tuple[torch.tensor, torch.tensor] = None) -> torch.tensor:
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


def ChanFeedForward(dim: int,
                    mult: int = 2) -> torch.nn.Sequential:
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv1d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv1d(hidden_dim, dim, 1, bias=False)
    )


class ChanLayerNorm(nn.Module):
    """
    LayerNorm for :class:`.ChanFeedForward`.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class CrossAttention(nn.Module):
    """
    Multi-headed cross attention.
    """
    def __init__(
            self,
            dim: int,
            *,
            context_dim: int = None,
            dim_head: int = 64,
            heads: int = 8,
            norm_context: bool = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def Downsample(dim: int, dim_out: int = None) -> torch.nn.Conv1d:
    dim_out = default(dim_out, dim)
    return nn.Conv1d(dim, dim_out, kernel_size=4, stride=2, padding=1)


class Identity(nn.Module):
    """
    Identity module - forward pass returns input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Parallel(nn.Module):
    """
    Passes input through parallel functions and then sums the result.
    """

    def __init__(self, *fns: tuple[Callable, ...]):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x: torch.tensor) -> torch.tensor:
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


class Residual(nn.Module):
    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return self.fn(x, **kwargs) + x


class ResnetBlock(nn.Module):
    """
    ResNet Block.
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            *,
            cond_dim: int = None,
            time_cond_dim: int = None,
            groups: int = 8,
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None
        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                'b c l',
                'b l c',
                CrossAttention(
                    dim=dim_out,
                    context_dim=cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x: torch.tensor, time_emb: torch.tensor = None, cond: torch.tensor = None) -> torch.tensor:
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


class SinusoidalPosEmb(nn.Module):
    '''
    Generates sinusoidal positional embedding tensor.
    '''

    def __init__(self, dim: int):
        """
        :param dim: Dimensionality of the embedding space
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerBlock(nn.Module):
    """
    Responsible for applying attention at the end of a chain of
    :class:`.ResnetBlock`s at each layer in the U-Met.
    """
    def __init__(
            self,
            dim: int,
            *,
            heads: int = 8,
            dim_head: int = 32,
            ff_mult: int = 2,
            context_dim: int = None
    ):
        super().__init__()
        self.attn = EinopsToAndFrom('b c l', 'b l c',
                                    Attention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim))
        self.ff = ChanFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x: torch.tensor, context: torch.tensor = None) -> torch.tensor:
        x = self.attn(x, context=context) + x
        x = self.ff(x) + x
        return x


def Upsample(dim: int, dim_out: int = None) -> torch.nn.Sequential:
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, dim_out, 3, padding=1)
    )


class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(144, hidden_dim)
        self.sid_embedding = nn.Embedding(1015, hidden_dim)
        self.eid_embedding = nn.Embedding(1015, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim * 3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        # Continuous attributes
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 6].long(), attr[:, 7].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed

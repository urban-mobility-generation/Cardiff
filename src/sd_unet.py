"""
---
revised based on
 Annotated PyTorch implementation/tutorial of the U-Net in stable diffusion.
 and
 [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
---
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from src.sd_unet_attention import SpatialTransformer


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


class Unet(nn.Module):
    """
    ## U-Net model
    """
    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond: int = 128,
            lowres_cond: bool = False
        ):

        super().__init__()
        self.channels = channels
        levels = len(channel_multipliers)
        # Maps time to time hidden state
        time_cond_dim = channels * 2 * (2 if lowres_cond else 1)
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        )
        self.lowres_cond = lowres_cond
        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(channels),
                nn.Linear(channels, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

        self.attr_embed = AttrBlock(embedding_dim=time_cond_dim)
        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv1d(in_channels,
                      # in_channels if not lowres_cond else in_channels * 2,
                      channels,
                      kernel_size=3,
                      padding=1)
        ))

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, time_cond_dim, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, time_cond_dim),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, time_cond_dim),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), time_cond_dim, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))

                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def _generate_t_embs(
            self,
            time: torch.tensor,
            lowres_noise_times: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:

        t = self.to_time_hiddens(time)

        # If lowres conditioning, add lowres time conditioning to `t`
        if self.lowres_cond:
            lowres_t = self.to_lowres_time_hiddens(lowres_noise_times)
            t = t + lowres_t
        return t


    def forward(self,
                x: torch.Tensor,
                time_steps: torch.Tensor,
                attr_embeds: Optional[torch.Tensor] = None,
                lowres_cond_embed: torch.tensor = None,
                lowres_noise_times: torch.tensor = None,
                cond_drop_prob: float = 0.
                ):

        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings + attribute embeddings
        # t_emb = self.to_time_hiddens(time_steps)
        t_emb = self._generate_t_embs(time_steps, lowres_noise_times)
        if attr_embeds is not None:
            attr_emb = self.attr_embed(attr_embeds)
            t_emb = t_emb + attr_emb

        # get low-level condition embeddings
        cond = lowres_cond_embed

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    """
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """
    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.op = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()

        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    #  Group normalization
    """
    return GroupNorm32(32, channels)


class AttrBlock(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(AttrBlock, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(2, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(144, hidden_dim)
        self.sid_embedding = nn.Embedding(1015, hidden_dim)
        self.eid_embedding = nn.Embedding(1015, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim * 3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        # Continuous attributes
        continuous_attrs = attr[:, 1:3]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 3].long(), attr[:, 4].long()

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



class Super(Unet):
    """
    Super-Resolution U-Net.
    """
    defaults = dict(
        in_channels=2,
        out_channels=2,
        channels=128,
        n_res_blocks=2,
        attention_levels=[0,1,2],
        channel_multipliers=[1, 1, 2, 2],
        n_heads=4,
        tf_layers=1,
        d_cond=128,
        lowres_cond=True
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Super.defaults, **kwargs})



class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )
        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        b, c, h = x.shape
        x_in = x
        # Normalize
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 1).view(b, h, c)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = x.view(b, h, c).permute(0, 2, 1)
        x = self.proj_out(x)
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        x = self.ff(self.norm3(x)) + x
        #
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        try:
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """
        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """
        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        # Compute attention
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, :self.d_head]
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """
        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(*out.shape[:2], -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

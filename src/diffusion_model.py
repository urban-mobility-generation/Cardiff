import torch
import torch.nn.functional as F
from torch import nn

from .helpers import default, log, extract


class GaussianDiffusion(nn.Module):

    def __init__(
            self,
            *,
            timesteps: int
    ):
        """
        :param timesteps: Number of timesteps in the Diffusion Process.
        """
        super().__init__()

        # Timesteps < 20 => scale > 50 => beta_end > 1 => alphas[-1] < 0 => sqrt_alphas_cumprod[-1] is NaN
        assert not timesteps < 20,  f'timsteps must be at least 20'
        self.num_timesteps = timesteps

        # Create variance schedule.
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        # Diffusion model constants/buffers. See https://arxiv.org/pdf/2006.11239.pdf
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # $\prod_{i=1}^t \alpha_i$
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # register buffer helper function
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # Register variance schedule related buffers
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Buffer for diffusion calculations q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # Posterior variance:
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Clipped because posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps=1e-20))

        # Buffers for calculating the q_posterior mean $\~{\mu}$.
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _get_times(self, batch_size: int, noise_level: float, *, device: torch.device) -> torch.tensor:
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device=device, dtype=torch.long)

    def _sample_random_times(self, batch_size: int, *, device: torch.device) -> torch.tensor:
        """
        Randomly sample `batch_size` timestep values uniformly from [0, 1, ..., `self.num_timesteps`]
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def _get_sampling_timesteps(self, batch: int, *, device: torch.device) -> list[torch.tensor]:
        time_transitions = []

        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))

        return time_transitions

    def q_posterior(self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor) -> tuple[torch.tensor,
                                                                                              torch.tensor,
                                                                                              torch.tensor]:
        """
        Calculates q_posterior parameters given a starting image
        :code:`x_start` (x_0) and a noised image :code:`x_t`.
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Extract the value corresponding to the current time from the buffers, and then reshape to (b, 1, 1, 1)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start: torch.tensor, t: torch.tensor, noise: torch.tensor = None) -> torch.tensor:

        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return noised

    def predict_start_from_noise(self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Given a noised image and its noise component, calculated the unnoised image :code:`x_0`.
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # for ddim
    def _get_ddim_sampling_timesteps(self, batch: int, *, device: torch.device, step: int) -> list[torch.tensor]:
        time_transitions = []

        skipped_time_steps = range(0, self.num_timesteps, step)

        for i in reversed(skipped_time_steps):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))

        return time_transitions

    # refer to https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
    def compute_alpha(self, t):
        beta = torch.cat([torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
        return a

    def ddim_sample(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                    eps: torch.Tensor, eta: float = 0.0) -> torch.Tensor:

        # alpha_bar_t
        # alpha_bar_t = extract(self.alphas_cumprod, t+1, x.shape)
        # # t-1
        # alpha_bar_prev = extract(self.alphas_cumprod, t_prev+1, x.shape)

        alpha_bar_t = self.compute_alpha(t)
        alpha_bar_prev = self.compute_alpha(t_prev)

        pred_x0 = torch.sqrt(1. / alpha_bar_t) * x - torch.sqrt(1. / alpha_bar_t - 1)*eps

        # sigma_t
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
            1 - alpha_bar_t / alpha_bar_prev)
        # sample noise
        noise = torch.randn_like(x) if eta > 0.0 else 0.
        # update x_{t-1}
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + \
                 torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps + \
                 sigma_t * noise

        return x_prev

    def ddim_sample_from_xstart(self, x, t, t_prev, x_start, eta=0.0):
        alpha_bar_t = self.compute_alpha(t)
        alpha_bar_prev = self.compute_alpha(t_prev)

        eps = (x - torch.sqrt(alpha_bar_t) * x_start) / torch.sqrt(1 - alpha_bar_t)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        noise = torch.randn_like(x) if eta > 0.0 else 0.
        x_prev = torch.sqrt(alpha_bar_prev) * x_start + \
                 torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps + \
                 sigma_t * noise
        return x_prev

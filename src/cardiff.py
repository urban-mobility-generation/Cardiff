from typing import List, Tuple, Union, Callable, Literal
from tqdm import tqdm
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T

from einops import rearrange, repeat
from einops_exts import check_shape

from src.dit import DiT
from src.sd_unet import Unet
from src.helpers import (cast_tuple,
                         default,
                         identity,
                         exists,
                         module_device,
                         maybe,
                         eval_decorator,
                         null_context)
from transformers.modeling_outputs import BaseModelOutput
from src.diffusion_model import GaussianDiffusion

def structure_loss_from_logits(logits: torch.Tensor, adjacency_map: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, L, V), decoder output logits
    adjacency_map: dict[str, set[str]], edge_id string adjacency
    vocab: list of vocab token strings, where vocab[i] is the str edge_id at index i
    """
    pred_ids = logits.argmax(dim=-1)  # (B, L)
    B, L = pred_ids.shape
    pred_ids = logits.argmax(dim=-1)  # (B, L),
    B, L = pred_ids.shape
    src_ids = pred_ids[:, :-1]  # (B, L-1)
    tgt_ids = pred_ids[:, 1:]   # (B, L-1)
    is_invalid = ~adjacency_map[src_ids, tgt_ids]
    structure_loss = is_invalid.float().mean()
    return torch.tensor(structure_loss)


class Cardiff(nn.Module):
    def __init__(
            self,
            denoisers,
            *,
            first_stage_dim,
            first_stage_len,
            second_stage_len,
            attr_embed_dim: int = None,
            channels: int = 2,
            timesteps: Union[int, List[int], Tuple[int, ...]] = 1000,
            cond_drop_prob: float = 0.1,
            loss_type: Literal["l1", "l2", "huber"] = 'l2',
            lowres_sample_noise_level: float = 0.2,
            only_train_unet_number: int = None
    ):

        super().__init__()

        # Set loss
        self.tokenizer_vocab = None
        self.adjacency_matrix = None
        self.structure_loss_percent = 0
        self.structure_loss_weight = 0.01
        self.lm = None

        self.loss_type = loss_type
        self.loss_fn = self._set_loss_fn(loss_type)

        self.channels = channels

        denoisers = cast_tuple(denoisers)
        num_denoisers = len(denoisers)

        # input size and dim
        self.first_stage_dim = first_stage_dim
        self.first_stage_len = first_stage_len
        self.second_stage_len = second_stage_len

        # Create noise schedulers for each stage
        self.noise_schedulers = self._make_noise_schedulers(num_denoisers, timesteps)
        # augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusion(timesteps=timesteps)

        self.lowres_sample_noise_level = lowres_sample_noise_level

        self.unet_being_trained_index = -1
        self.only_train_unet_number = only_train_unet_number

        # Cast the relevant hyperparameters to the input Unets, ensuring that the first Unet does not condition on
        self.denoisers = nn.ModuleList([])
        for ind, denoiser in enumerate(denoisers):
            self.denoisers.append(denoiser)

        # unet input size for sampling
        self.sample_sizes = cast_tuple([self.first_stage_len, self.second_stage_len]) # 需要修改
        self.sample_channels = cast_tuple(self.channels, num_denoisers)

        # Classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # one temp parameter for keeping track of device
        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)
        self.to(next(self.denoisers.parameters()).device)

    @property
    def device(self) -> torch.device:
        return self._temp.device

    @staticmethod
    def _set_loss_fn(loss_type: str) -> Callable:
        """
        :param loss_type: Type of loss to use. Either 'l1', 'l2', or 'huber'
        """
        # loss
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        return loss_fn

    def set_lm(self, lm, adjacency_matrix, tokenizer_vocab, structure_loss_percent=0.0, structure_loss_weight=0.01):
        self.lm=lm
        self.adjacency_matrix = adjacency_matrix.to(self.device)
        self.tokenizer_vocab = tokenizer_vocab
        self.structure_loss_percent = structure_loss_percent
        self.structure_loss_weight = structure_loss_weight

    @staticmethod
    def _make_noise_schedulers(
            num_denoisers: int,
            timesteps: Union[int, List[int], Tuple[int, ...]]
    ) -> Tuple[GaussianDiffusion, ...]:
        # determine noise schedules per denoiser
        timesteps = cast_tuple(timesteps, num_denoisers)
        # construct noise schedulers
        noise_schedulers = nn.ModuleList([])
        for timestep in timesteps:
            noise_scheduler = GaussianDiffusion(timesteps=timestep)
            noise_schedulers.append(noise_scheduler)
        return noise_schedulers

    def _reset_denoisers_all_one_device(self, device: torch.device = None):
        """
        Creates a ModuleList out of all Unets and places it on one device.
        """
        device = default(device, self.device)
        self.denoisers = nn.ModuleList([*self.denoisers])
        self.denoisers.to(device)
        # Resets relevant attribute to specify that no Unet is being trained at the moment
        self.unet_being_trained_index = -1

    def state_dict(self, *args, **kwargs):

        self._reset_denoisers_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self._reset_denoisers_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    @contextmanager
    def _one_unet_in_gpu(self, denoisers_number, denoiser):
        """
        for sampling
        """
        assert exists(denoisers_number) ^ exists(denoiser)
        if exists(denoisers_number):
            denoiser = self.denoisers[denoisers_number - 1]
        # Store which device each UNet is on, place them all on CPU except the specified one
        devices = [module_device(denoiser) for denoiser in self.denoisers]
        self.denoisers.cpu()
        denoiser.to(self.device)
        yield
        # Restore all UNets back to their original devices
        for denoiser, device in zip(self.denoisers, devices):
            denoiser.to(device)

    # ddpm
    def _p_mean_variance(self,
                         unet: Unet,
                         x: torch.tensor,
                         t: torch.tensor,
                         *,
                         noise_scheduler: GaussianDiffusion,
                         text_embeds: torch.tensor = None,
                         lowres_cond_img: torch.tensor = None,
                         lowres_noise_times: torch.tensor = None,
                         cond_scale: float = 1.,
                         model_output: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Predicts noise component of `x` with `unet`
        """
        # Get the prediction from the base unet
        if lowres_cond_img:
            pred = default(model_output, lambda: unet.forward_with_cond_scale(x,
                                                                              t,
                                                                              attr_embeds=text_embeds,
                                                                              cond_scale=cond_scale,
                                                                              lowres_cond_img=lowres_cond_img,
                                                                              lowres_noise_times=lowres_noise_times))

        else:
            pred = default(model_output, lambda: unet.forward(x,
                                                              t,
                                                              attr_embeds=text_embeds))

        # Calculate the x_0 from the noise
        x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        # Return the forward process posterior parameters given the predicted x_start
        return noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)

    @torch.no_grad()
    def _p_sample(self,
                  unet: Unet,
                  x: torch.tensor,
                  t: torch.tensor,
                  *,
                  noise_scheduler: GaussianDiffusion,
                  text_embeds: torch.tensor = None,
                  lowres_cond_img: torch.tensor = None,
                  lowres_noise_times: torch.tensor = None,
                  cond_scale: float = 1.
                  ) -> torch.tensor:
        """
        _p_sample： q(x_{t-1}|x_t)
        """
        b, *_, device = *x.shape, x.device
        # Calculate sampling distribution parameters
        model_mean, _, model_log_variance = self._p_mean_variance(unet, x=x, t=t,
                                                                  noise_scheduler=noise_scheduler,
                                                                  text_embeds=text_embeds,
                                                                  cond_scale=cond_scale,
                                                                  lowres_cond_img=lowres_cond_img,
                                                                  lowres_noise_times=lowres_noise_times)
        noise = torch.randn_like(x)
        # Don't denoise when t == 0
        is_last_sampling_timestep = (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # Calculate sample from posterior distribution. See
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def _p_sample_ddim(self,
                       unet: Unet,
                       x: torch.tensor,
                       t: torch.tensor,
                       t_prev: torch.tensor,
                       *,
                       noise_scheduler: GaussianDiffusion,
                       text_embeds: torch.tensor = None,
                       lowres_cond_img: torch.tensor = None,
                       lowres_noise_times: torch.tensor = None,
                       cond_scale: float = 1.,
                       eta: float = 1  # eta=0 as DDIM
                       ) -> torch.tensor:

        b, *_, device = *x.shape, x.device

        # predict eps
        if lowres_cond_img is not None:
            pred = unet.forward(x, t,
                                attr_embeds=text_embeds,
                                cond_drop_prob=cond_scale,
                                lowres_cond_embed=lowres_cond_img,
                                lowres_noise_times=lowres_noise_times)
        else:
            pred = unet.forward(x, t, attr_embeds=text_embeds)

        # ddim
        x_prev = noise_scheduler.ddim_sample(x, t, t_prev, pred, eta)

        return x_prev

    @torch.no_grad()
    def _p_sample_loop(self,
                       unet: Unet,
                       shape: tuple,
                       *,
                       noise_scheduler: GaussianDiffusion,
                       text_embeds: torch.tensor = None,
                       lowres_cond_img: torch.tensor = None,
                       lowres_noise_times: torch.tensor = None,
                       cond_scale: float = 1.,
                       sampling_type: str="ddim",
                       step: int=1
                       ):
        """
        :param unet: The Unet to use for reverse-diffusion.
        """
        device = self.device
        # Get reverse-diffusion timesteps (i.e. (T, T-1, T-2, ..., 2, 1, 0) )
        batch = shape[0]

        if sampling_type == "ddpm":
            timesteps = noise_scheduler._get_sampling_timesteps(batch, device=device)
            # Generate starting "noised samples"
            img = torch.randn(shape, device=device)
            for times in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps)):
                img = self._p_sample(
                    unet,
                    img,
                    times,
                    text_embeds=text_embeds,
                    cond_scale=cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                )
        elif sampling_type == "ddim":

            time_seq = noise_scheduler._get_ddim_sampling_timesteps(batch, device=device, step=step)

            seq_prev = time_seq[1:] + [torch.full((batch,), -1, device=device, dtype=torch.long)]


            img = torch.randn(shape, device=device)
            # denoising the trajectory
            for i in tqdm(range(0, len(time_seq)), desc="DDIM Sampling"):
                t = time_seq[i]
                t_prev = seq_prev[i]
                if t[0] >= noise_scheduler.num_timesteps -1:
                    continue
                img = self._p_sample_ddim(
                    unet,
                    img,
                    t,
                    t_prev,
                    text_embeds=text_embeds,
                    cond_scale=cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                )

        return img



    @torch.no_grad()
    @eval_decorator
    def sample(
            self,
            text_embeds: torch.tensor = None,
            cond_scale: float = 1.,
            lowres_sample_noise_level: float = None,
            device: torch.device = None,
            sampling_type: str = "ddim",
            step: int = 1
    ) -> Union[torch.tensor]:
        device = default(device, self.device)
        self._reset_denoisers_all_one_device(device=device)

        batch_size = text_embeds.shape[0]
        outputs = None
        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)
        # For each unet, sample with the appropriate conditioning
        cascaded_results = []
        for unet_number, unet, channel, sample_size, noise_scheduler in tqdm(
                zip(range(1, len(self.denoisers) + 1), self.denoisers, self.sample_channels, self.sample_sizes,
                    self.noise_schedulers)):

            # If GPU is available, place the Unet currently being sampled from on the GPU
            context = self._one_unet_in_gpu(unet=unet) if is_cuda else null_context()
            with context:
                lowres_cond_img = lowres_noise_times = None
                # If on a super-resolution model, noise the previously generated one
                if unet_number==2:
                    lowres_noise_times = self.lowres_noise_schedule._get_times(batch_size, lowres_sample_noise_level,
                                                                               device=device)
                    lowres_cond_img = img
                    # Here we set the sampling level as zero during sampling,
                    # lowres_cond_img = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img,
                    #                                                       t=lowres_noise_times,
                    #                                                       noise=torch.randn_like(lowres_cond_img))

                if unet_number == 1:
                    shape = (batch_size, sample_size, self.first_stage_dim)

                else:
                    shape = (batch_size, self.channels, sample_size)
                # Generate trajs with the current unet
                img = self._p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    cond_scale=cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    sampling_type = sampling_type,
                    step = step
                )
                # Output the sample trajectory
                cascaded_results.append(img)
                outputs = img if unet_number == len(self.denoisers) else None

        return outputs, cascaded_results

    def forward(self,
                trajs,
                attr_embeds: torch.tensor = None,
                segment_embeds: torch.tensor = None,
                denoiser_number: int = None,
                use_p_loss: bool = True,
                labels: torch.tensor =None):
        """
        give unet number, calculate loss of given unet
        """
        denoising_number = default(denoiser_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == denoising_number-1, \
            f'you can only train on unet #{self.only_train_unet_number}'

        # Get the proper models, objective, etc. for the unet to be trained.
        denoiser_index = denoiser_number - 1
        denoiser = self.denoisers[denoiser_index]

        noise_scheduler = self.noise_schedulers[denoiser_index]

        b, c, h, device, = *trajs.shape, trajs.device

        # Randomly sample a timestep value for each sample in the batch.
        times = noise_scheduler._sample_random_times(b, device=device)
        check_shape(trajs, 'b c h', c=self.channels)

        # obtain low-res conditioning information if a super-res model
        lowres_cond_road_embed = lowres_aug_times = None
        if denoiser_index==1:
            # fine-grained denoising
            x_start = trajs
            lowres_cond_road_embed = segment_embeds
            lowres_aug_time = self.lowres_noise_schedule._sample_random_times(1, device=device)
            lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)
        else:
            # coarse-grained denoising
            x_start = segment_embeds

        noise = torch.randn_like(x_start)
        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # noise the first-level latent conditioning
        lowres_cond_embed_noisy = None
        if exists(lowres_cond_road_embed):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_embed_noisy = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_road_embed,
                                                                        t=lowres_aug_times,
                                                                        noise=torch.randn_like(lowres_cond_road_embed))

        # Predict the noise
        if exists(lowres_cond_road_embed):
            # second_stage: fine-grained denoising, denoiser: unet w/ attention
            pred = denoiser.forward(
                x_noisy,
                times,
                attr_embeds=attr_embeds,
                lowres_noise_times=lowres_aug_times,
                lowres_cond_embed=lowres_cond_embed_noisy,
                cond_drop_prob=self.cond_drop_prob
            )
            return self.loss_fn(pred, noise)
        elif use_p_loss:
            # first_stage with ploss: coarse-grained denoising, transformers-based denoiser
            pred = denoiser.forward(
                x_noisy,
                times,
                attr_embeds=attr_embeds,
                cond_drop_prob=self.cond_drop_prob
            )
            diffusion_loss = self.loss_fn(pred, noise)
            num_steps = noise_scheduler.num_timesteps
            threshold_step = int((1.0 - self.structure_loss_percent) * num_steps)  # e.g., later 10% steps
            structure_mask = (times >= threshold_step)  # shape: (B,)
            if structure_mask.any():
                pred_selected = pred[structure_mask]
                labels_selected = labels[structure_mask]

                with torch.no_grad():
                    re_encoder_outputs = BaseModelOutput(last_hidden_state=self.lm.get_decoder_input(pred_selected.clone()))

                    decoder_out = self.lm(
                        encoder_outputs=re_encoder_outputs,
                        labels=labels_selected,
                        return_dict=True
                    )
                logits = decoder_out.logits  # (B', L, V)

                structure_loss = structure_loss_from_logits(
                    logits,
                    self.adjacency_matrix
                )
            else:
                structure_loss = torch.tensor(0.0, device=pred.device)
            return diffusion_loss + self.structure_loss_weight * structure_loss
        else:
            # first_stage: coarse-grained denoising, transformers-based denoiser
            pred = denoiser.forward(
                x_noisy,
                times,
                attr_embeds=attr_embeds,
                cond_drop_prob=self.cond_drop_prob
            )
            return self.loss_fn(pred, noise)
            # return self.loss_fn(pred, noise)



    @torch.no_grad()
    def sample_second_stage(
        self,
        first_stage_output: torch.Tensor,
        attr_embeds: torch.Tensor = None,
        cond_scale: float = 1.,
        noise_level: float = 0.2,
        sampling_type: str = "ddim",
        step: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        second-stage denoising
        """
        device = default(device, self.device)
        self._reset_denoisers_all_one_device(device=device)

        B, C, L = first_stage_output.shape
        lowres_cond_img = first_stage_output.to(device)
        noise_times = self.lowres_noise_schedule._get_times(B, noise_level, device=device)
        noisy_lowres = self.lowres_noise_schedule.q_sample(
            x_start=lowres_cond_img,
            t=noise_times,
            noise=torch.randn_like(lowres_cond_img)
        )

        unet = self.denoisers[1]
        scheduler = self.noise_schedulers[1]
        shape = (B, self.channels, self.second_stage_len)

        output = self._p_sample_loop(
            unet=unet,
            shape=shape,
            noise_scheduler=scheduler,
            text_embeds=attr_embeds,
            lowres_cond_img=first_stage_output,
            lowres_noise_times=noise_times,
            cond_scale=cond_scale,
            sampling_type=sampling_type,
            step=step
        )

        return output

    @torch.no_grad()
    def sample_first_stage(
        self,
        attr_embeds: torch.Tensor = None,
        cond_scale: float = 1.,
        sampling_type: str = "ddim",
        step: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        first-stage denoising
        """
        device = default(device, self.device)
        self._reset_denoisers_all_one_device(device=device)

        batch_size = attr_embeds.shape[0]
        shape = (batch_size, self.first_stage_len, self.first_stage_dim)

        unet = self.denoisers[0]
        scheduler = self.noise_schedulers[0]

        output = self._p_sample_loop(
            unet=unet,
            shape=shape,
            noise_scheduler=scheduler,
            text_embeds=attr_embeds,
            cond_scale=cond_scale,
            sampling_type=sampling_type,
            step=step
        )

        return output

    @torch.no_grad()
    def sample_first_stage_predict_x0(
            self,
            attr_embeds: torch.Tensor = None,
            cond_scale: float = 1.,
            sampling_type: str = "ddim",
            step: int = 1,
            device: torch.device = None,
    ) -> torch.Tensor:
        """
        Predict-x0 版本：first-stage denoising
        """
        device = default(device, self.device)
        self._reset_denoisers_all_one_device(device=device)

        batch_size = attr_embeds.shape[0]
        shape = (batch_size, self.first_stage_len, self.first_stage_dim)

        unet = self.denoisers[0]
        scheduler = self.noise_schedulers[0]

        img = torch.randn(shape, device=device)

        if sampling_type == "ddpm":
            timesteps = scheduler._get_sampling_timesteps(batch=batch_size, device=device)
            for t in tqdm(timesteps, desc="DDPM Sampling"):
                t_batch = t
                x_start = unet.forward(img, t_batch, attr_embeds=attr_embeds)
                model_mean, _, model_log_variance = scheduler.q_posterior(x_start=x_start, x_t=img, t=t_batch)

                noise = torch.randn_like(img)
                is_last = (t_batch == 0)
                nonzero_mask = (1 - is_last.float()).view(batch_size, *((1,) * (len(img.shape) - 1)))

                img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        elif sampling_type == "ddim":
            time_seq = scheduler._get_ddim_sampling_timesteps(batch=batch_size, device=device, step=step)
            seq_prev = time_seq[1:] + [torch.full((batch_size,), -1, device=device, dtype=torch.long)]

            for i in tqdm(range(len(time_seq)), desc="DDIM Sampling (x0)"):
                t = time_seq[i]
                t_prev = seq_prev[i]
                if t[0] >= scheduler.num_timesteps - 1:
                    continue

                x_start = unet.forward(img, t, attr_embeds=attr_embeds)
                img = scheduler.ddim_sample_from_xstart(img, t, t_prev, x_start,
                                                        eta=0. if sampling_type == "ddim" else 1.)

        return img

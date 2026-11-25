"""Pixel MeanFlow with the iMF boundary-velocity objective.

Based on https://arxiv.org/abs/2601.22158, Algorithm 1 and Table 8.
Like this repository's iMF, time runs from 0=noise to 1=data. The network
predicts pixels; u=(x_pred-z)/(1-t). The denominator floor follows the
official implementation and is also applied to the velocity target.
"""
import torch

from ldm.flow import pad_v_like_x
from ldm.imf import ImprovedMeanFlow


class PixelMeanFlow(ImprovedMeanFlow):
    def __init__(self, net_cfg, fm_ratio=0.5, time_mean=-0.8, time_std=0.8,
                 denominator_floor=0.05, uniform_fraction=0.0, noise_scale=1.0):
        super().__init__(net_cfg, fm_ratio, time_mean, time_std)
        if not 0 < denominator_floor <= 1:
            raise ValueError('denominator_floor must be in (0, 1].')
        if not 0 <= uniform_fraction <= 1 or noise_scale <= 0:
            raise ValueError('Invalid uniform_fraction or noise_scale.')
        self.denominator_floor = denominator_floor
        self.uniform_fraction = uniform_fraction
        self.noise_scale = noise_scale

    def sample_times(self, batch_size, device):
        times = torch.sigmoid(torch.randn(2, batch_size, device=device)
                              * self.time_std + self.time_mean)
        if self.uniform_fraction:
            mask = torch.rand(batch_size, device=device) < self.uniform_fraction
            times = torch.where(mask[None], torch.rand_like(times), times)
        # pMF's reference sets the diagonal BEFORE sorting. Thus its FM
        # samples retain the unsorted logit-normal marginal. iMF sorts first.
        count = int(batch_size * self.fm_ratio)
        times[1, :count] = times[0, :count]
        return times.min(dim=0).values, times.max(dim=0).values

    def denominator(self, t, x):
        return pad_v_like_x((1 - t).clamp_min(self.denominator_floor), x)

    def forward(self, x, t, s=None, **kwargs):
        prediction = super().forward(x, t, s, **kwargs)
        # Conversion arithmetic remains FP32 under BF16 network autocast.
        return (prediction.float() - x.float()) / self.denominator(t, x)

    def compute_ut(self, x0, x1, t):
        z = self.compute_xt(x0, x1, t)
        return (x1 - z) / self.denominator(t, z)

    def training_losses(self, x1, x0=None, **cond_kwargs):
        x1 = x1.float()
        x0 = (torch.randn_like(x1) * self.noise_scale if x0 is None else x0.float())
        t, s = self.sample_times(len(x1), x1.device)
        z = self.compute_xt(x0, x1, t)
        target = (x1 - z) / self.denominator(t, z)
        # Differentiate the complete x->u conversion, holding s/condition
        # fixed. Inherited iMF uses u(z,t,t) as its model-predicted tangent.
        return self.interval_loss(z, target, t, s, **cond_kwargs)

    def validation_losses(self, x1, x0=None, **kwargs):
        x0 = torch.randn_like(x1) * self.noise_scale if x0 is None else x0
        return super().validation_losses(x1, x0=x0, **kwargs)

    def generate(self, x, **kwargs):
        # Input is unit Gaussian noise, as supplied by RepTok/FrozenCodec.
        return super().generate(x * self.noise_scale, **kwargs)

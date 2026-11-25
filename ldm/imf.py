"""Stage-1 iMF, using the boundary-velocity variant of Algorithm 1.

Reference: https://arxiv.org/abs/2512.02012, Eq. 12 / Algorithm 1.
We retain this repository's time direction: 0=noise, 1=data. If t is
the current time and s the destination, t <= s, the identity becomes
v = u - (s-t) * (partial_t u + partial_z u * v).
Conditioning tokens and the destination s are held fixed in the JVP.
"""
import torch

from ldm.flow import FlowModel, pad_v_like_x


class ImprovedMeanFlow(FlowModel):
    def __init__(self, net_cfg, fm_ratio=0.5, time_mean=0.4, time_std=1.0,
                 uniform_fraction=0.0):
        super().__init__(net_cfg)
        if not 0 < fm_ratio < 1:
            raise ValueError('fm_ratio must be between 0 and 1.')
        if time_std <= 0:
            raise ValueError('time_std must be positive.')
        if not 0 <= uniform_fraction <= 1:
            raise ValueError('uniform_fraction must be in [0, 1].')
        self.fm_ratio = fm_ratio
        self.uniform_fraction = uniform_fraction
        # Paper uses -0.4 with 1=noise; reversing time changes this to +0.4.
        self.time_mean, self.time_std = time_mean, time_std

    def forward(self, x, t, s=None, **kwargs):
        if t.numel() == 1:
            t = t.expand(len(x))
        s = t if s is None else s.expand_as(t)
        return self.net(x, t, s, **kwargs)

    def sample_times(self, batch_size, device):
        times = torch.sigmoid(torch.randn(2, batch_size, device=device)
                              * self.time_std + self.time_mean)
        if self.uniform_fraction:
            mask = torch.rand(batch_size, device=device) < self.uniform_fraction
            times = torch.where(mask[None], torch.rand_like(times), times)
        t, s = times.min(dim=0).values, times.max(dim=0).values
        # Random membership also permits both objectives with microbatch size 1.
        fm_mask = torch.rand(batch_size, device=device) < self.fm_ratio
        return t, torch.where(fm_mask, t, s)

    def training_losses(self, x1, x0=None, **cond_kwargs):
        x1 = x1.float()
        x0 = torch.randn_like(x1) if x0 is None else x0.float()
        t, s = self.sample_times(len(x1), x1.device)
        z = self.compute_xt(x0, x1, t)
        return self.interval_loss(z, x1 - x0, t, s, **cond_kwargs)

    def interval_loss(self, z, target_velocity, t, s, **cond_kwargs):
        """Regress velocity with a detached, model-tangent JVP correction.

        Keep JVP in FP32 and outside the backward graph. Recompute u with
        gradients so the existing trainable CLS/projection still receive the
        decoder loss. No differentiation through the tangent or JVP is needed.
        """
        detached_cond = {
            key: value.detach().float() if isinstance(value, torch.Tensor)
            and value.is_floating_point() else value
            for key, value in cond_kwargs.items()
        }
        with torch.no_grad(), torch.autocast(z.device.type, enabled=False):
            z_jvp, t_jvp, s_jvp = z.detach().float(), t.float(), s.float()
            velocity = self(z_jvp, t_jvp, t_jvp, **detached_cond)

            def u_fn(state, current, destination):
                return self(state, current, destination, **detached_cond)

            _, derivative = torch.func.jvp(
                u_fn, (z_jvp, t_jvp, s_jvp),
                (velocity, torch.ones_like(t_jvp), torch.zeros_like(s_jvp)),
            )
        u = self(z, t, s, **cond_kwargs).float()
        prediction = u - pad_v_like_x(s - t, z) * derivative.detach()
        # Plain mean squared error retains the baseline scale relative to cos
        # loss. Adaptive weighting from the full iMF recipe is not enabled.
        return (prediction - target_velocity.detach().float()).square().mean()

    @torch.no_grad()
    def generate(self, x, num_steps=1, reverse=False, return_intermediates=False,
                 progress=False, **kwargs):
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError('num_steps must be a positive integer.')
        if reverse:
            raise ValueError('iMF is trained for noise-to-data intervals only.')
        from tqdm import tqdm

        times = torch.linspace(0, 1, num_steps + 1, device=x.device)
        intermediates = [x] if return_intermediates else None
        for i in tqdm(range(num_steps), disable=not progress):
            t, s = times[i].expand(len(x)), times[i + 1].expand(len(x))
            x = x + pad_v_like_x(s - t, x) * self(x, t, s, **kwargs)
            if return_intermediates:
                intermediates.append(x)
        return torch.stack(intermediates) if return_intermediates else x

    # Inherited validation_losses evaluates u(z,t,t), the instantaneous
    # velocity boundary. This is an FM diagnostic, not the interval objective;
    # it remains valid inside RepTok.validate's torch.inference_mode context.

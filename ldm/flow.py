"""Original RepTok linear flow objective and Euler reconstruction sampler."""
import torch
from torch import nn, Tensor
from tqdm import tqdm
from jutils import instantiate_from_config


def pad_v_like_x(v_, x_):
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


class LinearSchedule:
    def alpha_t(self, t):
        return t


    def alpha_dt_t(self, t):
        return 1


    def sigma_t(self, t):
        return 1 - t


    def sigma_dt_t(self, t):
        return -1



class FlowModel(nn.Module):
    def __init__(self, net_cfg, schedule="linear", sigma_min=0.0):
        super().__init__()
        if schedule != "linear":
            raise ValueError("This branch implements the cls-init-cos linear flow baseline only.")
        self.net = net_cfg if isinstance(net_cfg, nn.Module) else instantiate_from_config(net_cfg)
        self.sigma_min = sigma_min
        self.schedule = LinearSchedule()
        self.t_sampler = torch.rand

    def forward(self, x, t, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        return self.net(x, t, **kwargs)

    def generate(self, x: Tensor, num_steps: int = 50, reverse=False, return_intermediates=False, progress=True, **kwargs):
        """
        Classic Euler sampling from x0 to x1 in num_steps.

        Args:
            x: source minibatch (bs, *dim)
            num_steps: int, number of steps to take
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: bool, if true, return list of samples
            progress: bool, if true, show tqdm progress bar
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        bs, dev = x.shape[0], x.device

        timesteps = torch.linspace(0, 1, num_steps + 1)
        if reverse:
            timesteps = 1 - timesteps

        xt = x
        intermediates = [xt]
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            pred = self.forward(xt, t, **kwargs)

            dt = t_next - t_curr
            xt = xt + dt * pred

            if return_intermediates: intermediates.append(xt)

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return xt


    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt


    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        t = pad_v_like_x(t, x0)
        alpha_dt_t = self.schedule.alpha_dt_t(t)
        sigma_dt_t = self.schedule.sigma_dt_t(t)
        return alpha_dt_t * x1 + sigma_dt_t * x0


    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            t: shape (bs,), represents the time in [0, 1]. If None, we sample
                according to the t_sampler (default: U(0, 1)).
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None: x0 = torch.randn_like(x1)
        t = self.t_sampler(x1.shape[0], device=x1.device, dtype=x1.dtype)

        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.forward(x=xt, t=t, **cond_kwargs)

        return (vt - ut).square().mean()


    def validation_losses(self, x1: Tensor, x0: Tensor = None, num_segments: int = 8, **cond_kwargs):
        """
        SD3 & Meta Movie Gen show that val loss correlates well with human quality. They
        compute the loss in equidistant segments in (0, 1) to reduce variance and average
        them afterwards. Default number of segments: 8 (Esser et al., page 21, ICML 2024).
        """
        if x0 is None: x0 = torch.randn_like(x1)

        assert num_segments > 0, "Number of segments must be greater than 0"
        ts = torch.linspace(0, 1, num_segments+1)[:-1] + 1/(2*num_segments)
        losses_per_segment = []
        for t in ts:
            t = torch.ones(x1.shape[0], device=x1.device) * t
            xt = self.compute_xt(x0=x0, x1=x1, t=t)
            ut = self.compute_ut(x0=x0, x1=x1, t=t)
            vt = self.forward(x=xt, t=t, **cond_kwargs)
            losses_per_segment.append((vt - ut).square().mean())
        
        losses_per_segment = torch.stack(losses_per_segment)
        return losses_per_segment.mean(), losses_per_segment

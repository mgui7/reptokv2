"""Two-sided exact MMD using the official RDM Gaussian-kernel implementation."""
import torch
from torch import nn, distributed as dist
from torch.distributed.nn.functional import all_gather
from ldm.third_party.rdm.kernels import cross_kernel_mean, gamma_from_sigma, self_kernel_mean
from ldm.third_party.rdm.median_heuristic import median_bandwidth


def two_sided_mmd(real, fake, sigma, chunk=1024):
    # Unlike upstream mmd_exact_loss, the reference is trainable: no detach,
    # and its self-kernel term must carry encoder gradients as well.
    gamma = gamma_from_sigma(sigma)
    return (self_kernel_mean(fake, gamma, chunk)
            + self_kernel_mean(real, gamma, chunk)
            - 2 * cross_kernel_mean(fake, real, gamma, chunk))


class BatchMMD(nn.Module):
    def __init__(self, sigma=None, normalization_epsilon=1e-6, chunk=1024):
        super().__init__()
        if (sigma is not None and sigma <= 0) or normalization_epsilon <= 0 or chunk < 1:
            raise ValueError('Invalid MMD settings.')
        self.normalization_epsilon, self.chunk = normalization_epsilon, chunk
        self.register_buffer('sigma', torch.tensor(sigma or 0., dtype=torch.float64))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))

    def forward(self, real, fake):
        # Same autograd gather and FP64 statistic arithmetic as the FD variant.
        # All ranks evaluate the full global-batch loss before DDP averages grads.
        features = torch.stack([real.flatten(1), fake.flatten(1)]).double()
        if dist.is_initialized():
            features = torch.cat(all_gather(features), dim=1)
        with torch.autocast(features.device.type, enabled=False):
            if self.sigma.item() == 0:
                if features.shape[1] < 2:
                    raise ValueError('Bandwidth calibration needs at least two real tokens.')
                # Reuse upstream's single-scale median heuristic, once, on the
                # initial global encoder batch. Do not perturb training RNG.
                devices = [features.device.index] if features.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(2023)
                    self.sigma.fill_(median_bandwidth(features[0].detach()))
            raw = two_sided_mmd(features[0], features[1], self.sigma.item(), self.chunk)
            # V-statistic is nonnegative; guard tiny arithmetic round-off only.
            raw = raw.clamp_min(0)
            normalized = raw / (raw.detach() + self.normalization_epsilon)
            variance = features.detach().var(1, unbiased=False).mean(-1)
            with torch.no_grad():
                self.updates.add_(1)
        return normalized.float(), raw.detach().float(), variance.float()

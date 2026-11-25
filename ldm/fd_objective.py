"""Two-sided population FD with detached EMA history and exact Bures gradients."""
import torch
from torch import nn, distributed as dist
from torch.distributed.nn.functional import all_gather


def symmetric(x):
    return (x + x.mT) * .5


class Bures(torch.autograd.Function):
    """SPD covariance distance; avoid eigenvector derivatives at repeated roots.

    Inputs include fixed diagonal jitter. The transport map T solves T A T = B;
    gradients are I-T and I-T^{-1}. Only first derivatives are needed in training.
    """
    @staticmethod
    def forward(ctx, a, b):
        vals, vecs = torch.linalg.eigh(symmetric(a))
        root = (vecs * vals.sqrt()) @ vecs.mT
        invroot = (vecs * vals.rsqrt()) @ vecs.mT
        vals_mid, vecs_mid = torch.linalg.eigh(symmetric(root @ b @ root))
        # Jitter makes these SPD; reject numerical failure rather than hide it.
        if (vals <= 0).any() or (vals_mid <= 0).any():
            raise FloatingPointError('FD covariance square root is not positive definite.')
        midroot = (vecs_mid * vals_mid.sqrt()) @ vecs_mid.mT
        midinv = (vecs_mid * vals_mid.rsqrt()) @ vecs_mid.mT
        transport = symmetric(invroot @ midroot @ invroot)
        inverse = symmetric(root @ midinv @ root)
        identity = torch.eye(len(a), dtype=a.dtype, device=a.device)
        ctx.save_for_backward(identity - transport, identity - inverse)
        return a.trace() + b.trace() - 2 * vals_mid.sqrt().sum()

    @staticmethod
    def backward(ctx, grad):
        da, db = ctx.saved_tensors
        return grad * da, grad * db


def frechet(mean_a, cov_a, mean_b, cov_b, jitter=1e-5):
    identity = torch.eye(len(mean_a), dtype=mean_a.dtype, device=mean_a.device)
    return (mean_a - mean_b).square().sum() + Bures.apply(
        symmetric(cov_a) + jitter * identity, symmetric(cov_b) + jitter * identity)


class PopulationFD(nn.Module):
    def __init__(self, dim, decay=.99, jitter=1e-5, normalization_epsilon=1e-6):
        super().__init__()
        if not 0 <= decay < 1 or min(jitter, normalization_epsilon) <= 0:
            raise ValueError('Invalid FD statistic settings.')
        self.decay, self.jitter = decay, jitter
        self.normalization_epsilon = normalization_epsilon
        self.register_buffer('means', torch.zeros(2, dim, dtype=torch.float64))
        self.register_buffer('seconds', torch.zeros(2, dim, dim, dtype=torch.float64))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))

    def forward(self, real, fake):
        # All ranks see the same global population; autograd all-gather plus DDP
        # averaging gives the gradient of the global loss (checked in tests).
        features = torch.stack([real.flatten(1), fake.flatten(1)]).double()
        if dist.is_initialized():
            features = torch.cat(all_gather(features), dim=1)
        with torch.autocast(features.device.type, enabled=False):
            mean = features.mean(1)
            second = features.transpose(1, 2) @ features / features.shape[1]
            # Bootstrap from the first actual batch, not fictitious zero samples.
            decay = self.decay if self.updates.item() else 0.
            mean = decay * self.means.detach().clone() + (1-decay) * mean
            second = decay * self.seconds.detach().clone() + (1-decay) * second
            cov = second - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            raw = frechet(mean[0], cov[0], mean[1], cov[1], self.jitter) / features.shape[-1]
            # Tiny round-off can make the theoretical nonnegative FD negative.
            raw = raw.clamp_min(0)
            normalized = raw / (raw.detach() + self.normalization_epsilon)
            with torch.no_grad():
                self.means.copy_(mean)
                self.seconds.copy_(second)
                self.updates.add_(1)
        return normalized.float(), raw.detach().float(), cov.diagonal(dim1=-2, dim2=-1).mean(-1).detach().float()

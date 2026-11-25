"""UL diffusion KL in RepTok's linear-velocity parameterization."""
import torch
from torch.nn import functional as F


def vp(logsnr):
    return torch.sigmoid(logsnr).sqrt(), torch.sigmoid(-logsnr).sqrt()


def expand(value, data):
    return value.reshape(-1, *([1] * (data.ndim - 1)))


def flow_input(clean, noise, logsnr):
    # VP corruption / (alpha+sigma) matches the original flow network input.
    u = torch.sigmoid(logsnr / 2)
    return expand(u, clean) * clean + expand(1 - u, clean) * noise, u


def elbo_velocity_loss(velocity, clean, noise, logsnr, span, denominator):
    # Uniform log-SNR: A_clean=span/2*exp(lambda).
    # z_hat-z=(1-u)*(v_hat-v_target), exp(lambda)*(1-u)^2=u^2.
    u = torch.sigmoid(logsnr / 2)
    error = (velocity.float() - (clean.float() - noise.float())).square().flatten(1).sum(1)
    return (.5 * span * u.square() * error / denominator).mean()


def terminal_kl(tokens, logsnr_min, denominator):
    lam = tokens.new_tensor(logsnr_min, dtype=torch.float32)
    alpha2 = torch.sigmoid(lam)
    log_sigma2 = F.logsigmoid(-lam)
    per_dim = .5 * (alpha2 * tokens.float().square() - alpha2 - log_sigma2)
    return per_dim.flatten(1).sum(1).mean() / denominator


def sample_tokens(noise, predict_velocity, low, high, steps):
    """Euler in original flow coordinate u; return noisy VP z_0, not clean z."""
    endpoints = noise.new_tensor([low, high], dtype=torch.float32)
    alpha, sigma = vp(endpoints)
    u = alpha / (alpha + sigma)
    state = noise.float() / (alpha[0] + sigma[0])
    times = torch.linspace(u[0], u[1], steps + 1, device=noise.device)
    for current, following in zip(times[:-1], times[1:]):
        state = state + (following - current) * predict_velocity(state, current.expand(len(state))).float()
    return state * (alpha[1] + sigma[1])

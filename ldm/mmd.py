"""Joint RepTok with a one-call token generator and two-sided exact MMD."""
import torch
from torch import distributed as dist
from ldm.ul import ULRepTok
from ldm.mmd_objective import BatchMMD
from ldm.trainer import update_ema


class MMDRepTok(ULRepTok):
    extra_log_fields = [field for field in ULRepTok.extra_log_fields if field != 'train/terminal_kl'] + [
        'train/mmd_raw', 'train/encoder_batch_var', 'train/generator_batch_var',
        'train/mmd_sigma', 'train/generator_std', 'train/generator_rms', 'train/generator_pair_cos',
    ]

    def __init__(self, mmd_weight=.1, mmd_sigma=None, mmd_chunk=1024,
                 mmd_normalization_epsilon=1e-6, **kwargs):
        if kwargs.get('token_steps', 1) != 1 or kwargs.get('cfg_scale', 1.) != 1.:
            raise ValueError('MMD trial uses exactly one generator call, CFG=1.')
        kwargs['token_steps'] = 1
        super().__init__(prior_weight=mmd_weight, **kwargs)
        self.mmd = BatchMMD(mmd_sigma, mmd_normalization_epsilon, mmd_chunk)
        self.hyper_parameters.pop('prior_weight')
        self.hyper_parameters.update(mmd_weight=mmd_weight, mmd_sigma=mmd_sigma, mmd_chunk=mmd_chunk,
                                     mmd_normalization_epsilon=mmd_normalization_epsilon)

    def generate_tokens(self, noise, labels):
        # RepTok uses u=0 at noise and velocity z-eps. This residual map is
        # trained directly to the noisy decoder-token distribution, not an ODE.
        time = noise.new_zeros(len(noise))
        return noise.float() + self.prior(noise, time, y=labels, cfg_scale=1.).float()

    def losses(self, batch):
        image = self.image_latents(batch)
        tokens = self.tokenizer(batch['image'])['proj_token'].float()
        if tuple(tokens.shape[1:]) != self.token_shape:
            raise ValueError('Tokenizer and generator shapes differ.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Baseline seeds are equal across ranks. Give MMD independent pure-noise
        # samples per rank, reproducible from checkpointed update count.
        seed = 1000000 + self.hyper_parameters['prior_init_seed'] + int(self.mmd.updates) * world + rank
        rng = torch.Generator(device=tokens.device).manual_seed(seed)
        eta = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        condition = self.noisy_condition(tokens, eta)
        image_loss = self.model.training_losses(image, concat_tokens=condition)
        noise = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        generated = self.generate_tokens(noise, batch['label'].long())
        token_loss, raw, variance = self.mmd(condition, generated)
        return image_loss, token_loss, raw, variance, tokens, generated

    def forward(self, batch):
        image_loss, token_loss, raw, variance, tokens, generated = self.losses(batch)
        loss = image_loss + self.prior_weight * token_loss
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.ema_rate)
        std, rms, cosine = self.token_stats(tokens)
        gstd, grms, gcos = self.token_stats(generated)
        return loss, {'train/loss': loss.detach(), 'train/image_loss': image_loss.detach(),
                      'train/token_loss': token_loss.detach(), 'train/mmd_raw': raw,
                      'train/mmd_sigma': self.mmd.sigma.detach().float(),
                      'train/encoder_batch_var': variance[0], 'train/generator_batch_var': variance[1],
                      'train/token_std': std, 'train/token_rms': rms, 'train/token_pair_cos': cosine,
                      'train/generator_std': gstd, 'train/generator_rms': grms, 'train/generator_pair_cos': gcos}

    @torch.no_grad()
    def sample_tokens(self, labels, seed=0):
        noise = self.noise((len(labels), *self.token_shape), labels.device, seed)
        return self.generate_tokens(noise, labels)

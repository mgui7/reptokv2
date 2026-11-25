"""Joint RepTok with a one-call token generator and two-sided population FD."""
import math
import torch
from torch import distributed as dist
from ldm.ul import ULRepTok
from ldm.fd_objective import PopulationFD
from ldm.trainer import update_ema


class FDRepTok(ULRepTok):
    extra_log_fields = [field for field in ULRepTok.extra_log_fields if field != 'train/terminal_kl'] + [
        'train/fd_raw', 'train/encoder_population_var', 'train/generator_population_var',
        'train/generator_std', 'train/generator_rms', 'train/generator_pair_cos',
    ]

    def __init__(self, fd_weight=.1, fd_decay=.99, fd_jitter=1e-5,
                 fd_normalization_epsilon=1e-6, **kwargs):
        if kwargs.get('token_steps', 1) != 1 or kwargs.get('cfg_scale', 1.) != 1.:
            raise ValueError('FD trial uses exactly one generator call, CFG=1.')
        kwargs['token_steps'] = 1
        super().__init__(prior_weight=fd_weight, **kwargs)
        self.fd = PopulationFD(math.prod(self.token_shape), fd_decay, fd_jitter, fd_normalization_epsilon)
        self.hyper_parameters.pop('prior_weight')
        self.hyper_parameters.update(fd_weight=fd_weight, fd_decay=fd_decay, fd_jitter=fd_jitter,
                                     fd_normalization_epsilon=fd_normalization_epsilon)

    def generate_tokens(self, noise, labels):
        # An x-pred prior directly produces the token. Legacy v checkpoints
        # retain their one-step Euler residual. Both are trained by FD here.
        time = noise.new_zeros(len(noise))
        if getattr(self.prior, 'prediction_type', 'v') == 'x':
            return self.prior.predict_clean(noise, time, y=labels, cfg_scale=1.).float()
        return noise.float() + self.prior(noise, time, y=labels, cfg_scale=1.).float()

    def losses(self, batch):
        image = self.image_latents(batch)
        tokens = self.tokenizer(batch['image'])['proj_token'].float()
        if tuple(tokens.shape[1:]) != self.token_shape:
            raise ValueError('Tokenizer and generator shapes differ.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Baseline seeds are equal across ranks. Give FD independent pure-noise
        # samples per rank, reproducible from checkpointed update count.
        seed = 1000000 + self.hyper_parameters['prior_init_seed'] + int(self.fd.updates) * world + rank
        rng = torch.Generator(device=tokens.device).manual_seed(seed)
        eta = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        condition = self.noisy_condition(tokens, eta)
        image_loss = self.model.training_losses(image, concat_tokens=condition)
        noise = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        generated = self.generate_tokens(noise, batch['label'].long())
        token_loss, raw, variance = self.fd(condition, generated)
        return image_loss, token_loss, raw, variance, tokens, generated

    def forward(self, batch):
        image_loss, token_loss, raw, variance, tokens, generated = self.losses(batch)
        loss = image_loss + self.prior_weight * token_loss
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.ema_rate)
        std, rms, cosine = self.token_stats(tokens)
        gstd, grms, gcos = self.token_stats(generated)
        return loss, {'train/loss': loss.detach(), 'train/image_loss': image_loss.detach(),
                      'train/token_loss': token_loss.detach(), 'train/fd_raw': raw,
                      'train/encoder_population_var': variance[0], 'train/generator_population_var': variance[1],
                      'train/token_std': std, 'train/token_rms': rms, 'train/token_pair_cos': cosine,
                      'train/generator_std': gstd, 'train/generator_rms': grms, 'train/generator_pair_cos': gcos}

    @torch.no_grad()
    def sample_tokens(self, labels, seed=0):
        noise = self.noise((len(labels), *self.token_shape), labels.device, seed)
        return self.generate_tokens(noise, labels)

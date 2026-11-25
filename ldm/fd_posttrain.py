"""Clean-token FD post-training with fixed-scale decoder-only corruption."""
import math
import torch
from torch import distributed as dist
from torch.nn import functional as F
from ldm.fd import FDRepTok
from ldm.fd_objective import PopulationFD, frechet
from ldm.ul_posttrain import ULPosttrainRepTok, reptok_initialization


class FDPosttrainRepTok(FDRepTok, ULPosttrainRepTok):
    # FD supplies the objective; ULPosttrain supplies trained-component optimizer
    # groups, gradient diagnostics, and the optional decoder time-MLP freeze.
    extra_log_fields = FDRepTok.extra_log_fields + [
        'train/weighted_prior_loss', 'train/grad_norm_encoder',
        'train/grad_norm_decoder', 'train/grad_norm_prior',
        'train/decoder_clean_fraction', 'train/decoder_noise_std',
        'val/fd_fresh', 'val/fd_mean_gap', 'val/token_rms_std',
        'val/token_scale_ratio', 'val/corrupted_signal_noise_ratio',
        'val/token_corrupted_mse',
    ]

    def __init__(self, decoder_noise_reference_std, decoder_signal_scale=.8,
                 decoder_noise_scale=.6, decoder_clean_probability=.2, **kwargs):
        values = (decoder_noise_reference_std, decoder_signal_scale, decoder_noise_scale,
                  decoder_clean_probability)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Decoder corruption settings must be finite.')
        if decoder_noise_reference_std <= 0 or decoder_signal_scale <= 0 or decoder_noise_scale < 0:
            raise ValueError('Invalid decoder corruption scale.')
        if not 0 <= decoder_clean_probability <= 1:
            raise ValueError('Clean probability must be in [0,1].')
        super().__init__(**kwargs)
        self.decoder_noise_reference_std = float(decoder_noise_reference_std)
        self.decoder_signal_scale = float(decoder_signal_scale)
        self.decoder_noise_scale = float(decoder_noise_scale)
        self.decoder_clean_probability = float(decoder_clean_probability)
        self.hyper_parameters.update(
            decoder_noise_reference_std=self.decoder_noise_reference_std,
            decoder_signal_scale=self.decoder_signal_scale,
            decoder_noise_scale=self.decoder_noise_scale,
            decoder_clean_probability=self.decoder_clean_probability)

    def corrupt_tokens(self, tokens, noise, clean_mask):
        """No learned normalization or adaptive scale; mask is one choice per image."""
        if noise.shape != tokens.shape or clean_mask.shape != (len(tokens),):
            raise ValueError('Corruption noise/mask shape mismatch.')
        noisy = (self.decoder_signal_scale * tokens.float()
                 + self.decoder_noise_scale * self.decoder_noise_reference_std * noise.float())
        return torch.where(clean_mask.reshape(-1, *([1] * (tokens.ndim - 1))), tokens.float(), noisy)

    def noisy_condition(self, tokens, noise):
        # Compatibility with the parent validation API: evaluation uses clean
        # tokens. Reconstruction corruption is invoked explicitly only in losses.
        return tokens.float()

    def losses(self, batch):
        image = self.image_latents(batch)
        tokens = self.tokenizer(batch['image'])['proj_token'].float()
        if tuple(tokens.shape[1:]) != self.token_shape:
            raise ValueError('Tokenizer and generator shapes differ.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        seed = 1000000 + self.hyper_parameters['prior_init_seed'] + int(self.fd.updates) * world + rank
        rng = torch.Generator(device=tokens.device).manual_seed(seed)
        noise = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        clean = torch.rand(len(tokens), device=tokens.device, generator=rng) < self.decoder_clean_probability
        condition = self.corrupt_tokens(tokens, noise, clean)
        image_loss = self.model.training_losses(image, concat_tokens=condition)
        generator_noise = torch.randn(tokens.shape, device=tokens.device, generator=rng)
        generated = self.generate_tokens(generator_noise, batch['label'].long())
        # Crucial separation: the FD reference NEVER contains decoder noise.
        token_loss, raw, variance = self.fd(tokens, generated)
        self._last_clean_fraction = clean.float().mean().detach()
        return image_loss, token_loss, raw, variance, tokens, generated

    def forward(self, batch):
        loss, metrics = FDRepTok.forward(self, batch)
        metrics['train/weighted_prior_loss'] = self.prior_weight * metrics['train/token_loss']
        metrics['train/decoder_clean_fraction'] = self._last_clean_fraction
        metrics['train/decoder_noise_std'] = loss.new_tensor(
            self.decoder_noise_scale * self.decoder_noise_reference_std)
        return loss, metrics

    @torch.inference_mode()
    def validate(self, loader, device, limit, use_bf16=False):
        from ldm.metrics import ImageMetricTracker, un_normalize_ims
        from ldm.generation import GenerationMetrics
        from ldm.train_utils import move_batch
        self.eval()
        decoder = self.ema_model if self.ema_model is not None else self.model
        reconstruction, generation = ImageMetricTracker().to(device), GenerationMetrics().to(device)
        stats, panels, real_tokens, generated_tokens = [], {}, [], []
        iterator = iter(loader)
        for idx in range(limit):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                image = self.image_latents(batch)
                tokens = self.tokenizer(batch['image'])['proj_token'].float()
                fake_tokens = self.sample_tokens(batch['label'].long(), idx)
                recon = self.sample_images(tokens, idx)
                generated = self.sample_images(fake_tokens, 0)
                reconstruction(batch['image'], recon)
                eps = self.noise(image.shape, device, idx + 30000)
                time = image.new_full((len(image),), .5)
                noisy_image, target = .5 * (image + eps), image - eps
                correct = F.mse_loss(decoder(noisy_image, time, concat_tokens=tokens).float(), target.float())
                shuffled = F.mse_loss(decoder(noisy_image, time, concat_tokens=tokens.roll(1, 0)).float(), target.float())
                corrupted = self.corrupt_tokens(tokens, self.noise(tokens.shape, device, idx + 40000),
                                               torch.zeros(len(tokens), device=device, dtype=torch.bool))
                corruption_mse = F.mse_loss(decoder(noisy_image, time, concat_tokens=corrupted).float(), target.float())
            generation(batch['image'], generated)
            real_tokens.append(tokens.flatten(1))
            generated_tokens.append(fake_tokens.flatten(1))
            stats.append(torch.stack([*self.token_stats(tokens), shuffled, correct, corruption_mse]))
            if idx == 0:
                panels = {name: un_normalize_ims(value[:self.n_images_to_vis]) for name, value in
                          [('real', batch['image']), ('recon', recon), ('gen', generated)]}
        values = reconstruction.aggregate()
        values['gfid'] = generation.aggregate()['gfid']
        values.update(zip(('token_std', 'token_rms', 'token_pair_cos', 'token_shuffle_mse',
                           'token_correct_mse', 'token_corrupted_mse'), torch.stack(stats).mean(0)))
        features = torch.stack([torch.cat(real_tokens), torch.cat(generated_tokens)]).double()
        if dist.is_initialized():
            gathered = [torch.empty_like(features) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, features)
            features = torch.cat(gathered, dim=1)
        with torch.autocast(device.type, enabled=False):
            mean = features.mean(1)
            centered = features - mean[:, None]
            cov = centered.transpose(1, 2) @ centered / features.shape[1]
            values['fd_fresh'] = (frechet(mean[0], cov[0], mean[1], cov[1], self.fd.jitter)
                                  / features.shape[-1]).clamp_min(0).float()
            values['fd_mean_gap'] = (mean[0] - mean[1]).square().mean().float()
            variance = cov[0].diagonal().mean()
            values['token_rms_std'] = variance.sqrt().float()
            values['token_scale_ratio'] = (variance.sqrt() / self.decoder_noise_reference_std).float()
            noise_variance = (self.decoder_noise_scale * self.decoder_noise_reference_std)**2
            values['corrupted_signal_noise_ratio'] = (self.decoder_signal_scale**2 * variance /
                                                       max(noise_variance, 1e-30)).float()
        self.train()
        return values, panels


def clean_fd_initialization(checkpoint, settings):
    """Exact trained E/D/G mapping plus fresh, checkpointed FD moment state."""
    mapped = reptok_initialization(checkpoint, settings)
    params = mapped['hyper_parameters']
    prior = params['prior_cfg']['params']['net_cfg']['params']
    dim = prior['num_channels'] * prior['in_channels']
    objective = PopulationFD(dim, params['fd_decay'], params['fd_jitter'], params['fd_normalization_epsilon'])
    mapped['state_dict'].update({'fd.' + key: value for key, value in objective.state_dict().items()})
    return mapped

"""Baseline RepTok architecture with UL learned-prior regularization (stage 1).

Retains reconstruction FM. Not a full pixel-space UL decoder reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from jutils import instantiate_from_config, load_partial_from_config
from ldm.codec import plain, codec_config, canonical_state
from ldm.shared_unite import portable_config
from ldm.trainer import update_ema
from ldm.ul_objective import vp, flow_input, elbo_velocity_loss, terminal_kl, sample_tokens


class ULRepTok(nn.Module):
    extra_log_fields = [
        'train/image_loss', 'train/token_loss', 'train/terminal_kl',
        'train/token_std', 'train/token_rms', 'train/token_pair_cos',
        'val/token_std', 'val/token_rms', 'val/token_pair_cos',
        'val/token_shuffle_mse', 'val/token_correct_mse',
    ]

    def __init__(self, flow_cfg, tokenizer_cfg, prior_cfg, first_stage_cfg=None,
                 prior_logsnr_min=-10., prior_logsnr_max=5., prior_weight=.1,
                 lr=1e-4, weight_decay=0., ema_rate=.9995, lr_scheduler_cfg=None,
                 token_steps=50, decoder_steps=50, cfg_scale=1., n_images_to_vis=16,
                 prior_init_seed=2023):
        super().__init__()
        if not prior_logsnr_min < prior_logsnr_max or prior_weight <= 0:
            raise ValueError('Invalid prior noise range or loss weight.')
        if min(token_steps, decoder_steps) < 1:
            raise ValueError('Sampling steps must be positive.')
        self.model = instantiate_from_config(flow_cfg)
        self.ema_rate, self.ema_model = ema_rate, None
        if ema_rate > 0:
            # Independent construction avoids compiled-forward deepcopy closures.
            # Preserve RNG consumption for fresh baseline initialization.
            with torch.random.fork_rng(devices=[]):
                self.ema_model = instantiate_from_config(portable_config(flow_cfg))
            self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
            self.ema_model.requires_grad_(False).eval()
        self.first_stage = instantiate_from_config(first_stage_cfg) if first_stage_cfg else None
        if self.first_stage is not None:
            self.first_stage.requires_grad_(False).eval()
        tokenizer_cfg = plain(tokenizer_cfg)
        compile_encoder = tokenizer_cfg['params'].get('compile', False)
        tokenizer_cfg['params']['compile'] = False
        self.tokenizer = instantiate_from_config(tokenizer_cfg)
        if compile_encoder:
            # Compile forwards without changing serialization names.
            self.tokenizer.encoder.forward = torch.compile(self.tokenizer.encoder.forward)
            self.tokenizer.cls_proj.forward = torch.compile(self.tokenizer.cls_proj.forward)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(prior_init_seed)
            self.prior = instantiate_from_config(prior_cfg)
        self.prior_min, self.prior_max, self.prior_weight = prior_logsnr_min, prior_logsnr_max, prior_weight
        self.lr, self.weight_decay, self.lr_scheduler_cfg = lr, weight_decay, lr_scheduler_cfg
        self.token_steps, self.decoder_steps = token_steps, decoder_steps
        self.cfg_scale, self.n_images_to_vis = cfg_scale, n_images_to_vis
        p = flow_cfg['params']['net_cfg']['params']
        self.latent_shape = (p['in_channels'], p['input_size'], p['input_size'])
        p = prior_cfg['params']['net_cfg']['params']
        self.token_shape = (p['num_channels'], p['in_channels'])
        saved_codec = codec_config(dict(flow_cfg=flow_cfg, tokenizer_cfg=tokenizer_cfg,
                                       first_stage_cfg=first_stage_cfg), self.tokenizer)
        saved_codec['flow_cfg'] = portable_config(flow_cfg)
        saved_codec['tokenizer_cfg']['params']['compile'] = compile_encoder
        self.hyper_parameters = dict(**saved_codec, prior_cfg=portable_config(prior_cfg),
            prior_logsnr_min=prior_logsnr_min, prior_logsnr_max=prior_logsnr_max, prior_weight=prior_weight,
            lr=lr, weight_decay=weight_decay, ema_rate=ema_rate, lr_scheduler_cfg=plain(lr_scheduler_cfg),
            token_steps=token_steps, decoder_steps=decoder_steps, cfg_scale=cfg_scale,
            n_images_to_vis=n_images_to_vis, prior_init_seed=prior_init_seed)

    def train(self, mode=True):
        super().train(mode)
        if self.first_stage is not None:
            self.first_stage.eval()
        if self.ema_model is not None:
            self.ema_model.eval()
        return self

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                      lr=self.lr, weight_decay=self.weight_decay)
        scheduler = (load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
                     if self.lr_scheduler_cfg else None)
        return optimizer, scheduler

    @torch.no_grad()
    def image_latents(self, batch):
        if 'latent' in batch:
            return batch['latent']
        return self.first_stage.encode(batch['image']) if self.first_stage else batch['image']

    def decode_image(self, image):
        return self.first_stage.decode(image) if self.first_stage else image

    def noisy_condition(self, tokens, noise):
        alpha, sigma = vp(tokens.new_tensor(self.prior_max, dtype=torch.float32))
        return alpha * tokens.float() + sigma * noise.float()

    def losses(self, batch):
        image = self.image_latents(batch)
        tokens = self.tokenizer(batch['image'])['proj_token'].float()
        if tuple(tokens.shape[1:]) != self.token_shape:
            raise ValueError('Tokenizer and prior shapes differ.')
        condition = self.noisy_condition(tokens, torch.randn_like(tokens))
        image_loss = self.model.training_losses(image, concat_tokens=condition)
        lam = torch.empty(len(image), device=image.device).uniform_(self.prior_min, self.prior_max)
        eps = torch.randn_like(tokens)
        noisy, u = flow_input(tokens, eps, lam)
        velocity = self.prior(noisy, u, y=batch['label'].long())
        # Common dimensional denominator: image FM already means image coordinates.
        denominator = math.prod(self.latent_shape)
        diffusion = elbo_velocity_loss(velocity, tokens, eps, lam,
                                      self.prior_max - self.prior_min, denominator)
        terminal = terminal_kl(tokens, self.prior_min, denominator)
        return image_loss, diffusion + terminal, terminal, tokens

    @staticmethod
    def token_stats(tokens):
        flat = tokens.detach().float().flatten(1)
        return (flat.std(0, unbiased=False).mean(), flat.square().mean().sqrt(),
                F.cosine_similarity(flat, flat.roll(1, 0), dim=-1).mean())

    def forward(self, batch):
        image_loss, token_loss, terminal, tokens = self.losses(batch)
        loss = image_loss + self.prior_weight * token_loss
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.ema_rate)
        std, rms, cosine = self.token_stats(tokens)
        return loss, {'train/loss': loss.detach(), 'train/image_loss': image_loss.detach(),
                      'train/token_loss': token_loss.detach(), 'train/terminal_kl': terminal.detach(),
                      'train/token_std': std, 'train/token_rms': rms, 'train/token_pair_cos': cosine}

    @staticmethod
    def noise(shape, device, seed):
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device)

    @torch.no_grad()
    def sample_tokens(self, labels, seed=0):
        noise = self.noise((len(labels), *self.token_shape), labels.device, seed)
        prediction = lambda value, time: self.prior(value, time, y=labels, cfg_scale=self.cfg_scale)
        return sample_tokens(noise, prediction, self.prior_min, self.prior_max, self.token_steps)

    @torch.no_grad()
    def sample_images(self, condition, seed=0):
        decoder = self.ema_model if self.ema_model is not None else self.model
        noise = self.noise((len(condition), *self.latent_shape), condition.device, seed)
        return self.decode_image(decoder.generate(noise, concat_tokens=condition,
                                                  num_steps=self.decoder_steps, progress=False))

    @torch.inference_mode()
    def validate(self, loader, device, limit, use_bf16=False):
        from ldm.metrics import ImageMetricTracker, un_normalize_ims
        from ldm.generation import GenerationMetrics
        from ldm.train_utils import move_batch
        self.eval()
        decoder = self.ema_model if self.ema_model is not None else self.model
        reconstruction, generation = ImageMetricTracker().to(device), GenerationMetrics().to(device)
        stats, panels = [], {}
        iterator = iter(loader)
        for idx in range(limit):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                image = self.image_latents(batch)
                tokens = self.tokenizer(batch['image'])['proj_token'].float()
                condition = self.noisy_condition(tokens, self.noise(tokens.shape, device, idx + 10000))
                recon = self.sample_images(condition, idx)
                generated = self.sample_images(self.sample_tokens(batch['label'].long(), idx), 0)
                reconstruction(batch['image'], recon)
                # Cheap token-dependence diagnostic; not full shuffled reconstruction.
                eps = self.noise(image.shape, device, idx + 30000)
                time = image.new_full((len(image),), .5)
                noisy, target = .5 * (image + eps), image - eps
                correct = F.mse_loss(decoder(noisy, time, concat_tokens=condition).float(), target.float())
                shuffled = F.mse_loss(decoder(noisy, time, concat_tokens=condition.roll(1, 0)).float(), target.float())
            generation(batch['image'], generated)
            stats.append(torch.stack([*self.token_stats(tokens), shuffled, correct]))
            if idx == 0:
                panels = {name: un_normalize_ims(value[:self.n_images_to_vis]) for name, value in
                          [('real', batch['image']), ('recon', recon), ('gen', generated)]}
        values = reconstruction.aggregate()
        values['gfid'] = generation.aggregate()['gfid']
        values.update(zip(('token_std', 'token_rms', 'token_pair_cos', 'token_shuffle_mse',
                           'token_correct_mse'), torch.stack(stats).mean(0)))
        self.train()
        return values, panels

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        params = plain(state['hyper_parameters'])
        params['flow_cfg']['params']['net_cfg']['params']['compile'] = False
        params['tokenizer_cfg']['params']['compile'] = False
        module = cls(**params)
        module.load_state_dict(canonical_state(state['state_dict']), strict=True)
        return module.to(device).eval()

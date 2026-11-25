"""Experimental single-token UNITE-style training with a shared image decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from jutils import instantiate_from_config, load_partial_from_config

from ldm.codec import plain
from ldm.trainer import update_ema


def portable_config(config):
    config = plain(config)
    def clear(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('load_from_ckpt', 'encoder_path', 'decoder_path'):
                    node[key] = None
                else:
                    clear(value)
        elif isinstance(node, list):
            for value in node:
                clear(value)
    clear(config)
    return config


def latent_corruption(tokens, noise, t, grad_mode='stop'):
    """Control encoder gradients through the noisy input and clean target."""
    if grad_mode not in ('stop', 'input', 'full'):
        raise ValueError(f'Unknown token gradient mode: {grad_mode}')
    source = tokens.detach() if grad_mode == 'stop' else tokens
    target = tokens if grad_mode == 'full' else tokens.detach()
    return t * source + (1 - t) * noise, target


class SharedUNITE(nn.Module):
    # Additional CSV fields are opt-in; the RepTok log schema stays unchanged.
    extra_log_fields = [
        'train/image_loss', 'train/token_loss', 'train/token_std',
        'train/token_rms', 'train/token_pair_cos',
        'val/token_std', 'val/token_pair_cos', 'val/token_noise_mse',
    ]

    def __init__(self, net_cfg, first_stage_cfg=None, lr=1e-4, weight_decay=0.,
                 ema_rate=0.9995, lr_scheduler_cfg=None, token_loss_weight=None,
                 recon_token_noise=0.1, decoder_steps=50, token_steps=50,
                 cfg_scale=1., n_images_to_vis=16, token_grad_mode=None):
        super().__init__()
        token_loss_weight = 1. if token_loss_weight is None else float(token_loss_weight)
        if not math.isfinite(token_loss_weight) or token_loss_weight <= 0 or recon_token_noise < 0:
            raise ValueError('Invalid token loss weight or reconstruction noise.')
        self.model = instantiate_from_config(net_cfg)
        self.first_stage = instantiate_from_config(first_stage_cfg) if first_stage_cfg else None
        if self.first_stage is not None:
            self.first_stage.requires_grad_(False).eval()
        self.ema_rate = ema_rate
        self.ema_model = None
        if ema_rate > 0:
            # Construct independently: deepcopy of compiled bound forwards can
            # retain closures that reference the online model.
            self.ema_model = instantiate_from_config(portable_config(net_cfg))
            self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
            self.ema_model.requires_grad_(False).eval()
        self.lr, self.weight_decay = lr, weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.token_grad_mode = 'stop' if token_grad_mode is None else token_grad_mode
        if self.token_grad_mode not in ('stop', 'input', 'full'):
            raise ValueError(f'Unknown token gradient mode: {self.token_grad_mode}')
        self.token_loss_weight = token_loss_weight
        self.recon_token_noise = recon_token_noise
        self.decoder_steps, self.token_steps = decoder_steps, token_steps
        self.cfg_scale, self.n_images_to_vis = cfg_scale, n_images_to_vis
        p = net_cfg['params']
        self.latent_shape = (p['in_channels'], p['input_size'], p['input_size'])
        # A full checkpoint constructs the model without any initializer files.
        self.hyper_parameters = portable_config(dict(
            net_cfg=net_cfg, first_stage_cfg=first_stage_cfg, lr=lr,
            weight_decay=weight_decay, ema_rate=ema_rate,
            lr_scheduler_cfg=lr_scheduler_cfg, token_loss_weight=token_loss_weight,
            recon_token_noise=recon_token_noise, decoder_steps=decoder_steps,
            token_steps=token_steps, cfg_scale=cfg_scale, n_images_to_vis=n_images_to_vis,
            token_grad_mode=self.token_grad_mode,
        ))

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
                     if self.lr_scheduler_cfg is not None else None)
        return optimizer, scheduler

    @torch.no_grad()
    def image_latents(self, batch):
        if 'latent' in batch:
            return batch['latent']
        return self.first_stage.encode(batch['image']) if self.first_stage is not None else batch['image']

    def decode_image(self, latents):
        return self.first_stage.decode(latents) if self.first_stage is not None else latents

    @staticmethod
    def token_stats(tokens):
        flat = tokens.detach().float().flatten(1)
        cosine = F.cosine_similarity(flat, flat.roll(1, 0), dim=-1).mean()
        return flat.std(dim=0, unbiased=False).mean(), flat.square().mean().sqrt(), cosine

    def forward(self, batch):
        image = self.image_latents(batch)
        tokens = self.model.encode(image)
        batch_size = len(image)
        t_image = torch.rand(batch_size, device=image.device)
        noise_image = torch.randn_like(image)
        t = t_image[:, None, None, None]
        noisy_image = t * image + (1 - t) * noise_image
        conditioning = tokens + self.recon_token_noise * torch.randn_like(tokens)
        velocity = self.model(conditioning, t_image, 'image', image=noisy_image)
        image_loss = F.mse_loss(velocity.float(), (image - noise_image).float())
        t_token = torch.rand(batch_size, device=image.device)
        noisy_token, target = latent_corruption(
            tokens, torch.randn_like(tokens), t_token[:, None, None], self.token_grad_mode)
        prediction = self.model(noisy_token, t_token, 'token', labels=batch['label'].long())
        token_loss = F.mse_loss(prediction.float(), target.float())
        loss = image_loss + self.token_loss_weight * token_loss
        # Keep the existing native loop's pre-update EMA convention.
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.ema_rate)
        std, rms, cosine = self.token_stats(tokens)
        return loss, {'train/loss': loss.detach(), 'train/image_loss': image_loss.detach(),
                      'train/token_loss': token_loss.detach(), 'train/token_std': std,
                      'train/token_rms': rms, 'train/token_pair_cos': cosine}

    @staticmethod
    def noise(shape, device, seed):
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device)

    @torch.no_grad()
    def sample_tokens(self, labels, seed=0, model=None):
        model = model if model is not None else (self.ema_model or self.model)
        device = next(model.parameters()).device
        labels = labels.to(device=device, dtype=torch.long)
        z = self.noise((len(labels), 1, model.hidden_size), device, seed)
        if self.token_steps < 1:
            raise ValueError('token_steps must be positive.')
        # x-start prediction -> velocity: (z_clean - z_t)/(1-t).
        # Evaluate only t < 1, so the endpoint never divides by zero.
        for i in range(self.token_steps):
            t = torch.full((len(z),), i / self.token_steps, device=device)
            prediction = model(z, t, 'token', labels=labels)
            if self.cfg_scale != 1:
                null = torch.full_like(labels, model.y_embedder.num_classes)
                unconditional = model(z, t, 'token', labels=null)
                prediction = unconditional + self.cfg_scale * (prediction - unconditional)
            z = z + (prediction - z) / (self.token_steps - i)
        return z

    @torch.no_grad()
    def sample_images(self, tokens, seed=0, model=None):
        model = model if model is not None else (self.ema_model or self.model)
        image = self.noise((len(tokens), *self.latent_shape), tokens.device, seed)
        if self.decoder_steps < 1:
            raise ValueError('decoder_steps must be positive.')
        for i in range(self.decoder_steps):
            t = torch.full((len(image),), i / self.decoder_steps, device=image.device)
            image = image + model(tokens, t, 'image', image=image) / self.decoder_steps
        return self.decode_image(image)

    @torch.inference_mode()
    def validate(self, loader, device, limit, use_bf16=False):
        from ldm.metrics import ImageMetricTracker, un_normalize_ims
        from ldm.generation import GenerationMetrics
        from ldm.train_utils import move_batch
        self.eval()
        sampler = self.ema_model if self.ema_model is not None else self.model
        reconstruction = ImageMetricTracker().to(device)
        generation = GenerationMetrics().to(device)
        stats, panels = [], {}
        iterator = iter(loader)
        for idx in range(limit):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                image = self.image_latents(batch)
                encode_noise = self.noise((len(image), 1, sampler.hidden_size), device, idx + 10000)
                tokens = sampler.encode(image, encode_noise)
                # Same image, new encoder noise: detect an encoding dominated by
                # random register initialization rather than image information.
                alternate = sampler.encode(image, self.noise(encode_noise.shape, device, idx + 20000))
                recon = self.sample_images(tokens, seed=idx, model=sampler)
                generated = self.sample_images(
                    self.sample_tokens(batch['label'], seed=idx + 30000, model=sampler),
                    seed=idx, model=sampler)
                # Match recon baseline's metric autocast context; generation
                # metrics follow the existing stage-2 FP32 context below.
                reconstruction(batch['image'], recon)
            generation(batch['image'], generated)
            std, _, cosine = self.token_stats(tokens)
            noise_mse = F.mse_loss(tokens.float(), alternate.float())
            stats.append(torch.stack([std, cosine, noise_mse]))
            if idx == 0:
                n = self.n_images_to_vis
                panels = {'real': un_normalize_ims(batch['image'][:n]),
                          'recon': un_normalize_ims(recon[:n]),
                          'gen': un_normalize_ims(generated[:n])}
        values = reconstruction.aggregate()
        values['gfid'] = generation.aggregate()['gfid']
        values.update(zip(('token_std', 'token_pair_cos', 'token_noise_mse'), torch.stack(stats).mean(0)))
        self.train()
        return values, panels

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        params = portable_config(checkpoint['hyper_parameters'])
        params['net_cfg']['params']['compile'] = False
        module = cls(**params)
        module.load_state_dict(checkpoint['state_dict'], strict=True)
        return module.to(device).eval()

"""The original cls-init-cos reconstruction objective, without a training framework."""
from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn
from jutils import freeze, instantiate_from_config, load_partial_from_config

from ldm.metrics import ImageMetricTracker, un_normalize_ims


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_params[name.removeprefix('module.')].mul_(decay).add_(param.data, alpha=1 - decay)


class RepTok(nn.Module):
    def __init__(self, flow_cfg, tokenizer_cfg, first_stage_cfg=None,
                 lr=1e-4, weight_decay=0.0, ema_rate=0.9995,
                 lr_scheduler_cfg=None, dino_loss_weight=0.1,
                 n_images_to_vis=16, sample_kwargs=None):
        super().__init__()
        # Save before constructing the teacher: its frozen config must not mutate
        # the trainable tokenizer configuration stored in the checkpoint.
        self.hyper_parameters = deepcopy(dict(
            flow_cfg=flow_cfg, tokenizer_cfg=tokenizer_cfg,
            first_stage_cfg=first_stage_cfg, lr=lr, weight_decay=weight_decay,
            ema_rate=ema_rate, lr_scheduler_cfg=lr_scheduler_cfg,
            dino_loss_weight=dino_loss_weight, n_images_to_vis=n_images_to_vis,
            sample_kwargs=sample_kwargs,
        ))
        self.model = instantiate_from_config(flow_cfg)
        self.ema_rate = ema_rate
        self.ema_model = None
        if ema_rate > 0:
            self.ema_model = deepcopy(self.model)
            freeze(self.ema_model)
            self.ema_model.eval()
            update_ema(self.ema_model, self.model, decay=0)
        self.first_stage = None
        if first_stage_cfg is not None:
            self.first_stage = instantiate_from_config(first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval()
        self.tokenizer = instantiate_from_config(tokenizer_cfg)
        self.lr, self.weight_decay = lr, weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.sample_kwargs = sample_kwargs or {}
        self.n_images_to_vis = n_images_to_vis
        self.generator = torch.Generator()
        self.latent_shape = None
        self.metric_tracker = ImageMetricTracker()
        self.dino_loss_weight = dino_loss_weight
        teacher_cfg = deepcopy(tokenizer_cfg['params']['encoder'])
        teacher_cfg['params']['trainable_layers'] = []
        self.frozen_dino = instantiate_from_config(teacher_cfg)
        freeze(self.frozen_dino)

    def export_codec_config(self):
        from ldm.codec import codec_config
        return codec_config(self.hyper_parameters, self.tokenizer)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = None
        if self.lr_scheduler_cfg is not None:
            scheduler = load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
        return optimizer, scheduler

    @torch.no_grad()
    def encode(self, images):
        return self.first_stage.encode(images) if self.first_stage is not None else images

    @torch.no_grad()
    def decode(self, latents):
        return self.first_stage.decode(latents) if self.first_stage is not None else latents

    def forward(self, batch):
        images = batch['image']
        # Keep the eager encode call: the legacy dict.get evaluates it even when
        # cached latents exist. Removing it can change random-number consumption.
        latents = batch.get('latent', self.encode(images))
        tokens = self.tokenizer(images)
        with torch.no_grad():
            teacher = self.frozen_dino(images)['cls_token']
            teacher = torch.cat([teacher], dim=1)
        similarity = nn.functional.cosine_similarity(tokens['token'], teacher, dim=-1)
        cosine_loss = self.dino_loss_weight * (1 - similarity).mean()
        flow_loss = self.model.training_losses(latents, concat_tokens=tokens['proj_token'])
        if self.ema_model is not None:
            # Deliberately before backward/optimizer.step, as in training_step.
            update_ema(self.ema_model, self.model, self.ema_rate)
        return flow_loss + cosine_loss, {
            'train/loss': flow_loss.detach(),
            'train/cos_sim': similarity.mean().detach(),
            'train/cos_sim_loss': cosine_loss.detach(),
        }

    def reconstruction_batch(self, batch, batch_idx):
        images = batch['image']
        latents = batch.get('latent')
        tokens = self.tokenizer(images)['proj_token']
        encoded = self.encode(images)  # legacy default(latents, encode(images))
        latents = encoded if latents is None else latents
        _, losses = self.model.validation_losses(latents, concat_tokens=tokens)
        if self.latent_shape is None:
            self.latent_shape = self.encode(images).shape[1:]
        noise = torch.randn((len(images), *self.latent_shape),
                            generator=self.generator.manual_seed(batch_idx)).to(images.device)
        sampler = self.ema_model if self.ema_model is not None else self.model
        samples = self.decode(sampler.generate(noise, concat_tokens=tokens, **self.sample_kwargs))
        self.metric_tracker(images, samples)
        return samples, losses

    @torch.inference_mode()
    def validate(self, loader, device, limit, use_bf16=False):
        from ldm.train_utils import move_batch
        self.eval()
        losses, panels = [], {}
        iterator = iter(loader)
        for batch_idx in range(limit):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                samples, segments = self.reconstruction_batch(batch, batch_idx)
            losses.append(segments)
            if batch_idx == 0:
                panels = {
                    'real': un_normalize_ims(batch['image'][:self.n_images_to_vis]),
                    'fake': un_normalize_ims(samples[:self.n_images_to_vis]),
                }
        metrics = self.metric_tracker.aggregate()
        metrics['loss'] = torch.stack(losses).mean(0).mean()
        self.metric_tracker.reset()
        # Lightning recursively puts the entire module back into train mode,
        # including frozen children. Preserve that behavior.
        self.train()
        return metrics, panels

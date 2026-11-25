"""Stage-2 class-conditioned token flow with a self-contained frozen recon codec."""
from copy import deepcopy

import torch
from torch import nn
from jutils import instantiate_from_config, load_partial_from_config

from ldm.codec import FrozenCodec, plain
from ldm.flow import FlowModel
from ldm.metrics import un_normalize_ims
from ldm.trainer import update_ema


class TokenFlow(FlowModel):
    def forward(self, x, t, cfg_scale=1.0, **kwargs):
        if cfg_scale == 1.0:
            return super().forward(x, t, **kwargs)
        y = kwargs.pop('y')
        if self.net.y_embedder.dropout_prob <= 0:
            raise ValueError('CFG requires a trained unconditional class embedding.')
        if t.numel() == 1:
            t = t.expand(len(x))
        labels = torch.cat([y, torch.full_like(y, self.net.y_embedder.num_classes)])
        conditional, unconditional = self.net(
            torch.cat([x, x]), torch.cat([t, t]), y=labels, **kwargs,
        ).chunk(2)
        return unconditional + cfg_scale * (conditional - unconditional)


class GenerationMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        from torchmetrics import SumMetric
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False)
        self.count = SumMetric()

    def forward(self, real, fake):
        self.fid.update(un_normalize_ims(real), real=True)
        self.fid.update(un_normalize_ims(fake), real=False)
        self.count.update(len(real))

    def aggregate(self):
        return {'gfid': self.fid.compute(), 'n_metric_samples': self.count.compute()}


class TokenGenerator(nn.Module):
    validation_metric = 'gfid'

    def __init__(self, flow_cfg, recon_checkpoint=None, codec_config=None,
                 lr=1e-4, weight_decay=0.0, ema_rate=0.0, lr_scheduler_cfg=None,
                 n_images_to_vis=16, sample_kwargs=None, decoder_steps=50):
        super().__init__()
        self.model = instantiate_from_config(flow_cfg)
        self.ema_rate = ema_rate
        self.ema_model = deepcopy(self.model).requires_grad_(False) if ema_rate > 0 else None
        if codec_config is not None:
            self.codec = FrozenCodec(codec_config)
        elif recon_checkpoint:
            self.codec = FrozenCodec.from_recon_checkpoint(recon_checkpoint)
        else:
            raise ValueError('Fresh gen training needs recon_checkpoint; a full gen checkpoint embeds the codec.')
        self.lr, self.weight_decay = lr, weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.sample_kwargs = plain(sample_kwargs) if sample_kwargs else {'num_steps': 50, 'progress': False}
        self.decoder_steps = decoder_steps
        self.n_images_to_vis = n_images_to_vis
        net = self.model.net
        self.latent_shape = (net.num_channels, net.in_channels)
        # Architecture only: no recon path or initialization file dependency.
        self.hyper_parameters = plain(dict(
            flow_cfg=flow_cfg, codec_config=self.codec.config, lr=lr,
            weight_decay=weight_decay, ema_rate=ema_rate,
            lr_scheduler_cfg=lr_scheduler_cfg, n_images_to_vis=n_images_to_vis,
            sample_kwargs=self.sample_kwargs, decoder_steps=decoder_steps,
        ))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                      lr=self.lr, weight_decay=self.weight_decay)
        scheduler = (load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
                     if self.lr_scheduler_cfg is not None else None)
        return optimizer, scheduler

    def forward(self, batch):
        tokens = self.codec.encode(batch['image'])
        if tuple(tokens.shape[1:]) != self.latent_shape:
            raise ValueError(f'Generator expects {self.latent_shape} tokens, codec returned {tuple(tokens.shape[1:])}.')
        # Image-space latent.npy caches are not stage-2 token targets.
        loss = self.model.training_losses(tokens, y=batch['label'].long())
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.ema_rate)
        return loss, {'train/loss': loss.detach()}

    @torch.no_grad()
    def sample(self, labels, seed=0):
        self.eval()
        device = next(self.parameters()).device
        labels = labels.to(device=device, dtype=torch.long)
        noise = torch.randn((len(labels), *self.latent_shape),
                            generator=torch.Generator().manual_seed(seed)).to(device)
        sampler = self.ema_model if self.ema_model is not None else self.model
        tokens = sampler.generate(noise, y=labels, **self.sample_kwargs)
        return self.codec.decode(tokens, self.decoder_steps)

    @torch.inference_mode()
    def validate(self, loader, device, limit, use_bf16=False):
        from ldm.train_utils import move_batch
        self.eval()
        # Evaluation-only Inception is not part of the saved generative model.
        metrics = GenerationMetrics().to(device)
        losses, panels = [], {}
        iterator = iter(loader)
        for batch_idx in range(limit):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                tokens = self.codec.encode(batch['image'])
                _, segments = self.model.validation_losses(tokens, y=batch['label'].long())
                samples = self.sample(batch['label'], seed=batch_idx)
            metrics(batch['image'], samples)
            losses.append(segments)
            if batch_idx == 0:
                panels = {'real': un_normalize_ims(batch['image'][:self.n_images_to_vis]),
                          'fake': un_normalize_ims(samples[:self.n_images_to_vis])}
        values = metrics.aggregate()
        values['loss'] = torch.stack(losses).mean(0).mean()
        self.train()
        return values, panels

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        module = cls(**checkpoint['hyper_parameters'])
        module.load_state_dict(checkpoint['state_dict'], strict=True)
        return module.to(device).eval()

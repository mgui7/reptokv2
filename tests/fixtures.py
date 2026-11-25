"""Small offline components for loop parity; these are not benchmark models."""
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from omegaconf import OmegaConf


class TinyEncoder(nn.Module):
    hidden_dim = 8
    num_register_tokens = 0

    def __init__(self, trainable_layers=(), compile=False):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.proj = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.1)
        for name, p in self.named_parameters():
            p.requires_grad_(any(layer in name for layer in trainable_layers))

    def forward(self, x):
        z = self.proj(x.mean((2, 3))).unsqueeze(1) + self.embeddings.cls_token
        return {'cls_token': self.dropout(z)}


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 3, 1)

    def encode(self, x):
        return self.encoder(x) + torch.randn_like(self.encoder(x)) * 0.01

    def decode(self, z):
        return self.decoder(z).tanh()


class TinyMetricTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('total', torch.tensor(0.0), persistent=False)
        self.register_buffer('count', torch.tensor(0.0), persistent=False)

    def forward(self, target, prediction):
        self.total += (target - prediction).square().mean() * len(target)
        self.count += len(target)

    def aggregate(self):
        # Deliberately a proxy; never report these as actual FID measurements.
        total, count = self.total.clone(), self.count.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total)
            torch.distributed.all_reduce(count)
        return {'fid': total / count, 'mse': total / count, 'n_metric_samples': count}

    def reset(self):
        self.total.zero_()
        self.count.zero_()


class Batches(IterableDataset):
    def __iter__(self):
        g = torch.Generator().manual_seed(77)
        while True:
            yield {'image': torch.randn(2, 3, 4, 4, generator=g),
                   'latent': torch.randn(2, 4, 4, 4, generator=g)}


class TinyData:
    def __init__(self, **kwargs):
        pass

    def train_dataloader(self):
        return DataLoader(Batches(), batch_size=None)

    def val_dataloader(self):
        return DataLoader(Batches(), batch_size=None)


def model_params():
    return OmegaConf.create({
        'flow_cfg': {'target': 'ldm.flow.FlowModel', 'params': {
            'schedule': 'linear', 'sigma_min': 0.0,
            'net_cfg': {'target': 'ldm.models.transformer.mmdit.MMDiT', 'params': {
                'input_size': 4, 'in_channels': 4, 'depth': 2, 'hidden_size': 32,
                'patch_size': 2, 'num_heads': 4, 'num_classes': 1000,
                'learn_sigma': True, 'null_class_cond': True, 'compile': False,
            }}}},
        'tokenizer_cfg': {'target': 'ldm.tokenizer.ClsTokenizer', 'params': {
            'mlp_out_dim': 32, 'compile': False,
            'encoder': {'target': 'fixtures.TinyEncoder', 'params': {
                'trainable_layers': ['embeddings.cls_token']}}}},
        'first_stage_cfg': {'target': 'fixtures.TinyAutoencoder', 'params': {}},
        'lr': 0.001, 'weight_decay': 0.0, 'ema_rate': 0.9,
        'lr_scheduler_cfg': {'target': 'jutils.nn.lr_schedulers.get_constant_schedule_with_warmup',
                            'params': {'num_warmup_steps': 8}},
        'dino_loss_weight': 0.1, 'n_images_to_vis': 2,
        'sample_kwargs': {'num_steps': 3, 'progress': False},
    })


def build_reptok(**params):
    from ldm import trainer
    original = trainer.ImageMetricTracker
    trainer.ImageMetricTracker = TinyMetricTracker
    try:
        return trainer.RepTok(**params)
    finally:
        trainer.ImageMetricTracker = original


class GenBatches(Batches):
    def __iter__(self):
        for batch in super().__iter__():
            yield dict(batch, label=torch.tensor([0, 1]))


class TinyGenData(TinyData):
    def train_dataloader(self):
        return DataLoader(GenBatches(), batch_size=None)

    def val_dataloader(self):
        return DataLoader(GenBatches(), batch_size=None)

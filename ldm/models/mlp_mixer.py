"""RepTok's attention-free token MLP-Mixer for the native generator.

Based on image-ldm MLPMixer1D (33b4277) and
https://github.com/CompVis/RepTok/blob/main/reptok/models/mlp-mixer.py.
Mixer construction follows lucidrains/mlp-mixer-pytorch, as in that source.
Historical expansion_factor mixes tokens; expansion_factor_token mixes features.
"""
from functools import partial

import torch
from torch import nn

from ldm.models.transformer.dit import LabelEmbedder, TimestepEmbedder


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def feed_forward(dim, expansion, dropout, dense=nn.Linear):
    inner = int(dim * expansion)
    if inner < 1:
        raise ValueError('Mixer expansion must produce at least one hidden unit.')
    return nn.Sequential(
        dense(dim, inner), nn.GELU(), nn.Dropout(dropout),
        dense(inner, dim), nn.Dropout(dropout),
    )


class MLPMixer1D(nn.Module):
    def __init__(self, n_channels=1, channels=1152, dim=1280, depth=28,
                 expansion_factor=4, expansion_factor_token=2, dropout=0.,
                 num_classes=1000, class_dropout_prob=0.1, n_extra_tokens=0,
                 zero_init_last=False, compile=False):
        super().__init__()
        if min(n_channels, channels, dim, depth) < 1 or n_extra_tokens < 0 or num_classes < 0:
            raise ValueError('Invalid Mixer dimensions.')
        if not 0 <= class_dropout_prob <= 1:
            raise ValueError('Class dropout must be between zero and one.')
        self.n_channels, self.channels = n_channels, channels
        # Native TokenGenerator uses the same shape interface as token_dit.
        self.num_channels, self.in_channels = n_channels, channels
        self.n_extra_tokens = n_extra_tokens
        self.num_classes, self.class_dropout_prob = num_classes, class_dropout_prob
        count = n_channels + 1 + int(num_classes > 0) + n_extra_tokens
        token_linear = partial(nn.Conv1d, kernel_size=1)
        self.mixer = nn.Sequential(
            nn.Linear(channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, feed_forward(count, expansion_factor, dropout, token_linear)),
                PreNormResidual(dim, feed_forward(dim, expansion_factor_token, dropout)),
            ) for _ in range(depth)],
            nn.LayerNorm(dim), nn.Linear(dim, channels),
        )
        if zero_init_last:
            nn.init.zeros_(self.mixer[-1].weight)
            nn.init.zeros_(self.mixer[-1].bias)
        if compile:
            self.mixer = torch.compile(self.mixer)
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, channels))
        self.t_embedder = TimestepEmbedder(channels)
        self.label_embedder = (LabelEmbedder(num_classes, channels, class_dropout_prob)
                               if num_classes > 0 else None)

    @property
    def y_embedder(self):
        # A property avoids registering duplicate checkpoint state keys.
        return self.label_embedder

    def forward(self, x, t, concat_tokens=None, y=None):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_channels, self.channels):
            raise ValueError(f'Expected B x {self.n_channels} x {self.channels} tokens.')
        if t.numel() == 1:
            t = t.reshape(1).expand(len(x))
        if t.shape != (len(x),):
            raise ValueError('Expected one timestep per sample.')
        parts = [x + self.pos_embed]
        if self.label_embedder is not None:
            if y is None or y.shape != (len(x),):
                raise ValueError('Expected one class label per sample.')
            parts.append(self.label_embedder(y, self.training).unsqueeze(1))
        parts.append(self.t_embedder(t).unsqueeze(1))
        xs = torch.cat(parts, dim=1)
        if self.n_extra_tokens:
            if concat_tokens is None or concat_tokens.shape != (len(x), self.n_extra_tokens, self.channels):
                raise ValueError('Context token shape does not match n_extra_tokens.')
            xs = torch.cat([concat_tokens, xs], dim=1)
        elif concat_tokens is not None:
            raise ValueError('This Mixer was configured without context tokens.')
        output = self.mixer(xs)
        return output[:, self.n_extra_tokens:self.n_extra_tokens + self.n_channels]

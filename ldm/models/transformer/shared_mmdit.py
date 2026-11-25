"""One SiT backbone for image encoding, latent denoising and image flow."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .dit import DiT


class SharedMMDiT(DiT):
    def __init__(self, n_aux_registers=4, **kwargs):
        # Load the original SiT state before adding the shared-task parameters.
        super().__init__(**kwargs)
        if self.y_embedder is None or self.y_embedder.dropout_prob <= 0:
            raise ValueError('Shared training requires a null class embedding.')
        if n_aux_registers < 0:
            raise ValueError('n_aux_registers must be nonnegative.')
        self.aux_registers = nn.Parameter(torch.randn(1, n_aux_registers, self.hidden_size) * 0.02)
        self.task_embed = nn.Embedding(3, self.hidden_size)
        nn.init.normal_(self.task_embed.weight, std=0.02)
        # No token projection: the bottleneck width is the backbone width.
        self.token_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def forward(self, tokens, t, mode, image=None, labels=None):
        modes = {'encode': 0, 'token': 1, 'image': 2}
        if mode not in modes:
            raise ValueError(f'Unknown shared mode: {mode}')
        if tokens.ndim != 3 or tokens.shape[1:] != (1, self.hidden_size):
            raise ValueError('Exactly one latent token at backbone width is required.')
        if (image is None) != (mode == 'token'):
            raise ValueError('Image patches must be absent only in token denoising.')
        batch = len(tokens)
        if t.numel() == 1:
            t = t.expand(batch)
        if mode != 'token':
            # Never allow a label side channel into encoding or reconstruction.
            if labels is not None:
                raise ValueError('Only token denoising accepts image labels.')
            labels = torch.full((batch,), self.y_embedder.num_classes,
                                device=tokens.device, dtype=torch.long)
        elif labels is None:
            raise ValueError('Token denoising needs labels (or explicit null labels).')
        condition = self.t_embedder(t) + self.y_embedder(labels, self.training and mode == 'token')
        condition = condition + self.task_embed.weight[modes[mode]]
        # Reset to the learned initial embeddings on EVERY forward. Auxiliary
        # hidden states are never returned, cached or transferred between tasks.
        parts = [tokens, self.aux_registers.expand(batch, -1, -1)]
        if image is not None:
            parts.append(self.x_embedder(image) + self.pos_embed)
        hidden = torch.cat(parts, dim=1)
        for block in self.blocks:
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, condition, use_reentrant=False)
            else:
                hidden = block(hidden, condition)
        if mode != 'image':
            return self.token_norm(hidden[:, :1])
        patches = hidden[:, 1 + self.aux_registers.shape[1]:]
        velocity = self.unpatchify(self.final_layer(patches, condition))
        if self.learn_sigma and not self.return_sigma:
            velocity = velocity.chunk(2, dim=1)[0]
        return velocity

    def encode(self, image, noise=None):
        if noise is None:
            noise = torch.randn((len(image), 1, self.hidden_size), device=image.device, dtype=image.dtype)
        # Encoder token starts at pure noise (t=0); clean image availability is
        # indicated by the mode, not by the latent's timestep.
        return self(noise, image.new_zeros(len(image)), 'encode', image=image)

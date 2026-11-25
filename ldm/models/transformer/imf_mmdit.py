"""SiT-compatible MMDiT with an additional interval-length embedding."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ldm.models.transformer.mmdit import MMDiT
# MMDiT imports the legacy module by this name; share its compile setting.
from dit import TimestepEmbedder


class IMFMMDiT(MMDiT):
    def __init__(self, *args, compile=False, **kwargs):
        if compile:
            raise ValueError('IMFMMDiT requires compile=false for the FP32 JVP path.')
        # Load the existing SiT weights before adding new parameters.
        super().__init__(*args, compile=False, **kwargs)
        if self.y_embedder is not None and not self.null_class_cond:
            raise ValueError('IMFMMDiT supports stage-1 null-class conditioning only.')
        if self.return_sigma:
            raise ValueError('IMFMMDiT must predict velocity only (return_sigma=false).')
        # Do not shift initialization of the subsequently constructed tokenizer
        # projection relative to the same-seed FM baseline.
        with torch.random.fork_rng(devices=[]):
            self.interval_embedder = TimestepEmbedder(self.hidden_size)
            nn.init.normal_(self.interval_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.interval_embedder.mlp[0].bias)
            nn.init.zeros_(self.interval_embedder.mlp[2].weight)
            nn.init.zeros_(self.interval_embedder.mlp[2].bias)
        # Flash/efficient SDPA kernels may not implement forward-mode AD.
        # Explicit timm attention supports JVP on CPU and CUDA.
        for block in self.blocks:
            block.attn.fused_attn = False

    def forward(self, x, t, s, concat_tokens):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.interval_embedder(s - t)
        if self.y_embedder is not None:
            labels = torch.full_like(t, self.y_embedder.num_classes, dtype=torch.long)
            # Null labels need no random class dropout; all JVP passes must
            # evaluate the same deterministic conditional field.
            c = c + self.y_embedder(labels, train=False)
        n_extra_tokens = concat_tokens.shape[1]
        x = torch.cat([concat_tokens, x], dim=1)
        for block in self.blocks:
            if self.use_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        x = self.final_layer(x, c)[:, n_extra_tokens:]
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

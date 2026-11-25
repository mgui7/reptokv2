"""Original RepTok stage-2 token Transformer; shared DiT building blocks."""
import torch
from torch import nn
from einops import rearrange
from .dit import TimestepEmbedder, LabelEmbedder, DiTBlock, modulate


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class OneDimensionalTransformer(nn.Module):
    """
    A 1D transformer model for non-spatial data.
    """
    def __init__(
        self,
        num_channels=5,
        in_channels=1152,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=0,
        n_registers=0,
        out_channels=None,
        learn_sigma=False,          # LEGACY (True for DiT and SiT)
        return_sigma=False,         # LEGACY (True for DiT, False for SiT, but not used at all)
        use_checkpointing=False,
        load_from_ckpt=None,
        split_channels=1,           # number of channels for one channel to split into
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_channels = num_channels
        self.in_channels = in_channels
        if learn_sigma:
            self.out_channels = in_channels * 2
        else:
            self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.use_checkpointing = use_checkpointing
        self.hidden_size = hidden_size
        self.n_registers = n_registers
        self.split_channels = split_channels       # default to no channel splitting (*1)
        if n_registers > 0:
            print(f"[1D MMDiT] Using {n_registers} register tokens")
            self.registers = nn.Parameter(torch.randn(1, n_registers, hidden_size))

        self.return_sigma = return_sigma

        self.x_embedder = nn.Sequential(
            nn.Linear(in_channels, hidden_size * split_channels),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            uc = "+ unconditional" if class_dropout_prob > 0 else ""
            print(f"[DiT] Class-conditional ({num_classes} classes {uc})")
        else:
            self.y_embedder = None
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_channels, hidden_size * split_channels))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels // split_channels)
        self.initialize_weights()

        if load_from_ckpt is not None:
            dev = next(self.parameters()).device
            print(f"[DiT] Loading weights from {load_from_ckpt}")
            self.load_state_dict(torch.load(load_from_ckpt, map_location=dev))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize the positional embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder[0].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder[0].bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, concat_tokens=None, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        Split -> Registers -> context_selfattn
        """
        assert x.shape[1] == self.pos_embed.shape[1], f"Input shape {x.shape} does not match position embedding shape {self.pos_embed.shape}"
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        if self.split_channels > 1:
            x = rearrange(x, 'n t (d1 d2) -> n (t d1) d2', d1=self.split_channels)

        # define conditioning vector ----------- #
        t = self.t_embedder(t)                   # (N, D)
        if self.y_embedder is not None:
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                        # (N, D)
        else:
            c = t
        # ------------------------------------- #

        # prepend register tokens ------------- #
        if self.n_registers > 0:
            registers = self.registers.repeat(x.shape[0], 1, 1)
            x = torch.cat([registers, x], dim=1)
        # ------------------------------------- #

        # prepend context tokens -------------- #
        if concat_tokens is None:
            n_extra_tokens = 0
        else:
            n_extra_tokens = concat_tokens.shape[1]
            x = torch.cat([concat_tokens, x], dim=1)
        # ------------------------------------- #

        # iterate through blocks
        for block in self.blocks:
            if self.use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
            else:
                x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                   # (N, T, D)

        # remove context tokens --------------- #
        x = x[:, n_extra_tokens:]                  # (N, T, D)
        # ------------------------------------- #

        # remove register tokens -------------- #
        if self.n_registers > 0:
            x = x[:, self.n_registers:]
        # ------------------------------------- #
        if self.split_channels > 1:
            x = rearrange(x, 'n (t d1) d2 -> n t (d1 d2)', d1=self.split_channels)
        if self.learn_sigma and not self.return_sigma:        # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, cfg_scale, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, **kwargs)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

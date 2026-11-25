"""Frozen Self-Flow ImageNet features, with a deterministic SD-VAE input path."""
import torch
from torch import nn
from diffusers import AutoencoderKL
from einops import rearrange

from ldm.models.self_flow.model import SelfFlowPerTokenDiT


IMAGENET_CONFIG = dict(
    input_size=32, patch_size=2, in_channels=4, hidden_size=1152,
    depth=28, num_heads=16, mlp_ratio=4.0, num_classes=1001,
    learn_sigma=True, compatibility_mode=True,
)


class FrozenSelfFlowEncoder(nn.Module):
    """Pool raw block output, before the upstream self-distillation projector.

    Images are RGB in [-1, 1]. Encode the VAE posterior mode, without noise,
    at public forward timestep 0 (internal SiT timestep 1). Class conditioning
    is always null; no ground-truth label enters the representation.
    """

    def __init__(self, load_from_ckpt=None, vae_pretrained=None, vae_config=None,
                 model_config=None, layer=16, encode_batch_size=16):
        super().__init__()
        config = dict(IMAGENET_CONFIG if model_config is None else model_config)
        if not 1 <= layer <= config['depth']:
            raise ValueError('layer is one-based and must be within model depth')
        if encode_batch_size < 1:
            raise ValueError('encode_batch_size must be positive')
        self.layer = layer
        self.encode_batch_size = encode_batch_size
        self.hidden_dim = config['hidden_size']
        self.model = SelfFlowPerTokenDiT(**config)
        if load_from_ckpt is not None:
            state = torch.load(load_from_ckpt, map_location='cpu', weights_only=True)
            state = state.get('model', state.get('state_dict', state))
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
        if vae_pretrained is not None:
            self.vae = AutoencoderKL.from_pretrained(vae_pretrained, local_files_only=True)
        elif vae_config is not None:
            self.vae = AutoencoderKL.from_config(dict(vae_config))
        else:
            raise ValueError('Provide vae_pretrained or an embedded vae_config')
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode_latents(self, images):
        # Keep the VAE posterior numerically stable and deterministic. It is the
        # Self-Flow input encoder, independent of RepTok's reconstruction AE.
        with torch.autocast(images.device.type, enabled=False):
            return self.vae.encode(images.float()).latent_dist.mode() * 0.18215

    @torch.no_grad()
    def patch_features(self, latents):
        model = self.model
        p = model.patch_size
        expected = model.x_embedder.grid_size[0] * p
        if tuple(latents.shape[1:]) != (model.in_channels, expected, expected):
            raise ValueError(f'Unexpected Self-Flow latent shape: {tuple(latents.shape)}')
        x = rearrange(latents, 'b c (h p) (w q) -> b (h w) (c p q)', p=p, q=p)
        x = model.x_embedder(x) + model.pos_embed
        # Upstream forward applies 1 - timesteps. Clean public timestep 0
        # therefore means internal timestep 1, not 0.
        t = torch.ones(len(x), device=x.device)
        labels = torch.full((len(x),), model.y_embedder.num_classes - 1,
                            device=x.device, dtype=torch.long)
        c = model.t_embedder(t).unsqueeze(1) + model.y_embedder(labels, False).unsqueeze(1)
        for block in model.blocks[:self.layer]:
            x = block(x, c)
        return x

    @torch.no_grad()
    def forward(self, images):
        pooled = []
        for batch in images.split(self.encode_batch_size):
            features = self.patch_features(self.encode_latents(batch))
            pooled.append(features.mean(dim=1, keepdim=True))
        return {'cls_token': torch.cat(pooled)}

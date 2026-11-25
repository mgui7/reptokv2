"""Frozen RepTok codec reconstructed from architecture + weights, never old paths."""
from copy import deepcopy

import torch
from torch import nn
from omegaconf import OmegaConf
from jutils import instantiate_from_config


def plain(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return {key: plain(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [plain(value) for value in config]
    return deepcopy(config)


def uncompiled(module):
    while hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module


def codec_config(params, tokenizer=None):
    params = plain(params)
    config = {key: params[key] for key in ('flow_cfg', 'tokenizer_cfg', 'first_stage_cfg')}

    def clear_initializers(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'compile':
                    node[key] = False
                elif key in ('load_from_ckpt', 'encoder_path', 'decoder_path', 'vae_pretrained'):
                    node[key] = None
                else:
                    clear_initializers(value)
        elif isinstance(node, list):
            for value in node:
                clear_initializers(value)

    clear_initializers(config)
    encoder = config['tokenizer_cfg']['params']['encoder']
    if encoder['target'] == 'ldm.encoders.DINOImageEncoder':
        if tokenizer is not None:
            model = uncompiled(uncompiled(tokenizer.encoder).model)
            encoder['params']['model_config'] = model.config.to_dict()
        elif 'model_config' not in encoder['params']:
            # One-time conversion of older recon files lacking embedded HF config.
            from transformers import AutoConfig
            name = encoder['params'].get('model_name', 'facebook/dinov2-with-registers-base')
            encoder['params']['model_config'] = AutoConfig.from_pretrained(name).to_dict()
    if encoder['target'] == 'ldm.self_flow_encoder.FrozenSelfFlowEncoder':
        if tokenizer is not None:
            encoder['params']['vae_config'] = dict(uncompiled(tokenizer.encoder).vae.config)
        elif 'vae_config' not in encoder['params']:
            raise ValueError('Self-Flow codec requires embedded VAE architecture')
    return config


def recon_resume_params(checkpoint):
    """Drop external initializers while retaining training compile wrappers."""
    params = plain(checkpoint['hyper_parameters'])
    architecture = plain(checkpoint['codec_config'])

    def restore_compile(original, target):
        if isinstance(original, dict) and isinstance(target, dict):
            for key, value in original.items():
                if key == 'compile':
                    target[key] = value
                elif key in target:
                    restore_compile(value, target[key])

    restore_compile(params, architecture)
    params.update(architecture)
    return params


def canonical_state(state):
    # torch.compile changes serialization prefixes, not the underlying model.
    result = {}
    for name, value in state.items():
        key = '.'.join(part for part in name.split('.') if part != '_orig_mod')
        if key in result:
            raise ValueError(f'Duplicate state key after compile normalization: {key}')
        result[key] = value
    return result


class FrozenCodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = plain(config)
        self.flow = instantiate_from_config(self.config['flow_cfg'])
        self.tokenizer = instantiate_from_config(self.config['tokenizer_cfg'])
        ae = self.config['first_stage_cfg']
        self.first_stage = instantiate_from_config(ae) if ae is not None else nn.Identity()
        net = self.config['flow_cfg']['params']['net_cfg']['params']
        self.latent_shape = (net['in_channels'], net['input_size'], net['input_size'])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen stage-1 dropout/normalization must not change during gen training.
        return super().train(False)

    @classmethod
    def from_recon_checkpoint(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint.get('codec_config')
        if config is None:
            config = codec_config(checkpoint['hyper_parameters'])
        codec = cls(config)
        state = canonical_state(checkpoint['state_dict'])
        decoder = 'ema_model.' if any(k.startswith('ema_model.') for k in state) else 'model.'
        selected = {}
        for source, target in ((decoder, 'flow.'), ('tokenizer.', 'tokenizer.'),
                               ('first_stage.', 'first_stage.')):
            selected.update({target + key[len(source):]: value for key, value in state.items()
                             if key.startswith(source)})
        codec.load_state_dict(selected, strict=True)
        return codec

    @torch.no_grad()
    def encode(self, images):
        return self.tokenizer(images)['proj_token']

    @torch.no_grad()
    def decode(self, tokens, num_steps=50):
        noise = torch.randn((len(tokens), *self.latent_shape),
                            generator=torch.Generator().manual_seed(0)).to(tokens.device)
        latent = self.flow.generate(noise, concat_tokens=tokens, num_steps=num_steps, progress=False)
        if self.config['first_stage_cfg'] is None:
            return latent
        return self.first_stage.decode(latent)

import torch
from torch import nn
from typing import Union
from omegaconf import ListConfig, OmegaConf
from transformers import AutoConfig, AutoModel

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class DINOImageEncoder(nn.Module):
    def __init__(self, model_name='facebook/dinov2-with-registers-base', trainable_layers: Union[list, str] = [], compile=False, pretrained=True, model_config=None):
        """
        Potential trainable layers: ['embeddings.cls_token', 'embeddings.register_tokens']
        """
        super().__init__()
        print(f"[DINOImageEncoder] {model_name}")
        assert model_name in ['facebook/dinov2-with-registers-base', 'facebook/dinov2-base'], (
            "Only 'facebook/dinov2-with-registers-base' and 'facebook/dinov2-base' are supported"
        )
        print(f"[DINOImageEncoder] Trainable layers: {trainable_layers}")
        if model_config is not None:
            # A full checkpoint supplies architecture and weights; no Hub lookup.
            config = (OmegaConf.to_container(model_config, resolve=True)
                      if OmegaConf.is_config(model_config) else dict(model_config))
            model_type = config.pop('model_type')
            self.model = AutoModel.from_config(AutoConfig.for_model(model_type, **config)).eval()
        elif pretrained:
            self.model = AutoModel.from_pretrained(model_name).eval()
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
            trainable_layers = "all"  # if pretrained is False, all layers are trainable

        if compile:
            self.model = torch.compile(self.model)

        self.hidden_dim = self.model.config.hidden_size
        if "registers" in model_name:
            self.num_register_tokens = self.model.config.num_register_tokens
        else:
            self.num_register_tokens = 0

        # Freeze layers
        if isinstance(trainable_layers, str):
            if trainable_layers == "all":
                # avoid not used parameters warning
                self.model.embeddings.mask_token.requires_grad_(False)
            else:
                raise ValueError(f"Unknown string for trainable_layers: {trainable_layers}")
        elif isinstance(trainable_layers, ListConfig) or isinstance(trainable_layers, list):
            for name, param in self.model.named_parameters():
                if any([layer in name for layer in trainable_layers]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        print(f"[DINOImageEncoder] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self

    def forward(self, imgs: torch.Tensor):
        assert len(imgs.shape) == 4         # [B, C, H, W]
        imgs = (imgs + 1.0) / 2.0           # Normalize to [0, 1]
        imgs = torch.nn.functional.interpolate(imgs, [224, 224], mode="bilinear")

        # Normalize the images using the standard ImageNet mean and std
        # Sourced from https://github.com/facebookresearch/dinov2/issues/86
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(imgs.device)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(imgs.device)
        imgs = (imgs - mean) / std

        vision_outputs = self.model(imgs)

        cls_token = vision_outputs.last_hidden_state[:, :1]
        if self.num_register_tokens > 0:
            register_tokens = vision_outputs.last_hidden_state[:, 1:1 + self.num_register_tokens]
        else:
            register_tokens = None
        image_tokens = vision_outputs.last_hidden_state[:, 1 + self.num_register_tokens:]

        return {
            "cls_token": cls_token,                 # (1, 1, hidden_dim)
            "register_tokens": register_tokens,     # (1, num_register_tokens, hidden_dim)
            "image_tokens": image_tokens            # (1, 256, hidden_dim)
        }
    
    def encode(self, imgs: torch.Tensor):
        return self.forward(imgs)

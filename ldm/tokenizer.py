import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig
from jutils import instantiate_from_config, load_partial_from_config

class Projection(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.ln(self.fc(x))



class ClsTokenizer(nn.Module):
    def __init__(
        self,
        encoder: Union[dict, nn.Module],        # must have a `self.hidden_dim` attribute
        mlp_out_dim: int = 768,
        compile: bool = False,
        proj_layer: Union[dict, nn.Module] = Projection,
    ):
        super().__init__()
        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            self.encoder = instantiate_from_config(encoder)

        self.enc_hidden_dim = self.encoder.hidden_dim

        # projection layer
        h_dim = self.enc_hidden_dim
        if isinstance(proj_layer, (dict, DictConfig)):
            proj_layer = load_partial_from_config(proj_layer)
        self.cls_proj = proj_layer(in_features=h_dim, hidden_features=h_dim * 4, out_features=mlp_out_dim)

        if compile:
            self.encoder = torch.compile(self.encoder)
            self.cls_proj = torch.compile(self.cls_proj)
        
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ClsTokenizer] Trainable parameters: {train_params:,}")

    def forward(self, x, require_loss: bool = False):
        """
        Args:
            x: Images in [-1, 1] range, shape (b, c, h, w)
        Returns:
            dict:
            - token: shape (b, num_tokens, hidden_dim)
            - proj_token: shape (b, num_tokens, mlp_out_dim)
        """
        if not require_loss:
            enc_out = self.encoder(x)
        else:
            enc_out = self.encoder.forward_with_loss(x)
        cls_token = enc_out['cls_token']
        proj_cls_token = self.cls_proj(cls_token)
        return dict(token=cls_token, proj_token=proj_cls_token, verbose_info=enc_out)

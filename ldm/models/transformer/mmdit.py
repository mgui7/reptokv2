import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dit import DiT


class MMDiT(DiT):
    """
    Diffusion Transformer with concatenation of tokens.
    """
    def __init__(self, *args, null_class_cond: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # If true, the model will always use the null class token
        # from the pretrained class-conditional ImageNet model.
        self.null_class_cond = null_class_cond

    def forward(self, x, t, concat_tokens, y=None):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        if self.y_embedder is not None:
            # use null token    ------------------- #
            if self.null_class_cond:
                y = torch.ones(x.shape[0], dtype=torch.long).to(x.device) * 1000
            # ------------------------------------- #
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t

        # prepend context tokens -------------- #
        n_extra_tokens = concat_tokens.shape[1]
        x = torch.cat([concat_tokens, x], dim=1)
        # ------------------------------------- #
        
        for block in self.blocks:
            if self.use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
            else:
                x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        # remove context tokens --------------- #
        x = x[:, n_extra_tokens:]
        # ------------------------------------- #

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma and not self.return_sigma:        # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        raise NotImplementedError("Not implemented yet.")


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == "__main__":
    ipt = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 1000, (2,))
    
    """ Conditional """
    net = DiT_models['DiT-B/8']()
    out = net(ipt, t, y)

    print("Conditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")
    
    """ Unconditional """
    net = DiT_models['DiT-B/8'](num_classes=-1)
    out = net(ipt, t)

    print("Unconditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")

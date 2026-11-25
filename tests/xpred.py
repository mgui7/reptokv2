"""JiT readout/loss, legacy parity, direct FD and sequential checkpoint tests."""
import copy
import sys
import tempfile
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params, build_reptok
from ldm.codec import plain
from ldm.flow import FlowModel
from ldm.generation import TokenFlow, TokenGenerator
from ldm.fd_posttrain import FDPosttrainRepTok, clean_fd_initialization


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
    def forward(self, x, t, **kwargs):
        return torch.ones_like(x) * self.value


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2023)
    clean = torch.full((4, 1, 3), 2.)
    noise = torch.full_like(clean, -1.)
    t = torch.tensor([0., .5, .99, 1.])
    model = TokenFlow(Constant(2), prediction_type='x')
    # Perfect clean predictions have zero loss even inside the clipped region.
    assert model._jit_loss(clean, noise, t) == 0
    model.net.value.data.fill_(1.)
    loss = model._jit_loss(clean, noise, t)
    torch.testing.assert_close(loss, torch.tensor((1 + 4 + 400 + 400) / 4))
    loss.backward()
    torch.testing.assert_close(model.net.value.grad, -2 * loss)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mixed = model._jit_loss(clean.bfloat16(), noise.bfloat16(), t)
    assert mixed.dtype == torch.float32 and torch.isfinite(mixed)
    # Endpoint readout must reconstruct a perfect constant denoiser at any NFE;
    # 20-step grid reaches the 0.05 floor exactly, and one step is direct x.
    model.net.value.data.fill_(2.)
    for steps in [1, 5, 20]:
        torch.testing.assert_close(model.generate(noise, steps, progress=False), clean)
    values, segments = model.validation_losses(clean, noise, num_segments=100)
    assert values == 0 and segments.shape == (100,)
    # Existing default v training, validation and sampling are bitwise unchanged.
    old, new = FlowModel(Constant(.3)), TokenFlow(Constant(.3))
    for obj in [old, new]:
        torch.manual_seed(7)
        value = obj.training_losses(clean, noise)
        if obj is old: reference = value
        else: torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(old.generate(noise, 7, progress=False), new.generate(noise, 7, progress=False), rtol=0, atol=0)
    from ldm.models.transformer.token_dit import OneDimensionalTransformer
    net_args = dict(num_channels=1, in_channels=32, hidden_size=32, depth=2,
                    num_heads=4, num_classes=3, class_dropout_prob=.1, n_registers=1)
    flow = dict(target='ldm.generation.TokenFlow', params=dict(prediction_type='x', t_eps=.05,
                net_cfg=dict(target='ldm.models.transformer.token_dit.OneDimensionalTransformer', params=net_args)))
    model = TokenFlow(OneDimensionalTransformer(**net_args), prediction_type='x').eval()
    with torch.no_grad(): model.net.final_layer.linear.weight.normal_(std=.02)
    x, t, y = torch.randn(2, 1, 32), torch.tensor([.3, .99]), torch.tensor([0, 1])
    conditional = model.net(x, t, y=y)
    unconditional = model.net(x, t, y=torch.full_like(y, 3))
    expected = unconditional + 1.5 * (conditional - unconditional)
    torch.testing.assert_close(model.predict_clean(x, t, y=y, cfg_scale=1.5), expected)
    torch.testing.assert_close(model(x, t, y=y, cfg_scale=1.5),
                               (expected-x) / torch.tensor([.7, .05]).view(2,1,1))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        recon = build_reptok(**model_params()).eval()
        torch.save(dict(state_dict=recon.state_dict(), hyper_parameters=recon.hyper_parameters,
                        codec_config=recon.export_codec_config()), path/'recon.ckpt')
        gen = TokenGenerator(flow_cfg=flow, recon_checkpoint=path/'recon.ckpt',
                             sample_kwargs=dict(num_steps=3, progress=False), decoder_steps=3)
        opt, _ = gen.configure_optimizers()
        batch = dict(image=torch.randn(4,3,4,4), latent=torch.randn(4,4,4,4), label=torch.tensor([0,1,2,0]))
        frozen = {k:v.clone() for k,v in gen.codec.state_dict().items()}
        for _ in range(3):
            opt.zero_grad(); loss, _ = gen(batch); loss.backward(); opt.step()
            assert torch.isfinite(loss)
        assert all(p.grad is None for p in gen.codec.parameters())
        for key, value in frozen.items(): torch.testing.assert_close(value, gen.codec.state_dict()[key], rtol=0, atol=0)
        checkpoint = dict(state_dict=gen.state_dict(), hyper_parameters=gen.hyper_parameters, global_step=3)
        torch.save(checkpoint, path/'gen.ckpt')
        (path/'recon.ckpt').unlink()
        loaded = TokenGenerator.from_checkpoint(path/'gen.ckpt')
        assert loaded.model.prediction_type == 'x'
        gen.eval()
        torch.testing.assert_close(gen.sample(y), loaded.sample(y), rtol=0, atol=0)
        base = plain(model_params())
        settings = {k:base[k] for k in ['flow_cfg','tokenizer_cfg','first_stage_cfg','lr_scheduler_cfg']}
        settings.update(prior_cfg=flow, lr=1e-5, encoder_lr=1e-6, ema_rate=.9,
                        token_steps=1, decoder_steps=3, n_images_to_vis=2,
                        freeze_decoder_time_mlp=True, decoder_noise_reference_std=1.3,
                        fd_weight=.1, fd_decay=.99, fd_jitter=1e-5, fd_normalization_epsilon=1e-6)
        mapped = clean_fd_initialization(checkpoint, settings)
        fd = FDPosttrainRepTok(**mapped['hyper_parameters'])
        fd.load_state_dict(mapped['state_dict'], strict=True)
        assert fd.prior.prediction_type == 'x' and int(fd.fd.updates) == 0
        fd.eval()
        noise = torch.randn(2,1,32)
        direct = loaded.model.predict_clean(noise, torch.zeros(2), y=y)
        torch.testing.assert_close(fd.generate_tokens(noise,y), direct, rtol=0, atol=0)
        torch.testing.assert_close(fd.generate_tokens(noise,y), loaded.model.generate(noise,1,y=y,progress=False))
        assert not torch.allclose(fd.generate_tokens(noise,y), noise+direct)
        fd.train()
        for _ in range(2):
            fd.zero_grad()
            image_loss, token_loss, _, _, tokens, generated = fd.losses(batch)
            loss = image_loss + .1 * token_loss
            loss.backward()
            assert torch.isfinite(loss)
            for net in (fd.tokenizer, fd.prior, fd.model):
                grads=[p.grad for p in net.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(v).all() for v in grads)
            assert all(p.grad is None for p in fd.model.net.t_embedder.mlp.parameters())
        print('PASS: clipped JiT target/prediction; FP32 loss under autocast; CFG; Euler; v parity; frozen codec; trained x checkpoint reload; exact FD warm-start/direct output; joint gradients.')

if __name__ == '__main__': main()

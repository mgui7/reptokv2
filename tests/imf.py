"""Offline iMF/pMF math, gradient, warm-start and NFE checks; no quality claims."""
from pathlib import Path
import argparse
import sys
import tempfile

import torch
from torch import nn
from hydra import compose, initialize_config_dir

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from fixtures import imf_params, pmf_params, build_reptok
from ldm.imf import ImprovedMeanFlow
from ldm.pmf import PixelMeanFlow
from ldm.models.transformer.mmdit import MMDiT
from ldm.models.transformer.imf_mmdit import IMFMMDiT, PixelMMDiT


class AffineField(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3))
        self.calls = 0

    def forward(self, x, t, s, concat_tokens):
        self.calls += 1
        return self.a * x + 0.7 * t[:, None] + 1.1 * s[:, None] + concat_tokens


def check_identity():
    net = AffineField()
    model = ImprovedMeanFlow(net)
    z = torch.tensor([[0.2, 0.5], [0.1, -0.6]])
    t, s = torch.tensor([0.1, 0.4]), torch.tensor([0.8, 0.9])
    cond = torch.tensor([[0.2, 0.3], [0.4, -0.2]], requires_grad=True)
    target = torch.ones_like(z) * 2
    # Independent analytic directional derivative: a*v(z,t,t) + 0.7.
    velocity = 0.3 * z + 1.8 * t[:, None] + cond.detach()
    correction = (s-t)[:, None] * (0.3 * velocity + 0.7)
    prediction = 0.3 * z + 0.7*t[:, None] + 1.1*s[:, None] + cond.detach() - correction
    error = prediction - target
    loss = model.interval_loss(z, target, t, s, concat_tokens=cond)
    torch.testing.assert_close(loss, error.square().mean())
    loss.backward()
    torch.testing.assert_close(net.a.grad, (2 * error * z).mean())
    torch.testing.assert_close(cond.grad, 2 * error / error.numel())
    # At s=t the correction vanishes and this is ordinary FM.
    boundary = model.interval_loss(z, target, t, t, concat_tokens=cond)
    torch.testing.assert_close(boundary, (velocity-target).square().mean())
    for steps in (1, 2, 4):
        net.calls = 0
        result = model.generate(z, num_steps=steps, concat_tokens=cond)
        assert net.calls == steps and torch.isfinite(result).all()
        if steps == 1:
            torch.testing.assert_close(result, z + 0.3*z + 1.1 + cond.detach())
    for invalid in (0, -1, 1.5, True):
        try:
            model.generate(z, num_steps=invalid, concat_tokens=cond)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid step count accepted')
    t, s = model.sample_times(10000, torch.device('cpu'))
    assert (s >= t).all() and (t > 0).all() and (s < 1).all()
    assert 0.45 < (s == t).float().mean() < 0.55
    model.uniform_fraction = 1.0
    t, s = model.sample_times(100000, torch.device('cpu'))
    diagonal = t == s
    assert 0.49 < diagonal.float().mean() < 0.51
    # iMF retains sort-before-diagonal: the lower endpoint is min(U1,U2).
    assert abs(t.mean().item() - 1/3) < 0.01
    assert abs(s[~diagonal].mean().item() - 2/3) < 0.01
    assert (t >= 0).all() and (s < 1).all() and (t <= s).all()


def check_backbone(device):
    kwargs = dict(input_size=4, in_channels=4, depth=2, hidden_size=32,
                  patch_size=2, num_heads=4, num_classes=1000, learn_sigma=True,
                  null_class_cond=True, compile=False)
    initial_rng = torch.get_rng_state()
    baseline = MMDiT(**kwargs).eval()
    after_baseline_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    initialized_imf = IMFMMDiT(**kwargs)
    assert torch.equal(torch.get_rng_state(), after_baseline_rng)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(initialized_imf.state_dict()[key], value, rtol=0, atol=0)
    # Nonzero readout ensures the comparison and conditioning gradients matter.
    nn.init.normal_(baseline.final_layer.linear.weight, std=0.05)
    for block in baseline.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'sit.pt'
        torch.save(baseline.state_dict(), path)
        model = IMFMMDiT(**kwargs, load_from_ckpt=path).eval()
    baseline, model = baseline.to(device), model.to(device)
    x, c, t = (torch.randn(2, 4, 4, 4, device=device),
               torch.randn(2, 1, 32, device=device), torch.rand(2, device=device))
    torch.testing.assert_close(model(x, t, torch.ones_like(t), c), baseline(x, t, c))
    # Exercise the actual backbone JVP with nonzero time/interval dependence.
    nn.init.normal_(model.interval_embedder.mlp[2].weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
    for checkpointing in (False, True):
        model.use_checkpointing = checkpointing
        model.train()
        model.zero_grad(set_to_none=True)
        cond = c.clone().requires_grad_()
        loss = ImprovedMeanFlow(model).interval_loss(
            x, torch.randn_like(x), t, torch.ones_like(t), concat_tokens=cond)
        loss.backward()
        assert torch.isfinite(loss) and cond.grad.abs().sum() > 0
        assert model.interval_embedder.mlp[2].weight.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)


def check_reptok(device, pixel=False):
    params = pmf_params() if pixel else imf_params()
    # This unit check drives AdamW directly; entry-point smoke covers warmup.
    params.lr_scheduler_cfg = None
    model = build_reptok(**params).to(device)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer, _ = model.configure_optimizers()
    for step in range(3):
        batch = {'image': torch.randn(2, 3, 4, 4, device=device),
                 'latent': torch.randn(2, 4, 4, 4, device=device)}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, metrics = model(batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert 'train/cos_sim_loss' in metrics
        # A randomly initialized DiT has a zero readout and zero attention
        # gates; conditioning begins to affect its output after updates.
        if step == 2:
            assert model.tokenizer.cls_proj.fc.weight.grad.abs().sum() > 0
        optimizer.step()
    for name, value in frozen.items():
        if not name.startswith('ema_model.'):
            torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        samples, losses = model.reconstruction_batch(batch, 0)
    assert samples.shape == batch['image'].shape and torch.isfinite(losses).all()
    restored = build_reptok(**params).to(device)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        channels = 3 if pixel else 4
        noise, cond = torch.randn(2, channels, 4, 4, device=device), torch.randn(2, 1, 32, device=device)
        torch.testing.assert_close(model.model.generate(noise, concat_tokens=cond),
                                   restored.model.generate(noise, concat_tokens=cond), rtol=0, atol=0)


def check_pixel_math():
    net = AffineField()
    model = PixelMeanFlow(net)
    z = torch.tensor([[0.2, 0.5], [0.1, -0.6]])
    t, s = torch.tensor([0.2, 0.98]), torch.tensor([0.8, 0.99])
    cond = torch.tensor([[0.2, 0.3], [0.4, -0.2]], requires_grad=True)
    d = torch.tensor([[0.8], [0.05]])
    f = 0.3*z + 0.7*t[:, None] + 1.1*s[:, None] + cond.detach()
    u = (f-z)/d
    v = (0.3*z + 1.8*t[:, None] + cond.detach() - z)/d
    # Includes the derivative of the division and its clipped boundary.
    du = (-0.7*v + 0.7)/d + u/d * torch.tensor([[1.], [0.]])
    error = u - (s-t)[:, None]*du - 2
    loss = model.interval_loss(z, torch.full_like(z, 2), t, s, concat_tokens=cond)
    torch.testing.assert_close(loss, error.square().mean())
    loss.backward()
    torch.testing.assert_close(net.a.grad, (2*error*z/d).mean())
    torch.testing.assert_close(cond.grad, 2*error/d/error.numel())
    for steps in (1, 2, 4):
        net.calls = 0
        result = model.generate(z, num_steps=steps, concat_tokens=cond)
        assert net.calls == steps and torch.isfinite(result).all()
        if steps == 1:
            torch.testing.assert_close(result, 0.3*z + 1.1 + cond.detach())
    x0, x1 = torch.zeros_like(z), torch.ones_like(z)
    torch.testing.assert_close(model.compute_ut(x0, x1, t),
                               torch.tensor([[1., 1.], [0.4, 0.4]]))
    t, s = model.sample_times(100000, torch.device('cpu'))
    assert (t <= s).all() and (t[:50000] == s[:50000]).all()
    assert (s[50000:] > t[50000:]).float().mean() > 0.999
    logits = torch.logit(t[:50000])
    assert abs(logits.mean().item() + 0.8) < 0.02
    assert abs(logits.std().item() - 0.8) < 0.02
    model.uniform_fraction = 1.0
    t, s = model.sample_times(100000, torch.device('cpu'))
    assert abs(t[:50000].mean().item() - 0.5) < 0.01
    assert abs(t[:50000].std().item() - 12**-0.5) < 0.01
    # Reconstruct the mixture independently to check pair-level selection,
    # the 10% probability, and diagonal-before-sort ordering.
    model.uniform_fraction = 0.1
    torch.manual_seed(912)
    candidates = torch.sigmoid(torch.randn(2, 100000) * 0.8 - 0.8)
    mask = torch.rand(100000) < 0.1
    assert 0.095 < mask.float().mean() < 0.105
    candidates[:, mask] = torch.rand_like(candidates)[:, mask]
    candidates[1, :50000] = candidates[0, :50000]
    torch.manual_seed(912)
    t, s = model.sample_times(100000, torch.device('cpu'))
    torch.testing.assert_close(t, candidates.min(dim=0).values, rtol=0, atol=0)
    torch.testing.assert_close(s, candidates.max(dim=0).values, rtol=0, atol=0)


def check_pixel_backbone(device):
    kwargs = dict(input_size=4, in_channels=4, depth=2, hidden_size=32,
                  patch_size=2, num_heads=4, num_classes=1000, learn_sigma=True,
                  null_class_cond=True, compile=False)
    original = MMDiT(**kwargs)
    kwargs.update(input_size=8, patch_size=4, in_channels=3, learn_sigma=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'sit.pt'
        torch.save(original.state_dict(), path)
        pixel = PixelMMDiT(**kwargs, load_from_ckpt=path)
    for key, value in original.state_dict().items():
        if not key.startswith(('x_embedder.', 'final_layer.linear.')):
            torch.testing.assert_close(pixel.state_dict()[key], value, rtol=0, atol=0)
    assert not pixel.final_layer.linear.weight.any()
    nn.init.normal_(pixel.final_layer.linear.weight, std=0.02)
    for block in pixel.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    pixel = pixel.to(device)
    cond = torch.randn(2, 1, 32, device=device, requires_grad=True)
    model = PixelMeanFlow(pixel)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        loss = model.training_losses(torch.randn(2, 3, 8, 8, device=device), concat_tokens=cond)
    loss.backward()
    assert torch.isfinite(loss) and cond.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in pixel.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--pmf', action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(13)
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='pmf' if args.pmf else 'imf')
    assert cfg.trainer_module.params.sample_kwargs.num_steps == 1
    assert cfg.trainer_module.params.dino_loss_weight == 0.1
    assert not cfg.model.params.net_cfg.params.compile
    if args.pmf:
        assert cfg.trainer_module.params.first_stage_cfg is None
        assert cfg.model.params.time_mean == -0.8 and cfg.model.params.time_std == 0.8
        check_pixel_math()
        check_pixel_backbone(device)
    else:
        check_identity()
        check_backbone(device)
    check_reptok(device, pixel=args.pmf)
    print(('PMF' if args.pmf else 'IMF') + ' PASSED: identity/JVP gradients, NFE, SiT loading, real MMDiT, '
          f'CLS/projection gradients, frozen teacher, {device.type} BF16, validation and state restore.')

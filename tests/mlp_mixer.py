"""Offline Mixer contracts; optional parity against the historical implementation."""
import argparse
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldm.generation import TokenFlow
from ldm.models.mlp_mixer import MLPMixer1D

parser = argparse.ArgumentParser()
parser.add_argument('--legacy-source', type=Path)
args = parser.parse_args()
torch.set_num_threads(1)

with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
    standard = compose(config_name='gen')
    cfg = compose(config_name='gen_mlp')
net_cfg = OmegaConf.to_container(cfg.model.params.net_cfg.params, resolve=True)
# This experiment only changes the prior architecture, not the codec or protocol.
for key in ['data', 'trainer_params', 'checkpoint_params', 'lr_scheduler']:
    assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(standard[key], resolve=True)
for key in ['decoder_steps', 'sample_kwargs', 'lr', 'ema_rate']:
    assert cfg.trainer_module.params[key] == standard.trainer_module.params[key]
with torch.device('meta'):
    full = MLPMixer1D(**net_cfg)
    count = sum(p.numel() for p in full.parameters())
assert count == 189486852, count
assert not any(isinstance(m, torch.nn.MultiheadAttention) for m in full.modules())
del full

small = dict(n_channels=1, channels=32, dim=48, depth=2, num_classes=3,
             expansion_factor=4, expansion_factor_token=2, class_dropout_prob=.1)
torch.manual_seed(7)
net = MLPMixer1D(**small).eval()
x, t, labels = torch.randn(2, 1, 32), torch.tensor([.2, .6]), torch.tensor([0, 1])
out = net(x, t, y=labels)
assert out.shape == x.shape and torch.isfinite(out).all()
assert not torch.allclose(out, net(x, 1-t, y=labels)), 'Time conditioning is disconnected.'
assert not torch.allclose(out, net(x, t, y=labels.flip(0))), 'Class conditioning is disconnected.'
flow = TokenFlow(net)
null = torch.full_like(labels, 3)
expected = net(x, t, y=null) + 1.7 * (out - net(x, t, y=null))
torch.testing.assert_close(flow(x, t, y=labels, cfg_scale=1.7), expected)
# The null embedding used by CFG must also receive training gradients.
net.train()
with patch.object(net.label_embedder, 'dropout_prob', 1.):
    loss = flow.training_losses(x, y=labels)
    loss.backward()
assert net.label_embedder.embedding_table.weight.grad[3].abs().sum() > 0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
assert net.pos_embed.grad.abs().sum() > 0
assert net.t_embedder.mlp[0].weight.grad.abs().sum() > 0

if args.legacy_source:
    spec = importlib.util.spec_from_file_location('legacy_mixer', args.legacy_source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    with patch('torch.compile', side_effect=lambda module, **kwargs: module):
        torch.manual_seed(19)
        old = legacy.MLPMixer1D(**small)
        torch.manual_seed(19)
        new = MLPMixer1D(**small)
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    for training in [False, True]:
        old.train(training); new.train(training)
        old.zero_grad(); new.zero_grad()
        torch.manual_seed(23)
        a = old(x, t, y=labels)
        torch.manual_seed(23)
        b = new(x, t, y=labels)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for p, q in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    print('Historical Mixer: exact initialization, train/eval outputs, and gradients.')
print(f'MIXER PASSED: {count:,} parameters; configuration, shape, conditioning, CFG and gradients.')

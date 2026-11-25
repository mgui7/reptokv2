"""Compare against the actual old Lightning implementation from git history.

Run in a temporary test environment containing Lightning 2.5.0.post0. The
production trainer never imports it. No model weights or ImageNet are needed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import torch

p = argparse.ArgumentParser()
p.add_argument('--legacy-ref', default='33b4277437db35929c5468feecff76b2cf224ac2')
p.add_argument('--continuous-rng', action='store_true')
p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], default='32-true')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]


def compare(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda msg: f'{path}: {msg}')
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a: compare(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, f'{path}[{i}]')
    else:
        assert a == b, (path, a, b)


with tempfile.TemporaryDirectory(prefix='reptok-parity-') as temp:
    temp = Path(temp)
    archive = temp / 'legacy.tar'
    with archive.open('wb') as f:
        subprocess.run(['git', 'archive', args.legacy_ref], cwd=root, stdout=f, check=True)
    legacy = temp / 'legacy'
    legacy.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(legacy, filter='data')
    for accumulation in [1, 2]:
        for mode, source in [('legacy', legacy), ('native', root)]:
            subprocess.run([
                sys.executable, str(root / 'tests/parity_worker.py'), mode,
                '--source', str(source), '--output', str(temp / f'{mode}.pt'),
                '--accumulation', str(accumulation), '--device', args.device,
                '--precision', args.precision,
                *(['--continuous-rng'] if args.continuous_rng else []),
            ], cwd=source, check=True)
        old = torch.load(temp / 'legacy.pt', weights_only=False)
        new = torch.load(temp / 'native.pt', weights_only=False)
        compare(old, new)
        print(f'EXACT MATCH: {args.device}, {args.precision}, accumulation={accumulation}; '
              'initial state, losses, gradients, optimizer, LR, EMA, reconstructions and validation.', flush=True)

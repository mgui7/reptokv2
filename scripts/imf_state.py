"""Record a successful MeanFlow segment before scheduling its continuation."""
import csv
import json
import math
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

root, job, target = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
config = OmegaConf.load(sys.argv[4]) if len(sys.argv) > 4 else None
name = config.name if config is not None else 'recon'
folders = list((root / 'logs' / name).glob(f'*/{job}/checkpoints'))
if len(folders) != 1:
    raise RuntimeError(f'Expected one checkpoint directory, found {folders}')
folder = folders[0]
path = next((folder / name for name in ('final.ckpt', 'interrupted.ckpt')
             if (folder / name).is_file()), None)
if path is None:
    raise RuntimeError('No final or gracefully interrupted checkpoint; do not continue.')
state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
step = int(state['global_step'])
record = root / 'recon.json'
previous = json.loads(record.read_text())['step'] if record.exists() else 0
if not previous < step <= target:
    raise RuntimeError(f'Unexpected training progress: {previous} -> {step}/{target}')
if len(state['rng_states']) != 4:
    raise RuntimeError('Expected four GPU ranks.')
flow = state['hyper_parameters']['flow_cfg']
if config is not None:
    expected = OmegaConf.to_container(config.trainer_module.params.flow_cfg, resolve=True)
    if flow != expected:
        raise RuntimeError('Checkpoint decoder/sampler does not match the experiment config.')
elif flow['target'] != 'ldm.imf.ImprovedMeanFlow':
    raise RuntimeError('Unexpected decoder objective.')
with (folder.parent / 'metrics.csv').open() as stream:
    rows = list(csv.DictReader(stream))
losses = [float(row['train/loss']) for row in rows if row.get('train/loss')]
if not losses or not all(math.isfinite(loss) for loss in losses):
    raise RuntimeError('Missing or non-finite training losses; do not continue.')
for row in rows:
    for key, value in row.items():
        if key.startswith(('train/', 'val/')) and value and not math.isfinite(float(value)):
            raise RuntimeError(f'Non-finite metric {key}: {value}')
result = dict(job_id=job, step=step, target=target, complete=step == target,
              checkpoint=str(path), last_train_loss=losses[-1])
temporary = record.with_suffix('.tmp')
temporary.write_text(json.dumps(result, indent=2) + '\n')
temporary.replace(record)
print(step, path)

"""Track only completed, readable checkpoints from this bounded baseline run."""
import csv
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
stage, job, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
import torch
folders = list((BASE/'logs'/stage).glob(f'*/{job}/checkpoints'))
if len(folders) != 1:
    raise RuntimeError(f'Expected one checkpoint directory for {stage}/{job}, found {folders}')
folder = folders[0]
path = next((folder/name for name in ('final.ckpt','interrupted.ckpt') if (folder/name).is_file()), None)
if path is None:
    raise RuntimeError('No final or gracefully interrupted checkpoint; do not continue.')
state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
step = int(state['global_step'])
record = BASE/f'{stage}.json'
previous = json.loads(record.read_text())['step'] if record.exists() else 0
if step <= previous:
    raise RuntimeError(f'No training progress: {previous} -> {step}')
if len(state['rng_states']) != 4:
    raise RuntimeError('Unexpected world size')
with (folder.parent/'metrics.csv').open() as file:
    rows=list(csv.DictReader(file))
losses=[float(row['train/loss']) for row in rows if row.get('train/loss')]
if not losses or not all(math.isfinite(loss) for loss in losses):
    raise RuntimeError('Missing or non-finite training losses; do not continue.')
for row in rows:
    for key, value in row.items():
        if key.startswith('val/') and value and not math.isfinite(float(value)):
            raise RuntimeError(f'Non-finite validation metric {key}: {value}')
if stage.startswith('pilot') and step != target:
    raise RuntimeError(f'Pilot did not reach its target: {step}/{target}')
result=dict(stage=stage, job_id=job, step=step, target=target, complete=step>=target,
            checkpoint=str(path), last_train_loss=losses[-1])
temporary=record.with_suffix('.tmp')
temporary.write_text(json.dumps(result,indent=2)+'\n')
temporary.replace(record)
print(step, path)

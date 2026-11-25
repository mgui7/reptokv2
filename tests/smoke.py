"""Exercise the real entry point, checkpoints and DDP without external weights."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fixtures import model_params, imf_params, pmf_params

p = argparse.ArgumentParser()
p.add_argument('--ddp', action='store_true', help='Two CPU processes; no second GPU required.')
objective = p.add_mutually_exclusive_group()
objective.add_argument('--imf', action='store_true', help='Exercise the stage-1 iMF decoder.')
objective.add_argument('--pmf', action='store_true', help='Exercise the pixel pMF decoder.')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
    cfg = compose(config_name='config')
OmegaConf.set_struct(cfg, False)
cfg.name = 'smoke'
cfg.data = {'target': 'fixtures.TinyData', 'params': {'batch_size': 2, 'val_batch_size': 2}}
cfg.trainer_module = {'target': 'fixtures.build_reptok',
                      'params': pmf_params() if args.pmf else imf_params() if args.imf else model_params()}
cfg.trainer_params.max_steps = 2
cfg.trainer_params.val_check_interval = 2
cfg.trainer_params.limit_val_batches = 2
cfg.trainer_params.log_every_n_steps = 1
cfg.trainer_params.precision = '32-true' if args.ddp else 'bf16-mixed'
cfg.checkpoint_params.every_n_train_steps = 2
env = os.environ.copy()
env['PYTHONPATH'] = str(root / 'tests')
env['OMP_NUM_THREADS'] = '1'
if args.ddp:
    env['CUDA_VISIBLE_DEVICES'] = ''
command = [sys.executable]
if args.ddp:
    command += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2']
command += [str(root / 'train.py')]
with tempfile.TemporaryDirectory(prefix='reptok-smoke-') as temp:
    temp = Path(temp)
    cfg.log_root = str(temp / 'logs')
    OmegaConf.save(cfg, temp / 'smoke.yaml')
    command += ['--config-path', str(temp), '--config-name', 'smoke',
                'hydra.run.dir=' + str(temp / 'hydra'), 'hydra.output_subdir=null']
    subprocess.run(command + ['slurm_id=first'], cwd=root, env=env, check=True)
    checkpoint = next((temp / 'logs').glob('**/first/checkpoints/final.ckpt'))
    state = torch.load(checkpoint, weights_only=False)
    assert state['global_step'] == 2 and state['batch_step'] == 2
    assert len(state['rng_states']) == (2 if args.ddp else 1)
    assert state['hyper_parameters']['tokenizer_cfg']['params']['encoder']['params']['trainable_layers'] == ['embeddings.cls_token']
    subprocess.run(command + ['slurm_id=resumed', 'trainer_params.max_steps=4',
                             f'resume_checkpoint={checkpoint}'], cwd=root, env=env, check=True)
    final = next((temp / 'logs').glob('**/resumed/checkpoints/final.ckpt'))
    resumed = torch.load(final, weights_only=False)
    assert resumed['global_step'] == 4 and resumed['batch_step'] == 4
    assert resumed['lr_schedulers'][0]['last_epoch'] == 4
    assert all(v['step'].item() == 4 for v in resumed['optimizer_states'][0]['state'].values())
    assert any(not torch.equal(state['state_dict'][k], resumed['state_dict'][k])
               for k in state['state_dict'] if k.startswith('model.'))
    print('SMOKE PASSED:', 'pMF' if args.pmf else 'iMF' if args.imf else 'FM',
          'two-process CPU DDP' if args.ddp else 'single-process mixed precision',
          'train, validate, checkpoint, resume; metric values are synthetic proxies.')

"""Run one side of the comparison in its own Python process and source tree."""
import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['legacy', 'native'])
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
p.add_argument('--accumulation', type=int, default=1)
p.add_argument('--precision', default='32-true')
p.add_argument('--device', default='cpu')
p.add_argument('--continuous-rng', action='store_true')
args = p.parse_args()
sys.path.insert(0, args.source)
from fixtures import model_params, TinyMetricTracker, TinyData
import ldm.trainer as training
training.ImageMetricTracker = TinyMetricTracker

os.environ['PL_GLOBAL_SEED'] = '2023'
os.environ['PL_SEED_WORKERS'] = '0'
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')
params = model_params()
trace = []
validation = []


def snapshot(module, optimizer, scheduler, metrics):
    return {
        'state': {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
        'grads': {k: v.grad.detach().cpu().clone() for k, v in module.named_parameters() if v.grad is not None},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'metrics': {k: torch.as_tensor(v).detach().cpu().clone() for k, v in metrics.items()},
        'trainable': [k for k, v in module.named_parameters() if v.requires_grad],
    }


if args.mode == 'legacy':
    from lightning.pytorch import Trainer, Callback
    from lightning.pytorch.loggers import TensorBoardLogger
    from ldm.trainer_other import TrainerModuleDINOLoss

    class Capture(Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            # Identical forward RNG lets this check isolate training semantics
            # from framework loader-prefetch differences.
            if not args.continuous_rng:
                torch.manual_seed(1000 + batch_idx)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            metrics = {k: v for k, v in trainer.callback_metrics.items() if k.startswith('train/')}
            trace.append(snapshot(module, trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler, metrics))

        def on_validation_epoch_end(self, trainer, module):
            validation.append({
                'metric': module.metric_tracker.aggregate(),
                'loss': torch.stack(module.val_losses).mean(0).mean(),
                'panels': {k: v.cpu().clone() for k, v in module.val_images.items()},
            })

    module = TrainerModuleDINOLoss(**params)
    # Capture initializer and requires_grad before the framework changes modes.
    initial = snapshot(module, torch.optim.AdamW([p for p in module.parameters() if p.requires_grad]),
                       torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(module.parameters()), lambda _: 1), {})
    trainer = Trainer(accelerator=args.device, devices=1, max_steps=4,
                      strategy='ddp' if args.device == 'gpu' else 'auto',
                      accumulate_grad_batches=args.accumulation,
                      precision=args.precision, gradient_clip_val=1.0,
                      num_sanity_val_steps=0, val_check_interval=2 * args.accumulation,
                      limit_val_batches=2, log_every_n_steps=1,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, callbacks=[Capture()],
                      logger=TensorBoardLogger('/tmp/reptok-parity-tb'))
    trainer.fit(module, train_dataloaders=TinyData().train_dataloader(),
                val_dataloaders=TinyData().val_dataloader())
else:
    from ldm.train_utils import train_batch, move_batch
    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')
    module = training.RepTok(**params)
    initial = snapshot(module, torch.optim.AdamW([p for p in module.parameters() if p.requires_grad]),
                       torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(module.parameters()), lambda _: 1), {})
    module.to(device).train()
    if device.type == 'cuda':
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
    optimizer, scheduler = module.configure_optimizers()
    loader = iter(TinyData().train_dataloader())
    for batch_idx in range(4 * args.accumulation):
        batch = move_batch(next(loader), device)
        if not args.continuous_rng:
            torch.manual_seed(1000 + batch_idx)
        metrics, did_step = train_batch(module, batch, optimizer, scheduler, batch_idx,
                                       args.accumulation, 1.0, device,
                                       args.precision == 'bf16-mixed')
        trace.append(snapshot(module, optimizer, scheduler, metrics))
        if (batch_idx + 1) % (2 * args.accumulation) == 0:
            optimizer.zero_grad(set_to_none=True)
            values, panels = module.validate(TinyData().val_dataloader(), device, 2,
                                            args.precision == 'bf16-mixed')
            validation.append({'metric': {k: v for k, v in values.items() if k != 'loss'},
                               'loss': values['loss'], 'panels': {k: v.cpu() for k, v in panels.items()}})

# Move nested GPU state to CPU for portable comparisons.
def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value

torch.save(cpu(dict(initial=initial, trace=trace, validation=validation)), args.output)

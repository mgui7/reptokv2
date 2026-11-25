"""RepTok reconstruction training with native PyTorch / torchrun."""
import csv
import datetime
import signal
import time
from pathlib import Path

import hydra
import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from jutils import instantiate_from_config

from ldm.train_utils import (
    setup_distributed, seed_everything, move_batch, wrap_ddp, train_batch,
    reduce_metrics, save_checkpoint, load_checkpoint, restore_rng,
)


def check_config(cfg):
    if cfg.resume_checkpoint and cfg.load_weights:
        raise ValueError('Set only one of resume_checkpoint and load_weights.')
    p = cfg.trainer_params
    if p.precision not in ('bf16-mixed', '32-true', '32', 32):
        raise ValueError('Only the baseline bf16-mixed and full precision are supported.')
    if p.accumulate_grad_batches < 1 or p.max_steps == 0 or p.max_steps < -1:
        raise ValueError('Invalid accumulation or max_steps.')
    if p.limit_val_batches < 0 or p.val_check_interval < 1:
        raise ValueError('Invalid validation interval or batch count.')
    if p.limit_val_batches and p.val_check_interval % p.accumulate_grad_batches:
        raise ValueError('Validation must follow a complete accumulation window.')


class RunLogger:
    def __init__(self, directory, enabled):
        self.writer = SummaryWriter(directory) if enabled else None
        self.file = None
        if enabled:
            self.file = open(Path(directory) / 'metrics.csv', 'a', newline='')
            fields = ['step', 'batch_step', 'train/loss', 'train/cos_sim',
                      'train/cos_sim_loss', 'lr-AdamW', 'val/fid', 'val/gfid', 'val/ssim',
                      'val/psnr', 'val/mse', 'val/n_metric_samples', 'val/loss']
            self.csv = csv.DictWriter(self.file, fieldnames=fields)
            if self.file.tell() == 0:
                self.csv.writeheader()

    def log(self, values, step, batch_step):
        if self.writer is None:
            return
        self.csv.writerow(dict(step=step, batch_step=batch_step, **values))
        self.file.flush()
        for name, value in values.items():
            self.writer.add_scalar(name, value, step)

    def images(self, panels, step):
        if self.writer is None:
            return
        for name, images in panels.items():
            b, c, h, w = images.shape
            # Same grid grouping as the original ims_to_grid(split=4).
            if b % 4 == 0:
                grid = images.reshape(4, b // 4, c, h, w).permute(2, 0, 3, 1, 4)
                grid = grid.reshape(c, 4 * h, b // 4 * w)
            else:
                grid = images.permute(1, 0, 2, 3).reshape(c, b * h, w)
            self.writer.add_image(f'val/{name}/samples', grid.cpu(), step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.file.close()


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    started = time.monotonic()
    device, rank, world = setup_distributed()
    torch.set_float32_matmul_precision('high')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    check_config(cfg)
    seed_everything(cfg.seed)
    params = cfg.trainer_params
    bf16 = params.precision == 'bf16-mixed'
    data = instantiate_from_config(cfg.data)
    # Full stage-2 checkpoints already contain the frozen reconstruction model.
    # Construct its saved architecture without reopening the old recon path.
    source = cfg.resume_checkpoint or cfg.load_weights
    if source and cfg.trainer_module.target == 'ldm.generation.TokenGenerator':
        saved = torch.load(source, map_location='cpu', weights_only=False)
        cfg.trainer_module.params = OmegaConf.create(saved['hyper_parameters'])
        del saved
    elif source and cfg.trainer_module.target == 'ldm.trainer.RepTok':
        encoder = cfg.trainer_module.params.tokenizer_cfg.params.encoder.target
        if encoder == 'ldm.self_flow_encoder.FrozenSelfFlowEncoder':
            from ldm.codec import recon_resume_params
            saved = torch.load(source, map_location='cpu', weights_only=False)
            cfg.trainer_module.params = OmegaConf.create(recon_resume_params(saved))
            del saved
    module = instantiate_from_config(cfg.trainer_module).to(device).train()
    if cfg.trainer_module.target == 'ldm.generation.TokenGenerator':
        cfg.trainer_module.params = OmegaConf.create(module.hyper_parameters)
    config = OmegaConf.to_container(cfg, resolve=True)
    if device.type == 'cuda':
        # Legacy train.py selects DDP even on one GPU; DDPStrategy resets the
        # global seed after model construction, before creating data iterators.
        seed_everything(cfg.seed)
    optimizer, scheduler = module.configure_optimizers()
    step, batch_step, checkpoint = 0, 0, None
    if cfg.resume_checkpoint:
        checkpoint = load_checkpoint(cfg.resume_checkpoint, module, optimizer, scheduler)
        step = int(checkpoint['global_step'])
        batch_step = int(checkpoint.get('batch_step', step * params.accumulate_grad_batches))
    elif cfg.load_weights:
        load_checkpoint(cfg.load_weights, module, strict=cfg.load_weights_strict)
    wrapped = wrap_ddp(module, device)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader() if params.limit_val_batches else None
    train_iterator = iter(train_loader)
    if checkpoint is not None and 'rng_states' in checkpoint:
        if len(checkpoint['rng_states']) != world:
            raise ValueError('Resume requires the same world size to restore rank RNG states.')
        restore_rng(checkpoint['rng_states'][rank])

    stamp = datetime.datetime.now().strftime('%Y-%m-%d/T%H%M%S')
    if cfg.slurm_id is not None:
        stamp = datetime.datetime.now().strftime('%Y-%m-%d/') + str(cfg.slurm_id)
    if dist.is_initialized():
        values = [stamp]
        dist.broadcast_object_list(values, src=0)
        stamp = values[0]
    log_dir = Path(cfg.log_root) / cfg.name / stamp
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / 'config.yaml')
        OmegaConf.save(cfg.trainer_module.params, log_dir / 'hparams.yaml')
        global_batch = cfg.data.params.batch_size * world * params.accumulate_grad_batches
        samples = cfg.data.params.val_batch_size * world * params.limit_val_batches
        print(f'Log directory: {log_dir}\nGlobal batch: {global_batch}; validation samples: {samples}', flush=True)
    logger = RunLogger(str(log_dir), rank == 0)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)

    def validate():
        # Lightning clears gradients before validation at optimizer boundaries.
        optimizer.zero_grad(set_to_none=True)
        values, panels = module.validate(val_loader, device, params.limit_val_batches, bf16)
        values = reduce_metrics({f'val/{k}': v for k, v in values.items()}, device)
        logger.log(values, step, batch_step)
        logger.images(panels, step)
        if rank == 0:
            metric = getattr(module, 'validation_metric', 'fid')
            label = 'gFID' if metric == 'gfid' else 'rFID'
            print(f"Step {step}: {label} {values[f'val/{metric}']:.4f} @ {int(values['val/n_metric_samples'])} samples", flush=True)

    if checkpoint is not None and checkpoint.get('validation_pending'):
        validate()
    try:
        while params.max_steps < 0 or step < params.max_steps:
            # The original WebDataset repeats indefinitely. An exhausted loader is
            # an error, not a reason to silently change epoch/scheduler semantics.
            batch = move_batch(next(train_iterator), device)
            lr_before_forward = optimizer.param_groups[0]['lr']
            metrics, did_step = train_batch(
                wrapped, batch, optimizer, scheduler, batch_step,
                params.accumulate_grad_batches, params.gradient_clip_val, device, bf16,
                error_if_nonfinite=params.get('error_if_nonfinite', False),
            )
            batch_step += 1
            if not did_step:
                continue
            step += 1
            if params.log_every_n_steps > 0 and step % params.log_every_n_steps == 0:
                values = reduce_metrics(metrics, device)
                values['lr-AdamW'] = lr_before_forward
                logger.log(values, step, batch_step)
                if rank == 0:
                    message = f"Step {step}: flow={values['train/loss']:.6f}"
                    if 'train/cos_sim_loss' in values:
                        message += f", cosine={values['train/cos_sim_loss']:.6f}"
                    print(message, flush=True)
            should_validate = bool(params.limit_val_batches and batch_step % params.val_check_interval == 0)
            interval = cfg.checkpoint_params.every_n_train_steps
            if interval > 0 and step % interval == 0:
                path = log_dir / 'checkpoints' / f'step{step:06d}.ckpt'
                save_checkpoint(path, module, optimizer, scheduler, step, batch_step, config, should_validate)
                if rank == 0:
                    last = path.parent / 'last.ckpt'
                    if last.is_symlink() or last.exists():
                        last.unlink()
                    last.symlink_to(path.name)
            if should_validate:
                validate()
            max_time = params.get('max_time_seconds')
            if max_time is not None and time.monotonic() - started >= max_time:
                stopped = True
            flag = torch.tensor(int(stopped), device=device)
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stopped = bool(flag.item())
            if stopped:
                break
        # Keep an explicit final state; periodic checkpoints retain legacy timing.
        name = 'interrupted.ckpt' if stopped else 'final.ckpt'
        save_checkpoint(log_dir / 'checkpoints' / name, module, optimizer, scheduler,
                        step, batch_step, config)
    finally:
        logger.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

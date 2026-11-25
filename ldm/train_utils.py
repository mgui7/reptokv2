"""Small PyTorch loop utilities; checkpoint keys remain compatible with old runs."""
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook


def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world


def seed_everything(seed):
    # The legacy Lightning seed is identical on all ranks, not seed + rank.
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    os.environ['PL_SEED_WORKERS'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    if isinstance(batch, (torch.Tensor, np.ndarray)):
        return torch.as_tensor(batch).to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch(v, device) for v in batch)
    return batch


def wrap_ddp(module, device):
    if not dist.is_initialized():
        return module
    world = dist.get_world_size()
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', world))
    multi_node = world > local_world
    wrapped = DistributedDataParallel(
        module, device_ids=[device.index] if device.type == 'cuda' else None,
        gradient_as_bucket_view=multi_node,
    )
    if multi_node:
        # Match the original multi-node Lightning DDPStrategy.
        wrapped.register_comm_hook(None, fp16_compress_hook)
    return wrapped


def train_batch(module, batch, optimizer, scheduler, batch_idx, accumulation,
                clip_norm, device, use_bf16=False, error_if_nonfinite=False):
    """Match Lightning's closure: forward -> scheduler -> zero_grad -> backward."""
    step = (batch_idx + 1) % accumulation == 0
    context = module.no_sync() if hasattr(module, 'no_sync') and not step else nullcontext()
    with context:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss, metrics = module(batch)
            if scheduler is not None:
                # The old training_step calls this for every microbatch.
                scheduler.step()
            loss = loss / accumulation
        if batch_idx % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
        if error_if_nonfinite and not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
    if step:
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad], clip_norm,
                error_if_nonfinite=error_if_nonfinite,
            )
        optimizer.step()
    return metrics, step


def reduce_metrics(metrics, device):
    values = {k: torch.as_tensor(v, device=device).detach().clone() for k, v in metrics.items()}
    if dist.is_initialized():
        for value in values.values():
            dist.all_reduce(value)
            value /= dist.get_world_size()
    return {k: v.item() for k, v in values.items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, module, optimizer, scheduler, step, batch_step, config,
                    validation_pending=False):
    states = [rng_state()]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, rng_state())
    if not dist.is_initialized() or dist.get_rank() == 0:
        checkpoint = dict(
            global_step=step, batch_step=batch_step,
            state_dict=module.state_dict(), optimizer_states=[optimizer.state_dict()],
            lr_schedulers=[scheduler.state_dict()] if scheduler is not None else [],
            hyper_parameters=module.hyper_parameters, hparams_name='kwargs',
            config=config, rng_states=states, latent_shape=module.latent_shape,
            validation_pending=validation_pending, format_version=1,
        )
        if hasattr(module, 'export_codec_config'):
            checkpoint['codec_config'] = module.export_codec_config()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, module, optimizer=None, scheduler=None, strict=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    module.load_state_dict(checkpoint['state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    module.latent_shape = checkpoint.get('latent_shape')
    return checkpoint

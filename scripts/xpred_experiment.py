"""Prepare/audit the authorized sequential x-pred experiment. No submissions here."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jutils import instantiate_from_config
from ldm.codec import canonical_state, plain
from ldm.generation import TokenGenerator
from ldm.fd_posttrain import FDPosttrainRepTok, clean_fd_initialization
from ldm.train_utils import move_batch, load_checkpoint


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def configuration(stage, directory):
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        cfg = compose(config_name='gen_xpred' if stage == 'gen' else 'fd_xpred', overrides=[
            'data.params.tar_base=/e/project1/jureap52/schb/data/imagenet256_latents_wds',
            'trainer_module.params.decoder_noise_reference_std=1.3787409924304508',
        ] if stage == 'fd' else [
            'data.params.tar_base=/e/project1/jureap52/schb/data/imagenet256_latents_wds'])
    cfg.log_root = str(directory/'logs')
    cfg.load_weights = str(directory/'initialization.ckpt')
    return cfg


def construct(stage, hp):
    return (TokenGenerator if stage == 'gen' else FDPosttrainRepTok)(**hp)


def save_init(directory, cfg, mapped):
    assert not (directory/'initialization.ckpt').exists(), 'Never overwrite an initialization.'
    torch.save(mapped, directory/'initialization.ckpt')
    cfg.trainer_module.params = OmegaConf.create(mapped['hyper_parameters'])
    OmegaConf.save(cfg, directory/'resolved-config.yaml')
    module = construct('gen' if 'codec_config' in mapped['hyper_parameters'] else 'fd', mapped['hyper_parameters'])
    load_checkpoint(directory/'initialization.ckpt', module, strict=True)
    assert tuple(module.latent_shape) == tuple(mapped['latent_shape'])
    optimizer, _ = module.configure_optimizers()
    planned = {id(p) for p in module.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(actual) == len(set(actual)) and set(actual) == planned
    if isinstance(module, TokenGenerator):
        assert module.model.prediction_type == 'x'
        assert not any(p.requires_grad for p in module.codec.parameters())
    else:
        assert module.prior.prediction_type == 'x' and int(module.fd.updates) == 0
        assert all(a.data_ptr() != b.data_ptr() for a,b in zip(module.model.parameters(), module.ema_model.parameters()))
        assert not any(p.requires_grad for p in module.model.net.t_embedder.mlp.parameters())
    write(directory/'checks/initialization.json', dict(passed=True,
        strict_full_size_load=True, optimizer_covers_exact_trainable_parameters=True,
        trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad),
        **mapped['initialization']))


def prepare_gen(directory, checkpoint):
    base = load(checkpoint)
    hp = plain(base['hyper_parameters'])
    hp['flow_cfg']['params'].update(prediction_type='x', t_eps=.05)
    hp.update(lr=1e-4, ema_rate=0., sample_kwargs=dict(num_steps=50, progress=False, cfg_scale=1.), decoder_steps=50)
    # The complete G is initialized independently from seed 2023; no old G weights.
    torch.manual_seed(2023)
    module = TokenGenerator(**hp)
    state = canonical_state(base['state_dict'])
    codec = {k[len('codec.'):]:v for k,v in state.items() if k.startswith('codec.')}
    module.codec.load_state_dict(codec, strict=True)
    for k,v in module.codec.state_dict().items():
        assert torch.equal(v, codec[k]), k
    assert not torch.equal(module.model.net.final_layer.linear.weight,
                           state['model.net.final_layer.linear.weight'])
    mapped = dict(state_dict=module.state_dict(), hyper_parameters=module.hyper_parameters,
                  global_step=0, latent_shape=module.latent_shape,
                  initialization=dict(source_checkpoint=str(checkpoint),
                  source_step=int(base['global_step']), generator_initialization='fresh',
                  seed=2023, prediction_type='x', t_eps=.05, frozen_codec_exact=True,
                  optimizer_reset=True, scheduler_reset=True))
    # Avoid keeping a second full network resident during strict-load verification.
    del module, base, state, codec
    save_init(directory, configuration('gen', directory), mapped)


def prepare_fd(directory, gen_directory):
    result = json.loads((gen_directory/'result.json').read_text())
    assert result['passed'] and result['terminal_step'] > 0
    checkpoint = Path(result['terminal_checkpoint'])
    assert checkpoint.name in ('final.ckpt', 'interrupted.ckpt'), 'Use final state, never best.'
    base = load(checkpoint)
    assert base['hyper_parameters']['flow_cfg']['params']['prediction_type'] == 'x'
    cfg = configuration('fd', directory)
    settings = OmegaConf.to_container(cfg.trainer_module.params, resolve=True)
    mapped = clean_fd_initialization(base, settings)
    mapped['global_step'] = 0
    mapped['initialization'].update(source_checkpoint=str(checkpoint),
        source_step=int(base['global_step']), prediction_type='x',
        generator_initialization='trained_xpred_final', fd_statistics_reset=True,
        scheduler_reset=True, one_step_direct_output=True)
    save_init(directory, cfg, mapped)
    # Values are copied exactly, including train/EMA decoder and trained G.
    for source, target in [('model.', 'prior.'), ('codec.flow.', 'model.'),
                           ('codec.flow.', 'ema_model.'), ('codec.tokenizer.', 'tokenizer.'),
                           ('codec.first_stage.', 'first_stage.')]:
        for key,value in base['state_dict'].items():
            if key.startswith(source):
                assert torch.equal(value, mapped['state_dict'][target+key[len(source):]]), key
    print('FD initialized from final trained x-pred generator:', checkpoint, flush=True)


def preflight(directory, stage):
    state = load(directory/'initialization.ckpt')
    hp = plain(state['hyper_parameters'])
    if stage == 'fd':
        hp['flow_cfg']['params']['net_cfg']['params']['compile'] = False
        hp['tokenizer_cfg']['params']['compile'] = False
    module = construct(stage, hp)
    # Exercise the SAME loader as train.py, including runtime metadata.
    load_checkpoint(directory/'initialization.ckpt', module, strict=True)
    assert module.latent_shape is not None
    del state
    module = module.cuda().train()
    config = OmegaConf.load(directory/'resolved-config.yaml').data
    config.params.val_batch_size = 32
    config.params.val_num_workers = 0
    config.params.multinode = False
    batch = move_batch(next(iter(instantiate_from_config(config).val_dataloader())), torch.device('cuda'))
    routes = ['generation'] if stage == 'gen' else ['reconstruction','clean_fd']
    result = dict(passed=False, optimizer_update_performed=False, real_images=32,
                  precision='BF16 autocast; x-to-v calculation FP32', routes={})
    for route in routes:
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if stage == 'gen':
                loss, _ = module(batch)
            else:
                imloss, tokenloss, raw, variance, tokens, generated = module.losses(batch)
                loss = imloss if route == 'reconstruction' else tokenloss
        assert torch.isfinite(loss), route
        loss.backward()
        networks = [('generator',module.model), ('codec',module.codec)] if stage == 'gen' else [
            ('encoder',module.tokenizer), ('decoder',module.model), ('generator',module.prior)]
        norms = {}
        for name,net in networks:
            grads = [p.grad.float().norm() for p in net.parameters() if p.grad is not None]
            assert all(torch.isfinite(v) for v in grads), (route,name)
            norms[name] = float(torch.stack(grads).norm()) if grads else 0.
        if stage == 'gen':
            assert norms['generator'] > 0 and norms['codec'] == 0
        else:
            assert norms['encoder'] > 0
            assert norms['decoder' if route == 'reconstruction' else 'generator'] > 0
            assert norms['generator' if route == 'reconstruction' else 'decoder'] == 0
            assert all(p.grad is None for p in module.model.net.t_embedder.mlp.parameters())
            assert all(p.grad is None for p in module.first_stage.parameters())
        result['routes'][route] = dict(loss=float(loss), gradient_norms=norms)
    if stage == 'fd':
        module.eval()
        noise = torch.randn(4,*module.token_shape,device='cuda')
        labels = batch['label'][:4].long()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            direct = module.prior.predict_clean(noise, torch.zeros(4,device='cuda'), y=labels).float()
            generated = module.generate_tokens(noise, labels)
            torch.testing.assert_close(direct, generated, rtol=0, atol=0)
        result['direct_output_exact'] = True
    # Also exercise inference: backward alone did not catch missing latent_shape.
    module.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        labels = batch['label'][:2].long()
        samples = (module.sample(labels) if stage == 'gen' else
                   module.sample_images(module.sample_tokens(labels), 0))
    assert tuple(samples.shape) == tuple(batch['image'][:2].shape)
    assert torch.isfinite(samples).all()
    result['sample_check'] = dict(images=2, shape=list(samples.shape), finite=True,
                                  token_steps=50 if stage=='gen' else 1, decoder_steps=50)
    result['training_loader_used'] = True
    result['passed'] = True
    write(directory/'checks/gpu-preflight.json', result)
    print(json.dumps(result), flush=True)


def finish(directory, stage, job_id):
    paths = [p for p in (directory/'logs').glob('**/checkpoints/*.ckpt')
             if p.name in ('final.ckpt','interrupted.ckpt')]
    assert len(paths) == 1, paths
    terminal = paths[0]
    final, initial = load(terminal), load(directory/'initialization.ckpt')
    step = int(final['global_step'])
    assert 0 < step <= 100000
    assert final['hyper_parameters'] == initial['hyper_parameters']
    frozen = [k for k in initial['state_dict'] if k.startswith('codec.')] if stage == 'gen' else [
        k for k in initial['state_dict'] if k.startswith('first_stage.') or
        k.startswith('model.net.t_embedder.mlp.') or k.startswith('ema_model.net.t_embedder.mlp.')]
    for key in frozen:
        assert torch.equal(final['state_dict'][key], initial['state_dict'][key]), key
    prefix = 'model.' if stage == 'gen' else 'prior.'
    assert any(not torch.equal(v,final['state_dict'][k]) for k,v in initial['state_dict'].items() if k.startswith(prefix))
    rows = list(csv.DictReader((terminal.parent.parent/'metrics.csv').open()))
    for row in rows:
        for key,value in row.items():
            if value: assert math.isfinite(float(value)), (key,value)
    evaluations = [{k:float(v) for k,v in row.items() if v and (k=='step' or k.startswith('val/'))}
                   for row in rows if row.get('val/gfid')]
    assert evaluations and evaluations[0]['step'] == 0
    assert (terminal.parent/'best.ckpt').exists()
    result = dict(passed=True, job_id=str(job_id), terminal_step=step,
        terminal_checkpoint=str(terminal), terminal_reason='step_limit' if terminal.name=='final.ckpt' else 'trainer_time_or_signal_stop',
        generator_prediction='x', gpu_preflight=json.loads((directory/'checks/gpu-preflight.json').read_text()),
        frozen_weights_exact=True, finite_metrics=True, evaluation_images=1024,
        sampling_token_steps=50 if stage=='gen' else 1, decoder_steps=50, cfg_scale=1.,
        initial_evaluation=evaluations[0], best_evaluation=min(evaluations,key=lambda r:r['val/gfid']),
        latest_evaluation=evaluations[-1], best_checkpoint=str(terminal.parent/'best.ckpt'),
        note='1024-image monitoring. Historical v runs differ in initialization/budget; not a matched causal comparison.')
    write(directory/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare-gen','prepare-fd','preflight','finish'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--stage',choices=['gen','fd'])
    p.add_argument('--gen-root',type=Path)
    p.add_argument('--job-id',default='local')
    args=p.parse_args()
    torch.set_num_threads(8)
    if args.action=='prepare-gen': prepare_gen(args.root,args.checkpoint)
    elif args.action=='prepare-fd': prepare_fd(args.root,args.gen_root)
    elif args.action=='preflight': preflight(args.root,args.stage)
    elif args.action=='finish': finish(args.root,args.stage,args.job_id)

if __name__=='__main__': main()

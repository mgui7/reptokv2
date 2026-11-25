"""Post-training mapping, encoder gradients, LR groups and gFID selection/resume."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params
from ldm.codec import plain
from ldm.ul import ULRepTok
from ldm.ul_posttrain import ULPosttrainRepTok, reptok_initialization
from ldm.train_utils import load_checkpoint, save_checkpoint, train_batch


class SelectionFixture(ULPosttrainRepTok):
    """Synthetic scores for checkpoint selection only; NOT measured FID."""
    gradient_log_interval = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('validation_calls', torch.tensor(0))

    def validate(self, loader, device, limit, use_bf16=False):
        scores = [3., 1., 2.]
        value = scores[min(int(self.validation_calls), 2)]
        self.validation_calls.add_(1)
        return {'gfid': value, 'fid': 10. - value, 'n_metric_samples': 4.}, {}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    base = plain(model_params())
    params = {k: base[k] for k in ['flow_cfg', 'tokenizer_cfg', 'first_stage_cfg', 'lr_scheduler_cfg']}
    params.update(lr=1e-5, ema_rate=.9, prior_weight=3., token_steps=3, decoder_steps=3)
    params['prior_cfg'] = {'target': 'ldm.generation.TokenFlow', 'params': {
        'net_cfg': {'target': 'ldm.models.transformer.token_dit.OneDimensionalTransformer', 'params': {
            'num_channels': 1, 'in_channels': 32, 'hidden_size': 32, 'depth': 2, 'num_heads': 4,
            'num_classes': 3, 'class_dropout_prob': .1, 'n_registers': 1, 'split_channels': 1}}}}
    donor = ULRepTok(**params)
    with torch.no_grad():
        # Simulate trained, nonzero modulation and output heads.
        for net in (donor.model, donor.prior):
            for p in net.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * .03)
    checkpoint = dict(global_step=100000, latent_shape=(1, 32), hyper_parameters={
        'flow_cfg': donor.hyper_parameters['prior_cfg'],
        'codec_config': {k: donor.hyper_parameters[k] for k in ['flow_cfg', 'tokenizer_cfg', 'first_stage_cfg']}},
        state_dict={})
    for net, prefix in [(donor.model, 'codec.flow.'), (donor.tokenizer, 'codec.tokenizer.'),
                        (donor.first_stage, 'codec.first_stage.'), (donor.prior, 'model.')]:
        checkpoint['state_dict'].update({prefix+k: v for k,v in net.state_dict().items()})
    mapped = reptok_initialization(checkpoint, dict(params, encoder_lr=1e-6))
    assert mapped['latent_shape'] == (4, 4, 4)
    assert 'optimizer_states' not in mapped and 'global_step' not in mapped
    module = ULPosttrainRepTok(**mapped['hyper_parameters'])
    module.load_state_dict(mapped['state_dict'], strict=True)
    for old, new in [(donor.model, module.model), (donor.model, module.ema_model),
                     (donor.prior, module.prior), (donor.tokenizer, module.tokenizer)]:
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    batch = {'image': torch.randn(3,3,4,4), 'latent': torch.randn(3,4,4,4), 'label': torch.tensor([0,1,2])}
    for kind in ['recon', 'prior']:
        module.zero_grad(set_to_none=True)
        image_loss, prior_loss, _, _ = module.losses(batch)
        (image_loss if kind == 'recon' else prior_loss).backward()
        assert module.tokenizer.cls_proj.fc.weight.grad.abs().sum() > 0
        assert module.tokenizer.encoder.embeddings.cls_token.grad.abs().sum() > 0
        other = module.prior if kind == 'recon' else module.model
        assert all(p.grad is None for p in other.parameters())
    module.zero_grad(set_to_none=True)
    loss, metrics = module(batch)
    torch.testing.assert_close(loss, metrics['train/image_loss'] + 3 * metrics['train/token_loss'])
    torch.testing.assert_close(metrics['train/weighted_prior_loss'], 3 * metrics['train/token_loss'])
    loss.backward()
    assert all(torch.isfinite(v) and v > 0 for v in module.gradient_metrics().values())
    assert all(p.grad is None for p in module.first_stage.parameters())
    optimizer, scheduler = module.configure_optimizers()
    assert [g['initial_lr'] for g in optimizer.param_groups] == [1e-5, 1e-6]
    groups = [p for g in optimizer.param_groups for p in g['params']]
    assert len(groups) == len({id(p) for p in groups})
    assert {id(p) for p in groups} == {id(p) for p in module.parameters() if p.requires_grad}
    print('PASS: exact trained E/D/G mapping, decoder EMA seed, image latent shape, both E gradients, weight 3 and LR groups', flush=True)
    with tempfile.TemporaryDirectory(prefix='ul-posttrain-test-') as directory:
        directory = Path(directory)
        init = directory/'initial.ckpt'; torch.save(mapped, init)
        load_checkpoint(init, module)
        # Freeze only D's time MLP: real optimizer updates must preserve it and
        # its EMA copy while E, the rest of D, and G continue changing.
        frozen = ULPosttrainRepTok(**dict(mapped['hyper_parameters'], freeze_decoder_time_mlp=True))
        frozen.load_state_dict(mapped['state_dict'], strict=True)
        before = {k: v.clone() for k, v in frozen.state_dict().items()}
        opt, sched = frozen.configure_optimizers()
        ids = {id(p) for group in opt.param_groups for p in group['params']}
        assert all(not p.requires_grad and id(p) not in ids for p in frozen.model.net.t_embedder.mlp.parameters())
        assert all(p.requires_grad and id(p) in ids for p in frozen.prior.net.t_embedder.mlp.parameters())
        for index in range(2):
            train_batch(frozen, batch, opt, sched, index, 1, 1., torch.device('cpu'))
        after = frozen.state_dict()
        for key in before:
            if key.startswith(('model.net.t_embedder.mlp.', 'ema_model.net.t_embedder.mlp.')):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        for prefix in ['tokenizer.', 'model.net.blocks.', 'prior.net.t_embedder.mlp.']:
            assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix)), prefix
        frozen_path = directory/'frozen.ckpt'
        save_checkpoint(frozen_path, frozen, opt, sched, 2, 2, {})
        reloaded = ULPosttrainRepTok.from_checkpoint(frozen_path)
        assert reloaded.freeze_decoder_time_mlp
        assert all(not p.requires_grad for p in reloaded.model.net.t_embedder.mlp.parameters())
        ropt, rsched = reloaded.configure_optimizers()
        load_checkpoint(frozen_path, reloaded, ropt, rsched)
        assert all(v['step'].item() == 2 for v in ropt.state.values())
        print('PASS: frozen decoder time MLP/EMA stay exact after updates; E/rest-D/G update; flag and optimizer restore', flush=True)
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
            cfg = compose(config_name='ul_posttrain_frozen_time')
        assert cfg.trainer_module.params.prior_weight == 3
        assert cfg.trainer_params.validate_before_training
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.trainer_module.target = 'ul_posttrain.SelectionFixture'
        cfg.trainer_module.params = frozen.hyper_parameters
        cfg.data = {'target': 'fixtures.TinyGenData', 'params': {'batch_size': 2, 'val_batch_size': 2}}
        cfg.log_root = str(directory/'logs')
        cfg.trainer_params.precision = '32-true'
        cfg.trainer_params.max_steps = 2
        cfg.trainer_params.limit_val_batches = 1
        cfg.trainer_params.val_check_interval = 2
        cfg.trainer_params.log_every_n_steps = 1
        cfg.checkpoint_params.every_n_train_steps = 0
        OmegaConf.save(cfg, directory/'test.yaml')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'tests'), OMP_NUM_THREADS='1')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                   str(ROOT/'train.py'), '--config-path', str(directory), '--config-name', 'test']
        subprocess.run(command+['slurm_id=first'], cwd=ROOT, env=env, check=True)
        first = next((directory/'logs').glob('**/first/checkpoints/final.ckpt'))
        best = torch.load(first.parent/'best.ckpt', weights_only=False)
        assert best['global_step'] == 2 and best['best_validation'] == dict(metric='gfid', value=1., step=2)
        assert int(best['state_dict']['validation_calls']) == 2
        subprocess.run(command+['slurm_id=resumed', 'trainer_params.max_steps=4', f'resume_checkpoint={first}'],
                       cwd=ROOT, env=env, check=True)
        last = next((directory/'logs').glob('**/resumed/checkpoints/final.ckpt'))
        state = torch.load(last, weights_only=False)
        assert state['global_step'] == 4 and len(state['rng_states']) == 2
        assert state['hyper_parameters']['freeze_decoder_time_mlp']
        for key, value in best['state_dict'].items():
            if key.startswith(('model.net.t_embedder.mlp.', 'ema_model.net.t_embedder.mlp.')):
                torch.testing.assert_close(value, state['state_dict'][key], rtol=0, atol=0)
        assert state['best_validation'] == best['best_validation']
        assert int(state['state_dict']['validation_calls']) == 3  # no extra step-zero eval on resume
        assert all(v['step'].item() == 4 for v in state['optimizer_states'][0]['state'].values())
        rates = [g['lr'] for g in state['optimizer_states'][0]['param_groups']]
        assert abs(rates[1]/rates[0] - .1) < 1e-12
        for row in csv.DictReader((last.parent.parent/'metrics.csv').open()):
            if row['train/loss']:
                assert abs(float(row['train/loss']) - float(row['train/image_loss']) - 3*float(row['train/token_loss'])) < 1e-5
                assert all(float(row[f'train/grad_norm_{name}']) > 0 for name in ['encoder','decoder','prior'])
        print('PASS: two-rank training/resume, step-zero validation, gFID selection/persistence, gradients and LR ratio', flush=True)


if __name__ == '__main__':
    main()

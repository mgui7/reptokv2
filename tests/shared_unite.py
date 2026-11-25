"""Contracts for the shared bottleneck, gradients, CFG and portable DDP resume."""
import csv
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldm.models.transformer.dit import DiT
from ldm.shared_unite import SharedUNITE, latent_corruption
from ldm.train_utils import save_checkpoint, seed_everything, train_batch


def main():
    torch.set_num_threads(1)
    seed_everything(17)
    with tempfile.TemporaryDirectory(prefix='shared-unite-test-') as temp:
        temp = Path(temp)
        init = temp / 'initializer.pt'
        net = dict(input_size=4, patch_size=2, in_channels=4, hidden_size=32,
                   depth=2, num_heads=4, num_classes=3, class_dropout_prob=.1,
                   learn_sigma=True, compile=False, use_checkpointing=True)
        # A nonzero tiny SiT initializer exercises the real strict loading path
        # and image-to-register attention rather than zero-initialized gates.
        donor = DiT(**net)
        with torch.no_grad():
            for block in donor.blocks:
                block.adaLN_modulation[-1].weight.normal_(std=.1)
                block.adaLN_modulation[-1].bias.normal_(std=.1)
            donor.final_layer.linear.weight.normal_(std=.1)
        torch.save(donor.state_dict(), init)
        params = dict(net_cfg={'target': 'ldm.models.transformer.shared_mmdit.SharedMMDiT',
                              'params': dict(net, n_aux_registers=4, load_from_ckpt=str(init))},
                      lr=.001, ema_rate=.9, decoder_steps=2, token_steps=2)
        module = SharedUNITE(**params)
        batch = {'image': torch.randn(3, 4, 4, 4), 'label': torch.tensor([0, 1, 2])}
        lengths = []
        handle = module.model.blocks[0].register_forward_pre_hook(lambda m, a: lengths.append(a[0].shape[1]))
        loss, metrics = module(batch)
        assert lengths == [9, 9, 5], lengths
        handle.remove()
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in metrics.values())
        loss.backward()
        assert all(p.grad is not None for p in module.model.parameters() if p.requires_grad)
        assert module.model.x_embedder.proj.weight.grad.abs().sum() > 0
        assert module.model.aux_registers.grad.abs().sum() > 0
        assert module.model.token_norm.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in module.ema_model.parameters())
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(module.model.parameters(), module.ema_model.parameters()))
        # Isolate reconstruction's gradient through the encoder bottleneck.
        module.zero_grad(set_to_none=True)
        z = module.model.encode(batch['image']); z.retain_grad()
        assert z.shape == (3, 1, 32)
        output = module.model(z, torch.ones(3) * .5, 'image', image=torch.randn_like(batch['image']))
        output.square().mean().backward()
        assert z.grad is not None and z.grad.abs().sum() > 0
        source = torch.randn(3, 1, 32, requires_grad=True)
        noisy, target = latent_corruption(source, torch.randn_like(source), torch.rand(3, 1, 1))
        assert not noisy.requires_grad and not target.requires_grad
        noise = torch.randn_like(source)
        t = torch.rand(3, 1, 1)
        reference = latent_corruption(source, noise, t)
        for routing in ('input', 'full'):
            noisy, target = latent_corruption(source, noise, t, routing)
            torch.testing.assert_close(noisy, reference[0], rtol=0, atol=0)
            torch.testing.assert_close(target, reference[1], rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(noisy.sum(), source)[0], t.expand_as(source))
            assert target.requires_grad == (routing == 'full')
        # Isolate token loss: it must reach the encoder's output only when enabled.
        module.eval()
        for routing in ('stop', 'input', 'full'):
            module.zero_grad(set_to_none=True)
            z = module.model.encode(batch['image']); z.retain_grad()
            noisy, target = latent_corruption(z, torch.randn_like(z), torch.ones(3, 1, 1) * .4, routing)
            prediction = module.model(noisy, torch.ones(3) * .4, 'token', labels=batch['label'])
            (prediction - target).square().mean().backward()
            if routing == 'stop':
                assert z.grad is None
            else:
                assert z.grad is not None and z.grad.abs().sum() > 0
                assert module.model.x_embedder.proj.weight.grad.abs().sum() > 0
        module.train()
        for mode, image in [('token', batch['image']), ('encode', None)]:
            try:
                module.model(z, torch.zeros(3), mode, image=image, labels=batch['label'])
                raise AssertionError('Accepted an invalid image side channel')
            except ValueError:
                pass
        # No auxiliary-state transfer between unrelated images/calls.
        module.eval()
        noise = torch.randn_like(z)
        one = module.model.encode(batch['image'], noise)
        module.model.encode(torch.randn_like(batch['image']))
        two = module.model.encode(batch['image'], noise)
        torch.testing.assert_close(one, two, rtol=0, atol=0)
        assert not torch.allclose(one, module.model.encode(batch['image'].roll(1, 0), noise))
        # CFG's conditional and null passes keep the same task/time/weights.
        module.token_steps = 1; module.cfg_scale = 2.
        labels = batch['label']
        start = module.noise((3, 1, 32), torch.device('cpu'), 23)
        cond = module.model(start, torch.zeros(3), 'token', labels=labels)
        uncond = module.model(start, torch.zeros(3), 'token', labels=torch.full_like(labels, 3))
        torch.testing.assert_close(module.sample_tokens(labels, 23, module.model), 2 * cond - uncond)
        module.token_steps = 2; module.cfg_scale = 1.
        # Actual train loop states, then remove the initializer entirely.
        module.train(); optimizer, scheduler = module.configure_optimizers()
        optimizer.zero_grad(set_to_none=True); module(batch)[0].backward(); optimizer.step()
        path = temp / 'saved.ckpt'
        save_checkpoint(path, module, optimizer, scheduler, 1, 1, {})
        init.unlink()
        legacy = torch.load(path, weights_only=False)
        legacy['hyper_parameters'].pop('token_grad_mode')
        torch.save(legacy, path)
        loaded = SharedUNITE.from_checkpoint(path)
        assert loaded.hyper_parameters['net_cfg']['params']['load_from_ckpt'] is None
        module.eval()
        expected = module.sample_images(module.sample_tokens(labels, 37), seed=41)
        actual = loaded.sample_images(loaded.sample_tokens(labels, 37), seed=41)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print('PASS: shared sequence lengths, encoder gradients, detached targets, no side channels, CFG, portable sampling', flush=True)

        cfg = OmegaConf.load(ROOT / 'configs/unite.yaml')
        cfg.defaults = [{'override hydra/hydra_logging': 'disabled'},
                        {'override hydra/job_logging': 'disabled'}]
        cfg.trainer_module.params = loaded.hyper_parameters
        cfg.trainer_module.params.token_grad_mode = None
        cfg.trainer_module.params.token_loss_weight = None
        cfg.trainer_module.params.first_stage_cfg = {'target': 'fixtures.TinyAutoencoder'}
        cfg.data = {'target': 'fixtures.TinyGenData', 'params': {'batch_size': 2, 'val_batch_size': 2}}
        cfg.log_root = str(temp / 'logs')
        cfg.trainer_params.precision = '32-true'
        cfg.trainer_params.max_steps = 2
        cfg.trainer_params.limit_val_batches = 0
        cfg.trainer_params.log_every_n_steps = 1
        cfg.checkpoint_params.every_n_train_steps = 0
        OmegaConf.save(cfg, temp / 'test.yaml')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'tests'), OMP_NUM_THREADS='1')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                   str(ROOT / 'train.py'), '--config-path', str(temp), '--config-name', 'test']
        subprocess.run(command + ['slurm_id=first'], cwd=ROOT, env=env, check=True)
        first = next((temp / 'logs').glob('**/first/checkpoints/final.ckpt'))
        for routing in ('stop', 'input', 'full'):
            subprocess.run(command + [f'slurm_id=resumed-{routing}', 'trainer_params.max_steps=4',
                                      f'resume_checkpoint={first}',
                                      f'trainer_module.params.token_grad_mode={routing}'],
                           cwd=ROOT, env=env, check=True)
            last = next((temp / 'logs').glob(f'**/resumed-{routing}/checkpoints/final.ckpt'))
            state = torch.load(last, weights_only=False)
            assert state['global_step'] == 4 and len(state['rng_states']) == 2
            assert state['hyper_parameters']['token_grad_mode'] == routing
            assert state['config']['trainer_module']['params']['token_grad_mode'] == routing
            assert all(v['step'].item() == 4 for v in state['optimizer_states'][0]['state'].values())
            assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
        # Resume with a new loss weight, then resume again with null defaults.
        subprocess.run(command + ['slurm_id=weighted', 'trainer_params.max_steps=4',
                                  f'resume_checkpoint={first}',
                                  'trainer_module.params.token_loss_weight=3'],
                       cwd=ROOT, env=env, check=True)
        weighted = next((temp / 'logs').glob('**/weighted/checkpoints/final.ckpt'))
        subprocess.run(command + ['slurm_id=weighted-resume', 'trainer_params.max_steps=6',
                                  f'resume_checkpoint={weighted}'], cwd=ROOT, env=env, check=True)
        for name, expected_step in [('weighted', 4), ('weighted-resume', 6)]:
            state = torch.load(next((temp / 'logs').glob(f'**/{name}/checkpoints/final.ckpt')),
                               weights_only=False)
            assert state['global_step'] == expected_step
            assert state['hyper_parameters']['token_loss_weight'] == 3.
            assert state['config']['trainer_module']['params']['token_loss_weight'] == 3.
            assert all(v['step'].item() == expected_step for v in state['optimizer_states'][0]['state'].values())
            for row in csv.DictReader(next((temp / 'logs').glob(f'**/{name}/metrics.csv')).open()):
                if row.get('train/loss'):
                    assert abs(float(row['train/loss']) - float(row['train/image_loss'])
                               - 3 * float(row['train/token_loss'])) < 1e-5
        # A bad batch must fail before AdamW can corrupt trainable weights.
        loaded.train()
        optimizer, scheduler = loaded.configure_optimizers()
        before = {n: p.detach().clone() for n, p in loaded.named_parameters() if p.requires_grad}
        bad_batch = dict(batch, image=torch.full_like(batch['image'], float('nan')))
        try:
            train_batch(loaded, bad_batch, optimizer, scheduler, 0, 1, 1., torch.device('cpu'))
            raise AssertionError('Nonfinite gradients did not stop training')
        except RuntimeError as error:
            assert 'non-finite' in str(error)
        for n, p in loaded.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        print('PASS: DDP resume, loss-weight override/preservation, weighted loss and nonfinite-gradient guard', flush=True)


if __name__ == '__main__':
    main()

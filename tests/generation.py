"""Generation smoke, frozen-codec checks, and relocation without original weights."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import build_reptok, model_params, TinyMetricTracker
from ldm.codec import FrozenCodec
from ldm.generation import TokenGenerator
from ldm.models.transformer.token_dit import OneDimensionalTransformer


class FakeGenerationMetrics(TinyMetricTracker):
    def aggregate(self):
        values = super().aggregate()
        return {'gfid': values['fid'], 'n_metric_samples': values['n_metric_samples']}


if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    import ldm.generation
    ldm.generation.GenerationMetrics = FakeGenerationMetrics
    sys.argv = [str(ROOT / 'train.py'), *sys.argv[2:]]
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    raise SystemExit

p = argparse.ArgumentParser()
p.add_argument('--ddp', action='store_true')
p.add_argument('--backbone', choices=['transformer', 'mixer'], default='transformer')
args = p.parse_args()
torch.set_num_threads(1)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONPATH=str(ROOT / 'tests'))
if args.ddp:
    env['CUDA_VISIBLE_DEVICES'] = ''

with tempfile.TemporaryDirectory(prefix='reptok-gen-') as tmp:
    tmp = Path(tmp)
    # Exercise actual DINO/TAESD constructors, not just fixture substitutes.
    # No Hub config, pretrained download, SiT file, or AE file may be reopened.
    from transformers import Dinov2WithRegistersConfig
    real_params = model_params()
    real_params.tokenizer_cfg.params.encoder = {
        'target': 'ldm.encoders.DINOImageEncoder', 'params': {
            'model_config': Dinov2WithRegistersConfig(
                hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                image_size=224, patch_size=14, num_register_tokens=2,
            ).to_dict(), 'trainable_layers': ['embeddings.cls_token'],
        }}
    real_params.first_stage_cfg = {
        'target': 'jutils.nn.tiny_autoencoder.TinyAutoencoderKL', 'params': {
            'encoder_path': None, 'decoder_path': None, 'latent_channels': 4,
        }}
    with (patch('transformers.AutoConfig.from_pretrained', side_effect=AssertionError('external config')),
          patch('transformers.AutoModel.from_pretrained', side_effect=AssertionError('external weights'))):
        real_recon = build_reptok(**real_params).eval()
        saved_params = real_recon.hyper_parameters
        saved_params['flow_cfg']['params']['net_cfg']['params']['load_from_ckpt'] = '/missing/SiT.pt'
        saved_params['first_stage_cfg']['params']['encoder_path'] = '/missing/encoder.pth'
        saved_params['first_stage_cfg']['params']['decoder_path'] = '/missing/decoder.pth'
        real_path = tmp / 'real-components.ckpt'
        real_state = real_recon.state_dict()
        # Compiled checkpoints include these wrappers around the same tensors.
        real_state = {k.replace('tokenizer.encoder.', 'tokenizer.encoder._orig_mod.'): v
                      for k, v in real_state.items()}
        torch.save(dict(state_dict=real_state, hyper_parameters=saved_params,
                        codec_config=real_recon.export_codec_config()), real_path)
        real_codec = FrozenCodec.from_recon_checkpoint(real_path)
        real_images = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(real_codec.encode(real_images),
                                   real_recon.tokenizer(real_images)['proj_token'], rtol=0, atol=0)
        assert real_codec.decode(real_codec.encode(real_images), num_steps=2).shape == real_images.shape
        real_path.unlink()

    # Verify reuse of DiT blocks leaves the original generator computation intact.
    old_source = subprocess.check_output(['git', 'show', '33b4277:ldm/models/transformer/1d_mmdit.py'], cwd=ROOT, text=True)
    (tmp / 'legacy_token.py').write_text(old_source)
    spec = importlib.util.spec_from_file_location('legacy_token', tmp / 'legacy_token.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    net_args = dict(num_channels=1, in_channels=32, hidden_size=32, depth=2,
                    num_heads=4, num_classes=3, class_dropout_prob=0.1, n_registers=1)
    torch.manual_seed(5)
    old_net = legacy.OneDimensionalTransformer(**net_args)
    torch.manual_seed(5)
    new_net = OneDimensionalTransformer(**net_args)
    assert old_net.state_dict().keys() == new_net.state_dict().keys()
    for key, value in old_net.state_dict().items():
        torch.testing.assert_close(value, new_net.state_dict()[key], rtol=0, atol=0)
    # Nonzero readout tests the actual Transformer path beyond zero initialization.
    with torch.no_grad():
        old_net.final_layer.linear.weight.normal_()
    new_net.load_state_dict(old_net.state_dict())
    x, t, labels = torch.randn(2, 1, 32), torch.rand(2), torch.tensor([0, 1])
    torch.manual_seed(7)
    old_out = old_net(x, t, y=labels)
    torch.manual_seed(7)
    new_out = new_net(x, t, y=labels)
    torch.testing.assert_close(old_out, new_out, rtol=0, atol=0)
    old_out.square().mean().backward()
    new_out.square().mean().backward()
    for a, b in zip(old_net.parameters(), new_net.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)

    recon = build_reptok(**model_params()).eval()
    recon_path = tmp / 'original-recon.ckpt'
    torch.save(dict(state_dict=recon.state_dict(), hyper_parameters=recon.hyper_parameters,
                    codec_config=recon.export_codec_config()), recon_path)
    codec = FrozenCodec.from_recon_checkpoint(recon_path)
    images = torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(codec.encode(images), recon.tokenizer(images)['proj_token'], rtol=0, atol=0)
    for key, value in codec.flow.state_dict().items():
        torch.testing.assert_close(value, recon.ema_model.state_dict()[key], rtol=0, atol=0)
    codec.train()
    assert not any(m.training for m in codec.modules())
    assert not any(p.requires_grad for p in codec.parameters())

    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='gen_mlp' if args.backbone == 'mixer' else 'gen')
    OmegaConf.set_struct(cfg, False)
    cfg.data = dict(target='fixtures.TinyGenData', params=dict(batch_size=2, val_batch_size=2))
    cfg.model.params.net_cfg.params = (dict(
        n_channels=1, channels=32, dim=48, depth=2, num_classes=3,
        class_dropout_prob=0.1, compile=False,
    ) if args.backbone == 'mixer' else net_args)
    cfg.recon_checkpoint = str(recon_path)
    cfg.trainer_module.params.sample_kwargs.num_steps = 2
    cfg.trainer_module.params.sample_kwargs.cfg_scale = 1.5
    cfg.trainer_module.params.decoder_steps = 2
    cfg.trainer_params.max_steps = 2
    cfg.trainer_params.val_check_interval = 2
    cfg.trainer_params.limit_val_batches = 2
    cfg.trainer_params.log_every_n_steps = 1
    cfg.trainer_params.precision = '32-true' if args.ddp else 'bf16-mixed'
    cfg.checkpoint_params.every_n_train_steps = 2
    cfg.log_root = str(tmp / 'logs')
    OmegaConf.save(cfg, tmp / 'smoke.yaml')
    command = [sys.executable]
    if args.ddp:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2']
    command += [str(Path(__file__).resolve()), '--worker', '--config-path', str(tmp),
                '--config-name', 'smoke', 'hydra.run.dir=' + str(tmp / 'hydra')]
    subprocess.run(command + ['slurm_id=first'], env=env, cwd=ROOT, check=True)
    first = next((tmp / 'logs').glob('**/first/checkpoints/final.ckpt'))
    state = torch.load(first, map_location='cpu', weights_only=False)
    assert 'recon_checkpoint' not in state['hyper_parameters']
    assert state['global_step'] == 2
    for key, value in codec.state_dict().items():
        torch.testing.assert_close(value, state['state_dict']['codec.' + key], rtol=0, atol=0)
    inference = TokenGenerator.from_checkpoint(first)
    before = inference.sample(labels, seed=42)

    moved = tmp / 'another-machine' / 'generation.ckpt'
    moved.parent.mkdir()
    shutil.copy2(first, moved)
    recon_path.unlink()
    shutil.rmtree(first.parents[1])
    # Resume still sees the now-invalid original path in the runtime YAML.
    subprocess.run(command + ['slurm_id=resumed', 'trainer_params.max_steps=4',
                             f'resume_checkpoint={moved}'], env=env, cwd=ROOT, check=True)
    with (patch('transformers.AutoConfig.from_pretrained', side_effect=AssertionError('external config')),
          patch('transformers.AutoModel.from_pretrained', side_effect=AssertionError('external weights'))):
        loaded = TokenGenerator.from_checkpoint(moved)
        torch.testing.assert_close(before, loaded.sample(labels, seed=42), rtol=0, atol=0)
    resumed_path = next((tmp / 'logs').glob('**/resumed/checkpoints/final.ckpt'))
    resumed = torch.load(resumed_path, map_location='cpu', weights_only=False)
    assert resumed['global_step'] == 4 and resumed['lr_schedulers'][0]['last_epoch'] == 4
    assert all(v['step'].item() == 4 for v in resumed['optimizer_states'][0]['state'].values())
    assert any(not torch.equal(value, resumed['state_dict'][key])
               for key, value in state['state_dict'].items() if key.startswith('model.'))
    for key, value in state['state_dict'].items():
        if key.startswith('codec.'):
            torch.testing.assert_close(value, resumed['state_dict'][key], rtol=0, atol=0)
    print('GEN PASSED: original token weights/forward exact, gradients within FP32 tolerance (atol=1e-6, rtol=1e-5); frozen codec, CFG, training, validation, '
          'resume and identical inference after removing recon and moving gen checkpoint;',
          args.backbone, 'two-process CPU DDP' if args.ddp else 'BF16 mixed precision')

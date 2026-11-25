"""Offline checks against the actual upstream feature API and portable codecs."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from diffusers import AutoencoderKL
from einops import rearrange
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params, build_reptok
from ldm.self_flow_encoder import FrozenSelfFlowEncoder, IMAGENET_CONFIG
from ldm.codec import FrozenCodec, recon_resume_params
from ldm.generation import TokenGenerator

torch.set_num_threads(2)
torch.manual_seed(7)
config = dict(IMAGENET_CONFIG, input_size=4, hidden_size=32, depth=3, num_heads=4)
vae_config = dict(in_channels=3, out_channels=3,
                  down_block_types=['DownEncoderBlock2D'],
                  up_block_types=['UpDecoderBlock2D'], block_out_channels=[16],
                  layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=4)
encoder = FrozenSelfFlowEncoder(vae_config=vae_config, model_config=config,
                                layer=2, encode_batch_size=2)
# Upstream zero-initializes adaLN gates. Nonzero gates make the layer-index test
# distinguish actual block 2 from the embedding or another block.
with torch.no_grad():
    for block in encoder.model.blocks:
        block.adaLN_modulation[-1].weight.normal_(std=0.03)
images = torch.randn(3, 3, 4, 4).clamp(-1, 1)
latents = encoder.encode_latents(images)
patches = rearrange(latents, 'b c (h p) (w q) -> b (h w) (c p q)', p=2, q=2)
with torch.no_grad():
    _, reference = encoder.model(patches, torch.zeros(3), torch.full((3,), 1000),
                                 return_raw_features=2)
actual = encoder.patch_features(latents)
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
pooled = encoder(images)['cls_token']
torch.testing.assert_close(pooled, reference.mean(1, keepdim=True), rtol=1e-5, atol=1e-6)
assert pooled.shape == (3, 1, 32) and not pooled.requires_grad
encoder.train()
assert all(not child.training for child in encoder.modules())
assert all(not p.requires_grad for p in encoder.parameters())
torch.testing.assert_close(encoder(images)['cls_token'], pooled, rtol=0, atol=0)
assert not torch.equal(pooled[0], pooled[1])

with tempfile.TemporaryDirectory(prefix='reptok-self-flow-') as tmp:
    tmp = Path(tmp)
    torch.save(encoder.model.state_dict(), tmp / 'self-flow.pt')
    encoder.vae.save_pretrained(tmp / 'vae')
    params = model_params()
    params.dino_loss_weight = 0
    params.tokenizer_cfg.params.encoder = {'target': 'ldm.self_flow_encoder.FrozenSelfFlowEncoder',
        'params': dict(load_from_ckpt=str(tmp / 'self-flow.pt'), vae_pretrained=str(tmp / 'vae'),
                       model_config=config, layer=2, encode_batch_size=2)}
    recon = build_reptok(**params)
    assert recon.frozen_dino is None
    # Baseline decoder output starts at zero; make conditioning gradients visible.
    with torch.no_grad():
        for p in recon.model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)
    optimizer, scheduler = recon.configure_optimizers()
    before = {k: v.clone() for k, v in recon.tokenizer.encoder.state_dict().items()}
    projection = recon.tokenizer.cls_proj.fc.weight.detach().clone()
    scheduler.step()  # baseline advances warmup before the optimizer update
    loss, _ = recon({'image': images, 'latent': latents})
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in recon.tokenizer.encoder.parameters())
    assert recon.tokenizer.cls_proj.fc.weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(projection, recon.tokenizer.cls_proj.fc.weight)
    assert all(torch.equal(v, recon.tokenizer.encoder.state_dict()[k]) for k, v in before.items())
    recon.eval()
    tokens = recon.tokenizer(images)['proj_token'].detach()
    saved = dict(state_dict=recon.state_dict(), hyper_parameters=recon.hyper_parameters,
                 codec_config=recon.export_codec_config())
    path = tmp / 'recon.ckpt'
    torch.save(saved, path)
    (tmp / 'self-flow.pt').unlink()
    import shutil
    shutil.rmtree(tmp / 'vae')
    with patch.object(AutoencoderKL, 'from_pretrained', side_effect=AssertionError('external VAE')):
        codec = FrozenCodec.from_recon_checkpoint(path)
        torch.testing.assert_close(codec.encode(images), tokens, rtol=0, atol=0)
        restored = build_reptok(**recon_resume_params(saved))
        restored.load_state_dict(saved['state_dict'], strict=True)
        torch.testing.assert_close(restored.tokenizer(images)['proj_token'], tokens, rtol=0, atol=0)
        gen_flow = {'target': 'ldm.generation.TokenFlow', 'params': {
            'schedule': 'linear', 'sigma_min': 0.0, 'net_cfg': {
                'target': 'ldm.models.transformer.token_dit.OneDimensionalTransformer',
                'params': dict(num_channels=1, in_channels=32, hidden_size=32,
                               depth=2, num_heads=4, num_classes=1000, n_registers=1)}}}
        gen = TokenGenerator(flow_cfg=gen_flow, recon_checkpoint=str(path))
        gen.train()
        assert all(not p.requires_grad for p in gen.codec.parameters())
        assert not gen.codec.tokenizer.encoder.training
        torch.testing.assert_close(gen.codec.encode(images), tokens, rtol=0, atol=0)
        gen_loss, _ = gen({'image': images, 'label': torch.tensor([1, 2, 3])})
        gen_loss.backward()
        assert all(p.grad is None for p in gen.codec.parameters())
        gen_path = tmp / 'gen.ckpt'
        torch.save(dict(state_dict=gen.state_dict(), hyper_parameters=gen.hyper_parameters), gen_path)
        path.unlink()
        restored_gen = TokenGenerator.from_checkpoint(gen_path)
        torch.testing.assert_close(restored_gen.codec.encode(images), tokens, rtol=0, atol=0)
print('SELF-FLOW PASSED: raw feature parity, deterministic pooling, frozen encoder, trainable projection, offline recon/gen portability')

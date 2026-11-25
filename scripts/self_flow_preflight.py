"""Check released weights and raw layer-16 features on real validation images."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from jutils import instantiate_from_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldm.self_flow_encoder import FrozenSelfFlowEncoder

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')
encoder = FrozenSelfFlowEncoder(
    load_from_ckpt=str(a.base / 'assets/self-flow/selfflow_imagenet256.pt'),
    vae_pretrained=str(a.base / 'assets/sd-vae-ft-ema'),
).cuda()
assert len(encoder.model.blocks) == 28
with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
    cfg = compose(config_name='self_flow', overrides=[
        'data.params.tar_base=/e/project1/jureap52/schb/data/imagenet256_latents_wds',
        'data.params.val_batch_size=2', 'data.params.multinode=false'])
data = instantiate_from_config(cfg.data)
images = next(iter(data.val_dataloader()))['image'].cuda()
started = time.monotonic()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    latents = encoder.encode_latents(images)
    patches = rearrange(latents, 'b c (h p) (w q) -> b (h w) (c p q)', p=2, q=2)
    _, reference = encoder.model(patches, torch.zeros(len(images), device='cuda'),
                                 torch.full((len(images),), 1000, device='cuda'),
                                 return_raw_features=16)
    actual = encoder.patch_features(latents)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    pooled = encoder(images)['cls_token']
    torch.testing.assert_close(pooled, reference.mean(1, keepdim=True), rtol=0, atol=0)
    encoder.train()
    repeated = encoder(images)['cls_token']
    torch.testing.assert_close(repeated, pooled, rtol=0, atol=0)
assert all(not p.requires_grad for p in encoder.parameters())
assert not encoder.training and pooled.shape == (2, 1, 1152)
assert torch.isfinite(pooled).all() and not torch.equal(pooled[0], pooled[1])
result = dict(passed=True, depth=28, layer=16, patch_shape=list(actual.shape),
              token_shape=list(pooled.shape), upstream_max_abs_error=(actual-reference).abs().max().item(),
              deterministic=True, frozen=True, internal_clean_timestep=1,
              token_std=pooled.float().std().item(), seconds=time.monotonic()-started,
              peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
(a.base/'checks/preflight.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)

# UL post-training from trained RepTok

The experiment asks whether joint UL fine-tuning improves generation from an
already trained RepTok E/D/G triplet. Primary selection is lower gFID. The user
approved four GPUs for 12 hours and prior/generator weight **3**, with E trainable
from the first update. There is no frozen-encoder warmup or automatic continuation.

## Initialization and objective

Use the canonical base RepTok final generator checkpoint (100k generator updates,
300k reconstruction training) from `baselines/cos01-20260922`, job 1967449. Its
`codec.flow`, `codec.tokenizer`, and `model` contain the actual evaluated D, E and G.
Seed both the trainable decoder and its EMA from that codec decoder. Carry all
trained generator weights. Start fresh optimizer/scheduler states at update zero.
`reptok_initialization` produces a portable weights-only checkpoint; embedded DINO
configuration and weights remove dependencies on external initializer files.
Image latent shape is reconstructed from the decoder (4,32,32), not the source
generator checkpoint's token latent shape (1,1152).

Keep the existing UL adaptation: image reconstruction flow matching plus
`3 * (UL diffusion ELBO term + terminal KL)`. Both prior input and target gradients
reach E; no stop-gradient, cosine, FD, MMD or added variance constraint. This
coefficient is 30 times the original UL experiment's 0.1. It numerically matches
shared-UNITE lambda 3, but the objectives and resulting gradient scales differ.
The weighted prior term and pre-clipping E/D/G gradient norms are logged every 50
updates. This is the existing RepTok image-FM decoder adaptation, not a full
reproduction of the UL paper's decoder.

## Training and evaluation

- E: DINOv2-B, train CLS and 768-to-1152 projection (888,960 parameters), LR 1e-6.
- D: independent MMDiT-XL/2, LR 1e-5, EMA 0.9995.
- G: independent 1D Transformer XL, LR 1e-5, no generator EMA.
- AdamW, zero weight decay, 2,000-update LR warmup, gradient clipping 1, BF16.
- Global batch 128, same cached ImageNet center-crop/latent data as earlier UL.
- UL log-SNR range [-10,5], decoder token noise sigma approximately 0.08181.
- Evaluate at step zero and every 1,000 updates: 1,024 images, CFG 1, 50 token
  integration steps and 50 decoder steps, same existing seeds/metrics as UL.
- Select `best.ckpt` by **gFID**; record rFID/PSNR/token diagnostics separately.
  If all updates degrade gFID, the best checkpoint can remain at step zero.
- Periodic checkpoints every 10k and final state at graceful stop after 41,400s,
  within one 12h / four-GPU allocation. No automatic resubmission.

Base final gFID was 74.52217 under its clean-token sampling protocol. Step-zero UL
validation measures the effect of switching to UL's finite-noise endpoint before
any update. Report that value separately from the historical baseline; learning
improvement should first be assessed against the step-zero UL value. The old base
stage-2 training also used batch 512/raw augmentation, unlike this joint stream.
A 1,024-image, single-seed gFID and the best of frequent evaluations are exploratory;
any claimed improvement needs a later larger common-protocol evaluation.

## Validation

`tests/ul_posttrain.py` checks exact component mapping, image latent shape,
reconstruction/prior gradients into E, objective coefficient, optimizer coverage
and LR ratio, two-process training/resume, initial evaluation and gFID-based best
checkpoint persistence. Selection test scores are synthetic, never reported as
real FID. The run's `checks/production.json` records strict full-size trained-weight
loading and a synthetic BF16 forward/backward check without an optimizer update.

## Fixed decoder time MLP ablation (2026-10-05)

`ul_posttrain_frozen_time` starts from the same canonical trained RepTok final
checkpoint with prior weight3. Only the decoder's `net.t_embedder.mlp` is fixed
(1,624,320 parameters). Its EMA copy retains the same initialization. E's CLS and
projection, all other previously trainable decoder parameters, and all generator
parameters including its own time MLP remain trainable. The saved hyperparameter
`freeze_decoder_time_mlp` restores this policy on checkpoint loading/resume and
frozen parameters are excluded from AdamW. The default remains false for old runs.

This is a controlled test of the time-MLP instability identified by the read-only
20k/30k checkpoint swaps, not an established long-term fix. Stop at35k updates to
cross the earlier23–26k failure window; keep the four-GPU/12h allocation ceiling and
41,400s graceful stop. LR, warmup, global gradient clipping, UL noise, EMA, dataset,
step-zero/every1k evaluation and gFID selection remain identical. Do not restart
from a degraded late checkpoint. There is no automatic continuation.

Validation adds actual small-model optimizer updates: frozen decoder time weights
and EMA must remain bitwise unchanged, while E/rest-D/G change. Serialization,
optimizer restore and two-rank DDP resume must preserve freezing. The run's final
audit also compares full-size decoder and EMA time-MLP tensors against the initial
checkpoint. Whether this prevents later instability and improves gFID is the
experiment's outcome, not assumed by these plumbing checks.

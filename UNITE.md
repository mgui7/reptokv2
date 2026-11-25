# Shared image/token model: first experiment

This is an adaptation of UNITE's shared tokenizer/latent denoiser to RepTok's
single-token bottleneck, with the image flow decoder sharing the same backbone.
It is not a reproduction of the UNITE paper and is not trained entirely from scratch.

## Architecture

- One SiT-XL/2 backbone: 28 layers, width 1152, 16 heads, initialized from the
  same SiT checkpoint as the RepTok reconstruction baseline. All backbone weights train.
- One continuous image token, shape `1 x 1152`. No token input/output projection.
  A shared learnable LayerNorm produces the encoding and denoising outputs.
- Four learned auxiliary register embeddings reset to their initial embeddings
  on every forward. Their updated states are discarded. Only the image token
  travels from encoding to reconstruction; no patch or auxiliary state bypass.
- Frozen TAESD and the existing spatial-latent dataset. No DINO/MAE/CLIP teacher,
  alignment targets, or cosine regularizer.

Three calls per update:

1. **Encode:** clean spatial image latents, one Gaussian-noise latent register,
   four auxiliary registers -> normalized clean image token. The latent-register
   time is 0 (noise endpoint); the encode task specifies that image patches are clean.
2. **Image flow:** noisy spatial image latents + image token with small Gaussian
   perturbation + four reset auxiliary registers -> image velocity. This loss
   backpropagates through the token into the encoding pass.
3. **Token denoise:** noisy detached image token + four reset auxiliary registers,
   with image patches completely absent -> normalized clean-token prediction.
   Both the noising source and target are detached, but denoising updates the
   shared backbone. Loss is unweighted clean-token MSE, not velocity MSE.

Conditioning is `time_embedding + class_embedding + task_embedding` through the
existing AdaLN blocks. Encode/image modes always use null class ID 1000. Token
mode uses the ImageNet class, dropping it to null with probability 0.1. CFG compares
the same token task with real versus null class; task IDs are a separate embedding
and do not consume or replace class IDs.

## First-run choices

`configs/unite.yaml`: joint loss = image flow MSE + 1.0 * token clean-target MSE.
Reconstruction token noise has sigma 0.1 after normalization. This is a provisional
few-token setting, not the paper's noise recipe or an experimentally selected optimum.
No KL, adversarial or perceptual training losses are added.

Global batch 128 on four GPUs, AdamW lr 1e-4, warmup 2000, clip 1, BF16; shared-model
EMA 0.9995. Activation checkpointing applies to the same blocks in all modes. EMA
has its own separately constructed network, including its own compiled forwards.
EMA is updated before the optimizer, matching this project's existing convention.
Validation uses EMA for both encoding and decoding to keep their latent interface
consistent. This differs from baseline's current tokenizer + EMA image decoder.

The first trial is capped at 10k updates; this is a functional/stability experiment,
not a convergence run. Before it, a full-size real-data pilot checks training,
validation, saving and portable resume. Image/token samplers use 50 Euler steps
and CFG 1 in the trial, with both rFID and gFID at 1024 samples every 1000 steps.
FID at this sample count is for matched local comparisons, not paper scores.

Token diagnostics report per-rank batch dispersion, RMS and between-image cosine,
plus same-image token MSE under two different encoding-noise seeds during validation.
High dispersion alone does not exclude tokens dominated by the encoder's random input.
Image fidelity and generated samples must be assessed alongside these diagnostics.

At inference, generate the single token from noise/class first, then generate image
latents from fresh independent image noise conditioned only on that token. Token
sampling converts clean predictions to velocity as `(prediction - z_t) / (1-t)`;
it never evaluates the singular endpoint t=1. Auxiliary states reset each step.

## Code and loading

- `ldm/models/transformer/shared_mmdit.py`: three modes through the SiT blocks.
- `ldm/shared_unite.py`: joint losses, EMA, sampling, diagnostics, paired recon/gen validation.
- `configs/unite.yaml`: explicit experiment settings.
- `train.py`: opt-in extra CSV fields and architecture-only checkpoint construction.
- `tests/shared_unite.py`: bottleneck/gradient/CFG/portability contracts and native DDP resume.

Fresh training uses SiT/TAESD initializers. Full checkpoints embed their weights and
initializer-free architecture; resumed training and `SharedUNITE.from_checkpoint`
do not reopen those paths. Logs, caches, weights and run snapshots live on scratch.

Sources: https://arxiv.org/abs/2603.22283 and the existing RepTok baseline code.

## Token-gradient ablations

`trainer_module.params.token_grad_mode` controls the two paths through the encoded
latent: `stop` detaches both input and target (the original experiment); `input`
backpropagates only through the noising input; `full` backpropagates through both.
The forward values, losses, model architecture and frozen TAESD are identical.
Only gradient routing changes. The null default preserves the saved mode on
resume, or selects `stop` for fresh training/older checkpoints. An explicit mode
override on resume is applied to the saved architecture before model construction;
optimizer, scheduler, EMA and RNG are still restored from the checkpoint.

The official UNITE implementation detaches `z_for_flow` once, before using it
for both the noisy input and the loss target (`models/unite.py`,
`forward_denoising` and `_compute_flow_loss`). Removing that detach corresponds
to `full`; the public code does not expose a separate no-stop-gradient ablation
switch. `input` is an additional intermediate variant, not a claimed paper ablation.

`trainer_module.params.token_loss_weight` also accepts an explicit resume override.
Its null default preserves the checkpoint value (1.0 for fresh training). For a
threefold generation objective, set `token_grad_mode=stop` and
`token_loss_weight=3`; this changes the task weight, not the timestep weighting.
Gradient clipping now fails on nonfinite gradients before the optimizer update,
so a failed run cannot continue spending its budget on NaN updates.

# RepTok baseline with a jointly trained UL prior

## Scope and reference

User-approved scope (2026-09-28): three independent networks; reuse the **actual
current RepTok 1D Transformer**, not MLP-Mixer; retain baseline DINO encoder and
trainable scope for this controlled comparison; four GPUs for **12 hours maximum**,
with no continuation and no stage-2 training.

Reference: `/e/scratch/lift-virg/gui2/reptok/baselines/cos01-20260922`.
Pinned baseline source: `0d446f5a42a05d7fea11caab2823ad88af5e15a5`.
Completed recon job 1965038 (300k updates), generation job 1967449 (100k).
This experiment branches from native trainer revision 4996b04 (includes immediate
nonfinite-gradient rejection). The baseline source/configurations are unchanged.

This is an **UL-prior adaptation**, not a complete reproduction of Unified Latents:
we retain the original reconstruction FM loss, pretrained DINO and low-level VAE.
Source: https://arxiv.org/html/2602.17270, especially Eq. 3 and Algorithm 1.
No claim of a new method or demonstrated quality improvement is made.

## Architecture and initialization

| Component | Setting |
| --- | --- |
| Encoder | Original DINOv2-with-registers-base, 12 layers, width 768; pretrained frozen backbone, trainable CLS embedding and 768-to-1152 linear+LayerNorm projection |
| Bottleneck | One projected continuous 1x1152 token; no raw patches/register states passed to decoder |
| Reconstructor | Original SiT-initialized MMDiT-XL/2, width1152, depth28, heads16, patch2; 674,834,720 trainable parameters |
| Prior | Original OneDimensionalTransformer, width1152, depth28, heads16, MLP ratio4, 1 auxiliary register, split_channels1; 677,437,056 parameters |
| Encoder trainable parameters | 888,960 (CLS + projection); no unfreezing of DINO blocks |
| Low-level codec | Original frozen TAESD and cached spatial image latents, shape4x32x32 |
| Initialization | Fresh original DINO and SiT initializers; prior random initialization seed2023; no completed recon/gen/UNITE weights reused |
| Labels | Real class labels only enter prior; 10% unconditional dropout; encoder sees images, reconstructor uses null class1000 |
| Weight sharing | None between encoder, reconstructor and prior |

Prior is inexpensive per example because its sequence is short, not because its
parameter count is small. Two large trainable networks are present in joint training.

## Objective, precision and gradients

Let x denote the image, y the frozen spatial image latent, and z=E(x) the clean
1x1152 token. Decoder condition:

    z0 = alpha0*z + sigma0*eta
    alpha0 = sqrt(sigmoid(5)) = 0.99664796
    sigma0 = sqrt(sigmoid(-5)) = 0.08180985

Encoder is deterministic (apart from standard model behavior); no learned posterior
variance, entropy term, cosine teacher, LPIPS or adversarial loss is added.

    total = original_image_FM_MSE + 0.1 * prior_KL_upper_bound

The coefficient .1 is provisional and is NOT numerically equivalent to the old
cosine weight .1. The image FM uses uniform linear-flow time and its original mean
velocity MSE. The only change to its inputs is fixed-noise token conditioning.

Prior: sample lambda uniformly on [-10,5], alpha=sqrt(sigmoid(lambda)),
sigma=sqrt(sigmoid(-lambda)); u=alpha/(alpha+sigma)=sigmoid(lambda/2).
Feed the original prior a normalized VP corruption:

    input = u*z + (1-u)*epsilon
    target_velocity = z - epsilon
    prior_diffusion = mean_batch[ (15/2)*u^2 * sum_token_dims((v-target)^2) / 4096 ]

This equals Eq.3's clean-target ELBO coefficient `(15/2)*exp(lambda)` after the
linear-velocity conversion. It is NOT uniform clean-token/velocity MSE.
Add the finite high-noise endpoint KL:

    terminal = mean_batch[sum_dims(0.5*(alpha_min^2*z^2 + sigma_min^2
                                      - 1 - log(sigma_min^2))) / 4096]

Here "min" denotes lambda=-10, the high-noise end. Both prior terms retain gradients
through z. Decoder FM and prior KL use the SAME image-dimensional normalization,
4096=4*32*32. This avoids implicitly multiplying relative token pressure by4096/1152.
Prior loss is conditioned on the class (or randomly dropped class), so its KL
interpretation is conditional on the supplied conditioning, not an unconditional bitrate.
No bits/pixel claim is made for this weighted spatial-latent adaptation.

Gradient routes:
- Reconstruction -> decoder + encoder's trainable CLS/projection.
- Prior -> prior + encoder's trainable CLS/projection, through input AND target.
- No reconstruction gradient into prior; no prior gradient into decoder.
- Frozen DINO backbone still propagates gradients to the trainable CLS.

## Optimization and evaluation

Four GPUs, global batch128 (32/rank), AdamW lr1e-4, weight_decay0, betas(.9,.999),
2000-update warmup then constant LR, BF16 mixed precision, clip norm1 with immediate
nonfinite-gradient failure. Same seed2023. Update each network once per joint batch.
No weight schedule or sweep. EMA .9995 for decoder ONLY, retaining baseline's
pre-optimizer EMA timing; current tokenizer and current prior are used in evaluation,
matching the baseline's lack of tokenizer/prior EMA.

Use the baseline recon cached ImageNet-256 dataset, center crops, no random flips.
Evaluate every5000 updates with16 batches x16 images x4 ranks =1024 images.
Same FID/PSNR/SSIM implementation and real-reference images. Decoder Euler50 steps,
prior Euler50 steps, CFG1. Prior integrates from u(lambda=-10) to u(lambda=5), then
rescales to the VP endpoint z0; do NOT denoise to clean z or add noise a second time.
Reconstruction uses the same alpha0/sigma0 conditioning distribution as training.
Generation decoder noise uses seed0 on each evaluation batch, matching the baseline
FrozenCodec path; reconstruction uses batch-index seeds. Prior uses batch-index seeds,
matching baseline stage2. Encoding noise uses batch-index+10000, a new UL-only source.

Log image loss, unweighted prior loss, terminal KL, token RMS, mean coordinate std
and cyclic-neighbour cosine. Validation additionally compares decoder velocity MSE
at fixed flow time .5 with correct versus shuffled tokens (same noisy image).
This is a cheap token-dependence diagnostic, not a full shuffled-reconstruction metric.
Moments/cosines are rank-local batch statistics averaged over batches/ranks.

12h Slurm limit. Native loop requests graceful stop after41400 seconds (~11.5h),
reserving30 minutes for validation/checkpointing and teardown. No auto-resubmission.
Save every10k steps and on graceful stop. Full checkpoints embed all models,
optimizer, scheduler, rank RNG and initializer-free architecture/configuration.

## Comparability and remaining differences

Baseline final: rFID@1024=66.0, PSNR=11.606884; gFID@1024=74.522171.
Architectures, pretrained initializers, encoder trainability, reconstruction dataset,
optimizer defaults, validation reference/count and nominal sampling steps are matched.

Changes intentionally made: joint vs staged training; cosine replaced by learned-prior
constraint; fixed token noise and finite prior endpoint; prior time/loss weighting.
The original stage2 used global batch512 and a raw-image random-crop/flip stream;
joint training uses the reconstruction batch128 and aligned cached image stream.
The first joint prior is also the evaluated generator; no frozen-token second stage.
12h is NOT the same budget as baseline300k recon+100k gen. Equal parameter shape
and FID sample count do not establish a compute-matched causal comparison.

## Validation

`tests/ul.py`: loss and encoder gradients match direct clean-prediction ELBO;
analytic terminal KL; finite-noise sampler endpoint; independent modules; isolated
prior/recon gradients to CLS/projection; frozen parameters; portable sampling after
initializer deletion; two-process native DDP training and optimizer/RNG resume.
Config comparison checks the completed baseline run files, not only repository defaults.
These checks do not substitute for GPU execution or establish model quality.

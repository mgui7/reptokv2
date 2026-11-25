# Joint one-step token generation with population FD

Authorized 2026-09-29: three independent networks, jointly trained for one 12-hour
allocation on four GPUs, from the same original initialization as the UL trial.
The user explicitly rejected creating a new branch. Work stays on the existing
`codex/reptok-ul` branch. Existing UL-to-400k uses its immutable old snapshot.

## Objective and sampling

- Encoder: same pretrained DINOv2-B with registers; train CLS and 768→1152
  projection only. This is fresh experiment initialization, not all-random weights.
- Decoder: same SiT-initialized MMDiT-XL/2, same spatial reconstruction flow loss,
  frozen TAESD, null image class, EMA .9995.
- Generator: same randomly initialized 1D Transformer XL, one 1152-D token,
  depth28/width1152/heads16/one auxiliary register. No checkpoint warm start.
- `z0 = sqrt(sigmoid(5))*E(x) + sqrt(sigmoid(-5))*eta`, as in UL.
- Independent pure Gaussian epsilon enters exactly ONE generator network call:
  `G(epsilon,c) = epsilon + prior(epsilon,u=0,c)`. Existing velocity convention is
  z-epsilon, hence the plus sign. This residual map learns the noisy z0 distribution
  directly: no ODE integration, no noisy-data input, no additional output noise.
- Loss = original image FM MSE + .1 * normalized population FD. UL denoising and
  terminal KL are fully removed. No extra cosine, semantic, variance or class loss.
- FD uses raw z0 vectors, jointly learns both sides (no encoder stop-gradient),
  includes all 1152x1152 covariance entries, and matches the unconditional mixture.
  Class input and 10% class dropout are preserved, but mixture FD does NOT enforce
  class-conditional correctness; it also cannot detect higher-moment differences.
- No learned representation head is introduced. FD is a Gaussian-moment distance,
  not a proof that the full token distributions match or that image quality improves.

## Population estimator and numerics

Adaptation of https://arxiv.org/html/2604.28190v1 section3.2. That paper uses fixed
real-image feature statistics and post-trains pretrained generators; this trial
instead trains encoder/decoder/generator jointly from the original initializers.

Two EMA estimators (encoder and generator), decay .99, first actual batch as
initial state. This decay differs from the paper default .999: with our global
batch128 it has a roughly100-update /12800-sample e-folding history. It is a
provisional staleness/variance choice, not a tuned optimum. Maintain first and
second raw moments, covariance=M-mu mu^T; never EMA batch covariances directly.
Current batch retains autograd on both sides, previous moments detach. No gradient
rescaling to cancel 1-decay. Therefore the effective gradient scale changes after
bootstrap. Report raw FD, not only the normalized loss, which stays near1.

Gather features with autograd across all GPUs, then update identical moments on
all ranks. Generator/encoding noises use rank-distinct seeds derived from saved
population update count; baseline identical rank seeds would otherwise duplicate
pure-noise samples. These FD-specific draws do not consume the baseline global RNG.

Compute moments and covariance matrix distance in FP64 with fixed diagonal jitter
1e-5. Custom analytic Bures first derivatives use SPD transport maps, avoiding
unstable eigenvector backward for repeated eigenvalues; finite-difference tests
cover both covariance inputs. FD/dim is normalized as
`fd / (stopgrad(fd) + 1e-6)`, then weighted .1. The jitter, division by dimension and
epsilon are documented numerical conventions; this .1 is NOT gradient-equivalent
to the original UL prior weight .1. Small negative round-off FD is clamped at0.
Nonfinite gradients stop before optimizer updates. No loss-specific collapse stop.

## Training and evaluation

AdamW1e-4, wd0, warmup2000, original scheduler, globalbatch128, same cached
center-cropped ImageNet256 stream, BF16 network computation, clip1, seed2023.
All trainability, initializers and architecture configurations match UL. First
stage is frozen; independent encoder, decoder and generator receive the expected
loss gradients. Reconstruction loss has no generator path; FD has no decoder path.

Evaluate every5000 updates on1024 samples, same paired reconstruction and generation
metrics/reference as UL. Decoder50 Euler steps, generator1 NFE, CFG1. Current
encoder/generator plus EMA decoder. Save generated/reconstructed panels and
correct-vs-shuffled decoder-token MSE. Log raw FD, population variances, clean
encoder and generated-token batch spread/RMS/cosine. Image quality and token use
remain necessary even if FD falls. Comparisons are same runtime/architecture/data,
not identical compute or identical generator sampling cost.

One 12h Slurm allocation, graceful stop41400seconds, periodic checkpoint10k,
no automatic continuation. Checkpoints include both EMA moment sets and update
count as well as native optimizer/scheduler/model/EMA/RNG. Dataset iterator cursor
is not saved, matching the existing trainer. Native resume parity tests use a fixed
stream to isolate model/optimizer/statistic/RNG continuation.

## Verification

`tests/fd.py`: two-sided FD finite differences; repeated and singular covariance
stability; sample-permutation invariance; detached EMA history; global DDP gradient
and statistics equivalence to a concatenated batch; one pure-noise generator call;
encoder CLS/projection gradients from reconstruction and FD; decoder/generator
routing isolation; frozen first stage; portable checkpoint without initializers;
DDP training/resume including moments and exact fixed-stream continuation parity.

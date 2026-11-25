# Joint one-step RepTok: FD replaced by exact MMD

Requested 2026-09-30: replace only the distribution-matching objective, reuse the
RDM paper implementation where possible, and do not create another branch.
This implementation lives in the existing `codex/reptok-ul` worktree. It is an
exact-MMD adaptation, not the paper's final Nyström/frozen-reference iRDM recipe.

## Source and necessary adaptation

Official source: https://github.com/vita-epfl/RDM
Pinned commit: `391e2749d8fa078ce3644be2ff9e5afe83ce7a61`.
The MIT-licensed `kernels.py`, `median_heuristic.py`, and `mmd_exact.py` are vendored
unchanged in `ldm/third_party/rdm`, with their license and SHA256 provenance.

The upstream exact loss detaches the real reference and accepts a constant real
self-kernel term. Our encoder is trainable. `ldm/mmd_objective.py` therefore composes
the original kernel primitives as `k_gg + k_ee - 2*k_ge`, with gradients on BOTH
current sides, including `k_ee`. The official wrapper remains unchanged and is
used as a value/generator-gradient reference in tests.

- Raw noisy encoder tokens `z0 = alpha*E(x) + sigma_noise*eta` versus one-call
  generated tokens `G(epsilon,c) = epsilon + prior(epsilon,u=0,c)`.
- Single Gaussian/RBF kernel on raw features; no feature normalization.
- Biased MMD squared (V-statistic), diagonal included, same upstream convention.
- One fixed median bandwidth, calibrated once on the first global encoder batch
  (128 samples under the existing four-GPU configuration), using the upstream
  strict-upper-triangle distance median. No per-step bandwidth updates or
  multiscale kernel. Calibration preserves training RNG and is checkpointed.
  The paper calibrates a frozen reference pool; our first-batch calibration is
  an adaptation and estimates bandwidth from fewer reference samples.
- Exact global fresh-batch all-pairs kernels, gathered with autograd. No history
  queue and no EMA moments: exact nonlinear MMD cannot reuse FD moment buffers.
  Row chunking/checkpointing changes memory use, not the estimator or batch size.
- Kernel arithmetic remains FP64, as in the FD statistic path, with network BF16.
  This differs from the upstream wrapper's FP32, but reuses its kernel primitives
  verbatim and avoids cancellation/TF32 errors near small MMD values.
- Preserve the FD trial's `raw / (stopgrad(raw) + 1e-6)` loss normalization and
  weight 0.1. Clamp only negative round-off to zero. The upstream aggregate default
  epsilon is 1e-7; 1e-6 here intentionally preserves our existing FD convention.

## Controlled settings and limits

`configs/mmd.yaml` retains the FD configuration for E/D/G architectures,
trainability, original initializers, seed, token noise, data, AdamW/LR/warmup,
global batch128, decoder EMA, clipping, evaluation, and stopping budget.
No FD term remains in training; no variance/scale anchor, cosine, additional
semantic/class loss, multiple G updates, or change in encoder learning rate.

The config retains the prior 41400-second graceful stop inside a 12-hour/four-GPU
allocation, but adding the config does not submit a job. Fresh initialization
is the default; `resume_checkpoint` must be an MMD checkpoint, not an FD one.

Estimator and objective both change: FD used an EMA of first/second moments with
0.99 decay; this MMD arm uses the current global batch. It has more sampling noise
and no factor0.01 attenuation of current-batch gradients. Equal loss weights are
not equivalent gradient strengths. The fixed initial bandwidth can become too
broad if learned tokens contract. Joint collapse and class mismatch remain
possible; global-mixture MMD does not ensure class-conditional correctness.

Report raw MMD and saved bandwidth, current encoder/generator batch variances,
existing token std/RMS/cosine and image-quality/token-use diagnostics. The
normalized training scalar stays near1 and is not itself a convergence metric.

## Verification and use

`tests/mmd.py` checks independent kernel sums, two-sided finite differences,
upstream value/G-gradient parity, restored encoder-gradient parity, equal-moment
but different-distribution sensitivity, fixed bandwidth/RNG preservation,
global DDP gradient parity, single-call sampling, reconstruction/matching gradient
routing, frozen weights, portable checkpoints and exact fixed-stream DDP resume.
The native dataset iterator cursor remains uncheckpointed, as in the FD baseline.

Run with the existing trainer using `--config-name mmd`. As with FD, separate
microbatch loss accumulation is not an exact larger-batch MMD because it omits
cross-microbatch kernel pairs.

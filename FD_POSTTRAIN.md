# Clean-token FD post-training with decoder corruption

Start from canonical trained RepTok E/D/G (baseline final generator 1967449,
300k reconstruction + 100k generation), reset optimizer/scheduler and FD moments.
The generator's trained multi-step weights initialize the one-call residual
`epsilon + prior(epsilon, time=0, class)`. This changes the sampling protocol:
record step-zero one-call gFID, and do not equate it with baseline 50-step gFID.

FD compares **clean projected encoder tokens** with generated tokens. Both E and
G receive FD gradients. Reconstruction trains the existing E/D with independent
per-image corruption: 80% receive `.8*z + .6*s_ref*epsilon`, and 20% receive exactly
`z`. Noise is never included in the FD reference or added to generator outputs.
Validation reconstruction and generation both use clean decoder inputs. No new
normalization layer, auxiliary head, model, or semantic reference is introduced.

`s_ref = sqrt(mean_j Var_images(z_j))` is measured once from the initial trained
encoder and stored as an immutable hyperparameter. For the 2026-10-12 run,
49,681 unique cached FP32 baseline validation encoder outputs are projected through
the exact checkpoint's linear/LayerNorm head, then accumulated in FP64. Cache
checkpoint identity and encoder-source hashes are verified. s_ref=1.3787409924304508,
so noise std=.8272445954582705. This is a fixed scale calibration on validation
images, not an outcome-based hyperparameter selection. Do not use the old FD
checkpoint's contracted scale or adapt the scale to subsequent encoder shrinkage.

Keep original FD weight .1, self-detached loss normalization (epsilon 1e-6), decay
.99 and covariance jitter 1e-5. This experiment does not isolate normalization from
the FD objective, and does not promise that moment matching fixes generation.
Post-training LR E1e-6, D/G1e-5, warmup2000, AdamW wd0, global clip1, batch128,
BF16 and decoder EMA.9995. E trainability remains CLS+projection. The decoder time
MLP is fixed, following the separately diagnosed instability; the rest of D and
all G parameters, including its time MLP, train normally. This is an additional
stability setting, not evidence that noise alone causes any observed improvement.

One four-GPU allocation, 35k updates or 41400s trainer ceiling, whichever comes
first; scheduler wall time12h. No automatic continuation/resubmission. Evaluate
step0 and every1k on1024 images, CFG1, generator1NFE/decoder50NFE; save best gFID,
periodic10k and final. Also log fresh global token FD/mean gap, centered RMS std,
scale-to-reference ratio, corrupted-path SNR and fixed-time corrupted-token
velocity MSE. Fresh FD at1024 is sample-biased; it is not directly interchangeable
with the old8192-sample frozen diagnostic. Corrupted MSE is a diagnostic, not a
full image reconstruction score under corruption.

`tests/fd_posttrain.py` checks exact pretrained mapping, corruption formula and
clean mask, clean FD inputs, both E gradient paths, frozen time/EMA after real
optimizer updates, optimizer coverage, self-contained checkpoints, actual small
validation paths, two-rank resume with exact uninterrupted parity, and synthetic
gFID checkpoint-selection logic. Full-size CPU structural/exact-weight checks
run before submission. The real-data32-image BF16 gradient preflight runs inside
the same allocation without an optimizer update; training starts only if it passes.

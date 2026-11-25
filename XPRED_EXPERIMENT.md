# Sequential RepTok x-pred and clean-FD x-pred

Confirmed 2026-10-20: train RepTok G from scratch, with canonical pretrained E/D frozen.
Then initialize joint FD E/D/G from the **terminal**, not best-gFID, checkpoint of
that new x-pred stage. Each stage gets four GPUs, at most 100,000 optimizer steps
or 41,400 trainer seconds, within one 12-hour allocation. No continuation or retry.
The second job depends on successful completion and an audit of the first.

## Prediction and loss

`ldm.generation.TokenFlow` defaults to legacy `prediction_type="v"`. With `"x"`,
the raw Transformer output is a clean token prediction. For `z_t=t*z+(1-t)*eps`,
compute both `v_pred=(x_pred-z_t)/max(1-t,.05)` and
`v_target=(z-z_t)/max(1-t,.05)` in FP32 and take their MSE. This matches the
x-output/v-loss and denominator floor in
https://github.com/LTH14/JiT/blob/main/denoiser.py (consulted 2026-10-20).
The original uniform time sampling, unit Gaussian input noise, Transformer,
class dropout and Euler solver remain. This is JiT's loss, not a reproduction of
all JiT training/sampling settings. Sampling also uses the .05 denominator floor,
including the final steps of the 50-step Euler schedule.

`FDRepTok.generate_tokens` detects the saved x-pred configuration and uses
`prior.predict_clean(eps,0,y)` directly, with no epsilon addition. FD supervises
generated and encoded clean-token distributions and does not use a velocity MSE.
Legacy v checkpoints keep their original `eps+prior(eps,0,y)` output.
Both generation methods are checkpointed through `prior_cfg`/`flow_cfg`.

## Training and evaluation

RepTok: fresh whole G, seed2023, AdamW1e-4, no G EMA, frozen canonical codec.
FD: carry the terminal trained x-pred G plus unchanged E/D into joint training;
reset optimizer, scheduler and FD moments. Retain cleanFD weight .1, decay .99,
jitter1e-5, detached-self normalization epsilon1e-6. E LR1e-6, D/G1e-5,
CLS+projection encoder trainability, fixed decoder timeMLP, D EMA .9995.
Reconstruction corruption remains 80%(.8*z+.6*1.3787409924304508*eps),20%clean;
inference and FD real targets use clean tokens. s_ref comes from the same
canonical unchanged encoder as the previous cleanFD run.

Both: cached ImageNet256 center crops, batch128 across4GPUs, BF16, warmup2000,
weight decay0, global gradient clip1. Step0 and every1k evaluate1024 images,
CFG1, D50; RepTok G50 and FD G1. Save best-gFID, periodic10k and terminal state.
A trainer-time stop writes `interrupted.ckpt`; when it exits successfully and
passes the result audit, this is the terminal state used by the FD stage.

Report FD step-zero separately: switching G50 to direct G1 changes the sampler
although G weights are identical. Historical v results are contextual references,
not matched new controls: evaluation frequency/RNG consumption and FD's trained-G
initialization/budget differ. Single1024-image scores do not prove a causal gain.

## Reproducibility and checks

`scripts/xpred_experiment.py` prepares portable initialization checkpoints,
asserts exact codec/transition weights, checks optimizer coverage and independent
D/EMA storage, runs real-data BF16 gradient checks inside each allocation,
and audits terminal/frozen weights, finite metrics and checkpoint selection.
`tests/xpred.py` covers the clipped loss including t=1, mixed precision, CFG,
Euler, legacy v parity, actual optimization with a frozen codec, portable
checkpoint reload, exact trained-x-to-FD transfer and joint gradients.
`tests/fd_posttrain.py` retains native two-rank resume and validation regressions.
A source snapshot and hashes pin the complete run, including existing uncommitted
FD files. Initial checkpoints contain no optimizer state from old experiments.

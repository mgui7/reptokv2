# RepTok reconstruction and generation

## Repository branches

- `main`: native RepTok reconstruction and generation, including the ordinary MLP-Mixer generator.
- `recon`: original native reconstruction and stage-2 Transformer baseline snapshot.
- `self-flow`: frozen Self-Flow layer-16 encoder experiment.
- `unite`: shared image/token Transformer experiment.
- `ul`: joint UL, FD/MMD, post-training, and x-pred experiments.

This repository starts from source snapshots of the original worktrees. Checkpoints,
logs, generated outputs, and historical notebooks with embedded outputs are excluded.
Original source revisions are recorded in the import commit messages.

The retained tests run without importing the original Git history or Lightning.

A minimal PyTorch implementation of the existing `cls-init-cos` baseline. Stage 1 retains DINO CLS token adaptation, cosine regularization, and the MMDiT flow decoder. Stage 2 adds the original class-conditioned 1D token Transformer over frozen projected tokens. Both use one native training loop; no Lightning runtime, video models, notebooks, or alternative regularizers.

Reference: `jm/cont-tok` commit `33b4277437db35929c5468feecff76b2cf224ac2`, experiment `imnet256/tokenizer/cls-init-cos`. The native loop and container launch structure follow `qdino-v2`; numerical behavior follows the original RepTok implementation.

## Training

Use the existing `/e/home/jusers/gui2/jupiter/jenv-video.sif` container. All runtime dependencies were import-checked there; nothing needs to be installed into that image for this branch.

On an allocated GPU node, from this directory:

```bash
bash scripts/run.sh trainer_params.max_steps=100000
```

The baseline defaults are unchanged: cosine weight 0.1, LR 1e-4, warmup 2,000, EMA 0.9995, BF16, batch size 32 per GPU, and 50 Euler reconstruction steps. `max_steps=-1` runs until interrupted, so set an explicit budget for a bounded run. Override the container with `REPTOK_CONTAINER` if its location changes.

Required assets:

- ImageNet shards containing matching `jpeg`, `cls`, and `latent.npy`, including the validation shards. Set `data.params.tar_base=/your/latent/dataset` to the actual root. The historical paths are retained in the config. The legacy `/p` copy is incomplete; a complete Jupiter copy was verified at `/e/project1/jureap52/schb/data/imagenet256_latents_wds` (285 training shards and 34 `val-shards`). Raw-image shards are not a drop-in substitute for the cached SD latents.
- `checkpoints/SiT-XL-2-256x256.pt` for the original warm-start decoder.
- `checkpoints/taesd_encoder.pth` and `checkpoints/taesd_decoder.pth`.
- The `facebook/dinov2-with-registers-base` model and the Inception weights used by torchmetrics, locally cached or downloadable.

Jupiter initialization weights are available under `/e/scratch/jureap52/fischer13/checkpoints`; the local `checkpoints` symlink can point there. Dataset roots must be explicitly overridden for the complete Jupiter copies. No completed ImageNet baseline results are bundled with this code.

For Jupiter, submit from this directory:

```bash
mkdir -p logs/slurm
sbatch scripts/jupiter.sh trainer_params.max_steps=100000 data.params.tar_base=/your/latent/dataset
```

The script defaults to one node, four GPUs, account `lift-virg`, and 30 minutes. Override allocation/time through `sbatch`. It starts one torchrun launcher per node. The job is independent of the local connection once submitted. USR1/TERM requests a checkpoint at the next complete optimizer update.

## Stage-1 iMF decoder

`imf` adds the **boundary-velocity variant** of improved MeanFlow
([paper, Eq. 12 and Algorithm 1](https://arxiv.org/abs/2512.02012)) to the
existing MMDiT-XL/2 decoder. It retains the SiT warm start, frozen TAESD,
DINO backbone with trainable CLS embedding and projection, frozen DINO teacher,
cosine weight 0.1, optimizer and EMA behavior. The default FM config and stage-2
generator are unchanged.

```bash
bash scripts/run.sh --config-name imf trainer_params.max_steps=100000 data.params.tar_base=/your/latent/dataset
```

Use the same arguments with `sbatch scripts/jupiter.sh` to submit on Jupiter.
This is a new training run; an existing FM checkpoint is not an iMF checkpoint.
The SiT initialization file loads before the new interval embedding is added;
that embedding's output starts at zero. Resume an iMF run with `--config-name imf
resume_checkpoint=/path/to/imf.ckpt` and the original data/architecture overrides.

The model predicts average velocity `u(z,t,s,c)` between current time `t` and
destination `s`, using this repository's convention **0=noise, 1=data** (the
paper uses the opposite direction). Training uses `v=u(z,t,t,c)` as the JVP
tangent, holds `s` and conditioning `c` fixed, and minimizes
`mean((u - (s-t)*stopgrad(du/dt) - (data-noise))**2)`.
The total derivative is `du/dt = partial_t u + partial_z u * v`.
Half the samples have zero-length intervals in expectation, providing ordinary
FM supervision for the boundary velocity. Times use logit-normal sampling with
mean +0.4 and std 1.0, equivalent to the reference's -0.4 after time reversal.

This controlled adaptation uses the paper's no-auxiliary-head formulation and
plain MSE, preserving the baseline decoder/cosine loss scale. It does **not**
reproduce the complete iMF benchmark recipe (adaptive loss weighting, auxiliary
velocity head, learned flexible CFG, and replacement of adaLN by multi-token
conditioning). No pretrained iMF class-conditional weights or quality claims are
transferred to this DINO-conditioned reconstruction task.

Reconstruction defaults to **one decoder evaluation**:
`latent = noise + u(noise, 0, 1, projected_token)`, then one TAESD decode.
Inference needs no JVP or teacher evaluations. Fixed tokens and noise give
deterministic outputs; changing noise permits different reconstructions.
For a few-step comparison set `trainer_module.params.sample_kwargs.num_steps=2`
(or `4`); each interval costs one decoder evaluation. More steps are not
guaranteed to improve quality. Validation `val/loss` is the boundary FM diagnostic
`u(z,t,t)`, not the interval training objective; rFID/SSIM still evaluate the
configured reconstruction step count.

The decoder runs eagerly with explicit attention for forward-mode autodiff;
its `compile` option must remain false. JVP and its model-predicted tangent are
computed in FP32 without a backward graph. A separate forward uses the training
autocast setting and preserves gradients into the CLS/projection. This costs more
training compute than ordinary FM. Tokenizer compilation retains the baseline
setting. Optional `+model.params.net_cfg.params.use_checkpointing=true` checkpoints
the gradient-enabled decoder pass only.

Offline checks in the existing container (synthetic data, not ImageNet quality):

```bash
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/imf.py
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py --imf
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py --imf --ddp
```

These passed on CPU, including BF16, two-process DDP and save/resume. The checks
use a reduced real MMDiT and synthetic encoder/AE fixtures. GPU execution,
XL memory/throughput and ImageNet reconstruction quality have not been measured.

### Baseline-matched Jupiter run

`imf_jupiter` matches the completed **300,000-update reconstruction** baseline
(job 1965038); the baseline's 100,000-update budget applies to stage-2 generation.
It uses the same SiT/TAESD files, ImageNet latent shards, seed 2023, per-GPU batch
32, global batch 128 on four GH200s, LR 1e-4, 2,000-update warmup, EMA 0.9995,
cosine weight 0.1, and 1,024 validation images every 5,000 updates.
The necessary interval embedding adds 1,624,320 parameters while retaining the
28-layer, width-1152, 16-head, patch-2 backbone. Its initialization preserves the
RNG state so the tokenizer/projection initialization matches the same-seed baseline.
The objective/interval sampling, eager JVP implementation and 1-step validation
are the intentional differences.

`scripts/imf_jupiter.sh` runs a frozen source copy from an explicit run directory,
with the existing offline model caches. Submit it as that directory's `job.sh`,
passing the directory as its first argument and an absolute Slurm output path.
It runs CUDA sanity checks before training. Each allocation is limited to 12 hours,
with a graceful checkpoint after 41,400 seconds; successful finite segments
automatically submit dependent continuations until 300,000 total updates.
Failures are not retried automatically. The continuation cap is eight allocations,
matching the baseline's cap; reaching the cap preserves the latest checkpoint.
`recon.json` records progress and `jobs.tsv` records continuation job IDs.

## Stage-2 generation

The default command still runs reconstruction. To train the minimal generator after recon:

```bash
bash scripts/run.sh --config-name gen recon_checkpoint=/current/path/recon.ckpt trainer_params.max_steps=100000
```

Generation uses raw ImageNet images, encodes frozen **projected 1 x 1152 tokens** online, and trains the 1D XL Transformer with class dropout 0.1. The old `1d-xl` defaults are retained: LR 1e-4, warmup 2,000, EMA disabled, batch 128 per GPU, validation every 10,000 batches, checkpoint every 20,000 updates. The data root is separate from the recon latent-shard root; override `data.params.tar_base` as needed. Cached spatial `latent.npy` files are not token targets.

Validation reports **gFID**, sample count, and token flow loss; PSNR/SSIM between different generated images are not reported. Default validation uses 1,280 images per GPU (5,120 with four GPUs), not FID@50k. `trainer_module.params.sample_kwargs` controls token sampling, including `cfg_scale`; `decoder_steps` separately controls image decoding.

The frozen codec contains the recon tokenizer, selected EMA decoder (or decoder if no EMA exists), and image AE. It loads without `torch.compile` wrappers, normalizing their state-key prefixes; the recon training compile setting is unchanged. It stays in evaluation mode. Reconstruction is no longer instantiated through a nested Lightning checkpoint loader. Decoder sampling uses the current autocast context, without the old wrapper's forced inner FP16 context. This is a minimal stage-2 implementation, not a claim of full numerical equivalence to the old generation training loop.

### Ordinary RepTok MLP generator

`gen_mlp` replaces the token Transformer with the ordinary MLP-Mixer: **28 layers,
width 1280, 189,486,852 trainable generator parameters** for the local projected
`1 x 1152` tokens and 1,000 classes. It reproduces the historical ordinary
`mlp-mixer` configuration (token expansion 4, feature expansion 2), with no attention.
The existing tokenizer, projection, reconstruction decoder and image AE are loaded
from `recon_checkpoint` and frozen. The optimizer, flow target, sampling defaults
and evaluation protocol are inherited from `gen`.

```bash
bash scripts/run.sh --config-name gen_mlp recon_checkpoint=/current/path/recon.ckpt trainer_params.max_steps=100000
```

Use the same local reconstruction checkpoint and dataset overrides as the baseline
for a generator-only comparison. The existing local baseline codec is at
`/e/scratch/lift-virg/gui2/reptok/baselines/cos01-20260922/logs/recon/2026-09-23/1965038/checkpoints/final.ckpt`.
`gen_mlp_jupiter` selects that local codec and the verified Jupiter dataset paths,
with 100,000 updates and the completed baseline's 16 validation batches per GPU
(1,024 validation images on four GPUs). Use `--config-name gen_mlp_jupiter` when
launching through `scripts/jupiter.sh`. Logs and checkpoints follow `log_root`;
the local `logs` symlink can point to the experiment directory on scratch.

### Checkpoints and portability

**Old full Lightning gen checkpoints already included the nested recon weights.** Their path dependency came from reopening the original recon file during construction, before loading those saved tensors. Freezing weights does not exclude them from a state dictionary. This was established from the old source; no historical gen checkpoint was accessible for direct inspection.

New gen checkpoints contain the trainable generator, optional generator EMA, and frozen codec, together with the architecture needed to construct them. Resume and inference do not open the original recon, SiT, or TAESD paths:

```bash
bash scripts/run.sh --config-name gen resume_checkpoint=/new/location/gen.ckpt trainer_params.max_steps=200000
```

Standalone sampling needs only this gen checkpoint and the software environment:

```python
import torch
from ldm.generation import TokenGenerator

model = TokenGenerator.from_checkpoint('/new/location/gen.ckpt', device='cuda')
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    images = model.sample(torch.tensor([0, 1, 2, 3]), seed=42)
```

New recon saves embed the DINO architecture config for offline codec construction. Starting gen from an older recon save that lacks it may need the Hugging Face **config** cached/downloadable once; it does not reload pretrained DINO weights or the old SiT/TAESD files. After saving gen, its model loading is self-contained. Inception evaluation weights and training datasets are separate runtime assets. The loader targets this branch's gen checkpoints; old Lightning gen files with other architectures are not automatically converted.

UNITE/shared image-token denoising remains a design discussion; it is not implemented in this branch.

## Outputs and resume

Runs write `config.yaml`, `hparams.yaml`, TensorBoard events, `metrics.csv`, and checkpoints under `logs/<name>/<date>/<job-or-time>/`. Periodic checkpoints use `stepXXXXXX.ckpt` and `last.ckpt`; the last state of a bounded run is `final.ckpt`. Interrupted runs write `interrupted.ckpt`.

```bash
bash scripts/run.sh resume_checkpoint=/path/to/final.ckpt trainer_params.max_steps=200000
```

Checkpoint state includes model, EMA, AdamW, scheduler, batch/update counters, and per-rank RNG. Resume requires the same world size. WebDataset worker/shuffle positions are not serialized: resume restores optimization state, not a bitwise continuation of the input stream. Recon training can load legacy recon state dictionaries, but native checkpoints do not contain Lightning's internal loop metadata.

Reconstruction validation retains the old protocol and cadence. Its sample count is `val_batch_size × world_size × limit_val_batches`, and the actual count is logged as `val/n_metric_samples`. The default is **256 per GPU**, not FID@50k. Keep sample count, reference data, decoder NFE and precision fixed when comparing runs.

## Reconstruction behavior preserved

- Same DINO preprocessing and trainable CLS embedding, projection, MMDiT, SiT warm start, linear flow/noise target, cosine teacher, and loss weighting.
- Same AdamW defaults, clipping, per-microbatch warmup update, and gradient accumulation. The scheduler advances **before** the optimizer, as in the old code.
- Decoder EMA updates **before backward/optimizer.step**, on every microbatch. No tokenizer EMA is introduced.
- Same validation noise seeds, current tokenizer + EMA decoder, metric implementation, image quantization, and recursive train/eval modes of frozen children.
- Same seed on every rank, including Lightning DDP's post-construction seed reset. Multi-node DDP retains the old FP16 gradient-compression hook.
- The original eager AE call is retained even when a cached latent is present, preserving RNG consumption.

Scope is the selected infinite/repeating WebDataset configuration. Other architectures, epoch-based training, alternate precisions, online tracking and profiling were intentionally not ported. Checkpoint metadata now preserves the trainable encoder config instead of allowing teacher construction to mutate it. Signal checkpoints are saved at a complete update boundary.

## Reconstruction verification

During the original migration, the following passed in `jenv-video.sif` with tiny offline encoders/AE and the real reduced-size MMDiT, tokenizer projection and flow implementation:

- Exact equality (`rtol=0`, `atol=0`) against Lightning 2.5.0.post0 for initialization, losses, gradients, every parameter/EMA tensor, AdamW state, warmup state, reconstruction images and validation outputs.
- FP32 on CPU and BF16 on GH200, with accumulation 1 and 2; GPU checks also cover a continuous RNG stream across training/validation.
- Full entry-point training, validation, save and resume in the unmodified container; two-process CPU DDP smoke test.
- Resolved config equality and source comparisons against the original baseline.

These establish controlled short-run equivalence, not bitwise identity across different PyTorch versions, hardware, shuffled datasets or a full-scale compiled ImageNet run. Smoke-test metrics are synthetic proxies and are not FID results. Multi-node GPU execution has not been exercised.

Run the native checks without Lightning:

```bash
apptainer exec --nv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py
apptainer exec /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py --ddp
```

The one-off source/Lightning parity scripts have been removed after migration;
the retained smoke tests exercise the current training, validation and resume paths.

Stage-2 checks (synthetic metrics, no ImageNet training):

```bash
apptainer exec --cleanenv --nv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py --ddp
```

These verify frozen codec weights,
exercise actual small DINO + TAESD constructors with external loading blocked, and
remove the original recon checkpoint before restoring gen from another directory.

MLP-specific checks (the integration checks use a reduced model and synthetic data):

```bash
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/mlp_mixer.py
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py --backbone mixer
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py --backbone mixer --ddp
```

`tests/mlp_mixer.py --legacy-source /path/to/old/mlp-mixer.py` additionally checks
exact historical initialization, forward outputs and gradients on a small model.

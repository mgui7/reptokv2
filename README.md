# RepTok reconstruction and generation

## Repository branches

- `main` and `recon`: native RepTok reconstruction and stage-2 generation baseline.
- `self-flow`: frozen Self-Flow layer-16 encoder experiment.
- `unite`: shared image/token Transformer experiment.
- `ul`: joint UL, FD/MMD, post-training, and x-pred experiments.

This repository starts from source snapshots of the original worktrees. Checkpoints,
logs, generated outputs, and historical notebooks with embedded outputs are excluded.
Original source revisions are recorded in the import commit messages.

The legacy comparison checks (`tests/source_contract.py`, `tests/compare_legacy.py`,
and the legacy parity section of `tests/generation.py`) additionally need original
commit `33b4277437db35929c5468feecff76b2cf224ac2`. To run them, import that history
locally from an existing original checkout first:

```bash
git fetch /path/to/original/ziptok 33b4277437db35929c5468feecff76b2cf224ac2
```

The imported legacy history is only for local verification; do not push it to this
source-only repository. Training and sampling do not require that history.

A minimal PyTorch implementation of the existing `cls-init-cos` baseline. Stage 1 retains DINO CLS token adaptation, cosine regularization, and the MMDiT flow decoder. Stage 2 adds the original class-conditioned 1D token Transformer over frozen projected tokens. Both use one native training loop; no Lightning runtime, video models, or notebooks. The additional shared image/token experiment is described in [UNITE.md](UNITE.md).

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

## Stage-2 generation

The default command still runs reconstruction. To train the minimal generator after recon:

```bash
bash scripts/run.sh --config-name gen recon_checkpoint=/current/path/recon.ckpt trainer_params.max_steps=100000
```

Generation uses raw ImageNet images, encodes frozen **projected 1 x 1152 tokens** online, and trains the 1D XL Transformer with class dropout 0.1. The old `1d-xl` defaults are retained: LR 1e-4, warmup 2,000, EMA disabled, batch 128 per GPU, validation every 10,000 batches, checkpoint every 20,000 updates. The data root is separate from the recon latent-shard root; override `data.params.tar_base` as needed. Cached spatial `latent.npy` files are not token targets.

Validation reports **gFID**, sample count, and token flow loss; PSNR/SSIM between different generated images are not reported. Default validation uses 1,280 images per GPU (5,120 with four GPUs), not FID@50k. `trainer_module.params.sample_kwargs` controls token sampling, including `cfg_scale`; `decoder_steps` separately controls image decoding.

The frozen codec contains the recon tokenizer, selected EMA decoder (or decoder if no EMA exists), and image AE. It loads without `torch.compile` wrappers, normalizing their state-key prefixes; the recon training compile setting is unchanged. It stays in evaluation mode. Reconstruction is no longer instantiated through a nested Lightning checkpoint loader. Decoder sampling uses the current autocast context, without the old wrapper's forced inner FP16 context. This is a minimal stage-2 implementation, not a claim of full numerical equivalence to the old generation training loop.

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

This branch additionally implements an experimental shared image/token model; see [UNITE.md](UNITE.md). The default RepTok configurations and model computations remain the baseline.

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

In `jenv-video.sif`, the following passed with tiny offline encoders/AE and the real reduced-size MMDiT, tokenizer projection and flow implementation:

- Exact equality (`rtol=0`, `atol=0`) against Lightning 2.5.0.post0 for initialization, losses, gradients, every parameter/EMA tensor, AdamW state, warmup state, reconstruction images and validation outputs.
- FP32 on CPU and BF16 on GH200, with accumulation 1 and 2; GPU checks also cover a continuous RNG stream across training/validation.
- Full entry-point training, validation, save and resume in the unmodified container; two-process CPU DDP smoke test.
- Resolved config equality and source comparisons against the original baseline.

These establish controlled short-run equivalence, not bitwise identity across different PyTorch versions, hardware, shuffled datasets or a full-scale compiled ImageNet run. Smoke-test metrics are synthetic proxies and are not FID results. Multi-node GPU execution has not been exercised.

Run the native checks without Lightning:

```bash
apptainer exec --nv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/source_contract.py
apptainer exec --nv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py
apptainer exec /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/smoke.py --ddp
```

For the legacy comparison only, install Lightning into a temporary directory outside this project:

```bash
apptainer exec /e/home/jusers/gui2/jupiter/jenv-video.sif python -m pip install --no-deps \
  --target /tmp/reptok-parity-deps lightning==2.5.0.post0 pytorch-lightning==2.5.0.post0
apptainer exec --nv --env PYTHONPATH=/tmp/reptok-parity-deps \
  /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/compare_legacy.py \
  --device gpu --precision bf16-mixed --continuous-rng
```

The comparison extracts the old implementation from Git into a temporary directory; legacy code and Lightning are not shipped as training dependencies.

Stage-2 checks (synthetic metrics, no ImageNet training):

```bash
apptainer exec --cleanenv --nv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py
apptainer exec --cleanenv /e/home/jusers/gui2/jupiter/jenv-video.sif python tests/generation.py --ddp
```

These compare the token Transformer with the old source, verify frozen codec weights,
exercise actual small DINO + TAESD constructors with external loading blocked, and
remove the original recon checkpoint before restoring gen from another directory.

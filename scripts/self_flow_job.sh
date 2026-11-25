#!/usr/bin/env bash
#SBATCH --partition=booster
#SBATCH --account=lift-virg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=reptok-sf-pilot
#SBATCH --output=/e/scratch/lift-virg/gui2/reptok/self-flow/layer16-20261016/slurm/%x-%j.out
set -euo pipefail
BASE=/e/scratch/lift-virg/gui2/reptok/self-flow/layer16-20261016
REPTOK_CONTAINER=/e/home/jusers/gui2/jupiter/jenv-video.sif
export REPTOK_CONTAINER
export HF_HOME=/e/scratch/lift-virg/gui2/reptok/cache/huggingface
export TORCH_HOME=/e/scratch/lift-virg/gui2/reptok/cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=/e/scratch/lift-virg/gui2/reptok/cache/inductor
cd "$BASE/source"
export SLURM_SUBMIT_DIR="$PWD"
stage=${1:?stage required}
self_flow=(--config-name self_flow
    "self_flow_checkpoint=$BASE/assets/self-flow/selfflow_imagenet256.pt"
    "self_flow_vae=$BASE/assets/sd-vae-ft-ema")
attempt=${2:-1}
common=("log_root=$BASE/logs" 'trainer_params.limit_val_batches=16' '+trainer_params.max_time_seconds=41400')
recon_data=('data.params.tar_base=/e/project1/jureap52/schb/data/imagenet256_latents_wds')
gen_data=(
    'data.params.tar_base=/e/scratch/jureap52/datasets/imagenet_shards_raw_shuffled'
    'data.params.train.shards="imagenet_train_{000000..000146}.tar"'
    'data.params.train.image_key=jpg' 'data.params.train.rename.image=jpg'
    'data.params.validation.shards="/e/project1/jureap52/schb/data/imagenet256_latents_wds/val-shards/{000000..000033}.tar"'
)
record() {
    apptainer exec "$REPTOK_CONTAINER" python "$BASE/source/scripts/self_flow_state.py" "$1" "$SLURM_JOB_ID" "$2"
}
submit() {
    local next_stage=$1 next_attempt=$2 id
    id=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --job-name="reptok-sf-$next_stage" \
         "$BASE/source/scripts/self_flow_job.sh" "$next_stage" "$next_attempt")
    printf '%s\t%s\t%s\t%s\n' "$next_stage" "$next_attempt" "$id" "$SLURM_JOB_ID" >> "$BASE/jobs.tsv"
    echo "Queued $next_stage attempt $next_attempt: $id"
}
if [[ $stage == pilot ]]; then
    srun --nodes=1 --ntasks=1 --gpus-per-task=1 \
        apptainer exec --nv --bind /e:/e "$REPTOK_CONTAINER" \
        python scripts/self_flow_preflight.py --base "$BASE"
    common=("log_root=$BASE/logs" 'trainer_params.limit_val_batches=1' 'trainer_params.log_every_n_steps=1' \
            'checkpoint_params.every_n_train_steps=0' '+trainer_params.max_time_seconds=3600')
    bash scripts/jupiter.sh "${self_flow[@]}" "${common[@]}" "${recon_data[@]}" name=pilot-recon \
        trainer_params.max_steps=8 trainer_params.val_check_interval=8 trainer_module.params.sample_kwargs.num_steps=2
    read -r step recon_path < <(record pilot-recon 8)
    test -n "$recon_path"
    bash scripts/jupiter.sh "${self_flow[@]}" "${common[@]}" "${recon_data[@]}" name=pilot-recon-resume \
        "resume_checkpoint=$recon_path" trainer_params.max_steps=10 trainer_params.val_check_interval=10
    read -r step recon_path < <(record pilot-recon-resume 10)
    test -n "$recon_path"
    bash scripts/jupiter.sh --config-name gen "${common[@]}" "${gen_data[@]}" name=pilot-gen \
        "recon_checkpoint=$recon_path" trainer_params.max_steps=2 trainer_params.val_check_interval=2 \
        trainer_module.params.decoder_steps=2 trainer_module.params.sample_kwargs.num_steps=2 \
        +trainer_params.error_if_nonfinite=true
    read -r step gen_path < <(record pilot-gen 2)
    test -n "$gen_path"
    bash scripts/jupiter.sh --config-name gen "${common[@]}" "${gen_data[@]}" name=pilot-gen-resume \
        "resume_checkpoint=$gen_path" trainer_params.max_steps=4 trainer_params.val_check_interval=4 \
        +trainer_params.error_if_nonfinite=true
    record pilot-gen-resume 4
    submit recon 1
    exit 0
fi
resume=()
if [[ -f $BASE/$stage.json ]]; then
    checkpoint=$(python3 - "$BASE/$stage.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['checkpoint'])
PY
    )
    resume=("resume_checkpoint=$checkpoint")
fi
if [[ $stage == recon ]]; then
    target=300000
    max_attempts=8
    bash scripts/jupiter.sh "${self_flow[@]}" "${common[@]}" "${recon_data[@]}" "${resume[@]}" name=recon \
        "trainer_params.max_steps=$target"
elif [[ $stage == gen ]]; then
    target=100000
    max_attempts=4
    recon_path=$(python3 - "$BASE/recon.json" <<'PY'
import json,sys
record=json.load(open(sys.argv[1]))
assert record['complete'], 'Recon must finish before generation starts'
print(record['checkpoint'])
PY
    )
    bash scripts/jupiter.sh --config-name gen "${common[@]}" "${gen_data[@]}" "${resume[@]}" name=gen \
        "recon_checkpoint=$recon_path" "trainer_params.max_steps=$target" +trainer_params.error_if_nonfinite=true
else
    echo "Unknown stage: $stage" >&2
    exit 2
fi
read -r step checkpoint < <(record "$stage" "$target")
test -n "$checkpoint"
if ((step < target)); then
    if ((attempt >= max_attempts)); then
        echo "Reached continuation limit at $step/$target; checkpoint retained." >&2
        exit 1
    fi
    submit "$stage" "$((attempt + 1))"
elif [[ $stage == recon ]]; then
    submit gen 1
else
    echo "Self-Flow run complete: recon 300000, gen $step."
fi

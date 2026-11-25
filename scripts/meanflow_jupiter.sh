#!/usr/bin/env bash
# RUN_ROOT contains source/, resolved.yaml, logs/, and this copied job.sh.
#SBATCH --partition=booster
#SBATCH --account=lift-virg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
#SBATCH --job-name=reptok-meanflow
set -euo pipefail
RUN_ROOT=${1:?Pass the absolute run directory}
CONFIG=${2:?Pass the Hydra experiment config name}
attempt=${3:-1}
target=300000
max_attempts=8
case "$CONFIG" in
    imf_jupiter|imf_uniform_jupiter) preflight=() ;;
    pmf_jupiter|pmf_uniform_jupiter|pmf_mix10_jupiter) preflight=(--pmf) ;;
    *) echo "Unknown experiment: $CONFIG" >&2; exit 1 ;;
esac
export REPTOK_CONTAINER=/e/home/jusers/gui2/jupiter/jenv-video.sif
export HF_HOME=/e/scratch/lift-virg/gui2/reptok/cache/huggingface
export TORCH_HOME=/e/scratch/lift-virg/gui2/reptok/cache/torch
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
cd "$RUN_ROOT/source"
export SLURM_SUBMIT_DIR="$PWD"
resume=()
if [[ -f $RUN_ROOT/recon.json ]]; then
    read -r complete checkpoint < <(python3 - "$RUN_ROOT/recon.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(int(record['complete']), record['checkpoint'])
PY
    )
    if [[ $complete == 1 ]]; then
        echo "$CONFIG already reached $target steps."
        exit 0
    fi
    resume=("resume_checkpoint=$checkpoint")
fi
if [[ $attempt == 1 ]]; then
    srun --ntasks=1 --gpus-per-task=1 apptainer exec --nv --bind /e:/e \
        "$REPTOK_CONTAINER" python tests/imf.py --device cuda "${preflight[@]}"
fi
bash scripts/jupiter.sh --config-name "$CONFIG" \
    "log_root=$RUN_ROOT/logs" "${resume[@]}"
record=$(apptainer exec --bind /e:/e "$REPTOK_CONTAINER" \
    python scripts/imf_state.py "$RUN_ROOT" "$SLURM_JOB_ID" "$target" "$RUN_ROOT/resolved.yaml")
read -r step checkpoint <<< "$record"
if ((step < target)); then
    if ((attempt >= max_attempts)); then
        echo "Continuation limit reached at $step/$target; checkpoint: $checkpoint" >&2
        exit 1
    fi
    next_id=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" \
        --job-name="reptok-${CONFIG%_jupiter}" \
        --output="$RUN_ROOT/logs/slurm/%j.out" \
        "$RUN_ROOT/job.sh" "$RUN_ROOT" "$CONFIG" "$((attempt + 1))")
    printf '%s\t%s\t%s\n' "$((attempt + 1))" "$next_id" "$SLURM_JOB_ID" >> "$RUN_ROOT/jobs.tsv"
    echo "Checkpoint $step/$target saved; continuation queued: $next_id"
else
    echo "$CONFIG complete: $step steps; checkpoint: $checkpoint"
fi

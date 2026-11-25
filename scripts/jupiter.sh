#!/usr/bin/env bash
#SBATCH --partition=booster
#SBATCH --account=lift-virg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@120
#SBATCH --job-name=reptok-recon
#SBATCH --output=logs/slurm/%j.out
set -euo pipefail
# Submit from the worktree root after: mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
export REPTOK_CONTAINER=${REPTOK_CONTAINER:-/e/home/jusers/gui2/jupiter/jenv-video.sif}
export REPTOK_GPUS_PER_NODE=${REPTOK_GPUS_PER_NODE:-4}
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
# One torchrun launcher per node, following qdino-v2. The training loop keeps
# the old RepTok multi-node fp16 gradient-compression hook.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 --gpus-per-task="$REPTOK_GPUS_PER_NODE" \
    bash -c '
        bind_args=(--bind /e:/e)
        if [[ -d /p ]]; then bind_args+=(--bind /p:/p); fi
        exec apptainer exec --nv "${bind_args[@]}" "$REPTOK_CONTAINER" \
            torchrun --nnodes="$SLURM_NNODES" \
            --nproc-per-node="$REPTOK_GPUS_PER_NODE" \
            --node-rank="$SLURM_PROCID" \
            --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
            train.py "$@" slurm_id="$SLURM_JOB_ID"
    ' bash "$@"

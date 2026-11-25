#!/usr/bin/env bash
# Run on an allocated GPU node; pass Hydra overrides after this script.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER=${REPTOK_CONTAINER:-/e/home/jusers/gui2/jupiter/jenv-video.sif}
cd "$REPO_ROOT"
bind_args=(--bind /e:/e)
if [[ -d /p ]]; then bind_args+=(--bind /p:/p); fi
exec apptainer exec --cleanenv --nv "${bind_args[@]}" --bind "$REPO_ROOT:$REPO_ROOT" \
    "$CONTAINER" python train.py "$@"

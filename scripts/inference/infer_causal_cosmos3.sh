#!/usr/bin/env bash

# Single-card warm rollout benchmark, matching the Agibot IDE configuration.
# Extra CLI args select independent opt-in fusions, for example:
#   bash scripts/inference/infer_causal_cosmos3.sh --fuse-qkv --add-rms-norm-implementation torch_npu
# Baseline: --no-fuse-qkv --add-rms-norm-implementation eager
# Set sampling.residual_reuse_full_steps: null in CONFIG for full-backbone comparisons.
set -euo pipefail

CONFIG="${CONFIG:-configs/infer/causal_cosmos3_agibot.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/episode_iter_000020000_hf}"
DATASET="${DATASET:-/mnt/sfs_turbo/public/datasets/agibot_processed/v0/task_352/canonical_55d}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
OUTPUT="${OUTPUT:-outputs/profile_agibot_$(date +%Y%m%d_%H%M%S)_$$}"

# Prefer the repository venv, then the existing inference environment on this host.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    elif [[ -x /mnt/sfs_turbo/public/apps/miniforge3/envs/infer-wam-veomni/bin/python ]]; then
        PYTHON=/mnt/sfs_turbo/public/apps/miniforge3/envs/infer-wam-veomni/bin/python
    else
        PYTHON=python
    fi
fi

# Avoid loading libtorch from an unrelated activated Conda environment.
TORCH_LIB_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/torch/lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PYTHON" tasks/infer/infer_causal_cosmos3.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --mode rollout --task policy \
    --dataset "$DATASET" \
    --episode "${EPISODE:-0}" --start "${START:-0}" \
    --max-actions 640 --device npu:0 --dtype bfloat16 --steps 20 \
    --profile --profile-warmup-runs "${WARMUP_RUNS:-1}" \
    --output "$OUTPUT" "$@"

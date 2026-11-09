#!/usr/bin/env bash
# Launch causal Cosmos3 video training in an activated torch_npu/CANN environment.
# Usage: bash scripts/causal_cosmos3/training_scripts/train_video.sh [training CLI overrides]
# Four NPUs: ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 bash scripts/causal_cosmos3/training_scripts/train_video.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a VIDEO_DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#VIDEO_DEVICES[@]}}"
export NNODES="${NNODES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/video_datasets.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_video_sample}"
DP_SHARD_SIZE="${DP_SHARD_SIZE:-$NPROC_PER_NODE}"

exec bash "$REPO_ROOT/train.sh" tasks/train_causal_cosmos3.py \
    configs/dit/causal_cosmos3_edge_video.yaml \
    --model.model_path "$MODEL_PATH" \
    --data.train_path "$DATA_CONFIG" \
    --model.accelerator.dp_shard_size "$DP_SHARD_SIZE" \
    --train.checkpoint.output_dir "$OUTPUT_DIR" \
    --train.wandb.enable false \
    "$@"

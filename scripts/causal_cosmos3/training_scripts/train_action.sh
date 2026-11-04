#!/usr/bin/env bash
# Launch causal Cosmos3 action training in an activated torch_npu/CANN environment.
# Usage: bash scripts/causal_cosmos3/training_scripts/train_action.sh [training CLI overrides]
# Four NPUs: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/causal_cosmos3/training_scripts/train_action.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,6}"
IFS=',' read -r -a VIDEO_DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#VIDEO_DEVICES[@]}}"
export NNODES="${NNODES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline

MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/action_datasets.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_action_sample}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"
mkdir -p "$WANDB_DIR"
DP_SHARD_SIZE="${DP_SHARD_SIZE:-$NPROC_PER_NODE}"

exec bash "$REPO_ROOT/train.sh" tasks/train_causal_cosmos3.py \
    configs/dit/causal_cosmos3_edge_action.yaml \
    --model.model_path "$MODEL_PATH" \
    --data.train_path "$DATA_CONFIG" \
    --model.accelerator.dp_shard_size "$DP_SHARD_SIZE" \
    --train.checkpoint.output_dir "$OUTPUT_DIR" \
    "$@"

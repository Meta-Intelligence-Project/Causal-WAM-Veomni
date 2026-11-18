#!/usr/bin/env bash
# Launch single-NPU causal Cosmos3 Policy inference in an activated torch_npu/CANN environment.
# Usage: bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh [inference CLI overrides]
# Example: ASCEND_RT_VISIBLE_DEVICES=0 EPISODE=8 OUTPUT_DIR=outputs/policy_ep8 bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_CONFIG="${INFER_CONFIG:-configs/infer/causal_cosmos3_edge_action_policy.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/robotwin_adjust_bottle_policy_8npu/checkpoints/global_step_2300/model/ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__adjust_bottle/clean}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_action_policy_infer}"
EPISODE="${EPISODE:-5}"
START="${START:-0}"
MAX_ACTIONS="${MAX_ACTIONS:-160}"
NUM_STEPS="${NUM_STEPS:-20}"
HISTORY="${HISTORY:-32}"
GUIDANCE="${GUIDANCE:-1}"
SEED="${SEED:-42}"

exec "$PYTHON_BIN" tasks/infer/infer_causal_cosmos3.py \
    --config "$INFER_CONFIG" \
    --checkpoint "$CHECKPOINT_PATH" \
    --mode teacher_forcing \
    --task policy \
    --dataset "$DATASET_ROOT" \
    --episode "$EPISODE" \
    --start "$START" \
    --max-actions "$MAX_ACTIONS" \
    --steps "$NUM_STEPS" \
    --history "$HISTORY" \
    --guidance "$GUIDANCE" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR" \
    "$@"

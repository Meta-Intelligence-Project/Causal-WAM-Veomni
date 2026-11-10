#!/usr/bin/env bash
# Run in an activated torch_npu/CANN environment. CLI overrides are appended.
# EPISODES="0 1 2" MAX_ACTIONS=64 bash scripts/causal_cosmos3/infer_idm_robotwin.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/causal_cosmos3_idm_robotwin/checkpoints/global_step_4000}"
DATASET="${DATASET:-/mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__adjust_bottle/clean}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_idm_robotwin/inference_step_4000}"
read -r -a EPISODE_IDS <<< "${EPISODES:-0 1 2}"
read -r -a INFERENCE_MODES <<< "${MODES:-teacher_forcing rollout}"
for episode in "${EPISODE_IDS[@]}"; do
    for mode in "${INFERENCE_MODES[@]}"; do
        "$PYTHON_BIN" tasks/infer/infer_causal_cosmos3.py \
            --config configs/infer/causal_cosmos3_idm_robotwin.yaml \
            --checkpoint "$CHECKPOINT" --dataset "$DATASET" \
            --mode "$mode" --task inverse_dynamics --episode "$episode" \
            --start "${START:-0}" --max-actions "${MAX_ACTIONS:-64}" \
            --steps "${STEPS:-20}" \
            --output "$OUTPUT_DIR/episode_${episode}/$mode" "$@"
    done
done

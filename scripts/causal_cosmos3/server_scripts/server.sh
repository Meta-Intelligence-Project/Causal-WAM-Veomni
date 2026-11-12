#!/usr/bin/env bash
# Single-process unified55 policy service; run in the activated VeOmni/CANN environment.
# Usage: bash scripts/causal_cosmos3/server_scripts/server.sh [server CLI overrides]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_CONFIG="${INFER_CONFIG:-configs/server/causal_cosmos3_server.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/robotwin_adjust_bottle_policy_8npu/checkpoints/global_step_2300/model/ckpt}"

exec "$PYTHON_BIN" -m tasks.client_server.server.server \
    --config "$INFER_CONFIG" \
    --checkpoint "$CHECKPOINT_PATH" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --kv-cache \
    "$@"

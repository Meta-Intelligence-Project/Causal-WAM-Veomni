#!/usr/bin/env bash
# Run in the activated RoboTwin simulator environment; no VeOmni/Torch installation is required.
# Usage: bash scripts/causal_cosmos3/client_scripts/robotwin_client.sh [client CLI overrides]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/../RoboTwin}"
SERVER_URI="${SERVER_URI:-ws://127.0.0.1:8000}"
TASK_NAME="${TASK_NAME:-adjust_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CONTROL_MODE="${CONTROL_MODE:-joint}"
STATE_SOURCE="${STATE_SOURCE:-command}"
SEED="${SEED:-100000}"
EPISODES="${EPISODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/robotwin_sim}"

client_args=(
    --robotwin-root "$ROBOTWIN_ROOT"
    --server-uri "$SERVER_URI"
    --task-name "$TASK_NAME"
    --task-config "$TASK_CONFIG"
    --control-mode "$CONTROL_MODE"
    --state-source "$STATE_SOURCE"
    --seed "$SEED"
    --episodes "$EPISODES"
    --output "$OUTPUT_DIR"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
    client_args+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${INSTRUCTION:-}" ]]; then
    client_args+=(--instruction "$INSTRUCTION")
fi
if [[ -n "${INSTRUCTION_TYPE:-}" ]]; then
    client_args+=(--instruction-type "$INSTRUCTION_TYPE")
fi

exec "$PYTHON_BIN" -m tasks.client_server.client.robotwin "${client_args[@]}" "$@"

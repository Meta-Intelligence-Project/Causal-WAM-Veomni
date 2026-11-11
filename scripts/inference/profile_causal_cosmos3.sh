#!/usr/bin/env bash

# Record CPU/NPU operators with Python stacks, and compare wall time against a
# genuinely uninstrumented rollout after identical full-horizon warmups.
# Uses the same CONFIG/CHECKPOINT/DATASET/PYTHON/card variables as inference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/infer_causal_cosmos3.sh" \
    --profile-npu-trace --compare-profile-wall "$@"

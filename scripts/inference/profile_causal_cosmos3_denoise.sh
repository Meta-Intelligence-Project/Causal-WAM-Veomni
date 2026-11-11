#!/usr/bin/env bash

# Encode one measured history window once, then benchmark/profile only its
# denoising loop. Reuse the ordinary entry's model/data/device/fusion selectors.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WARMUP_RUNS="${WARMUP_RUNS:-3}"
exec bash "$SCRIPT_DIR/infer_causal_cosmos3.sh" \
    --denoise-only --profile-npu-trace --compare-profile-wall "$@"

#!/usr/bin/env bash
# Standalone IDM on ModelArts: text/history video FM plus action FM.
# Usage: bash scripts/causal_cosmos3/training_scripts/train_idm_modelarts.sh [training CLI overrides]
# Preview: DRY_RUN=1 bash scripts/causal_cosmos3/training_scripts/train_idm_modelarts.sh
# Set TRAIN_CONFIG to a matching IDM YAML to change video_stop_sigma / partial_video_steps.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_train_triple_modelarts.sh" idm "$@"

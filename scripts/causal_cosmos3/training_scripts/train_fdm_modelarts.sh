#!/usr/bin/env bash
# Standalone FDM on ModelArts: train video and clean action/state conditioning.
# Usage: bash scripts/causal_cosmos3/training_scripts/train_fdm_modelarts.sh [training CLI overrides]
# Preview: DRY_RUN=1 bash scripts/causal_cosmos3/training_scripts/train_fdm_modelarts.sh
# The default recipe samples text-only / action-state-only / both with equal probability.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_train_triple_modelarts.sh" fdm "$@"

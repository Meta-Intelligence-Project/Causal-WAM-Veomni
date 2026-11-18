#!/usr/bin/env bash
# Run in the existing Piper/ROS environment; --check never connects hardware.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source /opt/ros/noetic/setup.bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "${PIPER_PYTHON:-/home/agilex/miniconda3/envs/piper_new/bin/python}" \
  -m tasks.client_server.client.piper \
  --config-file "$REPO_ROOT/configs/client/causal_cosmos3_piper.json" "$@"

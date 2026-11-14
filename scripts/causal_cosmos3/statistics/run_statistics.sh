#!/usr/bin/env bash
set -euo pipefail

# 修改这里的路径和配置，然后运行本脚本。
DATASET_ROOT="/path/to/datasets"
PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/zheng/envs/veomni-cosmos3/bin/python}"
DATASET_PROCESSES=4
NUM_WORKERS=4
BATCH_SIZE=512
METHOD="reservoir"  # reservoir: 采样估计分位数；exact: 精确计算，内存占用更高。
BOUNDS="q01_q99"    # 归一化边界，也可填 min_max。
RESERVOIR_SIZE=50000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec "$PYTHON_BIN" -m scripts.causal_cosmos3.statistics.compute_causal_action_stats \
    --dataset-root "$DATASET_ROOT" \
    --dataset-processes "$DATASET_PROCESSES" \
    --num-workers "$NUM_WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --method "$METHOD" \
    --bounds "$BOUNDS" \
    --reservoir-size "$RESERVOIR_SIZE" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

# 在已安装 VeOmni 依赖的环境中执行；只需修改下面的配置。
CONFIG="${CONFIG:-configs/dit/causal_cosmos3_edge_action.yaml}"
# 可选路径覆盖；MODEL_PATH 仅用于定位 text_tokenizer，不会加载模型权重。
MODEL_PATH="${MODEL_PATH:-}"
DATA_CONFIG="${DATA_CONFIG:-}"
RANKS="${RANKS:-32}"                 # 逻辑 DP rank 数，不需要对应数量的物理卡或进程。
SEED_START="${SEED_START:-42}"
RUNS="${RUNS:-10}"                  # 模拟 SEED_START 起连续 RUNS 个 seed。
WORKERS="${WORKERS:-1}"             # CPU 进程数；小任务默认串行，较大任务可实测 2/4/8。
SAMPLER_EPOCH="${SAMPLER_EPOCH:-0}"
OUTPUT="${OUTPUT:-outputs/causal_cosmos3_epoch_iters.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd -- "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PATH_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then PATH_ARGS+=(--model-path "${MODEL_PATH}"); fi
if [[ -n "${DATA_CONFIG}" ]]; then PATH_ARGS+=(--train-path "${DATA_CONFIG}"); fi

exec "${PYTHON_BIN}" -m scripts.causal_cosmos3.statistics.estimate_epoch_iters \
  --config "${CONFIG}" \
  --ranks "${RANKS}" \
  --seed-start "${SEED_START}" \
  --runs "${RUNS}" \
  --workers "${WORKERS}" \
  --sampler-epoch "${SAMPLER_EPOCH}" \
  --output "${OUTPUT}" \
  "${PATH_ARGS[@]}" \
  "$@"

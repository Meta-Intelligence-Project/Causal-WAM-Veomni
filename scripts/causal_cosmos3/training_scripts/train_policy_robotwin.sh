#!/usr/bin/env bash
# Launch RoboTwin policy/TBPTT training on ModelArts or a local Ascend host.
# Usage: bash scripts/causal_cosmos3/training_scripts/train_policy_robotwin.sh [training CLI overrides]
# Four NPUs: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/causal_cosmos3/training_scripts/train_policy_robotwin.sh
# Preview: DRY_RUN=1 bash scripts/causal_cosmos3/training_scripts/train_policy_robotwin.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/_modelarts_utils.sh"

# 集群配置
host_count="$(modelarts_host_count)"
visible_count="$(modelarts_visible_count)"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${visible_count:-${MA_NUM_GPUS:-8}}}"
export NNODES="${NNODES:-${MA_NUM_HOSTS:-$host_count}}"
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-}}"
export MASTER_ADDR="${MASTER_ADDR:-${VC_WORKER_HOSTS:-}}"
export MASTER_ADDR="${MASTER_ADDR%%,*}"
export MASTER_PORT="${MASTER_PORT:-50012}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

# HCCL 调试及超时配置
export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export LOG_TIME="${LOG_TIME:-$(date +'%Y%m%d_%H%M%S')}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CLOSE_MATMUL_K_SHIFT="${CLOSE_MATMUL_K_SHIFT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"

# 训练路径和参数
MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/action_robotwin.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_policy_robotwin_tbptt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/dit/causal_cosmos3_edge_policy.yaml}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"
DP_SHARD_SIZE="${DP_SHARD_SIZE:-$NPROC_PER_NODE}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
DRY_RUN="${DRY_RUN:-0}"
modelarts_validate_cluster
modelarts_print_topology

# 切换 conda 环境（dry-run 不激活环境或检查资源路径）
CONDA_HOME="${CONDA_HOME:-/mnt/sfs_turbo/public/apps/miniforge3}"
CONDA_ENV="${CONDA_ENV:-causal-wam}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
python_override="${PYTHON_BIN:-}"
if [[ "$DRY_RUN" != 1 ]]; then
    modelarts_check_resources
    set +u
    source "$CONDA_HOME/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    source "$ASCEND_ENV"
    set -euo pipefail
    cd "$REPO_ROOT"
    unset TORCH_DEVICE_BACKEND_AUTOLOAD
fi

# 补充环境
modelarts_select_python "$python_override"
export PYTHON_BIN PYTHON="$PYTHON_BIN"
export TMPDIR="${TMPDIR:-/tmp}"
if [[ "$DRY_RUN" != 1 ]]; then
    export FFMPEG_ROOT="$CONDA_PREFIX"
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 启动训练（辅助函数负责 torchrun 参数、节点日志和退出码透传）
modelarts_torchrun tasks/train_causal_cosmos3.py \
    "$TRAIN_CONFIG" \
    --model.model_path "$MODEL_PATH" \
    --data.train_path "$DATA_CONFIG" \
    --model.accelerator.dp_shard_size "$DP_SHARD_SIZE" \
    --train.checkpoint.output_dir "$OUTPUT_DIR" \
    "$@"

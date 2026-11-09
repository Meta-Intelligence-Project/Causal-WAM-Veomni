#!/usr/bin/env bash
# Native decision-window IDM/TBPTT on ModelArts or a local Ascend host.
# ModelArts: run this script on every instance; MA_*/VC_* resolve the topology.
# Local: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/causal_cosmos3/training_scripts/train_idm_robotwin.sh
# Preview without activating environments, launching workers or creating output:
# DRY_RUN=1 bash scripts/causal_cosmos3/training_scripts/train_idm_robotwin.sh
# Training overrides are forwarded last, e.g. --train.max_steps 30.
set -euo pipefail

fail() { printf 'train_idm_robotwin: %s\n' "$*" >&2; exit 2; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer (got '$2')"; }
trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

worker_hosts=()
if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
    IFS=',' read -r -a worker_hosts <<< "$VC_WORKER_HOSTS"
    for index in "${!worker_hosts[@]}"; do
        worker_hosts[index]="$(trim "${worker_hosts[index]}")"
        [[ -n "${worker_hosts[index]}" ]] || fail 'VC_WORKER_HOSTS contains an empty host'
    done
fi

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail 'ASCEND_RT_VISIBLE_DEVICES must be comma-separated device IDs'
    IFS=',' read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    declare -A seen_devices=()
    for device in "${visible_devices[@]}"; do
        [[ -z "${seen_devices[$device]:-}" ]] || fail "duplicate visible device: $device"
        seen_devices[$device]=1
    done
    NPROC_PER_NODE="${NPROC_PER_NODE:-${#visible_devices[@]}}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}"
    positive_int NPROC_PER_NODE "$NPROC_PER_NODE"
    ASCEND_RT_VISIBLE_DEVICES=''
    for ((device = 0; device < NPROC_PER_NODE; device++)); do
        ASCEND_RT_VISIBLE_DEVICES+="${ASCEND_RT_VISIBLE_DEVICES:+,}$device"
    done
    IFS=',' read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
fi
positive_int NPROC_PER_NODE "$NPROC_PER_NODE"
((NPROC_PER_NODE <= ${#visible_devices[@]})) || fail 'NPROC_PER_NODE exceeds the number of visible devices'
if [[ -z "${NNODES:-}" ]]; then
    if [[ -n "${MA_NUM_HOSTS:-}" ]]; then
        NNODES="$MA_NUM_HOSTS"
    elif ((${#worker_hosts[@]})); then
        NNODES="${#worker_hosts[@]}"
    else
        NNODES=1
    fi
fi
positive_int NNODES "$NNODES"
if [[ -n "${NODE_RANK:-${VC_TASK_INDEX:-}}" ]]; then
    NODE_RANK="${NODE_RANK:-$VC_TASK_INDEX}"
elif ((NNODES == 1)); then
    NODE_RANK=0
else
    fail 'multi-node launch requires NODE_RANK or VC_TASK_INDEX on every instance'
fi
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'NODE_RANK must be a nonnegative integer'
((NODE_RANK < NNODES)) || fail 'NODE_RANK must be smaller than NNODES'
MASTER_ADDR="$(trim "${MASTER_ADDR:-${worker_hosts[0]:-}}")"
MASTER_ADDR="$(trim "${MASTER_ADDR%%,*}")"
if ((NNODES > 1)); then
    [[ -n "$MASTER_ADDR" ]] || fail 'multi-node launch requires MASTER_ADDR or VC_WORKER_HOSTS'
    case "${MASTER_ADDR,,}" in 127.*|localhost|::1|0.0.0.0) fail 'multi-node MASTER_ADDR must be reachable from the other instances' ;; esac
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
[[ "$MASTER_ADDR" != *[[:space:]]* ]] || fail 'MASTER_ADDR must be a single host address'
MASTER_PORT="${MASTER_PORT:-50012}"
positive_int MASTER_PORT "$MASTER_PORT"
((MASTER_PORT <= 65535)) || fail 'MASTER_PORT must be at most 65535'
DP_SHARD_SIZE="${DP_SHARD_SIZE:-$NPROC_PER_NODE}"
positive_int DP_SHARD_SIZE "$DP_SHARD_SIZE"
world_size=$((NNODES * NPROC_PER_NODE))
((world_size % DP_SHARD_SIZE == 0)) || fail 'DP_SHARD_SIZE must divide the total device count'
export ASCEND_RT_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT

CONDA_HOME="${CONDA_HOME:-/mnt/sfs_turbo/public/apps/miniforge3}"
CONDA_ENV="${CONDA_ENV:-causal-wam}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
python_override="${PYTHON_BIN:-}"
if [[ -n "$python_override" ]]; then
    PYTHON_BIN="$python_override"
elif [[ "$CONDA_ENV" == base ]]; then
    PYTHON_BIN="$CONDA_HOME/bin/python"
elif [[ "$CONDA_ENV" == */* ]]; then
    PYTHON_BIN="$CONDA_ENV/bin/python"
else
    PYTHON_BIN="$CONDA_HOME/envs/$CONDA_ENV/bin/python"
fi
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/dit/causal_cosmos3_edge_idm.yaml}"
MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/action_robotwin.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_cosmos3_idm_robotwin_tbptt}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
LOG_TIME="${LOG_TIME:-$(date +'%Y%m%d_%H%M%S')}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail 'DRY_RUN must be 0 or 1'
export CLOSE_MATMUL_K_SHIFT="${CLOSE_MATMUL_K_SHIFT:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"

launch_command=("$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK")
if ((NNODES == 1)); then
    launch_command+=(--standalone)
else
    launch_command+=(--master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
fi
launch_command+=(tasks/train_causal_cosmos3.py "$TRAIN_CONFIG"
    --model.model_path "$MODEL_PATH"
    --data.train_path "$DATA_CONFIG"
    --model.accelerator.dp_shard_size "$DP_SHARD_SIZE"
    --train.checkpoint.output_dir "$OUTPUT_DIR"
    "$@")
printf 'Launcher defaults: nodes=%s devices/node=%s node_rank=%s master=%s:%s shard=%s replicas=%s\n' \
    "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT" "$DP_SHARD_SIZE" "$((world_size / DP_SHARD_SIZE))"
if [[ "$DRY_RUN" == 1 ]]; then
    # This is a candidate path; normal launches resolve named envs after activation.
    printf 'COMMAND: '; printf '%q ' "${launch_command[@]}"; printf '\n'
    exit 0
fi

[[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]] || fail "missing conda activation script under $CONDA_HOME"
[[ -f "$ASCEND_ENV" ]] || fail "missing Ascend environment script: $ASCEND_ENV"
[[ -f "$TRAIN_CONFIG" && -f "$DATA_CONFIG" ]] || fail 'TRAIN_CONFIG and DATA_CONFIG must point to existing YAML files'
[[ -d "$MODEL_PATH" ]] || fail "MODEL_PATH must point to the Cosmos3 checkpoint root: $MODEL_PATH"
set +u
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source "$ASCEND_ENV"
set -euo pipefail
cd "$REPO_ROOT"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
if [[ -z "$python_override" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="$python_override"
fi
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
python_bin_dir="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)"
conda_bin_dir="$(cd -- "$CONDA_PREFIX/bin" && pwd -P)"
[[ "$python_bin_dir" == "$conda_bin_dir" ]] || fail 'PYTHON_BIN must belong to the activated CONDA_ENV; use its environment prefix as CONDA_ENV'
launch_command[0]="$PYTHON_BIN"
printf 'COMMAND: '; printf '%q ' "${launch_command[@]}"; printf '\n'
export PYTHON_BIN PYTHON="$PYTHON_BIN"
export FFMPEG_ROOT="$CONDA_PREFIX"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$LOG_DIR"
log_path="$LOG_DIR/train_node${NODE_RANK}_${LOG_TIME}.log"
printf 'Node %s training log: %s\n' "$NODE_RANK" "$log_path"
"${launch_command[@]}" 2>&1 | tee -a "$log_path"

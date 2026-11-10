#!/usr/bin/env bash
# Sourced helpers for the RoboTwin launcher; sourcing performs no setup or I/O.

modelarts_fail() { printf 'train_policy_robotwin_tbptt_modelarts: %s\n' "$*" >&2; exit 2; }
modelarts_positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || modelarts_fail "$1 must be a positive integer (got '$2')"; }
modelarts_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

modelarts_host_count() {
    local host
    local -a hosts=()
    if [[ -z "${VC_WORKER_HOSTS:-}" ]]; then
        printf '1'
        return
    fi
    IFS=',' read -r -a hosts <<< "$VC_WORKER_HOSTS"
    for host in "${hosts[@]}"; do
        [[ -n "$(modelarts_trim "$host")" ]] || modelarts_fail 'VC_WORKER_HOSTS contains an empty host'
    done
    printf '%s' "${#hosts[@]}"
}

modelarts_visible_count() {
    [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] || return 0
    [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || modelarts_fail 'ASCEND_RT_VISIBLE_DEVICES must be comma-separated device IDs'
    local device
    local -a devices=()
    local -A seen=()
    IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    for device in "${devices[@]}"; do
        [[ -z "${seen[$device]:-}" ]] || modelarts_fail "duplicate visible device: $device"
        seen[$device]=1
    done
    printf '%s' "${#devices[@]}"
}

modelarts_validate_cluster() {
    local device
    local -a devices=()
    modelarts_positive_int NPROC_PER_NODE "$NPROC_PER_NODE"
    if [[ -z "$ASCEND_RT_VISIBLE_DEVICES" ]]; then
        for ((device = 0; device < NPROC_PER_NODE; device++)); do
            ASCEND_RT_VISIBLE_DEVICES+="${ASCEND_RT_VISIBLE_DEVICES:+,}$device"
        done
    fi
    IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    ((NPROC_PER_NODE <= ${#devices[@]})) || modelarts_fail 'NPROC_PER_NODE exceeds the number of visible devices'
    modelarts_positive_int NNODES "$NNODES"
    if [[ -z "$NODE_RANK" ]]; then
        ((NNODES == 1)) || modelarts_fail 'multi-node launch requires NODE_RANK or VC_TASK_INDEX on every instance'
        NODE_RANK=0
    fi
    [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || modelarts_fail 'NODE_RANK must be a nonnegative integer'
    ((NODE_RANK < NNODES)) || modelarts_fail 'NODE_RANK must be smaller than NNODES'
    MASTER_ADDR="$(modelarts_trim "$MASTER_ADDR")"
    if ((NNODES > 1)); then
        [[ -n "$MASTER_ADDR" ]] || modelarts_fail 'multi-node launch requires MASTER_ADDR or VC_WORKER_HOSTS'
        case "${MASTER_ADDR,,}" in 127.*|localhost|::1|0.0.0.0) modelarts_fail 'multi-node MASTER_ADDR must be reachable from the other instances' ;; esac
    else
        MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    fi
    [[ "$MASTER_ADDR" != *[[:space:]]* ]] || modelarts_fail 'MASTER_ADDR must be a single host address'
    modelarts_positive_int MASTER_PORT "$MASTER_PORT"
    ((MASTER_PORT <= 65535)) || modelarts_fail 'MASTER_PORT must be at most 65535'
    modelarts_positive_int DP_SHARD_SIZE "$DP_SHARD_SIZE"
    ((NNODES * NPROC_PER_NODE % DP_SHARD_SIZE == 0)) || modelarts_fail 'DP_SHARD_SIZE must divide the total device count'
    [[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || modelarts_fail 'DRY_RUN must be 0 or 1'
}

modelarts_print_topology() {
    printf 'Launcher defaults: nodes=%s devices/node=%s node_rank=%s master=%s:%s shard=%s replicas=%s\n' \
        "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT" "$DP_SHARD_SIZE" "$((NNODES * NPROC_PER_NODE / DP_SHARD_SIZE))"
}

modelarts_check_resources() {
    [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]] || modelarts_fail "missing conda activation script under $CONDA_HOME"
    [[ -f "$ASCEND_ENV" ]] || modelarts_fail "missing Ascend environment script: $ASCEND_ENV"
    [[ -f "$TRAIN_CONFIG" && -f "$DATA_CONFIG" ]] || modelarts_fail 'TRAIN_CONFIG and DATA_CONFIG must point to existing YAML files'
    [[ -d "$MODEL_PATH" ]] || modelarts_fail "MODEL_PATH must point to the Cosmos3 checkpoint root: $MODEL_PATH"
}

modelarts_select_python() {
    local override="$1" python_bin_dir conda_bin_dir
    if [[ -n "$override" ]]; then
        PYTHON_BIN="$override"
    elif [[ "$DRY_RUN" != 1 ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ "$CONDA_ENV" == base ]]; then
        PYTHON_BIN="$CONDA_HOME/bin/python"
    elif [[ "$CONDA_ENV" == */* ]]; then
        PYTHON_BIN="$CONDA_ENV/bin/python"
    else
        PYTHON_BIN="$CONDA_HOME/envs/$CONDA_ENV/bin/python"
    fi
    if [[ "$DRY_RUN" != 1 ]]; then
        [[ -x "$PYTHON_BIN" ]] || modelarts_fail "Python is not executable: $PYTHON_BIN"
        python_bin_dir="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)"
        conda_bin_dir="$(cd -- "$CONDA_PREFIX/bin" && pwd -P)"
        [[ "$python_bin_dir" == "$conda_bin_dir" ]] || modelarts_fail 'PYTHON_BIN must belong to the activated CONDA_ENV; use its environment prefix as CONDA_ENV'
    fi
}

modelarts_prepare_directories() {
    local setting directory probe
    for setting in OUTPUT_DIR WANDB_DIR LOG_DIR TMPDIR; do
        directory="${!setting}"
        if ! mkdir -p -- "$directory"; then
            modelarts_fail "cannot create $setting='$directory' (cwd: $PWD). Set $setting to a writable directory; OUTPUT_DIR must be shared by all instances."
        fi
        if ! probe="$(mktemp -- "$directory/.veomni-write-check.XXXXXX")"; then
            modelarts_fail "cannot write to $setting='$directory' (cwd: $PWD). Set $setting to a writable directory; OUTPUT_DIR must be shared by all instances."
        fi
        rm -- "$probe" || modelarts_fail "cannot remove the write probe in $setting='$directory'"
    done
}

modelarts_torchrun() {
    local log_path
    local -a launch_command=("$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK")
    if ((NNODES == 1)); then
        launch_command+=(--standalone)
    else
        launch_command+=(--master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
    fi
    launch_command+=("$@")
    printf 'COMMAND: '; printf '%q ' "${launch_command[@]}"; printf '\n'
    [[ "$DRY_RUN" != 1 ]] || return 0
    modelarts_prepare_directories
    log_path="$LOG_DIR/train_node${NODE_RANK}_${LOG_TIME}.log"
    printf 'Node %s training log: %s\n' "$NODE_RANK" "$log_path"
    "${launch_command[@]}" 2>&1 | tee -a "$log_path"
}

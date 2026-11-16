#!/usr/bin/env bash
# Independent single-NPU replicas; keep this supervisor alive while clients run.
# CHECKPOINT_PATH=/path/to/checkpoint DEVICE_IDS=0,1,2,3,4,5,6,7 BASE_PORT=8000 \
#   bash scripts/causal_cosmos3/server_scripts/server_multi.sh --task inverse_dynamics
# Use --task policy for Policy weights; KV/config options are forwarded to server.sh.
# Wait for all eight logs to say "Causal inference listening" before starting clients.
# --dry-run only prints device/port mappings. Ctrl+C stops this replica group.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

IFS=',' read -r -a devices <<< "${DEVICE_IDS:-0,1,2,3,4,5,6,7}"
base_port="${BASE_PORT:-8000}"
log_dir="${LOG_DIR:-outputs/robotwin_servers_$(date +%Y%m%d_%H%M%S_%N)}"
dry_run=false
server_args=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --port|--port=*|--device|--device=*)
            echo "Use BASE_PORT and DEVICE_IDS; each replica uses logical npu:0." >&2; exit 2 ;;
        *) server_args+=("$arg") ;;
    esac
done
[[ "$base_port" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "Invalid BASE_PORT" >&2; exit 2; }
(( base_port + ${#devices[@]} - 1 <= 65535 )) || { echo "Port range exceeds 65535" >&2; exit 2; }
declare -A seen=()
for device in "${devices[@]}"; do
    [[ "$device" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid NPU device: $device" >&2; exit 2; }
    [[ ! -v "seen[$device]" ]] || { echo "Duplicate NPU device: $device" >&2; exit 2; }
    seen[$device]=1
done
[[ "${DEVICE_IDS:-x}" != *, ]] || { echo "Trailing comma in DEVICE_IDS" >&2; exit 2; }
if ! "$dry_run"; then
    command -v setsid >/dev/null
    mkdir -p "$log_dir"
fi
pids=()
cleanup() {
    trap '' INT TERM
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    if (( ${#pids[@]} )); then sleep 2; fi
    for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in "${!devices[@]}"; do
    port=$(( base_port + i ))
    log="$log_dir/server_${i}_npu${devices[$i]}_port${port}.log"
    echo "server $i: physical NPU ${devices[$i]} -> logical npu:0, port $port, log $log"
    if "$dry_run"; then continue; fi
    ASCEND_RT_VISIBLE_DEVICES="${devices[$i]}" PORT="$port" \
        setsid bash "$SCRIPT_DIR/server.sh" "${server_args[@]}" --device npu:0 >"$log" 2>&1 &
    pids+=("$!")
done
if "$dry_run"; then exit 0; fi
echo "Replicas launched. Wait for all server logs to report listening before starting clients."
# A replica exiting leaves its client without a server. Fail the group explicitly.
status=0
wait -n "${pids[@]}" || status=$?
echo "A server exited (status $status); stopping the remaining owned replicas. See $log_dir" >&2
exit 1

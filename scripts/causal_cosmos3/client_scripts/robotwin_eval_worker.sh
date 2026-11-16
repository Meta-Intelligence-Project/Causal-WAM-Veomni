#!/usr/bin/env bash
# Internal worker: reads its fixed assignment from plan.tsv. One server, one GPU.
# Failure/timeout stops this worker; other workers continue. No automatic retries.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/robotwin_eval_common.sh"
worker="${1:?worker index required}"
[[ "$worker" =~ ^[0-9]+$ && -f "$OUTPUT_DIR/plan.tsv" ]] || exit 2
exec 9>"$OUTPUT_DIR/worker_$worker.lock"
flock -n 9 || { echo "Worker $worker is already running" >&2; exit 2; }
child=''
cleanup() {
    trap '' INT TERM
    if [[ -n "$child" ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while IFS=$'\t' read -r config task assigned episodes control device server; do
    [[ "$assigned" == "$worker" ]] || continue
    log_dir="$OUTPUT_DIR/logs/$task/$config"
    mkdir -p "$log_dir"
    echo "worker $worker: $config / $task on GPU $device -> $server"
    args=(--robotwin-root "$ROBOTWIN_ROOT" --server-uri "$server" --task-name "$task"
          --task-config "$config" --control-mode "$control" --state-source "$STATE_SOURCE"
          --seed "$SEED" --episodes "$episodes" --output "$OUTPUT_DIR/runs")
    if [[ -n "${MAX_STEPS:-}" ]]; then args+=(--max-steps "$MAX_STEPS"); fi
    if [[ -n "${INSTRUCTION_TYPE:-}" ]]; then args+=(--instruction-type "$INSTRUCTION_TYPE"); fi
    # SAPIEN 3's legacy core.SapienRenderer ignores device kwargs. Hold the explicit
    # RenderSystem while running the UNMODIFIED client entry point, including resets.
    CUDA_VISIBLE_DEVICES="$device" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
        setsid timeout --foreground --kill-after=10s "$JOB_TIMEOUT" "$PYTHON_BIN" -u -c '
import runpy
import sapien.render
sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
renderer = sapien.render.RenderSystem(device="cuda:0")
print("SAPIEN render device:", renderer.device, flush=True)
runpy.run_module("tasks.client_server.client.robotwin", run_name="__main__")
' "${args[@]}" > "$log_dir/client.log" 2>&1 &
    child=$!
    code=0
    wait "$child" || code=$?
    printf '%s\n' "$code" > "$log_dir/exitcode"
    if (( code != 0 )); then
        echo "Task failed (exit=$code). Worker stops to avoid reusing a busy/dead server. See $log_dir/client.log" >&2
        exit 1
    fi
    child=''
done < "$OUTPUT_DIR/plan.tsv"

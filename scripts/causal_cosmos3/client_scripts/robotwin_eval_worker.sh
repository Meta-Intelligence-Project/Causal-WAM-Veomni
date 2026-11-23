#!/usr/bin/env bash
# Internal worker: reads its fixed assignment from plan.tsv. One server, one GPU.
# Episode recovery/retries are handled by robotwin_eval_episodes.sh; failed tasks
# stay incomplete while this worker continues its remaining fixed assignments.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/robotwin_eval_common.sh"
worker="${1:?worker index required}"
[[ "$worker" =~ ^[0-9]+$ && -f "$OUTPUT_DIR/plan.tsv" ]] || exit 2
# Hold a shared plan lock through the whole worker/child lifetime. Never read a
# plan while a resume coordinator is changing task ownership.
exec 6>"$OUTPUT_DIR/plan.lock"
flock -sn 6 || { echo 'Plan reassignment in progress; worker skipped.' >&2; exit 75; }
exec 9>"$OUTPUT_DIR/worker_$worker.lock"
flock -n 9 || { echo "Worker $worker is already running; skipped without touching its results." >&2; exit 75; }
child=''
cleanup() {
    trap '' INT TERM
    if [[ -n "$child" ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
status=0
while IFS=$'\t' read -r config task assigned episodes control device server; do
    [[ "$assigned" == "$worker" ]] || continue
    log_dir="$OUTPUT_DIR/logs/$task/$config"
    run_dir="$OUTPUT_DIR/runs/$task/$config/$control"
    if [[ "${RESUME:-0}" == 1 ]]; then
        if detail="$(bash "$SCRIPT_DIR/robotwin_eval_check.sh" "$OUTPUT_DIR" "$task" "$config" "$control" "$episodes")"; then
            echo "SKIP complete: $task / $config"
            continue
        fi
        echo "RETRY $task / $config: $detail"
    fi
    mkdir -p "$log_dir"
    echo "worker $worker: $config / $task on GPU $device -> $server"
    # Keep earlier logs and valid seed directories; the coordinator resumes episodes.
    setsid bash "$SCRIPT_DIR/robotwin_eval_episodes.sh" "$worker" "$task" "$config" "$control" "$episodes" "$device" "$server" >> "$log_dir/client.log" 2>&1 &
    child=$!
    code=0
    wait "$child" || code=$?
    printf '%s\n' "$code" > "$log_dir/exitcode"
    child=''
    if (( code != 0 )); then
        echo "Task incomplete (exit=$code); episode attempts are in $log_dir/client.log. Continuing next task." >&2
        status=1
        continue
    fi
    if ! detail="$(bash "$SCRIPT_DIR/robotwin_eval_check.sh" "$OUTPUT_DIR" "$task" "$config" "$control" "$episodes")"; then
        echo "Incomplete artifacts: $detail; continuing next task." >&2
        status=1
    fi
done < "$OUTPUT_DIR/plan.tsv"
exit "$status"

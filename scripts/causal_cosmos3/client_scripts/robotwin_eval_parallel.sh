#!/usr/bin/env bash
# Start at most eight independent clients, each paired with its own model server.
# SERVER_HOST=10.0.0.1 ROBOTWIN_ROOT=/path/to/RoboTwin OUTPUT_DIR=outputs/all_tasks \
#   bash scripts/causal_cosmos3/client_scripts/robotwin_eval_parallel.sh
# Settings: CLIENT_DEVICES=0,1,2,3,4,5,6,7 BASE_PORT=8000 EPISODES=10
#           TASK_NAME=all TASK_CONFIG=demo_clean,demo_randomized JOB_TIMEOUT=7200
# Test first: TASK_NAME=adjust_bottle,lift_pot EPISODES=1 CLIENT_DEVICES=0,1
# Requires bash, jq, flock, setsid, timeout; use a NEW OUTPUT_DIR for each run.
# Same checkpoint/task/sampling settings on all servers. Policy and IDM both work.
# Resume: OUTPUT_DIR=/old/run JOB_TIMEOUT=21600 bash this_script.sh --resume
# Resume reads the original plan, skips locked workers and reuses valid episodes.
# EPISODE_TIMEOUT=1800 EPISODE_RETRIES=1 RETRY_DELAY=15; MAX_CONSECUTIVE_FAILURES=3.
# MAX_EPISODE_ATTEMPTS defaults to 5*EPISODES. JOB_TIMEOUT is the per-episode fallback.
# Progress printed automatically; STATUS_INTERVAL=60 seconds (0 disables periodic output).
# STATUS_STALE_SECONDS=900 warns about quiet logs; no automatic restart.
set -euo pipefail
if [[ $# == 1 && "$1" == --resume ]]; then
    exec bash "$(dirname -- "${BASH_SOURCE[0]}")/robotwin_eval_resume.sh"
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/robotwin_eval_common.sh"
if [[ "${1:-}" == --help ]]; then sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0; fi
[[ $# == 0 || ( $# == 1 && "$1" == --dry-run ) ]] || { echo 'Use --help / --dry-run / --resume; settings via environment.' >&2; exit 2; }
source "$SCRIPT_DIR/../monitor.sh" --library
validate_status_settings
validate_settings
for i in "${!devices[@]}"; do
    assigned=()
    for t in "${!tasks[@]}"; do
        if (( t % ${#devices[@]} == i )); then assigned+=("${tasks[$t]}"); fi
    done
    echo "worker $i: GPU ${devices[$i]} -> ws://$SERVER_HOST:$((BASE_PORT+i)); tasks: ${assigned[*]}"
done
echo "Plan: ${#tasks[@]} tasks x ${#configs[@]} configs x $EPISODES episodes; output $OUTPUT_DIR"
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
for tool in jq flock setsid timeout; do command -v "$tool" >/dev/null; done
for task in "${tasks[@]}"; do
    [[ -f "$ROBOTWIN_ROOT/envs/$task.py" ]] || { echo "Missing task: $ROBOTWIN_ROOT/envs/$task.py" >&2; exit 2; }
done
[[ -f "$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py" ]] || { echo 'Missing RoboTwin eval helper' >&2; exit 2; }
for config in "${configs[@]}"; do
    found=false
    for folder in env_cfg/task_config task_config; do
        for ext in yml yaml; do
            if [[ -f "$ROBOTWIN_ROOT/$folder/$config.$ext" ]]; then found=true; fi
        done
    done
    "$found" || { echo "Missing task config: $config" >&2; exit 2; }
done
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR" # Never mix previous results into a new evaluation.
mkdir "$OUTPUT_DIR/logs" "$OUTPUT_DIR/runs"
jq -n --arg ROBOTWIN_ROOT "$ROBOTWIN_ROOT" --arg STATE_SOURCE "$STATE_SOURCE" --arg SEED "$SEED" \
    --arg MAX_STEPS "${MAX_STEPS:-}" --arg INSTRUCTION_TYPE "${INSTRUCTION_TYPE:-}" \
    '{ROBOTWIN_ROOT:$ROBOTWIN_ROOT, STATE_SOURCE:$STATE_SOURCE, SEED:$SEED, MAX_STEPS:$MAX_STEPS, INSTRUCTION_TYPE:$INSTRUCTION_TYPE}' \
    > "$OUTPUT_DIR/eval_settings.json"
printf 'config\ttask\tworker\tepisodes\tcontrol\tdevice\tserver\n' > "$OUTPUT_DIR/plan.tsv"
for config in "${configs[@]}"; do
    for t in "${!tasks[@]}"; do
        i=$((t % ${#devices[@]}))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$config" "${tasks[$t]}" "$i" "$EPISODES" "$CONTROL_MODE" "${devices[$i]}" "ws://$SERVER_HOST:$((BASE_PORT+i))" >> "$OUTPUT_DIR/plan.tsv"
    done
done
pids=()
cleanup() {
    trap '' INT TERM
    stop_status_monitor
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    # Workers first clean their separately-owned task process groups.
    if (( ${#pids[@]} )); then sleep 4; fi
    for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in "${!devices[@]}"; do
    setsid bash "$SCRIPT_DIR/robotwin_eval_worker.sh" "$i" > "$OUTPUT_DIR/logs/worker_$i.log" 2>&1 &
    pids+=("$!")
done
start_status_monitor client "$OUTPUT_DIR"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
pids=()
stop_status_monitor
show_final_client_status "$OUTPUT_DIR" || status=1
exit "$status"

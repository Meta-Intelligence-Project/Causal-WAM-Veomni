#!/usr/bin/env bash
# Shared settings for the shell-only RoboTwin evaluation scripts. Source, do not execute.
# Official tasks: https://robotwin-platform.github.io/doc/tasks/index.html
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

all_tasks=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic
    hanging_mug lift_pot move_can_pot move_playingcard_away move_stapler_pad open_laptop
    open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
    place_bread_basket place_bread_skillet place_burger_fries place_can_basket
    place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan
    place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand
    place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two
    stack_bowls_three stack_bowls_two stamp_seal turn_switch move_pillbottle_pad
)
tasks=("${all_tasks[@]}")
if [[ "${TASK_NAME:-all}" != all ]]; then IFS=',' read -r -a tasks <<< "$TASK_NAME"; fi
IFS=',' read -r -a configs <<< "${TASK_CONFIG:-demo_clean,demo_randomized}"
IFS=',' read -r -a devices <<< "${CLIENT_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/../RoboTwin}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export BASE_PORT="${BASE_PORT:-8000}"
export CONTROL_MODE="${CONTROL_MODE:-joint}"
export STATE_SOURCE="${STATE_SOURCE:-command}"
export EPISODES="${EPISODES:-10}"
export SEED="${SEED:-100000}"
export JOB_TIMEOUT="${JOB_TIMEOUT:-7200}" # fallback seconds per isolated episode; prefer EPISODE_TIMEOUT
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/robotwin_parallel_$(date +%Y%m%d_%H%M%S_%N)}"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
ROBOTWIN_ROOT="$(realpath -m -- "$ROBOTWIN_ROOT")"

validate_settings() {
    [[ "$EPISODES" =~ ^[1-9][0-9]*$ && "$JOB_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'EPISODES/JOB_TIMEOUT must be positive integers' >&2; return 2; }
    [[ "$BASE_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( BASE_PORT + ${#devices[@]} - 1 <= 65535 )) || { echo 'Invalid port range' >&2; return 2; }
    [[ "$CONTROL_MODE" == joint || "$CONTROL_MODE" == eef ]] || { echo 'CONTROL_MODE: joint or eef' >&2; return 2; }
    [[ "$STATE_SOURCE" == command || "$STATE_SOURCE" == measured ]] || { echo 'STATE_SOURCE: command or measured' >&2; return 2; }
    local list item
    for list in tasks configs devices; do
        local -n values="$list"
        local -A seen=()
        for item in "${values[@]}"; do
            [[ "$item" =~ ^[a-zA-Z0-9_-]+$ && ! -v "seen[$item]" ]] || { echo "Invalid/duplicate $list entry: $item" >&2; return 2; }
            seen[$item]=1
        done
    done
    [[ "${TASK_NAME:-x}" != *, && "${TASK_CONFIG:-x}" != *, && "${CLIENT_DEVICES:-x}" != *, ]] || { echo 'Remove trailing commas' >&2; return 2; }
}

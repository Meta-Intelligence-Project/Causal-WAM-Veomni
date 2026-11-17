#!/usr/bin/env bash
# Aggregate an existing run, using its plan (not the current shell's task settings).
# bash scripts/causal_cosmos3/client_scripts/robotwin_eval_summary.sh outputs/your_run
# Output: per_task.tsv, summary.json, summary.tsv. Rates in JSON are fractions.
# Missing/error jobs stay INCOMPLETE; they are never silently omitted from the average.
# Finished jobs also require episode JSON, actions and structurally complete videos.
# --progress writes progress.json only: record counts, WITHOUT artifact checks.
# Its separate lock is nonblocking (exit 75 = another progress scan in progress).
set -euo pipefail
trap 'exit 2' ERR # Distinguish scan errors from an ordinary incomplete report.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="${1:?Usage: robotwin_eval_summary.sh OUTPUT_DIR}"
[[ -f "$output/plan.tsv" ]] || { echo "Missing $output/plan.tsv" >&2; exit 2; }
progress=false
report_name=summary
if [[ "${2:-}" == --progress && $# == 2 ]]; then
    progress=true
    report_name=progress
elif [[ $# != 1 ]]; then
    echo 'Usage: robotwin_eval_summary.sh OUTPUT_DIR [--progress]' >&2
    exit 2
fi
command -v jq >/dev/null
exec 7>"$output/$report_name.lock"
if "$progress"; then
    flock -n 7 || { echo 'Progress scan already running; skipping this refresh.' >&2; exit 75; }
else
    flock 7 # Final verification remains serialized.
fi
rows="$(mktemp "$output/.summary_rows.XXXXXX")"
trap 'rm -f -- "$rows" "$rows.report"' EXIT
while IFS=$'\t' read -r config task worker expected control device server; do
    [[ "$config" != config ]] || continue
    metric="$output/runs/$task/$config/$control/metrics.json"
    code_file="$output/logs/$task/$config/exitcode"
    code=missing
    if [[ -f "$code_file" ]]; then code="$(cat "$code_file")"; fi
    data=null
    if [[ -f "$metric" ]]; then
        data="$(jq -sc 'try (
            if length != 1 then null else .[0] |
                if type != "object" then null
                elif (.episodes|type) != "array" then null
                elif all(.episodes[]; type == "object") then . else null end
            end) catch null' "$metric" 2>/dev/null)" || data=null
    fi
    artifacts=false
    detail='Process not complete'
    if [[ "$code" == 0 && "$progress" == true ]]; then detail='Not checked: progress snapshot only'; fi
    if [[ "$code" == 0 && "$progress" == false ]]; then
        if detail="$(bash "$SCRIPT_DIR/robotwin_eval_check.sh" "$output" "$task" "$config" "$control" "$expected")"; then artifacts=true; fi
    fi
    jq -nc --arg config "$config" --arg task "$task" --arg control "$control" \
        --arg worker "$worker" --arg device "$device" --arg server "$server" \
        --arg code "$code" --argjson progress "$progress" --argjson artifacts "$artifacts" --arg detail "$detail" --argjson expected "$expected" --argjson m "$data" '
        (if ($m.episodes | type) == "array" then $m.episodes else [] end) as $episodes |
        [$episodes[] | select(.status == "finished")] as $done |
        ([$done[] | select(.success == true)] | length) as $success |
        ($m.task == $task and $m.task_config == $config and $m.control_mode == $control
          and ($m.episodes | type) == "array"
          and all($done[]; (.success | type) == "boolean" and .seed != null)
          and ([$done[].seed] | unique | length) == ($done | length)) as $valid |
        ($valid and $code == "0" and ($done | length) == $expected
          and ($episodes | length) == $expected) as $records_complete |
        ($records_complete and $artifacts) as $complete |
        {config:$config, task:$task, control:$control, worker:$worker, device:$device,
         server:$server, exitcode:$code, expected:$expected, complete:$complete, records_complete:$records_complete, artifact_check:$detail,
         completed:(if $valid then ($done|length) else 0 end),
         successes:(if $valid then $success else 0 end),
         success_rate:(if $complete then $success/$expected else null end),
         status:(if $complete then "complete" elif $progress and $records_complete then "record_complete_unverified" elif $code == "missing" then "pending_or_interrupted" else "error" end)}
    ' >> "$rows"
done < "$output/plan.tsv"
jq -s --argjson progress "$progress" '
    . as $jobs |
    {check_level:(if $progress then "progress" else "full" end), generated_at:(now|todateiso8601), jobs:$jobs, groups: (group_by([.config,.control]) | map(
      . as $g | (all(.[]; .complete)) as $complete |
      {config:.[0].config, control:.[0].control, complete:$complete,
       tasks:length, record_complete_tasks:([.[]|select(.records_complete)]|length), complete_tasks:([.[]|select(.complete)]|length),
       expected_episodes:(map(.expected)|add), completed_episodes:(map(.completed)|add),
       successes:(map(.successes)|add),
       macro_success_rate:(if $complete then (map(.success_rate)|add)/length else null end),
       micro_success_rate:(if $complete then (map(.successes)|add)/(map(.completed)|add) else null end)}))}
' "$rows" > "$rows.report"
mv -- "$rows.report" "$output/$report_name.json"
if "$progress"; then exit 0; fi
jq -r '(["config","task","worker","device","completed","expected","successes","success_rate","status"] | @tsv),
    (.jobs[] | [.config,.task,.worker,.device,.completed,.expected,.successes,.success_rate,.status] | @tsv)' \
    "$output/summary.json" > "$output/per_task.tsv"
jq -r '(["config","control","complete_tasks","tasks","episodes","successes","macro_success_percent","micro_success_percent"] | @tsv),
    (.groups[] | [.config,.control,.complete_tasks,.tasks,.completed_episodes,.successes,
      (if .complete then .macro_success_rate*100 else "INCOMPLETE" end),
      (if .complete then .micro_success_rate*100 else "INCOMPLETE" end)] | @tsv)' \
    "$output/summary.json" | tee "$output/summary.tsv"
echo "Results: $output/summary.json ; $output/per_task.tsv ; $output/summary.tsv"
if jq -e '.groups | length > 0 and all(.[]; .complete)' "$output/summary.json" >/dev/null; then
    exit 0
else
    exit 1
fi

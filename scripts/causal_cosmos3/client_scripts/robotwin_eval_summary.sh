#!/usr/bin/env bash
# Aggregate an existing run, using its plan (not the current shell's task settings).
# bash scripts/causal_cosmos3/client_scripts/robotwin_eval_summary.sh outputs/your_run
# Output: per_task.tsv, summary.json, summary.tsv. Rates in JSON are fractions.
# Missing/error jobs stay INCOMPLETE; they are never silently omitted from the average.
set -euo pipefail
output="${1:?Usage: robotwin_eval_summary.sh OUTPUT_DIR}"
[[ -f "$output/plan.tsv" ]] || { echo "Missing $output/plan.tsv" >&2; exit 2; }
command -v jq >/dev/null
rows="$(mktemp "$output/.summary_rows.XXXXXX")"
trap 'rm -f -- "$rows"' EXIT
while IFS=$'\t' read -r config task worker expected control device server; do
    [[ "$config" != config ]] || continue
    metric="$output/runs/$task/$config/$control/metrics.json"
    code_file="$output/logs/$task/$config/exitcode"
    code=missing
    if [[ -f "$code_file" ]]; then code="$(cat "$code_file")"; fi
    data=null
    if [[ -f "$metric" ]]; then data="$(jq -c . "$metric" 2>/dev/null)" || data=null; fi
    jq -nc --arg config "$config" --arg task "$task" --arg control "$control" \
        --arg worker "$worker" --arg device "$device" --arg server "$server" \
        --arg code "$code" --argjson expected "$expected" --argjson m "$data" '
        (if ($m.episodes | type) == "array" then $m.episodes else [] end) as $episodes |
        [$episodes[] | select(.status == "finished")] as $done |
        ([$done[] | select(.success == true)] | length) as $success |
        ($m.task == $task and $m.task_config == $config and $m.control_mode == $control
          and ($m.episodes | type) == "array"
          and all($done[]; (.success | type) == "boolean" and .seed != null)
          and ([$done[].seed] | unique | length) == ($done | length)) as $valid |
        ($valid and $code == "0" and ($done | length) == $expected
          and ($episodes | length) == $expected) as $complete |
        {config:$config, task:$task, control:$control, worker:$worker, device:$device,
         server:$server, exitcode:$code, expected:$expected, complete:$complete,
         completed:(if $valid then ($done|length) else 0 end),
         successes:(if $valid then $success else 0 end),
         success_rate:(if $complete then $success/$expected else null end),
         status:(if $complete then "complete" elif $code == "missing" then "pending_or_interrupted" else "error" end)}
    ' >> "$rows"
done < "$output/plan.tsv"
jq -s '
    . as $jobs |
    {jobs:$jobs, groups: (group_by([.config,.control]) | map(
      . as $g | (all(.[]; .complete)) as $complete |
      {config:.[0].config, control:.[0].control, complete:$complete,
       tasks:length, complete_tasks:([.[]|select(.complete)]|length),
       expected_episodes:(map(.expected)|add), completed_episodes:(map(.completed)|add),
       successes:(map(.successes)|add),
       macro_success_rate:(if $complete then (map(.success_rate)|add)/length else null end),
       micro_success_rate:(if $complete then (map(.successes)|add)/(map(.completed)|add) else null end)}))}
' "$rows" > "$rows.report"
mv -- "$rows.report" "$output/summary.json"
jq -r '(["config","task","worker","device","completed","expected","successes","success_rate","status"] | @tsv),
    (.jobs[] | [.config,.task,.worker,.device,.completed,.expected,.successes,.success_rate,.status] | @tsv)' \
    "$output/summary.json" > "$output/per_task.tsv"
jq -r '(["config","control","complete_tasks","tasks","episodes","successes","macro_success_percent","micro_success_percent"] | @tsv),
    (.groups[] | [.config,.control,.complete_tasks,.tasks,.completed_episodes,.successes,
      (if .complete then .macro_success_rate*100 else "INCOMPLETE" end),
      (if .complete then .micro_success_rate*100 else "INCOMPLETE" end)] | @tsv)' \
    "$output/summary.json" | tee "$output/summary.tsv"
echo "Results: $output/summary.json ; $output/per_task.tsv ; $output/summary.tsv"
jq -e '.groups | length > 0 and all(.[]; .complete)' "$output/summary.json" >/dev/null

#!/usr/bin/env bash
# Resume the ORIGINAL plan, without changing worker/GPU/endpoint assignments.
# OUTPUT_DIR=/absolute/path/to/old_run JOB_TIMEOUT=21600 bash .../robotwin_eval_parallel.sh --resume
# Completed task/config groups are checked and untouched. Incomplete groups are
# recovered episode by episode; only incomplete episode artifacts are archived.
# Locked workers are skipped; they may still be running the old launcher.
# Keep server weights/task/KV/sampling config unchanged. No model hash is available.
# Legacy runs recover client settings from info.json; supply original settings if
# no info.json exists. Changing JOB_TIMEOUT is allowed. Episode failures have bounded retries; completed episodes are retained.
# Progress printed automatically; STATUS_INTERVAL=60 seconds (0 disables periodic output).
# STATUS_STALE_SECONDS=900 warns about quiet logs; no automatic restart.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"
[[ -n "${OUTPUT_DIR:-}" ]] || { echo 'Set OUTPUT_DIR to the existing evaluation directory.' >&2; exit 2; }
export OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
[[ -f "$OUTPUT_DIR/plan.tsv" ]] || { echo 'Missing original plan.tsv' >&2; exit 2; }
for tool in jq flock setsid timeout; do command -v "$tool" >/dev/null; done
# Only one resume launcher may schedule this output at a time. Old launchers do
# not use this lock; per-worker locks below protect their live clients as well.
exec 8>"$OUTPUT_DIR/resume.lock"
flock -n 8 || { echo 'Another resume launcher is active for this output.' >&2; exit 2; }
"${PYTHON_BIN:-python}" - "$OUTPUT_DIR" <<'PY'
import csv
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

root = Path(sys.argv[1])
with (root / 'plan.tsv').open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
if not rows:
    raise SystemExit('Empty plan')
identities, workers = set(), {}
for row in rows:
    for key in ('task', 'config', 'control'):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', row[key]):
            raise SystemExit('Unsafe plan path')
    if row['control'] not in ('joint', 'eef') or not row['worker'].isdigit() or int(row['episodes']) <= 0:
        raise SystemExit('Invalid plan values')
    uri = urlsplit(row['server'])
    if uri.scheme not in ('ws', 'wss') or not uri.hostname or not uri.port:
        raise SystemExit('Invalid server URI')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', row['device']):
        raise SystemExit('Invalid device')
    identity = row['task'], row['config']
    if identity in identities:
        raise SystemExit('Duplicate task/config in plan')
    identities.add(identity)
    pair = row['device'], row['server']
    old = workers.setdefault(row['worker'], pair)
    if old != pair:
        raise SystemExit('Worker maps to multiple devices/endpoints')
if len({p[0] for p in workers.values()}) != len(workers) or len({p[1] for p in workers.values()}) != len(workers):
    raise SystemExit('Workers must have distinct devices and servers')

fields = dict(ROBOTWIN_ROOT='robotwin_root', STATE_SOURCE='state_source', SEED='seed',
              MAX_STEPS='max_steps', INSTRUCTION_TYPE='instruction_type')
def settings(info):
    args = info['client_args']
    if not isinstance(args, dict) or not all(field in args for field in fields.values()):
        raise ValueError('Incomplete client_args')
    if not isinstance(args['robotwin_root'], str) or not args['robotwin_root'] or args['state_source'] not in ('command', 'measured'):
        raise ValueError('Invalid client settings')
    if type(args['seed']) is not int or (args['max_steps'] is not None and (type(args['max_steps']) is not int or args['max_steps'] <= 0)):
        raise ValueError('Invalid seed/max_steps')
    if args['instruction_type'] not in (None, 'seen', 'unseen'):
        raise ValueError('Invalid instruction_type')
    result = {key: '' if args[field] is None else str(args[field]) for key, field in fields.items()}
    if 'evaluation_seed' in info:
        if type(info['evaluation_seed']) is not int:
            raise ValueError('Invalid evaluation_seed')
        result['SEED'] = str(info['evaluation_seed'])
    return result
path = root / 'eval_settings.json'
known = json.loads(path.read_text()) if path.exists() else None
# Valid completed episode metadata is also available in the old shell-only runs.
for row in rows:
    for p in (root / 'runs' / row['task'] / row['config'] / row['control']).glob('seed*/info.json'):
        try:
            saved = settings(json.loads(p.read_text()))
        except (ValueError, KeyError, TypeError, OSError):
            continue  # Incomplete/corrupt artifacts are handled by the worker checker.
        if known is None:
            known = saved
        if saved != known:
            raise SystemExit(f'Conflicting client settings in {p}; refusing to mix experiments')
if known is None:
    if not all(k in os.environ for k in ('ROBOTWIN_ROOT', 'STATE_SOURCE', 'SEED')):
        raise SystemExit('No saved settings: explicitly supply original ROBOTWIN_ROOT, STATE_SOURCE and SEED')
    known = {key: os.environ.get(key, '') for key in fields}
    print('Warning: no saved episode settings; using explicitly supplied original settings.')
for key in fields:
    if key in os.environ:
        value = os.environ[key]
        if key == 'ROBOTWIN_ROOT':
            value = str(Path(value).expanduser().resolve())
        if value != known[key]:
            raise SystemExit(f'{key} differs from original run; use original settings')
    if '\n' in known[key] or '\t' in known[key]:
        raise SystemExit('Tabs/newlines are not supported in settings')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(known, indent=2)+'\n')
temporary.replace(path)
PY
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(jq -r 'to_entries[] | [.key,.value] | @tsv' "$OUTPUT_DIR/eval_settings.json")
source "$SCRIPT_DIR/robotwin_eval_common.sh"
export RESUME=1
[[ "$JOB_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'JOB_TIMEOUT must be positive' >&2; exit 2; }
[[ -f "$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py" ]] || { echo 'RoboTwin checkout missing' >&2; exit 2; }
source "$SCRIPT_DIR/../monitor.sh" --library
validate_status_settings
pids=()
cleanup() {
    trap '' INT TERM
    stop_status_monitor
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    if (( ${#pids[@]} )); then sleep 4; fi
    for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stamp="$(date +%Y%m%d_%H%M%S_%N)"
mapfile -t workers < <(awk -F '\t' 'NR>1 {print $3}' "$OUTPUT_DIR/plan.tsv" | sort -nu)
for worker in "${workers[@]}"; do
    log="$OUTPUT_DIR/logs/worker_${worker}.resume_${stamp}.log"
    echo "Resume worker $worker: $log"
    setsid bash "$SCRIPT_DIR/robotwin_eval_worker.sh" "$worker" > "$log" 2>&1 8>&- &
    pids+=("$!")
done
start_status_monitor client "$OUTPUT_DIR"
status=0
for pid in "${pids[@]}"; do
    code=0
    wait "$pid" || code=$?
    if (( code == 75 )); then
        echo 'Skipped an already-running worker; it continues in its original session.'
    elif (( code != 0 )); then status=1; fi
done
pids=()
stop_status_monitor
show_final_client_status "$OUTPUT_DIR" || status=1
exit "$status"

#!/usr/bin/env bash
# Internal helper, called by resume while it holds resume.lock.
# EXCLUDE_CLIENT_DEVICES matches plan.tsv device values, not worker IDs or NPU IDs.
# RESTORE_CLIENT_DEVICES restores saved pairings and balances incomplete jobs only.
# Only scheduling changes: result paths, seeds, episode targets and task settings stay intact.
set -euo pipefail
exec "${PYTHON_BIN:-python}" - "${1:?evaluation output directory required}" "$(dirname -- "${BASH_SOURCE[0]}")" <<'PY'
import csv
import fcntl
import io
import os
import subprocess
from pathlib import Path
import re
import sys
import tempfile
from contextlib import ExitStack

root = Path(sys.argv[1])
exclude_raw = os.environ.get('EXCLUDE_CLIENT_DEVICES', '')
restore_raw = os.environ.get('RESTORE_CLIENT_DEVICES', '')
if bool(exclude_raw) == bool(restore_raw):
    raise SystemExit('Set exactly one of EXCLUDE_CLIENT_DEVICES or RESTORE_CLIENT_DEVICES')
option = 'RESTORE_CLIENT_DEVICES' if restore_raw else 'EXCLUDE_CLIENT_DEVICES'
devices = (restore_raw or exclude_raw).split(',')
if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', d) for d in devices) or len(set(devices)) != len(devices):
    raise SystemExit(f'{option} must be a comma-separated list of distinct plan device IDs, without spaces')
excluded = set(devices) if exclude_raw else set()

with ExitStack() as locks:
    def lock(path):
        stream = locks.enter_context(path.open('a'))
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Cannot reassign while client workers are active ({path.name}); stop the client launcher and workers first')

    lock(root / 'plan.lock')
    plan = root / 'plan.tsv'
    original = plan.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode()), delimiter='\t')
    rows = list(reader)
    columns = reader.fieldnames
    if not rows or not {'task', 'config', 'worker', 'device', 'server', 'episodes', 'control'} <= set(columns or []):
        raise SystemExit('Invalid or empty plan.tsv')
    workers = {}
    for row in rows:
        if not row['worker'].isdigit():
            raise SystemExit('Invalid worker ID')
        pair = row['device'], row['server']
        if workers.setdefault(row['worker'], pair) != pair:
            raise SystemExit('Worker maps to multiple devices/endpoints')
    if len({p[0] for p in workers.values()}) != len(workers) or len({p[1] for p in workers.values()}) != len(workers):
        raise SystemExit('Workers must have distinct devices and servers')
    restored = {}
    if restore_raw:
        missing = set(devices) - {pair[0] for pair in workers.values()}
        # Recover exact original pairings; never guess server ports from GPU IDs.
        for backup in sorted(root.glob('plan.before_*.tsv')):
            with backup.open() as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    if row['device'] not in missing:
                        continue
                    worker, pair = row['worker'], (row['device'], row['server'])
                    if not worker.isdigit() or restored.setdefault(worker, pair) != pair:
                        raise SystemExit(f'Conflicting worker mapping in {backup}')
        if missing - {pair[0] for pair in restored.values()}:
            raise SystemExit('Cannot restore requested GPUs: original mappings missing from plan backups')
        for worker, pair in restored.items():
            if workers.setdefault(worker, pair) != pair:
                raise SystemExit('Restored worker conflicts with current plan')
        if len({p[0] for p in workers.values()}) != len(workers) or len({p[1] for p in workers.values()}) != len(workers):
            raise SystemExit('Restored GPU/server mappings conflict; plan.tsv was not changed')
    # Also protect against workers started from the previous script version.
    for worker in sorted(workers, key=int):
        lock(root / f'worker_{worker}.lock')
    healthy = {w: pair for w, pair in workers.items() if pair[0] not in excluded}
    if not healthy:
        raise SystemExit('No healthy worker remains; plan.tsv was not changed')
    affected = [row for row in rows if row['device'] in excluded]
    if restored:
        # Rebalance only incomplete jobs. Full checks preserve the existing
        # completion contract; a stalled check aborts before plan publication.
        checker = Path(sys.argv[2])
        for row in rows:
            try:
                check = subprocess.run(
                    ['bash', str(checker / 'robotwin_eval_check.sh'), str(root),
                     row['task'], row['config'], row['control'], row['episodes']],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                raise SystemExit(f"Completion check timed out for {row['task']}/{row['config']}; plan unchanged")
            if check.returncode == 1:
                affected.append(row)
            elif check.returncode != 0:
                raise SystemExit(f'Completion check failed; plan unchanged: {check.stderr}')
    if not affected:
        print('No task reassignment needed; keeping the saved plan unchanged.', flush=True)
        raise SystemExit(0)
    previous_assignments = [(row['worker'], row['device'], row['server']) for row in rows]
    # Balance task/config rows, not runtime estimates. Completed rows remain in
    # the plan for full summaries and are skipped by the normal resume checker.
    loads = {w: 0 if restored else sum(row['worker'] == w for row in rows) for w in healthy}
    for row in affected:
        old = row['worker']
        # Use the requested set, not only newly restored workers, so repeated
        # restoration stays stable even when there are fewer jobs than GPUs.
        worker = min(healthy, key=lambda w: (loads[w], bool(restore_raw) and healthy[w][0] not in devices, int(w)))
        row.update(worker=worker, device=healthy[worker][0], server=healthy[worker][1])
        loads[worker] += 1
        print(f"REASSIGN {row['task']}/{row['config']}: worker {old} -> {worker}, GPU {row['device']}, {row['server']}", flush=True)
    if previous_assignments == [(row['worker'], row['device'], row['server']) for row in rows]:
        print('Assignments already match; keeping the saved plan unchanged.', flush=True)
        raise SystemExit(0)
    # Retain the exact previous plan; publish the replacement only once complete.
    prefix = 'plan.before_restore.' if restore_raw else 'plan.before_exclude.'
    backup_fd, backup = tempfile.mkstemp(prefix=prefix, suffix='.tsv', dir=root)
    with os.fdopen(backup_fd, 'wb') as stream:
        stream.write(original)
    fd, temporary = tempfile.mkstemp(prefix='.plan.', suffix='.tsv', dir=root)
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, plan)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Plan updated: {len(affected)} task/config rows reassigned; {len(healthy)} workers remain. Backup: {backup}', flush=True)
PY

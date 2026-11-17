#!/usr/bin/env bash
# Read-only diagnostics (writes reports only; never connects to inference ports).
# bash scripts/causal_cosmos3/monitor.sh client OUTPUT_DIR [--watch]
# bash scripts/causal_cosmos3/monitor.sh server LOG_DIR [--watch]
# Launchers start this automatically. STATUS_INTERVAL=60; STATUS_STALE_SECONDS=900.
# STATUS_INTERVAL=0 disables automatic monitoring; standalone --watch needs >0.
# Client summaries verify artifacts; status.txt/status.log record progress. A quiet
# log or a live process does not prove a hang or a healthy inference service.
# Sourcing with --library exposes lifecycle helpers without starting a monitor.
ROBOTWIN_MONITOR_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/monitor.sh"
export ROBOTWIN_MONITOR_SCRIPTS="$(dirname -- "$ROBOTWIN_MONITOR_SCRIPT")"
status_monitor_pid=''
validate_status_settings() {
    [[ "${STATUS_INTERVAL:-60}" =~ ^(0|[1-9][0-9]*)$ && "${STATUS_STALE_SECONDS:-900}" =~ ^[1-9][0-9]*$ ]] || {
        echo 'STATUS_INTERVAL must be >=0; STATUS_STALE_SECONDS must be >0 (seconds).' >&2; return 2;
    }
}
start_status_monitor() {
    validate_status_settings
    if [[ "${STATUS_INTERVAL:-60}" == 0 ]]; then return; fi
    setsid bash "$ROBOTWIN_MONITOR_SCRIPT" "$1" "$2" --watch 8>&- 9>&- 7>&- &
    status_monitor_pid=$!
}
stop_status_monitor() {
    if [[ -n "$status_monitor_pid" ]]; then
        # The child may not have entered setsid yet: terminate its PID as well.
        kill -TERM "$status_monitor_pid" 2>/dev/null || true
        kill -TERM -- "-$status_monitor_pid" 2>/dev/null || true
        wait "$status_monitor_pid" 2>/dev/null || true
        status_monitor_pid=''
    fi
}
if [[ "${1:-}" == --library ]]; then return 0; fi
set -euo pipefail
exec "${PYTHON_BIN:-python}" - "$@" <<'PY'
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

mode, location, *options = sys.argv[1:]
if mode not in ('client', 'server') or options not in ([], ['--watch']):
    raise SystemExit('Usage: monitor.sh client OUTPUT_DIR | server LOG_DIR [--watch]')
root = Path(location).resolve()
watch = bool(options)
interval = int(os.environ.get('STATUS_INTERVAL', '60'))
stale = int(os.environ.get('STATUS_STALE_SECONDS', '900'))
if (watch and interval <= 0) or stale <= 0:
    raise SystemExit('Watch interval and stale threshold must be positive')
# Supplied by the shell wrapper below via environment, independent of caller cwd.
scripts = Path(os.environ['ROBOTWIN_MONITOR_SCRIPTS'])


def tail(path, limit=3000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode(errors='replace').strip()
    except OSError:
        return ''


def lock_table():
    # Never acquire worker locks. After the flock utility exits, inherited locks
    # may disappear from /proc/locks but remain in the owner's fdinfo/9 on Linux.
    texts = []
    unknown = False
    paths = [Path('/proc/locks')]
    paths += [p / 'fdinfo/9' for p in Path('/proc').iterdir() if p.name.isdigit()]
    for path in paths:
        try:
            texts.append(path.read_text())
        except FileNotFoundError:
            continue
        except OSError:
            unknown = True
    held = set()
    for text in texts:
        for line in text.splitlines():
            fields = line.split()
            if '->' in fields:
                continue
            for kind in ('FLOCK', 'POSIX', 'OFDLCK'):
                if kind in fields:
                    try:
                        device = fields[fields.index(kind) + 4].split(':')
                        held.add((int(device[0], 16), int(device[1], 16), int(device[2])))
                    except (ValueError, IndexError):
                        unknown = True
    return held, unknown


def locked(path, table):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    identity = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    held, unknown = table
    return True if identity in held else (None if unknown else False)


def latest_worker_log(worker):
    paths = list((root / 'logs').glob(f'worker_{worker}.resume_*.log'))
    paths += list((root / 'logs').glob(f'worker_{worker}.log'))
    return max(paths, key=lambda p: p.stat().st_mtime, default=None)


def client_status():
    # Summary returns 1 for an incomplete evaluation, not necessarily a failure.
    result = subprocess.run(['bash', str(scripts / 'client_scripts/robotwin_eval_summary.sh'), str(root)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError('Summary check failed: ' + (result.stderr or result.stdout)[-2000:])
    report = json.loads((root / 'summary.json').read_text())
    if not report['jobs']:
        raise RuntimeError('Empty evaluation plan')
    lines = ['[CLIENT] 完整任务 %d/%d；episode 记录 %d/%d（记录齐全仍需文件校验）' % (
        sum(j['complete'] for j in report['jobs']), len(report['jobs']),
        sum(j['completed'] for j in report['jobs']), sum(j['expected'] for j in report['jobs']))]
    for g in report['groups']:
        rate = f"{g['macro_success_rate'] * 100:.2f}%" if g['complete'] else 'INCOMPLETE'
        lines.append(f"  {g['config']}/{g['control']}: {g['complete_tasks']}/{g['tasks']}任务；平均成功率 {rate}")
    table = lock_table()
    for worker in sorted({j['worker'] for j in report['jobs']}, key=int):
        jobs = [j for j in report['jobs'] if j['worker'] == worker]
        done = sum(j['complete'] for j in jobs)
        busy = locked(root / f'worker_{worker}.lock', table)
        errors = [j for j in jobs if j['status'] == 'error']
        state = '全部完成' if done == len(jobs) else ('活动锁占用/运行或检查中' if busy else ('已停止但未完成' if busy is False else '本机锁状态未知'))
        lines.append(f"  worker {worker} [{state}] {done}/{len(jobs)}任务；episode={sum(j['completed'] for j in jobs)}/{sum(j['expected'] for j in jobs)}；异常 {len(errors)}；server={jobs[0]['server']}")
        log = latest_worker_log(worker)
        if done < len(jobs):
            if log:
                lines.append(f'    worker日志: {log}')
                # The last dispatch line identifies the current/last attempted job.
                recent = tail(log, 16384).splitlines()
                dispatch = [s for s in recent if s.startswith(f'worker {worker}: ')]
                if dispatch:
                    lines.append('    最近分配: ' + dispatch[-1])
                    label = dispatch[-1].split(': ', 1)[1].split(' on GPU ', 1)[0]
                    current = next((j for j in jobs if label == f"{j['config']} / {j['task']}"), None)
                    if current:
                        lines.append(f"    当前/最近任务episode记录: {current['completed']}/{current['expected']}")
                        lines.append(f"    任务日志: {root / 'logs' / current['task'] / current['config'] / 'client.log'}")
            for j in errors:
                detail = j['artifact_check']
                if j['exitcode'] == '124':
                    detail = '整组任务超过 JOB_TIMEOUT（不等于网络故障）'
                lines.append(f"    ERROR {j['task']}/{j['config']} exit={j['exitcode']} episode={j['completed']}/{j['expected']} {detail}")
                path = root / 'logs' / j['task'] / j['config'] / 'client.log'
                lines.append(f'      日志: {path}')
                lines.extend('      ' + s for s in tail(path).splitlines()[-3:])
            if busy is False:
                lines.append('    提示：检查任务日志和对应server后，可再次 --resume；完整任务保留。')
            elif busy:
                paths = [root / 'logs' / j['task'] / j['config'] / 'client.log' for j in jobs]
                if log:
                    paths.append(log)
                mtimes = [p.stat().st_mtime for p in paths if p.exists()]
                age = time.time() - max(mtimes, default=time.time())
                if age >= stale:
                    lines.append(f'    WARN 约{int(age)}秒无日志更新；可能在长计算/等待，并不能据此判定卡死。')
    if all(j['complete'] for j in report['jobs']):
        lines.append('ALL COMPLETE：原计划全部通过完整性检查。任务执行失败的episode也计入评测结果。')
    lines.append('服务端健康：未主动探测；端口/进程存活不保证推理正常。')
    return lines


def server_status():
    with (root / 'server_processes.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    lines = ['[SERVER] 进程与日志状态（不占用推理连接，不验证响应能力）']
    for row in rows:
        try:
            proc = Path('/proc') / row['pid'] / 'stat'
            fields = proc.read_text().rsplit(')', 1)[1].split()
            alive = fields[0] != 'Z' and fields[19] == row['start_ticks']
        except (OSError, IndexError):
            alive = False
        path = Path(row['log'])
        text = tail(path, 16384)
        state = '进程存活' if alive else '进程已退出'
        lines.append(f"  server {row['worker']} NPU={row['device']} port={row['port']} pid={row['pid']} [{state}]")
        lines.append(f'    日志: {path}')
        warnings = [s for s in text.splitlines() if any(k in s.lower() for k in ('error', 'traceback', 'timeout', 'out of memory', 'exception'))]
        for line in warnings[-2:]:
            lines.append('    日志提示（可能是此前记录）: ' + line)
    return lines


def stop(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
while True:
    try:
        lines = client_status() if mode == 'client' else server_status()
    except Exception as error:
        lines = [f'MONITOR ERROR: {error}；监控失败不代表任务已停止，请检查日志。']
    rendered = '\n'.join([time.strftime('[%Y-%m-%d %H:%M:%S %z]')] + lines) + '\n'
    print(rendered, flush=True)
    # Multiple read-only monitors may coexist; publish snapshots atomically.
    fd, temporary = tempfile.mkstemp(prefix='.status_', dir=root)
    with os.fdopen(fd, 'w') as stream:
        stream.write(rendered)
    os.replace(temporary, root / 'status.txt')
    with (root / 'status.log').open('a') as stream:
        stream.write(rendered)
    if not watch:
        break
    time.sleep(interval)
PY

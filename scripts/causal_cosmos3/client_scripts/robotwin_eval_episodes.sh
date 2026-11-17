#!/usr/bin/env bash
# Internal task coordinator: WORKER TASK CONFIG CONTROL EPISODES DEVICE SERVER.
# Each isolated simulator process runs one episode; valid old episodes are reused.
# EPISODE_TIMEOUT defaults to JOB_TIMEOUT (7200s), now per episode, not per group.
# EPISODE_RETRIES=1 retries the same starting seed once; then moves past that seed.
# MAX_EPISODE_ATTEMPTS defaults to 5*EPISODES; MAX_CONSECUTIVE_FAILURES=3.
# RETRY_DELAY=15 seconds. Limits leave the task incomplete; worker continues others.
# Infrastructure failures are logged separately and never counted as task success.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/robotwin_eval_common.sh"
export ROBOTWIN_EPISODE_SCRIPTS="$SCRIPT_DIR"
exec "$PYTHON_BIN" - "$@" <<'PY'
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

worker, task, config, control, target, device, server = sys.argv[1:]
target = int(target)
root = Path(os.environ['OUTPUT_DIR'])
run = root / 'runs' / task / config / control
logs = root / 'logs' / task / config
scripts = Path(os.environ['ROBOTWIN_EPISODE_SCRIPTS'])
base_seed = int(os.environ['SEED'])
def number(key, default, minimum=1):
    value = int(os.environ.get(key, default))
    if value < minimum:
        raise ValueError(f'{key} must be >= {minimum}')
    return value
limit = number('EPISODE_TIMEOUT', os.environ['JOB_TIMEOUT'])
retries = number('EPISODE_RETRIES', 1, 0)
max_attempts = number('MAX_EPISODE_ATTEMPTS', target * 5)
max_failures = number('MAX_CONSECUTIVE_FAILURES', 3)
delay = number('RETRY_DELAY', 15, 0)
run.mkdir(parents=True, exist_ok=True)
logs.mkdir(parents=True, exist_ok=True)
active = None
stopping = False


def stop(signum, frame):
    global stopping
    stopping = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def defer_stop(signum, frame):
    global stopping
    stopping = True


def terminate(process):
    handlers = {sig: signal.signal(sig, defer_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in (signal.SIGCONT, signal.SIGTERM):
            try: os.killpg(process.pid, sig)
            except ProcessLookupError: pass
        try: process.wait(timeout=0.5)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        try: process.wait(timeout=0.5)
        except subprocess.TimeoutExpired: pass
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        if stopping: raise SystemExit(143)


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.episode_', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def valid(check_root, seed):
    result = subprocess.run(['bash', str(scripts / 'robotwin_eval_check.sh'), str(check_root),
                             task, config, control, '1', '--episode', str(seed)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def archive(path):
    if not path.exists(): return
    parent = root / 'archive' / task / config / control
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='episode_' + time.strftime('%Y%m%d_%H%M%S_'), dir=parent))
    shutil.move(str(path), str(destination / path.name))
    print(f'Archived attempt metadata/partial files: {destination}', flush=True)


records = []
failures = []
skipped = []
previous = {}
old = run / 'metrics.json'
if old.exists():
    try:
        previous = json.loads(old.read_text())
        failures = previous.get('infrastructure_failures', [])
        skipped = previous.get('skipped_seeds', [])
    except (ValueError, AttributeError):
        previous = {}
if not isinstance(failures, list): failures = []
if not isinstance(skipped, list): skipped = []
# Recover records even if a prior process died before metrics.json was written.
for folder in sorted(run.glob('seed*'), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else -1):
    if not folder.is_dir() or not folder.name[4:].isdigit(): continue
    seed = int(folder.name[4:])
    if valid(root, seed):
        info = json.loads((folder / 'info.json').read_text())
        args = info['client_args']
        expected = dict(robotwin_root=os.environ['ROBOTWIN_ROOT'], state_source=os.environ['STATE_SOURCE'],
                        max_steps=int(os.environ['MAX_STEPS']) if os.environ.get('MAX_STEPS') else None,
                        instruction_type=os.environ.get('INSTRUCTION_TYPE') or None)
        if any(args.get(k) != v for k, v in expected.items()) or info.get('evaluation_seed', args['seed']) != base_seed:
            raise ValueError(f'Existing episode settings differ: {folder}')
        records.append(info)
        print(f'REUSE seed={seed} success={info["success"]}: {folder}', flush=True)
    else:
        archive(folder)
if len(records) > target:
    raise ValueError('More completed episodes than requested; refusing to discard results')


completed_seeds = {r['seed'] for r in records}


def promote(folder, code):
    info = json.loads((folder / 'info.json').read_text())
    args = info['client_args']
    expected = dict(robotwin_root=os.environ['ROBOTWIN_ROOT'], state_source=os.environ['STATE_SOURCE'],
                    max_steps=int(os.environ['MAX_STEPS']) if os.environ.get('MAX_STEPS') else None,
                    instruction_type=os.environ.get('INSTRUCTION_TYPE') or None)
    if any(args.get(k) != v for k, v in expected.items()) or info.get('evaluation_seed', base_seed) != base_seed:
        raise ValueError(f'Staged episode settings differ: {folder}')
    actual_seed = info['seed']
    if actual_seed in completed_seeds or actual_seed < base_seed:
        raise ValueError('Simulator returned duplicate or earlier seed')
    info['evaluation_seed'] = base_seed
    info['episode'] = len(records)
    info['attempt_exitcode'] = code
    final = run / folder.name
    def relocate(value):
        if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(str(folder) + '/'):
            return str(final) + value[len(str(folder)):]
        return value
    requests = json.loads((folder / 'requests.json').read_text())
    atomic_json(folder / 'requests.json', relocate(requests))
    for key in ('info_path', 'requests_path', 'action_path', 'video_path', 'prediction_video_path'):
        if info.get(key): info[key] = str(final / Path(info[key]).name)
    atomic_json(folder / 'info.json', info)
    shutil.move(str(folder), str(final))
    records.append(info)
    completed_seeds.add(actual_seed)
    return actual_seed


def attempt_metadata(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError): return {}


def merge_skips(items):
    if not isinstance(items, list): return
    existing = {json.dumps(item, sort_keys=True) for item in skipped}
    for item in items:
        if not isinstance(item, dict) or type(item.get('seed')) is not int: continue
        key = json.dumps(item, sort_keys=True)
        if key not in existing:
            skipped.append(item)
            existing.add(key)


# A process can be interrupted after writing complete files but before promotion.
staging = run / '.episode_attempt'
temp_run = staging / 'runs' / task / config / control
for folder in temp_run.glob('seed*'):
    if len(records) < target and folder.name[4:].isdigit() and int(folder.name[4:]) not in completed_seeds and valid(staging, int(folder.name[4:])):
        recovered = promote(folder, 'interrupted_after_save')
        print(f'RECOVER staged complete episode seed={recovered}', flush=True)
merge_skips(attempt_metadata(temp_run / 'metrics.json').get('skipped_seeds', []))
archive(staging)
seed = max([base_seed - 1] + [r['seed'] for r in records]) + 1
if type(previous.get('next_seed')) is int: seed = max(seed, previous['next_seed'])
retry = previous.get('retry_index', 0) if seed == previous.get('next_seed') else 0
if type(retry) is not int or not 0 <= retry <= retries: retry = 0
consecutive = 0


def publish():
    atomic_json(run / 'metrics.json', dict(task=task, task_config=config, control_mode=control,
        episodes=records, skipped_seeds=skipped, completed_episodes=len(records),
        successful_episodes=sum(r['success'] for r in records),
        success_rate=sum(r['success'] for r in records)/len(records) if records else None,
        infrastructure_failures=failures, evaluation_seed=base_seed, next_seed=seed, retry_index=retry))


publish()
launch = '''import os
import sapien.render
sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
renderer = sapien.render.RenderSystem(device="cuda:0")
print("SAPIEN render device:", renderer.device, flush=True)
from tasks.client_server.client.robotwin import simulator
# Keep the instruction candidate pool identical to the original multi-episode run.
original_load = simulator.load_robotwin
def load_with_original_instruction_count(root):
    helpers = original_load(root)
    build = helpers.build_instruction
    def build_instruction(task_args, episode_info, instruction_type, episodes, seed):
        return build(task_args, episode_info, instruction_type, int(os.environ["ROBOTWIN_TARGET_EPISODES"]), seed)
    helpers.build_instruction = build_instruction
    return helpers
simulator.load_robotwin = load_with_original_instruction_count
simulator.main()
'''
for attempt in range(max_attempts):
    if len(records) >= target: break
    while seed in completed_seeds: seed += 1
    # Preserve any abandoned staging from an interrupted older coordinator.
    staging = run / '.episode_attempt'
    archive(staging)
    staging.mkdir()
    temp_run = staging / 'runs' / task / config / control
    args = ['--robotwin-root', os.environ['ROBOTWIN_ROOT'], '--server-uri', server,
            '--task-name', task, '--task-config', config, '--control-mode', control,
            '--state-source', os.environ['STATE_SOURCE'], '--seed', str(seed),
            '--episodes', '1', '--output', str(staging / 'runs')]
    for env_key, flag in [('MAX_STEPS', '--max-steps'), ('INSTRUCTION_TYPE', '--instruction-type')]:
        if os.environ.get(env_key): args += [flag, os.environ[env_key]]
    attempt_log = logs / ('episode_' + time.strftime('%Y%m%d_%H%M%S_') + str(time.time_ns()) + '.log')
    print(f'EPISODE {len(records)}/{target} seed={seed} retry={retry}/{retries} timeout={limit}s log={attempt_log}', flush=True)
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, ROBOTWIN_TARGET_EPISODES=str(target),
                     OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'), MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS','1'))
    started = time.monotonic()
    code = 0
    with attempt_log.open('w') as stream:
        active = subprocess.Popen([os.environ['PYTHON_BIN'], '-u', '-c', launch, *args],
                                  env=child_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = active.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            terminate(active)
            active = None
    with (logs / 'attempts.jsonl').open('a') as journal:
        journal.write(json.dumps(dict(seed=seed, exitcode=code, seconds=time.monotonic()-started, log=str(attempt_log))) + '\n')
    # Accept a complete saved episode even if its process failed during teardown.
    found = []
    for folder in temp_run.glob('seed*'):
        if folder.name[4:].isdigit() and valid(staging, int(folder.name[4:])):
            found.append(folder)
    meta = attempt_metadata(temp_run / 'metrics.json')
    merge_skips(meta.get('skipped_seeds', []))
    if len(found) == 1:
        if code != 0:
            failures.append(dict(seed=seed, exitcode=code, reason='complete_artifacts_after_teardown_failure', log=str(attempt_log)))
        actual_seed = promote(found[0], code)
        seed = actual_seed + 1
        consecutive = retry = 0
        print(f'FINISHED {len(records)}/{target}: seed={actual_seed} success={records[-1]["success"]}', flush=True)
    else:
        consecutive += 1
        failures.append(dict(seed=seed, exitcode=code, reason='timeout' if code == 124 else 'process_or_artifact_failure',
                             seconds=time.monotonic()-started, log=str(attempt_log)))
        print(f'ATTEMPT FAILED seed={seed} exit={code}; consecutive={consecutive}/{max_failures}', flush=True)
        if retry < retries:
            retry += 1
        else:
            # Account for expert seeds already skipped by this simulator process.
            try:
                m = json.loads((temp_run / 'metrics.json').read_text())
                used = [int(r['seed']) for r in m.get('skipped_seeds', []) + m.get('episodes', [])]
                seed = max([seed] + used)
            except (OSError, ValueError, KeyError, TypeError): pass
            seed += 1
            retry = 0
    publish()
    archive(staging)
    if consecutive >= max_failures:
        print('Failure limit reached; task remains incomplete. Check server and attempt logs.', flush=True)
        break
    if len(records) < target and consecutive > 0:
        time.sleep(delay)
if len(records) != target:
    raise SystemExit(1)
print(f'TASK COMPLETE: {len(records)}/{target} episodes; infrastructure failures={len(failures)}', flush=True)
PY

#!/usr/bin/env bash
# Read-only completion check: OUTPUT_DIR TASK CONFIG CONTROL EXPECTED_EPISODES.
# Exit 0 = complete; 1 = retry needed.
# Checks JSON, float32 NPY length and MP4 container structure, not decoded video quality.
set -euo pipefail
exec "${PYTHON_BIN:-python}" - "$@" <<'PY'
import ast
import json
import math
import struct
import sys
from pathlib import Path

root, task, config, control, expected = sys.argv[1:]
root, expected = Path(root), int(expected)
run = root / 'runs' / task / config / control

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_json(path):
    with path.open() as stream:
        return json.load(stream)

def check_npy(path):
    with path.open('rb') as f:
        require(f.read(6) == b'\x93NUMPY', f'{path.name}: invalid NPY')
        version = f.read(2)
        require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'Unsupported NPY version')
        width = 2 if version[0] == 1 else 4
        length = int.from_bytes(f.read(width), 'little')
        require(0 < length < 65536, 'Invalid NPY header length')
        header = ast.literal_eval(f.read(length).decode('utf-8' if version[0] == 3 else 'latin1').strip())
        shape = header['shape']
        require(len(shape) == 2 and all(type(x) is int and x > 0 for x in shape), 'Invalid action shape')
        require(header['descr'] in ('<f4', '>f4', '=f4'), 'Expected float32 actions')
        require(path.stat().st_size == f.tell() + math.prod(shape) * 4, 'Truncated/invalid action array')

def check_mp4(path):
    size = path.stat().st_size
    boxes = set()
    with path.open('rb') as f:
        while f.tell() < size:
            start = f.tell()
            head = f.read(8)
            require(len(head) == 8, f'{path.name}: truncated MP4 header')
            length, kind = struct.unpack('>I4s', head)
            header_size = 8
            if length == 1:
                extended = f.read(8)
                require(len(extended) == 8, 'Truncated extended MP4 header')
                length, header_size = int.from_bytes(extended, 'big'), 16
            elif length == 0:
                length = size - start
            require(length >= header_size and start + length <= size, f'{path.name}: truncated MP4 box')
            boxes.add(kind)
            f.seek(start + length)
    require({b'ftyp', b'moov', b'mdat'} <= boxes, f'{path.name}: missing MP4 metadata/payload')

try:
    require((root / 'logs' / task / config / 'exitcode').read_text().strip() == '0', 'Previous process did not exit successfully')
    metrics = read_json(run / 'metrics.json')
    require((metrics['task'], metrics['task_config'], metrics['control_mode']) == (task, config, control), 'Metrics identity mismatch')
    episodes = metrics['episodes']
    require(len(episodes) == expected, f'Expected {expected} episodes, found {len(episodes)}')
    require(all(e['status'] == 'finished' and type(e['success']) is bool and type(e['seed']) is int for e in episodes), 'Unfinished/invalid episode records')
    require(len({e['seed'] for e in episodes}) == expected, 'Duplicate seeds')
    for episode in episodes:
        folder = run / f"seed{episode['seed']}"
        info = read_json(folder / 'info.json')
        require(isinstance(info, dict), 'Invalid episode info')
        args = info.get('client_args')
        require(isinstance(args, dict) and all(k in args for k in ('robotwin_root', 'state_source', 'seed', 'max_steps', 'instruction_type')), 'Incomplete client_args')
        require(isinstance(args['robotwin_root'], str) and bool(args['robotwin_root']) and args['state_source'] in ('command', 'measured'), 'Invalid client settings')
        require(type(args['seed']) is int and (args['max_steps'] is None or (type(args['max_steps']) is int and args['max_steps'] > 0)), 'Invalid seed/max_steps')
        require(args['instruction_type'] in (None, 'seen', 'unseen'), 'Invalid instruction_type')
        require((info['task'], info['task_config'], info['control_mode'], info['seed'], info['status'], info['success']) ==
                (task, config, control, episode['seed'], 'finished', episode['success']), 'Episode info mismatch')
        requests = read_json(folder / 'requests.json')
        require(isinstance(requests, list) and len(requests) > 0, 'Missing inference requests')
        require(isinstance(read_json(folder / 'structured_prompt.json'), (dict, list)), 'Invalid structured prompt')
        check_npy(folder / f'action_{control}.npy')
        check_mp4(folder / 'video.mp4')
        check_mp4(folder / 'prediction.mp4')
    print('complete')
except (OSError, ValueError, KeyError, TypeError, SyntaxError, OverflowError) as error:
    print(f'{type(error).__name__}: {error}')
    sys.exit(1)
PY

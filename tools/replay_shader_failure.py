r"""Replay one local MKW_SHADER_FAILURE_DUMP packet without the game or GPU.

Set MKW_SHADER_FAILURE_DUMP to an ignored build directory before running the game.
On a translator failure it retains <key>.request.json, .failed.code and .error.txt
in a newly claimed capture-<process>-<sequence> directory there.
The packet contains game-derived shader code/constants: keep all of it local.

  python tools/replay_shader_failure.py build/failures/<capture>/<key>.request.json \
      --translator build/xlate/totk_xlate.exe --out build/failure-replay

Reuses every recorded job option, rebinding only its code path and output folder.
Sets DMSH_STACK=1 in the child and saves its reply and stderr (including any stack)
in --out. Exit 0: translated; 1: translator failed; 2: invalid packet/process error.
Neither the packet nor the replay output should be attached to a public report.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def replay(request_path, translator, output, timeout):
    request_path, translator, output = request_path.resolve(), translator.resolve(), output.resolve()
    suffix = '.request.json'
    if not request_path.name.endswith(suffix):
        raise ValueError('expected a captured <64 hex digits>.request.json packet')
    key = request_path.name[:-len(suffix)]
    if not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('capture filename must contain a 64-digit lowercase hex key')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    job = request.get('job') if isinstance(request, dict) else None
    if not isinstance(job, dict) or job.get('key') != key:
        raise ValueError('recorded job key does not match the capture filename')
    code = request_path.with_name(key + '.failed.code')
    if not code.is_file():
        raise ValueError('captured sibling .failed.code file is missing; the original cache path is not used')
    if not translator.is_file():
        raise ValueError('translator executable does not exist')
    job['code'] = str(code)
    request['out'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['DMSH_STACK'] = '1'
    env['DOTNET_GENERATE_ASPNET_CERTIFICATE'] = 'false'
    run = subprocess.run([str(translator), '--serve'], input=json.dumps(request) + '\n',
                         capture_output=True, encoding='utf-8', errors='replace', timeout=timeout, env=env)
    (output / 'reply.txt').write_text(run.stdout, encoding='utf-8')
    (output / 'stderr.txt').write_text(run.stderr, encoding='utf-8')
    print(run.stdout, end='')
    print(run.stderr, end='', file=sys.stderr)
    if run.returncode:
        print(f'translator exited with {run.returncode}', file=sys.stderr)
        return 2
    return 0 if run.stdout.splitlines() == ['ok ' + key] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('request', type=Path)
    parser.add_argument('--translator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60, help='child timeout in seconds (default 60)')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        return replay(args.request, args.translator, args.out, args.timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'cannot replay shader failure: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

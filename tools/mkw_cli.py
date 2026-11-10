#!/usr/bin/env python3
"""Mario Kart World Windows recomp research driver (local game data only).

Stages: input audit -> local version lock -> ARM64 lift -> optional RomFS pack
-> incremental native build -> bounded boot experiment and diagnostic report.
This tool does not claim that a generated executable is playable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mkw_probe

BUILD_ID = 'mkw180'
TARGET_VERSION = '1.8.0'
UPDATE_BUILD_NUMBER = 917504
EXPECTED_TITLE = '0400C3F00006E000'
# Public 1.0.0 reference module IDs from kartnative's game-specific profile.
# NSO0 stores a 32-byte ID at header offset 0x40; the published value is 16 bytes.
# Different genuine revisions may have different IDs, so experimental overrides must be explicit.
KNOWN_BASE_MODULE_IDS = {
    'main': '9e244438c3d8ef1c703a6a85ef99c4f0',
    'subsdk0': 'd39a7a3d44530bb61b35690e5b0383d5',
    'sdk': '90ebba182862b9ecdc21fcb87b999012',
}


def nso_module_id(path: Path) -> str:
    with path.open('rb') as f:
        hdr = f.read(0x60)
    if len(hdr) < 0x60 or hdr[:4] != b'NSO0':
        raise ValueError(f'{path}: missing NSO0 header or module ID')
    return hdr[0x40:0x50].hex()


def check_module_ids(exefs: Path) -> tuple[bool, dict]:
    actual = {name: nso_module_id(exefs / name) for name in KNOWN_BASE_MODULE_IDS}
    return all(actual[name] == wanted for name, wanted in KNOWN_BASE_MODULE_IDS.items()), actual
MODULE_PATTERN = re.compile(r'^(?:main|sdk|subsdk\d+)$')
ERROR_PATTERNS = [
    ('game identity / loader', (r'wrong title', r'loader:', r'cannot read', r'NPDM', r'\bhash\b', r'module .*not loaded')),
    ('guest startup / constructors', (r'no DT_INIT', r'constructor', r'init::Start', r'not exported', r'fn ptr', r'could not resolve symbol')),
    ('Horizon / IPC', (r'unknown (?:svc|service)', r'ipc:', r'unsupported (?:svc|command)', r'kernel:', r'unknown port')),
    ('graphics / NVN', (r'NVN', r'vulkan', r'shader', r'pipeline', r'vk_', r'GPU')),
    ('memory / scheduling', (r'memory commit', r'virtual memory', r'allocation failed', r'deadlock', r'thread')),
]
MILESTONES = [
    ('module-load', [r'loader:.*loaded', r'ldr:.*loaded', r'modules linked']),
    ('main-thread', [r'boot: Switch2 NPDM main', r'boot: ENTER nn::init::Start']),
    ('guest-initialization', [r'boot: ENTER .* DT_INIT', r'boot: DT_INIT .*state=2']),
    ('first-nvn-call', [r'boot: Switch2 FIRST_NVNCALL']),
    ('first-present', [r'presented \(nvn calls', r'boot: frames \d+ presented']),
]


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            dig.update(block)
    return dig.hexdigest()


def add_common(ap):
    ap.add_argument('--repo', type=Path, required=True, help='mkw-recomp source checkout')
    ap.add_argument('--dump', type=Path, required=True, help='Extracted Mario Kart World update directory with exefs/ and romfs/')


def paths(args):
    repo = args.repo.resolve()
    exefs, romfs = mkw_probe.locate(args.dump.resolve())
    if not (repo / 'tools' / 'build_game.py').is_file() or not (repo / 'recomp' / 'gen.py').is_file():
        raise ValueError(f'{repo} is not an mkw-recomp source checkout')
    npdm = mkw_probe.npdm(exefs / 'main.npdm')
    if npdm['title_id'] != EXPECTED_TITLE:
        raise ValueError(f'Unexpected program ID {npdm["title_id"]}, need {EXPECTED_TITLE}')
    for required in ('main', 'sdk', 'subsdk0'):
        if not (exefs / required).is_file():
            raise ValueError(f'Missing MKW module {exefs / required}')
    names = [name for name in ['main'] + [f'subsdk{i}' for i in range(10)] + ['sdk']
             if (exefs / name).is_file()]
    return repo, exefs, romfs, names


def python_of(repo: Path) -> str:
    candidates = [repo / '.venv' / 'Scripts' / 'python.exe', repo / '.venv' / 'bin' / 'python']
    for f in candidates:
        if f.is_file():
            return str(f)
    return sys.executable


def local_hash_path(repo: Path) -> Path:
    return repo / 'build' / 'mkw-local' / 'expected-mkw180.json'


def create_lock(exefs: Path, dest: Path, attestation: bool, allow_unknown_ids=False) -> dict:
    if not attestation:
        raise ValueError('Update 1.8.0 is not proven by the program ID. Use --i-confirm-update-180 only when the supplied dump is the 1.8.0 update.')
    # There are no verified public 1.8.0 module IDs pinned in this devkit.
    # Reject a complete match to the known 1.0.0 IDs, but NEVER claim that
    # a non-match proves 1.8.0. SHA-256 values below are a LOCAL consistency lock.
    known_ids, base_ids = check_module_ids(exefs)
    if known_ids:
        raise ValueError('ExeFS matches the known base 1.0.0 module IDs. This is not evidence of a 1.8.0 update.')
    module_ids = {p.name: nso_module_id(p) for p in exefs.iterdir()
                  if p.is_file() and MODULE_PATTERN.fullmatch(p.name)}
    files = sorted([p for p in exefs.iterdir() if p.is_file() and not p.name.startswith('.')], key=lambda p: p.name)
    allowed = {'rtld', 'main.npdm'} | {n for n in [p.name for p in files] if MODULE_PATTERN.fullmatch(n)}
    unknown = [p.name for p in files if p.name not in allowed]
    if unknown:
        raise ValueError(f'Unexpected ExeFS files: {unknown}')
    expected = {
        '_comment': ('Mario Kart World LOCAL user-supplied standalone update claimed to be 1.8.0. '
                     'Version and completeness are NOT independently verified. SHA-256 locks input consistency only.'),
        'title_id': EXPECTED_TITLE,
        'display_version': TARGET_VERSION,
        'update_build_number_claimed': UPDATE_BUILD_NUMBER,
        'version_verified': False,
        'matches_known_base_100_ids': known_ids,
        'module_ids': module_ids,
        'mkw_version_attested_not_verified': True,
        'exefs_sha256': {p.name: sha256(p) for p in files},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        old = json.loads(dest.read_text(encoding='utf-8'))
        if old.get('exefs_sha256') != expected['exefs_sha256']:
            raise ValueError('Local version-lock hashes changed. Refusing overwrite; check game version or dump integrity.')
    else:
        dest.write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')
    return expected


def validate_lock(exefs: Path, lock: Path) -> None:
    if not lock.is_file():
        raise ValueError(f'Missing {lock}; run `prepare --i-confirm-update-180` first')
    expected = json.loads(lock.read_text(encoding='utf-8'))
    for name, wanted in expected['exefs_sha256'].items():
        file = exefs / name
        if not file.is_file() or sha256(file) != wanted:
            raise ValueError(f'Input changed from local version lock: {file}')


def command(argv: list[str], cwd: Path, dry_run=False) -> None:
    display = ' '.join('"'+a+'"' if ' ' in a else a for a in argv)
    print(f'[{cwd}] $ {display}', flush=True)
    if dry_run:
        return
    result = subprocess.run(argv, cwd=cwd, check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: {argv[1]}')


def do_prepare(args):
    repo, exefs, romfs, modules = paths(args)
    if romfs is None or not romfs.is_dir() or not any(romfs.iterdir()):
        raise ValueError('No non-empty RomFS directory found; an update-only dump must include all required game assets.')
    info = create_lock(exefs, local_hash_path(repo), args.i_confirm_update_180)
    output = {'program_id': EXPECTED_TITLE, 'version_claimed': TARGET_VERSION,
              'update_build_number_claimed': UPDATE_BUILD_NUMBER,
              'version_verified': False, 'module_ids': info['module_ids'],
              'exefs': str(exefs),
              'romfs': str(romfs) if romfs else None,
              'modules': modules, 'hashes': info['exefs_sha256']}
    report = repo / 'build' / 'mkw-local' / 'input.json'
    report.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(f'PREPARED: {report}\nLocal SHA-256 lock created. Claimed update: {TARGET_VERSION} (not independently verified). Modules: {", ".join(modules)}')


def do_xlate(args):
    repo = args.repo.resolve()
    source = repo/'tools'/'build_xlate.py'
    if not source.is_file():
        raise ValueError(f'Missing translator build script: {source}')
    # A child of the repository build tree, so NuGet discovery sees it for
    # build/xlate/xlate and build/xlate/ryujinx projects. No global NuGet edits.
    config = repo/'build'/'xlate'/'NuGet.Config'
    contents = ("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                "<configuration>\n  <packageSources>\n    <clear />\n"
                "    <add key=\"nuget.org\" value=\"https://api.nuget.org/v3/index.json\" />\n"
                "  </packageSources>\n  <auditSources>\n    <clear />\n"
                "    <add key=\"nuget.org\" value=\"https://api.nuget.org/v3/index.json\" />\n"
                "  </auditSources>\n</configuration>\n")
    if config.exists() and config.read_text(encoding='utf-8') != contents:
        raise ValueError(f'Refusing to overwrite a different local NuGet config: {config}')
    if args.dry_run:
        print(f'Would write project-local NuGet sources to: {config}')
    else:
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(contents, encoding='utf-8')
    command([python_of(repo), str(source)], repo, args.dry_run)


def do_generate(args):
    repo, exefs, _, modules = paths(args)
    validate_lock(exefs, local_hash_path(repo))
    cmd = [python_of(repo), str(repo/'recomp'/'gen.py'), '--exefs', str(exefs),
           '--out', str(repo/'generated'/BUILD_ID), '--modules', ','.join(modules)]
    if args.jobs:
        cmd += ['--jobs', str(args.jobs)]
    if args.limit:
        cmd += ['--limit', str(args.limit)]
        print('NOTE: --limit generates a partial CPU test, never a bootable game.', flush=True)
    command(cmd, repo, args.dry_run)
    if not args.dry_run:
        marker = repo/'generated'/BUILD_ID/'.mkw_partial'
        if args.limit:
            marker.write_text(f'Partial test generation with --limit {args.limit}; not a boot build.\n', encoding='utf-8')
        elif marker.exists():
            marker.unlink()
        if not args.limit:
            command([python_of(repo), str(repo/'tools'/'mkw_generated_overlays.py'),
                     '--repo', str(repo), '--exefs', str(exefs)], repo)


def do_pack(args):
    repo, _, romfs, _ = paths(args)
    if romfs is None:
        raise ValueError('No extracted RomFS found')
    packed = repo/'build'/'extracted'/BUILD_ID/'romfs'
    if (romfs/'romfs.bin').is_file() and (romfs/'index.json').is_file():
        print('Already packed. Use this folder with --romfs in build_game.py:')
        print(romfs)
        return
    if (packed/'romfs.bin').is_file() and (packed/'index.json').is_file() and not args.force:
        print(f'Reusing packed RomFS: {packed} (use --force to repack after changes)')
        return
    command([python_of(repo), str(repo/'tools'/'pack_romfs.py'), '--dir', str(romfs),
             '--out', str(packed)], repo, args.dry_run)


def chosen_romfs(repo: Path, romfs: Path | None) -> Path:
    selection = repo/'build'/'mkw-local'/'selected-romfs.json'
    if selection.is_file():
        spec = json.loads(selection.read_text(encoding='utf-8'))
        folder = Path(spec['path'])
        if not (folder/'index.json').is_file():
            raise ValueError(f'Selected RomFS disappeared: {folder}; rebuild')
        if spec['backend'] == 'mkw-loose-v1' and not (folder/'loose-root.txt').is_file():
            raise ValueError(f'Selected loose RomFS root missing: {folder}; rebuild')
        if spec['backend'] != 'mkw-loose-v1' and not (folder/'romfs.bin').is_file():
            raise ValueError(f'Selected packed RomFS image missing: {folder}; rebuild')
        return folder
    if romfs and (romfs/'index.json').is_file() and (romfs/'romfs.bin').is_file():
        return romfs
    # MKW builder's new zero-copy mode uses the native runtime's loose backend.
    loose = repo/'build'/'extracted'/BUILD_ID/'romfs-loose'
    if (loose/'index.json').is_file() and (loose/'loose-root.txt').is_file():
        doc = json.loads((loose/'index.json').read_text(encoding='utf-8'))
        if doc.get('backend') == 'mkw-loose-v1' and doc.get('status') == 'inventory-only':
            return loose
    p = repo/'build'/'extracted'/BUILD_ID/'romfs'
    if not (p/'index.json').is_file() or not (p/'romfs.bin').is_file():
        raise ValueError(f'RomFS not indexed at {loose} or packed at {p}; run builder first')
    return p


def do_build(args):
    repo, exefs, romfs, _ = paths(args)
    validate_lock(exefs, local_hash_path(repo))
    gen = repo/'generated'/BUILD_ID/'layout.json'
    if not gen.exists():
        raise ValueError(f'Missing {gen}; run generate first')
    if (gen.parent/'.mkw_partial').exists():
        raise ValueError('Generated code is from --limit partial test; rerun generate without --limit')
    command([python_of(repo), str(repo/'tools'/'mkw_generated_overlays.py'),
             '--repo', str(repo), '--exefs', str(exefs)], repo, args.dry_run)
    build_source = repo/'tools'/'build_game.py'
    # The runtime uses a nonzero rt_game_version for TotK-only constructor
    # and NVN return-address checks. MKW MUST have version 0 to use the
    # generic Switch 2 boot path. Earlier kits erroneously registered 180.
    if "'mkw180': '1.8.0'" in build_source.read_text(encoding='utf-8'):
        raise ValueError('Outdated mkw180=180 registration would activate TotK boot checks. Run tools/mkw_build_version.py --repo <repo> --apply')
    packed = chosen_romfs(repo, romfs)
    exe = repo/'build'/BUILD_ID/('mkw.exe' if os.name=='nt' else 'mkw')
    cmd = [python_of(repo), str(repo/'tools'/'build_game.py'), '--version', BUILD_ID,
           '--expected', str(local_hash_path(repo)), '--exefs', str(exefs),
           '--romfs', str(packed)]
    if args.jobs:
        cmd += ['--jobs', str(args.jobs)]
    if args.dry_run:
        print(f'Expected local executable: {exe}')
    command(cmd, repo, args.dry_run)
    if not args.dry_run:
        print(f'BUILT: {exe}\nNOTE: a successful link does not establish Mario Kart World boot compatibility.')


def analyze_text(log: str) -> dict:
    stages = {name: any(re.search(p, log, re.I) for p in patterns) for name, patterns in MILESTONES}
    fatals = [line.strip() for line in log.splitlines()
              if re.search(r'\bFATAL\b|\bfatal:|\bERROR\b|unsupported|not implemented', line, re.I)]
    categories = {group: [line for line in fatals if any(re.search(p, line, re.I) for p in patterns)]
                  for group, patterns in ERROR_PATTERNS}
    category = next((group for group, matches in categories.items() if matches), 'unclassified') if fatals else 'none'
    if stages['first-nvn-call']:
        level = 'M2 early graphics reached; NOT title-screen boot'
    elif stages['guest-initialization']:
        level = 'guest constructors started; boot not established'
    elif stages['main-thread']:
        level = 'main thread initialized; boot not established'
    elif stages['module-load']:
        level = 'modules linked; boot not established'
    else:
        level = 'no boot milestone proved'
    return {'stages': stages, 'boot_evidence': level, 'fatal_count': len(fatals),
            'first_fatals': fatals[:25], 'likely_category': category,
            'title_screen_seen': False, 'note': 'A text log cannot verify a rendered title screen.'}


def do_analyze(args):
    log = args.log.resolve()
    if not log.is_file():
        raise ValueError(f'Log missing: {log}')
    result = analyze_text(log.read_text(encoding='utf-8',errors='replace'))
    output = args.out or log.with_suffix('.summary.json')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'SUMMARY: {output}')


def do_run(args):
    repo, exefs, romfs, _ = paths(args)
    exe = repo/'build'/BUILD_ID/('mkw.exe' if os.name=='nt' else 'mkw')
    if not exe.is_file():
        raise ValueError(f'No executable found at {exe}; build first')
    lock = local_hash_path(repo)
    validate_lock(exefs, lock)
    packed = chosen_romfs(repo, romfs)
    mode = {'selftest': '--selftest', 'services': '--probe-services',
            'boot': '--boot', 'game': None, 'first-frame': None}[args.mode]
    cmd = [str(exe), '--exefs', str(exefs), '--romfs', str(packed),
           '--profile', str(repo/'build'/'profile-mkw180')]
    if mode:
        cmd.append(mode)
    env = os.environ.copy()
    env['MKW_BOOT_TRACE'] = '1'
    # An earlier diagnostic run must not leak its first-NVN stop into a game run.
    env.pop('MKW_STOP_ON_NVN', None)
    if args.mode == 'boot':
        env['MKW_STOP_ON_NVN'] = '1'
    env['TOTK_NATIVE_GS'] = '1'
    env['TOTK_ASTC'] = 'gpu'
    env['TOTK_ASTC_BC'] = '1'
    env['TOTK_MEMORY'] = 'auto'
    # Memory pool from the base 1.0.0 research is NOT imposed on the unverified 1.8.0 update.
    # Allow explicit testing when users have measured the update's requirements.
    if args.app_pool:
        env['TOTK_APP_POOL'] = args.app_pool
    if args.gpu:
        env['TOTK_GPU'] = args.gpu
    env['TOTK_TRACE'] = args.trace
    report_dir = repo/'build'/'mkw-local'/'runs'
    report_dir.mkdir(parents=True, exist_ok=True)
    label = time.strftime('%Y%m%d_%H%M%S')+'_'+args.mode
    log = report_dir/(label+'.log')
    shot_dir = report_dir/(label+'_guest_frames')
    if args.mode == 'first-frame':
        shot_dir.mkdir(parents=True, exist_ok=False)
        # These screenshots come from rt_frame_presented, not a fake host splash.
        # Use several frames because some runtimes count from zero.
        env['TOTK_SHOTS'] = '0-8:1'
        env['TOTK_SHOTS_DIR'] = str(shot_dir)
    print('RUN:', ' '.join(cmd), '\nLOG:', log, flush=True)
    t0 = time.monotonic()
    timed_out = False
    captured = None
    stopped_after_capture = False
    # Stream potentially huge host traces to a file, not to process memory.
    with log.open('w', encoding='utf-8') as output:
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
        if args.mode == 'first-frame':
            # A real guest-presented image may take >80s to appear.
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline and proc.poll() is None:
                candidates = sorted(shot_dir.glob('f*.png'))
                for file in candidates:
                    if file.stat().st_size >= 64:
                        with file.open('rb') as png:
                            start = png.read(8)
                            png.seek(-12, os.SEEK_END)
                            end = png.read(12)
                        if start == b'\x89PNG\r\n\x1a\n' and end[4:8] == b'IEND':
                            captured = file
                            break
                if captured:
                    # Diagnostic-only capture: don't report a successful game exit.
                    time.sleep(0.5)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    stopped_after_capture = True
                    break
                time.sleep(0.25)
            if proc.poll() is None:
                timed_out = True
                proc.kill()
                proc.wait()
                rc = None
            else:
                rc = proc.returncode
        else:
            try:
                rc = proc.wait(timeout=None if args.timeout <= 0 else args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()
                rc = None
    # Avoid loading a multi-gigabyte trace into memory. Keep head + tail.
    with log.open('rb') as stream:
        beginning = stream.read(2 << 20)
        if log.stat().st_size > 5 << 20:
            stream.seek(-3 << 20, 2)
            ending = stream.read(3 << 20)
            data = (beginning + b'\n[... log truncated for summary ...]\n' + ending).decode('utf-8', 'replace')
        else:
            data = (beginning + stream.read()).decode('utf-8', 'replace')
    # The game's own logfile might contain diagnostics not sent to the console.
    game_log = repo/'build'/BUILD_ID/'logs'/'totk.log'
    if game_log.is_file() and game_log.stat().st_mtime >= time.time() - (time.monotonic() - t0) - 5:
        with game_log.open('rb') as g:
            if game_log.stat().st_size > 3 << 20:
                g.seek(-3 << 20, 2)
            data += '\n[game log]\n' + g.read().decode('utf-8', 'replace')
    report = analyze_text(data)
    report.update({'exit_code': rc, 'timed_out': timed_out, 'elapsed_s': round(time.monotonic()-t0,2),
                   'mode': args.mode, 'log': str(log),
                   'first_guest_frame_png': str(captured) if captured else None,
                   'first_guest_frame_captured': bool(captured),
                   'stopped_for_diagnostic': stopped_after_capture})
    log.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n'.join(data.splitlines()[-45:]))
    print(f'\nRESULT: {report["boot_evidence"]}; first fatal: {(report["first_fatals"] or ["none"])[0]}')
    if captured:
        print(f'ACTUAL GUEST FRAME SAVED: {captured}')
        print('This proves one frame was presented, not that a title/loading screen is correct.')
    print('Timing:', 'timeout' if timed_out else f'exit code {rc}', 'in', report['elapsed_s'],'seconds')
    if timed_out:
        print('TIMEOUT: not automatically a successful boot, inspect the game log.')
    return 0 if (captured and stopped_after_capture) or (rc == 0 and not timed_out) else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare', help='Confirm identity and pin hashes of a locally supplied dump')
    add_common(p)
    p.add_argument('--i-confirm-update-180', action='store_true', help='Attest input is a 1.8.0 update; does not establish dump completeness')
    p.set_defaults(fn=do_prepare)
    p = sub.add_parser('xlate', help='Build shader translator with project-local nuget.org config')
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=do_xlate)
    p = sub.add_parser('generate', help='Lift local ARM64 modules using the gen.py of this checkout')
    add_common(p)
    p.add_argument('--jobs', type=int)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=do_generate)
    p = sub.add_parser('pack', help='Create the packed RomFS image, unless already present')
    add_common(p)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=do_pack)
    p = sub.add_parser('build', help='Build local Windows executable incrementally')
    add_common(p)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=do_build)
    p = sub.add_parser('run', help='Bounded host/boot experiment, never claims title screen')
    add_common(p)
    p.add_argument('--mode', choices=['selftest','services','boot','first-frame','game'], default='boot')
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--gpu', choices=['null','vulkan'], default=None)
    p.add_argument('--trace', default='ipc,svc')
    p.add_argument('--app-pool', help='Optional research override of TOTK_APP_POOL for this child process (e.g. 0x...)')
    p.set_defaults(fn=do_run)
    p = sub.add_parser('analyze-log', help='Summarize failure stage of an existing Windows log')
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.set_defaults(fn=do_analyze)
    args = parser.parse_args(argv)
    try:
        ret = args.fn(args)
        return ret if isinstance(ret, int) else 0
    except (ValueError, OSError, RuntimeError) as e:
        print('ERROR:', e, file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())

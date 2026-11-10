#!/usr/bin/env python3
"""Mario Kart World recompilation input audit (no game assets or keys included).

Read-only input validation for a decrypted ExeFS/RomFS dump. Generates an
informational JSON report. It does NOT recompile, extract, decrypt, or certify
that a dump is version 1.0.0. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_TITLE = 0x0400C3F00006E000
EXPECTED_NAME = 'Mario Kart World'
MODULE = re.compile(r'^(?:main|sdk|rtld|subsdk[0-9]+)$')


def byte_size(size: int) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            return f'{size:.1f} {unit}'
        size /= 1024
    return '?'


def digest(path: Path, bufsize: int = 8 * 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(bufsize), b''):
            sha.update(chunk)
    return sha.hexdigest()


def npdm(path: Path) -> dict:
    with path.open('rb') as f:
        # ACID may be larger, but the ACI0 header only needs 0x40 bytes.
        header = f.read(0x80)
        file_size = path.stat().st_size
        if len(header) < 0x80 or header[:4] != b'META':
            raise ValueError('main.npdm does not begin with a valid META header')
        aci_off, aci_size = struct.unpack_from('<II', header, 0x70)
        if aci_off < 0x80 or aci_size < 0x40 or aci_off + aci_size > file_size:
            raise ValueError('invalid ACI0 offset/size in main.npdm')
        f.seek(aci_off)
        aci = f.read(0x40)
        if aci[:4] != b'ACI0':
            raise ValueError('main.npdm points to a missing ACI0 block')
        title_id = struct.unpack_from('<Q', aci, 0x10)[0]
        name = header[0x20:0x30].split(b'\x00', 1)[0].decode('ascii', 'replace')
        return {
            'magic': 'META', 'aci0_magic': 'ACI0',
            'title_id': f'{title_id:016X}',
            'is_mario_kart_world': title_id == EXPECTED_TITLE,
            'process_name': name, 'npdm_meta_version': struct.unpack_from('<I', header, 0x18)[0],
            'main_thread_core': header[0x0F],
        }


def locate(dump: Path) -> tuple[Path, Path | None]:
    if dump.is_file():
        if dump.suffix.lower() == '.nup':
            raise ValueError('NUP container supplied. Extract your own game into exefs/ and romfs/ first; this tool is an audit, not a NUP decryptor.')
        raise ValueError('Expected a directory containing exefs/ and romfs/, not a file')
    # Prioritize a complete standalone update over a stale/empty base folder.
    # Some NUP extractors create empty base/exefs when only update content exists.
    candidates = [
        dump / 'updates' / '917504',
        dump / '917504',
        dump,
        dump / 'base',
    ]
    if dump.name.lower() == 'exefs':
        candidates.insert(0, dump.parent)
    for root in candidates:
        exefs = root / 'exefs'
        if (exefs / 'main.npdm').is_file() and (exefs / 'main').is_file():
            romfs = root / 'romfs'
            return exefs, romfs if romfs.is_dir() else None
    raise ValueError('Could not locate exefs/; try --dump H:\\Switch2\\MarioKartWorld\\updates\\917504')


def inspect_exefs(exefs: Path, hash_files: bool) -> tuple[list[dict], list[str]]:
    errors = []
    result = []
    for filename in ('main.npdm', 'main', 'sdk'):
        if not (exefs / filename).is_file():
            errors.append(f'Missing required ExeFS file: {filename}')
    for p in sorted(exefs.iterdir(), key=lambda p: p.name):
        if not p.is_file() or (p.name != 'main.npdm' and not MODULE.fullmatch(p.name)):
            continue
        st = p.stat()
        info = {'name': p.name, 'bytes': st.st_size}
        if p.name != 'main.npdm':
            with p.open('rb') as f:
                magic = f.read(4)
            # This is a warning only: Switch2 variants have evolved.
            info['magic_hex'] = magic.hex()
            info['nso0_magic'] = magic == b'NSO0'
            if not info['nso0_magic']:
                errors.append(f'{p.name} does not start with NSO0; may be unsupported or damaged')
        if hash_files:
            info['sha256'] = digest(p)
        result.append(info)
    return result, errors


def scan_romfs(romfs: Path | None, enumerate_files: bool) -> dict:
    if romfs is None:
        return {'present': False, 'scanned': False}
    result = {'present': True, 'path': str(romfs), 'scanned': enumerate_files}
    if enumerate_files:
        count = total = 0
        for parent, dirs, names in os.walk(romfs):
            dirs.sort()
            for name in names:
                if name.startswith('.'):
                    continue
                p = Path(parent, name)
                if p.is_file():
                    count += 1
                    total += p.stat().st_size
        result.update({'file_count': count, 'total_file_bytes': total})
    return result


def run_probe(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    dump = args.dump.expanduser().resolve()
    try:
        exefs, romfs = locate(dump)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    errors: list[str] = []
    npdm_record: dict | None = None
    try:
        npdm_record = npdm(exefs / 'main.npdm')
        if not npdm_record['is_mario_kart_world']:
            errors.append(f"Wrong title ID: {npdm_record['title_id']} (need {EXPECTED_TITLE:016X})")
    except (ValueError, OSError) as e:
        errors.append(str(e))
    modules, module_errors = inspect_exefs(exefs, args.sha256)
    errors.extend(module_errors)
    asset_info = scan_romfs(romfs, args.scan_romfs)
    if not asset_info['present']:
        errors.append('Missing RomFS folder')
    report = {
        'tool': 'mkw_probe', 'schema_version': 1,
        'game': EXPECTED_NAME, 'target_program_id': f'{EXPECTED_TITLE:016X}',
        'target_version_claimed': '1.8.0',
        'version_verified': False,
        'version_note': 'The NPDM title ID does not prove the update version or completeness. Current 1.8.0 module IDs are not independently pinned in this kit.',
        'exefs': str(exefs), 'npdm': npdm_record, 'modules': modules,
        'romfs': asset_info, 'errors': errors, 'input_layout_valid': len(errors) == 0,
        'seconds': round(time.monotonic() - t0, 3),
        'next_step': 'Establish version-specific module hashes and a game-specific code generation profile. This probe does not produce a playable EXE.'
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Game:       {EXPECTED_NAME}')
    print(f'ExeFS:      {exefs}')
    print(f'RomFS:      {romfs or "MISSING"}')
    print(f'Program ID: {npdm_record["title_id"] if npdm_record else "unavailable"}')
    print('Modules:    ' + ', '.join(x['name'] for x in modules))
    for m in modules:
        print(f"  {m['name']:12s} {byte_size(m['bytes']):>12s}" + (f" sha256={m['sha256']}" if 'sha256' in m else ''))
    if asset_info.get('scanned'):
        print(f"RomFS:      {asset_info['file_count']:,} files, {byte_size(asset_info['total_file_bytes'])}")
    print('Version:    not verified (hash baseline needed)')
    print(f'Report:     {args.report}')
    if errors:
        print('INPUT PROBLEMS:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 2
    print('PASS: game identity and extracted directory layout. This is NOT a recompilation or full data integrity test.')
    return 0


def run_tools(_: argparse.Namespace) -> int:
    candidates = [
        ('Python', [sys.executable, '--version']),
        ('Git', ['git', '--version']),
        ('.NET 10+', ['dotnet', '--list-sdks']),
        ('MSYS2 Clang', [r'C:\msys64\ucrt64\bin\clang.exe', '--version']),
        ('MSYS2 Ninja', [r'C:\msys64\ucrt64\bin\ninja.exe', '--version']),
    ]
    missing = False
    for label, cmd in candidates:
        found = bool(shutil.which(cmd[0]) or Path(cmd[0]).is_file())
        if not found:
            print(f'[MISSING] {label}: {cmd[0]}')
            missing = True
            continue
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines = (res.stdout + res.stderr).splitlines()
            msg = lines[0] if lines else '(no output)'
            if label == '.NET 10+':
                found = any(x.strip().startswith('10.') for x in lines)
            print(f'[{"PASS" if found else "MISSING"}] {label}: {msg}')
            if not found:
                missing = True
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f'[ERROR] {label}: {e}')
            missing = True
    return 2 if missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('probe', help='Read-only Mario Kart World ExeFS/RomFS validation')
    p.add_argument('--dump', required=True, type=Path, help='Game root, updates/917504/, or exefs/ directory')
    p.add_argument('--report', type=Path, default=Path('mkw-input-report.json'))
    p.add_argument('--sha256', action='store_true', help='Compute full hashes for each ExeFS module; slower')
    p.add_argument('--scan-romfs', action='store_true', help='Count RomFS files and their sizes; slower on HDD')
    p.set_defaults(fn=run_probe)
    t = sub.add_parser('tools', help='Check prerequisites without installing/changing anything')
    t.set_defaults(fn=run_tools)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the required MKW 1.8.0 heap-observer wrappers to freshly lifted C."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MARKER = 'MKW_GENERATED_OVERLAYS_M15_26'
MAIN_SHA256 = '29040941362bf30175134a580a06fff7ffb8bb27e9cfed60d9ace9e9e2d982fe'
SDK_SHA256 = 'acdc4a29bb809955d6a680f3f8515e0bd1d6eb1028e6a38182f00dea08c309bd'
FUNCTIONS = {
    0x70003E3550: 'primary_alloc',
    0x70001A58A0: 'dealloc',
    0x70003F24C0: 'region_vfunc',
    0x70003F17A0: 'region_direct',
    0x7000805830: 'region_string',
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verify_exefs(exefs: Path) -> None:
    for name, expected in (('main', MAIN_SHA256), ('sdk', SDK_SHA256)):
        path = exefs / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'{name}: expected the audited MKW 1.8.0 module')


def definition(address: int) -> re.Pattern[str]:
    return re.compile(r'\bvoid\s+f_' + f'{address:x}' + r'\s*\(Cpu\s*\*c\)\s*\{')


def locate(repo: Path) -> dict[Path, list[tuple[int, str]]]:
    root = repo / 'generated' / 'mkw180' / 'main'
    if not root.is_dir():
        raise ValueError(f'generated main module is missing: {root}')
    result: dict[Path, list[tuple[int, str]]] = {}
    found = {address: [] for address in FUNCTIONS}
    patterns = {address: definition(address) for address in FUNCTIONS}
    for path in root.glob('main_*.c'):
        source = path.read_text(encoding='utf-8')
        for address, pattern in patterns.items():
            if pattern.search(source):
                found[address].append(path)
    for address, matches in found.items():
        if len(matches) != 1:
            raise ValueError(f'expected one lifted definition for {address:#x}; found {len(matches)}')
        result.setdefault(matches[0], []).append((address, FUNCTIONS[address]))
    return result


def wrapper(address: int, kind: str, original: str) -> str:
    if kind == 'primary_alloc':
        body = '''
    uint64_t mkw_size = c->x[0];
    uint64_t mkw_owner = c->x[1];
    uint64_t mkw_align = c->x[2];
    ORIGINAL(c);
    mkw_heap_owner_record(c->x[0], mkw_owner, mkw_size, mkw_align);'''
    elif kind == 'dealloc':
        body = '''
    uint64_t mkw_pointer = c->x[1];
    mkw_heap_owner_discard(mkw_pointer);
    ORIGINAL(c);'''
    else:
        if kind == 'region_vfunc':
            requested, owner, size = 1, 0, 'mkw_requested'
        elif kind == 'region_direct':
            requested, owner, size = 0, 1, 'mkw_requested'
        elif kind == 'region_string':
            requested, owner, size = 0, 1, '(mkw_requested <= UINT64_MAX - 16u ? mkw_requested + 16u : mkw_requested)'
        else:
            raise ValueError('unknown generated overlay kind')
        body = f'''
    uint64_t mkw_requested = c->x[{requested}];
    uint64_t mkw_owner = c->x[{owner}];
    int32_t mkw_signed_align = (int32_t)c->x[2];
    uint64_t mkw_align = mkw_signed_align < 0
        ? (uint64_t)(-(int64_t)mkw_signed_align)
        : (uint64_t)mkw_signed_align;
    ORIGINAL(c);
    mkw_heap_owner_record(c->x[0], mkw_owner, {size}, mkw_align);'''
    body = body.replace('ORIGINAL', original)
    return f'''\n/* {MARKER}: observe the unmodified lifted allocator. */
void f_{address:x}(Cpu *c) {{{body}
}}
'''


def patch_chunk(source: str, entries: list[tuple[int, str]]) -> str:
    if MARKER in source:
        raise ValueError('partially or previously overlaid generated source')
    declarations = ('#include "cpu.h"\n'
                    'extern void mkw_heap_owner_record(uint64_t, uint64_t, uint64_t, uint64_t);\n'
                    'extern void mkw_heap_owner_discard(uint64_t);')
    if source.count('#include "cpu.h"') != 1:
        raise ValueError('unexpected generated source header')
    source = source.replace('#include "cpu.h"', declarations, 1)
    for address, kind in entries:
        pattern = definition(address)
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError(f'lifted function {address:#x} is missing or ambiguous')
        match = matches[0]
        original = f'mkw_original_{kind}_{address:x}'
        renamed = match.group(0).replace('void', 'static void', 1).replace(f'f_{address:x}', original, 1)
        source = source[:match.start()] + renamed + source[match.end():]
        source += wrapper(address, kind, original)
    return source


def apply(repo: Path, exefs: Path) -> dict:
    verify_exefs(exefs)
    state = repo / 'build' / 'mkw-local' / 'generated-overlays.json'
    if state.is_file():
        record = json.loads(state.read_text(encoding='utf-8'))
        for name, expected in record.get('files', {}).items():
            path = repo / name
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'generated overlay changed after installation: {name}')
        return record
    located = locate(repo)
    outputs = {path: patch_chunk(path.read_text(encoding='utf-8'), entries)
               for path, entries in located.items()}
    for path, output in outputs.items():
        path.write_text(output, encoding='utf-8', newline='\n')
    record = {
        'marker': MARKER,
        'files': {path.relative_to(repo).as_posix(): digest(path) for path in outputs},
        'functions': [f'{address:#x}' for address in FUNCTIONS],
    }
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--exefs', type=Path, required=True)
    args = parser.parse_args()
    try:
        record = apply(args.repo.resolve(), args.exefs.resolve())
        print(f'{MARKER}: {len(record["functions"])} functions in {len(record["files"])} chunks')
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

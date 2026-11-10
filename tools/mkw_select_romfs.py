#!/usr/bin/env python3
"""Select an existing user-created packed RomFS without copying or hashing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    packed = args.packed.resolve()
    image = packed / 'romfs.bin'
    index = packed / 'index.json'
    if not image.is_file() or not index.is_file():
        parser.error(f'{packed} must contain romfs.bin and index.json')
    manifest = json.loads(index.read_text(encoding='utf-8'))
    if manifest.get('status') != 'pass' or manifest.get('romfs_size') != image.stat().st_size:
        parser.error('packed RomFS index is incomplete or does not match the image size')
    output = repo / 'build' / 'mkw-local' / 'selected-romfs.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'path': str(packed), 'backend': 'packed'}, indent=2) + '\n', encoding='utf-8')
    print(f'Selected existing packed RomFS: {packed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

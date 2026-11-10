"""Pack an extracted RomFS directory into the flat image and index.json the runtime reads (runtime/romfs.c).

Dumps of Switch 2 games often come as a folder of loose files. The runtime only needs one image file and the offset
and size of each file in it (tools/romfs_index.py), so the files are concatenated in path order, each aligned to 16
bytes. Dot files the dump tool leaves (".listing") are skipped. The output is game data: it goes under build/.

    .venv\\Scripts\\python tools\\mkw_pack_romfs.py --dir <dump>\\romfs --out build\\extracted\\mkw180\\romfs

Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ALIGN = 16
CHUNK = 8 << 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', type=Path, required=True, help='extracted RomFS root')
    ap.add_argument('--out', type=Path, required=True, help='folder for romfs.bin and index.json')
    args = ap.parse_args()
    root = args.dir.resolve()
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.startswith('.'):
                p = Path(dirpath) / name
                files.append((p.relative_to(root).as_posix(), p))
    files.sort(key=lambda e: e[0].encode())
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / 'romfs.bin'
    sha = hashlib.sha256()
    entries, off, t0 = [], 0, time.time()
    with open(image, 'wb') as dst:
        for i, (rel, p) in enumerate(files):
            pad = -off % ALIGN
            if pad:
                dst.write(b'\0' * pad)
                sha.update(b'\0' * pad)
                off += pad
            size = 0
            with open(p, 'rb') as src:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
            entries.append({'path': rel, 'offset': off, 'size': size})
            off += size
    total_bytes = off
    index = {
        'format_version': 1,
        'image_sha256': sha.hexdigest(),
        'image_size': total_bytes,
        'files': entries,
    }
    index_path = args.out / 'index.json'
    index_path.write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    dt = time.time() - t0
    rate = (total_bytes / (1 << 20)) / dt if dt > 0 else 0
    print(f'mkw_pack_romfs: {len(entries)} files, {total_bytes / (1 << 20):.1f} MiB in {dt:.1f}s ({rate:.1f} MiB/s) -> {args.out}')


if __name__ == '__main__':
    main()

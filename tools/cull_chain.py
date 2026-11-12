# Follows MKW's GPU culling chain through an MKW_DUMP_CS capture (before / after snapshots):
#   14293e2831001e91 (0006): counts visible objects per (type, LOD) slot: c0+0x60 buffer, 8 bytes per slot (+4 count)
#   104bd69caf1f4c09 (0007): turns slot counts into draw commands (20-byte VkDrawIndexedIndirectCommand)
#   e8dae91b1570941b (0008): writes each visible object's 80-byte instance record
# Prints, per buffer the dispatch wrote, the words that changed, so the counts and commands can be lined up.
# Usage: python tools/cull_chain.py <dump dir>
import glob
import os
import struct
import sys

d = sys.argv[1]


def load(base):
    c0 = open(base + '_c0.bin', 'rb').read()
    files = {}
    for f in glob.glob(base + '_*.bin'):
        tag = f[len(base) + 1:-4]
        if tag == 'c0' or tag.startswith('post'):
            continue
        k, a = tag.split('_')
        files[k] = (int(a, 16), open(f, 'rb').read())
    return c0, files


for pre in sorted(glob.glob(os.path.join(d, 'cs*_c0.bin'))):
    if '_post_' in pre:
        continue
    base = pre[:-len('_c0.bin')]
    post = base + '_post'
    if not os.path.exists(post + '_c0.bin'):
        continue
    c0, a = load(base)
    _, b = load(post)
    name = os.path.basename(base)
    print(f'== {name}')
    for off in range(0x10, 0x100, 0x10):
        addr, size = struct.unpack_from('<QI', c0, off)
        if not addr:
            continue
        print(f'   c0+{off:#x}: {addr:#x}+{size:#x}', end='')
        k = next((k for k, (fa, _) in a.items() if fa == addr and k[0] == 's'), None)
        if k and k in b:
            x, y = a[k][1], b[k][1]
            n = min(len(x), len(y)) // 4
            ch = [i for i in range(n) if x[4 * i:4 * i + 4] != y[4 * i:4 * i + 4]]
            print(f' ({k}) {len(ch)} words changed', end='')
            if ch and len(ch) <= 400:
                print()
                for i in ch[:60]:
                    print(f'      +{4 * i:#07x}: {struct.unpack_from("<I", x, 4 * i)[0]:#010x} -> {struct.unpack_from("<I", y, 4 * i)[0]:#010x}')
                continue
        print()

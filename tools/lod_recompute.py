# Recomputes MKW's per-object LOD choice (compute program 14293e2831001e91, the culling / LOD selection pass) from an
# MKW_DUMP_CS snapshot, following the translated shader step by step: frustum planes c3[0..3], camera c3[4], LOD bias
# c3[5].z, object records (96 bytes: position + radius at +48, type +72, LOD halves +80, scale +92) at c0+0x30, per-type records (20 bytes) at c0+0x40, the per-type LOD scale and
# maximum distance in c4 (16 bytes per type). Prints the objects nearest a point (or the camera) and flags anomalies.
# Usage: python tools/lod_recompute.py <dump dir> [x y z]
import glob
import math
import os
import struct
import sys

d = sys.argv[1]
c0s = sorted(glob.glob(os.path.join(d, 'cs*_14293e2831001e91_c0.bin')))
if not c0s:
    sys.exit('no snapshot of 14293e2831001e91 in ' + d)


def half(h):
    return struct.unpack('<e', struct.pack('<H', h & 0xffff))[0]


for c0p in c0s:
    base = c0p[:-len('_c0.bin')]
    c0 = open(c0p, 'rb').read()
    files = {}
    for f in glob.glob(base + '_*.bin'):
        tag = f[len(base) + 1:-4]
        if tag == 'c0':
            continue
        kind_slot, addr = tag.split('_')
        files[kind_slot] = (int(addr, 16), open(f, 'rb').read())

    def table(off):
        a, n = struct.unpack_from('<QI', c0, off)
        for k, (fa, data) in files.items():
            if k[0] == 's' and fa == a:
                return a, n, data
        return a, n, None

    oa, on, objs = table(0x30)
    ta, tn, types = table(0x40)
    c3 = files.get('u0', (0, b''))[1]
    c4 = files.get('u1', (0, b''))[1]
    print(f'== {os.path.basename(base)}: objects {oa:#x}+{on:#x} ({on // 96}), types {ta:#x}+{tn:#x}, '
          f'c3 {len(c3)} B, c4 {len(c4)} B')
    if objs is None or types is None or len(c3) < 0x60:
        print('   missing buffers')
        continue
    v = lambda i: struct.unpack_from('<4f', c3, 16 * i)
    planes = [v(i) for i in range(4)]
    cam = v(4)[:3]
    bias = struct.unpack_from('<i', c3, 16 * 5 + 8)[0]
    print(f'   camera {cam[0]:.1f} {cam[1]:.1f} {cam[2]:.1f}  lod bias {bias}')
    ref = tuple(map(float, sys.argv[2:5])) if len(sys.argv) >= 5 else cam
    rows, anomalies = [], {}
    for i in range(min(on, len(objs)) // 96):
        r = objs[96 * i:96 * i + 96]
        x, y, z, rad = struct.unpack_from('<4f', r, 48)
        typ = struct.unpack_from('<I', r, 72)[0]
        hA, hB = struct.unpack_from('<II', r, 80)
        scale = struct.unpack_from('<f', r, 92)[0]
        dot = lambda p: p[0] * x + p[1] * y + p[2] * z + p[3]
        inside = dot(planes[3]) < rad and dot(planes[2]) < rad and dot(planes[0]) < rad and not dot(planes[1]) >= rad
        if typ * 20 + 20 > len(types):
            anomalies['type index past the type table'] = anomalies.get('type index past the type table', 0) + 1
            continue
        t0, = struct.unpack_from('<I', types, typ * 20)
        nlod, = struct.unpack_from('<I', types, typ * 20 + 16)
        ci = (t0 >> 16) << 4
        mul = struct.unpack_from('<f', c4, ci)[0] if ci + 8 <= len(c4) else 0.0
        maxd = struct.unpack_from('<f', c4, ci + 4)[0] if ci + 8 <= len(c4) else 0.0
        if ci + 8 > len(c4):
            anomalies['c4 index past the bound c4'] = anomalies.get('c4 index past the bound c4', 0) + 1
        d2 = sum((a - b) ** 2 for a, b in zip(cam, (x, y, z)))
        th = [(h * mul) ** 2 for h in (half(hA), half(hA >> 16), half(hB), half(hB >> 16))]
        lod = sum(1 for t in th if d2 >= t) + bias
        lod = max(0, min(lod, max(nlod, 1) - 1))
        culled = (not inside) or d2 > (scale * maxd) ** 2
        if mul == 0.0:
            anomalies['zero LOD scale'] = anomalies.get('zero LOD scale', 0) + 1
        dref = math.dist(ref, (x, y, z))
        rows.append((dref, i, x, y, z, typ, nlod, lod, math.sqrt(d2), [math.sqrt(t) for t in th], mul, maxd, scale, culled))
    rows.sort()
    for row in rows[:12]:
        dref, i, x, y, z, typ, nlod, lod, dist, ths, mul, maxd, scale, culled = row
        print(f'   obj {i:5} at {x:8.1f} {y:7.1f} {z:8.1f} type {typ:4} dist {dist:7.1f} lod {lod}/{nlod} '
              f'thresholds {" ".join(f"{t:.0f}" for t in ths)} (scale {mul:.3g}, max {maxd:.0f}, obj {scale:.2f})'
              f'{" culled" if culled else ""}')
    print('   anomalies:', anomalies or 'none', f'({len(rows)} objects)')

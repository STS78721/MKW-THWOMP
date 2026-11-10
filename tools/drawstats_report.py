# Summarises MKW_DRAWSTATS lines (runtime/vk_drawstats.c) from a log: per pass number, how the draw count,
# fingerprint, mesh draws and GPU indirect counts change from frame to frame. A "blink" is a value that differs from
# both its neighbours while the neighbours agree (A B A), the signature of something drawn every other frame.
# Usage: python tools/drawstats_report.py <host.log> [--pass N]
import re
import sys
from collections import defaultdict

log = sys.argv[1]
only = int(sys.argv[sys.argv.index('--pass') + 1]) if '--pass' in sys.argv else None
stat = defaultdict(dict)    # pass -> frame -> (draws, fp, mesh)
cnt = defaultdict(dict)     # pass -> frame -> tuple(counts)
for line in open(log, errors='replace'):
    m = re.search(r'dstat f(\d+) p(\d+) (\d+)x(\d+) draws (\d+) pipes (\d+) fp ([0-9a-f]+) icd (\d+) mesh (\d+)/(\d+)', line)
    if m:
        f, p = int(m.group(1)), int(m.group(2))
        stat[p][f] = (int(m.group(5)), m.group(7), int(m.group(9)), m.group(3) + 'x' + m.group(4), int(m.group(8)))
        continue
    m = re.search(r'dcnt f(\d+) p(\d+)((?: \d+)*)', line)
    if m:
        cnt[int(m.group(2))][int(m.group(1))] = tuple(int(x) for x in m.group(3).split())


def blinks(series):
    fs = sorted(series)
    n = 0
    for a, b, c in zip(fs, fs[1:], fs[2:]):
        if c - a == 2 and series[a] == series[c] and series[b] != series[a]:
            n += 1
    return n, len(fs)


for p in sorted(stat):
    if only is not None and p != only:
        continue
    s = stat[p]
    size = next(iter(s.values()))[3]
    draws = {f: v[0] for f, v in s.items()}
    fps = {f: v[1] for f, v in s.items()}
    mesh = {f: v[2] for f, v in s.items()}
    bd, nf = blinks(draws)
    bf, _ = blinks(fps)
    bm, _ = blinks(mesh)
    dv = sorted(set(draws.values()))
    line = (f'p{p:<3} {size:>10} frames {nf:>3}  draws {min(dv)}..{max(dv)} ({len(dv)} values, {bd} blinks)  '
            f'fp {len(set(fps.values()))} values ({bf} blinks)  mesh {min(mesh.values())}..{max(mesh.values())} ({bm} blinks)')
    if p in cnt:
        c = cnt[p]
        bc, _ = blinks(c)
        width = max(len(v) for v in c.values())
        varying = sum(1 for i in range(width) if len({v[i] if i < len(v) else -1 for v in c.values()}) > 1)
        line += f'  gpu-counts {width} per frame, {varying} vary, {bc} blink frames'
    print(line)

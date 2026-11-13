"""Differential test of the lifter, one instruction at a time, against Unicorn's AArch64.

Samples real instruction words from a module's code (default: the game's main), groups them by form (mnemonic and
operand shape from Capstone, register numbers and immediates folded), lifts up to --per-form words of each form with
recomp/lifter.py (simd='c', as the build does), compiles them against runtime/cpu.h and runtime/simd.h, runs each on
--states random register states, and runs the same word on the same states in Unicorn. Every difference in x0-x30,
sp, NZCV or v0-v31 is reported by form, with the word, its disassembly, the input and both outputs.

Loads, stores, branches, system instructions and anything whose lifted C calls into the runtime are not tested.
Differences only in NaN payloads (both sides NaN in every differing lane) are counted apart: they are real but rarely
change what a game does.

    py -3 tools/check_lift_diff.py --exefs "C:/Games/MKW PC/exefs" [--module main] [--filter fp|simd|int|all]
"""
import argparse
import collections
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'recomp'))
from lifter import Lifter, Unsupported  # noqa: E402
from nso import Nso  # noqa: E402

import capstone  # noqa: E402
import unicorn  # noqa: E402
from unicorn import arm64_const as A  # noqa: E402

BASES = {'main': 0x7000000000, 'subsdk0': 0x7003800000, 'subsdk1': 0x7003A00000, 'sdk': 0x7004400000}
STATE = struct.Struct('<31QQQ32s40s' + '16s' * 32)   # x0-30, sp, nzcv, pad, pad, v0-31 at offset 336 (16-aligned)
SKIP_CODE = re.compile(r'\b(LD|ST|rt_|ea\b|c->pc|excl|tpidr|__atomic|rt_dispatch|SWAP|CAS)')

F32_POOL = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 1e-3, 1e30, -1e30, 1e-40, 3.4e38, 65504.0, 1.5, -2.5, 0.1,
            float('inf'), float('-inf'), float('nan')]
F64_POOL = F32_POOL + [1e300, -1e-310, 2.0 ** 52, 2.0 ** 63, 4.5e15]


def rand64(r):
    k = r.randrange(9)
    if k == 0:
        return r.getrandbits(64)
    if k == 1:
        return r.randrange(0, 256)
    if k == 2:
        return (1 << 64) - r.randrange(1, 256)
    if k == 3:
        return r.getrandbits(32)
    if k == 4:
        return 1 << r.randrange(64)
    if k == 5:
        return struct.unpack('<Q', struct.pack('<d', r.choice(F64_POOL)))[0]
    if k == 6:
        a, b = (struct.unpack('<I', struct.pack('<f', r.choice(F32_POOL)))[0] for _ in range(2))
        return a | b << 32
    if k == 7:
        return 0x8000000000000000 if r.random() < 0.5 else 0x7fffffffffffffff
    return r.getrandbits(16) * 0x0001000100010001


def rand_state(r):
    xs = [rand64(r) for _ in range(31)]
    sp = (r.getrandbits(40) << 4) | 0x100000000
    nzcv = r.randrange(16) << 28
    vs = [struct.pack('<QQ', rand64(r), rand64(r)) for _ in range(32)]
    return xs, sp, nzcv, vs


def pack_state(s):
    xs, sp, nzcv, vs = s
    return STATE.pack(*xs, sp, nzcv, b'\0' * 32, b'\0' * 32, *vs)


def unpack_state(b):
    t = STATE.unpack(b)
    return list(t[:31]), t[31], t[32], list(t[35:])


FOLD = [(re.compile(r'\b[xw](\d+|zr)\b'), 'R'), (re.compile(r'\bsp\b'), 'SP'), (re.compile(r'\bv\d+'), 'V'),
        (re.compile(r'\b[qdshb]\d+\b'), lambda m: m.group(0)[0].upper()), (re.compile(r'#-?0x[0-9a-f]+|#-?\d+(\.\d+(e[+-]\d+)?)?'), '#')]


def form_of(ins):
    s = ins.op_str
    for rx, rep in FOLD:
        s = rx.sub(rep, s)
    return f'{ins.mnemonic} {s}'


def category(word):
    op0 = (word >> 25) & 0xF
    if (op0 & 0b0111) == 0b0111:
        return 'simd' if (word >> 28) & 1 == 0 else 'fp'
    if (op0 & 0b1110) == 0b1000 or (op0 & 0b0111) == 0b0101:
        return 'int'
    return None


def collect(nso, base, filt, per_form, seed, limit_words):
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.skipdata = True
    off, size = nso.text
    code = bytes(nso.image[off:off + size])
    forms = collections.defaultdict(list)
    seen = set()
    for ins in md.disasm(code, base + off):
        if ins.size != 4 or ins.mnemonic == '.byte':
            continue
        w = struct.unpack_from('<I', ins.bytes)[0]
        cat = category(w)
        if not cat or (filt != 'all' and cat != filt) or w in seen:
            continue
        seen.add(w)
        forms[form_of(ins)].append((w, ins.address, f'{ins.mnemonic} {ins.op_str}'))
        if limit_words and len(seen) >= limit_words:
            break
    r = random.Random(seed)
    picked = []
    for f, ws in sorted(forms.items()):
        r.shuffle(ws)
        picked += [(f, *x) for x in ws[:per_form]]
    return picked, len(forms)


def build_tests(picked, base):
    lf = Lifter(base, simd='c')
    tests, skipped = [], collections.Counter()
    for f, w, pc, text in picked:
        try:
            ins = lf.lift(w, pc)
        except Unsupported:
            skipped['unsupported'] += 1
            continue
        except Exception:
            skipped['lifter error'] += 1
            continue
        if ins.kind != 'op' or not ins.code or SKIP_CODE.search(ins.code):
            skipped[ins.kind if ins.kind != 'op' else 'memory/runtime'] += 1
            continue
        tests.append((f, w, pc, text, ins.code))
    return tests, skipped


def harness_c(tests):
    out = ['#include "cpu.h"', '#include "simd.h"', '#include <stdio.h>', '#include <stdlib.h>',
           'void rt_udf(Cpu *c, uint64_t pc) { (void)c; fprintf(stderr, "udf %llx\\n", (unsigned long long)pc); exit(3); }',
           'void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) { (void)c; (void)imm; (void)pc; exit(4); }',
           'void rt_fatal(const char *fmt, ...) { (void)fmt; exit(5); }']
    for i, (_, _, _, text, code) in enumerate(tests):
        out.append(f'static void t{i}(Cpu *c) {{ /* {text} */ {code} }}')
    out.append('static void (*const T[])(Cpu *) = {' + ', '.join(f't{i}' for i in range(len(tests))) + '};')
    out.append(r'''
typedef struct { uint64_t x[31], sp, nzcv, pad[9]; __uint128_t v[32]; } St;
_Static_assert(sizeof(St) == 848, "state layout");
int main(int argc, char **argv) {
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
    long ns = atol(argv[3]);
    St *s = malloc(sizeof(St) * ns);
    if (fread(s, sizeof(St), ns, in) != (size_t)ns) return 2;
    static Cpu c;
    for (size_t t = 0; t < sizeof T / sizeof *T; t++)
        for (long k = 0; k < ns; k++) {
            memset(&c, 0, sizeof c);
            memcpy(c.x, s[k].x, sizeof s[k].x);
            c.sp = s[k].sp, c.nzcv = s[k].nzcv, c.fpcr = 0, c.fpsr = 0;
            memcpy(c.v, s[k].v, sizeof c.v);
            T[t](&c);
            St o = {0};
            memcpy(o.x, c.x, sizeof o.x);
            o.sp = c.sp, o.nzcv = c.nzcv;
            memcpy(o.v, c.v, sizeof o.v);
            fwrite(&o, sizeof o, 1, out);
        }
    fclose(out);
    return 0;
}''')
    return '\n'.join(out)


def uc_run(uc, pc, s):
    xs, sp, nzcv, vs = s
    for i in range(29):
        uc.reg_write(A.UC_ARM64_REG_X0 + i, xs[i])
    uc.reg_write(A.UC_ARM64_REG_X29, xs[29])
    uc.reg_write(A.UC_ARM64_REG_X30, xs[30])
    uc.reg_write(A.UC_ARM64_REG_SP, sp)
    uc.reg_write(A.UC_ARM64_REG_NZCV, nzcv)
    uc.reg_write(A.UC_ARM64_REG_FPCR, 0)
    uc.reg_write(A.UC_ARM64_REG_FPSR, 0)
    for i in range(32):
        uc.reg_write(A.UC_ARM64_REG_Q0 + i, int.from_bytes(vs[i], 'little'))
    uc.emu_start(pc, pc + 4, count=1)
    ox = [uc.reg_read(A.UC_ARM64_REG_X0 + i) for i in range(29)]
    ox += [uc.reg_read(A.UC_ARM64_REG_X29), uc.reg_read(A.UC_ARM64_REG_X30)]
    ov = [uc.reg_read(A.UC_ARM64_REG_Q0 + i).to_bytes(16, 'little') for i in range(32)]
    return ox, uc.reg_read(A.UC_ARM64_REG_SP), uc.reg_read(A.UC_ARM64_REG_NZCV) & 0xF0000000, ov


def is_nan_word(b, esz):
    v = int.from_bytes(b, 'little')
    if esz == 4:
        return (v & 0x7f800000) == 0x7f800000 and (v & 0x7fffff)
    if esz == 8:
        return (v & 0x7ff0000000000000) == 0x7ff0000000000000 and (v & 0xfffffffffffff)
    return (v & 0x7c00) == 0x7c00 and (v & 0x3ff)


def nan_only(a, b):
    """Every differing lane is NaN on both sides, for some lane size."""
    for esz in (2, 4, 8):
        ok = True
        for i in range(0, 16, esz):
            if a[i:i + esz] != b[i:i + esz] and not (is_nan_word(a[i:i + esz], esz) and is_nan_word(b[i:i + esz], esz)):
                ok = False
                break
        if ok:
            return True
    return False


def compare(got, want):
    gx, gsp, gn, gv = got
    wx, wsp, wn, wv = want
    diffs, nan = [], True
    for i in range(31):
        if gx[i] != wx[i]:
            diffs.append(f'x{i} {gx[i]:#x} want {wx[i]:#x}')
            nan = False
    if gsp != wsp:
        diffs.append(f'sp {gsp:#x} want {wsp:#x}')
        nan = False
    if (gn & 0xF0000000) != wn:
        diffs.append(f'nzcv {gn >> 28:04b} want {wn >> 28:04b}')
        nan = False
    for i in range(32):
        if gv[i] != wv[i]:
            diffs.append(f'v{i} {gv[i][::-1].hex()} want {wv[i][::-1].hex()}')
            nan = nan and nan_only(gv[i], wv[i])
    return diffs, nan and bool(diffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exefs', required=True)
    ap.add_argument('--module', default='main', choices=sorted(BASES))
    ap.add_argument('--filter', default='all', choices=['all', 'fp', 'simd', 'int'])
    ap.add_argument('--per-form', type=int, default=3)
    ap.add_argument('--states', type=int, default=48)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--limit-words', type=int, default=0, help='stop collecting after this many distinct words')
    ap.add_argument('--out', default=None, help='report file (default: print)')
    ap.add_argument('--cc', default='C:/msys64/ucrt64/bin/clang.exe')
    a = ap.parse_args()

    base = BASES[a.module]
    nso = Nso(os.path.join(a.exefs, a.module))
    picked, nforms = collect(nso, base, a.filter, a.per_form, a.seed, a.limit_words)
    tests, skipped = build_tests(picked, base)
    print(f'{nforms} forms, {len(picked)} words picked, {len(tests)} testable; skipped {dict(skipped)}', flush=True)

    r = random.Random(a.seed)
    states = [rand_state(r) for _ in range(a.states)]
    work = Path(tempfile.mkdtemp(prefix='liftdiff-'))
    (work / 'h.c').write_text(harness_c(tests))
    exe = work / 'h.exe'
    cmd = [a.cc, '-O1', '-fno-strict-aliasing', '-Wno-everything', '-march=x86-64-v3', '-I', str(REPO / 'runtime'),
           str(work / 'h.c'), '-o', str(exe)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-4000:])
        sys.exit('harness does not compile')
    (work / 'in.bin').write_bytes(b''.join(pack_state(s) for s in states))
    p = subprocess.run([str(exe), str(work / 'in.bin'), str(work / 'out.bin'), str(len(states))], capture_output=True,
                       text=True)
    if p.returncode:
        sys.exit(f'harness failed ({p.returncode}): {p.stderr[-2000:]}')
    blob = (work / 'out.bin').read_bytes()

    uc = unicorn.Uc(unicorn.UC_ARCH_ARM64, unicorn.UC_MODE_ARM)
    uc.reg_write(A.UC_ARM64_REG_CPACR_EL1, uc.reg_read(A.UC_ARM64_REG_CPACR_EL1) | (3 << 20))
    mapped = set()
    bad = collections.defaultdict(list)
    nanf = collections.Counter()
    uc_err = collections.Counter()
    for t, (f, w, pc, text, code) in enumerate(tests):
        page = pc & ~0xFFF
        if page not in mapped:
            uc.mem_map(page, 0x1000)
            mapped.add(page)
        uc.mem_write(pc, struct.pack('<I', w))
        for k, s in enumerate(states):
            o = (t * len(states) + k) * STATE.size
            got = unpack_state(blob[o:o + STATE.size])
            got = (got[0], got[1], got[2], got[3])
            try:
                want = uc_run(uc, pc, s)
            except unicorn.UcError as e:
                uc_err[str(e)] += 1
                break
            diffs, nan = compare(got, want)
            if not diffs:
                continue
            if nan:
                nanf[f] += 1
                continue
            if len(bad[f]) < 3:
                xs, sp, nzcv, vs = s
                bad[f].append(f'  {w:08x} @ {pc:#x}  {text}\n    in: nzcv {nzcv >> 28:04b} '
                              + ' '.join(f'x{i}={xs[i]:#x}' for i in range(31) if f'x{i} ' in ' '.join(diffs) or
                                         re.search(rf'\b[xw]{i}\b', text))
                              + ' ' + ' '.join(f'v{i}={vs[i][::-1].hex()}' for i in range(32) if re.search(rf'\b[vqdsbh]{i}\b', text))
                              + '\n    ' + '\n    '.join(diffs))
    lines = [f'{len(tests)} words tested x {len(states)} states; {len(bad)} forms differ; {len(nanf)} forms differ only in NaN payloads',
             f'unicorn errors: {dict(uc_err)}' if uc_err else '']
    for f in sorted(bad):
        lines.append(f'FORM {f}')
        lines += bad[f]
    if nanf:
        lines.append('NaN-payload-only forms: ' + ', '.join(f'{f} ({n})' for f, n in sorted(nanf.items())))
    rep = '\n'.join(lines)
    if a.out:
        Path(a.out).write_text(rep)
        print(f'report: {a.out}')
    print(rep[:6000])


if __name__ == '__main__':
    main()

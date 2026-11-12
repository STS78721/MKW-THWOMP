"""CPU-only content-key differential test for the production mesh shader memo.

Compiles the actual NVN2 hash, mesh key functions and memo from the runtime;
compares memoized keys with fresh full-content hashes across program lifetimes.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
mesh = (ROOT / "runtime/vk_mesh.c").read_text(encoding="utf-8")
shader = (ROOT / "runtime/gfx_shader.c").read_text(encoding="utf-8")


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


hash_code = "\n".join(function(shader, signature) for signature in (
    "int gfx_shader_is_nvn2(", "static int nvn2_sizes(", "uint64_t gfx_shader_nvn2_hash("))
raw_key = function(mesh, "static uint64_t mesh_stage_key(")
memo = mesh[mesh.index("#define MESH_KEY_MEMOS "):mesh.index("static void mesh_dump_spirv(")]
prim_key = function(mesh, "static uint64_t prim_sph_key(")
# Keep primitive-header and synthetic-header dependencies at the call site, outside the raw stage memo.
assert "mesh_program_stage_key(ex->prog, 1, mctl, mcode) ^ prim_sph_key(prim_sph)" in mesh
assert "tcode ? mesh_program_stage_key(ex->prog, 0, tctl, tcode) : 0" in mesh

harness = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct { uint64_t serial; } prog_t;
@HASH@
static uint64_t hashes;
static uint64_t counted_hash(const uint8_t *ctl, const uint8_t *code) { hashes++; return gfx_shader_nvn2_hash(ctl, code); }
#define gfx_shader_nvn2_hash counted_hash
@RAW@
@MEMO@
#undef gfx_shader_nvn2_hash
@PRIM@

static uint32_t rng = 0x571b7ecdu;
static uint32_t random_word(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void random_bytes(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)random_word(); }
static void u32(uint8_t *p, uint32_t n) { memcpy(p, &n, 4); }
static void control(uint8_t *p, uint32_t n) {
    random_bytes(p, 0x880);
    u32(p, 0x98761234u); u32(p + 4, 0xc9);
    u32(p + 0x30, 0xd8); u32(p + 0x38, n); u32(p + 0x44, n);
}
static uint64_t fresh(const uint8_t *ctl, const uint8_t *code) {
    uint64_t h = gfx_shader_nvn2_hash(ctl, code);
    return (h ? h : (uint64_t)(uintptr_t)code) ^ 4;
}

int main(void) {
    uint8_t ctl[3][0x880], code[3][8192], prim[0x50];
    uint64_t comparisons = 0;
    for (uint64_t lifetime = 1; lifetime <= 2000; lifetime++) {
        // The same guest/program addresses are reused for different contents after a new ProgramSetShaders serial.
        prog_t p = {lifetime};
        control(ctl[0], 2048); control(ctl[1], 4096);
        random_bytes(code[0], sizeof code[0]); random_bytes(code[1], sizeof code[1]);
        int has_task = (int)(random_word() & 1);
        uint64_t h0 = hashes;
        for (unsigned draw = 0; draw < 100; draw++) {
            random_bytes(prim, sizeof prim); // primitive translation inputs can change independently
            int synth = (int)(draw & 1), has_prim = draw % 3 != 0;
            const uint8_t *sp = has_prim ? prim : NULL;
            uint64_t task = has_task ? mesh_program_stage_key(&p, 0, ctl[0], code[0]) : 0;
            uint64_t key = mesh_program_stage_key(&p, 1, ctl[1], code[1]) ^ prim_sph_key(sp) ^ (synth ? 0x5052494dull : 0);
            assert(task == (has_task ? fresh(ctl[0], code[0]) : 0));
            assert(key == (fresh(ctl[1], code[1]) ^ prim_sph_key(sp) ^ (synth ? 0x5052494dull : 0)));
            comparisons += 2;
        }
        assert(hashes - h0 == (uint64_t)(1 + has_task));
    }
    // A changed stage pointer is conservatively rehashed even under the same serial.
    prog_t p = {9000};
    control(ctl[2], 1024); random_bytes(code[2], sizeof code[2]);
    assert(mesh_program_stage_key(&p, 1, ctl[0], code[0]) == fresh(ctl[0], code[0]));
    assert(mesh_program_stage_key(&p, 1, ctl[2], code[2]) == fresh(ctl[2], code[2]));
    assert(mesh_program_stage_key(&p, 1, ctl[2], code[0]) == fresh(ctl[2], code[0]));
    // Header-only/invalid NVN2 stages preserve the pointer fallback, including reuse under a new serial.
    u32(ctl[2] + 0x38, 0); u32(ctl[2] + 0x44, 0); p.serial++;
    assert(mesh_program_stage_key(&p, 1, ctl[2], code[2]) == fresh(ctl[2], code[2]));
    // Serial zero offers no lifetime guarantee: in-place changes must always be hashed again.
    p.serial = 0; control(ctl[2], 1024);
    uint64_t h0 = hashes;
    for (unsigned i = 0; i < 100; i++) {
        code[2][i] ^= (uint8_t)(i + 1);
        assert(mesh_program_stage_key(&p, 1, ctl[2], code[2]) == fresh(ctl[2], code[2]));
    }
    assert(hashes - h0 == 100);
    // Find colliding serials and alternate them: collisions may only cost a fresh hash.
    prog_t a = {1}, b = {2};
    while ((a.serial * 0x9E3779B97F4A7C15ull >> 55) != (b.serial * 0x9E3779B97F4A7C15ull >> 55)) b.serial++;
    for (unsigned i = 0; i < 100; i++) {
        assert(mesh_program_stage_key(&a, 0, ctl[0], code[0]) == fresh(ctl[0], code[0]));
        assert(mesh_program_stage_key(&b, 1, ctl[1], code[1]) == fresh(ctl[1], code[1]));
    }
    printf("mesh key memo: passed %llu draw-key comparisons across 2000 address-reused lifetimes; "
           "missing tasks, changed primitive headers, pointer changes, zero serials and collisions\n",
           (unsigned long long)comparisons);
    return 0;
}
""".replace("@HASH@", hash_code).replace("@RAW@", raw_key).replace("@MEMO@", memo).replace("@PRIM@", prim_key)

compiler = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
if not compiler and Path("C:/msys64/ucrt64/bin/clang.exe").exists():
    compiler = "C:/msys64/ucrt64/bin/clang.exe"
if not compiler:
    raise SystemExit("a C compiler (clang or cc) is required")

with tempfile.TemporaryDirectory(prefix="mkw-mesh-keys-") as tmp:
    c = Path(tmp) / "keys.c"
    exe = Path(tmp) / ("keys.exe" if os.name == "nt" else "keys")
    c.write_text(harness, encoding="utf-8")
    subprocess.run([compiler, "-std=gnu11", "-O2", "-Wall", "-Wextra", str(c), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

"""CPU-only differential replay of Vulkan draw-state optimizations; never loads a Vulkan driver.

Compiles the production dynamic-state and vertex-binding span code with command-recording mocks. The baseline
records full state each time; the optimized path must leave exactly the same command state after every transition.
Usage: python tools/check_draw_state.py [--cc clang] [--vma-include PATH]
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"

typedef struct {
    uint32_t n_attr, n_bind;
    VkVertexInputAttributeDescription2EXT a[32];
    VkVertexInputBindingDescription2EXT b[17];
} VtxInput;
typedef struct {
    uint32_t scalar[13], stencil[2][4];
    VkBool32 en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
    VtxInput vi;
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
} Recorded;
static Recorded out;
static uint64_t calls;
static struct {
    int dyn_vi, dyn_clamp, dyn_poly, dyn_a2c, dyn_blend_en, dyn_blend_eq, dyn_wmask;
    PFN_vkCmdSetVertexInputEXT set_vi;
    PFN_vkCmdSetDepthClampEnableEXT set_clamp;
    PFN_vkCmdSetPolygonModeEXT set_poly;
    PFN_vkCmdSetAlphaToCoverageEnableEXT set_a2c;
    PFN_vkCmdSetColorBlendEnableEXT set_blend_en;
    PFN_vkCmdSetColorBlendEquationEXT set_blend_eq;
    PFN_vkCmdSetColorWriteMaskEXT set_wmask;
} P;
static int ensure_init(void) { return 1; }
#define SCALAR(fn, T, index) \
VKAPI_ATTR void VKAPI_CALL fn(VkCommandBuffer cb, T value) { (void)cb; out.scalar[index] = value; calls++; }
SCALAR(vkCmdSetCullMode, VkCullModeFlags, 0)
SCALAR(vkCmdSetFrontFace, VkFrontFace, 1)
SCALAR(vkCmdSetPrimitiveTopology, VkPrimitiveTopology, 2)
SCALAR(vkCmdSetPrimitiveRestartEnable, VkBool32, 3)
SCALAR(vkCmdSetDepthTestEnable, VkBool32, 4)
SCALAR(vkCmdSetDepthWriteEnable, VkBool32, 5)
SCALAR(vkCmdSetDepthCompareOp, VkCompareOp, 6)
SCALAR(vkCmdSetStencilTestEnable, VkBool32, 7)
SCALAR(vkCmdSetDepthBiasEnable, VkBool32, 8)
SCALAR(vkCmdSetRasterizerDiscardEnable, VkBool32, 9)
VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags face,
        VkStencilOp fail, VkStencilOp pass, VkStencilOp zfail, VkCompareOp compare) {
    (void)cb;
    unsigned i = face == VK_STENCIL_FACE_BACK_BIT;
    out.stencil[i][0] = fail; out.stencil[i][1] = pass; out.stencil[i][2] = zfail; out.stencil[i][3] = compare;
    calls++;
}
void rec_bool(PFN_vkCmdSetDepthClampEnableEXT fn, VkCommandBuffer cb, VkBool32 value) {
    (void)cb;
    out.scalar[fn == P.set_clamp ? 10 : 12] = value; calls++;
}
void rec_polygon_mode(PFN_vkCmdSetPolygonModeEXT fn, VkCommandBuffer cb, VkPolygonMode value) {
    (void)fn; (void)cb; out.scalar[11] = value; calls++;
}
void rec_vertex_input(PFN_vkCmdSetVertexInputEXT fn, VkCommandBuffer cb, uint32_t nb,
        const VkVertexInputBindingDescription2EXT *b, uint32_t na, const VkVertexInputAttributeDescription2EXT *a) {
    (void)fn; (void)cb;
    memset(&out.vi, 0, sizeof out.vi);
    out.vi.n_bind = nb; out.vi.n_attr = na;
    memcpy(out.vi.b, b, nb * sizeof *b); memcpy(out.vi.a, a, na * sizeof *a); calls++;
}
void rec_blend_enable(PFN_vkCmdSetColorBlendEnableEXT fn, VkCommandBuffer cb, uint32_t first,
        uint32_t n, const VkBool32 *v) { (void)fn; (void)cb; memcpy(out.en + first, v, n * sizeof *v); calls++; }
void rec_blend_equation(PFN_vkCmdSetColorBlendEquationEXT fn, VkCommandBuffer cb, uint32_t first,
        uint32_t n, const VkColorBlendEquationEXT *v) { (void)fn; (void)cb; memcpy(out.eq + first, v, n * sizeof *v); calls++; }
void rec_write_mask(PFN_vkCmdSetColorWriteMaskEXT fn, VkCommandBuffer cb, uint32_t first,
        uint32_t n, const VkColorComponentFlags *v) { (void)fn; (void)cb; memcpy(out.wm + first, v, n * sizeof *v); calls++; }
static VkCompareOp compare_op(unsigned x) { return (VkCompareOp)x; }
static VkStencilOp stencil_op(unsigned x) { return (VkStencilOp)x; }
static VkPolygonMode polygon_mode(unsigned x) { return (VkPolygonMode)x; }
static VkBlendFactor blend_factor(unsigned x) { return (VkBlendFactor)x; }
static VkBlendOp blend_op(unsigned x) { return (VkBlendOp)x; }
VkPrimitiveTopology vk_pipeline_topology(int x) { return (VkPrimitiveTopology)x; }
static int target_count(const VkPipeState *s) {
    int n = 0; for (int i = 0; i < 8; i++) if (s->color_fmt[i]) n = i + 1; return n;
}
static VkFormat target_format(int f, int *d, int *st) { (void)d; (void)st; return (VkFormat)f; }
static int blendable(VkFormat f) { return f != 0 && (f & 1); }
// Model immutable shader metadata and format mappings. The production function's guards must retain all these
// dependencies; the test compares exact emitted values, including targets that disappear and later return.
static uint32_t target_wmask(const VkPipeState *s, int i) {
    uint32_t mask = s->chmask.mask >> (i * 4) & 15;
    if (s->fs && !((uintptr_t)s->fs >> i & 1)) mask = 0;
    if (i == 1 && s->color.blend_en && s->blend[0].src_rgb == 3 && target_count(s) > 1 && blendable(s->color_fmt[0])) mask = 0;
    return mask;
}
static void vertex_input(const VkPipeState *s, VtxInput *v) {
    memset(v, 0, sizeof *v); v->n_attr = v->n_bind = 16;
    for (unsigned i = 0; i < 16; i++) {
        v->a[i].location = i; v->a[i].format = s->vattr[i]; v->a[i].offset = (uintptr_t)s->vs;
        v->b[i].binding = i; v->b[i].stride = s->vstream[i].stride; v->b[i].divisor = s->vstream[i].divisor;
    }
}
/* DYNAMIC_IMPLEMENTATION */

static struct { VkBuffer vb[17]; VkDeviceSize vo[17], vsz[17], vstr[17]; int vb_valid; } B;
typedef struct { VkCommandBuffer cb; } Exec;
VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n,
        const VkBuffer *vb, const VkDeviceSize *vo, const VkDeviceSize *vsz, const VkDeviceSize *vstr) {
    (void)cb;
    if (!n || first + n > 17) abort();
    memcpy(out.vb + first, vb, n * sizeof *vb); memcpy(out.vo + first, vo, n * sizeof *vo);
    memcpy(out.vsz + first, vsz, n * sizeof *vsz);
    if (vstr) memcpy(out.vstr + first, vstr, n * sizeof *vstr);
    calls += n;
}
static void bind_range(Exec *ex, VkBuffer *vb, VkDeviceSize *vo, VkDeviceSize *vsz, VkDeviceSize *vstr, int dyn) {
/* VERTEX_IMPLEMENTATION */
}
static uint32_t rng = 0x8324fb10;
static uint32_t random32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void check(const Recorded *want, unsigned caps, unsigned step, const char *what) {
    if (memcmp(want, &out, sizeof out)) {
        fprintf(stderr, "%s state mismatch caps=%u transition=%u\n", what, caps, step); exit(1);
    }
}
int main(void) {
    uint64_t full = 0, delta = 0, vb_full = 0, vb_delta = 0;
    P.set_clamp = (void *)1; P.set_a2c = (void *)2;
    for (unsigned caps = 0; caps < 128; caps++) {
        P.dyn_vi = caps & 1; P.dyn_clamp = caps & 2; P.dyn_poly = caps & 4; P.dyn_a2c = caps & 8;
        P.dyn_blend_en = caps & 16; P.dyn_blend_eq = caps & 32; P.dyn_wmask = caps & 64;
        Recorded expected = {0}, optimized = {0};
        VkPipeState state = {0}, previous = {0}; state.vs = (void *)1; state.fs = (void *)0xff;
        // Explicit attachment-count cycles cover stale values outside a temporarily shortened target list.
        static const unsigned targets[] = {0, 8, 0, 1, 8, 3, 0, 5, 0, 8};
        for (unsigned step = 0; step < sizeof targets / sizeof *targets; step++) {
            memset(state.color_fmt, 0, sizeof state.color_fmt);
            for (unsigned i = 0; i < targets[step]; i++) state.color_fmt[i] = 101 + i * 2;
            state.fs = (void *)(uintptr_t)(step * 17 + 1); state.chmask.mask = ~0u;
            out = expected; set_dynamic(NULL, &state, NULL, 0); expected = out;
            out = optimized; set_dynamic(NULL, &state, step % 3 ? &previous : NULL, 0);
            check(&expected, caps, step, "attachment cycle"); optimized = out; previous = state;
        }
        // Mesh draws omit topology, primitive restart and vertex input; the next conventional draw starts afresh.
        Recorded before_mesh = optimized;
        state.prim ^= 7; state.prim_restart ^= 1; state.vattr[3] ^= 127; state.poly.cull ^= 3;
        out = expected; set_dynamic(NULL, &state, NULL, 0); expected = out;
        expected.scalar[2] = before_mesh.scalar[2]; expected.scalar[3] = before_mesh.scalar[3]; expected.vi = before_mesh.vi;
        out = optimized; set_dynamic(NULL, &state, NULL, 1);
        check(&expected, caps, 0, "mesh omission"); optimized = out;
        out = expected; set_dynamic(NULL, &state, NULL, 0); expected = out;
        out = optimized; set_dynamic(NULL, &state, NULL, 0);
        check(&expected, caps, 0, "post-mesh reset"); optimized = out; previous = state;
        for (unsigned step = 0; step < 4096; step++) {
            unsigned pos = random32() % sizeof state;
            ((unsigned char *)&state)[pos] ^= 1u << (random32() & 7);
            if (!(step % 257)) memset(&expected, 0, sizeof expected), memset(&optimized, 0, sizeof optimized);
            out = expected; calls = 0; set_dynamic(NULL, &state, NULL, 0); full += calls; expected = out;
            out = optimized; calls = 0;
            set_dynamic(NULL, &state, step % 257 ? &previous : NULL, 0); delta += calls;
            check(&expected, caps, step, "dynamic"); optimized = out; previous = state;
            calls = 0; set_dynamic(NULL, &state, &previous, 0);
            if (calls) { fprintf(stderr, "unchanged state emitted commands\n"); return 1; }
            check(&expected, caps, step, "unchanged");
        }
    }
    for (int dyn = 0; dyn < 2; dyn++) {
        VkBuffer vb[17] = {0}; VkDeviceSize vo[17] = {0}, vsz[17] = {0}, vstr[17] = {0};
        Recorded expected = {0}, optimized = {0}; Exec ex = {0}; memset(&B, 0, sizeof B);
        for (unsigned step = 0; step < 32768; step++) {
            unsigned b = random32() % 17;
            switch (random32() % 5) {
            case 0: vb[b] = (VkBuffer)(uintptr_t)random32(); break;
            case 1: vo[b] = random32(); break;
            case 2: vsz[b] = random32(); break;
            case 3: vstr[b] = random32(); break;
            default: break; // unchanged draw
            }
            if (!(step % 257)) B.vb_valid = 0;
            out = expected; calls = 0; vkCmdBindVertexBuffers2(NULL, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
            vb_full += calls; expected = out;
            out = optimized; calls = 0; bind_range(&ex, vb, vo, vsz, vstr, dyn); vb_delta += calls;
            check(&expected, dyn, step, "vertex"); optimized = out;
            memcpy(B.vb, vb, sizeof vb); memcpy(B.vo, vo, sizeof vo);
            memcpy(B.vsz, vsz, sizeof vsz); memcpy(B.vstr, vstr, sizeof vstr); B.vb_valid = 1;
        }
    }
    printf("PASS: 524288 dynamic transitions, all 128 optional capability combinations, attachment cycles, resets and mesh omission; 65536 vertex transitions\n");
    printf("Mock command counts: dynamic full=%llu delta=%llu (%.1f%% fewer); vertex binding slots full=%llu range=%llu (%.1f%% fewer)\n",
           (unsigned long long)full, (unsigned long long)delta, 100.0 * (full - delta) / full,
           (unsigned long long)vb_full, (unsigned long long)vb_delta, 100.0 * (vb_full - vb_delta) / vb_full);
    return 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default=shutil.which("clang") or "clang")
    ap.add_argument("--vma-include", default="C:/msys64/ucrt64/include/vma")
    args = ap.parse_args()
    pipeline = (ROOT / "runtime/vk_pipeline.c").read_text()
    start = pipeline.index("static void set_dynamic(", pipeline.index("void vk_pipeline_set_dynamic_ex("))
    dynamic = pipeline[start:pipeline.index("// ---- statistics", start)]
    draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
    start = draw.index("    unsigned first = 0, end = 17;")
    vertex = draw[start:draw.index("    ex->recorded++;", start)]
    source = HARNESS.replace("/* DYNAMIC_IMPLEMENTATION */", dynamic).replace("/* VERTEX_IMPLEMENTATION */", vertex)
    with tempfile.TemporaryDirectory(prefix="mkw-draw-state-") as tmp:
        src = Path(tmp) / "check.c"
        exe = Path(tmp) / "check.exe"
        src.write_text(source)
        subprocess.run([args.cc, "-O2", "-Wall", "-Wextra", "-I", str(ROOT / "runtime"),
                        "-I", args.vma_include, str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()

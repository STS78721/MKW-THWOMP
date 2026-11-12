"""CPU-only differential checks for pointer-table scans, c0 banks and descriptor-array reuse.

Compiles production helpers against deterministic memory-pool mocks. No Vulkan driver is loaded.
Usage: python tools/check_pointer_tables.py [--cc clang] [--vma-include PATH]
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

OLD_ENUM = r'''
        uint32_t probed = 0;
        uint64_t dense = buf_sz <= 0x1000ull ? buf_sz : 0x400ull;
        uint64_t stride_end = buf_sz;
        if (stride_end > (256ull << 10)) stride_end = 256ull << 10;
        uint64_t offs[2048];
        uint32_t no = 0;
        for (uint64_t off = 0; off + 8 <= dense && no < 512; off += 8) offs[no++] = off;
        uint64_t mid = stride_end < (64ull << 10) ? stride_end : (64ull << 10);
        for (uint64_t off = dense; off + 8 <= mid && no < 1600; off += 0x20) offs[no++] = off;
        for (uint64_t off = mid; off + 8 <= stride_end && no < 2000; off += 0x80) offs[no++] = off;
        if (buf_sz > stride_end) {
            for (int q = 1; q <= 3 && no < 2048; q++) {
                uint64_t t = (buf_sz / 4) * (uint64_t)q;
                t &= ~7ull;
                if (t + 8 <= buf_sz && t >= stride_end) offs[no++] = t;
            }
            uint64_t tail = buf_sz > 8 ? (buf_sz - 8) & ~7ull : 0;
            if (tail >= stride_end && no < 2048) offs[no++] = tail;
        }
        for (uint32_t oi = 0; oi < no && *n_io < 48; oi++) {
            uint64_t off = offs[oi];
            uint64_t v = base[off / 8];
            probed++;
            reference_try_add(v, i, off, found, n_io, bound, nbound, &scan_log);
        }
'''

HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vk_exec.h"
#include "nvn2_c0.h"
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_TS 3
#define NVN_STAGE_MS 4
#define NVN_STAGE_CS 5

#define POOL_BYTES (2u << 20)
static uint8_t *memory;
static uint64_t resolves;
static int unavailable, device_local, readable = 1;
static VkCtx context;
VkCtx *vk_ctx(void) { return &context; }
void rt_log(const char *fmt, ...) { (void)fmt; }
__attribute__((noreturn)) void rt_fatal(const char *fmt, ...) { fprintf(stderr, "%s\n", fmt); abort(); }
int plat_mem_readable(uint64_t addr, uint64_t n) { (void)addr; (void)n; return readable; }
int vk_mem_resolve(uint64_t addr, uint64_t n, VkMemSpan *out) {
    resolves++; memset(out, 0, sizeof *out);
    uint64_t lo = (uintptr_t)memory;
    if (unavailable || addr < lo || addr - lo >= 4u * POOL_BYTES) return 0;
    uint64_t off = (addr - lo) % POOL_BYTES;
    if (n > POOL_BYTES - off) return 0;
    out->avail = POOL_BYTES - off; out->offset = off;
    out->buffer = (void *)(uintptr_t)(1 + (addr - lo) / POOL_BYTES);
    out->alloc = device_local ? (void *)1 : NULL;
    return 1;
}
static uint64_t mem_gen = 1, usnap_gen = 1;
static uint64_t ptr_cb_epoch = 1;
uint64_t vk_mem_gen(void) { return mem_gen; }
uint64_t vk_usnap_gen(void) { return usnap_gen; }
static void invalidate_pointer_cache(void) { mem_gen++; }
static const Nvn2C0Layout *c0_lay;
static const uint32_t (*c0_push)[NVN2_ROOT_WORDS];
static const uint64_t *c0_tex, *c0_img;
static uint32_t c0_dim[3], c0_binst;
static uint64_t c0_fills;
static int fix_c0_enabled = 1;
static int c0_on(void) { return fix_c0_enabled; }
static int find_stage(const prog_t *prog, int stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    (void)prog; (void)stage; (void)ctl; (void)code; (void)len; return 0;
}
static int ubo_ptr_res(const GfxShaderRes *r) { return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18; }
static uint64_t ubo_ptr(const VkExec *ex, int stage, const GfxShaderRes *r) {
    return ex->ubo[stage][r->cbuf_slot - 3].addr;
}
typedef struct { int slot; uint64_t off, value; } Sample;
static Sample samples[16 * 2048];
static uint32_t sample_count;
static void sample_note(uint64_t v, int slot, uint64_t off) {
    if (sample_count >= sizeof samples / sizeof *samples) abort();
    samples[sample_count++] = (Sample){slot, off, v};
}
/* PRODUCTION_HELPERS */
/* REFERENCE_TRY_ADD */
/* REFERENCE_SCAN */
static void reference_table4(const p_bind_buf *a, const p_bind_buf *b, const p_bind_buf *c, const p_bind_buf *d,
                             PtrEntry out[PTR_ENTRIES]) {
    memset(out, 0, sizeof(PtrEntry) * PTR_ENTRIES);
    uint64_t found[48]; uint32_t n = 0; const p_bind_buf *src[4] = {a, b, c, d};
    for (int i = 0; i < 4; i++) reference_scan(src[i], 0, 8, ~0ull, found, &n, src, 4);
    pointer_table_finish(found, n, src, 4, out);
}
/* POINTER_BINDING */
/* MESH_BANK_CACHE */
/* GUARDED_C0 */
/* REFERENCE_GUARDED_C0 */

static uint32_t rng = 0x64b809ad;
static uint32_t random32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void check(int ok, const char *what, unsigned step) {
    if (!ok) { fprintf(stderr, "FAIL %s case %u\n", what, step); exit(1); }
}
static uint64_t random64(void) { return (uint64_t)random32() << 32 | random32(); }
static volatile uint64_t benchmark_checksum;
static void check_exclusions(void) {
    p_bind_buf table[6][16]; const p_bind_buf *src[6]; PtrBoundSet flat;
    unsigned comparisons = 0;
    for (unsigned trial = 0; trial < 10000; trial++) {
        int count = trial % 7;
        for (int s = 0; s < 6; s++) {
            src[s] = random32() % 5 ? table[s] : NULL;
            for (int i = 0; i < 16; i++) {
                uint64_t lo = random64(), size = random64();
                if (random32() % 3 == 0) lo = 0;
                if (random32() % 4 == 0) size = 0;
                table[s][i] = (p_bind_buf){.addr = lo, .size = size};
            }
        }
        ptr_bounds_init(&flat, src, count);
        for (int s = 0; s < 6; s++) for (int i = 0; i < 16; i++) {
            uint64_t values[3] = {random64(), table[s][i].addr, table[s][i].addr + table[s][i].size};
            for (int k = 0; k < 3; k++) {
                check(ptr_exclude_sources(values[k], src, count) == ptr_exclude(values[k], &flat),
                      "flattened exclusion/uint64 endpoints", comparisons++);
            }
        }
    }
    // Unexpected callers with more than 64 active ranges must retain exclusions from every source.
    for (int s = 0; s < 6; s++) {
        src[s] = table[s];
        for (int i = 0; i < 16; i++)
            table[s][i] = (p_bind_buf){.addr = 0x100000000ull + (s * 16 + i) * 0x10000ull, .size = 0x8000};
    }
    ptr_bounds_init(&flat, src, 6);
    check(flat.n > PTR_BOUND_RANGES, "exclusion capacity fallback selected", 0);
    for (int s = 0; s < 6; s++) for (int i = 0; i < 16; i++)
        check(ptr_exclude(table[s][i].addr, &flat), "exclusion capacity fallback preserved all sources", s * 16 + i);
    printf("PASS: %u flattened exclusion comparisons (empty/missing sources, zero and wrapping ranges, exact endpoints); 96-range overflow fallback\n", comparisons);

    // A synthetic exclusion-only benchmark: include rebuilding the local list per 1024 sampled pointers.
    // It quantifies removed slot checks without predicting gameplay FPS or asserting timing thresholds.
    for (int count = 1; count <= 4; count *= 2) {
        uint64_t values[1024];
        memset(table, 0, sizeof table);
        for (int s = 0; s < count; s++) for (int i = 0; i < 2; i++)
            table[s][2 + 9 * i] = (p_bind_buf){.addr = 0x100000000ull + s * 0x100000ull + i * 0x10000ull, .size = 0x4000};
        for (unsigned i = 0; i < 1024; i++) values[i] = 0x100000000ull + (random32() % 8) * 0x100000ull + (random32() % 32) * 0x1000ull;
        uint64_t expected = 0, actual = 0;
        clock_t begin = clock();
        for (int repeat = 0; repeat < 10000; repeat++)
            for (unsigned i = 0; i < 1024; i++) expected += ptr_exclude_sources(values[i], src, count);
        double old_s = (double)(clock() - begin) / CLOCKS_PER_SEC;
        begin = clock();
        for (int repeat = 0; repeat < 10000; repeat++) {
            ptr_bounds_init(&flat, src, count);
            for (unsigned i = 0; i < 1024; i++) actual += ptr_exclude(values[i], &flat);
        }
        double new_s = (double)(clock() - begin) / CLOCKS_PER_SEC;
        check(expected == actual, "exclusion benchmark output", count);
        benchmark_checksum = expected + actual;
        printf("Synthetic exclusions (%d source arrays/%d active ranges): old %.3f s, flat list+scan %.3f s\n", count, count * 2, old_s, new_s);
    }
}
static void fill_memory(unsigned pattern) {
    uint64_t *p = (void *)memory;
    for (unsigned i = 0; i < 4u * POOL_BYTES / 8; i++) {
        uint32_t r = random32();
        if (!pattern) p[i] = 0;
        else if (pattern == 1) p[i] = (uintptr_t)memory + 3u * POOL_BYTES + 0x8000; // duplicates
        else if (r % 5 < 2) p[i] = (uintptr_t)memory + ((r >> 4) % (4u * POOL_BYTES / 8)) * 8;
        else p[i] = (r % 2) ? (r & 0xffff) : ((uint64_t)r << 32 | random32());
    }
}
int main(void) {
    check_exclusions();
    memory = malloc(4u * POOL_BYTES); if (!memory) abort();
    context.props.limits.maxStorageBufferRange = 1u << 27;
    const uint32_t bounds[] = {0,1,7,8,9,15,16,1023,1024,1025,4095,4096,4097,65535,65536,65537,
                               262143,262144,262145,524288,1048576,POOL_BYTES};
    unsigned cases = 0;
    uint64_t old_resolves = 0, new_resolves = 0;
    for (unsigned pattern = 0; pattern < 4; pattern++) {
        fill_memory(pattern);
        for (unsigned step = 0; step < 4096; step++) {
            p_bind_buf bufs[16] = {0};
            uint64_t size = step < sizeof bounds / sizeof *bounds ? bounds[step] : random32() % (POOL_BYTES + 1u);
            unsigned slot = random32() % 16;
            bufs[slot] = (p_bind_buf){.addr = (uintptr_t)memory, .size = size};
            const p_bind_buf *bound[1] = {bufs};
            uint64_t expected[48] = {0}, actual[48] = {0};
            uint32_t n = (step % 19 == 0) ? 47 : (step % 37 == 0) ? 48 : 0, m = n;
            for (unsigned i = 0; i < n; i++) expected[i] = actual[i] = (uintptr_t)memory + 3u * POOL_BYTES + i * 16;
            unavailable = step % 101 == 0; device_local = step % 3 == 0; readable = step % 103 != 0;
            uint64_t lo = step % 17 == 0 ? 0x1000 : 8, hi = step % 23 == 0 ? 0x10000 : ~0ull;
            sample_count = 0; resolves = 0;
            reference_scan(bufs, 0, lo, hi, expected, &n, bound, 1); old_resolves += resolves;
            Sample reference[16 * 2048]; uint32_t ns = sample_count;
            memcpy(reference, samples, ns * sizeof *samples);
            sample_count = 0; resolves = 0;
            PtrBoundSet excluded; ptr_bounds_init(&excluded, bound, 1);
            PtrFoundSet seen = {0};
            for (uint32_t i = 0; i < m; i++) ptr_found_put(&seen, actual[i]);
            pointer_table_scan_range(bufs, 0, lo, hi, actual, &m, &seen, &excluded); new_resolves += resolves;
            check(n == m && !memcmp(expected, actual, n * sizeof *expected), "scan results", cases);
            check(ns == sample_count && !memcmp(reference, samples, ns * sizeof *samples), "sample order/caps", cases);
            cases++;
        }
    }
    unavailable = device_local = 0; readable = 1; fill_memory(2);
    VkExec *ex = calloc(1, sizeof *ex); if (!ex) abort();
    for (int s = 0; s < 6; s++) for (int i = 0; i < 16; i++) {
        ex->ssbo[s][i] = (p_bind_buf){.addr = (uintptr_t)memory + i * 64, .size = 0x800 + i * 8};
        ex->ubo[s][i] = (p_bind_buf){.addr = (uintptr_t)memory + 0x10000 + i * 64, .size = 0x200};
    }
    uint32_t dims[3] = {7, 11, 13};
    uint32_t roots[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS] = {0};
    uint64_t textures[128] = {0}, images[32] = {0};
    Nvn2C0Layout layout = {.images = 4, .ubos = 8, .ssbos = 8, .textures = 12};
    for (unsigned r = 0; r < NVN2_ROOT_TABLES; r++) {
        layout.roots[r] = 1 + r;
        uint64_t pointer = (uintptr_t)memory + (r % 4) * POOL_BYTES + r * 16;
        memcpy(roots[r], &pointer, 8);
    }
    c0_push = roots; c0_tex = textures; c0_img = images;
    uint64_t bank_old = 0, bank_new = 0;
    for (unsigned pass = 0; pass < 64; pass++) {
        unavailable = pass % 11 == 0; device_local = pass % 5 == 0; readable = pass % 7 != 0;
        c0_lay = pass % 2 ? &layout : NULL;
        MeshPtrTable cache = {0};
        for (int stage = 0; stage < 6; stage++) {
            uint8_t expected[VK_C0_BYTES], actual[VK_C0_BYTES];
            resolves = 0; sample_count = 0;
            vk_c0_fill(expected, stage, ex->ssbo[stage], ex->ubo[stage], dims, pass);
            int mesh = stage == NVN_STAGE_TS || stage == NVN_STAGE_MS;
            if (mesh) c0_pointer_table_mesh(expected, ex->ubo[NVN_STAGE_TS], ex->ubo[NVN_STAGE_MS],
                ex->ssbo[NVN_STAGE_TS], ex->ssbo[NVN_STAGE_MS], ex->ssbo[NVN_STAGE_VS], ex->ssbo[NVN_STAGE_CS]);
            bank_old += resolves; resolves = 0; sample_count = 0;
            c0_fill(actual, stage, ex->ssbo[stage], ex->ubo[stage], dims, pass, !mesh);
            if (mesh) c0_mesh_table(ex, actual, &cache);
            bank_new += resolves;
            check(!memcmp(expected, actual, sizeof expected), "complete c0 bank", pass * 6 + stage);
        }
        ((uint64_t *)memory)[pass * 64] = (uintptr_t)memory + 2u * POOL_BYTES + pass * 16;
        invalidate_pointer_cache();
    }
    uint64_t array_old = 0, array_new = 0;
    for (unsigned step = 0; step < 256; step++) {
        unavailable = step % 11 == 0; device_local = step % 5 == 0; readable = step % 7 != 0;
        int stage = step % 6; p_bind_buf expected[3], actual[3];
        for (unsigned k = 0; k < 3; k++) {
            PtrEntry table[PTR_ENTRIES]; resolves = 0; sample_count = 0;
            if (stage == NVN_STAGE_TS || stage == NVN_STAGE_MS)
                reference_table4(ex->ssbo[NVN_STAGE_TS], ex->ssbo[NVN_STAGE_MS], ex->ssbo[NVN_STAGE_VS], ex->ssbo[NVN_STAGE_CS], table);
            else {
                memset(table, 0, sizeof table); uint64_t found[48]; uint32_t n = 0;
                const p_bind_buf *src[1] = {ex->ssbo[stage]};
                reference_scan(src[0], 0, 8, ~0ull, found, &n, src, 1); pointer_table_finish(found, n, src, 1, table);
            }
            expected[k] = (p_bind_buf){.addr = table[k].addr, .size = table[k].size}; array_old += resolves;
        }
        BindingPtrTable cache = {0}; resolves = 0; sample_count = 0;
        for (unsigned k = 0; k < 3; k++) actual[k] = pointer_binding(ex, stage, PTR_TABLE + k * 16, &cache);
        array_new += resolves;
        check(!memcmp(expected, actual, sizeof expected), "descriptor addresses/extents", step);
        // New binding gets fresh scratch; mutate scanned words and bump the pool generation so the production
        // result cache cannot reuse a prior CB's regions across this content change.
        ((uint64_t *)memory)[step % 128] = (uintptr_t)memory + (step % 4) * POOL_BYTES + step * 16;
        invalidate_pointer_cache();
    }
    uint64_t guarded_old = 0, guarded_new = 0;
    for (unsigned pass = 0; pass < 128; pass++) {
        unavailable = pass % 13 == 0; device_local = pass % 5 == 0; readable = pass % 7 != 0;
        fix_c0_enabled = pass % 11 != 0;
        c0_lay = pass % 17 ? &layout : NULL;
        c0_dim[0] = pass + 1; c0_dim[1] = pass + 3; c0_dim[2] = pass + 7; c0_binst = pass;
        for (unsigned r = 0; r < NVN2_ROOT_TABLES; r++) {
            layout.roots[r] = pass % 5 ? 1 + r : 0; // absent root tables are valid
            roots[r][0] ^= random32();
        }
        for (unsigned b = 0; b < 16; b++) {
            ex->ubo[NVN_STAGE_CS][b].size = (pass + b) % 3 ? 0x200 : 0;
            ex->ssbo[NVN_STAGE_CS][b].size = (pass + b) % 4 ? 0x800 + b * 8 : 0;
        }
        ex->ssbo[NVN_STAGE_CS][pass % 16].addr = (uintptr_t)memory + ((pass * 64) % 0x4000);
        ((uint64_t *)memory)[pass % 128] = (uintptr_t)memory + (pass % 4) * POOL_BYTES + pass * 32;
        invalidate_pointer_cache();
        GuardedC0Bank cache; cache.valid = 0; // a fresh descriptor-build scope after the source mutations above
        for (unsigned off = 0; off < VK_C0_BYTES + 16; off += 4) {
            GfxShaderRes resource = {.cbuf_slot = 0, .handle = off / 4};
            sample_count = 0; c0_fills = 0;
            p_bind_buf expected = reference_guarded_storage_binding(ex, NVN_STAGE_CS, &resource, &cache);
            guarded_old += c0_fills; c0_fills = 0; sample_count = 0;
            p_bind_buf actual = guarded_storage_binding(ex, NVN_STAGE_CS, &resource, &cache);
            guarded_new += c0_fills;
            check(!memcmp(&expected, &actual, sizeof actual), "guarded address and declared extent", pass * VK_C0_BYTES + off);
        }
        if (cache.valid) {
            uint8_t expected_bank[VK_C0_BYTES]; sample_count = 0;
            vk_c0_fill(expected_bank, NVN_STAGE_CS, ex->ssbo[NVN_STAGE_CS], ex->ubo[NVN_STAGE_CS], c0_dim, c0_binst);
            check(!memcmp(expected_bank, cache.bank, VK_C0_BYTES), "guarded complete bank/new build", pass);
        } else {
            check(!fix_c0_enabled || !c0_lay, "lazy guarded bank", pass);
        }
    }
    printf("PASS: %u scan cases (order, caps, early stops, duplicates, absent/unreadable backing); 384 complete c0 banks (native layouts and shared task/mesh tables); 256 three-entry descriptor arrays; 74240 guarded candidates across 128 fresh builds (source changes, absent roots, zero sizes)\n", cases);
    printf("Synthetic resolver calls: scan old=%llu optimized=%llu; mesh-bank old=%llu optimized=%llu; descriptor-array old=%llu optimized=%llu\n",
        (unsigned long long)old_resolves, (unsigned long long)new_resolves,
        (unsigned long long)bank_old, (unsigned long long)bank_new,
        (unsigned long long)array_old, (unsigned long long)array_new);
    printf("Synthetic guarded bank builds: old=%llu optimized=%llu\n", (unsigned long long)guarded_old, (unsigned long long)guarded_new);
    free(ex); free(memory); return 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default=shutil.which("clang") or "clang")
    ap.add_argument("--vma-include", default="C:/msys64/ucrt64/include/vma")
    args = ap.parse_args()
    draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
    helpers = draw[draw.index("enum { PTR_TABLE ="):draw.index("static int c0_on(void)")]
    helpers = helpers.replace("    c0_fill(bank, stage, ssbo, ubo, dispatch, base_instance, 1);", "    c0_fills++;\n    c0_fill(bank, stage, ssbo, ubo, dispatch, base_instance, 1);")
    # Keep the real acceptance rules, and record every attempted sample to compare visit order independently of hits.
    helpers = helpers.replace("    uint32_t n = *n_io;\n    if (n >= 48", "    sample_note(v, slot, off);\n    uint32_t n = *n_io;\n    if (n >= 48", 1)
    c0table = draw[draw.index("static void c0_table("):draw.index("// Hookshot vk_bank.cpp pointer_table:")]
    scan = helpers[helpers.index("static void pointer_table_scan_range("):helpers.index("static void pointer_table_finish(")]
    reference_scan = scan.replace("pointer_table_scan_range", "reference_scan", 1)
    reference_scan = reference_scan.replace("PtrFoundSet *seen, const PtrBoundSet *bound)",
                                            "const p_bind_buf *bound[], int nbound)", 1)
    begin = reference_scan.index("        // Dense head:")
    end = reference_scan.index("        if ((*n_io > before", begin)
    reference_scan = reference_scan[:begin] + OLD_ENUM + reference_scan[end:]
    reference_scan = reference_scan.replace("bufs[i].size, no,", "bufs[i].size, probed,")
    try_add = helpers[helpers.index("static inline void pointer_table_try_add("):helpers.index("static void pointer_table_scan_range(")]
    reference_try = try_add.replace("pointer_table_try_add", "reference_try_add", 1)
    reference_try = reference_try.replace("static inline void reference_try_add", "static void reference_try_add", 1)
    reference_try = reference_try.replace("PtrFoundSet *seen, const PtrBoundSet *bound, int *scan_log)",
                                          "const p_bind_buf *bound[], int nbound, int *scan_log)", 1)
    begin = reference_try.index("    // A pointer already collected")
    end = reference_try.index("    if (scan_log &&", begin)
    reference_try = reference_try[:begin] + r'''    if (ptr_exclude_sources(v, bound, nbound)) return;
    VkMemSpan hit;
    if (!vk_mem_resolve(v, 1, &hit)) return;
    for (uint32_t j = 0; j < n; j++) if (found[j] == v) return;
''' + reference_try[end:]
    reference_try = reference_try.replace("    ptr_found_put(seen, v);\n", "")
    binding = draw[draw.index("typedef struct { PtrEntry e[PTR_ENTRIES]; int valid; }"):draw.index("static VkDescriptorBufferInfo ssbo_info(")]
    mesh_cache = draw[draw.index("typedef struct { uint8_t bytes[16u * PTR_ENTRIES]; int valid; }"):draw.index("// c0 from replay scratch;")]
    guarded = draw[draw.index("typedef struct { uint8_t bank[VK_C0_BYTES]; int valid; }"):draw.index("// Explicit NVN 2 compute storage-buffer table slots")]
    reference_guarded = guarded[guarded.index("static p_bind_buf guarded_storage_binding("):]
    reference_guarded = reference_guarded.replace("guarded_storage_binding", "reference_guarded_storage_binding", 1)
    reference_guarded = reference_guarded.replace("    uint64_t off =", "    (void)scratch;\n    uint64_t off =", 1)
    reference_guarded = reference_guarded.replace("        const uint8_t *bank = guarded_c0_bank(ex, scratch);", r'''        uint8_t bank[VK_C0_BYTES];
        vk_c0_fill(bank, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);''')
    source = HARNESS.replace("/* PRODUCTION_HELPERS */", c0table + helpers).replace("/* REFERENCE_TRY_ADD */", reference_try)
    source = source.replace("/* REFERENCE_SCAN */", reference_scan).replace("/* POINTER_BINDING */", binding)
    source = source.replace("/* MESH_BANK_CACHE */", mesh_cache)
    source = source.replace("/* GUARDED_C0 */", guarded).replace("/* REFERENCE_GUARDED_C0 */", reference_guarded)
    with tempfile.TemporaryDirectory(prefix="mkw-pointer-tables-") as tmp:
        src = Path(tmp) / "check.c"
        exe = Path(tmp) / "check.exe"
        src.write_text(source)
        subprocess.run([args.cc, "-O2", "-Wall", "-Wextra", "-I", str(ROOT / "runtime"),
                        "-I", args.vma_include, str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()

// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to translated shaders (the translator's answers come from the bound state), asks vk_pipeline.c for the
// pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic state, vertex and index
// buffers and the draw. A dispatch ends the pass and does the same for the compute shader. UPDATE_UB is a buffer copy
// from transient memory between full barriers.
//
// Per command buffer the replay keeps transient memory (support buffers, copied uniform ranges, generated indices),
// descriptor pools, image views and the storage-buffer ranges the GPU may write; all of it is freed, and the ranges are
// reported with vk_gpu_written, from one completion callback once the submission's fence is done (the write watch
// does not see GPU writes). Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_geometry_control.h"
#include "nvn_indirect_reset.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define MAX_BINDINGS 128
// Last real fragment program across command buffers (mesh CBs often omit FS; see OP_BIND_PROGRAM).
static const prog_t *g_last_fs_prog;

// Header-only MeshPrimitiveHeader SPH: only raw NVN2 controls use +0x38 as Ampere code length.
// Lowered Maxwell / TKGSMETA often have +0x38==0 with a real FS at host slot 1 — those are NOT
// header-only (n249: never remembered FS because every Maxwell FS looked empty at +0x38).
static int control_is_header_only(const uint8_t *ctl, uint32_t flen_704) {
    if (!ctl) return 0;
    uint16_t nvn = 0;
    uint32_t nvn_len = 0;
    memcpy(&nvn, ctl + 0x56, 2);
    memcpy(&nvn_len, ctl + 0x38, 4);
    if (nvn > 4) return 0;
    if (!gfx_shader_is_nvn2(ctl)) return 0; // Maxwell / lowered: real shader
    (void)flen_704;
    return nvn_len <= 0x50u + 16u;
}
// NVN 2 compute parameter layout of the program being dispatched (gfx_shader_c0_layout), NULL for NVN 1: read by
// vk_c0_fill, set by the dispatch ops around bind_resources like c0_dim.
static const Nvn2C0Layout *c0_lay;
static const uint32_t (*c0_push)[NVN2_ROOT_WORDS];
static const uint64_t *c0_tex, *c0_img;
#define MAX_DESC 1024
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 1048576u   // 1 MiB dummy buffer (was 64 KiB) to safely absorb unbacked/out-of-bounds GPU buffer stores
#define SUPPORT_BYTES 1408u   // Ryujinx SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_TS 3
#define NVN_STAGE_MS 4
#define NVN_STAGE_CS 5
static uint8_t geometry_input;                              // normalized actual draw primitive; zero outside opt-in GS
#define ONESHOT_WAIT_MS 1500u   // prepare_draw: capped to 1.5s to prevent Windows TDR (2s watchdog timeout) crashes

int vk_dump_active(void);   // vk_dump.c: the replay is in the dump frame
const char *vk_dump_dir(void);
static void c0_lay_set(VkExec *ex, const GfxShader *cs) {
    c0_lay = gfx_shader_c0_layout(cs);
    c0_push = ex->push[1];
    c0_tex = ex->tex[NVN_STAGE_CS];
    c0_img = ex->img[NVN_STAGE_CS];
}

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_FAULT_SKIP, W_FAULT_SKIP2, W_SKIP_VS, W_SKIP_FS, W_RESET, W_GEOMETRY, W_LIST_RESTART, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

// ---- resources of one command buffer, freed by its completion callback

typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 40, "ViewKey is hashed as bytes: no padding");
typedef struct { uint64_t addr, size; } Range;
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
    uint32_t *written_slot;              // note_written: open-addressed index of written, entry = index + 1, 0 empty
    uint32_t written_slots;              // a power of two
} CbRes;

static CbRes *cur;
static uint64_t cur_fence;

// Descriptor pools and full-size transient chunks go back to these lists when their command buffer completes, instead
// of being destroyed and created again for the next one: each creation is a driver memory allocation (on Windows an
// allocation, a map and a paging-fence wait on the CPU), about ten of them per frame inside the replay.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 16
static PlatMutex spare_lock = PLAT_MUTEX_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
static struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[6][VK_C0_BYTES];          // vertex, fragment, compute, geometry, task, mesh
    VkBuffer c0_buf[6];
    VkDeviceSize c0_off[6];
    int c0_valid[6];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; Nvn2C0Layout lay; uint8_t has_lay;
             uint32_t roots[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS]; uint64_t tex[128], img[32]; } c0_in[6];
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
} B;

// Contents of B.ds_last: the set's packed descriptors (VkDescSlot), which with its layout determine every write. A draw
// whose allocated set would hold the same reuses the set instead of allocating and writing one (a per-draw
// cost). psig: the same for the last push per set number; a push equal to it under the same layout is not recorded
// again (the descriptors stay bound across pipeline binds of that layout).
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

// TOTK_BIND_STATS counters (bind_stats).
static struct {
    int on;
    uint64_t draws, bind[2], desc[2], push, push_skip, set_new, set_reuse, ac_miss, same_prog, same_all, bind_skip;
} BS = {-1};

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
static struct {
    int on;                              // TOTK_BIND_SKIP: 1 skip (default), 2 check, 0 off
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[32];                      // generations, descriptor tables, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N], pcnt[2 * VK_MCLS_N];
    int ncnt, pncnt, cnt_over;
    int cmds;                            // check mode: pushes, binds and set writes the build recorded
    uint32_t ckn;                        // check mode: the last build's packed descriptors (bytes in ck)
    uint8_t ck[DS_SIG_BYTES];
} RS = {-1};

// The bound-slot bytes of VkExec a pipeline's descriptors read, as ranges merged where they touch, built once per
// pipeline info: c0's storage table, the uniform and storage ranges of the bound slots and the texture and image
// handles. sup and c0: the support inputs and the base instance count too. p: storage buffers whose address the shader
// loads from a user uniform block; the address read from the block counts too. ok 0: never skipped (too many
// descriptors or such buffers).
#define RS_PTRS 4
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static const VkPipeInfo *rs_last;        // the last lookup
static const RsList *rs_last_l;

static uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
static uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++) vk_gpu_written(r->written[i].addr, r->written[i].size);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        plat_mutex_lock(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        plat_mutex_lock(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->written_slot), free(r->pools), free(r->chunks), free(r);
}

// Pointer-table result cache lifetime. Bumped on a new command buffer so soft guest writes that do not
// move pool/usnap gens cannot reuse a prior CB's regions across menu/submit boundaries.
static uint64_t ptr_cb_epoch = 1;

void vk_exec_draw_new_cb(void) {
    memset(&B, 0, sizeof B);
    ptr_cb_epoch++;
}

static CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            plat_mutex_lock(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            plat_mutex_unlock(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static uint32_t written_hash(uint64_t addr, uint64_t size) {
    uint64_t h = (addr ^ size * 0xc2b2ae3d27d4eb4full) * 0x9e3779b97f4a7c15ull;
    return (uint32_t)(h >> 32);
}

// Each distinct range once per command buffer, in first-write order. A scene with thousands of written ranges and tens
// of thousands of draws binding them spent most of the replay searching this list, so the search goes through a table.
static void note_written(VkExec *ex, uint64_t addr, uint64_t size) {
    CbRes *r = cb_res(ex);
    if ((uint32_t)r->nwritten * 2 >= r->written_slots) {
        uint32_t slots = r->written_slots ? r->written_slots * 2 : 256;
        uint32_t *table = calloc(slots, sizeof *table);
        if (!table) rt_fatal("vk: out of memory");
        for (int i = 0; i < r->nwritten; i++) {
            uint32_t at = written_hash(r->written[i].addr, r->written[i].size) & (slots - 1);
            while (table[at]) at = (at + 1) & (slots - 1);
            table[at] = (uint32_t)i + 1;
        }
        free(r->written_slot);
        r->written_slot = table, r->written_slots = slots;
    }
    uint32_t mask = r->written_slots - 1, at = written_hash(addr, size) & mask;
    for (; r->written_slot[at]; at = (at + 1) & mask) {
        const Range *w = &r->written[r->written_slot[at] - 1];
        if (w->addr == addr && w->size == size) return;
    }
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size};
    r->written_slot[at] = (uint32_t)r->nwritten;
    vk_mem_gpu_pending(addr, size);
}

static int reset_mapped(void *context, uint64_t addr, uint32_t size) {
    (void)context;
    VkMemSpan span;
    return vk_mem_resolve(addr, size, &span);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// Whether two nonnull targets share a byte (duplicates included). Their copies then need a barrier between them to
// keep the input order; disjoint targets can all be copied after one barrier.
static int reset_targets_overlap(const uint64_t *pointers, uint32_t count) {
    uint64_t *sorted = malloc((size_t)count * sizeof *sorted);
    if (!sorted) rt_fatal("vk: out of memory checking indirect-reset targets");
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++)
        if (pointers[i]) sorted[n++] = pointers[i];
    qsort(sorted, n, sizeof *sorted, cmp_u64);
    int overlap = 0;
    for (uint32_t i = 1; i < n && !overlap; i++) overlap = sorted[i] - sorted[i - 1] < NVN_INDIRECT_RESET_RECORD_BYTES;
    free(sorted);
    return overlap;
}

// The pointer list may have been written earlier in this submission. Read its GPU-visible contents after submitting
// that prefix, then validate every target before recording a write. Null pointers have no record to reset.
int vk_exec_reset_indirect(VkExec *ex, uint64_t list_addr, uint32_t descriptor_bytes, int64_t first, int64_t count) {
    NvnIndirectResetList list;
    if (!nvn_indirect_reset_list(list_addr, descriptor_bytes, first, count, &list)) return 0;
    uint64_t *pointers = malloc(list.bytes);
    if (!pointers) rt_fatal("vk: out of memory reading an indirect-reset pointer list");
    if (!vk_exec_read_buffer(ex, list.begin, list.bytes, pointers) ||
        !nvn_indirect_reset_targets(&list, pointers, reset_mapped, NULL)) {
        free(pointers);
        return 0;
    }
    VkBuffer source;
    VkDeviceSize source_offset;
    static const uint32_t record[5] = {0, 1, 0, 0, 0};
    memcpy(scratch(ex, sizeof record, &source, &source_offset), record, sizeof record);
    CbRes *resources = cb_res(ex);
    Chunk *chunk = &resources->chunks[resources->nchunks - 1];
    if (vmaFlushAllocation(vk_ctx()->vma, chunk->alloc, source_offset, sizeof record) != VK_SUCCESS)
        rt_fatal("vk: flushing indirect-reset values failed");
    int ordered = reset_targets_overlap(pointers, list.count);
    vk_exec_barrier(ex);   // the copies follow every earlier access to the records
    for (uint32_t i = 0; i < list.count; i++) {
        uint64_t addr = pointers[i];
        if (!addr) continue;
        VkMemSpan span;
        if (!vk_mem_resolve(addr, sizeof record, &span))
            rt_fatal("vk: an indirect-reset mapping changed during replay");
        // Commit first-use imports to their mirror before choosing the destination. A later indirect bind must
        // read this same backing rather than create a mirror from the old contents.
        VK_MEM_USE(span);
        if (ordered) vk_exec_barrier(ex);   // preserves the input order of duplicate or overlapping target ranges
        VkBufferCopy copy = {source_offset, span.offset, sizeof record};
        vkCmdCopyBuffer(ex->cb, source, span.buffer, 1, &copy);
        ex->recorded++;
        vk_mem_gpu_wrote(&span, sizeof record);
        vk_usnap_drop(addr, sizeof record);
        vk_mem_count(span.cls, VK_MUSE_GPU_WRITE, sizeof record);
        note_written(ex, addr, sizeof record);
    }
    free(pointers);
    vk_exec_barrier(ex);   // subsequent shader accesses and indirect reads see every completed record
    return 1;
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };

static struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBufferView zview;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL by the upload batch (which reaches the queue before
// the submission that uses it). *sview is the view a storage descriptor takes (a cube becomes a 2D array).
static void make_dummy(int i, VkFormat fmt, VkImage *image, VmaAllocation *alloc, VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | (ms ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (GfxShaderRes.format: Ryujinx TextureFormat order); Vulkan wants
// the view format to match the declaration. Created on first use.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);

// Releases the stand-ins and the depth twins. The device must be idle; the game path leaves them to process exit like
// the pipelines.
void vk_exec_draw_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    twins_release(c);
    for (int i = 0; Z.ready && i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    cur = NULL, cur_fence = 0;
    vk_usnap_shutdown();
    if (Z.ready) {
        vkDestroyBufferView(c->device, Z.zview, NULL);
        vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    }
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// Numeric type, byte size and optimal-tiling features for sampled views and integer/sRGB view reinterpretation.
static int vk_fmt_num_type(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_S8_UINT:
        return GFX_TYPE_UINT;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_B8G8R8A8_SINT:
        return GFX_TYPE_SINT;
    default: return GFX_TYPE_FLOAT;
    }
}

static int vk_fmt_is_compressed(VkFormat f) {
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}

static uint32_t vk_fmt_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB: case VK_FORMAT_S8_UINT:
        return 1;
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_A4B4G4R4_UNORM_PACK16:
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
        return 2;
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return 4;
    case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        return 16;
    default: return 0;
    }
}

static VkFormatFeatureFlags sampled_features_of(VkFormat f) {
    static VkFormatFeatureFlags known[256];
    static uint8_t have[256];
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    if ((unsigned)f < 256 && have[f]) return known[f];
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(c->phys, f, &p);
    if ((unsigned)f < 256) known[f] = p.optimalTilingFeatures, have[f] = 1;
    return p.optimalTilingFeatures;
}

static VkFormat sampled_view_format(const VkTexImage *img, const view_t *v, int hv, int storage, TexVkFormat *vf_out,
                                    int *used_vf) {
    if (vf_out) memset(vf_out, 0, sizeof *vf_out);
    if (used_vf) *used_vf = 0;
    VkFormat fmt = img->format;
    if (storage) {
        if (fmt == VK_FORMAT_R8G8B8A8_SRGB) fmt = VK_FORMAT_R8G8B8A8_UNORM;
        else if (fmt == VK_FORMAT_B8G8R8A8_SRGB) fmt = VK_FORMAT_B8G8R8A8_UNORM;
        return fmt;
    }
    if (hv && v->set_format && !img->block_views && (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        TexVkFormat vf;
        if (tex_vk_format(v->format, &vf) && !vf.depth && !vf.stencil && !vf.expand_rgb) {
            if (vf.astc_bw) {
                if (fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_SRGB)
                    fmt = vf.astc_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                else if (fmt == VK_FORMAT_BC7_UNORM_BLOCK || fmt == VK_FORMAT_BC7_SRGB_BLOCK)
                    fmt = vf.astc_srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
                else if (fmt == VK_FORMAT_BC3_UNORM_BLOCK || fmt == VK_FORMAT_BC3_SRGB_BLOCK)
                    fmt = vf.astc_srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
                else if (fmt == vf.vk)
                    fmt = vf.vk;
            } else if (!vf.astc_bw && !vk_fmt_is_compressed(fmt) && !vk_fmt_is_compressed(vf.vk) &&
                       vk_fmt_bytes(fmt) && vk_fmt_bytes(fmt) == vk_fmt_bytes(vf.vk)) {
                fmt = vf.vk;
                if (vf_out) *vf_out = vf;
                if (used_vf) *used_vf = 1;
            }
        }
    }
    return fmt;
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    int shape;
    if (hv && v->set_target) shape = vk_pipeline_tex_type(v->target);
    else {
        int ms = img->samples > 1;
        switch (img->view_type) {
        case VK_IMAGE_VIEW_TYPE_1D: shape = GFX_TEX_1D; break;
        case VK_IMAGE_VIEW_TYPE_1D_ARRAY: shape = GFX_TEX_1D_ARRAY; break;
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY: shape = ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY; break;
        case VK_IMAGE_VIEW_TYPE_3D: shape = GFX_TEX_3D; break;
        case VK_IMAGE_VIEW_TYPE_CUBE: shape = GFX_TEX_CUBE; break;
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: shape = GFX_TEX_CUBE_ARRAY; break;
        default: shape = ms ? GFX_TEX_2D_MS : GFX_TEX_2D; break;
        }
    }
    if (vk_exec_view_aspect(img, v, hv) == VK_IMAGE_ASPECT_STENCIL_BIT) return shape | GFX_TEX_UINT;
    VkFormat vfmt = sampled_view_format(img, v, hv, 0, NULL, NULL);
    int nt = vk_fmt_num_type(vfmt);
    if (nt == GFX_TYPE_UINT) shape |= GFX_TEX_UINT;
    else if (nt == GFX_TYPE_SINT) shape |= GFX_TEX_SINT;
    return shape;
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A view in STENCIL mode samples the stencil plane (TextureViewSetDepthStencilMode); otherwise depth wins. The
// translator declares every sampled image float and the shader takes the raw bits, as for integer formats (nxnative
// nvn_mtl.m).
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime: made per command buffer they were created again
// every frame (vkCreateImageView took about 8% of the fight replay on the 7900X). Used under the texture lock, as
// vk_exec_image_gone is (destroy_tex), which hands the views of a destroyed image to the deferred release with it.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
static VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    (void)ex;
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return VK_NULL_HANDLE;
    if (storage && !(img->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return VK_NULL_HANDLE;
    // CompactLayers: guest mip img->host_mip0 is host level 0; clamp guest levels into the trimmed image.
    uint32_t guest_l0 = hv && v->set_levels ? v->base_level : 0;
    uint32_t l0 = guest_l0 >= img->host_mip0 ? guest_l0 - img->host_mip0 : 0;
    if (l0 >= img->levels) return VK_NULL_HANDLE;
    uint32_t guest_nl = hv && v->set_levels && v->nlevels ? v->nlevels : 0;
    uint32_t nl = guest_nl ? guest_nl : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return VK_NULL_HANDLE;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return VK_NULL_HANDLE;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    TexVkFormat vf;
    int used_vf = 0;
    VkFormat fmt = sampled_view_format(img, v, hv, storage, &vf, &used_vf);
    if (storage && !(sampled_features_of(fmt) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) return VK_NULL_HANDLE;
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // MNG nvn_mtl.m tex_swizzle_sel: view selectors replace the texture's, then the format remap
        int alpha_one = used_vf ? vf.alpha_one : img->alpha_one;
        const uint8_t *chan = used_vf ? vf.chan : img->chan;
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (alpha_one || chan[0])
            for (int i = 0; i < 4; i++) {
                if (alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
    ViewKey k = {img->image, fmt, vt, aspect, l0, nl, y0, ny, swz};
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = fmt,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = img->block_views ? VK_IMAGE_USAGE_SAMPLED_BIT
                                                                : (img->usage & ~VK_IMAGE_USAGE_STORAGE_BIT)};
    // Restrict view usage for block-view BC images, sampled views over STORAGE images (sRGB views lack STORAGE_IMAGE_BIT),
    // or format-reinterpreted storage views (e.g. R32_UINT over R8G8B8A8_SRGB, where SAMPLED/ATTACHMENT bits may be invalid).
    if (img->block_views || (!storage && (img->usage & VK_IMAGE_USAGE_STORAGE_BIT))) ci.pNext = &use;
    else if (storage && fmt != img->format) {
        use.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        ci.pNext = &use;
    }
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, &k, (void *)view);
    return view;
}

// An attachment view of a registered texture, kept with its sampled views for the image's lifetime: made per command
// buffer, every render target's views were created again each frame (vkCreateImageView under vk_exec_pass_begin, 2%
// of the fight replay on the 7900X). Attachment keys carry a swizzle word no sampled key has (those hold four NVN
// selectors, each below 8). The texture lock guards the tables, as for get_view; vk_exec_pass_begin may already hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    ViewKey k = {img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu};
    vk_texture_lock();
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView view = (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                    .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
        VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        if (blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
        // Attachment views never use storage and sRGB images carry mutable storage usage from vk_texture.c create_tex.
        else if (img->usage & VK_IMAGE_USAGE_STORAGE_BIT)
            use.usage = img->usage & ~VK_IMAGE_USAGE_STORAGE_BIT, ci.pNext = &use;
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
            rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        ViewNode *n = malloc(sizeof *n);
        if (!n) rt_fatal("vk: out of memory");
        *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
        nvn_hm_put(&views_by_image, &img->image, n);
        nvn_hm_put(&view_of, &k, (void *)view);
    }
    vk_texture_unlock();
    return view;
}

void vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    if (!views_by_image.klen) return;
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next) {
        next = n->next;
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        uint32_t st;
        memcpy(&st, c + 0x714, 4);
        if ((int)st != nvn_stage) continue;
        memcpy(len, c + 0x704, 4);
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

static int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st;
            memcpy(&st, c + 0x714, 4);
            if (st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, memcpy(&m->len[st], c + 0x704, 4);
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under: a slot is looked up again, a hash probe, only when the handle changed or its slot may resolve
// otherwise (slot_still).
static struct { uint64_t context, tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32]; } AC[6];
static uint64_t descriptor_serial = 1;

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(const VkExec *ex, uint64_t h) {
    return h ? vk_texture_slot_token(&ex->tex_pool, (int)(h & 0xfffff)) : 1;
}

static void descriptor_state(VkExec *ex);

// Whether what handle h resolved to under the global generation *egen and its slot token etok (both read before that
// lookup) still holds at global generation gen: nothing at all changed since, or not its slot (the token, read only when
// the global generation moved, about every 16 draws). A token hit moves *egen to gen.
static int slot_still(const VkExec *ex, uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(ex, h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}
static int shader_memo_on(void);

// The slots of (program, stage, GS input topology) whose types its translation reads (gfx_shader_slot_mask), learned from its first
// shader; ~0u while unknown. The types of other slots select the same shader, so answers_for leaves them zero instead
// of looking up all 64 handles. Direct-mapped by serial and stage; a collision only relearns.
#define SLOTM_N 4096
static struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static int slot_mask_stage(int st) { return st == NVN_STAGE_GS ? st + 8 * geometry_input : st; }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    st = slot_mask_stage(st);                           // GS topology can change which texture queries survive translation
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    st = slot_mask_stage(st);
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

static uint8_t type_of_handle(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv) : NULL;
    return img ? (uint8_t)gfx_type_of(img, &v, hv) : 0;
}

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    descriptor_state(ex);
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
    else if (st == NVN_STAGE_GS) a->geometry_input = geometry_input;
    else if (st == NVN_STAGE_CS)
        for (int k = 0; k < 3; k++) {
            uint32_t l;
            memcpy(&l, ctl + 0x748 + 4 * k, 4);
            a->local[k] = (uint16_t)(l ? l : 1);
        }
    typeof(AC[0]) *c = &AC[st];
    if (c->context != descriptor_serial) {
        memset(c, 0, sizeof *c);
        c->context = descriptor_serial;
    }
    // The generation and tokens are read before the lookups (one that creates an entry changes them) and unlocked: a
    // registration racing with them is ordered after this draw, and a slot stored under older ones is only looked up
    // again.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0, check = shader_memo_on() == 2;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(ex, &c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(ex, &c->igen[slot], c->itok[slot], hi, gen);
        uint64_t tt = tmiss ? token_of(ex, h) : 0, it = imiss ? token_of(ex, hi) : 0;
        if ((tmiss || imiss || check) && !locked) vk_texture_lock(), locked = 1, BS.ac_miss++;
        if (tmiss) c->tt[slot] = type_of_handle(ex, h), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss) c->it[slot] = type_of_handle(ex, hi) & GFX_TEX_MASK, c->img[slot] = hi, c->itok[slot] = it, c->igen[slot] = gen;
        if (check && (type_of_handle(ex, h) != c->tt[slot] || (type_of_handle(ex, hi) & GFX_TEX_MASK) != c->it[slot]))   // MKW_SHADER_MEMO=check
            rt_fatal("vk_exec: kept texture types of stage %d slot %d differ from a lookup", st, slot);
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot];
    }
    if (locked) vk_texture_unlock();
}

// MKW_NATIVE_GS runs lowered native passthrough geometry stages through the portable path (issue #45). On by default
// (runtime/cfg.c): without it the destructible terrain draws black; MKW_NATIVE_GS=0 turns it off.
// Eglantine takes the varyings from the hardware skip mask (Maxwell register 0x1240, translate_program.cpp,
// LowerGeometryPassthrough); the host does not have that mask for NVN 2 programs, so this infers it: what the vertex
// stage writes and the fragment stage reads. That is enough for a fragment stage that reads nothing, as the terrain's
// does; a stage whose real mask differs would forward the wrong set.
static int native_gs_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_NATIVE_GS");
    return on;
}

static const GfxShader *native_gs_vs, *native_gs_fs;   // the draw's stages, set before its geometry lookup

static const uint8_t *native_gs_control(const uint8_t *ctl, const uint8_t *code, uint32_t len) {
    // nvn2_lower.cpp::nvn2_lower keeps these source controls for the process lifetime. Keep the packed copies
    // just as long, including after the first 256 contexts; shader_for borrows them even on shader-cache hits.
    static struct { const uint8_t *ctl; uint32_t mask; uint8_t *copy; } *made;
    static size_t count, capacity;
    GfxShaderIo io[32];
    const uint32_t fs_in = gfx_shader_inputs(native_gs_fs, io), vs_out = gfx_shader_outputs(native_gs_vs, io);
    uint32_t mask = fs_in & vs_out;
    if (vk_caps.amd_sc && nvn2_origin(code) == UINT64_C(0x3dbaaf649cb5f927))
        mask |= (UINT32_C(1) << 0) | (UINT32_C(1) << 19);
    for (size_t i = 0; i < count; i++)
        if (made[i].ctl == ctl && made[i].mask == mask) return made[i].copy;
    uint32_t skip[8];
    skip[0] = 0x0fffffffu;
    for (unsigned i = 1; i < 8; i++) skip[i] = UINT32_MAX;
    for (unsigned l = 0; l < 32; l++)
        if (mask >> l & 1) skip[1 + l / 8] &= ~(15u << (l % 8 * 4));
    uint8_t *copy = malloc(0x880);
    if (!copy) return NULL;
    memcpy(copy, ctl, 0x880);
    if (!nvn_geometry_control_pack_lowered(copy, 0x880, code, len, skip)) {
        rt_log("native GS: stage %016llx does not fit the portable control", (unsigned long long)nvn2_origin(code));
        free(copy);
        return NULL;
    }
    if (count == capacity) {
        if (capacity > SIZE_MAX / 2) { free(copy); return NULL; }
        size_t next = capacity ? capacity * 2 : 256;
        if (next > SIZE_MAX / sizeof *made) { free(copy); return NULL; }
        void *grown = realloc(made, next * sizeof *made);
        if (!grown) { free(copy); return NULL; }
        made = grown, capacity = next;
    }
    made[count].ctl = ctl, made[count].mask = mask, made[count].copy = copy;
    count++;
    rt_log("native GS: stage %016llx forwards locations %#x (vertex writes %#x, fragment reads %#x)",
           (unsigned long long)nvn2_origin(code), mask, vs_out, fs_in);
    return copy;
}

// *present: the program has this stage. Returns NULL when it has not or the translation failed. pending (graphics
// stages, vk_pipeline_async_on): a shader not translated yet is queued and NULL returned with *pending = 1. await: a
// shader queued earlier is waited for (gfx_shader_await) instead.
static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    if (st == NVN_STAGE_GS && native_gs_on() && nvn2_origin(code)) {
        if (!native_gs_vs || !native_gs_fs) {   // its varyings follow from the other stages, which are still pending
            if (pending) *pending = 1;
            return NULL;
        }
        if (!(ctl = native_gs_control(ctl, code, len))) return NULL;
    }
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS :
             st == NVN_STAGE_GS ? GFX_STAGE_GS : st == NVN_STAGE_CS ? GFX_STAGE_CS : -1;
    if (gs < 0) return NULL;
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && vk_pipeline_async_on() ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS :
           gfx_stage == GFX_STAGE_GS ? NVN_STAGE_GS : gfx_stage == GFX_STAGE_CS ? NVN_STAGE_CS :
           gfx_stage == GFX_STAGE_TASK ? NVN_STAGE_TS : gfx_stage == GFX_STAGE_MESH ? NVN_STAGE_MS : -1;
}

static int c0_stage_index(int st) {
    switch (st) {
    case NVN_STAGE_VS: return 0;
    case NVN_STAGE_FS: return 1;
    case NVN_STAGE_CS: return 2;
    case NVN_STAGE_GS: return 3;
    case NVN_STAGE_TS: return 4;
    case NVN_STAGE_MS: return 5;
    default: return -1;
    }
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing (a pool's VRAM mirror when it has one) when
// aligned, a copy in transient memory when the backing is host memory with an unaligned offset. 0 when neither works.
// written: the shader stores into it; writeback: schedule a mirror -> host copy at the end of the submission.
// direct_only: guarded aliases must share the real backing; a transient copy would lose writes and alias coherence.
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      int writeback, int direct_only, VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    // Bare+written with no size used to take sp.avail through the end of a multi-GiB pool and
    // write the whole remainder back every submit (~1 GiB observed on 0x1a9c8d000+0x94400000).
    if (written && !size && n > (1ull << 20)) n = 1ull << 20;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (writeback) vk_mem_gpu_wrote(&sp, n);
        if (written) vk_usnap_drop(addr, n);
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc || direct_only) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// Hookshot vk_bank.cpp pointer_table: task/mesh (and some compute) programs chase GPU addresses kept inside their
// small storage buffers. Eden's AddGlobalTables exposes up to three pool regions at c0+0x880 so GlobalToStorage can
// search them. Without this, skinned/instanced mesh reads return 0 and models collapse.
enum { PTR_TABLE = 0x880u, PTR_ENTRIES = 3u };
typedef struct { uint64_t addr, size; } PtrEntry;

static int ptr_bound(const p_bind_buf ssbo[16], uint64_t v) {
    for (int i = 0; i < 16; i++)
        if (ssbo[i].addr && v >= ssbo[i].addr && v < ssbo[i].addr + ssbo[i].size) return 1;
    return 0;
}

// Collect GPU pointers from NVN storage buffers (bones / instances / cull lists).
// Hookshot scanned the first 0x100 of tiny pointer blocks. MKW title foliage often keeps those
// blocks empty at bind (GPU-filled later) while the large SSBOs hold instance bases deeper than
// the first 512 bytes — a head-only scan left found=0 and GlobalToStorage only covered the
// fallback SSBO windows, so tree/grass pointers outside those windows read as 0.
static int ptr_exclude_sources(uint64_t v, const p_bind_buf *const bound[], int nbound) {
    for (int b = 0; b < nbound; b++)
        if (bound[b] && ptr_bound(bound[b], v)) return 1;
    return 0;
}

enum { PTR_BOUND_RANGES = 4 * 16 };
typedef struct {
    struct { uint64_t lo, hi; } r[PTR_BOUND_RANGES];
    unsigned n;
    const p_bind_buf *const *sources;
    int nsources;
} PtrBoundSet;

// Only active, non-wrapping intervals can exclude a candidate. Build their endpoints once for this table scan,
// rather than revisiting every empty slot (up to four 16-slot stage tables) for each sampled pointer. Preserve the
// original uint64 membership rule: an overflowing addr+size cannot match v>=addr && v<addr+size.
static void ptr_bounds_init(PtrBoundSet *out, const p_bind_buf *const sources[], int nsources) {
    out->n = 0, out->sources = sources, out->nsources = nsources;
    for (int s = 0; s < nsources; s++) {
        if (!sources[s]) continue;
        for (int i = 0; i < 16; i++) {
            uint64_t lo = sources[s][i].addr, hi = lo + sources[s][i].size;
            if (!lo || hi <= lo) continue;
            // All current callers supply at most four sources. Keep exact semantics for future callers with more
            // active intervals instead of silently dropping exclusions when the fixed local list is full.
            if (out->n == PTR_BOUND_RANGES) { out->n = PTR_BOUND_RANGES + 1; return; }
            out->r[out->n].lo = lo, out->r[out->n].hi = hi;
            out->n++;
        }
    }
}

static int ptr_exclude(uint64_t v, const PtrBoundSet *bound) {
    if (bound->n > PTR_BOUND_RANGES) return ptr_exclude_sources(v, bound->sources, bound->nsources);
    for (unsigned i = 0; i < bound->n; i++)
        if (v >= bound->r[i].lo && v < bound->r[i].hi) return 1;
    return 0;
}

static int ptr_candidate_ok(uint64_t v) {
    // Guest VirtualAlloc pools in MKW span 0x110000000..0x3ffffffff+ (including 64 KiB-aligned bases where the low
    // 16..32 bits are zero). Reject unaligned words and small packed 32-bit integer pairs like 0x100000003.
    if ((v & 3) || v < 0x10000000ull || v >= (1ull << 47)) return 0;
    if ((v >> 32) < 0x10ull && (v & 0xffffull) != 0 && (v & 0xffffffffull) < 0x10000ull) return 0;
    return 1;
}

// Finished pointer-table regions are reused across draws that bind the same storage ranges. Lifetime follows
// pool/usnap generations and command-buffer epoch. A cheap fingerprint (a handful of words per buffer) catches
// soft guest writes inside a CB — without the old full-sample content walk that dominated race CPU. UPDATE_UB
// overlays are read by ubo_ptr, not these guest-RAM scans (so UPD does not wipe the whole cache).
// 4-way set-associative: race draws thrash a direct-mapped 1K table; 512×4 keeps hot bindings without
// growing the fingerprint work on the hit path.
enum { PTR_RESULT_SETS = 512, PTR_RESULT_WAYS = 4 };
typedef struct {
    uint64_t src_key, life, finger;
    PtrEntry e[PTR_ENTRIES];
} PtrResultSlot;
typedef struct {
    PtrResultSlot way[PTR_RESULT_WAYS];
    uint8_t clock;
} PtrResultSet;
static PtrResultSet ptr_result_cache[PTR_RESULT_SETS];

static uint64_t ptr_mix(uint64_t h, uint64_t v) {
    return h ^ (v + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2));
}

static uint64_t ptr_result_life(void) {
    return vk_mem_gen() + vk_usnap_gen() + ptr_cb_epoch * UINT64_C(0x100000001b3);
}

static uint64_t ptr_bufs_src_key(const p_bind_buf bufs[16], int first_slot, uint64_t min_sz, uint64_t max_sz,
                                 uint64_t h) {
    if (!bufs) return ptr_mix(h, 0);
    for (int i = first_slot; i < 16; i++) {
        if (!bufs[i].addr) continue;
        uint64_t sz = bufs[i].size ? bufs[i].size : 0x200ull;
        if (sz < min_sz || sz > max_sz) continue;
        h = ptr_mix(h, bufs[i].addr);
        h = ptr_mix(h, bufs[i].size);
        h = ptr_mix(h, (uint64_t)(unsigned)(i + 1));
    }
    return h;
}

// A few fixed samples of the stable head (not the churning SSBO tail). Resolve once to clamp length; skip
// device-local unreadables. Computed only after an address-key hit.
static uint64_t ptr_bufs_quick_key(const p_bind_buf bufs[16], int first_slot, uint64_t min_sz, uint64_t max_sz,
                                   uint64_t h) {
    if (!bufs) return ptr_mix(h, 0);
    for (int i = first_slot; i < 16; i++) {
        if (!bufs[i].addr) continue;
        uint64_t sz = bufs[i].size ? bufs[i].size : 0x200ull;
        if (sz < min_sz || sz > max_sz) continue;
        VkMemSpan sp;
        if (!vk_mem_resolve(bufs[i].addr, 1, &sp)) {
            h = ptr_mix(h, UINT64_C(0x1111));
            continue;
        }
        if (sp.alloc && !plat_mem_readable(bufs[i].addr, 8)) {
            h = ptr_mix(h, UINT64_C(0x2222));
            continue;
        }
        uint64_t buf_sz = sz;
        if (buf_sz > sp.avail) buf_sz = sp.avail;
        if (buf_sz < 8) continue;
        const uint64_t *base = (const uint64_t *)(uintptr_t)bufs[i].addr;
        uint64_t words = buf_sz / 8;
        h = ptr_mix(h, base[0]);
        if (words > 1) h = ptr_mix(h, base[1]);
        if (words > 8) h = ptr_mix(h, base[8]);
        if (words > 32) h = ptr_mix(h, base[32]);
        if (words > 128) h = ptr_mix(h, base[128]);
        h = ptr_mix(h, (uint64_t)(unsigned)(i + 1) << 32 | words);
    }
    return h;
}

static PtrResultSlot *ptr_result_lookup(uint64_t src_key) {
    uint64_t life = ptr_result_life();
    PtrResultSet *s = &ptr_result_cache[src_key & (PTR_RESULT_SETS - 1)];
    for (int w = 0; w < PTR_RESULT_WAYS; w++)
        if (s->way[w].src_key == src_key && s->way[w].life == life) return &s->way[w];
    return NULL;
}

static void ptr_result_take(PtrResultSlot *s, PtrEntry out[PTR_ENTRIES]) {
    memcpy(out, s->e, sizeof(PtrEntry) * PTR_ENTRIES);
}

static void ptr_result_put(uint64_t src_key, uint64_t finger, const PtrEntry e[PTR_ENTRIES]) {
    uint64_t life = ptr_result_life();
    PtrResultSet *s = &ptr_result_cache[src_key & (PTR_RESULT_SETS - 1)];
    int slot = -1;
    for (int w = 0; w < PTR_RESULT_WAYS; w++) {
        if (s->way[w].src_key == src_key) {
            slot = w;
            break;
        }
        if (slot < 0 && (!s->way[w].src_key || s->way[w].life != life)) slot = w;
    }
    if (slot < 0) {
        slot = s->clock & (PTR_RESULT_WAYS - 1);
        s->clock++;
    }
    PtrResultSlot *dst = &s->way[slot];
    dst->src_key = src_key, dst->life = life, dst->finger = finger;
    memcpy(dst->e, e, sizeof(PtrEntry) * PTR_ENTRIES);
}

// Open-addressed set of collected pointer values (max 48). Avoids a linear found[] walk on every probe —
// bone/instance lists repeat the same bases thousands of times per scan.
typedef struct { uint64_t slot[64]; } PtrFoundSet;

static int ptr_found_has(const PtrFoundSet *s, uint64_t v) {
    uint32_t i = (uint32_t)((v * UINT64_C(0x9e3779b97f4a7c15)) >> 58);
    for (uint32_t n = 0; n < 64; n++, i = (i + 1) & 63) {
        uint64_t x = s->slot[i];
        if (!x) return 0;
        if (x == v) return 1;
    }
    return 1;
}

static void ptr_found_put(PtrFoundSet *s, uint64_t v) {
    uint32_t i = (uint32_t)((v * UINT64_C(0x9e3779b97f4a7c15)) >> 58);
    for (uint32_t n = 0; n < 64; n++, i = (i + 1) & 63) {
        if (!s->slot[i]) { s->slot[i] = v; return; }
        if (s->slot[i] == v) return;
    }
}

static inline void pointer_table_try_add(uint64_t v, uint64_t *found, uint32_t *n_io, PtrFoundSet *seen,
                                         PtrFoundSet *miss, const PtrBoundSet *bound) {
    uint32_t n = *n_io;
    if (n >= 48 || !ptr_candidate_ok(v)) return;
    // A pointer already collected cannot add another region. Test it before resolving: repeated bone/instance
    // pointers are common, and their final backing is checked again by pointer_table_finish.
    if (ptr_found_has(seen, v) || ptr_found_has(miss, v)) return;
    if (ptr_exclude(v, bound)) return;
    VkMemSpan hit;
    if (!vk_mem_resolve(v, 1, &hit)) {
        ptr_found_put(miss, v);   // same bad candidate appears thousands of times across bone lists
        return;
    }
    (void)hit;
    ptr_found_put(seen, v);
    found[n] = v;
    *n_io = n + 1;
}

static void pointer_table_scan_range(const p_bind_buf bufs[16], int first_slot, uint64_t min_sz, uint64_t max_sz,
                                     uint64_t *found, uint32_t *n_io, PtrFoundSet *seen, PtrFoundSet *miss,
                                     const PtrBoundSet *bound) {
    if (!bufs) return;
    for (int i = first_slot; i < 16 && *n_io < 48; i++) {
        if (!bufs[i].addr) continue;
        VkMemSpan sp;
        if (!vk_mem_resolve(bufs[i].addr, 1, &sp)) continue;
        if (sp.alloc && !plat_mem_readable(bufs[i].addr, 8)) continue;
        uint64_t buf_sz = bufs[i].size ? bufs[i].size : 0x200ull;
        if (buf_sz > sp.avail) buf_sz = sp.avail;
        if (buf_sz < 8 || buf_sz < min_sz || buf_sz > max_sz) continue;
        const uint64_t *base = (const uint64_t *)(uintptr_t)bufs[i].addr;
        // Dense head: full buffer when ≤4 KiB (UBO constants / pointer / cull lists); else first 1 KiB.
        uint64_t dense = buf_sz <= 0x1000ull ? buf_sz : 0x400ull;
        uint64_t stride_end = buf_sz;
        if (stride_end > (256ull << 10)) stride_end = 256ull << 10;
        // Visit the same ordered samples directly. Building their 16 KiB offset array first wrote and read it on
        // every scanned buffer, even when the first few samples filled found[] and the rest would not be read.
        uint32_t no = 0;
#define PTR_PROBE(offset_) do { \
            uint64_t v = base[(offset_) / 8]; \
            no++; \
            pointer_table_try_add(v, found, n_io, seen, miss, bound); \
        } while (0)
        for (uint64_t off = 0; off + 8 <= dense && no < 512 && *n_io < 48; off += 8) PTR_PROBE(off);
        uint64_t mid = stride_end < (64ull << 10) ? stride_end : (64ull << 10);
        for (uint64_t off = dense; off + 8 <= mid && no < 1600 && *n_io < 48; off += 0x20) PTR_PROBE(off);
        for (uint64_t off = mid; off + 8 <= stride_end && no < 2000 && *n_io < 48; off += 0x80) PTR_PROBE(off);
        if (buf_sz > stride_end && *n_io < 48) {
            for (int q = 1; q <= 3 && no < 2048 && *n_io < 48; q++) {
                uint64_t t = (buf_sz / 4) * (uint64_t)q;
                t &= ~7ull;
                if (t + 8 <= buf_sz && t >= stride_end) PTR_PROBE(t);
            }
            uint64_t tail = buf_sz > 8 ? (buf_sz - 8) & ~7ull : 0;
            if (tail >= stride_end && no < 2048 && *n_io < 48) PTR_PROBE(tail);
        }
#undef PTR_PROBE
    }
}

static void pointer_table_finish(uint64_t *found, uint32_t n, const p_bind_buf *src[], int nsrc,
                                 PtrEntry out[PTR_ENTRIES]);

static void pointer_table2(const p_bind_buf a[16], const p_bind_buf b[16], PtrEntry out[PTR_ENTRIES]) {
    // Match Hookshot vk_bank.cpp: each region starts at a found GPU pointer and covers toward the end
    // of that pool (folding later pointers in the same pool within 64 MiB). Starting at the pool base
    // then capping to 1 MiB left GlobalToStorage searching the wrong window — skinned / culled mesh
    // reads returned 0 (collapsed props, terrain holes, black strips).
    // Task and mesh stages each hold their own small SSBOs; scanning only the stage being bound left
    // the other stage's bone / instance pointers out of c0+0x880 (title foliage → green shards).
    uint64_t src_key = ptr_mix(UINT64_C(0x2a2a2a2a), 2);
    src_key = ptr_bufs_src_key(a, 0, 8, ~0ull, src_key);
    src_key = ptr_bufs_src_key(b, 0, 8, ~0ull, src_key);
    PtrResultSlot *hit = ptr_result_lookup(src_key);
    if (hit) {
        uint64_t finger = ptr_bufs_quick_key(a, 0, 8, ~0ull, 0);
        finger = ptr_bufs_quick_key(b, 0, 8, ~0ull, finger);
        if (hit->finger == finger) { ptr_result_take(hit, out); return; }
    }
    memset(out, 0, sizeof(PtrEntry) * PTR_ENTRIES);
    uint64_t found[48];
    uint32_t n = 0;
    PtrFoundSet seen = {0}, miss = {0};
    const p_bind_buf *bound[2] = {a, b};
    PtrBoundSet excluded;
    ptr_bounds_init(&excluded, bound, 2);
    // Visit order matches the pre-opt path (all sizes per source). Reordering small-then-large changed
    // which of >48 candidates survived and corrupted menu guest state via wrong write-back windows.
    pointer_table_scan_range(a, 0, 8, ~0ull, found, &n, &seen, &miss, &excluded);
    pointer_table_scan_range(b, 0, 8, ~0ull, found, &n, &seen, &miss, &excluded);
    pointer_table_finish(found, n, bound, 2, out);
    uint64_t finger = ptr_bufs_quick_key(a, 0, 8, ~0ull, 0);
    finger = ptr_bufs_quick_key(b, 0, 8, ~0ull, finger);
    ptr_result_put(src_key, finger, out);
}

static void pointer_table4(const p_bind_buf a[16], const p_bind_buf b[16], const p_bind_buf c[16],
                           const p_bind_buf d[16], PtrEntry out[PTR_ENTRIES]) {
    const p_bind_buf *src[4] = {a, b, c, d};
    uint64_t src_key = ptr_mix(UINT64_C(0x4a4a4a4a), 4);
    for (int i = 0; i < 4; i++) src_key = ptr_bufs_src_key(src[i], 0, 8, ~0ull, src_key);
    PtrResultSlot *hit = ptr_result_lookup(src_key);
    if (hit) {
        uint64_t finger = 0;
        for (int i = 0; i < 4; i++) finger = ptr_bufs_quick_key(src[i], 0, 8, ~0ull, finger);
        if (hit->finger == finger) { ptr_result_take(hit, out); return; }
    }
    memset(out, 0, sizeof(PtrEntry) * PTR_ENTRIES);
    uint64_t found[48];
    uint32_t n = 0;
    PtrFoundSet seen = {0}, miss = {0};
    PtrBoundSet excluded;
    ptr_bounds_init(&excluded, src, 4);
    for (int i = 0; i < 4; i++) pointer_table_scan_range(src[i], 0, 8, ~0ull, found, &n, &seen, &miss, &excluded);
    pointer_table_finish(found, n, src, 4, out);
    uint64_t finger = 0;
    for (int i = 0; i < 4; i++) finger = ptr_bufs_quick_key(src[i], 0, 8, ~0ull, finger);
    ptr_result_put(src_key, finger, out);
}

static void pointer_table_finish(uint64_t *found, uint32_t n, const p_bind_buf *src[], int nsrc,
                                 PtrEntry out[PTR_ENTRIES]) {
    for (uint32_t i = 1; i < n; i++) {
        uint64_t v = found[i];
        uint32_t j = i;
        while (j && found[j - 1] > v) found[j] = found[j - 1], j--;
        found[j] = v;
    }
    uint64_t most = vk_ctx()->props.limits.maxStorageBufferRange;
    if (most > 0xFFFFFF00ull) most = 0xFFFFFF00ull;
    most &= ~0xFFull;
    uint32_t regions = 0;
    for (uint32_t i = 0; i < n && regions < PTR_ENTRIES;) {
        VkMemSpan sp;
        if (!vk_mem_resolve(found[i], 1, &sp)) {
            i++;
            continue;
        }
        // Hookshot vk_bank.cpp: fold later pointers in the same pool within 64 MiB, then cover from the
        // first through last_found+tail (not first+1MiB). Truncating at first+1MiB left bone / instance
        // tables unbound → GlobalToStorage returned 0 and mesh foliage collapsed into giant spikes.
        uint64_t first = found[i], end = first + sp.avail;
        while (i + 1 < n && found[i + 1] < end && found[i + 1] - first < (64ull << 20)) i++;
        // Cover through last+4MiB (Hookshot was last+1MiB). n90 cluster at 0x28ab34030+0x40*
        // had only ~32 KiB of avail from first — also pull in the pool bytes *before* first when
        // the pointers sit mid-block (bone headers just below the first hit).
        uint64_t hi = found[i] + (4ull << 20);
        if (hi > end) hi = end;
        if (sp.offset > 0 && sp.offset < (1ull << 20)) {
            uint64_t lo = first - sp.offset;
            if (first - lo < (1ull << 20)) first = lo;
        }
        uint64_t sz = hi > first ? hi - first : 0;
        if (sz > most) sz = most;
        if (sz) out[regions++] = (PtrEntry){first, sz};
        i++;
    }
    // Fallback when tiny pointer blocks are still zero (GPU-written later / empty at bind): expose
    // the contiguous pool window around each large SSBO so GlobalToStorage can reach bone /
    // instance words that land beside the bound range (n101 deep-scan still found=0 — heads are
    // floats/zeros; foliage pointers target neighbouring allocations in the same pool).
    if (!regions) {
        typedef struct { uint64_t addr, size; } Cand;
        Cand cand[32];
        uint32_t nc = 0;
        for (int s = 0; s < nsrc; s++) {
            if (!src[s]) continue;
            for (int i = 0; i < 16 && nc < 32; i++) {
                if (!src[s][i].addr || src[s][i].size < 0x1000) continue;
                VkMemSpan sp;
                if (!vk_mem_resolve(src[s][i].addr, 1, &sp)) continue;
                // Pool window: back up at most 4 MiB before the bind (bone headers), cover forward
                // through the SSBO and up to 16 MiB of contiguous pool. Exact-SSBO-only (n85-era
                // fix) left neighbours unbound; full-pool 16 MiB from bind alone matched garbage
                // (rainbow stairs) — keep the window anchored on the real SSBO.
                uint64_t lo_back = sp.offset < (4ull << 20) ? sp.offset : (4ull << 20);
                uint64_t first = src[s][i].addr - lo_back;
                uint64_t forward = src[s][i].size ? src[s][i].size : sp.avail;
                if (forward < (16ull << 20) && sp.avail > forward) {
                    uint64_t extra = sp.avail - forward;
                    if (extra > (16ull << 20) - forward) extra = (16ull << 20) - forward;
                    forward += extra;
                }
                uint64_t sz = lo_back + forward;
                if (sz > sp.offset + sp.avail) sz = sp.offset + sp.avail;
                if (sz > most) sz = most;
                if (sz) cand[nc++] = (Cand){first, sz};
            }
        }
        for (uint32_t i = 1; i < nc; i++) {
            Cand v = cand[i];
            uint32_t j = i;
            while (j && cand[j - 1].size < v.size) cand[j] = cand[j - 1], j--;
            cand[j] = v;
        }
        for (uint32_t i = 0; i < nc && regions < PTR_ENTRIES; i++) {
            int dup = 0;
            for (uint32_t r = 0; r < regions; r++)
                if (out[r].addr == cand[i].addr) dup = 1;
            if (!dup) out[regions++] = (PtrEntry){cand[i].addr, cand[i].size};
        }
        (void)regions;
    }
}

static void pointer_table(const p_bind_buf ssbo[16], PtrEntry out[PTR_ENTRIES]) {
    pointer_table2(ssbo, NULL, out);
}

static void c0_pointer_table(uint8_t bank[VK_C0_BYTES], const p_bind_buf ssbo[16]) {
    if (VK_C0_BYTES < PTR_TABLE + 16u * PTR_ENTRIES) return;
    PtrEntry e[PTR_ENTRIES];
    pointer_table(ssbo, e);
    for (uint32_t i = 0; i < PTR_ENTRIES; i++) {
        uint32_t off = PTR_TABLE + 16u * i, bytes = (uint32_t)e[i].size;
        memcpy(bank + off, &e[i].addr, 8);
        memcpy(bank + off + 8, &bytes, 4);
    }
}

static void c0_pointer_table_cs(uint8_t bank[VK_C0_BYTES], const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                                const Nvn2C0Layout *lay, const uint32_t push[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS]) {
    if (VK_C0_BYTES < PTR_TABLE + 16u * PTR_ENTRIES) return;
    PtrEntry e[PTR_ENTRIES];
    uint64_t src_key = ptr_mix(UINT64_C(0xc500c500), lay ? 1 : 0);
    src_key = ptr_bufs_src_key(ssbo, 0, 8, 0x10000ull, src_key);
    src_key = ptr_bufs_src_key(ssbo, 0, 0x10001ull, ~0ull, src_key);
    src_key = ptr_bufs_src_key(ubo, 2, 8, ~0ull, src_key);
    if (lay && push) {
        for (unsigned r = 0; r < NVN2_ROOT_TABLES; r++) {
            src_key = ptr_mix(src_key, lay->roots[r]);
            for (unsigned w = 0; w + 1 < lay->roots[r]; w += 2) {
                uint64_t v = (uint64_t)push[r][w] | ((uint64_t)push[r][w + 1] << 32);
                src_key = ptr_mix(src_key, v);
            }
        }
    }
    PtrResultSlot *hit = ptr_result_lookup(src_key);
    if (hit) {
        uint64_t finger = ptr_bufs_quick_key(ssbo, 0, 8, 0x10000ull, 0);
        finger = ptr_bufs_quick_key(ssbo, 0, 0x10001ull, ~0ull, finger);
        finger = ptr_bufs_quick_key(ubo, 2, 8, ~0ull, finger);
        if (hit->finger == finger) ptr_result_take(hit, e);
        else hit = NULL;
    }
    if (!hit) {
        memset(e, 0, sizeof e);
        uint64_t found[48];
        uint32_t n = 0;
        PtrFoundSet seen = {0}, miss = {0};
        const p_bind_buf *bound[1] = {ssbo};
        PtrBoundSet excluded;
        ptr_bounds_init(&excluded, bound, 1);
        if (lay && push) {
            for (unsigned r = 0; r < NVN2_ROOT_TABLES; r++) {
                for (unsigned w = 0; w + 1 < lay->roots[r] && n < 48; w += 2) {
                    uint64_t v = (uint64_t)push[r][w] | ((uint64_t)push[r][w + 1] << 32);
                    pointer_table_try_add(v, found, &n, &seen, &miss, &excluded);
                }
            }
        }
        pointer_table_scan_range(ubo, 2, 8, ~0ull, found, &n, &seen, &miss, &excluded);
        pointer_table_scan_range(ssbo, 0, 8, 0x10000ull, found, &n, &seen, &miss, &excluded);
        pointer_table_scan_range(ssbo, 0, 0x10001ull, ~0ull, found, &n, &seen, &miss, &excluded);
        pointer_table_finish(found, n, bound, 1, e);
        uint64_t finger = ptr_bufs_quick_key(ssbo, 0, 8, 0x10000ull, 0);
        finger = ptr_bufs_quick_key(ssbo, 0, 0x10001ull, ~0ull, finger);
        finger = ptr_bufs_quick_key(ubo, 2, 8, ~0ull, finger);
        ptr_result_put(src_key, finger, e);
    }
    for (uint32_t i = 0; i < PTR_ENTRIES; i++) {
        uint32_t off = PTR_TABLE + 16u * i, bytes = (uint32_t)e[i].size;
        memcpy(bank + off, &e[i].addr, 8);
        memcpy(bank + off + 8, &bytes, 4);
    }
}

static void c0_pointer_table_mesh(uint8_t bank[VK_C0_BYTES], const p_bind_buf ts_ubo[16], const p_bind_buf ms_ubo[16],
                                  const p_bind_buf a[16], const p_bind_buf b[16], const p_bind_buf c[16],
                                  const p_bind_buf d[16]) {
    if (VK_C0_BYTES < PTR_TABLE + 16u * PTR_ENTRIES) return;
    PtrEntry e[PTR_ENTRIES];
    const p_bind_buf *src[4] = {a, b, c, d};
    uint64_t src_key = ptr_mix(UINT64_C(0x6e556e55), 6);
    src_key = ptr_bufs_src_key(ts_ubo, 2, 8, ~0ull, src_key);
    src_key = ptr_bufs_src_key(ms_ubo, 2, 8, ~0ull, src_key);
    for (int i = 0; i < 4; i++) {
        src_key = ptr_bufs_src_key(src[i], 0, 8, 0x10000ull, src_key);
        src_key = ptr_bufs_src_key(src[i], 0, 0x10001ull, ~0ull, src_key);
    }
    PtrResultSlot *hit = ptr_result_lookup(src_key);
    if (hit) {
        uint64_t finger = ptr_bufs_quick_key(ts_ubo, 2, 8, ~0ull, 0);
        finger = ptr_bufs_quick_key(ms_ubo, 2, 8, ~0ull, finger);
        for (int i = 0; i < 4; i++) {
            finger = ptr_bufs_quick_key(src[i], 0, 8, 0x10000ull, finger);
            finger = ptr_bufs_quick_key(src[i], 0, 0x10001ull, ~0ull, finger);
        }
        if (hit->finger == finger) ptr_result_take(hit, e);
        else hit = NULL;
    }
    if (!hit) {
        memset(e, 0, sizeof e);
        uint64_t found[48];
        uint32_t n = 0;
        PtrFoundSet seen = {0}, miss = {0};
        PtrBoundSet excluded;
        ptr_bounds_init(&excluded, src, 4);
        // Scan user constant buffers (c2..c15, e.g. c3/c4 meshlet & bone descriptors) and small SSBOs first so
        // real pointer tables populate found[] ahead of multi-megabyte vertex/material SSBOs.
        pointer_table_scan_range(ts_ubo, 2, 8, ~0ull, found, &n, &seen, &miss, &excluded);
        pointer_table_scan_range(ms_ubo, 2, 8, ~0ull, found, &n, &seen, &miss, &excluded);
        for (int i = 0; i < 4; i++) pointer_table_scan_range(src[i], 0, 8, 0x10000ull, found, &n, &seen, &miss, &excluded);
        for (int i = 0; i < 4; i++)
            pointer_table_scan_range(src[i], 0, 0x10001ull, ~0ull, found, &n, &seen, &miss, &excluded);
        pointer_table_finish(found, n, src, 4, e);
        uint64_t finger = ptr_bufs_quick_key(ts_ubo, 2, 8, ~0ull, 0);
        finger = ptr_bufs_quick_key(ms_ubo, 2, 8, ~0ull, finger);
        for (int i = 0; i < 4; i++) {
            finger = ptr_bufs_quick_key(src[i], 0, 8, 0x10000ull, finger);
            finger = ptr_bufs_quick_key(src[i], 0, 0x10001ull, ~0ull, finger);
        }
        ptr_result_put(src_key, finger, e);
    }
    for (uint32_t i = 0; i < PTR_ENTRIES; i++) {
        uint32_t off = PTR_TABLE + 16u * i, bytes = (uint32_t)e[i].size;
        memcpy(bank + off, &e[i].addr, 8);
        memcpy(bank + off + 8, &bytes, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): Ryujinx's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. Task/mesh: NVN tables at 0x670/0x770 plus Hookshot pointer table at 0x880.
// Unbound slots read zero, since the replay keeps them zero. Layout: nxnative dm/gfx/nvn/nvn_mtl.m (MIT).
static void c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                    const uint32_t dispatch[3], uint32_t base_instance, int mesh_table) {
    if (stage == NVN_STAGE_CS && c0_lay) {
        if (!nvn2_c0_pack_handles(bank, VK_C0_BYTES, c0_lay, c0_push, ssbo, ubo, c0_tex, c0_img))
            rt_fatal("vk: invalid NVN 2 compute parameter-block layout");
        // NVN 2 compute dispatch dimensions {X, Y, Z} live at c0+0x4..0xf, below the parameter block at 0x10
        // (MKW compute culling / LOD shaders index workgroups as ctaid.y * c[0][0x4] + ctaid.x).
        memcpy(bank + 0x4, dispatch, 12);
        // Layout ends before 0x880: same pointer table task/mesh get (culling passes reading root/UBO/SSBO ptrs).
        if (nvn2_c0_part(c0_lay, 5) <= PTR_TABLE) c0_pointer_table_cs(bank, ssbo, ubo, c0_lay, c0_push);
        return;
    }
    memset(bank, 0, VK_C0_BYTES);
    int base = stage == NVN_STAGE_VS ? 0x110
             : stage == NVN_STAGE_FS ? 0x510
             : stage == NVN_STAGE_CS ? 0x310
             : stage == NVN_STAGE_GS ? 0x410
             : stage == NVN_STAGE_TS ? 0x670
             : stage == NVN_STAGE_MS ? 0x770
             : -1;
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (mesh_table && (stage == NVN_STAGE_TS || stage == NVN_STAGE_MS)) c0_pointer_table(bank, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    c0_fill(bank, stage, ssbo, ubo, dispatch, base_instance, 1);
}

static int c0_on(void) {   // TOTK_FIX_C0=0: c0 reads zeros instead of the driver constants
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_FIX_C0");
    }
    return on;
}

typedef struct { uint8_t bytes[16u * PTR_ENTRIES]; int valid; } MeshPtrTable;

static void c0_mesh_table(VkExec *ex, uint8_t bank[VK_C0_BYTES], MeshPtrTable *table) {
    // Task and mesh banks of this descriptor build read one snapshot of the same six source arrays. Keep this
    // table separate from pointer_binding's SSBO-only table; it also scans the task and mesh user UBOs.
    if (!table->valid) {
        c0_pointer_table_mesh(bank, ex->ubo[NVN_STAGE_TS], ex->ubo[NVN_STAGE_MS], ex->ssbo[NVN_STAGE_TS],
                              ex->ssbo[NVN_STAGE_MS], ex->ssbo[NVN_STAGE_VS], ex->ssbo[NVN_STAGE_CS]);
        memcpy(table->bytes, bank + PTR_TABLE, sizeof table->bytes);
        table->valid = 1;
    } else {
        memcpy(bank + PTR_TABLE, table->bytes, sizeof table->bytes);
    }
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static VkDescriptorBufferInfo c0_info(VkExec *ex, int st, MeshPtrTable *mesh_table) {
    int k = c0_stage_index(st);
    if (k < 0) rt_fatal("vk_exec: synthetic c0 is unavailable for stage %d", st);
    // vk_c0_fill reads buffer bindings, compute root words and dispatch size, or the graphics base instance:
    // equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    typeof(B.c0_in[0]) in;
    memset(&in, 0, sizeof in);
    memcpy(in.ssbo, ex->ssbo[st], sizeof in.ssbo);
    if (st == NVN_STAGE_CS) memcpy(in.ubo, ex->ubo[st], sizeof in.ubo), memcpy(in.dim, c0_dim, sizeof in.dim);
    else if (st == NVN_STAGE_TS)
        // Cache key must include the sibling mesh-stage SSBOs: pointer_table merges both.
        memcpy(in.ubo, ex->ssbo[NVN_STAGE_MS], sizeof in.ubo), in.binst = c0_binst;
    else if (st == NVN_STAGE_MS)
        memcpy(in.ubo, ex->ssbo[NVN_STAGE_TS], sizeof in.ubo), in.binst = c0_binst;
    else
        in.binst = c0_binst;
    if (st == NVN_STAGE_CS && c0_lay) {
        in.lay = *c0_lay, in.has_lay = 1;
        for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++)
            memcpy(in.roots[i], ex->push[1][i], 4u * c0_lay->roots[i]);
        if (c0_lay->textures) memcpy(in.tex, ex->tex[NVN_STAGE_CS], 8u * c0_lay->textures);
        if (c0_lay->images) memcpy(in.img, ex->img[NVN_STAGE_CS], 8u * c0_lay->images);
    }
    int is_mesh = st == NVN_STAGE_TS || st == NVN_STAGE_MS;
    if (!is_mesh && B.c0_valid[k] && !memcmp(&in, &B.c0_in[k], sizeof in))
        return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    B.c0_in[k] = in;
    uint8_t tmp[VK_C0_BYTES];
    // The combined mesh table below overwrites all three pointer entries; omit the single-stage scan whose result
    // would be discarded. Public vk_c0_fill retains the single-stage table for its other callers and probes.
    c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst, !is_mesh);
    // Task/mesh GlobalToStorage: scan TS+MS UBOs (c3/c4 meshlet & bone descriptors) and TS+MS+VS+CS SSBOs.
    if (is_mesh) c0_mesh_table(ex, tmp, mesh_table);
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// The buffer range limits, read once: the device's properties are a cold cache line on every binding of every draw.
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot, MeshPtrTable *mesh_table) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0 && c0_on()) return c0_info(ex, st, mesh_table);
    if (slot == 1 && ex->prog) {   // compiler constants stored after the shader code
        const prog_t *p = ex->prog;
        const uint8_t *ctl = NULL, *code = NULL;
        uint32_t len = 0, c1 = 0, c1_bytes = 0;
        int found = find_stage(p, st, &ctl, &code, &len);
        // Mesh draws bind a [MeshPrimitiveHeader, task, mesh] program where host slot 1 is a header-only SPH,
        // while the live fragment stage comes from vk_exec_fragment_prog(ex).
        if (st == NVN_STAGE_FS && (!found || control_is_header_only(ctl, len))) {
            const prog_t *fp = vk_exec_fragment_prog(ex);
            if (fp && fp != p) found = find_stage(fp, st, &ctl, &code, &len);
        }
        if (found) {
            if (gfx_shader_is_nvn2(ctl)) {   // raw NVN 2 / DKMESH control: c1_size at +0x3c, c1_off at +0x40, total at +0x44
                uint32_t c1_size = 0, c1_off = 0, total = 0;
                memcpy(&c1_size, ctl + 0x3c, 4);
                memcpy(&c1_off, ctl + 0x40, 4);
                memcpy(&total, ctl + 0x44, 4);
                if (c1_size && c1_off <= total && c1_size <= total - c1_off) c1 = c1_off, c1_bytes = c1_size;
            } else {
                memcpy(&c1, ctl + 0x700, 4);
                if (c1 < len) c1_bytes = len - c1;
            }
            if (c1_bytes && bind_range(ex, (uint64_t)(uintptr_t)code + c1, c1_bytes, align, max, VK_MUSE_UBO, 0, 0, 0, &bi))
                return bi;
            if (c1_bytes && c1_bytes <= max) {   // NVN 2 stages: the lowered code (nvn2_lower.cpp) is host memory
                VkBuffer b;
                VkDeviceSize o;
                RS.volatile_ = 1;
                memcpy(scratch(ex, c1_bytes, &b, &o), code + c1, c1_bytes);
                return (VkDescriptorBufferInfo){b, o, c1_bytes};
            }
        }
    } else if (slot == 17 && st != NVN_STAGE_GS) {
        // Switch 2 full-screen blits read a parameter block the console supplies from bank 24, which the lowering moves
        // to bank 17 (never bound by games). For a blit over the whole render area (Eglantine, ir_opt/position_pass.cpp
        // BlitConstantsPass): texel size at 0x0/0x4 and 0x10/0x14, size at 0x8/0xc, UV scale 1 at 0x18/0x1c, UV clamp
        // rectangle 0,0..1,1 at 0x30-0x3c, zero dither bias, integer source rectangle 0,0..W-1,H-1 at 0x50-0x5c.
        // Bank 24 is NVN 2's graphics root table 0: the words the game pushed (OP_PUSH_CONST) replace the guess
        // (the NVN 2 reference title pushes 1/w, 1/h, w, h). No the NVN 2 reference title compute stage reads bank 24; one that did would get zeros plus
        // the compute pushes, never the graphics guess.
        float w = ex->pass_w ? (float)ex->pass_w : 1.0f, h = ex->pass_h ? (float)ex->pass_h : 1.0f;
        union { float f[64]; uint32_t u[64]; } blk = {{1 / w, 1 / h, w, h, 1 / w, 1 / h, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1}};
        blk.u[22] = (uint32_t)(w - 1), blk.u[23] = (uint32_t)(h - 1);
        int k = st == NVN_STAGE_CS;
        if (k) memset(&blk, 0, sizeof blk);
        for (int i = 0; i < 64; i++)
            if (ex->push_set[k][0] >> i & 1) blk.u[i] = ex->push[k][0][i];
        VkBuffer b;
        VkDeviceSize o;
        RS.volatile_ = 1;
        memcpy(scratch(ex, sizeof blk, &b, &o), &blk, sizeof blk);
        return (VkDescriptorBufferInfo){b, o, sizeof blk};
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, 0, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// The NVN slot of a storage binding. Ryujinx numbers a shader's storage buffers in the order it meets them
// (GfxShaderRes.slot); the NVN slot is the entry of the driver constant buffer c0 that holds the buffer's address
// (cbuf_slot 0, handle = word offset): 16-byte entries at byte 0x110 (vertex), 0x510 (fragment) and 0x310 (compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: the
// address is not in c0 (the shader loads it from a user uniform block: ubo_ptr) or the stage has no known table.
// Mapping: MacNativeGames games/totk/tools/totk_xlate/Service.cs:206-222 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou).
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    // Eglantine shader_recompiler/frontend/maxwell/translate_program.cpp::AddNVNStorageBuffers uses a distinct
    // ordinary geometry table at c0+0x410. This only maps declared
    // bindings; it does not synthesize that driver bank or establish the native NVN2 geometry ABI.
    int off = r->handle * 4, base = st == NVN_STAGE_VS ? 0x110 : st == NVN_STAGE_FS ? 0x510 :
                                  st == NVN_STAGE_GS ? 0x410 : st == NVN_STAGE_CS ? 0x310 :
                                  st == NVN_STAGE_TS ? 0x670 : st == NVN_STAGE_MS ? 0x770 : -1;
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    // Hookshot pointer table (c0+0x880): not an NVN bind slot; ssbo_info reads the filled bank / pointer_table.
    if (off >= (int)PTR_TABLE && off < (int)(PTR_TABLE + 16u * PTR_ENTRIES) && (off - (int)PTR_TABLE) % 16 == 0)
        return -2;
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., NVN uniform slot
// cbuf_slot - 3, byte offset handle * 4). The Depths terrain reads its vertices through one, about 500 draws a frame;
// with a zero buffer they collapse and the ground is missing. MacNativeGames resolves these the same way (NVN_RB_SBUFU,
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m @ b75be8cf).
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// The inline uniform updates of the submission being replayed, newest last (a ring of UPD_N). op_update_ub writes them
// into the buffer's GPU copy; guest memory gets them only from the write-back when the submission ends. The terrain
// rewrites its pointer block this way before every draw, all at one address, so ubo_ptr reads the latest update that
// covers the pointer, else memory. Payloads stay in the recorded command buffer for the whole replay.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n++ % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
}

// 1 when an update of this submission supplied [a, a+n).
static int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    for (uint32_t i = UPD.n, end = UPD.n > UPD_N ? UPD.n - UPD_N : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        if (a >= e->hi || a + n <= e->lo) continue;
        if (a < e->lo || a + n > e->hi) return 0;   // partly covered: memory
        memcpy(out, e->data + (a - e->lo), n);
        return 1;
    }
    return 0;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short. The
// block has no size field.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    const uint8_t *sb = vk_usnap_bytes(u->addr + off, 8);
    if (sb) { memcpy(&a, sb, 8); return a; }
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

// Guarded NVN 2 candidates use the same bytes that the dispatch binds as uniform banks. Bank 0 includes pushed
// roots and packed descriptor tables; bank 1 is the program's bounded compiler-constant data. No scalar is guessed
// to be a pointer here: bind_range determines whether a candidate has backing, and the shader checks the true range.
typedef struct { uint8_t bank[VK_C0_BYTES]; int valid; } GuardedC0Bank;

static const uint8_t *guarded_c0_bank(const VkExec *ex, GuardedC0Bank *scratch) {
    // Every c0 candidate of this descriptor build reads the same parameter bank. Rebuild on the next call to
    // bind_resources, so guest writes and intervening command-stream updates are observed by the next dispatch.
    if (!scratch->valid) {
        vk_c0_fill(scratch->bank, NVN_STAGE_CS, ex->ssbo[NVN_STAGE_CS], ex->ubo[NVN_STAGE_CS], c0_dim, c0_binst);
        scratch->valid = 1;
    }
    return scratch->bank;
}

static p_bind_buf guarded_storage_binding(const VkExec *ex, int st, const GfxShaderRes *r, GuardedC0Bank *scratch) {
    uint64_t off = (uint64_t)r->handle * 4;
    p_bind_buf b = {0};
    if (st != NVN_STAGE_CS || !c0_lay) return b;
    if (r->cbuf_slot == 0) {
        if (!c0_on() || off > VK_C0_BYTES - 8) return b;
        const uint8_t *bank = guarded_c0_bank(ex, scratch);
        memcpy(&b.addr, bank + off, 8);
        size_t ub = nvn2_c0_part(c0_lay, 1), end = nvn2_c0_part(c0_lay, 3);
        if (off >= ub && off < end && (off - ub) % 16 == 0) {
            uint32_t bytes;
            memcpy(&bytes, bank + off + 8, 4);
            b.size = bytes;   // preserve a typed table's declared extent, not the remainder of its pool
        }
    } else if (r->cbuf_slot == 1 && ex->prog) {
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (!find_stage(ex->prog, st, &ctl, &code, &len)) return b;
        memcpy(&c1, ctl + 0x700, 4);
        if (c1 > len || off > len - c1 || len - c1 - off < 8) return b;
        memcpy(&b.addr, code + c1 + off, 8);
    } else if (r->cbuf_slot == 17) {
        // ubo_info binds the synthetic compute push bank here, before its user-UBO mapping. Unwritten words are zero.
        if (off > sizeof ex->push[1][0] - 8) return b;
        uint32_t words[2] = {0};
        for (unsigned i = 0; i < 2; i++) {
            unsigned word = r->handle + i;
            if (ex->push_set[1][0] >> word & 1) words[i] = ex->push[1][0][word];
        }
        memcpy(&b.addr, words, 8);
    } else if (ubo_ptr_res(r)) {
        b.addr = ubo_ptr(ex, st, r);
    }
    return b;
}

// Explicit NVN 2 compute storage-buffer table slots (c0 part 2..3) carry the exact byte size passed to
// nvnCommandBufferBindStorageBuffer, unlike speculative root/c1/UBO pointer candidates whose unbounded pool remainder
// would overwrite adjacent CPU-written buffers if mirrored back to host RAM.
static int guarded_is_declared_ssbo(int st, const GfxShaderRes *r, const p_bind_buf *b) {
    if (st != NVN_STAGE_CS || !c0_lay || r->cbuf_slot != 0 || !b || !b->size) return 0;
    size_t sb = nvn2_c0_part(c0_lay, 2), end = nvn2_c0_part(c0_lay, 3), off = (size_t)r->handle * 4;
    return off >= sb && off < end && (off - sb) % 16 == 0;
}

typedef struct { PtrEntry e[PTR_ENTRIES]; int valid; } BindingPtrTable;

static p_bind_buf pointer_binding(VkExec *ex, int st, uint32_t off, BindingPtrTable *ptr) {
    // Array elements are entries of one table snapshot. The caller provides fresh scratch per binding, so no
    // result survives a draw, another binding, or an intervening op that could change its source buffers.
    if (!ptr->valid) {
        if (st == NVN_STAGE_TS || st == NVN_STAGE_MS)
            pointer_table4(ex->ssbo[NVN_STAGE_TS], ex->ssbo[NVN_STAGE_MS], ex->ssbo[NVN_STAGE_VS],
                           ex->ssbo[NVN_STAGE_CS], ptr->e);
        else
            pointer_table(ex->ssbo[st], ptr->e);
        ptr->valid = 1;
    }
    uint32_t i = (off - PTR_TABLE) / 16u;
    return i < PTR_ENTRIES ? (p_bind_buf){.addr = ptr->e[i].addr, .size = ptr->e[i].size} : (p_bind_buf){0};
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written, BindingPtrTable *ptr,
                                      GuardedC0Bank *guarded_c0) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
    int ubo = 0, slot = -1;
    p_bind_buf up = {0};
    const p_bind_buf *b = NULL;
    int guarded = !!(r->flags & GFX_RES_SSBO_GUARDED);
    if (!guarded) slot = ssbo_slot(r, st, &ubo);
    if (guarded) up = guarded_storage_binding(ex, st, r, guarded_c0), b = &up;
    else if (slot == -2) {   // c0+0x880 Hookshot pointer table entry
        uint32_t off = (uint32_t)r->handle * 4u;
        up = pointer_binding(ex, st, off, ptr);
        b = &up;
    } else if (slot >= 0 && slot < 16) b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    else if (ubo_ptr_res(r)) up.addr = ubo_ptr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    if (b) {
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        uint64_t at = b->addr - lead;
        int bare = !b->size;
        int mesh = st == NVN_STAGE_TS || st == NVN_STAGE_MS;
        int may_write = written && slot != -2;
        int real_write = may_write && (!guarded || guarded_is_declared_ssbo(st, r, b));
        int wb = real_write;
        // Read-only SSBOs bind to the end of their backing (up to maxStorageBufferRange): NVN shaders index LDG.E
        // past the nominal size in c0 (HUD glyph tables, MKW tree/instance/material records; adriadam10/mkw-recomp
        // vk_exec_bind.c, MIT, Copyright (c) 2026 Adrià). UBOs, written SSBOs, guarded slots, and pointer-table
        // entries keep their explicit size.
        uint64_t want = (!ubo && !may_write && !guarded && slot != -2) ? 0 : (bare ? 0 : b->size + lead);
        // A small read-only vertex storage buffer the game writes on the CPU: its bytes as of the submit call
        // (vk_usnap.c MKW_SSBO_SNAPSHOT; skinned pedestrians' bone matrices, Crown City sheets).
        if (st == NVN_STAGE_VS && !may_write && !guarded && !ubo && slot >= 0 && !bare &&
            vk_usnap_find_ssbo(at, b->size + lead, align, &bi)) {
            count_use(VK_MCLS_SCRATCH, VK_MUSE_SSBO, bi.range);
            return bi;
        }
        if (mesh && at) {
            VkMemSpan sp;
            if (!vk_mem_resolve(at, 1, &sp)) return zero_info(max);
            const uint64_t import_cap = 1ull << 20;
            if (!want) {
                uint64_t cap = written ? import_cap : max;
                if (cap > max) cap = max;
                want = sp.avail < cap ? sp.avail : cap;
            } else if (want > max) {
                want = max;
            }
            if (!sp.alloc && written && slot != -2 && want > import_cap) want = import_cap;
            // Bare+written: bind the guessed range without write-back, then stamp only the head (Hookshot).
            if (bare && wb) wb = 0;
        }
        int bound_ok = bind_range(ex, at, want, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO, may_write, wb, guarded, &bi);
        if (vk_dump_armed && st == NVN_STAGE_CS && vk_dump_active()) {   // capture frame: what each compute binding got
            static int nlog;
            const uint8_t *dctl, *dcode = NULL;
            uint32_t dlen;
            uint64_t dorg = ex->prog && find_stage(ex->prog, NVN_STAGE_CS, &dctl, &dcode, &dlen) ? nvn2_origin(dcode) : 0;
            if (nlog++ < 6000)
                rt_log("dbind cs %016llx %s cbuf %d handle %#x slot %d addr %#llx size %#llx lead %#llx want %#llx -> %s offset %#llx range %#llx",
                       (unsigned long long)dorg, ubo ? "ubo" : "ssbo", r->cbuf_slot, r->handle * 4, slot, (unsigned long long)b->addr, (unsigned long long)b->size,
                       (unsigned long long)lead, (unsigned long long)want, bound_ok ? "bound" : "FAILED",
                       (unsigned long long)bi.offset, (unsigned long long)bi.range);
        }
        if (bound_ok) {
            if (real_write && bi.buffer != VK_NULL_HANDLE) {
                uint64_t wb_n = bi.range;
                if (mesh && bare) {
                    uint64_t head = (b->addr - at) + 0x40;
                    if (head > wb_n) head = wb_n;
                    wb_n = (head + 3) & ~3ull;
                    if (wb_n) {
                        VkMemSpan sp;
                        if (vk_mem_resolve(at, 1, &sp)) {
                            VK_MEM_USE(sp);
                            vk_mem_gpu_wrote(&sp, wb_n);
                        }
                        note_written(ex, at, wb_n);
                    }
                } else {
                    note_written(ex, at, wb_n);
                }
                const uint8_t *ctl, *code;
                uint32_t len;
                if (vk_sync_noting() && ex->prog && find_stage(ex->prog, st, &ctl, &code, &len))
                    vk_sync_note_write(at, wb_n, nvn2_origin(code));
            }
            return bi;
        }
        if (b->addr && !guarded) warn_once(W_SSBO_ALIGN, "storage buffer %#llx is not backed or not aligned: a zero buffer is bound",
                               (unsigned long long)b->addr);
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
        if (vk_dump_armed && vk_dump_active()) {
            static int nz;
            const uint8_t *zc, *zcode = NULL;
            uint32_t zl;
            uint64_t zo = ex->prog && find_stage(ex->prog, st, &zc, &zcode, &zl) ? nvn2_origin(zcode) : 0;
            if (nz++ < 2000)
                rt_log("dbind st%d %016llx ssbo cbuf %d handle %#x NOT REPLAYED (zero buffer)%s", st, (unsigned long long)zo,
                       r->cbuf_slot, r->handle * 4, written ? " written" : "");
        }
    }
    bi = zero_info(max);
    // Vulkan requires a nonzero descriptor range. Both unguarded and guarded unbacked fallback users bind the full dummy buffer
    // so any out-of-bounds writes (such as offset 0x1000) land safely in mapped zero storage without triggering GPU MMU page faults on AMD LLPC.
    return bi;
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// TotK's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid Tulin's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) {
        uint32_t guest_l0 = v->base_level;
        uint32_t l0 = guest_l0 >= img->host_mip0 ? guest_l0 - img->host_mip0 : 0;
        uint32_t nl = v->nlevels ? v->nlevels : img->levels - l0;
        if (l0 >= img->levels) l0 = img->levels ? img->levels - 1 : 0;
        if (l0 + nl > img->levels) nl = img->levels - l0;
        r.l0 = l0;
        r.l1 = l0 + nl;
    }
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// TOTK_TEX_MEMO (tex_info, feedback_scan): 0 off, 1 on, 2 (=check) computes every hit again and stops on a difference.
static int td_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_value("TOTK_TEX_MEMO");
        on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return on;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
static struct { uint64_t context, h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    descriptor_state(ex);
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    int st = nvn_stage_of(gfx_shader_stage(fs));
    if (st < 0) return 0;
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode(), locked = 0;
    for (int r = 0; r < n; r++)
        for (int k = 0; k < (res[r].count ? res[r].count : 1); k++) {
            int slot = res[r].slot + k;
            uint64_t h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (memo && f->context == descriptor_serial && f->h == h && slot_still(ex, &f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    if (!locked) vk_texture_lock(), locked = 1;
                    SubRange c = sampled_sub(ex, h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(ex, h);   // before the lookup
            if (!locked) vk_texture_lock(), locked = 1;
            SubRange sr = sampled_sub(ex, h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.context = descriptor_serial, .h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

static int geometry_feedback_safe(VkExec *ex, const GfxShader *vs, const GfxShader *gs) {
    // The existing in-pass feedback dependency is fragment-only. Pre-raster reads of an attachment need a different
    // synchronization contract; reject them rather than reuse that by-region fragment barrier.
    if (gs && (feedback_scan(ex, vs) || feedback_scan(ex, gs))) {
        warn_once(W_GEOMETRY, "experimental geometry draw samples an attachment before rasterization: skipped");
        return 0;
    }
    return 1;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier). Even when no draw in
// the open pass has written the attachment yet (fb_dirty == 0, e.g. read-only depth decals while depth is bound as an
// attachment in GENERAL), Vulkan requires ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT and the pipeline feedback flag whenever
// a shader samples an attachment of the active render pass (VUID-vkCmdDraw-None-09000), otherwise Hi-Z/ZCULL or DCC
// compression on discrete GPUs can return stale or uncompressed values.
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    uint32_t learn = 0;
    if (mode == VK_FEEDBACK_LAYOUT && (m & ~ex->fbl_mask)) {
        for (uint32_t i = 0; i < 9; i++) {
            const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
            if (((m & ~ex->fbl_mask) >> i & 1) && t->img && t->img->fb_usage) {
                ((VkTexImage *)t->img)->fb_learned = 1;
                learn |= 1u << i;
            }
        }
    }
    if (learn || (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0))) {
        vk_exec_pass_end(ex);
        vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Marks the attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with
// depth writes or the stencil test on.
static void feedback_written(VkExec *ex, const VkPipeState *s) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    if (ex->depth.image && !(ex->fb_dirty & 0x100) && ((s->ds.dt && s->ds.dw) || s->ds.st)) ex->fb_dirty |= 0x100;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && !(ex->fb_dirty >> i & 1) && vk_pipeline_target_wmask(s, (int)i)) ex->fb_dirty |= 1u << i;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);

// Sampled-texture descriptors per (handle, resource type, separate), kept while nothing they derive from changed: the
// slot's resolution and texture entry (slot_still) and the samplers (vk_sampler_gen); the layout is set again on every
// hit from the images in the feedback layout (in_fbl of the image kept with it). The first binding of a slot in a submission (vk_texture_epoch) still makes the slot lookup
// for its storage check; a re-upload writes the same image, so the descriptor stays. Draws bind the same textures
// thousands of times a frame (15,000 lookups a frame in the monsters fight); a hit skips the slot lookup, the view key
// with its hash probe and the sampler lookup. Storage images, which note their binding per submission, are
// not kept. Direct-mapped; texture lock held. TOTK_TEX_MEMO=0 turns it off; =check computes every hit again and stops
// on a difference.
#define TD_N 8192
static struct {
    uint64_t context, h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    descriptor_state(ex);
    int td_on = td_mode();
    VkImage image;
    if (!h || storage || !td_on) return tex_info_full(ex, r, h, storage, separate, out, &image);
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->context == descriptor_serial && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(ex, &m->gen, m->tok, h, gen)) {
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        if (td_on == 2) {
            VkDescriptorImageInfo full;
            tex_info_full(ex, r, h, storage, separate, &full, &image);
            if (full.sampler != out->sampler || full.imageView != out->imageView || full.imageLayout != out->imageLayout)
                rt_fatal("vk_exec: texture descriptor memo of handle %#llx type %#x differs from a lookup",
                         (unsigned long long)h, r->type);
        }
        return;
    }
    uint64_t tok = token_of(ex, h);   // before the lookup
    tex_info_full(ex, r, h, storage, separate, out, &image);
    *m = (typeof(*m)){.context = descriptor_serial, .h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
        if (img) view = get_view(ex, img, &v, hv, r->type, storage), *image = img->image;
        if (img && storage) vk_texture_storage_bound_locked(img);
        if (img && storage && img->host_mip0)
            vk_ctrace(img->addr, "storage image CompactLayers %#llx bound for writes (view lv %u layers %u+%u)",
                      (unsigned long long)img->addr, hv && v.set_levels ? v.base_level : 0, hv && v.set_layers ? v.min_layer : 0,
                      hv && v.set_layers ? v.nlayers : 0);
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch, whatever the sampler's filter:
            // blending neighbouring depths puts reconstructed positions in mid-air at object edges (shadow masks).
            // Integer formats (UINT/SINT and STENCIL) cannot use linear filtering in Vulkan.
            int sid = (int)(h >> 20 & 0xfff);
            VkFormat vfmt = img ? sampled_view_format(img, &v, hv, 0, NULL, NULL) : VK_FORMAT_UNDEFINED;
            int is_int = img && (vk_exec_view_aspect(img, &v, hv) == VK_IMAGE_ASPECT_STENCIL_BIT ||
                                 vk_fmt_num_type(vfmt) != GFX_TYPE_FLOAT);
            int point = img && (is_int || ((img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                                           vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT));
            // CompactLayers: guest LOD is computed for the full guest pyramid; host level 0 is guest mip
            // host_mip0. Bias (and clamp) by -host_mip0 so distance LOD matches guest intent on the trimmed
            // stack — without it close terrain stays soft and LOD steps look like pop-in. Default off
            // (TOTK_COMPACT_LOD_BIAS=1 to enable). Sampler cache keys the adjusted samp_par (few host_mip0 values).
            // TOTK_COMPACT_LOD_BIAS: 0=off (n148 sand returned, cliffs still rainbow), 1=full -host_mip0
            // (n139 black near ground), 2=half -host_mip0/2 (compromise for overnight cliffs).
            static int compact_lod = -1;
            if (compact_lod < 0) {
                const char *e = cfg_get("TOTK_COMPACT_LOD_BIAS");
                if (!e || !*e) compact_lod = 0;   // unset: no bias, as Eglantine (derivative LOD fits the trimmed image)
                else if (*e == '0') compact_lod = 0;
                else if (*e == '2') compact_lod = 2;
                else compact_lod = 1;
            }
            samp_par sp;
            if (!point && compact_lod && img && img->host_mip0 && vk_sampler_params(&ex->samp_pool, sid, &sp)) {
                float skip = (float)img->host_mip0;
                if (compact_lod == 2) skip *= 0.5f;
                sp.lod_bias -= skip;
                sp.lod_min = fmaxf(0.f, sp.lod_min - skip);
                sp.lod_max = fmaxf(sp.lod_min, sp.lod_max - skip);
                samp = vk_sampler_for(&sp);
            } else {
                samp = point ? vk_sampler_point_by_id_locked(&ex->samp_pool, sid)
                             : vk_sampler_by_id_locked(&ex->samp_pool, sid);
            }
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// TotK's compute passes sample an R32_SFLOAT texture through a shadow sampler. Maxwell compares any single-channel
// float or unorm texel; Vulkan only formats with SAMPLED_IMAGE_DEPTH_COMPARISON (VUID-vkCmdDispatch-None-06479), which
// AMD's drivers do not give R32_SFLOAT, so the result there is undefined. Such an image gets a depth twin of the same
// size and bits (R32_SFLOAT -> D32_SFLOAT, R16_UNORM -> D16_UNORM), copied through a buffer right before each dispatch
// that samples it (a dispatch is outside any render pass), and the dispatch samples the twin. Draws keep the colour
// image: a copy cannot go inside their pass. Design: Savage Wind Recomp runtime/nvn/vulkan/exec.c depth_twin_update
// (https://github.com/Revivedx/SavageWindRecomp, MIT); the code is ours. TOTK_DEPTH_TWIN=0: off.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static int twin_dispatch;                // op_dispatch*: the resources being bound are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    static int on = -1;
    if (on < 0) on = rt_knob("DEPTH_TWIN", "1");
    if (!on || (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding img's current texels, the copies recorded into ex->cb; NULL when img needs none or it cannot
// be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin.
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        plat_mutex_lock(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        plat_mutex_unlock(&spare_lock);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
}

// TOTK_BIND_STATS=1: bindings and descriptors per draw of each stage (Task 2.2's choice of the pushed stage), and how
// often a draw repeats the previous one, logged every 20000 draws: pushes recorded and skipped, allocated sets written
// and reused, stage texture-type recomputations (answers_for), draws with the previous draw's program, bound state
// and textures (shaders and pipeline would be the same) and those that also have its buffers and viewport (all
// descriptors would be the same). BS is declared beside DS.
static int bind_stats_on(void) {
    if (BS.on < 0) {
        const char *e = cfg_get("TOTK_BIND_STATS");
        BS.on = e && *e && strcmp(e, "0");
    }
    return BS.on;
}

static void bind_stats(const VkPipeInfo *info) {
    if (!bind_stats_on() || info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        int fs = info->n_sets > 1 ? b->set : !(b->stages & VK_SHADER_STAGE_VERTEX_BIT);
        BS.bind[fs]++, BS.desc[fs] += b->count ? b->count : 1;
    }
    if (++BS.draws % 20000 == 0) {
        double d = (double)BS.draws;
        rt_log("vk_exec: per draw over %llu draws: vertex %.2f bindings %.2f descriptors, fragment %.2f bindings %.2f descriptors",
               (unsigned long long)BS.draws, BS.bind[0] / d, BS.desc[0] / d, BS.bind[1] / d, BS.desc[1] / d);
        rt_log("vk_exec: per draw: pushes %.3f skipped %.3f, sets written %.3f reused %.3f, texture types %.3f; "
               "same program and state as the draw before %.3f, also same buffers %.3f; descriptor builds skipped %.3f",
               BS.push / d, BS.push_skip / d, BS.set_new / d, BS.set_reuse / d, BS.ac_miss / d, BS.same_prog / d,
               BS.same_all / d, BS.bind_skip / d);
    }
}

// The bound state a draw reads, compared with the previous draw's (TOTK_BIND_STATS only).
static void repeat_stats(const VkExec *ex, const VkPipeInfo *info) {
    static struct {
        const prog_t *prog;
        VkPipeState pipe;
        uint64_t tex[2][128], img[2][32];
        const VkPipeInfo *info;
        p_bind_buf ubo[2][18], ssbo[2][16];
        float vp[6], point_size;
    } last, cur;
    if (!bind_stats_on()) return;
    memset(&cur, 0, sizeof cur);
    cur.prog = ex->prog, cur.pipe = ex->pipe, cur.info = info;
    const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int k = 0; k < 2; k++) {
        memcpy(cur.tex[k], ex->tex[st[k]], sizeof cur.tex[k]), memcpy(cur.img[k], ex->img[st[k]], sizeof cur.img[k]);
        memcpy(cur.ubo[k], ex->ubo[st[k]], sizeof cur.ubo[k]), memcpy(cur.ssbo[k], ex->ssbo[st[k]], sizeof cur.ssbo[k]);
    }
    memcpy(cur.vp, ex->vp[0], sizeof cur.vp), cur.point_size = ex->point_size;
    size_t prog_part = offsetof(typeof(cur), info);
    if (!memcmp(&cur, &last, prog_part)) {
        BS.same_prog++;
        if (!memcmp((const char *)&cur + prog_part, (const char *)&last + prog_part, sizeof cur - prog_part)) BS.same_all++;
    }
    last = cur;
}

static int push_skip_on(void) {   // TOTK_PUSH_SKIP=0: every graphics draw pushes its set again
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_PUSH_SKIP");
    }
    return on;
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its
// sets: every write would be the same, the push would be skipped and the allocated set reused without a bind (about a
// quarter of field and monsters2 draws, TOTK_BIND_STATS). The inputs, compared and stored in place, are what
// bind_resources and its helpers read beyond the pipeline info's fixed bindings and Z's stand-ins:
// - the info (bindings, layouts), the program and its serial (the compiler-constant range of UBO slot 1);
// - vk_texture_slots_gen, vk_texture_epoch (a slot's image, view and once-per-submit storage check), vk_sampler_gen;
// - vk_mem_gen (buffer and offset of a span, mirror creation and per-submission sync);
// - the images in the feedback layout (the sampled layout);
// - per binding: the support inputs, c0's storage table and base instance, the bound uniform and storage ranges, the
//   texture and image handles (RsList; compared only when a bind op ran since the last draw, bind_gen);
// - storage buffer addresses read from user uniform blocks (RsList.p, ubo_ptr): compared on every draw, since the
//   block's contents change without a bind op.
// The side effects of a build repeat as no-ops in the same command buffer and submission: c0 and support buffers kept
// in B, note_written's ranges, vk_mem_gpu_wrote's write-backs, the per-submission mirror sync and dirty census note,
// the views cached per submission. The census counts are added again. A build that copied guest bytes into transient
// memory (an unaligned host range) is never skipped after: the copy may differ. B.res_valid says the sets bound hold
// the last build; vk_exec_draw_new_cb's memset clears it, and every build clears it until it ends.
// TOTK_BIND_SKIP=0 builds every draw; =check builds every draw and stops when a draw with equal inputs gets other
// writes, census counts or descriptor commands.
static int bind_skip_on(void) {
    if (RS.on < 0) {
        const char *e = cfg_value("TOTK_BIND_SKIP");
        RS.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
        if (RS.on != 1) rt_log("vk_exec: descriptor build skip %s", RS.on ? "check" : "off");
    }
    return RS.on;
}

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        static RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings && ok; i++) {
            const VkPipeBinding *b = &info->bindings[i];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0 && c0_on()) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                int ubo, slot = ssbo_slot(r, st, &ubo);
                if (slot >= 0 && slot < 16) {
                    rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                } else if (ubo_ptr_res(r)) {
                    if (np == RS_PTRS) { ok = 0; break; }
                    rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. The bound-slot bytes
// are compared only when a bind op ran since they were stored or the pipeline changed: otherwise they are the same.
// Check mode compares them anyway and stops when the bind op generation said they could not differ but they do.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT(ex->tex_pool.addr);
    PUT(ex->tex_pool.count);
    PUT(ex->samp_pool.addr);
    PUT(ex->samp_pool.count);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
    for (int i = 0; i < l->np; i++) PUT(ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast && RS.on != 2) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    if (fast && bd) rt_fatal("vk_exec: descriptor build skip check: bound slots changed without a bind op");
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Writes the pipeline's sets from the bound state: set 0, and set 1 with independent layouts. Each is pushed into the
// command buffer, or allocated, written and bound; a graphics set whose contents equal the last one allocated for that
// set number in this command buffer is reused, and bound again only when another layout or a push came between. A
// graphics push equal to the last one at its set number, with no other layout or bind between, is not recorded.
// The descriptors go into one packed array d, a VkDescSlot per descriptor in binding order, which a set's update
// template reads directly (VkPipeInfo.tpl); write structs are filled only when a set has no template.
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    static VkWriteDescriptorSet w[MAX_BINDINGS];
    static VkDescSlot d[MAX_DESC];
    static VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    bind_stats(info);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    int skip = gfx ? bind_skip_on() : 0, same = 0, was_valid = B.res_valid;
    if (skip) {
        same = bind_sig(ex, info);
        if (same && was_valid && skip == 1) {
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            BS.bind_skip++;
            ex->recorded++;
            return;
        }
        B.res_valid = 0;   // until the build below ends
        if (skip == 2) memcpy(RS.pcnt, RS.cnt, sizeof RS.cnt), RS.pncnt = RS.ncnt;
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0, RS.cmds = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0}, d0[2] = {0, 0};   // bindings [lo, hi) and first descriptor per set
    int need_w = 0;                                             // a set without a template: fill the write structs
    for (uint32_t s = 0, i0 = 0; s < n_sets; s++) {
        uint32_t i1 = i0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s)) i1++;
        lo[s] = i0, hi[s] = i1, i0 = i1;
        if (i1 > lo[s] && !info->tpl[s]) need_w = 1;
    }
    uint32_t nd = 0;
    MeshPtrTable mesh_table = {0};      // shared only by task/mesh c0 banks built in this call
    BindingPtrTable ptr_mesh = {0};     // Hookshot table from TS+MS+VS+CS SSBOs
    BindingPtrTable ptr_other = {0};    // single-stage table for ordinary VS/FS/CS -2 slots
    GuardedC0Bank guarded_c0;           // lazily filled; ordinary graphics draws need none of its bank bytes
    guarded_c0.valid = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage), kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k, &mesh_table);
        } else if (b->kind == GFX_RES_SSBO) {
            BindingPtrTable *ptr = (st == NVN_STAGE_TS || st == NVN_STAGE_MS) ? &ptr_mesh : &ptr_other;
            for (uint32_t k = 0; k < cnt; k++) {
                GfxShaderRes rk = *r;
                // Array descriptors at one c0 base (AddGlobalTables count=3 at 0x880): consecutive 16-byte entries.
                if (cnt > 1 && r->cbuf_slot == 0 && (uint32_t)r->handle * 4u + 16u * k < VK_C0_BYTES)
                    rk.handle = (uint16_t)(r->handle + (int)k * 4);
                o[k].buffer = ssbo_info(ex, st, &rk, r->flags & 1, ptr, &guarded_c0);
            }
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                vk_texture_lock();   // tex_info looks up with the texture lock held
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    uint64_t h = 0;
                    if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    tex_info(ex, r, h, img, r->separate, &o[k].image);
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                }
                vk_texture_unlock();
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    // Set 1's first descriptor is noted when the loop reaches its first binding. A set 1 without bindings (a fragment
    // stage with no resources) starts at the end instead, so set 0's byte count (d0[1] - d0[0]) covers its descriptors.
    // Left at 0, it pushed set 0's template with 0 bytes copied into the recording, and the driver read the
    // descriptors from whatever followed in the recording.
    if (n_sets == 2 && lo[1] == info->n_bindings) d0[1] = nd;
    // Under the validation layer, a new graphics layout binds set 1 before set 0 is pushed or bound. The layer (SDK
    // 1.4.363, PipelineLayoutCompatDef::operator==) takes any two INDEPENDENT_SETS layouts as compatible, so
    // vkCmdPushDescriptorSetKHR keeps the push set it made for an earlier layout and drops writes to bindings that
    // layout lacks (false VUID-vkCmdDrawIndexed-None-08114, then an access violation inside the layer). A bind at set 1
    // disturbs set 0 by its canonical id, which drops the layer's push set: the push after it gets a new one.
    int set1_first = 0;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
        set1_first = n_sets == 2 && vk_ctx()->validation;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = set1_first ? 1 - k : k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (s && i1 == i0) {               // no fragment stage: set 1 is empty and unused
            if (set1_first) {              // bound only to drop the layer's push set (comment above)
                VkDescriptorSet set = alloc_set(ex, info->set_layout1);
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 1, 1, &set, 0, NULL);
                B.ds_bound[1] = set, B.ds_pushed[1] = 0;
                RS.cmds++;
            }
            continue;
        }
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx && push_skip_on() ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                BS.push_skip++;
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            BS.push++;
            RS.cmds++;
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
                BS.set_new++;
                RS.cmds++;
            } else {
                BS.set_reuse++;
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
                RS.cmds++;
            }
        }
    }
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push (TOTK_PUSH_SKIP on) and nothing made
        // the build unrepeatable; allocated sets are always bound to the one written last.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        if (skip == 2) {   // with the pipeline info equal (bind_sig), the packed descriptors determine every write
            uint32_t n = (uint32_t)(nd * sizeof *d);
            int wr = n != RS.ckn || memcmp(RS.ck, d, n);
            int cn = RS.ncnt != RS.pncnt || memcmp(RS.cnt, RS.pcnt, sizeof RS.cnt[0] * (size_t)RS.ncnt);
            if (same && was_valid && (wr || cn || RS.cmds))
                rt_fatal("vk_exec: descriptor build skip check: a draw of program %llu with the inputs of the previous one "
                         "gets %s", (unsigned long long)ex->prog->serial,
                         RS.cmds ? "descriptor commands" : wr ? "other writes" : "other census counts");
            if (same && was_valid) BS.bind_skip++;   // the draws a skip would have spared, compared instead
            RS.ckn = n;
            memcpy(RS.ck, d, n);
        }
        B.res_valid = valid;
    }
    ex->recorded++;
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    vkCmdBindPipeline(ex->cb, bp, p);
    if (vk_ctx()->has_nv_checkpoints && vk_checkpoints_on()) {   // MKW_GPU_CHECKPOINTS=1: name the shaders the checkpoint's pipeline holds
        uint64_t o[3] = {0};
        const uint8_t *ctl, *code;
        uint32_t len;
        static const int stages[2][3] = {{NVN_STAGE_VS, NVN_STAGE_GS, NVN_STAGE_FS}, {NVN_STAGE_CS, -1, -1}};
        for (int i = 0; i < 3 && ex->prog; i++)
            if (stages[k][i] >= 0 && find_stage(ex->prog, stages[k][i], &ctl, &code, &len)) o[i] = nvn2_origin(code);
        vk_checkpoint_note(p, o[0], o[1], o[2]);
        rec_checkpoint(ex->cb, p);
    }
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    if (ex->dirty & VK_EXEC_DIRTY_VIEWPORT) {
        float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
        if (ex->has_vp) x = ex->vp[0][0], y = ex->vp[0][1], w = ex->vp[0][2], h = ex->vp[0][3];
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        VkViewport v = {x, y + h, w, -h, ex->depth_range[0], ex->depth_range[1]};   // the y flip: negative height
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

static void bind_vertex_buffers(VkExec *ex) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range is the one its bound buffer was resolved from, with no pool or mirror change since
    // (vk_mem_gen, read before any resolve), resolves to that buffer again: most draws change one stream or none.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    // Vertex bindings persist independently: most draws replace just one stream. Keep one command, but send only
    // the span containing changed bindings instead of rebinding all 17 (including the constant-zero stream).
    unsigned first = 0, end = 17;
    if (B.vb_valid) {
        first = 17, end = 0;
        for (unsigned b = 0; b < 17; b++)
            if (B.vb[b] != vb[b] || B.vo[b] != vo[b] || B.vsz[b] != vsz[b] || (!dyn && B.vstr[b] != vstr[b])) {
                if (first == 17) first = b;
                end = b + 1;
            }
        if (!end) return;
    }
    vkCmdBindVertexBuffers2(ex->cb, first, end - first, vb + first, vo + first, vsz + first, dyn ? NULL : vstr + first);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = 1;
}

// TOTK_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait.
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("TOTK_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (MKW_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (MKW_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program, bound pipeline state, the vertex and fragment stages' first 32
// texture and image handles and every slot's resolution (vk_texture_slots_gen, read unlocked: a registration racing
// with it is ordered after this draw, as answers_for's own check would order it) stay the same: answers_for reads
// nothing else, so shader_for would return the same shaders. About 65% of field draws repeat them
// (TOTK_BIND_STATS). TOTK_SHADER_MEMO=0 turns it off; =check looks the shaders up on every hit too, and every slot's
// texture type in answers_for, and stops on a difference.
// tex_serial (bumped by every texture and image bind) equal to the key's means the handles were not rebound since
// they were compared, so the 1 KiB compare is skipped; a rebind to the same handles still matches by the compare.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[3][32], img[3][32];
    uint8_t geometry_input;
    VkPipeState pipe;
} MemoKey;

// Workaround for the GPU fault after the first guide movie (#9). On NVIDIA (drivers 617.14, 617.42 and 621.01) the
// draws of the NVN 2 reference title's instanced light quads whose vertex stage fetches ten per-instance streams (an input at Location
// 9) and whose fragment stage reads the storage buffer at c0+0x530 fault the device with an invalid instruction
// pointer, or crash the driver's shader compiler. Leaving out one such shader moves the fault to another of the
// family, and the same SPIR-V drawn on its own does not fault, so the cause is still open. Skipping these draws
// loses those lights; the family's quads without an input at Location 9 draw as before. It is a mitigation, not a fix:
// in 600 s runs on an RTX 5060 Ti the fault came minutes later in one run of two (another shader), where without it
// every run faulted right after the first guide movie. TOTK_FIX_VS_FAULT=0 draws them.
//
// The later fault, around seven minutes in, comes from one more quad of the same shape whose fragment stage does not
// read c0+0x530: NVN 2 vertex stage cae0ef8e9fe9147f, nine per-instance RGBA32F streams, one instance. In 9 runs of
// 600 s every run where it drew faulted, with its draw among those in flight at the device loss, and the runs where it
// never drew did not; skipping it gave 0 faults in 4 runs. Quads that never fault have its shape: vertex stages
// 782b2585777d2b4e, 392e2956038a9e64 and 7d85669ac430f01a read Location 9, feed every attribute per instance and carry
// the same integer code (four reads of c9 at a computed index), and skipping every such quad left out ten vertex stages
// that drew in all 17 logged runs. So it is matched by the name of its NVN 2 stage.
// LOCAL DIAGNOSTIC, not for a PR. MKW_PAIR_LOG=1 logs each new (vs, fs, gs) stage set at its first draw, to name the
// draw behind a lost device. MKW_SKIP_STAGE=<hex>[,<hex>] skips the draws that use one of those NVN 2 stages.
// ponytail: a fixed table and list, enough for one session's stage sets; it stops logging when full.
static int stage_diag_skip(const VkExec *ex, const GfxShader *vs) {
    static int on = -1, nskip;
    static uint64_t skip[16], seen[1 << 16];
    if (on < 0) {
        const char *e = getenv("MKW_SKIP_STAGE"), *l = getenv("MKW_PAIR_LOG");
        for (char *end; e && *e && nskip < 16; e = *end ? end + 1 : end) skip[nskip++] = strtoull(e, &end, 16);
        on = (l && *l == '1') ? 2 : nskip > 0;
    }
    if (!on) return 0;
    static const int st[3] = {NVN_STAGE_VS, NVN_STAGE_FS, NVN_STAGE_GS};
    uint64_t o[3] = {0}, h = 1469598103934665603ull;
    for (int i = 0; i < 3; i++) {
        const uint8_t *ctl, *code = NULL;
        uint32_t len;
        if (find_stage(ex->prog, st[i], &ctl, &code, &len)) o[i] = nvn2_origin(code);
        h = (h ^ o[i]) * 1099511628211ull;
        for (int k = 0; k < nskip; k++) if (o[i] && o[i] == skip[k]) return 1;
    }
    if (on == 2) {
        if (!h) h = 1;
        for (uint32_t i = (uint32_t)h & 0xffff, n = 0; n < 1 << 12; i = (i + 1) & 0xffff, n++) {
            if (seen[i] == h) break;
            if (!seen[i]) {
                seen[i] = h;
                GfxShaderIo io[32];
                rt_log("stage set: vs %016llx fs %016llx gs %016llx inputs %08llx", (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], vs ? (unsigned long long)gfx_shader_inputs(vs, io) : 0ull);
                break;
            }
        }
    }
    return 0;
}

static int light_quad_fault_skip(const VkExec *ex, const GfxShader *vs, const GfxShader *fs, int32_t count) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("MKW_FIX_VS_FAULT");
        on = e && *e == '1';
    }
    if (!on || !vs) return 0;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t orig = 0;
    if (find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len)) {
        orig = nvn2_origin(code);
        if (orig == 0xcae0ef8e9fe9147full ||
            orig == 0xa05d707357ffc917ull || orig == 0xc081588c78ee16f3ull || orig == 0x89300537d4fff15eull ||
            orig == 0xfb6e80e4f0bd1ac5ull || orig == 0x140a1eac75f7b5c0ull || orig == 0x163bd4735277403aull) {
            warn_once(W_FAULT_SKIP2, "the light quad of NVN 2 vertex stage %016llx is skipped (#9/cutscene)", (unsigned long long)orig);
            return 1;
        }
    }
    // Whitelist known-safe gameplay quads
    if (orig == 0x782b2585777d2b4eull || orig == 0x392e2956038a9e64ull || orig == 0x7d85669ac430f01aull)
        return 0;

    GfxShaderIo io[32];
    uint32_t used = gfx_shader_inputs(vs, io);
    if (!(used >> 9 & 1)) return 0;

    // Check if fragment shader reads light SSBO at c0+0x530
    int lights = 0;
    if (fs) {
        const GfxShaderRes *r;
        int n = gfx_shader_resources(fs, GFX_RES_SSBO, &r);
        for (int i = 0; i < n; i++) lights |= r[i].cbuf_slot == 0 && r[i].handle * 4 == 0x530;
    }
    if (lights) {
        warn_once(W_FAULT_SKIP, "light-quad draws with ten instance streams are skipped (#9; MKW_FIX_VS_FAULT=0 draws them)");
        return 1;
    }

    // Instanced light-quad draws in the NVN 2 reference title have count <= 6 (4-vertex quad or 6-vertex two-triangle quad)
    // and every active vertex attribute stream is per-instance (divisor != 0).
    // Legitimate 3D world meshes (skybox, cave walls, terrain) use per-vertex attributes (divisor == 0).
    if (count > 0 && count <= 6) {
        int all_instanced = 1, active_streams = 0;
        for (int l = 0; l < 16; l++) {
            if (!(used >> l & 1)) continue;
            va_st va = {0};
            memcpy(&va, &ex->pipe.vattr[l], sizeof va);
            if (!va.fmt || va.stream >= 16) continue;
            active_streams++;
            if (ex->pipe.vstream[va.stream].divisor == 0) {
                all_instanced = 0;
                break;
            }
        }
        if (all_instanced && active_streams >= 8) {
            warn_once(W_FAULT_SKIP, "instanced light-quad draw is skipped (#9/cutscene; MKW_FIX_VS_FAULT=0 draws them)");
            return 1;
        }
    }

    return 0;
}

// MKW_SKIP_VS / MKW_SKIP_FS (debug only, for bisecting the device loss of #75): comma-separated 16-digit hex NVN 2
// stage origins. A draw whose vertex (fragment) stage is listed in the matching variable is skipped, so a run that
// stops faulting names the culprit shader. Parsed once; unset costs one flag test per draw.
typedef struct { int n; uint64_t list[16]; } SkipList;
static int skip_listed(const VkExec *ex, int stage, SkipList *sl, const char *var, int warn, const char *msg) {
    if (sl->n < 0) {
        sl->n = 0;
        const char *e = cfg_get(var);
        while (e && *e && sl->n < 16) {
            char *end;
            uint64_t v = strtoull(e, &end, 16);
            if (end == e) break;
            sl->list[sl->n++] = v;
            e = *end == ',' ? end + 1 : end;
        }
    }
    if (!sl->n) return 0;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    if (!find_stage(ex->prog, stage, &ctl, &code, &len)) return 0;
    uint64_t o = nvn2_origin(code);
    for (int i = 0; i < sl->n; i++)
        if (sl->list[i] == o) return warn_once(warn, msg), 1;
    return 0;
}

static int skip_vs_listed(const VkExec *ex) {
    static SkipList sl = {-1};
    return skip_listed(ex, NVN_STAGE_VS, &sl, "TOTK_SKIP_VS", W_SKIP_VS, "MKW_SKIP_VS is active: draws of the listed NVN 2 vertex stages are skipped");
}

static int skip_fs_listed(const VkExec *ex) {
    static SkipList sl = {-1};
    return skip_listed(ex, NVN_STAGE_FS, &sl, "TOTK_SKIP_FS", W_SKIP_FS, "MKW_SKIP_FS is active: draws of the listed NVN 2 fragment stages are skipped");
}

static struct { int on, valid, has_fs, has_gs; MemoKey k; const GfxShader *vs, *fs, *gs; } SM = {-1};

// Numeric handles are relative to the active descriptor tables. Keep the per-handle memos only while those tables
// and their address mappings stay unchanged. A sampler table switch also changes combined image descriptors.
static void descriptor_state(VkExec *ex) {
    static struct { const VkExec *ex; dpool_ref tex, samp; uint64_t mem_gen; uint32_t slots_gen; } last;
    uint64_t mg = vk_mem_gen();
    uint32_t sg = vk_texture_slots_gen();
    if (last.ex != ex || last.tex.addr != ex->tex_pool.addr || last.tex.count != ex->tex_pool.count ||
        last.samp.addr != ex->samp_pool.addr || last.samp.count != ex->samp_pool.count || last.mem_gen != mg) {
        if (!++descriptor_serial) rt_fatal("vk_exec: descriptor context serial exhausted");
        SM.valid = 0;
        ex->fb_fs = NULL;
        last.ex = ex, last.tex = ex->tex_pool, last.samp = ex->samp_pool, last.mem_gen = mg;
    }
    if (last.slots_gen != sg) ex->fb_fs = NULL;
    last.slots_gen = sg;
}

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    k->geometry_input = geometry_input;
    if (geometry_input) {
        memcpy(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]);
        memcpy(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]);
    }
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(VkExec *ex, uint32_t gen, MemoKey *k) {
    descriptor_state(ex);
    if (!SM.valid) return 0;
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || k->geometry_input != geometry_input ||
        memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (geometry_input && (memcmp(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]) ||
                           memcmp(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]))) return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("TOTK_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

static struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("TOTK_PIPE_MEMO");
    return PM.on;
}

static int experimental_geometry_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_EXPERIMENTAL_GEOMETRY");
    return on;
}

// 1: an opted-in ordinary GS is required; 0: preserve the existing VS/FS path; -1: this opted-in draw is unsupported.
// Inspect raw NVN2 markers before any NVN1 control offsets. TOTK_EXPERIMENTAL_GEOMETRY admits no native shader; only
// MKW_NATIVE_GS does, for the passthrough subset checked first.
static int geometry_draw(VkExec *ex, uint32_t prim) {
    geometry_input = 0;
    if (native_gs_on() && ex->prog) {   // native failures keep VS/FS; ordinary stages retain their separate opt-in
        const uint8_t *ctl, *code, *fctl, *fcode;
        uint32_t len, flen, sph[5];
        if (!find_stage(ex->prog, NVN_STAGE_GS, &ctl, &code, &len)) return 0;
        if (nvn2_origin(code)) {
            if (len < 0x30 + 0x60 || !find_stage(ex->prog, NVN_STAGE_FS, &fctl, &fcode, &flen) ||
                vk_geometry_input(prim) != GFX_GEOMETRY_TRIANGLES) return 0;
            memcpy(sph, code + 0x30, sizeof sph);
            if (!(sph[0] & (1u << 24)) || sph[2] >> 24 != 1 || (sph[3] >> 24 & 15) != 7 || (sph[4] & 0xfff) != 1) {
                warn_once(W_GEOMETRY, "native GS: a stage outside the passthrough subset keeps the vertex and fragment path");
                return 0;
            }
            // Metal has no geometry stage (MoltenVK): skip before the pipeline and the lowered GS are built. The vertex
            // and fragment stages alone would draw black, so the draw is dropped and counted in draws skipped.
            if (!vk_geometry_supported()) {
                warn_once(W_GEOMETRY, "native GS: the device has no geometryShader: draws that need one are skipped");
                __atomic_fetch_add(&gfx_counters.draws_skipped, 1, __ATOMIC_RELAXED);
                return -1;
            }
            geometry_input = GFX_GEOMETRY_TRIANGLES;
            return 1;
        }
        // An ordinary GS still follows its own opt-in and validation below.
    }
    if (!experimental_geometry_on() || !ex->prog) return 0;
    int found = 0, portable = 0;
    uint32_t stages = 0;
    if (ex->prog->nstage > 6) goto unsupported;
    for (uint32_t i = 0; i < ex->prog->nstage; i++) {
        const uint8_t *ctl = ex->prog->sd[i].control;
        const uint8_t *code = (const uint8_t *)(uintptr_t)ex->prog->sd[i].data;
        if (!ctl || !code || gfx_shader_is_nvn2(ctl)) goto unsupported;
        uint32_t stage, len;
        memcpy(&stage, ctl + 0x714, 4);
        if (stage == NVN_STAGE_VS || stage == NVN_STAGE_FS) { stages |= 1u << stage; continue; }
        if (stage == NVN_STAGE_CS) continue;
        if (stage != NVN_STAGE_GS || found++) goto unsupported;   // no tessellation, duplicate or unknown stage
        stages |= 1u << stage;
        memcpy(&len, ctl + 0x704, 4);
        if (len < 0x30 + 0x60 || nvn2_origin(code) || c0_on()) goto unsupported;
        uint32_t sph;
        memcpy(&sph, code + 0x30, 4);
        if ((sph >> 10 & 15) != 4) goto unsupported;
        if (sph & (1u << 24)) {
            uint32_t skip[8];
            if (nvn_geometry_control_get(ctl, 0x880, code, len, skip) != 1 ||
                vk_geometry_input(prim) != GFX_GEOMETRY_TRIANGLES) goto unsupported;
            portable = 1;
        }
    }
    if (found) {
        // Replay currently replaces a complete program, not individual stages. Partial GS binds need a separate
        // merging contract; do not execute a GS which this bind did not request.
        if (!(stages & (1u << NVN_STAGE_VS)) || ((uint32_t)ex->prog_stages & stages) != stages) goto unsupported;
        // Portable metadata currently follows the verified whole-program SDK binding path.
        if (portable && ((uint32_t)ex->prog_stages != NVN_SHADER_STAGE_ALL_GRAPHICS_BITS ||
                         !(stages & (1u << NVN_STAGE_FS)))) goto unsupported;
        geometry_input = vk_geometry_input(prim);
        if (!geometry_input) goto unsupported;
    }
    return found;
unsupported:
    geometry_input = 0;
    warn_once(W_GEOMETRY, "experimental geometry draw rejected: ordinary GS, complete stage mask, supported primitive and FIX_C0=0 are required");
    return -1;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    if (ex->render_disabled) return vk_drawstats_skip(ex, ex->prog ? ex->prog->serial : 0), 0;
    // MKW_SKIP_PROG=<serial>[,<serial>...] (diagnostics): drop the draws of those shader programs, to see on screen what
    // a program the MKW_DRAWSTATS log names actually draws.
    static int nskip = -1;
    static uint64_t skip_prog[16];
    if (nskip < 0) {
        nskip = 0;
        const char *e = cfg_get("MKW_SKIP_PROG");
        for (char *q = (char *)e; q && *q && nskip < 16; q += *q == ',') skip_prog[nskip++] = strtoull(q, &q, 10);
        if (nskip) rt_log("vk_exec: MKW_SKIP_PROG: draws of %d shader program(s) are dropped", nskip);
    }
    for (int i = 0; i < nskip && ex->prog; i++)
        if (ex->prog->serial == skip_prog[i]) return 0;
    static int skip_all = -1;   // MKW_SKIP_DRAWS=1: drop every draw (bisecting GPU faults)
    if (skip_all < 0) skip_all = getenv("MKW_SKIP_DRAWS") != NULL;
    if (skip_all) return 0;
    int want_gs = geometry_draw(ex, (uint32_t)prim);
    if (want_gs < 0) return 0;
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return 0;
    if (!ex->prog) return warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int has_vs, has_fs, has_gs = 0, vs_wait = 0, fs_wait = 0, gs_wait = 0, memo = shader_memo_on();
    const GfxShader *vs, *fs, *gs = NULL;
    uint32_t gen = memo ? vk_texture_slots_gen() : 0;   // before shader_for, whose lookups may bump it
    if (memo && SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, gs = SM.gs, has_vs = 1, has_fs = SM.has_fs, has_gs = SM.has_gs;
        if (memo == 2) {
            int hv, hf, hg = 0, wv, wf, wg = 0;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv, &wv, 0), *f2 = shader_for(ex, NVN_STAGE_FS, &hf, &wf, 0);
            const GfxShader *g2 = want_gs ? shader_for(ex, NVN_STAGE_GS, &hg, &wg, 0) : NULL;
            if (v2 != vs || f2 != fs || g2 != gs || hf != has_fs || hg != has_gs || wv || wf || wg)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs, &vs_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        native_gs_vs = vs, native_gs_fs = fs;
        if (want_gs) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, &gs_wait, 0);
        if (memo && vs && !vs_wait && !fs_wait && !gs_wait && (!has_fs || fs) && (!want_gs || (has_gs && gs)))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.gs = gs, SM.has_fs = has_fs, SM.has_gs = has_gs, SM.valid = 1;
    }
    if (!has_vs) return warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    static int bis_lo = -1, bis_hi = 255;   // MKW_DRAW_BISECT=lo-hi (hex): only draws whose program byte is in range
    if (bis_lo < 0) {
        const char *e = getenv("MKW_DRAW_BISECT");
        bis_lo = 0;
        if (e) sscanf(e, "%x-%x", &bis_lo, &bis_hi);
    }
    if (vs && (bis_lo > 0 || bis_hi < 255)) {
        int b = gfx_shader_input_id(vs)[0] ^ (fs ? gfx_shader_input_id(fs)[0] : 0);
        if (b < bis_lo || b > bis_hi) return 0;
        if (bis_lo == bis_hi) {
            static int told;
            if (told++ < 4) {   // the cache file names (shader-cache/<key>.spv)
                char k[2][65] = {{0}};
                for (int i = 0; i < 32; i++) {
                    sprintf(k[0] + 2 * i, "%02x", gfx_shader_key(vs)[i]);
                    if (fs) sprintf(k[1] + 2 * i, "%02x", gfx_shader_key(fs)[i]);
                }
                const uint8_t *ctl, *vcode = NULL, *fcode = NULL;
                uint32_t len;
                find_stage(ex->prog, NVN_STAGE_VS, &ctl, &vcode, &len), find_stage(ex->prog, NVN_STAGE_FS, &ctl, &fcode, &len);
                rt_log("bisect: draw byte %#x vs %s (NVN 2 dump %016llx) fs %s (%016llx)", b, k[0],
                       (unsigned long long)nvn2_origin(vcode), k[1], (unsigned long long)nvn2_origin(fcode));
            }
        }
    }
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline:
    // one-shot draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the
    // session. A draw into per-frame targets is skipped until they are ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (vs_wait || fs_wait || gs_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, skip >= 1))) return perf_count(PERF_CNT_SHADER_SKIP), 0;   // skipped until ready
        perf_count(PERF_CNT_ONESHOT_WAIT);
        if (vs_wait) vs = shader_for(ex, NVN_STAGE_VS, &has_vs, NULL, 1);
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
        native_gs_vs = vs, native_gs_fs = fs;
        if (gs_wait) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, NULL, 1);
    }
    if (stage_diag_skip(ex, vs)) return 0;
    if (light_quad_fault_skip(ex, vs, fs, count) || skip_vs_listed(ex) || skip_fs_listed(ex)) return 0;
    if (!vs || (has_fs && !fs) || (want_gs && (!has_gs || !gs))) return 0;
    if (!geometry_feedback_safe(ex, vs, gs)) return 0;
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs, s.gs = gs;
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (TOTK_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static uint64_t wait_frame;
    static uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, TOTK_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) {   // a background worker is building it: the draw is skipped until it is ready
        vk_drawstats_pending(ex->prog ? ex->prog->serial : 0);
        return 0;
    }
    if (!p || !info) return warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic_changed(ex->cb, &s, B.st_valid ? &B.st : NULL);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    repeat_stats(ex, info);
    bind_resources(ex, info);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    bind_vertex_buffers(ex);
    if (vk_dump_armed) vk_dump_draw(ex, &s, p, info);
    vk_drawstats_draw(ex, (uint64_t)(uintptr_t)p, ex->prog ? ex->prog->serial : 0);
    vk_exec_oq_draw(ex);
    return 1;
}

// TOTK_TRACE=perf: the draw or dispatch just recorded, with the NVN 2 names of its stages, for the slow-part line
// (vk_sync.c vk_sync_note). kind: 0 draw, 1 indirect draw, 2 dispatch, 3 indirect dispatch.
static void note_work(const VkExec *ex, int kind, uint64_t work) {
    if (!vk_sync_noting() || !ex->prog) return;
    const uint8_t *ctl, *code;
    uint32_t len;
    int cs = kind >= 2;
    uint64_t a = find_stage(ex->prog, cs ? NVN_STAGE_CS : NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    uint64_t b = !cs && find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    vk_sync_note(kind, a, b, work);
}

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    ex->recorded++;
}

// The the NVN 2 reference title reset vertex program loses stores through pointers loaded from a list. Match both locally confirmed
// stages and the observed one-instance point-draw form. Other programs and parameter forms keep the shader path.
static int indirect_reset_draw(VkExec *ex, const p_draw *d) {
    // Both settings are fixed for the run (TOTK_FIX_INDIRECT_RESET needs a restart), so read them once: this
    // runs ahead of every draw.
    static int on = -1;
    if (on < 0) on = !rt_game_version && cfg_bool("TOTK_FIX_INDIRECT_RESET") && !getenv("MKW_SKIP_DRAWS");
    if (!on || !c0_on() || ex->render_disabled ||
        !ex->prog || ex->prog->nstage != 2 || d->prim != NVN_DRAW_PRIMITIVE_POINTS ||
        d->count <= 0 || d->first < 0 || d->base_instance || d->instances != 1) return 0;
    const uint8_t *ctl, *code;
    uint32_t length;
    if (!find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &length) || nvn2_origin(code) != 0xe78abde51f2d6bfaull ||
        !find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &length) || nvn2_origin(code) != 0x57b0d7b4959f03c0ull) return 0;
    const p_bind_buf *list = &ex->ssbo[NVN_STAGE_VS][0];
    // c0_table encodes the binding's size as u32 at +0x118; use the same extent, including its zero rejection.
    if (!vk_exec_reset_indirect(ex, list->addr, (uint32_t)list->size, d->first, d->count)) {
        warn_once(W_RESET, "the matched indirect reset has an unsupported list or record: using its shader");
        return 0;
    }
    return 1;
}



// MKW_DUMP_INST=<NVN 2 vertex origin, hex> | all (default 1cdcc0fd9dee5df9, the Mario Circuit tree that turned into
// phantom trunks): in the dump frame, every draw whose vertex stage has that origin (all: every draw of a pass at least
// 1000 wide that binds vertex storage slot 5) is traced. At the end of its render pass (inside the pass no copy is
// allowed; the draws do not change what they read) the GPU copies its indirect commands and draw count, and once per
// pass and buffer the storage buffer of slot 5 (c0+0x160, the 80-byte per-instance records), into readback buffers.
// On completion the log gets "dinst" lines: the dump pass and draw number (as in pick.txt, MKW_DUMP_PICK=1), the vertex
// origin, the commands and the records the drawn instances read (translation, row lengths, phase word).
enum { INST_CMDS = 8, INST_PENDING = 1024, INST_REC = 80 };
typedef struct {   // one slot-5 snapshot, shared by the traces of a pass that read the same range
    VkBuffer b;
    VmaAllocation a;
    int refs;
} InstSnap;
typedef struct {
    int kind;                 // 0 direct, 1 indexed, 2 indirect, 3 indirect count
    uint32_t base, count;     // direct draws: base instance, instances
    uint32_t ncmd, stride, max;
    VkMemSpan args, cnt, sb;  // spans to copy (buffer 0: none)
    uint64_t sb_size, origin, ind_addr;
    uint64_t vtx_addr, vtx_size, mat_addr, mat_size, idx_addr, idx_size;
    uint64_t u7_addr, u7_size, u9_addr, u9_size;
    uint32_t seq;
    int pass, draw;
    VkBuffer rb;
    VmaAllocation ra;
    uint64_t off_args, off_cnt;
    InstSnap *snap;
} InstTrace;
static InstTrace *inst_pending[INST_PENDING];
static int inst_npending;

enum { INST_ALL = 2 };
static uint64_t inst_origin(void) {
    static uint64_t o = 1;
    if (o == 1) {
        const char *e = cfg_get("MKW_DUMP_INST");
        o = e && !strcmp(e, "all") ? INST_ALL : e && *e ? strtoull(e, NULL, 16) : 0x1cdcc0fd9dee5df9ull;
    }
    return o;
}

static void inst_print(const InstTrace *t, const uint8_t *sb, uint32_t first, uint32_t n) {
    if (!sb) return;
    for (uint32_t k = 0; k < n && k < 6; k++) {
        uint64_t o = (uint64_t)(first + k) * INST_REC;
        if (o + INST_REC > t->sb_size) {
            rt_log("dinst #%u inst %u: past the end of the %#llx-byte buffer", t->seq, first + k,
                   (unsigned long long)t->sb_size);
            break;
        }
        const float *r = (const float *)(sb + o);
        float sx = sqrtf(r[0] * r[0] + r[4] * r[4] + r[8] * r[8]);
        float sy = sqrtf(r[1] * r[1] + r[5] * r[5] + r[9] * r[9]);
        float sz = sqrtf(r[2] * r[2] + r[6] * r[6] + r[10] * r[10]);
        rt_log("dinst #%u inst %u: pos %.1f %.1f %.1f scale %.3f %.3f %.3f phase %.3f", t->seq, first + k, r[3], r[7],
               r[11], sx, sy, sz, r[12]);
    }
}

static void inst_cb(void *arg) {   // completion thread (callbacks run one at a time)
    InstTrace *t = arg;
    static const char *const kn[] = {"draw", "draw-idx", "indirect", "indirect-count"};
    VkCtx *c = vk_ctx();
    void *map = NULL, *smap = NULL;
    if (!c || vmaMapMemory(c->vma, t->ra, &map) != VK_SUCCESS) goto out;
    vmaInvalidateAllocation(c->vma, t->ra, 0, VK_WHOLE_SIZE);
    if (t->snap && vmaMapMemory(c->vma, t->snap->a, &smap) == VK_SUCCESS)
        vmaInvalidateAllocation(c->vma, t->snap->a, 0, VK_WHOLE_SIZE);
    const uint8_t *m = map, *sb = smap;
    if (t->kind < 2) {
        rt_log("dinst #%u pass %04d draw %d vs %016llx %s base %u instances %u", t->seq, t->pass, t->draw,
               (unsigned long long)t->origin, kn[t->kind], t->base, t->count);
        inst_print(t, sb, t->base, t->count);
    } else {
        uint32_t n = t->kind == 3 ? *(const uint32_t *)(m + t->off_cnt) : 1;
        rt_log("dinst #%u pass %04d draw %d vs %016llx %s draws %u (max %u) args %#llx", t->seq, t->pass, t->draw,
               (unsigned long long)t->origin, kn[t->kind], n, t->max, (unsigned long long)t->ind_addr);
        for (uint32_t i = 0; i < n && i < t->ncmd; i++) {
            const uint32_t *a = (const uint32_t *)(m + t->off_args + (uint64_t)i * t->stride);
            rt_log("dinst #%u   cmd %u: indices %u instances %u first index %u vertex offset %d first instance %u", t->seq,
                   i, a[0], a[1], a[2], (int32_t)a[3], a[4]);
            inst_print(t, sb, a[4], a[1]);
        }
    }
    if (smap) vmaUnmapMemory(c->vma, t->snap->a);
    vmaUnmapMemory(c->vma, t->ra);
out:
    if (c) vmaDestroyBuffer(c->vma, t->rb, t->ra);
    if (t->snap && --t->snap->refs == 0) {
        if (c) vmaDestroyBuffer(c->vma, t->snap->b, t->snap->a);
        free(t->snap);
    }
    free(t);
}

int vk_dump_active(void);
void vk_dump_where(int *pass, int *draw);
static void inst_trace(VkExec *ex, int kind, uint32_t base, uint32_t count, uint64_t ind, uint64_t cnt, uint32_t max,
                       uint32_t stride, const p_draw_idx *direct_idx) {
    uint64_t want = inst_origin();
    if (!vk_dump_armed || !want || inst_npending == INST_PENDING || !ex->prog) return;
    static int auto_on = -1;
    static unsigned auto_n;
    if (auto_on < 0) {
        const char *e = cfg_get("MKW_INST_AUTOCAP");
        auto_on = e && *e == '1';
    }
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    if (!find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len)) return;
    uint64_t origin = nvn2_origin(code);
    const p_bind_buf *b = &ex->ssbo[NVN_STAGE_VS][5];
    if (want == INST_ALL ? ex->pass_w < 1000 || !b->addr : origin != want) return;
    int in_dump = vk_dump_active();
    if (!in_dump && (!auto_on || auto_n >= 16)) return;
    if (!in_dump) {
        if (!auto_n++) plat_file_mkdir_p(vk_dump_dir());
        if (auto_n == 16) rt_log("dinst: automatic capture reached 16 matching draws");
    }
    static uint32_t seq;
    InstTrace *t = calloc(1, sizeof *t);
    if (!t) return;
    t->kind = kind, t->base = base, t->count = count, t->max = max, t->stride = stride, t->seq = ++seq;
    t->origin = origin, t->ind_addr = ind;
    vk_dump_where(&t->pass, &t->draw);
    // A per-vertex transform index can turn one mesh into a screen-wide sheet when its stream or format is wrong.
    // Record the exact input declaration and ranges for a targeted F12 draw; ordinary frames never enter this path.
    rt_log("dinst #%u pass %04d draw %d vs %016llx vertex inputs:", t->seq, t->pass, t->draw,
           (unsigned long long)origin);
    for (int l = 0; l < 16; l++) {
        va_st a = {0};
        memcpy(&a, &ex->pipe.vattr[l], sizeof a);
        if (!a.fmt || a.stream >= 16) continue;
        const vs_st *s = &ex->pipe.vstream[a.stream];
        const NVNbufferRange *v = &ex->vbuf[a.stream];
        rt_log("dinst #%u   loc %d fmt %#x offset %u stream %u stride %u divisor %u buffer %#llx+%#llx",
               t->seq, l, (unsigned)a.fmt, (unsigned)a.offset, (unsigned)a.stream, s->stride, s->divisor,
               (unsigned long long)v->address, (unsigned long long)v->size);
    }
    for (int slot = 0; slot < 3; slot++) {
        const p_bind_buf *b = &ex->ssbo[NVN_STAGE_VS][slot];
        rt_log("dinst #%u   vs ssbo %d %#llx+%#llx", t->seq, slot,
               (unsigned long long)b->addr, (unsigned long long)b->size);
    }
    va_st index_attr = {0};
    memcpy(&index_attr, &ex->pipe.vattr[7], sizeof index_attr);
    if (index_attr.fmt && index_attr.stream < 16) {
        const NVNbufferRange *v = &ex->vbuf[index_attr.stream];
        t->vtx_addr = v->address, t->vtx_size = v->size;
    }
    // The skinning shader's c0+0x130 pointer selects vertex storage slot 2.
    const p_bind_buf *mat = &ex->ssbo[NVN_STAGE_VS][2];
    t->mat_addr = mat->addr, t->mat_size = mat->size;
    const p_bind_buf *u7 = &ex->ubo[NVN_STAGE_VS][7], *u9 = &ex->ubo[NVN_STAGE_VS][9];
    t->u7_addr = u7->addr, t->u7_size = u7->size;
    t->u9_addr = u9->addr, t->u9_size = u9->size;
    if (direct_idx && direct_idx->itype >= 0 && direct_idx->itype <= 2 && direct_idx->count > 0) {
        t->idx_addr = direct_idx->ibuf;
        t->idx_size = (uint64_t)direct_idx->count << direct_idx->itype;
        rt_log("dinst #%u   indices %d type %d base vertex %d buffer %#llx+%#llx restart %u",
               t->seq, direct_idx->count, direct_idx->itype, direct_idx->base_vertex,
               (unsigned long long)t->idx_addr, (unsigned long long)t->idx_size, (unsigned)ex->prim_restart.enable);
    }
    uint8_t bank[VK_C0_BYTES];
    vk_c0_fill(bank, NVN_STAGE_VS, ex->ssbo[NVN_STAGE_VS], ex->ubo[NVN_STAGE_VS], c0_dim, c0_binst);
    char path[480];
    snprintf(path, sizeof path, "%s/vi%u_p%04d_d%d_c0.bin", vk_dump_dir(), t->seq, t->pass, t->draw);
    FILE *out = fopen(path, "wb");
    if (out) fwrite(bank, 1, sizeof bank, out), fclose(out);
    t->ncmd = max < INST_CMDS ? max : INST_CMDS;
    if (ind && !vk_mem_resolve(ind, (uint64_t)stride * t->ncmd, &t->args)) t->args.buffer = VK_NULL_HANDLE;
    if (cnt && !vk_mem_resolve(cnt, 4, &t->cnt)) t->cnt.buffer = VK_NULL_HANDLE;
    t->sb_size = b->size;
    if (!b->addr || !b->size || !vk_mem_resolve(b->addr, b->size, &t->sb)) t->sb.buffer = VK_NULL_HANDLE;
    inst_pending[inst_npending++] = t;
}

static int inst_buffer(VkCtx *c, uint64_t size, VkBuffer *b, VmaAllocation *a) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    return c && vmaCreateBuffer(c->vma, &bi, &ai, b, a, NULL) == VK_SUCCESS;
}

// MKW_DUMP_CS=<NVN 2 compute origins, hex, comma separated> (default: MKW's object culling / LOD selection passes
// c287454e752c5e54, 0050df6cc6c1b5ce, 14293e2831001e91): in the dump frame, right before each dispatch of those
// programs, the c0 bank the dispatch binds and a GPU copy of every bound uniform and storage buffer (the state the
// shader starts from) go to the dump folder as cs<n>_<origin>_c0.bin, _u<slot>_<addr>.bin and _s<slot>_<addr>.bin;
// the log gets one "dcs" line per file. Used to recompute the LOD choice of each object offline.
enum { CS_SNAP_MAX = 24, CS_SNAP_BYTES = 64u << 20 };
typedef struct {
    char name[200];
    VkBuffer b;
    VmaAllocation a;
    uint64_t size;
} CsFile;

const char *vk_dump_dir(void);

static int cs_snap_wanted(uint64_t origin) {
    static uint64_t list[8];
    static int n = -1;
    if (n < 0) {
        n = 0;
        const char *e = cfg_get("MKW_DUMP_CS");
        if (!e || !*e) e = "c287454e752c5e54,0050df6cc6c1b5ce,14293e2831001e91,104bd69caf1f4c09,e8dae91b1570941b";
        while (*e && n < 8) {
            char *end;
            uint64_t v = strtoull(e, &end, 16);
            if (end == e) break;
            list[n++] = v;
            e = *end == ',' ? end + 1 : end;
        }
    }
    for (int i = 0; i < n; i++)
        if (list[i] == origin) return 1;
    return 0;
}

static void cs_file_cb(void *arg) {   // completion thread
    CsFile *f = arg;
    VkCtx *c = vk_ctx();
    void *map = NULL;
    if (c && vmaMapMemory(c->vma, f->a, &map) == VK_SUCCESS) {
        vmaInvalidateAllocation(c->vma, f->a, 0, VK_WHOLE_SIZE);
        FILE *o = fopen(f->name, "wb");
        if (o) fwrite(map, 1, (size_t)f->size, o), fclose(o);
        vmaUnmapMemory(c->vma, f->a);
    }
    if (c) vmaDestroyBuffer(c->vma, f->b, f->a);
    free(f);
}

static void cs_snap_copy(VkExec *ex, const char *base, char kind, int slot, const p_bind_buf *bb) {
    if (!bb->addr || !bb->size || bb->size > CS_SNAP_BYTES) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(bb->addr, bb->size, &sp)) return;
    CsFile *f = calloc(1, sizeof *f);
    if (!f) return;
    f->size = bb->size;
    snprintf(f->name, sizeof f->name, "%s_%c%d_%llx.bin", base, kind, slot, (unsigned long long)bb->addr);
    if (!inst_buffer(vk_ctx(), f->size, &f->b, &f->a)) {
        free(f);
        return;
    }
    VkBufferCopy cp = {sp.offset, 0, f->size};
    vkCmdCopyBuffer(ex->cb, sp.buffer, f->b, 1, &cp);
    ex->recorded++;
    rt_log("dcs %s %c%d addr %#llx size %#llx", base, kind, slot, (unsigned long long)bb->addr, (unsigned long long)bb->size);
    vk_completion_add(ex->fence, cs_file_cb, f);
}

// Before vkCmdDispatch / vkCmdDispatchIndirect, after bind_resources (outside any render pass); post: right after the
// dispatch (files cs<n>_<origin>_post_*), the same buffers as the dispatch left them.
static int cs_snap_n;
static void cs_snap(VkExec *ex, int post) {
    int done = cs_snap_n;
    if (!vk_dump_armed || (post ? done - 1 : done) >= CS_SNAP_MAX || (post && !done) || !ex->prog || !vk_dump_active()) return;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    if (!find_stage(ex->prog, NVN_STAGE_CS, &ctl, &code, &len)) return;
    uint64_t origin = nvn2_origin(code);
    if (!cs_snap_wanted(origin)) return;
    char base[160];
    snprintf(base, sizeof base, "%s/cs%d_%016llx%s", vk_dump_dir(), post ? done - 1 : done, (unsigned long long)origin,
             post ? "_post" : "");
    if (!post) cs_snap_n++;
    uint8_t bank[VK_C0_BYTES];
    vk_c0_fill(bank, NVN_STAGE_CS, ex->ssbo[NVN_STAGE_CS], ex->ubo[NVN_STAGE_CS], c0_dim, c0_binst);
    char path[200];
    snprintf(path, sizeof path, "%s_c0.bin", base);
    FILE *o = fopen(path, "wb");
    if (o) fwrite(bank, 1, sizeof bank, o), fclose(o);
    rt_log("dcs %s c0 (dispatch dims %u %u %u)", base, c0_dim[0], c0_dim[1], c0_dim[2]);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    for (int i = 0; i < 18; i++) cs_snap_copy(ex, base, 'u', i, &ex->ubo[NVN_STAGE_CS][i]);
    for (int i = 0; i < 16; i++) cs_snap_copy(ex, base, 's', i, &ex->ssbo[NVN_STAGE_CS][i]);
    vkCmdPipelineBarrier2(ex->cb, &di);   // the dispatch's own writes come after the copies
    ex->recorded += 2;
}

// vk_exec_pass_end, after vkCmdEndRendering: the copies of the traced draws of this pass.
void vk_exec_inst_pass_end(VkExec *ex) {
    VkCtx *c = vk_ctx();
    if (inst_npending) {
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                               .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded++;
    }
    for (int i = 0; i < inst_npending; i++) {
        InstTrace *t = inst_pending[i];
        uint64_t sz = t->args.buffer ? (uint64_t)t->stride * t->ncmd : 0;
        t->off_args = 0;
        sz = (sz + 15) & ~15ull, t->off_cnt = sz, sz += 16;
        if (!inst_buffer(c, sz, &t->rb, &t->ra)) {
            free(t);
            inst_pending[i] = NULL;
            continue;
        }
        if (t->sb.buffer) {   // share one snapshot per buffer range in this pass
            for (int j = 0; j < i && !t->snap; j++) {
                InstTrace *u = inst_pending[j];
                if (u && u->snap && u->sb.buffer == t->sb.buffer && u->sb.offset == t->sb.offset && u->sb_size == t->sb_size)
                    t->snap = u->snap, t->snap->refs++;
            }
            if (!t->snap) {
                InstSnap *s = calloc(1, sizeof *s);
                if (s && inst_buffer(c, t->sb_size, &s->b, &s->a)) {
                    VkBufferCopy cp = {t->sb.offset, 0, t->sb_size};
                    vkCmdCopyBuffer(ex->cb, t->sb.buffer, s->b, 1, &cp);
                    ex->recorded++;
                    s->refs = 1, t->snap = s;
                } else {
                    free(s);
                }
            }
        }
        if (t->args.buffer) {
            VkBufferCopy cp = {t->args.offset, t->off_args, (uint64_t)t->stride * t->ncmd};
            vkCmdCopyBuffer(ex->cb, t->args.buffer, t->rb, 1, &cp);
        }
        if (t->cnt.buffer) {
            VkBufferCopy cp = {t->cnt.offset, t->off_cnt, 4};
            vkCmdCopyBuffer(ex->cb, t->cnt.buffer, t->rb, 1, &cp);
        }
        char base[180];
        snprintf(base, sizeof base, "%s/vi%u_p%04d_d%d", vk_dump_dir(), t->seq, t->pass, t->draw);
        p_bind_buf vtx = {.addr = t->vtx_addr, .size = t->vtx_size < (1u << 20) ? t->vtx_size : (1u << 20)};
        p_bind_buf mat = {.addr = t->mat_addr, .size = t->mat_size < (1u << 20) ? t->mat_size : (1u << 20)};
        p_bind_buf idx = {.addr = t->idx_addr, .size = t->idx_size < (1u << 20) ? t->idx_size : (1u << 20)};
        p_bind_buf u7 = {.addr = t->u7_addr, .size = t->u7_size < (1u << 20) ? t->u7_size : (1u << 20)};
        p_bind_buf u9 = {.addr = t->u9_addr, .size = t->u9_size < (1u << 20) ? t->u9_size : (1u << 20)};
        cs_snap_copy(ex, base, 'v', 0, &vtx);
        cs_snap_copy(ex, base, 'm', 0, &mat);
        cs_snap_copy(ex, base, 'i', 0, &idx);
        cs_snap_copy(ex, base, 'u', 7, &u7);
        cs_snap_copy(ex, base, 'u', 9, &u9);
        ex->recorded += 2;
    }
    if (inst_npending) {
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded++;
        for (int i = 0; i < inst_npending; i++)
            if (inst_pending[i]) vk_completion_add(ex->fence, inst_cb, inst_pending[i]);
    }
    inst_npending = 0;
}

static void op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (indirect_reset_draw(ex, d)) return;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0)) return;
    vk_exec_frame.draws++;
    vk_drawstats_inst(d->instances > 0 ? (uint32_t)d->instances : 1);
    inst_trace(ex, 0, (uint32_t)d->base_instance, d->instances > 0 ? (uint32_t)d->instances : 1, 0, 0, 0, 0, NULL);
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
    } else {
        vkCmdDraw(ex->cb, n, inst, first, (uint32_t)d->base_instance);
    }
    ex->recorded++;
    note_work(ex, 0, (uint64_t)n * inst);
}

// Primitive restart of an indexed draw. NVN restarts wherever an index equals the SetPrimitiveRestart value, on lists
// as well as strips (docs/maxwell.md 5.2): in a list the restart drops the unfinished primitive and the next one starts
// after it. Vulkan restarts only at the all-ones value of the bound index type, and on lists only with
// primitiveTopologyListRestart, which Eglantine enables too (renderer_vulkan/vk_rasterizer.cpp,
// IsPrimitiveRestartSupported). Byte indices are widened to 16 bits, so their restart value becomes 0xffff
// (op_draw_idx). Without restart the all-ones entries are fetched as vertices: 65535 + base, or about 4 billion with
// 32-bit indices, out of the vertex buffer. In TotK's terrain and cliff strips the triangles around them dropped out;
// in a list every later triangle also loses its alignment, and the out-of-range vertices give long sheets across the
// screen (the NVN 2 reference title's map). TOTK_LIST_RESTART=0 keeps lists without restart, as before.
static int list_restart(void) {
    static int on = -1;
    if (on < 0) on = vk_ctx()->has_list_restart && cfg_bool("TOTK_LIST_RESTART");
    return on;
}

static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
        if (list_restart()) break;
        warn_once(W_LIST_RESTART, "primitive restart on a list draw is not replayed (no primitiveTopologyListRestart, or "
                  "TOTK_LIST_RESTART=0): its restart indices are fetched as vertices");
        return 0;
    default: return 0;   // patch lists: no restart
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

// MKW_SKIN_CHECK=1 (diagnostic, Crown City sheets): every draw of the skinned vertex stage 2367473af09b1016 checks what
// it will read. Bones per object = word 0 of vertex storage slot 2, object slot = c7[9].y (uniform slot 4, byte 0x94),
// matrix i at 16 + 48 * i. Logged ("dskin"): an object block past the end of the buffer, and matrices that are not
// finite or have a row longer than 1000, both in the submit-time copy (what the draw reads) and in live memory at
// record time and again when the submission completes. One summary line per 2000 checked draws.
typedef struct {
    uint64_t addr, slot, bones, n, seq;
    float snap_bad, live_bad;
} SkinCheck;

uint64_t vk_skin_buf_lo, vk_skin_buf_hi;   // the bone buffer the skinned pedestrians read (skin_check)

// MKW_SKIN_CHECK: data the replay writes into the bone buffer (inline UpdateUniformBuffer, buffer copies). Logs each
// write's offset and size and how many of its floats are broken (NaN, infinity, |x| > 1e6) as they arrive ("dskinu").
void vk_skin_write(const char *how, uint64_t dst, uint64_t size, const void *data) {
    if (!vk_skin_buf_hi || dst + size <= vk_skin_buf_lo || dst >= vk_skin_buf_hi) return;
    static uint64_t writes, bytes, bad_writes;
    static int logs;
    uint64_t bad = 0;
    if (data)
        for (uint64_t i = 0; i + 4 <= size; i += 4) {
            float f;
            memcpy(&f, (const uint8_t *)data + i, 4);
            bad += !(f == f) || f > 1e6f || f < -1e6f;
        }
    writes++, bytes += size, bad_writes += bad != 0;
    if (logs < 60 && (bad || logs < 10))
        logs++, rt_log("dskinu %s at +%#llx size %#llx: %llu broken floats%s", how, (unsigned long long)(dst - vk_skin_buf_lo),
                       (unsigned long long)size, (unsigned long long)bad, data ? "" : " (data not visible: GPU copy)");
    if (writes % 500 == 0)
        rt_log("dskinu summary: %llu writes, %llu bytes, %llu carried broken floats", (unsigned long long)writes,
               (unsigned long long)bytes, (unsigned long long)bad_writes);
}

void rt_skin_watch_arm(uint64_t addr, uint64_t size, uint64_t frame);   // ov_system.c MKW_SKIN_WATCH
void rt_skin_watch_poll(uint64_t frame);
void rt_skin_watch_seen(uint64_t addr, uint64_t size, uint32_t bones, uint32_t slot, int bad, uint64_t frame);
static int skin_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_SKIN_CHECK");
    return on;
}

// 0 good, else the first bad matrix + 1. m points at the object's first matrix (bones of them, 12 floats each).
static int skin_bad(const float *m, uint64_t bones) {
    for (uint64_t k = 0; k < bones; k++)
        for (int r = 0; r < 3; r++) {
            const float *row = m + 12 * k + 4 * r;
            float len = 0;
            for (int c = 0; c < 3; c++) {
                if (!isfinite(row[c])) return (int)k + 1;
                len += row[c] * row[c];
            }
            if (!isfinite(row[3]) || len > 1e6f || fabsf(row[3]) > 1e6f) return (int)k + 1;
        }
    return 0;
}

// MKW_SKIN_CHECK: does anything update the pedestrians? Once per frame per bone buffer (up to SKM_BUFS), every object's
// block (bones x 48 bytes) is hashed and compared with the last frame. Every 120 frames: "dskinm" with the objects
// whose block changed in that window, the ones that never changed since first seen, the broken ones, and how many of
// the broken ones never changed.
enum { SKM_BUFS = 6, SKM_SLOTS = 256 };
static struct {
    uint64_t addr, size, frame, first;
    uint32_t bones, nslot;
    uint64_t hash[SKM_SLOTS];
    uint8_t changed[SKM_SLOTS], ever[SKM_SLOTS], bad[SKM_SLOTS];
} skm[SKM_BUFS];

static uint64_t skm_hash(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

static void skin_motion(uint64_t addr, uint64_t size, uint64_t frame) {
    int k;
    for (k = 0; k < SKM_BUFS && skm[k].addr && skm[k].addr != addr; k++) {}
    if (k == SKM_BUFS) return;
    if (!skm[k].addr) skm[k].addr = addr, skm[k].size = size, skm[k].first = frame, skm[k].frame = 0;
    if (skm[k].frame == frame) return;
    int fresh = !skm[k].frame;
    skm[k].frame = frame;
    const uint8_t *m = (const uint8_t *)(uintptr_t)addr;
    uint32_t bones = *(const uint32_t *)m;
    if (!bones || bones > 64) return;
    uint64_t n = (size - 16) / 48 / bones;
    if (n > SKM_SLOTS) n = SKM_SLOTS;
    skm[k].bones = bones, skm[k].nslot = (uint32_t)n;
    for (uint64_t s = 0; s < n; s++) {
        const uint8_t *blk = m + 16 + 48 * bones * s;
        uint64_t h = skm_hash(blk, 48 * bones);
        if (!fresh && h != skm[k].hash[s]) skm[k].changed[s] = 1, skm[k].ever[s] = 1;
        skm[k].hash[s] = h;
        skm[k].bad[s] |= skin_bad((const float *)blk, bones) != 0;
    }
    if ((frame - skm[k].first) % 120 == 119) {
        uint32_t ch = 0, never = 0, bad = 0, badnever = 0, zero = 0;
        for (uint32_t s = 0; s < n; s++) {
            ch += skm[k].changed[s];
            never += !skm[k].ever[s];
            bad += skm[k].bad[s];
            badnever += skm[k].bad[s] && !skm[k].ever[s];
            const uint8_t *blk = m + 16 + 48 * bones * s;
            int z = 1;
            for (uint32_t i = 0; i < 48 * bones && z; i++) z = !blk[i];
            zero += z;
            skm[k].changed[s] = 0;
        }
        rt_log("dskinm buffer %#llx (%u objects x %u bones): %u changed in the last 120 frames, %u never changed, %u all-zero, "
               "%u broken (%u of them never changed)", (unsigned long long)addr, (uint32_t)n, bones, ch, never, zero, bad, badnever);
    }
}

static volatile int32_t skin_logs;
static void skin_done(void *arg) {   // completion thread
    SkinCheck *s = arg;
    const float *m = (const float *)(uintptr_t)(s->addr + 16 + 48 * s->slot * s->bones);
    int bad = skin_bad(m, s->bones);
    if (bad && __atomic_fetch_add(&skin_logs, 1, __ATOMIC_RELAXED) < 200)
        rt_log("dskin #%llu at completion: live matrix %d of slot %llu is bad", (unsigned long long)s->seq, bad - 1,
               (unsigned long long)s->slot);
    free(s);
}

static void skin_check(VkExec *ex) {
    if (!skin_on() || !ex->prog) return;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    if (!find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) || nvn2_origin(code) != UINT64_C(0x2367473af09b1016)) return;
    static uint64_t seq, checked, out_of_range, snap_bad, live_bad, differ, no_snap;
    seq++, checked++;
    rt_skin_watch_poll(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    const p_bind_buf *mb = &ex->ssbo[NVN_STAGE_VS][2], *ub = &ex->ubo[NVN_STAGE_VS][4];
    if (mb->addr && mb->size) vk_skin_buf_lo = mb->addr, vk_skin_buf_hi = mb->addr + mb->size;
    if (mb->addr && mb->size >= 64) skin_motion(mb->addr, mb->size, __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    if (vk_dump_armed && vk_dump_active()) {   // F12 frame: where this frame's bone buffer is (match against dbind lines)
        static uint64_t last;
        if (mb->addr != last) last = mb->addr, rt_log("dskin capture: bone buffer %#llx size %#llx", (unsigned long long)mb->addr,
                                                     (unsigned long long)mb->size);
    }
    if (mb->addr && mb->size >= 16 && ub->addr && ub->size >= 0x98) {
        const uint8_t *u = vk_usnap_bytes(ub->addr + 0x94, 4);
        uint32_t slot = u ? *(const uint32_t *)u : *(const uint32_t *)(uintptr_t)(ub->addr + 0x94);
        const uint8_t *sm = vk_usnap_bytes(mb->addr, mb->size);
        uint32_t bones = sm ? *(const uint32_t *)sm : *(const uint32_t *)(uintptr_t)mb->addr;
        uint64_t n = (mb->size - 16) / 48, need = ((uint64_t)slot + 1) * bones;
        if (!sm) no_snap++;
        if (!bones || need > n) {
            out_of_range++;
            if (__atomic_fetch_add(&skin_logs, 1, __ATOMIC_RELAXED) < 200)
                rt_log("dskin #%llu object block past the buffer: slot %u x %u bones = %llu of %llu matrices (%s)",
                       (unsigned long long)seq, slot, bones, (unsigned long long)need, (unsigned long long)n,
                       sm ? "submit copy" : "live");
        } else {
            uint64_t off = 16 + 48ull * slot * bones;
            const float *live = (const float *)(uintptr_t)(mb->addr + off);
            int lb = skin_bad(live, bones), sb = sm ? skin_bad((const float *)(sm + off), bones) : lb;
            if (sm && memcmp(sm + off, live, 48 * bones)) differ++;
            if (sb) snap_bad++;
            rt_skin_watch_seen(mb->addr, mb->size, bones, slot, sb || lb, __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
            if (lb) live_bad++;
            if ((sb || lb) && __atomic_fetch_add(&skin_logs, 1, __ATOMIC_RELAXED) < 200) {
                rt_log("dskin #%llu slot %u (%u bones): bad matrix %d in the %s, %d live at record time", (unsigned long long)seq,
                       slot, bones, (sb ? sb : lb) - 1, sm ? "submit copy" : "live buffer", lb - 1);
                static int vals;
                if (vals++ < 40) {   // the bad matrix and every bone of the object, as floats and raw bits of the first row
                    const float *base = sm ? (const float *)(sm + off) : live;
                    for (uint32_t k = 0; k < bones; k++) {
                        const float *mm = base + 12 * k;
                        uint32_t raw;
                        memcpy(&raw, mm, 4);
                        rt_log("dskin   bone %2u: %.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g (%08x)%s", k,
                               mm[0], mm[1], mm[2], mm[3], mm[4], mm[5], mm[6], mm[7], mm[8], mm[9], mm[10], mm[11], raw,
                               (int)k == (sb ? sb : lb) - 1 ? "  <- bad" : "");
                    }
                }
            }
            SkinCheck *s = calloc(1, sizeof *s);
            if (s) {
                *s = (SkinCheck){mb->addr, slot, bones, n, seq, 0, 0};
                vk_completion_add(ex->fence, skin_done, s);
            }
        }
    }
    if (checked % 2000 == 0)
        rt_log("dskin summary: %llu draws, %llu block past the buffer, %llu bad in the copy, %llu bad live, %llu copy != "
               "live at record, %llu without a copy", (unsigned long long)checked, (unsigned long long)out_of_range,
               (unsigned long long)snap_bad, (unsigned long long)live_bad, (unsigned long long)differ,
               (unsigned long long)no_snap);
}

static void op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype);
    int restart = restart_for(ex, d->prim, d->itype);
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, restart)) return;
    vk_exec_frame.draws++;
    vk_drawstats_inst(d->instances > 0 ? (uint32_t)d->instances : 1);
    inst_trace(ex, 1, (uint32_t)d->base_instance, d->instances > 0 ? (uint32_t)d->instances : 1, 0, 0, 0, 0, d);
    skin_check(ex);
    {   // MKW_SKIN_WATCH_TEST=1: self-test of the write-watch on a CPU-written vertex uniform block (frame 400 on)
        static int t = -1;
        if (t < 0) t = cfg_bool("MKW_SKIN_WATCH_TEST");
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (t && f >= 400) {
            const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][3];
            if (u->addr && u->size) rt_skin_watch_arm(u->addr, u->size, f);
            rt_skin_watch_poll(f);
        }
    }
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf);
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped");
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf);
    } else {
        VK_MEM_USE(sp);
        vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        ex->recorded++;
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    vkCmdDrawIndexed(ex->cb, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0, d->base_vertex,
                     (uint32_t)d->base_instance);
    ex->recorded++;
    note_work(ex, 0, (uint64_t)d->count * (uint64_t)(d->instances > 0 ? d->instances : 1));
}

// DrawElementsIndirect: the arguments are read by the GPU from the pool buffer (they are often written by an earlier
// dispatch), as nxnative nvn_mtl.m does. The base instance stays out of c0 (0x644): c0 cannot be patched by a transfer
// inside a pass; gl_BaseInstance of the Vulkan draw is the argument's. Byte indices need the count to be widened: skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    inst_trace(ex, 2, 0, 0, d->ind, 0, 1, sizeof(VkDrawIndexedIndirectCommand), NULL);
    ex->recorded += 2;
    note_work(ex, 1, 0);
    // An imported pool's GPU address is its host address (bind_range reads it the same way); device memory is not.
    if (!ip.alloc && vk_sync_noting()) vk_sync_note_args(d->ind);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

// MultiDrawElementsIndirectCount: GPU count buffer selects how many of the max_count indirect records to draw.
static void op_draw_idx_ind_count(VkExec *ex, const p_draw_idx_ind_count *d) {
    c0_binst = 0;
    if (d->max_count <= 0) return;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect-count draw with index type %d: skipped", d->itype);
    uint32_t stride = d->stride > 0 ? (uint32_t)d->stride : (uint32_t)sizeof(VkDrawIndexedIndirectCommand);
    if (stride < sizeof(VkDrawIndexedIndirectCommand) || (stride % 4))
        return warn_once(W_INDIRECT, "indirect-count stride %u is invalid: draws skipped", stride);
    VkMemSpan ib, ip, cp;
    uint64_t ind_bytes = (uint64_t)stride * (uint64_t)(uint32_t)d->max_count;
    if (!vk_mem_resolve(d->ind, ind_bytes, &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect-count args at %#llx are not backed or not aligned: draws skipped",
                         (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->count, sizeof(uint32_t), &cp) || cp.offset % 4)
        return warn_once(W_INDIRECT, "indirect-count counter at %#llx is not backed or not aligned: draws skipped",
                         (unsigned long long)d->count);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect-count draws skipped",
                         (unsigned long long)d->ibuf);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(cp);
    VK_MEM_USE(ib);
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset,
                         d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirectCount(ex->cb, ip.buffer, ip.offset, cp.buffer, cp.offset, (uint32_t)d->max_count, stride);
    vk_drawstats_count(ex, &cp);
    inst_trace(ex, 3, 0, 0, d->ind, d->count, (uint32_t)d->max_count, stride, NULL);
    ex->recorded += 2;
    note_work(ex, 1, 0);
    vk_mem_count(ip.cls, VK_MUSE_IND, ind_bytes);
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(cp);
    VK_MEM_BOUND(ib);
}

// ---- dispatch and uniform buffer updates

static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    static int skip = -1;   // MKW_SKIP_COMPUTE=1: drop every direct dispatch (bisecting GPU faults)
    if (skip < 0) skip = getenv("MKW_SKIP_COMPUTE") != NULL;
    if (skip) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);   // synchronous: a skipped dispatch changes results
    if (!cs) return;
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    c0_lay_set(ex, cs);
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    cs_snap(ex, 0);
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    cs_snap(ex, 1);
    ex->recorded++;
    note_work(ex, 2, (uint64_t)d->x * (uint64_t)d->y * (uint64_t)d->z);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, d);
}

// DispatchComputeIndirect: vkCmdDispatchIndirect from the pool buffer. NVN 1 shaders read the group counts at c0+0x200,
// so they are copied there on the GPU before dispatch. That NVN 1 offset can hold declared roots or resource tables
// in NVN 2; leave its packed bank unchanged rather than overwrite it with the legacy copy.
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);
    if (!cs) return;
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    c0_lay_set(ex, cs);
    B.c0_valid[2] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (B.c0_valid[2]) {   // copy GPU-written group counts into c0 (+0x4 on NVN 2, +0x200 on NVN 1)
        VkBufferCopy cp = {ip.offset, B.c0_off[2] + (c0_lay ? 0x4 : 0x200), sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[2], 1, &cp);
        ex->recorded++;
        vk_exec_barrier(ex);
        B.c0_valid[2] = 0;   // the range no longer holds B.c0[2]
    }
    cs_snap(ex, 0);
    vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset);
    cs_snap(ex, 1);
    ex->recorded++;
    note_work(ex, 3, 0);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, &(p_dispatch){0, 0, 0, d->caller});
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    if (skin_on()) vk_skin_write("inline update", u->dst, u->size, u->data);
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    vk_mem_gpu_wrote(&sp, u->size);
    vk_usnap_drop(u->dst, u->size);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size);
    upd_note(ex, u);
}

// ---- bind ops

static void geometry_context_buffer_log(unsigned record, const char *name, const p_bind_buf *binding) {
    VkMemInspect view = {0};
    int backed = binding->addr && binding->size && vk_mem_inspect(binding->addr, binding->size, &view);
    rt_log("geometry_buffer_context: record=%u range=%s snapshot=passive requested=%llu full_range_backed_now=%d "
           "pool_kind=%u pool_flags=%#x class_before_mirror_pick=%d offset=%llu available=%llu "
           "mirror_state=%d mirror_present=%d shadow_storage=%d buffer_present=%d contents=unread descriptor=unobserved",
           record, name, (unsigned long long)binding->size, backed, view.pool_kind, view.pool_flags, view.cls,
           (unsigned long long)view.offset, (unsigned long long)view.avail,
           view.mirror_state, view.mirror_present, view.shadow_storage, view.buffer_present);
}

// Opt-in draw-ingress metadata only. Replay is serialized; the cap is process-wide and later draws are unobserved.
static void geometry_context_log(const VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_DRAW && op != OP_DRAW_IDX && op != OP_DRAW_IDX_IND && op != OP_DRAW_IDX_IND_COUNT) return;
    static int on = -1, filtered;
    static uint64_t filter;
    static unsigned records;
    if (on < 0) {
        const char *e = getenv("MKW_GEOMETRY_CONTEXT"), *f = getenv("MKW_GEOMETRY_CONTEXT_ORIGIN");
        on = e && !strcmp(e, "1");
        if (on && f) {
            filtered = 1;
            if (f[0] == '0' && (f[1] == 'x' || f[1] == 'X')) f += 2;
            unsigned digits = 0;
            for (; *f; f++) {
                unsigned d = *f >= '0' && *f <= '9' ? (unsigned)(*f - '0') :
                             *f >= 'a' && *f <= 'f' ? (unsigned)(*f - 'a' + 10) :
                             *f >= 'A' && *f <= 'F' ? (unsigned)(*f - 'A' + 10) : 16;
                if (d == 16 || digits == 16) { on = 0; break; }
                filter = (filter << 4) | d;
                digits++;
            }
            if (!digits) on = 0;
            if (!on) rt_log("geometry_context: invalid origin filter; metadata logging disabled");
        }
    }
    if (!on || records == 12 || !ex || !payload) return;

    uint64_t origin[3] = {0};
    unsigned present = 0;
    int metadata = ex->prog && ex->prog->nstage <= 6;
    // Failed native lowering can leave a compact control: do not send it to the legacy-shaped stage getter.
    if (metadata)
        for (uint32_t i = 0; i < ex->prog->nstage; i++) {
            const uint8_t *ctl = ex->prog->sd[i].control;
            if (ctl && gfx_shader_is_nvn2(ctl)) { metadata = 0; break; }
        }
    if (metadata)
        for (int stage = 0; stage < 3; stage++) {
            const uint8_t *ctl, *code;
            uint32_t len;
            if (find_stage(ex->prog, stage, &ctl, &code, &len)) {
                present |= 1u << stage;
                origin[stage] = nvn2_origin(code);   // pointer-to-origin lookup, no shader bytes read
            }
        }
    if (filtered && !((present & 1) && origin[0] == filter) &&
        !((present & 2) && origin[1] == filter) && !((present & 4) && origin[2] == filter)) return;

    const char *kind;
    int prim;
    char draw[256], viewport[192], attachments[512], bindings[384];
    if (op == OP_DRAW) {
        const p_draw *d = payload;
        kind = "direct", prim = d->prim;
        snprintf(draw, sizeof draw, "first=%d count=%d instances_raw=%d base_instance=%d",
                 d->first, d->count, d->instances, d->base_instance);
    } else if (op == OP_DRAW_IDX) {
        const p_draw_idx *d = payload;
        kind = "indexed", prim = d->prim;
        snprintf(draw, sizeof draw, "count=%d instances_raw=%d base_vertex=%d base_instance=%d index_type=%d index_addr=%#llx",
                 d->count, d->instances, d->base_vertex, d->base_instance, d->itype, (unsigned long long)d->ibuf);
    } else if (op == OP_DRAW_IDX_IND_COUNT) {
        const p_draw_idx_ind_count *d = payload;
        kind = "indexed_indirect_count", prim = d->prim;
        snprintf(draw, sizeof draw,
                 "index_type=%d index_addr=%#llx indirect_addr=%#llx count_addr=%#llx max=%d stride=%d",
                 d->itype, (unsigned long long)d->ibuf, (unsigned long long)d->ind, (unsigned long long)d->count,
                 d->max_count, d->stride);
    } else {
        const p_draw_idx_ind *d = payload;
        kind = "indexed_indirect", prim = d->prim;
        snprintf(draw, sizeof draw, "index_type=%d index_addr=%#llx indirect_addr=%#llx count=unknown instances=unknown",
                 d->itype, (unsigned long long)d->ibuf, (unsigned long long)d->ind);
    }
    if (ex->has_vp)
        snprintf(viewport, sizeof viewport, "has_vp=1 viewport0_stored=[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g] depth_range=[%.9g,%.9g]",
                 ex->vp[0][0], ex->vp[0][1], ex->vp[0][2], ex->vp[0][3], ex->vp[0][4], ex->vp[0][5],
                 ex->depth_range[0], ex->depth_range[1]);
    else
        snprintf(viewport, sizeof viewport, "has_vp=0 viewport0_stored=unreported depth_range=[%.9g,%.9g]", ex->depth_range[0], ex->depth_range[1]);
    if (ex->rt_resolved) {
        size_t n = (size_t)snprintf(attachments, sizeof attachments, "attachments=resolved pass_noatt=%d pass_layers=%u layer_views=mip:base:count",
                                  ex->pass_noatt, ex->pass_layers);
        for (int i = 0; i < 8; i++)
            if (ex->color[i].image && n < sizeof attachments) {
                const VkExecTarget *t = &ex->color[i];
                n += (size_t)snprintf(attachments + n, sizeof attachments - n, " color%d=%u:%u:%u", i, t->level, t->base_layer, t->layers);
            }
        if (ex->depth.image && n < sizeof attachments)
            snprintf(attachments + n, sizeof attachments - n, " depth=%u:%u:%u", ex->depth.level, ex->depth.base_layer, ex->depth.layers);
    } else {
        snprintf(attachments, sizeof attachments, "attachments=unresolved pass_layers=unknown");
    }
    const p_bind_buf *s0 = &ex->ssbo[NVN_STAGE_GS][0], *s1 = &ex->ssbo[NVN_STAGE_GS][1], *u0 = &ex->ubo[NVN_STAGE_GS][0];
    snprintf(bindings, sizeof bindings, "gs_ranges=recorded_pod_unresolved gs_ssbo0=%#llx:%llu:%s gs_ssbo1=%#llx:%llu:%s gs_ubo0=%#llx:%llu:%s",
             (unsigned long long)s0->addr, (unsigned long long)s0->size, s0->addr && s0->size ? "range" : "empty_or_unset",
             (unsigned long long)s1->addr, (unsigned long long)s1->size, s1->addr && s1->size ? "range" : "empty_or_unset",
             (unsigned long long)u0->addr, (unsigned long long)u0->size, u0->addr && u0->size ? "range" : "empty_or_unset");
    rt_log("geometry_context: record=%u limit=12 kind=%s op=%u prim=%d %s fence=%llu program=%llu "
           "bound_stage_mask=%#x stage_metadata=%s present_stage_mask=%#x vs=%016llx gs=%016llx fs=%016llx "
           "render_disabled=%d render_cond_present=%d raster_discard=%u pass_active=%d %s %s %s "
           "clip_cull=untracked xfb=untracked guest_viewport_count=unknown guest_viewport_mask=unknown viewport0_update_provenance=untracked layer_routing=unknown",
           ++records, kind, op, prim, draw, (unsigned long long)ex->fence,
           ex->prog ? (unsigned long long)ex->prog->serial : 0ull, (unsigned)ex->prog_stages,
           metadata ? "available" : "unavailable", present,
           (unsigned long long)origin[0], (unsigned long long)origin[2], (unsigned long long)origin[1],
           ex->render_disabled, ex->has_render_cond, (unsigned)ex->pipe.raster_discard, ex->in_pass, viewport, attachments, bindings);
    geometry_context_buffer_log(records, "gs_ssbo0", s0);
    geometry_context_buffer_log(records, "gs_ssbo1", s1);
    geometry_context_buffer_log(records, "gs_ubo0", u0);
}

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= 6 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    memcpy(&dst[b->first], b->h, sizeof(uint64_t) * (size_t)b->n);
}

int vk_exec_find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    return find_stage(p, nvn_stage, ctl, code, len);
}

const GfxShader *vk_exec_shader_for(VkExec *ex, int nvn_stage, int *has) {
    return shader_for(ex, nvn_stage, has, NULL, 0);
}

const prog_t *vk_exec_fragment_prog(VkExec *ex) {
    if (ex && ex->stage_prog[NVN_SHADER_STAGE_FRAGMENT]) return ex->stage_prog[NVN_SHADER_STAGE_FRAGMENT];
    return g_last_fs_prog;
}

int vk_exec_mesh_prepare(VkExec *ex) {
    if (ex->render_disabled) return 0;
    cb_res(ex);
    dummies_init();
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    return sc.extent.width && sc.extent.height;
}

void vk_exec_mesh_bind(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info) {
    // Dynamic state that is static on a conventional pipeline may be set before the mesh bind;
    // after bind, only states declared dynamic on the mesh pipeline may be set (VUID-08608).
    if (!ex->in_pass) {
        static int pass_log;
        if (pass_log < 4) rt_log("vk_mesh: bind skipped: not in a render pass"), pass_log++;
        return;
    }
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vk_pipeline_set_dynamic_ex(ex->cb, s, 1);
    ex->recorded++;
    B.st = *s, B.st_valid = 0;   // next conventional draw must re-record topology/vertex input
    ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR | VK_EXEC_DIRTY_DEPTH_BIAS | VK_EXEC_DIRTY_BLEND_CONST |
                 VK_EXEC_DIRTY_STENCIL;
    record_fixed(ex);
    B.res_valid = 0;
    bind_resources(ex, info);
    B.res_valid = 0;   // conventional draws must not reuse mesh descriptor sets
    vk_exec_oq_draw(ex);
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    geometry_context_log(ex, op, payload);
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = payload;
        // Mesh CBs often BindProgram(mesh) without a prior FS in the same VkExec. Keep the last
        // real fragment program across command buffers (n239: stage_prog FS unset every mesh draw).
        ex->prog = b->prog, ex->prog_stages = b->stages;
        if (!ex->stage_prog[NVN_SHADER_STAGE_FRAGMENT] && g_last_fs_prog)
            ex->stage_prog[NVN_SHADER_STAGE_FRAGMENT] = g_last_fs_prog;
        // Stage-masked bind (NVN): update only stages named in the mask. A mesh ProgramSetShaders
        // often is [header-only SPH (type≤4, code len 0), task, mesh] with the SPH sitting in host
        // slot 1 — never promote that to the live fragment program (n238 foliage).
        if (b->prog) {
            static int bind_log;
            if (bind_log < 12)
                rt_log("vk_exec: BindProgram stages=%#x nstage=%u serial=%llu", (unsigned)b->stages,
                       b->prog->nstage, (unsigned long long)b->prog->serial),
                    bind_log++;
            for (int st = 0; st < 6; st++) {
                if (!(b->stages & (1 << st))) continue;
                if (st == NVN_SHADER_STAGE_FRAGMENT) {
                    const uint8_t *fctl = NULL, *fcode = NULL;
                    uint32_t flen = 0;
                    if (find_stage(b->prog, NVN_SHADER_STAGE_FRAGMENT, &fctl, &fcode, &flen)) {
                        if (control_is_header_only(fctl, flen)) {
                            static int skip_fs;
                            if (skip_fs++ < 4) {
                                uint32_t nvn_len = 0;
                                if (fctl) memcpy(&nvn_len, fctl + 0x38, 4);
                                rt_log("vk_exec: BindProgram keeps prior FS (host-1 header-only nvn_len=%u flen=%u)",
                                       nvn_len, flen);
                            }
                            continue;
                        }
                        g_last_fs_prog = b->prog;
                    } else {
                        // Mask names FRAGMENT but this program has no host-1 stage. Keep prior FS.
                        continue;
                    }
                }
                ex->stage_prog[st] = b->prog;
            }
            // Remember a real fragment whenever host slot 1 is not a header-only SPH.
            {
                const uint8_t *fctl = NULL, *fcode = NULL;
                uint32_t flen = 0;
                if (find_stage(b->prog, NVN_SHADER_STAGE_FRAGMENT, &fctl, &fcode, &flen) && fctl &&
                    !control_is_header_only(fctl, flen)) {
                    uint32_t nvn_len = 0;
                    memcpy(&nvn_len, fctl + 0x38, 4);
                    g_last_fs_prog = b->prog;
                    ex->stage_prog[NVN_SHADER_STAGE_FRAGMENT] = b->prog;
                    static int fs_remember;
                    if (fs_remember++ < 6)
                        rt_log("vk_exec: remembered FS program serial=%llu (nvn_len=%u flen=%u mask=%#x nvn2=%d)",
                               (unsigned long long)b->prog->serial, nvn_len, flen, (unsigned)b->stages,
                               gfx_shader_is_nvn2(fctl));
                }
            }
        }
        bind_gen++;
        break;
    }
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        case 7: memcpy(&ex->pipe.alpha_ref, b->bytes, sizeof ex->pipe.alpha_ref); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // MNG: the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        bind_gen++;
        break;
    }
    case OP_PUSH_CONST: {
        const p_push_const *p = payload;
        if (p->table < 0 || p->table >= NVN2_ROOT_TABLES || (!p->compute && p->table != 0)) {
            static int warned;
            if (!warned++) rt_log("vk: push constants into root table %d dropped", p->table);
            break;
        }
        int k = p->compute != 0, w0 = p->offset / 4, n = p->bytes / 4;
        memcpy(&ex->push[k][p->table][w0], p->w, 4 * (size_t)n);
        ex->push_set[k][p->table] |= (n >= 64 ? ~0ull : ((1ull << n) - 1)) << w0;
        bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        else {   // past the table: dropped (NVN 2 binds more storage buffers per stage than NVN 1's 16)
            static uint64_t seen[6];
            if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 64 && !(seen[b->stage] >> b->index & 1))
                seen[b->stage] |= 1ull << b->index, rt_log("vk: storage buffer %d of stage %d dropped", b->index, b->stage);
        }
        bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : NULL, 128, b);
        ex->tex_serial++;
        bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->samp[b->stage] : NULL, 128, b);
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->img[b->stage] : NULL, 32, b);
        ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        bind_gen++;
        break;
    }
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_DRAW: op_draw(ex, payload); break;
    case OP_DRAW_IDX: op_draw_idx(ex, payload); break;
    case OP_DISPATCH: op_dispatch(ex, payload); break;
    case OP_DRAW_IDX_IND: op_draw_idx_ind(ex, payload); break;
    case OP_DRAW_IDX_IND_COUNT: op_draw_idx_ind_count(ex, payload); break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); break;
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}

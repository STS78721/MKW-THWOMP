// Uniform blocks as of the submit call. NVN starts the GPU when nvnQueueSubmitCommands returns and the GPU reads the
// constants soon after; here the replay records for several milliseconds, and the GPU reads a pool's bytes when it
// runs the part that uses them (the VRAM mirror's sync copies run then, vk_memory.c "VRAM mirror"). Meanwhile the game,
// having handed the frame over, rewrites per-object uniform blocks on other threads with no fence: late draws read
// another object's matrix (phantom rocks, a flat far field; TotK's producer f_70008c6460, 14-24 ms after the
// submit). So at the start of a submission's recording every uniform range it binds is copied into an arena of its own,
// and the uniform bindings of the replay use the copy.
// - Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
//   minUniformBufferOffsetAlignment) and at most 64 KiB from the bound address, overlapping ones merged.
// - A range the GPU wrote in a submission that has not completed is left out: its bytes reach guest memory only with
//   that submission's write-back, the mirror holds them until then.
// - A GPU write within the submission (UpdateUniformBuffer, a storage buffer a shader writes, a buffer copy or fill,
//   an image-to-buffer copy: vk_usnap_drop) drops the copies it touches from then on, so later draws read the result.
// - The arena goes back to a free list once the submission's fence is done (completion thread).
// The design follows MacNativeGames' uniform-buffer snapshot (games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m usnap_take,
// usnap_drop; MIT, Copyright (c) 2026 jhaizhou); the code is ours.
// The copy is taken where vk_exec_record starts: at the submit call when the replay runs on the submitting thread, later
// with MKW_REPLAY_ASYNC. MKW_UBO_SNAPSHOT=0 binds live memory as before; =ab alternates (MKW_AB).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "platform/platform.h"
#include "vk_usnap_int.h"

#define ARENA_MIN (4u << 20)

int vk_usnap_host_wanted;   // a reader of the host copy besides MKW_SPEC: ubo_ptr, MKW_IND_CHECK_SETS

typedef struct Arena {
    struct Arena *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint64_t size;
} Arena;

// drop_at: the replay position (vk_exec_op_pos) of the first GPU write into the region in this submission, UINT64_MAX
// none: a binding at a later position takes guest memory. Positions, not a flag: MKW_REPLAY_PAR records a submission's
// chunks out of order, and a chunk that records first must still bind the copy where the serial replay would have.
typedef struct { uint64_t lo, hi, off, drop_at; } Reg;
typedef struct { uint64_t lo, hi, fence; } Wrote;

static struct {
    int mode;                   // rt_knob: 0 off, 1 on, 2 ab
    int on;                     // this submission has a snapshot
    uint64_t fence, gen;
    Arena *arena;
    uint8_t *host;              // the same bytes in cached host memory (vk_usnap_peek), with MKW_SPEC or MKW_CB_CENSUS
    uint64_t chost;
    int host_on;                // this submission filled host
    RngVec col;                 // collected ranges
    Reg *reg;                   // merged, sorted by lo, disjoint
    int nreg, creg;
    Wrote *wr;                  // GPU writes of submissions not known to be complete
    int nwr, cwr;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, drops, inflight, refused, off_subs, from_take;
    uint64_t walk_ns;           // of copy_ns, collecting, sorting and merging the ranges (the rest: the bytes)
} U;
static _Thread_local VkUsnapLog *ulog;   // MKW_REPLAY_PAR: this thread's snapshot hits and drops (vk_usnap_log)
static void log_add(int **a, int *n, int *cap, int v);

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;

static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("vk_usnap: out of memory");
    *cap = n;
    return q;
}

static Arena *arena_get(uint64_t need) {
    plat_mutex_lock(&free_lock);
    Arena **pp = &free_arenas, *a = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->size >= need) {
            a = *pp, *pp = a->next;
            break;
        }
    plat_mutex_unlock(&free_lock);
    if (a) return a;
    a = calloc(1, sizeof *a);
    if (!a) rt_fatal("vk_usnap: out of memory");
    a->size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a->size,
                             // storage too: a bank's word view (vk_pipe_layout.c flags 2) binds the same copy
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    // Device-local through the BAR when the device has it (the GPU reads constants on every draw), else host memory.
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &a->buf, &a->alloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk_usnap: arena of %llu bytes failed (%d): this submission binds live memory",
               (unsigned long long)a->size, (int)res);
        free(a);
        return NULL;
    }
    a->map = info.pMappedData;
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    return a;
}

static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
}

void usnap_collect(RngVec *v, const op_hdr *h, int depth) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < CALL_DEPTH) usnap_collect(v, nvn_op_first(((const p_jump *)a)->target), depth + 1);
        if (h->op == OP_BIND_UBO) {
            const p_bind_buf *b = a;
            if (b->addr && b->size) {
                if (v->n == v->cap) v->r = grow(v->r, &v->cap, sizeof *v->r);
                uint64_t n = b->size < MAX_RANGE ? b->size : MAX_RANGE;
                v->r[v->n++] = (Rng){b->addr & ~(uint64_t)(ALIGN - 1), b->addr + n};
            }
        }
        h = nvn_op_next(h);
    }
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
static int wr_by_lo(const void *a, const void *b) {
    uint64_t x = ((const Wrote *)a)->lo, y = ((const Wrote *)b)->lo;
    return x < y ? -1 : x > y;
}

// Sorts the collected ranges by lo. A town frame binds about 10,000 uniform ranges and qsort took half a millisecond
// of the copy's 0.96 ms on the game thread, so they go through an LSD radix sort on (lo - lowest) / ALIGN, three
// passes of 11 bits; qsort is left for a span that does not fit in 33 bits (2 TiB).
#define RADIX_BITS 11
void usnap_sort(RngVec *v) {
    static _Thread_local Rng *tmp;   // the replay thread and, with MKW_USNAP_AT_SUBMIT, the submitting one
    static _Thread_local int ctmp;
    int n = v->n;
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (v->r[i].lo < lo) lo = v->r[i].lo;
        if (v->r[i].lo > hi) hi = v->r[i].lo;
    }
    if ((hi - lo) / ALIGN >> (3 * RADIX_BITS)) {
        qsort(v->r, (size_t)n, sizeof *v->r, by_lo);
        return;
    }
    if (ctmp < n) {
        free(tmp);
        ctmp = v->cap;
        if (!(tmp = malloc((size_t)ctmp * sizeof *tmp))) rt_fatal("vk_usnap: out of memory");
    }
    Rng *src = v->r, *dst = tmp;
    for (int pass = 0; pass < 3; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != v->r) memcpy(v->r, src, (size_t)n * sizeof *src);   // three passes end in tmp
}

// Keeps the GPU writes of submissions that are not done, merged into disjoint ranges sorted by lo (the terrain rewrites
// one block before every draw: hundreds of equal entries), each with the newest fence among its parts.
static void wr_prune(void) {
    uint64_t done = vk_sync_fence_done();
    int n = 0;
    for (int i = 0; i < U.nwr; i++)
        if (U.wr[i].fence > done) U.wr[n++] = U.wr[i];
    qsort(U.wr, (size_t)n, sizeof *U.wr, wr_by_lo);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m && U.wr[i].lo <= U.wr[m - 1].hi) {
            if (U.wr[i].hi > U.wr[m - 1].hi) U.wr[m - 1].hi = U.wr[i].hi;
            if (U.wr[i].fence > U.wr[m - 1].fence) U.wr[m - 1].fence = U.wr[i].fence;
        } else {
            U.wr[m++] = U.wr[i];
        }
    }
    U.nwr = m;
}
static int wr_hits(uint64_t lo, uint64_t hi) {   // wr_prune ran: disjoint, sorted
    int a = 0, b = U.nwr;
    while (a < b) {
        int k = (a + b) / 2;
        if (U.wr[k].hi <= lo) a = k + 1;
        else b = k;
    }
    return a < U.nwr && U.wr[a].lo < hi;
}

static void summary(int force) {
    if (U.subs + U.off_subs < (force ? 1u : 3600u)) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms (%.3f of it collecting and merging the ranges), %.0f uniform bindings from the copy, %.1f dropped "
           "by GPU writes, %.1f left out (GPU writes in flight), %.1f refused (not an imported pool), %.1f regions "
           "from the submit's copy (MKW_USNAP_AT_SUBMIT)",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.walk_ns / s / 1e6, U.hits / s, U.drops / s, U.inflight / s, U.refused / s,
           U.from_take / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.walk_ns = U.hits = U.drops = U.inflight = U.refused =
        U.from_take = 0;
}

// MKW_USNAP_AT_SUBMIT=3: per frame, the ranges whose bytes the game rewrote between its submit and the recording, with
// the first few; logged at the next frame's first range.
static struct { uint64_t frame; uint32_t ranges, changed; uint64_t bytes; char first[200]; int k; } C;

static void cmp_note(uint64_t lo, uint64_t n, const uint8_t *took) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != C.frame) {
        if (C.changed)
            rt_log("vk_usnap: f%llu %u of %u uniform ranges (%llu bytes) changed between the submit and the recording:%s",
                   (unsigned long long)C.frame, C.changed, C.ranges, (unsigned long long)C.bytes, C.first);
        C.frame = f, C.ranges = C.changed = 0, C.bytes = 0, C.k = 0, C.first[0] = 0;
    }
    C.ranges++;
    uint8_t live[256];
    uint64_t diff = 0;
    for (uint64_t o = 0; o < n; o += sizeof live) {
        uint64_t m = n - o < sizeof live ? n - o : sizeof live;
        mem_copy_out(live, lo + o, m);
        for (uint64_t b = 0; b < m; b++) diff += live[b] != took[o + b];
    }
    if (!diff) return;
    C.changed++, C.bytes += diff;
    if (C.changed <= 4 && C.k < (int)sizeof C.first - 40)
        C.k += snprintf(C.first + C.k, sizeof C.first - (size_t)C.k, " %#llx+%#llx (%llu)", (unsigned long long)lo,
                        (unsigned long long)n, (unsigned long long)diff);
}

void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence) {
    if (!U.mode) U.mode = rt_knob("UBO_SNAPSHOT", "1") + 1;   // stored +1: 0 is "not read yet"
    U.on = 0, U.fence = fence, U.gen++;
    summary(0);
    UsnapTake *take = usnap_take_pop(handles, n);   // MKW_USNAP_AT_SUBMIT: the bytes as the submit found them
    if (!RT_KNOB_ON(U.mode - 1)) {
        usnap_take_free(take);
        U.off_subs++;
        return;
    }
    uint64_t t0 = kern_now_ns();
    U.col.n = 0;
    for (int i = 0; i < n; i++)
        if (handles[i]) usnap_collect(&U.col, nvn_op_first(handles[i]), 0);
    usnap_sort(&U.col);
    wr_prune();
    // merge, keeping only what lies wholly in one imported pool and no GPU write in flight touches
    U.nreg = 0;
    uint64_t total = 0;
    for (int i = 0; i < U.col.n;) {
        uint64_t lo = U.col.r[i].lo, hi = U.col.r[i].hi;
        for (i++; i < U.col.n && U.col.r[i].lo <= hi; i++)
            if (U.col.r[i].hi > hi) hi = U.col.r[i].hi;
        VkMemSpan sp;
        if (!vk_mem_resolve(lo, hi - lo, &sp) || sp.alloc || !sp.pool || sp.avail < hi - lo ||
            (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC)) {
            U.refused++;
            continue;
        }
        if (wr_hits(lo, hi)) {
            U.inflight++;
            continue;
        }
        if (U.nreg == U.creg) U.reg = grow(U.reg, &U.creg, sizeof *U.reg);
        U.reg[U.nreg++] = (Reg){lo, hi, total, UINT64_MAX};
        total += (hi - lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    }
    uint64_t t_walk = kern_now_ns() - t0;
    if (!U.nreg || !(U.arena = arena_get(total))) {
        usnap_take_free(take);
        return;
    }
    // The arena is write-combined (sequential write, device-local through the BAR): reading it back cost ~20 us a
    // draw for the constant words vk_spec.c looks at (01/11/2026, race at 95 ms a frame). The readers get a cached copy.
    if (ubo_ptr_snap < 0) {
        ubo_ptr_snap = rt_knob("UBO_PTR_SNAP", "1");
        rt_log("vk_usnap: storage addresses in uniform blocks from the snapshot %s (MKW_UBO_PTR_SNAP)",
               ubo_ptr_snap == 2 ? "in the B blocks" : ubo_ptr_snap ? "on" : "off");
    }
    U.host_on = vk_spec_mode() || vk_spec_census_on() || vk_usnap_host_wanted;
    if (U.host_on && U.chost < total) {
        free(U.host);
        U.chost = total * 2;
        if (!(U.host = malloc(U.chost))) rt_fatal("vk_usnap: out of memory");
    }
    for (int i = 0; i < U.nreg; i++) {
        const void *src = (const void *)(uintptr_t)U.reg[i].lo;
        uint64_t n = U.reg[i].hi - U.reg[i].lo;
        const uint8_t *took = take ? usnap_take_bytes(take, U.reg[i].lo, U.reg[i].hi) : NULL;
        if (took && usnap_take_compare()) {
            cmp_note(U.reg[i].lo, n, took);
            took = NULL;
        }
        if (took) src = took, U.from_take++;
        if (U.host_on) {
            if (took) memcpy(U.host + U.reg[i].off, took, n);
            else mem_copy_out(U.host + U.reg[i].off, U.reg[i].lo, n);
            src = U.host + U.reg[i].off;
        }
        if (src == (const void *)(uintptr_t)U.reg[i].lo) mem_copy_out(U.arena->map + U.reg[i].off, U.reg[i].lo, n);
        else memcpy(U.arena->map + U.reg[i].off, src, n);
    }
    usnap_take_free(take);
    vmaFlushAllocation(vk_ctx()->vma, U.arena->alloc, 0, total);   // a no-op on coherent memory
    U.on = 1;
    U.subs++, U.regions += (uint64_t)U.nreg, U.bytes += total, U.copy_ns += kern_now_ns() - t0, U.walk_ns += t_walk;
}

void vk_usnap_end(uint64_t fence) {
    if (U.arena) vk_completion_add(fence, arena_done, U.arena), U.arena = NULL;
    U.on = 0, U.gen++;
}

// First region whose end is above a.
static int first_above(uint64_t a) {
    int lo = 0, hi = U.nreg;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (U.reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    if (!U.on || !n) return 0;
    // Consecutive bindings mostly fall in the region found last (2.5 % of the replay went to the search, rt-z6).
    static _Thread_local int last;   // per thread: MKW_REPLAY_PAR
    int i = last < U.nreg && addr >= U.reg[last].lo && addr < U.reg[last].hi ? last : first_above(addr);
    last = i;
    if (i >= U.nreg) return 0;
    const Reg *r = &U.reg[i];
    if (__atomic_load_n(&r->drop_at, __ATOMIC_ACQUIRE) <= vk_exec_op_pos() || addr < r->lo || addr + n > r->hi) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    U.hits++;
    if (ulog) log_add(&ulog->hit, &ulog->nhit, &ulog->chit, i), log_add(&ulog->hpos, &ulog->nhpos, &ulog->chpos, (int)vk_exec_op_pos());
    return 1;
}

// As vk_usnap_find, uncounted: the cached host copy of addr when a binding of [addr, addr+n) would use the snapshot
// (NULL without MKW_SPEC or MKW_CB_CENSUS: no such copy).
const uint8_t *vk_usnap_peek(uint64_t addr, uint64_t n, uint64_t align) {
    if (!U.on || !n) return NULL;
    int i = first_above(addr);
    if (i >= U.nreg) return NULL;
    const Reg *r = &U.reg[i];
    if (__atomic_load_n(&r->drop_at, __ATOMIC_ACQUIRE) <= vk_exec_op_pos() || addr < r->lo || addr + n > r->hi) return NULL;
    uint64_t off = r->off + (addr - r->lo);
    return off % align || !U.host_on ? NULL : U.host + off;
}

// THWOMP camera jitter (device/vk_mv.c): the bytes the GPU will read for [addr, addr+n) in this submission's arena, to
// be rewritten before the part that binds them is submitted (vk_usnap_patched flushes them); NULL when the range was
// not copied (the binding reads guest memory) or a GPU write dropped the copy.
uint8_t *vk_usnap_gpu_bytes(uint64_t addr, uint64_t n) {
    if (!U.on || !n || !U.arena) return NULL;
    int i = first_above(addr);
    if (i >= U.nreg) return NULL;
    const Reg *r = &U.reg[i];
    if (__atomic_load_n(&r->drop_at, __ATOMIC_ACQUIRE) <= vk_exec_op_pos() || addr < r->lo || addr + n > r->hi) return NULL;
    return U.arena->map + r->off + (addr - r->lo);
}

void vk_usnap_patched(const uint8_t *p, uint64_t n) {
    if (U.arena && p >= U.arena->map) vmaFlushAllocation(vk_ctx()->vma, U.arena->alloc, (VkDeviceSize)(p - U.arena->map), n);
}

// Chunks of one submission record on several threads (MKW_REPLAY_PAR): drops go under drop_lock; vk_usnap_find reads
// a region's drop position without it.
static PlatMutex drop_lock = PLAT_MUTEX_INIT;
static _Thread_local uint64_t drops_here;

uint64_t vk_usnap_thread_drops(void) { return drops_here; }

static void log_add(int **a, int *n, int *cap, int v) {
    if (*n == *cap) *a = grow(*a, cap, sizeof **a);
    (*a)[(*n)++] = v;
}

void vk_usnap_log(VkUsnapLog *l) { ulog = l; }
int vk_usnap_regions(void) { return U.on ? U.nreg : 0; }
uint64_t vk_usnap_drop_at(int region) { return __atomic_load_n(&U.reg[region].drop_at, __ATOMIC_ACQUIRE); }

void vk_usnap_drop(uint64_t addr, uint64_t n) {
    if (!U.mode || !n) return;
    plat_mutex_lock(&drop_lock);
    if (U.nwr == U.cwr) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
    U.wr[U.nwr++] = (Wrote){addr, addr + n, U.fence};
    uint64_t pos = vk_exec_op_pos();
    if (U.on)
        for (int i = first_above(addr); i < U.nreg && U.reg[i].lo < addr + n; i++) {
            if (U.reg[i].drop_at == UINT64_MAX) U.drops++;
            if (pos < U.reg[i].drop_at) __atomic_store_n(&U.reg[i].drop_at, pos, __ATOMIC_RELEASE), U.gen++, drops_here++;
        }
    plat_mutex_unlock(&drop_lock);
}

uint64_t vk_usnap_gen(void) { return U.gen; }

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_summary(void) { summary(1); }

void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next) next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a);
    free_arenas = NULL;
    plat_mutex_unlock(&free_lock);
}

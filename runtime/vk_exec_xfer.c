// Transfers of the Vulkan op-stream replay: buffer and texture copies, texture
// clears, event signals, counter reports and conditional rendering. vk_exec.c calls vk_exec_op_xfer for these ops.
//
// Buffers are guest memory: vk_mem_resolve maps an address to the VkBuffer behind its pool, textures are the vk_texture.c images
// (linear texel layout, GENERAL layout; block-linear storage is only converted when a mirrored texture is uploaded). So a
// copy is one Vulkan transfer command and the pitch of a buffer side comes from the payload (OP_COPY_STRIDE or the
// tight row). Every copy ends the open pass and starts with a full barrier. Destinations: an image is marked
// GPU-written at record time; a CPU-visible buffer range is reported to vk_gpu_written once the GPU is done (write
// watch does not see GPU writes, textures stored there re-upload).
//
// Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
// vk_completion_add, in submit order, as gfx_null.c does: event WRITE / ADD, counter {0, timestamp} for every type but
// samples passed. A timestamp report puts 0 in the counter word: TOTK gates its clears on two such words being unequal.
//
// Samples passed (NVN_COUNTER_TYPE_SAMPLES_PASSED) is a running count on the queue: RESET_COUNTER zeroes it, every
// draw after it adds the samples that passed the depth and stencil tests, REPORT_COUNTER writes the running value.
// TOTK resets and reports it 3 to 8 times a frame in the field and its CPU reads the results back (f_70010c3298); a 0
// there culls what the query tested. The replay counts with Vulkan occlusion queries ("occlusion counts" below).
//
// Conditional rendering is a CPU-side skip, not VK_EXT_conditional_rendering: the extension compares one 32-bit word
// with zero, NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded
// from guest memory; a counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp,
// as unknown if it is samples passed. An unknown count renders: the report usually sits in the command buffer being
// recorded, so its value cannot be waited for, and the word in memory still holds an older query's count. Rendering
// costs at most the work the condition would have saved (MacNativeGames nvn_mtl.m decides the same); in the field
// TOTK conditions only on timestamp reports. The condition sets VkExec.render_disabled, which vk_exec.c honours for
// clears and vk_exec_draw.c for draws and dispatches. A condition that cannot be read (address outside every pool)
// renders.
// Copies and CLEAR_TEX are not conditional (as in MacNativeGames, nvn_mtl.m).
//
// The copy semantics (region and layer arithmetic, row and image strides, whole-level clears, copy / blit / resolve
// choice) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10080-10368, 10618-10642, 1700-1725
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou) and LAS Recomp runtime/nvn/vulkan/exec.c:3045-3235 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "cfg.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

enum { W_FORMAT, W_REGION, W_BUFFER, W_OVERLAP, W_COPY, W_CLEAR, W_QUERY, W_COUNT };

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

static uint32_t max1(uint32_t v) { return v ? v : 1; }

// TOTK_COMPACT_TRACE=1: log (first 600) copies that touch CompactLayers arrays or their big decoded staging.
static int ctrace_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("TOTK_COMPACT_TRACE");
        on = e && *e && *e != '0';
    }
    return on;
}
// Per key (a destination address): the first 48 lines each, 32 keys, so one busy array cannot starve the rest.
static void vctrace(uint64_t key, const char *fmt, va_list ap) {
    static PlatMutex m = PLAT_MUTEX_INIT;
    static uint64_t keys[512];
    static int cnt[512], nkeys;
    if (!ctrace_on()) return;
    plat_mutex_lock(&m);
    int i = 0;
    while (i < nkeys && keys[i] != key) i++;
    if (i == nkeys && nkeys < 512) keys[nkeys++] = key;
    int ok = i < nkeys && cnt[i]++ < (key >> 63 ? 3 : 48);
    plat_mutex_unlock(&m);
    if (!ok) return;
    char msg[512];
    vsnprintf(msg, sizeof msg, fmt, ap);
    rt_log("ctrace: %s", msg);
}
void vk_ctrace(uint64_t key, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vctrace(key, fmt, ap);
    va_end(ap);
}
#define ctrace(key, ...) vk_ctrace((uint64_t)(key), __VA_ARGS__)
static int big_img(const VkTexImage *i) { return i && (i->host_mip0 || (i->width >= 4096 && i->height >= 4096)); }

// ---- completion callbacks (completion thread)

typedef struct { uint64_t addr, size; } Range;

static void guest_written_cb(void *arg) {
    Range *r = arg;
    vk_gpu_written(r->addr, r->size);
    free(r);
}

// The GPU wrote [addr, addr+size) of CPU-visible memory: tell the texture cache once the fence is reached.
static void note_gpu_write(VkExec *ex, uint64_t addr, uint64_t size) {
    vk_usnap_drop(addr, size);
    if (nvn_gpu_only(addr)) return;
    Range *r = malloc(sizeof *r);
    if (!r) rt_fatal("vk: out of memory");
    *r = (Range){addr, size};
    vk_mem_gpu_pending(addr, size);
    vk_completion_add(ex->fence, guest_written_cb, r);
}

typedef struct { volatile uint32_t *val; uint32_t mode, value; } EventWrite;

static void event_cb(void *arg) {
    EventWrite *e = arg;
    if (e->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, e->value, __ATOMIC_RELEASE);
    else __atomic_store_n(e->val, e->value, __ATOMIC_RELEASE);
    free(e);
}

// NVNcounterData = {counter, timestamp}; timestamps are guest counter ticks (the base of GetSystemTick, rt_tick_hz).
static void write_counter_at(uint64_t addr, uint64_t counter, uint64_t ticks) {
    uint64_t rec[2] = {counter, ticks};
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
}
static void write_counter(uint64_t addr, uint64_t counter) {
    write_counter_at(addr, counter, rt_ns_to_ticks(kern_now_ns()));
}

// A timestamp report. On the console the GPU runs a submission as soon as the game submits it and its work takes a few
// milliseconds; here the replay comes first (up to a frame later with TOTK_REPLAY_ASYNC), and the completion is seen
// later still, so "now" at completion is mostly the replay's delay. the NVN 2 reference title paces itself on (end-of-frame timestamp -
// CPU frame start): with that delay in it, it measured every frame as longer than 1/60 s and dropped to a longer present
// interval. For a Switch 2 game the report keeps that existing entry-tick approximation, copied into its callback
// payload before work can complete. It does not measure GPU execution: TOP and END in one submit still share a tick.
// Concurrent callers may capture entry ticks in a different order from queueing; no global monotonicity is implied.
typedef struct { uint64_t addr, submitted_ticks; } CounterWrite;
static void counter_cb(void *arg) {
    CounterWrite *w = arg;
    if (!rt_game_version) write_counter_at(w->addr, 0, w->submitted_ticks);
    else write_counter(w->addr, 0);
    free(w);
}

// Reports queued by the replay that the GPU has not finished. The condition of RENDER_COND reads a timestamp one as 0
// and a samples-passed one as unknown.
#define PENDING 256
static struct { uint64_t addr, fence; int unknown; } pending[PENDING];
static unsigned pending_head;

// ---- occlusion counts (samples passed)
//
// After a SAMPLES_PASSED reset, the first draw of a pass begins an occlusion query (precise when the device has
// occlusionQueryPrecise); the query ends with the pass, or at the next report or reset of the counter, and the next
// draw begins another. A Vulkan query begun inside a render pass must end inside it. TOTK always resets, draws the
// tested proxies, reports, so counting stops at a report and restarts at the next reset: queries cover only the draws
// between the two, not every pass of the frame. Once the game reports twice without a reset in between, counting stays
// on for good (cumulative NVN semantics; that one report misses the draws since the previous one). The queries closed since the last
// report or reset form a segment, handed to the completion thread with that report or reset: in submit order it sums
// the segment's results into the running count (a reset drops the sum and zeroes the count) and writes the count of a
// report into guest memory. Slots come from one ring; the completion thread reads and host-resets them
// (hostQueryReset, Vulkan 1.2), so the replay records no reset commands. A full ring (no slot read back yet) leaves the
// draw uncounted and the segment's count at least 1, as does a query still unavailable after its fence. Without hostQueryReset the counts stay 0.
#define OQ_N 4096
#define LOG_ONCE(...) do { static volatile char once_; if (!__atomic_exchange_n(&once_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__); } while (0)
static VkQueryPool oq_pool;
static int oq_state;                     // 0 not set up, 1 counting available, -1 unavailable
static VkQueryControlFlags oq_flags;
static volatile uint8_t oq_busy[OQ_N];   // from begin until the completion thread read and reset the slot
static uint32_t oq_next;                 // replay: next ring slot
static int oq_counting;                  // replay: draws are counted (a reset since the last report, or oq_keep)
static int oq_reported;                  // replay: a samples-passed report since the last reset
static int oq_keep;                      // replay: the game reported twice without a reset: count every draw
static int oq_open = -1;                 // replay: the slot whose query is active in the open pass
static uint32_t *oq_seg;                 // replay: slots closed since the last report or reset
static uint32_t oq_nseg, oq_capseg;
static int oq_lost;                      // replay: the segment missed a draw (ring full)
static uint64_t oq_count;                // completion thread: samples since the last reset

typedef struct { uint64_t addr, frame; int reset, lost; uint32_t n; uint32_t q[]; } OqSegment;   // frame: recorded in

static int oq_ready(void) {
    if (oq_state) return oq_state > 0;
    VkCtx *v = vk_ctx();
    // vk_device.c enables every feature the device reports, so supported means enabled.
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(v->phys, &f2);
    oq_state = -1;
    if (!f12.hostQueryReset) {
        rt_log("vk: no hostQueryReset: samples-passed counters report 0");
        return 0;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                .queryCount = OQ_N};
    VkResult r = vkCreateQueryPool(v->device, &qi, NULL, &oq_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: occlusion query pool creation failed (%d): samples-passed counters report 0", (int)r);
        return 0;
    }
    vkResetQueryPool(v->device, oq_pool, 0, OQ_N);
    oq_flags = f2.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    oq_state = 1;
    return 1;
}

static void oq_close(VkExec *ex) {
    if (oq_open < 0) return;
    vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)oq_open);
    ex->recorded++;
    if (oq_nseg == oq_capseg) {
        oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
        oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg);
        if (!oq_seg) rt_fatal("vk: out of memory");
    }
    oq_seg[oq_nseg++] = (uint32_t)oq_open;
    oq_open = -1;
}

void vk_exec_oq_draw(VkExec *ex) {
    if (!oq_counting || oq_open >= 0) return;
    uint32_t q = oq_next % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        oq_lost = 1;
        return;
    }
    oq_busy[q] = 1;
    oq_next++;
    vkCmdBeginQuery(ex->cb, oq_pool, q, oq_flags);
    ex->recorded++;
    oq_open = (int)q;
}

void vk_exec_oq_pass_end(VkExec *ex) { oq_close(ex); }

void vk_exec_xfer_shutdown(void) {
    if (oq_state > 0) vkDestroyQueryPool(vk_ctx()->device, oq_pool, NULL);
    oq_pool = VK_NULL_HANDLE, oq_state = 0, oq_next = 0, oq_counting = oq_reported = oq_keep = 0, oq_open = -1;
    oq_nseg = 0, oq_lost = 0, oq_count = 0;
    memset((void *)oq_busy, 0, sizeof oq_busy);
}

static void oq_cb(void *arg) {
    OqSegment *s = arg;
    VkDevice dev = vk_ctx()->device;
    uint64_t sum = 0, v[64 * 2];
    for (uint32_t i = 0; i < s->n;) {
        uint32_t j = i + 1;
        while (j < s->n && j - i < 64 && s->q[j] == s->q[j - 1] + 1) j++;
        // The fence has passed, so a query that is still unavailable will not become available: waiting would block the
        // completion thread, and with it every later callback, for good. It counts as visible (one sample).
        VkResult r = vkGetQueryPoolResults(dev, oq_pool, s->q[i], j - i, sizeof v, v, 2 * sizeof v[0],
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY) {
            vk_device_fault_report(r);
            rt_fatal("vk: occlusion query results failed (%d)", (int)r);
        }
        vkResetQueryPool(dev, oq_pool, s->q[i], j - i);
        for (uint32_t k = i; k < j; k++) {
            if (v[2 * (k - i) + 1]) sum += v[2 * (k - i)];
            else {
                sum += 1;
                LOG_ONCE("vk: occlusion query slot %u not available after its fence, counted as visible", s->q[k]);
            }
            __atomic_store_n(&oq_busy[s->q[k]], 0, __ATOMIC_RELEASE);
        }
        i = j;
    }
    if (s->reset) oq_count = 0;
    else {
        oq_count += sum;
        if (s->lost && !oq_count) oq_count = 1;
        if (s->addr) write_counter(s->addr, oq_count);
        vk_drawstats_oq(s->frame, s->addr, oq_count, s->n);   // MKW_DRAWSTATS: when the answer lands, and what it is
    }
    free(s);
}

// Hands the segment to the completion thread with a report (addr; 0: no CPU-visible memory) or a reset.
static void oq_segment(VkExec *ex, uint64_t addr, int reset) {
    oq_close(ex);
    OqSegment *s = malloc(sizeof *s + sizeof *s->q * oq_nseg);
    if (!s) rt_fatal("vk: out of memory");
    s->addr = addr, s->reset = reset, s->lost = oq_lost, s->n = oq_nseg;
    s->frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    memcpy(s->q, oq_seg, sizeof *s->q * oq_nseg);
    oq_nseg = 0, oq_lost = 0;
    vk_completion_add(ex->fence, oq_cb, s);
}

static void reset_counter(VkExec *ex, const p_reset_counter *p) {
    if (p->type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) return;
    oq_segment(ex, 0, 1);
    oq_counting = 1, oq_reported = 0;
}

// A samples-passed report: the segment goes with it (addr 0: no CPU-visible memory) and counting stops until the next
// reset, unless reports came without resets before.
static void oq_report(VkExec *ex, uint64_t addr) {
    oq_segment(ex, addr, 0);
    if (oq_reported && !oq_keep) {
        oq_keep = 1;
        rt_log("vk: samples passed reported twice without a reset: counting every draw from now on");
    }
    oq_reported = 1;
    if (!oq_keep) oq_counting = 0;
}

static void signal_event(VkExec *ex, const p_signal_event *p) {
    EventWrite *e = malloc(sizeof *e);
    if (!e) rt_fatal("vk: out of memory");
    *e = (EventWrite){p->val, p->mode, p->value};
    vk_completion_add(ex->fence, event_cb, e);
}

static void report_counter(VkExec *ex, const p_report_counter *p) {
    int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && oq_ready();
    if (nvn_gpu_only(p->addr)) {   // CPU-invisible pool: no host memory behind it
        if (samples) oq_report(ex, 0);
        return;
    }
    VkMemSpan m;
    if (!vk_mem_resolve(p->addr, sizeof(NVNcounterData), &m))
        rt_fatal("vk: counter report at %#llx is in no memory pool", (unsigned long long)p->addr);
    vk_mem_count(m.cls, VK_MUSE_GPU_WRITE, sizeof(NVNcounterData));
    if (samples) oq_report(ex, p->addr);
    else {
        CounterWrite *a = malloc(sizeof *a);
        if (!a) rt_fatal("vk: out of memory");
        uint64_t ticks = ex->submitted_ticks;
        // MKW_GPU_TIME=zero (experiment, foliage plan): every timestamp of a frame reads that frame's first submission
        // tick, so the game measures its GPU frame time as 0. The default (each submission's entry tick) puts the
        // replay's CPU cost of the earlier submissions into the game's GPU timing, which a game that scales detail by
        // GPU time reads as an overloaded GPU.
        static int zero = -1;
        static uint64_t z_frame, z_ticks;
        if (zero < 0) {
            const char *e = cfg_get("MKW_GPU_TIME");
            zero = e && !strcmp(e, "zero");
            if (zero) rt_log("vk: MKW_GPU_TIME=zero: the timestamps of a frame all read its first submission's tick");
        }
        uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
        if (zero) {
            if (frame != z_frame) z_frame = frame, z_ticks = ticks;
            ticks = z_ticks;
        }
        vk_drawstats_ts(frame, p->addr, ticks);
        *a = (CounterWrite){p->addr, ticks};
        vk_completion_add(ex->fence, counter_cb, a);
    }
    pending[pending_head++ % PENDING] = (typeof(pending[0])){p->addr, ex->fence, samples};
}

// ---- conditional rendering

// 1: *v read, 0: not readable, 2: a samples-passed report not done yet (unknown)
static int counter_word(uint64_t addr, uint64_t *v) {
    uint64_t done = vk_sync_fence_done();
    for (unsigned k = 1; k <= PENDING; k++) {   // newest first: the last report to addr decides
        unsigned i = (pending_head - k) % PENDING;
        if (pending[i].addr == addr && pending[i].fence > done) return *v = 0, pending[i].unknown ? 2 : 1;
    }
    VkMemSpan m;
    if (nvn_gpu_only(addr) || !vk_mem_resolve(addr, sizeof *v, &m)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    vk_mem_count(m.cls, VK_MUSE_CPU_READ, sizeof *v);
    return 1;
}

static int cond_enabled(const p_render_cond *c) {
    uint64_t x, y;
    int rx = counter_word(c->addr, &x), ry = counter_word(c->addr + 16, &y);
    if (!rx || !ry) {
        warn_once(W_BUFFER, "conditional render counters at %#llx are not readable: rendering", (unsigned long long)c->addr);
        return 1;
    }
    if (rx == 2 || ry == 2) return vk_drawstats_cond(2), 1;   // a samples-passed count not done yet
    int on = c->mode == 0 ? x == y : x != y;
    vk_drawstats_cond(on);
    return on;
}

// ---- textures

typedef struct {
    const VkTexImage *img;
    VkImageAspectFlags aspect;
    TexVkFormat f;
    uint32_t level, layer0;
    int bpb, bw, bh;
    int32_t x, y, z;           // region origin
    uint32_t w, h, nz;         // region extent; nz: slices (3D) or layers
} Sub;

// Resolves tex, its view and a copy region to image coordinates. 0 (warned) when the copy cannot be replayed.
// allow_mode: 0 strict, 1 allow decoded/expanded (buffer->image), 2 allow promoted depth-stencil & dual aspect (clears/t2t).
static int sub_resolve(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level,
                       int allow_mode, Sub *s) {
    memset(s, 0, sizeof *s);
    if (!t) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f)) return 0;
    int promoted_ds = (allow_mode & 2) && (f.depth || f.stencil);
    if (!(allow_mode & 1) && !promoted_ds && (f.vk != f.ideal || f.expand_rgb || f.astc_bw)) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", t->d.format);
        return 0;
    }
    s->f = f;
    s->bpb = nvn_fmt_bpb(t->d.format, &s->bw, &s->bh);
    s->img = vk_texture_get(t);
    if (!s->img || !s->bpb) {
        warn_once(W_FORMAT, "copy of format %#x texture without an image: skipped", t->d.format);
        return 0;
    }
    const VkTexImage *img = s->img;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    else if (img->aspect == VK_IMAGE_ASPECT_DEPTH_BIT ||
             ((allow_mode & 2) && (img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))))
        s->aspect = img->aspect;
    else {
        warn_once(W_FORMAT, "copy of depth-stencil format %#x texture: skipped", t->d.format);
        return 0;
    }
    // CompactLayers: guest mip img->host_mip0 is host level 0 (vk_exec_draw.c sub_of / resolve_target).
    uint32_t guest_level = has_v && v->set_levels ? v->base_level : 0;
    if (guest_level < img->host_mip0) {
        warn_once(W_REGION, "copy of trimmed guest mip %u (host starts at %u) of a %ux%u texture: skipped",
                  guest_level, img->host_mip0, img->width, img->height);
        return 0;
    }
    s->level = guest_level - img->host_mip0;
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    s->x = r->xoffset, s->y = r->yoffset, s->z = r->zoffset;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    s->w = r->width > 0 ? (uint32_t)r->width : defaults_to_level ? (s->x < (int32_t)lw ? lw - (uint32_t)s->x : 0) : 0;
    s->h = r->height > 0 ? (uint32_t)r->height : defaults_to_level ? (s->y < (int32_t)lh ? lh - (uint32_t)s->y : 0) : 0;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t layers = is3d ? 1 : (uint32_t)s->z + s->nz, first = is3d ? 0 : s->layer0;
    if (s->level >= img->levels || s->x < 0 || s->y < 0 || s->z < 0 || !s->w || !s->h ||
        (uint64_t)s->x + s->w > lw || (uint64_t)s->y + s->h > lh || first + layers > img->layers ||
        (is3d && (uint64_t)s->z + s->nz > max1(img->depth >> s->level))) {
        warn_once(W_REGION, "copy region %d,%d,%d %ux%ux%u outside host level %u (guest %u) of a %ux%u texture: skipped",
                  s->x, s->y, s->z, s->w, s->h, s->nz, s->level, guest_level, img->width, img->height);
        return 0;
    }
    return 1;
}

static VkImageSubresourceLayers sub_layers(const Sub *s) {
    int is3d = s->img->type == VK_IMAGE_TYPE_3D;
    return (VkImageSubresourceLayers){s->aspect, s->level, is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
}

static VkOffset3D sub_offset(const Sub *s) {
    return (VkOffset3D){s->x, s->y, s->img->type == VK_IMAGE_TYPE_3D ? s->z : 0};
}

static VkExtent3D sub_extent(const Sub *s) {
    return (VkExtent3D){s->w, s->h, s->img->type == VK_IMAGE_TYPE_3D ? s->nz : 1};
}

// A compressed buffer copy covers whole blocks; Vulkan requires whole-block extents or reaching the mip edge (VUID 00207/00208).
static VkExtent3D sub_extent_blocks(const Sub *s) {
    VkExtent3D e = sub_extent(s);
    if (s->f.astc_bw || (s->bw == 1 && s->bh == 1)) return e;
    uint32_t bw = (uint32_t)s->bw, bh = (uint32_t)s->bh, rw = (s->w + bw - 1) / bw * bw, rh = (s->h + bh - 1) / bh * bh;
    uint32_t lw = s->img->width >> s->level, lh = s->img->height >> s->level;
    lw = lw ? lw : 1, lh = lh ? lh : 1;
    e.width = rw < lw - (uint32_t)s->x ? rw : lw - (uint32_t)s->x;
    e.height = rh < lh - (uint32_t)s->y ? rh : lh - (uint32_t)s->y;
    return e;
}

// Buffer side of a texture <-> buffer copy: bytes covered, pitches, Vulkan buffer span (a VRAM mirror when the pool
// has one).
static int buffer_side(const VkExec *ex, const Sub *s, uint64_t addr, uint64_t *size, VkBufferImageCopy *c,
                       VkMemSpan *span) {
    uint32_t wb = (s->w + (uint32_t)s->bw - 1) / (uint32_t)s->bw, hb = (s->h + (uint32_t)s->bh - 1) / (uint32_t)s->bh;
    uint64_t row = ex->copy_stride > 0 ? (uint64_t)ex->copy_stride : (uint64_t)wb * (uint64_t)s->bpb;
    if (row % (uint64_t)s->bpb || row < (uint64_t)wb * (uint64_t)s->bpb) {
        warn_once(W_REGION, "buffer copy row stride %llu does not fit %u blocks of %d bytes: skipped",
                  (unsigned long long)row, wb, s->bpb);
        return 0;
    }
    uint64_t img = row * hb;
    *size = img * (s->nz - 1) + row * (hb - 1) + (uint64_t)wb * (uint64_t)s->bpb;
    VkMemSpan m;
    if (!addr || !vk_mem_resolve(addr, *size, &m) || m.avail < *size) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)addr, (unsigned long long)*size);
        return 0;
    }
    VK_MEM_PICK(m);
    *span = m;
    *c = (VkBufferImageCopy){.bufferOffset = m.offset, .bufferRowLength = (uint32_t)(row / (uint64_t)s->bpb) * (uint32_t)s->bw,
                             .bufferImageHeight = hb * (uint32_t)s->bh, .imageSubresource = sub_layers(s),
                             .imageOffset = sub_offset(s), .imageExtent = sub_extent_blocks(s)};
    return 1;
}

static void copy_buffer_image(VkExec *ex, const p_copy_b2t *p, int to_image) {
    Sub s;
    int okr = sub_resolve(p->tex, &p->view, p->has_view, &p->r, 1, to_image, &s);
    {
        const VkTexImage *ti = p->tex ? vk_texture_get(p->tex) : NULL;
        if (ti && !big_img(ti) && p->tex->d.format >= 0x79 && p->tex->d.format <= 0x94)
            ctrace((uintptr_t)p->tex->mem | 1ull << 63, "b2t into ASTC tex %#llx %ux%u fmt %#x src %#llx rg %d,%d %dx%dx%d ok %d",
                   (unsigned long long)(uintptr_t)p->tex->mem, ti->width, ti->height, p->tex->d.format,
                   (unsigned long long)p->src, p->r.xoffset, p->r.yoffset, p->r.width, p->r.height, p->r.depth, okr);
        if (big_img(ti))
            ctrace((uintptr_t)p->tex->mem, "b2t%s fmt %#x tex %#llx %ux%u L%u x%u hm%u view(lv %u ly %u+%u) rg %d,%d,%d %dx%dx%d stride %d | ok %d s lv%u ly%u %d,%d %ux%ux%u",
                   to_image ? "" : "(readback)", p->tex->d.format, (unsigned long long)(uintptr_t)p->tex->mem, ti->width,
                   ti->height, ti->levels, ti->layers, ti->host_mip0, p->has_view ? p->view.base_level : 0,
                   p->has_view ? p->view.min_layer : 0, p->has_view ? p->view.nlayers : 0, p->r.xoffset, p->r.yoffset,
                   p->r.zoffset, p->r.width, p->r.height, p->r.depth, (int)ex->copy_stride, okr, s.level, s.layer0, s.x, s.y,
                   s.w, s.h, s.nz);
    }
    if (!okr) return;
    uint64_t size;
    VkBufferImageCopy c;
    VkMemSpan m;
    if (!buffer_side(ex, &s, p->src, &size, &c, &m)) return;
    vk_mem_count(m.cls, to_image ? VK_MUSE_COPY_SRC : VK_MUSE_GPU_WRITE, size);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    int written = 0;
    if (to_image && s.f.astc_bw && vk_astc_gpu()) {
        uint32_t wb = (s.w + (uint32_t)s.bw - 1) / (uint32_t)s.bw, hb = (s.h + (uint32_t)s.bh - 1) / (uint32_t)s.bh;
        uint64_t tight_row = (uint64_t)wb * (uint64_t)s.bpb;
        // vk_astc_record takes tight block rows (no row pitch); fall back to the skip warning on non-tight rows or layer strides.
        if ((ex->copy_stride > 0 && (uint64_t)ex->copy_stride != tight_row) ||
            (c.bufferRowLength && c.bufferRowLength != wb * (uint32_t)s.bw) ||
            (c.bufferImageHeight && c.bufferImageHeight != hb * (uint32_t)s.bh)) {
            warn_once(W_FORMAT, "copy of format %#x texture (non-tight stride): skipped", p->tex->d.format);
            return;
        }
        VkAstcRegion *ar = calloc(s.nz, sizeof *ar);
        if (!ar) return;
        uint64_t img_bytes = tight_row * (uint64_t)hb;
        int is3d = s.img->type == VK_IMAGE_TYPE_3D;
        uint32_t dlw = max1(s.img->width >> s.level), dlh = max1(s.img->height >> s.level);
        for (uint32_t z = 0; z < s.nz; z++) {
            ar[z] = (VkAstcRegion){
                .src = m.offset + z * img_bytes,
                .w = s.w,
                .h = s.h,
                .slices = 1,
                .level = s.level,
                .layer = is3d ? 0 : s.layer0 + (uint32_t)s.z + z,
                .lw = dlw,
                .lh = dlh,
                .offset = (VkOffset3D){s.x, s.y, is3d ? s.z + (int32_t)z : 0}
            };
        }
        int img_bc = s.f.astc_bc && s.img->format != VK_FORMAT_R8G8B8A8_UNORM &&
                     s.img->format != VK_FORMAT_R8G8B8A8_SRGB;
        written = vk_astc_record(ex->cb, m.buffer, size, ar, (int)s.nz, s.f.astc_bw, s.f.astc_bh, s.f.astc_srgb, img_bc,
                                 s.img->image, s.aspect, VK_NULL_HANDLE);
        free(ar);
        if (!written) {
            warn_once(W_FORMAT, "ASTC compute copy of format %#x texture failed to record: skipped", p->tex->d.format);
            return;
        }
        vk_exec_draw_new_cb();
        ex->dirty = VK_EXEC_DIRTY_ALL;
        vk_exec_barrier(ex);
    } else if (to_image && (s.f.vk != s.f.ideal || s.f.expand_rgb || s.f.astc_bw)) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", p->tex->d.format);
        return;
    } else if (to_image) {
        vkCmdCopyBufferToImage(ex->cb, m.buffer, s.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        written = 1;
    } else {
        if (s.f.vk != s.f.ideal || s.f.expand_rgb || s.f.astc_bw) {
            warn_once(W_FORMAT, "readback of format %#x texture (decoded or expanded on upload): skipped", p->tex->d.format);
            return;
        }
        vkCmdCopyImageToBuffer(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, m.buffer, 1, &c);
        written = 1;
    }
    ex->recorded++;
    if (to_image) {
        if (written) {
            vk_texture_gpu_written(p->tex);
            if (s.img->sparse)
                vk_sparse_written(s.img->image, s.level, s.layer0 + (uint32_t)s.z, s.nz, s.x, s.y, s.w, s.h);
        }
    } else {
        vk_mem_gpu_wrote(&m, size), note_gpu_write(ex, p->src, size);
    }
}

// Decoded ASTC copies operate on host RGBA8 texels, not on the guest's compressed blocks. Keep their view and
// block-edge checks separate from the native-format path, which also supports blits, resolves and other targets.
// fold_full: allow a whole-level guest write into a trimmed CompactLayers host (scale into host level 0).
// Off by default — void placeholder fills would wipe streamed tiles. On when the source is a full-resolution
// decoded staging image (host_mip0 == 0) copying into a CompactLayers dest.
static int decoded_astc_rgba8(VkFormat fmt) {
    return fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_SRGB;
}

static int decoded_astc_sub(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r,
                            const TexVkFormat *f, const VkTexImage *img, Sub *s, int fold_full) {
    // Host may be RGBA8 (decoded / CompactLayers) while f->vk still names BC3 from ASTC_BC.
    // Accept any RGBA8 host for decoded copies; fold_full also accepts the CompactLayers dest
    // when the TexVkFormat still advertises BC.
    int fmt_ok = img && (img->format == f->vk || decoded_astc_rgba8(img->format) ||
                         (fold_full && decoded_astc_rgba8(img->format)));
    if (!img || (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != NVN_TEXTURE_TARGET_2D_ARRAY) ||
        img->type != VK_IMAGE_TYPE_2D || img->samples != 1 || img->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        !fmt_ok || !img->image || img->bound ||
        (img->usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) !=
            (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    if (has_v && (v->set_format || v->set_target || v->set_swizzle || v->set_dsmode)) return 0;
    memset(s, 0, sizeof *s);
    s->img = img, s->aspect = VK_IMAGE_ASPECT_COLOR_BIT, s->f = *f;
    // CompactLayers: remap guest mip → host level (same rule as sub_resolve / sub_of).
    // Whole-level writes into trimmed mips are usually void placeholder fills; folding them into
    // host level 0 overwrites streamed tiles (pop-in / magenta voids). Partial tile updates that
    // still target a trimmed guest mip are folded with coords scaled by 2^(host_mip0 - guest_level).
    // fold_full: staging→CompactLayers T2T (8192 host_mip0=0 → 1024 host_mip0=3) must fold or terrain stays empty.
    uint32_t guest_level = has_v && v->set_levels ? v->base_level : 0;
    uint32_t shift = 0;
    // TOTK_COMPACT_FOLD=0: like Eglantine PR #570, a trimmed guest mip has no host storage — drop writes to it
    // instead of NEAREST-folding them into host level 0 (which overwrites the real guest-mip-host_mip0 data
    // there and showed as black near ground / road).
    static int compact_fold = -1;
    if (compact_fold < 0) {
        const char *e = cfg_get("TOTK_COMPACT_FOLD");
        compact_fold = !(e && *e == '0');
        rt_log("vk: CompactLayers folds of trimmed guest mips into host level 0: %s", compact_fold ? "on" : "off");
    }
    if (guest_level < img->host_mip0 && !compact_fold) return 0;
    if (guest_level < img->host_mip0) {
        uint32_t gw = max1(t->d.width >> guest_level), gh = max1(t->d.height >> guest_level);
        int full = r->xoffset <= 0 && r->yoffset <= 0 &&
                   (!r->width || (uint32_t)r->width >= gw) && (!r->height || (uint32_t)r->height >= gh);
        if (full && !fold_full) return 0;
        shift = img->host_mip0 - guest_level;
        s->level = 0;
    } else {
        s->level = guest_level - img->host_mip0;
    }
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    uint32_t levels = has_v && v->set_levels ? v->nlevels : img->levels;
    uint32_t layers = has_v && v->set_layers ? v->nlayers : img->layers;
    // CompactLayers fold: the guest view still names the full guest mip count (e.g. 14). Host only has
    // the trimmed pyramid (e.g. 9). Clamping lets staging→host copies through; rejecting them left
    // dst_ok=0 with 8192x8192 vs 0x0 and empty terrain.
    if (levels > img->levels - s->level) levels = img->levels - s->level;
    // Validate before shifting, and bound the complete view as well as the one selected mip/region.
    if (s->level >= img->levels || s->level >= 32 || !levels || levels > img->levels - s->level ||
        s->layer0 >= img->layers || !layers || layers > img->layers - s->layer0 ||
        r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 || r->width < 0 || r->height < 0 || r->depth < 0) return 0;
    uint32_t w = max1(img->width >> s->level), h = max1(img->height >> s->level);
    s->x = (int32_t)((uint32_t)r->xoffset >> shift);
    s->y = (int32_t)((uint32_t)r->yoffset >> shift);
    s->z = r->zoffset;
    if ((uint32_t)s->x >= w || (uint32_t)s->y >= h) return 0;
    s->w = r->width ? (uint32_t)r->width >> shift : w - (uint32_t)s->x;
    s->h = r->height ? (uint32_t)r->height >> shift : h - (uint32_t)s->y;
    s->nz = r->depth ? (uint32_t)r->depth : 1;
    if (!s->w) s->w = 1;
    if (!s->h) s->h = 1;
    if ((uint64_t)s->x + s->w > w || (uint64_t)s->y + s->h > h || (uint64_t)s->z + s->nz > layers) return 0;
    // Expand to whole blocks (BC3 4×4 / ASTC bw×bh). Streamed tile updates are often not block-aligned
    // (MKW logged 510×510 into a CompactLayers host); incomplete blocks stay legal only at the mip edge.
    // RGBA8 hosts (decoded / CompactLayers) are texel-addressed — do not apply BC/ASTC block rounding
    // just because TexVkFormat still advertises ASTC_BC from the guest format.
    uint32_t bbw = 0, bbh = 0;
    if (!decoded_astc_rgba8(img->format)) {
        bbw = f->astc_bc ? 4u : (uint32_t)f->astc_bw;
        bbh = f->astc_bc ? 4u : (uint32_t)f->astc_bh;
    }
    if (bbw && bbh) {
        uint32_t x0 = (uint32_t)s->x / bbw * bbw, y0 = (uint32_t)s->y / bbh * bbh;
        uint32_t x1 = ((uint32_t)s->x + s->w + bbw - 1) / bbw * bbw;
        uint32_t y1 = ((uint32_t)s->y + s->h + bbh - 1) / bbh * bbh;
        if (x1 > w) x1 = w;
        if (y1 > h) y1 = h;
        s->x = (int32_t)x0;
        s->y = (int32_t)y0;
        s->w = x1 - x0;
        s->h = y1 - y0;
        if (!s->w || !s->h) return 0;
        if (s->x % bbw || s->y % bbh || (s->w % bbw && (uint32_t)s->x + s->w != w) ||
            (s->h % bbh && (uint32_t)s->y + s->h != h)) return 0;
    }
    return 1;
}

// Returns one for decoded-ASTC sources, including deliberate unsupported skips. Native ASTC and all other source
// formats continue through copy_t2t unchanged. In create_tex, !bound means a separate vmaCreateImage allocation;
// different guest ranges alone would not prove nonaliasing for images bound into a pool.
// Whether f can be a vkCmdBlitImage source (BC formats on desktop GPUs: yes), cached per format.
static int blit_src_ok(VkFormat f) {
    static int8_t known[256];
    if ((unsigned)f >= 256) return 0;
    if (!known[f]) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &fp);
        known[f] = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? 1 : -1;
    }
    return known[f] > 0;
}

static int copy_decoded_astc(VkExec *ex, const p_copy_t2t *p) {
    if (!p->src || p->src->d.format < 0x79 || p->src->d.format > 0x94) return 0;
    TexVkFormat f;
    if (!tex_vk_format(p->src->d.format, &f) || !f.astc_bw) return 0;
    if (!p->dst || p->src->d.format != p->dst->d.format || p->flags ||
        (!f.astc_bc && f.vk != VK_FORMAT_R8G8B8A8_UNORM && f.vk != VK_FORMAT_R8G8B8A8_SRGB)) {
        warn_once(W_COPY, "decoded ASTC copy requires matching formats and no copy flags: skipped");
        return 1;
    }
    uint64_t sa = (uintptr_t)p->src->mem, da = (uintptr_t)p->dst->mem;
    if (!sa || !da || !p->src->L.size || !p->dst->L.size ||
        (sa <= da ? da - sa < p->src->L.size : sa - da < p->dst->L.size)) {
        warn_once(W_COPY, "decoded ASTC copy between overlapping guest storage: skipped");
        return 1;
    }
    const VkTexImage *si = vk_texture_get(p->src), *di = vk_texture_get(p->dst);
    Sub s, d;
    // Full-res decoded staging (host_mip0==0) into a CompactLayers dest: fold the whole guest level.
    // Always fold for CompactLayers dests (n115/n139 black/rainbow when the ≤4/≤16 caps skipped
    // thicker packs). n173 cliffs still rainbow with a 48-layer clamp — cliff materials use the
    // upper layers, which stayed UNDEFINED. Cap only to the host image layer count (68).
    uint32_t fold_layers = di && di->layers ? di->layers : 0;
    int fold_dst = si && di && !si->host_mip0 && di->host_mip0;
    view_t dv_clamp = p->has_dv ? p->dv : (view_t){0};
    NVNcopyRegion dr_clamp = p->dr;
    if (fold_dst && fold_layers) {
        if (p->has_dv && p->dv.set_layers && p->dv.nlayers > fold_layers) {
            dv_clamp = p->dv;
            dv_clamp.nlayers = fold_layers;
        }
        if (p->dr.depth > (int)fold_layers) dr_clamp.depth = (int)fold_layers;
    }
    int ok_s = decoded_astc_sub(p->src, &p->sv, p->has_sv, &p->sr, &f, si, &s, 0);
    int ok_d = decoded_astc_sub(p->dst, &dv_clamp, p->has_dv, &dr_clamp, &f, di, &d, fold_dst);
    if (di && di->host_mip0 && si) {
        uint32_t ups = 0;
        int lch = 0, pw = 0;
        vk_texture_debug(si, &ups, &lch, &pw);
        VkMemInspect mi;
        int mok = vk_mem_inspect((uintptr_t)p->src->mem, p->src->L.size, &mi);
        ctrace((uintptr_t)p->src->mem | 1ull << 63,
               "t2t src state %#llx %ux%u fmt %#x: mirrored %u gpu_owned %u uploads %u last_change %d pages_written %d | "
               "pool kind %u flags %#x mirror %u (inspect %d) gpu_only %d",
               (unsigned long long)(uintptr_t)p->src->mem, si->width, si->height, p->src->d.format, si->mirrored, si->gpu_owned,
               ups, lch, pw, mi.pool_kind, mi.pool_flags, mi.mirror_state, mok, nvn_gpu_only((uintptr_t)p->src->mem));
    }
    if (big_img(si) || big_img(di))
        ctrace((uintptr_t)p->dst->mem + (p->has_dv ? p->dv.base_level : 0), "t2t fmt %#x src %#llx %ux%u L%u x%u hm%u view(lv %u+%u ly %u+%u) rg %d,%d,%d %dx%dx%d -> dst %#llx %ux%u L%u x%u hm%u "
               "view(lv %u+%u ly %u+%u) rg %d,%d,%d %dx%dx%d | ok %d/%d fold %d | s lv%u ly%u %d,%d %ux%ux%u d lv%u ly%u %d,%d %ux%ux%u",
               p->src->d.format, (unsigned long long)(uintptr_t)p->src->mem, si ? si->width : 0, si ? si->height : 0,
               si ? si->levels : 0, si ? si->layers : 0, si ? si->host_mip0 : 0, p->has_sv ? p->sv.base_level : 0,
               p->has_sv ? p->sv.nlevels : 0, p->has_sv ? p->sv.min_layer : 0, p->has_sv ? p->sv.nlayers : 0, p->sr.xoffset,
               p->sr.yoffset, p->sr.zoffset, p->sr.width, p->sr.height, p->sr.depth, (unsigned long long)(uintptr_t)p->dst->mem,
               di ? di->width : 0, di ? di->height : 0, di ? di->levels : 0, di ? di->layers : 0, di ? di->host_mip0 : 0,
               p->has_dv ? p->dv.base_level : 0, p->has_dv ? p->dv.nlevels : 0, p->has_dv ? p->dv.min_layer : 0,
               p->has_dv ? p->dv.nlayers : 0, p->dr.xoffset, p->dr.yoffset, p->dr.zoffset, p->dr.width, p->dr.height, p->dr.depth,
               ok_s, ok_d, fold_dst, s.level, s.layer0, s.x, s.y, s.w, s.h, s.nz, d.level, d.layer0, d.x, d.y, d.w, d.h, d.nz);
    if (!ok_s || !ok_d || !si || !di || si->image == di->image || s.nz != d.nz) {
        warn_once(W_COPY,
                  "decoded ASTC copy skipped (src_ok=%d dst_ok=%d bound=%d/%d host_mip=%u/%u fmt %#x/%#x %ux%u vs %ux%u)",
                  ok_s, ok_d, si && si->bound, di && di->bound, si ? si->host_mip0 : 0, di ? di->host_mip0 : 0,
                  si ? (unsigned)si->format : 0, di ? (unsigned)di->format : 0, s.w, s.h, d.w, d.h);
        return 1;
    }
    // CompactLayers: dest may be a folded top mip (scaled down). BC/ASTC cannot vkCmdBlitImage, so pick a
    // same-size source or dest mip when the other image has one. Block expansion can also leave a 1-block
    // edge mismatch (429→432 vs 429, or 4x2→4x4 on a trimmed edge); take the aligned intersection.
    // Odd guest heights (4084, 8188, …) do not match power-of-two mip chains exactly; allow a few pixels so
    // CompactLayers folds (4096→512) still CopyImage instead of skipping as a size mismatch.
    if ((s.w != d.w || s.h != d.h) && si->levels > 1) {
        uint32_t old = s.level;
        for (uint32_t lv = old + 1; lv < si->levels; lv++) {
            uint32_t mw = max1(si->width >> lv), mh = max1(si->height >> lv);
            int w_ok = mw == d.w, h_ok = mh == d.h || (mh > d.h ? mh - d.h : d.h - mh) <= 4;
            if (w_ok && h_ok) {
                uint32_t sh = lv - old;
                s.level = lv;
                s.x = (int32_t)((uint32_t)s.x >> sh);
                s.y = (int32_t)((uint32_t)s.y >> sh);
                s.w = d.w;
                s.h = d.h;
                break;
            }
        }
    }
    if ((s.w != d.w || s.h != d.h) && di->levels > 1) {
        uint32_t old = d.level;
        for (uint32_t lv = old + 1; lv < di->levels; lv++) {
            uint32_t mw = max1(di->width >> lv), mh = max1(di->height >> lv);
            int w_ok = mw == s.w, h_ok = mh == s.h || (mh > s.h ? mh - s.h : s.h - mh) <= 4;
            if (w_ok && h_ok) {
                uint32_t sh = lv - old;
                d.level = lv;
                d.x = (int32_t)((uint32_t)d.x >> sh);
                d.y = (int32_t)((uint32_t)d.y >> sh);
                d.w = s.w;
                d.h = s.h;
                break;
            }
        }
    }
    int si_rgba8 = decoded_astc_rgba8(si->format);
    int di_rgba8 = decoded_astc_rgba8(di->format);
    int di_bc = f.astc_bc && !di_rgba8;
    if (s.w != d.w || s.h != d.h) {
        uint32_t bbw = di_bc ? 4u : (di_rgba8 ? 0u : (uint32_t)f.astc_bw);
        uint32_t bbh = di_bc ? 4u : (di_rgba8 ? 0u : (uint32_t)f.astc_bh);
        uint32_t nw = s.w < d.w ? s.w : d.w, nh = s.h < d.h ? s.h : d.h;
        uint32_t aw = bbw ? nw / bbw * bbw : nw, ah = bbh ? nh / bbh * bbh : nh;
        uint32_t src_w = max1(si->width >> s.level), src_h = max1(si->height >> s.level);
        uint32_t dst_w = max1(di->width >> d.level), dst_h = max1(di->height >> d.level);
        int origin_ok = (!bbw || (!(s.x % (int)bbw) && !(d.x % (int)bbw))) &&
                        (!bbh || (!(s.y % (int)bbh) && !(d.y % (int)bbh)));
        // CompactLayers scale fold (staging 8192 → host 1024): do not crop the source to the dest
        // size — that would copy the top-left tile only. Leave the mismatch for the RGBA8 blit
        // below (or a matching mip pick above).
        int scale_fold = s.w >= d.w * 2 || s.h >= d.h * 2 || d.w >= s.w * 2 || d.h >= s.h * 2;
        if (scale_fold) {
            /* keep s/d extents; blit or skip below */
        } else if (!di_bc && nw && nh) {
            // Host is RGBA8 (no BC recompress): any matching intersection is legal.
            s.w = d.w = nw;
            s.h = d.h = nh;
        } else if (aw && ah) {
            s.w = d.w = aw;
            s.h = d.h = ah;
        } else if (bbw && bbh) {
            // Grow to the next full block when both images contain it (3x3/4x2→4x4 tile writes).
            // Otherwise keep a partial extent only when each side is a full block or ends on a mip edge
            // (Vulkan compressed copies allow a trailing partial block only at the image edge).
            uint32_t mw = s.w > d.w ? s.w : d.w, mh = s.h > d.h ? s.h : d.h;
            uint32_t ew = (mw + bbw - 1) / bbw * bbw, eh = (mh + bbh - 1) / bbh * bbh;
            if (origin_ok && ew && eh && (uint32_t)s.x + ew <= src_w && (uint32_t)s.y + eh <= src_h &&
                (uint32_t)d.x + ew <= dst_w && (uint32_t)d.y + eh <= dst_h) {
                s.w = d.w = ew;
                s.h = d.h = eh;
            } else if (nw && nh) {
                int src_w_ok = !(nw % bbw) || (uint32_t)s.x + nw == src_w;
                int src_h_ok = !(nh % bbh) || (uint32_t)s.y + nh == src_h;
                int dst_w_ok = !(nw % bbw) || (uint32_t)d.x + nw == dst_w;
                int dst_h_ok = !(nh % bbh) || (uint32_t)d.y + nh == dst_h;
                if (src_w_ok && src_h_ok && dst_w_ok && dst_h_ok) {
                    s.w = d.w = nw;
                    s.h = d.h = nh;
                }
            }
        }
    }
    // Scale fold into CompactLayers: RGBA8 blit when formats allow. Prefer this over skipping when
    // ASTC_BC left a BC dest with no matching src mip (logged as 4092→512 holes / missing ground).
    // A BC-recompressed source into an RGBA8 dest (CompactLayers or <256x256 RGBA8): vkCmdCopyImage would
    // move the raw 1-byte-per-texel BC blocks into the 4-byte RGBA8 texels (invalid usage; NVIDIA copies the bytes),
    // which left every MKW terrain layer as a scrambled top quarter over black: black road / ground up close, rainbow
    // cliffs. Blit instead: the GPU decodes BC on the read and writes real texels.
    int fmt_mismatch = si->format != di->format && !(si_rgba8 && di_rgba8);
    int src_blit = si_rgba8 || blit_src_ok(si->format);
    int rgba8_blit = (s.w != d.w || s.h != d.h || fmt_mismatch) && src_blit &&
                     (di_rgba8 || (!di_bc && (f.vk == VK_FORMAT_R8G8B8A8_UNORM || f.vk == VK_FORMAT_R8G8B8A8_SRGB)));
    if (fmt_mismatch)
        ctrace((uintptr_t)p->dst->mem | 1ull << 62, "t2t format mismatch src vk %d -> dst vk %d: %s",
               (int)si->format, (int)di->format, rgba8_blit ? "blit (decode)" : "no blit path");
    if (rgba8_blit) {
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        VkImageBlit bl = {
            .srcSubresource = {.aspectMask = s.aspect, .mipLevel = s.level, .baseArrayLayer = s.layer0 + (uint32_t)s.z,
                               .layerCount = s.nz},
            .srcOffsets = {{s.x, s.y, 0}, {s.x + (int32_t)s.w, s.y + (int32_t)s.h, 1}},
            .dstSubresource = {.aspectMask = d.aspect, .mipLevel = d.level, .baseArrayLayer = d.layer0 + (uint32_t)d.z,
                               .layerCount = d.nz},
            .dstOffsets = {{d.x, d.y, 0}, {d.x + (int32_t)d.w, d.y + (int32_t)d.h, 1}},
        };
        // NEAREST: LINEAR over an 8× CompactLayers fold (8192→1024) blends across material
        // tiles and showed as rainbow stripes on MKW cliffs / boost pads (n173–n214).
        vkCmdBlitImage(ex->cb, si->image, VK_IMAGE_LAYOUT_GENERAL, di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       VK_FILTER_NEAREST);
        ex->recorded++;
        // CompactLayers hosts stay CPU-mirrored so MapVirtual / guest writes keep re-uploading tiles.
        if (!di->host_mip0) vk_texture_gpu_written(p->dst);
        return 1;
    }
    if (s.w != d.w || s.h != d.h) {
        // Mip-edge partial src (4x3, 15x3, …) into an interior full-block dest (4x4, 16x4). CopyImage cannot
        // grow past the src mip, so shuttle the compressed blocks through a staging buffer sized for the
        // rounded-up dest extent (BC = 16 bytes/block).
        uint32_t bbw = di_bc ? 4u : (di_rgba8 ? 0u : (uint32_t)f.astc_bw);
        uint32_t bbh = di_bc ? 4u : (di_rgba8 ? 0u : (uint32_t)f.astc_bh);
        uint32_t src_w = max1(si->width >> s.level), src_h = max1(si->height >> s.level);
        uint32_t dst_w = max1(di->width >> d.level), dst_h = max1(di->height >> d.level);
        uint32_t mw = s.w > d.w ? s.w : d.w, mh = s.h > d.h ? s.h : d.h;
        uint32_t ew = bbw ? (mw + bbw - 1) / bbw * bbw : mw, eh = bbh ? (mh + bbh - 1) / bbh * bbh : mh;
        int src_edge = (uint32_t)s.x + s.w == src_w || (uint32_t)s.y + s.h == src_h;
        int origins = (!bbw || (!(s.x % (int)bbw) && !(d.x % (int)bbw))) &&
                      (!bbh || (!(s.y % (int)bbh) && !(d.y % (int)bbh)));
        uint32_t nb = bbw && bbh ? (ew / bbw) * (eh / bbh) : 0;
        // Cap: 256 KiB of BC blocks (16384×16) covers large streamed foliage/terrain tiles past the old 64-block pad.
        enum { ASTC_PAD_BLOCKS = 16384u };
        if (di_bc && !si_rgba8 && bbw && bbh && origins && src_edge && s.nz == 1 && d.nz == 1 && nb && nb <= ASTC_PAD_BLOCKS &&
            (uint32_t)d.x + ew <= dst_w && (uint32_t)d.y + eh <= dst_h) {
            // Persistent device-local pad (not vk_staging): exec CB must not allocate into the upload batch.
            static VkBuffer pad_buf;
            static VmaAllocation pad_alloc;
            static VkDevice pad_dev;
            static VkDeviceSize pad_bytes;
            VkCtx *c = vk_ctx();
            VkDeviceSize need = (VkDeviceSize)nb * 16u;
            if (c && (pad_buf == VK_NULL_HANDLE || pad_dev != c->device || pad_bytes < need)) {
                if (pad_buf) vmaDestroyBuffer(c->vma, pad_buf, pad_alloc), pad_buf = VK_NULL_HANDLE;
                VkDeviceSize cap = (VkDeviceSize)ASTC_PAD_BLOCKS * 16u;
                if (need < 64u * 16u) need = 64u * 16u;
                if (need > cap) need = cap;
                VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need,
                                         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
                if (vmaCreateBuffer(c->vma, &bi, &ai, &pad_buf, &pad_alloc, NULL) == VK_SUCCESS)
                    pad_dev = c->device, pad_bytes = bi.size;
                else pad_buf = VK_NULL_HANDLE, pad_bytes = 0;
            }
            if (pad_buf) {
                VkBufferImageCopy to_buf = {.bufferRowLength = ew, .bufferImageHeight = eh,
                                           .imageSubresource = sub_layers(&s), .imageOffset = sub_offset(&s),
                                           .imageExtent = {s.w, s.h, 1}};
                VkBufferImageCopy from_buf = {.bufferRowLength = ew, .bufferImageHeight = eh,
                                             .imageSubresource = sub_layers(&d), .imageOffset = sub_offset(&d),
                                             .imageExtent = {ew, eh, 1}};
                vk_exec_pass_end(ex);
                vk_exec_barrier(ex);
                vkCmdCopyImageToBuffer(ex->cb, si->image, VK_IMAGE_LAYOUT_GENERAL, pad_buf, 1, &to_buf);
                vk_exec_barrier(ex);
                vkCmdCopyBufferToImage(ex->cb, pad_buf, di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &from_buf);
                ex->recorded++;
                if (!di->host_mip0) vk_texture_gpu_written(p->dst);
                return 1;
            }
        }
        warn_once(W_COPY,
                  "decoded ASTC CompactLayers size mismatch %ux%u -> %ux%u (no matching src mip, compressed blit unsupported): skipped",
                  s.w, s.h, d.w, d.h);
        return 1;
    }
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (fmt_mismatch && !(si->format == di->format)) {
        int to_bc = di->format == VK_FORMAT_BC7_SRGB_BLOCK || di->format == VK_FORMAT_BC7_UNORM_BLOCK ||
                    di->format == VK_FORMAT_BC3_SRGB_BLOCK || di->format == VK_FORMAT_BC3_UNORM_BLOCK;
        uint32_t dl0 = d.layer0 + (uint32_t)d.z;
        if (to_bc && s.w == d.w && s.h == d.h && !si->host_mip0 && !di->host_mip0 &&
            vk_texture_astc_copy(p->src, s.level, s.layer0 + (uint32_t)s.z, s.nz, (uint32_t)s.x, (uint32_t)s.y, s.w, s.h,
                                 di, d.aspect, d.level, dl0, d.x, d.y, 1)) {
            vk_texture_gpu_written(p->dst);
            if (di->sparse) vk_sparse_written(di->image, d.level, dl0, s.nz, d.x, d.y, s.w, s.h);
            return 1;
        }
        warn_once(W_COPY, "decoded ASTC copy between incompatible formats %d -> %d without a blit: skipped", (int)si->format,
                  (int)di->format);
        return 1;
    }
    VkImageCopy cp = {sub_layers(&s), sub_offset(&s), sub_layers(&d), sub_offset(&d), sub_extent(&s)};
    vkCmdCopyImage(ex->cb, si->image, VK_IMAGE_LAYOUT_GENERAL, di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    ex->recorded++;
    // CompactLayers: do not set gpu_owned — that freezes make_fresh and stops streamed tile re-uploads (AV / holes).
    if (!di->host_mip0) vk_texture_gpu_written(p->dst);
    if (di->sparse) vk_sparse_written(di->image, d.level, d.layer0 + (uint32_t)d.z, s.nz, d.x, d.y, s.w, s.h);
    return 1;
}

// An ASTC texture copied into a sparse texture re-encoded as BC7/BC3: MKW fills each layer of its 8192x8192x68 terrain
// arrays, level by level, from an 8192x8192 ASTC staging texture (and small ASTC textures for coarse mips). Decoding
// directly from p->src->mem into the sparse array avoids allocating redundant 85+ MiB staging VkImages in VRAM
// (adriadam10/mkw-recomp, MIT).
static int astc_into_sparse(const p_copy_t2t *p) {
    TexVkFormat fs, fd;
    if (!p->src || !p->dst || !(p->dst->d.flags & NVN_TEXTURE_FLAGS_SPARSE) || !tex_vk_format(p->src->d.format, &fs) ||
        !fs.astc_bw || !tex_vk_format(p->dst->d.format, &fd) || fd.astc_bw != fs.astc_bw || fd.astc_bh != fs.astc_bh ||
        p->src->d.target == NVN_TEXTURE_TARGET_3D)
        return 0;
    Sub d;
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, 1, &d) || !d.img->sparse ||
        (d.img->format != VK_FORMAT_BC7_SRGB_BLOCK && d.img->format != VK_FORMAT_BC7_UNORM_BLOCK &&
         d.img->format != VK_FORMAT_BC3_SRGB_BLOCK && d.img->format != VK_FORMAT_BC3_UNORM_BLOCK))
        return 0;
    uint32_t level = p->has_sv && p->sv.set_levels ? p->sv.base_level : 0;
    uint32_t layer = (p->has_sv && p->sv.set_layers ? p->sv.min_layer : 0) + (uint32_t)p->sr.zoffset;
    uint32_t lw = max1(p->src->d.width >> level), lh = max1(p->src->d.height >> level);
    uint32_t x = (uint32_t)p->sr.xoffset, y = (uint32_t)p->sr.yoffset;
    uint32_t w = p->sr.width > 0 ? (uint32_t)p->sr.width : lw - x, h = p->sr.height > 0 ? (uint32_t)p->sr.height : lh - y;
    uint32_t n = p->sr.depth > 0 ? (uint32_t)p->sr.depth : 1, dl0 = d.layer0 + (uint32_t)d.z;
    if (level >= p->src->d.levels || x + w > lw || y + h > lh || w != d.w || h != d.h) return 0;
    if (!vk_texture_astc_copy(p->src, level, layer, n, x, y, w, h, d.img, d.aspect, d.level, dl0, d.x, d.y, 1))
        return 0;
    vk_texture_gpu_written(p->dst);
    vk_sparse_written(d.img->image, d.level, dl0, n, d.x, d.y, w, h);
    return 1;
}

static void copy_t2t(VkExec *ex, const p_copy_t2t *p) {
    static int trace = -1;
    static volatile int32_t ntrace;
    if (trace < 0) trace = cfg_get("MKW_SPARSE_TRACE") != NULL;
    if (trace && p->src && p->dst && ((p->src->d.flags | p->dst->d.flags) & NVN_TEXTURE_FLAGS_SPARSE) &&
        plat_inc32(&ntrace) <= 8192)
        rt_log("sparse: copy %ux%ux%u fmt %#x flags %#x lv%u [%d,%d,%d %dx%dx%d] -> %#llx %ux%ux%u fmt %#x [%d,%d,%d "
               "%dx%dx%d] level %d layer %d", p->src->d.width, p->src->d.height, p->src->d.depth, p->src->d.format,
               p->src->d.flags, p->src->d.levels, p->sr.xoffset, p->sr.yoffset, p->sr.zoffset, p->sr.width,
               p->sr.height, p->sr.depth, (unsigned long long)(uintptr_t)p->dst->mem, p->dst->d.width,
               p->dst->d.height, p->dst->d.depth, p->dst->d.format, p->dr.xoffset, p->dr.yoffset, p->dr.zoffset,
               p->dr.width, p->dr.height, p->dr.depth, p->has_dv && p->dv.set_levels ? p->dv.base_level : -1,
               p->has_dv && p->dv.set_layers ? p->dv.min_layer : -1);
    if (astc_into_sparse(p)) return;
    if (copy_decoded_astc(ex, p)) return;
    Sub s, d;
    if (!sub_resolve(p->src, &p->sv, p->has_sv, &p->sr, 1, 2, &s)) return;
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, 2, &d)) return;
    d.nz = s.nz;
    int same_size = s.w == d.w && s.h == d.h;
    int same_format = s.img->format == d.img->format;
    if (s.img->samples != d.img->samples && !(s.img->samples > 1 && d.img->samples == 1 && same_size && same_format)) {
        warn_once(W_COPY, "texture copy between %u and %u samples: skipped", s.img->samples, d.img->samples);
        return;
    }
    if (s.aspect != d.aspect || (s.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !same_format)) {
        warn_once(W_COPY, "texture copy between formats %#x and %#x: skipped", p->src->d.format, p->dst->d.format);
        return;
    }
    VkImageSubresourceLayers sl = sub_layers(&s), dl = sub_layers(&d);
    // Between a 2D array and a 3D image, the layers of one side are the depth slices of the other (VUID 01790..01792),
    // which fills all 32 slices of MKW's 32x32x32 race colour-grading LUTs (adriadam10/mkw-recomp, MIT).
    int s3d = s.img->type == VK_IMAGE_TYPE_3D, d3d = d.img->type == VK_IMAGE_TYPE_3D;
    VkExtent3D ext = {s.w, s.h, s3d || d3d ? s.nz : 1};
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (s.img->samples > 1 && d.img->samples == 1) {
        VkImageResolve rv = {sl, sub_offset(&s), dl, sub_offset(&d), ext};
        vkCmdResolveImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
    } else if (same_size && same_format) {
        VkImageCopy cp = {sl, sub_offset(&s), dl, sub_offset(&d), ext};
        vkCmdCopyImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    } else {   // scaled or converting copy: nearest, or linear when asked for (colour only)
        int linear = (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
        if (s.aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            VkImageBlit bl[2];
            static const VkImageAspectFlags kAsp[2] = {VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT};
            for (int i = 0; i < 2; i++) {
                VkImageSubresourceLayers sla = sl, dla = dl;
                sla.aspectMask = dla.aspectMask = kAsp[i];
                bl[i] = (VkImageBlit){.srcSubresource = sla, .dstSubresource = dla};
                bl[i].srcOffsets[0] = sub_offset(&s);
                bl[i].srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, s3d ? s.z + (int32_t)s.nz : 1};
                bl[i].dstOffsets[0] = sub_offset(&d);
                bl[i].dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, d3d ? d.z + (int32_t)d.nz : 1};
            }
            vkCmdBlitImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 2, bl,
                           VK_FILTER_NEAREST);
        } else {
            uint32_t n = s3d != d3d ? s.nz : 1;
            for (uint32_t k = 0; k < n; k++) {
                VkImageBlit bl = {.srcSubresource = sl, .dstSubresource = dl};
                int32_t sz = s.z + (int32_t)k, dz = d.z + (int32_t)k, depth = n > 1 ? 1 : (int32_t)s.nz;
                if (n > 1) {
                    if (!s3d) bl.srcSubresource.baseArrayLayer = s.layer0 + (uint32_t)sz, bl.srcSubresource.layerCount = 1;
                    if (!d3d) bl.dstSubresource.baseArrayLayer = d.layer0 + (uint32_t)dz, bl.dstSubresource.layerCount = 1;
                }
                bl.srcOffsets[0] = (VkOffset3D){s.x, s.y, s3d ? sz : 0};
                bl.srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, s3d ? sz + depth : 1};
                bl.dstOffsets[0] = (VkOffset3D){d.x, d.y, d3d ? dz : 0};
                bl.dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, d3d ? dz + depth : 1};
                vkCmdBlitImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                               linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
            }
        }
    }
    ex->recorded++;
    vk_texture_gpu_written(p->dst);
    if (d.img->sparse)
        vk_sparse_written(d.img->image, d.level, d.layer0 + (d3d ? 0 : (uint32_t)d.z), d3d ? 1 : d.nz, d.x, d.y, d.w, d.h);
}

// ---- buffer to buffer

static void copy_b2b(VkExec *ex, const p_copy_b2b *p) {
    uint64_t src = p->src, dst = p->dst, left = p->size;
    int started = 0;
    while (left) {
        VkMemSpan s, d;
        if (!vk_mem_resolve(src, 1, &s) || !vk_mem_resolve(dst, 1, &d)) {
            warn_once(W_BUFFER, "buffer copy %#llx -> %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < s.avail ? left : s.avail;
        if (n > d.avail) n = d.avail;
        VK_MEM_PICK(s);
        VK_MEM_PICK(d);
        if (s.buffer == d.buffer && s.offset < d.offset + n && d.offset < s.offset + n) {
            warn_once(W_OVERLAP, "buffer copy %#llx -> %#llx overlaps within one buffer: skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst);
            break;
        }
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VkBufferCopy bc = {s.offset, d.offset, n};
        vkCmdCopyBuffer(ex->cb, s.buffer, d.buffer, 1, &bc);
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(s.cls, VK_MUSE_COPY_SRC, n);
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        src += n, dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->dst, p->size - left);
}

// ---- buffer fills (ClearBuffer, and ReportValue as a 4-byte one): vkCmdFillBuffer in GPU order after a full
// barrier, so the value lands after earlier work, as NVN's semaphore release does; CPU-visible ranges go to
// vk_guest_written at the fence, as copies do. Works in CPU-invisible pools.

static void clear_buf(VkExec *ex, const p_clear_buf *p) {
    if ((p->addr | p->size) & 3) {
        warn_once(W_BUFFER, "buffer fill at %#llx of %llu bytes is not 4-byte aligned: skipped", (unsigned long long)p->addr,
                  (unsigned long long)p->size);
        return;
    }
    uint64_t dst = p->addr, left = p->size;
    int started = 0;
    while (left) {
        VkMemSpan d;
        if (!vk_mem_resolve(dst, 1, &d) || d.offset % 4) {
            warn_once(W_BUFFER, "buffer fill at %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->addr, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < d.avail ? left : d.avail & ~3ull;
        if (!n) break;
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VK_MEM_PICK(d);
        vkCmdFillBuffer(ex->cb, d.buffer, d.offset, n, p->value);
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->addr, p->size - left);
}

// ---- texture clears

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return 1;
    default: return 0;
    }
}

// Whole levels only (vkCmdClear*Image has no rectangle); partial Z-slice clears of 2D-array-compatible 3D colour
// textures are cleared via a single dynamic rendering clear pass over their slice range.
static void clear_tex(VkExec *ex, const p_clear_tex *p) {
    const tex_t *t = p->tex;
    if (!t) return;
    Sub s;
    if (!sub_resolve(t, &p->view, p->has_view, &p->r, 1, 2, &s)) return;
    const VkTexImage *img = s.img;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t lw = max1(img->width >> s.level), lh = max1(img->height >> s.level);
    int reformat = p->has_view && p->view.set_format && p->view.format != t->d.format;
    int partial_3d = is3d && (s.z || s.nz != max1(img->depth >> s.level));
    if (partial_3d && !s.x && !s.y && s.w == lw && s.h == lh && (p->mask & 0xf) == 0xf && !reformat &&
        (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (img->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        img->samples == 1) {
        if (int_format(img->format) != (p->is_uint != 0)) {
            warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint,
                      (int)img->format);
            return;
        }
        VkClearColorValue cv;
        memcpy(&cv, &p->c, sizeof cv);
        VkImageView view = vk_exec_target_view(img, img->format,
                                               s.nz > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                                               VK_IMAGE_ASPECT_COLOR_BIT, s.level, (uint32_t)s.z, s.nz, 0);
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                        .imageView = view,
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                        .clearValue = {.color = cv}};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                              .renderArea = {{0, 0}, {lw, lh}},
                              .layerCount = s.nz,
                              .colorAttachmentCount = 1,
                              .pColorAttachments = &ca};
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdBeginRendering(ex->cb, &ri);
        vkCmdEndRendering(ex->cb);
        ex->recorded += 2;
        vk_texture_gpu_written(t);
        return;
    }
    if (s.x || s.y || s.w != lw || s.h != lh || (p->mask & 0xf) != 0xf || reformat || partial_3d) {
        warn_once(W_CLEAR, "texture clear of region %d,%d %ux%u mask %#x (level %ux%u, view format change %d): skipped",
                  s.x, s.y, s.w, s.h, p->mask, lw, lh, reformat);
        return;
    }
    VkImageSubresourceRange rng = {img->aspect, s.level, 1, is3d ? 0 : s.layer0 + (uint32_t)s.z, is3d ? 1 : s.nz};
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkClearColorValue cv;
        if (int_format(img->format) != (p->is_uint != 0)) {
            warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint, (int)img->format);
            return;
        }
        memcpy(&cv, &p->c, sizeof cv);
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearColorImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    } else {
        int both = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        VkClearDepthStencilValue dv = {0, 0};
        if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) dv.depth = p->c.f[0];
        if (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) dv.stencil = both ? p->c.u[1] : p->c.u[0];
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearDepthStencilImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rng);
    }
    ex->recorded++;
    vk_texture_gpu_written(t);
}

void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_COPY_B2B: copy_b2b(ex, payload); break;
    case OP_COPY_B2T: copy_buffer_image(ex, payload, 1); break;
    case OP_COPY_T2B: copy_buffer_image(ex, payload, 0); break;   // p_copy_b2t: src holds the destination address
    case OP_COPY_T2T: copy_t2t(ex, payload); break;
    case OP_CLEAR_TEX: clear_tex(ex, payload); break;
    case OP_SIGNAL_EVENT: signal_event(ex, payload); break;
    case OP_REPORT_COUNTER: report_counter(ex, payload); break;
    case OP_RESET_COUNTER: reset_counter(ex, payload); break;
    case OP_CLEAR_BUF: clear_buf(ex, payload); break;
    case OP_RENDER_ENABLE: ex->render_disabled = !((const p_flag *)payload)->value; break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        ex->render_disabled = !cond_enabled(payload);
        break;
    default: rt_fatal("vk: op %s is not a transfer", nvn_op_name(op));
    }
}

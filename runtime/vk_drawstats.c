// MKW_DRAWSTATS=<first frame>:<frames> (diagnostics, foliage pop-in plan Phase 0): a per-frame log of what the big
// render passes drew, to find objects that blink without screenshots. For each pass with at least DS_MIN_DRAWS draws,
// or any indirect-count or mesh-task draw, of a frame in the range, one line in the log:
//   dstat f<frame> p<seq> <w>x<h> draws <n> pipes <distinct pipelines> fp <fingerprint> icd <n> mesh <n>/<groups>
// fp hashes the pass's (pipeline, draw kind) sequence: equal from frame to frame while the same things are drawn the
// same way. The GPU-written counts of its indirect-count draws (the game's culling / LOD output) are copied out at the
// pass end and logged when the submission completes:
//   dcnt f<frame> p<seq> <count> <count> ...
// A blinking object shows as fp or a count that alternates between frames of a still scene (MKW_SEED fixes the scene).
// Phase 1: in passes of at least DS_DIFF_DRAWS draws, the shader programs whose draw count changed since the same pass of
// the previous frame, and the draws conditional rendering dropped there:
//   ddiff f<frame> p<seq> prog <serial> <before> -> <now> inst <before> -> <now>   dskip f<frame> p<seq> prog <serial> <n>
// (inst: the program's instances over its direct draws), and the draws skipped while their pipeline builds:
//   dpend f<frame> p<seq> prog <serial> <n>
// and per frame the conditional rendering decisions (render / skip / render because the query was not done yet):
//   dcond f<frame> render <n> skip <n> pending <n>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"

enum { DS_MIN_DRAWS = 40, DS_MAX_COUNTS = 512, DS_DIFF_DRAWS = 200, DS_PROGS = 512, DS_KEEP = 128, DS_LINES = 40 };

static int ds_state = -1;   // -1 not read, 0 off, 1 on
static uint64_t ds_from, ds_to;
static uint64_t ds_key_n;   // MKW_DRAWSTATS=key:<n>
static int ds_frame_on(uint64_t *frame);

// F9 (gfx_window_win.c): with MKW_DRAWSTATS=key:<n>, record the n frames from two presents on. The range is written by
// the window thread and read by the replay thread; a torn read only shifts the first recorded frame.
void vk_drawstats_key(void) {
    ds_frame_on(NULL);
    if (!ds_key_n) return;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 2;
    __atomic_store_n(&ds_to, f + ds_key_n, __ATOMIC_RELAXED);
    __atomic_store_n(&ds_from, f, __ATOMIC_RELAXED);
    rt_log("drawstats: F9 pressed: recording frames %llu..%llu", (unsigned long long)f,
           (unsigned long long)(f + ds_key_n - 1));
}

static int ds_frame_on(uint64_t *frame) {
    if (ds_state < 0) {
        const char *e = cfg_get("MKW_DRAWSTATS");
        unsigned long long a = 0, n = 0;
        if (e && sscanf(e, "key:%llu", &n) == 1 && n) {
            // MKW_DRAWSTATS=key:<frames>: armed with no range; F9 in the game window records the next <frames>
            ds_key_n = n;
            ds_from = ds_to = 0;
            ds_state = 1;
            rt_log("drawstats: armed, F9 records %llu frames (MKW_DRAWSTATS)", n);
        } else {
            ds_state = e && sscanf(e, "%llu:%llu", &a, &n) == 2 && n;
            if (ds_state) {
                ds_from = a, ds_to = a + n;
                rt_log("drawstats: frames %llu..%llu (MKW_DRAWSTATS)", a, a + n - 1);
            }
        }
    }
    if (!ds_state) return 0;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (frame) *frame = f;
    uint64_t from = __atomic_load_n(&ds_from, __ATOMIC_RELAXED), to = __atomic_load_n(&ds_to, __ATOMIC_RELAXED);
    return f >= from && f < to;
}

// The open pass (replay thread only).
static struct {
    uint64_t frame, fp, last_pipe;
    uint32_t draws, pipes, icd, mesh, groups, seq;
    uint32_t ncount;
    VkMemSpan counts[DS_MAX_COUNTS];
    uint32_t nprog;
    uint64_t prog[DS_PROGS];
    uint32_t pdraw[DS_PROGS], pskip[DS_PROGS], pinst[DS_PROGS], ppend[DS_PROGS];
    int last_k;
} P;

// The per-program counts of the big passes of the previous frame, by pass number.
static struct { int valid; uint32_t n; uint64_t prog[DS_PROGS]; uint32_t draw[DS_PROGS], inst[DS_PROGS]; } K[DS_KEEP];
static uint32_t C[3];   // this frame's conditional rendering decisions: skip, render, pending

static int prog_slot(uint64_t prog) {
    for (uint32_t i = 0; i < P.nprog; i++)
        if (P.prog[i] == prog) return (int)i;
    if (P.nprog == DS_PROGS) return -1;
    P.prog[P.nprog] = prog, P.pdraw[P.nprog] = P.pskip[P.nprog] = P.pinst[P.nprog] = P.ppend[P.nprog] = 0;
    return (int)P.nprog++;
}

void vk_drawstats_skip(VkExec *ex, uint64_t prog) {
    (void)ex;
    if (!ds_frame_on(NULL)) return;
    int k = prog_slot(prog);
    if (k >= 0) P.pskip[k]++;
}

void vk_drawstats_cond(int result) {
    if (ds_frame_on(NULL) && result >= 0 && result <= 2) C[result]++;
}

static void mix(uint64_t v) { P.fp = (P.fp ^ v) * 0x100000001b3ull; }

void vk_drawstats_draw(VkExec *ex, uint64_t pipe, uint64_t prog) {
    (void)ex;
    if (!ds_frame_on(&P.frame)) return;
    P.draws++;
    int k = prog_slot(prog);
    if (k >= 0) P.pdraw[k]++;
    P.last_k = k;
    if (pipe != P.last_pipe) P.pipes++, P.last_pipe = pipe;
    mix(pipe);
}

// Called right after vk_drawstats_draw for direct draws (indirect ones count 0 instances here).
void vk_drawstats_inst(uint32_t instances) {
    if (!ds_frame_on(NULL) || P.last_k < 0) return;
    P.pinst[P.last_k] += instances;
}

void vk_drawstats_pending(uint64_t prog) {
    if (!ds_frame_on(NULL)) return;
    int k = prog_slot(prog);
    if (k >= 0) P.ppend[k]++;
}

void vk_drawstats_count(VkExec *ex, const VkMemSpan *count) {
    (void)ex;
    if (!ds_frame_on(NULL)) return;
    P.icd++;
    mix(0x1cd);
    if (P.ncount < DS_MAX_COUNTS) P.counts[P.ncount++] = *count;
}

void vk_drawstats_mesh(VkExec *ex, uint32_t groups) {
    (void)ex;
    if (!ds_frame_on(NULL)) return;
    P.mesh++;
    P.groups += groups;
    mix(0x3e5 + groups);
}

typedef struct {
    uint64_t frame;
    uint32_t seq, n;
    VkBuffer buf;
    VmaAllocation alloc;
} CountJob;

static void count_cb(void *arg) {
    CountJob *j = arg;
    VkCtx *c = vk_ctx();
    void *map = NULL;
    if (c && vmaMapMemory(c->vma, j->alloc, &map) == VK_SUCCESS) {
        vmaInvalidateAllocation(c->vma, j->alloc, 0, VK_WHOLE_SIZE);
        const uint32_t *v = map;
        char line[4096];
        int o = snprintf(line, sizeof line, "dcnt f%llu p%u", (unsigned long long)j->frame, j->seq);
        for (uint32_t i = 0; i < j->n && o < (int)sizeof line - 16; i++) o += snprintf(line + o, sizeof line - o, " %u", v[i]);
        rt_log("%s", line);
        vmaUnmapMemory(c->vma, j->alloc);
    }
    if (c) vmaDestroyBuffer(c->vma, j->buf, j->alloc);
    free(j);
}

// After vkCmdEndRendering (copies are legal again): the line, and the count copies for the completion thread.
void vk_drawstats_pass_end(VkExec *ex) {
    if (!ds_frame_on(NULL) && !P.draws && !P.mesh && !P.icd) return;
    // a new frame restarts the pass numbering, so p<n> names the same pass in every frame
    static uint64_t seen_frame;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (f != seen_frame) {
        if (seen_frame && (C[0] || C[1] || C[2]))
            rt_log("dcond f%llu render %u skip %u pending %u", (unsigned long long)seen_frame, C[1], C[0], C[2]);
        C[0] = C[1] = C[2] = 0;
        seen_frame = f, P.seq = 0;
    }
    uint32_t seq = P.seq++;
    if (P.draws >= DS_DIFF_DRAWS && seq < DS_KEEP) {   // what changed per program since this pass of the last frame
        int lines = 0;
        if (K[seq].valid) {
            for (uint32_t i = 0; i < P.nprog && lines < DS_LINES; i++) {
                uint32_t before = 0, binst = 0;
                for (uint32_t j = 0; j < K[seq].n; j++)
                    if (K[seq].prog[j] == P.prog[i]) before = K[seq].draw[j], binst = K[seq].inst[j];
                if (before != P.pdraw[i] || binst != P.pinst[i])
                    rt_log("ddiff f%llu p%u prog %llu %u -> %u inst %u -> %u", (unsigned long long)P.frame, seq,
                           (unsigned long long)P.prog[i], before, P.pdraw[i], binst, P.pinst[i]), lines++;
            }
            for (uint32_t j = 0; j < K[seq].n && lines < DS_LINES; j++) {   // programs that vanished entirely
                int still = 0;
                for (uint32_t i = 0; i < P.nprog; i++) still |= P.prog[i] == K[seq].prog[j];
                if (!still && K[seq].draw[j])
                    rt_log("ddiff f%llu p%u prog %llu %u -> 0 inst %u -> 0", (unsigned long long)P.frame, seq,
                           (unsigned long long)K[seq].prog[j], K[seq].draw[j], K[seq].inst[j]), lines++;
            }
        }
        for (uint32_t i = 0; i < P.nprog && lines < DS_LINES; i++)
            if (P.pskip[i]) rt_log("dskip f%llu p%u prog %llu %u", (unsigned long long)P.frame, seq,
                                  (unsigned long long)P.prog[i], P.pskip[i]), lines++;
        for (uint32_t i = 0; i < P.nprog && lines < DS_LINES; i++)
            if (P.ppend[i]) rt_log("dpend f%llu p%u prog %llu %u", (unsigned long long)P.frame, seq,
                                  (unsigned long long)P.prog[i], P.ppend[i]), lines++;
        K[seq].valid = 1, K[seq].n = P.nprog;
        memcpy(K[seq].prog, P.prog, sizeof P.prog);
        memcpy(K[seq].draw, P.pdraw, sizeof P.pdraw);
        memcpy(K[seq].inst, P.pinst, sizeof P.pinst);
    }
    if (P.draws >= DS_MIN_DRAWS || P.icd || P.mesh) {
        char rt[160] = "";
        if (ex->pass_w <= 512) {   // small passes (impostor / probe tiles): where they render, as guest storage
            int o = 0;
            for (int i = 0; i < 9 && o < (int)sizeof rt - 48; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if (!t->tex) continue;
                o += snprintf(rt + o, sizeof rt - o, " %c%d %llx L%u/%u", i < 8 ? 'c' : 'd', i < 8 ? i : 0,
                              (unsigned long long)(uintptr_t)t->tex->mem, t->base_layer, t->tex->L.layers);
            }
        }
        rt_log("dstat f%llu p%u %ux%u draws %u pipes %u fp %016llx icd %u mesh %u/%u%s", (unsigned long long)P.frame, seq,
               ex->pass_w, ex->pass_h, P.draws, P.pipes, (unsigned long long)P.fp, P.icd, P.mesh, P.groups, rt);
        VkCtx *c = vk_ctx();
        if (P.ncount && c) {
            CountJob *j = calloc(1, sizeof *j);
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = P.ncount * 4u,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
            if (j && vmaCreateBuffer(c->vma, &bi, &ai, &j->buf, &j->alloc, NULL) == VK_SUCCESS) {
                for (uint32_t i = 0; i < P.ncount; i++) {
                    VkBufferCopy cp = {P.counts[i].offset, i * 4u, 4};
                    vkCmdCopyBuffer(ex->cb, P.counts[i].buffer, j->buf, 1, &cp);
                }
                VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                       .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                                       .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
                VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                                       .pMemoryBarriers = &mb};
                vkCmdPipelineBarrier2(ex->cb, &di);
                ex->recorded++;
                j->frame = P.frame, j->seq = seq, j->n = P.ncount;
                vk_completion_add(ex->fence, count_cb, j);
            } else {
                free(j);
            }
        }
    }
    memset(&P, 0, sizeof P);
    P.seq = seq + 1;
    P.frame = f;
}

// A samples-passed report reached guest memory (completion thread): the frame it was recorded in, the frame being
// presented now, the address the game reads and the count it gets. At most DS_OQ_LINES per recorded frame.
//   doq f<recorded> at f<now> a<address> v<count> q<queries>
enum { DS_OQ_LINES = 64 };
void vk_drawstats_oq(uint64_t rec_frame, uint64_t addr, uint64_t count, uint32_t queries) {
    if (ds_state <= 0 || rec_frame < ds_from || rec_frame >= ds_to) return;
    static uint64_t frame_seen;
    static int lines;
    if (rec_frame != frame_seen) frame_seen = rec_frame, lines = 0;
    if (lines++ >= DS_OQ_LINES) return;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    rt_log("doq f%llu at f%llu a%llx v%llu q%u", (unsigned long long)rec_frame, (unsigned long long)now,
           (unsigned long long)addr, (unsigned long long)count, queries);
}

// Timestamp reports (the game's GPU timing): per frame the number of reports and the spread between the first and the
// last tick the game will read, in microseconds. Logged when the next frame's first report comes.
//   dts f<frame> n<reports> span <us> us
void vk_drawstats_ts(uint64_t frame, uint64_t addr, uint64_t ticks) {
    (void)addr;
    static uint64_t cur, lo, hi;
    static uint32_t n;
    if (frame != cur) {
        if (cur && ds_state > 0 && cur >= ds_from && cur < ds_to)
            rt_log("dts f%llu n%u span %llu us", (unsigned long long)cur, n, (unsigned long long)(rt_ticks_to_ns(hi - lo) / 1000));
        cur = frame, lo = hi = ticks, n = 0;
    }
    if (ticks < lo) lo = ticks;
    if (ticks > hi) hi = ticks;
    n++;
}

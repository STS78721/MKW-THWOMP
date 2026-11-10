// MKW_DRAWSTATS=<first frame>:<frames> (diagnostics, foliage pop-in plan Phase 0): a per-frame log of what the big
// render passes drew, to find objects that blink without screenshots. For each pass with at least DS_MIN_DRAWS draws,
// or any indirect-count or mesh-task draw, of a frame in the range, one line in the log:
//   dstat f<frame> p<seq> <w>x<h> draws <n> pipes <distinct pipelines> fp <fingerprint> icd <n> mesh <n>/<groups>
// fp hashes the pass's (pipeline, draw kind) sequence: equal from frame to frame while the same things are drawn the
// same way. The GPU-written counts of its indirect-count draws (the game's culling / LOD output) are copied out at the
// pass end and logged when the submission completes:
//   dcnt f<frame> p<seq> <count> <count> ...
// A blinking object shows as fp or a count that alternates between frames of a still scene (MKW_SEED fixes the scene).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"

enum { DS_MIN_DRAWS = 40, DS_MAX_COUNTS = 512 };

static int ds_state = -1;   // -1 not read, 0 off, 1 on
static uint64_t ds_from, ds_to;

static int ds_frame_on(uint64_t *frame) {
    if (ds_state < 0) {
        const char *e = cfg_get("MKW_DRAWSTATS");
        unsigned long long a = 0, n = 0;
        ds_state = e && sscanf(e, "%llu:%llu", &a, &n) == 2 && n;
        if (ds_state) {
            ds_from = a, ds_to = a + n;
            rt_log("drawstats: frames %llu..%llu (MKW_DRAWSTATS)", a, a + n - 1);
        }
    }
    if (!ds_state) return 0;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (frame) *frame = f;
    return f >= ds_from && f < ds_to;
}

// The open pass (replay thread only).
static struct {
    uint64_t frame, fp, last_pipe;
    uint32_t draws, pipes, icd, mesh, groups, seq;
    uint32_t ncount;
    VkMemSpan counts[DS_MAX_COUNTS];
} P;

static void mix(uint64_t v) { P.fp = (P.fp ^ v) * 0x100000001b3ull; }

void vk_drawstats_draw(VkExec *ex, uint64_t pipe) {
    (void)ex;
    if (!ds_frame_on(&P.frame)) return;
    P.draws++;
    if (pipe != P.last_pipe) P.pipes++, P.last_pipe = pipe;
    mix(pipe);
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
    if (f != seen_frame) seen_frame = f, P.seq = 0;
    uint32_t seq = P.seq++;
    if (P.draws >= DS_MIN_DRAWS || P.icd || P.mesh) {
        rt_log("dstat f%llu p%u %ux%u draws %u pipes %u fp %016llx icd %u mesh %u/%u", (unsigned long long)P.frame, seq,
               ex->pass_w, ex->pass_h, P.draws, P.pipes, (unsigned long long)P.fp, P.icd, P.mesh, P.groups);
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

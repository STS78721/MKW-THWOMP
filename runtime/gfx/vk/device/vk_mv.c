// THWOMP motion vectors, prototype (the groundwork for DLSS): MKW draws no motion vectors and does not jitter its
// camera, but every draw of its 3D scene binds one camera block (vertex constant buffer c4, 0x980 bytes) that holds
// this frame's view-projection (rows 3-6) and the previous frame's (rows 24-27), with the render size in row 18. With
// the scene's depth buffer that gives each pixel's motion caused by the camera. What objects do by themselves (the
// karts) is not in it yet.
//
// Capture (vk_mv_draw, the replay's draw path, MKW_MV_DEBUG only): the first draw of a frame whose c4 block is that
// size and whose render size equals the pass's depth target keeps a copy of the block's 32 rows and the depth image.
// The frame's present (vk_mv_frame) makes that capture the current one. The present thread (vk_mv_record) then runs
// shaders/mv_debug.comp over the crop and shows its debug picture instead of the game's (MKW_MV_DEBUG=1|2|3, see the
// shader). MKW_MV_FLIPY=0 and MKW_MV_ZMODE=1 try the other conventions. MKW_MV_SHOT=lo-hi:step [MKW_MV_SHOT_DIR] saves
// the debug picture of those frames as mv_<frame>.png.
#include "vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "mv_debug_spv.h"
#include "mv_out_spv.h"
#include "mv_vel_fs_spv.h"
#include "mv_vel_vs_spv.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "vk_exec.h"

#define SLOTS 8
#define CAM_SIZE 0x980u

int vk_mv_armed = -1;   // -1: not read yet; 0 off; 1 the replay captures (the debug view or DLSS wants them)
static int mv_debug;     // MKW_MV_DEBUG: the debug view shown instead of the picture, 0 none

typedef struct {
    uint64_t frame;
    float r[32][4];
    VkImage depth;
    VkFormat dfmt;
    uint32_t w, h;
    int cut;
} Cam;

static PlatMutex ml = PLAT_MUTEX_INIT;
static Cam pend, ready, last;
static int flipy = 1, zmode = 0;
static int jitter_on;   // MKW_MV_JITTER (camera jitter, below)
static uint64_t shot_lo, shot_hi, shot_step;
static const char *shot_dir = ".";

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler nearest, linear;
    VkImage out, hist[2];
    VmaAllocation out_a, hist_a[2];
    VkImageView out_v, hist_v[2];
    uint32_t w, h;
    int cur;   // hist[cur] holds the last picture
    int have_hist;
    VkImageView src_view[SLOTS], depth_view[SLOTS];
    VkBuffer shot_buf, stat_buf;
    VmaAllocation stat_a;
    VmaAllocation shot_a;
    uint64_t shot_frame;   // the frame whose picture shot_buf receives; 0 none
    VkFormat last_fmt;     // the presented picture's view format (colour check)
} M;

static void read_cfg(void) {
    const char *e = getenv("MKW_MV_DEBUG");
    mv_debug = e ? atoi(e) : 0;
    if (mv_debug < 0 || mv_debug > 6) mv_debug = 0;
    // DLSS (vk_dlss.cpp) needs the captures from the first frame: read at start, so switching the upscaler to DLSS
    // in the menu takes a restart to begin capturing
    vk_mv_armed = mv_debug > 0 || !strcmp(cfg_value("MKW_UPSCALER"), "dlss");
    if ((e = getenv("MKW_MV_FLIPY"))) flipy = atoi(e);
    if ((e = getenv("MKW_MV_ZMODE"))) zmode = atoi(e);
    if ((e = getenv("MKW_MV_SHOT"))) {
        unsigned long long a, b, s;
        if (sscanf(e, "%llu-%llu:%llu", &a, &b, &s) == 3 && s) shot_lo = a, shot_hi = b, shot_step = s;
    }
    if ((e = getenv("MKW_MV_SHOT_DIR"))) shot_dir = e;
    if ((e = getenv("MKW_MV_JITTER"))) jitter_on = atoi(e) > 0;
    if (vk_mv_armed)
        rt_log("mv: captures on, debug view %d (flip y %d, depth %s), jitter %s", mv_debug, flipy, zmode ? "[0,1]" : "[-1,1]",
               jitter_on ? "on" : "off");
}

// ---- camera jitter (DLSS super resolution, MKW_MV_JITTER=1): each frame the scene's projection moves by a sub-pixel
// offset from a 16-step Halton (2, 3) sequence, written into the submission's copy of the camera block (rows 3-4 of
// the view-projection and 7-8 of the projection get the offset times their w row), so every scene draw of the frame
// sees it. The game's own copy in guest memory is left alone: the "previous" rows it writes next frame and the
// matrices the camera motion vectors use stay unjittered. Offsets in pixels, x right, y down.
static uint64_t jit_frame;
static const uint8_t *jit_done;

static float halton(int i, int b) {
    float f = 1, r = 0;
    for (; i > 0; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

void vk_mv_jitter_px(uint64_t frame, float *x, float *y) {
    if (!jitter_on || !frame) {
        *x = *y = 0;
        return;
    }
    int i = (int)(frame % 16) + 1;
    *x = halton(i, 2) - 0.5f, *y = halton(i, 3) - 0.5f;
}

static void jitter_apply(const p_bind_buf *u, uint64_t frame) {
    static uint64_t applied, missing, last_log;
    uint8_t *g = vk_usnap_gpu_bytes(u->addr, 32 * 16);
    if (frame >= last_log + 600) {
        if (last_log) rt_log("mv: jitter written into %llu camera block copies, %llu draws without a copy",
                             (unsigned long long)applied, (unsigned long long)missing);
        last_log = frame, applied = missing = 0;
    }
    if (!g) {
        missing++;
        return;
    }
    if (g == jit_done && frame == jit_frame) return;
    applied++;
    const float *f = (const float *)(uintptr_t)u->addr;   // the game's unjittered rows
    float r[32][4], ox, oy;
    memcpy(r, f, sizeof r);
    vk_mv_jitter_px(frame, &ox, &oy);
    float jx = 2.0f * ox / r[18][0], jy = -2.0f * oy / r[18][1];   // NDC, y up
    for (int k = 0; k < 4; k++) {
        r[3][k] += jx * r[6][k], r[4][k] += jy * r[6][k];      // view-projection: x, y += offset * w
        r[7][k] += jx * r[10][k], r[8][k] += jy * r[10][k];    // projection
    }
    memcpy(g + 3 * 16, r[3], 2 * 16);
    memcpy(g + 7 * 16, r[7], 2 * 16);
    vk_usnap_patched(g, 32 * 16);
    jit_done = g, jit_frame = frame;
}

void vk_mv_draw(VkExec *ex) {
    const p_bind_buf *u = &ex->ubo[0][1];   // vertex stage, c4
    if (u->size != CAM_SIZE || !u->addr || !ex->depth.image) return;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (jitter_on && ex->depth.width == (uint32_t)((const float *)(uintptr_t)u->addr)[18 * 4]) jitter_apply(u, frame);
    if (__atomic_load_n(&pend.frame, __ATOMIC_RELAXED) == frame) return;
    VkMemSpan ms;
    if (!vk_mem_resolve(u->addr, 1, &ms) || ms.avail < 32 * 16) return;
    const float *f = (const float *)(uintptr_t)u->addr;
    if ((uint32_t)f[18 * 4] != ex->depth.width || (uint32_t)f[18 * 4 + 1] != ex->depth.height) return;
    plat_mutex_lock(&ml);
    if (pend.frame != frame) {
        memcpy(pend.r, f, sizeof pend.r);
        pend.depth = ex->depth.image, pend.dfmt = ex->depth.format;
        pend.w = ex->depth.width, pend.h = ex->depth.height;
        pend.frame = frame;
    }
    plat_mutex_unlock(&ml);
}

static int is_camera_cut(const Cam *cur, const Cam *prev) {
    if (!prev->frame || prev->frame + 1 != cur->frame) return 1;
    // Game writes prev view-projection into rows 24..27; any jump relative to last frame's 3..6 means a camera cut.
    float worst = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) worst = fmaxf(worst, fabsf(cur->r[24 + i][j] - prev->r[3 + i][j]));
    if (worst > 0.05f) return 1;
    // Also detect same-frame map teleports or FOV snaps where the game updated rows 24..27 together with the jump.
    float dx = cur->r[11][3] - prev->r[11][3], dy = cur->r[12][3] - prev->r[12][3], dz = cur->r[13][3] - prev->r[13][3];
    if (dx * dx + dy * dy + dz * dz > 80.0f * 80.0f) return 1;
    float fdot = cur->r[11][2] * prev->r[11][2] + cur->r[12][2] * prev->r[12][2] + cur->r[13][2] * prev->r[13][2];
    if (fdot < 0.7f) return 1;
    if (fabsf(cur->r[8][1] - prev->r[8][1]) > 0.25f * fabsf(prev->r[8][1])) return 1;
    return 0;
}

// The replay thread at frame's present: its capture becomes the one the present uses.
void vk_mv_frame(uint64_t frame) {
    if (vk_mv_armed < 0) read_cfg();
    if (!vk_mv_armed) return;
    plat_mutex_lock(&ml);
    if (pend.frame == frame) {
        static int checks;
        if (last.frame + 1 == frame && checks < 8) {   // is "previous" in this block the last frame's "current"?
            float worst = 0;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) worst = fmaxf(worst, fabsf(pend.r[24 + i][j] - last.r[3 + i][j]));
            rt_log("mv: frame %llu: previous view-projection differs from the last frame's by %g at most",
                   (unsigned long long)frame, worst);
            checks++;
        }
        pend.cut = is_camera_cut(&pend, &last);
        last = pend;
        ready = pend;
    } else {
        ready.frame = 0;   // no 3D scene this frame (a menu): the debug view shows the picture
    }
    plat_mutex_unlock(&ml);
}

static int init(VkCtx *c) {
    M.tried = 1;
    M.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[8];
    for (uint32_t i = 0; i < 8; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .descriptorType = i < 3 || i >= 6 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                : i < 5 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 8, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 112};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkSamplerCreateInfo sl = si;
    sl.magFilter = sl.minFilter = VK_FILTER_LINEAR;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof mv_debug_spv,
                                   .pCode = mv_debug_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!M.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &M.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &M.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &M.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &M.nearest) != VK_SUCCESS ||
        vkCreateSampler(c->device, &sl, NULL, &M.linear) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = M.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &M.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 3840ull * 2160 * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ba = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (shot_step && vmaCreateBuffer(c->vma, &bi, &ba, &M.shot_buf, &M.shot_a, NULL) != VK_SUCCESS) M.shot_buf = VK_NULL_HANDLE;
    bi.size = 256, bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vmaCreateBuffer(c->vma, &bi, &ba, &M.stat_buf, &M.stat_a, NULL) != VK_SUCCESS) return 0;
    return M.ok = 1;
}

static int make_image(VkCtx *c, VkFormat fmt, VkImageUsageFlags usage, VkImage *img, VmaAllocation *a, VkImageView *v) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {M.w, M.h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, a, NULL) != VK_SUCCESS) return *img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, v) == VK_SUCCESS;
}

static int ensure_images(VkCtx *c, uint32_t w, uint32_t h) {
    if (M.out && M.w == w && M.h == h) return 1;
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    if (M.out) {
        vkDestroyImageView(c->device, M.out_v, NULL), vmaDestroyImage(c->vma, M.out, M.out_a);
        for (int i = 0; i < 2; i++) vkDestroyImageView(c->device, M.hist_v[i], NULL), vmaDestroyImage(c->vma, M.hist[i], M.hist_a[i]);
    }
    M.w = w, M.h = h, M.have_hist = 0;
    const VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    int ok = make_image(c, VK_FORMAT_R8G8B8A8_UNORM, u, &M.out, &M.out_a, &M.out_v);
    for (int i = 0; i < 2; i++) ok &= make_image(c, VK_FORMAT_R16G16B16A16_SFLOAT, u, &M.hist[i], &M.hist_a[i], &M.hist_v[i]);
    if (!ok) M.out = VK_NULL_HANDLE;
    return ok;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags asp, VkImageLayout from) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {asp, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

// 4x4 inverse (Gauss-Jordan with partial pivoting); 0 when singular
static int invert(double a[4][4], double o[4][4]) {
    double m[4][8];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) m[i][j] = j < 4 ? a[i][j] : (j - 4 == i);
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++) if (fabs(m[r][c]) > fabs(m[p][c])) p = r;
        if (fabs(m[p][c]) < 1e-12) return 0;
        for (int j = 0; j < 8; j++) { double t = m[c][j]; m[c][j] = m[p][j]; m[p][j] = t; }
        double d = m[c][c];
        for (int j = 0; j < 8; j++) m[c][j] /= d;
        for (int r = 0; r < 4; r++)
            if (r != c) {
                double f = m[r][c];
                for (int j = 0; j < 8; j++) m[r][j] -= f * m[c][j];
            }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) o[i][j] = m[i][j + 4];
    return 1;
}

static void write_shot(void) {
    if (!M.shot_frame || !M.shot_buf) return;
    VkCtx *c = vk_ctx();
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    VmaAllocationInfo ai;
    vmaGetAllocationInfo(c->vma, M.shot_a, &ai);
    vmaInvalidateAllocation(c->vma, M.shot_a, 0, VK_WHOLE_SIZE);
    char path[512];
    snprintf(path, sizeof path, "%s/mv_%llu.png", shot_dir, (unsigned long long)M.shot_frame);
    plat_file_mkdir_p(shot_dir);
    rt_write_png(path, ai.pMappedData, M.w, M.h);
    VmaAllocationInfo si;
    vmaGetAllocationInfo(c->vma, M.stat_a, &si);
    vmaInvalidateAllocation(c->vma, M.stat_a, 0, VK_WHOLE_SIZE);
    const uint32_t *s = si.pMappedData;
    rt_log("mv: saved %s; scene pixels %u: mean error %.4f warped by the vectors, %.4f unwarped (%.0f%% less)", path,
           s[2], s[2] ? s[0] / 1000.0 / s[2] : 0, s[2] ? s[1] / 1000.0 / s[2] : 0,
           s[1] ? 100.0 * (1.0 - (double)s[0] / s[1]) : 0);
    if (s[2])
        rt_log("mv: colour check (mean luma): presented %.4f (view format %d); copy as stored %.4f, sRGB-encoded %.4f, "
               "sRGB-decoded %.4f", s[3] / 1000.0 / s[2], (int)M.last_fmt, s[4] / 1000.0 / s[2], s[5] / 1000.0 / s[2],
               s[6] / 1000.0 / s[2]);
    M.shot_frame = 0;
}

// ---- object motion: every triangle draw of the scene's depth pre-pass is drawn a second time right after the game's,
// with the same vertex shader patched to hand its clip positions to transform feedback (spirv_xfb) and rasterization
// off. An object is its mesh (index buffer, count, base vertex, first vertex buffer) and its occurrence in the frame;
// last frame's capture of the same object holds its previous positions, so whatever the vertex shader does (the
// object's matrix, its bones, MKW's curved world, the camera) is in the motion. The present (vel_pass) draws each
// object again from the two captures and keeps the motion where the object is the visible surface.
#define OBJ_MAX 2048
#define XFB_BUFS 4                 // frame f captures into f % 4: the present of f reads f and f - 1 while the
                                   // replay records f + 1 (the game runs at most a frame ahead)
#define XFB_VERTS (3u << 20)       // vec4 each: 48 MiB per buffer
// key: the mesh (low byte: its occurrence in the frame); t: the translation of the draw's object block (vertex c6,
// rows 0-2 of a 3x4 matrix) when it has one, to tell apart objects of one mesh (a kart's four wheels)
typedef struct { uint64_t key; uint32_t first, count; float t[3]; uint8_t has_t; } Cap;
static struct {
    int tried, ok;
    PFN_vkCmdBindTransformFeedbackBuffersEXT bind;
    PFN_vkCmdBeginTransformFeedbackEXT begin;
    PFN_vkCmdEndTransformFeedbackEXT end;
    VkBuffer buf[XFB_BUFS];
    VmaAllocation a[XFB_BUFS];
    Cap caps[XFB_BUFS][OBJ_MAX];
    int n[XFB_BUFS];
    uint32_t used[XFB_BUFS];
    uint64_t frame[XFB_BUFS];
    struct { const void *vs; VkPipelineLayout lay; uint64_t fmt; VkPipeline p; } pipes[1024];
    int npipes;
    uint64_t draws, captured, no_pipe;
} X;

VkShaderModule make_module(const uint32_t *w, uint64_t words);                    // pipe/vk_pipeline.c
void vk_pipeline_dyn_list(const VkDynamicState **d, uint32_t *n, int *pv_last);   // pipe/vk_pipeline.c

static uint64_t mix64(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

// The real Vulkan entry points behind vk_rec.h's macros, for the capture's direct calls after rec_drain
#undef vkCmdBindPipeline
#undef vkCmdSetRasterizerDiscardEnable
#undef vkCmdDrawIndexed
static void rc_real_bind_pipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) { vkCmdBindPipeline(cb, bp, p); }
static void rc_real_discard(VkCommandBuffer cb, VkBool32 on) { vkCmdSetRasterizerDiscardEnable(cb, on); }
static void rc_real_draw_indexed(VkCommandBuffer cb, uint32_t n, uint32_t inst, uint32_t first, int32_t base, uint32_t binst) {
    vkCmdDrawIndexed(cb, n, inst, first, base, binst);
}
#define vkCmdBindPipeline rc_vkCmdBindPipeline
#define vkCmdSetRasterizerDiscardEnable rc_vkCmdSetRasterizerDiscardEnable
#define vkCmdDrawIndexed rc_vkCmdDrawIndexed

// The vertex shader with gl_Position captured by transform feedback (buffer 0, 16-byte stride, offset 0): capability
// TransformFeedback, execution mode Xfb, and the decorations on the Position variable or on its member of the output
// block. NULL when the shader has no Position output.
static uint32_t *spirv_xfb(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    uint32_t pos_var = 0, pos_struct = 0, pos_member = 0, ep = 0;
    uint64_t ep_end = 0, anno_at = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (!wc || i + wc > n) return NULL;
        if (op == 15 && !ep) ep = w[i + 2], ep_end = i + wc;                              // OpEntryPoint
        if ((op == 71 || op == 72) && !anno_at) anno_at = i;                             // first decoration
        if (op == 71 && wc >= 4 && w[i + 2] == 11 && w[i + 3] == 0) pos_var = w[i + 1];  // BuiltIn Position
        if (op == 72 && wc >= 5 && w[i + 3] == 11 && w[i + 4] == 0) pos_struct = w[i + 1], pos_member = w[i + 2];
        i += wc;
    }
    if (pos_struct && !pos_var) {   // the variable of the output block: OpVariable of an Output pointer to it
        uint32_t ptr = 0;
        for (uint64_t i = 5; i < n; i += w[i] >> 16)
            if ((w[i] & 0xffff) == 32 && w[i + 2] == 3 && w[i + 3] == pos_struct) ptr = w[i + 1];
        for (uint64_t i = 5; ptr && i < n; i += w[i] >> 16)
            if ((w[i] & 0xffff) == 59 && w[i + 1] == ptr && w[i + 3] == 3) pos_var = w[i + 2];
    }
    if (!pos_var || !ep || !anno_at) return NULL;
    uint32_t *o = malloc((n + 32) * 4);
    if (!o) return NULL;
    uint64_t k = 0;
    for (uint64_t i = 0; i < n;) {
        if (i == 5) o[k++] = 2u << 16 | 17, o[k++] = 53;                                // OpCapability TransformFeedback
        if (i == anno_at) {
            o[k++] = 4u << 16 | 71, o[k++] = pos_var, o[k++] = 36, o[k++] = 0;           // XfbBuffer 0
            o[k++] = 4u << 16 | 71, o[k++] = pos_var, o[k++] = 37, o[k++] = 16;          // XfbStride 16
            if (pos_struct)
                o[k++] = 5u << 16 | 72, o[k++] = pos_struct, o[k++] = pos_member, o[k++] = 35, o[k++] = 0;
            else
                o[k++] = 4u << 16 | 71, o[k++] = pos_var, o[k++] = 35, o[k++] = 0;      // Offset 0
        }
        uint32_t wc = i < 5 ? 1 : w[i] >> 16;
        memcpy(o + k, w + i, wc * 4), k += wc, i += wc;
        if (i == ep_end) o[k++] = 3u << 16 | 16, o[k++] = ep, o[k++] = 11;               // OpExecutionMode Xfb
    }
    *out_n = k;
    return o;
}

static int xfb_init(VkCtx *c) {
    X.tried = 1;
    if (!c->has_transform_feedback) return rt_log("mv: no VK_EXT_transform_feedback: object motion off"), 0;
    X.bind = (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(c->device, "vkCmdBindTransformFeedbackBuffersEXT");
    X.begin = (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(c->device, "vkCmdBeginTransformFeedbackEXT");
    X.end = (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(c->device, "vkCmdEndTransformFeedbackEXT");
    if (!X.bind || !X.begin || !X.end) return 0;
    for (int i = 0; i < XFB_BUFS; i++) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)XFB_VERTS * 16,
                                 .usage = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(c->vma, &bi, &ai, &X.buf[i], &X.a[i], NULL) != VK_SUCCESS) return 0;
    }
    rt_log("mv: object motion on (transform feedback of the depth pre-pass)");
    return X.ok = 1;
}

static VkPipeline capture_pipeline(VkExec *ex, const GfxShader *vs, VkPipelineLayout lay, int minus_w) {
    uint64_t fmt = mix64(mix64(mix64(mix64(0xcbf29ce484222325ull, ex->pass_ncolor), ex->pass_depth), ex->pass_stencil),
                         ex->pass_samples * 2 + (uint64_t)minus_w);
    for (uint32_t i = 0; i < ex->pass_ncolor; i++) fmt = mix64(fmt, ex->pass_color[i]);
    for (int i = 0; i < X.npipes; i++)
        if (X.pipes[i].vs == vs && X.pipes[i].lay == lay && X.pipes[i].fmt == fmt) return X.pipes[i].p;
    if (X.npipes == (int)(sizeof X.pipes / sizeof *X.pipes)) return VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    const VkDynamicState *dyn;
    uint32_t ndyn;
    int pv_last;
    vk_pipeline_dyn_list(&dyn, &ndyn, &pv_last);
    int dyn_vi = 0;
    for (uint32_t i = 0; i < ndyn; i++) dyn_vi |= dyn[i] == VK_DYNAMIC_STATE_VERTEX_INPUT_EXT;
    uint64_t vw = 0, pn = 0;
    const uint32_t *code = gfx_shader_spirv(vs, &vw);
    uint32_t *patched = dyn_vi && code ? spirv_xfb(code, vw, &pn) : NULL;
    VkShaderModule m = patched ? make_module(patched, pn) : VK_NULL_HANDLE;
    free(patched);
    if (m) {
        VkPipelineShaderStageCreateInfo st = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = m, .pName = "main"};
        VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineViewportDepthClipControlCreateInfoEXT clip = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, .negativeOneToOne = minus_w != 0};
        VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
        VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
            .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
        VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .pNext = pv_last ? &pvs : NULL, .rasterizerDiscardEnable = VK_TRUE,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
        VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = ex->pass_samples ? (VkSampleCountFlagBits)ex->pass_samples
                                                                                            : VK_SAMPLE_COUNT_1_BIT};
        VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                    .maxDepthBounds = 1.0f};
        VkPipelineColorBlendAttachmentState cba[8];
        memset(cba, 0, sizeof cba);
        VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                  .attachmentCount = ex->pass_ncolor, .pAttachments = cba};
        VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                               .dynamicStateCount = ndyn, .pDynamicStates = dyn};
        VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                            .colorAttachmentCount = ex->pass_ncolor, .pColorAttachmentFormats = ex->pass_color,
                                            .depthAttachmentFormat = ex->pass_depth, .stencilAttachmentFormat = ex->pass_stencil};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                           .stageCount = 1, .pStages = &st, .pVertexInputState = &vi,
                                           .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                           .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
                                           .pDynamicState = &dy, .layout = lay};
        if (vkCreateGraphicsPipelines(vk_ctx()->device, VK_NULL_HANDLE, 1, &pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk_ctx()->device, m, NULL);
    }
    X.pipes[X.npipes].vs = vs, X.pipes[X.npipes].lay = lay, X.pipes[X.npipes].fmt = fmt, X.pipes[X.npipes].p = pipe;
    X.npipes++;
    return pipe;
}

// After a game draw (vk_exec_ops.c op_draw_idx), the replay thread. 1: another pipeline was bound and rasterizer
// discard turned on; the caller binds the game's pipeline and its discard state again.
int vk_mv_capture(VkExec *ex, const void *vs, VkPipelineLayout lay, int minus_w, uint64_t ibuf, int32_t count,
                  int32_t base_vertex, int32_t instances, int32_t base_instance, int triangles) {
    if (!triangles || instances > 1 || !lay || !vs || !ex->pipe.ds.dw || !ex->depth.image || count <= 0) return 0;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (__atomic_load_n(&pend.frame, __ATOMIC_RELAXED) != frame || ex->depth.width != pend.w ||
        ex->depth.image != pend.depth)
        return 0;   // not the scene's depth pass
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&ml);
    if (!X.tried) xfb_init(c);
    int s = (int)(frame % XFB_BUFS);
    Cap *cp = NULL;
    VkPipeline p = VK_NULL_HANDLE;
    if (X.ok) {
        if (X.frame[s] != frame) X.frame[s] = frame, X.n[s] = 0, X.used[s] = 0;
        X.draws++;
        if (X.n[s] < OBJ_MAX && X.used[s] + (uint32_t)count <= XFB_VERTS) {
            uint64_t k = mix64(mix64(mix64(mix64(0xcbf29ce484222325ull, ibuf), (uint64_t)count),
                                     (uint64_t)(uint32_t)base_vertex),
                               ex->vbuf[0].address);
            int occ = 0;
            for (int i = 0; i < X.n[s]; i++) occ += (X.caps[s][i].key >> 8) == (k >> 8);
            k = (k & ~0xffull) | (uint64_t)(occ & 0xff);
            p = capture_pipeline(ex, (const GfxShader *)vs, lay, minus_w);
            if (!p) {
                X.no_pipe++;
            } else {
                cp = &X.caps[s][X.n[s]++];
                cp->key = k, cp->first = X.used[s], cp->count = (uint32_t)count;
                const p_bind_buf *u6 = &ex->ubo[0][3];
                VkMemSpan ms6;
                cp->has_t = u6->size == 0xc0 && u6->addr && vk_mem_resolve(u6->addr, 48, &ms6);
                if (cp->has_t) {
                    const float *m = (const float *)(uintptr_t)u6->addr;
                    cp->t[0] = m[3], cp->t[1] = m[7], cp->t[2] = m[11];
                }
                X.used[s] += (uint32_t)count;
                X.captured++;
            }
        }
        if (X.draws >= 600000) {
            rt_log("mv: object capture: %llu of %llu pre-pass draws captured, %llu without a capture pipeline, %d pipelines",
                   (unsigned long long)X.captured, (unsigned long long)X.draws, (unsigned long long)X.no_pipe, X.npipes);
            X.draws = X.captured = X.no_pipe = 0;
        }
    }
    plat_mutex_unlock(&ml);
    if (!cp) return 0;
    // The recorder thread (vk_rec.h) makes the replay's vkCmd* calls in order; the transform feedback commands have no
    // packets there, so the ring is emptied first and these calls are made here, all of them, in their order
    // (prototype: a drain per capture; rec_* wrappers for the three commands would keep the recorder running).
    rec_drain();
    (rc_real_bind_pipeline)(ex->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    (rc_real_discard)(ex->cb, VK_TRUE);
    VkDeviceSize off = (VkDeviceSize)cp->first * 16, size = (VkDeviceSize)count * 16;
    X.bind(ex->cb, 0, 1, &X.buf[s], &off, &size);
    X.begin(ex->cb, 0, 0, NULL, NULL);
    (rc_real_draw_indexed)(ex->cb, (uint32_t)count, 1, 0, base_vertex, (uint32_t)base_instance);
    X.end(ex->cb, 0, 0, NULL, NULL);
    ex->recorded += 6;
    return 1;
}

// ---- the velocity pass (present thread): each object of this frame with a capture last frame, drawn from both
static struct {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkImage img;
    VmaAllocation a;
    VkImageView view;
    uint32_t w, h;
    uint64_t objs, frames, reordered;
} V;

static int vel_init(VkCtx *c) {
    V.tried = 1;
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 24};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &V.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &V.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &V.layout) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mv = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof mv_vel_vs_spv,
                                   .pCode = mv_vel_vs_spv},
                             mf = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof mv_vel_fs_spv,
                                   .pCode = mv_vel_fs_spv};
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mv, NULL, &vm) != VK_SUCCESS || vkCreateShaderModule(c->device, &mf, NULL, &fm) != VK_SUCCESS)
        return 0;
    VkPipelineShaderStageCreateInfo st[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, .negativeOneToOne = VK_TRUE};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip,
                                            .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1, .pAttachments = &cba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkFormat f = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                        .colorAttachmentCount = 1, .pColorAttachmentFormats = &f};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                       .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
                                       .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy,
                                       .layout = V.layout};
    VkResult r = vkCreateGraphicsPipelines(c->device, VK_NULL_HANDLE, 1, &pi, NULL, &V.pipe);
    vkDestroyShaderModule(c->device, vm, NULL), vkDestroyShaderModule(c->device, fm, NULL);
    return V.ok = r == VK_SUCCESS;
}

static int vel_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (V.img && V.w == w && V.h == h) return 1;
    if (V.img) vkDestroyImageView(c->device, V.view, NULL), vmaDestroyImage(c->vma, V.img, V.a), V.img = VK_NULL_HANDLE;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &V.img, &V.a, NULL) != VK_SUCCESS) return V.img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = V.img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &V.view) != VK_SUCCESS) return 0;
    V.w = w, V.h = h;
    return 1;
}

// Records the velocity image of frame (cleared: no object motion when there is nothing to draw); its view, or
// VK_NULL_HANDLE when it cannot be made. depth_view: the scene depth, already in GENERAL and readable.
static VkImageView vel_pass(VkCommandBuffer cb, uint64_t frame, VkImageView depth_view, uint32_t w, uint32_t h, float near_z,
                            float far_z) {
    VkCtx *c = vk_ctx();
    if (!V.tried) vel_init(c);
    if (!V.ok || !vel_image(c, w, h)) return VK_NULL_HANDLE;
    barrier(cb, V.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = V.view,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    int s = (int)(frame % XFB_BUFS), q = (int)((frame + XFB_BUFS - 1) % XFB_BUFS), drawn = 0;
    if (X.ok && X.frame[s] == frame && X.frame[q] == frame - 1 && X.n[s] && X.n[q]) {
        VkViewport vp = {0, (float)h, (float)w, -(float)h, 0, 1};   // the replay's y-flipped viewport
        VkRect2D sc = {{0, 0}, {w, h}};
        vkCmdSetViewport(cb, 0, 1, &vp), vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.pipe);
        VkDescriptorBufferInfo bc = {X.buf[s], 0, VK_WHOLE_SIZE}, bp = {X.buf[q], 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo di = {M.nearest, depth_view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wr[3] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bc},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bp},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &di}};
        rec_push(M.push, cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.layout, 0, 3, wr);
        const Cap *pv = X.caps[q];
        int np = X.n[q];
        static uint8_t taken[OBJ_MAX];
        memset(taken, 0, (size_t)np);
        // Objects of one mesh are told apart by where they are, not by their order in the frame: the game sorts its
        // draws, so a kart's wheels swap places in the list as the camera turns, and pairing by order gave a wheel
        // another wheel's previous position (a smear two or three wheels long behind the kart).
        for (int i = 0; i < X.n[s]; i++) {
            const Cap *cc = &X.caps[s][i];
            int j = -1, same = -1;
            float best = 3.4e38f;
            for (int u = 0; u < np; u++) {
                if (taken[u] || (pv[u].key >> 8) != (cc->key >> 8) || pv[u].count != cc->count) continue;
                if (pv[u].key == cc->key) same = u;
                if (!cc->has_t || !pv[u].has_t) continue;
                float dx = pv[u].t[0] - cc->t[0], dy = pv[u].t[1] - cc->t[1], dz = pv[u].t[2] - cc->t[2];
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best) best = d2, j = u;
            }
            if (j < 0) j = same;
            if (j < 0) continue;
            if (same >= 0 && j != same) V.reordered++;
            taken[j] = 1;
            struct { uint32_t bc, bp; float size[2], nf[2]; } pc = {cc->first, pv[j].first, {(float)w, (float)h}, {near_z, far_z}};
            vkCmdPushConstants(cb, V.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
            vkCmdDraw(cb, cc->count, 1, 0, 0);
            drawn++;
        }
    }
    vkCmdEndRendering(cb);
    barrier(cb, V.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    V.objs += (uint64_t)drawn;
    if (++V.frames == 600) {
        rt_log("mv: velocity pass: %.1f objects a frame drawn from both captures, %.2f a frame paired out of draw order",
               V.objs / 600.0, V.reordered / 600.0);
        V.objs = V.frames = V.reordered = 0;
    }
    return V.view;
}

// ---- the picture without the HUD (DLSS's input; frame generation's "hudless" colour). MKW draws the finished 3D
// picture into a float image with the scene's depth attached (a dump of frame 2400: pass 275, r11g11b10f), renders the
// whole HUD (item boxes, minimap, positions) into an RGBA8 layer of its own (passes 276-277), then merges the two in
// one colour-only pass (278); the title screen's logo comes after (279). So the last render-size float target drawn
// with the scene's depth is the picture without the HUD: it is copied when the first colour-only pass at the render
// size begins. Frame f's copy goes to f % 3 (the present of f reads it while the replay records f + 1).
#define HUD_SLOTS 3
static struct {
    VkImage img[HUD_SLOTS];
    VmaAllocation a[HUD_SLOTS];
    VkImageView v[HUD_SLOTS];
    uint64_t frame[HUD_SLOTS];
    uint32_t w, h;
    VkFormat f;
    VkExecTarget scene;    // the frame's last float target drawn with the scene depth (tonemap+bloom)
    VkImage smaa_target;   // SMAA blend target (r11g11b10f, no depth, after H.scene): UI later blends onto this same image
    uint64_t scene_frame, smaa_frame, done_frame, copies;
} H;

static int hud_images(VkCtx *c, uint32_t w, uint32_t h, VkFormat f) {
    if (H.img[0] && H.w == w && H.h == h && H.f == f) return 1;
    for (int i = 0; i < HUD_SLOTS; i++) {
        if (!H.img[i]) continue;
        vkDestroyImageView(c->device, H.v[i], NULL), vmaDestroyImage(c->vma, H.img[i], H.a[i]);
        H.img[i] = VK_NULL_HANDLE, H.frame[i] = 0;
    }
    for (int i = 0; i < HUD_SLOTS; i++) {
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &H.img[i], &H.a[i], NULL) != VK_SUCCESS) return H.img[i] = VK_NULL_HANDLE, 0;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = H.img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &H.v[i]) != VK_SUCCESS) return 0;
    }
    H.w = w, H.h = h, H.f = f;
    rt_log("mv: picture without the HUD: %ux%u, format %d", w, h, (int)f);
    return 1;
}

static void hud_copy(VkExec *ex, const VkExecTarget *src, VkImageLayout src_layout, uint64_t frame, int after_pass) {
    if (!src->image || !hud_images(vk_ctx(), src->width, src->height, src->format)) return;
    int s = (int)(frame % HUD_SLOTS);
    VkImageMemoryBarrier2 bs[2];
    uint32_t nb = 0;
    if (after_pass) {
        bs[nb++] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                           .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                           .oldLayout = src_layout, .newLayout = src_layout,
                                           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                           .image = src->image,
                                           .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, src->level, 1, src->base_layer, 1}};
    }
    bs[nb++] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = H.img[s], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = nb, .pImageMemoryBarriers = bs};
    vkCmdPipelineBarrier2(ex->cb, &d);
    VkImageCopy r = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, src->level, src->base_layer, 1},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {src->width, src->height, 1}};
    vkCmdCopyImage(ex->cb, src->image, src_layout, H.img[s], VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    if (after_pass) {
        VkImageMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                      .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
                                      .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .image = H.img[s], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo pd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &post};
        vkCmdPipelineBarrier2(ex->cb, &pd);
        ex->recorded += 3;
    } else {
        ex->recorded += 2;
        vk_exec_barrier(ex);
    }
    H.frame[s] = frame;
    if (++H.copies == 1) rt_log("mv: picture without the HUD taken at frame %llu", (unsigned long long)frame);
}

// The replay thread, in vk_exec_pass_begin after its barrier and before the pass begins.
void vk_mv_pass_begin(VkExec *ex) {
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    const VkExecTarget *t = &ex->color[0];
    int render_size = ex->pass_ncolor >= 1 && t->image && !t->is3d && t->width == pend.w && t->height == pend.h;
    VkExecTarget src = {0};
    plat_mutex_lock(&ml);
    if (pend.frame == frame && H.done_frame != frame) {
        if (render_size) {
            if (ex->depth.image == pend.depth && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
                H.scene = *t, H.scene_frame = frame;
            else if (!ex->depth.image && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && H.scene_frame == frame) {
                // Skip mid-frame rg8/rgba8 passes; catch the post-tonemap SMAA blend target when its pass finishes,
                // or snapshot H.scene immediately if SMAA was skipped and UI blends straight onto H.scene.
                if (t->image != H.scene.image) H.smaa_target = t->image, H.smaa_frame = frame;
                else src = H.scene, H.done_frame = frame;
            }
        } else if (ex->pass_ncolor >= 1 && t->image && !t->is3d && !ex->depth.image &&
                   t->format == VK_FORMAT_R8G8B8A8_SRGB && H.scene_frame == frame) {
            // Fallback if both SMAA and UI passes were skipped before the final sRGB copy.
            src = H.scene, H.done_frame = frame;
        }
    }
    plat_mutex_unlock(&ml);
    hud_copy(ex, &src, VK_IMAGE_LAYOUT_GENERAL, frame, 0);
}

// The replay thread, in vk_exec_pass_end after vkCmdEndRendering when the pass actually completes (!split_ending).
void vk_mv_pass_end(VkExec *ex) {
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    const VkExecTarget *t = &ex->color[0];
    VkExecTarget src = {0};
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_GENERAL;
    plat_mutex_lock(&ml);
    if (pend.frame == frame && ex->pass_ncolor >= 1 && t->image && !t->is3d && !ex->depth.image &&
        t->width == pend.w && t->height == pend.h && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
        H.smaa_frame == frame && t->image == H.smaa_target && H.done_frame != frame) {
        src = *t, H.done_frame = frame;
        if (ex->fbl_mask & 1u) src_layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
    }
    plat_mutex_unlock(&ml);
    hud_copy(ex, &src, src_layout, frame, 1);
}

// THWOMP DLSS (vk_dlss.c): this frame's and last frame's finished pictures (render size, sRGB views, GENERAL), which the
// bias hint compares in place of the pictures without the HUD: the HUD is in DLSS's input, and where it stays on the
// screen while the scene moves behind it, the motion does not describe it either. Set before vk_mv_frame_inputs.
static VkImageView pic_cur, pic_prev;
void vk_mv_set_pictures(VkImageView cur, VkImageView prev) { pic_cur = cur, pic_prev = prev; }

static VkImageView hud_view(uint64_t frame) {
    int s = (int)(frame % HUD_SLOTS);
    return H.img[s] && H.frame[s] == frame ? H.v[s] : VK_NULL_HANDLE;
}

VkImage vk_mv_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]) {
    if (mv_debug <= 0) return VK_NULL_HANDLE;
    VkCtx *c = vk_ctx();
    if (!M.tried) init(c);
    write_shot();
    Cam cam;
    plat_mutex_lock(&ml);
    cam = ready;
    plat_mutex_unlock(&ml);
    if (!M.ok || !cam.frame || !ensure_images(c, (uint32_t)crop[2], (uint32_t)crop[3])) return VK_NULL_HANDLE;
    slot %= SLOTS;
    if (M.src_view[slot]) vkDestroyImageView(c->device, M.src_view[slot], NULL);
    if (M.depth_view[slot]) vkDestroyImageView(c->device, M.depth_view[slot], NULL);
    M.src_view[slot] = M.depth_view[slot] = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &M.src_view[slot]) != VK_SUCCESS) { M.src_view[slot] = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
    vi.image = cam.depth, vi.format = cam.dfmt, vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vkCreateImageView(c->device, &vi, NULL, &M.depth_view[slot]) != VK_SUCCESS) { M.depth_view[slot] = VK_NULL_HANDLE; return VK_NULL_HANDLE; }

    // M = previous view-projection x inverse(current view-projection); rows as the block stores them
    double cur[4][4], prev[4][4], inv[4][4], mm[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) cur[i][j] = cam.r[3 + i][j], prev[i][j] = cam.r[24 + i][j];
    if (!invert(cur, inv)) return VK_NULL_HANDLE;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            mm[i][j] = 0;
            for (int k = 0; k < 4; k++) mm[i][j] += prev[i][k] * inv[k][j];
        }
    struct { float m[16]; int32_t offset[2], size[2]; float opt[4], opt2[4]; } pc = {0};
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) pc.m[col * 4 + row] = (float)mm[row][col];
    pc.offset[0] = crop[0], pc.offset[1] = crop[1], pc.size[0] = crop[2], pc.size[1] = crop[3];
    pc.opt[0] = (float)mv_debug, pc.opt[1] = (float)flipy, pc.opt[2] = (float)zmode;
    pc.opt2[0] = fmt == VK_FORMAT_R8G8B8A8_SRGB || fmt == VK_FORMAT_B8G8R8A8_SRGB;
    M.last_fmt = fmt;
    pc.opt2[1] = (float)(cam.frame % 2 == 0);
    {   // the camera vectors come from unjittered matrices; the pictures moved by the jitter's change too
        float cx, cy, px, py;
        vk_mv_jitter_px(cam.frame, &cx, &cy), vk_mv_jitter_px(cam.frame - 1, &px, &py);
        pc.opt2[2] = cx - px, pc.opt2[3] = cy - py;
    }

    VkImageAspectFlags dasp = cam.dfmt == VK_FORMAT_D32_SFLOAT_S8_UINT || cam.dfmt == VK_FORMAT_D24_UNORM_S8_UINT
                                  ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    int rd = M.cur, wr = M.cur ^ 1;
    barrier(cb, src, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, cam.depth, dasp, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, M.out, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    barrier(cb, M.hist[rd], VK_IMAGE_ASPECT_COLOR_BIT, M.have_hist ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED);
    barrier(cb, M.hist[wr], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    int shoot = M.shot_buf && cam.frame >= shot_lo && cam.frame <= shot_hi && (cam.frame - shot_lo) % shot_step == 0;
    if (shoot) vkCmdFillBuffer(cb, M.stat_buf, 0, 256, 0);
    pc.opt[3] = (float)shoot;
    VkMemoryBarrier2 fb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo fd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &fb};
    vkCmdPipelineBarrier2(cb, &fd);   // also makes the transform feedback captures readable
    VkImageView velv = vel_pass(cb, cam.frame, M.depth_view[slot], (uint32_t)crop[2], (uint32_t)crop[3], cam.r[14][0], cam.r[14][1]);
    if (!velv) return VK_NULL_HANDLE;
    VkDescriptorImageInfo vdi = {M.nearest, velv, VK_IMAGE_LAYOUT_GENERAL};
    VkImageView hv = hud_view(cam.frame);   // the picture without the HUD, or last frame's picture when there is none
    VkDescriptorImageInfo hdi = {M.nearest, hv ? hv : M.hist_v[rd], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo sb = {M.stat_buf, 0, 256};
    VkDescriptorImageInfo di[5] = {{M.nearest, M.depth_view[slot], VK_IMAGE_LAYOUT_GENERAL},
                                   {M.nearest, M.src_view[slot], VK_IMAGE_LAYOUT_GENERAL},
                                   {M.linear, M.hist_v[rd], VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, M.out_v, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, M.hist_v[wr], VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[8];
    w[7] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 7, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &hdi};
    w[6] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 6, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &vdi};
    w[5] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 5, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &sb};
    for (uint32_t i = 0; i < 5; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                      .descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .pImageInfo = &di[i]};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.pipe);
    rec_push(M.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.layout, 0, 8, w);
    vkCmdPushConstants(cb, M.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, ((uint32_t)crop[2] + 7) / 8, ((uint32_t)crop[3] + 7) / 8, 1);
    barrier(cb, M.out, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, M.hist[wr], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, src, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, cam.depth, dasp, VK_IMAGE_LAYOUT_GENERAL);
    M.cur = wr, M.have_hist = 1;
    if (shoot && (uint64_t)M.w * M.h * 4 <= 3840ull * 2160 * 4) {
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {M.w, M.h, 1}};
        vkCmdCopyImageToBuffer(cb, M.out, VK_IMAGE_LAYOUT_GENERAL, M.shot_buf, 1, &bc);
        barrier(cb, M.out, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        M.shot_frame = cam.frame;
    }
    return M.out;
}

// ---- the inputs DLSS takes (vk_dlss.cpp), recorded on the present's command buffer
void vk_mv_set_jitter(int on) { jitter_on = on; }

static struct {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkImage img;
    VmaAllocation a;
    VkImageView view;
    uint32_t w, h;
    uint64_t last_frame;
    VkImage bias;             // DLSS's bias-current-colour hint (R16F), the render size
    VmaAllocation bias_a;
    VkImageView bias_view;
    VkImage depth_img;        // the scene depth image depth_view views
    VkImageView depth_view;
} O;

static int out_init(VkCtx *c) {
    O.tried = 1;
    VkDescriptorSetLayoutBinding b[8];
    for (uint32_t i = 0; i < 8; i++)   // depth, velocity, motion vectors out, this and last picture, bias out, the scene's two
        b[i] = (VkDescriptorSetLayoutBinding){i, i == 2 || i == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                  : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 8, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 88};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &O.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &O.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &O.layout) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof mv_out_spv,
                                   .pCode = mv_out_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = O.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &O.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    return O.ok = r == VK_SUCCESS;
}

static int out_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (O.img && O.w == w && O.h == h) return 1;
    if (O.img) vkDestroyImageView(c->device, O.view, NULL), vmaDestroyImage(c->vma, O.img, O.a), O.img = VK_NULL_HANDLE;
    if (O.bias) vkDestroyImageView(c->device, O.bias_view, NULL), vmaDestroyImage(c->vma, O.bias, O.bias_a), O.bias = VK_NULL_HANDLE;
    {
        VkImageCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &bi, &bai, &O.bias, &O.bias_a, NULL) != VK_SUCCESS) return O.bias = VK_NULL_HANDLE, 0;
        VkImageViewCreateInfo bvi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = O.bias,
                                     .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16_SFLOAT,
                                     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &bvi, NULL, &O.bias_view) != VK_SUCCESS) return 0;
    }
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R16G16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &O.img, &O.a, NULL) != VK_SUCCESS) return O.img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = O.img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16G16_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &O.view) != VK_SUCCESS) return 0;
    O.w = w, O.h = h;
    return 1;
}

// The present thread, before DLSS: the motion vectors of the frame being presented (velocity pass, then mv_out.comp),
// and what the frame's captures give. 0 when this frame has no 3D scene, no copy without the HUD, or the render size
// is not the scene's (the caller then presents as before).
int vk_mv_frame_inputs(VkCommandBuffer cb, unsigned slot, const int crop[4], VkMvFrame *f) {
    VkCtx *c = vk_ctx();
    if (vk_mv_armed <= 0) return 0;
    if (!M.tried) init(c);
    if (!O.tried) out_init(c);
    Cam cam;
    plat_mutex_lock(&ml);
    cam = ready;
    plat_mutex_unlock(&ml);
    uint32_t w = (uint32_t)crop[2], h = (uint32_t)crop[3];
    VkImageView hv = cam.frame ? hud_view(cam.frame) : VK_NULL_HANDLE;
    if (!M.ok || !O.ok || !cam.frame || !hv || cam.w != w || cam.h != h || !out_image(c, w, h)) return 0;
    (void)slot;
    // One view of the scene depth for as long as the depth image lasts: DLSS reads it through a view handle
    // (VK_NVX_image_view_handle) and keeps what it learned of a view, so a view made and destroyed per frame (whose
    // handle a later view could reuse) is what it must not get. A new depth image (a resolution change) waits for the
    // GPU before the old view goes.
    if (O.depth_img != cam.depth || !O.depth_view) {
        if (O.depth_view) {
            vk_queue_lock();
            vkDeviceWaitIdle(c->device);
            vk_queue_unlock();
            vkDestroyImageView(c->device, O.depth_view, NULL);
            O.depth_view = VK_NULL_HANDLE;
        }
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = cam.depth,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = cam.dfmt,
                                    .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &O.depth_view) != VK_SUCCESS) return O.depth_view = VK_NULL_HANDLE, 0;
        O.depth_img = cam.depth;
        rt_log("mv: scene depth view for image %p", (void *)cam.depth);
    }
    double cur[4][4], prev[4][4], inv[4][4], mm[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) cur[i][j] = cam.r[3 + i][j], prev[i][j] = cam.r[24 + i][j];
    if (!invert(cur, inv)) return 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            mm[i][j] = 0;
            for (int k = 0; k < 4; k++) mm[i][j] += prev[i][k] * inv[k][j];
        }
    // everything written before (the replay's targets, the captures) is visible to what follows
    VkMemoryBarrier2 fb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo fd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &fb};
    vkCmdPipelineBarrier2(cb, &fd);
    VkImageView velv = vel_pass(cb, cam.frame, O.depth_view, w, h, cam.r[14][0], cam.r[14][1]);
    if (!velv) return 0;
    barrier(cb, O.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    barrier(cb, O.bias, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    // with DLSS's finished pictures (pic_cur): the scene's own (without the HUD) too, so that a change of the finished
    // picture the scene does not have is told apart: the HUD and the menus, which DLSS must take as they are now
    VkImageView scene_cur = hv, scene_prev = hud_view(cam.frame - 1);
    VkImageView prevv = pic_cur ? pic_prev : scene_prev;   // last frame's picture, when it was this scene's
    if (pic_cur) hv = pic_cur;
    float has_prev = prevv && O.last_frame + 1 == cam.frame && !cam.cut;
    float has_scene = pic_cur && scene_prev && has_prev;
    VkDescriptorImageInfo di[8] = {{M.nearest, O.depth_view, VK_IMAGE_LAYOUT_GENERAL},
                                   {M.nearest, velv, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, O.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {M.nearest, hv, VK_IMAGE_LAYOUT_GENERAL},
                                   {M.linear, prevv ? prevv : hv, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, O.bias_view, VK_IMAGE_LAYOUT_GENERAL},
                                   {M.nearest, scene_cur, VK_IMAGE_LAYOUT_GENERAL},
                                   {M.linear, scene_prev ? scene_prev : scene_cur, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[8];
    for (uint32_t i = 0; i < 8; i++)
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                       .descriptorType = i == 2 || i == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                          : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .pImageInfo = &di[i]};
    float jc[2], jp[2];
    vk_mv_jitter_px(cam.frame, &jc[0], &jc[1]), vk_mv_jitter_px(cam.frame - 1, &jp[0], &jp[1]);
    struct { float m[16], size[2], jd[2], has_prev, has_scene; } pc;
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) pc.m[col * 4 + row] = (float)mm[row][col];
    pc.size[0] = (float)w, pc.size[1] = (float)h, pc.jd[0] = jc[0] - jp[0], pc.jd[1] = jc[1] - jp[1];
    pc.has_prev = has_prev, pc.has_scene = has_scene;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, O.pipe);
    rec_push(M.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, O.layout, 0, 8, wr);
    vkCmdPushConstants(cb, O.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb, O.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, O.bias, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    int s = (int)(cam.frame % HUD_SLOTS);
    memset(f, 0, sizeof *f);
    f->frame = cam.frame;
    f->reset = O.last_frame + 1 != cam.frame || cam.cut;   // gap (menu/loading) or camera cut/warp
    O.last_frame = cam.frame;
    f->w = w, f->h = h;
    f->depth = cam.depth, f->depth_view = O.depth_view, f->depth_fmt = cam.dfmt;
    f->mv = O.img, f->mv_view = O.view;
    f->bias = O.bias, f->bias_view = O.bias_view;
    f->color = H.img[s], f->color_view = H.v[s], f->color_fmt = H.f;
    f->jitter[0] = jc[0], f->jitter[1] = jc[1];
    memcpy(f->cam, cam.r, sizeof f->cam);
    return 1;
}

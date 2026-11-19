// THWOMP: NVIDIA DLSS super resolution (MKW_UPSCALER=dlss, MKW_DLSS_MODEL=cnn|transformer|transformer2) through Streamline
// (sl_shim.cpp). The game renders at the lines MKW_DLSS_QUALITY gives (sdk/fs/romfs_res.c: Quality two thirds of the
// window's, Balanced 58 %, Performance half, Ultra Performance a third, DLAA all); on the present, in place of FSR 1 or
// the blit's scale, DLSS upscales the game's finished picture (tone-mapped, HUD included: MKW's picture without the HUD
// is the scene before its tone mapping, 16/11/2026) to the window's size, with the inputs vk_mv.c gathers: the scene
// depth, motion vectors (camera and every object's own vertex shader), the bias hint (where this picture disagrees with
// last frame's moved by the motion: shadows, the HUD over the moving scene) and the camera jitter it adds to the game's
// projection. dlss_compose.comp encodes DLSS's linear picture to sRGB for the present.
//
// Streamline starts before the Vulkan device (vk_dlss_pre_device: its extensions are added to ours), learns the device
// once it exists (vk_dlss_post_device), and takes the swapchain calls (vk_dlss_proc: gfx_present.c). It starts only
// when MKW_UPSCALER is dlss at start on an NVIDIA GPU with sl.interposer.dll beside mkw.exe; anything missing leaves
// the game as it was (FSR 1 or the blit). The captures vk_mv.c needs are read at start too, so choosing DLSS in the
// menu takes effect after a restart.
//
// THWOMP: NVIDIA DLSS frame generation (MKW_FRAMEGEN_TYPE=dlss at start, MKW_DLSSG=off|2..6 live): Streamline's
// sl.dlss_g, loaded at start with Streamline's Reflex and PCL (vk_reflex.c then goes through them), makes frames in the
// swapchain proxy's present from the window's picture and the depth and motion vectors DLSS tags for the frame. It needs
// DLSS super resolution on (its inputs are gathered here) and an RTX 40 or newer; 3x and more need an RTX 50 (the
// plugin's numFramesToGenerateMax). While the menu is open, or a frame has no 3D scene (no motion vectors), it is off for
// that frame; turned on or off for good the swapchain is made again (gfx_present.c) with the plugin loaded or not.
// It and the port's own frame generation (vk_framegen.c) never run together: vk_fg_wanted is off with it.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "vk.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "dlss_compose_spv.h"
#include "host.h"
#include "menu.h"
#include "nvn_api.h"   // gfx_counters
#include "sl_shim.h"

static struct {
    int wanted, started, ok, inited;
    VkDevice dev;
    unsigned seen;
    int model;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler linear, nearest;
    VkImage out, comp, hudless[2], ui[2], in[2];
    VmaAllocation out_a, comp_a, hudless_a[2], ui_a[2], in_a[2];
    VkImageView out_v, comp_v, hudless_v[2], ui_v[2], in_v[2];
    uint32_t w, h, fg_img_w, fg_img_h, fg_i, in_w, in_h, in_n;   // fg_*: ping-pong HUD-less/UI; in[in_n%2]: this frame
    uint64_t in_frame, sl_eval_frame;                      // the game frame in[(in_n - 1) % 2] / last SL eval holds
    VkImageView src_view[8];
    uint64_t frames, fails, smaller, fmt_said;
    VkImageView last_bias;   // vk_dlss_last_bias
    VkQueryPool qp;           // GPU time of the DLSS work (motion vectors, Streamline, compose): two stamps per slot
    double ms_sum;
    uint32_t ms_n, qp_used;   // qp_used: bit per slot with stamps written
    int have;                 // SL_SHIM_* that run on the device
    uint32_t sl_gfx_queues;   // Streamline's own queues, after ours in the graphics family
    uint32_t sl_compute_queues;
    uint32_t fg_max;          // DLSS-G: the most frames it makes per game frame (numFramesToGenerateMax), 0 unknown
    int fg_tagged;            // this frame's DLSS ran with frame generation's tags
    uint32_t fg_set;          // the frames last asked of the plugin (0 off), ~0u never
    uint32_t fg_in_w, fg_in_h, fg_out_w, fg_out_h;
    uint64_t fg_status_said, fg_presented;
} D = {.inited = -1, .seen = ~0u, .fg_set = ~0u};

int vk_dlss_wanted(void) { return !strcmp(cfg_value("MKW_UPSCALER"), "dlss"); }

static int dlss_start_shim(void) {
    if (D.started) return 1;
#ifdef _WIN32
    wchar_t dir[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    while (n && dir[n - 1] != L'\\' && dir[n - 1] != L'/') n--;
    if (!n) return 0;
    dir[n - 1] = 0;
#else
    wchar_t dir[2] = L"";
#endif
    int fg = !strcmp(cfg_value("MKW_FRAMEGEN_TYPE"), "dlss");   // read at start: its plugins load with Streamline
    if (!sl_shim_init(dir, fg)) return 0;
    D.started = 1;
    return 1;
}

// vk_device.c create_instance, before vkCreateInstance: start Streamline early so its temporary physical-device query
// never nests inside our open VkInstance (which trips Vulkan hook layers), and add Streamline's instance extensions.
void vk_dlss_pre_instance(const VkExtensionProperties *avail, uint32_t n_avail, const char **ext, uint32_t *n_ext,
                          uint32_t cap) {
    D.wanted = vk_dlss_wanted();
    if (!D.wanted || !dlss_start_shim()) return;
    const char *const *ie = NULL;
    uint32_t ni = 0;
    if (!sl_shim_instance_extensions(&ie, &ni)) return;
    for (uint32_t i = 0; i < ni; i++) {
        int sup = 0;
        for (uint32_t a = 0; a < n_avail && !sup; a++) sup = !strcmp(avail[a].extensionName, ie[i]);
        if (!sup) {
            rt_log("dlss: instance lacks extension %s asked by Streamline", ie[i]);
            continue;
        }
        int have = 0;
        for (uint32_t k = 0; k < *n_ext && !have; k++) have = !strcmp(ext[k], ie[i]);
        if (have) continue;
        if (*n_ext >= cap) {
            rt_log("dlss: no room for instance extension %s for Streamline", ie[i]);
            continue;
        }
        ext[(*n_ext)++] = ie[i];
        rt_log("dlss: instance extension %s for Streamline", ie[i]);
    }
}

// vk_device.c create_device, before vkCreateDevice: Streamline's device extensions join ext (at most cap);
// *queues: the queues it wants made in the graphics family after ours (DLSS frame generation's)
void vk_dlss_pre_device(uint32_t vendor, const VkExtensionProperties *avail, uint32_t n_avail, const char **ext,
                        uint32_t *n_ext, uint32_t cap, uint32_t *queues) {
    *queues = 0;
    D.sl_gfx_queues = 0;
    D.sl_compute_queues = 0;
    D.wanted = vk_dlss_wanted();
    if (!D.wanted) return;
    if (vendor != 0x10de) {
        rt_log("dlss: not an NVIDIA GPU: DLSS unavailable, the upscaler is FSR 1");
        vk_dlss_abort_pre_device();
        return;
    }
    if (!dlss_start_shim()) return;
    const char *const *de = NULL, *const *ie = NULL;
    uint32_t nd = 0, ni = 0, gq = 0, cq = 0;
    if (!sl_shim_requirements(avail, n_avail, &de, &nd, &ie, &ni, &gq, &cq)) {
        vk_dlss_abort_pre_device();
        return;
    }
    // Streamline's queues (DLSS-G presents on one of its own): graphics then compute, in our graphics family
    // (graphics queues do compute too) after our own queue
    if (gq + cq > 15) {
        rt_log("dlss: Streamline asks for %u graphics and %u compute queues: too many, DLSS unavailable", gq, cq);
        vk_dlss_abort_pre_device();
        return;
    }
    uint32_t orig_n_ext = *n_ext;
    if (gq || cq) rt_log("dlss: %u graphics and %u compute queues for Streamline", gq, cq);
    for (uint32_t i = 0; i < nd; i++) {
        // the EXT form of buffer device address may not be enabled with the KHR one (core since 1.2, already on)
        if (!strcmp(de[i], "VK_EXT_buffer_device_address")) continue;
        int have = 0;
        for (uint32_t k = 0; k < *n_ext; k++) have |= !strcmp(ext[k], de[i]);
        if (have) continue;
        if (*n_ext == cap) {
            rt_log("dlss: no room for device extension %s: DLSS unavailable", de[i]);
            *n_ext = orig_n_ext;
            vk_dlss_abort_pre_device();
            return;
        }
        ext[(*n_ext)++] = de[i];
        rt_log("dlss: device extension %s for Streamline", de[i]);
    }
    D.sl_gfx_queues = gq;
    D.sl_compute_queues = cq;
    *queues = gq + cq;
}

void vk_dlss_abort_pre_device(void) {
    if (!D.started) return;
    D.started = 0;
    D.sl_gfx_queues = 0;
    D.sl_compute_queues = 0;
    sl_shim_shutdown();
}

void vk_dlss_shutdown(VkCtx *c) {
    // Streamline must shut down while VkDevice and VkInstance are still alive.
    sl_shim_shutdown();
    if (c && c->device) {
        for (int i = 0; i < 8; i++)
            if (D.src_view[i]) vkDestroyImageView(c->device, D.src_view[i], NULL);
        if (D.out_v) vkDestroyImageView(c->device, D.out_v, NULL);
        if (D.out && c->vma) vmaDestroyImage(c->vma, D.out, D.out_a);
        if (D.comp_v) vkDestroyImageView(c->device, D.comp_v, NULL);
        if (D.comp && c->vma) vmaDestroyImage(c->vma, D.comp, D.comp_a);
        for (int i = 0; i < 2; i++) {
            if (D.hudless_v[i]) vkDestroyImageView(c->device, D.hudless_v[i], NULL);
            if (D.hudless[i] && c->vma) vmaDestroyImage(c->vma, D.hudless[i], D.hudless_a[i]);
            if (D.ui_v[i]) vkDestroyImageView(c->device, D.ui_v[i], NULL);
            if (D.ui[i] && c->vma) vmaDestroyImage(c->vma, D.ui[i], D.ui_a[i]);
            if (D.in_v[i]) vkDestroyImageView(c->device, D.in_v[i], NULL);
            if (D.in[i] && c->vma) vmaDestroyImage(c->vma, D.in[i], D.in_a[i]);
        }
        if (D.qp) vkDestroyQueryPool(c->device, D.qp, NULL);
        if (D.pipe) vkDestroyPipeline(c->device, D.pipe, NULL);
        if (D.layout) vkDestroyPipelineLayout(c->device, D.layout, NULL);
        if (D.dsl) vkDestroyDescriptorSetLayout(c->device, D.dsl, NULL);
        if (D.linear) vkDestroySampler(c->device, D.linear, NULL);
        if (D.nearest) vkDestroySampler(c->device, D.nearest, NULL);
    }
    memset(&D, 0, sizeof D);
    D.inited = -1;
    D.seen = ~0u;
    D.fg_set = ~0u;
}

// vk_device.c, right after vkCreateDevice (vk_ctx() is not set yet)
void vk_dlss_post_device(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t gfx_family,
                         uint32_t gfx_queue_count) {
    if (!D.started) return;
    D.dev = dev;
    uint32_t gfx_idx = (D.sl_gfx_queues > 0 && 1 < gfx_queue_count) ? 1 : 0;
    uint32_t cmp_idx = (D.sl_compute_queues > 0 && 1 + D.sl_gfx_queues < gfx_queue_count) ? (1 + D.sl_gfx_queues) : 0;
    if ((D.sl_gfx_queues || D.sl_compute_queues) && 1 + D.sl_gfx_queues + D.sl_compute_queues > gfx_queue_count)
        rt_log("dlss: graphics family has %u queue(s); Streamline sharing queue 0", gfx_queue_count);
    D.have = sl_shim_set_vulkan(inst, phys, dev, gfx_family, gfx_idx, gfx_family, cmp_idx);
    D.ok = (D.have & SL_SHIM_DLSS) != 0;
    if (D.have & SL_SHIM_FG) {
        uint32_t mx = 0;
        if (sl_shim_fg_state(&mx, NULL, NULL, NULL)) D.fg_max = mx;
        rt_log("dlss: frame generation up to %ux on this GPU", D.fg_max + 1);
    }
}

// ---- DLSS frame generation

int vk_dlssg_loaded(void) { return (D.have & SL_SHIM_FG) != 0; }       // its plugin runs on this device
int vk_dlssg_reflex(void) { return (D.have & SL_SHIM_REFLEX) != 0; }   // Streamline's Reflex replaces the driver's
uint32_t vk_dlssg_max(void) { return D.fg_max ? D.fg_max + 1 : 0; }    // the highest multiplier, 0 unknown

// The frames per game frame the settings ask for and this GPU makes (multiplier - 1), 0 off. Needs DLSS super
// resolution running (its depth and motion vectors). Honours MKW_DLSSG (2/3/4/5/6); only clamps to the GPU's
// reported max. A soft log warns when the ask exceeds ~monitor_Hz/60, but does not hard-lock to 2x.
uint32_t vk_dlssg_frames(void) {
    if (!(D.have & SL_SHIM_FG) || !vk_dlss_active() || strcmp(cfg_value("MKW_FRAMEGEN_TYPE"), "dlss")) return 0;
    int m = atoi(cfg_value("MKW_DLSSG"));   // "off": 0
    if (m < 2) return 0;
    int asked = m;
    uint32_t n = (uint32_t)m - 1;
    if (D.fg_max && n > D.fg_max) n = D.fg_max;
    double hz = vk_present_refresh_hz();
    if (hz > 0.0) {
        int fit = (int)(hz / 60.0 + 0.25);
        if (fit < 2) fit = 2;
        if ((int)(n + 1) > fit) {
            static int warned_for = -1;
            if (warned_for != asked)
                rt_log("dlss: frame generation %ux may exceed monitor %.0f Hz at ~60 game fps (settings %ux; not capped)",
                       n + 1, hz, asked),
                    warned_for = asked;
        }
    }
    if (asked != (int)(n + 1)) {
        static int said_for = -1;
        if (said_for != asked) {
            said_for = asked;
            rt_log("dlss: frame generation using %ux (settings %ux; GPU max %ux)", n + 1, asked,
                   D.fg_max ? D.fg_max + 1 : 0);
        }
    }
    return n;
}

// gfx_present.c, before a swapchain is made: DLSS-G loaded (its proxy swapchain) only while it makes frames
void vk_dlssg_swapchain(int on) {
    sl_shim_fg_loaded(on);
    D.fg_set = ~0u;
}

static volatile uint32_t g_fg_overlay_pres;
uint32_t vk_dlssg_take_presented(void) { return __atomic_exchange_n(&g_fg_overlay_pres, 0, __ATOMIC_ACQ_REL); }

// gfx_present.c, after recording a game frame's present and before presenting it: the plugin's options for that
// present, off when this frame's DLSS did not run with frame generation's tags, the swapchain is HDR10 or the menu is
// open (docs/ProgrammingGuideDLSS_G.md 6.4)
void vk_dlssg_present(uint32_t frames, int hdr) {
    if (!(D.have & SL_SHIM_FG)) return;
    // Hold FG off for the whole cup opening + post-end settle (ov_cup_movie_guard): retest of 8e90582 hung while the
    // 2560 decoder was still live, so end-only settle never armed. 5x / THWOMP FG gate / compose path stay.
    uint32_t want = D.fg_tagged && !hdr && !menu_is_open() && !ov_cup_movie_guard() ? frames : 0;
    D.fg_tagged = 0;
    if (want != D.fg_set &&
        sl_shim_fg_set(want, D.fg_in_w, D.fg_in_h, D.fg_out_w, D.fg_out_h, (uint32_t)vk_present_image_count())) {
        if (want) {
            uint32_t drv = vk_dlssg_driver_mult();
            rt_log("dlss: frame generation %ux on", want + 1);
            if (drv && drv != want + 1) rt_log("dlss: the driver profile (NVIDIA App override) makes it %ux", drv);
        }
        else if (D.fg_set != ~0u && D.fg_set) rt_log("dlss: frame generation off for now");
        D.fg_set = want;
    }
    if (!want) {
        __atomic_store_n(&g_fg_overlay_pres, 0, __ATOMIC_RELEASE);
        return;
    }
    uint32_t mx = 0, st = 0, pres = 0, side = 0;
    if (!sl_shim_fg_state(&mx, &st, &pres, &side)) return;
    if (mx) D.fg_max = mx;
    D.fg_presented += pres;
    __atomic_add_fetch(&g_fg_overlay_pres, pres, __ATOMIC_RELEASE);
    static uint64_t n;
    if (++n % 600 == 0) {
        rt_log("dlss: frame generation %ux running, status %u (the plugin counted %llu presents over the last 600 game frames)", want + 1,
               st, (unsigned long long)D.fg_presented);
        D.fg_presented = 0;
    }
    if (st && D.fg_status_said++ < 4)
        rt_log("dlss: frame generation status %#x (1 resolution too low, 2 no Reflex, 4 HDR format, 8 constants)", st);
}

// The menu: the multiplier in effect now (0 none)
uint32_t vk_dlssg_running(void) { return D.fg_set == ~0u || !D.fg_set ? 0 : D.fg_set + 1; }
int vk_dlssg_preset_driver(int *own) { return sl_shim_fg_preset_read(own); }   // 0 default, 1 A, 2 B, 3 C, -1 unknown
// The NVIDIA App's fixed multiplier, which the driver applies over ours (nvngx_dlssg: "multi-frame count ... applied"),
// 0 none
uint32_t vk_dlssg_driver_mult(void) {
    int n = sl_shim_fg_frames_override();
    return n > 0 ? (uint32_t)n + 1 : 0;
}
int vk_dlssg_preset_write(int preset) { return sl_shim_fg_preset_write(preset); }

PFN_vkVoidFunction vk_dlss_proc(const char *name) {
    return D.ok ? sl_shim_device_proc(D.dev, name) : NULL;
}

// The window surface's calls through the interposer when DLSS frame generation runs (it finds its window there)
PFN_vkVoidFunction vk_dlss_instance_proc(VkInstance inst, const char *name) {
    return D.have & SL_SHIM_FG ? sl_shim_instance_proc(inst, name) : NULL;
}

void vk_wait_device_idle(VkDevice dev) {
    PFN_vkDeviceWaitIdle p = (PFN_vkDeviceWaitIdle)vk_dlss_proc("vkDeviceWaitIdle");
    (p ? p : vkDeviceWaitIdle)(dev);
}

int vk_dlss_active(void) { return D.ok && vk_dlss_wanted(); }
VkImageView vk_dlss_last_bias(void) { return D.last_bias; }
const char *vk_ctx_device_name(VkCtx *c) { return c ? c->props.deviceName : NULL; }   // the menu (C++)

static int init(VkCtx *c) {
    D.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[8];
    for (uint32_t i = 0; i < 8; i++)
        b[i] = (VkDescriptorSetLayoutBinding){i, (i == 2 || i == 6 || i == 7) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                              : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 8, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 40};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (!D.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &D.dsl) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &D.linear) != VK_SUCCESS)
        return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &D.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlss_compose_spv,
                                   .pCode = dlss_compose_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(c->device, &li, NULL, &D.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = D.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &D.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    return r == VK_SUCCESS;
}

static int make(VkCtx *c, VkFormat f, VkImageUsageFlags u, uint32_t w, uint32_t h, VkImage *img, VmaAllocation *a,
                VkImageView *v) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = u};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, a, NULL) != VK_SUCCESS) return *img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, v) == VK_SUCCESS;
}

#define OUT_USAGE (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
#define COMP_USAGE (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)

static int images(VkCtx *c, uint32_t w, uint32_t h) {
    if (D.out && D.w == w && D.h == h) return 1;
    vk_queue_lock();
    vk_wait_device_idle(c->device);
    vk_queue_unlock();
    if (D.out) vkDestroyImageView(c->device, D.out_v, NULL), vmaDestroyImage(c->vma, D.out, D.out_a);
    if (D.comp) vkDestroyImageView(c->device, D.comp_v, NULL), vmaDestroyImage(c->vma, D.comp, D.comp_a);
    for (int i = 0; i < 2; i++) {
        if (D.hudless[i])
            vkDestroyImageView(c->device, D.hudless_v[i], NULL), vmaDestroyImage(c->vma, D.hudless[i], D.hudless_a[i]);
        if (D.ui[i]) vkDestroyImageView(c->device, D.ui_v[i], NULL), vmaDestroyImage(c->vma, D.ui[i], D.ui_a[i]);
        D.hudless[i] = D.ui[i] = VK_NULL_HANDLE;
    }
    D.out = D.comp = VK_NULL_HANDLE;
    D.fg_img_w = D.fg_img_h = 0;
    if (!make(c, VK_FORMAT_R16G16B16A16_SFLOAT, OUT_USAGE, w, h, &D.out, &D.out_a, &D.out_v) ||
        !make(c, VK_FORMAT_R8G8B8A8_UNORM, COMP_USAGE, w, h, &D.comp, &D.comp_a, &D.comp_v))
        return 0;
    D.w = w, D.h = h;
    rt_log("dlss: output %ux%u", w, h);
    return 1;
}

// Two HUD-less/UI pairs so race FG can still read last frame's tags while we write this frame's (no present-thread stall).
static int fg_images(VkCtx *c, uint32_t w, uint32_t h) {
    if (D.hudless[0] && D.hudless[1] && D.ui[0] && D.ui[1] && D.fg_img_w == w && D.fg_img_h == h) return 1;
    if (D.hudless[0] || D.hudless[1] || D.ui[0] || D.ui[1]) {
        vk_queue_lock();
        vk_wait_device_idle(c->device);
        vk_queue_unlock();
    }
    for (int i = 0; i < 2; i++) {
        if (D.hudless[i])
            vkDestroyImageView(c->device, D.hudless_v[i], NULL), vmaDestroyImage(c->vma, D.hudless[i], D.hudless_a[i]);
        if (D.ui[i]) vkDestroyImageView(c->device, D.ui_v[i], NULL), vmaDestroyImage(c->vma, D.ui[i], D.ui_a[i]);
        D.hudless[i] = D.ui[i] = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; i++)
        if (!make(c, VK_FORMAT_R8G8B8A8_UNORM, COMP_USAGE, w, h, &D.hudless[i], &D.hudless_a[i], &D.hudless_v[i]) ||
            !make(c, VK_FORMAT_R8G8B8A8_UNORM, COMP_USAGE, w, h, &D.ui[i], &D.ui_a[i], &D.ui_v[i]))
            return 0;
    D.fg_img_w = w, D.fg_img_h = h, D.fg_i = 0;
    return 1;
}

// DLSS's colour: the game's finished picture (tone-mapped, sRGB-encoded) copied out of its larger present image into
// one of the render size, read through sRGB so DLSS sees linear light. One view for as long as the image lasts (DLSS
// keeps what it learned of a view handle).
#define IN_USAGE (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
static int input_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (D.in[0] && D.in[1] && D.in_w == w && D.in_h == h) return 1;
    if (D.in[0] || D.in[1]) {
        vk_queue_lock();
        vk_wait_device_idle(c->device);
        vk_queue_unlock();
    }
    for (int i = 0; i < 2; i++) {
        if (D.in[i]) vkDestroyImageView(c->device, D.in_v[i], NULL), vmaDestroyImage(c->vma, D.in[i], D.in_a[i]);
        D.in[i] = VK_NULL_HANDLE;
    }
    D.in_frame = 0;
    for (int i = 0; i < 2; i++)
        if (!make(c, VK_FORMAT_R8G8B8A8_SRGB, IN_USAGE, w, h, &D.in[i], &D.in_a[i], &D.in_v[i])) return 0;
    D.in_w = w, D.in_h = h;
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static int invert4(const double a[4][4], double o[4][4]) {
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

static void rows4(const float r[32][4], int at, double o[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) o[i][j] = r[at + i][j];
}

// Transpose column-vector matrix m into Streamline's row-major row-vector layout (v_row * M).
static void put_t(float *dst, const double m[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) dst[i * 4 + j] = (float)m[j][i];
}

// Convert an OpenGL [-1, 1] clip-Z column-vector matrix to Streamline's [0, 1] clip-Z (d = 0.5*z_ndc + 0.5).
static void to_dx_z(double m[4][4]) {
    for (int j = 0; j < 4; j++) m[2][j] = 0.5 * m[2][j] + 0.5 * m[3][j];
}

static void norm3(float v[3]) {
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-6f) v[0] /= l, v[1] /= l, v[2] /= l;
}

static void mul4(const double a[4][4], const double b[4][4], double o[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            o[i][j] = 0;
            for (int k = 0; k < 4; k++) o[i][j] += a[i][k] * b[k][j];
        }
}

// The present thread (gfx_present.c), in place of FSR or the blit's scale: the picture at out_w x out_h in RGBA8 (the
// game's encoding), HUD included, in GENERAL; VK_NULL_HANDLE when DLSS is off or this frame cannot use it.
VkImage vk_dlss_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4], int src_w,
                       int src_h, uint32_t out_w, uint32_t out_h, const int *bb_rect) {
    int on = vk_dlss_active();
    vk_mv_set_jitter(on);
    if (!on) return VK_NULL_HANDLE;
    VkCtx *c = vk_ctx();
    if (D.inited < 0) D.inited = init(c);
    if (!D.inited) return VK_NULL_HANDLE;
    // FG inputs wait is GPU-side on the present/DLSS submit (sl_shim_fg_inputs_take), not a CPU stall here.
    unsigned v = cfg_version();
    if (v != D.seen) {
        D.seen = v;
        const char *mod = cfg_value("MKW_DLSS_MODEL");
        D.model = !strcmp(mod, "cnn") ? 0 : !strcmp(mod, "transformer2") ? 2 : 1;
    }
    // DLSS's colour: the game's finished picture (tone-mapped, HUD included), its crop copied out of the larger present
    // image; the bias hint compares it with last frame's (vk_mv_set_pictures)
    if (fmt != VK_FORMAT_R8G8B8A8_UNORM && fmt != VK_FORMAT_R8G8B8A8_SRGB) {
        if (!D.fmt_said++) rt_log("dlss: the game's picture is format %d, not RGBA8: not upscaled by DLSS", (int)fmt);
        return VK_NULL_HANDLE;
    }
    uint32_t rw = (uint32_t)crop[2], rh = (uint32_t)crop[3];
    if (!input_image(c, rw, rh)) return VK_NULL_HANDLE;
    uint64_t gf = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    unsigned cur = D.in_n % 2;
    int has_prev = D.in_frame && D.in_frame + 1 == gf;
    {
        VkImageMemoryBarrier2 bs = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = src, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo dd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bs};
        vkCmdPipelineBarrier2(cb, &dd);
        barrier(cb, D.in[cur], VK_IMAGE_LAYOUT_UNDEFINED);
        VkImageCopy r = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffset = {crop[0], crop[1], 0},
                         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {rw, rh, 1}};
        vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, D.in[cur], VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        barrier(cb, D.in[cur], VK_IMAGE_LAYOUT_GENERAL);
    }
    vk_mv_set_pictures(D.in_v[cur], has_prev ? D.in_v[cur ^ 1] : VK_NULL_HANDLE);
    VkMvFrame f;
    int got = vk_mv_frame_inputs(cb, slot, crop, &f);
    vk_mv_set_pictures(VK_NULL_HANDLE, VK_NULL_HANDLE);
    D.in_n++, D.in_frame = gf;
    uint32_t want_fg = vk_dlssg_frames();
    if (!got) {
        if (want_fg > 0) sl_shim_fg_clear_tags(cb, (uint32_t)gf, out_w, out_h, bb_rect);
        return VK_NULL_HANDLE;
    }
    if (out_w < f.w || out_h < f.h) {   // DLSS only scales up: a window smaller than the picture keeps the usual path
        vk_mv_set_jitter(0);   // nothing would take the jitter back out
        if (want_fg > 0) sl_shim_fg_clear_tags(cb, (uint32_t)f.frame, out_w, out_h, bb_rect);
        if (!D.smaller++) rt_log("dlss: window %ux%u smaller than the picture %ux%u: not upscaled by DLSS", out_w, out_h, f.w, f.h);
        return VK_NULL_HANDLE;
    }
    if (!images(c, out_w, out_h)) return VK_NULL_HANDLE;
    unsigned qs = slot % 8;
    if (!D.qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                    .queryCount = 16};
        if (vkCreateQueryPool(c->device, &qi, NULL, &D.qp) != VK_SUCCESS) D.qp = VK_NULL_HANDLE;
    }
    if (D.qp && (D.qp_used >> qs & 1)) {   // this slot's last stamps: its submission has finished (the slot's fence)
        uint64_t t[2];
        if (vkGetQueryPoolResults(c->device, D.qp, qs * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
            t[1] > t[0]) {
            D.ms_sum += (double)(t[1] - t[0]) * c->props.limits.timestampPeriod * 1e-6, D.ms_n++;
            if (D.ms_n == 600) {
                rt_log("dlss: %.3f ms GPU a frame for motion vectors, DLSS and compose (%ux%u to %ux%u, mean of 600)",
                       D.ms_sum / 600, f.w, f.h, out_w, out_h);
                D.ms_sum = 0, D.ms_n = 0;
            }
        }
    }
    if (D.qp) {
        vkCmdResetQueryPool(cb, D.qp, qs * 2, 2);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, D.qp, qs * 2);
        D.qp_used |= 1u << qs;
    }
    slot %= 8;
    if (D.src_view[slot]) vkDestroyImageView(c->device, D.src_view[slot], NULL);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &D.src_view[slot]) != VK_SUCCESS) {
        D.src_view[slot] = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }

    SlShimFrame s;
    memset(&s, 0, sizeof s);
    // DLSS frame generation on this frame's present (vk_dlssg_present): the swapchain image's part with the picture
    s.fg = want_fg > 0;
    if (s.fg) {
        s.bb_rect[0] = bb_rect ? bb_rect[0] : 0, s.bb_rect[1] = bb_rect ? bb_rect[1] : 0;
        s.bb_rect[2] = bb_rect && bb_rect[2] > 0 ? bb_rect[2] : (int32_t)out_w;
        s.bb_rect[3] = bb_rect && bb_rect[3] > 0 ? bb_rect[3] : (int32_t)out_h;
        if (!fg_images(c, out_w, out_h)) s.fg = 0;
    }
    s.frame_index = (uint32_t)f.frame;
    s.ortho = fabsf(f.cam[10][3]) > 0.5f;
    // Ortho / 2D menus: drop DLSS history every frame so Multiplayer rows etc. do not smear in SR.
    // No HUD-less capture (common on character-select / some Free Roam panes): also reset — without it the
    // finished picture (UI included) accumulates trails that ParticleHint alone cannot clear.
    s.reset = f.reset || (D.sl_eval_frame + 1 != f.frame) || s.ortho || !f.color_view;
    // MKW_DLSS_ALWAYS_RESET=1 (diagnosis / stubborn menu smear): clear history every evaluate.
    static int always_reset = -1;
    if (always_reset < 0) always_reset = getenv("MKW_DLSS_ALWAYS_RESET") && atoi(getenv("MKW_DLSS_ALWAYS_RESET")) > 0;
    if (always_reset) s.reset = 1;
    // Prefer the player's quality setting; fall back to scale if the string is unknown.
    {
        const char *q = cfg_value("MKW_DLSS_QUALITY");
        float ratio = (float)out_w / (float)f.w;
        int by_scale = ratio < 1.2f ? 0 : ratio < 1.6f ? 1 : ratio < 1.85f ? 2 : ratio < 2.5f ? 3 : 4;
        s.mode = !strcmp(q, "dlaa")          ? 0
                 : !strcmp(q, "quality")     ? 1
                 : !strcmp(q, "balanced")    ? 2
                 : !strcmp(q, "performance") ? 3
                 : !strcmp(q, "ultra")       ? 4
                                             : by_scale;
    }
    s.model = D.model;
    s.in_w = f.w, s.in_h = f.h, s.out_w = out_w, s.out_h = out_h;
    s.depth = (SlShimImage){f.depth, f.depth_view, f.depth_fmt, f.w, f.h,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    s.mvec = (SlShimImage){f.mv, f.mv_view, VK_FORMAT_R16G16_SFLOAT, f.w, f.h,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    s.color_in = (SlShimImage){D.in[cur], D.in_v[cur], VK_FORMAT_R8G8B8A8_SRGB, f.w, f.h, IN_USAGE};
    s.color_out = (SlShimImage){D.out, D.out_v, VK_FORMAT_R16G16B16A16_SFLOAT, out_w, out_h, OUT_USAGE};
    s.bias = (SlShimImage){f.bias, f.bias_view, VK_FORMAT_R16_SFLOAT, f.w, f.h,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    // the camera block (vk_mv.c): rows 7-10 projection, 3-6 view-projection, 24-27 last frame's, 11-13 inverse view
    double p[4][4], ip[4][4], vp[4][4], ivp[4][4], pvp[4][4], c2p[4][4], p2c[4][4];
    rows4(f.cam, 7, p), rows4(f.cam, 3, vp), rows4(f.cam, 24, pvp);
    to_dx_z(p), to_dx_z(vp), to_dx_z(pvp);
    if (!invert4(p, ip) || !invert4(vp, ivp)) {
        if (s.fg) sl_shim_fg_clear_tags(cb, s.frame_index, out_w, out_h, bb_rect);
        return VK_NULL_HANDLE;
    }
    mul4(pvp, ivp, c2p);
    if (!invert4(c2p, p2c)) {
        if (s.fg) sl_shim_fg_clear_tags(cb, s.frame_index, out_w, out_h, bb_rect);
        return VK_NULL_HANDLE;
    }
    put_t(s.view_to_clip, p), put_t(s.clip_to_view, ip), put_t(s.clip_to_prev_clip, c2p), put_t(s.prev_clip_to_clip, p2c);
    for (int i = 0; i < 4; i++) s.identity[i * 5] = 1;
    s.jitter[0] = f.jitter[0], s.jitter[1] = f.jitter[1];
    for (int i = 0; i < 3; i++) {
        s.cam_right[i] = f.cam[11 + i][0], s.cam_up[i] = f.cam[11 + i][1], s.cam_fwd[i] = -f.cam[11 + i][2];
        s.cam_pos[i] = f.cam[11 + i][3];
    }
    norm3(s.cam_right), norm3(s.cam_up), norm3(s.cam_fwd);
    s.near_plane = f.cam[14][0] > 0.0f ? f.cam[14][0] : 1.0f;
    s.far_plane = f.cam[14][1] > s.near_plane ? f.cam[14][1] : 10000.0f;
    s.fov = fabsf(f.cam[8][1]) > 1e-6f ? 2.0f * atanf(1.0f / fabsf(f.cam[8][1])) : 1.0f;
    s.aspect = fabsf(f.cam[7][0]) > 1e-6f ? fabsf(f.cam[8][1] / f.cam[7][0]) : (float)f.w / (float)f.h;
    barrier(cb, D.out, VK_IMAGE_LAYOUT_UNDEFINED);
    static int skip = -1;   // MKW_DLSS_SKIPEVAL=1 (diagnosis): everything but DLSS itself, the copy shown in its place
    if (skip < 0) skip = getenv("MKW_DLSS_SKIPEVAL") && atoi(getenv("MKW_DLSS_SKIPEVAL")) > 0;
    if (!skip && !sl_shim_evaluate(cb, &s)) {
        if (s.fg) sl_shim_fg_clear_tags(cb, s.frame_index, out_w, out_h, bb_rect);
        if (D.fails++ < 4) rt_log("dlss: frame %llu not upscaled by DLSS", (unsigned long long)f.frame);
        return VK_NULL_HANDLE;
    }
    if (!skip) D.sl_eval_frame = f.frame;
    // FG must not run without HUD-less + UI tags: otherwise Streamline warps menus/HUD with scene MVs (smear).
    // Ortho / 2D menus: keep DLSS SR but pause FG for this present.
    unsigned fi = D.fg_i & 1;
    int write_fg = s.fg && !skip && !s.ortho && D.hudless[fi] && D.ui[fi] && f.color_view;
    if (s.fg && !skip && !write_fg) {
        sl_shim_fg_clear_tags(cb, s.frame_index, out_w, out_h, s.bb_rect);
        static uint64_t skip_n, skip_log;
        skip_n++;
        if (skip_log < 8 || (skip_n % 600) == 0)
            rt_log("dlss: frame generation skipped (ortho=%d hudless_src=%d, %llu skips so far)", s.ortho,
                   f.color_view != VK_NULL_HANDLE, (unsigned long long)skip_n), skip_log++;
    }
    // DLSS's picture -> the window's (sRGB-encoded); the game's own and the bias hint for MKW_DLSS_MASK
    barrier(cb, D.out, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, D.comp, VK_IMAGE_LAYOUT_UNDEFINED);
    if (write_fg) {
        barrier(cb, D.hudless[fi], VK_IMAGE_LAYOUT_UNDEFINED);
        barrier(cb, D.ui[fi], VK_IMAGE_LAYOUT_UNDEFINED);
    }
    VkDescriptorImageInfo di[8] = {
        {D.linear, skip ? D.in_v[cur] : D.out_v, VK_IMAGE_LAYOUT_GENERAL},
        {D.linear, D.src_view[slot], VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, D.comp_v, VK_IMAGE_LAYOUT_GENERAL},
        {D.linear, f.bias_view, VK_IMAGE_LAYOUT_GENERAL},
        {D.linear, f.color_view ? f.color_view : D.in_v[cur], VK_IMAGE_LAYOUT_GENERAL},
        {D.linear, D.in_v[cur], VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, write_fg ? D.hudless_v[fi] : D.comp_v, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, write_fg ? D.ui_v[fi] : D.comp_v, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[8];
    for (uint32_t i = 0; i < 8; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                      .descriptorType = (i == 2 || i == 6 || i == 7) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                                    : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &di[i]};
    static int mask = -1;   // MKW_DLSS_MASK=1 (diagnosis): DLSS's picture left, the game's (bilinear) right
    if (mask < 0) mask = getenv("MKW_DLSS_MASK") ? atoi(getenv("MKW_DLSS_MASK")) : 0;
    struct { float crop[4], tex[2], size[2], view, write_fg; } pc = {
        {(float)crop[0], (float)crop[1], (float)crop[2], (float)crop[3]},
        {(float)src_w, (float)src_h},
        {(float)out_w, (float)out_h},
        (float)mask,
        write_fg ? 1.0f : 0.0f};
    // Manual Streamline hooking: route CmdBindPipeline through the interposer after slEvaluateFeature so
    // sl.common's missing auto-hooks do not leave the command buffer in a bad state for DLSS-G.
    PFN_vkCmdBindPipeline bind_pipe = (PFN_vkCmdBindPipeline)vk_dlss_proc("vkCmdBindPipeline");
    if (!bind_pipe) bind_pipe = vkCmdBindPipeline;
    bind_pipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pipe);
    rec_push(D.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.layout, 0, 8, w);
    vkCmdPushConstants(cb, D.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (out_w + 7) / 8, (out_h + 7) / 8, 1);
    barrier(cb, D.comp, VK_IMAGE_LAYOUT_GENERAL);
    if (write_fg) {
        barrier(cb, D.hudless[fi], VK_IMAGE_LAYOUT_GENERAL);
        barrier(cb, D.ui[fi], VK_IMAGE_LAYOUT_GENERAL);
        s.hudless = (SlShimImage){D.hudless[fi], D.hudless_v[fi], VK_FORMAT_R8G8B8A8_UNORM, out_w, out_h, COMP_USAGE};
        s.ui = (SlShimImage){D.ui[fi], D.ui_v[fi], VK_FORMAT_R8G8B8A8_UNORM, out_w, out_h, COMP_USAGE};
        if (sl_shim_tag_fg(cb, &s))
            D.fg_tagged = 1, D.fg_in_w = f.w, D.fg_in_h = f.h, D.fg_out_w = out_w, D.fg_out_h = out_h,
            D.fg_i++;
    }
    if (D.qp) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, D.qp, qs * 2 + 1);
    D.last_bias = f.bias_view;
    if (++D.frames == 1) {
        static const char *modes[] = {"dlaa", "quality", "balanced", "performance", "ultra"};
        static const char *models[] = {"cnn", "transformer", "transformer2"};
        rt_log("dlss: first frame upscaled (%ux%u to %ux%u, quality %s, model %s)", f.w, f.h, out_w, out_h,
               modes[s.mode < 5 ? s.mode : 1], models[s.model < 3 ? s.model : 1]);
    } else {
        static int last_mode = -1, last_model = -1;
        if (s.mode != last_mode || s.model != last_model) {
            static const char *modes[] = {"dlaa", "quality", "balanced", "performance", "ultra"};
            static const char *models[] = {"cnn", "transformer", "transformer2"};
            rt_log("dlss: quality %s, model %s (%ux%u to %ux%u)", modes[s.mode < 5 ? s.mode : 1],
                   models[s.model < 3 ? s.model : 1], f.w, f.h, out_w, out_h);
            last_mode = s.mode, last_model = s.model;
        }
    }
    return D.comp;
}

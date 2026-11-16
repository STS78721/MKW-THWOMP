// THWOMP: NVIDIA Streamline as plain C functions (sl_shim.cpp), for vk_dlss.c and vk_reflex.c. 0 = failed / unavailable.
#pragma once
#include <stdint.h>
#include <wchar.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t w, h;
    uint32_t usage;   // VkImageUsageFlags the image was made with
} SlShimImage;        // in VK_IMAGE_LAYOUT_GENERAL when DLSS runs

typedef struct {
    uint32_t frame_index;
    int reset, ortho;
    int mode;    // 0 DLAA, 1 quality, 2 balanced, 3 performance, 4 ultra performance
    int model;   // 1 transformer (presets K), 2 transformer 2nd generation (M; L for ultra performance)
    uint32_t in_w, in_h, out_w, out_h;
    SlShimImage depth, mvec, color_in, color_out, bias, hudless, ui;   // bias: R16F; hudless+ui: RGBA8_UNORM out size
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16], identity[16];   // row-major
    float jitter[2];   // pixels
    float cam_pos[3], cam_up[3], cam_right[3], cam_fwd[3];
    float near_plane, far_plane, fov, aspect;
    int fg;               // DLSS frame generation runs on this frame's present: depth and motion vectors stay tagged
    int32_t bb_rect[4];   // with fg: where the picture lies in the swapchain image {x, y, w, h}
} SlShimFrame;

// Streamline's features: SL_DLSS always, SL_FG (DLSS frame generation with Streamline's Reflex and PCL) when asked
#define SL_SHIM_DLSS 1
#define SL_SHIM_FG 2
#define SL_SHIM_REFLEX 4

int sl_shim_init(const wchar_t *dir, int want_fg);   // dir: where sl.interposer.dll and the plugins are
// The device extensions and the extra queues every loaded feature asks for (graphics and compute queues: in the
// graphics family after the host's own queue)
int sl_shim_requirements(const char *const **dev_ext, uint32_t *n_dev, const char *const **inst_ext, uint32_t *n_inst,
                         uint32_t *gfx_queues, uint32_t *compute_queues);
// SL_SHIM_* bits of what runs on this device
int sl_shim_set_vulkan(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t gfx_family, uint32_t gfx_index,
                       uint32_t compute_family, uint32_t compute_index);
PFN_vkVoidFunction sl_shim_device_proc(VkDevice dev, const char *name);   // the interposer's proxies
PFN_vkVoidFunction sl_shim_instance_proc(VkInstance inst, const char *name);
int sl_shim_evaluate(VkCommandBuffer cb, const SlShimFrame *f);
int sl_shim_tag_fg(VkCommandBuffer cb, const SlShimFrame *f);   // tags post-compose HUD-less + UI buffers for DLSS-G
void sl_shim_fg_clear_tags(VkCommandBuffer cb, uint32_t frame_index, uint32_t out_w, uint32_t out_h,
                           const int32_t *bb_rect);
void sl_shim_shutdown(void);

// DLSS frame generation (sl.dlss_g). frames: generated per game frame (multiplier - 1), 0 off. Present thread.
int sl_shim_fg_set(uint32_t frames, uint32_t mvec_w, uint32_t mvec_h, uint32_t color_w, uint32_t color_h);
// The plugin's state: the most frames it can make (numFramesToGenerateMax), its status bits (DLSSGStatus), the frames
// presented since the last call and the smallest swapchain side it works at. 0: not loaded.
int sl_shim_fg_state(uint32_t *max_frames, uint32_t *status, uint32_t *presented, uint32_t *min_side);
void sl_shim_fg_loaded(int on);   // before a swapchain is made: on, the next one goes through DLSS-G
void sl_shim_fg_free(void);       // after turning it off for good: its memory back

// Streamline's Reflex and PCL markers (DLSS-G needs them, not the driver's own Reflex): mode 0 off, 1 on, 2 boost
int sl_shim_reflex_mode(int mode);
int sl_shim_reflex_sleep(uint32_t frame);
int sl_shim_pcl_marker(int marker, uint32_t frame);   // sl::PCLMarker
enum { SL_PCL_SIM_START = 0, SL_PCL_SIM_END = 1, SL_PCL_SUBMIT_START = 2, SL_PCL_SUBMIT_END = 3, SL_PCL_PRESENT_START = 4,
       SL_PCL_PRESENT_END = 5, SL_PCL_INPUT = 13 };

// The driver profile's DLSS frame generation preset (NVIDIA's key 0x10E41DF1, the NVIDIA App's "DLSS Override - Model
// Presets > Frame Generation"): 0 default (no override), 1 A, 2 B, 3 C. Read: the value that applies to this
// executable and whether it is its own profile's (1) or a global one (0); -1 unknown (no NVIDIA driver). Write: this
// executable's profile only (made when missing), 0 removes the override; 1 written.
int sl_shim_fg_preset_read(int *own);
int sl_shim_fg_preset_write(int preset);
// The NVIDIA App's fixed multiplier override in the driver profile: generated frames per game frame, 0 none
int sl_shim_fg_frames_override(void);

#ifdef __cplusplus
}
#endif

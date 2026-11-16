// THWOMP: the NVIDIA Streamline calls DLSS needs, as plain C functions (sl_shim.h) for vk_dlss.c and vk_reflex.c.
// Streamline's API is C++ (runtime/third_party/streamline, MIT); sl.interposer.dll is loaded at run time from beside
// mkw.exe, so the game starts without it (AMD and Intel GPUs, or the files missing) and only DLSS is then unavailable.
//
// Manual hooking (docs/ProgrammingGuideManualHooking.md of the SDK): slInit before the Vulkan device is made, the
// device extensions and queues the features ask for added to ours (sl_shim_requirements), slSetVulkanInfo once it
// exists, the swapchain calls through the interposer's proxies (sl_shim_device_proc). Streamline's over-the-air updates
// are off: only the DLLs shipped with the game are loaded, and nothing is downloaded.
//
// DLSS frame generation (sl.dlss_g, MKW_FRAMEGEN_TYPE=dlss): loaded with Streamline's Reflex and PCL plugins, which it
// requires (docs/ProgrammingGuideDLSS_G.md section 8: a Reflex outside Streamline does not count). It makes its frames in
// the swapchain proxy's present from the backbuffer and the depth and motion vectors DLSS super resolution tags.
//
// The frame generation preset has no Streamline option: nvngx_dlssg.dll reads the driver profile's key 0x10E41DF1
// (NVIDIA's NGX_DLSS_FG_OVERRIDE_RENDER_PRESET_SELECTION_ID, the NVIDIA App's "DLSS Override - Model Presets > Frame
// Generation") when frame generation starts. sl_shim_fg_preset_* read and write it in mkw.exe's own profile through
// NVAPI's driver settings (nvapi64.dll, installed with the driver; the functions by their published ids).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <string.h>

#include "sl.h"
#include "sl_dlss.h"
#include "sl_dlss_g.h"
#include "sl_helpers_vk.h"
#include "sl_pcl.h"
#include "sl_reflex.h"

#include "sl_shim.h"

extern "C" void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

namespace {
HMODULE g_mod;
bool g_init, g_want_fg;
int g_have;   // SL_SHIM_* that run
PFun_slInit *p_init;
PFun_slShutdown *p_shutdown;
PFun_slIsFeatureSupported *p_supported;
PFun_slIsFeatureLoaded *p_is_loaded;
PFun_slSetFeatureLoaded *p_set_loaded;
PFun_slGetFeatureRequirements *p_reqs;
PFun_slSetVulkanInfo *p_vkinfo;
PFun_slGetFeatureFunction *p_feature_fn;
PFun_slGetNewFrameToken *p_token;
PFun_slSetConstants *p_consts;
PFun_slSetTagForFrame *p_tag;
PFun_slEvaluateFeature *p_eval;
PFun_slFreeResources *p_free;
PFun_slDLSSGetOptimalSettings *p_optimal;
PFun_slDLSSSetOptions *p_options;
PFun_slDLSSGSetOptions *p_fg_options;
PFun_slDLSSGGetState *p_fg_state;
PFun_slReflexSetOptions *p_reflex_options;
PFun_slReflexSleep *p_reflex_sleep;
PFun_slPCLSetMarker *p_pcl_marker;
PFN_vkGetDeviceProcAddr p_gdpa;
PFN_vkGetInstanceProcAddr p_gipa;

void log_cb(sl::LogType type, const char *msg) {
    if (type != sl::LogType::eInfo) rt_log("streamline: %s", msg);
}

template <typename T> bool load(T *&p, const char *name) {
    p = reinterpret_cast<T *>(GetProcAddress(g_mod, name));
    return p != nullptr;
}

template <typename T> bool feature_fn(T *&p, sl::Feature f, const char *name) {
    void *v = nullptr;
    p_feature_fn(f, name, v);
    p = reinterpret_cast<T *>(v);
    return p != nullptr;
}

void fill(sl::float4x4 &m, const float *rows) { memcpy(&m, rows, 16 * sizeof(float)); }

sl::Resource to_sl_res(const SlShimImage &i) {
    sl::Resource r(sl::ResourceType::eTex2d, (void *)i.image, nullptr, (void *)i.view, VK_IMAGE_LAYOUT_GENERAL);
    r.width = i.w, r.height = i.h, r.nativeFormat = (uint32_t)i.format, r.mipLevels = 1, r.arrayLayers = 1;
    r.usage = i.usage;
    return r;
}

// Fills *out with the backbuffer subrect. Returns 1 when the extent is usable; 0 when the caller must pass a NULL
// extent pointer (Streamline then uses the full swapchain image). Never returns a 0x0 extent: that triggers
// sanitizeResourceExtent warnings and bad FG sampling.
int to_sl_ebb(const int32_t *bb_rect, uint32_t out_w, uint32_t out_h, sl::Extent *out) {
    if (bb_rect && bb_rect[2] > 0 && bb_rect[3] > 0) {
        *out = sl::Extent{(uint32_t)bb_rect[1], (uint32_t)bb_rect[0], (uint32_t)bb_rect[2], (uint32_t)bb_rect[3]};
        return 1;
    }
    if (out_w > 0 && out_h > 0) {
        *out = sl::Extent{0, 0, out_w, out_h};
        return 1;
    }
    return 0;
}

sl::FrameToken *token(uint32_t frame) {
    sl::FrameToken *tok = nullptr;
    uint32_t idx = frame;
    return p_token && p_token(tok, &idx) == sl::Result::eOk ? tok : nullptr;
}

// the extensions of every loaded feature, without repeats (they outlive the call: plugin-owned strings)
const char *g_dev_ext[64], *g_inst_ext[32];
uint32_t g_n_dev, g_n_inst;
void add(const char **list, uint32_t &n, uint32_t cap, const char *const *ext, uint32_t ne) {
    for (uint32_t i = 0; i < ne; i++) {
        bool have = false;
        for (uint32_t k = 0; k < n && !have; k++) have = !strcmp(list[k], ext[i]);
        if (!have && n < cap) list[n++] = ext[i];
    }
}
}  // namespace

extern "C" int sl_shim_init(const wchar_t *dir, int want_fg) {
    if (g_init) return 1;
    wchar_t path[MAX_PATH];
    _snwprintf(path, MAX_PATH, L"%ls\\sl.interposer.dll", dir);
    g_mod = LoadLibraryW(path);
    if (!g_mod) {
        rt_log("dlss: sl.interposer.dll is not beside mkw.exe: DLSS unavailable");
        return 0;
    }
    if (!load(p_init, "slInit") || !load(p_shutdown, "slShutdown") || !load(p_supported, "slIsFeatureSupported") ||
        !load(p_reqs, "slGetFeatureRequirements") || !load(p_vkinfo, "slSetVulkanInfo") ||
        !load(p_feature_fn, "slGetFeatureFunction") || !load(p_token, "slGetNewFrameToken") ||
        !load(p_consts, "slSetConstants") || !load(p_tag, "slSetTagForFrame") || !load(p_eval, "slEvaluateFeature") ||
        !load(p_is_loaded, "slIsFeatureLoaded") || !load(p_set_loaded, "slSetFeatureLoaded") ||
        !load(p_free, "slFreeResources")) {
        rt_log("dlss: sl.interposer.dll lacks a Streamline function: DLSS unavailable");
        return 0;
    }
    p_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(g_mod, "vkGetDeviceProcAddr"));
    p_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_mod, "vkGetInstanceProcAddr"));
    // DLSS-G's plugins beside the others: a missing one leaves DLSS super resolution as it was
    g_want_fg = false;
    if (want_fg) {
        static const wchar_t *need[] = {L"sl.dlss_g.dll", L"nvngx_dlssg.dll", L"sl.reflex.dll", L"sl.pcl.dll"};
        g_want_fg = true;
        for (const wchar_t *n : need) {
            _snwprintf(path, MAX_PATH, L"%ls\\%ls", dir, n);
            if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
                rt_log("dlss: %ls is not beside mkw.exe: DLSS frame generation unavailable", n);
                g_want_fg = false;
            }
        }
    }
    static const sl::Feature features[] = {sl::kFeatureDLSS, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    static const wchar_t *paths[1];
    paths[0] = dir;
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = log_cb;
    static wchar_t logdir[MAX_PATH];   // MKW_DLSS_LOG=<dir> (diagnosis): Streamline's and NGX's own verbose logs there
    if (GetEnvironmentVariableW(L"MKW_DLSS_LOG", logdir, MAX_PATH)) pref.logLevel = sl::LogLevel::eVerbose, pref.pathToLogsAndData = logdir;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    // no over-the-air updates; tags per frame (slSetTagForFrame)
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = g_want_fg ? 4 : 1;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "mkw-thwomp";
    pref.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";   // THWOMP's own id for Streamline's logs
    pref.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = p_init(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) {
        rt_log("dlss: slInit failed (%d): DLSS unavailable", (int)r);
        return 0;
    }
    g_init = true;
    rt_log("dlss: Streamline %d.%d.%d ready (manual hooking, no over-the-air updates%s)", SL_VERSION_MAJOR, SL_VERSION_MINOR,
           SL_VERSION_PATCH, g_want_fg ? "; DLSS frame generation, Reflex and PCL loaded" : "");
    return 1;
}

extern "C" int sl_shim_requirements(const char *const **dev_ext, uint32_t *n_dev, const char *const **inst_ext,
                                    uint32_t *n_inst, uint32_t *gfx_queues, uint32_t *compute_queues) {
    if (!g_init) return 0;
    sl::FeatureRequirements req{};
    sl::Result r = p_reqs(sl::kFeatureDLSS, req);
    if (r != sl::Result::eOk) {
        rt_log("dlss: no requirements for DLSS (%d): unavailable", (int)r);
        return 0;
    }
    g_n_dev = g_n_inst = 0;
    add(g_dev_ext, g_n_dev, 64, req.vkDeviceExtensions, req.vkNumDeviceExtensions);
    add(g_inst_ext, g_n_inst, 32, req.vkInstanceExtensions, req.vkNumInstanceExtensions);
    uint32_t gq = req.vkNumGraphicsQueuesRequired, cq = req.vkNumComputeQueuesRequired;
    sl::FeatureRequirements com{};   // the common plugin's own needs (queues it would make for itself)
    if (p_reqs(sl::kFeatureCommon, com) == sl::Result::eOk)
        rt_log("dlss: Streamline common plugin: %u graphics, %u compute, %u optical flow queues; DLSS: %u, %u, %u",
               com.vkNumGraphicsQueuesRequired, com.vkNumComputeQueuesRequired, com.vkNumOpticalFlowQueuesRequired,
               req.vkNumGraphicsQueuesRequired, req.vkNumComputeQueuesRequired, req.vkNumOpticalFlowQueuesRequired);
    if (g_want_fg) {
        static const sl::Feature more[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
        static const char *const names[] = {"DLSS frame generation", "Reflex", "PCL"};
        for (int i = 0; i < 3; i++) {
            sl::FeatureRequirements q{};
            if (p_reqs(more[i], q) != sl::Result::eOk) {
                rt_log("dlss: no requirements for %s: DLSS frame generation unavailable", names[i]);
                g_want_fg = false;
                continue;
            }
            // optical flow runs in interop mode: its own queue family is not made (ProgrammingGuideManualHooking 5.2.1)
            rt_log("dlss: %s asks for %u graphics, %u compute, %u optical flow queues, %u device extensions", names[i],
                   q.vkNumGraphicsQueuesRequired, q.vkNumComputeQueuesRequired, q.vkNumOpticalFlowQueuesRequired,
                   q.vkNumDeviceExtensions);
            add(g_dev_ext, g_n_dev, 64, q.vkDeviceExtensions, q.vkNumDeviceExtensions);
            add(g_inst_ext, g_n_inst, 32, q.vkInstanceExtensions, q.vkNumInstanceExtensions);
            gq += q.vkNumGraphicsQueuesRequired, cq += q.vkNumComputeQueuesRequired;
        }
    }
    *dev_ext = g_dev_ext, *n_dev = g_n_dev;
    *inst_ext = g_inst_ext, *n_inst = g_n_inst;
    *gfx_queues = gq, *compute_queues = cq;
    return 1;
}

extern "C" int sl_shim_set_vulkan(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t gfx_family,
                                  uint32_t gfx_index, uint32_t compute_family, uint32_t compute_index) {
    if (!g_init) return 0;
    sl::VulkanInfo info{};
    info.device = dev, info.instance = inst, info.physicalDevice = phys;
    info.graphicsQueueFamily = gfx_family, info.graphicsQueueIndex = gfx_index;
    info.computeQueueFamily = compute_family, info.computeQueueIndex = compute_index;
    sl::Result r = p_vkinfo(info);
    if (r != sl::Result::eOk) {
        rt_log("dlss: slSetVulkanInfo failed (%d)", (int)r);
        return 0;
    }
    sl::AdapterInfo ai{};
    ai.vkPhysicalDevice = phys;
    r = p_supported(sl::kFeatureDLSS, ai);
    if (r != sl::Result::eOk) {
        rt_log("dlss: DLSS is not supported on this GPU or driver (%d)", (int)r);
        return 0;
    }
    if (!feature_fn(p_optimal, sl::kFeatureDLSS, "slDLSSGetOptimalSettings") ||
        !feature_fn(p_options, sl::kFeatureDLSS, "slDLSSSetOptions"))
        return 0;
    g_have = SL_SHIM_DLSS;
    rt_log("dlss: DLSS super resolution supported");
    if (!g_want_fg) return g_have;
    if ((r = p_supported(sl::kFeatureReflex, ai)) == sl::Result::eOk && feature_fn(p_reflex_options, sl::kFeatureReflex, "slReflexSetOptions") &&
        feature_fn(p_reflex_sleep, sl::kFeatureReflex, "slReflexSleep") &&
        p_supported(sl::kFeaturePCL, ai) == sl::Result::eOk && feature_fn(p_pcl_marker, sl::kFeaturePCL, "slPCLSetMarker")) {
        g_have |= SL_SHIM_REFLEX;
        sl::ReflexOptions o{};   // set once even while off (ProgrammingGuideReflex 4.0); vk_reflex.c sets the mode
        p_reflex_options(o);
        rt_log("dlss: Streamline Reflex and PCL markers supported");
    } else
        rt_log("dlss: Streamline Reflex is not supported here (%d): DLSS frame generation unavailable", (int)r);
    r = p_supported(sl::kFeatureDLSS_G, ai);
    if (r != sl::Result::eOk) {
        // an RTX 30 or older, hardware-accelerated GPU scheduling off, an old driver or Windows
        rt_log("dlss: DLSS frame generation is not supported on this GPU, driver or Windows setup (%d)", (int)r);
    } else if (!(g_have & SL_SHIM_REFLEX) || !feature_fn(p_fg_options, sl::kFeatureDLSS_G, "slDLSSGSetOptions") ||
               !feature_fn(p_fg_state, sl::kFeatureDLSS_G, "slDLSSGGetState")) {
        rt_log("dlss: DLSS frame generation's functions or Reflex missing: unavailable");
    } else {
        g_have |= SL_SHIM_FG;
        // loaded only while it makes frames: its swapchain proxy renders off-screen (ProgrammingGuideDLSS_G 18.0)
        p_set_loaded(sl::kFeatureDLSS_G, false);
        rt_log("dlss: DLSS frame generation supported");
    }
    return g_have;
}

extern "C" PFN_vkVoidFunction sl_shim_device_proc(VkDevice dev, const char *name) {
    return g_init && p_gdpa ? p_gdpa(dev, name) : nullptr;
}

extern "C" PFN_vkVoidFunction sl_shim_instance_proc(VkInstance inst, const char *name) {
    return g_init && p_gipa ? p_gipa(inst, name) : nullptr;
}

extern "C" int sl_shim_evaluate(VkCommandBuffer cb, const SlShimFrame *f) {
    if (!g_init || !p_options) return 0;
    sl::ViewportHandle vp{0};
    sl::FrameToken *tok = token(f->frame_index);
    if (!tok) return 0;

    static const sl::DLSSMode modes[] = {sl::DLSSMode::eDLAA, sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced,
                                         sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance};
    sl::DLSSOptions o{};
    o.mode = modes[f->mode < 5 ? f->mode : 1];
    o.outputWidth = f->out_w, o.outputHeight = f->out_h;
    o.colorBuffersHDR = sl::Boolean::eFalse;   // the game's finished picture, read through sRGB: linear, within 0..1
    // No exposure texture: auto-exposure. useAutoExposure=false without a 1x1 exposure map ghosts moving contrast.
    o.useAutoExposure = sl::Boolean::eTrue;
    sl::DLSSPreset gen1 = sl::DLSSPreset::ePresetK, gen2 = sl::DLSSPreset::ePresetM;
    sl::DLSSPreset p = f->model == 2 ? gen2 : gen1;
    o.dlaaPreset = o.qualityPreset = o.balancedPreset = o.performancePreset = p;
    o.ultraPerformancePreset = f->model == 2 ? sl::DLSSPreset::ePresetL : gen1;
    static int fails;
    sl::Result rr = p_options(vp, o);
    if (rr != sl::Result::eOk) {
        if (fails++ < 4) rt_log("dlss: slDLSSSetOptions failed (%d)", (int)rr);
        return 0;
    }

    sl::Constants c{};
    fill(c.cameraViewToClip, f->view_to_clip);
    fill(c.clipToCameraView, f->clip_to_view);
    fill(c.clipToLensClip, f->identity);
    fill(c.clipToPrevClip, f->clip_to_prev_clip);
    fill(c.prevClipToClip, f->prev_clip_to_clip);
    c.jitterOffset = {f->jitter[0], f->jitter[1]};
    c.mvecScale = {1.0f / (float)f->in_w, 1.0f / (float)f->in_h};
    c.cameraPinholeOffset = {0, 0};
    c.cameraPos = {f->cam_pos[0], f->cam_pos[1], f->cam_pos[2]};
    c.cameraUp = {f->cam_up[0], f->cam_up[1], f->cam_up[2]};
    c.cameraRight = {f->cam_right[0], f->cam_right[1], f->cam_right[2]};
    c.cameraFwd = {f->cam_fwd[0], f->cam_fwd[1], f->cam_fwd[2]};
    c.cameraNear = f->near_plane, c.cameraFar = f->far_plane;
    c.cameraFOV = f->fov, c.cameraAspectRatio = f->aspect;
    c.motionVectorsInvalidValue = 0;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = f->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = f->ortho ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eTrue;   // mv_out.comp dilates toward closer depth (thin poles/lamps)
    c.motionVectorsJittered = sl::Boolean::eFalse;
    rr = p_consts(c, *tok, vp);
    if (rr != sl::Result::eOk) {
        if (fails++ < 4) rt_log("dlss: slSetConstants failed (%d)", (int)rr);
        return 0;
    }

    sl::Resource depth = to_sl_res(f->depth), mvec = to_sl_res(f->mvec), in = to_sl_res(f->color_in),
                 out = to_sl_res(f->color_out), bias = to_sl_res(f->bias);
    sl::Extent ein{0, 0, f->in_w, f->in_h}, eout{0, 0, f->out_w, f->out_h};
    // sl::Extent is {top, left, width, height}; bb_rect is {x, y, w, h}.
    // With frame generation, depth (overwritten on replay's next frame) and mvec (single shared image)
    // are reused before Present executes on Streamline's queue, so eOnlyValidNow snapshots them on cb.
    sl::Extent ebb{};
    int have_ebb = to_sl_ebb(f->bb_rect, f->out_w, f->out_h, &ebb);
    sl::ResourceLifecycle fg_life = f->fg ? sl::ResourceLifecycle::eOnlyValidNow : sl::ResourceLifecycle::eValidUntilEvaluate;
    // Transformer presets ignore BiasCurrentColorHint (Preset F only). NGX still consumes Particle + AnimatedTexture
    // masks — feed the reactive hint there. InvalidDepthMotion marks HUD/overlays whose depth/MVs do not match colour.
    sl::ResourceTag tags[12];
    uint32_t nt = 0;
    tags[nt++] = sl::ResourceTag(&depth, sl::kBufferTypeDepth, fg_life, &ein);
    tags[nt++] = sl::ResourceTag(&mvec, sl::kBufferTypeMotionVectors, fg_life, &ein);
    tags[nt++] = sl::ResourceTag(&in, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &ein);
    tags[nt++] = sl::ResourceTag(&out, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &eout);
    if (f->bias.image) {
        sl::ResourceLifecycle bl = sl::ResourceLifecycle::eValidUntilEvaluate;
        tags[nt++] = sl::ResourceTag(&bias, sl::kBufferTypeBiasCurrentColorHint, bl, &ein);   // legacy / FG compose
        tags[nt++] = sl::ResourceTag(&bias, sl::kBufferTypeParticleHint, bl, &ein);
        tags[nt++] = sl::ResourceTag(&bias, sl::kBufferTypeAnimatedTextureHint, bl, &ein);
        tags[nt++] = sl::ResourceTag(&bias, sl::kBufferTypeInvalidDepthMotionHint, bl, &ein);
    }
    // Backbuffer extent: pass a real subrect, or nullptr for the full swapchain — never 0x0.
    if (f->fg)
        tags[nt++] = sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle{},
                                     have_ebb ? &ebb : nullptr);
    sl::CommandBuffer *slcb = reinterpret_cast<sl::CommandBuffer *>(cb);
    rr = p_tag(*tok, vp, tags, nt, slcb);
    if (rr != sl::Result::eOk) {
        if (fails++ < 4) rt_log("dlss: slSetTagForFrame failed (%d)", (int)rr);
        return 0;
    }
    const sl::BaseStructure *inputs[] = {&vp};
    sl::Result r = p_eval(sl::kFeatureDLSS, *tok, inputs, 1, slcb);
    if (r != sl::Result::eOk) {
        static int logged;
        if (logged++ < 4) rt_log("dlss: slEvaluateFeature failed (%d)", (int)r);
        return 0;
    }
    return 1;
}

extern "C" int sl_shim_tag_fg(VkCommandBuffer cb, const SlShimFrame *f) {
    if (!(g_have & SL_SHIM_FG) || !f->fg || !f->hudless.image || !f->ui.image) return 0;
    sl::FrameToken *tok = token(f->frame_index);
    if (!tok) return 0;
    sl::ViewportHandle vp{0};
    sl::Resource hudless = to_sl_res(f->hudless), ui = to_sl_res(f->ui);
    sl::Extent eout{0, 0, f->out_w, f->out_h};
    sl::ResourceTag tags[2] = {
        sl::ResourceTag(&hudless, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eOnlyValidNow, &eout),
        sl::ResourceTag(&ui, sl::kBufferTypeUIColorAndAlpha, sl::ResourceLifecycle::eOnlyValidNow, &eout)};
    sl::Result rr = p_tag(*tok, vp, tags, 2, reinterpret_cast<sl::CommandBuffer *>(cb));
    if (rr != sl::Result::eOk) {
        static int fails;
        if (fails++ < 4) rt_log("dlss: slSetTagForFrame(fg) failed (%d)", (int)rr);
        return 0;
    }
    return 1;
}

extern "C" void sl_shim_fg_clear_tags(VkCommandBuffer cb, uint32_t frame_index, uint32_t out_w, uint32_t out_h,
                                      const int32_t *bb_rect) {
    if (!(g_have & SL_SHIM_FG)) return;
    sl::FrameToken *tok = token(frame_index);
    if (!tok) return;
    sl::ViewportHandle vp{0};
    sl::Extent ebb{};
    int have_ebb = to_sl_ebb(bb_rect, out_w, out_h, &ebb);
    sl::ResourceTag tags[5] = {
        sl::ResourceTag(nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, nullptr),
        sl::ResourceTag(nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, nullptr),
        sl::ResourceTag(nullptr, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent, nullptr),
        sl::ResourceTag(nullptr, sl::kBufferTypeUIColorAndAlpha, sl::ResourceLifecycle::eValidUntilPresent, nullptr),
        sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle{}, have_ebb ? &ebb : nullptr)};
    p_tag(*tok, vp, tags, 5u, reinterpret_cast<sl::CommandBuffer *>(cb));
}

extern "C" void sl_shim_shutdown(void) {
    if (g_init && p_shutdown) p_shutdown();
    g_init = false;
    g_have = 0;
}

// ---- DLSS frame generation

namespace {
void fg_error(const sl::APIError &e) {   // the present thread: logged and back at once
    static volatile LONG n;
    if (InterlockedIncrement(&n) <= 8) rt_log("dlss: frame generation's Vulkan call failed (%d)", (int)e.vkRes);
}
}  // namespace

extern "C" int sl_shim_fg_set(uint32_t frames, uint32_t mvec_w, uint32_t mvec_h, uint32_t color_w, uint32_t color_h) {
    if (!(g_have & SL_SHIM_FG)) return 0;
    sl::ViewportHandle vp{0};
    sl::DLSSGOptions o{};
    o.mode = frames ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    o.numFramesToGenerate = frames ? frames : 1;
    // off for a moment (a menu, a loading screen): its memory stays, so turning it back on does not stutter
    o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    o.mvecDepthWidth = mvec_w, o.mvecDepthHeight = mvec_h;
    o.colorWidth = color_w, o.colorHeight = color_h;
    o.onErrorCallback = fg_error;
    o.hudLessBufferFormat = (uint32_t)VK_FORMAT_R8G8B8A8_UNORM;
    o.uiBufferFormat = (uint32_t)VK_FORMAT_R8G8B8A8_UNORM;
    o.enableUserInterfaceRecomposition = sl::Boolean::eTrue;
    sl::Result r = p_fg_options(vp, o);
    if (r != sl::Result::eOk) {
        static int logged;
        if (logged++ < 4) rt_log("dlss: slDLSSGSetOptions failed (%d)", (int)r);
        return 0;
    }
    return 1;
}

extern "C" int sl_shim_fg_state(uint32_t *max_frames, uint32_t *status, uint32_t *presented, uint32_t *min_side) {
    if (!(g_have & SL_SHIM_FG)) return 0;
    sl::ViewportHandle vp{0};
    sl::DLSSGState s{};
    if (p_fg_state(vp, s, nullptr) != sl::Result::eOk) return 0;
    if (max_frames) *max_frames = s.numFramesToGenerateMax;
    if (status) *status = (uint32_t)s.status;
    if (presented) *presented = s.numFramesActuallyPresented;
    if (min_side) *min_side = s.minWidthOrHeight;
    return 1;
}

extern "C" void sl_shim_fg_loaded(int on) {
    if (!(g_have & SL_SHIM_FG)) return;
    bool now = false;
    p_is_loaded(sl::kFeatureDLSS_G, now);
    if (now == (on != 0)) return;
    sl::Result r = p_set_loaded(sl::kFeatureDLSS_G, on != 0);
    rt_log("dlss: frame generation %s (%d)", on ? "loaded for the next swapchain" : "unloaded", (int)r);
}

extern "C" void sl_shim_fg_free(void) {
    if (!(g_have & SL_SHIM_FG)) return;
    sl::ViewportHandle vp{0};
    p_free(sl::kFeatureDLSS_G, vp);
}

// ---- Streamline's Reflex and PCL

extern "C" int sl_shim_reflex_mode(int mode) {
    if (!(g_have & SL_SHIM_REFLEX)) return 0;
    sl::ReflexOptions o{};
    o.mode = mode == 2 ? sl::ReflexMode::eLowLatencyWithBoost : mode ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
    o.useMarkersToOptimize = true;
    return p_reflex_options(o) == sl::Result::eOk;
}

extern "C" int sl_shim_reflex_sleep(uint32_t frame) {
    if (!(g_have & SL_SHIM_REFLEX)) return 0;
    sl::FrameToken *tok = token(frame);
    return tok && p_reflex_sleep(*tok) == sl::Result::eOk;
}

extern "C" int sl_shim_pcl_marker(int marker, uint32_t frame) {
    if (!(g_have & SL_SHIM_REFLEX)) return 0;
    sl::FrameToken *tok = token(frame);
    return tok && p_pcl_marker((sl::PCLMarker)marker, *tok) == sl::Result::eOk;
}

// ---- the driver profile's frame generation preset (NVAPI driver settings)

namespace {
#pragma pack(push, 4)
typedef uint16_t NvUnicode[2048];
struct NvdrsSetting {   // NVDRS_SETTING_V1
    uint32_t version;
    NvUnicode settingName;
    uint32_t settingId, settingType, settingLocation, isCurrentPredefined, isPredefinedValid;
    union { uint32_t u32; uint8_t binary[4 + 4096]; } predefined, current;
};
struct NvdrsApplication {   // NVDRS_APPLICATION_V1
    uint32_t version, isPredefined;
    NvUnicode appName, userFriendlyName, launcher;
};
struct NvdrsProfile {   // NVDRS_PROFILE_V1
    uint32_t version;
    NvUnicode profileName;
    uint32_t gpuSupport, isPredefined, numOfApps, numOfSettings;
};
#pragma pack(pop)
static_assert(sizeof(NvdrsSetting) == 12320, "NVDRS_SETTING_V1");
static_assert(sizeof(NvdrsApplication) == 12296, "NVDRS_APPLICATION_V1");
static_assert(sizeof(NvdrsProfile) == 4116, "NVDRS_PROFILE_V1");
constexpr uint32_t kFgPresetKey = 0x10E41DF1;   // NGX_DLSS_FG_OVERRIDE_RENDER_PRESET_SELECTION_ID
// The NVIDIA App's "Frame Generation Mode: Fixed Nx" override: the generated frames per game frame (N - 1). Not in a
// published header: nvngx_dlssg 310.9.16 reads it beside the preset key and reports the same value as the multi-frame
// count it applies (its log, 16/11/2026: "drsReadKey id 104d6667, value 5" and "multi-frame count ... value=5").
constexpr uint32_t kFgFramesKey = 0x104D6667;

typedef void *(__cdecl *QueryInterface)(uint32_t id);
typedef int (__cdecl *FnVoid)(void);
typedef int (__cdecl *FnSession)(void **session);
typedef int (__cdecl *FnHandle)(void *session);
typedef int (__cdecl *FnFindApp)(void *session, const uint16_t *name, void **profile, NvdrsApplication *app);
typedef int (__cdecl *FnFindProfile)(void *session, const uint16_t *name, void **profile);
typedef int (__cdecl *FnCreateProfile)(void *session, NvdrsProfile *info, void **profile);
typedef int (__cdecl *FnCreateApp)(void *session, void *profile, NvdrsApplication *app);
typedef int (__cdecl *FnSetSetting)(void *session, void *profile, NvdrsSetting *s);
typedef int (__cdecl *FnGetSetting)(void *session, void *profile, uint32_t id, NvdrsSetting *s);
typedef int (__cdecl *FnDelSetting)(void *session, void *profile, uint32_t id);
typedef int (__cdecl *FnGlobal)(void *session, void **profile);

struct Drs {
    bool tried, ok;
    FnSession create;
    FnHandle destroy, load_settings, save;
    FnFindApp find_app;
    FnFindProfile find_profile;
    FnCreateProfile create_profile;
    FnCreateApp create_app;
    FnSetSetting set;
    FnGetSetting get;
    FnDelSetting del;
    FnGlobal global;
} D;
SRWLOCK g_drs_lock = SRWLOCK_INIT;

bool drs_load() {
    if (D.tried) return D.ok;
    D.tried = true;
    HMODULE m = LoadLibraryW(L"nvapi64.dll");   // the driver's, in System32
    QueryInterface q = m ? reinterpret_cast<QueryInterface>(GetProcAddress(m, "nvapi_QueryInterface")) : nullptr;
    if (!q) return false;
    // the ids of NVIDIA's nvapi_interface.h (github.com/NVIDIA/nvapi)
    FnVoid init = reinterpret_cast<FnVoid>(q(0x0150E828));
    D.create = reinterpret_cast<FnSession>(q(0x0694D52E));
    D.destroy = reinterpret_cast<FnHandle>(q(0xDAD9CFF8));
    D.load_settings = reinterpret_cast<FnHandle>(q(0x375DBD6B));
    D.save = reinterpret_cast<FnHandle>(q(0xFCBC7E14));
    D.find_app = reinterpret_cast<FnFindApp>(q(0xEEE566B2));
    D.find_profile = reinterpret_cast<FnFindProfile>(q(0x7E4A9A0B));
    D.create_profile = reinterpret_cast<FnCreateProfile>(q(0xCC176068));
    D.create_app = reinterpret_cast<FnCreateApp>(q(0x4347A9DE));
    D.set = reinterpret_cast<FnSetSetting>(q(0x577DD202));
    D.get = reinterpret_cast<FnGetSetting>(q(0x73BF8338));
    D.del = reinterpret_cast<FnDelSetting>(q(0xE4A26362));
    D.global = reinterpret_cast<FnGlobal>(q(0x617BFF9F));
    D.ok = init && D.create && D.destroy && D.load_settings && D.save && D.find_app && D.find_profile &&
           D.create_profile && D.create_app && D.set && D.get && D.del && D.global && init() == 0;
    if (!D.ok) rt_log("dlss: NVIDIA's driver settings (NVAPI) are not available: the frame generation preset is the driver's");
    return D.ok;
}

void wcopy(uint16_t *dst, const wchar_t *src) {
    size_t i = 0;
    for (; src[i] && i < 2047; i++) dst[i] = (uint16_t)src[i];
    dst[i] = 0;
}

// this executable's file name ("mkw.exe"): the profile's application
void exe_name(uint16_t *out) {
    wchar_t p[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, p, MAX_PATH);
    const wchar_t *s = p;
    for (DWORD i = 0; i < n; i++)
        if (p[i] == L'\\' || p[i] == L'/') s = p + i + 1;
    wcopy(out, n ? s : L"mkw.exe");
}

volatile int g_preset_cache = -2, g_preset_own, g_preset_last = -1;   // cache -2: not read yet
volatile int g_frames_override;   // the driver's multi-frame count override, 0 none
}  // namespace

extern "C" int sl_shim_fg_preset_read(int *own) {
    // the menu asks every frame: while a write runs, the last value (unknown until then)
    if (!TryAcquireSRWLockExclusive(&g_drs_lock)) {
        if (own) *own = g_preset_own;
        return g_preset_last;
    }
    if (g_preset_cache == -2) {
        g_preset_cache = -1;
        void *s = nullptr;
        if (drs_load() && D.create(&s) == 0) {
            if (D.load_settings(s) == 0) {
                static NvdrsApplication app;
                static NvdrsSetting set;
                static uint16_t name[2048];
                exe_name(name);
                memset(&app, 0, sizeof app);
                app.version = sizeof app | 1u << 16;
                void *prof = nullptr;
                if (D.find_app(s, name, &prof, &app) != 0) prof = nullptr;
                if (!prof) D.global(s, &prof);
                memset(&set, 0, sizeof set);
                set.version = sizeof set | 1u << 16;
                g_preset_cache = 0, g_preset_own = 0, g_frames_override = 0;
                if (prof && D.get(s, prof, kFgPresetKey, &set) == 0) {
                    g_preset_cache = (int)set.current.u32;
                    g_preset_own = set.settingLocation == 0 && app.appName[0];
                }
                memset(&set, 0, sizeof set);
                set.version = sizeof set | 1u << 16;
                if (prof && D.get(s, prof, kFgFramesKey, &set) == 0 && set.current.u32 < 16)
                    g_frames_override = (int)set.current.u32;
            }
            D.destroy(s);
        }
    }
    int v = g_preset_cache;
    g_preset_last = v;
    if (own) *own = g_preset_own;
    ReleaseSRWLockExclusive(&g_drs_lock);
    return v;
}

namespace {
DWORD WINAPI preset_write_thread(void *arg) {
    int preset = (int)(intptr_t)arg;
    AcquireSRWLockExclusive(&g_drs_lock);
    int ok = 0;
    void *s = nullptr;
    if (drs_load() && D.create(&s) == 0) {
        if (D.load_settings(s) == 0) {
            static NvdrsApplication app;
            static NvdrsProfile pi;
            static NvdrsSetting set;
            static uint16_t name[2048], pname[2048];
            exe_name(name);
            memset(&app, 0, sizeof app);
            app.version = sizeof app | 1u << 16;
            void *prof = nullptr;
            int r = D.find_app(s, name, &prof, &app);
            if (r != 0 || !prof) {
                prof = nullptr;
                if (preset) {   // a profile of THWOMP's own for this executable
                    wcopy(pname, L"MKW-THWOMP");
                    if (D.find_profile(s, pname, &prof) != 0) {
                        memset(&pi, 0, sizeof pi);
                        pi.version = sizeof pi | 1u << 16;
                        memcpy(pi.profileName, pname, sizeof pname);
                        if (D.create_profile(s, &pi, &prof) != 0) prof = nullptr;
                    }
                    if (prof) {
                        memset(&app, 0, sizeof app);
                        app.version = sizeof app | 1u << 16;
                        memcpy(app.appName, name, sizeof name);
                        memcpy(app.userFriendlyName, pname, sizeof pname);
                        if ((r = D.create_app(s, prof, &app)) != 0) rt_log("dlss: driver profile: application not added (%d)", r);
                    }
                }
            }
            if (prof && preset) {
                memset(&set, 0, sizeof set);
                set.version = sizeof set | 1u << 16;
                set.settingId = kFgPresetKey;
                set.settingType = 0;   // NVDRS_DWORD_TYPE
                set.current.u32 = (uint32_t)preset;
                r = D.set(s, prof, &set);
                ok = r == 0 && D.save(s) == 0;
                if (r) rt_log("dlss: driver profile: preset not set (%d)", r);
            } else if (prof) {
                r = D.del(s, prof, kFgPresetKey);
                ok = (r == 0 && D.save(s) == 0) || r != 0;   // nothing there to delete is fine too
            } else
                ok = !preset;
        }
        D.destroy(s);
    }
    static const char *const names[] = {"default", "A", "B", "C"};
    rt_log("dlss: frame generation preset %s %s in the driver profile of this executable",
           names[preset >= 0 && preset < 4 ? preset : 0], ok ? "written" : "NOT written");
    g_preset_cache = -2;   // read again
    ReleaseSRWLockExclusive(&g_drs_lock);
    return 0;
}
}  // namespace

extern "C" int sl_shim_fg_frames_override(void) {
    sl_shim_fg_preset_read(nullptr);
    return g_frames_override;
}

extern "C" int sl_shim_fg_preset_write(int preset) {
    if (preset < 0 || preset > 3) return 0;
    // the driver's settings take a moment to load and save: not on the present thread
    HANDLE h = CreateThread(nullptr, 0, preset_write_thread, (void *)(intptr_t)preset, 0, nullptr);
    if (!h) return 0;
    CloseHandle(h);
    return 1;
}

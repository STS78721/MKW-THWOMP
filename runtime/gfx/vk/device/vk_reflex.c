// THWOMP: NVIDIA Reflex low latency (MKW_REFLEX=off|on|boost) through VK_NV_low_latency2, the driver's own Vulkan
// Reflex: no NVIDIA SDK. Off by default, or where the driver lacks the extension (AMD, Intel, older NVIDIA drivers):
// then nothing here runs.
//
// What it does: the swapchain is made in latency mode (gfx_present.c swapchain_build); with Reflex on the driver keeps
// no frames queued ahead and, at the start of each game frame (nvnWindowAcquireTexture, after the pacer's own wait),
// vkLatencySleepNV holds the game until the latest moment its frame still makes its display time: the controller is
// read later, so less time passes between a press and the picture. Boost also keeps the GPU's clocks up.
// The markers tell the driver each game frame's stages, by its present id (the game's present count, the same number on
// the game thread and the replay thread: vk_backend.c game_frames and gfx_counters.frames):
//   SIMULATION_START  the game's frame begins (after the sleep)        game thread
//   INPUT_SAMPLE      the game reads the controller (input.c)          game thread
//   SIMULATION_END + RENDERSUBMIT_START  the game presents             game thread
//   RENDERSUBMIT_END  the replay thread has submitted the frame        replay thread
//   PRESENT_START / PRESENT_END  around the host present, whose VkPresentIdKHR is that id (made frames carry none)
// The NVIDIA overlay's latency figures (Alt+R) read them.
//
// With NVIDIA DLSS frame generation loaded (vk_dlss.c, MKW_FRAMEGEN_TYPE=dlss) the same calls go through Streamline's
// Reflex and PCL plugins instead (DLSS-G requires them: docs/ProgrammingGuideDLSS_G.md section 8): slReflexSleep and
// every marker on every frame whatever the mode (ProgrammingGuideReflex 4.0), the frame index the same id DLSS's
// constants carry, and Reflex at least on while frame generation makes frames. The driver's own calls stay unused then.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "sl_shim.h"

static struct {
    PFN_vkSetLatencySleepModeNV sleep_mode;
    PFN_vkLatencySleepNV sleep;
    PFN_vkSetLatencyMarkerNV marker;
    VkSemaphore sem;
    uint64_t sem_val;
    VkSwapchainKHR sc;      // the swapchain the markers go to; VK_NULL_HANDLE while there is none
    int mode;               // the mode the driver has: 0 off, 1 on, 2 boost; -1 not set yet
    unsigned cfg_seen;
    int loaded;
    volatile uint64_t sim_id;   // the frame the game is in (its next present's id)
} R = {.mode = -1};
static PlatMutex rl = PLAT_MUTEX_INIT;
static void note_start(uint64_t id);

int vk_reflex_supported(void) { return vk_ctx() && (vk_ctx()->has_low_latency || vk_dlssg_reflex()); }

static int wanted_mode(void) {
    const char *v = cfg_value("MKW_REFLEX");
    int m = !strcmp(v, "boost") ? 2 : !strcmp(v, "on") ? 1 : 0;
    return !m && vk_dlssg_frames() ? 1 : m;   // DLSS frame generation needs Reflex on (DLSSGStatus 2 otherwise)
}

// ---- through Streamline (DLSS frame generation loaded)
static int sl_mode = -1;
static void sl_frame_start(uint64_t id) {
    static unsigned seen = ~0u;
    unsigned v = cfg_version();
    if (v != seen || sl_mode < 0) {
        seen = v;
        int m = wanted_mode();
        if (m != sl_mode && sl_shim_reflex_mode(m)) {
            rt_log("reflex: %s through Streamline", m == 2 ? "on + boost" : m ? "on" : "off");
            sl_mode = m;
        }
    }
    __atomic_store_n(&R.sim_id, id, __ATOMIC_RELAXED);   // vk_reflex_input's frame
    uint64_t p0 = perf_begin();
    sl_shim_reflex_sleep((uint32_t)id);
    perf_end(PERF_PACE, p0);
    note_start(id);
    sl_shim_pcl_marker(SL_PCL_SIM_START, (uint32_t)id);
}

static int load(VkCtx *c) {
    if (R.loaded) return R.loaded > 0;
    R.loaded = -1;
    R.sleep_mode = (PFN_vkSetLatencySleepModeNV)vkGetDeviceProcAddr(c->device, "vkSetLatencySleepModeNV");
    R.sleep = (PFN_vkLatencySleepNV)vkGetDeviceProcAddr(c->device, "vkLatencySleepNV");
    R.marker = (PFN_vkSetLatencyMarkerNV)vkGetDeviceProcAddr(c->device, "vkSetLatencyMarkerNV");
    if (!R.sleep_mode || !R.sleep || !R.marker) {
        rt_log("reflex: VK_NV_low_latency2 functions missing: Reflex off");
        return 0;
    }
    VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
    if (vkCreateSemaphore(c->device, &si, NULL, &R.sem) != VK_SUCCESS) return 0;
    R.loaded = 1;
    return 1;
}

// rl held: the driver gets the mode the setting asks for (the swapchain is new, or the setting changed)
static void apply_mode_locked(VkCtx *c, int force) {
    int m = wanted_mode();
    if (!force && m == R.mode) return;
    VkLatencySleepModeInfoNV mi = {.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV,
                                   .lowLatencyMode = m > 0, .lowLatencyBoost = m > 1, .minimumIntervalUs = 0};
    VkResult r = R.sleep_mode(c->device, R.sc, &mi);
    if (m != R.mode) rt_log("reflex: %s (%d)", m == 2 ? "on + boost" : m ? "on" : "off", (int)r);
    R.mode = m;
}

// gfx_present.c: the swapchain was made (sc) or is about to go (VK_NULL_HANDLE), queue lock held
void vk_reflex_swapchain(VkSwapchainKHR sc) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_low_latency || vk_dlssg_reflex() || !load(c)) return;
    plat_mutex_lock(&rl);
    R.sc = sc;
    if (sc) apply_mode_locked(c, 1);
    plat_mutex_unlock(&rl);
}

static void mark_locked(VkLatencyMarkerNV what, uint64_t id) {
    VkSetLatencyMarkerInfoNV mi = {.sType = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV, .presentID = id,
                                   .marker = what};
    R.marker(vk_ctx()->device, R.sc, &mi);
}

static void mark(VkLatencyMarkerNV what, uint64_t id) {
    if (id && vk_dlssg_reflex()) {   // Streamline's PCL: the same stages, numbered alike up to PRESENT_END
        int pcl = what == VK_LATENCY_MARKER_INPUT_SAMPLE_NV ? SL_PCL_INPUT : (int)what;
        if (what <= VK_LATENCY_MARKER_PRESENT_END_NV || what == VK_LATENCY_MARKER_INPUT_SAMPLE_NV)
            sl_shim_pcl_marker(pcl, (uint32_t)id);
        return;
    }
    if (R.loaded <= 0 || !id) return;
    plat_mutex_lock(&rl);
    if (R.sc && R.mode > 0) mark_locked(what, id);
    plat_mutex_unlock(&rl);
}

// The game thread, at the start of a frame (nvnWindowAcquireTexture): the setting's change, the sleep, the marker.
void vk_reflex_frame_start(uint64_t id) {
    VkCtx *c = vk_ctx();
    if (c && vk_dlssg_reflex()) {
        sl_frame_start(id);
        return;
    }
    if (!c || R.loaded <= 0) {
        note_start(id);
        return;
    }
    __atomic_store_n(&R.sim_id, id, __ATOMIC_RELAXED);
    unsigned v = cfg_version();
    plat_mutex_lock(&rl);
    if (v != R.cfg_seen && R.sc) {
        R.cfg_seen = v;
        apply_mode_locked(c, 0);
    }
    if (!R.sc || R.mode <= 0) {
        plat_mutex_unlock(&rl);
        note_start(id);
        return;
    }
    uint64_t want = ++R.sem_val;
    VkLatencySleepInfoNV si = {.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV, .signalSemaphore = R.sem, .value = want};
    VkResult r = R.sleep(c->device, R.sc, &si);
    plat_mutex_unlock(&rl);
    if (r == VK_SUCCESS) {
        VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                                  .pSemaphores = &R.sem, .pValues = &want};
        uint64_t p0 = perf_begin();
        vkWaitSemaphores(c->device, &wi, 50000000ull);   // a swapchain rebuilt meanwhile: never more than 50 ms
        perf_end(PERF_PACE, p0);
    }
    note_start(id);
    mark(VK_LATENCY_MARKER_SIMULATION_START_NV, id);
}

void vk_reflex_input(void) {
    static uint64_t last;   // once per frame: the first read
    uint64_t id = __atomic_load_n(&R.sim_id, __ATOMIC_RELAXED);
    if (!id || id == last) return;
    last = id;
    mark(VK_LATENCY_MARKER_INPUT_SAMPLE_NV, id);
}

void vk_reflex_game_present(uint64_t id) {
    mark(VK_LATENCY_MARKER_SIMULATION_END_NV, id);
    mark(VK_LATENCY_MARKER_RENDERSUBMIT_START_NV, id);
}

void vk_reflex_submitted(uint64_t id) { mark(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, id); }

// The present thread, around the game frame's host present (gfx_present.c flip_one_locked)
int vk_reflex_on(void) { return vk_dlssg_reflex() ? sl_mode > 0 : R.loaded > 0 && R.mode > 0; }
void vk_reflex_present(uint64_t id, int end) {
    mark(end ? VK_LATENCY_MARKER_PRESENT_END_NV : VK_LATENCY_MARKER_PRESENT_START_NV, id);
}

// The frame's latency as the game sees it, Reflex on or off: from the frame's start (after any sleep: the game reads
// the controller right after) to its host present. Logged as an average every 600 frames: "reflex: start to present".
static uint64_t started[64];
static void note_start(uint64_t id) { started[id & 63] = kern_now_ns(); }
void vk_reflex_shown(uint64_t id) {
    static uint64_t sum, n, worst;
    uint64_t t0 = started[id & 63], now = kern_now_ns();
    if (!id || !t0 || now < t0 || now - t0 > 200000000ull) return;
    uint64_t d = now - t0;
    sum += d, n++, worst = d > worst ? d : worst;
    if (n == 600) {
        rt_log("reflex: start to present %.2f ms average, %.2f ms worst over 600 frames (Reflex %s)", sum / 6e8,
               worst / 1e6, vk_dlssg_reflex() ? (sl_mode == 2 ? "boost, Streamline" : sl_mode == 1 ? "on, Streamline" : "off")
                                              : R.mode == 2 ? "boost" : R.mode == 1 ? "on" : "off");
        sum = n = worst = 0;
    }
}

void vk_reflex_release(void) {
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&rl);
    R.sc = VK_NULL_HANDLE;
    if (c && R.sem) vkDestroySemaphore(c->device, R.sem, NULL);
    R.sem = VK_NULL_HANDLE;
    R.loaded = 0, R.mode = -1;
    plat_mutex_unlock(&rl);
}

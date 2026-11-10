// The in-game menu core: Dear ImGui state, the event queue the window adapter fills, gamepad navigation, the Vulkan back
// end on our device and the menu's pages. Free of any window system (menu.h). ImGui runs on the present path only,
// under gfx_present.c's present lock; the event functions only queue.
//
// The Vulkan set-up follows LAS Recomp's runtime/nvn/vulkan/imgui_glue.cpp (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai): dynamic rendering into the swapchain
// image, an internal descriptor pool, vertex buffers rotated over more frames than are in flight.
//
// The look (colours, rounding, the tab bar with L / R and Q / E, the window over a dimmed game) follows
// SavageWindRecomp's runtime/ui/menu.mm (https://github.com/Revivedx/SavageWindRecomp @
// 575ad88e02f9a89a9a885d90039d48af33c982f4, MIT, Copyright (c) 2026 Revivedx and the Savage Wind Recomp contributors,
// Copyright (c) 2026 Peiikai); the pages are rows of option_rows.cpp.
#include <algorithm>
#include <atomic>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // SetFocusID, NavCursorVisible
#include "backends/imgui_impl_vulkan.h"
// A private copy of stb_truetype, the parser Dear ImGui builds its font atlas with, to check a system font before the
// atlas gets it (font_usable). Its asserts only mark the font bad here.
static int g_font_bad;
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_assert(x) ((x) ? (void)0 : (void)(g_font_bad = 1))
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "imstb_truetype.h"
#pragma clang diagnostic pop
#include "ui/binds_page.h"
#include "ui/option_rows.h"
#include "ui/theme.h"

extern "C" {
#include "cfg.h"
#include "input.h"
#include "input_map.h"
#include "perfmon.h"
#include "platform/platform.h"
#include "sysinfo.h"
#include "ui/menu.h"
#include "ui_text.h"
// host.h and vk.h pull in cpu.h, which is C only: the functions the menu needs from them, as declared there.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void vk_queue_lock(void);
void vk_queue_unlock(void);
void res_sizes(unsigned out[2], unsigned scene[2]);
// Vulkan & GPU capabilities

double frame_step_limit_hz(void);
extern const int rt_game_version;
// fs.h (romfs.c): romfs-patches-xbox was next to romfs-index.bin at start, and was loaded.
int romfs_xbox_prompts_ready(void);
int romfs_xbox_prompts_active(void);
extern const char rt_port_version[];
}

// Roboto Medium (Apache-2.0, ui/fonts/LICENSE-Roboto.txt) with Latin and Cyrillic glyphs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"   // #embed in C++
static const unsigned char roboto_ttf[] = {
#embed "fonts/Roboto-Medium.ttf"
};
#pragma clang diagnostic pop

// ---- open state, any thread
static std::atomic<int> g_open;     // the menu is open
static std::atomic<int> g_reset;    // opened or closed since the last laid-out frame: drop stale input

// g_open, the input block and the log line change together under open_lock, so they cannot disagree when the window thread
// (F1, Esc), the polling thread (combo) and the present path (Continue, B) race.
static PlatMutex open_lock = PLAT_MUTEX_INIT;

// want 0 or 1 sets the state; -1 flips it. Returns the new state.
static int change_open(int want) {
    plat_mutex_lock(&open_lock);
    int open = want < 0 ? !g_open.load() : want;
    if (g_open.load() != open) {
        g_open.store(open);
        g_reset.store(1);
        input_menu(open);
        rt_log("menu: %s", open ? "open" : "closed");
    }
    plat_mutex_unlock(&open_lock);
    return open;
}

static void set_open(int open) { change_open(open); }

static int g_failed_flag();
extern "C" int menu_is_open(void) { return g_open.load(std::memory_order_relaxed); }
extern "C" void menu_toggle(void) {
    if (!g_failed_flag()) change_open(-1);
}

// ---- events from the window adapter, queued while the menu is open; the present path hands them to ImGui
enum { EV_KEY, EV_POS, EV_BUTTON, EV_WHEEL, EV_FOCUS, EV_MAX = 256 };
struct Ev { int type, a, b; float x, y; };
static PlatMutex ev_lock = PLAT_MUTEX_INIT;
static Ev ev_q[EV_MAX];
static int ev_n;

static void push(Ev e) {
    if (!menu_is_open()) return;
    plat_mutex_lock(&ev_lock);
    if (e.type == EV_POS && ev_n && ev_q[ev_n - 1].type == EV_POS) ev_q[ev_n - 1] = e;   // only the newest position
    else if (ev_n < EV_MAX) ev_q[ev_n++] = e;
    plat_mutex_unlock(&ev_lock);
}

extern "C" void menu_event_key(int key, int down) { push({EV_KEY, key, down, 0, 0}); }
extern "C" void menu_event_mouse_pos(float x, float y) { push({EV_POS, 0, 0, x, y}); }
extern "C" void menu_event_mouse_button(int button, int down) { push({EV_BUTTON, button, down, 0, 0}); }
extern "C" void menu_event_wheel(float dx, float dy) { push({EV_WHEEL, 0, 0, dx, dy}); }
extern "C" void menu_event_focus(int focused) { push({EV_FOCUS, focused, 0, 0, 0}); }

static void drain_events(ImGuiIO &io) {
    Ev q[EV_MAX];
    plat_mutex_lock(&ev_lock);
    int n = ev_n;
    for (int i = 0; i < n; i++) q[i] = ev_q[i];
    ev_n = 0;
    plat_mutex_unlock(&ev_lock);
    for (int i = 0; i < n; i++) {
        const Ev &e = q[i];
        switch (e.type) {
        case EV_KEY: io.AddKeyEvent((ImGuiKey)e.a, e.b != 0); break;
        case EV_POS: io.AddMousePosEvent(e.x, e.y); break;
        case EV_BUTTON: io.AddMouseButtonEvent(e.a, e.b != 0); break;
        case EV_WHEEL: io.AddMouseWheelEvent(e.x, e.y); break;
        case EV_FOCUS: io.AddFocusEvent(e.a != 0); break;
        }
    }
}

// ---- gamepad navigation: the Switch buttons of our SDL3 gamepad to ImGui's gamepad keys. Nintendo layout: A (east)
// activates, B (south) cancels.
static const struct { int bit; ImGuiKey key; } pad_keys[] = {
    {B_A, ImGuiKey_GamepadFaceDown}, {B_B, ImGuiKey_GamepadFaceRight}, {B_X, ImGuiKey_GamepadFaceUp},
    {B_Y, ImGuiKey_GamepadFaceLeft}, {B_L, ImGuiKey_GamepadL1}, {B_R, ImGuiKey_GamepadR1},
    {B_ZL, ImGuiKey_GamepadL2}, {B_ZR, ImGuiKey_GamepadR2}, {B_LEFT, ImGuiKey_GamepadDpadLeft},
    {B_UP, ImGuiKey_GamepadDpadUp}, {B_RIGHT, ImGuiKey_GamepadDpadRight}, {B_DOWN, ImGuiKey_GamepadDpadDown},
};
static uint64_t pad_prev;   // buttons at the last frame; a button held when the menu opened counts once released
static int stick_dir = -1;  // the D-pad button the left stick holds, -1 none

// The left stick moves the cursor as the D-pad does (ImGui's own stick handling only scrolls the page): past half way
// it holds the D-pad direction of its larger axis, kept until that axis falls under 0.35; ImGui repeats a held one.
static int stick_to_dpad(int32_t lx, int32_t ly) {
    float x = lx / 32767.0f, y = ly / 32767.0f;   // y up
    float along = stick_dir == B_LEFT ? -x : stick_dir == B_RIGHT ? x : stick_dir == B_UP ? y : stick_dir == B_DOWN ? -y : 0;
    if (stick_dir >= 0 && along > 0.35f) return stick_dir;
    if (fmaxf(fabsf(x), fabsf(y)) < 0.5f) return -1;
    return fabsf(x) > fabsf(y) ? (x < 0 ? B_LEFT : B_RIGHT) : (y > 0 ? B_UP : B_DOWN);
}

static void pad_events(ImGuiIO &io, int reset) {
    uint64_t b;
    int32_t lx, ly;
    int have = input_pad_read(&b, &lx, &ly);
    if (have) io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    else io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
    stick_dir = stick_to_dpad(lx, ly);
    if (stick_dir >= 0) b |= 1ull << stick_dir;
    if (reset) {
        pad_prev = b;
        return;
    }
    for (const auto &k : pad_keys) {
        uint64_t m = 1ull << k.bit;
        if ((b ^ pad_prev) & m) io.AddKeyEvent(k.key, (b & m) != 0);
    }
    pad_prev = b;
}

// ---- ImGui and its Vulkan back end (present path)
static int g_ctx_ok;              // context made
static std::atomic<int> g_failed; // set-up failed for good (the menu stays closed)
static int g_failed_flag() { return g_failed.load(std::memory_order_relaxed); }
static int g_vk_ok;               // back end initialised for g_vk_fmt
static VkFormat g_vk_fmt;
static float g_font_scale;        // the scale of the uploaded font atlas; 0 when none is uploaded
static float g_style_scale;       // the scale the style was last sized for
static ImGuiStyle g_base_style;   // the menu's style at 1080 lines
static uint64_t g_last_ns;
static uint64_t g_cost_ns, g_cost_frames;   // CPU time of layout and recording while open, logged per 600 frames

// Sizes are tuned for 1080 lines and follow the window's height. Under 720 lines at 100% display scale they stay at the
// 720 size; that floor grows with the window's DPI, so a small window on a 150% or 200% display (sizes are physical
// pixels) still gets text as large as the display scale asks for.
static float menu_scale(uint32_t h) {
    float least = 720.0f * (float)gfx_window_dpi() / 96.0f;
    return ((float)h >= least ? (float)h : least) / 1080.0f;
}

// The Mario Kart World PC theme (ui/theme.h): light (Switch 2 system settings, cyan) or dark (red accents),
// MKW_MENU_THEME. Re-run when the theme changes (menu_overlay_frame).
static void style_init(ImGuiStyle &s) {
    using namespace mkw_theme;
    bool dark = !strcmp(cfg_value("MKW_MENU_THEME"), "dark");
    apply(dark);
    if (dark) ImGui::StyleColorsDark(&s);
    else ImGui::StyleColorsLight(&s);
    s.WindowRounding = 18; s.FrameRounding = 20; s.GrabRounding = 8; s.ChildRounding = 10; s.PopupRounding = 14;
    s.WindowPadding = ImVec2(28, 22); s.FramePadding = ImVec2(18, 9); s.ItemSpacing = ImVec2(14, 12);
    s.WindowBorderSize = 0; s.PopupBorderSize = 1; s.FrameBorderSize = 1; s.ScrollbarSize = 14; s.ScrollbarRounding = 7;
    ImVec4 *c = s.Colors;
    c[ImGuiCol_WindowBg] = v4(kPanel);
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = v4(kHeader);
    c[ImGuiCol_Border] = v4(kDivider);
    c[ImGuiCol_Text] = v4(kText);
    c[ImGuiCol_TextDisabled] = v4(kTextDim);
    c[ImGuiCol_FrameBg] = v4(kRowSel);
    c[ImGuiCol_Button] = v4(kRowSel);
    c[ImGuiCol_ButtonHovered] = v4(kButtonHover);
    c[ImGuiCol_ButtonActive] = v4(kButtonActive);
    c[ImGuiCol_NavCursor] = v4(kAccent);
    c[ImGuiCol_Separator] = v4(kDivider);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = v4(kScroll);
    c[ImGuiCol_ScrollbarGrabHovered] = v4(kTextDim);
    c[ImGuiCol_ScrollbarGrabActive] = v4(kAccent);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.30f);
}

static void make_context(void) {
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendPlatformName = "mkw_menu";
    style_init(g_base_style);
    ImGui::GetStyle() = g_base_style;
    g_ctx_ok = 1;
}

static void vk_down(void) {
    if (!g_vk_ok) return;
    ImGui_ImplVulkan_Shutdown();
    g_vk_ok = 0;
    g_font_scale = 0;
}

static int vk_up(const MenuTarget *t) {
    if (g_vk_ok && g_vk_fmt == t->format) return 1;
    if (g_vk_ok) {   // the swapchain changed format: pipelines for the new one
        vk_queue_lock();
        vkQueueWaitIdle(t->queue);
        vk_queue_unlock();
        vk_down();
    }
    ImGui_ImplVulkan_InitInfo ii = {};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = t->instance;
    ii.PhysicalDevice = t->phys;
    ii.Device = t->device;
    ii.QueueFamily = t->queue_family;
    ii.Queue = t->queue;
    ii.DescriptorPoolSize = 16;
    ii.MinImageCount = 2;
    ii.ImageCount = 4;   // vertex buffers rotate per recorded frame: more than gfx_present.c's 2 frames in flight
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    g_vk_fmt = t->format;
    ii.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &g_vk_fmt;
    if (!ImGui_ImplVulkan_Init(&ii)) return 0;
    g_vk_ok = 1;
    return 1;
}

// Roboto Medium (Latin, Cyrillic, dashes and quotes) at 26 px per 1080 lines, with the glyphs of the Japanese, Korean
// and Chinese language names merged in from a system font when there is one (each language is named in its own script).
// Made again when the window height changes; the old texture goes after the queue is idle (CreateFontsTexture).
static const ImWchar kRanges[] = {0x0020, 0x00FF, 0x0400, 0x052F, 0x2010, 0x2027, 0};
static const struct { const char *lang; UiTextId name; } kCjk[] = {
    {"ja", UI_MENU_LANGNAME_JA}, {"ko", UI_MENU_LANGNAME_KO}, {"zh-Hans", UI_MENU_LANGNAME_ZH_HANS},
    {"zh-Hant", UI_MENU_LANGNAME_ZH_HANT},
};

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

// 1 when Dear ImGui can take this font file: more than the 100 bytes AddFontFromMemoryTTF asserts on, every table of its
// directory inside the file (stb_truetype reads tables without bounds checks, so a cut-off file faults when the glyphs
// are made), and stb_truetype parses it. ImGui asserts on a font it cannot parse, and an assert ends the process with
// 0xc0000409 and no line in the log.
static int font_usable(const uint8_t *d, uint64_t n) {
    if (!d || n <= 100 || n > INT_MAX) return 0;
    int off = stbtt_GetFontOffsetForIndex(d, 0);
    if (off < 0 || (uint64_t)off + 12 > n) return 0;
    uint32_t tables = (uint32_t)d[off + 4] << 8 | d[off + 5];
    if ((uint64_t)off + 12 + 16ull * tables > n) return 0;
    for (uint32_t i = 0; i < tables; i++) {
        const uint8_t *r = d + off + 12 + 16 * i;
        if ((uint64_t)be32(r + 8) + be32(r + 12) > n) return 0;
    }
    stbtt_fontinfo fi;
    g_font_bad = 0;
    return stbtt_InitFont(&fi, d, off) && !g_font_bad;
}

extern "C" int menu_font_check(void) {
    uint64_t n = sizeof roboto_ttf;
    uint8_t *bad = (uint8_t *)malloc(n);
    if (!bad) return 0;
    memcpy(bad, roboto_ttf, n);
    uint32_t tables = (uint32_t)bad[4] << 8 | bad[5];
    for (uint32_t i = 0; i < tables; i++)
        if (!memcmp(bad + 12 + 16 * i, "cmap", 4)) memset(bad + 12 + 16 * i, 0, 4);   // no character map
    int ok = font_usable(roboto_ttf, n) && !font_usable(roboto_ttf, 4096) && !font_usable(roboto_ttf, 100) &&
             !font_usable(bad, n);
    memset(bad, 0, 4);             // an unknown font tag
    ok = ok && !font_usable(bad, n);
    free(bad);
    return ok;
}

static int fonts_up(float scale) {
    enum { N = sizeof kCjk / sizeof *kCjk };
    ImFontAtlas *fa = ImGui::GetIO().Fonts;
    fa->Clear();
    ImFontConfig fc;
    fc.FontDataOwnedByAtlas = false;
    fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, 26.0f * scale, &fc, kRanges);
    uint8_t *data[N] = {};
    ImVector<ImWchar> ranges[N];
    int merged = 0;
    for (int i = 0; i < N; i++) {
        char path[512];
        uint64_t size = 0;
        if (!plat_font_path(kCjk[i].lang, path, sizeof path) || !(data[i] = plat_file_read(path, &size))) continue;
        if (!font_usable(data[i], size)) {
            rt_log("menu: system font %s left out: the font parser cannot read it (%llu bytes)", path,
                   (unsigned long long)size);
            free(data[i]);
            data[i] = NULL;
            continue;
        }
        merged++;
        ImFontGlyphRangesBuilder b;
        b.AddText(ui_text_in("en", kCjk[i].name));
        b.BuildRanges(&ranges[i]);
        ImFontConfig mc;
        mc.MergeMode = true;
        mc.FontDataOwnedByAtlas = false;
        fa->AddFontFromMemoryTTF(data[i], (int)size, 26.0f * scale, &mc, ranges[i].Data);
    }
    vk_queue_lock();   // the upload submits on the queue and waits for it
    int ok = ImGui_ImplVulkan_CreateFontsTexture();
    vk_queue_unlock();
    fa->ClearInputData();   // the glyphs are in the texture: the font files and the pixels are not needed again
    fa->ClearTexData();
    for (uint8_t *d : data) free(d);
    rt_log("menu: font atlas at %.1f px, language names from %d of %d system fonts", 26.0f * scale, merged, (int)N);
    g_font_scale = ok ? scale : 0;
    return ok;
}

extern "C" void menu_vk_release(void) {
    if (g_ctx_ok) vk_down();
}

// ---- test hooks
// MKW_MENU_TEST=<frame>[:<frames per page>]: at that presented frame the menu opens on its first tab; each tab stays
// <frames per page> frames (default 30) and its last frame's swapchain image goes to MKW_MENU_TEST_DIR (default: the
// working directory) as menu-test-<frame>-<page>.png; after the last tab the menu closes.
// MKW_MENU_TEST_SET=<frame>:<key>=<value>[,<key>=<value>...][;<frame>:...]: cfg_set of each key (TOTK_ prefix
// optional) at that presented frame; changes live settings with no input, menu open or not. mkw.ini is written only by
// the pseudo-key SAVE=1 (cfg_save, as a row change does; a failed write shows the menu's save note).
enum { TAB_DISPLAY, TAB_SOUND, TAB_CONTROLS, TAB_SYSTEM, TAB_ADVANCED, TAB_COUNT };
enum { TEST_PAGE_FRAMES = 30, TEST_SETS = 16 };
struct TestSet { uint64_t frame; char key[64], value[64]; };
static struct {
    int parsed;
    uint64_t at;          // MKW_MENU_TEST frame; 0 none
    uint64_t per_page;
    uint64_t opened;      // the frame the hook opened the menu; 0 not yet
    int page;             // the tab being shot; -1 none
    TestSet sets[TEST_SETS];
    int n_sets, next_set; // MKW_MENU_TEST_SET, in frame order
} g_test = {0, 0, 0, 0, -1, {}, 0, 0};

static void test_parse(void) {
    g_test.parsed = 1;
    if (const char *e = cfg_get("MKW_MENU_TEST")) {
        long long at = atoll(e), per = 0;
        if (const char *c = strchr(e, ':')) per = atoll(c + 1);
        g_test.at = at > 0 ? (uint64_t)at : 0;
        g_test.per_page = per > 0 ? (uint64_t)per : TEST_PAGE_FRAMES;
    }
    const char *s = cfg_get("MKW_MENU_TEST_SET");
    while (s && *s) {   // groups "<frame>:<key>=<value>,..." separated by ';'
        char *end;
        unsigned long long f = strtoull(s, &end, 10);
        if (*end != ':') break;
        s = end + 1;
        while (*s && *s != ';' && g_test.n_sets < TEST_SETS) {
            size_t n = strcspn(s, ",;");
            const char *eq = (const char *)memchr(s, '=', n);
            if (eq) {
                TestSet &t = g_test.sets[g_test.n_sets++];
                t.frame = f;
                int kn = (int)(eq - s), vn = (int)(s + n - eq - 1);
                snprintf(t.key, sizeof t.key, "%s%.*s", strncmp(s, "MKW_", 5) ? "MKW_" : "", kn, s);
                snprintf(t.value, sizeof t.value, "%.*s", vn, eq + 1);
            }
            s += n;
            if (*s == ',') s++;
        }
        while (*s && *s != ';') s++;
        if (*s == ';') s++;
    }
    if (g_test.at || g_test.n_sets)
        rt_log("menu: test hook: open at frame %llu, %llu frames per page, %d settings changes",
               (unsigned long long)g_test.at, (unsigned long long)g_test.per_page, g_test.n_sets);
}

static void test_apply_sets(uint64_t frame) {
    for (; g_test.next_set < g_test.n_sets && g_test.sets[g_test.next_set].frame <= frame; g_test.next_set++) {
        const TestSet &t = g_test.sets[g_test.next_set];
        rt_log("menu: test hook: frame %llu: %s=%s", (unsigned long long)frame, t.key, t.value);
        if (!strcmp(t.key, "MKW_SAVE")) rt_log("menu: test hook: cfg_save %s", cfg_save() ? "wrote" : "failed");
        else cfg_set(t.key, t.value);
    }
}

// ---- the lists the Screen and Game tabs build
enum { MAX_GPUS = 8 };
static char g_gpu_names[MAX_GPUS][VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
static int g_gpu_n = -1;   // -1: not listed yet

// The Vulkan devices in vkEnumeratePhysicalDevices order, which TOTK_GPU_INDEX counts in (vk_device.c pick_device).
static void list_gpus(VkInstance inst) {
    if (g_gpu_n >= 0) return;
    VkPhysicalDevice pd[MAX_GPUS];
    uint32_t n = MAX_GPUS;
    if (vkEnumeratePhysicalDevices(inst, &n, pd) < 0) n = 0;   // VK_INCOMPLETE keeps the first MAX_GPUS
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        snprintf(g_gpu_names[i], sizeof g_gpu_names[i], "%s", p.deviceName);
    }
    g_gpu_n = (int)n;
}

// Window sizes: the common 16:9 sizes that fit the monitor, the 1920x1080 default and the current size.
static const int kSizes[][2] = {{1280, 720}, {1366, 768}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3200, 1800},
                                {3840, 2160}};
enum { MAX_SIZES = sizeof kSizes / sizeof *kSizes + 1 };

static void row_window_size(void) {
    if (cfg_bool("MKW_FULLSCREEN")) opt_disable_next(UI_MENU_WINDOWED_ONLY);
    int sz[MAX_SIZES][2], n = 0, mw = 0, mh = 0, cw = 0, ch = 0;
    int have = gfx_window_monitor_size(&mw, &mh);
    int cur_ok = sscanf(cfg_value("MKW_WINDOW_SIZE"), "%dx%d", &cw, &ch) == 2;
    bool listed = false;
    for (size_t i = 0; i < sizeof kSizes / sizeof *kSizes; i++) {
        int w = kSizes[i][0], h = kSizes[i][1];
        bool cur = cur_ok && w == cw && h == ch;
        if (have && (w > mw || h > mh) && !cur && !(w == 1920 && h == 1080)) continue;
        listed |= cur;
        sz[n][0] = w, sz[n][1] = h, n++;
    }
    if (cur_ok && !listed) {   // the current size where its area puts it
        int at = n;
        while (at > 0 && (long)sz[at - 1][0] * sz[at - 1][1] > (long)cw * ch) at--;
        memmove(sz[at + 1], sz[at], sizeof sz[0] * (n - at));
        sz[at][0] = cw, sz[at][1] = ch, n++;
    }
    char vals[MAX_SIZES][24], shown[MAX_SIZES][32];
    const char *pv[MAX_SIZES], *pn[MAX_SIZES];
    for (int i = 0; i < n; i++) {
        snprintf(vals[i], sizeof vals[i], "%dx%d", sz[i][0], sz[i][1]);
        snprintf(shown[i], sizeof shown[i], "%d × %d", sz[i][0], sz[i][1]);   // U+00D7, the multiplication sign
        pv[i] = vals[i];
        pn[i] = shown[i];
    }
    opt_choice_list("MKW_WINDOW_SIZE", UI_MENU_WINDOW_SIZE, pv, pn, n, UI_MENU_WINDOW_SIZE_DESC);
}

static void row_gpu(void) {
    const char *vals[MAX_GPUS + 1], *names[MAX_GPUS + 1];
    char idx[MAX_GPUS][4];
    vals[0] = "";
    names[0] = menu_text(UI_MENU_GPU_AUTO);
    for (int i = 0; i < g_gpu_n; i++) {
        snprintf(idx[i], sizeof idx[i], "%d", i);
        vals[i + 1] = idx[i];
        names[i + 1] = g_gpu_names[i];
    }
    opt_choice_list("MKW_GPU_INDEX", UI_MENU_GPU, vals, names, g_gpu_n + 1, UI_MENU_GPU_DESC);
}

// The game language: one entry per language TotK ships (a Switch code that names it), each under its own name. The
// Switch's Portuguese is not among them.
static void row_game_lang(void) {
    static const struct { const char *code; UiTextId name; } langs[] = {
        {"en-US", UI_MENU_LANGNAME_EN}, {"ja", UI_MENU_LANGNAME_JA}, {"fr", UI_MENU_LANGNAME_FR},
        {"de", UI_MENU_LANGNAME_DE}, {"it", UI_MENU_LANGNAME_IT}, {"es", UI_MENU_LANGNAME_ES},
        {"nl", UI_MENU_LANGNAME_NL}, {"ru", UI_MENU_LANGNAME_RU},
        {"zh-Hans", UI_MENU_LANGNAME_ZH_HANS}, {"zh-Hant", UI_MENU_LANGNAME_ZH_HANT}, {"ko", UI_MENU_LANGNAME_KO},
    };
    enum { N = sizeof langs / sizeof *langs + 1 };
    const char *vals[N], *names[N];
    vals[0] = "";
    names[0] = menu_text(UI_MENU_GAME_LANG_AUTO);
    const char *now = cfg_value("MKW_LANG");
    int cur = *now ? -1 : 0;
    for (int i = 1; i < N; i++) {
        vals[i] = langs[i - 1].code;
        names[i] = menu_text(langs[i - 1].name);
        if (*now && !strcmp(ui_text_lang(now), ui_text_lang(vals[i]))) cur = i;   // en-GB is English too
    }
    opt_choice_list("MKW_LANG", UI_MENU_GAME_LANG, vals, names, N, UI_MENU_GAME_LANG_DESC, cur);
}

// The menu's name of a TOTK_VK_PRESENT mode (gfx_window_present_mode); NULL for another.
static const char *present_name(const char *mode) {
    static const struct { const char *mode; UiTextId name; } names[] = {
        {"mailbox", UI_MENU_PRESENT_MAILBOX}, {"fifo", UI_MENU_PRESENT_VSYNC}, {"immediate", UI_MENU_PRESENT_IMMEDIATE},
    };
    for (const auto &n : names)
        if (!strcmp(n.mode, mode)) return menu_text(n.name);
    return nullptr;
}

// "in use: VSync" under the Presentation value when the swapchain got another mode than the setting (a driver without
// mailbox gets fifo); NULL when they agree or before the first swapchain.
static const char *present_note(void) {
    const char *got = gfx_window_present_mode(), *name = present_name(got);
    if (!*got || !strcmp(got, cfg_value("MKW_VK_PRESENT")) || !name) return nullptr;
    static char note[96];
    snprintf(note, sizeof note, menu_text(UI_MENU_PRESENT_IN_USE), name);
    return note;
}

// ---- the tabs (2026-11 rewrite): what a Switch 2 owner knows (Display, Sound, Controls, System) plus Advanced.
// No upscalers, frame generation or latency extensions: the game renders at its own resolution and the present only
// scales its image to the window.

// The frame limit: 60 as on the console, or 30.
static void row_frame_limit(void) {
    static const char *const vals[] = {"60", "30"};
    const char *names[] = {menu_text(UI_MENU_FPS_60), menu_text(UI_MENU_FPS_30)};
    const char *set = cfg_value("MKW_FPS_LIMIT");
    opt_choice_list("MKW_FPS_LIMIT", UI_MENU_FPS_LIMIT, vals, names, 2, UI_MENU_FRAME_LIMIT_DESC,
                    !strcmp(set, "30") ? 1 : 0);
}

// V-Sync: On is fifo (as the console), Off is immediate. A mailbox value from an older mkw.ini shows as On (no tearing).
static void row_vsync(void) {
    static const char *const vals[] = {"fifo", "immediate"};
    const char *names[] = {menu_text(UI_MENU_ON), menu_text(UI_MENU_OFF)};
    int cur = !strcmp(cfg_value("MKW_VK_PRESENT"), "immediate") ? 1 : 0;
    opt_choice_list("MKW_VK_PRESENT", UI_MENU_VSYNC, vals, names, 2, UI_MENU_VSYNC_DESC, cur, present_note());
}

// Rumble in four steps over TOTK_RUMBLE_STRENGTH (percent): the nearest step shows.
static void row_rumble(void) {
    static const char *const vals[] = {"0", "60", "100", "150"};
    const char *names[] = {menu_text(UI_MENU_OFF), menu_text(UI_MENU_RUMBLE_LOW), menu_text(UI_MENU_RUMBLE_NORMAL),
                           menu_text(UI_MENU_RUMBLE_STRONG)};
    int v = cfg_int("MKW_RUMBLE_STRENGTH");
    int cur = v <= 0 ? 0 : v < 80 ? 1 : v < 125 ? 2 : 3;
    opt_choice_list("MKW_RUMBLE_STRENGTH", UI_MENU_RUMBLE, vals, names, 4, UI_MENU_RUMBLE_DESC, cur);
}

static void tab_display(float scale) {
    opt_begin("##display", scale);
    static const UiTextId mode[] = {UI_MENU_MODE_DOCKED, UI_MENU_MODE_HANDHELD};
    opt_choice("MKW_OPERATION_MODE", UI_MENU_OPERATION_MODE, mode, 2, UI_MENU_CONSOLE_MODE_DESC);
    opt_toggle("MKW_FULLSCREEN", UI_MENU_FULLSCREEN, UI_MENU_FULLSCREEN_DESC);
    row_window_size();
    row_vsync();
    row_frame_limit();
    static const UiTextId perf[] = {UI_MENU_PERF_OFF, UI_MENU_PERF_FPS, UI_MENU_PERF_FULL};
    opt_choice("MKW_PERF_OVERLAY", UI_MENU_PERF_OVERLAY, perf, 3, UI_MENU_PERF_OVERLAY_DESC);
    opt_end();
}

static void tab_sound(float scale) {
    opt_begin("##sound", scale);
    opt_slider("MKW_VOLUME", UI_MENU_VOLUME, 0, 100, UI_MENU_VOLUME_DESC, 5, "%.0f %%");
    static const UiTextId channels[] = {UI_MENU_CHANNELS_AUTO, UI_MENU_CHANNELS_STEREO, UI_MENU_CHANNELS_51};
    opt_choice("MKW_AUDIO_CHANNELS", UI_MENU_AUDIO_CHANNELS, channels, 3, UI_MENU_AUDIO_CHANNELS_DESC);
    opt_end();
}

// The note under the layout. Xbox prompts (the Nintendo layout with the A/B and X/Y prompts swapped, romfs.c
// overlay_init) change the buttons at once but the prompts only at the next start: "files missing" without
// romfs-patches-xbox, "prompts change after restart" while the prompts in use differ from the choice. Automatic with a
// gamepad connected: "in use: Xbox". NULL otherwise.
static const char *pad_layout_note(void) {
    bool want = !strcasecmp(cfg_value("MKW_PAD_LAYOUT"), "xbox_prompts");
    if (want && !romfs_xbox_prompts_ready()) return menu_text(UI_MENU_PAD_LAYOUT_PROMPTS_MISSING);
    if (want != (romfs_xbox_prompts_active() != 0)) return menu_text(UI_MENU_PAD_LAYOUT_PROMPTS_RESTART);
    int got = input_pad_layout();
    if (got < 0 || strcmp(cfg_value("MKW_PAD_LAYOUT"), "auto")) return nullptr;
    static char note[96];
    snprintf(note, sizeof note, menu_text(UI_MENU_PAD_LAYOUT_IN_USE),
             menu_text(got == PAD_LAYOUT_XBOX ? UI_MENU_PAD_LAYOUT_XBOX : UI_MENU_PAD_LAYOUT_NINTENDO));
    return note;
}

static void tab_controls(float scale) {
    opt_begin("##controls", scale);
    static const UiTextId layout[] = {UI_MENU_PAD_LAYOUT_AUTO, UI_MENU_PAD_LAYOUT_NINTENDO, UI_MENU_PAD_LAYOUT_XBOX,
                                      UI_MENU_PAD_LAYOUT_XBOX_PROMPTS};
    opt_choice("MKW_PAD_LAYOUT", UI_MENU_PAD_LAYOUT, layout, 4, UI_MENU_PAD_LAYOUT_DESC, pad_layout_note());
    row_rumble();
    opt_toggle("MKW_MOTION", UI_MENU_MOTION, UI_MENU_MOTION_DESC);
    opt_slider("MKW_STICK_DEADZONE", UI_MENU_STICK_DEADZONE, 0, 50, UI_MENU_STICK_DEADZONE_DESC, 1, "%.0f %%");
    binds_rows(scale);
    opt_end();
    binds_after_page();
}

// ---- System: languages and the reset of every setting (with a confirmation)
static bool g_modal;   // the reset confirmation is open: B and Esc close it, not the menu; L / R stay on the page

// Every setting the menu shows back to its default (the environment's keys stay), written to mkw.ini.
static void reset_all(void) {
    const char *keys[160];
    int n = 0;
    for (const CfgMeta *m; n < 160 && (m = cfg_meta_at(n)); n++) keys[n] = m->key;
    int done = cfg_reset(keys, n);
    opt_save_soon();
    rt_log("menu: reset %d settings to their defaults", done);
}

static void tab_system(float scale) {
    opt_begin("##system", scale);
    row_game_lang();
    static const UiTextId langs[] = {UI_MENU_MENU_LANG_AUTO, UI_MENU_LANGNAME_EN, UI_MENU_LANGNAME_RU};
    opt_choice("MKW_MENU_LANG", UI_MENU_MENU_LANG, langs, 3, UI_MENU_MENU_LANG_DESC);
    static const UiTextId themes[] = {UI_MENU_THEME_LIGHT, UI_MENU_THEME_DARK};
    opt_choice("MKW_MENU_THEME", UI_MENU_THEME, themes, 2, UI_MENU_THEME_DESC);
    bool ask = opt_button("##reset", UI_MENU_RESET, UI_MENU_RESET_DESC);
    opt_end();
    if (ask) ImGui::OpenPopup("##reset_confirm");   // at the level BeginPopupModal looks it up

    // the confirmation, over the menu; Cancel has the cursor first
    g_modal = false;
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("##reset_confirm", nullptr,
                               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize)) {
        g_modal = true;
        ImGui::TextUnformatted(menu_text(UI_MENU_RESET_CONFIRM));
        ImGui::Spacing();
        const char *yes = menu_text(UI_MENU_RESET_YES), *no = menu_text(UI_MENU_RESET_NO);
        const ImGuiStyle &st = ImGui::GetStyle();
        float bw = fmaxf(ImGui::CalcTextSize(yes).x, ImGui::CalcTextSize(no).x) + st.FramePadding.x * 4;
        bool close = false;
        if (ImGui::Button(no, ImVec2(bw, 0))) close = true;
        if (ImGui::IsWindowAppearing()) ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button(yes, ImVec2(bw, 0))) {
            reset_all();
            close = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))
            close = true;
        if (close) {
            ImGui::CloseCurrentPopup();
            opt_focus_key("##reset");   // back on the row it came from
        }
        ImGui::EndPopup();
    }
}

// ---- Advanced: the graphics card, the memory mode and the bug report helpers
// A button's result shown in its value for a few seconds: 1 done (done_text), -1 failed.
struct ActionNote { uint64_t until; int result; };

static const char *action_value(const ActionNote &a, UiTextId done_text) {
    if (!a.result || plat_now_ns() > a.until) return "";
    return menu_text(a.result > 0 ? done_text : UI_MENU_ACTION_FAILED);
}

static void action_done(ActionNote &a, bool ok) { a = {plat_now_ns() + 3000000000ull, ok ? 1 : -1}; }

static void tab_advanced(float scale) {
    static ActionNote copy_note, log_note;
    opt_begin("##advanced", scale);
    opt_text(UI_MENU_ADV_WARNING, true);
    row_gpu();
    static const char *const mem_vals[] = {"import", "mirror"};
    const char *mem_names[] = {menu_text(UI_MENU_MEMORY_IMPORT_REC), menu_text(UI_MENU_MEMORY_MIRROR)};
    opt_choice_list("MKW_MEMORY", UI_MENU_MEMORY, mem_vals, mem_names, 2, UI_MENU_MEMORY_MODE_DESC);
    if (opt_button("##open_log", UI_MENU_OPEN_LOG, UI_MENU_OPEN_LOG_DESC, action_value(log_note, UI_MENU_OPEN_LOG))) {
        char dir[512];
        bool ok = sysinfo_log_folder(dir, sizeof dir) && plat_open_folder(dir);
        if (!ok) action_done(log_note, false);   // success shows as the opened folder
    }
    if (opt_button("##copy_info", UI_MENU_COPY_INFO, UI_MENU_COPY_INFO_DESC,
                   action_value(copy_note, UI_MENU_COPY_INFO_DONE))) {
        static char text[16384];
        action_done(copy_note, sysinfo_text(text, sizeof text) > 0 && plat_clipboard_set(text));
    }
    opt_end();
}

static const UiTextId kTabNames[TAB_COUNT] = {UI_MENU_TAB_DISPLAY, UI_MENU_TAB_SOUND, UI_MENU_TAB_CONTROLS,
                                              UI_MENU_TAB_SYSTEM, UI_MENU_TAB_ADVANCED};
static int g_tab;               // the tab shown
static int g_tab_request = -1;  // select this tab at the next layout; -1 none
static int g_focus_page = -1;   // this tab's first row takes the cursor once it is shown; -1 none

// ---- the window (Switch 2 system settings look, ui/theme.h): a title bar, the category list on the left (L / R or
// Q / E step it, a click picks one), the page with its help box on the right, and a footer with the controller hints and
// "Continue" / "Quit game". B on the gamepad and Esc close it.

// A round button glyph with its letter (the footer hints).
static void hint_glyph(ImDrawList *dl, ImVec2 c, float rad, const char *letter) {
    using namespace mkw_theme;
    dl->AddCircleFilled(c, rad, kText);
    ImFont *font = ImGui::GetFont();
    float fs = rad * 1.25f;
    ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, letter);
    dl->AddText(font, fs, ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), kHeader, letter);
}

// A pill glyph (L, R) for the category switch.
static float hint_pill(ImDrawList *dl, ImVec2 p, float h, const char *letter) {
    using namespace mkw_theme;
    ImFont *font = ImGui::GetFont();
    float fs = h * 0.62f;
    ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, letter);
    float w = ts.x + h * 0.9f;
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), kText, h * 0.5f);
    dl->AddText(font, fs, ImVec2(p.x + (w - ts.x) * 0.5f, p.y + (h - ts.y) * 0.5f), kHeader, letter);
    return w;
}

static void layout_menu(uint32_t w, uint32_t h, float scale) {
    using namespace mkw_theme;
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2((float)w, (float)h), kBackdrop);
    float ww = fminf(w * 0.86f, 1500 * scale), wh = fminf(h * 0.88f, 880 * scale);
    ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(ww, wh), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImFont *font = ImGui::GetFont();
    const ImGuiStyle &st = ImGui::GetStyle();
    ImVec2 o = ImGui::GetWindowPos();
    float W = ImGui::GetWindowWidth(), H = ImGui::GetWindowHeight(), fs = ImGui::GetFontSize();
    float pad = 30 * scale, rounding = st.WindowRounding;
    float head_h = floorf(fs * 3.1f), foot_h = floorf(fs * 3.3f), side_w = floorf(fminf(W * 0.26f, 380 * scale));
    float body_y = o.y + head_h, body_h = H - head_h - foot_h;

    // title bar: an accent bar, "Mario Kart World", the page title, and the version on the right
    dl->AddRectFilled(o, ImVec2(o.x + W, o.y + head_h), kHeader, rounding, ImDrawFlags_RoundCornersTop);
    dl->AddLine(ImVec2(o.x, o.y + head_h - 0.5f), ImVec2(o.x + W, o.y + head_h - 0.5f), kDivider, 1.0f);
    float tfs = fs * 1.3f, ty = o.y + (head_h - tfs) * 0.5f;
    dl->AddRectFilled(ImVec2(o.x + pad, o.y + head_h * 0.27f), ImVec2(o.x + pad + 6 * scale, o.y + head_h * 0.73f),
                      kAccent, 3 * scale);
    float tx = o.x + pad + 20 * scale;
    const char *brand = "Mario Kart World";
    dl->AddText(font, tfs, ImVec2(tx, ty), kText, brand);
    tx += font->CalcTextSizeA(tfs, FLT_MAX, 0, brand).x + 14 * scale;
    dl->AddText(font, fs, ImVec2(tx, o.y + (head_h - fs) * 0.5f + fs * 0.08f), kTextSoft, menu_text(UI_MENU_TITLE));
    char version[96];
    const char *game = "1.8.0";   // this project builds the 1.8.0 update (tools/mkw_cli.py prepare --i-confirm-update-180)
    char gbuf[16];
    if (rt_game_version) {
        snprintf(gbuf, sizeof gbuf, "%d.%d.%d", rt_game_version / 100, rt_game_version / 10 % 10, rt_game_version % 10);
        game = gbuf;
    }
    snprintf(version, sizeof version, menu_text(UI_MENU_VERSION), rt_port_version, game);
    float vw = font->CalcTextSizeA(fs * 0.85f, FLT_MAX, 0, version).x;
    dl->AddText(font, fs * 0.85f, ImVec2(o.x + W - pad - vw, o.y + (head_h - fs * 0.85f) * 0.5f), kTextDim, version);

    // category list
    dl->AddRectFilled(ImVec2(o.x, body_y), ImVec2(o.x + side_w, body_y + body_h), kSidebar);
    dl->AddLine(ImVec2(o.x + side_w - 0.5f, body_y), ImVec2(o.x + side_w - 0.5f, body_y + body_h), kDivider, 1.0f);
    int step = g_modal || binds_modal()                                                                  ? 0
             : ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_E, false) ? 1
             : ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_Q, false) ? -1
                                                                                                         : 0;
    if (step) g_tab_request = (g_tab + step + TAB_COUNT) % TAB_COUNT;
    float ih = floorf(fs * 2.4f), ix = o.x + 14 * scale, iw = side_w - 28 * scale, iy = body_y + 18 * scale;
    for (int k = 0; k < TAB_COUNT; k++) {
        ImVec2 a(ix, iy + k * (ih + 6 * scale)), b(ix + iw, a.y + ih);
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(k);
        // not in the D-pad navigation (L / R switch pages); the mouse picks one
        if (ImGui::InvisibleButton("##cat", ImVec2(iw, ih)) && !g_modal && !binds_modal()) g_tab_request = k;
        bool hov = ImGui::IsItemHovered();
        ImGui::PopID();
        bool sel = k == (g_tab_request >= 0 ? g_tab_request : g_tab);
        if (sel) {
            dl->AddRectFilled(a, b, kRowSel, 8 * scale);
            dl->AddRectFilled(ImVec2(a.x + 8 * scale, a.y + ih * 0.22f), ImVec2(a.x + 13 * scale, b.y - ih * 0.22f),
                              kAccent, 3 * scale);
        } else if (hov) {
            dl->AddRectFilled(a, b, kRowHover, 8 * scale);
        }
        dl->AddText(ImVec2(a.x + 26 * scale, a.y + (ih - fs) * 0.5f), sel ? kAccentText : kText, menu_text(kTabNames[k]));
    }
    // L / R under the list: the controller's page switch
    {
        float ph = fs * 1.1f, py = body_y + body_h - ph - 20 * scale, px = ix + 12 * scale;
        px += hint_pill(dl, ImVec2(px, py), ph, "L") + 8 * scale;
        hint_pill(dl, ImVec2(px, py), ph, "R");
    }
    if (g_tab_request >= 0) {
        g_tab = g_tab_request;
        g_focus_page = g_tab_request;   // the new page's first row takes the cursor
        g_tab_request = -1;
    }

    // the page: a direct NavFlattened child of this window (option_rows.cpp), so the D-pad moves between its rows and
    // the footer buttons without a second child level in between
    ImGui::SetCursorScreenPos(ImVec2(o.x + side_w + pad, body_y + pad * 0.7f));
    if (g_focus_page == g_tab) {
        opt_focus_first();
        g_focus_page = -1;
    }
    opt_reserve_bottom(foot_h + pad * 0.5f);
    opt_reserve_right(pad);
    switch (g_tab) {
    case TAB_DISPLAY: tab_display(scale); break;
    case TAB_SOUND: tab_sound(scale); break;
    case TAB_CONTROLS: tab_controls(scale); break;
    case TAB_SYSTEM: tab_system(scale); break;
    case TAB_ADVANCED: tab_advanced(scale); break;
    }

    // footer: the hints (or the save failure) on the left, "Continue" and "Quit game" on the right
    float fy = o.y + H - foot_h;
    dl->AddRectFilled(ImVec2(o.x, fy), ImVec2(o.x + W, o.y + H), kHeader, rounding, ImDrawFlags_RoundCornersBottom);
    dl->AddLine(ImVec2(o.x, fy + 0.5f), ImVec2(o.x + W, fy + 0.5f), kDivider, 1.0f);
    float cy = fy + foot_h * 0.5f, rad = fs * 0.62f, hx = o.x + pad;
    if (!cfg_last_save_ok()) {   // stays until a save succeeds; the settings hold for the session
        dl->AddText(ImVec2(hx, cy - fs * 0.5f), kWarn, menu_text(UI_MENU_SAVE_FAILED));
    } else {
        const struct { const char *glyph; UiTextId text; } hints[] = {
            {"A", UI_MENU_HINT_OK}, {"B", UI_MENU_HINT_BACK}};
        for (const auto &e : hints) {
            hint_glyph(dl, ImVec2(hx + rad, cy), rad, e.glyph);
            hx += rad * 2 + 8 * scale;
            dl->AddText(ImVec2(hx, cy - fs * 0.5f), kTextSoft, menu_text(e.text));
            hx += ImGui::CalcTextSize(menu_text(e.text)).x + 26 * scale;
        }
    }
    const char *cont = menu_text(UI_MENU_CONTINUE), *quit = menu_text(UI_MENU_QUIT);
    float bw = fmaxf(ImGui::CalcTextSize(cont).x, ImGui::CalcTextSize(quit).x) + st.FramePadding.x * 4;
    float bh = ImGui::GetFrameHeight();
    ImGui::SetCursorScreenPos(ImVec2(o.x + W - pad - bw * 2 - 14 * scale, cy - bh * 0.5f));
    if (ImGui::Button(cont, ImVec2(bw, 0))) set_open(0);
    ImGui::SameLine(0, 14 * scale);
    if (ImGui::Button(quit, ImVec2(bw, 0))) gfx_window_request_close();

    bool was_modal = g_modal || binds_modal();
    if (g_tab != TAB_SYSTEM) g_modal = false;
    if (!was_modal && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)))
        set_open(0);
    ImGui::End();
}

static int g_save_pending;   // a changed setting still waits for its write to mkw.ini (opt_save_tick)

// ---- the performance overlay (TOTK_PERF_OVERLAY), in the top right corner. "fps": the frame rate over the presents of
// the last half second. "full": a panel of perfmon.c's numbers, in the style of SavageWindRecomp's perf_overlay_draw
// (menu.mm): the machine's load and temperatures, the frame rate and frame times with a graph, the render thread's waits.
// Off, it costs a cfg_version load per present.
static unsigned g_perf_ver = ~0u;
static int g_perf_mode;                  // 0 off, 1 fps, 2 full
static std::atomic<int> g_perf_save;     // F10 changed the mode: mkw.ini is written on the present thread
static uint64_t g_perf_last_ns, g_perf_win_ns, g_perf_max_ns;
static int g_perf_frames;
static char g_perf_text[96];
static char g_perf_fps_num[32];
static char g_perf_ms_num[32];
static uint32_t g_dt_ring[1000];
static int g_dt_count = 0;
static int g_dt_head = 0;
static char g_perf_1pct_num[32];
static char g_perf_01pct_num[32];

static int perf_mode_of(const char *s) { return !strcmp(s, "full") ? 2 : !strcmp(s, "fps") ? 1 : 0; }

extern "C" void menu_perf_cycle(void) {
    static const char *const modes[] = {"off", "fps", "full"};
    cfg_set("MKW_PERF_OVERLAY", modes[(perf_mode_of(cfg_value("MKW_PERF_OVERLAY")) + 1) % 3]);
    g_perf_save.store(1);
}

// The device the game renders on, for perfmon's GPU counters (by LUID), NVML (by name) and ADL (by PCI bus, else
// name); once, before full mode.
static void perf_gpu_once(const MenuTarget *t) {
    static bool done;
    if (done) return;
    done = true;
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(t->phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = (VkExtensionProperties *)calloc(ne ? ne : 1, sizeof *ep);
    bool has_bus = false;
    if (ep && vkEnumerateDeviceExtensionProperties(t->phys, NULL, &ne, ep) >= 0)
        for (uint32_t i = 0; i < ne && !has_bus; i++) has_bus = !strcmp(ep[i].extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    free(ep);
    VkPhysicalDevicePCIBusInfoPropertiesEXT bus = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, has_bus ? &bus : nullptr};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vkGetPhysicalDeviceProperties2(t->phys, &p2);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(t->phys, &mp);
    uint64_t vram = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
        if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += mp.memoryHeaps[h].size;
    perfmon_set_gpu(p2.properties.deviceName, id.deviceLUIDValid ? id.deviceLUID : NULL, has_bus ? (int)bus.pciBus : -1,
                    (float)(vram >> 20));
}

static void perf_tick(uint64_t now, const MenuTarget *t) {
    if (cfg_version() != g_perf_ver) {
        g_perf_ver = cfg_version();
        int m = perf_mode_of(cfg_value("MKW_PERF_OVERLAY"));
        if (m != g_perf_mode) {
            g_perf_last_ns = 0, g_perf_text[0] = 0;
            g_perf_fps_num[0] = 0;
            g_perf_1pct_num[0] = 0;
            g_perf_01pct_num[0] = 0;
            if (m == 2) perf_gpu_once(t);
            perfmon_mode(m);
        }
        g_perf_mode = m;
    }
    if (g_perf_mode != 1) return;
    uint64_t d = now - g_perf_last_ns;
    if (!g_perf_last_ns || d > 2000000000ull) {   // first frame, or a stall the window should not average in
        g_perf_win_ns = g_perf_max_ns = 0;
        g_perf_frames = 0;
    } else {
        g_perf_win_ns += d;
        g_perf_frames++;
        if (d > g_perf_max_ns) g_perf_max_ns = d;

        // Record into rolling 1000-frame ring buffer for 1% and 0.1% low calculation
        g_dt_ring[g_dt_head] = (uint32_t)(d > 2000000000ull ? 2000000000ull : d);
        g_dt_head = (g_dt_head + 1) % 1000;
        if (g_dt_count < 1000) g_dt_count++;

        if (g_perf_win_ns >= 500000000ull) {
            double base_fps = (g_perf_frames * 1e9) / (double)g_perf_win_ns;
            double base_ms = ((double)g_perf_win_ns / (double)g_perf_frames) / 1e6;

            if (base_fps >= 100.0) snprintf(g_perf_fps_num, sizeof g_perf_fps_num, "%.0f FPS", base_fps);
            else snprintf(g_perf_fps_num, sizeof g_perf_fps_num, "%.1f FPS", base_fps);
            snprintf(g_perf_ms_num, sizeof g_perf_ms_num, "%.1f ms", base_ms);
            if (g_dt_count >= 10) {
                uint32_t sorted_dt[1000];
                int n_samples = g_dt_count;
                memcpy(sorted_dt, g_dt_ring, n_samples * sizeof(uint32_t));
                std::sort(sorted_dt, sorted_dt + n_samples);
                int idx_1p = (int)(n_samples * 0.99f), idx_01p = (int)(n_samples * 0.999f);
                if (idx_1p >= n_samples) idx_1p = n_samples - 1;
                if (idx_01p >= n_samples) idx_01p = n_samples - 1;
                double low_1p = sorted_dt[idx_1p] ? 1e9 / (double)sorted_dt[idx_1p] : base_fps;
                double low_01p = sorted_dt[idx_01p] ? 1e9 / (double)sorted_dt[idx_01p] : base_fps;
                snprintf(g_perf_1pct_num, sizeof g_perf_1pct_num, "1%% %.0f", low_1p);
                snprintf(g_perf_01pct_num, sizeof g_perf_01pct_num, "0.1%% %.0f", low_01p);
            } else {
                g_perf_1pct_num[0] = g_perf_01pct_num[0] = 0;
            }
            snprintf(g_perf_text, sizeof g_perf_text, "%s", g_perf_fps_num);
            g_perf_win_ns = g_perf_max_ns = 0;
            g_perf_frames = 0;
        }
    }
    g_perf_last_ns = now;
}

// The panel keeps its width while the numbers change: every column is as wide as the widest text it can show, measured
// on samples of 8s (Roboto's digits share one width), and the numbers are right-aligned in it.
struct PerfCol { UiTextId fmt; float sample; };

static float perf_text_w(UiTextId fmt, float v) {
    char b[64];
    snprintf(b, sizeof b, menu_text(fmt), v);
    return ImGui::CalcTextSize(b).x;
}

// A fixed column as wide as the widest of the formats (with their samples) and labels given.
static void perf_column(const char *id, const PerfCol *cols, int n, const UiTextId *labels = nullptr, int nl = 0) {
    float w = ImGui::CalcTextSize(menu_text(UI_MENU_OV_NONE)).x;
    for (int i = 0; i < n; i++) w = fmaxf(w, perf_text_w(cols[i].fmt, cols[i].sample));
    for (int i = 0; i < nl; i++) w = fmaxf(w, ImGui::CalcTextSize(menu_text(labels[i])).x);
    ImGui::TableSetupColumn(id, ImGuiTableColumnFlags_WidthFixed, w);
}

static void perf_right(const char *s, bool dim) {
    float pad = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(s).x;
    if (pad > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
    if (dim) ImGui::TextDisabled("%s", s);
    else ImGui::TextUnformatted(s);
}

// One cell of the full panel's table: fmt (a catalog format, one number) with v, or "-" when the machine gives none.
static void perf_cell(UiTextId fmt, float v) {
    ImGui::TableNextColumn();
    if (v < 0) return perf_right(menu_text(UI_MENU_OV_NONE), true);
    char b[64];
    snprintf(b, sizeof b, menu_text(fmt), v);
    perf_right(b, false);
}

static void perf_label(UiTextId label, const ImVec4 &c) {
    ImGui::TableNextColumn();
    ImGui::TextColored(c, "%s", menu_text(label));
}

static float perf_gb(float mb) { return mb < 0 ? -1 : mb / 1024; }

// The picture lines: the window, the game image the present shows and the 3D scene it holds (resolution.c), how the
// present scales the image to the window (gfx_present.c, the last present), the frame limit and the presentation mode
// in use.
static void perf_picture(const MenuTarget *t, const ImVec4 &c) {
    char v[5][64];
    snprintf(v[0], sizeof v[0], menu_text(UI_MENU_OV_FMT_SIZE), t->width, t->height);
    snprintf(v[1], sizeof v[1], menu_text(UI_MENU_OV_FMT_SIZE), (unsigned)t->image_w, (unsigned)t->image_h);
    snprintf(v[2], sizeof v[2], "%s", menu_text(t->scaler ? UI_MENU_OV_SCALE_BILINEAR : UI_MENU_OV_SCALE_NONE));
    double hz = frame_step_limit_hz();
    snprintf(v[3], sizeof v[3], "%s", menu_text(UI_MENU_OV_NONE));
    if (hz > 0) snprintf(v[3], sizeof v[3], menu_text(UI_MENU_OV_FPS), hz);
    const char *pm = present_name(gfx_window_present_mode());
    snprintf(v[4], sizeof v[4], "%s", pm ? pm : menu_text(UI_MENU_OV_NONE));
    static const UiTextId labels[5] = {UI_MENU_OV_WINDOW, UI_MENU_OV_OUTPUT, UI_MENU_OV_SCALING, UI_MENU_OV_LIMIT,
                                       UI_MENU_OV_PRESENT};
    if (!ImGui::BeginTable("##perfp", 2, ImGuiTableFlags_SizingFixedFit)) return;
    float vw = 0;
    for (auto &x : v) vw = fmaxf(vw, ImGui::CalcTextSize(x).x);
    ImGui::TableSetupColumn("##p0", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("##p1", ImGuiTableColumnFlags_WidthFixed, vw);
    for (int i = 0; i < 5; i++) {
        ImGui::TableNextColumn();
        ImGui::TextColored(c, "%s", menu_text(labels[i]));
        ImGui::TableNextColumn();
        perf_right(v[i], false);
    }
    ImGui::EndTable();
}

static void perf_full_draw(const MenuTarget *t, float scale) {
    float w = (float)t->width;
    PerfStats s;
    perfmon_snapshot(&s);
    ImGui::SetNextWindowPos(ImVec2(w - 12 * scale, 12 * scale), ImGuiCond_Always, ImVec2(1, 0));
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::Begin("##perf", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                                        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings);
    // under the menu while it is open (the settings stay readable), else over everything
    if (g_open.load(std::memory_order_relaxed)) ImGui::BringWindowToDisplayBack(ImGui::GetCurrentWindow());
    else ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    ImGui::SetWindowFontScale(0.72f);
    const ImVec4 gpu_c(0.55f, 0.83f, 0.31f, 1), cpu_c(0.18f, 0.59f, 0.80f, 1), ram_c(0.80f, 0.45f, 0.70f, 1),
        fps_c(0.95f, 0.95f, 0.95f, 1), wait_c(0.93f, 0.76f, 0.33f, 1), pic_c(0.62f, 0.70f, 0.95f, 1);
    if (ImGui::BeginTable("##perft", 4, ImGuiTableFlags_SizingFixedFit)) {
        static const UiTextId labels[] = {UI_MENU_OV_GPU, UI_MENU_OV_VRAM, UI_MENU_OV_CPU, UI_MENU_OV_RAM};
        static const PerfCol c0[] = {{UI_MENU_OV_FPS, 888}};
        static const PerfCol c1[] = {{UI_MENU_OV_FMT_PCT, 100}, {UI_MENU_OV_FMT_GB, 888.8f}, {UI_MENU_OV_FRAME_MS, 888.8f}};
        static const PerfCol c2[] = {{UI_MENU_OV_FMT_TEMP, 188}, {UI_MENU_OV_FMT_GB_OF, 888.8f}, {UI_MENU_OV_MAX_MS, 8888.8f}};
        static const PerfCol c3[] = {{UI_MENU_OV_FMT_POWER, 888}, {UI_MENU_OV_FMT_MHZ, 8888}, {UI_MENU_OV_GAME_PCT, 100},
                                     {UI_MENU_OV_GAME_GB, 888.8f}};
        perf_column("##c0", c0, 1, labels, 4);
        perf_column("##c1", c1, 3);
        perf_column("##c2", c2, 3);
        perf_column("##c3", c3, 4);
        perf_label(UI_MENU_OV_GPU, gpu_c);
        perf_cell(UI_MENU_OV_FMT_PCT, s.gpu_pct);
        perf_cell(UI_MENU_OV_FMT_TEMP, s.gpu_temp_c);
        perf_cell(UI_MENU_OV_FMT_POWER, s.gpu_power_w);
        perf_label(UI_MENU_OV_VRAM, gpu_c);
        perf_cell(UI_MENU_OV_FMT_GB, perf_gb(s.vram_used_mb));
        perf_cell(UI_MENU_OV_FMT_GB_OF, perf_gb(s.vram_total_mb));
        perf_cell(UI_MENU_OV_FMT_MHZ, s.gpu_clock_mhz);
        perf_label(UI_MENU_OV_CPU, cpu_c);
        perf_cell(UI_MENU_OV_FMT_PCT, s.cpu_total_pct);
        perf_cell(UI_MENU_OV_FMT_TEMP, s.cpu_temp_c);
        perf_cell(UI_MENU_OV_GAME_PCT, s.cpu_game_pct);
        perf_label(UI_MENU_OV_RAM, ram_c);
        perf_cell(UI_MENU_OV_FMT_GB, perf_gb(s.ram_used_mb));
        perf_cell(UI_MENU_OV_FMT_GB_OF, perf_gb(s.ram_total_mb));
        perf_cell(UI_MENU_OV_GAME_GB, perf_gb(s.ram_game_mb));
        ImGui::TableNextColumn();   // the frame row: no label, the rate in its place
        char b[64];
        snprintf(b, sizeof b, menu_text(UI_MENU_OV_FPS), s.fps);
        ImGui::TextColored(fps_c, "%s", b);
        perf_cell(UI_MENU_OV_FRAME_MS, s.frame_ms_avg);
        perf_cell(UI_MENU_OV_MAX_MS, s.frame_ms_max);
        ImGui::EndTable();
    }
    float top = s.frame_ms_max > 50 ? s.frame_ms_max : 50;
    // as wide as the panel the tables size; the plot does not count toward that width, so the panel can still shrink
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    float max_x = win->DC.CursorMaxPos.x;
    ImGui::PlotLines("##ft", s.frame_ms, PERF_HISTORY, 0, nullptr, 0, top, ImVec2(-FLT_MIN, 48 * scale));
    win->DC.CursorMaxPos.x = max_x;
    // per frame, averaged over the last half second; the labels' column stretches, so the numbers end at the panel's
    // right edge, and the table still asks for no more than its widest label and number
    if (ImGui::BeginTable("##perfw", 2, ImGuiTableFlags_SizingFixedFit)) {
        static const PerfCol c1[] = {{UI_MENU_OV_FMT_MS, 888.88f}, {UI_MENU_OV_PER_FRAME, 888.8f}};
        ImGui::TableSetupColumn("##w0", ImGuiTableColumnFlags_WidthStretch);
        perf_column("##w1", c1, 2);
        perf_label(UI_MENU_OV_REPLAY, wait_c);
        perf_cell(UI_MENU_OV_FMT_MS, s.replay_ms);
        perf_label(UI_MENU_OV_GPU_TIME, wait_c);
        perf_cell(UI_MENU_OV_FMT_MS, s.gpu_ms);
        perf_label(UI_MENU_OV_PIPE_WAIT, wait_c);
        perf_cell(UI_MENU_OV_FMT_MS, s.pipe_wait_ms);
        perf_label(UI_MENU_OV_SHADER_WAIT, wait_c);
        perf_cell(UI_MENU_OV_FMT_MS, s.xlate_ms);
        perf_label(UI_MENU_OV_TEX_UPLOADS, wait_c);
        perf_cell(UI_MENU_OV_PER_FRAME, s.tex_uploads);
        ImGui::EndTable();
    }
    perf_picture(t, pic_c);
    const char *pname = input_gamepad_name();
    if (pname) {
        int pct = -1, chg = 0;
        int hp = input_pad_power_info(&pct, &chg);
        if (hp && pct >= 0) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1), "[Controller] %s [%s %d%%]", pname, chg ? "Charging" : "Battery", pct);
        } else {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1), "[Controller] %s", pname);
        }
    }
    if (s.gpu_name[0]) ImGui::TextDisabled("%s", s.gpu_name);
    if (s.cpu_name[0]) ImGui::TextDisabled("%s", s.cpu_name);
    ImGui::End();
}

static void perf_draw(const MenuTarget *t, float scale) {
    if (g_perf_mode == 2) return perf_full_draw(t, scale);
    float w = (float)t->width;
    if (!g_perf_mode || !g_perf_fps_num[0]) return;
    // with the menu open: under its window, over the dimming layout_menu drew before
    ImDrawList *dl = g_open.load(std::memory_order_relaxed) ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
    ImFont *font = ImGui::GetFont();
    float font_sz = 14.5f * scale, pad_x = 9.0f * scale, pad_y = 5.0f * scale, gap = 8.0f * scale;
    const char *parts[4] = {g_perf_fps_num, g_perf_1pct_num, g_perf_01pct_num, g_perf_ms_num};
    static const ImU32 cols[4] = {IM_COL32(255, 255, 255, 255), IM_COL32(210, 225, 245, 225), IM_COL32(180, 205, 235, 200),
                                  IM_COL32(160, 185, 215, 230)};
    float tw = 0;
    int n = 0;
    for (const char *p : parts)
        if (p[0]) tw += (n++ ? gap : 0) + font->CalcTextSizeA(font_sz, FLT_MAX, -1.0f, p).x;
    ImVec2 p_min(w - 12.0f * scale - tw - pad_x * 2, 10.0f * scale);
    ImVec2 p_max(p_min.x + tw + pad_x * 2, p_min.y + font_sz + pad_y * 2);
    dl->AddRectFilled(p_min, p_max, IM_COL32(12, 16, 24, 210), 4.0f * scale);
    float x = p_min.x + pad_x;
    for (int i = 0; i < 4; i++) {
        if (!parts[i][0]) continue;
        dl->AddText(font, font_sz, ImVec2(x, p_min.y + pad_y), cols[i], parts[i]);
        x += font->CalcTextSizeA(font_sz, FLT_MAX, -1.0f, parts[i]).x + gap;
    }
}

extern "C" int menu_overlay_frame(uint64_t frame, const MenuTarget *t, const char **shot) {
    *shot = NULL;
    if (!g_test.parsed) test_parse();
    if (g_test.next_set < g_test.n_sets && frame >= g_test.sets[g_test.next_set].frame) test_apply_sets(frame);
    if (g_perf_save.exchange(0)) cfg_save();
    perf_tick(plat_now_ns(), t);
    bool open = g_open.load(std::memory_order_relaxed);
    if (!open) {
        if (g_save_pending) g_save_pending = opt_save_tick(1);   // closed: the last change goes to mkw.ini now
        if (g_test.at && !g_test.opened && frame >= g_test.at) {
            g_test.opened = frame;
            g_test.page = 0;
            g_tab_request = 0;
            set_open(1);
            open = true;
        } else if (!g_perf_mode) {
            return 0;
        }
    }
    if (g_failed_flag()) {
        set_open(0);
        return 0;
    }
    uint64_t t0 = plat_now_ns();
    if (!g_ctx_ok) make_context();
    float scale = menu_scale(t->height);
    if (!vk_up(t) || (g_font_scale != scale && !fonts_up(scale))) {
        rt_log("menu: Dear ImGui Vulkan back end set-up failed; the menu stays closed");
        g_failed.store(1);
        set_open(0);
        return 0;
    }
    static unsigned theme_ver = ~0u;   // the cfg_version the style's theme was read at
    if (cfg_version() != theme_ver) {
        theme_ver = cfg_version();
        static int dark_now = -1;
        int dark = !strcmp(cfg_value("MKW_MENU_THEME"), "dark");
        if (dark != dark_now) {   // a theme switch rebuilds the style at the current scale
            dark_now = dark;
            style_init(g_base_style);
            g_style_scale = 0;
        }
    }
    if (g_style_scale != scale) {
        ImGui::GetStyle() = g_base_style;
        ImGui::GetStyle().ScaleAllSizes(scale);
        g_style_scale = scale;
    }
    list_gpus(t->instance);
    ImGuiIO &io = ImGui::GetIO();
    int reset = g_reset.exchange(0);
    if (reset) {
        io.ClearInputKeys();
        io.ClearInputMouse();
        g_last_ns = 0;
        g_focus_page = g_tab_request >= 0 ? g_tab_request : g_tab;   // opening: the first row has the cursor
        if (GImGui->OpenPopupStack.Size) ImGui::ClosePopupToLevel(0, false);   // a confirmation left open on closing
        g_modal = false;
    }
    io.DisplaySize = ImVec2((float)t->width, (float)t->height);
    double dt = g_last_ns ? (t0 - g_last_ns) / 1e9 : 1 / 60.0;
    io.DeltaTime = (float)(dt < 1e-4 ? 1e-4 : dt > 0.1 ? 0.1 : dt);
    g_last_ns = t0;
    if (!open) {   // the overlay alone: no input reaches ImGui
    } else if (g_test.page < 0) {
        drain_events(io);
        pad_events(io, reset);
    } else {   // the test hook's pages: the user's input in a background test window must not change settings
        plat_mutex_lock(&ev_lock);
        ev_n = 0;
        plat_mutex_unlock(&ev_lock);
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (open) layout_menu(t->width, t->height, scale);
    perf_draw(t, scale);
    ImGui::Render();
    if (open) g_save_pending = opt_save_tick(0);
    g_cost_ns += plat_now_ns() - t0;
    if (g_test.page >= 0 && frame >= g_test.opened + g_test.per_page * (uint64_t)(g_test.page + 1)) {
        static char path[512];   // this frame's image goes to a file; then the next tab, or the menu closes
        const char *dir = cfg_get("MKW_MENU_TEST_DIR");
        snprintf(path, sizeof path, "%s/menu-test-%llu-%d.png", dir ? dir : ".", (unsigned long long)frame, g_test.page);
        *shot = path;
        if (++g_test.page == TAB_COUNT) {
            g_test.page = -1;
            set_open(0);   // the next frame lays out nothing
        } else {
            g_tab_request = g_test.page;
        }
    }
    return ImGui::GetDrawData()->TotalVtxCount > 0;
}

extern "C" void menu_overlay_draw(VkCommandBuffer cb) {
    uint64_t t0 = plat_now_ns();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
    g_cost_ns += plat_now_ns() - t0;
    if (++g_cost_frames == 600) {
        rt_log("menu: %.3f ms CPU per open frame (layout and recording, mean of 600)", g_cost_ns / 600 / 1e6);
        g_cost_ns = g_cost_frames = 0;
    }
}

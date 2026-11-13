// Adapted from MacNativeGames games/totk/runtime/services.c (lines 208-302), applet_host.c, account.c, time_host.c and
// shared_font.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames contributors).
// System service overrides: nn::oe, applet, am, prepo, nfp, err, aoc, pctl, diag,
// account, time and pl, and the six observers. Nothing here talks to a Horizon service; each call is answered by the
// host. Stub rule: queries and initialisation succeed with neutral values, data paths stop with a named fatal.
//
// Environment:
//   TOTK_LANG=<Switch language code>   the game's language as given ("ja", "en-GB", "zh-Hant", ...)
//   TOTK_LOCALE=<BCP 47 tag>           else the language for this locale ("pt-BR"), else the Windows user locale;
//                                      a language the game does not ship becomes en-US
//   TOTK_OPERATION_MODE=docked|handheld   operation and performance mode reported to the game (default docked)
//   TOTK_ASSUME_FOCUS=1|0              1 (default, the Game tab's row): GetCurrentFocusState always reports in focus,
//                                      whatever the window says; 0: out of focus while the window is, as on a Switch
//   TOTK_SHARED_FONTS=<dir>            nn::pl shared fonts (tools/extract_fonts.py output), default
//                                      <romfs>/../../firmware/shared-fonts
#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "aoc_romfs.h"
#include "cfg.h"
#include "dispatch.h"
#include "fs.h"
#include "host.h"
#include "ov.h"
#include "platform/platform.h"

// Soft-catch near-null AVs. --wrap does not cover rt_dispatch (table holds the real lift), so a
// process-wide VEH skips the faulting host insn when the address is near-null / poison. Overnight
// headless can finish shots; real high-address bugs still FATAL.
#ifdef _WIN32
static jmp_buf s_mkw_soft_av_jmp;
static volatile int s_mkw_soft_av_armed;
static PVOID s_mkw_soft_av_veh;
static volatile int s_mkw_soft_skip_on;

// Minimal x64 length for the REX+ModRM loads/stores the lifts emit for LDR/STR of a field.
static int mkw_x64_insn_len(const uint8_t *p) {
    int i = 0;
    while (i < 4 && ((p[i] >= 0x40 && p[i] <= 0x4f) || p[i] == 0x66 || p[i] == 0x67 || p[i] == 0xf0 ||
                     p[i] == 0xf2 || p[i] == 0xf3))
        i++;
    if (p[i] == 0x0f) return -1;
    uint8_t op = p[i++];
    // Opcodes with ModRM used by the recompiler for guest LDR/STR of object fields.
    int has_modrm = op == 0x63 || op == 0x89 || op == 0x8b || op == 0x8d || op == 0x85 || op == 0x3b ||
                    op == 0x39 || op == 0x01 || op == 0x03 || op == 0x29 || op == 0x2b || op == 0x31 ||
                    op == 0x33 || op == 0x87 || (op >= 0x80 && op <= 0x83);
    if (!has_modrm) return -1;
    uint8_t modrm = p[i++];
    int mod = modrm >> 6, rm = modrm & 7;
    if (mod == 3) return i;   // reg-reg: should not fault
    if (rm == 4) i++;         // SIB
    if (mod == 1) i += 1;
    else if (mod == 2 || (mod == 0 && rm == 5)) i += 4;
    return i;
}

static LONG CALLBACK mkw_soft_av_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    if (ep->ExceptionRecord->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    uint64_t addr = (uint64_t)ep->ExceptionRecord->ExceptionInformation[1];
    ULONG_PTR av_kind = ep->ExceptionRecord->ExceptionInformation[0];   // 0 read, 1 write
    int near_null = addr < 0x10000ull;
    int poison = addr == ~0ull || addr >= 0x8000000000ull;
    // n29: write 0x4cd2e720 — a guest heap pointer with the high bits cleared (should be
    // 0x2…/0x24…). Unmapped low-4GiB addresses get the same skip as near-null.
    int truncated = !near_null && !poison && addr >= 0x10000ull && addr < 0x100000000ull &&
                    !plat_mem_readable(addr, 1);
    // n34: write into main .text/.rdata (0x70027a76b8). Only soft-skip RO stores inside the guest
    // image — a broad VirtualQuery RO check also matched live heap holes (0x5570… in n44) and
    // the title exited after two series frames.
    int ro_write = 0;
    if (!near_null && !poison && !truncated && av_kind == 1 && addr >= 0x7000000000ull &&
        addr < 0x7100000000ull) {
#if defined(_WIN32)
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery((const void *)(uintptr_t)addr, &mbi, sizeof mbi) && mbi.State == MEM_COMMIT) {
            DWORD p = mbi.Protect & 0xff;
            if (p == PAGE_READONLY || p == PAGE_EXECUTE || p == PAGE_EXECUTE_READ) ro_write = 1;
        } else {
            ro_write = 1;   // unmapped hole inside the guest image
        }
#endif
    }
    // n58/n79: unmapped hole past 4GiB (write 0x11d… / read 0x2a410752e7). Cap so a tight loop
    // cannot spin forever. Reads and writes both — foliage culling often AVs on a stale GPU VA.
    // n136: 0x1800000013 (tagged/stale VA below the 0x80_0000_0000 poison line) burned the old
    // 4096 budget mid-title → FATAL. Overnight needs more headroom.
    int hole = 0;
    if (!near_null && !poison && !truncated && !ro_write && !plat_mem_readable(addr, 1)) {
        // n215: sequential writes through 0x25bcd4… burned 65k mid-title then FATAL. Overnight
        // foliage / GPU-VA storms need more headroom; still capped so a true spin cannot run forever.
        static int hole_skips;
        if (hole_skips < 1048576) hole = 1, hole_skips++;
    }
    if (!near_null && !poison && !truncated && !ro_write && !hole) return EXCEPTION_CONTINUE_SEARCH;

    // Preferred: longjmp out of an armed wrap (clean).
    if (s_mkw_soft_av_armed) longjmp(s_mkw_soft_av_jmp, 1);

    // Always-on path: skip the faulting insn so rt_dispatch-hit lifts survive overnight.
    if (!s_mkw_soft_skip_on) return EXCEPTION_CONTINUE_SEARCH;
    const uint8_t *ip = (const uint8_t *)(uintptr_t)ep->ContextRecord->Rip;
    int len = mkw_x64_insn_len(ip);
    // Unknown opcode: still skip a common REX+op+ModRM+disp8 length so overnight survives
    // (n27 wrote address 0 with an insn the short decoder rejected → FATAL).
    if (len <= 0 || len > 15) len = 4;
    static int logged;
    if (logged++ < 32)
        rt_log("system: soft-skip near-null AV %#llx at rip %#llx +%d", (unsigned long long)addr,
               (unsigned long long)ep->ContextRecord->Rip, len);
    // Do not poke GPRs: wrong ModRM decoding zeroed live pointers and blacked the title (n17).
    // Skipping the load/store alone is enough for overnight survival.
    ep->ContextRecord->Rip += (DWORD64)(unsigned)len;
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void mkw_soft_av_call(void (*fn)(Cpu *), Cpu *c, const char *who) {
    if (!fn) return;
    if (!s_mkw_soft_av_veh) s_mkw_soft_av_veh = AddVectoredExceptionHandler(1, mkw_soft_av_veh);
    if (setjmp(s_mkw_soft_av_jmp) == 0) {
        s_mkw_soft_av_armed = 1;
        fn(c);
        s_mkw_soft_av_armed = 0;
    } else {
        s_mkw_soft_av_armed = 0;
        static int logged;
        if (logged++ < 16) rt_log("system: %s soft-caught near-null AV, skip", who);
    }
}

static void mkw_enable_soft_skip(void) {
    s_mkw_soft_skip_on = 1;
    if (!s_mkw_soft_av_veh) s_mkw_soft_av_veh = AddVectoredExceptionHandler(1, mkw_soft_av_veh);
    rt_log("system: MKW near-null soft-skip VEH armed (overnight)");
}
#else
static void mkw_soft_av_call(void (*fn)(Cpu *), Cpu *c, const char *who) {
    (void)who;
    if (fn) fn(c);
}
static void mkw_enable_soft_skip(void) {}
#endif

#define NFP_R_DEVICE_NOT_FOUND 0x8073u     // nfp: no NFC reader on the host
#define ACCOUNT_R_INVALID 0x7c67u          // MacNativeGames' result for an unknown user handle or uid
#define TIME_R_INVALID 0x74u
#define TIME_R_SOURCE_MISMATCH 0x18074u

static void need(int ok, const char *what) {
    if (!ok) rt_fatal("%s", what);
}

static const char *caller(Cpu *c, char *buf, int n) { return ldr_describe(c->x[30] - 4, buf, n); }

// ==== nn::oe (services.c)
// TOTK_OPERATION_MODE: 1 handheld (OperationMode 0, PerformanceMode Normal 0), 0 docked (1, Boost 1).
int ov_handheld(void) {
    const char *e = cfg_value("TOTK_OPERATION_MODE");
    if (!e || !strcmp(e, "docked")) return 0;
    if (!strcmp(e, "handheld")) return 1;
    return 0;
}

// The Switch language for a BCP 47 tag; en-US for a language the game does not ship.
static const char *switch_language(const char *tag) {
    char lang[8] = "", region[4] = "", script[8] = "";
    size_t n = strcspn(tag, "-_");
    if (n < sizeof lang)
        for (size_t i = 0; i < n; i++) lang[i] = (char)tolower((unsigned char)tag[i]);
    for (const char *p = tag + n; *p;) {
        p += strspn(p, "-_");
        size_t k = strcspn(p, "-_");
        if (k == 2 && isalpha((unsigned char)p[0]) && isalpha((unsigned char)p[1])) {
            region[0] = (char)toupper((unsigned char)p[0]);
            region[1] = (char)toupper((unsigned char)p[1]);
        } else if (k == 3 && isdigit((unsigned char)p[0])) {
            memcpy(region, p, 3);   // UN M.49 area, e.g. es-419
        } else if (k == 4 && !*script) {
            for (size_t i = 0; i < 4; i++) script[i] = (char)(i ? tolower((unsigned char)p[i]) : toupper((unsigned char)p[i]));
        }
        p += k;
    }
    // the NVN 2 reference title ships Portuguese only as Mals/USpt (the Americas), with no EUpt for plain "pt", so every Portuguese
    // locale, Portugal's included, asks for pt-BR; "pt" alone would fall back to English.
    if (!strcmp(lang, "pt") && !rt_game_version) return "pt-BR";
    if (!strcmp(lang, "ja") || !strcmp(lang, "de") || !strcmp(lang, "it") || !strcmp(lang, "nl") ||
        !strcmp(lang, "ru") || !strcmp(lang, "ko") || !strcmp(lang, "pt"))
        return !strcmp(lang, "ja") ? "ja" : !strcmp(lang, "de") ? "de" : !strcmp(lang, "it") ? "it"
             : !strcmp(lang, "nl") ? "nl" : !strcmp(lang, "ru") ? "ru" : !strcmp(lang, "ko") ? "ko" : "pt";
    if (!strcmp(lang, "fr")) return !strcmp(region, "CA") ? "fr-CA" : "fr";
    if (!strcmp(lang, "es")) return *region && strcmp(region, "ES") ? "es-419" : "es";
    if (!strcmp(lang, "zh"))
        return !strcmp(script, "Hant") || !strcmp(region, "TW") || !strcmp(region, "HK") || !strcmp(region, "MO")
                   ? "zh-Hant" : "zh-Hans";
    if (!strcmp(lang, "en"))
        return !strcmp(region, "GB") || !strcmp(region, "IE") || !strcmp(region, "AU") || !strcmp(region, "NZ")
                   ? "en-GB" : "en-US";
    return "en-US";
}

// The game's language as a Switch language code (TOTK_LANG, else TOTK_LOCALE, else the Windows user locale). TotK
// accepts "zh-Hans" and "zh-Hant" for Chinese and falls back to Japanese for "zh-CN" and "zh-TW", so those two
// spellings, which older totk.ini files and menus wrote, are mapped to them.
const char *ov_language(void) {
    static const char *const codes[] = {"ja", "en-US", "fr", "de", "it", "es", "ko", "nl", "pt", "ru",
                                        "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR"};
    const char *code = cfg_get("TOTK_LANG");   // no fixed default: TOTK_LOCALE or the Windows user locale decides when unset
    if (code && *code) {
        if (!strcmp(code, "zh-CN")) return "zh-Hans";
        if (!strcmp(code, "zh-TW")) return "zh-Hant";
        for (size_t i = 0; i < sizeof codes / sizeof *codes; i++)
            if (!strcmp(code, codes[i])) return codes[i];
        const char *norm = switch_language(code);
        if (norm && *norm) {
            rt_log("oe: language '%s' mapped to Switch code '%s'", code, norm);
            return norm;
        }
        rt_log("oe: warning: language '%s' not recognized, defaulting to en-US", code);
        return "en-US";
    }
    const char *tag = cfg_get("TOTK_LOCALE");
    char sys[85] = "en-US";
    if (!tag || !*tag) {
        if (*plat_user_locale()) snprintf(sys, sizeof sys, "%s", plat_user_locale());
        tag = sys;
    }
    return switch_language(tag);
}

// LanguageCode GetDesiredLanguage(): char[8] in x0
static void ov_desired_language(Cpu *c) {
    const char *code = ov_language();
    uint64_t v = 0;
    memcpy(&v, code, strlen(code));
    ret_x(c, v);
}

static void ov_oe_initialize(Cpu *c) { (void)c; rt_log("oe: application environment: a local host process"); }

// void SetPerformanceConfiguration(PerformanceMode mode, int config)
static void ov_performance_config(Cpu *c) {
    int64_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 1) rt_fatal("nn::oe::SetPerformanceConfiguration: invalid mode %lld", (long long)mode);
    rt_log("oe: performance request mode %lld configuration %#x: the host OS schedules the CPU", (long long)mode,
           (uint32_t)arg_x(c, 1));
}

// void SetCpuBoostMode(CpuBoostMode mode)
static void ov_cpu_boost(Cpu *c) {
    int64_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 2) rt_fatal("nn::oe::SetCpuBoostMode: unknown mode %lld", (long long)mode);
}

static void ov_operation_mode(Cpu *c) { ret_x(c, ov_handheld() ? 0 : 1); }
static void ov_performance_mode(Cpu *c) { ret_x(c, ov_handheld() ? 0 : 1); }

// void SetFocusHandlingMode(FocusHandlingMode mode): the nnMain checkpoint of --boot
static volatile int32_t focus_mode;
static void ov_focus_mode(Cpu *c) {
    int64_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 3) rt_fatal("nn::oe::SetFocusHandlingMode: unknown mode %lld", (long long)mode);
    plat_xchg32(&focus_mode, (int32_t)mode);
    if (rt_config()->boot) {
        rt_boot_event("checkpoint", c, "SetFocusHandlingMode");
    } else {
        char lr[96];
        rt_log("oe: focus handling mode %lld from %s", (long long)mode, caller(c, lr, sizeof lr));
    }
}

// FocusState GetCurrentFocusState(): 1 in focus, 2 out of focus (nn::oe::FocusState). The game sets focus handling
// mode 2: on a Switch the system then suspends the whole process while it is out of focus and sends Resume when it
// returns. Here only the game's main thread stops (it waits in PopNotificationMessage for the next message) while its
// loader threads, audio and clocks run on, and a load interrupted that way was seen to never finish (Linux, 1.4.3).
// So TOTK_ASSUME_FOCUS=1, the default, keeps it at 1, as MacNativeGames does; with 0 it follows the window through
// oe_set_focus. The caller of oe_set_focus also pushes OE_MSG_FOCUS_STATE_CHANGED.
//
// The setting is read on every focus change rather than once at start: a focus change is rare, the lookup is a short
// walk of the loaded ini entries, and the Game tab's row then takes effect the next time the window is left, without
// a restart.
static volatile int32_t focus_state = 1;
void oe_set_focus(int in_focus) {
    plat_xchg32(&focus_state, in_focus || atoi(cfg_value("TOTK_ASSUME_FOCUS")) ? 1 : 2);
}
static void ov_focus_state(Cpu *c) { ret_x(c, (uint64_t)focus_state); }

static volatile int32_t notify_resume, notify_operation, notify_performance;
static void ov_resume_enabled(Cpu *c) { plat_xchg32(&notify_resume, (int32_t)(arg_x(c, 0) & 0xff)); }
static void ov_operation_enabled(Cpu *c) { plat_xchg32(&notify_operation, (int32_t)(arg_x(c, 0) & 0xff)); }
static void ov_performance_enabled(Cpu *c) { plat_xchg32(&notify_performance, (int32_t)(arg_x(c, 0) & 0xff)); }

// ==== nn::oe, applet, am, prepo, nfp, err (applet_host.c)
static PlatMutex lock = PLAT_MUTEX_INIT;
static PlatCond message_cv = PLAT_COND_INIT;
static uint32_t messages[32];
static unsigned head, tail;
static float balance[2] = {1.0f, 1.0f};
static uint64_t start_ticks;
__attribute__((constructor)) static void started(void) { start_ticks = rt_ticks(); }

void oe_push_message(uint32_t message) {
    plat_mutex_lock(&lock);
    if (tail - head < 32) {
        messages[tail++ % 32] = message;
        plat_cond_broadcast(&message_cv);
    }
    plat_mutex_unlock(&lock);
}

// void GetExpectedVolumeBalance(float *left, float *right)
static void ov_get_balance(Cpu *c) {
    float *l = arg_p(c, 0), *r = arg_p(c, 1);
    plat_mutex_lock_shared(&lock);
    if (l) *l = balance[0];
    if (r) *r = balance[1];
    plat_mutex_unlock_shared(&lock);
}

// void SetExpectedVolumeBalance(float left, float right)
static void ov_set_balance(Cpu *c) {
    float l = arg_s(c, 0), r = arg_s(c, 1);
    if (!(l >= 0 && l <= 1 && r >= 0 && r <= 1))
        rt_fatal("nn::oe::SetExpectedVolumeBalance: balance %g/%g out of range", (double)l, (double)r);
    plat_mutex_lock(&lock);
    balance[0] = l;
    balance[1] = r;
    plat_mutex_unlock(&lock);
}

// TimeSpan GetProgramTotalActiveTime(): nanoseconds since the process started
static void ov_active_time(Cpu *c) {
    ret_x(c, rt_ticks_to_ns(rt_ticks() - start_ticks));
}

// void SetUserInactivityDetectionTimeExtended[Unsafe](bool): the host never puts the console to sleep
static void ov_inactivity_extended(Cpu *c) { (void)c; }

// void FinishStartupLogo(): no system logo is drawn on the host
static void ov_finish_logo(Cpu *c) { (void)c; rt_log("oe: the game finished its startup logo"); }

// bool TryPopLaunchParameter(size_t *size, void *buffer, size_t capacity): launched directly, no parameter
static void ov_launch_parameter(Cpu *c) {
    uint64_t *size = arg_p(c, 0);
    if (size) *size = 0;
    ret_x(c, 0);
}

// bool TryPopNotificationMessage(Message *out)
static void ov_try_pop(Cpu *c) {
    uint32_t *out = arg_p(c, 0);
    plat_mutex_lock(&lock);
    int ok = head != tail;
    if (ok) {
        uint32_t m = messages[head++ % 32];
        if (out) *out = m;
    }
    plat_mutex_unlock(&lock);
    ret_x(c, (uint64_t)ok);
}

// Message PopNotificationMessage(): blocks like a Horizon wait, so the thread gives its virtual core up meanwhile
static void ov_pop(Cpu *c) {
    int away = kern_core_away();
    plat_mutex_lock(&lock);
    while (head == tail) plat_cond_wait(&message_cv, &lock);
    uint32_t m = messages[head++ % 32];
    plat_mutex_unlock(&lock);
    kern_core_back(&away);
    ret_x(c, m);
}

// int applet::GetPreviousProgramIndex(): started directly, not from an ExecuteProgram chain
static void ov_previous_program(Cpu *c) { ret_x(c, (uint64_t)(int64_t)-1); }
// AppletResourceUserId applet:: / am::GetAppletResourceUserId(): one application, one stable non-zero id
static void ov_aruid(Cpu *c) { ret_x(c, 1); }

// nn::prepo: play reports stay local (no Nintendo servers)
static volatile int64_t reports;
static void ov_report_save(Cpu *c) {   // Result PlayReport::Save() and Save(const account::Uid &)
    int64_t n = plat_inc64(&reports);
    if (n % 100 == 1) rt_log("prepo: play report kept local (not sent), %lld so far", (long long)n);
    ret_x(c, 0);
}
static void ov_result_ok(Cpu *c) { ret_x(c, 0); }   // RequestImmediateTransmission, AddInternetConnectionStatus

// nn::nfp: no NFC reader
static volatile int32_t nfp_state;
static void ov_nfp_initialize(Cpu *c) { plat_xchg32(&nfp_state, 1); ret_x(c, 0); }
static void ov_nfp_finalize(Cpu *c) { plat_xchg32(&nfp_state, 0); ret_x(c, 0); }
static void ov_nfp_state(Cpu *c) { ret_x(c, (uint64_t)nfp_state); }
// Result ListDevices(DeviceHandle *out, int *count, int max)
static void ov_nfp_list(Cpu *c) {
    int32_t *count = arg_p(c, 1);
    if (count) *count = 0;
    ret_x(c, NFP_R_DEVICE_NOT_FOUND);
}
// Result AttachAvailabilityChangeEvent(os::SystemEventType *event)
static void ov_nfp_availability(Cpu *c) {
    uint64_t ev = arg_x(c, 0);
    if (!ev) rt_fatal("nn::nfp::AttachAvailabilityChangeEvent: null event");
    kern_bind_system_event(c, ev, 0);
    ret_x(c, 0);
}
static void ov_nfp_no_device(Cpu *c) { ret_x(c, NFP_R_DEVICE_NOT_FOUND); }
static void ov_nfp_device_state(Cpu *c) { ret_x(c, 0); }   // DeviceState GetDeviceState(const DeviceHandle &): none

// void err::ShowError(Result): the game's error dialog, logged
static void ov_show_error(Cpu *c) {
    uint32_t r = (uint32_t)arg_x(c, 0);
    char lr[96];
    rt_log("err: game error dialog %04u-%04u (result %#x) from %s", 2000 + (r & 0x1ff), (r >> 9) & 0x1fff, r,
           caller(c, lr, sizeof lr));
}

// ==== SDK-internal initialisation (services.c)
static void ov_diag_startup(Cpu *c) { (void)c; rt_log("diag: host diagnostics"); }
static void ov_aoc_initialize(Cpu *c) { (void)c; rt_log("aoc: %u add-on content(s)", aoc_count()); }
static void ov_aoc_count(Cpu *c) { ret_x(c, aoc_count()); }
// int ListAddOnContent(int *out, int offset, int count): the add-on content indices this machine has, paged.
static void ov_aoc_list(Cpu *c) {
    int32_t *out = arg_p(c, 0);
    int offset = (int32_t)arg_x(c, 1), count = (int32_t)arg_x(c, 2);
    if (offset < 0 || count < 0) rt_fatal("aoc: ListAddOnContent range %d+%d", offset, count);
    uint32_t ids[AOC_MAX];
    uint32_t n = aoc_list(ids, offset, count < AOC_MAX ? count : AOC_MAX);
    if (n && !(out && plat_mem_readable((uintptr_t)out, n * sizeof *out))) n = 0;   // nothing written, nothing reported
    for (uint32_t i = 0; i < n; i++) out[i] = (int32_t)ids[i];
    ret_x(c, n);
}
static void ov_err_context(Cpu *c) { (void)c; rt_log("err: error contexts stay local"); }
static void ov_pctl_initialize(Cpu *c) { (void)c; rt_log("pctl: no parental controls"); }
static void ov_abort_observer(Cpu *c) { (void)c; rt_log("diag: the host crash reporter observes aborts"); }

// ==== observers (services.c): log, then the guest definition
static void observe(Cpu *c, const char *name) {
    rt_log("ENTER %s", name);
    rt_original(c);
    rt_log("RETURN %s", name);
}
static void obs_os_initialize(Cpu *c) { observe(c, "nnosInitialize"); }
static void obs_sdk_initialize(Cpu *c) { observe(c, "nninitInitializeSdkModule"); }
static void obs_startup(Cpu *c) { observe(c, "nninitStartup"); }
static void obs_main(Cpu *c) { observe(c, "nnMain"); }

// Result os::AllocateAddressRegion(uintptr_t *out, size_t size)
static void obs_address_region(Cpu *c) {
    uint64_t *out = arg_p(c, 0), size = arg_x(c, 1);
    uint32_t r = (uint32_t)rt_original(c);
    if (rt_trace & RT_TRACE_HOSTCALL)   // 20,000 lines a session otherwise, each a write on a game thread
        rt_log("AllocateAddressRegion size %#llx result %#x address %#llx", (unsigned long long)size, r,
           (unsigned long long)(r || !out ? 0 : *out));
}

// Result os::AllocateMemoryPages(uintptr_t address, size_t size)
static void obs_memory_pages(Cpu *c) {
    uint64_t a = arg_x(c, 0), size = arg_x(c, 1);
    uint32_t r = (uint32_t)rt_original(c);
    if (rt_trace & RT_TRACE_HOSTCALL)
        rt_log("AllocateMemoryPages address %#llx size %#llx result %#x", (unsigned long long)a, (unsigned long long)size, r);
}

// ==== nn::account (account.c): one offline local user whose Uid survives restarts in <profile>/account.uid
typedef struct { uint64_t lo, hi; } Uid;
typedef struct { Uid uid; uint64_t token; } UserHandle;
static PlatMutex account_lock = PLAT_MUTEX_INIT;
static Uid local;
static uint64_t tokens[64], next_token = 1;

static void make_dirs(const char *dir) {
    plat_file_mkdir_p(dir);
    if (!plat_dir_exists(dir)) rt_fatal("account: cannot create %s", dir);
}

// Holding account_lock.
static void load_uid(void) {
    if (local.lo || local.hi) return;
    const char *dir = rt_config()->profile;
    char path[1100];
    make_dirs(dir);
    need(snprintf(path, sizeof path, "%s/account.uid", dir) < (int)sizeof path, "account: path too long");
    int64_t size = plat_file_size(path);
    if (size <= 0) {
        do need(plat_random(&local, sizeof local), "account: random uid");
        while (!local.lo && !local.hi);
        need(plat_file_write_atomic(path, &local, sizeof local), "account: persist the local account identity");
        rt_log("account: new local account in %s", path);
    } else {
        uint64_t got = 0;
        uint8_t *b = plat_file_read(path, &got);
        if (b && got == sizeof local) memcpy(&local, b, sizeof local);
        free(b);
        if (got != sizeof local || (!local.lo && !local.hi))
            rt_fatal("account: %s is not a local account identity (16 bytes)", path);
    }
}

// host.h: the save directory of fs_host.c is named after this Uid.
void account_uid(uint64_t out[2]) {
    plat_mutex_lock(&account_lock);
    load_uid();
    out[0] = local.lo;
    out[1] = local.hi;
    plat_mutex_unlock(&account_lock);
}

static int same_uid(const Uid *u) { return u && u->lo == local.lo && u->hi == local.hi; }

// Holding account_lock.
static int valid_handle(const UserHandle *h) {
    if (!h || !same_uid(&h->uid)) return 0;
    for (int i = 0; i < 64; i++)
        if (tokens[i] && tokens[i] == h->token) return 1;
    return 0;
}

// void Initialize()
static void ov_account_initialize(Cpu *c) {
    uint64_t uid[2];
    (void)c;
    account_uid(uid);
}

// bool TryOpenPreselectedUser(UserHandle *out)
static void ov_open_user(Cpu *c) {
    UserHandle *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::account::TryOpenPreselectedUser: null handle");
    plat_mutex_lock(&account_lock);
    load_uid();
    int ok = 0;
    for (int i = 0; i < 64 && !ok; i++)
        if (!tokens[i]) {
            tokens[i] = next_token++;
            *out = (UserHandle){local, tokens[i]};
            ok = 1;
        }
    plat_mutex_unlock(&account_lock);
    ret_x(c, (uint64_t)ok);
}

// void CloseUser(const UserHandle &)
static void ov_close_user(Cpu *c) {
    const UserHandle *h = arg_p(c, 0);
    plat_mutex_lock(&account_lock);
    if (!valid_handle(h)) rt_fatal("nn::account::CloseUser: not an open user handle");
    for (int i = 0; i < 64; i++)
        if (tokens[i] == h->token) tokens[i] = 0;
    plat_mutex_unlock(&account_lock);
}

// Result GetUserId(Uid *out, const UserHandle &)
static void ov_user_id(Cpu *c) {
    Uid *out = arg_p(c, 0);
    plat_mutex_lock_shared(&account_lock);
    int ok = out && valid_handle(arg_p(c, 1));
    if (ok) *out = local;
    plat_mutex_unlock_shared(&account_lock);
    ret_x(c, ok ? 0 : ACCOUNT_R_INVALID);
}

// Result GetLastOpenedUser(Uid *out): the local user, the only one the host ever opens (the NVN 2 reference title asks after
// TryOpenPreselectedUser)
static void ov_last_opened_user(Cpu *c) {
    Uid *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::account::GetLastOpenedUser: null output");
    plat_mutex_lock_shared(&account_lock);
    *out = local;
    plat_mutex_unlock_shared(&account_lock);
    ret_x(c, 0);
}

// Result GetUserExistence(bool *out, const Uid &)
static void ov_user_existence(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    const Uid *u = arg_p(c, 1);
    if (!out || !u) { ret_x(c, ACCOUNT_R_INVALID); return; }
    plat_mutex_lock(&account_lock);
    load_uid();
    *out = (uint8_t)same_uid(u);
    plat_mutex_unlock(&account_lock);
    ret_x(c, 0);
}

// Result GetNickname(Nickname *out, const Uid &): 33 bytes, NUL-terminated UTF-8
static void ov_nickname(Cpu *c) {
    char *out = arg_p(c, 0);
    plat_mutex_lock(&account_lock);
    load_uid();
    int ok = out && same_uid(arg_p(c, 1));
    plat_mutex_unlock(&account_lock);
    if (ok) {
        memset(out, 0, 33);
        memcpy(out, "Local", 5);
    }
    ret_x(c, ok ? 0 : ACCOUNT_R_INVALID);
}

static void ov_user_count(Cpu *c) {
    int32_t *out = arg_p(c, 0);
    if (out) *out = 1;
    ret_x(c, 0);
}

static void ov_list_all_users(Cpu *c) {
    int32_t *out = arg_p(c, 0);
    Uid *uids = arg_p(c, 1);
    int32_t max = (int32_t)arg_x(c, 2);
    plat_mutex_lock(&account_lock);
    load_uid();
    if (out) *out = 1;
    if (uids && max >= 1) {
        uids[0] = local;
    }
    plat_mutex_unlock(&account_lock);
    ret_x(c, 0);
}

static void ov_open_user_by_uid(Cpu *c) {
    UserHandle *out = arg_p(c, 0);
    const Uid *u = arg_p(c, 1);
    if (!out) rt_fatal("nn::account::OpenUser: null handle");
    plat_mutex_lock(&account_lock);
    load_uid();
    int ok = 0;
    if (!u || same_uid(u)) {
        for (int i = 0; i < 64 && !ok; i++)
            if (!tokens[i]) {
                tokens[i] = next_token++;
                *out = (UserHandle){local, tokens[i]};
                ok = 1;
            }
    }
    plat_mutex_unlock(&account_lock);
    ret_x(c, ok ? 0 : ACCOUNT_R_INVALID);
}

// ==== nn::time (time_host.c): the host clock and time zone. User and network clocks follow the host wall clock; the
// steady clock counts seconds since Windows started, with one random source id per process. ClockSnapshot has the
// public 0xD0-byte layout, because game code reads its fields through inline accessors.
typedef struct { int64_t value; uint8_t source[16]; } SteadyClockTimePoint;
typedef struct { int64_t offset; SteadyClockTimePoint point; } SystemClockContext;
typedef struct { int16_t year; int8_t month, day, hour, minute, second, pad; } CalendarTime;
typedef struct { int32_t day_of_week, day_of_year; char name[8]; uint8_t dst, pad[3]; int32_t utc_offset; } CalendarInfo;
typedef struct {
    SystemClockContext user, network;
    int64_t user_time, network_time;
    CalendarTime user_calendar, network_calendar;
    CalendarInfo user_info, network_info;
    SteadyClockTimePoint steady;
    char location[0x24];
    uint8_t automatic_correction, type, pad[2];
} ClockSnapshot;
_Static_assert(sizeof(SteadyClockTimePoint) == 0x18 && sizeof(CalendarTime) == 8 && sizeof(CalendarInfo) == 0x18,
               "nn::time types");
_Static_assert(sizeof(ClockSnapshot) == 0xd0, "nn::time::ClockSnapshot");

static uint8_t source_id[16];
static volatile int32_t time_ready;
static PlatOnce source_once = PLAT_ONCE_INIT;

static void make_source(void *param) {
    (void)param;
    need(plat_random(source_id, sizeof source_id), "time: source id");
    plat_tz_west();   // reads the zone once, before the first localtime
    return;
}

static SteadyClockTimePoint steady_now(void) {
    plat_once(&source_once, make_source, NULL);
    SteadyClockTimePoint p = {(int64_t)(plat_now_ms() / 1000), {0}};
    memcpy(p.source, source_id, 16);
    return p;
}

static int64_t posix_now(void) { return (int64_t)time(NULL); }

// Days since 1970-01-01 of a proleptic Gregorian date, and back (H. Hinnant's civil calendar algorithms).
static int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    int64_t era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

static void civil_from_days(int64_t z, int64_t *y, unsigned *m, unsigned *d) {
    z += 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = (int64_t)yoe + era * 400 + (*m <= 2);
}

static int64_t floor_div(int64_t a, int64_t b) { return a / b - (a % b != 0 && (a < 0) != (b < 0)); }

// UTC calendar fields of a POSIX time. 0 if the year does not fit CalendarTime.
static int utc_calendar(int64_t t, CalendarTime *c, CalendarInfo *info) {
    int64_t days = floor_div(t, 86400), sec = t - days * 86400, y;
    unsigned m, d;
    civil_from_days(days, &y, &m, &d);
    *c = (CalendarTime){(int16_t)y, (int8_t)m, (int8_t)d, (int8_t)(sec / 3600), (int8_t)(sec / 60 % 60),
                        (int8_t)(sec % 60), 0};
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = (int32_t)(days + 4 - floor_div(days + 4, 7) * 7);   // 1970-01-01 was a Thursday
        info->day_of_year = (int32_t)(days - days_from_civil(y, 1, 1));
        memcpy(info->name, "UTC", 3);
    }
    return y >= -32768 && y <= 32767;
}

// Local calendar fields of a POSIX time in the Windows time zone. The zone name is "UTC" or the UTC offset ("+03",
// "-0330"), as tzdata names zones without an abbreviation; Windows has no abbreviations. 0 outside localtime's range.
static int local_calendar(int64_t t, CalendarTime *c, CalendarInfo *info) {
    plat_once(&source_once, make_source, NULL);
    struct tm tm;
    if (!plat_localtime(t, &tm)) return 0;
    int64_t as_utc = days_from_civil(tm.tm_year + 1900, (unsigned)tm.tm_mon + 1, (unsigned)tm.tm_mday) * 86400 +
                     tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec;
    int32_t offset = (int32_t)(as_utc - t);
    *c = (CalendarTime){(int16_t)(tm.tm_year + 1900), (int8_t)(tm.tm_mon + 1), (int8_t)tm.tm_mday, (int8_t)tm.tm_hour,
                        (int8_t)tm.tm_min, (int8_t)tm.tm_sec, 0};
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = tm.tm_wday;
        info->day_of_year = tm.tm_yday;
        int a = offset < 0 ? -offset : offset;
        if (!offset) memcpy(info->name, "UTC", 3);
        else if (a % 3600) snprintf(info->name, sizeof info->name, "%c%02d%02d", offset < 0 ? '-' : '+', a / 3600, a / 60 % 60);
        else snprintf(info->name, sizeof info->name, "%c%02d", offset < 0 ? '-' : '+', a / 3600);
        info->dst = tm.tm_isdst > 0;
        info->utc_offset = offset;
    }
    return 1;
}

// IANA name of the standard offset: Etc/GMT+5 is UTC-5 (POSIX sign); "UTC" for a zone off whole hours.
static void location_name(char out[0x24]) {
    long west = plat_tz_west();
    memset(out, 0, 0x24);
    if (!west || west % 3600) snprintf(out, 0x24, "UTC");
    else snprintf(out, 0x24, "Etc/GMT%+ld", west / 3600);
}

static void ov_time_initialize(Cpu *c) {
    plat_once(&source_once, make_source, NULL);
    plat_xchg32(&time_ready, 1);
    ret_x(c, 0);
}
static void ov_time_finalize(Cpu *c) { plat_xchg32(&time_ready, 0); ret_x(c, 0); }
static void ov_time_ready(Cpu *c) { ret_x(c, (uint64_t)time_ready); }

// Result StandardUserSystemClock / StandardNetworkSystemClock::GetCurrentTime(PosixTime *out)
static void ov_current_time(Cpu *c) {
    int64_t *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    *out = posix_now();
    ret_x(c, 0);
}

// time_point StandardUserSystemClock / StandardNetworkSystemClock::now(): PosixTime seconds in x0
static void ov_posix_now(Cpu *c) { ret_x(c, (uint64_t)posix_now()); }

// time_point StandardSteadyClock::now(): the int64 seconds in x0 only. The SDK caller (sdk+0x3d1d88) reads x0;
// returning the 0x18-byte struct through x8 would write through a stale pointer (time_host.c).
static void ov_steady_now(Cpu *c) { ret_x(c, (uint64_t)steady_now().value); }

// Result StandardSteadyClock::GetCurrentTimePoint(SteadyClockTimePoint *out)
static void ov_steady_timepoint(Cpu *c) {
    SteadyClockTimePoint *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    *out = steady_now();
    ret_x(c, 0);
}

// Result GetStandardSteadyClockCurrentTimespec(timespec *out): called by clock_gettime(CLOCK_MONOTONIC) after nntimeStartup
static void ov_steady_timespec(Cpu *c) {
    int64_t *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    uint64_t ns = kern_now_ns();
    out[0] = (int64_t)(ns / 1000000000ull);
    out[1] = (int64_t)(ns % 1000000000ull);
    ret_x(c, 0);
}

// Result GetStandard{User,Network}SystemClockCurrentTimespec(timespec *out): called by clock_gettime(CLOCK_REALTIME)
static void ov_system_timespec(Cpu *c) {
    int64_t *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    out[0] = posix_now();
    out[1] = (int64_t)(kern_now_ns() % 1000000000ull);
    ret_x(c, 0);
}

// Result ToCalendarTime(CalendarTime *out, CalendarAdditionalInfo *info, const PosixTime &t): local time
static void ov_to_calendar(Cpu *c) {
    CalendarTime *out = arg_p(c, 0);
    CalendarInfo *info = arg_p(c, 1);
    const int64_t *t = arg_p(c, 2);
    if (!out || !t) { ret_x(c, TIME_R_INVALID); return; }
    CalendarTime ct;
    CalendarInfo ci;
    if (!local_calendar(*t, &ct, &ci)) { ret_x(c, TIME_R_INVALID); return; }
    *out = ct;
    if (info) *info = ci;
    ret_x(c, 0);
}

// CalendarTime ToCalendarTimeInUtc(const PosixTime &t): 8 bytes in x0
static void ov_to_calendar_utc(Cpu *c) {
    const int64_t *t = arg_p(c, 0);
    CalendarTime ct;
    utc_calendar(t ? *t : 0, &ct, NULL);
    uint64_t v;
    memcpy(&v, &ct, 8);
    ret_x(c, v);
}

// PosixTime ToPosixTimeFromUtc(const CalendarTime &c): months out of 1..12 carry into the year, like timegm
static void ov_to_posix_utc(Cpu *c) {
    const CalendarTime *ct = arg_p(c, 0);
    if (!ct) rt_fatal("nn::time::ToPosixTimeFromUtc: null calendar time");
    int64_t mon = ct->month - 1, y = ct->year + floor_div(mon, 12);
    mon -= floor_div(mon, 12) * 12;
    int64_t t = (days_from_civil(y, (unsigned)mon + 1, 1) + ct->day - 1) * 86400 + ct->hour * 3600 + ct->minute * 60 +
                ct->second;
    ret_x(c, (uint64_t)t);
}

static void ov_get_device_location_name(Cpu *c) {
    char *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    location_name(out);
    ret_x(c, 0);
}

static void ov_system_clock_context(Cpu *c) {
    SystemClockContext *out = arg_p(c, 0);
    if (!out) { ret_x(c, TIME_R_INVALID); return; }
    SteadyClockTimePoint steady = steady_now();
    int64_t now = posix_now();
    *out = (SystemClockContext){now - steady.value, steady};
    ret_x(c, 0);
}

// ClockSnapshot::ClockSnapshot()
static void ov_snapshot_init(Cpu *c) {
    ClockSnapshot *s = arg_p(c, 0);
    if (s) memset(s, 0, sizeof *s);
}

// static Result ClockSnapshot::CreateWithStandardSystemClock(ClockSnapshot *out)
static void ov_snapshot_create(Cpu *c) {
    ClockSnapshot *s = arg_p(c, 0);
    if (!s) { ret_x(c, TIME_R_INVALID); return; }
    SteadyClockTimePoint steady = steady_now();
    int64_t now = posix_now();
    ClockSnapshot n = {0};
    n.user = (SystemClockContext){now - steady.value, steady};
    n.network = n.user;
    n.user_time = n.network_time = now;
    if (!local_calendar(now, &n.user_calendar, &n.user_info)) rt_fatal("nn::time: the host clock is out of range");
    n.network_calendar = n.user_calendar;
    n.network_info = n.user_info;
    n.steady = steady;
    location_name(n.location);
    n.automatic_correction = 1;
    *s = n;
    ret_x(c, 0);
}

// Result CalculateSpanBetween(TimeSpan *out, const ClockSnapshot &a, const ClockSnapshot &b): nanoseconds
static void ov_span_between(Cpu *c) {
    int64_t *out = arg_p(c, 0);
    const ClockSnapshot *a = arg_p(c, 1), *b = arg_p(c, 2);
    if (!out || !a || !b) { ret_x(c, TIME_R_INVALID); return; }
    if (memcmp(a->steady.source, b->steady.source, 16)) { ret_x(c, TIME_R_SOURCE_MISMATCH); return; }
    *out = (b->steady.value - a->steady.value) * 1000000000ll;
    ret_x(c, 0);
}

// TimeSpan CalculateStandardUserSystemClockDifferenceByUser(const ClockSnapshot &a, const ClockSnapshot &b): the
// change of the user clock offset between the snapshots (manual adjustments) in nanoseconds
static void ov_user_difference(Cpu *c) {
    const ClockSnapshot *a = arg_p(c, 0), *b = arg_p(c, 1);
    if (!a || !b) rt_fatal("nn::time::CalculateStandardUserSystemClockDifferenceByUser: null snapshot");
    if (memcmp(a->user.point.source, b->user.point.source, 16)) { ret_x(c, 0); return; }
    ret_x(c, (uint64_t)((b->user.offset - a->user.offset) * 1000000000ll));
}

// ==== nn::pl shared fonts (shared_font.c). TotK asks for them only from its font loader, only with Simplified Chinese,
// Korean or Traditional Chinese text, and only for type 0; tools/extract_fonts.py writes
// them from the user's firmware. Each type is read on first use into guest-visible memory as the BFTTF file the pl
// shared memory holds; Address is 8 bytes past the file start (after the BFTTF header) and Size the decoded font size,
// as in MNG. The load state is always Loaded: the game spins until it is, so missing files stop with a named fatal at
// Address/Size instead of a hang. TOTK_SHARED_FONTS=<dir> overrides <romfs>/../../firmware/shared-fonts.
#define SHARED_FONT_TYPES 6
#define BFTTF_MAGIC 0x7f9a0218u
static const char *const font_files[SHARED_FONT_TYPES] = {
    "0-standard.bfttf", "1-chinese-simplified.bfttf", "2-chinese-simplified-extension.bfttf",
    "3-chinese-traditional.bfttf", "4-korean.bfttf", "5-nintendo-extension.bfttf"};
static struct { PlatMutex lock; const uint8_t *file[SHARED_FONT_TYPES]; uint64_t size[SHARED_FONT_TYPES]; } fonts;

const char *ov_shared_font_dir(char *buf, size_t n) {
    const char *e = cfg_get("TOTK_SHARED_FONTS");
    if (e && *e) snprintf(buf, n, "%s", e);
    else {
        char exe[PLAT_PATH_MAX], probe[PLAT_PATH_MAX + 40];   // shared-fonts next to totk.exe (tools/package_build.py), else the repo's
        int len = plat_exe_path(exe, sizeof exe);
        char *cut = len && len < sizeof exe ? plat_last_sep(exe) : NULL;
        if (cut) *cut = 0;
        snprintf(probe, sizeof probe, "%s/shared-fonts/0-standard.bfttf", exe);
        if (cut && plat_file_exists(probe)) snprintf(buf, n, "%s/shared-fonts", exe);
        else snprintf(buf, n, "%s/../../firmware/shared-fonts", rt_config()->romfs);
    }
    return buf;
}

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

// The decoded font size of a BFTTF image, 0 if it is not one: big-endian words XORed with key = word0 ^ BFTTF_MAGIC,
// word1 = font size, then an sfnt (TrueType or OpenType) font (MNG extract_shared_fonts.py bfttf_size).
uint64_t ov_bfttf_size(const uint8_t *p, uint64_t len) {
    if (len < 12) return 0;
    uint32_t key = be32(p) ^ BFTTF_MAGIC, size = be32(p + 4) ^ key, first = be32(p + 8) ^ key;
    return (uint64_t)size + 8 == len && (first == 0x00010000u || first == 0x4f54544fu) ? size : 0;
}

static const char *font_type_name(int64_t t) { return font_files[t] + 2; }

static int64_t font_type(Cpu *c, const char *fn) {
    int64_t t = (int32_t)arg_x(c, 0);
    if (t < 0 || t >= SHARED_FONT_TYPES) rt_fatal("nn::pl::%s: invalid shared font type %lld", fn, (long long)t);
    return t;
}
static void ov_font_state(Cpu *c) { font_type(c, "GetSharedFontLoadState"); ret_x(c, 1); }

// The font of type t, read and checked on first use; a named fatal when it is missing or damaged.
static const uint8_t *font_data(Cpu *c, const char *fn, int64_t t) {
    plat_mutex_lock(&fonts.lock);
    if (!fonts.file[t]) {
        char dir[PLAT_PATH_MAX], path[PLAT_PATH_MAX + 64], lr[96];
        snprintf(path, sizeof path, "%s/%s", ov_shared_font_dir(dir, sizeof dir), font_files[t]);
        FILE *f = fopen(path, "rb");
        if (!f)
            rt_fatal("nn::pl::%s(%lld) from %s: no shared font %s (needed for Chinese and Korean text); run "
                     "tools/extract_fonts.py --firmware <firmware NCA dir> or set TOTK_SHARED_FONTS", fn, (long long)t,
                     caller(c, lr, sizeof lr), path);
        plat_fseek64(f, 0, SEEK_END);
        int64_t len = plat_ftell64(f);
        plat_fseek64(f, 0, SEEK_SET);
        if (len < 12 || len > (64 << 20)) rt_fatal("nn::pl: shared font %s: size %lld", path, (long long)len);
        uint8_t *p = mem_kernel_alloc((uint64_t)len, 16);
        size_t got = fread(p, 1, (size_t)len, f);
        fclose(f);
        uint64_t size = got == (size_t)len ? ov_bfttf_size(p, (uint64_t)len) : 0;
        if (!size) rt_fatal("nn::pl: shared font %s is not a BFTTF font", path);
        fonts.size[t] = size;
        __atomic_store_n(&fonts.file[t], p, __ATOMIC_RELEASE);
        rt_log("pl: shared font %lld (%s): %llu bytes from %s", (long long)t, font_type_name(t),
               (unsigned long long)size, path);
    }
    const uint8_t *p = fonts.file[t];
    plat_mutex_unlock(&fonts.lock);
    return p;
}
static void ov_font_address(Cpu *c) {
    int64_t t = font_type(c, "GetSharedFontAddress");
    ret_x(c, (uint64_t)(uintptr_t)(font_data(c, "GetSharedFontAddress", t) + 8));
}
static void ov_font_size(Cpu *c) {
    int64_t t = font_type(c, "GetSharedFontSize");
    font_data(c, "GetSharedFontSize", t);
    ret_x(c, fonts.size[t]);
}

// int nn::audio::detail::ListVirtualInputDevices(AudioDeviceName *out, int max): no virtual microphone input on host
static void ov_list_virtual_input(Cpu *c) { ret_x(c, 0); }

// ---- Offline stubs for nifm, socket, ssl, nsd, ro, prepo::Send, account NSA
static void ov_noop(Cpu *c) { (void)c; }
static void ov_ret_zero(Cpu *c) { ret_x(c, 0); }
static void ov_ret_one(Cpu *c) { ret_x(c, 1); }
static void ov_nifm_error_result(Cpu *c) { ret_x(c, 0x271000u | (110u & 0x1ffu)); }   // nifm offline result
static void ov_nifm_proxy(Cpu *c) {
    void *out = arg_p(c, 0);
    if (out) memset(out, 0, 0xaa);
    ret_x(c, 0);
}
static void ov_nsd_env(Cpu *c) {
    char *out = arg_p(c, 0);
    if (out) { memset(out, 0, 8); memcpy(out, "lp1", 3); }
    ret_x(c, 0);
}
static void ov_nsd_resolve(Cpu *c) {
    char *out = arg_p(c, 0);
    const char *in = arg_p(c, 1);
    if (out && in && out != in) memcpy(out, in, 0x100);
    ret_x(c, 0);
}
static void ov_account_offline(Cpu *c) { ret_x(c, 0x1f4000u | 124u); }   // account network unavailable
static void ov_report_send(Cpu *c) {
    int64_t n = plat_inc64(&reports);
    if (n % 100 == 1) rt_log("prepo: play report kept local (not sent), %lld so far", (long long)n);
    void *out = (void *)(uintptr_t)c->x[8];
    if (out) memset(out, 0, 16);
    ret_x(c, c->x[8]);
}
static void ov_async_event(Cpu *c) {
    uint64_t ev = arg_x(c, 1);
    if (ev) {
        kern_bind_system_event(c, ev, 0);
        kern_signal_system_event(ev);
    }
    ret_x(c, 0);
}
static volatile uint32_t sock_errno;
static void ov_sock_fail(Cpu *c) { sock_errno = 38; ret_x(c, (uint64_t)(int64_t)-1); }
static void ov_sock_get_errno(Cpu *c) { ret_x(c, sock_errno); }
static void ov_sock_set_errno(Cpu *c) { sock_errno = (uint32_t)arg_x(c, 0); }
static void ov_sock_gai(Cpu *c) { ret_x(c, 8); }   // EAI_NONAME
static void ov_sock_gai_str(Cpu *c) {
    static char *msg;
    if (!msg) { msg = mem_kernel_alloc(16, 16); memcpy(msg, "offline", 8); }
    ret_x(c, (uint64_t)(uintptr_t)msg);
}
static void ov_sock_htonl(Cpu *c) { ret_x(c, __builtin_bswap32((uint32_t)arg_x(c, 0))); }
static void ov_sock_htons(Cpu *c) { ret_x(c, __builtin_bswap16((uint16_t)arg_x(c, 0))); }
static void ov_sock_eventfd(Cpu *c) { ret_x(c, 100); }
static void ov_sock_eventfd_rw(Cpu *c) {
    uint64_t *out = arg_p(c, 1);
    if (out) *out = 1;
    ret_x(c, 0);
}

// ==== registration
// ---- Switch 2 oe calls (the NVN 2 reference title)
// int nn::oe::GetCurrentGraphicsScale(): an enum 0-2; main+0x4686d4 maps it through a float table {3, 12, 16} and
// ignores anything above 2. The host is a docked console with the most to give, so the highest value. Unverified
// against hardware: what the three levels mean is not known.
static void ov_graphics_scale(Cpu *c) { ret_x(c, 2); }
// void nn::oe::SetGraphicsScaleChangedNotificationEnabled(bool): the scale never changes on the host
static void ov_graphics_scale_notify(Cpu *c) { (void)c; }
// void nn::oe::SetAlbumImageOrientation(album::ImageOrientation): there is no capture album
static void ov_album_orientation(Cpu *c) { (void)c; }
// void nn::oe::{Enter,Leave}ExitRequestHandlingSection(): no exit request ever arrives from a host system applet
static void ov_exit_section(Cpu *c) { (void)c; }
// void nn::oe::GetDisplayVersion(DisplayVersion *out): 16-byte null-terminated version string
static void ov_display_version(Cpu *c) {
    char *out = arg_p(c, 0);
    if (out) {
        memset(out, 0, 16);
        memcpy(out, "1.0.0", 5);
    }
}

HOST_OVERRIDE("_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE", ov_display_version);
HOST_OVERRIDE("_ZN2nn2oe23GetCurrentGraphicsScaleEv", ov_graphics_scale);
HOST_OVERRIDE("_ZN2nn2oe42SetGraphicsScaleChangedNotificationEnabledEb", ov_graphics_scale_notify);
HOST_OVERRIDE("_ZN2nn2oe24SetAlbumImageOrientationENS_5album16ImageOrientationE", ov_album_orientation);
HOST_OVERRIDE("_ZN2nn2oe31EnterExitRequestHandlingSectionEv", ov_exit_section);
HOST_OVERRIDE("_ZN2nn2oe31LeaveExitRequestHandlingSectionEv", ov_exit_section);

HOST_OVERRIDE("_ZN2nn2oe10InitializeEv", ov_oe_initialize);
HOST_OVERRIDE("_ZN2nn2oe18GetDesiredLanguageEv", ov_desired_language);
HOST_OVERRIDE("_ZN2nn2oe27SetPerformanceConfigurationENS0_15PerformanceModeEi", ov_performance_config);
HOST_OVERRIDE("_ZN2nn2oe15SetCpuBoostModeENS0_12CpuBoostModeE", ov_cpu_boost);
HOST_OVERRIDE("_ZN2nn2oe16GetOperationModeEv", ov_operation_mode);
HOST_OVERRIDE("_ZN2nn2oe18GetPerformanceModeEv", ov_performance_mode);
HOST_OVERRIDE("_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE", ov_focus_mode);
HOST_OVERRIDE("_ZN2nn2oe20GetCurrentFocusStateEv", ov_focus_state);
HOST_OVERRIDE("_ZN2nn2oe28SetResumeNotificationEnabledEb", ov_resume_enabled);
HOST_OVERRIDE("_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb", ov_operation_enabled);
HOST_OVERRIDE("_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb", ov_performance_enabled);

HOST_OVERRIDE("_ZN2nn6applet23GetPreviousProgramIndexEv", ov_previous_program);
HOST_OVERRIDE("_ZN2nn6applet23GetAppletResourceUserIdEv", ov_aruid);
HOST_OVERRIDE("_ZN2nn2am23GetAppletResourceUserIdEv", ov_aruid);
HOST_OVERRIDE("_ZN2nn2oe21TryPopLaunchParameterEPmPvm", ov_launch_parameter);
HOST_OVERRIDE("_ZN2nn2oe24GetExpectedVolumeBalanceEPfS1_", ov_get_balance);
HOST_OVERRIDE("_ZN2nn2oe24SetExpectedVolumeBalanceEff", ov_set_balance);
HOST_OVERRIDE("_ZN2nn2oe25GetProgramTotalActiveTimeEv", ov_active_time);
HOST_OVERRIDE("_ZN2nn2oe38SetUserInactivityDetectionTimeExtendedEb", ov_inactivity_extended);
HOST_OVERRIDE("_ZN2nn2oe44SetUserInactivityDetectionTimeExtendedUnsafeEb", ov_inactivity_extended);
HOST_OVERRIDE("_ZN2nn2oe17FinishStartupLogoEv", ov_finish_logo);
HOST_OVERRIDE("_ZN2nn2oe25TryPopNotificationMessageEPj", ov_try_pop);
HOST_OVERRIDE("_ZN2nn2oe22PopNotificationMessageEv", ov_pop);
HOST_OVERRIDE("_ZN2nn5prepo10PlayReport4SaveEv", ov_report_save);
HOST_OVERRIDE("_ZN2nn5prepo10PlayReport4SaveERKNS_7account3UidE", ov_report_save);
HOST_OVERRIDE("_ZN2nn5prepo28RequestImmediateTransmissionEv", ov_result_ok);
HOST_OVERRIDE("_ZN2nn5prepo27AddInternetConnectionStatusEPNS0_10PlayReportEPKcS4_S4_", ov_result_ok);
HOST_OVERRIDE("_ZN2nn3nfp10InitializeEv", ov_nfp_initialize);
HOST_OVERRIDE("_ZN2nn3nfp8FinalizeEv", ov_nfp_finalize);
HOST_OVERRIDE("_ZN2nn3nfp8GetStateEv", ov_nfp_state);
HOST_OVERRIDE("_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii", ov_nfp_list);
HOST_OVERRIDE("_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE", ov_nfp_availability);
HOST_OVERRIDE("_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE", ov_nfp_device_state);
HOST_OVERRIDE("_ZN2nn3nfp5MountERKNS0_12DeviceHandleENS0_9ModelTypeENS0_11MountTargetE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp19AttachActivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp21AttachDeactivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp10GetTagInfoEPNS0_7TagInfoERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE", ov_nfp_no_device);
HOST_OVERRIDE("_ZN2nn3err9ShowErrorENS_6ResultE", ov_show_error);

HOST_OVERRIDE("nndiagStartup", ov_diag_startup);
HOST_OVERRIDE("_ZN2nn3aoc10InitializeEv", ov_aoc_initialize);
HOST_OVERRIDE("_ZN2nn3aoc17CountAddOnContentEv", ov_aoc_count);
HOST_OVERRIDE("_ZN2nn3aoc16ListAddOnContentEPiii", ov_aoc_list);
HOST_OVERRIDE("_ZN2nn3err26InitializeContextWriterApiEv", ov_err_context);
HOST_OVERRIDE("_ZN2nn4pctl10InitializeEv", ov_pctl_initialize);
HOST_OVERRIDE("_ZN2nn4diag34InitializeApplicationAbortObserverEv", ov_abort_observer);

HOST_OBSERVER("_ZN2nn2os21AllocateAddressRegionEPmm", obs_address_region);
HOST_OBSERVER("_ZN2nn2os19AllocateMemoryPagesEmm", obs_memory_pages);
HOST_OBSERVER("nnosInitialize", obs_os_initialize);
HOST_OBSERVER("nninitInitializeSdkModule", obs_sdk_initialize);
HOST_OBSERVER("nninitStartup", obs_startup);
HOST_OBSERVER("nnMain", obs_main);

HOST_OVERRIDE("_ZN2nn7account10InitializeEv", ov_account_initialize);
HOST_OVERRIDE("_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE", ov_open_user);
HOST_OVERRIDE("_ZN2nn7account9CloseUserERKNS0_10UserHandleE", ov_close_user);
HOST_OVERRIDE("_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE", ov_user_id);
HOST_OVERRIDE("_ZN2nn7account17GetLastOpenedUserEPNS0_3UidE", ov_last_opened_user);
HOST_OVERRIDE("_ZN2nn7account16GetUserExistenceEPbRKNS0_3UidE", ov_user_existence);
HOST_OVERRIDE("_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE", ov_nickname);
HOST_OVERRIDE("_ZN2nn7account12GetUserCountEPi", ov_user_count);
HOST_OVERRIDE("_ZN2nn7account12ListAllUsersEPiPNS0_3UidEi", ov_list_all_users);
HOST_OVERRIDE("_ZN2nn7account13ListOpenUsersEPiPNS0_3UidEi", ov_list_all_users);
HOST_OVERRIDE("_ZN2nn7account18ListQualifiedUsersEPiPNS0_3UidEi", ov_list_all_users);
HOST_OVERRIDE("_ZN2nn7account8OpenUserEPNS0_10UserHandleERKNS0_3UidE", ov_open_user_by_uid);
HOST_OVERRIDE("_ZN2nn7account19OpenPreselectedUserEPNS0_10UserHandleE", ov_open_user);

HOST_OVERRIDE("_ZN2nn4time10InitializeEv", ov_time_initialize);
HOST_OVERRIDE("_ZN2nn4time8FinalizeEv", ov_time_finalize);
HOST_OVERRIDE("_ZN2nn4time13IsInitializedEv", ov_time_ready);
HOST_OVERRIDE("_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE", ov_current_time);
HOST_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE", ov_current_time);
HOST_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock20IsAccuracySufficientEv", ov_ret_one);
HOST_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock21GetSystemClockContextEPNS0_18SystemClockContextE", ov_system_clock_context);
HOST_OVERRIDE("_ZN2nn4time23StandardUserSystemClock21GetSystemClockContextEPNS0_18SystemClockContextE", ov_system_clock_context);
HOST_OVERRIDE("_ZN2nn4time23StandardUserSystemClock3nowEv", ov_posix_now);
HOST_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock3nowEv", ov_posix_now);
HOST_OVERRIDE("_ZN2nn4time19StandardSteadyClock3nowEv", ov_steady_now);
HOST_OVERRIDE("_ZN2nn4time19StandardSteadyClock19GetCurrentTimePointEPNS0_20SteadyClockTimePointE", ov_steady_timepoint);
HOST_OVERRIDE("_ZN2nn4time38GetStandardSteadyClockCurrentTimePointEPNS0_20SteadyClockTimePointE", ov_steady_timepoint);
HOST_OVERRIDE("_ZN2nn4time37GetStandardSteadyClockCurrentTimespecEP8timespec", ov_steady_timespec);
HOST_OVERRIDE("_ZN2nn4time41GetStandardUserSystemClockCurrentTimespecEP8timespec", ov_system_timespec);
HOST_OVERRIDE("_ZN2nn4time44GetStandardNetworkSystemClockCurrentTimespecEP8timespec", ov_system_timespec);
HOST_OVERRIDE("_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE",
              ov_to_calendar);
HOST_OVERRIDE("_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE", ov_to_calendar_utc);
HOST_OVERRIDE("_ZN2nn4time18ToPosixTimeFromUtcERKNS0_12CalendarTimeE", ov_to_posix_utc);
HOST_OVERRIDE("_ZN2nn4time13ClockSnapshotC1Ev", ov_snapshot_init);
HOST_OVERRIDE("_ZN2nn4time13ClockSnapshot29CreateWithStandardSystemClockEPS1_", ov_snapshot_create);
HOST_OVERRIDE("_ZN2nn4time20CalculateSpanBetweenEPNS_8TimeSpanERKNS0_13ClockSnapshotES5_", ov_span_between);
HOST_OVERRIDE("_ZN2nn4time48CalculateStandardUserSystemClockDifferenceByUserERKNS0_13ClockSnapshotES3_",
              ov_user_difference);
HOST_OVERRIDE("_ZN2nn4time21GetDeviceLocationNameEPNS0_12LocationNameE", ov_get_device_location_name);
HOST_OVERRIDE("_ZN2nn8settings6system29GetDeviceTimeZoneLocationNameEPNS_4time12LocationNameE", ov_get_device_location_name);
HOST_OVERRIDE("_ZN2nn8settings6detail29GetDeviceTimeZoneLocationNameEPNS_4time12LocationNameE", ov_get_device_location_name);

HOST_OVERRIDE("_ZN2nn2pl22GetSharedFontLoadStateENS0_14SharedFontTypeE", ov_font_state);
HOST_OVERRIDE("_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE", ov_font_address);
HOST_OVERRIDE("_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE", ov_font_size);

HOST_OVERRIDE("_ZN2nn5audio6detail23ListVirtualInputDevicesEPNS0_15AudioDeviceNameEi", ov_list_virtual_input);

HOST_OVERRIDE("_ZN2nn4nifm10InitializeEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn4nifm18IsNetworkAvailableEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn4nifm20CancelNetworkRequestEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE", ov_nifm_proxy);
HOST_OVERRIDE("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv", ov_nifm_error_result);
HOST_OVERRIDE("_ZN2nn6socket10InitializeERKNS0_6ConfigE", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn6socket8FinalizeEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn6socket13IsInitializedEv", ov_ret_one);
HOST_OVERRIDE("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj", ov_sock_fail);
HOST_OVERRIDE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE", ov_sock_eventfd);
HOST_OVERRIDE("_ZN2nn6socket11EventFdReadEiPm", ov_sock_eventfd_rw);
HOST_OVERRIDE("_ZN2nn6socket12EventFdWriteEim", ov_ret_zero);
HOST_OVERRIDE("nnsocketAccept", ov_sock_fail);
HOST_OVERRIDE("nnsocketBind", ov_sock_fail);
HOST_OVERRIDE("nnsocketClose", ov_ret_zero);
HOST_OVERRIDE("nnsocketConnect", ov_sock_fail);
HOST_OVERRIDE("nnsocketFcntl", ov_ret_zero);
HOST_OVERRIDE("nnsocketFreeAddrInfo", ov_noop);
HOST_OVERRIDE("nnsocketGAIStrError", ov_sock_gai_str);
HOST_OVERRIDE("nnsocketGetAddrInfo", ov_sock_gai);
HOST_OVERRIDE("nnsocketGetAddrInfoWithOptions", ov_sock_gai);
HOST_OVERRIDE("nnsocketGetLastErrno", ov_sock_get_errno);
HOST_OVERRIDE("nnsocketGetPeerName", ov_sock_fail);
HOST_OVERRIDE("nnsocketGetSockName", ov_sock_fail);
HOST_OVERRIDE("nnsocketGetSockOpt", ov_sock_fail);
HOST_OVERRIDE("nnsocketInetHtonl", ov_sock_htonl);
HOST_OVERRIDE("nnsocketInetHtons", ov_sock_htons);
HOST_OVERRIDE("nnsocketInetNtohl", ov_sock_htonl);
HOST_OVERRIDE("nnsocketInetNtohs", ov_sock_htons);
HOST_OVERRIDE("nnsocketListen", ov_sock_fail);
HOST_OVERRIDE("nnsocketPoll", ov_ret_zero);
HOST_OVERRIDE("nnsocketRecv", ov_sock_fail);
HOST_OVERRIDE("nnsocketRecvMsg", ov_sock_fail);
HOST_OVERRIDE("nnsocketResolverGetOption", ov_ret_zero);
HOST_OVERRIDE("nnsocketResolverSetOption", ov_ret_zero);
HOST_OVERRIDE("nnsocketSend", ov_sock_fail);
HOST_OVERRIDE("nnsocketSendMsg", ov_sock_fail);
HOST_OVERRIDE("nnsocketSetLastErrno", ov_sock_set_errno);
HOST_OVERRIDE("nnsocketSetSockOpt", ov_ret_zero);
HOST_OVERRIDE("nnsocketShutdown", ov_ret_zero);
HOST_OVERRIDE("nnsocketSocket", ov_sock_fail);
HOST_OVERRIDE("nnsslInitialize", ov_ret_zero);
HOST_OVERRIDE("nnsslFinalize", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionCreate", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionDestroy", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionDoHandshakeWithCertBuffer", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionGetConnectionId", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionGetNextAlpnProto", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionGetServerCertDetail", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionGetVerifyCertError", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionPeek", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionPending", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionRead", ov_sock_fail);
HOST_OVERRIDE("nnsslConnectionSetHostName", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetIoMode", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetNextAlpnProto", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetOption", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetServerCertBuffer", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetSessionCacheMode", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetSocketDescriptor", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionSetVerifyOption", ov_ret_zero);
HOST_OVERRIDE("nnsslConnectionWrite", ov_sock_fail);
HOST_OVERRIDE("nnsslContextDestroy", ov_ret_zero);
HOST_OVERRIDE("nnsslContextGetContextId", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn3ssl10InitializeEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn3ssl8FinalizeEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn3nsd24GetEnvironmentIdentifierEPNS0_21EnvironmentIdentifierE", ov_nsd_env);
HOST_OVERRIDE("_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_", ov_nsd_resolve);
HOST_OVERRIDE("_ZN2nn2ro12ProtectRelroEPKvS2_S2_S2_S2_", ov_noop);
HOST_OVERRIDE("_ZN2nn2pl17RequestLoadNxFontEPvm", ov_noop);
HOST_OVERRIDE("_ZN2nn5prepo10PlayReport4SendERKNS_7account3UidE", ov_report_send);
HOST_OVERRIDE("_ZN2nn5prepo12AsyncContext18GetCompletionEventEPNS_2os11SystemEventE", ov_async_event);
HOST_OVERRIDE("_ZN2nn5prepo12AsyncContext9GetResultEv", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn5prepo12AsyncContextD1Ev", ov_noop);
HOST_OVERRIDE("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE", ov_account_offline);
HOST_OVERRIDE("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE", ov_account_offline);
HOST_OVERRIDE("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE", ov_account_offline);
HOST_OVERRIDE("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE", ov_account_offline);
HOST_OVERRIDE("_ZN2nn7account12AsyncContext14GetSystemEventEPNS_2os11SystemEventE", ov_async_event);
HOST_OVERRIDE("_ZN2nn7account12AsyncContext9GetResultEv", ov_account_offline);

static void ov_nsa_available(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (out) *out = 0;
    ret_x(c, 0);
}
HOST_OVERRIDE("_ZN2nn7account32IsNetworkServiceAccountAvailableEPbRKNS0_10UserHandleE", ov_nsa_available);

static void ov_account_async_ctor(Cpu *c) {
    uint64_t *ctx = arg_p(c, 0);
    if (ctx) memset(ctx, 0, 16);
    ret_x(c, (uint64_t)(uintptr_t)ctx);
}
HOST_OVERRIDE("_ZN2nn7account12AsyncContextC1Ev", ov_account_async_ctor);
HOST_OVERRIDE("_ZN2nn7account12AsyncContextD1Ev", ov_noop);
HOST_OVERRIDE("_ZN2nn7account12AsyncContext6CancelEv", ov_noop);

static void ov_account_async_done(Cpu *c) {
    uint8_t *out = arg_p(c, 1);
    if (out) *out = 1;
    ret_x(c, 0);
}
HOST_OVERRIDE("_ZN2nn7account12AsyncContext7HasDoneEPb", ov_account_async_done);

static void ov_account_async_move_assign(Cpu *c) {
    uint64_t *dst = arg_p(c, 0);
    uint64_t *src = arg_p(c, 1);
    if (dst && src) memcpy(dst, src, 16);
    ret_x(c, (uint64_t)(uintptr_t)dst);
}
HOST_OVERRIDE("_ZN2nn7account12AsyncContextaSEOS1_", ov_account_async_move_assign);

static void ov_mem_debug_break(Cpu *c) {
    static int s_logged = 0;
    if (s_logged < 40) {
        s_logged++;
        rt_log("mem: nnmem_nlib_debug_break ignored (x0=%#llx x1=%#llx lr=%#llx)%s",
               (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[30],
               s_logged == 40 ? " (further breaks suppressed)" : "");
        // guest frame-pointer chain: who freed a pointer nmalloc does not own
        char bt[256]; int o = 0;
        uint64_t fp = c->x[29];
        for (int i = 0; i < 10 && fp >= 0x100000000ull && fp < 0x8000000000ull && !(fp & 7); i++) {
            uint64_t lr = *(const uint64_t *)(uintptr_t)(fp + 8);
            o += snprintf(bt + o, sizeof bt - o, " %#llx", (unsigned long long)lr);
            if (o >= (int)sizeof bt - 20) break;
            fp = *(const uint64_t *)(uintptr_t)fp;
        }
        rt_log("mem: debug_break backtrace%s", bt);
    }
}
HOST_OVERRIDE("nnmem_nlib_debug_break", ov_mem_debug_break);
HOST_OVERRIDE("_ZN2nn7nlibsdk4heap18NotNMallocPtrErrorEPKv", ov_mem_debug_break);

// nn::chat: Game Chat and Camera streaming stub (offline/disabled)
static void ov_chat_init(Cpu *c) { ret_x(c, 0); }
static void ov_chat_false(Cpu *c) { ret_x(c, 0); }
static void ov_chat_attach_event(Cpu *c) {
    uint64_t ev = arg_x(c, 0);
    if (ev) kern_bind_system_event(c, ev, 0);
    ret_x(c, 0);
}

HOST_OVERRIDE("_ZN2nn4chat10InitializeEv", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat10IsChattingEv", ov_chat_false);
HOST_OVERRIDE("_ZN2nn4chat8FinalizeEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat30InitializeWithoutVirtualCameraEv", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat28FinalizeWithoutVirtualCameraEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat11GetRoomInfoEPNS0_8RoomInfoE", ov_chat_false);
HOST_OVERRIDE("_ZN2nn4chat13SetSharingKeyERKNS0_10SharingKeyE", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat13ShowChatStartERKNS_7account3UidE", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat14GetChattingUidEv", ov_chat_false);
HOST_OVERRIDE("_ZN2nn4chat15ClearSharingKeyEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat20SetScreenSharingModeENS0_17ScreenSharingModeE", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat22ShowCameraSharingGuideEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat24StopLocalCameraStreamingEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat25StartLocalCameraStreamingEv", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat25StopRemoteCameraStreamingERKNS_7account23NetworkServiceAccountIdE", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat26StartRemoteCameraStreamingERKNS_7account23NetworkServiceAccountIdE", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat25GetLocalCameraImageHandleEPKNS0_20LocalCameraVideoTypeEi", ov_chat_false);
HOST_OVERRIDE("_ZN2nn4chat26GetRemoteCameraImageHandleEPKNS0_21RemoteCameraVideoTypeEi", ov_chat_false);
HOST_OVERRIDE("_ZN2nn4chat27RequestNextLocalCameraImageEPNS_2gs16NativeSyncHandleEPNS0_20LocalCameraVideoTypeEiS2_", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat28AttachToRoomInfoChangedEventEPNS_2os11SystemEventE", ov_chat_attach_event);
HOST_OVERRIDE("_ZN2nn4chat28RequestNextRemoteCameraImageEPNS_2gs16NativeSyncHandleEPNS0_21RemoteCameraVideoTypeEiS2_RKNS_7account23NetworkServiceAccountIdE", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat28SetScreenSharingDisallowListEPKNS_7account23NetworkServiceAccountIdEi", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat30ClearScreenSharingDisallowListEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat31FinalizeLocalCameraVideoStorageEPNS0_20LocalCameraVideoTypeE", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat32FinalizeRemoteCameraVideoStorageEPNS0_21RemoteCameraVideoTypeE", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat33AttachToChattingStateChangedEventEPNS_2os11SystemEventE", ov_chat_attach_event);
HOST_OVERRIDE("_ZN2nn4chat33InitializeLocalCameraVideoStorageEPNS0_20LocalCameraVideoTypeEiiiPvm", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat34InitializeRemoteCameraVideoStorageEPNS0_21RemoteCameraVideoTypeEiiiPvm", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat35CalculateLocalCameraVideoBufferSizeEiii", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn4chat36CalculateRemoteCameraVideoBufferSizeEiii", ov_ret_zero);
HOST_OVERRIDE("_ZN2nn4chat36ShowLocalCameraConfigurationVersion0ENS0_25CameraImageProcessingModeE", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat44EndFullScreenWithLocalCameraAvailableSectionEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat46BeginFullScreenWithLocalCameraAvailableSectionEb", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat46BeginFullScreenWithLocalCameraAvailableSectionEbb", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat6detail21InitializeUserServiceEv", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat6detail23CreateUserServiceByHipcEv", ov_chat_init);
HOST_OVERRIDE("_ZN2nn4chat6detail19FinalizeUserServiceEv", ov_noop);
HOST_OVERRIDE("_ZN2nn4chat6detail24DestroyUserServiceByHipcEv", ov_noop);

static void ov_gs_check_init(Cpu *c) {
    rt_log("gs: CheckGraphicsSystemInitialized called");
    if (rt_application_id == 0x0400C3F00006E000ull) {
        *(uint32_t *)(uintptr_t)0x70052db828ull = 1;
    }
}
HOST_OVERRIDE("_ZN2nn2gs3sdk30CheckGraphicsSystemInitializedEPKcm", ov_gs_check_init);

// nn::friends: Friends service stub (offline/no friends)
static void ov_friends_init(Cpu *c) {
    rt_log("friends: Initialize called");
    ret_x(c, 0);
}

static void ov_friends_async_ctor(Cpu *c) {
    uint64_t *ctx = arg_p(c, 0);
    if (ctx) memset(ctx, 0, 32);
    ret_x(c, (uint64_t)(uintptr_t)ctx);
}

static void ov_friends_async_done(Cpu *c) {
    uint8_t *out = arg_p(c, 1);
    if (out) *out = 1;
    ret_x(c, 0);
}

static void ov_friends_check_availability(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (out) *out = 0;
    ret_x(c, 0);
}

static void ov_friends_get_list(Cpu *c) {
    int32_t *out_count = arg_p(c, 0);
    if (out_count) *out_count = 0;
    ret_x(c, 0);
}

static void ov_friends_friend_ctor(Cpu *c) {
    void *p = arg_p(c, 0);
    if (p) memset(p, 0, 0x100);
    ret_x(c, (uint64_t)(uintptr_t)p);
}

static void ov_friends_friend_nickname(Cpu *c) {
    static const char s_empty_name[] = "";
    ret_x(c, (uint64_t)(uintptr_t)s_empty_name);
}

static void ov_friends_friend_false(Cpu *c) {
    ret_x(c, 0);
}

HOST_OVERRIDE("_ZN2nn7friends10InitializeEv", ov_friends_init);
HOST_OVERRIDE("_ZN2nn7friends8FinalizeEv", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends12SetAllocatorEPFPvmEPFvS1_mE", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContextC1Ev", ov_friends_async_ctor);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContextC2Ev", ov_friends_async_ctor);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContextD1Ev", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContextD2Ev", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContext7HasDoneEPb", ov_friends_async_done);
HOST_OVERRIDE("_ZN2nn7friends12AsyncContext6CancelEv", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends25EnsureFriendListAvailableEPNS0_12AsyncContextERKNS_7account3UidE", ov_result_ok);
HOST_OVERRIDE("_ZN2nn7friends27CheckFriendListAvailabilityEPbRKNS_7account3UidE", ov_friends_check_availability);
HOST_OVERRIDE("_ZN2nn7friends13GetFriendListEPiPNS0_6FriendERKNS_7account3UidEiiRKNS0_12FriendFilterE", ov_friends_get_list);
HOST_OVERRIDE("_ZN2nn7friends13GetFriendListEPiPNS0_6FriendEiiRKNS0_12FriendFilterE", ov_friends_get_list);
HOST_OVERRIDE("_ZN2nn7friends14AddPlayHistoryERKNS_7account3UidERKNS0_26PlayHistoryRegistrationKeyERKNS0_15InAppScreenNameESA_", ov_result_ok);
HOST_OVERRIDE("_ZN2nn7friends29GetPlayHistoryRegistrationKeyEPNS0_26PlayHistoryRegistrationKeyERKNS_7account3UidEb", ov_result_ok);
HOST_OVERRIDE("_ZN2nn7friends6FriendC1Ev", ov_friends_friend_ctor);
HOST_OVERRIDE("_ZN2nn7friends6FriendC2Ev", ov_friends_friend_ctor);
HOST_OVERRIDE("_ZNK2nn7friends6Friend11GetNicknameEv", ov_friends_friend_nickname);
HOST_OVERRIDE("_ZNK2nn7friends6Friend12GetAccountIdEv", ov_friends_friend_false);
HOST_OVERRIDE("_ZNK2nn7friends6Friend7IsValidEv", ov_friends_friend_false);
HOST_OVERRIDE("_ZN2nn7friends6detail3ipc10InitializeEv", ov_friends_init);
HOST_OVERRIDE("_ZN2nn7friends6detail3ipc8FinalizeEv", ov_noop);
HOST_OVERRIDE("_ZN2nn7friends6detail17ShimLibraryGlobal10InitializeEv", ov_friends_init);
HOST_OVERRIDE("_ZN2nn7friends6detail17ShimLibraryGlobal8FinalizeEv", ov_noop);

static PlatMutex s_fallback_lock;
static uint64_t g_heap_fallback_bump = 0;
static uint64_t g_heap_fallback_end = 0;
static uint64_t s_expheap_hook_addr = 0;

extern void __real_f_70003f24c0(Cpu *c) __attribute__((weak));
extern void __real_f_7000d32270(Cpu *c) __attribute__((weak));
extern void __real_f_7000d31b20(Cpu *c) __attribute__((weak));
extern void __real_f_7000cdc2f0(Cpu *c) __attribute__((weak));
extern void __real_f_70005e10a0(Cpu *c) __attribute__((weak));
extern void __real_f_70003f2d70(Cpu *c) __attribute__((weak));

// MKW_HEAP_TRACE=1 (diagnostics, foliage plan): the first allocations and free-size queries of the game's sead heaps.
// MKW_HEAP_LEGACY=1: the old workarounds for a starving game heap (report 32 MiB when a heap is full, satisfy a
// failed ExpHeap::alloc from a bump region). They were masking the loader binding the SDK's operator delete to the
// SDK's own _ZdlPv (game-heap blocks were never freed, see loader.c resolve); the bump region also lands inside a
// live game heap. Off by default: the game's own answers are used.
static int heap_legacy(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("MKW_HEAP_LEGACY");
        on = e && *e == '1';
    }
    return on;
}

// MKW_ANIM_CHECK heap tracker (defined at the end of this file): every ExpHeap alloc/free/resize is recorded so the
// events touching an animation keyframe buffer can be reported.
enum { AHT_ALLOC, AHT_FREE, AHT_RESIZE_FRONT, AHT_RESIZE_BACK };
static int anim_on(void);
static void aht_event(int kind, uint64_t heap, uint64_t ptr, uint64_t size, uint64_t res, Cpu *c, uint64_t lr);

static const char *heap_name(uint64_t heap, char *buf, size_t n);   // below

// MKW_SOUND_HEAP_TRACE: live bytes in the "SLinkUserCreate" heap (allocations minus frees, the freed block's size from
// its ExpHeap header at ptr-0x20+0x18), per allocating guest caller; reported when its getMaxAllocatableSize fails.
static uint64_t slink_heap;
static int64_t slink_live, slink_n_alloc, slink_n_free;
static struct { uint64_t lr; int64_t n, live; } slink_by[32];
static int slink_nby;
static SRWLOCK slink_l = SRWLOCK_INIT;
static struct { uint64_t ptr, lr; } slink_blk[8192];   // open addressing: ptr 0 empty, 1 deleted
static void slink_note(uint64_t heap, uint64_t ptr, uint64_t size, uint64_t lr, int alloc) {
    if (!slink_heap) {
        static uint64_t seen_h[512];
        uint32_t slot = (uint32_t)((heap >> 4) * 0x9E3779B1u) & 511;
        if (seen_h[slot] == heap) return;
        char nm[64];
        heap_name(heap, nm, sizeof nm);
        if (strncmp(nm, "SLinkUserCreate ", 16)) {
            seen_h[slot] = heap;
            return;
        }
        slink_heap = heap;
    }
    if (heap != slink_heap || !ptr) return;
    AcquireSRWLockExclusive(&slink_l);
    uint32_t h = (uint32_t)((ptr >> 4) * 0x9E3779B97F4A7C15ull >> 51) & 8191;
    if (alloc) {
        for (uint32_t k = 0; k < 8192; k++, h = (h + 1) & 8191)
            if (slink_blk[h].ptr <= 1) { slink_blk[h].ptr = ptr, slink_blk[h].lr = lr; break; }
    } else {
        lr = 0;
        for (uint32_t k = 0; k < 8192 && slink_blk[h].ptr; k++, h = (h + 1) & 8191)
            if (slink_blk[h].ptr == ptr) { lr = slink_blk[h].lr, slink_blk[h].ptr = 1; break; }
        if (plat_mem_readable(ptr - 0x20, 0x20)) memcpy(&size, (const void *)(uintptr_t)(ptr - 0x20 + 0x18), 8);
    }
    int64_t s = alloc ? (int64_t)size : -(int64_t)size;
    slink_live += s, alloc ? slink_n_alloc++ : slink_n_free++;
    int i;
    for (i = 0; i < slink_nby && slink_by[i].lr != lr; i++) {}
    if (i == slink_nby && slink_nby < 32) slink_by[slink_nby++] = (typeof(slink_by[0])){lr, 0, 0};
    if (i < slink_nby) slink_by[i].n += alloc ? 1 : -1, slink_by[i].live += s;
    ReleaseSRWLockExclusive(&slink_l);
}
static void slink_report(const char *why) {
    AcquireSRWLockShared(&slink_l);
    rt_log("dslink %s: heap %#llx, %lld allocations, %lld frees, %.1f KiB live", why, (unsigned long long)slink_heap,
           (long long)slink_n_alloc, (long long)slink_n_free, slink_live / 1024.0);
    for (int i = 0; i < slink_nby; i++)
        rt_log("dslink   by %#llx: %lld blocks live, %.1f KiB", (unsigned long long)slink_by[i].lr, (long long)slink_by[i].n,
               slink_by[i].live / 1024.0);
    ReleaseSRWLockShared(&slink_l);
}

static int heap_trace(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("MKW_HEAP_TRACE");
        on = e && *e == '1';
    }
    return on;
}

// A sead heap's name: the first pointer in its first 0x100 bytes to a short printable NUL-terminated string.
static const char *heap_name(uint64_t heap, char *buf, size_t n) {
    snprintf(buf, n, "?");
    if (!heap || !plat_mem_readable(heap, 0x100)) return buf;
    for (int o = 8; o < 0x100; o += 8) {
        uint64_t p;
        memcpy(&p, (const void *)(uintptr_t)(heap + o), 8);
        if (p < 0x100000000ull || p >= 0x8000000000ull || !plat_mem_readable(p, 24)) continue;
        const char *s = (const char *)(uintptr_t)p;
        int k = 0;
        while (k < 23 && s[k] >= 0x20 && s[k] < 0x7f) k++;
        if (k >= 3 && s[k] == 0) {
            snprintf(buf, n, "%s (+%#x)", s, o);
            return buf;
        }
    }
    return buf;
}

void __wrap_f_70005e10a0(Cpu *c) {
    uint64_t trace_heap = c->x[0], trace_lr = c->x[30], zheap = trace_heap, zlr = trace_lr, zchild = c->x[20];
    if (__real_f_70005e10a0) {
        __real_f_70005e10a0(c);
    } else {
        GuestFn orig = rt_dispatch_lookup(0x70005e10a0ull);
        if (orig) orig(c);
    }
    static volatile int32_t nq;
    if (heap_trace() && __atomic_fetch_add(&nq, 1, __ATOMIC_RELAXED) < 60)
        rt_log("htrace maxalloc heap %#llx -> %#llx (lr %#llx)", (unsigned long long)trace_heap,
               (unsigned long long)c->x[0], (unsigned long long)trace_lr);
    if (c->x[0] == 0) {
        static volatile int32_t nz;
        int32_t k = __atomic_fetch_add(&nz, 1, __ATOMIC_RELAXED) + 1;
        if (k <= 20 || !(k & (k - 1))) {   // the first 20, then at powers of two with the guest callers
            char pn[64], cn[64];
            if (slink_heap && (k == 1 || k == 16 || k == 128)) slink_report(k == 1 ? "at the first full report" : "later");
            if (slink_heap && (k == 1 || k == 64)) {
                for (int w = 0; w < 2; w++) {
                    uint64_t hp = w ? zchild : zheap, q[16] = {0};
                    if (hp && plat_mem_readable(hp, 0x80)) memcpy(q, (const void *)(uintptr_t)hp, 0x80);
                    rt_log("heap: %s %#llx words: %llx %llx %llx %llx | %llx %llx %llx %llx | %llx %llx %llx %llx | %llx %llx %llx %llx",
                           w ? "child" : "parent", (unsigned long long)hp, (unsigned long long)q[0], (unsigned long long)q[1],
                           (unsigned long long)q[2], (unsigned long long)q[3], (unsigned long long)q[4], (unsigned long long)q[5],
                           (unsigned long long)q[6], (unsigned long long)q[7], (unsigned long long)q[8], (unsigned long long)q[9],
                           (unsigned long long)q[10], (unsigned long long)q[11], (unsigned long long)q[12], (unsigned long long)q[13],
                           (unsigned long long)q[14], (unsigned long long)q[15]);
                }
            }
            rt_log("heap: f_70005e10a0 getMaxAllocatableSize heap %#llx '%s' returned 0 (lr %#llx; call %d; child heap x20 %#llx '%s'; "
                   "caller %#llx)%s",
                   (unsigned long long)zheap, heap_name(zheap, pn, sizeof pn), (unsigned long long)zlr, k, (unsigned long long)zchild,
                   heap_name(zchild, cn, sizeof cn),
                   (unsigned long long)(c->x[29] && plat_mem_readable(c->x[29], 16) ? *(const uint64_t *)(uintptr_t)(c->x[29] + 8) : 0),
                   heap_legacy() ? ", legacy: reporting 32 MiB" : "");
        }
        if (heap_legacy()) c->x[0] = 32 * 1024 * 1024;
    }
}

void __wrap_f_70003f2d70(Cpu *c) {
    uint64_t zheap = c->x[0], zlr = c->x[30];
    if (__real_f_70003f2d70) {
        __real_f_70003f2d70(c);
    } else {
        GuestFn orig = rt_dispatch_lookup(0x70003f2d70ull);
        if (orig) orig(c);
    }
    if (c->x[0] == 0) {
        static volatile int32_t nz;
        if (__atomic_fetch_add(&nz, 1, __ATOMIC_RELAXED) < 20)
            rt_log("heap: f_70003f2d70 getMaxAllocatableSize heap %#llx returned 0 (lr %#llx)%s", (unsigned long long)zheap,
                   (unsigned long long)zlr, heap_legacy() ? ", legacy: reporting 32 MiB" : "");
        if (heap_legacy()) c->x[0] = 32 * 1024 * 1024;
    }
}

void __wrap_f_7000cdc2f0(Cpu *c) {
    rt_log("camera: f_7000cdc2f0 PlatformLocalCameraMgr::Initialize bypassed");
}

void __wrap_f_7000d32270(Cpu *c) {
    rt_log("chat: f_7000d32270 camera video storage init bypassed");
}

void __wrap_f_7000d31b20(Cpu *c) {
    rt_log("chat: f_7000d31b20 remote camera video storage init bypassed");
}

void __wrap_f_70003f24c0(Cpu *c) {
    uint64_t heap = c->x[0];
    uint64_t size = c->x[1];
    int32_t raw_align = (int32_t)c->x[2];
    int64_t mag_align = raw_align < 0 ? -(int64_t)raw_align : (int64_t)raw_align;
    uint64_t align = (uint64_t)mag_align;
    uint64_t lr = c->x[30];

    // Try original guest ExpHeap::alloc first
    if (__real_f_70003f24c0) {
        __real_f_70003f24c0(c);
    } else {
        static GuestFn s_orig_expheap_alloc = NULL;
        if (!s_orig_expheap_alloc) {
            s_orig_expheap_alloc = rt_dispatch_lookup(0x70003f24c0ull);
        }
        if (s_orig_expheap_alloc) {
            s_orig_expheap_alloc(c);
        } else {
            c->x[0] = 0;
        }
    }

    static volatile int32_t na;
    if (heap_trace() && __atomic_fetch_add(&na, 1, __ATOMIC_RELAXED) < 300)
        rt_log("htrace alloc heap %#llx size %#llx align %lld -> %#llx (lr %#llx)", (unsigned long long)heap,
               (unsigned long long)size, (long long)raw_align, (unsigned long long)c->x[0], (unsigned long long)lr);
    if (c->x[0] != 0) {
        if (anim_on()) aht_event(AHT_ALLOC, heap, c->x[0], size, 0, c, lr);
        // Collision investigation (HANDOVER 12e): allocations of 64 KiB or more under the "Sound" heap, with the heap
        // chain and guest callers (the Sound heap is full at boot on PC, so SLinkUserCreate cannot grow).
        static int snd_on = -1;
        if (snd_on < 0) snd_on = cfg_bool("MKW_SOUND_HEAP_TRACE");
        // direct allocations in the Sound heap (any size), summed per guest caller: a line per 256 and per 4 MiB more
        static uint64_t snd_heap;
        if (snd_on && !snd_heap) {
            static uint64_t seen_h[512];   // heaps already known not to be Sound (a direct-mapped cache)
            uint32_t slot = (uint32_t)((heap >> 4) * 0x9E3779B1u) & 511;
            if (seen_h[slot] != heap) {
                char nm[64];
                heap_name(heap, nm, sizeof nm);
                if (!strncmp(nm, "Sound ", 6)) snd_heap = heap;
                else seen_h[slot] = heap;
            }
        }
        if (snd_on) slink_note(heap, c->x[0], size, lr, 1);
        if (snd_on && heap == snd_heap) {
            static struct { uint64_t lr, n, bytes; } agg[64];
            static int nagg;
            static uint64_t total, count, next_report = 4u << 20;
            static SRWLOCK l = SRWLOCK_INIT;
            AcquireSRWLockExclusive(&l);
            int i;
            for (i = 0; i < nagg && agg[i].lr != lr; i++) {}
            if (i == nagg && nagg < 64) agg[nagg++] = (typeof(agg[0])){lr, 0, 0};
            if (i < nagg) agg[i].n++, agg[i].bytes += size;
            total += size, count++;
            if (total >= next_report || !(count & 255)) {
                next_report = total + (4u << 20);
                uint64_t hs = 0;
                memcpy(&hs, (const void *)(uintptr_t)(heap + 0x30), 8);
                rt_log("dsound direct: %llu allocations, %.1f MiB of the %.1f MiB Sound heap so far; by caller:", (unsigned long long)count,
                       total / 1048576.0, hs / 1048576.0);
                for (int k = 0; k < nagg; k++)
                    if (agg[k].bytes >= (256u << 10))
                        rt_log("dsound   lr %#llx: %llu allocations, %.2f MiB", (unsigned long long)agg[k].lr, (unsigned long long)agg[k].n,
                               agg[k].bytes / 1048576.0);
            }
            ReleaseSRWLockExclusive(&l);
        }
        if (size >= 0x10000) {
            int on = snd_on && snd_heap;
            if (on) {
                char chain[200];
                int o = 0, under = 0;
                uint64_t h = heap;
                for (int dpt = 0; dpt < 6 && h && plat_mem_readable(h, 0x40); dpt++) {
                    char nm[64];
                    heap_name(h, nm, sizeof nm);
                    if (!strncmp(nm, "Sound ", 6)) under = 1;
                    char *sp = strstr(nm, " (+");
                    if (sp) *sp = 0;
                    o += snprintf(chain + o, sizeof chain - o, "%s%s", dpt ? "<" : "", nm);
                    if (o >= (int)sizeof chain - 8) break;
                    memcpy(&h, (const void *)(uintptr_t)(h + 0x38), 8);
                }
                if (under) {
                    uint64_t lr2 = 0, fp = c->x[29];
                    if (fp && plat_mem_readable(fp, 16)) lr2 = *(const uint64_t *)(uintptr_t)(fp + 8);
                    rt_log("dsound alloc %#llx bytes (%.1f MiB) in %s -> %#llx (lr %#llx, caller %#llx)", (unsigned long long)size,
                           size / 1048576.0, chain, (unsigned long long)c->x[0], (unsigned long long)lr, (unsigned long long)lr2);
                }
            }
        }
        return; // Normal guest allocation succeeded
    }
    if (!heap_legacy()) {
        static volatile int32_t nf;
        if (__atomic_fetch_add(&nf, 1, __ATOMIC_RELAXED) < 20)
            rt_log("heap: ExpHeap::alloc heap %#llx size %#llx align %d failed (lr %#llx)", (unsigned long long)heap,
                   (unsigned long long)size, raw_align, (unsigned long long)lr);
        return;
    }

    // Allocation failed - provide fallback allocation.
    // Large requests (hundreds of MiB / a few GiB) are required at boot; refusing them AVs on a null
    // block. VRAM blow-up is prevented by refusing to mirror pools ≥512 MiB (vk_memory.c), not by
    // starving the guest heap here. Cap the fallback region so a thrashing ExpHeap cannot commit
    // tens of GiB (n06 hit 32 GiB process commit before rt_dispatch pc=0).
    // Boot alone asks for multi‑GiB blocks via fallback; 2 GiB aborted before title (n07). 8 GiB
    // covers boot + early title; refuse past that so thrash cannot reach the 32 GiB n06 death.
    enum { FALLBACK_CAP = 8ull << 30 };
    plat_mutex_lock(&s_fallback_lock);

    const RtGuestLayout *L = mem_layout();
    if (!g_heap_fallback_bump) {
        // Start fallback region inside the reserved heap at 1152 MiB offset
        g_heap_fallback_bump = L->heap.base + 0x48000000ull;
        g_heap_fallback_end = g_heap_fallback_bump;
    }
    uint64_t fallback_base = L->heap.base + 0x48000000ull;
    if (g_heap_fallback_bump - fallback_base >= FALLBACK_CAP) {
        plat_mutex_unlock(&s_fallback_lock);
        static int logged;
        if (logged++ < 8)
            rt_log("heap: fallback cap %u MiB reached, refusing %#llx", (unsigned)(FALLBACK_CAP >> 20),
                   (unsigned long long)size);
        c->x[0] = 0;
        return;
    }

    if (size == 0) size = 16;
    if (align < 8) align = 8;
    if (align > 65536) align = 65536;
    if ((align & (align - 1)) != 0) {
        uint64_t p = 1;
        while (p < align) p <<= 1;
        align = p;
    }

    // Align user_ptr with 0x20 bytes reserved before it for the ExpHeap MemBlock header
    uint64_t user_ptr = (g_heap_fallback_bump + 0x20ull + align - 1) & ~(align - 1);
    uint64_t block_hdr = user_ptr - 0x20ull;
    uint64_t alloc_sz = (size + 15ull) & ~15ull;
    uint64_t next_bump = user_ptr + alloc_sz;

    if (next_bump > g_heap_fallback_end) {
        size_t need = (size_t)(next_bump - g_heap_fallback_end);
        size_t commit_sz = (need + (64ull << 20) - 1) & ~((64ull << 20) - 1);
        if (!plat_mem_commit(g_heap_fallback_end, commit_sz)) {
            plat_mutex_unlock(&s_fallback_lock);
            rt_log("heap: failed to commit fallback %#llx bytes at %#llx, refusing",
                   (unsigned long long)commit_sz, (unsigned long long)g_heap_fallback_end);
            c->x[0] = 0;
            return;
        }
        g_heap_fallback_end += commit_sz;
        mem_expand_heap(g_heap_fallback_end);
    }

    // Populate ExpHeap MemBlock header
    *(uint16_t *)(uintptr_t)(block_hdr + 0x10) = 0;
    *(uint16_t *)(uintptr_t)(block_hdr + 0x12) = 0x20;
    *(uint32_t *)(uintptr_t)(block_hdr + 0x14) = 0;
    *(uint64_t *)(uintptr_t)(block_hdr + 0x18) = size;
    memset((void *)(uintptr_t)user_ptr, 0, (size_t)size);

    g_heap_fallback_bump = next_bump;

    plat_mutex_unlock(&s_fallback_lock);

    rt_log("heap: fallback alloc for heap %#llx size %#llx align %#llx -> %#llx (lr %#llx)",
           (unsigned long long)heap, (unsigned long long)size, (unsigned long long)align,
           (unsigned long long)user_ptr, (unsigned long long)lr);

    c->x[0] = user_ptr;
}

static void ov_expheap_alloc_hook(Cpu *c) {
    __wrap_f_70003f24c0(c);
}

static uint64_t s_obj_dispose_hook_addr = 0;

void __wrap_f_700087b9b0(Cpu *c) {
    uint64_t obj = c->x[0];
    if (!obj) return;

    uint64_t x20 = 0;

    // Check condition 1 (+0xf0)
    uint64_t ea1 = obj + 0xf0ull;
    uint64_t v1 = *(uint64_t *)(uintptr_t)ea1;
    if (v1) {
        uint64_t a0 = *(uint64_t *)(uintptr_t)(v1 + 0x38ull);
        if (a0) {
            uint64_t vt = *(uint64_t *)(uintptr_t)a0;
            uint64_t fn = *(uint64_t *)(uintptr_t)(vt + 0xa0ull);
            uint64_t res = rt_call(c, fn, 2, a0, obj);
            if (res & 1) {
                uint64_t v = *(uint64_t *)(uintptr_t)ea1;
                x20 = *(uint64_t *)(uintptr_t)(v + 0x38ull);
            }
        }
    }

    // Check condition 2 (+0x118)
    if (!x20) {
        uint64_t ea2 = obj + 0x118ull;
        uint64_t v2 = *(uint64_t *)(uintptr_t)ea2;
        if (v2) {
            uint64_t a0 = *(uint64_t *)(uintptr_t)(v2 + 0x38ull);
            if (a0) {
                uint64_t vt = *(uint64_t *)(uintptr_t)a0;
                uint64_t fn = *(uint64_t *)(uintptr_t)(vt + 0xa0ull);
                uint64_t res = rt_call(c, fn, 2, a0, obj);
                if (res & 1) {
                    uint64_t v = *(uint64_t *)(uintptr_t)ea2;
                    x20 = *(uint64_t *)(uintptr_t)(v + 0x38ull);
                }
            }
        }
    }

    // Call vtable[0](obj) (destructor)
    uint64_t obj_vt = *(uint64_t *)(uintptr_t)obj;
    uint64_t dtor = *(uint64_t *)(uintptr_t)obj_vt;
    rt_call(c, dtor, 1, obj);

    // If x20 is valid, call vtable[0x38/8](x20, obj) (unlink)
    if (x20) {
        uint64_t cnt_vt = *(uint64_t *)(uintptr_t)x20;
        uint64_t unlink_fn = *(uint64_t *)(uintptr_t)(cnt_vt + 0x38ull);
        rt_call(c, unlink_fn, 2, x20, obj);
    } else {
        rt_log("heap: __wrap_f_700087b9b0 object %#llx unattached, unlink skipped", (unsigned long long)obj);
    }
}

void __real_f_700025a680(Cpu *c) __attribute__((weak));
void __wrap_f_700025a680(Cpu *c) {
    uint64_t ea = 0x70036f4148ull;
    uint64_t mgr_ptr = *(uint64_t *)(uintptr_t)ea;
    uint64_t mgr = mgr_ptr ? *(uint64_t *)(uintptr_t)mgr_ptr : 0;
    if (!mgr) {
        c->x[0] = 0;
        return;
    }
    if (__real_f_700025a680) {
        __real_f_700025a680(c);
    } else {
        c->x[0] = 0;
    }
}

void __real_f_70006cec80(Cpu *c) __attribute__((weak));
void __wrap_f_70006cec80(Cpu *c) {
    uint64_t obj = c->x[0];
    if (obj) {
        uint32_t count = *(const uint32_t *)(uintptr_t)(obj + 0x240);
        uint64_t arr_addr = *(const uint64_t *)(uintptr_t)(obj + 0x248);
        if (arr_addr && count > 0 && count < 0x10000) {
            uint64_t *arr = (uint64_t *)(uintptr_t)arr_addr;
            for (uint32_t i = 0; i < count; i++) {
                uint64_t item = arr[i];
                if (item) {
                    uint64_t vtbl = *(const uint64_t *)(uintptr_t)item;
                    if (!vtbl) {
                        rt_log("system: f_70006cec80 item %u (%#llx) null vtable, clearing", i, (unsigned long long)item);
                        arr[i] = 0;
                    } else {
                        uint64_t fn = *(const uint64_t *)(uintptr_t)(vtbl + 0x58);
                        if (!fn) {
                            rt_log("system: f_70006cec80 item %u (%#llx, vtbl %#llx) null fn+0x58, clearing", i, (unsigned long long)item, (unsigned long long)vtbl);
                            arr[i] = 0;
                        }
                    }
                }
            }
        }
    }
    if (__real_f_70006cec80) {
        __real_f_70006cec80(c);
    }
}

void __real_f_70006d2800(Cpu *c) __attribute__((weak));
void __wrap_f_70006d2800(Cpu *c) {
    uint64_t obj = c->x[0];
    if (!obj || obj < 0x1000) return;
    uint64_t vtable = *(const uint64_t *)(uintptr_t)obj;
    if (vtable < 0x7000000000ull || vtable >= 0x7005400000ull) {
        return;
    }
    if (__real_f_70006d2800) {
        __real_f_70006d2800(c);
    }
}

// Sibling of f_70006d2800: night AV read 0x111 with x8=0x101 (garbage "vtable" +0x10). Same guard.
void __real_f_70006d19a0(Cpu *c) __attribute__((weak));
void __wrap_f_70006d19a0(Cpu *c) {
    uint64_t obj = c->x[0];
    if (!obj || obj < 0x100000000ull || obj >= 0x8000000000ull) return;
    uint64_t vtable = *(const uint64_t *)(uintptr_t)obj;
    if (vtable < 0x7000000000ull || vtable >= 0x7005400000ull) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_70006d19a0 bad vt %#llx on %#llx, skip", (unsigned long long)vtable,
                   (unsigned long long)obj);
        return;
    }
    if (__real_f_70006d19a0) {
        __real_f_70006d19a0(c);
    }
}

// n07: after a refused fallback alloc, f_7000ae4020 ran with x0=0 and wrote address 0x8.
void __real_f_7000ae4020(Cpu *c) __attribute__((weak));
void __wrap_f_7000ae4020(Cpu *c) {
    if (!c->x[0] || c->x[0] < 0x100000000ull || c->x[0] >= 0x8000000000ull) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_7000ae4020 null this %#llx, skip", (unsigned long long)c->x[0]);
        return;
    }
    mkw_soft_av_call(__real_f_7000ae4020, c, "f_7000ae4020");
}

void __real_f_7000ae3ba0(Cpu *c) __attribute__((weak));
void __wrap_f_7000ae3ba0(Cpu *c) {
    if (!c->x[0] || c->x[0] < 0x100000000ull || c->x[0] >= 0x8000000000ull) return;
    mkw_soft_av_call(__real_f_7000ae3ba0, c, "f_7000ae3ba0");
}

void __real_f_70006d30b0(Cpu *c) __attribute__((weak));
void __wrap_f_70006d30b0(Cpu *c) {
    uint64_t obj = c->x[0];
    if (!obj || obj < 0x1000) return;
    uint64_t vtable = *(const uint64_t *)(uintptr_t)obj;
    if (vtable < 0x7000000000ull || vtable >= 0x7005400000ull) return;
    mkw_soft_av_call(__real_f_70006d30b0, c, "f_70006d30b0");
}

// n13: read 0x10 in f_70008c2610 (parent of the 6ed* heap path). Soft-catch near-null.
void __real_f_70008c2610(Cpu *c) __attribute__((weak));
void __wrap_f_70008c2610(Cpu *c) {
    mkw_soft_av_call(__real_f_70008c2610, c, "f_70008c2610");
}
void __real_f_70008c25a0(Cpu *c) __attribute__((weak));
void __wrap_f_70008c25a0(Cpu *c) {
    mkw_soft_av_call(__real_f_70008c25a0, c, "f_70008c25a0");
}

// n14: read 0x109 in f_70000c1400.
void __real_f_70000c1400(Cpu *c) __attribute__((weak));
void __wrap_f_70000c1400(Cpu *c) {
    mkw_soft_av_call(__real_f_70000c1400, c, "f_70000c1400");
}
void __real_f_70000be740(Cpu *c) __attribute__((weak));
void __wrap_f_70000be740(Cpu *c) {
    mkw_soft_av_call(__real_f_70000be740, c, "f_70000be740");
}

// Broad soft-catch: f_7004566d10 sits above many of the night AVs (6ed*, 8c*, 0c*).
void __real_f_7004566d10(Cpu *c) __attribute__((weak));
void __wrap_f_7004566d10(Cpu *c) {
    mkw_soft_av_call(__real_f_7004566d10, c, "f_7004566d10");
}

// Night AV: f_70006edb40 reading address 0x9 with x1=0 (null object field at +9). Skip when the
// second argument is a near-null / non-guest pointer; x0 alone was a valid heap object.
void __real_f_70006edb40(Cpu *c) __attribute__((weak));
void __wrap_f_70006edb40(Cpu *c) {
    uint64_t a0 = c->x[0], a1 = c->x[1];
    int a0_ok = a0 >= 0x100000000ull && a0 < 0x8000000000ull;
    int a1_ok = !a1 || (a1 >= 0x100000000ull && a1 < 0x8000000000ull);
    if (!a0_ok || !a1_ok) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_70006edb40 bad args %#llx %#llx, skip", (unsigned long long)a0,
                   (unsigned long long)a1);
        return;
    }
    mkw_soft_av_call(__real_f_70006edb40, c, "f_70006edb40");
}

// Sibling: n03 wrote address 0x40 inside f_70006ed4a0 after ~60s of good frames (null +0x40 store).
void __real_f_70006ed4a0(Cpu *c) __attribute__((weak));
void __wrap_f_70006ed4a0(Cpu *c) {
    uint64_t a0 = c->x[0];
    if (!a0 || a0 < 0x100000000ull || a0 >= 0x8000000000ull) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_70006ed4a0 bad this %#llx, skip", (unsigned long long)a0);
        return;
    }
    mkw_soft_av_call(__real_f_70006ed4a0, c, "f_70006ed4a0");
}

// n04: read 0xffffffffffffffff in f_70003624e0 (poison pointer chase). Soft-catch and continue.
void __real_f_70003624e0(Cpu *c) __attribute__((weak));
void __wrap_f_70003624e0(Cpu *c) {
    uint64_t a0 = c->x[0];
    if (!a0 || a0 < 0x100000000ull || a0 >= 0x8000000000ull) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_70003624e0 bad this %#llx, skip", (unsigned long long)a0);
        return;
    }
    mkw_soft_av_call(__real_f_70003624e0, c, "f_70003624e0");
}

void __real_f_7000361930(Cpu *c) __attribute__((weak));
void __wrap_f_7000361930(Cpu *c) {
    mkw_soft_av_call(__real_f_7000361930, c, "f_7000361930");
}

void __real_f_70006f9c50(Cpu *c) __attribute__((weak));
void __wrap_f_70006f9c50(Cpu *c) {
    uint64_t x19 = c->x[0];
    if (x19 >= 0x100000000ull && x19 < 0x8000000000ull) {
        uint64_t x24 = x19 + 128; // 0x80
        uint64_t x22 = *(const uint64_t *)(uintptr_t)(x24 + 8);
        if (x22 > 1 && x22 < 0x8000000000ull) {
            int32_t off = *(const int32_t *)(uintptr_t)(x22 + 0x7c);
            uint64_t x8 = x24 + off;
            if (x8 >= 0x100000000ull && x8 < 0x8000000000ull && (x8 & 7) == 0) {
                uint64_t next = *(const uint64_t *)(uintptr_t)(x8);
                uint64_t prev = *(const uint64_t *)(uintptr_t)(x8 + 8);
                int next_ok = (next == 0) || (next >= 0x100000000ull && next < 0x8000000000ull && (next & 7) == 0);
                int prev_ok = (prev == 0) || (prev >= 0x100000000ull && prev < 0x8000000000ull && (prev & 7) == 0);
                if (!next_ok || !prev_ok) {
                    rt_log("heap: f_70006f9c50 invalid list node at %#llx (next=%#llx, prev=%#llx), clearing",
                           (unsigned long long)x8, (unsigned long long)next, (unsigned long long)prev);
                    *(uint64_t *)(uintptr_t)x8 = 0;
                    *(uint64_t *)(uintptr_t)(x8 + 8) = 0;
                }
            }
        }
    }
    if (__real_f_70006f9c50) {
        __real_f_70006f9c50(c);
    }
}

void __real_f_70001a6e80(Cpu *c) __attribute__((weak));
void __wrap_f_70001a6e80(Cpu *c) {
    uint64_t obj = c->x[0];
    if (obj >= 0x100000000ull && obj < 0x8000000000ull) {
        for (int off = 0x128; off <= 0x130; off += 8) {
            uint64_t child = *(const uint64_t *)(uintptr_t)(obj + off);
            if (child >= 0x100000000ull && child < 0x8000000000ull) {
                uint64_t vt = *(const uint64_t *)(uintptr_t)child;
                if (vt >= 0x100000000ull && vt < 0x8000000000ull) {
                    uint64_t fn = *(const uint64_t *)(uintptr_t)(vt + 0x48);
                    if (!rt_dispatch_lookup(fn)) {
                        rt_log("system: f_70001a6e80 obj %#llx child+0x%x (%#llx, vt %#llx) invalid fn+0x48 %#llx, clearing",
                               (unsigned long long)obj, off, (unsigned long long)child, (unsigned long long)vt, (unsigned long long)fn);
                        *(uint64_t *)(uintptr_t)(obj + off) = 0;
                    }
                } else {
                    rt_log("system: f_70001a6e80 obj %#llx child+0x%x (%#llx) invalid vt %#llx, clearing",
                           (unsigned long long)obj, off, (unsigned long long)child, (unsigned long long)vt);
                    *(uint64_t *)(uintptr_t)(obj + off) = 0;
                }
            } else if (child != 0) {
                *(uint64_t *)(uintptr_t)(obj + off) = 0;
            }
        }
    }
    // n06: after clearing, the lifted body still chased a null vtable (rt_dispatch pc 0 at lr+0x70).
    mkw_soft_av_call(__real_f_70001a6e80, c, "f_70001a6e80");
}

static uint64_t s_noop_hook_addr = 0;

// Guest heap / module pointers used by the MKW system hooks below. Small integers like 0x101 show up when a
// vtable slot or CriticalSection pointer is corrupted; the lifted code then raw-dereferences them (including
// through __atomic_load_n) and the host AVs.
static int mkw_guest_ptr(uint64_t p) {
    return p >= 0x100000000ull && p < 0x8000000000ull;
}

// Shared by f_7000192d60 / f_7000193200: walk the delegate list at this+0x20 and replace undeliverable
// function pointers with the registered noop. Only the known slot10 layout is patched (see wrap comment).
static void mkw_sanitize_delegate_list(uint64_t x0, const char *who) {
    if (!mkw_guest_ptr(x0)) return;
    uint64_t obj22 = *(const uint64_t *)(uintptr_t)(x0 + 0x20);
    if (!mkw_guest_ptr(obj22)) return;
    uint64_t head = obj22 + 0x310;
    uint64_t end_val = *(const uint64_t *)(uintptr_t)(obj22 + 0x318);
    if (end_val == head || !mkw_guest_ptr(end_val)) return;
    int32_t off = *(const int32_t *)(uintptr_t)(obj22 + 0x324);
    uint64_t term = head - (uint64_t)(int64_t)off;
    uint64_t cur = end_val - (uint64_t)(int64_t)off;
    uint64_t noop = s_noop_hook_addr ? s_noop_hook_addr : 0x70029a7700ull;
    int safety = 0;
    while (cur != term && safety++ < 1000) {
        if (!mkw_guest_ptr(cur)) break;
        uint64_t vt = *(const uint64_t *)(uintptr_t)cur;
        if (mkw_guest_ptr(vt)) {
            uint64_t slot10 = *(const uint64_t *)(uintptr_t)(vt + 0x10);
            // Only the known delegate layout is safe to patch: writing noop into vt+0x10 for every
            // unknown slot10 corrupted heap "vtables" (slot10 values like 0, 1, or other object
            // pointers) and crashed soon after in f_70001920a0 (host AV on a garbage guest address).
            if (slot10 == 0x7002921330ull) {
                uint64_t *pfn = (uint64_t *)(uintptr_t)(cur + 64);
                if (!rt_dispatch_lookup(*pfn)) {
                    rt_log("system: %s delegate at %#llx invalid fn %#llx, replacing with noop", who,
                           (unsigned long long)cur, (unsigned long long)*pfn);
                    *pfn = noop;
                }
            } else if (!rt_dispatch_lookup(slot10)) {
                static int s_slot10_skip_logs;
                if (s_slot10_skip_logs++ < 8) {
                    rt_log("system: %s item at %#llx unknown slot10 %#llx, leaving untouched", who,
                           (unsigned long long)cur, (unsigned long long)slot10);
                }
            }
        }
        uint64_t next = *(const uint64_t *)(uintptr_t)(cur + (uint64_t)(int64_t)off + 8);
        if (!mkw_guest_ptr(next)) break;
        cur = next - (uint64_t)(int64_t)off;
    }
}

void __real_f_7000192d60(Cpu *c) __attribute__((weak));
void __wrap_f_7000192d60(Cpu *c) {
    mkw_sanitize_delegate_list(c->x[0], "f_7000192d60");
    mkw_soft_av_call(__real_f_7000192d60, c, "f_7000192d60");
}

// Sibling of f_7000192d60: same this+0x20 list, then a vtable dispatch. Playtest AV reading 0x101 in
// f_7000193200 (main+0x193200) with a null/garbage this or a corrupted +0x18 / +0x290 pointer; skip or
// clear those before the lifted body raw-loads them.
void __real_f_7000193200(Cpu *c) __attribute__((weak));
void __wrap_f_7000193200(Cpu *c) {
    uint64_t x0 = c->x[0];
    if (!mkw_guest_ptr(x0)) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_7000193200 bad this %#llx, skip", (unsigned long long)x0);
        return;
    }
    uint64_t p18 = *(const uint64_t *)(uintptr_t)(x0 + 0x18);
    if (p18 && !mkw_guest_ptr(p18)) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_7000193200 bad +0x18 %#llx, clearing and skip", (unsigned long long)p18);
        *(uint64_t *)(uintptr_t)(x0 + 0x18) = 0;
        return;
    }
    mkw_sanitize_delegate_list(x0, "f_7000193200");
    uint64_t obj22 = *(const uint64_t *)(uintptr_t)(x0 + 0x20);
    if (mkw_guest_ptr(obj22)) {
        uint64_t p290 = *(const uint64_t *)(uintptr_t)(obj22 + 0x290);
        if (p290 && !mkw_guest_ptr(p290)) {
            static int logged;
            if (logged++ < 8)
                rt_log("system: f_7000193200 bad +0x20/+0x290 %#llx, clearing", (unsigned long long)p290);
            *(uint64_t *)(uintptr_t)(obj22 + 0x290) = 0;
        }
    }
    mkw_soft_av_call(__real_f_7000193200, c, "f_7000193200");
}

// n05: read 0xb8 in f_70001974e0 under the 193200/192d60 path (null object +0xb8).
void __real_f_70001974e0(Cpu *c) __attribute__((weak));
void __wrap_f_70001974e0(Cpu *c) {
    mkw_soft_av_call(__real_f_70001974e0, c, "f_70001974e0");
}

void __real_f_7000193470(Cpu *c) __attribute__((weak));
void __wrap_f_7000193470(Cpu *c) {
    mkw_soft_av_call(__real_f_7000193470, c, "f_7000193470");
}

// Guest pc 0x700259a940 (called from the f_7000193200 vtable path): loads CriticalSection pointers from
// 0x70036e1000+{e98,bf8,c10} and __atomic_load_n-bytes them. A corrupt value like 0x101 is a host AV.
void __real_f_700259a940(Cpu *c) __attribute__((weak));
void __wrap_f_700259a940(Cpu *c) {
    static const uint32_t offs[] = {0xe98u, 0xbf8u, 0xc10u};
    const uint64_t base = 0x70036e1000ull;
    for (unsigned i = 0; i < sizeof offs / sizeof offs[0]; i++) {
        uint64_t p = *(const uint64_t *)(uintptr_t)(base + offs[i]);
        if (p && !mkw_guest_ptr(p)) {
            static int logged;
            if (logged++ < 8)
                rt_log("system: f_700259a940 bad lock +%#x = %#llx, skip", offs[i], (unsigned long long)p);
            c->x[0] = 0;
            return;
        }
    }
    mkw_soft_av_call(__real_f_700259a940, c, "f_700259a940");
}

void __real_f_700259aac0(Cpu *c) __attribute__((weak));
void __wrap_f_700259aac0(Cpu *c) {
    uint64_t p = *(const uint64_t *)(uintptr_t)(0x70036e1000ull + 0xe98ull);
    if (p && !mkw_guest_ptr(p)) {
        static int logged;
        if (logged++ < 8)
            rt_log("system: f_700259aac0 bad lock +0xe98 = %#llx, skip", (unsigned long long)p);
        c->x[0] = 0;
        return;
    }
    mkw_soft_av_call(__real_f_700259aac0, c, "f_700259aac0");
}


// MKW_HEAP_TRACE=1: logging pass-throughs on the ExpHeap vtable slots around alloc (sead Heap order): adjust (-1),
// free (+1), resizeFront (+2), resizeBack (+3). They call the game's own function and log heap, arguments and result.
static uint64_t ht_orig[4], ht_hook[4];
static const int ht_slot[4] = {-1, 1, 2, 3};
static const char *const ht_name[4] = {"adjust", "free", "resizeFront", "resizeBack"};
static void ht_call(Cpu *c, int k) {
    uint64_t heap = c->x[0], a1 = c->x[1], a2 = c->x[2], lr = c->x[30];
    if (k == 1 && slink_heap && heap == slink_heap) slink_note(heap, a1, 0, lr, 0);   // before the free rewrites the header
    GuestFn orig = rt_dispatch_lookup(ht_orig[k]);
    if (orig) orig(c);
    if (k && anim_on())
        aht_event(k == 1 ? AHT_FREE : k == 2 ? AHT_RESIZE_FRONT : AHT_RESIZE_BACK, heap, a1, a2, c->x[0], c, lr);
    if (!heap_trace()) return;
    static volatile int32_t n;
    // free is frequent: log only the large ones (a pointer into a heap header region is a heap being freed)
    int big = k != 1;
    if (big && __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED) < 400)
        rt_log("htrace %s heap %#llx a1 %#llx a2 %#llx -> %#llx (lr %#llx)", ht_name[k], (unsigned long long)heap,
               (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)c->x[0], (unsigned long long)lr);
}
// MKW_SOUND_HEAP_TRACE: the growable heap class at vtable 0x700363a7c0 (SLinkUserCreate is one): pass-throughs on its
// alloc (slot 6, f_70003f18e0) and free (slot 7, f_70001a5220) feed slink_note.
static void gh_alloc(Cpu *c) {
    uint64_t heap = c->x[0], size = c->x[1], lr = c->x[30];
    GuestFn orig = rt_dispatch_lookup(0x70003f18e0ull);
    if (orig) orig(c);
    slink_note(heap, c->x[0], size, lr, 1);
}
static void gh_free(Cpu *c) {
    uint64_t heap = c->x[0], ptr = c->x[1], lr = c->x[30];
    if (slink_heap && heap == slink_heap) slink_note(heap, ptr, 0, lr, 0);
    GuestFn orig = rt_dispatch_lookup(0x70001a5220ull);
    if (orig) orig(c);
}
static void ht_adjust(Cpu *c) { ht_call(c, 0); }
static void ht_free(Cpu *c) { ht_call(c, 1); }
static void ht_resize_front(Cpu *c) { ht_call(c, 2); }
static void ht_resize_back(Cpu *c) { ht_call(c, 3); }

void ov_system_post_load(void) {
    if (rt_application_id == 0x0400C3F00006E000ull) {
        mkw_enable_soft_skip();
        s_noop_hook_addr = hostfn_register("mkw_noop", ov_noop);
        s_expheap_hook_addr = hostfn_register("sead__ExpHeap__alloc", ov_expheap_alloc_hook);
        uint64_t *vtable1 = (uint64_t *)(uintptr_t)0x700363a330ull;
        uint64_t *vtable2 = (uint64_t *)(uintptr_t)0x70036e1e18ull;
        if (heap_trace() || anim_on() || cfg_bool("MKW_SOUND_HEAP_TRACE")) {
            static const GuestFn fn[4] = {ht_adjust, ht_free, ht_resize_front, ht_resize_back};
            for (int k = 0; k < 4; k++) {
                ht_orig[k] = vtable1[ht_slot[k]];
                uint64_t h = ht_hook[k] = hostfn_register(ht_name[k], fn[k]);
                if (vtable1[ht_slot[k]] == ht_orig[k]) vtable1[ht_slot[k]] = h;
                if (vtable2[ht_slot[k]] == ht_orig[k]) vtable2[ht_slot[k]] = h;
                rt_log("htrace hooked ExpHeap %s (%#llx)", ht_name[k], (unsigned long long)ht_orig[k]);
            }
        }
        if (cfg_bool("MKW_SOUND_HEAP_TRACE")) {
            uint64_t *gvt = (uint64_t *)(uintptr_t)0x700363a7c0ull;
            if (gvt[6] == 0x70003f18e0ull && gvt[7] == 0x70001a5220ull) {
                gvt[6] = hostfn_register("growheap_alloc", gh_alloc);
                gvt[7] = hostfn_register("growheap_free", gh_free);
                rt_log("dslink: hooked alloc/free of heap class 0x700363a7c0");
            } else
                rt_log("dslink: heap class 0x700363a7c0 slots are %#llx %#llx, not hooked", (unsigned long long)gvt[6],
                       (unsigned long long)gvt[7]);
        }
        rt_log("heap: patching ExpHeap::alloc vtable at %p (%#llx -> %#llx)",
               vtable1, (unsigned long long)*vtable1, (unsigned long long)s_expheap_hook_addr);
        *vtable1 = s_expheap_hook_addr;
        rt_log("heap: patching ExpHeap derived vtable at %p (%#llx -> %#llx)",
               vtable2, (unsigned long long)*vtable2, (unsigned long long)s_expheap_hook_addr);
        *vtable2 = s_expheap_hook_addr;

        s_obj_dispose_hook_addr = hostfn_register("mkw_object_dispose", __wrap_f_700087b9b0);
        uint64_t *vtable_disp = (uint64_t *)(uintptr_t)0x70036b6a18ull;
        rt_log("heap: patching dispose vtable at %p (%#llx -> %#llx)",
               vtable_disp, (unsigned long long)*vtable_disp, (unsigned long long)s_obj_dispose_hook_addr);
        *vtable_disp = s_obj_dispose_hook_addr;
    }
}

void ov_system_on_inits_done(void) {
    if (rt_application_id == 0x0400C3F00006E000ull) {
        if (s_expheap_hook_addr) {
            *(uint64_t *)(uintptr_t)0x700363a330ull = s_expheap_hook_addr;
            *(uint64_t *)(uintptr_t)0x70036e1e18ull = s_expheap_hook_addr;
        }
        if (s_obj_dispose_hook_addr) {
            *(uint64_t *)(uintptr_t)0x70036b6a18ull = s_obj_dispose_hook_addr;
        }
        // the heap-trace pass-throughs again, in case the tables were rewritten since post_load
        for (int k = 0; k < 4; k++) {
            uint64_t *vt[2] = {(uint64_t *)(uintptr_t)0x700363a330ull, (uint64_t *)(uintptr_t)0x70036e1e18ull};
            for (int v = 0; v < 2 && ht_hook[k]; v++)
                if (vt[v][ht_slot[k]] == ht_orig[k]) {
                    vt[v][ht_slot[k]] = ht_hook[k];
                    rt_log("htrace re-hooked ExpHeap %s in table %d", ht_name[k], v);
                }
        }
        *(uint32_t *)(uintptr_t)0x70052db828ull = 1;
        rt_log("gs: set 0x70052db828 = 1 in ov_system_on_inits_done");
    }
}

// ---- MKW_SKIN_WATCH=1 (diagnostic, Crown City): where the broken bone matrices of the skinned objects come from.
// vk_exec_draw.c skin_check reports each skinned draw's bone buffer (rt_skin_watch_seen). The buffers of one kind (same
// size and bones per object) rotate through a ring; once a kind has shown a broken matrix and SKW_RING of its buffers
// have been seen, they are made read-only. Each guest write there faults; the handler notes the lifted function, the
// host store site and the object slot, lets that one instruction run with the page writable (trap flag), checks the
// value it wrote, and protects the page again.
// Stage 1 (SKW_FRAMES frames): writers of the bone buffers. The writer is f_70001673c0's copy loop, which copies each
// dirty bone's 48-byte matrix from the model's bone array (0xa0-byte entries, matrix at +0x30) into the buffer; at its
// first store (f_70001673c0 + SKW_COPY_OFF) the source entry is rax + rbp. Bone entries that were copied broken are
// remembered (up to SKW_ENT).
// Stage 2 (SKW_FRAMES2 frames): the bone buffers stay watched, and the pages of those entries are watched too. Per
// entry: who wrote it (function, host site, broken or not), and each copy out of it: broken or not, and whether it was
// written since the previous copy (a broken copy with no write before it = the entry was never computed).
#ifdef _WIN32
enum { SKW_FRAMES = 8, SKW_FRAMES2 = 16, SKW_WRITERS = 64, SKW_BUFS = 8, SKW_RING = 4, SKW_SLOTS = 256, SKW_SITES = 64,
       SKW_ENT = 16, SKW_ENT_SIZE = 0xa0, SKW_ENT_W = 8 };
#define SKW_COPY_OFF 0x19ea1ull   // f_70001673c0's first store into the bone buffer (main_0020.o as built 2026-11-12)
uint64_t rt_lifted_function(uintptr_t rip);   // main.c
void f_70001673c0(void *c);
typedef struct {
    uint64_t addr, size;
    uint32_t bones;
    uint8_t written[SKW_SLOTS], bad[SKW_SLOTS];
} SkwBuf;
typedef struct {
    uint64_t addr;
    uint64_t writes, bad_writes, copies, bad_copies, bad_copies_unwritten, good_copies;
    int dirty;   // written since the last copy
    struct { uint64_t fn, site, n, bad; } w[SKW_ENT_W];
    int nw;
    uint32_t first[SKW_ENT_SIZE / 4];   // contents when first seen broken
} SkwEnt;
static SkwBuf skw_b[SKW_BUFS];
static int skw_nb;
static SkwEnt skw_e[SKW_ENT];
static int skw_ne;
static uint64_t skw_kind_size;   // the kind being watched (buffer size); 0 until a broken one is seen
static uint32_t skw_kind_bones;
static volatile LONG skw_armed, skw_done, skw_stage;
static uint64_t skw_until;
static struct { uint64_t fn, writes, bad; } skw_w[SKW_WRITERS];
static volatile LONG skw_nw;
static struct { uint64_t rip, writes, bad; } skw_s[SKW_SITES];   // host store sites (offset from the image base)
static int skw_ns;
static SRWLOCK skw_lock = SRWLOCK_INIT;
static _Thread_local uint64_t skw_page, skw_addr, skw_fn, skw_rip, skw_src;
static PVOID skw_veh;

static int skw_badf(uint64_t a) {
    float f;
    memcpy(&f, (const void *)(uintptr_t)(a & ~3ull), 4);
    return !(f == f) || f > 1e6f || f < -1e6f;
}

static int skw_bad_mat(uint64_t a) {   // a 48-byte matrix (3 rows of 4): huge/NaN values, or a zero row
    const float *m = (const float *)(uintptr_t)a;
    for (int i = 0; i < 12; i++)
        if (!(m[i] == m[i]) || m[i] > 1e6f || m[i] < -1e6f) return 1;
    for (int r = 0; r < 3; r++)
        if (m[r * 4] == 0 && m[r * 4 + 1] == 0 && m[r * 4 + 2] == 0) return 1;
    return 0;
}

static SkwBuf *skw_find(uint64_t a) {
    for (int i = 0; i < skw_nb; i++)
        if (a >= skw_b[i].addr && a < skw_b[i].addr + skw_b[i].size) return &skw_b[i];
    return NULL;
}

static SkwEnt *skw_find_ent(uint64_t a) {
    for (int i = 0; i < skw_ne; i++)
        if (a >= skw_e[i].addr && a < skw_e[i].addr + SKW_ENT_SIZE) return &skw_e[i];
    return NULL;
}

static int skw_in_pages(uint64_t a) {
    uint64_t p = a & ~0xfffull;
    for (int i = 0; i < skw_nb; i++)
        if (p >= (skw_b[i].addr & ~0xfffull) && p < ((skw_b[i].addr + skw_b[i].size + 0xfff) & ~0xfffull)) return 1;
    if (skw_stage == 2)
        for (int i = 0; i < skw_ne; i++)
            if (p >= (skw_e[i].addr & ~0xfffull) && p <= ((skw_e[i].addr + SKW_ENT_SIZE - 1) & ~0xfffull)) return 1;
    return 0;
}

static void skw_note(uint64_t fn, uint64_t a, int bad, uint64_t src) {
    AcquireSRWLockExclusive(&skw_lock);
    uint64_t r = skw_rip - (uint64_t)(uintptr_t)GetModuleHandleW(NULL);
    SkwBuf *b = skw_find(a);
    if (b) {
        int k;
        for (k = 0; k < skw_ns && skw_s[k].rip != r; k++) {}
        if (k == skw_ns && skw_ns < SKW_SITES) skw_s[skw_ns++] = (typeof(skw_s[0])){r, 0, 0};
        if (k < skw_ns) skw_s[k].writes++, skw_s[k].bad += bad;
        int i;
        for (i = 0; i < skw_nw && skw_w[i].fn != fn; i++) {}
        if (i == skw_nw && skw_nw < SKW_WRITERS) skw_w[skw_nw++] = (typeof(skw_w[0])){fn, 0, 0};
        if (i < skw_nw) skw_w[i].writes++, skw_w[i].bad += bad;
        if (a >= b->addr + 16 && b->bones) {
            uint64_t s = (a - b->addr - 16) / (48ull * b->bones);
            if (s < SKW_SLOTS) b->written[s] = 1;
        }
        if (src) {   // the copy loop's first store: one bone matrix copied out of entry src
            int badm = plat_mem_readable(src + 0x30, 48) && skw_bad_mat(src + 0x30);
            SkwEnt *e = skw_find_ent(src);
            if (!e && badm && skw_stage == 1 && skw_ne < SKW_ENT && plat_mem_readable(src, SKW_ENT_SIZE)) {
                e = &skw_e[skw_ne++];
                memset(e, 0, sizeof *e);
                e->addr = src;
                memcpy(e->first, (const void *)(uintptr_t)src, SKW_ENT_SIZE);
            }
            if (e && skw_stage == 2) {
                e->copies++;
                if (badm) e->bad_copies++, e->bad_copies_unwritten += !e->dirty;
                else e->good_copies++;
                e->dirty = 0;
            }
        }
    } else if (skw_stage == 2) {
        SkwEnt *e = skw_find_ent(a);
        if (e) {
            e->writes++, e->bad_writes += bad, e->dirty = 1;
            int k;
            for (k = 0; k < e->nw && (e->w[k].fn != fn || e->w[k].site != r); k++) {}
            if (k == e->nw && e->nw < SKW_ENT_W) e->w[e->nw++] = (typeof(e->w[0])){fn, r, 0, 0};
            if (k < e->nw) e->w[k].n++, e->w[k].bad += bad;
        }
    }
    ReleaseSRWLockExclusive(&skw_lock);
}

static void skw_protect(DWORD prot) {
    DWORD old;
    for (int i = 0; i < skw_nb; i++) {
        uint64_t lo = skw_b[i].addr & ~0xfffull, hi = (skw_b[i].addr + skw_b[i].size + 0xfff) & ~0xfffull;
        VirtualProtect((void *)(uintptr_t)lo, hi - lo, prot, &old);
    }
    if (skw_stage == 2)
        for (int i = 0; i < skw_ne; i++) {
            uint64_t lo = skw_e[i].addr & ~0xfffull, hi = (skw_e[i].addr + SKW_ENT_SIZE + 0xfff) & ~0xfffull;
            VirtualProtect((void *)(uintptr_t)lo, hi - lo, prot, &old);
        }
}

static LONG CALLBACK skw_veh_fn(PEXCEPTION_POINTERS ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION && skw_armed && ep->ExceptionRecord->NumberParameters >= 2 &&
        ep->ExceptionRecord->ExceptionInformation[0] == 1) {
        uint64_t a = ep->ExceptionRecord->ExceptionInformation[1];
        if (skw_in_pages(a)) {
            DWORD old;
            PCONTEXT x = ep->ContextRecord;
            skw_page = a & ~0xfffull, skw_addr = a;
            skw_fn = rt_lifted_function((uintptr_t)x->Rip);
            skw_rip = x->Rip;
            skw_src = x->Rip == (uint64_t)(uintptr_t)f_70001673c0 + SKW_COPY_OFF ? x->Rax + x->Rbp : 0;
            VirtualProtect((void *)(uintptr_t)skw_page, 0x1000, PAGE_READWRITE, &old);
            x->EFlags |= 0x100;   // trap after this one instruction
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (code == EXCEPTION_SINGLE_STEP && skw_page) {
        DWORD old;
        skw_note(skw_fn, skw_addr, skw_badf(skw_addr), skw_src);
        if (skw_armed) VirtualProtect((void *)(uintptr_t)skw_page, 0x1000, PAGE_READONLY, &old);
        skw_page = 0;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// Each skinned draw: its bone buffer, the object's slot, and whether that object's matrices are broken.
void rt_skin_watch_seen(uint64_t addr, uint64_t size, uint32_t bones, uint32_t slot, int bad, uint64_t frame) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_SKIN_WATCH");
    if (!on || skw_done || skw_armed || !bones) return;
    if (!skw_kind_size) {
        if (!bad) return;
        skw_kind_size = size, skw_kind_bones = bones;
        rt_log("dskinw: broken matrices in a %u-bone buffer of %#llx bytes; collecting its ring", bones,
               (unsigned long long)size);
    }
    if (size != skw_kind_size || bones != skw_kind_bones) return;
    AcquireSRWLockExclusive(&skw_lock);
    SkwBuf *b = skw_find(addr);
    if (!b && skw_nb < SKW_BUFS) b = &skw_b[skw_nb++], memset(b, 0, sizeof *b), b->addr = addr, b->size = size, b->bones = bones;
    if (b && bad && slot < SKW_SLOTS) b->bad[slot] = 1;
    int ready = skw_nb >= SKW_RING;
    ReleaseSRWLockExclusive(&skw_lock);
    if (!ready || InterlockedCompareExchange(&skw_armed, 1, 0)) return;
    if (!skw_veh) skw_veh = AddVectoredExceptionHandler(1, skw_veh_fn);
    skw_stage = 1;
    skw_until = frame + SKW_FRAMES;
    skw_protect(PAGE_READONLY);
    rt_log("dskinw: watching %d buffers for %d frames", skw_nb, SKW_FRAMES);
}

static void skw_report_stage1(void) {
    for (int i = 0; i < skw_nw; i++)
        rt_log("dskinw: writer f_%llx: %llu writes, %llu left a broken float", (unsigned long long)skw_w[i].fn,
               (unsigned long long)skw_w[i].writes, (unsigned long long)skw_w[i].bad);
    for (int i = 0; i < skw_ns; i++)
        rt_log("dskinw: store site image+%#llx: %llu writes, %llu broken", (unsigned long long)skw_s[i].rip,
               (unsigned long long)skw_s[i].writes, (unsigned long long)skw_s[i].bad);
    for (int i = 0; i < skw_nb; i++) {
        char w[600], bd[600];
        int ow = 0, ob = 0;
        uint64_t nslot = (skw_b[i].size - 16) / (48ull * skw_b[i].bones);
        for (uint64_t s = 0; s < nslot && s < SKW_SLOTS; s++) {
            if (skw_b[i].written[s] && ow < (int)sizeof w - 8) ow += snprintf(w + ow, sizeof w - ow, " %llu", (unsigned long long)s);
            if (skw_b[i].bad[s] && ob < (int)sizeof bd - 8) ob += snprintf(bd + ob, sizeof bd - ob, " %llu", (unsigned long long)s);
        }
        rt_log("dskinw: buffer %#llx (%llu slots): written slots:%s | broken slots seen:%s", (unsigned long long)skw_b[i].addr,
               (unsigned long long)nslot, ow ? w : " none", ob ? bd : " none");
    }
    for (int i = 0; i < skw_ne; i++) {
        char t[1400];
        int o = 0;
        for (int k = 0; k < SKW_ENT_SIZE / 4; k++) {
            float f;
            memcpy(&f, &skw_e[i].first[k], 4);
            o += snprintf(t + o, sizeof t - o, "%s%g", k % 4 ? " " : k ? " | " : "", f);
        }
        rt_log("dskinw: broken bone entry %#llx (copied broken): %s", (unsigned long long)skw_e[i].addr, t);
    }
}

static void skw_report_stage2(void) {
    for (int i = 0; i < skw_ne; i++) {
        SkwEnt *e = &skw_e[i];
        rt_log("dskinw2: entry %#llx: %llu writes (%llu broken), %llu copies out: %llu good, %llu broken "
               "(%llu of those with no write since the previous copy); now %s",
               (unsigned long long)e->addr, (unsigned long long)e->writes, (unsigned long long)e->bad_writes,
               (unsigned long long)e->copies, (unsigned long long)e->good_copies, (unsigned long long)e->bad_copies,
               (unsigned long long)e->bad_copies_unwritten, skw_bad_mat(e->addr + 0x30) ? "broken" : "good");
        for (int k = 0; k < e->nw; k++)
            rt_log("dskinw2:   writer f_%llx at image+%#llx: %llu writes, %llu broken", (unsigned long long)e->w[k].fn,
                   (unsigned long long)e->w[k].site, (unsigned long long)e->w[k].n, (unsigned long long)e->w[k].bad);
    }
}

// Each skinned draw: move to stage 2, or disarm and report.
void rt_skin_watch_poll(uint64_t frame) {
    if (!skw_armed || skw_done || frame < skw_until) return;
    if (skw_stage == 1) {
        AcquireSRWLockExclusive(&skw_lock);
        skw_report_stage1();
        ReleaseSRWLockExclusive(&skw_lock);
        if (skw_ne) {
            skw_stage = 2;   // the bone buffers stay protected; add the entries' pages
            skw_protect(PAGE_READONLY);
            skw_until = frame + SKW_FRAMES2;
            rt_log("dskinw: stage 2: watching %d broken bone entries for %d frames", skw_ne, SKW_FRAMES2);
            return;
        }
        rt_log("dskinw: no broken bone entry was caught at the copy; no stage 2");
    }
    skw_armed = 0, skw_done = 1;
    skw_protect(PAGE_READWRITE);
    AcquireSRWLockExclusive(&skw_lock);
    if (skw_stage == 2) skw_report_stage2();
    rt_log("dskinw: done");
    ReleaseSRWLockExclusive(&skw_lock);
}
void rt_skin_watch_arm(uint64_t addr, uint64_t size, uint64_t frame) { (void)addr, (void)size, (void)frame; }
#else
void rt_skin_watch_seen(uint64_t addr, uint64_t size, uint32_t bones, uint32_t slot, int bad, uint64_t frame) {
    (void)addr, (void)size, (void)bones, (void)slot, (void)bad, (void)frame;
}
void rt_skin_watch_poll(uint64_t frame) { (void)frame; }
void rt_skin_watch_arm(uint64_t addr, uint64_t size, uint64_t frame) { (void)addr, (void)size, (void)frame; }
#endif

// ---- MKW_TRIG_CHECK=1 (diagnostic, Crown City bone matrices): every guest call of sinf/cosf through a PLT runs the
// SDK's lifted routine, then compares its result with the host's (double precision). Logs the first mismatches
// ("dtrig": input, guest and host results, caller) and a count every 2^20 calls.
static void trig_check(Cpu *c, int cosine) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_TRIG_CHECK");
    uint32_t in_bits = (uint32_t)c->v[0];
    uint64_t lr = c->x[30];
    rt_original(c);
    if (!on) return;
    float x, got;
    uint32_t got_bits = (uint32_t)c->v[0];
    memcpy(&x, &in_bits, 4), memcpy(&got, &got_bits, 4);
    double want = cosine ? cos((double)x) : sin((double)x);
    static long calls, bad;
    long n = __atomic_add_fetch(&calls, 1, __ATOMIC_RELAXED);
    int wrong = (x == x && fabs(x) < 1e30f) ? !(fabs((double)got - want) <= 1e-5) : 0;
    if (wrong) {
        long b = __atomic_add_fetch(&bad, 1, __ATOMIC_RELAXED);
        if (b <= 40)
            rt_log("dtrig %s(%.9g [%08x]) = %.9g [%08x], host %.9g, caller %#llx", cosine ? "cosf" : "sinf", x, in_bits,
                   got, got_bits, want, (unsigned long long)lr);
    }
    if (!(n & 0xfffff)) rt_log("dtrig: %ld calls, %ld wrong", n, __atomic_load_n(&bad, __ATOMIC_RELAXED));
}
static void obs_sinf(Cpu *c) { trig_check(c, 0); }
static void obs_cosf(Cpu *c) { trig_check(c, 1); }
HOST_OBSERVER("sinf", obs_sinf);
HOST_OBSERVER("cosf", obs_cosf);

// ---- MKW_ANIM_CHECK=1 (diagnostic, Crown City bone matrices): probes inserted by hand into the generated
// f_700042bea0 (main_0049.c; original at the scratchpad copy main_0049.c.orig). rt_anim_t sees each curve sample time
// (base + offset, the two floats summed before the frame index is taken) and the curve header. rt_anim_q checks
// the quaternion matrix store join, not the separate Euler paths. Blends reach that join without rt_anim_t:
// their TLS time/header and scratch quaternion can be stale, so report only the matrix for those paths.
static _Thread_local float anim_base, anim_off;
static _Thread_local uint64_t anim_hdr;
static _Thread_local uint64_t anim_entry;
static _Thread_local int anim_path;
static struct {
    uint64_t hdr;
    float good_max, good_min;
    uint64_t good, bad, calls;
    int dumped, ndump, nsample, ndiff;
    uint64_t base, size, res;   // the keyframe buffer, resolved at first sight
    uint8_t *copy;              // its contents at first sight
    uint64_t diff_sig;          // hash of the live contents at the last reported difference
} anim_h[128];

// The keyframe buffer of a curve header: base = [[[hdr+0x40]+0x10]+0x100] + [hdr+0x28], size = [hdr+0x98]
// (= frames (u16 at hdr+0x8a) x stride (u32 at hdr+0x10c) in every header seen so far).
static int anim_buf(uint64_t hdr, uint64_t *base, uint64_t *size, uint64_t *res) {
    uint64_t a, b, x11, off;
    uint32_t sz, stride;
    uint16_t frames;
    if (!plat_mem_readable(hdr, 0x130)) return 0;
    memcpy(&a, (const void *)(uintptr_t)(hdr + 0x40), 8);
    memcpy(&off, (const void *)(uintptr_t)(hdr + 0x28), 8);
    memcpy(&sz, (const void *)(uintptr_t)(hdr + 0x98), 4);
    memcpy(&stride, (const void *)(uintptr_t)(hdr + 0x10c), 4);
    memcpy(&frames, (const void *)(uintptr_t)(hdr + 0x8a), 2);
    if (!sz) sz = (uint32_t)frames * stride;
    if (!a || !plat_mem_readable(a + 0x10, 8)) return 0;
    memcpy(&b, (const void *)(uintptr_t)(a + 0x10), 8);
    if (!b || !plat_mem_readable(b, 0x108)) return 0;
    memcpy(&x11, (const void *)(uintptr_t)(b + 0x100), 8);
    if (!x11 || !sz || sz > (64u << 20) || !plat_mem_readable(x11 + off, sz)) return 0;
    *base = x11 + off, *size = sz, *res = b;
    return 1;
}
static uint64_t anim_hash(const uint8_t *p, uint64_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x100000001b3ull;
    }
    return h;
}

// ---- heap tracker for MKW_ANIM_CHECK. aht_event sees every sead ExpHeap alloc (the __wrap_f_70003f24c0 hook) and
// free / resizeFront / resizeBack (the ht_* vtable pass-throughs). Events go into a ring; allocations of at least
// AHT_BIG_MIN bytes are also kept in a table until freed. aht_watch registers a keyframe buffer: it reports the big
// allocations overlapping it and the ring's past events touching it, and from then on every event touching it is
// logged as it happens ("daht"), with the guest return addresses up the frame-pointer chain.
typedef struct {
    uint64_t ptr, size, heap, res, old, lr[4];
    uint32_t seq, kind;
} AhtEv;
enum { AHT_RING = 1 << 19, AHT_BIG = 1 << 17, AHT_RANGES = 16, AHT_BIG_MIN = 0x8000, AHT_LIVE_LOGS = 300, AHT_PAST_LOGS = 60 };
static AhtEv aht_ring[AHT_RING];
static uint32_t aht_seq;
static struct { uint64_t ptr, size, lr[4]; uint32_t seq; } aht_big[AHT_BIG];   // open addressing: ptr 0 empty, 1 deleted
static SRWLOCK aht_lock = SRWLOCK_INIT;
static struct { uint64_t lo, hi, hdr; } aht_rng[AHT_RANGES];
static volatile LONG aht_nrng;
static long aht_live_logs, aht_count[4];
static const char *const aht_kind[4] = {"alloc", "free", "resizeFront", "resizeBack"};

static uint32_t aht_slot(uint64_t ptr) { return (uint32_t)(((ptr >> 4) * 0x9E3779B97F4A7C15ull) >> 47) & (AHT_BIG - 1); }
static int aht_big_find(uint64_t ptr) {   // index or -1; the caller holds aht_lock
    for (uint32_t i = aht_slot(ptr), n = 0; n < AHT_BIG; i = (i + 1) & (AHT_BIG - 1), n++) {
        if (aht_big[i].ptr == ptr) return (int)i;
        if (!aht_big[i].ptr) return -1;
    }
    return -1;
}
static void aht_big_put(const AhtEv *e) {   // the caller holds aht_lock exclusively
    int at = -1;
    for (uint32_t i = aht_slot(e->ptr), n = 0; n < AHT_BIG; i = (i + 1) & (AHT_BIG - 1), n++) {
        if (aht_big[i].ptr == e->ptr) { at = (int)i; break; }
        if (aht_big[i].ptr == 1 && at < 0) at = (int)i;
        if (!aht_big[i].ptr) {
            if (at < 0) at = (int)i;
            break;
        }
    }
    if (at < 0) return;
    aht_big[at].ptr = e->ptr, aht_big[at].size = e->size, aht_big[at].seq = e->seq;
    memcpy(aht_big[at].lr, e->lr, sizeof e->lr);
}
static int aht_hit_range(const AhtEv *e, int r) {
    uint64_t lo = aht_rng[r].lo, hi = aht_rng[r].hi;
    if (e->kind == AHT_ALLOC || e->kind == AHT_FREE) return e->ptr < hi && e->ptr + (e->size ? e->size : 1) > lo;
    // a resize: the block starts at ptr; its old extent is known only for big blocks
    return e->ptr < hi && e->ptr + (e->old ? e->old : (16ull << 20)) > lo;
}
static void aht_print(const AhtEv *e, const char *when, int r) {
    uint64_t lo = aht_rng[r].lo;
    rt_log("daht %s #%u %s heap %#llx ptr %#llx size %#llx (old %#llx) -> %#llx | callers %#llx %#llx %#llx %#llx | "
           "keyframe buffer of hdr %#llx [%#llx, %#llx): event at %s%#llx",
           when, e->seq, aht_kind[e->kind & 3], (unsigned long long)e->heap, (unsigned long long)e->ptr,
           (unsigned long long)e->size, (unsigned long long)e->old, (unsigned long long)e->res, (unsigned long long)e->lr[0],
           (unsigned long long)e->lr[1], (unsigned long long)e->lr[2], (unsigned long long)e->lr[3],
           (unsigned long long)aht_rng[r].hdr, (unsigned long long)lo, (unsigned long long)aht_rng[r].hi,
           e->ptr >= lo ? "+" : "-", (unsigned long long)(e->ptr >= lo ? e->ptr - lo : lo - e->ptr));
}
static void aht_event(int kind, uint64_t heap, uint64_t ptr, uint64_t size, uint64_t res, Cpu *c, uint64_t lr) {
    AhtEv e = {.ptr = ptr, .size = size, .heap = heap, .res = res, .kind = (uint32_t)kind};
    e.seq = __atomic_add_fetch(&aht_seq, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&aht_count[kind & 3], 1, __ATOMIC_RELAXED);
    e.lr[0] = lr;
    // the guest frame-pointer chain, kept to frames above sp on this stack
    uint64_t sp = c->sp, fp = c->x[29];
    for (int i = 1; i < 4 && fp >= sp && fp < sp + 0x80000 && !(fp & 7); i++) {
        e.lr[i] = *(const uint64_t *)(uintptr_t)(fp + 8);
        uint64_t up = *(const uint64_t *)(uintptr_t)fp;
        if (up <= fp) break;
        fp = up;
    }
    if (kind == AHT_ALLOC) {
        if (size >= AHT_BIG_MIN) {
            AcquireSRWLockExclusive(&aht_lock);
            aht_big_put(&e);
            ReleaseSRWLockExclusive(&aht_lock);
        }
    } else {
        AcquireSRWLockShared(&aht_lock);
        int i = aht_big_find(ptr);
        ReleaseSRWLockShared(&aht_lock);
        if (i >= 0) {
            AcquireSRWLockExclusive(&aht_lock);
            i = aht_big_find(ptr);
            if (i >= 0) {
                e.old = aht_big[i].size;
                if (kind == AHT_FREE) e.size = aht_big[i].size, aht_big[i].ptr = 1;
                else if (kind == AHT_RESIZE_BACK && res) aht_big[i].size = size;
            }
            ReleaseSRWLockExclusive(&aht_lock);
        }
    }
    aht_ring[e.seq & (AHT_RING - 1)] = e;
    int n = aht_nrng;
    for (int r = 0; r < n; r++)
        if (aht_hit_range(&e, r)) {
            if (__atomic_add_fetch(&aht_live_logs, 1, __ATOMIC_RELAXED) <= AHT_LIVE_LOGS) aht_print(&e, "live", r);
            break;
        }
}
// Registers a keyframe buffer: what overlaps it now, what touched it before, and from here on live reports.
static void aht_watch(uint64_t hdr, uint64_t lo, uint64_t hi) {
    AcquireSRWLockExclusive(&aht_lock);
    int r = aht_nrng;
    if (r >= AHT_RANGES) {
        ReleaseSRWLockExclusive(&aht_lock);
        return;
    }
    aht_rng[r].lo = lo, aht_rng[r].hi = hi, aht_rng[r].hdr = hdr;
    int found = 0;
    for (uint32_t i = 0; i < AHT_BIG; i++)
        if (aht_big[i].ptr > 1 && aht_big[i].ptr < hi && aht_big[i].ptr + aht_big[i].size > lo) {
            found++;
            rt_log("daht hdr %#llx buffer [%#llx, %#llx): overlaps live big allocation #%u ptr %#llx size %#llx (ends %#llx; "
                   "buffer starts at %s%#llx of it) callers %#llx %#llx %#llx %#llx",
                   (unsigned long long)hdr, (unsigned long long)lo, (unsigned long long)hi, aht_big[i].seq,
                   (unsigned long long)aht_big[i].ptr, (unsigned long long)aht_big[i].size,
                   (unsigned long long)(aht_big[i].ptr + aht_big[i].size), lo >= aht_big[i].ptr ? "+" : "-",
                   (unsigned long long)(lo >= aht_big[i].ptr ? lo - aht_big[i].ptr : aht_big[i].ptr - lo),
                   (unsigned long long)aht_big[i].lr[0], (unsigned long long)aht_big[i].lr[1],
                   (unsigned long long)aht_big[i].lr[2], (unsigned long long)aht_big[i].lr[3]);
        }
    if (!found)
        rt_log("daht hdr %#llx buffer [%#llx, %#llx): no live tracked ExpHeap allocation of %#x bytes or more overlaps it",
               (unsigned long long)hdr, (unsigned long long)lo, (unsigned long long)hi, AHT_BIG_MIN);
    uint32_t now = __atomic_load_n(&aht_seq, __ATOMIC_RELAXED), span = now < AHT_RING ? now : AHT_RING - 1;
    int hits = 0;
    for (uint32_t s = now - span + 1; s != now + 1; s++) {
        AhtEv e = aht_ring[s & (AHT_RING - 1)];
        if (e.seq != s || !aht_hit_range(&e, r)) continue;
        if (hits++ < AHT_PAST_LOGS) aht_print(&e, "past", r);
    }
    rt_log("daht hdr %#llx: %d past heap events touch its buffer (ring holds the last %u of %u events; so far %ld allocs, "
           "%ld frees, %ld resizeFront, %ld resizeBack)", (unsigned long long)hdr, hits, span, now, aht_count[0], aht_count[1],
           aht_count[2], aht_count[3]);
    aht_nrng = r + 1;
    ReleaseSRWLockExclusive(&aht_lock);
}

static void apl_report(uint64_t hdr, uint64_t pool, uint64_t lo, uint64_t hi);   // below
static void apw_report(uint64_t hdr, uint64_t pool);
static void apw_add(uint64_t pool, uint64_t mem, uint64_t size);
static void apw_retire(uint64_t pool);

// First sample from a curve header: resolve its keyframe buffer, keep a copy, and have the heap tracker watch it.
static void anim_first_sight(int h) {
    uint64_t hdr = anim_h[h].hdr, base, size, res;
    if (!anim_buf(hdr, &base, &size, &res)) {
        rt_log("danim: hdr %#llx: keyframe buffer not resolved", (unsigned long long)hdr);
        return;
    }
    anim_h[h].base = base, anim_h[h].size = size, anim_h[h].res = res;
    anim_h[h].copy = malloc(size);
    if (anim_h[h].copy) memcpy(anim_h[h].copy, (const void *)(uintptr_t)base, size);
    uint32_t stride;
    uint16_t frames, bones;
    uint8_t cs;
    memcpy(&stride, (const void *)(uintptr_t)(hdr + 0x10c), 4);
    memcpy(&frames, (const void *)(uintptr_t)(hdr + 0x8a), 2);
    memcpy(&bones, (const void *)(uintptr_t)(hdr + 0x88), 2);
    memcpy(&cs, (const void *)(uintptr_t)(hdr + 0x129), 1);
    rt_log("danim first sight: hdr %#llx keyframe buffer [%#llx, %#llx) size %#llx (%u frames x stride %u = %#x; word +0x88 %u; "
           "%u bytes per bone) resource %#llx, at heap event #%u",
           (unsigned long long)hdr, (unsigned long long)base, (unsigned long long)(base + size), (unsigned long long)size,
           frames, stride, frames * stride, bones, cs, (unsigned long long)res, __atomic_load_n(&aht_seq, __ATOMIC_RELAXED));
    aht_watch(hdr, base, base + size);
    apl_report(hdr, res, base, base + size);
    apw_report(hdr, res);
}
static void anim_write(const char *name, const void *p, uint64_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) return;
    fwrite(p, 1, n, f);
    fclose(f);
}
// Every 512 samples of a header: has its keyframe buffer changed since first sight? Reports the changed byte ranges
// once per distinct state (up to 12 states).
static void anim_diff(int h) {
    if (!anim_h[h].copy || anim_h[h].ndiff >= 12) return;
    const uint8_t *live = (const uint8_t *)(uintptr_t)anim_h[h].base, *copy = anim_h[h].copy;
    uint64_t size = anim_h[h].size;
    if (!memcmp(live, copy, size)) return;
    uint64_t sig = anim_hash(live, size);
    if (sig == anim_h[h].diff_sig) return;
    anim_h[h].diff_sig = sig;
    char s[1600];
    int o = 0, nr = 0;
    uint64_t changed = 0, first = 0, last_end = 0;
    for (uint64_t i = 0; i < size;) {
        if (live[i] == copy[i]) {
            i++;
            continue;
        }
        uint64_t j = i, last = i;
        for (; j < size && j - last < 64; j++)
            if (live[j] != copy[j]) last = j;
        uint64_t len = last - i + 1;
        if (!nr) first = i;
        last_end = last + 1;
        changed += len;
        if (nr < 10 && o < (int)sizeof s - 120) {
            o += snprintf(s + o, sizeof s - o, " | +%#llx len %#llx:", (unsigned long long)i, (unsigned long long)len);
            for (uint64_t k = i; k < i + 8 && k < size; k++) o += snprintf(s + o, sizeof s - o, " %02x", copy[k]);
            o += snprintf(s + o, sizeof s - o, " ->");
            for (uint64_t k = i; k < i + 8 && k < size; k++) o += snprintf(s + o, sizeof s - o, " %02x", live[k]);
        }
        nr++;
        i = last + 1;
    }
    rt_log("danim buffer CHANGED since first sight: hdr %#llx base %#llx, state %d at heap event #%u: %d ranges, %llu bytes, "
           "spanning +%#llx..+%#llx of %#llx%s",
           (unsigned long long)anim_h[h].hdr, (unsigned long long)anim_h[h].base, anim_h[h].ndiff,
           __atomic_load_n(&aht_seq, __ATOMIC_RELAXED), nr, (unsigned long long)changed, (unsigned long long)first,
           (unsigned long long)last_end, (unsigned long long)size, s);
    if (anim_h[h].ndiff < 3) {
        char name[160];
        CreateDirectoryA("anim-dump", NULL);
        snprintf(name, sizeof name, "anim-dump/h%llx_changed%d.bin", (unsigned long long)anim_h[h].hdr, anim_h[h].ndiff);
        anim_write(name, live, size);
    }
    anim_h[h].ndiff++;
}
// First bad rotation of a header: the live keyframe buffer with up to 0x2000 bytes either side, the first-sight
// copy, the resource object and the header, in anim-dump/.
static void anim_dump(int h, const char *why) {
    anim_h[h].ndump++;
    if (!anim_h[h].copy) return;
    uint64_t hdr = anim_h[h].hdr, base = anim_h[h].base, size = anim_h[h].size, lo = base, hi = base + size;
    if (plat_mem_readable(base - 0x2000, 0x2000)) lo = base - 0x2000;
    if (plat_mem_readable(hi, 0x2000)) hi += 0x2000;
    char name[160];
    CreateDirectoryA("anim-dump", NULL);
    snprintf(name, sizeof name, "anim-dump/h%llx_live_from%llx.bin", (unsigned long long)hdr, (unsigned long long)lo);
    anim_write(name, (const void *)(uintptr_t)lo, hi - lo);
    snprintf(name, sizeof name, "anim-dump/h%llx_first_from%llx.bin", (unsigned long long)hdr, (unsigned long long)base);
    anim_write(name, anim_h[h].copy, size);
    if (plat_mem_readable(anim_h[h].res, 0x400)) {
        snprintf(name, sizeof name, "anim-dump/h%llx_res%llx.bin", (unsigned long long)hdr, (unsigned long long)anim_h[h].res);
        anim_write(name, (const void *)(uintptr_t)anim_h[h].res, 0x400);
    }
    if (plat_mem_readable(hdr - 0x100, 0x500)) {
        snprintf(name, sizeof name, "anim-dump/h%llx_hdr_from%llx.bin", (unsigned long long)hdr, (unsigned long long)(hdr - 0x100));
        anim_write(name, (const void *)(uintptr_t)(hdr - 0x100), 0x500);
    }
    rt_log("danim dump (%s): hdr %#llx buffer [%#llx, %#llx) written from %#llx to %#llx; live %s the first-sight copy",
           why, (unsigned long long)hdr, (unsigned long long)base, (unsigned long long)(base + size), (unsigned long long)lo,
           (unsigned long long)hi, memcmp((const void *)(uintptr_t)base, anim_h[h].copy, size) ? "DIFFERS from" : "equals");
}
// A bad rotation's sample: where in the keyframe buffer it was read, the bytes there now and at first sight.
static void anim_sample(int h, uint64_t bone, float t) {
    anim_h[h].nsample++;
    if (!anim_h[h].copy) return;
    uint64_t hdr = anim_h[h].hdr, base = anim_h[h].base, size = anim_h[h].size;
    uint32_t stride;
    uint8_t cs;
    memcpy(&stride, (const void *)(uintptr_t)(hdr + 0x10c), 4);
    memcpy(&cs, (const void *)(uintptr_t)(hdr + 0x129), 1);
    int64_t idx = (int64_t)t;
    if ((float)idx != t && t < 0) idx--;
    uint32_t off = (uint32_t)(stride * (uint32_t)idx + (uint32_t)bone * cs);
    if ((uint64_t)off + 16 > size) {
        rt_log("danim sample: hdr %#llx frame %lld bone %llu: offset %#x is OUTSIDE the buffer (size %#llx)",
               (unsigned long long)hdr, (long long)idx, (unsigned long long)bone, off, (unsigned long long)size);
        return;
    }
    const uint8_t *live = (const uint8_t *)(uintptr_t)(base + off), *copy = anim_h[h].copy + off;
    char a[64], b[64];
    int oa = 0, ob = 0;
    for (int k = 0; k < 16; k++) oa += snprintf(a + oa, sizeof a - oa, "%02x%s", live[k], k == 7 ? " " : ""), ob += snprintf(b + ob, sizeof b - ob, "%02x%s", copy[k], k == 7 ? " " : "");
    rt_log("danim sample: hdr %#llx frame %lld bone %llu at %#llx = buffer +%#x (%u%% in): now %s | first sight %s | %s",
           (unsigned long long)hdr, (long long)idx, (unsigned long long)bone, (unsigned long long)(base + off), off,
           (unsigned)(100ull * off / size), a, b, memcmp(live, copy, 16) ? "CHANGED since first sight" : "same as at first sight");
}
static SRWLOCK anim_lock = SRWLOCK_INIT;
static int anim_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_ANIM_CHECK");
    return on;
}
void rt_anim_t(Cpu *c, float base, float off, uint64_t hdr, int path) {
    if (!anim_on()) return;
    anim_base = base, anim_off = off, anim_hdr = hdr, anim_path = path;
    anim_entry = c->x[24];
}
void rt_anim_q(Cpu *c, uint64_t fp, uint64_t entry) {
    if (!anim_on()) return;
    // The source-list header is saved at sp+56 by f_700042bea0. A count other than one takes the blend path
    // before the time hook. Matching the entry as well prevents attribution to another bone's last sample.
    uint64_t list = 0;
    uint32_t sources = 0;
    if (plat_mem_readable(c->sp + 56, 8)) memcpy(&list, (const void *)(uintptr_t)(c->sp + 56), 8);
    if (plat_mem_readable(list, 4)) memcpy(&sources, (const void *)(uintptr_t)list, 4);
    int source_valid = sources == 1 && anim_path == 1 && anim_entry == entry;
    float q[4] = {0}, r[12];
    if (source_valid) {
        memcpy(&q[0], (const void *)(uintptr_t)(fp - 0x58), 4);
        memcpy(&q[1], (const void *)(uintptr_t)(fp - 0x5c), 4);
        memcpy(&q[2], (const void *)(uintptr_t)(fp - 0x60), 4);
        memcpy(&q[3], (const void *)(uintptr_t)(fp - 0x64), 4);
    }
    memcpy(r, (const void *)(uintptr_t)(entry + 0x60), 48);   // the rotation just written (rows of 4)
    float n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        float rn = r[i * 4] * r[i * 4] + r[i * 4 + 1] * r[i * 4 + 1] + r[i * 4 + 2] * r[i * 4 + 2];
        bad |= !(rn > 0.8f && rn < 1.2f);
    }
    uint32_t raw[4] = {0};
    if (source_valid) {
        memcpy(&raw[0], (const void *)(uintptr_t)(fp - 0x70), 4);
        memcpy(&raw[1], (const void *)(uintptr_t)(fp - 0x54), 4);
        memcpy(&raw[2], (const void *)(uintptr_t)(fp - 0x50), 4);
        memcpy(&raw[3], (const void *)(uintptr_t)(fp - 0x4c), 4);
    }
    float t = anim_base + anim_off;
    static long checks, bads;
    long k = __atomic_add_fetch(&checks, 1, __ATOMIC_RELAXED);
    AcquireSRWLockExclusive(&anim_lock);
    int h = 128;
    if (source_valid)
        for (h = 0; h < 128 && anim_h[h].hdr && anim_h[h].hdr != anim_hdr; h++) {}
    if (h < 128 && !anim_h[h].hdr) {
        anim_h[h].hdr = anim_hdr, anim_h[h].good_max = -1e30f, anim_h[h].good_min = 1e30f;
        anim_first_sight(h);
    }
    if (h < 128) {
        if (!(++anim_h[h].calls & 511)) anim_diff(h);
        if (bad && !anim_h[h].ndump) anim_dump(h, "first bad rotation");
        if (bad && anim_h[h].nsample < 12) anim_sample(h, c->x[19], t);
        if (!bad) {
            anim_h[h].good++;
            if (t > anim_h[h].good_max) anim_h[h].good_max = t;
            if (t < anim_h[h].good_min) anim_h[h].good_min = t;
        } else anim_h[h].bad++;
    }
    if (bad) {
        __atomic_add_fetch(&bads, 1, __ATOMIC_RELAXED);
        static long reports[2];   // keep blended samples from exhausting the single-source report budget
        long b = ++reports[source_valid];
        if (b <= 40 && !source_valid) {
            rt_log("danim bad rotation rows (%g %g %g | %g %g %g | %g %g %g) entry %#llx sources %u: "
                   "matrix only; blend/source context not captured (time/header/quaternion omitted)",
                   r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9], r[10], (unsigned long long)entry, sources);
        }
        if (b <= 40 && source_valid) {
            uint32_t stride = 0;
            uint64_t data = 0, resource = 0, backing = 0, source_base = 0;
            if (plat_mem_readable(anim_hdr, 0x130)) {
                memcpy(&stride, (const void *)(uintptr_t)(anim_hdr + 0x10c), 4);
                memcpy(&data, (const void *)(uintptr_t)(anim_hdr + 0x28), 8);
                // hdr+0x28 is added to this external base by the guest; zero is a valid offset.
                memcpy(&resource, (const void *)(uintptr_t)(anim_hdr + 0x40), 8);
                if (resource && plat_mem_readable(resource + 0x10, 8))
                    memcpy(&backing, (const void *)(uintptr_t)(resource + 0x10), 8);
                if (backing && plat_mem_readable(backing + 0x100, 8))
                    memcpy(&source_base, (const void *)(uintptr_t)(backing + 0x100), 8);
            }
            rt_log("danim bad rotation rows (%g %g %g | %g %g %g | %g %g %g) raw %08x %08x %08x %08x quat (%g %g %g %g |q|^2=%g) path %d t=%.6g (base %.6g + off %.6g) hdr %#llx stride %u data_offset "
                   "%#llx source_base %#llx entry %#llx | this hdr: %llu good (observed t %g..%g; not clip bounds), %llu bad",
                   r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9], r[10], raw[0], raw[1], raw[2], raw[3],
                   q[0], q[1], q[2], q[3], n, anim_path, t, anim_base, anim_off, (unsigned long long)anim_hdr, stride,
                   (unsigned long long)data, (unsigned long long)source_base, (unsigned long long)entry,
                   h < 128 ? (unsigned long long)anim_h[h].good : 0ull,
                   h < 128 ? anim_h[h].good_min : 0.f, h < 128 ? anim_h[h].good_max : 0.f,
                   h < 128 ? (unsigned long long)anim_h[h].bad : 0ull);
            if (h < 128 && !anim_h[h].dumped && plat_mem_readable(anim_hdr, 0x130)) {
                anim_h[h].dumped = 1;
                char s[2400];
                int o = 0;
                for (int i = 0; i < 0x130; i += 4) {
                    uint32_t w;
                    memcpy(&w, (const void *)(uintptr_t)(anim_hdr + i), 4);
                    o += snprintf(s + o, sizeof s - o, "%s%08x", i % 32 ? " " : i ? " | " : "", w);
                }
                rt_log("danim hdr %#llx: %s", (unsigned long long)anim_hdr, s);
            }
        }
    }
    if (!(k & 0xfffff)) rt_log("danim: %ld quaternion-path rotations, %ld with non-unit rows (Euler paths not sampled)",
                             k, __atomic_load_n(&bads, __ATOMIC_RELAXED));
    ReleaseSRWLockExclusive(&anim_lock);
}

// ---- MKW_ANIM_CHECK: NVN memory pools (nvn_api_core.c calls rt_anim_pool_note at nvnMemoryPoolInitialize and
// nvnMemoryPoolFinalize). Each pool of at most APL_HASH_MAX bytes keeps the hash and first 32 bytes of its storage at
// initialization, and the guest return addresses of the call; anim_first_sight reports the record of the keyframe
// buffer's pool and earlier finalized pools over the same storage ("dapool").
enum { APL_N = 1 << 16, APL_HASH_MAX = 8u << 20 };
static struct {
    uint64_t pool, mem, size, hash, lr[4], fini_lr[4];
    uint32_t flags, seq, fini_seq;
    uint8_t head[32];
    uint8_t used, fini;
} apl[APL_N];
static uint32_t apl_next;
static SRWLOCK apl_lock = SRWLOCK_INIT;
static void apl_chain(Cpu *c, uint64_t lr[4]) {
    lr[0] = c->x[30];
    uint64_t sp = c->sp, fp = c->x[29];
    for (int i = 1; i < 4; i++) lr[i] = 0;
    for (int i = 1; i < 4 && fp >= sp && fp < sp + 0x80000 && !(fp & 7); i++) {
        lr[i] = *(const uint64_t *)(uintptr_t)(fp + 8);
        uint64_t up = *(const uint64_t *)(uintptr_t)fp;
        if (up <= fp) break;
        fp = up;
    }
}
void rt_anim_pool_note(int fini, uint64_t pool, uint64_t mem, uint64_t size, uint32_t flags, Cpu *c) {
    if (!anim_on() || !mem) return;
    uint64_t lr[4];
    apl_chain(c, lr);
    if (fini) apw_retire(pool);
    else if ((lr[0] == 0x70005d74a0ull || lr[0] == 0x70005d77c0ull) && flags == 0x24) apw_add(pool, mem, size);
    AcquireSRWLockExclusive(&apl_lock);
    if (fini) {   // newest live record of this pool object
        for (uint32_t k = 0; k < APL_N; k++) {
            uint32_t i = (apl_next - 1 - k) & (APL_N - 1);
            if (apl[i].used && !apl[i].fini && apl[i].pool == pool) {
                apl[i].fini = 1, apl[i].fini_seq = __atomic_load_n(&aht_seq, __ATOMIC_RELAXED);
                memcpy(apl[i].fini_lr, lr, sizeof lr);
                break;
            }
        }
    } else {
        uint32_t i = apl_next++ & (APL_N - 1);
        memset(&apl[i], 0, sizeof apl[i]);
        apl[i].pool = pool, apl[i].mem = mem, apl[i].size = size, apl[i].flags = flags, apl[i].used = 1;
        apl[i].seq = __atomic_load_n(&aht_seq, __ATOMIC_RELAXED);
        memcpy(apl[i].lr, lr, sizeof lr);
        if (size <= APL_HASH_MAX && plat_mem_readable(mem, size)) {
            apl[i].hash = anim_hash((const uint8_t *)(uintptr_t)mem, size);
            memcpy(apl[i].head, (const void *)(uintptr_t)mem, size < 32 ? size : 32);
        }
    }
    ReleaseSRWLockExclusive(&apl_lock);
}
static void apl_hex(char *s, size_t n, const uint8_t *p) {
    int o = 0;
    for (int k = 0; k < 32 && o < (int)n - 3; k++) o += snprintf(s + o, n - o, "%02x%s", p[k], k % 8 == 7 && k < 31 ? " " : "");
}
// Reports the pool records over [lo, hi): the live one and up to 8 finalized ones before it.
static void apl_report(uint64_t hdr, uint64_t pool, uint64_t lo, uint64_t hi) {
    AcquireSRWLockShared(&apl_lock);
    int shown = 0, olds = 0;
    for (uint32_t k = 0; k < APL_N && k < apl_next && (shown < 1 || olds < 8); k++) {
        uint32_t i = (apl_next - 1 - k) & (APL_N - 1);
        if (!apl[i].used || apl[i].mem >= hi || apl[i].mem + apl[i].size <= lo) continue;
        int mine = apl[i].pool == pool && !apl[i].fini;
        if (!mine && olds >= 8) continue;
        char a[80], b[80];
        apl_hex(a, sizeof a, apl[i].head);
        uint8_t now[32] = {0};
        if (plat_mem_readable(apl[i].mem, 32)) memcpy(now, (const void *)(uintptr_t)apl[i].mem, 32);
        apl_hex(b, sizeof b, now);
        uint64_t h = apl[i].size <= APL_HASH_MAX && plat_mem_readable(apl[i].mem, apl[i].size)
                         ? anim_hash((const uint8_t *)(uintptr_t)apl[i].mem, apl[i].size) : 0;
        rt_log("dapool hdr %#llx: %s pool %#llx mem [%#llx, %#llx) flags %#x init at heap event #%u callers %#llx %#llx %#llx "
               "%#llx%s | contents %s since init (init %s | now %s)",
               (unsigned long long)hdr, mine ? "ITS" : apl[i].fini ? "earlier finalized" : "other live",
               (unsigned long long)apl[i].pool, (unsigned long long)apl[i].mem, (unsigned long long)(apl[i].mem + apl[i].size),
               apl[i].flags, apl[i].seq, (unsigned long long)apl[i].lr[0], (unsigned long long)apl[i].lr[1],
               (unsigned long long)apl[i].lr[2], (unsigned long long)apl[i].lr[3], apl[i].fini ? " (finalized)" : "",
               !apl[i].hash ? "not hashed" : h == apl[i].hash ? "UNCHANGED" : "CHANGED", a, b);
        if (apl[i].fini)
            rt_log("dapool   finalized at heap event #%u by %#llx %#llx %#llx %#llx", apl[i].fini_seq,
                   (unsigned long long)apl[i].fini_lr[0], (unsigned long long)apl[i].fini_lr[1],
                   (unsigned long long)apl[i].fini_lr[2], (unsigned long long)apl[i].fini_lr[3]);
        if (mine) shown++;
        else olds++;
    }
    if (!shown) rt_log("dapool hdr %#llx: no live record of pool %#llx", (unsigned long long)hdr, (unsigned long long)pool);
    ReleaseSRWLockShared(&apl_lock);
}

// ---- MKW_ANIM_CHECK: who writes the animation keyframe pools. f_70005d72f0 creates one NVN pool per animation
// resource (rt_anim_pool_note sees nvnMemoryPoolInitialize return to 0x70005d74a0 / 0x70005d77c0); its storage is made
// read-only right after the call. The first write to each page faults: apw_veh records the writer (the lifted
// function, or for the host memcpy the guest caller and source) and makes that page writable again. File reads into
// a watched pool are recorded by rt_anim_preread (romfs.c, fs_host.c, aoc_romfs.c) and unprotect it first. GPU writes
// into imported memory do not fault and are not seen. anim_first_sight reports the record ("dapw").
#ifdef _WIN32
enum { APW_N = 1024, APW_W = 6 };
typedef struct {
    uint64_t pool, mem, size;
    uint32_t seq, pages_written, npages;
    struct { uint64_t fn, rip, lr, src, off, host[6]; uint32_t pages, seq; uint8_t kind; } w[APW_W];   // kind 0 cpu 1 memcpy 2 file
    int nw;
    uint8_t live;
} ApwRec;
static ApwRec apw[APW_N];
static volatile LONG apw_n;
static SRWLOCK apw_lock = SRWLOCK_INIT;
static PVOID apw_veh_h;
extern _Thread_local uint64_t rt_memcpy_lr, rt_memcpy_src;   // nvn_api_core.c ov_memcpy

static ApwRec *apw_find(uint64_t a) {   // apw_lock held
    int n = apw_n;
    for (int i = n - 1; i >= 0; i--)
        if (apw[i].live && a >= apw[i].mem && a < apw[i].mem + apw[i].size) return &apw[i];
    return NULL;
}
static _Thread_local uint64_t apw_host[6];   // apw_veh: mkw.exe code addresses found on the faulting thread's stack
static void apw_note(ApwRec *r, int kind, uint64_t fn, uint64_t rip, uint64_t lr, uint64_t src, uint64_t a, uint32_t pages) {
    int k;
    for (k = 0; k < r->nw; k++)
        if (r->w[k].kind == kind && r->w[k].fn == fn && r->w[k].lr == lr) break;
    if (k == r->nw && r->nw < APW_W) {
        r->nw++;
        r->w[k].kind = (uint8_t)kind, r->w[k].fn = fn, r->w[k].rip = rip, r->w[k].lr = lr, r->w[k].src = src;
        r->w[k].off = a - r->mem, r->w[k].pages = 0, r->w[k].seq = __atomic_load_n(&aht_seq, __ATOMIC_RELAXED);
        memcpy(r->w[k].host, apw_host, sizeof apw_host);
    }
    if (k < r->nw) r->w[k].pages += pages;
    r->pages_written += pages;
}
static LONG CALLBACK apw_veh(PEXCEPTION_POINTERS ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || ep->ExceptionRecord->NumberParameters < 2 ||
        ep->ExceptionRecord->ExceptionInformation[0] != 1 || !apw_n)
        return EXCEPTION_CONTINUE_SEARCH;
    uint64_t a = ep->ExceptionRecord->ExceptionInformation[1];
    AcquireSRWLockExclusive(&apw_lock);
    ApwRec *r = apw_find(a);
    if (!r) {
        ReleaseSRWLockExclusive(&apw_lock);
        return EXCEPTION_CONTINUE_SEARCH;
    }
    uint64_t rip = ep->ContextRecord->Rip, fn = rt_lifted_function((uintptr_t)rip);
    int kind = !fn && rt_memcpy_lr ? 1 : 0;
    // mkw.exe text addresses on the faulting stack (return addresses, most recent first): the host caller chain
    static uint64_t img_lo, img_hi;
    if (!img_lo) {
        img_lo = (uint64_t)(uintptr_t)GetModuleHandleW(NULL);
        IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(uintptr_t)(img_lo + ((IMAGE_DOS_HEADER *)(uintptr_t)img_lo)->e_lfanew);
        img_hi = img_lo + nt->OptionalHeader.SizeOfImage;
    }
    memset(apw_host, 0, sizeof apw_host);
    apw_host[0] = rip;
    const uint64_t *sp = (const uint64_t *)(uintptr_t)ep->ContextRecord->Rsp;
    for (int i = 0, n = 1; i < 512 && n < 6; i++)
        if (sp[i] >= img_lo && sp[i] < img_hi) apw_host[n++] = sp[i];
    apw_note(r, kind, fn, rip, kind ? rt_memcpy_lr : 0, kind ? rt_memcpy_src : 0, a, 1);
    DWORD old;
    VirtualProtect((void *)(uintptr_t)(a & ~0xfffull), 0x1000, PAGE_READWRITE, &old);
    ReleaseSRWLockExclusive(&apw_lock);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void apw_add(uint64_t pool, uint64_t mem, uint64_t size) {
    if ((mem & 0xfff) || (size & 0xfff) || !size || size > (16u << 20)) return;
    AcquireSRWLockExclusive(&apw_lock);
    if (!apw_veh_h) apw_veh_h = AddVectoredExceptionHandler(1, apw_veh);
    for (int i = 0; i < apw_n; i++)   // an older record over the same storage is retired
        if (apw[i].live && apw[i].mem < mem + size && mem < apw[i].mem + apw[i].size) apw[i].live = 0;
    if (apw_n < APW_N) {
        ApwRec *r = &apw[apw_n];
        memset(r, 0, sizeof *r);
        r->pool = pool, r->mem = mem, r->size = size, r->npages = (uint32_t)(size >> 12), r->live = 1;
        r->seq = __atomic_load_n(&aht_seq, __ATOMIC_RELAXED);
        DWORD old;
        if (VirtualProtect((void *)(uintptr_t)mem, size, PAGE_READONLY, &old)) apw_n++;
    }
    ReleaseSRWLockExclusive(&apw_lock);
}
static void apw_retire(uint64_t pool) {
    AcquireSRWLockExclusive(&apw_lock);
    for (int i = 0; i < apw_n; i++)
        if (apw[i].live && apw[i].pool == pool) {
            DWORD old;
            apw[i].live = 0;
            VirtualProtect((void *)(uintptr_t)apw[i].mem, apw[i].size, PAGE_READWRITE, &old);
        }
    ReleaseSRWLockExclusive(&apw_lock);
}
// mem_thaw (memory.c): the range must be writable for a kernel write or a Vulkan host-memory import. Watched pages in
// it are made writable; the record notes that its watch ended there (kind 3).
void rt_anim_unwatch(uint64_t a, uint64_t n) {
    if (!apw_n || !n) return;
    AcquireSRWLockExclusive(&apw_lock);
    for (int i = 0; i < apw_n; i++) {
        ApwRec *r = &apw[i];
        if (!r->live || a >= r->mem + r->size || a + n <= r->mem) continue;
        uint64_t lo = a > r->mem ? a : r->mem, hi = a + n < r->mem + r->size ? a + n : r->mem + r->size;
        lo &= ~0xfffull, hi = (hi + 0xfff) & ~0xfffull;
        apw_note(r, 3, 0, 0, 0, 0, lo, 0);
        DWORD old;
        VirtualProtect((void *)(uintptr_t)lo, hi - lo, PAGE_READWRITE, &old);
    }
    ReleaseSRWLockExclusive(&apw_lock);
}
void rt_anim_preread(uint64_t a, uint64_t n) {
    if (!apw_n || !n) return;
    AcquireSRWLockExclusive(&apw_lock);
    for (int i = 0; i < apw_n; i++) {
        ApwRec *r = &apw[i];
        if (!r->live || a >= r->mem + r->size || a + n <= r->mem) continue;
        uint64_t lo = a > r->mem ? a : r->mem, hi = a + n < r->mem + r->size ? a + n : r->mem + r->size;
        lo &= ~0xfffull, hi = (hi + 0xfff) & ~0xfffull;
        apw_note(r, 2, 0, 0, 0, 0, lo, (uint32_t)((hi - lo) >> 12));
        DWORD old;
        VirtualProtect((void *)(uintptr_t)lo, hi - lo, PAGE_READWRITE, &old);
    }
    ReleaseSRWLockExclusive(&apw_lock);
}
static void apw_report(uint64_t hdr, uint64_t pool) {
    static const char *const kn[4] = {"lifted code", "host memcpy", "file read", "watch ended by a thaw (import or read)"};
    AcquireSRWLockShared(&apw_lock);
    ApwRec *r = NULL;
    for (int i = apw_n - 1; i >= 0 && !r; i--)
        if (apw[i].pool == pool && apw[i].live) r = &apw[i];
    if (!r)
        rt_log("dapw hdr %#llx: pool %#llx was not watched (not created by f_70005d72f0 while watching)",
               (unsigned long long)hdr, (unsigned long long)pool);
    else {
        rt_log("dapw hdr %#llx: pool %#llx storage [%#llx, %#llx): %u of %u pages written by the CPU or a file read since "
               "creation (heap event #%u)%s",
               (unsigned long long)hdr, (unsigned long long)pool, (unsigned long long)r->mem,
               (unsigned long long)(r->mem + r->size), r->pages_written, r->npages, r->seq,
               r->pages_written ? "" : " -- NOTHING on the CPU side wrote it");
        for (int k = 0; k < r->nw; k++)
        {
            char mod[MAX_PATH] = "?";
            HMODULE hm = NULL;
            if (r->w[k].rip && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                  (LPCSTR)(uintptr_t)r->w[k].rip, &hm))
                GetModuleFileNameA(hm, mod, sizeof mod);
            uint64_t base = (uint64_t)(uintptr_t)GetModuleHandleW(NULL);
            rt_log("dapw   %s f_%llx at %#llx (%s), guest caller %#llx, source %#llx: first at +%#llx (heap event #%u), %u pages | "
                   "mkw.exe stack (image offsets) %#llx %#llx %#llx %#llx %#llx",
                   kn[r->w[k].kind], (unsigned long long)r->w[k].fn, (unsigned long long)r->w[k].rip, mod,
                   (unsigned long long)r->w[k].lr, (unsigned long long)r->w[k].src, (unsigned long long)r->w[k].off,
                   r->w[k].seq, r->w[k].pages, (unsigned long long)(r->w[k].host[1] ? r->w[k].host[1] - base : 0),
                   (unsigned long long)(r->w[k].host[2] ? r->w[k].host[2] - base : 0),
                   (unsigned long long)(r->w[k].host[3] ? r->w[k].host[3] - base : 0),
                   (unsigned long long)(r->w[k].host[4] ? r->w[k].host[4] - base : 0),
                   (unsigned long long)(r->w[k].host[5] ? r->w[k].host[5] - base : 0));
        }
    }
    ReleaseSRWLockShared(&apw_lock);
}
#else
static void apw_add(uint64_t pool, uint64_t mem, uint64_t size) { (void)pool, (void)mem, (void)size; }
static void apw_retire(uint64_t pool) { (void)pool; }
void rt_anim_preread(uint64_t a, uint64_t n) { (void)a, (void)n; }
void rt_anim_unwatch(uint64_t a, uint64_t n) { (void)a, (void)n; }
static void apw_report(uint64_t hdr, uint64_t pool) { (void)hdr, (void)pool; }
#endif

// ---- Collision investigation probe (HANDOVER 12e): hand-inserted calls in GENERATED code (main_0061.c
// f_7000514d90, the CrashFade2 controller setup; pristine copy main_0061.c.orig in the scratchpad). Counts calls per
// id and logs the first ones with x0 and the guest return addresses ("dprobe"). Regenerating the code removes the calls.
// Controllers seen by probe 0 (CrashFade2 setup, x0 = the controller); 25 s after the first, a background thread dumps
// the first 0x200 bytes of each controller and 0x800 of the object at [controller+0x20] (the actor) to
// probe-dump/ (cf2_<n>.bin), for working out where the actor keeps its world matrix.
static uint64_t probe_ctrl[4096];
static volatile LONG probe_nctrl;
static DWORD WINAPI probe_dump_thread(LPVOID arg) {
    (void)arg;
    Sleep(25000);
    CreateDirectoryA("probe-dump", NULL);
    int n = probe_nctrl < 4096 ? (int)probe_nctrl : 4096, written = 0;
    for (int i = 0; i < n && written < 64; i++) {
        uint64_t ctl = probe_ctrl[i], act = 0;
        if (!plat_mem_readable(ctl, 0x200)) continue;
        memcpy(&act, (const void *)(uintptr_t)(ctl + 0x20), 8);
        if (!plat_mem_readable(act, 0x800)) continue;
        char name[96];
        snprintf(name, sizeof name, "probe-dump/cf2_%03d.bin", written);
        FILE *f = fopen(name, "wb");
        if (!f) continue;
        fwrite(&ctl, 8, 1, f), fwrite(&act, 8, 1, f);
        fwrite((const void *)(uintptr_t)ctl, 1, 0x200, f);
        fwrite((const void *)(uintptr_t)act, 1, 0x800, f);
        fclose(f);
        written++;
    }
    rt_log("dprobe: dumped %d of %d CrashFade2 controllers and their actors to probe-dump/", written, n);
    // Where are the props' bodies? Each actor's world position (row-major 3x4 at actor+0x410, translation in column 3)
    // is searched for in all committed guest memory, to 0.05: a hit whose 48 bytes before hold three unit columns is a
    // column-major transform (Havok hkTransform: rotation columns, then the translation at +0x30).
    enum { NA = 4096 };
    static float pos[NA][3];
    static uint64_t act_of[NA];
    static int ncol[NA], nrow[NA], noth[NA];
    static uint64_t col_at[NA][4], oth_at[NA][6];
    int na = 0;
    for (int i = 0; i < n && na < NA; i++) {
        uint64_t ctl = probe_ctrl[i], act = 0;
        if (!plat_mem_readable(ctl + 0x20, 8)) continue;
        memcpy(&act, (const void *)(uintptr_t)(ctl + 0x20), 8);
        if (!plat_mem_readable(act + 0x410, 48)) continue;
        float m[12];
        memcpy(m, (const void *)(uintptr_t)(act + 0x410), 48);
        if (!(fabsf(m[3]) > 1.0f || fabsf(m[11]) > 1.0f) || !(fabsf(m[3]) < 1e5f && fabsf(m[7]) < 1e5f && fabsf(m[11]) < 1e5f)) continue;
        int dup = 0;
        for (int k = 0; k < na && !dup; k++) dup = act_of[k] == act;
        if (dup) continue;
        pos[na][0] = m[3], pos[na][1] = m[7], pos[na][2] = m[11], act_of[na] = act;
        na++;
    }
    uint64_t scanned = 0, arr_found = 0;
    for (uint64_t a = 0x100000000ull; a < 0x8000000000ull;) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mbi, sizeof mbi)) break;
        uint64_t lo = (uint64_t)(uintptr_t)mbi.BaseAddress, hi = lo + mbi.RegionSize;
        int ok = mbi.State == MEM_COMMIT && !(mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) &&
                 (mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE));
        if (ok && hi - lo <= (1ull << 34)) {
            const float *p = (const float *)(uintptr_t)lo;
            size_t cnt = (size_t)((hi - lo) / 4);
            for (size_t j = 12; j + 3 < cnt; j++) {
                if (!arr_found && !(j & 3) && j + 100 < cnt) {   // three consecutive body entries (ids n, n+1, n+2)
                    const uint32_t *u = (const uint32_t *)(p + j);
                    uint32_t i0 = u[0] & 0xffffff, f0 = u[0] >> 24;
                    if (i0 > 16 && i0 < 0x100000 && f0 && f0 < 16 && (u[48] & 0xffffff) == i0 + 1 && (u[96] & 0xffffff) == i0 + 2) {
                        const float *t = p + j + 4;
                        float l0 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2], l1 = t[4] * t[4] + t[5] * t[5] + t[6] * t[6];
                        if (l0 > 0.98f && l0 < 1.02f && l1 > 0.98f && l1 < 1.02f) arr_found = lo + 4 * (uint64_t)j;
                    }
                }
                float x = p[j];
                if (!(fabsf(x) > 1.0f && fabsf(x) < 1e5f)) continue;
                for (int k = 0; k < na; k++) {
                    if (fabsf(x - pos[k][0]) > 0.05f || fabsf(p[j + 1] - pos[k][1]) > 0.05f || fabsf(p[j + 2] - pos[k][2]) > 0.05f)
                        continue;
                    uint64_t at = lo + 4 * (uint64_t)j;
                    if (at >= act_of[k] && at < act_of[k] + 0x800) break;   // the actor's own copies
                    int col = 1;
                    for (int c3 = 0; c3 < 3 && col; c3++) {
                        const float *v = p + j - 12 + 4 * c3;
                        float l = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        col = l > 0.95f && l < 1.05f;
                    }
                    int row = !col && j >= 3 && fabsf(p[j - 3] * p[j - 3] + p[j - 2] * p[j - 2] + p[j - 1] * p[j - 1] - 1.0f) < 0.05f;
                    if (col) {
                        if (ncol[k] < 4) col_at[k][ncol[k]] = at;
                        ncol[k]++;
                    } else if (row) nrow[k]++;
                    else {
                        if (noth[k] < 6) oth_at[k][noth[k]] = at;
                        noth[k]++;
                    }
                    if (row && noth[k] < 6) oth_at[k][noth[k]++] = at | 1;   // keep the row-major hit too (bit 0 marks it)
                    break;
                }
            }
            scanned += hi - lo;
        }
        a = hi;
    }
    int nobody = 0;
    for (int k = 0; k < na; k++) {
        nobody += !ncol[k];
        rt_log("dprobe body: actor %#llx at (%.2f %.2f %.2f): %d column-major transforms (%#llx %#llx), %d row-major, %d other%s",
               (unsigned long long)act_of[k], pos[k][0], pos[k][1], pos[k][2], ncol[k], (unsigned long long)col_at[k][0],
               (unsigned long long)col_at[k][1], nrow[k], noth[k], ncol[k] ? "" : "  <-- NO BODY-LIKE TRANSFORM");
    }
    rt_log("dprobe body: %d actors, %d with no column-major transform at their position; %.1f GiB scanned", na, nobody,
           scanned / 1073741824.0);
    // Actors whose component arrays (+0x300) are empty: their state (+0x30..+0x40) and the list they sit in (+0x1a0,
    // +0x1a8); for the first few, the object around the list node and the nearest vtable-looking word before it.
    {
        int shown = 0, stuck = 0;
        static uint64_t heads[64];
        static int nheads[64], nh = 0;
        for (int k = 0; k < na; k++) {
            uint64_t act = act_of[k], vt = 0, l0 = 0, l1 = 0;
            uint32_t cnt0 = 0, st = 0;
            memcpy(&vt, (const void *)(uintptr_t)act, 8);
            memcpy(&cnt0, (const void *)(uintptr_t)(act + 0x308), 4);
            memcpy(&st, (const void *)(uintptr_t)(act + 0x34), 4);
            memcpy(&l0, (const void *)(uintptr_t)(act + 0x1a0), 8);
            memcpy(&l1, (const void *)(uintptr_t)(act + 0x1a8), 8);
            if (vt != 0x700352c148ull || cnt0) continue;
            stuck++;
            int h;
            for (h = 0; h < nh && heads[h] != l0; h++) {}
            if (h == nh && nh < 64) heads[nh] = l0, nheads[nh++] = 0;
            if (h < 64) nheads[h]++;
            if (shown++ < 6) {
                uint32_t w[16];
                memcpy(w, (const void *)(uintptr_t)(act + 0x20), 64);
                rt_log("dprobe stuck actor %#llx (%d body hits): +0x20.. %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                       "| list %#llx %#llx", (unsigned long long)act, ncol[k], w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9],
                       w[10], w[11], (unsigned long long)l0, (unsigned long long)l1);
            }
        }
        rt_log("dprobe stuck: %d actors of class 0x700352c148 with empty component arrays; %d distinct list words at +0x1a0", stuck,
               nh);
        for (int h = 0; h < nh && h < 8; h++) {
            uint64_t p = heads[h];
            if (!plat_mem_readable(p - 0x200, 0x300)) {
                rt_log("dprobe list %#llx (%d actors): not readable", (unsigned long long)p, nheads[h]);
                continue;
            }
            uint64_t vtab = 0, at = 0;
            for (uint64_t q = p; q >= p - 0x200 && !vtab; q -= 8) {
                uint64_t v;
                memcpy(&v, (const void *)(uintptr_t)q, 8);
                if (v >= 0x7003500000ull && v < 0x7003800000ull) vtab = v, at = q;
            }
            uint64_t qw[8];
            memcpy(qw, (const void *)(uintptr_t)(p - 0x20), 64);
            rt_log("dprobe list %#llx (%d actors): nearest vtable-like word %#llx at %#llx (-%#llx); around: %llx %llx %llx %llx [%llx] %llx "
                   "%llx %llx",
                   (unsigned long long)p, nheads[h], (unsigned long long)vtab, (unsigned long long)at, (unsigned long long)(p - at),
                   (unsigned long long)qw[0], (unsigned long long)qw[1], (unsigned long long)qw[2], (unsigned long long)qw[3],
                   (unsigned long long)qw[4], (unsigned long long)qw[5], (unsigned long long)qw[6], (unsigned long long)qw[7]);
        }
    }
    // The body array: entries of 0xc0 bytes whose transform (3 rotation columns, then the translation) starts at +0x10;
    // a column-major hit h is an entry's translation, so the entry is at h - 0x40. Walk +-20000 entries around the first
    // hit, keep the ones whose rotation columns are unit length, and match each actor without a hit by rotation alone.
    {
        uint64_t h0 = 0;
        for (int k = 0; k < na && !h0; k++)
            if (ncol[k]) h0 = col_at[k][0];
        if (!h0 && arr_found) h0 = arr_found + 0x40;   // an entry found by its id sequence
        if (h0) {
            enum { NB = 40001 };
            static float brot[NB][9], bpos[NB][3];
            static uint64_t baddr[NB];
            int nb = 0, near_origin = 0;
            uint64_t base = h0 - 0x40;
            for (int64_t e = -20000; e <= 20000 && nb < NB; e++) {
                uint64_t ent = base + (uint64_t)(e * 0xc0);
                if (!plat_mem_readable(ent, 0xc0)) continue;
                const float *t = (const float *)(uintptr_t)(ent + 0x10);
                int ok = 1;
                for (int c3 = 0; c3 < 3 && ok; c3++) {
                    float l = t[4 * c3] * t[4 * c3] + t[4 * c3 + 1] * t[4 * c3 + 1] + t[4 * c3 + 2] * t[4 * c3 + 2];
                    ok = l > 0.98f && l < 1.02f;
                }
                if (!ok || !(fabsf(t[12]) < 1e6f && fabsf(t[13]) < 1e6f && fabsf(t[14]) < 1e6f)) continue;
                for (int q = 0; q < 3; q++)
                    for (int r3 = 0; r3 < 3; r3++) brot[nb][q * 3 + r3] = t[4 * q + r3];
                bpos[nb][0] = t[12], bpos[nb][1] = t[13], bpos[nb][2] = t[14], baddr[nb] = ent;
                near_origin += fabsf(t[12]) < 2.0f && fabsf(t[13]) < 2.0f && fabsf(t[14]) < 2.0f;
                nb++;
            }
            rt_log("dprobe bodies: %d body-like entries around %#llx, %d within 2 m of the world origin", nb,
                   (unsigned long long)base, near_origin);
            // The real array: the hit entry's id (low 24 bits of its first word) is its index, so the array starts id
            // entries before it. Walk while the id matches the index (free slots are counted apart) and classify.
            uint32_t w00;
            memcpy(&w00, (const void *)(uintptr_t)base, 4);
            uint64_t arr = base - (uint64_t)(w00 & 0xffffff) * 0xc0;
            int n_body = 0, n_free = 0, n_ident_origin = 0, n_rot_origin = 0, n_zero = 0, n_bad = 0, n_world = 0, misses = 0;
            int shown = 0;
            for (uint32_t i = 0; i < 200000 && misses < 64; i++) {
                uint64_t ent = arr + (uint64_t)i * 0xc0;
                if (!plat_mem_readable(ent, 0xc0)) break;
                uint32_t w0;
                memcpy(&w0, (const void *)(uintptr_t)ent, 4);
                if ((w0 & 0xffffff) != i) {
                    n_free++, misses++;
                    continue;
                }
                misses = 0;
                n_body++;
                const float *t = (const float *)(uintptr_t)(ent + 0x10);
                int unit = 1, zero = 1, finite = 1;
                for (int k2 = 0; k2 < 16; k2++) finite &= t[k2] == t[k2] && fabsf(t[k2]) < 1e7f;
                for (int c3 = 0; c3 < 3; c3++) {
                    float l = t[4 * c3] * t[4 * c3] + t[4 * c3 + 1] * t[4 * c3 + 1] + t[4 * c3 + 2] * t[4 * c3 + 2];
                    unit &= l > 0.98f && l < 1.02f;
                    zero &= l < 1e-6f;
                }
                int origin = fabsf(t[12]) < 2.0f && fabsf(t[13]) < 2.0f && fabsf(t[14]) < 2.0f;
                int ident = unit && fabsf(t[0] - 1) < 1e-3f && fabsf(t[5] - 1) < 1e-3f && fabsf(t[10] - 1) < 1e-3f;
                const char *kind = NULL;
                if (!finite) n_bad++, kind = "non-finite";
                else if (zero) n_zero++, kind = "zero rotation";
                else if (unit && origin && ident) n_ident_origin++;
                else if (origin) n_rot_origin++, kind = "rotated at origin";
                else n_world++;
                if (kind && shown < 12) {
                    shown++;
                    uint32_t u[4];
                    memcpy(u, (const void *)(uintptr_t)ent, 16);
                    rt_log("dprobe slot %u %s: %08x %08x %08x %08x | rot (%.3f %.3f %.3f) pos (%.3f %.3f %.3f)", i, kind, u[0], u[1],
                           u[2], u[3], t[0], t[1], t[2], t[12], t[13], t[14]);
                }
            }
            rt_log("dprobe array %#llx: %d bodies (%d free/other slots): %d placed in the world, %d identity at the origin, "
                   "%d rotated at the origin, %d zero rotation, %d non-finite",
                   (unsigned long long)arr, n_body, n_free, n_world, n_ident_origin, n_rot_origin, n_zero, n_bad);
            for (int k = 0; k < na; k++) {
                if (ncol[k]) continue;
                float m[12];
                memcpy(m, (const void *)(uintptr_t)(act_of[k] + 0x410), 48);
                int best = -1;
                float bestd = 1e30f;
                int same_rot = 0;
                for (int b = 0; b < nb; b++) {
                    // actor rows m[0..2], m[4..6], m[8..10]; body columns brot[b][0..2] etc. (column j = row j of R^T)
                    float err = 0;
                    for (int q = 0; q < 3; q++)
                        for (int r3 = 0; r3 < 3; r3++) err += fabsf(brot[b][q * 3 + r3] - m[r3 * 4 + q]);
                    if (err > 0.01f) continue;
                    same_rot++;
                    float dx = bpos[b][0] - m[3], dy = bpos[b][1] - m[7], dz = bpos[b][2] - m[11];
                    float dd = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (dd < bestd) bestd = dd, best = b;
                }
                if (best >= 0)
                    rt_log("dprobe match: actor %#llx at (%.1f %.1f %.1f): %d bodies with its rotation; nearest at (%.2f %.2f %.2f), "
                           "%.2f m away (entry %#llx)",
                           (unsigned long long)act_of[k], m[3], m[7], m[11], same_rot, bpos[best][0], bpos[best][1], bpos[best][2],
                           bestd, (unsigned long long)baddr[best]);
                else
                    rt_log("dprobe match: actor %#llx at (%.1f %.1f %.1f): NO body with its rotation", (unsigned long long)act_of[k],
                           m[3], m[7], m[11]);
            }
        }
    }
    for (int k = 0, shown = 0; k < 0 && shown < 3; k++) {
        if (!ncol[k]) continue;
        shown++;
        uint64_t at = col_at[k][0] - 0x30;
        for (int row = -4; row < 12; row++) {
            uint64_t r = at + 16 * (int64_t)row;
            uint32_t u[4];
            float f[4];
            memcpy(u, (const void *)(uintptr_t)r, 16);
            memcpy(f, u, 16);
            rt_log("dprobe arr k%d %#llx: %08x %08x %08x %08x | %.4g %.4g %.4g %.4g", k, (unsigned long long)r, u[0], u[1], u[2], u[3],
                   f[0], f[1], f[2], f[3]);
        }
    }
    for (int k = 0; k < 0; k++)
        for (int h = 0; h < 6 && h < noth[k]; h++) {
            uint64_t at = oth_at[k][h] & ~1ull;
            char s[1200];
            int o = 0;
            for (int q = -16; q < 24; q++) {
                float f;
                uint32_t u;
                memcpy(&u, (const void *)(uintptr_t)(at + 4 * (int64_t)q), 4);
                memcpy(&f, &u, 4);
                o += snprintf(s + o, sizeof s - o, "%s%s%.4g", q % 4 == 0 ? " |" : " ", q == 0 ? ">" : "", f);
            }
            rt_log("dprobe ctx actor %d hit %#llx%s:%s", k, (unsigned long long)at, (oth_at[k][h] & 1) ? " (row)" : "", s);
        }
    return 0;
}
void rt_probe_hit(Cpu *c, int id) {
    static long n[8];
    if (id < 0 || id >= 8) return;
    long k = __atomic_add_fetch(&n[id], 1, __ATOMIC_RELAXED);
    if (id == 0) {
        void rt_aww_register(uint64_t act);
        uint64_t act = 0;
        if (plat_mem_readable(c->x[0] + 0x20, 8)) memcpy(&act, (const void *)(uintptr_t)(c->x[0] + 0x20), 8);
        rt_aww_register(act);
        LONG i = InterlockedIncrement(&probe_nctrl) - 1;
        if (i < 4096) probe_ctrl[i] = c->x[0];
        if (i == 0) CloseHandle(CreateThread(NULL, 0, probe_dump_thread, NULL, 0, NULL));
    }
    if (k > 12 && (k & (k - 1))) return;
    uint64_t lr[4];
    apl_chain(c, lr);
    rt_log("dprobe %d: call %ld x0 %#llx x1 %#llx | callers %#llx %#llx %#llx %#llx", id, k, (unsigned long long)c->x[0],
           (unsigned long long)c->x[1], (unsigned long long)lr[0], (unsigned long long)lr[1], (unsigned long long)lr[2],
           (unsigned long long)lr[3]);
}

// ---- Collision investigation: write-watch of CrashFade2 actors' state (+0x30..+0x40) and list node (+0x1a0..+0x1b0).
// rt_probe_hit registers the actor of each controller (first AWW_N); their pages are read-only for AWW_MS. A write to a
// watched field is logged with the lifted function, the field and the value it wrote ("daww"); every write to the page
// runs one instruction with the page writable (trap flag) and the page is protected again.
#ifdef _WIN32
enum { AWW_N = 48, AWW_MS = 25000, AWW_LOGS = 600 };
static uint64_t aww_act[AWW_N];
static volatile LONG aww_n, aww_logs;
static volatile LONG aww_on;
static ULONGLONG aww_until;
static _Thread_local uint64_t aww_page, aww_addr, aww_fn, aww_rip, aww_act_hit;
static _Thread_local int aww_field;
static PVOID aww_veh_h;
static int aww_find(uint64_t a, uint64_t *act, int *field) {
    int n = aww_n;
    for (int i = 0; i < n; i++) {
        uint64_t b = aww_act[i];
        if (a >= b + 0x30 && a < b + 0x40) { *act = b, *field = (int)(a - b); return 1; }
        if (a >= b + 0x1a0 && a < b + 0x1b0) { *act = b, *field = (int)(a - b); return 1; }
    }
    return 0;
}
static int aww_in_page(uint64_t a) {
    uint64_t p = a & ~0xfffull;
    int n = aww_n;
    for (int i = 0; i < n; i++) {
        uint64_t b = aww_act[i];
        if (p == ((b + 0x30) & ~0xfffull) || p == ((b + 0x1a0) & ~0xfffull)) return 1;
    }
    return 0;
}
static void aww_protect(DWORD prot) {
    int n = aww_n;
    for (int i = 0; i < n; i++) {
        DWORD old;
        VirtualProtect((void *)(uintptr_t)((aww_act[i] + 0x30) & ~0xfffull), 0x1000, prot, &old);
        VirtualProtect((void *)(uintptr_t)((aww_act[i] + 0x1a0) & ~0xfffull), 0x1000, prot, &old);
    }
}
static LONG CALLBACK aww_veh(PEXCEPTION_POINTERS ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION && aww_on && ep->ExceptionRecord->NumberParameters >= 2 &&
        ep->ExceptionRecord->ExceptionInformation[0] == 1) {
        uint64_t a = ep->ExceptionRecord->ExceptionInformation[1];
        if (!aww_in_page(a)) return EXCEPTION_CONTINUE_SEARCH;
        DWORD old;
        aww_page = a & ~0xfffull, aww_addr = a, aww_act_hit = 0, aww_field = -1;
        uint64_t act;
        int field;
        if (aww_find(a, &act, &field)) {
            aww_act_hit = act, aww_field = field;
            aww_rip = ep->ContextRecord->Rip;
            aww_fn = rt_lifted_function((uintptr_t)aww_rip);
        }
        VirtualProtect((void *)(uintptr_t)aww_page, 0x1000, PAGE_READWRITE, &old);
        ep->ContextRecord->EFlags |= 0x100;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (code == EXCEPTION_SINGLE_STEP && aww_page) {
        DWORD old;
        if (aww_act_hit && InterlockedIncrement(&aww_logs) <= AWW_LOGS) {
            uint64_t v = 0;
            memcpy(&v, (const void *)(uintptr_t)(aww_addr & ~7ull), 8);
            uint32_t st = 0;
            memcpy(&st, (const void *)(uintptr_t)(aww_act_hit + 0x34), 4);
            rt_log("daww actor %#llx +%#x <- %#llx (word at +%#llx) by f_%llx (image+%#llx); state +0x34 now %u",
                   (unsigned long long)aww_act_hit, aww_field, (unsigned long long)v,
                   (unsigned long long)((aww_addr & ~7ull) - aww_act_hit), (unsigned long long)aww_fn,
                   (unsigned long long)(aww_rip - (uint64_t)(uintptr_t)GetModuleHandleW(NULL)), st);
        }
        if (aww_on && GetTickCount64() < aww_until) VirtualProtect((void *)(uintptr_t)aww_page, 0x1000, PAGE_READONLY, &old);
        else if (aww_on) {
            aww_on = 0;
            aww_protect(PAGE_READWRITE);
            rt_log("daww: watch ended");
        }
        aww_page = 0;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
void rt_aww_register(uint64_t act) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_SOUND_HEAP_TRACE");
    if (!on || !act || aww_n >= AWW_N) return;
    for (int i = 0; i < aww_n; i++)
        if (aww_act[i] == act) return;
    if (!aww_veh_h) aww_veh_h = AddVectoredExceptionHandler(1, aww_veh), aww_until = GetTickCount64() + AWW_MS, aww_on = 1;
    if (!aww_on) return;
    LONG i = InterlockedIncrement(&aww_n) - 1;
    aww_act[i] = act;
    DWORD old;
    VirtualProtect((void *)(uintptr_t)((act + 0x30) & ~0xfffull), 0x1000, PAGE_READONLY, &old);
    VirtualProtect((void *)(uintptr_t)((act + 0x1a0) & ~0xfffull), 0x1000, PAGE_READONLY, &old);
}
#else
void rt_aww_register(uint64_t act) { (void)act; }
#endif

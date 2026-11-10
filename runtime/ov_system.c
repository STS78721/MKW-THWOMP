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
    if (s_logged < 10) {
        s_logged++;
        rt_log("mem: nnmem_nlib_debug_break ignored (x0=%#llx x1=%#llx lr=%#llx)%s",
               (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[30],
               s_logged == 10 ? " (further breaks suppressed)" : "");
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

void __wrap_f_70005e10a0(Cpu *c) {
    if (__real_f_70005e10a0) {
        __real_f_70005e10a0(c);
    } else {
        GuestFn orig = rt_dispatch_lookup(0x70005e10a0ull);
        if (orig) orig(c);
    }
    if (c->x[0] == 0) {
        rt_log("heap: f_70005e10a0 getMaxAllocatableSize returned 0, providing fallback 32 MiB");
        c->x[0] = 32 * 1024 * 1024;
    }
}

void __wrap_f_70003f2d70(Cpu *c) {
    if (__real_f_70003f2d70) {
        __real_f_70003f2d70(c);
    } else {
        GuestFn orig = rt_dispatch_lookup(0x70003f2d70ull);
        if (orig) orig(c);
    }
    if (c->x[0] == 0) {
        rt_log("heap: f_70003f2d70 getMaxAllocatableSize returned 0, providing fallback 32 MiB");
        c->x[0] = 32 * 1024 * 1024;
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

    if (c->x[0] != 0) {
        return; // Normal guest allocation succeeded
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

void ov_system_post_load(void) {
    if (rt_application_id == 0x0400C3F00006E000ull) {
        mkw_enable_soft_skip();
        s_noop_hook_addr = hostfn_register("mkw_noop", ov_noop);
        s_expheap_hook_addr = hostfn_register("sead__ExpHeap__alloc", ov_expheap_alloc_hook);
        uint64_t *vtable1 = (uint64_t *)(uintptr_t)0x700363a330ull;
        uint64_t *vtable2 = (uint64_t *)(uintptr_t)0x70036e1e18ull;
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
        *(uint32_t *)(uintptr_t)0x70052db828ull = 1;
        rt_log("gs: set 0x70052db828 = 1 in ov_system_on_inits_done");
    }
}

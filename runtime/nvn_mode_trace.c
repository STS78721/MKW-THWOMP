// TOTK_NVN2_MODE_TRACE: passive scalar metadata only; no mode semantics or command decoding.
#include <stdlib.h>

#include "cfg.h"
#include "host.h"
#include "nvn_mode_trace.h"
#include "platform/platform.h"

typedef struct {
    uint64_t sequence, command_buffer, handle, submit, elapsed_ns;
    uint32_t kind, thread, value;
} ModeEvent;
static const char *const names[NVN_MODE_EVENT_COUNT] = {
    "initialize", "finalize", "begin", "end", "recording_mode", "geometry_mode", "prepare", "handle"
};
static PlatOnce once = PLAT_ONCE_INIT;
static PlatMutex lock = PLAT_MUTEX_INIT;
static int enabled = -1, sealed;
static ModeEvent *events;
static uint64_t seen[NVN_MODE_EVENT_COUNT], emitted[NVN_MODE_EVENT_COUNT], sequence;
static uint32_t used;
static uint32_t start_ms, duration_ms;
static uint64_t origin_ns, start_ns, duration_ns, excluded_before, excluded_after;

static void trace_init(void *unused) {
    (void)unused;
    int on = cfg_bool("TOTK_NVN2_MODE_TRACE");
    if (on) {
        int start = cfg_int("TOTK_NVN2_MODE_TRACE_START_MS");
        int duration = cfg_int("TOTK_NVN2_MODE_TRACE_DURATION_MS");
        if (start < 0 || start > 600000 || duration < 0 || duration > 600000) {
            rt_log("nvn2-mode: window settings outside 0..600000; capture disabled");
            __atomic_store_n(&enabled, 0, __ATOMIC_RELEASE);
            return;
        }
        start_ms = (uint32_t)start;
        duration_ms = (uint32_t)duration;
        start_ns = (uint64_t)start_ms * 1000000;
        duration_ns = (uint64_t)duration_ms * 1000000;
        events = calloc(NVN_MODE_TRACE_LIMIT, sizeof *events);
        if (!events) {
            rt_log("nvn2-mode: buffer allocation failed; capture disabled");
            on = 0;
        } else {
            origin_ns = plat_now_ns(); // relative to the first enabled hook, not guest/game time
            atexit(nvn_mode_trace_finish);
        }
    }
    __atomic_store_n(&enabled, on, __ATOMIC_RELEASE);
}

static int trace_on(void) {
    int on = __atomic_load_n(&enabled, __ATOMIC_ACQUIRE);
    if (on < 0) {
        plat_once(&once, trace_init, NULL);
        on = __atomic_load_n(&enabled, __ATOMIC_ACQUIRE);
    }
    return on;
}

void nvn_mode_trace(unsigned kind, uint64_t command_buffer, uint64_t handle, uint32_t value, uint64_t submit) {
    if (!trace_on() || kind >= NVN_MODE_EVENT_COUNT) return;
    plat_mutex_lock(&lock);
    if (!sealed) {
        uint64_t elapsed = plat_now_ns() - origin_ns;
        if (elapsed < start_ns) excluded_before++;
        else if (duration_ns && elapsed >= start_ns + duration_ns) excluded_after++;
        else {
            uint64_t no = ++sequence;
            seen[kind]++;
            if (used < NVN_MODE_TRACE_LIMIT) {
                // Reserve and publish under one lock, so terminal sealing cannot
                // expose a reserved record before its fields are complete.
                events[used++] = (ModeEvent){no, command_buffer, handle, submit, elapsed, kind, plat_thread_id(), value};
                emitted[kind]++;
            }
        }
    }
    plat_mutex_unlock(&lock);
}

void nvn_mode_trace_finish(void) {
    if (!trace_on()) return;
    plat_mutex_lock(&lock);
    if (sealed) { plat_mutex_unlock(&lock); return; }
    sealed = 1;
    __atomic_store_n(&enabled, 0, __ATOMIC_RELEASE);
    rt_log("nvn2-mode window schema=2 origin=first_hook start_ms=%u duration_ms=%u elapsed_ns=%llu excluded_before=%llu excluded_after=%llu",
           start_ms, duration_ms, (unsigned long long)(plat_now_ns() - origin_ns),
           (unsigned long long)excluded_before, (unsigned long long)excluded_after);
    for (uint32_t i = 0; i < used; i++) {
        const ModeEvent *e = &events[i];
        rt_log("nvn2-mode event seq=%llu type=%s thread=%u cb=%#llx handle=%#llx value=%u submit=%llu elapsed_ns=%llu",
               (unsigned long long)e->sequence, names[e->kind], e->thread,
               (unsigned long long)e->command_buffer, (unsigned long long)e->handle, e->value,
               (unsigned long long)e->submit, (unsigned long long)e->elapsed_ns);
    }
    rt_log("nvn2-mode capture seen=%llu emitted=%u dropped=%llu limit=%u overflow=%u order=observational sealed=1 nested=untracked",
           (unsigned long long)sequence, used, (unsigned long long)(sequence - used), NVN_MODE_TRACE_LIMIT,
           sequence > used);
    for (unsigned i = 0; i < NVN_MODE_EVENT_COUNT; i++)
        rt_log("nvn2-mode summary type=%s seen=%llu emitted=%llu dropped=%llu", names[i],
               (unsigned long long)seen[i], (unsigned long long)emitted[i], (unsigned long long)(seen[i] - emitted[i]));
    plat_mutex_unlock(&lock);
}

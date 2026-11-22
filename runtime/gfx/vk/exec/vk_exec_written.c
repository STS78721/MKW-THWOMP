// MKW_REPLAY_PAR_WSHARE (default 1; "ab": in the B blocks): the GPU-written ranges of a submission recorded in chunks
// (vk_exec_par.c), shared by the threads that record it. Each thread keeps the ranges it binds writable in its own
// command buffer resources (note_written, vk_exec_cb.c: one CbRes per thread and fence), so a range that chunks on
// several threads wrote was noted pending and merged back once per thread at the same fence: 775 -> 1,175 MiB
// compared per 60 frames for the same 8.6-8.8 MiB the GPU changed, merge_back 2.65 -> 3.40 ms a frame (#610,
// i7-14700F, 2-player race, MKW_WB_CENSUS). Here the first thread to note a range at the fence owns it: only its
// CbRes reports it pending and written back, once, as the serial replay's single CbRes does. Keyed by the fence as
// well, so a range is never merged into an earlier fence's write-back, which completes before a later write of it.
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "platform/platform.h"

typedef struct { uint64_t addr, size; } WKey;   // size 0: an empty slot

static struct {
    PlatMutex lock;
    uint64_t fence;   // the submission being shared, 0: none (read without the lock first)
    WKey *slot;
    uint32_t cap, n;  // cap: a power of two, at least twice n
} W = {.lock = PLAT_MUTEX_INIT};
static int mode = -1;

static uint32_t slot_of(uint64_t addr, uint64_t size, uint32_t cap) {
    return (uint32_t)(((addr ^ size * 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull) >> 32) & (cap - 1);
}

static void put(WKey *slot, uint32_t cap, WKey k) {
    uint32_t i = slot_of(k.addr, k.size, cap);
    while (slot[i].size) i = (i + 1) & (cap - 1);
    slot[i] = k;
}

static void grow(void) {   // W.lock held
    uint32_t cap = W.cap ? W.cap * 2 : 1024;   // a race submission writes ~330 distinct ranges (rt-z5)
    WKey *slot = calloc(cap, sizeof *slot);
    if (!slot) rt_fatal("vk: out of memory (written ranges)");
    for (uint32_t i = 0; i < W.cap; i++)
        if (W.slot[i].size) put(slot, cap, W.slot[i]);
    free(W.slot);
    W.slot = slot, W.cap = cap;
}

void vk_exec_wshare_set(uint64_t fence) {
    if (mode < 0) mode = rt_knob("REPLAY_PAR_WSHARE", "1");
    plat_mutex_lock(&W.lock);
    if (W.n) memset(W.slot, 0, sizeof *W.slot * W.cap);
    W.n = 0;
    __atomic_store_n(&W.fence, RT_KNOB_ON(mode) ? fence : 0, __ATOMIC_RELEASE);
    plat_mutex_unlock(&W.lock);
}

int vk_exec_wshare_owner(uint64_t fence, uint64_t addr, uint64_t size) {
    if (!size || !fence || __atomic_load_n(&W.fence, __ATOMIC_ACQUIRE) != fence) return 1;
    int own = 1;
    plat_mutex_lock(&W.lock);
    if (W.fence == fence) {
        if (2 * (W.n + 1) > W.cap) grow();
        uint32_t i = slot_of(addr, size, W.cap);
        for (; W.slot[i].size; i = (i + 1) & (W.cap - 1))
            if (W.slot[i].addr == addr && W.slot[i].size == size) {
                own = 0;
                break;
            }
        if (own) W.slot[i] = (WKey){addr, size}, W.n++;
    }
    plat_mutex_unlock(&W.lock);
    return own;
}

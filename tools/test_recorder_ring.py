"""CPU-only differential capacity test for vk_rec.c's actual wait_space function.

Extracts and compiles the production function, replacing only its pause with a
delayed consumer and counting acquire loads. No Vulkan device or game is started.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "runtime/vk_rec.c").read_text(encoding="utf-8")
start = source.index("static void wait_space(uint64_t need) {")
end = source.index("\n}\n", start) + 3
wait_space = source[start:end]

harness = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define RING_WORDS 256u
static volatile uint64_t tail, head;
static uint64_t put_h, seen_tail;
static int unpub;
static struct { uint64_t full_waits; } RS_;
static uint64_t loads, pauses, wakes, cases, reservations;
static uint32_t rng = 0x7921b9d5u;
static uint32_t random_word(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint64_t load_tail(const volatile uint64_t *p, int order) {
    assert(p == &tail && order == __ATOMIC_ACQUIRE);
    loads++;
    return __atomic_load_n(p, order);
}
static void consume(uint64_t n) {
    assert(n <= head - tail); // the consumer cannot read unpublished words
    tail += n;
}
static void wake(void) { wakes++; }
static void pause_consumer(void) {
    pauses++;
    assert(pauses < 100000000ull);
    // Delay consumption on some polls, then move by an arbitrary number of published words.
    if ((random_word() & 3) && head != tail) consume(1 + random_word() % (head - tail));
}
#define _mm_pause() pause_consumer()
#define __atomic_load_n(p, order) load_tail(p, order)
@WAIT_SPACE@
#undef __atomic_load_n

// The reference uses the true consumer position on every reservation. A stale cache may cause an extra refresh,
// but it must neither overwrite unread words nor wait when the reference already has enough space.
static void reserve(uint64_t need) {
    uint64_t before_tail = tail, before_loads = loads, before_waits = RS_.full_waits;
    int actual_full = put_h + need - tail > RING_WORDS;
    int cached_full = put_h + need - seen_tail > RING_WORDS;
    unpub = 7;
    wait_space(need);
    assert(put_h + need - tail <= RING_WORDS);
    assert(put_h + need - seen_tail <= RING_WORDS);
    assert(tail - seen_tail <= RING_WORDS);
    assert(RS_.full_waits - before_waits == (uint64_t)actual_full);
    if (!actual_full) assert(tail == before_tail);
    else assert(unpub == 0 && head == put_h);
    if (!cached_full) assert(loads == before_loads);
    else assert(loads > before_loads);
    put_h += need;
    assert(put_h - tail <= RING_WORDS);
    reservations++;
}

int main(void) {
    const uint64_t bases[] = {0, 17, 0x20000000000001ull, UINT64_MAX - 1024, UINT64_MAX - 7};
    for (unsigned b = 0; b < sizeof bases / sizeof *bases; b++) {
        put_h = seen_tail = tail = head = bases[b];
        for (int i = 0; i < 200000; i++) {
            if (random_word() & 1) head = put_h; // publish batches of varying lengths
            if (head != tail && (random_word() & 1)) consume(random_word() % (head - tail + 1));
            uint64_t n = 1 + random_word() % (RING_WORDS / 4), pos = put_h % RING_WORDS;
            if (pos + n > RING_WORDS) reserve(RING_WORDS - pos); // production pad before a wrapping packet
            reserve(n);
            cases++;
        }
    }
    // A completely delayed consumer forces the full path and publication before it can free any space.
    put_h = head = RING_WORDS; seen_tail = tail = 0;
    reserve(64);
    assert(wakes && pauses && RS_.full_waits);
    printf("recorder ring: passed %llu packets / %llu reservations; %llu acquire loads, %llu full waits; "
           "wrapping, >2^53 indices, uint64 overflow and delayed consumption\n",
           (unsigned long long)cases, (unsigned long long)reservations, (unsigned long long)loads,
           (unsigned long long)RS_.full_waits);
    return 0;
}
""".replace("@WAIT_SPACE@", wait_space)

compiler = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
if not compiler and Path("C:/msys64/ucrt64/bin/clang.exe").exists():
    compiler = "C:/msys64/ucrt64/bin/clang.exe"
if not compiler:
    raise SystemExit("a C compiler (clang or cc) is required")

with tempfile.TemporaryDirectory(prefix="mkw-recorder-ring-") as tmp:
    c = Path(tmp) / "ring.c"
    exe = Path(tmp) / ("ring.exe" if os.name == "nt" else "ring")
    c.write_text(harness, encoding="utf-8")
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", str(c), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

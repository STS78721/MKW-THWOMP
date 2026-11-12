"""CPU-only differential checks/benchmark of vk_memory.c's descriptor-key cache.

Compiles the production lookup and pool index with mock pool metadata and Windows
SRW locks. No Vulkan loader, driver or game process is used.
"""
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_pool_index.h"
typedef SRWLOCK PlatMutex;
static PlatMutex pool_lock = SRWLOCK_INIT;
static uint64_t locks;
static void plat_mutex_lock_shared(PlatMutex *m) { locks++; AcquireSRWLockShared(m); }
static void plat_mutex_unlock_shared(PlatMutex *m) { ReleaseSRWLockShared(m); }
enum { POOL_IMPORT, POOL_PHYS, POOL_VIRT };
#define NVN_VPOOL_PAGE UINT64_C(0x10000)
typedef struct PoolRec PoolRec;
typedef struct { PoolRec *phys; uint64_t off; } VPage;
struct PoolRec { int kind; uint64_t lo, size, storage_serial, npages; VPage *pt; };
typedef struct { uint64_t storage, offset; } VkDescKey;
static uint64_t pool_gen = 1;
static pidx pool_index;
'''

TESTS = r'''
static unsigned checks;
#define REQUIRE(c) do { checks++; if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(1); } } while (0)
static void bump(void) { __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE); }
static void add(PoolRec *r) {
    bump();
    pidx_ent e = {.lo=r->lo, .hi=r->lo+r->size, .buf=r, .key=r};
    REQUIRE(pidx_insert(&pool_index, &e));
}
static void remove_pool(PoolRec *r) { bump(); REQUIRE(pidx_remove(&pool_index, r, NULL)); }
static int reference(uint64_t addr, uint64_t size, VkDescKey *key) {
    memset(key, 0, sizeof *key);
    if (!addr || !size || addr > UINT64_MAX-size) return 0;
    plat_mutex_lock_shared(&pool_lock);
    int ok=descriptor_key_locked(addr,size,key);
    plat_mutex_unlock_shared(&pool_lock);
    return ok;
}
static void compare(uint64_t addr, uint64_t size) {
    VkDescKey a, b;
    int got=vk_mem_descriptor_key(addr,size,&a), want=reference(addr,size,&b);
    REQUIRE(got == want && !memcmp(&a,&b,sizeof a));
    uint64_t was=locks;
    REQUIRE(vk_mem_descriptor_key(addr,size,&a) == want && !memcmp(&a,&b,sizeof a));
    REQUIRE(locks == was); // exact repeat, including a miss, never takes the pool lock
}
static void expect(uint64_t addr, uint64_t size, int ok, uint64_t storage, uint64_t offset) {
    compare(addr,size);
    VkDescKey k;
    REQUIRE(vk_mem_descriptor_key(addr,size,&k) == ok);
    REQUIRE(k.storage == storage && k.offset == offset);
}
static uint32_t rng=0x12345678;
static uint32_t random32(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void cases(void) {
    PoolRec outer={POOL_IMPORT,0x1000000,0x20000,1,0,NULL};
    PoolRec nested={POOL_PHYS,0x1008000,0x4000,2,0,NULL};
    PoolRec physical={POOL_PHYS,0x2000000,0x80000,3,0,NULL};
    VPage pages[4]={{&physical,0},{&physical,NVN_VPOOL_PAGE},{NULL,0},{&physical,3*NVN_VPOOL_PAGE}};
    PoolRec virt={POOL_VIRT,0x3000000,4*NVN_VPOOL_PAGE,4,4,pages};
    add(&outer); add(&nested); add(&physical); add(&virt);
    expect(0x1008020,32,1,2,32); // nested pool wins
    expect(0x1008020,0x8000,1,0,0x1008020); // larger range chooses its containing outer pool
    expect(0x101ffff,1,1,0,0x101ffff);
    expect(0x101ffff,2,0,0,0);
    expect(0x300fff0,32,1,3,0xfff0); // contiguous virtual pages
    expect(0x301fff0,32,0,0,0); // crosses into an unmapped page
    expect(0,32,0,0,0); expect(UINT64_MAX-15,32,0,0,0); expect(1,0,0,0,0);
    remove_pool(&nested);
    expect(0x1008020,32,1,0,0x1008020); // cached success invalidated by finalization
    nested.storage_serial=22; add(&nested);
    expect(0x1008020,32,1,22,32); // same address, new backing lifetime
    pages[1].off=3*NVN_VPOOL_PAGE; bump();
    expect(0x300fff0,32,0,0,0); // cached virtual answer invalidated by remap
    pages[1].off=NVN_VPOOL_PAGE; bump();
    expect(0x300fff0,32,1,3,0xfff0); // cached miss invalidated by remap
    expect(0x4000020,32,0,0,0);
    PoolRec fresh={POOL_PHYS,0x4000000,0x10000,5,0,NULL}; add(&fresh);
    expect(0x4000020,32,1,5,32); // cached miss invalidated by initialization
    for (unsigned i=0;i<100000;i++) {
        if (i%127==0) { // map/unmap, change backing lifetime, preserve exact selection
            pages[2]=(VPage){random32()&1 ? &physical : NULL,(random32()%5)*NVN_VPOOL_PAGE};
            physical.storage_serial++; bump();
        }
        uint64_t bases[]={0x1000000,0x2000000,0x3000000,0x4000000,0x5000000};
        uint64_t addr=bases[random32()%5]+random32()%0x50000;
        uint64_t size=(uint64_t[]){1,32,64,0x1000,0x10020}[random32()%5];
        compare(addr,size);
    }
    pool_index.n=0; bump(); // shutdown invalidates every cached key
    expect(0x4000020,32,0,0,0);
    free(pool_index.e); free(pool_index.maxhi); memset(&pool_index,0,sizeof pool_index);
}
static double seconds(void) { LARGE_INTEGER n,f; QueryPerformanceCounter(&n); QueryPerformanceFrequency(&f); return (double)n.QuadPart/f.QuadPart; }
static volatile uint64_t checksum;
static void bench(void) {
    enum { N=4096, HOT=128, ITER=4000000 };
    PoolRec *r=calloc(N,sizeof *r);
    for (unsigned i=0;i<N;i++) { r[i]=(PoolRec){POOL_IMPORT,UINT64_C(0x100000000)+i*NVN_VPOOL_PAGE,NVN_VPOOL_PAGE,i+1,0,NULL}; add(&r[i]); }
    uint64_t queries[HOT];
    for (unsigned i=0;i<HOT;i++) queries[i]=r[(i%8)*71].lo+((i/8)*NVN_VPOOL_PAGE/64)+32;
    for (unsigned i=0;i<HOT;i++) compare(queries[i],32);
    VkDescKey k; uint64_t sum=0;
    double t=seconds(); locks=0;
    for (unsigned i=0;i<ITER;i++) { reference(queries[i&(HOT-1)],32,&k); sum+=k.offset; }
    double baseline=seconds()-t; uint64_t baseline_locks=locks;
    t=seconds(); locks=0;
    for (unsigned i=0;i<ITER;i++) { vk_mem_descriptor_key(queries[i&(HOT-1)],32,&k); sum+=k.offset; }
    double cached=seconds()-t; checksum=sum;
    printf("descriptor-key benchmark: %u queries, uncached %.2f ms (%llu locks), cached %.2f ms (%llu locks), %.2fx\n",
           ITER,baseline*1000,(unsigned long long)baseline_locks,cached*1000,(unsigned long long)locks,baseline/cached);
    REQUIRE(locks < baseline_locks/2);
    free(r); free(pool_index.e); free(pool_index.maxhi);
}
int main(void) { cases(); bench(); printf("descriptor-key checks: %u passed\n",checks); return 0; }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=shutil.which('clang') or r'C:\msys64\ucrt64\bin\clang.exe')
    parser.add_argument('--out', type=Path, default=ROOT.parent / 'build' / 'mkw' / 'cpu-tests')
    args = parser.parse_args()
    source = (ROOT / 'runtime' / 'vk_memory.c').read_text()
    start = source.index('typedef struct { uint64_t gen, addr, size; VkDescKey key; int ok; } DescKeyHit;')
    end = source.index('\nint vk_mem_resolve(', start)
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out / 'descriptor-key-cache.c'
    exe = args.out / 'descriptor-key-cache.exe'
    src.write_text(PRELUDE + source[start:end] + TESTS)
    subprocess.run([args.cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                    str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()

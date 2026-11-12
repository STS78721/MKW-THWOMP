"""Compile the production GPU-address resolver against mock pool metadata.

Checks exact negative-cache invalidation, overlapping pools, virtual mappings,
and retrying a failed lazy import without any Vulkan calls or game process.
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
static uint64_t locks, imports;
static void plat_mutex_lock_shared(PlatMutex *m) { locks++; AcquireSRWLockShared(m); }
static void plat_mutex_unlock_shared(PlatMutex *m) { ReleaseSRWLockShared(m); }
static int32_t plat_xchg32(int32_t *p,int32_t n) { return __atomic_exchange_n(p,n,__ATOMIC_SEQ_CST); }
typedef uint64_t VkBuffer;
typedef void *VmaAllocation;
#define VK_NULL_HANDLE 0
#define NVN_VPOOL_PAGE UINT64_C(0x10000)
#define NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS 1
#define NVN_MEMORY_POOL_FLAGS_CPU_CACHED 2
enum { POOL_IMPORT=1, POOL_PHYS, POOL_VIRT };
enum { VK_MCLS_DEVICE, VK_MCLS_HOST_NOACC, VK_MCLS_HOST_CACHED, VK_MCLS_HOST_UNC };
typedef struct PoolRec PoolRec;
typedef struct { PoolRec *phys; uint64_t off; } VPage;
struct PoolRec {
    int kind; uint32_t flags; uint64_t lo, size, import_lo, import_hi;
    VkBuffer buf; VmaAllocation alloc; VPage *pt; uint64_t npages;
    int32_t used; uint8_t lazy, stouched; void *smap;
};
typedef struct {
    VkBuffer buffer; uint64_t offset, avail; VmaAllocation alloc;
    int cls; void *pool; uint64_t alloc_offset;
} VkMemSpan;
static uint64_t pool_gen=1;
static pidx pool_index;
static int shadow_on, check_on;
static const char *cfg_get(const char *key) { (void)key; return check_on ? "1" : NULL; }
#define rt_fatal(...) do { fprintf(stderr,__VA_ARGS__); exit(1); } while(0)
static void shadow_touch(PoolRec *r) { r->stouched=1; }
static void lazy_import(PoolRec *r) { imports++; if (imports > 1) r->buf=99, r->lazy=0; }
'''
TESTS = r'''
static unsigned checks;
#define REQUIRE(c) do { checks++; if (!(c)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); exit(1); } } while(0)
static void bump(void) { __atomic_add_fetch(&pool_gen,1,__ATOMIC_RELEASE); }
static void add(PoolRec *r) {
    bump(); pidx_ent e={.lo=r->lo,.hi=r->lo+r->size,.buf=r,.key=r}; REQUIRE(pidx_insert(&pool_index,&e));
}
static void remove_pool(PoolRec *r) { bump(); REQUIRE(pidx_remove(&pool_index,r,NULL)); }
static void compare(uint64_t a,uint64_t n) {
    VkMemSpan got,want; int x=vk_mem_resolve(a,n,&got), y=resolve_locked(a,n,&want);
    REQUIRE(x==y && !memcmp(&got,&want,sizeof got));
    REQUIRE(vk_mem_resolve(a,n,&got)==y && !memcmp(&got,&want,sizeof got));
}
static void miss(uint64_t a,uint64_t n) {
    compare(a,n); VkMemSpan out; uint64_t was=locks;
    REQUIRE(!vk_mem_resolve(a,n,&out)); REQUIRE(locks==was+(check_on ? 1 : 0));
    VkMemSpan zero={0}; REQUIRE(!memcmp(&out,&zero,sizeof out));
}
static uint32_t state=0x12345678;
static uint32_t random32(void) { state^=state<<13; state^=state>>17; state^=state<<5; return state; }
static void cases(void) {
    PoolRec outer={.kind=POOL_IMPORT,.lo=0x1000000,.size=0x20000,.import_lo=0x1000000,.import_hi=0x1020000,.buf=1};
    PoolRec nested={.kind=POOL_PHYS,.lo=0x1008000,.size=0x4000,.buf=2,.alloc=(void*)2};
    PoolRec physical={.kind=POOL_PHYS,.lo=0x2000000,.size=0x80000,.buf=3,.alloc=(void*)3};
    VPage pages[4]={{&physical,0},{&physical,NVN_VPOOL_PAGE},{NULL,0},{&physical,3*NVN_VPOOL_PAGE}};
    PoolRec virt={.kind=POOL_VIRT,.lo=0x3000000,.size=4*NVN_VPOOL_PAGE,.pt=pages,.npages=4};
    add(&outer); add(&nested); add(&physical); add(&virt);
    compare(0x1008020,32); compare(0x1008020,0x8000); compare(0x101ffff,1); miss(0x101ffff,2);
    miss(0x1000020,0x40000); compare(0x1000020,32); // exact request size participates in the miss key
    compare(0x1020000,0); miss(0x1020001,0); compare(0x101ffff,0);
    compare(0x300fff0,32); compare(0x301fff0,32); miss(0,32); miss(UINT64_MAX-15,32); miss(0x5000020,1);
    // Known virtual pools with an unmapped page must be retried even without a generation change.
    REQUIRE(!vk_mem_resolve(0x3020020,1,&(VkMemSpan){0}));
    pages[2]=(VPage){&physical,2*NVN_VPOOL_PAGE};
    REQUIRE(vk_mem_resolve(0x3020020,1,&(VkMemSpan){0}));
    pages[2]=(VPage){NULL,0}; bump();
    // Different requests deliberately occupying one direct-mapped cache slot cannot answer one another.
    uint64_t collision_a=0x88000000,collision_b=collision_a+4;
    while(rmiss_slot(collision_a,1)!=rmiss_slot(collision_b,1)) collision_b+=4;
    miss(collision_a,1); miss(collision_b,1); miss(collision_a,1);
    // A cached miss must become a hit as soon as a pool is registered.
    PoolRec fresh={.kind=POOL_PHYS,.lo=0x5000000,.size=0x10000,.buf=4,.alloc=(void*)4}; add(&fresh);
    VkMemSpan out; REQUIRE(vk_mem_resolve(0x5000020,1,&out) && out.buffer==4); compare(0x5000020,1);
    remove_pool(&fresh); miss(0x5000020,1); fresh.buf=44; add(&fresh);
    REQUIRE(vk_mem_resolve(0x5000020,1,&out) && out.buffer==44);
    remove_pool(&nested); compare(0x1008020,32); add(&nested); compare(0x1008020,32);
    pages[1].off=3*NVN_VPOOL_PAGE; bump(); compare(0x300fff0,32);
    pages[1].off=NVN_VPOOL_PAGE; bump(); compare(0x300fff0,32);
    // Known but unbacked pools are deliberately not negative-cached: retry without any generation change.
    PoolRec lazy={.kind=POOL_IMPORT,.lo=0x6000000,.size=0x10000,.import_lo=0x6000000,.import_hi=0x6010000,.lazy=1}; add(&lazy);
    REQUIRE(!vk_mem_resolve(lazy.lo,1,&out) && imports==1);
    REQUIRE(vk_mem_resolve(lazy.lo,1,&out) && imports==2 && out.buffer==99);
    PoolRec failed={.kind=POOL_IMPORT,.lo=0x7000000,.size=0x10000,.import_lo=0x7000000,.import_hi=0x7010000}; add(&failed);
    REQUIRE(!vk_mem_resolve(failed.lo,1,&out)); failed.buf=100;
    REQUIRE(vk_mem_resolve(failed.lo,1,&out) && out.buffer==100);
    for (unsigned i=0;i<100000;i++) {
        if (i%127==0) { pages[2]=(VPage){random32()&1 ? &physical : NULL,(random32()%5)*NVN_VPOOL_PAGE}; bump(); }
        uint64_t a=0x1000000+(random32()%8)*0x100000+random32()%0x50000;
        uint64_t n=(uint64_t[]){0,1,32,64,0x1000,0x10020}[random32()%6]; compare(a,n);
    }
    pool_index.n=0; bump(); miss(0x1008020,32);
    free(pool_index.e); free(pool_index.maxhi); memset(&pool_index,0,sizeof pool_index);
}
static double seconds(void) { LARGE_INTEGER n,f; QueryPerformanceCounter(&n); QueryPerformanceFrequency(&f); return (double)n.QuadPart/f.QuadPart; }
static volatile uint64_t checksum;
static void bench(void) {
    enum { N=4096,HOT=128,ITER=4000000 };
    PoolRec *r=calloc(N,sizeof *r);
    for (unsigned i=0;i<N;i++) { r[i]=(PoolRec){.kind=POOL_PHYS,.lo=UINT64_C(0x100000000)+i*NVN_VPOOL_PAGE,.size=NVN_VPOOL_PAGE,.buf=i+1}; add(&r[i]); }
    uint64_t q[HOT]; unsigned used[RMISS_SLOTS]={0};
    for(unsigned i=0;i<HOT;) { uint64_t a=((uint64_t)random32()<<20)|((uint64_t)random32()<<2); unsigned slot=(unsigned)(rmiss_slot(a,1)-rmiss); if(used[slot])continue; used[slot]=1; q[i++]=a; }
    VkMemSpan out; uint64_t sum=0; for(unsigned i=0;i<HOT;i++) miss(q[i],1);
    locks=0; double t=seconds();
    for(unsigned i=0;i<ITER;i++) sum+=resolve_locked(q[i&(HOT-1)],1,&out);
    double before=seconds()-t; uint64_t was=locks;
    locks=0; t=seconds();
    for(unsigned i=0;i<ITER;i++) sum+=vk_mem_resolve(q[i&(HOT-1)],1,&out);
    double after=seconds()-t; checksum=sum;
    printf("resolve-miss ideal hot-cache benchmark: %u probes, locked %.2f ms (%llu locks), cached %.2f ms (%llu locks), %.2fx\n",ITER,before*1000,(unsigned long long)was,after*1000,(unsigned long long)locks,before/after);
    REQUIRE(!locks); free(r); free(pool_index.e); free(pool_index.maxhi);
}
int main(int argc,char **argv) { (void)argv; check_on=argc>1; cases(); if(!check_on) bench(); printf("resolve-miss checks: %u passed (%s)\n",checks,check_on ? "checked" : "cached"); return 0; }
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=shutil.which('clang') or r'C:\msys64\ucrt64\bin\clang.exe')
    ap.add_argument('--out', type=Path, default=ROOT.parent / 'build' / 'mkw' / 'cpu-tests')
    args = ap.parse_args()
    source = (ROOT / 'runtime' / 'vk_memory.c').read_text()
    start = source.index('static int rec_class(')
    end = source.index('\nint vk_mem_inspect(', start)
    lookup = source.index('int vk_mem_resolve(')
    lookup_end = source.index('\nuint64_t vk_mem_gen(', lookup)
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out / 'resolve-miss-cache.c'
    exe = args.out / 'resolve-miss-cache.exe'
    src.write_text(PRELUDE + source[start:end] + source[lookup:lookup_end] + TESTS)
    subprocess.run([args.cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                    str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    subprocess.run([str(exe), '--check'], check=True)


if __name__ == '__main__':
    main()

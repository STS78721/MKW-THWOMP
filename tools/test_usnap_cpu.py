"""CPU-only differential tests and microbenchmarks for uniform snapshot bookkeeping.

Compiles the production sort/drop functions with host-only stand-ins. No Vulkan
loader, GPU or game process is used. Output compares with the previous drop
algorithm and a stable reference sort, including submission changes and overflow.
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
#define ALIGN 256u
typedef struct { uint64_t lo, hi; } Rng;
typedef struct { uint64_t lo, hi, off; int dropped; } Reg;
typedef struct { uint64_t lo, hi, fence; } Wrote;
typedef struct {
    int mode, on; uint64_t fence, gen;
    Rng *col; int ncol, ccol;
    Reg *reg; int nreg, creg;
    Wrote *wr; int nwr, cwr;
    uint64_t drops;
} State;
static State U;
static uint64_t done;
static uint64_t vk_sync_fence_done(void) { return done; }
static void rt_fatal(const char *message) { fprintf(stderr,"%s\n",message); exit(1); }
'''
TESTS = r'''
static unsigned checks;
#define REQUIRE(c) do { checks++; if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); exit(1); } } while(0)
static uint32_t rng=0x12345678;
static uint32_t random32(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void reference_drop(uint64_t addr,uint64_t n) {
    if (!U.mode || !n) return;
    if (U.nwr == U.cwr) U.wr=grow(U.wr,&U.cwr,sizeof *U.wr);
    U.wr[U.nwr++]=(Wrote){addr,addr+n,U.fence};
    if (!U.on) return;
    for (int i=first_above(addr);i<U.nreg && U.reg[i].lo<addr+n;i++)
        if (!U.reg[i].dropped) U.reg[i].dropped=1,U.drops++,U.gen++;
}
static Wrote *normalized(State *s,int *n) {
    State save=U; U=*s;
    U.wr=malloc((size_t)(U.nwr+1)*sizeof *U.wr);
    memcpy(U.wr,s->wr,(size_t)U.nwr*sizeof *U.wr);
    wr_prune(); Wrote *out=U.wr; *n=U.nwr; U=save; return out;
}
static int canonical_write(const void *a,const void *b) {
    const Wrote *x=a,*y=b;
    if(x->lo!=y->lo) return x->lo<y->lo?-1:1;
    if(x->hi!=y->hi) return x->hi<y->hi?-1:1;
    return x->fence<y->fence?-1:x->fence>y->fence;
}
static void compare_states(State *a,State *b) {
    REQUIRE(a->drops==b->drops && a->gen==b->gen && a->nreg==b->nreg);
    REQUIRE(!memcmp(a->reg,b->reg,(size_t)a->nreg*sizeof *a->reg));
    int na,nb; Wrote *wa=normalized(a,&na),*wb=normalized(b,&nb);
    // Wrapping inputs retain their malformed ranges; qsort has unspecified order for equal lo keys. Compare their
    // exact lo/hi/fence multiset after the real prune, without treating that unspecified order as a semantic change.
    qsort(wa,(size_t)na,sizeof *wa,canonical_write); qsort(wb,(size_t)nb,sizeof *wb,canonical_write);
    if (na!=nb || memcmp(wa,wb,(size_t)(na<nb?na:nb)*sizeof *wa)) {
        fprintf(stderr,"normalized writes differ fence %llu: old %d new %d\n",(unsigned long long)a->fence,na,nb);
        for(int i=0;i<na || i<nb;i++) {
            Wrote x=i<na?wa[i]:(Wrote){0},y=i<nb?wb[i]:(Wrote){0};
            if(memcmp(&x,&y,sizeof x)) fprintf(stderr,"%d: %llx..%llx@%llu vs %llx..%llx@%llu\n",i,
                (unsigned long long)x.lo,(unsigned long long)x.hi,(unsigned long long)x.fence,
                (unsigned long long)y.lo,(unsigned long long)y.hi,(unsigned long long)y.fence);
        }
        REQUIRE(0);
    }
    free(wa); free(wb);
}
static void drop_cases(void) {
    State old={.mode=1},now={.mode=1};
    old.nreg=now.nreg=64;
    old.reg=calloc(64,sizeof *old.reg); now.reg=calloc(64,sizeof *now.reg);
    for (unsigned sub=0;sub<200;sub++) {
        // Reusing a fence number must still reset the memo at the start of every snapshot.
        old.fence=now.fence=sub/2+1;
        old.gen++; now.gen++; old.on=now.on=sub%3!=0; old.mode=now.mode=sub%13!=0;
        done=sub/2>3 ? sub/2-3 : 0;
        U=old; wr_prune(); old=U; U=now; wr_prune(); now=U; drop_begin();
        for (int i=0;i<64;i++) old.reg[i]=now.reg[i]=(Reg){0x10000+(uint64_t)i*0x200,0x10100+(uint64_t)i*0x200,0,0};
        uint64_t last_addr=0,last_n=0;
        for (unsigned i=0;i<1200;i++) {
            uint64_t addr=0x10000+random32()%0x9000,n=random32()%0x400;
            if (i%3==0) addr=last_addr,n=last_n; // exact duplicates
            if (i%67==0) addr=UINT64_MAX-(random32()%128),n=256; // original unsigned-overflow behavior
            if (i%103==0) addr=0,n=32;
            if (i%131==0) n=0;
            U=old; reference_drop(addr,n); old=U;
            U=now; vk_usnap_drop(addr,n); now=U;
            last_addr=addr,last_n=n;
            if (i%79==0) compare_states(&old,&now);
        }
        compare_states(&old,&now);
    }
    // Epoch wrap must not resurrect an entry from an old submission.
    drop_epoch=UINT64_MAX; drop_begin(); REQUIRE(drop_epoch==1);
    REQUIRE(!drop_repeat(0x123400,32)); REQUIRE(drop_repeat(0x123400,32));
    drop_begin(); REQUIRE(!drop_repeat(0x123400,32));
    free(old.reg); free(now.reg); free(old.wr); free(now.wr);
}
static void stable_reference(Rng *a,int n) {
    for (int i=1;i<n;i++) { Rng x=a[i]; int j=i; while(j && a[j-1].lo>x.lo) { a[j]=a[j-1];j--; } a[j]=x; }
}
static void sort_cases(void) {
    enum { N=257 }; Rng want[N]; U=(State){0}; U.ccol=N; U.col=malloc(sizeof want);
    for (unsigned trial=0;trial<1500;trial++) {
        U.ncol=(int)(random32()%(N+1));
        unsigned bits=(unsigned[]){0,1,10,11,12,21,22,23,32,33,34}[trial%11];
        for (int i=0;i<U.ncol;i++) {
            uint64_t key=((uint64_t)random32()<<32)|random32();
            key&=bits ? (UINT64_C(1)<<bits)-1 : 0;
            if (bits>33) key=(uint64_t)(unsigned)i*(UINT64_C(1)<<34); // distinct keys for qsort fallback
            U.col[i]=(Rng){UINT64_C(0x200000000)+key*ALIGN,(uint64_t)(unsigned)i};
        }
        // Reverse some input runs; duplicate keys retain their exact order.
        if (trial&1) for (int i=0;i<U.ncol/2;i++) { Rng t=U.col[i];U.col[i]=U.col[U.ncol-1-i];U.col[U.ncol-1-i]=t; }
        memcpy(want,U.col,(size_t)U.ncol*sizeof *want); stable_reference(want,U.ncol);
        sort_col(); REQUIRE(!memcmp(want,U.col,(size_t)U.ncol*sizeof *want));
    }
    free(U.col);
}
static double seconds(void) { LARGE_INTEGER t,f;QueryPerformanceCounter(&t);QueryPerformanceFrequency(&f);return (double)t.QuadPart/f.QuadPart; }
static volatile uint64_t checksum;
static void drop_bench(void) {
    enum { N=2000000,HOT=64 }; State old={.mode=1,.fence=1},now=old;
    double t=seconds(); U=old;
    for(unsigned i=0;i<N;i++) reference_drop(0x100000+(uint64_t)(i&(HOT-1))*0x1000,0x1000);
    old=U; double baseline=seconds()-t;
    t=seconds(); U=now; drop_begin();
    for(unsigned i=0;i<N;i++) vk_usnap_drop(0x100000+(uint64_t)(i&(HOT-1))*0x1000,0x1000);
    now=U; double cached=seconds()-t; checksum=U.nwr;
    compare_states(&old,&now);
    printf("usnap-drop: %u writes, previous %.2f ms/%d queued, memo %.2f ms/%d queued (%.2fx; bookkeeping only)\n",
           N,baseline*1000,old.nwr,cached*1000,now.nwr,baseline/cached);
    REQUIRE(now.nwr<=HOT);
    free(old.wr);free(now.wr);
}
int main(void) { drop_cases();sort_cases();drop_bench();printf("usnap CPU checks: %u passed\n",checks);return 0; }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=shutil.which('clang') or r'C:\msys64\ucrt64\bin\clang.exe')
    parser.add_argument('--out', type=Path, default=ROOT.parent / 'build' / 'mkw' / 'cpu-tests')
    args = parser.parse_args()
    source = (ROOT / 'runtime' / 'vk_usnap.c').read_text()
    snippets = []
    for start, end in [
        ('typedef struct { uint64_t addr, size, epoch; } DropHit;', 'static Arena *arena_get('),
        ('static int by_lo(', 'static void summary('),
        ('static int first_above(', 'int vk_usnap_find('),
        ('void vk_usnap_drop(', 'uint64_t vk_usnap_gen('),
    ]:
        a = source.index(start)
        snippets.append(source[a:source.index(end, a)])
    args.out.mkdir(parents=True, exist_ok=True)
    src, exe = args.out / 'usnap-cpu.c', args.out / 'usnap-cpu.exe'
    src.write_text(PRELUDE + '\n'.join(snippets) + TESTS)
    subprocess.run([args.cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                    str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()

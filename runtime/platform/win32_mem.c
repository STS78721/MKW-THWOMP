// Windows implementation of platform.h, virtual memory and the write watch (the parts: win32_int.h).
#include <io.h>
#include <stdio.h>
#include <windows.h>
#include <bcrypt.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <timeapi.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

#include "platform.h"

#include "win32_int.h"

// ---- virtual memory: VirtualAlloc reservations, MEM_WRITE_WATCH for the watched ones.

int plat_mem_reserve(uint64_t addr, uint64_t n, int watch) {
    void *p = VirtualAlloc((void *)(uintptr_t)addr, n, MEM_RESERVE | (watch ? MEM_WRITE_WATCH : 0), PAGE_NOACCESS);
    if (p && p != (void *)(uintptr_t)addr) {
        VirtualFree(p, 0, MEM_RELEASE);
        p = NULL;
    }
    return p != NULL;
}

void *plat_mem_alloc(size_t n) { return VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }

int plat_mem_commit(uint64_t addr, size_t n) {
    return VirtualAlloc((void *)(uintptr_t)addr, n, MEM_COMMIT, PAGE_READWRITE) != NULL;
}

// Decommitted pages come back zeroed when committed again.
int plat_mem_decommit(uint64_t addr, uint64_t n) { return VirtualFree((void *)(uintptr_t)addr, n, MEM_DECOMMIT) != 0; }

int plat_mem_protect(uint64_t addr, uint64_t n, int prot) {
    DWORD old;
    DWORD p = prot == PLAT_PROT_RW ? PAGE_READWRITE : prot == PLAT_PROT_READ ? PAGE_READONLY : PAGE_NOACCESS;
    return VirtualProtect((void *)(uintptr_t)addr, n, p, &old) != 0;
}

int plat_mem_poke(uint64_t addr, const void *src, size_t n) { return (void)addr, (void)src, (void)n, 0; }
int plat_mem_peek(void *dst, uint64_t addr, size_t n) { return memcpy(dst, (const void *)(uintptr_t)addr, n), 1; }   // no read traps here

int plat_mem_state(uint64_t addr, uint64_t limit, uint64_t *end) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery((void *)(uintptr_t)addr, &mbi, sizeof mbi)) {
        *end = limit;
        return PLAT_MEM_FREE;
    }
    uint64_t e = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
    *end = e < limit ? e : limit;
    if (mbi.State == MEM_RESERVE) return PLAT_MEM_RESERVED;
    if (mbi.State != MEM_COMMIT) return PLAT_MEM_FREE;
    DWORD p = mbi.Protect & 0xff;
    return p == PAGE_READWRITE ? PLAT_PROT_RW : p == PAGE_READONLY ? PLAT_PROT_READ : PLAT_PROT_NONE;
}

// Adapted from LAS Recomp runtime/platform/windows.c:40-63 @ da35b83 (MIT, Copyright (c) 2026 Peiikai): GetWriteWatch
// with the reset flag.
uint64_t plat_mem_populated(uint64_t addr, uint64_t n, uint64_t (*runs)[2], uint64_t max) {
    (void)addr, (void)n, (void)runs, (void)max;
    return UINT64_MAX;
}

uint64_t plat_mem_written(uint64_t addr, uint64_t n, uint64_t *pages, uint64_t max, int reset) {
    if (!max || !n) return 0;
    ULONG_PTR got = (ULONG_PTR)max;
    DWORD gran;
    _Static_assert(sizeof(PVOID) == sizeof(uint64_t), "page addresses are written straight into a uint64_t array");
    if (GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, (void *)(uintptr_t)addr, (SIZE_T)n, (PVOID *)pages, &got,
                      &gran))
        return UINT64_MAX;
    return got;
}

static int (*fault_hook)(uint64_t addr);
static int (*lazy_hook)(uint64_t addr);   // plat_mem_lazy_hook: reads too
static int (*watch_fn)(uint64_t addr, uint64_t rip);   // plat_mem_watch_hook: write watches set for diagnostics

// A write access violation (ExceptionInformation[0] == 1) goes to the watch hook, then to the fault hook; registered
// first, before the crash handler. Reads are not trapped here (plat_mem_poke cannot, so memory.c sets no read traps).
static LONG CALLBACK fault_veh(EXCEPTION_POINTERS *ep) {
    const EXCEPTION_RECORD *er = ep->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    uint64_t addr = (uint64_t)er->ExceptionInformation[1];
    if (er->ExceptionInformation[0] <= 1 && lazy_hook && lazy_hook(addr)) return EXCEPTION_CONTINUE_EXECUTION;
    if (er->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    if (watch_fn && watch_fn(addr, (uint64_t)ep->ContextRecord->Rip)) return EXCEPTION_CONTINUE_EXECUTION;
    return fault_hook && fault_hook(addr) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

static void fault_install(void) {
    static volatile LONG done;
    if (!InterlockedExchange(&done, 1)) AddVectoredExceptionHandler(1, fault_veh);
}

_Thread_local int plat_fault_write;   // platform.h: not set here (no read traps on Windows)
_Thread_local uint64_t plat_fault_pc, plat_fault_ret;

void plat_mem_lazy_hook(int (*fn)(uint64_t addr)) {
    lazy_hook = fn;
    fault_install();
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_hook = fn;
    fault_install();
}

void plat_mem_watch_hook(int (*fn)(uint64_t addr, uint64_t rip)) {
    watch_fn = fn;
    fault_install();
}

static void (*crash_fn)(const PlatCrash *c);

static const char *crash_reason(DWORD code) {
    return code == EXCEPTION_ACCESS_VIOLATION ? "access violation"
         : code == EXCEPTION_ILLEGAL_INSTRUCTION ? "illegal instruction"
         : code == EXCEPTION_INT_DIVIDE_BY_ZERO ? "integer division by zero"
         : code == EXCEPTION_PRIV_INSTRUCTION ? "privileged instruction"
         : code == EXCEPTION_STACK_OVERFLOW ? "stack overflow"
         : code == EXCEPTION_IN_PAGE_ERROR ? "in-page error" : NULL;
}

static void fill_crash(PlatCrash *c, const char *what, EXCEPTION_POINTERS *ep, const char *mod,
                       uintptr_t base, int in_module) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    uintptr_t rip = (uintptr_t)ep->ContextRecord->Rip;
    *c = (PlatCrash){what, (uint32_t)code, rip, in_module ? rip - base : rip, -1, 0, mod, 0};
    uintptr_t sp = (uintptr_t)ep->ContextRecord->Rsp;   // a wild call left its return address there
    if (!in_module && sp && !(sp & 7) && plat_mem_readable(sp, 8)) c->ret = *(const uintptr_t *)sp;
    if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
        c->access = (int)ep->ExceptionRecord->ExceptionInformation[0],
        c->addr = ep->ExceptionRecord->ExceptionInformation[1];
}

static LONG CALLBACK crash_veh(EXCEPTION_POINTERS *ep) {
    const char *what = crash_reason(ep->ExceptionRecord->ExceptionCode);
    if (!what) return EXCEPTION_CONTINUE_SEARCH;
    uintptr_t rip = (uintptr_t)ep->ContextRecord->Rip;
    // Ours: mkw.exe, the game code library (mkwgame.dll), or no module at all (a jump to a wild address). Faults in
    // other modules (drivers) are left to the handler that expects them.
    HMODULE at = NULL, exe = GetModuleHandleW(NULL), game = GetModuleHandleW(L"mkwgame.dll");
    int in_module = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       (LPCWSTR)rip, &at);
    if (in_module && at != exe && (!game || at != game)) return EXCEPTION_CONTINUE_SEARCH;
    MODULEINFO mi = {0};
    if (in_module) GetModuleInformation(GetCurrentProcess(), at, &mi, sizeof mi);
    PlatCrash c;
    fill_crash(&c, what, ep, !in_module ? "no module" : at == exe ? "mkw" : "mkwgame", (uintptr_t)mi.lpBaseOfDll, in_module);
    crash_fn(&c);
    return EXCEPTION_CONTINUE_SEARCH;
}

// Fallback when an unhandled exception in an external DLL (Vulkan driver, layer, overlay) terminates the process,
// so the faulting module name and offset still land in mkw.log instead of ending silently.
static LONG WINAPI unhandled_crash_filter(EXCEPTION_POINTERS *ep) {
    if (!crash_fn || !ep || !ep->ExceptionRecord || !ep->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
    const char *what = crash_reason(ep->ExceptionRecord->ExceptionCode);
    if (!what) what = "unhandled exception";
    uintptr_t rip = (uintptr_t)ep->ContextRecord->Rip;
    HMODULE at = NULL;
    int in_module = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       (LPCWSTR)rip, &at);
    static char mod_name[128];
    const char *mod = "no module";
    MODULEINFO mi = {0};
    if (in_module) {
        GetModuleInformation(GetCurrentProcess(), at, &mi, sizeof mi);
        char path[MAX_PATH] = "";
        DWORD len = GetModuleFileNameA(at, path, sizeof path);
        const char *base = path;
        for (DWORD i = 0; i < len; i++)
            if (path[i] == '\\' || path[i] == '/') base = path + i + 1;
        snprintf(mod_name, sizeof mod_name, "%s", *base ? base : "unknown");
        mod = mod_name;
    }
    PlatCrash c;
    fill_crash(&c, what, ep, mod, (uintptr_t)mi.lpBaseOfDll, in_module);
    crash_fn(&c);
    return EXCEPTION_EXECUTE_HANDLER;
}

// The C runtime's check of a call's arguments (a time out of localtime_s's range, a null stream...) ends the process
// with 0xC0000409 and no exception the handler above sees: #310 died in ucrtbase.dll with an empty log while warming
// up. Reported as a crash instead, at the first frame of ours on the stack (a release CRT passes no expression or
// file), with the guest's frames.
static void crt_invalid_parameter(const wchar_t *expr, const wchar_t *fn, const wchar_t *file, unsigned line,
                                  uintptr_t reserved) {
    (void)expr, (void)fn, (void)file, (void)line, (void)reserved;
    void *frames[32];
    USHORT n = RtlCaptureStackBackTrace(1, 32, frames, NULL);
    HMODULE exe = GetModuleHandleW(NULL), game = GetModuleHandleW(L"mkwgame.dll"), at = NULL;
    PlatCrash c = {"C runtime invalid parameter", 0xC0000409u, 0, 0, -1, 0, "mkw", 0};
    for (USHORT i = 0; i < n && !c.rip; i++) {
        uintptr_t pc = (uintptr_t)frames[i];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)pc, &at) || (at != exe && at != game))
            continue;
        MODULEINFO mi = {0};
        GetModuleInformation(GetCurrentProcess(), at, &mi, sizeof mi);
        c.rip = pc, c.offset = pc - (uintptr_t)mi.lpBaseOfDll, c.module = at == exe ? "mkw" : "mkwgame";
    }
    crash_fn(&c);
}

void plat_crash_hook(void (*fn)(const PlatCrash *c)) {
    crash_fn = fn;
    AddVectoredExceptionHandler(0, crash_veh);
    SetUnhandledExceptionFilter(unhandled_crash_filter);
    _set_invalid_parameter_handler(crt_invalid_parameter);
}

// A fresh process gets a fresh layout: Windows places DLLs, heaps and thread stacks at random addresses. The parent
// waits and exits with the child's code.
void plat_reexec(void) {
    fflush(stdout);
    fflush(stderr);
    STARTUPINFOW si = {.cb = sizeof si};
    PROCESS_INFORMATION pi;
    wchar_t *cmd = _wcsdup(GetCommandLineW());   // CreateProcessW may write its command line
    if (!cmd || !CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "FATAL: re-running the process failed (error %lu)\n", GetLastError());
        ExitProcess(70);
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 70;
    GetExitCodeProcess(pi.hProcess, &code);
    ExitProcess(code);
}

int plat_mem_read(uint64_t addr, void *out, size_t n) {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void *)(uintptr_t)addr, out, n, &got) && got == n;
}

int plat_mem_readable(uint64_t addr, uint64_t n) {
    for (uint64_t a = addr, end = addr + n; a < end || a == addr;) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) return 0;
        if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return 0;
        a = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (a >= end) return 1;
    }
    return 1;
}

void *plat_aligned_alloc(size_t size, size_t align) { return _aligned_malloc(size, align); }
void plat_aligned_free(void *p) { _aligned_free(p); }


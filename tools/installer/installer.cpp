// MKW-THWOMP Setup: the Windows installer of Mario Kart World PC, and its uninstaller.
//
// What: one .exe (tools/installer/build.py) holding the player's folder (build/dist/windows: mkw.exe, its DLLs and
// setup/) compressed with zstd and appended after the program (THWOMPV1, read at the footer). Pages: Welcome, the
// install folder with the shortcuts, the copy with its progress, and the end, which can start the start screen; the
// start screen then sets the game up from the player's own dump (Game files). It registers in Windows' installed apps
// (HKCU, no administrator rights) with uninstall.exe: this program without the files, which removes what was installed
// and, if the player ticks it, the game data and saves too.
// Why: the port ships as a zip; THWOMP wants one setup in its own look: the dark theme (default, ui/menu/menu_theme.h),
// the MKW icon (runtime/platform/thwomp_icon.png), the README's words and a chequered flag (launcher_backdrop.cpp).
// Built with the start screen's pieces: SDL3 (linked statically, so the .exe stands alone), Dear ImGui with its SDL3 and
// SDL_Renderer back ends (runtime/third_party/imgui, MIT) and Roboto (runtime/ui/fonts, Apache-2.0).
// Test hooks: THWOMP_SETUP_SHOT=<frame>:<page> saves that page as THWOMP_SETUP_SHOT_OUT and quits (pages welcome,
// folder, copy, done, uninstall); THWOMP_SETUP_DIR=<folder> with THWOMP_SETUP_AUTO=1 installs there without input (as
// uninstall.exe: uninstalls without input); THWOMP_SETUP_NO_SHORTCUTS=1 is a test install: no shortcuts made or
// removed, and its own Windows apps entry (MKW-THWOMP-test).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <SDL3/SDL.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <float.h>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"   // #embed in C++
static const unsigned char kIconPng[] = {
#embed "../../runtime/platform/thwomp_icon.png"
};
static const unsigned char kRoboto[] = {
#embed "../../runtime/ui/fonts/Roboto-Medium.ttf"
};
#pragma clang diagnostic pop

static const char kName[] = "Mario Kart World PC";
static const wchar_t kNameW[] = L"Mario Kart World PC";
// A test install (THWOMP_SETUP_NO_SHORTCUTS) registers apart, so a test never touches the player's real install.
static const wchar_t *uninst_key(void) {
    return getenv("THWOMP_SETUP_NO_SHORTCUTS") ? L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MKW-THWOMP-test"
                                               : L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MKW-THWOMP";
}
static const char kMagic[8] = {'T', 'H', 'W', 'O', 'M', 'P', 'V', '1'};

// ---- the dark theme (ui/launcher/launcher_look.cpp's dark palette)
#define C_BG IM_COL32(14, 14, 17, 255)
#define C_PANEL IM_COL32(22, 22, 26, 255)
#define C_RAISED IM_COL32(38, 38, 45, 255)
#define C_HOVER IM_COL32(56, 40, 46, 255)
#define C_PRESSED IM_COL32(74, 44, 52, 255)
#define C_ACCENT IM_COL32(200, 28, 40, 255)
#define C_ACCENT_HOT IM_COL32(226, 44, 56, 255)
#define C_ACCENT_DOWN IM_COL32(170, 22, 32, 255)
#define C_TEXT IM_COL32(238, 238, 242, 255)
#define C_HEADING IM_COL32(186, 186, 196, 255)
#define C_DIM IM_COL32(150, 150, 162, 255)
#define C_OFF IM_COL32(86, 86, 96, 255)
#define C_LINE IM_COL32(255, 255, 255, 24)
#define C_OK IM_COL32(91, 201, 138, 255)
#define C_WARN IM_COL32(236, 160, 70, 255)
#define C_BAD IM_COL32(244, 112, 100, 255)

static ImVec4 v4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

// ---- text conversions
static std::wstring wide(const std::string &s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}
static std::string utf8(const std::wstring &w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(n ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}
static std::wstring self_path(void) {
    wchar_t p[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, p, (DWORD)(sizeof p / sizeof *p));
    return std::wstring(p, n);
}
static std::wstring dir_of(const std::wstring &p) {
    size_t k = p.find_last_of(L"\\/");
    return k == std::wstring::npos ? L"." : p.substr(0, k);
}
static void human(uint64_t b, char *out, size_t n) {
    if (b >= (1ull << 30)) snprintf(out, n, "%.1f GB", b / 1073741824.0);
    else snprintf(out, n, "%.0f MB", b / 1048576.0);
}

// ---- the payload: THWOMPV1, file count, total size, version; then each file's path, sizes and zstd frame. The footer
// (last 16 bytes): the payload's offset and the magic.
struct Entry { std::string path; uint64_t size, csize, at; };
struct Payload {
    uint64_t offset = 0, total = 0;
    std::string version;
    std::vector<Entry> files;
    bool ok = false;
};

static Payload read_payload(const std::wstring &exe) {
    Payload pl;
    FILE *f = _wfopen(exe.c_str(), L"rb");
    if (!f) return pl;
    char foot[16];
    if (_fseeki64(f, -16, SEEK_END) || fread(foot, 1, 16, f) != 16 || memcmp(foot + 8, kMagic, 8)) return fclose(f), pl;
    memcpy(&pl.offset, foot, 8);
    char head[8];
    uint32_t count = 0, vlen = 0;
    if (_fseeki64(f, (long long)pl.offset, SEEK_SET) || fread(head, 1, 8, f) != 8 || memcmp(head, kMagic, 8) ||
        fread(&count, 4, 1, f) != 1 || fread(&pl.total, 8, 1, f) != 1 || fread(&vlen, 4, 1, f) != 1 || vlen > 256)
        return fclose(f), pl;
    pl.version.resize(vlen);
    if (vlen && fread(pl.version.data(), 1, vlen, f) != vlen) return fclose(f), pl;
    for (uint32_t i = 0; i < count; i++) {
        uint16_t n = 0;
        Entry e;
        if (fread(&n, 2, 1, f) != 1) return fclose(f), pl;
        e.path.resize(n);
        if (fread(e.path.data(), 1, n, f) != n || fread(&e.size, 8, 1, f) != 1 || fread(&e.csize, 8, 1, f) != 1)
            return fclose(f), pl;
        e.at = (uint64_t)_ftelli64(f);
        if (_fseeki64(f, (long long)e.csize, SEEK_CUR)) return fclose(f), pl;
        pl.files.push_back(std::move(e));
    }
    fclose(f);
    pl.ok = true;
    return pl;
}

// ---- the work, on its own thread; the pages read its progress
struct Job {
    std::atomic<uint64_t> done{0}, total{0};
    std::atomic<int> state{0};   // 0 idle, 1 running, 2 finished, 3 failed
    std::mutex m;
    std::string current, error;
    void set_current(const std::string &s) { std::lock_guard<std::mutex> g(m); current = s; }
    std::string get_current(void) { std::lock_guard<std::mutex> g(m); return current; }
    void fail(const std::string &s) { { std::lock_guard<std::mutex> g(m); error = s; } state = 3; }
};
static Job g_job;

static bool make_dirs(const std::wstring &path) {
    int r = SHCreateDirectoryExW(nullptr, path.c_str(), nullptr);
    return r == ERROR_SUCCESS || r == ERROR_ALREADY_EXISTS || r == ERROR_FILE_EXISTS;
}

static bool shortcut(const std::wstring &lnk, const std::wstring &target, const std::wstring &dir) {
    IShellLinkW *sl = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW, (void **)&sl))) return false;
    sl->SetPath(target.c_str());
    sl->SetWorkingDirectory(dir.c_str());
    sl->SetIconLocation(target.c_str(), 0);
    sl->SetDescription(kNameW);
    IPersistFile *pf = nullptr;
    bool ok = SUCCEEDED(sl->QueryInterface(IID_IPersistFile, (void **)&pf)) && SUCCEEDED(pf->Save(lnk.c_str(), TRUE));
    if (pf) pf->Release();
    sl->Release();
    return ok;
}

static std::wstring known(REFKNOWNFOLDERID id) {
    PWSTR p = nullptr;
    std::wstring s;
    if (SUCCEEDED(SHGetKnownFolderPath(id, 0, nullptr, &p))) s = p;
    CoTaskMemFree(p);
    return s;
}
static std::wstring desktop_lnk(void) { return known(FOLDERID_Desktop) + L"\\Mario Kart World PC.lnk"; }
static std::wstring menu_lnk(void) { return known(FOLDERID_Programs) + L"\\Mario Kart World PC.lnk"; }

static void reg_str(HKEY k, const wchar_t *name, const std::wstring &v) {
    RegSetValueExW(k, name, 0, REG_SZ, (const BYTE *)v.c_str(), (DWORD)((v.size() + 1) * sizeof(wchar_t)));
}
static void reg_dword(HKEY k, const wchar_t *name, DWORD v) { RegSetValueExW(k, name, 0, REG_DWORD, (const BYTE *)&v, 4); }

static void install_run(Payload pl, std::wstring exe, std::wstring dir, bool desk, bool menu) {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    for (auto &ch : dir) if (ch == L'/') ch = L'\\';
    while (dir.size() > 3 && dir.back() == L'\\') dir.pop_back();
    g_job.total = pl.total;
    if (!make_dirs(dir)) return g_job.fail("The folder could not be created: " + utf8(dir));
    FILE *in = _wfopen(exe.c_str(), L"rb");
    if (!in) return g_job.fail("This setup could not read itself.");
    ZSTD_DCtx *dc = ZSTD_createDCtx();
    std::vector<char> ibuf(ZSTD_DStreamInSize()), obuf(ZSTD_DStreamOutSize());
    std::wstring manifest;
    for (const Entry &e : pl.files) {
        std::wstring rel = wide(e.path), out = dir + L"\\" + rel;
        for (auto &ch : out) if (ch == L'/') ch = L'\\';
        g_job.set_current(e.path);
        make_dirs(dir_of(out));
        FILE *o = _wfopen(out.c_str(), L"wb");
        if (!o) {
            fclose(in), ZSTD_freeDCtx(dc);
            return g_job.fail("Could not write " + e.path + ". Is the game or the start screen still open?");
        }
        _fseeki64(in, (long long)e.at, SEEK_SET);
        ZSTD_DCtx_reset(dc, ZSTD_reset_session_only);
        uint64_t left = e.csize;
        while (left) {
            size_t want = (size_t)std::min<uint64_t>(left, ibuf.size()), got = fread(ibuf.data(), 1, want, in);
            if (!got) break;
            left -= got;
            ZSTD_inBuffer ib = {ibuf.data(), got, 0};
            while (ib.pos < ib.size) {
                ZSTD_outBuffer ob = {obuf.data(), obuf.size(), 0};
                size_t r = ZSTD_decompressStream(dc, &ob, &ib);
                if (ZSTD_isError(r)) {
                    fclose(o), fclose(in), ZSTD_freeDCtx(dc);
                    return g_job.fail("This setup is damaged (" + e.path + "): download it again.");
                }
                fwrite(obuf.data(), 1, ob.pos, o);
                g_job.done += ob.pos;
            }
        }
        if (fclose(o)) return g_job.fail("Could not finish writing " + e.path + " (disk full?).");
        manifest += rel + L"\n";
    }
    ZSTD_freeDCtx(dc);
    // uninstall.exe: this program without its payload
    {
        std::vector<char> stub((size_t)pl.offset);
        _fseeki64(in, 0, SEEK_SET);
        size_t got = fread(stub.data(), 1, stub.size(), in);
        FILE *u = _wfopen((dir + L"\\uninstall.exe").c_str(), L"wb");
        if (u) fwrite(stub.data(), 1, got, u), fclose(u);
        manifest += L"uninstall.exe\n";
    }
    fclose(in);
    {
        FILE *m = _wfopen((dir + L"\\thwomp-install.txt").c_str(), L"wb");
        if (m) {
            std::string s = utf8(manifest);
            fwrite(s.data(), 1, s.size(), m), fclose(m);
        }
    }
    std::wstring game = dir + L"\\mkw.exe";
    if (desk) shortcut(desktop_lnk(), game, dir);
    if (menu) shortcut(menu_lnk(), game, dir);
    HKEY k;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, uninst_key(), 0, nullptr, 0, KEY_WRITE, nullptr, &k, nullptr) == ERROR_SUCCESS) {
        reg_str(k, L"DisplayName", kNameW);
        reg_str(k, L"DisplayVersion", wide(pl.version));
        reg_str(k, L"Publisher", L"MKW-THWOMP");
        reg_str(k, L"DisplayIcon", game);
        reg_str(k, L"InstallLocation", dir);
        reg_str(k, L"UninstallString", L"\"" + dir + L"\\uninstall.exe\"");
        reg_str(k, L"URLInfoAbout", L"https://github.com/STS78721/MKW-THWOMP");
        reg_dword(k, L"NoModify", 1);
        reg_dword(k, L"NoRepair", 1);
        reg_dword(k, L"EstimatedSize", (DWORD)(pl.total / 1024));
        RegCloseKey(k);
    }
    CoUninitialize();
    g_job.state = 2;
}

// A file still in use (the game or the start screen closing) is tried again for a few seconds.
// One still locked after that is handed to delete_self_later, which runs once this program has exited.
static std::mutex g_left_m;
static std::vector<std::wstring> g_left_files;
static void delete_retry(const std::wstring &p) {
    for (int i = 0; i < 30; i++) {
        if (DeleteFileW(p.c_str())) return;
        DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) return;
        Sleep(100);
    }
    std::lock_guard<std::mutex> g(g_left_m);
    g_left_files.push_back(p);
}

// The Python bytecode the setup's own Python wrote under setup/ (never installed, so not in the manifest).
static void remove_pycache(const std::wstring &dir) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring n = fd.cFileName;
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || n == L"." || n == L"..") continue;
        std::wstring sub = dir + L"\\" + n;
        if (n == L"__pycache__") {
            std::wstring from = sub + L'\0';   // SHFileOperation wants a double-NUL list
            SHFILEOPSTRUCTW op = {};
            op.wFunc = FO_DELETE;
            op.pFrom = from.c_str();
            op.fFlags = FOF_NO_UI;
            SHFileOperationW(&op);
        } else {
            remove_pycache(sub);
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}

// The uninstaller: the files the manifest lists (and, with everything, the whole folder), the shortcuts and the key.
static void uninstall_run(std::wstring dir, bool everything) {
    g_job.total = 1;
    std::wstring self = L"uninstall.exe";
    if (everything) {
        std::wstring from = dir + L'\0';   // SHFileOperation wants a double-NUL list
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                std::wstring n = fd.cFileName;
                if (n == L"." || n == L".." || !_wcsicmp(n.c_str(), self.c_str())) continue;
                g_job.set_current(utf8(n));
                std::wstring p = dir + L"\\" + n + L'\0';
                SHFILEOPSTRUCTW op = {};
                op.wFunc = FO_DELETE;
                op.pFrom = p.c_str();
                op.fFlags = FOF_NO_UI;
                SHFileOperationW(&op);
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
    } else {
        FILE *m = _wfopen((dir + L"\\thwomp-install.txt").c_str(), L"rb");
        std::string text;
        if (m) {
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, m))) text.append(buf, n);
            fclose(m);
        }
        std::vector<std::wstring> dirs;
        size_t a = 0;
        while (a < text.size()) {
            size_t b = text.find('\n', a);
            if (b == std::string::npos) b = text.size();
            std::wstring rel = wide(text.substr(a, b - a));
            a = b + 1;
            if (rel.empty() || !_wcsicmp(rel.c_str(), self.c_str())) continue;
            g_job.set_current(utf8(rel));
            std::wstring p = dir + L"\\" + rel;
            for (auto &ch : p) if (ch == L'/') ch = L'\\';
            delete_retry(p);
            for (std::wstring d = dir_of(p); d.size() > dir.size(); d = dir_of(d)) dirs.push_back(d);
        }
        DeleteFileW((dir + L"\\thwomp-install.txt").c_str());
        remove_pycache(dir + L"\\setup");
        std::sort(dirs.begin(), dirs.end(), [](const std::wstring &x, const std::wstring &y) { return x.size() > y.size(); });
        for (const auto &d : dirs) RemoveDirectoryW(d.c_str());   // only the ones left empty
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (!getenv("THWOMP_SETUP_NO_SHORTCUTS")) {   // a test uninstall leaves the player's shortcuts alone
        DeleteFileW(desktop_lnk().c_str());
        DeleteFileW(menu_lnk().c_str());
    }
    CoUninitialize();
    RegDeleteKeyW(HKEY_CURRENT_USER, uninst_key());
    g_job.done = 1;
    g_job.state = 2;
}

// After the window closes: uninstall.exe (and the folder, if now empty) removed by a hidden cmd once this has exited.
static void delete_self_later(const std::wstring &dir) {
    std::wstring cmd = L"cmd.exe /c ping -n 4 127.0.0.1 >nul";
    {
        std::lock_guard<std::mutex> g(g_left_m);
        for (const auto &f : g_left_files) cmd += L" & del /f /q \"" + f + L"\"";
        for (const auto &f : g_left_files)   // and the folders they kept, once empty
            for (std::wstring d = dir_of(f); d.size() > dir.size(); d = dir_of(d)) cmd += L" & rmdir \"" + d + L"\" 2>nul";
    }
    // the start screen's player folder only if it never held anything (rmdir without /s), then the install folder
    cmd += L" & del /f /q \"" + dir + L"\\uninstall.exe\" & rmdir \"" + dir + L"\\profile\" 2>nul & rmdir \"" + dir +
           L"\" 2>nul";
    STARTUPINFOW si = {sizeof si};
    PROCESS_INFORMATION pi = {};
    if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        CloseHandle(pi.hThread), CloseHandle(pi.hProcess);
}

// ---- drawing
struct Ui {
    SDL_Window *win = nullptr;
    SDL_Renderer *ren = nullptr;
    SDL_Texture *icon = nullptr;
    ImFont *body = nullptr, *head = nullptr, *big = nullptr, *small = nullptr;
    float scale = 1;
};
static Ui U;

// The MKW icon's chequered band as a cloth on the right, in the dark theme's red tones (launcher_backdrop.cpp's flag).
static void backdrop(ImDrawList *dl, float w, float h, float t) {
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), C_BG);
    const int cols = 8, rows = 7, sub = 4;
    float cell = h * 0.19f, amp = cell * 0.42f, rot = -0.16f, cr = cosf(rot), sr = sinf(rot);
    ImVec2 pivot(w * 0.86f, h * 0.48f), o(w * 0.86f - cell * cols * 0.42f, h * 0.48f - cell * rows * 0.5f);
    auto at = [&](float u, float v) {
        float x = o.x + u * cell, y = o.y + v * cell + amp * sinf(u * 0.62f - t) * (0.35f + u / cols);
        float dx = x - pivot.x, dy = y - pivot.y;
        return ImVec2(pivot.x + dx * cr - dy * sr, pivot.y + dx * sr + dy * cr);
    };
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i++)
            for (int k = 0; k < sub; k++) {
                float u0 = i + (float)k / sub, u1 = i + (float)(k + 1) / sub;
                float face = 0.5f + 0.5f * cosf((u0 + u1) * 0.31f - t);
                ImU32 c = (i + j) % 2 ? IM_COL32(12, 11, 13, 255)
                                      : ImGui::ColorConvertFloat4ToU32(ImVec4((30 + 28 * face) / 255.0f, (22 + 8 * face) / 255.0f,
                                                                               (26 + 10 * face) / 255.0f, 1));
                dl->AddQuadFilled(at(u0, (float)j), at(u1 + 0.02f, (float)j), at(u1 + 0.02f, j + 1.0f), at(u0, j + 1.0f), c);
            }
    ImU32 none = C_BG & ~IM_COL32_A_MASK;
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(w * 0.40f, h), C_BG);
    dl->AddRectFilledMultiColor(ImVec2(w * 0.40f - 1, 0), ImVec2(w * 0.72f, h), C_BG, none, none, C_BG);
    dl->AddRectFilledMultiColor(ImVec2(0, h * 0.74f), ImVec2(w, h * 0.86f), none, none, C_BG, C_BG);
    dl->AddRectFilled(ImVec2(0, h * 0.86f - 1), ImVec2(w, h), C_BG);
}

static bool accent_button(const char *label, ImVec2 size, bool enabled = true) {
    ImGui::PushStyleColor(ImGuiCol_Button, v4(enabled ? C_ACCENT : C_RAISED));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, v4(enabled ? C_ACCENT_HOT : C_RAISED));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, v4(enabled ? C_ACCENT_DOWN : C_RAISED));
    ImGui::PushStyleColor(ImGuiCol_Text, v4(enabled ? IM_COL32(255, 255, 255, 255) : C_OFF));
    bool r = ImGui::Button(label, size) && enabled;
    ImGui::PopStyleColor(4);
    return r;
}
static bool plain_button(const char *label, ImVec2 size) { return ImGui::Button(label, size); }

static void wrapped(ImU32 c, const char *s) {
    ImGui::PushStyleColor(ImGuiCol_Text, v4(c));
    ImGui::TextWrapped("%s", s);
    ImGui::PopStyleColor();
}

static void title_block(float S, const char *subtitle) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float mark = 84 * S;
    if (U.icon) dl->AddImage((ImTextureID)(intptr_t)U.icon, p, ImVec2(p.x + mark, p.y + mark));
    ImGui::SetCursorScreenPos(ImVec2(p.x + mark + 22 * S, p.y + 2 * S));
    ImGui::PushFont(U.big);
    ImGui::TextUnformatted(kName);
    ImGui::PopFont();
    ImGui::SetCursorScreenPos(ImVec2(p.x + mark + 24 * S, ImGui::GetCursorScreenPos().y - 2 * S));
    ImGui::PushStyleColor(ImGuiCol_Text, v4(C_HEADING));
    ImGui::TextUnformatted(subtitle);
    ImGui::PopStyleColor();
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + mark + 26 * S));
}

// A highlight from the README: a red bar, a title and a line.
static void card(float S, float width, const char *title, const char *text) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float th = U.body->FontSize, dh = U.small->CalcTextSizeA(U.small->FontSize, FLT_MAX, width - 32 * S, text).y;
    float h = 12 * S + th + 6 * S + dh + 14 * S;   // sized to its text
    dl->AddRectFilled(p, ImVec2(p.x + width, p.y + h), C_PANEL, 6 * S);
    dl->AddRectFilled(ImVec2(p.x, p.y + 12 * S), ImVec2(p.x + 4 * S, p.y + h - 12 * S), C_ACCENT, 2 * S);
    dl->AddText(U.body, th, ImVec2(p.x + 18 * S, p.y + 12 * S), C_TEXT, title);
    ImGui::SetCursorScreenPos(ImVec2(p.x + 18 * S, p.y + 12 * S + th + 6 * S));
    ImGui::PushFont(U.small);
    ImGui::PushStyleColor(ImGuiCol_Text, v4(C_DIM));
    ImGui::PushTextWrapPos(p.x + width - 14 * S - ImGui::GetWindowPos().x);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::TextUnformatted(text);
    ImGui::PopStyleVar();
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + 12 * S));
}

static void progress(float S, float width, float f) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = 10 * S;
    dl->AddRectFilled(p, ImVec2(p.x + width, p.y + h), C_RAISED, h * 0.5f);
    if (f > 0) dl->AddRectFilled(p, ImVec2(p.x + width * fminf(f, 1.0f), p.y + h), C_ACCENT, h * 0.5f);
    ImGui::Dummy(ImVec2(width, h + 8 * S));
}

// ---- the folder dialog's answer arrives on another thread
static std::mutex g_pick_m;
static std::string g_picked;
static void SDLCALL on_folder(void *, const char *const *list, int) {
    if (list && list[0]) {
        std::lock_guard<std::mutex> g(g_pick_m);
        g_picked = list[0];
    }
}

static uint64_t free_space(const std::wstring &dir) {
    std::wstring d = dir;
    ULARGE_INTEGER freeb;
    while (!d.empty()) {
        if (GetDiskFreeSpaceExW(d.c_str(), &freeb, nullptr, nullptr)) return freeb.QuadPart;
        std::wstring up = dir_of(d);
        if (up == d) break;
        d = up;
    }
    return 0;
}

enum Page { P_WELCOME, P_FOLDER, P_COPY, P_DONE, P_UNINSTALL, P_UNINSTALLING, P_UNINSTALLED };

int main(int argc, char **argv) {
    std::wstring exe = self_path(), exe_dir = dir_of(exe);
    std::wstring base = exe.substr(exe_dir.size() + 1);
    bool uninstall = !_wcsicmp(base.c_str(), L"uninstall.exe");
    for (int i = 1; i < argc; i++) uninstall |= !strcmp(argv[i], "--uninstall");
    Payload pl = uninstall ? Payload{} : read_payload(exe);

    int shot_frame = 0;
    std::string shot_page;
    if (const char *e = getenv("THWOMP_SETUP_SHOT")) {
        shot_frame = atoi(e);
        if (const char *c = strchr(e, ':')) shot_page = c + 1;
    }

    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "1");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        MessageBoxW(nullptr, L"This setup could not open its window.", kNameW, MB_ICONERROR);
        return 1;
    }
    float ds = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    if (ds <= 0) ds = 1;
    U.win = SDL_CreateWindow(uninstall ? "Mario Kart World PC: Uninstall" : "Mario Kart World PC Setup",
                             (int)(1180 * ds), (int)(720 * ds), SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);
    U.ren = U.win ? SDL_CreateRenderer(U.win, nullptr) : nullptr;
    if (!U.ren) {
        MessageBoxW(nullptr, L"This setup could not open its window.", kNameW, MB_ICONERROR);
        return 1;
    }
    SDL_SetRenderVSync(U.ren, 1);
    if (SDL_Surface *s = SDL_LoadPNG_IO(SDL_IOFromConstMem(kIconPng, sizeof kIconPng), true)) {
        SDL_SetWindowIcon(U.win, s);
        U.icon = SDL_CreateTextureFromSurface(U.ren, s);
        if (U.icon) SDL_SetTextureScaleMode(U.icon, SDL_SCALEMODE_LINEAR);
        SDL_DestroySurface(s);
    }
    SDL_ShowWindow(U.win);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    float S = ds;
    U.scale = S;
    ImFontConfig fc;
    fc.FontDataOwnedByAtlas = false;
    static const ImWchar ranges[] = {0x0020, 0x017F, 0x2010, 0x2027, 0x2190, 0x2193, 0};
    U.body = io.Fonts->AddFontFromMemoryTTF((void *)kRoboto, (int)sizeof kRoboto, 22 * S, &fc, ranges);
    U.small = io.Fonts->AddFontFromMemoryTTF((void *)kRoboto, (int)sizeof kRoboto, 18 * S, &fc, ranges);
    U.head = io.Fonts->AddFontFromMemoryTTF((void *)kRoboto, (int)sizeof kRoboto, 30 * S, &fc, ranges);
    U.big = io.Fonts->AddFontFromMemoryTTF((void *)kRoboto, (int)sizeof kRoboto, 46 * S, &fc, ranges);
    ImGuiStyle &st = ImGui::GetStyle();
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 0; st.FrameRounding = 4; st.GrabRounding = 4; st.WindowBorderSize = 0;
    st.FramePadding = ImVec2(16, 9); st.ItemSpacing = ImVec2(12, 12); st.WindowPadding = ImVec2(0, 0);
    st.ScaleAllSizes(S);
    ImVec4 *c = st.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_Text] = v4(C_TEXT);
    c[ImGuiCol_TextDisabled] = v4(C_DIM);
    c[ImGuiCol_Button] = v4(C_RAISED);
    c[ImGuiCol_ButtonHovered] = v4(C_HOVER);
    c[ImGuiCol_ButtonActive] = v4(C_PRESSED);
    c[ImGuiCol_FrameBg] = v4(C_RAISED);
    c[ImGuiCol_FrameBgHovered] = v4(C_HOVER);
    c[ImGuiCol_FrameBgActive] = v4(C_PRESSED);
    c[ImGuiCol_CheckMark] = v4(C_ACCENT_HOT);
    c[ImGuiCol_NavCursor] = v4(IM_COL32(255, 255, 255, 235));
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.78f, 0.11f, 0.16f, 0.45f);
    c[ImGuiCol_Border] = v4(C_LINE);
    ImGui_ImplSDL3_InitForSDLRenderer(U.win, U.ren);
    ImGui_ImplSDLRenderer3_Init(U.ren);

    Page page = uninstall ? P_UNINSTALL : P_WELCOME;
    if (shot_page == "folder") page = P_FOLDER;
    else if (shot_page == "copy") page = P_COPY, g_job.total = 100, g_job.done = 42, g_job.set_current("setup/zig/zig.exe");
    else if (shot_page == "done") page = P_DONE;
    else if (shot_page == "uninstall") page = P_UNINSTALL, uninstall = true;
    std::string sys_drive = "C:";
    if (const char *sd = getenv("SystemDrive")) sys_drive = sd;
    char path[1024];
    snprintf(path, sizeof path, "%s\\Games\\Mario Kart World PC", sys_drive.c_str());
    if (const char *d = getenv("THWOMP_SETUP_DIR")) snprintf(path, sizeof path, "%s", d);
    bool desk = true, menu = true, launch = true, everything = false, auto_run = getenv("THWOMP_SETUP_AUTO") != nullptr;
    if (getenv("THWOMP_SETUP_NO_SHORTCUTS")) desk = menu = false;   // (test) no shortcuts
    std::thread worker;
    float phase = 1.2f;
    bool quit = false;
    int frame = 0;
    while (!quit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_QUIT || ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                if (g_job.state != 1) quit = true;   // never in the middle of a copy
            }
        }
        {
            std::lock_guard<std::mutex> g(g_pick_m);
            if (!g_picked.empty()) {
                std::string p = g_picked;
                // a new folder of its own inside the one picked, unless it already is the game's
                if (p.find("Mario Kart World") == std::string::npos) p += "\\Mario Kart World PC";
                snprintf(path, sizeof path, "%s", p.c_str());
                g_picked.clear();
            }
        }
        if (!shot_frame && !(SDL_GetWindowFlags(U.win) & SDL_WINDOW_INPUT_FOCUS)) SDL_Delay(30);
        else if (!shot_frame) phase += io.DeltaTime * 0.55f;
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        int pw, ph;
        SDL_GetRenderOutputSize(U.ren, &pw, &ph);
        float W = (float)pw, H = (float)ph;
        backdrop(ImGui::GetBackgroundDrawList(), W, H, phase);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(W, H));
        ImGui::Begin("##setup", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground);
        ImGui::PushFont(U.body);
        float x0 = 70 * S, colw = fminf(W * 0.50f, 640 * S), top = 56 * S, foot = H - 86 * S;
        ImGui::Indent(x0);   // every line of the page starts under the title
        ImGui::SetCursorScreenPos(ImVec2(x0, top));
        ImVec2 bsz(170 * S, 0);
        auto footer = [&](void) { ImGui::SetCursorScreenPos(ImVec2(x0, foot)); };

        switch (page) {
        case P_WELCOME: {
            title_block(S, "MKW-THWOMP  ·  natively on Windows, not an emulator");
            card(S, colw, "Native code, not emulation",
                 "The game's ARM64 code is compiled into a Windows program on your PC. Nothing is emulated while you play.");
            card(S, colw, "Shaders ready before you race",
                 "All 11 604 shader programs are translated during setup, so worlds don't pop in while you drive.");
            card(S, colw, "60 fps, a PC settings menu, your controllers",
                 "Vulkan 1.3 with frame generation and FSR; a Switch-2-style menu (F1) in the dark theme; Xbox, DualSense and Switch 2 pads.");
            ImGui::SetCursorScreenPos(ImVec2(x0, ImGui::GetCursorScreenPos().y + 4 * S));
            ImGui::PushTextWrapPos(x0 + colw);
            ImGui::PushFont(U.small);
            wrapped(C_WARN, "No game files are included. After installing, the start screen sets the game up from your own dump "
                            "of Mario Kart World (1.0.0 or the 1.8.0 update).");
            ImGui::PopFont();
            ImGui::PopTextWrapPos();
            if (!pl.ok && !shot_frame) {
                ImGui::PushFont(U.small);
                wrapped(C_BAD, "This setup holds no game program (it was not built with tools/installer/build.py).");
                ImGui::PopFont();
            }
            footer();
            if (accent_button("Next", bsz, pl.ok || shot_frame)) page = P_FOLDER;
            ImGui::SameLine();
            if (plain_button("Cancel", bsz)) quit = true;
            break;
        }
        case P_FOLDER: {
            title_block(S, "Where should it go?");
            ImGui::PushFont(U.small);
            wrapped(C_HEADING, "Install folder");
            ImGui::PopFont();
            ImGui::SetNextItemWidth(colw - 150 * S);
            ImGui::InputText("##dir", path, sizeof path);
            ImGui::SameLine();
            if (plain_button("Browse…", ImVec2(138 * S, 0)))
                SDL_ShowOpenFolderDialog(on_folder, nullptr, U.win, path, false);
            uint64_t freeb = free_space(wide(path));
            char need[32], have[32], line[160];
            human(pl.total, need, sizeof need);
            human(freeb, have, sizeof have);
            snprintf(line, sizeof line, "Program: %s.  The game set up from your dump adds about 6 GB.  Free there: %s",
                     need, have);
            ImGui::PushFont(U.small);
            wrapped(freeb && freeb < pl.total + (6ull << 30) ? C_WARN : C_DIM, line);
            ImGui::PopFont();
            ImGui::Dummy(ImVec2(1, 10 * S));
            ImGui::Checkbox("Desktop shortcut", &desk);
            ImGui::Checkbox("Start menu shortcut", &menu);
            ImGui::Dummy(ImVec2(1, 10 * S));
            ImGui::PushTextWrapPos(x0 + colw);
            ImGui::PushFont(U.small);
            wrapped(C_DIM, "Installing over an earlier version updates the program and keeps your game files, saves, settings "
                           "and mods. No administrator rights are needed.");
            ImGui::PopFont();
            ImGui::PopTextWrapPos();
            footer();
            bool go = accent_button("Install", bsz, path[0] != 0) || (auto_run && pl.ok);
            if (go && !shot_frame) {
                auto_run = false;
                page = P_COPY;
                g_job.state = 1;
                worker = std::thread(install_run, pl, exe, wide(path), desk, menu);
            }
            ImGui::SameLine();
            if (plain_button("Back", bsz)) page = P_WELCOME;
            ImGui::SameLine();
            if (plain_button("Cancel", bsz)) quit = true;
            break;
        }
        case P_COPY: {
            title_block(S, g_job.state == 3 ? "Something went wrong" : "Installing…");
            uint64_t d = g_job.done, t = g_job.total;
            float f = t ? (float)d / (float)t : 0;
            progress(S, colw, f);
            char line[256];
            char a[32], b[32];
            human(d, a, sizeof a), human(t, b, sizeof b);
            snprintf(line, sizeof line, "%.0f %%   %s of %s", f * 100, a, b);
            ImGui::TextUnformatted(line);
            ImGui::PushFont(U.small);
            ImGui::PushTextWrapPos(x0 + colw);
            if (g_job.state == 3) {
                std::lock_guard<std::mutex> g(g_job.m);
                wrapped(C_BAD, g_job.error.c_str());
            } else {
                wrapped(C_DIM, g_job.get_current().c_str());
            }
            ImGui::PopTextWrapPos();
            ImGui::PopFont();
            if (g_job.state == 2) {
                if (worker.joinable()) worker.join();
                page = P_DONE;
            }
            footer();
            if (g_job.state == 3) {
                if (worker.joinable()) worker.join();
                if (plain_button("Back", bsz)) page = P_FOLDER, g_job.state = 0, g_job.done = 0;
                ImGui::SameLine();
                if (plain_button("Close", bsz)) quit = true;
            }
            break;
        }
        case P_DONE: {
            title_block(S, "Installed");
            ImGui::PushTextWrapPos(x0 + colw);
            wrapped(C_TEXT, "Mario Kart World PC is installed.");
            ImGui::PushFont(U.small);
            wrapped(C_DIM, "Next, on the start screen: open Game files, choose your dump and press Set up. It checks your copy, builds the game "
                           "code on this PC and translates the shaders (about 5 minutes). Then press Play.");
            ImGui::PopFont();
            ImGui::PopTextWrapPos();
            ImGui::Dummy(ImVec2(1, 8 * S));
            ImGui::Checkbox("Open the start screen now", &launch);
            footer();
            if (accent_button("Finish", bsz)) {
                if (launch && !shot_frame) {
                    std::wstring dir = wide(path);
                    ShellExecuteW(nullptr, L"open", (dir + L"\\mkw.exe").c_str(), nullptr, dir.c_str(), SW_SHOWNORMAL);
                }
                quit = true;
            }
            break;
        }
        case P_UNINSTALL: {
            title_block(S, "Uninstall");
            ImGui::PushTextWrapPos(x0 + colw);
            wrapped(C_TEXT, "This removes Mario Kart World PC's program files, its shortcuts and its entry in Windows' apps.");
            ImGui::PopTextWrapPos();
            ImGui::Dummy(ImVec2(1, 8 * S));
            ImGui::Checkbox("Also remove the game files set up from my dump, saves, settings and mods", &everything);
            ImGui::PushFont(U.small);
            ImGui::PushTextWrapPos(x0 + colw);
            wrapped(everything ? C_WARN : C_DIM, everything ? "Everything in this folder goes, your saves too. This cannot be undone."
                                                            : "Your saves and the set-up game stay in this folder.");
            ImGui::PopTextWrapPos();
            ImGui::PopFont();
            footer();
            if ((accent_button("Uninstall", bsz) || auto_run) && !shot_frame) {
                auto_run = false;
                page = P_UNINSTALLING;
                g_job.state = 1;
                worker = std::thread(uninstall_run, exe_dir, everything);
            }
            ImGui::SameLine();
            if (plain_button("Cancel", bsz)) quit = true;
            break;
        }
        case P_UNINSTALLING:
        case P_UNINSTALLED: {
            bool done = g_job.state == 2;
            if (done && worker.joinable()) worker.join();
            title_block(S, done ? "Uninstalled" : "Removing…");
            ImGui::PushFont(U.small);
            wrapped(C_DIM, done ? "Mario Kart World PC was removed. Thanks for racing!" : g_job.get_current().c_str());
            ImGui::PopFont();
            footer();
            if (done && (accent_button("Close", bsz) || getenv("THWOMP_SETUP_AUTO"))) {
                delete_self_later(exe_dir);
                quit = true;
            }
            break;
        }
        }
        ImGui::Unindent(x0);
        // the version, lower right
        if (!pl.version.empty()) {
            ImGui::PushFont(U.small);
            char v[96];
            snprintf(v, sizeof v, "MKW-THWOMP %s", pl.version.c_str());
            ImVec2 vs = ImGui::CalcTextSize(v);
            ImGui::GetWindowDrawList()->AddText(ImVec2(W - vs.x - 40 * S, H - vs.y - 34 * S), C_DIM, v);
            ImGui::PopFont();
        }
        ImGui::PopFont();
        ImGui::End();
        ImGui::Render();
        SDL_SetRenderDrawColor(U.ren, 14, 14, 17, 255);
        SDL_RenderClear(U.ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), U.ren);
        if (shot_frame && ++frame >= shot_frame) {
            SDL_Surface *s = SDL_RenderReadPixels(U.ren, nullptr);
            const char *out = getenv("THWOMP_SETUP_SHOT_OUT");
            if (s) SDL_SavePNG(s, out && *out ? out : "setup.png"), SDL_DestroySurface(s);
            quit = true;
        }
        SDL_RenderPresent(U.ren);
        if (auto_run && page == P_DONE) quit = true;
    }
    if (worker.joinable()) worker.join();
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    if (U.icon) SDL_DestroyTexture(U.icon);
    SDL_DestroyRenderer(U.ren);
    SDL_DestroyWindow(U.win);
    SDL_Quit();
    return g_job.state == 3 ? 1 : 0;
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    int argc = 0;
    LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::string> args;
    std::vector<char *> argv;
    for (int i = 0; i < argc; i++) args.push_back(utf8(wargv[i]));
    for (auto &a : args) argv.push_back(a.data());
    LocalFree(wargv);
    return main(argc, argv.data());
}

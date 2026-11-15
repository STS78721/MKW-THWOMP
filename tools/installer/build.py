"""Builds MKW-THWOMP's Windows installer: build/installer/MKW-THWOMP-Setup.exe, the player's folder inside it.

What: (1) SDL3 as a static library, once (build/win/sdl-static, from the source tools/build/windeps.py fetched), so the
setup is one file that needs no DLL next to it; (2) tools/installer/installer.cpp with Dear ImGui and its SDL3 /
SDL_Renderer back ends (runtime/third_party/imgui), zstd and the MKW icon, with llvm-mingw (build/win/llvm-mingw);
(3) the payload: every file of build/dist/windows (tools/build/dist.py) compressed with zstd, appended after the
program in installer.cpp's THWOMPV1 format, and the footer (payload offset, magic).
Why: one setup in THWOMP's own look instead of a zip (installer.cpp says what it does).
How: after `uv run tools/build/windeps.py`, `uv run tools/build/build.py --dist --runtime-only --target windows` and
`uv run tools/build/dist.py --target windows`: `uv run tools/installer/build.py [--level 19]` (Python 3.14: zstd is in
its standard library).
"""

import argparse
import struct
import subprocess
import sys
import tarfile
import zipfile
from compression import zstd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIN = ROOT / "build/win"
OUT = ROOT / "build/installer"
DIST = ROOT / "build/dist/windows"
EXE = ".exe" if sys.platform == "win32" else ""
CC = WIN / "llvm-mingw/bin/x86_64-w64-mingw32-clang++"
MAGIC = b"THWOMPV1"


def run(cmd: list[str], **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def sdl_static() -> Path:
    """SDL3 as libSDL3.a with its headers, built once with llvm-mingw (no libusb: the setup reads no controller)."""
    prefix = WIN / "sdl-static"
    if (prefix / "lib/libSDL3.a").exists():
        return prefix
    src = sorted((WIN / "dl").glob("SDL3-*.tar.gz"))[-1]
    work = WIN / "sdl-static-work"
    with tarfile.open(src) as t:
        t.extractall(work, filter="tar")
    (top,) = [p for p in work.iterdir() if p.is_dir() and p.name.startswith("SDL3-")]
    cc = WIN / "llvm-mingw/bin/x86_64-w64-mingw32-"
    fwd = lambda p: str(p).replace("\\", "/")  # noqa: E731  (CMake: no backslashes)
    run(["cmake", "-S", top, "-B", work / "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_SYSTEM_NAME=Windows", "-DCMAKE_SYSTEM_PROCESSOR=x86_64", f"-DCMAKE_C_COMPILER={fwd(cc)}clang{EXE}",
         f"-DCMAKE_CXX_COMPILER={fwd(cc)}clang++{EXE}", f"-DCMAKE_RC_COMPILER={fwd(cc)}windres{EXE}",
         f"-DCMAKE_INSTALL_PREFIX={fwd(prefix)}", "-DSDL_STATIC=ON", "-DSDL_SHARED=OFF", "-DSDL_TEST_LIBRARY=OFF",
         "-DSDL_TESTS=OFF", "-DSDL_EXAMPLES=OFF", "-DSDL_HIDAPI_LIBUSB=OFF"], stdout=subprocess.DEVNULL)  # fmt: skip
    run(["cmake", "--build", work / "build", "--target", "install"], stdout=subprocess.DEVNULL)
    return prefix


def compile_stub(sdl: Path) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    imgui = ROOT / "runtime/third_party/imgui"
    rc = OUT / "setup.rc"
    ico = OUT / "setup.ico"
    run([sys.executable, ROOT / "tools/build/icon.py", ico])
    rc.write_text('1 ICON "setup.ico"\n1 24 "setup.manifest"\n')
    (OUT / "setup.manifest").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n'
        '  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3"><security><requestedPrivileges>'
        '<requestedExecutionLevel level="asInvoker" uiAccess="false"/></requestedPrivileges></security></trustInfo>\n'
        '  <application xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings>'
        '<dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>'
        '<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>'
        '</windowsSettings></application>\n'
        '  <dependency><dependentAssembly><assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" '
        'version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>'
        '</dependentAssembly></dependency>\n</assembly>\n')
    res = OUT / "setup.res.o"
    run([str(WIN / f"llvm-mingw/bin/x86_64-w64-mingw32-windres{EXE}"), "-O", "coff", "-I", OUT, rc, "-o", res])
    stub = OUT / "setup-stub.exe"
    srcs = [ROOT / "tools/installer/installer.cpp", imgui / "imgui.cpp", imgui / "imgui_draw.cpp",
            imgui / "imgui_tables.cpp", imgui / "imgui_widgets.cpp", imgui / "backends/imgui_impl_sdl3.cpp",
            imgui / "backends/imgui_impl_sdlrenderer3.cpp"]  # fmt: skip
    deps = WIN / "deps"
    run([f"{CC}{EXE}", "-std=c++20", "-O2", "-mwindows", "-static", "-DUNICODE",
         f"-I{sdl / 'include'}", f"-I{imgui}", f"-I{deps / 'include'}", *srcs, res, "-o", stub,
         f"-L{sdl / 'lib'}", "-lSDL3", f"{deps / 'lib/libzstd.a'}", "-lsetupapi", "-lwinmm", "-limm32", "-lversion",
         "-lole32", "-loleaut32", "-luuid", "-lgdi32", "-luser32", "-lshell32", "-ladvapi32", "-lcfgmgr32"])  # fmt: skip
    return stub


def version() -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "dev"


def pack(stub: Path, level: int, ver_s: str) -> Path:
    if not (DIST / f"mkw{EXE or '.exe'}").exists() and not (DIST / "mkw.exe").exists():
        sys.exit(f"{DIST}: no player's folder; run tools/build/dist.py --target windows first")
    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    out = OUT / "MKW-THWOMP-Setup.exe"
    body = stub.read_bytes()
    ver = (ver_s or version()).encode()
    with out.open("wb") as f:
        f.write(body)
        offset = f.tell()
        f.write(MAGIC + struct.pack("<IQI", len(files), total, len(ver)) + ver)
        for i, p in enumerate(files):
            rel = p.relative_to(DIST).as_posix().encode()
            data = zstd.compress(p.read_bytes(), level=level)
            f.write(struct.pack("<H", len(rel)) + rel + struct.pack("<QQ", p.stat().st_size, len(data)))
            f.write(data)
            if i % 200 == 0:
                print(f"  {i}/{len(files)} {rel.decode()}", flush=True)
        f.write(struct.pack("<Q", offset) + MAGIC)
    print(f"{out}: {len(files)} files, {total / 2**20:.0f} MiB in {out.stat().st_size / 2**20:.0f} MiB, version {ver.decode()}")
    return out


def release_zips(ver: str) -> None:
    """The release's two zips in STS78721's naming: SteelTemple-MKW-THWOMP-<ver>-Portable-StandAlone.zip (a new install:
    extract anywhere) and -Update.zip (over an install; the start screen's updater, tools/prepare/update.py, takes this
    one). Both hold the player's folder under one MKW-THWOMP/ folder, as the updater expects."""
    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    for kind in ("Portable-StandAlone", "Update"):
        out = OUT / f"SteelTemple-MKW-THWOMP-{ver}-{kind}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                z.write(f, "MKW-THWOMP/" + f.relative_to(DIST).as_posix())
        print(f"{out}: {out.stat().st_size / 2**20:.0f} MiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, default=19, help="zstd level of the payload (default 19)")
    ap.add_argument("--zips", action="store_true", help="also the release zips (Portable-StandAlone, Update)")
    ap.add_argument("--version", default="", help="the release shown by the setup and in Windows' apps (default: git describe)")
    a = ap.parse_args()
    setup = pack(compile_stub(sdl_static()), a.level, a.version)
    if a.version:   # the release's name for it: SteelTemple-MKW-THWOMP-<ver>-Full-Setup-Wizard.exe
        named = OUT / f"SteelTemple-MKW-THWOMP-{a.version}-Full-Setup-Wizard.exe"
        named.write_bytes(setup.read_bytes())
        print(named)
    if a.zips:
        release_zips(a.version or version())


if __name__ == "__main__":
    main()

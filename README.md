# MKW-THWOMP: Mario Kart World PC

A static recompilation of **Mario Kart World** (Nintendo Switch 2, update 1.8.0) to native Windows with Vulkan.

**No game files are included.** You need your own legally dumped copy of the game (exefs and romfs of the 1.8.0
update). This repository holds only the recompiler, the runtime and the build tools.

> Status: work in progress. The intro, title screen and menus render and races load; graphical issues remain
> (see *Known issues*). Not yet fully playable.

## What it is

- `recomp/`: lifts the game's ARM64 code to C (`gen.py`).
- `runtime/`: the host side: the NVN 2 graphics API on Vulkan, the Switch 2 services, input, audio, the file system
  and the in-game settings menu.
- `tools/`: build, packing and diagnostic scripts.

## Build

Requirements: Windows 10/11, [MSYS2](https://www.msys2.org/) UCRT64 (clang, ninja, Vulkan headers, SDL3, Opus),
Python 3.12, a Vulkan 1.3 GPU.

```bat
set MKW_GAME=D:\path\to\your\mkw-1.8.0-dump
BUILD-MKW.bat
RUN-MKW.bat
```

Or build the runtime directly into a folder of your choice:

```bat
python tools\build_game.py --version mkw180 --out build\mkw
```

## Settings

Press **F1** or **Esc** (Minus + Plus on a controller) in game for the settings menu: Display, Sound, Controls,
System and Advanced, in a light (Switch 2 style) or dark theme. Settings are saved to `mkw.ini` next to `mkw.exe`.
Any setting can also be set as an environment variable with the `MKW_` prefix (for example `MKW_MEMORY=import`).

The game renders at its own resolution, as on the console. There are no upscalers or frame generation.

## Known issues

- Foliage (trees, bushes, cacti) pops in and flickers between levels of detail.
- The terrain arrays are reduced to 1024² on the host (the full 8192² × 68 layers does not fit in VRAM).
- The `mirror` GPU memory mode can corrupt game memory (crashes, flicker); `import` is the default.

## Credits

Built on the open-source recompiler runtime lineage (TotK Recomp and its contributors), Eden / Eglantine
(shader recompiler), Dear ImGui, Ryujinx and others. See `THIRD_PARTY_NOTICES.md` and the licence files.

Mario Kart World is a trademark of Nintendo. This project is not affiliated with or endorsed by Nintendo.

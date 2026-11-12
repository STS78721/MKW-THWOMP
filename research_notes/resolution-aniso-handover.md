# Resolution (1080p / 1440p / 4K) and anisotropic filtering — handover

## Codex continuation, 2026-11-12

- New visual report: `C:/Users/debee/Pictures/Screenshots/Screenshot 2026-11-12 133240.png` shows faint yellow/orange light halos over opaque foreground foliage and the stone wall; user identifies them as distant in-game lights showing through the world. The visible artifact is glow/visibility leakage. A depth/occlusion problem is a hypothesis, not a confirmed cause. The user cannot reproduce the view right now, and the existing `dump-f12/passes.txt` predates the screenshot. A future F12 capture of an affected view is needed to identify the exact draw, its depth state and sampled depth resources. Relevant paths: `vk_exec_draw.c` light quads / texture binding / depth twins, `vk_exec_xfer.c` occlusion counts and conditional-render fallback. Do not hide lights or weaken fidelity as a workaround. Screenshot overlay reads 25.3 fps / 39.5 ms; it is not an A/B benchmark.
- Current priority: intermittent track-transition loading loop (animation keeps moving; restarting then entering the same area works). User confirms it predates the optimization pass. Preserved the reported run as `build/mkw/loading-loop-2026-11-12.log`; the last 20.6 seconds present at 60 fps without a GPU fault. No live hung process was available.
- Found and corrected an existing kernel bug: SVC 0x34 arbitration type 3 is `WaitIfEqual64`, but `kernel.c` compared only the low 32 bits. The SDK has a full-width call at `sdk+0x16eda0` / `sdk+0x16edc0` in `generated/mkw180/sdk/sdk_0010.c`. Verified semantics against https://github.com/switchbrew/libnx/blob/master/nx/include/switch/kernel/svc.h (ArbitrationType and svcWaitForAddress). Regression checks for an upper-word-only change fail twice before the fix (`build/mkw/loading-wait-before-fix.log`) and pass after it, along with the full `--probe-kernel` (`loading-wait-after-fix.log`). The build passes (`loading-fix-build.log`). This is a real synchronization correction and a plausible cause, but the reported track-transition hang is not yet reproduced/confirmed fixed.
- Added Ctrl+F9 thread/wait capture in `gfx_window_win.c` / `gfx_window_sdl.c`, backed by `kernel.c`: three snapshots over four seconds into `build/mkw/logs/loading-waits-*.log`. Captures kernel-blocked guest stacks, wait addresses/values, synchronization handles, and virtual-core/host-call metadata; it does not wake or alter waits. Snapshot probe cases verify mutex, condition, address and sync wait state is preserved. If the loop recurs, capture while it is still hung.
- After the loading issue, user wants steady 60 fps at full fidelity, then investigation of unlocked / 120 fps modes with correct simulation and animation timing. Resolution remains unfinished.
- The user changed priority to optimizations before completing resolution. The first fidelity-preserving CPU pass is built, uncommitted: `nvn_api_cmd.c` records immutable wait/deferred-work flags while validating recordings, `vk_memory.c` caches exact descriptor-key queries until pool mappings/lifetimes change, and `vk_exec_draw.c` / `vk_pipeline.c` emit changed dynamic state and the changed vertex-binding range. `vk.h` declares the delta-state entry point. No graphical settings or render passes were reduced.
- Optimization validation: runtime-only 52-step build passed (`build/mkw/optimization-build.log`); `--probe-nvn` passed all 1,054 checks on the null backend (`optimization-nvn.log`), including deferred snapshots and inlined waits. The probe also fixes a pre-existing coverage failure for three newer op kinds. `tools/check_draw_state.py` passed 524,288 dynamic-state transitions and 65,536 vertex-binding transitions with mocked Vulkan commands; `tools/test_descriptor_key_cache.py` passed 304,568 checks including remaps and overlapping pools. Its synthetic four-million-lookup workload was 49.87 ms uncached vs 6.79 ms cached. These are CPU-only checks and synthetic measurements, not an in-game FPS result. GPU/game runs remain held after the timeout.
- `runtime/boot.c` now uses the presented window crop for `--shot` and series captures; `runtime/nvn_api_obj.c` passes that crop through and no longer substitutes 1704x960 when it is empty. The empty crop follows NVN's whole-image default. `runtime/gfx.h` carries the crop to the capture callback.
- `runtime/resolution.c` no longer logs the old Zelda 1920x1080/1600x900 sizes at startup. The performance overlay already uses the actual presented image size (`MenuTarget.image_w/image_h` in `runtime/ui/menu.cpp`).
- Build: `ninja -f C:/Games/MKW PC/build/mkw/build.ninja` succeeded. One 2160-line headless run exited 0: crop was 2560x1440 and `--shot` produced a 2560x1440 PNG at `build/mkw/dump-claude/codex-res-crop/final.png`. The 2/3 reduction remains.
- Found a writer of the crop rectangle in 1.8.0 in the `main+0x47193c` block (`generated/mkw180/main/main_0053.c` near line 32003): it sets the dirty byte at `obj+0xab`, clears `obj+0xac..0xb3`, then stores width and height at `obj+0xb4..0xbb`. The object comes from the global at `main+0x36d3a48`, then `+0x80`. The values were already packed in `x10` earlier in `f_7000471700`; this writer propagates a texture size, so the earlier producer still needs tracing.
- `f_7000471700` selects one of the 0xa8-byte entries at its object `+0xd0`, using the byte at `+0x25f`. On one path it reads the packed width/height at selected entry `+0x44`; on another it computes a packed size from entry floats and stores it at object `+0x10`. The code compares that computed size with known modes through 3840x2160 (`main+0x4719f4`). The table's initialization and the active path still need identifying; changing only the NVN crop would not change the scene render size.
- The shipped 1.8.0 GParam values are 1080 for Low and 1440 for Middle, except `ResolutionLowFor3DForDiv=900`. The current overlay changes all twelve fields to the same line count. The reference port's 1.0.0 measurement was ~65% for the **3D scene**; this work's 2/3 measurement is for the **final crop** on 1.8.0, so they are distinct observations.
- A separate 16x anisotropy headless run (`build/mkw/dump-claude/codex-aniso16/host.log`) logged the 16x setting, then ended with `VK_ERROR_DEVICE_LOST` / GPU fault at an invalid instruction pointer during an upload (`exit 70`). Cause is unknown. No further game runs were made after the GPU timeout; `mkw.exe` is not running. Treat 16x as unvalidated in game play.
- The headless run forced a windowed background window and reported a 1920x1080 swapchain. Static code shows `WM_SIZE` marks the swapchain dirty and `gfx_present.c` recreates it, but fullscreen 4K has not been observed.

Date: 2026-11-12. Status: **in progress, uncommitted, builds clean**. Nothing has been tested by the user yet.

## Goal (user request)

1. A real higher game resolution: 1080p, 1440p and 4K.
2. A Graphics settings tab with anisotropic filtering, up to 16x.

## What is done (uncommitted)

| File | Change |
|---|---|
| `runtime/resolution.c` | `res_mkw_lines()` (reads `MKW_RESOLUTION`) and `res_gparam_build()`: unpacks `System/GymlArchive/general_format.Product_180.sarc.zs`, sets every `*For3D*` / `*Screen*` int in `Gyml/GamePerformanceGraphicsConfigGParam/30_Extendted...bgyml` to the chosen lines, packs it again. Diagnostic knob `MKW_GAME_UPSCALE=0/1` sets `PostProcess.IsEnableUpscale` (it did nothing, see below). |
| `runtime/romfs.c` | `gparam_overlay()` (modelled on `heapdef_overlay`): writes the rebuilt archive to `<profile>/resolution/` and serves it in place of the original. Called from `overlay_init`. |
| `runtime/host.h` | prototypes for the two functions above. |
| `runtime/vk_texture.c` `make_sampler` | `MKW_ANISO` (`game|2|4|8|16`): minimum anisotropy on mipmapped linear-min samplers (idea from adriadam10/mkw-recomp `vk_tex_slots.c`, MIT). Takes effect at the next start. |
| `runtime/cfg.c` | known keys `RESOLUTION`, `ANISO`; metas `RESOLUTION original|720|900|1080|1440|2160`, `ANISO game|2|4|8|16` (both RESTART); section `graphics`. |
| `runtime/ui_text.h`, `ui_text.c` | `UI_MENU_TAB_GRAPHICS`, `UI_MENU_RESOLUTION(_DESC/_ORIGINAL)`, `UI_MENU_ANISO(_DESC/_GAME)`; English and Russian. |
| `runtime/ui/menu.cpp` | new **Graphics** tab (`TAB_GRAPHICS`, second tab) with the Resolution and Anisotropic filtering rows. |
| `runtime/nvn_api_obj.c` `nvnWindowSetCrop` | logs the first 12 crop changes with caller `lr` and present count (diagnostic; keep or trim). |

Anisotropic filtering should be complete. **Resolution is not**: see below.

## What the experiments showed (headless runs, `build/mkw/dump-claude/<tag>/host.log`)

- The game draws its **whole frame (3D and HUD)** into the top-left of a **3840x2160** window texture and sets
  `nvnWindowSetCrop` to the part it used. Stock: **1704/1706x960**. That is what the player sees, upscaled.
- `MKW_RESOLUTION=2160` (all 12 fields 2160): crop goes `3840x2160` then **`2560x1440`**. Render passes 2560x1440.
- `MKW_RESOLUTION=1440`: crop `2560x1440` then **`1706x960`** (same as stock).
- So the final frame is **2/3 of the field lines**. The first crop is the full field size, then it drops to 2/3.
- Both crop calls happen at present count 0 (setup time), from **lr `main+0x69d124`** in `f_700069cfa0`
  (`generated/mkw180/main/main_0073.c` ~line 7098): it calls a function pointer from `[0x70036e05b8]` with x1..x4
  read from `[obj+0xac..0xb8]` (x, y, w, h) of the object at `[x19+0x150]`. **Next step: find who writes
  `obj+0xac..0xb8` (the 2/3 factor) and why it drops from full size.**
- Not the cause: `MKW_GPU_TIME=zero` (GPU timestamps pinned) — still drops; `IsEnableUpscale=0` — no change.
- `GraphicsSystem` setup `main+0xb879f0` (1.8.0) reads the 12 GParam fields via getters and **clamps each to 2160
  (0x870)**, widths = h*16/9. So field lines > 2160 are impossible without a code patch.
- The 1.0.0 reference port's output-size site (`mov x10,#0xa00; movk x10,#0x5a0,lsl#32`) exists in 1.8.0 at
  `main+0xb87b9c` / `main+0xb87ba8`, but in 1.8.0 those defaults are overwritten by the GParam values right after.

### Consequence for the menu values

If the 2/3 cannot be removed, the mapping would be field = target × 3/2: 720→1080, 900→1350, 1080→1620,
1440→2160, and **4K would need the 0x870 clamp lifted** (12 csel sites in `f_7000b879f0`; patch set via
`recomp/patches-mkw180.txt` + regenerating the lifted C, a heavy rebuild — ask the user first). Better: find and
remove the 2/3 factor, then field = target lines directly (2160 = real 4K, no clamp change needed).

`res_gparam_build` currently writes the target lines straight into the fields (so "1440p" today gives 960p, "2160p"
gives 1440p). Fix the mapping once the 2/3 is understood.

## Other notes

- Screenshot capture and present now use the window's real crop; `--shot` was verified at 2560x1440 for the 2160-line setting.
- The old TotK startup resolution log was replaced. The overlay's picture-size values already came from the actual presented image (`MenuTarget.image_w/image_h`), not `res_sizes()`.
- The user's display is 3840x2160 at DPI 216. The log showed `swapchain 1920x1080` at start (created before the
  fullscreen switch); check that the swapchain is recreated at 3840x2160 in fullscreen.
- House rules: one light run at a time, ask before heavy work, MKW naming only, ask before commit/push
  (commit messages end with `Co-Authored-By`), never build into `build/mkw180`.

## Build / run

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
Set-Location "C:\Games\MKW PC\mkw-recomp"
& "C:\msys64\ucrt64\bin\ninja.exe" -f "C:/Games/MKW PC/build/mkw/build.ninja"
# headless (kills running mkw.exe first!)
$env:MKW_RESOLUTION='2160'; $env:MKW_SEED='1'
& 'C:\Games\MKW PC\build\mkw180\dump-claude\run_headless.ps1' -Tag t1 -Seconds 30 -Shots '0' -Dump '0' -Dir 'C:\Games\MKW PC\build\mkw'
Select-String -Path 'C:\Games\MKW PC\build\mkw\dump-claude\t1\host.log' -Pattern 'window crop|resolution'
```

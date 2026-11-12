# MKW-THWOMP handover (for Gemini Carbon)

Date: 2026-11-11. Written by Claude (Opus 5.5) at the end of a long debugging session. Everything below was checked
in this session unless it says otherwise.

## 1. What the project is

A static recompilation of **Mario Kart World** (Nintendo Switch 2, update 1.8.0) to native Windows with Vulkan.

- `recomp/` lifts the game's ARM64 code to C.
- `runtime/` is the host side: NVN 2 on Vulkan, Switch 2 services, input, audio, file system and the F1/Esc menu.
- `tools/` holds build, packing and diagnostic scripts.

The game boots, the menus work, and races load and play at about 38–40 fps on the user's NVIDIA PC. The main open
bug is described in section 5.

## 2. Where everything is

| What | Path |
|---|---|
| Source (git repo) | `C:\Games\MKW PC\mkw-recomp` |
| Lifted C code (generated) | `C:\Games\MKW PC\generated\mkw180` (`main/`, `sdk/`, `subsdk0/1/`) |
| Game files (user's dump) | `C:\Games\MKW PC\exefs`, `C:\Games\MKW PC\romfs` |
| **Working build** (exe, ini, logs) | `C:\Games\MKW PC\build\mkw` (`mkw.exe`, `mkw.ini`, `logs\mkw.log`) |
| User's F12 frame captures | `C:\Games\MKW PC\build\mkw\dump-f12` (previous one: `dump-f12.old`) |
| Headless run outputs | `C:\Games\MKW PC\build\mkw\dump-claude\<tag>\` (`host.log`, `f*.png`, `dump\`) |
| Headless runner script | `C:\Games\MKW PC\build\mkw180\dump-claude\run_headless.ps1` |
| Research notes | `mkw-recomp\research_notes\foliage-popin-plan.md` (findings log, keep it updated) |
| Backup of the old GitHub history | `C:\Games\MKW PC\backups\MKW-THWOMP-before-override-2026-11-10.bundle` |
| Original DK Bananza recomp (do not touch) | `C:\Games\DKB PC\dkb-BANANAS` |

`C:\Games\MKW PC\build\mkw180` is shared with another project that overwrites it. Do not build into it.

## 3. Build and run

**Incremental build** (about a minute; run it from the repo folder, because the ninja file uses relative paths):

```bash
cd "/c/Games/MKW PC/mkw-recomp"
PATH="/c/msys64/ucrt64/bin:$PATH" ninja -f "C:/Games/MKW PC/build/mkw/build.ninja"
```

**Full build:** `python tools/build_game.py --version mkw180 --out "C:\Games\MKW PC\build\mkw" --jobs 6` with the
system Python 3.12. Do not use `python -I`: that hides the build modules.

**Headless run** (PowerShell): hidden window, no input, title-screen attract mode, which drives a race on a random
course.

```powershell
& 'C:\Games\MKW PC\build\mkw180\dump-claude\run_headless.ps1' -Tag t1 -Seconds 30 -Shots '600-900:150' -Dir 'C:\Games\MKW PC\build\mkw'
```

- `-Shots 0` means no screenshots. `-Dump 400` captures frame 400 (`dump\` subfolder).
- Set any knob first with `$env:MKW_X = '...'`, and remove it afterwards.
- `MKW_SEED=1` makes the attract course the same every run (it fixes GetInfo random entropy).
- **The script kills every running `mkw.exe` before it starts.** Make sure the user is not playing.

**Settings:** every `MKW_X` (also `TOTK_X`, `DKB_X`) environment knob can be written in `mkw.ini` as `x = value`.
A key missing from the known list in `runtime/cfg.c` still works but logs "unknown key"; add new keys to that list.

### Hard rules from the user

1. **One light run at a time, at most about 20 screenshots, never batches.** Four back-to-back heavy runs once froze
   the user's PC and forced a hard reset while they were busy. They were furious. Ask before anything heavy.
2. Everything must say MKW / Mario Kart World, never DKB or TotK, in user-facing text.
3. Debug with evidence (captures, logs, traces), not guesses. Grok spent a night guessing and broke things.
4. Before saying a fix "doesn't work", check the user actually ran the new build. Compare `mkw.exe`'s time with
   `logs\mkw.log`'s. This happened twice in this session.
5. The user tests races by hand. Headless runs never reach Mario Circuit; the attract mode picks other courses.

## 4. Git state

- Remote `origin` = https://github.com/STS78721/MKW-THWOMP. Repo-local identity: `STS78721 <134802927+STS78721@users.noreply.github.com>`.
- Pushed: `5e0f19d`, `612e8e0`, `8e83a67`. **Local, not pushed:** `a8f683d` (heap fix).
- **Uncommitted** (all from this session, all built into the current `mkw.exe`):

  | File | Change |
  |---|---|
  | `runtime/vk_pipeline.c` | `spirv_base_instance`, the instance-id / base-instance SPIR-V rewrite |
  | `runtime/vk_exec_draw.c` | compute group counts in c0; `MKW_DUMP_INST` trace; drawstats instance/pending hooks |
  | `runtime/vk_dump.c` | `MKW_DUMP_PICK` depth-grid tracer, `vk_dump_active()` |
  | `runtime/vk_drawstats.c` | F9 trigger (`key:N`), instance counts, `dpend` lines, small-pass target addresses |
  | `runtime/gfx_window_win.c` | F9 key |
  | `runtime/vk_exec.c`, `vk_exec.h` | hook for the instance trace after a pass |
  | `runtime/cfg.c` | new known keys |
  | `research_notes/foliage-popin-plan.md` | findings |

- The user asks before every commit and push. Commit messages end with a `Co-Authored-By` line.
- Licence note: the user received a review of this repo from someone working on another MKW port ("MKWRecomp"),
  which is GPL-3. Take facts from it, never code.

## 5. Phantom / exploding trees and blinking foliage: FIXED 2026-11-12

**Root cause:** the shader translator's AST optimizer copied a warp `ballot(true)` into a one-lane branch of MKW's
instance-compaction compute shader (`104bd69caf1f4c09`). Each warp's instance offset became 0, so different object
types' instance ranges overlapped. **Fix:** translator patch `0027-keep-warp-operations-in-place.patch`, confirmed
by the user on Mario Circuit. Details: `research_notes/foliage-popin-plan.md`, last section. The history below is
kept for reference. The base-instance rewrite was also needed and stays.

### History of the investigation

### Symptom (Mario Circuit, user screenshots)

- Huge, dark tree trunks with leaf cut-outs stand where no tree belongs.
- Bushes turn into trees, and trees grow and change shape as you drive.
- Some distant trees stay half-dithered (a dotted LOD fade).
- Earlier the user called it "foliage pops in and out" and "very low polygons".

### What a capture proved

- The phantom is real geometry in the main depth prepass (1704×960), not a shadow. Depth shows it, and colour shows it
  as a bark-textured spike.
- With `MKW_DUMP_PICK=1` the depth grid named the draw: **vertex shader NVN 2 origin `1cdcc0fd9dee5df9`** (SPIR-V
  hash `5eeccb876afa784b`), fragment `16baadf04c78e4e0`, 12 draws per pass. Program serials change every run; the
  origin is stable.
- In that shader, each instance reads an **80-byte record** (a 3×4 transform plus a phase word) from the **vertex
  storage slot 5** buffer (address at c0+0x160, about 0x2a7061000, size 0x493e00 ≈ 4.8 MB) at `base + instance×80`.
  Wind sway (`vp_c9[988]`, `vp_c11[21]`) and a distance bend (`vp_c9[42]`) are applied on top.
- The instance number is `InstanceId + c0[0x644]` (c0[100].y).
- These draws are **indirect**: `vkCmdDrawIndexedIndirect` with 20-byte GPU-written commands, whose first instance is
  non-zero (1, 6, ... in a headless capture). Compute passes earlier in the frame also bind that slot-5 buffer, so the
  game most likely builds the instance lists and/or records on the GPU (culling / LOD).
- So a phantom = a tree mesh drawn with **another object's record**, from either a wrong instance index or a wrong
  record.

### Ruled out (with evidence)

| Idea | Test | Result |
|---|---|---|
| Game sheds detail because "GPU time" looks high | `gpu_time = zero` | still flickers |
| Old Zelda-port "light quad" draw skip (`FIX_VS_FAULT`) | `fix_vs_fault = 0` | still flickers |
| Mesh-shader foliage path | `mesh_discard = 1` | phantoms still there |
| CPU/GPU race on the instance buffer | `submit_wait = 1` (every submit waits) | phantoms still there |
| Deferred buffer binds read at the wrong time | MKW never calls them | n/a |
| Virtual-pool remap race | `vk_be_pool_map_virtual` drains the replay first | n/a |
| The flipping 128×128 pass | it is a reflection probe (one cube face per frame) | harmless |

### Fixes already in for it (uncommitted, built at 02:37, NOT yet tested by the user on Mario Circuit)

1. **Base instance from Vulkan's BaseInstance builtin** (`vk_pipeline.c` `spirv_base_instance`, final version
   2026-11-12). Every vertex-stage load of c0+0x644 (vp_c0 member 0 [100].y) becomes a load of `BuiltIn
   BaseInstance`. The input variable is synthesised when the module lacks it; 3 shaders read the base directly as
   `(16 + base) << 4`. The instance id stays zero-based (`InstanceIndex − BaseInstance`).

   Reason: the replay only fills c0+0x644 for direct draws (0 for indirect). An independent MIT port
   (github.com/adriadam10/mkw-recomp, `sm86/translate/glsl/core.py` and `mem.py`) found the same cause ("trees out of
   place and black") and confirms that NVN's instance id does not include the base. A short-lived variant that added
   the base to the instance id (from a GPL review) was reverted. Validated: all 28 base-instance vertex shaders in one
   capture and 26 in the other are rewritten, and all pass `spirv-val`; headless runs are clean.
   `MKW_BASE_INSTANCE=c0` turns it off. Watch out: the F12 capture's `sh_*.spv` files are the shaders **before** the
   rewrite.
2. **Compute group counts** at c0+0x0..0xb (`vk_c0_fill`, NVN 2 compute branch). Before this they were always 0.
   Indirect dispatches still get 0 (the size is GPU-written; it needs a GPU copy into c0).

The user's last "still broken" report ran an older build (log 02:26, exe 02:37). **First step: get one test on the
current build, with F12 on a phantom.**

### Where to look next, in order

1. **Read the F12 instance trace.** With the current build, pressing F12 logs `dinst` lines to `logs\mkw.log` for
   every draw of shader `1cdcc0fd9dee5df9` (override with `MKW_DUMP_INST=<origin hex>`). Each line pair shows the GPU
   snapshot, taken after the pass, of the draw command (`indices / instances / first index / vertex offset / first
   instance`) and the records those instances read (`pos`, `scale`, `phase`).
   - Absurd scale or position in a record → the record data is wrong → go to 3.
   - Plausible records but a first instance or count that points at another object's range → the command is wrong → go to 2.
2. **The indirect-reset workaround** (`vk_exec_draw.c` `indirect_reset_draw` → `vk_exec_reset_indirect`, knob
   `FIX_INDIRECT_RESET`, default on for MKW).
   - MKW resets its indirect commands each frame with a point draw: VS `e78abde51f2d6bfa`, FS `57b0d7b4959f03c0`,
     which stores through a pointer list. Grok replaced it with a copy of a fixed record `{0, 1, 0, 0, 0}` into every
     command.
   - The headless snapshot shows the game's commands keep their index counts and vertex offsets with the instance
     count at 0. So the real reset probably clears only the instance count, while Grok's version also zeroes the
     index count and first instance.
   - The code credits Eglantine, but **Eglantine's public source has no such reset**, so nothing confirms it.
   - Next: find what the real reset shader writes. Translate VS `e78abde51f2d6bfa` and read its stores, or snapshot
     the records before and after with `TOTK_DUMP_BUF`. Then make the replacement write only those fields, or test
     `fix_indirect_reset = 0` (the shader path; Grok noted it "loses stores through pointers loaded from a list",
     which is a translator bug worth fixing properly).
3. **The culling compute that writes the slot-5 records and the indirect commands.** Find the dispatches in
   `passes.txt` that bind slot-5 buffer 0x2a7061000 (and the indirect-command pool) as written. Check those programs
   for:
   - the indirect-dispatch group counts (still 0, see fix 2);
   - IABS (see section 7);
   - subgroup or atomic operations;
   - global stores through loaded pointers (the same "lost stores" class as item 2).
4. The **dotted LOD fade** on distant trees: check that the temporal AA history and motion-vector targets keep their
   contents across frames. Look for texture re-creation or "released under VRAM pressure".

## 6. What was fixed in this session (root causes)

| Problem | Root cause | Fix (file) |
|---|---|---|
| Black roads and ground | BC3 staging textures were raw-copied into RGBA8 terrain arrays (`vkCmdCopyImage`) | blit instead (`vk_exec_xfer.c` `copy_decoded_astc`, `blit_src_ok`) |
| Black cacti silhouettes | fragment shaders need Maxwell's FMZ (0·x = 0); the dither fade computed NaN | FMZ rewrite on all stages by default (`vk_pipeline.c`, `MKW_FMZ`) |
| Stale mesh-task counts | `DrawMeshTasksIndirect` read its count on the CPU at record time | GPU count path (`vk_mesh.c`; `MKW_MESH_CPU_COUNT=1` = old way) |
| Heap starvation, a 6.8 GB fallback allocator corrupting a live heap, crashes | the loader bound a module's own symbols first; the SDK's sized `operator delete` reached the SDK's weak `_ZdlPv` (nmalloc free) instead of the game's heap-aware one, so game-heap blocks leaked (`NotNMallocPtrError`) | rtld-style load-order lookup (`loader.c` `resolve`); Grok's heap hacks now opt-in with `MKW_HEAP_LEGACY=1` |
| Phantom trees (partial, untested) | base instance / instance id on indirect draws | see section 5 |
| Compute group counts read 0 | never written to c0 | `vk_c0_fill` |

Also done: the XeLL / DLSS / FSR / frame-gen / Reflex / AntiLag code was removed. The new F1/Esc menu
(`runtime/ui/menu.cpp`, `theme.h`, `option_rows.cpp`) has Display, Sound, Controls, System and Advanced tabs in a
light (Switch 2 cyan) or dark (red) theme. The project was renamed to MKW, and the GitHub repo history was replaced.

Defaults changed: `VK_PRESENT=fifo`, `MEMORY=import` (mirror mode corrupts game memory: write-back of whole SSBO
ranges), rumble 100, `COMPACT_LOD_BIAS=0`, `FMZ=all`.

## 7. Diagnostic tools added (use them)

| Knob / key | What it does |
|---|---|
| `MKW_SEED=1` | the same attract course every run |
| `MKW_DRAWSTATS=<frame>:<n>` or `key:<n>` + **F9** | per-pass, per-program draw/instance counts, GPU indirect counts, conditional render, pipeline-pending skips (`dstat`/`ddiff`/`dcnt`/`dpend` log lines); summarise with `python tools/drawstats_report.py logs\mkw.log` |
| `MKW_SKIP_PROG=<serial>` | drop one program's draws |
| `dump = key` + **F12** (`dump_dir = dump-f12`) | one-frame capture: `passes.txt` (every pass, pipeline, binding, texture), `images.txt`, PNGs of targets, `sh_<hash>.spv` (pre-rewrite SPIR-V) |
| `MKW_DUMP_PICK=1` | during the capture, a 32×18 depth grid before each draw of big passes → `pick.txt`: which draw wrote depth where. That is how the phantom draw was found. |
| `MKW_DUMP_INST=<vs origin>` | during the capture, GPU snapshot of that shader's indirect commands and slot-5 records → `dinst` log lines |
| `MKW_HEAP_TRACE=1` | logs ExpHeap alloc / maxalloc / adjust / resize; the nmalloc failure log carries a guest backtrace |
| `MKW_GPU_TIME=zero`, `TOTK_SUBMIT_WAIT=1`, `mesh_discard=1`, `fix_vs_fault=0` | the isolation tests from section 5 |

Tools: `spirv-dis` and `spirv-cross` from `C:\VulkanSDK\1.4.363.0\Bin`; Python with PIL for contact sheets.
`recomp/nso.py` parses NSOs. To decode the dynamic symbols/imports of `exefs\main` / `sdk`, parse `MOD0 → DT_SYMTAB /
DT_STRTAB / DT_JMPREL` (that is how the `_ZdlPvm` / `_ZdlPv` binding bug was found).

## 8. Other open issues

From the outside review of an older state of this repo. Some claims were wrong for our code (we do have a real mesh
path), so verify each one first.

- **Terrain sparse textures.** CompactLayers shrinks the 8192²×68 ASTC arrays to 1024 (host mip0 = 3) without
  adjusting explicit LOD or texture size. Copies into the dropped mips are squeezed into level 0.
  `vk_texture_sparse_pages_mapped` may do nothing for GPU-only textures. Not checked this session.
- **IABS lowered to `max(x, -x)`** in the shader translator (Ryujinx-based `mkw_xlate`, patches in `tools/xlate`).
  The review says this triggers an NVIDIA device-lost bug. Not checked.
- **The crash handler** reportedly lost its "not our code" check, so driver-handled faults kill the game. Not checked.
- **ASTC 5×5 / 6×6 partial updates** off a 4×4 block corner are skipped. ASTC goes to BC3, not BC7. Not checked.
- **All guest loads and stores are `volatile`** (`runtime/cpu.h:82–87`): a speed cost, confirmed.
- **The application pool is 9 GiB** (`memory.c` `DEFAULT_POOL`); the review says 9809 MiB. Unverified on hardware.
- **`tools/build_game.py`** hardcodes `C:\Games\MKW PC` paths (lines ~812–815).
- **Seven mesh-shader programs** give the fragment stage fewer outputs than it reads (`vk_mesh: iface ... missing=0xf`;
  the synthesised primitive header never fires).
- **VRAM overload in long headless runs:** one died with heap 0 at 19.5 GB against a 14.7 GB budget while decoding
  ASTC (device lost).
- **Grok's other guest patches** in `runtime/ov_system.c`: the dispose vtable hook `__wrap_f_700087b9b0`, the
  near-null soft-skip VEH, and "invalid list node / null vtable, clearing" wraps. Some may have been heap-corruption
  side effects. Re-test whether they are still needed now that the heap fix is in.

## 9. Suggested order

1. The user runs Mario Circuit on the current build with F12 on a phantom. Read the `dinst` lines (section 5).
2. Fix the indirect reset or the culling compute, depending on what the trace shows.
3. Commit (ask first): the base-instance rewrite, the compute group counts and the diagnostic tools.
4. Indirect-dispatch group counts (GPU copy into c0).
5. The sparse terrain items and the IABS check from section 8.

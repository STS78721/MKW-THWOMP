# Plan: tree / foliage pop-in and flicker

Date: 2026-11-10. Symptom: trees, bushes and cacti blink in and out, swap between their near and far model from one
frame to the next, and sometimes stay half-dithered (a dotted tree); the user also calls the look "blown out".

## What is already known (evidence, not guesses)

| Finding | How it was shown |
|---|---|
| The blinking is in the game's own rendered frames | `MKW_SHOTS` captures the game's presented image before any host post-processing; an 8-frame series shows a pine tree swap near model / far model / nothing while the camera barely moves |
| Not frame generation | FG is removed; shots are taken before it anyway |
| Not a CPU / GPU timing race of the submit | `TOTK_SUBMIT_WAIT=1` (each submit waits for the GPU) still blinked (a road sign and a cactus) |
| Not a missing compute → indirect-draw barrier | `barrier_to` (vk_exec.c) puts ALL_COMMANDS / MEMORY_READ+WRITE before every pass |
| `mirror` memory mode corrupts game memory | ~half the mirror runs crashed in `f_70006d2800` on garbage vtables, 0 of 4 `import` runs did; write-back copies whole bound SSBO ranges (up to 1 MiB) over CPU data |
| Unknown: does `import` also stop the blinking? | Never measured in a comparable scene (the attract mode picks a random course) |

## Suspects, most likely first

1. **Mirror write-back clobbering per-object / LOD data** (now default off, but must be confirmed as the cause or not).
2. **NVN 2 task / mesh shader foliage** (`DrawMeshTasks`). MKW draws bushes, grass and many trees with task + mesh
   shaders (nxnative: 7 task + 18 mesh programs). Our path is Grok's unfinished work: synthesised primitive headers
   (`no MeshPrimitiveHeader`, `synth-prim-SPH`), ISBE stage memory approximations, outputs capped at 128 vertices,
   interface mismatches logged as `iface mesh_out=0x30 fs_in=0x3f missing=0xf`. A mesh shader that outputs nothing on
   some frames is exactly "blinks out".
3. **GPU-driven LOD / culling reading wrong inputs**: the compute dispatches before the main pass pick a LOD and an
   instance list per object from the camera, last frame's depth pyramid (Hi-Z) and a LOD fade value. A stale or
   cleared Hi-Z (render target re-created, aliasing) culls objects that are visible.
4. **LOD cross-fade without its temporal resolve**: MKW fades LODs with a dither pattern that its temporal pass
   (motion vectors in the rg8 target, a history buffer) smooths. If the history is reset or the motion vectors are
   wrong, the dither shows (the dotted tree) and the swap looks like a pop.

## Phase 0: make it measurable (no guessing, light on the PC)

- **Deterministic scene**: find the attract-mode course pick (the RNG seed it reads: a time / tick service call) and
  add `MKW_ATTRACT_SEED` so every run shows the same course. Without this every A/B is a lottery.
- **Draw-level flicker detector** instead of screenshots: per frame, log for each pipeline in the main colour pass the
  number of draws and, for indirect-count draws, read back the GPU count. A tree that blinks shows as a count that
  alternates. Cheap: a few numbers per frame into the log, no images.
- **Rule for runs**: one headless run at a time, at most ~20 screenshots, never batches (the 4-run batch froze the PC).

## Phase 1: find which path the blinking objects take

1. With the fixed scene, dump two consecutive frames (`MKW_DUMP` + `MKW_DUMP_LOG`) where a tree is present in one and
   missing in the next.
2. Diff the frames' pass lists: which pipeline's draw (or indirect count, or `DrawMeshTasks`) differs.
3. Classify: mesh-shader draw (suspect 2), indirect-count draw fed by compute (suspect 3), or present in both with a
   dither (suspect 4).
4. In parallel, settle suspect 1: the detector in `import` vs `mirror` on the fixed scene.

## Phase 2: fix by class

- **Mesh shaders**: implement the task/mesh path properly instead of the approximations: real primitive headers from
  the program (not synthesised), ISBE reads/writes mapped to `EmitMeshTasksEXT` / `SetMeshOutputsEXT` per Mesa's ISBE
  layout, full vertex / primitive limits from the NVN 2 control section, and the FS interface built from the mesh
  outputs. References: Eglantine PR #426 / #772, nxnative `docs/reference/shaders.md` (stage ids 6/7, control +0x56..),
  Mesa NVK mesh notes. Validate one program at a time against a frame dump.
- **GPU culling / LOD**: dump the culling dispatch's inputs (camera UBO, Hi-Z mips, instance buffers) on a good and a
  bad frame; fix whatever differs (typically the Hi-Z source texture being a fresh or aliased image, or a UBO read
  after the CPU rewrote it: then extend the uniform snapshot to the storage ranges those dispatches read).
- **Temporal resolve**: check the history and motion-vector targets keep their contents across frames (texture
  re-creation, `released textures under VRAM pressure`), and that the jitter the game applies matches what it reads.
- **Mirror write-back** (if it is involved): write back only the bytes a shader actually changed (diff against the
  pre-submission copy) or keep pools the CPU writes during a submission imported.

## Phase 3: verify

- The detector shows no alternating counts over 1000 frames on the fixed scene, in both themes of camera motion
  (attract mode and a race start).
- One visual check run (≤20 shots) and a user check in a real race.

## Order of work

Phase 0 (seed + detector) first: about a day, and every later step depends on it. Then Phase 1 decides which of the
Phase 2 fixes is needed; the mesh-shader rewrite is the largest piece if that is where the blinking lives.

## Results so far (2026-11-10)

- Phase 0 tools: MKW_SEED (same course every run), MKW_DRAWSTATS + tools/drawstats_report.py, MKW_SKIP_PROG.
- The A-B-A blinking in the desert is the cacti (one shader program); one cactus at a time is added and dropped by the
  game's CPU logic. Not conditional rendering (steady 7/6), not occlusion queries (none in the scene), not GPU culling.
- **Black cacti fixed**: MKW fragment shaders need the FMZ rewrite (Maxwell 0 * x = 0). Without it the near-camera
  dither fade computes NaN and the cactus draws as an opaque black silhouette. MKW_FMZ now defaults to all stages.
- Fixed on the way: DrawMeshTasksIndirect read its group count on the CPU at record time (stale).
- Open: whether the remaining cactus toggles are visible on screen (edge-of-view / distance cut-off) and the dotted LOD
  fade on trees (check the temporal anti-aliasing history next).

## Heap starvation root cause (2026-11-10)

- Symptom: `getMaxAllocatableSize` returned 0 at boot, and Grok's workaround handed out a fake 32 MiB plus a bump
  allocator placed inside a live game heap (6.8 GB of fallback allocations per boot, memory corruption).
- Cause: the loader bound a module's references to its *own* definitions first. The SDK's sized `operator delete`
  (`_ZdlPvm`) calls `_ZdlPv` through its PLT; on the console, rtld does a global lookup in load order (no DT_SYMBOLIC),
  so that reaches the game's heap-aware `_ZdlPv` in main. Ours bound the SDK's weak `_ZdlPv` → `free` → nmalloc,
  which rejected every game-heap block (`NotNMallocPtrError`), so the blocks leaked, child heaps could not shrink
  back on `adjust`, and their parents starved.
- Fix: `loader.c` resolve() uses the first definition in load order. Result on a boot run: 40 → 0 bad frees, 467 → 0
  fallback allocations, no failed allocations in 40 s. The workarounds are now opt-in (`MKW_HEAP_LEGACY=1`).
- Still to check in a real race: whether pop-in, low-poly models and the old crashes improved.

## Phantom / exploding trees & bushes turning into trees: partial root causes & open leads (2026-11-11)

- Symptom (Mario Circuit): huge dark tree trunks where no tree stands, bushes/grass turning into giant trees, trees
  growing/changing shape while driving, and distant bushes/trees staying half-dithered.
- Decompiling the Mario Circuit F12 capture (`dump-f12.old`) exposed three bugs in the procedural foliage placement,
  culling, and instanced vertex pipeline (fixed in this commit, though an in-game Mario Circuit test at 22:42 showed
  giant trees/foliage and blurry world textures still occurring — see open leads at the bottom):

1. **Indirect compute dispatch group counts `c0+0x0..0xb` were 0 on NVN 2 (`vk_exec_draw.c:op_dispatch_ind`)**:
   - Mario Circuit `pass 0018` (`cs b5252fdd788a115f`, NVN 2 `e590d50a45b1c299`) and `pass 0019` (`cs 0cd68c4c50a54509`,
     NVN 2 `1df9dc86787a4b3b`) are **indirect** compute dispatches (`vkCmdDispatchIndirect`) that generate procedural
     foliage world positions and write the 80-byte per-instance records in slot-5 buffer `0x2a7061000` (`+0x493e00`).
   - Both shaders normalize dispatch coordinates by `1.0 / float(uint(cp_c0[0].y << 3))` (`c0+0x4`, `numWorkGroups.y`).
   - `op_dispatch_ind` guarded the GPU `vkCmdCopyBuffer` of `VkDispatchIndirectCommand` (`12` bytes) with `if (!c0_lay ...)`,
     skipping NVN 2 compute shaders completely (and copying to NVN 1's `+0x200` instead of NVN 2's `+0x0`). Every indirect
     NVN 2 compute dispatch therefore read `cp_c0[0].xyz = (0, 0, 0)`, dividing by zero (`+inf`) and writing corrupted
     transforms/scales into the foliage instance buffer.
   - **Fix**: `op_dispatch_ind` now copies the 12-byte indirect dispatch dimensions into `B.c0_off[2] + (c0_lay ? 0 : 0x200)`.

2. **`R8_UINT` foliage species density maps sampled as float `sampler2D` (`0025-integer-sampled-textures.patch`, `vk_exec_draw.c`, `gfx_shader.c`)**:
   - `pass 0018`, `pass 0019`, and `pass 0029` bind `tex 32113` / `tex 32129` (`VK_FORMAT_R8_UINT`, `2048×2048` species ID map)
     and execute `int id = floatBitsToInt(textureGather(...).x); uint species = min(uint(id), max_species);`.
   - Without integer-sampled texture specialization, `mkw_xlate` emitted float `sampler2D` and Vulkan sampled the view with
     float/UNORM semantics: every non-zero texel returned `1.0f` (`0x3f800000` = `1,065,353,216` as `uint`), clamping every
     grass tuft, flower, and crossfading bush to `max_species` (the largest tree mesh in the biome).
   - **Fix**: Ported `0025-integer-sampled-textures.patch` (plus `0024-guarded-subword-global-loads.patch` and
     `0026-more-storage-buffers-per-stage.patch`) into `tools/xlate`, rebuilt `mkw_xlate.exe` (`6c54957fc5737b20`), and wired
     `GFX_TEX_UINT` / `GFX_TEX_SINT` + point-sampled integer `VkImageView`s through `gfx_shader.c` and `vk_exec_draw.c`.

3. **Vertex `BaseInstance` (`c0+0x644`) vs zero-based `SR_InstanceId` (`0x2F8`) (`vk_pipeline.c:spirv_base_instance`)**:
   - On NVN 2, `SR_InstanceId` (`0x2F8`) is **zero-based** (`gl_InstanceIndex - gl_BaseInstance`), which `mkw_xlate` already
     emits accurately: e.g. `sh_bc07a00ae13ddd1b.spv` computes `(gl_InstanceIndex - gl_BaseInstance) + vp_c3[1].x` without
     reading `c0+0x644`. Rewriting `OpISub(InstanceIndex, BaseInstance)` to `InstanceIndex` corrupted those draws whenever
     `firstInstance != 0`.
   - Meanwhile, 4 of the 28 `vp_c0[0][100].y` (`c0+0x644`) loads in the Mario Circuit capture (e.g. `sh_3c3c126eec55bc40.spv`)
     did not reference `SR_InstanceId` at all (`base_var == 0`), so the earlier `spirv_base_instance` bailed out and left
     indirect draws reading `0` instead of `firstInstance`.
   - **Fix**: `spirv_base_instance` now leaves `InstanceIndex - BaseInstance` untouched and rewrites every vertex-stage load
     of `vp_c0[0][100].y` to load `BuiltIn BaseInstance`, synthesising the `Input` variable (`OpEntryPoint`, `OpDecorate`,
     `OpVariable`) when not already declared. Validated across all 367 captured shaders (26 rewritten) with `spirv-val`.

## Additional runtime & release fixes (2026-11-11)

- **Guarded SSBO speculative candidates (`vk_exec_draw.c:guarded_is_declared_ssbo`)**: prevented speculative pointer
  candidates scanned from `c1`/`c3`/`c6`/root banks from scheduling bogus GPU→CPU writebacks (`vk_mem_gpu_wrote`), fixing
  the `MEMORY=mirror` vtable corruption noted earlier.
- **Attachment feedback loop learning (`vk_exec_draw.c:feedback_prepare`)**: read-only attachment sampling inside an open
  render pass now learns `fb_learned = 1` and restarts in `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`.
- **Promoted depth-stencil copies/clears (`vk_exec_xfer.c`)**: `copy_t2t` splits dual-aspect depth+stencil blits cleanly;
  `clear_tex` handles promoted depth formats and partial 3D Z-slice clears.
- **F12 frame dump (`vk_dump.c`)**: `dump = key` now defers opening/truncating `passes.txt`, `images.txt`, and `bufs.txt`
  until F12 is pressed.
- **Crash handler VEH (`platform/win32.c`)**: first-chance `crash_veh` now ignores faults outside `mkw.exe` so driver/OS DLL
  `__try/__except` probes pass through normally, while `SetUnhandledExceptionFilter(crash_uef)` still logs unhandled external
  crashes.
- **VRAM pressure & ASTC BC3 OOM fallback (`vk_texture.c`, `vk_astc.c`)**: `vk_astc_mode` always builds `bc_pipe` so per-texture
  BC3 fallback works even when `ASTC_BC=0`, and `evict_vram_oom` / `vk_texture_free_for_vram` flush deferred uploads/frees
  (`vk_upload_flush(1)` + `vk_mem_collect()`) so VMA actually reclaims device-local memory before OOM.
- **Checked & ruled out from outside review**: `indirect_reset_draw` (`e78abde51f2d6bfa`) is absent from all 3,190 MKW NVN 2
  stages; `IABS` lowers to valid `GLSL.std.450 SMax(x, -x)` with 0 shader/pipeline failures across all 239 compute pipelines.

## Open leads for next session (from 2026-11-11 22:42 Mario Circuit test)

1. **Foliage / giant trees still occurring in Mario Circuit & open world**:
   - **Bindless compute texture handles in user UBOs**: `answers_for` (`vk_exec_draw.c`) only inspects `ex->tex[st][slot]`
     (up to 32 bound slots), whereas NVN 2 compute shaders like `pass 0018` / `pass 0019` can read bindless texture handles
     from UBO banks (`cbuf_slot >= 3`) or packed `c0` slots where `slot < 0`, so `a->tex_type` stayed `0` instead of
     `GFX_TEX_2D | GFX_TEX_UINT` and the float variant from `shader-answers.bin` / cache kept running.
   - **False-positive `ptr-scan` hit inside slot-5 foliage instance buffer**: at frame 929 (`mkw.log` lines 3206–3253),
     `ptr-scan` scanned `slot 5 addr=2a7061000 size=493e00` and matched float matrix data (`w0=3f800000`, `off=0x30 ->
     0x23f5bfeac`) as a guest pointer.
   - Capture one fresh F12 frame on Mario Circuit with the new build (`dinst` + `passes.txt` + `sh_*.spv`) to verify
     both whether `2d_uint` actually compiled for `e590d50a45b1c299` / `1df9dc86787a4b3b` and what `dinst` logs for
     `1cdcc0fd9dee5df9`.
2. **Blurry / low-resolution world textures**:
   - `CompactLayers` (`vk_texture.c`) clamped all four `8192×8192×68` ASTC terrain/world arrays down to `1024×1024`
     (skipping the top 3 guest mips: `8192`, `4096`, `2048`) under the default 512 MiB `MKW_COMPACT_ASTC_MB` budget,
     combined with `BC3 recompress` (`ASTC_BC` default on), even on a 16 GB RTX 5070 Ti.
   - `getMaxAllocatableSize` on streaming heap `0x283cb0440` returned `0` 20 times during boot/course load (`mkw.log`
     lines 2124–2149).

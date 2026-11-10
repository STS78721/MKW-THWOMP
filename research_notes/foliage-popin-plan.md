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

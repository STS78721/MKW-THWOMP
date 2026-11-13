# Mario Kart World PC — handover

Date: 2026-11-12, 18:30. Written for the next session. The open bug is distant lights shining through solid world (a blue spot on a cliff in a Boo race), plus races sitting at about 40–50 fps. Foliage phantoms are already fixed. Do not reopen them.

The user plays this build by hand. One light run at a time. Do not kill `mkw.exe` if it is running. Ask before any headless race, profile, or batch of screenshots. Four heavy runs once froze the PC. Commit and push only when asked.

User-facing strings say **MKW** or **Mario Kart World**. Do not put MTF, TotK, or DKB in the window title, menu, log lines the user reads, or the version resource. Internal file-format magics (`TOTKPKEY`, `TOTKSPRP`) and the `TOTK_` environment aliases in `cfg.c` still work. Do not rename those casually.

## 1. What this is

Static recompilation of Mario Kart World (Switch 2, update 1.8.0) to native Windows, Vulkan.

- `recomp/` lifts the game's ARM64 to C.
- `runtime/` is the host: NVN 2 on Vulkan, services, input, audio, files, the F1/Esc menu.
- `tools/` is build and diagnostics.

Fidelity stays. Do not hide lights, skip the cliff draw, drop resolution, or turn off the user's colour grade to make a screenshot look clean.

## 2. Paths

| What | Path |
|---|---|
| Source | `C:\Games\MKW PC\mkw-recomp` |
| Lifted C | `C:\Games\MKW PC\generated\mkw180` |
| Game files | `C:\Games\MKW PC\exefs`, `C:\Games\MKW PC\romfs` |
| **The build the user runs** | `C:\Games\MKW PC\build\mkw` (`mkw.exe`, `mkw.ini`, `logs\mkw.log`) |
| **The F12 capture of the blue spot** | `C:\Games\MKW PC\build\mkw\dump-f12` (rewritten 2026-11-12 18:13) |
| Last good CPU profile | `C:\Games\MKW PC\build\mkw\perf-runs\pass5-20261112-153214` |
| Foliage findings (closed) | `mkw-recomp\research_notes\foliage-popin-plan.md` |
| Resolution / aniso notes | `mkw-recomp\research_notes\resolution-aniso-handover.md` |
| Do not build into | `C:\Games\MKW PC\build\mkw180` (another project overwrites it) |

## 3. Build and run

From the repo, because `build.ninja` uses relative paths:

```bash
cd "/c/Games/MKW PC/mkw-recomp"
PATH="/c/msys64/ucrt64/bin:$PATH" ninja -f "C:/Games/MKW PC/build/mkw/build.ninja"
```

There is no ninja target named `ui_menu`. Build the objects or `mkw.exe` by path. `py -3 tools/make_icon.py` regenerates the icon.

Headless (kills every `mkw.exe` first — only when the user is not playing):

```powershell
$env:MKW_SEED = '1'
& 'C:\Games\MKW PC\build\mkw\mkw.exe' --seconds 12
```

`MKW_SEED=1` fixes the attract course. Attract mode is not the user's Boo race. A headless run does not reproduce the blue spot.

Before saying a fix failed, compare `mkw.exe`'s write time with `logs\mkw.log`. The user has tested an old exe before.

`spirv-dis` and `spirv-cross` are on `C:\VulkanSDK\1.4.363.0\Bin` and in `C:\msys64\ucrt64\bin`. Copy a `.spv` out of the path with spaces before invoking them.

## 4. The user's live settings (`build\mkw\mkw.ini`)

Do not edit these to "test" a theory.

- Display: fullscreen, `resolution = 2160`, `aniso = 16`, `fps_limit = 60`, `present = fifo` (code default).
- Colour, custom, live: brightness −2, contrast 115, saturation 135, gamma 0.95. Not the defaults, so the full-frame colour compute runs every present (`runtime/vk_color.c`, `shaders/color_grade.comp`, `gfx_present.c`). The Display tab already has the rows (`runtime/ui/menu.cpp` `rows_color`).
- Performance: `memory = import`, `pipe_prebuild = 1`, `lib_prebuild = 1`, `rec_batch = 16`. `LIB_PREBUILD_KEEP` is unset, so stage-lib prebuild keeps compute only.
- `dump = key`, `dump_dir = dump-f12`. F12 captures one frame. `MKW_DUMP_PICK` was **off** for the 18:13 capture. `pick.txt` in that folder is an older 11:31 file. Ignore it.

`resolution = 2160` renders a **2560×1440** crop (a 2/3 factor, unfinished). The blue-spot capture is 2560×1440. Do not treat "fix the crop" as part of this bug. See `research_notes/resolution-aniso-handover.md`.

## 5. The bug to fix

Distant lights shine through solid world. Earlier: white spots on asphalt, light source far below the track (`Screenshot 2026-11-12 133240.png` in the user's Pictures, predates this capture). Now: a blue spot on the cliff face in a Boo race, and the overlay at about 40–50 fps.

Do not hide the light. Do not weaken the material. The spot has to fail the game's own visibility test, the same way it would on the console.

### What the 18:13 F12 capture proves

Frame 11747, fence 58755. `passes.txt` is 1.4 MB. Scene colour is `r11g11b10f` at guest address `0x8000f00200`.

| Buffer | Result at the spot |
|---|---|
| `pass_0227_t0_2560x1440_122_r11g11b10f_tm.png` | Pixel (1333, 870) is **(0, 0, 0)**. Geometry colour, before lighting. |
| `pass_0227_td_2560x1440_130_d32fs8.png` | Depth is solid cliff. Inside the blob, depth (min–max scaled) is 115–137, mean ~133, matching the ring around it. **0 blue pixels sit on a depth outlier.** Not a hole. |
| `pass_0236` / `pass_0237` half-res light (`1280×720`) | Black at the matching texel (666, 435). The spot is not coming from the half-res light buffer. |
| `pass_0252_t0_2560x1440_122_r11g11b10f.png` | The spot exists. Unscaled read at (1333, 870) is **(18, 25, 131)**. Soft blob, bbox x 1293–1348, y 862–974 (**56×113**). |
| Later 2560 passes (`0281`, `0283`, `0286`, `0357`) | The same blob. They did not create it. |

Pass 0252 is the writer: **render pass, 353 draws, 2560×1440**, depth attachment still `0x8000000000` (the scene depth). It does not sample that depth image. It samples the encoded gbuffer and the light list.

Gbuffer, pass 0247, one fullscreen draw, FS hash `195f179a70a00a4a` (`sh_195f179a70a00a4a.spv`):

- Layer 0 (`pass_0247_t0_...png`) varies: the spot is (255, 255, 255), the road nearby is (0, 255, 255), other cliff is (255, 0, 255).
- Layer 1 (`pass_0247_t1_...png`) is **(255, 127, 127, 255) at every sampled pixel, including sky**. The shader does write location 1 (`_68.y/_68.z` from `texture(fp_t_tcb_14).yz` in the cross-compile). A constant (1, 0.5, 0.5) across the whole target is a lead, not a proof. Check whether binding 9 is sampling a border / the wrong image. Do not assume it is the normal buffer until the binding map says so.

Lighting shaders in pass 0252 (48 pipes, program serials change every run, the `fs` hash and the `nvn2` origin are stable):

| FS hash | Draws | Shadow sampler in the SPIR-V |
|---|---|---|
| `f6182e326875e0e1` | 36 | No. `sh_f6182e326875e0e1.spv` |
| `21c3ebeed13c0dfb` | 18 | No. `sh_21c3ebeed13c0dfb.spv` |
| a few others (`db966595dd305039` and similar) | 1 each | Yes: `d16` 3072×3072 array, sampler 458, compare |

The common cliff shader lights from SSBOs at `c0+0x520`, `c0+0x540`, `c0+0x560` and from the 88×48×256 cluster (`r8` at `0x8005c46a00`, `rgba16f` at `0x8005406800`). It samples the full-res `rg8` at `0x80069f7400` (pass 0233, nearest) and the gbuffer array `0x8007ad7a00`. It does **not** sample the scene `d32`.

The cross-compile of `f6182e326875e0e1` (temp GLSL, not saved) has the usual Maxwell `1 - x` bit trick:

```
intBitsToFloat(floatBitsToInt(texture(...).x) ^ 0x80000000) + 1.0
```

One use (around the cross-compile's line 985) samples `sampler2DArray` binding 15 at `z = 1` — gbuffer layer 1. Another (line 1108) samples `sampler2D` binding 14 at a screen coordinate and complements `.x`. Read the SPIR-V before changing either.

The cluster is written early, **pass 0042**, dispatch 22×12×1, CS `ad0de0fd37d310d2`, prog 34501. It samples two shadow arrays with compare (3072 `d16` sampler 458, and 512 `d16` sampler 460 with lod 3..15) and stores the 88×48×256 images. Depth twins (`vk_exec_draw.c` `twin_update`) apply only to **colour** `R32_SFLOAT` / `R16_UNORM` sampled with a compare in a compute dispatch. These shadow maps are real `D16`, so they are sampled directly. The dumped shadow map `pass_0150_td_3072x3072_124_d16.png` is mostly clear (far) with real casters on about 0.3% of texels. An empty map was not the finding. A compare that returns "lit" for every sample still would be.

`passes.txt` groups draws by pipeline. It does not say which of the 353 draws wrote the blue texel. `MKW_DUMP_PICK` only tracks depth, and the depth does not change at the spot, so another depth pick will not name this draw.

### Where to look, in order

1. Bisect the pass 0252 pipes on that view. `MKW_SKIP_PROG` drops one program serial, and serials change every launch, so prefer the stable FS hash / NVN 2 origin. A one-frame colour readback at (1333, 870) after each pipe of a 2560×1440 pass would name it. Do that on a capture frame, not on every race frame.
2. If the writer is `f6182e326875e0e1` or `21c3ebeed13c0dfb`, the light is in the SSBO list and survived the cluster. Then check pass 0042's shadow compare in compute: a compare that returns 1 (fully lit) leaves occluded lights in the cluster. Log or read back one tile of the `r8` cluster (tile about x=46, y=29 for this pixel: 2560/88 ≈ 29 px, 1440/48 = 30 px). The PNG is scaled by the image maximum, so do not trust its 8-bit values as light indices.
3. If gbuffer layer 1 is stuck because binding 9 sampled the wrong image, fix the bind. Confirm with the descriptor the draw actually used, not with the constant colour alone.
4. Depth sampled **without** compare is already forced to nearest (`vk_exec_draw.c` `tex_info_full`, the `point` flag, aspect depth, type bit `0x400` clear). That did not remove this spot, because pass 0252 does not sample the scene depth. Do not "fix" it again by blurring or by dropping samples. The point sampler still keeps the guest LOD range. Sampler 306 is linear mipmapped with lod −1000..1000 and is used on depth in other passes (pass 0236 slot 4, pass 0281). Clamping that LOD is a separate experiment. It is not what painted this blob.

## 6. Do not repeat these

| Attempt | What happened | Leave it |
|---|---|---|
| Occlusion conditional reads the last guest count | `counter_word` / `cond_enabled` in `vk_exec_xfer.c`. In-flight samples-passed returns status 2, and `cond_enabled` **draws** (line 387: `rx == 2 \|\| ry == 2` → return 1). Changing that to `memcpy` the guest counter tanked the frame rate. The read is `vk_mem_resolve` plus an uncached import-memory load on the record path. The comment at the top of the file explains why an unknown samples-passed count renders: the report is usually in the command buffer being recorded, and the word in memory is an older query. | Keep the status-2 → draw behaviour. A CPU-side cache of the last completed count is the same decision as the guest word, which already failed. Do not ship it without a race the user says is still 60 fps. |
| Blocking full shader warm | `lb_warm_prepared` plus `lb_start(..., blocking=1)` froze the window (replay frames of 27–42 s). Combined with a pointer fingerprint that hashed the SSBO tail, the pointer cache never hit. | `vk_pipeline_prewarm` joins before the guest. `lb_start(..., 0)` stays background. `MKW_LIB_PREBUILD_KEEP` default 0 (compute only). Do not block on it. |
| Pointer-scan reorder, or dropping `ptr_cb_epoch` | Menu crash, wild PC. Small-then-large scan order is required. The epoch is part of the life key. | `PTR_RESULT_SETS` 512, `PTR_RESULT_WAYS` 4. Fingerprint is `ptr_bufs_quick_key`: resolved base, not the SSBO tail. Failed resolves stay in the miss set. No per-hit `rt_log`. |
| `FIX_VS_FAULT=0` | Draws the instanced light quads that fault the NVIDIA driver (invalid instruction pointer). The skip loses those quads. It is a mitigation. | Default stays on (`vk_exec_draw.c` `light_quad_fault_skip`). Do not turn it off to "get the lights back" on this GPU. |
| Deleting `pipeline-cache.bin` | 466 MB. Rebuilding it stalls startup and can still fault if the SPIR-V is bad. | Leave the file. |

## 7. Frame rate (separate from the spot)

Pass5 (2026-11-12 15:32, attract course, seed) settled near 60 fps. Replay ~11.4–11.6 ms, GPU ~4.3 ms. Hot samples: `pointer_table_try_add` 8.24%, `pointer_table_scan_range` 4.73%. The 4-way cache, miss set, and head-only fingerprint are the fidelity-safe result of that. They are in the current exe.

The user's Boo race is 40–50. That is a different scene from pass5: 2560×1440, aniso 16, colour grade on, 353 lighting draws. There is no profile of that race. Do not start one without asking. `mkw.1.log` (18:02) is the session after the occlusion experiment: clean exit, 219 frames, mean **28.6 fps**. That number is the regression, not the baseline.

`MEMORY=import` stays. Mirror write-back is ~20 MiB/frame and corrupts guest SSBOs.

Console spam is capped: `vk_mesh.c` DrawMeshTasks logs stop at `mesh_log_ok < 4`. Pointer-table "found" / fallback logs are gone. Unlimited logs made a 60 fps run look frozen in a terminal.

## 8. Device lost on launch (intermittent, not this bug)

`logs\mkw.log` around 18:07: frame 181, one pipeline waited ~500 ms, then a large ASTC upload, GPU breadcrumb submission 927, invalid instruction pointer `0x2059d4d20`, `upload submit failed (-4)` (`VK_ERROR_DEVICE_LOST`). Process commit ~16.7 GB of ~80 GB. Not system OOM. VRAM heap 0 was 5637/15227. `pipeline-cache.bin` (~466,672,076 bytes) was written at 18:06:51 and left in place. The user got back in and pressed F12, so it is not every boot. Aniso 16 plus a 2160-line field has faulted during upload in an earlier headless run. Do not change their aniso as a fix. If it dies at the same point again, name the pipeline that waited 500 ms. Do not delete the cache as the first move.

## 9. Already done (leave it)

- Foliage phantoms: translator patch `0027-keep-warp-operations-in-place.patch`. Base-instance rewrite in `vk_pipeline.c` (`MKW_BASE_INSTANCE=c0` turns it off). F12 `sh_*.spv` files are the shaders **before** that rewrite.
- Colour profiles in the Display tab. Brand string "Mario Kart World" on the window, Vulkan application name, version resource, menu.
- Pointer result cache as in section 6.
- Background pipeline prewarm. Mesh and pointer-table log caps.
- `FMZ=all` default. `VK_PRESENT=fifo`. Black-road BC3 blit and the heap load-order fix are older and stay.

## 10. Diagnostics

| Knob | Use |
|---|---|
| F12, `dump=key` | One frame: `passes.txt`, PNGs, `sh_<hash>.spv` (pre base-instance rewrite). |
| `MKW_DUMP_PICK=1` | Depth grid per draw of wide passes → `pick.txt`. Useless for this spot: depth does not change there. |
| `MKW_DRAWSTATS=key:<n>` and F9 | Per-pass draw counts. `python tools/drawstats_report.py logs\mkw.log`. |
| `MKW_SKIP_PROG=<serial>` | Drop one program for a bisect. Serials are not stable across launches. |
| `MKW_SEED=1` | Same attract course. Not the Boo race. |

## 11. Suggested order

1. Read section 5. Open `dump-f12\passes.txt` at pass 0252 (line 3012) and the two SPIR-V files named there.
2. Name the draw that writes (1333, 870) on a new F12 of the same view, with a colour readback or a hash skip. Do not guess from pass 0227.
3. Fix that draw's visibility (cluster shadow compare, or the gbuffer binding if layer 1 is actually wrong). Keep every other light.
4. Do not touch `cond_enabled`, the pointer-scan order, `ptr_cb_epoch`, or the blocking warm path while doing it.
5. Ask the user before a profile run. The 40–50 fps race is not pass5.

## 12. New blocker: intermittent Crown City vertex explosion (2026-11-12 evening)

**Prioritize this before the Boo-race light occlusion.** The user reports violent flickering at Crown City's starting street. Their screenshot is `C:\Users\debee\AppData\Local\Temp\codex-clipboard-c3925114-c301-4071-9811-5da38afc5d48.png`: a saturated blue sheet covers most of the main view while Mario, road, and minimap remain. I observed the running `Mario Kart World` window through computer-use screenshots: among 8 successive samples at about 60 fps, multiple frames abruptly filled with enormous solid blue/cyan/white polygonal sheets, then cleared on the next frame. The HUD and minimap remained correct. This is a live, intermittent world-rendering fault, not just a bad still image. The user has now closed the game because the flickering caused a headache. Do not ask for another visual test until an offline-supported fix is ready.

Two F12 presses at 19:53 and 19:54 used `build\mkw\dump-f12`. F12 stalls several seconds and changes the timing. The first captured final image, `pass_0289_t0_3840x2160_43_rgba8srgb.png`, shows missing distant city geometry (sky where buildings should be), but not the blue wall. The second captured final image, `pass_0288_t0_3840x2160_43_rgba8srgb.png`, looks normal. `passes.txt` and most pass images in that directory were overwritten by the second F12; the first final image survives because its pass number was one higher. These captures reinforce the user's observation that a full F12 can hide the transient explosion. Preserve them; use separate dump directories for any future captures.

Earlier targeted color readback in `build\mkw\dump-light-pick\color-pick.txt` located one **actual blue-wall writer**: pass 0189, pipe 23, draw 50, program 39436, VS hash `8a1d616ebcc6c3b2`, NVN2 VS origin `2367473af09b1016`, FS origin `2b9e840c81c6c517`. It changed a sampled pixel from black to blue. This shader uses vertex attribute 7 as bone indices and VS storage at c0+0x130. `build\mkw\dump-vertex` confirmed vertex declarations: loc0 R32G32B32_FLOAT, loc7 R8G8B8A8_UINT, stride 40. `build\mkw\dump-auto-skinning` captured 16 ordinary-frame draws of this shader without F12. Mesh index ranges and vertex positions were valid (first mesh: 4732 vertices, indices at most 1774 for base vertex 0, local positions about ±1.5/0..3.1). This rules out a simple corrupt index list in that snapshot, **not** a timing-sensitive vertex or storage race.

Important caveat about the skinning dumps: `vk_exec_inst_pass_end` copies SSBO/UBO bytes **after the render pass**, so those bytes may differ from what an earlier draw read if the CPU or GPU rewrote the backing. The correct matrix buffer is VS SSBO **slot 2**, not slot 0: c0+0x130 holds its pointer (`vi10_p0000_d0_c0.bin` has low address `0x42d3d000`, high 2). `vi10_p0000_d0_m0_242d3d000.bin` begins with u32 `13`, followed by a long zero region and later matrices. The translated shader in `dump-light-pick\vs_2367473af09b1016.glsl` calculates a matrix index from that word and raw bits of `vp_c7[9].y`; the pass-end `vi10...u7...bin` has `vp_c7[9].y = 0x3f7e85f8` (float ~0.994). Taken literally, that multiplication indexes far beyond the 78,016-byte buffer. **Do not patch the shader from that alone**: the UBO/SSBO snapshots are post-pass, and the correct at-draw values must be checked first. The shader binary is `build\mkw\dump-light-pick\sh_8a1d616ebcc6c3b2.spv`; identical bytes are in cache file `shader-cache\056c7597f339128e594a11f8dbe8a956aa4d2e09c5b286cf8ef6b881e84cb664.spv` with its `.res.json` reflection.

Diagnostic edits are currently **uncommitted** on top of other pre-existing dirty work: `runtime/vk_dump.c` adds `MKW_DUMP_COLOR_PICK` and `MKW_DUMP_SKIP_IMAGES`; `runtime/vk_exec_draw.c` extends `MKW_DUMP_INST` snapshots and adds `MKW_INST_AUTOCAP`; `runtime/cfg.c` registers them. The auto-capture setting was removed from `build\mkw\mkw.ini` after one run became stuck in a loading animation. That hung session rendered ~60 fps indefinitely with no crash/device loss in `logs\mkw.log`, and the user closed it normally. The next normal-config launch reproduced the vertex explosion, so the auto-capture setting is not required for the fault. Current ini `[experiment]` is only `dump = key` and `dump_dir = dump-f12`; resolution 2160, aniso 16, user color grade unchanged. Remove or gate the diagnostic overhead after finding the fix. Do not assume the loading hang had the same cause.

Next investigation: capture the **actual at-draw** UBO7 and SSBO2 values or their GPU-visible binding for this shader in an ordinary frame, and correlate them with a bad output frame without a full F12. Consider an imported-memory read/write race: the live screenshots show frame-to-frame alternation, and F12's synchronization often makes it disappear. Verify this before changing memory mode or shader behavior. Keep graphics fidelity and every world draw.

While the user left the game running at the fault, I sampled the live window again without pressing F12. In eight successive screenshots, three showed large blue/cyan/white sheets, including one frame with only Mario, a strip of road, and the HUD visible; the following frames restored the city. The game stayed around 60 fps and `logs\mkw.log` advanced through 20:01 with no graphics error or device-loss message. The saved `dump-auto-skinning` pass-end UBO7 samples all have `c7[9].y` near `0.99423` and SSBO2 first words 13, 11, or 10, so the apparent out-of-range address is repeatable in those snapshots. These are still **post-pass** samples and cannot establish the bytes fetched at draw time. No executable rebuild or in-game setting change was made while the user left the game open.

### 12a. Root-cause trail, later on 2026-11-12 (Claude). Read this first; it supersedes the shader and race theories above.

**Symptoms are one bug.** The blue/white "sheets", the **invisible pedestrians** that you still collide with (they launch the kart), and probably the **green lamp posts that don't break**: in each case, skinned objects (Toads and other pedestrians) are drawn with broken bone matrices. Huge values give sheets; zero rows collapse the mesh to nothing. The bushes you can clip through are the same in the emulator, so not a bug. Pedestrians appear only on some streets, not at the start line. The bug is intermittent: 1 of 3 runs showed it.

**Corrections to the notes above.**
- The "u7" in dump file names is NVN uniform **slot 7 = c10**, not c7. Bank cN = uniform slot N−3. The skinning VS (origin `2367473af09b1016`) uses: matrix index = bones-per-object (word 0 of VS SSBO slot 2) × object slot (c7[9].y = uniform slot 4, byte 0x94, an integer) + bone index (attribute 7). Matrices are at 16 + 48·i, and there are 125 object slots per buffer. The shader and translator are **correct**.
- Bone buffers rotate in a ring of 4 per kind: 13-, 11-, 10- and 8-bone kinds, about 0xd000/0x15000 apart.

**Ruled out, with evidence:**
- GPU/CPU race: the at-submit snapshot matches the live data. `MKW_SSBO_SNAPSHOT`, default off, did not help.
- Inline updates or buffer copies into the bone buffer (logged with `dskinu`): none carry the bad data.
- A compute writer, or unresolved storage bindings.
- Resolution, including at Original.
- Allocation failures.
- Lifter fixed-point conversions (`shift_imm`/`fp_fixed`) and the runtime `fromint`/`vfromint`.
- Grok's soft-skip VEH: it never fired.

**Chain found with page-protect write-watches** (`MKW_SKIN_WATCH`, `runtime/ov_system.c`):
1. The only CPU writer of the bone buffers is **`f_70001673c0`**, a plain 48-byte copy loop (`generated/mkw180/main/main_0020.c` around line 640; guest `L_7000167b98`..`L_7000167be4`). For each dirty bone (bitmask at `[x26 + x27*4]`), it copies the matrix at **+0x30 of a 0xa0-byte bone entry** (array at `[unit+0x28]`, stride `x16 = 0xa0`) into the GPU buffer. The copy is translated correctly, so the **source bone entries are already broken**.
2. The broken entries hold values from 1e6 up to 1e23 that **grow over frames**. The same vector (-3.14e6, -6.83e6, 3.17e7) repeats across bones of one model, so a bad parent world matrix propagates to its children. 15 of 16 copies out of each broken entry over 16 frames were broken, and almost every copy had fresh writes before it, so these are **computed** garbage, not uninitialised memory.
3. The writers of those entries are the animation and world-matrix functions **`f_700042bea0`** (main_0049.c, 3773 lines, scalar FP: 90 mul, 51 add, 17 sub, 11 div, 41 fma, 39 int→float, 18 float→int, 32 compares) and **`f_7000429820`**. Each individual write is "fine" (under 1e6), so the blow-up is gradual and accumulates through feedback.
4. `f_700042bea0` calls **`sinf`** (18 times, PLT `f_7002bade50`, GOT 0x70036f5028) and **`cosf`** (15 times, `f_7002bade60`). They resolve to the SDK's ARM optimized-routines versions: sinf = `f_7004b7fa70`, cosf = `f_7004b79560` (`generated/mkw180/sdk/sdk_0047.c`), with helpers `f_7004c3ed40` (round), `f_7004b797d0` (large-arg reduction) and `f_7004b79770`/`f_7004b7edc0` (tails). Not yet verified.
   - A tool to name PLT imports and module exports from `exefs/*` is at `C:\Users\debee\AppData\Local\Temp\claude\30af027f6c9dbd63\609f7855-0c90-4c69-a552-a9911dcba7cc\scratchpad\plt.py` (`python plt.py exefs/main 0x70036f5028`, `python plt.py exefs/sdk exports`).

**Next step in progress:** `MKW_TRIG_CHECK` (`trig_check = 1` in mkw.ini, observer at the end of `runtime/ov_system.c`). It runs the guest sinf/cosf and compares each result with host `sin`/`cos` in double. It logs `dtrig` lines for the first 40 mismatches over 1e-5, plus a count every 2^20 calls.
- **If it reports mismatches:** the bug is in the lifted libm or one of its helpers. Fix the lifter or runtime op, or as a last resort add a `HOST_OVERRIDE("sinf"/"cosf")`. The fix belongs in the translation, not a hide.
- **If it reports none:** diff `f_700042bea0`'s FP ops against the ARM semantics, especially:
  - `fp_fma_s` operand order and negation (FMSUB/FNMADD/FNMSUB);
  - `FOP_NMUL`, `FOP_MAXNM` and `FOP_DIV`;
  - the `toint_s` rounding modes;
  - `cmp_s` with NaN, and the conditions that use it.

  Or extend the stage-2 watch to log each written value per site, and find the first write that departs from the emulator.

**Diagnostic state (all uncommitted, remove or gate before the installer):**
- `skin_check = 1` and `trig_check = 1` in `build/mkw/mkw.ini`. `skin_watch` is set to 0. It is a two-stage watch: stage 2 follows the source bone entries for 16 frames and causes a brief hitch.
- `build/mkw/build.ninja` has an extra `genflags = ... -g` line under the `main_0020.o` edge, for addr2line. The original is at `C:\Users\debee\AppData\Local\Temp\claude\30af027f6c9dbd63\609f7855-0c90-4c69-a552-a9911dcba7cc\scratchpad\build.ninja.bak`, and `tools/build_game.py` regenerates it anyway.
- `SKW_COPY_OFF 0x19ea1` in `ov_system.c` is the host offset of the copy store inside `f_70001673c0` for the current `main_0020.o` build. It must be recomputed (`nm` + `objdump`) if that object is rebuilt.
- Runtime files touched: `vk_usnap.c`, `vk.h`, `vk_exec_draw.c` (skin_check, dskinu, dskinm, dbind), `vk_exec_xfer.c`, `main.c` (`rt_lifted_function`), `ov_system.c` (watch and trig check), `cfg.c` (knobs).

**Update: tracing back from the `f_700042bea0` store sites** (line info via an extra `-g` on `main_0049.o` in build.ninja):
- Site `image+0x128ca16`..`0x128cb3d` (main_0049.c ~31281–31287) writes the bone's local matrix (+0x00) and its translation and scale (+0x6c/+0x7c/+0x8c, +0x90..+0x98). The rotation at +0x60..+0x88 is written by sites `0x1281f34`..`0x1282081` (~31598–31604): a quaternion→matrix with `s11 = 1.0`.
- The bone entry layout is: +0x00 local 3×4, +0x30 world 3×4 (copied to the GPU), +0x60 rotation 3×4 (column 3 = translation), +0x90 scale.
- In the broken entries, scale and translation are 0 and the rotation is huge. In one case the rotation was a clean rotation scaled by ~52, so the quaternion fed in is far from unit length (|q|² ≫ 1).
- The quaternion comes straight from the animation bitstream: per-component widths from a table, `(word << pos) >> (32 - width)`, 32-bit words. Each component is then turned from half-float bits into a float in software (L_700042c394..c480; exponent +0x70, mantissa << 13). Both are faithful to ARM semantics, including the LSLV/LSRV mod-32 shifts. So the **bits being read are bad**.
- The keyframe index = trunc(t) − (t < 0 and t has a fraction), with t = [frame control + 4 or +0x54] + [anim + 0x90]. The FCCMP immediate in the translation is #0. The offset is index × [hdr + 0x10c] + [hdr + 0x28]. The runtime's `toint` saturates and maps NaN → 0 like ARM.
- The leading hypotheses are now: (a) t runs past the curve's end, so the reads go beyond the keyframe data; or (b) the curve data is corrupt. That would be the pasted ZSTD or buffer-reuse theory, still unverified.
- **Probe in place:** `anim_check = 1`. Hand-inserted calls `rt_anim_t`/`rt_anim_q` in the GENERATED `main_0049.c` (3 lines plus 2 prototypes at the top; the pristine copy is `main_0049.c.orig` in the scratchpad), implemented at the end of `ov_system.c`. For each non-unit quaternion, a `danim` line logs t, the curve header, the stride, the data pointer, and the good-t range seen for that header, plus a one-time dump of the header. **Regenerating the code removes the probe.** Restore the original before shipping in any case.
- Lifted `sinf`/`cosf` check (`trig_check = 1`): no run has completed yet.
- **Result, 21:55 run:** the lifted `sinf`/`cosf` match the host in 6.3M calls with 0 mismatches, so **libm is ruled out** (`trig_check` is now 0). The first quaternion probe point (L_700042c480) was wrong: most calls see zeroed slots because other curve formats decode elsewhere. The probe has moved to right after the rotation stores (after L_700042c4fc). It now checks the written 3×3 rows for unit length and logs the raw packed words at x29-0x70/-0x54/-0x50/-0x4c.

### 12b. Offline audit and GitHub comparison (Codex, 2026-11-13)

Read [research_notes/crown-animation-audit.md](research_notes/crown-animation-audit.md) before another probe run.
The moved probe has not run yet. Its coverage also needs qualification: blends can reach it without the time
hook, and separate Euler paths bypass it. `ov_system.c` now omits stale time/header/quaternion attribution for
blends and labels `hdr+0x28` as an offset with an external base. That offset can validly be zero. The diagnostic
source passes syntax and four CPU-only regression checks; **mkw.exe has not been rebuilt**.

Offline original-ARM checks found no mismatches in 65,536 half conversions, 50,688 extraction cases and 20,023
time-conversion cases. GitHub's similar Crown City task/mesh-table fix is already present locally. Correction:
upstream adriadam10 PR #172 already added version-specific ZSTD hooks for MKW 1.8.0. This local fork has not
integrated them: `recomp/patches-mkw180.txt` is empty apart from comments, the generated 1.8.0 `patches.h` has
`RT_PATCH_COUNT 0`, and `runtime/ov_zstd.c` still targets TotK 1.4.3 guest addresses. Validate the upstream sites
against this dump and port the 1.8.0 entries before using the local decoder comparison option. No confirmed game fix
yet; capture actual sample inputs and bounds next.
- **Result, 01:04 run on 2026-11-13 (moved probe):** about 16% of rotations written by this path are non-orthonormal. The raw packed words behind them are **not keyframes**. They are pieces of heap pointers (`0x4030`, `0x00000002`, i.e. halves of `0x2_4030_7060`-style addresses), `0xffff` runs and small integers. The same junk repeats every frame for the same bone, so the junk is static.
  - Time is in range: t = 1036 with 1218 frames in the buffer (`hdr+0x98` high16 = 0x4c2, size `hdr+0xa8` = 0x26100 = 1218×128); t = 1394 of 1516 frames. So this is **not a time overrun**.
  - The read address = `[[[hdr+0x40]+0x10]+0x100]` + `[hdr+0x28]` + stride(`hdr+0x10c` = 128) × frame + bone × `[hdr+0x129]`.
  - Host zstd (`ov_zstd.c`) is **not active** for MKW (no `zstd:` log lines; its patch sites are for TotK 1.4.3 only), so the guest's own decoder fills the data.
  - **Conclusion so far:** the keyframe buffer's contents are wrong (unfilled, freed and reused, or overwritten). The decode is not at fault.
  - **Probe now also dumps:** on the first bad rotation for a header, it writes the whole keyframe buffer and 0x400 bytes of its resource object to `build/mkw/anim-dump/h<hdr>_b<base>_n<k>.bin`. It dumps again (up to 4 times) if the buffer's hash changes. Next: see where in the buffer the junk starts, and whether it changes over time, then find the loader or allocator responsible. Candidates: short read, async-load-not-finished, a use-after-free through the heap changes in `mkw-heap-root-cause`.

### 12c. Keyframe buffer holds foreign objects; heap tracker added (Claude, 2026-11-13)

**Evidence so far (runs at 01:04 and 01:08 with the matrix probe):**
- The raw packed words behind the bad rotations are not keyframes. Reassembled as the 8 bytes per bone that the game reads, the 01:04 samples are `0x0000000240300600`, `0x0000000240300600`, `0xffffffff00000000`, three good bones, `0x00000000ffffffff`, `0x0000000240300638`.
- **Those pointers point at themselves.** The layout is the same in every run: hdr, then `[hdr+0x40]` 0x3b8 later, then the keyframe buffer at the next page boundary (`source_base`), then `[hdr+0x38]` about 0xf60 past the buffer's end. For hdr `0x2402df358` that puts the buffer at `0x2402e0000`. Frame 1036 × stride 128 = +0x20600 → the read address is `0x240300600`, which is exactly the pointer value found there. So a live object with empty intrusive-list heads (prev = next = self) sits **inside** the keyframe buffer. This is an overlapping allocation, a use-after-free, or a shrunk block. It is not a bad decode and not bad file data.
- The same junk repeats every frame, and bones in between still hold valid unit quaternions, so the buffer is partly overwritten, not wholly replaced.
- All three bad samples so far are in the last quarter of their buffers (76%, 85%, 92%).
- The header fields: frames = u16 at `hdr+0x8a`; buffer size = u32 at **`hdr+0x98`** (= frames × stride; my first dump used +0xa8, which is why the 01:08 run logged "keyframe buffer not resolved" and wrote no dump); stride = `hdr+0x10c`; bytes per bone = `hdr+0x129`; component types at `hdr+0x114` (0x10 = half, 0x20 = float).
- Both formats are affected: stride 128 with halves (8 bytes per bone) and stride 512 with floats (16 bytes per bone, 01:08 run: raw floats 1,0,1,0 / 1,0,1,1 where a quaternion should be).
- Host zstd is not active for this build (Codex's 12b explains the missing 1.8.0 patch entries), so it cannot be the corrupter.
- The only heap message in the log is `getMaxAllocatableSize heap 0x283cb0440 returned 0` at boot (20 times, lr `0x700259a214`). None of the soft-catch wraps (`system:` / `heap: ... unattached`) fire in these runs.

**Probe now in the build (`anim_check = 1`, all in `runtime/ov_system.c`):**
- **Heap tracker (`daht` lines).** With `anim_check` on, the ExpHeap alloc hook and the free/resizeFront/resizeBack vtable pass-throughs (the `MKW_HEAP_TRACE` ones, now also installed for `anim_check` and re-applied in `ov_system_on_inits_done`) record every event into a 512K-entry ring with four guest return addresses each. Allocations ≥ 0x8000 bytes also go in a table until freed.
- **First sight of a curve header** (`danim first sight`): resolves the keyframe buffer, keeps a copy, and registers it with the tracker. The tracker then logs the live big allocations overlapping it, the past ring events touching it, and from then on every alloc/free/resize that touches it, as it happens.
- **Change detection** (`danim buffer CHANGED`): every 512 samples per header the live buffer is compared with the first-sight copy; changed byte ranges are logged with the heap event number, and the first three changed states are written to `build/mkw/anim-dump/`.
- **Bad sample** (`danim sample`): the exact read address (frame, bone = `c->x[19]`), the 16 bytes there now and at first sight.
- **Dump at the first bad rotation**: live buffer with 0x2000 bytes either side, the first-sight copy, the resource object and the header.

**What the next run should tell:** which allocation the keyframe buffer belongs to and who made it; whether it was freed or resized; and who allocates inside it (return addresses). If no ExpHeap event explains it, the dumps and the change log give the offsets and timing for a write-watch on the changed pages.

### 12d. ROOT CAUSE FOUND AND FIXED: animation bakes were never written back to the CPU (Claude, 2026-11-13 morning)

**Read this before 12a–12c; it supersedes their open hypotheses.**

**Mechanism (verified headless, several runs):**
- MKW bakes each skeletal animation on the GPU into an **RGBA16F texture** (format `0x29`, width = bones, height = frames: 13×1211, 8×1218, 11×1516 for the Crown City pedestrians), stored in a **CPU-cached NVN pool** (flags `0x24`) created per animation by the generic pool allocator `f_70005d72f0`.
- The CPU animation code (`f_700042bea0`) then reads the half-float keyframes **straight from that pool** after the fence, without `nvnMemoryPoolInvalidateMappedRange`. This is the same pattern as TotK's grass maps.
- The runtime writes GPU-rendered (mirrored) targets back to their storage when the CPU waits on a fence, but only up to `WB_FENCE_LIMIT` = **64 KiB** (`vk_texture_write_back_small`). The pedestrian bakes are 150–200 KiB, so they were **never written back**. The trace line `vk: fence write-back skipped-for-size ... 13x1211 fmt 0x29 ... pool cpu-cached` listed exactly these, 18–20 per session.
- So the CPU read whatever was in the pool memory: **zeros** (fresh pages: zero scale → collapsed mesh → **invisible pedestrians** that still collide and launch the kart) or **stale heap bytes** (old objects, pointers, float transforms → huge bone matrices → the **full-screen blue sheets**). Which one you got depended on what that memory held before, hence "1 in 3 runs".
- Small bakes (≤ 64 KiB) were written back and animated correctly. The pool write-watch (`dapw`) showed their only writer was `vk_texture_write_back_small → storage_texels → nvn_texels_box`, and the large pools were never written by anything on the CPU side.

**Fix (`runtime/vk_texture.c`, `wb_limit_of`):** a target in a **CPU-cached pool** is written back on fence waits up to **4 MiB** (`WB_CACHED_LIMIT`). Other pools keep the 64 KiB limit; `WB_FENCE_LIMIT = 0` still disables it.
- Verified with probes off and `MKW_TRACE=gfx`: no `skipped-for-size` lines; all 18 bakes appear as `over-default` write-backs; frame rate unchanged (60 fps, 1% low ~50).
- Two other CPU-cached targets that used to be skipped now go back too: 256×64 (128 KiB) and 256×32 (1 MiB, `0x2af350000`/`0x2af371000`). Their purpose is unknown. Possibly relevant to other "the CPU sees nothing" symptoms such as the **lamp posts that don't break** (not verified).
- **Confirmed fixed in-game by the user in Crown City (2026-11-13).** Lamp posts not re-checked separately.

**Cleanup done:**
- `generated/mkw180/main/main_0049.c` restored byte-for-byte (probe hooks removed).
- The per-file `-g` lines removed from `build/mkw/build.ninja`.
- Diagnostic keys (`skin_check`, `skin_watch`, `trig_check`, `anim_check`) removed from `build/mkw/mkw.ini`; the copy with them is in the scratchpad as `mkw.ini.with-diagnostics`.

**Diagnostics kept in source, all off by default:**
- Knobs: `MKW_SKIN_CHECK`, `MKW_SKIN_WATCH`, `MKW_SSBO_SNAPSHOT`, `MKW_TRIG_CHECK`, `MKW_ANIM_CHECK`.
- Under `anim_check`, `ov_system.c` holds the heap tracker (`daht`), the per-pool init records (`dapool`), the write-watch of `f_70005d72f0` pools (`dapw`), and the keyframe-buffer first-sight / dump / diff (`danim`).
- `rt_anim_t`/`rt_anim_q` only fire if hand-inserted into the generated `main_0049.c` again.
- Always-on hooks, with negligible cost:
  - `rt_anim_pool_note` at pool init/fini (returns at once when off);
  - `rt_anim_unwatch` in `mem_thaw` and `rt_anim_preread` in the three file-read paths (return when nothing is watched);
  - two thread-local stores in the host `memcpy` override.
- `build/test_anim_probe.py` passes (its harness now includes `windows.h`).

### 12e. Still open after the 12d fix: some objects have no collision (user report, 2026-11-13)

The pedestrians now draw and animate (a Yoshi pedestrian is visible on the pavement in the user's screenshot). But the kart still **drives through some objects** in Crown City:
- the **green lamp posts**, which should break and fall over;
- **metal road barriers / guard rails** on the kerb.

The original game was compared in an emulator, where these collide. Bushes pass through in the emulator too, so they are not a bug. Nothing has been investigated yet. A first idea, unverified: the 12d fix covered CPU-cached render targets of up to 4 MiB. Collision for breakable objects might depend on another GPU→CPU path that is still missing, such as a target in a non-CPU-cached pool, a buffer the GPU writes and the CPU reads, or a size over 4 MiB. The `MKW_TRACE=gfx` lines `fence write-back skipped-for-size` and the `dapw` pool write-watch (`anim_check`) are the tools to start with.

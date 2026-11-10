# MKW mesh / geometry gaps — research findings

Date: 2026-11-09  
Scope: Switch 2 / NVN2 mesh+task shaders, `nvnCommandBufferSetGeometryProcessingMode`, `DrawMeshTasks*`, Ampere SASS stage memory (`ISBERD`/`ISBEWR`), emulator behaviour, and what a Vulkan host that currently **accepts but does not record** these calls should do next.  
Primary game evidence: Mario Kart World (MKW) foliage / scenery LODs; cross-checked against Bananza, OoT Switch 2, and public emulator notes.

---

## Executive summary (actionable)

Missing MKW scenery that goes through **native NVN2 task/mesh programs** will stay missing until the host:

1. **Records** `SetGeometryProcessingMode` (state) and `DrawMeshTasks*` (draw ops), instead of the current accept/no-op stubs in `runtime/nvn_api_nvn2.c`.
2. **Classifies** Ampere programs whose control section marks stages **task (6) / mesh (7)** — SPH alone still looks like vertex/domain.
3. **Lowers** `ISBERD`/`ISBEWR` into a real stage-memory ABI (not Maxwell no-ops), then either:
   - **Preferred:** emit `VK_EXT_mesh_shader` pipelines and `vkCmdDrawMeshTasksEXT` / Indirect / IndirectCount, or
   - **Fallback:** geometry-as-compute (task+mesh → SSBO arenas → ordinary indexed draw) when the device lacks mesh shaders.

Do **not** confuse this with Bananza’s **passthrough geometry shader → Metal mesh / compute** path. That is classical GS emulation. MKW bushes/grass are real task/mesh programs launched by `DrawMeshTasks`.

---

## Local host gap (this repo)

| Call | Current behaviour | File |
|---|---|---|
| `nvnCommandBufferDrawMeshTasks` (+ 3D / Indirect / Multi* variants) | `NVN2_NOT_RECORDED` — accepted, first call logged, **nothing recorded** | `runtime/nvn_api_nvn2.c` |
| `nvnCommandBufferSetGeometryProcessingMode` | `NVN2_MODE_HANDLED(..., 0)` — mode **0** treated as handled no-op; **any nonzero** logged as unsupported | same |
| Proc table | All mesh/task names present | `runtime/nvn_procs_nvn2.inc` |
| Ampere `ISBERD` (`0x923`) | Lowered to a Maxwell move / attribute path for **geometry**, not full mesh stage memory | `runtime/third_party/eglantine/.../lower_to_maxwell.h` |
| Ampere `ISBEWR` (`0x927`) | **Accepted and dropped** | same |
| Docs | Mesh/task listed as still to handle | `docs/bananza-shaders.md`, `docs/bananza-plan.md`, `docs/nvn2-command-modes.md` |

Comment in `nvn_api_nvn2.c` (geometry mode):

> Geometry processing mode 0 is the traditional vertex pipeline… A nonzero value selects the mesh pipeline, which has no Vulkan counterpart here yet.

`docs/nvn2-command-modes.md` explicitly separates compute-overlap (handled) from `SetGeometryProcessingMode` (mesh-pipeline state, still open).

---

## NVN2 API shape (from public ports + GL/Vulkan analogues)

### `nvnCommandBufferSetGeometryProcessingMode`

- Signature (observed everywhere): `(NVNcommandBuffer *cmd, uint32_t mode)`.
- **Mode 0:** traditional VTG (vertex / tess / geometry) pipeline — only value this repo currently “handles”.
- **Mode ≠ 0:** selects the mesh processing path so subsequent `DrawMeshTasks*` are meaningful.

#### Enum values 0 / 1 / 2 — documentation status

| Value | What is known | Confidence |
|---|---|---|
| **0** | Traditional / mesh-control **off**. Matches local Bananza/tutorial captures and `NVN2_MODE_HANDLED(..., 0)`. Hardware analogue: `SET_MESH_CONTROL.ENABLE = FALSE` (`NVC797_SET_MESH_CONTROL` @ `0x114c` in [NVIDIA open-gpu-doc `clc797.h`](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h)). | High |
| **1** | Almost certainly “mesh pipeline on” (task optional). No public Nintendo header naming `NVN_GEOMETRY_PROCESSING_MODE_*` was found. | Medium (inferred) |
| **2** | **Not documented** in any public NVN2 header, SMO NVN1 docs, or open-gpu-doc method field (hardware enable is a single bit). Do not hard-code “2 = task+mesh” without a live MKW/SDK dump of `mode` values. | Unknown |

**Recommendation:** treat mode as `0` vs `nonzero` for the first recording pass; log the raw `mode` from MKW-THWOMP runs (`nvn_mode_trace` already tags `NVN_MODE_GEOMETRY`) and fill the enum table from observation.

### `nvnCommandBufferDrawMeshTasks*` family

Present in this repo’s proc list (same set nxnative still partially stubs):

| NVN2 entry | Likely host mapping |
|---|---|
| `DrawMeshTasks` | 1D task count → `vkCmdDrawMeshTasksEXT(cmd, count, 1, 1)` (or NV `DrawMeshTasksNV(first, count)`) |
| `DrawMeshTasksIndirect` | GPU record `{count, first}` (OoT) → rewrite to EXT `{x,y,z}` or NV 1D form |
| `DrawMeshTasks3D` / `*3DIndirect` | 3D workgroup counts (no public game argument dump yet; nxnative leaves them red) |
| `MultiDrawMeshTasks*` / `*Count` | Multi-draw / count-buffer variants |

OpenGL / Vulkan references (API model, not Switch headers):

- [NV_mesh_shader](https://registry.khronos.org/OpenGL/extensions/NV/NV_mesh_shader.txt) — `DrawMeshTasksNV(first, count)`; optional task stage; mesh replaces VS/tess/GS.
- [VK_EXT_mesh_shader proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_mesh_shader.adoc) — `vkCmdDrawMeshTasksEXT(x,y,z)`, Indirect, IndirectCount.
- [VK_NV_mesh_shader](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_mesh_shader.html) — older NV draw path.

---

## Ampere hardware: mesh/task as reused VTG stages + ISBE

Mesa NVK reverse-engineering (best public Ampere/Turing mesh notes):

- [Mesh Shading Notes](https://docs.mesa3d.org/drivers/nvk/mesh_shading_notes.html)
- [ISBE Layout](https://docs.mesa3d.org/drivers/nvk/isbe_layout.html)
- [XDC2024 NVK Mesh Shading slides](https://indico.freedesktop.org/event/6/contributions/307/attachments/216/295/XDC2024-NVK-MeshShading.pdf)

### Method programming ([`clc797.h`](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h))

| Method | Offset | Role |
|---|---|---|
| `SET_MESH_CONTROL` | `0x114c` | bit0 enable |
| `SET_MESH_SHADER_A` | `0x1150` | output topology, max vertex, max primitive |
| `SET_MESH_SHADER_B` | `0x1154` | shared mem lines, thread count (≤32) |
| `SET_MESH_INIT_SHADER` | `0x1158` | task: thread count, local buffer lines, output-to-mesh lines |

### Stage binding while mesh is on (Eden `NX2_NOTES` row 89)

While `SET_MESH_CONTROL` is enabled:

- **Task** program → **vertex** slot  
- **Mesh** program → **tessellation** slot  
- **Header-only passthrough GS** when per-primitive outputs exist  
- Draw count = task workgroup count  

This matches Mesa: task always as VS; mesh as VS if alone, else tess control.

### Stage-memory ops (Ampere SASS)

| Opcode | Name | Role |
|---|---|---|
| `0x923` | `ISBERD` | read map / patch / primitive / attribute (modes + skew) |
| `0x927` | `ISBEWR` | write same regions (Maxwell has **no** write equivalent) |

Mesa lowering of EXT mesh builtins (from mesh notes):

- `EmitMeshTasksEXT(x,y,z)` → `ISBEWR.O.ATTR` at `0x04` (count), `0x08/0x0C/0x10` (xyz)  
- `SetMeshOutputsEXT(v,p)` → `ISBEWR.O.MAP.32 [0x3]` = primitive count; U8 indices from `+0x4`

Eden/NX2 row 89 (public mirror [Frizti/Fritchii `NX2_NOTES.md`](https://github.com/Frizti/Fritchii/blob/nx2/NX2_NOTES.md)):

- Task: `ISBERD.ATTR.SKEW [RZ]` = workgroup index; payload `ISBEWR.O.ATTR [rank*stride+0x10]`; mesh count at ATTR `[0x4]`  
- Mesh: input index; payload from `0x10`; vertices in groups of 32 (128-byte lines per component); map count `ISBEWR.O.32 [0x3]`, U8 indices from 4  
- Planned host path: keep ISBE as IR pseudo-ops → epilogue emits `EmitMeshTasks` / `SetMeshOutputs`; pipelines keyed by `SET_MESH_SHADER_A/B` + `SET_MESH_INIT_SHADER`

nxnative control-section decode (more NVN-specific than Eden’s GPU-method view) — [docs/reference/shaders.md § NVN2 task and mesh shaders](https://github.com/seedds/nxnative/blob/master/docs/reference/shaders.md):

- Stages **6 = task, 7 = mesh** (control `+0x56`; SPH still type 1 or 3)  
- Workgroup threads `+0x58`, payload bytes `+0x5a`, mesh-has-task `+0x5e`, verts/prim `+0x5f`, max verts/prims `+0x60/+0x62`  
- ISBERD modes: workgroup `0x78`, payload word `0x38`; ISBEWR modes: task payload `0xb8`, mesh attr `0xf8`, prim count byte `0x88` (+3), prim index byte `0x80` (from +4)  
- Host binds these stages in **tessellation slots 3 and 4** (`hle_nvn.c` `st_host`); `c0` SSBO tables at **`0x670` (task)** and **`0x770` (mesh)**  

Hookshot/mkw-recomp state comment (public paste): NVN2 task stage **5 or** mesh **7**, `kNvnStages = 9` — treat nxnative’s 6/7 + control `+0x56` as the more detailed source when implementing.

---

## What open-source projects do

### Eglantine / Eden (Vulkan, Switch 2)

| Behaviour | Evidence |
|---|---|
| **Native path:** `VK_EXT_mesh_shader` + `vkCmdDrawMeshTasksEXT` | [PR #426](https://github.com/Djihads80/Eglantine/pull/426), [PR #772](https://github.com/Eglantine-Community/Eglantine/pull/772), [issue #730](https://github.com/Djihads80/Eglantine/issues/730) (`vk_rasterizer.cpp` ~812, `vk_pipeline_cache.cpp` ~1346) |
| **No extension:** skip task/mesh pipelines / draws (black/empty targets) | same; confirmed on llvmpipe (software mesh) vs ANV/Pascal |
| **Kill switch:** `EDEN_NX2_NO_MESH` / `EDEN_NX2_FORCE_MESH` | [PR #782](https://github.com/Eglantine-Community/Eglantine/pull/782) (also disables mesh on RADV GFX12 hangs) |
| **ISBERD/ISBEWR:** required for MKW native programs; flicker when rejected | [issue #158](https://github.com/Djihads80/Eglantine/issues/158), [PR #173](https://github.com/Djihads80/Eglantine/pull/173); this tree’s `lower_to_maxwell.h` still drops `ISBEWR` |
| **Geometry-as-compute for task/mesh:** **planned, not shipped** | PR #772 / issue #730 propose combined compute → SSBO → `DrawIndexedIndirect`; NX2_NOTES row 89 records the ISBE layout that lowering needs |
| **MKW symptoms without mesh:** pop-in / flicker / low-quality LODs | NX2_NOTES row 97; [Wael3rd known-issues](https://github.com/Wael3rd/Eglantine-MKW/blob/mkw-eglantine/docs/nx2-mkw/known-issues.md) |

### nxnative (Metal, Switch 2) — **most complete public implementation**

Primary docs:

- [MKW profile README](https://github.com/seedds/nxnative/blob/master/dm/games/0400C3F00006E000/README.md)  
- [shaders.md — NVN2 task and mesh shaders](https://github.com/seedds/nxnative/blob/master/docs/reference/shaders.md)  
- [OoT foliage / DrawMeshTasksIndirect](https://github.com/seedds/nxnative/blob/master/docs/reference/shaders.md) (§ Task and mesh shaders in BNSH…)

| Piece | What they do |
|---|---|
| Classification | Split stages as `ts`/`ms` from control, not SPH |
| Translation | Body as compute with **emulated stage memory** in threadgroup shared (`dmsh_meshstage`, `AIsberd` / `AStageOut`) |
| Host wrap | Metal `[[object]]` (task) / `[[mesh]]` (mesh); mesh shares vertex arg table; task uses object stage |
| Draw | `mesh_tasks_draw` → Metal mesh draw; **`DrawMeshTasks` works** on MKW (7 task + 18 mesh = bushes/grass) |
| Indirect | OoT: GPU rewrite `{count,first}` → Metal `{count,1,1}` (`mdx_run`) |
| Still red | `DrawMeshTasks3D`, multi-draw forms (no observed args) |
| Binding footgun | Wrong `c0` SSBO bases (`0x610/0x710` vs real `0x670/0x770`) → dummy buffer → sky panels ([issue #1186](https://github.com/seedds/nxnative/issues/1186)) |
| Env | `DM_NVN_SKIP_MESH=1` drops all task/mesh draws for timing A/B |

**Important:** nxnative’s `DM_NVN_GEO_MESH` / `geo_mesh.tsv` path is **passthrough classical GS → Metal mesh** (Bananza terrain). That is a different feature from NVN2 `DrawMeshTasks`.

### MacNativeGames (Ryujinx + nxnative patches)

[docs/ns2-port.md](https://github.com/ly05010419/MacNativeGames/blob/main/docs/ns2-port.md):

- Phase 5 still lists “mesh geometry” for Bananza level bring-up.  
- **Geometry-as-compute** is implemented for Bananza’s **one classical GS** (`5d537f76`) via Ryujinx `Translate(asCompute)` — not NVN2 task/mesh.  
- Native Metal object/mesh for that GS path is “to compare later”.  
- OoT Switch 2 bring-up notes cite `DrawMeshTasksIndirect` as a next wall ([issue #1773](https://github.com/ly05010419/MacNativeGames/issues/1773)).

### kartnative / pahorema

[geometry-metal-research.md](https://github.com/pahorema/kartnative/blob/main/docs/geometry-metal-research.md): classical GS → **compute + ordinary draw** (chosen); mesh pipeline as optional optimization. Same distinction: GS emulation ≠ guest mesh shaders.

### Ryujinx / yuzu forks

- Classical GS → compute on Metal is upstream Ryujinx ([#5551](https://github.com/Ryujinx/Ryujinx/pull/5551) era).  
- Switch 2 Ampere task/mesh work lives in **nxnative’s Ryujinx fork / patches**, not stock Ryujinx.  
- No public Ryujinx-NS2 tree was found that independently implements `DrawMeshTasks` beyond what nxnative documents.

### Hookshot / mkw-recomp lineage

[NotRexed/Hookshot `docs/design.md`](https://github.com/NotRexed/Hookshot/blob/main/docs/design.md):

- Records NVN2 natively; Vulkan replay via **Eden’s shader recompiler** with **`VK_EXT_mesh_shader`** for task/mesh.  
- Closest architectural sibling to this repo’s intended end state (record ops → Eden-style SPIR-V mesh stages → EXT draws).

---

## Geometry-as-compute: when it is / isn’t the answer

| Workload | Preferred path | Compute fallback? |
|---|---|---|
| Classical GS (Bananza terrain, NieR index-build, etc.) | Metal mesh compose **or** VS+GS-as-compute → draw | **Yes — production** on nxnative / MacNativeGames / kartnative |
| NVN2 task + mesh (`DrawMeshTasks`) | `VK_EXT_mesh_shader` / Metal object+mesh | **Yes — designed, rarely shipped** (Eden #730 / NX2_NOTES 89). Current Eden behaviour without EXT: **skip** |

For MKW-THWOMP on a desktop Vulkan host with mesh shaders (NVIDIA Turing+, recent AMD/Intel with EXT): implement the **native mesh path first**. Keep compute lowering as the Pascal / no-EXT escape hatch.

---

## Concrete implementation sketch for this Vulkan host

### Phase A — stop dropping the API (unblocks diagnosis)

1. Record `OP_SET_GEOMETRY_PROCESSING_MODE { mode }` from `SetGeometryProcessingMode`.  
2. Record `OP_DRAW_MESH_TASKS` (and Indirect) with raw args; log `mode`, bound program keys, and counts for MKW-THWOMP.  
3. Keep replay as no-op until shaders exist — but the stream must show how often MKW hits nonzero mode + mesh draws.

### Phase B — identify and translate mesh programs

1. Extend program/control parsing for NVN2 control version `0xc9`: stage id at `+0x56`, mesh limits at `+0x5e..+0x62` (nxnative).  
2. Stop treating ISBEWR as a silent no-op for stages 6/7; keep ISBE ops through IR (Eden plan) **or** map onto EXT mesh builtins / shared payloads (nxnative plan).  
3. Pipeline key must include mesh topology / max verts / max prims / thread counts (`SET_MESH_SHADER_*` / control fields).

### Phase C — draw path

1. If `geometry_processing_mode != 0` and device has `VK_EXT_mesh_shader`:  
   - Bind task (`VK_SHADER_STAGE_TASK_BIT_EXT`) + mesh (`VK_SHADER_STAGE_MESH_BIT_EXT`) + FS.  
   - `vkCmdDrawMeshTasksEXT` / Indirect / IndirectCount.  
2. Else: skip with a clear once-per-key warning **or** dispatch the compute lowering from issue #730.  
3. Bind storage using **task/mesh c0 bases `0x670` / `0x770`** (nxnative dummy-buffer bug is the cautionary tale).

### Phase D — MKW validation targets

From nxnative MKW notes:

- 7 task + 18 mesh shaders → bushes / grass tufts.  
- Without them: missing foliage; with wrong SSBO bases: sky panels.  
- OoT: hundreds of foliage mesh shaders + `DrawMeshTasksIndirect` — good stress for Indirect rewrite.  
- Eden A/B: `EDEN_NX2_NO_MESH` world-map stall / empty targets when mesh is required.

---

## Links (quick index)

| Topic | URL |
|---|---|
| This host stubs | `runtime/nvn_api_nvn2.c`, `runtime/nvn_procs_nvn2.inc` |
| Local Ampere ISBE stubs | `runtime/third_party/eglantine/shader_recompiler/frontend/ampere/lower_to_maxwell.h` (`0x923`/`0x927`) |
| NVIDIA Ampere 3D methods | https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h |
| Mesa ISBE / mesh | https://docs.mesa3d.org/drivers/nvk/isbe_layout.html · https://docs.mesa3d.org/drivers/nvk/mesh_shading_notes.html |
| GL NV mesh | https://registry.khronos.org/OpenGL/extensions/NV/NV_mesh_shader.txt |
| VK EXT mesh proposal | https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_mesh_shader.adoc |
| Eden NX2_NOTES (row 89/97) | https://github.com/Frizti/Fritchii/blob/nx2/NX2_NOTES.md |
| Eden skip / diagnostics | https://github.com/Djihads80/Eglantine/pull/426 · https://github.com/Eglantine-Community/Eglantine/pull/772 · https://github.com/Djihads80/Eglantine/issues/730 |
| Eden ISBERD/ISBEWR | https://github.com/Djihads80/Eglantine/issues/158 · https://github.com/Djihads80/Eglantine/pull/173 |
| nxnative MKW | https://github.com/seedds/nxnative/blob/master/dm/games/0400C3F00006E000/README.md |
| nxnative task/mesh docs | https://github.com/seedds/nxnative/blob/master/docs/reference/shaders.md |
| MacNativeGames NS2 plan | https://github.com/ly05010419/MacNativeGames/blob/main/docs/ns2-port.md |
| kartnative GS research | https://github.com/pahorema/kartnative/blob/main/docs/geometry-metal-research.md |
| Hookshot design | https://github.com/NotRexed/Hookshot/blob/main/docs/design.md |

---

## Open questions (need a live MKW-THWOMP trace)

1. Exact `SetGeometryProcessingMode` values MKW uses (only `0` vs `1`, or also `2+`?).  
2. Which of `DrawMeshTasks` vs `DrawMeshTasksIndirect` dominates foliage/scenery on THWOMP.  
3. Whether any THWOMP path hits `DrawMeshTasks3D` / Multi* (still undocumented args).  
4. Whether this host’s current `ISBEWR` drop already rejects the same MKW programs Eden rejected before PR #173, independent of draw recording.

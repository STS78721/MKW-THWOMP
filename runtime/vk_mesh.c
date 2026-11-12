// NVN2 DrawMeshTasks replay: Eden task/mesh SPIR-V + VK_EXT_mesh_shader.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eden_mesh.h"
#include "gfx.h"
#include "host.h"
#include "mesh_stub/stub_spv.h"
#include "nvn_obj.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_mesh.h"
#include "cfg.h"
#include "vk_rec.h"

// Keyed by gfx_shader_nvn2_hash of the raw DKMESH Ampere stages (nvn2_origin is 0: they are never lowered).
// Adopted SPIR-V is kept for the process lifetime: VkPipeBinding.res points into it.
#define MESH_SHADERS 256

typedef struct {
    uint64_t task_o, mesh_o;
    const GfxShader *task, *mesh;
    int failed;   // Eden translate or a prior pipeline build failed for this pair
} MeshShaders;

static MeshShaders mesh_shaders[MESH_SHADERS];
static int mesh_log_ok, mesh_log_fail;

int vk_mesh_available(void) {
    VkCtx *c = vk_ctx();
    return c && c->has_mesh_shader;
}

void vk_mesh_op_geo_mode(VkExec *ex, const void *payload) {
    ex->geo_mode = ((const p_geo_mode *)payload)->mode;
}

static int is_dkmesh(const uint8_t *ctl) { return ctl && !memcmp(ctl + 0x870, "DKMESH", 6); }

static int stage_nvn_type(const uint8_t *ctl) {
    uint16_t st;
    memcpy(&st, ctl + 0x56, 2);
    return (int)st;
}

// SET_MESH_SHADER_A/B + SET_MESH_INIT_SHADER packing (Hookshot vk_shader.cpp mesh_registers_of).
static void mesh_regs_from_controls(const uint8_t *task_ctl, const uint8_t *mesh_ctl, uint32_t regs[3]) {
    uint16_t mesh_thr = 0, task_thr = 0, payload = 0, max_prim = 0, max_vert = 0;
    uint8_t topo = 0;
    memcpy(&mesh_thr, mesh_ctl + 0x58, 2);
    memcpy(&max_prim, mesh_ctl + 0x60, 2);
    memcpy(&max_vert, mesh_ctl + 0x62, 2);
    topo = mesh_ctl[0x5f];
    if (topo == 3) topo = 4; // NVN triangles -> Maxwell topology 4
    if (task_ctl) {
        memcpy(&task_thr, task_ctl + 0x58, 2);
        memcpy(&payload, task_ctl + 0x5a, 2);
    }
    uint32_t lines = task_ctl ? (payload + 127u) / 128u : 0u;
    uint32_t most = max_vert > max_prim ? max_vert : max_prim;
    if (most > 0x100u) most = 0x100u;
    if (lines > 0x1ffu) lines = 0x1ffu;
    uint32_t mt = mesh_thr > 0x3fu ? 0x3fu : mesh_thr;
    uint32_t tt = task_thr > 0x3fu ? 0x3fu : task_thr;
    regs[0] = (uint32_t)(topo & 0xf) | (most << 4) | (most << 13);
    regs[1] = mt << 20;
    regs[2] = (task_ctl ? tt : 0u) | (lines << 12) | (lines << 22);
}

// Raw DKMESH Ampere code never goes through nvn2_lower, so nvn2_origin(code) is always 0.
// Key and adopt-origin must be the NVN2 content hash instead.
static uint64_t mesh_stage_key(const uint8_t *ctl, const uint8_t *code) {
    uint64_t h = gfx_shader_nvn2_hash(ctl, code);
    // n238: real MeshPrimitiveHeader from len=0 header-only stage (was misread as FS).
    return (h ? h : (uint64_t)(uintptr_t)code) ^ 0x4ull;
}

// ProgramSetShaders gives every successful stage replacement a new serial and owns the task/mesh controls. Shader
// code remains valid and immutable for that program lifetime, as for vk_exec_draw.c's serial-keyed shader table.
// Keep only the raw content keys here: the primitive SPH is a separate translation input and is hashed every draw.
// A collision simply hashes again; zero-serial programs retain the full-content lookup on every call.
#define MESH_KEY_MEMOS 512
static struct {
    uint64_t serial, key[2];
    const uint8_t *ctl[2], *code[2];
    uint8_t valid;
} mesh_key_memos[MESH_KEY_MEMOS];

static uint64_t mesh_program_stage_key(const prog_t *p, unsigned stage, const uint8_t *ctl, const uint8_t *code) {
    if (!p || !p->serial || stage >= 2) return mesh_stage_key(ctl, code);
    uint32_t index = (uint32_t)(p->serial * 0x9E3779B97F4A7C15ull >> 55);
    typeof(mesh_key_memos[0]) *m = &mesh_key_memos[index];
    if (m->serial != p->serial) m->serial = p->serial, m->valid = 0;
    if ((m->valid & (1u << stage)) && m->ctl[stage] == ctl && m->code[stage] == code) return m->key[stage];
    uint64_t key = mesh_stage_key(ctl, code);
    m->ctl[stage] = ctl, m->code[stage] = code, m->key[stage] = key;
    m->valid |= (uint8_t)(1u << stage);
    return key;
}

static void mesh_dump_spirv(uint64_t key, int is_task, const uint32_t *spirv, uint32_t words) {
    static int dumped;
    if (dumped >= 4 || !spirv || !words) return;
    const char *dir = getenv("TOTK_MESH_DUMP");
    if (!dir || !*dir) dir = getenv("MKW_MESH_DUMP");
    if (!dir || !*dir) return;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s_%016llx.spv", dir, is_task ? "task" : "mesh", (unsigned long long)key);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(spirv, 4, words, f);
    fclose(f);
    dumped++;
    rt_log("vk_mesh: wrote %s (%u words)", path, words);
}

static const GfxShader *translate_mesh_stage(const uint8_t *ctl, const uint8_t *code, uint32_t len, int is_task,
                                             const uint32_t mesh_regs[3], const uint8_t *prim_sph) {
    uint32_t *spirv = NULL;
    uint32_t words = 0;
    EdenMeshSsbo ssbos[EDEN_MESH_SSBO_MAX];
    EdenMeshCbuf cbufs[EDEN_MESH_CBUF_MAX];
    EdenMeshTex texs[EDEN_MESH_TEX_MAX];
    uint32_t n_ssbos = 0, n_cbufs = 0, n_texs = 0;
    if (eden_mesh_translate(ctl, code, len, is_task, mesh_regs, is_task ? NULL : prim_sph, &spirv, &words, ssbos,
                            &n_ssbos, cbufs, &n_cbufs, texs, &n_texs) != 0)
        return NULL;
    uint64_t key = mesh_stage_key(ctl, code);
    mesh_dump_spirv(key, is_task, spirv, words);
    const GfxShader *s =
        gfx_shader_adopt_spirv(is_task ? GFX_STAGE_TASK : GFX_STAGE_MESH, spirv, words, key);
    eden_mesh_free(spirv);
    if (s) {
        gfx_shader_apply_eden_ssbos(s, ssbos, n_ssbos);
        gfx_shader_apply_eden_cbufs(s, cbufs, n_cbufs);
        gfx_shader_apply_eden_texs(s, texs, n_texs);
        if (mesh_log_ok < 4 && (n_ssbos || n_cbufs || n_texs)) {
            char buf[160];
            size_t n = 0;
            for (uint32_t i = 0; i < n_ssbos && n + 24 < sizeof buf; i++)
                n += (size_t)snprintf(buf + n, sizeof buf - n, "%s%#x×%u", i ? "," : "", ssbos[i].cbuf_offset,
                                      ssbos[i].count);
            char cb[80];
            size_t cn = 0;
            for (uint32_t i = 0; i < n_cbufs && cn + 12 < sizeof cb; i++)
                cn += (size_t)snprintf(cb + cn, sizeof cb - cn, "%sc%u", i ? "," : "", cbufs[i].cbuf_index);
            rt_log("vk_mesh: %s Eden SSBOs %u (%s) cbufs %u (%s) texs %u", is_task ? "task" : "mesh", n_ssbos, buf,
                   n_cbufs, cb, n_texs);
        }
    }
    return s;
}

// True when any component of generic g is enabled in an SPH omap_generic_vector (byte offset 54).
static int sph_generic_any(const uint8_t *sph, int g) {
    if (!sph || g < 0 || g >= 32) return 0;
    int data = sph[54 + (g >> 1)] >> ((g & 1) * 4);
    return (data & 0xf) != 0;
}

// SPH usable for AddMeshLines generics: any omap_generic bit (offset 54..69), or systemb (offset 53)
// bits AddMeshLines understands. Word0 alone (type/version) is NOT enough — MKW's len=0 stage 0 often
// has that with an empty omap, which left prim_sph=1 but still mout holes (n256).
static int sph_has_mesh_outputs(const uint8_t *sph) {
    if (!sph) return 0;
    if (sph[53]) return 1;
    for (int i = 0; i < 16; i++)
        if (sph[54 + i]) return 1;
    return 0;
}

// Hookshot: a header-only graphics stage (code len ≤ SPH+16) is NVN2's per-primitive output map.
// Prefer end (+0x44) ≥ SPH. MKW foliage often has type 0 / len 0 / end 0 — accept data only when the
// SPH actually enables outputs (n256: "live" word0 with empty omap was a false positive).
static const uint8_t *find_mesh_prim_sph(const prog_t *p) {
    if (!p) return NULL;
    enum { SPH = 0x50, HDR_ONLY = SPH + 16 };
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        const uint8_t *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        if (!c || !code) continue;
        uint32_t host = 0, len = 0, end = 0;
        memcpy(&host, c + 0x714, 4);
        memcpy(&len, c + 0x38, 4);
        memcpy(&end, c + 0x44, 4);
        int nvn = stage_nvn_type(c);
        if (nvn == 5 || nvn == 6 || nvn == 7) continue; // compute / task / mesh
        if (is_dkmesh(c)) continue;
        if (nvn <= 4 && len <= HDR_ONLY) {
            if (end >= SPH && sph_has_mesh_outputs(code)) return code;
            if (end < SPH && sph_has_mesh_outputs(code)) {
                static int live_log;
                if (live_log++ < 4)
                    rt_log("vk_mesh: MeshPrimitiveHeader from len=%u end=%u live omap (stage %u)", len, end, i);
                return code;
            }
        }
        if ((host == 2u || nvn == 2) && end >= SPH && sph_has_mesh_outputs(code)) return code;
    }
    return NULL;
}

// MKW mesh ProgramSetShaders parks an empty type-0 stage (no Geometry.reserved tracking yet).
// Vertex SPH enables high generics (foliage: 4+5 → mout 0x30); FS still wants 0..3 as per-primitive.
// Synthesize a prim SPH that enables every generic below the first vertex generic — matching the
// usual NVN packing and what Eglantine would load from Geometry.reserved for those programs.
static const uint8_t *synth_prim_sph_from_mesh(const uint8_t *mcode, uint32_t mlen) {
    static uint8_t synth[0x50];
    if (!mcode || mlen < 0x50) return NULL;
    int min_vert = 32;
    for (int g = 0; g < 32; g++)
        if (sph_generic_any(mcode, g) && g < min_vert) min_vert = g;
    if (min_vert <= 0 || min_vert >= 32) return NULL;
    memset(synth, 0, sizeof synth);
    // SPH type 2 (VTG), version 4 — same shape Maxwell mesh headers use.
    uint32_t w0 = 2u | (4u << 5);
    memcpy(synth, &w0, 4);
    for (int g = 0; g < min_vert; g++)
        synth[54 + (g >> 1)] = (uint8_t)(synth[54 + (g >> 1)] | (0xfu << ((g & 1) * 4)));
    static int synth_log;
    if (synth_log++ < 6)
        rt_log("vk_mesh: synth MeshPrimitiveHeader generics 0..%d (mesh vertex min=%d)", min_vert - 1, min_vert);
    return synth;
}

static uint64_t prim_sph_key(const uint8_t *prim) {
    if (!prim) return 0;
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 0x50; i++) h = (h ^ prim[i]) * 0x100000001b3ull;
    return h;
}

// True when the program's host-slot stage is a raw NVN2 header-only SPH (not real shader code).
static int stage_is_header_only(const prog_t *p, int host_slot) {
    const uint8_t *ctl = NULL, *code = NULL;
    uint32_t flen = 0;
    if (!vk_exec_find_stage(p, host_slot, &ctl, &code, &flen) || !ctl) return 0;
    if (!gfx_shader_is_nvn2(ctl)) return 0; // Maxwell / lowered FS is real
    uint32_t nvn_len = 0;
    memcpy(&nvn_len, ctl + 0x38, 4);
    int nvn = stage_nvn_type(ctl);
    return nvn <= 4 && nvn_len <= 0x50u + 16u;
}

static int find_mesh_stages(const prog_t *p, const uint8_t **task_ctl, const uint8_t **task_code, uint32_t *task_len,
                            const uint8_t **mesh_ctl, const uint8_t **mesh_code, uint32_t *mesh_len,
                            const uint8_t **prim_sph) {
    // Host slots after st_host / prog_set_shaders: task→3, mesh→4 (tessellation places).
    // Fragment comes from stage_prog[FS] (prior BindProgram), not from the mesh program — MKW's
    // mesh ProgramSetShaders parks a header-only MeshPrimitiveHeader in host slot 1.
    enum { HOST_TS = 3, HOST_MS = 4 };
    if (prim_sph) *prim_sph = NULL;
    if (vk_exec_find_stage(p, HOST_TS, task_ctl, task_code, task_len) && is_dkmesh(*task_ctl) &&
        stage_nvn_type(*task_ctl) == 6) { /* optional task */ } else {
        *task_ctl = NULL;
        *task_code = NULL;
        *task_len = 0;
    }
    int ok_mesh = vk_exec_find_stage(p, HOST_MS, mesh_ctl, mesh_code, mesh_len) && is_dkmesh(*mesh_ctl) &&
                  stage_nvn_type(*mesh_ctl) == 7;
    if (ok_mesh && prim_sph) {
        *prim_sph = find_mesh_prim_sph(p);
        static int prim_log;
        if (prim_log < 6) {
            if (*prim_sph)
                rt_log("vk_mesh: MeshPrimitiveHeader from header-only stage (prim_sph=1)");
            else {
                char inv[160];
                size_t n = 0;
                for (uint32_t i = 0; i < p->nstage && i < 6 && n + 36 < sizeof inv; i++) {
                    const uint8_t *c = p->sd[i].control;
                    uint32_t host = 0, len = 0, end = 0;
                    if (c) {
                        memcpy(&host, c + 0x714, 4);
                        memcpy(&len, c + 0x38, 4);
                        memcpy(&end, c + 0x44, 4);
                    }
                    n += (size_t)snprintf(inv + n, sizeof inv - n, "%s%u:h%u/t%d/l%u/e%u", n ? " " : "", i, host,
                                          c ? stage_nvn_type(c) : -1, len, end);
                }
                rt_log("vk_mesh: no MeshPrimitiveHeader (nstage=%u [%s])", p->nstage, inv);
            }
            prim_log++;
        }
    }
    return ok_mesh;
}

static MeshShaders *mesh_shaders_get(uint64_t task_o, uint64_t mesh_o) {
    for (int i = 0; i < MESH_SHADERS; i++)
        if (mesh_shaders[i].mesh_o == mesh_o && mesh_shaders[i].task_o == task_o &&
            (mesh_shaders[i].task || mesh_shaders[i].mesh || mesh_shaders[i].failed))
            return &mesh_shaders[i];
    return NULL;
}

static MeshShaders *mesh_shaders_put(uint64_t task_o, uint64_t mesh_o, const GfxShader *task, const GfxShader *mesh,
                                     int failed) {
    MeshShaders *slot = NULL;
    for (int i = 0; i < MESH_SHADERS; i++) {
        if (mesh_shaders[i].mesh_o == mesh_o && mesh_shaders[i].task_o == task_o) {
            slot = &mesh_shaders[i];
            break;
        }
        if (!slot && !mesh_shaders[i].mesh_o && !mesh_shaders[i].task_o && !mesh_shaders[i].failed &&
            !mesh_shaders[i].task && !mesh_shaders[i].mesh)
            slot = &mesh_shaders[i];
    }
    if (!slot) {
        static int full_log;
        if (full_log < 2) rt_log("vk_mesh: shader keep table full (%d)", MESH_SHADERS), full_log++;
        return NULL;
    }
    // Never free a previously kept shader: pipeline bindings may still reference it.
    if (!slot->task) slot->task = task;
    else if (task && task != slot->task) gfx_shader_release_adopted(task);
    if (!slot->mesh) slot->mesh = mesh;
    else if (mesh && mesh != slot->mesh) gfx_shader_release_adopted(mesh);
    slot->task_o = task_o;
    slot->mesh_o = mesh_o;
    slot->failed = failed;
    return slot;
}

static int mesh_draw_cmds_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("MESH_DRAW", "1");
    return on;
}

// TOTK_MESH_EMPTY=1: record DrawMeshTasks(0,0,0) to test the entry point without shader work.
static int mesh_empty_draw(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("MESH_EMPTY", "0");
    return on;
}

// NVN indirect mesh args are {count, first} (8 bytes). VK_EXT_mesh_shader wants {x,y,z}. Feeding the NVN
// buffer as-is sets groupCountY = first (usually 0) → zero workgroups, silent empty draws (missing foliage /
// props / terrain strips). Rewrite into a large ring of {count, 1, 1} so multi-frame in-flight draws never
// overwrite earlier frames' slots or step past the end of mesh_args_buf.
enum { MESH_ARGS = 1u << 18 };
static VkBuffer mesh_args_buf;
static VmaAllocation mesh_args_alloc;
static uint32_t *mesh_args_map;
static uint32_t mesh_args_next;

static int mesh_args_ready(void) {
    if (mesh_args_buf) return 1;
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = (VkDeviceSize)MESH_ARGS * 12u,
                             .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &mesh_args_buf, &mesh_args_alloc, &info) != VK_SUCCESS) return 0;
    mesh_args_map = info.pMappedData;
    for (uint32_t i = 0; i < MESH_ARGS; i++)
        mesh_args_map[i * 3] = 0, mesh_args_map[i * 3 + 1] = 1, mesh_args_map[i * 3 + 2] = 1;
    return 1;
}

static void mesh_issue_draw(VkExec *ex, const p_draw_mesh *d, const VkPipeState *ps, VkPipeline pipe,
                            const VkPipeInfo *info) {
    // Must go through vk_rec: a direct vkCmdDrawMeshTasksEXT on the replay thread races the recorder
    // thread on the same command pool (validation THREADING ERROR → device lost).
    if (!mesh_draw_cmds_on()) {
        if (mesh_log_ok < 4)
            rt_log("vk_mesh: DrawMeshTasks skipped (TOTK_MESH_DRAW=0) program %llu",
                   (unsigned long long)ex->prog->serial),
                mesh_log_ok++;
        return;
    }
    if (mesh_empty_draw()) {
        rec_draw_mesh_tasks(ex->cb, 0, 0, 0);
        ex->recorded++;
        if (mesh_log_ok < 4)
            rt_log("vk_mesh: DrawMeshTasks empty (0,0,0) program %llu", (unsigned long long)ex->prog->serial),
                mesh_log_ok++;
        return;
    }
    if (d->kind == 0) {
        uint32_t count = d->count > 0 ? (uint32_t)d->count : 1u;
        rec_draw_mesh_tasks(ex->cb, count, 1, 1);
        ex->recorded++;
        vk_drawstats_mesh(ex, count);
        // Cap hard: a console-attached boot was scrolling these for seconds and looked frozen.
        if (mesh_log_ok < 4)
            rt_log("vk_mesh: DrawMeshTasks count=%u program %llu", count, (unsigned long long)ex->prog->serial),
                mesh_log_ok++;
        vk_exec_frame.draws++;
    } else if (d->kind == 1) {
        VkMemSpan sp;
        if (!vk_mem_resolve(d->indirect, 8, &sp)) {
            if (mesh_log_fail < 4) rt_log("vk_mesh: indirect draw buffer not backed"), mesh_log_fail++;
            return;
        }
        // The count word is usually written by the game's culling / LOD compute shader in this same submission, so the
        // CPU copy read here at record time is the previous frame's (or older): foliage drawn with a stale count blinks
        // and swaps LOD. Default: always take the count on the GPU (below). MKW_MESH_CPU_COUNT=1: the old CPU read,
        // for A/B tests.
        static int cpu_count = -1;
        if (cpu_count < 0) {
            const char *e = cfg_get("MKW_MESH_CPU_COUNT");
            cpu_count = e && *e == '1';
            rt_log("vk_mesh: DrawMeshTasksIndirect counts are read %s", cpu_count ? "on the CPU at record time (MKW_MESH_CPU_COUNT=1)"
                                                                                  : "by the GPU at draw time");
        }
        uint32_t count = 0;
        if (cpu_count && !sp.alloc && sp.avail >= 8)
            memcpy(&count, (const void *)(uintptr_t)d->indirect, 4);
        vk_drawstats_mesh(ex, count);
        if (count) {
            // CPU-visible NVN record: draw direct as {count,1,1}.
            rec_draw_mesh_tasks(ex->cb, count, 1, 1);
            ex->recorded++;
            vk_exec_frame.draws++;
            if (mesh_log_ok < 4)
                rt_log("vk_mesh: DrawMeshTasksIndirect → direct count=%u program %llu", count,
                       (unsigned long long)ex->prog->serial),
                    mesh_log_ok++;
            return;
        }
        // GPU-written count (culling): copy word0 into a {count,1,1} slot, then indirect-draw that slot.
        if (!mesh_args_ready()) {
            if (mesh_log_fail < 4) rt_log("vk_mesh: mesh args buffer unavailable"), mesh_log_fail++;
            return;
        }
        if (mesh_args_next >= MESH_ARGS) mesh_args_next = 0;
        uint32_t slot = mesh_args_next++;
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        VkBufferCopy cp = {.srcOffset = sp.offset, .dstOffset = (VkDeviceSize)slot * 12u, .size = 4};
        VK_MEM_USE(sp);
        vkCmdCopyBuffer(ex->cb, sp.buffer, mesh_args_buf, 1, &cp);
        // y/z stay 1 from mesh_args_ready init; count word may be GPU-written this submit.
        vk_exec_barrier(ex);
        vk_exec_pass_begin(ex);
        if (ps && pipe && info) vk_exec_mesh_bind(ex, ps, pipe, info);
        rec_draw_mesh_tasks_indirect(ex->cb, mesh_args_buf, (VkDeviceSize)slot * 12u, 1, 12);
        ex->recorded++;
        vk_exec_frame.draws++;
        vk_drawstats_count(ex, &sp);   // the GPU-written group count, read back for MKW_DRAWSTATS
        if (mesh_log_ok < 8)
            rt_log("vk_mesh: DrawMeshTasksIndirect rewritten slot %u program %llu", slot,
                   (unsigned long long)ex->prog->serial),
                mesh_log_ok++;
    } else if (d->kind == 2) {
        // Multi*Count: same {count,first} vs {x,y,z} mismatch per record. Rewrite up to MESH_ARGS entries.
        uint32_t n = d->max > 0 ? (uint32_t)d->max : 1u;
        uint32_t stride = d->stride > 0 ? (uint32_t)d->stride : 8u;
        if (n > MESH_ARGS) n = MESH_ARGS;
        VkMemSpan sp, cp;
        if (!vk_mem_resolve(d->indirect, (uint64_t)n * stride, &sp) || !vk_mem_resolve(d->count_addr, 4, &cp)) {
            if (mesh_log_fail < 4) rt_log("vk_mesh: indirect-count draw buffers not backed"), mesh_log_fail++;
            return;
        }
        if (!mesh_args_ready()) {
            if (mesh_log_fail < 4) rt_log("vk_mesh: mesh args buffer unavailable"), mesh_log_fail++;
            return;
        }
        if (mesh_args_next + n > MESH_ARGS) mesh_args_next = 0;
        uint32_t slot = mesh_args_next;
        mesh_args_next += n;
        if (mesh_args_next >= MESH_ARGS) mesh_args_next = 0;
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        VK_MEM_USE(sp);
        VK_MEM_USE(cp);
        VkBufferCopy stack_bcp[256];
        VkBufferCopy *bcp = n <= 256 ? stack_bcp : malloc((size_t)n * sizeof *bcp);
        if (bcp) {
            for (uint32_t i = 0; i < n; i++)
                bcp[i] = (VkBufferCopy){.srcOffset = sp.offset + (VkDeviceSize)i * stride,
                                        .dstOffset = (VkDeviceSize)(slot + i) * 12u, .size = 4};
            vkCmdCopyBuffer(ex->cb, sp.buffer, mesh_args_buf, n, bcp);
            if (bcp != stack_bcp) free(bcp);
        }
        vk_exec_barrier(ex);
        vk_exec_pass_begin(ex);
        if (ps && pipe && info) vk_exec_mesh_bind(ex, ps, pipe, info);
        rec_draw_mesh_tasks_indirect_count(ex->cb, mesh_args_buf, (VkDeviceSize)slot * 12u, cp.buffer, cp.offset, n, 12);
        ex->recorded++;
        vk_exec_frame.draws++;
    } else if (mesh_log_fail < 4)
        rt_log("vk_mesh: draw kind %d not implemented", d->kind), mesh_log_fail++;
}

void vk_mesh_op_draw(VkExec *ex, const void *payload) {
    const p_draw_mesh *d = payload;
    if (!vk_mesh_available() || ex->render_disabled || !ex->prog) return;
    if (ex->geo_mode == 0 && d->kind == 0 && d->first != 0 && mesh_log_fail < 4)
        rt_log("vk_mesh: DrawMeshTasks first=%d ignored (Hookshot uses count only)", d->first), mesh_log_fail++;

    // TOTK_MESH_STUB=1: trivial glslang mesh+frag (no Eden). Isolates pipeline/bind faults from Ampere SPIR-V.
    static int mesh_stub = -1;
    if (mesh_stub < 0) mesh_stub = rt_knob("MESH_STUB", "0");
    if (mesh_stub) {
        static MeshShaders stub;
        if (!stub.mesh) {
            stub.mesh = gfx_shader_adopt_spirv(GFX_STAGE_MESH, stub_mesh_spv, stub_mesh_words, 1);
            stub.task_o = 0;
            stub.mesh_o = 1;
            if (!stub.mesh) {
                if (mesh_log_fail < 4) rt_log("vk_mesh: stub mesh adopt failed"), mesh_log_fail++;
                return;
            }
        }
        static const GfxShader *stub_fs;
        if (!stub_fs) {
            stub_fs = gfx_shader_adopt_spirv(GFX_STAGE_FS, stub_frag_spv, stub_frag_words, 2);
            if (!stub_fs) {
                if (mesh_log_fail < 4) rt_log("vk_mesh: stub fragment adopt failed"), mesh_log_fail++;
                return;
            }
        }
        if (!vk_exec_mesh_prepare(ex)) return;
        VkPipeState ps = ex->pipe;
        ps.vs = NULL;
        ps.gs = NULL;
        ps.fs = stub_fs;
        ps.prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
        ps.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
        ps.y_flip = 1;
        static int mesh_discard = -1;
        if (mesh_discard < 0) mesh_discard = rt_knob("MESH_DISCARD", "0");
        if (mesh_discard) ps.raster_discard = 1;
        const VkPipeInfo *info = NULL;
        VkPipeline pipe = vk_pipeline_mesh_graphics(&ps, NULL, stub.mesh, &info);
        if (!pipe || !info) {
            if (mesh_log_fail < 8) rt_log("vk_mesh: stub pipeline creation failed"), mesh_log_fail++;
            return;
        }
        vk_exec_mesh_bind(ex, &ps, pipe, info);
        mesh_issue_draw(ex, d, &ps, pipe, info);
        return;
    }

    const uint8_t *tctl, *tcode, *mctl, *mcode, *prim_sph;
    uint32_t tlen, mlen;
    if (!find_mesh_stages(ex->prog, &tctl, &tcode, &tlen, &mctl, &mcode, &mlen, &prim_sph)) {
        if (mesh_log_fail < 8) rt_log("vk_mesh: program %llu missing NVN2 task/mesh stages",
                                      (unsigned long long)ex->prog->serial), mesh_log_fail++;
        return;
    }

    static int mesh_no_task = -1;
    if (mesh_no_task < 0) mesh_no_task = rt_knob("MESH_NO_TASK", "0");
    if (mesh_no_task) tctl = NULL, tcode = NULL, tlen = 0;

    // When ProgramSetShaders has no live header-only SPH (MKW len=0/end=0/zeros), invent one from the
    // mesh vertex omap gap so AddMeshLines emits per-primitive Locations FS expects (foliage Loc 0-3).
    int prim_synth = 0;
    if (!prim_sph) {
        prim_sph = synth_prim_sph_from_mesh(mcode, mlen);
        if (prim_sph) prim_synth = 1;
    }

    uint64_t task_o = tcode ? mesh_program_stage_key(ex->prog, 0, tctl, tcode) : 0;
    // Fold prim SPH into the cache key — otherwise a prior no-prim translate sticks for the process.
    uint64_t mesh_o = mesh_program_stage_key(ex->prog, 1, mctl, mcode) ^ prim_sph_key(prim_sph) ^
                      (prim_synth ? 0x5052494dull : 0);
    MeshShaders *sh = mesh_shaders_get(task_o, mesh_o);
    if (sh && sh->failed) return;

    if (!sh || !sh->mesh) {
        uint32_t mesh_regs[3];
        mesh_regs_from_controls(tctl, mctl, mesh_regs);
        // mesh_regs_from_controls already holds max(verts,prims) to 0x100 (Hookshot mesh_registers_of).
        // An earlier 128 clamp left meshlets short of primitives (holes in foliage / props) once banks and
        // the c0+0x880 pointer table stopped the guest-memory corruption that motivated it.
        if (mesh_log_ok < 4)
            rt_log("vk_mesh: regs %08x %08x %08x task=%d mesh_o=%016llx max_out=%u prim_sph=%d synth=%d",
                   mesh_regs[0], mesh_regs[1], mesh_regs[2], tctl != NULL, (unsigned long long)mesh_o,
                   (mesh_regs[0] >> 4) & 0x1ffu, prim_sph != NULL, prim_synth);
        const GfxShader *task = NULL, *mesh = NULL;
        if (tctl && tcode) {
            task = translate_mesh_stage(tctl, tcode, tlen, 1, mesh_regs, NULL);
            if (!task && mesh_log_fail < 8)
                rt_log("vk_mesh: task Eden translate failed origin %016llx", (unsigned long long)task_o),
                    mesh_log_fail++;
        }
        // Pass Geometry-stage SPH (header-only or synth) for per-primitive output lines (Eglantine).
        mesh = translate_mesh_stage(mctl, mcode, mlen, 0, mesh_regs, prim_sph);
        if (!mesh) {
            if (task) gfx_shader_release_adopted(task);
            mesh_shaders_put(task_o, mesh_o, NULL, NULL, 1);
            if (mesh_log_fail < 8)
                rt_log("vk_mesh: mesh Eden translate failed origin %016llx", (unsigned long long)mesh_o),
                    mesh_log_fail++;
            return;
        }
        sh = mesh_shaders_put(task_o, mesh_o, task, mesh, 0);
        if (!sh) {
            gfx_shader_release_adopted(mesh);
            if (task) gfx_shader_release_adopted(task);
            return;
        }
    }

    if (!vk_exec_mesh_prepare(ex)) return;
    // Fragment from the last real FS program (cross-CB), not the mesh program's header-only SPH.
    int has_fs = 0;
    const GfxShader *fs = NULL;
    {
        const prog_t *fs_prog = vk_exec_fragment_prog(ex);
        if (fs_prog && !stage_is_header_only(fs_prog, 1)) {
            const prog_t *saved = ex->prog;
            ex->prog = fs_prog;
            fs = vk_exec_shader_for(ex, 1 /* NVN fragment */, &has_fs);
            ex->prog = saved;
        }
    }
    if (!has_fs || !fs) {
        if (mesh_log_fail < 8)
            rt_log("vk_mesh: fragment stage missing for mesh draw (stage_prog FS unset)"), mesh_log_fail++;
        return;
    }
    // Mesh→FS Location mismatch leaves FS sampling unset varyings (black foliage silhouettes).
    {
        static int iface_log;
        if (iface_log < 8) {
            GfxShaderIo mio[32], fio[32];
            uint32_t mout = gfx_shader_outputs(sh->mesh, mio);
            uint32_t fin = gfx_shader_inputs(fs, fio);
            uint32_t miss = fin & ~mout;
            if (miss || iface_log < 4)
                rt_log("vk_mesh: iface mesh_out=%#x fs_in=%#x missing=%#x program %llu", mout, fin, miss,
                       (unsigned long long)ex->prog->serial);
            iface_log++;
        }
    }

    VkPipeState ps = ex->pipe;
    ps.vs = NULL;
    ps.gs = NULL;
    ps.fs = fs;
    ps.prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
    ps.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    ps.y_flip = 1;
    // Opt-in: rasterizer-discard mesh draws to isolate task/mesh GPU faults from the fragment path.
    static int mesh_discard = -1;
    if (mesh_discard < 0) mesh_discard = rt_knob("MESH_DISCARD", "0");
    if (mesh_discard) ps.raster_discard = 1;
    const VkPipeInfo *info = NULL;
    VkPipeline pipe = vk_pipeline_mesh_graphics(&ps, sh->task, sh->mesh, &info);
    if (!pipe || !info) {
        // Not sticky: fragment answers or pipe state may not be ready this frame.
        if (mesh_log_fail < 8) rt_log("vk_mesh: pipeline creation failed origin %016llx", (unsigned long long)mesh_o),
                                   mesh_log_fail++;
        return;
    }

    vk_exec_mesh_bind(ex, &ps, pipe, info);
    mesh_issue_draw(ex, d, &ps, pipe, info);
}

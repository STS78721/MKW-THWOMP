// NVN2 task/mesh stages -> SPIR-V via Eden's recompiler (runtime/third_party/eden, GPL-3.0-or-later).
#include "eden_mesh.h"

#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

extern "C" void rt_log(const char *fmt, ...);

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/stage.h"

namespace {

constexpr uint32_t kControlMagic = 0x98761234;
constexpr uint32_t kControlSize = 0xd8;
constexpr uint32_t kAmpereHeader = 128;

struct StageCode {
    std::vector<uint64_t> code;
    std::vector<uint8_t> c1;
    uint32_t texture_bound = ~0u;
    uint32_t shared_memory = 0;
};

bool control_ok(const uint8_t *ctl, uint32_t code_size, uint32_t *len_out) {
    uint32_t magic, len, c1o, end;
    uint16_t st;
    std::memcpy(&magic, ctl, 4);
    std::memcpy(&len, ctl + 0x38, 4);
    std::memcpy(&c1o, ctl + 0x40, 4);
    std::memcpy(&end, ctl + 0x44, 4);
    std::memcpy(&st, ctl + 0x56, 2);
    if (magic != kControlMagic || !len || end > (16u << 20) || c1o > end || len > end || st > 7)
        return false;
    if (code_size < end)
        return false;
    if (len_out)
        *len_out = len;
    return true;
}

bool prepare_stage_code(const uint8_t *ctl, const uint8_t *area, uint32_t area_size, StageCode *out) {
    uint32_t len, c1o;
    if (!control_ok(ctl, area_size, &len))
        return false;
    std::memcpy(&len, ctl + 0x38, 4);
    std::memcpy(&c1o, ctl + 0x40, 4);
    const bool graphics = true;
    const uint32_t hdr = kAmpereHeader;
    if (len <= hdr + 16)
        return false;

    using Shader::Ampere::Instruction;
    std::vector<Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    for (uint32_t i = 0; hdr + (i + 1) * sizeof(Instruction) <= len; i++) {
        Instruction &insn = native.emplace_back();
        std::memcpy(&insn, area + hdr + i * sizeof(Instruction), sizeof insn);
        const auto r = capture.Observe(insn, i);
        if (r == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget)
            return false;
        if (r == Shader::Ampere::ProgramCaptureState::Result::Complete)
            break;
    }

    Shader::Ampere::native_matrix_multiply = false;
    const auto lowered = Shader::Ampere::LowerToMaxwell(native);
    if (!lowered)
        return false;

    out->texture_bound = lowered.required_texture_buffer;
    out->code.assign(sizeof(Shader::ProgramHeader) / 8, 0);
    std::memcpy(out->code.data(), area, sizeof(Shader::ProgramHeader));
    for (const uint64_t instruction : lowered.instructions) {
        if (out->code.size() * 8 % 32 == 0)
            out->code.push_back(0);
        out->code.push_back(instruction);
    }
    out->code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(out->code, lowered.source_instruction_offsets);
    out->c1.assign(area + c1o, area + area_size);
    std::memcpy(&out->shared_memory, ctl + 0x64, 4);
    return true;
}

class MeshEnv final : public Shader::Environment {
public:
    MeshEnv(const StageCode &c, Shader::Stage st, const std::array<u32, 3> &mesh_regs, const uint8_t *prim_sph,
            uint32_t mesh_threads)
        : nvn(c), wg_threads{std::max(1u, mesh_threads)} {
        stage = st;
        start_address = 0;
        std::memcpy(&sph, reinterpret_cast<const uint8_t *>(nvn.code.data()), sizeof sph);
        local_memory = static_cast<uint32_t>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        is_proprietary_driver = true;
        mesh_registers = mesh_regs;
        if (prim_sph) {
            std::memcpy(&mesh_primitive_header, prim_sph, sizeof mesh_primitive_header);
            // MKW's header-only SPH (type 0, code len 0) often has VTG omap_systemc bits set.
            // AddMeshLines only implements systemb + generics and throws on systemc ("Mesh outputs
            // past the generic attributes"). Keep the generic (and systemb) prim map; drop systemc.
            const auto &sysc = mesh_primitive_header.vtg.omap_systemc;
            if (sysc.clip_distances || sysc.point_sprite_s || sysc.point_sprite_t || sysc.fog_coordinate ||
                sysc.tessellation_eval_point_u || sysc.tessellation_eval_point_v || sysc.instance_id ||
                sysc.vertex_id) {
                static int sysc_log;
                if (sysc_log++ < 4)
                    rt_log("eden_mesh: MeshPrimitiveHeader omap_systemc cleared (generics-only AddMeshLines)");
                std::memset(&mesh_primitive_header.vtg.omap_systemc, 0,
                            sizeof mesh_primitive_header.vtg.omap_systemc);
            }
            static int prim_omap_log;
            if (prim_omap_log < 4) {
                char hex[48];
                size_t n = 0;
                for (size_t i = 0; i < mesh_primitive_header.vtg.omap_generic_vector.size() && n + 3 < sizeof hex;
                     i++)
                    n += (size_t)snprintf(hex + n, sizeof hex - n, "%02x",
                                          mesh_primitive_header.vtg.omap_generic_vector[i]);
                rt_log("eden_mesh: MeshPrimitiveHeader omap_generic=%s systemb=%#x", hex,
                       (unsigned)mesh_primitive_header.vtg.omap_systemb.raw);
                prim_omap_log++;
            }
        }
    }

    u64 ReadInstruction(u32 address) override {
        return address / 8 < nvn.code.size() ? nvn.code[address / 8] : 0;
    }

    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto t = Shader::Ampere::NativeProgramAddress(nvn.code, address))
            return *t;
        throw Shader::RuntimeError("indirect branch to native address {:#x}, not in the program", address);
    }

    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (index == 1) {
            uint32_t v = 0;
            if ((size_t)offset + 4 <= nvn.c1.size())
                std::memcpy(&v, nvn.c1.data() + offset, 4);
            return v;
        }
        return 0;
    }

    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override { return Shader::TexturePixelFormat{}; }

    bool IsTexturePixelFormatInteger(u32) override { return false; }

    u32 ReadViewportTransformState() override { return 1; }

    u32 TextureBoundBuffer() const override {
        if (nvn.texture_bound != ~0u)
            return nvn.texture_bound;
        return 2;
    }

    u32 LocalMemorySize() const override { return local_memory; }

    u32 SharedMemorySize() const override { return nvn.shared_memory; }

    std::array<u32, 3> WorkgroupSize() const override { return {wg_threads, 1u, 1u}; }

    bool HasHLEMacroState() const override { return false; }

    bool HasNativeInstructions() const override { return true; }

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }

    void Dump(u64, u64) override {}

private:
    const StageCode &nvn;
    uint32_t local_memory = 0;
    uint32_t wg_threads = 1;
};

struct Translation {
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::IR::Program program;
};

// Fallback when the caller did not pack SET_MESH_* from the task+mesh pair (Hookshot mesh_registers_of).
static void mesh_registers_of(const uint8_t *control, int is_task, std::array<u32, 3> &regs) {
    uint16_t thr = 0, payload = 0, max_prim = 0, max_vert = 0;
    uint8_t topo = 0;
    std::memcpy(&thr, control + 0x58, 2);
    std::memcpy(&payload, control + 0x5a, 2);
    topo = control[0x5f];
    if (topo == 3) topo = 4;
    std::memcpy(&max_prim, control + 0x60, 2);
    std::memcpy(&max_vert, control + 0x62, 2);
    const u32 lines = is_task ? (payload + 127u) / 128u : 0u;
    const u32 most = (std::min)((std::max)((u32)max_vert, (u32)max_prim), 0x100u);
    const u32 threads = (std::min)((u32)thr, 0x3Fu);
    regs[0] = (static_cast<u32>(topo) & 0xfu) | (most << 4) | (most << 13);
    regs[1] = threads << 20;
    regs[2] = (is_task ? threads : 0u) | ((std::min)(lines, 0x1FFu) << 12) | ((std::min)(lines, 0x1FFu) << 22);
}

std::unique_ptr<Translation> translate_stage(MeshEnv &env) {
    auto t = std::make_unique<Translation>();
    Shader::Maxwell::Flow::CFG cfg{env, t->flow_blocks, env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader)),
                                   true};
    Shader::HostTranslateInfo h{};
    h.support_float64 = true;
    h.support_float16 = true;
    h.support_int64 = true;
    h.needs_demote_reorder = false;
    h.support_snorm_render_buffer = true;
    h.support_viewport_index_layer = true;
    h.min_ssbo_alignment = 256;
    h.support_geometry_shader_passthrough = false;
    h.support_conditional_barrier = true;
    t->program = Shader::Maxwell::TranslateProgram(t->insts, t->blocks, env, cfg, h);
    return t;
}

const Shader::Profile &rtx_profile() {
    static const Shader::Profile p = [] {
        Shader::Profile pr{};
        pr.supported_spirv = 0x00010600;
        pr.unified_descriptor_binding = true;
        pr.support_descriptor_aliasing = true;
        pr.support_int8 = pr.support_int16 = pr.support_int64 = true;
        pr.support_uniform_and_storage_buffer_8bit = pr.support_storage_buffer_8bit = true;
        pr.support_uniform_and_storage_buffer_16bit = pr.support_storage_buffer_16bit = true;
        pr.support_vertex_instance_id = false;
        pr.support_float_controls = true;
        pr.support_separate_denorm_behavior = pr.support_separate_rounding_mode = true;
        pr.support_fp32_denorm_preserve = pr.support_fp32_denorm_flush = true;
        pr.support_fp16_signed_zero_nan_preserve = pr.support_fp32_signed_zero_nan_preserve = true;
        pr.support_vote = true;
        pr.supported_subgroup_stages = VK_SHADER_STAGE_ALL;
        pr.support_viewport_index_layer_non_geometry = true;
        pr.support_typeless_image_loads = true;
        pr.support_demote_to_helper_invocation = true;
        pr.support_int64_atomics = true;
        pr.support_derivative_control = true;
        pr.support_native_ndc = false;
        pr.support_scaled_attributes = false;
        pr.support_multi_viewport = true;
        pr.support_sampled_image_array_nonuniform_indexing = true;
        pr.warp_size_potentially_larger_than_guest = false;
        pr.lower_left_origin_mode = false;
        pr.need_declared_frag_colors = false;
        pr.max_subgroup_size = 32;
        pr.min_ssbo_alignment = 256;
        return pr;
    }();
    return p;
}

} // namespace

// SPIR-V Binding numbers under unified+aliasing (matches EmitContext::DefineConstantBuffers then Define*).
static void collect_cbuf_bindings(const Shader::Info &info, EdenMeshCbuf *cbufs, uint32_t *n_cbufs) {
    if (!cbufs || !n_cbufs)
        return;
    *n_cbufs = 0;
    uint32_t binding = 0;
    for (const Shader::ConstantBufferDescriptor &desc : info.constant_buffer_descriptors) {
        if (*n_cbufs >= EDEN_MESH_CBUF_MAX)
            break;
        cbufs[*n_cbufs] = EdenMeshCbuf{binding, desc.index};
        (*n_cbufs)++;
        // Aliased uniforms: one Binding per descriptor (EmitContext::DefineConstantBuffers).
        binding += 1;
    }
}

static void collect_ssbo_bindings(const Shader::Info &info, EdenMeshSsbo *ssbos, uint32_t *n_ssbos) {
    if (!ssbos || !n_ssbos)
        return;
    *n_ssbos = 0;
    uint32_t binding = static_cast<uint32_t>(info.constant_buffer_descriptors.size());
    for (const Shader::StorageBufferDescriptor &desc : info.storage_buffers_descriptors) {
        if (*n_ssbos >= EDEN_MESH_SSBO_MAX)
            break;
        ssbos[*n_ssbos] = EdenMeshSsbo{binding, desc.cbuf_offset, desc.count, desc.is_written ? 1u : 0u};
        (*n_ssbos)++;
        binding += desc.count > 1 ? 1u : desc.count;
    }
}

// Ryujinx SamplerType bits used by GfxShaderRes.type / res_view_type.
static uint16_t ryujinx_tex_type(Shader::TextureType t, bool depth, bool ms) {
    uint16_t ty = 3; // 2D
    switch (t) {
    case Shader::TextureType::Color1D:
        ty = 1;
        break;
    case Shader::TextureType::ColorArray1D:
        ty = (uint16_t)(1u | 0x100u);
        break;
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect:
        ty = 3;
        break;
    case Shader::TextureType::ColorArray2D:
        ty = (uint16_t)(3u | 0x100u);
        break;
    case Shader::TextureType::Color3D:
        ty = 4;
        break;
    case Shader::TextureType::ColorCube:
        ty = 5;
        break;
    case Shader::TextureType::ColorArrayCube:
        ty = (uint16_t)(5u | 0x100u);
        break;
    case Shader::TextureType::Buffer:
        ty = 2;
        break;
    }
    if (ms)
        ty = (uint16_t)(ty | 0x200u);
    if (depth)
        ty = (uint16_t)(ty | 0x400u);
    return ty;
}

static void collect_tex_bindings(const Shader::Info &info, EdenMeshTex *texs, uint32_t *n_texs) {
    if (!texs || !n_texs)
        return;
    *n_texs = 0;
    // Unified binding order: cbufs, ssbos, texture buffers, image buffers, then textures (Define*).
    uint32_t binding = static_cast<uint32_t>(info.constant_buffer_descriptors.size());
    for (const Shader::StorageBufferDescriptor &desc : info.storage_buffers_descriptors)
        binding += desc.count > 1 ? 1u : desc.count;
    binding += static_cast<uint32_t>(info.texture_buffer_descriptors.size());
    binding += static_cast<uint32_t>(info.image_buffer_descriptors.size());
    for (const Shader::TextureDescriptor &desc : info.texture_descriptors) {
        if (*n_texs >= EDEN_MESH_TEX_MAX)
            break;
        texs[*n_texs] = EdenMeshTex{binding, desc.cbuf_index, desc.cbuf_offset, desc.count,
                                    ryujinx_tex_type(desc.type, desc.is_depth, desc.is_multisample)};
        (*n_texs)++;
        binding += 1;
    }
}

extern "C" int eden_mesh_translate(const uint8_t *control, const uint8_t *code, uint32_t code_size, int is_task,
                                   const uint32_t mesh_regs[3], const uint8_t *prim_sph, uint32_t **out_spirv,
                                   uint32_t *out_words, EdenMeshSsbo *ssbos, uint32_t *n_ssbos, EdenMeshCbuf *cbufs,
                                   uint32_t *n_cbufs, EdenMeshTex *texs, uint32_t *n_texs) {
    if (!control || !code || !out_spirv || !out_words)
        return -1;
    *out_spirv = nullptr;
    *out_words = 0;
    if (n_ssbos)
        *n_ssbos = 0;
    if (n_cbufs)
        *n_cbufs = 0;
    if (n_texs)
        *n_texs = 0;

    StageCode stage_code;
    if (!prepare_stage_code(control, code, code_size, &stage_code)) {
        static int prep_log;
        if (prep_log < 4) {
            uint32_t magic = 0, len = 0, end = 0;
            uint16_t st = 0;
            std::memcpy(&magic, control, 4);
            std::memcpy(&len, control + 0x38, 4);
            std::memcpy(&end, control + 0x44, 4);
            std::memcpy(&st, control + 0x56, 2);
            rt_log("eden_mesh: prepare failed is_task=%d code_size=%u magic=%#x len=%u end=%u stage=%u", is_task,
                   code_size, magic, len, end, (unsigned)st);
            prep_log++;
        }
        return -2;
    }

    std::array<u32, 3> regs{};
    if (mesh_regs)
        std::memcpy(regs.data(), mesh_regs, sizeof regs);
    if (!mesh_regs || (!regs[0] && !regs[1] && !regs[2]))
        mesh_registers_of(control, is_task != 0, regs);

    uint32_t w58 = 0;
    std::memcpy(&w58, control + 0x58, 4);
    const uint32_t mesh_threads = w58 & 0xffffu;

    try {
        const Shader::Stage st = is_task ? Shader::Stage::Task : Shader::Stage::Mesh;
        MeshEnv env{stage_code, st, regs, is_task ? nullptr : prim_sph, mesh_threads};
        // Overnight: log mesh SPH omap so Location holes (n181 missing 0xf) can be matched to the header.
        if (!is_task) {
            static int omap_log;
            if (omap_log < 6) {
                const auto& sph = env.SPH();
                char hex[48];
                size_t n = 0;
                for (size_t i = 0; i < sph.vtg.omap_generic_vector.size() && n + 3 < sizeof hex; i++)
                    n += (size_t)snprintf(hex + n, sizeof hex - n, "%02x", sph.vtg.omap_generic_vector[i]);
                rt_log("eden_mesh: mesh SPH omap_generic=%s systemb=%#x", hex, (unsigned)sph.vtg.omap_systemb.raw);
                omap_log++;
            }
        }
        auto tr = translate_stage(env);
        // Do not force c0+0x880 when uses_global_memory is false: n112 aborted in the guest
        // immediately after that descriptor was injected (SPIR-V / bind mismatch). Log occupancy
        // so overnight can see which mesh programs actually carry the Hookshot table.
        collect_cbuf_bindings(tr->program.info, cbufs, n_cbufs);
        collect_ssbo_bindings(tr->program.info, ssbos, n_ssbos);
        collect_tex_bindings(tr->program.info, texs, n_texs);
        {
            static int ssbo_log;
            if (ssbo_log++ < 16) {
                char buf[192];
                size_t n = 0;
                int has_880 = 0;
                if (n_ssbos && *n_ssbos) {
                    for (uint32_t i = 0; i < *n_ssbos && n + 28 < sizeof buf; i++) {
                        n += (size_t)snprintf(buf + n, sizeof buf - n, "%s%#x×%u", i ? "," : "", ssbos[i].cbuf_offset,
                                              ssbos[i].count);
                        if (ssbos[i].cbuf_offset == 0x880u) has_880 = 1;
                    }
                }
                char cbuf[96];
                size_t cn = 0;
                if (n_cbufs && *n_cbufs)
                    for (uint32_t i = 0; i < *n_cbufs && cn + 16 < sizeof cbuf; i++)
                        cn += (size_t)snprintf(cbuf + cn, sizeof cbuf - cn, "%s%u@%u", i ? "," : "", cbufs[i].cbuf_index,
                                               cbufs[i].binding);
                rt_log("eden_mesh: %s SSBOs %u [%s] cbufs %u [%s] texs %u global=%d ptr_table=%d",
                       is_task ? "task" : "mesh", n_ssbos ? *n_ssbos : 0, buf, n_cbufs ? *n_cbufs : 0, cbuf,
                       n_texs ? *n_texs : 0, (int)tr->program.info.uses_global_memory, has_880);
            }
        }
        Shader::Backend::Bindings bindings{};
        Shader::RuntimeInfo rt{};
        rt.previous_stage_stores.mask.set();
        rt.convert_depth_mode = false;
        const std::vector<uint32_t> spirv =
            Shader::Backend::SPIRV::EmitSPIRV(rtx_profile(), rt, tr->program, bindings);
        if (spirv.empty())
            return -3;
        auto *heap = new uint32_t[spirv.size()];
        std::memcpy(heap, spirv.data(), spirv.size() * sizeof(uint32_t));
        *out_spirv = heap;
        *out_words = static_cast<uint32_t>(spirv.size());
        return 0;
    } catch (const std::exception &e) {
        static int catch_log;
        if (catch_log < 4) {
            rt_log("eden_mesh: translate exception is_task=%d: %s", is_task, e.what());
            catch_log++;
        }
        return -4;
    }
}

extern "C" void eden_mesh_free(uint32_t *spirv) { delete[] spirv; }

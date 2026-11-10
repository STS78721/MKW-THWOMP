#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { EDEN_MESH_SSBO_MAX = 32, EDEN_MESH_CBUF_MAX = 16, EDEN_MESH_TEX_MAX = 16 };

// SPIR-V storage binding → NVN c0 word that holds the buffer's {addr,size} (Hookshot/Eden AddNVNStorageBuffers).
// binding is the SPIR-V Binding decoration; gfx_shader adopt stores binding+1 so 0 stays the support UBO.
typedef struct {
    uint32_t binding;
    uint32_t cbuf_offset;
    uint32_t count;
    uint32_t is_written;   // 1: shader stores (NVN tables); 0: Hookshot pointer-table search regions
} EdenMeshSsbo;

// SPIR-V uniform binding → Maxwell constant-buffer index (c0, c3, c4, …). ubo_info uses slot==index
// (0 → driver c0, ≥3 → NVN uniform slot index-3). Without this, spirv_descriptor_res maps by
// binding order and c3 lands on slot 1, which mesh_bind_skip_c1 zeroes → EmitMeshTasks(0).
typedef struct {
    uint32_t binding;
    uint32_t cbuf_index;
} EdenMeshCbuf;

// Combined-image sampler: SPIR-V Binding after cbufs/ssbos/texbufs. type is Ryujinx SamplerType bits
// (3=2D, |0x100 array, |0x200 MS, |0x400 shadow). Handle lives at cbuf_index[cbuf_offset].
typedef struct {
    uint32_t binding;
    uint32_t cbuf_index;
    uint32_t cbuf_offset;
    uint32_t count;
    uint16_t type;
} EdenMeshTex;

/// Translate one NVN2 task (is_task != 0) or mesh stage to SPIR-V. control is 0xd8 bytes; code is the GPU code area.
/// mesh_regs are SET_MESH_SHADER_A/B and SET_MESH_INIT_SHADER; prim_sph is the header-only primitive map stage (mesh only).
/// On success returns 0 and sets *out_spirv / *out_words; free with eden_mesh_free.
/// ssbos/n_ssbos, cbufs/n_cbufs and texs/n_texs may be NULL; when set, receive descriptors in SPIR-V binding order.
int eden_mesh_translate(const uint8_t *control, const uint8_t *code, uint32_t code_size, int is_task,
                        const uint32_t mesh_regs[3], const uint8_t *prim_sph /*nullable*/, uint32_t **out_spirv,
                        uint32_t *out_words, EdenMeshSsbo *ssbos, uint32_t *n_ssbos, EdenMeshCbuf *cbufs,
                        uint32_t *n_cbufs, EdenMeshTex *texs, uint32_t *n_texs);

void eden_mesh_free(uint32_t *spirv);

#ifdef __cplusplus
}
#endif

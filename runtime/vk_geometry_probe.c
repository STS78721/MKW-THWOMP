// SPDX-License-Identifier: MIT
// --probe-vkgeometry: original Maxwell VS/GS/FS through NVN recording and ordinary Vulkan replay.
// A completed readback proves the producer's exact bounded arguments before any indirect draw. The optional
// ordering round starts from a verified zero-count record and uses an explicit NVN dependency in one submission.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_geometry_control.h"
#include "vk_exec.h"

#define NVN_PROC(n) void nvnp_##n(Cpu *c);
#include "nvn_procs.inc"
#undef NVN_PROC
void vk_exec_draw_shutdown(void);
void vk_exec_xfer_shutdown(void);

enum { GP_CONTROL = 0x880, GP_CODE = 2048, GP_SKIP = 0x30, GP_WIDTH = 32, GP_HEIGHT = 32,
       GP_POOL = 0x100000, GP_COLOR = 0x10000, GP_UNIFORM = 0x20000, GP_VERTEX = 0x30000,
       GP_INDEX = 0x31000, GP_RECORD = 0x40100, GP_GUARD = GP_RECORD - 128, GP_RESET = 0x40200, GP_COMMAND = 0x50000,
       GP_COMMAND_CONTROL = 8192 };
static int gp_passthrough;

// Original program authoring. CPU preparation compiles these actual helpers without Vulkan and validates all modules.
static void gp_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static uint64_t gp_movi(unsigned r, uint32_t v) {
    return UINT64_C(0x0100000000000000) | (uint64_t)v << 20 | UINT64_C(15) << 12 | UINT64_C(7) << 16 | r;
}
static uint64_t gp_movc(unsigned r, unsigned bank, unsigned word) {
    return UINT64_C(0x4c98000000000000) | UINT64_C(15) << 39 | (uint64_t)bank << 34 |
           (uint64_t)word << 20 | UINT64_C(7) << 16 | r;
}
static uint64_t gp_attr(unsigned r, unsigned address, unsigned vertex, int store, unsigned words) {
    return (store ? UINT64_C(0xeff0000000000000) : UINT64_C(0xefd8000000000000)) |
           (uint64_t)vertex << 39 | (uint64_t)(words - 1) << 47 | (uint64_t)address << 20 |
           UINT64_C(7) << 16 | UINT64_C(255) << 8 | r;
}
static uint64_t gp_ipa(unsigned r, unsigned address) {
    return UINT64_C(0xe000000000000000) | UINT64_C(255) << 39 | UINT64_C(7) << 47 |
           (uint64_t)address << 28 | UINT64_C(255) << 20 | UINT64_C(7) << 16 | UINT64_C(255) << 8 | r;
}
static uint64_t gp_stg(unsigned r, unsigned offset) {
    return UINT64_C(0xeed8000000000000) | UINT64_C(4) << 48 | UINT64_C(1) << 45 |
           (uint64_t)offset << 20 | UINT64_C(7) << 16 | UINT64_C(10) << 8 | r;
}
static uint32_t gp_program(unsigned stage, uint8_t control[GP_CONTROL], uint8_t code[GP_CODE]) {
    uint64_t ops[128];
    unsigned n = 0;
    memset(control, 0, GP_CONTROL), memset(code, 0, GP_CODE);
    unsigned sph = stage == 0 ? 1 : stage == 1 ? 5 : 4;
    gp_u32(code + GP_SKIP, 1u | 3u << 5 | sph << 10 | (stage == 2 ? 1u << 16 : 0));
    if (stage == 0) {
        ops[n++] = gp_attr(0, 0x80, 255, 0, 4);  // input Position and color, both user vertex attributes
        ops[n++] = gp_attr(4, 0x90, 255, 0, 4);
        ops[n++] = gp_attr(0, 0x70, 255, 1, 4);
        ops[n++] = gp_movi(8, 0x3f800000u);
        ops[n++] = gp_attr(8, 0x6c, 255, 1, 1);  // valid PointSize in the GS-disabled points control
        ops[n++] = gp_attr(4, 0x80, 255, 1, 4);
    } else if (stage == 1) {
        gp_u32(code + GP_SKIP + 6 * 4, 0xff);    // Location0 screen-linear RGBA interpolation
        gp_u32(code + GP_SKIP + 18 * 4, 15);     // RGBA target0
        for (unsigned i = 0; i < 4; i++) ops[n++] = gp_ipa(i, 0x80 + i * 4);
    } else {
        gp_u32(code + GP_SKIP + 8, 1u << 24);    // one invocation
        gp_u32(code + GP_SKIP + 12, 7u << 24);   // triangle strip
        gp_u32(code + GP_SKIP + 16, gp_passthrough ? 1 : 3);
        if (gp_passthrough) {
            uint32_t sph;
            memcpy(&sph, code + GP_SKIP, 4);
            gp_u32(code + GP_SKIP, sph | (1u << 24));
        }
        ops[n++] = gp_movc(10, 3, 0), ops[n++] = gp_movc(11, 3, 1);
        static const uint32_t args[5] = {3, 1, 0, 0, 0};
        for (unsigned i = 0; i < 5; i++) {
            ops[n++] = gp_movi(8, args[i]);
            ops[n++] = gp_stg(8, i * 4);
        }
        if (!gp_passthrough) {
            ops[n++] = gp_movi(9, 0);
            ops[n++] = gp_attr(12, 0x80, 9, 0, 4); // actual VS->GS Location0 color from valid input vertex0
            static const uint32_t positions[3][4] = {
                {0xbf400000u, 0xbf400000u, 0, 0x3f800000u},
                {0x3f400000u, 0xbf400000u, 0, 0x3f800000u},
                {0, 0x3f400000u, 0, 0x3f800000u}};
            for (unsigned v = 0; v < 3; v++) {
                for (unsigned i = 0; i < 4; i++) ops[n++] = gp_movi(i, positions[v][i]);
                // Outputs are undefined after EmitVertex; rewrite Position and color for every vertex.
                ops[n++] = gp_attr(0, 0x70, 255, 1, 4);
                ops[n++] = gp_attr(12, 0x80, 255, 1, 4);
                ops[n++] = UINT64_C(0xf6e0000000000000) | (uint64_t)(v == 2 ? 3 : 1) << 39 |
                           UINT64_C(7) << 16 | UINT64_C(255) << 8 | 255;
            }
        }
    }
    ops[n++] = UINT64_C(0xe30000000007000f);
    unsigned groups = (n + 2) / 3;
    for (unsigned g = 0; g < groups; g++) {
        uint64_t words[4] = {UINT64_C(0x001f8000fc0007e0), 0, 0, 0};
        for (unsigned lane = 0; lane < 3; lane++) if (g * 3 + lane < n) words[lane + 1] = ops[g * 3 + lane];
        memcpy(code + GP_SKIP + 0x50 + g * 32, words, sizeof words);
    }
    uint32_t total = GP_SKIP + 0x50 + groups * 32;
    gp_u32(control + 0x700, total), gp_u32(control + 0x704, total), gp_u32(control + 0x714, stage);
    return total;
}

#define GP_U(p) ((uint64_t)(uintptr_t)(p))
#define GP_OBJECT(type, name) static _Alignas(16) type name
GP_OBJECT(NVNdeviceBuilder, gp_device_builder);
GP_OBJECT(NVNdevice, gp_device);
GP_OBJECT(NVNmemoryPoolBuilder, gp_memory_builder);
GP_OBJECT(NVNmemoryPool, gp_memory_pool);
GP_OBJECT(NVNtextureBuilder, gp_texture_builder);
GP_OBJECT(NVNtexture, gp_color);
GP_OBJECT(NVNprogram, gp_full_program);
GP_OBJECT(NVNprogram, gp_plain_program);
GP_OBJECT(NVNcommandBuffer, gp_command);
static _Alignas(16) uint8_t gp_control[3][GP_CONTROL], gp_code[3][GP_CODE];
static Cpu gp_cpu;
static unsigned gp_checks, gp_failures;

static uint64_t gp_call(void (*fn)(Cpu *), const uint64_t a[8]) {
    memcpy(gp_cpu.x, a, 8 * sizeof a[0]);
    fn(&gp_cpu);
    return gp_cpu.x[0];
}
#define GP_CALL(n, ...) gp_call(nvnp_##n, (const uint64_t[8]){__VA_ARGS__})
#define GP_REQUIRE(c, text) do { gp_checks++; if (!(c)) { \
    printf("probe-vkgeometry: FAIL %s\n", text); gp_failures++; goto done; } } while (0)

// Read the current GPU-visible backing after the preceding NVN submission completed. No guest VA is dereferenced.
static int gp_read_gpu(uint64_t address, void *out, uint64_t size) {
    VkCtx *c = vk_ctx();
    VkMemSpan source;
    if (!size || size > 4096 || (address | size) & 3 || !vk_mem_resolve(address, size, &source)) return 0;
    VK_MEM_PICK(source);
    if (!source.buffer || source.avail < size || (source.offset & 3)) return 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo mapped;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buffer, &allocation, &mapped) != VK_SUCCESS) return 0;
    if (!mapped.pMappedData) { vmaDestroyBuffer(c->vma, buffer, allocation); return 0; }
    VkCommandBuffer cb = vk_upload_begin();
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    VkBufferCopy copy = {source.offset, 0, size};
    vkCmdCopyBuffer(cb, source.buffer, buffer, 1, &copy);
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
    vk_upload_flush(1);
    int ok = vmaInvalidateAllocation(c->vma, allocation, 0, size) == VK_SUCCESS;
    if (ok) memcpy(out, mapped.pMappedData, (size_t)size);
    vmaDestroyBuffer(c->vma, buffer, allocation);
    return ok;
}

static int gp_submit(void) {
    uint64_t commands = GP_CALL(nvnCommandBufferEndRecording, GP_U(&gp_command));
    if (!commands) return 0;
    uint64_t fence = gfx_vulkan_backend.submit(&commands, 1, rt_ticks());
    if (!gfx_vulkan_backend.fence_wait(fence, UINT64_C(10000000000))) return 0;
    gfx_vulkan_backend.finish(); // recorder and completion ownership are idle before main-thread texture/readback work
    gfx_vulkan_backend.cpu_synced(fence);
    return 1;
}

static void gp_prefix(uint8_t *memory, void *control, int indirect) {
    GP_CALL(nvnCommandBufferAddControlMemory, GP_U(&gp_command), GP_U(control), GP_COMMAND_CONTROL);
    GP_CALL(nvnCommandBufferBeginRecording, GP_U(&gp_command));
    const NVNtexture *color = &gp_color;
    GP_CALL(nvnCommandBufferSetRenderTargets, GP_U(&gp_command), 1, GP_U(&color), 0, 0, 0);
    static const float clear[4] = {0, 0, 1, 1};
    GP_CALL(nvnCommandBufferClearColor, GP_U(&gp_command), 0, GP_U(clear), 15);
    GP_CALL(nvnCommandBufferSetViewport, GP_U(&gp_command), 0, 0, GP_WIDTH, GP_HEIGHT);
    GP_CALL(nvnCommandBufferSetScissor, GP_U(&gp_command), 0, 0, GP_WIDTH, GP_HEIGHT);
    GP_CALL(nvnCommandBufferBindProgram, GP_U(&gp_command), GP_U(indirect ? &gp_plain_program : &gp_full_program),
        indirect ? NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT : NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    NVNvertexAttribState attr[2];
    NVNvertexStreamState stream;
    for (unsigned i = 0; i < 2; i++) {
        GP_CALL(nvnVertexAttribStateSetDefaults, GP_U(&attr[i]));
        GP_CALL(nvnVertexAttribStateSetFormat, GP_U(&attr[i]), NVN_FORMAT_RGBA32F, i * 16);
        GP_CALL(nvnVertexAttribStateSetStreamIndex, GP_U(&attr[i]), 0);
    }
    GP_CALL(nvnVertexStreamStateSetDefaults, GP_U(&stream));
    GP_CALL(nvnVertexStreamStateSetStride, GP_U(&stream), 32);
    GP_CALL(nvnCommandBufferBindVertexAttribState, GP_U(&gp_command), 2, GP_U(attr));
    GP_CALL(nvnCommandBufferBindVertexStreamState, GP_U(&gp_command), 1, GP_U(&stream));
    GP_CALL(nvnCommandBufferBindVertexBuffer, GP_U(&gp_command), 0,
        GP_U(memory + GP_VERTEX + (indirect ? 0 : gp_passthrough ? 96 : 192)),
        indirect || gp_passthrough ? 96 : 32);
    if (!indirect) GP_CALL(nvnCommandBufferBindUniformBuffer, GP_U(&gp_command), NVN_SHADER_STAGE_GEOMETRY,
        0, GP_U(memory + GP_UNIFORM), 256); // actual shader c3, NVN user UBO slot0
}

// Sample centers never lie on the authored triangle's edges. NVN viewport orientation may flip the whole image.
static int gp_inside(unsigned x, unsigned y, int flip) {
    int xx = (int)x * 2 + 1, yy = (int)(flip ? GP_HEIGHT - 1 - y : y) * 2 + 1;
    return yy > 8 && yy < 56 && 2 * xx - yy > 8 && 2 * xx + yy < 120;
}

static int gp_pixels(uint32_t pixels[GP_WIDTH * GP_HEIGHT], uint32_t color, int *orientation) {
    NVNcopyRegion region = {0, 0, 0, GP_WIDTH, GP_HEIGHT, 1};
    memset(pixels, 0x5a, GP_WIDTH * GP_HEIGHT * sizeof *pixels); // a refused/short read cannot reuse previous pixels
    GP_CALL(nvnTextureReadTexels, GP_U(&gp_color), 0, GP_U(&region), GP_U(pixels));
    unsigned bad[2] = {0};
    for (unsigned y = 0; y < GP_HEIGHT; y++) for (unsigned x = 0; x < GP_WIDTH; x++) {
        for (int flip = 0; flip < 2; flip++)
            bad[flip] += pixels[y * GP_WIDTH + x] != (gp_inside(x, y, flip) ? color : 0xffff0000u);
    }
    *orientation = bad[0] == 0 ? 0 : bad[1] == 0 ? 1 : -1;
    gp_checks += GP_WIDTH * GP_HEIGHT;
    if (*orientation < 0) {
        printf("probe-vkgeometry: FAIL triangle pixels (%u normal / %u flipped mismatches)\n", bad[0], bad[1]);
        gp_failures++;
    }
    return *orientation >= 0;
}

// Both possible argument records are bounded: reset draws no indices, and the already validated original GS
// writes exactly three owned indices/one instance. Reset completion is checked before the combined recording;
// there is no CPU readback or wait between its producer and consumer. The normal NVN barrier supplies ordering.
static int gp_ordered(uint8_t *memory, void *control, uint64_t record_address,
                      const uint8_t verified[256], int expected_orientation) {
    int ok = 0;
    uint8_t expected_guard[256], reset[256], after[256];
    static const uint32_t produced[5] = {3, 1, 0, 0, 0}, initial[5] = {0, 1, 0, 0, 0};
    memset(expected_guard, 0xa5, sizeof expected_guard);
    memcpy(expected_guard + 128, produced, sizeof produced);
    GP_REQUIRE(memory && control && gp_passthrough && record_address == GP_U(memory + GP_RECORD) &&
        expected_orientation >= 0 && expected_orientation <= 1, "ordering round owned inputs");
    GP_REQUIRE(verified && !memcmp(verified, expected_guard, sizeof expected_guard), "ordering round prior bounded GPU proof");
    memcpy(memory + GP_RESET, initial, sizeof initial);
    GP_CALL(nvnMemoryPoolFlushMappedRange, GP_U(&gp_memory_pool), GP_RESET, sizeof initial);
    GP_CALL(nvnCommandBufferAddControlMemory, GP_U(&gp_command), GP_U(control), GP_COMMAND_CONTROL);
    GP_CALL(nvnCommandBufferBeginRecording, GP_U(&gp_command));
    GP_CALL(nvnCommandBufferCopyBufferToBuffer, GP_U(&gp_command), GP_U(memory + GP_RESET),
        record_address, sizeof initial, 0);
    GP_REQUIRE(gp_submit(), "ordering round reset submission completion");
    GP_REQUIRE(gp_read_gpu(GP_U(memory + GP_GUARD), reset, sizeof reset), "ordering round reset GPU readback");
    memcpy(expected_guard + 128, initial, sizeof initial);
    GP_REQUIRE(!memcmp(reset, expected_guard, sizeof reset), "ordering round exact zero-count record and canaries");

    gp_prefix(memory, control, 0);
    GP_CALL(nvnCommandBufferDrawArrays, GP_U(&gp_command), NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    GP_CALL(nvnCommandBufferBarrier, GP_U(&gp_command),
        NVN_BARRIER_ORDER_PRIMITIVES_BIT | NVN_BARRIER_ORDER_INDIRECT_DATA_BIT);
    GP_CALL(nvnCommandBufferBindProgram, GP_U(&gp_command), GP_U(&gp_plain_program),
        NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT);
    GP_CALL(nvnCommandBufferBindVertexBuffer, GP_U(&gp_command), 0, GP_U(memory + GP_VERTEX), 96);
    GP_CALL(nvnCommandBufferDrawElementsIndirect, GP_U(&gp_command), NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_SHORT, GP_U(memory + GP_INDEX), record_address);
    GP_REQUIRE(gp_submit(), "ordering round combined submission completion");
    uint32_t pixels[GP_WIDTH * GP_HEIGHT];
    int orientation;
    GP_REQUIRE(gp_pixels(pixels, 0xff00ff00u, &orientation) && orientation == expected_orientation,
        "ordering round producer/barrier/indirect green triangle");
    GP_REQUIRE(gp_read_gpu(GP_U(memory + GP_GUARD), after, sizeof after), "ordering round final GPU readback");
    GP_REQUIRE(!memcmp(after, verified, sizeof after), "ordering round exact produced record and canaries");
    puts("probe-vkgeometry: verified zero reset -> GS -> NVN barrier -> indirect in one NVN recording/submission");
    ok = 1;
done:
    return ok;
}

static int gp_run(int passthrough, int ordered) {
    gp_passthrough = passthrough;
    const GfxBackend *backend = &gfx_vulkan_backend;
    if (gfx_backend() != backend || cfg_bool("TOTK_FIX_C0")) {
        printf("probe-vkgeometry: requires TOTK_GPU=vulkan and TOTK_FIX_C0=0 (ordinary Maxwell only)\n");
        return 1;
    }
    gp_checks = gp_failures = 0;
    uint64_t heap = 0, shader_failures = gfx_shader_failures();
    uint8_t *memory = NULL;
    void *control = NULL;
    int device_live = 0, pool_live = 0, texture_live = 0, command_live = 0, programs_live = 0;
    GP_REQUIRE(!mem_set_heap_size(16ull << 20, &heap) && heap, "SetHeapSize");
    memory = (uint8_t *)(uintptr_t)heap;
    control = mem_kernel_alloc(GP_COMMAND_CONTROL, 16);
    GP_REQUIRE(control, "owned command control allocation");
    GP_CALL(nvnDeviceBuilderSetDefaults, GP_U(&gp_device_builder));
    GP_REQUIRE((uint8_t)GP_CALL(nvnDeviceInitialize, GP_U(&gp_device), GP_U(&gp_device_builder)), "DeviceInitialize");
    device_live = 1;
    GP_CALL(nvnMemoryPoolBuilderSetDefaults, GP_U(&gp_memory_builder));
    GP_CALL(nvnMemoryPoolBuilderSetDevice, GP_U(&gp_memory_builder), GP_U(&gp_device));
    GP_CALL(nvnMemoryPoolBuilderSetStorage, GP_U(&gp_memory_builder), GP_U(memory), GP_POOL);
    GP_CALL(nvnMemoryPoolBuilderSetFlags, GP_U(&gp_memory_builder),
        NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    GP_REQUIRE((uint8_t)GP_CALL(nvnMemoryPoolInitialize, GP_U(&gp_memory_pool), GP_U(&gp_memory_builder)), "MemoryPoolInitialize");
    pool_live = 1;
    printf("probe-vkgeometry: device %s driver %#x core-validation-active %d\n", vk_ctx()->props.deviceName,
        vk_ctx()->props.driverVersion, vk_ctx()->validation);
    GP_CALL(nvnTextureBuilderSetDefaults, GP_U(&gp_texture_builder));
    GP_CALL(nvnTextureBuilderSetDevice, GP_U(&gp_texture_builder), GP_U(&gp_device));
    GP_CALL(nvnTextureBuilderSetTarget, GP_U(&gp_texture_builder), NVN_TEXTURE_TARGET_2D);
    GP_CALL(nvnTextureBuilderSetSize2D, GP_U(&gp_texture_builder), GP_WIDTH, GP_HEIGHT);
    GP_CALL(nvnTextureBuilderSetLevels, GP_U(&gp_texture_builder), 1);
    GP_CALL(nvnTextureBuilderSetFormat, GP_U(&gp_texture_builder), NVN_FORMAT_RGBA8);
    GP_CALL(nvnTextureBuilderSetStorage, GP_U(&gp_texture_builder), GP_U(&gp_memory_pool), GP_COLOR);
    uint64_t texture_bytes = GP_CALL(nvnTextureBuilderGetStorageSize, GP_U(&gp_texture_builder));
    GP_REQUIRE(texture_bytes && texture_bytes <= 0x10000, "bounded color storage");
    GP_REQUIRE((uint8_t)GP_CALL(nvnTextureInitialize, GP_U(&gp_color), GP_U(&gp_texture_builder)), "TextureInitialize");
    texture_live = 1;
    static const float vertices[7][8] = {
        {-0.75f, -0.75f, 0, 1, 0, 1, 0, 1}, {0.75f, -0.75f, 0, 1, 0, 1, 0, 1},
        {0, 0.75f, 0, 1, 0, 1, 0, 1},
        {-0.75f, -0.75f, 0, 1, 1, 0, 0, 1}, {0.75f, -0.75f, 0, 1, 1, 0, 0, 1},
        {0, 0.75f, 0, 1, 1, 0, 0, 1}, {0, 0, 0, 1, 1, 0, 0, 1}};
    static const uint16_t indices[3] = {0, 1, 2};
    uint64_t record_address = GP_U(memory + GP_RECORD);
    memcpy(memory + GP_UNIFORM, &record_address, 8);
    memcpy(memory + GP_VERTEX, vertices, sizeof vertices), memcpy(memory + GP_INDEX, indices, sizeof indices);
    memset(memory + GP_GUARD, 0xa5, 256), memset(memory + GP_RECORD, 0, 20);
    GP_CALL(nvnMemoryPoolFlushMappedRange, GP_U(&gp_memory_pool), GP_UNIFORM, GP_POOL - GP_UNIFORM);
    NVNshaderData shaders[3];
    for (unsigned stage = 0; stage < 3; stage++) {
        uint32_t length = gp_program(stage, gp_control[stage], gp_code[stage]);
        if (stage == 2 && gp_passthrough) {
            static const uint32_t skip[8] = {0x0fffffffu, 0xfffffff0u, UINT32_MAX, UINT32_MAX,
                                             UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
            GP_REQUIRE(nvn_geometry_control_pack_portable(gp_control[stage], GP_CONTROL,
                gp_code[stage], length, skip), "canonical portable geometry metadata");
        }
        shaders[stage] = (NVNshaderData){GP_U(gp_code[stage]), gp_control[stage]};
    }
    GP_REQUIRE((uint8_t)GP_CALL(nvnProgramInitialize, GP_U(&gp_full_program), GP_U(&gp_device)), "full ProgramInitialize");
    programs_live = 1;
    GP_REQUIRE((uint8_t)GP_CALL(nvnProgramSetShaders, GP_U(&gp_full_program), 3, GP_U(shaders)), "original VS/FS/GS program");
    GP_REQUIRE((uint8_t)GP_CALL(nvnProgramInitialize, GP_U(&gp_plain_program), GP_U(&gp_device)), "plain ProgramInitialize");
    programs_live = 2;
    GP_REQUIRE((uint8_t)GP_CALL(nvnProgramSetShaders, GP_U(&gp_plain_program), 2, GP_U(shaders)), "original VS/FS program");
    GP_REQUIRE((uint8_t)GP_CALL(nvnCommandBufferInitialize, GP_U(&gp_command), GP_U(&gp_device)), "CommandBufferInitialize");
    command_live = 1;
    GP_CALL(nvnCommandBufferAddCommandMemory, GP_U(&gp_command), GP_U(&gp_memory_pool), GP_COMMAND, 0x10000);
    gp_prefix(memory, control, 0);
    GP_CALL(nvnCommandBufferDrawArrays, GP_U(&gp_command),
        gp_passthrough ? NVN_DRAW_PRIMITIVE_TRIANGLES : NVN_DRAW_PRIMITIVE_POINTS, 0, gp_passthrough ? 3 : 1);
    GP_REQUIRE(gp_submit(), "GS submission completion");
    uint8_t guard[256];
    GP_REQUIRE(gp_read_gpu(GP_U(memory + GP_GUARD), guard, sizeof guard), "GPU argument/canary readback");
    static const uint32_t expected[5] = {3, 1, 0, 0, 0};
    int safe = 1;
    for (unsigned i = 0; i < 5; i++) {
        uint32_t value;
        memcpy(&value, guard + 128 + i * 4, 4);
        gp_checks++;
        if (value != expected[i]) {
            printf("probe-vkgeometry: FAIL argument %u = %u expected %u\n", i, value, expected[i]);
            gp_failures++, safe = 0;
        }
    }
    for (unsigned i = 0; i < sizeof guard; i++) if (i < 128 || i >= 148) {
        gp_checks++;
        if (guard[i] != 0xa5) { gp_failures++, safe = 0; printf("probe-vkgeometry: FAIL canary %u\n", i); }
    }
    uint32_t first[GP_WIDTH * GP_HEIGHT], second[GP_WIDTH * GP_HEIGHT];
    int first_orientation, second_orientation;
    gp_pixels(first, 0xff0000ffu, &first_orientation);
    if (safe) {
        puts("probe-vkgeometry: GPU record {3,1,0,0,0} and all 236 surrounding canary bytes verified");
        // Only exact {3,1,0,0,0} can reach this call; all three uint16 indices and 96 vertex bytes are owned and valid.
        gp_prefix(memory, control, 1);
        GP_CALL(nvnCommandBufferDrawElementsIndirect, GP_U(&gp_command), NVN_DRAW_PRIMITIVE_TRIANGLES,
            NVN_INDEX_TYPE_UNSIGNED_SHORT, GP_U(memory + GP_INDEX), record_address);
        GP_REQUIRE(gp_submit(), "validated indexed-indirect submission completion");
        gp_pixels(second, 0xff00ff00u, &second_orientation);
        GP_REQUIRE(first_orientation >= 0 && first_orientation == second_orientation, "same direct-GS/indirect triangle coverage");
        uint8_t after[256];
        GP_REQUIRE(gp_read_gpu(GP_U(memory + GP_GUARD), after, sizeof after), "post-indirect GPU record readback");
        GP_REQUIRE(!memcmp(guard, after, sizeof guard), "indirect consumption preserves argument record/canaries");
        if (ordered) GP_REQUIRE(gp_ordered(memory, control, record_address, guard, first_orientation), "same-submission ordering round");
    } else printf("probe-vkgeometry: indexed-indirect SKIPPED: exact bounded argument preflight failed\n");
    GP_REQUIRE(gfx_shader_failures() == shader_failures, "no translation failures");
    if (gp_passthrough) {
        GfxShaderAnswers answers = {0};
        answers.geometry_input = GFX_GEOMETRY_TRIANGLES;
        uint32_t length = nvn_geometry_control_u32(gp_control[2] + 0x704);
        const GfxShader *source = gfx_shader_get(GFX_STAGE_GS, gp_control[2], gp_code[2], length, NULL, &answers);
        GP_REQUIRE(source && gfx_shader_geometry(source) &&
                   gfx_shader_geometry(source)->version == GFX_GEOMETRY_PASSTHROUGH_VERSION,
                   "real portable shader/cache result");
        int owned = -1;
        GP_REQUIRE(gfx_shader_load_private(gfx_shader_input_id(source), &owned) == source && !owned,
                   "borrowed portable shader validates its stored source");
        GP_REQUIRE(gfx_shader_load_input(gfx_shader_input_id(source), 0) == source,
                   "portable input-ID lookup preserves its source");
    }
done:
    if (pool_live) backend->finish();
    if (command_live) {
        if (control) GP_CALL(nvnCommandBufferAddControlMemory, GP_U(&gp_command), GP_U(control), GP_COMMAND_CONTROL);
        GP_CALL(nvnCommandBufferFinalize, GP_U(&gp_command));
    }
    if (programs_live >= 2) GP_CALL(nvnProgramFinalize, GP_U(&gp_plain_program));
    if (programs_live >= 1) GP_CALL(nvnProgramFinalize, GP_U(&gp_full_program));
    if (texture_live) GP_CALL(nvnTextureFinalize, GP_U(&gp_color));
    if (pool_live) GP_CALL(nvnMemoryPoolFinalize, GP_U(&gp_memory_pool));
    if (device_live) GP_CALL(nvnDeviceFinalize, GP_U(&gp_device));
    if (control) mem_kernel_free(control, GP_COMMAND_CONTROL);
    vk_pipeline_shutdown(), vk_exec_draw_shutdown(), vk_exec_xfer_shutdown(), vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-vkgeometry: %s (%u checks, %u failures, %ld validation errors, enabled %d)\n",
        gp_failures || errors ? "FAILED" : "passed", gp_checks, gp_failures, errors, cfg_bool("TOTK_EXPERIMENTAL_GEOMETRY"));
    printf("probe-vkgeometry: %s, user-c3 store, %s; ",
           gp_passthrough ? "portable triangles->triangle" : "ordinary points->triangle",
           ordered ? "explicit NVN dependency in one NVN submission" : "separate completed-submission indirect consumption");
    puts(
         "no native lowering, NV passthrough extension or other input topologies");
    return gp_failures || errors;
}

int probe_vkgeometry(void) { return gp_run(0, 0); }
int probe_vkgeometry_passthrough(void) { return gp_run(1, 0); }
int probe_vkgeometry_ordering(void) { return gp_run(1, 1); }

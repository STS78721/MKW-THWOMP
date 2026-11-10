// SPDX-License-Identifier: MIT
// --probe-vkpool: original sampled-texture shader through the real NVN recorder and Vulkan replay.
// No game/census shader or direct descriptor injection. CPU preparation checks this exact instruction array.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk_exec.h"

#define NVN_PROC(n) void nvnp_##n(Cpu *c);
#include "nvn_procs.inc"
#undef NVN_PROC

// Original Maxwell: MOV32I R0,1.25; MOV32I R1,0.5; MOV R2,c0[0x20];
// TEX.B.2D.LZ R8..R11,[R0,R1],R2; MOV R4/R5,c0[0x10/0x14]; STG.E.B128 [R4],R8; EXIT.
// DKC1: zero root words, one storage buffer and one texture. Its handle is reflected to slot 0, not read as a pointer.
static const uint64_t pool_probe_code[] = {
    UINT64_C(0x001f8000fc0007e0), UINT64_C(0x0103fa000007f000), UINT64_C(0x0103f0000007f001), UINT64_C(0x4c98078000870002),
    UINT64_C(0x001f8000fc0007e0), UINT64_C(0xdeb80027a0270008), UINT64_C(0x4c98078000470004), UINT64_C(0x4c98078000570005),
    UINT64_C(0x001f8000fc0007e0), UINT64_C(0xeede200000070408), UINT64_C(0xe30000000007000f), UINT64_C(0x0000000000000000),
};

enum { POOL_BYTES = 0x100000, TABLE_A = 0x40000, TABLE_B = 0x50000,
       TABLE_REPEAT = 0x60000, TABLE_CLAMP = 0x70000, COMMAND_OFF = 0x80000,
       OUTPUT_OFF = 0xa0000, OUTPUT_BYTES = 0x10000, CONTROL_BYTES = 4096 };
#define U(p) ((uint64_t)(uintptr_t)(p))
#define OBJECT(type, name) static _Alignas(16) type name
OBJECT(NVNdeviceBuilder, device_builder);
OBJECT(NVNdevice, device);
OBJECT(NVNmemoryPoolBuilder, memory_builder);
OBJECT(NVNmemoryPool, memory_pool);
OBJECT(NVNtextureBuilder, texture_builder);
OBJECT(NVNtexture, red_texture);
OBJECT(NVNtexture, green_texture);
OBJECT(NVNtexture, pattern_texture);
OBJECT(NVNsamplerBuilder, sampler_builder);
OBJECT(NVNsampler, repeat_sampler);
OBJECT(NVNsampler, clamp_sampler);
OBJECT(NVNtexturePool, texture_a);
OBJECT(NVNtexturePool, texture_b);
OBJECT(NVNsamplerPool, sampler_repeat);
OBJECT(NVNsamplerPool, sampler_clamp);
OBJECT(NVNprogram, program);
OBJECT(NVNcommandBuffer, command);
static _Alignas(16) uint8_t control[0x880];
static Cpu cpu;
static unsigned checks, failures;

static uint64_t call(void (*fn)(Cpu *), const uint64_t a[8]) {
    memcpy(cpu.x, a, sizeof cpu.x[0] * 8);
    fn(&cpu);
    return cpu.x[0];
}
#define CALL(n, ...) call(nvnp_##n, (const uint64_t[8]){__VA_ARGS__})
#define REQUIRE(cond, text) do { checks++; if (!(cond)) { \
    printf("probe-vkpool: FAIL %s\n", text); failures++; goto done; } } while (0)

static int make_texture(NVNtexture *texture, uint64_t off, const uint8_t pixels[8]) {
    CALL(nvnTextureBuilderSetDevice, U(&texture_builder), U(&device));
    CALL(nvnTextureBuilderSetDefaults, U(&texture_builder));
    CALL(nvnTextureBuilderSetTarget, U(&texture_builder), NVN_TEXTURE_TARGET_2D);
    CALL(nvnTextureBuilderSetSize2D, U(&texture_builder), 2, 1);
    CALL(nvnTextureBuilderSetLevels, U(&texture_builder), 1);
    CALL(nvnTextureBuilderSetFormat, U(&texture_builder), NVN_FORMAT_RGBA8);
    CALL(nvnTextureBuilderSetStorage, U(&texture_builder), U(&memory_pool), off);
    uint64_t bytes = CALL(nvnTextureBuilderGetStorageSize, U(&texture_builder));
    if (!bytes || bytes > 0x10000) return 0;
    if (!(uint8_t)CALL(nvnTextureInitialize, U(texture), U(&texture_builder))) return 0;
    NVNcopyRegion region = {0, 0, 0, 2, 1, 1};
    CALL(nvnTextureWriteTexels, U(texture), 0, U(&region), U(pixels));
    return 1;
}

static int make_sampler(NVNsampler *sampler, int wrap) {
    CALL(nvnSamplerBuilderSetDevice, U(&sampler_builder), U(&device));
    CALL(nvnSamplerBuilderSetDefaults, U(&sampler_builder));
    CALL(nvnSamplerBuilderSetMinMagFilter, U(&sampler_builder), NVN_MIN_FILTER_NEAREST, NVN_MAG_FILTER_NEAREST);
    CALL(nvnSamplerBuilderSetWrapMode, U(&sampler_builder), wrap, NVN_WRAP_MODE_CLAMP_TO_EDGE, NVN_WRAP_MODE_CLAMP_TO_EDGE);
    return (uint8_t)CALL(nvnSamplerInitialize, U(sampler), U(&sampler_builder));
}

static int run_case(const char *name, uint8_t *memory, void *command_control, uint64_t handle,
                    const NVNtexturePool *const textures[3], const NVNsamplerPool *const samplers[3],
                    const uint32_t expected[3][4], unsigned phases, size_t stride) {
    const GfxBackend *backend = &gfx_vulkan_backend;
    memset(memory + OUTPUT_OFF, 0xa5, OUTPUT_BYTES);
    CALL(nvnMemoryPoolFlushMappedRange, U(&memory_pool), OUTPUT_OFF, OUTPUT_BYTES);
    // Re-adding this exact owned range retires the preceding recording only after its fence completed below.
    CALL(nvnCommandBufferAddControlMemory, U(&command), U(command_control), CONTROL_BYTES);
    CALL(nvnCommandBufferBeginRecording, U(&command));
    CALL(nvnCommandBufferBindProgram, U(&command), U(&program), NVN_SHADER_STAGE_COMPUTE_BIT);
    CALL(nvnCommandBufferBindTexture, U(&command), NVN_SHADER_STAGE_COMPUTE, 0, handle);
    for (unsigned phase = 0; phase < phases; phase++) {
        CALL(nvnCommandBufferSetTexturePool, U(&command), U(textures[phase]));
        CALL(nvnCommandBufferSetSamplerPool, U(&command), U(samplers[phase]));
        CALL(nvnCommandBufferBindStorageBuffer, U(&command), NVN_SHADER_STAGE_COMPUTE, 0,
             U(memory + OUTPUT_OFF + (phase + 1) * stride), 16);
        // SetPool alone changes descriptor identity: no handle/program bind or registration occurs between dispatches.
        for (unsigned repeat = 0; repeat < 2; repeat++) {
            CALL(nvnCommandBufferDispatchCompute, U(&command), 1, 1, 1);
        }
    }
    uint64_t commands = CALL(nvnCommandBufferEndRecording, U(&command));
    checks++;
    if (!commands) { printf("probe-vkpool: FAIL %s: no recording\n", name); failures++; return 0; }
    uint64_t fence = backend->submit(&commands, 1, rt_ticks());
    checks++;
    if (!backend->fence_wait(fence, UINT64_C(10000000000))) {
        printf("probe-vkpool: FAIL %s: fence did not complete\n", name); failures++; return 0;
    }
    backend->cpu_synced(fence);
    // The completion frontier publishes SSBO writes before fence_wait returns. Compare every word and untouched byte.
    unsigned before = failures;
    for (unsigned phase = 0; phase < phases; phase++) {
        const uint32_t *actual = (const uint32_t *)(memory + OUTPUT_OFF + (phase + 1) * stride);
        for (unsigned component = 0; component < 4; component++) {
            checks++;
            if (actual[component] != expected[phase][component]) {
                printf("probe-vkpool: FAIL %s phase %u component %u: %08x expected %08x\n",
                    name, phase, component, actual[component], expected[phase][component]);
                failures++;
            }
        }
    }
    int canary_ok = 1;
    for (size_t off = 0; off < OUTPUT_BYTES; off++) {
        size_t slot = off / stride;
        if (slot >= 1 && slot <= phases && off % stride < 16) continue;
        if (memory[OUTPUT_OFF + off] != 0xa5) { canary_ok = 0; break; }
    }
    checks++;
    if (!canary_ok) { printf("probe-vkpool: FAIL %s: output canary changed\n", name); failures++; }
    printf("probe-vkpool: %s: %s (%u pool selections, two dispatches each)\n", name,
        failures == before ? "passed" : "FAILED", phases);
    return 1;   // A semantic failure is accumulated; the following independent case still runs.
}

void vk_exec_draw_shutdown(void);
void vk_exec_xfer_shutdown(void);

int probe_vkpool(void) {
    const GfxBackend *backend = &gfx_vulkan_backend;
    checks = failures = 0;
    if (gfx_backend() != backend || !cfg_bool("TOTK_FIX_C0")) {
        printf("probe-vkpool: requires TOTK_GPU=vulkan and TOTK_FIX_C0=1\n");
        return 1;
    }
    uint64_t heap = 0;
    uint8_t *memory = NULL;
    void *command_control = NULL;
    int device_live = 0, pool_live = 0, program_live = 0, command_live = 0;
    unsigned textures_live = 0, samplers_live = 0, descriptor_pools_live = 0;
    uint64_t shader_failures = gfx_shader_failures();
    REQUIRE(!mem_set_heap_size(16ull << 20, &heap) && heap, "SetHeapSize");
    memory = (uint8_t *)(uintptr_t)heap;
    command_control = mem_kernel_alloc(CONTROL_BYTES, 16);
    REQUIRE(command_control, "owned command control allocation");
    CALL(nvnDeviceBuilderSetDefaults, U(&device_builder));
    REQUIRE((uint8_t)CALL(nvnDeviceInitialize, U(&device), U(&device_builder)), "DeviceInitialize");
    device_live = 1;
    CALL(nvnMemoryPoolBuilderSetDefaults, U(&memory_builder));
    CALL(nvnMemoryPoolBuilderSetDevice, U(&memory_builder), U(&device));
    CALL(nvnMemoryPoolBuilderSetStorage, U(&memory_builder), U(memory), POOL_BYTES);
    CALL(nvnMemoryPoolBuilderSetFlags, U(&memory_builder),
        NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    REQUIRE((uint8_t)CALL(nvnMemoryPoolInitialize, U(&memory_pool), U(&memory_builder)), "MemoryPoolInitialize");
    pool_live = 1;
    static const uint8_t red[8] = {255, 0, 0, 255, 255, 0, 0, 255};
    static const uint8_t green[8] = {0, 255, 0, 255, 0, 255, 0, 255};
    static const uint8_t pattern[8] = {255, 0, 0, 255, 0, 255, 0, 255};
    REQUIRE(make_texture(&red_texture, 0x10000, red), "red texture"); textures_live = 1;
    REQUIRE(make_texture(&green_texture, 0x20000, green), "green texture"); textures_live = 2;
    REQUIRE(make_texture(&pattern_texture, 0x30000, pattern), "pattern texture"); textures_live = 3;
    REQUIRE(make_sampler(&repeat_sampler, NVN_WRAP_MODE_REPEAT), "repeat sampler"); samplers_live = 1;
    REQUIRE(make_sampler(&clamp_sampler, NVN_WRAP_MODE_CLAMP_TO_EDGE), "clamp sampler"); samplers_live = 2;
    int texture_reserved = 0, sampler_reserved = 0;
    CALL(nvnDeviceGetInteger, U(&device), NVN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, U(&texture_reserved));
    CALL(nvnDeviceGetInteger, U(&device), NVN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, U(&sampler_reserved));
    REQUIRE(texture_reserved >= 0 && texture_reserved < 2044 && sampler_reserved >= 0 && sampler_reserved < 2044,
        "bounded reserved descriptor counts");
    int ti = texture_reserved + 1, si = sampler_reserved + 1;
    REQUIRE((uint8_t)CALL(nvnTexturePoolInitialize, U(&texture_a), U(&memory_pool), TABLE_A, ti + 2), "TexturePool A");
    descriptor_pools_live = 1;
    REQUIRE((uint8_t)CALL(nvnTexturePoolInitialize, U(&texture_b), U(&memory_pool), TABLE_B, ti + 2), "TexturePool B");
    descriptor_pools_live = 2;
    REQUIRE((uint8_t)CALL(nvnSamplerPoolInitialize, U(&sampler_repeat), U(&memory_pool), TABLE_REPEAT, si + 2), "SamplerPool repeat");
    descriptor_pools_live = 3;
    REQUIRE((uint8_t)CALL(nvnSamplerPoolInitialize, U(&sampler_clamp), U(&memory_pool), TABLE_CLAMP, si + 2), "SamplerPool clamp");
    descriptor_pools_live = 4;
    CALL(nvnTexturePoolRegisterTexture, U(&texture_a), ti, U(&red_texture), 0);
    CALL(nvnSamplerPoolRegisterSampler, U(&sampler_repeat), si, U(&repeat_sampler));
    uint64_t handle = CALL(nvnDeviceGetTextureHandle, U(&device), ti, si);
    REQUIRE(handle == (uint64_t)ti + ((uint64_t)si << 20), "unchanged combined handle");
    uint32_t length = sizeof pool_probe_code, stage = NVN_SHADER_STAGE_COMPUTE, one = 1;
    memcpy(control + 0x700, &length, 4); memcpy(control + 0x704, &length, 4);
    memcpy(control + 0x714, &stage, 4);
    for (unsigned k = 0; k < 3; k++) memcpy(control + 0x748 + 4 * k, &one, 4);
    memcpy(control + 0x860, "DKC1", 4);
    control[0x866] = 1; control[0x867] = 1;   // storage buffers, texture handles; roots/shared remain zero
    REQUIRE((uint8_t)CALL(nvnProgramInitialize, U(&program), U(&device)), "ProgramInitialize"); program_live = 1;
    NVNshaderData shader = {U(pool_probe_code), control};
    REQUIRE((uint8_t)CALL(nvnProgramSetShaders, U(&program), 1, U(&shader)), "original ProgramSetShaders");
    REQUIRE((uint8_t)CALL(nvnCommandBufferInitialize, U(&command), U(&device)), "CommandBufferInitialize"); command_live = 1;
    CALL(nvnCommandBufferAddCommandMemory, U(&command), U(&memory_pool), COMMAND_OFF, 0x10000);
    size_t stride = vk_ctx()->props.limits.minStorageBufferOffsetAlignment;
    if (stride < 256) stride = 256;
    REQUIRE(!(stride & (stride - 1)) && stride <= OUTPUT_BYTES / 8, "bounded aligned output stride");
    static const uint32_t trip[3][4] = {{0x3f800000, 0, 0, 0x3f800000}, {0, 0x3f800000, 0, 0x3f800000},
                                       {0x3f800000, 0, 0, 0x3f800000}};
    const NVNtexturePool *tp[3] = {&texture_a, &texture_b, &texture_a};
    const NVNsamplerPool *sp[3] = {&sampler_repeat, &sampler_repeat, &sampler_repeat};
    REQUIRE(run_case("one-table red control", memory, command_control, handle, tp, sp, trip, 1, stride), "control completed");
    // No registration occurs between A -> B -> A. The handle and program stay unchanged.
    CALL(nvnTexturePoolRegisterTexture, U(&texture_b), ti, U(&green_texture), 0);
    REQUIRE(run_case("texture A/B/A", memory, command_control, handle, tp, sp, trip, 3, stride), "texture case completed");
    CALL(nvnTexturePoolRegisterTexture, U(&texture_a), ti, U(&pattern_texture), 0);
    CALL(nvnSamplerPoolRegisterSampler, U(&sampler_clamp), si, U(&clamp_sampler));
    tp[1] = &texture_a; sp[1] = &sampler_clamp;
    REQUIRE(run_case("sampler repeat/clamp/repeat", memory, command_control, handle, tp, sp, trip, 3, stride), "sampler case completed");
    REQUIRE(gfx_shader_failures() == shader_failures, "no shader translation failures");
done:
    if (pool_live) backend->finish();
    if (command_live) {
        if (command_control) CALL(nvnCommandBufferAddControlMemory, U(&command), U(command_control), CONTROL_BYTES);
        CALL(nvnCommandBufferFinalize, U(&command));
    }
    if (program_live) CALL(nvnProgramFinalize, U(&program));
    if (descriptor_pools_live >= 4) CALL(nvnSamplerPoolFinalize, U(&sampler_clamp));
    if (descriptor_pools_live >= 3) CALL(nvnSamplerPoolFinalize, U(&sampler_repeat));
    if (descriptor_pools_live >= 2) CALL(nvnTexturePoolFinalize, U(&texture_b));
    if (descriptor_pools_live >= 1) CALL(nvnTexturePoolFinalize, U(&texture_a));
    if (samplers_live >= 2) CALL(nvnSamplerFinalize, U(&clamp_sampler));
    if (samplers_live >= 1) CALL(nvnSamplerFinalize, U(&repeat_sampler));
    if (textures_live >= 3) CALL(nvnTextureFinalize, U(&pattern_texture));
    if (textures_live >= 2) CALL(nvnTextureFinalize, U(&green_texture));
    if (textures_live >= 1) CALL(nvnTextureFinalize, U(&red_texture));
    if (pool_live) CALL(nvnMemoryPoolFinalize, U(&memory_pool));
    if (device_live) CALL(nvnDeviceFinalize, U(&device));
    vk_pipeline_shutdown();
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-vkpool: %s (%u checks, %u failures, %ld validation errors)\n",
        failures || errors ? "FAILED" : "passed", checks, failures, errors);
    printf("probe-vkpool: same-type compute resources; graphics bind-skip and type-changing answers are not covered\n");
    return failures || errors;
}

// --probe-vkcopy-astc: original valid ASTC blocks through OP_COPY_B2T/OP_COPY_T2T and the real Vulkan replay.
// No game, shader census or translated guest program. Requires GPU decode to RGBA8, synchronous replay and no splits.
// The first texture is uploaded normally and waited for before replay: first-decode table staging in a replay
// submission has a separate known lifetime gap. This probe deliberately does not exercise that gap.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "tex.h"
#include "vk_exec.h"

enum { SOURCE_BYTES = 0x1000, TEXTURE_OFFSET = 0x10000, POOL_BYTES = 0x100000 };
static const uint8_t sentinel[4] = {17, 34, 51, 255};
static const uint8_t colors[8][4] = {
    {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255},
    {0, 255, 255, 255}, {255, 0, 255, 255}, {255, 255, 0, 255}, {0, 0, 0, 255},
};

typedef struct {
    const char *name;
    uint32_t width, height, levels, layers, source_offset;
    int32_t row_stride;
    int has_view, copied;
    view_t view;
    NVNcopyRegion region;
} CopyCase;

#define REQUIRE(cond, ...) do { \
    if (!(cond)) { printf("probe-vkcopy-astc: FAIL %s: ", one->name); printf(__VA_ARGS__); \
        printf("\n"); goto done; } \
    (*checks)++; \
} while (0)

// An LDR void-extent block with all four extent coordinates 0x1fff. Each channel is exact UNORM16 byte * 257.
// This is a constant-color ASTC block, independent of the decoder's own color/weight reconstruction.
static void solid_block(uint8_t block[16], const uint8_t rgba[4]) {
    const uint64_t low = UINT64_C(0xfffffffffffffdfc);
    memcpy(block, &low, 8);
    for (unsigned c = 0; c < 4; c++) {
        uint16_t value = (uint16_t)(rgba[c] * 257u);
        memcpy(block + 8 + 2 * c, &value, 2);
    }
}

static uint32_t level_size(uint32_t n, uint32_t level) { n >>= level; return n ? n : 1; }

// Every mip/layer starts with a distinct nonzero sentinel. nvn_swizzle supplies the normal texture upload layout;
// the separate buffer-to-texture source below is linear ASTC blocks, as the NVN copy command requires.
static int fill_storage(tex_t *t) {
    uint8_t blocks[8 * 8 * 16];
    for (uint32_t level = 0; level < t->d.levels; level++) {
        uint32_t w = level_size(t->d.width, level), h = level_size(t->d.height, level);
        uint32_t columns = (w + 3) / 4, rows = (h + 3) / 4;
        if ((uint64_t)columns * rows * 16 > sizeof blocks) return 0;
        for (uint32_t i = 0; i < columns * rows; i++) solid_block(blocks + i * 16, sentinel);
        for (uint32_t layer = 0; layer < t->d.depth; layer++)
            if (nvn_swizzle(&t->d, level, layer, blocks, columns * 16, t->mem)) return 0;
    }
    return 1;
}

static void fill_source(uint8_t *memory, const CopyCase *one) {
    memset(memory, 0xa5, SOURCE_BYTES);
    const uint8_t padding[4] = {241, 129, 65, 255};
    uint32_t row = one->row_stride ? (uint32_t)one->row_stride : 32;
    for (int z = 0; z < one->region.depth; z++)
        for (unsigned y = 0; y < 2; y++) {
            uint8_t *at = memory + one->source_offset + (uint32_t)z * row * 2 + y * row;
            solid_block(at, colors[z * 4 + y * 2]);
            solid_block(at + 16, colors[z * 4 + y * 2 + 1]);
            for (unsigned off = 32; off < row; off += 16) solid_block(at + off, padding);
        }
}

static uint8_t *put_op(uint8_t *p, int op, const void *payload, size_t size) {
    size_t n = (sizeof(op_hdr) + size + 7) & ~(size_t)7;
    memset(p, 0, n);
    *(op_hdr *)p = (op_hdr){(uint16_t)op, (uint32_t)n};
    if (size) memcpy(p + sizeof(op_hdr), payload, size);
    return p + n;
}

static void read_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// Read the actual decoded image, not read_texels: that API intentionally rejects ASTC image readback.
static int read_level(const VkTexImage *image, uint32_t level, uint8_t *out) {
    VkCtx *c = vk_ctx();
    uint32_t w = level_size(image->width, level), h = level_size(image->height, level);
    VkDeviceSize size = (VkDeviceSize)w * h * image->layers * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo mapped;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buffer, &allocation, &mapped) != VK_SUCCESS) return 0;
    VkCommandBuffer cb = vk_upload_begin();
    read_barrier(cb);
    VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, image->layers},
        .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, image->image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
    read_barrier(cb);
    vk_upload_end();
    vk_upload_flush(1);
    VkResult result = vmaInvalidateAllocation(c->vma, allocation, 0, VK_WHOLE_SIZE);
    if (result == VK_SUCCESS) memcpy(out, mapped.pMappedData, (size_t)size);
    vmaDestroyBuffer(c->vma, buffer, allocation);
    return result == VK_SUCCESS;
}

static const uint8_t *expected(const CopyCase *one, uint32_t level, uint32_t layer, uint32_t x, uint32_t y) {
    uint32_t target_level = one->has_view ? one->view.base_level : 0;
    uint32_t first_layer = (one->has_view ? one->view.min_layer : 0) + (uint32_t)one->region.zoffset;
    uint32_t rx = (uint32_t)one->region.xoffset, ry = (uint32_t)one->region.yoffset;
    if (!one->copied || level != target_level || layer < first_layer ||
        layer - first_layer >= (uint32_t)one->region.depth || x < rx || x >= rx + 8 || y < ry || y >= ry + 8)
        return sentinel;
    return colors[(layer - first_layer) * 4 + ((y - ry) / 4) * 2 + (x - rx) / 4];
}

static int copy_case(const GfxBackend *backend, uint8_t *guest, const CopyCase *one, unsigned index, unsigned *checks) {
    int bad = 1, initialized = 0;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest, .size = POOL_BYTES, .game_mem = guest};
    tex_t texture;
    memset(&texture, 0, sizeof texture);
    uint8_t *pixels = NULL;
    uint8_t source_before[SOURCE_BYTES];
    REQUIRE(backend->pool_init(&pool), "CPU-visible pool initialization");
    initialized = 1;
    REQUIRE(vk_astc_mode() == VK_ASTC_GPU && vk_astc_gpu() && !vk_astc_bc(),
        "GPU ASTC decode to RGBA8 is unavailable");
    REQUIRE(!gfx_replay_async(), "requires the synchronous copy replay path");
    texture.magic = NVN_TEX_MAGIC;
    texture.d.target = one->layers == 1 ? NVN_TEXTURE_TARGET_2D : NVN_TEXTURE_TARGET_2D_ARRAY;
    texture.d.format = NVN_FORMAT_RGBA_ASTC_4x4;
    texture.d.width = one->width, texture.d.height = one->height;
    texture.d.depth = one->layers, texture.d.levels = one->levels;
    nvn_tex_layout(&texture.d, &texture.L);
    texture.pool = &pool, texture.off = TEXTURE_OFFSET, texture.mem = pool.mem + TEXTURE_OFFSET;
    texture.serial = 0x41535400u + index;
    REQUIRE(texture.L.size <= POOL_BYTES - TEXTURE_OFFSET, "texture layout leaves its owned pool");
    REQUIRE(fill_storage(&texture), "sentinel storage layout");
    fill_source(pool.mem, one);
    memcpy(source_before, pool.mem, sizeof source_before);

    // First-use decode and its table staging belong to this normal upload batch, never to the replay below.
    uint64_t warm_before, warm_after, ignored_texels;
    vk_astc_stats(&warm_before, &ignored_texels);
    const VkTexImage *image = vk_texture_get(&texture);
    REQUIRE(image && image->format == VK_FORMAT_R8G8B8A8_UNORM && image->layers == one->layers &&
        image->levels == one->levels, "requires decoded RGBA8 with every requested mip/layer");
    vk_upload_flush(1);
    vk_astc_stats(&warm_after, &ignored_texels);
    REQUIRE(warm_after > warm_before, "normal upload did not warm the GPU decoder");
    int owned_before = !!image->gpu_owned;
    REQUIRE(!owned_before, "normal initial upload unexpectedly owns guest storage");
    uint64_t before, after;
    vk_astc_stats(&before, &ignored_texels);
    VkMemSpan source_span;
    REQUIRE(vk_mem_resolve((uint64_t)(uintptr_t)(pool.mem + one->source_offset), 64, &source_span), "source span");
    REQUIRE(source_span.offset == one->source_offset, "buffer offset %llu expected %u",
        (unsigned long long)source_span.offset, one->source_offset);

    _Alignas(8) uint8_t stream[512];
    uint8_t *end = put_op(stream, OP_BEGIN, NULL, 0);
    const p_copy_stride stride = {one->row_stride};
    end = put_op(end, OP_COPY_STRIDE, &stride, sizeof stride);
    p_copy_b2t copy;
    memset(&copy, 0, sizeof copy);
    copy.src = (uint64_t)(uintptr_t)(pool.mem + one->source_offset), copy.tex = &texture;
    copy.has_view = (uint8_t)one->has_view, copy.view = one->view, copy.r = one->region;
    end = put_op(end, OP_COPY_B2T, &copy, sizeof copy);
    end = put_op(end, OP_END, NULL, 0);
    REQUIRE((size_t)(end - stream) <= sizeof stream, "recording size");
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t fence = backend->submit(&handle, 1, rt_ticks());
    REQUIRE(backend->fence_wait(fence, 5000000000ull), "copy fence wait");
    vk_astc_stats(&after, &ignored_texels);
    uint64_t want_dispatches = one->copied ? (uint32_t)one->region.depth : 0;
    REQUIRE(after - before == want_dispatches, "ASTC dispatch delta %llu expected %llu",
        (unsigned long long)(after - before), (unsigned long long)want_dispatches);
    image = vk_texture_get(&texture);   // a subsequent normal lookup must retain the copy, not re-upload the sentinel
    REQUIRE(image && !!image->gpu_owned == (one->copied ? 1 : owned_before), "ownership after subsequent lookup");
    REQUIRE(!memcmp(source_before, pool.mem, sizeof source_before), "source/prefix/padding/suffix bytes changed");
    pixels = malloc((size_t)one->width * one->height * one->layers * 4);
    REQUIRE(pixels, "readback allocation");
    for (uint32_t level = 0; level < one->levels; level++) {
        uint32_t w = level_size(one->width, level), h = level_size(one->height, level);
        REQUIRE(read_level(image, level, pixels), "direct image readback mip %u", level);
        for (uint32_t layer = 0; layer < one->layers; layer++)
            for (uint32_t y = 0; y < h; y++)
                for (uint32_t x = 0; x < w; x++) {
                    const uint8_t *got = pixels + (((size_t)layer * h + y) * w + x) * 4;
                    const uint8_t *want = expected(one, level, layer, x, y);
                    REQUIRE(!memcmp(got, want, 4), "mip %u layer %u pixel %u,%u = %u,%u,%u,%u expected %u,%u,%u,%u",
                        level, layer, x, y, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                }
    }
    printf("probe-vkcopy-astc: %s passed (offset %u, row stride %d, %llu decoder dispatches, ownership %d)\n",
        one->name, one->source_offset, one->row_stride, (unsigned long long)(after - before), one->copied);
    if (!one->copied)
        printf("probe-vkcopy-astc: unsupported padded-row policy retained pixels/ownership without a decode\n");
    bad = 0;
done:
    if (initialized) {
        backend->finish();
        backend->texture_release(&texture);
        backend->pool_fini(&pool);
    }
    free(pixels);
    return bad;
}

typedef struct {
    const char *name;
    uint32_t width, height, levels, source_layers, destination_layers;
    int has_sv, has_dv, copied, same_image;
    view_t sv, dv;
    NVNcopyRegion sr, dr;
} T2tCase;

static const uint8_t *t2t_color(uint32_t level, uint32_t layer, uint32_t x, uint32_t y) {
    return colors[(level * 3 + layer * 5 + (y / 8) * 2 + x / 8) % 8];
}

static int fill_t2t_storage(tex_t *t, int source) {
    uint8_t blocks[4 * 4 * 16];
    for (uint32_t level = 0; level < t->d.levels; level++) {
        uint32_t w = level_size(t->d.width, level), h = level_size(t->d.height, level);
        uint32_t columns = (w + 7) / 8, rows = (h + 7) / 8;
        if ((uint64_t)columns * rows * 16 > sizeof blocks) return 0;
        for (uint32_t layer = 0; layer < t->d.depth; layer++) {
            for (uint32_t y = 0; y < rows; y++)
                for (uint32_t x = 0; x < columns; x++)
                    solid_block(blocks + (y * columns + x) * 16,
                        source ? t2t_color(level, layer, x * 8, y * 8) : sentinel);
            if (nvn_swizzle(&t->d, level, layer, blocks, columns * 16, t->mem)) return 0;
        }
    }
    return 1;
}

static const uint8_t *t2t_expected(const T2tCase *one, uint32_t level, uint32_t layer, uint32_t x, uint32_t y) {
    uint32_t dl = one->has_dv && one->dv.set_levels ? one->dv.base_level : 0;
    uint32_t first = (one->has_dv && one->dv.set_layers ? one->dv.min_layer : 0) + (uint32_t)one->dr.zoffset;
    uint32_t dx = (uint32_t)one->dr.xoffset, dy = (uint32_t)one->dr.yoffset;
    if (!one->copied || level != dl || layer < first || layer - first >= (uint32_t)one->dr.depth ||
        x < dx || x - dx >= (uint32_t)one->dr.width || y < dy || y - dy >= (uint32_t)one->dr.height) return sentinel;
    uint32_t sl = one->has_sv && one->sv.set_levels ? one->sv.base_level : 0;
    uint32_t source_layer = (one->has_sv && one->sv.set_layers ? one->sv.min_layer : 0) +
        (uint32_t)one->sr.zoffset + layer - first;
    return t2t_color(sl, source_layer, (uint32_t)one->sr.xoffset + x - dx, (uint32_t)one->sr.yoffset + y - dy);
}

static int t2t_case(const GfxBackend *backend, uint8_t *guest, const T2tCase *one, unsigned index, unsigned *checks) {
    int bad = 1, initialized[2] = {0};
    pool_t pools[2] = {{0}};
    tex_t textures[2] = {{0}};
    const VkTexImage *images[2] = {0};
    uint8_t *pixels = NULL, *storage_before = NULL;
    REQUIRE(vk_astc_mode() == VK_ASTC_GPU && vk_astc_gpu() && !vk_astc_bc(), "requires GPU ASTC decode to RGBA8");
    for (unsigned i = 0; i < 2; i++) {
        pool_t *pool = &pools[i];
        *pool = (pool_t){.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + i * POOL_BYTES,
            .size = POOL_BYTES, .game_mem = guest + i * POOL_BYTES};
        REQUIRE(backend->pool_init(pool), "T2T pool %u initialization", i);
        initialized[i] = 1;
        memset(pool->mem, 0xa5, POOL_BYTES);
        tex_t *t = &textures[i];
        t->magic = NVN_TEX_MAGIC;
        t->d.target = (i ? one->destination_layers : one->source_layers) == 1 ?
            NVN_TEXTURE_TARGET_2D : NVN_TEXTURE_TARGET_2D_ARRAY;
        t->d.format = NVN_FORMAT_RGBA_ASTC_8x8_SRGB;
        t->d.width = one->width, t->d.height = one->height, t->d.levels = one->levels;
        t->d.depth = i ? one->destination_layers : one->source_layers;
        t->pool = pool, t->off = TEXTURE_OFFSET, t->mem = pool->mem + TEXTURE_OFFSET;
        t->serial = 0x54325400u + index * 2 + i;
        nvn_tex_layout(&t->d, &t->L);
        REQUIRE(t->L.size <= POOL_BYTES - TEXTURE_OFFSET, "T2T texture %u storage bounds", i);
        REQUIRE(fill_t2t_storage(t, !i), "T2T texture %u original block layout", i);
        images[i] = vk_texture_get(t);
        REQUIRE(images[i] && images[i]->format == VK_FORMAT_R8G8B8A8_SRGB && !images[i]->bound &&
            images[i]->type == VK_IMAGE_TYPE_2D && images[i]->samples == 1 &&
            images[i]->levels == one->levels && images[i]->layers == t->d.depth && !images[i]->gpu_owned,
            "T2T texture %u must be a separately allocated decoded RGBA8 sRGB image", i);
    }
    REQUIRE(images[0]->image != images[1]->image, "T2T images must be distinct");
    vk_upload_flush(1);   // normal uploads, including first-decode staging, finish before replay
    pixels = malloc((size_t)one->width * one->height *
        (one->source_layers > one->destination_layers ? one->source_layers : one->destination_layers) * 4);
    storage_before = malloc(2 * POOL_BYTES);
    REQUIRE(pixels && storage_before, "T2T comparison allocations");
    memcpy(storage_before, pools[0].mem, POOL_BYTES);
    memcpy(storage_before + POOL_BYTES, pools[1].mem, POOL_BYTES);
    uint64_t before, after, ignored_texels;
    vk_astc_stats(&before, &ignored_texels);
    _Alignas(8) uint8_t stream[512];
    uint8_t *end = put_op(stream, OP_BEGIN, NULL, 0);
    p_copy_t2t copy = {.src = &textures[0], .dst = &textures[one->same_image ? 0 : 1],
        .sv = one->sv, .dv = one->dv, .has_sv = (uint8_t)one->has_sv, .has_dv = (uint8_t)one->has_dv,
        .sr = one->sr, .dr = one->dr};
    end = put_op(end, OP_COPY_T2T, &copy, sizeof copy);
    end = put_op(end, OP_END, NULL, 0);
    REQUIRE((size_t)(end - stream) <= sizeof stream, "T2T recording size");
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t fence = backend->submit(&handle, 1, rt_ticks());
    REQUIRE(backend->fence_wait(fence, 5000000000ull), "T2T fence wait");
    vk_astc_stats(&after, &ignored_texels);
    REQUIRE(after == before, "T2T must copy existing RGBA8 texels without another ASTC decode");
    for (unsigned i = 0; i < 2; i++) {
        images[i] = vk_texture_get(&textures[i]);
        REQUIRE(images[i], "T2T texture %u after subsequent lookup", i);
        REQUIRE(!memcmp(storage_before + i * POOL_BYTES, pools[i].mem, POOL_BYTES),
            "T2T texture %u guest storage/pool canaries changed", i);
        for (uint32_t level = 0; level < one->levels; level++) {
            uint32_t w = level_size(one->width, level), h = level_size(one->height, level);
            REQUIRE(read_level(images[i], level, pixels), "T2T texture %u mip %u direct readback", i, level);
            for (uint32_t layer = 0; layer < images[i]->layers; layer++)
                for (uint32_t y = 0; y < h; y++)
                    for (uint32_t x = 0; x < w; x++) {
                        const uint8_t *got = pixels + (((size_t)layer * h + y) * w + x) * 4;
                        const uint8_t *want = i ? t2t_expected(one, level, layer, x, y) : t2t_color(level, layer, x, y);
                        REQUIRE(!memcmp(got, want, 4),
                            "T2T texture %u mip %u layer %u pixel %u,%u = %u,%u,%u,%u expected %u,%u,%u,%u",
                            i, level, layer, x, y, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                    }
        }
        REQUIRE(!!images[i]->gpu_owned == (int)(i && one->copied),
            "T2T texture %u ownership after subsequent lookup", i);
    }
    printf("probe-vkcopy-astc: %s passed (%s; source and all untouched pixels/mips/layers preserved)\n",
        one->name, one->copied ? "T2T copied" : "unsupported T2T policy skipped");
    bad = 0;
done:
    backend->finish();
    for (unsigned i = 0; i < 2; i++)
        if (initialized[i]) {
            backend->texture_release(&textures[i]);
            backend->pool_fini(&pools[i]);
        }
    free(storage_before);
    free(pixels);
    return bad;
}

void vk_exec_draw_shutdown(void);
void vk_exec_xfer_shutdown(void);

int probe_vkcopy_astc(void) {
    // Keep configuration explicit: otherwise a passing CPU/native/BC fallback would not cover this decoder path.
    if (strcmp(cfg_value("TOTK_ASTC"), "gpu") || cfg_bool("TOTK_ASTC_BC") ||
        cfg_bool("TOTK_REPLAY_ASYNC") || strcmp(cfg_value("TOTK_SPLIT_DRAWS"), "0")) {
        printf("probe-vkcopy-astc: requires TOTK_ASTC=gpu TOTK_ASTC_BC=0 TOTK_REPLAY_ASYNC=0 TOTK_SPLIT_DRAWS=0\n");
        return 1;
    }
    uint64_t heap = 0;
    if (mem_set_heap_size(16ull << 20, &heap) || !heap) return 1;
    const CopyCase cases[] = {
        {.name = "tight rows", .width = 8, .height = 8, .levels = 1, .layers = 1, .copied = 1,
            .region = {0, 0, 0, 8, 8, 1}},
        {.name = "explicit tight rows/nonzero buffer offset", .width = 8, .height = 8, .levels = 1, .layers = 1,
            .source_offset = 0x100, .row_stride = 32, .copied = 1, .region = {0, 0, 0, 8, 8, 1}},
        {.name = "padded rows skipped", .width = 8, .height = 8, .levels = 1, .layers = 1,
            .source_offset = 0x100, .row_stride = 64, .region = {0, 0, 0, 8, 8, 1}},
        {.name = "two layers/derived tight layer pitch", .width = 8, .height = 8, .levels = 1, .layers = 2,
            .source_offset = 0x100, .copied = 1, .region = {0, 0, 0, 8, 8, 2}},
        {.name = "view/mip/layer/subrectangle", .width = 32, .height = 32, .levels = 2, .layers = 4,
            .source_offset = 0x100, .has_view = 1, .copied = 1,
            .view = {.set_levels = 1, .base_level = 1, .nlevels = 1, .set_layers = 1, .min_layer = 1, .nlayers = 3},
            .region = {4, 4, 1, 8, 8, 2}},
    };
    const T2tCase t2t_cases[] = {
        {.name = "T2T 8x8 sRGB 2D to array", .width = 16, .height = 16, .levels = 1,
            .source_layers = 1, .destination_layers = 2, .copied = 1,
            .sr = {0, 0, 0, 16, 16, 1}, .dr = {0, 0, 0, 16, 16, 1}},
        {.name = "T2T offset subrectangle", .width = 32, .height = 32, .levels = 3,
            .source_layers = 1, .destination_layers = 2, .copied = 1,
            .sr = {8, 0, 0, 16, 16, 1}, .dr = {0, 8, 1, 16, 16, 1}},
        {.name = "T2T selected mip and view layer", .width = 32, .height = 32, .levels = 3,
            .source_layers = 1, .destination_layers = 3, .copied = 1, .has_sv = 1, .has_dv = 1,
            .sv = {.set_levels = 1, .base_level = 1, .nlevels = 1},
            .dv = {.set_levels = 1, .base_level = 1, .nlevels = 1, .set_layers = 1, .min_layer = 1, .nlayers = 2},
            .sr = {0, 0, 0, 16, 16, 1}, .dr = {0, 0, 0, 16, 16, 1}},
        {.name = "T2T two layers with view offsets", .width = 16, .height = 16, .levels = 2,
            .source_layers = 4, .destination_layers = 4, .copied = 1, .has_sv = 1, .has_dv = 1,
            .sv = {.set_layers = 1, .min_layer = 1, .nlayers = 3},
            .dv = {.set_layers = 1, .min_layer = 0, .nlayers = 3},
            .sr = {0, 0, 1, 16, 16, 2}, .dr = {0, 0, 1, 16, 16, 2}},
        {.name = "T2T 4x4 mip edge", .width = 32, .height = 32, .levels = 6,
            .source_layers = 1, .destination_layers = 2, .copied = 1, .has_sv = 1, .has_dv = 1,
            .sv = {.set_levels = 1, .base_level = 3, .nlevels = 1},
            .dv = {.set_levels = 1, .base_level = 3, .nlevels = 1},
            .sr = {0, 0, 0, 4, 4, 1}, .dr = {0, 0, 1, 4, 4, 1}},
        {.name = "T2T 2x2 mip edge", .width = 32, .height = 32, .levels = 6,
            .source_layers = 1, .destination_layers = 2, .copied = 1, .has_sv = 1, .has_dv = 1,
            .sv = {.set_levels = 1, .base_level = 4, .nlevels = 1},
            .dv = {.set_levels = 1, .base_level = 4, .nlevels = 1},
            .sr = {0, 0, 0, 2, 2, 1}, .dr = {0, 0, 1, 2, 2, 1}},
        {.name = "T2T 1x1 mip edge", .width = 32, .height = 32, .levels = 6,
            .source_layers = 1, .destination_layers = 2, .copied = 1, .has_sv = 1, .has_dv = 1,
            .sv = {.set_levels = 1, .base_level = 5, .nlevels = 1},
            .dv = {.set_levels = 1, .base_level = 5, .nlevels = 1},
            .sr = {0, 0, 0, 1, 1, 1}, .dr = {0, 0, 1, 1, 1, 1}},
        {.name = "T2T explicit format view skipped", .width = 16, .height = 16, .levels = 1,
            .source_layers = 1, .destination_layers = 2, .has_sv = 1,
            .sv = {.set_format = 1, .format = NVN_FORMAT_RGBA_ASTC_8x8_SRGB},
            .sr = {0, 0, 0, 16, 16, 1}, .dr = {0, 0, 0, 16, 16, 1}},
        {.name = "T2T nonoverlapping self-copy skipped", .width = 16, .height = 16, .levels = 1,
            .source_layers = 1, .destination_layers = 2, .same_image = 1,
            .sr = {0, 0, 0, 8, 8, 1}, .dr = {8, 8, 0, 8, 8, 1}},
    };
    printf("probe-vkcopy-astc: normal upload warmup precedes replay; first-decode replay lifetime gap excluded\n");
    unsigned checks = 0, completed = 0, skipped = 0;
    int bad = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0] && !bad; i++) {
        bad = copy_case(&gfx_vulkan_backend, (uint8_t *)(uintptr_t)heap, &cases[i], i, &checks);
        if (!bad) completed++, skipped += !cases[i].copied;
    }
    for (unsigned i = 0; i < sizeof t2t_cases / sizeof t2t_cases[0] && !bad; i++) {
        bad = t2t_case(&gfx_vulkan_backend, (uint8_t *)(uintptr_t)heap, &t2t_cases[i], i, &checks);
        if (!bad) completed++, skipped += !t2t_cases[i].copied;
    }
    gfx_vulkan_backend.finish();
    vk_pipeline_shutdown();
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    unsigned total = sizeof cases / sizeof cases[0] + sizeof t2t_cases / sizeof t2t_cases[0];
    printf("probe-vkcopy-astc: %s (%u/%u cases, %u intentional skips, %u checks, %ld validation errors)\n",
        bad || errors ? "FAILED" : "passed", completed, total, skipped, checks, errors);
    printf("probe-vkcopy-astc: independent image pitch, compute-program restoration, native/CPU/BC modes not covered\n");
    return bad || errors;
}

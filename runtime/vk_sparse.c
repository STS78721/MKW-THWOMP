// Sparse textures (NVN_TEXTURE_FLAGS_SPARSE in a virtual pool) as Vulkan sparse images.
// Ported from adriadam10/mkw-recomp runtime/gfx/vk/tex/vk_sparse.c (MIT License, Copyright (c) 2026 Adrià),
// adapted to MKW-THWOMP's 4 KiB virtual-pool page table (vk_mem_virt_phys over 64 KiB SPARSE_PAGE spans) and sync queue.
//
// Each 64 KiB page mapped by nvnMemoryPoolMapVirtual binds the Vulkan sparse blocks holding those texels to zeroed
// 64 KiB device-memory slots keyed by (physical pool + offset, sub-block index, VkFormat), so virtual pages aliasing
// the same physical page share texels just as on Switch 2. Levels smaller than a whole NVN tile (including the Vulkan
// mip tail) receive dedicated tail memory (tail_alloc) so streamed/copied coarse mips (foliage, distant terrain) stay
// intact rather than reading zero.
//
// Sparse binds serialize with graphics-queue submissions and upload batches via timeline semaphore order_tl
// (vk_sparse_order under vk_queue_lock), preventing PTE-update races and RTX 50-series mem_mapper.c unmap faults.
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "vk.h"
#include "platform/platform.h"

#define SPARSE_PAGE 0x10000ull     // 64 KiB NVN sparse tile page (16 x 4 KiB NVN_VPOOL_PAGE entries)
#define BLOCK_SLOTS 256            // 16 MiB of 64 KiB slots
#define SMALL_SPARSE (4u << 20)    // storage of an image that gets slots past MKW_SPARSE_MIB

typedef struct SlotBlock {
    VkBuffer buf;                 // over the whole block, for the zero fill
    VmaAllocation alloc;
    VkDeviceMemory mem;
    VkDeviceSize base;            // offset of the block in mem
    VkDeviceSize slot_size;       // bytes of each of its slots
    uint32_t type_bits;           // memory types of the images it serves
    uint32_t used;
    uint8_t busy[BLOCK_SLOTS];
    struct SlotBlock *next;
} SlotBlock;

typedef struct { uint64_t pa; uint32_t sub; int32_t fmt; } SlotKey;
typedef struct Slot {
    SlotKey key;
    SlotBlock *blk;
    uint32_t idx;
    int32_t refs;                 // block bindings that name it
    int32_t reaps;                // frees queued behind a submission (slot_reap)
    int32_t written;              // a copy or a pass wrote into it (vk_sparse_written)
} Slot;

typedef struct SparseImg {
    VkImage img;
    uint64_t addr, size;          // NVN storage range
    VkImageAspectFlags aspect;
    VkFormat fmt;
    uint32_t w, h, layers, nlev;  // nlev: levels bound per tile (whole NVN tiles and below the Vulkan mip tail)
    uint32_t ntw, nth;            // NVN tile in texels
    uint32_t gw, gh;              // Vulkan sparse block in texels
    uint32_t nnvn;                // NVN levels with whole tiles (nnvn >= nlev)
    uint32_t tiles_x[16], tiles_y[16], tile0[16], stride;   // NVN tiles a row, rows, first page of each level; pages a layer
    uint32_t bx[16], blk0[16], nblk;                        // Vulkan blocks a row and first block of each level; blocks a layer
    VkDeviceSize slot_size;
    uint32_t type_bits;
    Slot **bound;                 // layers * nblk
    VmaAllocation tail_alloc;
    VkDeviceMemory tail_mem;
    VkDeviceSize tail_base;
    uint32_t nmips, tail_end, tail_blocks;
    VkSparseImageMemoryRequirements tail_req;
    int tail_pending;
    uint32_t tail_written;
    uint64_t bind_fence;          // protects VkImage destruction until queued vkQueueBindSparse calls retire

    uint8_t *dirty;
    uint32_t *dl;
    int ndl, cdl;
    struct SparseImg *next;
} SparseImg;

static PlatMutex sp_lock = PLAT_MUTEX_INIT;
static SparseImg *images;
static nvn_hmap slots = {.klen = sizeof(SlotKey)};
static SlotBlock *blocks;
static int ndirty;
static Slot **dying;
static int ndying, cdying;
static VkSemaphore order_tl;
static VkSemaphore bind_sem;
static uint64_t order_val, bind_val;
static int binary_pending;
static int64_t n_slots, n_failed, n_binds, n_images, n_blocks, n_over, n_written, n_tail_writes, n_host_blocks;
static VmaAllocation *dying_tails;
static int ndying_tails, cdying_tails, ntail_pending;

static void grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 256;
    if (!(*arr = realloc(*arr, (size_t)*cap * elem))) rt_fatal("vk: out of memory (sparse)");
}

static uint64_t default_sparse_mib(int discrete, uint64_t vram_mib) {
    if (!discrete) return 768;
    if (vram_mib >= 14336) return 2048;   // 16 GiB+ cards (e.g. RTX 5070 Ti): ample budget for all 3 8192x8192x68 arrays
    if (vram_mib >= 10240) return 1536;   // 12 GiB cards
    if (vram_mib >= 7168)  return 1024;   // 8 GiB cards
    if (vram_mib >= 5120)  return 768;    // 6 GiB cards
    return 512;
}

static uint64_t budget_default(void) {
    VkCtx *c = vk_ctx();
    if (!c) return default_sparse_mib(0, 0);
    uint64_t vram = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount; h++)
        if ((c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && c->mem_props.memoryHeaps[h].size > vram)
            vram = c->mem_props.memoryHeaps[h].size;
    return default_sparse_mib(c->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, vram >> 20);
}

static uint64_t budget(void) {
    static uint64_t v;
    if (!v) {
        const char *e = cfg_get("MKW_SPARSE_MIB");
        uint64_t mib = e && *e ? (uint64_t)atol(e) : budget_default();
        rt_log("vk: sparse: tile slots up to %llu MiB (MKW_SPARSE_MIB%s; 0 lifts it)", (unsigned long long)mib,
               e && *e ? "" : " unset, from the GPU");
        v = mib ? mib << 20 : UINT64_MAX;
    }
    return v;
}

int vk_sparse_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = cfg_get("MKW_SPARSE_IMAGES");
        v = !(e && *e && !atoi(e));
    }
    VkCtx *c = vk_ctx();
    return v && c && c->has_sparse;
}

static VkResult alloc_local_or_any(VkCtx *c, const VkMemoryRequirements *req, VmaAllocation *a, VmaAllocationInfo *info) {
    if (!req->memoryTypeBits) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VmaAllocationCreateInfo aci = {.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkResult r = vmaAllocateMemory(c->vma, req, &aci, a, info);
    if (r == VK_SUCCESS) return r;
    aci = (VmaAllocationCreateInfo){.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if ((r = vmaAllocateMemory(c->vma, req, &aci, a, info)) == VK_SUCCESS && !n_host_blocks++)
        rt_log("vk: sparse: memory outside the GPU's own (device-local full or too small)");
    return r;
}

static SlotBlock *new_block(VkCtx *c, uint32_t type_bits, VkDeviceSize slot_size) {
    SlotBlock *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("vk: out of memory (sparse block)");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = slot_size * BLOCK_SLOTS,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(c->device, &bi, NULL, &b->buf) != VK_SUCCESS) { free(b); return NULL; }
    VkMemoryRequirements rb;
    vkGetBufferMemoryRequirements(c->device, b->buf, &rb);
    VkMemoryRequirements req = {.size = slot_size * BLOCK_SLOTS, .alignment = slot_size > rb.alignment ? slot_size : rb.alignment,
                                .memoryTypeBits = rb.memoryTypeBits & type_bits};
    VmaAllocationInfo info;
    if (alloc_local_or_any(c, &req, &b->alloc, &info) != VK_SUCCESS || vmaBindBufferMemory(c->vma, b->alloc, b->buf) != VK_SUCCESS) {
        if (b->alloc) vmaFreeMemory(c->vma, b->alloc);
        vkDestroyBuffer(c->device, b->buf, NULL);
        free(b);
        return NULL;
    }
    b->mem = info.deviceMemory, b->base = info.offset, b->type_bits = type_bits, b->slot_size = slot_size;
    n_blocks++;
    b->next = blocks, blocks = b;
    return b;
}

static Slot *slot_get(VkCtx *c, const SparseImg *s, uint64_t pa, uint32_t sub) {
    SlotKey k = {pa, sub, (int32_t)s->fmt};
    Slot *sl = nvn_hm_get(&slots, &k);
    if (sl) return sl;
    SlotBlock *b = blocks;
    while (b && (b->used == BLOCK_SLOTS || b->type_bits != s->type_bits || b->slot_size != s->slot_size)) b = b->next;
    int big = s->size > SMALL_SPARSE;
    if (big && ((uint64_t)(n_slots + 1) * s->slot_size > budget() ||
                (!b && (uint64_t)(n_blocks + 1) * BLOCK_SLOTS * s->slot_size > budget()))) {
        if (++n_over <= 4 || !(n_over & (n_over - 1)))
            rt_log("vk: sparse: tile slots over MKW_SPARSE_MIB (%lld MiB in blocks; %lld tiles left non-resident)",
                   (long long)(budget() >> 20), (long long)n_over);
        return NULL;
    }
    if (!b && !(b = new_block(c, s->type_bits, s->slot_size))) {
        if (++n_failed <= 8 || !(n_failed & (n_failed - 1)))
            rt_log("vk: sparse: no memory for a tile slot (%lld failed, %lld slots live)", (long long)n_failed,
                   (long long)n_slots);
        return NULL;
    }
    uint32_t i = 0;
    while (b->busy[i]) i++;
    b->busy[i] = 1, b->used++;
    if (!(sl = calloc(1, sizeof *sl))) rt_fatal("vk: out of memory (sparse slot)");
    sl->key = k, sl->blk = b, sl->idx = i;
    nvn_hm_put(&slots, &k, sl);
    n_slots++;
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdFillBuffer(cb, b->buf, (VkDeviceSize)i * s->slot_size, s->slot_size, 0);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &dep);
    vk_upload_end();
    return sl;
}

static void slot_unref(Slot *sl) {
    if (--sl->refs) return;
    if (ndying == cdying) grow((void **)&dying, &cdying, sizeof *dying);
    dying[ndying++] = sl;
    sl->reaps++;
}

static void slot_reap(void *arg) {
    Slot *sl = arg;
    plat_mutex_lock(&sp_lock);
    if (!--sl->reaps && !sl->refs && sl->blk) {
        nvn_hm_del(&slots, &sl->key);
        sl->blk->busy[sl->idx] = 0, sl->blk->used--;
        n_slots--;
        n_written -= sl->written;
        free(sl);
    }
    plat_mutex_unlock(&sp_lock);
}

static void mark_dirty(SparseImg *s, size_t i) {
    if (s->dirty[i]) return;
    s->dirty[i] = 1;
    if (!s->ndl) ndirty++;
    if (s->ndl == s->cdl) grow((void **)&s->dl, &s->cdl, sizeof *s->dl);
    s->dl[s->ndl++] = (uint32_t)i;
}

static VkSparseImageMemoryBind make_bind(const SparseImg *s, uint32_t i) {
    uint32_t layer = i / s->nblk, r = i % s->nblk, l = 0;
    while (l + 1 < s->nlev && r >= s->blk0[l + 1]) l++;
    r -= s->blk0[l];
    uint32_t x = r % s->bx[l] * s->gw, y = r / s->bx[l] * s->gh;
    uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1;
    const Slot *sl = s->bound[i];
    return (VkSparseImageMemoryBind){
        .subresource = {s->aspect, l, layer},
        .offset = {(int32_t)x, (int32_t)y, 0},
        .extent = {lw - x < s->gw ? lw - x : s->gw, lh - y < s->gh ? lh - y : s->gh, 1},
        .memory = sl ? sl->blk->mem : VK_NULL_HANDLE,
        .memoryOffset = sl ? sl->blk->base + (VkDeviceSize)sl->idx * s->slot_size : 0};
}

static void bind_page(VkCtx *c, SparseImg *s, uint64_t p) {
    uint32_t layer = (uint32_t)(p / s->stride), rel = (uint32_t)(p % s->stride), l = 0;
    if (layer >= s->layers) return;
    while (l + 1 < s->nnvn && rel >= s->tile0[l + 1]) l++;
    if (l >= s->nlev) return;
    uint32_t t = rel - s->tile0[l], tx = t % s->tiles_x[l], ty = t / s->tiles_x[l];
    uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1;
    uint64_t pa = 0;
    int mapped = vk_mem_virt_phys(s->addr + p * SPARSE_PAGE, SPARSE_PAGE, &pa);

    for (uint32_t sy = 0; sy < s->nth / s->gh; sy++)
        for (uint32_t sx = 0; sx < s->ntw / s->gw; sx++) {
            uint32_t x = tx * s->ntw + sx * s->gw, y = ty * s->nth + sy * s->gh;
            if (x >= lw || y >= lh) continue;
            size_t i = (size_t)layer * s->nblk + s->blk0[l] + (y / s->gh) * s->bx[l] + x / s->gw;
            Slot *want = mapped ? slot_get(c, s, pa, sy * (s->ntw / s->gw) + sx) : NULL;
            if (want == s->bound[i]) continue;
            if (want) want->refs++;
            if (s->bound[i]) slot_unref(s->bound[i]);
            s->bound[i] = want;
            mark_dirty(s, i);
        }
}

static void bind_range(VkCtx *c, SparseImg *s, uint64_t lo, uint64_t hi) {
    if (lo < s->addr) lo = s->addr;
    if (hi > s->addr + s->size) hi = s->addr + s->size;
    for (uint64_t a = lo & ~(SPARSE_PAGE - 1); a < hi; a += SPARSE_PAGE) bind_page(c, s, (a - s->addr) / SPARSE_PAGE);
}

static uint32_t tail_count(const SparseImg *s) {
    if (s->tail_end == s->nmips || !s->tail_req.imageMipTailSize) return 0;
    return s->tail_req.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT ? 1 : s->layers;
}

static VkDeviceSize round_up(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

static void tail_alloc(VkCtx *c, SparseImg *s, const VkSparseImageMemoryRequirements *sr, uint32_t levels, uint32_t types) {
    s->nmips = levels, s->tail_end = sr && sr->imageMipTailFirstLod < levels ? sr->imageMipTailFirstLod : levels;
    if (sr) s->tail_req = *sr;
    for (uint32_t l = s->nlev; l < s->tail_end; l++) {
        uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1;
        s->tail_blocks += ((lw + s->gw - 1) / s->gw) * ((lh + s->gh - 1) / s->gh);
    }
    VkDeviceSize bytes = (VkDeviceSize)s->layers * s->tail_blocks * s->slot_size +
                         tail_count(s) * round_up(s->tail_req.imageMipTailSize, s->slot_size);
    if (!bytes) return;
    VkMemoryRequirements req = {.size = bytes, .alignment = s->slot_size, .memoryTypeBits = types};
    VmaAllocationInfo info;
    if (alloc_local_or_any(c, &req, &s->tail_alloc, &info) != VK_SUCCESS) {
        s->tail_alloc = NULL;
        rt_log("vk: sparse: no memory for the %llu KiB of levels outside the tiles (%ux%u layers %u)",
               (unsigned long long)(bytes >> 10), s->w, s->h, s->layers);
        return;
    }
    s->tail_mem = info.deviceMemory, s->tail_base = info.offset, s->tail_pending = 1;
    rt_log("vk: sparse: %ux%u layers %u: levels %u-%d in blocks (%u a layer), mip tail from %u: offset %#llx size %#llx "
           "stride %#llx flags %#x", s->w, s->h, s->layers, s->nlev, (int)s->tail_end - 1, s->tail_blocks, s->tail_end,
           (unsigned long long)s->tail_req.imageMipTailOffset, (unsigned long long)s->tail_req.imageMipTailSize,
           (unsigned long long)s->tail_req.imageMipTailStride, s->tail_req.formatProperties.flags);
}

static void tail_binds(const SparseImg *s, VkSparseImageMemoryBindInfo *ib, VkSparseImageMemoryBind *bb,
                       VkSparseImageOpaqueMemoryBindInfo *ob, VkSparseMemoryBind *mb) {
    uint32_t n = 0;
    VkDeviceSize off = s->tail_base;
    for (uint32_t layer = 0; layer < s->layers; layer++)
        for (uint32_t l = s->nlev; l < s->tail_end; l++) {
            uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1;
            for (uint32_t y = 0; y < lh; y += s->gh)
                for (uint32_t x = 0; x < lw; x += s->gw, off += s->slot_size)
                    bb[n++] = (VkSparseImageMemoryBind){.subresource = {s->aspect, l, layer},
                                                        .offset = {(int32_t)x, (int32_t)y, 0},
                                                        .extent = {lw - x < s->gw ? lw - x : s->gw, lh - y < s->gh ? lh - y : s->gh, 1},
                                                        .memory = s->tail_mem, .memoryOffset = off};
        }
    *ib = (VkSparseImageMemoryBindInfo){.image = s->img, .bindCount = n, .pBinds = bb};
    uint32_t nt = tail_count(s);
    for (uint32_t t = 0; t < nt; t++, off += round_up(s->tail_req.imageMipTailSize, s->slot_size))
        mb[t] = (VkSparseMemoryBind){.resourceOffset = s->tail_req.imageMipTailOffset + t * s->tail_req.imageMipTailStride,
                                     .size = s->tail_req.imageMipTailSize, .memory = s->tail_mem, .memoryOffset = off};
    *ob = (VkSparseImageOpaqueMemoryBindInfo){.image = s->img, .bindCount = nt, .pBinds = mb};
}

static void tail_free(void *arg) {
    VkCtx *c = vk_ctx();
    if (c && c->vma && arg) vmaFreeMemory(c->vma, arg);
}

static int order_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = cfg_get("MKW_SPARSE_ORDER");
        v = !(e && *e && !atoi(e));
    }
    return v;
}

static void tail_bind_now(VkCtx *c, SparseImg *s) {
    if (!s->tail_pending || !order_on()) return;
    VkSparseImageMemoryBindInfo ib;
    VkSparseImageOpaqueMemoryBindInfo ob;
    VkSparseImageMemoryBind *bb = malloc(sizeof *bb * ((size_t)s->layers * s->tail_blocks + 1));
    VkSparseMemoryBind *mb = malloc(sizeof *mb * (tail_count(s) + 1));
    if (!bb || !mb) rt_fatal("vk: out of memory (sparse tail binds)");
    tail_binds(s, &ib, bb, &ob, mb);
    vk_queue_lock();
    if (!order_tl) {
        VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
        if (vkCreateSemaphore(c->device, &si, NULL, &order_tl) != VK_SUCCESS) rt_fatal("vk: sparse order semaphore");
    }
    uint64_t wait = order_val, sig = ++order_val;
    VkTimelineSemaphoreSubmitInfo tl = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                        .waitSemaphoreValueCount = wait ? 1u : 0u, .pWaitSemaphoreValues = &wait,
                                        .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &sig};
    VkBindSparseInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .pNext = &tl,
                           .waitSemaphoreCount = wait ? 1u : 0u, .pWaitSemaphores = &order_tl,
                           .imageOpaqueBindCount = ob.bindCount ? 1u : 0u, .pImageOpaqueBinds = &ob,
                           .imageBindCount = ib.bindCount ? 1u : 0u, .pImageBinds = &ib,
                           .signalSemaphoreCount = 1, .pSignalSemaphores = &order_tl};
    VkResult r = vkQueueBindSparse(c->gfx_queue, 1, &bi, VK_NULL_HANDLE);
    bind_val = sig;
    s->bind_fence = vk_sync_issued() + 2;
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueBindSparse of the levels outside the tiles failed (%d)", (int)r);
    }
    free(bb);
    free(mb);
    s->tail_pending = 0;
}

int vk_sparse_create(const VkImageCreateInfo *ci0, const nvn_texdesc *d, uint64_t addr, uint64_t size, VkImage *out) {
    VkCtx *c = vk_ctx();
    static volatile int32_t nwarn;
    *out = VK_NULL_HANDLE;
    if (!vk_sparse_on() || !(d->flags & NVN_TEXTURE_FLAGS_SPARSE) || !nvn_gpu_only(addr) ||
        ci0->imageType != VK_IMAGE_TYPE_2D || ci0->samples != VK_SAMPLE_COUNT_1_BIT ||
        !(ci0->usage & VK_IMAGE_USAGE_SAMPLED_BIT))
        return 0;
    VkImageCreateInfo ci = *ci0;
    ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    ci.usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    uint32_t n = 0;
    VkSparseImageFormatProperties fp[4];
    vkGetPhysicalDeviceSparseImageFormatProperties(c->phys, ci.format, ci.imageType, ci.samples, ci.usage, ci.tiling, &n, NULL);
    if (n > 4) n = 4;
    vkGetPhysicalDeviceSparseImageFormatProperties(c->phys, ci.format, ci.imageType, ci.samples, ci.usage, ci.tiling, &n, fp);
    if (!n || (fp[0].flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT)) {
        if (plat_inc32(&nwarn) <= 8) rt_log("vk: sparse: VkFormat %d usage %#x has no standard sparse blocks", (int)ci.format, ci.usage);
        return 0;
    }
    int32_t lay[4];
    nvn_sparse_tile_layout(d, lay);
    if (lay[0] <= 0) return 0;
    SparseImg *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("vk: out of memory (sparse image)");
    s->addr = addr, s->size = size, s->fmt = ci.format, s->aspect = fp[0].aspectMask;
    s->w = ci.extent.width, s->h = ci.extent.height, s->layers = ci.arrayLayers;
    s->ntw = (uint32_t)lay[1], s->nth = (uint32_t)lay[2];
    s->gw = fp[0].imageGranularity.width, s->gh = fp[0].imageGranularity.height;
    if (!s->gw || !s->gh || s->ntw % s->gw || s->nth % s->gh || lay[0] > 16) {
        if (plat_inc32(&nwarn) <= 8)
            rt_log("vk: sparse: NVN tile %ux%u is not whole Vulkan blocks of %ux%u (VkFormat %d)", s->ntw, s->nth, s->gw,
                   s->gh, (int)ci.format);
        free(s);
        return 0;
    }
    if (vkCreateImage(c->device, &ci, NULL, &s->img) != VK_SUCCESS) {
        if (plat_inc32(&nwarn) <= 8) rt_log("vk: sparse: vkCreateImage failed (VkFormat %d %ux%u)", (int)ci.format, s->w, s->h);
        free(s);
        return 0;
    }
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(c->device, s->img, &mr);
    uint32_t nr = 1;
    VkSparseImageMemoryRequirements sr;
    vkGetImageSparseMemoryRequirements(c->device, s->img, &nr, &sr);
    s->slot_size = mr.alignment ? mr.alignment : SPARSE_PAGE;
    s->type_bits = mr.memoryTypeBits;
    s->nlev = (uint32_t)lay[0] < ci.mipLevels ? (uint32_t)lay[0] : ci.mipLevels;
    if (nr && sr.imageMipTailFirstLod < s->nlev) s->nlev = sr.imageMipTailFirstLod;
    s->nnvn = (uint32_t)lay[0];
    for (uint32_t l = 0, page = 0, blk = 0; l < s->nnvn; l++) {
        uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1;
        s->tiles_x[l] = (lw + s->ntw - 1) / s->ntw, s->tiles_y[l] = (lh + s->nth - 1) / s->nth, s->tile0[l] = page;
        page += s->tiles_x[l] * s->tiles_y[l];
        s->stride = page;
        if (l < s->nlev) {
            s->bx[l] = (lw + s->gw - 1) / s->gw, s->blk0[l] = blk;
            blk += s->bx[l] * ((lh + s->gh - 1) / s->gh);
            s->nblk = blk;
        }
    }
    size_t ntotal = (size_t)s->layers * s->nblk;
    if (ntotal && (!(s->bound = calloc(ntotal, sizeof *s->bound)) || !(s->dirty = calloc(ntotal, 1))))
        rt_fatal("vk: out of memory (sparse table)");
    tail_alloc(c, s, nr ? &sr : NULL, ci.mipLevels, mr.memoryTypeBits);
    tail_bind_now(c, s);
    plat_mutex_lock(&sp_lock);
    s->next = images, images = s;
    if (s->tail_pending) ntail_pending++;
    n_images++;
    // Coarse levels first, so a full budget leaves fine detail out rather than whole layers.
    for (int l = (int)s->nlev - 1; l >= 0; l--)
        for (uint32_t y = 0; y < s->layers; y++) {
            uint64_t lo = s->addr + ((uint64_t)y * s->stride + s->tile0[l]) * SPARSE_PAGE;
            bind_range(c, s, lo, lo + (uint64_t)s->tiles_x[l] * s->tiles_y[l] * SPARSE_PAGE);
        }
    int64_t ni = n_images, ns = n_slots;
    plat_mutex_unlock(&sp_lock);
    if (ni <= 16 || !(ni & (ni - 1)))
        rt_log("vk: sparse image %lld: %#llx %ux%u layers %u VkFormat %d, NVN tile %ux%u (%u levels, %u pages a layer), "
               "Vulkan block %ux%u of %llu bytes, mip tail from level %u; %lld slots live", (long long)ni,
               (unsigned long long)addr, s->w, s->h, s->layers, (int)ci.format, s->ntw, s->nth, (unsigned)lay[0],
               s->stride, s->gw, s->gh, (unsigned long long)s->slot_size, nr ? sr.imageMipTailFirstLod : 0, (long long)ns);
    *out = s->img;
    return 1;
}

uint64_t vk_sparse_destroy(VkImage img) {
    uint64_t bind_fence = 0;
    plat_mutex_lock(&sp_lock);
    SparseImg **pp = &images;
    while (*pp && (*pp)->img != img) pp = &(*pp)->next;
    SparseImg *s = *pp;
    if (s) {
        *pp = s->next;
        bind_fence = s->bind_fence;
        for (size_t i = 0; i < (size_t)s->layers * s->nblk; i++)
            if (s->bound[i]) slot_unref(s->bound[i]);
        if (s->ndl) ndirty--;
        if (s->tail_pending) ntail_pending--;
        if (s->tail_alloc) {
            if (ndying_tails == cdying_tails) grow((void **)&dying_tails, &cdying_tails, sizeof *dying_tails);
            dying_tails[ndying_tails++] = s->tail_alloc;
        }
        free(s->bound);
        free(s->dirty);
        free(s->dl);
        free(s);
        n_images--;
    }
    plat_mutex_unlock(&sp_lock);
    return bind_fence;
}

int vk_sparse_is(VkImage img) {
    plat_mutex_lock(&sp_lock);
    const SparseImg *s = images;
    while (s && s->img != img) s = s->next;
    plat_mutex_unlock(&sp_lock);
    return s != NULL;
}

void vk_sparse_remap(uint64_t lo, uint64_t size) {
    if (!size) return;
    plat_mutex_lock(&sp_lock);
    VkCtx *c = vk_ctx();
    if (c) {
        for (SparseImg *s = images; s; s = s->next)
            if (lo < s->addr + s->size && s->addr < lo + size) bind_range(c, s, lo, lo + size);
    }
    plat_mutex_unlock(&sp_lock);
}

void vk_sparse_written(VkImage img, uint32_t level, uint32_t layer0, uint32_t layers, int32_t x, int32_t y, uint32_t w,
                       uint32_t h) {
    plat_mutex_lock(&sp_lock);
    SparseImg *s = images;
    while (s && s->img != img) s = s->next;
    if (s && level >= s->nlev && level < 32 && !(s->tail_written >> level & 1)) {
        s->tail_written |= 1u << level, n_tail_writes++;
        rt_log("vk: sparse: first write to level %u of %ux%u layers %u (outside the tiles; %s)", level, s->w, s->h,
               s->layers, s->tail_alloc ? "has memory" : "no memory: lost");
    }
    if (s && level < s->nlev && s->bound) {
        uint32_t lw = s->w >> level ? s->w >> level : 1, lh = s->h >> level ? s->h >> level : 1;
        if (!w) x = y = 0, w = lw, h = lh;
        uint32_t bx0 = (uint32_t)x / s->gw, by0 = (uint32_t)y / s->gh;
        uint32_t bx1 = ((uint32_t)x + w + s->gw - 1) / s->gw, by1 = ((uint32_t)y + h + s->gh - 1) / s->gh;
        uint32_t rows = (lh + s->gh - 1) / s->gh;
        if (bx1 > s->bx[level]) bx1 = s->bx[level];
        if (by1 > rows) by1 = rows;
        for (uint32_t l = layer0; l < layer0 + layers && l < s->layers; l++)
            for (uint32_t by = by0; by < by1; by++)
                for (uint32_t bx = bx0; bx < bx1; bx++) {
                    Slot *sl = s->bound[(size_t)l * s->nblk + s->blk0[level] + by * s->bx[level] + bx];
                    if (sl && !sl->written) sl->written = 1, n_written++;
                }
    }
    plat_mutex_unlock(&sp_lock);
}

VkSparseOrder vk_sparse_order(uint64_t fence, int binds) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_sparse) return (VkSparseOrder){0};
    int on_ = order_on();
    if (!on_ && binds != 1) return (VkSparseOrder){0};
    if (on_ && !order_tl) {
        VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
        if (vkCreateSemaphore(c->device, &si, NULL, &order_tl) != VK_SUCCESS) rt_fatal("vk: sparse order semaphore");
    }
    if (!binds) return (VkSparseOrder){order_tl, order_tl, bind_val, ++order_val};
    plat_mutex_lock(&sp_lock);
    if (fence) {
        for (int i = 0; i < ndying; i++) vk_completion_add(fence, slot_reap, dying[i]);
        ndying = 0;
        for (int i = 0; i < ndying_tails; i++) vk_completion_add(fence, tail_free, dying_tails[i]);
        ndying_tails = 0;
    }
    if (ndirty || ntail_pending) {
        int ni = 0, nb = 0, total = 0, no = 0, nm = 0, mtotal = 0;
        for (SparseImg *s = images; s; s = s->next) {
            total += s->ndl;
            if (s->tail_pending) total += (int)(s->layers * s->tail_blocks), mtotal += (int)tail_count(s);
        }
        VkSparseImageMemoryBindInfo *ib = malloc(sizeof *ib * (size_t)(ndirty + ntail_pending));
        VkSparseImageMemoryBind *bb = malloc(sizeof *bb * (size_t)(total ? total : 1));
        VkSparseImageOpaqueMemoryBindInfo *ob = malloc(sizeof *ob * (size_t)(ntail_pending ? ntail_pending : 1));
        VkSparseMemoryBind *mb = malloc(sizeof *mb * (size_t)(mtotal ? mtotal : 1));
        if (!ib || !bb || !ob || !mb) rt_fatal("vk: out of memory (sparse binds)");
        for (SparseImg *s = images; s; s = s->next) {
            if (!s->tail_pending) continue;
            tail_binds(s, &ib[ni], bb + nb, &ob[no], mb + nm);
            s->bind_fence = fence ? fence : vk_sync_issued() + 2;
            nb += (int)ib[ni].bindCount, nm += (int)ob[no].bindCount;
            if (ib[ni].bindCount) ni++;
            if (ob[no].bindCount) no++;
            s->tail_pending = 0;
        }
        ntail_pending = 0;
        for (SparseImg *s = images; s; s = s->next) {
            if (!s->ndl) continue;
            ib[ni++] = (VkSparseImageMemoryBindInfo){.image = s->img, .bindCount = (uint32_t)s->ndl, .pBinds = bb + nb};
            s->bind_fence = fence ? fence : vk_sync_issued() + 2;
            for (int k = 0; k < s->ndl; k++) bb[nb++] = make_bind(s, s->dl[k]), s->dirty[s->dl[k]] = 0;
            s->ndl = 0;
        }
        ndirty = 0;
        uint64_t wait = on_ ? order_val : (fence > 1 ? fence - 1 : 0), sig = on_ ? ++order_val : 0;
        if (!on_ && !bind_sem) {
            VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            if (vkCreateSemaphore(c->device, &si, NULL, &bind_sem) != VK_SUCCESS) rt_fatal("vk: sparse bind semaphore");
        }
        VkTimelineSemaphoreSubmitInfo tl = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                            .waitSemaphoreValueCount = wait ? 1u : 0u, .pWaitSemaphoreValues = &wait,
                                            .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &sig};
        VkBindSparseInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .pNext = &tl,
                               .waitSemaphoreCount = wait ? 1u : 0u, .pWaitSemaphores = on_ ? &order_tl : &c->timeline,
                               .imageOpaqueBindCount = (uint32_t)no, .pImageOpaqueBinds = ob,
                               .imageBindCount = (uint32_t)ni, .pImageBinds = ib, .signalSemaphoreCount = 1,
                               .pSignalSemaphores = on_ ? &order_tl : &bind_sem};
        if (on_) bind_val = sig;
        else binary_pending = 1;
        uint64_t t0 = plat_now_ns();
        VkResult r = vkQueueBindSparse(c->gfx_queue, 1, &bi, VK_NULL_HANDLE);
        uint64_t dt = plat_now_ns() - t0;
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: vkQueueBindSparse of %d binds failed (%d)", nb, (int)r);
        }
        n_binds += nb;
        static int64_t ncalls;
        static uint64_t bind_ns;
        int64_t calls = ++ncalls;
        bind_ns += dt;
        if (calls <= 4 || !(calls & (calls - 1)))
            rt_log("vk: sparse bind %lld: %d blocks in %d images (%lld blocks so far, %lld slots live, %lld MiB, "
                   "%lld written), %.1f ms in the call (%.1f so far)", (long long)calls, nb, ni, (long long)n_binds,
                   (long long)n_slots, (long long)(n_slots * 65536 >> 20), (long long)n_written, dt / 1e6,
                   bind_ns / 1e6);
        free(ib);
        free(bb);
        free(ob);
        free(mb);
    }
    if (binds == 2) {
        plat_mutex_unlock(&sp_lock);
        return (VkSparseOrder){0};
    }
    VkSparseOrder o = {order_tl, order_tl, bind_val, ++order_val};
    if (!on_) o = (VkSparseOrder){binary_pending ? bind_sem : VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0}, binary_pending = 0;
    plat_mutex_unlock(&sp_lock);
    return o;
}

void vk_sparse_shutdown(void) {
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&sp_lock);
    while (images) {
        SparseImg *s = images;
        images = s->next;
        if (s->tail_alloc && c && c->vma) vmaFreeMemory(c->vma, s->tail_alloc);
        free(s->bound);
        free(s->dirty);
        free(s->dl);
        free(s);
    }
    for (int i = 0; i < ndying_tails; i++)
        if (dying_tails[i] && c && c->vma) vmaFreeMemory(c->vma, dying_tails[i]);
    free(dying_tails), dying_tails = NULL, ndying_tails = cdying_tails = 0;
    free(dying), dying = NULL, ndying = cdying = 0;
    if (slots.vals) {
        for (uint32_t i = 0; i < slots.cap; i++) free(slots.vals[i]);
    }
    free(slots.keys), free(slots.vals);
    slots = (nvn_hmap){.klen = sizeof(SlotKey)};
    while (blocks) {
        SlotBlock *b = blocks;
        blocks = b->next;
        if (c && c->device && b->buf) vkDestroyBuffer(c->device, b->buf, NULL);
        if (c && c->vma && b->alloc) vmaFreeMemory(c->vma, b->alloc);
        free(b);
    }
    if (c && c->device) {
        if (order_tl) vkDestroySemaphore(c->device, order_tl, NULL);
        if (bind_sem) vkDestroySemaphore(c->device, bind_sem, NULL);
    }
    order_tl = bind_sem = VK_NULL_HANDLE;
    order_val = bind_val = 0;
    binary_pending = ndirty = ntail_pending = 0;
    n_slots = n_failed = n_binds = n_images = n_blocks = n_over = n_written = n_tail_writes = n_host_blocks = 0;
    plat_mutex_unlock(&sp_lock);
}

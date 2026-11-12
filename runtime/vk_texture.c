// Vulkan textures of the graphics layer: a VkImage cache keyed by storage
// GPU address plus descriptor, uploads through staging buffers with block-linear deswizzle and ASTC decode (vk_astc.c),
// re-uploads when page write generations (vk_memory.c) show the storage written, ReadTexels / WriteTexels, the texture
// and sampler descriptor slots, and VkSamplers.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn @ b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile): which textures mirror their pool memory (nvn_mtl.m tex_mirrored, 3347-3362), the
// "GPU-owned" rule that stops re-uploads once the GPU rendered into a mirrored texture (nvn_mtl.m tex_mtl, 3366-3372),
// the live texture at a storage address that a slot of a finalized texture follows (hle_nvn.c:373-393), and
// ReadTexels / WriteTexels (nvn_mtl.m:10105-10195). Image creation, upload regions and samplers are ported from LAS
// Recomp runtime/nvn/vulkan/textures.c:235-330, 778-844 @ da35b83a (MIT, Copyright (c) 2026 Peiikai).
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL. Uploads are recorded into the upload batch of vk_memory.c; the replay
// submits it (vk_upload_flush) before a submission that samples the textures.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

typedef struct {
    uint64_t addr;
    nvn_texdesc d;
    uint8_t has_layout, layout[8], pad[7];
} TexKey;

typedef struct Tex {
    VkTexImage pub;               // first: vk_texture_get hands out &pub
    TexKey key;
    tex_t t;                      // the newest texture object seen with this key
    TexVkFormat f;
    VmaAllocation alloc;          // own memory; NULL when bound into pool memory
    VmaAllocation bound_to;       // the pool memory it is bound into
    uint32_t wgen;                // page generation of the last upload check
    uint32_t checked;             // submit_epoch of the last check from a draw binding
    int uploaded, released;
    uint32_t n_up;                // uploads of this entry (vk_texture_debug)
    int last_ch;                  // last vk_mem_changed result of make_fresh: 1 written, 0 clean, -1 not tracked
    uint8_t hashed;               // chash holds the storage hash taken just before the last upload (untracked pages)
    uint64_t chash;
    struct Tex *next_at;          // other entries at the same storage address
    uint8_t gpu_dirty;            // mirrored, and the GPU wrote the image after the last write-back to the storage
    uint32_t gpu_gen;             // page generation when the GPU write that set gpu_dirty was recorded
    uint64_t gpu_seq;             // order of that GPU write among all of them (gpu_writes)
    uint64_t gpu_fence;           // fence of the submission that made that write (0: not recorded in one)
    uint64_t wr_fence;            // any entry: newest submission recorded to write its image (render, copy, clear,
                                  // storage binding); ReadTexels waits for it instead of draining the replay queue
    uint8_t listed;               // in owned[]
    uint8_t sparse;               // Vulkan sparse image backed by tile slots (vk_sparse.c)
    uint8_t *snap;                // a texture that fills no page: its storage bytes when the GPU write was recorded
    uint32_t wt_n;                // WriteTexels into the storage since the last upload
    struct Tex *rel_prev, *rel_next;   // while released: the list of released entries, oldest first
} Tex;

// A slot keeps what vk_texture_by_id resolved for it, so a draw binding skips the key lookup and the address chain: the
// entry (ce), valid while the generation of its address (agen) is still cgen, the view and has_view handed out with it,
// and the submit_epoch of the last freshness check through the slot (cepoch). Hot fields first: a lookup that hits
// touches one or two cache lines.
typedef struct {
    struct Tex *ce;               // NULL: nothing cached (registered since, or never resolved)
    uint32_t cgen;                // agen of the slot's address when ce was resolved
    uint32_t cepoch;
    view_t cview;
    uint8_t chas_view, used, has_view;
    view_t v;
    tex_t t;
} TexSlot;
typedef struct { samp_par p; VkSampler s, sp; uint8_t used; } SampSlot;   // s: vk_sampler_for(&p) once looked up; sp:
                                                                           // its point-sampled form (vk_sampler_point_by_id_locked)
enum { DESC_TEXTURE = 1, DESC_SAMPLER };
typedef struct {
    uint32_t kind, serial;
    union { TexSlot tex; SampSlot sampler; } u;
} DescSlot;

static PlatRMutex tex_lock;
static PlatOnce tex_once = PLAT_ONCE_INIT;
static nvn_hmap by_key = {.klen = sizeof(TexKey)};
static nvn_hmap by_addr = {.klen = 8};
static nvn_hmap descriptors = {.klen = sizeof(VkDescKey)};
static nvn_hmap samplers = {.klen = sizeof(samp_par)};
static int nreleased;
static Tex *rel_head, *rel_tail;   // released entries in release order (sweep_released drops from the head)
static uint64_t n_uploads, n_reuploads;
// Re-uploads since the last perf line (tex_lock): all, and those of textures a WriteTexels wrote since their last upload.
static uint64_t ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns;
static uint64_t n_hash_kept;   // untracked checks whose storage hash matched the last upload's (untracked_same)
// TOTK_TEX_ALIAS (default on; =0: a new image over live storage starts empty): a new descriptor over the storage of a
// live image that the GPU wrote gets that image's content (alias_copy). Counts since the last log line.
static int tex_alias = -1;
static uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
static uint32_t submit_epoch = 1;   // bumped per queue submit: draw bindings check a texture's storage once per submit
// Mirrored textures the GPU wrote into, sorted by address (vk_texture_cpu_stale, vk_texture_write_back). Changed under
// tex_lock and owned_lock; read under either.
static PlatMutex owned_lock = PLAT_MUTEX_INIT;
static Tex **owned;
static int nowned, cowned;
static uint64_t max_owned;   // largest storage size in owned[]
static uint64_t gpu_writes;  // GPU writes into mirrored textures so far: Tex.gpu_seq
// Bumped whenever the entry a slot resolves to can change: an entry created or destroyed, released or taken back into
// use (the slot lookup and the follow of a finalized slot read only by_key, by_addr and the released flags).
static uint32_t tex_gen = 1;
// Bumped with tex_gen and whenever a slot is registered: what vk_texture_by_id returns for any slot may have changed.
static uint32_t slots_gen = 1;
static uint32_t samp_gen = 1;   // bumped whenever a sampler slot is registered: vk_sampler_by_id may answer otherwise
static uint64_t n_checks;    // draw-binding storage checks (make_fresh_snap) since the last perf line
// tex_gen per storage address (hashed): a slot resolves through the entries at its own address only (slot_resolve), so
// a slot's resolution stays valid while its address's generation does. TotK finalizes and initializes transient
// textures every frame: about 90 tex_gen and 160 slots_gen bumps per frame in the monsters fight, one every 16 draws,
// which emptied every cache keyed on the global generations.
#define AGEN_N 4096
static uint32_t agen[AGEN_N];
static uint32_t reg_serial;

static uint32_t agen_idx(uint64_t addr) { return (uint32_t)(addr * 0x9E3779B97F4A7C15ull >> 52); }

static void tex_gen_bump(uint64_t addr) {
    if (!++tex_gen) tex_gen = 1;
    slots_gen++;
    uint32_t *g = &agen[agen_idx(addr)];
    __atomic_store_n(g, *g + 1, __ATOMIC_RELEASE);
}

static volatile int32_t tex_ready;
static void tex_init(void *p) {
    (void)p;
    __atomic_store_n(&tex_ready, 1, __ATOMIC_RELEASE);
    return;
}
static void lock(void) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    plat_rmutex_lock(&tex_lock);
}
static void unlock(void) { plat_rmutex_unlock(&tex_lock); }

static uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t mip(uint32_t v, unsigned lv) { return umax(v >> lv, 1); }

// Host bytes for a mip chain (bpt ≈ 4 for RGBA8, 1 for BC3). Used to trim huge Switch 2 ASTC arrays.
static uint64_t est_tex_bytes(uint32_t w, uint32_t h, uint32_t depth, uint32_t levels, uint32_t layers, uint32_t bpt) {
    uint64_t sum = 0;
    for (uint32_t lv = 0; lv < levels; lv++) {
        uint32_t mw = mip(w, lv), mh = mip(h, lv), md = depth > 1 ? mip(depth, lv) : 1;
        sum += (uint64_t)mw * mh * md * layers * bpt;
    }
    return sum;
}

// Eglantine CompactLayers: drop top guest mips until the host image fits. Default 256 MiB keeps MKW's
// 8192×68 arrays at 1024 (~91 MiB) — stable with the ~2.3 GiB buffer mirror. Raise with
// TOTK_COMPACT_ASTC_MB=384 (edge still ≤2048) when VRAM allows; room is also clamped to the live
// device-local budget minus a mirror reserve so a high cap cannot device-lose mid-boot.
enum { COMPACT_ASTC_BUDGET_DEFAULT_MB = 512u };
enum { COMPACT_ASTC_MAX_EDGE = 2048u };
enum { COMPACT_ASTC_MIRROR_RESERVE = 3ull << 30 };   // leave room for MKW's large cached pool mirror

static uint64_t compact_astc_budget_cap(void) {
    static int64_t bytes = -1;
    if (bytes < 0) {
        const char *s = cfg_get("TOTK_COMPACT_ASTC_MB");
        int mb = s && *s ? atoi(s) : (int)COMPACT_ASTC_BUDGET_DEFAULT_MB;
        if (mb < 64) mb = 64;
        if (mb > 2048) mb = 2048;
        bytes = (int64_t)mb << 20;
        rt_log("vk: CompactLayers ASTC budget %d MiB (edge ≤ %u)%s", mb, COMPACT_ASTC_MAX_EDGE,
               s && *s ? "" : ", MKW_COMPACT_ASTC_MB= to raise when VRAM allows");
    }
    return (uint64_t)bytes;
}

// Per-texture: never take more than the configured cap, and never more than free VRAM past the mirror reserve.
static uint64_t compact_astc_budget(uint64_t want) {
    uint64_t cap = compact_astc_budget_cap();
    if (want && want < cap) cap = want;
    uint64_t used = 0, budget = 0;
    int exact = 0;
    if (vk_mem_vram(&used, &budget, &exact) && budget > COMPACT_ASTC_MIRROR_RESERVE) {
        uint64_t room = budget - COMPACT_ASTC_MIRROR_RESERVE;
        if (used < room) {
            uint64_t free = room - used;
            if (cap > free) cap = free;
        } else {
            // Past reserve: still allow one huge RGBA8 CompactLayers host (~363 MiB at 1024×68).
            // The old 64 MiB floor crushed 8192×68 arrays to 256² (soft / missing ground).
            cap = 400ull << 20;
        }
    }
    if (cap < (64ull << 20)) cap = 64ull << 20;
    return cap;
}

// Huge MKW terrain arrays: one locked host edge. Must be a power-of-two guest mip size (8192→1024 /
// 512 / …): a non-POT lock like 768 made over_edge keep skipping to 512 (n09/n10 soft ground).
// 1024 RGBA8 is ~363 MiB each; keep the live-budget trim from dropping huge arrays further (below).
// 2048 RGBA8×68 ≈ 1.45 GiB blew VRAM (n78: mirrors refused at 13/14.6 GiB, title lost all
// background / foliage into a dark void). 1024 (~363 MiB) keeps room for mesh SSBO mirrors.
enum { COMPACT_ASTC_HUGE_EDGE = 1024u };
// MKW_COMPACT_HUGE_EDGE=1024|2048|4096 overrides. Unset: 2048 when the largest device-local heap is 12 GiB or more
// (import memory mode keeps the pools in host memory, so the n78 mirror pressure above no longer applies), else 1024.
// adriadam10/mkw-recomp keeps every terrain level (4 GiB budget, sparse tiles); 1024 is visibly blurry ground.
static uint32_t huge_edge_default(void) {
    const char *s = cfg_get("TOTK_COMPACT_HUGE_EDGE");
    if (s && *s) {
        uint32_t v = (uint32_t)atoi(s);
        if (v == 512 || v == 1024 || v == 2048 || v == 4096) return v;
    }
    return COMPACT_ASTC_HUGE_EDGE;
}
static uint32_t huge_astc_host_edge(uint32_t guest_w, uint32_t guest_h, uint32_t guest_levels, uint32_t layers,
                                    uint32_t bpt) {
    static uint32_t locked;
    static int logged;
    (void)guest_w; (void)guest_h; (void)guest_levels; (void)layers; (void)bpt;
    if (locked) return locked;
    locked = huge_edge_default();
    if (!logged) {
        logged = 1;
        rt_log("vk: CompactLayers huge ASTC arrays locked to host edge %u (same for all)", locked);
    }
    return locked;
}

// Allocation-hint and usage flags that do not change guest byte layout on unified memory; create_tex already enables
// storage usage on single-sample colour formats, so descriptors differing only in these flags share one VkImage.
#define KEY_FREE_FLAGS \
    (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL | \
     NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL)

static void make_key(const tex_t *t, TexKey *k) {
    memset(k, 0, sizeof *k);
    k->addr = (uintptr_t)t->mem;
    k->d = t->d;
    k->d.flags &= ~KEY_FREE_FLAGS;
    k->has_layout = t->has_layout;
    memcpy(k->layout, t->layout, 8);
}

// Keys equal apart from the descriptor's swizzle selectors: the same storage, format and geometry.
static int same_but_swizzle(const TexKey *a, const TexKey *b) {
    TexKey x = *a, y = *b;
    x.d.swizzle_r = x.d.swizzle_g = x.d.swizzle_b = x.d.swizzle_a = 0;
    y.d.swizzle_r = y.d.swizzle_g = y.d.swizzle_b = y.d.swizzle_a = 0;
    return !memcmp(&x, &y, sizeof x);
}

// MNG nvn_mtl.m tex_mirrored: textures whose content the CPU provides in pool memory. Display, compressible, zcull,
// depth, multisample and GPU-only storage stays GPU-owned.
static int tex_mirrored(const tex_t *t) {
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem)) return 0;
    if (t->has_layout) return 1;
    const pool_t *p = t->pool;
    if (!p || !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL |
                      NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL))
        return 0;
    if (t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;   // stencil / depth formats
    return 1;
}

static void log_limited(volatile int32_t *n, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void log_limited(volatile int32_t *n, const char *fmt, ...) {
    if (plat_inc32(n) > 16) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    rt_log("vk: %s", buf);
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT

// ---------------------------------------------------------------- images
static VkFormatFeatureFlags features_of(VkFormat f) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &p);
    return p.optimalTilingFeatures;
}

static VkImageAspectFlags aspect_of(const TexVkFormat *f) {
    VkImageAspectFlags a = 0;
    if (f->depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f->stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Binds img into the pool memory behind its GPU address (physical pools, virtual pools through the page table) when
// the memory type, alignment and contiguous size allow; 0 otherwise.
// A depth / stencil image that needs more bytes than its NVN storage (guest_size) is not bound: AMD's RDNA2 driver
// (26.10.1) puts the depth compression metadata after the depth data, and above 1080p that tail lies over the next
// texture of the game's heap; each depth write then scrambled it (garbage band at the top, pink glow, blocks in the
// grass, a blue checkered scene in the menu). Colour images keep their place: the RX 9070 also sizes some of them
// above their storage, and nothing there showed that the GPU writes the excess.
static int format_compressed(VkFormat f) {   // vkCmdClearColorImage takes no block-compressed format
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK && f <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK);
}

static int bind_pool_memory(VkCtx *c, Tex *e, VkImage img, uint64_t guest_size) {
    // Decoded ASTC, RGB-expanded, and format-converted images never match guest byte layout in pool memory, and any
    // image whose host memory requirement exceeds guest_size would overwrite adjacent pool allocations when written.
    if (e->f.astc_bw || e->f.expand_rgb || e->f.vk != e->f.ideal) return 0;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, img, &req);
    if (!req.size || req.size > guest_size) return 0;
    VkMemSpan sp;
    if (!vk_mem_resolve(e->pub.addr, req.size, &sp) || !sp.alloc || sp.alloc_offset % req.alignment) return 0;
    VmaAllocationInfo info;
    vmaGetAllocationInfo(c->vma, sp.alloc, &info);
    if (!(req.memoryTypeBits >> info.memoryType & 1)) return 0;
    if (vmaBindImageMemory2(c->vma, sp.alloc, sp.alloc_offset, img, NULL) != VK_SUCCESS) return 0;
    e->bound_to = sp.alloc;
    return 1;
}

// agl's BC4 encoder draws a BC texture's blocks through an RGBA16UI view, one texel per 8-byte block (refs/nxnative
// docs/agents/04-graphics.md, Breath of the Wild and Tears of the Kingdom sections). Vulkan does it with an uncompressed
// view of a BLOCK_TEXEL_VIEW_COMPATIBLE image; EXTENDED_USAGE lets the BC image carry the COLOR_ATTACHMENT usage that
// only such views support. Whether the device takes that combination for this format and shape:
static int block_views_ok(const VkImageCreateInfo *ci) {
    VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                                           .format = ci->format, .type = ci->imageType, .tiling = ci->tiling,
                                           .usage = ci->usage, .flags = ci->flags};
    VkImageFormatProperties2 p = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    if (vkGetPhysicalDeviceImageFormatProperties2(vk_ctx()->phys, &fi, &p) != VK_SUCCESS) return 0;
    const VkImageFormatProperties *q = &p.imageFormatProperties;
    return ci->extent.width <= q->maxExtent.width && ci->extent.height <= q->maxExtent.height &&
           ci->extent.depth <= q->maxExtent.depth && ci->mipLevels <= q->maxMipLevels &&
           ci->arrayLayers <= q->maxArrayLayers;
}

// The memory the device wants for an image made from ci: 0 when it cannot lay it out. RADV on GFX9 (Vega, Renoir)
// hands back a 0-byte image for a mipmapped block-compressed 3D image at least as deep as it is wide in blocks, with
// COLOR_ATTACHMENT usage and no STORAGE: that usage asks for a thin 3D layout (Mesa radv_image.c), for which the GFX9
// address library has no swizzle mode left (gfx9addrlib.cpp), and the error is dropped; VMA then refuses the 0-byte
// allocation. The format properties call it supported, so only the size shows it.
static VkDeviceSize image_size(const VkImageCreateInfo *ci) {
    VkDeviceImageMemoryRequirements q = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = ci};
    VkMemoryRequirements2 r = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(vk_ctx()->device, &q, &r);
    return r.memoryRequirements.size;
}

// UNDEFINED -> GENERAL; new depth / stencil images start cleared (LAS textures.c: the game copies depth
// without stencil, and the stencil of those images must not be whatever the memory held).
static void init_image(Tex *e, VkCommandBuffer cb) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image, .subresourceRange = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    if (e->sparse) {
        // Sparse tiles get zeroed slot memory when bound; clearing the image would wipe shared physical pages.
    } else if (e->f.depth || e->f.stencil) {
        VkClearDepthStencilValue zero = {0.0f, 0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearDepthStencilImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else if (!e->bound_to && !format_compressed(e->f.vk) &&
               ((vk_fix_zero() && !e->pub.mirrored) || e->pub.host_mip0)) {
        // GPU-only own-memory starts at zero (pool alloc_physical). Mirrored images usually upload from
        // CPU storage on first use — except CompactLayers (host_mip0 > 0): those stay mirrored for
        // MapVirtual tile refresh but the GPU image is a trimmed RGBA8 fold whose layers are filled
        // only as ASTC decodes arrive. Leaving them UNDEFINED showed as rainbow cliffs (n173–n215).
        VkClearColorValue zero = {0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
}

// defer_init: the caller records the image's first transition (init_image or alias_copy).
static int evict_vram_oom(int include_gpu_owned);

static Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init) {
    static volatile int32_t n_unmapped, n_unsupported, n_failed, n_unbound, n_thick, n_oom_evict;
    VkCtx *c = vk_ctx();
    const nvn_texdesc *d = &t->d;
    if (!c || d->target == NVN_TEXTURE_TARGET_BUFFER) return NULL;
    TexVkFormat f;
    if (!tex_vk_format(d->format, &f)) {
        log_limited(&n_unmapped, "texture format %#x has no Vulkan format: %s", d->format, f.reason);
        return NULL;
    }
    int sparse_cand = (d->flags & NVN_TEXTURE_FLAGS_SPARSE) && nvn_gpu_only(k->addr);
    // Keep small ASTC textures (< 256x256) as lossless RGBA8: they are mostly LUTs / noise maps that lossy BC7
    // degrades visibly, and they take negligible VRAM (adriadam10/mkw-recomp, MIT).
    if (f.astc_bc && !sparse_cand && (uint64_t)d->width * d->height < 256u * 256u) {
        f.astc_bc = 0;
        f.vk = f.astc_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    }
    if (!tex_vk_format_supported(&f)) {
        log_limited(&n_unsupported, "texture format %#x: VkFormat %d lacks features %#x", d->format, (int)f.vk, (unsigned)f.need);
        return NULL;
    }
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    uint32_t w = umax(d->width, 1), h = umax(d->height, 1), depth = 1, layers = 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: type = VK_IMAGE_TYPE_1D; h = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: type = VK_IMAGE_TYPE_1D; h = 1; layers = L.layers; break;
    case NVN_TEXTURE_TARGET_3D: type = VK_IMAGE_TYPE_3D; depth = umax(d->depth, 1); break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    case NVN_TEXTURE_TARGET_CUBEMAP: case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = L.layers; break;
    default: break;
    }
    uint32_t levels = umax(d->levels, 1), most = 1;
    for (uint32_t s = umax(umax(w, h), depth); s > 1; s >>= 1) most++;
    if (levels > most) levels = most;
    uint32_t samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    if (samples > 1) levels = 1;
    VkFormatFeatureFlags ff = features_of(f.vk);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    int ds = f.depth || f.stencil;
    if (ds && (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!ds && (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Storage usage for every single-sample colour texture whose format allows it, not only those created with
    // NVN_TEXTURE_FLAGS_IMAGE: NVN 2 binds textures made without the flag as images (the NVN 2 reference title's upscaler writes its
    // 2880x1620 r11g11b10f output, flags 0), and an image without the usage got a stand-in, so the frame stayed the
    // clear colour. Vulkan cannot add the usage later.
    int mutable_storage = 0;
    if (!ds && samples == 1) {
        if (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        else if (f.vk == VK_FORMAT_R8G8B8A8_SRGB &&
                 (features_of(VK_FORMAT_R8G8B8A8_UNORM) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
            usage |= VK_IMAGE_USAGE_STORAGE_BIT, mutable_storage = 1;
        else if (f.vk == VK_FORMAT_B8G8R8A8_SRGB &&
                 (features_of(VK_FORMAT_B8G8R8A8_UNORM) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
            usage |= VK_IMAGE_USAGE_STORAGE_BIT, mutable_storage = 1;
    }
    VkImageCreateFlags flags = ds ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // format-reinterpreting views
    if (mutable_storage) flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    // Cube compatible only for cube targets: the replay makes cube views of nothing else (vk_exec_draw.c get_view checks
    // the image's own view type). The flag on every square array of 6+ layers changed how AMD's Windows driver lays such
    // an image out, and terrain material arrays sampled through it came out in rainbow blocks that moved with the mip
    // level (Savage Wind Recomp issue #59, BotW on an RX 580 and a 7800 XT).
    if (type == VK_IMAGE_TYPE_2D && layers >= 6 && w == h && samples == 1 &&
        (d->target == NVN_TEXTURE_TARGET_CUBEMAP || d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY))
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // NVN renders into a 3D texture slice by slice (a view's min layer is the depth plane, nvn_mtl.m), e.g. the
    // grading LUT of the menu background blur. Vulkan needs a 2D view of the slice for that.
    if (type == VK_IMAGE_TYPE_3D && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = type, .format = f.vk,
                            .extent = {w, h, depth}, .mipLevels = levels, .arrayLayers = layers,
                            .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    int bw, bh, block_views = 0;
    // TOTK_BLOCK_VIEWS_3D (default 0): a 3D BC image also gets block views. On RDNA2 cards (driver 26.10.1, issue #5)
    // the 32x32x32 BC4 volume the game creates at the logo, made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE,
    // took a GPU write just past the end of its memory: device lost at the first submit after it when the image ended
    // its memory block. Draws into a 3D BC image go through a stand-in instead (vk_exec.c bc_target).
    static int bv3d = -1;
    if (bv3d < 0) {
        bv3d = rt_knob("BLOCK_VIEWS_3D", "0");
        rt_log("vk: 3D BC textures %s (TOTK_BLOCK_VIEWS_3D=%d)", bv3d ? "get block views (renderable)" : "have no block views",
               bv3d);
    }
    if (!ds && samples == 1 && !sparse_cand && f.vk == f.ideal && nvn_fmt_bpb(d->format, &bw, &bh) && bw == 4 && bh == 4 &&
        (type != VK_IMAGE_TYPE_3D || bv3d)) {
        VkImageCreateInfo bc = ci;   // BC formats: renderable through views of their blocks (block_views_ok)
        bc.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        bc.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if (type == VK_IMAGE_TYPE_3D) bc.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        VkImageUsageFlags hidden = 0;   // creation usage the replay never uses (views restrict theirs, vk_exec.c)
        int ok = block_views_ok(&bc);
        if (ok && type == VK_IMAGE_TYPE_3D && !image_size(&bc)) {
            // STORAGE usage keeps RADV on GFX9 from the thin layout (image_size), so agl's BC4 encoder still renders
            // into the volume; else the image is made without block views and its encoder passes are left out
            bc.usage |= VK_IMAGE_USAGE_STORAGE_BIT, hidden = VK_IMAGE_USAGE_STORAGE_BIT;
            ok = block_views_ok(&bc) && image_size(&bc);
            log_limited(&n_thick, "texture fmt %#x %ux%ux%u: the device cannot lay out a 3D image with block views; %s",
                        d->format, w, h, depth, ok ? "made with storage usage (a thick layout)" : "made without them");
        }
        if (ok) ci = bc, usage = bc.usage & ~hidden, flags = bc.flags, block_views = 1;
    }
    // Feedback loops (vk.h VkCtx.feedback): every attachable image may be sampled by a draw of a pass that renders it,
    // so it gets the usage now; adding it when a loop shows up would mean recreating the image (that lost the
    // grass tile maps once).
    int fb_usage = c->feedback == VK_FEEDBACK_LAYOUT && !block_views &&
                   (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
    if (fb_usage) ci.usage = usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    uint32_t host_mip0 = 0, guest_w = w, guest_h = h, guest_levels = levels;
    Tex *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("vk: out of memory for a texture");
    e->key = *k;
    e->t = *t;
    e->f = f;
    e->pub.addr = k->addr;
    VkImage img = VK_NULL_HANDLE;
    // Sparse residency first (vk_sparse.c): keeps full 8192x8192 resolution as BC7/BC6H/D16 without allocating unmapped pages!
    if (sparse_cand && vk_sparse_create(&ci, d, k->addr, L.size, &img)) {
        e->sparse = 1;
    }
    // CompactLayers fallback when sparse residency is disabled or unavailable for this format/shape.
    if ((!vk_sparse_on() || sparse_cand) && !img && f.astc_bw && samples == 1 && type != VK_IMAGE_TYPE_3D) {
        uint32_t bpt_est = f.astc_bc || vk_astc_bc() ? 1u : 4u;
        uint64_t need = est_tex_bytes(w, h, depth, levels, layers, bpt_est);
        uint64_t budget = compact_astc_budget(need);
        int huge = layers >= 16 && umax(w, h) >= 4096;
        if (huge || need > budget) {
            uint32_t edge_cap = huge ? huge_astc_host_edge(guest_w, guest_h, guest_levels, layers, bpt_est)
                                     : COMPACT_ASTC_MAX_EDGE;
            uint32_t bpt = 4u;
            while (levels > 1) {
                int over_edge = umax(w, h) > edge_cap;
                int over_budget = !huge && est_tex_bytes(w, h, depth, levels, layers, bpt) > compact_astc_budget(0);
                if (!over_edge && !over_budget) break;
                if (huge && !over_edge) break;
                host_mip0++;
                w = mip(guest_w, host_mip0);
                h = mip(guest_h, host_mip0);
                levels = guest_levels > host_mip0 ? guest_levels - host_mip0 : 1;
            }
            if (f.astc_bc) {
                f.astc_bc = 0;
                f.vk = f.astc_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                ci.format = f.vk;
            }
            ci.extent.width = w;
            ci.extent.height = h;
            ci.mipLevels = levels;
            if (host_mip0)
                rt_log("vk: CompactLayers ASTC %#x %ux%u x%u -> host %ux%u levels %u (skip %u guest mips, ~%.0f MiB, RGBA8 folds) addr %#llx",
                       d->format, guest_w, guest_h, layers, w, h, levels, host_mip0,
                       est_tex_bytes(w, h, depth, levels, layers, bpt) / 1048576.0, (unsigned long long)k->addr);
        }
        e->f = f;
    }
    // Trimmed ASTC arrays are smaller than the guest pool allocation: always take own memory.
    if (!img && !host_mip0 && nvn_gpu_only(k->addr)) {   // storage in a physical or virtual pool: alias the pool memory (section 3.2)
        if (vkCreateImage(c->device, &ci, NULL, &img) == VK_SUCCESS && !bind_pool_memory(c, e, img, L.size)) {
            vkDestroyImage(c->device, img, NULL);
            img = VK_NULL_HANDLE;
            log_limited(&n_unbound, "texture %#llx (%ux%u fmt %#x) cannot alias its pool memory; own memory instead",
                        (unsigned long long)k->addr, w, h, d->format);
        }
    }
    if (!img) {
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        // Drop released textures first when CompactLayers is about to take a large own-memory image
        // (Ryujinx-style: prefer surviving device-local pressure over a mid-create device-lost).
        if (host_mip0 || (uint64_t)w * h * layers >= (256ull << 10)) {
            uint32_t bpt = f.astc_bw ? (f.astc_bc || vk_astc_bc() ? 1u : 4u) : 4u;
            uint64_t need = est_tex_bytes(w, h, depth > 1 ? depth : 1, levels, layers, bpt);
            vk_texture_free_for_vram(need);
        }
        uint64_t v_used = 0, v_budget = 0;
        int v_exact = 0;
        if (vk_mem_vram(&v_used, &v_budget, &v_exact) && v_budget && v_used > v_budget / 20 * 17) {
            if (rel_head) evict_vram_oom(0);
            int fold_src = layers <= 4 && umax(guest_w, guest_h) >= 4096;
            if (!host_mip0 && !fold_src && f.astc_bw && !f.astc_bc && (uint64_t)w * h >= 256u * 256u &&
                vk_astc_mode() == VK_ASTC_GPU && vk_mem_vram(&v_used, &v_budget, &v_exact) && v_used > v_budget / 20 * 17) {
                f.vk = vk_astc_bc_format(f.astc_srgb);
                f.astc_bc = 1;
                ci.format = f.vk;
                ci.usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                              VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
                if (!block_views) ci.flags &= ~VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
                e->f = f;
                log_limited(&n_oom_evict,
                            "vk: VRAM usage %llu/%llu MiB (>85%%); compressing ASTC %ux%u to BC",
                            (unsigned long long)(v_used >> 20), (unsigned long long)(v_budget >> 20), w, h);
            }
        }
        VkResult r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
        if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
            int n_evict = evict_vram_oom(0);
            r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
            if (r != VK_SUCCESS) {
                n_evict += evict_vram_oom(1);
                r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
            }
            if (r == VK_SUCCESS)
                log_limited(&n_oom_evict, "vk: VRAM OOM for %ux%ux%u fmt %#x; evicted %d released textures and recovered",
                            w, h, depth, d->format, n_evict);
        }
        if (r != VK_SUCCESS && f.astc_bw && !f.astc_bc) {
            f.vk = vk_astc_bc_format(f.astc_srgb);
            f.astc_bc = 1;
            ci.format = f.vk;
            ci.usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                          VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
            if (!block_views) ci.flags &= ~VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
            r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
            if (r == VK_SUCCESS) {
                e->f = f;
                rt_log("vk: mega-texture %ux%ux%u layers %u fell back to BC (%s)", guest_w, guest_h, depth, layers,
                       f.astc_srgb ? "SRGB" : "UNORM");
            }
        }
        // Still over budget: keep trimming top mips until create succeeds or only 1x1 remains.
        while (r != VK_SUCCESS && f.astc_bw && levels > 1) {
            host_mip0++;
            w = mip(guest_w, host_mip0);
            h = mip(guest_h, host_mip0);
            levels = guest_levels > host_mip0 ? guest_levels - host_mip0 : 1;
            ci.extent.width = w;
            ci.extent.height = h;
            ci.mipLevels = levels;
            r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
            if (r == VK_SUCCESS)
                rt_log("vk: CompactLayers retry ASTC %#x host %ux%u levels %u (skip %u)", d->format, w, h, levels,
                       host_mip0);
        }
        if (r != VK_SUCCESS) {
            log_limited(&n_failed, "image creation failed (%d): nvn fmt %#x %ux%ux%u target %d levels %u layers %u", (int)r,
                        d->format, guest_w, guest_h, depth, d->target, guest_levels, layers);
            free(e);
            return NULL;
        }
    }
    VkTexImage *p = &e->pub;
    p->image = img;
    p->format = f.vk;
    p->type = type;
    p->usage = usage;
    p->aspect = aspect_of(&f);
    p->width = w, p->height = h, p->depth = depth, p->levels = levels, p->layers = layers, p->samples = samples;
    p->host_mip0 = (uint8_t)(host_mip0 > 255 ? 255 : host_mip0);
    e->f = f;
    p->view_type = type == VK_IMAGE_TYPE_1D ? (d->target == NVN_TEXTURE_TARGET_1D_ARRAY ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                 : type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                 : layers > 1 || d->target == NVN_TEXTURE_TARGET_2D_ARRAY || d->target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    p->alpha_one = f.alpha_one;
    memcpy(p->chan, f.chan, sizeof p->chan);
    p->swz[0] = (uint8_t)d->swizzle_r, p->swz[1] = (uint8_t)d->swizzle_g;
    p->swz[2] = (uint8_t)d->swizzle_b, p->swz[3] = (uint8_t)d->swizzle_a;
    if (!(p->swz[0] | p->swz[1] | p->swz[2] | p->swz[3]))   // descriptors built without nvnTextureBuilderSetDefaults
        p->swz[0] = NVN_TEXTURE_SWIZZLE_R, p->swz[1] = NVN_TEXTURE_SWIZZLE_G, p->swz[2] = NVN_TEXTURE_SWIZZLE_B,
        p->swz[3] = NVN_TEXTURE_SWIZZLE_A;
    p->mirrored = (uint8_t)tex_mirrored(t);
    p->bound = e->bound_to != NULL;
    p->sparse = e->sparse;
    p->block_views = (uint8_t)block_views;
    p->fb_usage = (uint8_t)fb_usage;

    if (!defer_init) {
        init_image(e, vk_upload_begin());
        vk_upload_end();
    }
    return e;
}

static int owned_lower(uint64_t a) {
    int lo = 0, hi = nowned;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (owned[m]->pub.addr < a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static void own_add(Tex *e) {
    if (e->listed) return;
    plat_mutex_lock(&owned_lock);
    if (nowned == cowned) {
        cowned = cowned ? 2 * cowned : 256;
        if (!(owned = realloc(owned, sizeof *owned * (size_t)cowned))) rt_fatal("vk: out of memory for the owned-texture list");
    }
    int i = owned_lower(e->pub.addr);
    memmove(owned + i + 1, owned + i, sizeof *owned * (size_t)(nowned - i));
    owned[i] = e;
    __atomic_store_n(&nowned, nowned + 1, __ATOMIC_RELEASE);
    if (e->t.L.size > max_owned) max_owned = e->t.L.size;
    e->listed = 1;
    plat_mutex_unlock(&owned_lock);
}

static void own_remove(Tex *e) {
    if (!e->listed) return;
    plat_mutex_lock(&owned_lock);
    for (int i = owned_lower(e->pub.addr); i < nowned && owned[i]->pub.addr == e->pub.addr; i++)
        if (owned[i] == e) {
            memmove(owned + i, owned + i + 1, sizeof *owned * (size_t)(nowned - i - 1));
            __atomic_store_n(&nowned, nowned - 1, __ATOMIC_RELEASE);
            break;
        }
    e->listed = 0;
    plat_mutex_unlock(&owned_lock);
}

static void rel_append(Tex *e) {
    e->rel_next = NULL;
    e->rel_prev = rel_tail;
    if (rel_tail) rel_tail->rel_next = e;
    else rel_head = e;
    rel_tail = e;
}

static void rel_unlink(Tex *e) {
    if (e->rel_prev) e->rel_prev->rel_next = e->rel_next;
    else rel_head = e->rel_next;
    if (e->rel_next) e->rel_next->rel_prev = e->rel_prev;
    else rel_tail = e->rel_prev;
    e->rel_prev = e->rel_next = NULL;
}

static void destroy_tex(Tex *e) {
    own_remove(e);
    tex_gen_bump(e->key.addr);
    nvn_hm_del(&by_key, &e->key);
    Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head == e) {
        if (e->next_at) nvn_hm_put(&by_addr, &e->key.addr, e->next_at);
        else nvn_hm_del(&by_addr, &e->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == e) { p->next_at = e->next_at; break; }
    }
    if (e->released) nreleased--, rel_unlink(e);
    vk_exec_image_gone(e->pub.image);
    uint64_t bf = e->sparse ? vk_sparse_destroy(e->pub.image) : 0;
    if (bf) vk_mem_protect_until(bf);
    vk_defer_image(e->pub.image, e->alloc);
    free(e->snap);
    free(e);
}

// Released entries with another descriptor at the same address are replaced; past RELEASED_MAX released entries the
// oldest one goes at each release (it comes back on demand, a mirrored one re-uploaded from its storage). TotK
// finalizes and initializes transient textures every frame, so a released entry is mostly taken back within frames.
// Dropping all of them at once when the count passed 4096 destroyed thousands of images in one frame: 333-422 ms
// frames every 40-50 s on a GTX 1050 Ti, then the textures still in use were created and uploaded again.
#define RELEASED_MAX 2048
static void sweep_released(void) {
    while (nreleased > RELEASED_MAX && rel_head) destroy_tex(rel_head);
}

// TOTK_TEX_VRAM_PRESSURE (percent of the device-local budget, default 90; 0 = off): above it, each submit drops up to
// PRESSURE_STEP of the oldest released entries, so a 4 GiB card does not keep RELEASED_MAX released images (each ASTC
// one decoded to RGBA8) while the heap spills into system memory. A few per submit, for the reason above. Only entries
// that come back intact are dropped: mirrored ones the GPU never wrote, re-uploaded from their storage. A render target
// lives only in its image; dropping it brought the texture back empty (black flashes in a Depths boss fight, ~170
// "left uninitialised" aliases a minute).
#define PRESSURE_STEP 32
static int vram_pressure = -1;
static uint64_t n_pressure_drops;

static void sweep_pressure(void) {   // tex_lock held
    if (vram_pressure < 0) {
        const char *s = cfg_get("TOTK_TEX_VRAM_PRESSURE");
        vram_pressure = s && *s ? atoi(s) : 90;
        if (vram_pressure < 0 || vram_pressure > 100) vram_pressure = 90;
        rt_log("vk: released textures under VRAM pressure: %s%d%%", vram_pressure ? "dropped above " : "off", vram_pressure);
    }
    if (!vram_pressure || !rel_head || submit_epoch & 7) return;   // the budget query every 8 submits
    uint64_t used, budget;
    int exact;
    if (!vk_mem_vram(&used, &budget, &exact) || used <= budget / 100 * (uint64_t)vram_pressure) return;
    int dropped = 0;
    for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP; e = next) {
        next = e->rel_next;
        if (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty) continue;
        destroy_tex(e);
        dropped++, n_pressure_drops++;
    }
    if (dropped) vk_mem_collect();
}

static int evict_vram_oom(int include_gpu_owned) {
    int dropped = 0;
    for (Tex *e = rel_head, *next; e; e = next) {
        next = e->rel_next;
        if (!include_gpu_owned && (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty)) continue;
        destroy_tex(e);
        dropped++;
    }
    if (dropped || !include_gpu_owned) {
        vk_upload_flush(1);
        vk_mem_collect();
    }
    return dropped;
}

// ---------------------------------------------------------------- uploads
static uint32_t texel_alpha_one(int format, uint32_t *elem) {
    switch (format) {
    case 0x19: *elem = 1; return 0xFF;
    case 0x1A: *elem = 1; return 0x7F;
    case 0x1B: case 0x1C: *elem = 1; return 1;
    case 0x1D: *elem = 2; return 0x3C00;
    case 0x1E: *elem = 2; return 0xFFFF;
    case 0x1F: *elem = 2; return 0x7FFF;
    case 0x20: case 0x21: *elem = 2; return 1;
    case 0x22: *elem = 4; return 0x3F800000;
    default: *elem = 4; return 1;   // 0x23, 0x24
    }
}

// 3-component texels to 4 (n texels).
static void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e, one = texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 3 * e, dst += 4 * e) {
        memcpy(dst, src, 3 * e);
        memcpy(dst + 3 * e, &one, e);
    }
}

static void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e;
    texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 4 * e, dst += 3 * e) memcpy(dst, src, 3 * e);
}

// TOTK_ASTC=check: the compute decoder's texels of n regions (gpu: the regions one after the other) against
// astc_decode_ldr of the same raw blocks (raw: the staging bytes the regions point into).
static void astc_check(const Tex *e, const uint8_t *raw, const VkAstcRegion *ar, int n, const uint8_t *gpu) {
    static volatile int32_t n_bad;
    static uint64_t n_tex, n_texels, n_diff_tex;
    uint32_t bw = e->f.astc_bw, bh = e->f.astc_bh;
    uint64_t diff = 0, first = UINT64_MAX;
    uint32_t want = 0, got = 0;
    int first_k = 0;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        size_t sb = (size_t)((r->w + bw - 1) / bw) * ((r->h + bh - 1) / bh) * 16, st = (size_t)r->w * r->h * 4;
        uint8_t *cpu = malloc(st);
        for (uint32_t z = 0; z < r->slices; z++, gpu += st) {
            astc_decode_ldr(raw + r->src + z * sb, r->w, r->h, bw, bh, e->f.astc_srgb, cpu);
            if (!memcmp(cpu, gpu, st)) continue;
            for (size_t i = 0; i < st / 4; i++)
                if (memcmp(cpu + 4 * i, gpu + 4 * i, 4)) {
                    if (first == UINT64_MAX) {
                        first = i, first_k = k;
                        memcpy(&want, cpu + 4 * i, 4), memcpy(&got, gpu + 4 * i, 4);
                    }
                    diff++;
                }
        }
        free(cpu);
        n_texels += (uint64_t)r->w * r->h * r->slices;
    }
    n_tex++;
    if (diff) {
        n_diff_tex++;
        log_limited(&n_bad, "ASTC check: texture %#llx fmt %#x %ux%u: %llu texels differ, first region %d texel %llu "
                    "cpu %08x gpu %08x", (unsigned long long)e->pub.addr, e->t.d.format, bw, bh,
                    (unsigned long long)diff, first_k, (unsigned long long)first, want, got);
    }
    if (n_tex == 1 || !(n_tex & 63) || diff)
        rt_log("vk: ASTC check: %llu textures, %llu texels compared, %llu textures differ", (unsigned long long)n_tex,
               (unsigned long long)n_texels, (unsigned long long)n_diff_tex);
}

// Copies guest storage into the image. layer0/nlayers select a subset of array layers (nlayers 0 = all); used by
// sparse MapVirtual so streamed tiles refresh only the hit layers instead of flashing the whole CompactLayers array.
static void upload_layers(Tex *e, uint32_t layer0, uint32_t nlayers) {
    static volatile int32_t n_bad, n_undecoded;
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || !t->mem) return;
    uint64_t p0 = perf_begin(), u0 = kern_now_ns();
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    uint32_t host_levels = linear ? 1 : e->pub.levels, layers_all = linear ? 1 : e->pub.layers;
    if (layer0 >= layers_all) return;
    uint32_t layers = nlayers && layer0 + nlayers <= layers_all ? nlayers : layers_all - layer0;
    uint32_t skip = linear ? 0 : e->pub.host_mip0;   // CompactLayers: guest mip skip maps to host level 0
    int astc = e->f.astc_bw != 0, rgb = e->f.expand_rgb;
    int agpu = astc && vk_astc_gpu(), acheck = agpu && vk_astc_mode() == VK_ASTC_CHECK;   // raw blocks to the GPU
    uint32_t n = host_levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    VkAstcRegion *ar = agpu ? calloc(n, sizeof *ar) : NULL;
    VkDeviceSize total = 0, decoded = 0;
    for (uint32_t hl = 0; hl < host_levels; hl++) {
        uint32_t lv = hl + skip;   // guest level
        for (uint32_t i = 0; i < layers; i++) {
            uint32_t ly = layer0 + i;
            uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
            uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)(w + bw - 1) / bw * bpb * ((h + bh - 1) / bh) * sl;
            uint64_t out = agpu ? raw : astc ? texels * 4 : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            rg[k++] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, hl, ly, 1},
                                          .imageExtent = {w, h, sl}};
            total += (out + 15) & ~15ull;
            decoded += texels * 4;
        }
    }
    VkCommandBuffer cb = vk_upload_begin();
    VkBuffer sb, rbuf = VK_NULL_HANDLE;
    VmaAllocation ralloc = NULL;
    uint8_t *stg = vk_staging(total, &sb, NULL), *rmap = acheck ? vk_staging(decoded, &rbuf, &ralloc) : NULL;
    k = 0;
    for (uint32_t hl = 0; hl < host_levels; hl++) {
        uint32_t lv = hl + skip;
        for (uint32_t i = 0; i < layers; i++, k++) {
            uint32_t ly = layer0 + i;
            uint32_t w = rg[k].imageExtent.width, h = rg[k].imageExtent.height, sl = rg[k].imageExtent.depth;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            size_t raw = (size_t)rb * rows * sl;
            uint8_t *dst = stg + rg[k].bufferOffset, *tmp = (astc && !agpu) || rgb ? malloc(raw) : NULL;
            if (nvn_deswizzle(d, lv, ly, t->mem, tmp ? tmp : dst, rb)) {
                log_limited(&n_bad, "texture %#llx fmt %#x target %d: level %u layer %u has no CPU layout; uploaded as zeros",
                            (unsigned long long)e->pub.addr, d->format, d->target, lv, ly);
                memset(tmp ? tmp : dst, 0, raw);
            }
            if (agpu) {
                ar[k] = (VkAstcRegion){.src = rg[k].bufferOffset, .w = w, .h = h, .slices = sl, .level = hl, .layer = ly};
            } else if (astc) {
                uint64_t a0 = perf_begin();
                for (uint32_t z = 0; z < sl; z++)
                    astc_decode_ldr_mt(tmp + (size_t)z * rb * rows, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                       dst + (size_t)z * w * h * 4);
                perf_end(PERF_ASTC, a0);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else if (rgb) {
                expand_rgb(d->format, tmp, dst, (size_t)w * h * sl);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else {
                rg[k].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[k].bufferImageHeight = rows * (uint32_t)bh;
            }
            free(tmp);
        }
    }
    uint8_t *raw_copy = acheck ? malloc(total) : NULL;
    if (raw_copy) memcpy(raw_copy, stg, total);
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int checked = 0;
    if (agpu) {
        uint64_t a0 = perf_begin();
        checked = vk_astc_record(cb, sb, total, ar, (int)n, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc, e->pub.image,
                                 e->pub.aspect, rbuf);
        perf_end(PERF_ASTC, a0);
        if (!checked)
            log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: left as it was",
                        (unsigned long long)e->pub.addr, d->format, d->width, d->height);
    } else {
        vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    }
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    {   // TOTK_COMPACT_TRACE=1: CompactLayers arrays and big decoded staging uploads
        static int tr = -1;
        static volatile int32_t ntr;
        if (tr < 0) {
            const char *te = cfg_get("TOTK_COMPACT_TRACE");
            tr = te && *te && *te != '0';
        }
        if (tr && (skip || (d->width >= 4096 && d->height >= 4096) || (astc && d->width >= 512 && d->height >= 512)) && __atomic_fetch_add(&ntr, 1, __ATOMIC_RELAXED) < 1500)
            rt_log("ctrace: upload tex %#llx fmt %#x guest %ux%u L%u -> host %ux%u L%u skip %u layers %u+%u/%u regions %u "
                   "bytes %llu agpu %d decoded %d mirrored %d bound %d",
                   (unsigned long long)e->pub.addr, d->format, d->width, d->height, d->levels, e->pub.width, e->pub.height,
                   host_levels, skip, layer0, layers, layers_all, n, (unsigned long long)total, agpu, checked,
                   e->pub.mirrored, e->bound_to != NULL);
    }
    vk_upload_end();
    if (total > (64ull << 20)) {
        vk_upload_flush(1);
    }
    if (acheck) {   // waits for the GPU
        vk_upload_flush(1);
        if (checked) {
            vmaInvalidateAllocation(vk_ctx()->vma, ralloc, 0, VK_WHOLE_SIZE);
            astc_check(e, raw_copy, ar, (int)n, rmap);
        }
        vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
    }
    free(raw_copy);
    free(ar);
    free(rg);
    uint64_t dt = kern_now_ns() - u0;
    if (e->uploaded) {
        n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
        ru_n++, ru_bytes += t->L.size, ru_ns += dt;
        if (e->wt_n) {
            ru_wt_n++, ru_wt_bytes += t->L.size, ru_wt_ns += dt;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("perf: frame %llu: re-upload after %u WriteTexels: texture %#llx fmt %#x %ux%u, %.1f KiB, %.3f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), e->wt_n,
                       (unsigned long long)e->pub.addr, d->format, d->width, d->height, t->L.size / 1024.0, dt / 1e6);
        }
    } else {
        n_uploads++, perf_count(PERF_CNT_TEX_UP);
    }
    e->wt_n = 0;
    e->uploaded = 1;
    e->n_up++;
    perf_end(PERF_TEX, p0);
}

static void upload(Tex *e) { upload_layers(e, 0, 0); }

// Whether e's storage fills no 4 KiB page of its own, so the write tracking can only see writes to pages it shares.
static int fills_no_page(const Tex *e) {
    return ((e->pub.addr + e->t.L.size) & ~0xfffull) <= ((e->pub.addr + 0xfff) & ~0xfffull);
}

// A texture that fills no page keeps a copy of its storage bytes from when a GPU write to it was recorded
// (vk_texture_gpu_written), up to SNAP_MAX bytes; wb_exact (TOTK_WB_EXACT, default on) enables it.
enum { SNAP_MAX = 4096 };
static int wb_exact = -1;
static volatile int32_t n_kept;

// vk_mem_changed over the pages wholly inside e's storage (all of its pages when it fills none): a neighbour that shares
// the first or last page does not count as a write to e. A texture that fills no page shares all of its pages, so
// there a page write counts only when e's own bytes differ from the copy taken at the GPU write. TotK's Purah Pad
// camera renders the ID, screen rect and depth of what is under the reticle into 4x4 targets of 512 bytes, two
// slots used in turn, and the CPU reads them a frame later. One slot's rect target shares its page with data the CPU
// writes every frame: without the copy its renders were never written back, the game read a zero rect every other
// frame and the target's name label blinked. TOTK_WB_EXACT=0 restores the page-only decision.
static int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    uint64_t lo = (e->pub.addr + 0xfff) & ~0xfffull, hi = (e->pub.addr + e->t.L.size) & ~0xfffull;
    if (hi <= lo) lo = e->pub.addr, hi = e->pub.addr + e->t.L.size;
    int ch = vk_mem_changed(lo, hi - lo, since, now);
    if (ch != 1 || !since || !e->snap || memcmp(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size)) return ch;
    log_limited(&n_kept, "write-back of %#llx kept: fmt %#x %ux%u, %llu bytes; a neighbour wrote its page",
                (unsigned long long)e->pub.addr, e->t.d.format, e->t.d.width, e->t.d.height,
                (unsigned long long)e->t.L.size);
    return 0;
}

// ---- content hash for untracked storage (TOTK_TEX_HASH, default on; =0 uploads again on every untracked check)
// Where the pages have no write watch (macOS: vk_mem_changed answers -1), a mirrored texture used to be deswizzled and
// uploaded again in every submission that bound it. On an M4 that grew to 17,000-20,000 re-uploads (about 3 GiB) per
// 60 frames a minute into play, and the frame rate fell to 8-11 fps (#116). Instead, the storage bytes are hashed just
// before each upload, and an untracked check uploads again only when the hash differs.
// Why a match is safe: a match is treated as a clean write-watch answer (ch == 0), which is what the tracked platforms
// do for a texture whose pages did not change. Most GPU writes into the image (vk_texture_gpu_written, alias_copy)
// make it GPU-owned for good, so it is never uploaded again. Two writers do not: storage-image stores
// (tex_info_full in vk_exec_draw.c only calls vk_texture_storage_bound_locked) and the MoltenVK upscaler blit
// (mvk_upscale_blit). Their results survive an unchanged-storage check exactly as they survive ch == 0 on Windows,
// where the old macOS path wiped them on every submission. The hash is taken before the copy, so a guest write racing
// the copy leaves the stored hash older than the image and costs one extra upload, never a missed one. The one miss
// is an ABA: the guest changes a block A to B while it is hashed, the copy reads B, then the guest writes A back, so
// the next check matches and the image keeps B. A write watch would catch that; the odds are small.
static int tex_hash = -1;

static inline uint64_t rotl64(uint64_t x, int r) { return x << r | x >> (64 - r); }

// Four independent multiply-rotate lanes over 32-byte blocks (the xxHash64 round), then the tail and a final mix:
// several GB/s, where the byte-wise FNV-1a manages about one. Equality detection only, not a stable format.
static uint64_t storage_hash(const Tex *e) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)e->pub.addr;
    uint64_t n = e->t.L.size, i = 0;
    const uint64_t P1 = 0x9e3779b185ebca87ull, P2 = 0xc2b2ae3d27d4eb4full;
    uint64_t a = P1 + P2, b = P2, c = 0, d = 0 - P1;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        memcpy(w, p + i, 32);
        a = rotl64(a + w[0] * P2, 31) * P1;
        b = rotl64(b + w[1] * P2, 31) * P1;
        c = rotl64(c + w[2] * P2, 31) * P1;
        d = rotl64(d + w[3] * P2, 31) * P1;
    }
    uint64_t h = rotl64(a, 1) + rotl64(b, 7) + rotl64(c, 12) + rotl64(d, 18) + n;
    for (; i < n; i++) h = rotl64(h ^ (p[i] * P1), 11) * P2;
    h ^= h >> 33, h *= P2, h ^= h >> 29, h *= P1, h ^= h >> 32;
    return h;
}

// Called with the page answer ch just before e would be uploaded. Untracked (-1): takes the storage hash for the upload
// that may follow and returns 1 when it equals the one taken before e's last upload (the upload is then skipped).
// Tracked: forgets the hash, since an upload made on a write-watch answer has none.
static int untracked_same(Tex *e, int ch) {
    if (tex_hash < 0) tex_hash = cfg_bool("TOTK_TEX_HASH");
    if (ch != -1 || !tex_hash) { e->hashed = 0; return 0; }
    uint64_t h = storage_hash(e);
    int same = e->hashed && e->uploaded && h == e->chash;
    e->chash = h, e->hashed = 1;
    if (same) n_hash_kept++;
    return same;
}

// A mirrored texture is uploaded on first use and again when its storage pages carry a newer write generation.
static void make_fresh(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint64_t size = e->t.L.size;
    if (!e->uploaded) {
        int ch = vk_mem_changed(e->pub.addr, size, 0, &e->wgen);   // resets the write bits before the storage is read
        untracked_same(e, ch);   // takes the hash for the next untracked check
        upload(e);
        return;
    }
    uint32_t now;
    int ch = vk_mem_changed(e->pub.addr, size, e->wgen, &now);
    e->wgen = now;
    e->last_ch = ch;
    if (ch && !untracked_same(e, ch)) upload(e);   // 1 written, -1 not tracked and the bytes changed: upload again
}

// make_fresh for draw bindings: the page check answers from this submission's write-watch snapshot (vk_memory.c
// vk_mem_changed_snap) instead of a system call per texture. A write the snapshot of the texture's region missed (made
// after it, during this replay) keeps its write-watch bit and is caught by the next submission's snapshot.
static void make_fresh_snap(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint32_t now;
    n_checks++;
    int ch = vk_mem_changed_snap(e->pub.addr, e->t.L.size, e->uploaded ? e->wgen : 0, submit_epoch, &now);
    e->wgen = now;
    if (!e->uploaded) { untracked_same(e, ch); upload(e); return; }
    e->last_ch = ch;
    if (ch && !untracked_same(e, ch)) upload(e);
}

// ---------------------------------------------------------------- aliases (TOTK_TEX_ALIAS)
// Bytes per block and block size of the image's VkFormat (ASTC decodes to RGBA8, RGB expands to RGBA); 0 when the
// VkFormat substitutes another one of unknown size.
static int img_block(const Tex *e, int *bw, int *bh) {
    if (e->f.astc_bc) { *bw = *bh = 4; return 16; }
    if (e->f.astc_bw) { *bw = *bh = 1; return 4; }
    int b = nvn_fmt_bpb(e->t.d.format, bw, bh);
    if (e->f.expand_rgb) { *bw = *bh = 1; return b / 3 * 4; }
    return e->f.vk == e->f.ideal ? b : 0;
}

// The new entry e can take src's content texel for texel: colour, one sample, the same shape and storage layout, and
// image formats of one block size.
static int alias_compatible(const Tex *e, const Tex *src) {
    const VkTexImage *p = &e->pub, *q = &src->pub;
    if (e->sparse || src->sparse || e->f.depth || e->f.stencil || src->f.depth || src->f.stencil || p->samples > 1 || q->samples > 1) return 0;
    if (p->type != q->type || p->width != q->width || p->height != q->height || p->depth != q->depth) return 0;
    if (e->t.L.size != src->t.L.size || nvn_tex_is_linear(&e->t.d) != nvn_tex_is_linear(&src->t.d)) return 0;
    int ew, eh, sw, sh, eb = img_block(e, &ew, &eh), sb = img_block(src, &sw, &sh);
    return eb && eb == sb && ew == sw && eh == sh;
}

// src holds content the storage bytes do not: the GPU wrote it and, for a mirrored texture, it was not written back
// and the CPU has not rewritten the storage since (the new entry's upload then reads the newer bytes).
static int alias_has_gpu_content(const Tex *src) {
    if (!src->pub.gpu_owned) return 0;
    if (!src->pub.mirrored) return 1;
    uint32_t now;
    return src->gpu_dirty && !storage_rewritten(src, src->gpu_gen, &now);
}

// Maxwell reads the same bytes through both descriptors: src's content goes into the new image e at the replay's
// current position, before any later command of the submission uses e. Through a buffer, since bound images share
// their memory and image copies may not overlap; e leaves UNDEFINED between the two halves, so its first transition
// cannot clobber src's bytes before they are read. 0 when no replay is recording (e is then not initialised yet).
static int alias_copy(Tex *e, const Tex *src) {
    int reopen;
    VkCommandBuffer cb = vk_exec_inline_begin(&reopen);
    if (!cb) return 0;
    int bw, bh, bpb = img_block(e, &bw, &bh);
    uint32_t levels = e->pub.levels < src->pub.levels ? e->pub.levels : src->pub.levels;
    uint32_t layers = e->pub.layers < src->pub.layers ? e->pub.layers : src->pub.layers, n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for an alias copy");
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(e->pub.width, lv), h = mip(e->pub.height, lv), z = mip(e->pub.depth, lv);
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1},
                                        .imageExtent = {w, h, z}};
            total += ((VkDeviceSize)(w + bw - 1) / bw * ((h + bh - 1) / bh) * z * bpb + 15) & ~15ull;
        }
    VkBuffer buf = vk_exec_inline_buffer(total);
    if (!buf) {
        free(rg);
        vk_exec_inline_end(reopen);
        return 0;
    }
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, src->pub.image, VK_IMAGE_LAYOUT_GENERAL, buf, n, rg);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdCopyBufferToImage(cb, buf, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_exec_inline_end(reopen);
    free(rg);
    return 1;
}

static Tex *entry_of(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e) {
        if (tex_alias < 0) {
            tex_alias = cfg_bool("TOTK_TEX_ALIAS");
            rt_log("vk: texture aliases: %s", tex_alias ? "a new image takes the live image's content" : "a new image starts empty");
        }
        if (!(e = create_tex(t, &k, tex_alias))) return NULL;
        perf_count(PERF_CNT_TEX_NEW);
        // released entries of another descriptor at this address are replaced by the new one; a live one holding
        // GPU-written content is the alias source, preferring the one that differs only in swizzle (same_but_swizzle)
        int alias = 0;
        Tex *src = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr), *nx; p; p = nx) {
            nx = p->next_at;
            if (p->released) { destroy_tex(p); continue; }
            alias = 1;
            if (tex_alias && (!src || (same_but_swizzle(&p->key, &k) && !same_but_swizzle(&src->key, &k))) &&
                alias_compatible(e, p) && alias_has_gpu_content(p))
                src = p;
        }
        if (alias) perf_count(PERF_CNT_TEX_ALIAS);
        if (tex_alias) {
            if (src && alias_copy(e, src)) {
                n_alias_copied++;
                if (e->pub.mirrored) {   // the content is src's GPU render, newer than the storage bytes
                    vk_mem_changed(e->pub.addr, e->t.L.size, 0, &e->wgen);
                    e->uploaded = 1;
                    e->pub.gpu_owned = 1;
                }
            } else {
                init_image(e, vk_upload_begin());
                vk_upload_end();
                if (alias) {
                    if (e->pub.mirrored) n_alias_upload++;
                    else n_alias_uninit++;
                }
            }
        }
        e->next_at = nvn_hm_get(&by_addr, &k.addr);
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
    }
    if (e->released) e->released = 0, nreleased--, rel_unlink(e), tex_gen_bump(k.addr);
    e->t = *t;
    return e;
}

// vk_dump: upload bookkeeping of an image vk_texture_get / vk_texture_by_id returned. pages_written: whether its
// storage pages were written since the last check (1, 0, -1 not tracked); the query leaves that check's state alone.
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written) {
    lock();
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint32_t now;
    *uploads = e->n_up;
    *last_change = e->last_ch;
    *pages_written = e->pub.mirrored && e->uploaded ? vk_mem_changed(e->pub.addr, e->t.L.size, e->wgen, &now) : 0;
    unlock();
}

const VkTexImage *vk_texture_get(const tex_t *t) {
    if (!t || !t->mem) return NULL;
    lock();
    Tex *e = entry_of(t);
    if (e) make_fresh(e);
    unlock();
    return e ? &e->pub : NULL;
}

// Small render targets the CPU reads without NVN's invalidate (vk_texture_write_back_small); small_pending: one may be
// dirty. wb_limit is the largest target (bytes) that goes back when the CPU waits for a fence: TOTK_WB_FENCE_LIMIT,
// read once. wb_trace: TOTK_TRACE=gfx, read together with it.
enum { WB_SMALL_DEFAULT = 64 << 10 };
static int small_pending;
static uint64_t wb_limit = WB_SMALL_DEFAULT;
static int wb_trace;
static int wb_cfg_read;

static void wb_cfg_init(void) {
    if (wb_cfg_read) return;
    int v = cfg_int("TOTK_WB_FENCE_LIMIT");
    wb_limit = v < 0 ? 0 : (uint64_t)v;
    wb_trace = (rt_trace & RT_TRACE_GFX) != 0;
    wb_cfg_read = 1;
    if (wb_limit != WB_SMALL_DEFAULT) rt_log("vk: fence write-back limit %llu bytes", (unsigned long long)wb_limit);
}

void vk_texture_gpu_written(const tex_t *t) {
    if (!t || !t->mem) return;
    vk_exec_frame_written(t);
    lock();
    Tex *e = entry_of(t);
    if (e) {
        make_fresh(e);   // a partial render keeps what the storage held before
        e->pub.gpu_owned = 1;
        uint64_t f = vk_exec_recording_fence();
        if (f > e->wr_fence) e->wr_fence = f;
        if (e->pub.mirrored) {
            e->gpu_dirty = 1;
            storage_rewritten(e, 0, &e->gpu_gen);   // CPU writes up to here are older than this render
            if (wb_exact < 0) wb_exact = cfg_bool("TOTK_WB_EXACT");
            if (wb_exact && !e->snap && e->t.L.size <= SNAP_MAX && fills_no_page(e) && !(e->snap = malloc(SNAP_MAX)))
                rt_fatal("vk: out of memory for a texture's storage copy");
            if (e->snap) memcpy(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size);
            e->gpu_seq = ++gpu_writes;
            e->gpu_fence = vk_exec_recording_fence();
            wb_cfg_init();
            if (e->t.L.size <= wb_limit) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
            own_add(e);
        }
    }
    unlock();
}

void vk_texture_release(const tex_t *t) {
    if (!t || !t->mem) return;
    TexKey k;
    make_key(t, &k);
    lock();
    Tex *e = nvn_hm_get(&by_key, &k);
    if (e && !e->released) {
        e->released = 1;
        nreleased++;
        rel_append(e);
        tex_gen_bump(k.addr);
        sweep_released();
    }
    unlock();
}

void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc) {
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && ((e->key.addr >= lo && e->key.addr - lo < size) || (alloc && e->bound_to == alloc))) list[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    unlock();
}

void vk_texture_sparse_pages_mapped(uint64_t lo, uint64_t size) {
    if (!size || lo > UINT64_MAX - size) return;
    uint64_t hi = lo + size;
    // Stamp so later make_fresh_snap sees the newly backed pages even when we skip a full wipe.
    vk_guest_written(lo, size);
    uint32_t n_full = 0, n_part = 0;
    lock();
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (!e || e->released) continue;
        uint64_t a = e->key.addr, b = a + e->t.L.size;
        if (a >= hi || b <= lo) continue;
        // Pool-aliased images keep their create-time physical bind; destroying them on every 64 KiB
        // MapVirtual would recreate the whole array per streamed tile. Real fix is sparse residency.
        if (e->bound_to || e->sparse || (!e->uploaded && !e->pub.mirrored)) continue;
        // CompactLayers / own-memory arrays: re-upload only layers that overlap the mapped pages so
        // streamed tiles do not flash the whole 68-layer host (pop-in). Fall back to a full mark when
        // layer_size is unknown or the hit is not layered.
        uint64_t ls = e->t.L.layer_size;
        uint32_t layers = e->pub.layers;
        if (e->uploaded && ls && layers > 1) {
            uint32_t first = layers, last = 0;
            for (uint32_t ly = 0; ly < layers; ly++) {
                uint64_t la = a + (uint64_t)ly * ls, lb = la + ls;
                if (la < hi && lb > lo) {
                    if (ly < first) first = ly;
                    if (ly > last) last = ly;
                }
            }
            if (first <= last) {
                upload_layers(e, first, last - first + 1);
                n_part++;
                continue;
            }
        }
        e->uploaded = 0;
        e->checked = 0;
        e->wgen = 0;
        n_full++;
    }
    unlock();
    if (n_full || n_part) {
        static volatile int32_t logged;
        if (__atomic_fetch_add(&logged, 1, __ATOMIC_RELAXED) < 8)
            rt_log("vk: sparse MapVirtual %#llx+%#llx: %u layer-scoped re-upload(s), %u full invalidate(s)",
                   (unsigned long long)lo, (unsigned long long)size, n_part, n_full);
    }
}

// ---------------------------------------------------------------- descriptor slots
static int descriptor_key(const dpool_ref *pool, int id, uint32_t limit, VkDescKey *key) {
    if (!pool || !pool->addr || !pool->count || pool->count > limit || id < 0 || (uint32_t)id >= pool->count ||
        pool->addr % NVN_DESCRIPTOR_BYTES || pool->addr > UINT64_MAX - (uint64_t)pool->count * NVN_DESCRIPTOR_BYTES) return 0;
    return vk_mem_descriptor_key(pool->addr + (uint64_t)(uint32_t)id * NVN_DESCRIPTOR_BYTES, NVN_DESCRIPTOR_BYTES, key);
}

// Texture lock held. Entries are addressed by their bytes, so wrappers and overlapping table views share updates.
static DescSlot *descriptor_slot(const dpool_ref *pool, int id, uint32_t limit, int create) {
    VkDescKey key;
    if (!descriptor_key(pool, id, limit, &key)) return NULL;
    DescSlot *d = nvn_hm_get(&descriptors, &key);
    if (!d && create) {
        d = calloc(1, sizeof *d);
        if (!d) rt_fatal("vk: out of memory for a descriptor");
        nvn_hm_put(&descriptors, &key, d);
    }
    return d;
}

static void descriptor_registered(DescSlot *d, uint32_t kind) {
    memset(d, 0, sizeof *d);
    if (!++reg_serial) ++reg_serial;
    d->serial = reg_serial, d->kind = kind;
    slots_gen++, samp_gen++;   // a registration can replace bytes previously used by either kind
}

void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    static volatile int32_t n_unbacked;
    if (!pool || id < 0 || id >= (1 << 20) || (uint32_t)id >= pool->count) rt_fatal("vk: texture slot %d out of pool range", id);
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 1);
    if (!d) {
        log_limited(&n_unbacked, "texture descriptor slot %d has no mapped backing: not registered", id);
        unlock();
        return;
    }
    descriptor_registered(d, DESC_TEXTURE);
    TexSlot *s = &d->u.tex;
    s->t = *t;
    s->used = 1;
    s->ce = NULL;
    s->has_view = v != NULL;
    if (v) s->v = *v;
    else memset(&s->v, 0, sizeof s->v);
    unlock();
}

// The entry slot s resolves to, and the view and has_view a caller gets with it (the slot's, with the descriptor's
// swizzle when the entry is another descriptor at the same storage). NULL when the slot names no texture.
static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view) {
    *view = s->v;
    *has_view = s->has_view;
    TexKey k;
    make_key(&s->t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e || e->released) {   // MNG hle_nvn.c: a finalized slot follows the live texture at its storage
        // Prefer the live entry that differs from the slot descriptor only in its swizzle: transient render targets
        // share storage, and the newest one at the address is usually another size. The map screen samples its
        // 2250x1080 map target through an (R,G,B,1) descriptor and followed a 960x60 target at the same address.
        Tex *own = e, *any = NULL;
        e = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr); p; p = p->next_at) {
            if (p->released) continue;
            if (!any) any = p;
            if (same_but_swizzle(&p->key, &k)) { e = p; break; }
        }
        if (!e) e = any ? any : own;
    }
    if (!e) e = s->t.mem ? entry_of(&s->t) : NULL;
    // A followed texture is another descriptor at the same storage: its image holds the content, but sampling takes the
    // slot descriptor's swizzle. The UI renders text into RGBA8 and samples it through a (A,A,A,A) descriptor; with the
    // render target's RGBA swizzle the text blur saw R = 1 everywhere and drew solid boxes behind tips and labels.
    if (e && !(s->has_view && s->v.set_swizzle)) {
        const nvn_texdesc *d = &s->t.d;
        uint8_t sw[4] = {(uint8_t)d->swizzle_r, (uint8_t)d->swizzle_g, (uint8_t)d->swizzle_b, (uint8_t)d->swizzle_a};
        if (!(sw[0] | sw[1] | sw[2] | sw[3]))   // descriptors built without nvnTextureBuilderSetDefaults
            sw[0] = NVN_TEXTURE_SWIZZLE_R, sw[1] = NVN_TEXTURE_SWIZZLE_G, sw[2] = NVN_TEXTURE_SWIZZLE_B,
            sw[3] = NVN_TEXTURE_SWIZZLE_A;
        if (memcmp(sw, e->pub.swz, 4)) {
            if (!s->has_view) memset(view, 0, sizeof *view);
            memcpy(view->swz, sw, 4);
            view->set_swizzle = 1;
            *has_view = 1;
        }
    }
    return e;
}

void vk_texture_lock(void) { lock(); }
void vk_texture_unlock(void) { unlock(); }
uint32_t vk_texture_slots_gen(void) { return slots_gen; }

uint64_t vk_texture_slot_token(const dpool_ref *pool, int id) {
    // tex_lock is recursive: callers may already hold it, or may be checking a cache before acquiring it.
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 0);
    uint64_t token = d && d->kind == DESC_TEXTURE ? (uint64_t)d->serial << 32 |
        __atomic_load_n(&agen[agen_idx((uintptr_t)d->u.tex.t.mem)], __ATOMIC_ACQUIRE) : 0;
    unlock();
    return token;
}
uint32_t vk_sampler_gen(void) { return __atomic_load_n(&samp_gen, __ATOMIC_RELAXED); }

const VkTexImage *vk_texture_by_id(const dpool_ref *pool, int id, view_t *view, int *has_view) {
    lock();
    const VkTexImage *r = vk_texture_by_id_locked(pool, id, view, has_view);
    unlock();
    return r;
}

const VkTexImage *vk_texture_by_id_locked(const dpool_ref *pool, int id, view_t *view, int *has_view) {
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 0);
    if (!d || d->kind != DESC_TEXTURE) return NULL;
    TexSlot *s = &d->u.tex;
    uint32_t g = agen[agen_idx((uintptr_t)s->t.mem)];
    if (!s->ce || s->cgen != g) {   // resolved again: an entry at its address was created, dropped or released, or the
        view_t v;                   // slot registered (s->ce cleared); an empty result is not kept: image creation may
        int hv;                     // succeed later
        s->ce = slot_resolve(s, &v, &hv);   // entry_of may create an entry and bump g: g read before it
        s->cview = v, s->chas_view = (uint8_t)hv;
        s->cgen = g;
        s->cepoch = 0;
    }
    Tex *e = s->ce;
    // Bindings of one submit share one check, answered from the submit's write-watch snapshot (guest writes during a
    // replay race the GPU read on hardware too). After one check an entry is either uploaded or never uploads (not
    // mirrored, GPU-owned or no CPU layout), so a slot that checked its entry in this submit has nothing left to do.
    if (e && s->cepoch != submit_epoch) {
        if (!e->uploaded || e->checked != submit_epoch) {
            make_fresh_snap(e);
            e->checked = submit_epoch;
        }
        s->cepoch = submit_epoch;
    }
    if (view) *view = has_view ? s->cview : s->v;
    if (has_view) *has_view = view ? s->chas_view : s->has_view;
    return e ? &e->pub : NULL;
}

// vk_dump: the swizzle selectors of the descriptor registered in slot id; 0 when the slot is empty.
int vk_texture_slot_swz(const dpool_ref *pool, int id, uint8_t swz[4]) {
    lock();
    DescSlot *s = descriptor_slot(pool, id, 1u << 20, 0);
    int ok = s && s->kind == DESC_TEXTURE;
    if (ok) {
        const nvn_texdesc *d = &s->u.tex.t.d;
        swz[0] = (uint8_t)d->swizzle_r, swz[1] = (uint8_t)d->swizzle_g, swz[2] = (uint8_t)d->swizzle_b;
        swz[3] = (uint8_t)d->swizzle_a;
    }
    unlock();
    return ok;
}

uint32_t vk_texture_epoch(void) { return __atomic_load_n(&submit_epoch, __ATOMIC_RELAXED); }

void vk_texture_new_submit(void) {
    lock();
    if (!++submit_epoch) submit_epoch = 1;
    sweep_pressure();
    static uint64_t log_at;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) / 60;
    if ((rt_trace & RT_TRACE_PERF) && f != log_at && tex_alias > 0) {
        log_at = f;
        rt_log("perf: frame %llu: texture aliases: content copied %llu, filled from storage %llu, left uninitialised %llu",
               (unsigned long long)f * 60, (unsigned long long)n_alias_copied, (unsigned long long)n_alias_upload,
               (unsigned long long)n_alias_uninit);
        n_alias_copied = n_alias_upload = n_alias_uninit = 0;
        rt_log("perf: frame %llu: texture re-uploads %llu (%.1f MiB, %.2f ms), after WriteTexels %llu (%.1f MiB, %.2f ms); "
               "untracked and unchanged, kept %llu",
               (unsigned long long)f * 60, (unsigned long long)ru_n, ru_bytes / 1048576.0, ru_ns / 1e6,
               (unsigned long long)ru_wt_n, ru_wt_bytes / 1048576.0, ru_wt_ns / 1e6, (unsigned long long)n_hash_kept);
        ru_n = ru_bytes = ru_ns = ru_wt_n = ru_wt_bytes = ru_wt_ns = n_hash_kept = 0;
        uint64_t ck, calls, untr;
        vk_mem_snap_stats(&ck, &calls, &untr);
        rt_log("perf: frame %llu: binding storage checks %llu (texture %llu), write-watch snapshot queries %llu, untracked %llu",
               (unsigned long long)f * 60, (unsigned long long)ck, (unsigned long long)n_checks, (unsigned long long)calls,
               (unsigned long long)untr);
        n_checks = 0;
        rt_log("perf: frame %llu: released textures kept %d, dropped under VRAM pressure %llu", (unsigned long long)f * 60,
               nreleased, (unsigned long long)n_pressure_drops);
        n_pressure_drops = 0;
    }
    unlock();
}

// ---------------------------------------------------------------- texel reads and writes
typedef struct { unsigned level, layer0; int nz, is3d, bw, bh, bpb; size_t wb, hb, row, img, x0, y0; } Region;

static int region_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img, Region *g) {
    g->bpb = nvn_fmt_bpb(t->d.format, &g->bw, &g->bh);
    if (!g->bpb || r->width <= 0 || r->height <= 0 || r->depth < 0 || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 ||
        r->xoffset % g->bw || r->yoffset % g->bh)
        return 0;
    g->level = v && v->set_levels ? v->base_level : 0;
    g->layer0 = v && v->set_layers ? v->min_layer : 0;
    g->nz = r->depth ? r->depth : 1;
    g->is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
    g->wb = (size_t)((r->width + g->bw - 1) / g->bw) * (size_t)g->bpb;
    g->hb = (size_t)(r->height + g->bh - 1) / (size_t)g->bh;
    g->row = row ? row : g->wb;
    g->img = img ? img : g->row * g->hb;
    g->x0 = (size_t)(r->xoffset / g->bw) * (size_t)g->bpb;
    g->y0 = (size_t)(r->yoffset / g->bh);
    return 1;
}

// Pool bytes <-> caller rows of the region alone (nvn_texels_box). MacNativeGames' nvn_mtl.m goes through a
// linear copy of the whole level; for a small region of a large texture that cost tens of milliseconds while the guest
// thread held its virtual core and the caller the texture lock (monsters2 replay, frames 765-790).
static int storage_texels(const tex_t *t, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    for (int z = 0; z < g->nz; z++) {
        unsigned layer = g->is3d ? 0 : g->layer0 + (unsigned)r->zoffset + (unsigned)z;
        uint32_t slice = g->is3d ? (uint32_t)(r->zoffset + z) : 0;
        if (nvn_texels_box(&t->d, g->level, layer, (uint32_t)g->x0, (uint32_t)g->y0, slice, (uint32_t)g->wb,
                           (uint32_t)g->hb, t->mem, mem + (size_t)z * g->img, g->row, !write))
            return 0;
    }
    return 1;
}

static int read_aspect(const Tex *e, VkImageAspectFlags *a) {
    if (!e->f.depth && !e->f.stencil) { *a = VK_IMAGE_ASPECT_COLOR_BIT; return 1; }
    switch (e->t.d.format) {   // packed depth-stencil storage has no single-aspect copy
    case 0x31: *a = VK_IMAGE_ASPECT_STENCIL_BIT; return 1;
    case 0x32: case 0x34: *a = VK_IMAGE_ASPECT_DEPTH_BIT; return 1;
    default: return 0;
    }
}

// Image <-> caller rows through a staging buffer; the GPU copy is waited for.
static int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    static volatile int32_t n_astc, n_ds;
    VkCtx *c = vk_ctx();
    VkImageAspectFlags aspect;
    if (e->f.astc_bw && !write) {
        log_limited(&n_astc, "ReadTexels of ASTC texture %#llx: the image holds decoded RGBA8; zeros returned",
                    (unsigned long long)e->pub.addr);
        return 0;
    }
    if (!read_aspect(e, &aspect)) {
        log_limited(&n_ds, "texels of packed depth-stencil format %#x are not copied", e->t.d.format);
        return 0;
    }
    // CompactLayers: guest mip g->level → host level; trimmed mips have no host image (skip, same as native copies).
    uint32_t host_level = g->level;
    if (e->pub.host_mip0) {
        if (g->level < e->pub.host_mip0) return 0;
        host_level = g->level - e->pub.host_mip0;
    }
    if (host_level >= e->pub.levels) return 0;
    uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height;
    int rgb = e->f.expand_rgb, astc = e->f.astc_bw != 0, agpu = astc && write && vk_astc_gpu();
    size_t texels = (size_t)w * h, ib = agpu ? 0 : astc ? 4 : rgb ? (size_t)g->bpb / 3 * 4 : 0;
    size_t slice = ib ? texels * ib : g->wb * g->hb;   // agpu: the raw blocks, decoded by vk_astc_record
    VkBuffer sb;
    VmaAllocation sa;
    uint8_t *stg = vk_staging(slice * (size_t)g->nz, &sb, &sa);
    VkBufferImageCopy *rg = calloc((size_t)g->nz, sizeof *rg);
    for (int z = 0; z < g->nz; z++) {
        uint8_t *s = stg + slice * (size_t)z;
        const uint8_t *m = mem + (size_t)z * g->img;
        if (write) {
            if (agpu) {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            } else if (astc) {
                uint8_t *tight = malloc(g->wb * g->hb);
                for (size_t y = 0; y < g->hb; y++) memcpy(tight + y * g->wb, m + y * g->row, g->wb);
                uint64_t a0 = perf_begin();
                astc_decode_ldr_mt(tight, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, s);
                perf_end(PERF_ASTC, a0);
                free(tight);
            } else if (rgb) {
                for (size_t y = 0; y < g->hb; y++) expand_rgb(e->t.d.format, m + y * g->row, s + y * w * ib, w);
            } else {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            }
        }
        rg[z] = (VkBufferImageCopy){.bufferOffset = slice * (size_t)z,
                                    .imageSubresource = {aspect, host_level, g->is3d ? 0 : g->layer0 + (uint32_t)r->zoffset + (uint32_t)z, 1},
                                    .imageOffset = {r->xoffset, r->yoffset, g->is3d ? r->zoffset + z : 0},
                                    .imageExtent = {w, h, 1}};
    }
    VkCommandBuffer cb = vk_upload_begin();
    if (write) {
        vmaFlushAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        int done = 0;
        // A BC3 image takes whole 4x4 blocks, or a part that ends on the mip's edge; a region of a 5x5 or 6x6 footprint
        // may end inside a block (vk_astc_record checks the start corner). Host mip size, not guest.
        uint32_t mw = umax(e->pub.width >> host_level, 1), mh = umax(e->pub.height >> host_level, 1);
        int img_bc = e->f.astc_bc && e->pub.format != VK_FORMAT_R8G8B8A8_UNORM &&
                     e->pub.format != VK_FORMAT_R8G8B8A8_SRGB;
        int bc_cut = img_bc && (((w & 3) && (uint32_t)r->xoffset + w != mw) ||
                                ((h & 3) && (uint32_t)r->yoffset + h != mh));
        if (agpu && !bc_cut) {   // bc_cut: nothing recorded, logged below as not decoded
            VkAstcRegion *ar = calloc((size_t)g->nz, sizeof *ar);
            for (int z = 0; z < g->nz; z++)
                ar[z] = (VkAstcRegion){.src = slice * (size_t)z, .w = w, .h = h, .slices = 1, .level = rg[z].imageSubresource.mipLevel,
                                       .layer = rg[z].imageSubresource.baseArrayLayer, .offset = rg[z].imageOffset};
            uint64_t a0 = perf_begin();
            done = vk_astc_record(cb, sb, slice * (size_t)g->nz, ar, g->nz, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, img_bc,
                                  e->pub.image, aspect, VK_NULL_HANDLE);
            perf_end(PERF_ASTC, a0);
            free(ar);
        }
        if (!agpu) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)g->nz, rg);
        else if (!done) log_limited(&n_astc, "WriteTexels of ASTC texture %#llx not decoded on the GPU: left as it was",
                                    (unsigned long long)e->pub.addr);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, sb, (uint32_t)g->nz, rg);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                VK_ACCESS_2_HOST_READ_BIT);
    }
    vk_upload_end();
    vk_upload_flush(1);
    if (!write) {
        vmaInvalidateAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        for (int z = 0; z < g->nz; z++) {
            const uint8_t *s = stg + slice * (size_t)z;
            uint8_t *m = mem + (size_t)z * g->img;
            for (size_t y = 0; y < g->hb; y++) {
                if (rgb) shrink_rgb(e->t.d.format, s + y * w * ib, m + y * g->row, w);
                else memcpy(m + y * g->row, s + y * g->wb, g->wb);
            }
        }
    }
    free(rg);
    vmaDestroyBuffer(c->vma, sb, sa);
    return 1;
}

static void zero_rows(const Region *g, uint8_t *dst) {
    for (int z = 0; z < g->nz; z++)
        for (size_t y = 0; y < g->hb; y++) memset(dst + (size_t)z * g->img + y * g->row, 0, g->wb);
}

int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    Region g;
    if (!t || !region_of(t, v, r, row, img, &g)) return 0;
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
    return ok;
}

// ReadTexels: the pool bytes are the content of a mirrored texture the GPU has not rendered into; every other texture
// is read back from its image.
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g)) {
        log_limited(&n_bad, "ReadTexels: invalid region %d,%d,%d %dx%dx%d (format %#x)", r->xoffset, r->yoffset,
                    r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e && e->pub.mirrored && !e->pub.gpu_owned) ok = storage_texels(t, &g, r, dst, 0);
    else if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
}

// WriteTexels: a mirrored texture gets the texels in its pool bytes (and in its image once the GPU owns it); every
// other texture in its image.
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g) || (v && v->set_format && v->format != t->d.format)) {
        log_limited(&n_bad, "WriteTexels: invalid region %d,%d,%d %dx%dx%d or view format (format %#x)", r->xoffset,
                    r->yoffset, r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    if (e && e->pub.mirrored) {
        if (!storage_texels(t, &g, r, (uint8_t *)src, 1))
            log_limited(&n_bad, "WriteTexels: region outside level %u of texture %#llx", g.level, (unsigned long long)e->pub.addr);
        vk_guest_written((uintptr_t)t->mem, t->L.size);
        if (e->pub.gpu_owned) image_texels(e, &g, r, (uint8_t *)src, 1);
        else if (e->uploaded) e->wt_n++;
    } else if (e) {
        image_texels(e, &g, r, (uint8_t *)src, 1);
    }
    unlock();
}

// ---------------------------------------------------------------- CPU reads of GPU-written storage
// A mirrored texture the GPU rendered into holds its texels in its image only; its storage keeps what the CPU last put
// there. The game may still read that storage with the CPU: TotK moves its generated grass maps between pools with
// memcpy once the GPU is done, and the copy then carried zeros. Such a read writes the image back first.
static void write_back(Tex *e) {
    const nvn_texdesc *d = &e->t.d;
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            // CompactLayers: host level lv is guest mip lv+host_mip0 (image_texels remaps the same way).
            uint32_t guest_lv = lv + e->pub.host_mip0;
            view_t v = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)guest_lv, .min_layer = (uint16_t)ly};
            NVNcopyRegion r = {.width = (int)mip(d->width, guest_lv),
                               .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, guest_lv),
                               .depth = is3d ? (int)mip(d->depth, guest_lv) : 1};
            Region g;
            if (!region_of(&e->t, &v, &r, 0, 0, &g)) continue;
            uint8_t *buf = malloc(g.img * (size_t)g.nz);
            if (!buf) rt_fatal("vk: out of memory for a texture write-back");
            if (image_texels(e, &g, &r, buf, 0)) storage_texels(&e->t, &g, &r, buf, 1);
            free(buf);
        }
    e->gpu_dirty = 0;
}

// Mirrored textures overlapping [addr, addr+size) that the GPU wrote since their last write-back.
static int owned_dirty(uint64_t addr, uint64_t size, Tex **hit) {
    int n = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        Tex *e = owned[i];
        if (__atomic_load_n(&e->gpu_dirty, __ATOMIC_RELAXED) && e->pub.addr + e->t.L.size > addr) {
            if (!hit) return 1;
            hit[n++] = e;
        }
    }
    return n;
}

// A storage image may be written by the shaders of the submission being recorded (texture lock held, vk_exec_draw.c
// tex_info). Its storage is not marked GPU-owned: only the write fence for ReadTexels.
void vk_texture_storage_bound_locked(const VkTexImage *img) {
    Tex *e = (Tex *)img;   // pub is the first member
    uint64_t f = vk_exec_recording_fence();
    if (f > e->wr_fence) e->wr_fence = f;
}

uint64_t vk_texture_last_write(const tex_t *t) {
    lock();
    Tex *e = entry_of(t);
    uint64_t f = e ? e->wr_fence : 0;
    unlock();
    return f;
}

size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img) {
    Region g;
    if (!region_of(t, v, r, row, img, &g)) return 0;
    return (size_t)(g.nz - 1) * g.img + (g.hb - 1) * g.row + g.wb;
}

int vk_texture_cpu_stale(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    plat_mutex_lock_shared(&owned_lock);
    int dirty = owned_dirty(addr, size, NULL);
    plat_mutex_unlock_shared(&owned_lock);
    return dirty;
}

// The newest submission recorded to write a GPU-written mirrored texture overlapping [addr, addr+size); 0: none.
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    lock();
    uint64_t f = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        const Tex *e = owned[i];
        if (!e->gpu_dirty || e->pub.addr + e->t.L.size <= addr) continue;
        if (e->gpu_fence > f) f = e->gpu_fence;
        if (e->wr_fence > f) f = e->wr_fence;
    }
    unlock();
    return f;
}

// The image is the newest content only while nothing wrote the storage since the GPU render: TotK finalizes grass maps,
// reuses their storage for new data and memcpys that data on, and writing the old image back first handed the copy
// a stale map or stale resource bytes (wrong textures in the distance). Every hit is checked before any is written
// back, since a write-back is itself a CPU write to storage another hit may share; the hits then go back oldest render
// first, so where two descriptors share storage the newer render wins.
void vk_texture_write_back(uint64_t addr, uint64_t size) {
    lock();
    Tex **hit = malloc(sizeof *hit * (size_t)(nowned + 1));
    if (!hit) rt_fatal("vk: out of memory for a texture write-back");
    int n = owned_dirty(addr, size, hit), k = 0;   // owned[] cannot change while tex_lock is held
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// TotK's grass system renders a 128x128 RGBA8 map of cut and trampled grass into a CPU-cached pool every frame and its
// CPU code reads the pool bytes without nvnMemoryPoolInvalidateMappedRange (cut reactions: blades, insects, the cut
// itself). On the Switch the bytes are there once the frame's fence has passed. So when the
// CPU waits for a fence, mirrored targets up to wb_limit bytes rendered by work up to that fence go back to their
// storage, unless the CPU rewrote it since. Larger targets still go back on an invalidate, memcpy or ReadTexels.
int vk_texture_small_dirty(void) { return __atomic_load_n(&small_pending, __ATOMIC_ACQUIRE); }

// TOTK_TRACE=gfx: a dirty target the fence write-back leaves alone because it is over the limit (what a CPU read
// after the wait would miss), or sends back although it is over the default limit (what a raised limit changed). One
// line per target address and outcome; the table is small and a full one stops logging.
static void wb_note(const Tex *e, int skipped) {
    static uint64_t seen[2][64];
    static int n_seen[2];
    for (int i = 0; i < n_seen[skipped]; i++)
        if (seen[skipped][i] == e->pub.addr) return;
    if (n_seen[skipped] == 64) return;
    seen[skipped][n_seen[skipped]++] = e->pub.addr;
    const pool_t *p = e->t.pool;
    unsigned fl = p ? p->flags : 0;
    rt_log("vk: fence write-back %s %#llx %ux%u fmt %#x size %llu pool %s (limit %llu)",
           skipped ? "skipped-for-size" : "over-default", (unsigned long long)e->pub.addr, e->t.d.width, e->t.d.height,
           e->t.d.format, (unsigned long long)e->t.L.size,
           !p ? "none" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) ? "cpu-cached" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) ? "cpu-uncached" : "other",
           (unsigned long long)wb_limit);
}

void vk_texture_write_back_small(uint64_t fence) {
    lock();
    wb_cfg_init();
    __atomic_store_n(&small_pending, 0, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&owned_lock);
    int n = 0, later = 0;
    for (int i = 0; i < nowned; i++) n += owned[i]->gpu_dirty && owned[i]->t.L.size <= wb_limit;
    Tex **hit = n ? malloc(sizeof *hit * (size_t)n) : NULL;
    if (n && !hit) rt_fatal("vk: out of memory for a texture write-back");
    n = 0;
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        if (!e->gpu_dirty) continue;
        if (e->t.L.size > wb_limit) {
            if (wb_trace && e->gpu_fence <= fence) {
                wb_note(e, 1);
                perf_count(PERF_CNT_WB_SKIP);
            }
            continue;
        }
        if (e->gpu_fence > fence) later = 1;
        else {
            if (wb_trace && e->t.L.size > WB_SMALL_DEFAULT) wb_note(e, 0);
            hit[n++] = e;
        }
    }
    plat_mutex_unlock_shared(&owned_lock);
    if (later) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    int k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// ---------------------------------------------------------------- samplers (LAS textures.c)
static VkCompareOp compare_op(uint32_t f) {
    switch (f) {
    case NVN_COMPARE_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
    case NVN_COMPARE_FUNC_LESS: return VK_COMPARE_OP_LESS;
    case NVN_COMPARE_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
    case NVN_COMPARE_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case NVN_COMPARE_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
    case NVN_COMPARE_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case NVN_COMPARE_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static struct { int init, mirror_clamp_edge, minmax, border_no_format; } sf;

static void sampler_features(VkCtx *c) {
    if (sf.init) return;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                            .pNext = c->has_custom_border_color ? &cbf : NULL};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    sf.mirror_clamp_edge = f12.samplerMirrorClampToEdge;
    sf.minmax = f12.samplerFilterMinmax;
    sf.border_no_format = c->has_custom_border_color && cbf.customBorderColors && cbf.customBorderColorWithoutFormat;
    sf.init = 1;
}

static VkSampler make_sampler(VkCtx *c, const samp_par *d) {
    sampler_features(c);
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    uint32_t mf = d->minf;
    si.minFilter = mf == 1 || mf == 3 || mf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode modes[3];
    const uint8_t wrap[3] = {d->ws, d->wt, d->wr};
    int border = 0;
    for (int i = 0; i < 3; i++) {
        switch (wrap[i]) {
        case NVN_WRAP_MODE_CLAMP: case NVN_WRAP_MODE_CLAMP_TO_EDGE: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
            modes[i] = sf.mirror_clamp_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            break;
        case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: case NVN_WRAP_MODE_CLAMP_TO_BORDER:
            modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            border = 1;
            break;
        case NVN_WRAP_MODE_MIRRORED_REPEAT: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
    si.addressModeU = modes[0], si.addressModeV = modes[1], si.addressModeW = modes[2];
    si.minLod = d->lod_min;
    si.maxLod = d->lod_max > 1000 ? VK_LOD_CLAMP_NONE : d->lod_max;
    if (mf <= NVN_MIN_FILTER_LINEAR) si.maxLod = si.minLod + 0.25f;   // not mipmapped: stay on the base level
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    float lim = c->props.limits.maxSamplerLodBias;
    si.mipLodBias = fmaxf(-lim, fminf(lim, d->lod_bias));
    // MKW_ANISO=N (Graphics tab, "game" by default): at least N-times anisotropic filtering on mipmapped samplers with
    // linear minification, on top of what the game asks (adriadam10/mkw-recomp vk_tex_slots.c, MIT, Copyright (c) 2026
    // Adrià). Samplers are made once per state, so it takes effect at the next start.
    static float aniso_min = -1;
    if (aniso_min < 0) {
        aniso_min = (float)atof(cfg_value("MKW_ANISO"));   // "game": 0
        if (aniso_min > 1) rt_log("vk: anisotropic filtering at least %gx (MKW_ANISO)", (double)aniso_min);
    }
    float aniso = d->aniso;
    if (mf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST && si.minFilter == VK_FILTER_LINEAR && aniso < aniso_min) aniso = aniso_min;
    if (aniso > 1) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = fminf(aniso, c->props.limits.maxSamplerAnisotropy);
    }
    if (d->cmp_mode) {
        si.compareEnable = VK_TRUE;
        si.compareOp = compare_op(d->cmp_func);
    }
    VkSamplerCustomBorderColorCreateInfoEXT cb = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rm = {.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    const void **tail = &si.pNext;
    if (border) {
        const float *b = d->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (sf.border_no_format) {
            memcpy(cb.customBorderColor.float32, b, 16);
            si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            *tail = &cb;
            tail = &cb.pNext;
        } else si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    if (d->reduction && sf.minmax && !si.compareEnable) {
        rm.reductionMode = d->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        *tail = &rm;
    }
    VkSampler s;
    VkResult r = vkCreateSampler(c->device, &si, NULL, &s);
    if (r != VK_SUCCESS) rt_fatal("vk: vkCreateSampler failed (%d)", (int)r);
    return s;
}

void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    static volatile int32_t n_unbacked;
    if (!pool || id < 0 || id >= (1 << 12) || (uint32_t)id >= pool->count) rt_fatal("vk: sampler slot %d out of pool range", id);
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 1);
    if (!d) {
        log_limited(&n_unbacked, "sampler descriptor slot %d has no mapped backing: not registered", id);
        unlock();
        return;
    }
    descriptor_registered(d, DESC_SAMPLER);
    d->u.sampler.p = *s;
    d->u.sampler.used = 1;
    unlock();
}

// Caller holds the lock.
static VkSampler sampler_for(VkCtx *c, const samp_par *s) {
    samp_par k;
    memset(&k, 0, sizeof k);
    k = *s;
    VkSampler v = (VkSampler)nvn_hm_get(&samplers, &k);
    if (!v) {
        v = make_sampler(c, &k);
        nvn_hm_put(&samplers, &k, (void *)v);
    }
    return v;
}

VkSampler vk_sampler_for(const samp_par *s) {
    VkCtx *c = vk_ctx();
    if (!c || !s) return VK_NULL_HANDLE;
    lock();
    VkSampler v = sampler_for(c, s);
    unlock();
    return v;
}

// Samplers live until shutdown, so a slot keeps the one its parameters map to until it is registered again.
VkSampler vk_sampler_by_id_locked(const dpool_ref *pool, int id) {
    VkCtx *c = vk_ctx();
    if (!c) return VK_NULL_HANDLE;
    VkSampler v = VK_NULL_HANDLE;
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    if (d && d->kind == DESC_SAMPLER) {
        SampSlot *s = &d->u.sampler;
        if (!s->s) s->s = sampler_for(c, &s->p);
        v = s->s;
    }
    return v;
}

VkSampler vk_sampler_point_by_id_locked(const dpool_ref *pool, int id) {
    VkCtx *c = vk_ctx();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    if (!c || !d || d->kind != DESC_SAMPLER) return VK_NULL_HANDLE;
    SampSlot *s = &d->u.sampler;
    if (!s->sp) {
        samp_par k = s->p;
        k.magf = NVN_MAG_FILTER_NEAREST;
        k.minf = k.minf == NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST :
                 k.minf > NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : k.minf;
        k.aniso = 1;
        s->sp = sampler_for(c, &k);
    }
    return s->sp;
}

VkSampler vk_sampler_by_id(const dpool_ref *pool, int id) {
    lock();
    VkSampler v = vk_sampler_by_id_locked(pool, id);
    unlock();
    return v;
}

int vk_sampler_params(const dpool_ref *pool, int id, samp_par *out) {
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    int ok = d && d->kind == DESC_SAMPLER;
    if (ok) *out = d->u.sampler.p;
    unlock();
    return ok;
}

// A backing lifetime ended. Keep imported entries while another live imported alias still maps the same bytes.
// pool_lock is not held by the caller; taking it through descriptor_key while holding tex_lock preserves lock order.
void vk_texture_descriptors_retire(uint64_t addr, uint64_t size, uint64_t storage) {
    lock();
    VkDescKey *gone = malloc(sizeof *gone * (size_t)(descriptors.n + 1));
    if (!gone) rt_fatal("vk: out of memory retiring descriptors");
    uint32_t n = 0;
    for (uint32_t i = 0; i < descriptors.cap; i++) {
        if (!descriptors.vals[i]) continue;
        VkDescKey key;
        memcpy(&key, descriptors.keys + (size_t)i * descriptors.klen, sizeof key);
        if (key.storage != storage) continue;
        if (!storage) {
            if (key.offset >= addr ? key.offset - addr >= size : addr - key.offset >= NVN_DESCRIPTOR_BYTES) continue;
            VkDescKey live;
            if (vk_mem_descriptor_key(key.offset, NVN_DESCRIPTOR_BYTES, &live) && !memcmp(&live, &key, sizeof key)) continue;
        }
        gone[n++] = key;
    }
    for (uint32_t i = 0; i < n; i++) free(nvn_hm_del(&descriptors, &gone[i]));
    if (n) slots_gen++, samp_gen++;
    free(gone);
    unlock();
}

// ---------------------------------------------------------------- statistics and shutdown
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads) {
    lock();
    if (images) *images = by_key.n;
    if (uploads) *uploads = n_uploads;
    if (reuploads) *reuploads = n_reuploads;
    unlock();
}

void vk_texture_census(VkTexCensus *s) {
    VkCtx *c = vk_ctx();
    memset(s, 0, sizeof *s);
    lock();
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
        }
        if (e->released) s->released++, s->own_released += e->alloc != NULL, s->bytes_released += bytes;
        else s->live++, s->own_live += e->alloc != NULL, s->bytes_live += bytes;
    }
    unlock();
}

void vk_texture_free_for_vram(uint64_t need) {
    lock();
    uint64_t used, budget;
    int exact;
    vk_mem_collect();
    for (int pass = 0; pass < 64 && rel_head; pass++) {
        if (!vk_mem_vram(&used, &budget, &exact) || used + need <= budget) break;
        int dropped = 0;
        for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP; e = next) {
            next = e->rel_next;
            if (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty) continue;
            destroy_tex(e);
            dropped++, n_pressure_drops++;
        }
        if (!dropped) break;
        vk_upload_flush(1);
        vk_mem_collect();
    }
    unlock();
}

int vk_texture_astc_copy(const tex_t *src, uint32_t level, uint32_t layer, uint32_t layers, uint32_t x, uint32_t y,
                         uint32_t w, uint32_t h, const VkTexImage *dst, VkImageAspectFlags aspect, uint32_t dlevel,
                         uint32_t dlayer, int32_t dx, int32_t dy, int bc) {
    TexVkFormat f;
    if (!src || !dst || !tex_vk_format(src->d.format, &f) || !f.astc_bw || !src->mem ||
        x % f.astc_bw || y % f.astc_bh || (bc && (dx % 4 || dy % 4)))
        return 0;
    uint32_t rb = (w + f.astc_bw - 1) / f.astc_bw * 16, rows = (h + f.astc_bh - 1) / f.astc_bh;
    size_t per = (size_t)rb * rows, total = per * layers;
    if (!total) return 0;
    VkAstcRegion *ar = calloc(layers, sizeof *ar);
    if (!ar) return 0;
    VkCommandBuffer cb = vk_upload_begin();
    VkBuffer sb = VK_NULL_HANDLE;
    uint8_t *stg = vk_staging(total, &sb, NULL);
    int ok = stg != NULL;
    uint32_t dlw = dst->width >> dlevel ? dst->width >> dlevel : 1;
    uint32_t dlh = dst->height >> dlevel ? dst->height >> dlevel : 1;
    for (uint32_t l = 0; l < layers && ok; l++) {
        ok = !nvn_texels_box(&src->d, level, layer + l, x / f.astc_bw * 16, y / f.astc_bh, 0, rb, rows, src->mem,
                             stg + l * per, rb, 1);
        ar[l] = (VkAstcRegion){.src = l * per, .w = w, .h = h, .slices = 1, .level = dlevel, .layer = dlayer + l,
                               .lw = dlw, .lh = dlh, .offset = {dx, dy, 0}};
    }
    if (ok) {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        ok = vk_astc_record(cb, sb, total, ar, (int)layers, f.astc_bw, f.astc_bh, f.astc_srgb, bc, dst->image, aspect,
                            VK_NULL_HANDLE);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
    vk_upload_end();
    if (total > (64ull << 20)) vk_upload_flush(1);
    free(ar);
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) <= 8)
        rt_log("vk: ASTC %ux%u (fmt %#x) level %u layer %u x%u -> %s level %u layer %u at %d,%d: %s", w, h,
               src->d.format, level, layer, layers, bc ? "BC" : "RGBA8", dlevel, dlayer, dx, dy,
               ok ? "decoded" : "failed");
    return ok;
}

void vk_texture_shutdown(void) {
    VkCtx *c = vk_ctx();
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) list[n++] = by_key.vals[i];
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    for (uint32_t i = 0; i < samplers.cap; i++)
        if (samplers.vals[i] && c) vkDestroySampler(c->device, (VkSampler)samplers.vals[i], NULL);
    free(by_key.keys), free(by_key.vals), free(by_addr.keys), free(by_addr.vals), free(samplers.keys), free(samplers.vals);
    by_key = (nvn_hmap){.klen = sizeof(TexKey)};
    by_addr = (nvn_hmap){.klen = 8};
    samplers = (nvn_hmap){.klen = sizeof(samp_par)};
    for (uint32_t i = 0; i < descriptors.cap; i++) free(descriptors.vals[i]);
    free(descriptors.keys), free(descriptors.vals);
    descriptors = (nvn_hmap){.klen = sizeof(VkDescKey)};
    nreleased = 0;
    rel_head = rel_tail = NULL;
    unlock();
}

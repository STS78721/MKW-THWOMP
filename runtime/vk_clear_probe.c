// --probe-vkclear: original synthetic depth/stencil image, production masked-clear helper, GPU readback.
// No guest program, shader cache, screenshots or census is required.
#include <stdio.h>
#include <string.h>
#include "host.h"
#include "vk.h"

enum { CW = 8, CH = 6, CL = 3, PIXELS = CW * CH * CL, STENCIL_OFFSET = PIXELS * 4 };

static void probe_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static int clear_case(VkSampleCountFlagBits samples, unsigned *checks) {
    VkCtx *c = vk_ctx();
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation ia = NULL, ba = NULL;
    int bad = 1;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                             .format = VK_FORMAT_D32_SFLOAT_S8_UINT, .extent = {CW, CH, 1}, .mipLevels = 1,
                             .arrayLayers = CL, .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                             .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &image, &ia, NULL) != VK_SUCCESS) goto done;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ii.format,
                                 .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 1, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &view) != VK_SUCCESS) goto done;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PIXELS * 5,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    ai = (VmaAllocationCreateInfo){.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo mapped;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buffer, &ba, &mapped) != VK_SUCCESS) goto done;
    VkCommandBuffer cb = vk_upload_begin();
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, CL};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = image, .subresourceRange = range,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    VkClearDepthStencilValue initial = {.depth = 0.375f, .stencil = 0xa6};
    vkCmdClearDepthStencilImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &initial, 1, &range);
    probe_barrier(cb);
    vk_clear_stencil_masked(cb, ii.format, samples, range.aspectMask, view, CW, CH,
                             (VkRect2D){{2, 1}, {3, 4}}, 0x15, 0x11);
    probe_barrier(cb);
    // A second masked clear must preserve the first clear's other bits, including in their overlap.
    vk_clear_stencil_masked(cb, ii.format, samples, range.aspectMask, view, CW, CH,
                             (VkRect2D){{3, 2}, {4, 2}}, 0x42, 0x40);
    probe_barrier(cb);
    if (samples == VK_SAMPLE_COUNT_1_BIT) {
        VkBufferImageCopy copy[2] = {
            {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, CL}, .imageExtent = {CW, CH, 1}},
            {.bufferOffset = STENCIL_OFFSET, .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, CL},
             .imageExtent = {CW, CH, 1}},
        };
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 2, copy);
        probe_barrier(cb);
    }
    vk_upload_end();
    vk_upload_flush(1);
    if (samples == VK_SAMPLE_COUNT_1_BIT) {
        vmaInvalidateAllocation(c->vma, ba, 0, VK_WHOLE_SIZE);
        const float *depth = mapped.pMappedData;
        const uint8_t *stencil = (const uint8_t *)mapped.pMappedData + STENCIL_OFFSET;
        for (int layer = 0; layer < CL; layer++)
            for (int y = 0; y < CH; y++)
                for (int x = 0; x < CW; x++) {
                    int p = (layer * CH + y) * CW + x;
                    uint8_t want = 0xa6;
                    if (layer == 1 && x >= 2 && x < 5 && y >= 1 && y < 5) want = (want & ~0x15) | 0x11;
                    if (layer == 1 && x >= 3 && x < 7 && y >= 2 && y < 4) want = (want & ~0x42) | 0x40;
                    if (depth[p] != 0.375f || stencil[p] != want) {
                        printf("probe-vkclear: FAIL %d,%d layer %d depth %g stencil %#x expected %#x\n",
                               x, y, layer, depth[p], stencil[p], want);
                        goto done;
                    }
                    *checks += 2;
                }
    } else {
        // Multisampled images cannot be copied to a buffer. This case validates the actual all-sample draw path;
        // exact stencil/depth readback above covers single-sample images only.
        (*checks)++;
    }
    bad = 0;
done:
    if (view) vkDestroyImageView(c->device, view, NULL);
    if (image) vmaDestroyImage(c->vma, image, ia);
    if (buffer) vmaDestroyBuffer(c->vma, buffer, ba);
    return bad;
}

int probe_vkclear(void) {
    if (!vk_init()) return 1;
    unsigned checks = 0;
    int bad = clear_case(VK_SAMPLE_COUNT_1_BIT, &checks);
    VkImageFormatProperties format;
    VkResult supported = vkGetPhysicalDeviceImageFormatProperties(vk_ctx()->phys, VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        0, &format);
    if (!bad && supported == VK_SUCCESS && (format.sampleCounts & VK_SAMPLE_COUNT_4_BIT))
        bad = clear_case(VK_SAMPLE_COUNT_4_BIT, &checks);
    vk_mem_shutdown();
    vk_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-vkclear: %s (%u checks, %ld validation errors; multisample draw has no readback)\n",
           bad || errors ? "FAILED" : "passed", checks, errors);
    return bad || errors;
}

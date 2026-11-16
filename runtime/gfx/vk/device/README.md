# runtime/gfx/vk/device

Device and submits of the Vulkan back end: instance and capabilities, VMA, command recording on its own thread,
submits and fences, present.

| File | What it is |
|---|---|
| `vk.h`, `vk_backend.c` | Internal header and back end: ties memory, textures, submits and present together |
| `vk_device.c`, `vk_caps.c`, `vk_vma.cpp` | Instance, device, VMA (the only C++), and what the runtime decides from the GPU |
| `vk_defaults.h`, `vk_defaults.c` | Memory caps with nothing set, from the player's GPU and RAM (`MKW_SPARSE_MIB`, `MKW_GPU_MEM_LIMIT` on an integrated GPU, `MKW_VK_TEX_POOL_MIN`); pure, checked by `tests/test_defaults.py` |
| `vk_destroy_count.h` | Routes every buffer, view or sampler destruction through `vk_destroy_note`: the draw cache is flushed when a stored set named that handle |
| `vk_rec.h`, `vk_rec.c`, `vk_rec_cmds.c`, `vk_rec_int.h` | Recorder thread: the replay's `vkCmd*` calls are made by another thread in the same order; `vk_rec.c` holds the ring, the thread and the fixed-size commands, `vk_rec_cmds.c` the ones with arrays and the extension ones, and the internal header what they share |
| `vk_rec_reuse.h`, `vk_rec_reuse.c` | `MKW_REUSE`: the command buffer's bound state by slots, capture of a draw's packets and their re-emission to the ring |
| `vk_rec_diff.h`, `vk_rec_diff.c` | `MKW_REC_DIFF=1` (measurement): how much of each submit's resolved commands repeats an earlier one, per packet and per draw span |
| `vk_sync.c` | Submits, fences, timeline semaphore and completion thread |
| `vk_sync_ts.c`, `vk_sync_ts.h` | GPU time of each part of a submit (`MKW_TRACE=perf`), for `vk_sync.c` |
| `vk_fsr.c` | AMD FSR 1 at present time when the window is larger than the game's output |
| `vk_color.c` | THWOMP: the colour profiles (brightness, contrast, saturation, gamma) on the present, before FSR or the blit (from STS78721's DKB build) |
| `vk_hdr_ui.c` | The F1 menu on an HDR10 swapchain: drawn into an RGBA8 image and blended onto the game's PQ picture at the system's paper white (`runtime/shaders/hdr_ui.frag`; from totk-pc-recomp, MIT) |
| `vk_reflex.c` | THWOMP: NVIDIA Reflex (`MKW_REFLEX` off/on/boost) through `VK_NV_low_latency2`: the sleep before each game frame, the latency markers and present ids; NVIDIA GPUs only |
| `vk_mv.c` | THWOMP: the inputs DLSS needs that the game never makes: the camera's motion from its camera block (this and the last frame's view-projection), each object's own motion (its depth pre-pass draws captured again by transform feedback and paired with last frame's), the camera jitter (Halton 2,3 in the camera block's copy), the picture before the HUD, and the bias-current-colour hint; `MKW_MV_DEBUG` views |
| `vk_dlss.c`, `sl_shim.cpp`, `sl_shim.h` | THWOMP: DLSS super resolution (`MKW_UPSCALER=dlss`, `MKW_DLSS_MODEL`) through NVIDIA Streamline with manual hooking: Streamline's extensions at device creation, its swapchain proxies, the evaluate of the game's finished picture (HUD included) and its encoding for the present; `sl_shim.cpp` is the Streamline API as plain C (its headers in `runtime/third_party/streamline`) |
| `vk_framegen.c` | Frame generation (`MKW_FRAMEGEN`): frames made between the game's from its last two (optical flow), on the present thread and the GPU; the present thread shows them evenly over the game's frame time |
| `vk_guard.c` | `MKW_GPU_GUARD=1`: stops with `rt_fatal` before submitting what would fault the GPU (an object destroyed or queued for destruction before the last part that names it, a bind skipped by a cache the command buffer does not have, a pushed buffer address of 0); `MKW_GPU_GUARD_TEST=bind\|free` commits one on purpose |

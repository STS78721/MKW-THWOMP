# runtime/shaders/mem

Compute shaders for the optional GPU byte-merged VRAM mirror write-back path (`runtime/gfx/vk/mem/vk_mem_mmerge.c`, `MKW_MIRROR_MERGE`).

| File | What it does |
|---|---|
| `mirror_base.comp` | Copies each writable range's pre-submission VRAM mirror words into a base chunk before the GPU work runs |
| `mirror_merge.comp` | Compares the post-submission VRAM mirror against the base snapshot and writes back only the words/edge bytes modified by the GPU |

# runtime/shaders

The runtime's own shaders: a compute ASTC decoder (`astc_decode.comp`), compaction of the mesh groups of task/mesh as
compute (`mesh_compact.comp`), a mode 6 BC7 encoder (`bc7_encode.comp`: recompresses the decoded ASTC to 1 byte per
texel instead of 4), a BC1 encoder (`bc1_encode.comp`: half a byte per texel, for ASTC without alpha), `solid.frag`
(diagnostic `MKW_FS_SOLID=<hashes>|all`: replaces those fragment shaders, or all of them, with magenta), AMD FSR 1
(`fsr_easu.comp` scales, `fsr_rcas.comp` sharpens; AMD's headers in `ffx/`, MIT) and frame generation
(`gfx/vk/device/vk_framegen.c`: `fg_luma.comp` the luma pyramid, `fg_flow.comp` the block matching of each level,
`fg_median.comp` the vectors' median, `fg_synth.comp` a made frame) and the F1 menu on an HDR10 display
(`hdr_fullscreen.vert` + `hdr_ui.frag`: the menu's sRGB image to PQ at paper white, `gfx/vk/device/vk_hdr_ui.c`; from
totk-pc-recomp, MIT) and THWOMP's colour profiles (`color_grade.comp`: brightness, contrast, saturation and gamma
on the present, `gfx/vk/device/vk_color.c`, from STS78721's DKB build). Compiled into `gfx/vk/*_spv.h`; a
new or changed one is regenerated with `uv run tools/shaders/embed_spv.py runtime/shaders/<name>.comp`.

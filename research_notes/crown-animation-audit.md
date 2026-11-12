# Crown City animation audit — 2026-11-13 (Codex)

## Result

No confirmed game fix yet. The broken CPU-produced bone matrices remain the strongest local evidence. This audit
corrected the diagnostic's attribution, checked decoder operations offline, and reviewed other GitHub projects.
No game was launched, no settings or generated game code were changed, and `build/mkw/mkw.exe` was not rebuilt.

## Corrections to HANDOVER section 12a

- The current corrected matrix probe has not run: `logs/mkw.log` ends at 21:55:18 on November 12, before the
  executable rebuilt at 21:56:05. The millions of bad quaternion reports belong to the earlier invalid probe.
- The existing log actually reaches **8,388,608 sin/cos calls, zero mismatches**.
- `hdr+0x28` is an offset added to an external source base. Zero there is valid. In generated `main_0049.c`,
  lines 31336–31340 obtain that base through `hdr+0x40 -> +0x10 -> +0x100`; lines 31344–31350 calculate the sample:

  ```text
  source_base + data_offset + uint32(frame_index * stride + bone_index * component_stride)
  ```

  The fallback uses source_base=0. Component stride is the byte at `hdr+0x129`. The arithmetic wraps at 32 bits
  before the base/offset additions, as in the original ARM code. Log the actual computed guest address in any
  next sample hook rather than interpreting the offset as a pointer.
- `t=972` is the sum 17+955 in the old sample. Neither that number nor the empirical observed-good time range
  establishes the clip's valid bounds. Obtain the clip's actual frame range and wrapping behavior first.
- The matrix hook at generated line 31615 covers a quaternion store join. It **does not cover every format**.
  A blend branches at 31315 to 32964 before `rt_anim_t` at 31330, then reaches the matrix hook through 33110.
  Thus its old TLS time/header and scratch quaternion can describe a previous sample. Euler paths at 33466,
  34592–34606 and 34826–34887 bypass that matrix hook entirely.

## Diagnostic source change

`runtime/ov_system.c` now checks the saved source-list count, animation stage and bone entry before associating
TLS time/header or stack quaternion values with a matrix. Blends and unmatched contexts report matrix-only
evidence, with a separate bounded report budget. They do not read the scratch quaternion. Single-source
reports label `data_offset` correctly, include the externally resolved `source_base`, and describe the observed
good-time range as observations rather than bounds. The summary explicitly states Euler paths are not sampled.
The disabled time hook returns early.

This changes diagnostic output only. It does not normalize quaternions, suppress meshes, or alter animation
results. It also does not add the missing Euler/sample-address hooks. The running executable still has the
previous probe until an intentional incremental rebuild.

Validation: the complete `ov_system.c` passes Clang syntax checking with the build's runtime flags. A CPU-only
harness built from the actual probe source passes four cases: a zero data offset with an external base, stale
blend metadata, a sample from the wrong stage, and a sample from another bone. The last three use an invalid
scratch pointer to prove it is not accessed. Rerun with `py -3 build/test_anim_probe.py`.

## Offline decoder checks

`tools/check_anim_decode.py` executes selected original ARM blocks from the user's supplied ExeFS with Unicorn.
It compares those results with models of the lifted operations; it neither starts the game nor writes the dump.
Run from this repository: `py -3 tools/check_anim_decode.py --exefs "C:/Games/MKW PC/exefs"`.

- Half conversion: all 65,536 input patterns match the lifted semantics. All 63,488 finite IEEE binary16 values
  also match Python's independent half conversion exactly. The game's software converter treats exponent 31
  differently from IEEE Inf/NaN; that behavior exists in the original ARM code and is not a lifter defect.
- Packed extraction: 50,688 cases match, including shift positions, widths, pointer advancement and word values;
  25,344 additionally match an independent MSB bitfield model.
- Time conversion/floor sequence: 20,023 edge/random inputs match, including NaNs, infinities and saturation edges.
- All 221 scalar arithmetic/FMA instructions in `f_700042bea0` match the lifter's operation and operand order.

These results narrow the investigation. The comparisons use instruction-level models, not execution of the
entire compiled animation function, and do not establish that its live input buffers, clip bounds or ownership
are correct. FPCR is zero, and the time block adds +0 before conversion; other rounding/flush modes and
arbitrary time additions were not tested. They are not a proof that every CPU helper is correct.

## GitHub leads checked

1. [nxnative #1186](https://github.com/seedds/nxnative/issues/1186) fixed similar blue-grey panels on Crown City,
   including a reported 1.8.0 confirmation. Its task/mesh storage tables had been at c0+0x610/0x710 instead of
   **0x670/0x770**, corrupting a displacement array. Those correct offsets are already present locally in
   `runtime/vk_exec_draw.c` at 1726–1727 and 1904. Its implicated procedural shader is different from the locally
   identified pedestrian skinning shader. **Do not reapply it as this bug's fix.**
2. [adriadam10 PR #172](https://github.com/adriadam10/mkw-recomp/pull/172), merged November 12, adds MKW 1.8.0
   support. Its 1.8.0 patch set, `recomp/patches-mkw-180.txt`, already includes the version-specific ZSTD hooks;
   the PR says they were found again in the 1.8.0 executable. This confirms upstream has the hooks. Their exact sites
   still need comparison with this checkout's dump before porting.
3. **Those upstream hooks are not connected to this local fork/build.** Local `recomp/patches-mkw180.txt` contains
   only comments, and the generated `C:\Games\MKW PC\generated\mkw180\patches.h` says `RT_PATCH_COUNT 0`.
   `runtime/ov_zstd.c` still has TotK 1.4.3 guest addresses `0x700003a1f0`/`0x700003a1e0`. Enabling the local ZSTD
   comparison option therefore does not test MKW 1.8.0's decoder through this override. No `zstd:` counters appear
   in the latest log. This is an integration gap in this fork, not evidence that upstream lacks hooks or that
   decompression is faulty.
4. [ns2-assets' FSKA notes](https://github.com/dougchansan/ns2-assets/blob/main/docs/ANIM-FSKA.md) and
   [indexer](https://github.com/dougchansan/ns2-assets/blob/main/tools/bfres10_anim.py) provide clip/skeleton
   structures and frame counts. The indexer does not decode quaternion samples, and the notes leave curve
   sampling unresolved. It can help identify bounds, not serve as a complete independent decoder.

No published fix specifically for the local broken pedestrian matrices was found in the checked sources.

## Next evidence needed

At the first bad matrix, capture the **current path**, actual frame index, exact computed sample address,
descriptor bytes and input bytes **before** they are consumed. Include the animation resource identity, clip
bounds, loop/wrap state, bone index, and both source contexts for a blend. Cover the separate Euler stores too.
Keep the first bad output and its inputs together so it can be replayed offline against the original ARM.

If the packed bytes are already wrong, compare them with the same range in the decompressed asset. Validate
the verified MKW 1.8.0 decoder sites before using the ZSTD comparison knob. Do not clamp time, normalize broken
quaternions, hide draws, change memory mode, or ask for another flickering visual run based only on the old logs.

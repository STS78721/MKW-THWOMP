# Eglantine Ampere lowering (GPL-3.0-or-later)

`shader_recompiler/frontend/ampere/lower_to_maxwell.h` and `program_capture.h` originate from
[Franseven/PokeDinner-a-Eglantine-Fork](https://git.eden-emu.dev/Franseven/PokeDinner-a-Eglantine-Fork), a fork of
Eglantine (an Eden-derived Switch 2 emulator), at the commit in `UPSTREAM_COMMIT`. Copyright 2026 Eden Emulator
Project, GPL-3.0-or-later; their encoding reference is Mesa NAK's sm70_encode.rs (MIT, Copyright 2022 Collabora,
Ltd.).

They rewrite Switch 2 (Ampere, SM86) shader code as Maxwell instructions, which the shader translator this project
already uses can take. `common/common_types.h` is ours: the fixed-width type names the headers expect.

Local extension: `lower_to_maxwell.h` accepts the R G B component mask for 2D and array-2D TXD. It routes the third
component to the second destination without writing a fourth component. `tests/native_txd_mask_test.cpp` checks
register routing and predicates with owned instructions; it does not verify hardware texture filtering.

Because these files are GPL-3.0-or-later, DKB Recomp as a whole is distributed under GPL-3.0-or-later (see the
repository's LICENSE).

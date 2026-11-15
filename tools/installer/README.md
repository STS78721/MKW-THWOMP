# tools/installer

THWOMP's Windows installer: one `.exe` in the start screen's dark theme with the player's folder inside, and its
uninstaller. Release assets in STS78721's naming come out of the same script.

| File | What it is |
|---|---|
| `installer.cpp` | The setup program (SDL3 linked statically, Dear ImGui, zstd): Welcome, install folder and shortcuts, the copy, the end; as `uninstall.exe` (or `--uninstall`) it removes the program, its shortcuts and its entry in Windows' apps, and the game data and saves only if the player ticks it. Test hooks at the top of the file |
| `build.py` | Builds SDL3 statically once, compiles `installer.cpp` and appends `build/dist/windows` compressed with zstd: `build/installer/MKW-THWOMP-Setup.exe`; `--version X` also writes `SteelTemple-MKW-THWOMP-X-Full-Setup-Wizard.exe`, `--zips` the `-Portable-StandAlone.zip` and `-Update.zip` |

From the repo root, after `tools/build/windeps.py`, `tools/build/build.py --dist --runtime-only --target windows` and
`tools/build/dist.py --target windows`:

```sh
uv run tools/installer/build.py --version 0.1.1 --zips
```

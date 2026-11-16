// Adapted from MacNativeGames games/totk/runtime/fs_host.h @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// The nn::fs host layer: the read-only RomFS (romfs.c) and writable save, temporary and
// cache storage under the profile directory (fs_host.c). Both register their nn::fs overrides with HOST_OVERRIDE;
// this header is what the two files and the probe share.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

// ---- nn::fs results
#define FS_R_PATH_NOT_FOUND 0x202u
#define FS_R_ALREADY_EXISTS 0x402u
#define FS_R_NOT_EMPTY 0x1002u
#define FS_R_UNSUPPORTED_OPEN 0x6402u    // RomFS: any open mode but read
#define FS_R_INVALID 0xe402u             // RomFS: bad handle, offset or argument
#define FS_R_TARGET_NOT_FOUND 0x7d402u
#define FS_R_INVALID_ARGUMENT 0x2ee202u
#define FS_R_INVALID_PATH 0x2ee602u
#define FS_R_OPEN_MODE 0x307402u
#define FS_R_APPEND_REQUIRED 0x2f5e02u

enum { FS_OPEN_READ = 1, FS_OPEN_WRITE = 2, FS_OPEN_APPEND = 4 };   // nn::fs::OpenMode
enum { FS_ENTRY_DIRECTORY = 0, FS_ENTRY_FILE = 1 };                 // nn::fs::DirectoryEntryType

// ---- romfs.c
// Handles 0x1_0000_0000 + j; host storage uses 0x2_0000_0000 + j (files) and 0x3_0000_0000 + j (directories).
#define ROMFS_HANDLE_BASE 0x100000000ull
int romfs_owns(const char *path);                          // path is on the mounted RomFS ("<mount>:...")
int romfs_handle(uint64_t h);                              // h is a RomFS file handle
uint32_t romfs_mount(const char *name);                    // nn::fs::MountRom; a second name is fatal
uint32_t romfs_open(uint64_t *out, const char *path, int mode);
uint32_t romfs_read(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size);
uint32_t romfs_size(int64_t *out, uint64_t h);
void romfs_close(uint64_t h);
uint32_t romfs_entry_type(int *out, const char *path);
// The entries right under a RomFS folder, in name order: add(name, is_dir, size, ctx) for each (a nonzero return
// stops). FS_R_PATH_NOT_FOUND when nothing lies under it.
uint32_t romfs_list(const char *path, int (*add)(const char *name, int is_dir, uint64_t size, void *ctx), void *ctx);
uint64_t romfs_file_count(void);                           // files in the index (fs_init first)
int romfs_has_language(const char *lang);                  // the index has Mals/<region><lang>/ ("pl", "th"): 1.5.0 and 1.7.0
// ---- romfs_res.c
int romfs_render_lines(void);             // the lines the game renders at (MKW_RESOLUTION or DLSS quality), 0 its own
int romfs_dlss_lines(const char *mode);   // the lines for MKW_DLSS_QUALITY `mode` at this window, 0 unknown
const char *romfs_resolution_overlay(void);   // MKW_RESOLUTION's overlay, NULL for the game's own
void romfs_resolution_external(void);         // the overlay came from MKW_ROMFS_PATCHES instead
int romfs_resolution_lines(void);             // lines applied at start: 0 the game's own, -1 MKW_ROMFS_PATCHES's
const char *romfs_entry(uint64_t i, uint64_t *offset, uint64_t *size);   // path, image offset and size of index row i
// The host file holding row i's bytes (the row's own file under the loose RomFS, or the .nup package), overlays aside;
// NULL when buf is too small or the row is stored compressed in the package
const char *romfs_source_path(uint64_t i, char *buf, size_t n);
// A whole RomFS file by its path (no overlays), malloc'd; NULL when missing or unreadable. Inside fs_init too.
uint8_t *romfs_read_whole(const char *path, uint64_t *size);
uint8_t *romfs_read_top(const char *path, uint64_t *size);   // through the overlays loaded so far (overlay_init)
int romfs_original_sha(const char *path, char hex[65]);      // the RomFS's own file's SHA-256; 0 when it has none
size_t romfs_overlay_count(void);
const char *romfs_overlay(size_t k, const char **file, uint64_t *size, int *is_added);   // RomFS path of overlay k
void fs_trace(const char *what, const char *path);         // MKW_TRACE=fs: one line per open

// ---- fs_host.c
void fs_host_set_profile(const char *dir);                 // profile directory; before the first storage mount only
                                                           // (default rt_config()->profile)
int fs_save_busy(void);                                    // a save is being written: a save file open for writing,
                                                           // or save data changed in the last 3 s and not committed
void fs_save_flush(void);                                  // wait for a save Commit still running in the background;
                                                           // every exit path calls it before the process ends

// --probe-shader-geometry: original Maxwell GS -> real translator child -> runtime reflection/cache.
// The launcher supplies an empty owned cache and the translator. No GPU, census, game data or cache deletion.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "platform/platform.h"

uint64_t gfx_shader_requests(void);
uint64_t gfx_shader_cache_hits(void);
void gfx_shader_drop_memory(void);

enum { TOPOLOGIES = 5, CONTROL_BYTES = 0x880, CODE_SKIP = 0x30, PROGRAM_BYTES = 0x30 + 0x50 + 5 * 32 };
#define CHECK(cond, ...) do { if (!(cond)) { printf("probe-shader-geometry: FAILED: " __VA_ARGS__); puts(""); return 1; } } while (0)

static void word32(uint8_t *p, uint32_t value) { memcpy(p, &value, sizeof value); }
static void word64(uint8_t *p, uint64_t value) { memcpy(p, &value, sizeof value); }

static uint64_t movi(unsigned reg, uint32_t value) {
    return 0x0100000000000000ull | ((uint64_t)value << 20) | (15ull << 12) | (7ull << 16) | reg;
}

static uint64_t ast(unsigned reg, unsigned address, unsigned count) {
    return 0xeff0000000000000ull | (255ull << 39) | ((uint64_t)(count - 1) << 47) |
           ((uint64_t)address << 20) | (7ull << 16) | (255ull << 8) | reg;
}

// Same original authoring pattern as tests/geometry_shader_fixture.py::program(store=False): write Position and
// a red color, emit/cut one point, exit. Every primitive-input topology is accepted; no input vertex is fetched.
static void original_program(uint8_t control[CONTROL_BYTES], uint8_t code[PROGRAM_BYTES]) {
    memset(control, 0, CONTROL_BYTES), memset(code, 0, PROGRAM_BYTES);
    word32(code + CODE_SKIP, 1u | (3u << 5) | (4u << 10));  // Maxwell SPH v3, ordinary GS
    word32(code + CODE_SKIP + 8, 1u << 24);                // one invocation
    word32(code + CODE_SKIP + 12, 1u << 24);               // output points
    word32(code + CODE_SKIP + 16, 1);                     // one output vertex
    uint64_t insn[13];
    for (unsigned i = 0; i < 8; i++) insn[i] = movi(i, i == 3 || i == 4 || i == 7 ? 0x3f800000u : 0);
    insn[8] = ast(0, 0x70, 4), insn[9] = ast(3, 0x6c, 1), insn[10] = ast(4, 0x80, 4);
    insn[11] = 0xf6e0000000000000ull | (3ull << 39) | (7ull << 16) | (255ull << 8) | 255;
    insn[12] = 0xe30000000000000full | (7ull << 16);
    for (unsigned group = 0; group < 5; group++) {
        uint8_t *p = code + CODE_SKIP + 0x50 + group * 32;
        word64(p, 0x001f8000fc0007e0ull);
        for (unsigned lane = 0; lane < 3; lane++) {
            unsigned i = group * 3 + lane;
            word64(p + 8 + lane * 8, i < 13 ? insn[i] : 0);
        }
    }
    word32(control + 0x700, PROGRAM_BYTES), word32(control + 0x704, PROGRAM_BYTES);
    word32(control + 0x714, 2);                            // NVN geometry stage, distinct from GFX_STAGE_GS
}

static int cache_entry(const char *name, int kind, uint64_t size, void *context) {
    (void)name; (void)kind; (void)size;
    *(int *)context = 1;
    return 0;
}

static int shader_valid(const GfxShader *s, unsigned input, const uint8_t *key, const uint8_t *iid) {
    static const unsigned vertices[TOPOLOGIES] = {1, 2, 4, 3, 6};
    if (!s || gfx_shader_stage(s) != GFX_STAGE_GS) return 0;
    const GfxShaderGeometry *g = gfx_shader_geometry(s);
    if (!g || g->version != GFX_GEOMETRY_VERSION || g->input != input || g->input_vertices != vertices[input - 1] ||
        g->output != GFX_GEOMETRY_OUTPUT_POINTS || g->invocations != 1 || g->max_output_vertices != 1 || g->passthrough ||
        gfx_shader_c0_layout(s)) return 0;
    uint64_t words = 0;
    const uint32_t *spv = gfx_shader_spirv(s, &words);
    return spv && words >= 5 && spv[0] == 0x07230203u &&
           (!key || !memcmp(key, gfx_shader_key(s), 32)) && (!iid || !memcmp(iid, gfx_shader_input_id(s), 32));
}

static int stored_input_valid(const char *cache, const uint8_t iid[32], unsigned input,
                              const uint8_t control[CONTROL_BYTES], const uint8_t code[PROGRAM_BYTES]) {
    char hex[65], path[PLAT_PATH_MAX * 2];
    for (unsigned i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", iid[i]);
    int n = snprintf(path, sizeof path, "%s/in/%s.in", cache, hex);
    if (n < 0 || (size_t)n >= sizeof path) return 0;
    uint64_t bytes = 0;
    uint8_t *file = plat_file_read(path, &bytes);
    uint32_t head[6];
    int ok = file && bytes == sizeof head + 148 + CONTROL_BYTES + PROGRAM_BYTES;
    if (ok) {
        memcpy(head, file, sizeof head);
        uint8_t answers[148] = {0};
        answers[147] = (uint8_t)input;
        ok = head[0] == 0x49534b54u && head[1] == 2 && head[2] == GFX_STAGE_GS && head[3] == PROGRAM_BYTES &&
             head[4] == 0 && head[5] == 148 && !memcmp(file + sizeof head, answers, sizeof answers) &&
             !memcmp(file + sizeof head + 148, control, CONTROL_BYTES) &&
             !memcmp(file + sizeof head + 148 + CONTROL_BYTES, code, PROGRAM_BYTES);
    }
    free(file);
    return ok;
}

int probe_geometry_shader(void) {
    const char *cache = getenv("TOTK_SHADER_CACHE"), *translator = getenv("TOTK_XLATE");
    CHECK(cache && *cache && strlen(cache) < 1024, "launcher must set TOTK_SHADER_CACHE to a fresh owned directory");
    CHECK(translator && *translator && strlen(translator) < 1024 && plat_file_exists(translator),
          "launcher must set TOTK_XLATE to the native translator");
    int nonempty = 0;
    CHECK(plat_fs_kind(cache) == PLAT_FS_DIR && plat_fs_list(cache, cache_entry, &nonempty) == PLAT_FS_OK && !nonempty,
          "cache must already exist and be empty; no existing cache is deleted");
    uint64_t keys, translated;
    gfx_shader_counts(&keys, &translated);
    CHECK(!keys && !translated && !gfx_shader_requests() && !gfx_shader_failures(), "probe needs a fresh shader client");
    const uint32_t subgroup[3] = {32, 32, 32};
    gfx_shader_set_host(1, 0, subgroup, 0), gfx_shader_set_geometry_host(32);
    uint8_t control[CONTROL_BYTES], code[PROGRAM_BYTES], key[TOPOLOGIES][32], iid[TOPOLOGIES][32];
    original_program(control, code);
    for (unsigned i = 0; i < TOPOLOGIES; i++) {
        GfxShaderAnswers a = {.geometry_input = (uint8_t)(i + 1)};
        const GfxShader *s = gfx_shader_get(GFX_STAGE_GS, control, code, sizeof code, NULL, &a);
        CHECK(shader_valid(s, i + 1, NULL, NULL), "fresh topology %u failed", i + 1);
        memcpy(key[i], gfx_shader_key(s), 32), memcpy(iid[i], gfx_shader_input_id(s), 32);
        for (unsigned j = 0; j < i; j++)
            CHECK(memcmp(key[i], key[j], 32) && memcmp(iid[i], iid[j], 32), "topologies %u/%u share identity", i + 1, j + 1);
        CHECK(stored_input_valid(cache, iid[i], i + 1, control, code), "stored input %u is not an exact GS v2 record", i + 1);
    }
    gfx_shader_counts(&keys, &translated);
    CHECK(keys == TOPOLOGIES && translated == TOPOLOGIES && gfx_shader_requests() == TOPOLOGIES,
          "fresh pass expected exactly five cached shaders, translations and child requests");

    gfx_shader_drop_memory();
    for (unsigned i = 0; i < TOPOLOGIES; i++) {
        int owned = 0;
        const GfxShader *s = gfx_shader_load_private(iid[i], &owned);
        int ok = owned == 1 && shader_valid(s, i + 1, key[i], iid[i]);
        if (owned) gfx_shader_release_private(s);
        CHECK(ok, "private disk load %u did not return a valid owned module", i + 1);
    }
    gfx_shader_counts(&keys, &translated);
    CHECK(!keys && translated == TOPOLOGIES && gfx_shader_requests() == TOPOLOGIES, "private loads inserted or translated shaders");
    for (unsigned i = 0; i < TOPOLOGIES; i++) {
        const GfxShader *s = gfx_shader_load_input(iid[i], 0);
        CHECK(shader_valid(s, i + 1, key[i], iid[i]), "by-IID disk load %u failed", i + 1);
        int owned = 1;
        const GfxShader *borrowed = gfx_shader_load_private(iid[i], &owned);
        int ok = !owned && borrowed == s;
        if (owned) gfx_shader_release_private(borrowed);
        CHECK(ok, "private memory load %u did not borrow the cached module", i + 1);
    }

    gfx_shader_drop_memory();
    uint64_t hits = gfx_shader_cache_hits();
    for (unsigned i = 0; i < TOPOLOGIES; i++) {
        GfxShaderAnswers a = {.geometry_input = (uint8_t)(i + 1)};
        const GfxShader *s = gfx_shader_get(GFX_STAGE_GS, control, code, sizeof code, NULL, &a);
        CHECK(shader_valid(s, i + 1, key[i], iid[i]), "request-keyed disk load %u failed", i + 1);
        CHECK(gfx_shader_get(GFX_STAGE_GS, control, code, sizeof code, NULL, &a) == s,
              "memory repeat %u returned a different module", i + 1);
    }
    gfx_shader_counts(&keys, &translated);
    CHECK(keys == TOPOLOGIES && translated == TOPOLOGIES && gfx_shader_requests() == TOPOLOGIES &&
          gfx_shader_cache_hits() - hits == 2 * TOPOLOGIES && !gfx_shader_failures(),
          "warm passes expected ten hits, zero new child requests/translations/failures");
    gfx_shader_drop_memory();
    puts("probe-shader-geometry: passed (5 topologies, 5 real child requests/translations, 5 owned + 5 borrowed private loads, 10 warm hits)");
    puts("probe-shader-geometry: ordinary Maxwell translation/cache only; native lowering, passthrough, GPU pipelines, replay and terrain remain unverified");
    return 0;
}

// Integer vertex inputs of a vertex shader and integer colour outputs of a fragment shader (gfx_shader.c
// int_input_variant, int_output_variant).
// The translator declares every vertex input as vec4 and reads its raw bits (floatBitsToUint), as SM86's ALD does; the
// attribute's format decides what those bits are. Vulkan needs an integer input for an integer format (bone indices in
// R8G8B8A8_UINT, R16G16B16A16_SINT: run-race19 validation), and RADV converts otherwise: the bones read as floats and
// the race's skinned meshes drew nothing. Colour outputs are the same the other way: the fragment shader writes a vec4
// of raw bits and the target's format decides what they are; a float output into an R8_UINT target is undefined in
// Vulkan (the race's mesh draws, issue #64 on Intel Arc). spirv_int_io retypes the Input or Output variables in umask
// (uint) and smask (int) to u/ivec4, bitcasts each load back to float and each stored value to the integer type, so
// the bits pass unchanged. eden-nx2 types the outputs at translation from the target formats (idea only, GPL-3.0:
// runtime_info color_output_types, EmitSetFragColor).

#include <stdlib.h>

#include "gfx.h"
#include "host.h"

enum { IOP_TYPE_INT = 21, IOP_TYPE_FLOAT = 22, IOP_TYPE_VECTOR = 23, IOP_TYPE_POINTER = 32, IOP_FUNCTION = 54,
       IOP_VARIABLE = 59, IOP_LOAD = 61, IOP_STORE = 62, IOP_ACCESS_CHAIN = 65, IOP_IB_ACCESS_CHAIN = 66,
       IOP_DECORATE = 71, IOP_BITCAST = 124 };

uint32_t *spirv_int_io(const uint32_t *w, uint64_t n, uint32_t umask, uint32_t smask, uint32_t sc, uint64_t *out_n) {
    uint32_t bound = w[3], f32 = 0, i32 = 0, u32 = 0, v4f = 0, v4i = 0, v4u = 0, loc_of_max = 64;
    uint32_t *loc = calloc(bound, sizeof *loc);   // Location + 1 of Input variables, 0 none
    uint8_t *kind = calloc(bound, 1);              // variable / access-chain id: 1 uint, 2 int (vector var | 4: scalar ptr)
    uint32_t *out = malloc(sizeof *out * (size_t)(n * 3 + 64));
    if (!loc || !kind || !out) rt_fatal("gfx_shader: out of memory (int inputs)");
    (void)loc_of_max;
    uint64_t fn_at = n;
    for (uint64_t i = 5; i < n;) {   // types and Location decorations
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto fail;
        const uint32_t *a = w + i + 1;
        if (op == IOP_TYPE_FLOAT && len >= 3 && a[1] == 32) f32 = a[0];
        else if (op == IOP_TYPE_INT && len >= 4 && a[1] == 32) { if (a[2]) i32 = a[0]; else u32 = a[0]; }
        else if (op == IOP_DECORATE && len >= 4 && a[1] == 30 && a[0] < bound) loc[a[0]] = a[2] + 1;
        else if (op == IOP_FUNCTION && fn_at == n) fn_at = i;
        i += len;
    }
    for (uint64_t i = 5; i < fn_at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == IOP_TYPE_VECTOR && len >= 4 && a[2] == 4) {
            if (a[1] == f32) v4f = a[0];
            else if (a[1] == i32) v4i = a[0];
            else if (a[1] == u32) v4u = a[0];
        }
        i += len;
    }
    if (!f32 || !v4f) goto fail;
    uint64_t o = 0;
    // header, then everything before the first function, with the variables' types switched and new types added
    // right before the first function (after every type they use).
    uint32_t nu = bound;
    if (!u32) u32 = nu++;
    if (!i32) i32 = nu++;
    if (!v4u) v4u = nu++;
    if (!v4i) v4i = nu++;
    // Input pointers to u/ivec4 and u/int, by kind (initializers are not sequenced: one id at a time)
    uint32_t pv[3] = {0, nu, nu + 1}, ps[3] = {0, nu + 2, nu + 3};
    nu += 4;
    for (uint64_t i = 0; i < 5; i++) out[o++] = w[i];
    // The global section as it was, minus the retyped variables; those, the new types and the integer pointer types go
    // at its end, before the first function, where every type they name is already declared.
    uint32_t moved[32][3];
    int nmoved = 0;
    for (uint64_t i = 5; i < fn_at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == IOP_VARIABLE && len == 4 && a[2] == sc && a[1] < bound && loc[a[1]] && loc[a[1]] <= 32 && nmoved < 32) {
            uint32_t l = loc[a[1]] - 1, k = umask >> l & 1 ? 1 : smask >> l & 1 ? 2 : 0;
            if (k) {
                kind[a[1]] = (uint8_t)k;
                moved[nmoved][0] = a[1], moved[nmoved][1] = k, moved[nmoved++][2] = 1;
                i += len;
                continue;
            }
        }
        for (uint32_t j = 0; j < len; j++) out[o++] = w[i + j];
        i += len;
    }
    if (!nmoved) goto fail;
    if (u32 >= bound) out[o++] = 4u << 16 | IOP_TYPE_INT, out[o++] = u32, out[o++] = 32, out[o++] = 0;
    if (i32 >= bound) out[o++] = 4u << 16 | IOP_TYPE_INT, out[o++] = i32, out[o++] = 32, out[o++] = 1;
    if (v4u >= bound) out[o++] = 4u << 16 | IOP_TYPE_VECTOR, out[o++] = v4u, out[o++] = u32, out[o++] = 4;
    if (v4i >= bound) out[o++] = 4u << 16 | IOP_TYPE_VECTOR, out[o++] = v4i, out[o++] = i32, out[o++] = 4;
    {
        uint32_t vt[3] = {0, v4u, v4i}, st[3] = {0, u32, i32};
        for (int k = 1; k <= 2; k++) {
            out[o++] = 4u << 16 | IOP_TYPE_POINTER, out[o++] = pv[k], out[o++] = sc, out[o++] = vt[k];
            out[o++] = 4u << 16 | IOP_TYPE_POINTER, out[o++] = ps[k], out[o++] = sc, out[o++] = st[k];
        }
    }
    for (int m = 0; m < nmoved; m++)
        out[o++] = 4u << 16 | IOP_VARIABLE, out[o++] = pv[moved[m][1]], out[o++] = moved[m][0], out[o++] = sc;
    // function bodies: access chains into the variables take scalar integer pointers; loads through them load the
    // integer type into a new id and bitcast it to the float type the shader expects under the old id.
    for (uint64_t i = fn_at; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto fail;
        const uint32_t *a = w + i + 1;
        if ((op == IOP_ACCESS_CHAIN || op == IOP_IB_ACCESS_CHAIN) && len >= 4 && a[2] < bound && (kind[a[2]] & 3) &&
            !(kind[a[2]] & 4) && a[1] < bound) {
            uint32_t k = kind[a[2]] & 3;
            kind[a[1]] = (uint8_t)(k | 4);
            out[o++] = w[i], out[o++] = ps[k];
            for (uint32_t j = 1; j < len - 1; j++) out[o++] = a[j];
            i += len;
            continue;
        }
        if (op == IOP_LOAD && len >= 4 && a[2] < bound && (kind[a[2]] & 3)) {
            uint32_t k = kind[a[2]] & 3, scalar = kind[a[2]] & 4, tmp = nu++;
            out[o++] = w[i], out[o++] = scalar ? (k == 1 ? u32 : i32) : (k == 1 ? v4u : v4i), out[o++] = tmp;
            for (uint32_t j = 2; j < len - 1; j++) out[o++] = a[j];
            out[o++] = 4u << 16 | IOP_BITCAST, out[o++] = a[0], out[o++] = a[1], out[o++] = tmp;
            i += len;
            continue;
        }
        if (op == IOP_STORE && len >= 3 && a[0] < bound && (kind[a[0]] & 3)) {   // the value bitcast, then stored
            uint32_t k = kind[a[0]] & 3, scalar = kind[a[0]] & 4, tmp = nu++;
            out[o++] = 4u << 16 | IOP_BITCAST, out[o++] = scalar ? (k == 1 ? u32 : i32) : (k == 1 ? v4u : v4i);
            out[o++] = tmp, out[o++] = a[1];
            out[o++] = w[i], out[o++] = a[0], out[o++] = tmp;
            for (uint32_t j = 2; j < len - 1; j++) out[o++] = a[j];
            i += len;
            continue;
        }
        for (uint32_t j = 0; j < len; j++) out[o++] = w[i + j];
        i += len;
    }
    out[3] = nu;
    free(loc), free(kind);
    *out_n = o;
    // The variant lives as long as the process (gfx_shader_int.c): keep its words, not the worst-case buffer.
    uint32_t *fit = realloc(out, sizeof *out * (size_t)o);
    return fit ? fit : out;
fail:
    free(loc), free(kind), free(out);
    return NULL;
}


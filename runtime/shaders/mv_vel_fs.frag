#version 450
// THWOMP object motion (runtime/gfx/vk/device/vk_mv.c): the motion of the visible surface only (its depth equals the
// scene's), in pixels from last frame to this one, image rows downwards; alpha 1 marks an object pixel.
layout(binding = 2) uniform sampler2D depth;
layout(push_constant) uniform P { uint base_cur, base_prev; vec2 size; vec2 nf; } pc;   // nf: near and far planes
layout(location = 0) in vec4 v_cur;
layout(location = 1) in vec4 v_prev;
layout(location = 0) out vec4 o;

void main() {
    // The visible surface: within 2 cm or 0.2 % of the distance of the scene's depth, compared as distances (depth is
    // stored d = z_ndc / 2 + 1 / 2, z_ndc of -1..1). Exact equality left out what the game draws with a depth offset
    // (decals such as the logos on the tyres), whose pixels then took the camera's motion and smeared behind a kart.
    float n = pc.nf.x, f = pc.nf.y;
    float zs = 2.0 * n * f / (f + n - (texelFetch(depth, ivec2(gl_FragCoord.xy), 0).r * 2.0 - 1.0) * (f - n));
    float zf = 2.0 * n * f / (f + n - (gl_FragCoord.z * 2.0 - 1.0) * (f - n));
    if (abs(zf - zs) > max(0.02, 0.002 * zs)) discard;
    vec2 c = v_cur.xy / v_cur.w, p = v_prev.xy / v_prev.w;   // NDC, y up (the viewport flips it onto the rows)
    vec2 mv = vec2(c.x - p.x, p.y - c.y) * 0.5 * pc.size;
    o = vec4(mv, 0.0, 1.0);
}

#version 450
// THWOMP object motion (runtime/gfx/vk/device/vk_mv.c): one object's triangles again, from the clip positions its own
// vertex shader produced this frame (cur, captured by transform feedback) and last frame (prev, the same object's
// capture then). Rasterized at the current positions; both positions go to the fragment stage.
layout(std430, binding = 0) readonly buffer Cur { vec4 cur[]; };
layout(std430, binding = 1) readonly buffer Prev { vec4 prev[]; };
layout(push_constant) uniform P { uint base_cur, base_prev; vec2 size; vec2 nf; } pc;
layout(location = 0) out vec4 v_cur;
layout(location = 1) out vec4 v_prev;

void main() {
    vec4 c = cur[pc.base_cur + uint(gl_VertexIndex)];
    v_cur = c;
    v_prev = prev[pc.base_prev + uint(gl_VertexIndex)];
    gl_Position = c;
}

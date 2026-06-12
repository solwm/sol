#version 450

// Frosted-backdrop fragment shader. Samples the already-blurred FBO at
// the element's screen-rect UV and paints it underneath an inactive
// toplevel.
//
// The backdrop pipeline runs with blending DISABLED — NVIDIA's
// fractional-alpha read-modify-write path corrupts the bottom-right
// of the framebuffer (blends against stale/cleared destination
// pixels), visible as a dark cross-window region behind frosted
// inactive windows. Writing opaque pixels sidesteps the RMW path
// entirely (same workaround as niri-sol's manual-blend offscreen).
// Consequences, both accepted: the rounded-corner mask is gone (the
// ~radius px outside the window's rounding show frosted instead of
// sharp wallpaper, and the window's own mask still shapes the visible
// window), and `pc.alpha` fades degrade to full-strength frost (the
// fading window above still animates normally).

layout(push_constant) uniform PC {
    layout(offset = 0)  vec4 rect;
    layout(offset = 16) vec4 uv;
    layout(offset = 32) vec2 size;
    layout(offset = 40) float radius;
    layout(offset = 44) float alpha;
} pc;

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pos;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = vec4(texture(u_tex, v_uv).rgb, 1.0);
}

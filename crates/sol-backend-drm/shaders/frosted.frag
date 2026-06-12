#version 450

// Frosted-window fragment shader: composites an inactive toplevel over
// the blurred-wallpaper FBO entirely in-shader and writes OPAQUE
// pixels (pipeline blending disabled).
//
// This replaces the old two-draw scheme (backdrop quad + alpha-blended
// window quad). Both of those were fractional-alpha read-modify-write
// draws, and NVIDIA has a raster-order defect where such draws in the
// bottom-right region of the framebuffer blend against stale/cleared
// destination pixels — the dark cross-window "cursed region". With the
// whole frosted composite computed here and written opaque, no blended
// draw is left in the path.
//
// set 0: the window's texture. set 1: the blurred background FBO,
// sampled at the output pixel position (gl_FragCoord / fb_size — both
// are top-left-origin in Vulkan, no flip).
//
// Outside the rounded-corner mask the output degrades to the blurred
// background instead of the sharp wallpaper — ~radius px in each
// corner, invisible behind a translucent window edge.

layout(push_constant) uniform PC {
    layout(offset = 0)  vec4 rect;
    layout(offset = 16) vec4 uv;
    layout(offset = 32) vec2 size;    // output rect size in pixels
    layout(offset = 40) float radius; // corner radius in pixels
    layout(offset = 44) float alpha;  // effective window alpha (incl. inactive dim)
    layout(offset = 48) float opaque; // XRGB opt-in, same as quad.frag
    layout(offset = 56) vec2 fb_size; // framebuffer size in pixels
} pc;

layout(set = 0, binding = 0) uniform sampler2D u_win;
layout(set = 1, binding = 0) uniform sampler2D u_bg;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pos;

layout(location = 0) out vec4 o_color;

float rounded_alpha() {
    vec2 pos = v_pos * pc.size;
    vec2 half_size = pc.size * 0.5;
    vec2 q = abs(pos - half_size) - (half_size - vec2(pc.radius));
    float d = length(max(q, vec2(0.0)))
            + min(max(q.x, q.y), 0.0)
            - pc.radius;
    return clamp(0.5 - d, 0.0, 1.0);
}

void main() {
    vec4 t = texture(u_win, v_uv);
    vec3 bg = texture(u_bg, gl_FragCoord.xy / pc.fb_size).rgb;
    float a = mix(t.a, 1.0, pc.opaque) * pc.alpha * rounded_alpha();
    o_color = vec4(mix(bg, t.rgb, a), 1.0);
}

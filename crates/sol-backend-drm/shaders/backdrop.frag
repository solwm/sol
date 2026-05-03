#version 450

// Frosted-backdrop fragment shader. Samples the already-blurred FBO at
// the element's screen-rect UV and paints it underneath an inactive
// toplevel, with the same rounded-rect mask the textured quad uses so
// the backdrop and the window line up. `alpha` lets the caller fade
// the backdrop in/out (e.g. during a workspace crossfade).

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
    float a = pc.alpha * rounded_alpha();
    o_color = vec4(texture(u_tex, v_uv).rgb, a);
}

#version 450

// 5x5 box-blur fragment shader. Samples 25 taps around v_uv weighted equally.
// Cheaper than a Gaussian and visually indistinguishable after a couple of
// passes — `box (*) box (*) ...` approaches a Gaussian by the central limit
// theorem. `texel` is the per-pass UV-space offset for one source texel
// (1.0 / source_size, optionally scaled by the configured radius for wider
// reach without more samples).

layout(push_constant) uniform PC {
    layout(offset = 0)  vec4 rect;
    layout(offset = 16) vec4 uv;
    layout(offset = 32) vec2 texel;
} pc;

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pos;

layout(location = 0) out vec4 o_color;

void main() {
    vec4 sum = vec4(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 off = vec2(float(dx), float(dy)) * pc.texel;
            sum += texture(u_tex, v_uv + off);
        }
    }
    o_color = sum * (1.0 / 25.0);
}

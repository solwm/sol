#version 450

// Textured-quad fragment shader. Samples a single combined image sampler
// (bound at descriptor set 0, binding 0), applies a rounded-rect SDF mask
// for the corner radius, multiplies in the per-element alpha, and outputs
// non-premultiplied RGBA. Pair with a SRC_ALPHA / ONE_MINUS_SRC_ALPHA blend
// state on the pipeline.
//
// `opaque` is the XRGB opt-in: if 1.0, ignore the texture's alpha channel
// and treat the surface as fully opaque (Wayland XRGB8888 / DRM 'X' fourcc).
// Otherwise the texture's own alpha contributes.
//
// All textures upload as VK_FORMAT_B8G8R8A8_UNORM, which means the sampler
// already returns (R, G, B, A) ordered correctly for both SHM (Wayland
// little-endian ARGB) and dmabuf imports — no per-shader swizzle.

layout(push_constant) uniform PC {
    layout(offset = 0)  vec4 rect;
    layout(offset = 16) vec4 uv;
    layout(offset = 32) vec2 size;    // output rect size in pixels
    layout(offset = 40) float radius; // corner radius in pixels
    layout(offset = 44) float alpha;
    layout(offset = 48) float opaque;
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
    vec4 t = texture(u_tex, v_uv);
    float a = mix(t.a, 1.0, pc.opaque) * pc.alpha * rounded_alpha();
    // Non-premultiplied: caller's blend state weights src by src_alpha,
    // so colour stays at full strength here. Multiplying by pc.alpha as
    // well would double-fade and look dim instead of transparent.
    o_color = vec4(t.rgb, a);
}

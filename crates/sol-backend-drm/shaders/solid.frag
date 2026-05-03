#version 450

// Flat-colour rounded ring / rect shader for borders and overlays. Three
// shape modes selected by the trailing fields:
//   border == 0, radius == 0 -> solid filled rectangle
//   border == 0, radius >  0 -> solid rounded rect
//   border >  0              -> rounded ring of `border`-pixel thickness;
//                               the inner radius shrinks by `border`
//                               (clamped at 0) so both edges stay
//                               concentric.
//
// Outputs straight RGBA — caller's blend state composites it over the
// scene below.

layout(push_constant) uniform PC {
    layout(offset = 0)  vec4 rect;
    layout(offset = 16) vec4 color;   // straight RGBA
    layout(offset = 32) vec2 size;    // px
    layout(offset = 40) float radius; // outer corner radius (px)
    layout(offset = 44) float border; // ring thickness (px); 0 = filled
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pos;

layout(location = 0) out vec4 o_color;

void main() {
    vec2 pos = v_pos * pc.size;
    vec2 half_size = pc.size * 0.5;
    vec2 q = abs(pos - half_size) - (half_size - vec2(pc.radius));
    float d_outer = length(max(q, vec2(0.0)))
                  + min(max(q.x, q.y), 0.0)
                  - pc.radius;
    float alpha = clamp(0.5 - d_outer, 0.0, 1.0);

    if (pc.border > 0.0) {
        float inner_radius = max(pc.radius - pc.border, 0.0);
        vec2 inner_size = pc.size - vec2(2.0 * pc.border);
        vec2 inner_pos = pos - vec2(pc.border);
        vec2 ihalf = inner_size * 0.5;
        vec2 iq = abs(inner_pos - ihalf) - (ihalf - vec2(inner_radius));
        float d_inner = length(max(iq, vec2(0.0)))
                      + min(max(iq.x, iq.y), 0.0)
                      - inner_radius;
        // Want fragments OUTSIDE the inner shape (positive d_inner).
        alpha *= clamp(0.5 + d_inner, 0.0, 1.0);
    }

    o_color = vec4(pc.color.rgb, pc.color.a * alpha);
}

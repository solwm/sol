#version 450

// Shared vertex shader for the textured-quad family (quad, blur, backdrop).
// Generates the 4 corners of a unit quad from gl_VertexIndex (TriangleStrip),
// places them into the NDC rect supplied by the push-constant block, and
// passes the per-fragment UV + intra-quad position through.
//
// Vulkan's NDC has +Y pointing down; the presenter flips that on the CPU
// side by feeding the NDC rect with top-edge at `pc.rect.y` and bottom-edge
// at `pc.rect.y + pc.rect.w` (the GBM scanout BO's top scanline corresponds
// to NDC y = -1). The same convention applies to every pipeline that uses
// this vertex shader, so the fragment side never has to reason about
// orientation.

layout(push_constant) uniform PC {
    vec4 rect;   // x, y, w, h in NDC
    vec4 uv;     // x, y, w, h in [0, 1] texture space (top-left origin)
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec2 v_pos;

void main() {
    // (0,0) (1,0) (0,1) (1,1)
    vec2 a_pos = vec2(float(gl_VertexIndex & 1),
                      float((gl_VertexIndex >> 1) & 1));
    vec2 ndc = pc.rect.xy + a_pos * pc.rect.zw;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = pc.uv.xy + a_pos * pc.uv.zw;
    v_pos = a_pos;
}

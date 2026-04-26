//! Textured-quad GLES2 programs.
//!
//! Two variants share a vertex shader but differ in texture binding:
//! - `build()` — `sampler2D` on `GL_TEXTURE_2D`, used for SHM uploads.
//!   Swizzles R<->B because clients write ARGB8888 in memory and our
//!   glTex{Sub,}Image2D path interprets the bytes as GL_RGBA.
//! - `build_external()` — `samplerExternalOES` on `GL_TEXTURE_EXTERNAL_OES`,
//!   used for dmabuf-imported textures. Mesa's driver handles the format
//!   swizzle internally, so no BGR flip here. Required because Mesa returns
//!   external-only images for many dmabuf imports — binding those to
//!   `GL_TEXTURE_2D` silently yields black samples.

use anyhow::{Result, anyhow};
use glow::{HasContext, NativeBuffer, NativeProgram, NativeUniformLocation};

const VS: &str = r#"#version 100
attribute vec2 a_pos;
uniform vec4 u_rect;
uniform vec4 u_uv;    // (x, y, w, h) in normalized [0,1] texture coords
varying vec2 v_uv;
void main() {
    vec2 p = u_rect.xy + a_pos * u_rect.zw;
    gl_Position = vec4(p, 0.0, 1.0);
    // Sample a sub-rect of the texture, picked by the caller per draw.
    // Full-texture default is u_uv = (0, 0, 1, 1). Wayland surface coords
    // go top-to-bottom; GL texture origin is bottom, so flip the v term
    // inside the sub-rect: start at uv.y + uv.h at a_pos.y = 0, decrease
    // to uv.y at a_pos.y = 1.
    v_uv = vec2(
        u_uv.x + a_pos.x * u_uv.z,
        u_uv.y + (1.0 - a_pos.y) * u_uv.w
    );
}
"#;

const FS: &str = r#"#version 100
precision mediump float;
uniform sampler2D u_tex;
uniform float u_opaque;
uniform float u_alpha;
varying vec2 v_uv;
void main() {
    vec4 t = texture2D(u_tex, v_uv);
    float a = mix(t.a, 1.0, u_opaque) * u_alpha;
    // Non-premultiplied alpha: pair with `glBlendFunc(SRC_ALPHA,
    // ONE_MINUS_SRC_ALPHA)`. Multiplying the rgb by u_alpha here
    // would double-fade — the blend already weights src by
    // src_alpha, so an extra * u_alpha squashes the colour without
    // changing how much dst shows through (looks dim, not
    // transparent). Keep colour at full strength and let the blend
    // do its job.
    gl_FragColor = vec4(t.bgr, a);
}
"#;

const FS_EXTERNAL: &str = r#"#version 100
#extension GL_OES_EGL_image_external : require
precision mediump float;
uniform samplerExternalOES u_tex;
uniform float u_opaque;
uniform float u_alpha;
varying vec2 v_uv;
void main() {
    vec4 t = texture2D(u_tex, v_uv);
    float a = mix(t.a, 1.0, u_opaque) * u_alpha;
    // Driver returns (R, G, B, A) in fourcc order for external images,
    // so no channel swizzle needed here (unlike the SHM path).
    // Non-premultiplied — see FS above for the blend-function
    // pairing rationale.
    gl_FragColor = vec4(t.rgb, a);
}
"#;

/// Flat-color fragment shader for border/overlay rects. Writes
/// `u_color` straight through; source-over blending with the existing
/// framebuffer happens via the pipeline's `glBlendFunc`.
const FS_SOLID: &str = r#"#version 100
precision mediump float;
uniform vec4 u_color;
void main() { gl_FragColor = u_color; }
"#;

/// Box-blur fragment shader. Samples a 5×5 neighbourhood (25 taps)
/// around `v_uv`, weighted equally, and writes the average. Cheaper
/// than a Gaussian, looks identical to the eye after a couple of
/// passes (each pass is mathematically a convolution; box ⊛ box ⊛
/// ... approaches Gaussian by the central limit theorem).
///
/// `u_texel` is the 1-texel offset in UV space (1.0 / texture_size).
/// Caller computes it once based on the FBO's actual dimensions.
const FS_BLUR: &str = r#"#version 100
precision mediump float;
uniform sampler2D u_tex;
uniform vec2 u_texel;
varying vec2 v_uv;
void main() {
    vec4 sum = vec4(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 off = vec2(float(dx), float(dy)) * u_texel;
            sum += texture2D(u_tex, v_uv + off);
        }
    }
    gl_FragColor = sum * (1.0 / 25.0);
}
"#;

/// Backdrop fragment shader: draws a sub-rect of the
/// already-blurred FBO texture into the current viewport. No
/// channel swizzle (we wrote into the FBO with GL's native ordering),
/// alpha multiplied by `u_alpha` so the caller can fade the
/// backdrop in/out (e.g. during a workspace crossfade).
const FS_BACKDROP: &str = r#"#version 100
precision mediump float;
uniform sampler2D u_tex;
uniform float u_alpha;
varying vec2 v_uv;
void main() {
    // Non-premultiplied — same convention as the textured-quad
    // path so the blend function pairs correctly.
    gl_FragColor = vec4(texture2D(u_tex, v_uv).rgb, u_alpha);
}
"#;

pub struct QuadProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_rect: NativeUniformLocation,
    pub u_uv: NativeUniformLocation,
    pub u_tex: NativeUniformLocation,
    pub u_opaque: NativeUniformLocation,
    pub u_alpha: NativeUniformLocation,
}

/// Border / solid-color rect program. Same vertex pipeline and VBO
/// convention as the textured quads, but the fragment shader just
/// writes a uniform color. Kept separate from `QuadProgram` so we
/// don't carry unused `u_tex` / `u_opaque` locations around.
pub struct SolidProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_rect: NativeUniformLocation,
    pub u_color: NativeUniformLocation,
}

/// 5×5 box-blur ping-pong program. Operates on FBO-attached colour
/// textures.
pub struct BlurProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_rect: NativeUniformLocation,
    pub u_uv: NativeUniformLocation,
    pub u_tex: NativeUniformLocation,
    pub u_texel: NativeUniformLocation,
}

/// Frosted-backdrop draw program. Same vertex pipeline as `QuadProgram`
/// but the fragment shader just samples + alpha-multiplies; no opaque
/// fallback or channel swizzle (the FBO is GL-native).
pub struct BackdropProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_rect: NativeUniformLocation,
    pub u_uv: NativeUniformLocation,
    pub u_tex: NativeUniformLocation,
    pub u_alpha: NativeUniformLocation,
}

pub fn build(gl: &glow::Context) -> Result<QuadProgram> {
    build_with_fs(gl, FS)
}

/// Companion to [`build`] that uses the `samplerExternalOES` fragment
/// shader for dmabuf-imported textures bound to `GL_TEXTURE_EXTERNAL_OES`.
pub fn build_external(gl: &glow::Context) -> Result<QuadProgram> {
    build_with_fs(gl, FS_EXTERNAL)
}

pub fn build_blur(gl: &glow::Context) -> Result<BlurProgram> {
    unsafe {
        let vs = compile(gl, glow::VERTEX_SHADER, VS)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, FS_BLUR)?;
        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow!("blur link: {log}"));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let u_rect = gl
            .get_uniform_location(program, "u_rect")
            .ok_or_else(|| anyhow!("u_rect missing"))?;
        let u_uv = gl
            .get_uniform_location(program, "u_uv")
            .ok_or_else(|| anyhow!("u_uv missing"))?;
        let u_tex = gl
            .get_uniform_location(program, "u_tex")
            .ok_or_else(|| anyhow!("u_tex missing"))?;
        let u_texel = gl
            .get_uniform_location(program, "u_texel")
            .ok_or_else(|| anyhow!("u_texel missing"))?;

        let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow!("create_buffer: {e}"))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytes_of_f32(&verts),
            glow::STATIC_DRAW,
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(BlurProgram {
            program,
            vbo,
            u_rect,
            u_uv,
            u_tex,
            u_texel,
        })
    }
}

pub fn build_backdrop(gl: &glow::Context) -> Result<BackdropProgram> {
    unsafe {
        let vs = compile(gl, glow::VERTEX_SHADER, VS)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, FS_BACKDROP)?;
        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow!("backdrop link: {log}"));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let u_rect = gl
            .get_uniform_location(program, "u_rect")
            .ok_or_else(|| anyhow!("u_rect missing"))?;
        let u_uv = gl
            .get_uniform_location(program, "u_uv")
            .ok_or_else(|| anyhow!("u_uv missing"))?;
        let u_tex = gl
            .get_uniform_location(program, "u_tex")
            .ok_or_else(|| anyhow!("u_tex missing"))?;
        let u_alpha = gl
            .get_uniform_location(program, "u_alpha")
            .ok_or_else(|| anyhow!("u_alpha missing"))?;

        let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow!("create_buffer: {e}"))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytes_of_f32(&verts),
            glow::STATIC_DRAW,
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(BackdropProgram {
            program,
            vbo,
            u_rect,
            u_uv,
            u_tex,
            u_alpha,
        })
    }
}

pub fn build_solid(gl: &glow::Context) -> Result<SolidProgram> {
    unsafe {
        let vs = compile(gl, glow::VERTEX_SHADER, VS)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, FS_SOLID)?;
        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow!("solid link: {log}"));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let u_rect = gl
            .get_uniform_location(program, "u_rect")
            .ok_or_else(|| anyhow!("u_rect missing"))?;
        let u_color = gl
            .get_uniform_location(program, "u_color")
            .ok_or_else(|| anyhow!("u_color missing"))?;

        let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow!("create_buffer: {e}"))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytes_of_f32(&verts),
            glow::STATIC_DRAW,
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(SolidProgram {
            program,
            vbo,
            u_rect,
            u_color,
        })
    }
}

fn build_with_fs(gl: &glow::Context, fs_src: &str) -> Result<QuadProgram> {
    unsafe {
        let vs = compile(gl, glow::VERTEX_SHADER, VS)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, fs_src)?;
        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow!("quad link: {log}"));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let u_rect = gl
            .get_uniform_location(program, "u_rect")
            .ok_or_else(|| anyhow!("u_rect missing"))?;
        let u_uv = gl
            .get_uniform_location(program, "u_uv")
            .ok_or_else(|| anyhow!("u_uv missing"))?;
        let u_tex = gl
            .get_uniform_location(program, "u_tex")
            .ok_or_else(|| anyhow!("u_tex missing"))?;
        let u_opaque = gl
            .get_uniform_location(program, "u_opaque")
            .ok_or_else(|| anyhow!("u_opaque missing"))?;
        let u_alpha = gl
            .get_uniform_location(program, "u_alpha")
            .ok_or_else(|| anyhow!("u_alpha missing"))?;

        let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow!("create_buffer: {e}"))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytes_of_f32(&verts),
            glow::STATIC_DRAW,
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(QuadProgram {
            program,
            vbo,
            u_rect,
            u_uv,
            u_tex,
            u_opaque,
            u_alpha,
        })
    }
}

unsafe fn compile(gl: &glow::Context, kind: u32, src: &str) -> Result<glow::NativeShader> {
    let s = gl.create_shader(kind).map_err(|e| anyhow!("create_shader: {e}"))?;
    gl.shader_source(s, src);
    gl.compile_shader(s);
    if !gl.get_shader_compile_status(s) {
        let log = gl.get_shader_info_log(s);
        return Err(anyhow!("shader compile: {log}"));
    }
    Ok(s)
}

fn bytes_of_f32(f: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, std::mem::size_of_val(f)) }
}

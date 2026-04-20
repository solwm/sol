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
varying vec2 v_uv;
void main() {
    vec2 p = u_rect.xy + a_pos * u_rect.zw;
    gl_Position = vec4(p, 0.0, 1.0);
    // Wayland surface coordinates go top-to-bottom; GL texture origin is at
    // the bottom, so flip v to match.
    v_uv = vec2(a_pos.x, 1.0 - a_pos.y);
}
"#;

const FS: &str = r#"#version 100
precision mediump float;
uniform sampler2D u_tex;
uniform float u_opaque;
varying vec2 v_uv;
void main() {
    vec4 t = texture2D(u_tex, v_uv);
    float a = mix(t.a, 1.0, u_opaque);
    gl_FragColor = vec4(t.bgr, a);
}
"#;

const FS_EXTERNAL: &str = r#"#version 100
#extension GL_OES_EGL_image_external : require
precision mediump float;
uniform samplerExternalOES u_tex;
uniform float u_opaque;
varying vec2 v_uv;
void main() {
    vec4 t = texture2D(u_tex, v_uv);
    float a = mix(t.a, 1.0, u_opaque);
    // Driver returns (R, G, B, A) in fourcc order for external images,
    // so no channel swizzle needed here (unlike the SHM path).
    gl_FragColor = vec4(t.rgb, a);
}
"#;

pub struct QuadProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_rect: NativeUniformLocation,
    pub u_tex: NativeUniformLocation,
    pub u_opaque: NativeUniformLocation,
}

pub fn build(gl: &glow::Context) -> Result<QuadProgram> {
    build_with_fs(gl, FS)
}

/// Companion to [`build`] that uses the `samplerExternalOES` fragment
/// shader for dmabuf-imported textures bound to `GL_TEXTURE_EXTERNAL_OES`.
pub fn build_external(gl: &glow::Context) -> Result<QuadProgram> {
    build_with_fs(gl, FS_EXTERNAL)
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
        let u_tex = gl
            .get_uniform_location(program, "u_tex")
            .ok_or_else(|| anyhow!("u_tex missing"))?;
        let u_opaque = gl
            .get_uniform_location(program, "u_opaque")
            .ok_or_else(|| anyhow!("u_opaque missing"))?;

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
            u_tex,
            u_opaque,
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

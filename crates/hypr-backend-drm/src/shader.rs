//! Minimal GLES2 checkerboard program: two triangles covering clip space,
//! fragment shader that xors (floor(x/cell) ^ floor(y/cell)) to pick between
//! two tile colours and cycles the tile size by time.

use anyhow::{Result, anyhow};
use glow::{HasContext, NativeBuffer, NativeProgram, NativeUniformLocation};

const VS: &str = r#"#version 100
attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
"#;

const FS: &str = r#"#version 100
precision mediump float;
uniform vec2 u_resolution;
uniform float u_time;
void main() {
    // Cell size scales with resolution so it reads the same at 720p / 4K.
    float cell = (u_resolution.y / 30.0) + 16.0 * sin(u_time);
    vec2 c = floor(gl_FragCoord.xy / cell);
    float on = mod(c.x + c.y, 2.0);
    // Hue tint drifts across X so the image isn't monotone.
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec3 a = vec3(0.90, 0.32, 0.80) * (0.6 + 0.4 * uv.x);
    vec3 b = vec3(0.10, 0.12, 0.16 + 0.15 * uv.y);
    vec3 col = mix(b, a, on);
    gl_FragColor = vec4(col, 1.0);
}
"#;

pub struct CheckerProgram {
    pub program: NativeProgram,
    pub vbo: NativeBuffer,
    pub u_resolution: NativeUniformLocation,
    pub u_time: NativeUniformLocation,
}

pub fn build_checkerboard(gl: &glow::Context) -> Result<CheckerProgram> {
    unsafe {
        let vs = compile(gl, glow::VERTEX_SHADER, VS)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, FS)?;
        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow!("link: {log}"));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let u_resolution = gl
            .get_uniform_location(program, "u_resolution")
            .ok_or_else(|| anyhow!("u_resolution missing"))?;
        let u_time = gl
            .get_uniform_location(program, "u_time")
            .ok_or_else(|| anyhow!("u_time missing"))?;

        // Fullscreen triangle strip: (-1,-1) (1,-1) (-1,1) (1,1)
        let verts: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let vbo = gl.create_buffer().map_err(|e| anyhow!("create_buffer: {e}"))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck_slice(&verts),
            glow::STATIC_DRAW,
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(CheckerProgram {
            program,
            vbo,
            u_resolution,
            u_time,
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

fn bytemuck_slice(floats: &[f32]) -> &[u8] {
    // Safe: f32 has a defined bit layout and we're reading as bytes only for
    // GL buffer upload.
    unsafe {
        std::slice::from_raw_parts(floats.as_ptr() as *const u8, std::mem::size_of_val(floats))
    }
}

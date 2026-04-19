//! Frame-by-frame presenter built on the B3 DRM+GBM+EGL+GLES stack.
//!
//! DrmPresenter wraps modeset bring-up and exposes `render_scene`, which the
//! Wayland server calls each time its scene needs to be re-drawn. Each call
//! clears the GL framebuffer, uploads every scene element to a cached
//! `NativeTexture`, draws textured quads, swaps the EGL surface, schedules
//! a drmModePageFlip, and blocks on the DRM fd for the flip-complete event.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use drm::Device as BasicDevice;
use drm::control::{Device as ControlDevice, PageFlipFlags, framebuffer};
use glow::HasContext;
use voidptr_core::{PixelFormat, Scene};

use crate::{
    Card, GlStack, OutputSelection, get_or_add_fb, pick_output, quad::QuadProgram,
    wait_for_page_flip,
};

pub struct DrmPresenter {
    card: Card,
    gl_stack: GlStack,
    sel: OutputSelection,
    fb_cache: HashMap<usize, framebuffer::Handle>,
    scanned_out: Option<gbm::BufferObject<()>>,
    quad: QuadProgram,
    /// GPU textures keyed by Scene::buffer_key. Kept across frames so
    /// uploads are glTexSubImage2D after the first sight.
    textures: HashMap<u64, TextureEntry>,
    width: u32,
    height: u32,
}

struct TextureEntry {
    tex: glow::NativeTexture,
    width: i32,
    height: i32,
}

impl DrmPresenter {
    pub fn new(device: &Path) -> Result<Self> {
        let card = Card::open(device)?;
        card.acquire_master_lock().map_err(|e| {
            anyhow!(
                "could not become DRM master on {}: {e:?}\n\nHyprland (or another compositor) likely owns this VT. Switch to a free TTY first.",
                device.display()
            )
        })?;

        let sel = pick_output(&card)?;
        let (w_i16, h_i16) = sel.mode.size();
        let width = w_i16 as u32;
        let height = h_i16 as u32;

        let gl_stack = GlStack::new(card.clone(), width, height)?;
        let quad = crate::quad::build(&gl_stack.gl)?;

        let mut presenter = Self {
            card,
            gl_stack,
            sel,
            fb_cache: HashMap::new(),
            scanned_out: None,
            quad,
            textures: HashMap::new(),
            width,
            height,
        };
        presenter.initial_modeset()?;
        Ok(presenter)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn initial_modeset(&mut self) -> Result<()> {
        // Render a dark frame so the first thing on screen isn't undefined
        // GBM memory (some drivers show garbage).
        unsafe {
            self.gl_stack.gl.viewport(0, 0, self.width as i32, self.height as i32);
            self.gl_stack.gl.clear_color(0.02, 0.02, 0.04, 1.0);
            self.gl_stack.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.gl_stack
            .egl
            .swap_buffers(self.gl_stack.display, self.gl_stack.surface)
            .map_err(|e| anyhow!("initial swap_buffers: {e:?}"))?;
        let bo = unsafe { self.gl_stack.gbm_surface.lock_front_buffer() }
            .context("initial lock_front_buffer")?;
        let fb = get_or_add_fb(&self.card, &bo, &mut self.fb_cache)?;
        self.card
            .set_crtc(
                self.sel.crtc,
                Some(fb),
                (0, 0),
                &[self.sel.connector],
                Some(self.sel.mode),
            )
            .context("set_crtc (initial modeset)")?;
        self.scanned_out = Some(bo);
        tracing::info!(width = self.width, height = self.height, "modeset complete");
        Ok(())
    }

    /// Redraw the given scene and present it, blocking on vsync.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<()> {
        let w = self.width as i32;
        let h = self.height as i32;

        // Pass 1: make sure every scene element has an up-to-date texture.
        for elem in &scene.elements {
            upload_texture(&self.gl_stack.gl, &mut self.textures, elem)?;
        }

        // Pass 2: draw. After pass 1 nothing else needs &mut self.textures.
        let gl = &self.gl_stack.gl;
        unsafe {
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.02, 0.02, 0.04, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Alpha blending for ARGB surfaces (notably the software cursor).
            // XRGB surfaces set u_opaque=1.0 so alpha ends up 1.0 and the
            // blend is a no-op for them.
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.quad.program));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad.vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.quad.u_tex), 0);
            gl.active_texture(glow::TEXTURE0);
        }

        for elem in &scene.elements {
            let Some(entry) = self.textures.get(&elem.buffer_key) else {
                continue;
            };
            let x0 = (elem.x as f32 / w as f32) * 2.0 - 1.0;
            let y0 = 1.0 - ((elem.y + entry.height) as f32 / h as f32) * 2.0;
            let rw = entry.width as f32 / w as f32 * 2.0;
            let rh = entry.height as f32 / h as f32 * 2.0;
            let opaque = match elem.format {
                PixelFormat::Argb8888 => 0.0,
                PixelFormat::Xrgb8888 => 1.0,
            };
            unsafe {
                gl.bind_texture(glow::TEXTURE_2D, Some(entry.tex));
                gl.uniform_4_f32(Some(&self.quad.u_rect), x0, y0, rw, rh);
                gl.uniform_1_f32(Some(&self.quad.u_opaque), opaque);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        }

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.disable_vertex_attrib_array(0);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
            gl.disable(glow::BLEND);
        }

        self.present_and_flip()
    }

    fn present_and_flip(&mut self) -> Result<()> {
        self.gl_stack
            .egl
            .swap_buffers(self.gl_stack.display, self.gl_stack.surface)
            .map_err(|e| anyhow!("swap_buffers: {e:?}"))?;
        let next_bo = unsafe { self.gl_stack.gbm_surface.lock_front_buffer() }
            .context("lock_front_buffer in present")?;
        let next_fb = get_or_add_fb(&self.card, &next_bo, &mut self.fb_cache)?;
        self.card
            .page_flip(self.sel.crtc, next_fb, PageFlipFlags::EVENT, None)
            .context("page_flip")?;
        wait_for_page_flip(&self.card).context("wait_for_page_flip")?;
        self.scanned_out = Some(next_bo);
        Ok(())
    }
}

fn upload_texture(
    gl: &glow::Context,
    textures: &mut HashMap<u64, TextureEntry>,
    elem: &voidptr_core::SceneElement<'_>,
) -> Result<()> {
    let expected_bytes = (elem.stride as usize).saturating_mul(elem.height as usize);
    if elem.pixels.len() < expected_bytes {
        tracing::warn!("scene element buffer shorter than stride*height; skipping");
        return Ok(());
    }

    // If the source has padded rows, copy into a tight buffer (GLES2 has no
    // GL_UNPACK_ROW_LENGTH to pass stride directly).
    let tight_width_bytes = elem.width as usize * 4;
    let owned_tight;
    let pixels: &[u8] = if elem.stride as usize == tight_width_bytes {
        &elem.pixels[..tight_width_bytes * elem.height as usize]
    } else {
        let mut buf = Vec::with_capacity(tight_width_bytes * elem.height as usize);
        for row in 0..elem.height as usize {
            let start = row * elem.stride as usize;
            let end = start + tight_width_bytes;
            buf.extend_from_slice(&elem.pixels[start..end]);
        }
        owned_tight = buf;
        &owned_tight
    };

    let needs_new = textures
        .get(&elem.buffer_key)
        .is_none_or(|e| e.width != elem.width || e.height != elem.height);

    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        if needs_new {
            if let Some(e) = textures.remove(&elem.buffer_key) {
                gl.delete_texture(e.tex);
            }
            let tex = gl
                .create_texture()
                .map_err(|e| anyhow!("create_texture: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                elem.width,
                elem.height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(pixels),
            );
            textures.insert(
                elem.buffer_key,
                TextureEntry {
                    tex,
                    width: elem.width,
                    height: elem.height,
                },
            );
        } else {
            let entry = textures.get(&elem.buffer_key).unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(entry.tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                elem.width,
                elem.height,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(pixels),
            );
        }
    }
    Ok(())
}

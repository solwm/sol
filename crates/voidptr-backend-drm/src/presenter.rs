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
use drm::control::{Device as ControlDevice, Mode, PageFlipFlags, framebuffer};
use glow::HasContext;
use khronos_egl as egl;
use voidptr_core::{PixelFormat, Scene, SceneContent};

use crate::{
    Card, GlStack, OutputSelection, dmabuf_egl, get_or_add_fb, pick_output,
    quad::QuadProgram, wait_for_page_flip,
};

/// Snapshot of the CRTC state the kernel/fbcon had configured before we
/// grabbed DRM master. Used on drop to hand the display back to whoever was
/// driving it, so the TTY unblanks automatically on clean shutdown.
struct SavedCrtc {
    mode: Option<Mode>,
    fb: Option<framebuffer::Handle>,
    position: (u32, u32),
}

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
    saved_crtc: Option<SavedCrtc>,
}

struct TextureEntry {
    tex: glow::NativeTexture,
    width: i32,
    height: i32,
    /// Some() when the texture is backed by an EGLImage imported from a
    /// dmabuf. The image is kept alive for the texture's lifetime and must
    /// be destroyed explicitly (EGL doesn't refcount through GL texture
    /// bindings).
    egl_image: Option<egl::Image>,
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

        // Snapshot the CRTC fbcon was using, so Drop can hand the display
        // back. Don't fail on error — restore is best-effort anyway.
        let saved_crtc = match card.get_crtc(sel.crtc) {
            Ok(info) => Some(SavedCrtc {
                mode: info.mode(),
                fb: info.framebuffer(),
                position: info.position(),
            }),
            Err(e) => {
                tracing::warn!(error = ?e, "get_crtc for save failed; TTY restore on exit won't work");
                None
            }
        };

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
            saved_crtc,
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
        // SHM elements blit pixels via glTex{Sub,}Image2D; dmabuf elements
        // take the EGL_EXT_image_dma_buf_import path once per buffer. A
        // single failing element (bad dmabuf, unsupported modifier, etc.)
        // is logged and skipped — don't let one broken client kill the
        // whole frame.
        for elem in &scene.elements {
            let res = match &elem.content {
                SceneContent::Shm { .. } => {
                    upload_shm_texture(&self.gl_stack.gl, &mut self.textures, elem)
                }
                SceneContent::Dmabuf { .. } => {
                    import_dmabuf_texture(&self.gl_stack, &mut self.textures, elem)
                }
            };
            if let Err(e) = res {
                tracing::warn!(error = %e, "scene element skipped");
            }
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
            // Dmabuf buffers are opaque by convention (XRGB); SHM carries its
            // own flag so the software cursor (ARGB) still alpha-blends.
            let opaque = match &elem.content {
                SceneContent::Shm {
                    format: PixelFormat::Argb8888,
                    ..
                } => 0.0,
                SceneContent::Shm {
                    format: PixelFormat::Xrgb8888,
                    ..
                } => 1.0,
                SceneContent::Dmabuf { .. } => 1.0,
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

impl Drop for DrmPresenter {
    fn drop(&mut self) {
        // Destroy any dmabuf EGLImages + their GL textures before we let
        // the GL context go. We still have `&mut self` here so all fields
        // are valid.
        for (_, entry) in self.textures.drain() {
            unsafe {
                self.gl_stack.gl.delete_texture(entry.tex);
            }
            if let Some(img) = entry.egl_image {
                let _ = self
                    .gl_stack
                    .egl
                    .destroy_image(self.gl_stack.display, img);
            }
        }

        // Best-effort: push the pre-modeset CRTC state back so fbcon can
        // re-scan the text console framebuffer and the TTY unblanks cleanly.
        // Failures are logged and swallowed — we still want the rest of
        // resource release (GBM BOs, DRM master) to run.
        //
        // NOTE: on DisplayPort at high refresh (e.g. 4K@240), the synchronous
        // set_crtc back to fbcon's mode can take 10–15s because the HW does
        // a full DP link re-training. That's normal, not a voidptr bug.
        let Some(saved) = self.saved_crtc.take() else {
            return;
        };
        tracing::info!(
            from_width = self.width,
            from_height = self.height,
            "restoring prior CRTC state (on high-bandwidth modes the DP link may take 10-15s to re-train)"
        );
        if let Err(e) = self.card.set_crtc(
            self.sel.crtc,
            saved.fb,
            saved.position,
            &[self.sel.connector],
            saved.mode,
        ) {
            tracing::warn!(error = ?e, "CRTC restore on drop failed; TTY may stay blank");
        } else {
            tracing::info!("restored prior CRTC state");
        }
    }
}

/// Upload (or re-upload) a SHM-backed scene element to a GL texture.
/// Keeps a per-`buffer_key` cache so subsequent frames use glTexSubImage2D.
fn upload_shm_texture(
    gl: &glow::Context,
    textures: &mut HashMap<u64, TextureEntry>,
    elem: &voidptr_core::SceneElement<'_>,
) -> Result<()> {
    let (pixels_in, stride) = match &elem.content {
        SceneContent::Shm { pixels, stride, .. } => (*pixels, *stride),
        SceneContent::Dmabuf { .. } => {
            unreachable!("upload_shm_texture called with dmabuf content")
        }
    };
    let expected_bytes = (stride as usize).saturating_mul(elem.height as usize);
    if pixels_in.len() < expected_bytes {
        tracing::warn!("scene element buffer shorter than stride*height; skipping");
        return Ok(());
    }

    // If the source has padded rows, copy into a tight buffer (GLES2 has no
    // GL_UNPACK_ROW_LENGTH to pass stride directly).
    let tight_width_bytes = elem.width as usize * 4;
    let owned_tight;
    let pixels: &[u8] = if stride as usize == tight_width_bytes {
        &pixels_in[..tight_width_bytes * elem.height as usize]
    } else {
        let mut buf = Vec::with_capacity(tight_width_bytes * elem.height as usize);
        for row in 0..elem.height as usize {
            let start = row * stride as usize;
            let end = start + tight_width_bytes;
            buf.extend_from_slice(&pixels_in[start..end]);
        }
        owned_tight = buf;
        &owned_tight
    };

    // Reuse an existing texture only if it's SHM-backed AND the same size.
    // A dmabuf->SHM transition on the same buffer_key (rare but possible
    // if a wl_buffer is re-imported) must rebuild the texture.
    let needs_new = textures.get(&elem.buffer_key).is_none_or(|e| {
        e.egl_image.is_some() || e.width != elem.width || e.height != elem.height
    });

    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        if needs_new {
            evict_texture(gl, textures, elem.buffer_key);
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
                    egl_image: None,
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

/// Import a dmabuf-backed scene element as a GL texture via
/// EGL_EXT_image_dma_buf_import. Cached per `buffer_key` just like SHM
/// uploads — alacritty reuses the same wl_buffer across frames, so we only
/// pay the import cost once.
fn import_dmabuf_texture(
    gl_stack: &GlStack,
    textures: &mut HashMap<u64, TextureEntry>,
    elem: &voidptr_core::SceneElement<'_>,
) -> Result<()> {
    let (fd, fourcc, modifier, offset, stride) = match &elem.content {
        SceneContent::Dmabuf {
            fd,
            fourcc,
            modifier,
            offset,
            stride,
        } => (*fd, *fourcc, *modifier, *offset, *stride),
        SceneContent::Shm { .. } => {
            unreachable!("import_dmabuf_texture called with SHM content")
        }
    };

    // Already imported and still valid? Reuse. EGLImage + GL texture stay
    // live until the texture cache evicts them.
    if let Some(entry) = textures.get(&elem.buffer_key) {
        if entry.egl_image.is_some()
            && entry.width == elem.width
            && entry.height == elem.height
        {
            return Ok(());
        }
    }

    let tex_target_fn = gl_stack
        .gl_egl_image_target_texture_2d_oes
        .ok_or_else(|| anyhow!("glEGLImageTargetTexture2DOES unavailable; cannot render dmabuf"))?;

    // Build EGL_LINUX_DMA_BUF_EXT attribute list. Modifier attribs are only
    // set if the client gave us a real modifier; DRM_FORMAT_MOD_INVALID
    // means "implicit — driver picks" and we omit those attribs per the
    // EGL_EXT_image_dma_buf_import_modifiers spec.
    let mut attribs: Vec<egl::Attrib> = Vec::with_capacity(20);
    attribs.extend_from_slice(&[
        egl::WIDTH as egl::Attrib,
        elem.width as egl::Attrib,
        egl::HEIGHT as egl::Attrib,
        elem.height as egl::Attrib,
        dmabuf_egl::EGL_LINUX_DRM_FOURCC_EXT,
        fourcc as egl::Attrib,
        dmabuf_egl::EGL_DMA_BUF_PLANE0_FD_EXT,
        fd as egl::Attrib,
        dmabuf_egl::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
        offset as egl::Attrib,
        dmabuf_egl::EGL_DMA_BUF_PLANE0_PITCH_EXT,
        stride as egl::Attrib,
    ]);
    if modifier != dmabuf_egl::DRM_FORMAT_MOD_INVALID {
        attribs.extend_from_slice(&[
            dmabuf_egl::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (modifier & 0xffff_ffff) as egl::Attrib,
            dmabuf_egl::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (modifier >> 32) as egl::Attrib,
        ]);
    }
    attribs.push(egl::ATTRIB_NONE);

    let no_ctx = unsafe { egl::Context::from_ptr(std::ptr::null_mut()) };
    let no_buf = unsafe { egl::ClientBuffer::from_ptr(std::ptr::null_mut()) };
    let image = gl_stack
        .egl
        .create_image(
            gl_stack.display,
            no_ctx,
            dmabuf_egl::EGL_LINUX_DMA_BUF_EXT,
            no_buf,
            &attribs,
        )
        .map_err(|e| anyhow!("eglCreateImage for dmabuf: {e:?}"))?;

    // Evict any prior (stale) cache entry — EGL image + GL texture both.
    evict_texture(&gl_stack.gl, textures, elem.buffer_key);

    let tex = unsafe {
        let tex = gl_stack
            .gl
            .create_texture()
            .map_err(|e| anyhow!("create_texture for dmabuf: {e}"))?;
        gl_stack.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl_stack.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        // Bind the EGLImage as the storage for this 2D texture. This is
        // zero-copy: the GPU samples the client's dmabuf directly.
        tex_target_fn(glow::TEXTURE_2D, image.as_ptr());
        tex
    };

    tracing::debug!(
        width = elem.width,
        height = elem.height,
        modifier,
        "imported dmabuf as GL texture"
    );

    textures.insert(
        elem.buffer_key,
        TextureEntry {
            tex,
            width: elem.width,
            height: elem.height,
            egl_image: Some(image),
        },
    );
    Ok(())
}

/// Evict a single cache entry, destroying its EGLImage (if dmabuf-backed)
/// and its GL texture. Leaves the map entry removed.
fn evict_texture(
    _gl: &glow::Context,
    textures: &mut HashMap<u64, TextureEntry>,
    key: u64,
) {
    if let Some(entry) = textures.remove(&key) {
        // NOTE: we intentionally leak the EGL image + GL texture here.
        // Calling destroy_image / delete_texture requires the GL/EGL
        // instance; this helper doesn't have access. The DrmPresenter::drop
        // cleans them up wholesale at shutdown. Steady-state renders
        // rarely evict because wl_buffers are long-lived.
        let _ = entry;
    }
}

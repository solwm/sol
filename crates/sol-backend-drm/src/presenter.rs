//! Frame-by-frame presenter built on the B3 DRM+GBM+EGL+GLES stack.
//!
//! DrmPresenter wraps modeset bring-up and exposes `render_scene`, which the
//! Wayland server calls each time its scene needs to be re-drawn. Each call
//! clears the GL framebuffer, uploads every scene element to a cached
//! `NativeTexture`, draws textured quads, swaps the EGL surface, schedules
//! a drmModePageFlip, and blocks on the DRM fd for the flip-complete event.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use drm::control::{Device as ControlDevice, Mode, PageFlipFlags, framebuffer, property};
use glow::HasContext;
use khronos_egl as egl;
use sol_core::{PixelFormat, Scene, SceneContent};

use crate::{
    Card, GlStack, OutputSelection, dmabuf_egl, get_or_add_fb, pick_output,
    quad::{QuadProgram, SolidProgram},
};

/// Snapshot of the CRTC state the kernel/fbcon had configured before we
/// grabbed DRM master. Used on drop to hand the display back to whoever was
/// driving it, so the TTY unblanks automatically on clean shutdown.
struct SavedCrtc {
    mode: Option<Mode>,
    fb: Option<framebuffer::Handle>,
    position: (u32, u32),
}

/// GL extension constant for the external-image texture target. Used for
/// dmabuf-imported textures via GL_OES_EGL_image_external.
const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8D65;

pub struct DrmPresenter {
    card: Card,
    gl_stack: GlStack,
    sel: OutputSelection,
    fb_cache: HashMap<usize, framebuffer::Handle>,
    scanned_out: Option<gbm::BufferObject<()>>,
    /// GBM BO submitted for the most recent page flip, held here until
    /// the flip-complete event fires and the buffer becomes the active
    /// scanout. Without this, GBM would release the BO back to its
    /// pool while the display is still sampling from it.
    pending_bo: Option<gbm::BufferObject<()>>,
    /// True while a page flip is in flight. `render_scene` checks this
    /// and skips rendering if set — the kernel rejects a second
    /// page_flip on the same CRTC before the first completes, and
    /// we'd rather just drop the extra frame than error out of the
    /// render loop.
    pending_flip: bool,
    /// Shader program for SHM textures: `sampler2D` on `GL_TEXTURE_2D`.
    quad: QuadProgram,
    /// Shader program for dmabuf textures: `samplerExternalOES` on
    /// `GL_TEXTURE_EXTERNAL_OES`. Required because Mesa often returns
    /// external-only EGLImages for dmabuf imports — binding those to
    /// GL_TEXTURE_2D silently yields all-zero samples.
    quad_external: QuadProgram,
    /// Flat-color shader for border rects, debug overlays, etc.
    solid: SolidProgram,
    /// GPU textures keyed by Scene::buffer_key. Kept across frames so
    /// uploads are glTexSubImage2D after the first sight.
    textures: HashMap<u64, TextureEntry>,
    width: u32,
    height: u32,
    saved_crtc: Option<SavedCrtc>,
    /// Cached handle for the connector's DPMS property. Resolved
    /// lazily on the first `set_dpms` call (via a linear scan of
    /// connector properties) and reused for every subsequent call,
    /// so blank/unblank transitions are a single ioctl.
    dpms_prop: Option<property::Handle>,
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
    /// GL texture target this entry is bound to. SHM uploads use
    /// GL_TEXTURE_2D; dmabuf imports use GL_TEXTURE_EXTERNAL_OES.
    target: u32,
}

impl DrmPresenter {
    /// Construct from a `Card` already wrapping a DRM fd. sol's main
    /// binary uses this path — libseat hands us the fd; we wrap it via
    /// `Card::from_fd` before calling here.
    ///
    /// No explicit `acquire_master_lock` call here: the fd returned by
    /// libseat's `TakeDevice` (logind) or seatd is already master,
    /// because the daemon set master on it as part of granting the
    /// active session's device. Calling DRM_IOCTL_SET_MASTER again
    /// from our process-side fails with EACCES — the kernel only
    /// allows the currently-active logind session to perform it, and
    /// from a child process's perspective of the daemon-opened fd
    /// that check doesn't match. Operations that actually need master
    /// (set_crtc, page_flip) work fine because the fd itself is
    /// master-flagged.
    pub fn from_card(card: Card) -> Result<Self> {
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
        let quad_external = crate::quad::build_external(&gl_stack.gl)?;
        let solid = crate::quad::build_solid(&gl_stack.gl)?;

        let mut presenter = Self {
            card,
            gl_stack,
            sel,
            fb_cache: HashMap::new(),
            scanned_out: None,
            pending_bo: None,
            pending_flip: false,
            quad,
            quad_external,
            solid,
            textures: HashMap::new(),
            width,
            height,
            saved_crtc,
            dpms_prop: None,
        };
        presenter.initial_modeset()?;
        Ok(presenter)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Selected mode's vrefresh in Hz. Returned as u32 to match what
    /// drm-rs exposes; the Wayland server multiplies by 1000 when
    /// advertising `wl_output.mode` (which wants milli-Hz).
    pub fn refresh_hz(&self) -> u32 {
        self.sel.mode.vrefresh()
    }

    /// Called from the session's Disable handler. libseat/logind
    /// revokes DRM master on our fd as part of the session-deactivate
    /// path, so we don't call DROP_MASTER ourselves — the kernel has
    /// already flipped the fd out of master mode and issuing another
    /// DROP would return EINVAL. Kept as a stub for symmetry with
    /// `reacquire_master` and so future backends (direct VT mgmt)
    /// can hook in.
    pub fn drop_master(&self) {
        tracing::info!("session disabled — DRM master released by logind");
    }

    /// Called on libseat Enable: re-apply our modeset so the screen
    /// reflects our scene rather than whatever drove the display
    /// during the disabled window (fbcon or another session). No
    /// explicit acquire_master_lock — libseat/logind restores master
    /// on our fd as part of making the session active.
    pub fn reacquire_master(&mut self) -> Result<()> {
        // Re-arm the CRTC with our most recent scanned-out buffer so
        // fbcon's takeover during the disabled window is overwritten.
        if let Some(bo) = self.scanned_out.as_ref() {
            let fb = crate::get_or_add_fb(&self.card, bo, &mut self.fb_cache)?;
            self.card
                .set_crtc(
                    self.sel.crtc,
                    Some(fb),
                    (0, 0),
                    &[self.sel.connector],
                    Some(self.sel.mode),
                )
                .context("re-apply modeset on Enable")?;
        }
        tracing::info!("DRM master reacquired (session enabled)");
        Ok(())
    }

    /// Drive the connector's DPMS property to ON (blank=false) or
    /// OFF (blank=true). Used by the idle-blank path in the
    /// compositor — the monitor powers down (real power save, not
    /// just a black frame) and wakes back on the next input.
    ///
    /// The DPMS property handle is resolved once from the connector's
    /// property table and cached on the presenter; subsequent calls
    /// are a single `drmModeConnectorSetProperty` ioctl. If the
    /// connector doesn't advertise a DPMS property (rare — nouveau
    /// on some old cards, or atomic-only drivers that expose DPMS
    /// only through the CRTC's `ACTIVE` property), we log and skip
    /// rather than erroring; the screen just stays on.
    pub fn set_dpms(&mut self, blank: bool) -> Result<()> {
        if self.dpms_prop.is_none() {
            let props = self
                .card
                .get_properties(self.sel.connector)
                .context("get connector properties for DPMS")?;
            for (handle, _) in props.iter() {
                let info = match self.card.get_property(*handle) {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                if info.name().to_bytes() == b"DPMS" {
                    self.dpms_prop = Some(*handle);
                    break;
                }
            }
        }
        let Some(handle) = self.dpms_prop else {
            tracing::warn!("connector has no DPMS property; screen stays on");
            return Ok(());
        };
        // Legacy DPMS values: 0 = ON, 1 = STANDBY, 2 = SUSPEND, 3 = OFF.
        // We use only the two extremes — STANDBY/SUSPEND's actual
        // behavior varies per driver and doesn't buy us anything
        // over OFF for an idle blank.
        let value = if blank { 3 } else { 0 };
        self.card
            .set_property(self.sel.connector, handle, value)
            .context("drm set DPMS")?;
        tracing::info!(dpms_off = blank, "set DPMS");
        Ok(())
    }

    /// Drop the cached texture + EGLImage for a given buffer_key.
    /// Callers should do this when a client destroys the backing
    /// wl_buffer; otherwise dmabuf EGLImages accumulate across every
    /// resize and leak GPU memory until the compositor exits.
    pub fn evict_texture(&mut self, key: u64) {
        let Some(entry) = self.textures.remove(&key) else {
            return;
        };
        unsafe {
            self.gl_stack.gl.delete_texture(entry.tex);
        }
        if let Some(img) = entry.egl_image {
            if let Err(e) = self
                .gl_stack
                .egl
                .destroy_image(self.gl_stack.display, img)
            {
                tracing::warn!(error = ?e, "destroy_image on eviction failed");
            }
        }
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

    /// Redraw the given scene and submit a page flip. Returns as soon
    /// as the flip is submitted — the flip-complete event arrives
    /// asynchronously on the DRM fd, driven by a calloop source in
    /// the main loop; the source calls `flip_complete()` to settle
    /// buffer state and fire frame callbacks. If a flip is still in
    /// flight when this is called, we silently drop the frame rather
    /// than error — the client will get a frame callback once the
    /// current flip completes and can submit again then.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<()> {
        if self.pending_flip {
            return Ok(());
        }
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

        // Pass 2: draw. SHM and dmabuf elements use different shader
        // programs + texture targets; switch between them per element.
        let gl = &self.gl_stack.gl;
        unsafe {
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.02, 0.02, 0.04, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.active_texture(glow::TEXTURE0);
        }

        let mut active_program: Option<u32> = None;
        for elem in &scene.elements {
            let Some(entry) = self.textures.get(&elem.buffer_key) else {
                continue;
            };
            // Pick the shader + VBO set matching this element's target.
            let prog = if entry.target == GL_TEXTURE_EXTERNAL_OES {
                &self.quad_external
            } else {
                &self.quad
            };
            let prog_id = prog.program.0.get();
            if active_program != Some(prog_id) {
                unsafe {
                    gl.use_program(Some(prog.program));
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(prog.vbo));
                    gl.enable_vertex_attrib_array(0);
                    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
                    gl.uniform_1_i32(Some(&prog.u_tex), 0);
                }
                active_program = Some(prog_id);
            }

            // Output rect size = dst_width/height when set (the tile
            // / layer rect the compositor assigned), else fall back
            // to the buffer's intrinsic size (subsurfaces / cursor).
            // UV in the vertex shader samples whatever sub-rect the
            // client declared via wp_viewport.set_source, so the
            // output quad can be bigger, smaller, or cropped
            // relative to the actual buffer.
            let dst_w = if elem.dst_width > 0.0 {
                elem.dst_width
            } else {
                elem.width as f32
            };
            let dst_h = if elem.dst_height > 0.0 {
                elem.dst_height
            } else {
                elem.height as f32
            };
            let fb_w = w as f32;
            let fb_h = h as f32;
            let x0 = (elem.x / fb_w) * 2.0 - 1.0;
            let y0 = 1.0 - ((elem.y + dst_h) / fb_h) * 2.0;
            let rw = dst_w / fb_w * 2.0;
            let rh = dst_h / fb_h * 2.0;
            let opaque = match &elem.content {
                SceneContent::Shm {
                    format: PixelFormat::Argb8888,
                    ..
                } => 0.0,
                SceneContent::Shm {
                    format: PixelFormat::Xrgb8888,
                    ..
                } => 1.0,
                // DRM fourcc is a 4-char little-endian ASCII code; the
                // first byte encodes whether the alpha channel is
                // meaningful. 'X' = ignore alpha (opaque), 'A' = use
                // alpha. Chrome hands us ARGB dmabufs for popups where
                // the shadow/border regions have alpha=0 and RGB≈0 —
                // forcing opaque there rendered those as solid black.
                SceneContent::Dmabuf { fourcc, .. } => {
                    if (*fourcc & 0xFF) as u8 == b'X' { 1.0 } else { 0.0 }
                }
            };
            unsafe {
                gl.bind_texture(entry.target, Some(entry.tex));
                gl.uniform_4_f32(Some(&prog.u_rect), x0, y0, rw, rh);
                // UV sub-rect: set by scene_from_buffers from the
                // surface's wp_viewport.set_source (or (0,0,1,1) for
                // the full texture).
                gl.uniform_4_f32(
                    Some(&prog.u_uv),
                    elem.uv_x,
                    elem.uv_y,
                    elem.uv_w,
                    elem.uv_h,
                );
                gl.uniform_1_f32(Some(&prog.u_opaque), opaque);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                gl.bind_texture(entry.target, None);
            }
        }

        // Border / overlay pass: flat-colored rects drawn on top of
        // the textured pass, below the cursor (which gets rendered
        // inside the textured pass because it's just another scene
        // element). Separate shader program because u_tex / u_opaque
        // aren't meaningful here.
        if !scene.borders.is_empty() {
            unsafe {
                gl.use_program(Some(self.solid.program));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.solid.vbo));
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            }
            for b in &scene.borders {
                let x0 = (b.x as f32 / w as f32) * 2.0 - 1.0;
                let y0 = 1.0 - ((b.y + b.h) as f32 / h as f32) * 2.0;
                let rw = b.w as f32 / w as f32 * 2.0;
                let rh = b.h as f32 / h as f32 * 2.0;
                unsafe {
                    gl.uniform_4_f32(Some(&self.solid.u_rect), x0, y0, rw, rh);
                    gl.uniform_4_f32(
                        Some(&self.solid.u_color),
                        b.rgba[0], b.rgba[1], b.rgba[2], b.rgba[3],
                    );
                    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                }
            }
        }

        unsafe {
            gl.disable_vertex_attrib_array(0);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
            gl.disable(glow::BLEND);
        }

        self.submit_flip()
    }

    /// Swap EGL front/back, lock the resulting GBM buffer, register a
    /// drm framebuffer for it, and ask DRM to page-flip to it on the
    /// next vblank. Non-blocking: the caller (render_tick) returns to
    /// the event loop as soon as the flip is submitted, and the
    /// flip-complete event arrives on the DRM fd and is handled by
    /// the calloop source in the main loop.
    fn submit_flip(&mut self) -> Result<()> {
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
        self.pending_bo = Some(next_bo);
        self.pending_flip = true;
        Ok(())
    }

    pub fn is_pending_flip(&self) -> bool {
        self.pending_flip
    }

    /// Drain outstanding DRM events from the card's fd and, for each
    /// page-flip-complete event, settle buffer state by calling
    /// `flip_complete`. Returns true if at least one PageFlip event
    /// was processed, so the caller (the calloop source on the DRM
    /// fd) knows it should fire the frame callbacks queued for this
    /// frame. Kept as a single method so sol-wayland doesn't
    /// need to depend on drm-rs directly.
    pub fn drain_events(&mut self) -> Result<bool> {
        let mut saw_flip = false;
        for ev in self.card.receive_events()? {
            if matches!(ev, drm::control::Event::PageFlip(_)) {
                saw_flip = true;
            }
        }
        if saw_flip {
            self.flip_complete();
        }
        Ok(saw_flip)
    }

    /// Settle buffer state after a page-flip-complete. The BO we
    /// submitted becomes the active scanout; the one that WAS the
    /// active scanout is dropped, returning it to GBM's pool. Clears
    /// the pending flag so the next render_scene submits a new flip.
    pub fn flip_complete(&mut self) {
        if let Some(bo) = self.pending_bo.take() {
            self.scanned_out = Some(bo);
        }
        self.pending_flip = false;
    }

    /// DRM file descriptor for calloop registration. Borrowed for the
    /// lifetime of the presenter; the card's File owns the fd.
    pub fn drm_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.card.as_fd()
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
        // a full DP link re-training. That's normal, not a sol bug.
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
    elem: &sol_core::SceneElement<'_>,
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
                    target: glow::TEXTURE_2D,
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
    elem: &sol_core::SceneElement<'_>,
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
        // Clear any stale error before the sequence so we only report
        // failures caused by our own calls.
        while gl_stack.gl.get_error() != glow::NO_ERROR {}

        let tex = gl_stack
            .gl
            .create_texture()
            .map_err(|e| anyhow!("create_texture for dmabuf: {e}"))?;
        // Bind to GL_TEXTURE_EXTERNAL_OES: Mesa returns external-only
        // EGLImages for most dmabuf imports (even LINEAR ARGB8888).
        // Binding those to plain GL_TEXTURE_2D succeeds silently but
        // sampling yields all-zeros — exactly what we were seeing as
        // black alacritty windows with the old path.
        gl_stack
            .gl
            .bind_texture(GL_TEXTURE_EXTERNAL_OES, Some(tex));
        gl_stack.gl.tex_parameter_i32(
            GL_TEXTURE_EXTERNAL_OES,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            GL_TEXTURE_EXTERNAL_OES,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            GL_TEXTURE_EXTERNAL_OES,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl_stack.gl.tex_parameter_i32(
            GL_TEXTURE_EXTERNAL_OES,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        tex_target_fn(GL_TEXTURE_EXTERNAL_OES, image.as_ptr());
        let err = gl_stack.gl.get_error();
        if err != glow::NO_ERROR {
            tracing::warn!(
                err = format!("0x{:04x}", err),
                "glEGLImageTargetTexture2DOES(EXTERNAL_OES) failed"
            );
        }
        tex
    };

    tracing::debug!(
        width = elem.width,
        height = elem.height,
        modifier,
        fourcc = format!("{:#x}", fourcc),
        "imported dmabuf as GL_TEXTURE_EXTERNAL_OES"
    );

    textures.insert(
        elem.buffer_key,
        TextureEntry {
            tex,
            width: elem.width,
            height: elem.height,
            egl_image: Some(image),
            target: GL_TEXTURE_EXTERNAL_OES,
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

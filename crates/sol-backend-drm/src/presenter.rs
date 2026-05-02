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
use std::time::Instant;

use sol_core::{PixelFormat, RenderTiming, Scene, SceneBorder, SceneContent};

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
    /// 5×5 box-blur ping-pong program. Used to soften the wallpaper
    /// behind inactive toplevels.
    blur: crate::quad::BlurProgram,
    /// Shader for the final blurred-backdrop draw onto the screen
    /// FBO at a window's rect.
    backdrop: crate::quad::BackdropProgram,
    /// FBOs + colour textures used for the inactive-window blur path.
    /// Three full-resolution FBOs are pre-allocated at presenter-init
    /// (mode change requires a rebuild — TODO when live mode-set
    /// lands): one to receive the captured screen contents, two to
    /// ping-pong the blur passes between. Full-res is overkill on
    /// paper but a 4K box-blur on a discrete GPU is well under a
    /// frame's worth of budget, and the UV math stays trivial. If
    /// this ever shows up in a profile, switch capture_tex to full
    /// res with ping/pong at half and have the first blur pass do
    /// the downsample.
    blur_fbos: Option<BlurFbos>,
    /// Per-frame flag set by `prepare_blur_backdrop` so we only blur
    /// once even if multiple inactive-window backdrop elements appear
    /// in the same scene. Reset at the top of every `render_scene`.
    /// Also pre-set to `true` (without running anything) when the
    /// blur-cache check decides the previous frame's blur output is
    /// still valid — see `last_bg_sig` / `last_blur_params`.
    blur_ready_this_frame: bool,
    /// Hash of the last frame's background scene elements
    /// (buffer keys + on-screen rects). When the current frame's
    /// signature matches and the blur params haven't changed, the
    /// blur FBOs from the previous frame are still valid — we skip
    /// both the bg pre-pass *and* the blur passes. This is the big
    /// win for "active client but static wallpaper" cases (cmatrix
    /// in alacritty, scrolling browser, etc.) where the scene is
    /// changing every frame but the blur *input* is not.
    last_bg_sig: u64,
    /// Last frame's `(passes, radius_bits)`. `radius_bits` is the
    /// f32 radius reinterpreted as u32 so we can equality-test
    /// without worrying about NaN. Changing either invalidates the
    /// blur cache.
    last_blur_params: (u32, u32),
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

/// Three FBOs at half output resolution used by the inactive-window
/// frosted-backdrop pipeline. `capture` receives a copy of the screen
/// just after the wallpaper / lower layers have been drawn; `ping`
/// and `pong` are ping-ponged between blur passes. Each FBO has one
/// `GL_RGBA8` colour attachment, no depth/stencil — we don't need
/// either for a 2D scene.
struct BlurFbos {
    /// Full output resolution. The bg pre-pass renders into
    /// `capture_*` at this size so the on-screen blit (and the first
    /// blur sample) operate on the same spatial reference as the
    /// rest of the scene.
    full_width: i32,
    full_height: i32,
    /// Half-resolution dimensions used by `ping_*` / `pong_*`. The
    /// blur passes ping-pong here to cut per-pass cost by ~4× —
    /// rotating between 5×5 box-blur samples at half resolution
    /// produces a result that's visually indistinguishable from
    /// full-res after upscale through the texture's bilinear filter.
    /// Standard dual-Kawase trick.
    blur_width: i32,
    blur_height: i32,
    /// Render target for the background pass. The presenter draws the
    /// `Scene::background_count` leading elements (wallpaper + bottom
    /// layer-shell surfaces) into here instead of the default FBO,
    /// then blits this onto the screen via a fullscreen quad. That
    /// way the capture is *guaranteed* to have valid contents we can
    /// sample — going through `glCopyTexSubImage2D` from the
    /// GBM-backed default framebuffer was producing zero-RGB on at
    /// least one Mesa configuration the user reported.
    capture_fbo: glow::NativeFramebuffer,
    capture_tex: glow::NativeTexture,
    ping_fbo: glow::NativeFramebuffer,
    ping_tex: glow::NativeTexture,
    pong_fbo: glow::NativeFramebuffer,
    pong_tex: glow::NativeTexture,
}

impl BlurFbos {
    fn new(gl: &glow::Context, width: i32, height: i32) -> Result<Self> {
        let blur_width = (width / 2).max(1);
        let blur_height = (height / 2).max(1);
        unsafe {
            let alloc_tex = |gl: &glow::Context, w: i32, h: i32| -> Result<glow::NativeTexture> {
                let tex = gl
                    .create_texture()
                    .map_err(|e| anyhow!("create blur texture: {e}"))?;
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
                    w,
                    h,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    None,
                );
                Ok(tex)
            };
            let attach_fbo = |gl: &glow::Context, tex: glow::NativeTexture| -> Result<glow::NativeFramebuffer> {
                let fbo = gl
                    .create_framebuffer()
                    .map_err(|e| anyhow!("create blur fbo: {e}"))?;
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(tex),
                    0,
                );
                let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                if status != glow::FRAMEBUFFER_COMPLETE {
                    return Err(anyhow!("blur fbo incomplete: 0x{status:x}"));
                }
                Ok(fbo)
            };
            // Capture stays full-res so the bg pre-pass renders the
            // wallpaper at the same scale as the screen.
            let capture_tex = alloc_tex(gl, width, height)?;
            let capture_fbo = attach_fbo(gl, capture_tex)?;
            // Ping/pong are half-res — each pass costs ~1/4 of full
            // res, and the eye doesn't see the resolution loss after
            // the blur smooths things out.
            let ping_tex = alloc_tex(gl, blur_width, blur_height)?;
            let ping_fbo = attach_fbo(gl, ping_tex)?;
            let pong_tex = alloc_tex(gl, blur_width, blur_height)?;
            let pong_fbo = attach_fbo(gl, pong_tex)?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            Ok(Self {
                full_width: width,
                full_height: height,
                blur_width,
                blur_height,
                capture_fbo,
                capture_tex,
                ping_fbo,
                ping_tex,
                pong_fbo,
                pong_tex,
            })
        }
    }
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
    pub fn from_card(
        card: Card,
        mode_pref: Option<crate::ModePreference>,
    ) -> Result<Self> {
        let sel = pick_output(&card, mode_pref)?;
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
        let blur = crate::quad::build_blur(&gl_stack.gl)?;
        let backdrop = crate::quad::build_backdrop(&gl_stack.gl)?;
        // Half-resolution FBOs for the blur pipeline. Failure is
        // non-fatal — log and disable blur for this session, the
        // compositor still renders without the frosted-glass effect.
        let blur_fbos =
            match BlurFbos::new(&gl_stack.gl, width as i32, height as i32) {
                Ok(f) => Some(f),
                Err(e) => {
                    tracing::warn!(error = %e, "blur FBO allocation failed; inactive-window blur disabled");
                    None
                }
            };

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
            blur,
            backdrop,
            blur_fbos,
            blur_ready_this_frame: false,
            last_bg_sig: 0,
            last_blur_params: (0, 0),
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

        // On wake, also re-apply the modeset using the last bo that
        // was on screen. Some monitor / driver combinations (DP +
        // long DPMS_OFF in particular, especially with nvidia)
        // don't re-train the link on a bare DPMS=ON; the display
        // stays at "no signal" until a real modeset re-arms the
        // CRTC. Without this, the user's only recovery is a VT
        // round-trip (Ctrl+Alt+F2 → F3), since libseat's
        // drop-/reacquire-master cycle implicitly re-modesets via
        // `reacquire_master`. set_crtc with the already-active
        // mode + connector + fb is idempotent on hardware that
        // doesn't need it, so adding it costs nothing for the
        // happy path.
        if !blank {
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
                    .context("re-apply modeset on DPMS wake")?;
            }
        }

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
    pub fn render_scene(
        &mut self,
        scene: &Scene,
        timing: &mut RenderTiming,
    ) -> Result<()> {
        if self.pending_flip {
            return Ok(());
        }
        let w = self.width as i32;
        let h = self.height as i32;
        let t_textures = Instant::now();
        // Blur cache check: if the bg slice and the blur params
        // haven't changed since last frame, the blur FBOs from the
        // previous render still hold the right result — short-circuit
        // both the bg pre-pass *and* the actual blur passes by
        // marking blur_ready_this_frame up front. This is the
        // hot-path for "client churns its own buffer but the
        // wallpaper is static" (cmatrix etc.) — exactly the case
        // where the blur cost is otherwise wasted.
        //
        // Signature includes buffer_key + on-screen rect of every
        // background element so a moved / resized wallpaper layer
        // also invalidates. Blur params: (passes, radius_bits).
        let bg_sig = compute_bg_signature(
            &scene.elements[..scene.background_count],
        );
        let scene_blur_params = scene
            .elements
            .iter()
            .find_map(|e| match &e.content {
                SceneContent::BlurredBackdrop { passes, radius } => {
                    Some((*passes, radius.to_bits()))
                }
                _ => None,
            })
            .unwrap_or((0, 0));
        let scene_has_backdrop = scene_blur_params.0 > 0;
        let blur_cache_valid = scene_has_backdrop
            && bg_sig == self.last_bg_sig
            && scene_blur_params == self.last_blur_params;
        // When the cache is valid prepare_blur_backdrop becomes a
        // no-op for the rest of the frame — the previous frame's
        // ping/pong textures still hold the result, and
        // final_blur_tex(passes) will return the right one.
        self.blur_ready_this_frame = blur_cache_valid;
        self.last_bg_sig = bg_sig;
        self.last_blur_params = scene_blur_params;

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
                // Backdrops have no client-side buffer; they sample
                // from a presenter-owned FBO instead. Nothing to
                // upload.
                SceneContent::BlurredBackdrop { .. } => Ok(()),
            };
            if let Err(e) = res {
                tracing::warn!(error = %e, "scene element skipped");
            }
        }
        timing.textures_ns += t_textures.elapsed().as_nanos() as u64;
        let t_blur = Instant::now();

        // Pre-pass: render the background slice (wallpaper + bottom
        // layer-shell + their subsurfaces) into the blur capture FBO
        // first, so the blur pipeline samples a render target we
        // *wrote into* — not one we copy from the GBM-backed default
        // framebuffer (which has produced zero-RGB output on at least
        // one Mesa configuration). The same elements get drawn again
        // into the default FBO by the main loop below; the cost is
        // small (a handful of fullscreen quads) and the correctness
        // win is large.
        //
        // Skipped when there's no backdrop in the scene at all (no
        // inactive windows, or inactive_blur=off — capture_tex would
        // never be sampled), and skipped when the cache is valid
        // (capture_tex still holds the right content from a previous
        // frame).
        if let Some(fbos) = self.blur_fbos.as_ref() {
            if scene.background_count > 0
                && scene_has_backdrop
                && !blur_cache_valid
            {
                unsafe {
                    let gl = &self.gl_stack.gl;
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbos.capture_fbo));
                    gl.viewport(0, 0, fbos.full_width, fbos.full_height);
                    gl.clear_color(0.02, 0.02, 0.04, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT);
                    gl.enable(glow::BLEND);
                    gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                    gl.active_texture(glow::TEXTURE0);
                }
                let mut bg_active_program: Option<u32> = None;
                for elem in &scene.elements[..scene.background_count] {
                    self.draw_textured_element(
                        elem,
                        fbos.full_width as f32,
                        fbos.full_height as f32,
                        &mut bg_active_program,
                    );
                }
                unsafe {
                    let gl = &self.gl_stack.gl;
                    gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
            }
        }

        timing.blur_ns += t_blur.elapsed().as_nanos() as u64;
        let t_draw = Instant::now();

        // Pass 2: draw. SHM and dmabuf elements use different shader
        // programs + texture targets; switch between them per element.
        unsafe {
            let gl = &self.gl_stack.gl;
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.02, 0.02, 0.04, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.active_texture(glow::TEXTURE0);
        }

        let mut active_program: Option<u32> = None;
        let border_anchor = scene.border_anchor.min(scene.elements.len());
        let mut borders_drawn = false;
        for (idx, elem) in scene.elements.iter().enumerate() {
            // Inject the border pass at the configured z-anchor —
            // i.e. right between the tile pass and the dialogs /
            // popups / cursor that should sit visually above the
            // focus rings. The shader switch invalidates the
            // textured-element program tracking, so reset it.
            if !borders_drawn && idx == border_anchor && !scene.borders.is_empty() {
                self.draw_solid_borders(&scene.borders, w as f32, h as f32);
                borders_drawn = true;
                active_program = None;
            }
            // Frosted backdrop: capture-and-blur the screen-so-far
            // (lazily, once per frame) and draw the blurred FBO
            // sampled at this element's screen rect. Different
            // shader pipeline from the textured quad path, so
            // active_program is forced to refresh next iteration.
            if let SceneContent::BlurredBackdrop { passes, radius } = &elem.content {
                if let Err(e) = self.draw_blurred_backdrop(elem, *passes, *radius) {
                    tracing::warn!(error = %e, "backdrop draw skipped");
                }
                active_program = None;
                continue;
            }
            let Some(entry) = self.textures.get(&elem.buffer_key) else {
                continue;
            };
            let gl = &self.gl_stack.gl;
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
                // The BlurredBackdrop branch up the loop already
                // `continue`d past this point; this arm is just to
                // satisfy exhaustiveness.
                SceneContent::BlurredBackdrop { .. } => unreachable!(),
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
                gl.uniform_1_f32(Some(&prog.u_alpha), elem.alpha.clamp(0.0, 1.0));
                gl.uniform_2_f32(Some(&prog.u_size), dst_w, dst_h);
                gl.uniform_1_f32(
                    Some(&prog.u_radius),
                    elem.corner_radius.max(0.0),
                );
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                gl.bind_texture(entry.target, None);
            }
        }

        // Border pass fallback: if the anchor was past the last
        // element (or callers didn't set one — `usize::MAX`), draw
        // borders here at the very end. Matches the pre-anchor
        // behaviour for any caller that builds a Scene with default
        // `border_anchor`.
        if !borders_drawn && !scene.borders.is_empty() {
            self.draw_solid_borders(&scene.borders, w as f32, h as f32);
        }

        unsafe {
            let gl = &self.gl_stack.gl;
            gl.disable_vertex_attrib_array(0);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
            gl.disable(glow::BLEND);
        }
        timing.draw_ns += t_draw.elapsed().as_nanos() as u64;
        let t_present = Instant::now();
        let r = self.submit_flip();
        timing.present_ns += t_present.elapsed().as_nanos() as u64;
        r
    }

    /// Draw the solid-color border pass into the currently-bound
    /// framebuffer. Used at whatever z-anchor the scene specifies
    /// (between tiles and dialogs, typically) so the focus ring
    /// doesn't paint over a floating window. Caller is responsible
    /// for resetting `active_program` after this method returns,
    /// since it switches shaders.
    fn draw_solid_borders(
        &self,
        borders: &[SceneBorder],
        fb_w: f32,
        fb_h: f32,
    ) {
        let gl = &self.gl_stack.gl;
        unsafe {
            gl.use_program(Some(self.solid.program));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.solid.vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
        }
        for b in borders {
            let x0 = (b.x / fb_w) * 2.0 - 1.0;
            let y0 = 1.0 - ((b.y + b.h) / fb_h) * 2.0;
            let rw = b.w / fb_w * 2.0;
            let rh = b.h / fb_h * 2.0;
            unsafe {
                gl.uniform_4_f32(Some(&self.solid.u_rect), x0, y0, rw, rh);
                gl.uniform_4_f32(
                    Some(&self.solid.u_color),
                    b.rgba[0], b.rgba[1], b.rgba[2], b.rgba[3],
                );
                gl.uniform_2_f32(Some(&self.solid.u_size), b.w, b.h);
                gl.uniform_1_f32(
                    Some(&self.solid.u_radius),
                    b.corner_radius.max(0.0),
                );
                gl.uniform_1_f32(
                    Some(&self.solid.u_border_width),
                    b.border_width.max(0.0),
                );
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        }
    }

    /// Draw one textured-quad scene element into the currently-bound
    /// framebuffer, using `fb_w` / `fb_h` to compute NDC coordinates.
    /// `active_program` is the caller's tracking variable for which
    /// shader is currently bound — this method updates it lazily so
    /// consecutive elements with the same target reuse the binding.
    /// Backdrops, missing textures, and elements whose buffer hasn't
    /// uploaded yet are silently skipped.
    fn draw_textured_element(
        &self,
        elem: &sol_core::SceneElement<'_>,
        fb_w: f32,
        fb_h: f32,
        active_program: &mut Option<u32>,
    ) {
        if matches!(elem.content, SceneContent::BlurredBackdrop { .. }) {
            return;
        }
        let Some(entry) = self.textures.get(&elem.buffer_key) else {
            return;
        };
        let gl = &self.gl_stack.gl;
        let prog = if entry.target == GL_TEXTURE_EXTERNAL_OES {
            &self.quad_external
        } else {
            &self.quad
        };
        let prog_id = prog.program.0.get();
        if *active_program != Some(prog_id) {
            unsafe {
                gl.use_program(Some(prog.program));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(prog.vbo));
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
                gl.uniform_1_i32(Some(&prog.u_tex), 0);
            }
            *active_program = Some(prog_id);
        }
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
            SceneContent::Dmabuf { fourcc, .. } => {
                if (*fourcc & 0xFF) as u8 == b'X' { 1.0 } else { 0.0 }
            }
            SceneContent::BlurredBackdrop { .. } => unreachable!(),
        };
        unsafe {
            gl.bind_texture(entry.target, Some(entry.tex));
            gl.uniform_4_f32(Some(&prog.u_rect), x0, y0, rw, rh);
            gl.uniform_4_f32(
                Some(&prog.u_uv),
                elem.uv_x,
                elem.uv_y,
                elem.uv_w,
                elem.uv_h,
            );
            gl.uniform_1_f32(Some(&prog.u_opaque), opaque);
            gl.uniform_1_f32(Some(&prog.u_alpha), elem.alpha.clamp(0.0, 1.0));
            gl.uniform_2_f32(Some(&prog.u_size), dst_w, dst_h);
            gl.uniform_1_f32(
                Some(&prog.u_radius),
                elem.corner_radius.max(0.0),
            );
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_texture(entry.target, None);
        }
    }

    /// Capture the default framebuffer (which has the wallpaper / lower
    /// layers drawn but no toplevels yet, since collect_scene emits in
    /// back-to-front order and inactive backdrops are emitted before
    /// their windows) into `capture_tex`, then ping-pong `passes` rounds
    /// of 5×5 box blur between the ping/pong FBOs. Leaves the final
    /// blurred result in `ping_tex` (or `pong_tex`, depending on parity
    /// — caller asks `final_blur_tex`).
    ///
    /// Idempotent within a frame: `blur_ready_this_frame` short-circuits
    /// re-runs so multiple inactive windows share one blur.
    fn prepare_blur_backdrop(&mut self, passes: u32, radius: f32) -> Result<()> {
        if self.blur_ready_this_frame {
            return Ok(());
        }
        let Some(fbos) = self.blur_fbos.as_ref() else {
            anyhow::bail!("blur FBOs unavailable");
        };
        let gl = &self.gl_stack.gl;
        unsafe {
            // capture_tex is already populated by the bg pre-pass at
            // the top of render_scene — no copy from the default
            // framebuffer needed here. Just run the blur passes
            // ping-pong style.
            //
            // Set up blur draw state. Same VBO + attrib layout as the
            // textured quad, so we can switch programs with minimal
            // setup.
            gl.use_program(Some(self.blur.program));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.blur.vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.blur.u_tex), 0);
            // Blur is solid-source over solid-source — no need to
            // blend with whatever was in the destination FBO.
            gl.disable(glow::BLEND);

            // Each pass renders a fullscreen quad; rect = full NDC
            // clip space, uv = full texture.
            gl.uniform_4_f32(Some(&self.blur.u_rect), -1.0, -1.0, 2.0, 2.0);
            gl.uniform_4_f32(Some(&self.blur.u_uv), 0.0, 0.0, 1.0, 1.0);
            // Per-pass kernel reach: shader's hardcoded -2..2 offsets
            // are multiplied by u_texel, so scaling u_texel by `radius`
            // widens (or shrinks) each pass's effective area without
            // changing sample count. The first pass samples the
            // full-res capture and writes into the half-res ping FBO,
            // so its texel stride is twice as effective in source
            // pixels — the downsample is "free" blur reach. The
            // remaining passes operate ping↔pong at half res.
            let r = radius.max(0.0);

            // Ping-pong N times. Source = capture_tex on the first
            // pass, then swaps each iteration. Destination alternates
            // ping ↔ pong. With at least one pass, we always end up
            // with the final result in either ping_tex or pong_tex
            // — `final_blur_tex` reports which.
            let total = passes.max(1);
            for i in 0..total {
                let (src_tex, src_w, src_h) = if i == 0 {
                    (fbos.capture_tex, fbos.full_width, fbos.full_height)
                } else if i % 2 == 1 {
                    (fbos.ping_tex, fbos.blur_width, fbos.blur_height)
                } else {
                    (fbos.pong_tex, fbos.blur_width, fbos.blur_height)
                };
                let dst_fbo = if i % 2 == 0 {
                    fbos.ping_fbo
                } else {
                    fbos.pong_fbo
                };
                gl.uniform_2_f32(
                    Some(&self.blur.u_texel),
                    r / src_w as f32,
                    r / src_h as f32,
                );
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst_fbo));
                gl.viewport(0, 0, fbos.blur_width, fbos.blur_height);
                gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }

            // Restore default framebuffer for the rest of the scene.
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, self.width as i32, self.height as i32);
            gl.enable(glow::BLEND);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }

        self.blur_ready_this_frame = true;
        Ok(())
    }

    /// Which of `ping_tex` / `pong_tex` holds the final blurred result
    /// after `passes` ping-pong rounds. Pass 1 ends in ping, pass 2 in
    /// pong, etc.
    fn final_blur_tex(&self, passes: u32) -> Option<glow::NativeTexture> {
        let fbos = self.blur_fbos.as_ref()?;
        let total = passes.max(1);
        Some(if total % 2 == 1 { fbos.ping_tex } else { fbos.pong_tex })
    }

    /// Draw the prepared blurred-backdrop texture sampled at this
    /// element's screen rect. Backdrop has no associated client
    /// buffer; UVs are derived from the screen position so the
    /// rendered patch shows whatever was beneath the window after
    /// blur. Multiplied by `elem.alpha` so the caller can fade it
    /// in/out (e.g. during workspace crossfade).
    fn draw_blurred_backdrop(
        &mut self,
        elem: &sol_core::SceneElement<'_>,
        passes: u32,
        radius: f32,
    ) -> Result<()> {
        self.prepare_blur_backdrop(passes, radius)?;
        let Some(blur_tex) = self.final_blur_tex(passes) else {
            anyhow::bail!("blur tex unavailable");
        };
        let gl = &self.gl_stack.gl;
        let fb_w = self.width as f32;
        let fb_h = self.height as f32;
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
        // NDC rect (same math as the textured-quad path).
        let x0 = (elem.x / fb_w) * 2.0 - 1.0;
        let y0 = 1.0 - ((elem.y + dst_h) / fb_h) * 2.0;
        let rw = dst_w / fb_w * 2.0;
        let rh = dst_h / fb_h * 2.0;
        // UV sub-rect on the blur texture corresponds to the same
        // screen position. Two conventions collide here:
        //   - GL textures are bottom-up: blur tex y=0 is the bottom
        //     of the screen content, y=1 is the top.
        //   - The shared vertex shader applies a *flip* to v
        //     (`(1 - a_pos.y) * uv.h`) so a Wayland-style buffer
        //     (which is stored top-row-first → GL row 0 = wallpaper
        //     top) samples right-side-up.
        // For our GL-native blur tex that flip would invert the
        // sampling. We compensate by passing the v range *backwards*:
        // `v` is the top edge of the window's region in texture
        // coords (high v) and `uh` is *negative* so v + (1-a_pos.y)*uh
        // walks downward to the bottom edge. Net result: a_pos.y=1
        // (output top) samples texture top of window region, a_pos.y=0
        // (output bottom) samples texture bottom of window region.
        // GL does not care about negative interpolation widths — it
        // just runs the vertex linear-interp the other direction.
        let u = elem.x / fb_w;
        let v = (fb_h - elem.y) / fb_h;
        let uw = dst_w / fb_w;
        let uh = -dst_h / fb_h;
        tracing::trace!(
            elem_x = elem.x,
            elem_y = elem.y,
            dst_w,
            dst_h,
            fb_w,
            fb_h,
            u,
            v,
            uw,
            uh,
            "backdrop draw"
        );
        unsafe {
            gl.use_program(Some(self.backdrop.program));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.backdrop.vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.backdrop.u_tex), 0);
            gl.uniform_4_f32(Some(&self.backdrop.u_rect), x0, y0, rw, rh);
            gl.uniform_4_f32(Some(&self.backdrop.u_uv), u, v, uw, uh);
            gl.uniform_1_f32(
                Some(&self.backdrop.u_alpha),
                elem.alpha.clamp(0.0, 1.0),
            );
            gl.uniform_2_f32(Some(&self.backdrop.u_size), dst_w, dst_h);
            gl.uniform_1_f32(
                Some(&self.backdrop.u_radius),
                elem.corner_radius.max(0.0),
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(blur_tex));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        Ok(())
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

/// Hash the bg slice into a u64 the presenter can equality-compare
/// across frames. Captures buffer identity *and* on-screen rect of
/// every element so a moved or resized wallpaper (or a fresh buffer
/// from `wp-cycle.sh` swapping wallpapers) invalidates the blur
/// cache. Includes the slice length implicitly via the hashed
/// elements; an empty bg slice hashes to a stable "empty" value so
/// the cache check still works in degenerate scenes.
fn compute_bg_signature(elems: &[sol_core::SceneElement<'_>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    elems.len().hash(&mut h);
    for e in elems {
        e.buffer_key.hash(&mut h);
        e.x.to_bits().hash(&mut h);
        e.y.to_bits().hash(&mut h);
        e.dst_width.to_bits().hash(&mut h);
        e.dst_height.to_bits().hash(&mut h);
    }
    h.finish()
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
        SceneContent::BlurredBackdrop { .. } => {
            unreachable!("upload_shm_texture called with backdrop content")
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
        SceneContent::BlurredBackdrop { .. } => {
            unreachable!("import_dmabuf_texture called with backdrop content")
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

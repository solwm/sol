//! DRM/KMS + GBM + EGL/GLES smoke test for block B3.
//!
//! Opens a DRM card, selects a connected output, brings up a GBM surface and
//! a GLES2 EGL context on it, then runs a render loop that clears to magenta
//! for the first second and draws an animated checkerboard for the rest,
//! using drmModePageFlip for vsynced presentation.
//!
//! Must run from a VT where nothing else holds DRM master. Switch to a free
//! TTY (Ctrl+Alt+F2..F6) before running.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use drm::Device as BasicDevice;
use drm::control::{
    Device as ControlDevice, Event, ModeTypeFlags, PageFlipFlags, connector,
    framebuffer,
};
use gbm::{AsRaw, BufferObjectFlags, Format as GbmFormat};
use khronos_egl as egl;
use rustix::event::{PollFd, PollFlags, poll};

mod shader;

const EGL_PLATFORM_GBM_KHR: egl::Enum = 0x31D7;

/// Wrapper that makes `File` satisfy drm-rs's trait set. drm-rs relies on
/// implementors to provide an FD and lets the traits dispatch ioctls.
#[derive(Debug, Clone)]
pub struct Card(Arc<File>);

impl Card {
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Card(Arc::new(file)))
    }
}
impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl BasicDevice for Card {}
impl ControlDevice for Card {}

#[derive(Debug, Clone, Copy)]
pub struct OutputSelection {
    pub connector: connector::Handle,
    pub crtc: drm::control::crtc::Handle,
    pub mode: drm::control::Mode,
}

/// Describe what's connected without touching master. Safe to run while
/// another compositor (Hyprland, whatever) holds DRM master on the same VT.
pub fn describe_device(device: &Path) -> Result<()> {
    let card = Card::open(device)?;
    let res = card.resource_handles().context("resource_handles")?;

    println!("device: {}", device.display());
    println!("  connectors: {}", res.connectors().len());
    println!("  encoders:   {}", res.encoders().len());
    println!("  crtcs:      {}", res.crtcs().len());

    for &h in res.connectors() {
        let conn = match card.get_connector(h, false) {
            Ok(c) => c,
            Err(e) => {
                println!("  connector {h:?}: <get_connector failed: {e}>");
                continue;
            }
        };
        println!(
            "  connector {:?}: {:?} state={:?} modes={}",
            h,
            conn.interface(),
            conn.state(),
            conn.modes().len()
        );
        if conn.state() == connector::State::Connected {
            for (i, m) in conn.modes().iter().enumerate().take(3) {
                let (w, h) = m.size();
                println!(
                    "    mode[{i}]: {w}x{h}@{hz} {flags}",
                    hz = m.vrefresh(),
                    flags = if m.mode_type().contains(ModeTypeFlags::PREFERRED) {
                        "(preferred)"
                    } else {
                        ""
                    }
                );
            }
            if conn.modes().len() > 3 {
                println!("    ... {} more mode(s)", conn.modes().len() - 3);
            }
        }
    }
    Ok(())
}

pub fn pick_output(card: &Card) -> Result<OutputSelection> {
    let res = card.resource_handles().context("resource_handles")?;

    for &connector_h in res.connectors() {
        let conn = card
            .get_connector(connector_h, false)
            .with_context(|| format!("get_connector {connector_h:?}"))?;
        if conn.state() != connector::State::Connected {
            continue;
        }
        if conn.modes().is_empty() {
            continue;
        }
        // Preferred mode > first mode.
        let mode = conn
            .modes()
            .iter()
            .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .copied()
            .unwrap_or_else(|| conn.modes()[0]);

        // Find a CRTC that works with this connector's encoder.
        let encoders = conn.encoders().to_vec();
        for enc_h in encoders {
            let enc = match card.get_encoder(enc_h) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let compatible_crtcs = res.filter_crtcs(enc.possible_crtcs());
            if let Some(&crtc_h) = compatible_crtcs.first() {
                tracing::info!(
                    connector = ?conn.interface(),
                    width = mode.size().0,
                    height = mode.size().1,
                    vrefresh = mode.vrefresh(),
                    "selected output"
                );
                return Ok(OutputSelection {
                    connector: connector_h,
                    crtc: crtc_h,
                    mode,
                });
            }
        }
    }
    bail!("no connected connector with a usable CRTC found")
}

/// Wraps a `khronos_egl::DynamicInstance`, GBM surface, EGL display/context/
/// surface, and a `glow::Context`. Holds everything alive for the duration of
/// the smoke test.
pub struct GlStack {
    pub egl: egl::DynamicInstance<egl::EGL1_5>,
    pub display: egl::Display,
    pub context: egl::Context,
    pub surface: egl::Surface,
    pub gl: glow::Context,
    pub gbm: gbm::Device<Card>,
    pub gbm_surface: gbm::Surface<()>,
}

impl GlStack {
    pub fn new(card: Card, width: u32, height: u32) -> Result<Self> {
        let egl = unsafe {
            egl::DynamicInstance::<egl::EGL1_5>::load_required()
                .map_err(|e| anyhow!("load libEGL (need EGL 1.5): {e:?}"))?
        };

        let gbm = gbm::Device::new(card).context("gbm::Device::new")?;
        let gbm_surface = gbm
            .create_surface::<()>(
                width,
                height,
                GbmFormat::Xrgb8888,
                BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING,
            )
            .context("create gbm surface")?;

        // EGL_PLATFORM_GBM_KHR platform display.
        let raw_gbm = gbm.as_raw() as *mut std::ffi::c_void;
        let display = unsafe {
            egl.get_platform_display(EGL_PLATFORM_GBM_KHR, raw_gbm, &[egl::ATTRIB_NONE])
                .map_err(|e| anyhow!("get_platform_display: {e:?}"))?
        };
        let (major, minor) = egl
            .initialize(display)
            .map_err(|e| anyhow!("eglInitialize: {e:?}"))?;
        tracing::info!(major, minor, "EGL initialized");
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| anyhow!("bind_api: {e:?}"))?;

        let config_attrs = [
            egl::SURFACE_TYPE,
            egl::WINDOW_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            0,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES2_BIT,
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &config_attrs)
            .map_err(|e| anyhow!("choose config: {e:?}"))?
            .ok_or_else(|| anyhow!("no matching EGL config"))?;

        let ctx_attrs = [egl::CONTEXT_CLIENT_VERSION, 2, egl::NONE];
        let context = egl
            .create_context(display, config, None, &ctx_attrs)
            .map_err(|e| anyhow!("create_context: {e:?}"))?;

        let raw_gbm_surface = gbm_surface.as_raw() as *mut std::ffi::c_void;
        let surface = unsafe {
            egl.create_window_surface(display, config, raw_gbm_surface, None)
                .map_err(|e| anyhow!("create_window_surface: {e:?}"))?
        };

        egl.make_current(display, Some(surface), Some(surface), Some(context))
            .map_err(|e| anyhow!("make_current: {e:?}"))?;

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                egl.get_proc_address(s)
                    .map(|p| p as *const _)
                    .unwrap_or(std::ptr::null())
            })
        };

        Ok(Self {
            egl,
            display,
            context,
            surface,
            gl,
            gbm,
            gbm_surface,
        })
    }
}

/// Run the smoke test. Blocks until the duration elapses or the `should_quit`
/// flag flips.
pub fn run_smoke_test(
    device: &Path,
    duration: Duration,
    should_quit: Arc<AtomicBool>,
) -> Result<()> {
    let card = Card::open(device)?;

    if let Err(e) = card.acquire_master_lock() {
        tracing::warn!(error = ?e, "DRM_IOCTL_SET_MASTER failed — is another compositor on this VT?");
        return Err(anyhow!(
            "could not become DRM master on {}: {e:?}\n\
             \n\
             You're likely trying to run from the same VT as Hyprland (or any other\n\
             compositor). Switch to a free TTY (Ctrl+Alt+F2..F6), log in there, and\n\
             run the demo again.",
            device.display()
        ));
    }

    let sel = pick_output(&card)?;
    let (w, h) = sel.mode.size();
    let w = w as u32;
    let h = h as u32;

    let gl_stack = GlStack::new(card.clone(), w, h)?;
    let program = shader::build_checkerboard(&gl_stack.gl)?;

    let mut fb_cache: HashMap<usize, framebuffer::Handle> = HashMap::new();

    // Initial frame: clear to magenta, swap, lock front, add fb, modeset.
    // We keep the locked BO in `scanned_out` so it isn't returned to the
    // pool while it's on screen — dropping it later calls
    // gbm_surface_release_buffer.
    clear_magenta(&gl_stack);
    gl_stack
        .egl
        .swap_buffers(gl_stack.display, gl_stack.surface)
        .map_err(|e| anyhow!("initial swap_buffers: {e:?}"))?;

    let first_bo = unsafe { gl_stack.gbm_surface.lock_front_buffer() }
        .context("initial lock_front_buffer")?;
    let first_fb = get_or_add_fb(&card, &first_bo, &mut fb_cache)?;
    card.set_crtc(
        sel.crtc,
        Some(first_fb),
        (0, 0),
        &[sel.connector],
        Some(sel.mode),
    )
    .context("set_crtc (initial modeset)")?;
    tracing::info!(
        width = w, height = h, "modeset done; beginning render loop"
    );
    let mut scanned_out: Option<gbm::BufferObject<()>> = Some(first_bo);

    let start = Instant::now();
    let mut frame = 0u64;

    while start.elapsed() < duration && !should_quit.load(Ordering::SeqCst) {
        // Don't start a new frame if GBM has no free back buffer. Waiting
        // for the previous flip below is what frees one up.
        if !gl_stack.gbm_surface.has_free_buffers() {
            tracing::warn!("gbm pool exhausted — bug in buffer lifetime");
        }

        let t = start.elapsed().as_secs_f32();
        if t < 1.0 {
            clear_magenta(&gl_stack);
        } else {
            draw_checkerboard(&gl_stack, &program, w as i32, h as i32, t);
        }

        gl_stack
            .egl
            .swap_buffers(gl_stack.display, gl_stack.surface)
            .map_err(|e| anyhow!("swap_buffers: {e:?}"))?;

        let next_bo = unsafe { gl_stack.gbm_surface.lock_front_buffer() }
            .context("lock_front_buffer in loop")?;
        let next_fb = get_or_add_fb(&card, &next_bo, &mut fb_cache)?;
        card.page_flip(sel.crtc, next_fb, PageFlipFlags::EVENT, None)
            .context("page_flip")?;

        // Block until the flip has completed, then drop the previous
        // scanned-out buffer (now off-screen) back into the pool. Its Drop
        // impl calls gbm_surface_release_buffer.
        wait_for_page_flip(&card).context("wait_for_page_flip")?;
        scanned_out = Some(next_bo); // old one is dropped here

        frame += 1;
        if frame.is_multiple_of(60) {
            tracing::info!(frame, "rendered");
        }
    }

    // Hold scanned_out alive until we're done; kernel revokes master on fd
    // close, which causes the current scan-out to stop cleanly.
    drop(scanned_out);

    tracing::info!(frames = frame, "smoke test complete");
    Ok(())
}

fn wait_for_page_flip(card: &Card) -> Result<()> {
    // Poll with a generous timeout so a missed vblank doesn't hang forever.
    let mut pfd = [PollFd::new(card, PollFlags::IN)];
    match poll(&mut pfd, 2000) {
        Ok(_) => {}
        Err(e) => bail!("poll on drm fd: {e}"),
    }
    let events = card.receive_events().context("receive_events")?;
    for event in events {
        if matches!(event, Event::PageFlip(_)) {
            return Ok(());
        }
    }
    // No event yet — try once more briefly; some drivers deliver a vblank
    // ahead of the page flip event on the same readability.
    let _ = poll(&mut pfd, 200);
    for event in card.receive_events().context("receive_events")? {
        if matches!(event, Event::PageFlip(_)) {
            return Ok(());
        }
    }
    bail!("no page flip event within timeout")
}

/// Render a checkerboard frame via GBM+EGL+GLES2 without DRM master, read
/// pixels back with glReadPixels, and write them to a PNG. Lets us prove the
/// GPU path works (on the same hardware that would eventually scan out) while
/// the actual compositor still holds master.
pub fn run_offscreen_render(
    device: &Path,
    out_path: &Path,
    width: u32,
    height: u32,
) -> Result<()> {
    use glow::HasContext;

    let card = Card::open(device)?;
    // NB: no acquire_master_lock — offscreen render doesn't need it.

    let gl_stack = GlStack::new(card, width, height)?;
    let program = shader::build_checkerboard(&gl_stack.gl)?;

    // One frame, t picked so we're clearly in the animated region.
    draw_checkerboard(&gl_stack, &program, width as i32, height as i32, 2.5);
    // Force the driver to actually submit before we read.
    unsafe { gl_stack.gl.finish() };

    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
    unsafe {
        gl_stack.gl.read_pixels(
            0,
            0,
            width as i32,
            height as i32,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(&mut pixels),
        );
    }
    // GL origin is bottom-left; PNG is top-left, so flip row order.
    flip_rows_rgba(&mut pixels, width as usize, height as usize);

    // Also swap the GBM surface once so the driver isn't left with any
    // half-scheduled work; ignore errors.
    let _ = gl_stack
        .egl
        .swap_buffers(gl_stack.display, gl_stack.surface);

    let file = std::fs::File::create(out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    let mut enc = png::Encoder::new(file, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .context("png header")?
        .write_image_data(&pixels)
        .context("png write")?;
    tracing::info!(
        path = %out_path.display(),
        width,
        height,
        "offscreen render written"
    );
    Ok(())
}

fn flip_rows_rgba(buf: &mut [u8], w: usize, h: usize) {
    let stride = w * 4;
    for y in 0..(h / 2) {
        let top = y * stride;
        let bot = (h - 1 - y) * stride;
        for i in 0..stride {
            buf.swap(top + i, bot + i);
        }
    }
}

fn clear_magenta(stack: &GlStack) {
    use glow::HasContext;
    unsafe {
        stack.gl.clear_color(1.0, 0.0, 1.0, 1.0);
        stack.gl.clear(glow::COLOR_BUFFER_BIT);
    }
}

fn draw_checkerboard(
    stack: &GlStack,
    program: &shader::CheckerProgram,
    w: i32,
    h: i32,
    t: f32,
) {
    use glow::HasContext;
    unsafe {
        stack.gl.viewport(0, 0, w, h);
        stack.gl.clear_color(0.05, 0.05, 0.08, 1.0);
        stack.gl.clear(glow::COLOR_BUFFER_BIT);

        stack.gl.use_program(Some(program.program));
        stack.gl.uniform_2_f32(Some(&program.u_resolution), w as f32, h as f32);
        stack.gl.uniform_1_f32(Some(&program.u_time), t);
        stack.gl.bind_buffer(glow::ARRAY_BUFFER, Some(program.vbo));
        stack.gl.enable_vertex_attrib_array(0);
        stack
            .gl
            .vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
        stack.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        stack.gl.disable_vertex_attrib_array(0);
    }
}

/// Return the framebuffer id for a given GBM BO, adding one to DRM on first
/// sight. Keyed by the BO's raw pointer because GBM recycles the same bos
/// through its pool, so adding a fb for each iteration would leak.
fn get_or_add_fb(
    card: &Card,
    bo: &gbm::BufferObject<()>,
    cache: &mut HashMap<usize, framebuffer::Handle>,
) -> Result<framebuffer::Handle> {
    let key = bo.as_raw() as usize;
    if let Some(fb) = cache.get(&key).copied() {
        return Ok(fb);
    }
    let fb = card
        .add_framebuffer(bo, 24, 32)
        .context("add_framebuffer")?;
    cache.insert(key, fb);
    Ok(fb)
}

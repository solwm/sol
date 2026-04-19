//! Wayland server for hyperland-rs.
//!
//! Handles protocol traffic in all backends; rendering is delegated to a
//! `BackendState` value (software canvas -> PNG for headless, or a
//! `hypr_backend_drm::DrmPresenter` for real hardware).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic};
use hypr_backend_drm::DrmPresenter;
use hypr_core::{Scene, SceneElement};
use wayland_protocols::xdg::shell::server::xdg_wm_base::XdgWmBase;
use wayland_server::{
    Display, DisplayHandle, Resource, Weak,
    backend::{ClientData, ClientId, DisconnectReason, GlobalId},
    protocol::{
        wl_compositor::WlCompositor, wl_output::WlOutput, wl_seat::WlSeat, wl_shm::WlShm,
        wl_surface::WlSurface,
    },
};

mod compositor;
mod cursor;
mod input;
mod output;
mod render;
mod seat;
mod shm;
mod xdg_shell;

use compositor::{SurfaceData, SurfaceRole};
use input::{InputEvent, InputState};
use render::Canvas;

const COMPOSITOR_VERSION: u32 = 6;
const SHM_VERSION: u32 = 1;
const OUTPUT_VERSION: u32 = 4;
const SEAT_VERSION: u32 = 7;
const XDG_WM_BASE_VERSION: u32 = 5;

/// Compositor state shared across Dispatch impls. Backend-specific resources
/// (canvas, presenter) live on `Compositor` alongside this, not here.
pub struct State {
    pub display_handle: DisplayHandle,
    pub globals: Globals,
    pub clients_seen: u64,
    /// Mapped xdg_toplevels in stacking order (bottom to top). Held as weak
    /// refs so dead surfaces drop out on the next render.
    pub mapped_toplevels: Vec<Weak<WlSurface>>,
    /// Logical screen size. Drives toplevel placement and (eventually) the
    /// wl_output mode advertised to clients.
    pub screen_width: u32,
    pub screen_height: u32,
    pub needs_render: bool,
    pub started: Instant,
    pub next_serial: u32,
    pub cursor: Cursor,
    pub input: Option<InputState>,
}

/// Software cursor: a fixed-size ARGB sprite whose top-left in screen space
/// is always `(pos_x - hot_x, pos_y - hot_y)`. Rendered as the topmost scene
/// element when visible.
pub struct Cursor {
    pub pos_x: f64,
    pub pos_y: f64,
    pub visible: bool,
    pub pixels: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub hot_x: i32,
    pub hot_y: i32,
}

impl Cursor {
    pub fn new(centre_x: f64, centre_y: f64) -> Self {
        Self {
            pos_x: centre_x,
            pos_y: centre_y,
            visible: true,
            pixels: cursor::pixels(),
            width: cursor::CURSOR_W,
            height: cursor::CURSOR_H,
            hot_x: cursor::CURSOR_HOT_X,
            hot_y: cursor::CURSOR_HOT_Y,
        }
    }
}

#[derive(Debug)]
pub struct Globals {
    pub compositor: GlobalId,
    pub shm: GlobalId,
    pub output: GlobalId,
    pub seat: GlobalId,
    pub xdg_wm_base: GlobalId,
}

impl State {
    pub fn elapsed_ms(&self) -> u32 {
        self.started.elapsed().as_millis() as u32
    }
    pub fn next_serial(&mut self) -> u32 {
        self.next_serial = self.next_serial.wrapping_add(1);
        self.next_serial
    }
}

#[derive(Default)]
pub struct ClientState;

impl ClientData for ClientState {
    fn initialized(&self, client_id: ClientId) {
        tracing::info!(?client_id, "client connected");
    }
    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        tracing::info!(?client_id, ?reason, "client disconnected");
    }
}

/// Backend-specific render target. Chosen at startup; does not switch at
/// runtime.
pub enum BackendState {
    Headless { canvas: Canvas, png_path: PathBuf },
    Drm(DrmPresenter),
}

/// Pairs Display + State + backend so calloop callbacks can reach everything.
pub struct Compositor {
    pub state: State,
    pub display: Display<State>,
    pub backend: BackendState,
}

/// Collects scene elements from mapped xdg_toplevels and the WlBuffers that
/// back them. Returns the buffers alongside the scene so callers keep them
/// alive for the duration of rendering.
fn collect_scene(state: &State) -> Vec<wayland_server::protocol::wl_buffer::WlBuffer> {
    let mut buffers = Vec::new();
    for weak in state.mapped_toplevels.iter() {
        let Ok(surface) = weak.upgrade() else { continue };
        let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
            continue;
        };
        let sd = sd_arc.lock().unwrap();
        if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
            continue;
        }
        let Some(buf_weak) = sd.current.buffer.as_ref() else {
            continue;
        };
        if let Ok(buf) = buf_weak.upgrade() {
            buffers.push(buf);
        }
    }
    buffers
}

fn scene_from_buffers<'a>(
    buffers: &'a [wayland_server::protocol::wl_buffer::WlBuffer],
    cursor: &'a Cursor,
    screen_width: u32,
    screen_height: u32,
) -> Scene<'a> {
    let mut scene = Scene::new();
    for buf in buffers {
        let Some(bd) = buf.data::<shm::BufferData>() else { continue };
        let Some(bytes) = bd.bytes() else { continue };
        let Some(format) = bd.pixel_format() else { continue };
        // Centre the toplevel. Replace with a real layout in B6.
        let x = (screen_width as i32 - bd.width) / 2;
        let y = (screen_height as i32 - bd.height) / 2;
        let key = (bd as *const shm::BufferData) as usize as u64;
        scene.elements.push(SceneElement {
            buffer_key: key,
            pixels: bytes,
            format,
            width: bd.width,
            height: bd.height,
            stride: bd.stride,
            x,
            y,
        });
    }
    // Cursor last so it always draws on top.
    if cursor.visible {
        scene.elements.push(SceneElement {
            buffer_key: CURSOR_SCENE_KEY,
            pixels: &cursor.pixels,
            format: hypr_core::PixelFormat::Argb8888,
            width: cursor.width,
            height: cursor.height,
            stride: cursor.width * 4,
            x: cursor.pos_x as i32 - cursor.hot_x,
            y: cursor.pos_y as i32 - cursor.hot_y,
        });
    }
    scene
}

/// Stable key the DRM renderer uses to avoid re-uploading the cursor
/// sprite every frame. Anything outside the range of real BufferData
/// pointers works; pick something unlikely to collide.
const CURSOR_SCENE_KEY: u64 = 0xC0FFEE_C0FFEE;

fn render_tick(comp: &mut Compositor) -> Result<()> {
    // Prune dead weak refs each frame.
    comp.state.mapped_toplevels.retain(|w| w.upgrade().is_ok());

    let buffers = collect_scene(&comp.state);
    let scene = scene_from_buffers(
        &buffers,
        &comp.state.cursor,
        comp.state.screen_width,
        comp.state.screen_height,
    );
    let drawn = scene.elements.len();

    match &mut comp.backend {
        BackendState::Headless { canvas, png_path } => {
            canvas.clear();
            for e in &scene.elements {
                canvas.blit_argb(
                    e.x,
                    e.y,
                    e.pixels,
                    e.width,
                    e.height,
                    e.stride,
                    e.format,
                );
            }
            canvas
                .write_png(png_path)
                .with_context(|| format!("write {}", png_path.display()))?;
            tracing::info!(drawn, path = %png_path.display(), "headless frame");
        }
        BackendState::Drm(presenter) => {
            presenter.render_scene(&scene).context("drm render_scene")?;
            tracing::debug!(drawn, "drm frame");
        }
    }

    // Scene is done; drop buffers to release them back to clients.
    for buf in buffers {
        buf.release();
    }
    Ok(())
}

fn setup_event_loop(
    backend: BackendState,
    screen_width: u32,
    screen_height: u32,
    input: Option<(InputState, std::os::fd::OwnedFd)>,
) -> Result<()> {
    let mut event_loop: EventLoop<'static, Compositor> =
        EventLoop::try_new().context("create calloop event loop")?;
    let mut display: Display<State> = Display::new().context("create wayland display")?;
    let dh = display.handle();

    let globals = Globals {
        compositor: dh.create_global::<State, WlCompositor, ()>(COMPOSITOR_VERSION, ()),
        shm: dh.create_global::<State, WlShm, ()>(SHM_VERSION, ()),
        output: dh.create_global::<State, WlOutput, ()>(OUTPUT_VERSION, ()),
        seat: dh.create_global::<State, WlSeat, ()>(SEAT_VERSION, ()),
        xdg_wm_base: dh.create_global::<State, XdgWmBase, ()>(XDG_WM_BASE_VERSION, ()),
    };
    tracing::info!(?globals, "advertised globals");

    let listener = wayland_server::ListeningSocket::bind_auto("wayland", 1..33)
        .context("bind wayland socket")?;
    let socket_name = listener
        .socket_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<anonymous>".into());
    tracing::info!(socket = %socket_name, "listening");

    event_loop
        .handle()
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            |_ready, listener, comp| {
                while let Some(stream) = listener.accept()? {
                    match comp
                        .state
                        .display_handle
                        .insert_client(stream, Arc::new(ClientState))
                    {
                        Ok(_) => comp.state.clients_seen += 1,
                        Err(e) => tracing::warn!(error = %e, "insert_client failed"),
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert socket source: {e}"))?;

    let display_fd = display.backend().poll_fd().try_clone_to_owned()?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display_fd, Interest::READ, Mode::Level),
            |_ready, _fd, comp| {
                comp.display
                    .dispatch_clients(&mut comp.state)
                    .map_err(std::io::Error::other)?;
                comp.display
                    .flush_clients()
                    .map_err(std::io::Error::other)?;
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert display source: {e}"))?;

    let (input_state, input_fd) = match input {
        Some((s, fd)) => (Some(s), Some(fd)),
        None => (None, None),
    };
    if let Some(fd) = input_fd {
        event_loop
            .handle()
            .insert_source(
                Generic::new(fd, Interest::READ, Mode::Level),
                |_ready, _fd, comp| {
                    let events = match comp.state.input.as_mut() {
                        Some(i) => i.drain(),
                        None => return Ok(PostAction::Continue),
                    };
                    for ev in events {
                        apply_input(&mut comp.state, ev);
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert input source: {e}"))?;
    }

    let cursor = Cursor::new(screen_width as f64 / 2.0, screen_height as f64 / 2.0);
    let state = State {
        display_handle: dh,
        globals,
        clients_seen: 0,
        mapped_toplevels: Vec::new(),
        screen_width,
        screen_height,
        needs_render: false,
        started: Instant::now(),
        next_serial: 0,
        cursor,
        input: input_state,
    };
    let mut compositor = Compositor {
        state,
        display,
        backend,
    };
    let _ = compositor.display.flush_clients();

    tracing::info!(
        socket = %socket_name,
        "ready: export WAYLAND_DISPLAY={} and run a client",
        socket_name
    );

    event_loop
        .run(None, &mut compositor, |comp| {
            if comp.state.needs_render {
                comp.state.needs_render = false;
                if let Err(e) = render_tick(comp) {
                    tracing::error!(error = %e, "render failed");
                }
            }
            let _ = comp.display.flush_clients();
        })
        .context("event loop errored")?;

    Ok(())
}

/// Apply one translated libinput event to the state. Updates the cursor and
/// sets `needs_render`. Input->client routing lands in a later B5 substep.
fn apply_input(state: &mut State, ev: InputEvent) {
    match ev {
        InputEvent::PointerMotion { dx, dy } => {
            state.cursor.pos_x = (state.cursor.pos_x + dx)
                .clamp(0.0, state.screen_width as f64 - 1.0);
            state.cursor.pos_y = (state.cursor.pos_y + dy)
                .clamp(0.0, state.screen_height as f64 - 1.0);
            state.needs_render = true;
        }
        InputEvent::PointerMotionAbsolute { x_mm, y_mm } => {
            // libinput reports absolute device coords as millimetres on the
            // device surface. Without a device-size mapping we just treat
            // them as screen-space until we wire up proper calibration.
            state.cursor.pos_x = x_mm.clamp(0.0, state.screen_width as f64 - 1.0);
            state.cursor.pos_y = y_mm.clamp(0.0, state.screen_height as f64 - 1.0);
            state.needs_render = true;
        }
        InputEvent::PointerButton { button, pressed } => {
            tracing::info!(button, pressed, "pointer button (not forwarded yet)");
        }
        InputEvent::Key { keycode, pressed } => {
            tracing::info!(keycode, pressed, "key (not forwarded yet)");
        }
    }
}

/// Headless backend: software canvas, dumps a PNG every frame.
pub fn run_headless(png_path: PathBuf, width: u32, height: u32) -> Result<()> {
    tracing::info!(path = %png_path.display(), width, height, "headless backend");
    let backend = BackendState::Headless {
        canvas: Canvas::new(width, height),
        png_path,
    };
    setup_event_loop(backend, width, height, None)
}

/// DRM backend: takes master on `device`, renders via GLES to the screen.
/// Must be called from a VT that doesn't already have a compositor.
pub fn run_drm(device: &Path) -> Result<()> {
    let presenter = DrmPresenter::new(device).context("initialise DrmPresenter")?;
    let (w, h) = presenter.size();
    tracing::info!(width = w, height = h, "drm backend");

    // libinput is DRM-only: a headless PNG dump has nowhere to point a cursor.
    let input = match InputState::init("seat0") {
        Ok(pair) => Some(pair),
        Err(e) => {
            tracing::warn!(error = %e, "libinput init failed; running without input");
            None
        }
    };
    setup_event_loop(BackendState::Drm(presenter), w, h, input)
}

/// Back-compat entry: default to headless with the old PNG path.
pub fn run() -> Result<()> {
    let png_path = std::env::var_os("HYPRS_PNG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/hyprs-headless.png"));
    run_headless(png_path, 1920, 1080)
}

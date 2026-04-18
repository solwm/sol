//! Wayland server for hyperland-rs.
//!
//! B2 scope: bring up a Wayland display, accept clients, let them create
//! wl_shm buffers and map xdg_toplevels, and composite mapped toplevels onto
//! a headless canvas written to PNG.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic};
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
mod output;
mod render;
mod seat;
mod shm;
mod xdg_shell;

use compositor::{SurfaceData, SurfaceRole};
use render::Canvas;

const COMPOSITOR_VERSION: u32 = 6;
const SHM_VERSION: u32 = 1;
const OUTPUT_VERSION: u32 = 4;
const SEAT_VERSION: u32 = 7;
const XDG_WM_BASE_VERSION: u32 = 5;

const CANVAS_WIDTH: u32 = output::OUTPUT_WIDTH as u32;
const CANVAS_HEIGHT: u32 = output::OUTPUT_HEIGHT as u32;

/// Top-level compositor state. Everything globally mutable lives here; it's
/// what Dispatch impls receive as `&mut self`.
pub struct State {
    pub display_handle: DisplayHandle,
    pub globals: Globals,
    pub clients_seen: u64,
    /// Mapped xdg_toplevels in stacking order (bottom to top). Held as weak
    /// refs so dead surfaces drop out on the next scan.
    pub mapped_toplevels: Vec<Weak<WlSurface>>,
    pub canvas: Canvas,
    pub png_path: PathBuf,
    pub needs_render: bool,
    pub started: Instant,
    pub next_serial: u32,
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

/// Pairs Display + State so calloop callbacks can mutate both.
pub struct Compositor {
    pub state: State,
    pub display: Display<State>,
}

/// Walk mapped_toplevels, composite each surface's current buffer onto the
/// canvas, write PNG.
fn render_and_dump(comp: &mut Compositor) -> Result<()> {
    comp.state.canvas.clear();

    // Drop any toplevels whose WlSurface has been destroyed.
    comp.state
        .mapped_toplevels
        .retain(|w| w.upgrade().is_ok());

    let mut drawn = 0usize;
    for weak in comp.state.mapped_toplevels.clone() {
        let Ok(surface) = weak.upgrade() else {
            continue;
        };
        let Some(sd) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
            continue;
        };
        let sd = sd.lock().unwrap();
        if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
            continue;
        }
        let Some(buf_weak) = &sd.current.buffer else {
            continue;
        };
        let Ok(buf) = buf_weak.upgrade() else {
            continue;
        };
        let Some(buf_data) = buf.data::<shm::BufferData>() else {
            continue;
        };
        let Some(format) = buf_data.pixel_format() else {
            tracing::warn!(format = ?buf_data.format, "unsupported format, skipping blit");
            continue;
        };
        let Some(bytes) = buf_data.bytes() else {
            tracing::warn!("buffer range out of bounds, skipping");
            continue;
        };
        // Simple placement: centre the toplevel on the canvas.
        let x = (CANVAS_WIDTH as i32 - buf_data.width) / 2;
        let y = (CANVAS_HEIGHT as i32 - buf_data.height) / 2;
        comp.state.canvas.blit_argb(
            x,
            y,
            bytes,
            buf_data.width,
            buf_data.height,
            buf_data.stride,
            format,
        );
        // Tell the client its buffer is no longer needed so it can reuse it.
        buf.release();
        drawn += 1;
    }

    comp.state
        .canvas
        .write_png(&comp.state.png_path)
        .with_context(|| format!("write {}", comp.state.png_path.display()))?;
    tracing::info!(
        drawn,
        path = %comp.state.png_path.display(),
        "rendered frame"
    );
    Ok(())
}

pub fn run() -> Result<()> {
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

    let png_path = std::env::var_os("HYPRS_PNG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/hyprs-headless.png"));
    tracing::info!(path = %png_path.display(), "headless PNG output");

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

    let state = State {
        display_handle: dh,
        globals,
        clients_seen: 0,
        mapped_toplevels: Vec::new(),
        canvas: Canvas::new(CANVAS_WIDTH, CANVAS_HEIGHT),
        png_path,
        needs_render: false,
        started: Instant::now(),
        next_serial: 0,
    };
    let mut compositor = Compositor { state, display };
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
                if let Err(e) = render_and_dump(comp) {
                    tracing::error!(error = %e, "render failed");
                }
            }
            let _ = comp.display.flush_clients();
        })
        .context("event loop errored")?;

    Ok(())
}

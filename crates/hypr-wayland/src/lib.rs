//! Wayland server plumbing for hyperland-rs.
//!
//! Block B1: bring up a Wayland display, accept client connections, advertise
//! the smallest set of globals a real client checks on startup. No surface or
//! buffer handling yet — that lands in B2.

use std::sync::Arc;

use anyhow::{Context, Result};
use calloop::{
    EventLoop, Interest, Mode, PostAction,
    generic::Generic,
};
use wayland_server::{
    Client, DataInit, Dispatch, Display, DisplayHandle, GlobalDispatch, New, Resource,
    backend::{ClientData, ClientId, DisconnectReason, GlobalId},
    protocol::{
        wl_compositor::{self, WlCompositor},
        wl_output::{self, WlOutput},
        wl_seat::{self, WlSeat},
        wl_shm::{self, WlShm},
        wl_shm_pool::{self, WlShmPool},
        wl_surface::{self, WlSurface},
        wl_region::{self, WlRegion},
    },
};

mod globals;

/// Versions we advertise for the core globals. Chosen conservatively so a
/// wide range of clients will bind them. We don't implement every request at
/// these versions yet; we just stop the client from giving up before we can
/// observe it.
const COMPOSITOR_VERSION: u32 = 6;
const SHM_VERSION: u32 = 1;
const OUTPUT_VERSION: u32 = 4;
const SEAT_VERSION: u32 = 7;

/// Top-level compositor state. Holds the handle clients are bound to and any
/// global ids we want to be able to tear down later.
pub struct State {
    pub display_handle: DisplayHandle,
    pub globals: Globals,
    pub clients_seen: u64,
}

#[derive(Debug)]
pub struct Globals {
    pub compositor: GlobalId,
    pub shm: GlobalId,
    pub output: GlobalId,
    pub seat: GlobalId,
}

/// Per-client data. Logs connect + disconnect so we can watch the socket work.
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

/// Runs the compositor's main loop. Blocks until the event loop exits or an
/// unrecoverable error occurs.
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
    };
    tracing::info!(?globals, "advertised globals");

    let listener = wayland_server::ListeningSocket::bind_auto("wayland", 1..33)
        .context("bind wayland socket")?;
    let socket_name = listener
        .socket_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<anonymous>".into());
    tracing::info!(socket = %socket_name, "listening");

    // Accept new client connections on the socket.
    event_loop
        .handle()
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            |_readiness, listener, comp| {
                while let Some(stream) = listener.accept()? {
                    match comp
                        .state
                        .display_handle
                        .insert_client(stream, Arc::new(ClientState))
                    {
                        Ok(_client) => {
                            comp.state.clients_seen += 1;
                            tracing::debug!(
                                total = comp.state.clients_seen,
                                "inserted client"
                            );
                        }
                        Err(e) => tracing::warn!(error = %e, "insert_client failed"),
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert socket source: {e}"))?;

    // Dispatch Wayland protocol traffic whenever the display fd is readable.
    let display_fd = display.backend().poll_fd().try_clone_to_owned()?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display_fd, Interest::READ, Mode::Level),
            |_readiness, _fd, comp| {
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

    let mut compositor = Compositor {
        state: State {
            display_handle: dh,
            globals,
            clients_seen: 0,
        },
        display,
    };

    // Flush once so already-bound clients see initial events (e.g. globals).
    let _ = compositor.display.flush_clients();

    tracing::info!(
        "run: set WAYLAND_DISPLAY={} to connect a client",
        socket_name
    );

    event_loop
        .run(None, &mut compositor, |comp| {
            let _ = comp.display.flush_clients();
        })
        .context("event loop errored")?;

    Ok(())
}

/// Pairs the `Display` and mutable `State` so calloop callbacks can touch
/// both through a single generic parameter.
pub struct Compositor {
    pub state: State,
    pub display: Display<State>,
}

// --- protocol impls: stubs at B1 ----------------------------------------------
//
// At this block we only care that clients *bind* these globals without error.
// The request handlers log and otherwise do nothing useful. Real behaviour
// lands in later blocks.

impl GlobalDispatch<WlCompositor, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlCompositor>,
        _global_data: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let r = init.init(resource, ());
        tracing::info!(id = ?r.id(), "bind wl_compositor");
    }
}

impl Dispatch<WlCompositor, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlCompositor,
        request: wl_compositor::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_compositor::Request::CreateSurface { id } => {
                let _ = init.init(id, ());
                tracing::debug!("wl_compositor.create_surface (stub)");
            }
            wl_compositor::Request::CreateRegion { id } => {
                let _ = init.init(id, ());
                tracing::debug!("wl_compositor.create_region (stub)");
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSurface, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSurface,
        request: wl_surface::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        tracing::trace!(?request, "wl_surface request (ignored at B1)");
    }
}

impl Dispatch<WlRegion, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlRegion,
        request: wl_region::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        tracing::trace!(?request, "wl_region request (ignored at B1)");
    }
}

impl GlobalDispatch<WlShm, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlShm>,
        _global_data: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let shm = init.init(resource, ());
        // Advertise the two mandatory formats; that's all libwayland clients
        // typically check on init.
        shm.format(wl_shm::Format::Argb8888);
        shm.format(wl_shm::Format::Xrgb8888);
        tracing::info!(id = ?shm.id(), "bind wl_shm");
    }
}

impl Dispatch<WlShm, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlShm,
        request: wl_shm::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, fd: _, size: _ } = request {
            let _ = init.init(id, ());
            tracing::debug!("wl_shm.create_pool (stub)");
        }
    }
}

impl Dispatch<WlShmPool, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlShmPool,
        request: wl_shm_pool::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        tracing::trace!(?request, "wl_shm_pool request (ignored at B1)");
    }
}

impl GlobalDispatch<WlOutput, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlOutput>,
        _global_data: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let output = init.init(resource, ());
        // Send a minimal but realistic set of events so clients don't panic.
        output.geometry(
            0,
            0,
            300,
            200,
            wl_output::Subpixel::Unknown,
            "hyperland-rs".into(),
            "headless-0".into(),
            wl_output::Transform::Normal,
        );
        output.mode(
            wl_output::Mode::Current | wl_output::Mode::Preferred,
            1920,
            1080,
            60_000,
        );
        if output.version() >= 2 {
            output.scale(1);
            output.done();
        }
        if output.version() >= 4 {
            output.name("HEADLESS-0".into());
            output.description("hyperland-rs B1 stub output".into());
        }
        tracing::info!(id = ?output.id(), "bind wl_output");
    }
}

impl Dispatch<WlOutput, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlOutput,
        request: wl_output::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        tracing::trace!(?request, "wl_output request (ignored at B1)");
    }
}

impl GlobalDispatch<WlSeat, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSeat>,
        _global_data: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let seat = init.init(resource, ());
        // No capabilities yet — we haven't wired libinput.
        seat.capabilities(wl_seat::Capability::empty());
        if seat.version() >= 2 {
            seat.name("seat0".into());
        }
        tracing::info!(id = ?seat.id(), "bind wl_seat");
    }
}

impl Dispatch<WlSeat, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSeat,
        request: wl_seat::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        tracing::trace!(?request, "wl_seat request (ignored at B1)");
    }
}

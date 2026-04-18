//! Tiny Wayland client that paints a 256x256 window with three horizontal
//! stripes (red / green / blue) and waits. Used as the driver for the B2
//! headless PNG pipeline.

use std::os::fd::{AsFd, OwnedFd};
use std::time::Duration;

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use rustix::fs::{MemfdFlags, ftruncate, memfd_create};
use wayland_client::{
    Connection, Dispatch, EventQueue, Proxy, QueueHandle,
    protocol::{
        wl_buffer::WlBuffer,
        wl_callback::WlCallback,
        wl_compositor::WlCompositor,
        wl_registry::{self, WlRegistry},
        wl_shm::{self, Format, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
    },
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};

const W: i32 = 256;
const H: i32 = 256;

struct App {
    compositor: Option<WlCompositor>,
    shm: Option<WlShm>,
    wm_base: Option<XdgWmBase>,
    surface: Option<WlSurface>,
    xdg_surface: Option<XdgSurface>,
    toplevel: Option<XdgToplevel>,
    buffer: Option<WlBuffer>,
    configured: bool,
    done: bool,
}

impl App {
    fn new() -> Self {
        Self {
            compositor: None,
            shm: None,
            wm_base: None,
            surface: None,
            xdg_surface: None,
            toplevel: None,
            buffer: None,
            configured: false,
            done: false,
        }
    }
}

fn make_memfd(size: usize) -> Result<OwnedFd> {
    let fd = memfd_create("hyprs-paint", MemfdFlags::CLOEXEC).context("memfd_create")?;
    ftruncate(&fd, size as u64).context("ftruncate")?;
    Ok(fd)
}

/// Writes three horizontal stripes of R / G / B (opaque) in ARGB8888 (little
/// endian in-memory byte order: B G R A).
fn paint_stripes(bytes: &mut [u8]) {
    for y in 0..H {
        let (r, g, b, a): (u8, u8, u8, u8) = match y {
            y if y < H / 3 => (0xff, 0x30, 0x30, 0xff),
            y if y < 2 * H / 3 => (0x30, 0xff, 0x30, 0xff),
            _ => (0x30, 0x30, 0xff, 0xff),
        };
        for x in 0..W {
            let idx = (y as usize * W as usize + x as usize) * 4;
            bytes[idx] = b;
            bytes[idx + 1] = g;
            bytes[idx + 2] = r;
            bytes[idx + 3] = a;
        }
    }
}

fn build_buffer(shm: &WlShm, qh: &QueueHandle<App>) -> Result<WlBuffer> {
    let size = (W * H * 4) as usize;
    let fd = make_memfd(size)?;
    // Writable mapping on the client side; server keeps its own read-only map.
    {
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&fd)? };
        paint_stripes(&mut mmap);
        mmap.flush().ok();
    }
    let pool: WlShmPool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
    let buffer = pool.create_buffer(0, W, H, W * 4, Format::Argb8888, qh, ());
    pool.destroy();
    Ok(buffer)
}

fn main() -> Result<()> {
    let conn = Connection::connect_to_env().context("connect WAYLAND_DISPLAY")?;
    let display = conn.display();
    let mut queue: EventQueue<App> = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = display.get_registry(&qh, ());

    let mut app = App::new();
    queue.roundtrip(&mut app)?;

    let compositor = app.compositor.clone().context("no wl_compositor")?;
    let shm = app.shm.clone().context("no wl_shm")?;
    let wm_base = app.wm_base.clone().context("no xdg_wm_base")?;

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("hyprs-paint".into());
    toplevel.set_app_id("rs.hyperland.paint".into());
    surface.commit();

    app.surface = Some(surface);
    app.xdg_surface = Some(xdg_surface);
    app.toplevel = Some(toplevel);

    // Wait for the initial xdg_surface.configure.
    while !app.configured {
        queue.blocking_dispatch(&mut app)?;
    }

    let buffer = build_buffer(&shm, &qh)?;
    let surface = app.surface.as_ref().unwrap();
    surface.attach(Some(&buffer), 0, 0);
    surface.damage_buffer(0, 0, W, H);
    let _cb: WlCallback = surface.frame(&qh, ());
    surface.commit();
    app.buffer = Some(buffer);

    // Stay alive briefly so the server sees and renders the commit.
    let deadline = std::time::Instant::now() + Duration::from_millis(500);
    while std::time::Instant::now() < deadline && !app.done {
        queue.roundtrip(&mut app)?;
        std::thread::sleep(Duration::from_millis(50));
    }

    println!("paint: committed and rendered");
    Ok(())
}

impl Dispatch<WlRegistry, ()> for App {
    fn event(
        state: &mut Self,
        reg: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(reg.bind::<WlCompositor, _, _>(name, version.min(6), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(reg.bind::<WlShm, _, _>(name, version.min(1), qh, ()));
                }
                "xdg_wm_base" => {
                    state.wm_base = Some(reg.bind::<XdgWmBase, _, _>(name, version.min(5), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<WlCompositor, ()> for App {
    fn event(_: &mut Self, _: &WlCompositor, _: <WlCompositor as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlSurface, ()> for App {
    fn event(_: &mut Self, _: &WlSurface, _: <WlSurface as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlShm, ()> for App {
    fn event(_: &mut Self, _: &WlShm, event: wl_shm::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_shm::Event::Format { format } = event {
            let _ = format;
        }
    }
}
impl Dispatch<WlShmPool, ()> for App {
    fn event(_: &mut Self, _: &WlShmPool, _: <WlShmPool as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlBuffer, ()> for App {
    fn event(_: &mut Self, _: &WlBuffer, _: <WlBuffer as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlCallback, ()> for App {
    fn event(state: &mut Self, _: &WlCallback, _: <WlCallback as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        state.done = true;
    }
}
impl Dispatch<XdgWmBase, ()> for App {
    fn event(_: &mut Self, base: &XdgWmBase, event: xdg_wm_base::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            base.pong(serial);
        }
    }
}
impl Dispatch<XdgSurface, ()> for App {
    fn event(state: &mut Self, xs: &XdgSurface, event: xdg_surface::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if let xdg_surface::Event::Configure { serial } = event {
            xs.ack_configure(serial);
            state.configured = true;
        }
    }
}
impl Dispatch<XdgToplevel, ()> for App {
    fn event(state: &mut Self, _: &XdgToplevel, event: xdg_toplevel::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if matches!(event, xdg_toplevel::Event::Close) {
            state.done = true;
        }
    }
}

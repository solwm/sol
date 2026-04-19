//! Interactive Wayland test client. Paints a 256x256 toplevel with three
//! horizontal stripes and cycles the palette every time the server delivers
//! a pointer button or a key event. Used as the B5 demo driver.

use std::os::fd::{AsFd, OwnedFd};
use std::time::Duration;

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use rustix::fs::{MemfdFlags, ftruncate, memfd_create};
use wayland_client::{
    Connection, Dispatch, EventQueue, Proxy, QueueHandle,
    protocol::{
        wl_buffer::WlBuffer,
        wl_callback::WlCallback,
        wl_compositor::WlCompositor,
        wl_keyboard::{self, WlKeyboard},
        wl_pointer::{self, WlPointer},
        wl_registry::{self, WlRegistry},
        wl_seat::{self, Capability, WlSeat},
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
    seat: Option<WlSeat>,
    surface: Option<WlSurface>,
    xdg_surface: Option<XdgSurface>,
    toplevel: Option<XdgToplevel>,
    buffer: Option<WlBuffer>,
    mmap: Option<MmapMut>,
    configured: bool,
    palette: u8,
    done: bool,
    input_counter: u32,
    seat_has_input: bool,
}

impl App {
    fn new() -> Self {
        Self {
            compositor: None,
            shm: None,
            wm_base: None,
            seat: None,
            surface: None,
            xdg_surface: None,
            toplevel: None,
            buffer: None,
            mmap: None,
            configured: false,
            palette: 0,
            done: false,
            input_counter: 0,
            seat_has_input: false,
        }
    }

    fn repaint(&mut self) {
        if let Some(mmap) = self.mmap.as_mut() {
            paint_stripes(mmap, self.palette);
        }
        if let (Some(surface), Some(buffer)) = (self.surface.as_ref(), self.buffer.as_ref()) {
            surface.attach(Some(buffer), 0, 0);
            surface.damage_buffer(0, 0, W, H);
            surface.commit();
        }
    }

    fn cycle_palette(&mut self, reason: &str) {
        self.palette = (self.palette + 1) % 4;
        self.input_counter += 1;
        eprintln!(
            "paint: palette -> {} ({} event #{})",
            self.palette, reason, self.input_counter
        );
        self.repaint();
    }
}

fn make_memfd(size: usize) -> Result<OwnedFd> {
    let fd = memfd_create("hyprs-paint", MemfdFlags::CLOEXEC).context("memfd_create")?;
    ftruncate(&fd, size as u64).context("ftruncate")?;
    Ok(fd)
}

fn paint_stripes(bytes: &mut [u8], palette: u8) {
    // Three stripe palettes; each tuple is (top, middle, bottom), BGRA ordered
    // by u8 component directly. The 4th palette inverts the first for clear
    // visual difference after a full cycle.
    let ((r0, g0, b0), (r1, g1, b1), (r2, g2, b2)) = match palette {
        0 => ((0xff, 0x30, 0x30), (0x30, 0xff, 0x30), (0x30, 0x30, 0xff)),
        1 => ((0xff, 0xff, 0x30), (0x30, 0xff, 0xff), (0xff, 0x30, 0xff)),
        2 => ((0xff, 0xff, 0xff), (0x80, 0x80, 0x80), (0x20, 0x20, 0x20)),
        _ => ((0x00, 0xcf, 0xcf), (0xcf, 0x00, 0xcf), (0xcf, 0xcf, 0x00)),
    };
    for y in 0..H {
        let (r, g, b) = if y < H / 3 {
            (r0, g0, b0)
        } else if y < 2 * H / 3 {
            (r1, g1, b1)
        } else {
            (r2, g2, b2)
        };
        for x in 0..W {
            let idx = (y as usize * W as usize + x as usize) * 4;
            bytes[idx] = b;
            bytes[idx + 1] = g;
            bytes[idx + 2] = r;
            bytes[idx + 3] = 0xff;
        }
    }
}

fn build_buffer(
    shm: &WlShm,
    qh: &QueueHandle<App>,
) -> Result<(WlBuffer, MmapMut)> {
    let size = (W * H * 4) as usize;
    let fd = make_memfd(size)?;
    let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&fd)? };
    paint_stripes(&mut mmap, 0);
    let pool: WlShmPool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
    let buffer = pool.create_buffer(0, W, H, W * 4, Format::Argb8888, qh, ());
    pool.destroy();
    Ok((buffer, mmap))
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
    // wl_seat is optional — on headless backends it advertises no caps, but
    // get_pointer/get_keyboard still work even if they never fire events.
    if let Some(seat) = app.seat.as_ref() {
        let _pointer = seat.get_pointer(&qh, ());
        let _keyboard = seat.get_keyboard(&qh, ());
    }

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("hyprs-paint".into());
    toplevel.set_app_id("rs.hyperland.paint".into());
    surface.commit();
    app.surface = Some(surface);
    app.xdg_surface = Some(xdg_surface);
    app.toplevel = Some(toplevel);

    while !app.configured {
        queue.blocking_dispatch(&mut app)?;
    }

    let (buffer, mmap) = build_buffer(&shm, &qh)?;
    app.buffer = Some(buffer);
    app.mmap = Some(mmap);
    app.repaint();
    let surface = app.surface.as_ref().unwrap();
    let _first_cb: WlCallback = surface.frame(&qh, ());

    eprintln!("paint: ready. click in the window or press keys; close with xdg_toplevel.close");
    // Short deadline for servers that advertised no input caps (headless);
    // long deadline when real input is on the way.
    let short_deadline = std::time::Instant::now() + Duration::from_millis(800);
    let long_deadline = std::time::Instant::now() + Duration::from_secs(600);
    loop {
        queue.flush()?;
        queue.dispatch_pending(&mut app)?;
        if app.done {
            break;
        }
        let deadline = if app.seat_has_input {
            long_deadline
        } else {
            short_deadline
        };
        if std::time::Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}

// -- registry / globals --------------------------------------------------

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
                    state.compositor =
                        Some(reg.bind::<WlCompositor, _, _>(name, version.min(6), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(reg.bind::<WlShm, _, _>(name, version.min(1), qh, ()));
                }
                "wl_seat" => {
                    state.seat = Some(reg.bind::<WlSeat, _, _>(name, version.min(7), qh, ()));
                }
                "xdg_wm_base" => {
                    state.wm_base =
                        Some(reg.bind::<XdgWmBase, _, _>(name, version.min(5), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<WlCompositor, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlCompositor,
        _: <WlCompositor as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlSurface, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlSurface,
        _: <WlSurface as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShm, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShmPool, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlBuffer, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        _: <WlBuffer as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlCallback, ()> for App {
    fn event(
        _: &mut Self,
        _: &WlCallback,
        _: <WlCallback as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<XdgWmBase, ()> for App {
    fn event(
        _: &mut Self,
        base: &XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            base.pong(serial);
        }
    }
}
impl Dispatch<XdgSurface, ()> for App {
    fn event(
        state: &mut Self,
        xs: &XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xs.ack_configure(serial);
            state.configured = true;
        }
    }
}
impl Dispatch<XdgToplevel, ()> for App {
    fn event(
        state: &mut Self,
        _: &XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if matches!(event, xdg_toplevel::Event::Close) {
            state.done = true;
        }
    }
}

// -- input ---------------------------------------------------------------

impl Dispatch<WlSeat, ()> for App {
    fn event(
        state: &mut Self,
        _: &WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            if let wayland_client::WEnum::Value(c) = capabilities {
                let pointer = c.contains(Capability::Pointer);
                let keyboard = c.contains(Capability::Keyboard);
                eprintln!("paint: seat caps pointer={pointer} keyboard={keyboard}");
                state.seat_has_input = pointer || keyboard;
            }
        }
    }
}

impl Dispatch<WlPointer, ()> for App {
    fn event(
        state: &mut Self,
        _: &WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter { .. } => {
                eprintln!("paint: pointer enter");
            }
            wl_pointer::Event::Leave { .. } => {
                eprintln!("paint: pointer leave");
            }
            wl_pointer::Event::Button {
                button: _,
                state: bs,
                ..
            } => {
                if let wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed) = bs {
                    state.cycle_palette("button");
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlKeyboard, ()> for App {
    fn event(
        state: &mut Self,
        _: &WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Keymap { .. } => {
                eprintln!("paint: got keymap from server");
            }
            wl_keyboard::Event::Enter { .. } => {
                eprintln!("paint: keyboard enter");
            }
            wl_keyboard::Event::Leave { .. } => {
                eprintln!("paint: keyboard leave");
            }
            wl_keyboard::Event::Key {
                key,
                state: ks,
                ..
            } => {
                if let wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed) = ks {
                    state.cycle_palette(&format!("key {key}"));
                }
            }
            _ => {}
        }
    }
}

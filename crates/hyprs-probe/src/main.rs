//! Minimal Wayland client: connect, roundtrip, print every global the server
//! advertises, disconnect. Used to verify the hyperland-rs socket + globals.

use anyhow::{Context, Result};
use wayland_client::{Connection, Dispatch, QueueHandle, protocol::wl_registry};

struct App {
    globals: Vec<(u32, String, u32)>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for App {
    fn event(
        state: &mut Self,
        _reg: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            state.globals.push((name, interface, version));
        }
    }
}

fn main() -> Result<()> {
    let conn = Connection::connect_to_env().context("connect to WAYLAND_DISPLAY")?;
    let display = conn.display();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = display.get_registry(&qh, ());

    let mut app = App { globals: vec![] };
    queue.roundtrip(&mut app)?;

    println!("globals ({}):", app.globals.len());
    for (name, iface, ver) in &app.globals {
        println!("  {name:>3} {iface} v{ver}");
    }
    Ok(())
}

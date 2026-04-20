//! zxdg_decoration_manager_v1 — client/server-side decoration negotiation.
//!
//! Tiling compositors don't want client-side decorations (they'd draw title
//! bars that clash with the tile layout). We always respond `ServerSide`,
//! which tells the client to skip its own decorations. We don't actually
//! draw SSDs either — at tile borders the lack of chrome is intentional.
//! B10.6 scope: just enough stub to silence alacritty's CSD path.

use wayland_protocols::xdg::decoration::zv1::server::{
    zxdg_decoration_manager_v1::{self, ZxdgDecorationManagerV1},
    zxdg_toplevel_decoration_v1::{self, Mode, ZxdgToplevelDecorationV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::State;

pub const DECORATION_VERSION: u32 = 1;

impl GlobalDispatch<ZxdgDecorationManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZxdgDecorationManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let d = init.init(resource, ());
        tracing::info!(id = ?d.id(), "bind zxdg_decoration_manager_v1");
    }
}

impl Dispatch<ZxdgDecorationManagerV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZxdgDecorationManagerV1,
        request: zxdg_decoration_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_decoration_manager_v1::Request::GetToplevelDecoration {
                id,
                toplevel: _,
            } => {
                let deco = init.init(id, ());
                // Immediate configure so the client knows our preferred mode
                // before it commits its first buffer.
                deco.configure(Mode::ServerSide);
                tracing::debug!("xdg_decoration: configure(ServerSide)");
            }
            zxdg_decoration_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgToplevelDecorationV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &ZxdgToplevelDecorationV1,
        request: zxdg_toplevel_decoration_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_toplevel_decoration_v1::Request::SetMode { mode: _ } => {
                // Regardless of what the client asks for, we insist on SSD.
                // Re-configure to confirm.
                resource.configure(Mode::ServerSide);
            }
            zxdg_toplevel_decoration_v1::Request::UnsetMode => {
                resource.configure(Mode::ServerSide);
            }
            zxdg_toplevel_decoration_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

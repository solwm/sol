//! Minimal xdg-shell: xdg_wm_base, xdg_surface, xdg_toplevel. Just enough for
//! a client to create a toplevel, receive a configure, ack it, and commit a
//! first buffer.

use std::sync::{Arc, Mutex};

use wayland_protocols::xdg::shell::server::{
    xdg_popup::{self, XdgPopup},
    xdg_positioner::{self, XdgPositioner},
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_surface::WlSurface,
};

use crate::{State, compositor::SurfaceData};

/// Canonical `states` payload for every configure voidptr sends. Includes
/// MAXIMIZED (tells the client "obey this size, per the xdg-shell spec")
/// plus TILED_* on all four edges (tells tiling-aware clients that they
/// border other tiles/the screen so they shouldn't draw external shadows
/// or allow resize by dragging edges) plus ACTIVATED (styling hint).
///
/// Every u32 value is written in native byte order; the wire protocol
/// length prefix is added by wayland-server.
pub(crate) fn tile_state_bytes() -> Vec<u8> {
    const STATE_MAXIMIZED: u32 = 1;
    const STATE_ACTIVATED: u32 = 4;
    const STATE_TILED_LEFT: u32 = 5;
    const STATE_TILED_RIGHT: u32 = 6;
    const STATE_TILED_TOP: u32 = 7;
    const STATE_TILED_BOTTOM: u32 = 8;
    let vals = [
        STATE_MAXIMIZED,
        STATE_ACTIVATED,
        STATE_TILED_LEFT,
        STATE_TILED_RIGHT,
        STATE_TILED_TOP,
        STATE_TILED_BOTTOM,
    ];
    let mut buf = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        buf.extend_from_slice(&v.to_ne_bytes());
    }
    buf
}

/// Tracks the surface a given xdg_surface wraps, so toplevel requests can
/// reach through to modify the underlying compositor state.
pub struct XdgSurfaceData {
    pub wl_surface: WlSurface,
    pub surface_data: Arc<Mutex<SurfaceData>>,
}

impl GlobalDispatch<XdgWmBase, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<XdgWmBase>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let wm = init.init(resource, ());
        tracing::info!(id = ?wm.id(), "bind xdg_wm_base");
    }
}

impl Dispatch<XdgWmBase, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        wm: &XdgWmBase,
        request: xdg_wm_base::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_wm_base::Request::GetXdgSurface { id, surface } => {
                let sd = surface
                    .data::<Arc<Mutex<SurfaceData>>>()
                    .expect("wl_surface without SurfaceData");
                let _ = init.init(
                    id,
                    XdgSurfaceData {
                        wl_surface: surface.clone(),
                        surface_data: sd.clone(),
                    },
                );
            }
            xdg_wm_base::Request::CreatePositioner { id } => {
                let _ = init.init(id, ());
            }
            xdg_wm_base::Request::Pong { serial: _ } => {}
            xdg_wm_base::Request::Destroy => {}
            _ => {
                wm.post_error(xdg_wm_base::Error::InvalidPopupParent, "unsupported");
            }
        }
    }
}

impl Dispatch<XdgSurface, XdgSurfaceData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        xs: &XdgSurface,
        request: xdg_surface::Request,
        data: &XdgSurfaceData,
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_surface::Request::GetToplevel { id } => {
                let toplevel = init.init(id, data.wl_surface.clone());
                {
                    let mut sd = data.surface_data.lock().unwrap();
                    sd.role = crate::compositor::SurfaceRole::XdgToplevel { mapped: false };
                    sd.xdg_toplevel = Some(toplevel.downgrade());
                    sd.xdg_surface = Some(xs.downgrade());
                }
                // Give the client a real initial tile size with MAXIMIZED +
                // ACTIVATED set, so it commits its first buffer at this
                // size rather than whatever its hard-coded default is.
                // Once the window actually maps, apply_layout re-sends a
                // configure with the precise tile rect.
                let sw = state.screen_width as i32;
                let sh = state.screen_height as i32;
                toplevel.configure(sw, sh, tile_state_bytes());
                let serial = state.next_serial();
                xs.configure(serial);
                tracing::info!(width = sw, height = sh, "initial xdg_toplevel.configure");
            }
            xdg_surface::Request::GetPopup { id, .. } => {
                let _ = init.init(id, ());
                tracing::warn!("xdg popup not implemented at B2");
            }
            xdg_surface::Request::AckConfigure { serial } => {
                tracing::info!(serial, "client ack_configure");
            }
            xdg_surface::Request::SetWindowGeometry { .. } => {}
            xdg_surface::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgToplevel, WlSurface> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &XdgToplevel,
        request: xdg_toplevel::Request,
        surface: &WlSurface,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_toplevel::Request::SetTitle { title } => {
                tracing::info!(id = ?surface.id(), %title, "toplevel title");
            }
            xdg_toplevel::Request::SetAppId { app_id } => {
                tracing::info!(id = ?surface.id(), %app_id, "toplevel app_id");
            }
            xdg_toplevel::Request::Destroy => {
                state
                    .mapped_toplevels
                    .retain(|w| w.surface.upgrade().ok().as_ref() != Some(surface));
                state.needs_render = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<XdgPositioner, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &XdgPositioner,
        _request: xdg_positioner::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<XdgPopup, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &XdgPopup,
        _request: xdg_popup::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

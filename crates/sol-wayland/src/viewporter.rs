//! `wp_viewporter` — per-surface source crop + destination logical size.
//!
//! **Semantics** (matching Hyprland's interpretation, which matches
//! what Chromium/Ozone expects):
//!
//! - `set_source(x, y, w, h)` → sub-rect of the buffer that the
//!   compositor should sample from, in buffer coordinates. Stored
//!   on the surface's `SurfaceData`; at render time we translate to
//!   UVs by dividing by the buffer size, and the shader samples
//!   the corresponding texture patch.
//!
//! - `set_destination(w, h)` → the surface's **logical** size (for
//!   input hit-testing, damage math, a client's "how big am I?"
//!   answer). It is **not** the on-screen output rect. The
//!   compositor still places the surface at whatever position and
//!   size it chose (tile rect for toplevels, anchor rect for
//!   layer surfaces). Mis-interpreting destination as "stretch the
//!   buffer to this size and draw at that size" (which an earlier
//!   implementation here did) broke Chrome's move/resize path
//!   because Chrome's Ozone assumes the compositor respects the
//!   buffer-to-output geometry implied by its `xdg_toplevel` size.
//!
//! Wire-format note: source coords are `wl_fixed_t` (24.8). The
//! `wayland-protocols` bindings hand them to us as pre-converted
//! `f64`, so no manual shift.

use std::sync::{Arc, Mutex};

use wayland_protocols::wp::viewporter::server::{
    wp_viewport::{self, WpViewport},
    wp_viewporter::{self, WpViewporter},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_surface::WlSurface,
};

use crate::{State, compositor::SurfaceData};

pub const VIEWPORTER_VERSION: u32 = 1;

/// User-data attached to a `wp_viewport`: the surface it was
/// associated with. We need it on destroy so we can clear the
/// `viewport_dst` field on the surface's SurfaceData.
pub struct ViewportData {
    pub surface: WlSurface,
}

impl GlobalDispatch<WpViewporter, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpViewporter>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let v = init.init(resource, ());
        tracing::info!(id = ?v.id(), "bind wp_viewporter");
    }
}

impl Dispatch<WpViewporter, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpViewporter,
        request: wp_viewporter::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_viewporter::Request::GetViewport { id, surface } => {
                let _ = init.init(
                    id,
                    ViewportData {
                        surface: surface.clone(),
                    },
                );
            }
            wp_viewporter::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpViewport, ViewportData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpViewport,
        request: wp_viewport::Request,
        data: &ViewportData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_viewport::Request::SetDestination { width, height } => {
                // -1/-1 per spec means "unset." Anything ≤ 0 we treat
                // as unset so a client that set a dest and then
                // wants to clear it works correctly.
                let dst = if width > 0 && height > 0 {
                    Some((width, height))
                } else {
                    None
                };
                tracing::debug!(
                    surface = ?data.surface.id(),
                    width, height,
                    dst = ?dst,
                    "viewport.set_destination"
                );
                if let Some(sd_arc) =
                    data.surface.data::<Arc<Mutex<SurfaceData>>>()
                {
                    sd_arc.lock().unwrap().viewport_dst = dst;
                }
            }
            wp_viewport::Request::SetSource { x, y, width, height } => {
                tracing::debug!(
                    surface = ?data.surface.id(),
                    x, y, width, height,
                    "viewport.set_source"
                );
                // -1 on any field per spec means "unset" (clear the
                // source). A valid rect has all four >= 0 with w, h
                // strictly positive; otherwise bail and clear.
                let src = if x >= 0.0 && y >= 0.0 && width > 0.0 && height > 0.0
                {
                    Some((x, y, width, height))
                } else {
                    None
                };
                if let Some(sd_arc) =
                    data.surface.data::<Arc<Mutex<SurfaceData>>>()
                {
                    sd_arc.lock().unwrap().viewport_src = src;
                }
            }
            wp_viewport::Request::Destroy => {
                if let Some(sd_arc) =
                    data.surface.data::<Arc<Mutex<SurfaceData>>>()
                {
                    let mut sd = sd_arc.lock().unwrap();
                    sd.viewport_dst = None;
                    sd.viewport_src = None;
                }
            }
            _ => {}
        }
    }
}

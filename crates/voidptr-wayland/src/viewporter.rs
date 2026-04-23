//! `wp_viewporter` — per-surface source crop + destination scale.
//!
//! awww (and many wallpaper daemons, video apps, and browsers) require
//! this protocol *mandatorily* — awww unwraps the global reference
//! and panics at startup if we don't advertise it. Implementing at
//! least the stub unblocks those clients.
//!
//! Current scope: advertise the global, hand out `wp_viewport` per
//! surface on request, track the `set_destination` size on the
//! surface's `SurfaceData` so the scene walker can stretch the
//! buffer to that size. `set_source` (source-rect cropping in
//! fixed-point surface coords) is accepted silently — we don't
//! honor it yet, but no known client we care about needs it for
//! rendering correctness.

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
                if let Some(sd_arc) =
                    data.surface.data::<Arc<Mutex<SurfaceData>>>()
                {
                    sd_arc.lock().unwrap().viewport_dst = dst;
                }
            }
            wp_viewport::Request::SetSource { .. } => {
                // Source cropping isn't honored yet — see module doc.
            }
            wp_viewport::Request::Destroy => {
                if let Some(sd_arc) =
                    data.surface.data::<Arc<Mutex<SurfaceData>>>()
                {
                    sd_arc.lock().unwrap().viewport_dst = None;
                }
            }
            _ => {}
        }
    }
}

//! `wl_subcompositor` + `wl_subsurface`.
//!
//! Required for GTK/Qt clients (Firefox, most desktop apps) to finish
//! initializing — they create subsurfaces for tooltips, popups, and
//! cursor images during startup and refuse to proceed if the global
//! isn't advertised.
//!
//! B9+ polish scope: accept the protocol so clients can init. We
//! assign `SurfaceRole::Subsurface` to the child surface so the main
//! commit handler doesn't treat its buffers as a toplevel, but we
//! don't actually render subsurfaces yet — tooltips/popups will be
//! invisible. Main windows (xdg_toplevel) are unaffected.

use std::sync::{Arc, Mutex};

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::{
        wl_subcompositor::{self, WlSubcompositor},
        wl_subsurface::{self, WlSubsurface},
        wl_surface::WlSurface,
    },
};

use crate::{State, compositor::{SurfaceData, SurfaceRole}};

pub const SUBCOMPOSITOR_VERSION: u32 = 1;

pub struct SubsurfaceData {
    pub surface: WlSurface,
    pub parent: WlSurface,
}

impl GlobalDispatch<WlSubcompositor, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSubcompositor>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let sc = init.init(resource, ());
        tracing::info!(id = ?sc.id(), "bind wl_subcompositor");
    }
}

impl Dispatch<WlSubcompositor, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &WlSubcompositor,
        request: wl_subcompositor::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_subcompositor::Request::GetSubsurface {
                id,
                surface,
                parent,
            } => {
                let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
                    resource.post_error(
                        wl_subcompositor::Error::BadSurface,
                        "wl_surface missing SurfaceData",
                    );
                    return;
                };
                {
                    let mut sd = sd_arc.lock().unwrap();
                    if !matches!(sd.role, SurfaceRole::None) {
                        resource.post_error(
                            wl_subcompositor::Error::BadSurface,
                            "wl_surface already has a role",
                        );
                        return;
                    }
                    sd.role = SurfaceRole::Subsurface;
                }
                let _ = init.init(
                    id,
                    SubsurfaceData {
                        surface: surface.clone(),
                        parent,
                    },
                );
                tracing::debug!("wl_subcompositor.get_subsurface");
            }
            wl_subcompositor::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlSubsurface, SubsurfaceData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSubsurface,
        request: wl_subsurface::Request,
        _data: &SubsurfaceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // All set_position / place_above / place_below / set_sync / set_desync
        // / destroy requests are accepted but not acted on. This is enough
        // for clients to finish initialization — we'll wire real subsurface
        // rendering in a follow-up milestone.
        match request {
            wl_subsurface::Request::Destroy
            | wl_subsurface::Request::SetPosition { .. }
            | wl_subsurface::Request::PlaceAbove { .. }
            | wl_subsurface::Request::PlaceBelow { .. }
            | wl_subsurface::Request::SetSync
            | wl_subsurface::Request::SetDesync => {}
            _ => {}
        }
    }
}

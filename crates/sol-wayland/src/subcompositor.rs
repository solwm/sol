//! `wl_subcompositor` + `wl_subsurface`.
//!
//! Required for GTK/Qt clients (Firefox, most desktop apps) to finish
//! initializing — they create subsurfaces for tooltips, popups, and
//! cursor images during startup.
//!
//! Now actually renders subsurfaces: `get_subsurface` links the child
//! into the parent's `subsurface_children` list; `set_position` stores
//! a pending offset applied on commit; the scene walker in
//! sol-wayland/src/lib.rs recurses through the tree from any
//! mapped xdg_toplevel or layer_surface root, emitting each subsurface
//! at `parent_origin + offset`.
//!
//! Out of scope for now: explicit sync vs desync semantics (we treat
//! every commit as immediate), `place_above` / `place_below` sibling
//! ordering (we use registration order), and subsurface damage
//! tracking.

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

/// Walk upward from a surface toward its role-holding root (xdg_toplevel
/// or layer_surface). Used at destroy time so we can clean up weak refs
/// on the right parent's `subsurface_children`. Returns the immediate
/// parent for surfaces with a Subsurface role; None otherwise.
fn parent_of(surface: &WlSurface) -> Option<WlSurface> {
    let sd_arc = surface.data::<Arc<Mutex<SurfaceData>>>()?;
    let sd = sd_arc.lock().ok()?;
    let parent_weak = sd.subsurface_parent.clone()?;
    parent_weak.upgrade().ok()
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
                let Some(child_sd) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
                    resource.post_error(
                        wl_subcompositor::Error::BadSurface,
                        "child wl_surface missing SurfaceData",
                    );
                    return;
                };
                let Some(parent_sd) = parent.data::<Arc<Mutex<SurfaceData>>>() else {
                    resource.post_error(
                        wl_subcompositor::Error::BadSurface,
                        "parent wl_surface missing SurfaceData",
                    );
                    return;
                };
                {
                    let mut cs = child_sd.lock().unwrap();
                    if !matches!(cs.role, SurfaceRole::None) {
                        resource.post_error(
                            wl_subcompositor::Error::BadSurface,
                            "wl_surface already has a role",
                        );
                        return;
                    }
                    cs.role = SurfaceRole::Subsurface;
                    cs.subsurface_parent = Some(parent.downgrade());
                }
                // Register the child on the parent's children list so
                // the scene walker finds it when it recurses down from
                // the mapped root.
                parent_sd
                    .lock()
                    .unwrap()
                    .subsurface_children
                    .push(surface.downgrade());
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
        data: &SubsurfaceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_subsurface::Request::SetPosition { x, y } => {
                // Double-buffer the offset. The main commit handler in
                // compositor.rs promotes pending → current on the next
                // wl_surface.commit. Matches the spec's double-buffered
                // state model.
                if let Some(sd_arc) = data.surface.data::<Arc<Mutex<SurfaceData>>>() {
                    sd_arc.lock().unwrap().subsurface_pending_offset = Some((x, y));
                }
            }
            wl_subsurface::Request::Destroy => {
                // Remove ourselves from the parent's children list so
                // the scene walker doesn't try to upgrade a dead weak
                // ref every frame. Order matters: read the parent link
                // before we clear it below, otherwise parent_of would
                // return None.
                if let Some(parent) = parent_of(&data.surface) {
                    if let Some(parent_sd) =
                        parent.data::<Arc<Mutex<SurfaceData>>>()
                    {
                        let child_id = data.surface.id();
                        parent_sd.lock().unwrap().subsurface_children.retain(|w| {
                            w.upgrade()
                                .ok()
                                .map(|s| s.id() != child_id)
                                .unwrap_or(false)
                        });
                    }
                }
                // Reset the child's role back to None so the wl_surface
                // can be reassigned later. Chrome's popup machinery
                // (address bar dropdown, menus) recycles wl_surfaces
                // through create→destroy→reuse-as-subsurface cycles;
                // without this reset we'd fail the reuse with
                // "wl_surface already has a role" and Chrome's
                // fallback path draws with black edges / missing
                // fragments. Per spec: `wl_subsurface.destroy` removes
                // the role and unmaps the surface immediately.
                if let Some(sd_arc) = data.surface.data::<Arc<Mutex<SurfaceData>>>() {
                    let mut sd = sd_arc.lock().unwrap();
                    sd.role = SurfaceRole::None;
                    sd.subsurface_parent = None;
                    sd.subsurface_offset = (0, 0);
                    sd.subsurface_pending_offset = None;
                }
            }
            // place_above / place_below / set_sync / set_desync all
            // silently accepted — not implemented but also not fatal
            // to clients that call them.
            _ => {}
        }
    }
}

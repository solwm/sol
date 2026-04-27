//! Minimal xdg-shell: xdg_wm_base, xdg_surface, xdg_toplevel, xdg_popup.
//!
//! Toplevels get a tile-sized initial configure and join the layout on
//! their first buffered commit. Popups (right-click context menus,
//! dropdowns, tooltips) get an xdg_positioner-derived placement
//! relative to their parent and render as a separate top-of-z-order
//! pass so they appear over toplevels and overlay layer surfaces.

use std::sync::{Arc, Mutex};

use wayland_protocols::xdg::shell::server::{
    xdg_popup::{self, XdgPopup},
    xdg_positioner::{self, Anchor, Gravity, XdgPositioner},
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    WEnum, protocol::wl_surface::WlSurface,
};

use crate::{State, compositor::SurfaceData};

/// Canonical `states` payload for every configure sol sends. Includes
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

/// Per-`xdg_positioner` mutable state. The protocol is incremental:
/// the client sets fields one request at a time and `xdg_surface.get_popup`
/// snapshots them. Wrapped in a Mutex so the `Dispatch::request` impl
/// (which holds `&PositionerData`) can mutate.
#[derive(Default)]
pub struct PositionerData {
    pub state: Mutex<PositionerState>,
}

#[derive(Clone, Copy, Debug)]
pub struct PositionerState {
    /// Logical size the popup wants to be (`set_size`).
    pub size: (i32, i32),
    /// Anchor rect in parent-local coords (`set_anchor_rect`).
    pub anchor_rect: (i32, i32, i32, i32),
    /// Which edge of the anchor rect the popup attaches to.
    pub anchor: Anchor,
    /// Direction the popup grows from the anchor point.
    pub gravity: Gravity,
    /// Extra translation applied after anchor + gravity (`set_offset`).
    pub offset: (i32, i32),
}

impl Default for PositionerState {
    fn default() -> Self {
        Self {
            size: (0, 0),
            anchor_rect: (0, 0, 0, 0),
            anchor: Anchor::None,
            gravity: Gravity::None,
            offset: (0, 0),
        }
    }
}

/// User-data on each `XdgPopup`: lets `popup_done` dispatch find the
/// surface to clean up without re-walking the toplevel/popup tree.
pub struct XdgPopupData {
    pub wl_surface: WlSurface,
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
        _wm: &XdgWmBase,
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
                let _ = init.init(id, PositionerData::default());
            }
            xdg_wm_base::Request::Pong { serial: _ } => {}
            xdg_wm_base::Request::Destroy => {}
            _ => {}
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
                // Initial configure with size 0,0 and no states tells
                // the client "pick your own size for now". This is the
                // only signal we have at this point that lets dialogs
                // (xdg_toplevel with set_parent) actually use their
                // preferred size — if we sent MAXIMIZED+screen-sized
                // here, every save/discard prompt would draw at the
                // full screen and ignore its own intrinsic dims.
                // Tiled toplevels get their proper sized + MAXIMIZED
                // configure from `apply_layout` once they map; the
                // brief moment of "client at preferred size" before
                // that arrives is invisible because we don't render a
                // toplevel until it has a buffer anyway.
                toplevel.configure(0, 0, Vec::new());
                let serial = state.next_serial();
                xs.configure(serial);
                tracing::debug!(?serial, "initial xdg_toplevel.configure(0,0)");
            }
            xdg_surface::Request::GetPopup {
                id,
                parent,
                positioner,
            } => {
                let parent_surface = parent
                    .as_ref()
                    .and_then(|xp| xp.data::<XdgSurfaceData>().map(|d| d.wl_surface.clone()));
                let pos_state = positioner
                    .data::<PositionerData>()
                    .and_then(|d| d.state.lock().ok().map(|s| *s))
                    .unwrap_or_default();

                let popup = init.init(
                    id,
                    XdgPopupData {
                        wl_surface: data.wl_surface.clone(),
                    },
                );

                let (px, py) = compute_popup_position(&pos_state);
                let (pw, ph) = pos_state.size;

                {
                    let mut sd = data.surface_data.lock().unwrap();
                    sd.role = crate::compositor::SurfaceRole::XdgPopup {
                        mapped: false,
                        offset: (px, py),
                        size: (pw, ph),
                    };
                    sd.xdg_surface = Some(xs.downgrade());
                    sd.xdg_popup = Some(popup.downgrade());
                    sd.xdg_popup_parent = parent_surface.as_ref().map(|s| s.downgrade());
                }

                // Initial configure: position + size, then xdg_surface
                // serial. The client acks and commits its first buffer.
                popup.configure(px, py, pw.max(1), ph.max(1));
                let serial = state.next_serial();
                xs.configure(serial);
                tracing::debug!(
                    x = px,
                    y = py,
                    w = pw,
                    h = ph,
                    has_parent = parent_surface.is_some(),
                    "xdg_popup configured"
                );
            }
            xdg_surface::Request::AckConfigure { serial } => {
                tracing::debug!(serial, "client ack_configure");
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
            xdg_toplevel::Request::SetParent { parent } => {
                // Parent links a transient window (dialog, file
                // picker, preferences) to its owner toplevel. The
                // protocol lets clients call this at any time, and
                // GTK clients (GIMP among them) often call it AFTER
                // the dialog's first commit — we have to handle the
                // late case by reclassifying the surface live, not
                // just at first-map time.
                let parent_surface = parent.as_ref().and_then(|tl| {
                    tl.data::<WlSurface>().cloned()
                });
                if let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() {
                    let mut sd = sd_arc.lock().unwrap();
                    sd.xdg_toplevel_parent = parent_surface.as_ref().map(|s| s.downgrade());
                }
                tracing::info!(
                    id = ?surface.id(),
                    parent = ?parent_surface.as_ref().map(|s| s.id()),
                    "toplevel set_parent"
                );
                crate::reclassify_dialog(state, surface, parent_surface.as_ref());
            }
            xdg_toplevel::Request::SetTitle { title } => {
                tracing::info!(id = ?surface.id(), %title, "toplevel title");
            }
            xdg_toplevel::Request::SetAppId { app_id } => {
                tracing::info!(id = ?surface.id(), %app_id, "toplevel app_id");
            }
            xdg_toplevel::Request::Destroy => {
                crate::unmap_toplevel(state, surface);
            }
            _ => {}
        }
    }
}

impl Dispatch<XdgPositioner, PositionerData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &XdgPositioner,
        request: xdg_positioner::Request,
        data: &PositionerData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        let mut s = match data.state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        match request {
            xdg_positioner::Request::SetSize { width, height } => {
                s.size = (width, height);
            }
            xdg_positioner::Request::SetAnchorRect { x, y, width, height } => {
                s.anchor_rect = (x, y, width, height);
            }
            xdg_positioner::Request::SetAnchor { anchor } => {
                if let WEnum::Value(a) = anchor {
                    s.anchor = a;
                }
            }
            xdg_positioner::Request::SetGravity { gravity } => {
                if let WEnum::Value(g) = gravity {
                    s.gravity = g;
                }
            }
            xdg_positioner::Request::SetOffset { x, y } => {
                s.offset = (x, y);
            }
            // ConstraintAdjustment, SetReactive, SetParentSize,
            // SetParentConfigure: best-effort ignored. Without
            // reactive constraint solving popups can land off-screen
            // for some apps; clients that care provide a sane
            // anchor rect anyway.
            _ => {}
        }
    }
}

impl Dispatch<XdgPopup, XdgPopupData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &XdgPopup,
        request: xdg_popup::Request,
        data: &XdgPopupData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_popup::Request::Grab { seat: _, serial: _ } => {
                // Track this popup as the current grabbing popup so a
                // click outside its tree dismisses it via `popup_done`.
                state.popup_grab = Some(resource.downgrade());
            }
            xdg_popup::Request::Reposition { positioner, token } => {
                let pos_state = positioner
                    .data::<PositionerData>()
                    .and_then(|d| d.state.lock().ok().map(|s| *s))
                    .unwrap_or_default();
                let (px, py) = compute_popup_position(&pos_state);
                let (pw, ph) = pos_state.size;
                if let Some(sd_arc) = data.wl_surface.data::<Arc<Mutex<SurfaceData>>>() {
                    let mut sd = sd_arc.lock().unwrap();
                    sd.role = crate::compositor::SurfaceRole::XdgPopup {
                        mapped: matches!(
                            sd.role,
                            crate::compositor::SurfaceRole::XdgPopup { mapped: true, .. }
                        ),
                        offset: (px, py),
                        size: (pw, ph),
                    };
                }
                resource.repositioned(token);
                resource.configure(px, py, pw.max(1), ph.max(1));
            }
            xdg_popup::Request::Destroy => {
                state.mapped_popups.retain(|w| {
                    w.upgrade().ok().as_ref() != Some(&data.wl_surface)
                });
                if state
                    .popup_grab
                    .as_ref()
                    .and_then(|w| w.upgrade().ok())
                    .map(|p| p.id() == resource.id())
                    .unwrap_or(false)
                {
                    state.popup_grab = None;
                }
                state.needs_render = true;
            }
            _ => {}
        }
    }
}

/// Translate xdg_positioner state into a popup top-left in the
/// parent's surface-local coordinate frame.
///
/// `anchor` picks a point on `anchor_rect`; `gravity` picks which
/// corner of the popup sits on that anchor point. `offset` is added
/// at the end. We don't apply constraint adjustment — popups that
/// would land off-screen for one reason or another stay where the
/// client asked. Clients that want safety provide a generous anchor
/// rect / try multiple positioners on reposition.
pub fn compute_popup_position(p: &PositionerState) -> (i32, i32) {
    let (rx, ry, rw, rh) = p.anchor_rect;
    let cx = rx + rw / 2;
    let cy = ry + rh / 2;
    // Anchor point on the rect.
    let (ax, ay) = match p.anchor {
        Anchor::Top => (cx, ry),
        Anchor::Bottom => (cx, ry + rh),
        Anchor::Left => (rx, cy),
        Anchor::Right => (rx + rw, cy),
        Anchor::TopLeft => (rx, ry),
        Anchor::BottomLeft => (rx, ry + rh),
        Anchor::TopRight => (rx + rw, ry),
        Anchor::BottomRight => (rx + rw, ry + rh),
        // None / Center / unknown
        _ => (cx, cy),
    };
    let (pw, ph) = p.size;
    // Gravity is the direction the popup grows from the anchor point.
    // A "Bottom" gravity puts the anchor at the popup's top edge, so
    // popup top-left x = ax - pw/2, y = ay.
    let (gx, gy) = match p.gravity {
        Gravity::Top => (ax - pw / 2, ay - ph),
        Gravity::Bottom => (ax - pw / 2, ay),
        Gravity::Left => (ax - pw, ay - ph / 2),
        Gravity::Right => (ax, ay - ph / 2),
        Gravity::TopLeft => (ax - pw, ay - ph),
        Gravity::BottomLeft => (ax - pw, ay),
        Gravity::TopRight => (ax, ay - ph),
        Gravity::BottomRight => (ax, ay),
        // None / Center / unknown: popup centered on the anchor point.
        _ => (ax - pw / 2, ay - ph / 2),
    };
    (gx + p.offset.0, gy + p.offset.1)
}

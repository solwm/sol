//! `zwlr_layer_shell_v1` — surfaces that sit outside the tile layout and
//! anchor to output edges. This is what waybar, rofi, swaybg, and
//! mako use.
//!
//! A layer surface starts at one of four z-depths (background, bottom,
//! top, overlay), can anchor to any combination of edges, reserves an
//! optional "exclusive zone" along its anchored edge so tiled toplevels
//! don't draw over it, and can request keyboard focus via
//! `keyboard_interactivity` (`exclusive` for launchers like rofi, `none`
//! for bars).
//!
//! B9.1/B9.2 scope: protocol skeleton + configure/ack flow. The server
//! accepts every request, double-buffers the state, and drives the
//! configure sequence so clients proceed to map. Scene integration,
//! input routing, and exclusive-zone layout all land in B9.3–B9.5.

use std::sync::Mutex;

use wayland_protocols_wlr::layer_shell::v1::server::{
    zwlr_layer_shell_v1::{self, Layer, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, ZwlrLayerSurfaceV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    WEnum,
    protocol::wl_surface::WlSurface,
};

use crate::{State, compositor::SurfaceRole};

pub const LAYER_SHELL_VERSION: u32 = 4;

/// Double-buffered portion of a layer surface's own state. The spec
/// requires everything set via `set_*` requests to be latched on
/// `wl_surface.commit`, not on the request itself.
#[derive(Clone, Debug, Default)]
pub struct LayerState {
    pub size: (u32, u32),
    /// Anchor bitfield — `top|bottom|left|right` per the anchor enum.
    pub anchor: u32,
    pub margin: (i32, i32, i32, i32), // top, right, bottom, left
    pub exclusive_zone: i32,
    pub keyboard_interactivity: u32,
    pub layer: u32,
}


/// User-data for a live `zwlr_layer_surface_v1` resource.
pub struct LayerSurfaceData {
    pub wl_surface: WlSurface,
    pub namespace: String,
    pub inner: Mutex<LayerSurfaceInner>,
}

pub struct LayerSurfaceInner {
    pub pending: LayerState,
    pub current: LayerState,
    /// Serial of the most recent configure we sent; cleared when the
    /// client acks it.
    pub last_configure_serial: u32,
    /// Most recent serial the client acked. We require `acked == last`
    /// before treating a buffered commit as a map.
    pub last_acked_serial: u32,
}

impl LayerSurfaceInner {
    fn new(initial_layer: u32) -> Self {
        let pending = LayerState {
            layer: initial_layer,
            ..LayerState::default()
        };
        let current = pending.clone();
        Self {
            pending,
            current,
            last_configure_serial: 0,
            last_acked_serial: 0,
        }
    }
}

impl GlobalDispatch<ZwlrLayerShellV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwlrLayerShellV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let s = init.init(resource, ());
        tracing::info!(id = ?s.id(), "bind zwlr_layer_shell_v1");
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwlrLayerShellV1,
        request: zwlr_layer_shell_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_layer_shell_v1::Request::GetLayerSurface {
                id,
                surface,
                output: _,
                layer,
                namespace,
            } => {
                let layer_num = match layer.into_result() {
                    Ok(l) => l as u32,
                    Err(_) => {
                        resource.post_error(
                            zwlr_layer_shell_v1::Error::InvalidLayer,
                            "invalid layer value",
                        );
                        return;
                    }
                };

                // Reject if the surface already has a role or a buffer.
                let bad_role = crate::compositor::with_sol_data(&surface, |sd| {
                    !matches!(sd.role, SurfaceRole::None)
                })
                .unwrap_or(false);
                if bad_role {
                    resource.post_error(
                        zwlr_layer_shell_v1::Error::Role,
                        "wl_surface already has a role",
                    );
                    return;
                }
                let has_buffer = crate::compositor::with_sol_data(&surface, |sd| {
                    sd.current_buffer.is_some()
                })
                .unwrap_or(false);
                if has_buffer {
                    resource.post_error(
                        zwlr_layer_shell_v1::Error::AlreadyConstructed,
                        "wl_surface already has a buffer attached",
                    );
                    return;
                }

                let data = LayerSurfaceData {
                    wl_surface: surface.clone(),
                    namespace: namespace.clone(),
                    inner: Mutex::new(LayerSurfaceInner::new(layer_num)),
                };
                let ls = init.init(id, data);

                crate::compositor::with_sol_data_mut(&surface, |sd| {
                    sd.role = SurfaceRole::LayerSurface {
                        mapped: false,
                        initial_configure_sent: false,
                    };
                    sd.zwlr_layer_surface = Some(ls.downgrade());
                });

                // Track membership for the next render tick (scene assembly
                // + apply_layout). We use a separate list from mapped_toplevels
                // because layer surfaces live outside the tile layout.
                state
                    .pending_layer_surfaces
                    .push(surface.downgrade());
                state.needs_render = true;

                tracing::info!(
                    ?namespace,
                    layer = layer_num,
                    id = ?ls.id(),
                    "layer surface created"
                );
            }
            zwlr_layer_shell_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, LayerSurfaceData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZwlrLayerSurfaceV1,
        request: zwlr_layer_surface_v1::Request,
        data: &LayerSurfaceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        let mut inner = data.inner.lock().unwrap();
        match request {
            zwlr_layer_surface_v1::Request::SetSize { width, height } => {
                inner.pending.size = (width, height);
            }
            zwlr_layer_surface_v1::Request::SetAnchor { anchor } => {
                if let WEnum::Value(a) = anchor {
                    inner.pending.anchor = a.bits();
                }
            }
            zwlr_layer_surface_v1::Request::SetExclusiveZone { zone } => {
                inner.pending.exclusive_zone = zone;
            }
            zwlr_layer_surface_v1::Request::SetMargin {
                top,
                right,
                bottom,
                left,
            } => {
                inner.pending.margin = (top, right, bottom, left);
            }
            zwlr_layer_surface_v1::Request::SetKeyboardInteractivity {
                keyboard_interactivity,
            } => {
                if let WEnum::Value(k) = keyboard_interactivity {
                    inner.pending.keyboard_interactivity = k as u32;
                }
            }
            zwlr_layer_surface_v1::Request::SetLayer { layer } => {
                if let WEnum::Value(l) = layer {
                    inner.pending.layer = l as u32;
                }
            }
            zwlr_layer_surface_v1::Request::AckConfigure { serial } => {
                inner.last_acked_serial = serial;
            }
            zwlr_layer_surface_v1::Request::Destroy => {
                // Evicting the surface from state.mapped_layers happens
                // on the next render tick via the usual dead-weak prune.
                state.needs_render = true;
            }
            zwlr_layer_surface_v1::Request::GetPopup { popup: _ } => {
                // Popups on layer surfaces aren't implemented at B9.
            }
            _ => {}
        }
    }
}

/// Convert the raw numeric layer (as stored in `LayerState.layer`) into
/// a strongly-typed Layer for z-order comparisons.
pub fn layer_of(raw: u32) -> Layer {
    match raw {
        0 => Layer::Background,
        1 => Layer::Bottom,
        2 => Layer::Top,
        3 => Layer::Overlay,
        _ => Layer::Bottom,
    }
}

/// Compute the screen-space rect for a layer surface given its anchor
/// bitfield, margin, size, and the full screen rect. Mirrors what sway
/// and wlroots do: the anchor determines where the bar sits; margins
/// push it inward from that edge; set_size gives the intrinsic
/// dimensions on axes the surface isn't stretched along.
pub fn compute_layer_rect(current: &LayerState, screen: crate::Rect) -> crate::Rect {
    const TOP: u32 = 1;
    const BOTTOM: u32 = 2;
    const LEFT: u32 = 4;
    const RIGHT: u32 = 8;

    let (m_t, m_r, m_b, m_l) = current.margin;
    let (size_w, size_h) = current.size;
    let a = current.anchor;

    // Width: stretched if anchored left+right, else use set_size (or fall
    // back to a quarter of the screen so a forgot-to-set-size surface
    // still shows up for diagnostics).
    let w = if a & LEFT != 0 && a & RIGHT != 0 {
        (screen.w - m_l - m_r).max(0)
    } else if size_w != 0 {
        size_w as i32
    } else {
        screen.w / 4
    };
    let h = if a & TOP != 0 && a & BOTTOM != 0 {
        (screen.h - m_t - m_b).max(0)
    } else if size_h != 0 {
        size_h as i32
    } else {
        screen.h / 4
    };

    // X position: anchor determines which edge drives it; margin
    // offsets inward from that edge. LEFT (alone or paired with
    // RIGHT for full-width anchoring) drives from the left edge.
    let x = if a & LEFT != 0 {
        screen.x + m_l
    } else if a & RIGHT != 0 {
        screen.x + screen.w - w - m_r
    } else {
        screen.x + (screen.w - w) / 2
    };
    let y = if a & TOP != 0 {
        screen.y + m_t
    } else if a & BOTTOM != 0 {
        screen.y + screen.h - h - m_b
    } else {
        screen.y + (screen.h - h) / 2
    };

    crate::Rect { x, y, w, h }
}

/// One entry for a currently-mapped layer surface, with everything
/// callers typically need in one place: the WlSurface (for input
/// routing + cache keys), the WlBuffer (for rendering), the layer
/// (for z-order), the rect (positioned via anchor + margin + size),
/// the keyboard_interactivity setting (for focus policy), anchor +
/// exclusive_zone (for layout reservations so tiled toplevels shrink
/// to avoid the bar).
pub struct MappedLayer {
    pub surface: WlSurface,
    pub buffer: wayland_server::protocol::wl_buffer::WlBuffer,
    pub layer: Layer,
    pub rect: crate::Rect,
    pub keyboard_interactivity: u32,
    pub anchor: u32,
    pub exclusive_zone: i32,
}

/// Enumerate every currently-mapped layer surface. Dead weak refs are
/// skipped silently; caller is expected to prune `pending_layer_surfaces`
/// on a separate pass.
pub fn mapped_layers(state: &State, screen: crate::Rect) -> Vec<MappedLayer> {
    let mut out = Vec::new();
    for weak in state.pending_layer_surfaces.iter() {
        let Ok(surface) = weak.upgrade() else { continue };
        let Some((ls_weak, buffer)) = crate::compositor::with_sol_data(&surface, |sd| {
            if !matches!(sd.role, SurfaceRole::LayerSurface { mapped: true, .. }) {
                return None;
            }
            let buf = sd.current_buffer.clone()?;
            Some((sd.zwlr_layer_surface.clone(), buf))
        })
        .flatten() else {
            continue;
        };
        let Some(ls_weak) = ls_weak else { continue };
        let Ok(ls) = ls_weak.upgrade() else { continue };
        let Some(ls_data) = ls.data::<LayerSurfaceData>() else { continue };
        let (layer, rect, kbi, anchor, exclusive_zone) = {
            let inner = ls_data.inner.lock().unwrap();
            (
                layer_of(inner.current.layer),
                compute_layer_rect(&inner.current, screen),
                inner.current.keyboard_interactivity,
                inner.current.anchor,
                inner.current.exclusive_zone,
            )
        };
        out.push(MappedLayer {
            surface,
            buffer,
            layer,
            rect,
            keyboard_interactivity: kbi,
            anchor,
            exclusive_zone,
        });
    }
    out
}

/// Compute the usable area inside the screen after subtracting exclusive
/// zones reserved by anchored layer surfaces. A bar anchored to the top
/// with exclusive_zone=30 reserves 30 px off the top; the master-stack
/// layout then uses screen minus reservation so tiled toplevels don't
/// overlap the bar. Reservations from overlay/top layers win over
/// bottom/background for the same edge.
pub fn usable_area(mapped: &[MappedLayer], screen: crate::Rect) -> crate::Rect {
    const TOP: u32 = 1;
    const BOTTOM: u32 = 2;
    const LEFT: u32 = 4;
    const RIGHT: u32 = 8;

    let (mut r_t, mut r_b, mut r_l, mut r_r) = (0, 0, 0, 0);
    for ml in mapped {
        if ml.exclusive_zone <= 0 {
            continue;
        }
        // An exclusive zone only applies when the surface is anchored
        // to one edge, or to an edge plus both perpendicular edges —
        // i.e. it's acting as a bar on one side.
        let z = ml.exclusive_zone;
        // Top-anchored bar: top + (left? + right?) but not bottom.
        if ml.anchor & TOP != 0 && ml.anchor & BOTTOM == 0 {
            r_t = r_t.max(z);
        } else if ml.anchor & BOTTOM != 0 && ml.anchor & TOP == 0 {
            r_b = r_b.max(z);
        } else if ml.anchor & LEFT != 0 && ml.anchor & RIGHT == 0 {
            r_l = r_l.max(z);
        } else if ml.anchor & RIGHT != 0 && ml.anchor & LEFT == 0 {
            r_r = r_r.max(z);
        }
    }
    crate::Rect {
        x: screen.x + r_l,
        y: screen.y + r_t,
        w: (screen.w - r_l - r_r).max(0),
        h: (screen.h - r_t - r_b).max(0),
    }
}

/// Double-buffer-apply the pending LayerState into the current one.
/// Called from the wl_surface.commit handler for any surface whose role
/// is LayerSurface, before we decide whether to send configure / map.
pub fn promote_layer_state(surface: &WlSurface) {
    let ls_weak = crate::compositor::with_sol_data(surface, |sd| sd.zwlr_layer_surface.clone()).flatten();
    let Some(ls_weak) = ls_weak else { return };
    let Ok(ls) = ls_weak.upgrade() else { return };
    let Some(data) = ls.data::<LayerSurfaceData>() else { return };
    let mut inner = data.inner.lock().unwrap();
    inner.current = inner.pending.clone();
}

/// Send the initial configure event on a layer surface: computes a size
/// from the client's requested (set_size) dimensions, falling back to
/// the full output extent when the client asks for auto-sizing on an
/// axis by setting it to zero.
pub fn send_initial_configure(state: &mut State, surface: &WlSurface) {
    let (screen_w, screen_h) = (state.screen_width, state.screen_height);
    let ls_weak = crate::compositor::with_sol_data(surface, |sd| sd.zwlr_layer_surface.clone()).flatten();
    let Some(ls_weak) = ls_weak else { return };
    let Ok(ls) = ls_weak.upgrade() else { return };
    let Some(data) = ls.data::<LayerSurfaceData>() else { return };

    let (requested_w, requested_h, anchor) = {
        let inner = data.inner.lock().unwrap();
        (
            inner.current.size.0,
            inner.current.size.1,
            inner.current.anchor,
        )
    };

    // set_size(0, N) with anchors on opposite edges means "fill that
    // axis". Bits 4 | 8 = left+right (0x0c). Bits 1 | 2 = top+bottom (0x03).
    const ANCHOR_LEFT_RIGHT: u32 = 4 | 8;
    const ANCHOR_TOP_BOTTOM: u32 = 1 | 2;

    let w = if requested_w != 0 {
        requested_w
    } else if anchor & ANCHOR_LEFT_RIGHT == ANCHOR_LEFT_RIGHT {
        screen_w
    } else {
        // Client gave neither a width nor horizontal-fill anchors — send
        // back something sensible so it has a chance to proceed. Spec
        // says this is actually a protocol error, but we're lenient.
        screen_w / 4
    };
    let h = if requested_h != 0 {
        requested_h
    } else if anchor & ANCHOR_TOP_BOTTOM == ANCHOR_TOP_BOTTOM {
        screen_h
    } else {
        screen_h / 4
    };

    let serial = state.next_serial();
    ls.configure(serial, w, h);
    {
        let mut inner = data.inner.lock().unwrap();
        inner.last_configure_serial = serial;
    }
    tracing::info!(
        namespace = data.namespace.as_str(),
        layer = data.inner.lock().unwrap().current.layer,
        width = w,
        height = h,
        serial,
        "layer surface initial configure"
    );
}

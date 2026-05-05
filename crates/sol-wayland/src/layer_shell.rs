//! `zwlr_layer_shell_v1` — wire dispatch lives in smithay
//! (`delegate_layer_shell!` in lib.rs). This module provides the
//! sol-side helpers that map smithay's tracked layer surfaces into
//! the scene/layout conventions: resolving anchor + margin + size
//! into a screen-space `Rect`, listing currently-mapped layers, and
//! computing the usable area after subtracting exclusive zones.

use smithay::wayland::compositor::with_states;
use smithay::wayland::shell::wlr_layer::{
    Anchor, ExclusiveZone, KeyboardInteractivity, Layer, LayerSurfaceCachedState,
};
use wayland_server::Resource;
use wayland_server::protocol::wl_surface::WlSurface;

use crate::State;
use crate::compositor::SurfaceRole;

/// One entry for a currently-mapped layer surface, with everything
/// callers typically need in one place: the WlSurface (input routing
/// + cache keys), the WlBuffer (rendering), the layer (z-order), the
/// rect (anchor + margin + size), the keyboard_interactivity (focus
/// policy), anchor + exclusive_zone (layout reservations so tiled
/// toplevels shrink to avoid the bar).
pub struct MappedLayer {
    pub surface: WlSurface,
    pub buffer: wayland_server::protocol::wl_buffer::WlBuffer,
    pub layer: Layer,
    pub rect: crate::Rect,
    pub keyboard_interactivity: KeyboardInteractivity,
    pub anchor: Anchor,
    pub exclusive_zone: ExclusiveZone,
}

/// Compute the screen-space rect for a layer surface given its anchor
/// bitfield, margin, size, and the full screen rect. Mirrors what
/// sway and wlroots do: anchor decides which edge the bar attaches
/// to; margin pushes inward from that edge; size gives the intrinsic
/// dimensions on axes the surface isn't stretched along.
pub fn compute_layer_rect(
    cached: &LayerSurfaceCachedState,
    screen: crate::Rect,
) -> crate::Rect {
    let m = cached.margin;
    let size = cached.size;
    let a = cached.anchor;

    // Width: stretched if anchored both horizontally; else use
    // set_size; else fallback to a quarter-screen so a forgot-to-
    // set-size client still shows up for diagnostics.
    let w = if a.contains(Anchor::LEFT) && a.contains(Anchor::RIGHT) {
        (screen.w - m.left - m.right).max(0)
    } else if size.w != 0 {
        size.w
    } else {
        screen.w / 4
    };
    let h = if a.contains(Anchor::TOP) && a.contains(Anchor::BOTTOM) {
        (screen.h - m.top - m.bottom).max(0)
    } else if size.h != 0 {
        size.h
    } else {
        screen.h / 4
    };

    let x = if a.contains(Anchor::LEFT) {
        screen.x + m.left
    } else if a.contains(Anchor::RIGHT) {
        screen.x + screen.w - w - m.right
    } else {
        screen.x + (screen.w - w) / 2
    };
    let y = if a.contains(Anchor::TOP) {
        screen.y + m.top
    } else if a.contains(Anchor::BOTTOM) {
        screen.y + screen.h - h - m.bottom
    } else {
        screen.y + (screen.h - h) / 2
    };

    crate::Rect { x, y, w, h }
}

/// Enumerate every currently-mapped layer surface. Walks smithay's
/// `WlrLayerShellState::layer_surfaces()`, filters for surfaces our
/// commit handler has marked `mapped: true` (i.e. they have an
/// initial configure ack + a buffer), and resolves each into a
/// `MappedLayer` for the scene + layout passes.
pub fn mapped_layers(state: &State, screen: crate::Rect) -> Vec<MappedLayer> {
    let mut out = Vec::new();
    for ls in state.layer_shell_state.layer_surfaces() {
        let surface = ls.wl_surface().clone();
        if !surface.is_alive() {
            continue;
        }
        let buf_opt = crate::compositor::with_sol_data(&surface, |sd| {
            if !matches!(sd.role, SurfaceRole::LayerSurface { mapped: true, .. }) {
                return None;
            }
            sd.current_buffer.clone()
        })
        .flatten();
        let Some(buffer) = buf_opt else { continue };
        let cached = with_states(&surface, |s| {
            *s.cached_state
                .get::<LayerSurfaceCachedState>()
                .current()
        });
        let rect = compute_layer_rect(&cached, screen);
        out.push(MappedLayer {
            surface,
            buffer,
            layer: cached.layer,
            rect,
            keyboard_interactivity: cached.keyboard_interactivity,
            anchor: cached.anchor,
            exclusive_zone: cached.exclusive_zone,
        });
    }
    out
}

/// Compute the usable area inside the screen after subtracting
/// exclusive zones reserved by anchored layer surfaces. A bar
/// anchored to the top with `Exclusive(30)` reserves 30 px off the
/// top; the master-stack layout uses `screen - reservations` so
/// tiled toplevels don't overlap the bar.
pub fn usable_area(mapped: &[MappedLayer], screen: crate::Rect) -> crate::Rect {
    let (mut r_t, mut r_b, mut r_l, mut r_r) = (0i32, 0i32, 0i32, 0i32);
    for ml in mapped {
        let z = match ml.exclusive_zone {
            ExclusiveZone::Exclusive(z) => z as i32,
            ExclusiveZone::Neutral | ExclusiveZone::DontCare => continue,
        };
        if ml.anchor.contains(Anchor::TOP) && !ml.anchor.contains(Anchor::BOTTOM) {
            r_t = r_t.max(z);
        } else if ml.anchor.contains(Anchor::BOTTOM) && !ml.anchor.contains(Anchor::TOP) {
            r_b = r_b.max(z);
        } else if ml.anchor.contains(Anchor::LEFT) && !ml.anchor.contains(Anchor::RIGHT) {
            r_l = r_l.max(z);
        } else if ml.anchor.contains(Anchor::RIGHT) && !ml.anchor.contains(Anchor::LEFT) {
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

/// Send the initial configure event on a layer surface: computes a
/// size from the client's requested (set_size) dimensions, falling
/// back to the full output extent when the client asks for
/// auto-sizing on an axis (size == 0 and anchored to both opposite
/// edges).
pub fn send_initial_configure(state: &mut State, surface: &WlSurface) {
    let (screen_w, screen_h) =
        (state.screen_width as i32, state.screen_height as i32);
    let Some(ls) = state
        .layer_shell_state
        .layer_surfaces()
        .find(|ls| ls.wl_surface() == surface)
    else {
        return;
    };
    let cached = with_states(surface, |s| {
        *s.cached_state
            .get::<LayerSurfaceCachedState>()
            .current()
    });
    let anchor = cached.anchor;
    let (req_w, req_h) = (cached.size.w, cached.size.h);
    let w = if req_w != 0 {
        req_w
    } else if anchor.contains(Anchor::LEFT) && anchor.contains(Anchor::RIGHT) {
        screen_w
    } else {
        screen_w / 4
    };
    let h = if req_h != 0 {
        req_h
    } else if anchor.contains(Anchor::TOP) && anchor.contains(Anchor::BOTTOM) {
        screen_h
    } else {
        screen_h / 4
    };
    ls.with_pending_state(|state| {
        state.size = Some((w, h).into());
    });
    let _ = ls.send_configure();
    tracing::info!(
        ?w,
        ?h,
        "layer surface initial configure"
    );
}


//! `wp_fractional_scale_manager_v1` — tell clients the preferred
//! scale factor for their surfaces.
//!
//! This protocol exists to bridge a gap in the original Wayland
//! scale model: `wl_surface.set_buffer_scale` only accepts integer
//! scales, but real HiDPI monitors often want 1.25x / 1.5x / etc.
//! The scheme is:
//!
//! 1. Client asks for a per-surface `wp_fractional_scale_v1` via
//!    `wp_fractional_scale_manager_v1.get_fractional_scale(surface)`.
//! 2. Compositor sends `preferred_scale(scale_120)` with the
//!    integer scale multiplied by 120 (so 120 = 1.0×,
//!    150 = 1.25×, etc.).
//! 3. Client renders at `logical_size * scale`, attaches a buffer
//!    at that pixel size, and uses `wp_viewport.set_destination`
//!    to declare the logical size to the compositor.
//!
//! Chrome's Ozone-Wayland pairs fractional-scale with
//! `wp_viewporter` to decide its rendering pipeline. With
//! viewporter advertised but fractional_scale missing, Chrome
//! seems to enter a bad hybrid mode that produces stretched
//! transitional buffers during moves/resizes; advertising both
//! clears that up.
//!
//! voidptr currently runs at integer 1.0× everywhere, so the
//! preferred scale we report is always `120`. Output-specific
//! scales (for HiDPI) would plug in here once we add that.

use wayland_protocols::wp::fractional_scale::v1::server::{
    wp_fractional_scale_manager_v1::{self, WpFractionalScaleManagerV1},
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::State;

pub const FRACTIONAL_SCALE_VERSION: u32 = 1;

/// Fixed-point scale factor the protocol expects: `actual_scale * 120`.
/// 120 corresponds to 1.0×, which is what voidptr drives today.
const SCALE_120: u32 = 120;

impl GlobalDispatch<WpFractionalScaleManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpFractionalScaleManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let m = init.init(resource, ());
        tracing::info!(id = ?m.id(), "bind wp_fractional_scale_manager_v1");
    }
}

impl Dispatch<WpFractionalScaleManagerV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpFractionalScaleManagerV1,
        request: wp_fractional_scale_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_fractional_scale_manager_v1::Request::GetFractionalScale {
                id,
                surface: _,
            } => {
                // Per spec, we must send `preferred_scale` at least
                // once before the client can usefully participate.
                // At integer 1× we send 120 eagerly and keep silent
                // thereafter; if / when we grow multi-monitor or
                // HiDPI support, this is where per-output scale
                // tracking plugs in.
                let fs = init.init(id, ());
                fs.preferred_scale(SCALE_120);
            }
            wp_fractional_scale_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpFractionalScaleV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpFractionalScaleV1,
        request: wp_fractional_scale_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // Only `destroy` is defined client-side. No state to clean up.
        if let wp_fractional_scale_v1::Request::Destroy = request {}
    }
}

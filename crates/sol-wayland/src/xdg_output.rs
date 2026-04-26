//! `zxdg_output_manager_v1` — extended per-output info (logical
//! position, logical size, name, description).
//!
//! GTK/GDK uses this to derive correct cairo surface dimensions and
//! scale factors. Without it GDK falls back to legacy wl_output data
//! and sometimes trips internal invariants during window setup
//! (Firefox's `impl->staging_cairo_surface != cairo_surface`
//! assertion is one symptom). Minimal scope: advertise the global,
//! hand out a zxdg_output_v1 per wl_output binding, populate it
//! with the same dimensions we already send on wl_output.mode.

use wayland_protocols::xdg::xdg_output::zv1::server::{
    zxdg_output_manager_v1::{self, ZxdgOutputManagerV1},
    zxdg_output_v1::{self, ZxdgOutputV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::State;

pub const XDG_OUTPUT_MANAGER_VERSION: u32 = 3;

impl GlobalDispatch<ZxdgOutputManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZxdgOutputManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let m = init.init(resource, ());
        tracing::info!(id = ?m.id(), "bind zxdg_output_manager_v1");
    }
}

impl Dispatch<ZxdgOutputManagerV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZxdgOutputManagerV1,
        request: zxdg_output_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_output_manager_v1::Request::GetXdgOutput { id, output: _ } => {
                let xo = init.init(id, ());
                let (w, h) = (state.screen_width as i32, state.screen_height as i32);
                xo.logical_position(0, 0);
                xo.logical_size(w, h);
                if xo.version() >= 2 {
                    xo.name("SOL-0".into());
                    xo.description("sol primary output".into());
                }
                xo.done();
            }
            zxdg_output_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgOutputV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZxdgOutputV1,
        request: zxdg_output_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        if let zxdg_output_v1::Request::Destroy = request {}
    }
}

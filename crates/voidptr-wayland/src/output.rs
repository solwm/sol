//! wl_output global. Advertises the compositor's active screen size to
//! clients. Dimensions are read live from `State.screen_{width,height}` so
//! headless and DRM backends with different resolutions both work.

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_output::{self, Mode, Subpixel, Transform, WlOutput},
};

use crate::State;

impl GlobalDispatch<WlOutput, ()> for State {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        client: &Client,
        resource: New<WlOutput>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let o = init.init(resource, ());
        let w = state.screen_width as i32;
        let h = state.screen_height as i32;
        o.geometry(
            0,
            0,
            300,
            200,
            Subpixel::Unknown,
            "voidptr".into(),
            "voidptr-0".into(),
            Transform::Normal,
        );
        // Advertise the real refresh rate. Chrome / Firefox / anything
        // that uses the presentation-time protocol or wl_frame pacing
        // reads this to decide their rendering cadence. Before this
        // was dynamic, voidptr shipped a hardcoded 60_000 and Chrome
        // dutifully capped at 60 Hz even on a 240 Hz panel.
        o.mode(
            Mode::Current | Mode::Preferred,
            w,
            h,
            state.screen_refresh_mhz,
        );
        if o.version() >= 2 {
            o.scale(1);
        }
        if o.version() >= 4 {
            o.name("VOIDPTR-0".into());
            o.description("voidptr primary output".into());
        }
        if o.version() >= 2 {
            o.done();
        }
        tracing::info!(id = ?o.id(), width = w, height = h, "bind wl_output");

        // Track live outputs so `ext_workspace_v1` can emit
        // `output_enter` on the workspace group. Prune dead
        // references first to keep the list bounded across
        // reconnects.
        state.outputs.retain(|existing| existing.is_alive());
        state.outputs.push(o.clone());
        crate::ext_workspace::notify_output_bound(state, client, &o);
    }
}

impl Dispatch<WlOutput, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlOutput,
        _request: wl_output::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

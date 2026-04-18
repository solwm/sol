//! wl_output global. At B2 we advertise a single virtual 1920x1080 output
//! matching the headless canvas size.

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_output::{self, Mode, Subpixel, Transform, WlOutput},
};

use crate::State;

pub const OUTPUT_WIDTH: i32 = 1920;
pub const OUTPUT_HEIGHT: i32 = 1080;

impl GlobalDispatch<WlOutput, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlOutput>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let o = init.init(resource, ());
        o.geometry(
            0,
            0,
            300,
            200,
            Subpixel::Unknown,
            "hyperland-rs".into(),
            "headless-0".into(),
            Transform::Normal,
        );
        o.mode(Mode::Current | Mode::Preferred, OUTPUT_WIDTH, OUTPUT_HEIGHT, 60_000);
        if o.version() >= 2 {
            o.scale(1);
        }
        if o.version() >= 4 {
            o.name("HEADLESS-0".into());
            o.description("hyperland-rs headless virtual output".into());
        }
        if o.version() >= 2 {
            o.done();
        }
        tracing::info!(id = ?o.id(), "bind wl_output");
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

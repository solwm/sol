//! wl_seat global with zero capabilities. Real input wiring lands in B5.

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_seat::{self, Capability, WlSeat},
};

use crate::State;

impl GlobalDispatch<WlSeat, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSeat>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let s = init.init(resource, ());
        s.capabilities(Capability::empty());
        if s.version() >= 2 {
            s.name("seat0".into());
        }
        tracing::info!(id = ?s.id(), "bind wl_seat");
    }
}

impl Dispatch<WlSeat, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSeat,
        _request: wl_seat::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

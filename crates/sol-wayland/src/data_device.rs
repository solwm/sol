//! `wl_data_device_manager` — clipboard / drag-and-drop plumbing.
//!
//! GTK's Wayland seat initialization calls `wl_data_device_manager.get_data_device`
//! for each seat it sees, and partially-failing that path can leave the
//! GdkSeat object in an invalid state — which shows up downstream as
//! `GDK_IS_SEAT` assertion failures in Firefox and other GTK apps.
//!
//! Minimal scope: advertise the global, accept `create_data_source` and
//! `get_data_device` requests, keep every resource alive but inert (no
//! selection events, no drop targets, no DND). That's enough for clients
//! to finish init. Real clipboard + drag-and-drop plumbing lands later.
//!
//! One subtle detail: we never send a `selection(null)` event to the
//! client's data_device. The spec lets us; clients treat "no selection
//! event yet" as "clipboard is empty", which is correct for our stub.

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::{
        wl_data_device::{self, WlDataDevice},
        wl_data_device_manager::{self, WlDataDeviceManager},
        wl_data_offer::{self, WlDataOffer},
        wl_data_source::{self, WlDataSource},
    },
};

use crate::State;

pub const DATA_DEVICE_MANAGER_VERSION: u32 = 3;

impl GlobalDispatch<WlDataDeviceManager, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlDataDeviceManager>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let m = init.init(resource, ());
        tracing::info!(id = ?m.id(), "bind wl_data_device_manager");
    }
}

impl Dispatch<WlDataDeviceManager, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataDeviceManager,
        request: wl_data_device_manager::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device_manager::Request::CreateDataSource { id } => {
                let _ = init.init(id, ());
            }
            wl_data_device_manager::Request::GetDataDevice { id, seat: _ } => {
                let _ = init.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataSource, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataSource,
        _request: wl_data_source::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<WlDataDevice, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataDevice,
        _request: wl_data_device::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<WlDataOffer, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataOffer,
        _request: wl_data_offer::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

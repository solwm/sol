//! `zwp_primary_selection_device_manager_v1` — middle-click paste.
//!
//! The primary selection is the X11-style "select-to-copy, middle-click-
//! to-paste" buffer. Browsers (Firefox / Chromium), terminals (alacritty,
//! foot, kitty), and most GTK / Qt apps use it independently of Ctrl+C/V.
//! Without this protocol, middle-click paste silently does nothing on
//! Wayland — the button event is delivered, but no app has selection
//! data to consume.
//!
//! Wire-flow mirrors `wl_data_device` exactly: `set_selection` on a
//! source, broadcast offers to every live device, forward `receive` to
//! the source via `send`. See `data_device.rs` for the full flow.

use std::os::fd::AsFd;
use std::sync::Mutex;

use wayland_protocols::wp::primary_selection::zv1::server::{
    zwp_primary_selection_device_manager_v1::{self, ZwpPrimarySelectionDeviceManagerV1},
    zwp_primary_selection_device_v1::{self, ZwpPrimarySelectionDeviceV1},
    zwp_primary_selection_offer_v1::{self, ZwpPrimarySelectionOfferV1},
    zwp_primary_selection_source_v1::{self, ZwpPrimarySelectionSourceV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::State;

pub const PRIMARY_SELECTION_VERSION: u32 = 1;

#[derive(Default)]
pub struct SourceData {
    pub mimes: Mutex<Vec<String>>,
}

pub struct OfferData {
    pub source: ZwpPrimarySelectionSourceV1,
}

impl GlobalDispatch<ZwpPrimarySelectionDeviceManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpPrimarySelectionDeviceManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let m = init.init(resource, ());
        tracing::debug!(id = ?m.id(), "bind zwp_primary_selection_device_manager_v1");
    }
}

impl Dispatch<ZwpPrimarySelectionDeviceManagerV1, ()> for State {
    fn request(
        state: &mut Self,
        client: &Client,
        _resource: &ZwpPrimarySelectionDeviceManagerV1,
        request: zwp_primary_selection_device_manager_v1::Request,
        _data: &(),
        dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_device_manager_v1::Request::CreateSource { id } => {
                let _ = init.init(id, SourceData::default());
            }
            zwp_primary_selection_device_manager_v1::Request::GetDevice { id, seat: _ } => {
                let device = init.init(id, ());
                if let Some(src) = state.primary_selection_source.clone() {
                    if src.is_alive() {
                        send_selection_to(dh, client, &device, &src);
                    }
                }
                state.primary_devices.push(device);
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwpPrimarySelectionSourceV1, SourceData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwpPrimarySelectionSourceV1,
        request: zwp_primary_selection_source_v1::Request,
        data: &SourceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_source_v1::Request::Offer { mime_type } => {
                if let Ok(mut m) = data.mimes.lock() {
                    m.push(mime_type);
                }
            }
            zwp_primary_selection_source_v1::Request::Destroy => {
                if state
                    .primary_selection_source
                    .as_ref()
                    .map(|s| s.id() == resource.id())
                    .unwrap_or(false)
                {
                    state.primary_selection_source = None;
                    state.primary_devices.retain(|d| d.is_alive());
                    for d in &state.primary_devices {
                        d.selection(None);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwpPrimarySelectionDeviceV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwpPrimarySelectionDeviceV1,
        request: zwp_primary_selection_device_v1::Request,
        _data: &(),
        dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_device_v1::Request::SetSelection { source, serial: _ } => {
                set_selection(state, dh, source);
            }
            zwp_primary_selection_device_v1::Request::Destroy => {
                state.primary_devices.retain(|d| d.id() != resource.id());
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwpPrimarySelectionOfferV1, OfferData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpPrimarySelectionOfferV1,
        request: zwp_primary_selection_offer_v1::Request,
        data: &OfferData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_offer_v1::Request::Receive { mime_type, fd } => {
                if data.source.is_alive() {
                    data.source.send(mime_type, fd.as_fd());
                }
            }
            zwp_primary_selection_offer_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

fn set_selection(
    state: &mut State,
    dh: &DisplayHandle,
    new_source: Option<ZwpPrimarySelectionSourceV1>,
) {
    let same = match (state.primary_selection_source.as_ref(), new_source.as_ref()) {
        (Some(a), Some(b)) => a.id() == b.id(),
        (None, None) => true,
        _ => false,
    };
    if same {
        return;
    }
    if let Some(prev) = state.primary_selection_source.take() {
        if prev.is_alive() {
            prev.cancelled();
        }
    }
    state.primary_selection_source = new_source.clone();
    state.primary_devices.retain(|d| d.is_alive());

    match &new_source {
        Some(src) if src.is_alive() => {
            let devices: Vec<ZwpPrimarySelectionDeviceV1> = state.primary_devices.clone();
            for device in &devices {
                let Some(client) = device.client() else { continue };
                send_selection_to(dh, &client, device, src);
            }
        }
        _ => {
            for device in &state.primary_devices {
                device.selection(None);
            }
        }
    }
}

fn send_selection_to(
    dh: &DisplayHandle,
    client: &Client,
    device: &ZwpPrimarySelectionDeviceV1,
    source: &ZwpPrimarySelectionSourceV1,
) {
    let offer = match client
        .create_resource::<ZwpPrimarySelectionOfferV1, OfferData, State>(
            dh,
            device.version(),
            OfferData {
                source: source.clone(),
            },
        ) {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!(error = ?e, "primary_selection: create_resource(offer) failed");
            return;
        }
    };
    device.data_offer(&offer);
    if let Some(src_data) = source.data::<SourceData>() {
        if let Ok(mimes) = src_data.mimes.lock() {
            for m in mimes.iter() {
                offer.offer(m.clone());
            }
        }
    }
    device.selection(Some(&offer));
}

//! `wl_data_device_manager` — system clipboard (selection) plumbing.
//!
//! Implements enough of the protocol for Ctrl+C / Ctrl+V across clients.
//! DnD is intentionally inert; a click in `start_drag` does nothing.
//!
//! Wire-flow for a copy:
//!   1. Source client creates a `wl_data_source`, calls `offer(mime)` for
//!      every type it can produce, then `data_device.set_selection(source,
//!      serial)`.
//!   2. We stash the source on `State.selection_source`, send `cancelled`
//!      to whatever was previously selected, then for every live
//!      `wl_data_device` we (a) create a fresh `wl_data_offer`, (b) emit
//!      `data_device.data_offer(offer)`, (c) one `offer.offer(mime)` per
//!      advertised mime, (d) `data_device.selection(Some(offer))`.
//!
//! Wire-flow for a paste:
//!   1. Destination client picks a mime and calls `wl_data_offer.receive(
//!      mime, fd)` where `fd` is the write end of a pipe it owns.
//!   2. We forward to the backing source via `wl_data_source.send(mime, fd)`
//!      so the source-owning client writes the bytes; the destination
//!      reads from its pipe and closes when done.
//!
//! The current selection's source can survive across many pastes — it is
//! only `cancelled` when the source-owning client replaces the selection
//! with a new one or destroys its source.

use std::os::fd::AsFd;
use std::sync::Mutex;

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

/// Per-`wl_data_source` state: the mime types the source has advertised
/// via `offer`. Wrapped in a Mutex so `Dispatch::request` (which holds
/// `&Self`) can mutate it.
#[derive(Default)]
pub struct SourceData {
    pub mimes: Mutex<Vec<String>>,
}

/// Per-`wl_data_offer` state: the source this offer represents. When the
/// destination calls `receive`, we forward to this source's `send`.
pub struct OfferData {
    pub source: WlDataSource,
}

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
        tracing::debug!(id = ?m.id(), "bind wl_data_device_manager");
    }
}

impl Dispatch<WlDataDeviceManager, ()> for State {
    fn request(
        state: &mut Self,
        client: &Client,
        _resource: &WlDataDeviceManager,
        request: wl_data_device_manager::Request,
        _data: &(),
        dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device_manager::Request::CreateDataSource { id } => {
                let _ = init.init(id, SourceData::default());
            }
            wl_data_device_manager::Request::GetDataDevice { id, seat: _ } => {
                let device = init.init(id, ());
                // Send the current selection (if any) so the new device
                // sees the same clipboard everyone else does. Without
                // this, a freshly-launched paste-target sees an empty
                // clipboard until the next copy.
                if let Some(src) = state.selection_source.clone() {
                    if src.is_alive() {
                        send_selection_to(dh, client, &device, &src);
                    }
                }
                state.data_devices.push(device);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataSource, SourceData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WlDataSource,
        request: wl_data_source::Request,
        data: &SourceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_source::Request::Offer { mime_type } => {
                if let Ok(mut m) = data.mimes.lock() {
                    m.push(mime_type);
                }
            }
            wl_data_source::Request::SetActions { .. } => {
                // DnD only; selection sources never call this.
            }
            wl_data_source::Request::Destroy => {
                // If this source was the active selection, clear it
                // and notify every live device that the clipboard is
                // now empty. wayland-server tears the resource down
                // for us via the destructor annotation.
                if state
                    .selection_source
                    .as_ref()
                    .map(|s| s.id() == resource.id())
                    .unwrap_or(false)
                {
                    state.selection_source = None;
                    state.data_devices.retain(|d| d.is_alive());
                    for d in &state.data_devices {
                        d.selection(None);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataDevice, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WlDataDevice,
        request: wl_data_device::Request,
        _data: &(),
        dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device::Request::SetSelection { source, serial: _ } => {
                set_selection(state, dh, source);
            }
            wl_data_device::Request::StartDrag { .. } => {
                // DnD is out of scope; ignoring is harmless — clients
                // that try a drag will see no offers reach a target,
                // which matches "drag cancelled".
            }
            wl_data_device::Request::Release => {
                state.data_devices.retain(|d| d.id() != resource.id());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataOffer, OfferData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataOffer,
        request: wl_data_offer::Request,
        data: &OfferData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_offer::Request::Receive { mime_type, fd } => {
                if data.source.is_alive() {
                    // Hand the destination's pipe write-fd to the
                    // source so it can stream the selection contents
                    // directly into the destination's pipe.
                    data.source.send(mime_type, fd.as_fd());
                }
            }
            wl_data_offer::Request::Accept { .. }
            | wl_data_offer::Request::SetActions { .. }
            | wl_data_offer::Request::Finish => {
                // DnD-only; safe to ignore for selection offers.
            }
            wl_data_offer::Request::Destroy => {}
            _ => {}
        }
    }
}

/// Apply a `set_selection` request: replace the current selection
/// source, cancel the previous one, and broadcast new offers to every
/// live data_device.
fn set_selection(
    state: &mut State,
    dh: &DisplayHandle,
    new_source: Option<WlDataSource>,
) {
    // Skip if same source (idempotent re-set).
    let same = match (state.selection_source.as_ref(), new_source.as_ref()) {
        (Some(a), Some(b)) => a.id() == b.id(),
        (None, None) => true,
        _ => false,
    };
    if same {
        return;
    }

    // Cancel the previous source so its owning client can drop the
    // backing buffer. Must happen before we replace the slot — the
    // `cancelled` event is delivered to the OLD source, not the new.
    if let Some(prev) = state.selection_source.take() {
        if prev.is_alive() {
            prev.cancelled();
        }
    }

    state.selection_source = new_source.clone();
    state.data_devices.retain(|d| d.is_alive());

    match &new_source {
        Some(src) if src.is_alive() => {
            // Snapshot devices so we can iterate without holding a
            // borrow on `state` (the broadcast loop doesn't touch
            // `state` again, but the borrow checker doesn't know).
            let devices: Vec<WlDataDevice> = state.data_devices.clone();
            for device in &devices {
                let Some(client) = device.client() else { continue };
                send_selection_to(dh, &client, device, src);
            }
        }
        _ => {
            for device in &state.data_devices {
                device.selection(None);
            }
        }
    }
}

/// Build a fresh `wl_data_offer` on `client`, advertise the source's
/// mimes, and announce it to `device` as the current selection.
fn send_selection_to(
    dh: &DisplayHandle,
    client: &Client,
    device: &WlDataDevice,
    source: &WlDataSource,
) {
    let offer = match client.create_resource::<WlDataOffer, OfferData, State>(
        dh,
        device.version(),
        OfferData {
            source: source.clone(),
        },
    ) {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!(error = ?e, "data_device: create_resource(wl_data_offer) failed");
            return;
        }
    };

    // Order matters: clients first see `data_offer(new_id)`, then
    // `offer(mime)` calls populate it, then `selection(offer)` swaps
    // it in as the active clipboard.
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

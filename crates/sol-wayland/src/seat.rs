//! wl_seat + wl_pointer + wl_keyboard.
//!
//! Capabilities are driven by whether the compositor has libinput wired in.
//! With input enabled we advertise Pointer | Keyboard; without it (headless)
//! we advertise none, which clients handle by simply not calling
//! get_pointer / get_keyboard.

use std::os::fd::AsFd;

use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::{
        wl_keyboard::{self, WlKeyboard},
        wl_pointer::{self, WlPointer},
        wl_seat::{self, Capability, WlSeat},
    },
};

use crate::State;

impl GlobalDispatch<WlSeat, ()> for State {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSeat>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let s = init.init(resource, ());
        let caps = if state.input.is_some() {
            Capability::Pointer | Capability::Keyboard
        } else {
            Capability::empty()
        };
        s.capabilities(caps);
        if s.version() >= 2 {
            s.name("seat0".into());
        }
        tracing::info!(id = ?s.id(), ?caps, "bind wl_seat");
    }
}

impl Dispatch<WlSeat, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WlSeat,
        request: wl_seat::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_seat::Request::GetPointer { id } => {
                let pointer = init.init(id, ());
                tracing::debug!(id = ?pointer.id(), "create wl_pointer");
                state.pointers.push(pointer);
            }
            wl_seat::Request::GetKeyboard { id } => {
                let keyboard = init.init(id, ());
                tracing::debug!(id = ?keyboard.id(), "create wl_keyboard");
                // Send the keymap to this freshly bound keyboard so the
                // client can make sense of future key events.
                if let Some(km) = state.keymap.as_ref() {
                    keyboard.keymap(
                        wl_keyboard::KeymapFormat::XkbV1,
                        km.fd.as_fd(),
                        km.size,
                    );
                    if keyboard.version() >= 4 {
                        keyboard.repeat_info(
                            state.config.keyboard_repeat_rate,
                            state.config.keyboard_repeat_delay,
                        );
                    }
                }
                // If this client owns the currently focused surface, send
                // an enter right away so it can start receiving key events.
                let focus_clone = state
                    .keyboard_focus
                    .as_ref()
                    .filter(|f| {
                        f.client().map(|c| c.id())
                            == keyboard.client().map(|c| c.id())
                    })
                    .cloned();
                if let Some(focus) = focus_clone {
                    let serial = state.next_serial();
                    // Pass the actually-held keycodes so the client's
                    // xkb stays in sync with the modifiers event we
                    // send right after — see `pressed_keys_bytes`.
                    let keys = state.pressed_keys_bytes();
                    keyboard.enter(serial, &focus, keys);
                    if let Some(m) = state.keymap.as_ref().map(|km| {
                        use xkbcommon::xkb as x;
                        crate::xkb::ModifiersSnapshot {
                            depressed: km.state.serialize_mods(x::STATE_MODS_DEPRESSED),
                            latched: km.state.serialize_mods(x::STATE_MODS_LATCHED),
                            locked: km.state.serialize_mods(x::STATE_MODS_LOCKED),
                            group: km.state.serialize_layout(x::STATE_LAYOUT_EFFECTIVE),
                        }
                    }) {
                        let serial = state.next_serial();
                        keyboard.modifiers(serial, m.depressed, m.latched, m.locked, m.group);
                    }
                }
                state.keyboards.push(keyboard);
            }
            wl_seat::Request::Release => {}
            _ => {}
        }
    }
}

impl Dispatch<WlPointer, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WlPointer,
        request: wl_pointer::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_pointer::Request::SetCursor {
                serial: _,
                surface,
                hotspot_x,
                hotspot_y,
            } => {
                // Client is asking for a custom cursor while the
                // pointer is over its surface (Chrome's hand-pointer
                // over a link, a text I-beam in an editor, etc.) or,
                // when `surface` is None, asking for the cursor to be
                // hidden entirely (fullscreen video players after a
                // few idle seconds). Per spec the choice expires the
                // next time the pointer leaves the surface — see
                // `update_pointer_focus_and_motion`, which clears
                // `client_override_active` on focus change.
                state.cursor.client_override_active = true;
                state.cursor.client_surface =
                    surface.as_ref().map(|s| s.downgrade());
                state.cursor.client_hot_x = hotspot_x;
                state.cursor.client_hot_y = hotspot_y;
                state.needs_render = true;
            }
            wl_pointer::Request::Release => {}
            _ => {}
        }
    }
}

impl Dispatch<WlKeyboard, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlKeyboard,
        _request: wl_keyboard::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // Only has Release, which we don't need to act on.
    }
}

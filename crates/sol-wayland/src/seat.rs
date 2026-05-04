//! `wl_seat` + `wl_pointer` + `wl_keyboard` — wire dispatch lives in
//! smithay (`delegate_seat!`); event delivery into clients goes
//! through `KeyboardHandle::input_intercept` / `input_forward` and
//! `PointerHandle::motion` / `button` / `axis` from `apply_input`.
//!
//! `wl_pointer.set_cursor` is routed via
//! `SeatHandler::cursor_image` (in lib.rs); seat capabilities are
//! managed by smithay automatically based on whether
//! `seat.add_keyboard` / `seat.add_pointer` were called.

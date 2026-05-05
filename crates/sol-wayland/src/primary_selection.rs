//! `zwp_primary_selection_device_manager_v1` — middle-click paste.
//!
//! Wire dispatch lives in smithay (`delegate_primary_selection!`);
//! the `PrimarySelectionHandler` impl is in lib.rs. Keyboard-focus
//! routing for primary selection runs through
//! `SeatHandler::focus_changed` → `set_primary_focus`, mirroring
//! the data_device side.

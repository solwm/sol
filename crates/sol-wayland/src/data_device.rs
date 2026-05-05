//! `wl_data_device_manager` — clipboard + drag-and-drop.
//!
//! Wire dispatch lives in smithay (`delegate_data_device!`); the
//! handler impls (`SelectionHandler`, `DataDeviceHandler`,
//! `ClientDndGrabHandler`, `ServerDndGrabHandler`) are in lib.rs
//! alongside the seat wiring they depend on. We keep selection
//! routing aligned with keyboard focus via
//! `SeatHandler::focus_changed` → `set_data_device_focus`, which
//! is what tells smithay which client receives the active selection.
//!
//! Sol doesn't drive server-side selection or DnD; the
//! ClientDndGrabHandler / ServerDndGrabHandler impls use empty
//! defaults, and `SelectionHandler::send_selection` is unreachable
//! in practice (only fires for compositor-set selections, which we
//! never set via `set_data_device_selection`).

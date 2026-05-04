//! `zwp_idle_inhibit_manager_v1` — clients veto idle blanking.
//!
//! Smithay owns the wire dispatch via [`IdleInhibitManagerState`] +
//! `delegate_idle_inhibit!` (in lib.rs). We implement
//! [`IdleInhibitHandler::inhibit`] / `uninhibit` and track the
//! inhibiting surfaces ourselves on `State::idle_inhibitors`.
//!
//! Per spec, the inhibitor only matters while the surface is "visually
//! relevant" (mapped, on screen). We approximate that with "surface is
//! alive" — good enough in practice: video players destroy the
//! inhibitor on pause/tab-hide, so a stale inhibitor is unusual. A
//! disconnected client that didn't explicitly destroy its inhibitor
//! shows up here as a dead `Weak<WlSurface>` and is filtered out by
//! [`any_active`].

use wayland_server::{Resource, protocol::wl_surface::WlSurface};

use crate::State;

/// Called from `IdleInhibitHandler::inhibit`. Records the surface so
/// `any_active` can consult it later. If the system was already idle
/// when the client created its inhibitor, raise `pending_wake` so the
/// main loop drives DPMS back on after this dispatch returns.
pub fn on_inhibit(state: &mut State, surface: WlSurface) {
    state.idle_inhibitors.push(surface.downgrade());
    if state.idle {
        state.pending_wake = true;
    }
    tracing::debug!(
        count = state.idle_inhibitors.len(),
        "idle_inhibit: inhibitor created"
    );
}

/// Called from `IdleInhibitHandler::uninhibit`. Drops every entry that
/// matches the surface. A client that creates two inhibitors on the
/// same surface and then destroys one will see both removed here —
/// matches our hand-rolled behaviour where dead `Weak`s would also
/// be filtered out wholesale on the next `any_active` poll.
pub fn on_uninhibit(state: &mut State, surface: WlSurface) {
    state
        .idle_inhibitors
        .retain(|w| w.upgrade().ok().as_ref() != Some(&surface));
    tracing::debug!(
        count = state.idle_inhibitors.len(),
        "idle_inhibit: inhibitor destroyed"
    );
}

/// True if any live inhibitor currently exists. Called from the idle
/// timer before blanking — if this returns true, we skip the blank
/// and rearm. Prunes dead `Weak<WlSurface>`s in-place so the list
/// doesn't grow unbounded across client churn (notably clients that
/// crash before destroying their inhibitor).
pub fn any_active(state: &mut State) -> bool {
    state.idle_inhibitors.retain(|w| w.upgrade().is_ok());
    !state.idle_inhibitors.is_empty()
}

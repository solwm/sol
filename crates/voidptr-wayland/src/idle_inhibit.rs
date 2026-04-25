//! `zwp_idle_inhibit_manager_v1` — clients veto idle blanking.
//!
//! A surface (Chrome/Firefox while a video plays, mpv, vlc, presentation
//! software) creates an inhibitor; while any inhibitor's surface is live,
//! the compositor must not mark the user idle. Without this, a 5-minute
//! idle timer would turn the monitor off mid-YouTube.
//!
//! Per spec, the inhibitor only matters while the surface is "visually
//! relevant" (mapped, on screen). We approximate that with "surface is
//! alive" — good enough in practice: video players destroy the inhibitor
//! on pause/tab-hide, so a stale inhibitor is unusual. If a window on an
//! inactive workspace keeps an inhibitor live, we still respect it; the
//! alternative (idle-blank while the user has a video paused in another
//! workspace) surprises people more often than it helps.

use wayland_protocols::wp::idle_inhibit::zv1::server::{
    zwp_idle_inhibit_manager_v1::{self, ZwpIdleInhibitManagerV1},
    zwp_idle_inhibitor_v1::{self, ZwpIdleInhibitorV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, Weak,
    protocol::wl_surface::WlSurface,
};

pub const IDLE_INHIBIT_MANAGER_VERSION: u32 = 1;

/// UserData on each `zwp_idle_inhibitor_v1`, letting us reverse-look
/// up the surface this inhibitor guards when we evaluate "is anyone
/// inhibiting idle right now?" in the timer callback.
pub struct InhibitorData {
    pub surface: Weak<WlSurface>,
}

impl GlobalDispatch<ZwpIdleInhibitManagerV1, ()> for crate::State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpIdleInhibitManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let m = init.init(resource, ());
        tracing::debug!(id = ?m.id(), "bind zwp_idle_inhibit_manager_v1");
    }
}

impl Dispatch<ZwpIdleInhibitManagerV1, ()> for crate::State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZwpIdleInhibitManagerV1,
        request: zwp_idle_inhibit_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_idle_inhibit_manager_v1::Request::CreateInhibitor { id, surface } => {
                let inhibitor = init.init(
                    id,
                    InhibitorData {
                        surface: surface.downgrade(),
                    },
                );
                // Creating an inhibitor while idle should un-idle. The
                // spec says "the inhibitor isn't honored if the system
                // was already idled at the time", but taking that
                // literally means waking only on user input — which is
                // fine for screensavers, bad for a monitor powered off:
                // the client wants their video visible *now*. We don't
                // have backend access from here (Dispatch is State-only),
                // so raise the pending_wake flag; the display source in
                // the main loop picks it up after this dispatch returns
                // and drives DPMS back on.
                if state.idle {
                    state.pending_wake = true;
                }
                state.idle_inhibitors.push(inhibitor);
                tracing::debug!(
                    count = state.idle_inhibitors.len(),
                    "idle_inhibit: inhibitor created"
                );
            }
            zwp_idle_inhibit_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpIdleInhibitorV1, InhibitorData> for crate::State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwpIdleInhibitorV1,
        request: zwp_idle_inhibitor_v1::Request,
        _data: &InhibitorData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // Destroy is a type="destructor" request; wayland-server runs
        // the actual teardown for us. We just have to drop our own
        // tracking entry so the count stays accurate.
        if let zwp_idle_inhibitor_v1::Request::Destroy = request {
            let id = resource.id();
            state.idle_inhibitors.retain(|i| i.id() != id);
            tracing::debug!(
                count = state.idle_inhibitors.len(),
                "idle_inhibit: inhibitor destroyed"
            );
        }
    }
}

/// True if any live inhibitor currently exists. Called from the idle
/// timer before blanking — if this returns true, we skip the blank
/// and rearm. Prunes dead entries in-place so the list doesn't grow
/// unbounded across client churn.
pub fn any_active(state: &mut crate::State) -> bool {
    state.idle_inhibitors.retain(|i| i.is_alive());
    state.idle_inhibitors.iter().any(|i| {
        i.data::<InhibitorData>()
            .and_then(|d| d.surface.upgrade().ok())
            .is_some()
    })
}


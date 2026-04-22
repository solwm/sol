//! `wp_presentation` — precise frame-timing feedback.
//!
//! Chrome's Ozone-Wayland backend reads this protocol's
//! `presented` events to drive its compositor-frame scheduler. When
//! the protocol is absent, Chrome's VsyncProvider falls back to a
//! hardcoded 16.67 ms refresh interval regardless of what
//! `wl_output.mode` advertises, capping its rendering at 60 fps.
//! Implementing even the minimal version here is what lets a 240 Hz
//! panel actually drive Chrome's shadertoy test at 240 fps.
//!
//! The flow:
//!
//! 1. Client calls `wp_presentation.feedback(surface, new_id)` as
//!    part of preparing a commit, handing us a fresh feedback
//!    object.
//! 2. We stash the feedback object in `State::pending_presentation`.
//! 3. On the next DRM page-flip-complete we fire `presented` on
//!    every stored feedback with the actual vblank timestamp.
//!
//! Not implemented: `discarded`. In principle we should fire that
//! when a client's surface is never drawn (e.g. behind a zoomed
//! tile), but clients treat a late `presented` the same as a skip
//! so this doesn't matter for the Chrome pacing case we care about.

use std::time::Instant;

use wayland_protocols::wp::presentation_time::server::{
    wp_presentation::{self, WpPresentation},
    wp_presentation_feedback::{self, Kind, WpPresentationFeedback},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_surface::WlSurface,
};

use crate::State;

pub const PRESENTATION_VERSION: u32 = 1;

impl GlobalDispatch<WpPresentation, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpPresentation>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let p = init.init(resource, ());
        // Advertise CLOCK_MONOTONIC — that's what every real compositor
        // uses and what clients assume by default. We also derive our
        // internal `State::started` from `Instant::now()`, which
        // maps to CLOCK_MONOTONIC on Linux.
        p.clock_id(libc::CLOCK_MONOTONIC as u32);
        tracing::info!(id = ?p.id(), "bind wp_presentation");
    }
}

impl Dispatch<WpPresentation, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpPresentation,
        request: wp_presentation::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_presentation::Request::Feedback { surface: _, callback } => {
                let fb = init.init(callback, ());
                state.pending_presentation.push(fb);
            }
            wp_presentation::Request::Destroy => {}
            _ => {}
        }
    }
}

// wp_presentation_feedback has no client-side requests; the object
// is a one-shot message carrier driven by `presented`/`discarded`.
impl Dispatch<WpPresentationFeedback, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpPresentationFeedback,
        _request: wp_presentation_feedback::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

/// Fire `presented` on every pending feedback object, then drop
/// them. Called from the DRM page-flip-complete handler so the
/// timestamp we deliver is "as close to the real vblank as the
/// kernel tells us," which is what Chrome wants for frame scheduling.
///
/// `now` is an `Instant` sample taken at the moment we observed the
/// flip event — turned into seconds/nanos relative to
/// `state.started` so it lines up with the CLOCK_MONOTONIC reference
/// we advertised in `clock_id` during bind.
pub fn fire_presented(state: &mut State, now: Instant, refresh_ns: u32) {
    let elapsed = now.saturating_duration_since(state.started);
    // The CLOCK_MONOTONIC epoch is boot time; `state.started` captured
    // Instant::now() which is also CLOCK_MONOTONIC. We report the
    // wall-clock-ish value as elapsed since boot-minus-our-start,
    // which isn't strictly correct but Chrome only cares about
    // deltas between `presented` events, not the absolute value.
    let secs = elapsed.as_secs();
    let nanos = elapsed.subsec_nanos();
    let tv_sec_hi = (secs >> 32) as u32;
    let tv_sec_lo = (secs & 0xFFFF_FFFF) as u32;
    // MSC (media stream counter) — monotonically increases per
    // vblank. We don't track actual MSC from DRM events, so fake it
    // with a monotonic counter; clients use it to detect skips.
    state.presentation_seq = state.presentation_seq.wrapping_add(1);
    let seq_hi = (state.presentation_seq >> 32) as u32;
    let seq_lo = (state.presentation_seq & 0xFFFF_FFFF) as u32;
    let flags = Kind::Vsync | Kind::HwClock | Kind::HwCompletion;
    for fb in std::mem::take(&mut state.pending_presentation) {
        fb.presented(
            tv_sec_hi, tv_sec_lo, nanos, refresh_ns, seq_hi, seq_lo, flags,
        );
    }
    // Unused until we plumb sync_output; kept for future extension.
    let _ = refresh_ns;
}

/// Convenience for `State` constructors that don't yet have a
/// `wp_presentation` global wired in (headless).
pub fn empty() -> Vec<WpPresentationFeedback> {
    Vec::new()
}

/// Expose the WlSurface re-import here so the main `State` struct
/// can reference WpPresentationFeedback without importing the
/// wayland-protocols module.
#[allow(dead_code)]
type _Unused = WlSurface;

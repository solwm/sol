//! libinput plumbing.
//!
//! Opens input devices via udev, exposes the libinput context's epoll fd for
//! calloop integration (duplicated so libinput keeps its own), and drains
//! events into a simple `InputEvent` enum the render loop can consume.
//!
//! Device opens are routed through libseat via a shared `Session`, so
//! sol runs as an ordinary user (no `input` group, no sudo). libseat
//! also revokes these fds when our VT isn't active, which is what fixes
//! the cross-VT input leak we saw pre-B11.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use input::event::Event as LiEvent;
use input::event::keyboard::{KeyState, KeyboardEvent, KeyboardEventTrait};
use input::event::pointer::{Axis, ButtonState, PointerEvent, PointerScrollEvent};
use input::{Libinput, LibinputInterface};
// Smithay's Session::open takes its (pinned rustix 1.x) OFlags, not
// the workspace rustix 0.38.x — they're distinct types at the Rust
// level. Use smithay's re-export here so the trait method matches.
use smithay::reexports::rustix::fs::OFlags;

use crate::session::{LibSeatSession, Session, SharedSession};

/// Translated from libinput into something the render loop cares about.
#[derive(Debug)]
pub enum InputEvent {
    PointerMotion { dx: f64, dy: f64 },
    PointerMotionAbsolute { x_mm: f64, y_mm: f64 },
    PointerButton { button: u32, pressed: bool },
    /// Scroll from a mouse wheel, trackpad, or other continuous source.
    /// Each axis is `Some(value)` only if the libinput event carried data
    /// for it. `v120_*` is populated only for `Wheel` — it's the high-res
    /// discrete step count, 120 units per logical click, per the Windows
    /// Vista wheel convention libinput borrowed. Finger sources emit a
    /// terminating event with value 0, which we forward as `axis_stop`
    /// on the Wayland side.
    PointerAxis {
        source: AxisSource,
        v_value: Option<f64>,
        h_value: Option<f64>,
        v120_v: Option<f64>,
        v120_h: Option<f64>,
    },
    Key { keycode: u32, pressed: bool },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AxisSource {
    Wheel,
    Finger,
    Continuous,
}

/// libinput's open_restricted / close_restricted glue. Every call
/// goes through smithay's [`LibSeatSession`] which talks to the seat
/// daemon (logind / seatd). The session keeps its own `RawFd ->
/// libseat::Device` map so we don't need one of our own — `open`
/// returns the `OwnedFd` libinput wants directly, and `close` finds
/// the matching libseat handle by fd and revokes it cleanly.
struct Interface {
    session: LibSeatSession,
}

impl LibinputInterface for Interface {
    fn open_restricted(&mut self, path: &Path, flags: i32) -> Result<OwnedFd, i32> {
        // Smithay's libseat impl ignores the flags arg, but we pass
        // through what libinput requested for protocol cleanliness in
        // case a future smithay version starts honouring it.
        let oflags = OFlags::from_bits_truncate(flags as u32);
        self.session.open(path, oflags).map_err(|e| {
            tracing::warn!(?e, path = %path.display(), "libseat open failed");
            libc::EACCES
        })
    }

    fn close_restricted(&mut self, fd: OwnedFd) {
        if let Err(e) = self.session.close(fd) {
            tracing::warn!(?e, "libseat close");
        }
    }
}

/// Owns the libinput context. The watchable event fd is returned separately
/// from `init` so the caller can hand it to calloop while `Libinput` moves
/// into State.
pub struct InputState {
    pub li: Libinput,
}

impl InputState {
    pub fn init(seat_name: &str, session: SharedSession) -> Result<(Self, OwnedFd)> {
        let iface = Interface { session };
        let mut li = Libinput::new_with_udev(iface);
        li.udev_assign_seat(seat_name).map_err(|_| {
            anyhow!(
                "libinput udev_assign_seat({seat_name:?}) failed. \
                 The libseat session probably isn't owned by us yet — make \
                 sure sol was launched from an active user session."
            )
        })?;
        let raw: RawFd = li.as_raw_fd();
        let dup = unsafe { libc::dup(raw) };
        if dup < 0 {
            return Err(anyhow::Error::from(std::io::Error::last_os_error()))
                .context("dup libinput fd");
        }
        let event_fd = unsafe { OwnedFd::from_raw_fd(dup) };
        tracing::info!(seat = %seat_name, "libinput attached (via libseat)");
        Ok((Self { li }, event_fd))
    }

    /// Pull all available events from libinput, translating into InputEvent.
    pub fn drain(&mut self) -> Vec<InputEvent> {
        let mut out = Vec::new();
        if let Err(e) = self.li.dispatch() {
            tracing::warn!(error = ?e, "libinput dispatch");
            return out;
        }
        for ev in self.li.by_ref() {
            match ev {
                LiEvent::Pointer(p) => match p {
                    PointerEvent::Motion(m) => out.push(InputEvent::PointerMotion {
                        dx: m.dx(),
                        dy: m.dy(),
                    }),
                    PointerEvent::MotionAbsolute(a) => {
                        out.push(InputEvent::PointerMotionAbsolute {
                            x_mm: a.absolute_x(),
                            y_mm: a.absolute_y(),
                        });
                    }
                    PointerEvent::Button(b) => out.push(InputEvent::PointerButton {
                        button: b.button(),
                        pressed: matches!(b.button_state(), ButtonState::Pressed),
                    }),
                    PointerEvent::ScrollWheel(s) => {
                        let v = s.has_axis(Axis::Vertical).then(|| s.scroll_value(Axis::Vertical));
                        let h = s.has_axis(Axis::Horizontal).then(|| s.scroll_value(Axis::Horizontal));
                        let v120_v = s.has_axis(Axis::Vertical).then(|| s.scroll_value_v120(Axis::Vertical));
                        let v120_h = s.has_axis(Axis::Horizontal).then(|| s.scroll_value_v120(Axis::Horizontal));
                        out.push(InputEvent::PointerAxis {
                            source: AxisSource::Wheel,
                            v_value: v,
                            h_value: h,
                            v120_v,
                            v120_h,
                        });
                    }
                    PointerEvent::ScrollFinger(s) => {
                        let v = s.has_axis(Axis::Vertical).then(|| s.scroll_value(Axis::Vertical));
                        let h = s.has_axis(Axis::Horizontal).then(|| s.scroll_value(Axis::Horizontal));
                        out.push(InputEvent::PointerAxis {
                            source: AxisSource::Finger,
                            v_value: v,
                            h_value: h,
                            v120_v: None,
                            v120_h: None,
                        });
                    }
                    PointerEvent::ScrollContinuous(s) => {
                        let v = s.has_axis(Axis::Vertical).then(|| s.scroll_value(Axis::Vertical));
                        let h = s.has_axis(Axis::Horizontal).then(|| s.scroll_value(Axis::Horizontal));
                        out.push(InputEvent::PointerAxis {
                            source: AxisSource::Continuous,
                            v_value: v,
                            h_value: h,
                            v120_v: None,
                            v120_h: None,
                        });
                    }
                    _ => {}
                },
                LiEvent::Keyboard(k) => {
                    if let KeyboardEvent::Key(key) = k {
                        out.push(InputEvent::Key {
                            keycode: key.key(),
                            pressed: matches!(key.key_state(), KeyState::Pressed),
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }
}

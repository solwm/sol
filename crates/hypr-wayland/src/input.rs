//! libinput plumbing.
//!
//! Opens input devices via udev, exposes the libinput context's epoll fd for
//! calloop integration (duplicated so libinput keeps its own), and drains
//! events into a simple `InputEvent` enum the render loop can consume.
//!
//! Accepts whatever perms the kernel gives us on /dev/input/event*. On a
//! default Arch install that means either the user is in the `input` group
//! or the compositor runs with sudo. Logind/seatd handoff is a later block.

use std::fs::OpenOptions;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use input::event::Event as LiEvent;
use input::event::keyboard::{KeyState, KeyboardEvent, KeyboardEventTrait};
use input::event::pointer::{ButtonState, PointerEvent};
use input::{Libinput, LibinputInterface};

/// Translated from libinput into something the render loop cares about.
#[derive(Debug)]
pub enum InputEvent {
    PointerMotion { dx: f64, dy: f64 },
    PointerMotionAbsolute { x_mm: f64, y_mm: f64 },
    PointerButton { button: u32, pressed: bool },
    Key { keycode: u32, pressed: bool },
}

struct Interface;

impl LibinputInterface for Interface {
    fn open_restricted(&mut self, path: &Path, flags: i32) -> Result<OwnedFd, i32> {
        let access = flags & libc::O_ACCMODE;
        let read = access == libc::O_RDONLY || access == libc::O_RDWR;
        let write = access == libc::O_WRONLY || access == libc::O_RDWR;
        OpenOptions::new()
            .custom_flags(flags)
            .read(read)
            .write(write)
            .open(path)
            .map(|f| f.into())
            .map_err(|e| e.raw_os_error().unwrap_or(libc::EIO))
    }
    fn close_restricted(&mut self, fd: OwnedFd) {
        drop(fd);
    }
}

/// Owns the libinput context. The watchable event fd is returned separately
/// from `init` so the caller can hand it to calloop while `Libinput` moves
/// into State.
pub struct InputState {
    pub li: Libinput,
}

impl InputState {
    pub fn init(seat: &str) -> Result<(Self, OwnedFd)> {
        let mut li = Libinput::new_with_udev(Interface);
        li.udev_assign_seat(seat).map_err(|_| {
            anyhow!(
                "libinput udev_assign_seat({seat:?}) failed.\n\
                 Almost certainly a permission problem on /dev/input/event*.\n\
                 Either run with sudo -E, or add yourself to the 'input' group:\n\
                 \n\
                     sudo usermod -aG input $USER && newgrp input"
            )
        })?;
        let raw: RawFd = li.as_raw_fd();
        let dup = unsafe { libc::dup(raw) };
        if dup < 0 {
            return Err(anyhow::Error::from(std::io::Error::last_os_error()))
                .context("dup libinput fd");
        }
        let event_fd = unsafe { OwnedFd::from_raw_fd(dup) };
        tracing::info!("libinput attached to seat {seat:?}");
        Ok((Self { li }, event_fd))
    }

    /// Pull all available events from libinput, translating into InputEvent.
    pub fn drain(&mut self) -> Vec<InputEvent> {
        let mut out = Vec::new();
        if let Err(e) = self.li.dispatch() {
            tracing::warn!(error = ?e, "libinput dispatch");
            return out;
        }
        while let Some(ev) = self.li.next() {
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

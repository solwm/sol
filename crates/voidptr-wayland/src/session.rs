//! Session management via libseat.
//!
//! libseat outsources the fiddly bits of running a wayland compositor to
//! a privileged system daemon — either systemd-logind on systemd hosts,
//! or seatd elsewhere. The daemon owns /dev/dri and /dev/input; we ask
//! it for fds, and in return it:
//!   - opens devices as root on our behalf (so voidptr runs as the
//!     logged-in user with no sudo and no group membership tricks);
//!   - coordinates VT switches across every registered session
//!     (pressing Ctrl+Alt+F2 while we're on F3 means we get a
//!     `Disable` event and another session gets `Enable`);
//!   - revokes our device fds while our VT isn't active, so input
//!     events don't leak out of the inactive session.
//!
//! The Session wrapper here is single-threaded — the libseat handle
//! itself is !Send. We expose a calloop-friendly pollable fd, a
//! `dispatch_events` method that drains any pending events into a
//! `Vec`, and pass-through `open_device` / `close_device` /
//! `switch_session` / `disable`. Callers own the Session behind an
//! `Rc<RefCell<Session>>` so the libinput open_restricted callback can
//! share it with the main event loop.

use std::cell::RefCell;
use std::os::fd::BorrowedFd;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use libseat::{Device, Seat, SeatEvent};

/// Shareable handle: libinput's open_restricted callback needs this at
/// the same time the main event loop does.
pub type SharedSession = Rc<RefCell<Session>>;

pub struct Session {
    seat: Seat,
    /// Events are queued into this Vec from the libseat callback (which
    /// fires synchronously during dispatch). The main loop drains it
    /// after each dispatch call so we can process Enable/Disable with
    /// full mutable access to Compositor state.
    queue: Rc<RefCell<Vec<SeatEvent>>>,
    /// Set by the first Enable event and cleared by Disable.
    pub active: bool,
    name: String,
    /// Devices we opened and still want to keep live for the whole
    /// session (DRM device, etc.). Dropped in order on Drop to give
    /// the daemon a chance to revoke cleanly before the seat itself
    /// closes. libinput's per-device lifetimes are managed separately
    /// through its open_restricted / close_restricted callbacks.
    long_lived: Vec<Device>,
}

impl Session {
    /// Open the current seat. libseat picks the backend based on the
    /// `LIBSEAT_BACKEND` env var if set, otherwise tries logind, then
    /// seatd, then falls back to direct-with-setuid (only works if we're
    /// already root — which we aren't). On systemd hosts with an active
    /// graphical session (our TTY login) this just works.
    pub fn open() -> Result<SharedSession> {
        let queue: Rc<RefCell<Vec<SeatEvent>>> = Rc::new(RefCell::new(Vec::new()));
        let queue_for_cb = queue.clone();
        let mut seat = Seat::open(move |_, ev| {
            queue_for_cb.borrow_mut().push(ev);
        })
        .map_err(|e| anyhow!("libseat: open seat: {e}"))?;

        // Pull the seat name once — it never changes for our lifetime
        // and we'd rather not deal with the &mut self requirement every
        // time we want to log it.
        let name = seat.name().to_string();
        tracing::info!(seat = %name, "libseat session opened");

        // libseat always starts in "disabled" state — the first event
        // we'll see is Enable once the daemon confirms we own the seat.
        let session = Session {
            seat,
            queue,
            active: false,
            name,
            long_lived: Vec::new(),
        };
        Ok(Rc::new(RefCell::new(session)))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Pollable fd to wire into calloop. The main loop reads-readable
    /// events from this and then calls `dispatch_events`.
    pub fn poll_fd(&mut self) -> Result<BorrowedFd<'_>> {
        self.seat
            .get_fd()
            .map_err(|e| anyhow!("libseat: get_fd: {e}"))
    }

    /// Drain any libseat events and return them. Non-blocking: timeout
    /// 0 means "read whatever is immediately available and return".
    pub fn dispatch_events(&mut self) -> Result<Vec<SeatEvent>> {
        self.seat
            .dispatch(0)
            .map_err(|e| anyhow!("libseat: dispatch: {e}"))?;
        Ok(std::mem::take(&mut *self.queue.borrow_mut()))
    }

    /// Block up to `timeout_ms` waiting for events. Used at startup to
    /// wait for the initial Enable: logind's D-Bus reply can take a
    /// tens-of-ms round-trip, and with dispatch(0) we'd bail before
    /// the message arrives. Returns the accumulated events.
    pub fn dispatch_events_blocking(
        &mut self,
        timeout_ms: i32,
    ) -> Result<Vec<SeatEvent>> {
        self.seat
            .dispatch(timeout_ms)
            .map_err(|e| anyhow!("libseat: dispatch({timeout_ms}): {e}"))?;
        Ok(std::mem::take(&mut *self.queue.borrow_mut()))
    }

    /// Open a device via the seat daemon. Returns a `Device` owning the
    /// fd; callers must `close_device` when done so the daemon can
    /// revoke cleanly on session teardown.
    pub fn open_device(&mut self, path: &Path) -> Result<Device> {
        self.seat
            .open_device(&path)
            .with_context(|| format!("libseat: open_device({})", path.display()))
    }

    pub fn close_device(&mut self, device: Device) -> Result<()> {
        self.seat
            .close_device(device)
            .map_err(|e| anyhow!("libseat: close_device: {e}"))
    }

    /// Open a device and keep the seat's `Device` handle alive for the
    /// whole session, returning a dup'd `OwnedFd` to the caller. Use
    /// this for long-lived fds (DRM card) where the caller doesn't need
    /// per-device close tracking. libinput manages its own device
    /// lifetimes via its open_restricted / close_restricted callbacks
    /// and goes through `open_device` / `close_device` instead.
    pub fn open_device_keep_fd(
        &mut self,
        path: &Path,
    ) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::{AsFd, AsRawFd, FromRawFd};

        let dev = self.open_device(path)?;
        let raw = dev.as_fd().as_raw_fd();
        let dup = unsafe { libc::dup(raw) };
        if dup < 0 {
            let err = std::io::Error::last_os_error();
            // best-effort revoke so we don't leak on the daemon side.
            let _ = self.close_device(dev);
            return Err(anyhow!("dup libseat fd: {err}"));
        }
        self.long_lived.push(dev);
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(dup) })
    }

    /// Ack a Disable event. Must be called after we've released every
    /// privileged ioctl on our devices (DRM master, etc.); libseat will
    /// hang if we don't.
    pub fn ack_disable(&mut self) -> Result<()> {
        self.seat
            .disable()
            .map_err(|e| anyhow!("libseat: disable: {e}"))
    }

    /// Ask the seat daemon to switch to the given session number. On a
    /// VT-bound seat (the only kind we care about) this becomes a VT
    /// switch via VT_ACTIVATE. Call doesn't block; the actual switch
    /// arrives later as a Disable event.
    pub fn switch_session(&mut self, session: i32) -> Result<()> {
        self.seat
            .switch_session(session)
            .map_err(|e| anyhow!("libseat: switch_session({session}): {e}"))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Close every long-lived device before the seat itself goes
        // away so logind/seatd can revoke cleanly. Per-device libinput
        // fds were closed by their own close_restricted path already.
        for dev in self.long_lived.drain(..) {
            if let Err(e) = self.seat.close_device(dev) {
                tracing::warn!(%e, "libseat: close_device on session drop");
            }
        }
        // Seat's own Drop calls libseat_close_seat after this.
    }
}

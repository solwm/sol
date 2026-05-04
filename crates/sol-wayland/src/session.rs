//! Session management via libseat — backed by smithay.
//!
//! libseat outsources the fiddly bits of running a wayland compositor to
//! a privileged system daemon — either systemd-logind on systemd hosts,
//! or seatd elsewhere. The daemon owns /dev/dri and /dev/input; we ask
//! it for fds, and in return it:
//!   - opens devices as root on our behalf (so sol runs as the
//!     logged-in user with no sudo and no group membership tricks);
//!   - coordinates VT switches across every registered session
//!     (pressing Ctrl+Alt+F2 while we're on F3 means we get a
//!     `Disable` event and another session gets `Enable`);
//!   - revokes our device fds while our VT isn't active, so input
//!     events don't leak out of the inactive session.
//!
//! Smithay's [`LibSeatSession`] is `Clone` (it holds a `Weak` to the
//! shared inner state) and implements the `Session` trait — `open` /
//! `close` / `change_vt` / `is_active` — so we can hand the same
//! handle to libinput's `LibinputInterface`, the DRM device-open path,
//! and the calloop event source without an `Rc<RefCell<>>` wrapper of
//! our own.

use anyhow::{Result, anyhow};

pub use smithay::backend::session::Event as SessionEvent;
pub use smithay::backend::session::Session;
pub use smithay::backend::session::libseat::{LibSeatSession, LibSeatSessionNotifier};

/// Shareable session handle. Smithay's `LibSeatSession` is already
/// `Clone` (cheap weak-ref clone), so the type alias is for naming
/// continuity with the previous `Rc<RefCell<Session>>` callers — no
/// allocation cost of its own.
pub type SharedSession = LibSeatSession;

/// Open the seat. Returns the cloneable session handle plus the
/// calloop event source that drives its async Enable/Disable signals.
///
/// On a typical systemd host the initial Enable arrives during the
/// constructor's first dispatch and `session.is_active()` returns
/// `true` right away. If it doesn't (slower setups, unusual session
/// config), we bail with a clear error rather than racing the rest of
/// startup against a still-disabled seat — opening `/dev/dri` while
/// disabled would fail in a less-readable way.
pub fn open() -> Result<(LibSeatSession, LibSeatSessionNotifier)> {
    let (session, notifier) = LibSeatSession::new()
        .map_err(|e| anyhow!("libseat: open seat: {e}"))?;
    if !session.is_active() {
        anyhow::bail!(
            "libseat: seat opened but no Enable event arrived in the initial \
             dispatch. The daemon hasn't handed us the seat. Make sure sol \
             was launched from an active logind session (check \
             `loginctl show-session $XDG_SESSION_ID` from the TTY where \
             you're running it)."
        );
    }
    tracing::info!(seat = %session.seat(), "libseat session active");
    Ok((session, notifier))
}

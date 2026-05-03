//! Vulkan + DRM/KMS rendering backend for sol.
//!
//! The contract with the rest of the compositor is `DrmPresenter`: build
//! it from a libseat-granted DRM fd, hand it `Scene` values produced by
//! `sol-wayland`, get back an optional GPU-completion FD that the
//! wayland-side calloop registers so the page flip can defer until
//! rendering's done.
//!
//! What this crate is responsible for:
//! - DRM device introspection (`describe_device`, `pick_output`).
//! - Mode preference parsing (`ModePreference`, `parse_mode_pref`).
//! - The `Card` wrapper that hands a DRM fd to drm-rs through its
//!   trait set.
//! - The Vulkan stack, GBM-backed scan-out swap, graphics pipelines,
//!   texture cache, and blur FBOs (in their `vk_*` modules).
//! - The presenter that ties everything together (`presenter::DrmPresenter`).

#![allow(clippy::collapsible_if)]

use std::fs::{File, OpenOptions};
use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use drm::Device as BasicDevice;
use drm::control::{Device as ControlDevice, ModeTypeFlags, connector};

mod presenter;
mod vk_blur;
mod vk_pipe;
mod vk_stack;
mod vk_swap;
mod vk_texture;

pub use presenter::DrmPresenter;

/// Wrapper that makes `File` satisfy drm-rs's trait set. drm-rs relies on
/// implementors to provide an FD and lets the traits dispatch ioctls.
#[derive(Debug, Clone)]
pub struct Card(Arc<File>);

impl Card {
    /// Open the DRM device directly. Requires the caller to have access
    /// to `/dev/dri/cardN`; on a regular user that means `video` group
    /// membership, or sudo. The main `sol` binary takes the libseat
    /// path instead via `Card::from_fd`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Card(Arc::new(file)))
    }

    /// Wrap an already-opened DRM fd (typically obtained via
    /// `libseat::open_device`, which runs as the daemon's user and
    /// hands us the fd via the seat protocol). Lets sol run as an
    /// unprivileged user without any group membership.
    pub fn from_fd(fd: std::os::fd::OwnedFd) -> Self {
        Card(Arc::new(File::from(fd)))
    }
}

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl BasicDevice for Card {}
impl ControlDevice for Card {}

#[derive(Debug, Clone, Copy)]
pub struct OutputSelection {
    pub connector: connector::Handle,
    pub crtc: drm::control::crtc::Handle,
    pub mode: drm::control::Mode,
}

/// Mode selection preference: width × height at the given integer refresh.
/// Exposed as `SOL_MODE=WxH@Hz`. When absent, sol picks the
/// highest-refresh mode at the connector's PREFERRED resolution.
#[derive(Debug, Clone, Copy)]
pub struct ModePreference {
    pub width: u16,
    pub height: u16,
    pub refresh_hz: u32,
}

impl ModePreference {
    fn matches(&self, m: &drm::control::Mode) -> bool {
        let (w, h) = m.size();
        w == self.width && h == self.height && m.vrefresh() == self.refresh_hz
    }
}

/// Parse `WxH@Hz` (e.g., `3840x2160@240`). Returns None on any parse error.
fn parse_mode_pref(s: &str) -> Option<ModePreference> {
    let (wh, hz) = s.split_once('@')?;
    let (w, h) = wh.split_once('x')?;
    Some(ModePreference {
        width: w.trim().parse().ok()?,
        height: h.trim().parse().ok()?,
        refresh_hz: hz.trim().parse().ok()?,
    })
}

/// Describe what's connected without touching master. Safe to run while
/// another compositor (Hyprland, whatever) holds DRM master on the same VT.
pub fn describe_device(device: &Path) -> Result<()> {
    let card = Card::open(device)?;
    let res = card.resource_handles().context("resource_handles")?;

    println!("device: {}", device.display());
    println!("  connectors: {}", res.connectors().len());
    println!("  encoders:   {}", res.encoders().len());
    println!("  crtcs:      {}", res.crtcs().len());

    for &h in res.connectors() {
        let conn = match card.get_connector(h, false) {
            Ok(c) => c,
            Err(e) => {
                println!("  connector {h:?}: <get_connector failed: {e}>");
                continue;
            }
        };
        println!(
            "  connector {:?}: {:?} state={:?} modes={}",
            h,
            conn.interface(),
            conn.state(),
            conn.modes().len()
        );
        if conn.state() == connector::State::Connected {
            for (i, m) in conn.modes().iter().enumerate() {
                let (w, h) = m.size();
                println!(
                    "    mode[{i}]: {w}x{h}@{hz} {flags}",
                    hz = m.vrefresh(),
                    flags = if m.mode_type().contains(ModeTypeFlags::PREFERRED) {
                        "(preferred)"
                    } else {
                        ""
                    }
                );
            }
        }
    }
    Ok(())
}

pub fn pick_output(
    card: &Card,
    config_pref: Option<ModePreference>,
) -> Result<OutputSelection> {
    // Parse SOL_MODE if set. An unparseable value is rejected up front —
    // better to fail loud than silently ignore a typo like `3840X2160@240`.
    // SOL_MODE wins over the config-supplied preference: env is the
    // ad-hoc one-off, config is the persistent intent.
    let env_raw = std::env::var("SOL_MODE").ok();
    let env_pref = match env_raw.as_deref() {
        Some(s) => Some(parse_mode_pref(s).ok_or_else(|| {
            anyhow!("SOL_MODE={s:?} is not of the form WxH@Hz (e.g., 3840x2160@240)")
        })?),
        None => None,
    };
    let env_pref = env_pref.or(config_pref);

    let res = card.resource_handles().context("resource_handles")?;

    for &connector_h in res.connectors() {
        let conn = card
            .get_connector(connector_h, false)
            .with_context(|| format!("get_connector {connector_h:?}"))?;
        if conn.state() != connector::State::Connected {
            continue;
        }
        if conn.modes().is_empty() {
            continue;
        }
        for m in conn.modes() {
            let (mw, mh) = m.size();
            tracing::info!(
                connector = ?conn.interface(),
                width = mw,
                height = mh,
                vrefresh = m.vrefresh(),
                preferred = m.mode_type().contains(ModeTypeFlags::PREFERRED),
                "available mode"
            );
        }
        let preferred = conn
            .modes()
            .iter()
            .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .copied();
        let (mode, source) = if let Some(p) = env_pref {
            match conn.modes().iter().find(|m| p.matches(m)).copied() {
                Some(m) => (m, "env"),
                None => bail!(
                    "SOL_MODE {}x{}@{} not advertised by connector {:?}; \
                     run `just drm-info` to see what's available",
                    p.width,
                    p.height,
                    p.refresh_hz,
                    conn.interface(),
                ),
            }
        } else if let Some(pm) = preferred {
            let (pw, ph) = pm.size();
            let best = conn
                .modes()
                .iter()
                .filter(|m| m.size() == (pw, ph))
                .max_by_key(|m| m.vrefresh())
                .copied()
                .unwrap_or(pm);
            if best.vrefresh() > pm.vrefresh() {
                (best, "preferred-size-max-hz")
            } else {
                (best, "preferred")
            }
        } else {
            (conn.modes()[0], "first")
        };

        let encoders = conn.encoders().to_vec();
        for enc_h in encoders {
            let enc = match card.get_encoder(enc_h) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let compatible_crtcs = res.filter_crtcs(enc.possible_crtcs());
            if let Some(&crtc_h) = compatible_crtcs.first() {
                tracing::info!(
                    connector = ?conn.interface(),
                    width = mode.size().0,
                    height = mode.size().1,
                    vrefresh = mode.vrefresh(),
                    source,
                    "selected output"
                );
                return Ok(OutputSelection {
                    connector: connector_h,
                    crtc: crtc_h,
                    mode,
                });
            }
        }
    }
    bail!("no connected connector with a usable CRTC found")
}

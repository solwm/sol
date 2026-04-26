//! sol — the sol compositor binary.
//!
//! Usage:
//!   sol                       # default (headless) backend
//!   sol --backend=headless    # explicit
//!   sol --backend=drm         # real DRM+GBM+GLES, from a free VT
//!
//! Extra env knobs:
//!   SOL_PNG_PATH    (headless) where to dump the frame PNG
//!   SOL_DRM_DEVICE  (drm) path to /dev/dri/cardN, default /dev/dri/card2

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    let mut backend = "headless".to_string();
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--backend=") {
            backend = v.to_string();
        } else if a == "--help" || a == "-h" {
            print_help();
            return Ok(());
        } else {
            bail!("unknown arg: {a}");
        }
    }
    // Env var wins over arg so `just` recipes can override easily.
    if let Ok(b) = std::env::var("SOL_BACKEND") {
        backend = b;
    }

    tracing::info!(%backend, "sol starting");

    // Normalise the environment before the backend (and therefore every
    // spawned client) inherits it. Children pick up our env via
    // std::process::Command's default inheritance, so fixing it once
    // here is enough. `set_var` / `remove_var` are unsafe in modern
    // Rust (thread-safety), but we're pre-threads here so it's fine.
    unsafe {
        // Tell clients we're a Wayland session — some (notably GTK,
        // Firefox via MOZ_ENABLE_WAYLAND auto-detect, Chromium's
        // Ozone) check this to decide their display backend.
        std::env::set_var("XDG_SESSION_TYPE", "wayland");
        std::env::set_var("XDG_CURRENT_DESKTOP", "sol");
        // No X server here. Strip DISPLAY / XAUTHORITY so clients that
        // auto-detect either-or don't try an X path that'll never
        // work (and waste seconds failing at it — e.g. mplayer's
        // VO probe).
        std::env::remove_var("DISPLAY");
        std::env::remove_var("XAUTHORITY");
    }

    // Log what we're passing through so config-not-found complaints
    // are one line away from the cause.
    for key in ["HOME", "USER", "SHELL", "PATH", "XDG_RUNTIME_DIR", "LANG"] {
        tracing::info!(
            key,
            value = std::env::var(key).as_deref().unwrap_or("<unset>"),
            "env inherited"
        );
    }

    match backend.as_str() {
        "headless" => {
            let png_path = std::env::var_os("SOL_PNG_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/tmp/sol-headless.png"));
            sol_wayland::run_headless(png_path, 1920, 1080)
        }
        "drm" => {
            let device = std::env::var_os("SOL_DRM_DEVICE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/dev/dri/card2"));
            sol_wayland::run_drm(&device).context("run_drm")
        }
        other => bail!("unknown backend: {other}"),
    }
}

fn print_help() {
    println!(
        "sol [--backend=headless|drm]\n\
         \n\
         Env:\n\
           SOL_BACKEND     overrides --backend\n\
           SOL_PNG_PATH    (headless) PNG dump path\n\
           SOL_DRM_DEVICE  (drm) path to /dev/dri/cardN"
    );
}

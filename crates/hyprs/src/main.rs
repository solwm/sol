//! hyprs — the hyperland-rs compositor binary.
//!
//! Usage:
//!   hyprs                       # default (headless) backend
//!   hyprs --backend=headless    # explicit
//!   hyprs --backend=drm         # real DRM+GBM+GLES, from a free VT
//!
//! Extra env knobs:
//!   HYPRS_PNG_PATH    (headless) where to dump the frame PNG
//!   HYPRS_DRM_DEVICE  (drm) path to /dev/dri/cardN, default /dev/dri/card2

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
    if let Ok(b) = std::env::var("HYPRS_BACKEND") {
        backend = b;
    }

    tracing::info!(%backend, "hyperland-rs starting");

    match backend.as_str() {
        "headless" => {
            let png_path = std::env::var_os("HYPRS_PNG_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/tmp/hyprs-headless.png"));
            hypr_wayland::run_headless(png_path, 1920, 1080)
        }
        "drm" => {
            let device = std::env::var_os("HYPRS_DRM_DEVICE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/dev/dri/card2"));
            hypr_wayland::run_drm(&device).context("run_drm")
        }
        other => bail!("unknown backend: {other}"),
    }
}

fn print_help() {
    println!(
        "hyprs [--backend=headless|drm]\n\
         \n\
         Env:\n\
           HYPRS_BACKEND     overrides --backend\n\
           HYPRS_PNG_PATH    (headless) PNG dump path\n\
           HYPRS_DRM_DEVICE  (drm) path to /dev/dri/cardN"
    );
}

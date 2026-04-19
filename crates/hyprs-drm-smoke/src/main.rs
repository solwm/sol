//! CLI wrapper around `hypr_backend_drm`.
//!
//! Three modes:
//!   hyprs-drm-smoke info       [DEVICE]           # list connectors/modes (no master)
//!   hyprs-drm-smoke gpu-render [DEVICE] [OUT]     # GBM+EGL+GLES offscreen -> PNG (no master)
//!   hyprs-drm-smoke run        [DEVICE] [SECS]    # full modeset + render loop (needs free VT)

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "run".into());
    let device: PathBuf = args
        .next()
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HYPRS_DRM_DEVICE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/dev/dri/card1"));

    match mode.as_str() {
        "info" => hypr_backend_drm::describe_device(&device).context("describe_device"),
        "gpu-render" => {
            let out = args
                .next()
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HYPRS_DRM_GPU_PNG").map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("/tmp/hyprs-gpu.png"));
            hypr_backend_drm::run_offscreen_render(&device, &out, 1280, 720)
                .context("offscreen render")
        }
        "run" => {
            let seconds: u64 = args
                .next()
                .and_then(|s| s.parse().ok())
                .or_else(|| std::env::var("HYPRS_DRM_DURATION").ok().and_then(|s| s.parse().ok()))
                .unwrap_or(10);
            let should_quit = Arc::new(AtomicBool::new(false));
            {
                let flag = should_quit.clone();
                ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))?;
            }
            tracing::info!(
                device = %device.display(),
                seconds,
                "starting DRM smoke test"
            );
            if let Err(e) = hypr_backend_drm::run_smoke_test(
                &device,
                Duration::from_secs(seconds),
                should_quit,
            ) {
                eprintln!("\nERROR: {e}");
                let mut src = e.source();
                let mut depth = 1;
                while let Some(s) = src {
                    eprintln!("  caused by [{depth}]: {s}");
                    src = s.source();
                    depth += 1;
                }
                std::process::exit(1);
            }
            Ok(())
        }
        other => {
            eprintln!("unknown mode: {other}\nusage: hyprs-drm-smoke [info|run] [DEVICE] [SECS]");
            std::process::exit(2);
        }
    }
}

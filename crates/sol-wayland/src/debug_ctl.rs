//! Scriptable control socket for the `sol-bench` harness.
//!
//! Compiled in only when the `debug-ctl` feature is enabled. Sol
//! listens on `$XDG_RUNTIME_DIR/sol-ctl-<wayland-display>.sock` (one
//! file per running compositor) and accepts a single client at a
//! time — that's the harness. Each side speaks line-delimited JSON
//! over the stream:
//!
//!   client → server: one command per line (see `Command`)
//!   server → client: one response per line (see `Response`)
//!
//! Commands route through the same internal entry points as
//! keybinds (`focus_direction`, `switch_workspace`, …), so a
//! synthetic move exercises the real code path. Spawn uses the same
//! `spawn_client` as `bind = exec, …` — bench scripts can launch
//! real Wayland clients to exercise the full pipeline. Snapshot
//! returns the live `Metrics` dump without exiting; shutdown also
//! returns it, then asks calloop to stop.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;

use anyhow::Result;
use calloop::{Interest, LoopHandle, LoopSignal, Mode, PostAction, generic::Generic};
use serde::{Deserialize, Serialize};

use crate::{Compositor, config};

/// Lives in `State.debug_ctl` while the feature is enabled. Owns the
/// socket path (so Drop can unlink it) and the read buffer; the
/// listener and per-connection sources live inside calloop.
pub struct DebugCtl {
    pub socket_path: PathBuf,
    pub signal: LoopSignal,
    /// Receive buffer accumulating bytes between source firings —
    /// commands aren't guaranteed to arrive whole, especially when
    /// the harness pipelines a long script.
    pub recv_buf: Vec<u8>,
    /// True while a per-connection source is in calloop. Second
    /// connections while one is active get refused at accept time.
    pub conn_active: bool,
}

impl Drop for DebugCtl {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Bind the socket and register the accept source. Returned value
/// goes into `State.debug_ctl`; the per-connection source is added
/// later, on first accept.
pub fn install(
    handle: &LoopHandle<'static, Compositor>,
    signal: LoopSignal,
    socket_path: PathBuf,
) -> Result<DebugCtl> {
    // Stale socket from a previous crash would EADDRINUSE bind. The
    // socket file isn't shared with anything else, so unlinking
    // before bind is safe.
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;
    listener.set_nonblocking(true)?;
    tracing::info!(path = %socket_path.display(), "debug-ctl listening");

    let inner_handle = handle.clone();
    handle
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            move |_, listener, comp: &mut Compositor| {
                loop {
                    match listener.accept() {
                        Ok((stream, _)) => accept_one(&inner_handle, comp, stream),
                        Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                        Err(e) => {
                            tracing::warn!(error = %e, "debug-ctl accept failed");
                            break;
                        }
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert ctl listener source: {e}"))?;

    Ok(DebugCtl {
        socket_path,
        signal,
        recv_buf: Vec::with_capacity(4096),
        conn_active: false,
    })
}

fn accept_one(
    handle: &LoopHandle<'static, Compositor>,
    comp: &mut Compositor,
    stream: UnixStream,
) {
    let Some(ctl) = comp.state.debug_ctl.as_mut() else { return };
    if ctl.conn_active {
        tracing::warn!("debug-ctl: another connection already active; rejecting");
        return;
    }
    if let Err(e) = stream.set_nonblocking(true) {
        tracing::warn!(error = %e, "debug-ctl set_nonblocking failed");
        return;
    }
    let res = handle.insert_source(
        Generic::new(stream, Interest::READ, Mode::Level),
        // Generic hands us `&UnixStream` (its NoIoDrop wrapper isn't
        // DerefMut). UnixStream implements Read/Write for `&Self`, so
        // we can still I/O through a shared reference.
        |_, sock, comp: &mut Compositor| Ok(handle_readable(sock, comp)),
    );
    match res {
        Ok(_) => {
            ctl.conn_active = true;
            ctl.recv_buf.clear();
            tracing::info!("debug-ctl: client connected");
        }
        Err(e) => tracing::warn!(error = %e, "debug-ctl insert per-conn source failed"),
    }
}

fn handle_readable(stream: &UnixStream, comp: &mut Compositor) -> PostAction {
    // UnixStream's Read/Write impls for &Self use the same fd.
    let mut chunk = [0u8; 4096];
    loop {
        match (&*stream).read(&mut chunk) {
            Ok(0) => {
                // EOF — client closed.
                tracing::info!("debug-ctl: client disconnected");
                if let Some(ctl) = comp.state.debug_ctl.as_mut() {
                    ctl.conn_active = false;
                    ctl.recv_buf.clear();
                }
                return PostAction::Remove;
            }
            Ok(n) => {
                if let Some(ctl) = comp.state.debug_ctl.as_mut() {
                    ctl.recv_buf.extend_from_slice(&chunk[..n]);
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => break,
            Err(e) => {
                tracing::warn!(error = %e, "debug-ctl read failed");
                if let Some(ctl) = comp.state.debug_ctl.as_mut() {
                    ctl.conn_active = false;
                    ctl.recv_buf.clear();
                }
                return PostAction::Remove;
            }
        }
    }

    // Drain complete lines from the buffer, dispatch each, write
    // the response. We pop a line into a local before dispatch so
    // the borrow on State.debug_ctl.recv_buf doesn't clash with
    // dispatch's wider mut access to State.
    loop {
        let line = {
            let Some(ctl) = comp.state.debug_ctl.as_mut() else { return PostAction::Continue };
            let Some(nl) = ctl.recv_buf.iter().position(|&b| b == b'\n') else { break };
            let mut bytes: Vec<u8> = ctl.recv_buf.drain(..=nl).collect();
            bytes.pop(); // strip '\n'
            bytes
        };
        let line_str = std::str::from_utf8(&line).unwrap_or("").trim();
        if line_str.is_empty() {
            continue;
        }
        let response = dispatch(line_str, comp);
        let mut bytes = response.into_bytes();
        bytes.push(b'\n');
        if let Err(e) = (&*stream).write_all(&bytes) {
            tracing::warn!(error = %e, "debug-ctl write failed");
            if let Some(ctl) = comp.state.debug_ctl.as_mut() {
                ctl.conn_active = false;
                ctl.recv_buf.clear();
            }
            return PostAction::Remove;
        }
    }
    PostAction::Continue
}

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    Focus { dir: String },
    Move { dir: String },
    Workspace { n: u32 },
    MoveToWorkspace { n: u32 },
    ToggleZoom,
    ToggleFullscreen,
    Close,
    Spawn {
        program: String,
        #[serde(default)]
        args: Vec<String>,
    },
    Snapshot,
    Shutdown,
}

#[derive(Serialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<MetricsDump>,
}

#[derive(Serialize)]
struct MetricsDump {
    uptime_ms: u64,
    frames_rendered: u64,
    /// Render-tick calls that early-bailed because a page flip was
    /// still in flight. Cheap (sub-microsecond) so they don't show
    /// up in `frame_time_ms_buckets` — but high counts here mean
    /// many wakeups while we were waiting for vblank, which is
    /// usually a sign of `needs_render` being set redundantly.
    ticks_skipped: u64,
    /// Frames that early-returned because the scene digest matched
    /// the last successfully-flipped one. Each entry here is a
    /// ~2 ms `lock_front_buffer` we avoided.
    flips_skipped: u64,
    page_flips: u64,
    render_tick_total_ns: u64,
    render_tick_max_ns: u64,
    frame_time_ms_buckets: [u64; 6],
    /// Per-phase wall-clock totals over every real render. Compare
    /// across snapshots to see which phase a particular workload
    /// hit hardest. Phases: prune, layout, animations, scene
    /// collection, render (backend handoff).
    phase_prune_ns: u64,
    phase_layout_ns: u64,
    phase_animations_ns: u64,
    phase_collect_scene_ns: u64,
    phase_render_ns: u64,
    phase_render_textures_ns: u64,
    phase_render_blur_ns: u64,
    phase_render_draw_ns: u64,
    phase_render_present_ns: u64,
    phase_render_present_swap_ns: u64,
    phase_render_present_lock_ns: u64,
    phase_render_present_addfb_ns: u64,
    phase_render_present_pageflip_ns: u64,
    spring_ticks: u64,
    input_events: u64,
    ctl_commands: u64,
    clients_connected: u64,
    mapped_toplevels: usize,
    mapped_dialogs: usize,
}

fn dispatch(line: &str, comp: &mut Compositor) -> String {
    comp.state.metrics.ctl_commands += 1;
    let cmd: Command = match serde_json::from_str(line) {
        Ok(c) => c,
        Err(e) => return err_response(format!("parse: {e}")),
    };
    match cmd {
        Command::Focus { dir } => match parse_dir(&dir) {
            Ok(d) => {
                crate::focus_direction(&mut comp.state, d);
                comp.state.needs_render = true;
                ok_response()
            }
            Err(e) => err_response(e),
        },
        Command::Move { dir } => match parse_dir(&dir) {
            Ok(d) => {
                crate::move_direction(&mut comp.state, d);
                comp.state.needs_render = true;
                ok_response()
            }
            Err(e) => err_response(e),
        },
        Command::Workspace { n } => {
            crate::switch_workspace(&mut comp.state, n);
            comp.state.needs_render = true;
            ok_response()
        }
        Command::MoveToWorkspace { n } => {
            crate::move_focused_to_workspace(&mut comp.state, n);
            comp.state.needs_render = true;
            ok_response()
        }
        Command::ToggleZoom => {
            crate::toggle_zoom(&mut comp.state);
            ok_response()
        }
        Command::ToggleFullscreen => {
            crate::toggle_fullscreen(&mut comp.state);
            ok_response()
        }
        Command::Close => {
            crate::close_focused_window(&mut comp.state);
            ok_response()
        }
        Command::Spawn { program, args } => {
            let argrefs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            crate::spawn_client(&comp.state, &program, &argrefs, "debug-ctl");
            ok_response()
        }
        Command::Snapshot => snapshot_response(comp),
        Command::Shutdown => {
            let resp = snapshot_response(comp);
            if let Some(ctl) = comp.state.debug_ctl.as_ref() {
                ctl.signal.stop();
            }
            resp
        }
    }
}

fn parse_dir(s: &str) -> Result<config::Direction, String> {
    match s.to_ascii_lowercase().as_str() {
        "left" | "h" => Ok(config::Direction::Left),
        "down" | "j" => Ok(config::Direction::Down),
        "up" | "k" => Ok(config::Direction::Up),
        "right" | "l" => Ok(config::Direction::Right),
        other => Err(format!("unknown dir: {other}")),
    }
}

fn ok_response() -> String {
    serde_json::to_string(&Response { ok: true, error: None, metrics: None }).unwrap()
}

fn err_response(msg: String) -> String {
    serde_json::to_string(&Response { ok: false, error: Some(msg), metrics: None }).unwrap()
}

fn snapshot_response(comp: &Compositor) -> String {
    let m = &comp.state.metrics;
    let dump = MetricsDump {
        uptime_ms: comp.state.started.elapsed().as_millis() as u64,
        frames_rendered: m.frames_rendered,
        ticks_skipped: m.ticks_skipped,
        flips_skipped: m.flips_skipped,
        page_flips: m.page_flips,
        render_tick_total_ns: m.render_tick_total_ns,
        render_tick_max_ns: m.render_tick_max_ns,
        frame_time_ms_buckets: m.frame_time_ms_buckets,
        phase_prune_ns: m.phase_prune_ns,
        phase_layout_ns: m.phase_layout_ns,
        phase_animations_ns: m.phase_animations_ns,
        phase_collect_scene_ns: m.phase_collect_scene_ns,
        phase_render_ns: m.phase_render_ns,
        phase_render_textures_ns: m.phase_render_textures_ns,
        phase_render_blur_ns: m.phase_render_blur_ns,
        phase_render_draw_ns: m.phase_render_draw_ns,
        phase_render_present_ns: m.phase_render_present_ns,
        phase_render_present_swap_ns: m.phase_render_present_swap_ns,
        phase_render_present_lock_ns: m.phase_render_present_lock_ns,
        phase_render_present_addfb_ns: m.phase_render_present_addfb_ns,
        phase_render_present_pageflip_ns: m.phase_render_present_pageflip_ns,
        spring_ticks: m.spring_ticks,
        input_events: m.input_events,
        ctl_commands: m.ctl_commands,
        clients_connected: m.clients_connected,
        mapped_toplevels: comp.state.mapped_toplevels.len(),
        mapped_dialogs: comp.state.mapped_dialogs.len(),
    };
    serde_json::to_string(&Response { ok: true, error: None, metrics: Some(dump) }).unwrap()
}

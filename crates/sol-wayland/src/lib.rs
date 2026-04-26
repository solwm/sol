//! Wayland server for sol.
//!
//! Handles protocol traffic in all backends; rendering is delegated to a
//! `BackendState` value (software canvas -> PNG for headless, or a
//! `sol_backend_drm::DrmPresenter` for real hardware).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic};
use sol_backend_drm::DrmPresenter;
use std::os::fd::AsRawFd;
use sol_core::{Scene, SceneContent, SceneElement};
use wayland_protocols::xdg::shell::server::xdg_wm_base::XdgWmBase;
use wayland_server::{
    Display, DisplayHandle, Resource, Weak,
    backend::{ClientData, ClientId, DisconnectReason, GlobalId},
    protocol::{
        wl_compositor::WlCompositor, wl_output::WlOutput, wl_seat::WlSeat, wl_shm::WlShm,
        wl_surface::WlSurface,
    },
};

mod compositor;
mod config;
mod cursor;
mod data_device;
mod ext_workspace;
mod fractional_scale;
mod idle_inhibit;
mod input;
mod layer_shell;
mod linux_dmabuf;
mod output;
mod presentation_time;
mod render;
mod seat;
mod session;
mod shm;
mod subcompositor;
mod viewporter;
mod xdg_decoration;
mod xdg_output;
mod xdg_shell;
mod xkb;

use compositor::{SurfaceData, SurfaceRole};
use input::{AxisSource, InputEvent, InputState};
use render::Canvas;
use wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::xdg::decoration::zv1::server::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_protocols::wp::fractional_scale::v1::server::wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1;
use wayland_protocols::wp::presentation_time::server::wp_presentation::WpPresentation;
use wayland_protocols::wp::viewporter::server::wp_viewporter::WpViewporter;
use wayland_protocols::xdg::xdg_output::zv1::server::zxdg_output_manager_v1::ZxdgOutputManagerV1;
use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::ZwlrLayerShellV1;
use wayland_server::protocol::{
    wl_data_device_manager::WlDataDeviceManager,
    wl_keyboard::{self, WlKeyboard},
    wl_pointer::{self, WlPointer},
    wl_subcompositor::WlSubcompositor,
};
use wayland_protocols::ext::workspace::v1::server::ext_workspace_manager_v1::ExtWorkspaceManagerV1;
use wayland_protocols::wp::idle_inhibit::zv1::server::{
    zwp_idle_inhibit_manager_v1::ZwpIdleInhibitManagerV1,
    zwp_idle_inhibitor_v1::ZwpIdleInhibitorV1,
};
use xkb::KeymapState;

const COMPOSITOR_VERSION: u32 = 6;
const SHM_VERSION: u32 = 1;
const OUTPUT_VERSION: u32 = 4;
const SEAT_VERSION: u32 = 8;
const XDG_WM_BASE_VERSION: u32 = 5;

/// Screen-space rectangle assigned to a mapped window by the layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Sub-pixel-precision counterpart of [`Rect`], used everywhere the
/// value will be drawn to the GPU and could land between integer
/// pixels: animation interpolation (`render_rect`, `from_rect`) and
/// scene transport (`PlacedBuffer`, `SceneElement`).
///
/// Layout math, focus borders, and hit-testing keep using integer
/// `Rect` — those care about whole pixels — but interpolation needs
/// the float so a 36-frame ease-out (150 ms at 240 Hz) doesn't round
/// to 0-pixel-then-1-pixel steps near the curve's tail. Conversion
/// goes through `From`/`Into` and `RectF::round`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectF {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl From<Rect> for RectF {
    fn from(r: Rect) -> Self {
        RectF {
            x: r.x as f32,
            y: r.y as f32,
            w: r.w as f32,
            h: r.h as f32,
        }
    }
}

impl RectF {
    /// Round to the nearest whole pixel. Used by the few consumers
    /// (focus border, hit-test) that need integer rects.
    pub fn round(self) -> Rect {
        Rect {
            x: self.x.round() as i32,
            y: self.y.round() as i32,
            w: self.w.round() as i32,
            h: self.h.round() as i32,
        }
    }
}

/// A mapped xdg_toplevel together with the screen-space rect the layout
/// assigned to it.
///
/// - `rect`: the target rect — what the layout wants the tile to be,
///   right now. Updated by `apply_layout` when the layout changes
///   (tile closed, move/focus command, zoom, etc.).
/// - `render_rect`: where the tile is actually drawn this frame.
///   `tick_animations` interpolates from `from_rect` toward `rect` over
///   `config.animation_duration_ms` using `config.animation_curve`, so layout changes read
///   as motion rather than instant snaps. The buffer is GPU-scaled to
///   `render_rect` regardless of its intrinsic size — during the tween
///   it's stretched/squished smoothly, and the eye reads the scaling
///   as resize motion rather than artifact.
/// - `from_rect`: snapshot of `render_rect` at the moment the current
///   tween started. The interpolation origin.
/// - `anim_started_at`: wall-clock the current tween started, or
///   `None` when settled. `apply_layout` writes this whenever it
///   changes `rect`.
/// - `pending_size` is the (w, h) most recently sent to the client via
///   `xdg_toplevel.configure`; layout sends a fresh configure only
///   when the target rect differs, so we don't spam configures every
///   frame.
pub struct Window {
    pub surface: Weak<WlSurface>,
    pub rect: Rect,
    pub render_rect: RectF,
    pub from_rect: RectF,
    pub anim_started_at: Option<Instant>,
    pub pending_size: Option<(i32, i32)>,
    /// Which workspace this window belongs to. 1-based. A window
    /// renders and receives input only when its workspace matches
    /// `State.active_ws`; otherwise it stays mapped but hidden.
    pub workspace: u32,
}

/// In-flight workspace crossfade. While set, both the outgoing and
/// the (already-promoted) incoming workspace render simultaneously,
/// with alpha derived from `now - started_at` against
/// `config.workspace_animation_duration_ms` and the configured
/// easing curve.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceAnim {
    /// The workspace we're leaving. Its windows render on top during
    /// the animation, fading from alpha 1.0 → 0.0.
    pub from_ws: u32,
    /// The workspace we just switched to (already in `state.active_ws`).
    /// Fades 0.0 → 1.0.
    pub to_ws: u32,
    /// Wall-clock the switch was triggered.
    pub started_at: Instant,
}

/// Compositor state shared across Dispatch impls. Backend-specific resources
/// (canvas, presenter) live on `Compositor` alongside this, not here.
pub struct State {
    pub display_handle: DisplayHandle,
    pub globals: Globals,
    pub clients_seen: u64,
    /// Mapped xdg_toplevels in stacking order (bottom to top). Held as weak
    /// refs so dead surfaces drop out on the next render.
    pub mapped_toplevels: Vec<Window>,
    /// Logical screen size. Drives toplevel placement and the
    /// wl_output mode advertised to clients.
    pub screen_width: u32,
    pub screen_height: u32,
    /// Refresh rate of the selected DRM mode, in milli-Hz (so 240 Hz
    /// is 240_000). Advertised via `wl_output.mode`: browsers (Chrome,
    /// Firefox) and other animation-aware clients use this to decide
    /// how fast to commit buffers. Hardcoded to 60_000 in headless —
    /// nothing actually paces off it there.
    pub screen_refresh_mhz: i32,
    pub needs_render: bool,
    pub started: Instant,
    pub next_serial: u32,
    pub cursor: Cursor,
    pub input: Option<InputState>,
    pub keymap: Option<KeymapState>,
    /// wl_pointer resources bound by clients. Cleaned on each tick.
    pub pointers: Vec<WlPointer>,
    /// wl_keyboard resources bound by clients.
    pub keyboards: Vec<WlKeyboard>,
    /// Surface currently under the cursor, if any.
    pub pointer_focus: Option<WlSurface>,
    /// Surface currently receiving keyboard events.
    pub keyboard_focus: Option<WlSurface>,
    /// wl_callback objects clients requested via wl_surface.frame and are
    /// waiting on. We stash them here during commit and fire them from the
    /// render_tick once the backend has actually presented the frame; that
    /// implicit throttle keeps clients in lock-step with our vblank cadence
    /// instead of over-rendering at max CPU speed.
    pub pending_frame_callbacks: Vec<wayland_server::protocol::wl_callback::WlCallback>,
    /// DRM device path used by the backend, so zwp_linux_dmabuf_v1 v4
    /// feedback can stat it and hand Mesa the `main_device` dev_t. None
    /// in headless mode; dmabuf clients won't get useful feedback.
    pub drm_device_path: Option<PathBuf>,
    /// Wayland socket name (e.g. `wayland-1`). Stashed so built-in
    /// keybinds that spawn clients (Alt+Enter → alacritty) can set
    /// `WAYLAND_DISPLAY` on the child process.
    pub socket_name: String,
    /// Modifier-key tracking for config-driven keybinds. libinput hands
    /// us raw evdev keycodes; Alt is scancode 56 (left) or 100 (right),
    /// Ctrl is 29 / 97, Shift 42 / 54, Super 125 / 126.
    pub left_alt_down: bool,
    pub right_alt_down: bool,
    pub left_ctrl_down: bool,
    pub right_ctrl_down: bool,
    pub left_shift_down: bool,
    pub right_shift_down: bool,
    pub left_super_down: bool,
    pub right_super_down: bool,
    /// User config loaded once at startup. Drives which evdev
    /// keycodes get remapped (e.g. CapsLock→Escape) and which
    /// (mod, key) combos dispatch `exec` actions instead of being
    /// forwarded to the focused client.
    pub config: config::Config,
    /// If Some, the toplevel-like tile currently zoomed to the full
    /// usable area (outer gaps respected). While set, other mapped
    /// toplevels don't render. Cleared by a second toggle, by any
    /// focus or move direction command, and automatically if the
    /// zoomed surface dies.
    pub zoomed: Option<Weak<WlSurface>>,
    /// Keys of texture cache entries the DRM presenter should evict on
    /// the next render tick. Filled by `Dispatch<WlBuffer, _>::Destroy`
    /// handlers when a client tears down a buffer; drained (and acted
    /// on) inside `render_tick` which has `&mut` access to the backend.
    pub pending_texture_evictions: Vec<u64>,
    /// Weak refs to every `wl_surface` with a `zwlr_layer_surface_v1`
    /// role. Dead weaks are pruned on render tick; the role's mapped
    /// flag on `SurfaceData` determines whether we currently draw it.
    pub pending_layer_surfaces: Vec<Weak<WlSurface>>,
    /// libseat session handle. Used by VT-switch keybinds and by the
    /// backend's Enable/Disable transition logic. None in headless.
    pub session: Option<session::SharedSession>,
    /// Rolling count of page-flip events since the last FPS log. Reset
    /// every second in the DRM source handler; logged at INFO so
    /// users can sanity-check their refresh rate without external
    /// tools.
    pub flip_counter: u32,
    pub flip_counter_reset: Option<Instant>,
    /// `wp_presentation.feedback` objects clients have handed us that
    /// are waiting for their next `presented` event. Fired at the
    /// next DRM page-flip-complete (at which point they're dropped).
    /// Without this protocol, Chrome refuses to pace above 60 Hz.
    pub pending_presentation: Vec<
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::WpPresentationFeedback,
    >,
    /// Monotonically incrementing counter used as the MSC field in
    /// `presented`. Clients use it to detect missed vblanks.
    pub presentation_seq: u64,
    /// Currently-visible workspace. 1-based. Only toplevels whose
    /// `Window.workspace` matches this render, receive input, and
    /// participate in focus navigation. Layer surfaces (bars,
    /// wallpapers) and the cursor are global, i.e. not gated by
    /// this.
    pub active_ws: u32,
    /// In-flight workspace-switch animation state. While `Some`, the
    /// outgoing workspace is rendered on top of the incoming one with
    /// per-window alpha computed from the elapsed fraction of
    /// `config.workspace_animation_duration_ms`. When the animation
    /// finishes `tick_workspace_animation` clears this back to `None`
    /// and only the active workspace renders again.
    pub workspace_anim: Option<WorkspaceAnim>,
    /// All `wl_output` resources that clients have bound. Tracked so
    /// the `ext_workspace_v1` manager can emit `output_enter` on the
    /// workspace group for each client's output.
    pub outputs: Vec<WlOutput>,
    /// Per-binding state for `ext_workspace_v1` — the workspace
    /// protocol that taskbars use to show + activate our
    /// workspaces. Each binding owns a server-created group + N
    /// workspace handles. `switch_workspace` walks this list on
    /// every switch to emit state changes atomically.
    pub ext_workspace_managers: Vec<ext_workspace::ManagerBinding>,
    /// Live `zwp_idle_inhibitor_v1` resources. Non-empty → video
    /// players / presentation tools have asked us not to idle-blank.
    /// The idle timer consults this before DPMS-off. Cleaned up
    /// in the inhibitor's `Destroy` dispatch.
    pub idle_inhibitors: Vec<ZwpIdleInhibitorV1>,
    /// Monotonic timestamp of the most recent user input. Updated
    /// on every `apply_input` call; the idle timer compares it to
    /// `now - config.idle_timeout` to decide whether to blank. Also
    /// re-set to `now` whenever the screen wakes, so a wake doesn't
    /// immediately re-arm another blank countdown.
    pub last_input_at: Instant,
    /// True while the monitor is DPMS-off. Lets apply_input cheaply
    /// detect "first input since blank" and re-power the display.
    pub idle: bool,
    /// Set by `State`-only callers (e.g. idle-inhibit creation) that
    /// want the screen to wake but don't have access to the backend.
    /// The main loop reads+clears this after each client dispatch
    /// and flips DPMS back on. Input-driven wake doesn't need this
    /// path — the input source has direct backend access.
    pub pending_wake: bool,
    /// Process-group IDs of `exec-once` children. Each is spawned
    /// in its own group (via `setsid` in a pre_exec hook) so we
    /// can `killpg` the whole subtree on shutdown — otherwise
    /// wrapper-shell loops like `wp-cycle.sh` outlive sol,
    /// accumulate across restarts, and each orphan keeps firing
    /// its own 30-second wallpaper-cycle tick. After ~7 restarts
    /// that shows up as "transitions every few seconds."
    pub exec_once_pgids: Vec<libc::pid_t>,
}

/// Software cursor: a fixed-size ARGB sprite whose top-left in screen space
/// is always `(pos_x - hot_x, pos_y - hot_y)`. Rendered as the topmost scene
/// element when visible.
pub struct Cursor {
    pub pos_x: f64,
    pub pos_y: f64,
    pub visible: bool,
    pub pixels: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub hot_x: i32,
    pub hot_y: i32,
}

impl Cursor {
    pub fn new(centre_x: f64, centre_y: f64) -> Self {
        let sprite = cursor::load();
        Self {
            pos_x: centre_x,
            pos_y: centre_y,
            visible: true,
            pixels: sprite.pixels,
            width: sprite.width,
            height: sprite.height,
            hot_x: sprite.hot_x,
            hot_y: sprite.hot_y,
        }
    }
}

#[derive(Debug)]
pub struct Globals {
    pub compositor: GlobalId,
    pub shm: GlobalId,
    pub output: GlobalId,
    pub seat: GlobalId,
    pub xdg_wm_base: GlobalId,
    pub linux_dmabuf: GlobalId,
    pub xdg_decoration: GlobalId,
    pub layer_shell: GlobalId,
    pub subcompositor: GlobalId,
    pub data_device_manager: GlobalId,
    pub xdg_output_manager: GlobalId,
    pub presentation: GlobalId,
    pub viewporter: GlobalId,
    pub fractional_scale: GlobalId,
    pub ext_workspace: GlobalId,
    pub idle_inhibit: GlobalId,
}

impl State {
    pub fn elapsed_ms(&self) -> u32 {
        self.started.elapsed().as_millis() as u32
    }
    pub fn next_serial(&mut self) -> u32 {
        self.next_serial = self.next_serial.wrapping_add(1);
        self.next_serial
    }

    /// Grab keyboard focus on first map of a toplevel — the window
    /// the user just spawned (or that just popped up via Alt+Enter
    /// while another terminal had focus) is what they want to type
    /// into next, not the previous one.
    ///
    /// Exception: an Exclusive-keyboard layer surface (rofi while
    /// it's open, a lockscreen) keeps focus. Stealing focus from
    /// rofi while it's actively waiting for the user to pick a
    /// command would defeat the point of running rofi.
    pub fn on_toplevel_mapped(&mut self, surface: &WlSurface) {
        use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::Layer;
        use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_surface_v1::KeyboardInteractivity;
        let screen = Rect {
            x: 0,
            y: 0,
            w: self.screen_width as i32,
            h: self.screen_height as i32,
        };
        let layers = layer_shell::mapped_layers(self, screen);
        let exclusive_layer_focused = layers.iter().any(|m| {
            matches!(m.layer, Layer::Top | Layer::Overlay)
                && m.keyboard_interactivity
                    == KeyboardInteractivity::Exclusive as u32
        });
        if exclusive_layer_focused {
            return;
        }
        self.set_keyboard_focus(Some(surface.clone()));
    }

    pub fn set_keyboard_focus(&mut self, new: Option<WlSurface>) {
        if surface_eq(self.keyboard_focus.as_ref(), new.as_ref()) {
            return;
        }
        if let Some(old) = self.keyboard_focus.take() {
            if old.is_alive() {
                let serial = self.next_serial();
                for kb in &self.keyboards {
                    if same_client(kb, &old) {
                        kb.leave(serial, &old);
                    }
                }
            }
        }
        if let Some(new) = new.as_ref() {
            let serial = self.next_serial();
            for kb in &self.keyboards {
                if same_client(kb, new) {
                    kb.enter(serial, new, Vec::new());
                    // Send current modifier state so the client starts with
                    // a correct shift/ctrl/etc view of the world.
                    if let Some(km) = self.keymap.as_ref() {
                        use xkbcommon::xkb as x;
                        let serial = self.next_serial_const();
                        kb.modifiers(
                            serial,
                            km.state.serialize_mods(x::STATE_MODS_DEPRESSED),
                            km.state.serialize_mods(x::STATE_MODS_LATCHED),
                            km.state.serialize_mods(x::STATE_MODS_LOCKED),
                            km.state.serialize_layout(x::STATE_LAYOUT_EFFECTIVE),
                        );
                    }
                }
            }
        }
        self.keyboard_focus = new;
    }

    // Cheeky: hand back the current `next_serial` without mutating. Only
    // used inside `set_keyboard_focus` where we just bumped it above and
    // need a second matching value for the modifiers event.
    fn next_serial_const(&self) -> u32 {
        self.next_serial
    }
}

fn same_client<A: Resource, B: Resource>(a: &A, b: &B) -> bool {
    match (a.client(), b.client()) {
        (Some(ca), Some(cb)) => ca.id() == cb.id(),
        _ => false,
    }
}

fn surface_eq(a: Option<&WlSurface>, b: Option<&WlSurface>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (None, None) => true,
        _ => false,
    }
}

#[derive(Default)]
pub struct ClientState;

impl ClientData for ClientState {
    fn initialized(&self, client_id: ClientId) {
        tracing::info!(?client_id, "client connected");
    }
    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        tracing::info!(?client_id, ?reason, "client disconnected");
    }
}

/// Backend-specific render target. Chosen at startup; does not switch at
/// runtime.
pub enum BackendState {
    Headless { canvas: Canvas, png_path: PathBuf },
    Drm(DrmPresenter),
}

impl BackendState {
    /// Drive the display's DPMS state, if the backend supports it.
    /// Headless is a no-op — the idle-blank path still updates the
    /// internal `idle` flag so tests can exercise it, but there's
    /// no monitor to power down.
    pub fn set_dpms(&mut self, blank: bool) {
        if let BackendState::Drm(p) = self {
            if let Err(e) = p.set_dpms(blank) {
                tracing::warn!(error = %e, "set_dpms failed");
            }
        }
    }
}

/// Pairs Display + State + backend so calloop callbacks can reach everything.
pub struct Compositor {
    pub state: State,
    pub display: Display<State>,
    pub backend: BackendState,
}

/// If keyboard focus is missing or points at a dead surface (e.g. the
/// focused window was just closed), move focus to the topmost surviving
/// mapped toplevel. Keeps typing working without requiring the user to
/// click a new window to reclaim focus.
///
/// Also: if any mapped Top/Overlay layer surface sets
/// `keyboard_interactivity = exclusive` (e.g. rofi while it's visible),
/// hand focus to *that* instead, regardless of what was previously
/// focused. This matches sway/wlroots behavior and is what makes
/// launchers and lockscreens capture input reliably.
fn rebalance_keyboard_focus(state: &mut State) {
    use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::Layer;
    use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_surface_v1::KeyboardInteractivity;

    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };
    // Check if a Top/Overlay layer surface is currently demanding
    // exclusive keyboard focus.
    let layers = layer_shell::mapped_layers(state, screen);
    let exclusive_layer = layers.iter().rev().find(|m| {
        matches!(m.layer, Layer::Top | Layer::Overlay)
            && m.keyboard_interactivity == KeyboardInteractivity::Exclusive as u32
    });
    if let Some(ml) = exclusive_layer {
        // Steal focus unconditionally; exclusive means exclusive.
        let target = Some(ml.surface.clone());
        if !surface_eq(state.keyboard_focus.as_ref(), target.as_ref()) {
            state.set_keyboard_focus(target);
        }
        return;
    }

    let focus_alive = state
        .keyboard_focus
        .as_ref()
        .map(|s| s.is_alive())
        .unwrap_or(false);
    if focus_alive {
        return;
    }
    let active_ws = state.active_ws;
    let next = state
        .mapped_toplevels
        .iter()
        .rev()
        .filter(|w| w.workspace == active_ws)
        .find_map(|w| w.surface.upgrade().ok());
    state.set_keyboard_focus(next);
}

/// Install an inotify-based watcher on the user's `sol.conf` so edits
/// take effect without restarting. Watches the parent directory (so
/// rename-on-save editors don't escape the watch) for `CLOSE_WRITE`
/// and `MOVED_TO`, filtered by the basename of the resolved config
/// path. On a match, dispatches `apply_config_reload`.
///
/// Failure to install (no parent dir, inotify_init failed, etc.) is
/// logged but non-fatal — sol still runs, just without live reload.
fn install_config_watcher(
    event_loop: &EventLoop<'static, Compositor>,
) -> Result<()> {
    use std::os::fd::AsFd;

    use rustix::fs::inotify::{self, CreateFlags, ReadFlags, WatchFlags};

    let cfg_path = config::config_path();
    let parent = cfg_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("config path has no parent dir"))?
        .to_path_buf();
    let basename = cfg_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("config path has no file name"))?
        .to_owned();

    // Make sure the dir exists before we try to watch it. A user
    // running sol without a config dir at all is fine — inotify
    // returns ENOENT, we log and disable live reload.
    if !parent.exists() {
        anyhow::bail!("config dir {} does not exist", parent.display());
    }

    let inotify_fd = inotify::init(CreateFlags::CLOEXEC | CreateFlags::NONBLOCK)
        .context("inotify_init")?;
    inotify::add_watch(
        &inotify_fd,
        &parent,
        WatchFlags::CLOSE_WRITE | WatchFlags::MOVED_TO,
    )
    .with_context(|| format!("inotify_add_watch {}", parent.display()))?;

    tracing::info!(
        path = %cfg_path.display(),
        "config: watching for changes (live reload)"
    );

    event_loop
        .handle()
        .insert_source(
            Generic::new(inotify_fd, Interest::READ, Mode::Level),
            move |_ready, fd, comp| {
                use std::mem::MaybeUninit;
                let mut buf = [MaybeUninit::<u8>::uninit(); 4096];
                let mut reader = inotify::Reader::new(fd.as_fd(), &mut buf);
                let mut should_reload = false;
                loop {
                    match reader.next() {
                        Ok(ev) => {
                            // Filter by basename: directory watch fires
                            // for every file in there (waybar.json,
                            // wp-cycle.sh) — only the conf matters.
                            let matches = ev
                                .file_name()
                                .map(|n| n.to_bytes() == basename.as_encoded_bytes())
                                .unwrap_or(false);
                            if matches
                                && ev.events().intersects(
                                    ReadFlags::CLOSE_WRITE | ReadFlags::MOVED_TO,
                                )
                            {
                                should_reload = true;
                            }
                        }
                        Err(e)
                            if e.raw_os_error()
                                == rustix::io::Errno::AGAIN.raw_os_error() =>
                        {
                            // Drained the buffer; back to the event loop.
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "inotify read failed");
                            break;
                        }
                    }
                }
                if should_reload {
                    apply_config_reload(&mut comp.state);
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert inotify source: {e}"))?;

    Ok(())
}

/// Re-read `sol.conf` and replace `state.config` in place.
///
/// Settings that take effect immediately (just by virtue of being read
/// from `state.config` on the next render / next idle tick / next
/// keypress): `gaps_in`, `gaps_out`, `border_width`, `border_color`,
/// `idle_timeout`, `bindings`, `remaps`.
///
/// Deliberately NOT reapplied:
/// - `exec-once` — would spawn a duplicate waybar / awww-daemon every
///   save. Startup-only.
/// - `mode` — changing the output mode requires tearing down and
///   rebuilding the GBM/EGL surface to the new size; that work isn't
///   wired yet. We log a warning so the user knows their edit won't
///   take effect until restart.
fn apply_config_reload(state: &mut State) {
    let new_cfg = config::load();

    if new_cfg.mode != state.config.mode {
        match new_cfg.mode {
            Some(m) => tracing::warn!(
                width = m.width,
                height = m.height,
                refresh_hz = m.refresh_hz,
                "config: `mode` changed; restart sol to apply (live mode-set not yet implemented)"
            ),
            None => tracing::warn!(
                "config: `mode` cleared; restart sol to revert to the default mode-pick heuristic"
            ),
        }
    }

    let kb_repeat_changed = new_cfg.keyboard_repeat_rate
        != state.config.keyboard_repeat_rate
        || new_cfg.keyboard_repeat_delay != state.config.keyboard_repeat_delay;

    tracing::info!(
        bindings = new_cfg.bindings.len(),
        remaps = new_cfg.remaps.len(),
        gaps_in = new_cfg.gaps_in,
        gaps_out = new_cfg.gaps_out,
        border_width = new_cfg.border_width,
        idle_timeout = new_cfg.idle_timeout,
        keyboard_repeat_rate = new_cfg.keyboard_repeat_rate,
        keyboard_repeat_delay = new_cfg.keyboard_repeat_delay,
        "config reloaded"
    );
    state.config = new_cfg;

    // wl_keyboard.repeat_info is allowed to be re-sent at any time, so
    // push the new rate/delay to every already-bound keyboard. Dead
    // keyboards are skipped via is_alive. Clients pick the new values
    // up on their next keypress — no event loop dance needed.
    if kb_repeat_changed {
        for kb in state.keyboards.iter() {
            if kb.version() >= 4 && kb.is_alive() {
                kb.repeat_info(
                    state.config.keyboard_repeat_rate,
                    state.config.keyboard_repeat_delay,
                );
            }
        }
    }

    // Force a fresh layout pass so gap / border tweaks animate in.
    state.needs_render = true;
}

/// Master-stack layout. First window takes the left half (or full screen if
/// it's the only one); remaining windows split the right half evenly, top to
/// bottom in mapping order — so the most recently mapped toplevel is the
/// bottom of the stack. Pure function over window count + screen rect.
fn master_stack_layout(n: usize, screen: Rect) -> Vec<Rect> {
    match n {
        0 => Vec::new(),
        1 => vec![screen],
        _ => {
            let mid = screen.w / 2;
            let master = Rect {
                x: screen.x,
                y: screen.y,
                w: mid,
                h: screen.h,
            };
            let stack_w = screen.w - mid;
            let stack_x = screen.x + mid;
            let stack_n = (n - 1) as i32;
            let mut out = Vec::with_capacity(n);
            out.push(master);
            for i in 0..(n - 1) as i32 {
                let y0 = screen.y + (screen.h * i) / stack_n;
                let y1 = screen.y + (screen.h * (i + 1)) / stack_n;
                out.push(Rect {
                    x: stack_x,
                    y: y0,
                    w: stack_w,
                    h: y1 - y0,
                });
            }
            out
        }
    }
}

/// Assign a screen-space rect to each mapped toplevel using master-stack,
/// but compute the layout against the usable area — the screen minus any
/// edge-reserved exclusive zones from layer surfaces (a top-anchored bar
/// with exclusive_zone=30 shrinks tiled toplevels by 30 px off the top).
/// Gaps come from the user config: `gaps_out` shrinks the usable area
/// (so every tile sits at least that many pixels from the edge) and
/// `gaps_in` is split half-each-side so adjacent tiles are exactly
/// `gaps_in` apart.
fn apply_layout(state: &mut State, now: Instant) {
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };
    let layers = layer_shell::mapped_layers(state, screen);
    let usable = layer_shell::usable_area(&layers, screen);
    let inner = shrink_rect(usable, state.config.gaps_out);

    let active_ws = state.active_ws;

    // Zoom overrides the tile layout: the zoomed window gets the
    // full inner area and we short-circuit before running
    // master-stack. Other windows keep their last-known rects
    // (collect_scene will skip them anyway while zoom is active).
    // If the zoomed surface isn't on the active workspace (e.g. we
    // just switched away), treat zoom as cleared — zoom doesn't
    // follow the user across workspaces.
    if let Some(zs) = state.zoomed.as_ref().and_then(|w| w.upgrade().ok()) {
        let zoomed_exists = state
            .mapped_toplevels
            .iter_mut()
            .find(|w| {
                w.workspace == active_ws
                    && w.surface.upgrade().ok().as_ref() == Some(&zs)
            })
            .map(|w| set_target_rect(w, inner, now))
            .is_some();
        if zoomed_exists {
            return;
        }
        // Zoomed surface vanished or isn't on this workspace. Drop
        // the flag and fall through to normal layout.
        state.zoomed = None;
    }

    let n = state
        .mapped_toplevels
        .iter()
        .filter(|w| w.workspace == active_ws)
        .count();
    let rects = master_stack_layout(n, inner);
    let gaps_in = state.config.gaps_in;
    let mut rect_iter = rects.into_iter();
    for win in state.mapped_toplevels.iter_mut() {
        if win.workspace != active_ws {
            continue;
        }
        let Some(rect) = rect_iter.next() else { break };
        let new_rect = shrink_interior_edges(rect, inner, gaps_in);
        set_target_rect(win, new_rect, now);
    }
}

/// Update a window's target `rect` and, if it actually changed since
/// the last call, kick off a fresh tween: snapshot `from_rect` from
/// the current `render_rect` (so a re-trigger mid-animation feels
/// natural — it eases from wherever the window is right now, not
/// from the previous static target) and stamp `anim_started_at` to
/// drive `tick_animations`. No-op when the rect is unchanged so we
/// don't restart tweens every render tick.
fn set_target_rect(win: &mut Window, new_rect: Rect, now: Instant) {
    if win.rect == new_rect {
        return;
    }
    // Capture the live render_rect (in float) as the tween origin so
    // a re-trigger mid-animation eases from where the window is right
    // now, not from the previous static target.
    win.from_rect = win.render_rect;
    win.anim_started_at = Some(now);
    win.rect = new_rect;
}

fn shrink_rect(r: Rect, by: i32) -> Rect {
    Rect {
        x: r.x + by,
        y: r.y + by,
        w: (r.w - 2 * by).max(1),
        h: (r.h - 2 * by).max(1),
    }
}

/// Shrink each edge of `r` by half of `gaps_in` iff that edge is an
/// interior boundary (i.e. not sitting on the corresponding edge of
/// `outer`). An edge that is on the outer boundary is already
/// `gaps_out` away from the actual screen edge, so we leave it alone —
/// the result is: tile ↔ tile gap = `gaps_in`, tile ↔ edge gap =
/// `gaps_out`. Clean semantics, no compound.
fn shrink_interior_edges(r: Rect, outer: Rect, gaps_in: i32) -> Rect {
    let half = gaps_in / 2;
    let left = if r.x == outer.x { 0 } else { half };
    let right = if r.x + r.w == outer.x + outer.w { 0 } else { half };
    let top = if r.y == outer.y { 0 } else { half };
    let bottom = if r.y + r.h == outer.y + outer.h { 0 } else { half };
    Rect {
        x: r.x + left,
        y: r.y + top,
        w: (r.w - left - right).max(1),
        h: (r.h - top - bottom).max(1),
    }
}

/// For each window whose assigned rect differs from the size we most
/// recently told the client, send a fresh xdg_toplevel.configure +
/// xdg_surface.configure. Cached via `Window.pending_size` so steady-state
/// render ticks don't re-send configures.
///
/// States list comes from `xdg_shell::tile_state_bytes` (MAXIMIZED +
/// TILED_* all edges + ACTIVATED). MAXIMIZED is what forces clients to
/// obey the configured size; TILED_* is what tiling-aware clients use to
/// avoid drawing resize chrome.
fn send_pending_configures(state: &mut State) {
    let state_bytes = xdg_shell::tile_state_bytes();

    let mut todo: Vec<(usize, i32, i32)> = Vec::new();
    for (i, win) in state.mapped_toplevels.iter().enumerate() {
        let target = (win.rect.w, win.rect.h);
        if win.pending_size != Some(target) {
            todo.push((i, target.0, target.1));
        }
    }
    for (i, w, h) in todo {
        let (tl, xs) = {
            let win = &state.mapped_toplevels[i];
            let Ok(surface) = win.surface.upgrade() else { continue };
            let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
                continue;
            };
            let sd = sd_arc.lock().unwrap();
            let (Some(tl_weak), Some(xs_weak)) =
                (sd.xdg_toplevel.as_ref(), sd.xdg_surface.as_ref())
            else {
                continue;
            };
            let (Ok(tl), Ok(xs)) = (tl_weak.upgrade(), xs_weak.upgrade()) else {
                continue;
            };
            (tl, xs)
        };
        let serial = state.next_serial();
        tl.configure(w, h, state_bytes.clone());
        xs.configure(serial);
        state.mapped_toplevels[i].pending_size = Some((w, h));
        tracing::debug!(
            tile_index = i,
            width = w,
            height = h,
            serial,
            "sent xdg_toplevel.configure (tiled+maximized)"
        );
    }
}

/// Collects scene elements in back-to-front render order:
/// background + bottom layer surfaces → tiled xdg toplevels →
/// top + overlay layer surfaces. Cursor is added separately by
/// `scene_from_buffers`. Each entry pairs a `WlBuffer` with the rect
/// it should be drawn at; callers keep the buffers alive for the
/// duration of rendering.
/// Entry passed from `collect_scene` to `scene_from_buffers`. Carries
/// the buffer to draw, the on-screen rect to draw it in (compositor's
/// chosen position and size), and an optional `wp_viewport.set_source`
/// crop rect in buffer coordinates (x, y, w, h). When the source rect
/// is Some, the presenter samples only that sub-rect of the buffer;
/// otherwise the full texture is used.
/// What `collect_scene` hands to `scene_from_buffers`. Two flavours:
///
/// - `Buffer` — a textured surface. Carries the `wl_buffer` to sample
///   from, the on-screen rect to draw it in, an optional
///   `wp_viewport.set_source` crop in buffer coords, and an alpha
///   multiplier in `[0, 1]`.
/// - `Backdrop` — a frosted-glass element. No client buffer; the
///   presenter samples its own blur FBO at the screen rect. Carries
///   the rect, the alpha multiplier (used to fade the backdrop
///   in/out during workspace crossfade), and the blur-pass count
///   the user asked for.
///
/// Order in the `Vec<Placed>` is back-to-front; the presenter
/// renders straight through it.
enum Placed {
    Buffer {
        buf: wayland_server::protocol::wl_buffer::WlBuffer,
        rect: RectF,
        vsrc: Option<(f64, f64, f64, f64)>,
        alpha: f32,
    },
    Backdrop {
        rect: RectF,
        alpha: f32,
        passes: u32,
    },
}

fn collect_scene(state: &State, now: Instant) -> (Vec<Placed>, usize) {
    let mut out: Vec<Placed> = Vec::new();
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };

    use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::Layer;
    let layers = layer_shell::mapped_layers(state, screen);

    // 1. Background + Bottom layer surfaces. Layer-shell surfaces
    // are global (not per-workspace), so they always draw at full
    // alpha — workspace-switch crossfade only affects toplevels.
    for ml in &layers {
        if matches!(ml.layer, Layer::Background | Layer::Bottom) {
            let vsrc = surface_viewport_src(&ml.surface);
            let r: RectF = ml.rect.into();
            out.push(Placed::Buffer {
                buf: ml.buffer.clone(),
                rect: r,
                vsrc,
                alpha: 1.0,
            });
            emit_subsurface_tree(&mut out, &ml.surface, r.x, r.y, 1.0);
        }
    }
    // Mark the boundary: everything pushed so far (bg + bottom layers
    // and their subsurfaces) is the "background" the blur pipeline
    // captures and blurs. Toplevels and top-layer surfaces draw on
    // top of that, on the default framebuffer.
    let background_count = out.len();

    // 2. Tiled xdg_toplevels.
    //
    // While zoom is active, only the zoomed surface renders. During a
    // workspace-switch crossfade, both the outgoing (`from_ws`) and
    // incoming (`to_ws == active_ws`) workspaces render together
    // with descending / ascending alpha; the outgoing draws *after*
    // the incoming in the same back-to-front pass so it's painted
    // on top — that way as it fades it reveals the incoming
    // workspace beneath rather than the wallpaper.
    //
    // Position uses `render_rect`, not `rect`: `tick_animations` is
    // interpolating it toward `rect`, and the GPU scales the buffer
    // to whatever intermediate size we ask for — that's what makes
    // the layout transition read as motion rather than a snap.
    let zoomed = state.zoomed.as_ref().and_then(|w| w.upgrade().ok());
    let crossfade = workspace_anim_alphas(state, now);
    let active_ws = state.active_ws;
    let focused = state.keyboard_focus.clone();
    let inactive_alpha = state.config.inactive_alpha;
    let inactive_blur = state.config.inactive_blur && state.config.inactive_alpha < 1.0;
    let inactive_blur_passes = state.config.inactive_blur_passes;

    let emit_for_ws = |out: &mut Vec<Placed>, ws: u32, ws_alpha: f32| {
        for win in state.mapped_toplevels.iter() {
            if win.workspace != ws {
                continue;
            }
            let Ok(surface) = win.surface.upgrade() else { continue };
            if let Some(zs) = &zoomed {
                if surface != *zs {
                    continue;
                }
            }
            let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
                continue;
            };
            let (buf_opt, vsrc, vdst) = {
                let sd = sd_arc.lock().unwrap();
                if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
                    continue;
                }
                (
                    sd.current.buffer.clone(),
                    sd.viewport_src,
                    sd.viewport_dst,
                )
            };
            let _ = vdst;
            // Active = the surface owning keyboard focus. Active windows
            // render at full opacity (× workspace alpha); inactive get
            // multiplied by config.inactive_alpha and optionally
            // preceded by a frosted-glass backdrop. Zoomed surfaces
            // are treated as active for alpha purposes — zoom is the
            // "spotlight" gesture, looks weird if the zoomed window
            // is dimmed.
            let is_active = focused
                .as_ref()
                .map(|f| *f == surface)
                .unwrap_or(false)
                || zoomed.is_some();
            let win_alpha = ws_alpha
                * if is_active { 1.0 } else { inactive_alpha };

            if !is_active && inactive_blur {
                // Backdrop emitted with ws_alpha (not win_alpha) so
                // it covers the wallpaper fully outside crossfade,
                // and fades together with its window during one.
                // Subsurface contents don't get a separate backdrop —
                // only the toplevel rect itself.
                out.push(Placed::Backdrop {
                    rect: win.render_rect,
                    alpha: ws_alpha,
                    passes: inactive_blur_passes,
                });
            }

            if let Some(buf) = buf_opt {
                out.push(Placed::Buffer {
                    buf,
                    rect: win.render_rect,
                    vsrc,
                    alpha: win_alpha,
                });
            }
            emit_subsurface_tree(
                out,
                &surface,
                win.render_rect.x,
                win.render_rect.y,
                win_alpha,
            );
        }
    };

    if let Some((from_alpha, to_alpha)) = crossfade {
        let from_ws = state.workspace_anim.map(|a| a.from_ws).unwrap_or(active_ws);
        // Incoming workspace first (drawn underneath); outgoing on
        // top so it covers the incoming until its alpha falls.
        emit_for_ws(&mut out, active_ws, to_alpha);
        emit_for_ws(&mut out, from_ws, from_alpha);
    } else {
        emit_for_ws(&mut out, active_ws, 1.0);
    }

    // 3. Top + Overlay layer surfaces.
    for ml in &layers {
        if matches!(ml.layer, Layer::Top | Layer::Overlay) {
            let vsrc = surface_viewport_src(&ml.surface);
            let r: RectF = ml.rect.into();
            out.push(Placed::Buffer {
                buf: ml.buffer.clone(),
                rect: r,
                vsrc,
                alpha: 1.0,
            });
            emit_subsurface_tree(&mut out, &ml.surface, r.x, r.y, 1.0);
        }
    }
    (out, background_count)
}

fn surface_viewport_src(s: &WlSurface) -> Option<(f64, f64, f64, f64)> {
    s.data::<Arc<Mutex<SurfaceData>>>()?
        .lock()
        .ok()?
        .viewport_src
}

/// Walk a surface's subsurface_children recursively, pushing each mapped
/// child's current buffer at `parent_origin + child.subsurface_offset`.
/// Subsurfaces without a buffer attached yet are skipped but their own
/// children still descended into — the buffer may land a frame later.
/// Stacking follows registration order in `subsurface_children`;
/// `place_above` / `place_below` aren't implemented.
fn emit_subsurface_tree(
    out: &mut Vec<Placed>,
    parent: &WlSurface,
    parent_x: f32,
    parent_y: f32,
    alpha: f32,
) {
    let Some(sd_arc) = parent.data::<Arc<Mutex<SurfaceData>>>() else {
        return;
    };
    // Snapshot the child list so we don't hold the parent's lock while
    // we recurse into child locks (avoids any risk of the same surface
    // being double-locked if the topology is ever cyclic).
    let children: Vec<WlSurface> = {
        let sd = sd_arc.lock().unwrap();
        sd.subsurface_children
            .iter()
            .filter_map(|w| w.upgrade().ok())
            .collect()
    };
    for child in children {
        let Some(child_sd_arc) = child.data::<Arc<Mutex<SurfaceData>>>() else {
            continue;
        };
        let (buf_opt, child_x, child_y, vsrc, logical) = {
            let sd = child_sd_arc.lock().unwrap();
            let (ox, oy) = sd.subsurface_offset;
            let buf = sd.current.buffer.clone();
            (
                buf,
                parent_x + ox as f32,
                parent_y + oy as f32,
                sd.viewport_src,
                surface_logical_size(&sd),
            )
        };
        if let Some(buf) = buf_opt {
            // Output rect uses the surface's *logical* size, not the
            // buffer's intrinsic dims. Chrome's URL-bar autocomplete
            // popup, when the list shrinks, keeps the big buffer
            // allocated and crops with `wp_viewport.set_source`
            // instead of reallocating; logical size = source size.
            // Falling back to buffer dims here would stretch the
            // (smaller) cropped region into the (larger) buffer's
            // quad on every commit where the popup shrinks.
            //
            // `surface_logical_size` collapses destination → source
            // → buffer in the order the protocol layers them, so
            // this also covers `set_destination` and the legacy
            // no-viewport path.
            let (w, h) = logical.unwrap_or((0, 0));
            out.push(Placed::Buffer {
                buf,
                rect: RectF {
                    x: child_x,
                    y: child_y,
                    w: w as f32,
                    h: h as f32,
                },
                vsrc,
                alpha,
            });
        }
        emit_subsurface_tree(out, &child, child_x, child_y, alpha);
    }
}

fn scene_from_buffers<'a>(
    placed: &'a [Placed],
    background_count: usize,
    cursor: &'a Cursor,
) -> Scene<'a> {
    let mut scene = Scene::new();
    scene.background_count = background_count;
    for p in placed {
        let (buf, rect, vsrc, alpha) = match p {
            Placed::Buffer { buf, rect, vsrc, alpha } => (buf, rect, vsrc, alpha),
            Placed::Backdrop { rect, alpha, passes } => {
                // Frosted backdrop: no client buffer, no UV crop.
                // Use a sentinel buffer_key so the presenter's
                // texture map never matches; the BlurredBackdrop
                // arm in the draw loop short-circuits before any
                // texture lookup happens anyway.
                scene.elements.push(SceneElement {
                    buffer_key: BACKDROP_SCENE_KEY,
                    width: 0,
                    height: 0,
                    dst_width: rect.w,
                    dst_height: rect.h,
                    x: rect.x,
                    y: rect.y,
                    uv_x: 0.0,
                    uv_y: 0.0,
                    uv_w: 1.0,
                    uv_h: 1.0,
                    alpha: *alpha,
                    content: SceneContent::BlurredBackdrop { passes: *passes },
                });
                continue;
            }
        };
        if let Some(bd) = buf.data::<shm::BufferData>() {
            let Some(bytes) = bd.bytes() else { continue };
            let Some(format) = bd.pixel_format() else { continue };
            let key = (bd as *const shm::BufferData) as usize as u64;
            let (ux, uy, uw, uh) = compute_uv(*vsrc, bd.width, bd.height);
            scene.elements.push(SceneElement {
                buffer_key: key,
                width: bd.width,
                height: bd.height,
                dst_width: rect.w,
                dst_height: rect.h,
                x: rect.x,
                y: rect.y,
                uv_x: ux,
                uv_y: uy,
                uv_w: uw,
                uv_h: uh,
                alpha: *alpha,
                content: SceneContent::Shm {
                    pixels: bytes,
                    stride: bd.stride,
                    format,
                },
            });
        } else if let Some(db) = buf.data::<linux_dmabuf::DmabufBuffer>() {
            // B10.3: single-plane dmabuf only. Multi-plane (YUV etc.) lands
            // when we care about video, not for alacritty/terminals.
            let Some(p0) = db.planes.first() else { continue };
            let key = (db as *const linux_dmabuf::DmabufBuffer) as usize as u64;
            let (ux, uy, uw, uh) = compute_uv(*vsrc, db.width, db.height);
            scene.elements.push(SceneElement {
                buffer_key: key,
                width: db.width,
                height: db.height,
                dst_width: rect.w,
                dst_height: rect.h,
                x: rect.x,
                y: rect.y,
                uv_x: ux,
                uv_y: uy,
                uv_w: uw,
                uv_h: uh,
                alpha: *alpha,
                content: SceneContent::Dmabuf {
                    fd: p0.fd.as_raw_fd(),
                    fourcc: db.format,
                    modifier: p0.modifier,
                    offset: p0.offset,
                    stride: p0.stride,
                },
            });
        }
    }
    // Cursor last so it always draws on top. dst_* = 0.0 forces the
    // presenter to render at the sprite's intrinsic size.
    if cursor.visible {
        scene.elements.push(SceneElement {
            buffer_key: CURSOR_SCENE_KEY,
            width: cursor.width,
            height: cursor.height,
            dst_width: 0.0,
            dst_height: 0.0,
            x: cursor.pos_x as f32 - cursor.hot_x as f32,
            y: cursor.pos_y as f32 - cursor.hot_y as f32,
            uv_x: 0.0,
            uv_y: 0.0,
            uv_w: 1.0,
            uv_h: 1.0,
            alpha: 1.0,
            content: SceneContent::Shm {
                pixels: &cursor.pixels,
                stride: cursor.width * 4,
                format: sol_core::PixelFormat::Argb8888,
            },
        });
    }
    scene
}

/// Translate a `wp_viewport.set_source` rect (in buffer coords) plus
/// the buffer's intrinsic dimensions into normalized `[0, 1]` UVs
/// for the sampler. `None` means "sample the whole texture."
/// Clamped into range so a client that posts a slightly-out-of-bounds
/// source doesn't blow up the shader; protocol-level clamping should
/// also happen in the viewporter dispatch precommit check.
fn compute_uv(
    src: Option<(f64, f64, f64, f64)>,
    buf_w: i32,
    buf_h: i32,
) -> (f32, f32, f32, f32) {
    match src {
        Some((x, y, w, h)) if buf_w > 0 && buf_h > 0 => {
            let bw = buf_w as f64;
            let bh = buf_h as f64;
            let ux = (x / bw).clamp(0.0, 1.0) as f32;
            let uy = (y / bh).clamp(0.0, 1.0) as f32;
            let uw = (w / bw).clamp(0.0, 1.0) as f32;
            let uh = (h / bh).clamp(0.0, 1.0) as f32;
            (ux, uy, uw, uh)
        }
        _ => (0.0, 0.0, 1.0, 1.0),
    }
}

/// Stable key the DRM renderer uses to avoid re-uploading the cursor
/// sprite every frame. Anything outside the range of real BufferData
/// pointers works; pick something unlikely to collide.
const CURSOR_SCENE_KEY: u64 = 0xC0FFEE_C0FFEE;
/// Sentinel buffer_key for `BlurredBackdrop` scene elements. The
/// backdrop branch in the presenter's draw loop returns before any
/// texture lookup happens, but a value is required by the struct
/// shape — pick something else that can't collide with a real
/// BufferData / DmabufBuffer pointer.
const BACKDROP_SCENE_KEY: u64 = 0xB10B1B_DEADBEEF;

/// Apply the configured easing curve to a raw `[0, 1]` progress value.
/// Names follow the easings.net taxonomy. `CubicOut` (the default)
/// front-loads motion and slows to a settle, which reads as "snap
/// into place" without the harshness of an actual snap.
fn apply_easing(curve: config::AnimationCurve, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        config::AnimationCurve::Linear => t,
        config::AnimationCurve::CubicOut => {
            let inv = 1.0 - t;
            1.0 - inv * inv * inv
        }
        config::AnimationCurve::QuartOut => {
            let inv = 1.0 - t;
            1.0 - inv * inv * inv * inv
        }
        config::AnimationCurve::QuintOut => {
            let inv = 1.0 - t;
            1.0 - inv * inv * inv * inv * inv
        }
        config::AnimationCurve::ExpoOut => {
            // Closed-form ease-out exponential, with the t==1 case
            // pinned to exactly 1.0 (otherwise 2^-10 leaves a tiny
            // residual ~0.001 that would make the snap visible).
            if t >= 1.0 {
                1.0
            } else {
                1.0 - 2f32.powf(-10.0 * t)
            }
        }
        config::AnimationCurve::InOutCubic => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                let a = -2.0 * t + 2.0;
                1.0 - a * a * a / 2.0
            }
        }
    }
}

fn lerp_f32(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}

fn lerp_rectf(from: RectF, to: RectF, t: f32) -> RectF {
    RectF {
        x: lerp_f32(from.x, to.x, t),
        y: lerp_f32(from.y, to.y, t),
        w: lerp_f32(from.w, to.w, t).max(1.0),
        h: lerp_f32(from.h, to.h, t).max(1.0),
    }
}

/// Step every animating window's `render_rect` toward its target
/// `rect`. Returns `true` if any window is still mid-tween, so the
/// caller can keep `needs_render` set and the loop ticks at vblank
/// cadence until the animation settles.
///
/// Duration and curve come from `state.config`; saving the conf with
/// new values applies on the *next* tick — including in-flight tweens
/// (a shrunk duration just snaps any tween whose elapsed already
/// exceeds the new total). `animation_duration_ms = 0` disables
/// tweening entirely and snaps to target on the same frame the rect
/// changes.
///
/// Newly-mapped windows (render_rect zero) snap to their assigned
/// rect on first sight — there's nothing on screen to interpolate
/// from, and a "spawn from a point" animation is a polish for later.
fn tick_animations(state: &mut State, now: Instant) -> bool {
    let duration_ms = state.config.animation_duration_ms as u128;
    let curve = state.config.animation_curve;
    let mut any_active = false;
    for win in state.mapped_toplevels.iter_mut() {
        let target: RectF = win.rect.into();
        // Newly mapped: snap, no animation.
        if win.render_rect.w == 0.0 || win.render_rect.h == 0.0 {
            win.render_rect = target;
            win.from_rect = target;
            win.anim_started_at = None;
            continue;
        }
        // Tweening disabled (or already settled): keep render_rect
        // pinned to the target and never report active.
        if duration_ms == 0 {
            win.render_rect = target;
            win.anim_started_at = None;
            continue;
        }
        let Some(started) = win.anim_started_at else {
            // No tween in progress. Idempotent guard: keep
            // render_rect == rect even if some other code path
            // mutated rect without going through apply_layout.
            if win.render_rect != target {
                win.render_rect = target;
            }
            continue;
        };
        let elapsed = now.duration_since(started).as_millis();
        if elapsed >= duration_ms {
            win.render_rect = target;
            win.anim_started_at = None;
        } else {
            let t = elapsed as f32 / duration_ms as f32;
            let eased = apply_easing(curve, t);
            win.render_rect = lerp_rectf(win.from_rect, target, eased);
            any_active = true;
        }
    }
    any_active
}

/// Per-workspace alphas for the active crossfade, if any.
///
/// Returns `(from_alpha, to_alpha)` where:
/// - `from_alpha` is what windows on `state.workspace_anim.from_ws`
///   should multiply their output by (fading 1 → 0 over the duration)
/// - `to_alpha` is what windows on `state.workspace_anim.to_ws`
///   should use (fading 0 → 1)
///
/// `None` means no animation is in flight; the caller renders the
/// active workspace at full alpha as usual.
fn workspace_anim_alphas(state: &State, now: Instant) -> Option<(f32, f32)> {
    let anim = state.workspace_anim?;
    let duration = state.config.workspace_animation_duration_ms as u128;
    if duration == 0 {
        return None;
    }
    let elapsed = now.duration_since(anim.started_at).as_millis();
    if elapsed >= duration {
        return None;
    }
    let t = elapsed as f32 / duration as f32;
    let eased = apply_easing(state.config.animation_curve, t);
    Some((1.0 - eased, eased))
}

/// Mark the workspace animation as finished if the elapsed time has
/// exceeded the configured duration. Called from render_tick after
/// drawing so the next frame stops compositing the outgoing
/// workspace. Returns `true` if an animation is still in flight, so
/// the caller can keep `needs_render` high.
fn tick_workspace_animation(state: &mut State, now: Instant) -> bool {
    let Some(anim) = state.workspace_anim else {
        return false;
    };
    let duration = state.config.workspace_animation_duration_ms as u128;
    if duration == 0 {
        state.workspace_anim = None;
        return false;
    }
    if now.duration_since(anim.started_at).as_millis() >= duration {
        state.workspace_anim = None;
        false
    } else {
        true
    }
}

/// A surface's logical size per the Wayland protocol layering:
///
/// 1. `wp_viewport.set_destination` — the client explicitly declared it.
/// 2. `wp_viewport.set_source` — implied from the source-rect size.
/// 3. Buffer intrinsic size — the legacy path, still the common case.
///
/// Chrome exploits path (2) when it's cheaper to shrink the
/// displayed window: instead of reallocating a smaller buffer, it
/// keeps the big buffer and crops with `set_source`. If we compared
/// raw buffer dims to the target rect, Chrome's resize would never
/// look "caught up" and `render_rect` would get stuck at the old
/// position, which is exactly the ~1-second hang we were debugging.
fn surface_logical_size(sd: &compositor::SurfaceData) -> Option<(i32, i32)> {
    if let Some((w, h)) = sd.viewport_dst {
        return Some((w, h));
    }
    if let Some((_, _, w, h)) = sd.viewport_src {
        return Some((w as i32, h as i32));
    }
    let buf = sd.current.buffer.as_ref()?;
    surface_buffer_dims(buf)
}

impl PartialEq for Rect {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.w == other.w && self.h == other.h
    }
}
impl Eq for Rect {}

/// Pick the index (in `mapped_toplevels`) of the nearest tile to
/// `from` in the given direction. Scoring is primary-axis distance
/// plus 2× orthogonal distance, so from the master tile in a
/// master-stack layout Alt+L lands on the vertically-nearest stack
/// tile rather than bouncing between top and bottom. Returns None
/// if nothing qualifies — the caller is expected to interpret that
/// as "do nothing," i.e. no wrap-around at the edges.
fn neighbor_in_direction(
    windows: &[Window],
    from_idx: usize,
    dir: config::Direction,
    ws: u32,
) -> Option<usize> {
    use config::Direction;
    let from = windows[from_idx].rect;
    let cx = from.x + from.w / 2;
    let cy = from.y + from.h / 2;
    let mut best: Option<(usize, i32)> = None;
    for (i, w) in windows.iter().enumerate() {
        if i == from_idx {
            continue;
        }
        if w.workspace != ws {
            continue;
        }
        if w.surface.upgrade().is_err() {
            continue;
        }
        let tx = w.rect.x + w.rect.w / 2;
        let ty = w.rect.y + w.rect.h / 2;
        let dx = tx - cx;
        let dy = ty - cy;
        let in_dir = match dir {
            Direction::Left => dx < 0,
            Direction::Right => dx > 0,
            Direction::Up => dy < 0,
            Direction::Down => dy > 0,
        };
        if !in_dir {
            continue;
        }
        let score = match dir {
            Direction::Left | Direction::Right => dx.abs() + 2 * dy.abs(),
            Direction::Up | Direction::Down => dy.abs() + 2 * dx.abs(),
        };
        if best.map_or(true, |(_, s)| score < s) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

/// Find the `mapped_toplevels` index of the currently keyboard-
/// focused toplevel, or None if focus is empty / on a layer surface
/// / on a popup. Both focus and move commands short-circuit on None.
fn focused_toplevel_index(state: &State) -> Option<usize> {
    let focus = state.keyboard_focus.as_ref()?;
    state
        .mapped_toplevels
        .iter()
        .position(|w| w.surface.upgrade().ok().as_ref() == Some(focus))
}

fn focus_direction(state: &mut State, dir: config::Direction) {
    let Some(idx) = focused_toplevel_index(state) else { return };
    let Some(n) =
        neighbor_in_direction(&state.mapped_toplevels, idx, dir, state.active_ws)
    else {
        return;
    };
    let Ok(target) = state.mapped_toplevels[n].surface.upgrade() else {
        return;
    };
    // Moving focus while zoomed would leave the user looking at a
    // hidden tile, so drop zoom before shifting focus. Explicit
    // Alt+Tab is the only path to re-zoom on the new window.
    state.zoomed = None;
    state.set_keyboard_focus(Some(target));
    // Border follows focus, so trigger a redraw.
    state.needs_render = true;
}

/// Swap the focused tile's position in `mapped_toplevels` with its
/// neighbor in the given direction. `apply_layout` on the next
/// render tick recomputes every window's rect from its new slot and
/// `send_pending_configures` fires fresh configures so the clients
/// resize into their new tiles.
fn move_direction(state: &mut State, dir: config::Direction) {
    let Some(idx) = focused_toplevel_index(state) else { return };
    let Some(n) =
        neighbor_in_direction(&state.mapped_toplevels, idx, dir, state.active_ws)
    else {
        return;
    };
    // Same reasoning as focus_direction: rearranging the tile layout
    // while zoom hides everything but one tile would be confusing.
    state.zoomed = None;
    state.mapped_toplevels.swap(idx, n);
    state.needs_render = true;
}

/// Ctrl+Q handler: send `xdg_toplevel.close` to whichever tile has
/// keyboard focus. The client decides how to handle it — terminals
/// and most CLI-ish apps exit immediately, GUI apps may show a
/// save-changes prompt. If the client chooses to ignore the event,
/// we don't escalate to SIGTERM; that's a different action.
fn close_focused_window(state: &mut State) {
    let Some(focus) = state.keyboard_focus.as_ref() else { return };
    let Some(sd_arc) = focus.data::<Arc<Mutex<SurfaceData>>>() else { return };
    let tl = sd_arc
        .lock()
        .unwrap()
        .xdg_toplevel
        .as_ref()
        .and_then(|w| w.upgrade().ok());
    let Some(tl) = tl else {
        // Not a mapped xdg_toplevel — layer surface, popup, etc.
        // Those don't have a close request in the same sense.
        return;
    };
    tracing::info!(id = ?focus.id(), "close_window: sending xdg_toplevel.close");
    tl.close();
}

/// Alt+Tab handler: toggle fullscreen-in-the-layout for the focused
/// toplevel. First press zooms, second press (or any focus/move
/// command) restores the master-stack.
fn toggle_zoom(state: &mut State) {
    let Some(focus) = state.keyboard_focus.as_ref() else { return };
    // Only zoom actual tiled toplevels — popups, layer surfaces,
    // etc. aren't in mapped_toplevels and zooming them would have
    // no visible effect beyond breaking the layout invariant.
    if !state
        .mapped_toplevels
        .iter()
        .any(|w| w.surface.upgrade().ok().as_ref() == Some(focus))
    {
        return;
    }
    let already_zoomed = state
        .zoomed
        .as_ref()
        .and_then(|w| w.upgrade().ok())
        .as_ref()
        == Some(focus);
    state.zoomed = if already_zoomed { None } else { Some(focus.downgrade()) };
    state.needs_render = true;
}

/// Switch to the given workspace. Zoom doesn't follow the user
/// across workspaces — it's tied to the window that was zoomed, so
/// a second Alt+Tab on the new workspace is required to re-zoom.
/// Keyboard focus moves to the topmost (= most recently mapped)
/// toplevel on the target workspace; if it's empty, focus drops to
/// None, which means subsequent keystrokes go nowhere until a
/// window maps or a focus command is issued.
pub(crate) fn switch_workspace(state: &mut State, n: u32) {
    if state.active_ws == n {
        return;
    }
    let old = state.active_ws;
    tracing::info!(from = old, to = n, "workspace switch");
    state.active_ws = n;
    state.zoomed = None;
    ext_workspace::notify_active_changed(state, old, n);
    // Windows coming back into view get the "first-frame" treatment
    // in `tick_animations` — render_rect.w/h == 0 means "snap to the
    // assigned rect, no interpolation" — so the workspace appears
    // already laid out instead of animating in from a stale rect.
    for win in state.mapped_toplevels.iter_mut() {
        if win.workspace == n {
            win.render_rect = RectF::default();
            win.from_rect = RectF::default();
            win.anim_started_at = None;
        }
    }
    // Kick off the configured workspace transition. Crossfade keeps
    // the outgoing workspace rendered on top during the animation
    // with descending alpha; None just doesn't set the field and
    // the swap is instant. A duration of 0 also short-circuits to
    // the instant path.
    state.workspace_anim = match state.config.workspace_animation {
        config::WorkspaceAnimation::Crossfade
            if state.config.workspace_animation_duration_ms > 0 =>
        {
            Some(WorkspaceAnim {
                from_ws: old,
                to_ws: n,
                started_at: Instant::now(),
            })
        }
        _ => None,
    };
    let new_focus = state
        .mapped_toplevels
        .iter()
        .rev()
        .filter(|w| w.workspace == n)
        .find_map(|w| w.surface.upgrade().ok());
    state.set_keyboard_focus(new_focus);
    state.needs_render = true;
    update_pointer_focus_and_motion(state);
}

/// Send the keyboard-focused window to the given workspace. No-op
/// if the target is already the current workspace. The window keeps
/// whatever size it had — layout + configure for it land the next
/// time that workspace becomes active (apply_layout sees a changed
/// tile count on both the source and destination workspaces and
/// recomputes rects accordingly).
fn move_focused_to_workspace(state: &mut State, n: u32) {
    if n == state.active_ws {
        return;
    }
    let Some(focus) = state.keyboard_focus.clone() else {
        return;
    };
    let moved = state
        .mapped_toplevels
        .iter_mut()
        .find(|w| w.surface.upgrade().ok().as_ref() == Some(&focus))
        .map(|w| {
            w.workspace = n;
            // Clear render_rect so the window gets first-frame
            // treatment when the user switches to `n`.
            w.render_rect = RectF::default();
            w.from_rect = RectF::default();
            w.anim_started_at = None;
        })
        .is_some();
    if !moved {
        // Focus is on a layer surface or popup, not a tiled
        // toplevel — nothing to move.
        return;
    }
    // If the moved window was zoomed, zoom goes with it semantically
    // (it's cleared on the next switch anyway).
    let was_zoomed = state
        .zoomed
        .as_ref()
        .and_then(|w| w.upgrade().ok())
        .as_ref()
        == Some(&focus);
    if was_zoomed {
        state.zoomed = None;
    }
    // Focus falls to the topmost remaining window on the current
    // (source) workspace.
    let new_focus = state
        .mapped_toplevels
        .iter()
        .rev()
        .filter(|w| w.workspace == state.active_ws)
        .find_map(|w| w.surface.upgrade().ok());
    state.set_keyboard_focus(new_focus);
    state.needs_render = true;
    update_pointer_focus_and_motion(state);
    tracing::info!(to_ws = n, "moved focused window to workspace");
}

/// Build four thin rects outlining the keyboard-focused toplevel's
/// tile — the visual cue for "this is where typing goes." Empty vec
/// if no toplevel is focused, if the border is disabled
/// (`border_width = 0`), or if the focused surface isn't a mapped
/// toplevel (popup focus, layer-surface focus, etc. — those don't
/// get a tile border).
fn focus_border(state: &State) -> Vec<sol_core::SceneBorder> {
    let w = state.config.border_width;
    if w <= 0 {
        return Vec::new();
    }
    let Some(focus) = state.keyboard_focus.as_ref() else {
        return Vec::new();
    };
    let Some(win) = state
        .mapped_toplevels
        .iter()
        .find(|ww| {
            ww.workspace == state.active_ws
                && ww.surface.upgrade().ok().as_ref() == Some(focus)
        })
    else {
        return Vec::new();
    };
    // Border tracks what's actually on screen, so it uses render_rect
    // and stays visually glued to the tile during resize transitions.
    // Round to int — borders only need whole-pixel placement; the
    // sub-pixel render_rect is for the texture scale, not the border.
    let r = win.render_rect.round();
    let c = state.config.border_color;
    // Top / bottom span the full width including corners; left / right
    // are insets so the four rects don't double-draw at the corners.
    vec![
        sol_core::SceneBorder { x: r.x, y: r.y, w: r.w, h: w, rgba: c },
        sol_core::SceneBorder { x: r.x, y: r.y + r.h - w, w: r.w, h: w, rgba: c },
        sol_core::SceneBorder { x: r.x, y: r.y + w, w, h: r.h - 2 * w, rgba: c },
        sol_core::SceneBorder {
            x: r.x + r.w - w,
            y: r.y + w,
            w,
            h: r.h - 2 * w,
            rgba: c,
        },
    ]
}

fn render_tick(comp: &mut Compositor) -> Result<()> {
    // If a page flip is already in flight, skip this tick. The flip
    // will complete asynchronously on the DRM fd; the calloop source
    // there will fire frame callbacks and re-trigger a render if more
    // work has piled up in the meantime.
    if let BackendState::Drm(presenter) = &comp.backend {
        if presenter.is_pending_flip() {
            return Ok(());
        }
    }

    // Free GPU-side resources for buffers clients destroyed since the
    // last tick. Dmabuf-backed entries hold EGLImages that don't get
    // freed via Drop; they'd otherwise accumulate per client-resize.
    if let BackendState::Drm(presenter) = &mut comp.backend {
        for key in std::mem::take(&mut comp.state.pending_texture_evictions) {
            presenter.evict_texture(key);
        }
    } else {
        // Headless has no per-buffer GPU state; just discard the list.
        comp.state.pending_texture_evictions.clear();
    }

    // Prune dead weak refs, recompute tile rects, then push fresh configures
    // to any client whose tile size has changed since the last we told them.
    comp.state
        .mapped_toplevels
        .retain(|w| w.surface.upgrade().is_ok());
    rebalance_keyboard_focus(&mut comp.state);
    let now = Instant::now();
    apply_layout(&mut comp.state, now);
    send_pending_configures(&mut comp.state);
    // Step animations (lerp render_rect toward rect with cubic ease).
    // If anything is still mid-tween, keep the render loop firing —
    // page-flip-complete normally drives needs_render in DRM mode, so
    // this is a belt-and-braces guard for headless and for ticks
    // where no other event is pending.
    let animating = tick_animations(&mut comp.state, now);
    let ws_animating = tick_workspace_animation(&mut comp.state, now);
    if animating || ws_animating {
        comp.state.needs_render = true;
    }

    let (placed, background_count) = collect_scene(&comp.state, now);
    let mut scene = scene_from_buffers(&placed, background_count, &comp.state.cursor);
    scene.borders.extend(focus_border(&comp.state));
    let drawn = scene.elements.len();

    match &mut comp.backend {
        BackendState::Headless { canvas, png_path } => {
            canvas.clear();
            for e in &scene.elements {
                match &e.content {
                    SceneContent::Shm {
                        pixels,
                        stride,
                        format,
                    } => {
                        canvas.blit_argb(
                            e.x.round() as i32,
                            e.y.round() as i32,
                            pixels,
                            e.width,
                            e.height,
                            *stride,
                            *format,
                        );
                    }
                    SceneContent::Dmabuf { .. } => {
                        // Headless backend has no EGL context, can't sample
                        // the dmabuf. Skip — dmabuf clients are a DRM-backend
                        // feature.
                    }
                    SceneContent::BlurredBackdrop { .. } => {
                        // Headless has no GL context, no FBO, no blur path.
                        // Inactive-window frosted glass is a DRM-backend
                        // feature; the headless PNG dump just shows
                        // un-blurred wallpaper behind the inactive
                        // window's transparency.
                    }
                }
            }
            for b in &scene.borders {
                canvas.fill_rect(b.x, b.y, b.w, b.h, b.rgba);
            }
            canvas
                .write_png(png_path)
                .with_context(|| format!("write {}", png_path.display()))?;
            tracing::info!(drawn, path = %png_path.display(), "headless frame");
        }
        BackendState::Drm(presenter) => {
            presenter.render_scene(&scene).context("drm render_scene")?;
            tracing::debug!(drawn, "drm frame");
        }
    }

    // Don't release buffers here — release fires once, when a client
    // attaches a new buffer that replaces this one (handled in the
    // commit dispatch). Releasing per-frame was producing spurious
    // events for buffers still in use, which tripped GDK's
    // buffer_release_callback assertion and crashed Firefox.
    drop(placed);

    // Fire frame callbacks: for headless there's no page-flip concept so
    // we do it right after the PNG write; for DRM the callbacks fire in
    // the flip-complete handler (below, on the DRM fd source) so that
    // client redraws pace to vblank instead of to our render timing.
    if matches!(comp.backend, BackendState::Headless { .. }) {
        let ts = comp.state.elapsed_ms();
        for cb in std::mem::take(&mut comp.state.pending_frame_callbacks) {
            cb.done(ts);
        }
    }
    Ok(())
}

fn setup_event_loop(
    backend: BackendState,
    screen_width: u32,
    screen_height: u32,
    screen_refresh_mhz: i32,
    input: Option<(InputState, std::os::fd::OwnedFd)>,
    drm_device_path: Option<PathBuf>,
    session: Option<session::SharedSession>,
    cfg: config::Config,
) -> Result<()> {
    let mut event_loop: EventLoop<'static, Compositor> =
        EventLoop::try_new().context("create calloop event loop")?;
    let mut display: Display<State> = Display::new().context("create wayland display")?;
    let dh = display.handle();

    let globals = Globals {
        compositor: dh.create_global::<State, WlCompositor, ()>(COMPOSITOR_VERSION, ()),
        shm: dh.create_global::<State, WlShm, ()>(SHM_VERSION, ()),
        output: dh.create_global::<State, WlOutput, ()>(OUTPUT_VERSION, ()),
        seat: dh.create_global::<State, WlSeat, ()>(SEAT_VERSION, ()),
        xdg_wm_base: dh.create_global::<State, XdgWmBase, ()>(XDG_WM_BASE_VERSION, ()),
        linux_dmabuf: dh.create_global::<State, ZwpLinuxDmabufV1, ()>(
            linux_dmabuf::DMABUF_VERSION,
            (),
        ),
        xdg_decoration: dh.create_global::<State, ZxdgDecorationManagerV1, ()>(
            xdg_decoration::DECORATION_VERSION,
            (),
        ),
        layer_shell: dh.create_global::<State, ZwlrLayerShellV1, ()>(
            layer_shell::LAYER_SHELL_VERSION,
            (),
        ),
        subcompositor: dh.create_global::<State, WlSubcompositor, ()>(
            subcompositor::SUBCOMPOSITOR_VERSION,
            (),
        ),
        data_device_manager: dh.create_global::<State, WlDataDeviceManager, ()>(
            data_device::DATA_DEVICE_MANAGER_VERSION,
            (),
        ),
        xdg_output_manager: dh.create_global::<State, ZxdgOutputManagerV1, ()>(
            xdg_output::XDG_OUTPUT_MANAGER_VERSION,
            (),
        ),
        presentation: dh.create_global::<State, WpPresentation, ()>(
            presentation_time::PRESENTATION_VERSION,
            (),
        ),
        viewporter: dh.create_global::<State, WpViewporter, ()>(
            viewporter::VIEWPORTER_VERSION,
            (),
        ),
        fractional_scale: dh
            .create_global::<State, WpFractionalScaleManagerV1, ()>(
                fractional_scale::FRACTIONAL_SCALE_VERSION,
                (),
            ),
        ext_workspace: dh.create_global::<State, ExtWorkspaceManagerV1, ()>(
            ext_workspace::EXT_WORKSPACE_MANAGER_VERSION,
            (),
        ),
        idle_inhibit: dh.create_global::<State, ZwpIdleInhibitManagerV1, ()>(
            idle_inhibit::IDLE_INHIBIT_MANAGER_VERSION,
            (),
        ),
    };
    tracing::info!(?globals, "advertised globals");

    let listener = wayland_server::ListeningSocket::bind_auto("wayland", 1..33)
        .context("bind wayland socket")?;
    let socket_name = listener
        .socket_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<anonymous>".into());
    tracing::info!(socket = %socket_name, "listening");
    // Set WAYLAND_DISPLAY in our own process env so every future
    // spawn_client inherits it via Command's default env inheritance.
    // No more per-call .env("WAYLAND_DISPLAY", ...) overrides needed.
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", &socket_name);
    }

    event_loop
        .handle()
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            |_ready, listener, comp| {
                while let Some(stream) = listener.accept()? {
                    match comp
                        .state
                        .display_handle
                        .insert_client(stream, Arc::new(ClientState))
                    {
                        Ok(_) => comp.state.clients_seen += 1,
                        Err(e) => tracing::warn!(error = %e, "insert_client failed"),
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert socket source: {e}"))?;

    // Watch sol.conf for changes so layout knobs / keybinds / colors
    // tweak live. We watch the parent *directory*, not the file
    // itself: editors that save via "write to tmp, rename over" (vim
    // with `set backupcopy=auto`, neovim's default, kate, gedit) would
    // delete and replace the inode, taking our file-level watch with
    // it. Catching CLOSE_WRITE + MOVED_TO on the directory and
    // filtering by basename covers both write-in-place editors and
    // rename-on-save editors.
    if let Err(e) = install_config_watcher(&event_loop) {
        tracing::warn!(error = %e, "config file watch unavailable; live reload disabled");
    }

    let display_fd = display.backend().poll_fd().try_clone_to_owned()?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display_fd, Interest::READ, Mode::Level),
            |_ready, _fd, comp| {
                comp.display
                    .dispatch_clients(&mut comp.state)
                    .map_err(std::io::Error::other)?;
                comp.display
                    .flush_clients()
                    .map_err(std::io::Error::other)?;
                // A client dispatch may have created an idle
                // inhibitor while the screen was blanked — the
                // inhibit handler raises `pending_wake` because
                // it can't reach the backend. Act on it here,
                // where we do have backend access, before yielding
                // back to the event loop.
                if comp.state.pending_wake {
                    comp.state.pending_wake = false;
                    comp.state.idle = false;
                    comp.state.last_input_at = Instant::now();
                    comp.backend.set_dpms(false);
                    comp.state.needs_render = true;
                }
                // Any client request/disconnect can dirty our layout
                // or focus state — e.g. the client that just exited
                // left its toplevel dead in mapped_toplevels and
                // keyboard_focus pointing at an invalidated surface.
                // Flagging a render triggers the prune + rebalance
                // path on the next tick at negligible cost.
                comp.state.needs_render = true;
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert display source: {e}"))?;

    let (input_state, input_fd) = match input {
        Some((s, fd)) => (Some(s), Some(fd)),
        None => (None, None),
    };
    if let Some(fd) = input_fd {
        event_loop
            .handle()
            .insert_source(
                Generic::new(fd, Interest::READ, Mode::Level),
                |_ready, _fd, comp| {
                    let events = match comp.state.input.as_mut() {
                        Some(i) => i.drain(),
                        None => return Ok(PostAction::Continue),
                    };
                    if !events.is_empty() {
                        comp.state.last_input_at = Instant::now();
                        if comp.state.idle {
                            // Wake the screen on the first input after a
                            // blank. We don't forward this input to any
                            // client — a keystroke meant to wake up the
                            // monitor shouldn't also, say, close the
                            // currently-focused app because it happened
                            // to be Alt+Q. Consume-and-drop is what sway
                            // and hyprland do for the same reason.
                            comp.state.idle = false;
                            comp.backend.set_dpms(false);
                            comp.state.needs_render = true;
                            return Ok(PostAction::Continue);
                        }
                    }
                    for ev in events {
                        apply_input(&mut comp.state, ev);
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert input source: {e}"))?;
    }

    // DRM fd: kernel delivers page-flip-complete events here after a
    // page_flip ioctl lands on vblank. Keeping the wait async (instead
    // of the old blocking wait_for_page_flip at the end of every
    // render) is what lets us stay responsive — if a client stalls
    // inside render_scene, the compositor no longer gets wedged.
    if let BackendState::Drm(presenter) = &backend {
        let drm_fd = presenter.drm_fd().try_clone_to_owned()?;
        event_loop
            .handle()
            .insert_source(
                Generic::new(drm_fd, Interest::READ, Mode::Level),
                |_ready, _fd, comp| {
                    let saw_flip = if let BackendState::Drm(p) = &mut comp.backend {
                        p.drain_events().unwrap_or(false)
                    } else {
                        return Ok(PostAction::Continue);
                    };
                    if saw_flip {
                        let ts = comp.state.elapsed_ms();
                        for cb in std::mem::take(
                            &mut comp.state.pending_frame_callbacks,
                        ) {
                            cb.done(ts);
                        }
                        // Fire wp_presentation.feedback on every
                        // pending feedback object with a real-time
                        // vblank timestamp. Chrome uses this for its
                        // VSyncProvider; without it, its scheduler
                        // caps at 60 fps regardless of what
                        // wl_output.mode reports.
                        let refresh_ns = (1_000_000_000u64
                            / (comp.state.screen_refresh_mhz as u64 / 1000).max(1))
                            as u32;
                        presentation_time::fire_presented(
                            &mut comp.state,
                            Instant::now(),
                            refresh_ns,
                        );
                        // Another commit may have piled up while we
                        // were waiting; trigger a fresh render so the
                        // pipeline stays full.
                        comp.state.needs_render = true;

                        // Rough FPS counter: tally page flips and log
                        // once a second so the user can confirm the
                        // display is actually running at the selected
                        // refresh. Fires at INFO so it lands in the
                        // default log filter.
                        comp.state.flip_counter += 1;
                        let now = Instant::now();
                        match comp.state.flip_counter_reset {
                            Some(t) if now.duration_since(t).as_secs() >= 1 => {
                                let secs = now.duration_since(t).as_secs_f64();
                                let fps = comp.state.flip_counter as f64 / secs;
                                tracing::info!(
                                    fps = format!("{:.1}", fps),
                                    "display refresh"
                                );
                                comp.state.flip_counter = 0;
                                comp.state.flip_counter_reset = Some(now);
                            }
                            None => comp.state.flip_counter_reset = Some(now),
                            _ => {}
                        }
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert drm source: {e}"))?;
    }

    // libseat session: dispatch Enable/Disable events, handle DRM
    // master transitions. Without this source, VT switches would
    // leave us in a half-disabled state and libseat would block.
    if let Some(sess) = session.clone() {
        let poll_fd = {
            let mut s = sess.borrow_mut();
            s.poll_fd()?.try_clone_to_owned()?
        };
        event_loop
            .handle()
            .insert_source(
                Generic::new(poll_fd, Interest::READ, Mode::Level),
                move |_ready, _fd, comp| {
                    let events = match comp.state.session.as_ref() {
                        Some(s) => s
                            .borrow_mut()
                            .dispatch_events()
                            .unwrap_or_default(),
                        None => return Ok(PostAction::Continue),
                    };
                    for ev in events {
                        handle_session_event(comp, ev);
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert session source: {e}"))?;
    }

    let cursor = Cursor::new(screen_width as f64 / 2.0, screen_height as f64 / 2.0);
    // Only build the keymap when we have libinput. Headless has no keys to
    // translate and no clients to hand the fd to in any useful way.
    let keymap = if input_state.is_some() {
        match KeymapState::new_us() {
            Ok(km) => Some(km),
            Err(e) => {
                tracing::warn!(error = %e, "xkb keymap init failed; keyboard input disabled");
                None
            }
        }
    } else {
        None
    };
    let state = State {
        display_handle: dh,
        globals,
        clients_seen: 0,
        mapped_toplevels: Vec::new(),
        active_ws: 1,
        workspace_anim: None,
        outputs: Vec::new(),
        ext_workspace_managers: Vec::new(),
        idle_inhibitors: Vec::new(),
        last_input_at: Instant::now(),
        idle: false,
        pending_wake: false,
        screen_width,
        screen_height,
        screen_refresh_mhz,
        needs_render: false,
        started: Instant::now(),
        next_serial: 0,
        cursor,
        input: input_state,
        keymap,
        pointers: Vec::new(),
        keyboards: Vec::new(),
        pointer_focus: None,
        keyboard_focus: None,
        pending_frame_callbacks: Vec::new(),
        drm_device_path,
        socket_name: socket_name.clone(),
        left_alt_down: false,
        right_alt_down: false,
        left_ctrl_down: false,
        right_ctrl_down: false,
        left_shift_down: false,
        right_shift_down: false,
        left_super_down: false,
        right_super_down: false,
        config: cfg,
        zoomed: None,
        flip_counter: 0,
        flip_counter_reset: None,
        pending_presentation: presentation_time::empty(),
        presentation_seq: 0,
        exec_once_pgids: Vec::new(),
        pending_texture_evictions: Vec::new(),
        pending_layer_surfaces: Vec::new(),
        session: session.clone(),
    };
    let mut compositor = Compositor {
        state,
        display,
        backend,
    };
    let _ = compositor.display.flush_clients();

    // Idle timer: 1 Hz tick that checks "have we been idle longer
    // than the configured threshold, and is no client holding an
    // inhibitor?" — if yes, DPMS-off. Skipped entirely when
    // `idle_timeout == 0` (the default) so the feature stays out of
    // the way for users who haven't opted in. 1 Hz bounds worst-case
    // blank latency to a second after the threshold expires while
    // keeping the callback nearly free.
    let idle_timeout_secs = compositor.state.config.idle_timeout;
    if idle_timeout_secs > 0 {
        use calloop::timer::{TimeoutAction, Timer};
        let tick = std::time::Duration::from_secs(1);
        event_loop
            .handle()
            .insert_source(
                Timer::from_duration(tick),
                move |_deadline, _meta, comp| {
                    if !comp.state.idle {
                        let elapsed = comp.state.last_input_at.elapsed().as_secs();
                        if elapsed >= idle_timeout_secs as u64
                            && !idle_inhibit::any_active(&mut comp.state)
                        {
                            tracing::info!(
                                elapsed,
                                inhibitors = comp.state.idle_inhibitors.len(),
                                "idle: blanking display"
                            );
                            comp.state.idle = true;
                            comp.backend.set_dpms(true);
                        }
                    }
                    TimeoutAction::ToDuration(tick)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert idle timer: {e}"))?;
    }

    tracing::info!(
        socket = %socket_name,
        "ready: export WAYLAND_DISPLAY={} and run a client",
        socket_name
    );

    // Fire `exec-once` commands from the config. Runs after the socket
    // is bound and WAYLAND_DISPLAY is exported (above) so children can
    // connect immediately; a wallpaper daemon (swaybg / hyprpaper /
    // mpvpaper) listed here comes up without any manual launch.
    // Clone first so we can borrow &mut state inside the loop for
    // PID tracking.
    let exec_once: Vec<_> = compositor.state.config.exec_once.clone();
    for cmd in &exec_once {
        let argrefs: Vec<&str> = cmd.args.iter().map(|s| s.as_str()).collect();
        spawn_exec_once(&mut compositor.state, &cmd.program, &argrefs);
    }

    // External stop signals (SIGTERM, SIGHUP — the ones `killall sol`
    // and `kill <pid>` send) trigger a clean shutdown: calloop returns
    // from `run`, Compositor drops, DrmPresenter's Drop restores the
    // saved CRTC. If the first stop can't make progress (e.g. the
    // event loop is wedged on a blocking DRM ioctl), a second signal
    // within ~3s calls libc::_exit so the user can always kill us
    // without resorting to SIGKILL. SIGINT is separately set to
    // SIG_IGN so Ctrl+C doesn't stop the compositor at all — it gets
    // forwarded to the focused client via libinput.
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_STOP: AtomicU64 = AtomicU64::new(0);
    let loop_signal = event_loop.get_signal();
    if let Err(e) = ctrlc::set_handler(move || {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let prev = LAST_STOP.swap(now, Ordering::SeqCst);
        if prev != 0 && now.saturating_sub(prev) < 3000 {
            // Second signal in 3s — event loop clearly can't unwedge.
            // Bypass Drop and exit immediately; TTY may stay blank
            // until logind takes over, but we're not killable-able.
            tracing::error!("second stop signal in under 3s; forcing exit");
            unsafe { libc::_exit(130) };
        }
        tracing::info!("shutdown signal received; stopping event loop");
        loop_signal.stop();
    }) {
        tracing::warn!(error = %e, "ctrlc handler install failed; `killall sol` won't cleanly restore the TTY");
    }
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_IGN);
    }

    event_loop
        .run(None, &mut compositor, |comp| {
            if comp.state.needs_render {
                comp.state.needs_render = false;
                if let Err(e) = render_tick(comp) {
                    tracing::error!(error = %e, "render failed");
                }
            }
            let _ = comp.display.flush_clients();
        })
        .context("event loop errored")?;

    // SIGTERM every exec-once pgroup so wallpaper daemons / cycler
    // shells don't linger as orphans across sol restarts.
    terminate_exec_once(&compositor.state);

    Ok(())
}

/// Handle a single libseat event. Keeps both the DRM backend's master
/// state AND libinput's device fds in sync with our session ownership,
/// then acks the Disable so the daemon can proceed with the VT switch.
///
/// libinput suspend/resume is mandatory here: logind revokes our
/// /dev/input/event* fds on Disable, so libinput is left with stale
/// fds that stop producing events even after we come back. Calling
/// `suspend` before we ack Disable closes them via our
/// close_restricted callback, and `resume` on Enable re-opens fresh
/// ones via open_restricted (which routes through libseat).
fn handle_session_event(comp: &mut Compositor, ev: libseat::SeatEvent) {
    match ev {
        libseat::SeatEvent::Disable => {
            tracing::info!("libseat: Disable — suspending input + releasing DRM master");
            if let Some(input) = comp.state.input.as_mut() {
                input.li.suspend();
            }
            if let BackendState::Drm(presenter) = &mut comp.backend {
                presenter.drop_master();
            }
            if let Some(s) = comp.state.session.as_ref() {
                let mut s = s.borrow_mut();
                s.active = false;
                if let Err(e) = s.ack_disable() {
                    tracing::warn!(error = %e, "libseat: ack_disable");
                }
            }
        }
        libseat::SeatEvent::Enable => {
            tracing::info!("libseat: Enable — reacquiring DRM master + resuming input");
            if let Some(s) = comp.state.session.as_ref() {
                s.borrow_mut().active = true;
            }
            if let BackendState::Drm(presenter) = &mut comp.backend {
                if let Err(e) = presenter.reacquire_master() {
                    tracing::warn!(error = %e, "DRM reacquire on Enable");
                }
            }
            if let Some(input) = comp.state.input.as_mut() {
                if input.li.resume().is_err() {
                    tracing::warn!("libinput resume failed — input may be dead");
                }
            }
            // A VT switch back to sol should land on a lit
            // screen regardless of where the idle state was when
            // we left. Reset idle + reset the input clock so the
            // user doesn't immediately re-blank on resume.
            comp.state.idle = false;
            comp.state.last_input_at = Instant::now();
            comp.backend.set_dpms(false);
            comp.state.needs_render = true;
        }
    }
}

/// Apply one translated libinput event to the state. Updates the cursor,
/// sets `needs_render`, and forwards pointer/key events to the right
/// Wayland client based on current focus.
fn apply_input(state: &mut State, ev: InputEvent) {
    // Drop dead resources so we don't try to send to them.
    state.pointers.retain(|p| p.is_alive());
    state.keyboards.retain(|k| k.is_alive());

    match ev {
        InputEvent::PointerMotion { dx, dy } => {
            state.cursor.pos_x = (state.cursor.pos_x + dx)
                .clamp(0.0, state.screen_width as f64 - 1.0);
            state.cursor.pos_y = (state.cursor.pos_y + dy)
                .clamp(0.0, state.screen_height as f64 - 1.0);
            state.needs_render = true;
            update_pointer_focus_and_motion(state);
        }
        InputEvent::PointerMotionAbsolute { x_mm, y_mm } => {
            state.cursor.pos_x = x_mm.clamp(0.0, state.screen_width as f64 - 1.0);
            state.cursor.pos_y = y_mm.clamp(0.0, state.screen_height as f64 - 1.0);
            state.needs_render = true;
            update_pointer_focus_and_motion(state);
        }
        InputEvent::PointerButton { button, pressed } => {
            send_pointer_button(state, button, pressed);
        }
        InputEvent::PointerAxis {
            source,
            v_value,
            h_value,
            v120_v,
            v120_h,
        } => {
            send_pointer_axis(state, source, v_value, h_value, v120_v, v120_h);
        }
        InputEvent::Key { keycode, pressed } => {
            // Apply user-declared scancode remaps before anything else
            // so modifier tracking, config bindings, and the forwarded
            // key all see the rewritten code. This is what makes
            // `remap = CapsLock, Escape` behave as an Escape press end
            // to end (xkb sees ESC, the client sees ESC).
            let keycode = state.config.remap(keycode);

            // Track modifiers so keybinds see accurate state. Released
            // modifiers stop participating.
            match keycode {
                KEY_LEFTALT => state.left_alt_down = pressed,
                KEY_RIGHTALT => state.right_alt_down = pressed,
                KEY_LEFTCTRL => state.left_ctrl_down = pressed,
                KEY_RIGHTCTRL => state.right_ctrl_down = pressed,
                KEY_LEFTSHIFT => state.left_shift_down = pressed,
                KEY_RIGHTSHIFT => state.right_shift_down = pressed,
                KEY_LEFTMETA => state.left_super_down = pressed,
                KEY_RIGHTMETA => state.right_super_down = pressed,
                _ => {}
            }
            let alt = state.left_alt_down || state.right_alt_down;
            let ctrl = state.left_ctrl_down || state.right_ctrl_down;
            let shift = state.left_shift_down || state.right_shift_down;
            let sup = state.left_super_down || state.right_super_down;

            // Ctrl+Alt+F1..F12 → libseat VT switch. Intentionally not
            // overridable from the config: a misconfigured .conf must
            // never be able to trap the user on the compositor.
            if pressed && ctrl && alt {
                if let Some(vt) = vt_number_for_f_key(keycode) {
                    if let Some(s) = state.session.as_ref() {
                        if let Err(e) = s.borrow_mut().switch_session(vt) {
                            tracing::warn!(%e, vt, "libseat switch_session");
                        } else {
                            tracing::info!(vt, "Ctrl+Alt+F{vt}: switching VT");
                        }
                    }
                    return;
                }
            }

            // Config-driven bindings. Exact match on the modifier mask
            // so Alt+Return doesn't also fire on Ctrl+Alt+Return.
            if pressed {
                let mut mask = 0u8;
                if alt { mask |= config::MOD_ALT; }
                if ctrl { mask |= config::MOD_CTRL; }
                if shift { mask |= config::MOD_SHIFT; }
                if sup { mask |= config::MOD_SUPER; }
                if let Some(b) =
                    state.config.bindings.iter().find(|b| b.mods == mask && b.key == keycode)
                {
                    match b.action.clone() {
                        config::Action::Spawn(cmd) => {
                            let argrefs: Vec<&str> =
                                cmd.args.iter().map(|s| s.as_str()).collect();
                            let label = format!(
                                "{}+{}",
                                config::mods_to_label(mask),
                                cmd.program
                            );
                            spawn_client(state, &cmd.program, &argrefs, &label);
                        }
                        config::Action::FocusDir(dir) => {
                            focus_direction(state, dir);
                        }
                        config::Action::MoveDir(dir) => {
                            move_direction(state, dir);
                        }
                        config::Action::ToggleZoom => {
                            toggle_zoom(state);
                        }
                        config::Action::CloseWindow => {
                            close_focused_window(state);
                        }
                        config::Action::Workspace(n) => {
                            switch_workspace(state, n);
                        }
                        config::Action::MoveToWorkspace(n) => {
                            move_focused_to_workspace(state, n);
                        }
                    }
                    return;
                }
            }
            send_keyboard_key(state, keycode, pressed);
        }
    }
}

/// Map an evdev F1..F12 keycode to the 1-based VT number libseat expects.
/// F1..F10 are contiguous (59..68), F11 and F12 jump to 87 and 88.
fn vt_number_for_f_key(keycode: u32) -> Option<i32> {
    match keycode {
        59..=68 => Some((keycode - 58) as i32), // F1 → VT 1 ... F10 → VT 10
        87 => Some(11),
        88 => Some(12),
        _ => None,
    }
}

/// evdev scancodes for the modifier keys we track internally. Non-
/// modifier scancodes live in `config::key_from_name`.
const KEY_LEFTALT: u32 = 56;
const KEY_RIGHTALT: u32 = 100;
const KEY_LEFTCTRL: u32 = 29;
const KEY_RIGHTCTRL: u32 = 97;
const KEY_LEFTSHIFT: u32 = 42;
const KEY_RIGHTSHIFT: u32 = 54;
const KEY_LEFTMETA: u32 = 125;
const KEY_RIGHTMETA: u32 = 126;

/// Fork/exec a Wayland client connected to our own socket. Env is
/// inherited wholesale from sol's process, which was normalised at
/// startup (XDG_SESSION_TYPE=wayland, XDG_CURRENT_DESKTOP=sol,
/// DISPLAY/XAUTHORITY unset, WAYLAND_DISPLAY set). Child handle is
/// intentionally dropped — no reap, no wait; kernel cleans up on
/// sol exit.
/// Spawn an `exec-once` child in its own session / process group
/// and record the pgid on `State` for shutdown cleanup. Using
/// `setsid` (via `pre_exec`) puts the child and any grandchildren
/// (e.g. `sleep` / `find` spawned by a wrapper shell like
/// `wp-cycle.sh`) in a fresh pgroup that we can blanket-kill with
/// `killpg(pgid, SIGTERM)` when sol exits. Without this, those
/// wrapper scripts outlive every sol restart, accumulate as
/// orphans, and each runs its own independent wallpaper-cycle
/// timer — which surfaces as "transitions every few seconds"
/// after a handful of compositor restarts.
fn spawn_exec_once(state: &mut State, program: &str, args: &[&str]) {
    use std::os::unix::process::CommandExt;
    let socket = state.socket_name.clone();
    let mut cmd = std::process::Command::new(program);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    unsafe {
        cmd.pre_exec(|| {
            // New session → new process group with the child as
            // leader. Safe in the forked child; returns -1 with
            // EPERM only if we're already a session leader, which
            // we aren't (we just forked from sol).
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id() as libc::pid_t;
            state.exec_once_pgids.push(pid);
            tracing::info!(
                pid,
                socket = %socket,
                program,
                "exec-once: spawned (new pgroup)"
            );
            // Drop the Child handle — we don't want to wait on it,
            // we'll signal the pgroup directly at shutdown. The
            // kernel reaps the zombie once sol exits (our own
            // process parent will reap us; our children left around
            // get re-parented to init which reaps them).
            std::mem::forget(child);
        }
        Err(e) => {
            tracing::warn!(error = %e, program, "exec-once: spawn failed");
        }
    }
}

/// Send SIGTERM to every pgroup we spawned via `exec-once`. Called
/// after the event loop returns so wrapper-shell loops and
/// wallpaper daemons don't linger as orphans after sol exits.
fn terminate_exec_once(state: &State) {
    for &pgid in &state.exec_once_pgids {
        // kill(-pgid, SIGTERM) addresses the whole process group.
        let rc = unsafe { libc::kill(-pgid, libc::SIGTERM) };
        if rc == -1 {
            let err = std::io::Error::last_os_error();
            // ESRCH just means the group is already gone — fine.
            if err.raw_os_error() != Some(libc::ESRCH) {
                tracing::warn!(error = %err, pgid, "exec-once: killpg failed");
            }
        } else {
            tracing::info!(pgid, "exec-once: killpg SIGTERM");
        }
    }
}

fn spawn_client(state: &State, program: &str, args: &[&str], label: &str) {
    let socket = state.socket_name.clone();
    match std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => tracing::info!(
            pid = child.id(),
            socket = %socket,
            program,
            "{label}: spawned"
        ),
        Err(e) => {
            tracing::warn!(error = %e, program, "{label}: spawn failed");
        }
    }
}

/// Hit-test the cursor top-to-bottom through the full z-stack:
/// Overlay → Top layer surfaces → tiled toplevels → Bottom → Background
/// layer surfaces. Returns the surface plus cursor position in
/// surface-local coords.
fn surface_under_cursor(state: &State) -> Option<(WlSurface, f64, f64)> {
    use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::Layer;
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };
    let cx = state.cursor.pos_x;
    let cy = state.cursor.pos_y;

    let hit_rect = |r: Rect| -> Option<(f64, f64)> {
        let lx = cx - r.x as f64;
        let ly = cy - r.y as f64;
        if lx >= 0.0 && lx < r.w as f64 && ly >= 0.0 && ly < r.h as f64 {
            Some((lx, ly))
        } else {
            None
        }
    };

    // Given a parent surface that the cursor is currently over (at
    // `lx, ly` in the parent's local coords), return the topmost
    // subsurface descendant that also contains the cursor — or the
    // parent itself if none do. This mirrors our scene walker, which
    // draws subsurfaces on top of their parent: pointer input has to
    // follow the same stacking so clicks actually reach popups,
    // tooltips, and the close boxes on toast dialogs.
    let resolve_hit = |parent: WlSurface, lx: f64, ly: f64| -> (WlSurface, f64, f64) {
        hit_subsurfaces(&parent, lx, ly).unwrap_or((parent, lx, ly))
    };

    let layers = layer_shell::mapped_layers(state, screen);

    // Pass 1: Overlay > Top. Iterate overlay-first (descending priority).
    for ml in layers
        .iter()
        .filter(|m| matches!(m.layer, Layer::Overlay))
        .chain(layers.iter().filter(|m| matches!(m.layer, Layer::Top)))
    {
        if let Some((lx, ly)) = hit_rect(ml.rect) {
            return Some(resolve_hit(ml.surface.clone(), lx, ly));
        }
    }

    // Pass 2: tiled toplevels (top of stack first). Honor zoom: while
    // zoomed, the non-zoomed tiles don't render and mustn't catch
    // clicks either.
    let zoomed = state.zoomed.as_ref().and_then(|w| w.upgrade().ok());
    for win in state.mapped_toplevels.iter().rev() {
        if win.workspace != state.active_ws {
            continue;
        }
        let Ok(surface) = win.surface.upgrade() else { continue };
        if let Some(zs) = &zoomed {
            if surface != *zs {
                continue;
            }
        }
        let sd_arc = match surface.data::<Arc<Mutex<SurfaceData>>>() {
            Some(s) => s,
            None => continue,
        };
        if !matches!(
            sd_arc.lock().unwrap().role,
            SurfaceRole::XdgToplevel { mapped: true }
        ) {
            continue;
        }
        // Hit test against what the user can actually see (render_rect)
        // — clicking on the visible tile should hit, regardless of
        // whether the layout target has moved past it already.
        if let Some((lx, ly)) = hit_rect(win.render_rect.round()) {
            return Some(resolve_hit(surface, lx, ly));
        }
    }

    // Pass 3: Bottom > Background.
    for ml in layers
        .iter()
        .filter(|m| matches!(m.layer, Layer::Bottom))
        .chain(layers.iter().filter(|m| matches!(m.layer, Layer::Background)))
    {
        if let Some((lx, ly)) = hit_rect(ml.rect) {
            return Some(resolve_hit(ml.surface.clone(), lx, ly));
        }
    }

    None
}

/// Walk `parent`'s subsurface tree topmost-first and return the
/// deepest descendant whose buffer-sized rect covers the cursor
/// (given `lx, ly` relative to `parent`'s origin). Without this,
/// Chrome's toast-style dialogs (attached as subsurfaces) receive
/// no pointer events because we deliver clicks to the outer
/// toplevel instead of the subsurface the client is actually
/// listening on.
fn hit_subsurfaces(
    parent: &WlSurface,
    lx: f64,
    ly: f64,
) -> Option<(WlSurface, f64, f64)> {
    let sd_arc = parent.data::<Arc<Mutex<SurfaceData>>>()?;
    // Snapshot children while holding the parent lock only briefly;
    // recurse outside so child locks don't nest.
    let children: Vec<(WlSurface, (i32, i32), (i32, i32))> = {
        let sd = sd_arc.lock().ok()?;
        sd.subsurface_children
            .iter()
            .filter_map(|w| {
                let s = w.upgrade().ok()?;
                // Lock the child's SurfaceData, read out everything we
                // need, then drop the guard before moving `s` into
                // the tuple — otherwise the borrow checker sees the
                // guard still holding a reference to `s` via its arc.
                let (offset, dims) = {
                    let csd_arc = s.data::<Arc<Mutex<SurfaceData>>>()?;
                    let csd = csd_arc.lock().ok()?;
                    let offset = csd.subsurface_offset;
                    let buf = csd.current.buffer.as_ref()?;
                    let dims = surface_buffer_dims(buf)?;
                    (offset, dims)
                };
                Some((s, offset, dims))
            })
            .collect()
    };
    // Topmost first: collect_scene draws in registration order, so
    // last-registered is on top, so we iterate in reverse.
    for (child, (ox, oy), (cw, ch)) in children.into_iter().rev() {
        let child_lx = lx - ox as f64;
        let child_ly = ly - oy as f64;
        if child_lx >= 0.0
            && child_lx < cw as f64
            && child_ly >= 0.0
            && child_ly < ch as f64
        {
            // Recurse so nested popups (rare but legal) still route
            // correctly; if none of THIS child's children claim the
            // cursor, the child itself gets the hit.
            return Some(
                hit_subsurfaces(&child, child_lx, child_ly)
                    .unwrap_or((child, child_lx, child_ly)),
            );
        }
    }
    None
}

fn surface_buffer_dims(
    buf: &wayland_server::protocol::wl_buffer::WlBuffer,
) -> Option<(i32, i32)> {
    if let Some(bd) = buf.data::<shm::BufferData>() {
        return Some((bd.width, bd.height));
    }
    if let Some(db) = buf.data::<linux_dmabuf::DmabufBuffer>() {
        return Some((db.width, db.height));
    }
    None
}

fn update_pointer_focus_and_motion(state: &mut State) {
    let hit = surface_under_cursor(state);
    let new_focus = hit.as_ref().map(|(s, _, _)| s.clone());
    let focus_changed = !surface_eq(state.pointer_focus.as_ref(), new_focus.as_ref());

    if focus_changed {
        if let Some(old) = state.pointer_focus.take() {
            if old.is_alive() {
                let serial = state.next_serial();
                for p in &state.pointers {
                    if same_client(p, &old) {
                        p.leave(serial, &old);
                        pointer_frame(p);
                    }
                }
            }
        }
        if let Some((surface, lx, ly)) = hit.as_ref() {
            let serial = state.next_serial();
            for p in &state.pointers {
                if same_client(p, surface) {
                    p.enter(serial, surface, *lx, *ly);
                    pointer_frame(p);
                }
            }
        }
        state.pointer_focus = new_focus;
    } else if let Some((surface, lx, ly)) = hit.as_ref() {
        // Ordinary motion within the same surface.
        let time = state.elapsed_ms();
        for p in &state.pointers {
            if same_client(p, surface) {
                p.motion(time, *lx, *ly);
                pointer_frame(p);
            }
        }
    }
}

fn pointer_frame(p: &WlPointer) {
    if p.version() >= 5 {
        p.frame();
    }
}

/// Forward a libinput scroll event to the focused client as a
/// wl_pointer axis frame. Scroll always targets the surface currently
/// under the cursor (`pointer_focus`) — it does not change focus.
///
/// Wire format:
///   - v5+: axis_source(source)
///   - for each axis with data:
///       * if the value is 0 and the source is a kinetic one (Finger /
///         Continuous), emit axis_stop — this is libinput's contract
///         for end-of-gesture and clients use it to trigger kinetic
///         deceleration
///       * otherwise emit axis(time, axis, value)
///   - wheel sources additionally emit high-res step info:
///       * v8+: axis_value120(axis, v120) — preferred
///       * v5..=v7: axis_discrete(axis, v120 / 120) — best-effort
///         fallback; loses sub-click resolution
///   - v5+: frame() to commit the group
fn send_pointer_axis(
    state: &mut State,
    source: AxisSource,
    v_value: Option<f64>,
    h_value: Option<f64>,
    v120_v: Option<f64>,
    v120_h: Option<f64>,
) {
    let Some(focus) = state.pointer_focus.clone() else { return };
    if !focus.is_alive() {
        return;
    }
    let time = state.elapsed_ms();
    let wl_source = match source {
        AxisSource::Wheel => wl_pointer::AxisSource::Wheel,
        AxisSource::Finger => wl_pointer::AxisSource::Finger,
        AxisSource::Continuous => wl_pointer::AxisSource::Continuous,
    };
    let kinetic = matches!(source, AxisSource::Finger | AxisSource::Continuous);
    for p in &state.pointers {
        if !same_client(p, &focus) {
            continue;
        }
        let v = p.version();
        if v >= 5 {
            p.axis_source(wl_source);
        }
        if let Some(val) = v_value {
            if val == 0.0 && kinetic && v >= 5 {
                p.axis_stop(time, wl_pointer::Axis::VerticalScroll);
            } else {
                p.axis(time, wl_pointer::Axis::VerticalScroll, val);
            }
        }
        if let Some(val) = h_value {
            if val == 0.0 && kinetic && v >= 5 {
                p.axis_stop(time, wl_pointer::Axis::HorizontalScroll);
            } else {
                p.axis(time, wl_pointer::Axis::HorizontalScroll, val);
            }
        }
        if matches!(source, AxisSource::Wheel) {
            if let Some(step) = v120_v {
                if v >= 8 {
                    p.axis_value120(wl_pointer::Axis::VerticalScroll, step as i32);
                } else if v >= 5 {
                    let discrete = (step / 120.0).round() as i32;
                    if discrete != 0 {
                        p.axis_discrete(wl_pointer::Axis::VerticalScroll, discrete);
                    }
                }
            }
            if let Some(step) = v120_h {
                if v >= 8 {
                    p.axis_value120(wl_pointer::Axis::HorizontalScroll, step as i32);
                } else if v >= 5 {
                    let discrete = (step / 120.0).round() as i32;
                    if discrete != 0 {
                        p.axis_discrete(wl_pointer::Axis::HorizontalScroll, discrete);
                    }
                }
            }
        }
        pointer_frame(p);
    }
}

fn send_pointer_button(state: &mut State, button: u32, pressed: bool) {
    let Some(focus) = state.pointer_focus.clone() else { return };
    if !focus.is_alive() {
        return;
    }
    // Click-to-focus: on press, move keyboard focus to the window under the
    // cursor. No-op if that window already has focus (set_keyboard_focus
    // compares first).
    if pressed {
        state.set_keyboard_focus(Some(focus.clone()));
    }
    let serial = state.next_serial();
    let time = state.elapsed_ms();
    let button_state = if pressed {
        wl_pointer::ButtonState::Pressed
    } else {
        wl_pointer::ButtonState::Released
    };
    for p in &state.pointers {
        if same_client(p, &focus) {
            p.button(serial, time, button, button_state);
            pointer_frame(p);
        }
    }
}

fn send_keyboard_key(state: &mut State, keycode: u32, pressed: bool) {
    // Always feed the key to xkb — even when focus is dead or missing —
    // so the modifier state stays consistent with the physical
    // keyboard. If we skipped the release of a modifier because the
    // focused client had just died (e.g. user typed Ctrl+D, the shell
    // exited, the Ctrl release event then arrived with no live focus),
    // xkb would think that modifier is still held forever. The next
    // focused client would then receive a `modifiers` event with
    // stale depressed bits and misinterpret every subsequent keystroke
    // as Ctrl+<whatever>.
    let mods = state
        .keymap
        .as_mut()
        .map(|km| km.feed_key(keycode, pressed));

    let Some(focus) = state.keyboard_focus.clone() else { return };
    if !focus.is_alive() {
        return;
    }
    let time = state.elapsed_ms();
    let key_state = if pressed {
        wl_keyboard::KeyState::Pressed
    } else {
        wl_keyboard::KeyState::Released
    };

    if let Some(mods) = mods {
        let changed = state
            .keymap
            .as_mut()
            .map(|km| km.mods_changed(mods))
            .unwrap_or(false);
        if changed {
            let serial = state.next_serial();
            for kb in &state.keyboards {
                if same_client(kb, &focus) {
                    kb.modifiers(serial, mods.depressed, mods.latched, mods.locked, mods.group);
                }
            }
        }
    }

    let serial = state.next_serial();
    for kb in &state.keyboards {
        if same_client(kb, &focus) {
            kb.key(serial, time, keycode, key_state);
        }
    }
}

/// Headless backend: software canvas, dumps a PNG every frame.
pub fn run_headless(png_path: PathBuf, width: u32, height: u32) -> Result<()> {
    tracing::info!(path = %png_path.display(), width, height, "headless backend");
    let backend = BackendState::Headless {
        canvas: Canvas::new(width, height),
        png_path,
    };
    let cfg = config::load();
    // 60 mHz is a stand-in; headless has no real output and nothing
    // paces off wl_output.mode here.
    setup_event_loop(backend, width, height, 60_000, None, None, None, cfg)
}

/// DRM backend, managed via libseat.
///
/// Startup sequence:
///   1. Open a libseat session. On systemd hosts this talks to
///      systemd-logind; otherwise seatd. The daemon owns device files.
///   2. Pump the session once to consume the initial Enable event so
///      we know the daemon has handed us the seat.
///   3. Ask libseat for the DRM device fd and wrap it into a Card.
///      Build the DrmPresenter off that.
///   4. Initialise libinput with a LibinputInterface that routes every
///      open_restricted through libseat.
///   5. Hand the session + Wayland bits to setup_event_loop; it wires
///      the session poll fd into calloop and handles enable/disable.
pub fn run_drm(device: &Path) -> Result<()> {
    let session = session::Session::open().context("libseat: open session")?;

    // Wait for the initial Enable. Under logind the first event can
    // be delayed by a D-Bus round-trip; a non-blocking dispatch(0)
    // poll-and-sleep loop can miss the notify and bail. dispatch with
    // a positive timeout lets libseat block on its own fd until the
    // daemon sends something. Loop in case other events (e.g. an
    // unrelated Disable from a prior session, though unusual) arrive
    // first.
    {
        let mut s = session.borrow_mut();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !s.active {
            let remaining = deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_millis() as i32;
            if remaining <= 0 {
                anyhow::bail!(
                    "libseat: no Enable event within 5s — the daemon hasn't \
                     handed us the seat. Make sure sol was launched from \
                     an active logind session (check `loginctl show-session \
                     $XDG_SESSION_ID` from the TTY where you're running it)."
                );
            }
            for ev in s
                .dispatch_events_blocking(remaining)
                .context("libseat: waiting for Enable")?
            {
                if matches!(ev, libseat::SeatEvent::Enable) {
                    s.active = true;
                }
            }
        }
        tracing::info!(seat = %s.name(), "libseat session active");
    }

    // Load config first so the user's `mode = WxH@Hz` (if any) drives
    // initial mode selection. The DRM backend's pick_output uses this
    // as a fallback when SOL_MODE isn't set.
    let cfg = config::load();
    let mode_pref = cfg.mode.map(|m| sol_backend_drm::ModePreference {
        width: m.width,
        height: m.height,
        refresh_hz: m.refresh_hz,
    });

    // Open the DRM device through libseat, wrap into a Card, build the
    // presenter off it.
    let drm_fd = session
        .borrow_mut()
        .open_device_keep_fd(device)
        .context("libseat: open DRM device")?;
    let card = sol_backend_drm::Card::from_fd(drm_fd);
    let presenter = DrmPresenter::from_card(card, mode_pref)
        .context("initialise DrmPresenter")?;
    let (w, h) = presenter.size();
    let refresh_mhz = presenter.refresh_hz() as i32 * 1000;
    tracing::info!(width = w, height = h, refresh_mhz, "drm backend");

    // libinput is DRM-only: a headless PNG dump has nowhere to point a cursor.
    let input = match InputState::init("seat0", session.clone()) {
        Ok(pair) => Some(pair),
        Err(e) => {
            tracing::warn!(error = %e, "libinput init failed; running without input");
            None
        }
    };
    setup_event_loop(
        BackendState::Drm(presenter),
        w,
        h,
        refresh_mhz,
        input,
        Some(device.to_path_buf()),
        Some(session),
        cfg,
    )
}

/// Back-compat entry: default to headless with the old PNG path.
pub fn run() -> Result<()> {
    let png_path = std::env::var_os("SOL_PNG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sol-headless.png"));
    run_headless(png_path, 1920, 1080)
}

//! Wayland server for sol.
//!
//! Handles protocol traffic in all backends; rendering is delegated to a
//! `BackendState` value (software canvas -> PNG for headless, or a
//! `sol_backend_drm::DrmPresenter` for real hardware).

// `collapsible_if` produces long `&& let Some(x) = ...` chains that
// hurt readability for the protocol-dispatch style this crate uses.
// `collapsible_match` rewrites legible nested matches into denser if
// lets that obscure intent. `too_many_arguments` fires on the few
// rendering helpers where threading state through is intentional.
#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::too_many_arguments
)]

// Re-export smithay so per-protocol modules can reach it via
// `crate::smithay::wayland::shm::...` etc as they migrate, without
// each one needing its own `use smithay::...` and without the rest
// of the workspace pulling smithay in directly. PR 0 only links it;
// PR 1 (wl_shm) is the first real consumer.
pub use smithay;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction, generic::Generic};
// Smithay's Session trait takes its own (re-exported) rustix's OFlags;
// our workspace `rustix = "0.38"` and smithay's pinned `rustix = "1.x"`
// are different crates at the type level. Use smithay's re-export at
// the only callsite that needs it (`run_drm`'s `session.open`).
use smithay::reexports::rustix::fs::OFlags;
use sol_backend_drm::DrmPresenter;
use std::os::fd::AsRawFd;
use sol_core::{RenderTiming, Scene, SceneContent, SceneElement};
use wayland_protocols::xdg::shell::server::xdg_wm_base::XdgWmBase;
use wayland_server::{
    Client, Display, DisplayHandle, Resource, Weak,
    backend::{ClientData, ClientId, DisconnectReason, GlobalId, ObjectId},
    protocol::{
        wl_output::WlOutput, wl_surface::WlSurface,
    },
};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::shm::{ShmHandler, ShmState};
use smithay::wayland::compositor::{
    self as smithay_compositor, BufferAssignment, CompositorClientState,
    CompositorHandler, CompositorState, SubsurfaceCachedState, SurfaceAttributes,
};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::input::keyboard::{FilterResult, KeyboardHandle, XkbConfig};
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorImageStatus, MotionEvent, PointerHandle,
};
use smithay::utils::SERIAL_COUNTER;

mod compositor;
mod config;
mod cursor;
mod data_device;
#[cfg(feature = "debug-ctl")]
mod debug_ctl;
mod ext_workspace;
mod fractional_scale;
mod idle_inhibit;
mod input;
mod layer_shell;
mod linux_dmabuf;
mod output;
mod presentation_time;
mod primary_selection;
mod render;
mod seat;
mod session;
mod shm;
mod subcompositor;
mod viewporter;
mod xdg_decoration;
mod xdg_output;
mod xdg_shell;

use compositor::{SolSurfaceData, SurfaceRole};
use input::{AxisSource, InputEvent, InputState};
use render::Canvas;
use wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::xdg::decoration::zv1::server::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_protocols::wp::fractional_scale::v1::server::wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1;
use wayland_protocols::wp::presentation_time::server::wp_presentation::WpPresentation;
use wayland_protocols::wp::viewporter::server::wp_viewporter::WpViewporter;
use wayland_protocols::xdg::xdg_output::zv1::server::zxdg_output_manager_v1::ZxdgOutputManagerV1;
use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::ZwlrLayerShellV1;
use wayland_server::protocol::wl_data_device_manager::WlDataDeviceManager;
use wayland_protocols::ext::workspace::v1::server::ext_workspace_manager_v1::ExtWorkspaceManagerV1;
use wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1;
use smithay::wayland::idle_inhibit::{IdleInhibitHandler, IdleInhibitManagerState};

/// Hand out a fresh, never-recycled key for the DRM presenter's
/// texture cache. Each `shm::BufferData` and `linux_dmabuf::DmabufBuffer`
/// pulls one at construction. The previous scheme (`(self as *const _)
/// as u64`) recycled keys whenever the heap allocator put a new buffer
/// at a freed buffer's address before the eviction queue drained,
/// which is exactly what happens during rapid resize churn — moving a
/// tile across columns repeatedly creates+destroys buffers fast — and
/// surfaces visually as two windows briefly rendering the same content.
/// 64 bits is enough headroom that we'll never wrap in any realistic
/// session lifetime.
pub(crate) fn next_buffer_cache_key() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

const OUTPUT_VERSION: u32 = 4;
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
/// pixels: animation interpolation (`render_rect`, `velocity`) and
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
///   `tick_animations` runs a per-component spring (x, y, w, h) that
///   pulls `render_rect` toward `rect` with the configured stiffness
///   and damping. The buffer is GPU-scaled to `render_rect`
///   regardless of its intrinsic size — during the tween it's
///   stretched smoothly, and the eye reads the scaling as resize
///   motion rather than artifact. Mid-flight target changes (rapid
///   move / close cascades) preserve the current `velocity` instead
///   of restarting from `t=0`, so the motion stays continuous.
/// - `velocity`: per-component spring velocity, in pixels-per-second.
///   Carries momentum across `apply_layout` calls; settled when the
///   spring decays under threshold.
/// - `pending_size` is the (w, h) most recently sent to the client via
///   `xdg_toplevel.configure`; layout sends a fresh configure only
///   when the target rect differs, so we don't spam configures every
///   frame.
pub struct Window {
    pub surface: Weak<WlSurface>,
    pub rect: Rect,
    pub render_rect: RectF,
    pub velocity: RectF,
    /// Alpha multiplier for the open animation: starts at 0.0 on
    /// first map, springs to 1.0. Static at 1.0 once settled. The
    /// overall window alpha emitted to the scene is
    /// `ws_alpha * render_alpha * (focused ? 1.0 : inactive_alpha)`.
    pub render_alpha: f32,
    pub vel_alpha: f32,
    /// Uniform scale around `render_rect`'s center for the open
    /// animation: starts at 0.7, springs to 1.0. The drawn rect is
    /// `render_rect` scaled by this factor about its center, so
    /// the effect is "pops out from a smaller copy of itself"
    /// rather than from a corner.
    pub render_scale: f32,
    pub vel_scale: f32,
    /// Alpha multiplier for this tile's focus-border ring. Springs
    /// to `1.0` while the tile is the focused tile (and not
    /// fullscreened), to `0.0` otherwise — so focus changes
    /// produce a cross-fade between the old and new border instead
    /// of an instant swap. Initial `0.0` so newly mapped tiles
    /// fade their border in alongside the open animation.
    pub border_alpha: f32,
    pub vel_border_alpha: f32,
    /// Set on both tiles involved in a `move_dir(Up | Down)` swap.
    /// While true, `tick_animations` overrides the layout spring
    /// with the dedicated swap spring (`spring_stiffness_swap` /
    /// `spring_damping_swap`) for all four rect components, so a
    /// vertical reorder of two stacked tiles has its own feel
    /// distinct from horizontal master/stack swaps and from
    /// resize-driven motion. Cleared once the tile's springs
    /// settle, so a follow-up non-swap motion uses the regular
    /// (per-axis) layout spring.
    pub swap_active: bool,
    /// Most recent `(width, height)` configured to the client via
    /// `xdg_toplevel.configure`. Cached so steady-state render
    /// ticks don't re-send configures when the layout target's
    /// dims haven't changed.
    pub pending_size: Option<(i32, i32)>,
    /// True after `apply_layout` changed `rect` but before the client
    /// committed a buffer at the new configured size. While set,
    /// `tick_animations` holds `render_rect` at its previous value
    /// (no snap, no tween) so we keep drawing the existing buffer at
    /// its actual size — the alternative is scaling an old-size
    /// buffer up to a growing rect and then popping in the new buffer
    /// mid-tween, which is the visible glitch on close-then-expand.
    /// Cleared in the commit handler the moment the buffer dims
    /// match `pending_size`; that's also where the tween is kicked
    /// off if `render_rect != rect`. Effectively makes the layout
    /// transition follow the protocol's configure → ack → buffer
    /// handshake instead of front-running it.
    pub pending_layout: bool,
    /// Which workspace this window belongs to. 1-based. A window
    /// renders and receives input only when its workspace matches
    /// `State.active_ws`; otherwise it stays mapped but hidden.
    pub workspace: u32,
}

/// A floating window — either a transient dialog anchored to a
/// parent tile (save/discard prompts, file pickers) or an unparented
/// fixed-size window (splash screens, "About" boxes, password
/// prompts). Lives outside the master-stack layout: no `rect` /
/// `velocity` / animation state, no configure round-trip — the
/// client picked its own size on its initial configure (we send
/// `(0,0)` for the initial), so its current buffer dims drive the
/// rendered size.
///
/// Position is centered: over the parent's `render_rect` when
/// `parent` is `Some`, on the workspace's screen rect otherwise.
/// `position` overrides this when the user has dragged the dialog
/// (xdg_toplevel.move) — top-left in screen coords, persists until
/// the dialog unmaps.
#[derive(Debug, Clone)]
pub struct DialogWindow {
    pub surface: Weak<WlSurface>,
    /// `None` → free-floater (splash, unparented modal), centered on
    /// the screen. `Some` → transient, centered over the parent's
    /// current render_rect so it follows the parent through tile
    /// resize tweens.
    pub parent: Option<Weak<WlSurface>>,
    /// Workspace the dialog belongs to — inherited from the parent
    /// at map time, or the active workspace for parentless floats.
    /// We don't track parent-workspace changes; if the user moves
    /// the parent across workspaces while a dialog is open, the
    /// dialog stays on the original workspace.
    pub workspace: u32,
    /// User-overridden top-left in screen coords, set by an
    /// interactive drag (`xdg_toplevel.move`). When `None` we
    /// recompute the centered rect every frame.
    pub position: Option<(f32, f32)>,
}

/// State of a compositor-driven interactive window drag, kicked off
/// by `xdg_toplevel.move` and ended by the next pointer button
/// release. While set, pointer motion updates the dragged dialog's
/// `position` and pointer events are not dispatched to clients
/// normally — the compositor "captures" the pointer for the move.
#[derive(Debug, Clone)]
pub struct DialogDrag {
    pub surface: Weak<WlSurface>,
    /// Vector from the dragged dialog's top-left to the cursor at
    /// drag start; constant for the duration of the drag so motion
    /// deltas don't accumulate floating-point error or drift on the
    /// first pointer event after the press.
    pub offset: (f32, f32),
}

/// A tile that's been removed from the layout but is still on screen
/// while its close animation plays out. Holds a strong ref to the
/// last `WlBuffer` so the surface dying doesn't take the buffer with
/// it; we keep rendering that buffer while `render_alpha` springs
/// toward 0 and `render_scale` toward ~0.7. Dropped from
/// `state.closing_windows` once the spring settles below the
/// invisible threshold.
#[derive(Debug)]
pub struct ClosingWindow {
    pub buffer: wayland_server::protocol::wl_buffer::WlBuffer,
    /// Static through the close animation — the rect doesn't move,
    /// only the scale around its center and the alpha change.
    pub render_rect: RectF,
    pub workspace: u32,
    pub render_alpha: f32,
    pub vel_alpha: f32,
    pub render_scale: f32,
    pub vel_scale: f32,
    /// Optional viewport src — preserved so the same crop the live
    /// window had keeps working through the fade.
    pub vsrc: Option<(f64, f64, f64, f64)>,
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
    /// Owned fence FD produced by the most recent DRM render, parked
    /// here in transit between `render_tick_inner` (which can't
    /// register a calloop source while it holds `&mut comp.backend`)
    /// and the source-registration pass that runs immediately after.
    /// Always `None` outside of `render_tick_inner`.
    pub pending_fence_fd: Option<std::os::fd::OwnedFd>,
    pub started: Instant,
    pub next_serial: u32,
    pub cursor: Cursor,
    pub input: Option<InputState>,
    /// Smithay's seat state. Owns the wl_seat global, the per-keyboard
    /// xkb context, and the per-pointer focus tracking.
    pub seat_state: SeatState<State>,
    /// The compositor's single seat. Cloneable; we stash it here so
    /// `apply_input` and helpers can reach it without threading.
    pub seat: Seat<State>,
    /// `KeyboardHandle` returned by `seat.add_keyboard`. None until
    /// libinput is wired up (DRM backend); headless never adds one.
    pub keyboard: Option<KeyboardHandle<State>>,
    /// `PointerHandle` returned by `seat.add_pointer`. Same gating
    /// as `keyboard`.
    pub pointer: Option<PointerHandle<State>>,
    /// Surface currently under the cursor, if any. Mirror of smithay's
    /// per-pointer focus that we update on every `pointer.motion` so
    /// readers (popup-grab dismiss, focus-on-click, etc.) don't have
    /// to crack open the PointerHandle.
    pub pointer_focus: Option<WlSurface>,
    /// Surface currently receiving keyboard events. Mirror of smithay's
    /// keyboard focus, written from `set_keyboard_focus`.
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
    /// Currently-pressed evdev keycodes (post-`config.remap`). Mirrors
    /// xkb's internal key-state and gets passed to `wl_keyboard.enter`
    /// so a newly-focused client knows what's *physically* held.
    /// Without this, a client receiving `enter(keys=[])` followed by
    /// `modifiers(depressed=Alt)` sees those two events as
    /// contradictory and may treat Alt as sticky/latched —
    /// symptomatically, hovering rofi after Alt+D and pressing Enter
    /// got interpreted as Alt+Enter, firing the alacritty bind
    /// instead of selecting in rofi.
    pub pressed_keys: std::collections::HashSet<u32>,
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
    /// Master/stack split ratio in [0.1, 0.9] — fraction of the
    /// usable width given to the master tile. Adjusted live in
    /// `resize_mode` (Alt+R then H/L). Persists across config
    /// reloads but resets on restart.
    pub master_ratio: f32,
    /// True while a modal resize loop is active: H/L tweak
    /// `master_ratio`, Escape exits, all other keys are swallowed
    /// (we do NOT forward them to the focused client). Entered via
    /// the `resize_mode` action; intentionally modal so the user
    /// can rapid-fire adjustments without holding a chord.
    pub resize_mode: bool,
    /// If Some, the tile currently fullscreened to the raw output
    /// rect — no outer gaps, no border, no rounded corners, drawn
    /// above Top-layer surfaces so it covers waybar etc. Overlay
    /// layers (lockscreen, OSD) still draw on top. Mutually
    /// exclusive with `zoomed`: toggling either clears the other.
    /// Same lifecycle rules as `zoomed`.
    pub fullscreened: Option<Weak<WlSurface>>,
    /// Keys of texture cache entries the DRM presenter should evict on
    /// the next render tick. Filled by `BufferHandler::buffer_destroyed`
    /// (SHM, via smithay) and the `Dispatch<WlBuffer, DmabufBuffer>`
    /// `Destroy` handler when a client tears down a buffer; drained
    /// (and acted on) inside `render_tick` which has `&mut` access to
    /// the backend.
    pub pending_texture_evictions: Vec<u64>,
    /// Smithay's `wl_shm` state: owns the global, the per-pool mmap,
    /// and the per-buffer wire dispatch. Created in `setup_event_loop`
    /// before the rest of `Globals` so its `global()` ID can populate
    /// `globals.shm`.
    pub shm_state: ShmState,
    /// Smithay's compositor state: owns the wl_compositor +
    /// wl_subcompositor globals, plus the per-surface tree
    /// (`SurfaceAttributes`, role, `cached_state`, `data_map` slot
    /// for our `SolSurfaceData`).
    pub compositor_state: CompositorState,
    /// Per-`wl_buffer` compositor-owned state, keyed by the buffer's
    /// `ObjectId`. Lazily populated at commit time in `compositor.rs`
    /// (see `shm::is_shm_buffer` gate); entries removed by
    /// `BufferHandler::buffer_destroyed`. See `shm.rs` for the
    /// rationale on why a side table beats trying to share the
    /// wl_buffer's user-data slot with smithay.
    pub shm_cache: std::collections::HashMap<ObjectId, shm::ShmCacheEntry>,
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
    /// Wall-clock of the previous `tick_animations` call. Used to
    /// compute `dt` for the spring integrator. `None` means "the
    /// next tick is the first one" — start with dt=0 (no motion
    /// step), just record the timestamp.
    pub last_anim_tick: Option<Instant>,
    /// Tiles in their close animation. Each holds a strong WlBuffer
    /// ref so the surface destruction underneath them doesn't drop
    /// the pixels. Walked in `tick_animations` (alpha+scale springs
    /// toward 0/0.7) and `collect_scene` (drawn over the same
    /// z-layer as live tiles); pruned once a window's alpha falls
    /// below the invisible threshold and velocities decay.
    pub closing_windows: Vec<ClosingWindow>,
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
    /// Surfaces with at least one live inhibitor. Non-empty →
    /// video players / presentation tools have asked us not to
    /// idle-blank. The idle timer consults this (via
    /// `idle_inhibit::any_active`) before DPMS-off. Smithay's
    /// `IdleInhibitHandler::inhibit` / `uninhibit` add and remove
    /// entries; dead `Weak`s are pruned lazily by `any_active`.
    pub idle_inhibitors: Vec<Weak<WlSurface>>,
    /// Smithay's idle-inhibit manager state (owns the global). The
    /// `delegate_idle_inhibit!` macro routes Dispatch into this; the
    /// only reason it lives on `State` is to keep the global ID
    /// reachable for shutdown.
    pub idle_inhibit_manager_state: IdleInhibitManagerState,
    /// Monotonic timestamp of the most recent user input. Updated
    /// on every `apply_input` call; the idle timer compares it to
    /// `now - config.idle_timeout` to decide whether to blank. Also
    /// re-set to `now` whenever the screen wakes, so a wake doesn't
    /// immediately re-arm another blank countdown.
    pub last_input_at: Instant,
    /// Timestamp of the last page-flip-complete event we saw. Used to
    /// compute frame-to-frame intervals for `Metrics::frame_interval_ms_buckets`
    /// — the difference between two consecutive flips is the actual
    /// refresh cadence the user sees on screen, which matters more
    /// than wall-clock render time once the GPU is overlapping with
    /// scan-out.
    pub last_flip_at: Option<Instant>,
    /// Last time the periodic perf log line was emitted (env-var-gated
    /// via `SOL_PERF_LOG_INTERVAL_MS`). Snapshot of `metrics` taken at
    /// emit time lets the next line print deltas, not lifetime totals.
    pub last_perf_log_at: Option<Instant>,
    /// Snapshot of `metrics` at the previous perf-log emit. The
    /// difference between this and the current value is what gets
    /// logged — averages over the most recent interval rather than
    /// since startup.
    pub last_perf_log_metrics: Option<Box<Metrics>>,
    /// Period in ms between automatic perf-log emits, set from
    /// `SOL_PERF_LOG_INTERVAL_MS` at startup. `None` means disabled
    /// (the default — keeps quiet in interactive sessions).
    pub perf_log_interval_ms: Option<u64>,
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
    /// Every live `wl_data_device` clients have bound. Selection
    /// changes broadcast to all of them. Cleaned of dead resources
    /// each time `set_selection` runs.
    pub data_devices: Vec<wayland_server::protocol::wl_data_device::WlDataDevice>,
    /// Source backing the current clipboard selection (Ctrl+C). The
    /// owning client streams paste contents on demand via
    /// `wl_data_source.send`. Cleared when the source destroys
    /// itself or another client replaces the selection.
    pub selection_source:
        Option<wayland_server::protocol::wl_data_source::WlDataSource>,
    /// `zwp_primary_selection_device_v1`s clients have bound.
    /// Counterpart of `data_devices` for the X11-style middle-click
    /// primary selection.
    pub primary_devices: Vec<
        wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
    >,
    /// Source backing the primary (middle-click) selection.
    pub primary_selection_source: Option<
        wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1,
    >,
    /// Mapped dialog/transient toplevels — `xdg_toplevel`s that
    /// declared a parent via `set_parent` before mapping. Live
    /// outside the master-stack tile layout: they self-size from
    /// their own buffer and float centered over the parent's tile.
    /// Drawn above tiles + Top-layer surfaces, below popups + Overlay.
    /// Pruned of dead-surface entries each render tick. Cleared
    /// alongside their parent if the parent unmaps.
    pub mapped_dialogs: Vec<DialogWindow>,
    /// In-progress interactive dialog move kicked off by
    /// `xdg_toplevel.move`. `None` means no drag — pointer events
    /// flow normally. `Some` captures the pointer to a specific
    /// dialog until the next button release.
    pub dragging: Option<DialogDrag>,
    /// Mapped xdg_popups in creation order (topmost last). Each entry
    /// is a weak ref to the popup's wl_surface; per-popup data
    /// (parent, offset, size) lives on `SurfaceData`. Dead weaks are
    /// pruned on render tick. Drawn after toplevels and after
    /// Top/Overlay layers so context menus reliably appear over
    /// every other surface.
    pub mapped_popups: Vec<Weak<WlSurface>>,
    /// The most recently grabbing xdg_popup, if any. A click outside
    /// this popup's chain dismisses it (and every popup above it on
    /// the parent stack) by sending `popup_done`. Set by
    /// `xdg_popup.grab`, cleared when the grabbing popup is
    /// destroyed.
    pub popup_grab: Option<
        Weak<wayland_protocols::xdg::shell::server::xdg_popup::XdgPopup>,
    >,
    /// Per-workspace last-focused toplevel. Saved when leaving a
    /// workspace via `switch_workspace`, consulted on return so the
    /// user lands back on the same window they were on. Pruned in
    /// `unmap_toplevel` and on workspace move so dead/relocated
    /// surfaces don't sit around as ghosts in this map.
    pub last_focus_per_workspace: std::collections::HashMap<u32, Weak<WlSurface>>,
    /// Per-workspace memory of the last keyboard-focused STACK
    /// tile (any toplevel that wasn't the master tile in its
    /// workspace at focus time). Used by `focus_direction(Right)`
    /// from master so going master → stack returns to the tile the
    /// user was on, not whatever's geometrically closest. Updated
    /// in `set_keyboard_focus` whenever focus lands on a non-
    /// master tile; entries are weak so tiles that close or move
    /// away naturally drop out on lookup.
    pub last_stack_focus_per_workspace: std::collections::HashMap<u32, Weak<WlSurface>>,
    /// Per-workspace memory of the tile that was demoted from
    /// master to stack on the most recent `move_dir(Left)` from a
    /// stack tile. `move_dir(Right)` from the new master uses this
    /// to swap the pair back, so a Left/Right loop is a true noop
    /// instead of geometric drift. Overwritten on each new
    /// Left-from-stack; weak ref so it self-invalidates.
    pub last_master_swap_partner_per_workspace:
        std::collections::HashMap<u32, Weak<WlSurface>>,
    /// Per-workspace memory of the zoomed tile. `state.zoomed` always
    /// reflects the active workspace's zoom (so the rest of the code
    /// doesn't have to thread `active_ws` through every check); on
    /// `switch_workspace` we save it here under the workspace we're
    /// leaving and restore the entry for the workspace we're entering,
    /// so a round-trip preserves zoom instead of dropping back to the
    /// tile layout.
    pub zoomed_per_workspace: std::collections::HashMap<u32, Weak<WlSurface>>,
    /// Per-workspace memory of the fullscreened tile. Same lifecycle
    /// as `zoomed_per_workspace` — switch saves the leaving
    /// workspace's mode and restores the arriving workspace's mode.
    pub fullscreened_per_workspace:
        std::collections::HashMap<u32, Weak<WlSurface>>,
    /// Cumulative compositor metrics. Always populated (overhead is
    /// a handful of integer adds per frame); the `debug-ctl` feature
    /// surfaces these via the control socket so a benchmark harness
    /// can read them out without scraping log lines.
    pub metrics: Metrics,
    /// Control-socket state when sol is built with `--features
    /// debug-ctl`. Bench harness writes line-delimited JSON commands
    /// here; sol writes line-delimited responses back. None in
    /// normal release builds (the field stays for type uniformity
    /// but never gets populated).
    #[cfg(feature = "debug-ctl")]
    pub debug_ctl: Option<debug_ctl::DebugCtl>,
}

/// Cumulative counters covering one sol process's lifetime. Reset
/// only on restart. The `frame_time_ms_buckets` histogram is
/// intentionally coarse — six buckets is enough to spot regressions
/// (everything under 8ms vs everything spilling into 33ms+) without
/// turning the dump into a CSV.
///
/// `frames_rendered` counts only ticks where real work happened —
/// `ticks_skipped` covers the early-bail path where a page flip was
/// still pending and we re-armed `needs_render`. Without this split
/// the histogram was diluted by sub-microsecond no-ops in the `<8ms`
/// bucket, hiding the real distribution.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub frames_rendered: u64,
    pub ticks_skipped: u64,
    /// Splits `ticks_skipped` by reason. `_pending_render` means the
    /// last submit's GPU work is still finishing (sync-FD not yet
    /// signalled — GPU is the bottleneck). `_pending_flip` means the
    /// kernel still hasn't completed the previous page-flip (scan-out
    /// / vblank is the bottleneck). Should sum to `ticks_skipped`.
    pub ticks_skipped_pending_render: u64,
    pub ticks_skipped_pending_flip: u64,
    pub page_flips: u64,
    pub render_tick_total_ns: u64,
    pub render_tick_max_ns: u64,
    /// Frame-time histogram tuned for high-refresh analysis. Boundaries
    /// (in ms): [<2, <3, <4, <5, <8, <16, <33, >=33].
    pub frame_time_ms_buckets: [u64; 8],
    /// Histogram of intervals between consecutive page-flip-complete
    /// events — the *real* refresh cadence the user sees. Boundaries
    /// (in ms): [<3.5, <4.0, <4.5, <5.0, <6.0, <8.5, <16.7, >=16.7].
    /// At 240Hz the budget per slot is 4.16ms; at 144Hz 6.94ms; at
    /// 60Hz 16.67ms. Most frames should park in the bucket matching
    /// the connector's refresh rate.
    pub frame_interval_ms_buckets: [u64; 8],
    /// Per-phase wall-clock totals, summed over every real render.
    pub phase_prune_ns: u64,
    pub phase_layout_ns: u64,
    pub phase_animations_ns: u64,
    pub phase_collect_scene_ns: u64,
    pub phase_render_ns: u64,
    /// Sub-breakdown of `phase_render_ns` for the DRM backend. Each
    /// part is the total CPU time spent in that phase across all
    /// frames; headless leaves them at zero.
    pub phase_render_textures_ns: u64,
    pub phase_render_blur_ns: u64,
    pub phase_render_draw_ns: u64,
    pub phase_render_present_ns: u64,
    /// Sub-breakdown of `phase_render_present_ns`. With the Vulkan
    /// pipeline, `present_swap_buffers_ns` actually measures
    /// `vkQueueSubmit`, and `present_lock_front_ns` is either the
    /// fence wait (sync fallback) or near-zero (deferred path).
    pub phase_render_present_swap_buffers_ns: u64,
    pub phase_render_present_lock_front_ns: u64,
    pub phase_render_present_add_fb_ns: u64,
    pub phase_render_present_page_flip_ns: u64,
    /// CPU sub-splits inside `render_scene`'s prologue / epilogue.
    pub phase_render_cpu_wait_fence_ns: u64,
    pub phase_render_cpu_queue_submit_ns: u64,
    pub phase_render_cpu_export_sync_fd_ns: u64,
    /// GPU-side phase totals from VK timestamp queries. Lag the CPU
    /// totals by one frame (we read frame N's GPU times during frame
    /// N+1). For "is the GPU the bottleneck?" compare
    /// `gpu_total_ns / frames_rendered` against the refresh-rate
    /// budget (4.16ms @ 240Hz).
    pub gpu_uploads_ns: u64,
    pub gpu_blur_ns: u64,
    pub gpu_draw_ns: u64,
    pub gpu_total_ns: u64,
    /// Per-frame operation count totals, summed over every real
    /// render. Divide by `frames_rendered` for averages.
    pub n_textured_draws: u64,
    pub n_solid_draws: u64,
    pub n_backdrop_draws: u64,
    pub n_blur_passes: u64,
    pub n_pipeline_binds: u64,
    pub n_descriptor_binds: u64,
    pub n_shm_uploads: u64,
    pub n_shm_upload_bytes: u64,
    /// Running maximum of any single SHM upload's byte count since the
    /// last perf-log emit. Reset to 0 each time the perf log line
    /// fires (so the displayed value reflects the largest upload in
    /// the just-reported interval, not all-time). Lifetime peak is
    /// not tracked separately — add if needed.
    pub n_shm_upload_max_bytes_running: u64,
    pub n_dmabuf_imports_new: u64,
    /// Last sample of the texture-cache size (gauge, not counter).
    pub last_textures_cached_total: u32,
    pub last_textures_cached_dmabuf: u32,
    /// Last sample of swap-slot occupancy. Persistent gauge so the
    /// most-recent snapshot is always inspectable.
    pub last_slot_scanned: u32,
    pub last_slot_pending: u32,
    pub last_slot_free: u32,
    pub texture_uploads: u64,
    pub texture_evictions: u64,
    pub spring_ticks: u64,
    pub input_events: u64,
    pub ctl_commands: u64,
    pub clients_connected: u64,
}

impl Metrics {
    /// Slot a render-tick wall-clock duration into the histogram +
    /// total. Caller measures around `render_tick`.
    pub fn record_frame(&mut self, dur_ns: u64) {
        self.frames_rendered += 1;
        self.render_tick_total_ns = self.render_tick_total_ns.saturating_add(dur_ns);
        if dur_ns > self.render_tick_max_ns {
            self.render_tick_max_ns = dur_ns;
        }
        let ms = dur_ns / 1_000_000;
        let idx = if ms < 2 { 0 }
            else if ms < 3 { 1 }
            else if ms < 4 { 2 }
            else if ms < 5 { 3 }
            else if ms < 8 { 4 }
            else if ms < 16 { 5 }
            else if ms < 33 { 6 }
            else { 7 };
        self.frame_time_ms_buckets[idx] += 1;
    }

    /// Slot a frame-to-frame interval (time between consecutive
    /// page-flip-complete events) into the cadence histogram. Boundaries
    /// in ms: [<3.5, <4.0, <4.5, <5.0, <6.0, <8.5, <16.7, >=16.7].
    /// At 240Hz a healthy stream parks in bucket 1 (<4.5ms) or 2; bucket
    /// 4 (<6ms) means an occasional miss; anything ≥ bucket 5 means
    /// the kernel waited an extra vblank.
    pub fn record_frame_interval(&mut self, ns: u64) {
        // Compare in microseconds so the half-millisecond boundaries
        // don't get rounded away.
        let us = ns / 1_000;
        let idx = if us < 3_500 { 0 }
            else if us < 4_000 { 1 }
            else if us < 4_500 { 2 }
            else if us < 5_000 { 3 }
            else if us < 6_000 { 4 }
            else if us < 8_500 { 5 }
            else if us < 16_700 { 6 }
            else { 7 };
        self.frame_interval_ms_buckets[idx] += 1;
    }

    /// Fold one frame's `RenderTiming` into the cumulative totals.
    /// One place to add per-frame numbers so adding a new field doesn't
    /// require chasing every accumulation site.
    pub fn fold_render_timing(&mut self, t: &sol_core::RenderTiming) {
        self.phase_render_textures_ns += t.textures_ns;
        self.phase_render_blur_ns += t.blur_ns;
        self.phase_render_draw_ns += t.draw_ns;
        self.phase_render_present_ns += t.present_ns;
        self.phase_render_present_swap_buffers_ns += t.present_swap_buffers_ns;
        self.phase_render_present_lock_front_ns += t.present_lock_front_ns;
        self.phase_render_present_add_fb_ns += t.present_add_fb_ns;
        self.phase_render_present_page_flip_ns += t.present_page_flip_ns;
        self.phase_render_cpu_wait_fence_ns += t.cpu_wait_fence_ns;
        self.phase_render_cpu_queue_submit_ns += t.cpu_queue_submit_ns;
        self.phase_render_cpu_export_sync_fd_ns += t.cpu_export_sync_fd_ns;
        self.gpu_uploads_ns += t.gpu_uploads_ns;
        self.gpu_blur_ns += t.gpu_blur_ns;
        self.gpu_draw_ns += t.gpu_draw_ns;
        self.gpu_total_ns += t.gpu_total_ns;
        self.n_textured_draws += t.n_textured_draws as u64;
        self.n_solid_draws += t.n_solid_draws as u64;
        self.n_backdrop_draws += t.n_backdrop_draws as u64;
        self.n_blur_passes += t.n_blur_passes as u64;
        self.n_pipeline_binds += t.n_pipeline_binds as u64;
        self.n_descriptor_binds += t.n_descriptor_binds as u64;
        self.n_shm_uploads += t.n_shm_uploads as u64;
        self.n_shm_upload_bytes += t.n_shm_upload_bytes;
        if t.n_shm_upload_max_bytes > self.n_shm_upload_max_bytes_running {
            self.n_shm_upload_max_bytes_running = t.n_shm_upload_max_bytes;
        }
        self.n_dmabuf_imports_new += t.n_dmabuf_imports_new as u64;
        self.last_textures_cached_total = t.n_textures_cached_total;
        self.last_textures_cached_dmabuf = t.n_textures_cached_dmabuf;
        self.last_slot_scanned = t.slot_scanned;
        self.last_slot_pending = t.slot_pending;
        self.last_slot_free = t.slot_free;
    }
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
    /// Same role as `BufferData::upload_seq`: lets the renderer skip
    /// re-uploading the cursor sprite when its bitmap hasn't changed.
    /// The default sprite is loaded once and never mutated, so this
    /// stays at 1 forever and the GPU texture is uploaded exactly
    /// once per process.
    pub upload_seq: u64,
    /// Set when the focused client called `wl_pointer.set_cursor`
    /// during the current pointer-enter. Cleared on the next focus
    /// change because per spec the cursor association expires the
    /// moment the pointer leaves the surface that received the
    /// enter event.
    pub client_override_active: bool,
    /// `client_override_active` + `Some(s)` → render `s` as the
    /// cursor (e.g. Chrome's hand pointer over a link).
    /// `client_override_active` + `None` → cursor is hidden (e.g.
    /// Chrome over a fullscreen video, after a few seconds idle).
    /// Both flags off → render the default sprite.
    pub client_surface: Option<Weak<WlSurface>>,
    pub client_hot_x: i32,
    pub client_hot_y: i32,
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
            client_override_active: false,
            client_surface: None,
            client_hot_x: 0,
            client_hot_y: 0,
            upload_seq: 1,
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
    pub primary_selection: GlobalId,
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

        // A new TILE mapping means the user wanted to bring up
        // another window in the layout, which they can't see while
        // one tile is taking the whole screen — drop zoom /
        // fullscreen so the layout reshuffles and the new tile is
        // actually visible. Floats (dialogs, splash screens, etc.)
        // are explicitly NOT a layout change: a save prompt over a
        // fullscreened editor should keep the editor fullscreen and
        // float on top, since the dialog is a continuation of the
        // user's interaction with that window, not a separate one.
        let is_float = self
            .mapped_dialogs
            .iter()
            .any(|d| d.surface.upgrade().ok().as_ref() == Some(surface));
        if !is_float {
            self.zoomed = None;
            self.fullscreened = None;
        }

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

    /// Set keyboard focus to the given surface. Smithay's
    /// `KeyboardHandle::set_focus` does the wl_keyboard.leave +
    /// wl_keyboard.enter + wl_keyboard.modifiers wire dispatch; the
    /// per-client modifier state comes from smithay's xkb that
    /// `KeyboardHandle::input_intercept` keeps in sync with each
    /// physical key event in `apply_input`. The mirror in
    /// `state.keyboard_focus` is updated by `SeatHandler::focus_changed`
    /// (smithay calls back into us as part of `set_focus`).
    pub fn set_keyboard_focus(&mut self, new: Option<WlSurface>) {
        if surface_eq(self.keyboard_focus.as_ref(), new.as_ref()) {
            return;
        }
        if let Some(kbd) = self.keyboard.clone() {
            let serial = SERIAL_COUNTER.next_serial();
            kbd.set_focus(self, new.clone(), serial);
        } else {
            // No keyboard handle (headless): just mirror the focus
            // ourselves so the rest of the codebase reads what it
            // expects.
            self.keyboard_focus = new.clone();
        }
        // Remember the focused tile's slot per workspace if it's a
        // STACK tile (any tile that isn't the master / first entry
        // in mapped_toplevels for its workspace). `focus_direction`
        // uses this to make Right-from-master return to the same
        // stack tile the user was on, instead of whatever's
        // geometrically closest.
        if let Some(s) = new.as_ref() {
            if let Some((ws, is_master)) = self.position_in_workspace(s) {
                if !is_master {
                    self.last_stack_focus_per_workspace
                        .insert(ws, s.downgrade());
                }
            }
        }
    }

    /// If `surface` is a tiled toplevel, return its workspace and
    /// whether it's the master tile (first non-skipped entry on
    /// that workspace). `None` for layer surfaces, popups, dialogs,
    /// or anything not in `mapped_toplevels`.
    pub fn position_in_workspace(&self, surface: &WlSurface) -> Option<(u32, bool)> {
        let mut master_seen_for: std::collections::HashMap<u32, bool> =
            std::collections::HashMap::new();
        for w in self.mapped_toplevels.iter() {
            let same = w.surface.upgrade().ok().as_ref() == Some(surface);
            let entry = master_seen_for.entry(w.workspace).or_insert(false);
            if same {
                let is_master = !*entry;
                return Some((w.workspace, is_master));
            }
            *entry = true;
        }
        None
    }


    /// Send `wl_keyboard.leave` to the focused client without
    /// updating our `keyboard_focus` mirror. Used by `resize_mode`
    /// to suspend key delivery while still remembering which surface
    /// should regain focus on exit.
    ///
    /// Implementation: clear smithay's keyboard focus (sends leave),
    /// then write the pre-leave focus back into our mirror so
    /// `keyboard_resume_focused` knows where to re-target.
    pub fn keyboard_suspend_focused(&mut self) {
        let Some(focus) = self.keyboard_focus.clone() else { return };
        if !focus.is_alive() {
            return;
        }
        if let Some(kbd) = self.keyboard.clone() {
            let serial = SERIAL_COUNTER.next_serial();
            kbd.set_focus(self, None, serial);
        }
        // `focus_changed` callback wrote `keyboard_focus = None`;
        // restore the mirror so resume() finds the surface.
        self.keyboard_focus = Some(focus);
    }

    /// Counterpart of `keyboard_suspend_focused`: re-attach the
    /// keyboard to the saved focus surface. Smithay's `set_focus`
    /// sends enter + modifiers automatically with the right state.
    pub fn keyboard_resume_focused(&mut self) {
        let Some(focus) = self.keyboard_focus.clone() else { return };
        if !focus.is_alive() {
            return;
        }
        if let Some(kbd) = self.keyboard.clone() {
            // Clear our mirror first so set_focus's "no-change" check
            // (in case smithay's internal focus is also None) doesn't
            // short-circuit. Smithay's `focus_changed` will rewrite
            // the mirror to Some(focus) on success.
            self.keyboard_focus = None;
            let serial = SERIAL_COUNTER.next_serial();
            kbd.set_focus(self, Some(focus), serial);
        }
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
pub struct ClientState {
    /// Per-client compositor state for smithay's `CompositorHandler`.
    /// Used by the per-client scale shim and transaction queue plumbing.
    pub compositor: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, client_id: ClientId) {
        tracing::info!(?client_id, "client connected");
    }
    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        tracing::info!(?client_id, ?reason, "client disconnected");
    }
}

// Smithay shm wiring. `delegate_shm!` attaches the WlShm /
// WlShmPool / WlBuffer (with smithay's `ShmBufferUserData`)
// Dispatch impls; we provide `ShmHandler` to hand out the state and
// `BufferHandler` to learn about destruction. The dmabuf module
// keeps its own `Dispatch<WlBuffer, DmabufBuffer>` for the dmabuf
// kind — those two impls coexist because each WlBuffer carries its
// own user-data type that wayland-server uses to route dispatch.
impl ShmHandler for State {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl BufferHandler for State {
    fn buffer_destroyed(
        &mut self,
        buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
    ) {
        // Only fires for buffer kinds whose Dispatch routes into a
        // smithay module that calls us — currently just SHM. Dmabuf
        // eviction still flows through `Dispatch<WlBuffer,
        // DmabufBuffer>::Destroy` in linux_dmabuf.rs.
        if let Some(entry) = self.shm_cache.remove(&buffer.id()) {
            self.pending_texture_evictions.push(entry.cache_key);
        }
    }
}

smithay::delegate_shm!(State);

// Smithay compositor wiring. `delegate_compositor!` claims the
// wl_compositor / wl_subcompositor / wl_surface / wl_subsurface /
// wl_region / wl_callback Dispatch impls. We provide the handler
// and `client_compositor_state` to expose the per-client state
// stored on our `ClientState`.
impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor
    }
    fn new_surface(&mut self, surface: &WlSurface) {
        // Attach our per-surface state behind a Mutex on the smithay
        // data_map. Done lazily on first call rather than at create
        // time because smithay's data_map insert API is "set if
        // missing" and idempotent — easier to put it here than to
        // override CompositorState::create_surface.
        smithay_compositor::with_states(surface, |states| {
            states
                .data_map
                .insert_if_missing_threadsafe(|| Mutex::new(SolSurfaceData::default()));
        });
    }
    fn new_subsurface(&mut self, surface: &WlSurface, parent: &WlSurface) {
        // Mark the child's role and link it into the parent's
        // children list so the scene walker finds it when recursing
        // down from the mapped root. Smithay enforces "one role per
        // surface" via `give_role` for us, so by the time we get
        // here the role slot is already exclusive.
        compositor::with_sol_data_mut(surface, |sd| {
            sd.role = SurfaceRole::Subsurface;
            sd.subsurface_parent = Some(parent.downgrade());
        });
        compositor::with_sol_data_mut(parent, |psd| {
            psd.subsurface_children.push(surface.downgrade());
        });
    }
    fn commit(&mut self, surface: &WlSurface) {
        commit_surface(self, surface);
    }
    fn destroyed(&mut self, surface: &WlSurface) {
        // If this was a subsurface, remove it from the parent's
        // children list so the scene walker doesn't try to upgrade
        // a dead Weak every frame. Read the parent link before
        // any cleanup since it's stored on `surface`'s SolSurfaceData
        // which is also being torn down.
        let parent = compositor::with_sol_data(surface, |sd| {
            sd.subsurface_parent.as_ref().and_then(|w| w.upgrade().ok())
        })
        .flatten();
        if let Some(parent) = parent {
            let child_id = surface.id();
            compositor::with_sol_data_mut(&parent, |psd| {
                psd.subsurface_children.retain(|w| {
                    w.upgrade().ok().map(|s| s.id() != child_id).unwrap_or(false)
                });
            });
        }
    }
}

smithay::delegate_compositor!(State);

// Smithay idle-inhibit wiring. The handler bodies live in
// `idle_inhibit.rs` so the State impl here stays a one-liner — we
// don't want a second copy of the surface-tracking logic.
impl IdleInhibitHandler for State {
    fn inhibit(&mut self, surface: WlSurface) {
        idle_inhibit::on_inhibit(self, surface);
    }
    fn uninhibit(&mut self, surface: WlSurface) {
        idle_inhibit::on_uninhibit(self, surface);
    }
}

smithay::delegate_idle_inhibit!(State);

// Smithay seat wiring. `delegate_seat!` claims the wl_seat,
// wl_pointer, wl_keyboard, wl_touch Dispatch impls. We drive
// keyboard/pointer events into the corresponding handles from
// `apply_input`. `cursor_image` is smithay's notification of
// `wl_pointer.set_cursor` — replaces the SetCursor branch in our
// previous hand-rolled `Dispatch<WlPointer>` impl.
impl SeatHandler for State {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<State> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, _seat: &Seat<State>, focused: Option<&WlSurface>) {
        // Mirror smithay's keyboard focus into `state.keyboard_focus`
        // so the dozens of reader sites in the rest of this file stay
        // unchanged. `set_keyboard_focus` is the only writer that
        // calls `keyboard.set_focus(...)` (which routes here), so the
        // mirror stays consistent without us forwarding the focus
        // value twice.
        self.keyboard_focus = focused.cloned();
    }

    fn cursor_image(&mut self, _seat: &Seat<State>, image: CursorImageStatus) {
        // Replaces the `wl_pointer::Request::SetCursor` branch in the
        // pre-PR-5 seat.rs. Surface = client-supplied cursor surface;
        // Hidden = client wants the cursor invisible; Named =
        // client requested the default sprite.
        match image {
            CursorImageStatus::Surface(surface) => {
                self.cursor.client_override_active = true;
                self.cursor.client_surface = Some(surface.downgrade());
            }
            CursorImageStatus::Hidden => {
                self.cursor.client_override_active = true;
                self.cursor.client_surface = None;
            }
            CursorImageStatus::Named(_) => {
                self.cursor.client_override_active = false;
                self.cursor.client_surface = None;
            }
        }
        self.needs_render = true;
    }
}

smithay::delegate_seat!(State);

/// Backend-specific render target. Chosen at startup; does not switch at
/// runtime. `DrmPresenter` is boxed because it carries the Vulkan
/// device / GBM scan-out / texture-cache state and is a couple of
/// orders of magnitude larger than the headless canvas — without the
/// box every `BackendState`-shaped local would carry that fixed
/// overhead.
pub enum BackendState {
    Headless { canvas: Canvas, png_path: PathBuf },
    Drm(Box<DrmPresenter>),
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
    /// Loop handle for inserting / removing event sources from
    /// dispatch contexts that don't see calloop directly. Concretely:
    /// each render produces a GPU-completion fence FD that has to be
    /// registered with the loop, and the source removes itself once
    /// it fires. None on backends that never need this (headless).
    pub loop_handle: Option<LoopHandle<'static, Compositor>>,
}

/// True when this surface declared a non-zero, equal min/max size
/// via `xdg_toplevel.set_min_size` + `set_max_size`. That's the
/// strongest "I'm not a tile-friendly window" signal a client can
/// send through xdg-shell — splash screens, GTK dialogs, file
/// pickers, password prompts all do this. Caught here so we float
/// them even if `set_parent` isn't called (splash screens) or only
/// arrives after the first commit (GIMP's New Image dialog —
/// previously the surface mapped as a tile, was given half the
/// screen, then jumped to a centered float once set_parent landed,
/// producing a visible flash).
pub(crate) fn is_fixed_size_floater(sd: &compositor::SolSurfaceData) -> bool {
    let (min_w, min_h) = sd.xdg_min_size;
    let (max_w, max_h) = sd.xdg_max_size;
    min_w > 0 && min_h > 0 && min_w == max_w && min_h == max_h
}

/// Compositor-side commit handler. Smithay calls this from
/// `CompositorHandler::commit` after merging pending → current
/// surface state (buffer assignment, damage, frame callbacks, role
/// state). We:
///
/// 1. Read smithay's `SurfaceAttributes` (cached_state) and
///    `SubsurfaceCachedState` for the just-promoted values.
/// 2. Update our `SolSurfaceData::current_buffer` based on the
///    `BufferAssignment` (NewBuffer vs Removed vs no-attach) and
///    mirror the subsurface offset so renderers don't have to crack
///    open `with_states` per-frame.
/// 3. Bump `state.shm_cache.upload_seq` for SHM buffers when there's
///    a real pixel change (attach OR damage since last commit) — bare
///    commits used for ack_configure / frame-callback handshakes
///    don't invalidate any upload cache.
/// 4. Drive role-specific map/unmap transitions (XdgToplevel,
///    LayerSurface, XdgPopup), running side-effects (push to
///    `mapped_toplevels` / `mapped_dialogs` / `mapped_popups`,
///    `unmap_toplevel`, `settle_pending_layout`,
///    `send_initial_configure`, `on_toplevel_mapped`) outside any
///    lock on `SolSurfaceData` — those helpers themselves call into
///    `with_states` and would re-enter the surface mutex.
/// 5. Drain frame callbacks into `state.pending_frame_callbacks` for
///    firing post-render.
/// 6. Set `needs_render = true`.
fn commit_surface(state: &mut State, surface: &WlSurface) {
    use std::sync::atomic::Ordering;

    // Phase 1 — single pass through smithay's cached_state. Drains
    // damage and frame_callbacks, reads the (already merged) buffer
    // assignment, reads the subsurface offset, and updates our
    // SolSurfaceData under the surface's outer Mutex (which smithay
    // already holds for us inside this closure). Captures the role +
    // new current_buffer for downstream phases.
    //
    // We do NOT `.take()` the SurfaceAttributes::buffer field —
    // smithay's `merge_into` (handlers.rs:125) reads `into.buffer`
    // (the old current value) to decide whether to fire
    // `wl_buffer.release` when the next attach replaces it. Clearing
    // it here would short-circuit that release path, leading to the
    // client's swapchain running out of free buffers and the window
    // stalling (or visibly stretching while still on its old buffer).
    // Instead we mirror the assignment into our own
    // `SolSurfaceData::current_buffer` and detect "fresh attach this
    // commit" by comparing the WlBuffer ID against our previous
    // mirrored value.
    struct Snapshot {
        attached_now: bool,
        damaged_now: bool,
        frame_callbacks: Vec<wayland_server::protocol::wl_callback::WlCallback>,
        role: SurfaceRole,
        new_current_buffer: Option<wayland_server::protocol::wl_buffer::WlBuffer>,
    }
    let snap = smithay_compositor::with_states(surface, |states| {
        let mut attrs_guard = states.cached_state.get::<SurfaceAttributes>();
        let attrs = attrs_guard.current();
        let buffer_now: Option<wayland_server::protocol::wl_buffer::WlBuffer> =
            match &attrs.buffer {
                Some(BufferAssignment::NewBuffer(b)) => Some(b.clone()),
                Some(BufferAssignment::Removed) | None => None,
            };
        let damaged_now = !attrs.damage.is_empty();
        attrs.damage.clear();
        let frame_callbacks = std::mem::take(&mut attrs.frame_callbacks);
        drop(attrs_guard);

        let mut ss_guard = states.cached_state.get::<SubsurfaceCachedState>();
        let ss_loc = ss_guard.current().location;
        drop(ss_guard);

        let sd_arc = states
            .data_map
            .get::<Mutex<compositor::SolSurfaceData>>()
            .expect("SolSurfaceData missing — did new_surface run?");
        let mut sd = sd_arc.lock().unwrap();
        // Compare new vs prev to detect "fresh attach this commit".
        // Smithay's merge_into only updates current.buffer when the
        // client's pending state had a buffer assignment, so a
        // difference here always corresponds to a real attach (or a
        // NULL-attach unmap).
        let attached_now = match (sd.current_buffer.as_ref(), buffer_now.as_ref()) {
            (Some(a), Some(b)) => a.id() != b.id(),
            (None, Some(_)) | (Some(_), None) => true,
            (None, None) => false,
        };
        sd.current_buffer = buffer_now.clone();
        sd.subsurface_offset = (ss_loc.x, ss_loc.y);
        let role = sd.role.clone();
        Snapshot {
            attached_now,
            damaged_now,
            frame_callbacks,
            role,
            new_current_buffer: buffer_now,
        }
    });

    let pixels_changed = snap.attached_now || snap.damaged_now;

    // Phase 2 — bump shm_cache upload_seq when the client signalled a
    // real pixel change. The cache is currently cursor-only anyway
    // (see vk_texture.rs / project memory note on upload-skip), but
    // keeping the gate correct here means we're ready to re-enable a
    // broader skip path without compositor-side changes.
    if pixels_changed {
        if let Some(buf) = &snap.new_current_buffer {
            if crate::shm::is_shm_buffer(buf) {
                let entry = state.shm_cache.entry(buf.id()).or_default();
                entry.upload_seq.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // Phase 3 — layer-shell state promotion. This helper itself calls
    // into `with_states`, so it must run with no SolSurfaceData lock
    // held. We already dropped ours when `with_states` returned.
    if matches!(snap.role, SurfaceRole::LayerSurface { .. }) {
        crate::layer_shell::promote_layer_state(surface);
    }

    // Phase 4 — role-specific map/unmap transitions. Read role flags,
    // mutate them, and decide which side-effects to run, all under
    // one with_sol_data_mut. The actual side-effects (push to
    // mapped_*, unmap_toplevel, etc.) need `&mut state` and helpers
    // that call back into `with_states`, so we collect "what to do"
    // here and execute below.
    let has_buffer = snap.new_current_buffer.is_some();
    let mut just_mapped_toplevel = false;
    let mut should_unmap_toplevel = false;
    let mut layer_needs_configure = false;
    let mut new_window: Option<Window> = None;
    let mut new_dialog: Option<DialogWindow> = None;
    let mut popup_unmap: bool = false;

    compositor::with_sol_data_mut(surface, |sd| {
        match &mut sd.role {
            SurfaceRole::XdgToplevel { mapped } => {
                if has_buffer {
                    if !*mapped {
                        *mapped = true;
                        just_mapped_toplevel = true;
                        // Dialog vs tile fork. A toplevel floats if it
                        // has a parent OR is fixed-size (min == max).
                        let dialog_parent = sd
                            .xdg_toplevel_parent
                            .as_ref()
                            .and_then(|w| w.upgrade().ok());
                        let fixed = is_fixed_size_floater(sd);
                        if dialog_parent.is_some() || fixed {
                            tracing::info!(
                                id = ?surface.id(),
                                parent = ?dialog_parent.as_ref().map(|s| s.id()),
                                fixed,
                                "dialog mapped"
                            );
                            new_dialog = Some(DialogWindow {
                                surface: surface.downgrade(),
                                parent: dialog_parent
                                    .as_ref()
                                    .map(|s| s.downgrade()),
                                workspace: state.active_ws,
                                position: None,
                            });
                        } else {
                            tracing::info!(id = ?surface.id(), "toplevel mapped");
                            new_window = Some(Window {
                                surface: surface.downgrade(),
                                rect: Rect::default(),
                                render_rect: RectF::default(),
                                velocity: RectF::default(),
                                render_alpha: 0.0,
                                vel_alpha: 0.0,
                                render_scale: 0.7,
                                vel_scale: 0.0,
                                border_alpha: 0.0,
                                vel_border_alpha: 0.0,
                                swap_active: false,
                                pending_size: None,
                                pending_layout: false,
                                workspace: state.active_ws,
                            });
                        }
                    }
                } else if *mapped {
                    *mapped = false;
                    should_unmap_toplevel = true;
                }
            }
            SurfaceRole::LayerSurface {
                mapped,
                initial_configure_sent,
            } => {
                if !*initial_configure_sent && !has_buffer {
                    layer_needs_configure = true;
                    *initial_configure_sent = true;
                } else if has_buffer && !*mapped && *initial_configure_sent {
                    *mapped = true;
                    tracing::info!(id = ?surface.id(), "layer surface mapped");
                } else if !has_buffer && *mapped {
                    *mapped = false;
                }
            }
            SurfaceRole::XdgPopup { mapped, .. } => {
                if has_buffer && !*mapped {
                    *mapped = true;
                    tracing::info!(id = ?surface.id(), "xdg_popup mapped");
                } else if !has_buffer && *mapped {
                    *mapped = false;
                    popup_unmap = true;
                }
            }
            SurfaceRole::None | SurfaceRole::Subsurface => {}
        }
    });

    // Phase 5 — run the deferred side-effects. None of these may run
    // with a SolSurfaceData lock held; helpers like `unmap_toplevel`,
    // `settle_pending_layout`, `on_toplevel_mapped` re-enter
    // `with_states` themselves.
    if let Some(d) = new_dialog {
        state.mapped_dialogs.push(d);
    }
    if let Some(w) = new_window {
        state.mapped_toplevels.push(w);
    }
    if popup_unmap {
        state
            .mapped_popups
            .retain(|w| w.upgrade().ok().as_ref() != Some(surface));
    }
    if matches!(snap.role, SurfaceRole::XdgPopup { .. })
        && has_buffer
        && !state.mapped_popups.iter().any(|w| {
            w.upgrade().ok().as_ref() == Some(surface)
        })
    {
        state.mapped_popups.push(surface.downgrade());
    }
    if should_unmap_toplevel {
        unmap_toplevel(state, surface);
    }
    if has_buffer && !just_mapped_toplevel {
        settle_pending_layout(state, surface);
    }
    if layer_needs_configure {
        crate::layer_shell::send_initial_configure(state, surface);
    }
    if just_mapped_toplevel {
        state.on_toplevel_mapped(surface);
    }

    // Phase 6 — frame callbacks. Defer firing until the next
    // successful render: firing them here would let clients commit
    // the next buffer before we've presented the current one, so
    // they'd just over-render at full CPU speed.
    state.pending_frame_callbacks.extend(snap.frame_callbacks);
    state.needs_render = true;
}

/// Classify a mapped surface as tile vs floating dialog based on its
/// current `xdg_toplevel_parent` and min/max-size hints, and move it
/// between `mapped_toplevels` / `mapped_dialogs` if its current slot
/// disagrees. Idempotent: pre-mapping calls and already-correct
/// classifications are silent no-ops. Called from
/// `xdg_toplevel.set_parent` / `set_min_size` / `set_max_size` so
/// GTK clients that finalize their dialog identity AFTER the first
/// commit still end up floating.
pub(crate) fn reclassify_window(state: &mut State, surface: &WlSurface) {
    let in_tiles = state
        .mapped_toplevels
        .iter()
        .position(|w| w.surface.upgrade().ok().as_ref() == Some(surface));
    let in_dialogs = state
        .mapped_dialogs
        .iter()
        .position(|d| d.surface.upgrade().ok().as_ref() == Some(surface));

    let Some((parent_weak, fixed)) = compositor::with_sol_data(surface, |sd| {
        (sd.xdg_toplevel_parent.clone(), is_fixed_size_floater(sd))
    }) else {
        return;
    };
    let parent_alive = parent_weak.as_ref().and_then(|w| w.upgrade().ok());
    let should_float = parent_alive.is_some() || fixed;

    match (in_tiles, in_dialogs, should_float) {
        // Tile that should be floating → demote. Drop tile state
        // (rect, animation, pending_layout) and re-fire an
        // unconstrained (0,0,no-states) configure so the client
        // lets go of the MAXIMIZED+TILED state we'd imposed and
        // redraws at its preferred size — without this the dialog
        // stays visually huge until the client decides on its own
        // to resize.
        (Some(idx), None, true) => {
            let win = state.mapped_toplevels.remove(idx);
            state.mapped_dialogs.push(DialogWindow {
                surface: surface.downgrade(),
                parent: parent_alive.as_ref().map(|s| s.downgrade()),
                workspace: win.workspace,
                position: None,
            });
            send_unconstrained_configure(state, surface);
            state.needs_render = true;
            tracing::info!(id = ?surface.id(), "tile → dialog");
        }
        // Dialog that should be a tile → promote. Push with default
        // rect; tick_animations snaps it on the next frame
        // (render_rect.w is 0). apply_layout will send a sized +
        // MAXIMIZED + TILED configure on the next render tick.
        (None, Some(idx), false) => {
            let dlg = state.mapped_dialogs.remove(idx);
            state.mapped_toplevels.push(Window {
                surface: surface.downgrade(),
                rect: Rect::default(),
                render_rect: RectF::default(),
                velocity: RectF::default(),
                // Demoted dialog: skip the open animation since the
                // surface has already been on screen as a float; an
                // alpha-fade-in here would look jarring.
                render_alpha: 1.0,
                vel_alpha: 0.0,
                render_scale: 1.0,
                vel_scale: 0.0,
                // Promoted dialog → tile: skip the open border-fade
                // for the same reason as alpha/scale (already on
                // screen).
                border_alpha: 1.0,
                vel_border_alpha: 0.0,
                swap_active: false,
                pending_size: None,
                pending_layout: false,
                workspace: dlg.workspace,
            });
            state.needs_render = true;
            tracing::info!(id = ?surface.id(), "dialog → tile");
        }
        // Dialog staying a dialog: refresh the parent ref in case
        // set_parent re-anchored to a different toplevel, or
        // promoted from "unparented float" to "child of X".
        (None, Some(idx), true) => {
            let new_parent = parent_alive.as_ref().map(|s| s.downgrade());
            if !weak_eq(state.mapped_dialogs[idx].parent.as_ref(), new_parent.as_ref()) {
                state.mapped_dialogs[idx].parent = new_parent;
                // Re-anchoring may move the centered position;
                // also reset any user drag-position so the dialog
                // re-snaps to the new center.
                state.mapped_dialogs[idx].position = None;
                state.needs_render = true;
            }
        }
        // Tile staying a tile, or surface not yet mapped (the
        // first-commit fork in compositor.rs handles those): no-op.
        _ => {}
    }
}

fn weak_eq(a: Option<&Weak<WlSurface>>, b: Option<&Weak<WlSurface>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x.upgrade().ok() == y.upgrade().ok(),
        (None, None) => true,
        _ => false,
    }
}

fn send_unconstrained_configure(state: &mut State, surface: &WlSurface) {
    let Some((tl, xs)) = compositor::with_sol_data(surface, |sd| {
        let tl = sd.xdg_toplevel.as_ref().and_then(|w| w.upgrade().ok())?;
        let xs = sd.xdg_surface.as_ref().and_then(|w| w.upgrade().ok())?;
        Some((tl, xs))
    })
    .flatten() else {
        return;
    };
    tl.configure(0, 0, Vec::new());
    let serial = state.next_serial();
    xs.configure(serial);
}

/// Kick off a compositor-driven interactive move for a dialog —
/// invoked from `xdg_toplevel.move` (GTK clients fire this when the
/// user click-and-drags their CSD titlebar). Captures the pointer
/// until the next button release; pointer motion updates the
/// dialog's `position`. Silently ignored for tiles (their position
/// is owned by the layout, not draggable) and for surfaces that
/// aren't currently mapped as a dialog at all.
pub(crate) fn start_dialog_drag(state: &mut State, surface: &WlSurface) {
    let Some(idx) = state
        .mapped_dialogs
        .iter()
        .position(|d| d.surface.upgrade().ok().as_ref() == Some(surface))
    else {
        return;
    };
    let Some((dx, dy)) = dialog_render_origin(state, idx) else {
        return;
    };
    let cursor_x = state.cursor.pos_x as f32;
    let cursor_y = state.cursor.pos_y as f32;
    state.dragging = Some(DialogDrag {
        surface: surface.downgrade(),
        offset: (cursor_x - dx, cursor_y - dy),
    });
    tracing::debug!(id = ?surface.id(), "dialog drag started");
}

/// Compute the current top-left of the dialog at `idx` as it would
/// be rendered: user-overridden `position` if set, else centered
/// over the parent's `render_rect`, else centered on screen.
/// Returns `None` if any required input (surface, parent, logical
/// size) is missing — caller should skip drawing / hit-testing.
fn dialog_render_origin(state: &State, idx: usize) -> Option<(f32, f32)> {
    let dlg = state.mapped_dialogs.get(idx)?;
    let surface = dlg.surface.upgrade().ok()?;
    let (dw, dh) = compositor::with_sol_data(&surface, surface_logical_size).flatten()?;
    let dw_f = dw as f32;
    let dh_f = dh as f32;

    if let Some(pos) = dlg.position {
        return Some(pos);
    }

    let host_rect = match dlg.parent.as_ref().and_then(|w| w.upgrade().ok()) {
        Some(parent) => state
            .mapped_toplevels
            .iter()
            .find(|w| w.surface.upgrade().ok().as_ref() == Some(&parent))
            .map(|w| w.render_rect)?,
        None => RectF {
            x: 0.0,
            y: 0.0,
            w: state.screen_width as f32,
            h: state.screen_height as f32,
        },
    };
    Some((
        host_rect.x + (host_rect.w - dw_f) * 0.5,
        host_rect.y + (host_rect.h - dh_f) * 0.5,
    ))
}

/// Remove a toplevel from `mapped_toplevels`, and — if it was the
/// keyboard-focused one — pick the next-down-the-stack tile on the
/// same workspace as the new focus. `mapped_toplevels` is ordered
/// master-first then stack top → stack bottom, so the entry that
/// shifts up into the closed tile's index after removal is exactly
/// the one that was visually below it. We try that first, then walk
/// upward as a fallback when the closed tile sat at the bottom of
/// its workspace's slice. Also clears the closed surface from any
/// per-workspace focus memory.
pub(crate) fn unmap_toplevel(state: &mut State, surface: &WlSurface) {
    let was_focused = surface_eq(state.keyboard_focus.as_ref(), Some(surface));

    // Dialog path: floating transient, never lived in
    // mapped_toplevels. Focus falls back to the parent tile if it's
    // still alive — that's where the user was working before the
    // dialog popped up.
    if let Some(dlg_idx) = state
        .mapped_dialogs
        .iter()
        .position(|d| d.surface.upgrade().ok().as_ref() == Some(surface))
    {
        let parent_surface = state.mapped_dialogs[dlg_idx]
            .parent
            .as_ref()
            .and_then(|w| w.upgrade().ok());
        state
            .last_focus_per_workspace
            .retain(|_, w| w.upgrade().ok().as_ref() != Some(surface));
        state.mapped_dialogs.remove(dlg_idx);
        state.needs_render = true;
        if was_focused {
            state.set_keyboard_focus(parent_surface);
        }
        return;
    }

    // Tile path: pick the next-down stack tile after removal.
    let active_ws = state.active_ws;
    let removed_idx = state
        .mapped_toplevels
        .iter()
        .position(|w| w.surface.upgrade().ok().as_ref() == Some(surface));

    state
        .last_focus_per_workspace
        .retain(|_, w| w.upgrade().ok().as_ref() != Some(surface));
    // Same hygiene for the per-workspace zoom / fullscreen memory
    // — without it a workspace would stash a dead weak that the
    // next round-trip restores into `state.zoomed`, which then
    // fails to upgrade and falls through to a tile-layout flash.
    state
        .zoomed_per_workspace
        .retain(|_, w| w.upgrade().ok().as_ref() != Some(surface));
    state
        .fullscreened_per_workspace
        .retain(|_, w| w.upgrade().ok().as_ref() != Some(surface));

    // Snapshot the dying tile for the close animation: clone its
    // last buffer (Strong WlBuffer), capture render_rect + alpha
    // + scale at the moment of removal so the spring doesn't have
    // to interpolate from a stale start. The other tiles
    // simultaneously spring into the freed slot via the upcoming
    // apply_layout — no extra work for them.
    if let Some(idx) = removed_idx {
        let win = &state.mapped_toplevels[idx];
        let buffer = win.surface.upgrade().ok().and_then(|s| {
            compositor::with_sol_data(&s, |sd| {
                let buf = sd.current_buffer.clone()?;
                Some((buf, sd.viewport_src))
            })
            .flatten()
        });
        if let Some((buffer, vsrc)) = buffer {
            state.closing_windows.push(ClosingWindow {
                buffer,
                render_rect: win.render_rect,
                workspace: win.workspace,
                render_alpha: win.render_alpha,
                vel_alpha: win.vel_alpha,
                render_scale: win.render_scale,
                vel_scale: win.vel_scale,
                vsrc,
            });
        }
    }

    state
        .mapped_toplevels
        .retain(|w| w.surface.upgrade().ok().as_ref() != Some(surface));
    // Dialogs anchored to this tile lose their host — drop them so
    // they don't render in mid-air. Most clients destroy their
    // dialogs alongside the parent, but well-behaved is not
    // universally true.
    state.mapped_dialogs.retain(|d| {
        d.parent
            .as_ref()
            .and_then(|w| w.upgrade().ok())
            .as_ref()
            != Some(surface)
    });
    state.needs_render = true;

    if !was_focused {
        return;
    }
    let Some(idx) = removed_idx else { return };
    let mt = &state.mapped_toplevels;
    let next = mt
        .iter()
        .skip(idx)
        .find(|w| w.workspace == active_ws)
        .or_else(|| mt.iter().take(idx).rev().find(|w| w.workspace == active_ws))
        .and_then(|w| w.surface.upgrade().ok());
    state.set_keyboard_focus(next);
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
/// - `mode` — changing the output mode requires tearing down the
///   GBM scan-out ring + the Vulkan images imported from it and
///   rebuilding both at the new size; that work isn't wired yet.
///   We log a warning so the user knows their edit won't take
///   effect until restart.
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

    // wl_keyboard.repeat_info is allowed to be re-sent at any time;
    // smithay's `change_repeat_info` walks every bound v4+ wl_keyboard
    // and pushes the new rate/delay. Clients pick it up on their
    // next keypress — no event loop dance needed.
    if kb_repeat_changed {
        if let Some(kbd) = state.keyboard.as_ref() {
            kbd.change_repeat_info(
                state.config.keyboard_repeat_rate,
                state.config.keyboard_repeat_delay,
            );
        }
    }

    // Force a fresh layout pass so gap / border tweaks animate in.
    state.needs_render = true;
}

/// Master-stack layout. First window takes the left half (or full screen if
/// it's the only one); remaining windows split the right half evenly, top to
/// bottom in mapping order — so the most recently mapped toplevel is the
/// bottom of the stack. Pure function over window count + screen rect.
fn master_stack_layout(n: usize, screen: Rect, master_ratio: f32) -> Vec<Rect> {
    match n {
        0 => Vec::new(),
        1 => vec![screen],
        _ => {
            // Clamp guards against values outside [0.1, 0.9] making
            // either pane vanish or go negative; resize_mode
            // already clamps, this is a belt-and-braces.
            let r = master_ratio.clamp(0.1, 0.9);
            let mid = (screen.w as f32 * r).round() as i32;
            let mid = mid.clamp(1, screen.w - 1);
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

    // Fullscreen overrides everything else: the focused tile
    // expands to the raw output rect (no outer gaps, no usable-area
    // shrink — this is the path that hides waybar). collect_scene
    // skips other tiles while fullscreen is active. Same workspace
    // rules as zoom: if the fullscreened surface isn't on the
    // active workspace, treat fullscreen as cleared.
    if let Some(fs) = state.fullscreened.as_ref().and_then(|w| w.upgrade().ok()) {
        let exists = state
            .mapped_toplevels
            .iter_mut()
            .find(|w| {
                w.workspace == active_ws
                    && w.surface.upgrade().ok().as_ref() == Some(&fs)
            })
            .map(|w| set_target_rect(w, screen, now))
            .is_some();
        if exists {
            return;
        }
        state.fullscreened = None;
    }

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
    let rects = master_stack_layout(n, inner, state.master_ratio);
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

/// Update a window's target `rect`. Marks `pending_layout` so the
/// resize tween only kicks off when the client commits a buffer at
/// the size we're about to configure (see `settle_pending_layout`).
/// No-op when the rect is unchanged so we don't re-arm the wait.
fn set_target_rect(win: &mut Window, new_rect: Rect, _now: Instant) {
    if win.rect == new_rect {
        return;
    }
    win.rect = new_rect;
    win.pending_layout = true;
}

/// Scale a `RectF` uniformly by `s` about its center. `s == 1.0`
/// is a no-op and returns the same rect; small `s` shrinks the
/// rect toward its midpoint without moving its center. Used by the
/// open / close animations so the buffer pops in/out from a
/// smaller copy of itself rather than from one corner.
fn scale_about_center(r: RectF, s: f32) -> RectF {
    if s == 1.0 {
        return r;
    }
    let cx = r.x + r.w * 0.5;
    let cy = r.y + r.h * 0.5;
    let nw = r.w * s;
    let nh = r.h * s;
    RectF {
        x: cx - nw * 0.5,
        y: cy - nh * 0.5,
        w: nw,
        h: nh,
    }
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
        let win = &state.mapped_toplevels[i];
        let Ok(surface) = win.surface.upgrade() else { continue };
        let Some((tl, xs)) = compositor::with_sol_data(&surface, |sd| {
            let tl = sd.xdg_toplevel.as_ref()?.upgrade().ok()?;
            let xs = sd.xdg_surface.as_ref()?.upgrade().ok()?;
            Some((tl, xs))
        })
        .flatten() else {
            continue;
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

    // Move-only layout changes (e.g. swapping two equal-size tiles in
    // the master-stack's right column) need no configure — the client
    // already drew at the right size — so the configure→ack→commit
    // path that normally fires `settle_pending_layout` will never run
    // and `pending_layout` would stay stuck `true` forever, holding
    // `render_rect` at the old slot. Symptom: the tile doesn't visibly
    // move, and the slot it was assigned to renders as wallpaper.
    // Settle these inline now: pending_size already matches the target
    // dims, so the existing buffer is correct for the new rect — kick
    // off the position tween directly.
    for win in state.mapped_toplevels.iter_mut() {
        if !win.pending_layout {
            continue;
        }
        let target_dims = (win.rect.w, win.rect.h);
        if win.pending_size != Some(target_dims) {
            continue;
        }
        win.pending_layout = false;
        // No explicit kickoff under springs — `tick_animations`
        // naturally drives `render_rect` toward `win.rect` on the
        // next pass. Carry-over velocity from a previous tween is
        // preserved, so a move chained on top of a settling resize
        // looks continuous.
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
        /// Corner-radius in pixels. `0.0` for layer surfaces, cursor,
        /// subsurfaces; `config.corner_radius` for the toplevel's own
        /// quad. Subsurfaces stay rectangular and rely on being
        /// inset from the toplevel's corners — clipping a subsurface
        /// to the parent's rounded shape would require a stencil
        /// pass we don't have yet.
        corner_radius: f32,
    },
    Backdrop {
        rect: RectF,
        alpha: f32,
        passes: u32,
        radius: f32,
        corner_radius: f32,
    },
}

fn collect_scene(state: &State, now: Instant) -> (Vec<Placed>, usize, usize) {
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
                corner_radius: 0.0,
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
    let fullscreened = state.fullscreened.as_ref().and_then(|w| w.upgrade().ok());
    let crossfade = workspace_anim_alphas(state, now);
    let active_ws = state.active_ws;
    let focused = state.keyboard_focus.clone();
    let inactive_alpha = state.config.inactive_alpha;
    let inactive_blur = state.config.inactive_blur && state.config.inactive_alpha < 1.0;
    let inactive_blur_passes = state.config.inactive_blur_passes;
    let inactive_blur_radius = state.config.inactive_blur_radius;
    let corner_radius = state.config.corner_radius as f32;

    let emit_for_ws = |out: &mut Vec<Placed>, ws: u32, ws_alpha: f32| {
        // Render order: non-promoted tiles first, then the zoomed
        // tile last so it paints on top of any still-fading peers.
        // (When zoom isn't active, zoomed_idx stays None and the
        // single pass walks every tile in mapped_toplevels order.)
        let mut indices: Vec<usize> = Vec::with_capacity(state.mapped_toplevels.len());
        let mut zoomed_idx: Option<usize> = None;
        for (i, win) in state.mapped_toplevels.iter().enumerate() {
            if win.workspace != ws {
                continue;
            }
            let Ok(surface) = win.surface.upgrade() else { continue };
            // Fullscreen tile is drawn separately, above Top-layer
            // surfaces, so it can cover waybar etc. Skip it here.
            if let Some(fs) = &fullscreened {
                if surface == *fs {
                    continue;
                }
            }
            if let Some(zs) = &zoomed {
                if surface == *zs {
                    zoomed_idx = Some(i);
                    continue;
                }
            }
            indices.push(i);
        }
        if let Some(i) = zoomed_idx {
            indices.push(i);
        }

        for i in indices {
            let win = &state.mapped_toplevels[i];
            let Ok(surface) = win.surface.upgrade() else { continue };
            let Some((buf_opt, vsrc, vdst)) = compositor::with_sol_data(&surface, |sd| {
                if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
                    return None;
                }
                Some((sd.current_buffer.clone(), sd.viewport_src, sd.viewport_dst))
            })
            .flatten() else {
                continue;
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
                * win.render_alpha
                * if is_active { 1.0 } else { inactive_alpha };

            // Open animation scales the tile around its center,
            // so the buffer pops into view from a slightly smaller
            // copy of itself rather than appearing at full size.
            // Settled `render_scale == 1.0` is a no-op.
            let scaled = scale_about_center(win.render_rect, win.render_scale);

            if !is_active && inactive_blur {
                // Backdrop emitted with ws_alpha (not win_alpha) so
                // it covers the wallpaper fully outside crossfade,
                // and fades together with its window during one.
                // Subsurface contents don't get a separate backdrop —
                // only the toplevel rect itself. Scaled along with
                // the window so the frosted area follows the
                // open-animation shrink.
                out.push(Placed::Backdrop {
                    rect: scaled,
                    alpha: ws_alpha * win.render_alpha,
                    passes: inactive_blur_passes,
                    radius: inactive_blur_radius,
                    corner_radius,
                });
            }

            if let Some(buf) = buf_opt {
                out.push(Placed::Buffer {
                    buf,
                    rect: scaled,
                    vsrc,
                    alpha: win_alpha,
                    corner_radius,
                });
            }
            emit_subsurface_tree(
                out,
                &surface,
                scaled.x,
                scaled.y,
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

    // 2c. Closing tiles. Same z-level as live tiles (drawn over
    // any live tile that springs into the freed slot — the closing
    // copy fades + shrinks while the new occupant grows in
    // underneath, which is the visual continuity we want). Workspace
    // gating uses the captured `workspace`, not the live one, so a
    // close on workspace 1 still fades there even if the user has
    // already switched to workspace 2.
    for cw in &state.closing_windows {
        if cw.workspace != active_ws {
            continue;
        }
        let scaled = scale_about_center(cw.render_rect, cw.render_scale);
        out.push(Placed::Buffer {
            buf: cw.buffer.clone(),
            rect: scaled,
            vsrc: cw.vsrc,
            alpha: cw.render_alpha,
            corner_radius,
        });
    }

    // Border z-anchor: focus rings draw at this point in the scene
    // — i.e. above tiles + closing tiles, BELOW Top layers,
    // fullscreen tiles, floating dialogs, popups, Overlay, and the
    // cursor. Without this anchor the border pass ran after every
    // textured element and would paint yellow rings over a
    // floating window underneath, which reads as a glitch.
    let border_anchor = out.len();

    // 3. Top layer surfaces. Emitted before any fullscreen tile so
    // a fullscreened toplevel can cover them — taskbars / launchers
    // belong above tiled toplevels but below a "give me the whole
    // screen" tile.
    for ml in &layers {
        if matches!(ml.layer, Layer::Top) {
            let vsrc = surface_viewport_src(&ml.surface);
            let r: RectF = ml.rect.into();
            out.push(Placed::Buffer {
                buf: ml.buffer.clone(),
                rect: r,
                vsrc,
                alpha: 1.0,
                corner_radius: 0.0,
            });
            emit_subsurface_tree(&mut out, &ml.surface, r.x, r.y, 1.0);
        }
    }

    // 4. Fullscreen tile, if any. Drawn above Top-layer surfaces so
    // it covers them visually, but below Overlay (lockscreen / OSD
    // stay on top of everything). No backdrop, no rounded corners,
    // alpha 1.0 — this is the "raw real estate" path. The client
    // doesn't know it's fullscreen at the protocol level; we just
    // configured it at screen size.
    if let Some(fs) = &fullscreened {
        if let Some(win) = state
            .mapped_toplevels
            .iter()
            .find(|w| w.surface.upgrade().ok().as_ref() == Some(fs))
        {
            if let Some((buf, vsrc)) = compositor::with_sol_data(fs, |sd| {
                (sd.current_buffer.clone(), sd.viewport_src)
            }) {
                // Honor open animation in fullscreen too — popping
                // up scaled and fading in still reads natural even
                // at full size.
                let scaled = scale_about_center(win.render_rect, win.render_scale);
                if let Some(buf) = buf {
                    out.push(Placed::Buffer {
                        buf,
                        rect: scaled,
                        vsrc,
                        alpha: win.render_alpha,
                        corner_radius: 0.0,
                    });
                }
                emit_subsurface_tree(
                    &mut out,
                    fs,
                    scaled.x,
                    scaled.y,
                    win.render_alpha,
                );
            }
        }
    }

    // 4b. Floating dialogs (xdg_toplevels with set_parent). Self-
    // sized from their committed buffer, centered over their
    // parent's render_rect, drawn above the tile pass and any
    // fullscreen tile. Skipped for the wrong workspace, for dead
    // surfaces, and when the parent is dead (the cleanup path
    // drops orphans, but a render between unmap and cleanup is
    // possible).
    for (idx, dlg) in state.mapped_dialogs.iter().enumerate() {
        if dlg.workspace != active_ws {
            continue;
        }
        let Ok(surface) = dlg.surface.upgrade() else { continue };
        let Some((dx, dy)) = dialog_render_origin(state, idx) else { continue };
        let Some((buf, vsrc, logical)) = compositor::with_sol_data(&surface, |sd| {
            (
                sd.current_buffer.clone(),
                sd.viewport_src,
                surface_logical_size(sd),
            )
        }) else { continue };
        let Some((dw, dh)) = logical else { continue };
        let dw_f = dw as f32;
        let dh_f = dh as f32;
        if let Some(buf) = buf {
            out.push(Placed::Buffer {
                buf,
                rect: RectF { x: dx, y: dy, w: dw_f, h: dh_f },
                vsrc,
                alpha: 1.0,
                corner_radius,
            });
        }
        emit_subsurface_tree(&mut out, &surface, dx, dy, 1.0);
    }

    // 5. Mapped xdg_popups. Stacked above tiled toplevels, Top
    // layers, and any fullscreen tile so a context menu in a
    // fullscreened browser still appears in front of its content.
    // Each popup's screen origin is its parent's render origin plus
    // the popup's surface-local offset (computed from xdg_positioner
    // at GetPopup time). The popup *buffer* may extend beyond the
    // logical popup rect when the client draws drop-shadow inside
    // it — we honor xdg_surface.set_window_geometry to align the
    // geometry rect with the positioner's anchor and let the
    // shadow spill out around it. Subsurface trees beneath the
    // popup recurse the same way as toplevels, anchored to the
    // BUFFER origin (not the geometry rect) so subsurface
    // positions in surface-local coords land where the client
    // expects.
    for popup in &state.mapped_popups {
        let Ok(surface) = popup.upgrade() else { continue };
        let Some((offset, buf, vsrc, geom, buf_dims)) = compositor::with_sol_data(&surface, |sd| {
            let SurfaceRole::XdgPopup { mapped: true, offset, .. } = sd.role else {
                return None;
            };
            let buf = sd.current_buffer.clone();
            let buf_dims = buf.as_ref().and_then(surface_buffer_dims);
            Some((offset, buf, sd.viewport_src, sd.xdg_window_geometry, buf_dims))
        })
        .flatten() else { continue };
        let Some((origin_x, origin_y)) = popup_screen_origin(state, &surface) else {
            continue;
        };
        let logical_x = origin_x + offset.0 as f32;
        let logical_y = origin_y + offset.1 as f32;
        // Pull the geometry offset (shadow padding inside the
        // buffer) and the buffer dims; default to a 0,0,buf,buf
        // geometry when the client hasn't set one — that matches
        // pre-window_geometry behavior.
        let (geom_x, geom_y) = match geom {
            Some((gx, gy, _, _)) => (gx as f32, gy as f32),
            None => (0.0, 0.0),
        };
        let buffer_x = logical_x - geom_x;
        let buffer_y = logical_y - geom_y;
        let (bw, bh) = match buf_dims {
            Some((w, h)) => (w as f32, h as f32),
            None => continue,
        };
        if let Some(buf) = buf {
            out.push(Placed::Buffer {
                buf,
                rect: RectF { x: buffer_x, y: buffer_y, w: bw, h: bh },
                vsrc,
                alpha: 1.0,
                corner_radius: 0.0,
            });
        }
        emit_subsurface_tree(&mut out, &surface, buffer_x, buffer_y, 1.0);
    }

    // 6. Overlay layer surfaces — always on top, even of fullscreen.
    for ml in &layers {
        if matches!(ml.layer, Layer::Overlay) {
            let vsrc = surface_viewport_src(&ml.surface);
            let r: RectF = ml.rect.into();
            out.push(Placed::Buffer {
                buf: ml.buffer.clone(),
                rect: r,
                vsrc,
                alpha: 1.0,
                corner_radius: 0.0,
            });
            emit_subsurface_tree(&mut out, &ml.surface, r.x, r.y, 1.0);
        }
    }

    // 7. Client-supplied cursor surface, if any. Drawn last so it
    // sits above every other layer including Overlay — pointer
    // visuals must always be on top. The default sprite path in
    // `scene_from_buffers` skips itself when this fires; if the
    // client requested cursor=None (hidden), we emit nothing here
    // and the skip path keeps the screen cursor-less.
    if state.cursor.client_override_active {
        if let Some(cur) = state
            .cursor
            .client_surface
            .as_ref()
            .and_then(|w| w.upgrade().ok())
        {
            if let Some((buf, vsrc, dims)) = compositor::with_sol_data(&cur, |sd| {
                let buf = sd.current_buffer.clone();
                let dims = buf.as_ref().and_then(surface_buffer_dims);
                (buf, sd.viewport_src, dims)
            }) {
                if let (Some(buf), Some((w, h))) = (buf, dims) {
                    let cx = state.cursor.pos_x as f32
                        - state.cursor.client_hot_x as f32;
                    let cy = state.cursor.pos_y as f32
                        - state.cursor.client_hot_y as f32;
                    out.push(Placed::Buffer {
                        buf,
                        rect: RectF {
                            x: cx,
                            y: cy,
                            w: w as f32,
                            h: h as f32,
                        },
                        vsrc,
                        alpha: 1.0,
                        corner_radius: 0.0,
                    });
                    emit_subsurface_tree(&mut out, &cur, cx, cy, 1.0);
                }
            }
        }
    }
    (out, background_count, border_anchor)
}

fn surface_viewport_src(s: &WlSurface) -> Option<(f64, f64, f64, f64)> {
    compositor::with_sol_data(s, |sd| sd.viewport_src).flatten()
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
    // Snapshot the child list so we don't re-enter `with_states` for
    // the parent while recursing into children — smithay's surface
    // mutex is per-surface, but locking the parent and a child
    // simultaneously across nested `with_states` calls is fragile.
    let Some(children) = compositor::with_sol_data(parent, |sd| {
        sd.subsurface_children
            .iter()
            .filter_map(|w| w.upgrade().ok())
            .collect::<Vec<WlSurface>>()
    }) else {
        return;
    };
    for child in children {
        let Some((buf_opt, child_x, child_y, vsrc, logical)) =
            compositor::with_sol_data(&child, |sd| {
                let (ox, oy) = sd.subsurface_offset;
                let buf = sd.current_buffer.clone();
                (
                    buf,
                    parent_x + ox as f32,
                    parent_y + oy as f32,
                    sd.viewport_src,
                    surface_logical_size(sd),
                )
            })
        else {
            continue;
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
                // Subsurfaces stay rectangular within their parent
                // toplevel — see Placed::Buffer doc-comment for the
                // reasoning. Rounded clipping inside the parent
                // would need a stencil pass.
                corner_radius: 0.0,
            });
        }
        emit_subsurface_tree(out, &child, child_x, child_y, alpha);
    }
}

fn scene_from_buffers<'a>(
    state: &'a State,
    placed: &'a [Placed],
    background_count: usize,
    border_anchor: usize,
    cursor: &'a Cursor,
) -> Scene<'a> {
    let mut scene = Scene::new();
    scene.background_count = background_count;
    // border_anchor is in `placed` index space; since we map
    // Placed → SceneElement 1:1, the index transfers unchanged. The
    // trailing cursor element is appended after every Placed entry,
    // so it always lands above the border pass — which is what we
    // want (cursor on top).
    scene.border_anchor = border_anchor;
    for p in placed {
        let (buf, rect, vsrc, alpha, corner_radius) = match p {
            Placed::Buffer { buf, rect, vsrc, alpha, corner_radius } => {
                (buf, rect, vsrc, alpha, corner_radius)
            }
            Placed::Backdrop {
                rect,
                alpha,
                passes,
                radius,
                corner_radius,
            } => {
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
                    corner_radius: *corner_radius,
                    content: SceneContent::BlurredBackdrop {
                        passes: *passes,
                        radius: *radius,
                    },
                });
                continue;
            }
        };
        if let Some((bytes, meta)) = shm::shm_pixels(buf) {
            // Cache entry is populated lazily at commit time in
            // compositor.rs. Missing here would mean we somehow saw a
            // committed buffer that the commit handler didn't run on
            // — drop the element rather than allocating a key without
            // a destruction signal to clean it up.
            let Some(entry) = state.shm_cache.get(&buf.id()) else {
                continue;
            };
            let key = entry.cache_key;
            let upload_seq = entry
                .upload_seq
                .load(std::sync::atomic::Ordering::Relaxed);
            let (ux, uy, uw, uh) = compute_uv(*vsrc, meta.width, meta.height);
            scene.elements.push(SceneElement {
                buffer_key: key,
                width: meta.width,
                height: meta.height,
                dst_width: rect.w,
                dst_height: rect.h,
                x: rect.x,
                y: rect.y,
                uv_x: ux,
                uv_y: uy,
                uv_w: uw,
                uv_h: uh,
                alpha: *alpha,
                corner_radius: *corner_radius,
                content: SceneContent::Shm {
                    pixels: bytes,
                    stride: meta.stride,
                    format: meta.format,
                    upload_seq,
                },
            });
        } else if let Some(db) = buf.data::<linux_dmabuf::DmabufBuffer>() {
            // B10.3: single-plane dmabuf only. Multi-plane (YUV etc.) lands
            // when we care about video, not for alacritty/terminals.
            let Some(p0) = db.planes.first() else { continue };
            let key = db.cache_key;
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
                corner_radius: *corner_radius,
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
    // Default cursor sprite — drawn last so it always sits on
    // top. Skipped when the focused client has called
    // wl_pointer.set_cursor: collect_scene already pushed the
    // client-supplied surface (or, if the client explicitly asked
    // for cursor=None, pushed nothing — the pointer is hidden).
    // dst_* = 0.0 forces the presenter to render at the sprite's
    // intrinsic size.
    if cursor.visible && !cursor.client_override_active {
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
            corner_radius: 0.0,
            content: SceneContent::Shm {
                pixels: &cursor.pixels,
                stride: cursor.width * 4,
                format: sol_core::PixelFormat::Argb8888,
                upload_seq: cursor.upload_seq,
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

/// Re-export of the shared sentinel so the rest of this file can keep
/// using the unqualified name. The canonical definition lives in
/// `sol-core` since the DRM backend needs to recognise it too.
const CURSOR_SCENE_KEY: u64 = sol_core::CURSOR_SCENE_KEY;
/// Sentinel buffer_key for `BlurredBackdrop` scene elements. The
/// backdrop branch in the presenter's draw loop returns before any
/// texture lookup happens, but a value is required by the struct
/// shape — pick something else that can't collide with a real
/// BufferData / DmabufBuffer pointer.
const BACKDROP_SCENE_KEY: u64 = 0xB10B1B_DEADBEEF;

/// Step every window's render_rect spring toward its target rect.
/// Returns `true` if anything is still in motion, so the render
/// loop knows to keep ticking.
///
/// Springs run a damped harmonic oscillator per component (x, y,
/// w, h):
///
/// ```text
/// force = stiffness * (target - current)
/// damp  = -damping * velocity
/// velocity += (force + damp) * dt
/// current  += velocity * dt
/// ```
///
/// vs the previous fixed-duration eased tween, this gets us:
/// - mid-flight target changes preserve velocity instead of
///   restarting from t=0 (the rapid-move stutter goes away);
/// - settle time is implicit in the spring constants, so
///   different-magnitude targets self-pace (a tiny resize finishes
///   sooner than a full half-screen swap);
/// - slight overshoot (we run sub-critically damped) reads as
///   "alive" rather than scripted.
fn tick_animations(state: &mut State, now: Instant) -> bool {
    state.metrics.spring_ticks += 1;
    // dt for the integrator. Cap at ~33 ms so a frame missed for
    // 200 ms (e.g. a stall, VT switch, suspend) doesn't catapult
    // the spring across the screen on resume — clamp keeps each
    // step linearizable.
    let dt = match state.last_anim_tick {
        Some(prev) => now.duration_since(prev).as_secs_f32().min(1.0 / 30.0),
        None => 0.0,
    };
    state.last_anim_tick = Some(now);

    let stiffness = state.config.spring_stiffness;
    let damping = state.config.spring_damping;
    // Per-axis overrides for the y + h springs. Default: same
    // values as the base. Lets the user tune right-stack vertical
    // reorders independently of horizontal master/stack swaps.
    let stiffness_v = state
        .config
        .spring_stiffness_vertical
        .unwrap_or(stiffness);
    let damping_v = state.config.spring_damping_vertical.unwrap_or(damping);
    // Dedicated swap spring used while a tile is mid-Up/Down
    // reorder (`swap_active`). Lives outside the per-axis split so
    // the user can dial in "two stack tiles trading places" as a
    // distinct motion from any other layout change.
    let swap_k = state.config.spring_stiffness_swap;
    let swap_c = state.config.spring_damping_swap;
    // Optional override for the open/close fade springs (alpha +
    // scale on live tiles, alpha + scale on ClosingWindow). Falls
    // back to the base when unset, so existing configs see no
    // change. Use a softer fade than the layout for a "windows
    // breathe in/out" feel, or match the layout for tightness.
    let fade_k = state.config.spring_stiffness_fade.unwrap_or(stiffness);
    let fade_c = state.config.spring_damping_fade.unwrap_or(damping);
    // Settled thresholds: under half a pixel away with sub-pixel-
    // per-second velocity is indistinguishable from a snap.
    const POS_EPS: f32 = 0.5;
    const VEL_EPS: f32 = 0.5;

    // Snapshot focus + zoom + fullscreen state for the border-alpha
    // and visibility-alpha target lookups inside the loop; doing it
    // once up here means the per-window check is a couple of cheap
    // pointer compares.
    let focus = state.keyboard_focus.clone();
    let active_ws = state.active_ws;
    let zoomed = state.zoomed.as_ref().and_then(|w| w.upgrade().ok());
    let fullscreened = state.fullscreened.as_ref().and_then(|w| w.upgrade().ok());
    // Promoted-tile lookup for inactive workspaces. Each workspace
    // remembers its own zoom / fullscreen mode via the per-workspace
    // maps; without this lookup, tiles on a workspace where zoom is
    // active (but the user has switched away) would have their
    // alpha pulled back to 1.0 by the active workspace's zoom state
    // — and then yanked to 0.0 again on return, causing a flicker.
    let promoted_other_ws: std::collections::HashMap<u32, WlSurface> = state
        .zoomed_per_workspace
        .iter()
        .chain(state.fullscreened_per_workspace.iter())
        .filter_map(|(&ws, w)| {
            if ws == active_ws {
                return None;
            }
            w.upgrade().ok().map(|s| (ws, s))
        })
        .collect();

    let mut any_active = false;

    for win in state.mapped_toplevels.iter_mut() {
        let target: RectF = win.rect.into();

        // Newly mapped: snap, no animation. Clearing pending_layout
        // is fine — there's nothing to interpolate from, and a
        // commit-driven settle right after would have nothing to do.
        if win.render_rect.w == 0.0 || win.render_rect.h == 0.0 {
            win.render_rect = target;
            win.velocity = RectF::default();
            win.pending_layout = false;
            continue;
        }

        // Holding for the client to commit at the configured size.
        // No spring step — `settle_pending_layout` clears the flag
        // once the buffer arrives and the next tick takes over.
        if win.pending_layout {
            continue;
        }

        // Step the four rect springs (x, y, w, h) plus alpha and
        // scale for the open animation. While `swap_active` is set
        // (the tile is in flight from a vertical Up/Down reorder),
        // every component uses the dedicated swap spring; otherwise
        // y + h pick up the `_vertical` overrides if set, x + w
        // run the base spring.
        let (k_x, c_x, k_y, c_y) = if win.swap_active {
            (swap_k, swap_c, swap_k, swap_c)
        } else {
            (stiffness, damping, stiffness_v, damping_v)
        };
        let mut win_active = false;
        for (cur, vel, tgt, k, c) in [
            (&mut win.render_rect.x, &mut win.velocity.x, target.x, k_x, c_x),
            (&mut win.render_rect.y, &mut win.velocity.y, target.y, k_y, c_y),
            (&mut win.render_rect.w, &mut win.velocity.w, target.w, k_x, c_x),
            (&mut win.render_rect.h, &mut win.velocity.h, target.h, k_y, c_y),
        ] {
            let force = (tgt - *cur) * k;
            let damp = -*vel * c;
            *vel += (force + damp) * dt;
            *cur += *vel * dt;
            if (tgt - *cur).abs() > POS_EPS || vel.abs() > VEL_EPS {
                any_active = true;
                win_active = true;
            } else {
                *cur = tgt;
                *vel = 0.0;
            }
        }
        // Clear the swap flag once the rect springs settle so a
        // subsequent non-swap motion (resize, master ↔ stack swap)
        // uses the regular layout spring tuning instead of the
        // swap one.
        if win.swap_active && !win_active {
            win.swap_active = false;
        }
        // Alpha + scale springs use tighter epsilons since their
        // values live in [0, 1] — half-a-pixel doesn't translate
        // here.
        const ALPHA_EPS: f32 = 0.005;
        // Border-alpha target: 1.0 only for the focused, non-
        // fullscreened tile. Fullscreen suppresses the ring (the
        // mode is "raw real estate"), and unfocused tiles fade
        // their ring out. Computed inline because we need access
        // to the focus / fullscreen snapshots above.
        let surface_alive = win.surface.upgrade().ok();
        let is_focused = focus
            .as_ref()
            .zip(surface_alive.as_ref())
            .map(|(f, s)| f == s)
            .unwrap_or(false);
        // Promoted (zoom OR fullscreen) tile for THIS window's
        // workspace. Active workspace reads from the live snapshots
        // above; other workspaces read the per-workspace map.
        let promoted: Option<&WlSurface> = if win.workspace == active_ws {
            zoomed.as_ref().or(fullscreened.as_ref())
        } else {
            promoted_other_ws.get(&win.workspace)
        };
        let is_promoted = promoted
            .zip(surface_alive.as_ref())
            .map(|(p, s)| p == s)
            .unwrap_or(false);
        let is_fullscreened = fullscreened
            .as_ref()
            .zip(surface_alive.as_ref())
            .map(|(f, s)| f == s)
            .unwrap_or(false);
        let border_target = if is_focused && !is_fullscreened {
            1.0_f32
        } else {
            0.0_f32
        };
        // Visibility-alpha target. While a zoom or fullscreen tile
        // is "spotlight"-promoted on this window's workspace, every
        // other tile on that workspace fades out — the promoted one
        // keeps full opacity, untouched tiles fade to 0 and ride
        // the fade spring back to 1 once the mode clears. This
        // replaces the older "instantly skip the others" path in
        // collect_scene with a smooth crossfade.
        let alpha_target = if promoted.is_some() {
            if is_promoted { 1.0 } else { 0.0 }
        } else {
            1.0
        };
        // Fade springs (open animation: alpha + scale → 1.0) use
        // the dedicated fade tuning, border-alpha keeps the base.
        for (cur, vel, tgt, k, c) in [
            (&mut win.render_alpha, &mut win.vel_alpha, alpha_target, fade_k, fade_c),
            (&mut win.render_scale, &mut win.vel_scale, 1.0_f32, fade_k, fade_c),
            (&mut win.border_alpha, &mut win.vel_border_alpha, border_target, stiffness, damping),
        ] {
            let force = (tgt - *cur) * k;
            let damp = -*vel * c;
            *vel += (force + damp) * dt;
            *cur += *vel * dt;
            if (tgt - *cur).abs() > ALPHA_EPS || vel.abs() > ALPHA_EPS {
                any_active = true;
            } else {
                *cur = tgt;
                *vel = 0.0;
            }
        }
    }

    // Tick closing windows: alpha → 0, scale → 0.7, using the same
    // fade spring tuning as the open animation so the close mirrors
    // the open feel by default. Drop entries that have decayed
    // below the visible threshold AND have damped to a near-stop,
    // since beyond that point further ticking is invisible
    // bookkeeping.
    state.closing_windows.retain_mut(|cw| {
        const ALPHA_EPS: f32 = 0.005;
        for (cur, vel, tgt) in [
            (&mut cw.render_alpha, &mut cw.vel_alpha, 0.0_f32),
            (&mut cw.render_scale, &mut cw.vel_scale, 0.7_f32),
        ] {
            let force = (tgt - *cur) * fade_k;
            let damp = -*vel * fade_c;
            *vel += (force + damp) * dt;
            *cur += *vel * dt;
        }
        // Alive while still visibly opaque OR still moving fast
        // enough that another step will change the picture.
        let keep = cw.render_alpha > ALPHA_EPS || cw.vel_alpha.abs() > ALPHA_EPS;
        if keep {
            any_active = true;
        }
        keep
    });

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
    let t = (elapsed as f32 / duration as f32).clamp(0.0, 1.0);
    // Cubic-out: front-loaded fade with a soft settle. Slight
    // start, slow finish — reads as "thing slides in / out" rather
    // than the mechanical bisection of a linear lerp.
    let inv = 1.0 - t;
    let eased = 1.0 - inv * inv * inv;
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

/// True iff any layout tween or workspace crossfade is currently in
/// flight. The DRM page-flip-complete handler consults this to
/// decide whether to keep auto-rendering at vblank cadence (so the
/// animation actually progresses) or let the compositor go idle
/// (event-driven renders only). Cheap O(N) scan over mapped
/// toplevels — N is at most a handful and this only fires once per
/// flip event, not per frame body.
fn has_active_animation(state: &State) -> bool {
    if state.workspace_anim.is_some() {
        return true;
    }
    // Closing windows are always "active" until tick_animations
    // drops them.
    if !state.closing_windows.is_empty() {
        return true;
    }
    // A spring is "active" while either (a) render_rect hasn't
    // reached rect, or (b) velocity is non-zero (still settling
    // even if pos==target due to overshoot crossing through). Same
    // for the alpha + scale springs that drive the open animation.
    // Thresholds mirror `tick_animations` so both functions agree
    // on "settled".
    const POS_EPS: f32 = 0.5;
    const VEL_EPS: f32 = 0.5;
    const A_EPS: f32 = 0.005;
    let focus = state.keyboard_focus.clone();
    let fullscreened = state.fullscreened.as_ref().and_then(|w| w.upgrade().ok());
    state.mapped_toplevels.iter().any(|w| {
        let target: RectF = w.rect.into();
        let r = w.render_rect;
        let v = w.velocity;
        // Border-alpha target, mirroring tick_animations.
        let surf = w.surface.upgrade().ok();
        let is_focused = focus.as_ref().zip(surf.as_ref()).map(|(f, s)| f == s).unwrap_or(false);
        let is_fs = fullscreened.as_ref().zip(surf.as_ref()).map(|(f, s)| f == s).unwrap_or(false);
        let border_target = if is_focused && !is_fs { 1.0 } else { 0.0 };
        (target.x - r.x).abs() > POS_EPS
            || (target.y - r.y).abs() > POS_EPS
            || (target.w - r.w).abs() > POS_EPS
            || (target.h - r.h).abs() > POS_EPS
            || v.x.abs() > VEL_EPS
            || v.y.abs() > VEL_EPS
            || v.w.abs() > VEL_EPS
            || v.h.abs() > VEL_EPS
            || (1.0 - w.render_alpha).abs() > A_EPS
            || w.vel_alpha.abs() > A_EPS
            || (1.0 - w.render_scale).abs() > A_EPS
            || w.vel_scale.abs() > A_EPS
            || (border_target - w.border_alpha).abs() > A_EPS
            || w.vel_border_alpha.abs() > A_EPS
    })
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
/// Resolve a popup surface's screen origin by walking its parent
/// chain to the toplevel (or layer surface) that anchors the chain.
/// Returns the parent-of-popup's render origin in screen coords;
/// callers add the popup's surface-local offset and pixel-snap at
/// the end. Returns `None` if the chain leads to a non-mapped root
/// or terminates without one.
fn popup_screen_origin(state: &State, popup: &WlSurface) -> Option<(f32, f32)> {
    let parent = compositor::with_sol_data(popup, |sd| sd.xdg_popup_parent.clone())
        .flatten()?
        .upgrade()
        .ok()?;
    let role = compositor::with_sol_data(&parent, |sd| sd.role.clone())?;
    match role {
        SurfaceRole::XdgToplevel { mapped: true } => state
            .mapped_toplevels
            .iter()
            .find(|w| w.surface.upgrade().ok().as_ref() == Some(&parent))
            .map(|w| (w.render_rect.x, w.render_rect.y)),
        SurfaceRole::XdgPopup { mapped: true, offset, .. } => {
            let (gx, gy) = popup_screen_origin(state, &parent)?;
            Some((gx + offset.0 as f32, gy + offset.1 as f32))
        }
        SurfaceRole::LayerSurface { mapped: true, .. } => {
            let screen = Rect {
                x: 0,
                y: 0,
                w: state.screen_width as i32,
                h: state.screen_height as i32,
            };
            let layers = layer_shell::mapped_layers(state, screen);
            layers
                .iter()
                .find(|m| m.surface == parent)
                .map(|m| (m.rect.x as f32, m.rect.y as f32))
        }
        _ => None,
    }
}

/// Called from the wl_surface commit handler whenever a mapped
/// toplevel attaches a buffer. If the toplevel has a pending layout
/// change AND the buffer's logical size matches the pending
/// configure, clear the hold flag so `tick_animations` resumes
/// driving `render_rect` toward `win.rect`. Until this runs the
/// window keeps drawing at its old rect with its old buffer — the
/// protocol-correct "client owns its size" behaviour.
pub(crate) fn settle_pending_layout(state: &mut State, surface: &WlSurface) {
    // Pull logical size out of the surface's SurfaceData while we
    // already have a Mutex guard available; cheap and read-only.
    let logical = compositor::with_sol_data(surface, surface_logical_size).flatten();

    let mut kicked = false;
    for win in state.mapped_toplevels.iter_mut() {
        if win.surface.upgrade().ok().as_ref() != Some(surface) {
            continue;
        }
        if !win.pending_layout {
            return;
        }
        // Only settle when the client's buffer is at the size we
        // configured. If they're still on the old size — e.g. they
        // committed before processing our configure — keep waiting.
        let matches = match (logical, win.pending_size) {
            (Some(buf), Some(cfg)) => buf == cfg,
            _ => false,
        };
        if !matches {
            return;
        }
        win.pending_layout = false;
        // Spring picks up automatically on the next tick — no
        // explicit kickoff. Velocity that built up before the hold
        // (e.g. a settle-while-moving sequence) carries over.
        let target: RectF = win.rect.into();
        if win.render_rect != target {
            kicked = true;
        }
        break;
    }
    if kicked {
        state.needs_render = true;
    }
}

fn surface_logical_size(sd: &compositor::SolSurfaceData) -> Option<(i32, i32)> {
    if let Some((w, h)) = sd.viewport_dst {
        return Some((w, h));
    }
    if let Some((_, _, w, h)) = sd.viewport_src {
        return Some((w as i32, h as i32));
    }
    let buf = sd.current_buffer.as_ref()?;
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
        if best.is_none_or(|(_, s)| score < s) {
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

/// If the keyboard-focused tile is the master of its workspace AND
/// we have a remembered last-stack-tile for that workspace AND that
/// tile is still in the stack, return it. Returns `None` otherwise
/// — caller falls back to geometric `neighbor_in_direction`.
fn focus_to_remembered_stack_tile(state: &State) -> Option<WlSurface> {
    let focus = state.keyboard_focus.as_ref()?;
    let (ws, is_master) = state.position_in_workspace(focus)?;
    if !is_master {
        return None;
    }
    let saved = state.last_stack_focus_per_workspace.get(&ws)?.upgrade().ok()?;
    let (saved_ws, saved_is_master) = state.position_in_workspace(&saved)?;
    if saved_ws != ws || saved_is_master {
        return None;
    }
    Some(saved)
}

fn focus_direction(state: &mut State, dir: config::Direction) {
    // Right-from-master shortcut: prefer the last stack tile the
    // user was on over geometric "nearest". Without this,
    // master → right always picks the vertically-closest stack tile
    // (typically the middle one in a 3-tile stack), even if the
    // user just came from the top or bottom — frustrating in a
    // 2-pane workflow where you bounce between master and "the
    // tile you were just reading".
    if matches!(dir, config::Direction::Right) {
        if let Some(saved) = focus_to_remembered_stack_tile(state) {
            state.zoomed = None;
            state.fullscreened = None;
            state.set_keyboard_focus(Some(saved));
            state.needs_render = true;
            return;
        }
    }

    let Some(idx) = focused_toplevel_index(state) else { return };
    let Some(n) =
        neighbor_in_direction(&state.mapped_toplevels, idx, dir, state.active_ws)
    else {
        return;
    };
    let Ok(target) = state.mapped_toplevels[n].surface.upgrade() else {
        return;
    };
    // Moving focus while zoomed/fullscreened would leave the user
    // looking at a hidden tile, so drop both modes before shifting
    // focus. Explicit Alt+Tab / Ctrl+Tab is the only path to re-
    // enter on the new window.
    state.zoomed = None;
    state.fullscreened = None;
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
    let active_ws = state.active_ws;

    // Right-from-master inverse: if the most recent move was a
    // Left-from-stack-to-master, do the exact opposite — swap the
    // current master with the tile we just demoted. Without this,
    // geometric `neighbor_in_direction` picks the vertically-
    // closest stack tile (usually the middle one in a 3-tile
    // stack), so a Left/Right loop drifts windows instead of being
    // a noop.
    if matches!(dir, config::Direction::Right) {
        if let Some(partner_idx) = master_swap_partner_index(state, idx, active_ws) {
            state.zoomed = None;
            state.fullscreened = None;
            state.mapped_toplevels.swap(idx, partner_idx);
            state.needs_render = true;
            return;
        }
    }

    let Some(n) =
        neighbor_in_direction(&state.mapped_toplevels, idx, dir, active_ws)
    else {
        return;
    };
    // Same reasoning as focus_direction: rearranging the tile layout
    // while zoom/fullscreen hides everything but one tile would be
    // confusing.
    state.zoomed = None;
    state.fullscreened = None;
    state.mapped_toplevels.swap(idx, n);

    // After Left-from-stack-to-master, remember the demoted master
    // (now sitting at the focused tile's old slot) so the next
    // Right-from-master can swap back to it. Overwriting any
    // previous saved partner — only the most recent Left counts as
    // the inverse target.
    if matches!(dir, config::Direction::Left) && n == 0 && idx != 0 {
        if let Ok(prev_master) = state.mapped_toplevels[idx].surface.upgrade() {
            state
                .last_master_swap_partner_per_workspace
                .insert(active_ws, prev_master.downgrade());
        }
    }

    // Up/Down stack reorders run a dedicated "swap" spring with
    // its own stiffness/damping — see `tick_animations`. Mark both
    // tiles so the next animation tick picks up the override; the
    // flag self-clears on settle.
    if matches!(dir, config::Direction::Up | config::Direction::Down) {
        if let Some(w) = state.mapped_toplevels.get_mut(idx) {
            w.swap_active = true;
        }
        if let Some(w) = state.mapped_toplevels.get_mut(n) {
            w.swap_active = true;
        }
    }

    state.needs_render = true;
}

/// Resolve the saved master-swap partner to an index in
/// `mapped_toplevels` iff (a) `focus_idx` IS the master of `ws`,
/// (b) a partner is recorded for `ws` and is still alive, and
/// (c) the partner currently sits in the stack (not master).
/// `None` otherwise; the caller falls through to geometric
/// `neighbor_in_direction`. Once partner ↔ master are swapped on
/// the next call, the partner is now master and (c) fails on the
/// following call, so this naturally only "consumes" itself once.
fn master_swap_partner_index(state: &State, focus_idx: usize, ws: u32) -> Option<usize> {
    let master_pos = state
        .mapped_toplevels
        .iter()
        .position(|w| w.workspace == ws)?;
    if master_pos != focus_idx {
        return None;
    }
    let partner = state
        .last_master_swap_partner_per_workspace
        .get(&ws)?
        .upgrade()
        .ok()?;
    let (partner_ws, partner_is_master) = state.position_in_workspace(&partner)?;
    if partner_ws != ws || partner_is_master {
        return None;
    }
    state
        .mapped_toplevels
        .iter()
        .position(|w| w.surface.upgrade().ok().as_ref() == Some(&partner))
}

/// Ctrl+Q handler: send `xdg_toplevel.close` to whichever tile has
/// keyboard focus. The client decides how to handle it — terminals
/// and most CLI-ish apps exit immediately, GUI apps may show a
/// save-changes prompt. If the client chooses to ignore the event,
/// we don't escalate to SIGTERM; that's a different action.
fn close_focused_window(state: &mut State) {
    let Some(focus) = state.keyboard_focus.as_ref() else { return };
    let tl = compositor::with_sol_data(focus, |sd| {
        sd.xdg_toplevel.as_ref().and_then(|w| w.upgrade().ok())
    })
    .flatten();
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
    // Zoom and fullscreen are mutually exclusive — entering either
    // one clears the other so the user only ever has one
    // "spotlight" mode in flight.
    state.fullscreened = None;
    state.needs_render = true;
}

/// Ctrl+Tab handler (configurable): toggle fullscreen on the
/// focused tile. Unlike zoom this gives the tile the raw output
/// rect — no outer gaps, no border, no rounded corners — and
/// renders it above Top-layer surfaces so it covers waybar etc.
/// Overlay-layer surfaces (lockscreens, OSD) still draw on top.
/// We do NOT send `xdg_toplevel.set_fullscreen` to the client, so
/// it doesn't enter its own fullscreen UI mode (Chrome's controls
/// stay visible, etc.); the client just receives a screen-sized
/// configure.
fn toggle_fullscreen(state: &mut State) {
    let Some(focus) = state.keyboard_focus.as_ref() else { return };
    if !state
        .mapped_toplevels
        .iter()
        .any(|w| w.surface.upgrade().ok().as_ref() == Some(focus))
    {
        return;
    }
    let already = state
        .fullscreened
        .as_ref()
        .and_then(|w| w.upgrade().ok())
        .as_ref()
        == Some(focus);
    state.fullscreened = if already { None } else { Some(focus.downgrade()) };
    // Mutually exclusive with zoom — see toggle_zoom.
    state.zoomed = None;
    state.needs_render = true;
}

/// Switch to the given workspace. Zoom and fullscreen are remembered
/// per workspace: leaving a zoomed/fullscreened workspace stashes
/// the mode in `zoomed_per_workspace` / `fullscreened_per_workspace`,
/// and returning restores it so the user lands back in the same
/// "spotlight" view they left.
/// Keyboard focus moves to the topmost (= most recently mapped)
/// toplevel on the target workspace; if it's empty, focus drops to
/// None, which means subsequent keystrokes go nowhere until a
/// window maps or a focus command is issued.
pub(crate) fn switch_workspace(state: &mut State, n: u32) {
    if state.active_ws == n {
        return;
    }
    let old = state.active_ws;

    // Remember the focused tile on the workspace we're leaving so a
    // round-trip back lands on the same window. Only stash if focus
    // is on a tiled toplevel that actually belongs to `old` — layer
    // surfaces and popups don't belong to a workspace in the same
    // sense, and a focus that's on a tile from a different
    // workspace (corner case during rapid switches) shouldn't mask
    // the real history of `old`.
    if let Some(focus) = state.keyboard_focus.clone() {
        if state.mapped_toplevels.iter().any(|w| {
            w.workspace == old && w.surface.upgrade().ok().as_ref() == Some(&focus)
        }) {
            state.last_focus_per_workspace.insert(old, focus.downgrade());
        }
    }

    tracing::info!(from = old, to = n, "workspace switch");

    // Save the leaving workspace's zoom / fullscreen mode so a
    // round-trip preserves it. `take()` clears the live field; we
    // either insert under `old` (mode was on) or remove its entry
    // (mode was off) so the map only carries live state.
    if let Some(w) = state.zoomed.take() {
        state.zoomed_per_workspace.insert(old, w);
    } else {
        state.zoomed_per_workspace.remove(&old);
    }
    if let Some(w) = state.fullscreened.take() {
        state.fullscreened_per_workspace.insert(old, w);
    } else {
        state.fullscreened_per_workspace.remove(&old);
    }

    state.active_ws = n;

    // Restore the arriving workspace's zoom / fullscreen, dropping
    // dead weaks (tile closed since last visit). Mutually exclusive
    // by construction at toggle time, so at most one of the two
    // ends up Some.
    state.zoomed = state
        .zoomed_per_workspace
        .get(&n)
        .filter(|w| w.upgrade().is_ok())
        .cloned();
    state.fullscreened = state
        .fullscreened_per_workspace
        .get(&n)
        .filter(|w| w.upgrade().is_ok())
        .cloned();

    ext_workspace::notify_active_changed(state, old, n);
    // Windows coming back into view get the "first-frame" treatment
    // in `tick_animations` — render_rect.w/h == 0 means "snap to the
    // assigned rect, no interpolation" — so the workspace appears
    // already laid out instead of animating in from a stale rect.
    for win in state.mapped_toplevels.iter_mut() {
        if win.workspace == n {
            win.render_rect = RectF::default();
            win.velocity = RectF::default();
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
    // Prefer the per-workspace remembered focus; fall back to the
    // topmost tile on the workspace if it's gone (closed since the
    // last time we were here, or never set on a fresh workspace).
    let saved = state
        .last_focus_per_workspace
        .get(&n)
        .and_then(|w| w.upgrade().ok())
        .filter(|s| {
            state.mapped_toplevels.iter().any(|w| {
                w.workspace == n && w.surface.upgrade().ok().as_ref() == Some(s)
            })
        });
    let new_focus = saved.or_else(|| {
        state
            .mapped_toplevels
            .iter()
            .rev()
            .filter(|w| w.workspace == n)
            .find_map(|w| w.surface.upgrade().ok())
    });
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
            // treatment when the user switches to `n` (snap-on-zero
            // in tick_animations).
            w.render_rect = RectF::default();
            w.velocity = RectF::default();
        })
        .is_some();
    if moved {
        // Mark the moved window as the remembered focus on its new
        // home, so a later switch to `n` lands on it instead of
        // whatever happened to be remembered there before.
        state.last_focus_per_workspace.insert(n, focus.downgrade());
    }
    if !moved {
        // Focus is on a layer surface or popup, not a tiled
        // toplevel — nothing to move.
        return;
    }
    // If the moved window was zoomed/fullscreened, the mode
    // semantically goes with it but is cleared on the next switch
    // anyway, so just drop it now.
    let was_zoomed = state
        .zoomed
        .as_ref()
        .and_then(|w| w.upgrade().ok())
        .as_ref()
        == Some(&focus);
    if was_zoomed {
        state.zoomed = None;
    }
    let was_fullscreen = state
        .fullscreened
        .as_ref()
        .and_then(|w| w.upgrade().ok())
        .as_ref()
        == Some(&focus);
    if was_fullscreen {
        state.fullscreened = None;
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

/// Emit one rounded-ring border per visible tile, modulated by
/// each window's `border_alpha` spring. Focus changes show up as
/// a cross-fade between the old and the new ring instead of an
/// instant swap; tiles whose alpha has decayed below an invisible
/// threshold are skipped so the renderer doesn't spend cycles on
/// fully-transparent quads.
///
/// Empty vec if `border_width = 0` (config-disabled) or no tile on
/// the active workspace currently has a non-zero border alpha
/// (initial state, or steady-state with focus on a layer surface
/// / popup / fullscreened tile). Fullscreen is enforced by the
/// per-tile `border_alpha` target in `tick_animations`, not here:
/// the spring drives the fullscreened tile's alpha toward 0 so the
/// transition into and out of fullscreen reads as a fade.
fn focus_border(state: &State) -> Vec<sol_core::SceneBorder> {
    let bw = state.config.border_width;
    if bw <= 0 {
        return Vec::new();
    }
    const ALPHA_EPS: f32 = 0.005;
    let active_ws = state.active_ws;
    let mut out = Vec::new();
    for win in state.mapped_toplevels.iter() {
        if win.workspace != active_ws {
            continue;
        }
        if win.border_alpha <= ALPHA_EPS {
            continue;
        }
        // Same scale-about-center the buffer pass uses, so the
        // ring tracks the tile's open / fullscreen-toggle scale
        // animation instead of floating where the tile used to be.
        let r = scale_about_center(win.render_rect, win.render_scale);
        let mut rgba = state.config.border_color;
        rgba[3] *= win.border_alpha;
        out.push(sol_core::SceneBorder {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
            rgba,
            corner_radius: state.config.corner_radius as f32,
            border_width: bw as f32,
        });
    }
    out
}

fn render_tick(comp: &mut Compositor) -> Result<()> {
    // If a page flip is already in flight, skip this tick and re-arm
    // `needs_render` so the next loop iteration retries once the flip
    // lands. Pre-idle-skip the page-flip handler set `needs_render`
    // unconditionally on every flip, which masked the loss; with the
    // idle optimisation in place it no longer does (`has_active_animation`
    // gates it), so the request would otherwise be dropped on the floor.
    // That dropped request is what produces the ~1s stall on `Ctrl+D`
    // close: the disconnect dispatch sets `needs_render`, the loop
    // callback clears it on the way into `render_tick`, we bail here on
    // the still-pending blink/cursor flip, and nothing re-arms us until
    // an unrelated event (next blink, input, etc.) ticks the loop.
    //
    // Counted as `ticks_skipped` so the histogram below only reflects
    // real renders. Mixing the two diluted the `<8ms` bucket with
    // sub-microsecond no-ops and made the tail unreadable.
    if let BackendState::Drm(presenter) = &comp.backend {
        if presenter.is_pending_flip() {
            comp.state.metrics.ticks_skipped += 1;
            // Split by reason so we can tell whether the GPU
            // (`pending_render` set, sync-FD not yet signalled) or the
            // kernel scan-out (`pending_flip` set, page-flip-complete
            // not yet fired) is the bottleneck. `is_pending_render`
            // would be a cleaner API; for now infer from the flag pair
            // exposed today.
            if presenter.is_render_in_flight() {
                comp.state.metrics.ticks_skipped_pending_render += 1;
            } else {
                comp.state.metrics.ticks_skipped_pending_flip += 1;
            }
            comp.state.needs_render = true;
            return Ok(());
        }
    }
    let tick_started = Instant::now();
    let result = render_tick_inner(comp);
    let elapsed_ns = tick_started.elapsed().as_nanos() as u64;
    comp.state.metrics.record_frame(elapsed_ns);
    maybe_emit_perf_log(&mut comp.state);
    result
}

/// One-line periodic perf summary, env-var-gated via
/// `SOL_PERF_LOG_INTERVAL_MS`. Computes deltas against the previous
/// snapshot so the numbers describe the most recent window, not
/// process-lifetime cumulative totals.
fn maybe_emit_perf_log(state: &mut State) {
    let Some(interval_ms) = state.perf_log_interval_ms else {
        return;
    };
    let now = Instant::now();
    let due = match state.last_perf_log_at {
        Some(prev) => now.duration_since(prev).as_millis() as u64 >= interval_ms,
        None => true,
    };
    if !due {
        return;
    }
    let cur = state.metrics.clone();
    let elapsed_ms = match state.last_perf_log_at {
        Some(prev) => now.duration_since(prev).as_millis() as u64,
        None => interval_ms.max(1),
    };
    let prev = state
        .last_perf_log_metrics
        .as_ref()
        .map(|b| (**b).clone())
        .unwrap_or_default();
    state.last_perf_log_at = Some(now);
    state.last_perf_log_metrics = Some(Box::new(cur.clone()));

    let frames = cur.frames_rendered.saturating_sub(prev.frames_rendered);
    let flips = cur.page_flips.saturating_sub(prev.page_flips);
    let skipped = cur.ticks_skipped.saturating_sub(prev.ticks_skipped);
    let skipped_render = cur
        .ticks_skipped_pending_render
        .saturating_sub(prev.ticks_skipped_pending_render);
    let skipped_flip = cur
        .ticks_skipped_pending_flip
        .saturating_sub(prev.ticks_skipped_pending_flip);
    if frames == 0 && flips == 0 {
        // Idle window — nothing to say. Don't spam the log.
        return;
    }
    let fps = (frames as f64 * 1000.0 / elapsed_ms as f64) as u64;

    // Per-frame averages. Avoid div-by-zero when `frames == 0` by
    // dividing by `frames.max(1)`; that bucket goes to "0" naturally
    // when nothing rendered.
    let f = frames.max(1);
    let avg_us = |cur: u64, prev: u64| (cur.saturating_sub(prev)) / f / 1_000;
    let avg_n = |cur: u64, prev: u64| (cur.saturating_sub(prev)) / f;

    let tick_us = avg_us(cur.render_tick_total_ns, prev.render_tick_total_ns);
    let render_us = avg_us(cur.phase_render_ns, prev.phase_render_ns);
    let tex_us = avg_us(cur.phase_render_textures_ns, prev.phase_render_textures_ns);
    let blur_us = avg_us(cur.phase_render_blur_ns, prev.phase_render_blur_ns);
    let draw_us = avg_us(cur.phase_render_draw_ns, prev.phase_render_draw_ns);
    let pres_us = avg_us(cur.phase_render_present_ns, prev.phase_render_present_ns);
    let wait_us = avg_us(
        cur.phase_render_cpu_wait_fence_ns,
        prev.phase_render_cpu_wait_fence_ns,
    );
    let submit_us = avg_us(
        cur.phase_render_cpu_queue_submit_ns,
        prev.phase_render_cpu_queue_submit_ns,
    );
    let export_us = avg_us(
        cur.phase_render_cpu_export_sync_fd_ns,
        prev.phase_render_cpu_export_sync_fd_ns,
    );

    let gpu_total_us = avg_us(cur.gpu_total_ns, prev.gpu_total_ns);
    let gpu_up_us = avg_us(cur.gpu_uploads_ns, prev.gpu_uploads_ns);
    let gpu_blur_us = avg_us(cur.gpu_blur_ns, prev.gpu_blur_ns);
    let gpu_draw_us = avg_us(cur.gpu_draw_ns, prev.gpu_draw_ns);

    let n_textured = avg_n(cur.n_textured_draws, prev.n_textured_draws);
    let n_solid = avg_n(cur.n_solid_draws, prev.n_solid_draws);
    let n_backdrop = avg_n(cur.n_backdrop_draws, prev.n_backdrop_draws);
    let n_blur_passes = avg_n(cur.n_blur_passes, prev.n_blur_passes);
    let n_pipe = avg_n(cur.n_pipeline_binds, prev.n_pipeline_binds);
    let n_desc = avg_n(cur.n_descriptor_binds, prev.n_descriptor_binds);
    // Uploads are bursty (zero most frames, several MB on others), so
    // per-frame integer averages truncate to 0 and lose the signal.
    // Show interval totals + the largest single upload seen, which is
    // what actually distinguishes "static wallpaper happy path" from
    // "animated wallpaper recommit" or "browser dropped to SHM mode".
    let up_total = cur.n_shm_uploads.saturating_sub(prev.n_shm_uploads);
    let up_kb = cur
        .n_shm_upload_bytes
        .saturating_sub(prev.n_shm_upload_bytes)
        / 1024;
    let up_max_kb = cur.n_shm_upload_max_bytes_running / 1024;
    // Reset the running max so the next interval starts fresh.
    state.metrics.n_shm_upload_max_bytes_running = 0;
    let dmabuf_new = cur
        .n_dmabuf_imports_new
        .saturating_sub(prev.n_dmabuf_imports_new);

    // Frame-interval histogram delta — show only the buckets that
    // changed in the window so the line stays compact.
    let mut iv_delta = [0u64; 8];
    for i in 0..8 {
        iv_delta[i] = cur.frame_interval_ms_buckets[i]
            .saturating_sub(prev.frame_interval_ms_buckets[i]);
    }

    tracing::info!(
        target: "sol_perf",
        fps,
        flips,
        skipped,
        skipped_r = skipped_render,
        skipped_f = skipped_flip,
        cpu_tick_us = tick_us,
        cpu_render_us = render_us,
        cpu_tex_us = tex_us,
        cpu_blur_us = blur_us,
        cpu_draw_us = draw_us,
        cpu_pres_us = pres_us,
        cpu_wait_us = wait_us,
        cpu_submit_us = submit_us,
        cpu_export_us = export_us,
        gpu_total_us,
        gpu_up_us,
        gpu_blur_us,
        gpu_draw_us,
        n_textured,
        n_solid,
        n_backdrop,
        n_blur_passes,
        n_pipe,
        n_desc,
        up_total,
        up_kb,
        up_max_kb,
        dmabuf_new,
        cached_total = cur.last_textures_cached_total,
        cached_dmabuf = cur.last_textures_cached_dmabuf,
        slot_s = cur.last_slot_scanned,
        slot_p = cur.last_slot_pending,
        slot_f = cur.last_slot_free,
        ?iv_delta,
        "perf"
    );
}

fn render_tick_inner(comp: &mut Compositor) -> Result<()> {
    // Phase 1 — prune. Free GPU resources for buffers clients
    // destroyed since the last tick (dmabuf-backed VkImages don't
    // get freed via Drop), drop dead weak refs from the mapped
    // collections, and rebalance focus if it pointed at a now-dead
    // surface.
    let phase_prune_t = Instant::now();
    if let BackendState::Drm(presenter) = &mut comp.backend {
        for key in std::mem::take(&mut comp.state.pending_texture_evictions) {
            presenter.evict_texture(key);
        }
    } else {
        comp.state.pending_texture_evictions.clear();
    }
    comp.state
        .mapped_toplevels
        .retain(|w| w.surface.upgrade().is_ok());
    comp.state
        .mapped_popups
        .retain(|w| w.upgrade().is_ok());
    // Prune dead-surface dialogs. Parented dialogs whose parent
    // died also drop — orphan dialogs have no sensible position
    // since we anchor them to the parent's render_rect. Parentless
    // floats (splash screens, About boxes) have no parent to
    // outlive, so the parent check is skipped for them.
    comp.state.mapped_dialogs.retain(|d| {
        if d.surface.upgrade().is_err() {
            return false;
        }
        match &d.parent {
            Some(weak) => weak.upgrade().is_ok(),
            None => true,
        }
    });
    rebalance_keyboard_focus(&mut comp.state);
    comp.state.metrics.phase_prune_ns += phase_prune_t.elapsed().as_nanos() as u64;

    // Phase 2 — layout. Recompute every tile's target rect from
    // master-stack (or zoom / fullscreen overrides), then push
    // configures to any client whose tile size has changed.
    let phase_layout_t = Instant::now();
    let now = Instant::now();
    apply_layout(&mut comp.state, now);
    send_pending_configures(&mut comp.state);
    comp.state.metrics.phase_layout_ns += phase_layout_t.elapsed().as_nanos() as u64;

    // Phase 3 — animations. Step springs (rect, alpha, scale,
    // border, swap, fade), advance the workspace crossfade. If
    // anything is still moving keep the loop firing — DRM page-flip
    // does this normally, this guard covers headless and ticks
    // where no other event is pending.
    let phase_anim_t = Instant::now();
    let animating = tick_animations(&mut comp.state, now);
    let ws_animating = tick_workspace_animation(&mut comp.state, now);
    if animating || ws_animating {
        comp.state.needs_render = true;
    }
    comp.state.metrics.phase_animations_ns += phase_anim_t.elapsed().as_nanos() as u64;

    // Phase 4 — scene collection. Walk State and produce the flat
    // Scene the backend will draw — subsurface trees, popups,
    // dialogs, layers, cursor. The borrow-the-buffers step inside
    // `scene_from_buffers` is part of this phase.
    let phase_scene_t = Instant::now();
    let (placed, background_count, border_anchor) = collect_scene(&comp.state, now);
    let mut scene = scene_from_buffers(
        &comp.state,
        &placed,
        background_count,
        border_anchor,
        &comp.state.cursor,
    );
    scene.borders.extend(focus_border(&comp.state));
    let drawn = scene.elements.len();
    // Capture the elapsed time now but defer the metric write until
    // the scene's borrow on `comp.state` is released (after the
    // render arm below). Folding into `Metrics::fold_render_timing`
    // keeps a similar pattern; this one is a single assignment
    // rather than a struct-fold, so we keep it inline.
    let phase_collect_scene_ns = phase_scene_t.elapsed().as_nanos() as u64;

    // Phase 5 — render. Hand the Scene to the backend (PNG dump for
    // headless; DRM presenter does texture upload, blur, draw, page
    // flip schedule for real hardware). Capture the wall-clock time
    // even on error so a failing frame still attributes correctly.
    let phase_render_t = Instant::now();
    let render_result: Result<()> = match &mut comp.backend {
        BackendState::Headless { canvas, png_path } => {
            canvas.clear();
            for e in &scene.elements {
                match &e.content {
                    SceneContent::Shm {
                        pixels,
                        stride,
                        format,
                        upload_seq: _,
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
                        // Headless backend has no GPU import path, can't
                        // sample the dmabuf. Skip — dmabuf clients are a
                        // DRM-backend feature.
                    }
                    SceneContent::BlurredBackdrop { .. } => {
                        // Headless has no GPU FBO and no blur path.
                        // Inactive-window frosted glass is a DRM-backend
                        // feature; the headless PNG dump just shows
                        // un-blurred wallpaper behind the inactive
                        // window's transparency.
                    }
                }
            }
            for b in &scene.borders {
                // Headless backend has no shader; it draws a plain
                // rectangular fill regardless of corner_radius /
                // border_width. The PNG dump is for debug, not
                // shipping pretty pixels — close enough.
                canvas.fill_rect(
                    b.x.round() as i32,
                    b.y.round() as i32,
                    b.w.round() as i32,
                    b.h.round() as i32,
                    b.rgba,
                );
            }
            let r = canvas
                .write_png(png_path)
                .with_context(|| format!("write {}", png_path.display()));
            if r.is_ok() {
                tracing::info!(drawn, path = %png_path.display(), "headless frame");
            }
            r
        }
        BackendState::Drm(presenter) => {
            let mut t = RenderTiming::default();
            let r = presenter
                .render_scene(&scene, &mut t)
                .context("drm render_scene");
            // Fold per-frame numbers (CPU + GPU phase totals + draw/upload
            // counts + slot occupancy) into the cumulative metrics. One
            // call instead of a wall of `+=` so adding a new field
            // doesn't mean chasing every accumulation site.
            comp.state.metrics.fold_render_timing(&t);
            if r.is_ok() {
                tracing::debug!(drawn, "drm frame");
            }
            // Strip the Option<OwnedFd> off the Result here so the
            // surrounding Result<()> shape is preserved; the FD is
            // routed back to the caller via a side channel because
            // we can't register a calloop source while still holding
            // `&mut comp.backend`.
            r.map(|fd| {
                if let Some(fd) = fd {
                    comp.state.pending_fence_fd = Some(fd);
                }
            })
        }
    };
    comp.state.metrics.phase_collect_scene_ns += phase_collect_scene_ns;
    comp.state.metrics.phase_render_ns += phase_render_t.elapsed().as_nanos() as u64;
    render_result?;

    // Hand off any GPU-completion fence FD produced by the DRM render
    // to a calloop source. The source fires once when the FD becomes
    // readable (GPU finished the queued work), at which point the
    // deferred `lock_front_buffer + page_flip` runs.
    if let Some(fd) = comp.state.pending_fence_fd.take() {
        if let Some(handle) = comp.loop_handle.clone() {
            // OneShot semantics: we want the source to fire exactly
            // once and then take itself out of the loop. Level mode
            // matches what other one-shot fd sources in the codebase
            // use; since we PostAction::Remove on first fire it
            // doesn't loop.
            let res = handle.insert_source(
                Generic::new(fd, Interest::READ, Mode::Level),
                |_ready, _fd, comp| {
                    if let BackendState::Drm(p) = &mut comp.backend {
                        match p.submit_flip_after_fence() {
                            Ok(t) => {
                                // Fold the deferred-phase numbers
                                // through the same path the synchronous
                                // submit uses so the bench sees a
                                // consistent attribution.
                                comp.state.metrics.fold_render_timing(&t);
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "submit_flip_after_fence");
                            }
                        }
                    }
                    Ok(PostAction::Remove)
                },
            );
            if let Err(e) = res {
                tracing::warn!(error = %e, "insert fence source");
                // Couldn't register the source — fall back to the
                // synchronous path inline so the frame doesn't hang
                // forever. submit_flip_after_fence works fine even
                // when called before the GPU is done; it'll just
                // block inside `lock_front_buffer` like the old code.
                if let BackendState::Drm(p) = &mut comp.backend {
                    if let Err(e) = p.submit_flip_after_fence() {
                        tracing::warn!(error = %e, "fallback submit_flip");
                    }
                }
            }
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
    session_notifier: Option<session::LibSeatSessionNotifier>,
    cfg: config::Config,
) -> Result<()> {
    let mut event_loop: EventLoop<'static, Compositor> =
        EventLoop::try_new().context("create calloop event loop")?;
    let mut display: Display<State> = Display::new().context("create wayland display")?;
    let dh = display.handle();

    // Smithay owns the wl_shm global. We construct ShmState first so
    // its `global()` ID can drop into the rest of `Globals`. ARGB8888
    // and XRGB8888 are advertised automatically by ShmState::new — no
    // need to pass them in the formats list.
    let shm_state = ShmState::new::<State>(&dh, []);
    let shm_global = shm_state.global();

    // Smithay owns the zwp_idle_inhibit_manager_v1 global. Same
    // construct-before-Globals shape so we can hand the GlobalId in.
    let idle_inhibit_manager_state = IdleInhibitManagerState::new::<State>(&dh);
    let idle_inhibit_global = idle_inhibit_manager_state.global();

    // Smithay owns the wl_compositor (v6) + wl_subcompositor globals.
    // The constructor creates both; the IDs drop into Globals below.
    let compositor_state = CompositorState::new_v6::<State>(&dh);
    let compositor_global = compositor_state.compositor_global();
    let subcompositor_global = compositor_state.subcompositor_global();

    // Smithay owns the wl_seat global. We add keyboard/pointer
    // capabilities below once we know whether libinput is wired up
    // (DRM mode). Headless leaves the seat with empty caps — same
    // behaviour as the pre-PR-5 hand-rolled bind.
    let mut seat_state: SeatState<State> = SeatState::new();
    let mut seat = seat_state.new_wl_seat(&dh, "seat0");
    let seat_global = seat.global().expect("new_wl_seat global");
    let mut keyboard_handle: Option<KeyboardHandle<State>> = None;
    let mut pointer_handle: Option<PointerHandle<State>> = None;
    if input.is_some() {
        match seat.add_keyboard(
            XkbConfig {
                layout: "us",
                ..XkbConfig::default()
            },
            cfg.keyboard_repeat_delay,
            cfg.keyboard_repeat_rate,
        ) {
            Ok(kb) => keyboard_handle = Some(kb),
            Err(e) => tracing::warn!(error = ?e, "seat.add_keyboard failed"),
        }
        pointer_handle = Some(seat.add_pointer());
    }

    let globals = Globals {
        compositor: compositor_global,
        shm: shm_global,
        output: dh.create_global::<State, WlOutput, ()>(OUTPUT_VERSION, ()),
        seat: seat_global,
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
        subcompositor: subcompositor_global,
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
        idle_inhibit: idle_inhibit_global,
        primary_selection: dh
            .create_global::<State, ZwpPrimarySelectionDeviceManagerV1, ()>(
                primary_selection::PRIMARY_SELECTION_VERSION,
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
                        .insert_client(stream, Arc::new(ClientState::default()))
                    {
                        Ok(_) => {
                            comp.state.clients_seen += 1;
                            comp.state.metrics.clients_connected += 1;
                        }
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
                        comp.state.metrics.page_flips += 1;
                        // Real refresh cadence: the gap between the
                        // *previous* flip-complete event and this one
                        // is what the monitor actually displayed at.
                        // Bucketed at 0.5ms granularity around the
                        // 240Hz / 144Hz / 60Hz budgets.
                        let now = Instant::now();
                        if let Some(prev) = comp.state.last_flip_at {
                            let dt = now.duration_since(prev).as_nanos() as u64;
                            comp.state.metrics.record_frame_interval(dt);
                        }
                        comp.state.last_flip_at = Some(now);
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
                        // Used to set needs_render unconditionally
                        // here, which kept the compositor at the full
                        // display refresh rate even with no scene
                        // changes — at 4K@240 with the visual effects
                        // pipeline (transparency + per-frame blur,
                        // even cached) that's a sustained ~70% GPU
                        // load just sitting idle. Now we only
                        // continue auto-rendering when there's
                        // actually motion to advance: any toplevel
                        // tween in flight, or a workspace crossfade.
                        // Static scenes coast on event-driven
                        // renders (client commits, input, focus
                        // changes) instead of pumping at vblank.
                        if has_active_animation(&comp.state) {
                            comp.state.needs_render = true;
                        }

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

    // libseat session: dispatch ActivateSession/PauseSession events,
    // handle DRM master transitions. Without this source, VT switches
    // would leave us in a half-disabled state and libseat would block.
    // Smithay's `LibSeatSessionNotifier` is itself a calloop
    // EventSource — no Generic wrapper needed; it dispatches the seat
    // and emits `SessionEvent::ActivateSession` / `PauseSession` to
    // the closure below.
    if let Some(notifier) = session_notifier {
        event_loop
            .handle()
            .insert_source(notifier, move |ev, _, comp| {
                handle_session_event(comp, ev);
            })
            .map_err(|e| anyhow::anyhow!("insert session source: {e}"))?;
    }

    let cursor = Cursor::new(screen_width as f64 / 2.0, screen_height as f64 / 2.0);
    let state = State {
        display_handle: dh,
        globals,
        clients_seen: 0,
        mapped_toplevels: Vec::new(),
        active_ws: 1,
        workspace_anim: None,
        last_anim_tick: None,
        closing_windows: Vec::new(),
        outputs: Vec::new(),
        ext_workspace_managers: Vec::new(),
        idle_inhibitors: Vec::new(),
        last_input_at: Instant::now(),
        last_flip_at: None,
        last_perf_log_at: None,
        last_perf_log_metrics: None,
        perf_log_interval_ms: std::env::var("SOL_PERF_LOG_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&v| v > 0),
        idle: false,
        pending_wake: false,
        screen_width,
        screen_height,
        screen_refresh_mhz,
        needs_render: false,
        pending_fence_fd: None,
        started: Instant::now(),
        next_serial: 0,
        cursor,
        input: input_state,
        seat_state,
        seat,
        keyboard: keyboard_handle,
        pointer: pointer_handle,
        pointer_focus: None,
        keyboard_focus: None,
        pending_frame_callbacks: Vec::new(),
        drm_device_path,
        socket_name: socket_name.clone(),
        left_alt_down: false,
        right_alt_down: false,
        pressed_keys: std::collections::HashSet::new(),
        left_ctrl_down: false,
        right_ctrl_down: false,
        left_shift_down: false,
        right_shift_down: false,
        left_super_down: false,
        right_super_down: false,
        config: cfg,
        zoomed: None,
        fullscreened: None,
        master_ratio: 0.5,
        resize_mode: false,
        flip_counter: 0,
        flip_counter_reset: None,
        pending_presentation: presentation_time::empty(),
        presentation_seq: 0,
        exec_once_pgids: Vec::new(),
        pending_texture_evictions: Vec::new(),
        shm_state,
        compositor_state,
        shm_cache: std::collections::HashMap::new(),
        idle_inhibit_manager_state,
        pending_layer_surfaces: Vec::new(),
        session: session.clone(),
        data_devices: Vec::new(),
        selection_source: None,
        primary_devices: Vec::new(),
        primary_selection_source: None,
        mapped_popups: Vec::new(),
        mapped_dialogs: Vec::new(),
        dragging: None,
        popup_grab: None,
        last_focus_per_workspace: std::collections::HashMap::new(),
        last_stack_focus_per_workspace: std::collections::HashMap::new(),
        last_master_swap_partner_per_workspace: std::collections::HashMap::new(),
        zoomed_per_workspace: std::collections::HashMap::new(),
        fullscreened_per_workspace: std::collections::HashMap::new(),
        metrics: Metrics::default(),
        #[cfg(feature = "debug-ctl")]
        debug_ctl: None,
    };
    let mut compositor = Compositor {
        state,
        display,
        backend,
        loop_handle: Some(event_loop.handle()),
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
    // Debug control socket — only present when sol was built with
    // `--features debug-ctl`. Bench harness reads the path off
    // $XDG_RUNTIME_DIR/sol-ctl-<wayland-display>.sock.
    #[cfg(feature = "debug-ctl")]
    {
        let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let ctl_path = runtime_dir.join(format!("sol-ctl-{}.sock", socket_name));
        match debug_ctl::install(&event_loop.handle(), loop_signal.clone(), ctl_path) {
            Ok(ctl) => compositor.state.debug_ctl = Some(ctl),
            Err(e) => tracing::warn!(error = %e, "debug-ctl install failed"),
        }
    }
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
/// state AND libinput's device fds in sync with our session ownership.
///
/// libinput suspend/resume is mandatory here: logind revokes our
/// /dev/input/event* fds on PauseSession, so libinput is left with
/// stale fds that stop producing events even after we come back.
/// Calling `suspend` before we yield closes them via our
/// close_restricted callback, and `resume` on ActivateSession re-opens
/// fresh ones via open_restricted (which routes through libseat).
///
/// Smithay's notifier internally calls `seat.disable()` on
/// PauseSession before emitting the event, so the daemon's ack is
/// already done by the time we run — we just need to clean up our
/// own subsystems.
fn handle_session_event(comp: &mut Compositor, ev: session::SessionEvent) {
    match ev {
        session::SessionEvent::PauseSession => {
            tracing::info!("libseat: pause — suspending input + releasing DRM master");
            if let Some(input) = comp.state.input.as_mut() {
                input.li.suspend();
            }
            if let BackendState::Drm(presenter) = &mut comp.backend {
                presenter.drop_master();
            }
        }
        session::SessionEvent::ActivateSession => {
            tracing::info!("libseat: activate — reacquiring DRM master + resuming input");
            if let BackendState::Drm(presenter) = &mut comp.backend {
                if let Err(e) = presenter.reacquire_master() {
                    tracing::warn!(error = %e, "DRM reacquire on activate");
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
    state.metrics.input_events += 1;
    // Smithay's KeyboardHandle / PointerHandle track their bound
    // wl_keyboard / wl_pointer resources internally and prune dead
    // ones automatically — no per-tick `retain` needed.

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
            // Track pressed keys so wl_keyboard.enter can hand the new
            // client the actually-held keycodes (per spec). Updated for
            // every key (modifier or not) so the set is exhaustive.
            // Bind-eaten keys are still tracked here — the bind handler
            // returns before send_keyboard_key, but the user still
            // physically pressed the key, and the *next* focused client
            // shouldn't be told otherwise.
            if pressed {
                state.pressed_keys.insert(keycode);
            } else {
                state.pressed_keys.remove(&keycode);
            }
            let alt = state.left_alt_down || state.right_alt_down;
            let ctrl = state.left_ctrl_down || state.right_ctrl_down;
            let shift = state.left_shift_down || state.right_shift_down;
            let sup = state.left_super_down || state.right_super_down;

            // Modal resize loop. While active, H / L bump the
            // master/stack split ratio by ±5% per press, Escape
            // exits, every other key (including modifiers) is
            // swallowed instead of being forwarded to the focused
            // client — the user pressed `resize_mode` to *adjust*,
            // not to type. Modifier *tracking* above still runs so
            // the next bind after exit sees correct state.
            //
            // We still feed xkb every key here so its modifier
            // state matches physical reality on exit — without
            // this, a release we swallow leaves xkb thinking the
            // mod is held, and the modifier event we re-send to
            // the client on resume would be wrong.
            if state.resize_mode {
                if let Some(kbd) = state.keyboard.clone() {
                    let xkb_code =
                        smithay::input::keyboard::Keycode::new(keycode + 8);
                    let kstate = if pressed {
                        smithay::backend::input::KeyState::Pressed
                    } else {
                        smithay::backend::input::KeyState::Released
                    };
                    // input_intercept updates smithay's xkb without
                    // forwarding to any client.
                    let _: ((), bool) =
                        kbd.input_intercept(state, xkb_code, kstate, |_, _, _| ());
                }
                if pressed {
                    match keycode {
                        KEY_ESC => {
                            state.resize_mode = false;
                            // Re-attach the keyboard to the focused
                            // client and re-broadcast modifier state
                            // from xkb. Without this the client's
                            // view of held keys is stale (we sent it
                            // a leave on entry) and typing reads as
                            // Alt+letter / Ctrl+letter for any modifier
                            // that was held when the user entered the
                            // mode.
                            state.keyboard_resume_focused();
                            tracing::debug!("resize_mode: exit");
                        }
                        KEY_L => {
                            state.master_ratio =
                                (state.master_ratio + 0.05).clamp(0.1, 0.9);
                            state.needs_render = true;
                            tracing::debug!(
                                ratio = state.master_ratio,
                                "resize_mode: L"
                            );
                        }
                        KEY_H => {
                            state.master_ratio =
                                (state.master_ratio - 0.05).clamp(0.1, 0.9);
                            state.needs_render = true;
                            tracing::debug!(
                                ratio = state.master_ratio,
                                "resize_mode: H"
                            );
                        }
                        _ => {}
                    }
                }
                return;
            }

            // Ctrl+Alt+F1..F12 → libseat VT switch. Intentionally not
            // overridable from the config: a misconfigured .conf must
            // never be able to trap the user on the compositor.
            if pressed && ctrl && alt {
                if let Some(vt) = vt_number_for_f_key(keycode) {
                    if let Some(s) = state.session.as_mut() {
                        use session::Session as _;
                        if let Err(e) = s.change_vt(vt) {
                            tracing::warn!(?e, vt, "libseat change_vt");
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
                        config::Action::ToggleFullscreen => {
                            toggle_fullscreen(state);
                        }
                        config::Action::ToggleZoom => {
                            toggle_zoom(state);
                        }
                        config::Action::ResizeMode => {
                            state.resize_mode = true;
                            // Suspend key delivery to the focused
                            // client so the H / L / Esc presses we
                            // intercept here don't reach it as
                            // unmatched events. wl_keyboard.leave
                            // releases all keys for that surface, so
                            // any modifier the user was holding at
                            // entry won't get stuck "down" on the
                            // client side.
                            state.keyboard_suspend_focused();
                            tracing::debug!("resize_mode: enter");
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
/// Used by the modal resize loop. Match the values in
/// `config::key_from_name`.
const KEY_ESC: u32 = 1;
const KEY_H: u32 = 35;
/// Linux evdev BTN_LEFT — the primary mouse button. Used by the
/// compositor-driven move gesture (Super + left click on a
/// floating window).
const BTN_LEFT: u32 = 0x110;
const KEY_L: u32 = 38;

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
    if let Some(home) = std::env::var_os("HOME") {
        cmd.current_dir(home);
    }
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
    // Children inherit our cwd by default, which is wherever the
    // user launched sol from — not what a freshly-spawned terminal
    // or app should land in. Chdir to $HOME so Alt+Enter alacritty
    // opens at ~ regardless of how sol was started.
    let mut cmd = std::process::Command::new(program);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    if let Some(home) = std::env::var_os("HOME") {
        cmd.current_dir(home);
    }
    match cmd.spawn()
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

    // Pass 0: xdg_popups, topmost first. Popups draw on top of
    // toplevels and over Top/Overlay layers, so click resolution
    // must check them before anything else — otherwise right-click
    // on an open context menu lands on the toplevel underneath.
    //
    // Hit-test against the LOGICAL popup rect (positioner-sized,
    // matches what the user perceives as the menu — clicks on the
    // shadow padding outside fall through). The surface-local
    // coords we hand to `resolve_hit`, however, are buffer-space:
    // we add the window_geometry offset back in, so a hover
    // halfway down the visible menu lands on the menu item Chrome
    // drew at that surface position rather than one shifted by
    // the shadow_top of however much the client padded.
    for popup in state.mapped_popups.iter().rev() {
        let Ok(surface) = popup.upgrade() else { continue };
        let Some((offset, size, geom)) = compositor::with_sol_data(&surface, |sd| {
            match sd.role.clone() {
                SurfaceRole::XdgPopup { mapped: true, offset, size } => {
                    Some((offset, size, sd.xdg_window_geometry))
                }
                _ => None,
            }
        })
        .flatten() else { continue };
        let Some((ox, oy)) = popup_screen_origin(state, &surface) else { continue };
        let popup_rect = Rect {
            x: (ox + offset.0 as f32).round() as i32,
            y: (oy + offset.1 as f32).round() as i32,
            w: size.0,
            h: size.1,
        };
        if let Some((lx, ly)) = hit_rect(popup_rect) {
            // Translate from "logical popup" coords into
            // surface-buffer coords. resolve_hit walks subsurfaces
            // off this point, so they land at the right offsets too.
            let (gx, gy) = match geom {
                Some((gx, gy, _, _)) => (gx as f64, gy as f64),
                None => (0.0, 0.0),
            };
            return Some(resolve_hit(surface, lx + gx, ly + gy));
        }
    }

    // Pass 1a: Overlay layers. Always topmost (lockscreen / OSD must
    // catch clicks even with a fullscreened tile underneath).
    for ml in layers.iter().filter(|m| matches!(m.layer, Layer::Overlay)) {
        if let Some((lx, ly)) = hit_rect(ml.rect) {
            return Some(resolve_hit(ml.surface.clone(), lx, ly));
        }
    }

    // Pass 1aa: floating dialogs. Iterate newest-first so a dialog
    // opened on top of another dialog catches its own clicks.
    for (idx, dlg) in state.mapped_dialogs.iter().enumerate().rev() {
        if dlg.workspace != state.active_ws {
            continue;
        }
        let Ok(surface) = dlg.surface.upgrade() else { continue };
        let Some((dx, dy)) = dialog_render_origin(state, idx) else { continue };
        let Some((dw, dh)) = compositor::with_sol_data(&surface, surface_logical_size)
            .flatten()
        else {
            continue;
        };
        let dlg_rect = Rect {
            x: dx.round() as i32,
            y: dy.round() as i32,
            w: dw,
            h: dh,
        };
        if let Some((lx, ly)) = hit_rect(dlg_rect) {
            return Some(resolve_hit(surface, lx, ly));
        }
    }

    // Pass 1b: a fullscreened tile, if any. Drawn above Top layers
    // in the render pass, so the click model has to match — clicking
    // the area where waybar would normally sit must reach the
    // fullscreen content.
    let fullscreened = state.fullscreened.as_ref().and_then(|w| w.upgrade().ok());
    if let Some(fs) = &fullscreened {
        if let Some(win) = state.mapped_toplevels.iter().find(|w| {
            w.surface.upgrade().ok().as_ref() == Some(fs)
                && w.workspace == state.active_ws
        }) {
            if let Some((lx, ly)) = hit_rect(win.render_rect.round()) {
                return Some(resolve_hit(fs.clone(), lx, ly));
            }
        }
    }

    // Pass 1c: Top layers. Skipped while a fullscreen tile is active —
    // they're visually covered, so clicks on the area where waybar
    // sits should fall through to the fullscreen tile (handled
    // above), not the now-invisible bar.
    if fullscreened.is_none() {
        for ml in layers.iter().filter(|m| matches!(m.layer, Layer::Top)) {
            if let Some((lx, ly)) = hit_rect(ml.rect) {
                return Some(resolve_hit(ml.surface.clone(), lx, ly));
            }
        }
    }

    // Pass 2: tiled toplevels (top of stack first). Honor zoom and
    // fullscreen: while either is active, only that one tile can
    // catch clicks; the others aren't rendered.
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
        if fullscreened.is_some() {
            // Fullscreen was already hit-tested above; skip the
            // generic tile pass entirely so non-fullscreen tiles
            // don't catch clicks they can't visually receive.
            continue;
        }
        let mapped = compositor::with_sol_data(&surface, |sd| {
            matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true })
        })
        .unwrap_or(false);
        if !mapped {
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
    // (child surface, parent-relative offset, buffer dims).
    type Child = (WlSurface, (i32, i32), (i32, i32));

    // Snapshot children with their offsets + buffer dims. Each
    // `with_sol_data` call locks the per-surface mutex briefly; we
    // never nest a child lookup inside the parent's, so smithay's
    // tree mutex never sees re-entry.
    let child_weaks: Vec<Weak<WlSurface>> = compositor::with_sol_data(parent, |sd| {
        sd.subsurface_children.clone()
    })?;
    let children: Vec<Child> = child_weaks
        .into_iter()
        .filter_map(|w| {
            let s = w.upgrade().ok()?;
            let (offset, dims) = compositor::with_sol_data(&s, |csd| {
                let buf = csd.current_buffer.as_ref()?;
                let dims = surface_buffer_dims(buf)?;
                Some((csd.subsurface_offset, dims))
            })
            .flatten()?;
            Some((s, offset, dims))
        })
        .collect();
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
    if let Some(dims) = shm::shm_dims(buf) {
        return Some(dims);
    }
    if let Some(db) = buf.data::<linux_dmabuf::DmabufBuffer>() {
        return Some((db.width, db.height));
    }
    None
}

fn update_pointer_focus_and_motion(state: &mut State) {
    // Compositor-driven dialog drag captures the pointer until the
    // button release ends it. While set, motion updates the dialog's
    // `position` (top-left = cursor − captured offset) and we skip
    // the regular focus + motion dispatch so the dragged client
    // doesn't see motion events from outside its own surface.
    if let Some(drag) = state.dragging.clone() {
        let cursor_x = state.cursor.pos_x as f32;
        let cursor_y = state.cursor.pos_y as f32;
        if let Ok(surface) = drag.surface.upgrade() {
            for d in state.mapped_dialogs.iter_mut() {
                if d.surface.upgrade().ok().as_ref() == Some(&surface) {
                    d.position =
                        Some((cursor_x - drag.offset.0, cursor_y - drag.offset.1));
                    break;
                }
            }
        } else {
            // Dragged surface died mid-drag — just drop the capture.
            state.dragging = None;
        }
        state.needs_render = true;
        return;
    }

    let hit = surface_under_cursor(state);
    let new_focus = hit.as_ref().map(|(s, _, _)| s.clone());
    let focus_changed = !surface_eq(state.pointer_focus.as_ref(), new_focus.as_ref());
    if focus_changed {
        // Per `wl_pointer.set_cursor` spec, the client's cursor
        // choice is bound to the surface that received the most
        // recent `enter` event — the moment the pointer leaves, we
        // revert to the default sprite. (Smithay's `cursor_image`
        // callback re-asserts the override the next time the new
        // client calls set_cursor.)
        state.cursor.client_override_active = false;
        state.cursor.client_surface = None;
    }
    state.pointer_focus = new_focus.clone();

    // Hand off to smithay's PointerHandle: it tracks per-client
    // wl_pointer resources and emits enter/leave/motion + frame
    // grouping based on the focus/location pair we hand it. None
    // pointer (headless) skips event delivery; we still maintain
    // `state.pointer_focus` for the rest of the codebase.
    let Some(pointer) = state.pointer.clone() else { return };
    let now_ms = state.elapsed_ms();
    let cursor = (state.cursor.pos_x, state.cursor.pos_y).into();
    let focus = hit.as_ref().map(|(s, lx, ly)| (s.clone(), (*lx, *ly).into()));
    pointer.motion(
        state,
        focus,
        &MotionEvent {
            location: cursor,
            serial: SERIAL_COUNTER.next_serial(),
            time: now_ms,
        },
    );
    pointer.frame(state);
}

/// Forward a libinput scroll event to the focused client as a
/// wl_pointer axis frame. Scroll always targets the surface currently
/// under the cursor — `PointerHandle` knows that surface internally
/// from the most recent `motion`, so we don't pass it. Smithay's
/// `axis(AxisFrame)` issues `axis_source` / `axis` / `axis_stop` /
/// `axis_value120` / `axis_discrete` based on what's set in the
/// frame and the bound wl_pointer's version.
fn send_pointer_axis(
    state: &mut State,
    source: AxisSource,
    v_value: Option<f64>,
    h_value: Option<f64>,
    v120_v: Option<f64>,
    v120_h: Option<f64>,
) {
    let Some(pointer) = state.pointer.clone() else { return };
    let time = state.elapsed_ms();
    let smithay_source = match source {
        AxisSource::Wheel => smithay::backend::input::AxisSource::Wheel,
        AxisSource::Finger => smithay::backend::input::AxisSource::Finger,
        AxisSource::Continuous => smithay::backend::input::AxisSource::Continuous,
    };
    let kinetic = matches!(source, AxisSource::Finger | AxisSource::Continuous);
    let mut frame = AxisFrame::new(time).source(smithay_source);
    if let Some(val) = v_value {
        if val == 0.0 && kinetic {
            frame = frame.stop(smithay::backend::input::Axis::Vertical);
        } else {
            frame = frame.value(smithay::backend::input::Axis::Vertical, val);
        }
    }
    if let Some(val) = h_value {
        if val == 0.0 && kinetic {
            frame = frame.stop(smithay::backend::input::Axis::Horizontal);
        } else {
            frame = frame.value(smithay::backend::input::Axis::Horizontal, val);
        }
    }
    if matches!(source, AxisSource::Wheel) {
        if let Some(step) = v120_v {
            frame = frame.v120(smithay::backend::input::Axis::Vertical, step as i32);
        }
        if let Some(step) = v120_h {
            frame = frame.v120(smithay::backend::input::Axis::Horizontal, step as i32);
        }
    }
    pointer.axis(state, frame);
    pointer.frame(state);
}

/// True if the cursor's current focus surface is one of our mapped
/// popups or a subsurface descendant of one. Used to decide whether
/// a button press dismisses an active popup grab.
fn pointer_focus_in_popup_chain(state: &State) -> bool {
    let Some(focus) = state.pointer_focus.as_ref() else { return false };
    for popup in &state.mapped_popups {
        let Ok(p) = popup.upgrade() else { continue };
        if &p == focus || surface_in_subtree(&p, focus) {
            return true;
        }
    }
    false
}

fn surface_in_subtree(root: &WlSurface, target: &WlSurface) -> bool {
    let Some(children) = compositor::with_sol_data(root, |sd| {
        sd.subsurface_children
            .iter()
            .filter_map(|w| w.upgrade().ok())
            .collect::<Vec<WlSurface>>()
    }) else {
        return false;
    };
    for child in children {
        if &child == target {
            return true;
        }
        if surface_in_subtree(&child, target) {
            return true;
        }
    }
    false
}

/// Send `popup_done` to every currently-mapped popup so the client
/// tears the chain down. Safer than only dismissing the explicit
/// grabbing popup: if a client somehow grabbed nested popups
/// non-monotonically, this still leaves no orphans on screen. The
/// xdg_popup destroy dispatch prunes `mapped_popups` and the grab
/// slot — we don't need to mutate them here.
fn dismiss_popup_grab(state: &mut State) {
    let popups: Vec<WlSurface> = state
        .mapped_popups
        .iter()
        .filter_map(|w| w.upgrade().ok())
        .collect();
    for surface in popups {
        let popup = compositor::with_sol_data(&surface, |sd| sd.xdg_popup.clone()).flatten();
        if let Some(weak) = popup {
            if let Ok(p) = weak.upgrade() {
                p.popup_done();
            }
        }
    }
    state.popup_grab = None;
}

fn send_pointer_button(state: &mut State, button: u32, pressed: bool) {
    // While an interactive dialog drag is in progress, the next
    // button release ends it; intervening presses (multi-button
    // mash) and the release itself are swallowed instead of being
    // forwarded to the dragged client, so it doesn't see a release
    // it didn't get a press for. After clearing the drag we return
    // — the click was the user "letting go of the window", not a
    // click on whatever's under the cursor.
    if state.dragging.is_some() {
        if !pressed {
            state.dragging = None;
            state.needs_render = true;
        }
        return;
    }

    // Compositor-driven floating-window move: Super + left click
    // on a floating window starts a drag without the client
    // having to call `xdg_toplevel.move`. Catches the
    // no-titlebar / no-CSD case (game windows, lots of native
    // Rust GUI apps) where the protocol's move request never
    // fires. The press is swallowed so the click doesn't also
    // reach the client.
    let super_held = state.left_super_down || state.right_super_down;
    if pressed && button == BTN_LEFT && super_held {
        if let Some(focus) = state.pointer_focus.clone() {
            let dialog_surface = state.mapped_dialogs.iter().find_map(|d| {
                let s = d.surface.upgrade().ok()?;
                if s == focus || surface_in_subtree(&s, &focus) {
                    Some(s)
                } else {
                    None
                }
            });
            if let Some(target) = dialog_surface {
                start_dialog_drag(state, &target);
                return;
            }
        }
    }

    // If a popup is currently grabbing and the press landed outside
    // its surface tree, dismiss it before forwarding the click. The
    // client tears the popup chain down in response to `popup_done`,
    // and the click still propagates to whatever surface was hit so
    // the user doesn't lose a "click somewhere else" gesture.
    if pressed && state.popup_grab.is_some() && !pointer_focus_in_popup_chain(state) {
        dismiss_popup_grab(state);
    }
    let Some(focus) = state.pointer_focus.clone() else { return };
    if !focus.is_alive() {
        return;
    }
    // Click-to-focus: on press, move keyboard focus to the window
    // under the cursor. No-op if that window already has focus
    // (set_keyboard_focus compares first).
    //
    // BUT skip the focus change when the click is inside a popup
    // chain. A popup is a transient surface — it'll be torn down by
    // the click that activates a menu item — and moving keyboard
    // focus to it produces wl_keyboard.leave + wl_keyboard.enter
    // events on the parent client mid-click, then another pair when
    // the popup destroys, which is exactly the focus-dance pattern
    // Chrome interprets as "the user clicked elsewhere, abandon the
    // menu action". Symptom: Save Image As doesn't open its file
    // picker. Keep keyboard focus on whatever already had it (the
    // main app window) so the click is a clean pointer-only event.
    let is_popup_click = state.mapped_popups.iter().any(|p| {
        let Ok(s) = p.upgrade() else { return false };
        s == focus || surface_in_subtree(&s, &focus)
    });
    if pressed && !is_popup_click {
        state.set_keyboard_focus(Some(focus.clone()));
    }
    let Some(pointer) = state.pointer.clone() else { return };
    let time = state.elapsed_ms();
    let serial = SERIAL_COUNTER.next_serial();
    let button_state = if pressed {
        smithay::backend::input::ButtonState::Pressed
    } else {
        smithay::backend::input::ButtonState::Released
    };
    pointer.button(
        state,
        &ButtonEvent {
            button,
            state: button_state,
            serial,
            time,
        },
    );
    pointer.frame(state);
}

/// Feed a libinput key event into smithay's keyboard. `intercept`
/// always runs to keep smithay's xkb / pressed-keys state in sync
/// with physical reality (so a modifier release while focus is dead
/// doesn't leave xkb thinking it's still held); `forward` runs only
/// when we want the event delivered to the focused client. Sol's
/// keybind dispatch in `apply_input` early-returns before calling
/// this when a binding eats the keypress, so binding handlers
/// don't accidentally double-dispatch.
fn send_keyboard_key(state: &mut State, keycode: u32, pressed: bool) {
    let Some(kbd) = state.keyboard.clone() else { return };
    let time = state.elapsed_ms();
    let key_state = if pressed {
        smithay::backend::input::KeyState::Pressed
    } else {
        smithay::backend::input::KeyState::Released
    };
    // libinput emits evdev keycodes (0-based); xkbcommon adds 8 for
    // the X11 wire convention. Smithay's keyboard handle takes the
    // xkb keycode; on send to wl_keyboard.key it subtracts 8 again.
    let xkb_code = smithay::input::keyboard::Keycode::new(keycode + 8);
    let serial = SERIAL_COUNTER.next_serial();
    let _: Option<()> = kbd.input(
        state,
        xkb_code,
        key_state,
        serial,
        time,
        |_, _, _| FilterResult::Forward,
    );
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
    setup_event_loop(backend, width, height, 60_000, None, None, None, None, cfg)
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
    // Smithay's `LibSeatSession::new` does an initial seat dispatch
    // and sets `is_active()` if the daemon already had an Enable
    // queued. On systemd hosts launched from an active TTY this is
    // the common case. `session::open` bails with a helpful error if
    // it isn't, rather than racing the rest of startup against a
    // still-disabled seat (opening `/dev/dri` while disabled would
    // fail in a less-readable way).
    let (mut session, notifier) =
        session::open().context("libseat: open session")?;

    // Load config first so the user's `mode = WxH@Hz` (if any) drives
    // initial mode selection. The DRM backend's pick_output uses this
    // as a fallback when SOL_MODE isn't set.
    let cfg = config::load();
    let mode_pref = cfg.mode.map(|m| sol_backend_drm::ModePreference {
        width: m.width,
        height: m.height,
        refresh_hz: m.refresh_hz,
    });

    // Open the DRM device through smithay's session, wrap into a
    // Card, build the presenter off it. Smithay's `LibSeatSession`
    // tracks the libseat::Device internally keyed by raw fd, so the
    // OwnedFd we get back is dup-managed and stays alive for the
    // session's lifetime — no `open_device_keep_fd` hand-rolled
    // dup-and-store needed.
    use session::Session as _;
    let drm_fd = session
        .open(device, OFlags::RDWR | OFlags::CLOEXEC)
        .map_err(|e| anyhow::anyhow!("libseat: open DRM device: {e:?}"))?;
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
        BackendState::Drm(Box::new(presenter)),
        w,
        h,
        refresh_mhz,
        input,
        Some(device.to_path_buf()),
        Some(session),
        Some(notifier),
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

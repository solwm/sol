//! Wayland server for voidptr.
//!
//! Handles protocol traffic in all backends; rendering is delegated to a
//! `BackendState` value (software canvas -> PNG for headless, or a
//! `voidptr_backend_drm::DrmPresenter` for real hardware).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic};
use voidptr_backend_drm::DrmPresenter;
use std::os::fd::AsRawFd;
use voidptr_core::{Scene, SceneContent, SceneElement};
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
mod cursor;
mod data_device;
mod input;
mod layer_shell;
mod linux_dmabuf;
mod output;
mod render;
mod seat;
mod session;
mod shm;
mod subcompositor;
mod xdg_decoration;
mod xdg_output;
mod xdg_shell;
mod xkb;

use compositor::{SurfaceData, SurfaceRole};
use input::{InputEvent, InputState};
use render::Canvas;
use wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::xdg::decoration::zv1::server::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_protocols::xdg::xdg_output::zv1::server::zxdg_output_manager_v1::ZxdgOutputManagerV1;
use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::ZwlrLayerShellV1;
use wayland_server::protocol::{
    wl_data_device_manager::WlDataDeviceManager,
    wl_keyboard::{self, WlKeyboard},
    wl_pointer::{self, WlPointer},
    wl_subcompositor::WlSubcompositor,
};
use xkb::KeymapState;

const COMPOSITOR_VERSION: u32 = 6;
const SHM_VERSION: u32 = 1;
const OUTPUT_VERSION: u32 = 4;
const SEAT_VERSION: u32 = 7;
const XDG_WM_BASE_VERSION: u32 = 5;

/// Screen-space rectangle assigned to a mapped window by the layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// A mapped xdg_toplevel together with the screen-space rect the layout
/// assigned to it. `pending_size` is the (w, h) most recently sent to the
/// client via xdg_toplevel.configure; layout sends a fresh configure only
/// when the target rect differs, so we don't spam configures every frame.
pub struct Window {
    pub surface: Weak<WlSurface>,
    pub rect: Rect,
    pub pending_size: Option<(i32, i32)>,
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
    /// Logical screen size. Drives toplevel placement and (eventually) the
    /// wl_output mode advertised to clients.
    pub screen_width: u32,
    pub screen_height: u32,
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
    /// Modifier-key tracking for built-in keybinds. libinput hands us raw
    /// evdev keycodes; Alt is scancode 56 (left) or 100 (right), Ctrl
    /// is 29 (left) / 97 (right).
    pub left_alt_down: bool,
    pub right_alt_down: bool,
    pub left_ctrl_down: bool,
    pub right_ctrl_down: bool,
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
        Self {
            pos_x: centre_x,
            pos_y: centre_y,
            visible: true,
            pixels: cursor::pixels(),
            width: cursor::CURSOR_W,
            height: cursor::CURSOR_H,
            hot_x: cursor::CURSOR_HOT_X,
            hot_y: cursor::CURSOR_HOT_Y,
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
}

impl State {
    pub fn elapsed_ms(&self) -> u32 {
        self.started.elapsed().as_millis() as u32
    }
    pub fn next_serial(&mut self) -> u32 {
        self.next_serial = self.next_serial.wrapping_add(1);
        self.next_serial
    }

    /// Give this surface keyboard focus if no other surface currently has it.
    /// Used when a toplevel first maps.
    pub fn on_toplevel_mapped(&mut self, surface: &WlSurface) {
        if self.keyboard_focus.is_none() {
            self.set_keyboard_focus(Some(surface.clone()));
        }
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
    let next = state
        .mapped_toplevels
        .iter()
        .rev()
        .find_map(|w| w.surface.upgrade().ok());
    state.set_keyboard_focus(next);
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
fn apply_layout(state: &mut State) {
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };
    let layers = layer_shell::mapped_layers(state, screen);
    let usable = layer_shell::usable_area(&layers, screen);
    let rects = master_stack_layout(state.mapped_toplevels.len(), usable);
    for (win, rect) in state.mapped_toplevels.iter_mut().zip(rects.into_iter()) {
        win.rect = rect;
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
fn collect_scene(
    state: &State,
) -> Vec<(wayland_server::protocol::wl_buffer::WlBuffer, Rect)> {
    let mut out = Vec::new();
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };

    use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_shell_v1::Layer;
    let layers = layer_shell::mapped_layers(state, screen);

    // 1. Background + Bottom layer surfaces.
    for ml in &layers {
        if matches!(ml.layer, Layer::Background | Layer::Bottom) {
            out.push((ml.buffer.clone(), ml.rect));
            emit_subsurface_tree(&mut out, &ml.surface, ml.rect.x, ml.rect.y);
        }
    }

    // 2. Tiled xdg_toplevels.
    for win in state.mapped_toplevels.iter() {
        let Ok(surface) = win.surface.upgrade() else { continue };
        let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
            continue;
        };
        let buf_opt = {
            let sd = sd_arc.lock().unwrap();
            if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
                continue;
            }
            sd.current.buffer.as_ref().and_then(|w| w.upgrade().ok())
        };
        if let Some(buf) = buf_opt {
            out.push((buf, win.rect));
        }
        emit_subsurface_tree(&mut out, &surface, win.rect.x, win.rect.y);
    }

    // 3. Top + Overlay layer surfaces.
    for ml in &layers {
        if matches!(ml.layer, Layer::Top | Layer::Overlay) {
            out.push((ml.buffer.clone(), ml.rect));
            emit_subsurface_tree(&mut out, &ml.surface, ml.rect.x, ml.rect.y);
        }
    }

    out
}

/// Walk a surface's subsurface_children recursively, pushing each mapped
/// child's current buffer at `parent_origin + child.subsurface_offset`.
/// Subsurfaces without a buffer attached yet are skipped but their own
/// children still descended into — the buffer may land a frame later.
/// Stacking follows registration order in `subsurface_children`;
/// `place_above` / `place_below` aren't implemented.
fn emit_subsurface_tree(
    out: &mut Vec<(wayland_server::protocol::wl_buffer::WlBuffer, Rect)>,
    parent: &WlSurface,
    parent_x: i32,
    parent_y: i32,
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
        let (buf_opt, child_x, child_y) = {
            let sd = child_sd_arc.lock().unwrap();
            let (ox, oy) = sd.subsurface_offset;
            let buf = sd.current.buffer.as_ref().and_then(|w| w.upgrade().ok());
            (buf, parent_x + ox, parent_y + oy)
        };
        if let Some(buf) = buf_opt {
            // rect.w/h aren't used when rendering — scene_from_buffers
            // pulls the intrinsic size from the buffer's own metadata
            // (SHM width/height, dmabuf width/height). Only x/y matters.
            out.push((
                buf,
                Rect {
                    x: child_x,
                    y: child_y,
                    w: 0,
                    h: 0,
                },
            ));
        }
        emit_subsurface_tree(out, &child, child_x, child_y);
    }
}

fn scene_from_buffers<'a>(
    placed: &'a [(wayland_server::protocol::wl_buffer::WlBuffer, Rect)],
    cursor: &'a Cursor,
) -> Scene<'a> {
    let mut scene = Scene::new();
    for (buf, rect) in placed {
        if let Some(bd) = buf.data::<shm::BufferData>() {
            let Some(bytes) = bd.bytes() else { continue };
            let Some(format) = bd.pixel_format() else { continue };
            let key = (bd as *const shm::BufferData) as usize as u64;
            scene.elements.push(SceneElement {
                buffer_key: key,
                width: bd.width,
                height: bd.height,
                x: rect.x,
                y: rect.y,
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
            scene.elements.push(SceneElement {
                buffer_key: key,
                width: db.width,
                height: db.height,
                x: rect.x,
                y: rect.y,
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
    // Cursor last so it always draws on top.
    if cursor.visible {
        scene.elements.push(SceneElement {
            buffer_key: CURSOR_SCENE_KEY,
            width: cursor.width,
            height: cursor.height,
            x: cursor.pos_x as i32 - cursor.hot_x,
            y: cursor.pos_y as i32 - cursor.hot_y,
            content: SceneContent::Shm {
                pixels: &cursor.pixels,
                stride: cursor.width * 4,
                format: voidptr_core::PixelFormat::Argb8888,
            },
        });
    }
    scene
}

/// Stable key the DRM renderer uses to avoid re-uploading the cursor
/// sprite every frame. Anything outside the range of real BufferData
/// pointers works; pick something unlikely to collide.
const CURSOR_SCENE_KEY: u64 = 0xC0FFEE_C0FFEE;

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
    apply_layout(&mut comp.state);
    send_pending_configures(&mut comp.state);

    let placed = collect_scene(&comp.state);
    let scene = scene_from_buffers(&placed, &comp.state.cursor);
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
                            e.x, e.y, pixels, e.width, e.height, *stride, *format,
                        );
                    }
                    SceneContent::Dmabuf { .. } => {
                        // Headless backend has no EGL context, can't sample
                        // the dmabuf. Skip — dmabuf clients are a DRM-backend
                        // feature.
                    }
                }
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
    input: Option<(InputState, std::os::fd::OwnedFd)>,
    drm_device_path: Option<PathBuf>,
    session: Option<session::SharedSession>,
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
                        // Another commit may have piled up while we
                        // were waiting; trigger a fresh render so the
                        // pipeline stays full.
                        comp.state.needs_render = true;
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
        screen_width,
        screen_height,
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

    tracing::info!(
        socket = %socket_name,
        "ready: export WAYLAND_DISPLAY={} and run a client",
        socket_name
    );

    // External stop signals (SIGTERM, SIGHUP — the ones `killall voidptr`
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
        tracing::warn!(error = %e, "ctrlc handler install failed; `killall voidptr` won't cleanly restore the TTY");
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
        InputEvent::Key { keycode, pressed } => {
            // Track modifiers before anything else so keybinds see
            // accurate state. Released modifiers stop participating.
            match keycode {
                KEY_LEFTALT => state.left_alt_down = pressed,
                KEY_RIGHTALT => state.right_alt_down = pressed,
                KEY_LEFTCTRL => state.left_ctrl_down = pressed,
                KEY_RIGHTCTRL => state.right_ctrl_down = pressed,
                _ => {}
            }
            let alt = state.left_alt_down || state.right_alt_down;
            let ctrl = state.left_ctrl_down || state.right_ctrl_down;

            // Ctrl+Alt+F1..F12 → libseat VT switch. Intercept FIRST so
            // clients don't see the Fn press. Matches sway/Hyprland.
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

            // Built-in keybinds:
            //   Alt+Enter → alacritty (terminal)
            //   Alt+D     → rofi (application launcher, layer-shell client)
            if pressed && alt && !ctrl {
                match keycode {
                    KEY_ENTER => {
                        spawn_client(state, "alacritty", &[], "Alt+Enter");
                        return;
                    }
                    KEY_D => {
                        spawn_client(
                            state,
                            "rofi",
                            &["-show", "drun"],
                            "Alt+D",
                        );
                        return;
                    }
                    _ => {}
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

/// evdev scancodes, matching what libinput hands us via `key.key()`.
const KEY_ENTER: u32 = 28;
const KEY_D: u32 = 32;
const KEY_LEFTALT: u32 = 56;
const KEY_RIGHTALT: u32 = 100;
const KEY_LEFTCTRL: u32 = 29;
const KEY_RIGHTCTRL: u32 = 97;

/// Fork/exec a Wayland client connected to our own socket. Env is
/// inherited wholesale from voidptr's process, which was normalised at
/// startup (XDG_SESSION_TYPE=wayland, XDG_CURRENT_DESKTOP=voidptr,
/// DISPLAY/XAUTHORITY unset, WAYLAND_DISPLAY set). Child handle is
/// intentionally dropped — no reap, no wait; kernel cleans up on
/// voidptr exit.
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

    let layers = layer_shell::mapped_layers(state, screen);

    // Pass 1: Overlay > Top. Iterate overlay-first (descending priority).
    for ml in layers
        .iter()
        .filter(|m| matches!(m.layer, Layer::Overlay))
        .chain(layers.iter().filter(|m| matches!(m.layer, Layer::Top)))
    {
        if let Some((lx, ly)) = hit_rect(ml.rect) {
            return Some((ml.surface.clone(), lx, ly));
        }
    }

    // Pass 2: tiled toplevels (top of stack first).
    for win in state.mapped_toplevels.iter().rev() {
        let Ok(surface) = win.surface.upgrade() else { continue };
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
        if let Some((lx, ly)) = hit_rect(win.rect) {
            return Some((surface, lx, ly));
        }
    }

    // Pass 3: Bottom > Background.
    for ml in layers
        .iter()
        .filter(|m| matches!(m.layer, Layer::Bottom))
        .chain(layers.iter().filter(|m| matches!(m.layer, Layer::Background)))
    {
        if let Some((lx, ly)) = hit_rect(ml.rect) {
            return Some((ml.surface.clone(), lx, ly));
        }
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
    setup_event_loop(backend, width, height, None, None, None)
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
                     handed us the seat. Make sure voidptr was launched from \
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

    // Open the DRM device through libseat, wrap into a Card, build the
    // presenter off it.
    let drm_fd = session
        .borrow_mut()
        .open_device_keep_fd(device)
        .context("libseat: open DRM device")?;
    let card = voidptr_backend_drm::Card::from_fd(drm_fd);
    let presenter =
        DrmPresenter::from_card(card).context("initialise DrmPresenter")?;
    let (w, h) = presenter.size();
    tracing::info!(width = w, height = h, "drm backend");

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
        input,
        Some(device.to_path_buf()),
        Some(session),
    )
}

/// Back-compat entry: default to headless with the old PNG path.
pub fn run() -> Result<()> {
    let png_path = std::env::var_os("VOIDPTR_PNG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/voidptr-headless.png"));
    run_headless(png_path, 1920, 1080)
}

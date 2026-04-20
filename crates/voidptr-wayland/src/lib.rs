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
mod input;
mod linux_dmabuf;
mod output;
mod render;
mod seat;
mod shm;
mod xdg_decoration;
mod xdg_shell;
mod xkb;

use compositor::{SurfaceData, SurfaceRole};
use input::{InputEvent, InputState};
use render::Canvas;
use wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::xdg::decoration::zv1::server::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_server::protocol::{
    wl_keyboard::{self, WlKeyboard},
    wl_pointer::{self, WlPointer},
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
fn rebalance_keyboard_focus(state: &mut State) {
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

/// Assign a screen-space rect to each mapped toplevel using master-stack.
fn apply_layout(state: &mut State) {
    let screen = Rect {
        x: 0,
        y: 0,
        w: state.screen_width as i32,
        h: state.screen_height as i32,
    };
    let rects = master_stack_layout(state.mapped_toplevels.len(), screen);
    for (win, rect) in state.mapped_toplevels.iter_mut().zip(rects.into_iter()) {
        win.rect = rect;
    }
}

/// For each window whose assigned rect differs from the size we most
/// recently told the client, send a fresh xdg_toplevel.configure +
/// xdg_surface.configure. Cached via `Window.pending_size` so steady-state
/// render ticks don't re-send configures.
fn send_pending_configures(state: &mut State) {
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
        tl.configure(w, h, Vec::new());
        xs.configure(serial);
        state.mapped_toplevels[i].pending_size = Some((w, h));
    }
}

/// Collects scene elements from mapped xdg_toplevels. Returns each toplevel's
/// backing `WlBuffer` paired with the `Rect` the layout assigned to it, so
/// callers keep the buffers alive for the duration of rendering.
fn collect_scene(
    state: &State,
) -> Vec<(wayland_server::protocol::wl_buffer::WlBuffer, Rect)> {
    let mut out = Vec::new();
    for win in state.mapped_toplevels.iter() {
        let Ok(surface) = win.surface.upgrade() else { continue };
        let Some(sd_arc) = surface.data::<Arc<Mutex<SurfaceData>>>() else {
            continue;
        };
        let sd = sd_arc.lock().unwrap();
        if !matches!(sd.role, SurfaceRole::XdgToplevel { mapped: true }) {
            continue;
        }
        let Some(buf_weak) = sd.current.buffer.as_ref() else {
            continue;
        };
        if let Ok(buf) = buf_weak.upgrade() {
            out.push((buf, win.rect));
        }
    }
    out
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

    // Scene is done; drop buffers to release them back to clients.
    for (buf, _) in placed {
        buf.release();
    }
    Ok(())
}

fn setup_event_loop(
    backend: BackendState,
    screen_width: u32,
    screen_height: u32,
    input: Option<(InputState, std::os::fd::OwnedFd)>,
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
    };
    tracing::info!(?globals, "advertised globals");

    let listener = wayland_server::ListeningSocket::bind_auto("wayland", 1..33)
        .context("bind wayland socket")?;
    let socket_name = listener
        .socket_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<anonymous>".into());
    tracing::info!(socket = %socket_name, "listening");

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

    // Wire SIGINT/SIGTERM to stop the calloop run cleanly. Stopping here
    // causes `run()` to return, which drops Compositor -> BackendState ->
    // DrmPresenter, whose Drop restores the saved CRTC so the TTY unblanks.
    // ctrlc::set_handler panics if called twice in the same process; that's
    // fine, voidptr only sets up one event loop per invocation.
    let loop_signal = event_loop.get_signal();
    if let Err(e) = ctrlc::set_handler(move || {
        tracing::info!("shutdown signal received; stopping event loop");
        loop_signal.stop();
    }) {
        tracing::warn!(error = %e, "ctrlc handler install failed; Ctrl+C may not cleanly restore the TTY");
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
            send_keyboard_key(state, keycode, pressed);
        }
    }
}

/// Hit-test the cursor against mapped toplevels (top of stack first) using
/// each window's layout-assigned `Rect`. Returns the surface plus cursor
/// position in surface-local coords.
fn surface_under_cursor(state: &State) -> Option<(WlSurface, f64, f64)> {
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
        let r = win.rect;
        let lx = state.cursor.pos_x - r.x as f64;
        let ly = state.cursor.pos_y - r.y as f64;
        if lx >= 0.0 && lx < r.w as f64 && ly >= 0.0 && ly < r.h as f64 {
            return Some((surface, lx, ly));
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
    let Some(focus) = state.keyboard_focus.clone() else { return };
    if !focus.is_alive() {
        return;
    }
    // Update modifier state via xkb and emit wl_keyboard.modifiers if it
    // changed (e.g. shift pressed/released).
    let mods = state
        .keymap
        .as_mut()
        .map(|km| km.feed_key(keycode, pressed));
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
    setup_event_loop(backend, width, height, None)
}

/// DRM backend: takes master on `device`, renders via GLES to the screen.
/// Must be called from a VT that doesn't already have a compositor.
pub fn run_drm(device: &Path) -> Result<()> {
    let presenter = DrmPresenter::new(device).context("initialise DrmPresenter")?;
    let (w, h) = presenter.size();
    tracing::info!(width = w, height = h, "drm backend");

    // libinput is DRM-only: a headless PNG dump has nowhere to point a cursor.
    let input = match InputState::init("seat0") {
        Ok(pair) => Some(pair),
        Err(e) => {
            tracing::warn!(error = %e, "libinput init failed; running without input");
            None
        }
    };
    setup_event_loop(BackendState::Drm(presenter), w, h, input)
}

/// Back-compat entry: default to headless with the old PNG path.
pub fn run() -> Result<()> {
    let png_path = std::env::var_os("VOIDPTR_PNG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/voidptr-headless.png"));
    run_headless(png_path, 1920, 1080)
}

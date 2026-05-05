//! Compositor-owned per-surface state.
//!
//! Smithay owns the wl_compositor / wl_subcompositor / wl_region /
//! wl_surface / wl_callback / wl_subsurface wire dispatch
//! (`delegate_compositor!` in lib.rs). Per-surface state that smithay
//! doesn't model — role tracking, viewport src/dst, popup positioner
//! result, dialog parent ref, the cached xdg/layer-shell handles —
//! lives in `SolSurfaceData`, attached to each surface via smithay's
//! `UserDataMap` slot at `CompositorHandler::new_surface` time.
//!
//! The wl_surface Commit body that used to live here has moved to
//! `CompositorHandler::commit` in lib.rs — it consumes smithay's
//! double-buffered `SurfaceAttributes` (buffer assignment, damage,
//! frame callbacks) and our `SolSurfaceData` (role + role state) to
//! drive the same map/unmap/configure pipeline as before.

use std::sync::Mutex;

use wayland_protocols::xdg::shell::server::{
    xdg_popup::XdgPopup, xdg_toplevel::XdgToplevel,
};
use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1;
use wayland_server::Weak;
use wayland_server::protocol::wl_surface::WlSurface;

/// Roles a surface can take on.
#[derive(Clone, Debug, Default)]
pub enum SurfaceRole {
    #[default]
    None,
    XdgToplevel {
        mapped: bool,
    },
    /// `zwlr_layer_surface_v1` role. Layer surfaces live outside the
    /// tile layout — they're anchored to output edges and stack in a
    /// separate z-order (background/bottom under toplevels, top/overlay
    /// over them).
    LayerSurface {
        mapped: bool,
        /// Set to true once we've sent the first `configure` event but
        /// haven't yet seen the corresponding `ack_configure` + buffered
        /// commit. The client is allowed to commit without a buffer in
        /// this state to request reconfiguration.
        initial_configure_sent: bool,
    },
    /// `wl_subsurface` role. Subsurfaces are children of another
    /// wl_surface, positioned at an offset, and drawn as part of the
    /// parent's tree (popups, tooltips, cursor images). The scene
    /// walker recurses through `subsurface_children` when rendering
    /// any mapped root (xdg_toplevel or layer_surface).
    Subsurface,
    /// `xdg_popup` role: a transient surface anchored to a parent
    /// (toplevel or another popup) used for context menus, dropdowns,
    /// tooltips. `offset` is the popup's top-left in the parent's
    /// surface-local coords (computed from xdg_positioner when the
    /// popup is created). `size` is what we configured the client to
    /// draw at.
    XdgPopup {
        mapped: bool,
        offset: (i32, i32),
        size: (i32, i32),
    },
}

/// Per-surface state we track that smithay's `SurfaceData` /
/// `SurfaceAttributes` don't already cover. Stored as a
/// `Mutex<SolSurfaceData>` in smithay's `data_map`, keyed by type.
/// Initialized in `CompositorHandler::new_surface`.
///
/// The currently-attached `WlBuffer` lives in
/// `current_buffer`. We mirror it from
/// `SurfaceAttributes::current().buffer` on each commit so the
/// renderer can read "is this surface mapped right now" without
/// cracking open smithay's cached_state every frame.
#[derive(Default)]
pub struct SolSurfaceData {
    pub role: SurfaceRole,
    /// Strong handle to the currently-attached `wl_buffer`, **not** a
    /// `Weak`. Holding a strong `WlBuffer` keeps the resource (and
    /// its user_data — `ShmBufferUserData` / `DmabufBuffer` — and
    /// the underlying SHM mapping or imported `VkImage`) alive even
    /// after the client destroys its end. Alacritty (and any GL
    /// client managing its own swapchain) destroys old buffers ahead
    /// of committing the new one across a resize; with `Weak<WlBuffer>`
    /// that race produced ~5 frames where `upgrade()` failed and we
    /// skipped drawing the surface entirely → wallpaper bleed-through.
    /// Keeping a strong ref means we still have a valid texture
    /// source until the next attach replaces it.
    pub current_buffer: Option<wayland_server::protocol::wl_buffer::WlBuffer>,
    /// Populated in `XdgShellHandler::new_toplevel` so the compositor
    /// can send directive configure events during layout without
    /// threading the XdgToplevel handle through every code path.
    pub xdg_toplevel: Option<Weak<XdgToplevel>>,
    /// Populated by zwlr_layer_shell_v1.get_layer_surface — same purpose
    /// as the xdg handles above, but for layer surfaces so apply_layout
    /// can configure the bar/launcher/wallpaper.
    pub zwlr_layer_surface: Option<Weak<ZwlrLayerSurfaceV1>>,
    /// Populated by xdg_surface.get_popup. Lets the compositor send
    /// `popup_done` to dismiss the popup (e.g. when the user clicks
    /// outside a grabbing popup).
    pub xdg_popup: Option<Weak<XdgPopup>>,
    /// For surfaces with role=XdgPopup: weak ref to the parent surface
    /// the popup is anchored to (a toplevel or another popup). Used to
    /// compute the popup's screen-space origin and walk up the chain
    /// when dismissing nested popup grabs.
    pub xdg_popup_parent: Option<Weak<WlSurface>>,
    /// Set by `xdg_toplevel.set_parent`. When `Some` at first map
    /// time, this toplevel is treated as a dialog/transient.
    pub xdg_toplevel_parent: Option<Weak<WlSurface>>,
    /// Most recent (width, height) seen via `xdg_toplevel.set_min_size`.
    pub xdg_min_size: (i32, i32),
    /// Most recent (width, height) seen via `xdg_toplevel.set_max_size`.
    pub xdg_max_size: (i32, i32),
    /// Most recent rect set via `xdg_surface.set_window_geometry`,
    /// in surface-local coords.
    pub xdg_window_geometry: Option<(i32, i32, i32, i32)>,
    /// For surfaces with role=Subsurface: weak ref to the parent the
    /// child hangs off of.
    pub subsurface_parent: Option<Weak<WlSurface>>,
    /// Subsurface position relative to its parent's origin, in
    /// surface-local coords. We mirror smithay's
    /// `SubsurfaceCachedState::location` here on every commit so
    /// subsurface readers don't have to crack open `with_states`.
    pub subsurface_offset: (i32, i32),
    /// Every wl_surface can have subsurface children attached via
    /// wl_subcompositor.get_subsurface. Iteration order is the order
    /// of registration.
    pub subsurface_children: Vec<Weak<WlSurface>>,
    /// Logical destination size declared via
    /// `wp_viewport.set_destination`, in surface-local pixels.
    pub viewport_dst: Option<(i32, i32)>,
    /// Source crop rect declared via `wp_viewport.set_source`, in
    /// buffer coordinates. Tuple layout is `(x, y, w, h)`.
    pub viewport_src: Option<(f64, f64, f64, f64)>,
}

/// Borrow `SolSurfaceData` for a surface, with an immutable closure.
/// Returns `None` if smithay hasn't yet populated the slot for this
/// surface (only happens during the gap between resource creation and
/// `CompositorHandler::new_surface`, so very narrow).
pub fn with_sol_data<R>(
    surface: &WlSurface,
    f: impl FnOnce(&SolSurfaceData) -> R,
) -> Option<R> {
    smithay::wayland::compositor::with_states(surface, |states| {
        states
            .data_map
            .get::<Mutex<SolSurfaceData>>()
            .map(|m| f(&m.lock().unwrap()))
    })
}

/// Mutating counterpart of [`with_sol_data`].
pub fn with_sol_data_mut<R>(
    surface: &WlSurface,
    f: impl FnOnce(&mut SolSurfaceData) -> R,
) -> Option<R> {
    smithay::wayland::compositor::with_states(surface, |states| {
        states
            .data_map
            .get::<Mutex<SolSurfaceData>>()
            .map(|m| f(&mut m.lock().unwrap()))
    })
}

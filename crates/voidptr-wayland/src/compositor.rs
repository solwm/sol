//! wl_compositor + wl_surface + wl_region + wl_callback.
//!
//! Surfaces hold pending and current double-buffered state. `commit` promotes
//! pending into current and, if the surface has an xdg_toplevel role with a
//! buffer, marks the scene dirty so the compositor re-renders.

use std::sync::{Arc, Mutex};

use wayland_protocols::xdg::shell::server::{
    xdg_surface::XdgSurface, xdg_toplevel::XdgToplevel,
};
use wayland_protocols_wlr::layer_shell::v1::server::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1;
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, Weak,
    protocol::{
        wl_buffer::WlBuffer,
        wl_callback::{self, WlCallback},
        wl_compositor::{self, WlCompositor},
        wl_region::{self, WlRegion},
        wl_surface::{self, WlSurface},
    },
};

use crate::State;

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
    /// walker recurses through SurfaceData.subsurface_children when
    /// rendering any mapped root (xdg_toplevel or layer_surface).
    Subsurface,
}

#[derive(Default)]
pub struct SurfaceState {
    pub buffer: Option<Weak<WlBuffer>>,
    /// Accumulated damage since last commit (surface-local coords).
    pub damage: Vec<(i32, i32, i32, i32)>,
}

#[derive(Default)]
pub struct SurfaceData {
    pub role: SurfaceRole,
    pub pending: SurfaceState,
    pub current: SurfaceState,
    /// True iff wl_surface.attach was called since the last commit.
    /// Without this flag, a commit-without-attach would overwrite
    /// current.buffer with an empty pending.buffer — unmapping the
    /// surface for one frame, then remapping it on the next
    /// attach+commit. Firefox's GDK commits subsurfaces this way for
    /// frame-callback delivery, which showed up as rapid blinking
    /// until we started honoring "no attach since last commit" as
    /// "leave current.buffer alone."
    pub pending_attach: bool,
    pub frame_callbacks: Vec<WlCallback>,
    /// Populated by xdg_surface.get_toplevel so the compositor can send
    /// directive configure events during layout without threading the
    /// XdgToplevel handle through every code path.
    pub xdg_toplevel: Option<Weak<XdgToplevel>>,
    pub xdg_surface: Option<Weak<XdgSurface>>,
    /// Populated by zwlr_layer_shell_v1.get_layer_surface — same purpose
    /// as the xdg handles above, but for layer surfaces so apply_layout
    /// can configure the bar/launcher/wallpaper.
    pub zwlr_layer_surface: Option<Weak<ZwlrLayerSurfaceV1>>,
    /// For surfaces with role=Subsurface: weak ref to the parent the
    /// child hangs off of. The scene walker follows this in reverse
    /// (parent → children) but carries the parent link so cleanup on
    /// destroy can remove this child from its parent's list.
    pub subsurface_parent: Option<Weak<WlSurface>>,
    /// Current offset of this subsurface from its parent's origin, in
    /// surface-local coords. Applied on commit; pending_subsurface_offset
    /// holds the value set via wl_subsurface.set_position between commits.
    pub subsurface_offset: (i32, i32),
    pub subsurface_pending_offset: Option<(i32, i32)>,
    /// Every wl_surface can have subsurface children attached via
    /// wl_subcompositor.get_subsurface. Iteration order is the order
    /// of registration; wl_subsurface.place_above/below isn't
    /// implemented yet so the stacking matches the client's
    /// attachment sequence.
    pub subsurface_children: Vec<Weak<WlSurface>>,
    /// Logical destination size declared via
    /// `wp_viewport.set_destination`, in surface-local pixels.
    /// **Not** an on-screen output rect — the compositor still owns
    /// that (tile rect / layer-shell anchor). Destination affects
    /// how the client labels its own logical size for input / damage
    /// math; we store it for protocol correctness but don't use it
    /// to drive output placement.
    pub viewport_dst: Option<(i32, i32)>,
    /// Source crop rect declared via `wp_viewport.set_source`, in
    /// buffer coordinates (fixed-point 24.8 on the wire; we store
    /// as f64). Tuple layout is `(x, y, w, h)`. When Some, the
    /// compositor samples this sub-rect of the buffer instead of
    /// the full texture; UVs are computed at render time as
    /// `source / buffer_size`. Used by video / shell integration
    /// clients that render into a larger buffer than they want
    /// displayed.
    pub viewport_src: Option<(f64, f64, f64, f64)>,
}

impl GlobalDispatch<WlCompositor, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlCompositor>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let c = init.init(resource, ());
        tracing::info!(id = ?c.id(), "bind wl_compositor");
    }
}

impl Dispatch<WlCompositor, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlCompositor,
        request: wl_compositor::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_compositor::Request::CreateSurface { id } => {
                let data = Arc::new(Mutex::new(SurfaceData::default()));
                let s = init.init(id, data);
                tracing::debug!(id = ?s.id(), "create_surface");
            }
            wl_compositor::Request::CreateRegion { id } => {
                let _ = init.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSurface, Arc<Mutex<SurfaceData>>> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        surface: &WlSurface,
        request: wl_surface::Request,
        data: &Arc<Mutex<SurfaceData>>,
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_surface::Request::Attach { buffer, x: _, y: _ } => {
                let mut sd = data.lock().unwrap();
                sd.pending.buffer = buffer.as_ref().map(|b| b.downgrade());
                sd.pending_attach = true;
            }
            wl_surface::Request::Damage {
                x,
                y,
                width,
                height,
            }
            | wl_surface::Request::DamageBuffer {
                x,
                y,
                width,
                height,
            } => {
                let mut sd = data.lock().unwrap();
                sd.pending.damage.push((x, y, width, height));
            }
            wl_surface::Request::Frame { callback } => {
                let cb = init.init(callback, ());
                data.lock().unwrap().frame_callbacks.push(cb);
            }
            wl_surface::Request::Commit => {
                let mut sd = data.lock().unwrap();
                // Promote pending -> current, but only touch the buffer
                // if the client actually called wl_surface.attach since
                // the last commit. A commit without attach is a no-op
                // on the content (used to deliver frame callbacks or
                // ack state changes); treating it as "buffer = None"
                // would unmap the surface for one frame, blinking it.
                // Hold onto the previous buffer so we can release it
                // only once it's been definitively replaced.
                let old_buffer = if sd.pending_attach {
                    let old = sd.current.buffer.take();
                    sd.current.buffer = sd.pending.buffer.take();
                    sd.pending_attach = false;
                    Some(old)
                } else {
                    // Still drain pending.buffer so a stale pending
                    // value doesn't leak into the next real attach.
                    sd.pending.buffer = None;
                    None
                };
                sd.current.damage = std::mem::take(&mut sd.pending.damage);

                // Subsurface position is double-buffered too: commits
                // promote the pending offset recorded by
                // wl_subsurface.set_position. We treat every commit as
                // desynced (see subcompositor.rs), so the offset goes
                // live on the child's own commit rather than waiting
                // for the parent to commit.
                if let Some(off) = sd.subsurface_pending_offset.take() {
                    sd.subsurface_offset = off;
                }

                // Release the replaced buffer so the client can reuse it,
                // per the wl_buffer.release "no longer used" contract.
                // Only when old != new (same-buffer re-attach keeps the
                // current reference live) and only after we've released
                // the SurfaceData lock so any handlers don't re-enter us.
                let new_buffer_id = sd
                    .current
                    .buffer
                    .as_ref()
                    .and_then(|w| w.upgrade().ok())
                    .map(|b| b.id());
                drop(sd);
                // old_buffer is Option<Option<Weak>>: outer None means
                // no attach happened (nothing to release); outer Some
                // with inner None means we promoted but had no prior
                // buffer; outer Some(Some) is the actual replaced
                // buffer.
                if let Some(Some(old_weak)) = old_buffer {
                    if let Ok(old_buf) = old_weak.upgrade() {
                        if Some(old_buf.id()) != new_buffer_id {
                            old_buf.release();
                        }
                    }
                }
                let sd = data.lock().unwrap();

                // If this surface has a layer-shell role, double-buffer
                // its per-role state (anchor, size, margin, etc.) the
                // same way we did with the buffer above.
                let is_layer_role =
                    matches!(sd.role, SurfaceRole::LayerSurface { .. });
                drop(sd);
                if is_layer_role {
                    crate::layer_shell::promote_layer_state(surface);
                }
                let mut sd = data.lock().unwrap();

                // Mark as mapped once we have a buffer + role.
                let has_buffer = sd.current.buffer.is_some();
                let mut just_mapped = false;
                let mut layer_needs_configure = false;
                match &mut sd.role {
                    SurfaceRole::XdgToplevel { mapped } => {
                        if has_buffer {
                            if !*mapped {
                                *mapped = true;
                                just_mapped = true;
                                tracing::info!(id = ?surface.id(), "toplevel mapped");
                                state.mapped_toplevels.push(crate::Window {
                                    surface: surface.downgrade(),
                                    rect: crate::Rect::default(),
                                    render_rect: crate::Rect::default(),
                                    pending_size: None,
                                });
                            }
                        } else {
                            *mapped = false;
                        }
                    }
                    SurfaceRole::LayerSurface {
                        mapped,
                        initial_configure_sent,
                    } => {
                        // Protocol flow:
                        // 1. Client creates layer_surface, sets state,
                        //    commits with NO buffer.
                        // 2. Server sends configure(serial, w, h).
                        // 3. Client acks + commits WITH a buffer → mapped.
                        if !*initial_configure_sent && !has_buffer {
                            layer_needs_configure = true;
                            *initial_configure_sent = true;
                        } else if has_buffer && !*mapped && *initial_configure_sent {
                            *mapped = true;
                            just_mapped = true;
                            tracing::info!(id = ?surface.id(), "layer surface mapped");
                        } else if !has_buffer && *mapped {
                            // Null buffer unmaps per spec.
                            *mapped = false;
                        }
                    }
                    SurfaceRole::None | SurfaceRole::Subsurface => {}
                }
                drop(sd);

                if layer_needs_configure {
                    crate::layer_shell::send_initial_configure(state, surface);
                }

                if just_mapped {
                    // Give this toplevel keyboard focus if no one else has
                    // it yet, so the user can type into it immediately.
                    state.on_toplevel_mapped(surface);
                }

                // Defer frame callbacks until the next successful render:
                // firing them here would let clients commit the next buffer
                // before we've presented the current one, so they'd just
                // over-render at full CPU speed. Delaying to post-present
                // paces them to our vblank cadence.
                let callbacks = std::mem::take(&mut data.lock().unwrap().frame_callbacks);
                state.pending_frame_callbacks.extend(callbacks);

                state.needs_render = true;
            }
            wl_surface::Request::Destroy => {}
            _ => {}
        }
    }
}

// wl_callback has no requests; the impl is required by the type system.
impl Dispatch<WlCallback, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlCallback,
        _request: wl_callback::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<WlRegion, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlRegion,
        _request: wl_region::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
    }
}

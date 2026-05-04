//! wl_compositor + wl_surface + wl_region + wl_callback.
//!
//! Surfaces hold pending and current double-buffered state. `commit` promotes
//! pending into current and, if the surface has an xdg_toplevel role with a
//! buffer, marks the scene dirty so the compositor re-renders.

use std::sync::{Arc, Mutex};

use wayland_protocols::xdg::shell::server::{
    xdg_popup::XdgPopup, xdg_surface::XdgSurface, xdg_toplevel::XdgToplevel,
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
    /// `xdg_popup` role: a transient surface anchored to a parent
    /// (toplevel or another popup) used for context menus, dropdowns,
    /// tooltips. `offset` is the popup's top-left in the parent's
    /// surface-local coords (computed from xdg_positioner when the
    /// popup is created). `size` is what we configured the client to
    /// draw at. Both stay constant across the popup's lifetime —
    /// reposition is handled by sending a fresh configure.
    XdgPopup {
        mapped: bool,
        offset: (i32, i32),
        size: (i32, i32),
    },
}

#[derive(Default)]
pub struct SurfaceState {
    /// Strong handle to the currently-attached wl_buffer, **not** a
    /// `Weak`. Holding a strong `WlBuffer` keeps the resource (and its
    /// user_data — BufferData / DmabufBuffer — and the underlying
    /// SHM mapping or imported `VkImage`) alive even after the client
    /// destroys its end. Alacritty (and any GL client managing its
    /// own swapchain) destroys old buffers ahead of committing the
    /// new one across a resize; with `Weak<WlBuffer>` that race
    /// produced ~5 frames where `upgrade()` failed and we skipped
    /// drawing the surface entirely → wallpaper bleed-through.
    /// Keeping a strong ref means we still have a valid texture
    /// source until the next attach replaces it.
    pub buffer: Option<WlBuffer>,
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
    /// Populated by xdg_surface.get_popup. Lets the compositor send
    /// `popup_done` to dismiss the popup (e.g. when the user clicks
    /// outside a grabbing popup) without threading the resource handle
    /// through every code path.
    pub xdg_popup: Option<Weak<XdgPopup>>,
    /// For surfaces with role=XdgPopup: weak ref to the parent surface
    /// the popup is anchored to (a toplevel or another popup). Used to
    /// compute the popup's screen-space origin (parent's render
    /// origin plus popup_offset) and to walk up the chain when
    /// dismissing nested popup grabs.
    pub xdg_popup_parent: Option<Weak<WlSurface>>,
    /// Set by `xdg_toplevel.set_parent`. When `Some` at first map
    /// time, this toplevel is treated as a dialog/transient: it
    /// stays out of the tile layout and floats centered over the
    /// parent's tile. Used by save/discard prompts, file pickers,
    /// preferences windows, etc. — anything the protocol marks as
    /// belonging to another window.
    pub xdg_toplevel_parent: Option<Weak<WlSurface>>,
    /// Most recent (width, height) seen via `xdg_toplevel.set_min_size`.
    /// (0, 0) means "no minimum declared". Combined with `xdg_max_size`
    /// to detect fixed-size windows (splash screens, GTK dialogs, file
    /// pickers) that should float rather than tile.
    pub xdg_min_size: (i32, i32),
    /// Most recent (width, height) seen via `xdg_toplevel.set_max_size`.
    /// (0, 0) means "no maximum declared". A window with a non-zero
    /// `min == max` is a fixed-size window — see
    /// `is_fixed_size_floater` for the routing rule that uses this.
    pub xdg_max_size: (i32, i32),
    /// Most recent rect set via `xdg_surface.set_window_geometry`,
    /// in surface-local coords (so within the buffer). When `Some`,
    /// it carves out the "logical" window inside a possibly-larger
    /// buffer — clients use this to draw decoration / shadow that
    /// extends outside the window's user-perceived bounds. Chrome
    /// and Firefox set this on popups so their drop-shadow rounds
    /// the menu. We use it to:
    /// - render the buffer at `(popup_x - geom.x, popup_y - geom.y)`
    ///   so the geometry rect aligns with the positioner's anchor
    ///   instead of the buffer origin, and
    /// - convert pointer events into surface-local coords by
    ///   adding `(geom.x, geom.y)` to the in-rect offset so the
    ///   client highlights the item the user is actually pointing
    ///   at.
    ///
    /// `None` → treat as `(0, 0, buffer_w, buffer_h)`.
    pub xdg_window_geometry: Option<(i32, i32, i32, i32)>,
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
                sd.pending.buffer = buffer;
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
                // Capture the client's "did pixel content actually
                // change?" signals before the buffer-swap block clears
                // them. Two ways for the client to say yes: re-attach
                // (new wl_buffer or same wl_buffer with intent of new
                // pixels) and/or damage (`wl_surface.damage` /
                // `damage_buffer` was called since the last commit).
                // A bare commit (no attach + no damage) is a no-op
                // for pixels — it's how clients deliver
                // wl_callback/ack_configure handshakes without asking
                // us to repaint.
                let attached_now = sd.pending_attach;
                let damaged_now = !sd.pending.damage.is_empty();
                let pixels_changed = attached_now || damaged_now;
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

                // Bump upload_seq when the client signalled a real
                // pixel change — `attach` (new buffer or
                // intent-to-replace) OR `damage` since last commit.
                // A bare commit (frame-callback ack / ack_configure)
                // doesn't change pixels and shouldn't invalidate any
                // upload cache. The cache is currently cursor-only
                // anyway (see vk_texture.rs / project memory note on
                // upload-skip), but keeping the gate correct here
                // means we're ready to re-enable a broader skip
                // path without compositor-side changes.
                //
                // Lazily populate the per-buffer cache entry on first
                // sight: smithay owns the wl_buffer's user-data slot,
                // so our `cache_key` + `upload_seq` live in a side
                // table keyed by the buffer's ObjectId. Eviction
                // happens in `BufferHandler::buffer_destroyed`.
                if pixels_changed {
                    if let Some(buf) = sd.current.buffer.as_ref() {
                        if crate::shm::is_shm_buffer(buf) {
                            let entry = state.shm_cache.entry(buf.id()).or_default();
                            entry.upload_seq.fetch_add(
                                1,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                    }
                }

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
                let new_buffer_id = sd.current.buffer.as_ref().map(|b| b.id());
                drop(sd);
                // old_buffer is Option<Option<WlBuffer>>: outer None means
                // no attach happened (nothing to release); outer Some
                // with inner None means we promoted but had no prior
                // buffer; outer Some(Some) is the actual replaced
                // buffer.
                if let Some(Some(old_buf)) = old_buffer {
                    if Some(old_buf.id()) != new_buffer_id {
                        old_buf.release();
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
                let mut just_mapped_toplevel = false;
                let mut should_unmap_toplevel = false;
                let mut layer_needs_configure = false;
                match &mut sd.role {
                    SurfaceRole::XdgToplevel { mapped } => {
                        if has_buffer {
                            if !*mapped {
                                *mapped = true;
                                just_mapped_toplevel = true;
                                // Dialog vs tile fork. A toplevel
                                // floats if it has a parent OR is
                                // fixed-size (min == max). Catches
                                // save/discard prompts, file pickers,
                                // GIMP's New Image dialog (parent
                                // arrives late but min/max are set
                                // early), and unparented splash
                                // screens (parent never set, but
                                // they're fixed-size). See
                                // `is_fixed_size_floater`.
                                let dialog_parent = sd
                                    .xdg_toplevel_parent
                                    .as_ref()
                                    .and_then(|w| w.upgrade().ok());
                                let fixed = crate::is_fixed_size_floater(&sd);
                                if dialog_parent.is_some() || fixed {
                                    tracing::info!(
                                        id = ?surface.id(),
                                        parent = ?dialog_parent.as_ref().map(|s| s.id()),
                                        fixed,
                                        "dialog mapped"
                                    );
                                    state.mapped_dialogs.push(crate::DialogWindow {
                                        surface: surface.downgrade(),
                                        parent: dialog_parent
                                            .as_ref()
                                            .map(|s| s.downgrade()),
                                        workspace: state.active_ws,
                                        position: None,
                                    });
                                } else {
                                    tracing::info!(id = ?surface.id(), "toplevel mapped");
                                    state.mapped_toplevels.push(crate::Window {
                                        surface: surface.downgrade(),
                                        rect: crate::Rect::default(),
                                        render_rect: crate::RectF::default(),
                                        velocity: crate::RectF::default(),
                                        // Open animation: window pops in from
                                        // 0% alpha + 70% scale up to fully
                                        // opaque + native size. Springs in
                                        // tick_animations drive both toward 1.
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
                            // Null-buffer commit unmaps per wl_surface
                            // spec. Drop from the layout immediately:
                            // alacritty (and others) unmap this way
                            // before issuing xdg_toplevel.destroy, and
                            // if we leave the entry in mapped_toplevels
                            // apply_layout keeps its tile rect reserved
                            // while collect_scene's mapped-filter skips
                            // drawing it — wallpaper bleeds through
                            // until destroy lands. The cleanup itself
                            // happens after we drop the SurfaceData
                            // lock below — `unmap_toplevel` picks the
                            // next-down-stack tile to focus and we
                            // can't re-enter the guard.
                            *mapped = false;
                            should_unmap_toplevel = true;
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
                        // Note: layer mapping deliberately does NOT set
                        // just_mapped_toplevel — taking keyboard focus is
                        // a toplevel concern. waybar / awww-daemon are
                        // exec-once'd before any toplevel maps, so if
                        // they grabbed focus on first map, the user's
                        // first Alt+Enter alacritty would never become
                        // focused (on_toplevel_mapped only sets focus
                        // when keyboard_focus is None). Layers with
                        // KeyboardInteractivity::Exclusive are still
                        // honoured per-frame by rebalance_keyboard_focus.
                        if !*initial_configure_sent && !has_buffer {
                            layer_needs_configure = true;
                            *initial_configure_sent = true;
                        } else if has_buffer && !*mapped && *initial_configure_sent {
                            *mapped = true;
                            tracing::info!(id = ?surface.id(), "layer surface mapped");
                        } else if !has_buffer && *mapped {
                            // Null buffer unmaps per spec.
                            *mapped = false;
                        }
                    }
                    SurfaceRole::XdgPopup { mapped, .. } => {
                        if has_buffer && !*mapped {
                            *mapped = true;
                            tracing::info!(id = ?surface.id(), "xdg_popup mapped");
                            state.mapped_popups.push(surface.downgrade());
                        } else if !has_buffer && *mapped {
                            *mapped = false;
                            state
                                .mapped_popups
                                .retain(|w| w.upgrade().ok().as_ref() != Some(surface));
                        }
                    }
                    SurfaceRole::None | SurfaceRole::Subsurface => {}
                }
                drop(sd);

                if should_unmap_toplevel {
                    crate::unmap_toplevel(state, surface);
                }

                // For an already-mapped toplevel, this commit may be
                // the one that lands a buffer at the size we just
                // configured. If so, kick off the resize tween now —
                // see `settle_pending_layout` for the rationale.
                // Skipped for newly mapped toplevels: tick_animations
                // snaps them on the first frame anyway.
                if has_buffer && !just_mapped_toplevel {
                    crate::settle_pending_layout(state, surface);
                }

                if layer_needs_configure {
                    crate::layer_shell::send_initial_configure(state, surface);
                }

                if just_mapped_toplevel {
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

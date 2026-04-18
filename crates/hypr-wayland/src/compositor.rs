//! wl_compositor + wl_surface + wl_region + wl_callback.
//!
//! Surfaces hold pending and current double-buffered state. `commit` promotes
//! pending into current and, if the surface has an xdg_toplevel role with a
//! buffer, marks the scene dirty so the compositor re-renders.

use std::sync::{Arc, Mutex};

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

/// Roles a surface can take on. Only xdg_toplevel at B2.
#[derive(Clone, Debug, Default)]
pub enum SurfaceRole {
    #[default]
    None,
    XdgToplevel {
        mapped: bool,
    },
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
    pub frame_callbacks: Vec<WlCallback>,
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
                // Promote pending -> current.
                sd.current.buffer = sd.pending.buffer.take();
                sd.current.damage = std::mem::take(&mut sd.pending.damage);

                // Mark as mapped once we have a buffer + role.
                let has_buffer = sd.current.buffer.is_some();
                if let SurfaceRole::XdgToplevel { mapped } = &mut sd.role {
                    if has_buffer {
                        if !*mapped {
                            *mapped = true;
                            tracing::info!(id = ?surface.id(), "toplevel mapped");
                            state
                                .mapped_toplevels
                                .push(surface.downgrade());
                        }
                    } else {
                        *mapped = false;
                    }
                }
                drop(sd);

                // Fire frame callbacks immediately (headless, no real vblank).
                let callbacks = std::mem::take(&mut data.lock().unwrap().frame_callbacks);
                let ts = state.elapsed_ms();
                for cb in callbacks {
                    cb.done(ts);
                }

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

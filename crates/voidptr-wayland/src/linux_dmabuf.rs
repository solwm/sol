//! zwp_linux_dmabuf_v1 — dmabuf-backed wl_buffers.
//!
//! B10.2 scope: protocol skeleton only. We advertise the global with a
//! minimal format/modifier table (ARGB8888 + XRGB8888, LINEAR) so EGL
//! clients pick this path over SHM. Plane info is collected via
//! `create_params` → `add`, and `create_immed` hands back a `wl_buffer`
//! whose user-data is a `DmabufBuffer` struct. We do NOT import the
//! dmabuf server-side yet — rendering stays empty for dmabuf-backed
//! surfaces until B10.3/B10.4 land EGLImage import and the renderer fork.

use std::os::fd::OwnedFd;
use std::sync::Mutex;

use wayland_protocols::wp::linux_dmabuf::zv1::server::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::{self, ZwpLinuxDmabufV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_buffer::{self, WlBuffer},
};

use crate::State;

/// DRM fourcc: ARGB8888 ('AR24').
pub const FOURCC_ARGB8888: u32 = 0x34325241;
/// DRM fourcc: XRGB8888 ('XR24').
pub const FOURCC_XRGB8888: u32 = 0x34325258;
/// Linear layout — no tiling, no compression.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// Sentinel for "no explicit modifier; server accepts whatever the driver
/// uses as the implicit layout." Advertising this alongside LINEAR lets
/// Mesa pick a fast tiled format on supported GPUs (radv, anv, etc.)
/// without us having to enumerate driver-specific modifiers by hand.
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// Version 3 is the widely-supported level and avoids the v4+ feedback
/// object machinery. B10.5 can bump to v4 if we want per-surface feedback.
pub const DMABUF_VERSION: u32 = 3;

#[derive(Debug)]
pub struct DmabufPlane {
    pub fd: OwnedFd,
    pub plane_idx: u32,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
}

/// Mutable state of an in-progress `zwp_linux_buffer_params_v1` object.
/// The Mutex is internal so the Dispatch impl can take an immutable ref.
#[derive(Default)]
pub struct ParamsData {
    pub inner: Mutex<ParamsInner>,
}

#[derive(Default)]
pub struct ParamsInner {
    pub planes: Vec<DmabufPlane>,
    pub consumed: bool,
}

/// User-data attached to a dmabuf-backed `wl_buffer`. Stores the plane
/// descriptors plus logical pixel dimensions and the DRM fourcc. Rendering
/// code (from B10.3 onwards) will import these into an EGLImage on first
/// use.
#[derive(Debug)]
pub struct DmabufBuffer {
    pub width: i32,
    pub height: i32,
    pub format: u32,
    pub planes: Vec<DmabufPlane>,
}

impl GlobalDispatch<ZwpLinuxDmabufV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpLinuxDmabufV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let d = init.init(resource, ());
        tracing::info!(id = ?d.id(), version = d.version(), "bind zwp_linux_dmabuf_v1");

        // v1–v3 legacy format / modifier events. Clients binding at v4+
        // expect to use get_default_feedback; we only advertise up to v3.
        d.format(FOURCC_ARGB8888);
        d.format(FOURCC_XRGB8888);
        if d.version() >= 3 {
            for modifier in [DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_MOD_INVALID] {
                let mhi = (modifier >> 32) as u32;
                let mlo = (modifier & 0xffff_ffff) as u32;
                d.modifier(FOURCC_ARGB8888, mhi, mlo);
                d.modifier(FOURCC_XRGB8888, mhi, mlo);
            }
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpLinuxDmabufV1,
        request: zwp_linux_dmabuf_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_linux_dmabuf_v1::Request::CreateParams { params_id } => {
                let _ = init.init(params_id, ParamsData::default());
                tracing::debug!("dmabuf.create_params");
            }
            zwp_linux_dmabuf_v1::Request::Destroy => {}
            // v4 feedback objects — we advertise v3 so clients shouldn't
            // reach here. If they do, just log and drop.
            _ => {
                tracing::warn!("unhandled zwp_linux_dmabuf_v1 request");
            }
        }
    }
}

impl Dispatch<ZwpLinuxBufferParamsV1, ParamsData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &ZwpLinuxBufferParamsV1,
        request: zwp_linux_buffer_params_v1::Request,
        data: &ParamsData,
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_linux_buffer_params_v1::Request::Add {
                fd,
                plane_idx,
                offset,
                stride,
                modifier_hi,
                modifier_lo,
            } => {
                let modifier = ((modifier_hi as u64) << 32) | modifier_lo as u64;
                let mut inner = data.inner.lock().unwrap();
                if inner.consumed {
                    resource.post_error(
                        zwp_linux_buffer_params_v1::Error::AlreadyUsed,
                        "params already consumed",
                    );
                    return;
                }
                inner.planes.push(DmabufPlane {
                    fd,
                    plane_idx,
                    offset,
                    stride,
                    modifier,
                });
                tracing::debug!(plane_idx, offset, stride, modifier, "dmabuf_params.add");
            }
            zwp_linux_buffer_params_v1::Request::CreateImmed {
                buffer_id,
                width,
                height,
                format,
                flags: _,
            } => {
                let planes = {
                    let mut inner = data.inner.lock().unwrap();
                    if inner.consumed {
                        resource.post_error(
                            zwp_linux_buffer_params_v1::Error::AlreadyUsed,
                            "params already consumed",
                        );
                        return;
                    }
                    inner.consumed = true;
                    std::mem::take(&mut inner.planes)
                };
                tracing::info!(
                    width,
                    height,
                    format = format_as_str(format),
                    planes = planes.len(),
                    "dmabuf.create_immed -> wl_buffer"
                );
                let _ = init.init(
                    buffer_id,
                    DmabufBuffer {
                        width,
                        height,
                        format,
                        planes,
                    },
                );
            }
            zwp_linux_buffer_params_v1::Request::Create {
                width: _,
                height: _,
                format: _,
                flags: _,
            } => {
                // Async form sends `created`/`failed` events. Implementing
                // `created` requires constructing a wl_buffer resource on
                // the client's side without a DataInit; Mesa/alacritty use
                // create_immed anyway, so B10.2 fails this path and lets
                // the client fall back.
                let mut inner = data.inner.lock().unwrap();
                inner.consumed = true;
                resource.failed();
                tracing::debug!("dmabuf_params.create (async) -> failed (use create_immed)");
            }
            zwp_linux_buffer_params_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlBuffer, DmabufBuffer> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlBuffer,
        request: wl_buffer::Request,
        _data: &DmabufBuffer,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        if let wl_buffer::Request::Destroy = request {
            tracing::debug!("dmabuf wl_buffer destroyed");
        }
    }
}

fn format_as_str(f: u32) -> &'static str {
    match f {
        FOURCC_ARGB8888 => "ARGB8888",
        FOURCC_XRGB8888 => "XRGB8888",
        _ => "unknown",
    }
}

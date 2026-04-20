//! zwp_linux_dmabuf_v1 — dmabuf-backed wl_buffers.
//!
//! Advertises the global at version 4 with full `zwp_linux_dmabuf_feedback_v1`
//! support. Mesa's EGL on Wayland requires knowing the `main_device`
//! (the DRM device the compositor drives) before it can open a render
//! node and use hardware GL — without feedback it falls back to llvmpipe
//! software rasterization, writes SHM buffers, and everything looks
//! terrible. The feedback object hands over the format+modifier table
//! (via memfd) and the device dev_t so Mesa picks the correct path.
//!
//! Legacy `format`/`modifier` events are still emitted for v1-v3 binds.
//! Plane info collected via `create_params → add → create_immed` becomes
//! a `DmabufBuffer` attached to the new `wl_buffer` as user-data. Import
//! into GL textures happens in `voidptr_backend_drm::presenter`.

use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::sync::Mutex;

use anyhow::{Context, Result};
use rustix::fs::{MemfdFlags, ftruncate, memfd_create};
use wayland_protocols::wp::linux_dmabuf::zv1::server::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_feedback_v1::{self, ZwpLinuxDmabufFeedbackV1},
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

/// Version 4 introduces per-surface feedback with main_device — required
/// for Mesa EGL on Wayland to pick the hardware GL driver instead of
/// falling back to llvmpipe.
pub const DMABUF_VERSION: u32 = 4;

/// Format + modifier pairs we advertise. `(format, modifier)`. Keep this
/// minimal: Mesa deadlocks on empty buffer commits if it can't satisfy
/// any advertised tranche with the paths it actually supports on the
/// main device. XRGB8888 with INVALID (implicit) modifier is the
/// maximally-compatible choice — Mesa lets the driver pick whatever
/// tiling it prefers — and LINEAR is the universal fallback.
fn supported_format_pairs() -> &'static [(u32, u64)] {
    &[
        (FOURCC_XRGB8888, DRM_FORMAT_MOD_INVALID),
        (FOURCC_ARGB8888, DRM_FORMAT_MOD_INVALID),
        (FOURCC_XRGB8888, DRM_FORMAT_MOD_LINEAR),
        (FOURCC_ARGB8888, DRM_FORMAT_MOD_LINEAR),
    ]
}

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

        // Legacy format / modifier events for v1-v3 clients. The spec marks
        // these deprecated at v4+ (clients use feedback instead); send them
        // only for older binds so v4 clients get a clean feedback flow.
        if d.version() < 4 {
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
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for State {
    fn request(
        state: &mut Self,
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
            zwp_linux_dmabuf_v1::Request::GetDefaultFeedback { id } => {
                let fb = init.init(id, ());
                if let Err(e) = send_feedback(&fb, state.drm_device_path.as_deref()) {
                    tracing::warn!(error = %e, "dmabuf get_default_feedback failed");
                } else {
                    tracing::info!("dmabuf.get_default_feedback -> sent");
                }
            }
            zwp_linux_dmabuf_v1::Request::GetSurfaceFeedback { id, surface: _ } => {
                let fb = init.init(id, ());
                if let Err(e) = send_feedback(&fb, state.drm_device_path.as_deref()) {
                    tracing::warn!(error = %e, "dmabuf get_surface_feedback failed");
                } else {
                    tracing::info!("dmabuf.get_surface_feedback -> sent");
                }
            }
            zwp_linux_dmabuf_v1::Request::Destroy => {}
            _ => {
                tracing::warn!("unhandled zwp_linux_dmabuf_v1 request");
            }
        }
    }
}

impl Dispatch<ZwpLinuxDmabufFeedbackV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpLinuxDmabufFeedbackV1,
        request: zwp_linux_dmabuf_feedback_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        if let zwp_linux_dmabuf_feedback_v1::Request::Destroy = request {
            // feedback is stateless for us — nothing to release.
        }
    }
}

/// Build + send a complete feedback event sequence:
/// format_table → main_device → tranche_target_device → tranche_formats →
/// tranche_flags → tranche_done → done. Mesa relies on main_device to
/// pick the hardware GL path; without it, EGL-on-Wayland falls back to
/// llvmpipe.
fn send_feedback(
    fb: &ZwpLinuxDmabufFeedbackV1,
    drm_device_path: Option<&std::path::Path>,
) -> Result<()> {
    let device_path = drm_device_path
        .ok_or_else(|| anyhow::anyhow!("no DRM device path on State (headless mode?)"))?;

    // 1. Format table in a memfd. Each entry is 16 bytes: u32 format, u32
    //    padding, u64 modifier. Mesa mmaps this read-only and indexes by
    //    u16 position sent later in tranche_formats.
    let pairs = supported_format_pairs();
    let mut table = Vec::with_capacity(pairs.len() * 16);
    for (fmt, modifier) in pairs {
        table.extend_from_slice(&fmt.to_ne_bytes());
        table.extend_from_slice(&0u32.to_ne_bytes());
        table.extend_from_slice(&modifier.to_ne_bytes());
    }
    let fd = memfd_create("voidptr-dmabuf-format-table", MemfdFlags::CLOEXEC)
        .context("memfd_create")?;
    ftruncate(&fd, table.len() as u64).context("ftruncate format table")?;
    {
        let mut f = std::fs::File::from(fd);
        f.write_all(&table).context("write format table")?;
        fb.format_table(f.as_fd(), table.len() as u32);
    } // f closes here; wayland-rs already dup'd the fd.

    // 2. main_device: dev_t of the DRM device we drive. Mesa uses this to
    //    open the matching render node via libdrm.
    let stat = rustix::fs::stat(device_path).context("stat drm device")?;
    let rdev: u64 = stat.st_rdev;
    let dev_bytes = rdev.to_ne_bytes().to_vec();
    fb.main_device(dev_bytes.clone());

    // 3. One tranche, same device, with all our format/modifier indices.
    fb.tranche_target_device(dev_bytes);
    let mut indices = Vec::with_capacity(pairs.len() * 2);
    for i in 0..pairs.len() as u16 {
        indices.extend_from_slice(&i.to_ne_bytes());
    }
    fb.tranche_formats(indices);
    fb.tranche_flags(zwp_linux_dmabuf_feedback_v1::TrancheFlags::empty());
    fb.tranche_done();
    fb.done();

    tracing::info!(
        device = %device_path.display(),
        rdev,
        pairs = pairs.len(),
        "dmabuf feedback sent"
    );
    Ok(())
}

impl Dispatch<ZwpLinuxBufferParamsV1, ParamsData> for State {
    fn request(
        _state: &mut Self,
        client: &Client,
        resource: &ZwpLinuxBufferParamsV1,
        request: zwp_linux_buffer_params_v1::Request,
        data: &ParamsData,
        dh: &DisplayHandle,
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
                width,
                height,
                format,
                flags: _,
            } => {
                // Async form: server-side allocates a wl_buffer resource on
                // behalf of the client, then signals via `created(buffer)`.
                // Mesa uses this path on Wayland (not create_immed); failing
                // it was why alacritty's EGL allocator stalled and the
                // client never committed a frame.
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
                match client.create_resource::<WlBuffer, DmabufBuffer, State>(
                    dh,
                    1,
                    DmabufBuffer {
                        width,
                        height,
                        format,
                        planes,
                    },
                ) {
                    Ok(buffer) => {
                        resource.created(&buffer);
                        tracing::info!(
                            width,
                            height,
                            format = format_as_str(format),
                            "dmabuf.create -> wl_buffer (async)"
                        );
                    }
                    Err(e) => {
                        resource.failed();
                        tracing::warn!(error = ?e, "dmabuf.create: create_resource failed");
                    }
                }
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

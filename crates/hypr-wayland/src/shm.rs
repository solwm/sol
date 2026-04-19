//! wl_shm + wl_shm_pool + wl_buffer.
//!
//! Pools are mmapped read-only from the server's side; clients retain their
//! own writable mapping via the fd they passed in, so buffer updates propagate
//! through the shared memory region without server-side copies.
//!
//! Resize is not yet supported (B2 clients either pick a size up-front or
//! allocate a larger pool). Attempting to resize just logs a warning.

use std::os::fd::OwnedFd;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::{
        wl_buffer::{self, WlBuffer},
        wl_shm::{self, Format, WlShm},
        wl_shm_pool::{self, WlShmPool},
    },
};

use crate::State;
use hypr_core::PixelFormat;

/// User data attached to a wl_shm_pool. Holds the mmap so buffers can read
/// pixels even after the client destroys the pool resource (spec: pool is
/// just a name for the mapping, buffers keep the mapping alive).
pub struct PoolData {
    pub mmap: Arc<Mmap>,
    pub size: usize,
}

/// User data attached to a wl_buffer. Buffers clone the pool's `Arc<Mmap>` so
/// they outlive the wl_shm_pool resource if needed.
#[derive(Clone)]
pub struct BufferData {
    pub mmap: Arc<Mmap>,
    pub offset: i32,
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    pub format: Format,
}

impl BufferData {
    pub fn pixel_format(&self) -> Option<PixelFormat> {
        match self.format {
            Format::Argb8888 => Some(PixelFormat::Argb8888),
            Format::Xrgb8888 => Some(PixelFormat::Xrgb8888),
            _ => None,
        }
    }

    /// Borrow the pixel bytes for this buffer. Returns None if the buffer's
    /// range escapes the mapping (malicious or broken client).
    pub fn bytes(&self) -> Option<&[u8]> {
        let offset = self.offset as usize;
        let len = self.stride as usize * self.height as usize;
        let end = offset.checked_add(len)?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[offset..end])
    }
}

impl GlobalDispatch<WlShm, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlShm>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let shm = init.init(resource, ());
        shm.format(Format::Argb8888);
        shm.format(Format::Xrgb8888);
        tracing::info!(id = ?shm.id(), "bind wl_shm");
    }
}

impl Dispatch<WlShm, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        shm: &WlShm,
        request: wl_shm::Request,
        _data: &(),
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, fd, size } = request {
            match mmap_pool(fd, size) {
                Ok(pool_data) => {
                    let _ = init.init(id, pool_data);
                    tracing::debug!(size, "wl_shm.create_pool");
                }
                Err(e) => {
                    shm.post_error(wl_shm::Error::InvalidFd, format!("mmap failed: {e}"));
                }
            }
        }
    }
}

fn mmap_pool(fd: OwnedFd, size: i32) -> anyhow::Result<PoolData> {
    if size <= 0 {
        anyhow::bail!("non-positive pool size {size}");
    }
    let mmap = unsafe { MmapOptions::new().len(size as usize).map(&fd)? };
    Ok(PoolData {
        mmap: Arc::new(mmap),
        size: size as usize,
    })
}

impl Dispatch<WlShmPool, PoolData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        pool: &WlShmPool,
        request: wl_shm_pool::Request,
        data: &PoolData,
        _dh: &DisplayHandle,
        init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_shm_pool::Request::CreateBuffer {
                id,
                offset,
                width,
                height,
                stride,
                format,
            } => {
                let format = match format.into_result() {
                    Ok(f) => f,
                    Err(_) => {
                        pool.post_error(wl_shm::Error::InvalidFormat, "unknown format");
                        return;
                    }
                };
                let buf = BufferData {
                    mmap: data.mmap.clone(),
                    offset,
                    width,
                    height,
                    stride,
                    format,
                };
                let _ = init.init(id, buf);
                tracing::trace!(width, height, stride, ?format, "wl_shm_pool.create_buffer");
            }
            wl_shm_pool::Request::Resize { size: _ } => {
                tracing::warn!("wl_shm_pool.resize ignored at B2");
            }
            wl_shm_pool::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlBuffer, BufferData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlBuffer,
        request: wl_buffer::Request,
        _data: &BufferData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        if let wl_buffer::Request::Destroy = request {
            // No extra bookkeeping; user data drop releases the Arc<Mmap>.
        }
    }
}

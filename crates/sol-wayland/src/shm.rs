//! wl_shm + wl_shm_pool + wl_buffer.
//!
//! Pools are mmapped read-only from the server's side; clients retain their
//! own writable mapping via the fd they passed in, so buffer updates propagate
//! through the shared memory region without server-side copies. Resize is
//! supported (grow only, per spec); existing wl_buffers keep their original
//! mapping via their private `Arc<Mmap>` ref, while subsequently-created
//! buffers allocate against the new, larger mapping.

use std::os::fd::OwnedFd;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

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
use sol_core::PixelFormat;

/// User data attached to a wl_shm_pool. The mmap lives behind a mutex so the
/// Resize request (which sees only `&PoolData` via the Dispatch signature)
/// can swap it for a bigger mapping.
pub struct PoolData {
    pub inner: Mutex<PoolInner>,
    /// Kept alive so Resize can re-mmap at a larger size.
    pub fd: OwnedFd,
}

pub struct PoolInner {
    pub mmap: Arc<Mmap>,
    pub size: usize,
}

/// User data attached to a wl_buffer. Buffers clone the pool's `Arc<Mmap>` so
/// they outlive the wl_shm_pool resource if needed.
pub struct BufferData {
    pub mmap: Arc<Mmap>,
    pub offset: i32,
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    pub format: Format,
    /// Stable, globally unique ID for the texture cache. Pulled from
    /// `crate::next_buffer_cache_key` at construction. We can't key
    /// the cache by `(self as *const _) as u64` — wayland-server
    /// boxes BufferData in heap memory that gets reused as soon as
    /// the resource is dropped, and a new buffer landing at the same
    /// address before the eviction queue drains aliases to the old
    /// texture. Symptom of that aliasing: two windows briefly render
    /// the same content during rapid resize/move churn.
    pub cache_key: u64,
    /// Bumped on each wl_surface.commit that promotes this buffer
    /// from pending to current. Backends compare against their cached
    /// per-key value to decide whether the SHM upload is redundant.
    /// Starts at 1 so first-frame uploads (which see a default-zero
    /// cached value) always run.
    pub upload_seq: AtomicU64,
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
        inner: Mutex::new(PoolInner {
            mmap: Arc::new(mmap),
            size: size as usize,
        }),
        fd,
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
                let mmap = {
                    let inner = data.inner.lock().unwrap();
                    inner.mmap.clone()
                };
                let buf = BufferData {
                    mmap,
                    offset,
                    width,
                    height,
                    stride,
                    format,
                    cache_key: crate::next_buffer_cache_key(),
                    upload_seq: AtomicU64::new(1),
                };
                let _ = init.init(id, buf);
                tracing::trace!(width, height, stride, ?format, "wl_shm_pool.create_buffer");
            }
            wl_shm_pool::Request::Resize { size } => {
                if size <= 0 {
                    pool.post_error(
                        wl_shm::Error::InvalidStride,
                        "wl_shm_pool.resize: non-positive size",
                    );
                    return;
                }
                let new_size = size as usize;
                let mut inner = data.inner.lock().unwrap();
                if new_size <= inner.size {
                    // Spec: resize can only grow. Silently ignore
                    // shrink — clients sometimes send the same size
                    // after a buffer destroy; not a protocol error.
                    return;
                }
                match unsafe { MmapOptions::new().len(new_size).map(&data.fd) } {
                    Ok(new_mmap) => {
                        inner.mmap = Arc::new(new_mmap);
                        inner.size = new_size;
                        tracing::debug!(new_size, "wl_shm_pool.resize applied");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, new_size, "wl_shm_pool.resize: mmap failed");
                    }
                }
            }
            wl_shm_pool::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlBuffer, BufferData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WlBuffer,
        request: wl_buffer::Request,
        data: &BufferData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        if let wl_buffer::Request::Destroy = request {
            // Queue the cache entry for eviction on the next render
            // tick using the stable counter-based key (see
            // BufferData::cache_key for why a pointer won't do).
            state.pending_texture_evictions.push(data.cache_key);
        }
    }
}

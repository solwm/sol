//! wl_shm + wl_shm_pool + wl_buffer.
//!
//! Backed by smithay's `ShmState`. The `WlShm` global, pool mmap, and
//! per-buffer wire dispatch (Create/Resize/Destroy) all live in
//! smithay; we keep a parallel `shm_cache` side table on `State`,
//! keyed by the `wl_buffer`'s `ObjectId`, that holds the two pieces
//! of compositor-owned per-buffer state we still need:
//!
//! - `cache_key`: stable, never-recycled u64 the DRM presenter uses
//!   to index its texture cache. We can't key the cache by anything
//!   the wayland-server crate hands out for free — `WlBuffer`'s
//!   address is recycled by the heap allocator and the protocol's
//!   integer ID is recycled by the protocol itself, both of which
//!   alias old textures to new buffers during rapid resize churn.
//! - `upload_seq`: bumped on each `wl_surface.commit` that promotes
//!   a real pixel change. Backends compare against their cached
//!   per-key value to decide whether to re-upload (currently only
//!   the cursor sentinel actually consults this — see the
//!   `project_shm_upload_skip_disabled` memory note).
//!
//! Cache entries are populated lazily at commit time in
//! `compositor.rs` and removed by `BufferHandler::buffer_destroyed`
//! in `lib.rs`, which also queues the `cache_key` into
//! `pending_texture_evictions` for the next render tick to act on.

use std::sync::atomic::AtomicU64;

use smithay::wayland::shm::{
    BufferAccessError, BufferData as SmithayShmData, ShmBufferUserData, with_buffer_contents,
};
use wayland_server::Resource;
use wayland_server::protocol::wl_buffer::WlBuffer;
use wayland_server::protocol::wl_shm::Format;

use sol_core::PixelFormat;

/// Per-buffer compositor-owned state. Lives in `State::shm_cache`
/// keyed by the `WlBuffer`'s `ObjectId`.
#[derive(Debug)]
pub struct ShmCacheEntry {
    pub cache_key: u64,
    pub upload_seq: AtomicU64,
}

impl ShmCacheEntry {
    pub fn new() -> Self {
        Self {
            cache_key: crate::next_buffer_cache_key(),
            // Starts at 1 so first-frame uploads (which see a default-
            // zero cached value) always run.
            upload_seq: AtomicU64::new(1),
        }
    }
}

impl Default for ShmCacheEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Sub-rect metadata returned alongside a borrowed pixel slice.
/// Mirrors smithay's `BufferData` minus the wayland-server format
/// wrapping; we translate the format into our own `PixelFormat` enum
/// so callers outside this module never see the wire-protocol type.
#[derive(Copy, Clone, Debug)]
pub struct ShmMeta {
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    pub format: PixelFormat,
}

/// True if `buf` is an SHM buffer (vs a dmabuf or other kind).
/// Used at commit time to gate the `shm_cache` insert and at
/// hit-test time so dmabuf surfaces fall through to their own
/// metadata path.
pub fn is_shm_buffer(buf: &WlBuffer) -> bool {
    buf.data::<ShmBufferUserData>().is_some()
}

/// Borrow this wl_buffer's pixel bytes plus its layout metadata.
/// Returns `None` if the buffer isn't SHM, the format isn't one we
/// know how to sample, or smithay refused the access (bad pool size).
///
/// The returned slice borrows from `buf` for `'b`. Smithay's
/// `with_buffer_contents` is closure-scoped — it only documents the
/// pointer as valid for the duration of the call — but the pool's
/// mmap is held behind an `Arc<Pool>` stored in the wl_buffer's
/// `ShmBufferUserData`, so it stays alive as long as the `WlBuffer`
/// resource itself, which outlives `'b`. We bridge from the closure
/// back to the buffer's borrow with a `transmute`. The compositor is
/// single-threaded; a client mutating its mapping concurrently with
/// this read is the same race smithay's own clients race against, and
/// the worst case is a torn frame, not memory unsafety.
pub fn shm_pixels<'b>(buf: &'b WlBuffer) -> Option<(&'b [u8], ShmMeta)> {
    let result = with_buffer_contents(buf, |ptr, len, data: SmithayShmData| {
        let format = match data.format {
            Format::Argb8888 => PixelFormat::Argb8888,
            Format::Xrgb8888 => PixelFormat::Xrgb8888,
            _ => return None,
        };
        let start = data.offset as usize;
        let extent = (data.stride as usize).checked_mul(data.height as usize)?;
        let end = start.checked_add(extent)?;
        if end > len {
            return None;
        }
        // SAFETY: see function-level doc — the pool's mmap outlives
        // the WlBuffer borrow, and we slice within `[start, end)`
        // which is bounds-checked above.
        let slice: &[u8] = unsafe { std::slice::from_raw_parts(ptr.add(start), extent) };
        let bytes: &'b [u8] = unsafe { std::mem::transmute::<&[u8], &'b [u8]>(slice) };
        Some((
            bytes,
            ShmMeta {
                width: data.width,
                height: data.height,
                stride: data.stride,
                format,
            },
        ))
    });
    match result {
        Ok(opt) => opt,
        Err(BufferAccessError::NotManaged) => None,
        Err(e) => {
            tracing::warn!(error = ?e, "shm_pixels: unexpected access error");
            None
        }
    }
}

/// Buffer dimensions for SHM. Cheaper than `shm_pixels` because the
/// closure doesn't dereference the mmap — used by hit-testing where
/// we only need the surface's intrinsic area.
pub fn shm_dims(buf: &WlBuffer) -> Option<(i32, i32)> {
    with_buffer_contents(buf, |_ptr, _len, data: SmithayShmData| {
        (data.width, data.height)
    })
    .ok()
}

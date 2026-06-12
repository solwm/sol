//! Per-buffer-key texture cache.
//!
//! For each `SceneElement::buffer_key` the renderer keeps one entry holding
//! a `VkImage` + view + descriptor set; the entry is reused across frames
//! so re-painting a long-lived `wl_buffer` is the same `vkCmdBindDescriptorSets`
//! every time. Two backings:
//!
//! - **SHM** — host-visible staging buffer + `vkCmdCopyBufferToImage` into
//!   a device-local `VK_FORMAT_B8G8R8A8_UNORM` image. The cursor sentinel
//!   is on the cached path: same `upload_seq` ⇒ skip the copy entirely.
//!   Other SHM buffers re-upload every commit (Chrome's repaint cadence
//!   violates the cache assumption — the cause was never pinned down,
//!   the workaround is to stay narrow).
//! - **dmabuf** — import via `VK_EXT_external_memory_dma_buf` +
//!   `VK_EXT_image_drm_format_modifier`. Once-only per `wl_buffer`.

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use sol_core::{CURSOR_SCENE_KEY, RenderTiming, SceneContent, SceneElement};
use std::collections::HashMap;
use std::os::fd::{IntoRawFd, OwnedFd};

use crate::vk_stack::SharedStack;

/// Vulkan format we always use for client-buffer textures, matching
/// little-endian Wayland ARGB / XRGB and Linux DRM `AR24` / `XR24`
/// dmabufs (BGRA byte order). The shader's sampled value comes back
/// straight as `(R, G, B, A)` — no swizzle needed.
pub const TEXTURE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub struct TextureEntry {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub descriptor: vk::DescriptorSet,
    /// Pool `descriptor` was allocated from — frees must go back to
    /// the originating pool now that the cache grows new pools on
    /// exhaustion.
    pub descriptor_pool: vk::DescriptorPool,
    pub width: i32,
    pub height: i32,
    /// Last `SceneContent::Shm::upload_seq` we copied for this entry.
    /// Compared against the next frame's seq to skip the upload when
    /// the cursor sentinel matches.
    pub uploaded_seq: u64,
    /// True iff this entry was imported from a dmabuf (no per-frame
    /// upload work; image is read-only from our side).
    pub is_dmabuf: bool,
    /// Persistent host-visible staging buffer for SHM uploads. Sized
    /// to `width * height * 4`; reallocated when the surface resizes.
    /// `None` for dmabuf entries.
    pub staging: Option<StagingBuf>,
}

pub struct StagingBuf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /// Persistent CPU mapping. SHM uploads memcpy into this slice
    /// and the matching `vkCmdCopyBufferToImage` runs in the per-frame
    /// command buffer. HOST_COHERENT memory means no manual flushes.
    pub mapped: *mut u8,
}

// SAFETY: the mapped pointer is into Vulkan-owned memory we hold for
// the lifetime of the StagingBuf; it's not Send/Sync by default just
// because of the raw pointer. The texture cache is single-threaded
// (the whole compositor is) so this never crosses threads.
unsafe impl Send for StagingBuf {}
unsafe impl Sync for StagingBuf {}

/// Tiny upload record: which entry, where to copy from, what region.
/// Built up during `prepare_uploads` and consumed by `record_uploads`
/// when we encode the per-frame command buffer.
pub struct PendingUpload {
    pub key: u64,
    pub width: u32,
    pub height: u32,
    /// Old image layout (UNDEFINED for fresh allocations,
    /// SHADER_READ_ONLY_OPTIMAL for re-uploads).
    pub from_layout: vk::ImageLayout,
}

pub struct TextureCache {
    stack: SharedStack,
    pub entries: HashMap<u64, TextureEntry>,
    /// Descriptor pools, oldest first; allocations come from the last
    /// one and a fresh pool is pushed when it runs out. Old pools
    /// stick around because long-lived sets (blur FBOs, cached
    /// textures) still live in them.
    descriptor_pools: Vec<vk::DescriptorPool>,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    pending: Vec<PendingUpload>,
    /// buffer_keys whose dmabuf import failed. The wl_buffer (and its
    /// key) are immutable once created, so retrying can't succeed —
    /// without this the dup + create_image + warn cycle repeats every
    /// frame for as long as the buffer stays in the scene. Cleared by
    /// `evict` when the buffer dies.
    failed_imports: std::collections::HashSet<u64>,
    /// Sync-file fds exported from dmabufs whose content advanced
    /// this frame (commit_seq changed) — each carries the producer's
    /// pending GPU writes. The presenter imports them as submit wait
    /// semaphores so our sampling can't race the client's rendering.
    /// NVIDIA does not reliably honour dma-buf implicit sync for
    /// sampled imports of *re-used* buffers; the symptom was the
    /// clear color bleeding through the wallpaper's last-written
    /// rows (and the blur backdrop built from them) during motion.
    pending_wait_fences: Vec<OwnedFd>,
}

impl TextureCache {
    /// Per-pool capacity — generous enough that even a busy session
    /// with dozens of clients + the blur FBOs (which also pull from
    /// this pool) fits in one. When a pool does run out, a new one is
    /// pushed and allocation continues there.
    const POOL_CAPACITY: u32 = 256;

    fn create_pool(stack: &SharedStack) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(Self::POOL_CAPACITY)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(Self::POOL_CAPACITY)
            .pool_sizes(&pool_sizes);
        unsafe {
            stack
                .device
                .create_descriptor_pool(&pool_info, None)
                .context("create_descriptor_pool (texture cache)")
        }
    }

    pub fn new(
        stack: SharedStack,
        set_layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
    ) -> Result<Self> {
        let descriptor_pools = vec![Self::create_pool(&stack)?];
        Ok(Self {
            stack,
            entries: HashMap::new(),
            descriptor_pools,
            set_layout,
            sampler,
            pending: Vec::new(),
            failed_imports: std::collections::HashSet::new(),
            pending_wait_fences: Vec::new(),
        })
    }

    /// Allocate a fresh descriptor set + immediately point it at the
    /// given image view. Used by the texture cache for client buffers
    /// and by the blur module for the capture / ping / pong FBOs.
    /// Returns the set together with the pool it came from (frees
    /// must target the originating pool). Grows a new pool when the
    /// current one is exhausted.
    pub fn allocate_descriptor(
        &mut self,
        view: vk::ImageView,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool)> {
        let layouts = [self.set_layout];
        let set = loop {
            let pool = *self
                .descriptor_pools
                .last()
                .expect("at least one descriptor pool always exists");
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            match unsafe { self.stack.device.allocate_descriptor_sets(&alloc_info) } {
                Ok(sets) => break (sets[0], pool),
                Err(
                    vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL,
                ) => {
                    tracing::info!(
                        pools = self.descriptor_pools.len() + 1,
                        "descriptor pool exhausted; growing"
                    );
                    let fresh = Self::create_pool(&self.stack)?;
                    self.descriptor_pools.push(fresh);
                    // Loop retries against the fresh pool, which
                    // cannot itself be exhausted.
                }
                Err(e) => return Err(e).context("allocate_descriptor_sets"),
            }
        };
        let (set, pool) = set;
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];
        unsafe {
            self.stack.device.update_descriptor_sets(&writes, &[]);
        }
        Ok((set, pool))
    }

    /// Drop the cached entry for `key`. Caller must have already
    /// queued a `vkDeviceWaitIdle` (or otherwise be sure the GPU is
    /// no longer using the resources). Used by the wl_buffer destroy
    /// dispatch — there's a tick between the destroy and the next
    /// render so by the time we actually free, the GPU is past it.
    pub fn evict(&mut self, key: u64) {
        self.failed_imports.remove(&key);
        let Some(entry) = self.entries.remove(&key) else {
            return;
        };
        unsafe {
            let device = &self.stack.device;
            let _ = device.free_descriptor_sets(entry.descriptor_pool, &[entry.descriptor]);
            device.destroy_image_view(entry.view, None);
            device.destroy_image(entry.image, None);
            device.free_memory(entry.memory, None);
            if let Some(s) = entry.staging {
                device.destroy_buffer(s.buffer, None);
                device.unmap_memory(s.memory);
                device.free_memory(s.memory, None);
            }
        }
    }

    /// Walk every scene element and ensure there's an up-to-date entry
    /// for it. SHM uploads go into the entry's staging buffer (memcpy
    /// only — the `vkCmdCopyBufferToImage` is recorded later by
    /// `record_uploads` into the per-frame command buffer). Dmabuf
    /// imports happen synchronously here on first sight; subsequent
    /// frames are no-ops.
    ///
    /// `timing` accumulates per-frame counts: `n_dmabuf_imports_new`
    /// bumps on fresh imports, the cached-texture gauges are populated
    /// at the end so callers see end-of-frame totals.
    pub fn prepare_uploads(
        &mut self,
        scene: &sol_core::Scene<'_>,
        timing: &mut RenderTiming,
    ) -> Result<()> {
        self.pending.clear();
        for elem in &scene.elements {
            let res = match &elem.content {
                SceneContent::Shm { .. } => self.prepare_shm(elem, timing),
                SceneContent::Dmabuf { .. } => {
                    if self.failed_imports.contains(&elem.buffer_key) {
                        continue;
                    }
                    self.prepare_dmabuf(elem, timing)
                }
                SceneContent::BlurredBackdrop { .. } => Ok(()),
            };
            if let Err(e) = res {
                if matches!(elem.content, SceneContent::Dmabuf { .. }) {
                    self.failed_imports.insert(elem.buffer_key);
                    tracing::warn!(
                        error = %e,
                        key = elem.buffer_key,
                        "dmabuf import failed; buffer marked dead, not retrying"
                    );
                } else {
                    tracing::warn!(error = %e, key = elem.buffer_key, "scene element skipped");
                }
            }
        }
        // Snapshot end-of-frame cache occupancy (cheap: HashMap len +
        // a single pass to count is_dmabuf entries). Lets the metrics
        // export show "how many client buffers do we have GPU memory
        // tied up for".
        timing.n_textures_cached_total = self.entries.len() as u32;
        timing.n_textures_cached_dmabuf = self
            .entries
            .values()
            .filter(|e| e.is_dmabuf)
            .count() as u32;
        Ok(())
    }

    fn prepare_shm(
        &mut self,
        elem: &SceneElement<'_>,
        timing: &mut RenderTiming,
    ) -> Result<()> {
        let SceneContent::Shm {
            pixels,
            stride,
            format: _,
            upload_seq,
            trust_seq,
        } = &elem.content
        else {
            unreachable!();
        };
        let upload_seq = *upload_seq;
        let stride = *stride;

        // Cursor-only upload-skip. The compositor cursor sentinel is
        // allocated once at startup and never mutated, so reusing the
        // cached GPU texture is provably safe. Every other SHM
        // surface re-uploads on every render — the skip
        // optimisations (universal seq-skip, per-role gate, and
        // Hyprland-style commit-time snapshot) all produced visible
        // glitches on this NVIDIA setup; see the project memory note
        // for the full investigation. Cost is roughly ~32 MB / frame
        // of wallpaper-class re-upload at 4K, which puts the
        // compositor at 120-180 fps under shadertoy + Chrome load
        // instead of 240. Future work direction (per-surface damage
        // tracking + sub-rect uploads, dmabuf explicit-sync, …)
        // lives in the same memory note.
        // `trust_seq` extends the skip to background-layer surfaces
        // (wallpaper) — the compositor only sets it where the seq is
        // a complete content signal, so the NVIDIA glitch class the
        // universal skip hit can't apply.
        if elem.buffer_key == CURSOR_SCENE_KEY || *trust_seq {
            if let Some(e) = self.entries.get(&elem.buffer_key) {
                if !e.is_dmabuf
                    && e.width == elem.width
                    && e.height == elem.height
                    && e.uploaded_seq == upload_seq
                {
                    timing.n_shm_uploads_skipped += 1;
                    timing.n_shm_upload_skipped_bytes +=
                        (elem.width as u64) * (elem.height as u64) * 4;
                    return Ok(());
                }
            }
        }

        let needs_new = self
            .entries
            .get(&elem.buffer_key)
            .map(|e| e.is_dmabuf || e.width != elem.width || e.height != elem.height)
            .unwrap_or(true);
        if needs_new {
            self.evict(elem.buffer_key);
            let entry =
                create_shm_image(&self.stack, elem.width as u32, elem.height as u32)?;
            let (descriptor, descriptor_pool) = self.allocate_descriptor(entry.view)?;
            self.entries.insert(
                elem.buffer_key,
                TextureEntry {
                    image: entry.image,
                    view: entry.view,
                    memory: entry.memory,
                    descriptor,
                    descriptor_pool,
                    width: elem.width,
                    height: elem.height,
                    uploaded_seq: 0,
                    is_dmabuf: false,
                    staging: Some(entry.staging),
                },
            );
        }
        let entry = self.entries.get_mut(&elem.buffer_key).unwrap();

        // Memcpy host pixels into staging. Wayland buffers are
        // top-row-first so we copy row-by-row to the matching position
        // in the staging buffer.
        let staging = entry.staging.as_ref().expect("SHM entry must have staging");
        let tight_row = (elem.width as usize) * 4;
        let dst_row = tight_row;
        let h = elem.height as usize;
        let total = dst_row * h;
        let want = (stride as usize).saturating_mul(h);
        if pixels.len() < want {
            bail!(
                "scene buffer {} bytes < stride*height = {}",
                pixels.len(),
                want
            );
        }
        // smithay's wl_shm dispatch already enforces stride >= width*4,
        // but the row copy below is unsafe — keep it locally sound
        // rather than trusting a different crate's protocol validation.
        if (stride as usize) < tight_row {
            bail!("shm stride {stride} < tight row {tight_row}");
        }
        if (staging.size as usize) < total {
            bail!(
                "staging buffer {} bytes < dst {}",
                staging.size,
                total
            );
        }
        unsafe {
            if stride as usize == tight_row {
                // Tightly packed (the common case) — one memcpy for
                // the whole buffer instead of one per row.
                std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging.mapped, total);
            } else {
                for row in 0..h {
                    let src = pixels.as_ptr().add(row * stride as usize);
                    let dst = staging.mapped.add(row * dst_row);
                    std::ptr::copy_nonoverlapping(src, dst, tight_row);
                }
            }
        }

        let from_layout = if entry.uploaded_seq == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        entry.uploaded_seq = upload_seq;
        self.pending.push(PendingUpload {
            key: elem.buffer_key,
            width: elem.width as u32,
            height: elem.height as u32,
            from_layout,
        });
        Ok(())
    }

    fn prepare_dmabuf(
        &mut self,
        elem: &SceneElement<'_>,
        timing: &mut RenderTiming,
    ) -> Result<()> {
        let SceneContent::Dmabuf {
            num_planes,
            fds,
            offsets,
            strides,
            fourcc: _,
            modifier,
            commit_seq,
        } = &elem.content
        else {
            unreachable!();
        };
        // New content since we last sampled this buffer (or first
        // sight): export the producer's pending-write fence so the
        // presenter's submit waits for the client's GPU to finish —
        // see `pending_wait_fences`. `uploaded_seq` doubles as the
        // last commit_seq we synced against for dmabuf entries.
        let content_advanced = match self.entries.get(&elem.buffer_key) {
            Some(e) if e.is_dmabuf => e.uploaded_seq != *commit_seq,
            _ => true,
        };
        if content_advanced {
            if let Some(fence) = export_dmabuf_read_fence(fds[0]) {
                self.pending_wait_fences.push(fence);
                timing.n_dmabuf_fence_waits += 1;
            }
        }
        // Already imported and matches dimensions? Skip.
        if let Some(e) = self.entries.get_mut(&elem.buffer_key) {
            if e.is_dmabuf && e.width == elem.width && e.height == elem.height {
                e.uploaded_seq = *commit_seq;
                return Ok(());
            }
        }
        self.evict(elem.buffer_key);

        // How many memory-plane layouts the explicit-modifier create
        // needs. The modifier is raw client input: anything the driver
        // didn't advertise for our texture format is rejected here —
        // feeding an unsupported modifier into
        // `VkImageDrmFormatModifierExplicitCreateInfoEXT` is undefined
        // behaviour, not just a failed create.
        //
        // DRM_FORMAT_MOD_INVALID (the implicit-modifier path — we
        // advertise it in the dmabuf feedback) is never in the driver
        // list; keep the long-standing single-plane passthrough for it
        // since real clients on this path have worked for the life of
        // the project. If a driver rejects it, the create fails cleanly
        // and the buffer lands in `failed_imports`.
        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
        let layout_planes = if *modifier == DRM_FORMAT_MOD_INVALID {
            1
        } else {
            match self.stack.texture_modifier_planes.get(modifier) {
                Some(&n) => n as usize,
                None => bail!(
                    "client dmabuf modifier 0x{modifier:016x} not supported by the driver"
                ),
            }
        };
        if layout_planes > *num_planes {
            bail!(
                "modifier 0x{modifier:016x} needs {layout_planes} memory planes, buffer has {num_planes}"
            );
        }
        // One VkDeviceMemory is bound for the whole (non-disjoint)
        // image, so every memory plane must live in the same fd.
        if fds[..layout_planes].iter().any(|&f| f != fds[0]) {
            bail!("disjoint multi-fd dmabuf import not supported");
        }

        // Wayland hands us a borrowed RawFd; dup it because Vulkan
        // takes ownership on import success. CLOEXEC so clients
        // spawned while the import is alive don't inherit the fd.
        let owned = {
            let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fds[0]) };
            rustix::io::fcntl_dupfd_cloexec(borrowed, 0).context("dup dmabuf fd")?
        };

        let image_data = import_dmabuf_external(
            &self.stack,
            owned,
            elem.width as u32,
            elem.height as u32,
            *modifier,
            &offsets[..layout_planes],
            &strides[..layout_planes],
        )?;
        let (descriptor, descriptor_pool) = self.allocate_descriptor(image_data.view)?;
        self.entries.insert(
            elem.buffer_key,
            TextureEntry {
                image: image_data.image,
                view: image_data.view,
                memory: image_data.memory,
                descriptor,
                descriptor_pool,
                width: elem.width,
                height: elem.height,
                uploaded_seq: *commit_seq,
                is_dmabuf: true,
                staging: None,
            },
        );
        // Fresh import — the texture cache previously had nothing for
        // this `buffer_key`. Steady-state should be 0 (clients re-use
        // their wl_buffers); a non-zero number every frame would
        // indicate cache thrash.
        timing.n_dmabuf_imports_new += 1;
        Ok(())
    }

    /// Drain the pending-upload list into the supplied command buffer.
    /// Each upload is bracketed by image-layout barriers (UNDEFINED ⇒
    /// TRANSFER_DST, then TRANSFER_DST ⇒ SHADER_READ_ONLY_OPTIMAL).
    /// `timing` accumulates the upload count + total bytes copied.
    pub fn record_uploads(&mut self, cb: vk::CommandBuffer, timing: &mut RenderTiming) {
        if self.pending.is_empty() {
            return;
        }
        let device = &self.stack.device;
        for up in self.pending.drain(..) {
            let Some(entry) = self.entries.get(&up.key) else {
                continue;
            };
            let staging = match &entry.staging {
                Some(s) => s,
                None => continue,
            };

            // Pre-copy barrier: whatever the image was → TRANSFER_DST.
            let pre = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(up.from_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(entry.image)
                .subresource_range(color_range());
            let dep_pre =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&pre));
            unsafe { device.cmd_pipeline_barrier2(cb, &dep_pre) };

            let bytes = (up.width as u64) * (up.height as u64) * 4;
            timing.n_shm_uploads += 1;
            timing.n_shm_upload_bytes += bytes;
            if bytes > timing.n_shm_upload_max_bytes {
                timing.n_shm_upload_max_bytes = bytes;
            }

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(up.width)
                .buffer_image_height(up.height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: up.width,
                    height: up.height,
                    depth: 1,
                });
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cb,
                    staging.buffer,
                    entry.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }

            // Post-copy barrier: TRANSFER_DST → SHADER_READ_ONLY_OPTIMAL.
            let post = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(entry.image)
                .subresource_range(color_range());
            let dep_post =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&post));
            unsafe { device.cmd_pipeline_barrier2(cb, &dep_post) };
        }
    }

    pub fn get(&self, key: u64) -> Option<&TextureEntry> {
        self.entries.get(&key)
    }

    /// Drain the producer fences exported during `prepare_uploads`.
    /// The presenter imports each into a wait semaphore for this
    /// frame's submit.
    pub fn take_wait_fences(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.pending_wait_fences)
    }
}

/// Export a sync-file fd carrying every pending *write* fence on the
/// dmabuf (`DMA_BUF_SYNC_READ`: "I want to read; give me the writers").
/// Returns `None` (with a once-per-process warning) on kernels or
/// drivers that don't support `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` —
/// behaviour then degrades to the old implicit-sync hope, not an error.
fn export_dmabuf_read_fence(fd: std::os::fd::RawFd) -> Option<OwnedFd> {
    use std::os::fd::FromRawFd;
    #[repr(C)]
    struct DmaBufExportSyncFile {
        flags: u32,
        fd: i32,
    }
    const DMA_BUF_SYNC_READ: u32 = 1 << 0;
    // _IOWR('b', 2, struct dma_buf_export_sync_file) — dma-buf core
    // ioctl, kernel >= 5.18.
    const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: u64 = 0xC008_6202;
    unsafe extern "C" {
        fn ioctl(fd: i32, request: u64, arg: *mut core::ffi::c_void) -> i32;
    }
    let mut arg = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };
    let r = unsafe {
        ioctl(
            fd,
            DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
            &mut arg as *mut _ as *mut core::ffi::c_void,
        )
    };
    if r != 0 || arg.fd < 0 {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "DMA_BUF_IOCTL_EXPORT_SYNC_FILE unavailable; sampling \
                 client dmabufs relies on driver implicit sync (may tear \
                 on NVIDIA)"
            );
        }
        return None;
    }
    Some(unsafe { OwnedFd::from_raw_fd(arg.fd) })
}

impl Drop for TextureCache {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stack.device.device_wait_idle();
            let keys: Vec<u64> = self.entries.keys().copied().collect();
            for k in keys {
                self.evict(k);
            }
            for pool in self.descriptor_pools.drain(..) {
                self.stack.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

struct ShmImageData {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    staging: StagingBuf,
}

fn create_shm_image(stack: &SharedStack, width: u32, height: u32) -> Result<ShmImageData> {
    let device = &stack.device;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(TEXTURE_FORMAT)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None)? };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = stack.find_memtype(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(TEXTURE_FORMAT)
        .subresource_range(color_range());
    let view = unsafe { device.create_image_view(&view_info, None)? };

    // Staging buffer for the per-frame copy. Sized to a tight
    // width*height*4 layout (matches `buffer_row_length` in the
    // CopyBufferToImage region, so source rows are not strided).
    let staging_size = (width as u64) * (height as u64) * 4;
    let buf_info = vk::BufferCreateInfo::default()
        .size(staging_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buf_info, None)? };
    let breq = unsafe { device.get_buffer_memory_requirements(buffer) };
    let bmt = stack.find_memtype(
        breq.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let balloc = vk::MemoryAllocateInfo::default()
        .allocation_size(breq.size)
        .memory_type_index(bmt);
    let bmem = unsafe { device.allocate_memory(&balloc, None)? };
    unsafe { device.bind_buffer_memory(buffer, bmem, 0)? };
    let mapped = unsafe {
        device
            .map_memory(bmem, 0, breq.size, vk::MemoryMapFlags::empty())
            .context("map staging memory")? as *mut u8
    };

    Ok(ShmImageData {
        image,
        view,
        memory,
        staging: StagingBuf {
            buffer,
            memory: bmem,
            size: breq.size,
            mapped,
        },
    })
}

struct DmabufImageData {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

fn import_dmabuf_external(
    stack: &SharedStack,
    fd: OwnedFd,
    width: u32,
    height: u32,
    modifier: u64,
    plane_offsets: &[u32],
    plane_strides: &[u32],
) -> Result<DmabufImageData> {
    let device = &stack.device;
    // One entry per memory plane of the modifier (the caller sized
    // the slices from the driver's modifier query). size must be 0
    // and arrayPitch/depthPitch are ignored per the VU.
    let plane_layouts: Vec<vk::SubresourceLayout> = plane_offsets
        .iter()
        .zip(plane_strides)
        .map(|(&offset, &stride)| {
            vk::SubresourceLayout::default()
                .offset(offset as u64)
                .size(0)
                .row_pitch(stride as u64)
        })
        .collect();
    let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);
    let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(TEXTURE_FORMAT)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut modifier_info)
        .push_next(&mut external_info);
    let image = unsafe { device.create_image(&image_info, None)? };

    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    use std::os::fd::{AsFd, AsRawFd};
    let probe_fd = fd.as_fd().try_clone_to_owned()?;
    unsafe {
        stack.ext_mem_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            probe_fd.as_raw_fd(),
            &mut fd_props,
        )?;
    }
    drop(probe_fd);
    let allowed = mem_reqs.memory_type_bits & fd_props.memory_type_bits;
    if allowed == 0 {
        unsafe { device.destroy_image(image, None) };
        bail!(
            "no memtype for dmabuf import (img=0x{:x} fd=0x{:x})",
            mem_reqs.memory_type_bits,
            fd_props.memory_type_bits
        );
    }
    let mem_type = stack
        .find_memtype(allowed, vk::MemoryPropertyFlags::empty())
        .or_else(|_| stack.find_memtype(allowed, vk::MemoryPropertyFlags::DEVICE_LOCAL))?;

    let raw_fd = fd.into_raw_fd();
    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_fd);
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type)
        .push_next(&mut dedicated_info)
        .push_next(&mut import_info);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) };
    let memory = match memory {
        Ok(m) => m,
        Err(e) => {
            unsafe {
                device.destroy_image(image, None);
                use std::os::fd::FromRawFd;
                let _ = OwnedFd::from_raw_fd(raw_fd); // close on failure
            }
            return Err(anyhow!("allocate_memory dmabuf: {e}"));
        }
    };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(TEXTURE_FORMAT)
        .subresource_range(color_range());
    let view = unsafe { device.create_image_view(&view_info, None)? };
    Ok(DmabufImageData {
        image,
        view,
        memory,
    })
}

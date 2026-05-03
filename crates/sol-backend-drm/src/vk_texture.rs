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
    pub width: i32,
    pub height: i32,
    /// Last `SceneContent::Shm::upload_seq` we copied for this entry.
    /// Compared against the next frame's seq to skip the upload when
    /// the cursor sentinel matches.
    pub uploaded_seq: u64,
    /// True iff this entry was imported from a dmabuf (no per-frame
    /// upload work; image is read-only from our side).
    pub is_dmabuf: bool,
    /// For dmabuf entries only: true once we've seen the image at
    /// least once and its layout is `SHADER_READ_ONLY_OPTIMAL`. The
    /// first-frame acquire is a plain UNDEFINED → SHADER_READ_ONLY
    /// transition with no QF transfer; subsequent frames issue a
    /// FOREIGN_EXT → graphics-queue acquire so the dma-buf implicit
    /// fence (signalled by the client's GPU process when its writes
    /// land) is honoured before we sample. Without this, Chrome
    /// rapidly rotating its video buffers can produce tearing /
    /// stutter as we read partial writes.
    pub dmabuf_seen: bool,
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
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    pending: Vec<PendingUpload>,
    /// When true, skip the SHM upload for any buffer whose `upload_seq`
    /// matches the last copied seq. Toggleable via `SOL_TRUST_UPLOAD_SEQ`
    /// env var (default `1` / on).
    trust_upload_seq: bool,
}

impl TextureCache {
    /// Pool capacity — generous enough that even a busy session with
    /// dozens of clients + the blur FBOs (which also pull from this
    /// pool) stays under the limit. Going past this count panics the
    /// allocate; we'll add growth if it ever comes up.
    const POOL_CAPACITY: u32 = 256;

    pub fn new(
        stack: SharedStack,
        set_layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
    ) -> Result<Self> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(Self::POOL_CAPACITY)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(Self::POOL_CAPACITY)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            stack
                .device
                .create_descriptor_pool(&pool_info, None)
                .context("create_descriptor_pool (texture cache)")?
        };
        // Default ON. The skip was previously blamed for Chrome
        // glitches but the actual root cause was missing FOREIGN_EXT
        // queue-family transfers on client dmabufs (Chrome's video
        // frames could tear at the consumer side without an explicit
        // dma-buf fence acquire). With the per-frame dmabuf acquire
        // barrier in place — see `TextureCache::record_dmabuf_acquires`
        // — the upload-seq skip is safe again, and worth ≈ 32 MB / frame
        // of wallpaper-upload savings at 4K. Set SOL_TRUST_UPLOAD_SEQ=0
        // to disable as a debugging knob if a regression suspect
        // points back here.
        let trust_upload_seq = std::env::var("SOL_TRUST_UPLOAD_SEQ")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        tracing::info!(
            trust_upload_seq,
            "SHM upload-skip configured (set SOL_TRUST_UPLOAD_SEQ=0 to disable)"
        );
        Ok(Self {
            stack,
            entries: HashMap::new(),
            descriptor_pool,
            set_layout,
            sampler,
            pending: Vec::new(),
            trust_upload_seq,
        })
    }

    /// Allocate a fresh descriptor set + immediately point it at the
    /// given image view. Used by the texture cache for client buffers
    /// and by the blur module for the capture / ping / pong FBOs.
    pub fn allocate_descriptor(
        &self,
        view: vk::ImageView,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [self.set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe {
            self.stack
                .device
                .allocate_descriptor_sets(&alloc_info)
                .context("allocate_descriptor_sets")?
        };
        let set = sets[0];
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
        Ok(set)
    }

    /// Drop the cached entry for `key`. Caller must have already
    /// queued a `vkDeviceWaitIdle` (or otherwise be sure the GPU is
    /// no longer using the resources). Used by the wl_buffer destroy
    /// dispatch — there's a tick between the destroy and the next
    /// render so by the time we actually free, the GPU is past it.
    pub fn evict(&mut self, key: u64) {
        let Some(entry) = self.entries.remove(&key) else {
            return;
        };
        unsafe {
            let device = &self.stack.device;
            let _ = device.free_descriptor_sets(self.descriptor_pool, &[entry.descriptor]);
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
                SceneContent::Shm { .. } => self.prepare_shm(elem),
                SceneContent::Dmabuf { .. } => self.prepare_dmabuf(elem, timing),
                SceneContent::BlurredBackdrop { .. } => Ok(()),
            };
            if let Err(e) = res {
                tracing::warn!(error = %e, key = elem.buffer_key, "scene element skipped");
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

    fn prepare_shm(&mut self, elem: &SceneElement<'_>) -> Result<()> {
        let SceneContent::Shm {
            pixels,
            stride,
            format: _,
            upload_seq,
        } = &elem.content
        else {
            unreachable!();
        };
        let upload_seq = *upload_seq;
        let stride = *stride;

        // Upload-skip: if the source buffer hasn't been re-committed
        // since the last upload, the GPU texture is still current and
        // the entire memcpy + vkCmdCopyBufferToImage pair can be
        // skipped. `upload_seq` is bumped from the wl_surface.commit
        // dispatch on every commit (see compositor.rs), so equal seqs
        // mean the client hasn't asked us to re-paint it.
        //
        // Set `SOL_TRUST_UPLOAD_SEQ=0` to fall back to the old
        // cursor-only behavior — kept as an escape hatch in case some
        // pathological client (the prior incident pointed at Chrome on
        // YouTube) violates the "no commit since last upload" ⇒
        // "pixels unchanged" assumption. Default ON because at 4K
        // every wallpaper / layer-shell buffer that doesn't re-commit
        // costs ≈32 MB of needless upload per frame, which dominates
        // the renderer at 240 Hz.
        //
        // The `upload_seq != 0` guard avoids skipping on the very first
        // frame for a freshly-attached buffer: an entry can exist with
        // `uploaded_seq = 0` after creation but before any copy was
        // recorded.
        if let Some(e) = self.entries.get(&elem.buffer_key) {
            if !e.is_dmabuf
                && e.width == elem.width
                && e.height == elem.height
                && upload_seq != 0
                && e.uploaded_seq == upload_seq
                && self.trust_upload_seq
            {
                return Ok(());
            }
            // The strict cursor-only path stays as a safety net even
            // when `trust_upload_seq` is off: the cursor sprite is
            // provably immutable post-init, so its skip is always
            // correct.
            if !self.trust_upload_seq && elem.buffer_key == CURSOR_SCENE_KEY {
                if !e.is_dmabuf
                    && e.width == elem.width
                    && e.height == elem.height
                    && e.uploaded_seq == upload_seq
                {
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
            let descriptor = self.allocate_descriptor(entry.view)?;
            self.entries.insert(
                elem.buffer_key,
                TextureEntry {
                    image: entry.image,
                    view: entry.view,
                    memory: entry.memory,
                    descriptor,
                    width: elem.width,
                    height: elem.height,
                    uploaded_seq: 0,
                    is_dmabuf: false,
                    dmabuf_seen: false,
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
        if (staging.size as usize) < total {
            bail!(
                "staging buffer {} bytes < dst {}",
                staging.size,
                total
            );
        }
        unsafe {
            for row in 0..h {
                let src = pixels.as_ptr().add(row * stride as usize);
                let dst = staging.mapped.add(row * dst_row);
                std::ptr::copy_nonoverlapping(src, dst, tight_row);
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
            fd,
            fourcc: _,
            modifier,
            offset,
            stride,
        } = &elem.content
        else {
            unreachable!();
        };
        // Already imported and matches dimensions? Skip.
        if let Some(e) = self.entries.get(&elem.buffer_key) {
            if e.is_dmabuf && e.width == elem.width && e.height == elem.height {
                return Ok(());
            }
        }
        self.evict(elem.buffer_key);

        // Wayland hands us a borrowed RawFd; dup it because Vulkan
        // takes ownership on import success.
        use std::os::fd::FromRawFd;
        let dup = unsafe {
            libc_dup(*fd).context("dup dmabuf fd")?
        };
        let owned = unsafe { OwnedFd::from_raw_fd(dup) };

        let image_data = import_dmabuf_external(
            &self.stack,
            owned,
            elem.width as u32,
            elem.height as u32,
            *modifier,
            *offset as u64,
            *stride as u64,
        )?;
        let descriptor = self.allocate_descriptor(image_data.view)?;
        self.entries.insert(
            elem.buffer_key,
            TextureEntry {
                image: image_data.image,
                view: image_data.view,
                memory: image_data.memory,
                descriptor,
                width: elem.width,
                height: elem.height,
                uploaded_seq: 0,
                is_dmabuf: true,
                dmabuf_seen: false,
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

    /// Per-frame foreign-queue acquire on every dmabuf entry actually
    /// sampled this frame. Pairs with the producer's release (which is
    /// implicit, signalled via the dma-buf fence when the client's
    /// commit lands). Without this barrier, NVIDIA in particular may
    /// sample a partially-written buffer — observable as Chrome video
    /// stutter at the per-buffer level, and as UI flicker for the rest
    /// of the browser chrome.
    ///
    /// First-time acquire on a freshly-imported dmabuf is a plain
    /// UNDEFINED → SHADER_READ_ONLY_OPTIMAL transition with no QF
    /// transfer (the image was never owned by the foreign side from
    /// Vulkan's view). Subsequent acquires emit the real
    /// FOREIGN_EXT → graphics-queue barrier.
    pub fn record_dmabuf_acquires(
        &mut self,
        scene: &sol_core::Scene<'_>,
        cb: vk::CommandBuffer,
        queue_family: u32,
    ) {
        let device = &self.stack.device;
        let mut barriers: Vec<vk::ImageMemoryBarrier2<'_>> = Vec::new();
        // Track which keys we've already acquired this frame so the
        // same buffer referenced twice (e.g. via subsurface) doesn't
        // get a duplicate barrier.
        let mut acquired: std::collections::HashSet<u64> =
            std::collections::HashSet::new();
        for elem in &scene.elements {
            if !matches!(elem.content, SceneContent::Dmabuf { .. }) {
                continue;
            }
            if !acquired.insert(elem.buffer_key) {
                continue;
            }
            let Some(entry) = self.entries.get_mut(&elem.buffer_key) else {
                continue;
            };
            if !entry.is_dmabuf {
                continue;
            }
            let (src_qf, dst_qf, old_layout) = if entry.dmabuf_seen {
                (
                    vk::QUEUE_FAMILY_FOREIGN_EXT,
                    queue_family,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            } else {
                (
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::ImageLayout::UNDEFINED,
                )
            };
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_queue_family_index(src_qf)
                    .dst_queue_family_index(dst_qf)
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(entry.image)
                    .subresource_range(color_range()),
            );
            entry.dmabuf_seen = true;
        }
        if barriers.is_empty() {
            return;
        }
        let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(cb, &dep) };
    }
}

impl Drop for TextureCache {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stack.device.device_wait_idle();
            let keys: Vec<u64> = self.entries.keys().copied().collect();
            for k in keys {
                self.evict(k);
            }
            self.stack
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
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
    offset: u64,
    row_pitch: u64,
) -> Result<DmabufImageData> {
    let device = &stack.device;
    let plane_layouts = [vk::SubresourceLayout::default()
        .offset(offset)
        .size(0)
        .row_pitch(row_pitch)];
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

unsafe fn libc_dup(fd: std::os::fd::RawFd) -> Result<i32> {
    let r = unsafe { libc_dup_impl(fd) };
    if r < 0 {
        Err(anyhow!("dup({fd}): {}", std::io::Error::last_os_error()))
    } else {
        Ok(r)
    }
}

// Minimal shim so we don't drag the full libc crate dep in here for one
// call. `dup` is `int dup(int)` in <unistd.h>.
unsafe extern "C" {
    fn dup(fd: i32) -> i32;
}
unsafe fn libc_dup_impl(fd: i32) -> i32 {
    unsafe { dup(fd) }
}

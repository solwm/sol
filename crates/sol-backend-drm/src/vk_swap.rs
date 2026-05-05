//! GBM-backed scan-out swap.
//!
//! We run our own swap chain instead of using `VkSwapchainKHR` because the
//! latter assumes a window-system surface (Wayland / X / display); we're a
//! compositor — we own the display directly via DRM/KMS. The buffers we
//! eventually `drmModePageFlip` to are GBM `BufferObject`s, allocated
//! with `SCANOUT | RENDERING | LINEAR`. Each one gets imported into Vulkan
//! as a `VkImage` via `VK_EXT_image_drm_format_modifier` +
//! `VK_EXT_external_memory_dma_buf`, so the GPU can use the same memory
//! the kernel scans out.
//!
//! The slot count (3) gives us one BO on screen + one queued for the next
//! flip + one we're rendering into, so the GPU can build frame N+1 while
//! the kernel still has frame N–1 latched.

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use gbm::{AsRaw, BufferObjectFlags, Format as GbmFormat};
use smithay::backend::drm::DrmDeviceFd;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

use crate::vk_stack::SharedStack;

/// Slot count for the scan-out ring. Three is the minimum that lets the
/// kernel hold one frame on screen, one queued, and the GPU render a
/// third concurrently.
pub const SWAP_SLOTS: usize = 3;

/// Format we render into and scan out. XRGB8888 stored little-endian
/// (BGRA in memory order on x86), which maps cleanly onto
/// `VK_FORMAT_B8G8R8A8_UNORM`. The DRM `add_framebuffer` call takes
/// 24 bpp / 32 stride for this layout.
pub const SCANOUT_VK_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// One scan-out slot. Holds the GBM BO that owns the kernel-visible
/// memory plus the Vulkan handles wrapping it. The DRM framebuffer
/// handle is built lazily on first use by the presenter's
/// `add_or_get_fb` (`AddFB2WithModifiers` under the hood) and cached
/// there — the slot doesn't track it directly so we don't have two
/// caches that can diverge.
pub struct Slot {
    pub bo: gbm::BufferObject<()>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    /// Tracked image layout. Starts in `UNDEFINED` (Vulkan's "contents
    /// are garbage" state) and the renderer transitions through
    /// `COLOR_ATTACHMENT_OPTIMAL` → `GENERAL` each frame so the
    /// kernel can scan out (DRM doesn't track Vulkan layouts; GENERAL
    /// is the spec-blessed "any operation" layout).
    pub layout: vk::ImageLayout,
    /// True once the slot has gone through a release-to-FOREIGN
    /// barrier (i.e. the kernel has scanned it out at least once).
    /// On its next use, the renderer must issue a matching acquire-
    /// from-FOREIGN barrier so the dma-buf fence is honoured. False
    /// initially — first-time use just transitions from UNDEFINED.
    pub needs_foreign_acquire: bool,
}

#[allow(dead_code)] // gbm/width/height kept for future modeset / resize hooks
pub struct GbmSwap {
    stack: SharedStack,
    pub gbm: gbm::Device<DrmDeviceFd>,
    pub width: u32,
    pub height: u32,
    pub slots: Vec<Slot>,
    /// Idx of the slot currently being scanned out (kernel is sampling
    /// from it). `None` until the first flip completes.
    pub scanned: Option<usize>,
    /// Idx of the slot whose flip is queued in the kernel — its memory
    /// must not be touched until the page-flip-complete event fires.
    pub pending: Option<usize>,
}

impl GbmSwap {
    pub fn new(
        stack: SharedStack,
        drm_fd: DrmDeviceFd,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let gbm = gbm::Device::new(drm_fd).context("gbm::Device::new")?;
        let mut slots = Vec::with_capacity(SWAP_SLOTS);
        for i in 0..SWAP_SLOTS {
            // Two-step allocation. First try without `LINEAR`: NVIDIA's GBM
            // rejects `SCANOUT | RENDERING | LINEAR` outright (especially
            // at 4K), and on Mesa drivers a tiled modifier scans out
            // faster anyway. The actual modifier picked is whatever the
            // driver's preferred one is, queried via `bo.modifier()` and
            // passed straight to Vulkan via `VK_EXT_image_drm_format_modifier`.
            // The LINEAR fallback is for any obscure setup where the
            // driver only knows how to scan out linear memory.
            let bo = match gbm.create_buffer_object::<()>(
                width,
                height,
                GbmFormat::Xrgb8888,
                BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING,
            ) {
                Ok(bo) => bo,
                Err(e) => {
                    tracing::warn!(
                        slot = i,
                        error = %e,
                        "gbm SCANOUT|RENDERING failed, retrying with LINEAR"
                    );
                    gbm.create_buffer_object::<()>(
                        width,
                        height,
                        GbmFormat::Xrgb8888,
                        BufferObjectFlags::SCANOUT
                            | BufferObjectFlags::RENDERING
                            | BufferObjectFlags::LINEAR,
                    )
                    .with_context(|| {
                        format!("gbm create_buffer_object slot {i} (both modifiers failed)")
                    })?
                }
            };
            let modifier_u64 = u64::from(bo.modifier());
            let stride = bo.stride();

            let fd = bo
                .fd()
                .map_err(|e| anyhow!("gbm bo fd (slot {i}): {e:?}"))?;
            let (image, view, memory) =
                import_dmabuf_image(&stack, fd, width, height, modifier_u64, stride as u64)
                    .with_context(|| format!("import slot {i}"))?;

            tracing::info!(
                slot = i,
                modifier = format!("0x{:x}", modifier_u64),
                stride,
                bo_ptr = bo.as_raw() as usize,
                "scan-out slot allocated"
            );

            slots.push(Slot {
                bo,
                image,
                view,
                memory,
                layout: vk::ImageLayout::UNDEFINED,
                needs_foreign_acquire: false,
            });
        }
        // The DrmDeviceFd we got is consumed by gbm::Device above —
        // no further DRM ops happen here (presenter adds framebuffers
        // lazily on first scan-out via add_or_get_fb).
        Ok(Self {
            stack,
            gbm,
            width,
            height,
            slots,
            scanned: None,
            pending: None,
        })
    }

    /// Grab a slot we're free to render into: not the one being scanned
    /// out and not the one with a flip queued. With three slots and at
    /// most one of each in flight, one is always free.
    pub fn acquire(&self) -> Option<usize> {
        for i in 0..self.slots.len() {
            if Some(i) == self.scanned || Some(i) == self.pending {
                continue;
            }
            return Some(i);
        }
        None
    }

    /// Mark slot `idx` as having an in-flight page flip. Caller did the
    /// `drmModePageFlip(EVENT)` and is now waiting for the kernel's
    /// completion event on the DRM fd.
    pub fn submit_flip(&mut self, idx: usize) {
        self.pending = Some(idx);
    }

    /// Page-flip-complete event arrived: the previously-pending slot is
    /// the new active scan-out, and whatever was on screen before is
    /// free for the next render.
    pub fn flip_complete(&mut self) {
        if let Some(p) = self.pending.take() {
            self.scanned = Some(p);
        }
    }
}

impl Drop for GbmSwap {
    fn drop(&mut self) {
        unsafe {
            // Wait for the GPU before destroying images — safer than
            // tracking submit fences across a teardown path.
            let _ = self.stack.device.device_wait_idle();
            for slot in self.slots.drain(..) {
                self.stack.device.destroy_image_view(slot.view, None);
                self.stack.device.destroy_image(slot.image, None);
                self.stack.device.free_memory(slot.memory, None);
                // The DRM framebuffer release happens via the fb_cache
                // teardown in DrmPresenter::Drop; the GBM BO drops here.
                drop(slot.bo);
            }
        }
    }
}

/// Import a dmabuf-backed scan-out BO as a `VkImage`. The image is
/// created with `DRM_FORMAT_MODIFIER_EXT` tiling and an explicit modifier
/// (`DRM_FORMAT_MOD_LINEAR` here, since we forced GBM to LINEAR), so the
/// driver knows the memory layout matches the BO.
fn import_dmabuf_image(
    stack: &SharedStack,
    fd: OwnedFd,
    width: u32,
    height: u32,
    modifier: u64,
    row_pitch: u64,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    // One plane (single-plane RGB), explicit layout: row pitch from GBM,
    // offset 0, size 0 (driver computes).
    let plane_layouts = [vk::SubresourceLayout::default()
        .offset(0)
        .size(0)
        .row_pitch(row_pitch)
        .array_pitch(0)
        .depth_pitch(0)];
    let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);
    let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(SCANOUT_VK_FORMAT)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut modifier_info)
        .push_next(&mut external_info);

    let image = unsafe {
        stack
            .device
            .create_image(&image_info, None)
            .context("create scan-out VkImage")?
    };

    let mem_reqs = unsafe { stack.device.get_image_memory_requirements(image) };

    // The dmabuf's memory type bits — restrict our `find_memtype` query
    // to types compatible with the imported FD, which on Mesa is usually
    // a single host-visible type plus device-local. The query takes the
    // FD by reference (no transfer of ownership), so we hand it a dup
    // and let the dup drop at end of scope rather than into_raw_fd-
    // ing it (which would leak).
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    let probe_fd = fd.as_fd().try_clone_to_owned()?;
    unsafe {
        stack.ext_mem_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            probe_fd.as_raw_fd(),
            &mut fd_props,
        )?;
    }
    drop(probe_fd);
    let allowed_bits = mem_reqs.memory_type_bits & fd_props.memory_type_bits;
    if allowed_bits == 0 {
        unsafe {
            stack.device.destroy_image(image, None);
        }
        bail!(
            "no memory type satisfies both image (0x{:x}) and dmabuf (0x{:x})",
            mem_reqs.memory_type_bits,
            fd_props.memory_type_bits,
        );
    }
    let mem_type = stack
        .find_memtype(allowed_bits, vk::MemoryPropertyFlags::empty())
        .or_else(|_| stack.find_memtype(allowed_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL))?;

    // Vulkan takes ownership of the FD on success, so this is a one-shot.
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

    let memory = unsafe { stack.device.allocate_memory(&alloc_info, None) };
    let memory = match memory {
        Ok(m) => m,
        Err(e) => {
            unsafe {
                stack.device.destroy_image(image, None);
            }
            // On failure, Vulkan didn't take the FD; close it.
            unsafe {
                let _ = std::os::fd::OwnedFd::from_raw_fd(raw_fd);
            }
            return Err(anyhow!("allocate_memory for dmabuf import: {e}"));
        }
    };

    unsafe {
        stack
            .device
            .bind_image_memory(image, memory, 0)
            .context("bind_image_memory for scan-out")?;
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(SCANOUT_VK_FORMAT)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe {
        stack
            .device
            .create_image_view(&view_info, None)
            .context("create scan-out image view")?
    };

    Ok((image, view, memory))
}


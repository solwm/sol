//! Frame-by-frame presenter built on Vulkan + DRM/KMS.
//!
//! `DrmPresenter` brings up a Vulkan device, allocates a triple of GBM-backed
//! scan-out images (imported as `VkImage`s via the DRM-format-modifier dmabuf
//! path), and exposes `render_scene` — the one entry point the Wayland side
//! calls each tick.
//!
//! The flow per frame:
//!
//!   render_scene:
//!     bail if a flip / render is still in flight
//!     check the blur cache (skip the bg pre-pass + blur passes if valid)
//!     pick a free scan-out slot
//!     reset the per-frame command buffer
//!     prepare SHM uploads (memcpy host→staging)
//!     record SHM upload copies + image barriers
//!     if needed: bg pre-pass into the blur capture FBO + ping-pong blur
//!     main draw into the slot's image — back-to-front element loop with
//!       border anchor injection, backdrop draws sampling the blurred FBO
//!     transition slot image to GENERAL (kernel can scan it out)
//!     submit with a sync-FD-exportable semaphore signal
//!     export the sync FD, return it up to the Wayland side
//!
//! When the FD signals (GPU done), `submit_flip_after_fence` runs the
//! `add_framebuffer` + `drmModePageFlip(EVENT)` pair. Kernel later wakes
//! us via the DRM fd; `flip_complete` settles state.
//!
//! Synchronous fallback (driver doesn't support `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD`):
//! `vkQueueWaitIdle` after submit, then add_framebuffer + page_flip inline,
//! and we go straight to `pending_flip`.

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use ash::vk;
use drm::control::{
    Device as ControlDevice, FbCmd2Flags, Mode, PageFlipFlags, framebuffer, property,
};
use sol_core::{PixelFormat, RenderTiming, Scene, SceneBorder, SceneContent};

use crate::vk_blur::{BlurChain, set_viewport_scissor, transition as transition_blur, bytes_of};
use crate::vk_pipe::{BackdropPC, Pipelines, QuadPC, SolidPC};
use crate::vk_stack::{SharedStack, VkStack};
use crate::vk_swap::GbmSwap;
use crate::vk_texture::TextureCache;
use crate::{Card, ModePreference, OutputSelection, pick_output};

/// Snapshot of the CRTC state that was active before we grabbed master,
/// pushed back on `Drop` so the TTY restores cleanly. Same shape as the
/// pre-rewrite version — only the rendering changed.
struct SavedCrtc {
    mode: Option<Mode>,
    fb: Option<framebuffer::Handle>,
    position: (u32, u32),
}

pub struct DrmPresenter {
    card: Card,
    sel: OutputSelection,
    width: u32,
    height: u32,
    saved_crtc: Option<SavedCrtc>,
    dpms_prop: Option<property::Handle>,

    stack: SharedStack,
    swap: GbmSwap,
    pipelines: Pipelines,
    textures: TextureCache,
    blur: Option<BlurChain>,

    /// Per-frame command buffer. Reset at the top of each render, recorded
    /// inline, submitted at the end. Not double-buffered — render_scene's
    /// `is_busy` guard guarantees the previous frame's submit has finished
    /// (sync FD signalled) before we touch this again.
    command_buffer: vk::CommandBuffer,
    /// Fence we always signal alongside the sync-FD semaphore so the
    /// fallback path (no sync-FD support) can `wait_for_fences` and the
    /// Drop path can `device_wait_idle` knowing the last submit drained.
    frame_fence: vk::Fence,
    /// Binary semaphore created with `VkExportSemaphoreCreateInfo` for
    /// `SYNC_FD`. `vkQueueSubmit` puts a pending signal payload here;
    /// `vkGetSemaphoreFdKHR` transfers the payload out as a pollable FD
    /// the wayland-side calloop registers. `None` when the driver
    /// declined to create an exportable semaphore — we run the
    /// synchronous fallback in that case.
    sync_semaphore: Option<vk::Semaphore>,

    /// Cache mapping `bo.handle().into()` → DRM framebuffer handle so
    /// `add_framebuffer` runs once per BO. Same shape as the pre-rewrite
    /// cache, just keyed off the BO's GEM handle.
    fb_cache: HashMap<u32, framebuffer::Handle>,

    /// Slot index acquired for the current render. `Some` between
    /// `render_scene`'s queue submit and `submit_flip_after_fence`, then
    /// settled into `swap.pending` and cleared.
    in_flight_slot: Option<usize>,
    pending_flip: bool,
    pending_render: bool,

    /// Blur-cache state. Same shape as the pre-rewrite version.
    blur_ready_this_frame: bool,
    last_bg_sig: u64,
    last_blur_params: (u32, u32),
}

impl DrmPresenter {
    pub fn from_card(card: Card, mode_pref: Option<ModePreference>) -> Result<Self> {
        let sel = pick_output(&card, mode_pref)?;
        let (w_i16, h_i16) = sel.mode.size();
        let width = w_i16 as u32;
        let height = h_i16 as u32;

        let saved_crtc = match card.get_crtc(sel.crtc) {
            Ok(info) => Some(SavedCrtc {
                mode: info.mode(),
                fb: info.framebuffer(),
                position: info.position(),
            }),
            Err(e) => {
                tracing::warn!(error = ?e, "get_crtc for save failed; TTY restore on exit won't work");
                None
            }
        };

        let stack = VkStack::new()?;
        let swap = GbmSwap::new(stack.clone(), &card, width, height)?;
        let pipelines = Pipelines::new(stack.clone())?;
        let textures = TextureCache::new(
            stack.clone(),
            pipelines.sampled_set_layout,
            pipelines.linear_sampler,
        )?;
        let blur = match BlurChain::new(stack.clone(), &textures, width, height) {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(error = %e, "blur chain init failed; inactive-window blur disabled");
                None
            }
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(stack.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            stack
                .device
                .allocate_command_buffers(&alloc_info)
                .context("allocate command buffer")?[0]
        };

        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let frame_fence = unsafe {
            stack
                .device
                .create_fence(&fence_info, None)
                .context("create frame fence")?
        };

        let sync_semaphore = create_export_semaphore(&stack);
        if sync_semaphore.is_none() {
            tracing::warn!(
                "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD unavailable; \
                 falling back to vkQueueWaitIdle before page_flip"
            );
        } else {
            tracing::info!(
                "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD available; \
                 page_flip deferred until GPU done"
            );
        }

        let mut presenter = Self {
            card,
            sel,
            width,
            height,
            saved_crtc,
            dpms_prop: None,
            stack,
            swap,
            pipelines,
            textures,
            blur,
            command_buffer,
            frame_fence,
            sync_semaphore,
            fb_cache: HashMap::new(),
            in_flight_slot: None,
            pending_flip: false,
            pending_render: false,
            blur_ready_this_frame: false,
            last_bg_sig: 0,
            last_blur_params: (0, 0),
        };
        presenter.initial_modeset()?;
        Ok(presenter)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn refresh_hz(&self) -> u32 {
        self.sel.mode.vrefresh()
    }

    /// libseat hands master back to us via the kernel-side ioctl when our
    /// session re-activates; this method just re-applies the modeset so
    /// the screen reflects our scene rather than whatever drove the
    /// display during the disabled window. Same semantics as the
    /// pre-rewrite version.
    pub fn reacquire_master(&mut self) -> Result<()> {
        if let Some(idx) = self.swap.scanned {
            let fb = self.add_or_get_fb(idx)?;
            self.card
                .set_crtc(
                    self.sel.crtc,
                    Some(fb),
                    (0, 0),
                    &[self.sel.connector],
                    Some(self.sel.mode),
                )
                .context("re-apply modeset on Enable")?;
        }
        tracing::info!("DRM master reacquired (session enabled)");
        Ok(())
    }

    pub fn drop_master(&self) {
        tracing::info!("session disabled — DRM master released by logind");
    }

    /// DPMS on/off — same as pre-rewrite. Resolves the `DPMS` connector
    /// property lazily and caches the handle.
    pub fn set_dpms(&mut self, blank: bool) -> Result<()> {
        if self.dpms_prop.is_none() {
            let props = self
                .card
                .get_properties(self.sel.connector)
                .context("get connector properties for DPMS")?;
            for (handle, _) in props.iter() {
                let info = match self.card.get_property(*handle) {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                if info.name().to_bytes() == b"DPMS" {
                    self.dpms_prop = Some(*handle);
                    break;
                }
            }
        }
        let Some(handle) = self.dpms_prop else {
            tracing::warn!("connector has no DPMS property; screen stays on");
            return Ok(());
        };
        let value = if blank { 3 } else { 0 };
        self.card
            .set_property(self.sel.connector, handle, value)
            .context("drm set DPMS")?;

        if !blank {
            if let Some(idx) = self.swap.scanned {
                let fb = self.add_or_get_fb(idx)?;
                self.card
                    .set_crtc(
                        self.sel.crtc,
                        Some(fb),
                        (0, 0),
                        &[self.sel.connector],
                        Some(self.sel.mode),
                    )
                    .context("re-apply modeset on DPMS wake")?;
            }
        }
        tracing::info!(dpms_off = blank, "set DPMS");
        Ok(())
    }

    pub fn evict_texture(&mut self, key: u64) {
        // The wayland side only calls this on wl_buffer destroy, which
        // happens after the buffer was last used in a frame and at
        // least one render_tick later — so the GPU can't still be
        // sampling. Safe to free immediately.
        self.textures.evict(key);
    }

    pub fn is_busy(&self) -> bool {
        self.pending_flip || self.pending_render
    }

    pub fn is_pending_flip(&self) -> bool {
        self.is_busy()
    }

    pub fn drm_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.card.as_fd()
    }

    /// Drain DRM events. For each page-flip-complete event, settle slot
    /// state via `flip_complete()`. Returns true if at least one
    /// page-flip event was consumed so the caller can fire frame
    /// callbacks.
    pub fn drain_events(&mut self) -> Result<bool> {
        let mut saw_flip = false;
        for ev in self.card.receive_events()? {
            if matches!(ev, drm::control::Event::PageFlip(_)) {
                saw_flip = true;
            }
        }
        if saw_flip {
            self.flip_complete();
        }
        Ok(saw_flip)
    }

    pub fn flip_complete(&mut self) {
        self.swap.flip_complete();
        self.pending_flip = false;
    }

    /// Build out the DRM framebuffer handle for a swap slot, caching the
    /// result so re-flipping to the same BO doesn't re-add. Keyed off
    /// the BO's GEM handle (a u32 id stable for the BO's lifetime).
    ///
    /// Uses `add_planar_framebuffer` with `FbCmd2Flags::MODIFIERS` so
    /// the kernel can ingest a tiled BO — the legacy `add_framebuffer`
    /// (DRM_IOCTL_MODE_ADDFB) only knows the implicit-modifier path
    /// and rejects anything non-LINEAR with EINVAL, which is what
    /// NVIDIA's GBM hands us at 4K.
    fn add_or_get_fb(&mut self, idx: usize) -> Result<framebuffer::Handle> {
        let slot = &self.swap.slots[idx];
        // gbm BufferObjectHandle is essentially a wrapped u32; convert
        // via the gbm API.
        let key: u32 = unsafe { slot.bo.handle().u32_ };
        if let Some(fb) = self.fb_cache.get(&key) {
            return Ok(*fb);
        }
        let fb = self
            .card
            .add_planar_framebuffer(&slot.bo, FbCmd2Flags::MODIFIERS)
            .context("drm add_planar_framebuffer")?;
        self.fb_cache.insert(key, fb);
        Ok(fb)
    }

    /// Populate slot 0 with a dark frame and modeset to it. Without this,
    /// the first thing on screen would be uninitialised GPU memory.
    fn initial_modeset(&mut self) -> Result<()> {
        let idx = 0_usize;
        // Render a clear-only frame into slot 0.
        unsafe {
            self.stack
                .device
                .reset_command_buffer(
                    self.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .context("reset cb (initial modeset)")?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.stack
                .device
                .begin_command_buffer(self.command_buffer, &begin)
                .context("begin cb (initial modeset)")?;
        }
        let slot = &mut self.swap.slots[idx];
        let dst_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        record_layout_transition(
            &self.stack.device,
            self.command_buffer,
            slot.image,
            slot.layout,
            dst_layout,
        );
        slot.layout = dst_layout;
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(slot.view)
            .image_layout(dst_layout)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.04, 1.0],
                },
            })
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        unsafe {
            self.stack
                .device
                .cmd_begin_rendering(self.command_buffer, &info);
            self.stack.device.cmd_end_rendering(self.command_buffer);
        }
        record_layout_transition(
            &self.stack.device,
            self.command_buffer,
            slot.image,
            slot.layout,
            vk::ImageLayout::GENERAL,
        );
        slot.layout = vk::ImageLayout::GENERAL;

        unsafe {
            self.stack
                .device
                .end_command_buffer(self.command_buffer)
                .context("end cb (initial modeset)")?;

            let cb = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&cb);
            self.stack
                .device
                .reset_fences(&[self.frame_fence])
                .ok();
            self.stack
                .device
                .queue_submit(self.stack.queue, &[submit], self.frame_fence)
                .context("queue_submit (initial modeset)")?;
            self.stack
                .device
                .wait_for_fences(&[self.frame_fence], true, u64::MAX)
                .context("wait_for_fences (initial modeset)")?;
        }

        let fb = self.add_or_get_fb(idx)?;
        self.card
            .set_crtc(
                self.sel.crtc,
                Some(fb),
                (0, 0),
                &[self.sel.connector],
                Some(self.sel.mode),
            )
            .context("set_crtc (initial modeset)")?;
        self.swap.scanned = Some(idx);
        tracing::info!(width = self.width, height = self.height, "modeset complete");
        Ok(())
    }

    /// The render entry point. See module docs for the full sequence.
    pub fn render_scene(
        &mut self,
        scene: &Scene,
        timing: &mut RenderTiming,
    ) -> Result<Option<OwnedFd>> {
        if self.is_busy() {
            return Ok(None);
        }

        let Some(slot_idx) = self.swap.acquire() else {
            tracing::warn!("no free swap slot — bug in slot bookkeeping");
            return Ok(None);
        };

        // Wait for the previous frame's submit to drain so we can reuse
        // the command buffer + staging buffers safely. Should be near-
        // instant: by the time we got out of `pending_render` /
        // `pending_flip` the GPU has already finished.
        unsafe {
            self.stack
                .device
                .wait_for_fences(&[self.frame_fence], true, u64::MAX)
                .ok();
            let _ = self.stack.device.reset_fences(&[self.frame_fence]);
        }

        let t_textures = Instant::now();

        // Blur-cache check: same as pre-rewrite — bg signature + blur
        // params equal across frames means the previous run's blurred
        // FBO still holds the right pixels.
        let bg_sig = compute_bg_signature(&scene.elements[..scene.background_count]);
        let scene_blur_params = scene
            .elements
            .iter()
            .find_map(|e| match &e.content {
                SceneContent::BlurredBackdrop { passes, radius } => {
                    Some((*passes, radius.to_bits()))
                }
                _ => None,
            })
            .unwrap_or((0, 0));
        let scene_has_backdrop = scene_blur_params.0 > 0;
        let blur_cache_valid = scene_has_backdrop
            && bg_sig == self.last_bg_sig
            && scene_blur_params == self.last_blur_params;
        self.blur_ready_this_frame = blur_cache_valid;
        self.last_bg_sig = bg_sig;
        self.last_blur_params = scene_blur_params;

        // SHM uploads: memcpy host pixels into staging, queue
        // PendingUpload records. dmabuf imports happen synchronously
        // inside prepare_uploads — they're once-per-buffer.
        self.textures.prepare_uploads(scene)?;

        unsafe {
            self.stack
                .device
                .reset_command_buffer(
                    self.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .context("reset cb")?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.stack
                .device
                .begin_command_buffer(self.command_buffer, &begin)
                .context("begin cb")?;
        }

        // Record the queued `vkCmdCopyBufferToImage`s for SHM uploads.
        // Each pair of barriers transitions the texture between
        // SHADER_READ_ONLY and TRANSFER_DST.
        self.textures.record_uploads(self.command_buffer);

        timing.textures_ns += t_textures.elapsed().as_nanos() as u64;

        // -- bg pre-pass + blur --
        let t_blur = Instant::now();
        let need_bg_capture = scene.background_count > 0
            && scene_has_backdrop
            && !blur_cache_valid
            && self.blur.is_some();
        if need_bg_capture {
            self.draw_background_capture(scene)?;
        }
        if scene_has_backdrop && !blur_cache_valid {
            if let Some(blur) = self.blur.as_mut() {
                blur.run(
                    self.command_buffer,
                    &self.pipelines,
                    scene_blur_params.0,
                    f32::from_bits(scene_blur_params.1),
                );
                self.blur_ready_this_frame = true;
            }
        }
        // Make sure the final-blur FBO is sampleable for the main draw.
        if scene_has_backdrop {
            if let Some(blur) = self.blur.as_mut() {
                let final_fbo = blur.final_fbo_mut(scene_blur_params.0);
                transition_blur(
                    self.command_buffer,
                    &self.stack.device,
                    final_fbo,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        }
        timing.blur_ns += t_blur.elapsed().as_nanos() as u64;

        // -- main pass: scene back-to-front into the slot's image --
        let t_draw = Instant::now();
        self.draw_main_pass(scene, slot_idx)?;
        timing.draw_ns += t_draw.elapsed().as_nanos() as u64;

        // Slot must be in a layout the kernel can scan out from. GENERAL
        // is the spec's catch-all; DRM doesn't track Vulkan layouts so
        // any "any operation" layout works.
        let slot = &mut self.swap.slots[slot_idx];
        record_layout_transition(
            &self.stack.device,
            self.command_buffer,
            slot.image,
            slot.layout,
            vk::ImageLayout::GENERAL,
        );
        slot.layout = vk::ImageLayout::GENERAL;

        unsafe {
            self.stack
                .device
                .end_command_buffer(self.command_buffer)
                .context("end cb")?;
        }

        // -- submit + sync FD export --
        let t_present = Instant::now();
        let r = self.submit_render(slot_idx, timing);
        timing.present_ns += t_present.elapsed().as_nanos() as u64;
        r
    }

    fn draw_background_capture(&mut self, scene: &Scene) -> Result<()> {
        let blur = self.blur.as_mut().expect("checked by caller");
        let cap = &mut blur.capture;
        let device = &self.stack.device;

        transition_blur(
            self.command_buffer,
            device,
            cap,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(cap.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.04, 1.0],
                },
            })
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: cap.width,
                    height: cap.height,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        unsafe {
            device.cmd_begin_rendering(self.command_buffer, &info);
            set_viewport_scissor(device, self.command_buffer, cap.width, cap.height);
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.quad.pipeline,
            );
        }
        let fb_w = cap.width as f32;
        let fb_h = cap.height as f32;
        for elem in &scene.elements[..scene.background_count] {
            self.draw_textured_element(elem, fb_w, fb_h);
        }
        unsafe { device.cmd_end_rendering(self.command_buffer) };
        Ok(())
    }

    fn draw_main_pass(&mut self, scene: &Scene, slot_idx: usize) -> Result<()> {
        let device = &self.stack.device;
        let slot = &mut self.swap.slots[slot_idx];
        record_layout_transition(
            device,
            self.command_buffer,
            slot.image,
            slot.layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        slot.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;

        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(slot.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.04, 1.0],
                },
            })
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);

        unsafe {
            device.cmd_begin_rendering(self.command_buffer, &info);
            set_viewport_scissor(device, self.command_buffer, self.width, self.height);
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.quad.pipeline,
            );
        }

        let fb_w = self.width as f32;
        let fb_h = self.height as f32;
        let border_anchor = scene.border_anchor.min(scene.elements.len());
        let mut borders_drawn = false;

        // Track which pipeline is currently bound so we only switch on
        // demand. Saves per-element bind churn for the common case
        // (long runs of textured quads).
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Bound {
            Quad,
            Backdrop,
        }
        let mut bound = Bound::Quad;

        for (idx, elem) in scene.elements.iter().enumerate() {
            if !borders_drawn && idx == border_anchor && !scene.borders.is_empty() {
                unsafe { device.cmd_end_rendering(self.command_buffer) };
                self.draw_solid_borders_pass(slot_idx, &scene.borders);
                // Re-begin the colour attachment for the rest of the
                // scene. LOAD = LOAD so we keep what we drew.
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(self.swap.slots[slot_idx].view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);
                let attachments = [attachment];
                let info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        },
                    })
                    .layer_count(1)
                    .color_attachments(&attachments);
                unsafe {
                    device.cmd_begin_rendering(self.command_buffer, &info);
                    set_viewport_scissor(device, self.command_buffer, self.width, self.height);
                    device.cmd_bind_pipeline(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.quad.pipeline,
                    );
                }
                bound = Bound::Quad;
                borders_drawn = true;
            }

            if let SceneContent::BlurredBackdrop { passes, radius } = &elem.content {
                if bound != Bound::Backdrop {
                    unsafe {
                        device.cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.backdrop.pipeline,
                        );
                    }
                    bound = Bound::Backdrop;
                }
                if let Err(e) = self.draw_backdrop_element(elem, *passes, *radius) {
                    tracing::warn!(error = %e, "backdrop draw skipped");
                }
                continue;
            }

            if bound != Bound::Quad {
                unsafe {
                    device.cmd_bind_pipeline(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.quad.pipeline,
                    );
                }
                bound = Bound::Quad;
            }
            self.draw_textured_element(elem, fb_w, fb_h);
        }

        unsafe { device.cmd_end_rendering(self.command_buffer) };

        // Border fallback: anchor was past the last element / not set.
        if !borders_drawn && !scene.borders.is_empty() {
            self.draw_solid_borders_pass(slot_idx, &scene.borders);
        }
        Ok(())
    }

    fn draw_solid_borders_pass(&self, slot_idx: usize, borders: &[SceneBorder]) {
        let device = &self.stack.device;
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swap.slots[slot_idx].view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        let fb_w = self.width as f32;
        let fb_h = self.height as f32;
        unsafe {
            device.cmd_begin_rendering(self.command_buffer, &info);
            set_viewport_scissor(device, self.command_buffer, self.width, self.height);
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.solid.pipeline,
            );
        }
        for b in borders {
            let (rx, ry, rw, rh) = ndc_rect(b.x, b.y, b.w, b.h, fb_w, fb_h);
            let pc = SolidPC {
                rect: [rx, ry, rw, rh],
                color: b.rgba,
                size: [b.w, b.h],
                radius: b.corner_radius.max(0.0),
                border: b.border_width.max(0.0),
            };
            unsafe {
                device.cmd_push_constants(
                    self.command_buffer,
                    self.pipelines.solid.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytes_of(&pc),
                );
                device.cmd_draw(self.command_buffer, 4, 1, 0, 0);
            }
        }
        unsafe { device.cmd_end_rendering(self.command_buffer) };
    }

    fn draw_textured_element(
        &self,
        elem: &sol_core::SceneElement<'_>,
        fb_w: f32,
        fb_h: f32,
    ) {
        if matches!(elem.content, SceneContent::BlurredBackdrop { .. }) {
            return;
        }
        let Some(entry) = self.textures.get(elem.buffer_key) else {
            return;
        };
        let dst_w = if elem.dst_width > 0.0 {
            elem.dst_width
        } else {
            elem.width as f32
        };
        let dst_h = if elem.dst_height > 0.0 {
            elem.dst_height
        } else {
            elem.height as f32
        };
        let opaque = match &elem.content {
            SceneContent::Shm {
                format: PixelFormat::Argb8888,
                ..
            } => 0.0,
            SceneContent::Shm {
                format: PixelFormat::Xrgb8888,
                ..
            } => 1.0,
            SceneContent::Dmabuf { fourcc, .. } => {
                if (*fourcc & 0xFF) as u8 == b'X' {
                    1.0
                } else {
                    0.0
                }
            }
            SceneContent::BlurredBackdrop { .. } => unreachable!(),
        };
        let (rx, ry, rw, rh) = ndc_rect(elem.x, elem.y, dst_w, dst_h, fb_w, fb_h);
        let pc = QuadPC {
            rect: [rx, ry, rw, rh],
            uv: [elem.uv_x, elem.uv_y, elem.uv_w, elem.uv_h],
            size: [dst_w, dst_h],
            radius: elem.corner_radius.max(0.0),
            alpha: elem.alpha.clamp(0.0, 1.0),
            opaque,
            _pad: [0.0; 3],
        };
        unsafe {
            self.stack.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.quad.layout,
                0,
                &[entry.descriptor],
                &[],
            );
            self.stack.device.cmd_push_constants(
                self.command_buffer,
                self.pipelines.quad.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes_of(&pc),
            );
            self.stack
                .device
                .cmd_draw(self.command_buffer, 4, 1, 0, 0);
        }
    }

    fn draw_backdrop_element(
        &self,
        elem: &sol_core::SceneElement<'_>,
        passes: u32,
        _radius: f32,
    ) -> Result<()> {
        let Some(blur) = self.blur.as_ref() else {
            anyhow::bail!("blur unavailable");
        };
        let final_fbo = blur.final_fbo(passes);
        let dst_w = if elem.dst_width > 0.0 {
            elem.dst_width
        } else {
            elem.width as f32
        };
        let dst_h = if elem.dst_height > 0.0 {
            elem.dst_height
        } else {
            elem.height as f32
        };
        let fb_w = self.width as f32;
        let fb_h = self.height as f32;
        let (rx, ry, rw, rh) = ndc_rect(elem.x, elem.y, dst_w, dst_h, fb_w, fb_h);
        // Vulkan-native UVs: (top-left origin), so the screen-rect
        // sub-rect maps directly to the blur tex coordinates without
        // any V-flip — every image we sample (scan-out, capture,
        // ping/pong, client buffers) is stored top-row-first.
        let pc = BackdropPC {
            rect: [rx, ry, rw, rh],
            uv: [
                elem.x / fb_w,
                elem.y / fb_h,
                dst_w / fb_w,
                dst_h / fb_h,
            ],
            size: [dst_w, dst_h],
            radius: elem.corner_radius.max(0.0),
            alpha: elem.alpha.clamp(0.0, 1.0),
        };
        unsafe {
            self.stack.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.backdrop.layout,
                0,
                &[final_fbo.descriptor],
                &[],
            );
            self.stack.device.cmd_push_constants(
                self.command_buffer,
                self.pipelines.backdrop.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes_of(&pc),
            );
            self.stack
                .device
                .cmd_draw(self.command_buffer, 4, 1, 0, 0);
        }
        Ok(())
    }

    fn submit_render(
        &mut self,
        slot_idx: usize,
        timing: &mut RenderTiming,
    ) -> Result<Option<OwnedFd>> {
        let t_swap = Instant::now();
        let cb_array = [self.command_buffer];

        if let Some(sem) = self.sync_semaphore {
            // Fast path: signal the exportable semaphore, dup the FD,
            // hand it back so the wayland-side calloop can defer the
            // page flip.
            let signals = [sem];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&cb_array)
                .signal_semaphores(&signals);
            unsafe {
                self.stack
                    .device
                    .queue_submit(self.stack.queue, &[submit], self.frame_fence)
                    .context("queue_submit (fence path)")?;
            }
            timing.present_swap_buffers_ns += t_swap.elapsed().as_nanos() as u64;

            match export_sync_fd(&self.stack, sem) {
                Ok(fd) => {
                    self.in_flight_slot = Some(slot_idx);
                    self.pending_render = true;
                    Ok(Some(fd))
                }
                Err(e) => {
                    // Export failed mid-flight — fall back to a wait
                    // and inline page_flip so the frame still lands.
                    tracing::warn!(error = %e, "vkGetSemaphoreFdKHR failed; inline page_flip");
                    unsafe {
                        let _ = self.stack.device.wait_for_fences(
                            &[self.frame_fence],
                            true,
                            u64::MAX,
                        );
                    }
                    self.inline_flip(slot_idx, timing)
                }
            }
        } else {
            // Synchronous fallback: submit, fence-wait, flip inline.
            let submit = vk::SubmitInfo::default().command_buffers(&cb_array);
            unsafe {
                self.stack
                    .device
                    .queue_submit(self.stack.queue, &[submit], self.frame_fence)
                    .context("queue_submit (fallback)")?;
            }
            timing.present_swap_buffers_ns += t_swap.elapsed().as_nanos() as u64;
            let t_lock = Instant::now();
            unsafe {
                self.stack
                    .device
                    .wait_for_fences(&[self.frame_fence], true, u64::MAX)
                    .context("wait_for_fences (fallback)")?;
            }
            timing.present_lock_front_ns += t_lock.elapsed().as_nanos() as u64;
            self.inline_flip(slot_idx, timing)
        }
    }

    fn inline_flip(
        &mut self,
        slot_idx: usize,
        timing: &mut RenderTiming,
    ) -> Result<Option<OwnedFd>> {
        let t_fb = Instant::now();
        let fb = self.add_or_get_fb(slot_idx)?;
        timing.present_add_fb_ns += t_fb.elapsed().as_nanos() as u64;
        let t_flip = Instant::now();
        self.card
            .page_flip(self.sel.crtc, fb, PageFlipFlags::EVENT, None)
            .context("page_flip (inline)")?;
        timing.present_page_flip_ns += t_flip.elapsed().as_nanos() as u64;
        self.swap.submit_flip(slot_idx);
        self.pending_flip = true;
        Ok(None)
    }

    /// Deferred half of the fast path: sync FD signalled, we can lock
    /// the slot's framebuffer and queue the flip without blocking on
    /// the GPU.
    pub fn submit_flip_after_fence(&mut self) -> Result<RenderTiming> {
        let mut t = RenderTiming::default();
        let Some(slot_idx) = self.in_flight_slot else {
            return Ok(t);
        };
        let result = (|| -> Result<()> {
            let t_lock = Instant::now();
            // Sanity: the GPU is done because the sync-FD signalled.
            // Reset the fence here (it was signalled on submit).
            t.present_lock_front_ns = t_lock.elapsed().as_nanos() as u64;

            let t_fb = Instant::now();
            let fb = self.add_or_get_fb(slot_idx)?;
            t.present_add_fb_ns = t_fb.elapsed().as_nanos() as u64;

            let t_flip = Instant::now();
            self.card
                .page_flip(self.sel.crtc, fb, PageFlipFlags::EVENT, None)
                .context("page_flip after fence signal")?;
            t.present_page_flip_ns = t_flip.elapsed().as_nanos() as u64;

            self.swap.submit_flip(slot_idx);
            self.pending_flip = true;
            Ok(())
        })();
        self.in_flight_slot = None;
        self.pending_render = false;
        result.map(|_| t)
    }
}

impl Drop for DrmPresenter {
    fn drop(&mut self) {
        unsafe {
            // Best-effort drain so we can free GPU resources without
            // tripping validation. The Arc'd VkStack keeps the device
            // alive until the last component drops.
            let _ = self.stack.device.device_wait_idle();
            if let Some(sem) = self.sync_semaphore {
                self.stack.device.destroy_semaphore(sem, None);
            }
            self.stack.device.destroy_fence(self.frame_fence, None);
            // Command buffer is freed when the pool is destroyed (by VkStack::Drop).
        }
        // Restore the pre-modeset CRTC so fbcon can re-scan the text
        // console framebuffer and the TTY unblanks cleanly. Best-
        // effort: log on failure, don't propagate.
        if let Some(saved) = self.saved_crtc.take() {
            tracing::info!(
                from_width = self.width,
                from_height = self.height,
                "restoring prior CRTC state (on high-bandwidth modes the DP link may take 10-15s to re-train)"
            );
            if let Err(e) = self.card.set_crtc(
                self.sel.crtc,
                saved.fb,
                saved.position,
                &[self.sel.connector],
                saved.mode,
            ) {
                tracing::warn!(error = ?e, "CRTC restore on drop failed; TTY may stay blank");
            } else {
                tracing::info!("restored prior CRTC state");
            }
        }
    }
}

/// Hash the bg slice into a u64 for the blur-cache equality check.
/// Same convention as the pre-rewrite version: buffer_key + on-screen
/// rect of every element, plus slice length to disambiguate empty.
fn compute_bg_signature(elems: &[sol_core::SceneElement<'_>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    elems.len().hash(&mut h);
    for e in elems {
        e.buffer_key.hash(&mut h);
        e.x.to_bits().hash(&mut h);
        e.y.to_bits().hash(&mut h);
        e.dst_width.to_bits().hash(&mut h);
        e.dst_height.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Convert a screen-space rect (top-left + size in pixels) to a Vulkan
/// NDC rect (top-left at NDC y = -1, +y down). The vertex shader feeds
/// `a_pos` ∈ [0,1]² across the quad, so `(rx + a_pos.x*rw, ry + a_pos.y*rh)`
/// lands in the right place without any flip or per-vertex math.
fn ndc_rect(x: f32, y: f32, w: f32, h: f32, fb_w: f32, fb_h: f32) -> (f32, f32, f32, f32) {
    let rx = x / fb_w * 2.0 - 1.0;
    let ry = y / fb_h * 2.0 - 1.0;
    let rw = w / fb_w * 2.0;
    let rh = h / fb_h * 2.0;
    (rx, ry, rw, rh)
}

fn record_layout_transition(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    if from == to {
        return;
    }
    let (src_stage, src_access) = pick_access(from);
    let (dst_stage, dst_access) = pick_access(to);
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let dep = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cb, &dep) };
}

fn pick_access(layout: vk::ImageLayout) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        vk::ImageLayout::GENERAL => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
    }
}

/// Create a binary semaphore exportable as a sync FD. Returns `None` if
/// the driver doesn't advertise the necessary handle-type pair —
/// common path on Mesa is to succeed.
fn create_export_semaphore(stack: &VkStack) -> Option<vk::Semaphore> {
    let mut export_info = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
    let sem = unsafe { stack.device.create_semaphore(&info, None).ok()? };
    Some(sem)
}

fn export_sync_fd(stack: &VkStack, sem: vk::Semaphore) -> Result<OwnedFd> {
    use std::os::fd::FromRawFd;
    let info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(sem)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let raw = unsafe {
        stack
            .ext_sem_fd
            .get_semaphore_fd(&info)
            .map_err(|e| anyhow!("vkGetSemaphoreFdKHR: {e:?}"))?
    };
    if raw < 0 {
        anyhow::bail!("vkGetSemaphoreFdKHR returned negative FD ({raw})");
    }
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

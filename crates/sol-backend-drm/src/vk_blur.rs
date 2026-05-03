//! Blur FBOs for the inactive-window frosted-glass backdrop.
//!
//! Three offscreen images:
//! - `capture` at full output resolution — receives the background slice
//!   (wallpaper + bottom layers + their subsurfaces) as a render-to-texture
//!   pre-pass at the start of each frame.
//! - `ping` / `pong` at half resolution — ping-ponged between blur passes;
//!   each pass samples one and renders into the other.
//!
//! Three is the minimum that lets us keep the capture intact across the
//! blur loop (pass 0 reads `capture` and writes `ping`; passes 1..N
//! alternate ping/pong). Half resolution costs a quarter the fragment
//! work at a quality loss the eye can't see after a couple of passes.

use anyhow::{Result, anyhow};
use ash::vk;

use crate::vk_pipe::{BlurPC, Pipelines};
use crate::vk_stack::SharedStack;
use crate::vk_swap::SCANOUT_VK_FORMAT;
use crate::vk_texture::TextureCache;

pub struct BlurFbo {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub descriptor: vk::DescriptorSet,
    pub width: u32,
    pub height: u32,
    pub layout: vk::ImageLayout,
}

pub struct BlurChain {
    stack: SharedStack,
    pub capture: BlurFbo,
    pub ping: BlurFbo,
    pub pong: BlurFbo,
}

impl BlurChain {
    pub fn new(
        stack: SharedStack,
        cache: &TextureCache,
        full_w: u32,
        full_h: u32,
    ) -> Result<Self> {
        let half_w = (full_w / 2).max(1);
        let half_h = (full_h / 2).max(1);
        let capture = create_fbo(&stack, cache, full_w, full_h)?;
        let ping = create_fbo(&stack, cache, half_w, half_h)?;
        let pong = create_fbo(&stack, cache, half_w, half_h)?;
        Ok(Self {
            stack,
            capture,
            ping,
            pong,
        })
    }

    /// Run `passes` rounds of blur ping-ponging between `ping` and `pong`,
    /// sourcing the capture image on the first pass. Caller is responsible
    /// for having already populated `capture` (typically by drawing the
    /// background slice into it earlier in the same command buffer).
    /// Returns the descriptor set + image of whichever FBO holds the
    /// final result, so the back-drop draw can sample it.
    pub fn run(&mut self, cb: vk::CommandBuffer, pipelines: &Pipelines, passes: u32, radius: f32) {
        let device = &self.stack.device;
        let total = passes.max(1);
        for i in 0..total {
            let (src, dst) = if i == 0 {
                (&mut self.capture, &mut self.ping)
            } else if i % 2 == 1 {
                (&mut self.ping, &mut self.pong)
            } else {
                (&mut self.pong, &mut self.ping)
            };

            transition(cb, device, src, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            transition(cb, device, dst, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(dst.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let attachments = [attachment];
            let render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: dst.width,
                        height: dst.height,
                    },
                })
                .layer_count(1)
                .color_attachments(&attachments);

            unsafe {
                device.cmd_begin_rendering(cb, &render_info);
                set_viewport_scissor(device, cb, dst.width, dst.height);
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.blur.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.blur.layout,
                    0,
                    &[src.descriptor],
                    &[],
                );

                let pc = BlurPC {
                    rect: [-1.0, -1.0, 2.0, 2.0],
                    uv: [0.0, 0.0, 1.0, 1.0],
                    texel: [
                        radius.max(0.0) / src.width as f32,
                        radius.max(0.0) / src.height as f32,
                    ],
                    _pad: [0.0, 0.0],
                };
                device.cmd_push_constants(
                    cb,
                    pipelines.blur.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytes_of(&pc),
                );
                device.cmd_draw(cb, 4, 1, 0, 0);
                device.cmd_end_rendering(cb);
            }
        }
    }

    /// Which of `ping` / `pong` (or `capture` if `passes` was 0) holds
    /// the final blurred result. The `total = passes.max(1)`
    /// convention matches `run`, so callers stay in sync.
    pub fn final_fbo(&self, passes: u32) -> &BlurFbo {
        let total = passes.max(1);
        if total % 2 == 1 {
            &self.ping
        } else {
            &self.pong
        }
    }

    pub fn final_fbo_mut(&mut self, passes: u32) -> &mut BlurFbo {
        let total = passes.max(1);
        if total % 2 == 1 {
            &mut self.ping
        } else {
            &mut self.pong
        }
    }
}

impl Drop for BlurChain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stack.device.device_wait_idle();
            for f in [&mut self.capture, &mut self.ping, &mut self.pong] {
                self.stack.device.destroy_image_view(f.view, None);
                self.stack.device.destroy_image(f.image, None);
                self.stack.device.free_memory(f.memory, None);
                // Descriptor set is freed via the texture cache's pool
                // reset on its own Drop; nothing to release here.
                let _ = f.descriptor;
            }
        }
    }
}

fn create_fbo(
    stack: &SharedStack,
    cache: &TextureCache,
    width: u32,
    height: u32,
) -> Result<BlurFbo> {
    let device = &stack.device;
    let info = vk::ImageCreateInfo::default()
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
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = stack.find_memtype(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(SCANOUT_VK_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_info, None)? };
    let descriptor = cache.allocate_descriptor(view).map_err(|e| {
        unsafe {
            device.destroy_image_view(view, None);
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        anyhow!("blur fbo descriptor: {e}")
    })?;
    Ok(BlurFbo {
        image,
        view,
        memory,
        descriptor,
        width,
        height,
        layout: vk::ImageLayout::UNDEFINED,
    })
}

/// Drive `fbo.layout` to `target` via a synchronization2 pipeline barrier.
/// All transitions go through `cmd_pipeline_barrier2`; we never reuse the
/// legacy `cmd_pipeline_barrier` form.
pub fn transition(
    cb: vk::CommandBuffer,
    device: &ash::Device,
    fbo: &mut BlurFbo,
    target: vk::ImageLayout,
) {
    if fbo.layout == target {
        return;
    }
    let (src_stage, src_access) = layout_access(fbo.layout, false);
    let (dst_stage, dst_access) = layout_access(target, true);
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(fbo.layout)
        .new_layout(target)
        .image(fbo.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let dep = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cb, &dep) };
    fbo.layout = target;
}

/// Pick stage / access masks from a layout for both src (just-finished
/// usage) and dst (about-to-start usage). Coverage is enough for our
/// pipeline: we only ever shuttle between UNDEFINED, COLOR_ATTACHMENT,
/// SHADER_READ_ONLY, and GENERAL.
fn layout_access(
    layout: vk::ImageLayout,
    is_dst: bool,
) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match layout {
        vk::ImageLayout::UNDEFINED => (
            if is_dst {
                vk::PipelineStageFlags2::TOP_OF_PIPE
            } else {
                vk::PipelineStageFlags2::TOP_OF_PIPE
            },
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

pub fn set_viewport_scissor(device: &ash::Device, cb: vk::CommandBuffer, w: u32, h: u32) {
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: w as f32,
        height: h as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: w,
            height: h,
        },
    };
    unsafe {
        device.cmd_set_viewport(cb, 0, &[viewport]);
        device.cmd_set_scissor(cb, 0, &[scissor]);
    }
}

/// Reinterpret a `T: Sized + Copy` as bytes for `cmd_push_constants`.
/// Matches the GLSL push-constant block layout (manual `layout(offset=N)`
/// in the shaders, no padding tricks needed beyond the explicit `_pad`
/// fields in the Rust structs).
pub fn bytes_of<T: Copy>(t: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((t as *const T) as *const u8, std::mem::size_of::<T>())
    }
}

//! Graphics pipelines.
//!
//! Four shaders produce four pipelines, all sharing the `quad.vert` vertex
//! shader: a fullscreen TriangleStrip generated from `gl_VertexIndex` (no
//! vertex buffers), a 2D NDC rect picked by the per-draw push constants,
//! and per-fragment UV / position passthrough. The fragment differs per
//! pipeline:
//!
//! - `quad`     — textured surface with rounded-rect mask + per-element
//!                alpha + XRGB opaque flag. Used for SHM and dmabuf
//!                clients alike (we always import to `VK_FORMAT_B8G8R8A8_UNORM`).
//! - `solid`    — flat-colour rounded ring or filled rounded rect. Used
//!                for the focused-tile border and any future overlays.
//! - `blur`     — 5×5 box-blur ping-pong. Sampled-source, opaque write.
//! - `backdrop` — sample the blurred FBO at a window's screen rect and
//!                composite under inactive windows.
//!
//! All textured pipelines use one descriptor set with one combined image
//! sampler binding (`set = 0, binding = 0`). The solid pipeline uses no
//! descriptor sets at all (push constants only) — empty pipeline layout.
//!
//! `dynamic_rendering` (Vulkan 1.3 core) means we don't carry around
//! `VkRenderPass` / `VkFramebuffer` objects; pipelines just declare the
//! colour-attachment format(s) they're compatible with at create time.

use anyhow::{Context, Result};
use ash::vk;

use crate::vk_stack::SharedStack;
use crate::vk_swap::SCANOUT_VK_FORMAT;

/// Pre-compiled SPIR-V binaries from `build.rs`. The `.spv` files are
/// produced by `glslc` and embedded at compile time.
const QUAD_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quad.vert.spv"));
const QUAD_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quad.frag.spv"));
const SOLID_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/solid.frag.spv"));
const BLUR_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blur.frag.spv"));
const BACKDROP_FRAG_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/backdrop.frag.spv"));

/// Per-draw push constants for `quad.frag`. Match the GLSL block layout
/// (std140-ish, manual offsets via `layout(offset=N)`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QuadPC {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub size: [f32; 2],
    pub radius: f32,
    pub alpha: f32,
    pub opaque: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SolidPC {
    pub rect: [f32; 4],
    pub color: [f32; 4],
    pub size: [f32; 2],
    pub radius: f32,
    pub border: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BlurPC {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub texel: [f32; 2],
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BackdropPC {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub size: [f32; 2],
    pub radius: f32,
    pub alpha: f32,
}

pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

pub struct Pipelines {
    stack: SharedStack,
    pub sampled_set_layout: vk::DescriptorSetLayout,
    pub linear_sampler: vk::Sampler,
    pub quad: GraphicsPipeline,
    pub solid: GraphicsPipeline,
    pub blur: GraphicsPipeline,
    pub backdrop: GraphicsPipeline,
    quad_vert: vk::ShaderModule,
}

impl Pipelines {
    pub fn new(stack: SharedStack) -> Result<Self> {
        let device = &stack.device;

        // One descriptor set layout: a single combined image sampler at
        // (set=0, binding=0). Used by quad / blur / backdrop. The
        // `solid` pipeline has no sampled images, so its layout has no
        // descriptor sets and we don't bind one.
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let sampled_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_info, None)
                .context("create_descriptor_set_layout")?
        };

        // Shared LINEAR sampler with CLAMP_TO_EDGE — same parameters as
        // every previous GL `tex_parameter_i32` call. No per-texture
        // sampler tuning: every surface samples the same way.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .anisotropy_enable(false)
            .unnormalized_coordinates(false);
        let linear_sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .context("create sampler")?
        };

        // Shared vertex shader. Loaded once and shared across all four
        // pipelines — saves a few KB of driver state vs. compiling
        // separate copies.
        let quad_vert = create_shader(device, QUAD_VERT_SPV)?;

        let quad = build_pipeline(
            &stack,
            quad_vert,
            QUAD_FRAG_SPV,
            std::mem::size_of::<QuadPC>() as u32,
            Some(sampled_set_layout),
            BlendMode::SrcAlpha,
        )?;
        let solid = build_pipeline(
            &stack,
            quad_vert,
            SOLID_FRAG_SPV,
            std::mem::size_of::<SolidPC>() as u32,
            None,
            BlendMode::SrcAlpha,
        )?;
        let blur = build_pipeline(
            &stack,
            quad_vert,
            BLUR_FRAG_SPV,
            std::mem::size_of::<BlurPC>() as u32,
            Some(sampled_set_layout),
            BlendMode::Opaque,
        )?;
        let backdrop = build_pipeline(
            &stack,
            quad_vert,
            BACKDROP_FRAG_SPV,
            std::mem::size_of::<BackdropPC>() as u32,
            Some(sampled_set_layout),
            BlendMode::SrcAlpha,
        )?;

        Ok(Self {
            stack,
            sampled_set_layout,
            linear_sampler,
            quad,
            solid,
            blur,
            backdrop,
            quad_vert,
        })
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        unsafe {
            let device = &self.stack.device;
            for gp in [&self.quad, &self.solid, &self.blur, &self.backdrop] {
                device.destroy_pipeline(gp.pipeline, None);
                device.destroy_pipeline_layout(gp.layout, None);
            }
            device.destroy_shader_module(self.quad_vert, None);
            device.destroy_sampler(self.linear_sampler, None);
            device.destroy_descriptor_set_layout(self.sampled_set_layout, None);
        }
    }
}

#[derive(Clone, Copy)]
enum BlendMode {
    /// Standard non-premultiplied blend: SRC_ALPHA + ONE_MINUS_SRC_ALPHA.
    SrcAlpha,
    /// No blending — destination is overwritten. Used for the blur
    /// passes where source already covers every pixel.
    Opaque,
}

fn create_shader(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    // SPIR-V words are 4-byte aligned in the file; reinterpret the byte
    // slice as &[u32]. Safe because the contents come from include_bytes!
    // of a glslc-produced .spv file, which is always a whole number of
    // u32 words.
    debug_assert_eq!(spv.len() % 4, 0);
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&info, None)? })
}

fn build_pipeline(
    stack: &SharedStack,
    vs: vk::ShaderModule,
    fs_spv: &[u8],
    push_constant_size: u32,
    sampled_layout: Option<vk::DescriptorSetLayout>,
    blend: BlendMode,
) -> Result<GraphicsPipeline> {
    let device = &stack.device;
    let fs = create_shader(device, fs_spv)?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs)
            .name(entry),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
        .primitive_restart_enable(false);

    // One viewport / scissor; both dynamic so the same pipeline drives
    // any framebuffer size (full-res capture, half-res blur, scan-out).
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_state = match blend {
        BlendMode::SrcAlpha => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
        BlendMode::Opaque => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
    };
    let blend_attachments = [blend_state];
    let blend_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);

    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    // Pipeline layout: one descriptor set (or zero) + push constant range
    // covering both vertex and fragment stages so we can write `pc.rect`
    // from the vertex shader and `pc.size` from the fragment shader
    // through the same block.
    let set_layouts: &[vk::DescriptorSetLayout] = if let Some(l) = sampled_layout.as_ref() {
        std::slice::from_ref(l)
    } else {
        &[]
    };
    let pc_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(push_constant_size)];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(set_layouts)
        .push_constant_ranges(&pc_ranges);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    // Dynamic rendering: declare the colour attachment formats the
    // pipeline will be used with. All four pipelines render into
    // `B8G8R8A8_UNORM` images (scan-out, capture FBO, blur ping/pong),
    // so one entry suffices.
    let color_formats = [SCANOUT_VK_FORMAT];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend_info)
        .dynamic_state(&dyn_state)
        .layout(layout)
        .push_next(&mut rendering_info);

    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    }
    .map_err(|(_, e)| anyhow::anyhow!("create_graphics_pipelines: {e:?}"))?;

    // Fragment shader module is no longer needed once the pipeline is
    // baked. Vertex shader is shared and freed at `Pipelines::Drop`.
    unsafe { device.destroy_shader_module(fs, None) };

    Ok(GraphicsPipeline {
        pipeline: pipelines[0],
        layout,
    })
}

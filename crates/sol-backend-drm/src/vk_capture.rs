//! Frame capture ring — the "dashcam" for visual-glitch hunting.
//!
//! Two tiers fed by one mechanism (see doc/nvidia-rendering.md for the
//! debugging story that motivated this):
//!
//! - a RAM ring of the last ~2 s of downscaled frames, dumped to disk
//!   only when `mark()` fires (a debug-ctl command or the
//!   `capture_mark` keybind) — catches single-frame transients;
//! - a sampled persist tier (default 10 Hz) that streams every Nth
//!   captured frame straight to disk for "there was a glitch at
//!   second 1.32" workflows.
//!
//! Per frame: blit the finished scan-out image (downscaled, linear
//! filter) into a small persistent image, copy that to a host-visible
//! staging buffer, and — once the frame's fence has been waited (the
//! pipeline is fully serialized, so "start of the next render") —
//! memcpy the bytes into the CPU ring together with metadata: frame
//! index, timestamp, the frame's `RenderTiming` snapshot, and a JSON
//! description of every scene element (rect/kind/alpha/key). The
//! element list is what turns glitch detection into invariant checking
//! ("no pixel inside an opaque window's rect may match the clear
//! color") instead of guesswork.
//!
//! PNG encoding happens on a detached writer thread; the compositor
//! thread only ever does the ~2 MB memcpy.

use anyhow::{Context, Result};
use ash::vk;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::mpsc;
use std::time::Instant;

use crate::vk_stack::SharedStack;

/// Metadata captured alongside each frame's pixels.
#[derive(Clone)]
pub struct CaptureMeta {
    pub frame_index: u64,
    /// Milliseconds since capture was started.
    pub t_ms: f64,
    /// The frame's RenderTiming as of blit-record time. CPU phase
    /// fields up to the draw pass are filled; `gpu_*` lag one frame
    /// (see vk_perf).
    pub timing: sol_core::RenderTiming,
    /// JSON array describing the scene elements (hand-rolled — the
    /// backend deliberately has no serde dependency).
    pub scene_json: String,
}

struct CapturedFrame {
    bytes: Vec<u8>,
    meta: CaptureMeta,
}

/// What the writer thread receives.
struct WriteJob {
    bytes: Vec<u8>,
    meta: CaptureMeta,
    width: u32,
    height: u32,
    src_width: u32,
    src_height: u32,
}

pub struct CaptureRing {
    stack: SharedStack,
    image: vk::Image,
    memory: vk::DeviceMemory,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_size: usize,
    pub width: u32,
    pub height: u32,
    src_width: u32,
    src_height: u32,
    /// Pending = a blit+copy was recorded this frame; finalized into
    /// `ring` by `collect()` after the fence wait.
    pending: Option<CaptureMeta>,
    ring: VecDeque<CapturedFrame>,
    ring_cap: usize,
    /// Capture every Nth rendered frame into the ring.
    pub every: u32,
    /// Persist a frame to disk when at least this much time has
    /// passed since the last persisted one (the 10 Hz tier).
    pub persist_interval_ms: f64,
    started_at: Instant,
    last_persist_ms: f64,
    frames_seen: u64,
    writer: mpsc::Sender<WriteJob>,
    pub out_dir: std::path::PathBuf,
}

// The staging pointer is into Vulkan memory owned for the ring's
// lifetime; the compositor is single-threaded (writer thread only
// receives owned Vecs).
unsafe impl Send for CaptureRing {}

impl CaptureRing {
    pub fn new(
        stack: SharedStack,
        src_width: u32,
        src_height: u32,
        every: u32,
        persist_hz: f32,
        ring_seconds: f32,
        refresh_hz: f32,
    ) -> Result<Self> {
        // Quarter-res capture, even dimensions for the blit.
        let width = (src_width / 4).max(2) & !1;
        let height = (src_height / 4).max(2) & !1;
        let device = &stack.device;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&image_info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let mem_type =
            stack.find_memtype(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        let staging_size = (width as usize) * (height as usize) * 4;
        let buf_info = vk::BufferCreateInfo::default()
            .size(staging_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buf = unsafe { device.create_buffer(&buf_info, None)? };
        let breq = unsafe { device.get_buffer_memory_requirements(staging_buf) };
        let bmt = stack.find_memtype(
            breq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let balloc = vk::MemoryAllocateInfo::default()
            .allocation_size(breq.size)
            .memory_type_index(bmt);
        let staging_mem = unsafe { device.allocate_memory(&balloc, None)? };
        unsafe { device.bind_buffer_memory(staging_buf, staging_mem, 0)? };
        let staging_ptr = unsafe {
            device
                .map_memory(staging_mem, 0, breq.size, vk::MemoryMapFlags::empty())
                .context("map capture staging")? as *mut u8
        };

        let ring_cap = ((ring_seconds * refresh_hz / every as f32).ceil() as usize).max(8);

        let out_dir = std::env::var_os("SOL_CAPTURE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp/sol-capture"));
        std::fs::create_dir_all(&out_dir).context("create capture dir")?;
        let writer = spawn_writer(out_dir.clone());

        tracing::info!(
            width,
            height,
            every,
            ring_cap,
            persist_hz,
            dir = %out_dir.display(),
            "capture ring armed"
        );
        Ok(Self {
            stack,
            image,
            memory,
            staging_buf,
            staging_mem,
            staging_ptr,
            staging_size,
            width,
            height,
            src_width,
            src_height,
            pending: None,
            ring: VecDeque::new(),
            ring_cap,
            every: every.max(1),
            persist_interval_ms: if persist_hz > 0.0 { 1000.0 / persist_hz as f64 } else { f64::MAX },
            started_at: Instant::now(),
            last_persist_ms: f64::MIN,
            frames_seen: 0,
            writer,
            out_dir,
        })
    }

    /// Should this frame be captured? Called once per rendered frame.
    pub fn want_frame(&mut self) -> bool {
        let n = self.frames_seen;
        self.frames_seen += 1;
        n % self.every as u64 == 0
    }

    /// Record the blit + buffer copy into the frame's command buffer.
    /// `src` must currently be COLOR_ATTACHMENT_OPTIMAL; it is
    /// returned to that layout afterwards so the scan-out release
    /// logic stays oblivious.
    pub fn record(
        &mut self,
        cb: vk::CommandBuffer,
        src: vk::Image,
        timing: &sol_core::RenderTiming,
        scene_json: String,
    ) {
        let device = &self.stack.device;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |image, old_l, new_l, src_stage, src_access, dst_stage, dst_access| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(old_l)
                .new_layout(new_l)
                .image(image)
                .subresource_range(range)
        };
        unsafe {
            // src: COLOR_ATTACHMENT -> TRANSFER_SRC; capture image -> TRANSFER_DST.
            let pre = [
                barrier(
                    src,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                barrier(
                    self.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            ];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&pre);
            device.cmd_pipeline_barrier2(cb, &dep);

            let sub = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let blit = vk::ImageBlit::default()
                .src_subresource(sub)
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.src_width as i32,
                        y: self.src_height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(sub)
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ]);
            device.cmd_blit_image(
                cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            // capture image -> TRANSFER_SRC; src back to COLOR_ATTACHMENT.
            let mid = [
                barrier(
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                barrier(
                    src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
            ];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&mid);
            device.cmd_pipeline_barrier2(cb, &dep);

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(self.width)
                .buffer_image_height(self.height)
                .image_subresource(sub)
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cb,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.staging_buf,
                &[region],
            );
        }
        self.pending = Some(CaptureMeta {
            frame_index: self.frames_seen.saturating_sub(1),
            t_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
            timing: *timing,
            scene_json,
        });
    }

    /// Finalize the previous frame's capture. Must be called after the
    /// frame fence has been waited (start of the next render tick).
    pub fn collect(&mut self) {
        let Some(meta) = self.pending.take() else { return };
        let bytes =
            unsafe { std::slice::from_raw_parts(self.staging_ptr, self.staging_size) }.to_vec();

        // 10 Hz tier: stream to disk on its own cadence.
        if meta.t_ms - self.last_persist_ms >= self.persist_interval_ms {
            self.last_persist_ms = meta.t_ms;
            let _ = self.writer.send(WriteJob {
                bytes: bytes.clone(),
                meta: meta.clone(),
                width: self.width,
                height: self.height,
                src_width: self.src_width,
                src_height: self.src_height,
            });
        }

        self.ring.push_back(CapturedFrame { bytes, meta });
        while self.ring.len() > self.ring_cap {
            self.ring.pop_front();
        }
    }

    /// Dump the whole RAM ring to disk — the "I just saw it" button.
    pub fn mark(&mut self) -> usize {
        let n = self.ring.len();
        for f in self.ring.drain(..) {
            let _ = self.writer.send(WriteJob {
                bytes: f.bytes,
                meta: f.meta,
                width: self.width,
                height: self.height,
                src_width: self.src_width,
                src_height: self.src_height,
            });
        }
        tracing::info!(frames = n, dir = %self.out_dir.display(), "capture ring marked + dumped");
        n
    }

    pub fn status(&self) -> String {
        format!(
            "ring {}/{} frames, every {}, persist {:.1}ms, dir {}",
            self.ring.len(),
            self.ring_cap,
            self.every,
            self.persist_interval_ms,
            self.out_dir.display()
        )
    }
}

impl Drop for CaptureRing {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stack.device.device_wait_idle();
            self.stack.device.destroy_buffer(self.staging_buf, None);
            self.stack.device.unmap_memory(self.staging_mem);
            self.stack.device.free_memory(self.staging_mem, None);
            self.stack.device.destroy_image(self.image, None);
            self.stack.device.free_memory(self.memory, None);
        }
    }
}

/// Detached PNG/JSONL writer. BGRA→RGB conversion and PNG encoding
/// happen here, never on the compositor thread.
fn spawn_writer(dir: std::path::PathBuf) -> mpsc::Sender<WriteJob> {
    let (tx, rx) = mpsc::channel::<WriteJob>();
    std::thread::Builder::new()
        .name("sol-capture-writer".into())
        .spawn(move || {
            let jsonl_path = dir.join("frames.jsonl");
            for job in rx {
                let name = format!(
                    "frame-{:06}-t{:09.3}.png",
                    job.meta.frame_index, job.meta.t_ms / 1000.0
                );
                let mut rgb = Vec::with_capacity((job.width * job.height * 3) as usize);
                for px in job.bytes.chunks_exact(4) {
                    rgb.extend_from_slice(&[px[2], px[1], px[0]]); // BGRA -> RGB
                }
                let path = dir.join(&name);
                let write_png = || -> std::io::Result<()> {
                    let file = std::fs::File::create(&path)?;
                    let mut enc = png::Encoder::new(
                        std::io::BufWriter::new(file),
                        job.width,
                        job.height,
                    );
                    enc.set_color(png::ColorType::Rgb);
                    enc.set_depth(png::BitDepth::Eight);
                    let mut w = enc.write_header()?;
                    w.write_image_data(&rgb)?;
                    Ok(())
                };
                if let Err(e) = write_png() {
                    tracing::warn!(error = %e, "capture png write failed");
                    continue;
                }
                let t = &job.meta.timing;
                let row = format!(
                    concat!(
                        "{{\"frame\":{},\"t_ms\":{:.3},\"file\":\"{}\",",
                        "\"src\":[{},{}],\"cap\":[{},{}],",
                        "\"cpu_ns\":{{\"textures\":{},\"blur\":{},\"draw\":{}}},",
                        "\"gpu_ns\":{{\"uploads\":{},\"blur\":{},\"draw\":{},\"total\":{}}},",
                        "\"shm_upload_bytes\":{},\"shm_skipped_bytes\":{},",
                        "\"dmabuf_fence_waits\":{},\"n_textured_draws\":{},",
                        "\"n_backdrop_draws\":{},\"elements\":{}}}\n"
                    ),
                    job.meta.frame_index,
                    job.meta.t_ms,
                    name,
                    job.src_width,
                    job.src_height,
                    job.width,
                    job.height,
                    t.textures_ns,
                    t.blur_ns,
                    t.draw_ns,
                    t.gpu_uploads_ns,
                    t.gpu_blur_ns,
                    t.gpu_draw_ns,
                    t.gpu_total_ns,
                    t.n_shm_upload_bytes,
                    t.n_shm_upload_skipped_bytes,
                    t.n_dmabuf_fence_waits,
                    t.n_textured_draws,
                    t.n_backdrop_draws,
                    job.meta.scene_json,
                );
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&jsonl_path)
                {
                    let _ = f.write_all(row.as_bytes());
                }
            }
        })
        .expect("spawn capture writer");
    tx
}

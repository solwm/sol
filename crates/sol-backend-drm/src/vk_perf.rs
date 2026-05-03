//! GPU-side timing via `VkQueryPool` timestamps.
//!
//! Vulkan exposes `VK_QUERY_TYPE_TIMESTAMP` queries that the GPU writes
//! at well-defined pipeline stages. Wall-clock CPU time inside the
//! renderer can lie — `vkCmdDraw` returns instantly because draws are
//! recorded, not executed — so to know whether the *GPU* is the
//! bottleneck we need to ask the GPU directly. This module owns one
//! query pool, exposes `cmd_write_*` helpers the presenter calls at
//! phase boundaries, and reads results back at the start of the next
//! frame (after the previous submit's fence has signalled).
//!
//! Lag: a frame's `gpu_*_ns` fields in `RenderTiming` are the *previous*
//! frame's GPU times. Acceptable for steady-state perf analysis — at
//! 240Hz one-frame lag is 4ms.
//!
//! Timestamps come back as integers in `timestamp_period`-nanosecond
//! ticks; we convert to ns up front before exposing.

use anyhow::{Context, Result};
use ash::vk;
use sol_core::RenderTiming;

use crate::vk_stack::SharedStack;

/// Slot indices in the query pool. Order matches the phase boundaries
/// recorded inside `record_frame`.
const TS_FRAME_START: u32 = 0;
const TS_AFTER_UPLOADS: u32 = 1;
const TS_AFTER_BLUR: u32 = 2;
const TS_FRAME_END: u32 = 3;
const TS_COUNT: u32 = 4;

pub struct GpuTimings {
    stack: SharedStack,
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick on this device. Some hardware
    /// reports `timestamp_period = 1.0` (already ns); discrete GPUs
    /// often report 1.0 too, but mobile / Mesa software stacks vary —
    /// always honour the value rather than assume.
    period_ns: f64,
    /// First frame has nothing in the pool yet; second frame onwards
    /// can read the previous frame's results.
    have_previous: bool,
}

impl GpuTimings {
    pub fn new(stack: SharedStack) -> Result<Option<Self>> {
        // Some drivers expose timestamp_period == 0 to signal "no
        // timestamp support on graphics queues" — fall back to
        // disabled rather than divide by zero.
        let period_ns = stack.limits.timestamp_period as f64;
        if period_ns <= 0.0 {
            tracing::warn!(
                "device timestamp_period == 0; GPU timestamps disabled"
            );
            return Ok(None);
        }
        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TS_COUNT);
        let pool = unsafe {
            stack
                .device
                .create_query_pool(&pool_info, None)
                .context("create timestamp query pool")?
        };
        Ok(Some(Self {
            stack,
            pool,
            period_ns,
            have_previous: false,
        }))
    }

    /// Read back the previous frame's timestamps and fold the GPU
    /// phase totals into `timing`. No-op on the first frame. Caller
    /// must have already waited on the previous submit's fence (we
    /// already do, at the top of `render_scene`), so results are
    /// available immediately — we pass `WITH_AVAILABILITY` to detect
    /// the rare case where they aren't.
    pub fn collect_previous(&mut self, timing: &mut RenderTiming) {
        if !self.have_previous {
            return;
        }
        // Five u64 slots: four timestamps + one availability word per
        // query when WITH_AVAILABILITY is set. We pass it AS WAIT-free
        // because the fence we already waited on guarantees the GPU
        // wrote them.
        let mut data: [u64; TS_COUNT as usize] = [0; TS_COUNT as usize];
        let res = unsafe {
            self.stack.device.get_query_pool_results(
                self.pool,
                0,
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if let Err(e) = res {
            tracing::trace!(error = ?e, "GPU timestamp results not ready");
            return;
        }
        let to_ns = |ticks: u64| (ticks as f64 * self.period_ns) as u64;
        let start = data[TS_FRAME_START as usize];
        let after_uploads = data[TS_AFTER_UPLOADS as usize];
        let after_blur = data[TS_AFTER_BLUR as usize];
        let end = data[TS_FRAME_END as usize];
        // Defensive: if the GPU reset between frames or wrapped a 32-bit
        // counter, skip rather than emit nonsense numbers.
        if after_uploads >= start && after_blur >= after_uploads && end >= after_blur {
            timing.gpu_uploads_ns = to_ns(after_uploads - start);
            timing.gpu_blur_ns = to_ns(after_blur - after_uploads);
            timing.gpu_draw_ns = to_ns(end - after_blur);
            timing.gpu_total_ns = to_ns(end - start);
        }
    }

    /// Reset the query pool and write the "frame start" timestamp.
    /// Called once per frame, just after `vkBeginCommandBuffer`.
    pub fn cmd_begin(&mut self, cb: vk::CommandBuffer) {
        unsafe {
            self.stack
                .device
                .cmd_reset_query_pool(cb, self.pool, 0, TS_COUNT);
            self.stack.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.pool,
                TS_FRAME_START,
            );
        }
    }

    /// "Uploads done" boundary — write after the SHM
    /// `vkCmdCopyBufferToImage` block + barriers.
    pub fn cmd_after_uploads(&self, cb: vk::CommandBuffer) {
        unsafe {
            self.stack.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                self.pool,
                TS_AFTER_UPLOADS,
            );
        }
    }

    /// "Blur done" boundary — write after the bg pre-pass + ping/pong
    /// blur passes.
    pub fn cmd_after_blur(&self, cb: vk::CommandBuffer) {
        unsafe {
            self.stack.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                self.pool,
                TS_AFTER_BLUR,
            );
        }
    }

    /// "Frame end" boundary — write at the very end of the cb, after
    /// the final layout transition to `GENERAL`.
    pub fn cmd_end(&mut self, cb: vk::CommandBuffer) {
        unsafe {
            self.stack.device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                TS_FRAME_END,
            );
        }
        self.have_previous = true;
    }
}

impl Drop for GpuTimings {
    fn drop(&mut self) {
        unsafe {
            self.stack.device.destroy_query_pool(self.pool, None);
        }
    }
}

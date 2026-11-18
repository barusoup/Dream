//! Per-pass GPU timing of the scene and post work.
//!
//! A frame records named marks, each a timestamp at its point in the command
//! buffer. Mark `i` starts the pass labelled `i`, so the pass lasts until mark
//! `i + 1`. The last mark only ends the previous pass. Each frame slot has its
//! own range of the query pool. The marks are read back when the slot's fence
//! has signalled, the same point the aggregate timestamps are read.

use super::timing::elapsed_timestamp_ticks;
use super::vulkan::GpuContext;
use ash::vk;
use std::sync::Arc;

/// Most marks one frame slot records.
pub(crate) const MAX_MARKS: u32 = 32;

pub(crate) struct PassTimer {
    ctx: Arc<GpuContext>,
    pool: vk::QueryPool,
    /// Labels of the marks each slot recorded in its last frame.
    labels: Vec<Vec<&'static str>>,
    /// Pass timings of the last frame read back, in recording order.
    timings: Vec<(&'static str, f32)>,
}

impl PassTimer {
    /// A pool of [`MAX_MARKS`] timestamps for each of `slots` frame slots.
    pub(crate) fn new(ctx: &Arc<GpuContext>, slots: usize) -> Self {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_MARKS * slots as u32);
        let pool =
            unsafe { ctx.device().create_query_pool(&info, None) }.expect("pass timer query pool");
        Self {
            ctx: Arc::clone(ctx),
            pool,
            labels: vec![Vec::new(); slots],
            timings: Vec::new(),
        }
    }

    /// Start a frame of `slot`: reset its queries and forget its marks.
    pub(crate) fn begin(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        let slots = self.labels.len();
        self.labels[slot % slots].clear();
        unsafe {
            device.cmd_reset_query_pool(
                cmd,
                self.pool,
                (slot % slots) as u32 * MAX_MARKS,
                MAX_MARKS,
            );
        }
    }

    /// Record a mark: the pass that follows it is called `label`.
    pub(crate) fn mark(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        label: &'static str,
    ) {
        let slots = self.labels.len();
        let slot = slot % slots;
        let labels = &mut self.labels[slot];
        if labels.len() as u32 >= MAX_MARKS {
            return;
        }
        let index = slot as u32 * MAX_MARKS + labels.len() as u32;
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                index,
            );
        }
        labels.push(label);
    }

    /// Read back the marks of `slot`'s last frame into [`Self::timings`]. The
    /// slot's fence must have signalled.
    pub(crate) fn collect(&mut self, slot: usize) {
        let slots = self.labels.len();
        let slot = slot % slots;
        let count = self.labels[slot].len();
        if count < 2 {
            return;
        }
        let mut ticks = vec![0u64; count];
        let device = self.ctx.device();
        let got = unsafe {
            device.get_query_pool_results(
                self.pool,
                slot as u32 * MAX_MARKS,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if got.is_err() {
            return;
        }
        let bits = self.ctx.timestamp_valid_bits();
        let period_ms = self.ctx.timestamp_period() * 1e-6;
        self.timings = (0..count - 1)
            .map(|i| {
                let ms = elapsed_timestamp_ticks(ticks[i], ticks[i + 1], bits)
                    .map_or(0.0, |t| t as f32 * period_ms);
                let ms = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
                (self.labels[slot][i], ms)
            })
            .collect();
    }

    /// The pass timings of the last frame read back.
    pub(crate) fn timings(&self) -> Vec<(&'static str, f32)> {
        self.timings.clone()
    }
}

impl Drop for PassTimer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_query_pool(self.pool, None);
        }
    }
}

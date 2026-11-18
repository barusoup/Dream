//! Native-resolution temporal anti-aliasing (`AaMode::Taa`).
//!
//! Two compute passes per frame, built on [`PostPass`]:
//!
//! - **Resolve** reads this frame's jittered colour, depth and velocity and the
//!   previous history, and writes the new history (ping-ponged between two
//!   images). It reprojects with the dilated velocity, filters the history with
//!   a five-tap Catmull-Rom, clips it to the neighbourhood's YCoCg box, and blends
//!   in a luminance-compressed space.
//! - **Sharpen** reads the new history and writes the image the blit samples.
//!   It never writes back into the history. With `sharpness == 0` it is skipped
//!   and the blit reads the history directly.
//!
//! Motion vectors follow the scene contract in [`crate::temporal`]: velocity is
//! `prev_uv - curr_uv` in bottom-up uv, so the resolve flips its y sign to read
//! top-down texels. Everything stays in `GENERAL` layout, as the other scene
//! passes do.

use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use crate::render::TaaSettings;
use ash::vk;
use std::sync::Arc;

const RESOLVE_WGSL: &str = include_str!("taa_resolve.wgsl");
const SHARPEN_WGSL: &str = include_str!("taa_sharpen.wgsl");
const HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Size of the `TaaParams` block both shaders read.
const PARAMS_SIZE: u64 = std::mem::size_of::<TaaParams>() as u64;

/// CPU mirror of `TaaParams` in `taa_resolve.wgsl` / `taa_sharpen.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParams {
    /// Render width, height, 1/width, 1/height.
    size: [f32; 4],
    /// xy: this frame's jitter in texels.
    jitter: [f32; 4],
    /// x: history weight, y: variance gamma, z: sharpness, w: anti-flicker.
    blend: [f32; 4],
    /// x: 1 when history is invalid.
    flags: [f32; 4],
}

/// The scene images the resolve reads, as sampled views in `GENERAL` layout.
pub(crate) struct TaaInputs {
    pub(crate) color: vk::ImageView,
    /// The depth aspect of the scene depth target.
    pub(crate) depth: vk::ImageView,
    pub(crate) velocity: vk::ImageView,
}

/// TAA resources for one render size. Built lazily with the scene targets.
pub(crate) struct Taa {
    ctx: Arc<GpuContext>,
    resolve: PostPass,
    sharpen: PostPass,
    /// Ping-pong history: the resolve writes `history[write]` and reads the other.
    history: [GpuImage; 2],
    output: GpuImage,
    /// Next history index to write; flips after every record.
    write: usize,
    /// False until a record has run; the first record resets history.
    primed: bool,
    sampler: vk::Sampler,
    params: Vec<GpuBuffer>,
    size: (u32, u32),
    /// The image the blit samples after the last record, and its view.
    blit_source: (vk::Image, vk::ImageView),
}

impl Taa {
    /// Build the passes and images for a `size` (render size) target.
    pub(crate) fn new(ctx: &Arc<GpuContext>, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let size = (w, h);
        let resolve_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampler,
            BindKind::Storage(HISTORY_FORMAT),
        ];
        let resolve = PostPass::new(ctx, "taa resolve", RESOLVE_WGSL, c"resolve", &resolve_kinds);
        let sharpen_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Storage(HISTORY_FORMAT),
        ];
        let sharpen = PostPass::new(ctx, "taa sharpen", SHARPEN_WGSL, c"sharpen", &sharpen_kinds);
        let history = [
            create_general_image(ctx, "taa history 0", size, HISTORY_FORMAT),
            create_general_image(ctx, "taa history 1", size, HISTORY_FORMAT),
        ];
        let output = create_general_image(ctx, "taa output", size, HISTORY_FORMAT);
        let blit_source = (history[0].image, history[0].view);
        Self {
            ctx: Arc::clone(ctx),
            resolve,
            sharpen,
            history,
            output,
            write: 0,
            primed: false,
            sampler: ctx.create_sampler("taa history sampler"),
            params: host_uniform_slots(ctx, "taa params", PARAMS_SIZE),
            size,
            blit_source,
        }
    }

    /// The image the blit samples after the last [`Self::record`]: the sharpened
    /// output, or the history itself when sharpening was off.
    pub(crate) fn blit_source(&self) -> (vk::Image, vk::ImageView) {
        self.blit_source
    }

    /// Record both passes for `slot`. `jitter` is this frame's sub-pixel offset
    /// in texels, `reset` marks history as invalid. Returns the blit source.
    /// The caller has waited the slot's fence and recorded the scene barrier.
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        inputs: &TaaInputs,
        settings: &TaaSettings,
        jitter: [f32; 2],
        reset: bool,
    ) -> (vk::Image, vk::ImageView) {
        let reset = reset || !self.primed;
        self.primed = true;
        let (w, h) = self.size;
        let (wf, hf) = (w as f32, h as f32);
        let params = TaaParams {
            size: [wf, hf, 1.0 / wf, 1.0 / hf],
            jitter: [jitter[0], jitter[1], 0.0, 0.0],
            blend: [
                settings.history_weight,
                settings.variance_gamma,
                settings.sharpness,
                if settings.anti_flicker { 1.0 } else { 0.0 },
            ],
            flags: [if reset { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        let slots = self.params.len();
        let buffer = &mut self.params[slot % slots];
        buffer.write(0, bytemuck::bytes_of(&params));

        let write = self.write;
        let read = 1 - write;
        self.write = read;
        let slot_params = &self.params[slot % slots];

        // Resolve: scene inputs, previous history (read), new history (written).
        self.resolve.bind_uniform(slot, 0, slot_params, PARAMS_SIZE);
        self.resolve.bind_sampled(slot, 1, inputs.color);
        self.resolve.bind_sampled(slot, 2, inputs.depth);
        self.resolve.bind_sampled(slot, 3, inputs.velocity);
        self.resolve.bind_sampled(slot, 4, self.history[read].view);
        self.resolve.bind_sampler(slot, 5, self.sampler);
        self.resolve
            .bind_storage(slot, 6, self.history[write].view, HISTORY_FORMAT);
        self.resolve.dispatch(cmd, slot, w, h);

        if settings.sharpness <= 0.0 {
            self.blit_source = (self.history[write].image, self.history[write].view);
        } else {
            self.sharpen.bind_uniform(slot, 0, slot_params, PARAMS_SIZE);
            self.sharpen.bind_sampled(slot, 1, self.history[write].view);
            self.sharpen
                .bind_storage(slot, 2, self.output.view, HISTORY_FORMAT);
            self.sharpen.dispatch(cmd, slot, w, h);
            self.blit_source = (self.output.image, self.output.view);
        }
        self.blit_source
    }
}

impl Drop for Taa {
    fn drop(&mut self) {
        // The passes and images drop after this; the GPU must be idle first so
        // no in-flight frame still reads them.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.sampler, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taa_params_block_matches_wgsl_layout() {
        // Four vec4s, read identically by the resolve and sharpen shaders.
        assert_eq!(std::mem::size_of::<TaaParams>(), 64);
        assert_eq!(std::mem::offset_of!(TaaParams, jitter), 16);
        assert_eq!(std::mem::offset_of!(TaaParams, blend), 32);
        assert_eq!(std::mem::offset_of!(TaaParams, flags), 48);
    }
}

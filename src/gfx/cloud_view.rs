//! Per-pixel view clouds: a half-resolution staggered march, a full-resolution
//! reprojected history and a composite over the scene colour.
//!
//! Three passes per frame, built on [`PostPass`]:
//!
//! - **March** (`cloud_view_march.wgsl`) marches one half-resolution texel per
//!   frame, which stands for one full-resolution pixel. The offset of the
//!   sampled full-resolution pixels cycles through a 2 x 2 pattern, so each
//!   pixel is marched once every four frames.
//! - **Resolve** (`cloud_view_resolve.wgsl`) writes the full-resolution history
//!   (ping-ponged). Marched pixels blend their exact sample into reprojected
//!   history; unmarched pixels take a depth-class-aware neighbourhood estimate
//!   blended with clamped reprojected history.
//! - **Composite** (`cloud_view_composite.wgsl`) blends the full-resolution
//!   history over the scene colour, after the sky and before fog and
//!   transparent geometry. It writes depth and velocity never.
//!
//! The history is valid after one frame and is reset by a history reset, a
//! resize and when the clouds are switched on.

use super::SCENE_FORMAT;
use super::clouds::{ATMO_PARAMS_SIZE, CloudParams, MarchInputs, PARAMS_SIZE, compose};
use super::post::{BindKind, FullscreenTarget, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use crate::render::CloudSettings;
use crate::render::quality_params;
use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;

const MARCH_WGSL: &str = include_str!("cloud_view_march.wgsl");
const RESOLVE_WGSL: &str = include_str!("cloud_view_resolve.wgsl");
const COMPOSITE_WGSL: &str = include_str!("cloud_view_composite.wgsl");
/// Colour history and quarter-resolution radiance: rgb radiance, a transmittance.
const COLOUR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Front and back distances (metres along the ray, see `CloudHit`).
const DIST_FORMAT: vk::Format = vk::Format::R32G32_SFLOAT;
/// Size of the `ViewParams` block the three passes read.
const VIEW_PARAMS_SIZE: u64 = std::mem::size_of::<ViewParams>() as u64;

/// CPU mirror of `ViewParams` in `cloud_common.wgsl`: 240 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ViewParams {
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    cam: [f32; 4],
    fwd: [f32; 4],
    layer: [f32; 4],
    screen: [f32; 4],
    res: [f32; 4],
    state: [f32; 4],
    march: [f32; 4],
}

/// The per-frame view inputs of [`CloudView::record`], besides the cloud frame.
pub(crate) struct ViewFrame {
    /// Unjittered inverse of the camera's view-projection (clip to world).
    pub inv_vp: Mat4,
    /// Unjittered view-projection of the previous frame (world to clip).
    pub prev_vp: Mat4,
    /// Unit camera forward direction, world space.
    pub forward: Vec3,
}

/// The scene and atmosphere resources one record reads and writes.
pub(crate) struct ViewInputs<'a> {
    pub atmo: MarchInputs<'a>,
    pub base: vk::ImageView,
    pub detail: vk::ImageView,
    /// The repeat sampler of the noise volumes.
    pub sampler: vk::Sampler,
    /// Linear view depth of the scene (metres, R32F), read by the march and composite.
    pub depth: vk::ImageView,
    /// Scene colour: the composite blends onto it.
    pub colour: vk::ImageView,
    /// The scene viewport and scissor the composite draws in.
    pub region: (vk::Viewport, vk::Rect2D),
}

/// Per-pixel view cloud resources for one render size.
pub(crate) struct CloudView {
    ctx: Arc<GpuContext>,
    /// Render size the history is built for.
    size: (u32, u32),
    /// Half-resolution march grid (`ceil(size / 2)`).
    low: (u32, u32),
    low_colour: GpuImage,
    low_dist: GpuImage,
    /// Ping-ponged history at half resolution.
    hist_colour: [GpuImage; 2],
    hist_dist: [GpuImage; 2],
    /// Index of the history the last record wrote.
    last: usize,
    /// The history holds a previous frame's data.
    valid: bool,
    /// Frames recorded so far: the offset pattern of the next frame.
    frames: u64,
    params: Vec<GpuBuffer>,
    cloud_params: Vec<GpuBuffer>,
    march: PostPass,
    resolve: PostPass,
    composite: PostPass,
}

impl CloudView {
    /// Build the images and passes for a render target of `size` pixels.
    pub(crate) fn new(ctx: &Arc<GpuContext>, size: (u32, u32)) -> Self {
        let size = (size.0.max(1), size.1.max(1));
        let low = (size.0.div_ceil(2), size.1.div_ceil(2));
        let march = PostPass::new(
            ctx,
            "cloud view march",
            &compose(MARCH_WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Uniform,
                BindKind::Uniform,
                BindKind::Sampled3d,
                BindKind::Sampled3d,
                BindKind::Sampler,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampler,
                BindKind::Sampled,
                BindKind::Storage(COLOUR_FORMAT),
                BindKind::Storage(DIST_FORMAT),
            ],
        );
        let resolve = PostPass::new(
            ctx,
            "cloud view resolve",
            &compose(RESOLVE_WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Storage(COLOUR_FORMAT),
                BindKind::Storage(DIST_FORMAT),
                BindKind::Sampled,
            ],
        );
        let composite = PostPass::new_fullscreen(
            ctx,
            "cloud view composite",
            &compose(COMPOSITE_WGSL),
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampled,
            ],
            FullscreenTarget {
                colour: &[SCENE_FORMAT],
                blend: true,
                depth_equal: None,
            },
        );
        Self {
            ctx: Arc::clone(ctx),
            size,
            low,
            low_colour: create_general_image(ctx, "cloud view low colour", low, COLOUR_FORMAT),
            low_dist: create_general_image(ctx, "cloud view low distance", low, DIST_FORMAT),
            hist_colour: [
                create_general_image(ctx, "cloud view history colour 0", size, COLOUR_FORMAT),
                create_general_image(ctx, "cloud view history colour 1", size, COLOUR_FORMAT),
            ],
            hist_dist: [
                create_general_image(ctx, "cloud view history distance 0", size, DIST_FORMAT),
                create_general_image(ctx, "cloud view history distance 1", size, DIST_FORMAT),
            ],
            last: 1,
            valid: false,
            frames: 0,
            params: host_uniform_slots(ctx, "cloud view params", VIEW_PARAMS_SIZE),
            cloud_params: host_uniform_slots(ctx, "cloud view cloud params", PARAMS_SIZE),
            march,
            resolve,
            composite,
        }
    }

    /// The render size the history is built for.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Force the next record to reset history (e.g., on quality or settings changes).
    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Record the march, the resolve and the composite of this frame into `cmd`
    /// for frame `slot`. `cp` is the cloud parameter block (its layer and camera
    /// are the view's), `reset` invalidates the history.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        inputs: &ViewInputs<'_>,
        cp: &CloudParams,
        view: &ViewFrame,
        settings: &CloudSettings,
        reset: bool,
        mark: &mut dyn FnMut(&'static str),
    ) {
        let slots = self.params.len();
        let invalid = reset || !self.valid;
        let steps = quality_params(settings.quality).0 as f32;
        let block = ViewParams {
            inv_vp: view.inv_vp.to_cols_array_2d(),
            prev_vp: view.prev_vp.to_cols_array_2d(),
            cam: cp.cam,
            fwd: [view.forward.x, view.forward.y, view.forward.z, 0.0],
            layer: cp.layer,
            screen: [
                self.size.0 as f32,
                self.size.1 as f32,
                1.0 / self.size.0 as f32,
                1.0 / self.size.1 as f32,
            ],
            res: [
                self.size.0 as f32,
                self.size.1 as f32,
                self.low.0 as f32,
                self.low.1 as f32,
            ],
            state: [
                (self.frames % 4096) as f32,
                if invalid { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            march: [steps, 0.0, 0.0, 0.0],
        };
        self.params[slot % slots].write(0, bytemuck::bytes_of(&block));
        self.cloud_params[slot % slots].write(0, bytemuck::bytes_of(cp));
        self.frames += 1;

        let cur = 1 - self.last;
        let prev = self.last;
        let params = &self.params[slot % slots];
        let cloud = &self.cloud_params[slot % slots];

        let m = &self.march;
        m.bind_uniform(slot, 0, params, VIEW_PARAMS_SIZE);
        m.bind_uniform(slot, 1, cloud, PARAMS_SIZE);
        m.bind_uniform(slot, 2, inputs.atmo.params, ATMO_PARAMS_SIZE);
        m.bind_sampled(slot, 3, inputs.base);
        m.bind_sampled(slot, 4, inputs.detail);
        m.bind_sampler(slot, 5, inputs.sampler);
        m.bind_sampled(slot, 6, inputs.atmo.transmittance);
        m.bind_sampled(slot, 7, inputs.atmo.skyview);
        m.bind_sampler(slot, 8, inputs.atmo.lut_sampler);
        m.bind_sampled(slot, 9, inputs.depth);
        m.bind_storage(slot, 10, self.low_colour.view, COLOUR_FORMAT);
        m.bind_storage(slot, 11, self.low_dist.view, DIST_FORMAT);
        mark("cloud march");
        m.dispatch(cmd, slot, self.low.0, self.low.1);

        let r = &self.resolve;
        r.bind_uniform(slot, 0, params, VIEW_PARAMS_SIZE);
        r.bind_sampled(slot, 1, self.low_colour.view);
        r.bind_sampled(slot, 2, self.low_dist.view);
        r.bind_sampled(slot, 3, self.hist_colour[prev].view);
        r.bind_storage(slot, 4, self.hist_colour[cur].view, COLOUR_FORMAT);
        r.bind_storage(slot, 5, self.hist_dist[cur].view, DIST_FORMAT);
        r.bind_sampled(slot, 6, inputs.depth);
        mark("cloud resolve");
        r.dispatch(cmd, slot, self.size.0, self.size.1);

        let c = &self.composite;
        c.bind_uniform(slot, 0, params, VIEW_PARAMS_SIZE);
        c.bind_sampled(slot, 1, inputs.depth);
        c.bind_sampled(slot, 2, self.hist_colour[cur].view);
        c.bind_sampled(slot, 3, self.hist_dist[cur].view);
        mark("cloud composite");
        let (viewport, scissor) = inputs.region;
        let colour = vk::RenderingAttachmentInfo::default()
            .image_view(inputs.colour)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let colours = [colour];
        let info = vk::RenderingInfo::default()
            .render_area(scissor)
            .layer_count(1)
            .color_attachments(&colours);
        let device = self.ctx.device();
        unsafe {
            device.cmd_begin_rendering(cmd, &info);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
        }
        c.draw_fullscreen(cmd, slot);
        unsafe { device.cmd_end_rendering(cmd) };

        self.last = cur;
        self.valid = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_view_params_block_matches_wgsl_layout() {
        // Two mat4x4 and seven vec4s: 240 bytes, the `ViewParams` of cloud_common.wgsl.
        assert_eq!(VIEW_PARAMS_SIZE, 240);
        assert_eq!(std::mem::offset_of!(ViewParams, prev_vp), 64);
        assert_eq!(std::mem::offset_of!(ViewParams, state), 208);
        assert_eq!(std::mem::offset_of!(ViewParams, march), 224);
    }
}

//! Linear view depth of the scene, shared by the screen-space chains.
//!
//! One compute pass per frame, recorded after the scene pass. It writes the
//! linear view depth (metres) at the depth resolution, which is the render size
//! unless every chain that runs uses half resolution. The indirect chain (AO and
//! GI), the reflection trace and the lighting set (binding 7, the depth of the
//! frame whose results are bound) all read it. A chain at a coarser working
//! resolution maps each of its texels onto the depth texel at its top-left.
//!
//! The depth is ping-ponged: [`LinearDepth::now`] is this frame's, and
//! [`LinearDepth::prev`] the one before, for temporal rejection.

use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use ash::vk;
use std::sync::Arc;

const WGSL: &str = include_str!("linear_depth.wgsl");
/// Storage format of the linear depth. R16F is not a WGSL storage format.
const DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

/// CPU mirror of `DepthParams` in `linear_depth.wgsl` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DepthParams {
    /// Depth width, height, 1/width, 1/height.
    pub size: [f32; 4],
    /// Render width, height, 1/width, 1/height.
    pub full: [f32; 4],
    /// Near, far, tan(fov_y / 2), aspect.
    pub proj: [f32; 4],
}

impl DepthParams {
    pub(crate) fn new(
        size: (u32, u32),
        full: (u32, u32),
        near: f32,
        far: f32,
        fov_y: f32,
        aspect: f32,
    ) -> Self {
        let (dw, dh) = (size.0.max(1) as f32, size.1.max(1) as f32);
        let (fw, fh) = (full.0.max(1) as f32, full.1.max(1) as f32);
        Self {
            size: [dw, dh, 1.0 / dw, 1.0 / dh],
            full: [fw, fh, 1.0 / fw, 1.0 / fh],
            proj: [near, far, (fov_y * 0.5).tan(), aspect],
        }
    }
}

/// The shared linear depth a frame reads, and the one before it.
#[derive(Clone, Copy)]
pub(crate) struct DepthViews {
    pub now: vk::ImageView,
    pub prev: vk::ImageView,
}

/// The shared linear-depth images for one depth resolution.
pub(crate) struct LinearDepth {
    ctx: Arc<GpuContext>,
    prepare: PostPass,
    /// Ping-ponged: the current record writes `depth[1 - last]`.
    depth: [GpuImage; 2],
    params: Vec<GpuBuffer>,
    size: (u32, u32),
    half: bool,
    /// Index of the image the last record wrote.
    last: usize,
    recorded: bool,
}

impl LinearDepth {
    /// Build the pass and images for a `full` (render size) target, at half
    /// resolution per axis when `half`.
    pub(crate) fn new(ctx: &Arc<GpuContext>, full: (u32, u32), half: bool) -> Self {
        let full = (full.0.max(1), full.1.max(1));
        let size = if half {
            (full.0.div_ceil(2), full.1.div_ceil(2))
        } else {
            full
        };
        let kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Storage(DEPTH_FORMAT),
        ];
        Self {
            ctx: Arc::clone(ctx),
            prepare: PostPass::new(ctx, "linear depth", WGSL, c"linear_depth", &kinds),
            depth: [
                create_general_image(ctx, "linear depth 0", size, DEPTH_FORMAT),
                create_general_image(ctx, "linear depth 1", size, DEPTH_FORMAT),
            ],
            params: host_uniform_slots(
                ctx,
                "linear depth params",
                std::mem::size_of::<DepthParams>() as u64,
            ),
            size,
            half,
            last: 0,
            recorded: false,
        }
    }

    /// Depth resolution.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Whether the images were built at half resolution.
    pub(crate) fn half_res(&self) -> bool {
        self.half
    }

    /// Record the depth of this frame into the image not read as `prev`.
    /// `scene_depth` is the scene depth attachment (depth aspect) at render size.
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        scene_depth: vk::ImageView,
        params: &DepthParams,
    ) {
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(params));
        let block = &self.params[slot % slots];
        let size = std::mem::size_of::<DepthParams>() as u64;
        let w = 1 - self.last;
        self.prepare.bind_uniform(slot, 0, block, size);
        self.prepare.bind_sampled(slot, 1, scene_depth);
        self.prepare
            .bind_storage(slot, 2, self.depth[w].view, DEPTH_FORMAT);
        self.prepare.dispatch(cmd, slot, self.size.0, self.size.1);
        self.last = w;
        self.recorded = true;
    }

    /// The depth the last record wrote, or `None` before the first record.
    pub(crate) fn now(&self) -> Option<vk::ImageView> {
        self.recorded.then(|| self.depth[self.last].view)
    }

    /// The depth the record before the last wrote (valid after two records).
    pub(crate) fn prev(&self) -> vk::ImageView {
        self.depth[1 - self.last].view
    }
}

impl Drop for LinearDepth {
    fn drop(&mut self) {
        // In-flight frames may still read these through a lighting set or a chain.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
        }
    }
}

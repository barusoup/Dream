//! Screen-space ambient occlusion and one-bounce indirect diffuse light.
//!
//! Three compute passes per frame, built on [`PostPass`], recorded after the
//! scene pass of frame N and after the shared linear depth ([`super::linear_depth`]):
//!
//! 1. **Horizon** marches the depth, normals and lit colour of frame N and
//!    writes indirect radiance (rgb) and ambient visibility (a). The method is
//!    the visibility bitmask of Therrien et al. 2023.
//! 2. **Denoise** (when [`IndirectQuality::denoise`]) filters the raw result
//!    with depth-weighted neighbours.
//! 3. **Accumulate** (when denoising) blends it into a history reprojected with
//!    the scene velocity.
//!
//! The final image is what frame N+1's forward shader samples (see
//! `lighting_feedback` in `lighting.wgsl`), so ambient light is attenuated and
//! brightened by a frame-old result. Nothing here is multiplied over direct
//! light, and the forward pass of a frame never reads an image written in it.
//!
//! Working resolution is the render size, or half of it in each axis when
//! [`IndirectQuality::half_res`] is set. Working texel `q` describes the
//! full-resolution texel `q * scale` (its top-left in each 2x2 block), and the
//! depth texel `q * depth_ratio` of the shared linear depth.
//!
//! Visibility and light are weighted by the cosine lobe around the normal with
//! four 8-sector bands (see [`band_weight`]), so occluders overhead count more
//! than occluders near the tangent plane.

use super::linear_depth::DepthViews;
use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::rt::RtFrameBinds;
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use crate::render::{AoSettings, GiSettings, IndirectQuality};
use ash::vk;
use glam::Mat4;
use std::sync::Arc;

const HORIZON_WGSL: &str = include_str!("indirect_horizon.wgsl");
const DENOISE_WGSL: &str = include_str!("indirect_denoise.wgsl");
const ACCUMULATE_WGSL: &str = include_str!("indirect_accumulate.wgsl");
const RT_HORIZON_WGSL: &str = include_str!("rt_indirect.wgsl");
const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Weight of the current frame in the temporal blend when history is valid.
const TEMPORAL_WEIGHT: f32 = 0.1;
/// Relative depth difference beyond which reprojected history is rejected.
const TEMPORAL_DEPTH_TOLERANCE: f32 = 0.15;

/// CPU mirror of `IndirectParams` in the three `indirect_*.wgsl` files (176 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct IndirectParams {
    /// Working width, height, 1/width, 1/height.
    pub size: [f32; 4],
    /// Full-resolution width, height, 1/width, 1/height.
    pub full: [f32; 4],
    /// Near, far, tan(fov_y / 2), aspect.
    pub proj: [f32; 4],
    /// World to view rotation (unjittered camera).
    pub view_from_world: [[f32; 4]; 4],
    /// x: AO radius (m), y: thickness (m), z: AO on, w: unused.
    pub ao: [f32; 4],
    /// x: GI radius (m), y: GI on, z: slices, w: steps per side.
    pub gi: [f32; 4],
    /// x: frame index, y: history reset, z/w: depth texels per working texel.
    pub march: [f32; 4],
    /// x: weight of this frame, y: relative depth tolerance, z/w: unused.
    pub temporal: [f32; 4],
    pub world_from_view: [[f32; 4]; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    pub env_valid: [f32; 4],
}

/// Everything the camera and settings contribute to one frame's chain.
pub(crate) struct IndirectFrameSetup<'a> {
    pub view: Mat4,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub full: (u32, u32),
    pub ao: Option<&'a AoSettings>,
    pub gi: Option<&'a GiSettings>,
    pub quality: IndirectQuality,
    pub frame: u32,
    pub reset: bool,
    pub env: crate::lighting::LightingEnv,
    pub env_valid: bool,
}

impl IndirectParams {
    /// Build the block for one frame. `working` is the chain's resolution and
    /// `depth` the shared linear depth's (see [`super::linear_depth`]).
    pub(crate) fn new(
        setup: &IndirectFrameSetup<'_>,
        working: (u32, u32),
        depth: (u32, u32),
    ) -> Self {
        let (fw, fh) = (setup.full.0.max(1) as f32, setup.full.1.max(1) as f32);
        let (ww, wh) = (working.0.max(1) as f32, working.1.max(1) as f32);
        let (dw, dh) = (depth.0.max(1) as f32, depth.1.max(1) as f32);
        let quality = setup.quality.sanitised();
        let tan_half = (setup.fov_y * 0.5).tan();
        let ao = setup.ao.map(AoSettings::sanitised);
        let gi = setup.gi.map(GiSettings::sanitised);
        Self {
            size: [ww, wh, 1.0 / ww, 1.0 / wh],
            full: [fw, fh, 1.0 / fw, 1.0 / fh],
            proj: [setup.near, setup.far, tan_half, setup.aspect],
            view_from_world: setup.view.to_cols_array_2d(),
            ao: [
                ao.as_ref().map_or(0.0, |a| a.radius),
                ao.as_ref().map_or(0.0, |a| a.thickness),
                if ao.is_some() { 1.0 } else { 0.0 },
                0.0,
            ],
            gi: [
                gi.as_ref().map_or(0.0, |g| g.radius),
                if gi.is_some() { 1.0 } else { 0.0 },
                quality.slices as f32,
                quality.steps as f32,
            ],
            march: [
                (setup.frame % 65536) as f32,
                if setup.reset { 1.0 } else { 0.0 },
                dw / ww,
                dh / wh,
            ],
            temporal: [TEMPORAL_WEIGHT, TEMPORAL_DEPTH_TOLERANCE, 0.0, 0.0],
            world_from_view: setup.view.inverse().to_cols_array_2d(),
            sun_direction: setup.env.sun_direction,
            sun_color: setup.env.sun_color,
            sky_zenith: setup.env.sky_zenith,
            sky_horizon: setup.env.sky_horizon,
            sky_ground: setup.env.sky_ground,
            env_valid: [if setup.env_valid { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        }
    }
}

/// Scene targets the chain reads, as views in `GENERAL` layout.
pub(crate) struct IndirectInputs<'a> {
    pub colour: vk::ImageView,
    pub surface: vk::ImageView,
    pub velocity: vk::ImageView,
    pub rt: Option<RtFrameBinds<'a>>,
}

/// Indirect chain resources for one working size.
pub(crate) struct Indirect {
    ctx: Arc<GpuContext>,
    horizon: PostPass,
    rt_horizon: Option<PostPass>,
    denoise: PostPass,
    accumulate: PostPass,
    /// Raw horizon output.
    raw: GpuImage,
    /// Denoised horizon output.
    filtered: GpuImage,
    /// Accumulated result, ping-ponged: the current frame writes `history[write]`.
    history: [GpuImage; 2],
    sampler: vk::Sampler,
    params: Vec<GpuBuffer>,
    working: (u32, u32),
    half_res: bool,
    /// Index of the history image the next record writes.
    write: usize,
    /// False until a record has run; the first record resets history.
    primed: bool,
    /// The last record's gi/ao view, for the next frame's lighting.
    output: Option<vk::ImageView>,
    /// The image behind `output`, for diagnostics readback.
    output_image: Option<vk::Image>,
}

impl Indirect {
    /// Build the passes and images for a `full` (render size) target. Working
    /// resolution is halved in each axis when `half_res`.
    pub(crate) fn new(ctx: &Arc<GpuContext>, full: (u32, u32), half_res: bool) -> Self {
        let full = (full.0.max(1), full.1.max(1));
        let working = if half_res {
            (full.0.div_ceil(2), full.1.div_ceil(2))
        } else {
            full
        };
        let horizon_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Storage(TARGET_FORMAT),
        ];
        let denoise_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Storage(TARGET_FORMAT),
            BindKind::Sampled,
        ];
        let accumulate_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampler,
            BindKind::Storage(TARGET_FORMAT),
        ];
        Self {
            ctx: Arc::clone(ctx),
            horizon: PostPass::new(
                ctx,
                "indirect horizon",
                HORIZON_WGSL,
                c"horizon",
                &horizon_kinds,
            ),
            rt_horizon: None,
            denoise: PostPass::new(
                ctx,
                "indirect denoise",
                DENOISE_WGSL,
                c"denoise",
                &denoise_kinds,
            ),
            accumulate: PostPass::new(
                ctx,
                "indirect accumulate",
                ACCUMULATE_WGSL,
                c"accumulate",
                &accumulate_kinds,
            ),
            raw: create_general_image(ctx, "indirect raw", working, TARGET_FORMAT),
            filtered: create_general_image(ctx, "indirect filtered", working, TARGET_FORMAT),
            history: [
                create_general_image(ctx, "indirect history 0", working, TARGET_FORMAT),
                create_general_image(ctx, "indirect history 1", working, TARGET_FORMAT),
            ],
            sampler: ctx.create_sampler("indirect history sampler"),
            params: host_uniform_slots(
                ctx,
                "indirect params",
                std::mem::size_of::<IndirectParams>() as u64,
            ),
            working,
            half_res,
            write: 0,
            primed: false,
            output: None,
            output_image: None,
        }
    }

    /// Whether the chain was built at half resolution.
    pub(crate) fn half_res(&self) -> bool {
        self.half_res
    }

    /// Force the next record to reset temporal history (e.g., when depth history was rebuilt).
    pub(crate) fn invalidate_history(&mut self) {
        self.primed = false;
    }

    /// The image of the last record's gi/ao output (diagnostics readback).
    pub(crate) fn output_image(&self) -> Option<vk::Image> {
        self.output_image
    }

    /// Working resolution of the chain.
    pub(crate) fn working_size(&self) -> (u32, u32) {
        self.working
    }

    /// The gi/ao view the last record produced, or `None` before the first
    /// record. Bound to the lighting set for the next frame.
    pub(crate) fn output(&self) -> Option<vk::ImageView> {
        self.output
    }

    /// Record the chain for `slot`. The caller has waited the slot's fence,
    /// recorded the scene barriers and recorded the shared linear depth.
    /// `params` comes from [`IndirectParams::new`].
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        inputs: &IndirectInputs<'_>,
        depth: DepthViews,
        params: &IndirectParams,
        denoise: bool,
    ) {
        let rt = inputs.rt.as_ref();
        let mut params = *params;
        if !self.primed {
            params.march[1] = 1.0;
        }
        let (ww, wh) = self.working;
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(&params));
        let block = &self.params[slot % slots];
        let size = std::mem::size_of::<IndirectParams>() as u64;

        let w = self.write;
        let r = 1 - w;
        self.write = r;
        self.primed = true;

        // 1. Horizon march or hardware ray query.
        if let Some(rt) = rt {
            if self.rt_horizon.is_none() {
                let kinds = [
                    BindKind::Uniform,
                    BindKind::Sampled,
                    BindKind::Sampled,
                    BindKind::Storage(TARGET_FORMAT),
                    BindKind::Accel,
                    BindKind::StorageBuffer,
                    BindKind::Sampled,
                    BindKind::Sampler,
                ];
                self.rt_horizon = Some(PostPass::new(
                    &self.ctx,
                    "ray traced ambient",
                    RT_HORIZON_WGSL,
                    c"trace",
                    &kinds,
                ));
            }
            let pass = self.rt_horizon.as_ref().expect("created RT ambient pass");
            pass.bind_uniform(slot, 0, block, size);
            pass.bind_sampled(slot, 1, depth.now);
            pass.bind_sampled(slot, 2, inputs.surface);
            pass.bind_storage(slot, 3, self.raw.view, TARGET_FORMAT);
            pass.bind_accel(slot, 4, rt.tlas);
            pass.bind_storage_buffer(slot, 5, rt.materials);
            pass.bind_sampled(slot, 6, rt.env_view);
            pass.bind_sampler(slot, 7, rt.env_sampler);
            pass.dispatch(cmd, slot, ww, wh);
        } else {
            self.horizon.bind_uniform(slot, 0, block, size);
            self.horizon.bind_sampled(slot, 1, depth.now);
            self.horizon.bind_sampled(slot, 2, inputs.surface);
            self.horizon.bind_sampled(slot, 3, inputs.colour);
            self.horizon
                .bind_storage(slot, 4, self.raw.view, TARGET_FORMAT);
            self.horizon.dispatch(cmd, slot, ww, wh);
        }

        let (output_view, output_image) = if denoise {
            // 2. Spatial filter.
            self.denoise.bind_uniform(slot, 0, block, size);
            self.denoise.bind_sampled(slot, 1, self.raw.view);
            self.denoise.bind_sampled(slot, 2, depth.now);
            self.denoise
                .bind_storage(slot, 3, self.filtered.view, TARGET_FORMAT);
            self.denoise.bind_sampled(slot, 4, inputs.surface);
            self.denoise.dispatch(cmd, slot, ww, wh);

            // 3. Temporal accumulation: history[w] from history[r].
            self.accumulate.bind_uniform(slot, 0, block, size);
            self.accumulate.bind_sampled(slot, 1, self.filtered.view);
            self.accumulate.bind_sampled(slot, 2, depth.now);
            self.accumulate.bind_sampled(slot, 3, inputs.velocity);
            self.accumulate.bind_sampled(slot, 4, self.history[r].view);
            self.accumulate.bind_sampled(slot, 5, depth.prev);
            self.accumulate.bind_sampler(slot, 6, self.sampler);
            self.accumulate
                .bind_storage(slot, 7, self.history[w].view, TARGET_FORMAT);
            self.accumulate.dispatch(cmd, slot, ww, wh);
            (self.history[w].view, self.history[w].image)
        } else {
            (self.raw.view, self.raw.image)
        };
        self.output = Some(output_view);
        self.output_image = Some(output_image);
    }
}

impl Drop for Indirect {
    fn drop(&mut self) {
        // The GPU must be idle before the images and sampler go: an in-flight
        // frame may still read them through its lighting set.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.sampler, None);
        }
    }
}

/// Cosine-lobe weight of the sectors set in a 32-sector bitmask, as a fraction
/// of the whole hemisphere (1.0 when every sector is set). Sector `i` spans the
/// angles `-90° + i * 5.625°` to `+5.625°` further, measured from the projected
/// normal of the slice, so sector 0 is at the grazing angle on one side and
/// sector 31 on the other. The sectors form four 8-sector bands of 45° each; a
/// band's weight is the integral of cos over it divided by 2 (the integral over
/// the whole hemisphere): 0.14645, 0.35355, 0.35355, 0.14645. The weights are
/// symmetric, so the side does not matter. Matches `band_weight` in
/// `indirect_horizon.wgsl`.
#[cfg(test)]
pub(crate) fn band_weight(bits: u32) -> f32 {
    const WEIGHTS: [f32; 4] = [0.146_446_6, 0.353_553_4, 0.353_553_4, 0.146_446_6];
    (0..4)
        .map(|band| {
            let mask = 0xFFu32 << (8 * band);
            (bits & mask).count_ones() as f32 / 8.0 * WEIGHTS[band as usize]
        })
        .sum()
}

/// The 32-sector bitmask of `[lo, hi]` (fractions of the hemisphere). Sector
/// `i` covers `[i/32, (i+1)/32)`. Matches `sector_mask` in `indirect_horizon.wgsl`.
#[cfg(test)]
pub(crate) fn sector_mask(lo: f32, hi: f32) -> u32 {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    // Nearest-boundary rounding at both ends (see the shader): rounding the
    // span up would give every sample at least one sector.
    let start = (lo * 32.0 + 0.5).floor().clamp(0.0, 32.0) as u32;
    let end = (hi * 32.0 + 0.5).floor().clamp(0.0, 32.0) as u32;
    if end <= start {
        return 0;
    }
    let count = end - start;
    (0xFFFF_FFFFu32 >> (32 - count)) << start
}

/// Occluded sectors of one arc, weighted by distance. Matches
/// `arc_weight_mask` in `indirect_horizon.wgsl`: sector `k` is occluded when its
/// threshold `fract(seed + k * golden)` is below the ray weight
/// `1 - (d / radius)^2`, with `d` interpolated along the arc from `d_front` at
/// `h_front` to `d_back` at `h_back`.
#[cfg(test)]
pub(crate) fn arc_weight_mask(
    h_front: f32,
    h_back: f32,
    d_front: f32,
    d_back: f32,
    radius: f32,
    seed: f32,
) -> u32 {
    let lo = h_front.min(h_back);
    let hi = h_front.max(h_back);
    let start = (lo * 32.0 + 0.5).floor().clamp(0.0, 32.0) as u32;
    let end = (hi * 32.0 + 0.5).floor().clamp(0.0, 32.0) as u32;
    let span = h_back - h_front;
    let mut bits = 0u32;
    for k in start..end {
        let along = if span.abs() < 1e-6 {
            0.0
        } else {
            ((k as f32 + 0.5) / 32.0 - h_front) / span
        }
        .clamp(0.0, 1.0);
        let d = (d_front + (d_back - d_front) * along) / radius;
        let weight = (1.0 - d * d).clamp(0.0, 1.0);
        let threshold = seed + k as f32 * 0.618_033_95;
        if threshold - threshold.floor() < weight {
            bits |= 1 << k;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_params_block_matches_wgsl_layout() {
        // The original indirect WGSL passes read the 176-byte prefix; the RT
        // pass appends world-space and environment data after that prefix.
        assert_eq!(std::mem::size_of::<IndirectParams>(), 336);
        assert_eq!(std::mem::offset_of!(IndirectParams, full), 16);
        assert_eq!(std::mem::offset_of!(IndirectParams, proj), 32);
        assert_eq!(std::mem::offset_of!(IndirectParams, view_from_world), 48);
        assert_eq!(std::mem::offset_of!(IndirectParams, ao), 112);
        assert_eq!(std::mem::offset_of!(IndirectParams, gi), 128);
        assert_eq!(std::mem::offset_of!(IndirectParams, march), 144);
        assert_eq!(std::mem::offset_of!(IndirectParams, temporal), 160);
    }

    #[test]
    fn indirect_arc_weight_mask_full_weight_is_the_sector_mask() {
        for (lo, hi) in [(0.0, 0.5), (0.25, 0.75), (0.1, 0.9), (0.4, 0.4)] {
            assert_eq!(
                arc_weight_mask(lo, hi, 0.0, 0.0, 1.0, 0.3),
                sector_mask(lo, hi)
            );
        }
    }

    #[test]
    fn indirect_arc_weight_mask_beyond_the_radius_is_empty() {
        assert_eq!(arc_weight_mask(0.0, 1.0, 1.5, 1.5, 1.5, 0.2), 0);
    }

    #[test]
    fn indirect_arc_weight_mask_occupies_sectors_in_proportion_to_weight() {
        // Half the hemisphere at distance 0.5 radius: weight 0.75 per sector,
        // so 12 of the 16 sectors on average over the per-pixel seeds.
        let seeds = 1000;
        let total: u32 = (0..seeds)
            .map(|i| arc_weight_mask(0.0, 0.5, 0.5, 0.5, 1.0, i as f32 / seeds as f32).count_ones())
            .sum();
        let mean = total as f32 / seeds as f32;
        assert!((mean - 12.0).abs() < 0.3, "mean occupied sectors {mean}");
    }

    #[test]
    fn indirect_arc_weight_masks_union_takes_the_largest_weight() {
        // Two arcs share one seed. Where they overlap, the union occludes exactly
        // what the heavier arc occludes: the nearest occluder decides the weight.
        let seed = 0.37;
        let faint = arc_weight_mask(0.0, 0.5, 0.9, 0.9, 1.0, seed);
        let heavy = arc_weight_mask(0.25, 0.75, 0.4, 0.4, 1.0, seed);
        let overlap = sector_mask(0.25, 0.5);
        assert_eq!(
            (faint | heavy) & overlap,
            heavy & overlap,
            "a lighter arc must not add sectors inside a heavier one"
        );
    }

    #[test]
    fn indirect_sector_mask_matches_hand_computed_cases() {
        // Lower half of the hemisphere: sectors 0..16.
        assert_eq!(sector_mask(0.0, 0.5), 0x0000_FFFF);
        // Upper half.
        assert_eq!(sector_mask(0.5, 1.0), 0xFFFF_0000);
        // Centre band: sectors 8..24.
        assert_eq!(sector_mask(0.25, 0.75), 0x00FF_FF00);
        // Whole hemisphere.
        assert_eq!(sector_mask(0.0, 1.0), 0xFFFF_FFFF);
        // One sector, first and last.
        assert_eq!(sector_mask(0.0, 1.0 / 32.0), 0x1);
        assert_eq!(sector_mask(31.0 / 32.0, 1.0), 0x8000_0000);
        // Empty and degenerate intervals.
        assert_eq!(sector_mask(0.4, 0.4), 0);
        assert_eq!(sector_mask(1.0, 1.0), 0);
        // Reversed endpoints mark the same sectors.
        assert_eq!(sector_mask(0.75, 0.25), sector_mask(0.25, 0.75));
    }

    #[test]
    fn indirect_band_weights_sum_to_one() {
        // Each band alone, then all four: the weights partition the hemisphere.
        let band_sum: f32 = (0..4).map(|b| band_weight(0xFFu32 << (8 * b))).sum();
        assert!((band_sum - 1.0).abs() < 1e-5, "band sum {band_sum}");
        assert!((band_weight(0xFFFF_FFFF) - 1.0).abs() < 1e-6);
        // The outer bands carry less cosine weight than the inner ones.
        assert!(band_weight(0xFF) < band_weight(0xFF00));
        assert_eq!(band_weight(0), 0.0);
    }

    #[test]
    fn indirect_band_weight_of_a_full_mask_is_one() {
        // Every sector set is the whole hemisphere; a half covers the inner bands.
        assert!((band_weight(sector_mask(0.0, 1.0)) - 1.0).abs() < 1e-6);
        let upper_half = band_weight(sector_mask(0.5, 1.0));
        assert!((upper_half - 0.5).abs() < 1e-5, "upper half {upper_half}");
    }

    #[test]
    fn indirect_params_follow_the_settings() {
        let setup = IndirectFrameSetup {
            view: Mat4::IDENTITY,
            near: 0.1,
            far: 100.0,
            fov_y: 1.0,
            aspect: 2.0,
            full: (1280, 720),
            ao: None,
            gi: None,
            quality: IndirectQuality::low(),
            frame: 70000,
            reset: true,
            env: crate::lighting::LightingEnv::default(),
            env_valid: false,
        };
        let p = IndirectParams::new(&setup, (640, 360), (1280, 720));
        // Both features off: their enable flags are 0 and the march runs anyway.
        assert_eq!(p.ao[2], 0.0);
        assert_eq!(p.gi[1], 0.0);
        assert_eq!(p.gi[2], 1.0);
        assert_eq!(p.gi[3], 4.0);
        assert_eq!(p.march[1], 1.0);
        assert_eq!(p.march[0], (70000 % 65536) as f32);
        assert_eq!(p.size[..2], [640.0, 360.0]);
        // Half-resolution working texels read the full-resolution depth at 2x.
        assert_eq!(p.march[2..], [2.0, 2.0]);
        let ao = AoSettings::default();
        let gi = GiSettings::default();
        let setup = IndirectFrameSetup {
            ao: Some(&ao),
            gi: Some(&gi),
            reset: false,
            ..setup
        };
        let p = IndirectParams::new(&setup, (1280, 720), (1280, 720));
        assert_eq!(p.ao[2], 1.0);
        assert_eq!(p.ao[0], 1.5);
        assert_eq!(p.gi[0], 6.0);
        assert_eq!(p.gi[1], 1.0);
        assert_eq!(p.march[1], 0.0);
        assert_eq!(p.march[2..], [1.0, 1.0]);
    }
}

//! Screen-space reflections: traced against the shared linear depth, filtered,
//! and bound to the lighting set for the next frame.
//!
//! Three compute passes per frame, recorded after the scene pass of frame N and
//! after the shared linear depth ([`super::linear_depth`]):
//!
//! 1. **Trace** marches a reflection ray per working texel through the depth of
//!    frame N and gathers the lit colour of frame N at the hit (see
//!    `reflection_trace.wgsl`).
//! 2. **Temporal** (when [`ReflectionSettings::denoise`]) reprojects the previous
//!    accumulated result and blends the trace into it.
//! 3. **Spatial** (when denoising) filters the accumulated result with a
//!    roughness-dependent radius.
//!
//! Frame N+1's forward shader samples the output through the shared reprojection
//! (`lighting_reflection_at` in `lighting.wgsl`) and replaces the analytic sky
//! reflection with it where the confidence is high. Inter-reflections build up
//! over frames because each trace gathers the previous frame's lit colour.

use super::linear_depth::DepthViews;
use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::rt::RtFrameBinds;
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use crate::render::ReflectionSettings;
use ash::vk;
use glam::Mat4;
use std::sync::Arc;

const TRACE_WGSL: &str = include_str!("reflection_trace.wgsl");
const TEMPORAL_WGSL: &str = include_str!("reflection_temporal.wgsl");
const SPATIAL_WGSL: &str = include_str!("reflection_spatial.wgsl");
const RT_TRACE_WGSL: &str = include_str!("rt_reflection.wgsl");
const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Weight of the current trace in the temporal blend when history is valid.
const TEMPORAL_WEIGHT: f32 = 0.12;
/// Relative depth difference beyond which reprojected history is rejected.
const TEMPORAL_DEPTH_TOLERANCE: f32 = 0.15;

/// CPU mirror of `ReflectionParams` in the three `reflection_*.wgsl` files
/// (192 bytes). Every block is a `vec4`, so the layout has no padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ReflectionParams {
    /// Working width, height, 1/width, 1/height.
    pub size: [f32; 4],
    /// Full-resolution width, height, 1/width, 1/height.
    pub full: [f32; 4],
    /// Linear depth width, height, 1/width, 1/height.
    pub depth: [f32; 4],
    /// Near, far, tan(fov_y / 2), aspect.
    pub proj: [f32; 4],
    /// World to view rotation (unjittered camera).
    pub view_from_world: [[f32; 4]; 4],
    /// x: max distance (m), y: thickness (m), z: steps, w: refinement steps.
    pub trace: [f32; 4],
    /// x: max roughness, y: edge fade, z: intensity, w: unused.
    pub surface: [f32; 4],
    /// x: frame index mod 64 (0 unless noise animates), y: history reset, z/w: unused.
    pub frame: [f32; 4],
    /// x: current weight, y: relative depth tolerance, z/w: unused.
    pub temporal: [f32; 4],
    /// World-from-view transform, used by the hardware ray-query path.
    pub world_from_view: [[f32; 4]; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    pub env_valid: [f32; 4],
}

/// Everything the camera and settings contribute to one frame's reflections.
pub(crate) struct ReflectionFrameSetup {
    pub view: Mat4,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub full: (u32, u32),
    pub settings: ReflectionSettings,
    /// Whether the noise may animate: true when temporal filtering or temporal
    /// AA integrates it.
    pub animated: bool,
    pub frame: u32,
    pub reset: bool,
    pub env: crate::lighting::LightingEnv,
    pub env_valid: bool,
}

impl ReflectionParams {
    /// Build the block for one frame. `working` is the chain's resolution and
    /// `depth` the shared linear depth's.
    pub(crate) fn new(
        setup: &ReflectionFrameSetup,
        working: (u32, u32),
        depth: (u32, u32),
    ) -> Self {
        let (fw, fh) = (setup.full.0.max(1) as f32, setup.full.1.max(1) as f32);
        let (ww, wh) = (working.0.max(1) as f32, working.1.max(1) as f32);
        let (dw, dh) = (depth.0.max(1) as f32, depth.1.max(1) as f32);
        let s = &setup.settings;
        Self {
            size: [ww, wh, 1.0 / ww, 1.0 / wh],
            full: [fw, fh, 1.0 / fw, 1.0 / fh],
            depth: [dw, dh, 1.0 / dw, 1.0 / dh],
            proj: [
                setup.near,
                setup.far,
                (setup.fov_y * 0.5).tan(),
                setup.aspect,
            ],
            view_from_world: setup.view.to_cols_array_2d(),
            trace: [
                s.max_distance,
                s.thickness,
                s.steps as f32,
                s.refine_steps as f32,
            ],
            surface: [s.max_roughness, s.edge_fade, s.intensity, 0.0],
            frame: [
                if setup.animated {
                    (setup.frame % 64) as f32
                } else {
                    0.0
                },
                if setup.reset { 1.0 } else { 0.0 },
                0.0,
                0.0,
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

/// Scene targets the reflections read, as views in `GENERAL` layout.
pub(crate) struct ReflectionInputs<'a> {
    pub colour: vk::ImageView,
    pub surface: vk::ImageView,
    pub velocity: vk::ImageView,
    pub rt: Option<RtFrameBinds<'a>>,
}

/// Reflection chain resources for one working size.
pub(crate) struct Reflection {
    ctx: Arc<GpuContext>,
    trace: PostPass,
    rt_trace: Option<PostPass>,
    temporal: PostPass,
    spatial: PostPass,
    /// This frame's trace.
    raw: GpuImage,
    /// Temporal output, ping-ponged: the current frame writes `history[write]`.
    history: [GpuImage; 2],
    /// Spatially filtered output.
    filtered: GpuImage,
    sampler: vk::Sampler,
    params: Vec<GpuBuffer>,
    working: (u32, u32),
    half_res: bool,
    write: usize,
    primed: bool,
    output: Option<vk::ImageView>,
    output_image: Option<vk::Image>,
}

impl Reflection {
    /// Build the passes and images for a `full` (render size) target. Working
    /// resolution is halved in each axis when `half_res`.
    pub(crate) fn new(ctx: &Arc<GpuContext>, full: (u32, u32), half_res: bool) -> Self {
        let full = (full.0.max(1), full.1.max(1));
        let working = if half_res {
            (full.0.div_ceil(2), full.1.div_ceil(2))
        } else {
            full
        };
        let trace_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Storage(TARGET_FORMAT),
        ];
        let temporal_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampler,
            BindKind::Storage(TARGET_FORMAT),
        ];
        let spatial_kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Sampled,
            BindKind::Storage(TARGET_FORMAT),
        ];
        Self {
            ctx: Arc::clone(ctx),
            trace: PostPass::new(ctx, "reflection trace", TRACE_WGSL, c"trace", &trace_kinds),
            rt_trace: None,
            temporal: PostPass::new(
                ctx,
                "reflection temporal",
                TEMPORAL_WGSL,
                c"temporal",
                &temporal_kinds,
            ),
            spatial: PostPass::new(
                ctx,
                "reflection spatial",
                SPATIAL_WGSL,
                c"spatial",
                &spatial_kinds,
            ),
            raw: create_general_image(ctx, "reflection raw", working, TARGET_FORMAT),
            history: [
                create_general_image(ctx, "reflection history 0", working, TARGET_FORMAT),
                create_general_image(ctx, "reflection history 1", working, TARGET_FORMAT),
            ],
            filtered: create_general_image(ctx, "reflection filtered", working, TARGET_FORMAT),
            sampler: ctx.create_sampler("reflection history sampler"),
            params: host_uniform_slots(
                ctx,
                "reflection params",
                std::mem::size_of::<ReflectionParams>() as u64,
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

    /// Working resolution of the chain.
    pub(crate) fn working_size(&self) -> (u32, u32) {
        self.working
    }

    /// The image behind the last record's output (diagnostics readback).
    pub(crate) fn output_image(&self) -> Option<vk::Image> {
        self.output_image
    }

    /// The view the last record produced, or `None` before the first record.
    /// Bound to the lighting set for the next frame.
    pub(crate) fn output(&self) -> Option<vk::ImageView> {
        self.output
    }

    /// Record the chain for `slot`, after the shared linear depth.
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        inputs: &ReflectionInputs<'_>,
        depth: DepthViews,
        params: &ReflectionParams,
        denoise: bool,
    ) {
        let rt = inputs.rt.as_ref();
        let mut params = *params;
        if !self.primed {
            params.frame[1] = 1.0;
        }
        let (ww, wh) = self.working;
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(&params));
        let block = &self.params[slot % slots];
        let size = std::mem::size_of::<ReflectionParams>() as u64;

        let w = self.write;
        let r = 1 - w;
        self.write = r;
        self.primed = true;

        // 1. Trace.
        if let Some(rt) = rt {
            if self.rt_trace.is_none() {
                let kinds = [
                    BindKind::Uniform,
                    BindKind::Sampled,
                    BindKind::Sampled,
                    BindKind::Sampled,
                    BindKind::Storage(TARGET_FORMAT),
                    BindKind::Accel,
                    BindKind::StorageBuffer,
                    BindKind::Sampled,
                    BindKind::Sampler,
                ];
                self.rt_trace = Some(PostPass::new(
                    &self.ctx,
                    "ray traced reflections",
                    RT_TRACE_WGSL,
                    c"trace",
                    &kinds,
                ));
            }
            let trace = self.rt_trace.as_ref().expect("created RT trace pass");
            trace.bind_uniform(slot, 0, block, size);
            trace.bind_sampled(slot, 1, depth.now);
            trace.bind_sampled(slot, 2, inputs.surface);
            trace.bind_sampled(slot, 3, inputs.colour);
            trace.bind_storage(slot, 4, self.raw.view, TARGET_FORMAT);
            trace.bind_accel(slot, 5, rt.tlas);
            trace.bind_storage_buffer(slot, 6, rt.materials);
            trace.bind_sampled(slot, 7, rt.env_view);
            trace.bind_sampler(slot, 8, rt.env_sampler);
            trace.dispatch(cmd, slot, ww, wh);
        } else {
            self.trace.bind_uniform(slot, 0, block, size);
            self.trace.bind_sampled(slot, 1, depth.now);
            self.trace.bind_sampled(slot, 2, inputs.surface);
            self.trace.bind_sampled(slot, 3, inputs.colour);
            self.trace
                .bind_storage(slot, 4, self.raw.view, TARGET_FORMAT);
            self.trace.dispatch(cmd, slot, ww, wh);
        }

        let (output, output_image) = if denoise {
            // 2. Temporal: history[w] from this trace and history[r].
            self.temporal.bind_uniform(slot, 0, block, size);
            self.temporal.bind_sampled(slot, 1, self.raw.view);
            self.temporal.bind_sampled(slot, 2, depth.now);
            self.temporal.bind_sampled(slot, 3, inputs.velocity);
            self.temporal.bind_sampled(slot, 4, self.history[r].view);
            self.temporal.bind_sampled(slot, 5, depth.prev);
            self.temporal.bind_sampler(slot, 6, self.sampler);
            self.temporal
                .bind_storage(slot, 7, self.history[w].view, TARGET_FORMAT);
            self.temporal.dispatch(cmd, slot, ww, wh);

            // 3. Spatial filter of the accumulated result.
            self.spatial.bind_uniform(slot, 0, block, size);
            self.spatial.bind_sampled(slot, 1, self.history[w].view);
            self.spatial.bind_sampled(slot, 2, depth.now);
            self.spatial.bind_sampled(slot, 3, inputs.surface);
            self.spatial
                .bind_storage(slot, 4, self.filtered.view, TARGET_FORMAT);
            self.spatial.dispatch(cmd, slot, ww, wh);
            (self.filtered.view, self.filtered.image)
        } else {
            (self.raw.view, self.raw.image)
        };
        self.output = Some(output);
        self.output_image = Some(output_image);
    }
}

impl Drop for Reflection {
    fn drop(&mut self) {
        // The GPU must be idle before the images and sampler go: an in-flight
        // frame may still read them through its lighting set.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.sampler, None);
        }
    }
}

/// Orthonormal basis around the unit vector `n` (Duff et al. 2017). Mirror of
/// the basis in `reflection_direction` in `reflection_trace.wgsl`.
#[cfg(test)]
fn basis(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        glam::Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        glam::Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// CPU port of the visible-normal sample of `reflection_direction`: the GGX
/// half vector (world space, unit length) for view vector `v` (towards the
/// viewer) about `n` with `alpha` = roughness^2 and random numbers `u1`, `u2`.
#[cfg(test)]
pub(crate) fn vndf_half(v: glam::Vec3, n: glam::Vec3, alpha: f32, u1: f32, u2: f32) -> glam::Vec3 {
    use std::f32::consts::TAU;
    let (tb, bb) = basis(n);
    let ve = glam::Vec3::new(v.dot(tb), v.dot(bb), v.dot(n));
    let vh = glam::Vec3::new(alpha * ve.x, alpha * ve.y, ve.z).normalize();
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        glam::Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        glam::Vec3::X
    };
    let t2 = vh.cross(t1);
    let u1 = u1.clamp(0.075, 0.925);
    let u2 = u2.clamp(0.075, 0.925);
    let r = u1.sqrt();
    let phi = TAU * u2;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let hs = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - hs) * (1.0 - p1 * p1).sqrt() + hs * p2;
    let p3 = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = p1 * t1 + p2 * t2 + p3 * vh;
    let he = glam::Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize();
    (he.x * tb + he.y * bb + he.z * n).normalize()
}

/// CPU port of `reflection_direction`: the sampled reflection direction, mirror
/// of `v` about the visible-normal half vector, falling back to the mirror about
/// `n` when that lies below the surface. Unit length, and above the surface.
#[cfg(test)]
pub(crate) fn reflection_direction(
    v: glam::Vec3,
    n: glam::Vec3,
    alpha: f32,
    u1: f32,
    u2: f32,
) -> glam::Vec3 {
    let h = vndf_half(v, n, alpha, u1, u2);
    let dir = -v - 2.0 * (-v).dot(h) * h;
    if dir.dot(n) <= 0.0 {
        let mirror = -v - 2.0 * (-v).dot(n) * n;
        return mirror.normalize();
    }
    dir.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn reflection_params_block_matches_wgsl_layout() {
        // Mirrors `ReflectionParams` in the three reflection_*.wgsl files.
        assert_eq!(std::mem::size_of::<ReflectionParams>(), 352);
        assert_eq!(std::mem::offset_of!(ReflectionParams, depth), 32);
        assert_eq!(std::mem::offset_of!(ReflectionParams, proj), 48);
        assert_eq!(std::mem::offset_of!(ReflectionParams, view_from_world), 64);
        assert_eq!(std::mem::offset_of!(ReflectionParams, trace), 128);
        assert_eq!(std::mem::offset_of!(ReflectionParams, surface), 144);
        assert_eq!(std::mem::offset_of!(ReflectionParams, frame), 160);
        assert_eq!(std::mem::offset_of!(ReflectionParams, temporal), 176);
    }

    fn setup(settings: ReflectionSettings, animated: bool) -> ReflectionFrameSetup {
        ReflectionFrameSetup {
            view: Mat4::IDENTITY,
            near: 0.1,
            far: 100.0,
            fov_y: 1.0,
            aspect: 2.0,
            full: (1280, 720),
            settings,
            animated,
            frame: 65 + 3,
            reset: false,
            env: crate::lighting::LightingEnv::default(),
            env_valid: false,
        }
    }

    #[test]
    fn reflection_params_follow_the_settings() {
        let p = ReflectionParams::new(
            &setup(ReflectionSettings::medium(), true),
            (640, 360),
            (1280, 720),
        );
        assert_eq!(p.size[..2], [640.0, 360.0]);
        assert_eq!(p.depth[..2], [1280.0, 720.0]);
        assert_eq!(p.trace, [40.0, 0.35, 32.0, 4.0]);
        assert_eq!(p.surface[..3], [0.6, 0.1, 1.0]);
        // The animated frame index wraps at 64, as the other chains' does.
        assert_eq!(p.frame[0], 4.0);
        let still = ReflectionParams::new(
            &setup(ReflectionSettings::medium(), false),
            (640, 360),
            (1280, 720),
        );
        assert_eq!(still.frame[0], 0.0);
    }

    #[test]
    fn reflection_vndf_half_is_unit_length_over_a_grid() {
        let normals = [
            Vec3::Z,
            Vec3::new(0.3, 0.2, 0.93).normalize(),
            Vec3::new(-0.6, 0.5, 0.62).normalize(),
            Vec3::new(0.0, -0.1, -1.0).normalize(),
        ];
        let views = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.1, 0.7).normalize(),
            Vec3::new(-0.9, 0.2, 0.05).normalize(),
        ];
        let mut checked = 0;
        for n in normals {
            for v in views {
                for alpha in [0.01f32, 0.1, 0.5, 1.0] {
                    for i in 0..9 {
                        for j in 0..9 {
                            let u1 = i as f32 / 8.0;
                            let u2 = j as f32 / 8.0;
                            let h = vndf_half(v, n, alpha, u1, u2);
                            assert!((h.length() - 1.0).abs() < 1e-4, "|h| = {}", h.length());
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn reflection_direction_stays_above_the_surface_over_a_grid() {
        let normals = [
            Vec3::Y,
            Vec3::new(0.3, 0.2, 0.93).normalize(),
            Vec3::new(-0.6, 0.5, 0.62).normalize(),
            Vec3::new(0.8, -0.3, 0.52).normalize(),
        ];
        let views = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.2, -0.4, 0.9).normalize(),
            Vec3::new(-0.9, 0.2, 0.05).normalize(),
            Vec3::new(0.95, 0.1, 0.1).normalize(),
        ];
        for n0 in normals {
            for v in views {
                // The shader flips the normal to face the viewer first.
                let n = if n0.dot(v) < 0.0 { -n0 } else { n0 };
                for alpha in [0.005f32, 0.09, 0.3, 0.9] {
                    for i in 0..17 {
                        for j in 0..17 {
                            let dir =
                                reflection_direction(v, n, alpha, i as f32 / 16.0, j as f32 / 16.0);
                            assert!((dir.length() - 1.0).abs() < 1e-4);
                            assert!(dir.dot(n) > 0.0, "dir {dir:?} below {n:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reflection_mirror_is_sampled_at_zero_roughness_limit() {
        // With the two random numbers at the lobe centre the sample is the normal.
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 0.6, 0.8).normalize();
        let mirror = -v - 2.0 * (-v).dot(n) * n;
        let dir = reflection_direction(v, n, 0.0001, 0.5, 0.5);
        assert!(dir.dot(mirror) > 0.99, "dir {dir:?} mirror {mirror:?}");
    }
}

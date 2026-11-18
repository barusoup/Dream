//! Volumetric fog: a view-aligned froxel volume (inject, then integrate) and the
//! apply pass that blends it onto the scene. The inject pass reads the sun
//! shadow map once per froxel; the apply pass continues beyond the volume with
//! the closed-form height-fog integral. The maths is in `fog_common.wgsl`.
//!
//! Depth slices: slice `s` (continuous) sits at view depth
//! `near * (distance / near)^(s / slices)`; see [`slice_depth`] and
//! [`depth_to_slice`]. Texel `k` of the integrated volume holds the total up
//! to the end of slice `k`, and the apply pass reads it at `(s - 0.5) / slices`.

use super::SCENE_FORMAT;
use super::post::{
    BindKind, FullscreenTarget, PostPass, create_general_image, create_general_volume,
    host_uniform_slots,
};
use super::vulkan::{self, GpuBuffer, GpuContext, GpuImage};
use crate::lighting::LightingFrame;
use crate::render::FogSettings;
use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;

const COMMON_WGSL: &str = include_str!("fog_common.wgsl");
const INJECT_WGSL: &str = include_str!("fog_inject.wgsl");
const INTEGRATE_WGSL: &str = include_str!("fog_integrate.wgsl");
const APPLY_WGSL: &str = include_str!("fog_apply.wgsl");
/// Froxel texel format: scattering rgb and extinction a.
const VOLUME_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Size of the `FogParams` block all three passes read.
const PARAMS_SIZE: u64 = std::mem::size_of::<FogParams>() as u64;
/// Size of the per-slot copy of the shadow lighting block.
const LIGHTING_SIZE: u64 = std::mem::size_of::<LightingFrame>() as u64;
/// Sky pixels integrate the height fog this many distances past the volume.
const SKY_PATH_FACTOR: f32 = 2.0;
/// Fraction of the history kept each frame when temporal reprojection is on.
const HISTORY_WEIGHT: f32 = 0.9;

/// CPU mirror of `FogParams` in `fog_common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FogParams {
    grid: [f32; 4],
    range: [f32; 4],
    density: [f32; 4],
    scatter: [f32; 4],
    light: [f32; 4],
    wind: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    cam_pos: [f32; 4],
    tan_fov: [f32; 4],
    cam_to_world: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
}

/// The per-frame inputs of [`params`].
pub(crate) struct FogFrame {
    /// Camera space to world space, unjittered.
    pub cam_to_world: Mat4,
    pub cam_pos: Vec3,
    /// Tangent of half the vertical field of view, and the aspect ratio.
    pub tan_half_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// Previous unjittered view-projection, world to clip.
    pub prev_view_proj: Mat4,
    /// Direction to the sun, the sun's colour, and the sky ambient radiance.
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub ambient: Vec3,
    /// Scene time in seconds (drives the noise drift) and the frame index (drives
    /// the slice jitter).
    pub time: f32,
    pub frame: u64,
    /// The history is invalid: the inject pass ignores it.
    pub reset: bool,
}

/// Build the parameter block of one frame.
pub(crate) fn params(settings: &FogSettings, frame: &FogFrame) -> FogParams {
    let (w, h, n) = settings.resolution.size();
    let jitter = if settings.temporal {
        // Golden-ratio sequence: each frame's slices sample at a new offset.
        (frame.frame as f64 * 0.618_033_988_75).fract() as f32
    } else {
        0.5
    };
    let drift = -Vec3::from(settings.wind) * frame.time;
    let c = frame.cam_to_world.to_cols_array_2d();
    let p = frame.prev_view_proj.to_cols_array_2d();
    let sun = frame.sun_dir.normalize_or_zero();
    FogParams {
        grid: [w as f32, h as f32, n as f32, frame.near],
        range: [
            settings.distance,
            SKY_PATH_FACTOR * settings.distance,
            0.0,
            if frame.reset { 1.0 } else { 0.0 },
        ],
        density: [
            settings.density,
            settings.height_falloff,
            settings.base_height,
            settings.noise,
        ],
        scatter: [
            settings.albedo[0],
            settings.albedo[1],
            settings.albedo[2],
            settings.anisotropy,
        ],
        light: [
            settings.ambient,
            settings.sun_scatter,
            settings.noise_scale,
            jitter,
        ],
        wind: [
            drift.x,
            drift.y,
            drift.z,
            if settings.temporal {
                HISTORY_WEIGHT
            } else {
                0.0
            },
        ],
        sun_dir: [sun.x, sun.y, sun.z, 0.0],
        sun_color: [frame.sun_color.x, frame.sun_color.y, frame.sun_color.z, 0.0],
        ambient: [frame.ambient.x, frame.ambient.y, frame.ambient.z, 0.0],
        cam_pos: [frame.cam_pos.x, frame.cam_pos.y, frame.cam_pos.z, 0.0],
        tan_fov: [
            frame.tan_half_y * frame.aspect,
            frame.tan_half_y,
            frame.far,
            0.0,
        ],
        cam_to_world: c,
        prev_view_proj: p,
    }
}

/// CPU mirror of the slice mapping in `fog_common.wgsl` (checked by the tests).
#[cfg_attr(not(test), allow(dead_code))]
/// Depth-slice coordinate `s` to view depth: `near * (dist / near)^(s / slices)`.
pub(crate) fn slice_depth(s: f32, near: f32, dist: f32, slices: f32) -> f32 {
    near * (dist / near).powf(s / slices)
}

/// Inverse of [`slice_depth`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn depth_to_slice(z: f32, near: f32, dist: f32, slices: f32) -> f32 {
    slices * (z.max(near) / near).ln() / (dist / near).ln()
}

/// Closed-form optical depth of exponential height fog (checked against numeric integration). along the ray segment
/// `t` in `[t0, t1]`, with the ray at height `y0 + t * dy`: density `density`
/// below `base` and `density * exp(-falloff * (y - base))` above it. Mirrors
/// `fog_height_od` in `fog_common.wgsl`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn height_fog_optical_depth(
    t0: f32,
    t1: f32,
    y0: f32,
    dy: f32,
    density: f32,
    falloff: f32,
    base: f32,
) -> f32 {
    if t1 <= t0 {
        return 0.0;
    }
    if dy.abs() <= 1e-6 {
        let height = if y0 < base {
            1.0
        } else {
            (-falloff * (y0 - base)).exp()
        };
        return density * height * (t1 - t0);
    }
    let tc = ((base - y0) / dy).clamp(t0, t1);
    let (below, above) = if dy < 0.0 {
        ((tc, t1), (t0, tc))
    } else {
        ((t0, tc), (tc, t1))
    };
    let mut od = density * (below.1 - below.0);
    let len = above.1 - above.0;
    if len > 0.0 {
        let ya = (y0 + dy * above.0 - base).max(0.0);
        let kd = falloff * dy;
        let integral = if kd.abs() > 1e-6 {
            (1.0 - (-kd * len).exp()) / kd
        } else {
            len
        };
        od += density * (-falloff * ya).exp() * integral;
    }
    od
}

/// Fog resources for one settings value: the volume (inject ping-pong and the
/// integrated volume), the passes and the per-slot parameter blocks.
pub(crate) struct Fog {
    ctx: Arc<GpuContext>,
    built_for: FogSettings,
    size: (u32, u32, u32),
    /// Inject output, two images: this frame writes `volumes[write]` and reads
    /// last frame's `volumes[1 - write]` as history.
    volumes: [GpuImage; 2],
    integrated: GpuImage,
    write: usize,
    /// False until a record has run; the first record resets temporal history.
    primed: bool,
    inject: PostPass,
    integrate: PostPass,
    apply: PostPass,
    linear: vk::Sampler,
    shadow_compare: vk::Sampler,
    /// Stand-in for the shadow map while shadows are off (never sampled then).
    dummy_shadow: GpuImage,
    /// Stand-in for the cloud shadow map while clouds are off (never sampled then).
    dummy_cloud: GpuImage,
    params: Vec<GpuBuffer>,
    lighting: Vec<GpuBuffer>,
}

/// Pass inputs of [`Fog::record`].
pub(crate) struct FogTargets {
    /// Scene colour (blended onto).
    pub colour: vk::ImageView,
    /// Scene depth, depth aspect, sampled.
    pub depth: vk::ImageView,
    /// Sun shadow map array, when shadows are on.
    pub shadow: Option<vk::ImageView>,
    /// Cloud shadow map of this frame slot, when clouds shadow the sun.
    pub cloud_shadow: Option<vk::ImageView>,
}

/// Target of the apply pass: scene colour, blended.
const APPLY_TARGET: FullscreenTarget = FullscreenTarget {
    colour: &[SCENE_FORMAT],
    blend: true,
    depth_equal: None,
};

impl Fog {
    /// Build the volume and passes for `settings`.
    pub(crate) fn new(ctx: &Arc<GpuContext>, settings: &FogSettings) -> Self {
        let settings = settings.sanitised();
        let size = settings.resolution.size();
        let history_size = if settings.temporal { size } else { (1, 1, 1) };
        let volumes = [
            create_general_volume(ctx, "fog volume 0", size, VOLUME_FORMAT),
            create_general_volume(ctx, "fog volume 1", history_size, VOLUME_FORMAT),
        ];
        let integrated = create_general_volume(ctx, "fog integrated", size, VOLUME_FORMAT);
        let inject_kinds = [
            BindKind::Uniform,
            BindKind::Sampled3d,
            BindKind::Sampler,
            BindKind::Storage3d(VOLUME_FORMAT),
            BindKind::Sampled,
            BindKind::ComparisonSampler,
            BindKind::Uniform,
            BindKind::Sampled,
        ];
        let inject = PostPass::new(
            ctx,
            "fog inject",
            &format!("{COMMON_WGSL}\n{INJECT_WGSL}"),
            c"main",
            &inject_kinds,
        );
        let integrate = PostPass::new(
            ctx,
            "fog integrate",
            &format!("{COMMON_WGSL}\n{INTEGRATE_WGSL}"),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Sampled3d,
                BindKind::Storage3d(VOLUME_FORMAT),
            ],
        );
        let apply = PostPass::new_fullscreen(
            ctx,
            "fog apply",
            &format!("{COMMON_WGSL}\n{APPLY_WGSL}"),
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampled3d,
                BindKind::Sampler,
            ],
            APPLY_TARGET,
        );
        let linear = ctx.create_sampler("fog linear sampler");
        let shadow_compare = create_shadow_compare(ctx);
        let dummy_shadow = ctx.create_image_array_mips(
            "fog dummy shadow",
            1,
            1,
            ctx.shadow_depth_format(),
            vk::ImageUsageFlags::SAMPLED,
            1,
            1,
        );
        ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                ctx.device(),
                cmd,
                dummy_shadow.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
                vk::ImageAspectFlags::DEPTH,
            );
        });
        let dummy_cloud = create_general_image(
            ctx,
            "fog dummy cloud shadow",
            (1, 1),
            vk::Format::R32_SFLOAT,
        );
        Self {
            ctx: Arc::clone(ctx),
            built_for: settings,
            size,
            volumes,
            integrated,
            write: 0,
            primed: false,
            inject,
            integrate,
            apply,
            linear,
            shadow_compare,
            dummy_shadow,
            dummy_cloud,
            params: host_uniform_slots(ctx, "fog params", PARAMS_SIZE),
            lighting: host_uniform_slots(ctx, "fog lighting", LIGHTING_SIZE),
        }
    }

    /// The settings the volume was built for (sanitised).
    pub(crate) fn built_for(&self) -> &FogSettings {
        &self.built_for
    }

    /// Froxel grid size (width, height, slices).
    pub(crate) fn size(&self) -> (u32, u32, u32) {
        self.size
    }

    /// The integrated volume, which the apply pass reads.
    pub(crate) fn integrated_image(&self) -> vk::Image {
        self.integrated.image
    }

    /// The inject output of the last recorded frame (per-froxel scattering and extinction).
    pub(crate) fn injected_image(&self) -> vk::Image {
        self.volumes[1 - self.write].image
    }

    /// Store this frame's shadow lighting block for `slot`: the fog's shadow
    /// lookups read a copy of the scene's `LightingFrame`.
    pub(crate) fn upload_lighting(&mut self, slot: usize, block: &LightingFrame) {
        let slots = self.lighting.len();
        self.lighting[slot % slots].write(0, bytemuck::bytes_of(block));
    }

    /// Record the inject, integrate and apply passes for `slot`. `block` is the
    /// frame's parameter block (see [`params`]). The apply pass runs in a
    /// rendering scope of its own over `region` (viewport and scissor).
    pub(crate) fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        block: &FogParams,
        targets: &FogTargets,
        region: (vk::Viewport, vk::Rect2D),
        mark: &mut dyn FnMut(&'static str),
    ) {
        let mut block = *block;
        if !self.primed {
            // An unused history volume may contain stale VRAM until first inject.
            block.range[3] = 1.0;
        }
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(&block));
        self.primed = true;
        let params = &self.params[slot % slots];
        let (w, h, n) = self.size;
        let write = self.write;
        let read = 1 - write;

        mark("fog inject");
        let inject = &self.inject;
        inject.bind_uniform(slot, 0, params, PARAMS_SIZE);
        inject.bind_sampled(slot, 1, self.volumes[read].view);
        inject.bind_sampler(slot, 2, self.linear);
        inject.bind_storage(slot, 3, self.volumes[write].view, VOLUME_FORMAT);
        inject.bind_sampled_in(
            slot,
            4,
            targets.shadow.unwrap_or(self.dummy_shadow.view),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        inject.bind_sampler(slot, 5, self.shadow_compare);
        inject.bind_uniform(slot, 6, &self.lighting[slot % slots], LIGHTING_SIZE);
        inject.bind_sampled(
            slot,
            7,
            targets.cloud_shadow.unwrap_or(self.dummy_cloud.view),
        );
        inject.dispatch_layers(cmd, slot, w, h, n);

        mark("fog integrate");
        let integrate = &self.integrate;
        integrate.bind_uniform(slot, 0, params, PARAMS_SIZE);
        integrate.bind_sampled(slot, 1, self.volumes[write].view);
        integrate.bind_storage(slot, 2, self.integrated.view, VOLUME_FORMAT);
        integrate.dispatch_after_compute(cmd, slot, w, h);

        mark("fog apply");
        let apply = &self.apply;
        apply.bind_uniform(slot, 0, params, PARAMS_SIZE);
        apply.bind_sampled(slot, 1, targets.depth);
        apply.bind_sampled(slot, 2, self.integrated.view);
        apply.bind_sampler(slot, 3, self.linear);
        let colour = vk::RenderingAttachmentInfo::default()
            .image_view(targets.colour)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let colours = [colour];
        let info = vk::RenderingInfo::default()
            .render_area(region.1)
            .layer_count(1)
            .color_attachments(&colours);
        unsafe {
            device.cmd_begin_rendering(cmd, &info);
            device.cmd_set_viewport(cmd, 0, &[region.0]);
            device.cmd_set_scissor(cmd, 0, &[region.1]);
        }
        apply.draw_fullscreen(cmd, slot);
        unsafe { device.cmd_end_rendering(cmd) };

        if self.built_for.temporal {
            self.write = read;
        }
    }
}

/// Comparison sampler for the shadow map: hardware PCF-style compare, clamped.
fn create_shadow_compare(ctx: &Arc<GpuContext>) -> vk::Sampler {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL);
    unsafe { ctx.device().create_sampler(&info, None) }
        .unwrap_or_else(|err| panic!("fog shadow compare sampler: {err:?}"))
}

impl Drop for Fog {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.linear, None);
            self.ctx.device().destroy_sampler(self.shadow_compare, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fog_params_block_matches_wgsl_layout() {
        // Eleven vec4s, then two mat4s: 176 + 128 bytes.
        assert_eq!(std::mem::size_of::<FogParams>(), 304);
        assert_eq!(std::mem::offset_of!(FogParams, tan_fov), 160);
        assert_eq!(std::mem::offset_of!(FogParams, cam_to_world), 176);
        assert_eq!(std::mem::offset_of!(FogParams, prev_view_proj), 240);
        // The shadow lighting block is the scene's LightingFrame: 512 bytes.
        assert_eq!(LIGHTING_SIZE, 512);
    }

    #[test]
    fn fog_slice_mapping_round_trips() {
        let (near, dist, n) = (0.1f32, 160.0f32, 64.0f32);
        let mut s = 0.0f32;
        while s <= n {
            let z = slice_depth(s, near, dist, n);
            let back = depth_to_slice(z, near, dist, n);
            assert!((back - s).abs() < 1e-3, "s {s} -> z {z} -> {back}");
            s += 0.37;
        }
        assert!((slice_depth(0.0, near, dist, n) - near).abs() < 1e-6);
        assert!((slice_depth(n, near, dist, n) - dist).abs() < 1e-2);
    }

    #[test]
    fn fog_height_optical_depth_matches_numerical_integration() {
        let cases: [(f32, f32, f32, f32); 6] = [
            (0.0, 50.0, 3.5, 0.0),
            (0.0, 200.0, 3.5, -0.2),
            (0.0, 200.0, 3.5, 0.3),
            (10.0, 90.0, -2.0, 0.05),
            (0.0, 300.0, 40.0, -0.4),
            (5.0, 500.0, 1.0, 1.0),
        ];
        let (density, falloff, base) = (0.02f32, 0.12f32, 0.0f32);
        for (t0, t1, y0, dy) in cases {
            let closed = height_fog_optical_depth(t0, t1, y0, dy, density, falloff, base);
            let steps = 200_000;
            let dt = (t1 - t0) / steps as f32;
            let mut numeric = 0.0f64;
            for i in 0..steps {
                let t = t0 + (i as f32 + 0.5) * dt;
                let y = y0 + dy * t;
                numeric += (density * (-falloff * (y - base).max(0.0)).exp() * dt) as f64;
            }
            let rel = (closed as f64 - numeric).abs() / numeric.max(1e-9);
            assert!(
                rel < 0.01,
                "case {t0} {t1} {y0} {dy}: closed {closed} numeric {numeric}"
            );
        }
    }

    #[test]
    fn fog_params_follow_settings() {
        let settings = FogSettings::medium();
        let frame = FogFrame {
            cam_to_world: Mat4::IDENTITY,
            cam_pos: Vec3::new(1.0, 2.0, 3.0),
            tan_half_y: 0.5,
            aspect: 2.0,
            near: 0.1,
            far: 500.0,
            prev_view_proj: Mat4::IDENTITY,
            sun_dir: Vec3::Y,
            sun_color: Vec3::ONE,
            ambient: Vec3::splat(0.4),
            time: 2.0,
            frame: 7,
            reset: true,
        };
        let block = params(&settings, &frame);
        assert_eq!(block.grid, [160.0, 90.0, 64.0, 0.1]);
        assert_eq!(block.range[3], 1.0);
        assert_eq!(block.tan_fov[0], 1.0);
        assert!((block.wind[0] + 3.0).abs() < 1e-5);
        assert!(block.light[3] >= 0.0 && block.light[3] < 1.0);
    }
}

//! GPU clouds: the tiling noise volumes (built once) and the sky environment
//! panorama with its march. The view clouds and the cloud shadow map have their
//! own passes (`cloud_view`, `cloud_shadow`). All three share the maths in
//! `cloud_common.wgsl`; the settings and panorama mapping are in `render::clouds`.
//!
//! Each frame the panorama march refreshes one texel in each `N x N` block of the
//! panorama, so a texel is refreshed every `N * N` frames. A full refresh (on
//! creation, on a settings change and after a history reset) marches every texel
//! in one dispatch.

use super::atmosphere::LUT_FORMAT;
use super::cloud_shadow::{CloudShadow, ShadowGrid};
use super::cloud_view::{CloudView, ViewFrame};
use super::post::{
    BindKind, PostPass, create_general_image, create_general_volume, host_uniform_slots,
};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use crate::render::{
    CloudSettings, MAX_MARCH_DISTANCE, PANORAMA_SIZE, PANORAMA_STEPS, PANORAMA_STRIDE,
    quality_params, refresh_offset,
};
use ash::vk;
use glam::Vec3;
use std::sync::Arc;

const ATMO_WGSL: &str = include_str!("atmo_common.wgsl");
const COMMON_WGSL: &str = include_str!("cloud_common.wgsl");
const NOISE_BASE_WGSL: &str = include_str!("cloud_noise_base.wgsl");
const NOISE_DETAIL_WGSL: &str = include_str!("cloud_noise_detail.wgsl");
const MARCH_WGSL: &str = include_str!("cloud_march.wgsl");

/// Base shape volume size (texels per side).
const BASE_SIZE: (u32, u32, u32) = (96, 96, 96);
/// Detail erosion volume size.
const DETAIL_SIZE: (u32, u32, u32) = (32, 32, 32);
/// Storage format of both noise volumes.
const NOISE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Light steps per panorama sample: the panorama feeds reflections, not the view.
const PANORAMA_LIGHT_STEPS: u32 = 2;

/// Size of the `CloudParams` block the cloud passes read.
pub(crate) const PARAMS_SIZE: u64 = std::mem::size_of::<CloudParams>() as u64;
/// Size of the `AtmoParams` block the march reads (the atmosphere's per-slot block).
pub(crate) const ATMO_PARAMS_SIZE: u64 = 352;

/// CPU mirror of `CloudParams` in `cloud_common.wgsl`: seven vec4s, 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CloudParams {
    /// xyz: camera world position, w: sea level.
    pub cam: [f32; 4],
    /// Planet radius, base altitude, thickness and max distance.
    pub layer: [f32; 4],
    pub shape: [f32; 4],
    pub drift: [f32; 4],
    pub look: [f32; 4],
    pub refresh: [f32; 4],
    pub panorama: [f32; 4],
}

/// Per-frame scene inputs of the cloud passes.
pub(crate) struct CloudFrame {
    /// Camera world position, metres.
    pub camera: Vec3,
    /// Sea level, metres (world y of the planet's mean surface).
    pub sea_level: f32,
    /// Planet radius, metres.
    pub planet_radius: f32,
    /// Scene time, seconds.
    pub time: f32,
}

/// Atmosphere resources the cloud passes read, for one frame slot.
pub(crate) struct MarchInputs<'a> {
    /// The slot's `AtmoParams` block (written by the atmosphere this frame).
    pub params: &'a GpuBuffer,
    pub transmittance: vk::ImageView,
    pub skyview: vk::ImageView,
    pub lut_sampler: vk::Sampler,
}

/// The `CloudParams` block of one pass: the camera, the layer and the look, with
/// the pass's own `refresh` and `panorama` fields.
pub(crate) fn cloud_params(
    settings: &CloudSettings,
    frame: &CloudFrame,
    steps: u32,
    light_steps: u32,
    refresh: [f32; 4],
    panorama: [f32; 4],
) -> CloudParams {
    let s = settings;
    CloudParams {
        cam: [
            frame.camera.x,
            frame.camera.y,
            frame.camera.z,
            frame.sea_level,
        ],
        layer: [
            frame.planet_radius,
            s.base_altitude,
            s.thickness,
            MAX_MARCH_DISTANCE,
        ],
        shape: [s.coverage, s.density, s.scale, s.detail],
        drift: [s.wind[0], s.wind[1], s.evolution, frame.time],
        look: [s.silver_lining, s.ambient, steps as f32, light_steps as f32],
        refresh,
        panorama,
    }
}

/// Cloud resources for one settings value.
pub(crate) struct Clouds {
    ctx: Arc<GpuContext>,
    settings: CloudSettings,
    /// Radiance (rgb) and transmittance (a), one image, updated in place.
    /// The environment panorama, ping-ponged: `panorama[last]` was written last.
    panorama: [GpuImage; 2],
    last: usize,
    base: GpuImage,
    detail: GpuImage,
    /// Repeat in both axes, trilinear; the noise and the panorama's azimuth.
    repeat: vk::Sampler,
    params: Vec<GpuBuffer>,
    march: PostPass,
    /// Next dispatch refreshes every texel.
    pending_full: bool,
    /// Panorama frames recorded so far: the refresh offset of the next frame.
    frame: u64,
    /// The per-pixel view clouds, sized to the render target.
    pub(crate) view: CloudView,
    /// The top-down cloud shadow map of the sun.
    pub(crate) shadow: CloudShadow,
}

impl Clouds {
    /// Build the noise volumes (one-shot submission), the panorama and the view
    /// and shadow passes for a render target of `render_size`. A full panorama
    /// refresh is pending for the first frame.
    pub(crate) fn new(
        ctx: &Arc<GpuContext>,
        settings: &CloudSettings,
        render_size: (u32, u32),
    ) -> Self {
        let settings = settings.sanitised();
        let base = create_general_volume(ctx, "cloud base shape", BASE_SIZE, NOISE_FORMAT);
        let detail = create_general_volume(ctx, "cloud detail erosion", DETAIL_SIZE, NOISE_FORMAT);
        let panorama = [
            create_general_image(ctx, "cloud panorama 0", PANORAMA_SIZE, LUT_FORMAT),
            create_general_image(ctx, "cloud panorama 1", PANORAMA_SIZE, LUT_FORMAT),
        ];
        let repeat = create_repeat_sampler(ctx);
        let params = host_uniform_slots(ctx, "cloud params", PARAMS_SIZE);

        let base_pass = PostPass::new(
            ctx,
            "cloud noise base",
            &compose(NOISE_BASE_WGSL),
            c"main",
            &[BindKind::Storage3d(NOISE_FORMAT)],
        );
        let detail_pass = PostPass::new(
            ctx,
            "cloud noise detail",
            &compose(NOISE_DETAIL_WGSL),
            c"main",
            &[BindKind::Storage3d(NOISE_FORMAT)],
        );
        base_pass.bind_storage(0, 0, base.view, NOISE_FORMAT);
        detail_pass.bind_storage(0, 0, detail.view, NOISE_FORMAT);
        ctx.one_shot(|cmd| {
            base_pass.dispatch_layers(cmd, 0, BASE_SIZE.0, BASE_SIZE.1, BASE_SIZE.2);
            detail_pass.dispatch_layers(cmd, 0, DETAIL_SIZE.0, DETAIL_SIZE.1, DETAIL_SIZE.2);
        });
        drop(base_pass);
        drop(detail_pass);

        let march = PostPass::new(
            ctx,
            "cloud march",
            &compose(MARCH_WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Uniform,
                BindKind::Sampled3d,
                BindKind::Sampled3d,
                BindKind::Sampler,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampler,
                BindKind::Storage(LUT_FORMAT),
                BindKind::Sampled,
            ],
        );
        Self {
            ctx: Arc::clone(ctx),
            settings,
            panorama,
            last: 1,
            base,
            detail,
            repeat,
            params,
            march,
            pending_full: true,
            frame: 0,
            view: CloudView::new(ctx, render_size),
            shadow: CloudShadow::new(ctx),
        }
    }

    /// Follow a settings value and the render size: any settings change schedules
    /// a full panorama refresh, and a new render size rebuilds the view passes
    /// (waiting the GPU idle first). The noise volumes never change.
    pub(crate) fn apply(&mut self, settings: &CloudSettings, render_size: (u32, u32)) {
        let settings = settings.sanitised();
        if settings != self.settings {
            self.settings = settings;
            self.pending_full = true;
            self.view.invalidate();
        }
        if self.view.size() != render_size {
            unsafe { self.ctx.device().device_wait_idle() }.expect("wait idle before cloud resize");
            self.view = CloudView::new(&self.ctx, render_size);
        }
    }

    /// The sanitised settings the clouds are built for.
    pub(crate) fn settings(&self) -> &CloudSettings {
        &self.settings
    }

    /// The sampler the sky environment map uses for the panorama.
    pub(crate) fn sampler(&self) -> vk::Sampler {
        self.repeat
    }

    /// The panorama's view (for the sky environment map).
    pub(crate) fn panorama_view(&self) -> vk::ImageView {
        self.panorama[self.last].view
    }

    /// The panorama image, for readback.
    pub(crate) fn panorama_image(&self) -> (vk::Image, (u32, u32)) {
        (self.panorama[self.last].image, PANORAMA_SIZE)
    }

    /// The cloud shadow map of frame `slot`, sampled by the lighting and fog.
    pub(crate) fn shadow_view(&self, slot: usize) -> vk::ImageView {
        self.shadow.view(slot)
    }

    /// The noise volumes' views and the repeat sampler, for the passes that march.
    fn base_detail(&self) -> (vk::ImageView, vk::ImageView, vk::Sampler) {
        (self.base.view, self.detail.view, self.repeat)
    }

    /// Record the per-pixel view clouds (march, resolve, composite into `colour`)
    /// for frame `slot`. Needs this frame's sky-view LUT and linear depth.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_view(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        atmo: MarchInputs<'_>,
        frame: &CloudFrame,
        view: &ViewFrame,
        depth: vk::ImageView,
        colour: vk::ImageView,
        region: (vk::Viewport, vk::Rect2D),
        reset: bool,
        mark: &mut dyn FnMut(&'static str),
    ) {
        let (base, detail, sampler) = self.base_detail();
        let params = cloud_params(
            &self.settings,
            frame,
            0,
            quality_params(self.settings.quality).1,
            [0.0; 4],
            [0.0; 4],
        );
        let inputs = super::cloud_view::ViewInputs {
            atmo,
            base,
            detail,
            sampler,
            depth,
            colour,
            region,
        };
        self.view.record(
            cmd,
            slot,
            &inputs,
            &params,
            view,
            &self.settings,
            reset,
            mark,
        );
    }

    /// Record the cloud shadow map of frame `slot` for the sun `sun` (direction
    /// to the sun). The map is centred on `grid`.
    pub(crate) fn record_shadow(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        frame: &CloudFrame,
        grid: &ShadowGrid,
        sun: Vec3,
    ) {
        let (base, detail, sampler) = self.base_detail();
        let params = cloud_params(
            &self.settings,
            frame,
            0,
            quality_params(self.settings.quality).1,
            [0.0; 4],
            [0.0; 4],
        );
        self.shadow
            .record(cmd, slot, &params, grid, sun, (base, detail, sampler));
    }

    /// Record this frame's panorama march into `cmd` for frame `slot`. `reset`
    /// forces a full refresh (history was reset). Every texel refreshed here is in
    /// world directions, so camera rotation does not invalidate the panorama.
    pub(crate) fn record_panorama(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        atmo: MarchInputs<'_>,
        frame: &CloudFrame,
        reset: bool,
    ) {
        let full = std::mem::take(&mut self.pending_full) || reset;
        let (w, h) = PANORAMA_SIZE;
        let seed = self.frame;
        self.frame += 1;
        let stride = if full { 1 } else { PANORAMA_STRIDE };
        let (ox, oy) = if full {
            (0, 0)
        } else {
            refresh_offset(seed, stride)
        };
        let block = cloud_params(
            &self.settings,
            frame,
            PANORAMA_STEPS,
            PANORAMA_LIGHT_STEPS,
            [stride as f32, ox as f32, oy as f32, (seed % 65_536) as f32],
            [w as f32, h as f32, if full { 1.0 } else { 0.0 }, 0.0],
        );
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(&block));

        let pass = &self.march;
        pass.bind_uniform(slot, 0, atmo.params, ATMO_PARAMS_SIZE);
        pass.bind_uniform(slot, 1, &self.params[slot % slots], PARAMS_SIZE);
        pass.bind_sampled(slot, 2, self.base.view);
        pass.bind_sampled(slot, 3, self.detail.view);
        pass.bind_sampler(slot, 4, self.repeat);
        pass.bind_sampled(slot, 5, atmo.transmittance);
        pass.bind_sampled(slot, 6, atmo.skyview);
        pass.bind_sampler(slot, 7, atmo.lut_sampler);
        let write = 1 - self.last;
        pass.bind_storage(slot, 8, self.panorama[write].view, LUT_FORMAT);
        pass.bind_sampled(slot, 9, self.panorama[self.last].view);
        pass.dispatch(cmd, slot, w, h);
        self.last = write;
    }
}

/// Prepend the atmosphere and cloud maths to a cloud pass, in the order build.rs uses.
pub(super) fn compose(pass: &str) -> String {
    format!("{ATMO_WGSL}\n{COMMON_WGSL}\n{pass}")
}

/// Trilinear, repeat in every axis: the noise volumes tile, and the panorama
/// repeats in azimuth (the environment map clamps the zenith and horizon rows itself).
fn create_repeat_sampler(ctx: &Arc<GpuContext>) -> vk::Sampler {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);
    unsafe { ctx.device().create_sampler(&info, None) }
        .unwrap_or_else(|err| panic!("cloud repeat sampler: {err:?}"))
}

impl Drop for Clouds {
    fn drop(&mut self) {
        // The march and the images drop after this; the GPU must be idle first.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.repeat, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clouds_params_block_matches_wgsl_layout() {
        // Seven vec4s: 112 bytes, each field at 16-byte steps.
        assert_eq!(std::mem::size_of::<CloudParams>(), 112);
        assert_eq!(std::mem::offset_of!(CloudParams, drift), 48);
        assert_eq!(std::mem::offset_of!(CloudParams, panorama), 96);
    }
}

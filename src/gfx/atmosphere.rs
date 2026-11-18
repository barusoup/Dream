//! GPU atmosphere (Hillaire 2020): the transmittance and multiple-scattering
//! LUTs, built once per settings value; the sky-view LUT, rebuilt every frame
//! the atmosphere is on; and the sky pass that draws it where the scene has no
//! geometry. The maths is in `atmo_common.wgsl` and mirrors
//! `render::AtmosphereModel`.

use super::clouds::MarchInputs;
use super::post::{BindKind, FullscreenTarget, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage, image_barrier_mips};
use super::{DEPTH_FORMAT, SCENE_FORMAT, VELOCITY_FORMAT};
use crate::lighting::LightingEnv;
use crate::render::{
    AtmosphereSettings, MULTISCATTER_SIZE, PlanetKm, SKY_ENV_MIPS, SKY_ENV_SIZE, SKYVIEW_SIZE,
    TRANSMITTANCE_SIZE,
};
use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const COMMON_WGSL: &str = include_str!("atmo_common.wgsl");
const TRANSMITTANCE_WGSL: &str = include_str!("atmo_transmittance.wgsl");
const MULTISCATTER_WGSL: &str = include_str!("atmo_multiscatter.wgsl");
const SKYVIEW_WGSL: &str = include_str!("atmo_skyview.wgsl");
const SKY_WGSL: &str = include_str!("atmo_sky.wgsl");
const ENVMAP_WGSL: &str = include_str!("atmo_envmap.wgsl");
const ENVMAP_DOWN_WGSL: &str = include_str!("atmo_envmap_down.wgsl");
const CLOUD_COMMON_WGSL: &str = include_str!("cloud_common.wgsl");
/// Each built atmosphere takes a fresh generation, so the lighting set can tell
/// its environment map from one a reused handle value used to name.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
/// LUT texel format.
pub(crate) const LUT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Size of the `AtmoParams` block all the passes read.
const PARAMS_SIZE: u64 = std::mem::size_of::<AtmoParams>() as u64;
/// Size of the environment-map block (`EnvMapParams` in `atmo_envmap.wgsl`).
const ENV_PARAMS_SIZE: u64 = 16;

/// CPU mirror of `AtmoParams` in `atmo_common.wgsl`. Vectors are `[f32; 4]`
/// so the layout is the WGSL one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AtmoParams {
    planet: [f32; 4],
    rayleigh: [f32; 4],
    mie: [f32; 4],
    ozone: [f32; 4],
    ground: [f32; 4],
    sun_dir: [f32; 4],
    sun_irradiance: [f32; 4],
    camera: [f32; 4],
    cam_pos: [f32; 4],
    screen: [f32; 4],
    inv_view_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
}

/// The camera state the sky pass needs for one frame.
pub(crate) struct SkyCamera {
    pub position: Vec3,
    /// Inverse of the jittered view-projection (scene pixel to world).
    pub inv_view_proj: Mat4,
    /// Rotation-only view-projection, unjittered, this frame and the last.
    pub view_proj: Mat4,
    pub prev_view_proj: Mat4,
    /// Scene render size in pixels.
    pub size: (u32, u32),
    /// Whether the sky environment map composites the cloud panorama.
    pub clouds: bool,
}

/// Pixel targets of the sky pass (the scene's colour and velocity, and depth
/// read as a read-only attachment).
#[derive(Clone, Copy)]
pub(crate) struct SkyTargets {
    pub colour: vk::ImageView,
    pub velocity: vk::ImageView,
    pub depth: vk::ImageView,
}

/// Build the per-frame block for `settings`, the environment and the camera.
pub(crate) fn frame_params(
    settings: &AtmosphereSettings,
    env: &LightingEnv,
    camera: &SkyCamera,
) -> AtmoParams {
    let planet = PlanetKm::from_settings(settings);
    let sun = Vec3::from_slice(&env.sun_direction[..3]).normalize_or_zero();
    let altitude = (camera.position.y - settings.sea_level).max(1.0);
    let night = settings.night_sky;
    let r_cam = planet.rg + altitude / 1000.0;
    let irradiance = Vec3::from_slice(&env.sun_color[..3]) * crate::render::SKY_IRRADIANCE_SCALE;
    let clouds = if camera.clouds { 1.0 } else { 0.0 };
    let (w, h) = (camera.size.0.max(1) as f32, camera.size.1.max(1) as f32);
    AtmoParams {
        planet: [planet.rg, planet.rt, 8.0, 1.2],
        // The night sky radiance rides in the unused w of the Rayleigh, Mie and
        // ozone blocks, so the block keeps its layout (`atmo_night` reads them).
        rayleigh: [
            planet.rayleigh.x,
            planet.rayleigh.y,
            planet.rayleigh.z,
            night[0],
        ],
        mie: [
            planet.mie_scatter,
            planet.mie_extinction,
            planet.mie_g,
            night[1],
        ],
        ozone: [planet.ozone.x, planet.ozone.y, planet.ozone.z, night[2]],
        // w: 1 to draw the planet ground below the horizon (`show_ground`).
        ground: [
            planet.albedo.x,
            planet.albedo.y,
            planet.albedo.z,
            if settings.show_ground { 1.0 } else { 0.0 },
        ],
        sun_dir: [sun.x, sun.y, sun.z, 0.0],
        sun_irradiance: [irradiance.x, irradiance.y, irradiance.z, clouds],
        camera: [
            r_cam,
            if settings.sun_disk { 1.0 } else { 0.0 },
            settings.sun_angular_radius,
            settings.sky_intensity,
        ],
        cam_pos: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        screen: [w, h, 1.0 / w, 1.0 / h],
        inv_view_proj: camera.inv_view_proj.to_cols_array_2d(),
        view_proj: camera.view_proj.to_cols_array_2d(),
        prev_view_proj: camera.prev_view_proj.to_cols_array_2d(),
    }
}

/// Target formats of the sky pass: the scene colour and velocity, depth tested
/// `EQUAL` and never written.
const SKY_TARGET: FullscreenTarget = FullscreenTarget {
    colour: &[SCENE_FORMAT, VELOCITY_FORMAT],
    blend: false,
    depth_equal: Some(DEPTH_FORMAT),
};

/// The sky environment map of one atmosphere: per frame slot, a 256 x 128
/// equirectangular image of the sky with a full mip chain (the roughness blur).
/// Mip 0 is written every frame; the chain is reduced from it.
struct SkyEnv {
    /// One image per frame slot, viewed with the full chain (what lighting samples).
    images: Vec<GpuImage>,
    /// Single-level views per slot and mip: storage targets and reduction sources.
    level_views: Vec<Vec<vk::ImageView>>,
    /// Ground radiance per slot, read by the mip-0 pass.
    params: Vec<GpuBuffer>,
    /// Mip 0 from the sky-view LUT and the cloud panorama.
    map_pass: PostPass,
    /// One 2x2 reduction per mip from 1 to `SKY_ENV_MIPS - 1`.
    reduce_passes: Vec<PostPass>,
    /// Frames recorded since the atmosphere was built.
    recorded: u64,
}

/// The atmosphere's GPU resources for one settings value.
pub(crate) struct Atmosphere {
    ctx: Arc<GpuContext>,
    /// Settings the static LUTs were built for (sanitised).
    built_for: AtmosphereSettings,
    /// Identifies this build; see `NEXT_GENERATION`.
    generation: u64,
    transmittance: GpuImage,
    multiscatter: GpuImage,
    /// Sky-view LUT, one per frame slot: a slot's copy is rewritten only while
    /// the slot is idle, so the sky pass of an earlier frame still in flight
    /// keeps reading its own.
    skyview: Vec<GpuImage>,
    lut_sampler: vk::Sampler,
    params: Vec<GpuBuffer>,
    skyview_pass: PostPass,
    sky_pass: PostPass,
    /// Present when the render settings enable the atmosphere itself (not only
    /// clouds): the sky environment map that reflections and ambient light read.
    env: Option<SkyEnv>,
}

impl Atmosphere {
    /// Build the static LUTs for `settings` (a one-shot GPU submission) and the
    /// per-frame passes. `env_map` also allocates the sky environment map. Nothing
    /// here depends on the scene's size.
    pub(crate) fn new(ctx: &Arc<GpuContext>, settings: &AtmosphereSettings, env_map: bool) -> Self {
        let settings = settings.sanitised();
        let transmittance = create_general_image(
            ctx,
            "atmosphere transmittance",
            TRANSMITTANCE_SIZE,
            LUT_FORMAT,
        );
        let multiscatter = create_general_image(
            ctx,
            "atmosphere multiscatter",
            MULTISCATTER_SIZE,
            LUT_FORMAT,
        );
        let skyview = (0..ctx.max_frames_in_flight())
            .map(|slot| {
                create_general_image(
                    ctx,
                    &format!("atmosphere skyview {slot}"),
                    SKYVIEW_SIZE,
                    LUT_FORMAT,
                )
            })
            .collect();
        let lut_sampler = ctx.create_sampler("atmosphere lut sampler");
        let mut params = host_uniform_slots(ctx, "atmosphere params", PARAMS_SIZE);

        let transmittance_pass = PostPass::new(
            ctx,
            "atmosphere transmittance",
            &compose(TRANSMITTANCE_WGSL),
            c"main",
            &[BindKind::Uniform, BindKind::Storage(LUT_FORMAT)],
        );
        let multiscatter_pass = PostPass::new(
            ctx,
            "atmosphere multiscatter",
            &compose(MULTISCATTER_WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampler,
                BindKind::Storage(LUT_FORMAT),
            ],
        );
        let skyview_pass = PostPass::new(
            ctx,
            "atmosphere skyview",
            &compose(SKYVIEW_WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampler,
                BindKind::Storage(LUT_FORMAT),
            ],
        );
        let sky_pass = PostPass::new_fullscreen(
            ctx,
            "atmosphere sky",
            &compose_sky(),
            &[
                BindKind::Uniform,
                BindKind::Sampled,
                BindKind::Sampled,
                BindKind::Sampler,
            ],
            SKY_TARGET,
        );

        // The static LUTs: transmittance first, then multiple scattering, which
        // samples it. Slot 0's uniform block serves both builds.
        let block = frame_params(
            &settings,
            &LightingEnv::default(),
            &SkyCamera {
                position: Vec3::ZERO,
                inv_view_proj: Mat4::IDENTITY,
                view_proj: Mat4::IDENTITY,
                prev_view_proj: Mat4::IDENTITY,
                size: (1, 1),
                clouds: false,
            },
        );
        params[0].write(0, bytemuck::bytes_of(&block));
        transmittance_pass.bind_uniform(0, 0, &params[0], PARAMS_SIZE);
        transmittance_pass.bind_storage(0, 1, transmittance.view, LUT_FORMAT);
        multiscatter_pass.bind_uniform(0, 0, &params[0], PARAMS_SIZE);
        multiscatter_pass.bind_sampled(0, 1, transmittance.view);
        multiscatter_pass.bind_sampler(0, 2, lut_sampler);
        multiscatter_pass.bind_storage(0, 3, multiscatter.view, LUT_FORMAT);
        ctx.one_shot(|cmd| {
            transmittance_pass.dispatch(cmd, 0, TRANSMITTANCE_SIZE.0, TRANSMITTANCE_SIZE.1);
            multiscatter_pass.dispatch(cmd, 0, MULTISCATTER_SIZE.0, MULTISCATTER_SIZE.1);
        });
        // Only slot 0 of the two static passes was used; they are dropped here.
        drop(transmittance_pass);
        drop(multiscatter_pass);

        let env = env_map.then(|| build_sky_env(ctx));

        Self {
            ctx: Arc::clone(ctx),
            built_for: settings,
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            transmittance,
            multiscatter,
            skyview,
            lut_sampler,
            params,
            skyview_pass,
            sky_pass,
            env,
        }
    }

    /// Whether this atmosphere keeps a sky environment map.
    pub(crate) fn has_env_map(&self) -> bool {
        self.env.is_some()
    }

    /// The environment map frame `slot`'s lighting set samples: `(generation,
    /// full-chain view)` of the map recorded into `slot`. `None` until one has been
    /// recorded, or when there is no map. The caller passes the previous frame's
    /// slot, as the screen-space chains' outputs are bound.
    pub(crate) fn env_map(&self, slot: usize) -> Option<(u64, vk::ImageView)> {
        let env = self.env.as_ref()?;
        if env.recorded == 0 {
            return None;
        }
        Some((self.generation, env.images[slot % env.images.len()].view))
    }

    /// The mip-0 image of frame `slot`'s environment map, for readback.
    pub(crate) fn env_map_image(&self, slot: usize) -> Option<vk::Image> {
        let env = self.env.as_ref()?;
        Some(env.images[slot % env.images.len()].image)
    }

    /// Record the sky environment map for `slot` into `cmd`: mip 0 from the
    /// sky-view LUT recorded for this slot (with the cloud panorama composited in
    /// when given), then the chain reduced from it. `ground` is the ground radiance
    /// below the horizon. Does nothing without an environment map.
    pub(crate) fn record_env_map(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        ground: [f32; 3],
        clouds: Option<(vk::ImageView, vk::Sampler)>,
    ) {
        let slots = self.params.len();
        let sky_block = &self.params[slot % slots];
        let skyview = self.skyview[slot % self.skyview.len()].view;
        let lut_sampler = self.lut_sampler;
        let Some(env) = self.env.as_mut() else {
            return;
        };
        let slot = slot % env.images.len();
        env.params[slot].write(
            0,
            bytemuck::bytes_of(&[ground[0], ground[1], ground[2], 0.0]),
        );
        // With clouds off the sky-view image stands in for the panorama; the
        // shader does not read it (`sun_irradiance.w` is 0).
        let (panorama, panorama_sampler) = clouds.unwrap_or((skyview, lut_sampler));
        let map = &env.map_pass;
        map.bind_uniform(slot, 0, sky_block, PARAMS_SIZE);
        map.bind_uniform(slot, 1, &env.params[slot], ENV_PARAMS_SIZE);
        map.bind_sampled(slot, 2, skyview);
        map.bind_sampler(slot, 3, lut_sampler);
        map.bind_sampled(slot, 4, panorama);
        map.bind_sampler(slot, 5, panorama_sampler);
        map.bind_storage(slot, 6, env.level_views[slot][0], LUT_FORMAT);
        map.dispatch(cmd, slot, SKY_ENV_SIZE.0, SKY_ENV_SIZE.1);
        // Each reduction reads the level above it; `dispatch` orders them.
        let (mut w, mut h) = SKY_ENV_SIZE;
        for (level, pass) in env.reduce_passes.iter().enumerate() {
            let level = level + 1;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            pass.bind_sampled(slot, 0, env.level_views[slot][level - 1]);
            pass.bind_storage(slot, 1, env.level_views[slot][level], LUT_FORMAT);
            pass.dispatch(cmd, slot, w, h);
        }
        env.recorded += 1;
    }

    /// The settings the static LUTs were built for.
    pub(crate) fn built_for(&self) -> &AtmosphereSettings {
        &self.built_for
    }

    /// The sky-view image of frame `slot`.
    pub(crate) fn skyview_image(&self, slot: usize) -> vk::Image {
        self.skyview[slot % self.skyview.len()].image
    }

    /// The atmosphere resources the cloud march reads in frame `slot`. Call after
    /// [`Self::record_skyview`] for the slot, so the sky-view LUT is this frame's.
    pub(crate) fn march_inputs(&self, slot: usize) -> MarchInputs<'_> {
        let slots = self.params.len();
        MarchInputs {
            params: &self.params[slot % slots],
            transmittance: self.transmittance.view,
            skyview: self.skyview[slot % self.skyview.len()].view,
            lut_sampler: self.lut_sampler,
        }
    }

    /// Record the per-frame sky-view LUT for `slot` from `block`.
    pub(crate) fn record_skyview(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        block: &AtmoParams,
    ) {
        let slots = self.params.len();
        self.params[slot % slots].write(0, bytemuck::bytes_of(block));
        let buffer = &self.params[slot % slots];
        let pass = &self.skyview_pass;
        pass.bind_uniform(slot, 0, buffer, PARAMS_SIZE);
        pass.bind_sampled(slot, 1, self.transmittance.view);
        pass.bind_sampled(slot, 2, self.multiscatter.view);
        pass.bind_sampler(slot, 3, self.lut_sampler);
        pass.bind_storage(
            slot,
            4,
            self.skyview[slot % self.skyview.len()].view,
            LUT_FORMAT,
        );
        pass.dispatch(cmd, slot, SKYVIEW_SIZE.0, SKYVIEW_SIZE.1);
    }

    /// Draw the sky into `targets` for `slot`: a rendering scope of its own with
    /// the scene colour and velocity loaded, and depth tested `EQUAL` against
    /// the cleared far plane. `block` must match the one recorded for the slot
    /// by [`Self::record_skyview`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_sky(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        targets: SkyTargets,
        region: (vk::Viewport, vk::Rect2D),
        extent: vk::Rect2D,
    ) {
        let slots = self.params.len();
        let buffer = &self.params[slot % slots];
        let pass = &self.sky_pass;
        let skyview = self.skyview[slot % self.skyview.len()].view;
        pass.bind_uniform(slot, 0, buffer, PARAMS_SIZE);
        pass.bind_sampled(slot, 1, skyview);
        pass.bind_sampled(slot, 2, self.transmittance.view);
        pass.bind_sampler(slot, 3, self.lut_sampler);

        let colour = vk::RenderingAttachmentInfo::default()
            .image_view(targets.colour)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let velocity = vk::RenderingAttachmentInfo::default()
            .image_view(targets.velocity)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(targets.depth)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::NONE);
        let colours = [colour, velocity];
        let info = vk::RenderingInfo::default()
            .render_area(extent)
            .layer_count(1)
            .color_attachments(&colours)
            .depth_attachment(&depth);
        unsafe {
            device.cmd_begin_rendering(cmd, &info);
            device.cmd_set_viewport(cmd, 0, &[region.0]);
            device.cmd_set_scissor(cmd, 0, &[region.1]);
        }
        pass.draw_fullscreen(cmd, slot);
        unsafe { device.cmd_end_rendering(cmd) };
    }
}

/// The sky pass shader: the atmosphere maths, as build.rs prepends them.
fn compose_sky() -> String {
    compose(SKY_WGSL)
}

/// Prepend the shared atmosphere maths to a pass shader, as build.rs validates it.
fn compose(pass: &str) -> String {
    format!("{COMMON_WGSL}\n{pass}")
}

/// The environment-map shader: cloud maths, then atmosphere maths, as build.rs
/// prepends them to `atmo_envmap.wgsl`.
fn compose_envmap() -> String {
    format!("{CLOUD_COMMON_WGSL}\n{}", compose(ENVMAP_WGSL))
}

/// Allocate the sky environment map of every frame slot, move it to `GENERAL`
/// with all its levels, and build the passes that fill it.
fn build_sky_env(ctx: &Arc<GpuContext>) -> SkyEnv {
    let slots = ctx.max_frames_in_flight();
    let images: Vec<GpuImage> = (0..slots)
        .map(|slot| {
            ctx.create_image_mips(
                &format!("sky environment map {slot}"),
                SKY_ENV_SIZE.0,
                SKY_ENV_SIZE.1,
                LUT_FORMAT,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                SKY_ENV_MIPS,
            )
        })
        .collect();
    ctx.one_shot(|cmd| {
        for image in &images {
            unsafe {
                image_barrier_mips(
                    ctx.device(),
                    cmd,
                    image.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::ImageAspectFlags::COLOR,
                    SKY_ENV_MIPS,
                );
            }
        }
    });
    let level_views = images
        .iter()
        .map(|image| {
            (0..SKY_ENV_MIPS)
                .map(|level| {
                    let info = vk::ImageViewCreateInfo::default()
                        .image(image.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(LUT_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: level,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    unsafe { ctx.device().create_image_view(&info, None) }
                        .expect("sky environment map level view")
                })
                .collect()
        })
        .collect();
    let params = host_uniform_slots(ctx, "sky environment map params", ENV_PARAMS_SIZE);
    let map_pass = PostPass::new(
        ctx,
        "sky environment map",
        &compose_envmap(),
        c"main",
        &[
            BindKind::Uniform,
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Sampler,
            BindKind::Sampled,
            BindKind::Sampler,
            BindKind::Storage(LUT_FORMAT),
        ],
    );
    let reduce_passes = (1..SKY_ENV_MIPS)
        .map(|_| {
            PostPass::new(
                ctx,
                "sky environment map reduce",
                &compose(ENVMAP_DOWN_WGSL),
                c"main",
                &[BindKind::Sampled, BindKind::Storage(LUT_FORMAT)],
            )
        })
        .collect();
    SkyEnv {
        images,
        level_views,
        params,
        map_pass,
        reduce_passes,
        recorded: 0,
    }
}

impl Drop for Atmosphere {
    fn drop(&mut self) {
        // The passes and images drop after this; the GPU must be idle first so
        // no in-flight frame still samples them. The level views go with the
        // images they view, so they are destroyed here, before the images drop.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_sampler(self.lut_sampler, None);
            if let Some(env) = self.env.as_ref() {
                for views in &env.level_views {
                    for &view in views {
                        self.ctx.device().destroy_image_view(view, None);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atmosphere_params_block_matches_wgsl_layout() {
        // Ten vec4s, then three mat4s: 160 + 192 bytes.
        assert_eq!(std::mem::size_of::<AtmoParams>(), 352);
        assert_eq!(std::mem::offset_of!(AtmoParams, camera), 112);
        assert_eq!(std::mem::offset_of!(AtmoParams, inv_view_proj), 160);
        assert_eq!(std::mem::offset_of!(AtmoParams, prev_view_proj), 288);
    }
}

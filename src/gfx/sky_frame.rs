//! Per-frame orchestration of the sky and fog: lazy build and free on settings
//! changes, the sun-driven environment, and the recording of the sky and fog
//! passes into the scene pass's command buffer.

use super::Renderer;
use super::atmosphere::{Atmosphere, SkyCamera, SkyTargets, frame_params};
use super::cloud_shadow::ShadowGrid;
use super::cloud_view::ViewFrame;
use super::clouds::{CloudFrame, Clouds};
use super::fog::{Fog, FogFrame, FogTargets, params as fog_params};
use crate::camera::Camera;
use crate::lighting::LightingEnv;
use crate::render::{AtmosphereModel, AtmosphereSettings, SKYVIEW_SIZE};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};

/// The cloud shadow of one frame: the lighting block fields that read it, and its
/// map view (see `Renderer::record_cloud_shadow`).
#[derive(Clone, Copy)]
pub(crate) struct CloudShadowFrame {
    /// Centre (world x, z), 1 / extent, strength.
    pub grid: [f32; 4],
    /// Layer bottom and top (world y).
    pub layer: [f32; 4],
    pub view: vk::ImageView,
}

/// The CPU atmosphere and its last driven environment, so the sun-driven
/// lighting is recomputed only when the sun, the altitude or the settings move.
#[derive(Default)]
pub(crate) struct SkyCpu {
    model: Option<(AtmosphereSettings, AtmosphereModel)>,
    driven: Option<DrivenCache>,
}

struct DrivenCache {
    settings: AtmosphereSettings,
    sun: Vec3,
    sun_color: Vec3,
    altitude: f32,
    /// Cloud cover the sky colours were blended with.
    coverage: f32,
    /// The driven sun colour and sky colours, to be substituted into the caller's env.
    sun_out: [f32; 4],
    zenith: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
}

/// Coarse thresholds under which a cached driven environment is reused.
const DRIVE_SUN_DOT: f32 = 0.999_99;
const DRIVE_COLOUR_EPS: f32 = 1e-4;
/// Altitude bucket, metres, of the driven-environment cache.
const DRIVE_ALTITUDE_BUCKET: f32 = 10.0;

pub(super) fn barrier_all(device: &ash::Device, cmd: vk::CommandBuffer) {
    let memory = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
    }
}

impl Renderer {
    /// Bring the sky and fog resources in line with the render settings: free any
    /// that are off or stale (waiting the GPU idle first), and, when `build` is
    /// set, create the ones that are on. Called with `build` on each scene frame,
    /// so resources are created on first use.
    pub(super) fn sync_sky(&mut self, build: bool) {
        let want_atmosphere = self.sky_settings();
        let stale_atmosphere = match (&want_atmosphere, &self.atmosphere) {
            (Some(settings), Some(atmosphere)) => atmosphere.built_for() != settings,
            (None, Some(_)) => true,
            _ => false,
        };
        // The environment map exists only while the atmosphere feature is on, so a
        // change of that flag rebuilds too.
        let env_wanted = self.render_settings.atmosphere.is_some();
        let stale_atmosphere = stale_atmosphere
            || self
                .atmosphere
                .as_ref()
                .is_some_and(|atmosphere| atmosphere.has_env_map() != env_wanted);
        if stale_atmosphere {
            unsafe { self.ctx.device().device_wait_idle() }.expect("wait idle before sky rebuild");
            self.atmosphere = None;
        }
        if let (true, Some(settings), None) =
            (build, want_atmosphere.as_ref(), self.atmosphere.as_ref())
        {
            self.atmosphere = Some(Atmosphere::new(&self.ctx, settings, env_wanted));
        }

        let want_fog = self.render_settings.fog.clone();
        let stale_fog = match (&want_fog, &self.fog) {
            (Some(settings), Some(fog)) => fog.built_for() != settings,
            (None, Some(_)) => true,
            _ => false,
        };
        if stale_fog {
            unsafe { self.ctx.device().device_wait_idle() }.expect("wait idle before fog rebuild");
            self.fog = None;
        }
        if let (true, Some(settings), None) = (build, want_fog.as_ref(), self.fog.as_ref()) {
            self.fog = Some(Fog::new(&self.ctx, settings));
        }

        // Clouds follow their settings: a change re-targets the panorama and the view
        // passes (a new render size rebuilds them); off frees everything.
        match (self.render_settings.clouds.as_ref(), self.clouds.as_mut()) {
            (Some(settings), Some(clouds)) => clouds.apply(settings, self.render_size),
            (None, Some(_)) => self.clouds = None,
            _ => {}
        }
        if let (true, Some(settings), None) = (
            build,
            self.render_settings.clouds.as_ref(),
            self.clouds.as_ref(),
        ) {
            self.clouds = Some(Clouds::new(&self.ctx, settings, self.render_size));
        }
    }

    /// The atmosphere the sky is drawn with: the explicit setting or, when clouds
    /// are on without one, the default atmosphere with its lighting drive off.
    fn sky_settings(&self) -> Option<AtmosphereSettings> {
        self.render_settings.atmosphere.clone().or_else(|| {
            self.render_settings
                .clouds
                .as_ref()
                .map(|_| AtmosphereSettings {
                    drive_lighting: false,
                    ..AtmosphereSettings::default()
                })
        })
    }

    /// The environment the forward pass uses this frame. With the atmosphere's
    /// `drive_lighting` on, the sun colour and sky colours come from the CPU
    /// atmosphere; otherwise `env` passes through unchanged. Cached while the sun,
    /// altitude and settings are steady.
    pub(super) fn drive_env(&mut self, camera: &Camera, env: LightingEnv) -> LightingEnv {
        let Some(settings) = self.render_settings.atmosphere.clone() else {
            return env;
        };
        if !settings.drive_lighting {
            return env;
        }
        let sun = Vec3::from_slice(&env.sun_direction[..3]).normalize_or_zero();
        let sun_color = Vec3::from_slice(&env.sun_color[..3]);
        // The altitude is compared in 10 m buckets: the driven sky changes little
        // within one.
        let altitude = (camera.position.y - settings.sea_level) / DRIVE_ALTITUDE_BUCKET;
        let altitude = altitude.round() * DRIVE_ALTITUDE_BUCKET;
        let coverage = self
            .render_settings
            .clouds
            .as_ref()
            .map_or(0.0, |clouds| clouds.sanitised().coverage);
        let cpu = &mut self.sky_cpu;
        if let Some(cache) = cpu.driven.as_ref() {
            let steady = cache.settings == settings
                && cache.sun.dot(sun) > DRIVE_SUN_DOT
                && (cache.sun_color - sun_color).abs().max_element() < DRIVE_COLOUR_EPS
                && cache.altitude == altitude
                && cache.coverage == coverage;
            if steady {
                return LightingEnv {
                    sun_color: cache.sun_out,
                    sky_zenith: cache.zenith,
                    sky_horizon: cache.horizon,
                    sky_ground: cache.ground,
                    params: [env.params[0], 1.0, env.params[2], env.params[3]],
                    ..env
                };
            }
        }
        if !matches!(&cpu.model, Some((s, _)) if *s == settings) {
            cpu.model = Some((settings.clone(), AtmosphereModel::new(&settings)));
        }
        let Some((_, model)) = cpu.model.as_ref() else {
            return env;
        };
        let driven = model.driven_environment(&env, altitude.max(1.0), coverage);
        cpu.driven = Some(DrivenCache {
            settings: settings.clone(),
            sun,
            sun_color,
            altitude,
            coverage,
            sun_out: driven.sun_color,
            zenith: driven.sky_zenith,
            horizon: driven.sky_horizon,
            ground: driven.sky_ground,
        });
        LightingEnv {
            sun_color: driven.sun_color,
            sky_zenith: driven.sky_zenith,
            sky_horizon: driven.sky_horizon,
            sky_ground: driven.sky_ground,
            params: driven.params,
            ..env
        }
    }

    /// Record the sky pass, then the fog pass, for this frame's scene pass. Both
    /// run only when their resources exist. The sky reads `sky_env` (the caller's
    /// sun, unattenuated) and the fog `env` (sun-driven when that is on). Ends with
    /// a memory barrier, so the chains that follow read the final colour.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_sky_and_fog(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        camera: &Camera,
        jp: [f32; 2],
        region: (vk::Viewport, vk::Rect2D),
        full: vk::Rect2D,
        sky_env: &LightingEnv,
        env: &LightingEnv,
        reset: bool,
        cloud_shadow: Option<vk::ImageView>,
    ) {
        let (rw, rh) = self.render_size;
        let size = [rw as f32, rh as f32];
        let view = camera.view();
        let proj = camera.jittered_projection([0.0, 0.0], size);
        // Rotation-only view: the sky is at infinity, so camera translation drops out.
        let rot_view = Mat4::from_cols(view.x_axis, view.y_axis, view.z_axis, Vec4::W);
        let rot_vp = proj * rot_view;
        let prev_rot = if reset {
            rot_vp
        } else {
            self.prev_sky_view_proj
        };
        self.prev_sky_view_proj = rot_vp;
        // The previous frame's unjittered view-projection (the current one on a reset).
        let prev_vp = if reset {
            camera.view_projection()
        } else {
            self.prev_view_projection
        };

        let sky_settings = self.sky_settings();
        if let (Some(settings), Some(atmosphere)) =
            (sky_settings.as_ref(), self.atmosphere.as_mut())
        {
            let sky_camera = SkyCamera {
                position: camera.position,
                inv_view_proj: camera.jittered_view_projection(jp, size).inverse(),
                view_proj: rot_vp,
                prev_view_proj: prev_rot,
                size: (rw, rh),
                clouds: self.clouds.is_some(),
            };
            let block = frame_params(settings, sky_env, &sky_camera);
            self.pass_timer.mark(device, cmd, frame, "sky LUTs");
            atmosphere.record_skyview(cmd, frame, &block);
            // The panorama march reads this frame's sky-view LUT, so it follows the
            // LUT. It feeds the sky environment map only.
            self.pass_timer.mark(device, cmd, frame, "cloud panorama");
            let cloud_bind = self.clouds.as_mut().map(|clouds| {
                let inputs = CloudFrame {
                    camera: camera.position,
                    sea_level: settings.sea_level,
                    planet_radius: settings.planet_radius,
                    time: self.scene_time as f32,
                };
                clouds.record_panorama(cmd, frame, atmosphere.march_inputs(frame), &inputs, reset);
                (clouds.panorama_view(), clouds.sampler())
            });
            self.pass_timer.mark(device, cmd, frame, "env map");
            atmosphere.record_env_map(
                cmd,
                frame,
                env.sky_ground[..3].try_into().unwrap(),
                cloud_bind,
            );
            let targets = SkyTargets {
                colour: self.targets.color.view,
                velocity: self.targets.velocity.view,
                depth: self.targets.depth.view,
            };
            self.pass_timer.mark(device, cmd, frame, "sky pass");
            atmosphere.record_sky(device, cmd, frame, targets, region, full);
            // The per-pixel view clouds composite over the sky, before fog and the
            // transparent pass. They read the linear depth recorded before this pass.
            if let (Some(clouds), Some(depth)) = (
                self.clouds.as_mut(),
                self.linear_depth.as_ref().and_then(|d| d.now()),
            ) {
                let inputs = CloudFrame {
                    camera: camera.position,
                    sea_level: settings.sea_level,
                    planet_radius: settings.planet_radius,
                    time: self.scene_time as f32,
                };
                let forward = (view.inverse() * Vec4::new(0.0, 0.0, -1.0, 0.0))
                    .truncate()
                    .normalize_or_zero();
                let view_frame = ViewFrame {
                    inv_vp: camera.view_projection().inverse(),
                    prev_vp,
                    forward,
                };
                clouds.record_view(
                    cmd,
                    frame,
                    atmosphere.march_inputs(frame),
                    &inputs,
                    &view_frame,
                    depth,
                    self.targets.color.view,
                    region,
                    reset,
                    &mut |label| self.pass_timer.mark(device, cmd, frame, label),
                );
            }
            barrier_all(device, cmd);
        }

        let Some(fog_settings) = self.render_settings.fog.clone() else {
            return;
        };
        if self.fog.is_none() {
            return;
        }
        let cam_to_world = view.inverse();
        let ambient = (Vec3::from_slice(&env.sky_zenith[..3])
            + Vec3::from_slice(&env.sky_horizon[..3]))
            * 0.5;
        let fog_frame = FogFrame {
            cam_to_world,
            cam_pos: camera.position,
            tan_half_y: (camera.fov_y() * 0.5).tan(),
            aspect: camera.aspect(),
            near: camera.clip_planes().0,
            far: camera.clip_planes().1,
            prev_view_proj: prev_vp,
            sun_dir: Vec3::from_slice(&env.sun_direction[..3]),
            sun_color: Vec3::from_slice(&env.sun_color[..3]),
            ambient,
            time: self.scene_time as f32,
            frame: self.frame_id,
            reset,
        };
        let block = fog_params(&fog_settings, &fog_frame);
        let shadow = if self.render_settings.shadows.is_some() {
            self.shadows.as_ref().map(|map| map.image.view)
        } else {
            None
        };
        let targets = FogTargets {
            colour: self.targets.color.view,
            depth: self.targets.depth.view,
            shadow,
            cloud_shadow,
        };
        if let Some(fog) = self.fog.as_mut() {
            fog.record(device, cmd, frame, &block, &targets, region, &mut |label| {
                self.pass_timer.mark(device, cmd, frame, label);
            });
        }
        barrier_all(device, cmd);
    }

    /// Record the cloud shadow map of frame `frame` for the sun direction `sun` (to
    /// the sun), ahead of the scene pass, so the lighting and fog of this frame
    /// read it. `None` when the clouds are off or their shadow strength is zero;
    /// then the lighting and fog see no clouds on the sun.
    pub(super) fn record_cloud_shadow(
        &mut self,
        cmd: vk::CommandBuffer,
        frame: usize,
        camera: &Camera,
        sun: Vec3,
    ) -> Option<CloudShadowFrame> {
        let atmosphere = self.sky_settings()?;
        let frame_time = self.scene_time as f32;
        let clouds = self.clouds.as_mut()?;
        let settings = clouds.settings().clone();
        if settings.shadow_strength <= 0.0 {
            return None;
        }
        let grid = ShadowGrid::for_camera(camera.position);
        let cloud = CloudFrame {
            camera: camera.position,
            sea_level: atmosphere.sea_level,
            planet_radius: atmosphere.planet_radius,
            time: frame_time,
        };
        clouds.record_shadow(cmd, frame, &cloud, &grid, sun);
        let base = atmosphere.sea_level + settings.base_altitude;
        Some(CloudShadowFrame {
            grid: [
                grid.centre[0],
                grid.centre[1],
                1.0 / grid.extent,
                settings.shadow_strength,
            ],
            layer: [base, base + settings.thickness, 0.0, 0.0],
            view: clouds.shadow_view(frame),
        })
    }

    /// Float copy of the sky-view LUT the last scene frame wrote: four floats per
    /// texel (radiance rgb, 1), texel row 0 first (uv.y near 0), with the width and
    /// height. Empty
    /// when the atmosphere is off or never built.
    pub fn capture_sky_view_rgba16f(&mut self) -> (Vec<f32>, u32, u32) {
        let Some(slot) = self.last_scene_slot() else {
            return (Vec::new(), 0, 0);
        };
        let Some(atmosphere) = self.atmosphere.as_ref() else {
            return (Vec::new(), 0, 0);
        };
        let image = atmosphere.skyview_image(slot);
        let (w, h) = SKYVIEW_SIZE;
        let raw = self.read_half_image(image, (w, h, 1), "sky-view readback");
        (raw, w, h)
    }

    /// Float copy of the integrated fog volume of the last scene frame, four floats
    /// per froxel (accumulated in-scatter rgb, transmittance a), x fastest, then y,
    /// then slice. `None` when the fog is off.
    pub fn capture_fog_volume(&mut self) -> Option<(Vec<f32>, (u32, u32, u32))> {
        let fog = self.fog.as_ref()?;
        let size = fog.size();
        let image = fog.integrated_image();
        let raw = self.read_half_image(image, size, "fog volume readback");
        Some((raw, size))
    }

    /// Float copy of the inject output of the last scene frame: per-froxel
    /// scattering per metre (rgb) and extinction per metre (a), in the same layout
    /// as [`Self::capture_fog_volume`]. `None` when the fog is off.
    pub fn capture_fog_injected(&mut self) -> Option<(Vec<f32>, (u32, u32, u32))> {
        let fog = self.fog.as_ref()?;
        let size = fog.size();
        let image = fog.injected_image();
        let raw = self.read_half_image(image, size, "fog inject readback");
        Some((raw, size))
    }

    /// Float copy of the cloud panorama of the last scene frame: four floats per
    /// texel (in-scattered cloud radiance rgb, transmittance a), texel row 0 at the
    /// zenith (uv.y 0) and the last row at the horizon, with the width and height.
    /// Empty when the clouds are off.
    pub fn capture_cloud_panorama(&mut self) -> Option<(Vec<f32>, u32, u32)> {
        let (image, (w, h)) = self.clouds.as_ref()?.panorama_image();
        let raw = self.read_half_image(image, (w, h, 1), "cloud panorama readback");
        Some((raw, w, h))
    }

    /// The slot the last recorded scene frame used.
    pub(super) fn last_scene_slot(&self) -> Option<usize> {
        let slots = self.ctx.max_frames_in_flight();
        (slots > 0).then(|| (self.frame + slots - 1) % slots)
    }
}

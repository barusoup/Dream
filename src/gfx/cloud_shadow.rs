//! The cloud shadow map: the sun's transmittance through the cloud layer, on a
//! square of the ground centred near the camera.
//!
//! One compute pass per frame slot (`cloud_shadow.wgsl`), recorded before the
//! scene pass so the lighting and the fog of the same frame read it. Each texel
//! marches the sun ray through the layer (see `cloud_sun_transmittance`). The
//! square is snapped to whole texels, so the shadows do not swim as the camera
//! moves. Lighting multiplies the sun by the map (`lighting_cloud_visibility`).

use super::clouds::{CloudParams, PARAMS_SIZE, compose};
use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use ash::vk;
use glam::Vec3;
use std::sync::Arc;

const WGSL: &str = include_str!("cloud_shadow.wgsl");
/// Texels per side of the map.
pub(crate) const SHADOW_SIZE: u32 = 256;
/// Side of the square the map covers, metres.
pub(crate) const SHADOW_EXTENT: f32 = 16_000.0;
/// Storage format of the map.
const SHADOW_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
/// Size of the `ShadowParams` block.
const SHADOW_PARAMS_SIZE: u64 = std::mem::size_of::<ShadowParams>() as u64;

/// CPU mirror of `ShadowParams` in `cloud_shadow.wgsl`: two vec4s.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowParams {
    /// x, y: centre (world x, z), z: extent (m), w: texels per side.
    grid: [f32; 4],
    /// xyz: direction to the sun, w: sea level (m).
    sun: [f32; 4],
}

/// Where the shadow map lies this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShadowGrid {
    /// Centre of the square on the ground (world x, z), a whole number of texels.
    pub centre: [f32; 2],
    /// Side of the square, metres.
    pub extent: f32,
}

impl ShadowGrid {
    /// The square for a camera at `camera`, snapped to the texel grid.
    pub(crate) fn for_camera(camera: Vec3) -> Self {
        let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
        Self {
            centre: [
                (camera.x / texel).round() * texel,
                (camera.z / texel).round() * texel,
            ],
            extent: SHADOW_EXTENT,
        }
    }
}

/// The shadow maps, one per frame slot, and their pass.
pub(crate) struct CloudShadow {
    maps: Vec<GpuImage>,
    /// Per slot: the `CloudParams` block and the `ShadowParams` block.
    cloud: Vec<GpuBuffer>,
    params: Vec<GpuBuffer>,
    pass: PostPass,
}

impl CloudShadow {
    /// Build the maps (one per frame slot) and the pass.
    pub(crate) fn new(ctx: &Arc<GpuContext>) -> Self {
        let maps = (0..ctx.max_frames_in_flight())
            .map(|slot| {
                create_general_image(
                    ctx,
                    &format!("cloud shadow map {slot}"),
                    (SHADOW_SIZE, SHADOW_SIZE),
                    SHADOW_FORMAT,
                )
            })
            .collect();
        let pass = PostPass::new(
            ctx,
            "cloud shadow",
            &compose(WGSL),
            c"main",
            &[
                BindKind::Uniform,
                BindKind::Uniform,
                BindKind::Sampled3d,
                BindKind::Sampled3d,
                BindKind::Sampler,
                BindKind::Storage(SHADOW_FORMAT),
            ],
        );
        Self {
            maps,
            cloud: host_uniform_slots(ctx, "cloud shadow cloud params", PARAMS_SIZE),
            params: host_uniform_slots(ctx, "cloud shadow params", SHADOW_PARAMS_SIZE),
            pass,
        }
    }

    /// The map of frame `slot`, a `GENERAL` `R32_SFLOAT` view.
    pub(crate) fn view(&self, slot: usize) -> vk::ImageView {
        self.maps[slot % self.maps.len()].view
    }

    /// Record the map of frame `slot` for the sun direction `sun` (to the sun), with
    /// the cloud block `cp` (the camera of this frame) and the square `grid`.
    /// `textures` are the noise volumes' base and detail views and their sampler.
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        cp: &CloudParams,
        grid: &ShadowGrid,
        sun: Vec3,
        textures: (vk::ImageView, vk::ImageView, vk::Sampler),
    ) {
        let slots = self.cloud.len();
        let (base, detail, sampler) = textures;
        let block = ShadowParams {
            grid: [
                grid.centre[0],
                grid.centre[1],
                grid.extent,
                SHADOW_SIZE as f32,
            ],
            sun: [sun.x, sun.y, sun.z, cp.cam[3]],
        };
        self.cloud[slot % slots].write(0, bytemuck::bytes_of(cp));
        self.params[slot % slots].write(0, bytemuck::bytes_of(&block));
        let pass = &self.pass;
        pass.bind_uniform(slot, 0, &self.cloud[slot % slots], PARAMS_SIZE);
        pass.bind_uniform(slot, 1, &self.params[slot % slots], SHADOW_PARAMS_SIZE);
        pass.bind_sampled(slot, 2, base);
        pass.bind_sampled(slot, 3, detail);
        pass.bind_sampler(slot, 4, sampler);
        pass.bind_storage(slot, 5, self.view(slot), SHADOW_FORMAT);
        pass.dispatch(cmd, slot, SHADOW_SIZE, SHADOW_SIZE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_shadow_grid_snaps_to_texels() {
        let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
        let a = ShadowGrid::for_camera(Vec3::new(1000.0, 50.0, -3000.0));
        // Whole texels, nearest to the camera.
        assert_eq!(a.centre[0] % texel, 0.0);
        assert_eq!(a.centre[1] % texel, 0.0);
        assert!((a.centre[0] - 1000.0).abs() <= texel / 2.0 + 1e-3);
        // Moving within a texel does not move the square.
        let b = ShadowGrid::for_camera(Vec3::new(1010.0, 50.0, -3000.0));
        assert_eq!(a, b);
    }

    #[test]
    fn cloud_shadow_params_block_matches_wgsl_layout() {
        assert_eq!(SHADOW_PARAMS_SIZE, 32);
    }
}

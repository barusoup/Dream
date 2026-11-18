//! Sun shadow maps: one depth image with a layer per cascade, a view per
//! layer, and a camera set per (cascade, frame slot) so the scene shaders run
//! unchanged in the shadow passes. The passes are recorded ahead of the main
//! scene pass; the matrices they render with are published to the lighting
//! block (see [`super::Renderer::lighting_frame`]).

use super::pipeline::{depth_attachment_in, make_viewport, uniform_write, write_sets};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage};
use super::{DrawCtx, Renderer, ScenePass, ViewUniform};
use crate::camera::Camera;
use crate::lighting::{LightingEnv, LightingFrame};
use crate::render::{CascadeFit, ShadowFilter, ShadowSettings, fit_cascades};
use ash::vk;
use bytemuck::Zeroable;
use std::sync::Arc;

/// Most cascades a shadow map holds.
const MAX_CASCADES: usize = 4;

/// The GPU side of the sun shadow map. Built on the first scene frame that
/// needs it and dropped when the settings that size it change.
pub(super) struct ShadowMap {
    ctx: Arc<GpuContext>,
    pub(super) resolution: u32,
    pub(super) cascades: u32,
    /// `D16_UNORM` (or `D32_SFLOAT` fallback) array, one layer per cascade.
    /// Its `view` is the array view the lighting set samples.
    pub(super) image: GpuImage,
    /// One single-layer view per cascade, used as the pass's depth target.
    layer_views: Vec<vk::ImageView>,
    pool: vk::DescriptorPool,
    /// `scene_sets[cascade][slot]`: the camera set of that pass.
    scene_sets: Vec<Vec<vk::DescriptorSet>>,
    /// Backing buffers of `scene_sets`, same indexing.
    uniforms: Vec<Vec<GpuBuffer>>,
    /// The fit each cascade's contents were last rendered with. Staggered
    /// cascades keep theirs between updates, so the lighting block always
    /// matches the map.
    fits: Vec<Option<CascadeFit>>,
}

impl ShadowMap {
    pub(super) fn new(
        ctx: &Arc<GpuContext>,
        camera_layout: vk::DescriptorSetLayout,
        resolution: u32,
        cascades: u32,
    ) -> Self {
        let device = ctx.device();
        let format = ctx.shadow_depth_format();
        let cascades = cascades.clamp(1, MAX_CASCADES as u32);
        let image = ctx.create_image_array_mips(
            "sun shadow map",
            resolution,
            resolution,
            format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            cascades,
            1,
        );
        ctx.one_shot(|cmd| unsafe {
            map_barrier(
                device,
                cmd,
                image.image,
                cascades,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
            );
        });
        let layer_views: Vec<_> = (0..cascades)
            .map(|layer| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(image.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&info, None) }.expect("shadow layer view")
            })
            .collect();

        let slots = ctx.max_frames_in_flight();
        let count = cascades as usize * slots;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(count as u32);
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count as u32)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )
        }
        .expect("shadow descriptor pool");
        let layouts = vec![camera_layout; count];
        let flat = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("shadow camera sets");

        let mut scene_sets = Vec::with_capacity(cascades as usize);
        let mut uniforms = Vec::with_capacity(cascades as usize);
        for cascade in 0..cascades as usize {
            let mut sets = Vec::with_capacity(slots);
            let mut buffers = Vec::with_capacity(slots);
            for slot in 0..slots {
                let set = flat[cascade * slots + slot];
                let buffer = ctx.create_host_buffer(
                    &format!("shadow camera {cascade}.{slot}"),
                    std::mem::size_of::<ViewUniform>() as u64,
                );
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .range(std::mem::size_of::<ViewUniform>() as u64);
                write_sets(ctx, &[uniform_write(set, 0, &info)]);
                sets.push(set);
                buffers.push(buffer);
            }
            scene_sets.push(sets);
            uniforms.push(buffers);
        }

        Self {
            ctx: Arc::clone(ctx),
            resolution,
            cascades,
            image,
            layer_views,
            pool,
            scene_sets,
            uniforms,
            fits: vec![None; cascades as usize],
        }
    }
}

impl Drop for ShadowMap {
    fn drop(&mut self) {
        // The renderer waits for the GPU before dropping a map.
        let device = self.ctx.device();
        unsafe {
            for view in self.layer_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Image barrier over every layer of a shadow map. The shared barrier helpers
/// only cover layer 0, which would leave the other cascades untransitioned.
#[allow(clippy::too_many_arguments)]
unsafe fn map_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    layers: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Whether cascade `c` renders this frame. Staggering updates cascade 2 every
/// second frame and cascade 3 every fourth; `all` forces every cascade.
fn cascade_updated(c: usize, frame_id: u64, stagger: bool, all: bool) -> bool {
    if all || !stagger {
        return true;
    }
    match c {
        0 | 1 => true,
        2 => frame_id.is_multiple_of(2),
        _ => frame_id.is_multiple_of(4),
    }
}

impl Renderer {
    /// Drop the shadow map when the settings no longer match its size, filter
    /// or existence. Creation waits for a scene frame ([`Self::ensure_shadow_map`]),
    /// so a settings change allocates nothing up front.
    pub(super) fn drop_stale_shadow_map(&mut self) {
        let wanted = self
            .render_settings
            .shadows
            .as_ref()
            .map(|s| (s.resolution, s.cascades));
        let current = self.shadows.as_ref().map(|m| (m.resolution, m.cascades));
        if current.is_none() || current == wanted {
            return;
        }
        self.quiesce_gpu();
        self.shadows = None;
        self.shadow_generation += 1;
        self.acquire.respawn();
    }

    /// Create the shadow map when shadows are on and it does not exist yet.
    pub(super) fn ensure_shadow_map(&mut self) {
        let Some((resolution, cascades)) = self
            .render_settings
            .shadows
            .as_ref()
            .map(|s| (s.resolution, s.cascades))
        else {
            return;
        };
        if self.shadows.is_some() {
            return;
        }
        self.shadows = Some(ShadowMap::new(
            &self.ctx,
            self.scene_dsl,
            resolution,
            cascades,
        ));
        self.shadow_generation += 1;
        self.shadow_all_dirty = true;
    }

    /// Record this frame's shadow passes into `cmd`, ahead of the main pass.
    /// The game's `draw` runs once per cascade that updates, with
    /// [`ScenePass::Shadow`]. Updated cascades store their fit for
    /// [`Self::lighting_frame`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_shadow_passes(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        camera: &Camera,
        env: &LightingEnv,
        settings: &ShadowSettings,
        draw: &mut dyn FnMut(&mut DrawCtx<'_>) -> usize,
    ) {
        let sun = glam::Vec3::new(
            env.sun_direction[0],
            env.sun_direction[1],
            env.sun_direction[2],
        );
        let (near, far) = camera.clip_planes();
        let fits = fit_cascades(
            camera.view(),
            camera.view_projection(),
            near,
            far.min(settings.distance),
            sun,
            settings,
        );
        let all = self.shadow_all_dirty;
        let frame_id = self.frame_id;
        let scene_pipeline = self.shadow_scene_pipeline;
        let scene_layout = self.scene_layout;
        let pulling = match (self.shadow_pulling_pipeline, self.pulling) {
            (Some(pipeline), Some((_, layout, _))) => Some((pipeline, layout)),
            _ => None,
        };
        let Some(map) = self.shadows.as_mut() else {
            return;
        };
        let res = map.resolution;
        let extent = vk::Extent2D {
            width: res,
            height: res,
        };
        let image = map.image.image;
        let layers = map.cascades;
        // Last frame's shadow reads must finish before this frame writes the map.
        unsafe {
            map_barrier(
                device,
                cmd,
                image,
                layers,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        }

        for (c, fit) in fits.iter().enumerate().take(map.cascades as usize) {
            if !cascade_updated(c, frame_id, settings.stagger_far_cascades, all) {
                continue;
            }
            // This slot's fence has recycled, so its camera buffer is free to write.
            let view = ViewUniform {
                view_projection: fit.view_projection.to_cols_array_2d(),
                curr_view_projection: fit.view_projection.to_cols_array_2d(),
                prev_view_projection: fit.view_projection.to_cols_array_2d(),
            };
            map.uniforms[c][frame].write(0, bytemuck::bytes_of(&view));

            let depth = depth_attachment_in(
                map.layer_views[c],
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                1.0,
            );
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                })
                .layer_count(1)
                .depth_attachment(&depth);
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[make_viewport((res, res))]);
                device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    }],
                );
                device.cmd_set_depth_bias(cmd, settings.depth_bias, 0.0, settings.slope_bias);
            }
            let mut ctx = DrawCtx {
                device,
                cmd,
                scene_pipeline,
                scene_layout,
                scene_set: map.scene_sets[c][frame],
                pulling,
                frame_slot: frame,
                frame_id,
                starter: std::cell::RefCell::new(None),
                pass: ScenePass::Shadow { cascade: c as u32 },
                has_transparent: true,
                view_proj: fit.view_projection,
                occlusion: None,
            };
            draw(&mut ctx);
            unsafe { device.cmd_end_rendering(cmd) };
            map.fits[c] = Some(*fit);
        }

        // The map becomes readable by the fog and main passes.
        unsafe {
            map_barrier(
                device,
                cmd,
                image,
                layers,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
        }
        if all {
            self.shadow_all_dirty = false;
        }
    }

    /// The per-frame lighting block for the frame being recorded. Shadow fields
    /// come from the stored fits, so call this after the shadow passes.
    pub(super) fn lighting_frame(
        &self,
        camera: &Camera,
        settings: Option<&ShadowSettings>,
    ) -> LightingFrame {
        let mut frame = LightingFrame::zeroed();
        if let (Some(map), Some(settings)) = (self.shadows.as_ref(), settings) {
            for (c, fit) in map.fits.iter().enumerate().take(map.cascades as usize) {
                if let Some(fit) = fit {
                    frame.cascade_view_proj[c] = fit.view_projection.to_cols_array_2d();
                    frame.cascade_params[c] = [fit.texel_size, fit.depth_range, 0.0, 0.0];
                }
            }
            let filter = match settings.filter {
                ShadowFilter::Hard => 0.0,
                ShadowFilter::Pcf => 1.0,
                ShadowFilter::Soft => 2.0,
            };
            frame.shadow = [
                map.cascades as f32,
                1.0 / map.resolution as f32,
                filter,
                settings.taps as f32,
            ];
            frame.shadow2 = [
                settings.softness,
                settings.normal_bias,
                settings.strength,
                if settings.debug_cascades { 1.0 } else { 0.0 },
            ];
            frame.shadow_fade = [
                settings.distance * (1.0 - settings.fade),
                settings.distance,
                0.0,
                0.0,
            ];
        }
        let (w, h) = (
            self.render_size.0.max(1) as f32,
            self.render_size.1.max(1) as f32,
        );
        frame.screen = [w, h, 1.0 / w, 1.0 / h];
        // Later phases read these; neutral strengths until they exist.
        // The shadow filter's noise is animated only under temporal AA, which
        // integrates it; without that the index stays 0 so penumbrae do not flicker.
        let animated = self.temporal || self.taa_active();
        let noise_frame = if animated {
            (self.frame_id % 65536) as f32
        } else {
            0.0
        };
        // Indirect strengths are zero unless the chain ran last frame and that
        // frame's result is the one bound, so a stale result is never shaded.
        let chain_ready = self.history_valid
            && self
                .indirect
                .as_ref()
                .is_some_and(|chain| chain.output().is_some());
        let ao = self.render_settings.ao.as_ref().filter(|_| chain_ready);
        let gi = self.render_settings.gi.as_ref().filter(|_| chain_ready);
        // The reflection strength follows the same rule for its own chain.
        let reflection_ready = self.history_valid
            && self
                .reflection
                .as_ref()
                .is_some_and(|chain| chain.output().is_some());
        let reflections = self
            .render_settings
            .reflections
            .as_ref()
            .filter(|_| reflection_ready);
        frame.features = [
            ao.map_or(0.0, |a| a.intensity),
            gi.map_or(0.0, |g| g.intensity),
            reflections.map_or(0.0, |r| r.intensity),
            noise_frame,
        ];
        frame.indirect = [
            ao.map_or(0.0, |a| if a.multi_bounce { 1.0 } else { 0.0 }),
            ao.map_or(0.0, |a| if a.specular_occlusion { 1.0 } else { 0.0 }),
            0.0,
            0.0,
        ];
        let prev = if self.history_valid {
            self.prev_view_projection
        } else {
            camera.view_projection()
        };
        frame.prev_view_proj = prev.to_cols_array_2d();
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::cascade_updated;

    #[test]
    fn shadows_stagger_updates_far_cascades_less_often() {
        let updates = |c: usize| {
            (0..8u64)
                .filter(|&f| cascade_updated(c, f, true, false))
                .count()
        };
        assert_eq!(updates(0), 8);
        assert_eq!(updates(1), 8);
        assert_eq!(updates(2), 4);
        assert_eq!(updates(3), 2);
        // Without stagger, or when forced, everything renders every frame.
        assert!((0..8u64).all(|f| cascade_updated(3, f, false, false)));
        assert!((0..8u64).all(|f| cascade_updated(3, f, true, true)));
    }
}

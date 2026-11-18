//! Shared PBR lighting: one directional sun plus an analytic sky, evaluated by
//! the WGSL in [`SHADER`] which is concatenated into consumer shaders before
//! `compile_wgsl` (the engine's WGSL-to-SPIR-V step). The environment is a uniform at
//! **set 3 binding 0** for every pipeline that uses it, so one
//! [`Lighting`] instance serves all consumers. Set 3 also carries the per-frame
//! `LightingFrame` block, the sun shadow map and the reserved GI/AO and
//! reflection inputs and the sky environment map (see the binding table on
//! [`Lighting`]).
//!
//! [`LightingEnv`] is the CPU mirror of the uniform block — all vec4s, so
//! alignment matches WGSL's `vec4` fields.

use crate::gfx::vulkan::{self, GpuContext};
use crate::gfx::{GpuBuffer, GpuImage, SCENE_FORMAT};
use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The WGSL snippet; prepend to any shader that calls `lighting_eval`.
pub const SHADER: &str = include_str!("lighting.wgsl");

/// Descriptor set index the lighting uniform binds at in every consumer.
pub const SET_INDEX: usize = 3;
/// Binding index within that set.
pub const BINDING: u32 = 0;

/// Bindings of set 3, all visible to the fragment stage. Their order and types
/// are the contract with `lighting.wgsl`.
const BINDING_ENV: u32 = 0;
const BINDING_FRAME: u32 = 1;
const BINDING_SHADOW_MAP: u32 = 2;
const BINDING_SHADOW_SAMPLER: u32 = 3;
const BINDING_GI_AO: u32 = 4;
const BINDING_REFLECTION: u32 = 5;
const BINDING_LINEAR: u32 = 6;
const BINDING_FEEDBACK_DEPTH: u32 = 7;
const BINDING_ENV_MAP: u32 = 8;
const BINDING_ENV_SAMPLER: u32 = 9;
const BINDING_CLOUD_SHADOW: u32 = 10;
/// Format of the sky environment map (see `gfx::atmosphere`).
const ENV_MAP_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// The previous frame's screen-space results, as bound to one slot's set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenSpaceViews {
    /// Indirect diffuse (rgb) and visibility (a), binding 4. `None` binds the
    /// neutral dummy.
    pub gi: Option<vk::ImageView>,
    /// Traced reflection (rgb) and confidence (a), binding 5. `None` binds the
    /// zero dummy.
    pub reflection: Option<vk::ImageView>,
    /// The linear depth both were computed from, binding 7.
    pub depth: vk::ImageView,
}

/// Chain generation and the views a slot's bindings 4, 5 and 7 hold.
type ScreenKey = (u64, Option<ScreenSpaceViews>);
/// Sky environment map generation and view a slot's binding 8 holds; `None`
/// is the dummy.
type EnvKey = (u64, Option<vk::ImageView>);
/// Depth stored for sky and for the neutral dummy: far beyond any scene depth.
pub(crate) const FEEDBACK_SKY_DEPTH: f32 = 65000.0;

/// CPU mirror of `LightingEnv` in `lighting.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingEnv {
    /// xyz: direction TO the sun (normalized); w unused.
    pub sun_direction: [f32; 4],
    /// rgb: sun colour × intensity.
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    /// xyz: world-space camera position (shading point's view vector).
    pub cam_pos: [f32; 4],
    /// x: exposure. y: 1 when the sky colours are sun-driven irradiance
    /// (`AtmosphereModel::driven_environment`), 0 when they are a radiance gradient.
    pub params: [f32; 4],
}

impl Default for LightingEnv {
    /// Day-ish sky with a warm sun 45° up — matches the lab's defaults.
    fn default() -> Self {
        Self::with_sun(0.6, 0.7, 3.0)
    }
}

impl LightingEnv {
    /// Build with the sun at `azimuth`/`elevation` radians and `intensity`.
    pub fn with_sun(azimuth: f32, elevation: f32, intensity: f32) -> Self {
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
            0.0,
        ];
        Self {
            sun_direction: dir,
            sun_color: [intensity, intensity * 0.96, intensity * 0.88, 0.0],
            sky_zenith: [0.18, 0.35, 0.65, 0.0],
            sky_horizon: [0.55, 0.60, 0.68, 0.0],
            sky_ground: [0.10, 0.09, 0.08, 0.0],
            cam_pos: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

/// CPU mirror of `LightingFrame` in `lighting.wgsl`, written per frame by the
/// renderer. Field order and offsets match the WGSL struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightingFrame {
    /// World to light clip space, per cascade.
    pub cascade_view_proj: [[[f32; 4]; 4]; 4],
    /// x: texel world size, y: depth range in metres.
    pub cascade_params: [[f32; 4]; 4],
    /// x: cascade count (0 = shadows off), y: 1/resolution, z: filter, w: taps.
    pub shadow: [f32; 4],
    /// x: softness, y: normal bias (texels), z: strength, w: debug cascades.
    pub shadow2: [f32; 4],
    /// x: fade start (m), y: fade end (m).
    pub shadow_fade: [f32; 4],
    /// Render width, height, 1/width, 1/height.
    pub screen: [f32; 4],
    /// x: AO strength, y: GI strength, z: reflection strength, w: frame index.
    pub features: [f32; 4],
    /// Previous unjittered view-projection.
    pub prev_view_proj: [[f32; 4]; 4],
    /// x: multi-bounce AO (1 on), y: specular occlusion (1 on), z: sky
    /// environment map bound and valid (1), w: its last mip.
    pub indirect: [f32; 4],
    /// x, y: centre of the cloud shadow map on the ground (world x, z), z: 1 / its
    /// extent (1/m), w: strength (0 when the map is off).
    pub cloud_shadow: [f32; 4],
    /// x, y: cloud layer bottom and top (world y, metres). The shadow fades out
    /// between them.
    pub cloud_layer: [f32; 4],
}

/// Per-frame-slot lighting state shared by every consumer pipeline.
///
/// Bindings of set 3 (all fragment-stage):
/// - 0: [`LightingEnv`] uniform.
/// - 1: `LightingFrame` uniform.
/// - 2: sun shadow map (`texture_depth_2d_array`, one layer per cascade).
/// - 3: comparison sampler for the shadow map (clamp to white border).
/// - 4: `lighting_gi_ao` (indirect diffuse rgb, visibility a; see
///   `Lighting::bind_screen_space` (crate-internal)). Neutral (0,0,0,1) until bound.
/// - 5: `lighting_reflection` (traced reflection rgb, confidence a). Zero until bound.
/// - 6: linear clamp sampler.
/// - 7: `lighting_feedback_depth` (the linear depth that bindings 4 and 5 were
///   computed from; may be finer than either). Neutral: a large sentinel until bound.
/// - 8: `lighting_env_map`, the previous frame's sky environment map with its full
///   mip chain. A one-texel dummy until bound (see `Self::bind_env_map`).
/// - 9: trilinear sampler for it: repeat in u, clamp in v, mips 0 to
///   `SKY_ENV_MIPS - 1`.
/// - 10: `lighting_cloud_shadow`, the sun transmittance through the cloud layer
///   on the ground (R32F, sampled with binding 6). Bound per frame slot by
///   `Self::bind_cloud_shadow`; a one-texel dummy otherwise.
///
/// Until `Self::bind_shadow_map` binds a real map, binding 2 is a one-layer
/// dummy cleared to depth 1.0 (fully lit), so existing users never see an
/// unbound slot.
pub struct Lighting {
    ctx: Arc<GpuContext>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    buffers: Vec<Mutex<GpuBuffer>>,
    frame_buffers: Vec<Mutex<GpuBuffer>>,
    shadow_compare: vk::Sampler,
    linear: vk::Sampler,
    dummy_shadow: GpuImage,
    /// Kept alive for the sets that view them.
    _dummy_gi_ao: GpuImage,
    _dummy_reflection: GpuImage,
    dummy_depth: GpuImage,
    /// Per slot, the chain generation and views the set points at, so an
    /// unchanged binding is not rewritten every frame.
    screen_bound: Mutex<Vec<ScreenKey>>,
    /// Per slot, the sky environment map generation and view bound at binding 8.
    env_bound: Mutex<Vec<EnvKey>>,
    env_sampler: vk::Sampler,
    /// One texel, bound at binding 8 until a sky environment map is.
    dummy_env: GpuImage,
    /// One texel, bound at binding 10 (the cloud shadow map of a slot replaces it).
    dummy_cloud: GpuImage,
    /// Shadow generation the sets currently point at (see `bind_shadow_map`).
    bound_generation: AtomicU64,
}

impl Lighting {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let device = ctx.device();
        let entry = |binding: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        };
        let bindings = [
            entry(BINDING_ENV, vk::DescriptorType::UNIFORM_BUFFER),
            entry(BINDING_FRAME, vk::DescriptorType::UNIFORM_BUFFER),
            entry(BINDING_SHADOW_MAP, vk::DescriptorType::SAMPLED_IMAGE),
            entry(BINDING_SHADOW_SAMPLER, vk::DescriptorType::SAMPLER),
            entry(BINDING_GI_AO, vk::DescriptorType::SAMPLED_IMAGE),
            entry(BINDING_REFLECTION, vk::DescriptorType::SAMPLED_IMAGE),
            entry(BINDING_LINEAR, vk::DescriptorType::SAMPLER),
            entry(BINDING_FEEDBACK_DEPTH, vk::DescriptorType::SAMPLED_IMAGE),
            entry(BINDING_ENV_MAP, vk::DescriptorType::SAMPLED_IMAGE),
            entry(BINDING_ENV_SAMPLER, vk::DescriptorType::SAMPLER),
            entry(BINDING_CLOUD_SHADOW, vk::DescriptorType::SAMPLED_IMAGE),
        ];
        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("lighting descriptor set layout");

        let slots = ctx.max_frames_in_flight();
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2 * slots as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(6 * slots as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(3 * slots as u32),
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(slots as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .expect("lighting descriptor pool");
        let layouts: Vec<_> = (0..slots).map(|_| layout).collect();
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("lighting descriptor sets");

        let shadow_compare = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                None,
            )
        }
        .expect("shadow comparison sampler");
        let linear = ctx.create_sampler("lighting linear sampler");
        let env_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod((crate::render::SKY_ENV_MIPS - 1) as f32),
                None,
            )
        }
        .expect("sky environment map sampler");
        let dummy_env = ctx.create_image(
            "lighting env map dummy",
            1,
            1,
            ENV_MAP_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );
        // Read only when the cloud shadow strength is above zero.
        let dummy_cloud = ctx.create_image(
            "lighting cloud shadow dummy",
            1,
            1,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );

        // Neutral dummies: depth 1.0 (lit), indirect (0,0,0,1) and zero reflection.
        let dummy_shadow = ctx.create_image_array_mips(
            "lighting shadow dummy",
            1,
            1,
            ctx.shadow_depth_format(),
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST,
            1,
            1,
        );
        let dummy_gi_ao = ctx.create_image(
            "lighting gi/ao dummy",
            1,
            1,
            SCENE_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );
        let dummy_reflection = ctx.create_image(
            "lighting reflection dummy",
            1,
            1,
            SCENE_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );
        let dummy_depth = ctx.create_image(
            "lighting feedback depth dummy",
            1,
            1,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );
        ctx.one_shot(|cmd| unsafe {
            let colour = vk::ImageAspectFlags::COLOR;
            let depth = vk::ImageAspectFlags::DEPTH;
            for (image, aspect) in [
                (dummy_shadow.image, depth),
                (dummy_gi_ao.image, colour),
                (dummy_reflection.image, colour),
                (dummy_depth.image, colour),
                (dummy_env.image, colour),
                (dummy_cloud.image, colour),
            ] {
                vulkan::image_barrier(
                    device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    aspect,
                );
            }
            let range = |aspect| vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            device.cmd_clear_depth_stencil_image(
                cmd,
                dummy_shadow.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
                &[range(depth)],
            );
            device.cmd_clear_color_image(
                cmd,
                dummy_gi_ao.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
                &[range(colour)],
            );
            device.cmd_clear_color_image(
                cmd,
                dummy_reflection.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                &[range(colour)],
            );
            device.cmd_clear_color_image(
                cmd,
                dummy_depth.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    float32: [FEEDBACK_SKY_DEPTH, 0.0, 0.0, 0.0],
                },
                &[range(colour)],
            );
            device.cmd_clear_color_image(
                cmd,
                dummy_env.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                &[range(colour)],
            );
            for (image, aspect) in [
                (dummy_shadow.image, depth),
                (dummy_gi_ao.image, colour),
                (dummy_reflection.image, colour),
                (dummy_depth.image, colour),
                (dummy_env.image, colour),
                (dummy_cloud.image, colour),
            ] {
                let target_layout = if image == dummy_shadow.image {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::GENERAL
                };
                vulkan::image_barrier(
                    device,
                    cmd,
                    image,
                    vk::ImageLayout::GENERAL,
                    target_layout,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::MEMORY_READ,
                    aspect,
                );
            }
        });

        let env_size = std::mem::size_of::<LightingEnv>() as u64;
        let frame_size = std::mem::size_of::<LightingFrame>() as u64;
        let buffers: Vec<_> = (0..slots)
            .map(|slot| {
                Mutex::new(ctx.create_host_buffer(&format!("lighting env {slot}"), env_size))
            })
            .collect();
        let frame_buffers: Vec<_> = (0..slots)
            .map(|slot| {
                Mutex::new(ctx.create_host_buffer(&format!("lighting frame {slot}"), frame_size))
            })
            .collect();

        for slot in 0..slots {
            let env_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffers[slot].lock().unwrap().buffer)
                .offset(0)
                .range(env_size)];
            let frame_info = [vk::DescriptorBufferInfo::default()
                .buffer(frame_buffers[slot].lock().unwrap().buffer)
                .offset(0)
                .range(frame_size)];
            let shadow_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_shadow.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let compare_info = [vk::DescriptorImageInfo::default().sampler(shadow_compare)];
            let gi_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_gi_ao.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let reflection_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_reflection.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let feedback_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_depth.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let linear_info = [vk::DescriptorImageInfo::default().sampler(linear)];
            let env_map_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_env.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let env_sampler_info = [vk::DescriptorImageInfo::default().sampler(env_sampler)];
            let cloud_info = [vk::DescriptorImageInfo::default()
                .image_view(dummy_cloud.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let set = sets[slot];
            let writes = [
                write(
                    set,
                    BINDING_ENV,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &[],
                    &env_info,
                ),
                write(
                    set,
                    BINDING_FRAME,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &[],
                    &frame_info,
                ),
                write(
                    set,
                    BINDING_SHADOW_MAP,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &shadow_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_SHADOW_SAMPLER,
                    vk::DescriptorType::SAMPLER,
                    &compare_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_GI_AO,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &gi_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_REFLECTION,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &reflection_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_LINEAR,
                    vk::DescriptorType::SAMPLER,
                    &linear_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_FEEDBACK_DEPTH,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &feedback_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_ENV_MAP,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &env_map_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_ENV_SAMPLER,
                    vk::DescriptorType::SAMPLER,
                    &env_sampler_info,
                    &[],
                ),
                write(
                    set,
                    BINDING_CLOUD_SHADOW,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &cloud_info,
                    &[],
                ),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        Self {
            ctx: Arc::clone(ctx),
            layout,
            pool,
            sets,
            buffers,
            frame_buffers,
            shadow_compare,
            linear,
            dummy_shadow,
            _dummy_gi_ao: dummy_gi_ao,
            _dummy_reflection: dummy_reflection,
            dummy_depth,
            screen_bound: Mutex::new(vec![(0, None); slots]),
            env_bound: Mutex::new(vec![(0, None); slots]),
            env_sampler,
            dummy_env,
            dummy_cloud,
            bound_generation: AtomicU64::new(0),
        }
    }

    /// Set layout to place at [`SET_INDEX`] in consumer pipeline layouts.
    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Write this frame slot's environment. Call before recording draws.
    pub fn upload(&self, frame_slot: usize, env: &LightingEnv) {
        let len = self.buffers.len();
        self.buffers[frame_slot % len]
            .lock()
            .unwrap()
            .write(0, bytemuck::bytes_of(env));
    }

    /// Write this frame slot's per-frame lighting block (shadow matrices,
    /// screen size, previous view-projection).
    pub(crate) fn upload_frame(&self, frame_slot: usize, frame: &LightingFrame) {
        let len = self.frame_buffers.len();
        self.frame_buffers[frame_slot % len]
            .lock()
            .unwrap()
            .write(0, bytemuck::bytes_of(frame));
    }

    /// Point binding 2 at `shadow` (a layer-array view), or back at the lit
    /// dummy when `None`. `generation` identifies the map: the sets are only
    /// rewritten when it differs from the last one bound, so repeated calls
    /// are free. The GPU goes idle first because the sets may be in flight.
    pub(crate) fn bind_shadow_map(&self, shadow: Option<vk::ImageView>, generation: u64) {
        if self.bound_generation.load(Ordering::Acquire) == generation {
            return;
        }
        let device = self.ctx.device();
        unsafe { device.device_wait_idle() }.expect("wait idle before rebinding the shadow map");
        let info = [vk::DescriptorImageInfo::default()
            .image_view(shadow.unwrap_or(self.dummy_shadow.view))
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        for &set in &self.sets {
            let writes = [write(
                set,
                BINDING_SHADOW_MAP,
                vk::DescriptorType::SAMPLED_IMAGE,
                &info,
                &[],
            )];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        self.bound_generation.store(generation, Ordering::Release);
    }

    /// Point this frame slot's bindings 4 (indirect diffuse and visibility), 5
    /// (traced reflection) and 7 (the depth both were computed from) at `views`,
    /// or back at the neutral dummies when `None`. `generation` identifies the
    /// chains that produced `views` (use 0 with `None`). Call at record time for
    /// the slot being recorded: that slot's set is idle there, so no device-wide
    /// wait is needed.
    pub(crate) fn bind_screen_space(
        &self,
        frame_slot: usize,
        views: Option<ScreenSpaceViews>,
        generation: u64,
    ) {
        let slot = frame_slot % self.sets.len();
        let mut bound = self.screen_bound.lock().unwrap();
        // A new chain can reuse freed handle values, so the generation is part of the key.
        let key = (generation, views);
        if bound[slot] == key {
            return;
        }
        let views = views.unwrap_or(ScreenSpaceViews {
            gi: None,
            reflection: None,
            depth: self.dummy_depth.view,
        });
        let gi_info = [vk::DescriptorImageInfo::default()
            .image_view(views.gi.unwrap_or(self._dummy_gi_ao.view))
            .image_layout(vk::ImageLayout::GENERAL)];
        let reflection_info = [vk::DescriptorImageInfo::default()
            .image_view(views.reflection.unwrap_or(self._dummy_reflection.view))
            .image_layout(vk::ImageLayout::GENERAL)];
        let depth_info = [vk::DescriptorImageInfo::default()
            .image_view(views.depth)
            .image_layout(vk::ImageLayout::GENERAL)];
        let set = self.sets[slot];
        let writes = [
            write(
                set,
                BINDING_GI_AO,
                vk::DescriptorType::SAMPLED_IMAGE,
                &gi_info,
                &[],
            ),
            write(
                set,
                BINDING_REFLECTION,
                vk::DescriptorType::SAMPLED_IMAGE,
                &reflection_info,
                &[],
            ),
            write(
                set,
                BINDING_FEEDBACK_DEPTH,
                vk::DescriptorType::SAMPLED_IMAGE,
                &depth_info,
                &[],
            ),
        ];
        unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };
        bound[slot] = key;
    }

    /// Point this frame slot's binding 8 at a sky environment map, or back at
    /// the dummy when `None`. `env` is `(generation, mip-chain view)`: the
    /// generation identifies the atmosphere that made the map, so a reused
    /// handle is not mistaken for the old one. Call at record time for the slot
    /// being recorded, as [`Self::bind_screen_space`] is.
    pub(crate) fn bind_env_map(&self, frame_slot: usize, env: Option<(u64, vk::ImageView)>) {
        let slot = frame_slot % self.sets.len();
        let mut bound = self.env_bound.lock().unwrap();
        let key: EnvKey = match env {
            Some((generation, view)) => (generation, Some(view)),
            None => (0, None),
        };
        if bound[slot] == key {
            return;
        }
        let view = key.1.unwrap_or(self.dummy_env.view);
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [write(
            self.sets[slot],
            BINDING_ENV_MAP,
            vk::DescriptorType::SAMPLED_IMAGE,
            &info,
            &[],
        )];
        unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };
        bound[slot] = key;
    }

    /// Point this frame slot's binding 10 at its cloud shadow map (a `R32_SFLOAT`
    /// view in `GENERAL` layout), or back at the dummy when `None`. Call at record
    /// time for the slot being recorded, as [`Self::bind_env_map`] is.
    pub(crate) fn bind_cloud_shadow(&self, frame_slot: usize, view: Option<vk::ImageView>) {
        let slot = frame_slot % self.sets.len();
        let view = view.unwrap_or(self.dummy_cloud.view);
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [write(
            self.sets[slot],
            BINDING_CLOUD_SHADOW,
            vk::DescriptorType::SAMPLED_IMAGE,
            &info,
            &[],
        )];
        unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };
    }

    /// The descriptor set for frame slot `frame_slot`.
    pub fn set(&self, frame_slot: usize) -> vk::DescriptorSet {
        self.sets[frame_slot % self.sets.len()]
    }
}

/// One descriptor write for set 3, using whichever info list the type needs.
fn write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    image: &'a [vk::DescriptorImageInfo],
    buffer: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty);
    match ty {
        vk::DescriptorType::UNIFORM_BUFFER => write.buffer_info(buffer),
        _ => write.image_info(image),
    }
}

impl Drop for Lighting {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let device = self.ctx.device();
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_sampler(self.shadow_compare, None);
            device.destroy_sampler(self.linear, None);
            device.destroy_sampler(self.env_sampler, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // CPU mirror of the WGSL math for sanity tests.
    fn sky(dir: Vec3, env: &LightingEnv) -> Vec3 {
        let up = dir.y.clamp(-1.0, 1.0);
        let zenith = Vec3::from_array(env.sky_zenith[..3].try_into().unwrap());
        let horizon = Vec3::from_array(env.sky_horizon[..3].try_into().unwrap());
        let ground = Vec3::from_array(env.sky_ground[..3].try_into().unwrap());
        let sky = horizon.lerp(zenith, (up * 1.6).clamp(0.0, 1.0));
        let gnd = horizon.lerp(ground, (-up * 1.6).clamp(0.0, 1.0));
        if up >= 0.0 { sky } else { gnd }
    }

    fn env_brdf(rough: f32, nov: f32) -> [f32; 2] {
        let c0 = [-1.0f32, -0.0275, -0.572, 0.022];
        let c1 = [1.0f32, 0.0425, 1.04, -0.04];
        let r: [f32; 4] = std::array::from_fn(|i| rough * c0[i] + c1[i]);
        let a004 = (r[0] * r[0]).min(2f32.powf(-9.28 * nov)) * r[0] + r[1];
        [(-1.04) * a004 + r[2], 1.04 * a004 + r[3]]
    }

    #[test]
    fn composed_shader_validates_with_naga() {
        // lighting snippet prepended to a trivial consumer compiles cleanly.
        let source = format!(
            "{SHADER}\nstruct O {{ @location(0) c: vec4<f32> }};\n\
             @fragment fn fs_main(@builtin(position) p: vec4<f32>) -> O {{\n\
             \x20 var o: O; o.c = vec4(lighting_eval(vec3(0.5), 0.0, 0.5,\n\
             \x20 vec3(0,1,0), vec3(0,0,1), p.xyz, 1.0), 1.0); return o; }}"
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("composed lighting shader must validate");
    }

    #[test]
    fn sky_gradient_is_bounded_and_analytic() {
        let env = LightingEnv::default();
        for dir in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::new(0.4, 0.7, -0.6).normalize(),
        ] {
            let c = sky(dir.normalize(), &env);
            assert!(c.cmplt(Vec3::ONE).all() && c.cmpge(Vec3::ZERO).all());
        }
        // Zenith reflects zenith colour, ground reflects ground colour.
        assert!(sky(Vec3::Y, &env).z > sky(Vec3::Y, &env).x);
        assert!(sky(Vec3::NEG_Y, &env).length() < sky(Vec3::Y, &env).length());
    }

    #[test]
    fn env_brdf_is_energy_conserving() {
        for rough in [0.0, 0.3, 0.6, 1.0] {
            for nov in [0.05, 0.5, 1.0] {
                let [scale, bias] = env_brdf(rough, nov);
                // |f0 * scale + bias| <= 1.2 for f0 in [0,1] (Karis approx
                // slightly exceeds 1 near grazing angles).
                for f0 in [0.04f32, 0.5, 1.0] {
                    let v = f0 * scale + bias;
                    assert!(
                        (0.0..=1.25).contains(&v),
                        "rough={rough} nov={nov} f0={f0} -> {v}"
                    );
                }
            }
        }
        // Mirror view, mirror surface: scale ~1 (full F0 response).
        let [scale, _] = env_brdf(0.0, 1.0);
        assert!((scale - 1.0).abs() < 0.02);
        // Very rough at grazing incidence: response is smooth and bounded.
        let [scale, bias] = env_brdf(1.0, 0.05);
        assert!(scale.is_finite() && bias.is_finite() && scale.abs() < 1.0);
    }

    #[test]
    fn uniform_block_is_all_vec4() {
        assert_eq!(std::mem::size_of::<LightingEnv>(), 7 * 16);
    }

    #[test]
    fn lighting_frame_block_matches_wgsl_layout() {
        // WGSL: 4 mat4 (256) + 4 vec4 (64) + 5 vec4 (80) + mat4 (64).
        assert_eq!(std::mem::size_of::<LightingFrame>(), 512);
        assert_eq!(std::mem::offset_of!(LightingFrame, cloud_shadow), 480);
        assert_eq!(std::mem::offset_of!(LightingFrame, cloud_layer), 496);
        assert_eq!(std::mem::offset_of!(LightingFrame, cascade_view_proj), 0);
        assert_eq!(std::mem::offset_of!(LightingFrame, cascade_params), 256);
        assert_eq!(std::mem::offset_of!(LightingFrame, shadow), 320);
        assert_eq!(std::mem::offset_of!(LightingFrame, shadow2), 336);
        assert_eq!(std::mem::offset_of!(LightingFrame, shadow_fade), 352);
        assert_eq!(std::mem::offset_of!(LightingFrame, screen), 368);
        assert_eq!(std::mem::offset_of!(LightingFrame, features), 384);
        assert_eq!(std::mem::offset_of!(LightingFrame, prev_view_proj), 400);
        assert_eq!(std::mem::offset_of!(LightingFrame, indirect), 464);
    }

    #[test]
    fn composed_shader_declares_frame_block_and_shadow_bindings() {
        for needle in [
            "@group(3) @binding(1) var<uniform> lighting_frame",
            "@group(3) @binding(2) var lighting_shadow_map: texture_depth_2d_array",
            "@group(3) @binding(3) var lighting_shadow_sampler: sampler_comparison",
            "@group(3) @binding(6) var lighting_linear: sampler",
            "fn lighting_begin(",
            "fn lod_dither(",
        ] {
            assert!(SHADER.contains(needle), "missing {needle}");
        }
    }
}

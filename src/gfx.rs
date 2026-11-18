//! Native Vulkan renderer.
//!
//! Chain: scene pass -> FSR 3.1 temporal upscale (see `fsr3.rs`) -> blit to the
//! swapchain.
//!
//! All offscreen images live in `GENERAL` layout, which is legal for every use
//! and lets us skip per-image layout bookkeeping. The swapchain image is the
//! only one that gets explicit transitions.
//!
//! Coordinate convention: shaders author to wgpu-style NDC (y up, depth 0..1),
//! which we preserve in Vulkan by using a negative viewport height.

#[cfg(not(target_os = "macos"))]
mod acquire;
#[cfg(not(target_os = "macos"))]
mod atmosphere;
#[cfg(not(target_os = "macos"))]
mod cloud_shadow;
#[cfg(not(target_os = "macos"))]
mod cloud_view;
#[cfg(not(target_os = "macos"))]
mod clouds;
#[cfg_attr(target_os = "macos", path = "gfx/metal/draw.rs")]
mod draw;
#[cfg(not(target_os = "macos"))]
mod fog;
#[cfg_attr(target_os = "macos", path = "gfx/metal/frame.rs")]
mod frame;
#[cfg(not(target_os = "macos"))]
mod hiz;
#[cfg(not(target_os = "macos"))]
mod indirect;
#[cfg(not(target_os = "macos"))]
mod linear_depth;
#[cfg(not(target_os = "macos"))]
mod pass_timer;
#[cfg(not(target_os = "macos"))]
mod pipeline;
pub mod present;
#[cfg(target_os = "macos")]
use metal::pipeline;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(not(target_os = "macos"))]
mod post;
#[cfg_attr(target_os = "macos", path = "gfx/metal/readback.rs")]
mod readback;
#[cfg(not(target_os = "macos"))]
mod reflection;
#[cfg_attr(target_os = "macos", path = "gfx/metal/renderer.rs")]
pub(crate) mod renderer;
#[cfg(not(target_os = "macos"))]
mod rt;
#[cfg(not(target_os = "macos"))]
mod shader;
#[cfg(not(target_os = "macos"))]
mod shadow;
#[cfg(not(target_os = "macos"))]
mod sky_frame;
#[cfg(not(target_os = "macos"))]
mod taa;
#[cfg_attr(target_os = "macos", path = "gfx/metal/thumbnail.rs")]
mod thumbnail;
#[cfg(not(target_os = "macos"))]
mod timing;
#[cfg(not(target_os = "macos"))]
pub mod vulkan;

#[cfg(target_os = "macos")]
pub(crate) use metal::align_up;
#[cfg(target_os = "macos")]
pub use metal::{DynamicBuffers, GpuBuffer, GpuContext, GpuImage, PulledMesh};
#[cfg(not(target_os = "macos"))]
pub(crate) use pipeline::{create_shadow_pipeline, create_shadow_pipeline_cull};
pub use present::{GpuPreference, PresentMode};
#[cfg(not(target_os = "macos"))]
pub use rt::{RayTracingScene, RtInstance, RtMaterial, RtMeshHandle};
#[cfg(not(target_os = "macos"))]
pub use vulkan::{DynamicBuffers, GpuBuffer, GpuContext, GpuImage, PulledMesh};

use crate::camera::Camera;
use crate::fsr3::{Fsr3, JitterSequence};
use crate::lighting::{Lighting, LightingEnv};
use crate::render::RenderSettings;
use crate::settings::AaMode;
#[cfg(not(target_os = "macos"))]
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(not(target_os = "macos"))]
pub const SCENE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
#[cfg(not(target_os = "macos"))]
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
#[cfg(not(target_os = "macos"))]
pub const VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
/// Per-pixel surface data for later lighting passes (AO, GI, reflections).
///
/// Encoding, written by the scene fragment shaders:
/// - `rg`: octahedral-encoded world-space unit normal, remapped to 0..1.
/// - `b`: perceptual roughness.
/// - `a`: surface class. 0 = no surface data (sky, unlit debug geometry);
///   1 = dielectric surface; 2/3 = metallic surface (metallic > 0.5).
///
/// Cleared to all zero at the start of each scene pass.
#[cfg(not(target_os = "macos"))]
pub const SURFACE_FORMAT: vk::Format = vk::Format::A2B10G10R10_UNORM_PACK32;
/// Colour attachment formats of the scene pass, in attachment order:
/// scene colour, motion vectors, surface data (see [`SURFACE_FORMAT`]).
#[cfg(not(target_os = "macos"))]
pub const SCENE_COLOR_FORMATS: [vk::Format; 3] = [SCENE_FORMAT, VELOCITY_FORMAT, SURFACE_FORMAT];

#[cfg(target_os = "macos")]
pub const SCENE_FORMAT: metal::PixelFormat = metal::PixelFormat::RGBA16Float;
#[cfg(target_os = "macos")]
pub const DEPTH_FORMAT: metal::PixelFormat = metal::PixelFormat::Depth32Float;
#[cfg(target_os = "macos")]
pub const VELOCITY_FORMAT: metal::PixelFormat = metal::PixelFormat::RG16Float;
#[cfg(target_os = "macos")]
pub const SURFACE_FORMAT: metal::PixelFormat = metal::PixelFormat::RGB10A2Unorm;
#[cfg(target_os = "macos")]
pub const SCENE_COLOR_FORMATS: [metal::PixelFormat; 3] =
    [SCENE_FORMAT, VELOCITY_FORMAT, SURFACE_FORMAT];

/// Shading values of a mesh for hardware ray tracing.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtMaterial {
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
}

#[cfg(target_os = "macos")]
impl Default for RtMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8],
            roughness: 0.5,
            emissive: [0.0; 3],
            metallic: 0.0,
        }
    }
}

/// Handle of a mesh registered with [`RayTracingScene`].
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtMeshHandle {
    _private: u32,
}

/// One placed copy of a registered mesh for hardware ray tracing.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtInstance {
    pub mesh: RtMeshHandle,
    pub transform: glam::Mat4,
}

/// Hardware ray-tracing scene (unavailable on Metal; `Renderer::ray_tracing_scene_mut` returns `None`).
#[cfg(target_os = "macos")]
pub struct RayTracingScene;

#[cfg(target_os = "macos")]
impl RayTracingScene {
    pub fn register_mesh(
        &mut self,
        _positions: &[[f32; 3]],
        _indices: &[u32],
        _material: RtMaterial,
    ) -> Option<RtMeshHandle> {
        None
    }

    pub fn set_instances(&mut self, _instances: &[RtInstance]) {}

    pub fn id(&self) -> u64 {
        0
    }
}

/// Diagnostic counts of the shared texture objects of one [`GpuContext`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureObjectStats {
    /// Distinct samplers alive in the shared cache.
    pub samplers: usize,
    /// Descriptor pool chunks alive.
    pub descriptor_pools: usize,
    /// Textures currently holding descriptor sets (one per texture).
    pub live_texture_sets: usize,
}

/// Mirror of the errors the swapchain path can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceError {
    Lost,
    Outdated,
    OutOfMemory,
    Timeout,
    Other,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorVertex {
    pub position: [f32; 3],
    pub color: [u8; 4],
}

/// Scene-pass uniform block. The scene renders with the jittered projection;
/// motion vectors use the two unjittered projections.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    view_projection: [[f32; 4]; 4],
    curr_view_projection: [[f32; 4]; 4],
    prev_view_projection: [[f32; 4]; 4],
}

/// Summary of one rendered frame, read back from the GPU. Used by the game's
/// `DREAM_FRAME_STATS` hook and by scripted capture runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMetrics {
    pub size: (u32, u32),
    pub mean: [f32; 3],
    pub sky_pct: f32,
    pub sky_top_pct: f32,
    pub sky_bottom_pct: f32,
    pub nonblack_pct: f32,
    /// Largest motion-vector magnitude in the render-resolution target, in
    /// normalized render-target units.
    pub mv_max: f32,
    /// Mean motion-vector magnitude.
    pub mv_mean: f32,
    /// Fraction of motion vectors pointing more than a quarter of the screen in
    /// one frame, which for this scene means a broken projection.
    pub mv_large_pct: f32,
    /// Fraction of motion vectors with a non-negligible magnitude.
    pub mv_nonzero_pct: f32,
    /// Mean of each channel of the FSR dilated reactive masks, at render
    /// resolution: REACTIVE, DISOCCLUSION, SHADING_CHANGE, ACCUMULATION. All
    /// four should fall back to ~0 once the camera stops; a channel that stays
    /// high is a permanent bias on the upscaled output rather than a transient.
    pub reactive_mean: [f32; 4],
}

impl FrameMetrics {
    pub fn summary(&self) -> String {
        format!(
            "{}x{} mean=({:.3},{:.3},{:.3}) sky={:.1}% (top={:.1}% bottom={:.1}%) nonblack={:.1}% mv_max={:.4} mv_mean={:.5} mv_large={:.2}% mv_nonzero={:.1}% reactive={:.4} disocclusion={:.4} shading={:.4} accumulation={:.4}",
            self.size.0,
            self.size.1,
            self.mean[0],
            self.mean[1],
            self.mean[2],
            self.sky_pct,
            self.sky_top_pct,
            self.sky_bottom_pct,
            self.nonblack_pct,
            self.mv_max,
            self.mv_mean,
            self.mv_large_pct,
            self.mv_nonzero_pct,
            self.reactive_mean[0],
            self.reactive_mean[1],
            self.reactive_mean[2],
            self.reactive_mean[3],
        )
    }
}

/// CPU-side time the last `render` call spent at each engine checkpoint, in
/// microseconds. A frame whose wall time is large while every game-side timer
/// is small blocked inside the renderer: `token_wait` is the frame-start stall
/// the app feels, `fence_wait` means the GPU was behind, `acquire` or `present`
/// means the presentation engine (compositor) held the swapchain (both run
/// off the frame's critical path or after it). `record` includes the caller's
/// draw callback.
#[derive(Clone, Copy, Debug, Default)]
pub struct FramePhases {
    /// Time the frame's slot fence took to signal, measured on the
    /// acquire-ahead worker (the acquire is gated on that fence).
    pub fence_wait_us: f32,
    /// Time the worker spent inside `vkAcquireNextImageKHR` for this frame's
    /// image -- the presentation engine holding the swapchain. Off the
    /// frame's critical path; the frame only pays the residual in
    /// [`Self::token_wait_us`].
    pub acquire_us: f32,
    /// Time the frame itself blocked waiting for its token: the frame-start
    /// stall the app feels. Near zero when the acquire-ahead kept up; a
    /// skipped frame reports its whole wait budget here.
    pub token_wait_us: f32,
    pub record_us: f32,
    pub submit_us: f32,
    pub present_us: f32,
}

#[cfg(not(target_os = "macos"))]
struct SceneTargets {
    color: GpuImage,
    velocity: GpuImage,
    surface: GpuImage,
    depth: GpuImage,
}

#[cfg(target_os = "macos")]
pub(crate) struct SceneTargets {
    pub(crate) color: GpuImage,
    pub(crate) velocity: GpuImage,
    pub(crate) surface: GpuImage,
    pub(crate) depth: GpuImage,
}

/// Which pass a [`DrawCtx`] records. A scene is drawn once per shadow cascade
/// that is updated this frame, then once for the main pass, then once more
/// for the transparent pass after the sky and fog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenePass {
    /// The lit colour pass (colour, motion vectors and surface data).
    Main,
    /// A depth-only pass into one cascade of the sun shadow map.
    Shadow {
        /// Index of the cascade being rendered, `0..cascades`.
        cascade: u32,
    },
    /// Alpha-blended geometry, composited over the sky and fog. It loads the
    /// scene colour, motion vectors and surface data and writes only colour;
    /// depth is tested but not written.
    Transparent,
}

/// Inputs of one frame drawn through [`Renderer::render_scene`].
pub struct SceneFrame<'a> {
    /// Camera. Its unjittered matrices fit the shadow cascades.
    pub camera: &'a Camera,
    /// Lighting set this frame's shaders read. Keep using the same `Lighting`
    /// with `render_scene` (the shadow map is bound into it on first use; a
    /// plain `render` with it afterwards may see a freed map).
    pub lighting: &'a Lighting,
    /// Sun and sky. The renderer fills `cam_pos` from `camera`.
    pub env: LightingEnv,
    /// Screen rectangle the scene occupies (see [`Renderer::render_ui`]).
    pub scene_rect: Option<crate::ui::Rect>,
    /// UI drawn over the composed frame.
    pub ui: Option<&'a crate::ui::draw::UiFrame>,
}

/// Draw commands handed to the game for the scene pass.
#[cfg(not(target_os = "macos"))]
pub struct DrawCtx<'a> {
    device: &'a ash::Device,
    cmd: vk::CommandBuffer,
    scene_pipeline: vk::Pipeline,
    scene_layout: vk::PipelineLayout,
    scene_set: vk::DescriptorSet,
    pulling: Option<(vk::Pipeline, vk::PipelineLayout)>,
    frame_slot: usize,
    frame_id: u64,
    starter: std::cell::RefCell<Option<&'a mut dyn FnMut()>>,
    pass: ScenePass,
    /// The matrix geometry is transformed by in this pass.
    view_proj: glam::Mat4,
    /// The occlusion grid the main pass may test against. `None` in shadow passes.
    occlusion: Option<&'a crate::cull::OcclusionBuffer>,
    /// Whether a transparent pass follows the main pass in this frame. True
    /// for [`Renderer::render_scene`](crate::gfx::Renderer::render_scene) only.
    has_transparent: bool,
}

/// Draw commands handed to the game for the scene pass (native Metal).
#[cfg(target_os = "macos")]
pub struct DrawCtx<'a> {
    pub(crate) encoder:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    pub(crate) scene_pipeline:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    pub(crate) scene_depth_stencil:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>,
    pub(crate) scene_uniform: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub(crate) pulling:
        Option<&'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>,
    pub(crate) index_buffer: Option<(
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        u64,
    )>,
    pub(crate) frame_slot: usize,
    pub(crate) frame_id: u64,
    pub(crate) starter: std::cell::RefCell<Option<&'a mut dyn FnMut()>>,
    pub(crate) pass: ScenePass,
    pub(crate) view_proj: glam::Mat4,
    pub(crate) has_transparent: bool,
}

#[cfg(not(target_os = "macos"))]
pub struct Renderer {
    ctx: Arc<GpuContext>,
    swapchain: vulkan::Swapchain,

    scene_pipeline: vk::Pipeline,
    scene_layout: vk::PipelineLayout,
    scene_dsl: vk::DescriptorSetLayout,
    /// Vertex-pulling pipeline variant: zero vertex input, quad storage at
    /// set 1, a 16-byte vertex-stage origin push constant. Present only when
    /// the renderer was built by [`Self::new_with_pulling`].
    pulling: Option<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)>,
    blit_pipeline: vk::Pipeline,
    blit_layout: vk::PipelineLayout,
    blit_dsl: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    /// Per frame slot: slot N's set/uniform are only touched after its
    /// fence recycled, so a submitted frame never sees them mutate.
    scene_sets: Vec<vk::DescriptorSet>,
    /// Per frame slot, like `scene_sets`: rewritten each frame with the image
    /// the blit samples (scene colour, FSR output or TAA output).
    blit_sets: Vec<vk::DescriptorSet>,

    sampler: vk::Sampler,
    scene_uniforms: Vec<GpuBuffer>,

    // Per-frame resources.
    frame_pool: vk::CommandPool,
    cmd: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    /// One per SWAPCHAIN IMAGE (indexed by the acquired image index), not per
    /// frame slot: a presented image's semaphore stays referenced by the
    /// presentation engine until that image is re-acquired, so reusing one per
    /// slot violates VUID-vkQueueSubmit-pSignalSemaphores-00067. Rebuilt with
    /// the swapchain in `Renderer::resize`.
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    /// Serial carried by each slot's fence; confirmed after the wait.
    frame_serials: Vec<u64>,
    frame: usize,
    /// Monotonic frame counter handed to [`DrawCtx::frame_id`]; bumped only
    /// after a submission succeeds, so a failed frame does not retire slot
    /// resources it never used.
    frame_id: u64,
    /// Per-slot timestamps: a pair bracketing the scene pass (0 = before
    /// begin_rendering, 1 = after end_rendering) and a pair bracketing the post
    /// work (2 = at the scene's end, 3 = before the blit). `scene_ms[slot]` and
    /// `post_ms[slot]` hold the measured GPU times of the last completed frame.
    gpu_pool: vk::QueryPool,
    /// Per-pass GPU timestamps of the scene and post work (see `pass_gpu_ms`).
    pass_timer: pass_timer::PassTimer,
    scene_ms: Vec<f32>,
    post_ms: Vec<f32>,
    shadow_ms: Vec<f32>,
    scene_sample: Option<(u64, f32)>,
    scene_recorded: Vec<bool>,
    /// Hi-Z occlusion pass, present while `RenderSettings::occlusion` is set.
    occlusion: Option<hiz::HiZ>,
    /// The latest occlusion grid read back from the GPU.
    occlusion_data: Option<Arc<crate::cull::OcclusionBuffer>>,
    /// Per frame slot: the grid recorded into that slot, with the generation it
    /// belongs to and what it was rendered with.
    occlusion_pending: Vec<Option<(u64, crate::cull::OcclusionMeta)>>,
    /// Bumped whenever occlusion history is dropped, so stale grids are ignored.
    occlusion_generation: u64,

    targets: SceneTargets,
    targets_ready: bool,
    /// GPU downsample of the scene target for card thumbnails (built on demand).
    thumb: Option<thumbnail::ThumbnailPass>,
    fsr: Option<Fsr3>,
    fsr_enabled: bool,
    /// Anti-aliasing mode the renderer was asked for (see `set_aa_mode`).
    aa_mode: AaMode,
    /// Native-resolution TAA passes, present while `aa_mode` is `Taa` and the
    /// scene targets exist.
    taa: Option<taa::Taa>,
    /// Shared linear depth, present while any screen-space chain is enabled.
    linear_depth: Option<linear_depth::LinearDepth>,
    /// Screen-space AO and GI chain, present while either is enabled.
    indirect: Option<indirect::Indirect>,
    /// Screen-space reflection chain, present while reflections are enabled.
    reflection: Option<reflection::Reflection>,
    /// Physically based sky, present while the atmosphere setting is on.
    atmosphere: Option<atmosphere::Atmosphere>,
    /// Volumetric fog, present while the fog setting is on.
    fog: Option<fog::Fog>,
    /// Volumetric clouds and their sky panorama, present while the clouds setting is on.
    clouds: Option<clouds::Clouds>,
    /// Optional hardware ray-tracing scene. Created only after explicit access
    /// on a supporting device; UI-only renderers never allocate RT resources.
    rt_scene: Option<rt::RayTracingScene>,
    /// CPU atmosphere and the lighting it drives (see `atmosphere::SkyCpu`).
    sky_cpu: sky_frame::SkyCpu,
    /// Scene time in seconds: drives the fog noise. Advanced by the frame delta times the scale.
    scene_time: f64,
    /// Multiplier on the frame delta that advances [`Self::scene_time`].
    scene_time_scale: f32,
    /// Previous frame's rotation-only view-projection, for the sky's velocity.
    prev_sky_view_proj: glam::Mat4,
    /// Bumped whenever a screen-space chain is rebuilt; keys the lighting bindings.
    screen_generation: u64,
    /// Render scale of the FSR preset, restored when `aa_mode` returns to FSR.
    fsr_render_scale: f32,
    /// A swapchain image the acquire worker had already reserved when the GPU
    /// was quiesced. The next frame presents it instead of acquiring again.
    parked_token: Option<acquire::Token>,
    /// Composed display image (swapchain format, screen-row order), used only
    /// by frames that cannot be composed straight into the swapchain: a
    /// crosshair, frame capture, or a colour-space change. There the blit
    /// writes it, the UI pass draws over it and the present pass copies it to
    /// the swapchain. Otherwise it is a 1x1 placeholder that the present set
    /// can still sample.
    display: GpuImage,
    /// Whether `display` is the full-size image rather than the placeholder.
    display_live: bool,
    /// Whether a frame has been composed into `display` since it was built:
    /// the only state in which the display holds something to read back.
    display_written: bool,
    /// Keep the composed frame for CPU readbacks (see
    /// [`Renderer::set_frame_capture`]).
    frame_capture: bool,
    /// Requested swapchain present mode (see [`Renderer::set_present_mode`]).
    present_mode: vulkan::PresentMode,
    /// Set once a readback has found no retained frame and said so.
    capture_notice: bool,
    present_pipeline: vk::Pipeline,
    present_layout: vk::PipelineLayout,
    present_set: vk::DescriptorSet,
    /// UI overlay pass; recorded between the blit and the present pass.
    ui: crate::ui::render::UiRenderer,
    /// Colour space the UI pass blends in (fixed at construction: it
    /// decides the display image's format and the pipelines' entries).
    ui_blend: crate::ui::render::UiBlend,
    display_size: (u32, u32),
    render_size: (u32, u32),
    jitter: JitterSequence,
    prev_view_projection: glam::Mat4,
    history_valid: bool,
    temporal: bool,
    /// Draw the centred plus crosshair in the present pass. Off by default.
    show_crosshair: bool,
    sharpness: f32,
    render_scale: f32,
    last_frame: Instant,
    /// Segment timings of the most recent completed `render*` call.
    phases: FramePhases,
    /// Managed acquire-ahead worker: resolves the next frame's swapchain
    /// image (gated on the slot's fence) while the app keeps running.
    acquire: acquire::AcquireWorker,
    /// Wall-clock bound on the frame-start wait for the token. `None` -- the
    /// default -- waits as long as the presentation engine needs, capped at
    /// the liveness bound (`acquire::UNBUDGETED_WAIT_CAP`).
    frame_wait_budget: Option<Duration>,
    /// Requested render settings; see [`Renderer::set_render_settings`].
    render_settings: RenderSettings,
    /// Sun shadow map, present only while shadows are on and it has been built.
    shadows: Option<shadow::ShadowMap>,
    /// Bumped whenever the shadow map is created or dropped, so each `Lighting`
    /// knows to rebind its shadow binding (see `Lighting::bind_shadow_map`).
    shadow_generation: u64,
    /// Set when every cascade must render this frame (first use, settings
    /// change, resize, history reset); cleared once a frame renders them all.
    shadow_all_dirty: bool,
    /// Depth-only variants of the scene pipeline, and of the pulling pipeline
    /// when one exists.
    shadow_scene_pipeline: vk::Pipeline,
    shadow_pulling_pipeline: Option<vk::Pipeline>,
}

#[cfg(target_os = "macos")]
pub struct Renderer {
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) swapchain: metal::Swapchain,
    pub(crate) scene_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) scene_depth_stencil:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) no_depth_stencil:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) pulling: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
        >,
    >,
    pub(crate) blit_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) present_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) scene_uniforms: Vec<GpuBuffer>,
    pub(crate) slots: Vec<Arc<renderer::SlotSync>>,
    pub(crate) frame: usize,
    pub(crate) frame_id: u64,
    pub(crate) scene_ms: Vec<f32>,
    pub(crate) scene_sample: Option<(u64, f32)>,
    pub(crate) scene_recorded: Vec<bool>,
    pub(crate) targets: SceneTargets,
    pub(crate) targets_ready: bool,
    pub(crate) thumb: Option<thumbnail::ThumbnailPass>,
    pub(crate) fsr: Option<Fsr3>,
    pub(crate) fsr_enabled: bool,
    pub(crate) display: GpuImage,
    pub(crate) display_live: bool,
    pub(crate) display_written: bool,
    pub(crate) frame_capture: bool,
    pub(crate) present_mode: PresentMode,
    pub(crate) capture_notice: bool,
    pub(crate) ui: crate::ui::render::UiRenderer,
    pub(crate) ui_blend: crate::ui::render::UiBlend,
    pub(crate) display_size: (u32, u32),
    pub(crate) render_size: (u32, u32),
    pub(crate) jitter: JitterSequence,
    pub(crate) prev_view_projection: glam::Mat4,
    pub(crate) history_valid: bool,
    pub(crate) temporal: bool,
    pub(crate) show_crosshair: bool,
    pub(crate) sharpness: f32,
    pub(crate) render_scale: f32,
    pub(crate) last_frame: Instant,
    pub(crate) phases: FramePhases,
    pub(crate) frame_wait_budget: Option<Duration>,
    pub(crate) aa_mode: AaMode,
    pub(crate) render_settings: RenderSettings,
    pub(crate) scene_time: f64,
    pub(crate) scene_time_scale: f32,
}

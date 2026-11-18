//! A plug-and-play scene: models with levels of detail and impostors,
//! instances, culling and drawing, behind one `render` call.
//!
//! Each frame, [`World::render`]:
//!
//! 1. updates every instance exactly once (animation and motion history);
//! 2. frustum-tests every instance against the camera, and decides each one's
//!    mesh level and impostor state from its screen coverage, once, before any
//!    pass runs (so shadow passes cannot change what the main pass shows);
//! 3. renders the sun shadow passes: frustum-tested against each cascade,
//!    drawing the shadow mesh of every caster;
//! 4. renders the main pass: occlusion-tests the frustum survivors against the
//!    previous frames' depth, then draws meshes grouped by model and level and
//!    impostors batched once per asset.
//!
//! The pieces stay usable on their own: [`crate::cull`] for the tests,
//! [`crate::impostor::LodPolicy`] for the thresholds, and
//! [`crate::model::render::ModelRenderer`] / [`crate::impostor::ImpostorRenderer`]
//! for drawing without a world.
//!
//! Instances of a model whose level is opaque and not animating are drawn
//! together: a level with two or more of them is one instanced draw per mesh
//! (see [`WorldSettings::instancing`]).
//!
//! A LoD switch crossfades for `lod::CROSSFADE_SECONDS` (0.25 s) and then
//! settles: a still camera has no crossfading placements at all. During the
//! transition the mesh and the impostor both draw in the main pass, each
//! keeping the pixels a per-pixel dither threshold assigns to it
//! (`lod_dither` in `lighting.wgsl`), so the two tile the object without
//! blending: depth, velocity and surface are written as for any opaque
//! object, and the fade is a stipple that TAA integrates. Crossfading
//! placements draw one by one. Only models with alpha-blended materials draw
//! in the transparent pass, after the sky and fog; a blended model that is
//! crossfading is dithered in the main pass like any other.
//!
//! # Usage
//!
//! ```no_run
//! use dream::camera::Camera;
//! use dream::gfx::Renderer;
//! use dream::model::shapes;
//! use dream::world::{LodModel, World};
//! use glam::{Mat4, Vec3};
//! use std::sync::Arc;
//!
//! fn setup(renderer: &Renderer) -> World {
//!     let mut world = World::new(renderer);
//!     let rock = Arc::new(shapes::uv_sphere(1.0, 32, 16, Default::default()));
//!     // Add a mesh model with a coarser level and a baked impostor when far away:
//!     // `.lod(..)` and `.impostor(..)` on the builder.
//!     let model = world
//!         .add_model(LodModel::builder(rock).build())
//!         .expect("model");
//!     world.spawn(model, Mat4::from_translation(Vec3::new(0.0, 1.0, -5.0)));
//!     world
//! }
//!
//! // Once per frame: `dt` in seconds, `ui` the UI frame or `None`.
//! fn frame(world: &mut World, renderer: &mut Renderer, camera: &Camera, dt: f32) {
//!     let stats = world
//!         .render(renderer, camera, dt, None, None)
//!         .expect("frame");
//!     assert!(stats.instances >= stats.frustum_culled);
//! }
//! ```

mod cache;
mod lod;

pub use lod::{ImpostorSource, LodModel, LodModelBuilder};

use crate::camera::Camera;
use crate::cull::{Bounds, Frustum};
use crate::gfx::{
    DEPTH_FORMAT, DrawCtx, Renderer, RtInstance, RtMeshHandle, SCENE_COLOR_FORMATS, SceneFrame,
    ScenePass, SurfaceError,
};
use crate::impostor::{
    AssetDebug, BakeSettings, FallbackReason, ImpostorAsset, ImpostorAssetError,
    ImpostorAssetLimits, ImpostorBaker, ImpostorBounds, ImpostorInstance, ImpostorRenderer,
    LodChoice, LodPolicy, static_capture_supported,
};
use crate::lighting::{Lighting, LightingEnv};
use crate::model::render::{InstanceTransform, ModelGpu, ModelRenderer};
use crate::model::{Model, ModelInstance};
use crate::ui::Rect;
use crate::ui::draw::UiFrame;
use glam::{Mat4, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Handle of a model added with [`World::add_model`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelHandle(u32);

/// Handle of a spawned instance. Stale after [`World::despawn`]: every
/// accessor then ignores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceHandle {
    index: u32,
    generation: u32,
}

/// Switches for the world's culling and LOD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSettings {
    /// Skip instances outside the camera frustum (and outside each shadow
    /// cascade). Default `true`.
    pub frustum_culling: bool,
    /// Skip instances hidden behind last frames' depth. Effective only when
    /// [`crate::render::RenderSettings::occlusion`] is set. Default `true`.
    pub occlusion_culling: bool,
    /// Scales screen coverage before the LOD decision: above 1 keeps finer
    /// levels longer. Default `1.0`.
    pub lod_bias: f32,
    /// Debug: keep culling and LOD against the camera from the frame this was
    /// switched on. Default `false`.
    pub freeze_culling: bool,
    /// Draw every model level that has two or more opaque, non-animated
    /// placements with one instanced draw per mesh (main and shadow passes).
    /// Off, every placement is its own draw. Default `true`.
    pub instancing: bool,
    /// Impostors whose projected diameter is below this many pixels take the
    /// far path: one baked view, no ray march. `0.0` disables it. Default
    /// [`crate::impostor::DEFAULT_FAR_PX`].
    pub impostor_far_px: f32,
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            frustum_culling: true,
            occlusion_culling: true,
            lod_bias: 1.0,
            freeze_culling: false,
            instancing: true,
            impostor_far_px: crate::impostor::DEFAULT_FAR_PX,
        }
    }
}

/// What one [`World::render`] call did. Counts are per frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldStats {
    /// Live instances.
    pub instances: u32,
    /// Instances outside the camera frustum.
    pub frustum_culled: u32,
    /// Instances behind the occlusion grid (main pass, two consecutive tests).
    pub occlusion_culled: u32,
    /// Instances beyond their model's `max_distance`.
    pub distance_culled: u32,
    /// Main-pass mesh draws per level (levels 3 and up share the last slot).
    pub mesh_draws: [u32; 4],
    /// Main-pass instances drawn as impostors, including crossfades.
    pub impostor_draws: u32,
    /// Instances drawn as a mesh and an impostor at once.
    pub crossfading: u32,
    /// Shadow-pass mesh draws, all cascades.
    pub shadow_draws: u32,
    /// Mesh draw calls issued in the main pass: an instanced draw counts once
    /// per mesh, a per-instance draw once per mesh. Impostors are not included.
    pub draw_calls: u32,
    /// Mesh draw calls issued in shadow passes, all cascades.
    pub shadow_draw_calls: u32,
    /// Placements drawn in the transparent pass, after the sky and fog: only
    /// models with alpha-blended materials at full mesh weight. Crossfades are
    /// opaque (dithered) and count in [`Self::mesh_draws`] instead.
    pub transparent_draws: u32,
    /// CPU time of the culling work: bounds, frustum, LOD, instance updates
    /// before the passes, plus the occlusion test in the main pass. Microseconds.
    pub cull_us: f32,
}

/// Errors from [`World::add_model`].
#[derive(Debug)]
pub enum WorldError {
    /// The impostor could not be baked (see [`FallbackReason`]) and
    /// [`LodModelBuilder::impostor_optional`] was not set.
    Impostor(FallbackReason),
    /// A baked impostor file could not be read or written.
    Asset(ImpostorAssetError),
    /// Writing the impostor cache failed.
    Io(std::io::Error),
    /// Mesh levels are not in decreasing-detail order, or a threshold is not
    /// a positive finite number. `level` is the offending index.
    InvalidLod { level: usize },
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorldError::Impostor(reason) => write!(f, "impostor bake failed: {reason}"),
            WorldError::Asset(e) => write!(f, "impostor asset: {e}"),
            WorldError::Io(e) => write!(f, "impostor cache: {e}"),
            WorldError::InvalidLod { level } => {
                write!(
                    f,
                    "mesh level {level} is out of order or has an invalid threshold"
                )
            }
        }
    }
}

impl std::error::Error for WorldError {}

/// Camera data culling and LOD use: the unjittered matrix, position and the
/// vertical half field-of-view tangent of the camera.
#[derive(Clone, Copy, Debug)]
struct CullView {
    vp: Mat4,
    position: Vec3,
    half_fov_tan: f32,
}

impl CullView {
    fn of(camera: &Camera) -> Self {
        Self {
            vp: camera.view_projection(),
            position: camera.position,
            half_fov_tan: (camera.fov_y() * 0.5).tan(),
        }
    }
}

/// One mesh level of a model on the GPU.
struct LevelGpu {
    model: Arc<Model>,
    gpu: ModelGpu,
    rt_meshes: Option<Vec<Option<RtMeshHandle>>>,
    below: f32,
}

/// A model added to the world.
struct ModelSlot {
    /// Some level has an alpha-blended material: its meshes draw in the
    /// transparent pass.
    has_blend: bool,
    levels: Vec<LevelGpu>,
    /// `levels[k].below`, kept as a flat array for the LOD search.
    belows: Vec<f32>,
    impostor: Option<ImpostorAsset>,
    /// Object-space box enclosing every level and the impostor.
    local: ImpostorBounds,
    policy: LodPolicy,
    cast_shadows: bool,
    shadow_lod: Option<usize>,
    max_distance: Option<f32>,
    fallback: Option<FallbackReason>,
}

/// One placed object.
struct Instance {
    model: u32,
    transform: Mat4,
    /// Transform latched by the last update, and the one before it: the
    /// current and previous impostor matrices.
    last_transform: Mat4,
    prev_transform: Mat4,
    bounds: Bounds,
    bounds_for: Mat4,
    bounds_valid: bool,
    impostor_capable: bool,
    /// Per level: the animated instance drawn at that level, created on first use.
    levels: Vec<Option<ModelInstance>>,
    /// Hysteresis state of the mesh level and of the impostor.
    mesh_level: u8,
    /// The representation the policy settles on (`Mesh` or `Impostor`). The
    /// hand-over thresholds live here; `choice` is what gets drawn.
    settled: LodChoice,
    /// Mesh weight moving toward `settled` over `lod::CROSSFADE_SECONDS`;
    /// `None` before the first frame.
    mesh_weight: Option<f32>,
    choice: LodChoice,
    /// Consecutive main-pass occlusion tests that found the instance hidden.
    occ_streak: u8,
    // Decisions made in this frame's pre-pass and main pass.
    main_visible: bool,
    distance_culled: bool,
    occluded: bool,
    shadow_level: u8,
}

struct Slot {
    generation: u32,
    instance: Option<Instance>,
}

/// The scene container. See the [module documentation](self).
pub struct World {
    ctx: Arc<crate::gfx::GpuContext>,
    settings: WorldSettings,
    env: LightingEnv,
    lighting: Lighting,
    model_renderer: ModelRenderer,
    impostor_renderer: ImpostorRenderer,
    models: Vec<ModelSlot>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    baker: Option<ImpostorBaker>,
    /// Impostor bakes performed, and impostors loaded from the cache.
    bakes: u32,
    cache_loads: u32,
    /// Culling view kept while `freeze_culling` is on.
    frozen: Option<CullView>,
    stats: WorldStats,
    // Per-frame scratch, reused so the frame path does not allocate.
    active: Vec<u32>,
    spheres: Vec<[f32; 4]>,
    visible: Vec<u32>,
    shadow_visible: Vec<u32>,
    jobs: Vec<u64>,
    transforms: Vec<InstanceTransform>,
    impostor_scratch: Vec<ImpostorInstance>,
    rt_scratch: Vec<RtInstance>,
    rt_scene_id: u64,
}

impl World {
    /// An empty world for `renderer`'s device.
    pub fn new(renderer: &Renderer) -> Self {
        let ctx = Arc::clone(renderer.context());
        let lighting = Lighting::new(&ctx);
        let model_renderer = ModelRenderer::new(&ctx, &SCENE_COLOR_FORMATS, DEPTH_FORMAT);
        let impostor_renderer =
            ImpostorRenderer::new(&ctx, &SCENE_COLOR_FORMATS, DEPTH_FORMAT, &lighting);
        Self {
            ctx,
            settings: WorldSettings::default(),
            env: LightingEnv::default(),
            lighting,
            model_renderer,
            impostor_renderer,
            models: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
            baker: None,
            bakes: 0,
            cache_loads: 0,
            frozen: None,
            stats: WorldStats::default(),
            active: Vec::new(),
            spheres: Vec::new(),
            visible: Vec::new(),
            shadow_visible: Vec::new(),
            jobs: Vec::new(),
            transforms: Vec::new(),
            impostor_scratch: Vec::new(),
            rt_scratch: Vec::new(),
            rt_scene_id: 0,
        }
    }

    /// Upload the model's levels and bake or load its impostor.
    pub fn add_model(&mut self, model: LodModel) -> Result<ModelHandle, WorldError> {
        let ctx = Arc::clone(&self.ctx);
        let mut previous = f32::INFINITY;
        for (level, mesh) in model.levels.iter().enumerate() {
            let ok = if level == 0 {
                mesh.below_coverage == f32::INFINITY
            } else {
                mesh.below_coverage.is_finite()
                    && mesh.below_coverage > 0.0
                    && mesh.below_coverage < previous
            };
            if !ok {
                return Err(WorldError::InvalidLod { level });
            }
            previous = mesh.below_coverage;
        }

        let policy = model.policy.sanitised();
        let base = Arc::clone(&model.levels[0].model);
        let mut fallback = None;
        let mut impostor = None;
        if let Some(source) = model.impostor {
            match self.impostor_from(&ctx, &base, source) {
                Ok(asset) => impostor = Some(asset),
                Err(WorldError::Impostor(reason)) if model.impostor_optional => {
                    fallback = Some(reason);
                }
                Err(e) => return Err(e),
            }
        }

        let mut local_min = Vec3::splat(f32::INFINITY);
        let mut local_max = Vec3::splat(f32::NEG_INFINITY);
        let mut levels = Vec::with_capacity(model.levels.len());
        for mesh in &model.levels {
            let (lo, hi) = mesh.model.static_bounds();
            local_min = local_min.min(lo);
            local_max = local_max.max(hi);
            levels.push(LevelGpu {
                gpu: ModelGpu::upload(&ctx, &mesh.model),
                model: Arc::clone(&mesh.model),
                rt_meshes: None,
                below: mesh.below_coverage,
            });
        }
        if let Some(asset) = &impostor {
            local_min = local_min.min(asset.bounds.min);
            local_max = local_max.max(asset.bounds.max);
        }
        let belows = levels.iter().map(|l| l.below).collect();
        let has_blend = model.levels.iter().any(|level| {
            level
                .model
                .materials
                .iter()
                .any(|m| m.alpha_mode == crate::model::AlphaMode::Blend)
        });
        let handle = ModelHandle(self.models.len() as u32);
        self.models.push(ModelSlot {
            has_blend,
            levels,
            belows,
            impostor,
            local: ImpostorBounds {
                min: local_min,
                max: local_max,
            },
            policy,
            cast_shadows: model.cast_shadows,
            shadow_lod: model.shadow_lod,
            max_distance: model.max_distance,
            fallback,
        });
        Ok(handle)
    }

    fn impostor_from(
        &mut self,
        ctx: &Arc<crate::gfx::GpuContext>,
        model: &Arc<Model>,
        source: ImpostorSource,
    ) -> Result<ImpostorAsset, WorldError> {
        match source {
            ImpostorSource::Asset(asset) => Ok(asset),
            ImpostorSource::Baked(path) => {
                ImpostorAsset::load_baked(ctx, &path, ImpostorAssetLimits::default())
                    .map_err(WorldError::Asset)
            }
            ImpostorSource::Bake(settings) => self.bake(ctx, model, settings),
            ImpostorSource::BakeCached { dir, settings } => {
                let key = cache::bake_key(model, &settings);
                let path: PathBuf = dir.join(cache::file_name(key));
                if path.is_file()
                    && let Ok(asset) =
                        ImpostorAsset::load_baked(ctx, &path, ImpostorAssetLimits::default())
                {
                    self.cache_loads += 1;
                    return Ok(asset);
                }
                let asset = self.bake(ctx, model, settings)?;
                std::fs::create_dir_all(&dir).map_err(WorldError::Io)?;
                asset.write_baked(&path).map_err(WorldError::Asset)?;
                Ok(asset)
            }
        }
    }

    fn bake(
        &mut self,
        ctx: &Arc<crate::gfx::GpuContext>,
        model: &Arc<Model>,
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, WorldError> {
        static_capture_supported(model).map_err(WorldError::Impostor)?;
        if self.baker.is_none() {
            self.baker = Some(ImpostorBaker::new(ctx));
        }
        let baker = self.baker.as_ref().expect("baker was just created");
        self.bakes += 1;
        baker.bake(model, settings).map_err(WorldError::Impostor)
    }

    /// Impostor bakes performed by this world, and impostors loaded from the
    /// cache instead. Lets a caller prove a cache hit.
    pub fn impostor_cache_counts(&self) -> (u32, u32) {
        (self.bakes, self.cache_loads)
    }

    /// Why a model's impostor is missing, when it was optional and could not
    /// be produced.
    pub fn impostor_fallback(&self, model: ModelHandle) -> Option<&FallbackReason> {
        self.models.get(model.0 as usize)?.fallback.as_ref()
    }

    /// Place an instance of `model`.
    ///
    /// # Panics
    /// If `model` did not come from this world's [`Self::add_model`].
    pub fn spawn(&mut self, model: ModelHandle, transform: Mat4) -> InstanceHandle {
        let slot_model = self
            .models
            .get(model.0 as usize)
            .expect("ModelHandle from another world");
        let count = slot_model.levels.len();
        let mut levels: Vec<Option<ModelInstance>> = (0..count).map(|_| None).collect();
        // Level 0 always exists: it is the instance `instance_mut` returns.
        let mut first = ModelInstance::new(Arc::clone(&slot_model.levels[0].model));
        first.set_transform(transform);
        first.update(0.0);
        levels[0] = Some(first);
        let instance = Instance {
            model: model.0,
            transform,
            last_transform: transform,
            prev_transform: transform,
            bounds: Bounds::from_aabb(Vec3::ZERO, Vec3::ZERO),
            bounds_for: transform,
            bounds_valid: false,
            impostor_capable: false,
            levels,
            mesh_level: 0,
            settled: LodChoice::Mesh,
            mesh_weight: None,
            choice: LodChoice::Mesh,
            occ_streak: 0,
            main_visible: false,
            distance_culled: false,
            occluded: false,
            shadow_level: 0,
        };
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.instance = Some(instance);
            InstanceHandle {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                instance: Some(instance),
            });
            InstanceHandle {
                index,
                generation: 0,
            }
        }
    }

    fn live_mut(&mut self, handle: InstanceHandle) -> Option<&mut Instance> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.instance.as_mut()
    }

    /// Move an instance. Ignored for a stale handle.
    pub fn set_transform(&mut self, instance: InstanceHandle, transform: Mat4) {
        if let Some(inst) = self.live_mut(instance) {
            inst.transform = transform;
        }
    }

    /// Remove an instance. Its handle becomes stale.
    pub fn despawn(&mut self, instance: InstanceHandle) {
        let Some(slot) = self.slots.get_mut(instance.index as usize) else {
            return;
        };
        if slot.generation != instance.generation || slot.instance.is_none() {
            return;
        }
        slot.instance = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(instance.index);
    }

    /// The instance's level-0 animation state, for playing clips or reading
    /// the pose. Other levels do not share it: see [`Self::spawn`].
    pub fn instance_mut(&mut self, instance: InstanceHandle) -> Option<&mut ModelInstance> {
        self.live_mut(instance)?.levels[0].as_mut()
    }

    /// The lighting environment the scene is drawn with.
    pub fn set_environment(&mut self, env: LightingEnv) {
        self.env = env;
    }

    /// The lighting environment the scene is drawn with.
    pub fn environment(&self) -> &LightingEnv {
        &self.env
    }

    pub fn settings_mut(&mut self) -> &mut WorldSettings {
        &mut self.settings
    }

    /// Statistics of the last [`Self::render_with`] call.
    pub fn stats(&self) -> &WorldStats {
        &self.stats
    }

    /// The lighting set the world draws with.
    pub fn lighting(&self) -> &Lighting {
        &self.lighting
    }

    /// The mesh renderer, for drawing extra geometry with the same pipelines.
    pub fn models(&mut self) -> &mut ModelRenderer {
        &mut self.model_renderer
    }

    /// Render one frame. See [`Self::render_with`].
    pub fn render(
        &mut self,
        renderer: &mut Renderer,
        camera: &Camera,
        dt: f32,
        ui: Option<&UiFrame>,
        scene_rect: Option<Rect>,
    ) -> Result<WorldStats, SurfaceError> {
        self.render_with(renderer, camera, dt, ui, scene_rect, |_| {})
    }

    /// [`Self::render`] plus `extra`, called in every pass after the world's
    /// own draws: custom geometry goes there, and it is drawn in shadow passes
    /// too when it casts shadows (check [`DrawCtx::is_shadow_pass`]).
    pub fn render_with<F>(
        &mut self,
        renderer: &mut Renderer,
        camera: &Camera,
        dt: f32,
        ui: Option<&UiFrame>,
        scene_rect: Option<Rect>,
        mut extra: F,
    ) -> Result<WorldStats, SurfaceError>
    where
        F: FnMut(&mut DrawCtx<'_>),
    {
        let screen_px = renderer.render_size().1 as f32;
        let live = CullView::of(camera);
        let cull = if self.settings.freeze_culling {
            *self.frozen.get_or_insert(live)
        } else {
            self.frozen = None;
            live
        };
        let settings = self.settings;
        let lod_bias = if settings.lod_bias.is_finite() {
            settings.lod_bias.max(0.0)
        } else {
            1.0
        };
        let env = self.env;

        let World {
            models,
            slots,
            active,
            spheres,
            visible,
            shadow_visible,
            jobs,
            transforms,
            impostor_scratch,
            rt_scratch,
            rt_scene_id,
            model_renderer,
            impostor_renderer,
            lighting,
            stats,
            ..
        } = self;

        // --- Pre-pass: bounds, frustum, LOD, then one update per instance ---
        let cull_start = Instant::now();
        *stats = WorldStats::default();
        active.clear();
        spheres.clear();
        for (index, slot) in slots.iter_mut().enumerate() {
            let Some(inst) = slot.instance.as_mut() else {
                continue;
            };
            let model = &models[inst.model as usize];
            if !inst.bounds_valid || inst.bounds_for != inst.transform {
                inst.bounds = Bounds::from_local(model.local.min, model.local.max, inst.transform);
                inst.bounds_for = inst.transform;
                inst.bounds_valid = true;
                inst.impostor_capable = model.impostor.is_some()
                    && ImpostorInstance::try_new(inst.transform, inst.transform, 1.0).is_ok();
            }
            let c = inst.bounds.centre;
            spheres.push([c.x, c.y, c.z, inst.bounds.radius]);
            active.push(index as u32);
            inst.main_visible = false;
            inst.occluded = false;
            inst.distance_culled = false;
        }
        stats.instances = active.len() as u32;

        let frustum = Frustum::from_view_projection(cull.vp);
        visible.clear();
        if settings.frustum_culling {
            frustum.cull_spheres(spheres, visible);
        } else {
            visible.extend(0..active.len() as u32);
        }
        stats.frustum_culled = active.len() as u32 - visible.len() as u32;
        for &a in visible.iter() {
            let slot = active[a as usize] as usize;
            if let Some(inst) = slots[slot].instance.as_mut() {
                inst.main_visible = true;
            }
        }

        for &s in active.iter() {
            let Some(inst) = slots[s as usize].instance.as_mut() else {
                continue;
            };
            let model = &models[inst.model as usize];
            let bounds = inst.bounds;
            let distance = (bounds.centre - cull.position).length();
            if !inst.main_visible {
                inst.occ_streak = 0;
            }
            if model
                .max_distance
                .is_some_and(|limit| distance - bounds.radius > limit)
            {
                inst.distance_culled = true;
                inst.occ_streak = 0;
                stats.distance_culled += 1;
                inst.choice = LodChoice::Mesh;
                inst.settled = LodChoice::Mesh;
                inst.mesh_weight = None;
                continue;
            }
            let coverage =
                LodPolicy::screen_coverage_fov(bounds.radius, distance, cull.half_fov_tan)
                    * lod_bias;
            let inside = ImpostorBounds {
                min: bounds.aabb_min,
                max: bounds.aabb_max,
            }
            .contains(cull.position);

            // Impostor state. The hand-over is the policy's hysteresis, with
            // its max-distance rule applied as `LodPolicy::select` applies it.
            // That decides the settled representation; the mesh weight then
            // moves to it over `CROSSFADE_SECONDS`, so a crossfade is a timed
            // transition and a still camera draws one representation.
            let capable = inst.impostor_capable;
            if capable && !inside {
                let policy = model.policy;
                let synthetic = (policy.impostor_below as f64 * policy.max_distance as f64
                    / distance.max(1e-3) as f64) as f32;
                // `fade: 0` makes the coverage decision a pure hand-over.
                inst.settled = LodPolicy {
                    fade: 0.0,
                    ..policy
                }
                .select_coverage(coverage.min(synthetic.max(0.0)), inst.settled);
                inst.choice = lod::timed_choice(&mut inst.mesh_weight, inst.settled, dt);
            } else {
                inst.settled = LodChoice::Mesh;
                inst.mesh_weight = Some(1.0);
                inst.choice = LodChoice::Mesh;
            }

            let level = lod::pick_level(&model.belows, inst.mesh_level as usize, coverage);
            inst.mesh_level = level as u8;
            let last = model.levels.len() - 1;
            let shadow_level =
                model
                    .shadow_lod
                    .map(|l| l.min(last))
                    .unwrap_or(if inst.choice.draws_mesh() {
                        level
                    } else {
                        last
                    });
            inst.shadow_level = shadow_level as u8;
        }

        // Create the animated instances this frame needs, then update each
        // exactly once (new ones were initialised at level zero time).
        for &s in active.iter() {
            let Some(inst) = slots[s as usize].instance.as_mut() else {
                continue;
            };
            if inst.distance_culled {
                continue;
            }
            let model = &models[inst.model as usize];
            let mut wanted = [usize::MAX; 2];
            if inst.choice.draws_mesh() {
                wanted[0] = inst.mesh_level as usize;
            }
            if model.cast_shadows {
                wanted[1] = inst.shadow_level as usize;
            }
            for level in wanted.into_iter().filter(|&l| l != usize::MAX) {
                if inst.levels[level].is_none() {
                    let mut created = ModelInstance::new(Arc::clone(&model.levels[level].model));
                    created.set_transform(inst.transform);
                    created.update(0.0);
                    inst.levels[level] = Some(created);
                }
            }
        }
        for &s in active.iter() {
            let Some(inst) = slots[s as usize].instance.as_mut() else {
                continue;
            };
            inst.prev_transform = inst.last_transform;
            inst.last_transform = inst.transform;
            for level in inst.levels.iter_mut().flatten() {
                level.set_transform(inst.transform);
                level.update(dt);
            }
        }
        let mut cull_us = cull_start.elapsed().as_secs_f32() * 1e6;

        // Build a distance-limited RT instance list from all live world
        // objects, including those outside the camera frustum or occluded in
        // screen space. Keep the mesh LOD chosen above; impostors never enter
        // the TLAS. Only static opaque geometry has a supported BLAS.
        let render_settings = renderer.render_settings();
        let rt_active = render_settings.ray_tracing.filter(|rt| {
            (rt.reflections && render_settings.reflections.is_some())
                || (rt.ambient && (render_settings.ao.is_some() || render_settings.gi.is_some()))
        });
        if let Some(rt_settings) = rt_active
            && let Some(rt_scene) = renderer.ray_tracing_scene_mut()
        {
            if *rt_scene_id != rt_scene.id() {
                for model in models.iter_mut() {
                    for level in &mut model.levels {
                        level.rt_meshes = None;
                    }
                }
                *rt_scene_id = rt_scene.id();
            }
            rt_scratch.clear();
            let max_distance = rt_settings.distance.clamp(10.0, 2000.0);
            for &slot_index in active.iter() {
                let Some(inst) = slots[slot_index as usize].instance.as_ref() else {
                    continue;
                };
                if inst.distance_culled {
                    continue;
                }
                let model = &models[inst.model as usize];
                let distance = (inst.bounds.centre - cull.position).length();
                if !distance.is_finite() || distance - inst.bounds.radius > max_distance {
                    continue;
                }
                let level_index = (inst.mesh_level as usize).min(model.levels.len() - 1);
                let level = &mut models[inst.model as usize].levels[level_index];
                if level.rt_meshes.is_none() {
                    level.rt_meshes = Some(level.gpu.register_rt_meshes(rt_scene));
                }
                let Some(handles) = level.rt_meshes.as_ref() else {
                    continue;
                };
                for (mesh_index, handle) in handles.iter().enumerate() {
                    let Some(mesh) = handle else { continue };
                    let node = level.model.mesh_node_transform(mesh_index);
                    rt_scratch.push(RtInstance {
                        mesh: *mesh,
                        transform: inst.transform * node,
                    });
                }
            }
            rt_scene.set_instances(rt_scratch);
        }

        let lighting: &Lighting = lighting;
        let frame = SceneFrame {
            camera,
            lighting,
            env,
            scene_rect,
            ui,
        };
        let result = renderer.render_scene(&frame, |ctx: &mut DrawCtx<'_>| {
            if ctx.is_shadow_pass() {
                // --- Shadow cascade: every caster in this cascade, no occlusion ---
                let cascade = Frustum::from_view_projection(ctx.view_projection());
                shadow_visible.clear();
                if settings.frustum_culling {
                    cascade.cull_spheres(spheres, shadow_visible);
                } else {
                    shadow_visible.extend(0..active.len() as u32);
                }
                jobs.clear();
                for &a in shadow_visible.iter() {
                    let s = active[a as usize] as usize;
                    let Some(inst) = slots[s].instance.as_ref() else {
                        continue;
                    };
                    if inst.distance_culled || !models[inst.model as usize].cast_shadows {
                        continue;
                    }
                    let level = inst.shadow_level as u64;
                    jobs.push(((inst.model as u64) << 40) | (level << 32) | s as u64);
                }
                jobs.sort_unstable();
                let calls_before = model_renderer.draw_calls();
                for_each_group(&jobs[..], |group| {
                    let level = group_level(group);
                    let model = group_model(group);
                    let placed = draw_mesh_group(
                        model_renderer,
                        ctx,
                        &mut models[model].levels[level].gpu,
                        lighting,
                        group,
                        slots,
                        transforms,
                        settings.instancing,
                        true,
                    );
                    stats.shadow_draws += placed.drawn;
                });
                stats.shadow_draw_calls += (model_renderer.draw_calls() - calls_before) as u32;
                extra(ctx);
                return 0;
            }

            // --- Transparent pass: after the sky and fog, the placements of
            // models whose materials alpha-blend, at full mesh weight. Their
            // per-frame state (LOD, occlusion, instance transforms) was decided
            // by the main pass; nothing here updates it again. Crossfades are
            // opaque and dithered in the main pass instead. ---
            if ctx.pass() == ScenePass::Transparent {
                jobs.clear();
                for &a in visible.iter() {
                    let s = active[a as usize] as usize;
                    let Some(inst) = slots[s].instance.as_ref() else {
                        continue;
                    };
                    if inst.distance_culled
                        || inst.occluded
                        || !inst.choice.draws_mesh()
                        || inst.choice.mesh_weight() < 1.0
                    {
                        continue;
                    }
                    if models[inst.model as usize].has_blend {
                        let level = inst.mesh_level as u64;
                        jobs.push(((inst.model as u64) << 40) | (level << 32) | s as u64);
                    }
                }
                jobs.sort_unstable();
                let mut placed = 0u32;
                for_each_group(&jobs[..], |group| {
                    let level = group_level(group);
                    let model = group_model(group);
                    for &key in group {
                        let inst = job_instance(slots, key);
                        let mi = inst.levels[level].as_ref().expect("main level created");
                        model_renderer.draw_pbr_with_opacity(
                            ctx,
                            &mut models[model].levels[level].gpu,
                            mi,
                            lighting,
                            1.0,
                        );
                        placed += 1;
                    }
                });
                stats.transparent_draws = placed;
                return placed as usize;
            }

            // --- Main pass ---
            // Occlusion: the grid is last frames' depth, tested from where the
            // camera was. Two consecutive hidden results are needed to cull.
            let occlusion_start = Instant::now();
            let grid = if settings.occlusion_culling {
                ctx.occlusion()
            } else {
                None
            };
            if let Some(grid) = grid {
                let motion = grid.camera_motion(cull.position);
                for &a in visible.iter() {
                    let s = active[a as usize] as usize;
                    let Some(inst) = slots[s].instance.as_mut() else {
                        continue;
                    };
                    if inst.distance_culled {
                        continue;
                    }
                    // Camera motion since the grid, times three, plus a quarter of
                    // the radius. Nearer occluders shift more than the tested box
                    // does, so motion alone leaves slivers at silhouettes: with 1x
                    // the world_check culling diff reached 0.109% strong pixels
                    // (limit 0.05%), 2x 0.041%, 3x 0.024%.
                    let extra_size = 3.0 * motion + 0.25 * inst.bounds.radius;
                    let hidden = grid.is_occluded_inflated(&inst.bounds, extra_size);
                    inst.occ_streak = if hidden {
                        inst.occ_streak.saturating_add(1)
                    } else {
                        0
                    };
                    inst.occluded = hidden && inst.occ_streak >= 2;
                }
            }
            cull_us += occlusion_start.elapsed().as_secs_f32() * 1e6;

            let mut drawn = 0u32;
            jobs.clear();
            for &a in visible.iter() {
                let s = active[a as usize] as usize;
                let Some(inst) = slots[s].instance.as_ref() else {
                    continue;
                };
                if inst.distance_culled || inst.occluded {
                    continue;
                }
                // Blended models at full weight draw in the transparent pass.
                // Crossfades of any model are dithered opaque here.
                let crossfade = inst.choice.mesh_weight() < 1.0;
                let blended = models[inst.model as usize].has_blend;
                if inst.choice.draws_mesh() && (crossfade || !blended) {
                    let level = inst.mesh_level as u64;
                    jobs.push(((inst.model as u64) << 40) | (level << 32) | s as u64);
                }
            }
            if grid.is_some() {
                stats.occlusion_culled = visible
                    .iter()
                    .filter(|&&a| {
                        slots[active[a as usize] as usize]
                            .instance
                            .as_ref()
                            .is_some_and(|i| i.occluded)
                    })
                    .count() as u32;
            }
            jobs.sort_unstable();
            let calls_before = model_renderer.draw_calls();
            for_each_group(&jobs[..], |group| {
                let level = group_level(group);
                let model = group_model(group);
                let placed = draw_mesh_group(
                    model_renderer,
                    ctx,
                    &mut models[model].levels[level].gpu,
                    lighting,
                    group,
                    slots,
                    transforms,
                    settings.instancing,
                    false,
                );
                stats.mesh_draws[level.min(3)] += placed.drawn;
                // Mesh and impostor both drawn this frame: a crossfade.
                stats.crossfading += placed.crossfading;
                drawn += placed.drawn;
            });
            stats.draw_calls = (model_renderer.draw_calls() - calls_before) as u32;

            // Impostors: one draw per asset, with every instance using it.
            jobs.clear();
            for &a in visible.iter() {
                let s = active[a as usize] as usize;
                let Some(inst) = slots[s].instance.as_ref() else {
                    continue;
                };
                if inst.distance_culled || inst.occluded || !inst.choice.draws_impostor() {
                    continue;
                }
                if models[inst.model as usize].impostor.is_some() {
                    jobs.push(((inst.model as u64) << 32) | s as u64);
                }
            }
            jobs.sort_unstable();
            let mut i = 0;
            while i < jobs.len() {
                let model = (jobs[i] >> 32) as usize;
                let mut end = i;
                impostor_scratch.clear();
                while end < jobs.len() && (jobs[end] >> 32) as usize == model {
                    let slot = (jobs[end] & 0xffff_ffff) as usize;
                    let inst = slots[slot].instance.as_ref().expect("job is live");
                    let alpha = inst.choice.impostor_weight();
                    if let Ok(instance) =
                        ImpostorInstance::try_new(inst.last_transform, inst.prev_transform, alpha)
                    {
                        impostor_scratch.push(instance);
                        stats.impostor_draws += 1;
                        drawn += 1;
                    }
                    end += 1;
                }
                if !impostor_scratch.is_empty()
                    && let Some(asset) = models[model].impostor.as_mut()
                {
                    impostor_renderer.far_px = settings.impostor_far_px;
                    impostor_renderer.draw(
                        ctx,
                        asset,
                        impostor_scratch,
                        lighting,
                        screen_px,
                        cull.half_fov_tan,
                        AssetDebug::Final,
                    );
                }
                i = end;
            }

            extra(ctx);
            drawn as usize
        });
        stats.cull_us = cull_us;
        result.map(|_| *stats)
    }
}

/// Result of [`draw_mesh_group`].
#[derive(Clone, Copy, Debug, Default)]
struct GroupDraw {
    /// Placements drawn, instanced or not (zero-opacity ones count too, as the
    /// per-instance statistics always did).
    drawn: u32,
    /// Placements drawn as a mesh and an impostor at once.
    crossfading: u32,
}

/// Job key layout: `model << 40 | level << 32 | slot`. Jobs sorted by key come
/// out grouped by model and level.
fn group_model(group: &[u64]) -> usize {
    (group[0] >> 40) as usize
}

fn group_level(group: &[u64]) -> usize {
    ((group[0] >> 32) & 0xff) as usize
}

/// Calls `f` once per run of keys sharing a model and level.
fn for_each_group(keys: &[u64], mut f: impl FnMut(&[u64])) {
    let mut start = 0;
    while start < keys.len() {
        let group = keys[start] >> 32;
        let len = keys[start..]
            .iter()
            .take_while(|&&key| key >> 32 == group)
            .count();
        f(&keys[start..start + len]);
        start += len;
    }
}

/// The placed level instance of a job key, which must be live.
fn job_instance(slots: &[Slot], key: u64) -> &Instance {
    let slot = (key & 0xffff_ffff) as usize;
    slots[slot].instance.as_ref().expect("job is live")
}

/// Whether a placement's level may join an instanced draw: opaque (the mesh
/// weight is one, always in shadow passes) and not animated.
fn instanceable(inst: &Instance, level: usize, shadow: bool) -> bool {
    let opaque = shadow || inst.choice.mesh_weight() >= 1.0;
    let animated = inst.levels[level]
        .as_ref()
        .is_some_and(|mi| mi.is_animating());
    opaque && !animated
}

/// Draw one (model, level) group of placements. Two or more instanceable
/// placements go out as one instanced draw; everything else is drawn one by
/// one, as before instancing existed.
#[allow(clippy::too_many_arguments)]
fn draw_mesh_group(
    renderer: &mut ModelRenderer,
    ctx: &mut DrawCtx<'_>,
    gpu: &mut ModelGpu,
    lighting: &Lighting,
    group: &[u64],
    slots: &[Slot],
    scratch: &mut Vec<InstanceTransform>,
    instancing: bool,
    shadow: bool,
) -> GroupDraw {
    let level = group_level(group);
    let mut out = GroupDraw::default();
    let instanceable_count = if instancing {
        group
            .iter()
            .filter(|&&key| instanceable(job_instance(slots, key), level, shadow))
            .count()
    } else {
        0
    };
    let use_instanced = instanceable_count >= 2;
    if use_instanced {
        scratch.clear();
        for &key in group {
            let inst = job_instance(slots, key);
            if !instanceable(inst, level, shadow) {
                continue;
            }
            let mi = inst.levels[level].as_ref().expect("level created");
            let (model, prev_model) = mi.motion_matrices();
            scratch.push(InstanceTransform {
                model: model.to_cols_array_2d(),
                prev_model: prev_model.to_cols_array_2d(),
            });
        }
        renderer.draw_pbr_instanced(ctx, gpu, scratch, lighting);
        out.drawn += instanceable_count as u32;
    }
    for &key in group {
        let inst = job_instance(slots, key);
        if use_instanced && instanceable(inst, level, shadow) {
            continue;
        }
        let mi = inst.levels[level].as_ref().expect("level created");
        // Shadow passes always cast the full mesh. A crossfade in the main pass
        // is the mesh's dithered share of the pixels; the impostor takes the rest.
        let weight = inst.choice.mesh_weight();
        if shadow || weight >= 1.0 {
            renderer.draw_pbr_with_opacity(ctx, gpu, mi, lighting, 1.0);
        } else {
            renderer.draw_pbr_dithered(ctx, gpu, mi, lighting, weight);
        }
        out.drawn += 1;
        if !shadow && inst.choice.draws_impostor() {
            out.crossfading += 1;
        }
    }
    out
}

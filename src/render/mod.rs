//! Scene render settings and the cascaded sun-shadow maths.
//!
//! Set options with [`crate::gfx::Renderer::set_render_settings`]. Nothing is
//! allocated for a setting that is off, so UI-only apps pay nothing.
//!
//! # Starting points
//!
//! [`RenderSettings::off`], [`RenderSettings::low`], [`RenderSettings::medium`],
//! [`RenderSettings::high`] and [`RenderSettings::ultra`] are one-call
//! starting points. Each documents what it enables; edit them field by field.
//! Pair a preset with [`AaMode::Taa`](crate::settings::AaMode::Taa) for native
//! rendering, or [`AaMode::Fsr`](crate::settings::AaMode::Fsr) when upscaling.
//!
//! # Optional hardware rays
//!
//! Request ray-traced AO, GI and reflections independently of the presets:
//!
//! ```no_run
//! # fn enable(renderer: &mut dream::gfx::Renderer) {
//! use dream::render::{RayTracingSettings, RenderSettings};
//! let mut settings = RenderSettings::high();
//! settings.ray_tracing = Some(RayTracingSettings::default());
//! renderer.set_render_settings(settings);
//! # }
//! ```
//!
//! [`crate::world::World`] registers the selected static mesh levels within
//! [`RayTracingSettings::distance`], including objects outside the view or
//! hidden by others. It uses meshes for rays even when the view uses impostors.
//! Per-mesh acceleration structures are cached; the per-frame structure is
//! rebuilt in its frame slot after the slot fence signals. The GPU timing
//! breakdown includes this work as `TLAS build`.
//!
//! Custom scene owners can register buffers and stage instances through
//! [`crate::gfx::Renderer::ray_tracing_scene_mut`]. Unsupported devices and
//! scenes with no registered instances use the screen-space traces. Support
//! requires the full ray-query and triangle-position-fetch feature set; check
//! [`crate::gfx::GpuContext::ray_tracing_supported`]. RT resources are created
//! on demand. Turning RT off releases its geometry cache; call
//! [`crate::gfx::Renderer::clear_ray_tracing_scene`] when discarding a world
//! while keeping RT enabled. World re-registers after a reset; manual handles
//! must be registered again.
//!
//! The RT ambient pass shares one cosine-weighted ray per working pixel for AO
//! and GI, followed by the same spatial and temporal filters. Reflections use
//! one roughness-dependent GGX ray and reuse the current scene radiance when a
//! hit agrees with its depth. Other hits use material factors, emission, sky
//! and a shadow-tested sun. Both tiers feed the next frame's forward shading.
//!
//! # The frame
//!
//! [`crate::gfx::Renderer::render_scene`] runs these stages in order. Each
//! stage is on only when its setting is `Some` (or `true`) in
//! [`RenderSettings`].
//! The optional per-frame ray-tracing structure is built before these passes.
//!
//! 1. **Shadow cascades** ([`RenderSettings::shadows`]): the scene is drawn
//!    depth-only once per cascade that updates this frame, with
//!    [`crate::gfx::DrawCtx::pass`] returning [`crate::gfx::ScenePass::Shadow`].
//! 2. **Scene pass** ([`ScenePass::Main`](crate::gfx::ScenePass::Main)): opaque
//!    and alpha-masked geometry is drawn into the scene colour, velocity and
//!    surface attachments and the depth buffer. Alpha-blended geometry waits for
//!    step 6.
//! 3. **Hi-Z** ([`RenderSettings::occlusion`]): the depth is reduced to a small
//!    grid, read back a few frames later for [`crate::cull::OcclusionBuffer`].
//! 4. **Sky** ([`RenderSettings::atmosphere`]): the physically based sky is drawn
//!    where depth is still the far plane. With the atmosphere on, the sky-view LUT
//!    and the cloud panorama also fill a 256 x 128 equirectangular environment map
//!    with a mip chain (one per frame slot). Ambient specular and rough reflections
//!    read it, so they see the sky and a low-resolution version of the clouds.
//!    Lighting reads the map recorded in the previous frame, as it reads the
//!    screen-space results.
//! 5. **Clouds** ([`RenderSettings::clouds`]): the view clouds are ray-marched per
//!    pixel at quarter resolution, reprojected into a half-resolution history, and
//!    composited over the sky. They stop at the scene's depth, so geometry in or
//!    behind a cloud is hidden by it. Before this pass, a top-down shadow map of
//!    the cloud layer is recorded, and lighting and fog read it.
//! 6. **Fog** ([`RenderSettings::fog`]): volumetric in-scatter and transmittance
//!    blended over the scene colour, using the depth buffer.
//! 7. **Transparent pass** ([`ScenePass::Transparent`](crate::gfx::ScenePass::Transparent)):
//!    alpha-blended geometry is composited over the sky and fog. It loads colour,
//!    velocity, surface and depth, tests depth without writing it, and writes
//!    colour only. Begun only when something draws in it. It runs after fog, so
//!    blended surfaces are not fogged (see the limitations).
//! 8. **AO and GI** ([`RenderSettings::ao`], [`RenderSettings::gi`],
//!    [`RenderSettings::indirect_quality`]): a horizon march over the depth and
//!    surface data, or a scene ray query when RT ambient is enabled.
//! 9. **Reflections** ([`RenderSettings::reflections`]): a screen-space trace of
//!    the current scene's lit colour, denoised over time and space.
//!    RT reflections replace the screen trace with a scene ray query.
//! 10. **Anti-aliasing**: native TAA ([`RenderSettings::taa`]) or FSR 3.1
//!     upscaling, chosen by [`crate::settings::AaMode`].
//! 11. **Tone curve and blit** ([`RenderSettings::tonemap`],
//!     [`RenderSettings::exposure`], [`RenderSettings::dither`]) to the
//!     swapchain.
//! 12. **UI**: the [`crate::ui`] frame is drawn last, on top.
//!
//! Linear depth is recorded at the start of the post work, before the sky and
//! the clouds, because the view clouds read it per pixel. It is full resolution
//! whenever the clouds are on.
//!
//! The order is deliberate. Transparent draws must sit over the sky and fog,
//! so they come after both. They must not touch the surface and velocity data
//! that the screen-space chains and TAA read, so their pipelines mask those
//! two attachments and the pass never writes depth. The chains therefore see
//! the opaque surfaces only, and a blended pixel's colour enters them as the
//! colour it was composited to.
//!
//! # Feedback is one frame late
//!
//! AO, GI and reflections are computed from this frame's buffers, but the
//! forward shading of frame N+1 reads their results. Shading therefore lags a
//! fast change in the camera or the light by one frame, and the indirect
//! results converge over several frames.
//!
//! # Cost
//!
//! Measured with the `preset_*` cases of `cargo run --release --bin
//! scene_check` on the development machine (NVIDIA GeForce RTX 5080) at
//! 1280x720 with TAA. Three GPU times are reported separately, each read from
//! the last frame of each case: the sun shadow cascades, the scene pass, and the
//! post-pass work (sky, fog, the transparent pass, AO, GI, reflections, TAA and
//! blit). Two full runs gave the ranges shown:
//!
//! | Preset | Shadow cascades | Scene pass | Post pass |
//! |--------|-----------------|------------|-----------|
//! | `low`    | 0.010 to 0.013 ms | 0.050 to 0.055 ms | 0.32 ms |
//! | `medium` | 0.023 ms          | 0.069 to 0.084 ms | 0.53 to 0.54 ms |
//! | `high`   | 0.025 to 0.027 ms | 0.081 to 0.082 ms | 0.74 to 1.25 ms |
//! | `ultra`  | 0.049 ms          | 0.085 to 0.087 ms | 1.82 to 2.21 ms |
//!
//! These are rough guidance for that machine, not a budget. The scene pass
//! barely changes with the preset because the test scene is small, and the
//! shadow passes are cheap at this scale. The post-pass range is the widest
//! and the least stable. The split between features was not measured
//! separately.
//!
//! Clouds add to the post pass (the cloud shadow map is recorded before the
//! scene pass). Measured by `scene_check`'s view-cloud checks on the development
//! machine with TAA, over three runs, the post-pass GPU time was as follows.
//! Ground camera on the horizon, 1280x720: 0.08 to 0.09 ms with clouds off, 0.33
//! to 0.37 ms with [`CloudQuality::Low`], 0.43 to 1.01 ms with `Medium` and 0.49
//! to 0.65 ms with `High`. Inside the layer, 1280x720: 0.08 to 0.10, 0.28 to 0.30,
//! 0.35 to 0.38 and 0.40 to 0.45 ms. Horizon at 2560x1440: 0.21 to 0.22 ms with
//! clouds off, and 0.80 to 3.5 ms with clouds on, which varied most between runs.
//! The view cost scales with the render size and the view steps; the panorama and
//! the shadow map are fixed-size.
//!
//! # Custom scene shaders
//!
//! A shader that draws into the scene pass writes three colour outputs, in
//! this order:
//!
//! - `@location(0)`: linear HDR colour;
//! - `@location(1)`: velocity, `prev_uv - curr_uv` in the offscreen targets'
//!   texture space, from the current and previous unjittered matrices;
//! - `@location(2)`: the surface encoding from `surface_encode` in
//!   `lighting.wgsl` (octahedral normal, roughness, and a class: 1 dielectric,
//!   2 or 3 metallic). Zero is reserved for "no surface", so a shader that
//!   writes it must write a non-zero class.
//!
//! Shadow passes need depth only. Check `ctx.pass()` (or
//! [`crate::gfx::DrawCtx::is_shadow_pass`]) and skip the colour work there; the
//! shadow pipelines have no colour attachments. A fragment that shades calls
//! `lighting_begin(frag_coord.xy)` first and then `lighting_eval(...)`, so the
//! screen-space noise and feedback reads use the right pixel.
//!
//! `lighting_eval` reads the sky environment map through set 3 bindings 8 (the
//! map) and 9 (its sampler), so a shader that includes the lighting module needs
//! no change for it. The binding table is on [`crate::lighting::Lighting`]. When
//! the atmosphere drives the lighting, the sky colours are irradiance over pi,
//! and diffuse ambient uses them through `sky_irradiance`.
//!
//! Transparent geometry has its own pass. When
//! [`crate::gfx::DrawCtx::has_transparent_pass`] is true (always, under
//! `render_scene`), draw alpha-blended geometry only when `ctx.pass()` is
//! [`ScenePass::Transparent`](crate::gfx::ScenePass::Transparent), and opaque
//! geometry only in `Main`. The transparent pipeline must blend colour and
//! mask both the velocity and the surface attachments to zero writes, as
//! [`crate::model::render::ModelRenderer`]'s transparent pipelines do. The
//! built-in vertex helpers ([`crate::gfx::DrawCtx::draw_indexed`] and the
//! pulled draws) do nothing in the transparent pass, because they bind the
//! opaque scene pipeline; `DrawCtx::raw` does not check the pass.
//!
//! # Known limitations
//!
//! - The default screen-space AO, GI and reflections only see what is on screen. Off-screen
//!   geometry neither occludes nor bounces light. GI has no off-screen bounce.
//!   Where a reflection ray leaves the screen or misses, the result falls back
//!   to the sky environment map (or the gradient with no atmosphere).
//! - The feedback is one frame late (see above).
//! - Impostors receive sun shadows but do not cast them. [`crate::world::World`]
//!   draws a mesh level in shadow passes instead.
//! - The cloud shadow map covers a 16 km square around the camera at 62.5 m a
//!   texel, with one sun path of 12 samples per texel. It is redrawn every frame.
//!   Clouds beyond the square, and shadows of geometry on clouds, are not
//!   modelled. Clouds do not shadow the sky or the sun's disk.
//! - The view clouds blend valid history with a weight of 0.94 (an effective
//!   history of about seventeen frames). The
//!   reprojection assumes each cloud point stays at its marched distance, so a
//!   cloud seen at a new parallax smears while the camera translates fast. Cloud
//!   motion has no velocity in the TAA buffers (sky velocity is rotation-only), so
//!   TAA does not correct it.
//! - Reflection misses use the sky environment map, a 256 x 128 panorama of the
//!   clouds refreshed a sixteenth of its texels per frame. It is not reprojected,
//!   so reflected clouds lag a translation and are blurrier than the view clouds.
//!   Local geometry does not show in it: the map is sky only, so a reflection
//!   that leaves the screen shows the sky where a building stands. With the
//!   atmosphere off, misses use the three-colour gradient.
//!   Enable RT reflections to trace registered local geometry outside the view.
//! - RT geometry is static and opaque: skinned and alpha-blended meshes are
//!   skipped, and alpha masks are treated as solid triangles. Instance transforms
//!   can move each frame, but mesh deformation is not traced. Secondary hits use
//!   flat triangle normals and material factors without textures or normal maps.
//!   GI gathers one diffuse bounce; there are no recursive specular reflections.
//!   The registration radius bounds which objects can cast secondary shadows or
//!   contribute to rays, and meshes outside it are absent. RT still has temporal
//!   convergence noise and the one-frame feedback delay.
//! - Inside the layer the cloud in front of a surface is treated as homogeneous
//!   between the front and back distances, so the in-scatter of a short path
//!   is an approximation of the marched radiance.
//! - Only one directional light, the sun. There are no point or spot lights.
//! - Skinned meshes use the simple non-PBR path ([`crate::model::render::ModelRenderer::draw`]).
//!   [`crate::model::render::ModelRenderer::draw_pbr`] and its instanced form
//!   skip them.
//! - Occlusion culling under fast camera motion relies on bounds inflation
//!   (three times the camera motion plus a quarter of the radius). Without it,
//!   silhouettes show slivers, which the `world_check` culling diff measures.
//! - Alpha-blended surfaces are not fogged. The transparent pass runs after
//!   fog, so the fog integral never sees them. A blended surface is drawn over
//!   the fogged opaque scene, and the fog it would carry is missing.
//! - Alpha-blended pixels keep the motion vectors and surface data of the
//!   opaque surface behind them, so TAA reprojects a blended pixel with the
//!   background's motion. Screen-space AO, GI and reflections see the blended
//!   colour, but the depth and surface data behind it.
//! - LoD crossfades are dithered, not blended: the mesh and the impostor each
//!   keep a per-pixel share of the object, so there is no see-through halo and
//!   both write depth, velocity and surface. The fade is a stipple until TAA
//!   or FSR integrates it. Without temporal AA the stipple stays visible on
//!   the crossfading band.

mod atmosphere;
mod clouds;
mod fog;
mod settings;
mod shadows;

pub use atmosphere::{
    AtmosphereModel, AtmosphereSettings, SKY_ENV_MIPS, SKY_ENV_SIZE, SKYVIEW_SIZE,
    sky_env_direction, sky_env_uv, sky_view_direction, sky_view_uv,
};
#[cfg(not(target_os = "macos"))]
pub(crate) use atmosphere::{
    MULTISCATTER_SIZE, PlanetKm, SKY_IRRADIANCE_SCALE, TRANSMITTANCE_SIZE,
};
pub use clouds::{CloudQuality, CloudSettings};
#[cfg(not(target_os = "macos"))]
pub(crate) use clouds::{
    MAX_MARCH_DISTANCE, PANORAMA_SIZE, PANORAMA_STEPS, PANORAMA_STRIDE, quality_params,
    refresh_offset,
};
pub use fog::{FogResolution, FogSettings};
pub use settings::{
    AoSettings, GiSettings, IndirectQuality, OcclusionSettings, RayTracingSettings,
    ReflectionSettings, RenderSettings, ShadowFilter, ShadowSettings, TaaSettings, Tonemap,
};
#[cfg(not(target_os = "macos"))]
pub(crate) use shadows::{CascadeFit, fit_cascades};

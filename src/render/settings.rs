//! Render settings the host chooses. Everything here is off by default:
//! [`RenderSettings::default`] allocates no GPU resources at all.

use super::atmosphere::AtmosphereSettings;
use super::clouds::{CloudQuality, CloudSettings};
use super::fog::{FogResolution, FogSettings};

/// Scene-wide rendering options, applied with
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RenderSettings {
    /// Dynamic sun shadows. `None` (the default) turns them off and frees the
    /// shadow map.
    pub shadows: Option<ShadowSettings>,
    /// Tuning of the native TAA resolve; used when the renderer runs with
    /// [`crate::settings::AaMode::Taa`].
    pub taa: TaaSettings,
    /// Tone curve applied by the final blit. [`Tonemap::None`] (the default)
    /// passes linear scene colour through unchanged.
    pub tonemap: Tonemap,
    /// Multiplier on scene colour before [`Self::tonemap`]. Applied only when
    /// a tone curve is on. Default `1.0`.
    pub exposure: f32,
    /// Add `±0.5` LSB noise before the final 8-bit quantisation, to break up
    /// banding in smooth gradients. Default `false`.
    pub dither: bool,
    /// Screen-space ambient occlusion on ambient light. `None` (the default)
    /// turns it off. Runs the indirect chain together with [`Self::gi`].
    pub ao: Option<AoSettings>,
    /// Screen-space one-bounce indirect diffuse light. `None` (the default)
    /// turns it off. Runs the indirect chain together with [`Self::ao`].
    pub gi: Option<GiSettings>,
    /// Shared sample budget of the AO and GI march; see [`IndirectQuality`].
    pub indirect_quality: IndirectQuality,
    /// Screen-space reflections. `None` (the default) turns them off and frees
    /// their images; see [`ReflectionSettings`].
    pub reflections: Option<ReflectionSettings>,
    /// Physically based sky and sun-driven lighting. `None` (the default) turns
    /// it off and frees its LUTs; see [`AtmosphereSettings`].
    pub atmosphere: Option<AtmosphereSettings>,
    /// Volumetric fog with shadowed light shafts. `None` (the default) turns it
    /// off and frees the fog volume; see [`FogSettings`].
    pub fog: Option<FogSettings>,
    /// Volumetric clouds, marched into a sky panorama and composited over the
    /// sky pass. `None` (the default) turns them off and frees their images.
    /// They need the atmosphere for their colours: when `atmosphere` is `None`
    /// the sky uses `AtmosphereSettings::default()` with `drive_lighting` off.
    /// See [`CloudSettings`].
    pub clouds: Option<CloudSettings>,
    /// Hi-Z occlusion data: the scene depth is reduced to a small grid and
    /// read back a few frames later for `crate::cull::OcclusionBuffer`.
    /// `None` (the default) allocates nothing and records no pass.
    pub occlusion: Option<OcclusionSettings>,
    /// Hardware ray tracing of the reflection and ambient traces. `None` (the
    /// default, and in every preset) keeps the screen-space traces. Only takes
    /// effect where the device supports ray tracing; see [`RayTracingSettings`].
    pub ray_tracing: Option<RayTracingSettings>,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            shadows: None,
            taa: TaaSettings::default(),
            tonemap: Tonemap::None,
            exposure: 1.0,
            dither: false,
            ao: None,
            gi: None,
            indirect_quality: IndirectQuality::default(),
            reflections: None,
            ray_tracing: None,
            atmosphere: None,
            fog: None,
            clouds: None,
            occlusion: None,
        }
    }
}

/// Hardware ray tracing for the reflection and ambient (AO and GI) traces.
///
/// Turns the traces into ray queries against the scene's acceleration
/// structures (see [`crate::gfx::RayTracingScene`]): geometry off screen and
/// hidden behind other objects is then seen too. The existing
/// [`ReflectionSettings`] and [`AoSettings`] / [`GiSettings`] still set the
/// strength, roughness cut-off, resolution and filtering; these fields only
/// choose the trace.
///
/// It takes effect only on devices that support ray tracing, and only while
/// the scene has instances (see [`crate::world::World`] or
/// [`crate::gfx::Renderer::ray_tracing_scene_mut`]).
/// Elsewhere the screen-space traces run unchanged. The presets leave it off.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RayTracingSettings {
    /// Reflection rays query the scene. Default `true`.
    pub reflections: bool,
    /// AO and GI rays query the scene. Default `true`.
    pub ambient: bool,
    /// Radius in metres around the camera within which live instances are
    /// placed in the ray-traced scene. Clamped to `10..=2000`. Default `150.0`.
    pub distance: f32,
    /// Longest reflection ray, in metres, clamped to `1..=2000`. Default `150.0`.
    pub reflection_distance: f32,
    /// Longest ambient (GI) ray, in metres, clamped to `0.5..=500`. Default `30.0`.
    pub ambient_distance: f32,
}

impl Default for RayTracingSettings {
    fn default() -> Self {
        Self {
            reflections: true,
            ambient: true,
            distance: 150.0,
            reflection_distance: 150.0,
            ambient_distance: 30.0,
        }
    }
}

impl RayTracingSettings {
    /// Copy with every distance finite and clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        let default = Self::default();
        Self {
            reflections: self.reflections,
            ambient: self.ambient,
            distance: finite(self.distance, default.distance).clamp(10.0, 2000.0),
            reflection_distance: finite(self.reflection_distance, default.reflection_distance)
                .clamp(1.0, 2000.0),
            ambient_distance: finite(self.ambient_distance, default.ambient_distance)
                .clamp(0.5, 500.0),
        }
    }
}

/// Settings of the Hi-Z depth readback behind occlusion culling. Build with
/// `Default` and set fields; the struct is `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct OcclusionSettings {
    /// Width of the readback grid in texels (clamped to `64..=512`). The
    /// height follows the render aspect ratio.
    pub width: u32,
    /// Screen-space inflation of every tested rectangle, in grid texels.
    pub margin_texels: f32,
    /// Relative slack on the depth comparison. Larger values cull less.
    pub depth_bias: f32,
}

impl Default for OcclusionSettings {
    fn default() -> Self {
        Self {
            width: 256,
            margin_texels: 1.5,
            depth_bias: 0.002,
        }
    }
}

impl OcclusionSettings {
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        let default = Self::default();
        Self {
            width: self.width.clamp(64, 512),
            margin_texels: finite(self.margin_texels, default.margin_texels).clamp(0.0, 16.0),
            depth_bias: finite(self.depth_bias, default.depth_bias).clamp(0.0, 0.1),
        }
    }
}

/// Screen-space ambient occlusion, computed by a horizon march over the depth
/// and normals of the previous scene pass. It attenuates ambient light only;
/// direct sunlight is never touched. Build from [`Default::default`] and assign
/// the fields you change; the struct is `#[non_exhaustive]`.
///
/// Values outside their documented range are clamped by
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AoSettings {
    /// Search radius in metres, `0.05..=50`. Occluders further away do not
    /// darken a point. Default `1.5`.
    pub radius: f32,
    /// Strength, `0..=2`: the visibility is raised to this power, so `0` turns
    /// the effect off and `2` darkens creases more. Default `1.0`.
    pub intensity: f32,
    /// Assumed thickness of occluders in metres, `0..=10`. Limits how far
    /// behind a depth edge a surface can still count as occluding. Default `0.5`.
    pub thickness: f32,
    /// Brighten occlusion on bright albedo (Jimenez et al. 2016), so a white
    /// surface in a corner does not look as dark as a black one. Default `true`.
    pub multi_bounce: bool,
    /// Also attenuate ambient specular with the occlusion (Lagarde's
    /// approximation, scaled by roughness). Default `true`.
    pub specular_occlusion: bool,
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            radius: 1.5,
            intensity: 1.0,
            thickness: 0.5,
            multi_bounce: true,
            specular_occlusion: true,
        }
    }
}

impl AoSettings {
    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        Self {
            radius: finite(self.radius, 1.5).clamp(0.05, 50.0),
            intensity: finite(self.intensity, 1.0).clamp(0.0, 2.0),
            thickness: finite(self.thickness, 0.5).clamp(0.0, 10.0),
            multi_bounce: self.multi_bounce,
            specular_occlusion: self.specular_occlusion,
        }
    }
}

/// One-bounce indirect diffuse light gathered from the previous lit frame.
/// Multiple bounces build up over frames, because each frame gathers the
/// result of the last. Build from [`Default::default`] and assign the fields
/// you change; the struct is `#[non_exhaustive]`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GiSettings {
    /// Multiplier on the gathered light, `0..=4`. Default `1.0`.
    pub intensity: f32,
    /// Gather radius in metres, `0.1..=50`: points further than this do not
    /// contribute light. Default `6.0`.
    pub radius: f32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            radius: 6.0,
        }
    }
}

impl GiSettings {
    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        Self {
            intensity: finite(self.intensity, 1.0).clamp(0.0, 4.0),
            radius: finite(self.radius, 6.0).clamp(0.1, 50.0),
        }
    }
}

/// Sample budget of the screen-space AO and GI march, shared by both.
///
/// Build from a preset ([`Self::low`], [`Self::medium`], [`Self::high`],
/// [`Self::ultra`]) and adjust the fields you change. Values are clamped by
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct IndirectQuality {
    /// Directions marched per pixel, `1..=4`.
    pub slices: u32,
    /// Steps per direction and side, `2..=16`.
    pub steps: u32,
    /// March and filter at half resolution in each axis (`low` and `medium`).
    pub half_res: bool,
    /// Run the edge-aware spatial and temporal filters over the raw result.
    /// Off, the raw horizon output is used as is (noisier).
    pub denoise: bool,
}

impl IndirectQuality {
    /// 1 slice, 4 steps per side, half resolution.
    pub fn low() -> Self {
        Self {
            slices: 1,
            steps: 4,
            half_res: true,
            denoise: true,
        }
    }

    /// 2 slices, 6 steps per side, half resolution. The default.
    pub fn medium() -> Self {
        Self {
            slices: 2,
            steps: 6,
            half_res: true,
            denoise: true,
        }
    }

    /// 2 slices, 8 steps per side, full resolution.
    pub fn high() -> Self {
        Self {
            slices: 2,
            steps: 8,
            half_res: false,
            denoise: true,
        }
    }

    /// 4 slices, 12 steps per side, full resolution.
    pub fn ultra() -> Self {
        Self {
            slices: 4,
            steps: 12,
            half_res: false,
            denoise: true,
        }
    }

    /// Copy with `slices` and `steps` clamped to their documented ranges.
    pub(crate) fn sanitised(&self) -> Self {
        Self {
            slices: self.slices.clamp(1, 4),
            steps: self.steps.clamp(2, 16),
            ..*self
        }
    }
}

impl Default for IndirectQuality {
    fn default() -> Self {
        Self::medium()
    }
}

/// Screen-space reflections: rays are marched through the depth of the previous
/// scene pass and the lit colour found at each hit is gathered. Where a ray hits
/// a surface the result replaces the analytic sky reflection. Off by default.
///
/// Build from a preset ([`Self::low`], [`Self::medium`], [`Self::high`],
/// [`Self::ultra`]) or [`Default::default`] (medium), and adjust the fields you
/// change. The struct is `#[non_exhaustive]`. Values are clamped by
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReflectionSettings {
    /// Blend of the traced reflection over the sky fallback, `0..=1`. Default `1.0`.
    pub intensity: f32,
    /// Surfaces at or above this perceptual roughness get no traced reflection.
    /// The confidence fades in over the 0.15 below it. Default `0.6`.
    pub max_roughness: f32,
    /// Longest ray, in metres. Default `40.0`.
    pub max_distance: f32,
    /// Assumed depth in metres of the surfaces behind the depth buffer, which
    /// bounds how far a ray may pass behind a visible surface. Default `0.35`.
    pub thickness: f32,
    /// March steps per ray, clamped to `8..=96`.
    pub steps: u32,
    /// Binary refinement steps after the first hit, clamped to `0..=8`.
    pub refine_steps: u32,
    /// Trace and filter at half resolution in each axis.
    pub half_res: bool,
    /// Temporal and spatial filtering of the traced result. Off, the raw trace
    /// is used as is (noisier on rough surfaces).
    pub denoise: bool,
    /// Fraction of the screen over which hits fade out toward the borders,
    /// clamped to `0..=0.5`. Default `0.1`.
    pub edge_fade: f32,
}

impl ReflectionSettings {
    /// 16 steps, 2 refinement steps, half resolution.
    pub fn low() -> Self {
        Self {
            steps: 16,
            refine_steps: 2,
            half_res: true,
            ..Self::medium()
        }
    }

    /// 32 steps, 4 refinement steps, half resolution. The default.
    pub fn medium() -> Self {
        Self {
            intensity: 1.0,
            max_roughness: 0.6,
            max_distance: 40.0,
            thickness: 0.35,
            steps: 32,
            refine_steps: 4,
            half_res: true,
            denoise: true,
            edge_fade: 0.1,
        }
    }

    /// 48 steps, 5 refinement steps, full resolution.
    pub fn high() -> Self {
        Self {
            steps: 48,
            refine_steps: 5,
            half_res: false,
            ..Self::medium()
        }
    }

    /// 80 steps, 6 refinement steps, full resolution.
    pub fn ultra() -> Self {
        Self {
            steps: 80,
            refine_steps: 6,
            half_res: false,
            ..Self::medium()
        }
    }

    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
        Self {
            intensity: finite(self.intensity, 1.0).clamp(0.0, 1.0),
            max_roughness: finite(self.max_roughness, 0.6).clamp(0.05, 1.0),
            max_distance: finite(self.max_distance, 40.0).clamp(1.0, 500.0),
            thickness: finite(self.thickness, 0.35).clamp(0.01, 5.0),
            steps: self.steps.clamp(8, 96),
            refine_steps: self.refine_steps.min(8),
            edge_fade: finite(self.edge_fade, 0.1).clamp(0.0, 0.5),
            ..*self
        }
    }
}

impl Default for ReflectionSettings {
    fn default() -> Self {
        Self::medium()
    }
}

/// Tuning of temporal anti-aliasing (`AaMode::Taa`). Build from
/// [`Default::default`] and assign the fields you change; the struct is
/// `#[non_exhaustive]`.
///
/// Values outside their range are clamped by
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TaaSettings {
    /// Weight of the reprojected history when the image is static, `0..=0.98`.
    /// Higher is smoother and slower to react to change. Default `0.93`.
    pub history_weight: f32,
    /// Size of the neighbourhood clip box in standard deviations, `0.25..=4`.
    /// Smaller clips history harder (less ghosting, more shimmer). Default `1.0`.
    pub variance_gamma: f32,
    /// Post-resolve adaptive sharpening strength, `0..=1`. `0` skips the
    /// sharpen pass and blits the resolved history. Default `0.35`.
    pub sharpness: f32,
    /// Lower the new-frame weight near the clip box boundary, so highlights
    /// that hover at the edge of the box stop flickering. Default `true`.
    pub anti_flicker: bool,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            history_weight: 0.93,
            variance_gamma: 1.0,
            sharpness: 0.35,
            anti_flicker: true,
        }
    }
}

impl TaaSettings {
    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        Self {
            history_weight: self.history_weight.clamp(0.0, 0.98),
            variance_gamma: self.variance_gamma.clamp(0.25, 4.0),
            sharpness: self.sharpness.clamp(0.0, 1.0),
            anti_flicker: self.anti_flicker,
        }
    }
}

/// Tone curve for the final blit. Applied in linear light, before the sRGB
/// encode of the display path. Curves other than `None` also map scene
/// colour above 1.0 into the display range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tonemap {
    /// Linear pass-through (the default; output is unchanged).
    #[default]
    None,
    /// Narkowicz's fitted ACES filmic curve (an approximation of the ACES RRT
    /// and ODT, not the full matrix transform).
    Aces,
    /// The minimal AgX approximation with its base look: inset, log2 encode,
    /// the sigmoid contrast fit, outset, then a 2.2 power to linear.
    AgX,
    /// Khronos PBR Neutral: keeps base colours, desaturates highlights only.
    Neutral,
}

/// How shadow edges are filtered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowFilter {
    /// One comparison per pixel: aliased edges, cheapest.
    Hard,
    /// Percentage-closer filtering: `taps` comparisons on a rotated disk of
    /// radius [`ShadowSettings::softness`] texels.
    Pcf,
    /// Contact-hardening: a blocker search estimates the penumbra per pixel,
    /// then PCF runs with that radius. Shadows widen with distance from their
    /// caster.
    Soft,
}

/// Parameters of the directional sun shadow map. Build one with a preset
/// ([`Self::low`] .. [`Self::ultra`]) and assign the fields you want to change;
/// the struct is `#[non_exhaustive]`, so it cannot be built field by field
/// from outside the crate.
///
/// Values outside their documented range are clamped by
/// [`crate::gfx::Renderer::set_render_settings`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ShadowSettings {
    /// Number of cascades, `1..=4`. Each cascade covers a slice of the view
    /// frustum with its own map layer; more cascades give sharper shadows near
    /// the camera at a cost of one extra depth pass per cascade.
    pub cascades: u32,
    /// Side length of each cascade's square depth map, in texels, `256..=8192`.
    pub resolution: u32,
    /// Distance from the camera, in metres, that shadows reach. Beyond it the
    /// scene is lit without shadows (see [`Self::fade`]).
    pub distance: f32,
    /// Split distribution between the cascades: `0.0` gives uniform slices,
    /// `1.0` gives logarithmic slices (more resolution near the camera).
    pub split_lambda: f32,
    /// Edge filter; see [`ShadowFilter`].
    pub filter: ShadowFilter,
    /// Filter samples per pixel, `1..=32`. Used by `Pcf` and `Soft`.
    pub taps: u32,
    /// For `Pcf`: kernel radius in texels. For `Soft`: the light's angular
    /// size, as penumbra width per metre between caster and receiver. The
    /// real sun is about `0.01`; `0.03..0.06` reads as a soft sunny day and
    /// larger values as an overcast one.
    pub softness: f32,
    /// Constant depth bias added to every shadow-map write, in the
    /// rasteriser's depth units. Raise it if shadows show acne.
    pub depth_bias: f32,
    /// Depth bias proportional to the triangle's slope, applied with
    /// [`Self::depth_bias`].
    pub slope_bias: f32,
    /// Offset of each lookup along the surface normal, in shadow texels. This
    /// removes acne at grazing angles without the peter-panning a large
    /// `depth_bias` causes.
    pub normal_bias: f32,
    /// Shadow darkness: `0.0` disables the shadow's effect on the sun term,
    /// `1.0` makes shadowed areas receive no sunlight.
    pub strength: f32,
    /// Fraction of [`Self::distance`], `0..=1`, over which shadows fade out
    /// toward the far end of their range.
    pub fade: f32,
    /// How far, in metres, toward the sun casters outside the camera's view can
    /// be and still cast into it. Anything further out is not captured.
    pub caster_distance: f32,
    /// Re-render cascade 2 every second frame and cascade 3 every fourth. Their
    /// far-away content changes slowly, so this trades a little shadow latency
    /// for GPU time.
    pub stagger_far_cascades: bool,
    /// Tint shaded surfaces by the cascade that covers them (debug aid).
    pub debug_cascades: bool,
}

impl ShadowSettings {
    /// Cheapest preset: 2 cascades of 1024 texels over 60 m, 4-tap PCF,
    /// far cascades staggered.
    pub fn low() -> Self {
        Self {
            cascades: 2,
            resolution: 1024,
            distance: 60.0,
            split_lambda: 0.75,
            filter: ShadowFilter::Pcf,
            taps: 4,
            softness: 1.5,
            depth_bias: 1.25,
            slope_bias: 1.75,
            normal_bias: 1.0,
            strength: 1.0,
            fade: 0.2,
            caster_distance: 60.0,
            stagger_far_cascades: true,
            debug_cascades: false,
        }
    }

    /// Balanced preset: 3 cascades of 2048 texels over 120 m, 8-tap PCF.
    pub fn medium() -> Self {
        Self {
            cascades: 3,
            resolution: 2048,
            distance: 120.0,
            taps: 8,
            caster_distance: 120.0,
            stagger_far_cascades: false,
            ..Self::low()
        }
    }

    /// Sharp preset: 4 cascades of 2048 texels over 200 m, contact-hardening
    /// soft shadows with 12 taps: crisp at the contact, widening at 4 cm per
    /// metre of caster distance.
    pub fn high() -> Self {
        Self {
            cascades: 4,
            distance: 200.0,
            filter: ShadowFilter::Soft,
            taps: 12,
            softness: 0.04,
            caster_distance: 200.0,
            ..Self::medium()
        }
    }

    /// Highest preset: 4 cascades of 4096 texels over 300 m, soft shadows with
    /// 16 taps.
    pub fn ultra() -> Self {
        Self {
            resolution: 4096,
            distance: 300.0,
            taps: 16,
            caster_distance: 300.0,
            ..Self::high()
        }
    }

    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let mut s = self.clone();
        s.cascades = s.cascades.clamp(1, 4);
        s.resolution = s.resolution.clamp(256, 8192);
        s.taps = s.taps.clamp(1, 32);
        s.distance = s.distance.max(1.0);
        s.split_lambda = s.split_lambda.clamp(0.0, 1.0);
        s.strength = s.strength.clamp(0.0, 1.0);
        s.fade = s.fade.clamp(0.0, 1.0);
        s.caster_distance = s.caster_distance.max(0.0);
        s.softness = s.softness.max(0.0);
        s.normal_bias = s.normal_bias.max(0.0);
        s
    }
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self::medium()
    }
}

impl RenderSettings {
    /// Copy with every nested value clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        Self {
            shadows: self.shadows.as_ref().map(ShadowSettings::sanitised),
            taa: self.taa.sanitised(),
            tonemap: self.tonemap,
            exposure: if self.exposure.is_finite() {
                self.exposure.max(0.0)
            } else {
                1.0
            },
            dither: self.dither,
            ao: self.ao.as_ref().map(AoSettings::sanitised),
            gi: self.gi.as_ref().map(GiSettings::sanitised),
            indirect_quality: self.indirect_quality.sanitised(),
            reflections: self.reflections.as_ref().map(ReflectionSettings::sanitised),
            atmosphere: self.atmosphere.as_ref().map(AtmosphereSettings::sanitised),
            fog: self.fog.as_ref().map(FogSettings::sanitised),
            clouds: self.clouds.as_ref().map(CloudSettings::sanitised),
            occlusion: self.occlusion.as_ref().map(OcclusionSettings::sanitised),
            ray_tracing: self.ray_tracing.as_ref().map(RayTracingSettings::sanitised),
        }
    }
}

impl RenderSettings {
    /// Every optional feature off. Same as [`Default::default`]: nothing is
    /// allocated, and the tone curve and exposure are the pass-through values.
    pub fn off() -> Self {
        Self::default()
    }

    /// Lowest preset: sun shadows (low), ambient occlusion at the low indirect
    /// budget, sky and sun lighting (earth atmosphere), light haze at the low
    /// fog grid, clouds at low quality, Hi-Z occlusion data, ACES tone curve.
    /// No global illumination or reflections.
    ///
    /// Starting point: edit field by field. Pair with `AaMode::Taa` on native
    /// rendering or `AaMode::Fsr` when upscaling.
    pub fn low() -> Self {
        let mut fog = FogSettings::light_haze();
        fog.resolution = FogResolution::Low;
        let mut clouds = CloudSettings::fair_weather();
        clouds.quality = CloudQuality::Low;
        Self {
            shadows: Some(ShadowSettings::low()),
            ao: Some(AoSettings::default()),
            indirect_quality: IndirectQuality::low(),
            atmosphere: Some(AtmosphereSettings::earth()),
            fog: Some(fog),
            clouds: Some(clouds),
            occlusion: Some(OcclusionSettings::default()),
            tonemap: Tonemap::Aces,
            ..Self::default()
        }
    }

    /// Middle preset: medium sun shadows, ambient occlusion and one-bounce
    /// global illumination at the medium indirect budget, low screen-space
    /// reflections, sky and sun lighting, medium fog at the medium grid,
    /// medium clouds, Hi-Z occlusion data, ACES tone curve.
    ///
    /// Starting point: edit field by field. Pair with `AaMode::Taa` on native
    /// rendering or `AaMode::Fsr` when upscaling.
    pub fn medium() -> Self {
        Self {
            shadows: Some(ShadowSettings::medium()),
            ao: Some(AoSettings::default()),
            gi: Some(GiSettings::default()),
            indirect_quality: IndirectQuality::medium(),
            reflections: Some(ReflectionSettings::low()),
            atmosphere: Some(AtmosphereSettings::earth()),
            fog: Some(FogSettings::medium()),
            clouds: Some(CloudSettings::fair_weather()),
            occlusion: Some(OcclusionSettings::default()),
            tonemap: Tonemap::Aces,
            ..Self::default()
        }
    }

    /// Upper-middle preset: high sun shadows, ambient occlusion and global
    /// illumination at the high indirect budget (full resolution), high
    /// reflections (full resolution), sky and sun lighting, medium fog at the
    /// medium grid, medium clouds, Hi-Z occlusion data, ACES tone curve.
    ///
    /// Starting point: edit field by field. Pair with `AaMode::Taa` on native
    /// rendering or `AaMode::Fsr` when upscaling.
    pub fn high() -> Self {
        Self {
            shadows: Some(ShadowSettings::high()),
            ao: Some(AoSettings::default()),
            gi: Some(GiSettings::default()),
            indirect_quality: IndirectQuality::high(),
            reflections: Some(ReflectionSettings::high()),
            atmosphere: Some(AtmosphereSettings::earth()),
            fog: Some(FogSettings::medium()),
            clouds: Some(CloudSettings::fair_weather()),
            occlusion: Some(OcclusionSettings::default()),
            tonemap: Tonemap::Aces,
            ..Self::default()
        }
    }

    /// Top preset: every feature at its highest preset: ultra sun shadows,
    /// ultra indirect budget for AO and GI, ultra reflections, sky and sun
    /// lighting, medium fog at the high fog grid, high-quality clouds,
    /// Hi-Z occlusion data, ACES tone curve. Costs the most GPU time.
    ///
    /// Starting point: edit field by field. Pair with `AaMode::Taa` on native
    /// rendering or `AaMode::Fsr` when upscaling.
    pub fn ultra() -> Self {
        let mut fog = FogSettings::medium();
        fog.resolution = FogResolution::High;
        let mut clouds = CloudSettings::fair_weather();
        clouds.quality = CloudQuality::High;
        Self {
            shadows: Some(ShadowSettings::ultra()),
            ao: Some(AoSettings::default()),
            gi: Some(GiSettings::default()),
            indirect_quality: IndirectQuality::ultra(),
            reflections: Some(ReflectionSettings::ultra()),
            atmosphere: Some(AtmosphereSettings::earth()),
            fog: Some(fog),
            clouds: Some(clouds),
            occlusion: Some(OcclusionSettings::default()),
            tonemap: Tonemap::Aces,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadows_presets_are_ordered_and_default_is_off() {
        assert_eq!(RenderSettings::default().shadows, None);
        let presets = [
            ShadowSettings::low(),
            ShadowSettings::medium(),
            ShadowSettings::high(),
            ShadowSettings::ultra(),
        ];
        for pair in presets.windows(2) {
            assert!(pair[0].cascades <= pair[1].cascades);
            assert!(pair[0].resolution <= pair[1].resolution);
            assert!(pair[0].distance < pair[1].distance);
        }
        assert_eq!(ShadowSettings::default(), ShadowSettings::medium());
    }

    #[test]
    fn render_settings_defaults_leave_output_unchanged() {
        let settings = RenderSettings::default();
        assert_eq!(settings.tonemap, Tonemap::None);
        assert_eq!(settings.exposure, 1.0);
        assert!(!settings.dither);
        assert_eq!(settings.taa, TaaSettings::default());
        assert_eq!(TaaSettings::default().history_weight, 0.93);
        assert!(TaaSettings::default().anti_flicker);
    }

    #[test]
    fn ray_tracing_settings_default_off_and_sanitise_clamp() {
        assert!(RenderSettings::default().ray_tracing.is_none());
        assert!(RenderSettings::off().ray_tracing.is_none());
        assert!(RenderSettings::low().ray_tracing.is_none());
        assert!(RenderSettings::ultra().ray_tracing.is_none());
        let defaults = RayTracingSettings::default();
        assert!(defaults.reflections && defaults.ambient);
        assert_eq!(defaults.distance, 150.0);
        assert_eq!(defaults.reflection_distance, 150.0);
        assert_eq!(defaults.ambient_distance, 30.0);
        let mut wild = RayTracingSettings {
            distance: 1.0e9,
            reflection_distance: f32::NAN,
            ambient_distance: -4.0,
            ..RayTracingSettings::default()
        };
        let clean = wild.sanitised();
        assert_eq!(clean.distance, 2000.0);
        assert_eq!(clean.reflection_distance, 150.0);
        assert_eq!(clean.ambient_distance, 0.5);
        wild.distance = 0.0;
        assert_eq!(wild.sanitised().distance, 10.0);
    }

    #[test]
    fn taa_and_exposure_sanitise_clamp_documented_ranges() {
        let mut settings = RenderSettings::default();
        settings.taa.history_weight = 2.0;
        settings.taa.variance_gamma = 0.0;
        settings.taa.sharpness = -1.0;
        settings.exposure = -3.0;
        let s = settings.sanitised();
        assert_eq!(s.taa.history_weight, 0.98);
        assert_eq!(s.taa.variance_gamma, 0.25);
        assert_eq!(s.taa.sharpness, 0.0);
        assert_eq!(s.exposure, 0.0);
        settings.exposure = f32::NAN;
        assert_eq!(settings.sanitised().exposure, 1.0);
    }

    #[test]
    fn shadows_sanitise_clamps_documented_ranges() {
        let mut s = ShadowSettings::medium();
        s.cascades = 9;
        s.resolution = 10;
        s.taps = 0;
        s.strength = 3.0;
        let s = s.sanitised();
        assert_eq!(s.cascades, 4);
        assert_eq!(s.resolution, 256);
        assert_eq!(s.taps, 1);
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn indirect_quality_presets_are_ordered_and_default_is_medium() {
        let (low, medium, high, ultra) = (
            IndirectQuality::low(),
            IndirectQuality::medium(),
            IndirectQuality::high(),
            IndirectQuality::ultra(),
        );
        assert_eq!(IndirectQuality::default(), medium);
        assert!(low.slices <= medium.slices && medium.slices <= high.slices);
        assert!(high.slices <= ultra.slices);
        assert!(low.steps < medium.steps && medium.steps < high.steps);
        assert!(high.steps < ultra.steps);
        assert!(low.half_res && medium.half_res);
        assert!(!high.half_res && !ultra.half_res);
        assert_eq!((ultra.slices, ultra.steps), (4, 12));
    }

    #[test]
    fn indirect_quality_and_ao_gi_sanitise_clamp_documented_ranges() {
        let quality = IndirectQuality {
            slices: 0,
            steps: 99,
            ..IndirectQuality::medium()
        }
        .sanitised();
        assert_eq!((quality.slices, quality.steps), (1, 16));
        let quality = IndirectQuality {
            slices: 40,
            steps: 0,
            ..IndirectQuality::medium()
        }
        .sanitised();
        assert_eq!((quality.slices, quality.steps), (4, 2));

        let ao = AoSettings {
            radius: 1000.0,
            intensity: -1.0,
            thickness: f32::NAN,
            ..AoSettings::default()
        }
        .sanitised();
        assert_eq!(ao.radius, 50.0);
        assert_eq!(ao.intensity, 0.0);
        assert_eq!(ao.thickness, 0.5);

        let gi = GiSettings {
            intensity: 9.0,
            radius: -3.0,
        }
        .sanitised();
        assert_eq!(gi.intensity, 4.0);
        assert_eq!(gi.radius, 0.1);
    }

    #[test]
    fn indirect_defaults_leave_the_chain_off() {
        let settings = RenderSettings::default();
        assert!(settings.ao.is_none());
        assert!(settings.gi.is_none());
        assert_eq!(settings.indirect_quality, IndirectQuality::medium());
        assert_eq!(AoSettings::default().radius, 1.5);
        assert_eq!(GiSettings::default().radius, 6.0);
    }

    #[test]
    fn reflection_default_is_off_and_medium_is_the_default_preset() {
        assert_eq!(RenderSettings::default().reflections, None);
        assert_eq!(ReflectionSettings::default(), ReflectionSettings::medium());
    }

    #[test]
    fn reflection_presets_are_ordered_by_cost() {
        let presets = [
            ReflectionSettings::low(),
            ReflectionSettings::medium(),
            ReflectionSettings::high(),
            ReflectionSettings::ultra(),
        ];
        for pair in presets.windows(2) {
            assert!(pair[0].steps < pair[1].steps);
            assert!(pair[0].refine_steps <= pair[1].refine_steps);
            // Half resolution is only ever dropped, never added, going up.
            assert!(pair[0].half_res || !pair[1].half_res);
        }
        assert!(ReflectionSettings::low().half_res);
        assert!(!ReflectionSettings::ultra().half_res);
    }

    #[test]
    fn reflection_settings_clamp_to_documented_ranges() {
        let wild = ReflectionSettings {
            intensity: 3.0,
            max_roughness: -1.0,
            max_distance: 1.0e9,
            thickness: f32::NAN,
            steps: 1000,
            refine_steps: 99,
            edge_fade: 0.9,
            ..ReflectionSettings::medium()
        };
        let clean = wild.sanitised();
        assert_eq!(clean.intensity, 1.0);
        assert_eq!(clean.max_roughness, 0.05);
        assert_eq!(clean.max_distance, 500.0);
        assert_eq!(clean.thickness, 0.35);
        assert_eq!(clean.steps, 96);
        assert_eq!(clean.refine_steps, 8);
        assert_eq!(clean.edge_fade, 0.5);
        let low = ReflectionSettings {
            steps: 0,
            intensity: -2.0,
            ..ReflectionSettings::medium()
        }
        .sanitised();
        assert_eq!(low.steps, 8);
        assert_eq!(low.intensity, 0.0);
        // Sanitising a sanitised value changes nothing.
        assert_eq!(clean.sanitised(), clean);
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn render_presets_survive_the_sanitiser_unchanged() {
        for (name, preset) in [
            ("off", RenderSettings::off()),
            ("low", RenderSettings::low()),
            ("medium", RenderSettings::medium()),
            ("high", RenderSettings::high()),
            ("ultra", RenderSettings::ultra()),
        ] {
            assert_eq!(preset.sanitised(), preset, "preset {name} changed");
        }
    }

    #[test]
    fn render_preset_off_is_default() {
        assert_eq!(RenderSettings::off(), RenderSettings::default());
    }

    #[test]
    fn render_presets_step_up_in_cost_fields() {
        let presets = [
            RenderSettings::low(),
            RenderSettings::medium(),
            RenderSettings::high(),
            RenderSettings::ultra(),
        ];
        for pair in presets.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(a.indirect_quality.steps <= b.indirect_quality.steps);
            let res = |s: &Option<ShadowSettings>| s.as_ref().map_or(0, |s| s.resolution);
            assert!(res(&a.shadows) <= res(&b.shadows));
        }
        assert!(RenderSettings::low().gi.is_none());
        assert!(RenderSettings::medium().gi.is_some());
        assert!(RenderSettings::low().reflections.is_none());
        assert!(RenderSettings::ultra().reflections.is_some());
    }
}

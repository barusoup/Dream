//! Renders the 3D feature test scene per named configuration and writes PNGs
//! to `target/scene-check/`: the composed frame (`<name>.png`), a 4x
//! nearest-neighbour crop of the edge region (`<name>_crop.png`) and the surface
//! attachment (`surface_png`, see `SURFACE_FORMAT` for the encoding).
//!
//! Run `cargo run --release --bin scene_check [-- --only <name>]`. The process
//! exits non-zero when a check fails: a composed frame that is more than 98%
//! one colour (nothing drew), the shadow darkening range, the soft-shadow
//! regression (lit pixels must match `base`), the TAA static stability bound, or
//! the TAA motion ghosting bound.
#[path = "scene_check/clouds.rs"]
mod clouds;
#[path = "scene_check/env_map.rs"]
mod env_map;
#[path = "scene_check/ray_tracing.rs"]
mod ray_tracing;
#[path = "scene_check/reference.rs"]
mod reference;
#[path = "scene_check/sky_fog.rs"]
mod sky_fog;
#[path = "scene_check/view_clouds.rs"]
mod view_clouds;

use dream::camera::Camera;
use dream::gfx::{DEPTH_FORMAT, GpuContext, Renderer, SCENE_COLOR_FORMATS, SceneFrame};
use dream::lighting::{Lighting, LightingEnv};
use dream::model::render::{ModelGpu, ModelRenderer};
use dream::model::shapes;
use dream::model::{AlphaMode, Material, Model, ModelInstance};
use dream::render::{
    AoSettings, GiSettings, IndirectQuality, ReflectionSettings, RenderSettings, ShadowFilter,
    ShadowSettings, TaaSettings, Tonemap,
};
use dream::render::{AtmosphereSettings, CloudSettings, FogSettings};
use dream::settings::{AaMode, Settings, UpscalePreset};
use glam::{Mat4, Vec3};
use reference::{
    Category, Hit, MIRROR_FLOOR, Ray, Scene, SceneItem, Shape, View, ao_visibility, linear_depth,
    mirror_reflection, scene_items,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const W: u32 = 1280;
const H: u32 = 720;
/// Frames rendered before each capture, so temporal state settles.
const FRAMES: usize = 8;
/// Frames averaged into [`Captured::mean_last`] (the last ones of a shot).
const MEAN_FRAMES: usize = 8;
/// Frames for the TAA cases, which need history to converge.
const TAA_FRAMES: usize = 24;
const OUT_DIR: &str = "target/scene-check";
/// A composed frame more than this fraction of one colour counts as "nothing drew".
const MAX_SINGLE_COLOUR: f64 = 0.98;
/// Largest allowed mean per-channel difference, in 8-bit steps, between two
/// consecutive static TAA frames.
const MAX_TAA_STATIC_DIFF: f64 = 1.5;
/// Largest allowed mean per-channel difference, in 8-bit steps, between the
/// TAA orbit frame and the no-AA render of the same pose.
const MAX_TAA_MOTION_DIFF: f64 = 12.0;
/// Largest allowed mean per-channel difference, in 8-bit steps, between the
/// lit pixels of `shadows_high` and `base`.
const MAX_SOFT_LIT_DIFF: f64 = 1.5;
/// Largest allowed consecutive-frame difference, in 8-bit steps, for the static
/// indirect-light cases (AO and GI).
const MAX_INDIRECT_STATIC_DIFF: f64 = 2.0;
/// Largest allowed difference between the moving GI case and its still reference.
const MAX_INDIRECT_MOTION_DIFF: f64 = 6.0;
/// A pixel counts as darkened by AO when it is this much darker than `taa`.
const AO_DARKER_BY: f32 = 0.03;
/// Accepted share of AO-darkened pixels against `taa`.
const AO_DARKER_RANGE: (f64, f64) = (0.01, 0.5);
/// Crop of the edge-quality region: 240x135 px, shown at 4x.
const CROP: (u32, u32, u32, u32) = (548, 293, 240, 135);
const CROP_SCALE: u32 = 4;
/// Sun for every case: azimuth, elevation, intensity. Azimuth 2.2 puts it at
/// -x and +z, so faces toward the camera (+z) are lit and shadows fall to the
/// right and away.
const SUN: (f32, f32, f32) = (2.2, 0.6, 3.0);
/// Least mean abs difference, in 8-bit steps, between the mirror floor patch
/// and `taa` for the forward shader to count as showing reflections.
const MIN_REFL_PATCH_DIFF: f64 = 5.0;
/// Reference AO grid: every this many debug-image texels.
const AO_GRID_STEP: u32 = 2;
/// Discontinuity guard: a grid point is skipped when a neighbour this many
/// debug texels away is a different surface or sky.
const EDGE_TEXELS: i64 = 2;
/// AO bounds: per-category mean signed error (GPU minus reference) and overall
/// mean absolute error, for categories with at least `AO_MIN_SAMPLES` samples.
/// Sphere lower halves are gated on the screen-visible reference (see
/// `check_ao`); the full-reference residual there is hidden geometry, reported.
const AO_MAX_SIGNED: f64 = 0.06;
const AO_MAX_MAE: f64 = 0.07;

/// Per-category signed-error bound.
fn ao_bound(category: Category) -> f64 {
    match category {
        Category::Vertical | Category::SphereUpper => 0.04,
        Category::SphereLower => 0.12,
        _ => AO_MAX_SIGNED,
    }
}
const AO_MIN_SAMPLES: u64 = 50;
/// Reflection reference grid: every this many texels of the reflection dump.
const REFL_GRID_STEP: u32 = 4;
/// A reflection texel counts as confident when its confidence exceeds this.
const REFL_HIT_CONF: f64 = 0.5;
/// Bounds: share of expected-hit texels with confidence above `REFL_HIT_CONF`,
/// mean abs RGB error (8-bit steps, display space) against the frame at the
/// reference hit, share of expected-miss texels below `REFL_MISS_CONF`.
const REFL_MIN_HIT_FRACTION: f64 = 0.8;
const REFL_MAX_COLOUR_ERROR: f64 = 20.0;
const REFL_MISS_CONF: f64 = 0.2;
const REFL_MIN_MISS_FRACTION: f64 = 0.9;
/// Depth-agreement bound: share of probe pixels whose reference hit and depth
/// buffer disagree.
const MAX_DEPTH_MISMATCH: f64 = 0.005;
/// Largest allowed consecutive-frame difference for the reflection cases.
const MAX_REFL_STATIC_DIFF: f64 = 2.5;
/// Largest allowed difference between the moving full-feature case and its still reference.
const MAX_ALL_MOTION_DIFF: f64 = 8.0;
/// Least R-G increase, on 0..255, that the one-bounce GI must add to the
/// white cube and ground in front of the red wall, against the AO case.
const MIN_GI_BLEED_DELTA: f64 = 2.0;
/// Screen box of the cube and ground in front of the red wall: centre and half extent.
const BLEED_REGION: ([f32; 3], [f32; 3]) = ([-4.4, 0.35, -1.9], [0.6, 0.35, 0.5]);
/// Reflection settings presets used by the cases.
const REFL_MEDIUM: Option<fn() -> ReflectionSettings> = Some(ReflectionSettings::medium);
const REFL_HIGH: Option<fn() -> ReflectionSettings> = Some(ReflectionSettings::high);
/// Camera for every case: eye, look-at target.
const EYE: [f32; 3] = [6.0, 3.5, 9.0];
const TARGET: [f32; 3] = [0.0, 0.7, 0.0];
/// The case every comparison is made against.
const BASE: &str = "base";
/// Orbit of the `taa_motion` case, degrees per frame about the vertical axis.
const ORBIT_DEG: f32 = 1.5;

/// Frames of the cloud cases: the panorama refreshes fully at least twice (2 * N * N + 8, N = 4).
const CLOUD_FRAMES: usize = 2 * 16 + 8;

/// Indirect-light options of a case: screen-space AO, one-bounce GI and the
/// quality both share.
#[derive(Clone, Copy)]
struct IndirectCase {
    ao: bool,
    gi: bool,
    quality: fn() -> IndirectQuality,
}

const NO_INDIRECT: IndirectCase = IndirectCase {
    ao: false,
    gi: false,
    quality: IndirectQuality::medium,
};
const AO_ONLY: IndirectCase = IndirectCase {
    ao: true,
    gi: false,
    quality: IndirectQuality::medium,
};
const AO_GI: IndirectCase = IndirectCase {
    ao: true,
    gi: true,
    quality: IndirectQuality::medium,
};
const AO_GI_HIGH: IndirectCase = IndirectCase {
    ao: true,
    gi: true,
    quality: IndirectQuality::high,
};

/// One named render configuration.
struct Case {
    name: &'static str,
    /// File name of this case's surface-attachment dump.
    surface_png: &'static str,
    /// Shadow settings for the case; `None` renders with shadows off.
    shadows: Option<fn() -> ShadowSettings>,
    aa: AaMode,
    taa: fn() -> TaaSettings,
    tonemap: Tonemap,
    frames: usize,
    /// Orbit about the target, degrees per frame (0 = static camera).
    orbit_deg: f32,
    /// Screen-space AO and GI settings.
    indirect: IndirectCase,
    /// Screen-space reflection settings; `None` renders with reflections off.
    reflections: Option<fn() -> ReflectionSettings>,
    /// Physically based sky settings; `None` renders with the sky off.
    atmosphere: Option<fn() -> AtmosphereSettings>,
    /// Volumetric fog settings; `None` renders with fog off.
    fog: Option<fn() -> FogSettings>,
    /// Cloud settings; `None` renders with clouds off.
    clouds: Option<fn() -> CloudSettings>,
    /// Scene time of the first frame, seconds.
    time_origin: f64,
    /// Camera pitch in degrees above the horizon; 0 keeps the look-at pose.
    look_up_deg: f32,
    /// Sun azimuth and elevation in radians; `None` uses `SUN`.
    sun: Option<(f32, f32)>,
    /// Scene time added per frame, in seconds (0 holds the time still).
    time_step: f64,
}

fn hard_shadows() -> ShadowSettings {
    let mut settings = ShadowSettings::medium();
    settings.filter = ShadowFilter::Hard;
    settings
}

fn debug_shadows() -> ShadowSettings {
    let mut settings = ShadowSettings::medium();
    settings.debug_cascades = true;
    settings
}

fn taa_default() -> TaaSettings {
    TaaSettings::default()
}

fn taa_sharp0() -> TaaSettings {
    let mut settings = TaaSettings::default();
    settings.sharpness = 0.0;
    settings
}

const fn medium() -> Option<fn() -> ShadowSettings> {
    Some(ShadowSettings::medium)
}

fn earth_sky() -> AtmosphereSettings {
    AtmosphereSettings::earth()
}

fn hazy_sky() -> AtmosphereSettings {
    AtmosphereSettings::hazy()
}

fn medium_fog() -> FogSettings {
    FogSettings::medium()
}

fn dense_fog() -> FogSettings {
    FogSettings::dense()
}

fn mist_fog() -> FogSettings {
    FogSettings::ground_mist()
}

/// Fog with strong sun in-scattering, so the green wall's shadow shows as a shaft.
fn shaft_fog() -> FogSettings {
    let mut settings = FogSettings::medium();
    settings.sun_scatter = 3.0;
    settings
}

/// The one-call presets, by the name of the `preset_*` case that renders each.
fn preset_for(name: &str) -> Option<fn() -> RenderSettings> {
    match name {
        "preset_low" => Some(RenderSettings::low),
        "preset_medium" => Some(RenderSettings::medium),
        "preset_high" => Some(RenderSettings::high),
        "preset_ultra" => Some(RenderSettings::ultra),
        _ => None,
    }
}

const CASES: &[Case] = &[
    Case {
        name: BASE,
        surface_png: "surface.png",
        shadows: None,
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "shadows_medium",
        surface_png: "surface_shadows_medium.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "shadows_high",
        surface_png: "surface_shadows_high.png",
        shadows: Some(ShadowSettings::high),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "shadows_cascades",
        surface_png: "surface_shadows_cascades.png",
        shadows: Some(debug_shadows),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "shadows_hard",
        surface_png: "surface_shadows_hard.png",
        shadows: Some(hard_shadows),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "taa",
        surface_png: "surface_taa.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "taa_sharp0",
        surface_png: "surface_taa_sharp0.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_sharp0,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "taa_motion",
        surface_png: "surface_taa_motion.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: ORBIT_DEG,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "fsr_native",
        surface_png: "surface_fsr_native.png",
        shadows: medium(),
        aa: AaMode::Fsr,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "tonemap_aces",
        surface_png: "surface_tonemap_aces.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "tonemap_agx",
        surface_png: "surface_tonemap_agx.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::AgX,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "tonemap_neutral",
        surface_png: "surface_tonemap_neutral.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::Neutral,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "ao",
        surface_png: "surface_ao.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_ONLY,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "ao_only_noaa",
        surface_png: "surface_ao_only_noaa.png",
        shadows: None,
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_ONLY,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "gi",
        surface_png: "surface_gi.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_GI,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "gi_high",
        surface_png: "surface_gi_high.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_GI_HIGH,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "gi_motion",
        surface_png: "surface_gi_motion.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: ORBIT_DEG,
        indirect: AO_GI,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "refl",
        surface_png: "surface_refl.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: REFL_MEDIUM,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "refl_high",
        surface_png: "surface_refl_high.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: REFL_HIGH,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "refl_noaa",
        surface_png: "surface_refl_noaa.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::None,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: REFL_MEDIUM,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "all",
        surface_png: "surface_all.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_GI,
        reflections: REFL_MEDIUM,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "all_motion",
        surface_png: "surface_all_motion.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: ORBIT_DEG,
        indirect: AO_GI,
        reflections: REFL_MEDIUM,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "sky_noon",
        surface_png: "surface_sky_noon.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "sky_sunset",
        surface_png: "surface_sky_sunset.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: Some((SUN.0, 0.06)),
        time_step: 0.0,
    },
    Case {
        name: "sky_hazy",
        surface_png: "surface_sky_hazy.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(hazy_sky),
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "fog",
        surface_png: "surface_fog.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: Some(medium_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "fog_dense",
        surface_png: "surface_fog_dense.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: Some(dense_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "fog_mist",
        surface_png: "surface_fog_mist.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: Some(mist_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "fog_shafts",
        surface_png: "surface_fog_shafts.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: Some(shaft_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: Some((-1.0, 0.25)),
        time_step: 0.0,
    },
    Case {
        name: "fog_noaa",
        surface_png: "surface_fog_noaa.png",
        shadows: medium(),
        aa: AaMode::None,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: Some(medium_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "everything",
        surface_png: "surface_everything.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_GI,
        reflections: REFL_MEDIUM,
        atmosphere: Some(earth_sky),
        fog: Some(medium_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "everything_motion",
        surface_png: "surface_everything_motion.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: ORBIT_DEG,
        indirect: AO_GI,
        reflections: REFL_MEDIUM,
        atmosphere: Some(earth_sky),
        fog: Some(medium_fog),
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 1.0 / 60.0,
    },
    Case {
        name: "clouds",
        surface_png: "surface_clouds.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "clouds_overcast",
        surface_png: "surface_clouds_overcast.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::overcast),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "clouds_wispy",
        surface_png: "surface_clouds_wispy.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::wispy),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "clouds_sunset",
        surface_png: "surface_clouds_sunset.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: Some((SUN.0, 0.08)),
        time_step: 0.0,
    },
    Case {
        name: "clouds_motion",
        surface_png: "surface_clouds_motion.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: ORBIT_DEG,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "clouds_time",
        surface_png: "surface_clouds_time.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "clouds_closeup",
        surface_png: "surface_clouds_closeup.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 35.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "everything_clouds",
        surface_png: "surface_everything_clouds.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: CLOUD_FRAMES,
        orbit_deg: 0.0,
        indirect: AO_GI,
        reflections: REFL_MEDIUM,
        atmosphere: Some(earth_sky),
        fog: Some(medium_fog),
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "preset_low",
        surface_png: "preset_low_surface.png",
        shadows: None,
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "preset_medium",
        surface_png: "preset_medium_surface.png",
        shadows: None,
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "preset_high",
        surface_png: "preset_high_surface.png",
        shadows: None,
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "preset_ultra",
        surface_png: "preset_ultra_surface.png",
        shadows: None,
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: None,
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "blend_sky",
        surface_png: "blend_sky_surface.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
    Case {
        name: "blend_crossfade",
        surface_png: "blend_crossfade_surface.png",
        shadows: medium(),
        aa: AaMode::Taa,
        taa: taa_default,
        tonemap: Tonemap::Aces,
        frames: TAA_FRAMES,
        orbit_deg: 0.0,
        indirect: NO_INDIRECT,
        reflections: None,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: None,
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    },
];

/// Pixel boxes of the reflective floor patches (the small mirror, the glossy
/// patch and the large mirror floor) and the five gold spheres, which the
/// reflection metric leaves out.
fn excluded_boxes(eye: Vec3, scene: &Scene) -> Vec<(u32, u32, u32, u32)> {
    let mut boxes: Vec<(u32, u32, u32, u32)> = scene
        .items()
        .iter()
        .filter(|item| matches!(item.shape, Shape::Plane { .. }) && item.name != "ground")
        .map(|item| plane_box(eye, item))
        .collect();
    for i in 0..5 {
        let x = i as f32 - 2.0;
        boxes.push(screen_box(eye, Vec3::new(x, 0.5, 1.0), Vec3::splat(0.5)));
    }
    boxes
}

/// Pixel box of a horizontal patch, from its projected extent.
fn plane_box(eye: Vec3, item: &SceneItem) -> (u32, u32, u32, u32) {
    let Shape::Plane { half_x, half_z } = item.shape else {
        panic!("{} is not a plane", item.name);
    };
    screen_box(
        eye,
        Vec3::from(item.centre),
        Vec3::new(half_x, 0.02, half_z),
    )
}

/// Mean abs per-channel difference over the pixels `keep(x, y)` accepts.
fn mean_abs_diff_where(a: &[u8], b: &[u8], keep: impl Fn(u32, u32) -> bool) -> f64 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for (i, (pa, pb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let (x, y) = (i as u32 % W, i as u32 / W);
        if !keep(x, y) {
            continue;
        }
        for c in 0..3 {
            sum += u64::from(pa[c].abs_diff(pb[c]));
            count += 1;
        }
    }
    sum as f64 / count.max(1) as f64
}

fn in_box(x: u32, y: u32, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> bool {
    x >= x0 && x <= x1 && y >= y0 && y <= y1
}

/// Mean of one channel of a diagnostic image over the scene-pixel box `rect`,
/// which is mapped onto the image's own size.
fn mean_channel_in(dump: &Dump, channel: usize, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> f64 {
    let (rgba, w, h) = dump;
    if *w == 0 || *h == 0 {
        return 0.0;
    }
    let (sx, sy) = (*w as f32 / W as f32, *h as f32 / H as f32);
    let mut sum = 0f64;
    let mut count = 0u64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let wx = ((x as f32 * sx) as u32).min(w - 1);
            let wy = ((y as f32 * sy) as u32).min(h - 1);
            sum += f64::from(rgba[((wy * w + wx) * 4) as usize + channel]);
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

/// The colour channels of a diagnostic image, opaque.
fn opaque(dump: &Dump) -> Dump {
    let (rgba, w, h) = dump;
    let out = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    (out, *w, *h)
}

/// The alpha channel of a diagnostic image as greyscale, opaque.
fn alpha_grey(dump: &Dump) -> Dump {
    let (rgba, w, h) = dump;
    let out = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[3], p[3], p[3], 255])
        .collect();
    (out, *w, *h)
}

/// Write an RGBA8 image of any size and print its path.
fn save_dump(path: &Path, (rgba, w, h): &Dump) {
    if rgba.is_empty() {
        eprintln!("no data for {}", path.display());
        return;
    }
    match dream::png::write(path, *w, *h, rgba) {
        Ok(()) => println!("saved {} ({w}x{h})", path.display()),
        Err(error) => eprintln!("failed to write {}: {error}", path.display()),
    }
}

/// A scene object: its GPU mesh plus the instance that carries its transform.
struct Object {
    gpu: ModelGpu,
    instance: ModelInstance,
    /// Set on the alpha-blended probes, which only their own case draws.
    probe: Option<BlendProbe>,
}

/// Alpha-blended probes for the `blend_*` cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlendProbe {
    /// A translucent slab standing across the sky and the ground.
    Quad,
    /// An opaque sphere drawn with explicit half opacity against the sky.
    Sphere,
}

const BLEND_QUAD_CENTRE: [f32; 3] = [-1.5, 3.0, 2.0];
const BLEND_QUAD_HALF: [f32; 3] = [3.0, 3.0, 0.04];
const BLEND_SPHERE_CENTRE: [f32; 3] = [-2.0, 6.0, -4.0];
const BLEND_SPHERE_RADIUS: f32 = 1.0;
const BLEND_OPACITY: f32 = 0.5;
/// Mean RGB change (0-255) the probe must make to the pixels it covers.
const BLEND_MIN_CHANGE: f64 = 12.0;
/// Fewest pixels a part (sky or ground) needs for its check to count.
const BLEND_MIN_PIXELS: usize = 200;

fn blend_probe_for(name: &str) -> Option<BlendProbe> {
    match name {
        "blend_sky" => Some(BlendProbe::Quad),
        "blend_crossfade" => Some(BlendProbe::Sphere),
        _ => None,
    }
}

fn probe_bounds(probe: BlendProbe) -> (Vec3, Vec3) {
    match probe {
        BlendProbe::Quad => (Vec3::from(BLEND_QUAD_CENTRE), Vec3::from(BLEND_QUAD_HALF)),
        BlendProbe::Sphere => (
            Vec3::from(BLEND_SPHERE_CENTRE),
            Vec3::splat(BLEND_SPHERE_RADIUS),
        ),
    }
}

fn blend_probe(ctx: &Arc<GpuContext>, probe: BlendProbe) -> Object {
    let mut object = match probe {
        BlendProbe::Quad => {
            let mat = Material {
                alpha_mode: AlphaMode::Blend,
                ..material("blend-quad", [0.9, 0.15, 0.1, 0.5], 0.0, 0.5)
            };
            object(ctx, shapes::cuboid(BLEND_QUAD_HALF, mat), BLEND_QUAD_CENTRE)
        }
        BlendProbe::Sphere => object(
            ctx,
            shapes::uv_sphere(
                BLEND_SPHERE_RADIUS,
                32,
                16,
                // This probe tests pass order and opacity, so use emission to
                // keep foreground/sky contrast independent of ambient tuning.
                Material {
                    emissive_factor: [1.0, 0.0, 0.0],
                    ..material("opacity-sphere", [0.0, 0.0, 0.0, 1.0], 0.0, 0.5)
                },
            ),
            BLEND_SPHERE_CENTRE,
        ),
    };
    object.probe = Some(probe);
    object
}

fn object(ctx: &Arc<GpuContext>, model: Model, position: [f32; 3]) -> Object {
    let model = Arc::new(model);
    let gpu = ModelGpu::upload(ctx, &model);
    let mut instance = ModelInstance::new(model);
    instance.set_transform(Mat4::from_translation(Vec3::from(position)));
    instance.update(0.0);
    Object {
        gpu,
        instance,
        probe: None,
    }
}

fn material(name: &str, base_color: [f32; 4], metallic: f32, roughness: f32) -> Material {
    Material {
        name: name.to_string(),
        base_color,
        metallic_factor: metallic,
        roughness_factor: roughness,
        ..Material::default()
    }
}

/// Camera of the mirror-fallback case, above and in front of its floor.
const MIRROR_EYE: [f32; 3] = [0.0, 2.0, 8.0];

/// The mirror-fallback floor: a metal of white base colour and near-zero
/// roughness, centred at the origin.
fn mirror_probe(ctx: &Arc<GpuContext>) -> Object {
    let floor = shapes::plane_rect(
        env_map::MIRROR_HALF,
        env_map::MIRROR_HALF,
        material("mirror probe", [1.0, 1.0, 1.0, 1.0], 1.0, 0.03),
    );
    object(ctx, floor, [0.0, 0.0, 0.0])
}

/// The mirror-fallback shot: reflections and clouds on, the atmosphere on,
/// no shadows, identity tone curve so the frame is comparable with the map.
fn mirror_shot() -> Shot {
    Shot {
        preset: None,
        blend: None,
        eye: MIRROR_EYE,
        orbit_deg: 0.0,
        frames: 12,
        aa: AaMode::None,
        taa: taa_default(),
        shadows: None,
        tonemap: Tonemap::None,
        indirect: NO_INDIRECT,
        reflections: REFL_MEDIUM,
        atmosphere: Some(earth_sky),
        fog: None,
        clouds: Some(CloudSettings::fair_weather),
        time_origin: 0.0,
        look_up_deg: 0.0,
        sun: None,
        time_step: 0.0,
    }
}

/// The fixed feature scene, built from [`scene_items`]: the same list the
/// CPU reference traces.
fn build_scene(ctx: &Arc<GpuContext>) -> Vec<Object> {
    scene_items()
        .into_iter()
        .map(|item| {
            let mat = material(item.name, item.base_color, item.metallic, item.roughness);
            let model = match item.shape {
                Shape::Plane { half_x, half_z } => shapes::plane_rect(half_x, half_z, mat),
                Shape::Box { half } => shapes::cuboid(half, mat),
                Shape::Sphere { radius } => shapes::uv_sphere(radius, 32, 16, mat),
            };
            object(ctx, model, item.centre)
        })
        .collect()
}
/// One render: camera path, lighting and post settings.
struct Shot {
    /// A one-call preset rendered instead of the per-feature fields below.
    preset: Option<fn() -> RenderSettings>,
    /// The blended probe this shot draws; `None` leaves the probes out.
    blend: Option<BlendProbe>,
    /// Camera position at frame 0; the orbit turns it about `TARGET`.
    eye: [f32; 3],
    /// Orbit per frame in degrees about the vertical axis through `TARGET`.
    orbit_deg: f32,
    frames: usize,
    aa: AaMode,
    taa: TaaSettings,
    shadows: Option<fn() -> ShadowSettings>,
    tonemap: Tonemap,
    indirect: IndirectCase,
    reflections: Option<fn() -> ReflectionSettings>,
    atmosphere: Option<fn() -> AtmosphereSettings>,
    fog: Option<fn() -> FogSettings>,
    /// Cloud settings; `None` renders with clouds off.
    clouds: Option<fn() -> CloudSettings>,
    /// Scene time of the first frame, seconds.
    time_origin: f64,
    /// Camera pitch in degrees above the horizon; 0 keeps the look-at pose.
    look_up_deg: f32,
    sun: Option<(f32, f32)>,
    time_step: f64,
}

/// An RGBA8 diagnostic image with its size, as `capture_*_rgba8` returns it.
type Dump = (Vec<u8>, u32, u32);

/// What a shot captured: the last two composed frames (for consecutive-frame
/// stability), the surface attachment of the last frame, its camera position,
/// and the indirect and reflection chain outputs of the last frame when those
/// chains ran.
#[derive(Clone)]
struct Captured {
    last: Vec<u8>,
    /// Mean of the last [`MEAN_FRAMES`] frames, one value per byte of `last`'s
    /// layout (display steps 0..=255). Checks that compare colour, not noise,
    /// read this instead of one frame.
    mean_last: Vec<f32>,
    previous: Vec<u8>,
    surface: Vec<u8>,
    eye: Vec3,
    indirect: Option<Dump>,
    reflection: Option<Dump>,
    /// The scene depth buffer of the last frame: raw little-endian f32, top
    /// row first, render resolution.
    depth: Dump,
    /// Sky-view LUT of the last frame: floats, four per texel, and its size.
    sky_view: Option<(Vec<f32>, u32, u32)>,
    /// Integrated fog volume of the last frame, and the per-froxel inject output.
    fog_volume: Option<(Vec<f32>, (u32, u32, u32))>,
    fog_injected: Option<(Vec<f32>, (u32, u32, u32))>,
    /// Post-pass GPU time of the last frame, in milliseconds.
    post_ms: f32,
    /// Cloud panorama of the last frame (RGBA float, width, height), when clouds are on.
    cloud_panorama: Option<(Vec<f32>, u32, u32)>,
    /// Sky environment map of the last frame (mip 0, RGBA float, width, height), when the atmosphere is on.
    sky_env: Option<(Vec<f32>, u32, u32)>,
}

/// Camera position for orbit step `frame` (0-based) about `TARGET`.
fn orbit_eye(eye: [f32; 3], orbit_deg: f32, frame: usize) -> Vec3 {
    let target = Vec3::from(TARGET);
    let rel = Vec3::from(eye) - target;
    let angle = (orbit_deg * frame as f32).to_radians();
    let (sin, cos) = angle.sin_cos();
    target + Vec3::new(rel.x * cos + rel.z * sin, rel.y, -rel.x * sin + rel.z * cos)
}

fn render_shot(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    objects: &mut [Object],
    shot: &Shot,
) -> Captured {
    renderer.set_aa_mode(shot.aa);
    // Scene time is set per frame from the case, so the noise is deterministic.
    renderer.set_scene_time_scale(0.0);
    let settings = match shot.preset {
        Some(make) => make(),
        None => {
            let mut settings = RenderSettings::default();
            settings.shadows = shot.shadows.map(|make| make());
            settings.taa = shot.taa.clone();
            settings.tonemap = shot.tonemap;
            settings.ao = shot.indirect.ao.then(AoSettings::default);
            settings.gi = shot.indirect.gi.then(GiSettings::default);
            settings.indirect_quality = (shot.indirect.quality)();
            settings.reflections = shot.reflections.map(|make| make());
            settings.atmosphere = shot.atmosphere.map(|make| make());
            settings.fog = shot.fog.map(|make| make());
            settings.clouds = shot.clouds.map(|make| make());
            settings
        }
    };
    renderer.set_render_settings(settings);
    renderer.reset_history();

    let (azimuth, elevation, intensity) = (
        shot.sun.map_or(SUN.0, |sun| sun.0),
        shot.sun.map_or(SUN.1, |sun| sun.1),
        SUN.2,
    );
    let mut captured = Captured {
        last: Vec::new(),
        mean_last: Vec::new(),
        previous: Vec::new(),
        surface: Vec::new(),
        eye: Vec3::from(shot.eye),
        indirect: None,
        reflection: None,
        depth: (Vec::new(), 0, 0),
        sky_view: None,
        fog_volume: None,
        fog_injected: None,
        post_ms: 0.0,
        cloud_panorama: None,
        sky_env: None,
    };
    let mut mean_sum: Vec<f32> = Vec::new();
    let mut mean_count = 0u32;
    for frame_index in 0..shot.frames {
        let eye = orbit_eye(shot.eye, shot.orbit_deg, frame_index);
        let mut camera = Camera::default();
        camera.set_aspect(W as f32 / H as f32);
        camera.position = eye;
        let forward = (Vec3::from(TARGET) - eye).normalize();
        let pitch = if shot.look_up_deg != 0.0 {
            shot.look_up_deg.to_radians()
        } else {
            forward.y.asin()
        };
        camera.set_look(forward.z.atan2(forward.x), pitch);
        let env = LightingEnv {
            cam_pos: eye.extend(0.0).to_array(),
            ..LightingEnv::with_sun(azimuth, elevation, intensity)
        };
        let frame = SceneFrame {
            camera: &camera,
            lighting,
            env,
            scene_rect: None,
            ui: None,
        };
        // Latch each instance's previous transform once per frame, as the frame
        // loop does: otherwise the motion vectors compare with the identity.
        for object in objects.iter_mut() {
            object.instance.update(0.0);
        }
        renderer.set_scene_time(shot.time_origin + frame_index as f64 * shot.time_step);
        renderer
            .render_scene(&frame, |draw| {
                for object in objects.iter_mut() {
                    match (object.probe, shot.blend) {
                        (None, _) => {
                            models.draw_pbr(draw, &mut object.gpu, &object.instance, lighting);
                        }
                        (Some(probe), Some(chosen)) if probe == chosen => match probe {
                            BlendProbe::Quad => {
                                models.draw_pbr(draw, &mut object.gpu, &object.instance, lighting);
                            }
                            BlendProbe::Sphere => models.draw_pbr_with_opacity(
                                draw,
                                &mut object.gpu,
                                &object.instance,
                                lighting,
                                BLEND_OPACITY,
                            ),
                        },
                        _ => {}
                    }
                }
                objects.len()
            })
            .expect("render scene frame");
        captured.eye = eye;
        // Read back only the last MEAN_FRAMES frames: the readback is the
        // expensive part. All of them are summed; the last two are kept.
        if frame_index + MEAN_FRAMES >= shot.frames {
            let rgba = renderer.capture_rgba8().0;
            if mean_sum.len() != rgba.len() {
                mean_sum = vec![0.0; rgba.len()];
            }
            for (sum, &v) in mean_sum.iter_mut().zip(&rgba) {
                *sum += f32::from(v);
            }
            mean_count += 1;
            if frame_index + 2 >= shot.frames {
                captured.previous = std::mem::take(&mut captured.last);
                captured.last = rgba;
            }
        }
    }
    captured.mean_last = mean_sum
        .iter()
        .map(|sum| sum / mean_count.max(1) as f32)
        .collect();
    captured.surface = renderer.capture_surface_rgba8().0;
    captured.depth = renderer.capture_depth_f32();
    if shot.atmosphere.is_some() {
        captured.sky_view = Some(renderer.capture_sky_view_rgba16f());
        captured.sky_env = Some(renderer.capture_sky_env_rgba16f());
    }
    if shot.fog.is_some() {
        captured.fog_volume = renderer.capture_fog_volume();
        captured.fog_injected = renderer.capture_fog_injected();
    }
    if shot.clouds.is_some() {
        captured.cloud_panorama = renderer.capture_cloud_panorama();
    }
    captured.post_ms = renderer.post_gpu_ms();
    if shot.indirect.ao || shot.indirect.gi {
        captured.indirect = Some(renderer.capture_indirect_rgba8());
    }
    if shot.reflections.is_some() {
        captured.reflection = Some(renderer.capture_reflection_rgba8());
    }
    captured
}

/// Fraction of pixels whose mean RGB is more than 10% below the same pixel of
/// `base`: a count of the shadow edges and interiors a case added.
fn darker_fraction(shadowed: &[u8], base: &[u8]) -> f64 {
    let mut darker = 0u64;
    let mut total = 0u64;
    for (a, b) in shadowed.chunks_exact(4).zip(base.chunks_exact(4)) {
        let lum_a = (u32::from(a[0]) + u32::from(a[1]) + u32::from(a[2])) as f32 / 3.0;
        let lum_b = (u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2])) as f32 / 3.0;
        if lum_a < 0.9 * lum_b {
            darker += 1;
        }
        total += 1;
    }
    darker as f64 / total.max(1) as f64
}

/// Per-channel mean of `shadowed` and of `base` over the pixels that are not
/// darker than `base` by more than 10%, as the difference of the two means.
/// The mask comes from `shadowed`, so the same pixels are averaged in both.
fn non_darker_mean_diff(shadowed: &[u8], base: &[u8]) -> ([f64; 3], u64) {
    let mut sum_a = [0f64; 3];
    let mut sum_b = [0f64; 3];
    let mut count = 0u64;
    for (a, b) in shadowed.chunks_exact(4).zip(base.chunks_exact(4)) {
        let lum_a = (u32::from(a[0]) + u32::from(a[1]) + u32::from(a[2])) as f32 / 3.0;
        let lum_b = (u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2])) as f32 / 3.0;
        if lum_a < 0.9 * lum_b {
            continue;
        }
        for c in 0..3 {
            sum_a[c] += f64::from(a[c]);
            sum_b[c] += f64::from(b[c]);
        }
        count += 1;
    }
    let n = count.max(1) as f64;
    (
        [
            sum_a[0] / n - sum_b[0] / n,
            sum_a[1] / n - sum_b[1] / n,
            sum_a[2] / n - sum_b[2] / n,
        ],
        count,
    )
}

/// Mean absolute per-channel difference between two RGBA8 images.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            sum += u64::from(pa[c].abs_diff(pb[c]));
            count += 1;
        }
    }
    sum as f64 / count.max(1) as f64
}

/// Mean of R, G, B and the share of pixels taken by the most common RGB value.
fn stats(rgba: &[u8]) -> ([f64; 3], f64) {
    let mut sum = [0u64; 3];
    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();
    let mut total = 0u64;
    for px in rgba.chunks_exact(4) {
        for (channel, value) in sum.iter_mut().zip(px) {
            *channel += u64::from(*value);
        }
        *counts.entry([px[0], px[1], px[2]]).or_default() += 1;
        total += 1;
    }
    let total = total.max(1) as f64;
    let mean = sum.map(|s| s as f64 / total);
    let top = counts.values().copied().max().unwrap_or(0) as f64 / total;
    (mean, top)
}

/// Mean absolute luma difference between horizontally adjacent pixels in the
/// crop: a blur and sharpness proxy (higher is crisper).
fn edge_energy(rgba: &[u8]) -> f64 {
    let (x0, y0, cw, ch) = CROP;
    let luma = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        0.2126 * f64::from(rgba[i])
            + 0.7152 * f64::from(rgba[i + 1])
            + 0.0722 * f64::from(rgba[i + 2])
    };
    let mut sum = 0.0;
    let mut count = 0u64;
    for y in y0..y0 + ch {
        for x in x0..x0 + cw - 1 {
            sum += (luma(x + 1, y) - luma(x, y)).abs();
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

/// The crop region, upscaled by `CROP_SCALE` with nearest-neighbour sampling.
fn crop_upscaled(rgba: &[u8]) -> Vec<u8> {
    let (x0, y0, cw, ch) = CROP;
    let (ow, oh) = (cw * CROP_SCALE, ch * CROP_SCALE);
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let sx = x0 + ox / CROP_SCALE;
            let sy = y0 + oy / CROP_SCALE;
            let src = ((sy * W + sx) * 4) as usize;
            let dst = ((oy * ow + ox) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
        }
    }
    out
}

/// Write one PNG and print its path and mean colour. Returns the single-colour
/// share so the caller can judge the composed frame.
fn save(path: &Path, rgba: &[u8]) -> Option<f64> {
    if let Err(error) = dream::png::write(path, W, H, rgba) {
        eprintln!("failed to write {}: {error}", path.display());
        return None;
    }
    let (mean, top) = stats(rgba);
    println!(
        "saved {} mean=({:.1}, {:.1}, {:.1})",
        path.display(),
        mean[0],
        mean[1],
        mean[2]
    );
    Some(top)
}

fn save_crop(path: &Path, rgba: &[u8]) {
    let (_, _, cw, ch) = CROP;
    let (ow, oh) = (cw * CROP_SCALE, ch * CROP_SCALE);
    let scaled = crop_upscaled(rgba);
    if let Err(error) = dream::png::write(path, ow, oh, &scaled) {
        eprintln!("failed to write {}: {error}", path.display());
    }
}

fn shot_of(case: &Case) -> Shot {
    Shot {
        preset: preset_for(case.name),
        blend: blend_probe_for(case.name),
        eye: EYE,
        orbit_deg: case.orbit_deg,
        frames: case.frames,
        aa: case.aa,
        taa: (case.taa)(),
        shadows: case.shadows,
        tonemap: case.tonemap,
        indirect: case.indirect,
        reflections: case.reflections,
        atmosphere: case.atmosphere,
        fog: case.fog,
        clouds: case.clouds,
        time_origin: case.time_origin,
        look_up_deg: case.look_up_deg,
        sun: case.sun,
        time_step: case.time_step,
    }
}

/// Camera used by the still cases: the same look-at as `render_shot`.
fn camera_at(eye: Vec3) -> Camera {
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    camera.position = eye;
    let forward = (Vec3::from(TARGET) - eye).normalize();
    camera.set_look(forward.z.atan2(forward.x), forward.y.asin());
    camera
}

/// Pixel box (x0, y0, x1, y1), top row first, of an axis-aligned box.
fn screen_box(eye: Vec3, centre: Vec3, half: Vec3) -> (u32, u32, u32, u32) {
    let vp = camera_at(eye).view_projection();
    let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for corner in 0..8u32 {
        let sign = |bit: u32| if corner & bit != 0 { 1.0 } else { -1.0 };
        let p = centre + half * Vec3::new(sign(1), sign(2), sign(4));
        let ndc = vp.project_point3(p);
        let x = (ndc.x * 0.5 + 0.5) * W as f32;
        let y = (1.0 - (ndc.y * 0.5 + 0.5)) * H as f32;
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x),
            bounds[3].max(y),
        ];
    }
    let clamp = |v: f32, max: u32| v.clamp(0.0, max as f32 - 1.0) as u32;
    (
        clamp(bounds[0], W),
        clamp(bounds[1], H),
        clamp(bounds[2], W),
        clamp(bounds[3], H),
    )
}

/// Mean of R minus G over the pixels of `rgba` inside the box.
fn mean_red_minus_green(rgba: &[u8], (x0, y0, x1, y1): (u32, u32, u32, u32)) -> f64 {
    let mut sum = 0f64;
    let mut count = 0u64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = ((y * W + x) * 4) as usize;
            sum += f64::from(rgba[i]) - f64::from(rgba[i + 1]);
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

/// Fraction of pixels whose mean RGB is more than `by` below the same pixel of
/// `base`.
fn darker_fraction_by(shadowed: &[u8], base: &[u8], by: f32) -> f64 {
    let mut darker = 0u64;
    let mut total = 0u64;
    for (a, b) in shadowed.chunks_exact(4).zip(base.chunks_exact(4)) {
        let lum_a = (u32::from(a[0]) + u32::from(a[1]) + u32::from(a[2])) as f32 / 3.0;
        let lum_b = (u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2])) as f32 / 3.0;
        if lum_a < (1.0 - by) * lum_b {
            darker += 1;
        }
        total += 1;
    }
    darker as f64 / total.max(1) as f64
}

/// The reference camera for `eye`, with the readback row order `top_first`.
fn make_view(eye: Vec3, size: (u32, u32), top_first: bool) -> View {
    let camera = camera_at(eye);
    View::new(
        camera.view_projection(),
        eye.as_dvec3(),
        size,
        camera.clip_planes(),
        top_first,
    )
}

/// Share of probe pixels (a grid of 16 texels) where the reference primary hit
/// and the depth buffer disagree: sky versus surface, a hit that does not
/// project back to its pixel, or linear depth off by more than 1% (plus 5 mm).
fn depth_mismatch(scene: &Scene, eye: Vec3, depth: &Dump, top_first: bool) -> f64 {
    let (bytes, w, h) = depth;
    let view = make_view(eye, (*w, *h), top_first);
    let (mut bad, mut total) = (0u64, 0u64);
    for y in (0..*h).step_by(16) {
        for x in (0..*w).step_by(16) {
            let (fx, fy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let i = ((y * w + x) * 4) as usize;
            let d = f64::from(f32::from_le_bytes([
                bytes[i],
                bytes[i + 1],
                bytes[i + 2],
                bytes[i + 3],
            ]));
            let sky = d >= 0.99999;
            let wrong = match scene.trace(view.pixel_ray(fx, fy), f64::INFINITY) {
                None => !sky,
                Some(_) if sky => true,
                Some(hit) => {
                    let (px, py, l_ref) = view
                        .project(hit.position)
                        .expect("reference hit is in front of the eye");
                    let projects_back = (px - fx).abs() + (py - fy).abs() < 0.05;
                    let l_gpu = linear_depth(d, view.near, view.far);
                    !projects_back || (l_ref - l_gpu).abs() > 0.01 * l_ref + 0.005
                }
            };
            bad += u64::from(wrong);
            total += 1;
        }
    }
    bad as f64 / total.max(1) as f64
}

/// Sample count, GPU and reference sums and absolute error of one category.
#[derive(Default, Clone, Copy)]
struct Acc {
    n: u64,
    gpu: f64,
    reference: f64,
    signed: f64,
    abs: f64,
}

impl Acc {
    fn add(&mut self, gpu: f64, reference: f64) {
        self.n += 1;
        self.gpu += gpu;
        self.reference += reference;
        self.signed += gpu - reference;
        self.abs += (gpu - reference).abs();
    }
}

/// The reference primary hit at texel `(x, y)` of a readback of `size`.
fn primary_at(scene: &Scene, view: &View, size: (u32, u32), x: i64, y: i64) -> Option<Hit> {
    if x < 0 || y < 0 || x >= i64::from(size.0) || y >= i64::from(size.1) {
        return None;
    }
    let (fx, fy) = view.texel_to_pixel(x as u32, y as u32, size);
    scene.trace(view.pixel_ray(fx, fy), f64::INFINITY)
}

/// True when the reference surface at `(x, y)` continues to its neighbours
/// `EDGE_TEXELS` away: same object, and no sky beside it.
fn interior(scene: &Scene, view: &View, size: (u32, u32), centre: &Hit, x: i64, y: i64) -> bool {
    [(-1, 0), (1, 0), (0, -1), (0, 1)]
        .into_iter()
        .all(|(dx, dy)| {
            let (nx, ny) = (x + dx * EDGE_TEXELS, y + dy * EDGE_TEXELS);
            primary_at(scene, view, size, nx, ny)
                .is_some_and(|neighbour| neighbour.object_index == centre.object_index)
        })
}

/// Compares the GPU ambient visibility (alpha of the indirect dump) with the
/// reference on a grid of working texels, per surface category, and writes the
/// reference as `ao_reference.png`. Returns false when a bound is exceeded.
fn check_ao(scene: &Scene, view: &View, ao: &Dump, out: &Path) -> bool {
    let (rgba, w, h) = ao;
    let size = (*w, *h);
    let radius = f64::from(AoSettings::default().radius);
    let mut cats = [Acc::default(); 6];
    let mut seen_cats = [Acc::default(); 6];
    let mut seen_all = Acc::default();
    let mut all = Acc::default();
    let mut reference_img = vec![0u8; (*w * *h * 4) as usize];
    for px in reference_img.chunks_exact_mut(4) {
        px[3] = 255;
    }
    for y in (0..*h).step_by(AO_GRID_STEP as usize) {
        for x in (0..*w).step_by(AO_GRID_STEP as usize) {
            let (xi, yi) = (i64::from(x), i64::from(y));
            let Some(centre) = primary_at(scene, view, size, xi, yi) else {
                continue;
            };
            if !interior(scene, view, size, &centre, xi, yi) {
                continue;
            }
            let reference = ao_visibility(scene, centre.position, centre.normal, radius, None);
            let reference_seen = ao_visibility(
                scene,
                centre.position,
                centre.normal,
                radius,
                Some(view.eye),
            );
            let gpu = f64::from(rgba[((y * w + x) * 4) as usize + 3]) / 255.0;
            cats[scene.category(&centre) as usize].add(gpu, reference);
            seen_cats[scene.category(&centre) as usize].add(gpu, reference_seen);
            seen_all.add(gpu, reference_seen);
            all.add(gpu, reference);
            let grey = (reference * 255.0 + 0.5) as u8;
            for by in 0..AO_GRID_STEP.min(*h - y) {
                for bx in 0..AO_GRID_STEP.min(*w - x) {
                    let j = (((y + by) * w + x + bx) * 4) as usize;
                    reference_img[j..j + 3].fill(grey);
                }
            }
        }
    }
    save_dump(out, &(reference_img, *w, *h));
    println!(
        "ao reference ({} samples, every {AO_GRID_STEP} texels):",
        all.n
    );
    println!("  category         n    gpu    ref  signed    mae   seen-ref  signed    mae");
    let mut ok = true;
    for category in Category::ALL {
        let acc = cats[category as usize];
        if acc.n == 0 {
            continue;
        }
        let n = acc.n as f64;
        let seen = seen_cats[category as usize];
        println!(
            "  {:<14} {:>5}  {:.3}  {:.3}  {:+.3}  {:.3}   {:.3}  {:+.3}  {:.3}",
            category.label(),
            acc.n,
            acc.gpu / n,
            acc.reference / n,
            acc.signed / n,
            acc.abs / n,
            seen.reference / n,
            seen.signed / n,
            seen.abs / n
        );
        // Sphere lower halves: the full reference counts the ground under the
        // sphere, which no screen-space method sees. Gate on the visible reference.
        let (gated_signed, gated_n) = if category == Category::SphereLower {
            (seen.signed / n, seen.n)
        } else {
            (acc.signed / n, acc.n)
        };
        if gated_n >= AO_MIN_SAMPLES && gated_signed.abs() > ao_bound(category) {
            eprintln!(
                "ao: {} mean signed error {:+.3} exceeds {}",
                category.label(),
                gated_signed,
                ao_bound(category)
            );
            ok = false;
        }
    }
    let mae = all.abs / all.n.max(1) as f64;
    println!(
        "  all              {:>5}  mae {mae:.3}   seen-ref mae {:.3}",
        all.n,
        seen_all.abs / seen_all.n.max(1) as f64
    );
    if mae >= AO_MAX_MAE {
        eprintln!("ao: overall mean absolute error {mae:.3} is not below {AO_MAX_MAE}");
        ok = false;
    }
    ok
}

/// Display sRGB byte to linear light.
fn srgb_linear(byte: u8) -> f64 {
    let c = f64::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Undoes the readback's Reinhard curve (`v / (1 + v)`) on one 8-bit channel.
fn reinhard_inverse(byte: u8) -> f64 {
    let c = (f64::from(byte) / 255.0).min(0.999);
    c / (1.0 - c)
}

/// Linear light to 8-bit sRGB steps.
fn srgb_steps(linear: f64) -> f64 {
    let v = linear.clamp(0.0, 1.0);
    let encoded = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    encoded * 255.0
}

/// Checks the reflection output over the large mirror floor against the
/// reference mirror ray. Expected hits (visible on screen) must have confidence
/// above `REFL_HIT_CONF` and a colour near the frame at the hit; expected misses
/// (sky, off-screen or hidden hits) must have confidence below `REFL_MISS_CONF`.
/// Writes `refl_reference.png`. Returns false when a bound is exceeded.
fn check_reflections(scene: &Scene, view: &View, refl: &Dump, frame: &[u8], out: &Path) -> bool {
    let (rgba, w, h) = refl;
    let size = (*w, *h);
    let mirror = scene.index_of(MIRROR_FLOOR);
    let mut reference_img = vec![40u8; (*w * *h * 4) as usize];
    for px in reference_img.chunks_exact_mut(4) {
        px[3] = 255;
    }
    let (mut hits, mut hits_confident, mut colour_err) = (0u64, 0u64, 0f64);
    let (mut misses, mut misses_low) = (0u64, 0u64);
    // End-to-end composition: the patch pixel's frame colour against the frame
    // colour at the reflected hit, times the mirror's F0, in linear light.
    let f0 = f64::from(scene.item(mirror).base_color[0]);
    let mut composition_err = 0f64;
    let mut composition_n = 0u64;
    let mut composition_signed = 0f64;
    for y in (0..*h).step_by(REFL_GRID_STEP as usize) {
        for x in (0..*w).step_by(REFL_GRID_STEP as usize) {
            let Some(primary) = primary_at(scene, view, size, i64::from(x), i64::from(y)) else {
                continue;
            };
            if primary.object_index != mirror {
                continue;
            }
            let (fx, fy) = view.texel_to_pixel(x, y, size);
            let ray: Ray = view.pixel_ray(fx, fy);
            let i = ((y * w + x) * 4) as usize;
            let confidence = f64::from(rgba[i + 3]) / 255.0;
            let target: Option<Hit> = mirror_reflection(scene, ray, &primary).filter(|target| {
                view.project(target.position).is_some_and(|(px, py, _)| {
                    view.on_screen(px, py) && scene.visible_from(view.eye, target.position)
                })
            });
            let colour = match target {
                Some(target) => {
                    let (px, py, _) = view.project(target.position).expect("projected");
                    let fi = (py as usize * view.width as usize + px as usize) * 4;
                    hits += 1;
                    if confidence > REFL_HIT_CONF {
                        hits_confident += 1;
                    }
                    // Both sides in display space: the readback's Reinhard curve
                    // is undone and the radiance sRGB-encoded, as the frame is.
                    let err: f64 = (0..3)
                        .map(|c| {
                            let gpu = srgb_steps(reinhard_inverse(rgba[i + c]));
                            (gpu - f64::from(frame[fi + c])).abs()
                        })
                        .sum();
                    colour_err += err / 3.0;
                    let patch = ((fy as usize) * view.width as usize + fx as usize) * 4;
                    let diffs: Vec<f64> = (0..3)
                        .map(|c| {
                            let hit_lin = srgb_linear(frame[fi + c]) * f0;
                            srgb_steps(hit_lin) - f64::from(frame[patch + c])
                        })
                        .collect();
                    composition_err += diffs.iter().map(|d| d.abs()).sum::<f64>() / 3.0;
                    composition_signed += diffs.iter().sum::<f64>() / 3.0;
                    composition_n += 1;
                    [0u8, 255, 0]
                }
                None => {
                    misses += 1;
                    if confidence < REFL_MISS_CONF {
                        misses_low += 1;
                    }
                    [0u8, 0, 0]
                }
            };
            for by in 0..REFL_GRID_STEP.min(*h - y) {
                for bx in 0..REFL_GRID_STEP.min(*w - x) {
                    let j = (((y + by) * w + x + bx) * 4) as usize;
                    reference_img[j..j + 3].copy_from_slice(&colour);
                }
            }
        }
    }
    save_dump(out, &(reference_img, *w, *h));
    let hit_fraction = hits_confident as f64 / hits.max(1) as f64;
    let mean_colour = colour_err / hits.max(1) as f64;
    let miss_fraction = misses_low as f64 / misses.max(1) as f64;
    println!(
        "refl reference: expected hits {hits}: confidence > {REFL_HIT_CONF} in {:.1}%, mean abs RGB error {mean_colour:.2}/255 vs frame at hit",
        hit_fraction * 100.0
    );
    println!(
        "refl reference: expected misses {misses}: confidence < {REFL_MISS_CONF} in {:.1}%",
        miss_fraction * 100.0
    );
    let composition = composition_err / composition_n.max(1) as f64;
    println!(
        "refl composition: patch pixel vs hit pixel x F0 {f0:.2} over {composition_n} expected hits: mean abs error {composition:.2}/255, signed (patch minus predicted) {:+.2}",
        composition_signed / composition_n.max(1) as f64
    );
    let mut ok = true;
    if composition_n == 0 {
        eprintln!("refl: no expected hits to check the composition on");
        ok = false;
    }
    if hits == 0 || hit_fraction < REFL_MIN_HIT_FRACTION {
        eprintln!("refl: confident hit fraction {hit_fraction:.3} below {REFL_MIN_HIT_FRACTION}");
        ok = false;
    }
    if hits > 0 && mean_colour >= REFL_MAX_COLOUR_ERROR {
        eprintln!("refl: mean colour error {mean_colour:.2} not below {REFL_MAX_COLOUR_ERROR}");
        ok = false;
    }
    if misses == 0 || miss_fraction < REFL_MIN_MISS_FRACTION {
        eprintln!(
            "refl: low-confidence miss fraction {miss_fraction:.3} below {REFL_MIN_MISS_FRACTION}"
        );
        ok = false;
    }
    ok
}

/// Visibility of a blended probe: the change in mean RGB (0-255, averaged over
/// the channels) that the probe makes to the pixels of its projected box,
/// measured separately against the sky (baseline depth at the far plane) and
/// against everything else. Returns `false` when a checked part fails.
fn blend_visibility(
    name: &str,
    probe: BlendProbe,
    with: &Captured,
    without: &Captured,
    top_first: bool,
) -> bool {
    let (centre, half) = probe_bounds(probe);
    let (x0, y0, x1, y1) = screen_box(with.eye, centre, half);
    let width = W as usize;
    let mut sums = [[0f64; 3]; 2];
    let mut counts = [0usize; 2];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let row = if top_first { y } else { H - 1 - y } as usize;
            let depth_at = (row * width + x as usize) * 4;
            let depth_bytes = &without.depth.0[depth_at..depth_at + 4];
            let depth = f32::from_le_bytes(depth_bytes.try_into().expect("four bytes"));
            let part = usize::from(depth < 1.0);
            let texel = (y as usize * width + x as usize) * 4;
            for (sum, (a, b)) in sums[part].iter_mut().zip(
                with.last[texel..texel + 3]
                    .iter()
                    .zip(&without.last[texel..texel + 3]),
            ) {
                *sum += f64::from(*a) - f64::from(*b);
            }
            counts[part] += 1;
        }
    }
    let mut ok = true;
    for (part, label) in [(0usize, "sky"), (1usize, "ground")] {
        let n = counts[part];
        let change = if n == 0 {
            0.0
        } else {
            sums[part].iter().map(|s| (s / n as f64).abs()).sum::<f64>() / 3.0
        };
        // The sphere is only required to show against the sky.
        let required = match probe {
            BlendProbe::Quad => true,
            BlendProbe::Sphere => part == 0,
        };
        let pass = change > BLEND_MIN_CHANGE && n >= BLEND_MIN_PIXELS;
        println!(
            "{name}: {label} pixels {n}, mean RGB change {change:.2}/255 (min {BLEND_MIN_CHANGE}){}",
            if required { "" } else { " (not required)" }
        );
        if required && !pass {
            eprintln!("{name}: {label} part: probe not visible (change {change:.2})");
            ok = false;
        }
    }
    ok
}

fn run(event_loop: &ActiveEventLoop, only: Option<&str>) -> bool {
    std::fs::create_dir_all(OUT_DIR).expect("create scene-check output dir");
    env_map::print_ambient();
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Dream Scene Check")
                    .with_visible(false)
                    .with_inner_size(PhysicalSize::new(W, H)),
            )
            .expect("create hidden window"),
    );
    let mut renderer = Renderer::new(
        Arc::clone(&window),
        dream::DEFAULT_SCENE_WGSL,
        Settings {
            aa: AaMode::None,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
        },
    );
    renderer.set_frame_capture(true);
    if only == Some("rt") {
        return ray_tracing::run(&mut renderer, true);
    }
    let ctx = renderer.context().clone();
    let lighting = Lighting::new(&ctx);
    let mut models = ModelRenderer::new(&ctx, &SCENE_COLOR_FORMATS, DEPTH_FORMAT);
    if only == Some("view_clouds") {
        return view_clouds::run(&mut renderer, &mut models, &lighting, &ctx, &window);
    }
    let mut objects = build_scene(&ctx);
    objects.push(blend_probe(&ctx, BlendProbe::Quad));
    objects.push(blend_probe(&ctx, BlendProbe::Sphere));
    let scene = Scene::new();
    // Row order of the readbacks, decided by the depth-agreement probe below.
    let mut top_first = true;
    let mut passed = true;
    let dir = Path::new(OUT_DIR);
    let mut base_composed: Option<Vec<u8>> = None;
    let mut taa_composed: Option<Vec<u8>> = None;
    let mut ao_composed: Option<Vec<u8>> = None;
    let mut kept: HashMap<&'static str, Captured> = HashMap::new();
    for case in CASES
        .iter()
        .filter(|case| only.is_none_or(|name| name == case.name || case.name == BASE))
    {
        let shot = shot_of(case);
        let captured = render_shot(&mut renderer, &mut models, &lighting, &mut objects, &shot);
        if let Some(probe) = shot.blend {
            let without = render_shot(
                &mut renderer,
                &mut models,
                &lighting,
                &mut objects,
                &Shot {
                    blend: None,
                    ..shot_of(case)
                },
            );
            passed &= blend_visibility(case.name, probe, &captured, &without, top_first);
        }
        let sun_angles = (
            shot.sun.map_or(SUN.0, |sun| sun.0),
            shot.sun.map_or(SUN.1, |sun| sun.1),
        );
        passed &= sky_fog::check_case(case.name, &captured, &kept, Path::new(OUT_DIR), sun_angles);
        let sun_dir = Vec3::from_slice(
            &LightingEnv::with_sun(sun_angles.0, sun_angles.1, SUN.2).sun_direction[..3],
        )
        .normalize_or_zero();
        passed &= clouds::check_case(case.name, &captured, &kept, sun_dir);
        if case.name == "sky_noon" {
            let sun_colour = Vec3::from_slice(
                &LightingEnv::with_sun(sun_angles.0, sun_angles.1, SUN.2).sun_color[..3],
            );
            passed &= env_map::check_sky_env(case.name, &captured, sun_dir, sun_colour);
        }
        println!("{}: post_gpu_ms={:.3}", case.name, captured.post_ms);
        if sky_fog::KEPT.contains(&case.name) || clouds::KEPT.contains(&case.name) {
            kept.insert(case.name, captured.clone());
        }
        let composed = &captured.last;
        if case.name == BASE {
            // Pin the conventions first: the reference's primary hits must agree
            // with the depth buffer under one of the two row orders.
            let rows = depth_mismatch(&scene, captured.eye, &captured.depth, true);
            let cols = depth_mismatch(&scene, captured.eye, &captured.depth, false);
            println!(
                "depth agreement: top-first rows {:.3}% mismatch, bottom-first {:.3}%",
                rows * 100.0,
                cols * 100.0
            );
            top_first = rows <= cols;
            let best = rows.min(cols);
            if best > MAX_DEPTH_MISMATCH {
                eprintln!(
                    "depth agreement: best mismatch {:.3}% above {:.1}%",
                    best * 100.0,
                    MAX_DEPTH_MISMATCH * 100.0
                );
                passed = false;
            }
        }
        match save(&dir.join(format!("{}.png", case.name)), composed) {
            Some(top) if top > MAX_SINGLE_COLOUR => {
                eprintln!(
                    "case {}: {:.2}% one colour, nothing drew",
                    case.name,
                    top * 100.0
                );
                passed = false;
            }
            Some(_) => {}
            None => passed = false,
        }
        save_crop(&dir.join(format!("{}_crop.png", case.name)), composed);
        println!("{}: edge_energy={:.3}", case.name, edge_energy(composed));
        println!(
            "{}: gpu shadow {:.3} ms, scene {:.3} ms, post {:.3} ms",
            case.name,
            renderer.shadow_gpu_ms(),
            renderer.scene_gpu_ms(),
            renderer.post_gpu_ms()
        );
        if let Some(dump) = captured.indirect.as_ref() {
            save_dump(&dir.join(format!("{}_gi.png", case.name)), &opaque(dump));
            save_dump(
                &dir.join(format!("{}_ao.png", case.name)),
                &alpha_grey(dump),
            );
        }
        if let Some(dump) = captured.reflection.as_ref() {
            save_dump(&dir.join(format!("{}_refl.png", case.name)), &opaque(dump));
            save_dump(
                &dir.join(format!("{}_reflconf.png", case.name)),
                &alpha_grey(dump),
            );
        }
        if save(&dir.join(case.surface_png), &captured.surface).is_none() {
            passed = false;
        }

        if case.name == BASE {
            base_composed = Some(composed.clone());
        }
        match case.name {
            "taa" => taa_composed = Some(composed.clone()),
            "ao" => ao_composed = Some(composed.clone()),
            _ => {}
        }
        // Consecutive-frame stability: the static shadow case without AA, the
        // static TAA cases, and the static indirect-light cases.
        if matches!(
            case.name,
            "shadows_medium" | "taa" | "taa_sharp0" | "fsr_native" | "ao" | "ao_only_noaa" | "gi"
        ) {
            let diff = mean_abs_diff(&captured.last, &captured.previous);
            println!("{}: consecutive-frame diff={diff:.3}/255", case.name);
            if case.name == "taa" && diff > MAX_TAA_STATIC_DIFF {
                eprintln!("case taa: static-frame diff {diff:.3} above {MAX_TAA_STATIC_DIFF}");
                passed = false;
            }
            let indirect_static = matches!(case.name, "ao" | "ao_only_noaa" | "gi");
            if indirect_static && diff > MAX_INDIRECT_STATIC_DIFF {
                eprintln!(
                    "case {}: static-frame diff {diff:.3} above {MAX_INDIRECT_STATIC_DIFF}",
                    case.name
                );
                passed = false;
            }
        }
        if case.name == "ao"
            && let Some(taa) = taa_composed.as_deref()
        {
            let fraction = darker_fraction_by(composed, taa, AO_DARKER_BY);
            let mean = |rgba: &[u8]| stats(rgba).0.iter().sum::<f64>() / 3.0;
            let ratio = mean(composed) / mean(taa).max(1e-6);
            println!(
                "ao: darker than taa by >{:.0}% in {:.2}% of pixels; mean brightness ratio to taa {ratio:.4}",
                AO_DARKER_BY * 100.0,
                fraction * 100.0
            );
            if !(AO_DARKER_RANGE.0..=AO_DARKER_RANGE.1).contains(&fraction) {
                eprintln!(
                    "case ao: darkened fraction {:.2}% outside {:.0}%..{:.0}%",
                    fraction * 100.0,
                    AO_DARKER_RANGE.0 * 100.0,
                    AO_DARKER_RANGE.1 * 100.0
                );
                passed = false;
            }
        }
        if case.name == "gi"
            && let Some(ao) = ao_composed.as_deref()
        {
            // The white cube and the ground in front of the red wall, from the
            // still camera.
            let (centre, half) = BLEED_REGION;
            let boxed = screen_box(Vec3::from(EYE), Vec3::from(centre), Vec3::from(half));
            let rg_ao = mean_red_minus_green(ao, boxed);
            let rg_gi = mean_red_minus_green(composed, boxed);
            let delta = rg_gi - rg_ao;
            println!(
                "gi: red-green on bleed region {boxed:?}: ao={rg_ao:.3} gi={rg_gi:.3} delta={delta:+.3}"
            );
            if let Some(dump) = captured.indirect.as_ref() {
                println!(
                    "gi: mean of the gi debug image in the region: ({:.2}, {:.2}, {:.2})",
                    mean_channel_in(dump, 0, boxed),
                    mean_channel_in(dump, 1, boxed),
                    mean_channel_in(dump, 2, boxed)
                );
            }
            if delta < MIN_GI_BLEED_DELTA {
                eprintln!(
                    "case gi: red bleeding raised R-G by {delta:+.3} in the bleed region, below {MIN_GI_BLEED_DELTA}"
                );
                passed = false;
            }
        }

        if case.name == "ao_only_noaa"
            && let Some(ao) = captured.indirect.as_ref()
        {
            let view = make_view(captured.eye, (W, H), top_first);
            if !check_ao(&scene, &view, ao, &dir.join("ao_reference.png")) {
                passed = false;
            }
        }
        if matches!(case.name, "refl_noaa" | "refl")
            && let Some(refl) = captured.reflection.as_ref()
        {
            let view = make_view(captured.eye, (W, H), top_first);
            if !check_reflections(
                &scene,
                &view,
                refl,
                composed,
                &dir.join("refl_reference.png"),
            ) {
                passed = false;
            }
        }
        if case.name == "refl" {
            let mirror = plane_box(Vec3::from(EYE), scene.item(scene.index_of(MIRROR_FLOOR)));
            if let Some(taa) = taa_composed.as_deref() {
                let inside = mean_abs_diff_where(composed, taa, |x, y| in_box(x, y, mirror));
                let excluded = excluded_boxes(Vec3::from(EYE), &scene);
                let outside = mean_abs_diff_where(composed, taa, |x, y| {
                    !excluded.iter().any(|b| in_box(x, y, *b))
                });
                println!(
                    "refl: mirror patch {mirror:?} mean abs diff vs taa {inside:.3}/255; \
                     outside patches and metal spheres {outside:.3}/255"
                );
                if inside <= MIN_REFL_PATCH_DIFF {
                    eprintln!(
                        "case refl: mirror patch differs from taa by {inside:.3}, not above {MIN_REFL_PATCH_DIFF}"
                    );
                    passed = false;
                }
            } else {
                println!(
                    "refl: no taa reference in this run; run without --only for the diff metrics"
                );
            }
            if let Some(dump) = captured.reflection.as_ref() {
                println!(
                    "refl: mean confidence inside the mirror patch {:.3}",
                    mean_channel_in(dump, 3, mirror) / 255.0
                );
            }
        }
        if matches!(case.name, "refl" | "all") {
            let diff = mean_abs_diff(&captured.last, &captured.previous);
            println!("{}: consecutive-frame diff={diff:.3}/255", case.name);
            if diff > MAX_REFL_STATIC_DIFF {
                eprintln!(
                    "case {}: consecutive-frame diff {diff:.3} above {MAX_REFL_STATIC_DIFF}",
                    case.name
                );
                passed = false;
            }
        }
        if matches!(case.name, "all_motion" | "everything_motion") {
            // Reference: the same final pose held still with the same settings.
            let reference = Shot {
                preset: None,
                blend: None,
                eye: captured.eye.to_array(),
                orbit_deg: 0.0,
                frames: TAA_FRAMES,
                aa: AaMode::Taa,
                taa: TaaSettings::default(),
                shadows: shot.shadows,
                tonemap: shot.tonemap,
                indirect: shot.indirect,
                reflections: shot.reflections,
                atmosphere: shot.atmosphere,
                fog: shot.fog,
                clouds: shot.clouds,
                time_origin: 0.0,
                look_up_deg: 0.0,
                sun: shot.sun,
                time_step: shot.time_step,
            };
            let ref_captured = render_shot(
                &mut renderer,
                &mut models,
                &lighting,
                &mut objects,
                &reference,
            );
            save(
                &dir.join(format!("{}_ref.png", case.name)),
                &ref_captured.last,
            );
            let diff = mean_abs_diff(composed, &ref_captured.last);
            println!("{}: diff vs {}_ref={diff:.3}/255", case.name, case.name);
            if diff > MAX_ALL_MOTION_DIFF {
                eprintln!(
                    "case {}: diff {diff:.3} above {MAX_ALL_MOTION_DIFF} (reprojection)",
                    case.name
                );
                passed = false;
            }
        }
        if case.name == "clouds_motion" {
            // Reference: the same final pose held still with the same settings.
            let reference = Shot {
                preset: None,
                blend: None,
                eye: captured.eye.to_array(),
                orbit_deg: 0.0,
                frames: CLOUD_FRAMES,
                aa: AaMode::Taa,
                taa: shot.taa.clone(),
                shadows: shot.shadows,
                tonemap: shot.tonemap,
                indirect: shot.indirect,
                reflections: shot.reflections,
                atmosphere: shot.atmosphere,
                fog: shot.fog,
                clouds: shot.clouds,
                time_origin: 0.0,
                look_up_deg: 0.0,
                sun: shot.sun,
                time_step: shot.time_step,
            };
            let ref_captured = render_shot(
                &mut renderer,
                &mut models,
                &lighting,
                &mut objects,
                &reference,
            );
            save(&dir.join("clouds_motion_ref.png"), &ref_captured.last);
            let diff = mean_abs_diff(composed, &ref_captured.last);
            println!("clouds_motion: diff vs clouds_motion_ref={diff:.3}/255");
            if diff > clouds::MAX_MOTION_DIFF {
                eprintln!(
                    "case clouds_motion: diff {diff:.3} above {} (reprojection)",
                    clouds::MAX_MOTION_DIFF
                );
                passed = false;
            }
        }
        if case.name == "clouds_time" {
            // The same still camera, 30 s of scene time later.
            let mut later = shot_of(case);
            later.time_origin = 30.0;
            let later_captured =
                render_shot(&mut renderer, &mut models, &lighting, &mut objects, &later);
            passed &= clouds::check_time(&captured, &later_captured);
        }
        if case.name == "gi_motion" {
            // Reference: the same final pose held still with the same settings.
            let reference = Shot {
                preset: None,
                blend: None,
                eye: captured.eye.to_array(),
                orbit_deg: 0.0,
                frames: TAA_FRAMES,
                aa: AaMode::Taa,
                taa: TaaSettings::default(),
                shadows: shot.shadows,
                tonemap: Tonemap::None,
                indirect: shot.indirect,
                reflections: shot.reflections,
                atmosphere: shot.atmosphere,
                fog: shot.fog,
                clouds: shot.clouds,
                time_origin: 0.0,
                look_up_deg: 0.0,
                sun: shot.sun,
                time_step: shot.time_step,
            };
            let ref_captured = render_shot(
                &mut renderer,
                &mut models,
                &lighting,
                &mut objects,
                &reference,
            );
            save(&dir.join("gi_motion_ref.png"), &ref_captured.last);
            let diff = mean_abs_diff(composed, &ref_captured.last);
            println!("gi_motion: diff vs gi_motion_ref={diff:.3}/255");
            if diff > MAX_INDIRECT_MOTION_DIFF {
                eprintln!(
                    "case gi_motion: diff {diff:.3} above {MAX_INDIRECT_MOTION_DIFF} (reprojection)"
                );
                passed = false;
            }
        }
        if case.name == "taa_motion" {
            // Reference: the same final pose, rendered without anti-aliasing.
            let reference = Shot {
                preset: None,
                blend: None,
                eye: captured.eye.to_array(),
                orbit_deg: 0.0,
                frames: FRAMES,
                aa: AaMode::None,
                taa: TaaSettings::default(),
                shadows: shot.shadows,
                tonemap: Tonemap::None,
                indirect: NO_INDIRECT,
                reflections: None,
                atmosphere: None,
                fog: None,
                clouds: None,
                time_origin: 0.0,
                look_up_deg: 0.0,
                sun: None,
                time_step: 0.0,
            };
            let ref_captured = render_shot(
                &mut renderer,
                &mut models,
                &lighting,
                &mut objects,
                &reference,
            );
            save(&dir.join("motion_ref.png"), &ref_captured.last);
            let diff = mean_abs_diff(composed, &ref_captured.last);
            println!("taa_motion: diff vs motion_ref={diff:.3}/255");
            if diff > MAX_TAA_MOTION_DIFF {
                eprintln!("case taa_motion: diff {diff:.3} above {MAX_TAA_MOTION_DIFF} (ghosting)");
                passed = false;
            }
        }
        if case.shadows.is_some()
            && case.name != BASE
            && let Some(base) = base_composed.as_deref()
        {
            let fraction = darker_fraction(composed, base);
            println!(
                "{}: darker than {BASE} by >10% in {:.2}% of pixels",
                case.name,
                fraction * 100.0
            );
            if case.name == "shadows_medium" && !(0.0025..=0.6).contains(&fraction) {
                eprintln!(
                    "case {}: darker fraction {:.2}% outside 0.25%..60%",
                    case.name,
                    fraction * 100.0
                );
                passed = false;
            }
            if case.name == "shadows_high" {
                let (diff, count) = non_darker_mean_diff(composed, base);
                println!(
                    "shadows_high: non-darker pixels ({count}) mean diff vs {BASE} = ({:+.2}, {:+.2}, {:+.2})",
                    diff[0], diff[1], diff[2]
                );
                if diff.iter().any(|d| d.abs() > MAX_SOFT_LIT_DIFF) {
                    eprintln!(
                        "case shadows_high: lit pixels differ from {BASE} by more than {MAX_SOFT_LIT_DIFF}/255"
                    );
                    passed = false;
                }
            }
        }
    }
    if only.is_some() {
        return passed;
    }
    // Mirror fallback: a white metal floor, the only object, so its reflections
    // see only the sky. Reflections and clouds are on; the frame's floor pixels
    // should match the sky environment map sampled along their mirror directions.
    let mut mirror = vec![mirror_probe(&ctx)];
    let captured = render_shot(
        &mut renderer,
        &mut models,
        &lighting,
        &mut mirror,
        &mirror_shot(),
    );
    save(&dir.join("mirror_fallback.png"), &captured.last);
    passed &= env_map::check_mirror(&captured, top_first);
    passed &= ray_tracing::run(&mut renderer, top_first);
    // Per-pixel view clouds: stability, fly-through, depth, shadows and timing.
    passed &= view_clouds::run(&mut renderer, &mut models, &lighting, &ctx, &window);
    passed
}

struct App {
    only: Option<String>,
    finished: bool,
    passed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.passed = run(event_loop, self.only.as_deref());
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: winit::event::WindowEvent) {}
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<&str> = CASES
        .iter()
        .map(|case| case.name)
        .chain(["rt", "view_clouds"])
        .collect();
    let only = match args.iter().position(|arg| arg == "--only") {
        Some(index) => match args.get(index + 1) {
            Some(name) if names.contains(&name.as_str()) => Some(name.clone()),
            Some(name) => {
                eprintln!("unknown case {name}; cases: {}", names.join(", "));
                std::process::exit(2);
            }
            None => {
                eprintln!("--only needs a case name; cases: {}", names.join(", "));
                std::process::exit(2);
            }
        },
        None => None,
    };
    let mut app = App {
        only,
        finished: false,
        passed: false,
    };
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut app)
        .expect("scene check");
    if !app.passed {
        std::process::exit(1);
    }
}

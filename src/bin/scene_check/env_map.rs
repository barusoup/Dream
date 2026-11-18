//! Sky environment map checks of scene_check: the GPU map against the CPU
//! atmosphere, the mirror fallback (a reflection that misses the scene shows the
//! map), and the driven ambient values across sun elevations.

use super::reference::{Ray, View};
use super::{Captured, H, W, make_view, srgb_steps};
use dream::lighting::LightingEnv;
use dream::render::{
    AtmosphereModel, AtmosphereSettings, SKY_ENV_SIZE, sky_env_direction, sky_env_uv,
};
use glam::{DVec3, Vec3};

/// Mean relative error (per texel, summed channels) of the GPU map against the
/// CPU atmosphere over the sampled upper-hemisphere texels, with clouds off. The
/// sky-view LUT is coarse near the horizon, so a few percent is expected.
const MAX_ENV_CPU_ERROR: f64 = 0.10;
/// Mean absolute error, in 8-bit display steps, of mirror pixels against the map
/// sampled along their reflection direction, measured on the average of the
/// last eight frames: the criterion is colour agreement, not per-frame noise.
// The remaining difference is spatial, not a colour bias (signed channel means
// are within 2/255): the mirror reflects the 1024 x 512 environment panorama's
// clouds, sampled bilinearly, against a per-texel prediction.
pub const MIRROR_MAX_MAE: f64 = 14.0;
/// Half extent of the mirror floor, metres.
pub const MIRROR_HALF: f32 = 30.0;
/// Pixel stride of the mirror sample grid.
const MIRROR_STEP: u32 = 4;
/// Upper-hemisphere sample grid of the CPU comparison: columns x rows (200 texels).
const GRID: (u32, u32) = (20, 10);
/// Sun elevations, degrees, of the ambient printout and test.
pub const AMBIENT_ELEVATIONS: [f32; 5] = [60.0, 20.0, 5.0, -5.0, -20.0];

/// Check 1: the captured sky environment map (clouds off) against the CPU
/// atmosphere, `sky_radiance` plus the night floor, at sampled upper-hemisphere
/// texels. The camera altitude is the capture's eye height.
pub fn check_sky_env(name: &str, captured: &Captured, sun_dir: Vec3, sun_color: Vec3) -> bool {
    let Some((raw, w, h)) = captured.sky_env.as_ref() else {
        eprintln!("sky env {name}: no environment map captured");
        return false;
    };
    let (w, h) = (*w, *h);
    if w != SKY_ENV_SIZE.0 || h != SKY_ENV_SIZE.1 || raw.len() != (w * h * 4) as usize {
        eprintln!("sky env {name}: unexpected map size {w}x{h}");
        return false;
    }
    let settings = AtmosphereSettings::earth();
    let model = AtmosphereModel::new(&settings);
    let night = Vec3::from(settings.night_sky);
    let altitude = captured.eye.y;
    let (mut sum, mut worst, mut n) = (0.0f64, 0.0f64, 0u32);
    for j in 0..GRID.1 {
        for i in 0..GRID.0 {
            // Texel centres in the upper half of the map (v < 0.5 is above the horizon).
            let x = ((i as f32 + 0.5) / GRID.0 as f32 * w as f32) as u32;
            let y = ((j as f32 + 0.5) / GRID.1 as f32 * 0.5 * h as f32) as u32;
            let uv = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
            let dir = sky_env_direction(uv);
            let cpu = model.sky_radiance(altitude, dir, sun_dir, sun_color) + night;
            let i0 = ((y * w + x) * 4) as usize;
            let gpu = Vec3::new(raw[i0], raw[i0 + 1], raw[i0 + 2]);
            let err =
                f64::from((gpu - cpu).abs().element_sum()) / f64::from(cpu.element_sum().max(1e-6));
            sum += err;
            worst = worst.max(err);
            n += 1;
        }
    }
    let mean = sum / f64::from(n);
    println!(
        "sky env {name}: {n} upper-hemisphere texels vs CPU, mean relative error {:.2}%, worst {:.1}% (limit {:.0}% mean)",
        mean * 100.0,
        worst * 100.0,
        MAX_ENV_CPU_ERROR * 100.0
    );
    if mean > MAX_ENV_CPU_ERROR {
        eprintln!(
            "sky env {name}: mean error {:.1}% above the limit",
            mean * 100.0
        );
        return false;
    }
    true
}

/// Mirror pixels of a frame of the mirror scene: for each sampled pixel whose
/// view ray reaches the floor (y = 0), the ray's mirror direction. `None` for
/// sky pixels and pixels off the floor.
fn mirror_direction(view: &View, x: u32, y: u32) -> Option<Vec3> {
    let Ray { origin, dir } = view.pixel_ray(f64::from(x) + 0.5, f64::from(y) + 0.5);
    if dir.y >= -1e-6 {
        return None;
    }
    let t = -origin.y / dir.y;
    if t <= 0.0 {
        return None;
    }
    let p: DVec3 = origin + dir * t;
    if p.x.abs() > f64::from(MIRROR_HALF) || p.z.abs() > f64::from(MIRROR_HALF) {
        return None;
    }
    Some(Vec3::new(dir.x as f32, -dir.y as f32, dir.z as f32).normalize())
}

/// Check 2: the mirror floor's pixels in the frame against the sky environment
/// map sampled along each pixel's mirror direction, both in 8-bit display steps
/// after the same (identity) tone curve. The mirror has no geometry in its
/// reflection, so every reflection misses and falls back to the map.
pub fn check_mirror(captured: &Captured, top_first: bool) -> bool {
    let Some((raw, mw, mh)) = captured.sky_env.as_ref() else {
        eprintln!("mirror: no environment map captured");
        return false;
    };
    let view = make_view(captured.eye, (W, H), top_first);
    let (mut sum, mut n) = (0.0f64, 0u64);
    let mut signed = [0.0f64; 3];
    let mut pred_mean = [0.0f64; 3];
    for y in (0..H).step_by(MIRROR_STEP as usize) {
        for x in (0..W).step_by(MIRROR_STEP as usize) {
            let Some(mirror) = mirror_direction(&view, x, y) else {
                continue;
            };
            let uv = sky_env_uv(mirror);
            let tx = ((uv[0] * *mw as f32) as u32).min(mw - 1);
            let ty = ((uv[1] * *mh as f32) as u32).min(mh - 1);
            let i0 = ((ty * mw + tx) * 4) as usize;
            let frame_i = ((y * W + x) * 4) as usize;
            for c in 0..3 {
                let expected = srgb_steps(f64::from(raw[i0 + c]));
                let got = f64::from(captured.mean_last[frame_i + c]);
                signed[c] += got - expected;
                pred_mean[c] += expected;
                sum += (got - expected).abs();
            }
            n += 3;
        }
    }
    if n == 0 {
        eprintln!("mirror: no floor pixels in the frame");
        return false;
    }
    let mae = sum / n as f64;
    let samples = (n / 3) as f64;
    println!(
        "mirror fallback: signed mean (frame - map) per channel {:.2?}, map mean {:.2?}",
        signed.map(|v| v / samples),
        pred_mean.map(|v| v / samples)
    );
    println!(
        "mirror fallback: {} mirror samples, mean of 8 frames, mean abs difference to the map along the reflection {mae:.2}/255 (limit {MIRROR_MAX_MAE})",
        n / 3
    );
    if mae > MIRROR_MAX_MAE {
        eprintln!("mirror fallback: {mae:.2}/255 above {MIRROR_MAX_MAE}");
        return false;
    }
    true
}

/// Driven ambient values at each sun elevation of [`AMBIENT_ELEVATIONS`], with
/// default settings, printed for the report. Returns the up-facing irradiance as
/// a fraction of the direct horizontal irradiance at each, by luminance (NaN with
/// the sun below the horizon).
pub fn print_ambient() -> Vec<(f32, f64)> {
    let model = AtmosphereModel::new(&AtmosphereSettings::earth());
    let mut ratios = Vec::new();
    for deg in AMBIENT_ELEVATIONS {
        let env = LightingEnv::with_sun(0.6, deg.to_radians(), 3.0);
        let driven = model.driven_environment(&env, 2.0, 0.0);
        let sun = Vec3::from_slice(&env.sun_direction[..3]);
        let sun_colour = Vec3::from_slice(&env.sun_color[..3]);
        let direct_h = sun_colour * sun.y;
        let zenith = Vec3::from_slice(&driven.sky_zenith[..3]);
        let horizon = Vec3::from_slice(&driven.sky_horizon[..3]);
        let ground = Vec3::from_slice(&driven.sky_ground[..3]);
        let luma = |c: Vec3| f64::from(c.dot(Vec3::new(0.2126, 0.7152, 0.0722)));
        // The ratio is only meaningful with the sun above the horizon.
        let ratio = (sun.y > 0.0).then(|| luma(zenith * std::f32::consts::PI) / luma(direct_h));
        let ratio_text = ratio.map_or("n/a (sun below the horizon)".to_string(), |r| {
            format!("{:.1}%", r * 100.0)
        });
        println!(
            "ambient {deg:>5} deg: zenith=({:.4}, {:.4}, {:.4}) horizon=({:.4}, {:.4}, {:.4}) ground=({:.4}, {:.4}, {:.4}) sun*sin=({:.4}, {:.4}, {:.4}) up-irradiance/direct (luma)={ratio_text}",
            zenith.x,
            zenith.y,
            zenith.z,
            horizon.x,
            horizon.y,
            horizon.z,
            ground.x,
            ground.y,
            ground.z,
            direct_h.x,
            direct_h.y,
            direct_h.z,
        );
        ratios.push((deg, ratio.unwrap_or(f64::NAN)));
    }
    ratios
}

//! Cloud checks: the cloud panorama's values and fractions, the sun-facing
//! brightness, the sunset colour, and the sky of the composed frame against
//! `sky_noon`. The panorama mapping mirrors `render::clouds` (Lambert
//! cylindrical equal-area projection of the upper hemisphere).

use super::{Captured, OUT_DIR, mean_abs_diff, mean_abs_diff_where};
use glam::Vec3;
use std::collections::HashMap;
use std::path::Path;

/// Cases whose captures the cloud checks read back.
pub(super) const KEPT: &[&str] = &["clouds", "sky_noon"];

/// Band of elevations, degrees, over which the gradient ratio is measured.
const RATIO_BAND_DEG: (f64, f64) = (5.0, 45.0);
/// Allowed range of the vertical / horizontal angular gradient ratio for `clouds`.
/// Oblique foreshortening of a horizontal cloud layer over `5-45 deg` gives
/// ~2.0x at `1024x512`; values `< 0.6` catch vertical columns and `> 2.2` catch
/// horizontal banding.
const RATIO_RANGE: (f64, f64) = (0.6, 2.2);
/// Transmittance below which a panorama texel counts as cloud.
const CLOUD_T: f32 = 0.5;
/// Largest mean abs difference of `clouds_motion` against its still reference (8-bit units).
pub(super) const MAX_MOTION_DIFF: f64 = 10.0;
/// Largest mean abs difference of consecutive frames at a fixed scene time.
const MAX_CONSECUTIVE_DIFF: f64 = 1.5;
/// Mean abs difference of the sky region, `clouds` against `sky_noon`, that the
/// composed clouds must exceed (8-bit units).
const MIN_SKY_CLOUD_DIFF: f64 = 4.0;

/// Direction of a panorama texel (mirror of `render::clouds::panorama_dir`).
fn panorama_dir(uv: [f32; 2]) -> Vec3 {
    let phi = (uv[0] - 0.5) * std::f32::consts::TAU;
    let y = 1.0 - uv[1];
    let s = (1.0 - y * y).max(0.0).sqrt();
    Vec3::new(s * phi.cos(), y, s * phi.sin())
}

/// Panorama statistics over the cloud texels (transmittance below `CLOUD_T`).
#[derive(Default)]
struct Stats {
    /// Texels that are cloud, of all texels.
    cloud_fraction: f64,
    /// Mean luminance of cloud texels, sun-facing and opposite halves of the sky.
    sun_luma: f64,
    opposite_luma: f64,
    /// Mean R / B of the cloud texels.
    red_over_blue: f64,
    /// Any NaN, infinity, negative radiance or transmittance outside [0, 1].
    bad_values: u64,
}

/// Compute the panorama statistics for `sun` (a direction towards the sun).
fn stats(pano: &(Vec<f32>, u32, u32), sun: Vec3) -> Stats {
    let (data, w, h) = pano;
    let (w, h) = (*w as usize, *h as usize);
    let mut out = Stats::default();
    let (mut n, mut sun_n, mut opp_n) = (0u64, 0u64, 0u64);
    let (mut sun_sum, mut opp_sum, mut r_sum, mut b_sum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let sun_h = Vec3::new(sun.x, 0.0, sun.z).normalize_or_zero();
    for y in 0..h {
        for x in 0..w {
            let texel = &data[(y * w + x) * 4..(y * w + x) * 4 + 4];
            let t = texel[3];
            if !texel.iter().all(|v| v.is_finite())
                || texel[..3].iter().any(|v| *v < 0.0)
                || !(0.0..=1.0).contains(&t)
            {
                out.bad_values += 1;
                continue;
            }
            if t >= CLOUD_T {
                continue;
            }
            n += 1;
            let luma = f64::from(texel[0]) * 0.2126
                + f64::from(texel[1]) * 0.7152
                + f64::from(texel[2]) * 0.0722;
            r_sum += f64::from(texel[0]);
            b_sum += f64::from(texel[2]);
            let dir = panorama_dir([(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32]);
            let facing = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero().dot(sun_h);
            if facing > 0.0 {
                sun_n += 1;
                sun_sum += luma;
            } else {
                opp_n += 1;
                opp_sum += luma;
            }
        }
    }
    let total = (w * h) as f64;
    out.cloud_fraction = n as f64 / total;
    out.sun_luma = sun_sum / sun_n.max(1) as f64;
    out.opposite_luma = opp_sum / opp_n.max(1) as f64;
    out.red_over_blue = r_sum / b_sum.max(1e-9);
    out
}

/// Mean absolute transmittance difference of vertically and horizontally adjacent
/// texels, each divided by the angle between the two texel directions, over the
/// elevation band. Returns (vertical, horizontal, vertical / horizontal). Columns
/// make the vertical gradient small, so the ratio falls far below one.
fn gradient_ratio(pano: &(Vec<f32>, u32, u32)) -> (f64, f64, f64) {
    let (data, w, h) = pano;
    let (w, h) = (*w as usize, *h as usize);
    let dir = |x: usize, y: usize| {
        panorama_dir([(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32])
    };
    let t = |x: usize, y: usize| f64::from(data[(y * w + x) * 4 + 3]);
    let in_band = |y: usize| {
        let el = f64::from(dir(0, y).y).asin().to_degrees();
        (RATIO_BAND_DEG.0..=RATIO_BAND_DEG.1).contains(&el)
    };
    let angle = |a: Vec3, b: Vec3| f64::from(a.dot(b).clamp(-1.0, 1.0)).acos().max(1e-9);
    let (mut vs, mut vn) = (0.0f64, 0u64);
    for y in 0..h.saturating_sub(1) {
        if !in_band(y) || !in_band(y + 1) {
            continue;
        }
        for x in 0..w {
            let a = angle(dir(x, y), dir(x, y + 1));
            vs += (t(x, y) - t(x, y + 1)).abs() / a;
            vn += 1;
        }
    }
    let (mut hs, mut hn) = (0.0f64, 0u64);
    for y in 0..h {
        if !in_band(y) {
            continue;
        }
        for x in 0..w {
            let xn = (x + 1) % w;
            let a = angle(dir(x, y), dir(xn, y));
            hs += (t(x, y) - t(xn, y)).abs() / a;
            hn += 1;
        }
    }
    let vertical = vs / vn.max(1) as f64;
    let horizontal = hs / hn.max(1) as f64;
    (vertical, horizontal, vertical / horizontal.max(1e-12))
}

/// Depth of the scene at a pixel (top row first), raw from the depth dump.
fn depth_value(depth: &(Vec<u8>, u32, u32), x: u32, y: u32) -> f32 {
    let i = ((y * depth.1 + x) * 4) as usize;
    f32::from_le_bytes([depth.0[i], depth.0[i + 1], depth.0[i + 2], depth.0[i + 3]])
}

/// Write the panorama as `<name>_clouds.png` (radiance, Reinhard, sRGB-ish) and
/// `<name>_cloudT.png` (transmittance).
fn dump_panorama(name: &str, pano: &(Vec<f32>, u32, u32)) {
    let (data, w, h) = pano;
    let mut rgb = Vec::with_capacity(data.len());
    let mut transmittance = Vec::with_capacity(data.len() / 4 * 4);
    for texel in data.chunks_exact(4) {
        for &c in &texel[..3] {
            let c = c.max(0.0);
            let v = (c / (1.0 + c)).powf(1.0 / 2.2);
            rgb.push((v * 255.0).round().clamp(0.0, 255.0) as u8);
        }
        rgb.push(255);
        let t = (texel[3].clamp(0.0, 1.0) * 255.0).round() as u8;
        transmittance.extend_from_slice(&[t, t, t, 255]);
    }
    let dir = Path::new(OUT_DIR);
    if let Err(error) = dream::png::write(&dir.join(format!("{name}_clouds.png")), *w, *h, &rgb) {
        eprintln!("failed to write {name}_clouds.png: {error}");
    }
    if let Err(error) = dream::png::write(
        &dir.join(format!("{name}_cloudT.png")),
        *w,
        *h,
        &transmittance,
    ) {
        eprintln!("failed to write {name}_cloudT.png: {error}");
    }
}

/// Check one case. Returns false when a cloud threshold fails.
pub(super) fn check_case(
    name: &str,
    captured: &Captured,
    kept: &HashMap<&'static str, Captured>,
    sun: Vec3,
) -> bool {
    let Some(pano) = captured.cloud_panorama.as_ref() else {
        return true;
    };
    let mut passed = true;
    dump_panorama(name, pano);
    let st = stats(pano, sun);
    println!(
        "{name}: cloud texels (T<0.5) {:.3} of the panorama",
        st.cloud_fraction
    );
    println!(
        "{name}: cloud luminance sun-facing {:.4} opposite {:.4}; R/B {:.4}",
        st.sun_luma, st.opposite_luma, st.red_over_blue
    );
    if st.bad_values > 0 {
        eprintln!(
            "case {name}: {} texels with NaN, Inf, negative or out-of-range values",
            st.bad_values
        );
        passed = false;
    }
    let (vertical, horizontal, ratio) = gradient_ratio(pano);
    println!(
        "{name}: transmittance gradient vertical {vertical:.5} horizontal {horizontal:.5} ratio {ratio:.3} (elevation {}-{} deg)",
        RATIO_BAND_DEG.0, RATIO_BAND_DEG.1
    );
    if name == "clouds" && !(RATIO_RANGE.0..=RATIO_RANGE.1).contains(&ratio) {
        eprintln!("case {name}: gradient ratio {ratio:.3} outside {RATIO_RANGE:?} (columns)");
        passed = false;
    }
    let fraction_ok = match name {
        "clouds" => (0.15..=0.75).contains(&st.cloud_fraction),
        "clouds_overcast" => st.cloud_fraction > 0.8,
        "clouds_wispy" => st.cloud_fraction < 0.35,
        _ => true,
    };
    if !fraction_ok {
        eprintln!(
            "case {name}: cloud fraction {:.3} outside its range",
            st.cloud_fraction
        );
        passed = false;
    }
    if matches!(name, "clouds" | "clouds_sunset") && st.sun_luma <= st.opposite_luma {
        eprintln!(
            "case {name}: sun-facing cloud luminance {:.4} not above opposite {:.4}",
            st.sun_luma, st.opposite_luma
        );
        passed = false;
    }
    if name == "clouds_sunset"
        && let Some(fair) = kept.get("clouds").and_then(|c| c.cloud_panorama.as_ref())
    {
        let base = stats(fair, sun);
        println!(
            "clouds_sunset: lit cloud R/B {:.4} vs clouds {:.4}",
            st.red_over_blue, base.red_over_blue
        );
        if st.red_over_blue <= base.red_over_blue {
            eprintln!("case clouds_sunset: R/B not warmer than clouds");
            passed = false;
        }
    }
    if name == "clouds"
        && let Some(noon) = kept.get("sky_noon")
    {
        let depth = &captured.depth;
        let sky = |x: u32, y: u32| depth_value(depth, x, y) >= 0.99999;
        let diff = mean_abs_diff_where(&captured.last, &noon.last, sky);
        println!("clouds: sky region diff vs sky_noon {diff:.3}/255");
        if diff <= MIN_SKY_CLOUD_DIFF {
            eprintln!("case clouds: sky diff {diff:.3} not above {MIN_SKY_CLOUD_DIFF}");
            passed = false;
        }
    }
    passed
}

/// `clouds_time`: the sky must change between scene times 30 s apart, while
/// consecutive frames at one time stay nearly identical.
pub(super) fn check_time(first: &Captured, later: &Captured) -> bool {
    let depth = &first.depth;
    let sky = |x: u32, y: u32| depth_value(depth, x, y) >= 0.99999;
    let sky_diff = mean_abs_diff_where(&first.last, &later.last, sky);
    let consecutive = mean_abs_diff(&first.last, &first.previous);
    println!("clouds_time: sky diff between t=0 and t=30s {sky_diff:.3}/255");
    println!("clouds_time: consecutive-frame diff at fixed time {consecutive:.3}/255");
    let mut passed = true;
    if sky_diff <= 0.0 {
        eprintln!("case clouds_time: the sky did not change over 30 s of scene time");
        passed = false;
    }
    if consecutive >= MAX_CONSECUTIVE_DIFF {
        eprintln!(
            "case clouds_time: consecutive-frame diff {consecutive:.3} above {MAX_CONSECUTIVE_DIFF}"
        );
        passed = false;
    }
    passed
}

//! Sky and fog checks of scene_check: the GPU sky-view LUT against the CPU
//! atmosphere, the sky and fog metrics, the fog-slice dumps, and the light-shaft
//! test against the reference scene's shadow of the green wall.

use super::reference::{Ray, Scene};
use super::{Captured, H, TARGET, W, camera_at, mean_abs_diff, save_dump};
use dream::lighting::LightingEnv;
use dream::render::{AtmosphereModel, AtmosphereSettings, sky_view_direction};
use glam::{DVec3, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// Fog volume of the scene-check cases: the medium preset's distance and the camera near plane.
const FOG_DISTANCE: f32 = 160.0;
const FOG_NEAR: f32 = 0.1;

/// Cases whose captures later checks read.
pub(super) const KEPT: &[&str] = &["sky_noon", "fog"];

/// Largest accepted GPU-versus-CPU sky radiance error, relative.
const MAX_SKY_CPU_ERROR: f32 = 0.15;
/// Largest accepted consecutive-frame difference of the static fog and everything cases.
const MAX_STATIC_DIFF: f64 = 2.5;
/// Largest accepted mean abs difference of the nearest tenth of the frame, fog against no fog.
const MAX_NEAREST_DIFF: f64 = 15.0;
/// Largest accepted ratio of green-wall-shadowed in-scatter to lit in-scatter at the same distance.
const MAX_SHAFT_RATIO: f64 = 0.75;
/// Fraction of the frame, from the top, that the sky check averages.
const TOP_FRACTION: f64 = 0.15;

/// The sun direction (to the sun) of a case's sun angles.
fn sun_dir(azimuth: f32, elevation: f32) -> Vec3 {
    Vec3::from_slice(&LightingEnv::with_sun(azimuth, elevation, 1.0).sun_direction[..3])
}

/// Mean RGB of the top `fraction` of the frame.
fn top_mean(rgba: &[u8], fraction: f64) -> [f64; 3] {
    let rows = (H as f64 * fraction) as usize;
    let mut sum = [0.0f64; 3];
    for y in 0..rows {
        for x in 0..W as usize {
            let p = pixel(rgba, x, y);
            for c in 0..3 {
                sum[c] += p[c] as f64;
            }
        }
    }
    let n = (rows * W as usize).max(1) as f64;
    sum.map(|v| v / n)
}

/// Slice depth of the fog volume's exponential mapping (the shader's `fog_slice_depth`).
fn slice_depth(s: f32, near: f32, dist: f32, slices: f32) -> f32 {
    near * (dist / near).powf(s / slices)
}

fn luma(rgb: [u8; 3]) -> f64 {
    0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64
}

fn pixel(rgba: &[u8], x: usize, y: usize) -> [u8; 3] {
    let i = (y * W as usize + x) * 4;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// Raw depth at a pixel of the depth dump (top row first, little-endian f32).
fn depth_at(depth: &[u8], x: usize, y: usize) -> f32 {
    let i = (y * W as usize + x) * 4;
    f32::from_le_bytes([depth[i], depth[i + 1], depth[i + 2], depth[i + 3]])
}

/// Row of the horizon in the frame: where the view ray is level.
fn horizon_row(eye: Vec3) -> f64 {
    let camera = camera_at(eye);
    let forward = (Vec3::from(TARGET) - eye).normalize();
    let level = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
    let far = eye + level * 1000.0;
    let clip = camera.view_projection() * far.extend(1.0);
    let ndc_y = (clip.y / clip.w) as f64;
    (1.0 - (ndc_y * 0.5 + 0.5)) * H as f64
}

/// Column of a world direction on the frame, if it is in front of the camera.
fn screen_column(eye: Vec3, direction: Vec3) -> Option<f64> {
    let camera = camera_at(eye);
    let clip = camera.view_projection() * (eye + direction * 1000.0).extend(1.0);
    (clip.w > 0.0).then(|| ((clip.x / clip.w) as f64 * 0.5 + 0.5) * W as f64)
}

/// Mean RGB and count of the sky pixels (raw depth at the far plane) in a box.
fn sky_mean(
    rgba: &[u8],
    depth: &[u8],
    rows: (usize, usize),
    cols: (usize, usize),
) -> ([f64; 3], u64) {
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    for y in rows.0..rows.1.min(H as usize) {
        for x in cols.0..cols.1.min(W as usize) {
            if depth_at(depth, x, y) < 0.99999 {
                continue;
            }
            let p = pixel(rgba, x, y);
            for c in 0..3 {
                sum[c] += p[c] as f64;
            }
            n += 1;
        }
    }
    if n > 0 {
        for v in &mut sum {
            *v /= n as f64;
        }
    }
    (sum, n)
}

/// Sky pixels in the band just above the horizon, near the sun's column.
fn horizon_band_rb(rgba: &[u8], depth: &[u8], eye: Vec3, sun: Vec3) -> f64 {
    let row = horizon_row(eye);
    let rows = (
        (row - 0.08 * H as f64).max(0.0) as usize,
        row.max(1.0) as usize,
    );
    let cols = match screen_column(eye, Vec3::new(sun.x, 0.0, sun.z).normalize_or_zero()) {
        Some(x) => {
            let half = 0.25 * W as f64;
            (
                ((x - half).max(0.0)) as usize,
                ((x + half).min(W as f64)) as usize,
            )
        }
        None => (0, W as usize),
    };
    let (mean, n) = sky_mean(rgba, depth, rows, cols);
    if n == 0 {
        return f64::NAN;
    }
    mean[0] / mean[2].max(1.0)
}

/// Standard deviation of luma over the geometry pixels below the horizon and
/// deeper than the median of those pixels (the far half of the ground).
fn far_ground_contrast(rgba: &[u8], depth: &[u8], eye: Vec3) -> f64 {
    let row = horizon_row(eye).max(0.0) as usize;
    let mut ground: Vec<(f32, f64)> = Vec::new();
    for y in row..H as usize {
        for x in 0..W as usize {
            let d = depth_at(depth, x, y);
            if d < 0.99999 {
                ground.push((d, luma(pixel(rgba, x, y))));
            }
        }
    }
    if ground.len() < 16 {
        return f64::NAN;
    }
    let mut depths: Vec<f32> = ground.iter().map(|g| g.0).collect();
    depths.sort_by(|a, b| a.total_cmp(b));
    let median = depths[depths.len() / 2];
    let far: Vec<f64> = ground
        .iter()
        .filter(|g| g.0 >= median)
        .map(|g| g.1)
        .collect();
    let mean = far.iter().sum::<f64>() / far.len() as f64;
    (far.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / far.len() as f64).sqrt()
}

/// Mean abs RGB difference of the nearest tenth of the geometry pixels.
fn nearest_tenth_diff(a: &[u8], b: &[u8], depth: &[u8]) -> f64 {
    let mut geometry: Vec<(f32, usize)> = Vec::new();
    for y in 0..H as usize {
        for x in 0..W as usize {
            let d = depth_at(depth, x, y);
            if d < 0.99999 {
                geometry.push((d, y * W as usize + x));
            }
        }
    }
    geometry.sort_by(|p, q| p.0.total_cmp(&q.0));
    let keep = (geometry.len() / 10).max(1);
    let mut sum = 0.0f64;
    for &(_, index) in geometry.iter().take(keep) {
        for c in 0..3 {
            sum += (a[index * 4 + c] as f64 - b[index * 4 + c] as f64).abs();
        }
    }
    sum / (keep * 3) as f64
}

/// Relative error of the GPU sky-view texel `(x, y)` against the CPU sky at the
/// same direction. Prints both; returns whether it is within tolerance.
#[allow(clippy::too_many_arguments)]
fn check_texel(
    label: &str,
    model: &AtmosphereModel,
    gpu: &[f32],
    (w, h): (u32, u32),
    (x, y): (u32, u32),
    altitude: f32,
    sun: Vec3,
    sun_color: Vec3,
) -> bool {
    let uv = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
    let dir = sky_view_direction(model, altitude, uv, sun);
    let cpu = model.sky_radiance(altitude, dir, sun, sun_color);
    let i = ((y * w + x) * 4) as usize;
    let g = Vec3::new(gpu[i], gpu[i + 1], gpu[i + 2]);
    let err = (g - cpu).abs().element_sum() / cpu.element_sum().max(1e-6);
    println!(
        "sky {label}: gpu=({:.4}, {:.4}, {:.4}) cpu=({:.4}, {:.4}, {:.4}) error={:.1}%",
        g.x,
        g.y,
        g.z,
        cpu.x,
        cpu.y,
        cpu.z,
        err * 100.0
    );
    if err > MAX_SKY_CPU_ERROR {
        eprintln!(
            "sky {label}: GPU and CPU differ by {:.1}% (limit {:.0}%)",
            err * 100.0,
            MAX_SKY_CPU_ERROR * 100.0
        );
        return false;
    }
    true
}

/// Reinhard-tone-mapped 8-bit image of linear radiance (`rgb` per texel, texel row 0 at the bottom).
fn reinhard_rgba(raw: &[f32], (w, h): (u32, u32), scale: f32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let src = ((y * w + x) * 4) as usize;
            // Row 0 holds uv.y near 0 (the bottom of the sky): show it at the bottom.
            let dst = (((h - 1 - y) * w + x) * 4) as usize;
            for c in 0..3 {
                let v = (raw[src + c] * scale).max(0.0);
                out[dst + c] = ((v / (1.0 + v)) * 255.0 + 0.5) as u8;
            }
            out[dst + 3] = 255;
        }
    }
    out
}

/// The fog slice through the middle of the frame: columns are screen x, rows
/// are depth slices (near at the bottom). Returns (in-scatter rgba, transmittance rgba).
fn fog_slice_images(volume: &[f32], (w, h, n): (u32, u32, u32)) -> (Vec<u8>, Vec<u8>) {
    let mut scatter = vec![0u8; (w * n * 4) as usize];
    let mut trans = vec![0u8; (w * n * 4) as usize];
    let mid = h / 2;
    for k in 0..n {
        for x in 0..w {
            let src = (((k * h + mid) * w + x) * 4) as usize;
            let dst = (((n - 1 - k) * w + x) * 4) as usize;
            for c in 0..3 {
                let v = (volume[src + c] * 4.0).clamp(0.0, 1.0);
                scatter[dst + c] = (v.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            }
            scatter[dst + 3] = 255;
            let t = volume[src + 3].clamp(0.0, 1.0);
            let g = (t.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            trans[dst..dst + 3].copy_from_slice(&[g, g, g]);
            trans[dst + 3] = 255;
        }
    }
    (scatter, trans)
}

/// Mean injected luma of the froxels that the reference scene's green wall
/// shadows from the sun, against those lit, at the same slice (distance).
/// Returns the mean ratio over the slices where both have samples.
fn green_wall_shadow_ratio(
    injected: &[f32],
    (w, h, n): (u32, u32, u32),
    eye: Vec3,
    sun: Vec3,
    near: f32,
) -> Option<(f64, usize)> {
    let scene = Scene::new();
    let wall = scene.index_of("green wall");
    let camera = camera_at(eye);
    let cam_to_world = camera.view().inverse();
    let tan_y = (camera.fov_y() * 0.5).tan();
    let aspect = camera.aspect();
    let dist = FOG_DISTANCE;
    let to_sun = DVec3::new(sun.x as f64, sun.y as f64, sun.z as f64).normalize();
    let mid = h / 2;
    let mut ratios = Vec::new();
    for k in 0..n {
        let z = slice_depth(k as f32 + 0.5, near, dist, n as f32);
        let mut shadow = (0.0f64, 0u32);
        let mut lit = (0.0f64, 0u32);
        for x in 0..w {
            let ndc_x = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let ndc_y = (mid as f32 + 0.5) / h as f32 * 2.0 - 1.0;
            let view = glam::Vec4::new(ndc_x * tan_y * aspect * z, ndc_y * tan_y * z, -z, 1.0);
            let p = (cam_to_world * view).truncate();
            let index = ((k * h + mid) * w + x) as usize * 4;
            let value = &injected[index..index + 3];
            let l = 0.2126 * value[0] as f64 + 0.7152 * value[1] as f64 + 0.0722 * value[2] as f64;
            let ray = Ray {
                origin: DVec3::new(p.x as f64, p.y as f64, p.z as f64),
                dir: to_sun,
            };
            match scene.trace(ray, 200.0) {
                Some(hit) if hit.object_index == wall => {
                    shadow.0 += l;
                    shadow.1 += 1;
                }
                None => {
                    lit.0 += l;
                    lit.1 += 1;
                }
                _ => {}
            }
        }
        if shadow.1 >= 5 && lit.1 >= 5 {
            let s = shadow.0 / shadow.1 as f64;
            let l = lit.0 / lit.1 as f64;
            if l > 1e-9 {
                ratios.push(s / l);
            }
        }
    }
    if ratios.is_empty() {
        return None;
    }
    Some((
        ratios.iter().sum::<f64>() / ratios.len() as f64,
        ratios.len(),
    ))
}

/// Run the sky and fog checks of one case. Returns whether they all pass;
/// `kept` holds the captures of earlier cases that some checks compare against.
pub(super) fn check_case(
    name: &str,
    captured: &Captured,
    kept: &HashMap<&'static str, Captured>,
    out: &Path,
    sun_angles: (f32, f32),
) -> bool {
    let mut ok = true;
    let eye = captured.eye;
    let sun = sun_dir(sun_angles.0, sun_angles.1);
    let rgba = &captured.last;
    let depth = &captured.depth.0;

    if let Some((raw, w, h)) = captured.sky_view.as_ref() {
        save_dump(
            &out.join(format!("{name}_skyview.png")),
            &(reinhard_rgba(raw, (*w, *h), 1.0), *w, *h),
        );
        let settings = if name == "sky_hazy" {
            AtmosphereSettings::hazy()
        } else {
            AtmosphereSettings::earth()
        };
        let model = AtmosphereModel::new(&settings);
        let altitude = eye.y;
        let sun_color = Vec3::new(3.0, 2.88, 2.64);
        let (w, h) = (*w, *h);
        // The LUT is a bottom-up texel grid: the zenith is near row 0, the horizon near row h/2.
        let zenith = check_texel(
            &format!("{name} zenith"),
            &model,
            raw,
            (w, h),
            (w / 2, 0),
            altitude,
            sun,
            sun_color,
        );
        let horizon = check_texel(
            &format!("{name} horizon (sun side)"),
            &model,
            raw,
            (w, h),
            (w / 2, h / 2 - 1),
            altitude,
            sun,
            sun_color,
        );
        let anti = check_texel(
            &format!("{name} horizon (away)"),
            &model,
            raw,
            (w, h),
            (0, h / 2 - 1),
            altitude,
            sun,
            sun_color,
        );
        ok &= zenith && horizon && anti;
    }

    if name == "sky_noon" {
        let mean = top_mean(rgba, TOP_FRACTION);
        println!(
            "sky_noon: top {:.0}% mean rgb=({:.1}, {:.1}, {:.1}) B>R {}",
            TOP_FRACTION * 100.0,
            mean[0],
            mean[1],
            mean[2],
            mean[2] > mean[0]
        );
        if mean[2] <= mean[0] {
            eprintln!("case sky_noon: top band is not blue");
            ok = false;
        }
        let rb = horizon_band_rb(rgba, depth, eye, sun);
        println!("sky_noon: horizon band R/B near the sun = {rb:.3}");
    }
    if name == "sky_sunset" {
        if let Some(noon) = kept.get("sky_noon") {
            let rb = horizon_band_rb(rgba, depth, eye, sun);
            let noon_rb = horizon_band_rb(&noon.last, &noon.depth.0, noon.eye, sun);
            println!("sky_sunset: horizon band R/B near the sun = {rb:.3} (sky_noon {noon_rb:.3})");
            if rb.is_nan() || rb <= noon_rb {
                eprintln!("case sky_sunset: horizon R/B {rb:.3} not above noon {noon_rb:.3}");
                ok = false;
            }
        } else {
            println!("sky_sunset: sky_noon not in this run; skipped comparison");
        }
    }
    if name == "sky_hazy" {
        let mean = top_mean(rgba, TOP_FRACTION);
        println!(
            "sky_hazy: top band mean rgb=({:.1}, {:.1}, {:.1})",
            mean[0], mean[1], mean[2]
        );
    }

    if let ("fog", Some(noon)) = (name, kept.get("sky_noon")) {
        let contrast = far_ground_contrast(rgba, depth, eye);
        let noon_contrast = far_ground_contrast(&noon.last, &noon.depth.0, noon.eye);
        println!("fog: far-ground luma std {contrast:.2} (sky_noon {noon_contrast:.2})");
        if contrast.is_nan() || contrast >= noon_contrast {
            eprintln!(
                "case fog: far-ground contrast {contrast:.2} not below sky_noon {noon_contrast:.2}"
            );
            ok = false;
        }
        let diff = nearest_tenth_diff(rgba, &noon.last, depth);
        println!("fog: nearest 10% mean abs diff vs sky_noon = {diff:.2}/255");
        if diff > MAX_NEAREST_DIFF {
            eprintln!("case fog: nearest tenth changed by {diff:.2} (limit {MAX_NEAREST_DIFF})");
            ok = false;
        }
    }
    if name == "fog" || name == "everything" {
        let previous = &captured.previous;
        if !previous.is_empty() {
            let diff = mean_abs_diff(rgba, previous);
            println!("{name}: consecutive-frame diff={diff:.3}/255 (limit {MAX_STATIC_DIFF})");
            if diff > MAX_STATIC_DIFF {
                eprintln!("case {name}: static-frame diff {diff:.3} above {MAX_STATIC_DIFF}");
                ok = false;
            }
        }
    }

    if let Some((volume, size)) = captured.fog_volume.as_ref() {
        let (scatter, trans) = fog_slice_images(volume, *size);
        let (w, _, n) = *size;
        save_dump(&out.join(format!("{name}_fogslice.png")), &(scatter, w, n));
        save_dump(&out.join(format!("{name}_fogslice_t.png")), &(trans, w, n));
    }
    if name == "fog_shafts" {
        match (captured.fog_injected.as_ref(), captured.fog_volume.as_ref()) {
            (Some((injected, size)), Some(_)) => {
                match green_wall_shadow_ratio(injected, *size, eye, sun, FOG_NEAR) {
                    Some((ratio, slices)) => {
                        println!(
                            "fog_shafts: green-wall shadow in-scatter / lit at the same depth = {ratio:.3} over {slices} slices (limit {MAX_SHAFT_RATIO})"
                        );
                        if ratio > MAX_SHAFT_RATIO {
                            eprintln!(
                                "case fog_shafts: shadowed/lit in-scatter {ratio:.3} above {MAX_SHAFT_RATIO}"
                            );
                            ok = false;
                        }
                    }
                    None => {
                        eprintln!("case fog_shafts: no slices with both shadowed and lit froxels");
                        ok = false;
                    }
                }
            }
            _ => {
                eprintln!("case fog_shafts: fog volume not captured");
                ok = false;
            }
        }
    }
    ok
}

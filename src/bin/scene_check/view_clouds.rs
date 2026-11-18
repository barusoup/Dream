//! Per-pixel view cloud checks: the static and motion stability of the
//! reprojected clouds, a fly-through inside the layer, depth against a tall box,
//! the cloud shadow on the ground, and the cost of each quality level. Each check
//! renders its own short sequence through the renderer and the scene's models.
//! Images go to `target/scene-check/`.

use super::{
    H, OUT_DIR, Object, W, earth_sky, material, mean_abs_diff_where, object, save, taa_default,
};
use dream::camera::Camera;
use dream::gfx::{GpuContext, Renderer, SceneFrame};
use dream::lighting::{Lighting, LightingEnv};
use dream::model::render::ModelRenderer;
use dream::model::shapes;
use dream::render::{CloudQuality, CloudSettings, RenderSettings};
use dream::settings::AaMode;
use glam::Vec3;
use std::path::Path;
use std::sync::Arc;
use winit::dpi::PhysicalSize;

/// Sun of the view checks: azimuth, elevation (radians) and intensity.
const SUN: (f32, f32, f32) = (2.2, 0.6, 3.0);
/// Scene time of every view check, seconds: the clouds hold still in time, so
/// only the camera and the history move.
const TIME: f64 = 100.0;
/// Largest mean abs difference of frames 8 apart at a still camera (8-bit units).
const MAX_STATIC_DIFF: f64 = 1.0;
/// Cloud base and thickness of the fair-weather preset, metres.
const FAIR_BASE: f32 = 1500.0;
const FAIR_THICKNESS: f32 = 2500.0;

/// The frames of one run: the captures asked for, the last depth dump, and the
/// post-pass GPU time sampled after every frame.
struct Run {
    captures: Vec<(usize, Vec<u8>)>,
    depth: (Vec<u8>, u32, u32),
    post_ms: Vec<f32>,
}

/// Depth of pixel (x, y) from a depth dump (raw, top row first).
fn depth_at(depth: &(Vec<u8>, u32, u32), x: u32, y: u32) -> f32 {
    let i = ((y * depth.1 + x) * 4) as usize;
    f32::from_le_bytes([depth.0[i], depth.0[i + 1], depth.0[i + 2], depth.0[i + 3]])
}

/// Pixels with no geometry (the sky).
fn sky_of(depth: &(Vec<u8>, u32, u32)) -> impl Fn(u32, u32) -> bool + '_ {
    move |x, y| depth_at(depth, x, y) >= 0.99999
}

/// Pixels that hold geometry.
fn solid_of(depth: &(Vec<u8>, u32, u32)) -> impl Fn(u32, u32) -> bool + '_ {
    move |x, y| depth_at(depth, x, y) < 0.99999
}

/// Mean Rec. 709 luminance of the pixels `keep` selects (8-bit units).
fn mean_luma(rgba: &[u8], keep: impl Fn(u32, u32) -> bool) -> f64 {
    mean_rgb(rgba, keep)
        .iter()
        .zip([0.2126, 0.7152, 0.0722])
        .map(|(c, w)| c * w)
        .sum()
}

/// Mean RGB of the pixels `keep` selects (8-bit units).
fn mean_rgb(rgba: &[u8], keep: impl Fn(u32, u32) -> bool) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    for y in 0..H {
        for x in 0..W {
            if !keep(x, y) {
                continue;
            }
            let i = ((y * W + x) * 4) as usize;
            for c in 0..3 {
                sum[c] += f64::from(rgba[i + c]);
            }
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    sum.map(|s| s / n)
}

/// Render settings of the view checks: TAA, the earth sky, and the given clouds.
fn settings(clouds: Option<CloudSettings>) -> RenderSettings {
    let mut settings = RenderSettings::default();
    settings.taa = taa_default();
    settings.atmosphere = Some(earth_sky());
    settings.clouds = clouds;
    settings
}

/// Render `frames` frames of `objects` under the camera `pose(frame)` (eye,
/// target), with clouds `clouds`. `reset_each` invalidates the history before
/// every frame. The frames in `captures` are read back.
#[allow(clippy::too_many_arguments)]
fn run_frames(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    objects: &mut [Object],
    clouds: Option<CloudSettings>,
    frames: usize,
    pose: &dyn Fn(usize) -> (Vec3, Vec3),
    reset_each: bool,
    captures: &[usize],
) -> Run {
    renderer.set_aa_mode(AaMode::Taa);
    renderer.set_scene_time_scale(0.0);
    renderer.set_scene_time(TIME);
    renderer.set_render_settings(settings(clouds));
    renderer.reset_history();
    let mut out = Run {
        captures: Vec::new(),
        depth: (Vec::new(), 0, 0),
        post_ms: Vec::new(),
    };
    for frame_index in 0..frames {
        if reset_each {
            renderer.reset_history();
        }
        let (eye, target) = pose(frame_index);
        let mut camera = Camera::default();
        camera.set_aspect(W as f32 / H as f32);
        camera.position = eye;
        let forward = (target - eye).normalize();
        camera.set_look(forward.z.atan2(forward.x), forward.y.asin());
        let env = LightingEnv {
            cam_pos: eye.extend(0.0).to_array(),
            ..LightingEnv::with_sun(SUN.0, SUN.1, SUN.2)
        };
        let frame = SceneFrame {
            camera: &camera,
            lighting,
            env,
            scene_rect: None,
            ui: None,
        };
        for object in objects.iter_mut() {
            object.instance.update(0.0);
        }
        renderer
            .render_scene(&frame, |draw| {
                for object in objects.iter_mut() {
                    models.draw_pbr(draw, &mut object.gpu, &object.instance, lighting);
                }
                objects.len()
            })
            .expect("render scene frame");
        out.post_ms.push(renderer.post_gpu_ms());
        if captures.contains(&frame_index) {
            out.captures.push((frame_index, renderer.capture_rgba8().0));
        }
    }
    out.depth = renderer.capture_depth_f32();
    out
}

/// The capture of frame `index` in a run.
fn capture_of(run: &Run, index: usize) -> &[u8] {
    &run.captures
        .iter()
        .find(|(i, _)| *i == index)
        .expect("captured frame")
        .1
}

/// Mean of the post-pass GPU time over the last `count` frames that reported one.
fn recent_post_ms(run: &Run, count: usize) -> f64 {
    let samples: Vec<f32> = run.post_ms.iter().copied().filter(|ms| *ms > 0.0).collect();
    let tail = &samples[samples.len().saturating_sub(count)..];
    if tail.is_empty() {
        return 0.0;
    }
    tail.iter().map(|ms| f64::from(*ms)).sum::<f64>() / tail.len() as f64
}

/// A camera on the ground, `pitch_deg` above the horizon, looking along -z.
fn ground_pose(pitch_deg: f32) -> impl Fn(usize) -> (Vec3, Vec3) {
    move |_| {
        let eye = Vec3::new(0.0, 2.0, 0.0);
        let p = pitch_deg.to_radians();
        (eye, eye + Vec3::new(0.0, p.sin(), -p.cos()) * 100.0)
    }
}

/// Check a: a still camera, TAA on, 60 frames. Frames 8 apart over the sky.
fn check_static(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    objects: &mut [Object],
) -> bool {
    let pose = ground_pose(15.0);
    let run = run_frames(
        renderer,
        models,
        lighting,
        objects,
        Some(CloudSettings::fair_weather()),
        61,
        &pose,
        false,
        &[52, 60],
    );
    let diff = mean_abs_diff_where(
        capture_of(&run, 52),
        capture_of(&run, 60),
        sky_of(&run.depth),
    );
    println!(
        "clouds_static: sky mean abs diff, frame 60 vs frame 52 (still camera): {diff:.3}/255 (limit {MAX_STATIC_DIFF})"
    );
    // Cloud texture: the clouds-off frame at the same pose gives the opacity
    // proxy the metric needs.
    let clear = run_frames(
        renderer,
        models,
        lighting,
        objects,
        None,
        61,
        &pose,
        false,
        &[60],
    );
    let (hf, pixels) = cloud_hf_energy(
        capture_of(&run, 60),
        capture_of(&clear, 60),
        sky_of(&run.depth),
    );
    println!(
        "clouds_static: cloud high-frequency energy after 60 frames: {hf:.3}/255 over {pixels} cloud px (opacity proxy 0.3-1.0)"
    );
    save(
        &Path::new(OUT_DIR).join("clouds_static_frame60.png"),
        capture_of(&run, 60),
    );
    if diff >= MAX_STATIC_DIFF {
        eprintln!("clouds_static: sky diff {diff:.3} not below {MAX_STATIC_DIFF}");
        return false;
    }
    true
}

/// High-frequency energy inside cloud pixels: the mean absolute difference
/// between a pixel's luma and the mean of its four neighbours, over sky pixels
/// whose opacity proxy lies in [0.3, 1.0]. Opacity is not read back, so the
/// proxy is the luma the clouds add against the clouds-off frame at the same
/// pose, relative to the 99th percentile of that change over the sky. Returns
/// the mean and the number of pixels measured.
fn cloud_hf_energy(on: &[u8], off: &[u8], keep: impl Fn(u32, u32) -> bool) -> (f64, usize) {
    let (w, h) = (super::W as usize, super::H as usize);
    let luma = |buf: &[u8], x: usize, y: usize| {
        let i = (y * w + x) * 4;
        0.2126 * f64::from(buf[i]) + 0.7152 * f64::from(buf[i + 1]) + 0.0722 * f64::from(buf[i + 2])
    };
    let mut change = vec![0.0f64; w * h];
    let mut kept = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if keep(x as u32, y as u32) {
                let d = (luma(on, x, y) - luma(off, x, y)).abs();
                change[y * w + x] = d;
                kept.push(d);
            }
        }
    }
    if kept.is_empty() {
        return (0.0, 0);
    }
    kept.sort_by(|a, b| a.total_cmp(b));
    let p99 = kept[(kept.len() - 1) * 99 / 100].max(1e-6);
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if !keep(x as u32, y as u32) {
                continue;
            }
            let alpha = change[y * w + x] / p99;
            if !(0.3..=1.0).contains(&alpha) {
                continue;
            }
            let avg = 0.25
                * (luma(on, x - 1, y)
                    + luma(on, x + 1, y)
                    + luma(on, x, y - 1)
                    + luma(on, x, y + 1));
            sum += (luma(on, x, y) - avg).abs();
            n += 1;
        }
    }
    (if n > 0 { sum / n as f64 } else { 0.0 }, n)
}

/// Check b: the camera moves at 40 m/s, 300 m below the cloud base, for 60 frames.
/// Frame-to-frame sky difference with the history, minus the same with the
/// history reset every frame.
fn check_motion(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    objects: &mut [Object],
) -> bool {
    let pose = |f: usize| {
        let eye = Vec3::new(40.0 * f as f32 / 60.0, FAIR_BASE - 300.0, 0.0);
        let p = 5.0f32.to_radians();
        (eye, eye + Vec3::new(p.cos(), p.sin(), 0.0) * 100.0)
    };
    let captures: Vec<usize> = (30..60).collect();
    let with = run_frames(
        renderer,
        models,
        lighting,
        objects,
        Some(CloudSettings::fair_weather()),
        60,
        &pose,
        false,
        &captures,
    );
    let without = run_frames(
        renderer,
        models,
        lighting,
        objects,
        Some(CloudSettings::fair_weather()),
        60,
        &pose,
        true,
        &captures,
    );
    let consecutive = |run: &Run| {
        let keep = sky_of(&run.depth);
        let pairs: Vec<f64> = (30..59)
            .map(|i| mean_abs_diff_where(capture_of(run, i), capture_of(run, i + 1), &keep))
            .collect();
        pairs.iter().sum::<f64>() / pairs.len() as f64
    };
    let a = consecutive(&with);
    let b = consecutive(&without);
    println!(
        "clouds_motion: sky frame-to-frame diff at 40 m/s: history {a:.3}/255, history reset each frame {b:.3}/255, excess {:.3}/255",
        a - b
    );
    save(
        &Path::new(OUT_DIR).join("clouds_near.png"),
        capture_of(&with, 59),
    );
    true
}

/// Check c: a camera inside the layer, looking horizontally. A sphere at 150 m and
/// one at 600 m are each drawn alone, clear air and in the clouds. The camera
/// position is searched on a grid so that cloud lies in front of the 600 m sphere.
/// Black probes isolate the cloud's in-scattered light, which must be positive
/// and grow with distance. Coloured probes also receive cloud shadows, so their
/// clear-air and cloud luminances are reported for inspection only.
fn check_inside(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    ctx: &Arc<GpuContext>,
) -> bool {
    let dir = Vec3::new(0.0, 0.0, -1.0);
    let mut best = (f64::MIN, Vec3::ZERO);
    for height in [
        FAIR_BASE + 0.4 * FAIR_THICKNESS,
        FAIR_BASE + 0.6 * FAIR_THICKNESS,
        FAIR_BASE + 0.8 * FAIR_THICKNESS,
    ] {
        for x in [-1500.0f32, 0.0, 1500.0] {
            for z in [-1500.0f32, 0.0, 1500.0] {
                let eye = Vec3::new(x, height, z);
                let pose = move |_: usize| (eye, eye + dir * 100.0);
                let probe = || {
                    vec![object(
                        ctx,
                        shapes::uv_sphere(
                            100.0,
                            32,
                            16,
                            material("probe", [0.9, 0.35, 0.1, 1.0], 0.0, 0.8),
                        ),
                        (eye + dir * 600.0).to_array(),
                    )]
                };
                let mut clear_objects = probe();
                let clear = run_frames(
                    renderer,
                    models,
                    lighting,
                    &mut clear_objects,
                    None,
                    24,
                    &pose,
                    false,
                    &[23],
                );
                let mut cloud_objects = probe();
                let cloud = run_frames(
                    renderer,
                    models,
                    lighting,
                    &mut cloud_objects,
                    Some(CloudSettings::fair_weather()),
                    24,
                    &pose,
                    false,
                    &[23],
                );
                let fog = (mean_luma(capture_of(&cloud, 23), solid_of(&clear.depth))
                    - mean_luma(capture_of(&clear, 23), solid_of(&clear.depth)))
                .abs();
                println!(
                    "clouds_inside: eye ({x:.0}, {height:.0}, {z:.0}) m: 600 m sphere changes by {fog:.2}/255"
                );
                if fog > best.0 {
                    best = (fog, eye);
                }
            }
        }
    }
    let eye = best.1;
    println!(
        "clouds_inside: camera at ({:.0}, {:.0}, {:.0}) m, looking -z",
        eye.x, eye.y, eye.z
    );
    let pose = move |_: usize| (eye, eye + dir * 100.0);
    // The coloured probe is shadowed by the cloud as well as lit by it, so its
    // colour is information only; the verdict is the black-probe in-scatter.
    let mut shares = Vec::new();
    for (distance, radius) in [(150.0f32, 25.0f32), (600.0, 100.0)] {
        let probe = || {
            vec![object(
                ctx,
                shapes::uv_sphere(
                    radius,
                    32,
                    16,
                    material("probe", [0.9, 0.35, 0.1, 1.0], 0.0, 0.8),
                ),
                (eye + dir * distance).to_array(),
            )]
        };
        let mut clear_objects = probe();
        let clear = run_frames(
            renderer,
            models,
            lighting,
            &mut clear_objects,
            None,
            40,
            &pose,
            false,
            &[39],
        );
        let mut cloud_objects = probe();
        let cloud = run_frames(
            renderer,
            models,
            lighting,
            &mut cloud_objects,
            Some(CloudSettings::fair_weather()),
            40,
            &pose,
            false,
            &[39],
        );
        let sphere_clear = mean_luma(capture_of(&clear, 39), solid_of(&clear.depth));
        let sphere = mean_luma(capture_of(&cloud, 39), solid_of(&clear.depth));
        let sky_cloud = mean_luma(capture_of(&cloud, 39), sky_of(&cloud.depth));
        let sky_clear = mean_luma(capture_of(&clear, 39), sky_of(&clear.depth));
        println!(
            "clouds_inside: sphere at {distance:.0} m: clear air {sphere_clear:.2}, in clouds {sphere:.2}; sky clear {sky_clear:.2}, in clouds {sky_cloud:.2}"
        );
        if distance == 150.0 {
            save(
                &Path::new(OUT_DIR).join("clouds_inside.png"),
                capture_of(&cloud, 39),
            );
        }
        let (lo, hi) = (sphere_clear.min(sky_cloud), sphere_clear.max(sky_cloud));
        let between = (lo - 0.5..=hi + 0.5).contains(&sphere);
        let share = (sphere - sphere_clear) / (sky_cloud - sphere_clear).abs().max(1e-6)
            * (sky_cloud - sphere_clear).signum();
        println!(
            "clouds_inside: info only: coloured probe at {distance:.0} m between its clear and cloud luminance: {between}; share of the cloud colour {share:.3}"
        );
        shares.push(share);
    }
    println!(
        "clouds_inside: info only: coloured share {:.3} at 150 m, {:.3} at 600 m",
        shares[0], shares[1]
    );
    // In-scatter alone: a black probe (no albedo) has almost no lighting of its own,
    // so its cloud-minus-clear difference is the composite's cloud radiance.
    let mut scatter = Vec::new();
    for (distance, radius) in [(150.0f32, 25.0f32), (600.0, 100.0)] {
        let probe = || {
            vec![object(
                ctx,
                shapes::uv_sphere(
                    radius,
                    32,
                    16,
                    material("black probe", [0.0, 0.0, 0.0, 1.0], 0.0, 1.0),
                ),
                (eye + dir * distance).to_array(),
            )]
        };
        let mut clear_objects = probe();
        let clear = run_frames(
            renderer,
            models,
            lighting,
            &mut clear_objects,
            None,
            40,
            &pose,
            false,
            &[39],
        );
        let mut cloud_objects = probe();
        let cloud = run_frames(
            renderer,
            models,
            lighting,
            &mut cloud_objects,
            Some(CloudSettings::fair_weather()),
            40,
            &pose,
            false,
            &[39],
        );
        let delta = mean_luma(capture_of(&cloud, 39), solid_of(&clear.depth))
            - mean_luma(capture_of(&clear, 39), solid_of(&clear.depth));
        println!("clouds_inside: black probe at {distance:.0} m: in-scatter delta {delta:.2}/255");
        scatter.push(delta);
    }
    // The criterion: in-scatter is positive at both distances and grows.
    let scatter_grows = scatter[0] > 0.0 && scatter[1] > scatter[0];
    println!(
        "clouds_inside: in-scatter grows with distance ({:.2} at 150 m, {:.2} at 600 m): {scatter_grows}",
        scatter[0], scatter[1]
    );
    scatter_grows
}

/// Check d: a tall red box rising into an overcast layer, seen from the ground.
/// The box pixels that the cloud in front of them covers must change.
fn check_occlusion(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    ctx: &Arc<GpuContext>,
) -> bool {
    let make = || {
        vec![object(
            ctx,
            shapes::cuboid(
                [25.0, 1000.0, 25.0],
                material("box", [0.8, 0.05, 0.05, 1.0], 0.0, 0.7),
            ),
            [0.0, 1000.0, -600.0],
        )]
    };
    let pose = |_: usize| {
        let eye = Vec3::new(0.0, 2.0, 0.0);
        (eye, Vec3::new(0.0, 1500.0, -600.0))
    };
    let mut with_objects = make();
    let with = run_frames(
        renderer,
        models,
        lighting,
        &mut with_objects,
        Some(CloudSettings::overcast()),
        24,
        &pose,
        false,
        &[23],
    );
    let mut without_objects = make();
    let without = run_frames(
        renderer,
        models,
        lighting,
        &mut without_objects,
        None,
        24,
        &pose,
        false,
        &[23],
    );
    let red = |rgba: &[u8], x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        rgba[i] > 60 && u16::from(rgba[i]) > 2 * u16::from(rgba[i + 1]) + 20
    };
    let box_mask =
        |x: u32, y: u32| red(capture_of(&without, 23), x, y) && solid_of(&without.depth)(x, y);
    let (mut boxed, mut changed) = (0u64, 0u64);
    for y in 0..H {
        for x in 0..W {
            if !box_mask(x, y) {
                continue;
            }
            boxed += 1;
            // Attenuated: the cloud in front has covered the red of the box. Cloud
            // lighting also shifts the box's colour, so only the loss of red
            // dominance counts.
            if !red(capture_of(&with, 23), x, y) {
                changed += 1;
            }
        }
    }
    let fraction = changed as f64 / boxed.max(1) as f64;
    println!(
        "clouds_occlude: {changed} of {boxed} box pixels lost their red to the cloud in front ({:.1}%)",
        fraction * 100.0
    );
    save(
        &Path::new(OUT_DIR).join("clouds_occlude.png"),
        capture_of(&with, 23),
    );
    let ok = boxed > 0 && changed > 0;
    if !ok {
        eprintln!("clouds_occlude: no box pixel is covered by the cloud in front of it");
    }
    ok
}

/// Check e: a sunlit plane. Overcast cover against clear sky, and a picture of
/// the shadows under fair-weather clouds.
fn check_shadow(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    ctx: &Arc<GpuContext>,
) -> bool {
    let plane = || {
        vec![object(
            ctx,
            shapes::plane_rect(
                4000.0,
                4000.0,
                material("ground", [0.7, 0.7, 0.7, 1.0], 0.0, 0.9),
            ),
            [0.0, 0.0, 0.0],
        )]
    };
    let pose = |_: usize| (Vec3::new(0.0, 300.0, 300.0), Vec3::ZERO);
    let mut clear_objects = plane();
    let clear = run_frames(
        renderer,
        models,
        lighting,
        &mut clear_objects,
        None,
        24,
        &pose,
        false,
        &[23],
    );
    let mut overcast_objects = plane();
    let overcast = run_frames(
        renderer,
        models,
        lighting,
        &mut overcast_objects,
        Some(CloudSettings::overcast()),
        24,
        &pose,
        false,
        &[23],
    );
    let lit_clear = mean_luma(capture_of(&clear, 23), solid_of(&clear.depth));
    let lit_cloud = mean_luma(capture_of(&overcast, 23), solid_of(&overcast.depth));
    let ratio = lit_cloud / lit_clear.max(1e-6);
    println!(
        "clouds_shadow: sunlit plane luminance clear {lit_clear:.2}, under overcast {lit_cloud:.2}, ratio {ratio:.3} (expect < 0.9)"
    );

    let pose = |_: usize| (Vec3::new(0.0, 1000.0, 1000.0), Vec3::ZERO);
    let mut fair_objects = plane();
    let fair = run_frames(
        renderer,
        models,
        lighting,
        &mut fair_objects,
        Some(CloudSettings::fair_weather()),
        30,
        &pose,
        false,
        &[29],
    );
    save(
        &Path::new(OUT_DIR).join("clouds_shadow.png"),
        capture_of(&fair, 29),
    );
    if ratio >= 0.9 {
        eprintln!("clouds_shadow: overcast ratio {ratio:.3} not below 0.9");
        return false;
    }
    true
}

/// Check f: post-pass GPU time per quality level, on the horizon and inside the
/// layer, at the window size and at 2560 x 1440 when the window takes it.
fn check_timing(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    ctx: &Arc<GpuContext>,
    window: &winit::window::Window,
) {
    let levels: [(&str, Option<CloudQuality>); 4] = [
        ("off", None),
        ("low", Some(CloudQuality::Low)),
        ("medium", Some(CloudQuality::Medium)),
        ("high", Some(CloudQuality::High)),
    ];
    let inside_pose = |_: usize| {
        let eye = Vec3::new(0.0, FAIR_BASE + 0.5 * FAIR_THICKNESS, 0.0);
        (eye, eye + Vec3::new(0.0, 0.0, -100.0))
    };
    let horizon = ground_pose(0.0);
    let scene = || {
        vec![object(
            ctx,
            shapes::plane_rect(
                4000.0,
                4000.0,
                material("ground", [0.4, 0.45, 0.4, 1.0], 0.0, 0.9),
            ),
            [0.0, 0.0, 0.0],
        )]
    };
    let sizes = [(W, H), (2560u32, 1440u32)];
    for (w, h) in sizes {
        let resized = if (w, h) == (W, H) {
            true
        } else {
            let _ = window.request_inner_size(PhysicalSize::new(w, h));
            renderer.resize(PhysicalSize::new(w, h));
            renderer.render_size() == (w, h)
        };
        if !resized {
            println!(
                "clouds_timing: {w}x{h} not measured (the hidden window did not take the size)"
            );
            continue;
        }
        for (view, pose) in [
            ("horizon", &horizon as &dyn Fn(usize) -> (Vec3, Vec3)),
            ("inside", &inside_pose),
        ] {
            for (name, quality) in levels {
                let mut objects = scene();
                let clouds = quality.map(|q| {
                    let mut settings = CloudSettings::fair_weather();
                    settings.quality = q;
                    settings
                });
                let run = run_frames(
                    renderer,
                    models,
                    lighting,
                    &mut objects,
                    clouds,
                    120,
                    pose,
                    false,
                    &[],
                );
                println!(
                    "clouds_timing: {w}x{h} {view} clouds {name}: post_gpu_ms {:.3} (mean of the last 60 frames)",
                    recent_post_ms(&run, 60)
                );
            }
        }
        if (w, h) != (W, H) {
            let _ = window.request_inner_size(PhysicalSize::new(W, H));
            renderer.resize(PhysicalSize::new(W, H));
        }
    }
}

/// Run every view-cloud check. Returns false when a gated check fails.
pub(super) fn run(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    ctx: &Arc<GpuContext>,
    window: &winit::window::Window,
) -> bool {
    let mut ok = true;
    let mut objects = super::build_scene(ctx);
    ok &= check_static(renderer, models, lighting, &mut objects);
    check_motion(renderer, models, lighting, &mut objects);
    ok &= check_inside(renderer, models, lighting, ctx);
    ok &= check_occlusion(renderer, models, lighting, ctx);
    ok &= check_shadow(renderer, models, lighting, ctx);
    check_timing(renderer, models, lighting, ctx, window);
    ok
}

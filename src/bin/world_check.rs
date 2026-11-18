//! Verifies the culling, LOD and impostor world (`dream::world`).
//!
//! Five checks, printed with every number and failing with exit code 1:
//!
//! 1. culling correctness: the same 60-frame camera path rendered with all
//!    culling off and then fully on; every frame pair must match closely
//!    (mean and fraction of strongly differing pixels).
//! 2. effectiveness: frustum culling removes a share of the instances, and
//!    occlusion culling removes some of the survivors while the camera is
//!    behind the wall. `WorldStats` is printed every 10 frames.
//! 3. LOD: a dolly from 5 m to 120 m from the grid centre shifts draws from
//!    mesh level 0 toward the coarser levels and impostors, and frame-to-frame
//!    differences stay small.
//! 4. impostor cache: a second `add_model` with `BakeCached` loads the file
//!    instead of baking, and renders the same.
//! 5. timing: CPU cull time, world CPU time and GPU scene/post times with
//!    culling off and on. Reported, never asserted.
//!
//! Run `cargo run --release --bin world_check`. PNGs go to `target/world-check/`.

use dream::camera::Camera;
use dream::gfx::{Renderer, SurfaceError};
use dream::impostor::{BakeSettings, LodPolicy};
use dream::lighting::LightingEnv;
use dream::model::{Material, Model, shapes};
use dream::render::{OcclusionSettings, RenderSettings, ShadowSettings};
use dream::settings::{AaMode, Settings, UpscalePreset};
use dream::world::{ImpostorSource, InstanceHandle, LodModel, ModelHandle, World, WorldStats};
use glam::{Mat4, Vec3};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const W: u32 = 960;
const H: u32 = 540;
const OUT_DIR: &str = "target/world-check";
const CACHE_DIR: &str = "target/world-check/impostor-cache";
const DT: f32 = 1.0 / 60.0;
const GRID_SIDE: u32 = 40;
const GRID_SPACING: f32 = 2.0;
const GRID_CENTRE: Vec3 = Vec3::new(0.0, 0.0, -36.0);
const WALL_CENTRE: Vec3 = Vec3::new(0.0, 2.5, -6.0);
const PATH_FRAMES: usize = 60;

/// Check 1 thresholds (0-255 scale).
const DIFF_MEAN_LIMIT: f64 = 0.6;
const DIFF_STRONG_LEVEL: f64 = 24.0;
const DIFF_STRONG_FRACTION: f64 = 0.0005;
/// Check 3 and 4 thresholds (0-255 scale).
const DOLLY_MEAN_LIMIT: f64 = 3.0;
const CACHE_MEAN_LIMIT: f64 = 0.5;

#[derive(Clone, Copy)]
struct Pose {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

struct Scene {
    world: World,
    wall: InstanceHandle,
    instances: usize,
}

/// Failures are collected, printed, and turned into the exit code at the end.
#[derive(Default)]
struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, ok: bool, message: String) {
        if ok {
            println!("[world-check] PASS {message}");
        } else {
            println!("[world-check] FAIL {message}");
            self.failures.push(message);
        }
    }
}

fn material(rgb: [f32; 3]) -> Material {
    Material {
        base_color: [rgb[0], rgb[1], rgb[2], 1.0],
        ..Material::default()
    }
}

fn sphere(segments: u32, rings: u32) -> Arc<Model> {
    Arc::new(shapes::uv_sphere(
        1.0,
        segments,
        rings,
        material([0.7, 0.5, 0.3]),
    ))
}

/// Diagnostic switch: `WC_EXPERIMENT=noshadows | frustum-only | no-impostor`.
fn experiment() -> String {
    std::env::var("WC_EXPERIMENT").unwrap_or_default()
}

fn policy() -> LodPolicy {
    let no_impostor = experiment() == "no-impostor";
    LodPolicy {
        impostor_below: if no_impostor { 1e-6 } else { 0.04 },
        mesh_above: if no_impostor { 2e-6 } else { 0.06 },
        fade: 0.6,
        max_distance: 4096.0,
    }
}

/// The LOD sphere: 48x24, 16x8 and 8x4 meshes, plus a compact impostor.
fn grid_model(impostor: ImpostorSource) -> LodModel {
    sphere_model(impostor, policy())
}

/// The same LOD sphere under an explicit policy.
fn sphere_model(impostor: ImpostorSource, policy: LodPolicy) -> LodModel {
    LodModel::builder(sphere(48, 24))
        .lod(sphere(16, 8), 0.25)
        .lod(sphere(8, 4), 0.10)
        .impostor(impostor)
        .policy(policy)
        .build()
}

fn renderer_settings(occlusion: bool) -> RenderSettings {
    let mut settings = RenderSettings::default();
    if experiment() != "noshadows" {
        settings.shadows = Some(ShadowSettings::medium());
    }
    settings.occlusion = occlusion.then(OcclusionSettings::default);
    settings
}

fn make_renderer(event_loop: &ActiveEventLoop, occlusion: bool) -> Renderer {
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Dream World Check")
                    .with_visible(false)
                    .with_inner_size(PhysicalSize::new(W, H)),
            )
            .expect("create hidden window"),
    );
    let mut renderer = Renderer::new(
        window,
        dream::DEFAULT_SCENE_WGSL,
        Settings {
            aa: AaMode::Taa,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
        },
    );
    renderer.set_frame_capture(true);
    renderer.set_render_settings(renderer_settings(occlusion));
    assert_eq!(renderer.render_size(), (W, H), "render size");
    renderer
}

fn build_scene(renderer: &Renderer) -> Scene {
    let mut world = World::new(renderer);
    world.set_environment(LightingEnv::default());
    let ground = world
        .add_model(
            LodModel::builder(Arc::new(shapes::plane(15.0, material([0.35, 0.4, 0.3])))).build(),
        )
        .expect("ground");
    world.spawn(ground, Mat4::IDENTITY);

    let wall_model = world
        .add_model(
            LodModel::builder(Arc::new(shapes::cuboid(
                [6.0, 2.5, 0.2],
                material([0.8, 0.75, 0.7]),
            )))
            .build(),
        )
        .expect("wall");
    let wall = world.spawn(wall_model, Mat4::from_translation(WALL_CENTRE));

    let model = world
        .add_model(grid_model(ImpostorSource::BakeCached {
            dir: CACHE_DIR.into(),
            settings: BakeSettings::compact(),
        }))
        .expect("grid model");
    if let Some(reason) = world.impostor_fallback(model) {
        panic!("grid impostor could not be baked: {reason}");
    }
    let half = (GRID_SIDE as f32 - 1.0) * 0.5;
    let mut instances = 0;
    for j in 0..GRID_SIDE {
        for i in 0..GRID_SIDE {
            let x = (i as f32 - half) * GRID_SPACING;
            let z = (j as f32 - half) * GRID_SPACING + GRID_CENTRE.z;
            world.spawn(model, Mat4::from_translation(Vec3::new(x, 1.0, z)));
            instances += 1;
        }
    }
    Scene {
        world,
        wall,
        instances,
    }
}

/// Check 1 and 2 path: 40 frames walking forward 15 m while yawing 40 degrees
/// (staying in front of the wall), then 20 frames strafing right past the
/// wall's edge so that the hidden objects are revealed.
fn behind_wall_path() -> Vec<Pose> {
    let mut poses = Vec::with_capacity(PATH_FRAMES);
    let mut position = Vec3::new(0.0, 1.7, 12.0);
    let yaw_start = -90f32.to_radians();
    let yaw_end = -50f32.to_radians();
    for frame in 0..40 {
        let s = frame as f32 / 39.0;
        let yaw = yaw_start + (yaw_end - yaw_start) * s;
        poses.push(Pose {
            position,
            yaw,
            pitch: 0.0,
        });
        let forward = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        position += forward * (15.0 / 40.0);
    }
    let yaw = yaw_end;
    let right = Vec3::new(-yaw.sin(), 0.0, yaw.cos());
    for _ in 0..20 {
        poses.push(Pose {
            position,
            yaw,
            pitch: 0.0,
        });
        position += right * (10.0 / 20.0);
    }
    poses
}

fn apply(camera: &mut Camera, pose: Pose) {
    camera.position = pose.position;
    camera.set_look(pose.yaw, pose.pitch);
}

fn render_frame(world: &mut World, renderer: &mut Renderer, camera: &Camera) -> WorldStats {
    loop {
        match world.render(renderer, camera, DT, None, None) {
            Ok(stats) => return stats,
            Err(SurfaceError::Timeout) => continue,
            Err(e) => panic!("render failed: {e:?}"),
        }
    }
}

/// Mean absolute channel difference (0-255 units) and the fraction of pixels
/// whose largest channel difference exceeds [`DIFF_STRONG_LEVEL`].
fn compare(a: &[u8], b: &[u8]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "frame size");
    let mut sum = 0u64;
    let mut strong = 0u64;
    let mut pixels = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut worst = 0u8;
        for c in 0..3 {
            let d = pa[c].abs_diff(pb[c]);
            sum += u64::from(d);
            worst = worst.max(d);
        }
        if f64::from(worst) > DIFF_STRONG_LEVEL {
            strong += 1;
        }
        pixels += 1;
    }
    (
        sum as f64 / (pixels * 3) as f64,
        strong as f64 / pixels as f64,
    )
}

fn save(name: &str, rgba: &[u8]) {
    let path = Path::new(OUT_DIR).join(name);
    dream::png::write(&path, W, H, rgba).expect("write png");
    println!("[world-check] wrote {}", path.display());
}

/// Diagnostics for a failing pair: both frames and an amplified difference
/// image, so the region that differs can be located.
fn save_pair(prefix: &str, a: &[u8], b: &[u8]) {
    save(&format!("{prefix}-a.png"), a);
    save(&format!("{prefix}-b.png"), b);
    let mut diff = vec![255u8; a.len()];
    for (out, (pa, pb)) in diff
        .chunks_exact_mut(4)
        .zip(a.chunks_exact(4).zip(b.chunks_exact(4)))
    {
        let worst = (0..3).map(|c| pa[c].abs_diff(pb[c])).max().unwrap_or(0);
        let level = (u32::from(worst) * 8).min(255) as u8;
        out[0] = level;
        out[1] = 0;
        out[2] = 0;
    }
    save(&format!("{prefix}-diff.png"), &diff);
}

#[derive(Default)]
struct SessionOutput {
    frames: Vec<Vec<u8>>,
    stats: Vec<WorldStats>,
    world_ms: Vec<f32>,
    scene_ms: Vec<f32>,
    post_ms: Vec<f32>,
    shadow_ms: Vec<f32>,
}

/// Renders the behind-wall path once, with culling fully off or fully on.
fn run_path(event_loop: &ActiveEventLoop, culling: bool) -> SessionOutput {
    let occlusion = culling && experiment() != "frustum-only";
    let mut renderer = make_renderer(event_loop, occlusion);
    let mut scene = build_scene(&renderer);
    scene.world.settings_mut().frustum_culling = culling;
    scene.world.settings_mut().occlusion_culling = occlusion;
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    let mut out = SessionOutput::default();
    let mut latencies = Vec::new();
    for (index, pose) in behind_wall_path().into_iter().enumerate() {
        apply(&mut camera, pose);
        let start = Instant::now();
        let stats = render_frame(&mut scene.world, &mut renderer, &camera);
        out.world_ms.push(start.elapsed().as_secs_f32() * 1e3);
        out.scene_ms.push(renderer.scene_gpu_ms());
        out.shadow_ms.push(renderer.shadow_gpu_ms());
        out.post_ms.push(renderer.post_gpu_ms());
        out.stats.push(stats);
        out.frames.push(renderer.capture_rgba8().0);
        // Frame ids count from zero, so this frame's id is `index`.
        if let Some(grid) = renderer.occlusion() {
            latencies.push(index as u64 - grid.frame_id());
        }
    }
    if culling {
        let (min, max) = (
            latencies.iter().min().copied().unwrap_or(0),
            latencies.iter().max().copied().unwrap_or(0),
        );
        println!(
            "[world-check] occlusion grid latency over the path: {min}..={max} frames ({} samples)",
            latencies.len()
        );
    }
    assert_eq!(scene.instances, (GRID_SIDE * GRID_SIDE) as usize);
    out
}

fn check_culling_correctness(report: &mut Report, off: &SessionOutput, on: &SessionOutput) {
    let mut worst_mean = 0.0f64;
    let mut worst_strong = 0.0f64;
    for (index, (a, b)) in off.frames.iter().zip(&on.frames).enumerate() {
        let (mean, strong) = compare(a, b);
        worst_mean = worst_mean.max(mean);
        worst_strong = worst_strong.max(strong);
        println!(
            "[world-check] check1 frame {index:2}: mean-abs-diff={mean:.3}/255 strong-fraction={:.5}%",
            strong * 100.0
        );
        let ok = mean < DIFF_MEAN_LIMIT && strong < DIFF_STRONG_FRACTION;
        if !ok {
            save_pair(&format!("check1-frame{index:02}"), a, b);
        }
        report.check(
            ok,
            format!(
                "check1 frame {index} culling on/off mean {mean:.3} < {DIFF_MEAN_LIMIT} and strong {:.5}% < {:.2}%",
                strong * 100.0,
                DIFF_STRONG_FRACTION * 100.0
            ),
        );
    }
    println!(
        "[world-check] check1 worst over {} frames: mean-abs-diff={worst_mean:.3}/255 strong-fraction={:.5}%",
        off.frames.len(),
        worst_strong * 100.0
    );
}

fn check_effectiveness(report: &mut Report, on: &SessionOutput) {
    let total = (GRID_SIDE * GRID_SIDE) as f64;
    let mut frustum_share = 0.0;
    for (index, stats) in on.stats.iter().enumerate() {
        frustum_share += f64::from(stats.frustum_culled) / f64::from(stats.instances.max(1));
        if index % 10 == 0 {
            println!("[world-check] check2 stats frame {index:2}: {stats:?}");
        }
    }
    let mean_frustum = frustum_share / on.stats.len() as f64;
    report.check(
        mean_frustum >= 0.30,
        format!(
            "check2 frustum culling mean share {:.1}% >= 30%",
            mean_frustum * 100.0
        ),
    );
    // Occlusion: frames 5..40 are behind the wall; the first frames have no grid yet.
    let mut occ_share = 0.0;
    let mut occ_frames = 0;
    for stats in on.stats.iter().take(40).skip(5) {
        let survivors = f64::from(stats.instances - stats.frustum_culled);
        if survivors > 0.0 {
            occ_share += f64::from(stats.occlusion_culled) / survivors;
            occ_frames += 1;
        }
    }
    let mean_occ = occ_share / occ_frames.max(1) as f64;
    report.check(
        mean_occ >= 0.15,
        format!(
            "check2 occlusion culling share of frustum survivors behind the wall {:.1}% >= 15% ({total} instances)",
            mean_occ * 100.0
        ),
    );
}

/// Positions for the LOD dolly: from 5 m to 120 m from the grid centre, looking
/// at the centre from the +Z side, with the wall removed.
fn dolly_poses() -> Vec<(f32, Pose)> {
    // About one pixel of image motion per frame at 120 m (focal length near
    // 385 px for this field of view), so that the frame-to-frame difference
    // measures popping rather than camera travel.
    let steps = 1200;
    let (near, far) = (5.0f32, 120.0f32);
    (0..=steps)
        .map(|k| {
            let d = near * (far / near).powf(k as f32 / steps as f32);
            let pitch = -(6.0f32).atan2(d);
            let pose = Pose {
                position: GRID_CENTRE + Vec3::new(0.0, 6.0, d),
                yaw: -90f32.to_radians(),
                pitch,
            };
            (d, pose)
        })
        .collect()
}

fn check_lod_dolly(report: &mut Report, scene: &mut Scene, renderer: &mut Renderer) {
    scene.world.despawn(scene.wall);
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    let probes = [5.0f32, 15.0, 35.0, 70.0, 120.0];
    let mut previous: Option<Vec<u8>> = None;
    let mut worst_step = 0.0f64;
    let mut worst_pair: Option<(Vec<u8>, Vec<u8>, f32)> = None;
    let mut histogram: Vec<(f32, WorldStats, f64)> = Vec::new();
    let mut next_probe = 0;
    for (d, pose) in dolly_poses() {
        apply(&mut camera, pose);
        let stats = render_frame(&mut scene.world, renderer, &camera);
        let frame = renderer.capture_rgba8().0;
        if let Some(prev) = &previous {
            let (mean, _) = compare(prev, &frame);
            if mean > worst_step {
                worst_step = mean;
                worst_pair = Some((prev.clone(), frame.clone(), d));
            }
        }
        previous = Some(frame.clone());
        // The dolly distance grows monotonically: each probe is taken once, at
        // the first frame at or beyond it.
        if next_probe < probes.len() && d >= probes[next_probe] - 1e-3 {
            let visible: u32 = stats.mesh_draws.iter().sum::<u32>() + stats.impostor_draws;
            let impostor_share = f64::from(stats.impostor_draws) / f64::from(visible.max(1));
            histogram.push((d, stats, impostor_share));
            save(
                &format!("lod-{:03}m.png", probes[next_probe].round() as u32),
                &frame,
            );
            next_probe += 1;
        }
    }
    println!("[world-check] check3 LOD histogram (main-pass draws):");
    for (d, stats, share) in &histogram {
        println!(
            "[world-check]   d={d:6.1} m mesh L0={:4} L1={:4} L2={:4} L3+={:4} impostor={:4} crossfade={:3} frustum-culled={:4} occlusion-culled={:4} impostor-share={:.1}%",
            stats.mesh_draws[0],
            stats.mesh_draws[1],
            stats.mesh_draws[2],
            stats.mesh_draws[3],
            stats.impostor_draws,
            stats.crossfading,
            stats.frustum_culled,
            stats.occlusion_culled,
            share * 100.0
        );
    }
    // Monotone: the mesh-level-0 share falls and the impostor share rises.
    let mut monotone = true;
    for pair in histogram.windows(2) {
        let (a, b) = (&pair[0].1, &pair[1].1);
        let share0 = |s: &WorldStats| {
            let v = s.mesh_draws.iter().sum::<u32>() + s.impostor_draws;
            f64::from(s.mesh_draws[0]) / f64::from(v.max(1))
        };
        monotone &= share0(b) <= share0(a) + 0.02;
        monotone &= pair[1].2 + 0.02 >= pair[0].2;
        let _ = (a, b);
    }
    report.check(
        monotone,
        "check3 mesh LOD0 share falls and impostor share rises with distance".to_string(),
    );
    if let Some((_, far, share)) = histogram.last() {
        let _ = far;
        report.check(
            *share >= 0.9,
            format!(
                "check3 impostor share at 120 m {:.1}% >= 90%",
                share * 100.0
            ),
        );
    }
    if let Some((a, b, d)) = &worst_pair {
        println!("[world-check] check3 worst step ends at d={d:.2} m");
        save_pair("dolly-worst", a, b);
    }
    println!(
        "[world-check] check3 worst frame-to-frame mean-abs-diff during the dolly: {worst_step:.3}/255"
    );
    report.check(
        worst_step < DOLLY_MEAN_LIMIT,
        format!("check3 dolly frame-to-frame {worst_step:.3} < {DOLLY_MEAN_LIMIT}/255"),
    );
}

fn check_cache(report: &mut Report, event_loop: &ActiveEventLoop) {
    let mut renderer = make_renderer(event_loop, false);
    let mut world = World::new(&renderer);
    world.set_environment(LightingEnv::default());
    let source = || ImpostorSource::BakeCached {
        dir: CACHE_DIR.into(),
        settings: BakeSettings::compact(),
    };
    let first = world.add_model(grid_model(source())).expect("first add");
    let after_first = world.impostor_cache_counts();
    let second = world.add_model(grid_model(source())).expect("second add");
    let after_second = world.impostor_cache_counts();
    println!(
        "[world-check] check4 bakes/loads after first add {after_first:?}, after second add {after_second:?}"
    );
    report.check(
        after_second.0 == after_first.0 && after_second.1 == after_first.1 + 1,
        "check4 second BakeCached add loads from disk without baking".to_string(),
    );

    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    apply(
        &mut camera,
        Pose {
            position: Vec3::new(0.0, 2.0, 9.0),
            yaw: -90f32.to_radians(),
            pitch: -0.1,
        },
    );
    let place = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
    let mut render_with = |model: ModelHandle, renderer: &mut Renderer| {
        renderer.reset_history();
        let instance = world.spawn(model, place);
        render_frame(&mut world, renderer, &camera);
        let frame = renderer.capture_rgba8().0;
        world.despawn(instance);
        frame
    };
    let baked = render_with(first, &mut renderer);
    let loaded = render_with(second, &mut renderer);
    let (mean, _) = compare(&baked, &loaded);
    save("cache-first.png", &baked);
    report.check(
        mean < CACHE_MEAN_LIMIT,
        format!("check4 cached render mean-abs-diff {mean:.3} < {CACHE_MEAN_LIMIT}/255"),
    );
}

fn check_timing(off: &SessionOutput, on: &SessionOutput) {
    let window = 10..40;
    let avg = |values: &[f32]| {
        let slice = &values[window.clone()];
        slice.iter().sum::<f32>() / slice.len() as f32
    };
    let avg_cull = |out: &SessionOutput| {
        let slice = &out.stats[window.clone()];
        slice.iter().map(|s| s.cull_us).sum::<f32>() / slice.len() as f32
    };
    println!(
        "[world-check] check5 behind-wall frames 10..40 (not asserted): culling off: cull={:.0} us world-cpu={:.2} ms shadow-gpu={:.2} ms scene-gpu={:.2} ms post-gpu={:.2} ms",
        avg_cull(off),
        avg(&off.world_ms),
        avg(&off.shadow_ms),
        avg(&off.scene_ms),
        avg(&off.post_ms)
    );
    println!(
        "[world-check] check5 behind-wall frames 10..40 (not asserted): culling on:  cull={:.0} us world-cpu={:.2} ms shadow-gpu={:.2} ms scene-gpu={:.2} ms post-gpu={:.2} ms",
        avg_cull(on),
        avg(&on.world_ms),
        avg(&on.shadow_ms),
        avg(&on.scene_ms),
        avg(&on.post_ms)
    );
}

/// Check 6 limits: the instanced and per-instance draws of the same path must
/// agree in colour and in motion vectors.
const INSTANCING_MEAN_LIMIT: f64 = 0.3;
const INSTANCING_VELOCITY_LIMIT: f64 = 1e-4;

/// One behind-wall path rendered with instancing on or off.
struct InstancingRun {
    rgba: Vec<u8>,
    velocity: Vec<u8>,
    last: WorldStats,
    main_calls: u64,
    shadow_calls: u64,
    world_ms: f32,
}

fn run_instancing(event_loop: &ActiveEventLoop, instancing: bool) -> InstancingRun {
    let mut renderer = make_renderer(event_loop, true);
    let mut scene = build_scene(&renderer);
    scene.world.settings_mut().instancing = instancing;
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    let (mut main_calls, mut shadow_calls, mut world_ms, mut frames) = (0u64, 0u64, 0f32, 0u32);
    let mut last = WorldStats::default();
    for pose in behind_wall_path() {
        apply(&mut camera, pose);
        let start = Instant::now();
        last = render_frame(&mut scene.world, &mut renderer, &camera);
        world_ms += start.elapsed().as_secs_f32() * 1e3;
        frames += 1;
        main_calls += u64::from(last.draw_calls);
        shadow_calls += u64::from(last.shadow_draw_calls);
    }
    // The last frame is the fixed view that is compared.
    let (rgba, _, _) = renderer.capture_rgba8();
    let (velocity, _, _) = renderer.capture_velocity_rg16f();
    InstancingRun {
        rgba,
        velocity,
        last,
        main_calls,
        shadow_calls,
        world_ms: world_ms / frames as f32,
    }
}

/// Decode one IEEE 754 half-precision value.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    };
    sign * magnitude
}

/// Mean absolute value per component of an `rg16f` velocity capture.
fn velocity_mean_abs(a: &[u8]) -> f64 {
    let sum: f64 = a
        .chunks_exact(2)
        .map(|p| f64::from(half_to_f32(u16::from_le_bytes([p[0], p[1]])).abs()))
        .sum();
    sum / (a.len() / 2).max(1) as f64
}

/// Mean absolute difference per component of two `rg16f` velocity captures.
fn velocity_mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "velocity size");
    let mut sum = 0f64;
    let mut count = 0u64;
    for (pa, pb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let va = half_to_f32(u16::from_le_bytes([pa[0], pa[1]]));
        let vb = half_to_f32(u16::from_le_bytes([pb[0], pb[1]]));
        sum += f64::from((va - vb).abs());
        count += 1;
    }
    sum / count.max(1) as f64
}

fn check_instancing(report: &mut Report, event_loop: &ActiveEventLoop) {
    let off = run_instancing(event_loop, false);
    let on = run_instancing(event_loop, true);
    let frames = behind_wall_path().len() as f64;
    let (mean, _) = compare(&off.rgba, &on.rgba);
    let velocity = velocity_mean_abs_diff(&off.velocity, &on.velocity);
    if mean >= INSTANCING_MEAN_LIMIT {
        save_pair("instancing", &off.rgba, &on.rgba);
    }
    report.check(
        mean < INSTANCING_MEAN_LIMIT,
        format!(
            "check6 instancing colour: mean-abs-diff={mean:.4}/255 (limit {INSTANCING_MEAN_LIMIT})"
        ),
    );
    report.check(
        velocity < INSTANCING_VELOCITY_LIMIT,
        format!(
            "check6 instancing velocity: mean-abs-diff={velocity:.2e} (limit {INSTANCING_VELOCITY_LIMIT:.0e})"
        ),
    );
    let motion = velocity_mean_abs(&on.velocity);
    report.check(
        motion > 1e-4,
        format!("check6 motion vectors are non-trivial: mean |v| = {motion:.2e}"),
    );
    report.check(
        off.main_calls > 0 && on.main_calls < off.main_calls && on.shadow_calls < off.shadow_calls,
        format!(
            "check6 instancing lowers draw calls: main {} -> {}, shadow {} -> {} (path total)",
            off.main_calls, on.main_calls, off.shadow_calls, on.shadow_calls
        ),
    );
    println!(
        "[world-check] check6 timing over {frames} frames (not asserted): instancing off: main draw calls/frame {:.1}, shadow {:.1}, World::render cpu {:.3} ms",
        off.main_calls as f64 / frames,
        off.shadow_calls as f64 / frames,
        off.world_ms
    );
    println!(
        "[world-check] check6 timing over {frames} frames (not asserted): instancing on:  main draw calls/frame {:.1}, shadow {:.1}, World::render cpu {:.3} ms",
        on.main_calls as f64 / frames,
        on.shadow_calls as f64 / frames,
        on.world_ms
    );
    println!(
        "[world-check] check6 last frame: mesh draws per level {:?}, impostor draws {}, instances {}",
        on.last.mesh_draws, on.last.impostor_draws, on.last.instances
    );
}

/// Check 7: one sphere drawn as its mesh and as its impostor from the same
/// camera, sun and settings. The thresholds are pinned so the level cannot
/// change: coverage is far above both for the mesh and far below both for the
/// impostor.
/// Near and far: far impostors shade through the integrated path.
const BRIGHTNESS_DISTANCES: [f32; 3] = [15.0, 60.0, 120.0];
/// Keeps the finest mesh level (LOD 0, the full sphere) at that distance.
const BRIGHTNESS_LOD_BIAS: f32 = 1000.0;
const BRIGHTNESS_MEAN_LIMIT: f64 = 3.0;
/// Per-pixel channel difference above which a pixel counts as covered by the
/// sphere (against the same frame with no sphere).
const BRIGHTNESS_COVER_LEVEL: u8 = 8;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Forced {
    Mesh,
    Impostor,
}

fn forced_policy(forced: Forced) -> LodPolicy {
    let (impostor_below, mesh_above) = match forced {
        Forced::Mesh => (1e-9, 2e-9),
        Forced::Impostor => (1e9, 2e9),
    };
    LodPolicy {
        impostor_below,
        mesh_above,
        fade: 0.6,
        max_distance: 4096.0,
    }
}

struct BrightnessRun {
    rgba: Vec<u8>,
    last: WorldStats,
}

/// Renders the fixed view after warming TAA history. `None` leaves the sphere
/// out, which gives the background the sphere's pixels are compared against.
fn run_brightness(
    event_loop: &ActiveEventLoop,
    forced: Option<Forced>,
    distance: f32,
) -> BrightnessRun {
    let mut renderer = make_renderer(event_loop, true);
    let mut world = World::new(&renderer);
    world.set_environment(LightingEnv::default());
    world.settings_mut().lod_bias = BRIGHTNESS_LOD_BIAS;
    let ground = world
        .add_model(
            LodModel::builder(Arc::new(shapes::plane(15.0, material([0.35, 0.4, 0.3])))).build(),
        )
        .expect("ground");
    world.spawn(ground, Mat4::IDENTITY);
    if let Some(forced) = forced {
        let model = world
            .add_model(sphere_model(
                ImpostorSource::BakeCached {
                    dir: CACHE_DIR.into(),
                    settings: BakeSettings::compact(),
                },
                forced_policy(forced),
            ))
            .expect("sphere model");
        if let Some(reason) = world.impostor_fallback(model) {
            panic!("sphere impostor could not be baked: {reason}");
        }
        world.spawn(
            model,
            Mat4::from_translation(Vec3::new(0.0, 1.0, -distance)),
        );
    }
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    apply(
        &mut camera,
        Pose {
            position: Vec3::new(0.0, 1.0, 0.0),
            yaw: -90f32.to_radians(),
            pitch: 0.0,
        },
    );
    let mut last = WorldStats::default();
    for _ in 0..30 {
        last = render_frame(&mut world, &mut renderer, &camera);
    }
    let (rgba, _, _) = renderer.capture_rgba8();
    BrightnessRun { rgba, last }
}

/// Difference of two representations over the pixels both of them cover.
struct BrightnessDiff {
    pixels: u64,
    mean_abs: f64,
    /// Mean of (impostor - mesh) per channel, 0-255 units.
    mean_signed: [f64; 3],
    mesh_mean: [f64; 3],
    impostor_mean: [f64; 3],
    /// Pixels covered by exactly one of the two, as a fraction of the union.
    mismatch_fraction: f64,
}

fn brightness_diff(mesh: &[u8], impostor: &[u8], background: &[u8]) -> BrightnessDiff {
    let covers = |a: &[u8], b: &[u8]| (0..3).any(|c| a[c].abs_diff(b[c]) > BRIGHTNESS_COVER_LEVEL);
    let (mut pixels, mut abs_sum, mut mismatch, mut union) = (0u64, 0u64, 0u64, 0u64);
    let mut signed = [0f64; 3];
    let mut mesh_sum = [0f64; 3];
    let mut imp_sum = [0f64; 3];
    for ((pm, pi), pb) in mesh
        .chunks_exact(4)
        .zip(impostor.chunks_exact(4))
        .zip(background.chunks_exact(4))
    {
        let in_mesh = covers(pm, pb);
        let in_impostor = covers(pi, pb);
        if in_mesh || in_impostor {
            union += 1;
        }
        if in_mesh != in_impostor {
            mismatch += 1;
        }
        if !(in_mesh && in_impostor) {
            continue;
        }
        pixels += 1;
        for c in 0..3 {
            abs_sum += u64::from(pm[c].abs_diff(pi[c]));
            signed[c] += f64::from(pi[c]) - f64::from(pm[c]);
            mesh_sum[c] += f64::from(pm[c]);
            imp_sum[c] += f64::from(pi[c]);
        }
    }
    let n = pixels.max(1) as f64;
    BrightnessDiff {
        pixels,
        mean_abs: abs_sum as f64 / (n * 3.0),
        mean_signed: signed.map(|s| s / n),
        mesh_mean: mesh_sum.map(|s| s / n),
        impostor_mean: imp_sum.map(|s| s / n),
        mismatch_fraction: mismatch as f64 / union.max(1) as f64,
    }
}

fn check_brightness(report: &mut Report, event_loop: &ActiveEventLoop) {
    let background = run_brightness(event_loop, None, BRIGHTNESS_DISTANCES[0]);
    for distance in BRIGHTNESS_DISTANCES {
        let mesh = run_brightness(event_loop, Some(Forced::Mesh), distance);
        let impostor = run_brightness(event_loop, Some(Forced::Impostor), distance);
        // The ground plane is one main-pass mesh draw in every run (LOD 0).
        report.check(
            mesh.last.mesh_draws == [2, 0, 0, 0] && mesh.last.impostor_draws == 0,
            format!(
                "check7 {distance} m forced mesh draws the LOD 0 sphere (plus the ground): mesh draws {:?}, impostor draws {}",
                mesh.last.mesh_draws, mesh.last.impostor_draws
            ),
        );
        report.check(
            impostor.last.mesh_draws == [1, 0, 0, 0] && impostor.last.impostor_draws == 1,
            format!(
                "check7 {distance} m forced impostor draws only the impostor (plus the ground): mesh draws {:?}, impostor draws {}",
                impostor.last.mesh_draws, impostor.last.impostor_draws
            ),
        );
        let d = brightness_diff(&mesh.rgba, &impostor.rgba, &background.rgba);
        println!(
            "[world-check] check7 {distance} m: sphere covers {} px in both; mesh mean rgb {:.1?}, impostor mean rgb {:.1?}, signed (impostor - mesh) {:.2?}; coverage mismatch {:.2}% of the union",
            d.pixels,
            d.mesh_mean,
            d.impostor_mean,
            d.mean_signed,
            100.0 * d.mismatch_fraction
        );
        if d.mean_abs >= BRIGHTNESS_MEAN_LIMIT {
            save_pair(
                &format!("brightness-{distance}m"),
                &mesh.rgba,
                &impostor.rgba,
            );
        }
        report.check(
            d.pixels > 0 && d.mean_abs < BRIGHTNESS_MEAN_LIMIT,
            format!(
                "check7 {distance} m impostor vs mesh brightness: mean-abs RGB diff {:.2}/255 over {} px (limit {BRIGHTNESS_MEAN_LIMIT})",
                d.mean_abs, d.pixels
            ),
        );
    }
}

fn run(event_loop: &ActiveEventLoop) -> bool {
    std::fs::create_dir_all(CACHE_DIR).expect("create world-check output dir");
    let mut report = Report::default();

    let off = run_path(event_loop, false);
    let on = run_path(event_loop, true);
    save("first-frame.png", &on.frames[0]);
    save("behind-wall.png", &on.frames[25]);
    save("revealed.png", &on.frames[PATH_FRAMES - 1]);

    check_culling_correctness(&mut report, &off, &on);
    check_effectiveness(&mut report, &on);
    check_timing(&off, &on);
    check_instancing(&mut report, event_loop);
    check_brightness(&mut report, event_loop);

    // Check 3 runs on a fresh culled world with the wall removed.
    let mut renderer = make_renderer(event_loop, true);
    let mut scene = build_scene(&renderer);
    check_lod_dolly(&mut report, &mut scene, &mut renderer);
    drop(scene);
    drop(renderer);

    check_cache(&mut report, event_loop);

    if report.failures.is_empty() {
        println!("[world-check] all checks passed");
        true
    } else {
        println!("[world-check] {} check(s) failed", report.failures.len());
        false
    }
}

struct App {
    finished: bool,
    passed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.passed = run(event_loop);
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: winit::event::WindowEvent) {}
}

fn main() {
    let mut app = App {
        finished: false,
        passed: false,
    };
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut app)
        .expect("world check");
    if !app.passed {
        std::process::exit(1);
    }
}

//! World-level hardware-ray checks. Off-screen geometry must contribute to
//! reflections and diffuse bounce; AO is compared against the full CPU scene.

use super::reference::{Scene, Shape, View, ao_visibility, scene_items};
use super::{EYE, H, OUT_DIR, TARGET, W, interior, material, primary_at, save};
use dream::camera::Camera;
use dream::gfx::Renderer;
use dream::model::{Material, Model, shapes};
use dream::render::{AoSettings, GiSettings, IndirectQuality, ReflectionSettings, RenderSettings};
use dream::settings::AaMode;
use dream::world::{LodModel, World};
use glam::{Mat4, Vec3};
use std::path::Path;
use std::sync::Arc;

fn camera(eye: Vec3, target: Vec3) -> Camera {
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    camera.position = eye;
    let forward = (target - eye).normalize();
    camera.set_look(forward.z.atan2(forward.x), forward.y.asin());
    camera
}

fn add(world: &mut World, model: Model, position: Vec3) -> dream::world::InstanceHandle {
    let model = world
        .add_model(LodModel::builder(Arc::new(model)).build())
        .expect("RT check model");
    world.spawn(model, Mat4::from_translation(position))
}

fn settle(renderer: &mut Renderer, world: &mut World, camera: &Camera) {
    renderer.reset_history();
    for _ in 0..96 {
        world
            .render(renderer, camera, 1.0 / 60.0, None, None)
            .expect("RT check frame");
    }
}

/// Mean RGBA of a central patch (Reinhard radiance, visibility/confidence alpha).
fn patch(dump: &(Vec<u8>, u32, u32)) -> [f64; 4] {
    let (bytes, w, h) = dump;
    assert!(*w > 0 && *h > 0 && bytes.len() == (w * h * 4) as usize);
    let mut sum = [0.0; 4];
    let mut count = 0;
    for y in (h / 2 - h / 32)..(h / 2 + h / 32) {
        for x in (w / 2 - w / 32)..(w / 2 + w / 32) {
            let i = ((y * w + x) * 4) as usize;
            for c in 0..4 {
                sum[c] += f64::from(bytes[i + c]);
            }
            count += 1;
        }
    }
    sum.map(|value| value / f64::from(count))
}

fn offscreen(renderer: &mut Renderer) -> bool {
    let mut world = World::new(renderer);
    add(
        &mut world,
        shapes::cuboid([3.0, 3.0, 0.05], material("RT mirror", [1.0; 4], 1.0, 0.0)),
        Vec3::new(0.0, 2.0, -4.0),
    );
    let emitter = add(
        &mut world,
        shapes::cuboid(
            [1.5, 1.5, 0.3],
            Material {
                base_color: [0.0, 0.0, 0.0, 1.0],
                emissive_factor: [3.0, 0.02, 0.02],
                ..Material::default()
            },
        ),
        Vec3::new(0.0, 2.0, 7.0),
    );
    let camera = camera(Vec3::new(0.0, 2.0, 3.0), Vec3::new(0.0, 2.0, -4.0));
    let mut settings = RenderSettings::off();
    settings.reflections = Some(ReflectionSettings::high());
    let mut gi = GiSettings::default();
    gi.radius = 20.0;
    settings.gi = Some(gi);
    settings.indirect_quality = IndirectQuality::high();
    renderer.set_aa_mode(AaMode::None);
    renderer.set_render_settings(settings.clone());
    settle(renderer, &mut world, &camera);
    let ssr = patch(&renderer.capture_reflection_rgba8());
    let ss_gi = patch(&renderer.capture_indirect_rgba8());

    settings.ray_tracing = Some(Default::default());
    renderer.set_render_settings(settings);
    settle(renderer, &mut world, &camera);
    let rt = patch(&renderer.capture_reflection_rgba8());
    let rt_gi = patch(&renderer.capture_indirect_rgba8());
    let composed = renderer.capture_rgba8().0;
    save(&Path::new(OUT_DIR).join("rt_offscreen.png"), &composed);
    let timings = renderer.pass_gpu_ms();
    println!("rt offscreen: SSR {ssr:?}, RT {rt:?}; diffuse SSR {ss_gi:?}, RT {rt_gi:?}");
    println!("rt offscreen: GPU pass ms {timings:?}");
    // Emission is strongly red. A fallback sky or a pass that never runs
    // cannot satisfy both its colour and its increase over screen-space rays.
    let reflected = rt[0] > 100.0 && rt[0] - rt[1] > 80.0 && rt[0] - ssr[0] > 60.0;
    let bounced = rt_gi[0] - rt_gi[1] > 2.0 && (rt_gi[0] - rt_gi[1]) - (ss_gi[0] - ss_gi[1]) > 2.0;

    renderer.clear_ray_tracing_scene();
    settle(renderer, &mut world, &camera);
    let rebuilt = patch(&renderer.capture_reflection_rgba8());
    let recovered = (rebuilt[0] - rt[0]).abs() < 2.0 && rebuilt[0] - rebuilt[1] > 80.0;
    println!(
        "rt scene reset: World registered again, reflection {rebuilt:?}, recovered={recovered}"
    );

    world.despawn(emitter);
    settle(renderer, &mut world, &camera);
    let removed = patch(&renderer.capture_reflection_rgba8());
    let cleared = rt[0] - removed[0] > 60.0;
    println!(
        "rt offscreen: reflection={reflected}, GI={bounced}, removed {removed:?}, cleared={cleared}"
    );

    // A broad slab entirely behind the camera covers much of the mirror's
    // forward-facing hemisphere. Screen-space AO sees only the mirror itself.
    add(
        &mut world,
        shapes::cuboid([8.0, 8.0, 0.3], Material::default()),
        Vec3::new(0.0, 2.0, 4.5),
    );
    let mut ao_settings = RenderSettings::off();
    let mut ao = AoSettings::default();
    ao.radius = 20.0;
    ao_settings.ao = Some(ao);
    ao_settings.indirect_quality = IndirectQuality::high();
    renderer.set_render_settings(ao_settings.clone());
    settle(renderer, &mut world, &camera);
    let ss_visibility = patch(&renderer.capture_indirect_rgba8())[3] / 255.0;
    ao_settings.ray_tracing = Some(Default::default());
    renderer.set_render_settings(ao_settings);
    settle(renderer, &mut world, &camera);
    let rt_visibility = patch(&renderer.capture_indirect_rgba8())[3] / 255.0;
    let occluded = ss_visibility - rt_visibility > 0.15;
    println!(
        "rt offscreen: AO visibility screen-space {ss_visibility:.3}, RT {rt_visibility:.3}, hidden slab occludes={occluded}"
    );
    reflected && bounced && recovered && cleared && occluded
}

fn ambient_reference(renderer: &mut Renderer, top_first: bool) -> bool {
    let mut world = World::new(renderer);
    for item in scene_items() {
        let mat = material(item.name, item.base_color, item.metallic, item.roughness);
        let model = match item.shape {
            Shape::Plane { half_x, half_z } => shapes::plane_rect(half_x, half_z, mat),
            Shape::Box { half } => shapes::cuboid(half, mat),
            Shape::Sphere { radius } => shapes::uv_sphere(radius, 64, 32, mat),
        };
        add(&mut world, model, Vec3::from_array(item.centre));
    }
    let mut settings = RenderSettings::off();
    settings.ao = Some(AoSettings::default());
    settings.indirect_quality = IndirectQuality::high();
    settings.ray_tracing = Some(Default::default());
    renderer.set_render_settings(settings);
    let camera = camera(Vec3::from_array(EYE), Vec3::from_array(TARGET));
    settle(renderer, &mut world, &camera);
    let (rgba, width, height) = renderer.capture_indirect_rgba8();
    if width == 0 || height == 0 {
        eprintln!("rt AO: missing indirect output");
        return false;
    }
    let view = View::new(
        camera.view_projection(),
        camera.position.as_dvec3(),
        (W, H),
        camera.clip_planes(),
        top_first,
    );
    let scene = Scene::new();
    let mut error = 0.0;
    let mut signed = 0.0;
    let mut count = 0;
    for y in (0..height).step_by(12) {
        for x in (0..width).step_by(12) {
            let Some(hit) = primary_at(&scene, &view, (width, height), i64::from(x), i64::from(y))
            else {
                continue;
            };
            if !interior(
                &scene,
                &view,
                (width, height),
                &hit,
                i64::from(x),
                i64::from(y),
            ) {
                continue;
            }
            let reference = ao_visibility(
                &scene,
                hit.position,
                hit.normal,
                f64::from(AoSettings::default().radius),
                None,
            );
            let gpu = f64::from(rgba[((y * width + x) * 4) as usize + 3]) / 255.0;
            error += (gpu - reference).abs();
            signed += gpu - reference;
            count += 1;
        }
    }
    let mae = error / f64::from(count.max(1));
    let bias = signed / f64::from(count.max(1));
    println!(
        "rt AO: {count} full-scene CPU probes, MAE {mae:.4}, bias {bias:+.4} (limits 0.08 / 0.04)"
    );
    count >= 500 && mae < 0.08 && bias.abs() < 0.04
}

pub fn run(renderer: &mut Renderer, top_first: bool) -> bool {
    if !renderer.context().ray_tracing_supported() {
        println!("rt checks: skipped (adapter lacks the complete ray-query feature set)");
        return true;
    }
    let offscreen = offscreen(renderer);
    let ambient = ambient_reference(renderer, top_first);
    renderer.set_render_settings(RenderSettings::off());
    offscreen && ambient
}

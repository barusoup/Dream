//! Showcase stills of the renderer's recent systems: ambient occlusion,
//! one-bounce global illumination, screen-space reflections, the atmosphere,
//! volumetric fog with shadowed light shafts, and volumetric clouds.
//!
//! ```text
//! cargo run --release --example showcase              # every shot
//! cargo run --release --example showcase -- --only hero
//! ```
//!
//! The window is hidden; each shot renders a settling sequence and writes
//! `target/showcase/<name>.png`.

use dream::camera::Camera;
use dream::gfx::{DEPTH_FORMAT, GpuContext, Renderer, SCENE_COLOR_FORMATS, SceneFrame};
use dream::lighting::{Lighting, LightingEnv};
use dream::model::render::{ModelGpu, ModelRenderer};
use dream::model::shapes;
use dream::model::{Material, Model, ModelInstance};
use dream::render::{
    AtmosphereSettings, CloudQuality, CloudSettings, FogResolution, FogSettings, RenderSettings,
    Tonemap,
};
use dream::settings::{AaMode, Settings, UpscalePreset};
use glam::{Mat4, Vec3};
use std::path::Path;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

const W: u32 = 1920;
const H: u32 = 1080;
const OUT_DIR: &str = "target/showcase";

/// A scene object: its GPU mesh plus the instance that carries its transform.
struct Object {
    gpu: ModelGpu,
    instance: ModelInstance,
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

fn object(ctx: &Arc<GpuContext>, model: Model, transform: Mat4) -> Object {
    let model = Arc::new(model);
    let gpu = ModelGpu::upload(ctx, &model);
    let mut instance = ModelInstance::new(model);
    instance.set_transform(transform);
    instance.update(0.0);
    Object { gpu, instance }
}

fn pbr(rgb: [f32; 3], metallic: f32, roughness: f32) -> Material {
    material("pbr", [rgb[0], rgb[1], rgb[2], 1.0], metallic, roughness)
}

/// A colonnade around a reflecting pool, metallic spheres for the reflection
/// chain, a bronze statue, and a ring of distant trees for depth.
fn build_scene(ctx: &Arc<GpuContext>) -> Vec<Object> {
    let mut objects = Vec::new();

    // Ground and plaza.
    objects.push(object(
        ctx,
        shapes::plane(160.0, pbr([0.16, 0.19, 0.14], 0.0, 0.95)),
        Mat4::IDENTITY,
    ));
    objects.push(object(
        ctx,
        shapes::plane_rect(30.0, 30.0, pbr([0.52, 0.50, 0.45], 0.0, 0.8)),
        Mat4::from_translation(Vec3::new(0.0, 0.015, 0.0)),
    ));

    // The pool: a smooth water surface for the reflection pass.
    objects.push(object(
        ctx,
        shapes::plane_rect(7.0, 7.0, pbr([0.10, 0.20, 0.24], 0.0, 0.04)),
        Mat4::from_translation(Vec3::new(0.0, 0.035, 0.0)),
    ));

    // Statue: a stone sphere on a plinth at the centre of the pool.
    objects.push(object(
        ctx,
        shapes::cuboid([1.9, 0.7, 1.9], pbr([0.46, 0.44, 0.40], 0.0, 0.75)),
        Mat4::from_translation(Vec3::new(0.0, 0.7, 0.0)),
    ));
    objects.push(object(
        ctx,
        shapes::uv_sphere(1.7, 48, 24, pbr([0.72, 0.70, 0.64], 0.0, 0.75)),
        Mat4::from_translation(Vec3::new(0.0, 2.9, 0.0)),
    ));

    // A lacquered red wall and two marble benches against it: warm light bleeds
    // from the wall onto the white stone, which is what the GI pass shows.
    let red = material("red lacquer", [0.55, 0.05, 0.035, 1.0], 0.0, 0.45);
    objects.push(object(
        ctx,
        shapes::cuboid([0.45, 4.5, 9.0], red),
        Mat4::from_translation(Vec3::new(-18.0, 4.5, 5.0)),
    ));
    for z in [1.5f32, 8.5] {
        objects.push(object(
            ctx,
            shapes::cuboid([3.0, 0.45, 0.9], pbr([0.80, 0.78, 0.72], 0.0, 0.6)),
            Mat4::from_translation(Vec3::new(-14.4, 0.45, z)),
        ));
    }

    // Colonnade: columns, capitals, and architrave segments carrying the ring.
    let columns = 16u32;
    let radius = 15.0f32;
    let stone = [0.60, 0.57, 0.50];
    for i in 0..columns {
        let theta = i as f32 / columns as f32 * std::f32::consts::TAU;
        let (sin, cos) = theta.sin_cos();
        let at = Vec3::new(radius * cos, 0.0, radius * sin);
        objects.push(object(
            ctx,
            shapes::cuboid([0.62, 5.2, 0.62], pbr(stone, 0.0, 0.72)),
            Mat4::from_translation(at + Vec3::Y * 5.2),
        ));
        objects.push(object(
            ctx,
            shapes::cuboid([1.0, 0.3, 1.0], pbr([0.50, 0.47, 0.41], 0.0, 0.78)),
            Mat4::from_translation(at + Vec3::Y * 10.7),
        ));
        // A lintel spanning to the next column, rotated to the tangent.
        let mid_theta = (i as f32 + 0.5) / columns as f32 * std::f32::consts::TAU;
        let span = 2.0 * radius * (std::f32::consts::PI / columns as f32).sin();
        let mid = Vec3::new(radius * mid_theta.cos(), 11.35, radius * mid_theta.sin());
        let rotation = Mat4::from_rotation_y(-(mid_theta + std::f32::consts::FRAC_PI_2));
        objects.push(object(
            ctx,
            shapes::cuboid([span * 0.52, 0.35, 0.42], pbr(stone, 0.0, 0.75)),
            Mat4::from_translation(mid) * rotation,
        ));
    }

    // Distant trees: a ring of dark green spheres of varied size.
    let tree = shapes::uv_sphere(1.0, 10, 6, pbr([0.12, 0.10, 0.075], 0.0, 1.0));
    let tree_model = Arc::new(tree);
    for i in 0..90u32 {
        let angle = i as f32 * 2.399_963; // golden angle spreads them evenly
        let spread = 60.0 + ((i as f32 * 7.13).sin() * 0.5 + 0.5) * 50.0;
        let height = 5.0 + ((i as f32 * 3.71).cos() * 0.5 + 0.5) * 6.0;
        let at = Vec3::new(spread * angle.cos(), height * 0.9, spread * angle.sin());
        let model = tree_model.clone();
        let gpu = ModelGpu::upload(ctx, &model);
        let mut instance = ModelInstance::new(model);
        let width = 1.6 + ((i as f32 * 1.7).sin() * 0.5 + 0.5) * 1.4;
        instance.set_transform(
            Mat4::from_translation(at) * Mat4::from_scale(Vec3::new(width, height, width)),
        );
        instance.update(0.0);
        objects.push(Object { gpu, instance });
    }

    objects
}

/// One still: camera pose, sun, feature settings.
struct Shot {
    name: &'static str,
    eye: [f32; 3],
    target: [f32; 3],
    /// Sun azimuth and elevation, radians.
    sun: (f32, f32),
    sun_intensity: f32,
    /// Frames rendered before the capture, so TAA, GI and clouds settle.
    frames: usize,
    exposure: f32,
    atmosphere: fn() -> AtmosphereSettings,
    clouds: Option<fn() -> CloudSettings>,
    fog: Option<fn() -> FogSettings>,
}

fn sunset_fog() -> FogSettings {
    let mut fog = FogSettings::medium();
    fog.density = 0.018;
    fog.height_falloff = 0.09;
    fog.anisotropy = 0.80;
    fog.sun_scatter = 2.2;
    fog.ambient = 0.45;
    fog.noise = 0.5;
    fog.noise_scale = 16.0;
    fog.distance = 240.0;
    fog.resolution = FogResolution::High;
    fog
}

fn hazy_atmosphere() -> AtmosphereSettings {
    let mut atmosphere = AtmosphereSettings::hazy();
    atmosphere.mie_scale = 1.6;
    atmosphere.ground_albedo = [0.20, 0.19, 0.18];
    atmosphere
}

fn fair_clouds() -> CloudSettings {
    let mut clouds = CloudSettings::fair_weather();
    clouds.quality = CloudQuality::High;
    clouds.coverage = 0.42;
    clouds.base_altitude = 1800.0;
    clouds.thickness = 1800.0;
    clouds.scale = 7000.0;
    clouds.detail = 0.65;
    clouds.silver_lining = 0.45;
    clouds.ambient = 0.8;
    clouds
}

fn wispy_clouds() -> CloudSettings {
    let mut clouds = CloudSettings::wispy();
    clouds.quality = CloudQuality::High;
    clouds
}

fn shots() -> Vec<Shot> {
    vec![
        Shot {
            name: "hero",
            eye: [4.0, 2.8, 34.0],
            target: [0.0, 7.5, 0.0],
            sun: (std::f32::consts::PI * 1.5 + 0.18, 0.30),
            sun_intensity: 3.2,
            frames: 28,
            exposure: 0.62,
            atmosphere: hazy_atmosphere,
            clouds: Some(fair_clouds),
            fog: Some(sunset_fog),
        },
        Shot {
            name: "cloudscape",
            eye: [0.0, 6.0, 42.0],
            target: [0.0, 30.0, -10.0],
            sun: (std::f32::consts::PI * 1.15, 0.35),
            sun_intensity: 2.4,
            frames: 28,
            exposure: 0.72,
            atmosphere: AtmosphereSettings::earth,
            clouds: Some(fair_clouds),
            fog: Some(sunset_fog),
        },
        Shot {
            name: "detail",
            eye: [11.0, 2.6, 16.0],
            target: [-9.0, 2.8, 3.0],
            sun: (0.85, 0.52),
            sun_intensity: 3.0,
            frames: 28,
            exposure: 0.78,
            atmosphere: AtmosphereSettings::clear,
            clouds: Some(wispy_clouds),
            fog: Some(FogSettings::light_haze),
        },
        Shot {
            name: "shafts",
            eye: [7.0, 2.2, 27.0],
            target: [0.0, 5.0, -3.0],
            sun: (std::f32::consts::PI * 1.5 + 0.16, 0.20),
            sun_intensity: 3.2,
            frames: 28,
            exposure: 0.62,
            atmosphere: hazy_atmosphere,
            clouds: Some(fair_clouds),
            fog: Some(sunset_fog),
        },
    ]
}

fn render_shot(
    renderer: &mut Renderer,
    models: &mut ModelRenderer,
    lighting: &Lighting,
    objects: &mut [Object],
    shot: &Shot,
) {
    let mut settings = RenderSettings::ultra();
    settings.exposure = shot.exposure;
    settings.tonemap = Tonemap::Aces;
    settings.dither = true;
    settings.atmosphere = Some((shot.atmosphere)());
    settings.clouds = shot.clouds.map(|make| make());
    settings.fog = shot.fog.map(|make| make());
    // A lower sun means the fog's shadowed shafts need a wider reach.
    if let Some(fog) = settings.fog.as_mut() {
        fog.distance = fog.distance.max(180.0);
    }
    renderer.set_aa_mode(AaMode::Taa);
    renderer.set_render_settings(settings);
    renderer.reset_history();
    renderer.set_scene_time_scale(0.0);

    let (azimuth, elevation) = shot.sun;
    let eye = Vec3::from(shot.eye);
    let target = Vec3::from(shot.target);
    for frame in 0..shot.frames {
        renderer.set_scene_time(frame as f64 * (1.0 / 60.0));
        let mut camera = Camera::default();
        camera.set_aspect(W as f32 / H as f32);
        camera.position = eye;
        let forward = (target - eye).normalize();
        camera.set_look(forward.z.atan2(forward.x), forward.y.asin());
        let env = LightingEnv {
            cam_pos: eye.extend(0.0).to_array(),
            ..LightingEnv::with_sun(azimuth, elevation, shot.sun_intensity)
        };
        let scene = SceneFrame {
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
            .render_scene(&scene, |draw| {
                for object in objects.iter_mut() {
                    models.draw_pbr(draw, &mut object.gpu, &object.instance, lighting);
                }
                objects.len()
            })
            .expect("render scene frame");
    }
    let dir = Path::new(OUT_DIR);
    std::fs::create_dir_all(dir).expect("create showcase output dir");
    let path = dir.join(format!("{}.png", shot.name));
    renderer.save_png(&path).expect("save showcase still");
    println!("saved {}", path.display());
}

struct ShowcaseApp {
    only: Option<String>,
    finished: bool,
    passed: bool,
}

impl ApplicationHandler for ShowcaseApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.passed = run_shots(event_loop, self.only.as_deref());
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::window::WindowId,
        _: winit::event::WindowEvent,
    ) {
    }
}

fn run_shots(event_loop: &ActiveEventLoop, only: Option<&str>) -> bool {
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Dream Showcase")
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
    let ctx = renderer.context().clone();
    let lighting = Lighting::new(&ctx);
    let mut models = ModelRenderer::new(&ctx, &SCENE_COLOR_FORMATS, DEPTH_FORMAT);
    let mut objects = build_scene(&ctx);
    for shot in shots()
        .iter()
        .filter(|shot| only.is_none_or(|name| name == shot.name))
    {
        println!("rendering {}", shot.name);
        render_shot(&mut renderer, &mut models, &lighting, &mut objects, shot);
    }
    true
}

fn main() {
    let only = std::env::args()
        .skip_while(|arg| arg != "--only")
        .nth(1)
        .filter(|name| shots().iter().any(|shot| shot.name == name));
    let mut app = ShowcaseApp {
        only,
        finished: false,
        passed: false,
    };
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut app)
        .expect("showcase");
    if !app.passed {
        std::process::exit(1);
    }
}

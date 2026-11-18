//! A small 3D scene in a Dream app: the 3D counterpart of `examples/hello.rs`.
//!
//! ```text
//! cargo run --release --example game
//! cargo run --release --example game -- --frames 120   (exit after 120 frames)
//! cargo run --release --example game -- --rt           (optional hardware rays)
//! ```
//!
//! Copy this file to start a game. The pieces:
//!
//! - [`dream::ui::app::run`] owns the window, renderer, input and frame loop.
//!   The app implements [`UiApp`]: `view` builds the HUD, and `draw_scene`
//!   draws the 3D [`World`] with that HUD in the same submission.
//! - The [`World`] holds the ground, the shapes and a field of LOD spheres
//!   with cached impostors. Models are added once, instances are spawned and
//!   moved, and `World::render` does culling, LOD and every pass.
//! - [`RenderSettings`] and [`AaMode`] are the knobs. The keys below change
//!   them at run time.
//!
//! Keys: WASD, Space and Ctrl fly, Shift is faster, mouse looks; click captures the cursor, Escape releases it.
//! `1`..`8` toggle shadows, AO, GI, reflections, atmosphere, fog, clouds and
//! occlusion. `R` requests hardware ray tracing for enabled AO, GI and reflections
//! (with screen-space fallback). `P` cycles the presets, `M` the AA mode,
//! `T` pauses the sun. Presets reset the ray-tracing request to off.

use dream::camera::Camera;
use dream::gfx::{Renderer, SurfaceError};
use dream::impostor::{BakeSettings, LodPolicy};
use dream::lighting::LightingEnv;
use dream::model::{Material, Model, TexelFilter, TextureRef, Wrap, shapes};
use dream::render::RenderSettings;
use dream::settings::AaMode;
use dream::texture::DecodedTexture;
use dream::ui::Color;
use dream::ui::app::{AppContext, AppOptions, RunError, UiApp, run};
use dream::ui::draw::UiFrame;
use dream::ui::layout::{LayoutAlign, Style};
use dream::ui::tokens::TypeRole;
use dream::ui::view::Paint;
use dream::ui::widgets::Ui;
use dream::ui::widgets::adapters::Text;
use dream::world::{ImpostorSource, LodModel, World, WorldStats};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::keyboard::{KeyCode, KeyCode::*};

/// Where the impostor bake is cached between runs.
const CACHE_DIR: &str = "target/game-example";
/// The feature keys, in the order `1`..`8` name them.
const FEATURE_KEYS: [KeyCode; 8] = [
    Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8,
];
const FEATURE_NAMES: [&str; 8] = [
    "shadows",
    "ao",
    "gi",
    "reflections",
    "atmosphere",
    "fog",
    "clouds",
    "occlusion",
];
const PRESETS: [&str; 5] = ["off", "low", "medium", "high", "ultra"];
const FLY_KEYS: [KeyCode; 7] = [KeyW, KeyA, KeyS, KeyD, Space, ControlLeft, ShiftLeft];

struct Game {
    mouse_captured: bool,
    world: Option<World>,
    camera: Camera,
    settings: RenderSettings,
    preset: usize,
    aa: AaMode,
    sun_running: bool,
    sun_angle: f32,
    stats: WorldStats,
    fps: f32,
    last_frame: Option<Instant>,
    /// Keys held on the previous frame, for edge detection.
    was_down: HashSet<KeyCode>,
    frames: u32,
    /// Per-pass GPU times of every frame read back (for the median table at exit).
    pass_samples: Vec<Vec<(&'static str, f32)>>,
    max_frames: Option<u32>,
    /// `--capture <png>`: save the last frame there before exiting.
    capture: Option<String>,
    /// `--pos x,y,z` and `--look yaw,pitch`: the starting camera.
    start_pos: Option<Vec3>,
    start_look: Option<(f32, f32)>,
}

impl Game {
    fn new(max_frames: Option<u32>) -> Self {
        let preset = arg("--preset")
            .and_then(|name| PRESETS.iter().position(|p| *p == name))
            .unwrap_or(3);
        let floats = |name: &str| -> Vec<f32> {
            arg(name)
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_default()
        };
        let pos = floats("--pos");
        let look = floats("--look");
        let sun = arg("--sun").and_then(|v| v.parse::<f32>().ok());
        Self {
            capture: arg("--capture"),
            start_pos: (pos.len() == 3).then(|| Vec3::new(pos[0], pos[1], pos[2])),
            start_look: (look.len() == 2).then(|| (look[0], look[1])),
            mouse_captured: false,
            world: None,
            camera: Camera::default(),
            settings: {
                let mut settings = preset_settings(preset);
                // `--clouds off|low|medium|high` overrides the preset's clouds.
                match arg("--clouds").as_deref() {
                    Some("off") => settings.clouds = None,
                    Some(level) => {
                        let mut clouds = settings.clouds.clone().unwrap_or_default();
                        clouds.quality = match level {
                            "low" => dream::render::CloudQuality::Low,
                            "medium" => dream::render::CloudQuality::Medium,
                            _ => dream::render::CloudQuality::High,
                        };
                        settings.clouds = Some(clouds);
                    }
                    None => {}
                }
                settings.ray_tracing = std::env::args().any(|a| a == "--rt").then(Default::default);
                settings
            },
            preset,
            aa: match arg("--aa").as_deref() {
                Some("none") => AaMode::None,
                Some("fsr") => AaMode::Fsr,
                _ => AaMode::Taa,
            },
            sun_running: sun.is_none(),
            sun_angle: sun.unwrap_or(0.6),
            stats: WorldStats::default(),
            fps: 0.0,
            last_frame: None,
            was_down: HashSet::new(),
            frames: 0,
            pass_samples: Vec::new(),
            max_frames,
        }
    }

    /// True once per press of `code`, not while it is held.
    fn pressed(&mut self, cx: &AppContext<'_>, code: KeyCode) -> bool {
        let down = cx.key_held(code);
        let was = self.was_down.contains(&code);
        if down {
            self.was_down.insert(code);
        } else {
            self.was_down.remove(&code);
        }
        down && !was
    }
}

/// A PBR material with the given factors.
fn pbr(rgb: [f32; 3], metallic: f32, roughness: f32) -> Material {
    Material {
        base_color: [rgb[0], rgb[1], rgb[2], 1.0],
        metallic_factor: metallic,
        roughness_factor: roughness,
        ..Material::default()
    }
}

/// A sphere with a grid of LODs and a baked impostor for distant views.
///
/// `--tree mesh` drops the impostor and `--tree impostor` forces it wherever it
/// is allowed (thresholds pushed past any coverage). Both are measurement aids.
fn tree_model() -> LodModel {
    let variant = arg("--tree").unwrap_or_default();
    let mesh = |segments, rings| {
        Arc::new(shapes::uv_sphere(
            1.0,
            segments,
            rings,
            pbr([0.3, 0.55, 0.25], 0.0, 0.8),
        ))
    };
    let mut builder = LodModel::builder(mesh(40, 20))
        .lod(mesh(24, 12), 0.25)
        .lod(mesh(14, 7), 0.10);
    if variant != "mesh" {
        builder = builder.impostor(ImpostorSource::BakeCached {
            dir: CACHE_DIR.into(),
            settings: BakeSettings::compact(),
        });
    }
    let policy = if variant == "impostor" {
        LodPolicy {
            impostor_below: 1.0e4,
            mesh_above: 1.1e4,
            fade: 0.6,
            max_distance: 4096.0,
        }
    } else {
        LodPolicy {
            impostor_below: 0.04,
            mesh_above: 0.06,
            fade: 0.6,
            max_distance: 4096.0,
        }
    };
    builder.policy(policy).build()
}

/// Repeat-tiled `TextureRef` backed by procedural RGBA8 pixels.
fn tiled_texture(pixels: Vec<u8>, size: u32) -> TextureRef {
    TextureRef {
        texture: DecodedTexture {
            pixels,
            width: size,
            height: size,
        },
        tex_coord: 0,
        transform: None,
        wrap_s: Wrap::Repeat,
        wrap_t: Wrap::Repeat,
        mag_filter: TexelFilter::Linear,
        min_filter: TexelFilter::Linear,
    }
}

/// Procedural architectural slate-stone courtyard material (1m tiles with
/// subtle surface variation and beveled mortar seams).
fn courtyard_ground_material() -> Material {
    let n = 256u32;
    let half = n / 2;
    let mut albedo = vec![0u8; (n * n * 4) as usize];
    let mut normal = vec![0u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            // 2x2 sub-tiles per 1m repeat, alternating offset.
            let row = (y >= half) as u32;
            let sx = (x + row * (half / 2)) % n;
            let lx = sx % half;
            let ly = y % half;
            let tx = sx / half;
            let ty = y / half;
            let edge = lx.min(half - 1 - lx).min(ly.min(half - 1 - ly));
            let h = ((x.wrapping_mul(374_761_393))
                ^ (y.wrapping_mul(668_265_263))
                ^ (tx.wrapping_mul(1_274_126_177))
                ^ (ty.wrapping_mul(2_246_822_519)))
            .wrapping_mul(1_103_515_245);
            let grain = ((h >> 24) as f32 / 255.0 - 0.5) * 0.035;
            let tile_tint = (((tx ^ ty) & 1) as f32) * 0.025;
            let seam = if edge < 3 { 0.58 } else { 1.0 };
            let shade = (0.31 + tile_tint + grain) * seam;
            let c = (shade.clamp(0.0, 1.0) * 255.0) as u8;
            let idx = ((y * n + x) * 4) as usize;
            albedo[idx] = c;
            albedo[idx + 1] = ((c as f32 * 0.98) as u8).max(1);
            albedo[idx + 2] = ((c as f32 * 0.94) as u8).max(1);
            albedo[idx + 3] = 255;

            // Beveled normal along seams + subtle stone grain.
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            if lx < 4 {
                nx = -0.35 * (4 - lx) as f32 / 4.0;
            } else if lx + 4 >= half {
                nx = 0.35 * (lx + 5 - half) as f32 / 4.0;
            }
            if ly < 4 {
                ny = -0.35 * (4 - ly) as f32 / 4.0;
            } else if ly + 4 >= half {
                ny = 0.35 * (ly + 5 - half) as f32 / 4.0;
            }
            nx += grain * 1.4;
            ny -= grain * 1.4;
            let v = Vec3::new(nx, ny, 1.0).normalize();
            normal[idx] = ((v.x * 0.5 + 0.5) * 255.0) as u8;
            normal[idx + 1] = ((v.y * 0.5 + 0.5) * 255.0) as u8;
            normal[idx + 2] = ((v.z * 0.5 + 0.5) * 255.0) as u8;
            normal[idx + 3] = 255;
        }
    }
    Material {
        name: "courtyard_stone".into(),
        base_color: [1.0, 1.0, 1.0, 1.0],
        metallic_factor: 0.0,
        roughness_factor: 0.72,
        normal_scale: 0.8,
        base_color_map: Some(tiled_texture(albedo, n)),
        normal_texture: Some(tiled_texture(normal, n)),
        ..Material::default()
    }
}

/// Ground, shapes, a centered reflective basin, pedestals, monoliths and the LOD field.
fn build_world(renderer: &Renderer) -> World {
    let mut world = World::new(renderer);
    let add = |world: &mut World, model: Model| {
        world
            .add_model(LodModel::builder(Arc::new(model)).build())
            .expect("add model")
    };

    let ground = add(&mut world, shapes::plane(60.0, courtyard_ground_material()));
    world.spawn(ground, Mat4::IDENTITY);

    let basalt = pbr([0.14, 0.15, 0.17], 0.08, 0.52);

    // Centered reflective obsidian/chrome basin framed by dark stone curbing.
    let mirror = add(
        &mut world,
        shapes::plane_rect(5.6, 1.8, pbr([0.62, 0.70, 0.78], 1.0, 0.025)),
    );
    world.spawn(mirror, Mat4::from_translation(Vec3::new(0.0, 0.015, -3.6)));

    let curb_x = add(
        &mut world,
        shapes::cuboid([5.8, 0.08, 0.18], basalt.clone()),
    );
    world.spawn(curb_x, Mat4::from_translation(Vec3::new(0.0, 0.08, -1.72)));
    world.spawn(curb_x, Mat4::from_translation(Vec3::new(0.0, 0.08, -5.48)));
    let curb_z = add(
        &mut world,
        shapes::cuboid([0.18, 0.08, 1.98], basalt.clone()),
    );
    world.spawn(curb_z, Mat4::from_translation(Vec3::new(-5.7, 0.08, -3.6)));
    world.spawn(curb_z, Mat4::from_translation(Vec3::new(5.7, 0.08, -3.6)));

    // Framing architectural monolith pillars + lintel + warm emissive beacons.
    let pillar = add(
        &mut world,
        shapes::cuboid([0.55, 2.3, 0.55], basalt.clone()),
    );
    world.spawn(pillar, Mat4::from_translation(Vec3::new(-6.8, 2.3, -3.6)));
    world.spawn(pillar, Mat4::from_translation(Vec3::new(6.8, 2.3, -3.6)));
    let lintel = add(&mut world, shapes::cuboid([7.4, 0.35, 0.6], basalt.clone()));
    world.spawn(lintel, Mat4::from_translation(Vec3::new(0.0, 4.75, -3.6)));
    let beacon = Material {
        base_color: [1.0, 0.85, 0.6, 1.0],
        emissive_factor: [1.0, 0.62, 0.24],
        emissive_strength: 6.0,
        roughness_factor: 0.3,
        ..Material::default()
    };
    let strip = add(&mut world, shapes::cuboid([0.08, 1.6, 0.14], beacon));
    world.spawn(strip, Mat4::from_translation(Vec3::new(-6.22, 2.3, -3.6)));
    world.spawn(strip, Mat4::from_translation(Vec3::new(6.22, 2.3, -3.6)));

    // Row of 5 hero spheres on dark basalt pedestals (metallic & roughness 0..1).
    let plinth = add(&mut world, shapes::cuboid([0.58, 0.22, 0.58], basalt));
    for i in 0..5 {
        let x = -4.8 + i as f32 * 2.4;
        let metallic = i as f32 / 4.0;
        let roughness = 0.08 + 0.92 * (i as f32 / 4.0);
        world.spawn(plinth, Mat4::from_translation(Vec3::new(x, 0.22, 0.0)));
        let sphere = add(
            &mut world,
            shapes::uv_sphere(0.76, 48, 24, pbr([0.92, 0.66, 0.32], metallic, roughness)),
        );
        world.spawn(sphere, Mat4::from_translation(Vec3::new(x, 1.20, 0.0)));
    }

    // A field of LOD spheres with cached impostors.
    let tree = world.add_model(tree_model()).expect("tree model");
    if let Some(reason) = world.impostor_fallback(tree) {
        eprintln!("tree impostor unavailable ({reason}); drawing meshes only");
    }
    for j in 0..20 {
        for i in 0..20 {
            let x = (i as f32 - 9.5) * 4.0;
            let z = -12.0 - j as f32 * 4.0;
            world.spawn(tree, Mat4::from_translation(Vec3::new(x, 1.0, z)));
        }
    }
    world
}

impl UiApp for Game {
    fn window_input(&mut self, event: &winit::event::WindowEvent, window: &winit::window::Window) {
        use winit::event::{ElementState, MouseButton, WindowEvent};
        use winit::keyboard::PhysicalKey;
        use winit::window::CursorGrabMode;
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_captured = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                    .is_ok();
                window.set_cursor_visible(!self.mouse_captured);
            }
            WindowEvent::Focused(false)
            | WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.mouse_captured = false;
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
                self.camera.clear_keys();
            }
            _ => {}
        }
    }

    fn device_input(&mut self, event: &winit::event::DeviceEvent) {
        self.camera.device_event(event, self.mouse_captured);
    }

    fn init(&mut self, _ui: &mut Ui, cx: &mut AppContext<'_>) {
        let renderer = cx.renderer();
        renderer.set_aa_mode(self.aa);
        renderer.set_render_settings(self.settings.clone());
        let mut world = build_world(renderer);
        // `--far-px N` sets the impostor far-path threshold (0 forces the exact path).
        if let Some(px) = arg("--far-px").and_then(|v| v.parse::<f32>().ok()) {
            world.settings_mut().impostor_far_px = px;
        }
        self.world = Some(world);
        self.camera.position = self.start_pos.unwrap_or(Vec3::new(0.0, 3.0, 12.0));
        let (yaw, pitch) = self
            .start_look
            .unwrap_or((-std::f32::consts::FRAC_PI_2, -0.1));
        self.camera.set_look(yaw, pitch);
        self.camera.set_fov_y(55_f32.to_radians());
        if self.capture.is_some() {
            cx.renderer().set_frame_capture(true);
        }
    }

    fn view(&mut self, ui: &mut Ui, cx: &mut AppContext<'_>) {
        cx.request_frame(); // the scene animates every frame
        let r = cx.renderer();
        let (shadow_ms, scene_ms, post_ms) = (r.shadow_gpu_ms(), r.scene_gpu_ms(), r.post_gpu_ms());
        let s = self.stats;
        let on = [
            self.settings.shadows.is_some(),
            self.settings.ao.is_some(),
            self.settings.gi.is_some(),
            self.settings.reflections.is_some(),
            self.settings.atmosphere.is_some(),
            self.settings.fog.is_some(),
            self.settings.clouds.is_some(),
            self.settings.occlusion.is_some(),
        ];
        let features: Vec<String> = FEATURE_NAMES
            .iter()
            .zip(on)
            .enumerate()
            .map(|(i, (name, on))| {
                format!(
                    "{}:{}",
                    i + 1,
                    if on {
                        name.to_uppercase()
                    } else {
                        name.to_string()
                    }
                )
            })
            .collect();
        let lines = [
            format!(
                "{:.0} fps   GPU shadow {shadow_ms:.2} ms   scene {scene_ms:.2} ms   post {post_ms:.2} ms",
                self.fps
            ),
            format!(
                "instances {}   mesh draws {:?}   impostors {}   draw calls {}   cull {:.0} us",
                s.instances, s.mesh_draws, s.impostor_draws, s.draw_calls, s.cull_us
            ),
            format!(
                "preset {} (P)   AA {:?} (M)   sun {} (T)   RT request {} (R)",
                PRESETS[self.preset],
                self.aa,
                if self.sun_running { "moving" } else { "paused" },
                if self.settings.ray_tracing.is_some() {
                    "on"
                } else {
                    "off"
                }
            ),
            features.join("  "),
            "WASD move | Mouse look | Click to capture | Esc to release".to_string(),
        ];
        ui.layout_root(
            cx.screen(),
            Style::column().pad(12.0).align(LayoutAlign::Start),
            |ui| {
                let chrome = Paint::new()
                    .bg(Color::rgba(12, 16, 22, 192))
                    .border(1.0, Color::rgba(255, 255, 255, 38))
                    .radius(8.0);
                ui.container(
                    Style::column().pad_x(12.0).pad_y(8.0).gap(3.0),
                    chrome,
                    |ui| {
                        for (idx, line) in lines.iter().enumerate() {
                            let color = if idx == 0 {
                                Color::rgb(245, 248, 255)
                            } else {
                                Color::rgb(206, 214, 226)
                            };
                            ui.add(Text::new(line).role(TypeRole::Caption).color(color));
                        }
                    },
                );
            },
        );
    }

    fn draw_scene(
        &mut self,
        ui: &UiFrame,
        cx: &mut AppContext<'_>,
    ) -> Option<Result<(), SurfaceError>> {
        let now = Instant::now();
        let dt = self.last_frame.map_or(1.0 / 60.0, |at| {
            now.duration_since(at).as_secs_f32().min(0.1)
        });
        self.last_frame = Some(now);
        self.fps = if self.fps == 0.0 {
            1.0 / dt
        } else {
            self.fps * 0.95 + 0.05 / dt
        };

        // Input: the fly keys feed the camera, the arrows turn it.
        for code in FLY_KEYS {
            self.camera.set_key(code, cx.key_held(code));
        }
        let turn = 1.5 * dt;
        let (mut yaw, mut pitch) = (self.camera.yaw(), self.camera.pitch());
        for (code, dyaw, dpitch) in [
            (ArrowLeft, -1.0, 0.0),
            (ArrowRight, 1.0, 0.0),
            (ArrowUp, 0.0, 1.0),
            (ArrowDown, 0.0, -1.0),
        ] {
            if cx.key_held(code) {
                yaw += dyaw * turn;
                pitch += dpitch * turn;
            }
        }
        self.camera.set_look(yaw, pitch);
        self.camera.update_fly(dt);
        let (w, h) = cx.renderer().render_size();
        self.camera.set_aspect(w as f32 / h as f32);

        // Keys that change settings. Each applies to the renderer once.
        let mut changed = false;
        for (i, code) in FEATURE_KEYS.iter().enumerate() {
            if self.pressed(cx, *code) {
                let high = RenderSettings::high();
                match i {
                    0 => toggle(&mut self.settings.shadows, high.shadows),
                    1 => toggle(&mut self.settings.ao, high.ao),
                    2 => toggle(&mut self.settings.gi, high.gi),
                    3 => toggle(&mut self.settings.reflections, high.reflections),
                    4 => toggle(&mut self.settings.atmosphere, high.atmosphere),
                    5 => toggle(&mut self.settings.fog, high.fog),
                    6 => toggle(&mut self.settings.clouds, high.clouds),
                    _ => toggle(&mut self.settings.occlusion, high.occlusion),
                }
                changed = true;
            }
        }
        if self.pressed(cx, KeyP) {
            self.preset = (self.preset + 1) % 5;
            self.settings = preset_settings(self.preset);
            changed = true;
        }
        if self.pressed(cx, KeyR) {
            toggle(&mut self.settings.ray_tracing, Some(Default::default()));
            changed = true;
        }
        if self.pressed(cx, KeyM) {
            self.aa = match self.aa {
                AaMode::None => AaMode::Taa,
                AaMode::Taa => AaMode::Fsr,
                _ => AaMode::None,
            };
            cx.renderer().set_aa_mode(self.aa);
        }
        if self.pressed(cx, KeyT) {
            self.sun_running = !self.sun_running;
        }
        if changed {
            cx.renderer().set_render_settings(self.settings.clone());
        }

        // The sun: a slow arc through the day, or held still.
        if self.sun_running {
            self.sun_angle += 0.15 * dt;
        }
        let elevation = 0.7 * self.sun_angle.sin();
        let azimuth = 0.5 * self.sun_angle;
        let eye = self.camera.position;
        let env = LightingEnv {
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            ..LightingEnv::with_sun(azimuth, elevation, 3.0)
        };
        let world = self.world.as_mut()?;
        world.set_environment(env);
        self.frames += 1;
        let passes = cx.renderer().pass_gpu_ms();
        if !passes.is_empty() {
            self.pass_samples.push(passes);
        }
        let result = match world.render(cx.renderer(), &self.camera, dt, Some(ui), None) {
            Ok(stats) => {
                self.stats = stats;
                Ok(())
            }
            Err(err) => Err(err),
        };
        if self.max_frames.is_some_and(|max| self.frames >= max) {
            let r = cx.renderer();
            println!(
                "fps {:.0}  gpu ms: shadow {:.2} scene {:.2} post {:.2}  {:?}",
                self.fps,
                r.shadow_gpu_ms(),
                r.scene_gpu_ms(),
                r.post_gpu_ms(),
                self.stats
            );
            // A pass may be absent on some frames (for example, staggered
            // cascades). Match labels rather than indices so later passes do
            // not get attributed to whichever pass occupied that index first.
            if !self.pass_samples.is_empty() {
                println!(
                    "  passes over {} frames: median ms, mean ms",
                    self.pass_samples.len()
                );
                let mut labels = Vec::new();
                for frame in &self.pass_samples {
                    for &(label, _) in frame {
                        if !labels.contains(&label) {
                            labels.push(label);
                        }
                    }
                }
                let mut total = 0.0f32;
                for label in labels {
                    let mut values: Vec<f32> = self
                        .pass_samples
                        .iter()
                        .map(|frame| {
                            frame
                                .iter()
                                .filter(|pass| pass.0 == label)
                                .map(|pass| pass.1)
                                .sum()
                        })
                        .collect();
                    values.sort_by(f32::total_cmp);
                    let median = values[values.len() / 2];
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    total += median;
                    println!("  pass {label:<18} {median:.3}  {mean:.3}");
                }
                println!("  passes median total {total:.3} ms");
            }
            if let Some(path) = &self.capture
                && let Err(err) = r.save_png(std::path::Path::new(path))
            {
                eprintln!("capture failed: {err}");
            }
            cx.exit();
        }
        Some(result)
    }
}

/// Turn a feature on (with the high preset's settings) or off.
fn toggle<T>(slot: &mut Option<T>, high: Option<T>) {
    *slot = if slot.is_some() { None } else { high };
}

/// The value following `name` on the command line.
fn arg(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

fn preset_settings(index: usize) -> RenderSettings {
    match index {
        0 => RenderSettings::off(),
        1 => RenderSettings::low(),
        2 => RenderSettings::medium(),
        4 => RenderSettings::ultra(),
        _ => RenderSettings::high(),
    }
}

fn main() -> Result<(), RunError> {
    let max_frames = arg("--frames").and_then(|value| value.parse().ok());
    run(
        AppOptions::new("Dream game example").size(1280.0, 720.0),
        Game::new(max_frames),
    )
}

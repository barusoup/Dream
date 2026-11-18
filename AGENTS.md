# Dream (Dream Engine)

Rust 2024 Vulkan engine (Vulkan 1.3, ash). Custom renderer: shadow cascades -> scene pass (colour, velocity, surface, depth) -> sky, fog, screen-space AO/GI/reflections -> TAA or FSR 3.1 -> tonemap blit; see `dream::render`. BedFrame (`dream::bedframe`, re-exported as `dream::ui`) is the UI and HUD framework. WGSL shaders are validated at build time by `build.rs` via naga; do not move `.wgsl` files without updating `include_str!` paths.

## Verify

- `cargo test --all-targets` — lib tests plus `tests/public_api.rs` (public path compatibility).
- `cargo test --doc` — doc tests.
- `cargo doc --no-deps --lib` — docs; keep intra-doc links resolving.
- `cargo check --all-targets`
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets` — warnings are permitted; do not add new ones.

## Testing

Run only the tests a change can realistically affect while iterating;
the full suite is the final confirmation, not the edit loop.

- `cargo test --lib <substr>` filters by name/module path — e.g.
  `cargo test --lib bedframe::widgets::tests::` for the widget suite,
  `cargo test --lib text_field` for one feature.
- `cargo test --test public_api` runs just that integration target.
- Keep this selectable: tests live in the module they cover
  (`widgets/tests.rs` for `widgets/`, inline `mod tests` otherwise) and
  names carry a feature prefix (`text_field_…`) so a change maps to a
  filter.

## Rules

- Keep public API paths, GPU layouts (vertex/uniform/descriptor), archive formats, and queue synchronization stable. Structural changes must not alter behavior.
- Small private child modules with explicit imports; parent facades re-export moved public items at their old paths.
- No new dependencies without a concrete use. Comments and docs move with their code.
- To start a new app, copy `examples/hello.rs` and run it with `dream::ui::app::run` (`src/bedframe/app.rs`); do not hand-write a winit event loop.

## Rendering features

- `dream::render` (`src/render/mod.rs`) has the frame order, measured costs, the custom-shader contract and the known limitations. Read it before changing a pass.
- `RenderSettings::{off, low, medium, high, ultra}` are one-call starting points; `AaMode::Taa` native, `AaMode::Fsr` upscaling.
- `dream::world::World` is the scene container: models with LODs and impostors, instancing, culling and every pass behind `World::render`. Start from `examples/game.rs` (3D) or `examples/hello.rs` (UI).
- Scene shaders write three colour outputs: colour, velocity and the surface encoding (`surface_encode`, `lighting.wgsl`). Shadow passes are depth-only: check `ctx.pass()`. Alpha-blended geometry belongs to `ScenePass::Transparent` (after sky and fog; write no velocity or surface). LoD crossfades are opaque: mesh and impostor both draw in the main pass and split pixels with `lod_dither` (`lighting.wgsl`), never by alpha. Call `lighting_begin` before `lighting_eval`.
- Checks: `cargo run --release --bin scene_check` (per-feature and preset images, CPU references) and `cargo run --release --bin world_check` (culling, LOD and instancing).

## Profiling

- `cargo run --release --bin bedframe_bench -- --apps --unicode` — CPU frame-build timings, allocation counts, cache pressure, animated controls, and editor workloads at 1x/2x. No Vulkan renderer; requested Rust heap and process working set are separate measurements. Portable Unicode workloads without registered script fonts intentionally hide unsupported clusters.
- `cargo test --release --lib preparation_profile -- --ignored --nocapture` — CPU geometry preparation for instanced, gradient, and mixed frames.
- `cargo test --release --lib long_text_layout_profile -- --ignored --nocapture` — direct text-layout scaling with warmed glyphs.
- `cargo run --release --bin bedframe_gpu_bench` — real Vulkan UI-only versus scene-under-opaque-UI CPU recording/submission timings and live allocator bytes. These are not GPU execution timings or complete driver memory.
- `cargo run --release --bin bedframe_check` — GPU readback/pixel regressions; screenshots go under `target/bedframe-check/`. Any readback tool must call `Renderer::set_frame_capture(true)` before rendering the frames it reads.
- `cargo run --release --bin dream_app -- --selfcheck` — app navigation, thumbnails, rendering, idle/wake, and hidden-window checks.
- `cargo run --release --bin bedframe_mem_probe` — resident memory of a UI-only Vulkan app at each start-up stage, then the hot set after `dream::memory::trim_working_set`.
- `cargo run --release --bin scene_check` — renders the 3D feature test scene per configuration to `target/scene-check/*.png`.
- `cargo run --release --bin world_check` — culling and LOD world: culling on/off frame diffs, frustum and occlusion effectiveness, LOD dolly popping, impostor disk cache and timings; PNGs and failing-frame diagnostics go to `target/world-check/`.

// Shared linear depth, one pass per frame. Written at the depth resolution
// (the finest the enabled screen-space chains use); the chains map their
// working texels onto it, see `depth_at` in the indirect and reflection shaders.

// Mirror of `DepthParams` (48 bytes) in linear_depth.rs.
struct DepthParams {
    size: vec4<f32>,  // depth width, height, 1/width, 1/height
    full: vec4<f32>,  // render width, height, 1/width, 1/height
    proj: vec4<f32>,  // near, far, tan(fov_y / 2), aspect
};

@group(0) @binding(0) var<uniform> P: DepthParams;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var linear_out: texture_storage_2d<r32float, write>;

// Stored for sky. Far beyond any scene distance, and finite in every format.
const SKY_DEPTH: f32 = 65000.0;

@compute @workgroup_size(8, 8)
fn linear_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let q = min(vec2<i32>(gid.xy) * scale, full_max);
    // Depth is 0 at the near plane and 1 at the far plane (perspective_rh).
    let z = textureLoad(scene_depth, q, 0);
    var linear = SKY_DEPTH;
    if z < 1.0 {
        let n = P.proj.x;
        let f = P.proj.y;
        linear = n * f / (f - z * (f - n));
    }
    textureStore(linear_out, vec2<i32>(gid.xy), vec4<f32>(linear, 0.0, 0.0, 0.0));
}

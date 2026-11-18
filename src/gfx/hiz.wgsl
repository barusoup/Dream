// Hi-Z reduction: the farthest scene depth of every block of the occlusion
// grid. One invocation per grid texel. Source texels that only partly overlap
// a block are included, so the result is conservative for any size ratio.
// Mirror of `HiZParams` in occlusion.rs (32 bytes).

struct HiZParams {
    src: vec4<u32>, // source (depth) width, height, 0, 0
    dst: vec4<u32>, // grid width, height, 0, 0
};

@group(0) @binding(0) var<uniform> P: HiZParams;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var hiz_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn hiz_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.dst.x || gid.y >= P.dst.y {
        return;
    }
    // Source texels overlapping this block: [floor(a), ceil(b)) in source texels,
    // with at least one texel and never past the edge.
    let x0 = min(gid.x * P.src.x / P.dst.x, P.src.x - 1u);
    let x1 = min(max(((gid.x + 1u) * P.src.x + P.dst.x - 1u) / P.dst.x, x0 + 1u), P.src.x);
    let y0 = min(gid.y * P.src.y / P.dst.y, P.src.y - 1u);
    let y1 = min(max(((gid.y + 1u) * P.src.y + P.dst.y - 1u) / P.dst.y, y0 + 1u), P.src.y);
    // Sky texels hold depth 1.0, so an empty block stays at the far plane.
    var far = 0.0;
    for (var y = y0; y < y1; y = y + 1u) {
        for (var x = x0; x < x1; x = x + 1u) {
            far = max(far, textureLoad(scene_depth, vec2<i32>(i32(x), i32(y)), 0));
        }
    }
    textureStore(hiz_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(far, 0.0, 0.0, 0.0));
}

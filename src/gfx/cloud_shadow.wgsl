// Cloud shadow map (gfx/cloud_shadow.rs): the sun's transmittance through the
// cloud layer for each texel of a square on the sea-level plane, centred near
// the camera. Lighting multiplies the sun by it (lighting_cloud_visibility).
// Prepended with atmo_common.wgsl and cloud_common.wgsl, which hold the density.

struct ShadowParams {
    // x, y: centre of the square (world x, z, m), z: extent (m), w: texels per side.
    grid: vec4<f32>,
    // xyz: direction to the sun, w: sea level (m).
    sun: vec4<f32>,
};

// Samples along the sun ray per texel.
const SHADOW_STEPS: u32 = 12u;

@group(0) @binding(0) var<uniform> cp: CloudParams;
@group(0) @binding(1) var<uniform> sp: ShadowParams;
@group(0) @binding(2) var cloud_base: texture_3d<f32>;
@group(0) @binding(3) var cloud_detail: texture_3d<f32>;
@group(0) @binding(4) var cloud_samp: sampler;
@group(0) @binding(5) var out_map: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(sp.grid.w);
    if gid.x >= n || gid.y >= n {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / f32(n);
    let g = sp.grid.xy + (uv - 0.5) * sp.grid.z;
    var t = 1.0;
    if sp.sun.y > 0.0 {
        t = cloud_sun_transmittance(cp, cloud_base, cloud_detail, cloud_samp,
            vec3<f32>(g.x, sp.sun.w, g.y), sp.sun.xyz, SHADOW_STEPS);
    }
    textureStore(out_map, vec2<i32>(gid.xy), vec4<f32>(t, 0.0, 0.0, 0.0));
}

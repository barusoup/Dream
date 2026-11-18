// Sky-view LUT pass: radiance around the camera, without the sun disk. Runs
// every frame. Prepended with atmo_common.wgsl at load.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var tlut: texture_2d<f32>;
@group(0) @binding(2) var mslut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var skyview_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(192u, 108u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let sun = atmo.sun_dir.xyz;
    let dir = atmo_skyview_dir(atmo, uv, sun);
    let l = atmo_sky_radiance(atmo, tlut, mslut, lut_sampler, atmo.camera.x, dir, sun);
    textureStore(skyview_out, vec2<i32>(gid.xy), vec4<f32>(l, 1.0));
}

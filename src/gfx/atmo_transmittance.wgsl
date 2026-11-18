// Transmittance LUT pass. Prepended with atmo_common.wgsl at load.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var transmittance_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(256u, 64u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let rm = atmo_tlut_params(atmo, uv);
    let t = atmo_transmittance_integral(atmo, rm.x, rm.y);
    textureStore(transmittance_out, vec2<i32>(gid.xy), vec4<f32>(t, 1.0));
}

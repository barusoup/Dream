// Fog integrate pass: front-to-back accumulation per froxel column with the
// energy-conserving slice integral (Hillaire 2015). Stores the accumulated
// in-scattered light (rgb) and the transmittance (a) at the end of each slice.
// Prepended with fog_common.wgsl at load. Dispatch: (width / 8, height / 8, 1).

@group(0) @binding(0) var<uniform> fog: FogParams;
@group(0) @binding(1) var injected: texture_3d<f32>;
@group(0) @binding(2) var integrated: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(fog.grid.x);
    let height = u32(fog.grid.y);
    let slices = u32(fog.grid.z);
    if gid.x >= width || gid.y >= height {
        return;
    }
    let near = fog.grid.w;
    let dist = fog.range.x;
    let n = fog.grid.z;
    let ratio = exp2(log2(dist / near) / n);
    let ratio_m1 = ratio - 1.0;
    var z = near;
    var acc = vec3<f32>(0.0);
    var transmittance = 1.0;
    let xy = vec2<i32>(gid.xy);
    for (var k = 0u; k < slices; k++) {
        let v = textureLoad(injected, vec3<i32>(xy, i32(k)), 0);
        let dz = z * ratio_m1;
        z *= ratio;
        let sigma = v.a;
        let step = exp(-sigma * dz);
        let s_int = select(v.rgb * dz, v.rgb * (1.0 - step) / max(sigma, 1e-6), sigma > 1e-6);
        acc += transmittance * s_int;
        transmittance *= step;
        textureStore(integrated, vec3<i32>(xy, i32(k)), vec4<f32>(acc, transmittance));
    }
}

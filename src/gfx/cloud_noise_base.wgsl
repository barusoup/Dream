// Base cloud shape volume, 96^3 rgba8unorm, built once. R: Perlin-Worley (a
// Perlin base remapped by inverted Worley fbm). G, B, A: Worley at 4, 8 and 16
// cells per tile. Every channel tiles seamlessly. Prepended with cloud_common.wgsl.

@group(0) @binding(0) var out_vol: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_vol);
    if gid.x >= size.x || gid.y >= size.y || gid.z >= size.z {
        return;
    }
    let p = (vec3<f32>(gid) + 0.5) / vec3<f32>(size);
    let perlin = cloud_perlin(p * 4.0, 4) * 0.625
        + cloud_perlin(p * 8.0, 8) * 0.25
        + cloud_perlin(p * 16.0, 16) * 0.125;
    let pn = clamp(0.5 + perlin, 0.0, 1.0);
    let w = cloud_worley_fbm(p);
    let r = cloud_remap(pn, (1.0 - w) * 0.15, 1.0, 0.0, 1.0);
    let g = cloud_worley(p * 4.0, 4);
    let b = cloud_worley(p * 8.0, 8);
    let a = cloud_worley(p * 16.0, 16);
    textureStore(out_vol, vec3<i32>(gid), vec4<f32>(r, g, b, a));
}

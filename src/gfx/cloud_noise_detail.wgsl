// Detail erosion volume, 32^3 rgba8unorm, built once. R: three octaves of tiling
// Worley fbm (the erosion term). G, B, A: Worley at 4, 8 and 16 cells per tile.
// Prepended with cloud_common.wgsl.

@group(0) @binding(0) var out_vol: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_vol);
    if gid.x >= size.x || gid.y >= size.y || gid.z >= size.z {
        return;
    }
    let p = (vec3<f32>(gid) + 0.5) / vec3<f32>(size);
    let r = cloud_worley_fbm(p);
    let g = cloud_worley(p * 4.0, 4);
    let b = cloud_worley(p * 8.0, 8);
    let a = cloud_worley(p * 16.0, 16);
    textureStore(out_vol, vec3<i32>(gid), vec4<f32>(r, g, b, a));
}

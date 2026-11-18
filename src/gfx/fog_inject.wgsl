// Fog inject pass: scattering (rgb, per metre) and extinction (a, per metre)
// per froxel, with temporal reprojection of the previous volume. Prepended
// with fog_common.wgsl at load. Dispatch: (width / 8, height / 8, slices).

@group(0) @binding(0) var<uniform> fog: FogParams;
@group(0) @binding(1) var history: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var current: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var shadow_map: texture_depth_2d_array;
@group(0) @binding(5) var shadow_cmp: sampler_comparison;
@group(0) @binding(6) var<uniform> lf: LightingFrame;
// Cloud shadow map of this frame slot (R32F), sampled with `lin`.
@group(0) @binding(7) var cloud_shadow_map: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(fog.grid.x);
    let height = u32(fog.grid.y);
    let slices = u32(fog.grid.z);
    if gid.x >= width || gid.y >= height || gid.z >= slices {
        return;
    }
    let near = fog.grid.w;
    let dist = fog.range.x;
    let n = fog.grid.z;
    let k = f32(gid.z);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(width), f32(height));
    let ndc = uv * 2.0 - vec2<f32>(1.0);
    let tan_x = fog.tan_fov.x;
    let tan_y = fog.tan_fov.y;

    // World position of a froxel at slice coordinate s (view-depth z).
    let world = fog_world_at(fog, ndc, fog_slice_depth(k + fog.light.w, near, dist, n));
    let density = fog_density(fog, world);
    var value = vec4<f32>(0.0);
    if density > 1e-7 {
        let cam = fog.cam_pos.xyz;
        let dir = normalize(world - cam);
        let cos_sd = dot(dir, fog.sun_dir.xyz);
        let vis = fog_shadow(lf, shadow_map, shadow_cmp, world)
            * fog_cloud_visibility(lf, cloud_shadow_map, lin, world);
        let albedo = fog.scatter.rgb;
        let sun_term = fog.sun_color.rgb * vis * fog_hg(cos_sd, fog.scatter.w) * fog.light.y;
        let ambient = fog.ambient.rgb * fog.light.x;
        let scatter = albedo * density * (sun_term + ambient);
        value = vec4<f32>(scatter, density);
    }

    // Reproject the unjittered centre of this froxel into the previous volume.
    if fog.wind.w > 0.0 && fog.range.w < 0.5 {
        let centre = fog_world_at(fog, ndc, fog_slice_depth(k + 0.5, near, dist, n));
        let prev = fog.prev_view_proj * vec4<f32>(centre, 1.0);
        if prev.w > near && prev.w < dist {
            let uvp = prev.xy / prev.w * 0.5 + vec2<f32>(0.5);
            let sp = fog_depth_to_slice(prev.w, near, dist, n);
            if uvp.x >= 0.0 && uvp.x <= 1.0 && uvp.y >= 0.0 && uvp.y <= 1.0 {
                // Injected texels are centred on their slice: texel k sits at s = k + 0.5.
                let hist = textureSampleLevel(history, lin, vec3<f32>(uvp, sp / n), 0.0);
                value = mix(value, hist, fog.wind.w);
            }
        }
    }
    textureStore(current, vec3<i32>(vec2<i32>(gid.xy), i32(gid.z)), value);
}

// World position of the view-space point at ndc (x, y) and view depth z.
fn fog_world_at(p: FogParams, ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let view = vec4<f32>(ndc.x * p.tan_fov.x * z, ndc.y * p.tan_fov.y * z, -z, 1.0);
    return (p.cam_to_world * view).xyz;
}

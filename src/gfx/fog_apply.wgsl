// Fog apply pass: a fullscreen draw blended onto the scene colour with
// out.rgb = in-scatter and out.a = transmittance (blend ONE, SRC_ALPHA).
// Within the volume it reads the integrated froxels; beyond it, and on sky
// pixels, it continues with the closed-form height-fog integral from the
// volume's last slice. Prepended with fog_common.wgsl at load.

@group(0) @binding(0) var<uniform> fog: FogParams;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var integrated: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;

struct ApplyVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> ApplyVsOut {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let c = corners[i];
    var out: ApplyVsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.ndc = c;
    return out;
}

@fragment
fn fs_main(in: ApplyVsOut) -> @location(0) vec4<f32> {
    let uv = in.ndc * 0.5 + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let d = textureLoad(scene_depth, vec2<i32>(floor(uv * dims)), 0);
    let near = fog.grid.w;
    let far = fog.tan_fov.z;
    let dist = fog.range.x;
    let n = fog.grid.z;
    let tan_x = fog.tan_fov.x;
    let tan_y = fog.tan_fov.y;
    // Euclidean distance along the view ray per unit of view depth.
    let k = length(vec3<f32>(in.ndc.x * tan_x, in.ndc.y * tan_y, -1.0));
    let dir = normalize((fog.cam_to_world * vec4<f32>(in.ndc.x * tan_x, in.ndc.y * tan_y, -1.0, 0.0)).xyz);
    let sky = d >= 1.0;
    var z = far;
    if !sky {
        z = near * far / (far - d * (far - near));
    }

    if !sky && z <= dist {
        // Integrated froxels: texel k holds the total up to the end of slice k.
        let s = fog_depth_to_slice(z, near, dist, n);
        let w = clamp((s - 0.5) / n, 0.0, 1.0);
        let v = textureSampleLevel(integrated, lin, vec3<f32>(uv, w), 0.0);
        return vec4<f32>(v.rgb, v.a);
    }

    // Beyond the volume: continue from its last slice with the analytic
    // height-fog term. In-scatter of a constant-albedo medium is albedo * L *
    // (1 - T), exact along the segment.
    let v_end = textureSampleLevel(integrated, lin, vec3<f32>(uv, 1.0 - 0.5 / n), 0.0);
    let t_start = dist * k;
    var t_end = t_start + fog.range.y;
    if !sky {
        t_end = z * k;
    }
    let cam = fog.cam_pos.xyz;
    let od = fog_height_od(t_start, t_end, cam.y, dir.y, fog.density.x, fog.density.y, fog.density.z);
    let tf = exp(-od);
    let cos_sd = dot(dir, fog.sun_dir.xyz);
    let source = fog.ambient.rgb * fog.light.x + fog.sun_color.rgb * fog.light.y * fog_hg(cos_sd, fog.scatter.w);
    let in_scatter = v_end.rgb + v_end.a * fog.scatter.rgb * source * (1.0 - tf);
    return vec4<f32>(in_scatter, v_end.a * tf);
}

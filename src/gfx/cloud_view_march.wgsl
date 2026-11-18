// Per-pixel view cloud march (gfx/cloud_view.rs). Each texel of the half-
// resolution grid marches the cloud layer along the ray through one full-
// resolution pixel: `fp = 2 * gid.xy + o` for this frame's 2x2 offset `o`.
// The march stops at that pixel's scene depth and writes radiance/transmittance
// plus front/back distances (negative back = geometry). Prepended with
// atmo_common.wgsl and cloud_common.wgsl.

@group(0) @binding(0) var<uniform> vp: ViewParams;
@group(0) @binding(1) var<uniform> cp: CloudParams;
@group(0) @binding(2) var<uniform> atmo: AtmoParams;
@group(0) @binding(3) var cloud_base: texture_3d<f32>;
@group(0) @binding(4) var cloud_detail: texture_3d<f32>;
@group(0) @binding(5) var cloud_samp: sampler;
@group(0) @binding(6) var tlut: texture_2d<f32>;
@group(0) @binding(7) var skyview: texture_2d<f32>;
@group(0) @binding(8) var lut_sampler: sampler;
// Linear view depth of the scene (metres), at its own resolution.
@group(0) @binding(9) var scene_depth: texture_2d<f32>;
@group(0) @binding(10) var low_colour: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var low_dist: texture_storage_2d<rg32float, write>;

// Farthest linear depth of the render-target pixels at q, read from the depth
// texture, which may be at a coarser resolution than the render target.
fn view_depth_at(q: vec2<u32>) -> f32 {
    let dims = vec2<u32>(textureDimensions(scene_depth));
    let full = vec2<u32>(vp.screen.xy);
    let t = min(q * dims / full, dims - 1u);
    return textureLoad(scene_depth, vec2<i32>(t), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let low = vec2<u32>(vp.res.zw);
    if gid.x >= low.x || gid.y >= low.y {
        return;
    }
    let frame = u32(vp.state.x);
    var offs = array<vec2<u32>, 4>(vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u));
    let o = offs[frame % 4u];
    let full = vec2<u32>(vp.screen.xy);
    let fp = gid.xy * 2u + o;
    let out_c = vec2<i32>(gid.xy);
    if fp.x >= full.x || fp.y >= full.y {
        textureStore(low_colour, out_c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(low_dist, out_c, vec4<f32>(CLOUD_NO_HIT, CLOUD_NO_HIT, 0.0, 0.0));
        return;
    }

    let z = view_depth_at(fp);
    let geom = z < CLOUD_SKY_DEPTH;
    let dir = cloud_view_ray(vp, vec2<f32>(fp) + 0.5);

    let sun = atmo.sun_dir.xyz;
    let zenith_uv = atmo_skyview_uv(atmo, vec3<f32>(0.0, 1.0, 0.0), sun);
    let amb_zenith = textureSampleLevel(skyview, lut_sampler, zenith_uv, 0.0).rgb * atmo.camera.w;

    var hit = CloudHit(vec3<f32>(0.0), 1.0, CLOUD_NO_HIT, CLOUD_NO_HIT);
    var span = cloud_layer_span(cloud_planet_centre(cp.cam, cp.layer), dir, cp.layer);
    if geom {
        span.y = min(span.y, z / max(dot(dir, vp.fwd.xyz), 1e-3));
    }
    if cp.shape.x > 0.0 && span.y > span.x {
        // Near-camera fine/coarse steps scaled with distance in `cloud_integrate`
        // so close/inside views stay finely sampled without overflowing budget on
        // 60 km horizon rays.
        let fine = clamp(cp.layer.z / max(vp.march.x * 1.1, 24.0), 16.0, 48.0);
        let coarse = fine * 2.5;
        let budget = u32(vp.march.x) * 5u;
        let jitter = fract(
            0.7548776662 * f32(fp.x)
                + 0.5698402910 * f32(fp.y)
                + 0.61803398875 * f32(frame / 4u),
        );
        hit = cloud_integrate(cp, atmo, cloud_base, cloud_detail, cloud_samp, tlut, skyview,
            lut_sampler, dir, span.x, span.y, fine, coarse, budget, jitter, amb_zenith, 0.00018);
    }

    var back = max(hit.back, 1.0);
    if geom {
        back = -back;
    }
    textureStore(low_colour, out_c, vec4<f32>(hit.light, hit.trans));
    textureStore(low_dist, out_c, vec4<f32>(hit.front, back, 0.0, 0.0));
}

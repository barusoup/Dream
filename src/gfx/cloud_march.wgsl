// Panorama cloud march: fills the sky environment panorama for this frame. Each
// texel of the panorama is written every frame: the texels of this frame's
// refresh block are marched and blended into the previous panorama, and the rest
// carry their previous value. A full refresh (first fill, settings change, history
// reset) replaces every texel. Writes rgb = in-scattered cloud radiance and
// a = transmittance through the layer. The panorama feeds the sky environment map
// only; the view clouds are marched per pixel (cloud_view_march.wgsl). Prepended
// with atmo_common.wgsl and cloud_common.wgsl, which hold the density and march core.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var<uniform> cp: CloudParams;
@group(0) @binding(2) var cloud_base: texture_3d<f32>;
@group(0) @binding(3) var cloud_detail: texture_3d<f32>;
@group(0) @binding(4) var cloud_samp: sampler;
@group(0) @binding(5) var tlut: texture_2d<f32>;
@group(0) @binding(6) var skyview: texture_2d<f32>;
@group(0) @binding(7) var lut_sampler: sampler;
@group(0) @binding(8) var panorama: texture_storage_2d<rgba16float, write>;
// The previous frame's panorama.
@group(0) @binding(9) var prev_panorama: texture_2d<f32>;

// Weight of a refreshed texel's new sample against its previous value.
const PANORAMA_BLEND: f32 = 0.35;

fn cloud_march(dir: vec3<f32>, texel: vec2<u32>, amb_zenith: vec3<f32>, jitter: f32) -> vec4<f32> {
    if cp.shape.x <= 0.0 || dir.y < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let o = cloud_planet_centre(cp.cam, cp.layer);
    let span = cloud_layer_span(o, dir, cp.layer);
    if span.y <= span.x {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Uniform panorama march along `[span.x, min(span.y, span.x + max_dist)]`.
    let steps = cp.look.z;
    let t1 = min(span.y, span.x + cp.layer.w);
    if t1 <= span.x {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let dt = (t1 - span.x) / steps;
    let hit = cloud_integrate(cp, atmo, cloud_base, cloud_detail, cloud_samp, tlut, skyview,
        lut_sampler, dir, span.x, t1, dt, dt, u32(steps), jitter, amb_zenith, 0.0);
    return vec4<f32>(hit.light, hit.trans);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u32(cp.panorama.x), u32(cp.panorama.y));
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let texel = gid.xy;
    let prev = textureLoad(prev_panorama, vec2<i32>(texel), 0);
    let full = cp.panorama.z > 0.5;
    // This texel is refreshed when it lies in the frame's block of the refresh stride.
    let stride = u32(cp.refresh.x);
    let refreshed = full || (texel.x % stride == u32(cp.refresh.y) && texel.y % stride == u32(cp.refresh.z));
    var out = prev;
    if refreshed {
        let sun = atmo.sun_dir.xyz;
        let zenith_uv = atmo_skyview_uv(atmo, vec3<f32>(0.0, 1.0, 0.0), sun);
        let amb_zenith = textureSampleLevel(skyview, lut_sampler, zenith_uv, 0.0).rgb * atmo.camera.w;
        let uv = (vec2<f32>(texel) + 0.5) / vec2<f32>(size);
        let dir = cloud_panorama_dir(uv);
        // Full-frame resets use midpoint sampling to avoid freezing per-texel
        // dither noise into unrefreshed tiles; incremental refreshes dither in
        // time with golden-ratio offsets so they converge smoothly.
        let temporal_jitter = fract(cp.refresh.w * 0.6180339887 + cloud_ign(vec2<f32>(texel)));
        let jitter = select(temporal_jitter, 0.5, full);
        let sample = cloud_march(dir, texel, amb_zenith, jitter);
        if full {
            out = sample;
        } else {
            out = mix(prev, sample, PANORAMA_BLEND);
        }
    }
    textureStore(panorama, vec2<i32>(texel), out);
}

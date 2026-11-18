// Per-pixel view cloud composite (gfx/cloud_view.rs). A fullscreen blend over the
// scene colour, after the sky and before fog: out = (L, T) with blend
// colour * T + L. Reads the full-resolution history texel directly and keeps only
// the portion of the cloud span in front of the pixel's scene depth.
// Prepended with cloud_common.wgsl (which holds ViewParams).

@group(0) @binding(0) var<uniform> vp: ViewParams;
// Linear view depth of the scene, at its own resolution.
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
@group(0) @binding(2) var hist_colour: texture_2d<f32>;
@group(0) @binding(3) var hist_dist: texture_2d<f32>;

struct CompVsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> CompVsOut {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: CompVsOut;
    out.pos = vec4<f32>(corners[i], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: CompVsOut) -> @location(0) vec4<f32> {
    let full = vec2<u32>(vp.screen.xy);
    let pix = min(vec2<u32>(in.pos.xy), full - 1u);
    let px = vec2<f32>(pix) + 0.5;
    let dims = vec2<u32>(textureDimensions(scene_depth));
    let zl = textureLoad(scene_depth, vec2<i32>(min(pix * dims / full, dims - 1u)), 0).r;
    let sky = zl >= CLOUD_SKY_DEPTH;
    let dir = cloud_view_ray(vp, px);
    // Distance along this pixel's ray (sky is at infinity: k = 1 below).
    let dist = zl / max(dot(dir, vp.fwd.xyz), 1e-3);

    let t = vec2<i32>(pix);
    let c = textureLoad(hist_colour, t, 0);
    let pair = textureLoad(hist_dist, t, 0).xy;
    let front = pair.x;
    let back = abs(pair.y);
    var kf = 1.0;
    if !sky {
        kf = clamp((dist - front) / max(back - front, 1.0), 0.0, 1.0);
    }
    let tr = clamp(c.a, 1e-6, 1.0);
    var tk = tr;
    if kf <= 0.0 {
        tk = 1.0;
    } else if kf < 1.0 {
        tk = exp(kf * log(tr));
    }
    let lk = c.rgb * (1.0 - tk) / max(1.0 - c.a, 1e-4);
    return vec4<f32>(lk, tk);
}

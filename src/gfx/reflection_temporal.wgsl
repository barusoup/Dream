// Screen-space reflections, pass 2: temporal accumulation. The previous
// accumulated result is reprojected with the surface velocity, rejected where its
// stored depth disagrees with the current surface, clamped to the current 3x3
// neighbourhood (widened by 25%) and blended with the current trace. The blend
// is premultiplied by confidence, so misses do not darken hits, and renormalised.
//
// Mirror of `ReflectionParams`; see reflection_trace.wgsl.

struct ReflectionParams {
    size: vec4<f32>,
    full: vec4<f32>,
    depth: vec4<f32>,
    proj: vec4<f32>,
    view_from_world: mat4x4<f32>,
    trace: vec4<f32>,
    surface: vec4<f32>,
    frame: vec4<f32>,
    temporal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: ReflectionParams;
// This frame's trace (rgb colour, a confidence).
@group(0) @binding(1) var raw: texture_2d<f32>;
@group(0) @binding(2) var linear_depth: texture_2d<f32>;
// Full-resolution motion: prev_uv - curr_uv, bottom-up uv.
@group(0) @binding(3) var velocity: texture_2d<f32>;
// The previous accumulated result (same format as `raw`).
@group(0) @binding(4) var history: texture_2d<f32>;
@group(0) @binding(5) var depth_prev: texture_2d<f32>;
@group(0) @binding(6) var history_sampler: sampler;
@group(0) @binding(7) var out_tex: texture_storage_2d<rgba16float, write>;

const SKY_DEPTH: f32 = 65000.0;

fn depth_at(q: vec2<i32>) -> f32 {
    let wq = clamp(q, vec2<i32>(0), vec2<i32>(P.size.xy) - vec2<i32>(1));
    let dmax = vec2<i32>(round(P.depth.xy)) - vec2<i32>(1);
    let dq = clamp(vec2<i32>(vec2<f32>(wq) * P.depth.xy * P.size.zw), vec2<i32>(0), dmax);
    return textureLoad(linear_depth, dq, 0).r;
}

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let max_q = vec2<i32>(P.size.xy) - vec2<i32>(1);
    let qf = vec2<f32>(q) + vec2<f32>(0.5);
    let cur = textureLoad(raw, q, 0);
    let dc = depth_at(q);
    if dc >= SKY_DEPTH * 0.5 {
        textureStore(out_tex, q, cur);
        return;
    }

    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let vel = textureLoad(velocity, min(q * scale, full_max), 0).xy;
    let prev_uv = qf * P.size.zw + vel;

    // Current neighbourhood of confident samples, for clamping the history.
    var mn = vec3<f32>(1e9);
    var mx = vec3<f32>(-1e9);
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = textureLoad(raw, clamp(q + vec2<i32>(dx, dy), vec2<i32>(0), max_q), 0);
            if s.a > 0.01 {
                mn = min(mn, s.rgb);
                mx = max(mx, s.rgb);
                count += 1.0;
            }
        }
    }

    // Blend weight of the current frame: 1 starts a fresh history.
    var weight = 1.0;
    if P.frame.y < 0.5 && all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
        let dmax = vec2<i32>(round(P.depth.xy)) - vec2<i32>(1);
        let pq = clamp(vec2<i32>(floor(prev_uv * P.depth.xy)), vec2<i32>(0), dmax);
        let dp = textureLoad(depth_prev, pq, 0).r;
        if dp < SKY_DEPTH * 0.5 && abs(dp - dc) <= P.temporal.y * dc {
            weight = P.temporal.x;
        }
    }
    // A fresh history is never read: an uninitialised image may hold NaN.
    if weight >= 1.0 {
        textureStore(out_tex, q, cur);
        return;
    }

    let hist = textureSampleLevel(history, history_sampler, prev_uv, 0.0);
    var hist_rgb = hist.rgb;
    if count > 0.0 {
        let centre = 0.5 * (mn + mx);
        let half_range = 0.5 * (mx - mn) * 1.25 + vec3<f32>(1e-3);
        hist_rgb = clamp(hist_rgb, centre - half_range, centre + half_range);
    }
    let pm = mix(hist_rgb * hist.a, cur.rgb * cur.a, weight);
    let conf = mix(hist.a, cur.a, weight);
    let rgb = select(vec3<f32>(0.0), pm / max(conf, 1e-4), conf > 1e-4);
    textureStore(out_tex, q, vec4<f32>(rgb, conf));
}

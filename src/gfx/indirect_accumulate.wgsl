// Indirect chain, pass 4: temporal accumulation. Reprojects the previous
// accumulated result with this frame's velocity, rejects history whose stored
// depth disagrees with the current surface, and blends in the current frame.

// Mirror of `IndirectParams` (176 bytes); see `indirect_horizon.wgsl`.
struct IndirectParams {
    size: vec4<f32>,
    full: vec4<f32>,
    proj: vec4<f32>,
    view_from_world: mat4x4<f32>,
    ao: vec4<f32>,
    gi: vec4<f32>,
    march: vec4<f32>,
    temporal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: IndirectParams;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var depth_now: texture_2d<f32>;
// Full-resolution motion: prev_uv - curr_uv, bottom-up uv.
@group(0) @binding(3) var velocity: texture_2d<f32>;
@group(0) @binding(4) var history: texture_2d<f32>;
@group(0) @binding(5) var depth_prev: texture_2d<f32>;
@group(0) @binding(6) var history_sampler: sampler;
@group(0) @binding(7) var out_tex: texture_storage_2d<rgba16float, write>;

const SKY_DEPTH: f32 = 65000.0;

// Linear depth of working texel q: the shared depth texel at its top-left
// (`march.zw` is depth texels per working texel). Clamped to the chain's edges.
fn depth_at(q: vec2<i32>) -> f32 {
    let wq = clamp(q, vec2<i32>(0), vec2<i32>(P.size.xy) - vec2<i32>(1));
    let dmax = vec2<i32>(round(P.size.xy * P.march.zw)) - vec2<i32>(1);
    let dq = clamp(vec2<i32>(vec2<f32>(wq) * P.march.zw), vec2<i32>(0), dmax);
    return textureLoad(depth_now, dq, 0).r;
}

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let max_q = vec2<i32>(P.size.xy) - vec2<i32>(1);
    let cur = textureLoad(current, q, 0);
    let dc = depth_at(q);
    if dc >= SKY_DEPTH * 0.5 {
        textureStore(out_tex, q, cur);
        return;
    }

    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let vel = textureLoad(velocity, min(q * scale, full_max), 0).xy;
    let uv = (vec2<f32>(q) + vec2<f32>(0.5)) * P.size.zw;
    let prev_uv = uv + vel;

    // Weight of the current frame: 1 starts a fresh history.
    var weight = 1.0;
    if P.march.y < 0.5 && all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
        let dmax = vec2<i32>(round(P.size.xy * P.march.zw)) - vec2<i32>(1);
        let pq = clamp(vec2<i32>(floor(prev_uv * P.size.xy * P.march.zw)), vec2<i32>(0), dmax);
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
    var mn = cur;
    var mx = cur;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nq = clamp(q + vec2<i32>(dx, dy), vec2<i32>(0), max_q);
            let ds = depth_at(nq);
            if ds < SKY_DEPTH * 0.5 && abs(ds - dc) <= 0.2 * dc + 0.05 {
                let s = textureLoad(current, nq, 0);
                mn = min(mn, s);
                mx = max(mx, s);
            }
        }
    }
    let centre = 0.5 * (mn + mx);
    let half_range = max(0.5 * (mx - mn) * 1.4, vec4<f32>(0.025, 0.025, 0.025, 0.04));
    let hist = clamp(textureSampleLevel(history, history_sampler, prev_uv, 0.0), centre - half_range, centre + half_range);
    let out = mix(hist, cur, weight);
    textureStore(out_tex, q, out);
}

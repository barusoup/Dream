// Indirect chain, pass 3: bilateral spatial filter over the raw horizon output.
// A Gaussian 5x5 kernel weighted by linear-depth similarity, world/view-normal
// alignment, and tangent-plane distance so GI and AO never bleed across walls,
// floors, or foreground edges.

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
@group(0) @binding(1) var raw: texture_2d<f32>;
@group(0) @binding(2) var linear_depth: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var surface_tex: texture_2d<f32>;

const SKY_DEPTH: f32 = 65000.0;
const DEPTH_EDGE: f32 = 0.08;

fn decode_world_normal(s: vec4<f32>) -> vec3<f32> {
    let e = s.xy * 2.0 - vec2<f32>(1.0);
    let folded = abs(e.x) + abs(e.y) > 1.0;
    var o = e;
    if folded {
        let sign_e = vec2<f32>(select(-1.0, 1.0, e.x >= 0.0), select(-1.0, 1.0, e.y >= 0.0));
        o = (vec2<f32>(1.0) - abs(e.yx)) * sign_e;
    }
    let ny = (1.0 - abs(o.x) - abs(o.y)) * select(1.0, -1.0, folded);
    return normalize(vec3<f32>(o.x, ny, o.y));
}

fn depth_texel(q: vec2<i32>) -> vec2<i32> {
    let wq = clamp(q, vec2<i32>(0), vec2<i32>(P.size.xy) - vec2<i32>(1));
    let dmax = vec2<i32>(round(P.size.xy * P.march.zw)) - vec2<i32>(1);
    return clamp(vec2<i32>(vec2<f32>(wq) * P.march.zw), vec2<i32>(0), dmax);
}

fn depth_at(q: vec2<i32>) -> f32 {
    return textureLoad(linear_depth, depth_texel(q), 0).r;
}

fn view_of(q: vec2<i32>, d: f32) -> vec3<f32> {
    let dq = depth_texel(q);
    let depth_size = P.size.xy * P.march.zw;
    let ndc = (vec2<f32>(dq) + vec2<f32>(0.5)) / depth_size * 2.0 - vec2<f32>(1.0);
    return vec3<f32>(ndc.x * P.proj.z * P.proj.w * d, ndc.y * P.proj.z * d, -d);
}

fn normal_at(q: vec2<i32>, scale: vec2<i32>, full_max: vec2<i32>, fallback: vec3<f32>) -> vec3<f32> {
    let fq = min(q * scale, full_max);
    let s = textureLoad(surface_tex, fq, 0);
    if s.a > 0.1 {
        return normalize((P.view_from_world * vec4<f32>(decode_world_normal(s), 0.0)).xyz);
    }
    return fallback;
}

@compute @workgroup_size(8, 8)
fn denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let max_q = vec2<i32>(P.size.xy) - vec2<i32>(1);
    let dc = depth_at(q);
    let centre = textureLoad(raw, q, 0);
    if dc >= SKY_DEPTH * 0.5 {
        textureStore(out_tex, q, centre);
        return;
    }
    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let pv = view_of(q, dc);
    let facing = normalize(-pv);
    let nc = normal_at(q, scale, full_max, facing);
    let plane_tol = 0.03 * dc + 0.015;

    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sq = clamp(q + vec2<i32>(dx, dy), vec2<i32>(0), max_q);
            let ds = depth_at(sq);
            if ds >= SKY_DEPTH * 0.5 {
                continue;
            }
            let rel = abs(ds - dc) / max(dc, 1e-3);
            if rel >= DEPTH_EDGE {
                continue;
            }
            let sv = view_of(sq, ds);
            let ns = normal_at(sq, scale, full_max, nc);
            let plane = abs(dot(sv - pv, nc));
            let gw = exp(-f32(dx * dx + dy * dy) / 4.5);
            let dw = 1.0 - rel / DEPTH_EDGE;
            let nw = pow(max(dot(nc, ns), 0.0), 16.0);
            let pw = exp(-plane * plane / (2.0 * plane_tol * plane_tol));
            let w = gw * dw * nw * pw;
            sum += textureLoad(raw, sq, 0) * w;
            total += w;
        }
    }
    let out = select(centre, sum / max(total, 1e-6), total > 1e-5);
    textureStore(out_tex, q, out);
}

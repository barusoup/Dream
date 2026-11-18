// Screen-space reflections, pass 3: edge-aware spatial filter. A 5x5 kernel
// whose radius grows with roughness (0 below roughness 0.05, about 3 texels at
// max_roughness); mirror-like pixels are copied unfiltered so sharp reflections
// stay sharp. Neighbours are weighted by a Gaussian, by linear-depth similarity
// and by normal similarity, and blended premultiplied by confidence.
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
// Temporal output (rgb colour, a confidence).
@group(0) @binding(1) var acc: texture_2d<f32>;
@group(0) @binding(2) var linear_depth: texture_2d<f32>;
@group(0) @binding(3) var surface_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

const SKY_DEPTH: f32 = 65000.0;
// Relative depth difference at which a neighbour stops contributing.
const DEPTH_EDGE: f32 = 0.1;

fn depth_at(q: vec2<i32>) -> f32 {
    let wq = clamp(q, vec2<i32>(0), vec2<i32>(P.size.xy) - vec2<i32>(1));
    let dmax = vec2<i32>(round(P.depth.xy)) - vec2<i32>(1);
    let dq = clamp(vec2<i32>(vec2<f32>(wq) * P.depth.xy * P.size.zw), vec2<i32>(0), dmax);
    return textureLoad(linear_depth, dq, 0).r;
}

fn surface_at(q: vec2<i32>) -> vec4<f32> {
    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    return textureLoad(surface_tex, min(q * scale, full_max), 0);
}

fn view_normal(s: vec4<f32>) -> vec3<f32> {
    let e = s.xy * 2.0 - vec2<f32>(1.0);
    let folded = abs(e.x) + abs(e.y) > 1.0;
    var o = e;
    if folded {
        let sign_e = vec2<f32>(select(-1.0, 1.0, e.x >= 0.0), select(-1.0, 1.0, e.y >= 0.0));
        o = (vec2<f32>(1.0) - abs(e.yx)) * sign_e;
    }
    let ny = (1.0 - abs(o.x) - abs(o.y)) * select(1.0, -1.0, folded);
    let world = normalize(vec3<f32>(o.x, ny, o.y));
    return normalize((P.view_from_world * vec4<f32>(world, 0.0)).xyz);
}

@compute @workgroup_size(8, 8)
fn spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let max_q = vec2<i32>(P.size.xy) - vec2<i32>(1);
    let centre = textureLoad(acc, q, 0);
    let surf = surface_at(q);
    let rough = surf.b;
    let max_rough = P.surface.x;
    let radius = 3.0 * clamp((rough - 0.05) / max(max_rough - 0.05, 1e-3), 0.0, 1.0);
    let dc = depth_at(q);
    if rough < 0.05 || radius < 0.5 || dc >= SKY_DEPTH * 0.5 || surf.a <= 0.1 {
        textureStore(out_tex, q, centre);
        return;
    }
    let nc = view_normal(surf);
    var sum_pm = vec3<f32>(0.0);
    var sum_c = 0.0;
    var sum_w = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let offset = vec2<i32>(round(vec2<f32>(f32(dx), f32(dy)) * radius * 0.5));
            let sq = clamp(q + offset, vec2<i32>(0), max_q);
            let ds = depth_at(sq);
            if ds >= SKY_DEPTH * 0.5 {
                continue;
            }
            let ss = surface_at(sq);
            var ns = nc;
            if ss.a > 0.1 {
                ns = view_normal(ss);
            }
            let s = textureLoad(acc, sq, 0);
            let w_gauss = exp(-0.25 * f32(dx * dx + dy * dy));
            let w_depth = clamp(1.0 - abs(ds - dc) / (dc * DEPTH_EDGE + 1e-3), 0.0, 1.0);
            let w_normal = pow(max(dot(nc, ns), 0.0), 16.0);
            let w = w_gauss * w_depth * w_normal;
            sum_pm += s.rgb * s.a * w;
            sum_c += s.a * w;
            sum_w += w;
        }
    }
    if sum_w <= 1e-6 {
        textureStore(out_tex, q, centre);
        return;
    }
    let conf = sum_c / sum_w;
    let rgb = select(vec3<f32>(0.0), sum_pm / max(sum_c, 1e-6), sum_c > 1e-6);
    textureStore(out_tex, q, vec4<f32>(rgb, conf));
}

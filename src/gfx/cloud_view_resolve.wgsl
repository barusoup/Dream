// Per-pixel view cloud resolve (gfx/cloud_view.rs). Writes the full-resolution
// history of this frame. Every pixel gathers a depth-class-aware 3x3 estimate
// around its continuous marched lattice and blends with clamped reprojected
// history. Marched pixels blend their exact sample for crisp 1:1 silhouettes.
// Prepended with atmo_common.wgsl and cloud_common.wgsl.

@group(0) @binding(0) var<uniform> vp: ViewParams;
@group(0) @binding(1) var low_colour: texture_2d<f32>;
@group(0) @binding(2) var low_dist: texture_2d<f32>;
@group(0) @binding(3) var hist_prev: texture_2d<f32>;
@group(0) @binding(4) var out_colour: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_dist: texture_storage_2d<rg32float, write>;
@group(0) @binding(6) var scene_depth: texture_2d<f32>;

const CLAMP_WIDEN: f32 = 1.15;
const HISTORY_WEIGHT_SKY: f32 = 0.92;
const HISTORY_WEIGHT_GEOM: f32 = 0.86;
const HISTORY_WEIGHT_MARKED: f32 = 0.72;

fn view_depth_at(q: vec2<u32>) -> f32 {
    let dims = vec2<u32>(textureDimensions(scene_depth));
    let full = vec2<u32>(vp.screen.xy);
    let t = min(q * dims / full, dims - 1u);
    return textureLoad(scene_depth, vec2<i32>(t), 0).r;
}

// Bilinear read of a full-resolution history texture at pixel coordinates pp.
fn hist_bilinear(tex: texture_2d<f32>, pp: vec2<f32>, full: vec2<i32>) -> vec4<f32> {
    let p0 = floor(pp);
    let f = pp - p0;
    let i0 = clamp(vec2<i32>(p0), vec2<i32>(0), full - 1);
    let i1 = clamp(vec2<i32>(p0) + 1, vec2<i32>(0), full - 1);
    let c00 = textureLoad(tex, vec2<i32>(i0.x, i0.y), 0);
    let c10 = textureLoad(tex, vec2<i32>(i1.x, i0.y), 0);
    let c01 = textureLoad(tex, vec2<i32>(i0.x, i1.y), 0);
    let c11 = textureLoad(tex, vec2<i32>(i1.x, i1.y), 0);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full = vec2<u32>(vp.screen.xy);
    if gid.x >= full.x || gid.y >= full.y {
        return;
    }
    let frame = u32(vp.state.x);
    var offs = array<vec2<u32>, 4>(vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u));
    let o = offs[frame % 4u];
    let marked = all((gid.xy % 2u) == o);
    let p = vec2<i32>(gid.xy);
    let full_i = vec2<i32>(full);
    let o_i = vec2<i32>(o);
    let z_now = view_depth_at(gid.xy);
    let sky_now = z_now >= CLOUD_SKY_DEPTH;

    // Continuous nearest-lattice centre removes tile-boundary discontinuities.
    let lo = vec2<i32>(round((vec2<f32>(p) - vec2<f32>(o_i)) * 0.5));
    let low_max = vec2<i32>(vec2<u32>(vp.res.zw)) - 1;

    var mn = vec4<f32>(1e30);
    var mx = vec4<f32>(-1e30);
    var mn_any = vec4<f32>(1e30);
    var mx_any = vec4<f32>(-1e30);
    var acc = vec4<f32>(0.0);
    var accw = 0.0;
    var acc_any = vec4<f32>(0.0);
    var accw_any = 0.0;
    var near_pair = vec2<f32>(CLOUD_NO_HIT, CLOUD_NO_HIT);
    var near_d2 = 1e30;
    var near_pair_any = vec2<f32>(CLOUD_NO_HIT, CLOUD_NO_HIT);
    var near_d2_any = 1e30;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let li = clamp(lo + vec2<i32>(dx, dy), vec2<i32>(0), low_max);
            let fp = li * 2 + o_i;
            if fp.x >= full_i.x || fp.y >= full_i.y {
                continue;
            }
            let c = textureLoad(low_colour, li, 0);
            let pair = textureLoad(low_dist, li, 0).xy;
            let same_class = (pair.y < 0.0) == !sky_now;
            let delta = vec2<f32>(fp - p);
            let d2 = dot(delta, delta);
            let gw = exp(-d2 / 2.2);

            mn_any = min(mn_any, c);
            mx_any = max(mx_any, c);
            acc_any += c * gw;
            accw_any += gw;
            if pair.x < CLOUD_NO_HIT * 0.5 && d2 < near_d2_any {
                near_d2_any = d2;
                near_pair_any = pair;
            }

            if same_class {
                mn = min(mn, c);
                mx = max(mx, c);
                acc += c * gw;
                accw += gw;
                if pair.x < CLOUD_NO_HIT * 0.5 && d2 < near_d2 {
                    near_d2 = d2;
                    near_pair = pair;
                }
            }
        }
    }

    if accw <= 1e-6 {
        mn = mn_any;
        mx = mx_any;
        acc = acc_any;
        accw = accw_any;
        near_pair = near_pair_any;
        near_d2 = near_d2_any;
    }

    let dir = cloud_view_ray(vp, vec2<f32>(gid.xy) + 0.5);
    let geom_dist = z_now / max(dot(dir, vp.fwd.xyz), 1e-3);
    var d_rep = 50000.0;
    if near_d2 < 1e30 {
        d_rep = near_pair.x;
    } else {
        let rb = vp.layer.x + vp.layer.y;
        let entry = cloud_sphere(cloud_planet_centre(vp.cam, vp.layer), dir, rb);
        if entry.y > 0.0 {
            d_rep = max(entry.x, 0.0);
        }
    }

    let colour_est = acc / max(accw, 1e-6);
    var colour = colour_est;
    var pair = near_pair;
    if marked {
        let own_idx = vec2<i32>(gid.xy / 2u);
        colour = textureLoad(low_colour, own_idx, 0);
        pair = textureLoad(low_dist, own_idx, 0).xy;
        mn = min(mn, colour);
        mx = max(mx, colour);
        if pair.x < CLOUD_NO_HIT * 0.5 {
            d_rep = pair.x;
        }
    }
    if !sky_now {
        d_rep = min(d_rep, geom_dist);
        pair.y = -abs(pair.y);
    } else {
        pair.y = abs(pair.y);
    }

    // Reprojection: blend clamped history at full resolution.
    if vp.state.y < 0.5 {
        let world = vp.cam.xyz + dir * d_rep;
        let clip = vp.prev_vp * vec4<f32>(world, 1.0);
        if clip.w > 0.0 {
            let ndc = clip.xy / clip.w;
            let pp = (ndc * 0.5 + 0.5) * vec2<f32>(full) - 0.5;
            let inside = all(pp >= vec2<f32>(-0.5)) && all(pp <= vec2<f32>(full) - 0.5);
            if inside {
                let mid = (mn + mx) * 0.5;
                let range = max((mx - mn) * CLAMP_WIDEN, vec4<f32>(0.015));
                let hist = clamp(hist_bilinear(hist_prev, pp, full_i), mid - range, mid + range);
                let base_w = select(HISTORY_WEIGHT_SKY, HISTORY_WEIGHT_GEOM, !sky_now);
                let w = select(base_w, HISTORY_WEIGHT_MARKED, marked);
                colour = mix(colour, hist, w);
            }
        }
    }

    textureStore(out_colour, p, colour);
    textureStore(out_dist, p, vec4<f32>(pair, 0.0, 0.0));
}

// Indirect chain, pass 2: horizon march for ambient occlusion (AO) and one
// bounce of indirect diffuse light (GI), after Therrien et al. 2023, "Screen
// Space Indirect Lighting with Visibility Bitmask".
//
// Output: rgb = indirect diffuse radiance (before the forward shader's GI
// intensity), a = ambient visibility in [0, 1] (before its AO intensity).
// Both are in view space's frame of the current scene pass.

// Mirror of `IndirectParams` (176 bytes); see `indirect_denoise.wgsl`.
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
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
// Full-resolution scene surface: rg octahedral world normal (0..1), b roughness,
// a class (0 no surface, 1 dielectric, 2/3 metal).
@group(0) @binding(2) var surface_tex: texture_2d<f32>;
@group(0) @binding(3) var colour_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const SKY_DEPTH: f32 = 65000.0;
// Soft-knee firefly ceiling so bright emissives keep bounce energy without
// single-texel blowouts.
const FIREFLY_LIMIT: f32 = 24.0;
const FIREFLY_KNEE: f32 = 12.0;

// Inverse of `surface_encode` (lighting.wgsl): octahedral unwrap, with the
// lower hemisphere folded into the corners of the square. A folded point has
// |e|_1 > 1.
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

// Depth texel that working texel q reads: its top-left texel in the shared
// full-resolution depth (`march.zw` is depth texels per working texel). Clamped
// to the chain's edges.
fn depth_texel(q: vec2<i32>) -> vec2<i32> {
    let wq = clamp(q, vec2<i32>(0), vec2<i32>(P.size.xy) - vec2<i32>(1));
    let dmax = vec2<i32>(round(P.size.xy * P.march.zw)) - vec2<i32>(1);
    return clamp(vec2<i32>(vec2<f32>(wq) * P.march.zw), vec2<i32>(0), dmax);
}

// View-space position of depth texel `dq` at linear depth `d`. The position is
// taken at the centre of the same texel that `d` was read from, so a point and
// its depth describe one surface point. The depth texture's own size is used
// (it is the working size times `march.zw`, not the render size). uv is
// bottom-up (`uv = ndc * 0.5 + 0.5`), so +y in texels is +y in view space.
fn view_at_texel(dq: vec2<i32>, d: f32) -> vec3<f32> {
    let depth_size = P.size.xy * P.march.zw;
    let ndc = (vec2<f32>(dq) + vec2<f32>(0.5)) / depth_size * 2.0 - vec2<f32>(1.0);
    return vec3<f32>(ndc.x * P.proj.z * P.proj.w * d, ndc.y * P.proj.z * d, -d);
}

// Linear depth of working texel q (see `depth_texel`).
fn depth_at(q: vec2<i32>) -> f32 {
    return textureLoad(linear_depth, depth_texel(q), 0).r;
}

// View-space position of working texel q with linear depth d.
fn view_of(q: vec2<i32>, d: f32) -> vec3<f32> {
    return view_at_texel(depth_texel(q), d);
}

fn world_to_view_normal(n: vec3<f32>) -> vec3<f32> {
    return normalize((P.view_from_world * vec4<f32>(n, 0.0)).xyz);
}

// Normal from depth neighbours, for pixels whose surface class is 0. Each
// axis uses the neighbour on the side with the closer depth.
fn reconstruct_normal(q: vec2<i32>, d: f32, pv: vec3<f32>, facing: vec3<f32>) -> vec3<f32> {
    let dl = depth_at(q - vec2<i32>(1, 0));
    let dr = depth_at(q + vec2<i32>(1, 0));
    let du = depth_at(q - vec2<i32>(0, 1));
    let dd = depth_at(q + vec2<i32>(0, 1));
    let pl = view_of(q - vec2<i32>(1, 0), dl);
    let pr = view_of(q + vec2<i32>(1, 0), dr);
    let pu = view_of(q - vec2<i32>(0, 1), du);
    let pd = view_of(q + vec2<i32>(0, 1), dd);
    let dx = select(pr - pv, pv - pl, abs(dl - d) < abs(dr - d));
    let dy = select(pd - pv, pv - pu, abs(du - d) < abs(dd - d));
    var n = cross(dx, dy);
    if length(n) < 1e-12 {
        return facing;
    }
    n = normalize(n);
    if dot(n, facing) < 0.0 {
        n = -n;
    }
    return n;
}

// Fast minimax atan2 approximation (max error < 2e-4 rad, ~1/500th of a sector).
fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let mn = min(ax, ay);
    let mx = max(ax, ay);
    let a = mn / max(mx, 1e-12);
    let s = a * a;
    var r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if ay > ax {
        r = 1.57079637 - r;
    }
    if x < 0.0 {
        r = 3.14159274 - r;
    }
    if y < 0.0 {
        r = -r;
    }
    return r;
}

// Signed angle of `v` from `facing` in the slice plane spanned by `facing` and
// the unit vector `in_u`, positive toward `in_u`.
fn slice_angle(v: vec3<f32>, facing: vec3<f32>, in_u: vec3<f32>) -> f32 {
    return fast_atan2(dot(v, in_u), dot(v, facing));
}

// Interleaved gradient noise (Jimenez 2014).
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Bits covering the sectors of [lo, hi] (fractions of the hemisphere) in a
// 32-sector bitmask. Sector i covers [i/32, (i+1)/32).
fn sector_mask(lo: f32, hi: f32) -> u32 {
    // Round both ends to the nearest sector boundary. Rounding the span up
    // (floor/ceil) makes every sample cover at least one whole sector however
    // thin it is, which adds a constant occlusion per sample and defeats the
    // distance falloff; nearest-boundary rounding is unbiased.
    let start = u32(clamp(floor(lo * 32.0 + 0.5), 0.0, 32.0));
    let end = u32(clamp(floor(hi * 32.0 + 0.5), 0.0, 32.0));
    if end <= start {
        return 0u;
    }
    let count = end - start;
    return (0xFFFFFFFFu >> (32u - count)) << start;
}

// Occluded sectors of one occluder arc, from the sample's horizon `h_front`
// (the occluder's near surface, at distance `d_front`) to its far side `h_back`
// (at `d_back`). Along the arc the ray distance is interpolated, and a direction
// is occluded with the reference's ray weight `1 - (d / radius)^2`.
//
// A sector is occluded when its threshold, fract(seed + k * golden), is below
// that weight. `seed` is one draw per pixel and slice, shared by every sample of
// the slice, so a sector is occluded with probability equal to the largest weight
// of any arc covering it. That is the weight of the nearest occluder along those
// directions, which is what a ray gets in the reference, rather than a full
// sector for every sample that reaches it.
fn arc_weight_mask(h_front: f32, h_back: f32, d_front: f32, d_back: f32, radius: f32, seed: f32) -> u32 {
    let lo = min(h_front, h_back);
    let hi = max(h_front, h_back);
    let start = u32(clamp(floor(lo * 32.0), 0.0, 32.0));
    let end = u32(clamp(ceil(hi * 32.0), 0.0, 32.0));
    var bits = 0u;
    let span = h_back - h_front;
    for (var k = start; k < end; k++) {
        let along = select(clamp(((f32(k) + 0.5) / 32.0 - h_front) / span, 0.0, 1.0), 0.0, abs(span) < 1e-6);
        let d = mix(d_front, d_back, along) / radius;
        let w = clamp(1.0 - d * d, 0.0, 1.0);
        if fract(seed + f32(k) * 0.6180339887) < w {
            bits = bits | (1u << k);
        }
    }
    return bits;
}

// Cosine-lobe weight of the sectors set in `bits`, as a fraction of the
// hemisphere. Sector i spans the angles -90 deg + i * 5.625 deg from the
// projected normal, so bands [0,8), [8,16), [16,24), [24,32) cover 45 deg each
// and their weights are the integral of cos over the band divided by 2. The
// weights sum to 1, so a full mask gives 1. Mirror: `band_weight` in indirect.rs.
fn sector_weight(k: u32, n_angle: f32) -> f32 {
    let beta = -PI * 0.5 + (f32(k) + 0.5) * PI / 32.0;
    return cos(beta) * abs(sin(beta + n_angle));
}

fn sector_bits_weight(bits: u32, n_angle: f32) -> f32 {
    var acc = 0.0;
    var b = bits;
    while b != 0u {
        let k = countTrailingZeros(b);
        acc += sector_weight(k, n_angle);
        b = b & (b - 1u);
    }
    return acc;
}

fn band_weight(bits: u32) -> f32 {
    let b0 = f32(countOneBits(bits & 0x000000FFu)) * 0.146446609;
    let b1 = f32(countOneBits(bits & 0x0000FF00u)) * 0.353553391;
    let b2 = f32(countOneBits(bits & 0x00FF0000u)) * 0.353553391;
    let b3 = f32(countOneBits(bits & 0xFF000000u)) * 0.146446609;
    return (b0 + b1 + b2 + b3) / 8.0;
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if l < 1e-12 {
        return fallback;
    }
    return v / l;
}

@compute @workgroup_size(8, 8)
fn horizon(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let qf = vec2<f32>(q);
    let d = depth_at(q);
    if d >= SKY_DEPTH * 0.5 {
        textureStore(out_tex, q, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let scale = vec2<i32>(round(P.full.xy / P.size.xy));
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let fq = min(q * scale, full_max);
    let pv = view_of(q, d);
    let facing = normalize(-pv);
    let surf = textureLoad(surface_tex, fq, 0);
    var n = facing;
    if surf.a > 0.1 {
        n = world_to_view_normal(decode_world_normal(surf));
    } else {
        n = reconstruct_normal(q, d, pv, facing);
    }

    let ao_radius = P.ao.x;
    let thickness = P.ao.y;
    let ao_on = P.ao.z > 0.5;
    let gi_radius = P.gi.x;
    let gi_on = P.gi.y > 0.5;
    let slices = max(u32(P.gi.z + 0.5), 1u);
    let steps = max(u32(P.gi.w + 0.5), 1u);

    // Per-frame animated noise: temporal accumulation averages it out.
    let frame_offset = 5.588238 * f32(u32(P.march.x) % 64u);
    let noise_p = qf + vec2<f32>(frame_offset);
    let noise1 = ign(noise_p);
    let noise2 = ign(noise_p + vec2<f32>(17.0, 9.0));

    // Screen-space march length: allow up to 65% of vertical resolution so
    // `gi_radius` (6 m) still reaches nearby courtyard walls and pillars.
    let projection_scale = 0.5 * P.size.y / P.proj.z;
    let radius_px = clamp(max(ao_radius, gi_radius) * projection_scale / d,
        8.0, max(8.0, 0.65 * P.size.y));

    let active_radius = max(select(0.0, ao_radius, ao_on), select(0.0, gi_radius, gi_on));
    let active_radius_sq = active_radius * active_radius;

    var visible = 0.0;
    var visible_norm = 0.0;
    var gi_acc = vec3<f32>(0.0);
    for (var s = 0u; s < slices; s++) {
        let phi = (f32(s) + noise1) / f32(slices) * PI;
        let omega = vec2<f32>(cos(phi), sin(phi));
        let dir_slice = vec3<f32>(omega, 0.0);
        // The slice plane spans `facing` (the direction to the eye) and `in_u`.
        // Every horizon angle is the signed angle from `facing` toward `in_u`,
        // taken from the sample vector itself.
        let in_u = safe_normalize(dir_slice - facing * dot(dir_slice, facing), vec3<f32>(0.0, 1.0, 0.0));
        let n_angle = slice_angle(n, facing, in_u);
        let slice_len = length(vec2<f32>(dot(n, facing), dot(n, in_u)));

        let step_a = PI / 32.0;
        let dc_a = cos(step_a);
        let ds_a = sin(step_a);
        let b0 = -PI * 0.5 + 0.5 * step_a;
        var cb = cos(b0);
        var sb = sin(b0);
        let cn = cos(n_angle);
        let sn = sin(n_angle);
        var slice_total = 0.0;
        for (var k = 0u; k < 32u; k++) {
            slice_total += cb * abs(sb * cn + cb * sn);
            let next_cb = cb * dc_a - sb * ds_a;
            let next_sb = sb * dc_a + cb * ds_a;
            cb = next_cb;
            sb = next_sb;
        }
        let inv_slice_total = 1.0 / max(slice_total, 1e-6);

        var occ_ao = 0u;
        var occ_gi = 0u;
        var gi_slice = vec3<f32>(0.0);
        let ao_seed = ign(noise_p + vec2<f32>(f32(s) * 41.0, 7.0));
        for (var side = -1.0; side <= 1.0; side += 2.0) {
            for (var j = 0u; j < steps; j++) {
                let t0 = (f32(j) + noise2) / f32(steps);
                let t = pow(t0, 1.55);
                let step_px = max(t * radius_px, 1.0 + f32(j));
                let sq = qf + vec2<f32>(0.5) + side * omega * step_px;
                let sqi = vec2<i32>(floor(sq));
                if sqi.x < 0 || sqi.y < 0 || f32(sqi.x) >= P.size.x || f32(sqi.y) >= P.size.y {
                    break;
                }
                let sd = depth_at(sqi);
                if sd >= SKY_DEPTH * 0.5 {
                    continue;
                }
                let sv = view_of(sqi, sd);
                let dv = sv - pv;
                let dist_sq = dot(dv, dv);
                if dist_sq >= active_radius_sq {
                    continue;
                }
                let dist = sqrt(dist_sq);
                if dot(dv, n) <= 0.01 * dist {
                    continue;
                }
                let dir = dv / max(dist, 1e-6);
                let back_v = dv + sv * (thickness * inverseSqrt(max(dot(sv, sv), 1e-12)));
                let angles = vec2<f32>(slice_angle(dv, facing, in_u), slice_angle(back_v, facing, in_u));
                let h = clamp((angles - vec2<f32>(n_angle) + vec2<f32>(PI * 0.5)) / PI,
                    vec2<f32>(0.0), vec2<f32>(1.0));
                let bits = sector_mask(min(h.x, h.y), max(h.x, h.y));

                if gi_on && dist < gi_radius {
                    let newly = bits & ~occ_gi;
                    if newly != 0u {
                        let sfq = min(sqi * scale, full_max);
                        var radiance = textureLoad(colour_tex, sfq, 0).rgb;
                        let lum = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                        if lum > FIREFLY_KNEE {
                            let head = FIREFLY_LIMIT - FIREFLY_KNEE;
                            let soft = FIREFLY_KNEE + head * (1.0 - exp(-(lum - FIREFLY_KNEE) / head));
                            radiance = radiance * (soft / max(lum, 1e-4));
                        }
                        let ss = textureLoad(surface_tex, sfq, 0);
                        var ns = -dir;
                        if ss.a > 0.1 {
                            ns = world_to_view_normal(decode_world_normal(ss));
                        }
                        let dist_fade = 1.0 - smoothstep(0.65 * gi_radius, gi_radius, dist);
                        let w = sector_bits_weight(newly, n_angle)
                            * inv_slice_total
                            * dist_fade
                            * clamp(dot(ns, -dir) * 4.0, 0.0, 1.0);
                        gi_slice += radiance * w;
                    }
                    occ_gi = occ_gi | bits;
                }
                if ao_on && dist < ao_radius {
                    let far_dist = max(dist, length(back_v));
                    occ_ao = occ_ao | arc_weight_mask(h.x, h.y, dist, far_dist, ao_radius, ao_seed);
                }
            }
        }
        if ao_on {
            if occ_ao == 0u {
                visible += slice_len;
            } else if occ_ao != 0xFFFFFFFFu {
                var cb_o = cos(b0);
                var sb_o = sin(b0);
                var slice_occ = 0.0;
                for (var k = 0u; k < 32u; k++) {
                    if ((occ_ao >> k) & 1u) == 1u {
                        slice_occ += cb_o * abs(sb_o * cn + cb_o * sn);
                    }
                    let next_cb = cb_o * dc_a - sb_o * ds_a;
                    let next_sb = sb_o * dc_a + cb_o * ds_a;
                    cb_o = next_cb;
                    sb_o = next_sb;
                }
                visible += slice_len * (1.0 - slice_occ * inv_slice_total);
            }
        }
        if gi_on {
            gi_acc += gi_slice * slice_len;
        }
        visible_norm += slice_len;
    }

    var vis = visible / max(visible_norm, 1e-6);
    if !ao_on {
        vis = 1.0;
    }
    let gi_out = gi_acc / max(visible_norm, 1e-6);
    textureStore(out_tex, q, vec4<f32>(gi_out, clamp(vis, 0.0, 1.0)));
}

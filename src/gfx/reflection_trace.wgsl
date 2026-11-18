// Screen-space reflections, pass 1: trace. Per working texel, a reflection
// direction (mirror, or a visible-normal GGX sample for rough surfaces) is
// marched through the linear depth of the scene pass, and the lit colour of the
// hit is gathered. Output: rgb = gathered colour (0 on a miss), a = confidence.
//
// Mirror of `ReflectionParams` (192 bytes) in reflection.rs; see that file for
// the field meanings. Every block is a vec4 so there is no padding to track.

struct ReflectionParams {
    size: vec4<f32>,         // working width, height, 1/width, 1/height
    full: vec4<f32>,         // render width, height, 1/width, 1/height
    depth: vec4<f32>,        // linear depth width, height, 1/width, 1/height
    proj: vec4<f32>,         // near, far, tan(fov_y / 2), aspect
    view_from_world: mat4x4<f32>,
    trace: vec4<f32>,        // max distance (m), thickness (m), steps, refinement steps
    surface: vec4<f32>,      // max roughness, edge fade, intensity, unused
    frame: vec4<f32>,        // frame index (0 unless animated), history reset, unused
    temporal: vec4<f32>,     // current weight, relative depth tolerance, unused
};

@group(0) @binding(0) var<uniform> P: ReflectionParams;
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
// Full-resolution scene surface: rg octahedral world normal, b roughness, a class.
@group(0) @binding(2) var surface_tex: texture_2d<f32>;
// Full-resolution lit colour of frame N.
@group(0) @binding(3) var colour_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const SKY_DEPTH: f32 = 65000.0;
// Soft-knee firefly ceiling so bright highlights stay vivid without single-texel blowouts.
const FIREFLY_LIMIT: f32 = 24.0;
const FIREFLY_KNEE: f32 = 12.0;
// The two random numbers are kept in the central ~85% of the lobe.
const VNDF_LO: f32 = 0.075;
const VNDF_HI: f32 = 0.925;

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

// View-space position of working texel centre `q` (float index) at linear depth `d`.
fn view_at(q: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = (q + vec2<f32>(0.5)) * P.size.zw * 2.0 - vec2<f32>(1.0);
    return vec3<f32>(ndc.x * P.proj.z * P.proj.w * d, ndc.y * P.proj.z * d, -d);
}

// Continuous working-texel position of a view-space point (texel q covers [q, q+1)).
fn view_to_texel(v: vec3<f32>) -> vec2<f32> {
    let ndc = vec2<f32>(v.x / (-v.z * P.proj.z * P.proj.w), v.y / (-v.z * P.proj.z));
    return (ndc * 0.5 + 0.5) * P.size.xy;
}

// Linear depth at the working-texel-space point `sp` (texel q covers [q, q+1)):
// the depth texel that contains the same point, so a ray sample and the depth it
// is compared with describe the same place on the screen.
fn depth_at_point(sp: vec2<f32>) -> f32 {
    let dmax = vec2<i32>(round(P.depth.xy)) - vec2<i32>(1);
    let dq = clamp(vec2<i32>(floor(sp * P.depth.xy * P.size.zw)), vec2<i32>(0), dmax);
    return textureLoad(linear_depth, dq, 0).r;
}

// Full-resolution texel containing the working-texel-space point `sp`.
fn full_texel_at_point(sp: vec2<f32>) -> vec2<i32> {
    let full_max = vec2<i32>(P.full.xy) - vec2<i32>(1);
    return clamp(vec2<i32>(floor(sp * P.full.xy * P.size.zw)), vec2<i32>(0), full_max);
}

// Surface at the working-texel-space point `sp`.
fn surface_at_point(sp: vec2<f32>) -> vec4<f32> {
    return textureLoad(surface_tex, full_texel_at_point(sp), 0);
}

fn world_to_view_normal(n: vec3<f32>) -> vec3<f32> {
    return normalize((P.view_from_world * vec4<f32>(n, 0.0)).xyz);
}

// Interleaved gradient noise (Jimenez 2014).
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Reflection direction for a GGX lobe of roughness alpha (= roughness^2) around
// `n`, for a view vector `v` pointing at the viewer (both unit, view space).
// The half vector is a visible-normal sample (Heitz 2018); the direction is the
// mirror of `v` about it, falling back to the mirror about `n` when the sample
// lies below the surface. Mirror: `reflection_direction` in reflection.rs.
fn reflection_direction(v: vec3<f32>, n: vec3<f32>, alpha: f32, u1_in: f32, u2_in: f32) -> vec3<f32> {
    // Orthonormal basis around n (Duff et al. 2017).
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let tb = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bb = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    let ve = vec3<f32>(dot(v, tb), dot(v, bb), dot(v, n));
    // Sample the visible normals in the local frame where n is +z.
    let vh = normalize(vec3<f32>(alpha * ve.x, alpha * ve.y, ve.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if lensq > 0.0 {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
    }
    let t2 = cross(vh, t1);
    let u1 = clamp(u1_in, VNDF_LO, VNDF_HI);
    let u2 = clamp(u2_in, VNDF_LO, VNDF_HI);
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let hs = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - hs) * sqrt(1.0 - p1 * p1) + hs * p2;
    let p3 = sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
    let nh = p1 * t1 + p2 * t2 + p3 * vh;
    let he = normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
    let h = he.x * tb + he.y * bb + he.z * n;
    var dir = reflect(-v, h);
    if dot(dir, n) <= 0.0 {
        dir = reflect(-v, n);
    }
    return normalize(dir);
}

// True when a depth sample is a surface the ray can hit (not sky).
fn is_surface_depth(d: f32) -> bool {
    return d < SKY_DEPTH * 0.5;
}

// True when the ray, at working-texel-space point `sp` and ray depth `ray_d`,
// is behind the scene surface it lands on.
fn ray_behind(sp: vec2<f32>, ray_d: f32) -> bool {
    let sd = depth_at_point(sp);
    return is_surface_depth(sd) && sd < ray_d;
}

// True when the ray is no more than the thickness allowance behind the surface.
fn within_thickness(sp: vec2<f32>, ray_d: f32) -> bool {
    let sd = depth_at_point(sp);
    return ray_d - sd < P.trace.y * (1.0 + 0.02 * ray_d);
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if f32(gid.x) >= P.size.x || f32(gid.y) >= P.size.y {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let qf = vec2<f32>(q) + vec2<f32>(0.5);
    let d = depth_at_point(qf);
    let surf = surface_at_point(qf);
    let rough = surf.b;
    let max_rough = P.surface.x;
    if !is_surface_depth(d) || surf.a <= 0.1 || rough >= max_rough {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    let pv = view_at(qf, d);
    let view_dir = normalize(-pv);
    var n = world_to_view_normal(decode_world_normal(surf));
    if dot(n, view_dir) < 0.0 {
        n = -n;
    }
    let alpha = rough * rough;
    // Noise animates with the frame only when temporal filtering runs.
    let frame_offset = 5.588238 * P.frame.x;
    let noise1 = ign(qf + vec2<f32>(frame_offset));
    let noise2 = ign(qf + vec2<f32>(frame_offset) + vec2<f32>(17.0, 9.0));
    var dir = reflect(-view_dir, n);
    if alpha >= 0.005 {
        dir = reflection_direction(view_dir, n, alpha, noise1, noise2);
    }
    // Far end of the ray, clipped in front of the near plane.
    let near = P.proj.x;
    var len = P.trace.x;
    if dir.z > 0.0 {
        len = min(len, max((-near - pv.z) / dir.z, 0.0));
    }
    if len <= 1e-4 {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    let far_pt = pv + dir * len;
    // Start one working texel's footprint off the surface toward the viewer, so
    // the ray does not hit its own surface and does not skip nearby hits.
    let start = pv + n * (2.0 * d * P.proj.z / P.size.y);
    let inv0 = 1.0 / -start.z;
    let inv1 = 1.0 / -far_pt.z;
    let s0 = view_to_texel(start);
    let s1 = view_to_texel(far_pt);
    let seg = s1 - s0;
    let seg_len = length(seg);
    if seg_len < 0.5 {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    let steps = u32(P.trace.z);
    let refine = u32(P.trace.w);
    // Samples are denser near the start: t = (i + jitter)^2 / steps^2. Far steps
    // can be many texels long, so a hit is the crossing from in front of a surface
    // to behind it, found by refinement. A sample can sit farther behind a thin
    // surface than the thickness allowance when the step is long, and the crossing
    // is still the hit; the allowance is checked at the refined hit.
    // Keep mirror rays (alpha < 0.005) free of per-frame ray-march jitter when
    // enough steps are available so crisp mirror reflections never shimmer.
    let jitter = select(noise2, 0.5, alpha < 0.005 && steps >= 24u);
    var prev_t = 0.0;
    var prev_front = true;
    var hit = false;
    var lo = 0.0;
    var hi = 0.0;
    for (var i = 0u; i < steps; i++) {
        let tt = (f32(i) + jitter) / f32(steps);
        let t = max(tt * tt, prev_t + 1.0 / seg_len);
        if t > 1.0 {
            break;
        }
        let sp = s0 + seg * t;
        if any(sp < vec2<f32>(0.0)) || any(sp >= P.size.xy) {
            break;
        }
        let behind = ray_behind(sp, 1.0 / mix(inv0, inv1, t));
        if behind && prev_front {
            hit = true;
            lo = prev_t;
            hi = t;
            break;
        }
        prev_front = !behind;
        prev_t = t;
    }
    if !hit {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    // Binary refinement between the last sample in front and the first behind.
    for (var k = 0u; k < refine; k++) {
        let mid = 0.5 * (lo + hi);
        if ray_behind(s0 + seg * mid, 1.0 / mix(inv0, inv1, mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let hit_t = hi;
    let hit_sp = s0 + seg * hit_t;
    if !within_thickness(hit_sp, 1.0 / mix(inv0, inv1, hit_t)) {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    let hit_surf = surface_at_point(hit_sp);
    if hit_surf.a <= 0.1 {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    // The hit surface must face the ray; its back is not a reflection.
    let hit_n = world_to_view_normal(decode_world_normal(hit_surf));
    if dot(hit_n, dir) > 0.0 {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    // Distance along the 3D ray: the screen parameter maps to the ray parameter
    // through the ray depth at the hit (perspective-correct).
    let ray_d = 1.0 / mix(inv0, inv1, hit_t);
    let tau = clamp(hit_t * ray_d / -far_pt.z, 0.0, 1.0);
    let dist = tau * length(far_pt - start);
    let hit_uv = hit_sp * P.size.zw;
    let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    var c_edge = 1.0;
    if P.surface.y > 0.0 {
        c_edge = smoothstep(0.0, P.surface.y, edge);
    }
    let max_dist = P.trace.x;
    let c_rough = clamp((max_rough - rough) / 0.15, 0.0, 1.0);
    let c_dist = 1.0 - smoothstep(0.8 * max_dist, max_dist, dist);
    let c_view = 1.0 - smoothstep(0.6, 0.95, dot(dir, view_dir));
    let conf = c_rough * c_edge * c_dist * c_view;
    if conf <= 0.0 {
        textureStore(out_tex, q, vec4<f32>(0.0));
        return;
    }
    // Bilinear surface-aware fetch from full-resolution `colour_tex` at `hit_sp`.
    let fc = hit_sp * P.full.xy * P.size.zw - vec2<f32>(0.5);
    let f0 = floor(fc);
    let ff = fc - f0;
    let fmax = vec2<i32>(P.full.xy) - vec2<i32>(1);
    let i0 = clamp(vec2<i32>(f0), vec2<i32>(0), fmax);
    let i1 = clamp(vec2<i32>(f0) + vec2<i32>(1), vec2<i32>(0), fmax);
    let q00 = vec2<i32>(i0.x, i0.y);
    let q10 = vec2<i32>(i1.x, i0.y);
    let q01 = vec2<i32>(i0.x, i1.y);
    let q11 = vec2<i32>(i1.x, i1.y);
    let w00 = (1.0 - ff.x) * (1.0 - ff.y) * select(0.0, 1.0, textureLoad(surface_tex, q00, 0).a > 0.1);
    let w10 = ff.x * (1.0 - ff.y) * select(0.0, 1.0, textureLoad(surface_tex, q10, 0).a > 0.1);
    let w01 = (1.0 - ff.x) * ff.y * select(0.0, 1.0, textureLoad(surface_tex, q01, 0).a > 0.1);
    let w11 = ff.x * ff.y * select(0.0, 1.0, textureLoad(surface_tex, q11, 0).a > 0.1);
    let wsum = w00 + w10 + w01 + w11;
    var col = textureLoad(colour_tex, full_texel_at_point(hit_sp), 0).rgb;
    if wsum > 1e-4 {
        col = (textureLoad(colour_tex, q00, 0).rgb * w00
            + textureLoad(colour_tex, q10, 0).rgb * w10
            + textureLoad(colour_tex, q01, 0).rgb * w01
            + textureLoad(colour_tex, q11, 0).rgb * w11) / wsum;
    }
    let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    if lum > FIREFLY_KNEE {
        let head = FIREFLY_LIMIT - FIREFLY_KNEE;
        let soft = FIREFLY_KNEE + head * (1.0 - exp(-(lum - FIREFLY_KNEE) / head));
        col = col * (soft / max(lum, 1e-4));
    }
    textureStore(out_tex, q, vec4<f32>(col, conf));
}

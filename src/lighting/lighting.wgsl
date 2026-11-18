// Shared PBR lighting — function-only snippet, concatenated by the Rust side
// (see crate::lighting) ahead of the shader that consumes it. Contract: the
// environment uniform lives at set 3 binding 0 for every pipeline that uses
// this module, so one descriptor set serves models and impostors alike.

struct LightingEnv {
    sun_direction: vec4<f32>,   // xyz: direction TO the sun (normalized)
    sun_color: vec4<f32>,       // rgb: colour * intensity
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_ground: vec4<f32>,
    cam_pos: vec4<f32>,         // xyz: world-space camera position
    params: vec4<f32>,          // x: exposure, y: 1 when the sky colours are sun-driven irradiance
};
@group(3) @binding(0) var<uniform> lighting_env: LightingEnv;

// Per-frame lighting state written by `Renderer::render_scene`. A zeroed block
// (shadow.x == 0) means shadows are off.
struct LightingFrame {
    cascade_view_proj: array<mat4x4<f32>, 4>, // world -> light clip, per cascade
    cascade_params: array<vec4<f32>, 4>,      // x: texel world size, y: depth range (m)
    shadow: vec4<f32>,      // x: cascade count (0 = off), y: 1/resolution, z: filter (0 hard, 1 pcf, 2 soft), w: taps
    shadow2: vec4<f32>,     // x: softness, y: normal bias (texels), z: strength, w: debug cascades (0/1)
    shadow_fade: vec4<f32>, // x: fade start (m), y: fade end (m)
    screen: vec4<f32>,      // render width, height, 1/width, 1/height
    features: vec4<f32>,    // x: ao strength, y: gi strength, z: reflection strength, w: frame index
    prev_view_proj: mat4x4<f32>, // previous unjittered view-projection
    indirect: vec4<f32>,    // x: multi-bounce AO, y: specular occlusion (1 on), z: env map valid, w: its last mip
    cloud_shadow: vec4<f32>, // x, y: map centre (world xz), z: 1 / extent, w: strength (0 off)
    cloud_layer: vec4<f32>,  // x, y: cloud layer bottom and top (world y)
};
@group(3) @binding(1) var<uniform> lighting_frame: LightingFrame;
// Cascaded sun shadow map: one depth layer per cascade. Uv follows the scene
// targets' convention (`screen_uv` in the model shaders): uv.y = ndc.y / 2 + 0.5.
@group(3) @binding(2) var lighting_shadow_map: texture_depth_2d_array;
// Comparison sampler with a white border, so anything outside a map is lit.
@group(3) @binding(3) var lighting_shadow_sampler: sampler_comparison;
// Indirect diffuse (rgb) and ambient visibility (a) of the previous frame, in
// the scene's bottom-up uv convention. Written by the indirect chain.
@group(3) @binding(4) var lighting_gi_ao: texture_2d<f32>;
// Traced screen-space reflection of the previous frame: radiance (rgb) and
// confidence (a). Written by the reflection chain (see reflection.rs).
@group(3) @binding(5) var lighting_reflection: texture_2d<f32>;
@group(3) @binding(6) var lighting_linear: sampler;
// Linear view depth (metres) of the frame that produced `lighting_gi_ao`.
@group(3) @binding(7) var lighting_feedback_depth: texture_2d<f32>;
// Sky environment map of the previous frame (see atmosphere.rs, `sky_env_uv`):
// radiance over the sphere, with a full mip chain. Valid when
// `lighting_frame.indirect.z` is 1; `indirect.w` is its last mip. Repeats in u,
// clamps in v, trilinear.
@group(3) @binding(8) var lighting_env_map: texture_2d<f32>;
@group(3) @binding(9) var lighting_env_sampler: sampler;
// Sun transmittance through the cloud layer on the ground (R32F, one per frame
// slot; see gfx/cloud_shadow.rs). Sampled with binding 6.
@group(3) @binding(10) var lighting_cloud_shadow: texture_2d<f32>;

// Fraction of the sun that reaches a world point through the clouds, from the
// shadow map. Points outside the map, and points above the layer top, get 1;
// the shadow fades out between the layer bottom and top.
fn lighting_cloud_visibility(world: vec3<f32>) -> f32 {
    let s = lighting_frame.cloud_shadow;
    if s.w <= 0.0 {
        return 1.0;
    }
    let uv = (world.xz - s.xy) * s.z + vec2<f32>(0.5);
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return 1.0;
    }
    let cloud_t = textureSampleLevel(lighting_cloud_shadow, lighting_linear, uv, 0.0).r;
    let layer = lighting_frame.cloud_layer;
    // The map is referenced to the ground, so it applies below the layer base and
    // fades out over the 50 m above it: a point inside or above the layer is not
    // shadowed by the ground transmittance.
    let fade = 1.0 - smoothstep(layer.x, layer.x + 50.0, world.y);
    return mix(1.0, cloud_t, s.w * fade);
}

// Pixel position of the fragment being shaded, set by `lighting_begin`. Shaders
// that never call it see (0, 0), which only changes the noise pattern.
var<private> lighting_pixel: vec2<f32>;

// Call first in a fragment entry point with its `@builtin(position).xy`.
fn lighting_begin(frag_coord: vec2<f32>) {
    lighting_pixel = frag_coord;
}

const PI: f32 = 3.14159265359;
const SHADOW_TAU: f32 = 6.28318530718;
// Fraction of a cascade's map, measured from its edge, over which it hands
// over to the next, larger cascade.
const SHADOW_EDGE_BLEND: f32 = 0.1;
// Depth margin (normalised) a blocker must clear below the receiver plane.
const SHADOW_BLOCKER_EPS: f32 = 1e-4;
// Largest caster-receiver gap, in metres, the soft filter's blocker search is
// sized for: its radius is the penumbra such a gap would produce.
const SHADOW_SEARCH_GAP: f32 = 12.0;
// Limit on the receiver-plane slope, depth per uv: grazing planes stay finite.
const SHADOW_MAX_SLOPE: f32 = 64.0;

// Interleaved gradient noise, animated per frame so temporal accumulation can
// integrate the PCF disk.
fn shadow_noise() -> f32 {
    let p = lighting_pixel + vec2<f32>(lighting_frame.features.w * 5.588238,
        lighting_frame.features.w * 3.14159);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Per-pixel threshold for the LoD crossfade. A mesh fragment survives while
// `t < mesh_weight`, an impostor fragment while `t >= 1 - impostor_weight`, so
// exactly one of the two covers each pixel. Both shaders must pass
// `@builtin(position).xy`; the same pixel then sees the same `t`. The frame
// index (features.w, zero unless TAA or temporal filtering is on) shifts the
// pattern so temporal accumulation integrates the stipple into a smooth blend.
fn lod_dither(frag_coord: vec2<f32>) -> f32 {
    let p = frag_coord + vec2<f32>(5.588238 * (lighting_frame.features.w % 64.0));
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Vogel (sunflower) disk: sample i of n, unit radius, rotated by phi.
fn shadow_vogel(i: u32, n: u32, phi: f32) -> vec2<f32> {
    let r = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * 2.39996323 + phi;
    return vec2<f32>(cos(theta), sin(theta)) * r;
}

// Light-space coordinates of a world point in cascade c: uv (0..1, matching the
// rasterised rows) and depth (0..1). The lookup first moves along the normal by
// normal_bias texels, scaled down as the surface turns toward the sun.
fn shadow_coords(c: u32, world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let sun = normalize(lighting_env.sun_direction.xyz);
    let texel = lighting_frame.cascade_params[c].x;
    let offset = n * (lighting_frame.shadow2.y * texel * (1.0 - clamp(dot(n, sun), 0.0, 1.0)));
    let clip = lighting_frame.cascade_view_proj[c] * vec4<f32>(world_pos + offset, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5, ndc.z);
}

// Distance from the nearest map edge, in uv units. Positive inside.
fn shadow_edge(p: vec3<f32>) -> f32 {
    return min(min(p.x, p.y), min(1.0 - p.x, 1.0 - p.y));
}

fn shadow_inside(p: vec3<f32>) -> bool {
    return shadow_edge(p) > 0.0 && p.z >= 0.0 && p.z <= 1.0;
}

// Depth slope of the receiver plane in cascade c: (dz/du, dz/dv) for a surface
// with normal n, so a flat receiver's depth at a uv offset d from p is
// p.z + dot(slope, d). The cascade transform is orthographic, so u, v and z are
// linear in world space and the slope follows from the plane alone. Zero when
// the plane is parallel to the light's rays.
fn shadow_plane_slope(c: u32, n: vec3<f32>) -> vec2<f32> {
    let m = lighting_frame.cascade_view_proj[c];
    // Rows of the light transform: uv (scaled from ndc) and depth.
    let ru = 0.5 * vec3<f32>(m[0].x, m[1].x, m[2].x);
    let rv = 0.5 * vec3<f32>(m[0].y, m[1].y, m[2].y);
    let rz = vec3<f32>(m[0].z, m[1].z, m[2].z);
    // Solve rz = a * ru + b * rv + k * n for (a, b) by Cramer's rule.
    let det = dot(ru, cross(rv, n));
    if abs(det) < 1e-6 {
        return vec2<f32>(0.0);
    }
    let slope = vec2<f32>(dot(rz, cross(rv, n)), dot(ru, cross(rz, n))) / det;
    return clamp(slope, vec2<f32>(-SHADOW_MAX_SLOPE), vec2<f32>(SHADOW_MAX_SLOPE));
}

// Visibility (0..1) of cascade c at uv/depth p with the frame's filter, for a
// receiver with normal `n`. Every tap is compared against the receiver plane's
// depth at its own offset, so a flat surface never reads its neighbours as
// occluders, and a tilted one gets no slope acne. The caller guarantees the
// point lies inside cascade c.
fn shadow_sample(c: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let layer = i32(c);
    let filter_mode = u32(lighting_frame.shadow.z + 0.5);
    if filter_mode == 0u {
        return textureSampleCompareLevel(lighting_shadow_map, lighting_shadow_sampler,
            p.xy, layer, p.z);
    }
    let slope = shadow_plane_slope(c, n);
    let texel = lighting_frame.shadow.y;
    let taps = max(u32(lighting_frame.shadow.w + 0.5), 1u);
    let phi = shadow_noise() * SHADOW_TAU;
    var radius = lighting_frame.shadow2.x; // texels
    if filter_mode == 2u {
        // Blocker search: a few plain depth reads over the widest penumbra the
        // light's angular size (softness) can produce in this cascade, each
        // against the receiver plane at its own offset. No blocker means the
        // point is fully lit, so skip the filter.
        let dims = vec2<f32>(textureDimensions(lighting_shadow_map));
        let search = clamp(lighting_frame.shadow2.x * SHADOW_SEARCH_GAP
            / max(lighting_frame.cascade_params[c].x, 1e-6), 2.0, 24.0);
        var gap_sum = 0.0;
        var blockers = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let o = shadow_vogel(i, 8u, phi) * search; // texels
            let coord = clamp(vec2<i32>(floor(p.xy * dims + o)), vec2<i32>(0),
                vec2<i32>(dims) - vec2<i32>(1));
            let d = textureLoad(lighting_shadow_map, coord, layer, 0);
            let plane = p.z + dot(slope, o * texel);
            if d < plane - SHADOW_BLOCKER_EPS {
                gap_sum += plane - d;
                blockers += 1.0;
            }
        }
        if blockers == 0.0 {
            return 1.0;
        }
        // Penumbra width: mean receiver-blocker gap in metres, times softness,
        // in texels of this cascade.
        let gap = gap_sum / blockers;
        let penumbra = lighting_frame.shadow2.x * gap * lighting_frame.cascade_params[c].y
            / max(lighting_frame.cascade_params[c].x, 1e-6);
        radius = clamp(penumbra, 1.0, 32.0);
    }
    var sum = 0.0;
    for (var i = 0u; i < taps; i++) {
        let o = shadow_vogel(i, taps, phi) * radius * texel;
        sum += textureSampleCompareLevel(lighting_shadow_map, lighting_shadow_sampler,
            p.xy + o, layer, p.z + dot(slope, o));
    }
    return sum / f32(taps);
}

// Shadow visibility of a point with shading normal n: 1 lit, 0 shadowed.
// Fades to lit toward the far end of the shadow distance, and blends into the
// next cascade across the last SHADOW_EDGE_BLEND of each map.
fn shadow_visibility(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    let count = u32(lighting_frame.shadow.x + 0.5);
    if count == 0u {
        return 1.0;
    }
    let range = lighting_frame.shadow_fade;
    let fade = smoothstep(range.x, max(range.y, range.x + 1e-3),
        distance(world_pos, lighting_env.cam_pos.xyz));
    if fade >= 1.0 {
        return 1.0;
    }
    var c = count;
    var p = vec3<f32>(0.0);
    for (var i = 0u; i < count; i++) {
        let candidate = shadow_coords(i, world_pos, n);
        if shadow_inside(candidate) {
            c = i;
            p = candidate;
            break;
        }
    }
    if c == count {
        return 1.0;
    }
    var vis = shadow_sample(c, p, n);
    let edge = shadow_edge(p);
    if edge < SHADOW_EDGE_BLEND && c + 1u < count {
        let next = shadow_coords(c + 1u, world_pos, n);
        if shadow_inside(next) {
            let t = smoothstep(0.0, SHADOW_EDGE_BLEND, edge);
            vis = mix(shadow_sample(c + 1u, next, n), vis, t);
        }
    }
    return mix(vis, 1.0, fade);
}

// Debug palette (`shadow2.w`): tint of the cascade covering a point, white
// outside every cascade.
fn shadow_debug_tint(world_pos: vec3<f32>) -> vec3<f32> {
    let count = u32(lighting_frame.shadow.x + 0.5);
    for (var i = 0u; i < count; i++) {
        if shadow_inside(shadow_coords(i, world_pos, vec3<f32>(0.0))) {
            if i == 0u { return vec3<f32>(1.0, 0.35, 0.35); }
            if i == 1u { return vec3<f32>(0.35, 1.0, 0.35); }
            if i == 2u { return vec3<f32>(0.35, 0.55, 1.0); }
            return vec3<f32>(1.0, 1.0, 0.35);
        }
    }
    return vec3<f32>(1.0);
}

// Analytic sky: gradient hemisphere. dir is world-space (unit).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    let sky = mix(lighting_env.sky_horizon.rgb, lighting_env.sky_zenith.rgb,
        clamp(up * 1.6, 0.0, 1.0));
    let ground = mix(lighting_env.sky_horizon.rgb, lighting_env.sky_ground.rgb,
        clamp(-up * 1.6, 0.0, 1.0));
    return select(ground, sky, up >= 0.0);
}

// Hemisphere average for rough reflections.
fn sky_average() -> vec3<f32> {
    return (lighting_env.sky_zenith.rgb + lighting_env.sky_horizon.rgb * 2.0
        + lighting_env.sky_ground.rgb) * 0.25;
}

// Sky environment map uv of a unit direction (y up): u = atan2(z, x) / 2pi + 1/2,
// v = acos(y) / pi, so v = 0 is straight up. Mirrors `sky_env_uv` in atmosphere.rs.
fn sky_env_uv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.z, dir.x);
    return vec2<f32>(phi / (2.0 * PI) + 0.5, acos(clamp(dir.y, -1.0, 1.0)) / PI);
}

// Diffuse irradiance over pi for a surface with unit normal n, from the driven
// environment: the zenith value for up-facing normals, the ground value for
// down-facing ones, and the horizon value at n.y = 0, blended by n.y.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let horizon = lighting_env.sky_horizon.rgb;
    if n.y >= 0.0 {
        return mix(horizon, lighting_env.sky_zenith.rgb, n.y);
    }
    return mix(horizon, lighting_env.sky_ground.rgb, -n.y);
}

fn f_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    let x2 = x * x;
    return f0 + (vec3<f32>(1.0) - f0) * (x2 * x2 * x);
}

fn d_ggx(noh: f32, rough: f32) -> f32 {
    let a = max(rough * rough, 1e-4);
    let a2 = a * a;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith height-correlated G2/GGX (Schlick approximation).
fn g_smith(nov: f32, nol: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let ggx_l = nol / (nol * (1.0 - a) + a);
    let ggx_v = nov / (nov * (1.0 - a) + a);
    return ggx_l * ggx_v / (4.0 * nov * nol + 1e-6);
}

// Karis's analytic EnvBRDF split-sum approximation.
fn env_brdf(rough: f32, nov: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Scene surface attachment (SURFACE_FORMAT, A2B10G10R10): rg = octahedral
// world-space unit normal remapped to 0..1, b = perceptual roughness,
// a = class (1 dielectric, 2/3 metallic; 0 is reserved for no surface data).
// Written by every shader whose output goes to the scene pass.
fn surface_encode(n: vec3<f32>, roughness: f32, metallic: f32) -> vec4<f32> {
    let a = abs(n.x) + abs(n.y) + abs(n.z);
    var o = n.xz / max(a, 1e-6);
    if n.y < 0.0 {
        let s = vec2<f32>(select(-1.0, 1.0, o.x >= 0.0), select(-1.0, 1.0, o.y >= 0.0));
        o = (vec2<f32>(1.0) - abs(o.yx)) * s;
    }
    return vec4<f32>(o * 0.5 + 0.5, clamp(roughness, 0.0, 1.0), select(1.0, 0.6667, metallic > 0.5));
}

// Jimenez et al. 2016 multi-bounce fit: occlusion on bright albedo is lifted,
// per channel, so light albedo is not darkened as much as dark albedo.
fn multi_bounce_ao(ao: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - vec3<f32>(0.3324);
    let b = -4.7951 * albedo + vec3<f32>(0.6417);
    let c = 2.7552 * albedo + vec3<f32>(0.6903);
    return max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
}

// Screen-space reprojection of a world position into the previous frame's
// results (the frame whose gi/ao and reflection textures are bound). `valid`
// is false behind the camera and off screen. Computed once per fragment.
struct Reprojection {
    uv: vec2<f32>,
    depth: f32,
    valid: bool,
};

fn lighting_reproject(world_pos: vec3<f32>) -> Reprojection {
    var r: Reprojection;
    let clip = lighting_frame.prev_view_proj * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 {
        return r;
    }
    r.uv = clip.xy / clip.w * 0.5 + 0.5;
    // For the engine's perspective projection clip.w is the view distance.
    r.depth = clip.w;
    r.valid = all(r.uv >= vec2<f32>(0.0)) && all(r.uv <= vec2<f32>(1.0));
    return r;
}

// Bilinear 4-tap gather of `tex` at the reprojected uv. Each tap is weighted
// down where the depth of the texel it lands on, read from
// `lighting_feedback_depth` (which may be finer than `tex`, at the texel the
// tap's own texel maps to), disagrees with the point's depth. Returns
// `fallback` when no tap agrees.
fn lighting_gather(tex: texture_2d<f32>, rp: Reprojection, fallback: vec4<f32>) -> vec4<f32> {
    let size_u = textureDimensions(tex);
    let depth_u = textureDimensions(lighting_feedback_depth);
    let ratio = vec2<f32>(depth_u) / vec2<f32>(size_u);
    let p = rp.uv * vec2<f32>(size_u) - vec2<f32>(0.5);
    let base = floor(p);
    let f = p - base;
    let max_px = vec2<i32>(size_u) - vec2<i32>(1);
    let max_dq = vec2<i32>(depth_u) - vec2<i32>(1);
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let c = clamp(vec2<i32>(base) + vec2<i32>(i, j), vec2<i32>(0), max_px);
            let dq = clamp(vec2<i32>(vec2<f32>(c) * ratio), vec2<i32>(0), max_dq);
            let d = textureLoad(lighting_feedback_depth, dq, 0).r;
            let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let depth_diff = abs(d - rp.depth) / (0.05 * rp.depth + 0.05);
            let dw = exp(-depth_diff * depth_diff * 2.0);
            let w = bw * dw;
            sum += textureLoad(tex, c, 0) * w;
            total += w;
        }
    }
    if total < 1e-4 {
        return fallback;
    }
    return sum / total;
}

// Indirect diffuse (rgb) and ambient visibility (a) of the previous frame at the
// point. Neutral (no light, full visibility) when the features are off or nothing
// agrees.
fn lighting_feedback(rp: Reprojection) -> vec4<f32> {
    let neutral = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (lighting_frame.features.x <= 0.0 && lighting_frame.features.y <= 0.0) || !rp.valid {
        return neutral;
    }
    return lighting_gather(lighting_gi_ao, rp, neutral);
}

// Traced reflection (rgb) and its confidence (a) of the previous frame at the
// point. Zero when reflections are off or nothing agrees.
fn lighting_reflection_at(rp: Reprojection) -> vec4<f32> {
    if lighting_frame.features.z <= 0.0 || !rp.valid {
        return vec4<f32>(0.0);
    }
    return lighting_gather(lighting_reflection, rp, vec4<f32>(0.0));
}

// Full BRDF for one point: sun + analytic sky (diffuse irradiance, split-sum
// specular). n, v are world-space and normalized; ao multiplies ambient only.
fn lighting_eval(
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    ao: f32,
) -> vec3<f32> {
    let rough = clamp(roughness, 0.02, 1.0);
    let metal = clamp(metallic, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metal);
    let l = normalize(lighting_env.sun_direction.xyz);
    let h = normalize(v + l);
    let nov = clamp(dot(n, v), 1e-4, 1.0);
    let nol = clamp(dot(n, l), 0.0, 1.0);
    let noh = clamp(dot(n, h), 0.0, 1.0);

    var color = vec3<f32>(0.0);
    if nol > 0.0 {
        let f = f_schlick(f0, clamp(dot(h, v), 0.0, 1.0));
        let spec = d_ggx(noh, rough) * g_smith(nov, nol, rough) * f;
        let diffuse = albedo * (vec3<f32>(1.0) - f) * (1.0 - metal) / PI;
        let sun_visibility = mix(1.0, shadow_visibility(world_pos, n), lighting_frame.shadow2.z)
            * lighting_cloud_visibility(world_pos);
        color += (diffuse + spec) * lighting_env.sun_color.rgb * nol * sun_visibility;
    }

    // Ambient: irradiance approximated by the sky gradient at the normal;
    // specular by a roughness-blurred lookup of the reflection direction.
    // Screen-space AO (previous frame, reprojected) attenuates ambient light
    // only; screen-space GI adds one-bounce diffuse light to it.
    // The previous frame's screen-space results share one reprojection.
    let rp = lighting_reproject(world_pos);
    let feedback = lighting_feedback(rp);
    var visibility = 1.0;
    if lighting_frame.features.x > 0.0 {
        visibility = pow(clamp(feedback.a, 0.0, 1.0), lighting_frame.features.x);
    }
    let occlusion = ao * visibility;
    var occlusion_rgb = vec3<f32>(occlusion);
    if lighting_frame.indirect.x > 0.5 {
        occlusion_rgb = multi_bounce_ao(occlusion, albedo);
    }
    // The sky environment map (when valid) gives the specular lookup a sharp
    // mirror and a blurred diffuse-like lobe. The sun-driven colours are irradiance,
    // so they shade diffuse through `sky_irradiance`; otherwise the gradient is used.
    let env_valid = lighting_frame.indirect.z > 0.5;
    let driven = lighting_env.params.y > 0.5;
    var diffuse_sky = sky_radiance(n);
    if env_valid && driven {
        diffuse_sky = sky_irradiance(n);
    }
    let ambient_diffuse = (diffuse_sky * occlusion_rgb
        + feedback.rgb * lighting_frame.features.y) * albedo * (1.0 - metal);
    let ab = env_brdf(rough, nov);
    // Traced reflections (previous frame) replace the sky fallback, and its
    // specular occlusion, where their confidence is high. Zero strength leaves
    // the sky term unchanged; full confidence skips sky env lookup entirely.
    let traced = lighting_reflection_at(rp);
    let traced_k = clamp(traced.a * lighting_frame.features.z, 0.0, 1.0);
    var spec_light = traced.rgb;
    if traced_k < 0.999 {
        let reflect_dir = reflect(-v, n);
        var ambient_spec = mix(sky_radiance(reflect_dir), sky_average(),
            rough * rough * 0.8);
        if env_valid {
            let mip = clamp(roughness, 0.0, 1.0) * lighting_frame.indirect.w;
            ambient_spec = textureSampleLevel(lighting_env_map, lighting_env_sampler,
                sky_env_uv(reflect_dir), mip).rgb;
        }
        var spec_occlusion = occlusion;
        if lighting_frame.indirect.y > 0.5 {
            spec_occlusion = clamp(pow(nov + occlusion, exp2(-16.0 * rough - 1.0)) - 1.0 + occlusion,
                0.0, 1.0);
        }
        spec_light = mix(ambient_spec * spec_occlusion, traced.rgb, traced_k);
    }
    let ambient_specular = spec_light * (f0 * ab.x + ab.y);
    color += ambient_diffuse + ambient_specular;
    if lighting_frame.shadow2.w > 0.5 {
        color *= shadow_debug_tint(world_pos);
    }
    return color * lighting_env.params.x;
}

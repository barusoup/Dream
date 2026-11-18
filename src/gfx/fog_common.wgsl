// Shared volumetric fog maths. Prepended to every `fog_*` pass by the Rust side
// and by build.rs. Depth slices run from `near` to `dist` with
// depth(s) = near * (dist / near)^(s / slices), where s is a continuous slice
// coordinate (slice k covers s in [k, k + 1]). Mirrors `gfx::fog`.

struct FogParams {
    // x: width, y: height, z: slices, w: near plane (m).
    grid: vec4<f32>,
    // x: distance covered by the volume (m), y: sky path beyond it (m),
    // z: unused, w: 1 when the history is invalid (reset).
    range: vec4<f32>,
    // x: density at base height (1/m), y: height falloff (1/m), z: base height (m),
    // w: noise amount (0..1).
    density: vec4<f32>,
    // rgb: scattering albedo, w: Henyey-Greenstein g.
    scatter: vec4<f32>,
    // x: ambient multiplier, y: sun scatter multiplier, z: noise scale (m),
    // w: this frame's slice jitter in [0, 1).
    light: vec4<f32>,
    // xyz: noise offset (-wind * time), w: history weight (0 when temporal is off).
    wind: vec4<f32>,
    // xyz: direction TO the sun.
    sun_dir: vec4<f32>,
    // rgb: sun colour (the environment's, possibly sky-driven).
    sun_color: vec4<f32>,
    // rgb: sky ambient radiance (average of the zenith and horizon).
    ambient: vec4<f32>,
    // xyz: camera position.
    cam_pos: vec4<f32>,
    // x: tan(half fov) x aspect, y: tan(half fov), z: far plane (m), w: unused.
    tan_fov: vec4<f32>,
    // Camera space to world space (unjittered).
    cam_to_world: mat4x4<f32>,
    // Previous unjittered view-projection, world to clip.
    prev_view_proj: mat4x4<f32>,
};

// Copy of `LightingFrame` in lighting.wgsl; same field order and layout.
struct LightingFrame {
    cascade_view_proj: array<mat4x4<f32>, 4>,
    cascade_params: array<vec4<f32>, 4>,
    shadow: vec4<f32>,
    shadow2: vec4<f32>,
    shadow_fade: vec4<f32>,
    screen: vec4<f32>,
    features: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
    indirect: vec4<f32>,
    cloud_shadow: vec4<f32>,
    cloud_layer: vec4<f32>,
};

// Sun transmittance through the clouds at a world point (see
// lighting_cloud_visibility in lighting.wgsl; the fog reads the same map).
fn fog_cloud_visibility(lf: LightingFrame, map: texture_2d<f32>, samp: sampler, world: vec3<f32>) -> f32 {
    let s = lf.cloud_shadow;
    if s.w <= 0.0 {
        return 1.0;
    }
    let uv = (world.xz - s.xy) * s.z + vec2<f32>(0.5);
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return 1.0;
    }
    let cloud_t = textureSampleLevel(map, samp, uv, 0.0).r;
    // Ground-referenced, as in lighting_cloud_visibility: full below the base, none
    // 50 m above it.
    let fade = 1.0 - smoothstep(lf.cloud_layer.x, lf.cloud_layer.x + 50.0, world.y);
    return mix(1.0, cloud_t, s.w * fade);
}

const FOG_PI: f32 = 3.14159265358979;

fn fog_slice_depth(s: f32, near: f32, dist: f32, n: f32) -> f32 {
    return near * exp2(log2(dist / near) * (s / n));
}

fn fog_depth_to_slice(z: f32, near: f32, dist: f32, n: f32) -> f32 {
    return n * log2(max(z, near) / near) / log2(dist / near);
}

fn fog_hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + vec3<f32>(33.33));
    return fract((q.x + q.y) * q.z);
}

// Trilinear value noise in [0, 1].
fn fog_value_noise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = fog_hash(i);
    let n100 = fog_hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = fog_hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = fog_hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = fog_hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = fog_hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = fog_hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = fog_hash(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Two-octave fbm in [0, 1].
fn fog_fbm(x: vec3<f32>) -> f32 {
    return 0.65 * fog_value_noise(x) + 0.35 * fog_value_noise(x * 2.03 + vec3<f32>(17.1, 3.7, 9.2));
}

// Extinction (1/m) at a world position. The noise term has mean 1, so the
// average density matches the analytic height falloff used beyond the volume.
fn fog_density(p: FogParams, pos: vec3<f32>) -> f32 {
    let height = exp(-p.density.y * max(pos.y - p.density.z, 0.0));
    let base = p.density.x * height;
    if p.density.w <= 0.0 || base <= 1e-7 {
        return base;
    }
    let n = fog_fbm((pos + p.wind.xyz) / p.light.z);
    return base * (1.0 - p.density.w + 2.0 * p.density.w * n);
}

// Henyey-Greenstein phase function; cos is the angle between the view ray and the sun.
fn fog_hg(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * c, 1e-4);
    return (1.0 - g2) / (4.0 * FOG_PI * d * sqrt(d));
}

// Visibility of the sun at a world position: one comparison sample of the
// cascade that contains it (1 when shadows are off or outside every cascade).
fn fog_shadow(lf: LightingFrame, map: texture_depth_2d_array, cmp: sampler_comparison, world: vec3<f32>) -> f32 {
    let count = u32(lf.shadow.x + 0.5);
    for (var c = 0u; c < count; c++) {
        let clip = lf.cascade_view_proj[c] * vec4<f32>(world, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
        if uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0 {
            return textureSampleCompareLevel(map, cmp, uv, i32(c), ndc.z);
        }
    }
    return 1.0;
}

// Integral of the height-fog density along a ray segment: the density is
// `density` below `base` and `density * exp(-k * (y - base))` above it, with
// the ray at height y0 + t * dy for t in [t0, t1]. Closed form, split at the
// crossing of `base`.
fn fog_height_od(t0: f32, t1: f32, y0: f32, dy: f32, density: f32, k: f32, base: f32) -> f32 {
    if t1 <= t0 {
        return 0.0;
    }
    if abs(dy) <= 1e-6 {
        if y0 < base {
            return density * (t1 - t0);
        }
        return density * exp(-k * (y0 - base)) * (t1 - t0);
    }
    // Parameter where the ray crosses the base height, clamped to the segment.
    let tc = clamp((base - y0) / dy, t0, t1);
    // Below the base the ray is at `density`; above it decays. Which side comes
    // first depends on the sign of dy.
    var below = vec2<f32>(t0, tc);
    var above = vec2<f32>(tc, t1);
    if dy < 0.0 {
        below = vec2<f32>(tc, t1);
        above = vec2<f32>(t0, tc);
    }
    var od = density * (below.y - below.x);
    let len = above.y - above.x;
    if len > 0.0 {
        let ya = max(y0 + dy * above.x - base, 0.0);
        let kd = k * dy;
        var integral = len;
        if abs(kd) > 1e-6 {
            integral = (1.0 - exp(-kd * len)) / kd;
        }
        od += density * exp(-k * ya) * integral;
    }
    return od;
}

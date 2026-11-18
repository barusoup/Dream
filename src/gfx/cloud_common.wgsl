// Shared cloud maths. Prepended to every `cloud_*` pass and to `atmo_envmap.wgsl`
// by the Rust side and by build.rs. Mirrors `render::clouds` (the panorama
// mapping, the height profile, the coverage remap) and the noise generators.

const CLOUD_PI: f32 = 3.14159265358979;

// Panorama mapping: Lambert cylindrical equal-area projection of the upper
// hemisphere. u = azimuth / 2pi + 1/2 (azimuth atan2(z, x)); v = 1 - y, so v = 0
// is the zenith and v = 1 the horizon. Directions must have y >= 0.
fn cloud_panorama_uv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.z, dir.x);
    return vec2<f32>(phi / (2.0 * CLOUD_PI) + 0.5, 1.0 - clamp(dir.y, 0.0, 1.0));
}

// Composite the cloud panorama over a sky radiance seen along `dir`: the sky is
// attenuated by the clouds' transmittance and the cloud radiance is added. Below
// the horizon there are no clouds and the sky is returned unchanged. Shared by
// the sky pass and the sky environment map.
fn cloud_composite(sky: vec3<f32>, dir: vec3<f32>, panorama: texture_2d<f32>,
    samp: sampler) -> vec3<f32> {
    var c = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if dir.y >= 0.0 {
        let size = vec2<f32>(textureDimensions(panorama));
        var cuv = cloud_panorama_uv(dir);
        // Repeat in azimuth only; keep the zenith and horizon rows inside the image.
        cuv.y = clamp(cuv.y, 0.5 / size.y, 1.0 - 0.5 / size.y);
        c = textureSampleLevel(panorama, samp, cuv, 0.0);
    }
    return sky * c.a + c.rgb;
}

fn cloud_panorama_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * 2.0 * CLOUD_PI;
    let y = 1.0 - uv.y;
    let s = sqrt(max(1.0 - y * y, 0.0));
    return vec3<f32>(s * cos(phi), y, s * sin(phi));
}

// Remap x from [a, b] to [c, d], clamped to the output range.
fn cloud_remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    if abs(b - a) < 1e-6 {
        return c;
    }
    return clamp(c + (x - a) * (d - c) / (b - a), min(c, d), max(c, d));
}

// Vertical dome profile at height fraction h for a column whose top is `top`:
// rounded over the first 10% of the layer, tapering to zero at the column top.
fn cloud_dome(h: f32, top: f32) -> f32 {
    return smoothstep(0.0, 0.1, h) * (1.0 - smoothstep(top * 0.5, top, h));
}

// Base shape remapped with a soft threshold so that `coverage` of the noise
// range is filled. The ramp is 0.25 wide, so there are no hard plateaus.
fn cloud_coverage_shape(shape: f32, coverage: f32) -> f32 {
    if coverage <= 0.0 {
        return 0.0;
    }
    let thr = (1.0 - coverage) * 0.8;
    return smoothstep(thr, min(thr + 0.25, 1.0), shape);
}

// Henyey-Greenstein phase function, normalised to 1 / (4 pi) at isotropy.
fn cloud_hg(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * CLOUD_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// Integer hash of a lattice cell (wrapping u32 arithmetic).
fn cloud_hash_u(v: vec3<u32>) -> u32 {
    var h = v.x * 374761393u + v.y * 668265263u + v.z * 2147483647u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

// Three hashed values in [0, 1) for a lattice cell, wrapped to `period` cells.
fn cloud_hash3(c: vec3<i32>, period: i32) -> vec3<f32> {
    let p = vec3<i32>(period);
    let w = ((c % p) + p) % p;
    let u = vec3<u32>(w);
    let k = 1.0 / 4294967295.0;
    return vec3<f32>(
        f32(cloud_hash_u(u)) * k,
        f32(cloud_hash_u(u + vec3<u32>(7919u, 104729u, 1299709u))) * k,
        f32(cloud_hash_u(u.zxy + vec3<u32>(31u, 65537u, 3u))) * k,
    );
}

// Tiling Worley (cellular) noise: 1 - F1 in [0, 1], with `period` cells per tile.
fn cloud_worley(p: vec3<f32>, period: i32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - vec3<f32>(i);
    var d = 1e9;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let c = i + vec3<i32>(dx, dy, dz);
                let feature = cloud_hash3(c, period);
                let r = vec3<f32>(f32(dx), f32(dy), f32(dz)) + feature - f;
                d = min(d, dot(r, r));
            }
        }
    }
    return clamp(1.0 - sqrt(d), 0.0, 1.0);
}

// Tiling gradient (Perlin) noise in about [-1, 1], with `period` cells per tile.
fn cloud_perlin(p: vec3<f32>, period: i32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let ii = vec3<i32>(i);
    var acc = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let o = vec3<i32>(vec3<u32>(k, k >> 1u, k >> 2u) & vec3<u32>(1u));
        let g = cloud_hash3(ii + o, period) * 2.0 - vec3<f32>(1.0);
        let d = f - vec3<f32>(o);
        let w = vec3<f32>(
            select(1.0 - u.x, u.x, o.x == 1),
            select(1.0 - u.y, u.y, o.y == 1),
            select(1.0 - u.z, u.z, o.z == 1),
        );
        acc += dot(g, d) * w.x * w.y * w.z;
    }
    return acc;
}

// Three octaves of Worley noise (weights 0.625, 0.25, 0.125) at cell counts
// 4, 8 and 16 per unit of `p`, which is in [0, 1)^3.
fn cloud_worley_fbm(p: vec3<f32>) -> f32 {
    return cloud_worley(p * 4.0, 4) * 0.625
        + cloud_worley(p * 8.0, 8) * 0.25
        + cloud_worley(p * 16.0, 16) * 0.125;
}

// ---------------------------------------------------------------------------
// Cloud march core, shared by the panorama march (cloud_march.wgsl), the
// per-pixel view march (cloud_view_march.wgsl) and the cloud shadow map
// (cloud_shadow.wgsl). Textures, samplers and parameter blocks are arguments,
// so this file declares no bindings.

// CPU mirror: `CloudParams` in gfx/clouds.rs (112 bytes).
struct CloudParams {
    // xyz: camera world position (m), w: sea level (m).
    cam: vec4<f32>,
    // x: planet radius (m), y: base altitude above the planet (m), z: thickness (m),
    // w: max march distance (m).
    layer: vec4<f32>,
    // x: coverage, y: density, z: noise scale (m), w: detail erosion.
    shape: vec4<f32>,
    // xy: wind (m/s), z: evolution (1/s), w: scene time (s).
    drift: vec4<f32>,
    // x: silver lining, y: ambient scale, z: panorama view steps, w: light steps.
    look: vec4<f32>,
    // Panorama only. x: refresh stride, y and z: texel offset, w: frame seed.
    refresh: vec4<f32>,
    // Panorama only. x, y: panorama size, z: 1 for a full refresh.
    panorama: vec4<f32>,
};

// CPU mirror: `ViewParams` in gfx/cloud_view.rs (240 bytes).
struct ViewParams {
    // Unjittered world-from-clip, and the previous frame's clip-from-world.
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    // xyz: camera world position, w: sea level (m).
    cam: vec4<f32>,
    // xyz: camera forward (world, unit).
    fwd: vec4<f32>,
    // As `CloudParams.layer`.
    layer: vec4<f32>,
    // x, y: render size (pixels), z, w: their reciprocals.
    screen: vec4<f32>,
    // x, y: resolve history size (full render size), z, w: march grid size (half res).
    res: vec4<f32>,
    // x: frame index, y: 1 when the history is invalid.
    state: vec4<f32>,
    // x: view march steps, the cap of the step count per ray.
    march: vec4<f32>,
};

const CLOUD_EXTINCTION: f32 = 0.03;
// Scale from unit-normalised phase to radiance from a unit-irradiance sun.
const CLOUD_SUN_GAIN: f32 = 3.2;
// Distance over which cloud radiance takes on the sky colour, metres.
const CLOUD_AERIAL_M: f32 = 40000.0;
// Distance stored where a ray met no cloud: beyond any march or scene distance.
const CLOUD_NO_HIT: f32 = 1.0e7;
// Linear depth (metres) at or beyond which a pixel holds no geometry (the depth
// pass stores 65000 for sky).
const CLOUD_SKY_DEPTH: f32 = 60000.0;

// Entry and exit distances of the ray with the sphere of radius r centred at o.
// (-1, -1) when there is no hit.
fn cloud_sphere(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// Planet centre in the march's frame: straight below the camera, so the camera
// sits at radius rp + (cam.y - sea).
fn cloud_planet_centre(cam: vec4<f32>, layer: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(0.0, layer.x + (cam.y - cam.w), 0.0);
}

// The ray's segment [t0, t1] inside the cloud layer: outside the bottom sphere
// and the ground, inside the top sphere, within the max distance. Returns
// (-1, -1) when the ray misses the layer. Downward rays from above are kept.
fn cloud_layer_span(o: vec3<f32>, dir: vec3<f32>, layer: vec4<f32>) -> vec2<f32> {
    let rp = layer.x;
    let rb = rp + layer.y;
    let rt = rb + layer.z;
    let outer = cloud_sphere(o, dir, rt);
    if outer.y <= 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    var t0 = max(outer.x, 0.0);
    var t1 = outer.y;
    let inner = cloud_sphere(o, dir, rb);
    if inner.y > 0.0 {
        if length(o) < rb {
            t0 = max(t0, inner.y);
        } else {
            t1 = min(t1, inner.x);
        }
    }
    let ground = cloud_sphere(o, dir, rp);
    if ground.y > 0.0 && ground.x > 0.0 {
        t1 = min(t1, ground.x);
    }
    t1 = min(t1, layer.w);
    if t1 <= t0 {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(t0, t1);
}

// Per-column cloud top, as a height fraction. A horizontal slice of the base
// volume at period 2.5 * scale gives a low-frequency column signal, so tops vary
// across the sky; overcast cover flattens them toward the layer top.
fn cloud_column_top(shape_tex: texture_3d<f32>, samp: sampler, xz: vec2<f32>, cov: f32,
    scale: f32) -> f32 {
    let col = textureSampleLevel(shape_tex, samp, vec3<f32>(xz / (scale * 2.5), 0.37), 0.0).r;
    let tall = smoothstep(0.35, 0.75, col);
    let flat = smoothstep(0.5, 1.0, cov);
    return mix(0.3 + 0.7 * tall, 1.0, 0.7 * flat);
}

// Extinction density (per metre, before `density`) at a world point and height
// fraction h. Every noise lookup is a true 3D sample at (x, altitude, z) divided
// by a period, so features are isotropic; wind shifts x and z, evolution shifts
// the noise along its y axis (shapes change in place).
fn cloud_density(cp: CloudParams, shape_tex: texture_3d<f32>, detail_tex: texture_3d<f32>,
    samp: sampler, world: vec3<f32>, h: f32) -> f32 {
    if h < 0.0 || h > 1.0 {
        return 0.0;
    }
    let t = cp.drift.w;
    let cov = cp.shape.x;
    let scale = cp.shape.z;
    let p = vec3<f32>(world.x + cp.drift.x * t, world.y, world.z + cp.drift.y * t);
    let top = cloud_column_top(shape_tex, samp, p.xz, cov, scale);
    if h > top {
        return 0.0;
    }
    // Dome: rounded over the first 10% of the layer, tapering to this column's top.
    let dome = cloud_dome(h, top);
    if dome <= 0.0 {
        return 0.0;
    }
    let evo = vec3<f32>(0.0, cp.drift.z * t, 0.0);
    // Low-frequency Perlin-Worley at period scale: the cloud map.
    let base_v = textureSampleLevel(shape_tex, samp, p / scale + evo, 0.0);
    // Billows: Worley fbm (detail R) at period scale / 3, carving rounded lobes.
    let billow = textureSampleLevel(detail_tex, samp, p * (3.0 / scale) + evo, 0.0).r;
    let shape = cloud_remap(base_v.r, (1.0 - billow) * 0.3, 1.0, 0.0, 1.0);
    let d = cloud_coverage_shape(shape, cov) * dome;
    if d <= 0.0 {
        return 0.0;
    }
    // Erosion at period scale / 20, stronger toward the bottom and edges.
    let fine = textureSampleLevel(detail_tex, samp, p * (20.0 / scale) + evo, 0.0).r;
    let erode = cp.shape.w * 0.5 * (1.0 + 0.5 * (1.0 - smoothstep(0.0, 0.3, h)));
    return cloud_remap(d, fine * erode, 1.0, 0.0, 1.0) * cp.shape.y;
}

// Optical depth toward the sun from a point (planet frame, metres), over 60% of the layer.
fn cloud_light_depth(cp: CloudParams, shape_tex: texture_3d<f32>, detail_tex: texture_3d<f32>,
    samp: sampler, local: vec3<f32>, sun: vec3<f32>, o: vec3<f32>, cam: vec3<f32>) -> f32 {
    // Light steps grow geometrically (x1.6) from 6% of the layer thickness, so the
    // first samples resolve the cloud near the point and the last reach the sun.
    let steps = u32(cp.look.w);
    let rb = cp.layer.x + cp.layer.y;
    var tau = 0.0;
    var s = 0.0;
    var dt = cp.layer.z * 0.06;
    for (var j = 0u; j < steps; j++) {
        let q = local + sun * (s + 0.5 * dt);
        let h = (length(q) - rb) / cp.layer.z;
        tau += cloud_density(cp, shape_tex, detail_tex, samp, cam + (q - o), h)
            * CLOUD_EXTINCTION * dt;
        s += dt;
        dt *= 1.6;
    }
    return tau;
}

// Two Henyey-Greenstein lobes (forward and back), anisotropy scaled by s.
fn cloud_lobes(c: f32, s: f32) -> f32 {
    return mix(cloud_hg(c, 0.8 * s), cloud_hg(c, -0.3 * s), 0.3);
}

// Multiple-scattering sun term (three octaves, each attenuated and blurred
// toward isotropy), with the silver-lining highlight on the first octave.
fn cloud_sun_scatter(tau: f32, c: f32, silver: f32) -> f32 {
    var sum = 0.0;
    var a = 1.0;
    var b = 1.0;
    var s = 1.0;
    for (var n = 0; n < 3; n++) {
        var ph = cloud_lobes(c, s);
        if n == 0 {
            ph += silver * cloud_hg(c, 0.95);
        }
        sum += a * exp(-tau * b) * ph;
        a *= 0.5;
        b *= 0.5;
        s *= 0.5;
    }
    return sum * CLOUD_SUN_GAIN;
}

// Result of one march through the layer. `front` is the distance along the ray
// where the transmittance first drops below 0.99, `back` the distance of the
// last sample that met cloud. Both are CLOUD_NO_HIT when no sample met cloud.
struct CloudHit {
    light: vec3<f32>,
    trans: f32,
    front: f32,
    back: f32,
};

// March the layer between t0 and t1 along dir. `fine` is the step inside cloud,
// `coarse` the step through empty space (equal to `fine` for a uniform march),
// `budget` the most samples taken, `jitter` in [0, 1) offsets the first sample.
// Radiance includes the aerial fade toward the sky colour and the fade toward
// the max distance, as the panorama and the view pass both show it.
fn cloud_integrate(cp: CloudParams, atmo: AtmoParams, shape_tex: texture_3d<f32>,
    detail_tex: texture_3d<f32>, samp: sampler, tlut: texture_2d<f32>,
    skyview: texture_2d<f32>, lut_samp: sampler, dir: vec3<f32>, t0: f32, t1: f32,
    fine: f32, coarse: f32, budget: u32, jitter: f32, amb_zenith: vec3<f32>,
    dist_growth: f32) -> CloudHit {
    let cam = cp.cam.xyz;
    let rb = cp.layer.x + cp.layer.y;
    let o = cloud_planet_centre(cp.cam, cp.layer);
    let sun = atmo.sun_dir.xyz;
    // Fine steps through the layer; coarse steps through empty space, with a
    // jittered rewind onto the entry bracket. When `dist_growth > 0`, step size
    // scales gently with distance `t` so near/inside views keep fine steps
    // while far rays reach the 60 km horizon within budget.
    let fine0 = fine * (1.0 + t0 * dist_growth);
    var t = t0 + jitter * min(fine0, max(t1 - t0, 0.0));
    var prev_t = t0;
    var trans = 1.0;
    var radiance = vec3<f32>(0.0);
    let adaptive = coarse > fine * 1.001;
    var fine_mode = !adaptive;
    var empty = 0u;
    var wsum = 0.0;
    var tsum = 0.0;
    var front = CLOUD_NO_HIT;
    var back = CLOUD_NO_HIT;
    let cos_sun = dot(dir, sun);
    for (var i = 0u; i < budget; i++) {
        if t >= t1 || trans < 0.01 {
            break;
        }
        let dist_scale = 1.0 + t * dist_growth;
        let fine_t = fine * dist_scale;
        let coarse_t = coarse * dist_scale;
        let local = o + dir * t;
        let r = length(local);
        let h = (r - rb) / cp.layer.z;
        let d = cloud_density(cp, shape_tex, detail_tex, samp, cam + dir * t, h);
        if !fine_mode {
            if d > 0.0 {
                fine_mode = true;
                if i > 0u {
                    t = max(prev_t + jitter * fine_t, t0);
                    continue;
                }
            } else {
                prev_t = t;
                t += coarse_t;
                continue;
            }
        }
        if d <= 0.0 {
            if adaptive {
                empty += 1u;
                if empty > 4u {
                    fine_mode = false;
                    empty = 0u;
                }
            }
            prev_t = t;
            t += fine_t;
            continue;
        }
        empty = 0u;
        let sig = d * CLOUD_EXTINCTION;
        let step = exp(-sig * fine_t);
        let w = trans * (1.0 - step);
        // The sun term is skipped where the sample carries almost no weight.
        var sun_l = vec3<f32>(0.0);
        if w > 0.002 {
            let tau_l = cloud_light_depth(cp, shape_tex, detail_tex, samp, local, sun, o, cam);
            let tr = atmo_transmittance(atmo, tlut, lut_samp, r * 0.001, dot(local, sun) / r);
            // Beer-powder: thin edges scatter more forward than they absorb.
            let powder = mix(1.0, 1.0 - exp(-2.0 * sig * fine_t), 0.5);
            sun_l = atmo.sun_irradiance.rgb * tr
                * cloud_sun_scatter(tau_l, cos_sun, cp.look.x) * powder;
        }
        let ambient = amb_zenith * cp.look.y * mix(0.6, 1.0, clamp(h, 0.0, 1.0));
        radiance += w * (sun_l + ambient);
        wsum += w;
        tsum += w * t;
        trans *= step;
        if front >= CLOUD_NO_HIT && trans < 0.99 {
            front = t;
        }
        back = t;
        prev_t = t;
        t += fine_t;
    }

    // Aerial perspective: the cloud radiance takes on the sky colour with
    // distance, and only where the cloud is opaque.
    let mean_t = select(0.0, tsum / max(wsum, 1e-6), wsum > 0.0);
    let aerial = exp(-mean_t / CLOUD_AERIAL_M);
    let tint = textureSampleLevel(skyview, lut_samp, atmo_skyview_uv(atmo, dir, sun), 0.0).rgb
        * atmo.camera.w;
    let light = radiance * aerial + tint * (1.0 - aerial) * (1.0 - trans);
    // Fade toward the max distance so the horizon blends into the sky.
    let fade = 1.0 - smoothstep(0.5, 1.0, mean_t / cp.layer.w);
    return CloudHit(light * fade, mix(1.0, trans, fade), front, back);
}

// Sun transmittance through the cloud layer above a ground point (world space,
// on the sea-level plane): `steps` samples along the sun ray between its exit
// from the layer bottom and its exit from the top. 1 when nothing is in the way.
fn cloud_sun_transmittance(cp: CloudParams, shape_tex: texture_3d<f32>,
    detail_tex: texture_3d<f32>, samp: sampler, ground: vec3<f32>, sun: vec3<f32>,
    steps: u32) -> f32 {
    let o = cloud_planet_centre(cp.cam, cp.layer);
    let local = o + (ground - cp.cam.xyz);
    let rb = cp.layer.x + cp.layer.y;
    let rt = rb + cp.layer.z;
    let outer = cloud_sphere(local, sun, rt);
    if outer.y <= 0.0 {
        return 1.0;
    }
    let inner = cloud_sphere(local, sun, rb);
    let t0 = max(inner.y, 0.0);
    let t1 = outer.y;
    if t1 <= t0 {
        return 1.0;
    }
    let dt = (t1 - t0) / f32(steps);
    var tau = 0.0;
    for (var j = 0u; j < steps; j++) {
        let t = t0 + (f32(j) + 0.5) * dt;
        let q = local + sun * t;
        let h = (length(q) - rb) / cp.layer.z;
        tau += cloud_density(cp, shape_tex, detail_tex, samp, cp.cam.xyz + (q - o), h)
            * CLOUD_EXTINCTION * dt;
    }
    return exp(-tau);
}

// Unit world direction through the centre of render-target pixel px (continuous
// pixel coordinates), with the unjittered view. Row 0 is ndc y = -1, as the
// scene targets are laid out.
fn cloud_view_ray(vp: ViewParams, px: vec2<f32>) -> vec3<f32> {
    let ndc = px / vp.screen.xy * 2.0 - 1.0;
    let w = vp.inv_vp * vec4<f32>(ndc, 0.5, 1.0);
    return normalize(w.xyz / w.w - vp.cam.xyz);
}

// Interleaved gradient noise in [0, 1), per pixel.
fn cloud_ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

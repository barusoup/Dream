// Shared atmosphere maths (Hillaire 2020). Prepended to every `atmo_*` pass by
// the Rust side and by build.rs. Geometry is in kilometres; coefficients are
// per kilometre. Mirrors `render::atmosphere::AtmosphereModel`.

const ATMO_PI: f32 = 3.14159265358979;
const ATMO_TLUT: vec2<f32> = vec2<f32>(256.0, 64.0);
const ATMO_MS: f32 = 32.0;
const ATMO_TRANSMITTANCE_STEPS: u32 = 40u;
const ATMO_MS_DIRECTIONS: u32 = 64u;
const ATMO_MS_STEPS: u32 = 20u;
const ATMO_SKY_STEPS: u32 = 40u;

struct AtmoParams {
    // x: ground radius, y: top radius, z: Rayleigh scale height, w: Mie scale height (km).
    planet: vec4<f32>,
    // rgb: Rayleigh scattering per km at scale 1 (already scaled). w: night sky red.
    rayleigh: vec4<f32>,
    // x: Mie scattering, y: Mie extinction (per km), z: Mie anisotropy g. w: night sky green.
    mie: vec4<f32>,
    // rgb: ozone absorption per km at the density peak. w: night sky blue.
    ozone: vec4<f32>,
    // rgb: ground albedo. w: 1 to draw the planet ground below the horizon (sky pass).
    ground: vec4<f32>,
    // xyz: direction TO the sun.
    sun_dir: vec4<f32>,
    // rgb: top-of-atmosphere irradiance (sun colour times the sky scale).
    // w: 1 when clouds are on (the sky environment map then composites the cloud panorama).
    sun_irradiance: vec4<f32>,
    // x: camera radius from the planet centre (km), y: sun disk on (0/1),
    // z: sun angular radius (rad), w: sky intensity.
    camera: vec4<f32>,
    // xyz: world camera position, w unused.
    cam_pos: vec4<f32>,
    // width, height, 1/width, 1/height of the scene target.
    screen: vec4<f32>,
    // Sky pass only: inverse of the jittered view-projection.
    inv_view_proj: mat4x4<f32>,
    // Sky pass only: rotation-only view-projection, current and previous, unjittered.
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
};

fn atmo_unit_to_tex(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn atmo_tex_to_unit(u: f32, n: f32) -> f32 {
    return (u - 0.5 / n) / (1.0 - 1.0 / n);
}

fn atmo_top_distance(p: AtmoParams, r: f32, mu: f32) -> f32 {
    return -r * mu + sqrt(max(r * r * mu * mu - r * r + p.planet.y * p.planet.y, 0.0));
}

fn atmo_ground_distance(p: AtmoParams, r: f32, mu: f32) -> f32 {
    return -r * mu - sqrt(max(r * r * mu * mu - r * r + p.planet.x * p.planet.x, 0.0));
}

fn atmo_hits_ground(p: AtmoParams, r: f32, mu: f32) -> bool {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + p.planet.x * p.planet.x >= 0.0;
}

struct AtmoMedium {
    ray: vec3<f32>,
    mie: f32,
    ext: vec3<f32>,
};

fn atmo_medium(p: AtmoParams, h: f32) -> AtmoMedium {
    let hr = exp(-h / p.planet.z);
    let hm = exp(-h / p.planet.w);
    let ozone = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    var m: AtmoMedium;
    m.ray = p.rayleigh.rgb * hr;
    m.mie = p.mie.x * hm;
    m.ext = m.ray + vec3<f32>(p.mie.y * hm) + p.ozone.rgb * ozone;
    return m;
}

fn atmo_scatter(m: AtmoMedium) -> vec3<f32> {
    return m.ray + vec3<f32>(m.mie);
}

// Transmittance LUT: x = zenith cosine (through the top-distance range), y = radius.
fn atmo_tlut_uv(p: AtmoParams, r: f32, mu: f32) -> vec2<f32> {
    let rg = p.planet.x;
    let rt = p.planet.y;
    let h = sqrt(rt * rt - rg * rg);
    let rho = sqrt(max(r * r - rg * rg, 0.0));
    let d = atmo_top_distance(p, r, mu);
    let d_min = rt - r;
    let d_max = rho + h;
    let x_mu = (d - d_min) / max(d_max - d_min, 1e-6);
    return vec2<f32>(atmo_unit_to_tex(x_mu, ATMO_TLUT.x), atmo_unit_to_tex(rho / h, ATMO_TLUT.y));
}

fn atmo_tlut_params(p: AtmoParams, uv: vec2<f32>) -> vec2<f32> {
    let rg = p.planet.x;
    let rt = p.planet.y;
    let h = sqrt(rt * rt - rg * rg);
    let x_mu = atmo_tex_to_unit(uv.x, ATMO_TLUT.x);
    let x_r = atmo_tex_to_unit(uv.y, ATMO_TLUT.y);
    let rho = h * x_r;
    let r = sqrt(rho * rho + rg * rg);
    let d_min = rt - r;
    let d_max = rho + h;
    let d = d_min + x_mu * (d_max - d_min);
    var mu = 1.0;
    if d > 1e-6 {
        mu = clamp((rt * rt - r * r - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

// Transmittance to the top, integrated directly (used to fill the LUT).
fn atmo_transmittance_integral(p: AtmoParams, r: f32, mu: f32) -> vec3<f32> {
    let dist = atmo_top_distance(p, r, mu);
    var od = vec3<f32>(0.0);
    var prev = 0.0;
    for (var i = 0u; i < ATMO_TRANSMITTANCE_STEPS; i++) {
        let t = dist * f32(i + 1u) / f32(ATMO_TRANSMITTANCE_STEPS);
        let dt = t - prev;
        let tm = prev + 0.5 * dt;
        let rr = sqrt(max(r * r + 2.0 * r * mu * tm + tm * tm, 0.0));
        od += atmo_medium(p, rr - p.planet.x).ext * dt;
        prev = t;
    }
    return exp(-od);
}

// Transmittance from the LUT; zero when the planet blocks the ray.
fn atmo_transmittance(p: AtmoParams, tlut: texture_2d<f32>, samp: sampler, r: f32, mu: f32) -> vec3<f32> {
    if atmo_hits_ground(p, r, mu) {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(tlut, samp, atmo_tlut_uv(p, r, mu), 0.0).rgb;
}

// Multiple-scattering LUT: x = sun cosine, y = altitude fraction of the shell.
fn atmo_ms_uv(p: AtmoParams, r: f32, mu_s: f32) -> vec2<f32> {
    let alt = clamp((r - p.planet.x) / (p.planet.y - p.planet.x), 0.0, 1.0);
    return vec2<f32>(atmo_unit_to_tex(mu_s * 0.5 + 0.5, ATMO_MS), atmo_unit_to_tex(alt, ATMO_MS));
}

fn atmo_ms_params(p: AtmoParams, uv: vec2<f32>) -> vec2<f32> {
    let mu_s = atmo_tex_to_unit(uv.x, ATMO_MS) * 2.0 - 1.0;
    let alt = atmo_tex_to_unit(uv.y, ATMO_MS);
    return vec2<f32>(p.planet.x + alt * (p.planet.y - p.planet.x), mu_s);
}

fn atmo_multiscatter(p: AtmoParams, mslut: texture_2d<f32>, samp: sampler, r: f32, mu_s: f32) -> vec3<f32> {
    return textureSampleLevel(mslut, samp, atmo_ms_uv(p, r, mu_s), 0.0).rgb;
}

fn atmo_phase_rayleigh(c: f32) -> f32 {
    return 3.0 / (16.0 * ATMO_PI) * (1.0 + c * c);
}

fn atmo_phase_mie(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return 3.0 / (8.0 * ATMO_PI) * (1.0 - g2) / (2.0 + g2) * (1.0 + c * c)
        / pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5);
}

// Night-sky radiance floor (AtmosphereSettings::night_sky), read from the unused w
// components of the Rayleigh, Mie and ozone blocks so the uniform keeps its layout.
fn atmo_night(p: AtmoParams) -> vec3<f32> {
    return vec3<f32>(p.rayleigh.w, p.mie.w, p.ozone.w);
}

// Sky-view mapping. Azimuth u in [0, 1] (u = 0.5 faces the sun); zenith angle
// concentrated at the horizon: v in [0, 0.5] is the sky above the horizon.
fn atmo_sun_basis(sun: vec3<f32>) -> mat2x3<f32> {
    let h = vec3<f32>(sun.x, 0.0, sun.z);
    var e1 = vec3<f32>(1.0, 0.0, 0.0);
    if length(h) >= 1e-4 {
        e1 = normalize(h);
    }
    return mat2x3<f32>(e1, cross(vec3<f32>(0.0, 1.0, 0.0), e1));
}

fn atmo_horizon_zenith(p: AtmoParams, r: f32) -> f32 {
    return acos(-sqrt(max(1.0 - (p.planet.x / r) * (p.planet.x / r), 0.0)));
}

fn atmo_skyview_dir(p: AtmoParams, uv: vec2<f32>, sun: vec3<f32>) -> vec3<f32> {
    let r = p.camera.x;
    let basis = atmo_sun_basis(sun);
    let theta_h = atmo_horizon_zenith(p, r);
    var theta = theta_h + (ATMO_PI - theta_h) * (2.0 * uv.y - 1.0);
    if uv.y <= 0.5 {
        theta = theta_h * sqrt(2.0 * uv.y);
    }
    let phi = (uv.x - 0.5) * 2.0 * ATMO_PI;
    let st = sin(theta);
    return basis[0] * (st * cos(phi)) + basis[1] * (st * sin(phi)) + vec3<f32>(0.0, cos(theta), 0.0);
}

fn atmo_skyview_uv(p: AtmoParams, dir: vec3<f32>, sun: vec3<f32>) -> vec2<f32> {
    let r = p.camera.x;
    let basis = atmo_sun_basis(sun);
    let theta_h = atmo_horizon_zenith(p, r);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    var v = 0.5 + 0.5 * (theta - theta_h) / (ATMO_PI - theta_h);
    if theta <= theta_h {
        v = 0.5 * (theta / theta_h) * (theta / theta_h);
    }
    let phi = atan2(dot(dir, basis[1]), dot(dir, basis[0]));
    return vec2<f32>(phi / (2.0 * ATMO_PI) + 0.5, v);
}

// Sky-view LUT radiance along a direction, scaled by the sky intensity. No sun
// disk and no night floor. The sky pass and the environment map both read it.
fn atmo_sky_view_radiance(p: AtmoParams, skyview: texture_2d<f32>, samp: sampler,
    dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(skyview, samp, atmo_skyview_uv(p, dir, sun), 0.0).rgb * p.camera.w;
}

// Radiance along a view ray from the camera (planet-frame, y up), without the
// sun disk. Shared by the sky-view pass.
fn atmo_sky_radiance(p: AtmoParams, tlut: texture_2d<f32>, mslut: texture_2d<f32>, samp: sampler,
    r0: f32, dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let mu = dir.y;
    let hits = atmo_hits_ground(p, r0, mu);
    var tmax = atmo_top_distance(p, r0, mu);
    if hits {
        tmax = atmo_ground_distance(p, r0, mu);
    }
    let cos_sd = dot(dir, sun);
    let pr = atmo_phase_rayleigh(cos_sd);
    let pm = atmo_phase_mie(cos_sd, p.mie.z);
    let origin = vec3<f32>(0.0, r0, 0.0);
    var l = vec3<f32>(0.0);
    var t_acc = vec3<f32>(1.0);
    var prev = 0.0;
    let n = f32(ATMO_SKY_STEPS);
    for (var i = 0u; i < ATMO_SKY_STEPS; i++) {
        let t1 = tmax * pow(f32(i + 1u) / n, 2.0);
        let dt = t1 - prev;
        let tm = 0.5 * (prev + t1);
        let pos = origin + dir * tm;
        let rr = length(pos);
        let m = atmo_medium(p, rr - p.planet.x);
        let mu_s = dot(pos, sun) / rr;
        let sun_t = atmo_transmittance(p, tlut, samp, rr, mu_s);
        let ms = atmo_multiscatter(p, mslut, samp, rr, mu_s);
        let s = p.sun_irradiance.rgb
            * (sun_t * (m.ray * pr + vec3<f32>(m.mie * pm)) + atmo_scatter(m) * ms);
        let step = exp(-m.ext * dt);
        l += t_acc * s * (vec3<f32>(1.0) - step) / m.ext;
        t_acc *= step;
        prev = t1;
    }
    if hits {
        let pos = origin + dir * tmax;
        let mu_g = dot(normalize(pos), sun);
        let e = p.sun_irradiance.rgb * atmo_transmittance(p, tlut, samp, p.planet.x, mu_g) * max(mu_g, 0.0);
        l += t_acc * p.ground.rgb / ATMO_PI * e;
    }
    return l;
}

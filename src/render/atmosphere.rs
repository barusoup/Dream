//! Physically based sky (Hillaire 2020, "A Scalable and Production Ready Sky
//! and Atmosphere Rendering Technique"): the settings, and a CPU mirror of the
//! GPU model in `gfx/atmo_*.wgsl`.
//!
//! The GPU builds a transmittance LUT and a multiple-scattering LUT once per
//! settings change, and a sky-view LUT every frame. [`AtmosphereModel`] builds
//! the same two static LUTs on the CPU and evaluates the same integrals, so the
//! CPU sky agrees with the GPU sky-view texels. The CPU model is used to drive
//! the forward lighting ([`AtmosphereSettings::drive_lighting`]) and by the
//! scene-check reference.
//!
//! Geometry runs in kilometres inside the maths (as in the paper) to keep the
//! `f32` ray-sphere terms accurate; the settings are in metres.

use crate::lighting::LightingEnv;
use glam::Vec3;

/// Earth Rayleigh scattering at sea level, per metre (RGB).
pub const RAYLEIGH_SCATTERING_M: [f32; 3] = [5.802e-6, 13.558e-6, 33.1e-6];
/// Rayleigh density scale height, metres.
pub const RAYLEIGH_HEIGHT_M: f32 = 8_000.0;
/// Mie scattering per metre.
pub const MIE_SCATTERING_M: f32 = 3.996e-6;
/// Mie extinction (scattering plus absorption) per metre.
pub const MIE_EXTINCTION_M: f32 = 4.44e-6;
/// Mie density scale height, metres.
pub const MIE_HEIGHT_M: f32 = 1_200.0;
/// Ozone absorption per metre at the density peak (RGB).
pub const OZONE_ABSORPTION_M: [f32; 3] = [0.650e-6, 1.881e-6, 0.085e-6];
/// The single scale from the sun colour to the sky's top-of-atmosphere
/// irradiance. The visible sky (sky-view LUT, sky pass), the sky environment map
/// that reflections and ambient specular read, and the ambient irradiance that
/// [`AtmosphereModel::driven_environment`] derives all use it, so they agree.
/// Sun-lit surfaces still use `sun_color` unscaled.
///
/// A scale of 1 is physically exact relative to the light on surfaces, but at
/// an exposure that keeps sunlit white below clipping a physical sky renders as
/// a very dark blue (zenith radiance is roughly 1% of the sun's irradiance).
/// This is an artistic lift, the same job a camera's tone curve does: 6 puts the
/// default sun (intensity 3) at a mid-blue zenith and a bright but unclipped
/// horizon. Use [`AtmosphereSettings::sky_intensity`] to move the visible sky
/// either way (`1.0 / 6.0` there gives the physical ratio).
pub const SKY_IRRADIANCE_SCALE: f32 = 2.5;
/// Size of the sky environment map (equirectangular, see [`sky_env_uv`]).
pub const SKY_ENV_SIZE: (u32, u32) = (1024, 512);
/// Mip levels of the sky environment map, down to 4 x 2.
pub const SKY_ENV_MIPS: u32 = 9;
/// Transmittance LUT size (view-zenith cosine x altitude).
pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
/// Multiple-scattering LUT size (sun-zenith cosine x altitude).
pub const MULTISCATTER_SIZE: (u32, u32) = (32, 32);
/// Sky-view LUT size (longitude relative to the sun x latitude).
pub const SKYVIEW_SIZE: (u32, u32) = (192, 108);

const TRANSMITTANCE_STEPS: usize = 40;
const MS_DIRECTIONS: usize = 64;
const MS_STEPS: usize = 20;
/// Raymarch steps of the sky-view integral (GPU and CPU alike).
pub const SKY_STEPS: usize = 40;

/// Atmosphere of the sky pass and lighting drive. Build from one of the
/// presets and assign the fields you change; the struct is `#[non_exhaustive]`.
/// Values are clamped by the renderer.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AtmosphereSettings {
    /// Planet radius, metres.
    pub planet_radius: f32,
    /// Height of the atmosphere above the planet surface, metres.
    pub atmosphere_height: f32,
    /// Multiplier on Earth's Rayleigh scattering (blue skies at 1).
    pub rayleigh_scale: f32,
    /// Multiplier on Earth's Mie scattering and extinction (haze).
    pub mie_scale: f32,
    /// Mie phase anisotropy `g` (Cornette-Shanks), in `[-0.99, 0.99]`.
    pub mie_anisotropy: f32,
    /// Multiplier on Earth's ozone absorption.
    pub ozone_scale: f32,
    /// Lambertian albedo of the ground seen below the horizon and bouncing light.
    pub ground_albedo: [f32; 3],
    /// Draw the sun disk in the sky.
    pub sun_disk: bool,
    /// Angular radius of the sun disk, radians. The real sun is about 0.00465;
    /// larger values give a stylised, dimmer sun.
    pub sun_angular_radius: f32,
    /// World y of altitude zero.
    pub sea_level: f32,
    /// Replace the environment `render_scene` uploads with one matching the
    /// sky: the sun colour is attenuated by the transmittance toward the sun,
    /// and `sky_zenith`, `sky_horizon` and `sky_ground` come from the sky. The
    /// caller's `SceneFrame::env` is not changed.
    pub drive_lighting: bool,
    /// Artistic multiplier on the sky radiance (the sun disk is not scaled).
    pub sky_intensity: f32,
    /// Draw the model's virtual planet ground below the horizon. `false` (the
    /// default) continues the horizon colour downward as distant haze in the sky
    /// pass, which otherwise shows the planet's flat ground as a band wherever
    /// scene geometry ends. The driven lighting and the LUTs are unaffected.
    pub show_ground: bool,
    /// Radiance of the night sky, linear RGB, added everywhere the sky is: the
    /// sky pass (upper hemisphere and horizon haze), the sky environment map and
    /// the driven ambient irradiance. A moonless night is dark but not black.
    /// Not scaled by `sky_intensity`, and not attenuated by the transmittance.
    /// Non-finite or negative components become zero.
    pub night_sky: [f32; 3],
}

impl Default for AtmosphereSettings {
    fn default() -> Self {
        Self::earth()
    }
}

impl AtmosphereSettings {
    /// Earth's atmosphere with the paper's coefficients.
    pub fn earth() -> Self {
        Self {
            planet_radius: 6_360_000.0,
            atmosphere_height: 100_000.0,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            mie_anisotropy: 0.8,
            ozone_scale: 1.0,
            ground_albedo: [0.3; 3],
            sun_disk: true,
            sun_angular_radius: 0.00465,
            sea_level: 0.0,
            drive_lighting: true,
            sky_intensity: 1.0,
            show_ground: false,
            night_sky: [0.010, 0.016, 0.032],
        }
    }

    /// Earth with four times the aerosol load: a milky, washed-out sky.
    pub fn hazy() -> Self {
        Self {
            mie_scale: 4.0,
            mie_anisotropy: 0.76,
            ..Self::earth()
        }
    }

    /// Earth with little aerosol: a deep, saturated sky.
    pub fn clear() -> Self {
        Self {
            mie_scale: 0.5,
            ..Self::earth()
        }
    }

    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        let albedo = self.ground_albedo.map(|c| finite(c, 0.3).clamp(0.0, 1.0));
        let planet_radius = finite(self.planet_radius, 6_360_000.0).clamp(100_000.0, 1.0e8);
        Self {
            planet_radius,
            atmosphere_height: finite(self.atmosphere_height, 100_000.0)
                .clamp(1_000.0, planet_radius * 0.25),
            rayleigh_scale: finite(self.rayleigh_scale, 1.0).clamp(0.0, 20.0),
            mie_scale: finite(self.mie_scale, 1.0).clamp(0.0, 50.0),
            mie_anisotropy: finite(self.mie_anisotropy, 0.8).clamp(-0.99, 0.99),
            ozone_scale: finite(self.ozone_scale, 1.0).clamp(0.0, 20.0),
            ground_albedo: albedo,
            sun_disk: self.sun_disk,
            sun_angular_radius: finite(self.sun_angular_radius, 0.00465).clamp(1e-4, 0.2),
            sea_level: finite(self.sea_level, 0.0),
            drive_lighting: self.drive_lighting,
            sky_intensity: finite(self.sky_intensity, 1.0).clamp(0.0, 100.0),
            show_ground: self.show_ground,
            night_sky: self.night_sky.map(|c| finite(c, 0.0).max(0.0)),
        }
    }
}

/// Planet and coefficients in kilometres, as the shaders read them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanetKm {
    pub rg: f32,
    pub rt: f32,
    pub rayleigh: Vec3,
    pub mie_scatter: f32,
    pub mie_extinction: f32,
    pub ozone: Vec3,
    pub mie_g: f32,
    pub albedo: Vec3,
}

impl PlanetKm {
    pub(crate) fn from_settings(s: &AtmosphereSettings) -> Self {
        let per_km = 1000.0;
        let r = RAYLEIGH_SCATTERING_M.map(|c| c * per_km * s.rayleigh_scale);
        let oz = OZONE_ABSORPTION_M.map(|c| c * per_km * s.ozone_scale);
        Self {
            rg: s.planet_radius / 1000.0,
            rt: (s.planet_radius + s.atmosphere_height) / 1000.0,
            rayleigh: Vec3::from(r),
            mie_scatter: MIE_SCATTERING_M * per_km * s.mie_scale,
            mie_extinction: MIE_EXTINCTION_M * per_km * s.mie_scale,
            ozone: Vec3::from(oz),
            mie_g: s.mie_anisotropy,
            albedo: Vec3::from(s.ground_albedo),
        }
    }
}

/// Density of the three media at one altitude.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Medium {
    pub ray: Vec3,
    pub mie: f32,
    pub ext: Vec3,
}

impl Medium {
    /// Total scattering coefficient.
    pub(crate) fn scatter(&self) -> Vec3 {
        self.ray + Vec3::splat(self.mie)
    }
}

pub(crate) fn medium_at(p: &PlanetKm, h_km: f32) -> Medium {
    let hr = (-h_km / (RAYLEIGH_HEIGHT_M / 1000.0)).exp();
    let hm = (-h_km / (MIE_HEIGHT_M / 1000.0)).exp();
    let ozone = (1.0 - (h_km - 25.0).abs() / 15.0).max(0.0);
    let ray = p.rayleigh * hr;
    let mie = p.mie_scatter * hm;
    Medium {
        ray,
        mie,
        ext: ray + Vec3::splat(p.mie_extinction * hm) + p.ozone * ozone,
    }
}

/// Rayleigh phase function of the cosine between view ray and sun.
pub(crate) fn phase_rayleigh(c: f32) -> f32 {
    3.0 / (16.0 * std::f32::consts::PI) * (1.0 + c * c)
}

/// Cornette-Shanks Mie phase function.
pub(crate) fn phase_mie(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    3.0 / (8.0 * std::f32::consts::PI) * (1.0 - g2) / (2.0 + g2) * (1.0 + c * c)
        / (1.0 + g2 - 2.0 * g * c).max(1e-4).powf(1.5)
}

/// Texel-centre coordinate of a unit-range value on an `n`-texel axis.
pub(crate) fn unit_to_tex(x: f32, n: f32) -> f32 {
    0.5 / n + x * (1.0 - 1.0 / n)
}

/// Inverse of [`unit_to_tex`].
pub(crate) fn tex_to_unit(u: f32, n: f32) -> f32 {
    (u - 0.5 / n) / (1.0 - 1.0 / n)
}

impl PlanetKm {
    /// Distance to the top of the atmosphere from radius `r` along `mu`.
    pub(crate) fn top_distance(&self, r: f32, mu: f32) -> f32 {
        -r * mu
            + (r * r * mu * mu - r * r + self.rt * self.rt)
                .max(0.0)
                .sqrt()
    }

    /// Distance to the ground along a ray that hits it.
    pub(crate) fn ground_distance(&self, r: f32, mu: f32) -> f32 {
        -r * mu
            - (r * r * mu * mu - r * r + self.rg * self.rg)
                .max(0.0)
                .sqrt()
    }

    pub(crate) fn hits_ground(&self, r: f32, mu: f32) -> bool {
        mu < 0.0 && r * r * (mu * mu - 1.0) + self.rg * self.rg >= 0.0
    }

    /// Transmittance LUT coordinates of (radius, zenith cosine), in `[0, 1]^2`.
    pub(crate) fn tlut_uv(&self, r: f32, mu: f32) -> (f32, f32) {
        let h = (self.rt * self.rt - self.rg * self.rg).sqrt();
        let rho = (r * r - self.rg * self.rg).max(0.0).sqrt();
        let d = self.top_distance(r, mu);
        let d_min = self.rt - r;
        let d_max = rho + h;
        let x_mu = (d - d_min) / (d_max - d_min).max(1e-6);
        (
            unit_to_tex(x_mu, TRANSMITTANCE_SIZE.0 as f32),
            unit_to_tex(rho / h, TRANSMITTANCE_SIZE.1 as f32),
        )
    }

    /// (radius, zenith cosine) of a transmittance LUT texel-space uv.
    pub(crate) fn tlut_params(&self, u: f32, v: f32) -> (f32, f32) {
        let h = (self.rt * self.rt - self.rg * self.rg).sqrt();
        let x_mu = tex_to_unit(u, TRANSMITTANCE_SIZE.0 as f32);
        let x_r = tex_to_unit(v, TRANSMITTANCE_SIZE.1 as f32);
        let rho = h * x_r;
        let r = (rho * rho + self.rg * self.rg).sqrt();
        let d_min = self.rt - r;
        let d_max = rho + h;
        let d = d_min + x_mu * (d_max - d_min);
        let mu = if d <= 1e-6 {
            1.0
        } else {
            ((self.rt * self.rt - r * r - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
        };
        (r, mu)
    }

    /// Optical depth to the top along (radius, zenith cosine), integrated
    /// directly: the transmittance LUT's texel value.
    pub(crate) fn transmittance_integral(&self, r: f32, mu: f32) -> Vec3 {
        let dist = self.top_distance(r, mu);
        let mut od = Vec3::ZERO;
        let mut prev = 0.0;
        for i in 0..TRANSMITTANCE_STEPS {
            let t = dist * (i + 1) as f32 / TRANSMITTANCE_STEPS as f32;
            let dt = t - prev;
            let tm = prev + 0.5 * dt;
            let rr = (r * r + 2.0 * r * mu * tm + tm * tm).max(0.0).sqrt();
            od += medium_at(self, rr - self.rg).ext * dt;
            prev = t;
        }
        (-od).exp()
    }
}

/// Bilinear fetch of a LUT with clamp-to-edge, as `textureSampleLevel` does.
pub(crate) fn bilinear(lut: &[Vec3], w: usize, h: usize, u: f32, v: f32) -> Vec3 {
    let x = u * w as f32 - 0.5;
    let y = v * h as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let at = |xi: f32, yi: f32| {
        let xc = (xi.max(0.0) as usize).min(w - 1);
        let yc = (yi.max(0.0) as usize).min(h - 1);
        lut[yc * w + xc]
    };
    let top = at(x0, y0).lerp(at(x0 + 1.0, y0), fx);
    let bottom = at(x0, y0 + 1.0).lerp(at(x0 + 1.0, y0 + 1.0), fx);
    top.lerp(bottom, fy)
}

/// CPU atmosphere: the static LUTs and the sky integral, for one settings value.
///
/// Build once per settings change ([`Self::new`] takes about tens of
/// milliseconds in release). Evaluation is cheap enough to call for a handful of
/// directions per frame.
#[derive(Clone, Debug)]
pub struct AtmosphereModel {
    settings: AtmosphereSettings,
    planet: PlanetKm,
    transmittance: Vec<Vec3>,
    multiscatter: Vec<Vec3>,
}

impl AtmosphereModel {
    /// Build the LUTs for `settings` (clamped as the renderer clamps them).
    pub fn new(settings: &AtmosphereSettings) -> Self {
        let settings = settings.sanitised();
        let planet = PlanetKm::from_settings(&settings);
        let (tw, th) = TRANSMITTANCE_SIZE;
        let mut transmittance = Vec::with_capacity((tw * th) as usize);
        for y in 0..th {
            for x in 0..tw {
                let u = (x as f32 + 0.5) / tw as f32;
                let v = (y as f32 + 0.5) / th as f32;
                let (r, mu) = planet.tlut_params(u, v);
                transmittance.push(planet.transmittance_integral(r, mu));
            }
        }
        let mut model = Self {
            settings,
            planet,
            transmittance,
            multiscatter: Vec::new(),
        };
        model.multiscatter = model.build_multiscatter();
        model
    }

    /// The settings this model was built from (sanitised).
    pub fn settings(&self) -> &AtmosphereSettings {
        &self.settings
    }

    fn lut_transmittance(&self, r: f32, mu: f32) -> Vec3 {
        if self.planet.hits_ground(r, mu) {
            return Vec3::ZERO;
        }
        let (u, v) = self.planet.tlut_uv(r, mu);
        bilinear(
            &self.transmittance,
            TRANSMITTANCE_SIZE.0 as usize,
            TRANSMITTANCE_SIZE.1 as usize,
            u,
            v,
        )
    }

    fn lut_multiscatter(&self, r: f32, mu_s: f32) -> Vec3 {
        let (u, v) = multiscatter_uv(&self.planet, r, mu_s);
        bilinear(
            &self.multiscatter,
            MULTISCATTER_SIZE.0 as usize,
            MULTISCATTER_SIZE.1 as usize,
            u,
            v,
        )
    }

    fn build_multiscatter(&self) -> Vec<Vec3> {
        let (w, h) = MULTISCATTER_SIZE;
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                let (r, mu_s) = multiscatter_params(&self.planet, u, v);
                out.push(self.multiscatter_integral(r, mu_s));
            }
        }
        out
    }

    /// Multiple-scattering factor psi (sun irradiance 1) at (radius, sun cosine).
    fn multiscatter_integral(&self, r: f32, mu_s: f32) -> Vec3 {
        let p = &self.planet;
        let sun = Vec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), mu_s, 0.0);
        let origin = Vec3::new(0.0, r, 0.0);
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        let mut acc_l = Vec3::ZERO;
        let mut acc_f = Vec3::ZERO;
        for i in 0..MS_DIRECTIONS {
            let z = 1.0 - (2 * i + 1) as f32 / MS_DIRECTIONS as f32;
            let s = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f32 * golden;
            let dir = Vec3::new(s * phi.cos(), z, s * phi.sin());
            let tmax = if p.hits_ground(r, dir.y) {
                p.ground_distance(r, dir.y)
            } else {
                p.top_distance(r, dir.y)
            };
            let dt = tmax / MS_STEPS as f32;
            let mut t_acc = Vec3::ONE;
            let mut l = Vec3::ZERO;
            let mut f = Vec3::ZERO;
            for j in 0..MS_STEPS {
                let t = (j as f32 + 0.5) * dt;
                let pos = origin + dir * t;
                let rr = pos.length();
                let m = medium_at(p, rr - p.rg);
                let mu_s_t = pos.dot(sun) / rr;
                let sun_t = self.lut_transmittance(rr, mu_s_t);
                let sint = (Vec3::ONE - (-m.ext * dt).exp()) / m.ext;
                let scat = m.scatter();
                l += t_acc * scat * sun_t * sint;
                f += t_acc * scat * sint;
                t_acc *= (-m.ext * dt).exp();
            }
            acc_l += l;
            acc_f += f;
        }
        let n = MS_DIRECTIONS as f32;
        let mean_l = acc_l / n;
        let f = (acc_f / n).clamp(Vec3::ZERO, Vec3::splat(0.99));
        mean_l / (4.0 * std::f32::consts::PI) / (Vec3::ONE - f)
    }

    /// Sky radiance (no sun disk, no night floor) seen from `altitude` metres
    /// above the sea level along `dir` (y up), for the sun direction `sun_dir`
    /// and the engine's sun colour `sun_color`. Matches the GPU sky-view LUT. The
    /// GPU sky adds [`AtmosphereSettings::night_sky`] on top of this.
    pub fn sky_radiance(&self, altitude: f32, dir: Vec3, sun_dir: Vec3, sun_color: Vec3) -> Vec3 {
        let p = &self.planet;
        let r0 = p.rg + (altitude.max(1.0) / 1000.0);
        let irradiance = sun_color * SKY_IRRADIANCE_SCALE;
        let mut radiance = self.sky_integral(r0, dir.normalize(), sun_dir.normalize(), irradiance);
        radiance *= self.settings.sky_intensity;
        radiance
    }

    fn sky_integral(&self, r0: f32, dir: Vec3, sun: Vec3, irradiance: Vec3) -> Vec3 {
        let p = &self.planet;
        let mu = dir.y;
        let hits = p.hits_ground(r0, mu);
        let tmax = if hits {
            p.ground_distance(r0, mu)
        } else {
            p.top_distance(r0, mu)
        };
        let cos_sd = dir.dot(sun);
        let pr = phase_rayleigh(cos_sd);
        let pm = phase_mie(cos_sd, p.mie_g);
        let origin = Vec3::new(0.0, r0, 0.0);
        let mut l = Vec3::ZERO;
        let mut t_acc = Vec3::ONE;
        let mut prev = 0.0;
        let n = SKY_STEPS as f32;
        for i in 0..SKY_STEPS {
            let t1 = tmax * ((i + 1) as f32 / n).powi(2);
            let t0 = prev;
            let dt = t1 - t0;
            let tm = 0.5 * (t0 + t1);
            let pos = origin + dir * tm;
            let rr = pos.length();
            let m = medium_at(p, rr - p.rg);
            let mu_s = pos.dot(sun) / rr;
            let sun_t = self.lut_transmittance(rr, mu_s);
            let ms = self.lut_multiscatter(rr, mu_s);
            let s =
                irradiance * (sun_t * (m.ray * pr + Vec3::splat(m.mie * pm)) + m.scatter() * ms);
            let step = (-m.ext * dt).exp();
            l += t_acc * s * (Vec3::ONE - step) / m.ext;
            t_acc *= step;
            prev = t1;
        }
        if hits {
            let pos = origin + dir * tmax;
            let mu_g = pos.normalize().dot(sun);
            let e = irradiance * self.lut_transmittance(p.rg, mu_g) * mu_g.max(0.0);
            l += t_acc * p.albedo / std::f32::consts::PI * e;
        }
        l
    }

    /// Transmittance from `altitude` toward the sun direction `sun_dir` (zero
    /// when the planet blocks it).
    pub fn transmittance(&self, altitude: f32, sun_dir: Vec3) -> Vec3 {
        let r0 = self.planet.rg + altitude.max(1.0) / 1000.0;
        self.lut_transmittance(r0, sun_dir.normalize().y)
    }

    /// The environment the forward pass uses when
    /// [`AtmosphereSettings::drive_lighting`] is on: `env.sun_color` attenuated
    /// by the sun's transmittance (with a short fade through the horizon), and
    /// the three sky colours as irradiance-derived values, with `params.y` set to
    /// 1 to say so. Other fields pass through unchanged.
    ///
    /// The sky colours are irradiance over pi, so a Lambertian surface with albedo
    /// `a` and normal `n` receives `a * sky_irradiance(n)` (see `lighting.wgsl`):
    /// - `sky_zenith` = `E_up / pi`, the cosine-weighted irradiance of the upper
    ///   hemisphere on an upward surface;
    /// - `sky_horizon` = `E_side / pi`, averaged over eight azimuths, where
    ///   `E_side` integrates the full sphere's radiance against a vertical normal
    ///   (sky above the horizon, ground below);
    /// - `sky_ground` = `ground_albedo * (E_up + E_sun) / pi`, the radiance of the
    ///   ground the sun and sky light.
    ///
    /// Every sky sample uses the caller's sun colour, so the values match the
    /// visible sky. The night floor ([`AtmosphereSettings::night_sky`]) is part of
    /// each radiance sample, so it reaches the irradiance without special cases.
    /// Costs 128 sky evaluations; `Renderer` caches the result while the sun,
    /// the sun colour, the altitude (to 10 m) and the settings are steady.
    pub fn driven_environment(
        &self,
        env: &LightingEnv,
        altitude: f32,
        coverage: f32,
    ) -> LightingEnv {
        use std::f32::consts::{FRAC_PI_2, PI, TAU};
        let sun = Vec3::from_slice(&env.sun_direction[..3]).normalize_or_zero();
        let sun = if sun == Vec3::ZERO { Vec3::Y } else { sun };
        let sun_color = Vec3::from_slice(&env.sun_color[..3]);
        let trans = self.transmittance(altitude, sun);
        let fade = smoothstep(-0.03, 0.03, sun.y);
        let sun_out = sun_color * trans * fade;
        let night = Vec3::from(self.settings.night_sky);

        // Upper hemisphere on a midpoint grid: each sample's solid angle and true
        // radiance (sky plus night floor). Reused for every normal below.
        const POLAR: usize = 8;
        const AZIMUTH: usize = 16;
        let dtheta = FRAC_PI_2 / POLAR as f32;
        let dphi = TAU / AZIMUTH as f32;
        let mut samples = Vec::with_capacity(POLAR * AZIMUTH);
        let mut e_up = Vec3::ZERO;
        for j in 0..POLAR {
            let theta = (j as f32 + 0.5) * dtheta;
            let (st, ct) = theta.sin_cos();
            for i in 0..AZIMUTH {
                let phi = (i as f32 + 0.5) * dphi;
                let dir = Vec3::new(st * phi.cos(), ct, st * phi.sin());
                let l = self.sky_radiance(altitude, dir, sun, sun_color) + night;
                let w = st * dtheta * dphi;
                e_up += l * ct * w;
                samples.push((dir, w, l));
            }
        }
        let e_sun = sun_out * sun.y.max(0.0);
        let ground = self.settings.ground_albedo_vec() * ((e_up + e_sun) / PI);
        let mut zenith = e_up / PI;

        // Vertical normals: the upper samples weighted by the cosine to the
        // normal, plus the uniform ground radiance over the lower half, whose
        // cosine integral is pi / 2 for any horizontal normal.
        let azimuths = 8;
        let mut horizon = Vec3::ZERO;
        for a in 0..azimuths {
            let az = a as f32 / azimuths as f32 * TAU;
            let n = Vec3::new(az.cos(), 0.0, az.sin());
            let mut e_side = ground * (PI / 2.0);
            for &(dir, w, l) in &samples {
                e_side += l * n.dot(dir).max(0.0) * w;
            }
            horizon += e_side / PI;
        }
        horizon /= azimuths as f32;

        // Clouds: the cover blends the sky colours toward a neutral overcast deck. Its
        // irradiance is the sun's horizontal part (a quarter, as the deck diffuses it)
        // plus the sky's, and only its luminance is kept, so it is white-grey.
        let coverage = coverage.clamp(0.0, 1.0);
        if coverage > 0.0 {
            let e_overcast = e_sun * 0.25 + e_up;
            let luma = e_overcast.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            let overcast = Vec3::splat(luma / PI);
            zenith = zenith.lerp(overcast, coverage);
            horizon = horizon.lerp(overcast, coverage);
        }

        let mut out = *env;
        out.sun_color = [sun_out.x, sun_out.y, sun_out.z, 0.0];
        out.sky_zenith = zenith.extend(0.0).to_array();
        out.sky_horizon = horizon.extend(0.0).to_array();
        out.sky_ground = ground.extend(0.0).to_array();
        out.params[1] = 1.0;
        out
    }
}

impl AtmosphereSettings {
    fn ground_albedo_vec(&self) -> Vec3 {
        Vec3::from(self.ground_albedo)
    }
}

fn multiscatter_uv(p: &PlanetKm, r: f32, mu_s: f32) -> (f32, f32) {
    let x = unit_to_tex(mu_s * 0.5 + 0.5, MULTISCATTER_SIZE.0 as f32);
    let alt = ((r - p.rg) / (p.rt - p.rg)).clamp(0.0, 1.0);
    (x, unit_to_tex(alt, MULTISCATTER_SIZE.1 as f32))
}

fn multiscatter_params(p: &PlanetKm, u: f32, v: f32) -> (f32, f32) {
    let mu_s = tex_to_unit(u, MULTISCATTER_SIZE.0 as f32) * 2.0 - 1.0;
    let alt = tex_to_unit(v, MULTISCATTER_SIZE.1 as f32);
    (p.rg + alt * (p.rt - p.rg), mu_s)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Sky-view LUT mapping. The texel grid covers the full sphere around the
/// camera: `u` is the azimuth relative to the sun (`u = 0.5` faces the sun), and
/// `v` the zenith angle, concentrated at the horizon. Returns the view direction
/// (y up) of a uv for the camera at `altitude` metres above sea level.
pub fn sky_view_direction(
    model: &AtmosphereModel,
    altitude: f32,
    uv: [f32; 2],
    sun_dir: Vec3,
) -> Vec3 {
    let r = model.planet.rg + altitude.max(1.0) / 1000.0;
    let (e1, e2) = sun_basis(sun_dir);
    let theta_h = horizon_zenith(&model.planet, r);
    let v = uv[1];
    let theta = if v <= 0.5 {
        theta_h * (2.0 * v).sqrt()
    } else {
        theta_h + (std::f32::consts::PI - theta_h) * (2.0 * v - 1.0)
    };
    let phi = (uv[0] - 0.5) * std::f32::consts::TAU;
    let (st, ct) = theta.sin_cos();
    (e1 * (st * phi.cos()) + e2 * (st * phi.sin()) + Vec3::Y * ct).normalize()
}

/// Inverse of [`sky_view_direction`]: the sky-view uv a view direction maps to.
pub fn sky_view_uv(model: &AtmosphereModel, altitude: f32, dir: Vec3, sun_dir: Vec3) -> [f32; 2] {
    let r = model.planet.rg + altitude.max(1.0) / 1000.0;
    let (e1, e2) = sun_basis(sun_dir);
    let d = dir.normalize();
    let theta_h = horizon_zenith(&model.planet, r);
    let theta = d.y.clamp(-1.0, 1.0).acos();
    let v = if theta <= theta_h {
        0.5 * (theta / theta_h).powi(2)
    } else {
        0.5 + 0.5 * (theta - theta_h) / (std::f32::consts::PI - theta_h)
    };
    let phi = d.dot(e2).atan2(d.dot(e1));
    [phi / std::f32::consts::TAU + 0.5, v]
}

/// Sky environment map mapping (the texture reflections and ambient specular
/// read). Equirectangular with `u = atan2(z, x) / 2pi + 1/2` and
/// `v = acos(y) / pi`, so `v = 0` is straight up. The WGSL twin is
/// `sky_env_uv` in `lighting.wgsl`.
pub fn sky_env_uv(dir: Vec3) -> [f32; 2] {
    let d = dir.normalize_or_zero();
    let u = d.z.atan2(d.x) / std::f32::consts::TAU + 0.5;
    let v = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    [u, v]
}

/// Inverse of [`sky_env_uv`]: the unit direction (y up) of a map uv.
pub fn sky_env_direction(uv: [f32; 2]) -> Vec3 {
    let phi = (uv[0] - 0.5) * std::f32::consts::TAU;
    let theta = uv[1] * std::f32::consts::PI;
    let (st, ct) = theta.sin_cos();
    Vec3::new(st * phi.cos(), ct, st * phi.sin())
}

/// Zenith angle of the horizon ray (the grazing ray that touches the ground).
fn horizon_zenith(p: &PlanetKm, r: f32) -> f32 {
    (-(1.0 - (p.rg / r) * (p.rg / r)).max(0.0).sqrt()).acos()
}

/// Orthonormal horizontal basis of the sky-view LUT: `e1` toward the sun and
/// `e2 = up x e1`.
pub(crate) fn sun_basis(sun: Vec3) -> (Vec3, Vec3) {
    let h = Vec3::new(sun.x, 0.0, sun.z);
    let e1 = if h.length() < 1e-4 {
        Vec3::X
    } else {
        h.normalize()
    };
    (e1, Vec3::Y.cross(e1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn atmosphere_settings_sanitise_and_presets() {
        assert_eq!(AtmosphereSettings::default(), AtmosphereSettings::earth());
        let wild = AtmosphereSettings {
            planet_radius: f32::NAN,
            mie_anisotropy: 5.0,
            ground_albedo: [2.0, -1.0, f32::INFINITY],
            sun_angular_radius: 0.0,
            ..AtmosphereSettings::earth()
        }
        .sanitised();
        assert_eq!(wild.planet_radius, 6_360_000.0);
        assert_eq!(wild.mie_anisotropy, 0.99);
        assert_eq!(wild.ground_albedo, [1.0, 0.0, 0.3]);
        assert!(wild.sun_angular_radius >= 1e-4);
        assert!(AtmosphereSettings::hazy().mie_scale > AtmosphereSettings::clear().mie_scale);
    }

    #[test]
    fn atmosphere_uniform_coefficients_are_in_range() {
        let p = PlanetKm::from_settings(&AtmosphereSettings::earth());
        assert!((p.rayleigh.z - 0.0331).abs() < 1e-5);
        assert!((p.rt - p.rg - 100.0).abs() < 1e-3);
    }

    #[test]
    fn atmosphere_zenith_sky_is_blue_and_horizon_is_brighter() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let sun = Vec3::new(0.0, 60f32.to_radians().sin(), 0.0)
            + Vec3::new(60f32.to_radians().cos(), 0.0, 0.0);
        let sun_color = Vec3::splat(1.0);
        let zenith = model.sky_radiance(0.0, Vec3::Y, sun, sun_color);
        assert!(zenith.z > zenith.y && zenith.y > zenith.x, "{zenith:?}");
        assert!(zenith.z / zenith.x > 2.0, "{zenith:?}");
        let horizon =
            model.sky_radiance(0.0, Vec3::new(1.0, 0.01, 0.0).normalize(), sun, sun_color);
        assert!(horizon.length() > zenith.length(), "{horizon:?} {zenith:?}");
        let sat = |c: Vec3| (c.z - c.x) / c.z.max(1e-6);
        assert!(sat(horizon) < sat(zenith));
    }

    #[test]
    fn atmosphere_low_sun_is_red_and_dim() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let low = Vec3::new(2f32.to_radians().cos(), 2f32.to_radians().sin(), 0.0);
        let noon = Vec3::new(60f32.to_radians().cos(), 60f32.to_radians().sin(), 0.0);
        let t_low = model.transmittance(0.0, low);
        let t_noon = model.transmittance(0.0, noon);
        assert!(t_low.x > t_low.y && t_low.y > t_low.z, "{t_low:?}");
        assert!(
            t_low.length() < 0.5 * t_noon.length(),
            "{t_low:?} {t_noon:?}"
        );
    }

    #[test]
    fn atmosphere_driven_sun_is_zero_below_horizon() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let az = 0.6f32;
        let below = (-10f32).to_radians();
        let env = LightingEnv::with_sun(az, below, 3.0);
        let driven = model.driven_environment(&env, 0.0, 0.0);
        let sun_max = driven.sun_color[..3].iter().fold(0.0f32, |m, &c| m.max(c));
        assert!(sun_max < 1e-3, "driven sun {sun_max}");
    }

    #[test]
    fn atmosphere_transmittance_straight_up_is_plausible() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let t = model.transmittance(0.0, Vec3::Y);
        for (c, expect) in [(t.x, 0.9), (t.y, 0.85), (t.z, 0.75)] {
            assert!((c - expect).abs() < 0.08, "{t:?}");
        }
    }

    /// Driven ambient at a range of sun elevations: finite, the up-facing
    /// irradiance a plausible fraction of the direct horizontal irradiance at
    /// 60 degrees, and nothing exactly zero below the horizon.
    #[test]
    fn atmosphere_driven_ambient_is_plausible_across_sun_elevations() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        for deg in [60.0f32, 20.0, 5.0, -5.0, -20.0] {
            let env = LightingEnv::with_sun(0.6, deg.to_radians(), 3.0);
            let driven = model.driven_environment(&env, 2.0, 0.0);
            for c in [driven.sky_zenith, driven.sky_horizon, driven.sky_ground] {
                assert!(c[..3].iter().all(|v| v.is_finite()), "{deg}: {c:?}");
            }
            let sun = Vec3::from_slice(&env.sun_direction[..3]);
            let luma = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            let direct_h = luma(Vec3::from_slice(&env.sun_color[..3])) * sun.y;
            let zenith_e = luma(Vec3::from_slice(&driven.sky_zenith[..3])) * PI;
            if deg == 60.0 {
                let ratio = zenith_e / direct_h;
                assert!((0.05..=0.40).contains(&ratio), "60 deg: ratio {ratio}");
            }
            if deg < 0.0 {
                for c in [driven.sky_zenith, driven.sky_horizon, driven.sky_ground] {
                    assert!(c[..3].iter().all(|&v| v > 0.0), "{deg}: {c:?}");
                }
            }
        }
    }

    /// Per-call cost of [`AtmosphereModel::driven_environment`]:
    /// `cargo test --release --lib driven_environment_profile -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn driven_environment_profile() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let env = LightingEnv::default();
        let start = std::time::Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..1000 {
            sink += model.driven_environment(&env, 2.0, 0.0).sky_zenith[0];
        }
        let per_call = start.elapsed().as_secs_f64() * 1e6 / 1000.0;
        println!("driven_environment: {per_call:.1} us per call (sink {sink})");
    }

    #[test]
    fn driven_environment_cloud_cover_blends_sky_toward_overcast() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let env = LightingEnv::with_sun(2.2, 0.6, 3.0);
        let clear = model.driven_environment(&env, 2.0, 0.0);
        let fair = model.driven_environment(&env, 2.0, 0.45);
        let full = model.driven_environment(&env, 2.0, 1.0);
        println!("sky_zenith clear {:?}", &clear.sky_zenith[..3]);
        println!(
            "sky_zenith fair weather (coverage 0.45) {:?}",
            &fair.sky_zenith[..3]
        );
        println!(
            "sky_zenith overcast (coverage 1) {:?}",
            &full.sky_zenith[..3]
        );
        let chroma = |v: [f32; 4]| (v[0] - v[2]).abs();
        assert_eq!(
            clear.sky_zenith,
            model.driven_environment(&env, 2.0, 0.0).sky_zenith
        );
        assert!(chroma(fair.sky_zenith) < chroma(clear.sky_zenith));
        assert!(chroma(full.sky_zenith) < 1e-4);
    }

    #[test]
    fn atmosphere_sky_env_mapping_round_trips() {
        // Directions on a grid over the whole sphere, away from the poles where
        // u is degenerate.
        for i in 0..24 {
            for j in 1..12 {
                let phi = (i as f32 + 0.25) / 24.0 * std::f32::consts::TAU;
                let theta = j as f32 / 12.0 * std::f32::consts::PI;
                let dir = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                let back = sky_env_direction(sky_env_uv(dir));
                assert!((back - dir).length() < 1e-4, "{dir:?} {back:?}");
            }
        }
        assert_eq!(sky_env_uv(Vec3::Y), [0.5, 0.0]);
        assert!((sky_env_uv(Vec3::NEG_Y)[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn atmosphere_sky_view_mapping_round_trips() {
        let model = AtmosphereModel::new(&AtmosphereSettings::earth());
        let sun = Vec3::new(0.3, 0.6, 0.74).normalize();
        for &uv in &[[0.5, 0.1], [0.2, 0.4], [0.8, 0.7], [0.5, 0.9]] {
            let dir = sky_view_direction(&model, 2.0, uv, sun);
            let back = sky_view_uv(&model, 2.0, dir, sun);
            assert!(
                (back[0] - uv[0]).abs() < 1e-3 && (back[1] - uv[1]).abs() < 1e-3,
                "{uv:?} {back:?}"
            );
        }
    }
}

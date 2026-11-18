//! Volumetric clouds: a per-pixel march for the view and an amortised panorama for
//! the sky environment map (see `gfx::clouds`, `gfx::cloud_view`, `gfx::cloud_shadow`).
//!
//! The view clouds are ray-marched per pixel at quarter resolution and
//! reprojected between frames. They are composited over the sky, before fog and
//! transparent geometry, so they respect scene depth. The cloud layer also
//! shadows the sun on the ground, through a top-down shadow map.
//!
//! The environment panorama is a fixed 1024 x 512 map of the cloud layer in world
//! directions, refreshed a fraction of its texels per frame. It feeds only the
//! sky environment map, which reflections and ambient light read.
//!
//! Panorama mapping: Lambert's cylindrical equal-area projection of the upper
//! hemisphere. `u = azimuth / 2pi + 1/2` (azimuth `atan2(z, x)`), and
//! `v = 1 - y`, so `v = 0` is the zenith and `v = 1` the horizon. Every texel is
//! an upper-hemisphere direction, and equal `(u, v)` areas are equal solid
//! angles. The GLSL-free maths lives in `gfx/cloud_common.wgsl`; the functions
//! here mirror it.

#[cfg(test)]
use glam::Vec3;

/// Cloud layer and its lighting. Build from one of the presets and assign the
/// fields you change; the struct is `#[non_exhaustive]`.
///
/// Clouds take their sun and sky colours from [`crate::render::AtmosphereSettings`].
/// When `RenderSettings::clouds` is on and `RenderSettings::atmosphere` is `None`,
/// the sky is drawn with `AtmosphereSettings::default()` but with
/// `drive_lighting` off, so the forward lighting is not changed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CloudSettings {
    /// Fraction of the sky covered, `0..=1`.
    pub coverage: f32,
    /// Extinction scale of the cloud material, `0..=10`.
    pub density: f32,
    /// Metres above sea level of the layer bottom.
    pub base_altitude: f32,
    /// Metres between the layer bottom and top.
    pub thickness: f32,
    /// Metres per base-noise period.
    pub scale: f32,
    /// Erosion of the cloud edges by high-frequency noise, `0..=1`.
    pub detail: f32,
    /// Drift of the cloud shapes in world x and z, metres per second.
    pub wind: [f32; 2],
    /// Change of the shapes in place (the noise's vertical scroll), per second.
    pub evolution: f32,
    /// Forward-scatter highlight strength (the silver lining), `0..=5`.
    pub silver_lining: f32,
    /// Sky-light contribution scale, `0..=5`.
    pub ambient: f32,
    /// March steps of the view clouds (see [`CloudQuality`]).
    pub quality: CloudQuality,
    /// How much the cloud layer darkens the sun on the ground, fog and the
    /// clouds' own shading, `0..=1`. `0` turns the cloud shadow map off, so the
    /// pass is not recorded.
    pub shadow_strength: f32,
}

/// Step counts of the view clouds. Each ray takes about one sample per 60 m of
/// cloud it crosses, between 8 and the view steps of the level. The panorama
/// used for reflections does not change with the level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CloudQuality {
    /// 32 view steps and 3 light steps per sample.
    Low,
    /// 48 view steps and 4 light steps per sample (the default).
    #[default]
    Medium,
    /// 64 view steps and 4 light steps per sample.
    High,
}

/// View steps per ray and light steps per sample of a quality level.
pub(crate) fn quality_params(quality: CloudQuality) -> (u32, u32) {
    match quality {
        CloudQuality::Low => (32, 3),
        CloudQuality::Medium => (48, 4),
        CloudQuality::High => (64, 4),
    }
}

/// Size of the sky environment panorama, texels.
pub(crate) const PANORAMA_SIZE: (u32, u32) = (1024, 512);
/// March steps per panorama ray.
pub(crate) const PANORAMA_STEPS: u32 = 48;
/// Refresh stride of the panorama: each frame refreshes one texel in each `N x N` block.
pub(crate) const PANORAMA_STRIDE: u32 = 8;

/// Largest distance the view rays march, metres. Clouds fade out toward it.
pub(crate) const MAX_MARCH_DISTANCE: f32 = 60_000.0;

impl Default for CloudSettings {
    fn default() -> Self {
        Self::fair_weather()
    }
}

impl CloudSettings {
    /// Scattered cumulus: the default.
    pub fn fair_weather() -> Self {
        Self {
            coverage: 0.45,
            density: 1.0,
            base_altitude: 1500.0,
            thickness: 2500.0,
            scale: 9000.0,
            detail: 0.5,
            wind: [12.0, 4.0],
            evolution: 0.02,
            silver_lining: 0.6,
            ambient: 1.0,
            quality: CloudQuality::Medium,
            shadow_strength: 0.8,
        }
    }

    /// A continuous grey-white deck with flat tops.
    pub fn overcast() -> Self {
        Self {
            coverage: 0.9,
            density: 0.8,
            base_altitude: 800.0,
            thickness: 1500.0,
            detail: 0.3,
            silver_lining: 0.2,
            ..Self::fair_weather()
        }
    }

    /// Thin, fibrous streaks with little cover.
    pub fn wispy() -> Self {
        Self {
            coverage: 0.25,
            density: 0.6,
            base_altitude: 5000.0,
            thickness: 1500.0,
            detail: 0.8,
            wind: [18.0, 6.0],
            evolution: 0.03,
            ..Self::fair_weather()
        }
    }

    /// Dense, towering cover.
    pub fn stormy() -> Self {
        Self {
            coverage: 0.95,
            density: 2.0,
            base_altitude: 600.0,
            thickness: 5000.0,
            detail: 0.7,
            silver_lining: 0.3,
            ambient: 0.6,
            ..Self::fair_weather()
        }
    }

    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        let d = Self::fair_weather();
        Self {
            coverage: finite(self.coverage, d.coverage).clamp(0.0, 1.0),
            density: finite(self.density, d.density).clamp(0.0, 10.0),
            base_altitude: finite(self.base_altitude, d.base_altitude).clamp(0.0, 20_000.0),
            thickness: finite(self.thickness, d.thickness).clamp(100.0, 20_000.0),
            scale: finite(self.scale, d.scale).clamp(500.0, 200_000.0),
            detail: finite(self.detail, d.detail).clamp(0.0, 1.0),
            wind: self.wind.map(|c| finite(c, 0.0).clamp(-200.0, 200.0)),
            evolution: finite(self.evolution, d.evolution).clamp(0.0, 1.0),
            silver_lining: finite(self.silver_lining, d.silver_lining).clamp(0.0, 5.0),
            ambient: finite(self.ambient, d.ambient).clamp(0.0, 5.0),
            quality: self.quality,
            shadow_strength: finite(self.shadow_strength, d.shadow_strength).clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
/// Unit direction to panorama uv in `[0, 1]^2`. Directions below the horizon
/// are not representable: they fold onto the horizon row.
pub(crate) fn panorama_uv(dir: Vec3) -> [f32; 2] {
    let d = dir.normalize_or_zero();
    let phi = d.z.atan2(d.x);
    let u = phi / std::f32::consts::TAU + 0.5;
    let v = 1.0 - d.y.clamp(0.0, 1.0);
    [u, v]
}

#[cfg(test)]
/// Panorama uv to unit direction (inverse of [`panorama_uv`]).
pub(crate) fn panorama_dir(uv: [f32; 2]) -> Vec3 {
    let phi = (uv[0] - 0.5) * std::f32::consts::TAU;
    let y = 1.0 - uv[1];
    let s = (1.0 - y * y).max(0.0).sqrt();
    Vec3::new(s * phi.cos(), y, s * phi.sin())
}

/// Texel offset `(x, y)` refreshed in frame `frame` with refresh stride `n`
/// (`n` a power of two). Over `n * n` consecutive frames the offsets form a
/// permutation of the `n x n` block (a Bayer-like ordering by bit interleave).
pub(crate) fn refresh_offset(frame: u64, n: u32) -> (u32, u32) {
    let n = n.max(1);
    let bits = n.trailing_zeros();
    let index = (frame % (u64::from(n) * u64::from(n))) as u32;
    let (mut x, mut y) = (0u32, 0u32);
    for b in 0..bits {
        x |= ((index >> (2 * b)) & 1) << b;
        y |= ((index >> (2 * b + 1)) & 1) << b;
    }
    (x, y)
}

#[cfg(test)]
/// CPU port of `cloud_remap` in `cloud_common.wgsl`: `x` from `[a, b]` to `[c, d]`, clamped.
pub(crate) fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    if (b - a).abs() < 1e-6 {
        return c;
    }
    (c + (x - a) * (d - c) / (b - a)).clamp(c.min(d), c.max(d))
}

#[cfg(test)]
/// CPU port of `cloud_smoothstep`.
pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
/// CPU port of `cloud_dome`: the dome profile at height fraction `h` for a column
/// whose top is `top`.
pub(crate) fn dome(h: f32, top: f32) -> f32 {
    smoothstep(0.0, 0.1, h) * (1.0 - smoothstep(top * 0.5, top, h))
}

/// CPU port of `cloud_coverage_shape`: the base shape with a soft threshold filling
/// `coverage` of the noise range.
#[cfg(test)]
pub(crate) fn coverage_shape(shape: f32, coverage: f32) -> f32 {
    if coverage <= 0.0 {
        return 0.0;
    }
    let thr = (1.0 - coverage) * 0.8;
    smoothstep(thr, (thr + 0.25).min(1.0), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clouds_remap_maps_and_clamps_like_the_shader_port() {
        assert_eq!(remap(0.5, 0.0, 1.0, 10.0, 20.0), 15.0);
        // Outside the input range the result is clamped to the output range.
        assert_eq!(remap(-1.0, 0.0, 1.0, 10.0, 20.0), 10.0);
        assert_eq!(remap(3.0, 0.0, 1.0, 10.0, 20.0), 20.0);
        // Inverted output ranges clamp to the lower and upper bound alike.
        assert_eq!(remap(0.25, 0.0, 1.0, 1.0, 0.0), 0.75);
        // A degenerate input range returns the output start.
        assert_eq!(remap(0.3, 0.5, 0.5, 4.0, 9.0), 4.0);
    }

    #[test]
    fn clouds_settings_sanitise_and_presets() {
        assert_eq!(CloudSettings::default(), CloudSettings::fair_weather());
        assert_eq!(CloudQuality::default(), CloudQuality::Medium);
        let wild = CloudSettings {
            coverage: 3.0,
            density: f32::NAN,
            thickness: -5.0,
            wind: [1.0e6, f32::INFINITY],
            shadow_strength: 7.0,
            ..CloudSettings::fair_weather()
        }
        .sanitised();
        assert_eq!(wild.coverage, 1.0);
        assert_eq!(wild.density, 1.0);
        assert_eq!(wild.thickness, 100.0);
        assert_eq!(wild.wind, [200.0, 0.0]);
        assert_eq!(wild.shadow_strength, 1.0);
        let nan_shadow = CloudSettings {
            shadow_strength: f32::NAN,
            ..CloudSettings::fair_weather()
        }
        .sanitised();
        assert_eq!(nan_shadow.shadow_strength, 0.8);
        assert_eq!(CloudSettings::fair_weather().shadow_strength, 0.8);
        assert!(CloudSettings::overcast().coverage > CloudSettings::fair_weather().coverage);
        assert!(CloudSettings::wispy().coverage < CloudSettings::fair_weather().coverage);
        assert!(CloudSettings::stormy().density > CloudSettings::fair_weather().density);
        let (low, med, high) = (
            quality_params(CloudQuality::Low),
            quality_params(CloudQuality::Medium),
            quality_params(CloudQuality::High),
        );
        assert!(low.0 < med.0 && med.0 < high.0);
        assert_eq!(med, (48, 4));
    }

    #[test]
    fn clouds_panorama_round_trips_and_stays_in_unit_square() {
        let mut worst = 0.0f32;
        for i in 0..64 {
            for j in 0..33 {
                let phi = i as f32 / 64.0 * std::f32::consts::TAU - std::f32::consts::PI;
                // Elevation from the horizon (0) to the zenith (pi / 2).
                let el = j as f32 / 32.0 * std::f32::consts::FRAC_PI_2;
                let dir = Vec3::new(el.cos() * phi.cos(), el.sin(), el.cos() * phi.sin());
                let uv = panorama_uv(dir);
                assert!(
                    (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                    "uv {uv:?} for {dir:?}"
                );
                let back = panorama_dir(uv);
                worst = worst.max((back - dir).abs().max_element());
            }
        }
        assert!(worst < 1e-4, "round trip error {worst}");
        // The zenith maps to the top row, the horizon to the bottom row.
        assert!(panorama_uv(Vec3::Y)[1].abs() < 1e-6);
        assert!((panorama_uv(Vec3::X)[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clouds_refresh_sequence_visits_every_offset_once_per_cycle() {
        for n in [2u32, 4] {
            for cycle in 0..3u64 {
                let mut seen = std::collections::HashSet::new();
                for f in 0..u64::from(n * n) {
                    let (x, y) = refresh_offset(cycle * u64::from(n * n) + f, n);
                    assert!(x < n && y < n);
                    assert!(seen.insert((x, y)), "offset repeated in cycle");
                }
                assert_eq!(seen.len(), (n * n) as usize);
            }
        }
    }

    #[test]
    fn clouds_cpu_ports_stay_in_unit_range() {
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            for j in 0..=20 {
                let c = j as f32 / 20.0;
                let p = dome(x, 0.3 + 0.7 * c);
                let s = coverage_shape(x, c);
                assert!((0.0..=1.0).contains(&p), "dome {p}");
                assert!((0.0..=1.0).contains(&s), "shape {s}");
            }
        }
        assert_eq!(coverage_shape(0.9, 0.0), 0.0);
        // A dome is rounded at the base, zero above its own top, and tapers below it.
        assert_eq!(dome(0.0, 0.8), 0.0);
        assert!(dome(0.05, 0.8) < dome(0.5, 0.8));
        assert_eq!(dome(0.9, 0.8), 0.0);
        assert!(dome(0.35, 1.0) > dome(0.7, 0.8));
    }
}

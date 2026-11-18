//! Volumetric fog settings (see `gfx::fog` for the froxel volume and its passes).

/// Froxel grid size of the fog volume: screen columns and rows, and depth
/// slices. Higher resolutions cost more in the inject and integrate passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FogResolution {
    /// 96 x 54 x 48.
    Low,
    /// 160 x 90 x 64 (the default).
    #[default]
    Medium,
    /// 240 x 135 x 96.
    High,
}

impl FogResolution {
    /// Grid size as `(width, height, slices)`.
    pub fn size(self) -> (u32, u32, u32) {
        match self {
            Self::Low => (96, 54, 48),
            Self::Medium => (160, 90, 64),
            Self::High => (240, 135, 96),
        }
    }
}

/// Participating fog with shadowed light shafts. Build from one of the presets
/// and assign the fields you change; the struct is `#[non_exhaustive]`.
///
/// Depth slices run from the camera's near plane to [`Self::distance`] with
/// the exponential mapping `depth(i) = near * (distance / near)^(i / slices)`,
/// so slices are thin near the camera. Beyond `distance`, an analytic
/// height-fog term continues to the surface or, for sky pixels, to a short
/// path length.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FogSettings {
    /// Extinction at `base_height`, per metre.
    pub density: f32,
    /// Exponential falloff above `base_height`, per metre (0 = uniform).
    pub height_falloff: f32,
    /// World y at which `density` applies.
    pub base_height: f32,
    /// Scattering albedo (rgb) in `[0, 1]`.
    pub albedo: [f32; 3],
    /// Henyey-Greenstein anisotropy `g` in `[-0.95, 0.95]`.
    pub anisotropy: f32,
    /// Multiplier on the sky ambient in-scattering.
    pub ambient: f32,
    /// Multiplier on sun in-scattering (the strength of light shafts).
    pub sun_scatter: f32,
    /// Metres covered by the volume.
    pub distance: f32,
    /// Density variation from animated noise, in `[0, 1]`.
    pub noise: f32,
    /// Metres per noise period.
    pub noise_scale: f32,
    /// Metres per second the noise drifts.
    pub wind: [f32; 3],
    /// Froxel grid size.
    pub resolution: FogResolution,
    /// Reproject and blend the previous volume (and jitter the slices).
    pub temporal: bool,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self::medium()
    }
}

impl FogSettings {
    /// A faint haze that thins the far distance.
    pub fn light_haze() -> Self {
        Self {
            density: 0.006,
            height_falloff: 0.08,
            albedo: [0.95, 0.95, 0.95],
            ..Self::medium()
        }
    }

    /// The default medium fog.
    pub fn medium() -> Self {
        Self {
            density: 0.02,
            height_falloff: 0.12,
            base_height: 0.0,
            albedo: [0.9, 0.9, 0.9],
            anisotropy: 0.6,
            ambient: 1.0,
            sun_scatter: 1.0,
            distance: 160.0,
            noise: 0.35,
            noise_scale: 18.0,
            wind: [1.5, 0.0, 0.6],
            resolution: FogResolution::Medium,
            temporal: true,
        }
    }

    /// Thick fog that hides the far scene.
    pub fn dense() -> Self {
        Self {
            density: 0.06,
            height_falloff: 0.05,
            ..Self::medium()
        }
    }

    /// Low mist that hugs the ground: strong height falloff and a low base.
    pub fn ground_mist() -> Self {
        Self {
            density: 0.05,
            height_falloff: 0.6,
            base_height: 0.0,
            ..Self::medium()
        }
    }

    /// Copy with every field clamped to its documented range.
    pub(crate) fn sanitised(&self) -> Self {
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        let albedo = self.albedo.map(|c| finite(c, 0.9).clamp(0.0, 1.0));
        let wind = self.wind.map(|c| finite(c, 0.0).clamp(-50.0, 50.0));
        Self {
            density: finite(self.density, 0.02).clamp(0.0, 2.0),
            height_falloff: finite(self.height_falloff, 0.12).clamp(0.0, 5.0),
            base_height: finite(self.base_height, 0.0),
            albedo,
            anisotropy: finite(self.anisotropy, 0.6).clamp(-0.95, 0.95),
            ambient: finite(self.ambient, 1.0).clamp(0.0, 10.0),
            sun_scatter: finite(self.sun_scatter, 1.0).clamp(0.0, 50.0),
            distance: finite(self.distance, 160.0).clamp(10.0, 1000.0),
            noise: finite(self.noise, 0.35).clamp(0.0, 1.0),
            noise_scale: finite(self.noise_scale, 18.0).clamp(0.5, 1000.0),
            wind,
            resolution: self.resolution,
            temporal: self.temporal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fog_settings_sanitise_and_presets() {
        assert_eq!(FogSettings::default(), FogSettings::medium());
        assert_eq!(FogResolution::default(), FogResolution::Medium);
        let wild = FogSettings {
            density: -1.0,
            anisotropy: 3.0,
            distance: 1.0e9,
            noise: f32::NAN,
            albedo: [2.0, 0.5, -3.0],
            ..FogSettings::medium()
        }
        .sanitised();
        assert_eq!(wild.density, 0.0);
        assert_eq!(wild.anisotropy, 0.95);
        assert_eq!(wild.distance, 1000.0);
        assert_eq!(wild.noise, 0.35);
        assert_eq!(wild.albedo, [1.0, 0.5, 0.0]);
        assert!(FogSettings::ground_mist().height_falloff > FogSettings::medium().height_falloff);
        assert!(FogSettings::dense().density > FogSettings::light_haze().density);
        let sizes = [
            FogResolution::Low,
            FogResolution::Medium,
            FogResolution::High,
        ]
        .map(|r| r.size().0 * r.size().2);
        assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2]);
    }
}

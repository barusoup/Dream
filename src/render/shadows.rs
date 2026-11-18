//! Cascaded shadow-map fitting. Pure functions: no GPU state.
//!
//! Each cascade covers one slice of the camera frustum. A slice is enclosed in
//! a bounding sphere, the sphere in an orthographic light box, and the box's
//! origin is snapped to whole shadow texels so the map does not shimmer while
//! the camera moves. The projection uses the engine's convention: right-handed
//! view space, depth 0..1, and y up in NDC (the renderer's negative viewport
//! height makes that match the image rows).

use super::ShadowSettings;
use glam::{Mat4, Vec3};

/// Radius quantisation: radii round up to a multiple of `1 / RADIUS_STEPS`
/// metres so small camera rotations do not change the map's extent.
const RADIUS_STEPS: f32 = 32.0;

/// Fitted light matrix of one cascade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CascadeFit {
    /// World to light clip space (depth 0..1).
    pub view_projection: Mat4,
    /// World-space size of one shadow texel at the cascade's plane.
    pub texel_size: f32,
    /// Depth extent of the light box in metres (maps depth 0..1).
    pub depth_range: f32,
    /// Bounding-sphere radius of the cascade's frustum slice, in metres.
    pub radius: f32,
}

/// `count + 1` view-space distances from `near` to `far`, practical split
/// scheme: `lambda * logarithmic + (1 - lambda) * uniform`.
pub(crate) fn split_distances(near: f32, far: f32, count: u32, lambda: f32) -> Vec<f32> {
    let near = near.max(1e-3);
    let far = far.max(near + 1e-2);
    let n = count.max(1) as f32;
    let lambda = lambda.clamp(0.0, 1.0);
    (0..=count.max(1))
        .map(|i| {
            let p = i as f32 / n;
            let log = near * (far / near).powf(p);
            let uni = near + (far - near) * p;
            if i == 0 {
                near
            } else if i as f32 >= n {
                far
            } else {
                lambda * log + (1.0 - lambda) * uni
            }
        })
        .collect()
}

/// Fits one light matrix per cascade of `settings` for a camera whose
/// unjittered `view` and `view_projection` are given. `near` and `far` are the
/// camera's clip planes, with `far` already limited to the shadow distance by
/// the caller. `sun` points TO the sun.
pub(crate) fn fit_cascades(
    view: Mat4,
    view_projection: Mat4,
    near: f32,
    far: f32,
    sun: Vec3,
    settings: &ShadowSettings,
) -> Vec<CascadeFit> {
    let count = settings.cascades.clamp(1, 4);
    let frustum = FrustumScale::new(view, view_projection);
    let splits = split_distances(near, far, count, settings.split_lambda);
    let sun = sun.try_normalize().unwrap_or(Vec3::Y);
    (0..count as usize)
        .map(|i| {
            let corners = frustum.slice_corners(splits[i], splits[i + 1]);
            fit_slice(&corners, sun, settings.resolution, settings.caster_distance)
        })
        .collect()
}

/// The camera's view inverse and projection scale, enough to unproject any
/// slice. The projection's x/y scale is read from `view_projection * view^-1`
/// rather than inverting the full view-projection: at far-plane distances the
/// full inverse loses enough f32 precision to change slice radii between
/// rotations.
pub(crate) struct FrustumScale {
    inv_view: Mat4,
    scale_x: f32,
    scale_y: f32,
}

impl FrustumScale {
    pub(crate) fn new(view: Mat4, view_projection: Mat4) -> Self {
        let inv_view = view.inverse();
        let projection = view_projection * inv_view;
        Self {
            inv_view,
            scale_x: projection.x_axis.x,
            scale_y: projection.y_axis.y,
        }
    }

    /// The eight world-space corners of the slice between view depths `d0` and
    /// `d1`: four at each plane, ordered like the NDC square.
    pub(crate) fn slice_corners(&self, d0: f32, d1: f32) -> [Vec3; 8] {
        let mut out = [Vec3::ZERO; 8];
        for (k, d) in [d0, d1].into_iter().enumerate() {
            for (j, (x, y)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .into_iter()
                .enumerate()
            {
                // Perspective: ndc = scale * view_xy / depth.
                let view_point = Vec3::new(x * d / self.scale_x, y * d / self.scale_y, -d);
                out[k * 4 + j] = self.inv_view.transform_point3(view_point);
            }
        }
        out
    }
}

/// Centre (mean of the corners) and the exact radius of the enclosing sphere
/// around that centre, before quantisation.
pub(crate) fn slice_sphere(corners: &[Vec3; 8]) -> (Vec3, f32) {
    let centre = corners.iter().copied().sum::<Vec3>() / 8.0;
    let radius = corners
        .iter()
        .map(|c| c.distance(centre))
        .fold(0.0, f32::max);
    (centre, radius)
}

/// Light matrix for one slice. `resolution` is the square map's side in texels;
/// `caster_distance` extends the light's depth range toward the sun.
pub(crate) fn fit_slice(
    corners: &[Vec3; 8],
    sun: Vec3,
    resolution: u32,
    caster_distance: f32,
) -> CascadeFit {
    let (centre, raw) = slice_sphere(corners);
    let radius = (raw * RADIUS_STEPS).ceil().max(1.0) / RADIUS_STEPS;
    let caster = caster_distance.max(0.0);
    let res = resolution.max(1) as f32;
    // One texel of padding per side: snapping the origin moves the box by up to
    // half a texel, which must not expose a corner outside the map.
    let half = radius * (res + 2.0) / res;
    let texel = 2.0 * half / res;

    // The up vector only has to be stable; switch it when the sun is near the
    // zenith, where `Y` would be parallel to the view direction.
    let up = if sun.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    // Snap the centre to the texel grid of the light's own x/y axes. Those
    // axes depend only on `sun`, so the grid is fixed in world space: moving the
    // camera changes the snapped centre by whole texels only.
    let basis = Mat4::look_to_rh(Vec3::ZERO, -sun, up);
    let local = basis.transform_point3(centre);
    let snapped = Vec3::new(
        (local.x / texel).round() * texel,
        (local.y / texel).round() * texel,
        local.z,
    );
    let centre = basis.inverse().transform_point3(snapped);
    let eye = centre + sun * (radius + caster);
    let view = Mat4::look_to_rh(eye, -sun, up);

    let depth = 2.0 * radius + caster + 2.0 * texel;
    let proj = Mat4::orthographic_rh(-half, half, -half, half, 0.0, depth);
    CascadeFit {
        view_projection: proj * view,
        texel_size: texel,
        depth_range: depth,
        radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    fn camera_at(position: Vec3, yaw: f32) -> Camera {
        let mut camera = Camera::default();
        camera.position = position;
        camera.set_look(yaw, -0.3);
        camera
    }

    fn settings() -> ShadowSettings {
        let mut s = ShadowSettings::medium();
        s.distance = 120.0;
        s
    }

    fn fits_for(camera: &Camera, s: &ShadowSettings) -> Vec<CascadeFit> {
        let (near, far) = camera.clip_planes();
        fit_cascades(
            camera.view(),
            camera.view_projection(),
            near,
            far.min(s.distance),
            Vec3::new(0.4, 0.7, -0.5),
            s,
        )
    }

    #[test]
    fn shadows_splits_strictly_increase_and_end_at_distance() {
        for lambda in [0.0, 0.5, 1.0] {
            for count in 1..=4 {
                let d = split_distances(0.1, 120.0, count, lambda);
                assert_eq!(d.len(), count as usize + 1);
                assert!((d[0] - 0.1).abs() < 1e-6);
                assert!((d[count as usize] - 120.0).abs() < 1e-3);
                assert!(d.windows(2).all(|w| w[1] > w[0]), "lambda {lambda}: {d:?}");
            }
        }
    }

    #[test]
    fn shadows_corners_project_inside_clip_cube() {
        let s = settings();
        let camera = camera_at(Vec3::new(3.0, 44.0, 70.0), 0.4);
        let (near, far) = camera.clip_planes();
        let far = far.min(s.distance);
        let splits = split_distances(near, far, s.cascades, s.split_lambda);
        let fits = fits_for(&camera, &s);
        let frustum = FrustumScale::new(camera.view(), camera.view_projection());
        for (i, fit) in fits.iter().enumerate() {
            let corners = frustum.slice_corners(splits[i], splits[i + 1]);
            for corner in corners {
                let p = fit.view_projection * corner.extend(1.0);
                let p = p.truncate() / p.w;
                assert!(p.x.abs() <= 1.0 + 1e-4, "cascade {i}: x {}", p.x);
                assert!(p.y.abs() <= 1.0 + 1e-4, "cascade {i}: y {}", p.y);
                assert!(
                    (-1e-4..=1.0 + 1e-4).contains(&p.z),
                    "cascade {i}: z {}",
                    p.z
                );
            }
        }
    }

    #[test]
    fn shadows_sub_texel_camera_moves_shift_in_whole_texels() {
        let s = settings();
        let base = Vec3::new(3.0, 44.0, 70.0);
        let fixed = Vec3::new(10.0, 2.0, 40.0);
        let first = fits_for(&camera_at(base, 0.4), &s);
        for step in [0.13_f32, 0.37, 0.61, 0.9] {
            let moved = fits_for(&camera_at(base + Vec3::new(step, 0.0, step * 0.5), 0.4), &s);
            for (a, b) in first.iter().zip(&moved) {
                assert!((a.radius - b.radius).abs() < 1e-3, "radius changed");
                let texel_a = a.view_projection.project_point3(fixed);
                let texel_b = b.view_projection.project_point3(fixed);
                let res = s.resolution as f32;
                let delta = (texel_b.x - texel_a.x) * res * 0.5;
                let delta_y = (texel_b.y - texel_a.y) * res * 0.5;
                assert!(
                    (delta - delta.round()).abs() < 1e-3,
                    "x delta {delta} texels (step {step})"
                );
                assert!(
                    (delta_y - delta_y.round()).abs() < 1e-3,
                    "y delta {delta_y} texels (step {step})"
                );
            }
        }
    }

    #[test]
    fn shadows_rotation_keeps_cascade_radius() {
        let s = settings();
        let (near, far) = camera_at(Vec3::ZERO, 0.0).clip_planes();
        let far = far.min(s.distance);
        let splits = split_distances(near, far, s.cascades, s.split_lambda);
        let pos = Vec3::new(3.0, 44.0, 70.0);
        let a = camera_at(pos, 0.2);
        let b = camera_at(pos, 1.9);
        let raw = |camera: &Camera| -> Vec<f32> {
            let frustum = FrustumScale::new(camera.view(), camera.view_projection());
            (0..s.cascades as usize)
                .map(|i| {
                    let corners = frustum.slice_corners(splits[i], splits[i + 1]);
                    slice_sphere(&corners).1
                })
                .collect()
        };
        for (x, y) in raw(&a).iter().zip(raw(&b)) {
            assert!((x - y).abs() < 1e-3, "raw radius {x} vs {y}");
        }
        for (x, y) in fits_for(&a, &s).iter().zip(fits_for(&b, &s)) {
            assert!((x.radius - y.radius).abs() <= 1.0 / RADIUS_STEPS + 1e-4);
        }
    }

    #[test]
    fn shadows_sphere_is_nearer_the_light_than_ground_below_it() {
        let s = settings();
        let mut camera = Camera::default();
        camera.set_aspect(1280.0 / 720.0);
        camera.position = Vec3::new(6.0, 3.5, 9.0);
        let forward = (Vec3::new(0.0, 0.7, 0.0) - camera.position).normalize();
        camera.set_look(forward.z.atan2(forward.x), forward.y.asin());
        let sun = crate::lighting::LightingEnv::with_sun(0.6, 0.7, 3.0).sun_direction;
        let (near, far) = camera.clip_planes();
        let fits = fit_cascades(
            camera.view(),
            camera.view_projection(),
            near,
            far.min(s.distance),
            Vec3::new(sun[0], sun[1], sun[2]),
            &s,
        );
        let sphere = Vec3::new(0.0, 0.5, -2.0);
        let ground = Vec3::new(0.0, 0.0, -2.0);
        for (i, fit) in fits.iter().enumerate() {
            let a = fit.view_projection.project_point3(sphere);
            let b = fit.view_projection.project_point3(ground);
            assert!(
                a.z < b.z,
                "cascade {i}: sphere depth {} vs ground {}",
                a.z,
                b.z
            );
        }
    }

    #[test]
    fn shadows_fit_survives_zenith_sun() {
        let s = settings();
        let camera = camera_at(Vec3::new(0.0, 44.0, 0.0), 0.0);
        let (near, far) = camera.clip_planes();
        let fits = fit_cascades(
            camera.view(),
            camera.view_projection(),
            near,
            far.min(s.distance),
            Vec3::new(0.0, 1.0, 0.0),
            &s,
        );
        for fit in fits {
            assert!(
                fit.view_projection
                    .to_cols_array()
                    .iter()
                    .all(|v| v.is_finite())
            );
            assert!(fit.texel_size > 0.0 && fit.depth_range > 0.0);
        }
    }
}

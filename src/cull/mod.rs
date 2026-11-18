//! Culling primitives: a view-frustum test and a Hi-Z occlusion test.
//!
//! [`Frustum`] and [`Bounds`] are pure arithmetic and work for perspective
//! cameras and for the orthographic sun-cascade matrices alike, because both
//! are standard `0..1` depth clip volumes (`glam`'s `*_rh` constructors). The
//! occlusion side lives in [`occlusion`]: it tests boxes against the depth the
//! renderer read back from an earlier frame.
//!
//! # Usage
//!
//! Frustum culling needs only the camera's view-projection matrix:
//!
//! ```
//! use dream::cull::{Bounds, Frustum};
//! use glam::{Mat4, Vec3};
//!
//! // Stands in for `camera.view_projection()`; this matrix maps the unit
//! // cube, which is the clip volume with `0..1` depth.
//! let frustum = Frustum::from_view_projection(Mat4::IDENTITY);
//! let inside = Bounds::from_aabb(Vec3::new(-0.5, -0.5, 0.25), Vec3::new(0.5, 0.5, 0.75));
//! assert!(frustum.contains_aabb(inside.aabb_min, inside.aabb_max));
//! ```
//!
//! [`OcclusionBuffer`] is the Hi-Z grid of the previous frames. The renderer
//! builds it when [`crate::render::RenderSettings::occlusion`] is set, and the
//! scene pass reads it back through [`crate::gfx::DrawCtx::occlusion`].
//! [`crate::world::World`] applies it for you.

pub mod occlusion;

pub use occlusion::{OcclusionBuffer, OcclusionMeta};

use glam::{Mat4, Vec3};

/// Six inward-pointing planes `(normal, d)` with `dot(normal, p) + d >= 0`
/// for points inside. Normals are unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract the planes from a combined view-projection matrix whose clip
    /// volume is `-1..1` in x and y and `0..1` in z (Gribb-Hartmann, adapted
    /// to the `0..1` depth convention: the near plane is the third clip row
    /// alone, not row 3 plus row 2).
    pub fn from_view_projection(vp: Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);
        let raw = [
            r3 + r0, // left:   x >= -w
            r3 - r0, // right:  x <=  w
            r3 + r1, // bottom: y >= -w
            r3 - r1, // top:    y <=  w
            r2,      // near:   z >=  0
            r3 - r2, // far:    z <=  w
        ];
        let mut planes = [[0.0f32; 4]; 6];
        for (out, p) in planes.iter_mut().zip(raw) {
            let len = p.truncate().length();
            // A degenerate plane (zero normal) constrains nothing: keep it
            // as a trivially-satisfied plane rather than dividing by zero.
            *out = if len > 1e-12 && len.is_finite() {
                (p / len).to_array()
            } else {
                [0.0, 0.0, 0.0, 1.0]
            };
        }
        Self { planes }
    }

    /// Conservative sphere test: `true` when the sphere may touch the volume.
    pub fn contains_sphere(&self, centre: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p[0] * centre.x + p[1] * centre.y + p[2] * centre.z + p[3] >= -radius)
    }

    /// Conservative box test: `true` unless the box lies wholly outside one
    /// plane. Tests only the box corner furthest along each plane normal.
    pub fn contains_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|p| {
            let x = if p[0] >= 0.0 { max.x } else { min.x };
            let y = if p[1] >= 0.0 { max.y } else { min.y };
            let z = if p[2] >= 0.0 { max.z } else { min.z };
            p[0] * x + p[1] * y + p[2] * z + p[3] >= 0.0
        })
    }

    /// Batch sphere test over `[x, y, z, radius]` rows. Appends the index of
    /// every sphere that may be visible to `visible`; the caller clears it.
    pub fn cull_spheres(&self, spheres: &[[f32; 4]], visible: &mut Vec<u32>) {
        let planes = &self.planes;
        for (index, s) in spheres.iter().enumerate() {
            let inside = planes
                .iter()
                .all(|p| p[0] * s[0] + p[1] * s[1] + p[2] * s[2] + p[3] >= -s[3]);
            if inside {
                visible.push(index as u32);
            }
        }
    }
}

/// World-space bounds of one object: a sphere for coarse tests and the
/// axis-aligned box for tight ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub centre: Vec3,
    pub radius: f32,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

impl Bounds {
    /// Bounds of a box given directly in world space.
    pub fn from_aabb(min: Vec3, max: Vec3) -> Self {
        let centre = (min + max) * 0.5;
        Self {
            centre,
            radius: (max - centre).length(),
            aabb_min: min,
            aabb_max: max,
        }
    }

    /// World-space bounds of the local box `[local_min, local_max]` under
    /// `transform`. The sphere is centred on the transformed box's centre and
    /// reaches its farthest corner, so it encloses the whole box.
    pub fn from_local(local_min: Vec3, local_max: Vec3, transform: Mat4) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for corner in 0..8u32 {
            let p = Vec3::new(
                if corner & 1 != 0 {
                    local_max.x
                } else {
                    local_min.x
                },
                if corner & 2 != 0 {
                    local_max.y
                } else {
                    local_min.y
                },
                if corner & 4 != 0 {
                    local_max.z
                } else {
                    local_min.z
                },
            );
            let w = transform.transform_point3(p);
            min = min.min(w);
            max = max.max(w);
        }
        let mut bounds = Self::from_aabb(min, max);
        // The box centre is not necessarily the centre of the enclosing sphere
        // for every transform; measure the corners from the centre to be safe.
        let mut radius = 0.0f32;
        for corner in 0..8u32 {
            let c = Vec3::new(
                if corner & 1 != 0 { max.x } else { min.x },
                if corner & 2 != 0 { max.y } else { min.y },
                if corner & 4 != 0 { max.z } else { min.z },
            );
            radius = radius.max((c - bounds.centre).length());
        }
        bounds.radius = radius;
        bounds
    }

    /// The box grown by `amount` on every side (the sphere grows with it).
    pub fn inflated(&self, amount: f32) -> Self {
        let d = Vec3::splat(amount.max(0.0));
        Self {
            centre: self.centre,
            radius: self.radius + amount.max(0.0),
            aabb_min: self.aabb_min - d,
            aabb_max: self.aabb_max + d,
        }
    }
}

#[cfg(test)]
mod tests;

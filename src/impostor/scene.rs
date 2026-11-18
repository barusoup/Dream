//! Per-frame LoD bookkeeping: which objects draw as mesh, fade or card.
//!
//! [`LodBatch`] tracks every object's hysteresis state ([`LodChoice`])
//! across frames. Each frame the driver calls [`LodBatch::select`] per object;
//! the result splits into three lists — full mesh, mesh+card crossfade,
//! card-only — which the lab submits through `ModelRenderer::draw_pbr*`
//! and `ImpostorRenderer::draw` respectively.
//!
//! Guards beyond the pure policy:
//!
//! - **camera inside the box** => always [`LodChoice::Mesh`]. A card renders
//!   a bounding shell; inside it the "impostor" would cover the whole screen
//!   with garbage.
//! - **no asset** (bake fell back or wasn't run) => always mesh.
//! - **non-finite distance/coverage** => keep the previous choice rather
//!   than flapping.

use super::ImpostorBounds;
use super::lod::{LodChoice, LodPolicy, LodStats};
use glam::{Mat4, Vec3};

/// One tracked object: its asset bounds (for coverage + inside tests) and
/// last choice.
#[derive(Clone, Debug)]
pub struct LodEntry {
    /// Object-space bounds of the asset (`ImpostorAsset::bounds`); the
    /// transformed box is what "camera inside" tests.
    pub bounds: ImpostorBounds,
    /// Whether an asset exists at all — `static_capture_supported` failures
    /// and missing bakes stay meshes forever.
    pub has_asset: bool,
    /// Hysteresis state across frames.
    pub choice: LodChoice,
}

impl LodEntry {
    pub fn new(bounds: ImpostorBounds, has_asset: bool) -> Self {
        Self {
            bounds,
            has_asset,
            choice: LodChoice::Mesh,
        }
    }
}

/// Policy + rolling stats shared by all objects.
pub struct LodBatch {
    pub policy: LodPolicy,
    pub stats: LodStats,
}

impl Default for LodBatch {
    fn default() -> Self {
        Self {
            policy: LodPolicy::default(),
            stats: LodStats::default(),
        }
    }
}

impl LodBatch {
    pub fn new(policy: LodPolicy) -> Self {
        Self {
            policy,
            stats: LodStats::default(),
        }
    }

    /// The choice for one object this frame, for the engine's default
    /// 70 degree field of view. See [`Self::select_fov`].
    pub fn select(&mut self, entry: &mut LodEntry, transform: Mat4, camera_pos: Vec3) -> LodChoice {
        self.select_fov(entry, transform, camera_pos, LodPolicy::HALF_FOV_TAN)
    }

    /// The choice for one object this frame: coverage from the transformed
    /// bounding sphere, measured with a camera whose vertical half field of
    /// view has tangent `half_fov_tan`, with the `max_distance` limit applied
    /// as [`LodPolicy::select`] applies it. The camera-inside,
    /// unsupported-transform and no-asset guards come first.
    pub fn select_fov(
        &mut self,
        entry: &mut LodEntry,
        transform: Mat4,
        camera_pos: Vec3,
        half_fov_tan: f32,
    ) -> LodChoice {
        if !entry.has_asset {
            entry.choice = LodChoice::Mesh;
            return entry.choice;
        }
        // The card shader supports affine proper rotation + positive uniform
        // scale only — anything else (non-uniform scale, shear, reflection)
        // stays on the mesh so the opt-in path can't pick a card it can't
        // draw. Manual `draw` calls are unaffected.
        if super::render::ImpostorInstance::try_new(transform, transform, 1.0).is_err() {
            entry.choice = LodChoice::Mesh;
            return entry.choice;
        }
        let world_box = entry.bounds.transformed(transform);
        if world_box.contains(camera_pos) {
            if entry.choice != LodChoice::Mesh {
                self.stats.record(entry.choice, LodChoice::Mesh);
            }
            entry.choice = LodChoice::Mesh;
            return entry.choice;
        }
        let center = world_box.center();
        let distance = (camera_pos - center).length();
        let coverage = LodPolicy::screen_coverage_fov(world_box.radius(), distance, half_fov_tan);
        let coverage = coverage.min(self.policy.sanitised().max_distance_coverage(distance));
        if !coverage.is_finite() || coverage <= 0.0 {
            return entry.choice;
        }
        let next = self.policy.select_coverage(coverage, entry.choice);
        self.stats.record(entry.choice, next);
        entry.choice = next;
        entry.choice
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn bounds() -> ImpostorBounds {
        ImpostorBounds {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        }
    }

    #[test]
    fn camera_inside_forces_mesh() {
        let mut batch = LodBatch::default();
        let mut entry = LodEntry::new(bounds(), true);
        // Far enough to be a card normally; inside the box it must not be.
        entry.choice = LodChoice::Impostor;
        let choice = batch.select(&mut entry, Mat4::IDENTITY, Vec3::ZERO);
        assert_eq!(choice, LodChoice::Mesh);
    }

    #[test]
    fn missing_asset_stays_mesh() {
        let mut batch = LodBatch::default();
        let mut entry = LodEntry::new(bounds(), false);
        let choice = batch.select(
            &mut entry,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 500.0)),
            Vec3::ZERO,
        );
        assert_eq!(choice, LodChoice::Mesh);
    }

    #[test]
    fn distant_object_becomes_card() {
        let mut batch = LodBatch::default();
        let mut entry = LodEntry::new(bounds(), true);
        let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, 500.0));
        let choice = batch.select(&mut entry, transform, Vec3::ZERO);
        assert!(choice.draws_impostor(), "expected impostor, got {choice:?}");
    }

    #[test]
    fn unsupported_transform_stays_mesh() {
        use glam::Mat4;
        let mut batch = LodBatch::new(LodPolicy::default());
        let mut entry = LodEntry {
            bounds: bounds(),
            has_asset: true,
            choice: LodChoice::Mesh,
        };
        let cam = Vec3::new(0.0, 0.0, -1000.0); // far: coverage -> impostor
        // Valid far transform still reaches the impostor.
        let c = batch.select(&mut entry, Mat4::IDENTITY, cam);
        assert_eq!(c, LodChoice::Impostor);
        // Non-uniform scale: card shader can't draw it — stays mesh.
        let mut entry = LodEntry {
            bounds: bounds(),
            has_asset: true,
            choice: LodChoice::Mesh,
        };
        let c = batch.select(&mut entry, Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0)), cam);
        assert_eq!(c, LodChoice::Mesh);
        // Reflection likewise.
        let c = batch.select(&mut entry, Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)), cam);
        assert_eq!(c, LodChoice::Mesh);
    }
}

#[cfg(test)]
mod fov_tests {
    use super::*;

    fn unit_bounds() -> ImpostorBounds {
        ImpostorBounds {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        }
    }

    /// Distance at which a unit box's radius (sqrt 3) projects to `coverage`
    /// with the given half field-of-view tangent.
    fn distance_for(coverage: f32, half_fov_tan: f32) -> f32 {
        3f32.sqrt() / (coverage * half_fov_tan)
    }

    #[test]
    fn lod_batch_max_distance_limit_is_honoured() {
        // A large box far away measures a mesh-sized coverage; past
        // max_distance the policy's cap must send it to the impostor, as
        // LodPolicy::select does.
        let policy = LodPolicy {
            max_distance: 100.0,
            ..LodPolicy::default()
        };
        let far = 500.0;
        let big = ImpostorBounds {
            min: Vec3::splat(-100.0),
            max: Vec3::splat(100.0),
        };
        let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, far));
        let mut batch = LodBatch::new(policy);
        let mut entry = LodEntry::new(big, true);
        let choice = batch.select(&mut entry, transform, Vec3::ZERO);
        assert_eq!(choice, LodChoice::Impostor, "past max_distance");

        // Without the limit the same box is a mesh: the limit is what changed it.
        let mut unlimited = LodBatch::new(LodPolicy {
            max_distance: 1e9,
            ..policy
        });
        let mut entry = LodEntry::new(big, true);
        assert_eq!(
            unlimited.select(&mut entry, transform, Vec3::ZERO),
            LodChoice::Mesh
        );
    }

    #[test]
    fn lod_batch_default_select_is_select_fov_at_default_tan() {
        let mut batch = LodBatch::default();
        for d in [3.0f32, 20.0, 80.0, 250.0, 1000.0] {
            let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, d));
            let mut a = LodEntry::new(unit_bounds(), true);
            let mut b = LodEntry::new(unit_bounds(), true);
            let x = batch.select(&mut a, transform, Vec3::ZERO);
            let y = batch.select_fov(&mut b, transform, Vec3::ZERO, LodPolicy::HALF_FOV_TAN);
            assert_eq!(x, y, "distance {d}");
        }
    }

    #[test]
    fn lod_batch_fov_changes_the_choice_at_the_same_distance() {
        // At the default field of view this distance projects below
        // impostor_below; with a narrow lens it projects above mesh_above.
        let d = distance_for(0.01, LodPolicy::HALF_FOV_TAN);
        let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, d));
        let mut batch = LodBatch::default();
        let mut entry = LodEntry::new(unit_bounds(), true);
        assert_eq!(
            batch.select(&mut entry, transform, Vec3::ZERO),
            LodChoice::Impostor
        );
        let narrow = 0.2;
        assert!(distance_for(0.035, narrow) > 0.0);
        let mut entry = LodEntry::new(unit_bounds(), true);
        assert_eq!(
            batch.select_fov(&mut entry, transform, Vec3::ZERO, narrow),
            LodChoice::Mesh
        );
    }
}

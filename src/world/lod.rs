//! Describing one object's levels of detail for a [`super::World`].
//!
//! A [`LodModel`] lists mesh levels in decreasing detail, each with the screen
//! coverage below which it takes over, plus an optional impostor for the
//! farthest range. Coverage is the fraction of the viewport height a bounding
//! sphere spans (see [`crate::impostor::LodPolicy::screen_coverage_fov`]).

use crate::impostor::{BakeSettings, ImpostorAsset, LodChoice, LodOwner, LodPolicy};
use crate::model::Model;
use std::path::PathBuf;
use std::sync::Arc;

/// One mesh level: the model to draw and the coverage below which it is used.
#[derive(Clone, Debug)]
pub(crate) struct MeshLevel {
    pub(crate) model: Arc<Model>,
    /// Level 0 has `f32::INFINITY` here: it is used whenever no finer
    /// threshold applies.
    pub(crate) below_coverage: f32,
}

/// Where an object's impostor comes from.
// `Asset` holds the baked data inline, as the builder API documents; the size
// gap to the path variants only matters for a handful of descriptions per run.
#[allow(clippy::large_enum_variant)]
pub enum ImpostorSource {
    /// Bake now, in memory.
    Bake(BakeSettings),
    /// Load `dir/impostor-<key>.imp1` when it exists and is valid; otherwise
    /// bake and write it there. The key covers the geometry, materials and
    /// settings (the key is FNV-1a, see `world::cache` in the source).
    BakeCached {
        dir: PathBuf,
        settings: BakeSettings,
    },
    /// Load a file written by [`ImpostorAsset::write_baked`].
    Baked(PathBuf),
    /// An asset already in memory.
    Asset(ImpostorAsset),
}

/// The complete description of one object for [`super::World::add_model`].
/// Build it with [`LodModel::builder`].
pub struct LodModel {
    pub(crate) levels: Vec<MeshLevel>,
    pub(crate) impostor: Option<ImpostorSource>,
    pub(crate) impostor_optional: bool,
    pub(crate) policy: LodPolicy,
    pub(crate) cast_shadows: bool,
    pub(crate) shadow_lod: Option<usize>,
    pub(crate) max_distance: Option<f32>,
}

impl LodModel {
    /// Start from the full-detail mesh `model` (level 0).
    pub fn builder(model: Arc<Model>) -> LodModelBuilder {
        LodModelBuilder {
            model: LodModel {
                levels: vec![MeshLevel {
                    model,
                    below_coverage: f32::INFINITY,
                }],
                impostor: None,
                impostor_optional: false,
                policy: LodPolicy::default(),
                cast_shadows: true,
                shadow_lod: None,
                max_distance: None,
            },
        }
    }
}

/// Builder for [`LodModel`]. Every setter returns the builder.
pub struct LodModelBuilder {
    model: LodModel,
}

impl LodModelBuilder {
    /// Draw `model` instead of the previous level when screen coverage falls
    /// below `below_coverage`. Call in decreasing-detail order with strictly
    /// decreasing thresholds; [`super::World::add_model`] rejects anything else.
    pub fn lod(mut self, model: Arc<Model>, below_coverage: f32) -> Self {
        self.model.levels.push(MeshLevel {
            model,
            below_coverage,
        });
        self
    }

    /// The farthest representation: an impostor for the object.
    pub fn impostor(mut self, source: ImpostorSource) -> Self {
        self.model.impostor = Some(source);
        self
    }

    /// Continue without an impostor when the source cannot be baked, instead
    /// of failing [`super::World::add_model`]. Default `false`.
    pub fn impostor_optional(mut self, yes: bool) -> Self {
        self.model.impostor_optional = yes;
        self
    }

    /// Impostor switch thresholds, hysteresis and crossfade band.
    pub fn policy(mut self, policy: LodPolicy) -> Self {
        self.model.policy = policy;
        self
    }

    /// Whether the object casts sun shadows. Default `true`.
    pub fn cast_shadows(mut self, yes: bool) -> Self {
        self.model.cast_shadows = yes;
        self
    }

    /// Mesh level drawn in shadow passes. Default: the level drawn in the
    /// main pass, or the coarsest mesh when the main pass shows the impostor.
    pub fn shadow_lod(mut self, level: usize) -> Self {
        self.model.shadow_lod = Some(level);
        self
    }

    /// Not drawn at all beyond this distance (metres from the camera to the
    /// bounding sphere's surface). Default: no limit. Unlike the policy's
    /// `max_distance`, this removes the object instead of switching it to its
    /// impostor.
    pub fn max_distance(mut self, metres: f32) -> Self {
        self.model.max_distance = Some(metres);
        self
    }

    /// Finish the description.
    pub fn build(self) -> LodModel {
        self.model
    }
}

/// Mesh level for `coverage` with 10 % hysteresis, starting from `current`.
///
/// Level `k` (k >= 1) is used while coverage is below `belows[k]`. Moving to a
/// coarser level needs coverage under `belows[k + 1] * 0.9`; moving to a finer
/// one needs coverage at or above `belows[k] * 1.1`. Each step is taken one
/// level at a time, so the result never skips a level it did not pass.
pub(crate) fn pick_level(belows: &[f32], current: usize, coverage: f32) -> usize {
    let count = belows.len().max(1);
    let mut level = current.min(count - 1);
    if !coverage.is_finite() {
        return level;
    }
    loop {
        if level + 1 < count && coverage < belows[level + 1] * 0.9 {
            level += 1;
            continue;
        }
        if level > 0 && coverage >= belows[level] * 1.1 {
            level -= 1;
            continue;
        }
        return level;
    }
}

/// Seconds a representation switch takes to crossfade. A still camera settles
/// on one representation within this time, so a crossfade is a transition and
/// never a steady state.
pub(crate) const CROSSFADE_SECONDS: f32 = 0.25;

/// Advance one instance's mesh weight toward the representation `settled`
/// names (a `Mesh` or `Impostor` choice from hysteresis alone) and return the
/// choice to draw this frame. `weight` is `None` before the first frame, which
/// takes the settled representation at once: an object that appears does not
/// fade in from nothing.
pub(crate) fn timed_choice(weight: &mut Option<f32>, settled: LodChoice, dt: f32) -> LodChoice {
    let target = if settled.is_impostor() { 0.0 } else { 1.0 };
    let w = match *weight {
        None => target,
        Some(w) => {
            let step = if dt.is_finite() {
                dt.max(0.0) / CROSSFADE_SECONDS
            } else {
                0.0
            };
            if w < target {
                (w + step).min(target)
            } else {
                (w - step).max(target)
            }
        }
    };
    *weight = Some(w);
    choice_from_weight(w, target)
}

/// The draw choice for mesh weight `w` heading to `target` (0 or 1).
fn choice_from_weight(w: f32, target: f32) -> LodChoice {
    const EPSILON: f32 = 1e-4;
    if w >= 1.0 - EPSILON {
        LodChoice::Mesh
    } else if w <= EPSILON {
        LodChoice::Impostor
    } else {
        LodChoice::Crossfade {
            t: 1.0 - w,
            owner: if target >= 1.0 {
                LodOwner::Impostor
            } else {
                LodOwner::Mesh
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CROSSFADE_SECONDS, pick_level, timed_choice};
    use crate::impostor::LodChoice;

    #[test]
    fn timed_crossfade_settles_within_its_duration_and_never_persists() {
        // A still camera parked inside the coverage band: the hysteresis
        // settles on the impostor, and the weight must reach it in
        // CROSSFADE_SECONDS, after which no frame may crossfade.
        let dt = 1.0 / 60.0;
        let frames = (CROSSFADE_SECONDS / dt).ceil() as usize + 1;
        let mut weight = Some(1.0); // was a mesh
        let mut crossfades = 0;
        let mut last_t = 0.0f32;
        for frame in 0..frames {
            let choice = timed_choice(&mut weight, LodChoice::Impostor, dt);
            if let LodChoice::Crossfade { t, .. } = choice {
                crossfades += 1;
                // Monotonic: the impostor only ever gains weight.
                assert!(t >= last_t, "frame {frame}: weight went backwards");
                last_t = t;
            }
        }
        assert!(crossfades > 0, "a switch must crossfade");
        assert_eq!(
            timed_choice(&mut weight, LodChoice::Impostor, dt),
            LodChoice::Impostor,
            "a still camera must settle on the impostor"
        );
        // A static camera now costs nothing extra for any number of frames.
        for _ in 0..1000 {
            let choice = timed_choice(&mut weight, LodChoice::Impostor, dt);
            assert!(!matches!(choice, LodChoice::Crossfade { .. }));
        }
    }

    #[test]
    fn timed_crossfade_takes_the_first_frame_without_a_fade() {
        let mut weight = None;
        assert_eq!(
            timed_choice(&mut weight, LodChoice::Impostor, 1.0 / 60.0),
            LodChoice::Impostor
        );
        assert_eq!(
            timed_choice(&mut weight, LodChoice::Mesh, 1.0 / 60.0),
            LodChoice::Crossfade {
                t: 1.0 - (1.0 / 60.0) / CROSSFADE_SECONDS,
                owner: crate::impostor::LodOwner::Impostor,
            }
        );
    }

    #[test]
    fn timed_crossfade_on_a_stalled_clock_holds_and_nan_dt_holds() {
        let mut weight = Some(0.5);
        let _ = timed_choice(&mut weight, LodChoice::Mesh, 0.0);
        assert_eq!(weight, Some(0.5));
        let _ = timed_choice(&mut weight, LodChoice::Mesh, f32::NAN);
        assert_eq!(weight, Some(0.5));
    }

    #[test]
    fn world_lod_levels_follow_thresholds_with_hysteresis() {
        // Level 1 below 0.2, level 2 below 0.08.
        let belows = [f32::INFINITY, 0.2, 0.08];
        assert_eq!(pick_level(&belows, 0, 0.5), 0);
        assert_eq!(pick_level(&belows, 0, 0.15), 1);
        assert_eq!(pick_level(&belows, 0, 0.05), 2);
        // Just above the level-1 threshold: stays coarse until 10 % above it.
        assert_eq!(pick_level(&belows, 1, 0.21), 1);
        assert_eq!(pick_level(&belows, 1, 0.23), 0);
        // Just below: the coarser level needs 10 % below the threshold.
        assert_eq!(pick_level(&belows, 0, 0.19), 0);
        assert_eq!(pick_level(&belows, 0, 0.17), 1);
        assert_eq!(pick_level(&belows, 1, 0.19), 1);
        assert_eq!(pick_level(&belows, 1, 0.17), 1);
        assert_eq!(pick_level(&belows, 1, 0.16), 1);
        // Non-finite coverage keeps the current level rather than flapping.
        assert_eq!(pick_level(&belows, 2, f32::NAN), 2);
    }

    #[test]
    fn world_lod_never_skips_a_level_it_did_not_pass() {
        let belows = [f32::INFINITY, 0.2, 0.08];
        // Coverage falls from far to very small in one frame: it lands on the
        // coarsest level in one step, which is what the threshold says.
        assert_eq!(pick_level(&belows, 0, 0.01), 2);
        // And back: from coarse to fine must pass the finer thresholds too.
        assert_eq!(pick_level(&belows, 2, 0.3), 0);
    }
}

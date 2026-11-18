//! CPU side of Hi-Z occlusion culling.
//!
//! The renderer reduces the scene depth to a small grid of farthest depths
//! (one per block) and reads it back a few frames later. [`OcclusionBuffer`]
//! holds that grid, builds a max-depth pyramid from it and answers "is this
//! box hidden behind what was drawn?". Every answer errs towards "visible": a
//! box is only culled when it provably lies behind the farthest depth of every
//! texel its screen rectangle touches.

use super::Bounds;
use glam::{Mat4, Vec3};

/// What the GPU data was rendered with, copied at record time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionMeta {
    /// Unjittered view-projection of the frame the depth came from. Depth
    /// `0..1`, `-1..1` in x and y, as the renderer's camera matrices are.
    pub view_projection: Mat4,
    /// Camera position of that frame.
    pub camera_position: Vec3,
    /// Frame counter of that frame ([`crate::gfx::DrawCtx::frame_id`]).
    pub frame_id: u64,
    /// Screen-space inflation of tested rectangles, in depth texels.
    pub margin_texels: f32,
    /// Relative slack on the depth comparison.
    pub depth_bias: f32,
}

/// One level of the max pyramid: `data[y * width + x]`.
#[derive(Clone, Debug)]
struct Level {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

/// The farthest depth per block of the scene, with a max pyramid over it.
///
/// Texel `(x, y)` covers the normalised device area
/// `x / width * 2 - 1 ..`, `y / height * 2 - 1 ..` (the texel uv is
/// `ndc * 0.5 + 0.5`, rows bottom-up as the scene depth stores them).
#[derive(Clone, Debug)]
pub struct OcclusionBuffer {
    levels: Vec<Level>,
    meta: OcclusionMeta,
}

impl OcclusionBuffer {
    /// Build from a `width x height` grid of farthest depths, row-major.
    ///
    /// # Panics
    /// If `depth.len() != width * height` or either dimension is zero.
    pub fn new(width: u32, height: u32, depth: Vec<f32>, meta: OcclusionMeta) -> Self {
        assert!(width > 0 && height > 0, "occlusion grid must not be empty");
        assert_eq!(
            depth.len(),
            width as usize * height as usize,
            "occlusion grid size mismatch"
        );
        let mut levels = vec![Level {
            width: width as usize,
            height: height as usize,
            data: depth,
        }];
        loop {
            let top = levels.last().expect("at least one level");
            if top.width <= 1 && top.height <= 1 {
                break;
            }
            let next = reduce(top);
            levels.push(next);
        }
        Self { levels, meta }
    }

    /// Grid width in texels.
    pub fn width(&self) -> u32 {
        self.levels[0].width as u32
    }

    /// Grid height in texels.
    pub fn height(&self) -> u32 {
        self.levels[0].height as u32
    }

    /// Everything the depth was rendered with.
    pub fn meta(&self) -> &OcclusionMeta {
        &self.meta
    }

    /// Frame counter of the frame the depth came from.
    pub fn frame_id(&self) -> u64 {
        self.meta.frame_id
    }

    /// The farthest depth of the level-0 texel at `(x, y)`.
    pub fn texel(&self, x: u32, y: u32) -> f32 {
        let l = &self.levels[0];
        l.data[y as usize * l.width + x as usize]
    }

    /// Distance the camera has moved since the depth was rendered.
    pub fn camera_motion(&self, current_position: Vec3) -> f32 {
        (current_position - self.meta.camera_position).length()
    }

    /// Whether `bounds` is hidden behind the stored depth, as seen from the
    /// stored camera. Not culled when the box touches a corner behind the near
    /// plane, or when any part of its screen rectangle lies outside the grid.
    pub fn is_occluded(&self, bounds: &Bounds) -> bool {
        self.is_occluded_inflated(bounds, 0.0)
    }

    /// [`Self::is_occluded`] with the box first grown by `extra` world units
    /// on every side. Use it for fast movers and for the camera's motion since
    /// the depth was rendered.
    ///
    /// The depth describes the scene from the stored camera, so a box is only
    /// hidden for the camera that moved if the occluders stayed put in view.
    /// Occluders are nearer than the box and shift more when the camera moves,
    /// so `extra` must cover the camera's motion with a margin: growing the box
    /// by the motion alone leaves slivers at silhouettes. [`crate::world::World`]
    /// uses three times the motion.
    pub fn is_occluded_inflated(&self, bounds: &Bounds, extra: f32) -> bool {
        let grow = Vec3::splat(extra.max(0.0));
        let Some((rect, nearest)) = self.project(bounds.aabb_min - grow, bounds.aabb_max + grow)
        else {
            return false;
        };
        let (w, h) = (self.width() as f32, self.height() as f32);
        let margin = self.meta.margin_texels.max(0.0);
        let x0 = rect[0] - margin;
        let y0 = rect[1] - margin;
        let x1 = rect[2] + margin;
        let y1 = rect[3] + margin;
        if x0 < 0.0 || y0 < 0.0 || x1 > w || y1 > h {
            return false;
        }
        let bx0 = x0.floor() as usize;
        let by0 = y0.floor() as usize;
        let bx1 = (x1.floor() as usize).min(self.width() as usize - 1);
        let by1 = (y1.floor() as usize).min(self.height() as usize - 1);
        let far = self.max_over(bx0, by0, bx1, by1);
        let bias = self.meta.depth_bias.max(0.0);
        nearest > far * (1.0 + bias) + 1e-6
    }

    /// Project the eight corners of `[min, max]` with the stored matrix.
    /// Returns the texel rectangle `[x0, y0, x1, y1]` and the nearest corner
    /// depth, or `None` when any corner is outside the `0..1` depth range or
    /// behind the eye.
    fn project(&self, min: Vec3, max: Vec3) -> Option<([f32; 4], f32)> {
        let (w, h) = (self.width() as f32, self.height() as f32);
        let vp = self.meta.view_projection;
        let extent = max - min;
        let c0 = vp * min.extend(1.0);
        let dx = vp.x_axis * extent.x;
        let dy = vp.y_axis * extent.y;
        let dz = vp.z_axis * extent.z;
        let c1 = c0 + dx;
        let c2 = c0 + dy;
        let c3 = c1 + dy;
        let corners = [c0, c1, c2, c3, c0 + dz, c1 + dz, c2 + dz, c3 + dz];

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut nearest = f32::INFINITY;
        for clip in corners {
            if !clip.w.is_finite() || clip.w <= 1e-6 {
                return None;
            }
            let inv_w = 1.0 / clip.w;
            let nz = clip.z * inv_w;
            if !(0.0..=1.0).contains(&nz) {
                return None;
            }
            let tx = (clip.x * inv_w * 0.5 + 0.5) * w;
            let ty = (clip.y * inv_w * 0.5 + 0.5) * h;
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
            nearest = nearest.min(nz);
        }
        Some(([min_x, min_y, max_x, max_y], nearest))
    }

    /// Maximum depth over the base texels `[x0, x1] x [y0, y1]` (inclusive),
    /// read from the finest pyramid level that needs at most a 2x2 lookup.
    fn max_over(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
        for (lod, level) in self.levels.iter().enumerate() {
            let (lx0, lx1) = (x0 >> lod, x1 >> lod);
            let (ly0, ly1) = (y0 >> lod, y1 >> lod);
            if lx1 - lx0 < 2 && ly1 - ly0 < 2 {
                let mut far = 0.0f32;
                for y in ly0..=ly1.min(level.height - 1) {
                    for x in lx0..=lx1.min(level.width - 1) {
                        far = far.max(level.data[y * level.width + x]);
                    }
                }
                return far;
            }
        }
        // The top level is 1x1, so the loop above always returns.
        1.0
    }

    /// Level `lod` of the pyramid as `(width, height, data)`, for tests and
    /// diagnostics.
    pub fn level(&self, lod: usize) -> Option<(u32, u32, &[f32])> {
        self.levels
            .get(lod)
            .map(|l| (l.width as u32, l.height as u32, l.data.as_slice()))
    }
}

/// Max-reduce `src` by two in each axis; odd edges keep the partial block.
fn reduce(src: &Level) -> Level {
    let width = src.width.div_ceil(2).max(1);
    let height = src.height.div_ceil(2).max(1);
    let mut data = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut far = 0.0f32;
            for sy in (2 * y)..(2 * y + 2).min(src.height) {
                for sx in (2 * x)..(2 * x + 2).min(src.width) {
                    far = far.max(src.data[sy * src.width + sx]);
                }
            }
            data[y * width + x] = far;
        }
    }
    Level {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests;

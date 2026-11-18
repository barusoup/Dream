use super::{OcclusionBuffer, OcclusionMeta};
use crate::cull::Bounds;
use glam::{Mat4, Vec3};

const W: u32 = 128;
const H: u32 = 72;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Camera at the origin looking down -Z, near 1 m so a wall at NDC depth 0.5
/// sits about 1.98 m away.
fn camera() -> Mat4 {
    Mat4::perspective_rh(70f32.to_radians(), W as f32 / H as f32, 1.0, 100.0)
        * Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y)
}

fn meta(vp: Mat4, margin: f32, bias: f32) -> OcclusionMeta {
    OcclusionMeta {
        view_projection: vp,
        camera_position: Vec3::ZERO,
        frame_id: 1,
        margin_texels: margin,
        depth_bias: bias,
    }
}

/// Depth image of a wall covering the left half of the screen at NDC depth
/// 0.5; the right half is empty sky (depth 1).
fn wall_depth() -> Vec<f32> {
    let mut depth = vec![0.0f32; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let ndc_x = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            depth[(y * W + x) as usize] = if ndc_x < 0.0 { 0.5 } else { 1.0 };
        }
    }
    depth
}

fn wall_buffer(margin: f32) -> OcclusionBuffer {
    let vp = camera();
    OcclusionBuffer::new(W, H, wall_depth(), meta(vp, margin, 0.002))
}

fn boxed(centre: Vec3, half: Vec3) -> Bounds {
    Bounds::from_aabb(centre - half, centre + half)
}

#[test]
fn occlusion_pyramid_is_the_max_of_each_block() {
    let (w, h) = (37u32, 19u32); // odd sizes exercise the partial edge blocks
    let mut rng = Lcg(11);
    let depth: Vec<f32> = (0..w * h).map(|_| rng.range(0.0, 1.0)).collect();
    let buffer = OcclusionBuffer::new(w, h, depth.clone(), meta(Mat4::IDENTITY, 0.0, 0.0));
    let mut lod = 0;
    let (mut lw, mut lh) = (w as usize, h as usize);
    while let Some((gw, gh, data)) = buffer.level(lod) {
        assert_eq!((gw as usize, gh as usize), (lw, lh), "level {lod} size");
        let block = 1usize << lod;
        for y in 0..lh {
            for x in 0..lw {
                let mut far = 0.0f32;
                for sy in (y * block)..((y + 1) * block).min(h as usize) {
                    for sx in (x * block)..((x + 1) * block).min(w as usize) {
                        far = far.max(depth[sy * w as usize + sx]);
                    }
                }
                assert_eq!(data[y * lw + x], far, "level {lod} texel {x},{y}");
            }
        }
        lod += 1;
        lw = lw.div_ceil(2).max(1);
        lh = lh.div_ceil(2).max(1);
    }
    assert_eq!(buffer.level(lod - 1).map(|(_, _, d)| d.len()), Some(1));
}

#[test]
fn occlusion_box_behind_the_wall_is_culled() {
    let buffer = wall_buffer(1.5);
    let behind_left = boxed(Vec3::new(-5.0, 0.0, -10.0), Vec3::splat(0.5));
    assert!(buffer.is_occluded(&behind_left));
    let far_behind = boxed(Vec3::new(-8.0, 1.0, -30.0), Vec3::splat(1.0));
    assert!(buffer.is_occluded(&far_behind));
}

#[test]
fn occlusion_box_in_front_of_the_wall_is_kept() {
    let buffer = wall_buffer(1.5);
    let in_front = boxed(Vec3::new(-0.5, 0.0, -1.3), Vec3::splat(0.1));
    assert!(!buffer.is_occluded(&in_front));
}

#[test]
fn occlusion_box_on_the_empty_side_or_over_the_edge_is_kept() {
    let buffer = wall_buffer(1.5);
    let right = boxed(Vec3::new(5.0, 0.0, -10.0), Vec3::splat(0.5));
    assert!(!buffer.is_occluded(&right));
    // Straddles x = 0: the right half of its rectangle sees sky.
    let edge = boxed(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(0.5));
    assert!(!buffer.is_occluded(&edge));
}

#[test]
fn occlusion_box_touching_the_screen_border_is_kept() {
    let buffer = wall_buffer(1.5);
    // Partly off the left edge of the grid: the hidden part is unknown.
    let off_screen = boxed(Vec3::new(-30.0, 0.0, -10.0), Vec3::splat(0.5));
    assert!(!buffer.is_occluded(&off_screen));
}

#[test]
fn occlusion_box_with_a_corner_behind_the_eye_is_kept() {
    let buffer = wall_buffer(1.5);
    let straddling_eye = boxed(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.5));
    assert!(!buffer.is_occluded(&straddling_eye));
}

#[test]
fn occlusion_inflation_makes_culling_more_conservative() {
    let buffer = wall_buffer(1.5);
    // A box just behind the wall is culled only while it is not inflated
    // towards the wall.
    let near_wall = boxed(Vec3::new(-5.0, 0.0, -2.1), Vec3::splat(0.2));
    let plain = buffer.is_occluded(&near_wall);
    let inflated = buffer.is_occluded_inflated(&near_wall, 1.0);
    assert!(!inflated || plain, "inflation must not add culls");
    let camera_moved = buffer.camera_motion(Vec3::new(3.0, 0.0, 0.0));
    assert!((camera_moved - 3.0).abs() < 1e-6);
}

/// Scene depth of the brute-force reference: the same wall, one texel per
/// pixel, plus an extra occluder block so some boxes are only partly hidden.
fn reference_depth() -> Vec<f32> {
    let mut depth = wall_depth();
    for y in 10..20 {
        for x in 90..100 {
            depth[(y * W + x) as usize] = 0.3;
        }
    }
    depth
}

/// Whether the box's front surface is visible at any pixel centre, using
/// exact ray-box tests against the per-pixel scene depth.
fn any_pixel_visible(vp: Mat4, depth: &[f32], min: Vec3, max: Vec3) -> bool {
    let inv = vp.inverse();
    let origin = Vec3::ZERO;
    for y in 0..H {
        for x in 0..W {
            let ndc = Vec3::new(
                (x as f32 + 0.5) / W as f32 * 2.0 - 1.0,
                (y as f32 + 0.5) / H as f32 * 2.0 - 1.0,
                1.0,
            );
            let far = inv * ndc.extend(1.0);
            let dir = (far.truncate() / far.w - origin).normalize();
            let Some(t) = ray_box(origin, dir, min, max) else {
                continue;
            };
            let hit = origin + dir * t;
            let clip = vp * hit.extend(1.0);
            let box_depth = clip.z / clip.w;
            if box_depth < depth[(y * W + x) as usize] {
                return true;
            }
        }
    }
    false
}

fn ray_box(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let mut t0 = f32::NEG_INFINITY;
    let mut t1 = f32::INFINITY;
    for axis in 0..3 {
        let (o, d, lo, hi) = (origin[axis], dir[axis], min[axis], max[axis]);
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let a = (lo - o) / d;
            let b = (hi - o) / d;
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
    }
    (t1 >= t0.max(0.0)).then_some(t0.max(0.0))
}

#[test]
fn occlusion_never_culls_a_box_with_a_visible_pixel() {
    let vp = camera();
    let depth = reference_depth();
    let buffer = OcclusionBuffer::new(W, H, depth.clone(), meta(vp, 1.5, 0.002));
    let mut rng = Lcg(0xdec0de);
    let mut hidden = 0;
    let mut culled_hidden = 0;
    for _ in 0..1500 {
        let centre = Vec3::new(
            rng.range(-25.0, 25.0),
            rng.range(-12.0, 12.0),
            rng.range(-40.0, -3.0),
        );
        let half = Vec3::new(
            rng.range(0.1, 2.0),
            rng.range(0.1, 2.0),
            rng.range(0.1, 2.0),
        );
        let bounds = boxed(centre, half);
        let occluded = buffer.is_occluded(&bounds);
        let visible = any_pixel_visible(vp, &depth, bounds.aabb_min, bounds.aabb_max);
        assert!(
            !(occluded && visible),
            "culled a visible box {centre:?} half {half:?}"
        );
        // "With margin": well inside the wall's half, so the grid's margin and
        // the wall edge do not decide the outcome.
        if !visible && centre.x <= -3.0 && centre.z <= -8.0 {
            hidden += 1;
            if occluded {
                culled_hidden += 1;
            }
        }
    }
    assert!(hidden > 200, "too few hidden samples: {hidden}");
    assert!(
        culled_hidden * 10 >= hidden * 7,
        "culled only {culled_hidden} of {hidden} hidden boxes"
    );
}

#[test]
fn occlusion_hidden_boxes_with_margin_are_mostly_culled() {
    let vp = camera();
    let buffer = OcclusionBuffer::new(W, H, reference_depth(), meta(vp, 1.5, 0.002));
    let mut rng = Lcg(42);
    let mut tested = 0;
    let mut culled = 0;
    for _ in 0..2000 {
        // Behind the left wall and well inside its half, so the rectangle
        // keeps the margin clear of the wall's edge and of the grid border.
        let centre = Vec3::new(
            rng.range(-6.0, -2.0),
            rng.range(-2.0, 2.0),
            rng.range(-30.0, -9.0),
        );
        let bounds = boxed(centre, Vec3::splat(rng.range(0.1, 0.6)));
        tested += 1;
        if buffer.is_occluded(&bounds) {
            culled += 1;
        }
    }
    assert!(culled * 10 >= tested * 7, "culled {culled} of {tested}");
}

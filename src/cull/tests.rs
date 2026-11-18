use super::{Bounds, Frustum};
use glam::{Mat4, Vec3};

fn perspective_camera() -> Mat4 {
    let fov = 70f32.to_radians();
    let aspect = 16.0 / 9.0;
    Mat4::perspective_rh(fov, aspect, 0.1, 100.0)
        * Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y)
}

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

#[test]
fn cull_points_inside_and_outside_each_plane() {
    let frustum = Frustum::from_view_projection(perspective_camera());
    let inside = Vec3::new(0.0, 0.0, -10.0);
    assert!(frustum.contains_sphere(inside, 0.0));
    // Behind the eye.
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, 10.0), 0.0));
    // Beyond the far plane.
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, -200.0), 0.0));
    // In front of the near plane.
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, -0.05), 0.0));
    // Right plane: half-width at depth 10 is tan(35) * 16/9 * 10 = 12.446.
    assert!(frustum.contains_sphere(Vec3::new(12.0, 0.0, -10.0), 0.0));
    assert!(!frustum.contains_sphere(Vec3::new(13.0, 0.0, -10.0), 0.0));
    // Left plane.
    assert!(!frustum.contains_sphere(Vec3::new(-13.0, 0.0, -10.0), 0.0));
    // Top and bottom: half-height at depth 10 is tan(35) * 10 = 7.002.
    assert!(frustum.contains_sphere(Vec3::new(0.0, 6.9, -10.0), 0.0));
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 7.1, -10.0), 0.0));
    assert!(!frustum.contains_sphere(Vec3::new(0.0, -7.1, -10.0), 0.0));
}

#[test]
fn cull_sphere_straddling_a_plane_is_kept() {
    let frustum = Frustum::from_view_projection(perspective_camera());
    // The right plane passes through the origin and (tan(hfov/2) * 10, 0, -10),
    // with unit normal proportional to (1, 0, tan(hfov/2)).
    let tan_h = (70f32.to_radians() * 0.5).tan() * (16.0 / 9.0);
    let normal = Vec3::new(1.0, 0.0, tan_h).normalize();
    let on_plane = Vec3::new(tan_h * 10.0, 0.0, -10.0);
    // 0.5 m outside the plane: a 0.6 m sphere reaches it, a 0.4 m one does not.
    let centre = on_plane + normal * 0.5;
    assert!(frustum.contains_sphere(centre, 0.6));
    assert!(!frustum.contains_sphere(centre, 0.4));
    // Straddling the near plane.
    assert!(frustum.contains_sphere(Vec3::new(0.0, 0.0, -0.1), 0.2));
}

#[test]
fn cull_orthographic_cascade_matrix() {
    let ortho = Mat4::orthographic_rh(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0)
        * Mat4::look_to_rh(Vec3::new(0.0, 0.0, 50.0), Vec3::NEG_Z, Vec3::Y);
    let frustum = Frustum::from_view_projection(ortho);
    assert!(frustum.contains_sphere(Vec3::new(4.9, 4.9, 0.0), 0.0));
    assert!(!frustum.contains_sphere(Vec3::new(5.5, 0.0, 0.0), 0.0));
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, 60.0), 0.0));
    assert!(frustum.contains_aabb(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, 10.0)));
    assert!(!frustum.contains_aabb(Vec3::new(6.0, -1.0, -10.0), Vec3::new(7.0, 1.0, 10.0)));
}

#[test]
fn cull_batch_matches_single_tests() {
    let frustum = Frustum::from_view_projection(perspective_camera());
    let mut rng = Lcg(7);
    let spheres: Vec<[f32; 4]> = (0..500)
        .map(|_| {
            [
                rng.range(-30.0, 30.0),
                rng.range(-20.0, 20.0),
                rng.range(-120.0, 10.0),
                rng.range(0.0, 3.0),
            ]
        })
        .collect();
    let mut visible = vec![999u32];
    frustum.cull_spheres(&spheres, &mut visible);
    assert_eq!(visible[0], 999, "batch appends, it does not clear");
    let expected: Vec<u32> = spheres
        .iter()
        .enumerate()
        .filter(|(_, s)| frustum.contains_sphere(Vec3::new(s[0], s[1], s[2]), s[3]))
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(&visible[1..], expected.as_slice());
}

#[test]
fn cull_never_rejects_a_box_with_a_corner_inside_the_clip_volume() {
    let vp = perspective_camera();
    let frustum = Frustum::from_view_projection(vp);
    let mut rng = Lcg(0x5eed);
    let mut checked_visible = 0;
    let mut checked_culled = 0;
    for _ in 0..3000 {
        let centre = Vec3::new(
            rng.range(-40.0, 40.0),
            rng.range(-30.0, 30.0),
            rng.range(-150.0, 20.0),
        );
        let half = Vec3::new(
            rng.range(0.05, 6.0),
            rng.range(0.05, 6.0),
            rng.range(0.05, 6.0),
        );
        let (min, max) = (centre - half, centre + half);
        // Brute force: does any corner land inside the clip volume?
        let mut any_inside = false;
        for corner in 0..8u32 {
            let p = Vec3::new(
                if corner & 1 != 0 { max.x } else { min.x },
                if corner & 2 != 0 { max.y } else { min.y },
                if corner & 4 != 0 { max.z } else { min.z },
            );
            let c = vp * p.extend(1.0);
            if c.w > 0.0 {
                let n = c.truncate() / c.w;
                if n.x.abs() <= 1.0 && n.y.abs() <= 1.0 && (0.0..=1.0).contains(&n.z) {
                    any_inside = true;
                }
            }
        }
        let kept = frustum.contains_aabb(min, max);
        if any_inside {
            assert!(kept, "culled a box with a corner inside: {min:?} {max:?}");
            checked_visible += 1;
        } else if !kept {
            checked_culled += 1;
        }
    }
    assert!(
        checked_visible > 100,
        "too few visible samples: {checked_visible}"
    );
    assert!(
        checked_culled > 100,
        "too few culled samples: {checked_culled}"
    );
}

#[test]
fn cull_bounds_from_local_encloses_the_transformed_box() {
    let transform = Mat4::from_rotation_y(0.7) * Mat4::from_scale(Vec3::new(2.0, 1.0, 0.5));
    let bounds = Bounds::from_local(Vec3::splat(-1.0), Vec3::new(1.0, 2.0, 3.0), transform);
    for corner in 0..8u32 {
        let p = Vec3::new(
            if corner & 1 != 0 { 1.0 } else { -1.0 },
            if corner & 2 != 0 { 2.0 } else { -1.0 },
            if corner & 4 != 0 { 3.0 } else { -1.0 },
        );
        let w = transform.transform_point3(p);
        assert!(w.cmpge(bounds.aabb_min - Vec3::splat(1e-4)).all());
        assert!(w.cmple(bounds.aabb_max + Vec3::splat(1e-4)).all());
        assert!((w - bounds.centre).length() <= bounds.radius + 1e-4);
    }
    let grown = bounds.inflated(0.5);
    assert!((grown.radius - bounds.radius - 0.5).abs() < 1e-5);
    assert!(
        (grown.aabb_max - bounds.aabb_max - Vec3::splat(0.5))
            .abs()
            .max_element()
            < 1e-5
    );
}

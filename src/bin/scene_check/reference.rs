//! CPU ray-traced reference for the `scene_check` test scene.
//!
//! [`scene_items`] is the one list the scene is built from: `build_scene`
//! turns each item into a GPU model, and [`Scene::trace`] intersects the same
//! analytic primitive. The camera mapping, [`View`], reproduces the renderer's
//! unjittered view-projection, so a reference hit can be projected back to a
//! pixel and compared with the GPU's depth and its screen-space outputs.
//!
//! Everything here runs in `f64` on the CPU. Hit positions are analytic, so the
//! reference includes geometry that screen-space methods cannot see.
use glam::{DMat4, DVec3, Mat4};

/// Ray origins are pushed this far along the surface normal, so a ray leaving a
/// surface does not hit that surface again.
const SURFACE_OFFSET: f64 = 1e-4;
/// Smallest accepted ray parameter.
const T_MIN: f64 = 1e-6;
/// Samples of the cosine-weighted hemisphere used for reference AO.
pub const AO_SAMPLES: usize = 256;

/// A scene primitive. Planes face +Y and are two-sided for intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Horizontal rectangle with half extents on X and Z.
    Plane { half_x: f32, half_z: f32 },
    /// Axis-aligned box with half extents on each axis.
    Box { half: [f32; 3] },
    /// Sphere.
    Sphere { radius: f32 },
}

/// One object of the test scene: its primitive, where it sits, and its
/// material (base colour, metallic and roughness factors).
#[derive(Clone, Copy, Debug)]
pub struct SceneItem {
    pub name: &'static str,
    pub shape: Shape,
    pub centre: [f32; 3],
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

/// The objects of the test scene, in draw order. Index 0 is the ground.
pub fn scene_items() -> Vec<SceneItem> {
    let roughness = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut items = vec![SceneItem {
        name: "ground",
        shape: Shape::Plane {
            half_x: 30.0,
            half_z: 30.0,
        },
        centre: [0.0, 0.0, 0.0],
        base_color: [0.5, 0.5, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.6,
    }];
    for (i, rough) in roughness.into_iter().enumerate() {
        let x = i as f32 - 2.0;
        items.push(SceneItem {
            name: "red dielectric",
            shape: Shape::Sphere { radius: 0.5 },
            centre: [x, 0.5, -2.0],
            base_color: [0.8, 0.1, 0.1, 1.0],
            metallic: 0.0,
            roughness: rough,
        });
        items.push(SceneItem {
            name: "gold metal",
            shape: Shape::Sphere { radius: 0.5 },
            centre: [x, 0.5, 1.0],
            base_color: [1.0, 0.77, 0.34, 1.0],
            metallic: 1.0,
            roughness: rough,
        });
    }
    let cube = Shape::Box { half: [0.5; 3] };
    items.push(SceneItem {
        name: "white cube",
        shape: cube,
        centre: [3.5, 0.5, 0.0],
        base_color: [0.95, 0.95, 0.95, 1.0],
        metallic: 0.0,
        roughness: 0.5,
    });
    items.push(SceneItem {
        name: "pillar",
        shape: Shape::Box {
            half: [0.1, 1.5, 0.1],
        },
        centre: [-7.4, 1.5, -4.0],
        base_color: [0.7, 0.7, 0.75, 1.0],
        metallic: 0.0,
        roughness: 0.5,
    });
    // Bottom face at 0.5 m above the ground, for shadow and AO overhang cases.
    items.push(SceneItem {
        name: "slab",
        shape: Shape::Box {
            half: [1.5, 0.1, 1.0],
        },
        centre: [0.0, 0.6, -4.0],
        base_color: [0.6, 0.5, 0.4, 1.0],
        metallic: 0.0,
        roughness: 0.5,
    });
    // Colour bleeding: a saturated red wall facing the camera, a white cube at
    // its left end, and the open ground in front of it.
    items.push(SceneItem {
        name: "red wall",
        shape: Shape::Box {
            half: [1.5, 1.25, 0.1],
        },
        centre: [-5.2, 1.25, -2.6],
        base_color: [0.8, 0.05, 0.05, 1.0],
        metallic: 0.0,
        roughness: 0.8,
    });
    items.push(SceneItem {
        name: "white twin",
        shape: cube,
        centre: [-5.6, 0.5, -1.8],
        base_color: [0.95, 0.95, 0.95, 1.0],
        metallic: 0.0,
        roughness: 0.5,
    });
    items.push(SceneItem {
        name: "green wall",
        shape: Shape::Box {
            half: [1.5, 1.25, 0.1],
        },
        centre: [4.2, 1.25, -2.4],
        base_color: [0.05, 0.7, 0.1, 1.0],
        metallic: 0.0,
        roughness: 0.8,
    });
    // Smooth dark patches just above the ground, for the reflection cases.
    items.push(SceneItem {
        name: "dark floor",
        shape: Shape::Plane {
            half_x: 1.2,
            half_z: 1.2,
        },
        centre: [0.6, 0.01, 0.6],
        base_color: [0.03, 0.03, 0.035, 1.0],
        metallic: 0.0,
        roughness: 0.05,
    });
    items.push(SceneItem {
        name: "glossy floor",
        shape: Shape::Plane {
            half_x: 1.2,
            half_z: 1.2,
        },
        centre: [-1.8, 0.01, 0.6],
        base_color: [0.03, 0.03, 0.035, 1.0],
        metallic: 0.0,
        roughness: 0.3,
    });
    // The mirror floor: about 6 x 4 m, in front of the green wall and the
    // white cube, so the camera sees both reflected.
    items.push(SceneItem {
        name: MIRROR_FLOOR,
        shape: Shape::Plane {
            half_x: 3.0,
            half_z: 2.0,
        },
        centre: [5.0, 0.01, 0.4],
        // Metal: F0 is the base colour, so the reflection is 75% of the
        // reflected light.
        base_color: [0.75, 0.75, 0.75, 1.0],
        metallic: 1.0,
        roughness: 0.05,
    });
    items
}

/// Name of the large mirror floor patch in [`scene_items`].
pub const MIRROR_FLOOR: &str = "mirror floor";

/// Ray with a unit direction.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: DVec3,
    pub dir: DVec3,
}

/// The nearest intersection along a ray.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    /// Distance along the ray (the direction is unit length).
    pub t: f64,
    pub position: DVec3,
    /// Unit surface normal, facing the ray.
    pub normal: DVec3,
    /// Index into [`scene_items`].
    pub object_index: usize,
}

/// Surface categories of the statistics tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Ground,
    Patch,
    SphereUpper,
    SphereLower,
    /// Box faces with |n.y| < 0.3: walls and cube sides.
    Vertical,
    /// Other box faces (tops and bottoms).
    Top,
}

impl Category {
    pub const ALL: [Category; 6] = [
        Category::Ground,
        Category::Patch,
        Category::SphereUpper,
        Category::SphereLower,
        Category::Vertical,
        Category::Top,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Ground => "ground",
            Category::Patch => "patches",
            Category::SphereUpper => "sphere upper",
            Category::SphereLower => "sphere lower",
            Category::Vertical => "vertical faces",
            Category::Top => "box tops",
        }
    }
}

/// The analytic scene.
pub struct Scene {
    items: Vec<SceneItem>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            items: scene_items(),
        }
    }

    /// Index of the item with `name`.
    pub fn index_of(&self, name: &str) -> usize {
        self.items
            .iter()
            .position(|item| item.name == name)
            .unwrap_or_else(|| panic!("no scene item named {name}"))
    }

    pub fn item(&self, index: usize) -> &SceneItem {
        &self.items[index]
    }

    pub fn items(&self) -> &[SceneItem] {
        &self.items
    }

    /// Nearest hit with `t < t_max`, if any.
    pub fn trace(&self, ray: Ray, t_max: f64) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (index, item) in self.items.iter().enumerate() {
            let limit = best.map_or(t_max, |hit| hit.t);
            if let Some(hit) = intersect(item, ray, limit) {
                best = Some(Hit {
                    object_index: index,
                    ..hit
                });
            }
        }
        best
    }

    /// Surface category of a hit.
    pub fn category(&self, hit: &Hit) -> Category {
        match self.items[hit.object_index].shape {
            Shape::Plane { .. } if hit.object_index == 0 => Category::Ground,
            Shape::Plane { .. } => Category::Patch,
            Shape::Sphere { .. } if hit.normal.y >= 0.0 => Category::SphereUpper,
            Shape::Sphere { .. } => Category::SphereLower,
            Shape::Box { .. } if hit.normal.y.abs() < 0.3 => Category::Vertical,
            Shape::Box { .. } => Category::Top,
        }
    }

    /// True when nothing lies between `eye` and `point` (the point's own
    /// surface is not counted).
    pub fn visible_from(&self, eye: DVec3, point: DVec3) -> bool {
        let offset = point - eye;
        let dist = offset.length();
        let ray = Ray {
            origin: eye,
            dir: offset / dist,
        };
        self.trace(ray, dist - 2e-3).is_none()
    }
}

fn intersect(item: &SceneItem, ray: Ray, t_max: f64) -> Option<Hit> {
    let c = DVec3::from_array(item.centre.map(f64::from));
    let hit = |t: f64, normal: DVec3| {
        Some(Hit {
            t,
            position: ray.origin + ray.dir * t,
            normal,
            object_index: 0,
        })
    };
    match item.shape {
        Shape::Plane { half_x, half_z } => {
            if ray.dir.y.abs() < 1e-12 {
                return None;
            }
            let t = (c.y - ray.origin.y) / ray.dir.y;
            if t <= T_MIN || t >= t_max {
                return None;
            }
            let p = ray.origin + ray.dir * t;
            if (p.x - c.x).abs() > f64::from(half_x) || (p.z - c.z).abs() > f64::from(half_z) {
                return None;
            }
            let normal = if ray.dir.y < 0.0 { DVec3::Y } else { -DVec3::Y };
            hit(t, normal)
        }
        Shape::Box { half } => {
            let h = DVec3::from_array(half.map(f64::from));
            let (lo, hi) = (c - h, c + h);
            let mut t_near = f64::NEG_INFINITY;
            let mut t_far = f64::INFINITY;
            let mut near_axis = 0;
            for axis in 0..3 {
                let (o, d) = (ray.origin[axis], ray.dir[axis]);
                let (mut t0, mut t1) = if d.abs() < 1e-15 {
                    if o < lo[axis] || o > hi[axis] {
                        return None;
                    }
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    ((lo[axis] - o) / d, (hi[axis] - o) / d)
                };
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                if t0 > t_near {
                    t_near = t0;
                    near_axis = axis;
                }
                t_far = t_far.min(t1);
            }
            if t_near > t_far || t_far <= T_MIN || t_near >= t_max {
                return None;
            }
            if t_near <= T_MIN {
                // The origin is inside the box: not a visible surface.
                return None;
            }
            let mut normal = DVec3::ZERO;
            normal[near_axis] = if ray.dir[near_axis] > 0.0 { -1.0 } else { 1.0 };
            hit(t_near, normal)
        }
        Shape::Sphere { radius } => {
            let r = f64::from(radius);
            let oc = ray.origin - c;
            let b = oc.dot(ray.dir);
            let disc = b * b - (oc.dot(oc) - r * r);
            if disc < 0.0 {
                return None;
            }
            let sq = disc.sqrt();
            let mut t = -b - sq;
            if t <= T_MIN {
                t = -b + sq;
            }
            if t <= T_MIN || t >= t_max {
                return None;
            }
            let p = ray.origin + ray.dir * t;
            hit(t, (p - c) / r)
        }
    }
}

/// The renderer's camera as seen by the reference: unjittered view-projection
/// and the eye, with the texel-row order of the readbacks.
pub struct View {
    pub width: u32,
    pub height: u32,
    pub eye: DVec3,
    pub near: f64,
    pub far: f64,
    vp: DMat4,
    inv: DMat4,
    /// Row 0 of the readbacks is the top of the image (NDC y = +1).
    pub top_first: bool,
}

impl View {
    pub fn new(vp: Mat4, eye: DVec3, size: (u32, u32), clip: (f32, f32), top_first: bool) -> Self {
        let vp = DMat4::from_cols_array(&vp.to_cols_array().map(f64::from));
        Self {
            width: size.0,
            height: size.1,
            eye,
            near: f64::from(clip.0),
            far: f64::from(clip.1),
            vp,
            inv: vp.inverse(),
            top_first,
        }
    }

    /// Ray through texel `(x, y)` of the readback (texel centres are `+ 0.5`).
    pub fn pixel_ray(&self, x: f64, y: f64) -> Ray {
        let ndc_x = x / f64::from(self.width) * 2.0 - 1.0;
        let v = y / f64::from(self.height) * 2.0 - 1.0;
        let ndc_y = if self.top_first { -v } else { v };
        let near = self.unproject(DVec3::new(ndc_x, ndc_y, 0.0));
        let far = self.unproject(DVec3::new(ndc_x, ndc_y, 1.0));
        Ray {
            origin: self.eye,
            dir: (far - near).normalize(),
        }
    }

    fn unproject(&self, ndc: DVec3) -> DVec3 {
        let p = self.inv * ndc.extend(1.0);
        p.truncate() / p.w
    }

    /// Projects `p` to texel coordinates and returns them with the view depth
    /// (distance in front of the eye). `None` when `p` is behind the eye.
    pub fn project(&self, p: DVec3) -> Option<(f64, f64, f64)> {
        let clip = self.vp * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        let x = (nx * 0.5 + 0.5) * f64::from(self.width);
        let y = if self.top_first {
            (1.0 - (ny * 0.5 + 0.5)) * f64::from(self.height)
        } else {
            (ny * 0.5 + 0.5) * f64::from(self.height)
        };
        Some((x, y, clip.w))
    }

    /// Whether texel `(x, y)` of a readback of `size` lies on the screen.
    pub fn on_screen(&self, x: f64, y: f64) -> bool {
        x >= 0.0 && y >= 0.0 && x < f64::from(self.width) && y < f64::from(self.height)
    }

    /// Full-resolution texel centre of texel `(x, y)` of a readback of `size`
    /// (which may be a fraction of the frame's resolution).
    pub fn texel_to_pixel(&self, x: u32, y: u32, size: (u32, u32)) -> (f64, f64) {
        (
            (f64::from(x) + 0.5) * f64::from(self.width) / f64::from(size.0),
            (f64::from(y) + 0.5) * f64::from(self.height) / f64::from(size.1),
        )
    }
}

/// Linear view depth of a 0..1 depth-buffer value under this near and far.
pub fn linear_depth(d: f64, near: f64, far: f64) -> f64 {
    near * far / (far - d * (far - near))
}

/// Reference ambient visibility at `p` (normal `n`): cosine-weighted hemisphere
/// rays, a hit at distance `t < radius` weighted `1 - (t / radius)^2`, and
/// visibility `1 - mean weight`.
pub fn ao_visibility(scene: &Scene, p: DVec3, n: DVec3, radius: f64, camera: Option<DVec3>) -> f64 {
    let helper = if n.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let tangent = helper.cross(n).normalize();
    let bitangent = n.cross(tangent);
    let origin = p + n * SURFACE_OFFSET;
    // R2 low-discrepancy sequence: the same 256 directions for every point.
    const A1: f64 = 0.754_877_666_246_692_7;
    const A2: f64 = 0.569_840_290_998_053_3;
    let mut weight_sum = 0.0;
    for i in 0..AO_SAMPLES {
        let k = i as f64 + 0.5;
        let u1 = (0.5 + k * A1).fract();
        let u2 = (0.5 + k * A2).fract();
        let r = u1.sqrt();
        let phi = std::f64::consts::TAU * u2;
        let dir = tangent * (r * phi.cos()) + bitangent * (r * phi.sin()) + n * (1.0 - u1).sqrt();
        let ray = Ray {
            origin,
            dir: dir.normalize(),
        };
        if let Some(hit) = scene.trace(ray, radius) {
            // A screen-space method can only use occluders the camera sees.
            if camera.is_some_and(|eye| !scene.visible_from(eye, hit.position)) {
                continue;
            }
            let k = hit.t / radius;
            weight_sum += 1.0 - k * k;
        }
    }
    1.0 - weight_sum / AO_SAMPLES as f64
}

/// Mirror reflection of `ray` at `hit`: the first hit of the reflected ray, if
/// it hits anything. Visibility from the eye is the caller's test.
pub fn mirror_reflection(scene: &Scene, ray: Ray, hit: &Hit) -> Option<Hit> {
    let n = hit.normal;
    let dir = ray.dir - n * (2.0 * ray.dir.dot(n));
    let reflected = Ray {
        origin: hit.position + n * SURFACE_OFFSET,
        dir: dir.normalize(),
    };
    scene.trace(reflected, f64::INFINITY)
}

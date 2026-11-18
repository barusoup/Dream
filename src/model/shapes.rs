//! Procedural primitives built on the CPU, without glTF.
//!
//! Each builder returns a single-mesh [`Model`] with one node at the identity
//! transform, so it works with [`ModelGpu::upload`](super::render::ModelGpu::upload),
//! [`ModelRenderer::draw_pbr`](super::render::ModelRenderer::draw_pbr) and
//! [`Model::static_bounds`] like a loaded file.
//!
//! Conventions match the glTF path the PBR pipelines expect:
//! - Front faces are counter-clockwise seen from outside the surface (the
//!   pipelines cull back faces with a flipped viewport, see `gfx::pipeline`).
//! - Positions are in metres, normals are unit length.
//! - `uvs` run 0..1 per face for the cuboid and sphere. The plane's UVs are its
//!   world XZ position in metres, so a texture tiles once per metre.
//! - `tangents` are filled. `w` is the bitangent sign such that
//!   `cross(normal, tangent.xyz) * w` points along increasing `v`.
//! - `uvs1` stays empty, as for a glTF primitive without `TEXCOORD_1`.

use super::{Material, Mesh, Model, Node};
use glam::{Quat, Vec3};

/// Horizontal square on the XZ plane centred on the origin, facing +Y.
/// It spans `-half_extent..=half_extent` on both X and Z.
pub fn plane(half_extent: f32, material: Material) -> Model {
    plane_rect(half_extent, half_extent, material)
}

/// Horizontal rectangle on the XZ plane centred on the origin, facing +Y. It
/// spans `-half_x..=half_x` on X and `-half_z..=half_z` on Z.
pub fn plane_rect(half_x: f32, half_z: f32, material: Material) -> Model {
    let mut builder = Builder::default();
    // CCW seen from +Y. UVs are the world XZ position, so the texture tiles
    // once per metre.
    let corners = [
        Vec3::new(-half_x, 0.0, half_z),
        Vec3::new(half_x, 0.0, half_z),
        Vec3::new(half_x, 0.0, -half_z),
        Vec3::new(-half_x, 0.0, -half_z),
    ];
    let uvs = corners.map(|c| [c.x, c.z]);
    builder.quad(corners, uvs, Vec3::Y, Vec3::X, Vec3::Z);
    builder.finish("plane", material)
}

/// Axis-aligned box centred on the origin with the given half extents. Each
/// face has its own four vertices and a flat normal (24 vertices in total).
pub fn cuboid(half_extents: [f32; 3], material: Material) -> Model {
    let h = half_extents;
    // (normal, u, v) with u x v = normal, so each face winds CCW from outside.
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (-Vec3::X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (-Vec3::Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (-Vec3::Z, Vec3::Y, Vec3::X),
    ];
    let mut builder = Builder::default();
    for (n, u, v) in faces {
        let center = n * h[axis(n)];
        let (hu, hv) = (h[axis(u)], h[axis(v)]);
        let corner = |su: f32, sv: f32| center + u * (su * hu) + v * (sv * hv);
        builder.quad(
            [
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            n,
            u,
            v,
        );
    }
    builder.finish("cuboid", material)
}

/// Sphere of `radius` centred on the origin, with `segments` slices around Y
/// and `rings` bands from pole to pole. Counts below 3 segments or 2 rings are
/// raised to those minimums.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32, material: Material) -> Model {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut builder = Builder::default();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let (sp, cp) = (std::f32::consts::PI * v).sin_cos();
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let (st, ct) = (std::f32::consts::TAU * u).sin_cos();
            let normal = Vec3::new(sp * ct, cp, sp * st);
            let du = Vec3::new(-st, 0.0, ct);
            let dv = Vec3::new(cp * ct, -sp, cp * st);
            builder.vertex(normal * radius, normal, [u, v], tangent_of(normal, du, dv));
        }
    }
    // Quad corners (a, b, c, d) run CCW from outside. The pole rows collapse
    // to a point, so each pole row keeps only its non-degenerate triangle.
    let row = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * row + s;
            let b = a + 1;
            let d = (r + 1) * row + s;
            let c = d + 1;
            if r > 0 {
                builder.tri(a, b, c);
            }
            if r + 1 < rings {
                builder.tri(a, c, d);
            }
        }
    }
    builder.finish("uv_sphere", material)
}

/// Index of the axis (0 = X, 1 = Y, 2 = Z) a unit vector points along.
fn axis(v: Vec3) -> usize {
    if v.x.abs() > 0.5 {
        0
    } else if v.y.abs() > 0.5 {
        1
    } else {
        2
    }
}

/// Tangent for a vertex with normal `n`, `t` the direction of increasing `u`
/// and `bt` the direction of increasing `v`.
fn tangent_of(n: Vec3, t: Vec3, bt: Vec3) -> [f32; 4] {
    let w = if n.cross(t).dot(bt) < 0.0 { -1.0 } else { 1.0 };
    [t.x, t.y, t.z, w]
}

#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    tangents: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Builder {
    fn vertex(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], tangent: [f32; 4]) {
        self.positions.push(position.to_array());
        self.normals.push(normal.to_array());
        self.uvs.push(uv);
        self.tangents.push(tangent);
    }

    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    /// Four vertices in CCW order seen from `n`, split into two triangles.
    /// `t` and `bt` are the directions of increasing `u` and `v`.
    fn quad(&mut self, corners: [Vec3; 4], uvs: [[f32; 2]; 4], n: Vec3, t: Vec3, bt: Vec3) {
        let base = self.positions.len() as u32;
        let tangent = tangent_of(n, t, bt);
        for (corner, uv) in corners.into_iter().zip(uvs) {
            self.vertex(corner, n, uv, tangent);
        }
        self.tri(base, base + 1, base + 2);
        self.tri(base, base + 2, base + 3);
    }

    fn finish(self, name: &str, material: Material) -> Model {
        let mesh = Mesh {
            name: name.to_string(),
            positions: self.positions,
            normals: self.normals,
            uvs: self.uvs,
            uvs1: Vec::new(),
            tangents: self.tangents,
            joints: Vec::new(),
            weights: Vec::new(),
            indices: self.indices,
            material: 0,
            node: Some(0),
        };
        Model {
            meshes: vec![mesh],
            materials: vec![material],
            nodes: vec![Node {
                name: name.to_string(),
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                children: Vec::new(),
                mesh: Some(0),
                skin: None,
            }],
            roots: vec![0],
            ..Model::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_unit_normals(model: &Model) {
        for normal in &model.meshes[0].normals {
            let length = Vec3::from(*normal).length();
            assert!((length - 1.0).abs() < 1e-5, "normal {normal:?} not unit");
        }
    }

    /// Every triangle has area and its geometric normal agrees with the
    /// vertex normals, i.e. front faces point outward.
    fn assert_front_faces_outward(model: &Model) {
        let mesh = &model.meshes[0];
        for tri in mesh.indices.chunks_exact(3) {
            let p: Vec<Vec3> = tri
                .iter()
                .map(|&i| Vec3::from(mesh.positions[i as usize]))
                .collect();
            let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
            let smooth = tri.iter().fold(Vec3::ZERO, |acc, &i| {
                acc + Vec3::from(mesh.normals[i as usize])
            });
            assert!(geometric.length() > 1e-9, "degenerate triangle {tri:?}");
            assert!(geometric.dot(smooth) > 0.0, "triangle {tri:?} winds inward");
        }
    }

    fn assert_indices_in_range(model: &Model) {
        let mesh = &model.meshes[0];
        assert!(
            mesh.indices
                .iter()
                .all(|&i| (i as usize) < mesh.positions.len())
        );
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert_eq!(mesh.tangents.len(), mesh.positions.len());
    }

    #[test]
    fn shapes_plane_counts_normals_and_bounds() {
        let model = plane(30.0, Material::default());
        let mesh = &model.meshes[0];
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
        assert_indices_in_range(&model);
        assert_unit_normals(&model);
        assert!(mesh.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]));
        let (min, max) = model.static_bounds();
        assert_eq!(min, Vec3::new(-30.0, 0.0, -30.0));
        assert_eq!(max, Vec3::new(30.0, 0.0, 30.0));
        assert_front_faces_outward(&model);
    }

    #[test]
    fn shapes_cuboid_counts_normals_and_bounds() {
        let model = cuboid([0.5, 1.5, 0.25], Material::default());
        let mesh = &model.meshes[0];
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert_indices_in_range(&model);
        assert_unit_normals(&model);
        // Flat shading: each face's four vertices share one axis-aligned normal.
        for face in mesh.normals.chunks_exact(4) {
            assert!(face.iter().all(|n| *n == face[0]));
            let n = Vec3::from(face[0]);
            assert!(n.abs().max_element() > 0.999 && n.abs().min_element() < 1e-6);
        }
        let (min, max) = model.static_bounds();
        assert!((min - Vec3::new(-0.5, -1.5, -0.25)).length() < 1e-6);
        assert!((max - Vec3::new(0.5, 1.5, 0.25)).length() < 1e-6);
        assert_front_faces_outward(&model);
    }

    #[test]
    fn shapes_uv_sphere_counts_normals_and_bounds() {
        let (segments, rings) = (16, 8);
        let model = uv_sphere(0.5, segments, rings, Material::default());
        let mesh = &model.meshes[0];
        assert_eq!(
            mesh.positions.len(),
            ((rings + 1) * (segments + 1)) as usize
        );
        // Two pole rows lose one triangle per segment each.
        assert_eq!(mesh.indices.len(), (6 * segments * (rings - 1)) as usize);
        assert_indices_in_range(&model);
        assert_unit_normals(&model);
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let expected = Vec3::from(*n) * 0.5;
            assert!((Vec3::from(*p) - expected).length() < 1e-5);
        }
        let (min, max) = model.static_bounds();
        assert!((min - Vec3::splat(-0.5)).length() < 1e-4, "min {min:?}");
        assert!((max - Vec3::splat(0.5)).length() < 1e-4, "max {max:?}");
        assert_front_faces_outward(&model);
    }

    #[test]
    fn shapes_uv_sphere_clamps_tiny_tessellation() {
        let model = uv_sphere(1.0, 0, 0, Material::default());
        assert_eq!(model.meshes[0].positions.len(), 3 * 4);
        assert_front_faces_outward(&model);
    }

    #[test]
    fn shapes_models_have_one_mesh_node_and_root() {
        for model in [
            plane(1.0, Material::default()),
            cuboid([1.0; 3], Material::default()),
            uv_sphere(1.0, 8, 4, Material::default()),
        ] {
            assert_eq!(model.meshes.len(), 1);
            assert_eq!(model.materials.len(), 1);
            assert_eq!(model.meshes[0].node, Some(0));
            assert_eq!(model.roots, vec![0]);
            assert_eq!(model.nodes[0].mesh, Some(0));
        }
    }
}

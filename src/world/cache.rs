//! Stable keys for the on-disk impostor cache (`ImpostorSource::BakeCached`).
//!
//! The key is a 64-bit FNV-1a hash over everything a bake depends on: the
//! static geometry of every mesh, its rest-node placement, the material
//! parameters and texel data, the bake settings, and a format version. FNV is
//! used because it is tiny, dependency-free and stable across Rust releases
//! (`std`'s hashers are not).

use crate::impostor::BakeSettings;
use crate::model::{Model, TextureRef};

/// Bump when the baked format or the bake's meaning changes, so old files are
/// re-baked instead of being loaded with stale assumptions.
pub const CACHE_FORMAT: u32 = 1;

/// 64-bit FNV-1a.
#[derive(Clone, Copy, Debug)]
pub struct Fnv64(u64);

impl Default for Fnv64 {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv64 {
    pub fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    pub fn f32(&mut self, value: f32) {
        self.bytes(&value.to_bits().to_le_bytes());
    }

    /// Length-prefixed, so adjacent strings cannot run together.
    pub fn str(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.bytes(value.as_bytes());
    }

    /// Length-prefixed raw bytes of a plain-old-data slice.
    pub fn pod<T: bytemuck::Pod>(&mut self, values: &[T]) {
        self.u32(values.len() as u32);
        self.bytes(bytemuck::cast_slice(values));
    }

    pub fn finish(self) -> u64 {
        self.0
    }
}

fn hash_texture(hash: &mut Fnv64, texture: &Option<TextureRef>) {
    match texture {
        None => hash.u32(0),
        Some(tref) => {
            hash.u32(1);
            hash.u32(tref.texture.width);
            hash.u32(tref.texture.height);
            hash.pod(&tref.texture.pixels);
            hash.u32(tref.tex_coord);
            hash.str(&format!(
                "{:?}{:?}{:?}{:?}{:?}",
                tref.transform, tref.wrap_s, tref.wrap_t, tref.mag_filter, tref.min_filter
            ));
        }
    }
}

/// Key of `model` baked with `settings`. Equal keys mean equal bake inputs.
pub fn bake_key(model: &Model, settings: &BakeSettings) -> u64 {
    let mut hash = Fnv64::default();
    hash.str(env!("CARGO_PKG_VERSION"));
    hash.u32(CACHE_FORMAT);
    hash.u32(settings.resolution);
    hash.u32(settings.supersample);
    hash.u32(settings.voxel_resolution);
    hash.u32(settings.material_resolution);

    hash.u32(model.meshes.len() as u32);
    for (index, mesh) in model.meshes.iter().enumerate() {
        hash.str(&mesh.name);
        hash.pod(&mesh.positions);
        hash.pod(&mesh.normals);
        hash.pod(&mesh.uvs);
        hash.pod(&mesh.uvs1);
        hash.pod(&mesh.tangents);
        hash.pod(&mesh.joints);
        hash.pod(&mesh.weights);
        hash.pod(&mesh.indices);
        hash.u32(mesh.material as u32);
        hash.u32(mesh.node.map_or(u32::MAX, |n| n as u32));
        hash.pod(&model.mesh_node_transform(index).to_cols_array());
    }

    hash.u32(model.materials.len() as u32);
    for material in &model.materials {
        hash.str(&material.name);
        hash.pod(&material.base_color);
        match &material.base_color_texture {
            None => hash.u32(0),
            Some(texture) => {
                hash.u32(1);
                hash.u32(texture.width);
                hash.u32(texture.height);
                hash.pod(&texture.pixels);
            }
        }
        hash.u32(u32::from(material.double_sided));
        hash_texture(&mut hash, &material.metallic_roughness_texture);
        hash_texture(&mut hash, &material.normal_texture);
        hash.f32(material.normal_scale);
        hash_texture(&mut hash, &material.occlusion_texture);
        hash.f32(material.occlusion_strength);
        hash_texture(&mut hash, &material.emissive_texture);
        hash.pod(&material.emissive_factor);
        hash.f32(material.emissive_strength);
        hash_texture(&mut hash, &material.base_color_map);
        hash.f32(material.metallic_factor);
        hash.f32(material.roughness_factor);
        hash.str(&format!("{:?}", material.alpha_mode));
        hash.f32(material.alpha_cutoff);
        hash.u32(material.unsupported.len() as u32);
        for extension in &material.unsupported {
            hash.str(extension);
        }
    }
    hash.finish()
}

/// File name of the cached bake for `key`.
pub fn file_name(key: u64) -> String {
    format!("impostor-{key:016x}.imp1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_fnv_matches_the_reference_vectors() {
        // Published FNV-1a 64 test vectors.
        let mut empty = Fnv64::default();
        empty.bytes(b"");
        assert_eq!(empty.finish(), 0xcbf2_9ce4_8422_2325);
        let mut a = Fnv64::default();
        a.bytes(b"a");
        assert_eq!(a.finish(), 0xaf63_dc4c_8601_ec8c);
        let mut foobar = Fnv64::default();
        foobar.bytes(b"foobar");
        assert_eq!(foobar.finish(), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn cache_key_changes_with_geometry_and_settings() {
        let model = crate::model::shapes::uv_sphere(1.0, 8, 4, crate::model::Material::default());
        let base = bake_key(&model, &BakeSettings::compact());
        assert_eq!(base, bake_key(&model, &BakeSettings::compact()));
        assert_ne!(base, bake_key(&model, &BakeSettings::default()));
        let mut moved = model.clone();
        moved.meshes[0].positions[0][0] += 1e-3;
        assert_ne!(base, bake_key(&moved, &BakeSettings::compact()));
    }
}

use super::anim::{AnimClip, Channel, Interpolation, Key, Target};
use super::material::SUPPORTED_MATERIAL_EXTENSIONS;
use super::{
    AlphaMode, Material, Mesh, Model, Node, Skin, TexelFilter, TextureRef, UvTransform, Wrap,
};
use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;

impl Model {
    /// Load a `.gltf` or `.glb` from bytes.
    pub fn load_gltf(bytes: &[u8]) -> Result<Arc<Self>, String> {
        let (document, buffers, images) =
            gltf::import_slice(bytes).map_err(|err| format!("import gltf: {err}"))?;
        let mut model = Model::default();

        // Materials first; meshes reference them by index.
        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            let base_color_map = pbr
                .base_color_texture()
                .and_then(|info| texture_ref(&info, &images));
            let base_color_texture = base_color_map.as_ref().map(|map| map.texture.clone());
            let mut unsupported = Vec::new();
            if let Some(extensions) = material.extensions() {
                for key in extensions.keys() {
                    if !SUPPORTED_MATERIAL_EXTENSIONS.contains(&key.as_str()) {
                        unsupported.push(key.clone());
                    }
                }
            }
            unsupported.sort();
            unsupported.dedup();
            let normal_map = material.normal_texture();
            let occlusion_map = material.occlusion_texture();
            model.materials.push(Material {
                name: material.name().unwrap_or_default().to_string(),
                base_color: pbr.base_color_factor(),
                base_color_texture,
                double_sided: material.double_sided(),
                metallic_roughness_texture: pbr
                    .metallic_roughness_texture()
                    .and_then(|info| texture_ref(&info, &images)),
                normal_texture: normal_map.as_ref().and_then(|info| {
                    map_ref(
                        &info.texture(),
                        info.tex_coord(),
                        info.extension_value("KHR_texture_transform"),
                        &images,
                    )
                }),
                normal_scale: normal_map.map(|info| info.scale()).unwrap_or(1.0),
                occlusion_texture: occlusion_map.as_ref().and_then(|info| {
                    map_ref(
                        &info.texture(),
                        info.tex_coord(),
                        info.extension_value("KHR_texture_transform"),
                        &images,
                    )
                }),
                occlusion_strength: occlusion_map.map(|info| info.strength()).unwrap_or(1.0),
                emissive_texture: material
                    .emissive_texture()
                    .and_then(|info| texture_ref(&info, &images)),
                emissive_factor: material.emissive_factor(),
                emissive_strength: material.emissive_strength().unwrap_or(1.0),
                base_color_map,
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                alpha_mode: match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                },
                alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
                unsupported,
            });
        }
        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive
                    .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
                let Some(positions) = reader.read_positions() else {
                    continue; // non-drawable primitive
                };
                let normals = reader
                    .read_normals()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();
                let uvs = reader
                    .read_tex_coords(0)
                    .map(|coords| coords.into_f32().collect())
                    .unwrap_or_default();
                let uvs1 = reader
                    .read_tex_coords(1)
                    .map(|coords| coords.into_f32().collect())
                    .unwrap_or_default();
                let tangents = reader
                    .read_tangents()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();
                let joints = reader
                    .read_joints(0)
                    .map(|joints| joints.into_u16().collect())
                    .unwrap_or_default();
                let weights = reader
                    .read_weights(0)
                    .map(|weights| weights.into_f32().collect())
                    .unwrap_or_default();
                let indices = reader
                    .read_indices()
                    .map(|indices| indices.into_u32().collect())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());
                model.meshes.push(Mesh {
                    name: mesh.name().unwrap_or_default().to_string(),
                    positions: positions.collect(),
                    normals,
                    uvs,
                    uvs1,
                    tangents,
                    joints,
                    weights,
                    indices,
                    material: primitive.material().index().unwrap_or(0),
                    node: None,
                });
            }
        }

        // Meshes are flattened per primitive; a glTF mesh's primitives are
        // consecutive, so each node's `mesh` index maps to a contiguous range.
        for node in document.nodes() {
            let Some(mesh_index) = node.mesh().map(|mesh| mesh.index()) else {
                continue;
            };
            let flattened = first_primitive_index(&document, mesh_index);
            let primitives = document
                .meshes()
                .nth(mesh_index)
                .map(|mesh| mesh.primitives().count())
                .unwrap_or(0);
            for flattened in flattened..flattened + primitives {
                if let Some(mesh) = model.meshes.get_mut(flattened) {
                    if mesh.node.is_none() {
                        mesh.node = Some(node.index());
                    }
                }
            }
        }

        for node in document.nodes() {
            let (translation, rotation, scale) = node.transform().decomposed();
            model.nodes.push(Node {
                name: node.name().unwrap_or_default().to_string(),
                translation: Vec3::from(translation),
                rotation: Quat::from_array(rotation),
                scale: Vec3::from(scale),
                children: node.children().map(|child| child.index()).collect(),
                mesh: node
                    .mesh()
                    .map(|mesh| first_primitive_index(&document, mesh.index())),
                skin: node.skin().map(|skin| skin.index()),
            });
        }

        for skin in document.skins() {
            let reader =
                skin.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
            let inverse_binds = reader
                .read_inverse_bind_matrices()
                .map(|matrices| matrices.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_default();
            model.skins.push(Skin {
                joints: skin.joints().map(|joint| joint.index()).collect(),
                inverse_binds,
            });
        }

        for animation in document.animations() {
            let mut channels = Vec::new();
            let mut duration = 0.0f32;
            for channel in animation.channels() {
                let reader = channel
                    .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
                let Some(times) = reader.read_inputs() else {
                    continue;
                };
                let times: Vec<f32> = times.collect();
                if times.is_empty() {
                    continue;
                }
                duration = duration.max(*times.last().unwrap());
                let Some(outputs) = reader.read_outputs() else {
                    continue;
                };
                let target = match channel.target().property() {
                    gltf::animation::Property::Translation => Target::Translation,
                    gltf::animation::Property::Rotation => Target::Rotation,
                    gltf::animation::Property::Scale => Target::Scale,
                    gltf::animation::Property::MorphTargetWeights => continue,
                };
                let interpolation = match channel.sampler().interpolation() {
                    gltf::animation::Interpolation::Step => Interpolation::Step,
                    gltf::animation::Interpolation::Linear => Interpolation::Linear,
                    gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                };
                let values: Vec<[f32; 4]> = match outputs {
                    gltf::animation::util::ReadOutputs::Translations(iter) => {
                        iter.map(|v| [v[0], v[1], v[2], 0.0]).collect()
                    }
                    gltf::animation::util::ReadOutputs::Rotations(iter) => {
                        iter.into_f32().map(|v| [v[0], v[1], v[2], v[3]]).collect()
                    }
                    gltf::animation::util::ReadOutputs::Scales(iter) => {
                        iter.map(|v| [v[0], v[1], v[2], 0.0]).collect()
                    }
                    gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
                };
                let keys = keys_from_values(&times, &values, interpolation);
                if !keys.is_empty() {
                    channels.push(Channel {
                        node: channel.target().node().index(),
                        target,
                        interpolation,
                        keys,
                    });
                }
            }
            model.clips.push(AnimClip {
                name: animation.name().unwrap_or_default().to_string(),
                duration,
                channels,
            });
        }

        model.roots = document
            .scenes()
            .next()
            .map(|scene| scene.nodes().map(|node| node.index()).collect())
            .unwrap_or_default();
        Ok(Arc::new(model))
    }

    /// Load a `.gltf` or `.glb` file from disk.
    pub fn load_gltf_file(path: impl AsRef<std::path::Path>) -> Result<Arc<Self>, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
        Self::load_gltf(&bytes)
    }
}

/// Map a glTF mesh index to the flattened mesh list (primitives are split).
fn first_primitive_index(document: &gltf::Document, mesh_index: usize) -> usize {
    let mut index = 0;
    for mesh in document.meshes() {
        let primitives = mesh.primitives().count();
        if mesh.index() == mesh_index {
            return index;
        }
        index += primitives;
    }
    0
}

fn keys_from_values(times: &[f32], values: &[[f32; 4]], interpolation: Interpolation) -> Vec<Key> {
    match interpolation {
        Interpolation::CubicSpline => {
            // Values arrive as (in-tangent, value, out-tangent) triples.
            let mut keys = Vec::with_capacity(times.len());
            for (index, &time) in times.iter().enumerate() {
                let base = index * 3;
                if base + 2 >= values.len() {
                    break;
                }
                keys.push(Key {
                    time,
                    value: values[base + 1],
                    in_tangent: values[base],
                    out_tangent: values[base + 2],
                });
            }
            keys
        }
        _ => times
            .iter()
            .zip(values)
            .map(|(&time, &value)| Key {
                time,
                value,
                in_tangent: [0.0; 4],
                out_tangent: [0.0; 4],
            })
            .collect(),
    }
}

/// Decode a `textureInfo` (the kind with `texture()`/`tex_coord()`) into a
/// [`TextureRef`], returning `None` when the image has no decoded pixels.
fn texture_ref(info: &gltf::texture::Info<'_>, images: &[gltf::image::Data]) -> Option<TextureRef> {
    let texture = info.texture();
    let transform = info.texture_transform().map(|t| UvTransform {
        offset: t.offset(),
        rotation: t.rotation(),
        scale: t.scale(),
        tex_coord: t.tex_coord(),
    });
    Some(map_ref_parts(
        &texture,
        info.tex_coord(),
        transform,
        images,
    )?)
}

/// Same as [`texture_ref`] for the `NormalTexture`/`OcclusionTexture` shapes,
/// which carry the transform as an opaque extension value.
fn map_ref(
    texture: &gltf::Texture<'_>,
    tex_coord: u32,
    transform_value: Option<&gltf::json::Value>,
    images: &[gltf::image::Data],
) -> Option<TextureRef> {
    Some(map_ref_parts(
        texture,
        tex_coord,
        transform_value.and_then(parse_uv_transform),
        images,
    )?)
}

fn map_ref_parts(
    texture: &gltf::Texture<'_>,
    tex_coord: u32,
    transform: Option<UvTransform>,
    images: &[gltf::image::Data],
) -> Option<TextureRef> {
    let image = images.get(texture.source().index())?;
    let sampler = texture.sampler();
    let wrap = |mode| match mode {
        gltf::texture::WrappingMode::ClampToEdge => Wrap::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => Wrap::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => Wrap::Repeat,
    };
    let mag = |f| match f {
        Some(gltf::texture::MagFilter::Nearest) => TexelFilter::Nearest,
        _ => TexelFilter::Linear,
    };
    let min = |f| match f {
        Some(gltf::texture::MinFilter::Nearest)
        | Some(gltf::texture::MinFilter::NearestMipmapNearest)
        | Some(gltf::texture::MinFilter::NearestMipmapLinear) => TexelFilter::Nearest,
        _ => TexelFilter::Linear,
    };
    Some(TextureRef {
        texture: rgba8_from_gltf(image),
        tex_coord,
        transform,
        wrap_s: wrap(sampler.wrap_s()),
        wrap_t: wrap(sampler.wrap_t()),
        mag_filter: mag(sampler.mag_filter()),
        min_filter: min(sampler.min_filter()),
    })
}

/// Parse a `KHR_texture_transform` extension blob found on a textureInfo.
fn parse_uv_transform(value: &gltf::json::Value) -> Option<UvTransform> {
    let f2 = |v: &gltf::json::Value| -> Option<[f32; 2]> {
        let arr = v.as_array()?;
        Some([arr.first()?.as_f64()? as f32, arr.get(1)?.as_f64()? as f32])
    };
    Some(UvTransform {
        offset: value.get("offset").and_then(f2).unwrap_or([0.0, 0.0]),
        rotation: value
            .get("rotation")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        scale: value.get("scale").and_then(f2).unwrap_or([1.0, 1.0]),
        tex_coord: value
            .get("texCoord")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    })
}

fn rgba8_from_gltf(image: &gltf::image::Data) -> crate::texture::DecodedTexture {
    use gltf::image::Format;
    let (width, height) = (image.width, image.height);
    let pixels = match image.format {
        Format::R8G8B8A8 => image.pixels.clone(),
        Format::R8G8B8 => image
            .pixels
            .chunks_exact(3)
            .flat_map(|px| [px[0], px[1], px[2], 255])
            .collect(),
        Format::R8G8 => image
            .pixels
            .chunks_exact(2)
            .flat_map(|px| [px[0], px[1], px[0], 255])
            .collect(),
        Format::R8 => image
            .pixels
            .iter()
            .flat_map(|&px| [px, px, px, 255])
            .collect(),
        _ => image.pixels.clone(),
    };
    crate::texture::DecodedTexture {
        pixels,
        width,
        height,
    }
}

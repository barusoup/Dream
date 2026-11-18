//! Hardware ray tracing: one bottom-level acceleration structure (BLAS) per
//! static mesh, and one top-level structure (TLAS) per frame slot, rebuilt in
//! that slot's command buffer every frame from the scene's instances.
//!
//! Compute passes read the TLAS through ray queries (`rt_*.wgsl`); no
//! ray-tracing pipelines are used. Everything here exists only when
//! [`GpuContext::ray_tracing_supported`] is true. Without it, the screen-space
//! passes run unchanged and no RT buffer or structure is created.
//!
//! Per-slot ownership: a slot's TLAS, instance, scratch and material buffers
//! are written and built only while that slot's fence has signalled, so the
//! GPU never reads them while they change.

use super::vulkan::RtDevice;
use super::{GpuBuffer, GpuContext, GpuImage};
use ash::vk;
use glam::Mat4;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Largest instance count the TLAS is sized for (one instance per draw of a
/// model's chosen level). Above it the surplus instances are not traced.
const MAX_INSTANCES: usize = 1 << 20;
/// Bytes of one `VkAccelerationStructureInstanceKHR`.
const INSTANCE_BYTES: usize = 64;
/// Bytes of one [`RtMaterial`] as the shaders read it (three `vec4`s).
const MATERIAL_BYTES: usize = 48;

/// Shading values of a mesh, used when a ray hits it: the rays do not sample
/// textures. Built from the glTF material factors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtMaterial {
    /// Base colour, linear RGB.
    pub base_color: [f32; 3],
    /// Perceptual roughness, `0..=1`.
    pub roughness: f32,
    /// Emitted radiance, linear RGB (factor times strength).
    pub emissive: [f32; 3],
    /// Metalness, `0..=1`.
    pub metallic: f32,
}

impl Default for RtMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8],
            roughness: 0.5,
            emissive: [0.0; 3],
            metallic: 0.0,
        }
    }
}

impl RtMaterial {
    /// The GPU record: `base` (rgb), `emissive` (rgb), `params` (roughness, metallic).
    fn bytes(&self) -> [u8; MATERIAL_BYTES] {
        let mut out = [0.0_f32; 12];
        out[0..3].copy_from_slice(&self.base_color);
        out[4..7].copy_from_slice(&self.emissive);
        out[8] = self.roughness;
        out[9] = self.metallic;
        let mut bytes = [0u8; MATERIAL_BYTES];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip(out) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// A mesh registered with a [`RayTracingScene`]. Valid for that scene only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtMeshHandle {
    scene: u64,
    index: u32,
}

/// One placed copy of a registered mesh for the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtInstance {
    /// The mesh to place.
    pub mesh: RtMeshHandle,
    /// Object to world.
    pub transform: Mat4,
}

/// Descriptor inputs of the ray-traced passes for one frame slot.
pub(crate) struct RtFrameBinds<'a> {
    pub tlas: vk::AccelerationStructureKHR,
    pub materials: &'a GpuBuffer,
    pub env_view: vk::ImageView,
    pub env_sampler: vk::Sampler,
}

/// A bottom-level structure: one static mesh.
struct Blas {
    ctx: Arc<GpuContext>,
    handle: vk::AccelerationStructureKHR,
    address: u64,
    _buffer: GpuBuffer,
}

impl Drop for Blas {
    fn drop(&mut self) {
        // In-flight frames may still trace this structure: wait before it goes.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            if let Some(rt) = self.ctx.rt_device() {
                rt.accel.destroy_acceleration_structure(self.handle, None);
            }
        }
    }
}

/// A registered mesh; its shading record lives in the scene's material bytes.
struct MeshEntry {
    blas: Blas,
}

/// Per-slot TLAS, instance, scratch and material storage. Grown on demand and
/// reused, never shrunk.
#[derive(Default)]
struct TlasSlot {
    handle: Option<vk::AccelerationStructureKHR>,
    tlas_buffer: Option<GpuBuffer>,
    /// Instance capacity the TLAS and scratch were sized for.
    capacity: usize,
    instances: Option<GpuBuffer>,
    scratch: Option<GpuBuffer>,
    materials: Option<GpuBuffer>,
    material_capacity: usize,
    material_count: usize,
}

/// The ray-traced scene: registered meshes, the instances placed this frame,
/// and the per-slot TLAS. Create it through
/// [`super::Renderer::ray_tracing_scene_mut`]. Registrations are cached until
/// [`super::Renderer::clear_ray_tracing_scene`] or RT is disabled.
pub struct RayTracingScene {
    ctx: Arc<GpuContext>,
    id: u64,
    meshes: Vec<MeshEntry>,
    staged: Vec<RtInstance>,
    slots: Vec<TlasSlot>,
    env_sampler: vk::Sampler,
    dummy_env: GpuImage,
    instance_bytes: Vec<u8>,
    material_bytes: Vec<u8>,
    /// Instances traced in the last recorded build, and that build's CPU time.
    last_instances: usize,
    last_build_us: f32,
}

impl RayTracingScene {
    /// An empty scene on `ctx`. Only call this on a device that supports ray
    /// tracing (see [`GpuContext::ray_tracing_supported`]).
    pub(crate) fn new(ctx: &Arc<GpuContext>) -> Self {
        debug_assert!(ctx.ray_tracing_supported());
        let env_sampler = {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            unsafe { ctx.device().create_sampler(&info, None) }.expect("ray tracing env sampler")
        };
        let dummy_env = super::post::create_general_image(
            ctx,
            "ray tracing dummy environment",
            (1, 1),
            vk::Format::R16G16B16A16_SFLOAT,
        );
        Self {
            ctx: Arc::clone(ctx),
            id: {
                static NEXT_SCENE: AtomicU64 = AtomicU64::new(1);
                NEXT_SCENE.fetch_add(1, Ordering::Relaxed)
            },
            meshes: Vec::new(),
            staged: Vec::new(),
            slots: (0..ctx.max_frames_in_flight())
                .map(|_| TlasSlot::default())
                .collect(),
            env_sampler,
            dummy_env,
            instance_bytes: Vec::new(),
            material_bytes: Vec::new(),
            last_instances: 0,
            last_build_us: 0.0,
        }
    }

    /// Register a static triangle mesh: a bottom-level structure is built now,
    /// synchronously. `vertices` must hold the position as the first three
    /// floats of each `vertex_stride`-byte vertex; `indices` are `u32`. Both
    /// buffers need the usage from [`GpuContext::rt_geometry_usage`]. Returns
    /// `None` for empty meshes, invalid strides or counts exceeding the buffers.
    /// Index values must be less than `vertex_count`.
    pub fn add_mesh(
        &mut self,
        vertices: &GpuBuffer,
        vertex_stride: u32,
        vertex_count: u32,
        indices: &GpuBuffer,
        index_count: u32,
        material: RtMaterial,
    ) -> Option<RtMeshHandle> {
        if vertex_count == 0
            || vertex_stride < 12
            || !vertex_stride.is_multiple_of(4)
            || u64::from(vertex_count) * u64::from(vertex_stride) > vertices.size
            || u64::from(index_count) * 4 > indices.size
            || index_count < 3
            || !index_count.is_multiple_of(3)
        {
            return None;
        }
        let blas = build_blas(
            &self.ctx,
            vertices,
            vertex_stride,
            vertex_count,
            indices,
            index_count,
        );
        let index = self.meshes.len() as u32;
        self.material_bytes.extend_from_slice(&material.bytes());
        self.meshes.push(MeshEntry { blas });
        Some(RtMeshHandle {
            scene: self.id,
            index,
        })
    }

    /// The instances for the next frame (replaces the previous list). Instances
    /// of unknown handles, non-affine matrices and singular transforms are skipped.
    pub fn set_instances(&mut self, instances: &[RtInstance]) {
        self.staged.clear();
        self.staged.extend_from_slice(instances);
    }

    /// Instances staged for the next frame.
    pub fn instance_count(&self) -> usize {
        self.staged.len()
    }

    /// Instances traced by the last recorded TLAS build.
    pub fn last_instances(&self) -> usize {
        self.last_instances
    }

    /// CPU time of the last recorded TLAS build, in microseconds (recording
    /// and host writes; the GPU time is the `TLAS build` pass timer).
    pub fn last_build_us(&self) -> f32 {
        self.last_build_us
    }

    /// Identity used to invalidate World registrations after a scene reset.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Record the TLAS build for `slot` at the current point of `cmd`, from
    /// the staged instances. Call only once the slot's previous frame is done.
    pub(crate) fn record_build(&mut self, cmd: vk::CommandBuffer, slot: usize) {
        let started = Instant::now();
        let rt = self.ctx.rt_device().expect("ray tracing device");
        let slot_index = slot % self.slots.len();
        let instance_count = self.staged.len().min(MAX_INSTANCES);
        let mesh_count = self.meshes.len();

        // Instance records: 3x4 row-major transform, custom index = mesh index
        // (the material slot), mask 0xff, force-opaque, the BLAS address.
        self.instance_bytes.clear();
        let mut placed = 0usize;
        for instance in &self.staged[..instance_count] {
            let m = instance.mesh.index as usize;
            if instance.mesh.scene != self.id
                || m >= mesh_count
                || !valid_transform(instance.transform)
            {
                continue;
            }
            let t = instance.transform;
            for row in 0..3 {
                for col in 0..4 {
                    let v = t.col(col)[row];
                    self.instance_bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            let custom = (m as u32) & 0x00ff_ffff;
            let mask = 0xffu32 << 24;
            self.instance_bytes
                .extend_from_slice(&(custom | mask).to_le_bytes());
            let sbt_and_flags = 0x04u32 << 24; // VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR
            self.instance_bytes
                .extend_from_slice(&sbt_and_flags.to_le_bytes());
            self.instance_bytes
                .extend_from_slice(&self.meshes[m].blas.address.to_le_bytes());
            placed += 1;
        }
        debug_assert_eq!(self.instance_bytes.len(), placed * INSTANCE_BYTES);
        if placed == 0 {
            self.last_instances = 0;
            self.last_build_us = started.elapsed().as_secs_f32() * 1e6;
            return;
        }

        let s = &mut self.slots[slot_index];
        // Vulkan does not permit a TLAS build with zero instances. Leave a
        // previously built slot intact; it is not bound while the scene is
        // empty and will be rebuilt before its next use.
        if placed == 0 {
            self.last_instances = 0;
            self.last_build_us = started.elapsed().as_secs_f32() * 1e6;
            return;
        }
        let wanted = placed.max(64).next_power_of_two();
        if s.capacity < wanted {
            if let Some(handle) = s.handle.take() {
                unsafe { rt.accel.destroy_acceleration_structure(handle, None) };
            }
            s.tlas_buffer = None;
            s.scratch = None;
            s.instances = None;
            let sizes = tlas_sizes(rt, instance_geometry(0), wanted as u32);
            let tlas_buffer = self.ctx.create_rt_buffer(
                "ray tracing TLAS",
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                false,
            );
            let info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(tlas_buffer.buffer)
                .offset(0)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
            let handle = unsafe { rt.accel.create_acceleration_structure(&info, None) }
                .expect("create TLAS");
            s.handle = Some(handle);
            s.tlas_buffer = Some(tlas_buffer);
            s.scratch = Some(self.ctx.create_rt_buffer(
                "ray tracing TLAS scratch",
                sizes.build_scratch_size + rt.scratch_alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                false,
            ));
            s.instances = Some(self.ctx.create_rt_buffer(
                "ray tracing instances",
                (wanted * INSTANCE_BYTES) as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                true,
            ));
            s.capacity = wanted;
        }
        if s.material_capacity < mesh_count.max(1) {
            let cap = mesh_count.max(16).next_power_of_two();
            s.materials = Some(self.ctx.create_host_storage_buffer(
                "ray tracing materials",
                (cap * MATERIAL_BYTES) as u64,
            ));
            s.material_capacity = cap;
        }
        if placed > 0 {
            s.instances
                .as_mut()
                .expect("instance buffer")
                .write(0, &self.instance_bytes);
        }
        if mesh_count > 0 && s.material_count != mesh_count {
            s.materials
                .as_mut()
                .expect("material buffer")
                .write(0, &self.material_bytes);
            s.material_count = mesh_count;
        }

        let instances_addr = self
            .ctx
            .buffer_address(s.instances.as_ref().expect("instance buffer").buffer);
        let scratch = s.scratch.as_ref().expect("scratch buffer");
        let scratch_addr = align_up(
            self.ctx.buffer_address(scratch.buffer),
            rt.scratch_alignment,
        );
        let geometries = [instance_geometry(instances_addr)];
        let handle = s.handle.expect("TLAS handle");
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(handle)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });
        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(placed as u32);
        let ranges = [range];
        unsafe {
            let device = self.ctx.device();
            // Host writes to the instance and material buffers, then the build
            // and the shader reads of the materials.
            let host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[host],
                &[],
                &[],
            );
            rt.accel
                .cmd_build_acceleration_structures(cmd, &[info], &[&ranges]);
            let built = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[built],
                &[],
                &[],
            );
        }
        self.last_instances = placed;
        self.last_build_us = started.elapsed().as_secs_f32() * 1e6;
    }

    /// Descriptor inputs of the ray-traced passes for `slot`. `env` is the sky
    /// environment map of the previous frame, when there is one.
    pub(crate) fn binds(
        &self,
        slot: usize,
        env: Option<(u64, vk::ImageView)>,
    ) -> Option<RtFrameBinds<'_>> {
        if self.last_instances == 0 {
            return None;
        }
        let s = &self.slots[slot % self.slots.len()];
        let handle = s.handle?;
        let materials = s.materials.as_ref()?;
        let env_view = env.map_or(self.dummy_env.view, |(_, view)| view);
        Some(RtFrameBinds {
            tlas: handle,
            materials,
            env_view,
            env_sampler: self.env_sampler,
        })
    }
}

impl Drop for RayTracingScene {
    fn drop(&mut self) {
        // In-flight frames may still trace these structures.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            if let Some(rt) = self.ctx.rt_device() {
                for slot in &mut self.slots {
                    if let Some(handle) = slot.handle.take() {
                        rt.accel.destroy_acceleration_structure(handle, None);
                    }
                }
            }
            self.ctx.device().destroy_sampler(self.env_sampler, None);
        }
    }
}

/// Geometry description of a TLAS whose instance records are at `address`.
fn instance_geometry(address: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: address,
                }),
        })
}

/// Sizes for a TLAS of up to `max_instances` instances described by `geometry`.
fn tlas_sizes(
    rt: &RtDevice,
    geometry: vk::AccelerationStructureGeometryKHR<'_>,
    max_instances: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let geometries = [geometry];
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    query_sizes(rt, &info, &[max_instances])
}

/// Size query of `info` for the given primitive counts.
fn query_sizes(
    rt: &RtDevice,
    info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    counts: &[u32],
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        rt.accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            info,
            counts,
            &mut sizes,
        );
    }
    sizes
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

// A zero scale can hide raster geometry, but Vulkan instances must be invertible.
fn valid_transform(transform: Mat4) -> bool {
    let determinant = transform.determinant();
    transform.is_finite()
        && determinant.is_finite()
        && determinant != 0.0
        && transform.row(3).abs_diff_eq(glam::Vec4::W, 1e-6)
}

/// Build the bottom-level structure of an indexed triangle list, to completion.
fn build_blas(
    ctx: &Arc<GpuContext>,
    vertices: &GpuBuffer,
    vertex_stride: u32,
    vertex_count: u32,
    indices: &GpuBuffer,
    index_count: u32,
) -> Blas {
    let rt = ctx.rt_device().expect("ray tracing device");
    let vertex_addr = ctx.buffer_address(vertices.buffer);
    let index_addr = ctx.buffer_address(indices.buffer);
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_addr,
        })
        .vertex_stride(vertex_stride as u64)
        .max_vertex(vertex_count - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_addr,
        });
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles });
    let geometries = [geometry];
    let primitive_count = index_count / 3;
    let base_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        // Position-fetch ray queries are used to recover hit normals. Vulkan
        // requires this build flag before the geometry data can be fetched.
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
        )
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let sizes = query_sizes(rt, &base_info, &[primitive_count]);
    let buffer = ctx.create_rt_buffer(
        "ray tracing BLAS",
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    );
    let create = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer.buffer)
        .offset(0)
        .size(sizes.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let handle =
        unsafe { rt.accel.create_acceleration_structure(&create, None) }.expect("create BLAS");
    let scratch = ctx.create_rt_buffer(
        "ray tracing BLAS scratch",
        sizes.build_scratch_size + rt.scratch_alignment,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    );
    let scratch_addr = align_up(ctx.buffer_address(scratch.buffer), rt.scratch_alignment);
    let info =
        base_info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });
    let range =
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
    let ranges = [range];
    ctx.one_shot(|cmd| unsafe {
        rt.accel
            .cmd_build_acceleration_structures(cmd, &[info], &[&ranges]);
        let built = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        ctx.device().cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[built],
            &[],
            &[],
        );
    });
    let address = unsafe {
        rt.accel.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(handle),
        )
    };
    Blas {
        ctx: Arc::clone(ctx),
        handle,
        address,
        _buffer: buffer,
    }
}

#[cfg(test)]
mod tests {
    use super::valid_transform;
    use glam::{Mat4, Vec3};

    #[test]
    fn rt_transforms_reject_noninvertible_or_projective_instances() {
        assert!(valid_transform(Mat4::IDENTITY));
        assert!(valid_transform(Mat4::from_scale(Vec3::new(-2.0, 0.5, 3.0))));
        assert!(!valid_transform(Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0))));
        assert!(!valid_transform(Mat4::from_translation(Vec3::splat(
            f32::NAN
        ))));
        assert!(!valid_transform(Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)));
    }
}

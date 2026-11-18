//! Compute and fullscreen post-pass layer for the screen-space passes that
//! follow the scene.
//!
//! A [`PostPass`] is one WGSL entry point (a compute `main`, or a fullscreen
//! `vs_main`/`fs_main` pair) with an ordered list of bindings ([`BindKind`];
//! binding `i` is the `i`-th kind). It owns its descriptor set layout, pipeline
//! layout, pipeline, descriptor pool and one set per frame slot, so a pass is
//! described by its shader and its bindings instead of a hand-built set of
//! Vulkan objects (FSR predates it and keeps its own).
//!
//! Each frame slot's set is rewritten only while that slot is idle: the
//! renderer waits a slot's fence before it records into the slot, so writes
//! through [`PostPass::bind_sampled`] and friends never race a submission that
//! reads the set. All images bound by these passes live in `GENERAL` layout.

use super::pipeline::create_pipeline_layout;
use super::vulkan::{self, GpuBuffer, GpuContext, GpuImage};
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

/// One binding of a [`PostPass`], in the order the kinds are listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindKind {
    /// Read-only uniform buffer (`var<uniform>`).
    Uniform,
    /// Sampled image (`texture_2d` or `texture_depth_2d`), read by `textureLoad`
    /// or through a sampler.
    Sampled,
    /// Storage image written by the shader (`texture_storage_2d<format, write>`).
    /// The format must match the WGSL declaration and the image's own format.
    Storage(vk::Format),
    /// Sampled 3D image (`texture_3d<f32>`), read by `textureLoad` or a sampler.
    Sampled3d,
    /// Storage 3D image (`texture_storage_3d<format, write>`).
    Storage3d(vk::Format),
    /// Filtering sampler (`sampler`).
    Sampler,
    /// Comparison sampler (`sampler_comparison`), for depth-array shadow lookups.
    ComparisonSampler,
    /// Top-level acceleration structure (`acceleration_structure`), read by ray
    /// queries. Only on devices with ray tracing (see `rt.rs`).
    Accel,
    /// Read-only storage buffer (`var<storage, read>`).
    StorageBuffer,
}

impl BindKind {
    fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            BindKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            BindKind::Sampled | BindKind::Sampled3d => vk::DescriptorType::SAMPLED_IMAGE,
            BindKind::Storage(_) | BindKind::Storage3d(_) => vk::DescriptorType::STORAGE_IMAGE,
            BindKind::Sampler | BindKind::ComparisonSampler => vk::DescriptorType::SAMPLER,
            BindKind::Accel => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            BindKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

/// Attachment setup of a fullscreen pass built by [`PostPass::new_fullscreen`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FullscreenTarget {
    /// Colour attachment formats, in attachment order.
    pub colour: &'static [vk::Format],
    /// Blend every colour attachment with `src * ONE + dst * SRC_ALPHA` (colour)
    /// and `src * ONE + dst * ZERO` (alpha). Replace when false.
    pub blend: bool,
    /// Depth format of a read-only depth attachment tested `EQUAL` (no writes).
    pub depth_equal: Option<vk::Format>,
}

/// A compute or fullscreen pass over a fixed list of bindings; see the module docs.
pub(crate) struct PostPass {
    ctx: Arc<GpuContext>,
    kinds: Vec<BindKind>,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bind_point: vk::PipelineBindPoint,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl PostPass {
    /// Compile `wgsl` (through the build-time precompiled table when it is a
    /// built-in shader) and build the pass around compute entry point `entry`.
    pub(crate) fn new(
        ctx: &Arc<GpuContext>,
        label: &str,
        wgsl: &str,
        entry: &CStr,
        kinds: &[BindKind],
    ) -> Self {
        let (dsl, layout) = Self::layouts(ctx, label, kinds, vk::ShaderStageFlags::COMPUTE);
        let module = ctx.compile_shader(wgsl, label);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = unsafe {
            ctx.device()
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        }
        .unwrap_or_else(|(_, err)| panic!("{label}: compute pipeline: {err:?}"))[0];
        unsafe { ctx.device().destroy_shader_module(module, None) };
        Self::finish(
            ctx,
            label,
            kinds,
            dsl,
            layout,
            pipeline,
            vk::PipelineBindPoint::COMPUTE,
        )
    }

    /// A fullscreen graphics pass: vertex entry `vs_main` (a fullscreen
    /// triangle) and fragment entry `fs_main`, drawn into the attachments of
    /// `target`. The caller begins the rendering scope and sets the viewport and
    /// scissor; [`Self::draw_fullscreen`] records the draw.
    pub(crate) fn new_fullscreen(
        ctx: &Arc<GpuContext>,
        label: &str,
        wgsl: &str,
        kinds: &[BindKind],
        target: FullscreenTarget,
    ) -> Self {
        let stage = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let (dsl, layout) = Self::layouts(ctx, label, kinds, stage);
        let device = ctx.device();
        let module = ctx.compile_shader(wgsl, label);
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(c"fs_main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
        if target.depth_equal.is_some() {
            depth_state = depth_state
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::EQUAL);
        }
        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = target
            .colour
            .iter()
            .map(|_| {
                let state = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                if target.blend {
                    state
                        .blend_enable(true)
                        .src_color_blend_factor(vk::BlendFactor::ONE)
                        .dst_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                        .color_blend_op(vk::BlendOp::ADD)
                        .src_alpha_blend_factor(vk::BlendFactor::ONE)
                        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                        .alpha_blend_op(vk::BlendOp::ADD)
                } else {
                    state.blend_enable(false)
                }
            })
            .collect();
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(target.colour)
            .depth_attachment_format(target.depth_equal.unwrap_or(vk::Format::UNDEFINED));
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_state)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering)];
        let pipeline =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) }
                .unwrap_or_else(|(_, err)| panic!("{label}: graphics pipeline: {err:?}"))[0];
        unsafe { device.destroy_shader_module(module, None) };
        Self::finish(
            ctx,
            label,
            kinds,
            dsl,
            layout,
            pipeline,
            vk::PipelineBindPoint::GRAPHICS,
        )
    }

    fn layouts(
        ctx: &Arc<GpuContext>,
        label: &str,
        kinds: &[BindKind],
        stage: vk::ShaderStageFlags,
    ) -> (vk::DescriptorSetLayout, vk::PipelineLayout) {
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(index as u32)
                    .descriptor_type(kind.descriptor_type())
                    .descriptor_count(1)
                    .stage_flags(stage)
            })
            .collect();
        let dsl = unsafe {
            ctx.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .unwrap_or_else(|err| panic!("{label}: descriptor set layout: {err:?}"));
        (dsl, create_pipeline_layout(ctx, label, &[dsl], &[]))
    }

    fn finish(
        ctx: &Arc<GpuContext>,
        label: &str,
        kinds: &[BindKind],
        dsl: vk::DescriptorSetLayout,
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
        bind_point: vk::PipelineBindPoint,
    ) -> Self {
        let device = ctx.device();
        let slots = ctx.max_frames_in_flight() as u32;
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for kind in kinds {
            let ty = kind.descriptor_type();
            match sizes.iter_mut().find(|size| size.ty == ty) {
                Some(size) => size.descriptor_count += slots,
                None => sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(ty)
                        .descriptor_count(slots),
                ),
            }
        }
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(slots)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .unwrap_or_else(|err| panic!("{label}: descriptor pool: {err:?}"));
        let layouts = vec![dsl; slots as usize];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .unwrap_or_else(|err| panic!("{label}: descriptor sets: {err:?}"));

        Self {
            ctx: Arc::clone(ctx),
            kinds: kinds.to_vec(),
            dsl,
            layout,
            pipeline,
            bind_point,
            pool,
            sets,
        }
    }

    fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.sets[slot % self.sets.len()]
    }

    fn write(&self, slot: usize, binding: u32, write: vk::WriteDescriptorSet<'_>) {
        debug_assert!(
            (binding as usize) < self.kinds.len(),
            "binding out of range"
        );
        let write = write.dst_set(self.set(slot)).dst_binding(binding);
        unsafe { self.ctx.device().update_descriptor_sets(&[write], &[]) };
    }

    /// Bind a sampled image (in `GENERAL` layout) to `binding` for `slot`.
    pub(crate) fn bind_sampled(&self, slot: usize, binding: u32, view: vk::ImageView) {
        self.bind_sampled_in(slot, binding, view, vk::ImageLayout::GENERAL);
    }

    /// Bind a sampled image in `layout` to `binding` for `slot`.
    pub(crate) fn bind_sampled_in(
        &self,
        slot: usize,
        binding: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        debug_assert!(matches!(
            self.kinds[binding as usize],
            BindKind::Sampled | BindKind::Sampled3d
        ));
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)];
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&info),
        );
    }

    /// Bind a storage image (in `GENERAL` layout) to `binding` for `slot`.
    /// `format` must be the format the binding was declared with.
    pub(crate) fn bind_storage(
        &self,
        slot: usize,
        binding: u32,
        view: vk::ImageView,
        format: vk::Format,
    ) {
        debug_assert!(matches!(
            self.kinds[binding as usize],
            BindKind::Storage(f) | BindKind::Storage3d(f) if f == format
        ));
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&info),
        );
    }

    /// Bind a filtering or comparison sampler to `binding` for `slot`.
    pub(crate) fn bind_sampler(&self, slot: usize, binding: u32, sampler: vk::Sampler) {
        let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&info),
        );
    }

    /// Bind a top-level acceleration structure to `binding` for `slot`.
    pub(crate) fn bind_accel(&self, slot: usize, binding: u32, tlas: vk::AccelerationStructureKHR) {
        debug_assert_eq!(self.kinds[binding as usize], BindKind::Accel);
        let handles = [tlas];
        let mut accel = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&handles);
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut accel),
        );
    }

    /// Bind a whole storage buffer to `binding` for `slot`.
    pub(crate) fn bind_storage_buffer(&self, slot: usize, binding: u32, buffer: &GpuBuffer) {
        debug_assert_eq!(self.kinds[binding as usize], BindKind::StorageBuffer);
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info),
        );
    }

    /// Bind a uniform buffer (its first `size` bytes) to `binding` for `slot`.
    pub(crate) fn bind_uniform(&self, slot: usize, binding: u32, buffer: &GpuBuffer, size: u64) {
        debug_assert_eq!(self.kinds[binding as usize], BindKind::Uniform);
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(size)];
        self.write(
            slot,
            binding,
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info),
        );
    }

    /// Record one dispatch covering `width` x `height` with 8x8 groups for
    /// `slot`, between a shader-write to shader-read barrier before it and one
    /// after it. Bind every binding for the slot first.
    pub(crate) fn dispatch(&self, cmd: vk::CommandBuffer, slot: usize, width: u32, height: u32) {
        self.dispatch_layers(cmd, slot, width, height, 1);
    }

    /// Record one dispatch covering `width` x `height` with 8x8 groups for
    /// `slot`, skipping the pre-barrier when the immediately preceding command
    /// was already a compute dispatch with a post-barrier.
    pub(crate) fn dispatch_after_compute(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        width: u32,
        height: u32,
    ) {
        self.dispatch_layers_inner(cmd, slot, width, height, 1, false);
    }

    /// [`Self::dispatch`] with `layers` workgroup layers along z, for 3D volumes
    /// (the shader uses `workgroup_size(8, 8, 1)`; `gid.z` is the layer).
    pub(crate) fn dispatch_layers(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        width: u32,
        height: u32,
        layers: u32,
    ) {
        self.dispatch_layers_inner(cmd, slot, width, height, layers, true);
    }

    fn dispatch_layers_inner(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        width: u32,
        height: u32,
        layers: u32,
        pre_barrier: bool,
    ) {
        let device = self.ctx.device();
        if pre_barrier {
            barrier(
                device,
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set(slot)],
                &[],
            );
            device.cmd_dispatch(
                cmd,
                width.max(1).div_ceil(8),
                height.max(1).div_ceil(8),
                layers.max(1),
            );
        }
        barrier(
            device,
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::TRANSFER,
        );
    }

    /// Record the fullscreen triangle of a [`Self::new_fullscreen`] pass for
    /// `slot`, inside a rendering scope the caller began.
    pub(crate) fn draw_fullscreen(&self, cmd: vk::CommandBuffer, slot: usize) {
        unsafe {
            let device = self.ctx.device();
            device.cmd_bind_pipeline(cmd, self.bind_point, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                self.bind_point,
                self.layout,
                0,
                &[self.set(slot)],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}

/// Shader-write to shader-read (and write) memory barrier, as FSR's `barrier`.
fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let memory = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
    }
}

impl Drop for PostPass {
    fn drop(&mut self) {
        let device = self.ctx.device();
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

/// One host-visible uniform buffer per frame slot, `size` bytes each. Write a
/// slot's block with [`GpuBuffer::write`] before recording the slot.
pub(crate) fn host_uniform_slots(ctx: &Arc<GpuContext>, label: &str, size: u64) -> Vec<GpuBuffer> {
    (0..ctx.max_frames_in_flight())
        .map(|slot| ctx.create_host_buffer(&format!("{label} {slot}"), size))
        .collect()
}

/// A `GENERAL`-layout image that is both sampled and storage. Its contents are
/// undefined until written; the layout transition runs in a one-shot submission.
pub(crate) fn create_general_image(
    ctx: &Arc<GpuContext>,
    label: &str,
    (width, height): (u32, u32),
    format: vk::Format,
) -> GpuImage {
    // TRANSFER_SRC lets diagnostics read the image back (`capture_*_rgba8`).
    let image = ctx.create_image(
        label,
        width,
        height,
        format,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    general_layout(ctx, image.image, vk::ImageAspectFlags::COLOR);
    image
}

/// A `GENERAL`-layout 3D image (`width` x `height` x `depth`), sampled and
/// storage, for the fog volume.
pub(crate) fn create_general_volume(
    ctx: &Arc<GpuContext>,
    label: &str,
    (width, height, depth): (u32, u32, u32),
    format: vk::Format,
) -> GpuImage {
    let image = ctx.create_volume(
        label,
        (width, height, depth),
        format,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    general_layout(ctx, image.image, vk::ImageAspectFlags::COLOR);
    image
}

/// Move a freshly created image of any aspect into `GENERAL` layout.
pub(crate) fn general_layout(
    ctx: &Arc<GpuContext>,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
) {
    ctx.one_shot(|cmd| unsafe {
        vulkan::image_barrier(
            ctx.device(),
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            aspect,
        );
    });
}

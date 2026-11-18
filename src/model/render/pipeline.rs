use super::{ModelVertex, PbrVertex, SkinnedVertex};
use crate::gfx::vulkan::GpuContext;
use ash::vk;

/// Vertex buffers and attributes of the simple model path: static or skinned.
fn model_vertex_input(
    skinned: bool,
) -> (
    Vec<vk::VertexInputBindingDescription>,
    Vec<vk::VertexInputAttributeDescription>,
) {
    let bindings = if skinned {
        vec![
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<SkinnedVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
        ]
    } else {
        vec![
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<ModelVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
        ]
    };
    let mut attrs = vec![
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
    ];
    if skinned {
        attrs.push(
            vk::VertexInputAttributeDescription::default()
                .location(3)
                .binding(0)
                .format(vk::Format::R16G16B16A16_UINT)
                .offset(32),
        );
        attrs.push(
            vk::VertexInputAttributeDescription::default()
                .location(4)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(40),
        );
    }
    (bindings, attrs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_model_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_formats: &[vk::Format],
    depth_format: vk::Format,
    entry: &std::ffi::CStr,
    skinned: bool,
    blended: bool,
    transparent: bool,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(c"fs_main"),
    ];
    let (bindings, attrs) = model_vertex_input(skinned);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(!blended)
        .depth_compare_op(if blended {
            vk::CompareOp::LESS_OR_EQUAL
        } else {
            vk::CompareOp::LESS
        });
    // One blend entry per colour attachment (colour, velocity, surface): a
    // single entry leaves the other attachments' write masks undefined, and
    // those targets are never written. Blended (transparent) surfaces keep
    // their surface data out of attachment 2 so they cannot overwrite the
    // opaque surface behind them.
    let blend_attachments: Vec<_> = color_formats
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            // Transparent draws do not write motion vectors either: they are
            // composited over the opaque velocity, which must survive.
            if transparent && index == 1 {
                attachment = attachment.color_write_mask(vk::ColorComponentFlags::empty());
            }
            if blended && index == 2 {
                attachment = attachment.color_write_mask(vk::ColorComponentFlags::empty());
            }
            if blended && index == 0 {
                attachment
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
            } else {
                attachment.blend_enable(false)
            }
        })
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = color_formats.to_vec();
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
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
        .push_next(&mut rendering);
    unsafe {
        ctx.device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, err)| err)
            .expect("create model pipeline")[0]
    }
}

/// Vertex layout of the PBR path (`PbrVertex`).
fn pbr_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 5],
) {
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<PbrVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(32),
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(40),
    ];
    (bindings, attrs)
}

/// PBR pipeline: static meshes only, no culling (glTF `doubleSided`; the
/// shader flips the normal on back faces), same blend contract as
/// `create_model_pipeline`.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_pbr_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_formats: &[vk::Format],
    depth_format: vk::Format,
    blended: bool,
    vertex_entry: &std::ffi::CStr,
    transparent: bool,
    cull: vk::CullModeFlags,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(vertex_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(c"fs_main"),
    ];
    let (bindings, attrs) = pbr_vertex_input();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(!blended)
        .depth_compare_op(if blended {
            vk::CompareOp::LESS_OR_EQUAL
        } else {
            vk::CompareOp::LESS
        });
    // Attachment 2 is the surface target: blended surfaces leave it unwritten
    // (see the matching comment in `create_model_pipeline`).
    let blend_attachments: Vec<_> = color_formats
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            // Transparent draws do not write motion vectors either: they are
            // composited over the opaque velocity, which must survive.
            if transparent && index == 1 {
                attachment = attachment.color_write_mask(vk::ColorComponentFlags::empty());
            }
            if blended && index == 2 {
                attachment = attachment.color_write_mask(vk::ColorComponentFlags::empty());
            }
            if blended && index == 0 {
                attachment
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
            } else {
                attachment.blend_enable(false)
            }
        })
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = color_formats.to_vec();
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
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
        .push_next(&mut rendering);
    unsafe {
        ctx.device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, err)| err)
            .expect("create pbr pipeline")[0]
    }
}

/// Depth-only twin of [`create_model_pipeline`] for sun shadow passes. Only
/// opaque casters use it; blended draws are skipped by the caller.
pub(super) fn create_model_shadow_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    depth_format: vk::Format,
    entry: &std::ffi::CStr,
    skinned: bool,
) -> vk::Pipeline {
    let (bindings, attrs) = model_vertex_input(skinned);
    crate::gfx::create_shadow_pipeline(
        ctx,
        shader,
        entry,
        &bindings,
        &attrs,
        layout,
        depth_format,
        None,
    )
}

/// Shadow pass pipeline for PBR casters: vertex stage only for opaque
/// materials, plus `fs_shadow` (alpha-mask cut-outs) when `mask` is set.
pub(super) fn create_pbr_shadow_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    depth_format: vk::Format,
    mask: bool,
    vertex_entry: &std::ffi::CStr,
    cull: vk::CullModeFlags,
) -> vk::Pipeline {
    let (bindings, attrs) = pbr_vertex_input();
    crate::gfx::create_shadow_pipeline_cull(
        ctx,
        shader,
        vertex_entry,
        &bindings,
        &attrs,
        layout,
        depth_format,
        mask.then_some(c"fs_shadow"),
        cull,
    )
}

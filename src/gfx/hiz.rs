//! GPU side of Hi-Z occlusion culling.
//!
//! One compute pass per frame, recorded right after the scene pass: it reduces
//! the scene depth to a small `R32F` grid of farthest depths (see `hiz.wgsl`)
//! and copies that grid into a host-visible buffer of the same frame slot. The
//! renderer reads the buffer back when the slot's fence has signalled, three
//! frames later at most (see `Renderer::read_occlusion`). Nothing waits on the
//! device inside the frame.
//!
//! Every frame slot owns its own grid image and readback buffer, so a later
//! frame can never overwrite a grid that an earlier copy has not finished
//! reading.

use super::post::{BindKind, PostPass, create_general_image, host_uniform_slots};
use super::vulkan::{GpuBuffer, GpuContext, GpuImage, image_barrier};
use crate::render::OcclusionSettings;
use ash::vk;
use std::sync::Arc;

const WGSL: &str = include_str!("hiz.wgsl");
const GRID_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

/// CPU mirror of `HiZParams` in `hiz.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct HiZParams {
    src: [u32; 4],
    dst: [u32; 4],
}

/// The per-slot Hi-Z grids and their readback buffers for one render size.
pub(crate) struct HiZ {
    ctx: Arc<GpuContext>,
    reduce: PostPass,
    images: Vec<GpuImage>,
    params: Vec<GpuBuffer>,
    readback: Vec<GpuBuffer>,
    /// Grid size in texels.
    grid: (u32, u32),
    /// Depth size the grid was built for.
    render: (u32, u32),
    settings: OcclusionSettings,
}

impl HiZ {
    /// Build the pass for a `render` sized depth target. The grid is
    /// `settings.width` texels wide, with the render aspect ratio for its
    /// height.
    pub(crate) fn new(
        ctx: &Arc<GpuContext>,
        render: (u32, u32),
        settings: OcclusionSettings,
    ) -> Self {
        let render = (render.0.max(1), render.1.max(1));
        let width = settings.width.clamp(64, 512);
        let height = ((width as f64 * render.1 as f64 / render.0 as f64).round() as u32).max(1);
        let grid = (width, height);
        let kinds = [
            BindKind::Uniform,
            BindKind::Sampled,
            BindKind::Storage(GRID_FORMAT),
        ];
        let slots = ctx.max_frames_in_flight();
        let bytes = width as u64 * height as u64 * 4;
        Self {
            ctx: Arc::clone(ctx),
            reduce: PostPass::new(ctx, "hi-z reduce", WGSL, c"hiz_reduce", &kinds),
            images: (0..slots)
                .map(|slot| {
                    create_general_image(ctx, &format!("hi-z grid {slot}"), grid, GRID_FORMAT)
                })
                .collect(),
            params: host_uniform_slots(ctx, "hi-z params", std::mem::size_of::<HiZParams>() as u64),
            readback: (0..slots)
                .map(|slot| {
                    ctx.create_cached_readback_buffer(&format!("hi-z readback {slot}"), bytes)
                })
                .collect(),
            grid,
            render,
            settings,
        }
    }

    /// Grid size in texels.
    pub(crate) fn grid(&self) -> (u32, u32) {
        self.grid
    }

    /// Render size the grid was built for.
    pub(crate) fn render_size(&self) -> (u32, u32) {
        self.render
    }

    /// Settings the grid was built with.
    pub(crate) fn settings(&self) -> &OcclusionSettings {
        &self.settings
    }

    /// Record the reduction of `scene_depth` (the depth attachment at render
    /// size, already readable) and the copy into this slot's readback buffer.
    pub(crate) fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        scene_depth: vk::ImageView,
    ) {
        let slots = self.params.len();
        let index = slot % slots;
        let params = HiZParams {
            src: [self.render.0, self.render.1, 0, 0],
            dst: [self.grid.0, self.grid.1, 0, 0],
        };
        self.params[index].write(0, bytemuck::bytes_of(&params));
        let block = &self.params[index];
        let size = std::mem::size_of::<HiZParams>() as u64;
        self.reduce.bind_uniform(slot, 0, block, size);
        self.reduce.bind_sampled(slot, 1, scene_depth);
        self.reduce
            .bind_storage(slot, 2, self.images[index].view, GRID_FORMAT);
        self.reduce.dispatch(cmd, slot, self.grid.0, self.grid.1);

        let device = self.ctx.device().clone();
        let image = self.images[index].image;
        unsafe {
            image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.grid.0,
                    height: self.grid.1,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                self.readback[index].buffer,
                &[region],
            );
        }
    }

    /// The grid of `slot`'s last recorded frame, row-major, as farthest depths.
    /// Only call once the slot's fence has signalled.
    pub(crate) fn read(&self, slot: usize) -> Option<Vec<f32>> {
        let buffer = self.readback.get(slot % self.readback.len())?;
        let bytes = buffer.mapped()?;
        let count = self.grid.0 as usize * self.grid.1 as usize;
        if bytes.len() < count * 4 {
            return None;
        }
        Some(
            bytes[..count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

impl Drop for HiZ {
    fn drop(&mut self) {
        // In-flight frames may still write the grids and buffers.
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
        }
    }
}

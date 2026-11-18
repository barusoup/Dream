//! Shared PBR lighting for the native Metal backend.

use crate::gfx::metal::{GpuBuffer, GpuContext, GpuImage};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCompareFunction, MTLDevice, MTLPixelFormat, MTLRenderCommandEncoder,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLTexture, MTLTextureUsage,
};
use std::sync::{Arc, Mutex};

pub const SHADER: &str = include_str!("lighting.wgsl");
pub const SET_INDEX: usize = 3;
pub const BINDING: u32 = 0;

/// CPU mirror of `LightingFrame` in `lighting.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightingFrame {
    pub cascade_view_proj: [[[f32; 4]; 4]; 4],
    pub cascade_params: [[f32; 4]; 4],
    pub shadow: [f32; 4],
    pub screen: [f32; 4],
    pub env_map: [f32; 4],
    pub cloud_shadow: [f32; 4],
}

impl Default for LightingFrame {
    fn default() -> Self {
        Self {
            cascade_view_proj: [[[0.0; 4]; 4]; 4],
            cascade_params: [[0.0; 4]; 4],
            shadow: [0.0; 4],
            screen: [0.0, 0.0, 65000.0, 0.0],
            env_map: [0.0; 4],
            cloud_shadow: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingEnv {
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    pub cam_pos: [f32; 4],
    pub params: [f32; 4],
}

impl Default for LightingEnv {
    fn default() -> Self {
        Self::with_sun(0.6, 0.7, 3.0)
    }
}

impl LightingEnv {
    pub fn with_sun(azimuth: f32, elevation: f32, intensity: f32) -> Self {
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
            0.0,
        ];
        Self {
            sun_direction: dir,
            sun_color: [intensity, intensity * 0.96, intensity * 0.88, 0.0],
            sky_zenith: [0.18, 0.35, 0.65, 0.0],
            sky_horizon: [0.55, 0.60, 0.68, 0.0],
            sky_ground: [0.10, 0.09, 0.08, 0.0],
            cam_pos: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

pub struct Lighting {
    buffers: Vec<Mutex<GpuBuffer>>,
    raw_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    frame_buffers: Vec<GpuBuffer>,
    dummy_shadow: GpuImage,
    dummy_2d: GpuImage,
    shadow_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    linear_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

impl Lighting {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let slots = ctx.max_frames_in_flight();
        let mut buffers = Vec::with_capacity(slots);
        let mut raw_buffers = Vec::with_capacity(slots);
        for slot in 0..slots {
            let buf = ctx.create_host_buffer(
                &format!("lighting env {slot}"),
                std::mem::size_of::<LightingEnv>() as u64,
            );
            raw_buffers.push(buf.buffer.clone());
            buffers.push(Mutex::new(buf));
        }
        let default_frame = LightingFrame::default();
        let frame_buffers = (0..slots)
            .map(|slot| {
                let mut buf = ctx.create_host_buffer(
                    &format!("lighting frame {slot}"),
                    std::mem::size_of::<LightingFrame>() as u64,
                );
                buf.write_bytes(0, bytemuck::bytes_of(&default_frame));
                buf
            })
            .collect();
        let dummy_shadow = ctx.create_texture_2d_array(
            "lighting dummy shadow",
            1,
            1,
            1,
            1,
            MTLPixelFormat::Depth32Float,
            true,
        );
        let dummy_2d = ctx.create_image(
            "lighting dummy 2d",
            1,
            1,
            MTLPixelFormat::RGBA16Float,
            MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
        );
        let sd = MTLSamplerDescriptor::new();
        sd.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sd.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sd.setMipFilter(MTLSamplerMipFilter::Nearest);
        sd.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sd.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sd.setCompareFunction(MTLCompareFunction::LessEqual);
        let shadow_sampler = ctx
            .device()
            .newSamplerStateWithDescriptor(&sd)
            .expect("lighting shadow sampler");
        let ld = MTLSamplerDescriptor::new();
        ld.setMagFilter(MTLSamplerMinMagFilter::Linear);
        ld.setMinFilter(MTLSamplerMinMagFilter::Linear);
        ld.setMipFilter(MTLSamplerMipFilter::Nearest);
        ld.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        ld.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let linear_sampler = ctx
            .device()
            .newSamplerStateWithDescriptor(&ld)
            .expect("lighting linear sampler");
        Self {
            buffers,
            raw_buffers,
            frame_buffers,
            dummy_shadow,
            dummy_2d,
            shadow_sampler,
            linear_sampler,
        }
    }

    pub fn upload(&self, frame_slot: usize, env: &LightingEnv) {
        let idx = frame_slot % self.buffers.len();
        self.buffers[idx]
            .lock()
            .unwrap()
            .write_bytes(0, bytemuck::bytes_of(env));
    }

    pub fn buffer(&self, frame_slot: usize) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw_buffers[frame_slot % self.raw_buffers.len()]
    }

    #[allow(dead_code)]
    pub(crate) fn bind_shadow_map(&self, _view: Option<()>, _gen: u64) {}

    /// Bind all `@group(3)` lighting buffers/textures/samplers into a Metal render encoder.
    pub(crate) fn bind_fragment_group3(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        frame_slot: usize,
        buf_base: usize,
        tex_base: usize,
        samp_base: usize,
    ) {
        let idx = frame_slot % self.raw_buffers.len();
        let d2d: &ProtocolObject<dyn MTLTexture> = &self.dummy_2d.texture;
        let dsh: &ProtocolObject<dyn MTLTexture> = &self.dummy_shadow.texture;
        unsafe {
            enc.setFragmentBuffer_offset_atIndex(Some(&self.raw_buffers[idx]), 0, buf_base);
            enc.setFragmentBuffer_offset_atIndex(
                Some(&self.frame_buffers[idx].buffer),
                0,
                buf_base + 1,
            );
            enc.setFragmentTexture_atIndex(Some(dsh), tex_base);
            enc.setFragmentTexture_atIndex(Some(d2d), tex_base + 1);
            enc.setFragmentTexture_atIndex(Some(d2d), tex_base + 2);
            enc.setFragmentTexture_atIndex(Some(d2d), tex_base + 3);
            enc.setFragmentTexture_atIndex(Some(d2d), tex_base + 4);
            enc.setFragmentTexture_atIndex(Some(d2d), tex_base + 5);
            enc.setFragmentSamplerState_atIndex(Some(&self.shadow_sampler), samp_base);
            enc.setFragmentSamplerState_atIndex(Some(&self.linear_sampler), samp_base + 1);
            enc.setFragmentSamplerState_atIndex(Some(&self.linear_sampler), samp_base + 2);
        }
    }
}

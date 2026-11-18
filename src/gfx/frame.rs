use super::acquire::AcquireOutcome;
use super::indirect::{IndirectFrameSetup, IndirectInputs, IndirectParams};
use super::linear_depth::{DepthParams, DepthViews};
use super::pipeline::{
    attachment, attachment_in, depth_attachment, general_barrier, make_viewport, map_surface_error,
    scene_region,
};
use super::reflection::{ReflectionFrameSetup, ReflectionInputs, ReflectionParams};
use super::taa::TaaInputs;
use super::timing::elapsed_timestamp_ticks;
use super::vulkan;
use super::{DrawCtx, FramePhases, Renderer, SceneFrame, ScenePass, SurfaceError};
use crate::camera::Camera;
use crate::lighting::{Lighting, LightingEnv, ScreenSpaceViews};
use crate::render::Tonemap;
use ash::vk;
use std::sync::Arc;
use std::time::Instant;

/// Fragment push constants of the blit pipeline (`struct BlitPush` in
/// `blit.wgsl`). With `tonemap == 0` and `dither == 0` the blit is a copy.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BlitPush {
    tonemap: u32,
    exposure: f32,
    dither: u32,
    pad: u32,
}

impl BlitPush {
    fn from_settings(settings: &crate::render::RenderSettings) -> Self {
        let tonemap = match settings.tonemap {
            Tonemap::None => 0,
            Tonemap::Aces => 1,
            Tonemap::AgX => 2,
            Tonemap::Neutral => 3,
        };
        Self {
            tonemap,
            exposure: settings.exposure,
            dither: u32::from(settings.dither),
            pad: 0,
        }
    }
}

/// What [`Renderer::render_scene`] adds to a frame: the lighting set and the
/// sun and sky, which the shadow passes and the lighting block need.
pub(super) struct SceneRequest<'a> {
    pub(super) lighting: &'a Lighting,
    /// The environment the lit passes and the fog use (sun-driven when the atmosphere drives lighting).
    pub(super) env: LightingEnv,
    /// The caller's environment, for the sky: the atmosphere computes the sun's
    /// transmittance itself.
    pub(super) source: LightingEnv,
}

impl Renderer {
    /// The previous frame's screen-space results as the lighting set binds them,
    /// with the generation of the chains that produced them. `None` until the
    /// shared depth has been recorded once.
    fn screen_space_views(&self) -> (Option<ScreenSpaceViews>, u64) {
        let Some(depth) = self.linear_depth.as_ref().and_then(|depth| depth.now()) else {
            return (None, 0);
        };
        let views = ScreenSpaceViews {
            gi: self.indirect.as_ref().and_then(|chain| chain.output()),
            reflection: self.reflection.as_ref().and_then(|chain| chain.output()),
            depth,
        };
        (Some(views), self.screen_generation)
    }

    /// Render one frame with no UI overlay.
    pub fn render<F>(&mut self, camera: &Camera, draw: F) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        let mut draw = Some(draw);
        let mut call = |ctx: &mut DrawCtx<'_>| draw.take().map_or(0, |f| f(ctx));
        self.render_inner(camera, None, &mut call, None, true, None)
    }

    /// Render one frame with an optional [`crate::ui::draw::UiFrame`] drawn
    /// over the composed image. The overlay renders at display resolution
    /// *after* upscaling, so text stays crisp at every `upscale` preset and
    /// is immune to temporal reprojection.
    pub fn render_ui<F>(
        &mut self,
        camera: &Camera,
        scene_rect: Option<crate::ui::Rect>,
        draw: F,
        ui: Option<&crate::ui::draw::UiFrame>,
    ) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        let mut draw = Some(draw);
        let mut call = |ctx: &mut DrawCtx<'_>| draw.take().map_or(0, |f| f(ctx));
        self.render_inner(camera, scene_rect, &mut call, ui, true, None)
    }

    pub fn render_ui_only(
        &mut self,
        ui: Option<&crate::ui::draw::UiFrame>,
    ) -> Result<(), SurfaceError> {
        let mut none = |_: &mut DrawCtx<'_>| 0;
        self.render_inner(&Camera::default(), None, &mut none, ui, false, None)
            .map(|_| ())
    }

    /// Render one frame of a lit scene, with the sun shadow passes the current
    /// [`Self::render_settings`] ask for.
    ///
    /// `draw` runs once for each shadow cascade that updates this frame, with
    /// [`ScenePass::Shadow`], and then once for the main pass. Its main-pass
    /// return value is what this returns. Use one `Lighting` for every
    /// `render_scene` call; it is bound to the shadow map when that map is
    /// created or rebuilt.
    pub fn render_scene<F>(
        &mut self,
        frame: &SceneFrame<'_>,
        mut draw: F,
    ) -> Result<usize, SurfaceError>
    where
        F: FnMut(&mut DrawCtx<'_>) -> usize,
    {
        self.ensure_shadow_map();
        let p = frame.camera.position;
        let mut source = frame.env;
        source.cam_pos = [p.x, p.y, p.z, 0.0];
        let mut env = self.drive_env(frame.camera, source);
        env.cam_pos = [p.x, p.y, p.z, 0.0];
        let shadow_view = self.shadows.as_ref().map(|map| map.image.view);
        frame
            .lighting
            .bind_shadow_map(shadow_view, self.shadow_generation);
        let request = SceneRequest {
            lighting: frame.lighting,
            env,
            source,
        };
        self.render_inner(
            frame.camera,
            frame.scene_rect,
            &mut draw,
            frame.ui,
            true,
            Some(&request),
        )
    }

    fn render_inner(
        &mut self,
        camera: &Camera,
        scene_rect: Option<crate::ui::Rect>,
        draw: &mut dyn FnMut(&mut DrawCtx<'_>) -> usize,
        ui: Option<&crate::ui::draw::UiFrame>,
        scene_enabled: bool,
        scene: Option<&SceneRequest<'_>>,
    ) -> Result<usize, SurfaceError> {
        self.scene_sample = None;
        self.phases = FramePhases::default();
        self.sync_occlusion();
        // The setters keep the display image in step with `needs_display`;
        // this guards the invariant so a frame never records through the
        // 1x1 placeholder.
        if self.needs_display() && !self.display_live {
            self.ensure_display();
        }
        // Frame start. The acquire-ahead worker already waited this slot's
        // fence and reserved its swapchain image, so a ready token means the
        // slot is free and the frame renders without blocking. Within the
        // wait budget a not-yet-ready token skips the frame
        // (`SurfaceError::Timeout`) instead of freezing the app loop behind a
        // stalled presentation engine; the request stays outstanding and a
        // later frame picks the image up.
        // A token parked by a quiesce is this frame's image already.
        let parked = self.parked_token.take();
        if parked.is_none() && !self.acquire.has_request() {
            self.request_acquire();
        }
        let wait_start = Instant::now();
        let outcome = match parked {
            Some(token) => AcquireOutcome::Ready(token),
            None => self.acquire.wait(self.frame_wait_budget),
        };
        let token = match outcome {
            AcquireOutcome::Ready(token) => token,
            AcquireOutcome::Pending => {
                self.phases.token_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
                return Err(SurfaceError::Timeout);
            }
            AcquireOutcome::Failed(err) => return Err(err),
            AcquireOutcome::Lost => return Err(SurfaceError::Other),
        };
        self.phases.fence_wait_us = token.fence_wait_us;
        self.phases.acquire_us = token.acquire_us;
        self.phases.token_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
        let device = self.ctx.device().clone();
        let frame = token.slot;
        let image_index = token.image_index;
        let suboptimal = token.suboptimal;
        // This slot's fence signalled its serial; the in-order queue makes
        // every earlier submission complete as well.
        self.ctx.confirm_completed(self.frame_serials[frame]);
        // Buffers dropped while earlier frames were in flight can go now.
        self.ctx.collect_retired_buffers();
        self.finish_scene_thumbnail(frame);
        if self.frame_serials[frame] > 0
            && self.scene_recorded[frame]
            && self.ctx.timestamp_valid_bits() > 0
        {
            self.pass_timer.collect(frame);
            let mut ticks = [0u64; 6];
            let got = unsafe {
                device.get_query_pool_results(
                    self.gpu_pool,
                    frame as u32 * 6,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            let bits = self.ctx.timestamp_valid_bits();
            let period_ms = self.ctx.timestamp_period() * 1e-6;
            if got.is_ok() {
                if let Some(elapsed) = elapsed_timestamp_ticks(ticks[0], ticks[1], bits) {
                    let ms = elapsed as f32 * period_ms;
                    if ms.is_finite() && ms > 0.0 {
                        self.scene_ms[frame] = ms;
                        self.scene_sample = Some((self.frame_serials[frame], ms));
                    }
                }
                if let Some(elapsed) = elapsed_timestamp_ticks(ticks[2], ticks[3], bits) {
                    let ms = elapsed as f32 * period_ms;
                    self.post_ms[frame] = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
                }
                if let Some(elapsed) = elapsed_timestamp_ticks(ticks[4], ticks[5], bits) {
                    let ms = elapsed as f32 * period_ms;
                    self.shadow_ms[frame] = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
                }
            }
        } else if self.frame_serials[frame] > 0 {
            self.scene_ms[frame] = 0.0;
            self.post_ms[frame] = 0.0;
            self.shadow_ms[frame] = 0.0;
        }

        // The slot's fence has signalled, so its earlier occlusion grid is
        // readable, and the copy from this slot can be read after this frame.
        self.read_occlusion(frame);
        let occlusion_data = self.occlusion_data.clone();
        let mut occlusion_meta: Option<(u64, crate::cull::OcclusionMeta)> = None;

        unsafe {
            device
                .reset_fences(&[self.in_flight[frame]])
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            device
                .reset_command_buffer(self.cmd[frame], vk::CommandBufferResetFlags::empty())
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
        }

        let cmd = self.cmd[frame];
        let render_size = self.render_size;
        let display_size = self.display_size;
        let swapchain_extent = (self.swapchain.extent.width, self.swapchain.extent.height);
        // Both FSR and TAA integrate jittered samples, so both draw with jitter.
        let jp = if scene_enabled && (self.temporal || self.taa_active()) {
            self.jitter.next(render_size.0, display_size.0)
        } else {
            [0.0, 0.0]
        };
        let curr_view_projection = camera.view_projection();

        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        let reset = !self.history_valid;
        if scene_enabled {
            self.sync_sky(true);
            self.write_uniforms(camera, jp);
            if self.temporal
                && let Some(fsr) = self.fsr.as_mut()
            {
                fsr.update(camera, jp, dt, reset, frame);
            }
        }

        let mark = Instant::now();
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
        }

        // This slot's six queries are reset before anything is timestamped, and
        // the shadow bracket opens here, ahead of the cascades.
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                self.pass_timer.begin(&device, cmd, frame);
                device.cmd_reset_query_pool(cmd, self.gpu_pool, frame as u32 * 6, 6);
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6 + 4,
                );
            }
        }
        // The slot fence was waited above, so its TLAS and build inputs can be
        // safely rewritten. Mark the build separately in the GPU timings.
        let rt_active = self.render_settings.ray_tracing.is_some_and(|rt| {
            (rt.reflections && self.render_settings.reflections.is_some())
                || (rt.ambient
                    && (self.render_settings.ao.is_some() || self.render_settings.gi.is_some()))
        });
        let rt_build_recorded = scene_enabled && rt_active && self.rt_scene.is_some();
        if rt_build_recorded && let Some(rt_scene) = self.rt_scene.as_mut() {
            if self.ctx.timestamp_valid_bits() > 0 {
                self.pass_timer.mark(&device, cmd, frame, "TLAS build");
            }
            rt_scene.record_build(cmd, frame);
            if self.ctx.timestamp_valid_bits() > 0 {
                self.pass_timer
                    .mark(&device, cmd, frame, "cloud shadow map");
            }
        }
        // Sun shadows: each updated cascade is a rendering scope of its own,
        // recorded ahead of the main pass. The lighting block then carries the
        // fits those passes rendered with.
        // The cloud shadow map is recorded first: the lighting and fog of this
        // frame read it.
        if !rt_build_recorded {
            self.pass_timer
                .mark(&device, cmd, frame, "cloud shadow map");
        }
        let cloud_shadow = match scene {
            Some(request) => self.record_cloud_shadow(
                cmd,
                frame,
                camera,
                glam::Vec3::from_slice(&request.env.sun_direction[..3]),
            ),
            None => None,
        };
        if let Some(request) = scene {
            self.pass_timer.mark(&device, cmd, frame, "sun cascades");
            let settings = self.render_settings.shadows.clone();
            if let (Some(settings), true) = (settings.as_ref(), self.shadows.is_some()) {
                self.record_shadow_passes(
                    &device,
                    cmd,
                    frame,
                    camera,
                    &request.env,
                    settings,
                    draw,
                );
            }
            // The chains' last outputs are last frame's results: bind them for this
            // slot's forward pass, before this frame's chains overwrite anything.
            let (views, generation) = self.screen_space_views();
            request.lighting.bind_screen_space(frame, views, generation);
            // The sky environment map is the previous frame's, as the chains'
            // outputs are; its valid flag and last mip go in the block's spare slots.
            let slots = self.ctx.max_frames_in_flight();
            let env_map = self
                .atmosphere
                .as_ref()
                .and_then(|atmosphere| atmosphere.env_map((frame + slots - 1) % slots));
            request.lighting.bind_env_map(frame, env_map);
            request
                .lighting
                .bind_cloud_shadow(frame, cloud_shadow.map(|shadow| shadow.view));
            let mut block = self.lighting_frame(camera, settings.as_ref());
            block.indirect[2] = if env_map.is_some() { 1.0 } else { 0.0 };
            block.indirect[3] = (crate::render::SKY_ENV_MIPS - 1) as f32;
            if let Some(shadow) = cloud_shadow {
                block.cloud_shadow = shadow.grid;
                block.cloud_layer = shadow.layer;
            }
            request.lighting.upload(frame, &request.env);
            request.lighting.upload_frame(frame, &block);
            if let Some(fog) = self.fog.as_mut() {
                fog.upload_lighting(frame, &block);
            }
        }

        // The scene owns one region of the render target - the app's viewport.
        // Outside it are its panels, which are drawn later at display
        // resolution: shading those pixels would be work nobody sees. The
        // render area stays the whole target so every pixel is still cleared;
        // only the scissor is narrowed.
        let (scene_viewport, scene_scissor) = scene_region(scene_rect, render_size, display_size);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: render_size.0,
                height: render_size.1,
            },
        };
        let swap_viewport = make_viewport(swapchain_extent);
        let swap_scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: swapchain_extent.0,
                height: swapchain_extent.1,
            },
        };

        // Swapchain image: undefined -> color attachment (the blit overwrites
        // every pixel, so the previous contents do not matter).
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        unsafe {
            // The image was acquired by `image_available`, which this submit
            // waits on at COLOR_ATTACHMENT_OUTPUT. The transition must be ordered
            // after that wait, so its source stage has to include that stage;
            // TOP_OF_PIPE runs before the wait (SYNC-HAZARD-WRITE-AFTER-READ vs
            // vkAcquireNextImageKHR).
            vulkan::image_barrier(
                &device,
                cmd,
                swapchain_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
        }

        // Direct composition: the blit and UI passes draw into the swapchain
        // image itself, and no present pass runs. Otherwise they draw into the
        // composed display image, which the present pass then copies out.
        let direct = !self.needs_display();
        let (target_view, target_size, target_layout) = if direct {
            (
                swapchain_view,
                swapchain_extent,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )
        } else {
            (self.display.view, display_size, vk::ImageLayout::GENERAL)
        };
        let target_viewport = make_viewport(target_size);
        let target_scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: target_size.0,
                height: target_size.1,
            },
        };

        // --- Scene pass ---
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6 + 5,
                );
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6,
                );
            }
        }
        let scene_pipeline = self.scene_pipeline;
        let scene_layout = self.scene_layout;
        let scene_set = self.scene_sets[frame];
        let pulling = self.pulling.map(|(pipeline, layout, _)| (pipeline, layout));
        let frame_id = self.frame_id;
        let already_ready = self.targets_ready;
        let mut scene_pass_begun = false;

        let begin_scene = |this: &mut Self, begun: &mut bool| {
            if *begun {
                return;
            }
            *begun = true;
            if !this.targets_ready {
                this.ensure_scene_targets();
                if this.temporal
                    && let Some(fsr) = this.fsr.as_mut()
                {
                    fsr.update(camera, jp, dt, reset, frame);
                }
            }
            let color = attachment(
                this.targets.color.view,
                vk::AttachmentLoadOp::CLEAR,
                [0.48, 0.69, 0.89, 1.0],
            );
            let velocity = attachment(
                this.targets.velocity.view,
                vk::AttachmentLoadOp::CLEAR,
                [0.0, 0.0, 0.0, 0.0],
            );
            let surface = attachment(
                this.targets.surface.view,
                vk::AttachmentLoadOp::CLEAR,
                [0.0, 0.0, 0.0, 0.0],
            );
            let colors = [color, velocity, surface];
            let depth = depth_attachment(this.targets.depth.view, 1.0);
            let info = vk::RenderingInfo::default()
                .render_area(scissor)
                .layer_count(1)
                .color_attachments(&colors)
                .depth_attachment(&depth);
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[scene_viewport]);
                device.cmd_set_scissor(cmd, 0, &[scene_scissor]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, scene_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    scene_layout,
                    0,
                    &[scene_set],
                    &[],
                );
            }
        };

        self.pass_timer.mark(&device, cmd, frame, "scene pass");

        let visible = if scene_enabled {
            if already_ready {
                begin_scene(self, &mut scene_pass_begun);
            }
            let mut starter = || {
                begin_scene(self, &mut scene_pass_begun);
            };
            let mut ctx = DrawCtx {
                device: &device,
                cmd,
                scene_pipeline,
                scene_layout,
                scene_set,
                pulling,
                frame_slot: frame,
                frame_id,
                starter: std::cell::RefCell::new(if already_ready {
                    None
                } else {
                    Some(&mut starter)
                }),
                pass: ScenePass::Main,
                has_transparent: scene.is_some(),
                view_proj: camera
                    .jittered_view_projection(jp, [render_size.0 as f32, render_size.1 as f32]),
                occlusion: occlusion_data.as_deref(),
            };
            let visible = draw(&mut ctx);
            let _ = ctx.starter.borrow_mut().take();
            // An explicit `render_scene` owns an empty scene too: initialize
            // targets for sky/cloud/fog and post work. A legacy `render_ui`
            // call with an empty scene closure must remain UI-only and avoid
            // lazily creating 3D targets on its first frame.
            if !scene_pass_begun && scene.is_some() {
                begin_scene(self, &mut scene_pass_begun);
            }
            if scene_pass_begun {
                unsafe { device.cmd_end_rendering(cmd) };
            }
            visible
        } else {
            0
        };
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6 + 1,
                );
                // The post work starts where the scene pass ends.
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6 + 2,
                );
            }
        }

        if scene_pass_begun {
            // Scene colour, depth and motion vectors become readable.
            unsafe {
                let barriers = [
                    general_barrier(
                        self.targets.color.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    general_barrier(
                        self.targets.depth.image,
                        vk::ImageAspectFlags::DEPTH,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    general_barrier(
                        self.targets.velocity.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    general_barrier(
                        self.targets.surface.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                ];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }

            // --- Hi-Z occlusion grid: the farthest depth per block of this frame's
            // scene, copied into this slot's readback. Recorded before the sky so
            // that sky (depth 1) and fog do not change what is read back. ---
            self.pass_timer.mark(&device, cmd, frame, "hi-z");
            if let Some(hiz) = self.occlusion.as_mut() {
                hiz.record(cmd, frame, self.targets.depth.view);
                let settings = *hiz.settings();
                occlusion_meta = Some((
                    self.occlusion_generation,
                    crate::cull::OcclusionMeta {
                        // The unjittered matrix: the grid is tested as "hidden
                        // from where the camera was", not from the jittered pixel.
                        view_projection: camera.view_projection(),
                        camera_position: camera.position,
                        frame_id: self.frame_id,
                        margin_texels: settings.margin_texels,
                        depth_bias: settings.depth_bias,
                    },
                ));
            }

            self.pass_timer.mark(&device, cmd, frame, "linear depth");

            // --- Shared linear depth. The view clouds read it before the sky pass,
            // and the screen-space chains below read it after the transparent pass.
            // Depth is not changed by either, so recording it first is exact. ---
            let (near, far) = camera.clip_planes();
            if let Some(depth) = self.linear_depth.as_mut() {
                let params = DepthParams::new(
                    depth.size(),
                    render_size,
                    near,
                    far,
                    camera.fov_y(),
                    camera.aspect(),
                );
                depth.record(cmd, frame, self.targets.depth.view, &params);
            }

            // --- Physically based sky, then the per-pixel view clouds, then volumetric
            // fog: over the scene's empty pixels and through its geometry, before the
            // chains below read it ---
            if let Some(request) = scene {
                self.record_sky_and_fog(
                    &device,
                    cmd,
                    frame,
                    camera,
                    jp,
                    (scene_viewport, scene_scissor),
                    scissor,
                    &request.source,
                    &request.env,
                    reset,
                    cloud_shadow.map(|shadow| shadow.view),
                );
            }

            self.pass_timer.mark(&device, cmd, frame, "transparent");

            // --- Transparent pass: alpha-blended draws composited over the sky and
            // fog. It loads colour, motion vectors and surface data, tests depth
            // without writing it, and writes colour only (the blended pipelines
            // mask the other two). Begun lazily, so a frame without transparent
            // draws records nothing. ---
            if scene.is_some() {
                let mut began = false;
                let mut starter = || {
                    if began {
                        return;
                    }
                    began = true;
                    let color = attachment(
                        self.targets.color.view,
                        vk::AttachmentLoadOp::LOAD,
                        [0.0; 4],
                    );
                    let keep = |view| {
                        attachment(view, vk::AttachmentLoadOp::LOAD, [0.0; 4])
                            .store_op(vk::AttachmentStoreOp::NONE)
                    };
                    let colours = [
                        color,
                        keep(self.targets.velocity.view),
                        keep(self.targets.surface.view),
                    ];
                    let depth = vk::RenderingAttachmentInfo::default()
                        .image_view(self.targets.depth.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::NONE);
                    let info = vk::RenderingInfo::default()
                        .render_area(scissor)
                        .layer_count(1)
                        .color_attachments(&colours)
                        .depth_attachment(&depth);
                    unsafe {
                        device.cmd_begin_rendering(cmd, &info);
                        device.cmd_set_viewport(cmd, 0, &[scene_viewport]);
                        device.cmd_set_scissor(cmd, 0, &[scene_scissor]);
                    }
                };
                let mut ctx = DrawCtx {
                    device: &device,
                    cmd,
                    scene_pipeline,
                    scene_layout,
                    scene_set,
                    pulling,
                    frame_slot: frame,
                    frame_id,
                    starter: std::cell::RefCell::new(Some(&mut starter)),
                    pass: ScenePass::Transparent,
                    view_proj: camera
                        .jittered_view_projection(jp, [render_size.0 as f32, render_size.1 as f32]),
                    occlusion: None,
                    has_transparent: true,
                };
                draw(&mut ctx);
                let _ = ctx.starter.borrow_mut().take();
                if began {
                    unsafe { device.cmd_end_rendering(cmd) };
                    super::sky_frame::barrier_all(&device, cmd);
                }
            }

            self.record_scene_thumbnail(cmd, frame);

            self.pass_timer.mark(&device, cmd, frame, "AO and GI");

            // --- Screen-space AO/GI and reflections (read by frame N+1) ---
            let depth_size = self.linear_depth.as_ref().map_or(render_size, |d| d.size());
            let depth_views = self.linear_depth.as_ref().and_then(|depth| {
                Some(DepthViews {
                    now: depth.now()?,
                    prev: depth.prev(),
                })
            });
            if let (Some(views), Some(chain)) = (depth_views, self.indirect.as_mut()) {
                let mut inputs = IndirectInputs {
                    colour: self.targets.color.view,
                    surface: self.targets.surface.view,
                    velocity: self.targets.velocity.view,
                    rt: None,
                };
                let quality = self.render_settings.indirect_quality;
                let setup = IndirectFrameSetup {
                    view: camera.view(),
                    near,
                    far,
                    fov_y: camera.fov_y(),
                    aspect: camera.aspect(),
                    full: render_size,
                    ao: self.render_settings.ao.as_ref(),
                    gi: self.render_settings.gi.as_ref(),
                    quality,
                    frame: self.frame_id as u32,
                    reset,
                    env: scene.map_or_else(crate::lighting::LightingEnv::default, |s| s.env),
                    env_valid: self.atmosphere.as_ref().is_some_and(|a| a.has_env_map()),
                };
                let mut params = IndirectParams::new(&setup, chain.working_size(), depth_size);
                let rt_binds = if self
                    .render_settings
                    .ray_tracing
                    .is_some_and(|rt| rt.ambient)
                {
                    let slots = self.ctx.max_frames_in_flight();
                    let env = self
                        .atmosphere
                        .as_ref()
                        .and_then(|atmosphere| atmosphere.env_map((frame + slots - 1) % slots));
                    self.rt_scene.as_ref().and_then(|rt| rt.binds(frame, env))
                } else {
                    None
                };
                if rt_binds.is_some()
                    && let Some(rt) = self.render_settings.ray_tracing
                {
                    let limit = if rt.ambient_distance.is_finite() {
                        rt.ambient_distance.clamp(0.5, 500.0)
                    } else {
                        30.0
                    };
                    params.ao[0] = params.ao[0].min(limit);
                    params.gi[0] = params.gi[0].min(limit);
                }
                inputs.rt = rt_binds;
                chain.record(cmd, frame, &inputs, views, &params, quality.denoise);
            }

            self.pass_timer.mark(&device, cmd, frame, "reflections");

            // --- Screen-space reflections (read by frame N+1) ---
            let refl_animated = self.taa_active() || self.temporal;
            if let (Some(views), Some(chain), Some(settings)) = (
                depth_views,
                self.reflection.as_mut(),
                self.render_settings.reflections,
            ) {
                let mut inputs = ReflectionInputs {
                    colour: self.targets.color.view,
                    surface: self.targets.surface.view,
                    velocity: self.targets.velocity.view,
                    rt: None,
                };
                let setup = ReflectionFrameSetup {
                    view: camera.view(),
                    near,
                    far,
                    fov_y: camera.fov_y(),
                    aspect: camera.aspect(),
                    full: render_size,
                    settings,
                    animated: refl_animated || settings.denoise,
                    frame: self.frame_id as u32,
                    reset,
                    env: scene.map_or_else(crate::lighting::LightingEnv::default, |s| s.env),
                    env_valid: self.atmosphere.as_ref().is_some_and(|a| a.has_env_map()),
                };
                let mut params = ReflectionParams::new(&setup, chain.working_size(), depth_size);
                let rt_binds = if self
                    .render_settings
                    .ray_tracing
                    .is_some_and(|rt| rt.reflections)
                {
                    let slots = self.ctx.max_frames_in_flight();
                    let env = self
                        .atmosphere
                        .as_ref()
                        .and_then(|atmosphere| atmosphere.env_map((frame + slots - 1) % slots));
                    self.rt_scene.as_ref().and_then(|rt| rt.binds(frame, env))
                } else {
                    None
                };
                if rt_binds.is_some()
                    && let Some(rt) = self.render_settings.ray_tracing
                {
                    params.trace[0] = if rt.reflection_distance.is_finite() {
                        rt.reflection_distance.clamp(1.0, 2000.0)
                    } else {
                        150.0
                    };
                }
                inputs.rt = rt_binds;
                chain.record(cmd, frame, &inputs, views, &params, settings.denoise);
            }

            self.pass_timer
                .mark(&device, cmd, frame, "temporal upscale");

            // --- FSR 3.1 temporal upscale ---
            if self.temporal
                && let Some(fsr) = self.fsr.as_mut()
            {
                fsr.record(cmd, !reset, frame);
            }

            self.pass_timer.mark(&device, cmd, frame, "TAA");

            // --- Native temporal anti-aliasing (resolve, then sharpen) ---
            if self.taa_active()
                && let Some(taa) = self.taa.as_mut()
            {
                let inputs = TaaInputs {
                    color: self.targets.color.view,
                    depth: self.targets.depth.view,
                    velocity: self.targets.velocity.view,
                };
                let settings = self.render_settings.taa.clone();
                taa.record(cmd, frame, &inputs, &settings, jp, reset);
            }
        }

        if scene_enabled {
            unsafe {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    self.blit_source().0,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageAspectFlags::COLOR,
                );
            }
        }

        // The post work ends where the blit starts.
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 6 + 3,
                );
            }
        }

        self.pass_timer.mark(&device, cmd, frame, "blit");

        // --- Blit pass (FSR output -> target: the swapchain image when composing
        // directly, otherwise the composed display image) ---
        {
            let load = if scene_enabled {
                vk::AttachmentLoadOp::DONT_CARE
            } else {
                vk::AttachmentLoadOp::CLEAR
            };
            let color = attachment_in(target_view, target_layout, load, [0.0, 0.0, 0.0, 1.0]);
            let info = vk::RenderingInfo::default()
                .render_area(target_scissor)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color));
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[target_viewport]);
                device.cmd_set_scissor(cmd, 0, &[target_scissor]);
                if scene_enabled {
                    // This slot's set is free: its fence has been waited.
                    self.bind_blit_source(frame, self.blit_source().1);
                    let push = BlitPush::from_settings(&self.render_settings);
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.blit_pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.blit_layout,
                        0,
                        &[self.blit_sets[frame]],
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        self.blit_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    device.cmd_draw(cmd, 3, 1, 0, 0);
                }
                device.cmd_end_rendering(cmd);
            }
        }

        self.pass_timer.mark(&device, cmd, frame, "ui");

        // --- UI overlay (display resolution, over the target) ---
        if let Some(ui) = ui {
            if direct {
                // The blit wrote the swapchain image that the UI pass loads
                // in its own rendering scope; order the two explicitly.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    );
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        std::slice::from_ref(&barrier),
                        &[],
                        &[],
                    );
                }
            }
            // The glass backdrop copies the target, so hand the UI the image behind the view. A
            // direct frame whose swapchain cannot be blitted from passes null: the capture is skipped
            // and glass shows its last backdrop.
            let target_image = if !direct {
                self.display.image
            } else if self.swapchain.supports_transfer_src() {
                swapchain_image
            } else {
                vk::Image::null()
            };
            self.ui.record_in_layout(
                cmd,
                frame,
                target_image,
                target_view,
                target_size,
                ui,
                target_layout,
            );
        }

        if !direct {
            // The composed display image holds this frame now and becomes
            // readable for the present pass.
            self.display_written = true;
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[general_barrier(
                        self.display.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    )],
                );
            }

            self.pass_timer.mark(&device, cmd, frame, "present");

            // --- Present pass (display image -> swapchain) ---
            // VUID-vkCmdBeginRendering-pRenderingInfo-09592: the attachment
            // layout must match the layout the image is actually in. The
            // swapchain image was moved to COLOR_ATTACHMENT_OPTIMAL above, so
            // declare that here rather than GENERAL.
            let color = attachment_in(
                swapchain_view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AttachmentLoadOp::DONT_CARE,
                [0.0, 0.0, 0.0, 1.0],
            );
            let info = vk::RenderingInfo::default()
                .render_area(swap_scissor)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color));
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[swap_viewport]);
                device.cmd_set_scissor(cmd, 0, &[swap_scissor]);
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.present_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.present_layout,
                    0,
                    &[self.present_set],
                    &[],
                );
                let crosshair = if self.show_crosshair { 1.0 } else { 0.0 };
                let push = [display_size.0 as f32, display_size.1 as f32, crosshair];
                device.cmd_push_constants(
                    cmd,
                    self.present_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw(cmd, 3, 1, 0, 0);
                device.cmd_end_rendering(cmd);
            }
        }

        unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                swapchain_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
                vk::ImageAspectFlags::COLOR,
            );
            device
                .end_command_buffer(cmd)
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            self.phases.record_us = mark.elapsed().as_secs_f32() * 1e6;

            let wait_semaphores = [self.image_available[frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            // Keyed by swapchain image, not frame slot: see `Renderer::render_finished`
            // (VUID-vkQueueSubmit-pSignalSemaphores-00067).
            let signal_semaphores = [self.render_finished[image_index as usize]];
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signal_semaphores);
            // A failed submit gets no serial; the frame's serial slot only
            // records submissions that actually entered the queue.
            let mark = Instant::now();
            let serial = self
                .ctx
                .try_submit_tracked(&submit, self.in_flight[frame])
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            self.phases.submit_us = mark.elapsed().as_secs_f32() * 1e6;
            self.frame_serials[frame] = serial;
            self.scene_recorded[frame] = scene_pass_begun;
            self.occlusion_pending[frame] = occlusion_meta;
            // The submitted frame owns this id now; a failed present below
            // must not reuse it for slot bookkeeping.
            self.frame_id += 1;
            self.frame = (frame + 1) % self.ctx.max_frames_in_flight();

            let swapchains = [self.swapchain.swapchain];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices);
            // Present on the same queue must be serialized with submits so
            // the serial tracker mirrors real queue order.
            {
                let mark = Instant::now();
                let _queue = self.ctx.queue_lock.lock().unwrap();
                // Present reports the same resize/loss conditions as
                // acquire: `OUT_OF_DATE` means the surface moved on while
                // this frame was recorded, so callers must be able to resize
                // and retry rather than see an undiagnosable `Other`.
                self.ctx
                    .swapchain_loader
                    .queue_present(self.ctx.queue(), &present)
                    .map_err(map_surface_error)?;
                self.phases.present_us = mark.elapsed().as_secs_f32() * 1e6;
            }
            // Request the next acquire only AFTER presenting. The image this
            // frame just presented is what frees the swapchain image the next
            // acquire needs: requesting earlier over-commits the pool (this
            // frame still holds its recording image, the presentation engine
            // holds the queued ones) and the acquire then throttles to the
            // compositor's latch cadence, dragging the whole loop with it.
            // The request still resolves on the worker while the app runs its
            // next update.
            self.request_acquire();
        }

        if scene_enabled {
            self.scene_time += f64::from(dt) * f64::from(self.scene_time_scale);
            self.prev_view_projection = curr_view_projection;
            self.history_valid = true;
        } else {
            self.history_valid = false;
        }

        if suboptimal {
            Err(SurfaceError::Outdated)
        } else {
            Ok(visible)
        }
    }

    /// Segment timings of the most recent `render*` call, in microseconds --
    /// including a skipped one, whose phases show where its wait budget went.
    /// See [`FramePhases`] for how to read them.
    pub fn frame_phases(&self) -> FramePhases {
        self.phases
    }

    /// Post the acquire-ahead request for the next frame slot. Only called
    /// with no request outstanding; the worker gates the acquire on the
    /// slot's fence, so a delivered token proves the slot is free.
    fn request_acquire(&mut self) {
        let frame = self.frame;
        self.acquire.request(
            Arc::clone(&self.ctx),
            self.swapchain.swapchain,
            frame,
            self.in_flight[frame],
            self.image_available[frame],
        );
    }

    /// A frame that aborts before its submission entered the queue has
    /// already acquired its swapchain image: `image_available[frame]` is
    /// signalled but unwaited, and a binary semaphore must be waited before
    /// it can be signalled again. [`Self::reclaim_slot`] waits it out.
    fn frame_abort(&mut self, frame: usize, err: SurfaceError) -> SurfaceError {
        self.reclaim_slot(frame);
        err
    }

    /// Return an aborted frame's slot to a reusable state with an empty batch
    /// that waits `image_available[frame]` out. The batch carries the slot's
    /// fence -- reset first: the aborted frame may have reset it without
    /// submitting -- so the acquire-ahead worker's fence gate covers this
    /// wait before it can signal the semaphore again. Safe to call for any
    /// slot not currently in flight; a failed submit means none is.
    pub(super) fn reclaim_slot(&mut self, frame: usize) {
        let device = self.ctx.device().clone();
        if unsafe { device.reset_fences(&[self.in_flight[frame]]) }.is_err() {
            return; // device lost; nothing here can be salvaged
        }
        let wait = [self.image_available[frame]];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages);
        if let Ok(serial) = self.ctx.try_submit_tracked(&submit, self.in_flight[frame]) {
            self.frame_serials[frame] = serial;
            self.scene_recorded[frame] = false;
            self.occlusion_pending[frame] = None;
        }
    }

    /// Publish the occlusion grid this slot recorded, if any. Called once the
    /// slot's fence has signalled, so the readback buffer is complete.
    fn read_occlusion(&mut self, slot: usize) {
        let Some((generation, meta)) = self.occlusion_pending[slot].take() else {
            return;
        };
        if generation != self.occlusion_generation {
            return;
        }
        let Some(hiz) = self.occlusion.as_ref() else {
            return;
        };
        let Some(values) = hiz.read(slot) else {
            return;
        };
        let (width, height) = hiz.grid();
        self.occlusion_data = Some(Arc::new(crate::cull::OcclusionBuffer::new(
            width, height, values, meta,
        )));
    }

    /// The image and view the blit samples this frame: the TAA output (or the
    /// resolved history) when TAA runs, the FSR output when FSR runs, and
    /// otherwise the scene colour.
    fn blit_source(&self) -> (vk::Image, vk::ImageView) {
        if let Some(taa) = self
            .taa
            .as_ref()
            .filter(|_| self.taa_active() && self.targets_ready)
        {
            return taa.blit_source();
        }
        match self
            .fsr
            .as_ref()
            .filter(|_| self.temporal && self.targets_ready)
        {
            Some(fsr) => (fsr.output_image(), fsr.output_view()),
            None => (self.targets.color.image, self.targets.color.view),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlitPush;

    #[test]
    fn blit_push_block_matches_wgsl_layout() {
        // Four 32-bit words: tonemap, exposure, dither, pad.
        assert_eq!(std::mem::size_of::<BlitPush>(), 16);
        assert_eq!(std::mem::offset_of!(BlitPush, exposure), 4);
        assert_eq!(std::mem::offset_of!(BlitPush, dither), 8);
    }
}

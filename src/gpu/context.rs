use std::sync::Arc;

use winit::window::Window;

use super::blur::BlurResources;
use super::glyph_atlas::GlyphAtlas;
use super::image_atlas::{ImageAtlas, ImageHandle};
use super::instance::{FrameUniform, ShapeInstance, SHAPE_KIND_GLYPH, SHAPE_KIND_IMAGE};
use super::overdraw::OverdrawResources;
use super::pipeline::ShapePipeline;
use crate::node::{ImageRef, TextRef};
use crate::text::TextResources;

// `TextResources` is owned by `SceneCtx`, not the renderer — shape/measure
// passes need it too, and keeping it scene-side avoids a borrow split.
use super::timing::{
    FrameTiming, PassAlloc, Timing, PASS_FINAL, PASS_OD_COMPOSE, PASS_OD_COUNT,
    PASS_OPAQUE,
};

/// Owns every wgpu handle the renderer touches.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,

    pub shape: ShapePipeline,
    pub blur: BlurResources,
    pub overdraw: OverdrawResources,
    pub glyph_atlas: GlyphAtlas,
    pub image_atlas: ImageAtlas,
    overdraw_mode: bool,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    shape_bg: wgpu::BindGroup,
    glass_bg: wgpu::BindGroup,

    /// Secondary instance buffer for debug overlays (HUD bar gauges,
    /// future inspector outlines). Drawn at the end of the final pass
    /// over the regular scene. Has no effect on the backdrop pass.
    overlay_buffer: wgpu::Buffer,
    overlay_capacity: u64,
    overlay_bg: wgpu::BindGroup,
    overlay_count: u32,

    instance_count: u32,
    /// Count of glass-kind instances in the most recent upload. Used
    /// to gate the backdrop+blur passes (no glass → skip entirely)
    /// and to populate `FrameStats.glass_count`. Non-glass count is
    /// derived as `instance_count - glass_count`.
    glass_count: u32,
    /// Mirror of the most recent instance list uploaded to the GPU.
    /// `set_instances` diffs against it to compute partial-upload
    /// ranges; cleared (then rebuilt) on buffer grow or when the slot
    /// count changes within the existing capacity.
    prev_instances: Vec<ShapeInstance>,
    /// Needs a re-render of the opaque pass into `backdrop_tex` on the
    /// next frame. Set by `set_instances`; cleared after render.
    backdrop_dirty: bool,

    /// Timestamp query resources. `Some` when the adapter advertises
    /// `Features::TIMESTAMP_QUERY`, `None` otherwise. Reads happen on
    /// demand via `take_last_timing`.
    timing: Option<Timing>,
    /// Render-pass + compute drawcall counter for the most recent frame.
    last_drawcalls: u32,
    /// Cached frame timing read at the end of the last render. `None`
    /// when timing isn't available or hasn't been read yet.
    last_timing: Option<FrameTiming>,
}

impl GpuContext {
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("no suitable adapter");

        let adapter_features = adapter.features();
        let want_timing = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = wgpu::Features::empty();
        if want_timing {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frostify-gfx device"),
                required_features,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device request failed");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                *f == wgpu::TextureFormat::Rgba8UnormSrgb
                    || *f == wgpu::TextureFormat::Bgra8UnormSrgb
            })
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::PreMultiplied)
            .unwrap_or(caps.alpha_modes[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        log::info!(
            "gpu init: format={format:?} alpha={alpha_mode:?} size={width}x{height}"
        );

        let glyph_atlas = GlyphAtlas::new(&device, 1024);
        // 2048² (16 MiB RGBA) so multiple 640px album covers coexist
        // without thrashing the eviction path. A single 640 cover nearly
        // fills a 1024² atlas, forcing a repack on every track change.
        let image_atlas = ImageAtlas::new(&device, 2048);
        let shape = ShapePipeline::new(
            &device,
            format,
            glyph_atlas.layout(),
            image_atlas.layout(),
        );
        let blur = BlurResources::new(&device, width, height);
        let overdraw =
            OverdrawResources::new(&device, width, height, format, &shape.shape_bgl);
        let timing = if want_timing {
            Some(Timing::new(&device, &queue))
        } else {
            None
        };
        log::info!("gpu timing: {}", if timing.is_some() { "on" } else { "off" });

        // Allocate an initial instance buffer with room for one shape.
        let instance_capacity: u64 = 16;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.instance ssbo"),
            size: instance_capacity * std::mem::size_of::<ShapeInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shape_bg = make_shape_bg(&device, &shape, &instance_buffer);
        let glass_bg = make_glass_bg(&device, &shape, &blur);

        let overlay_capacity: u64 = 16;
        let overlay_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.overlay ssbo"),
            size: overlay_capacity * std::mem::size_of::<ShapeInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_bg = make_shape_bg(&device, &shape, &overlay_buffer);

        // Write initial frame uniform.
        queue.write_buffer(
            &shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [width as f32, height as f32],
                max_backdrop_lod: blur.mip_count().saturating_sub(1) as f32,
                _pad: 0.0,
            }),
        );

        Self {
            instance,
            device,
            queue,
            surface,
            surface_config,
            window,
            shape,
            blur,
            overdraw,
            glyph_atlas,
            image_atlas,
            overdraw_mode: false,
            instance_buffer,
            instance_capacity,
            shape_bg,
            glass_bg,
            overlay_buffer,
            overlay_capacity,
            overlay_bg,
            overlay_count: 0,
            instance_count: 0,
            glass_count: 0,
            prev_instances: Vec::new(),
            backdrop_dirty: true,
            timing,
            last_drawcalls: 0,
            last_timing: None,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.blur
            .resize(&self.device, self.surface_config.width, self.surface_config.height);
        self.queue.write_buffer(
            &self.shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [
                    self.surface_config.width as f32,
                    self.surface_config.height as f32,
                ],
                max_backdrop_lod: self.blur.mip_count().saturating_sub(1) as f32,
                _pad: 0.0,
            }),
        );
        // Blurred view changed — rebuild the glass bind group.
        self.glass_bg = make_glass_bg(&self.device, &self.shape, &self.blur);
        self.overdraw.resize(
            &self.device,
            self.surface_config.width,
            self.surface_config.height,
        );
        self.backdrop_dirty = true;
    }

    pub fn overdraw_mode(&self) -> bool {
        self.overdraw_mode
    }

    pub fn set_overdraw(&mut self, on: bool) {
        self.overdraw_mode = on;
    }

    /// Upload a complete instance list in painter's (declared) order.
    /// Both passes draw the same range; `fs_opaque` discards glass so
    /// it stays out of the backdrop, while every other kind enters the
    /// blurred backdrop and shows up behind glass panels.
    /// `glass_count` is the number of `SHAPE_KIND_GLASS` entries in
    /// the slice — used to gate the backdrop+blur passes and to
    /// populate frame stats. `backdrop_hint` is OR'd into the
    /// existing `backdrop_dirty` state and cleared when the blur
    /// pass runs.
    pub fn set_instances(
        &mut self,
        instances: &[ShapeInstance],
        glass_count: u32,
        backdrop_hint: bool,
    ) {
        let needed = instances.len() as u64;
        let stride = std::mem::size_of::<ShapeInstance>() as u64;
        let grew = needed > self.instance_capacity;
        if grew {
            let mut new_cap = self.instance_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frostify.instance ssbo"),
                size: new_cap * stride,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
            self.shape_bg = make_shape_bg(&self.device, &self.shape, &self.instance_buffer);
        }

        // Full upload on buffer grow or on any slot-count change (new
        // instance count ≠ cached count) — slot indices may have shifted
        // so per-slot diffing isn't safe. Otherwise diff byte-wise
        // against `prev_instances` and coalesce contiguous dirty ranges
        // into individual `write_buffer` calls.
        if grew || instances.len() != self.prev_instances.len() {
            if !instances.is_empty() {
                self.queue
                    .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
            }
            self.prev_instances.clear();
            self.prev_instances.extend_from_slice(instances);
        } else {
            let mut i = 0;
            while i < instances.len() {
                if bytemuck::bytes_of(&instances[i])
                    == bytemuck::bytes_of(&self.prev_instances[i])
                {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < instances.len()
                    && bytemuck::bytes_of(&instances[i])
                        != bytemuck::bytes_of(&self.prev_instances[i])
                {
                    i += 1;
                }
                let end = i;
                self.queue.write_buffer(
                    &self.instance_buffer,
                    (start as u64) * stride,
                    bytemuck::cast_slice(&instances[start..end]),
                );
                self.prev_instances[start..end].copy_from_slice(&instances[start..end]);
            }
        }

        self.instance_count = instances.len() as u32;
        self.glass_count = glass_count.min(self.instance_count);
        if backdrop_hint {
            self.backdrop_dirty = true;
        }
    }

    pub fn glass_count(&self) -> u32 {
        self.glass_count
    }

    /// Shape + rasterize each [`TextRef`] into glyph-kind shape
    /// instances. Glyphs that miss the atlas cache are uploaded here
    /// via `queue.write_texture`. Output is meant to be appended after
    /// the existing glass instances (so it all draws in the final pass).
    pub fn build_glyph_instances(
        &mut self,
        text: &mut TextResources,
        refs: &[TextRef],
    ) -> Vec<ShapeInstance> {
        if refs.is_empty() {
            return Vec::new();
        }
        let atlas_size = self.glyph_atlas.size() as f32;
        let mut out = Vec::new();
        for r in refs {
            if r.content.is_empty() {
                continue;
            }
            let shaped = match r.max_width {
                Some(mw) => text.shape_constrained(&r.content, r.font_size, r.line_height, mw),
                None => text.shape(&r.content, r.font_size, r.line_height),
            };
            for g in shaped {
                let Some(entry) =
                    self.glyph_atlas
                        .get_or_insert(&self.queue, text, g.cache_key)
                else {
                    continue;
                };
                if entry.width == 0 || entry.height == 0 {
                    continue;
                }
                let px = r.position[0] + g.x as f32 + entry.left as f32;
                let py = r.position[1] + g.y as f32 - entry.top as f32;
                let uv_w = entry.width as f32 / atlas_size;
                let uv_h = entry.height as f32 / atlas_size;
                out.push(ShapeInstance {
                    color: r.color,
                    border_color: [0.0; 4],
                    shadow_color: [0.0; 4],
                    border_radius: [0.0; 4],
                    backdrop_uv_rect: [entry.uv[0], entry.uv[1], uv_w, uv_h],
                    clip_rect: r.clip_rect,
                    position: [px, py],
                    size: [entry.width as f32, entry.height as f32],
                    shadow_offset: [0.0; 2],
                    shape_kind: SHAPE_KIND_GLYPH,
                    roughness: 0.0,
                    border_width: 0.0,
                    shadow_blur: 0.0,
                    shadow_opacity: 0.0,
                    opacity: r.opacity,
                    scale: [1.0, 1.0],
                    _pad1: [0.0, 0.0],
                });
            }
        }
        out
    }

    /// Resolve each [`ImageRef`] against the image atlas, emitting one
    /// `SHAPE_KIND_IMAGE` instance per known handle. Stale handles
    /// (atlas reset / never uploaded) are silently skipped — caller
    /// already chose to render them, missing texture would be worse.
    pub fn build_image_instances(&self, refs: &[ImageRef]) -> Vec<ShapeInstance> {
        if refs.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(refs.len());
        for r in refs {
            let Some(entry) = self.image_atlas.get(r.handle) else {
                continue;
            };
            let uv_w = entry.uv[2] - entry.uv[0];
            let uv_h = entry.uv[3] - entry.uv[1];
            out.push(ShapeInstance {
                color: r.color,
                border_color: [0.0; 4],
                shadow_color: [0.0; 4],
                border_radius: r.border_radius,
                backdrop_uv_rect: [entry.uv[0], entry.uv[1], uv_w, uv_h],
                clip_rect: r.clip_rect,
                position: r.position,
                size: r.size,
                shadow_offset: [0.0; 2],
                shape_kind: SHAPE_KIND_IMAGE,
                roughness: 0.0,
                border_width: 0.0,
                shadow_blur: 0.0,
                shadow_opacity: 0.0,
                opacity: r.opacity,
                scale: [1.0, 1.0],
                _pad1: [0.0, 0.0],
            });
        }
        out
    }

    /// Upload a list of overlay instances drawn after the main scene.
    /// Pass an empty slice to clear. Same growth scheme as the main
    /// instance buffer.
    pub fn set_overlay_instances(&mut self, instances: &[ShapeInstance]) {
        let needed = instances.len() as u64;
        if needed > self.overlay_capacity {
            let mut new_cap = self.overlay_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.overlay_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frostify.overlay ssbo"),
                size: new_cap * std::mem::size_of::<ShapeInstance>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.overlay_capacity = new_cap;
            self.overlay_bg = make_shape_bg(&self.device, &self.shape, &self.overlay_buffer);
        }
        if !instances.is_empty() {
            self.queue
                .write_buffer(&self.overlay_buffer, 0, bytemuck::cast_slice(instances));
        }
        self.overlay_count = instances.len() as u32;
    }

    pub fn mark_backdrop_dirty(&mut self) {
        self.backdrop_dirty = true;
    }

    /// Drop every cached glyph from the atlas. Call when the
    /// physical glyph size changes (DPI scale flip) — old cache_keys
    /// are tied to a specific size, new ones won't match. The next
    /// `build_glyph_instances` call refills lazily.
    pub fn reset_glyph_atlas(&mut self) {
        self.glyph_atlas.reset();
    }

    /// Decode a PNG and upload it into the image atlas. Returns a
    /// handle the scene can pass to [`crate::node::Node::image`]. Fails
    /// (returns `None`) on malformed PNG, oversize image, or atlas
    /// exhaustion.
    pub fn upload_image_png(&mut self, bytes: &[u8]) -> Option<ImageHandle> {
        self.image_atlas.upload_png(&self.queue, bytes)
    }

    /// Encode the opaque pass, downsample dispatches (if needed), and
    /// final pass into `encoder`. `final_view` is the render target for
    /// the surface pass.
    fn encode_frame(&mut self, encoder: &mut wgpu::CommandEncoder, final_view: &wgpu::TextureView) {
        let mut drawcalls: u32 = 0;
        let timing_qs = self.timing.as_ref().map(|t| &t.query_set);
        let mut alloc = PassAlloc::new();
        // The opaque pass exists only to populate `backdrop_tex` mip 0
        // (and the rest of the pyramid via downsample) for glass shapes
        // to sample in the final pass. If there's no glass, the backdrop
        // is never sampled; if backdrop content hasn't changed since the
        // last submit the existing pyramid is still valid.
        let has_glass = self.glass_count() > 0;
        let run_backdrop = has_glass && self.backdrop_dirty;
        // Pre-allocate query pairs for every pass that will run this
        // frame. The pair indices are dense so `resolve_query_set` can
        // cover a contiguous prefix.
        let (opaque_begin, opaque_end) = match (timing_qs, run_backdrop) {
            (Some(_), true) => {
                let (b, e) = alloc.alloc(PASS_OPAQUE);
                (Some(b), Some(e))
            }
            _ => (None, None),
        };
        let (final_begin, final_end) = match timing_qs {
            Some(_) => {
                let (b, e) = alloc.alloc(PASS_FINAL);
                (Some(b), Some(e))
            }
            None => (None, None),
        };
        let (od_count_begin, od_count_end) = match (timing_qs, self.overdraw_mode) {
            (Some(_), true) => {
                let (b, e) = alloc.alloc(PASS_OD_COUNT);
                (Some(b), Some(e))
            }
            _ => (None, None),
        };
        let (od_compose_begin, od_compose_end) = match (timing_qs, self.overdraw_mode) {
            (Some(_), true) => {
                let (b, e) = alloc.alloc(PASS_OD_COMPOSE);
                (Some(b), Some(e))
            }
            _ => (None, None),
        };

        // ---- Pass A: opaque-class shapes → backdrop_tex ------------------
        // Draws the full instance list — `fs_opaque` discards glass
        // instances so they stay out of the blur input. Painter's
        // order is preserved so layered text/images appear in the
        // backdrop in the same order they appear on the surface.
        // Skipped when no glass exists (nothing samples backdrop_tex)
        // or when the prior submit's backdrop is still valid.
        if run_backdrop {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.backdrop pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur.backdrop_mip0_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: opaque_begin,
                    end_of_pass_write_index: opaque_end,
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count > 0 {
                rpass.set_pipeline(&self.shape.opaque_pipeline);
                rpass.set_bind_group(0, &self.shape_bg, &[]);
                rpass.set_bind_group(2, self.glyph_atlas.bind_group(), &[]);
                rpass.set_bind_group(3, self.image_atlas.bind_group(), &[]);
                rpass.draw(0..6, 0..self.instance_count);
                drawcalls += 1;
            }
        }

        // Downsample the freshly-rendered mip 0 into the rest of the
        // backdrop pyramid. Glass shapes sample this with fractional
        // LOD for variable-radius blur.
        if run_backdrop {
            self.blur.run_downsample(encoder);
            self.backdrop_dirty = false;
        }

        // ---- Pass C: final surface ------------------------------------
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.final pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: final_begin,
                    end_of_pass_write_index: final_end,
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count > 0 {
                rpass.set_pipeline(&self.shape.final_pipeline);
                rpass.set_bind_group(0, &self.shape_bg, &[]);
                rpass.set_bind_group(1, &self.glass_bg, &[]);
                rpass.set_bind_group(2, self.glyph_atlas.bind_group(), &[]);
                rpass.set_bind_group(3, self.image_atlas.bind_group(), &[]);
                rpass.draw(0..6, 0..self.instance_count);
                drawcalls += 1;
            }
            if self.overlay_count > 0 {
                rpass.set_pipeline(&self.shape.final_pipeline);
                rpass.set_bind_group(0, &self.overlay_bg, &[]);
                rpass.set_bind_group(1, &self.glass_bg, &[]);
                rpass.set_bind_group(2, self.glyph_atlas.bind_group(), &[]);
                rpass.set_bind_group(3, self.image_atlas.bind_group(), &[]);
                rpass.draw(0..6, 0..self.overlay_count);
                drawcalls += 1;
            }
        }

        // ---- Pass D (optional): overdraw count + compose --------------
        // When toggled on, count shape coverage into an Rgba16Float
        // accumulator, then re-render the swapchain with a heatmap of the
        // count. The final pass already cleared and drew the scene; the
        // compose pass overwrites it with the heatmap (LoadOp::Clear).
        if self.overdraw_mode {
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("frostify.overdraw count"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.overdraw.count_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: od_count_begin,
                        end_of_pass_write_index: od_count_end,
                    }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if self.instance_count > 0 {
                    rpass.set_pipeline(&self.overdraw.count_pipeline);
                    rpass.set_bind_group(0, &self.shape_bg, &[]);
                    rpass.draw(0..6, 0..self.instance_count);
                    drawcalls += 1;
                }
            }
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("frostify.overdraw compose"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: final_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: od_compose_begin,
                        end_of_pass_write_index: od_compose_end,
                    }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.overdraw.compose_pipeline);
                rpass.set_bind_group(0, &self.overdraw.compose_bg, &[]);
                rpass.draw(0..6, 0..1);
                drawcalls += 1;
            }
        }

        if let Some(t) = self.timing.as_mut() {
            t.encode_resolve(encoder, alloc);
        }

        self.last_drawcalls = drawcalls;
    }

    /// Acquire, render, present.
    pub fn render_frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("surface validation error");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frostify-gfx frame"),
            });

        self.encode_frame(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.poll_timing_after_submit();
        self.window.pre_present_notify();
        frame.present();
    }

    /// Kick async map for the slot the most recent `encode_frame` wrote
    /// into, then non-blocking poll. Updates `last_timing` in-place
    /// with whatever slot completed this tick (possibly a prior frame).
    fn poll_timing_after_submit(&mut self) {
        let Some(t) = self.timing.as_mut() else {
            self.last_timing = None;
            return;
        };
        t.kick_map_async();
        t.poll(&self.device);
        self.last_timing = t.last();
    }

    /// Last-frame stats. Drawcall + timing values come from the encoder /
    /// query readback; instance counts mirror the most recent
    /// `set_instances` call.
    pub fn last_frame_stats(&self) -> super::timing::FrameStats {
        let t = self.last_timing.unwrap_or_default();
        super::timing::FrameStats {
            cpu_ms: 0.0,
            gpu_ms: t.total_ms,
            opaque_ms: t.opaque_ms,
            final_ms: t.final_ms,
            overdraw_ms: t.overdraw_ms,
            instance_count: self.instance_count,
            // Non-glass count = everything that enters the backdrop
            // pass. Reported as `opaque_count` for back-compat with the
            // FrameStats struct (renaming would ripple through sidecar
            // sidecar consumers).
            opaque_count: self.instance_count - self.glass_count,
            glass_count: self.glass_count,
            drawcalls: self.last_drawcalls,
            dirty_mask: 0,
        }
    }

    /// True when `Features::TIMESTAMP_QUERY` is active and `last_frame_stats`
    /// will return a meaningful `gpu_ms`.
    pub fn timing_enabled(&self) -> bool {
        self.timing.is_some()
    }

    /// Snapshot of currently-allocated GPU-backed memory. Counts the
    /// instance + overlay SSBOs, blur/overdraw textures, timing
    /// query/readback buffers, and the CPU-side `prev_instances`
    /// shadow. Values reflect *allocated* capacity, not in-use size.
    pub fn memory_report(&self) -> MemoryReport {
        let stride = std::mem::size_of::<ShapeInstance>() as u64;
        let (bw, bh) = self.blur.resolution();
        let blur_px = bw as u64 * bh as u64;
        // Mipmap pyramid: each level is 1/4 the previous. Geometric
        // series sum ≈ 4/3 of base. Rgba8Unorm = 4 B/px.
        let blur_textures = (blur_px * 4 * 4) / 3;
        let (ow, oh) = self.overdraw.resolution();
        // 1 texture, Rgba16Float → 8 B/px.
        let overdraw_textures = (ow as u64) * (oh as u64) * 8;
        let params_buffers: u64 = 0;
        // Timing: 1× resolve (256) + 2× readback (256 each) when active.
        let timing = if self.timing.is_some() { 256 * 3 } else { 0 };
        MemoryReport {
            instance_buffer: self.instance_capacity * stride,
            overlay_buffer: self.overlay_capacity * stride,
            prev_instances_cpu: (self.prev_instances.capacity() as u64) * stride,
            blur_textures,
            overdraw_textures,
            timing,
            params_buffers,
            glyph_atlas: self.glyph_atlas.memory_bytes(),
            image_atlas: self.image_atlas.memory_bytes(),
            image_sources_cpu: self.image_atlas.source_bytes(),
        }
    }

    /// Render one frame into an offscreen RGBA texture and return raw
    /// pixels + dimensions. Used by the F2 screenshot path. Blocks on the
    /// GPU map. Non-hot path.
    pub fn capture_rgba(&mut self) -> (Vec<u8>, u32, u32) {
        let width = self.surface_config.width;
        let height = self.surface_config.height;

        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify.capture target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Readback buffer. Row pitch must be 256-aligned (COPY_BYTES_PER_ROW_ALIGNMENT).
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let readback_size = (padded_bpr as u64) * height as u64;

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.capture readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frostify.capture encoder"),
            });

        self.encode_frame(&mut encoder, &view);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        self.poll_timing_after_submit();

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv()
            .expect("map channel closed")
            .expect("map failed");

        let view = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded_bpr * height) as usize);
        for row in 0..height {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            out.extend_from_slice(&view[start..end]);
        }
        drop(view);
        readback.unmap();

        if matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in out.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }

        (out, width, height)
    }
}

/// Breakdown of currently-allocated GPU memory in bytes. Reported
/// values reflect buffer/texture *capacity*, not in-use counts — this
/// is a ceiling for debug/profiling, not an exact live watermark.
#[derive(Copy, Clone, Debug, Default)]
pub struct MemoryReport {
    pub instance_buffer: u64,
    pub overlay_buffer: u64,
    pub prev_instances_cpu: u64,
    pub blur_textures: u64,
    pub overdraw_textures: u64,
    pub timing: u64,
    pub params_buffers: u64,
    pub glyph_atlas: u64,
    pub image_atlas: u64,
    /// CPU-side cache of source bytes for every uploaded image
    /// (`ImageAtlas::source_bytes()`). Required so the eviction path
    /// can re-pack survivors when the atlas fills.
    pub image_sources_cpu: u64,
}

impl MemoryReport {
    pub fn total(&self) -> u64 {
        self.instance_buffer
            + self.overlay_buffer
            + self.prev_instances_cpu
            + self.blur_textures
            + self.overdraw_textures
            + self.timing
            + self.params_buffers
            + self.glyph_atlas
            + self.image_atlas
            + self.image_sources_cpu
    }
}

fn make_shape_bg(
    device: &wgpu::Device,
    shape: &ShapePipeline,
    instance_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frostify.shape bg"),
        layout: &shape.shape_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: shape.frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instance_buffer.as_entire_binding(),
            },
        ],
    })
}

fn make_glass_bg(
    device: &wgpu::Device,
    shape: &ShapePipeline,
    blur: &BlurResources,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frostify.glass bg"),
        layout: &shape.glass_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&blur.backdrop_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&blur.sampler),
            },
        ],
    })
}

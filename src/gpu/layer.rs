//! GPU-side compositor resources (P2 + P3 scroll).
//!
//! Each [`crate::layer::Layer`] caches a node subtree's pixels in an
//! offscreen color texture; [`LayerResources`] owns those textures plus
//! the **composite** pipeline (`shaders/composite.wgsl`) that blits them
//! to the surface in z-order.
//!
//! Two kinds of layer:
//!   - **identity / root layers** — full-surface-sized texture, rastered
//!     in absolute screen coords against the *global* shape bind group
//!     (frame UBO = surface size, corner clip applied in `fs_main`).
//!     Composited at `offset`/`scale`/`opacity` over the surface. This is
//!     the P2/P3 parity path and is byte-exact with the old final pass.
//!   - **scroll layers (P3 2a)** — a possibly **taller** texture sized to
//!     the scroll content. Their subtree rasters in **content-local**
//!     coords, so the texture must be mapped with the *texture's* size,
//!     not the surface — hence a **per-layer frame UBO** (`screen_size =
//!     tex_size`, `window_corner_radius = 0`) + a per-layer raster bind
//!     group pairing it with the shared instance SSBO. The composite then
//!     samples a 1:1-px **window** of that texture at the scroll offset
//!     and scissor-clips to the container rect ([`ScrollWindow`]).
//!
//! Premultiplied-over blend (`One, OneMinusSrcAlpha`): layers stack
//! correctly, and a single identity layer over a transparent surface is a
//! byte-exact passthrough.
//!
//! The GPU win — skipping the raster on composite-only / scroll frames —
//! comes from the per-slot content signature (see `GpuContext`): a scroll
//! layer's content-local instances don't change as it scrolls, so its
//! signature is stable and the raster pass is skipped; only the composite
//! window (`src_origin`) moves.

use std::ops::Range;

use super::instance::{FrameUniform, NO_CLIP};

/// Per-layer composite parameters uploaded to `composite.wgsl`. 96 B,
/// 16-aligned (the six leading `vec2`s fill 0..48 so the `vec4`
/// `clip_rect` lands on its 16-byte boundary; a 16-byte scalar row, then
/// the trailing `vec4` `edge_fade`). Assert in tests.
///
/// The quad covers screen rect `dst_origin .. dst_origin + dst_size`
/// and samples the texture window `src_origin .. src_origin + src_extent`
/// (in `tex_size` px), discarding fragments outside `clip_rect`. The
/// **full-surface identity** case (`dst_origin = offset`, `dst_size =
/// surface*scale`, `src_origin = 0`, `src_extent = tex_size = surface`,
/// `clip = NO_CLIP`) yields `uv = corner` with no clip — byte-exact P2
/// parity + the P3 offset/scale/opacity composite-move.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniform {
    dst_origin: [f32; 2],
    dst_size: [f32; 2],
    src_origin: [f32; 2],
    src_extent: [f32; 2],
    tex_size: [f32; 2],
    surface_size: [f32; 2],
    clip_rect: [f32; 4],
    opacity: f32,
    /// Rounds the clip rect's corners (physical px) — external video
    /// layers honouring a rounded container. 0 = square (every other layer).
    corner_radius: f32,
    /// Edge-fade falloff exponent (1 = linear).
    edge_fade_falloff: f32,
    _pad: f32,
    /// Per-edge fade fractions `[top, right, bottom, left]` (0..1 of the
    /// dst rect extent). Fade alpha to 0 near each edge. All-0 = no fade.
    edge_fade: [f32; 4],
}

/// Composite-time window for a **scroll layer**: place the quad at a
/// screen rect and sample a `dst_size`-sized (1:1 px) window of a
/// possibly-tall layer texture starting at `src_origin` (the scroll
/// offset), clipped to `clip_rect`. `None` on a [`LayerDraw`] selects
/// the full-surface identity path (the whole texture stretches over the
/// offset/scaled surface quad — P2/P3 semantics).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollWindow {
    /// Screen-space top-left of the composite quad (physical px).
    pub dst_origin: [f32; 2],
    /// Screen-space size of the composite quad — the container viewport.
    pub dst_size: [f32; 2],
    /// Content-space sample origin (physical px) — the scroll offset.
    pub src_origin: [f32; 2],
    /// Layer texture dimensions (physical px); typically `[w, content_h]`.
    pub tex_size: [f32; 2],
    /// Screen-space scissor `(min_x, min_y, max_x, max_y)`.
    pub clip_rect: [f32; 4],
}

/// One layer's per-frame draw description: which instance sub-range to
/// rasterize into the layer texture, plus the composite transform /
/// opacity / z-order. Built CPU-side from [`crate::layer::LayerTree`]
/// and handed to `GpuContext::set_layers`.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerDraw {
    /// Painter-order slice of the global instance buffer this layer owns.
    pub instances: Range<u32>,
    /// Composite-time screen offset (physical px). Full-surface path only.
    pub offset: [f32; 2],
    /// Composite-time scale about the layer's top-left. Identity = 1.
    /// Full-surface path only.
    pub scale: [f32; 2],
    /// Composite-time opacity multiplier.
    pub opacity: f32,
    /// Composite z-order; lower composites first (further back).
    pub z: i32,
    /// `Some` → composite a scrolled window of a tall texture with a
    /// scissor clip + raster the subtree content-local (scroll layer);
    /// `None` → full-surface identity using `offset`/`scale` (parity
    /// path), rastered in absolute screen coords via the global shape BG.
    pub window: Option<ScrollWindow>,
    /// `Some(node)` → **external-texture layer** (P6): no raster; the
    /// composite samples the caller's texture registered under `node`
    /// (via `GpuContext::set_external_texture`) instead of this slot's own
    /// raster texture. The composite quad still uses `window` for its
    /// screen rect. `None` for every normal layer.
    pub external: Option<crate::node::NodeId>,
    /// Corner radius (physical px) the composite rounds the clipped region
    /// by — lets an external (video) layer honour a rounded container. 0 =
    /// square. Only consulted for external layers.
    pub corner_radius: f32,
    /// Composite-time edge fade `[top, right, bottom, left]` (0..1 of the
    /// rect extent on each axis) — fade alpha to 0 near those edges. Any
    /// promoted layer. All-0 = no fade.
    pub edge_fade: [f32; 4],
    /// Falloff exponent for the edge fade (1 = linear).
    pub edge_fade_falloff: f32,
}

impl Default for LayerDraw {
    fn default() -> Self {
        LayerDraw {
            instances: 0..0,
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            opacity: 1.0,
            z: 0,
            window: None,
            external: None,
            corner_radius: 0.0,
            edge_fade: [0.0; 4],
            edge_fade_falloff: 1.0,
        }
    }
}

impl LayerDraw {
    /// Physical-pixel size this layer's texture must be: the window's
    /// `tex_size` for a scroll layer, else the surface size (identity
    /// layers are full-surface-sized for byte-parity).
    pub(crate) fn texture_size(&self, surface: [u32; 2]) -> [u32; 2] {
        match self.window {
            Some(w) => [
                (w.tex_size[0].ceil() as u32).max(1),
                (w.tex_size[1].ceil() as u32).max(1),
            ],
            None => surface,
        }
    }
}

struct LayerTexture {
    // The `wgpu::Texture` handle is intentionally not stored: the view
    // holds a strong ref to the underlying texture, so it stays alive.
    view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Per-layer frame UBO for the **scroll-layer** raster path
    /// (`screen_size = this texture's size`, corner clip = 0). Unused by
    /// identity layers (they raster against the global shape BG).
    frame: wgpu::Buffer,
    /// Shape bind group pairing `frame` (@0) with the shared instance
    /// SSBO (@1). Rebuilt when the instance buffer grows.
    raster_bg: wgpu::BindGroup,
    /// Physical-pixel size of this slot's texture.
    size: [u32; 2],
}

/// Owns the composite pipeline + one offscreen texture per live layer.
pub struct LayerResources {
    pipeline: wgpu::RenderPipeline,
    /// Same composite pipeline targeting the **linear backdrop format**
    /// (`BACKDROP_FORMAT`) instead of the sRGB surface format. P4 generic
    /// glass composites the below-glass content into the linear
    /// `backdrop_tex` (blur input) with this; the premultiplied-over math
    /// is identical, only the render-target format differs.
    backdrop_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// Shape bind-group layout (frame UBO + instance SSBO), cloned from
    /// the shape pipeline so per-layer raster bind groups can be (re)built
    /// here without threading it through every call.
    shape_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Linear-filtered composite sampler, used **only** for external-texture
    /// layers (video / Canvas). Their source texture is at the decoder's
    /// resolution, not the on-screen rect, so the composite quad scales it —
    /// nearest sampling there aliases hard on diagonals. Normal UI layers
    /// stay on the nearest [`Self::sampler`] (rastered 1:1, keeps text crisp).
    linear_sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    /// Default texture size for identity layers (= current surface size).
    surface_size: [u32; 2],
    textures: Vec<LayerTexture>,
    /// Content signature of the instance bytes last rastered into each
    /// slot's texture (P3 raster-skip). `None` = invalid (newly
    /// allocated / resized) → must raster. A slot whose signature
    /// matches the current frame's reuses its texture (raster skipped).
    sigs: Vec<Option<u64>>,
}

impl LayerResources {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        shape_bgl: &wgpu::BindGroupLayout,
        instance_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frostify.composite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frostify.composite bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CompositeUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Identity composite is exact; the layer texture
                        // is the surface format (filterable).
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frostify.composite pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // Premultiplied-over: layers stack; a single layer over a
        // transparent surface passes through unchanged.
        let blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        });

        let make_pipeline = |target: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline(format, "frostify.composite pipeline");
        // P4: composites below-glass content into the linear backdrop
        // texture (blur input). Unused until the segment walk lands.
        let backdrop_pipeline = make_pipeline(
            super::blur::BACKDROP_FORMAT,
            "frostify.composite→backdrop pipeline",
        );

        // Nearest sampling: a 1:1-px composite window maps the quad to the
        // layer texels exactly (identity OR a pixel-aligned scroll window),
        // so nearest is byte-exact and avoids edge bleed at the window
        // boundary. Sub-pixel scroll offsets land between texels; nearest
        // snaps, which reads crisp (no half-pixel smear).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify.composite sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // Linear sampler for external (video / Canvas) layers — see the
        // `linear_sampler` field doc. Clamp to edge so the bilinear tap at
        // the frame border doesn't wrap.
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify.composite sampler (linear)"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let mut res = LayerResources {
            pipeline,
            backdrop_pipeline,
            bgl,
            shape_bgl: shape_bgl.clone(),
            sampler,
            linear_sampler,
            format,
            surface_size: [width.max(1), height.max(1)],
            textures: Vec::new(),
            sigs: Vec::new(),
        };
        res.ensure(device, &[res.surface_size], instance_buffer);
        res
    }

    fn make_texture(
        &self,
        device: &wgpu::Device,
        size: [u32; 2],
        instance_buffer: &wgpu::Buffer,
    ) -> LayerTexture {
        let size = [size[0].max(1), size[1].max(1)];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify.layer texture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.composite ubo"),
            size: std::mem::size_of::<CompositeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frostify.composite bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.layer frame ubo"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_bg = make_raster_bg(device, &self.shape_bgl, &frame, instance_buffer);
        LayerTexture {
            view,
            uniform,
            bind_group,
            frame,
            raster_bg,
            size,
        }
    }

    /// Grow / shrink / resize the texture pool so slot `i` has a texture
    /// of `sizes[i]` (physical px). Slots whose size already matches are
    /// reused (no realloc, signature preserved); resized slots are rebuilt
    /// and their signature invalidated. Always at least one slot.
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        sizes: &[[u32; 2]],
        instance_buffer: &wgpu::Buffer,
    ) {
        let count = sizes.len().max(1);
        if self.textures.len() > count {
            self.textures.truncate(count);
            self.sigs.truncate(count);
        }
        for i in 0..count {
            let want = sizes.get(i).copied().unwrap_or(self.surface_size);
            let want = [want[0].max(1), want[1].max(1)];
            match self.textures.get(i) {
                Some(t) if t.size == want => {}
                Some(_) => {
                    self.textures[i] = self.make_texture(device, want, instance_buffer);
                    self.sigs[i] = None;
                }
                None => {
                    let t = self.make_texture(device, want, instance_buffer);
                    self.textures.push(t);
                    self.sigs.push(None);
                }
            }
        }
    }

    /// Record the new surface size (identity-layer default). Textures are
    /// re-sized lazily by [`Self::ensure`] on the next frame from the live
    /// draw list, so this just updates the default + drops every cached
    /// texture (surface changed → all content invalid).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_size = [width.max(1), height.max(1)];
        self.textures.clear();
        self.sigs.clear();
    }

    /// Repoint every per-layer raster bind group at a freshly-grown
    /// instance buffer. Call from `set_instances` whenever the instance
    /// SSBO is reallocated.
    pub fn rebuild_raster_bgs(&mut self, device: &wgpu::Device, instance_buffer: &wgpu::Buffer) {
        for t in &mut self.textures {
            t.raster_bg = make_raster_bg(device, &self.shape_bgl, &t.frame, instance_buffer);
        }
    }

    /// Whether slot `i` must re-raster: forced (e.g. backdrop changed for
    /// a glass-bearing layer), or its content signature differs from what
    /// is cached in the texture. A match means the cached texture is
    /// still valid → skip the raster pass and reuse it.
    pub fn needs_raster(&self, i: usize, sig: u64, force: bool) -> bool {
        force || self.sigs.get(i).copied().flatten() != Some(sig)
    }

    /// Record that slot `i`'s texture now holds the render of `sig`.
    pub fn mark_rastered(&mut self, i: usize, sig: u64) {
        if let Some(slot) = self.sigs.get_mut(i) {
            *slot = Some(sig);
        }
    }

    /// Upload layer `i`'s composite parameters for this frame.
    /// `surface_size` is the physical swapchain size (px → NDC map).
    pub fn write_uniform(
        &self,
        queue: &wgpu::Queue,
        i: usize,
        draw: &LayerDraw,
        surface_size: [f32; 2],
    ) {
        let u = match draw.window {
            // Scroll layer: quad at the container rect, sampling a 1:1-px
            // window of the tall texture at the scroll offset, clipped to
            // the container rect.
            Some(w) => CompositeUniform {
                dst_origin: w.dst_origin,
                dst_size: w.dst_size,
                src_origin: w.src_origin,
                src_extent: w.dst_size,
                tex_size: w.tex_size,
                surface_size,
                clip_rect: w.clip_rect,
                opacity: draw.opacity,
                corner_radius: draw.corner_radius,
                edge_fade_falloff: draw.edge_fade_falloff,
                _pad: 0.0,
                edge_fade: draw.edge_fade,
            },
            // Full-surface identity (P2/P3 parity): quad = offset +
            // corner*surface*scale, uv = corner (whole texture). The
            // layer texture is surface-sized, so tex_size = surface.
            None => CompositeUniform {
                dst_origin: draw.offset,
                dst_size: [
                    surface_size[0] * draw.scale[0],
                    surface_size[1] * draw.scale[1],
                ],
                src_origin: [0.0, 0.0],
                src_extent: surface_size,
                tex_size: surface_size,
                surface_size,
                clip_rect: NO_CLIP,
                opacity: draw.opacity,
                corner_radius: 0.0,
                edge_fade_falloff: draw.edge_fade_falloff,
                _pad: 0.0,
                edge_fade: draw.edge_fade,
            },
        };
        queue.write_buffer(&self.textures[i].uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Upload slot `i`'s per-layer frame UBO for the scroll-layer raster
    /// path: `screen_size` = the texture size so content-local instance
    /// positions map into the (possibly tall) texture; corner clip is 0
    /// (the window edge isn't the surface edge). `max_lod` is the backdrop
    /// pyramid max LOD (unused — scroll layers are glass-free — but kept
    /// correct for safety).
    pub fn write_frame(&self, queue: &wgpu::Queue, i: usize, screen_size: [f32; 2], max_lod: f32) {
        let u = FrameUniform {
            screen_size,
            max_backdrop_lod: max_lod,
            window_corner_radius: 0.0,
        };
        queue.write_buffer(&self.textures[i].frame, 0, bytemuck::bytes_of(&u));
    }

    /// Render-target view for layer `i` (the raster pass writes here).
    pub fn raster_view(&self, i: usize) -> &wgpu::TextureView {
        &self.textures[i].view
    }

    /// Per-layer raster bind group (frame UBO @0 + instance SSBO @1) —
    /// bound at group 0 when rasterizing a **scroll layer** in
    /// content-local space.
    pub fn raster_bg(&self, i: usize) -> &wgpu::BindGroup {
        &self.textures[i].raster_bg
    }

    /// Composite bind group for layer `i` (sampled in the composite pass).
    pub fn bind_group(&self, i: usize) -> &wgpu::BindGroup {
        &self.textures[i].bind_group
    }

    /// Build an ad-hoc composite bind group pairing slot `i`'s composite
    /// uniform (this frame's offset/window) with an **external** texture
    /// view + the composite sampler. P6: an external-texture layer
    /// composites the caller's view (a video frame) through the slot's
    /// uniform, instead of the slot's own raster texture. Built per-frame
    /// (the external view changes as the decoder swaps frames).
    pub fn external_bind_group(
        &self,
        device: &wgpu::Device,
        i: usize,
        external_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frostify.composite bg (external)"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.textures[i].uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(external_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    // Linear: the external view is at decoder resolution and
                    // gets scaled by the composite quad — nearest aliases.
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        })
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Composite pipeline targeting the linear `BACKDROP_FORMAT` — P4
    /// composites below-glass content into `backdrop_tex` (the blur input).
    /// Wired in by the segment-walk increment.
    #[allow(dead_code)]
    pub fn backdrop_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.backdrop_pipeline
    }

    /// Total bytes the layer textures occupy (4 B/texel, surface format),
    /// summed across the per-slot sizes (scroll layers can be taller).
    pub fn memory_bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|t| t.size[0] as u64 * t.size[1] as u64 * 4)
            .sum()
    }
}

fn make_raster_bg(
    device: &wgpu::Device,
    shape_bgl: &wgpu::BindGroupLayout,
    frame: &wgpu::Buffer,
    instance_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frostify.layer raster bg"),
        layout: shape_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instance_buffer.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_uniform_is_96_bytes() {
        // WGSL `Composite` struct must equal the Rust size (M2-style
        // stride landmine). Six leading vec2 (0..48) + vec4 clip_rect
        // (48..64) + (opacity, corner_radius, edge_fade_falloff, pad)
        // (64..80) + vec4 edge_fade (80..96) = 96 B, 16-aligned.
        assert_eq!(std::mem::size_of::<CompositeUniform>(), 96);
    }

    #[test]
    fn layer_draw_default_is_identity() {
        let d = LayerDraw::default();
        assert_eq!(d.offset, [0.0, 0.0]);
        assert_eq!(d.scale, [1.0, 1.0]);
        assert_eq!(d.opacity, 1.0);
        assert_eq!(d.instances, 0..0);
        assert_eq!(d.window, None);
    }

    #[test]
    fn texture_size_picks_window_or_surface() {
        let surface = [800u32, 600];
        let identity = LayerDraw::default();
        assert_eq!(identity.texture_size(surface), surface);
        let scroll = LayerDraw {
            window: Some(ScrollWindow {
                dst_origin: [0.0, 0.0],
                dst_size: [300.0, 500.0],
                src_origin: [0.0, 120.0],
                tex_size: [300.0, 4000.0],
                clip_rect: NO_CLIP,
            }),
            ..Default::default()
        };
        assert_eq!(scroll.texture_size(surface), [300, 4000]);
    }
}

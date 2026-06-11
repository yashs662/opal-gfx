use std::sync::Arc;

use winit::window::Window;

use super::blur::BlurResources;
use super::glyph_atlas::GlyphAtlas;
use super::image_atlas::{ImageAtlas, ImageHandle};
use super::instance::{
    FrameUniform, SHAPE_KIND_GLASS, SHAPE_KIND_GLYPH, SHAPE_KIND_IMAGE, SHAPE_KIND_MASK,
    ShapeInstance,
};
use super::overdraw::OverdrawResources;
use super::pipeline::ShapePipeline;
use crate::node::{ImageRef, TextRef};
use crate::text::TextResources;

// `TextResources` is owned by `SceneCtx`, not the renderer — shape/measure
// passes need it too, and keeping it scene-side avoids a borrow split.
use super::timing::{
    FrameTiming, PASS_FINAL, PASS_OD_COMPOSE, PASS_OD_COUNT, PASS_OPAQUE, PassAlloc, Timing,
};

/// A node's resident Canvas frames: every frame of the loop uploaded once
/// into its own VRAM texture, replayed by re-binding one of `views`. The
/// `textures` are kept alongside their `views` to keep them alive.
#[derive(Default)]
struct ExternalFrameSet {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    /// Sum of frame sizes (bytes) for the memory report.
    bytes: u64,
}

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
    /// Offscreen layer textures + the composite pipeline (compositor
    /// P2). Each layer rasterizes its instance sub-range here, then the
    /// composite pass blits all layers to the surface in z-order.
    pub layers: super::layer::LayerResources,
    /// Per-frame layer draw list set by `set_layers` (from the CPU
    /// `LayerTree`). Single root layer today → one entry covering the
    /// whole instance stream at identity transform.
    layer_draws: Vec<super::layer::LayerDraw>,
    /// External-texture registry (P6): caller-owned textures (video /
    /// Canvas decoder output) keyed by the `.external()` node id. The
    /// composite pass samples these for external layers instead of the
    /// slot's own raster texture. We keep the view (composite sampling)
    /// alongside the texture (keeps it alive).
    external_textures: std::collections::HashMap<crate::node::NodeId, wgpu::TextureView>,
    /// Engine-owned frame textures backing [`Self::upload_external_frame`].
    /// Reused across frames (re-created only when the frame size changes)
    /// so a 30fps video doesn't re-allocate a texture every frame.
    external_owned: std::collections::HashMap<crate::node::NodeId, wgpu::Texture>,
    /// Resident per-node frame sets (P6, looping Canvas): the whole clip
    /// uploaded once into VRAM, replayed by re-binding `external_textures`
    /// to one of these views — no per-frame CPU→GPU transfer. Keyed by the
    /// `.external()` node id; migrated across rebuilds, dropped on clear.
    external_frame_sets: std::collections::HashMap<crate::node::NodeId, ExternalFrameSet>,
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
    /// Global backdrop content invalidation — a `blur_source` node's pixels
    /// changed (set by `set_instances` from `dirty::BACKDROP`, or explicitly).
    /// `blur_source` sits below every glass, so this rebuilds them all.
    /// Cleared after render.
    backdrop_dirty: bool,
    /// Per-layer (by draw index) "composite params changed since the last
    /// frame" — offset / scale / opacity / scroll-window. Recomputed each
    /// `set_layers`. The per-glass backdrop pass rebuilds a glass iff any
    /// layer *below* it is flagged here (or re-rastered this frame). This is
    /// the GENERAL dirty rule — no special-casing of an "ambient" glass; a
    /// glass simply re-frosts when its own backdrop (everything beneath it)
    /// changes, whatever that content is.
    layer_composite_changed: Vec<bool>,

    /// Timestamp query resources. `Some` when the adapter advertises
    /// `Features::TIMESTAMP_QUERY`, `None` otherwise. Reads happen on
    /// demand via `take_last_timing`.
    timing: Option<Timing>,
    /// Render-pass + compute drawcall counter for the most recent frame.
    last_drawcalls: u32,
    /// Compositor accounting for the most recent `encode_frame`: live
    /// layers, layers actually re-rastered (the rest reused their cached
    /// texture), and layers composited. Fed into `FrameStats`.
    last_layer_count: u32,
    last_raster_count: u32,
    last_composite_count: u32,
    /// Backdrop pyramid (re)builds this frame — one full-screen opaque pass
    /// + downsample each. The dominant glass cost; surfaced in `FrameStats`
    /// so scroll-time blur churn is measurable (ambient skipped, only the
    /// upper glass should rebuild on scroll).
    last_backdrop_builds: u32,
    /// Cached frame timing read at the end of the last render. `None`
    /// when timing isn't available or hasn't been read yet.
    last_timing: Option<FrameTiming>,
    /// Window-level corner radius in logical px applied by the final
    /// shader as a clip SDF. `0.0` = square corners (no clip). Set via
    /// [`Self::set_window_corner_radius`]; consumed by the next frame
    /// uniform upload.
    window_corner_radius: f32,
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

        log::info!("gpu init: format={format:?} alpha={alpha_mode:?} size={width}x{height}");

        let glyph_atlas = GlyphAtlas::new(&device, 1024);
        // Allocate the image atlas **large up front**. Growing re-uploads
        // every cached source into the new (bigger) texture synchronously
        // on the UI thread — a multi-100ms hitch for a large list whose
        // covers stream in. Starting at the adapter's max 2D dimension
        // (clamped to 8192² = 256 MiB) means growth — and the eviction
        // fallback that growth would otherwise reach at the ceiling —
        // effectively never fires: every uploaded cover stays resident, so
        // none "never load" from a dangling/evicted handle. VRAM is the
        // cheap axis here (the compositor doctrine); 256 MiB holds ~1700
        // 300 px covers, far past any real playlist's working set.
        let atlas_size = device.limits().max_texture_dimension_2d.min(8192);
        let image_atlas = ImageAtlas::new(&device, atlas_size);
        let shape = ShapePipeline::new(&device, format, glyph_atlas.layout(), image_atlas.layout());
        let blur = BlurResources::new(&device, width, height);
        let overdraw = OverdrawResources::new(&device, width, height, format, &shape.shape_bgl);
        let timing = if want_timing {
            Some(Timing::new(&device, &queue))
        } else {
            None
        };
        log::info!(
            "gpu timing: {}",
            if timing.is_some() { "on" } else { "off" }
        );

        // Allocate an initial instance buffer with room for one shape.
        let instance_capacity: u64 = 16;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.instance ssbo"),
            size: instance_capacity * std::mem::size_of::<ShapeInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Layer resources need the instance SSBO for the per-layer raster
        // bind groups (scroll-layer content-local path), so build after it.
        let layers = super::layer::LayerResources::new(
            &device,
            format,
            width,
            height,
            &shape.shape_bgl,
            &instance_buffer,
        );

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
                window_corner_radius: 0.0,
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
            layers,
            layer_draws: Vec::new(),
            external_textures: std::collections::HashMap::new(),
            external_owned: std::collections::HashMap::new(),
            external_frame_sets: std::collections::HashMap::new(),
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
            layer_composite_changed: Vec::new(),
            prev_instances: Vec::new(),
            backdrop_dirty: true,
            timing,
            last_drawcalls: 0,
            last_layer_count: 1,
            last_raster_count: 0,
            last_composite_count: 0,
            last_backdrop_builds: 0,
            last_timing: None,
            window_corner_radius: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.blur.resize(
            &self.device,
            self.surface_config.width,
            self.surface_config.height,
        );
        self.queue.write_buffer(
            &self.shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [
                    self.surface_config.width as f32,
                    self.surface_config.height as f32,
                ],
                max_backdrop_lod: self.blur.mip_count().saturating_sub(1) as f32,
                window_corner_radius: self.window_corner_radius,
            }),
        );
        // Blurred view changed — rebuild the glass bind group.
        self.glass_bg = make_glass_bg(&self.device, &self.shape, &self.blur);
        self.overdraw.resize(
            &self.device,
            self.surface_config.width,
            self.surface_config.height,
        );
        // Layer textures are physical-px sized — drop cached textures so
        // the next frame re-allocates them at the new size from the live
        // draw list (identity = surface, scroll = its content size).
        self.layers
            .resize(self.surface_config.width, self.surface_config.height);
        self.backdrop_dirty = true;
    }

    /// Set the per-frame layer draw list (from the CPU `LayerTree`).
    /// `encode_frame` rasters each layer's instance sub-range into its
    /// texture, then composites them to the surface in z-order. An empty
    /// list falls back to a single root layer spanning every instance.
    /// Register (or replace) the external texture for an `.external()`
    /// node (P6). The composite pass samples it for that node's layer
    /// instead of a rastered slot texture. Pass a `TextureView` over the
    /// caller's decoder output; call again each frame the video advances
    /// (and request a redraw to recomposite). Unregistered external nodes
    /// composite empty until set.
    pub fn set_external_texture(&mut self, node: crate::node::NodeId, view: wgpu::TextureView) {
        self.external_textures.insert(node, view);
    }

    /// Drop a node's external texture + any resident frame set (e.g. video
    /// stopped / node gone), freeing the VRAM.
    pub fn clear_external_texture(&mut self, node: crate::node::NodeId) {
        self.external_textures.remove(&node);
        self.external_owned.remove(&node);
        self.external_frame_sets.remove(&node);
    }

    /// Append one decoded frame to `node`'s resident frame set, uploading it
    /// to a new VRAM texture **once**, and bind it as the shown texture
    /// (first-pass live build). The set is created on the first push. Bytes
    /// are sRGB-encoded (`Rgba8UnormSrgb`), matching the surface.
    pub fn push_external_frame(
        &mut self,
        node: crate::node::NodeId,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        debug_assert_eq!(rgba.len(), (width * height * 4) as usize);
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas frame"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let set = self.external_frame_sets.entry(node).or_default();
        set.bytes += (width as u64) * (height as u64) * 4;
        set.views.push(view.clone());
        set.textures.push(tex);
        // Show the just-uploaded frame.
        self.external_textures.insert(node, view);
    }

    /// Bind `node`'s shown texture to frame `index` of its resident set.
    /// Cheap — re-binds a cached view, no pixel transfer. No-op if the node
    /// has no set or the index is out of range.
    pub fn select_external_frame(&mut self, node: crate::node::NodeId, index: usize) {
        if let Some(set) = self.external_frame_sets.get(&node)
            && let Some(view) = set.views.get(index)
        {
            self.external_textures.insert(node, view.clone());
        }
    }

    /// Move a resident frame set from `old` to `new` (a rebuild reassigned
    /// the `.external()` node id). No re-upload; rebinds the shown texture to
    /// the migrated set's last frame (the next `select` corrects the index).
    pub fn migrate_external_frames(&mut self, old: crate::node::NodeId, new: crate::node::NodeId) {
        if old == new {
            return;
        }
        if let Some(set) = self.external_frame_sets.remove(&old) {
            self.external_textures.remove(&old);
            if let Some(view) = set.views.last() {
                self.external_textures.insert(new, view.clone());
            }
            self.external_frame_sets.insert(new, set);
        }
    }

    /// Upload tightly-packed `width * height * 4` RGBA8 pixels (a decoder
    /// frame) as `node`'s external texture, then register its view. The
    /// backing texture is engine-owned and reused across calls — only
    /// re-created when `width`/`height` change — so a video that pushes a
    /// new frame each tick doesn't churn allocations. Bytes are treated as
    /// sRGB-encoded (the texture is `Rgba8UnormSrgb`), matching the surface
    /// so no channel swizzle or colour-space fixup is needed. The caller
    /// still requests a redraw to recomposite.
    pub fn upload_external_frame(
        &mut self,
        node: crate::node::NodeId,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        debug_assert_eq!(rgba.len(), (width * height * 4) as usize);
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // Reuse the existing texture unless the frame size changed.
        let tex = match self.external_owned.get(&node) {
            Some(t) if t.width() == width && t.height() == height => t,
            _ => {
                let t = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("external frame"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.external_owned.insert(node, t);
                self.external_owned.get(&node).unwrap()
            }
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.external_textures.insert(node, view);
    }

    pub fn set_layers(&mut self, draws: &[super::layer::LayerDraw]) {
        // Generic glass (P4) sources its backdrop from the composite of the
        // layers *below* it. Those layers can change composite-only (a scroll
        // window moving, a `set_layer_offset`, a crossfade's `layer_opacity`)
        // without touching any instance bytes — which wouldn't otherwise
        // re-run the backdrop pass. Record per layer whether its composite
        // params moved since last frame; the per-glass backdrop pass
        // (`encode_frame`) rebuilds a glass iff any layer below it is flagged
        // (or re-rastered). No glass is special — this is purely "did the
        // stuff beneath you move?", so glasses can be added/removed freely.
        self.layer_composite_changed.clear();
        self.layer_composite_changed.reserve(draws.len());
        for (i, d) in draws.iter().enumerate() {
            let changed = match self.layer_draws.get(i) {
                Some(old) => {
                    d.offset != old.offset
                        || d.scale != old.scale
                        || d.opacity != old.opacity
                        || d.window != old.window
                }
                // New layer (count grew / first frame) → treat as changed.
                None => true,
            };
            self.layer_composite_changed.push(changed);
        }
        self.layer_draws.clear();
        self.layer_draws.extend_from_slice(draws);
    }

    /// Content signature of a layer's instance sub-range + whether it
    /// contains glass. The signature (FNV-1a over the instance bytes in
    /// `prev_instances`, the CPU shadow of the uploaded buffer) is the
    /// raster-skip key: identical bytes ⇒ identical rasterized pixels ⇒
    /// reuse the cached texture. `has_glass` lets the caller force a
    /// re-raster when the backdrop pyramid changed (glass samples it).
    fn layer_signature(&self, range: &std::ops::Range<u32>) -> (u64, bool) {
        let start = (range.start as usize).min(self.prev_instances.len());
        let end = (range.end as usize).min(self.prev_instances.len());
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut has_glass = false;
        for inst in &self.prev_instances[start..end] {
            for &b in bytemuck::bytes_of(inst) {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
            if inst.shape_kind & SHAPE_KIND_MASK == SHAPE_KIND_GLASS {
                has_glass = true;
            }
        }
        (hash, has_glass)
    }

    pub fn overdraw_mode(&self) -> bool {
        self.overdraw_mode
    }

    pub fn set_overdraw(&mut self, on: bool) {
        self.overdraw_mode = on;
    }

    /// Set the window-level corner radius (logical px). `0.0` disables
    /// the clip. Re-uploads the frame uniform so the next render picks
    /// it up; callers should pair this with `request_redraw` if the
    /// loop is idle.
    pub fn set_window_corner_radius(&mut self, r: f32, scale: f32) {
        self.window_corner_radius = r.max(0.0) * scale;
        self.queue.write_buffer(
            &self.shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [
                    self.surface_config.width as f32,
                    self.surface_config.height as f32,
                ],
                max_backdrop_lod: self.blur.mip_count().saturating_sub(1) as f32,
                window_corner_radius: self.window_corner_radius,
            }),
        );
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
            // The per-layer raster bind groups (scroll-layer content-local
            // path) reference the instance SSBO too — repoint them at the
            // freshly-grown buffer or they'd sample the dropped one.
            self.layers
                .rebuild_raster_bgs(&self.device, &self.instance_buffer);
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
                if bytemuck::bytes_of(&instances[i]) == bytemuck::bytes_of(&self.prev_instances[i])
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
        // Per-glass split is computed per-layer in `encode_frame` (P4
        // generic glass) from `prev_instances`, so no global first-glass
        // index is cached here anymore.
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
                let Some(entry) = self
                    .glyph_atlas
                    .get_or_insert(&self.queue, text, g.cache_key)
                else {
                    continue;
                };
                if entry.width == 0 || entry.height == 0 {
                    continue;
                }
                // Snap each glyph to the physical pixel grid. The atlas
                // bitmap is rastered at integer size; if the destination is
                // sub-pixel, linear filtering smears the coverage across two
                // rows/cols (blurry text). Rounding keeps 1 texel ≈ 1 pixel.
                let px = (r.position[0] + g.x as f32 + entry.left as f32).round();
                let py = (r.position[1] + g.y as f32 - entry.top as f32).round();
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
                    clip_radius: 0.0,
                    _pad1: 0.0,
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
            let (mut u0, mut v0) = (entry.uv[0], entry.uv[1]);
            let (mut uv_w, mut uv_h) = (entry.uv[2] - entry.uv[0], entry.uv[3] - entry.uv[1]);
            // Cover fit: crop a centred sub-region of the source whose
            // aspect matches the node rect, so the image fills without
            // stretching (overflow is cropped, not squished). Images are
            // packed at native resolution, so the atlas region's `uv_w/uv_h`
            // ratio is the source pixel aspect.
            if r.cover && uv_w > 0.0 && uv_h > 0.0 && r.size[0] > 0.0 && r.size[1] > 0.0 {
                let img_aspect = uv_w / uv_h;
                let node_aspect = r.size[0] / r.size[1];
                if node_aspect > img_aspect {
                    // Node wider than image → keep full width, crop height.
                    let nh = uv_h * (img_aspect / node_aspect);
                    v0 += (uv_h - nh) * 0.5;
                    uv_h = nh;
                } else {
                    // Node taller/narrower → keep full height, crop width.
                    let nw = uv_w * (node_aspect / img_aspect);
                    u0 += (uv_w - nw) * 0.5;
                    uv_w = nw;
                }
            }
            out.push(ShapeInstance {
                color: r.color,
                border_color: [0.0; 4],
                shadow_color: [0.0; 4],
                border_radius: r.border_radius,
                backdrop_uv_rect: [u0, v0, uv_w, uv_h],
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
                clip_radius: r.clip_radius,
                _pad1: 0.0,
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
        self.glyph_atlas.reset(&self.queue);
    }

    /// Decode a PNG and upload it into the image atlas. Returns a
    /// handle the scene can pass to [`crate::node::Node::image`]. Fails
    /// (returns `None`) on malformed PNG, oversize image, or atlas
    /// exhaustion.
    pub fn upload_image_png(&mut self, bytes: &[u8]) -> Option<ImageHandle> {
        self.image_atlas.upload_png(&self.queue, bytes)
    }

    /// Rasterize one layer's instance sub-range into its offscreen texture
    /// (`fs_main`, cleared transparent). Honors raster-skip: returns
    /// `false` (no pass encoded) when the slot's content signature is
    /// unchanged and `force` is false. `ts_pair` is the optional
    /// timestamp write-index pair (used only by the root layer's
    /// PASS_FINAL slot); the query set is read from `self.timing` here so
    /// the caller doesn't hold a `&self.timing` borrow across this
    /// `&mut self` call.
    ///
    /// Split out of the old single raster loop so P4 can raster the
    /// below-glass layers **before** the backdrop build (which composites
    /// them) and the rest after.
    fn raster_layer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        draw: &super::layer::LayerDraw,
        force: bool,
        ts_pair: Option<(u32, u32)>,
    ) -> bool {
        let (sig, _) = self.layer_signature(&draw.instances);
        if draw.instances.is_empty() || !self.layers.needs_raster(i, sig, force) {
            return false;
        }
        {
            let ts = match (self.timing.as_ref(), ts_pair) {
                (Some(t), Some((b, e))) => Some(wgpu::RenderPassTimestampWrites {
                    query_set: &t.query_set,
                    beginning_of_pass_write_index: Some(b),
                    end_of_pass_write_index: Some(e),
                }),
                _ => None,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.layer raster"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.layers.raster_view(i),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.shape.final_pipeline);
            // Identity layers raster in absolute screen coords against the
            // global shape BG (frame = surface); scroll layers raster
            // content-local against their per-layer BG. Group 1 (glass
            // backdrop) is always bound; scroll layers never sample it.
            let group0 = match draw.window {
                Some(_) => self.layers.raster_bg(i),
                None => &self.shape_bg,
            };
            rpass.set_bind_group(0, group0, &[]);
            rpass.set_bind_group(1, &self.glass_bg, &[]);
            rpass.set_bind_group(2, self.glyph_atlas.bind_group(), &[]);
            rpass.set_bind_group(3, self.image_atlas.bind_group(), &[]);
            rpass.draw(0..6, draw.instances.clone());
        }
        self.layers.mark_rastered(i, sig);
        true
    }

    /// Encode the opaque pass, downsample dispatches (if needed), and
    /// final pass into `encoder`. `final_view` is the render target for
    /// the surface pass.
    fn encode_frame(&mut self, encoder: &mut wgpu::CommandEncoder, final_view: &wgpu::TextureView) {
        let mut drawcalls: u32 = 0;
        // Whether GPU timestamp timing is active. Used only for the dense
        // query-pair allocation below; each pass reads `self.timing`
        // locally when building its `RenderPassTimestampWrites` so no
        // long-lived `&self.timing` borrow spans the `&mut self`
        // `raster_layer` calls (P4 split the raster into two phases).
        let has_timing = self.timing.is_some();
        let mut alloc = PassAlloc::new();
        // The opaque pass exists only to populate `backdrop_tex` mip 0
        // (and the rest of the pyramid via downsample) for glass shapes
        // to sample in the final pass. If there's no glass, the backdrop
        // is never sampled; if backdrop content hasn't changed since the
        // last submit the existing pyramid is still valid.
        let has_glass = self.glass_count() > 0;
        // Whether any backdrop build will *likely* run this frame (a
        // blur_source changed, or some layer's composite moved). Per-glass
        // `need` below decides which glasses actually rebuild; this only
        // gates the shared timing-query allocation. (A build triggered purely
        // by a below-layer re-raster may go untimed — timing is debug-only.)
        let run_backdrop =
            has_glass && (self.backdrop_dirty || self.layer_composite_changed.iter().any(|&c| c));
        // Pre-allocate query pairs for every pass that will run this
        // frame. The pair indices are dense so `resolve_query_set` can
        // cover a contiguous prefix.
        let (opaque_begin, opaque_end) = if has_timing && run_backdrop {
            let (b, e) = alloc.alloc(PASS_OPAQUE);
            (Some(b), Some(e))
        } else {
            (None, None)
        };
        let (final_begin, final_end) = if has_timing {
            let (b, e) = alloc.alloc(PASS_FINAL);
            (Some(b), Some(e))
        } else {
            (None, None)
        };
        let (od_count_begin, od_count_end) = if has_timing && self.overdraw_mode {
            let (b, e) = alloc.alloc(PASS_OD_COUNT);
            (Some(b), Some(e))
        } else {
            (None, None)
        };
        let (od_compose_begin, od_compose_end) = if has_timing && self.overdraw_mode {
            let (b, e) = alloc.alloc(PASS_OD_COMPOSE);
            (Some(b), Some(e))
        } else {
            (None, None)
        };

        // Effective draw list + per-layer texture sizing/uniforms, hoisted
        // **above** the backdrop pass (P4): generic glass sources its
        // backdrop from the composite of layers below it, so the layer
        // textures + composite uniforms must be ready before the backdrop
        // step runs. With a single root layer (no layers below a mid-stream
        // glass) the backdrop step still falls back to the raw pre-glass
        // draw below → byte-identical to the pre-P4 path.
        let screen = [
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        ];
        let draws: Vec<super::layer::LayerDraw> = if self.layer_draws.is_empty() {
            vec![super::layer::LayerDraw {
                instances: 0..self.instance_count,
                ..Default::default()
            }]
        } else {
            self.layer_draws.clone()
        };
        let surface_px = [self.surface_config.width, self.surface_config.height];
        let sizes: Vec<[u32; 2]> = draws.iter().map(|d| d.texture_size(surface_px)).collect();
        self.layers
            .ensure(&self.device, &sizes, &self.instance_buffer);
        let max_lod = self.blur.mip_count().saturating_sub(1) as f32;
        for (i, d) in draws.iter().enumerate() {
            self.layers.write_uniform(&self.queue, i, d, screen);
            if let Some(w) = d.window.as_ref() {
                self.layers.write_frame(&self.queue, i, w.tex_size, max_lod);
            }
        }

        // ---- Backdrop build (P4 generic glass) → backdrop_tex -----------
        // The backdrop a glass samples = composite of every layer painted
        // *below* it + that layer's own instances *before* the glass
        // (`fs_opaque` discards any glass in that pre-range). Skipped when
        // no glass exists or the prior submit's backdrop is still valid.
        // Index of the first glass-bearing layer in paint order, and the
        // within-layer instance offset of its first glass shape. The
        // backdrop a glass samples = composite of every layer painted
        // *below* it + that layer's own instances *before* the glass.
        let order_paint: Vec<usize> = {
            let mut o: Vec<usize> = (0..draws.len()).collect();
            o.sort_by_key(|&i| draws[i].z);
            o
        };

        // ---- Per-glass-layer backdrops (P4, generalized to N glass depths)
        // Each glass-bearing layer samples a backdrop = the composite of
        // every layer painted *below* it (+ its own pre-glass instances).
        // Walk glass layers in z-order; for each: raster the not-yet-
        // rastered layers below it (its backdrop inputs), rebuild the single
        // backdrop pyramid from them, then raster the glass layer — it reads
        // the pyramid immediately and writes its own texture, so the next
        // glass can overwrite the pyramid (sequential reuse, one pyramid).
        // One glass layer ⇒ one build ⇒ byte-identical to the old single-
        // backdrop path. This is what lets e.g. a sticky header glass frost
        // the list scrolling beneath it, not just the ambient backdrop.
        let mut raster_count = 0u32;
        let mut rastered = vec![false; draws.len()];
        let glass_pps: Vec<usize> = order_paint
            .iter()
            .enumerate()
            .filter_map(|(pp, &li)| self.layer_signature(&draws[li].instances).1.then_some(pp))
            .collect();
        // Raster a single layer (once) with the shared timing on layer 0.
        // Evaluates to `true` iff it actually (re)rastered this call.
        macro_rules! raster_once {
            ($li:expr, $force:expr) => {{
                let li = $li;
                if rastered[li] {
                    false
                } else {
                    let ts = if li == 0 {
                        final_begin.zip(final_end)
                    } else {
                        None
                    };
                    let d = draws[li].clone();
                    let did = self.raster_layer(encoder, li, &d, $force, ts);
                    if did {
                        drawcalls += 1;
                        raster_count += 1;
                    }
                    rastered[li] = true;
                    did
                }
            }};
        }
        let mut built_backdrop = false;
        let mut backdrop_builds = 0u32;
        // Did any layer processed so far (in ascending z) change since last
        // frame — re-rastered (content) or moved (composite)? Accumulated as
        // we walk up, so each glass sees "did anything beneath me change?".
        // This is the whole dirty rule — general, glass-agnostic.
        let mut any_below_changed = false;
        for &gp in &glass_pps {
            let glass_layer = order_paint[gp];
            // Raster this glass layer's backdrop inputs (everything below it),
            // and fold their content/composite changes into `any_below_changed`.
            for &li in &order_paint[..gp] {
                if raster_once!(li, false) {
                    any_below_changed = true;
                }
                if self
                    .layer_composite_changed
                    .get(li)
                    .copied()
                    .unwrap_or(true)
                {
                    any_below_changed = true;
                }
            }
            // Rebuild this glass's backdrop iff a `blur_source` changed
            // (global) OR anything below it moved/re-rastered. No glass is
            // privileged: the ambient album glass re-frosts when its art
            // crossfades; the sticky header re-frosts when the list scrolls;
            // remove either and the rule still holds.
            //
            // ALSO rebuild when this glass layer's own content changed: it
            // will re-raster below regardless of `need`, and the shared
            // pyramid may still hold a *different* glass's backdrop from an
            // earlier frame (sequential reuse). Sampling that is a feedback
            // loop — e.g. a hover-state change in a glass-bearing layer made
            // the glass re-blur a pyramid containing the layer's own prior
            // output, darkening the backdrop a little more every toggle.
            let glass_content_dirty = {
                let (sig, _) = self.layer_signature(&draws[glass_layer].instances);
                !draws[glass_layer].instances.is_empty()
                    && self.layers.needs_raster(glass_layer, sig, false)
            };
            let need = self.backdrop_dirty || any_below_changed || glass_content_dirty;
            if need {
                // Build the backdrop into `backdrop_tex` mip0 (linear):
                // composite layers `order_paint[0..gp]` (premultiplied-over)
                // then raw-draw the glass layer's pre-glass instances on top
                // (`fs_opaque` discards glass). Single glass at gp=0 → the
                // composite loop is empty and only the raw draw runs → the
                // exact pre-generalization opaque pass.
                let gstart = draws[glass_layer].instances.start as usize;
                let gend = draws[glass_layer].instances.end as usize;
                let pre_glass_end = {
                    let s = gstart.min(self.prev_instances.len());
                    let e = gend.min(self.prev_instances.len());
                    let rel = crate::gpu::instance::first_glass_index(&self.prev_instances[s..e]);
                    gstart as u32 + rel
                };
                // GPU timing attaches to the first build of the frame only.
                let backdrop_ts = match (self.timing.as_ref(), opaque_begin.zip(opaque_end)) {
                    (Some(t), Some((b, e))) if !built_backdrop => {
                        Some(wgpu::RenderPassTimestampWrites {
                            query_set: &t.query_set,
                            beginning_of_pass_write_index: Some(b),
                            end_of_pass_write_index: Some(e),
                        })
                    }
                    _ => None,
                };
                {
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
                        timestamp_writes: backdrop_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if gp > 0 {
                        rpass.set_pipeline(self.layers.backdrop_pipeline());
                        for &li in &order_paint[..gp] {
                            if draws[li].instances.is_empty() {
                                continue;
                            }
                            rpass.set_bind_group(0, self.layers.bind_group(li), &[]);
                            rpass.draw(0..6, 0..1);
                            drawcalls += 1;
                        }
                    }
                    if pre_glass_end > gstart as u32 {
                        rpass.set_pipeline(&self.shape.opaque_pipeline);
                        rpass.set_bind_group(0, &self.shape_bg, &[]);
                        rpass.set_bind_group(2, self.glyph_atlas.bind_group(), &[]);
                        rpass.set_bind_group(3, self.image_atlas.bind_group(), &[]);
                        rpass.draw(0..6, (gstart as u32)..pre_glass_end);
                        drawcalls += 1;
                    }
                }
                // Downsample mip0 → pyramid for this glass layer to sample.
                self.blur.run_downsample(encoder);
                built_backdrop = true;
                backdrop_builds += 1;
            }
            // Raster the glass layer (samples the freshly-built pyramid;
            // forced when its backdrop rebuilt). It's a backdrop input for any
            // higher glass, so fold its (re)raster + composite move upward.
            if raster_once!(glass_layer, need) {
                any_below_changed = true;
            }
            if self
                .layer_composite_changed
                .get(glass_layer)
                .copied()
                .unwrap_or(true)
            {
                any_below_changed = true;
            }
        }
        // Clear the global blur_source flag once its builds ran — but only
        // when glass exists to consume it (with no glass the backdrop is
        // never built, so it must persist until a glass appears). Per-layer
        // composite flags are rebuilt wholesale each `set_layers`.
        if has_glass {
            self.backdrop_dirty = false;
        }
        self.last_backdrop_builds = backdrop_builds;
        // Trailing sweep: layers above the last glass (or ALL layers when
        // there's no glass at all) that weren't rastered as backdrop inputs.
        for &li in &order_paint {
            raster_once!(li, false);
        }

        // Composite: blit each layer to the surface back-to-front, then
        // draw the debug overlay (HUD) on top.
        let mut order: Vec<usize> = (0..draws.len()).collect();
        order.sort_by_key(|&i| draws[i].z);
        // Per-layer composite bind groups. External-texture layers (P6)
        // get an ad-hoc bg pairing the slot uniform with the caller's view;
        // every other layer uses its stored slot bg. Built before the pass
        // so the owned externals outlive `rpass`.
        let composite_bgs: Vec<CompositeBg> = draws
            .iter()
            .enumerate()
            .map(|(i, d)| match d.external {
                Some(node) => match self.external_textures.get(&node) {
                    Some(view) => {
                        CompositeBg::Owned(self.layers.external_bind_group(&self.device, i, view))
                    }
                    // No texture registered yet → composite the (empty)
                    // slot texture so the layer reads transparent.
                    None => CompositeBg::Slot(i),
                },
                None => CompositeBg::Slot(i),
            })
            .collect();
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.composite pass"),
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(self.layers.pipeline());
            for &i in &order {
                let bg = match &composite_bgs[i] {
                    CompositeBg::Owned(b) => b,
                    CompositeBg::Slot(s) => self.layers.bind_group(*s),
                };
                rpass.set_bind_group(0, bg, &[]);
                rpass.draw(0..6, 0..1);
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
        self.last_layer_count = draws.len() as u32;
        self.last_raster_count = raster_count;
        self.last_composite_count = order.len() as u32;

        // ---- Pass D (optional): overdraw count + compose --------------
        // When toggled on, count shape coverage into an Rgba16Float
        // accumulator, then re-render the swapchain with a heatmap of the
        // count. The final pass already cleared and drew the scene; the
        // compose pass overwrites it with the heatmap (LoadOp::Clear).
        if self.overdraw_mode {
            {
                let od_count_ts = match (self.timing.as_ref(), od_count_begin.zip(od_count_end)) {
                    (Some(t), Some((b, e))) => Some(wgpu::RenderPassTimestampWrites {
                        query_set: &t.query_set,
                        beginning_of_pass_write_index: Some(b),
                        end_of_pass_write_index: Some(e),
                    }),
                    _ => None,
                };
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
                    timestamp_writes: od_count_ts,
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
                let od_compose_ts =
                    match (self.timing.as_ref(), od_compose_begin.zip(od_compose_end)) {
                        (Some(t), Some((b, e))) => Some(wgpu::RenderPassTimestampWrites {
                            query_set: &t.query_set,
                            beginning_of_pass_write_index: Some(b),
                            end_of_pass_write_index: Some(e),
                        }),
                        _ => None,
                    };
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
                    timestamp_writes: od_compose_ts,
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
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
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
            // Compositor metrics — the GPU is authoritative (it decides
            // what to actually raster/composite and what it allocated).
            layer_count: self.last_layer_count,
            raster_count: self.last_raster_count,
            composite_count: self.last_composite_count,
            backdrop_builds: self.last_backdrop_builds,
            layer_vram: self.layers.memory_bytes(),
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
            layer_textures: self.layers.memory_bytes(),
            timing,
            params_buffers,
            glyph_atlas: self.glyph_atlas.memory_bytes(),
            image_atlas: self.image_atlas.memory_bytes(),
            image_sources_cpu: self.image_atlas.source_bytes(),
            external_frames: self.external_frame_sets.values().map(|s| s.bytes).sum(),
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
        rx.recv().expect("map channel closed").expect("map failed");

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
    /// Offscreen compositor layer textures (4 B/texel × surface size ×
    /// layer count). One full-surface root layer in P2.
    pub layer_textures: u64,
    pub timing: u64,
    pub params_buffers: u64,
    pub glyph_atlas: u64,
    pub image_atlas: u64,
    /// CPU-side cache of source bytes for every uploaded image
    /// (`ImageAtlas::source_bytes()`). Required so the eviction path
    /// can re-pack survivors when the atlas fills.
    pub image_sources_cpu: u64,
    /// Resident Canvas frame sets in VRAM (whole loops uploaded once,
    /// replayed by view re-bind). 4 B/px × every cached frame.
    pub external_frames: u64,
}

impl MemoryReport {
    pub fn total(&self) -> u64 {
        self.instance_buffer
            + self.overlay_buffer
            + self.prev_instances_cpu
            + self.blur_textures
            + self.overdraw_textures
            + self.layer_textures
            + self.timing
            + self.params_buffers
            + self.glyph_atlas
            + self.image_atlas
            + self.image_sources_cpu
            + self.external_frames
    }
}

/// Per-layer composite bind-group selection for one frame (P6). Most
/// layers reuse their stored slot bind group (`Slot`); external-texture
/// layers get a freshly-built one over the caller's view (`Owned`), held
/// here so it outlives the composite render pass.
enum CompositeBg {
    Slot(usize),
    Owned(wgpu::BindGroup),
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

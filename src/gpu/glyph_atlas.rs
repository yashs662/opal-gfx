//! R8Unorm glyph atlas.
//!
//! Backed by a single square `R8Unorm` texture and an `etagere`
//! shelf-pack allocator. Keyed on cosmic-text `CacheKey`. When the
//! atlas fills up we drop everything and start over — [`generation`]
//! bumps so stale UVs cached in instance buffers are invalidated.
//!
//! Stage-1 is R8 only (no color emoji); color glyphs short-circuit
//! in [`crate::text::TextResources::rasterize`] and never reach here.

use std::collections::HashMap;

use cosmic_text::CacheKey;
use etagere::{size2, AtlasAllocator};

use crate::text::TextResources;

/// UV + metrics for a single glyph laid into the atlas.
#[derive(Copy, Clone, Debug)]
pub struct AtlasEntry {
    /// UV rect in `[0, 1]^2` — `[x0, y0, x1, y1]`.
    pub uv: [f32; 4],
    pub width: u32,
    pub height: u32,
    /// Horizontal bearing (pen → left edge of bitmap).
    pub left: i32,
    /// Vertical bearing (pen → top edge of bitmap, positive = above baseline).
    pub top: i32,
}


pub struct GlyphAtlas {
    size: u32,
    texture: wgpu::Texture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    allocator: AtlasAllocator,
    occupants: HashMap<CacheKey, AtlasEntry>,
    /// Bumped on every reset. Consumers (scene flush) use it to know
    /// when to re-resolve cached UVs.
    generation: u32,
    /// Transparent border around each glyph to keep linear filtering
    /// from bleeding across neighbours.
    padding: i32,
}

impl GlyphAtlas {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify-gfx glyph atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify-gfx glyph sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frostify-gfx glyph atlas bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frostify-gfx glyph atlas bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let allocator = AtlasAllocator::new(size2(size as i32, size as i32));
        Self {
            size,
            texture,
            layout,
            bind_group,
            allocator,
            occupants: HashMap::new(),
            generation: 0,
            padding: 1,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn get(&self, key: CacheKey) -> Option<AtlasEntry> {
        self.occupants.get(&key).copied()
    }

    /// Fetch `key` from the atlas, rasterizing + uploading on miss.
    /// Returns `None` for color-emoji glyphs or missing glyphs — caller
    /// should skip them. On allocator-full we reset the atlas once and
    /// retry; if the glyph still doesn't fit (larger than the atlas)
    /// we give up and return `None`.
    pub fn get_or_insert(
        &mut self,
        queue: &wgpu::Queue,
        text: &mut TextResources,
        key: CacheKey,
    ) -> Option<AtlasEntry> {
        if let Some(e) = self.occupants.get(&key) {
            return Some(*e);
        }
        let raster = text.rasterize(key)?;
        if raster.width == 0 || raster.height == 0 {
            // Whitespace glyphs can rasterize to an empty bitmap — still
            // a valid cache entry so we don't retry every frame.
            let entry = AtlasEntry {
                uv: [0.0; 4],
                width: 0,
                height: 0,
                left: raster.left,
                top: raster.top,
            };
            self.occupants.insert(key, entry);
            return Some(entry);
        }
        let w = raster.width as i32 + 2 * self.padding;
        let h = raster.height as i32 + 2 * self.padding;
        if w > self.size as i32 || h > self.size as i32 {
            return None;
        }
        let alloc = match self.allocator.allocate(size2(w, h)) {
            Some(a) => a,
            None => {
                self.reset(queue);
                self.allocator.allocate(size2(w, h))?
            }
        };
        let rect = alloc.rectangle;
        let gx = rect.min.x + self.padding;
        let gy = rect.min.y + self.padding;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: gx as u32,
                    y: gy as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &raster.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(raster.width),
                rows_per_image: Some(raster.height),
            },
            wgpu::Extent3d {
                width: raster.width,
                height: raster.height,
                depth_or_array_layers: 1,
            },
        );
        let inv = 1.0 / self.size as f32;
        let entry = AtlasEntry {
            uv: [
                gx as f32 * inv,
                gy as f32 * inv,
                (gx + raster.width as i32) as f32 * inv,
                (gy + raster.height as i32) as f32 * inv,
            ],
            width: raster.width,
            height: raster.height,
            left: raster.left,
            top: raster.top,
        };
        let _ = alloc.id;
        self.occupants.insert(key, entry);
        Some(entry)
    }

    /// Drop every cached glyph and the shelf-pack state. Bumps
    /// [`Self::generation`] so any consumer that cached UVs from a
    /// prior frame can detect the invalidation. Triggered when the
    /// physical glyph size changes (DPI scale flip) or the atlas
    /// fills up.
    ///
    /// Also **zeroes the texture**: cells get reused by the new
    /// generation, and glyph writes only cover the bitmap — not the 1px
    /// padding border — so without clearing, stale coverage from the
    /// previous generation lingers in those borders and bleeds into a
    /// neighbour under linear filtering (a faint mark beside a glyph).
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.allocator.clear();
        self.occupants.clear();
        self.generation = self.generation.wrapping_add(1);
        let zeros = vec![0u8; (self.size as usize) * (self.size as usize)];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &zeros,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size),
                rows_per_image: Some(self.size),
            },
            wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Reported GPU bytes used by the atlas texture.
    pub fn memory_bytes(&self) -> u64 {
        self.size as u64 * self.size as u64
    }
}

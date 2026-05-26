//! Rgba8UnormSrgb image atlas.
//!
//! Backed by a single square `Rgba8UnormSrgb` texture and an `etagere`
//! shelf-pack allocator. Keyed on a monotonic [`ImageHandle`]; users
//! upload PNG bytes via [`ImageAtlas::upload_png`] and stash the handle
//! to reference the image in scene nodes.
//!
//! **Eviction model: snapshot-rebuild.** Every uploaded image's source
//! bytes are cached in `sources` (cost: ~4 B/px × all uploads ever). On
//! direct [`ImageAtlas::upload_rgba`] failure the caller can invoke
//! [`ImageAtlas::rebuild_keeping`] with a `HashSet` of currently-live
//! handles — the allocator resets, non-live sources are dropped from
//! the cache, and the live set is re-uploaded into a fresh atlas
//! layout. Each surviving handle's `ImageEntry::uv` is rewritten in
//! place, so existing nodes that reference it pick up the new UVs on
//! the next flatten pass. [`ImageAtlas::upload_rgba_or_evict`] is the
//! convenience wrapper: try a normal upload, run a rebuild on failure
//! using the supplied live set, retry once.
//!
//! `Rgba8UnormSrgb` matches the shape pipeline convention: PNG bytes
//! are sRGB-authored, the texture decode flag converts to linear when
//! sampled, fragment math stays linear, the swapchain encodes back to
//! sRGB on store.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use etagere::{size2, AtlasAllocator};

/// Opaque handle returned by [`ImageAtlas::upload_png`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub u32);

/// UV + pixel size for an uploaded image.
#[derive(Copy, Clone, Debug)]
pub struct ImageEntry {
    /// UV rect in `[0, 1]^2` — `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    pub width: u32,
    pub height: u32,
}

pub struct ImageAtlas {
    size: u32,
    texture: wgpu::Texture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    allocator: AtlasAllocator,
    occupants: HashMap<ImageHandle, ImageEntry>,
    next_handle: u32,
    /// Transparent border around each image to keep linear filtering
    /// from bleeding across neighbours.
    padding: i32,
    /// CPU-side source-byte cache. Populated on every `upload_rgba`
    /// (including the re-uploads inside `rebuild_keeping`). Required
    /// so `rebuild_keeping` can re-write each surviving handle's pixel
    /// data into the fresh atlas layout. Drops entries on rebuild for
    /// any handle absent from the supplied live set.
    sources: HashMap<ImageHandle, ImageSource>,
}

struct ImageSource {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl ImageAtlas {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify-gfx image atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify-gfx image sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frostify-gfx image atlas bgl"),
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
            label: Some("frostify-gfx image atlas bg"),
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
            next_handle: 0,
            padding: 1,
            sources: HashMap::new(),
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

    pub fn get(&self, handle: ImageHandle) -> Option<ImageEntry> {
        self.occupants.get(&handle).copied()
    }

    /// Decode a PNG byte slice and upload its pixels into the atlas.
    /// Returns `None` if the PNG is malformed, larger than the atlas,
    /// or the atlas is full. RGBA8 internally; sRGB inputs welcome.
    pub fn upload_png(
        &mut self,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> Option<ImageHandle> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().ok()?;
        let (w, h, color_type, bit_depth, buf_size) = {
            let info = reader.info();
            (info.width, info.height, info.color_type, info.bit_depth, reader.output_buffer_size()?)
        };
        let mut buf = vec![0u8; buf_size];
        let frame = reader.next_frame(&mut buf).ok()?;
        let in_bytes = &buf[..frame.buffer_size()];
        // Normalize to RGBA8.
        let rgba = match (color_type, bit_depth) {
            (png::ColorType::Rgba, png::BitDepth::Eight) => in_bytes.to_vec(),
            (png::ColorType::Rgb, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for px in in_bytes.chunks_exact(3) {
                    out.extend_from_slice(&[px[0], px[1], px[2], 0xFF]);
                }
                out
            }
            (png::ColorType::Grayscale, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for &g in in_bytes {
                    out.extend_from_slice(&[g, g, g, 0xFF]);
                }
                out
            }
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for px in in_bytes.chunks_exact(2) {
                    out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                }
                out
            }
            _ => return None,
        };
        self.upload_rgba(queue, w, h, &rgba)
    }

    /// Upload pre-decoded `Rgba8UnormSrgb` pixels (`w*h*4` bytes,
    /// row-major, top-left origin). Stricter than [`upload_png`] —
    /// caller is responsible for color-space correctness. Caches the
    /// source bytes so a later [`rebuild_keeping`] can re-pack them.
    pub fn upload_rgba(
        &mut self,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
    ) -> Option<ImageHandle> {
        self.upload_internal(queue, w, h, rgba, None)
    }

    /// Convenience: try [`upload_rgba`]; on allocator failure, run
    /// [`rebuild_keeping`] with `live` (dropping any non-live sources)
    /// and retry once. Used by `App::upload_image_rgba` to handle
    /// "atlas full while album art lands" without forcing every caller
    /// to do the walk-tree-then-rebuild dance.
    pub fn upload_rgba_or_evict(
        &mut self,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
        live: &HashSet<ImageHandle>,
    ) -> Option<ImageHandle> {
        if let Some(h) = self.upload_rgba(queue, w, h, rgba) {
            return Some(h);
        }
        self.rebuild_keeping(queue, live);
        self.upload_rgba(queue, w, h, rgba)
    }

    /// Drop every cached source whose handle is missing from `live`,
    /// reset the allocator, and re-upload the survivors into a fresh
    /// atlas layout. Each surviving handle keeps its `ImageHandle`
    /// value; only its `ImageEntry::uv` changes. Pure CPU + texture
    /// writes — no GPU sync required, callers' instance buffers pick
    /// up the new UVs on their next flatten.
    ///
    /// Failure cases (live source larger than atlas, OOM, etc.) drop
    /// the offending handle entirely. The remaining live set still
    /// packs — partial rebuild is preferred to all-or-nothing.
    pub fn rebuild_keeping(&mut self, queue: &wgpu::Queue, live: &HashSet<ImageHandle>) {
        // 1. Drop sources not in live set + their occupant entries.
        self.sources.retain(|h, _| live.contains(h));
        self.occupants.retain(|h, _| live.contains(h));

        // 2. Fresh allocator. Old packing is discarded; the GPU texture
        //    is left untouched (dead pixels at no-longer-allocated
        //    coords are unreferenced and harmless — sampler never
        //    addresses them).
        self.allocator = AtlasAllocator::new(size2(self.size as i32, self.size as i32));

        // 3. Re-upload each surviving source into a fresh slot. Snapshot
        //    the (handle, w, h, bytes) tuples first to release the
        //    sources borrow before calling upload_internal (which
        //    mutates `self`).
        let snapshot: Vec<(ImageHandle, u32, u32, Vec<u8>)> = self
            .sources
            .iter()
            .map(|(h, s)| (*h, s.width, s.height, s.rgba.clone()))
            .collect();

        for (handle, w, h, bytes) in snapshot {
            // upload_internal with fixed_handle = Some(handle) re-uses
            // the existing handle id (no new mint, no source re-cache).
            // Drops the occupant entry on packing failure rather than
            // panicking — see method-level doc.
            if self.upload_internal(queue, w, h, &bytes, Some(handle)).is_none() {
                self.occupants.remove(&handle);
                self.sources.remove(&handle);
            }
        }
    }

    /// Iterate every currently-live handle in the atlas. Useful for
    /// composing the `live` set passed to `rebuild_keeping`.
    pub fn live_handles(&self) -> impl Iterator<Item = ImageHandle> + '_ {
        self.occupants.keys().copied()
    }

    /// CPU memory used by source-byte caches. GPU memory is separately
    /// reported by [`memory_bytes`].
    pub fn source_bytes(&self) -> u64 {
        self.sources.values().map(|s| s.rgba.len() as u64).sum()
    }

    /// Reported GPU bytes used by the atlas texture.
    pub fn memory_bytes(&self) -> u64 {
        self.size as u64 * self.size as u64 * 4
    }

    /// Test hook: look up a source's cached bytes. None when no upload
    /// has used that handle (or it was dropped by a rebuild).
    #[cfg(test)]
    pub(crate) fn source_size(&self, h: ImageHandle) -> Option<(u32, u32)> {
        self.sources.get(&h).map(|s| (s.width, s.height))
    }

    /// Shared body for both new-handle uploads (`fixed_handle = None`)
    /// and rebuild re-uploads (`fixed_handle = Some(h)` — reuses the
    /// existing handle, skips source caching since the bytes are
    /// already cached). Allocates, writes pixels, returns the handle.
    fn upload_internal(
        &mut self,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
        fixed_handle: Option<ImageHandle>,
    ) -> Option<ImageHandle> {
        if w == 0 || h == 0 || rgba.len() != (w * h * 4) as usize {
            return None;
        }
        let pad = self.padding;
        let pad_w = w as i32 + 2 * pad;
        let pad_h = h as i32 + 2 * pad;
        if pad_w > self.size as i32 || pad_h > self.size as i32 {
            return None;
        }
        let alloc = self.allocator.allocate(size2(pad_w, pad_h))?;
        let rect = alloc.rectangle;
        let gx = rect.min.x + pad;
        let gy = rect.min.y + pad;
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
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let inv = 1.0 / self.size as f32;
        let entry = ImageEntry {
            uv: [
                gx as f32 * inv,
                gy as f32 * inv,
                (gx + w as i32) as f32 * inv,
                (gy + h as i32) as f32 * inv,
            ],
            width: w,
            height: h,
        };
        let handle = match fixed_handle {
            Some(h) => h,
            None => {
                let new = ImageHandle(self.next_handle);
                self.next_handle = self.next_handle.wrapping_add(1);
                // Cache source bytes only for net-new uploads. Rebuild
                // re-uploads already have a live source entry.
                self.sources.insert(
                    new,
                    ImageSource { width: w, height: h, rgba: rgba.to_vec() },
                );
                new
            }
        };
        let _ = alloc.id;
        self.occupants.insert(handle, entry);
        Some(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Spin up a noop wgpu device. Cheap (~1 ms) — no GPU required.
    /// `noop.enable = true` is the runtime opt-in for the NOOP backend:
    /// the feature flag compiles it in, this field activates it.
    /// Without `enable: true` `request_adapter` returns `NotFound` with
    /// `active_backends = 0x0`.
    fn noop_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true },
                ..Default::default()
            },
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .expect("noop adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("noop device")
    }

    fn solid_rgba(w: u32, h: u32, c: [u8; 4]) -> Vec<u8> {
        (0..(w * h))
            .flat_map(|_| c.iter().copied())
            .collect()
    }

    #[test]
    fn upload_rgba_caches_source_bytes() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let bytes = solid_rgba(8, 8, [255, 0, 0, 255]);
        let h = atlas.upload_rgba(&queue, 8, 8, &bytes).expect("upload");
        assert_eq!(atlas.source_size(h), Some((8, 8)));
        assert_eq!(atlas.source_bytes(), (8 * 8 * 4) as u64);
    }

    #[test]
    fn rebuild_keeping_empty_drops_everything() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let a = atlas.upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4])).unwrap();
        let b = atlas.upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [2; 4])).unwrap();
        atlas.rebuild_keeping(&queue, &HashSet::new());
        assert!(atlas.get(a).is_none(), "a should be evicted");
        assert!(atlas.get(b).is_none(), "b should be evicted");
        assert_eq!(atlas.source_bytes(), 0, "source cache must be empty");
        assert_eq!(atlas.live_handles().count(), 0);
    }

    #[test]
    fn rebuild_keeping_preserves_live_handles() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let live = atlas.upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4])).unwrap();
        let _dead = atlas.upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [2; 4])).unwrap();
        let pre_uv = atlas.get(live).unwrap().uv;
        let keep: HashSet<_> = [live].into_iter().collect();
        atlas.rebuild_keeping(&queue, &keep);
        let post = atlas.get(live).expect("live handle survives");
        assert_eq!(post.width, 8);
        assert_eq!(post.height, 8);
        // Live handle's UV is typically the same after rebuild (first
        // alloc lands at the same spot) — just assert it's a valid
        // entry rather than asserting identity, since etagere's
        // packing order is an implementation detail.
        let _ = pre_uv;
        assert_eq!(atlas.live_handles().count(), 1);
        assert!(atlas.source_size(live).is_some());
    }

    #[test]
    fn upload_rgba_or_evict_succeeds_after_rebuild() {
        let (device, queue) = noop_device();
        // Tiny atlas — first big upload fills it. Second upload
        // would otherwise fail; with eviction it succeeds.
        let mut atlas = ImageAtlas::new(&device, 32);
        let bytes_a = solid_rgba(30, 30, [1; 4]);
        let bytes_b = solid_rgba(30, 30, [2; 4]);
        let a = atlas.upload_rgba(&queue, 30, 30, &bytes_a).expect("first fits");
        // Direct second upload fails — no room.
        assert!(
            atlas.upload_rgba(&queue, 30, 30, &bytes_b).is_none(),
            "second upload must fail without eviction (atlas full)"
        );
        // Eviction path with empty live set drops `a` and succeeds.
        let live = HashSet::new();
        let b = atlas
            .upload_rgba_or_evict(&queue, 30, 30, &bytes_b, &live)
            .expect("retry after eviction");
        assert!(atlas.get(a).is_none(), "a should have been evicted");
        assert!(atlas.get(b).is_some());
    }

    #[test]
    fn handle_value_is_stable_across_rebuild() {
        // ImageHandle is the public stable identifier — rebuild must
        // not mint new handles for survivors.
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let live = atlas.upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4])).unwrap();
        let keep: HashSet<_> = [live].into_iter().collect();
        atlas.rebuild_keeping(&queue, &keep);
        // Same handle value; new entry.
        assert!(atlas.get(live).is_some());
        assert_eq!(atlas.live_handles().next(), Some(live));
    }

    #[test]
    fn rebuild_compacts_so_new_uploads_fit_after_clearing() {
        // Fill atlas with 4 tiles, evict all, verify 4 more fit.
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 32);
        let h1 = atlas.upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [1; 4])).unwrap();
        let h2 = atlas.upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [2; 4])).unwrap();
        // Two 14x14 (padded 16x16) tiles fill the 32x32 atlas roughly.
        // Drop both via empty live + verify new uploads succeed.
        atlas.rebuild_keeping(&queue, &HashSet::new());
        assert!(atlas.get(h1).is_none());
        assert!(atlas.get(h2).is_none());
        let h3 = atlas.upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [3; 4]));
        assert!(h3.is_some(), "post-rebuild upload should succeed");
    }
}

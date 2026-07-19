//! Rgba8UnormSrgb image atlas.
//!
//! Backed by a single square `Rgba8UnormSrgb` texture and an `etagere`
//! shelf-pack allocator. Keyed on a monotonic [`ImageHandle`]; users
//! upload PNG bytes via [`ImageAtlas::upload_png`] and stash the handle
//! to reference the image in scene nodes.
//!
//! **Eviction model: GPU repack.** Pixels live *only* in the atlas
//! texture — no CPU-side source copies (an earlier snapshot-rebuild
//! design cached every upload's bytes forever: ~4 B/px × all uploads,
//! hundreds of MB of covers in a real session). On grow or evict a
//! fresh texture is created, each survivor's rect is
//! `copy_texture_to_texture`-blitted across into a fresh allocator
//! layout, and the texture/bind group swap in place. Each surviving
//! handle's `ImageEntry::uv` is rewritten, so existing nodes that
//! reference it pick up the new UVs on the next flatten pass.
//! [`ImageAtlas::rebuild_keeping`] drops any handle absent from the
//! caller's live set before repacking.
//!
//! `Rgba8UnormSrgb` matches the shape pipeline convention: PNG bytes
//! are sRGB-authored, the texture decode flag converts to linear when
//! sampled, fragment math stays linear, the swapchain encodes back to
//! sRGB on store.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use etagere::{AtlasAllocator, size2};

/// Opaque handle returned by [`ImageAtlas::upload_png`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub u32);

/// Result of [`ImageAtlas::upload_rgba_growing`]: the new handle plus
/// whether the upload **repacked** the atlas (grow / evict), which moves
/// every existing handle's UV. When `false`, only the node(s) bound to
/// `handle` need repainting; when `true`, the caller must re-flatten so
/// every image instance picks up its shifted UV.
#[derive(Copy, Clone, Debug)]
pub struct UploadOutcome {
    pub handle: ImageHandle,
    pub layout_changed: bool,
}

/// UV + pixel size for an uploaded image.
#[derive(Copy, Clone, Debug)]
pub struct ImageEntry {
    /// UV rect in `[0, 1]^2` — `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    pub width: u32,
    pub height: u32,
    /// Top-left texel of the image in the atlas (post-padding). The
    /// repack path blits `origin_px + width×height` between textures —
    /// kept in px so it never round-trips through the f32 UVs.
    pub origin_px: [u32; 2],
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
    sampler: wgpu::Sampler,
}

/// Texture usage shared by the initial atlas and every repack target.
/// `COPY_SRC` is what lets grow/evict blit pixels across on the GPU
/// instead of retaining CPU source copies.
const ATLAS_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::COPY_SRC);

impl ImageAtlas {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("opal-gfx image atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: ATLAS_USAGE,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("opal-gfx image sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal-gfx image atlas bgl"),
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
            label: Some("opal-gfx image atlas bg"),
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
            sampler,
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
    pub fn upload_png(&mut self, queue: &wgpu::Queue, bytes: &[u8]) -> Option<ImageHandle> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().ok()?;
        let (w, h, color_type, bit_depth, buf_size) = {
            let info = reader.info();
            (
                info.width,
                info.height,
                info.color_type,
                info.bit_depth,
                reader.output_buffer_size()?,
            )
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
    /// caller is responsible for color-space correctness.
    pub fn upload_rgba(
        &mut self,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
    ) -> Option<ImageHandle> {
        self.upload_internal(queue, w, h, rgba, None)
    }

    /// Auto-managed upload — the recommended path for consumer apps. Try
    /// [`upload_rgba`]; on failure **grow** the atlas (doubling up to
    /// `max_size`), retrying after each step; evict (`rebuild_keeping`)
    /// only as a last resort once the atlas is already at `max_size`.
    /// Returns `None` only when even a max-sized atlas (post-evict)
    /// can't fit the bytes, or when `w`/`h` exceeds `max_size` itself.
    ///
    /// **Why grow before evict?** Consumer-side bursts (e.g. a Home
    /// screen's worth of album art arriving back-to-back in one frame)
    /// upload many images before any are rendered. The caller-supplied
    /// `live` set comes from a tree walk, which only sees handles that
    /// have been bound to a node + flushed — i.e. handles from the
    /// previous frame's render. Just-uploaded handles from the same
    /// drain pass are invisible to `live`. Evicting at that point
    /// silently drops those handles, leaving consumers with stale
    /// handles that point at recycled atlas slots. Growing keeps every
    /// cached source intact, so a burst always lands.
    pub fn upload_rgba_growing(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
        // Lazy: computing the live set walks the whole node tree, and the
        // overwhelmingly common path (atlas has room) never needs it. Only
        // the last-resort evict calls this — so a burst of uploads pays the
        // tree walk at most once, not per upload.
        live: impl FnOnce() -> HashSet<ImageHandle>,
        max_size: u32,
    ) -> Option<UploadOutcome> {
        // Fast path: the atlas has room. The new handle gets a fresh slot
        // and **no existing UV moves**, so the caller only needs to repaint
        // the node(s) bound to this one handle — not re-flatten the scene.
        if let Some(handle) = self.upload_rgba(queue, w, h, rgba) {
            return Some(UploadOutcome {
                handle,
                layout_changed: false,
            });
        }
        // Try growing first — preserves every cached source, so an in-
        // flight burst of uploads stays intact regardless of how stale
        // the caller's `live` snapshot is. Grow/evict both **repack** the
        // atlas → every surviving handle's UV moves → `layout_changed`.
        while self.size < max_size {
            let next = (self.size * 2).min(max_size);
            if !self.grow(device, queue, next) {
                break;
            }
            if let Some(handle) = self.upload_rgba(queue, w, h, rgba) {
                return Some(UploadOutcome {
                    handle,
                    layout_changed: true,
                });
            }
        }
        // Last resort at max texture dimension: evict non-live + retry.
        // At this point we know `live` is the best signal we have, and
        // dropping anything not in it is the only way to make room. This
        // is the only path that needs the live set, so compute it now.
        self.rebuild_keeping(device, queue, &live());
        self.upload_rgba(queue, w, h, rgba)
            .map(|handle| UploadOutcome {
                handle,
                layout_changed: true,
            })
    }

    /// Replace the underlying GPU texture with one of `new_size`,
    /// GPU-blitting every occupant across into a fresh allocator
    /// layout. Bind group is rebuilt; render code reads
    /// [`Self::bind_group`] per pass, so callers don't need to rebind.
    /// No-op (returns `false`) when `new_size <= self.size`.
    pub fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, new_size: u32) -> bool {
        if new_size <= self.size {
            return false;
        }
        self.repack(device, queue, new_size, None);
        true
    }

    /// Drop every occupant whose handle is missing from `live`, then
    /// repack the survivors into a fresh same-size texture (GPU blits —
    /// no CPU pixel copies exist). Each surviving handle keeps its
    /// `ImageHandle` value; only its `ImageEntry::uv` changes, and
    /// callers' instance buffers pick up the new UVs on their next
    /// flatten.
    ///
    /// Failure cases (survivor no longer packs, OOM, etc.) drop the
    /// offending handle entirely — partial rebuild is preferred to
    /// all-or-nothing.
    pub fn rebuild_keeping(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        live: &HashSet<ImageHandle>,
    ) {
        self.repack(device, queue, self.size, Some(live));
    }

    /// Shared grow/evict body: create a `new_size` texture, allocate
    /// each survivor a slot in a fresh allocator, blit its pixels
    /// old-texture → new-texture on the GPU, swap texture + bind group
    /// in. `keep = None` keeps every occupant (grow); `Some(live)`
    /// drops the rest first (evict). A survivor that fails to allocate
    /// in the new layout is dropped rather than aborting the repack.
    fn repack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_size: u32,
        keep: Option<&HashSet<ImageHandle>>,
    ) {
        if let Some(live) = keep {
            self.occupants.retain(|h, _| live.contains(h));
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("opal-gfx image atlas"),
            size: wgpu::Extent3d {
                width: new_size,
                height: new_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: ATLAS_USAGE,
            view_formats: &[],
        });
        let mut allocator = AtlasAllocator::new(size2(new_size as i32, new_size as i32));
        let pad = self.padding;
        let inv = 1.0 / new_size as f32;
        // Deterministic repack order (handle id) — packing outcome must
        // not depend on HashMap iteration order.
        let mut survivors: Vec<(ImageHandle, ImageEntry)> =
            self.occupants.iter().map(|(h, e)| (*h, *e)).collect();
        survivors.sort_by_key(|(h, _)| h.0);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("opal-gfx atlas repack"),
        });
        for (handle, entry) in survivors {
            let (w, h) = (entry.width, entry.height);
            let Some(alloc) = allocator.allocate(size2(w as i32 + 2 * pad, h as i32 + 2 * pad))
            else {
                self.occupants.remove(&handle);
                continue;
            };
            let gx = (alloc.rectangle.min.x + pad) as u32;
            let gy = (alloc.rectangle.min.y + pad) as u32;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: entry.origin_px[0],
                        y: entry.origin_px[1],
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: gx, y: gy, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            self.occupants.insert(
                handle,
                ImageEntry {
                    uv: [
                        gx as f32 * inv,
                        gy as f32 * inv,
                        (gx + w) as f32 * inv,
                        (gy + h) as f32 * inv,
                    ],
                    width: w,
                    height: h,
                    origin_px: [gx, gy],
                },
            );
        }
        queue.submit(std::iter::once(encoder.finish()));
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opal-gfx image atlas bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.size = new_size;
        self.texture = texture;
        self.allocator = allocator;
    }

    /// Iterate every currently-live handle in the atlas. Useful for
    /// composing the `live` set passed to `rebuild_keeping`.
    pub fn live_handles(&self) -> impl Iterator<Item = ImageHandle> + '_ {
        self.occupants.keys().copied()
    }

    /// Reported GPU bytes used by the atlas texture.
    pub fn memory_bytes(&self) -> u64 {
        self.size as u64 * self.size as u64 * 4
    }

    /// Shared upload body. `fixed_handle = Some(h)` reuses an existing
    /// handle id (test-only affordance today — repack blits on the GPU
    /// instead of re-uploading). Allocates, writes pixels, returns the
    /// handle.
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
            origin_px: [gx as u32, gy as u32],
        };
        let handle = match fixed_handle {
            Some(h) => h,
            None => {
                let new = ImageHandle(self.next_handle);
                self.next_handle = self.next_handle.wrapping_add(1);
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
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("noop adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("noop device")
    }

    fn solid_rgba(w: u32, h: u32, c: [u8; 4]) -> Vec<u8> {
        (0..(w * h)).flat_map(|_| c.iter().copied()).collect()
    }

    #[test]
    fn rebuild_keeping_empty_drops_everything() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let a = atlas
            .upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4]))
            .unwrap();
        let b = atlas
            .upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [2; 4]))
            .unwrap();
        atlas.rebuild_keeping(&device, &queue, &HashSet::new());
        assert!(atlas.get(a).is_none(), "a should be evicted");
        assert!(atlas.get(b).is_none(), "b should be evicted");
        assert_eq!(atlas.live_handles().count(), 0);
    }

    #[test]
    fn rebuild_keeping_preserves_live_handles() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let live = atlas
            .upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4]))
            .unwrap();
        let _dead = atlas
            .upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [2; 4]))
            .unwrap();
        let keep: HashSet<_> = [live].into_iter().collect();
        atlas.rebuild_keeping(&device, &queue, &keep);
        let post = atlas.get(live).expect("live handle survives");
        assert_eq!(post.width, 8);
        assert_eq!(post.height, 8);
        assert_eq!(atlas.live_handles().count(), 1);
    }

    #[test]
    fn handle_value_is_stable_across_rebuild() {
        // ImageHandle is the public stable identifier — rebuild must
        // not mint new handles for survivors.
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 64);
        let live = atlas
            .upload_rgba(&queue, 8, 8, &solid_rgba(8, 8, [1; 4]))
            .unwrap();
        let keep: HashSet<_> = [live].into_iter().collect();
        atlas.rebuild_keeping(&device, &queue, &keep);
        // Same handle value; new entry.
        assert!(atlas.get(live).is_some());
        assert_eq!(atlas.live_handles().next(), Some(live));
    }

    #[test]
    fn rebuild_compacts_so_new_uploads_fit_after_clearing() {
        // Fill atlas with 4 tiles, evict all, verify 4 more fit.
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 32);
        let h1 = atlas
            .upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [1; 4]))
            .unwrap();
        let h2 = atlas
            .upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [2; 4]))
            .unwrap();
        // Two 14x14 (padded 16x16) tiles fill the 32x32 atlas roughly.
        // Drop both via empty live + verify new uploads succeed.
        atlas.rebuild_keeping(&device, &queue, &HashSet::new());
        assert!(atlas.get(h1).is_none());
        assert!(atlas.get(h2).is_none());
        let h3 = atlas.upload_rgba(&queue, 14, 14, &solid_rgba(14, 14, [3; 4]));
        assert!(h3.is_some(), "post-rebuild upload should succeed");
    }

    #[test]
    fn grow_preserves_handles_and_makes_room() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 32);
        // 28² padded → 30×30; one fills the 32² atlas, a 2nd can't fit
        // anywhere in the residual 2-px slivers.
        let h1 = atlas
            .upload_rgba(&queue, 28, 28, &solid_rgba(28, 28, [1; 4]))
            .unwrap();
        assert!(
            atlas
                .upload_rgba(&queue, 28, 28, &solid_rgba(28, 28, [2; 4]))
                .is_none()
        );
        assert!(atlas.grow(&device, &queue, 64), "grow 32→64 should succeed");
        assert_eq!(atlas.size(), 64);
        // Original handle survives verbatim, re-uploaded into the bigger
        // texture from the cached source bytes.
        assert!(atlas.get(h1).is_some());
        // Now there's room for the second.
        assert!(
            atlas
                .upload_rgba(&queue, 28, 28, &solid_rgba(28, 28, [2; 4]))
                .is_some()
        );
        // Growing to ≤ current size is a no-op.
        assert!(!atlas.grow(&device, &queue, 64));
        assert!(!atlas.grow(&device, &queue, 32));
    }

    #[test]
    fn burst_uploads_with_stale_live_keep_every_handle() {
        // Consumer scenario: many image uploads land in the same drain
        // pass, before any of them have been bound + flushed into the
        // tree. The caller's `live` set therefore omits them all. The
        // growing path must keep every burst-uploaded handle alive
        // (grow-before-evict) rather than silently dropping them.
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 32);
        let empty_live = HashSet::new();
        let mut handles = Vec::new();
        for i in 0..8 {
            let h = atlas
                .upload_rgba_growing(
                    &device,
                    &queue,
                    28,
                    28,
                    &solid_rgba(28, 28, [i as u8 + 1; 4]),
                    || empty_live.clone(),
                    1024,
                )
                .expect("burst upload must succeed via grow")
                .handle;
            handles.push(h);
        }
        // Every handle from the burst remains live in the atlas.
        for h in &handles {
            assert!(atlas.get(*h).is_some(), "burst handle {h:?} was evicted");
        }
        // Atlas grew at least once to fit the burst.
        assert!(atlas.size() > 32, "atlas should have grown");
    }

    #[test]
    fn upload_rgba_growing_doubles_until_it_fits() {
        let (device, queue) = noop_device();
        let mut atlas = ImageAtlas::new(&device, 32);
        // Pin a 28² live handle so eviction can't reclaim its space —
        // the grow path is the only way out for a second upload.
        let live_h = atlas
            .upload_rgba(&queue, 28, 28, &solid_rgba(28, 28, [1; 4]))
            .unwrap();
        let live: HashSet<_> = [live_h].into_iter().collect();
        // Second 28² won't fit at 32² and eviction can't help (live_h
        // pinned). Growing doubles 32→64 where both fit.
        let h2 = atlas
            .upload_rgba_growing(
                &device,
                &queue,
                28,
                28,
                &solid_rgba(28, 28, [2; 4]),
                || live.clone(),
                256,
            )
            .expect("growing path should succeed");
        assert!(h2.layout_changed, "grow path must report a repack");
        let h2 = h2.handle;
        assert!(atlas.size() > 32, "atlas must have grown");
        assert!(atlas.get(h2).is_some());
        assert!(
            atlas.get(live_h).is_some(),
            "live handle preserved across grow"
        );
    }
}

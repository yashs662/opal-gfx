use bytemuck::{Pod, Zeroable};

/// GPU shape kinds. Mirror the constants in `shape.wgsl`.
///
/// `ShapeInstance.shape_kind` is a packed u32:
/// - bits 0-7: shape kind (Rect/Glass/Glyph/Image) — masked via [`SHAPE_KIND_MASK`].
/// - bits 8-11: border-side mask (TRBL); 0b1111 = all sides (the default,
///   and what the WGSL falls back to when sides == 0 — see
///   [`crate::node::BorderSides::ALL`]).
/// Other bits reserved.
pub const SHAPE_KIND_MASK: u32 = 0xFF;
pub const SHAPE_KIND_RECT: u32 = 0;
pub const SHAPE_KIND_GLASS: u32 = 1;
/// Glyph blit: samples the `R8Unorm` glyph atlas at
/// `backdrop_uv_rect` (repurposed as atlas `(u0, v0, w, h)`),
/// multiplied by `color`. Shadows/borders disabled; `position`+`size`
/// bound the bitmap quad exactly.
pub const SHAPE_KIND_GLYPH: u32 = 2;
/// Image blit: samples the `Rgba8UnormSrgb` image atlas at
/// `backdrop_uv_rect` (repurposed as atlas `(u0, v0, w, h)`),
/// multiplied by `color` (used as tint; `[1,1,1,1]` for unmodified).
pub const SHAPE_KIND_IMAGE: u32 = 3;

/// Sentinel clip rect = no clipping. Min way below screen, max way
/// above. The shader compares `px` against these bounds and discards
/// outside; with the sentinel, nothing is ever discarded.
pub const NO_CLIP: [f32; 4] = [-1.0e30, -1.0e30, 1.0e30, 1.0e30];

/// Painter-order index of the first glass shape, or `instances.len()`
/// when there is none. The backdrop pass draws only `0..first_glass` so
/// shapes painted *in front of* the glass (drawn after it) can't bleed
/// into the blur the glass samples — they're simply not in the backdrop.
pub(crate) fn first_glass_index(instances: &[ShapeInstance]) -> u32 {
    instances
        .iter()
        .position(|s| s.shape_kind & SHAPE_KIND_MASK == SHAPE_KIND_GLASS)
        .map(|i| i as u32)
        .unwrap_or(instances.len() as u32)
}

/// GPU shape instance. Layout must match `ShapeInstance` in `shape.wgsl`.
/// std430-compatible. **160 bytes, 16-aligned.** WGSL
/// `array<ShapeInstance>` stride = roundUp(16, last_offset + last_size).
/// With `scale: vec2<f32>` at offset 144 (size 8), stride rounds to 160.
/// The Rust struct must equal that stride — the trailing `_pad: [f32; 2]`
/// closes the gap. Without it the stride-vs-size mismatch would corrupt
/// per-instance reads (M2 landmine).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShapeInstance {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub shadow_color: [f32; 4],
    pub border_radius: [f32; 4], // tl, tr, bl, br
    /// Multi-purpose by `shape_kind`:
    ///   - Glass: `(blur_px, refraction_px, _, _)`
    ///   - Glyph: atlas `(u0, v0, w, h)` into the R8 glyph atlas
    ///   - Image: atlas `(u0, v0, w, h)` into the Rgba8 image atlas
    ///   - Rect: ignored
    pub backdrop_uv_rect: [f32; 4],
    /// Scissor rect in physical px: `(min_x, min_y, max_x, max_y)`. The
    /// fragment shader discards any fragment outside these bounds.
    /// Default `NO_CLIP` = no clipping. Set by the flatten pass when the
    /// instance lives under a Scroll/Hidden overflow container.
    pub clip_rect: [f32; 4],
    pub position: [f32; 2], // top-left, pixels
    pub size: [f32; 2],     // pixels
    pub shadow_offset: [f32; 2],
    pub shape_kind: u32,
    /// Glass-only. Per-fragment LOD jitter for frosted-texture variation:
    /// sample UV is scattered by `hash(frag_coord) * roughness` pixels at
    /// the chosen mip. 0 = mirror-smooth glass; ~1 = noticeable
    /// frosted-pane texture; ~3 = pebbled. Authored in physical px;
    /// scales with display factor like `border_width`.
    pub roughness: f32,
    pub border_width: f32,
    pub shadow_blur: f32,
    pub shadow_opacity: f32,
    pub opacity: f32,
    /// Per-instance visual scale around the rect centre. `[1.0, 1.0]`
    /// = identity. Vertex shader expands the quad bounds; fragment
    /// shader rescales SDF coords so border + radius scale together.
    /// Layout + hit-test are unaffected (style-level transform only).
    pub scale: [f32; 2],
    /// Corner radius (physical px) of the [`Self::clip_rect`] — when an
    /// overflow container with a corner radius clips this instance, the
    /// fragment shader rounds the scissor by this radius (rounded overflow
    /// clipping). 0 = square clip. Set by the flatten pass.
    pub clip_radius: f32,
    /// Padding to align Rust `size_of` with WGSL array stride (160).
    pub _pad1: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(kind: u32) -> ShapeInstance {
        let mut s = ShapeInstance::zeroed();
        s.shape_kind = kind;
        s
    }

    #[test]
    fn first_glass_picks_earliest_glass_in_painter_order() {
        // [rect, image, glass, rect] → first glass at index 2, so only
        // indices 0..2 (the behind-glass backdrop) feed the blur.
        let list = [
            inst(SHAPE_KIND_RECT),
            inst(SHAPE_KIND_IMAGE),
            inst(SHAPE_KIND_GLASS),
            inst(SHAPE_KIND_RECT),
        ];
        assert_eq!(first_glass_index(&list), 2);
    }

    #[test]
    fn first_glass_honours_kind_mask() {
        // Glass with upper bits set (e.g. border-side mask) still counts.
        let glassy = inst(SHAPE_KIND_GLASS | (0b1111 << 8));
        let list = [inst(SHAPE_KIND_RECT), glassy];
        assert_eq!(first_glass_index(&list), 1);
    }

    #[test]
    fn no_glass_yields_full_length() {
        let list = [inst(SHAPE_KIND_RECT), inst(SHAPE_KIND_IMAGE)];
        assert_eq!(first_glass_index(&list), 2);
        assert_eq!(first_glass_index(&[]), 0);
    }
}

const _: () = assert!(std::mem::size_of::<ShapeInstance>() == 160);

impl Default for ShapeInstance {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            border_radius: [0.0; 4],
            backdrop_uv_rect: [0.0; 4],
            clip_rect: NO_CLIP,
            position: [0.0; 2],
            size: [0.0; 2],
            shadow_offset: [0.0; 2],
            shape_kind: SHAPE_KIND_RECT,
            roughness: 0.0,
            border_width: 0.0,
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
            scale: [1.0, 1.0],
            clip_radius: 0.0,
            _pad1: 0.0,
        }
    }
}

impl ShapeInstance {
    pub fn rounded_rect(
        position: [f32; 2],
        size: [f32; 2],
        radius: f32,
        color: [f32; 4],
    ) -> Self {
        Self {
            color,
            position,
            size,
            border_radius: [radius; 4],
            ..Default::default()
        }
    }
}

/// Per-frame uniform: viewport size in physical pixels + max usable
/// LOD for the backdrop mip pyramid (so the glass shader can clamp
/// `log2(blur_px)` without overflowing into a 1×1 mip).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub screen_size: [f32; 2],
    pub max_backdrop_lod: f32,
    /// Logical-to-physical scale applied for the window-corner clip
    /// SDF in the shader. `0.0` disables the corner clip entirely
    /// (square window, fullscreen, etc.). Otherwise the fragment shader
    /// in the final pass discards pixels outside a rounded rect with
    /// this corner radius — gives the whole window rounded corners
    /// without an extra render pass.
    pub window_corner_radius: f32,
}

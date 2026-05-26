//! Text shaping + glyph rasterization.
//!
//! Thin wrapper around cosmic-text's `FontSystem` + `SwashCache`.
//! Produces [`ShapedGlyph`]s (position + `CacheKey`) and on-demand
//! R8 masks via [`TextResources::rasterize`]. The atlas that holds
//! those masks and the draw-instance emission both live outside this
//! module — see [`crate::gpu`] for the atlas and
//! [`crate::node::ShapeKind::Text`] for the scene-graph hookup.
//!
//! Stage 1 uses system fonts (SansSerif family, sans bundled assets)
//! and rejects color-emoji glyphs so the atlas can stay R8Unorm.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use cosmic_text::{
    Attrs, Buffer, CacheKey, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent,
};

/// Owns the shaping engine + rasterized-glyph cache. One per `App`.
pub struct TextResources {
    font_system: FontSystem,
    swash_cache: SwashCache,
    /// Memoized shape outputs keyed by `(content, size, line_height)`
    /// hash. cosmic-text's `Buffer::new` + `set_text` + `shape_until_scroll`
    /// allocates + does real work on every call; for static UI labels
    /// (HUD stats, list rows, button captions) the same triple repeats
    /// every flatten — caching saves 50–100µs per flatten on a Spotify-
    /// scale scene. Hash collisions are theoretically possible (one in
    /// 2^64) but practically irrelevant for UI text volumes.
    shape_cache: HashMap<u64, Vec<ShapedGlyph>>,
    /// Same shape, but for [`Self::measure`] (intrinsic-size queries
    /// during the layout pass). The layout pass calls `measure_text`
    /// multiple times per Auto-sized text node during flex resolution.
    measure_cache: HashMap<u64, TextMetrics>,
}

/// One shaped glyph: an atlas key and its physical-pixel pen position.
/// The final screen position is `pen + (placement.left, -placement.top)`
/// where `placement` comes from [`TextResources::rasterize`].
#[derive(Copy, Clone, Debug)]
pub struct ShapedGlyph {
    pub cache_key: CacheKey,
    pub x: i32,
    pub y: i32,
}

/// Result of rasterizing one glyph to an R8 coverage mask.
#[derive(Debug)]
pub struct RasterizedGlyph {
    pub width: u32,
    pub height: u32,
    /// Horizontal bearing from pen origin to left of bitmap.
    pub left: i32,
    /// Vertical bearing from pen origin to top of bitmap (positive = above baseline).
    pub top: i32,
    /// Row-major R8 coverage, `width * height` bytes.
    pub data: Vec<u8>,
}

/// Bounding box of a shaped run, in physical pixels relative to the
/// pen origin at `(0, 0)` on the first baseline.
#[derive(Copy, Clone, Debug, Default)]
pub struct TextMetrics {
    pub width: f32,
    pub height: f32,
}

impl TextResources {
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            shape_cache: HashMap::new(),
            measure_cache: HashMap::new(),
        }
    }

    /// Drop the shape + measure caches. Call after font system mutation
    /// (font swap, axis change) or whenever cached shapes might no
    /// longer match what the current `FontSystem` would produce. The
    /// glyph atlas reset path doesn't need to call this — atlas cache
    /// keys are physical-px-tied and shape cache keys are too, so a
    /// DPI change naturally bypasses old shape entries via key mismatch.
    pub fn clear_shape_cache(&mut self) {
        self.shape_cache.clear();
        self.measure_cache.clear();
    }

    /// Shape `text` at `size_px` line-height `line_height_px`. Returns
    /// physical glyph positions and cache keys (for atlas lookup).
    /// Memoized — repeat calls with the same triple return a cloned
    /// cached `Vec` instead of re-allocating a `Buffer` and re-shaping.
    pub fn shape(&mut self, text: &str, size_px: f32, line_height_px: f32) -> Vec<ShapedGlyph> {
        let key = shape_key(text, size_px, line_height_px);
        if let Some(cached) = self.shape_cache.get(&key) {
            return cached.clone();
        }
        let out = shape_uncached(&mut self.font_system, text, size_px, line_height_px);
        self.shape_cache.insert(key, out.clone());
        out
    }

    /// Shape + measure bounding box in one call. Memoized like
    /// [`Self::shape`].
    pub fn measure(&mut self, text: &str, size_px: f32, line_height_px: f32) -> TextMetrics {
        let key = shape_key(text, size_px, line_height_px);
        if let Some(&cached) = self.measure_cache.get(&key) {
            return cached;
        }
        let m = measure_uncached(&mut self.font_system, text, size_px, line_height_px);
        self.measure_cache.insert(key, m);
        m
    }

    /// Pick the longest character prefix of `text` whose `prefix + "…"`
    /// renders at ≤ `max_width_px`. Used by both `measure_constrained`
    /// and `shape_constrained` so the truncation point is consistent
    /// across measure + render. When the full string already fits,
    /// returns the original `text` unchanged (no ellipsis).
    pub fn truncated_text(
        &mut self,
        text: &str,
        size_px: f32,
        line_height_px: f32,
        max_width_px: f32,
    ) -> String {
        let full = self.measure(text, size_px, line_height_px);
        if full.width <= max_width_px {
            return text.to_string();
        }
        // Binary-search the longest *character* prefix. We binary search
        // on the character index, not byte index, so we never split a
        // multi-byte codepoint. Cosmic-text's shaping respects
        // grapheme-cluster boundaries internally; this binary search
        // can occasionally land on a non-grapheme break (combining
        // marks), but the visual artifact is a missing accent at the
        // truncation point — acceptable for v1.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut lo = 0usize;
        let mut hi = chars.len();
        let mut best = 0usize;
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let byte_end = chars.get(mid).map(|(b, _)| *b).unwrap_or(text.len());
            let candidate = format!("{}\u{2026}", &text[..byte_end]);
            let m = self.measure(&candidate, size_px, line_height_px);
            if m.width <= max_width_px {
                best = mid;
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let byte_end = chars.get(best).map(|(b, _)| *b).unwrap_or(text.len());
        format!("{}\u{2026}", &text[..byte_end])
    }

    /// Measure `text` constrained to `max_width_px`. If the unconstrained
    /// measurement fits, returns it; otherwise returns the measurement
    /// of the truncated `prefix + "…"` form.
    pub fn measure_constrained(
        &mut self,
        text: &str,
        size_px: f32,
        line_height_px: f32,
        max_width_px: f32,
    ) -> TextMetrics {
        let s = self.truncated_text(text, size_px, line_height_px, max_width_px);
        self.measure(&s, size_px, line_height_px)
    }

    /// Shape `text` constrained to `max_width_px`. Identical to
    /// [`Self::shape`] when the original fits; otherwise shapes the
    /// `prefix + "…"` truncation.
    pub fn shape_constrained(
        &mut self,
        text: &str,
        size_px: f32,
        line_height_px: f32,
        max_width_px: f32,
    ) -> Vec<ShapedGlyph> {
        let s = self.truncated_text(text, size_px, line_height_px, max_width_px);
        self.shape(&s, size_px, line_height_px)
    }

    /// Rasterize one glyph. Returns `None` for color-emoji glyphs
    /// (stage-1 atlas is R8 only) or missing glyphs.
    pub fn rasterize(&mut self, key: CacheKey) -> Option<RasterizedGlyph> {
        let img = self
            .swash_cache
            .get_image(&mut self.font_system, key)
            .as_ref()?;
        if !matches!(img.content, SwashContent::Mask) {
            return None;
        }
        Some(RasterizedGlyph {
            width: img.placement.width,
            height: img.placement.height,
            left: img.placement.left,
            top: img.placement.top,
            data: img.data.clone(),
        })
    }
}

impl Default for TextResources {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::layout::Measurer for TextResources {
    fn measure_text(&mut self, content: &str, font_size: f32, line_height: f32) -> [f32; 2] {
        let m = self.measure(content, font_size, line_height);
        [m.width, m.height]
    }
    fn measure_text_constrained(
        &mut self,
        content: &str,
        font_size: f32,
        line_height: f32,
        max_width: f32,
    ) -> [f32; 2] {
        let m = self.measure_constrained(content, font_size, line_height, max_width);
        [m.width, m.height]
    }
}

fn shape_key(content: &str, size_px: f32, line_height_px: f32) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut h);
    h.write_u32(size_px.to_bits());
    h.write_u32(line_height_px.to_bits());
    h.finish()
}

fn shape_uncached(
    fs: &mut FontSystem,
    text: &str,
    size_px: f32,
    line_height_px: f32,
) -> Vec<ShapedGlyph> {
    let mut buf = Buffer::new(fs, Metrics::new(size_px, line_height_px));
    let attrs = Attrs::new().family(Family::SansSerif);
    buf.set_text(text, &attrs, Shaping::Advanced, None);
    buf.shape_until_scroll(fs, false);
    let mut out = Vec::new();
    for run in buf.layout_runs() {
        for g in run.glyphs.iter() {
            let phys = g.physical((0.0, run.line_y), 1.0);
            out.push(ShapedGlyph {
                cache_key: phys.cache_key,
                x: phys.x,
                y: phys.y,
            });
        }
    }
    out
}

fn measure_uncached(
    fs: &mut FontSystem,
    text: &str,
    size_px: f32,
    line_height_px: f32,
) -> TextMetrics {
    let mut buf = Buffer::new(fs, Metrics::new(size_px, line_height_px));
    let attrs = Attrs::new().family(Family::SansSerif);
    buf.set_text(text, &attrs, Shaping::Advanced, None);
    buf.shape_until_scroll(fs, false);
    let mut w: f32 = 0.0;
    let mut lines: u32 = 0;
    for run in buf.layout_runs() {
        w = w.max(run.line_w);
        lines += 1;
    }
    TextMetrics {
        width: w,
        height: lines as f32 * line_height_px,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_produces_glyph_per_ascii_letter() {
        let mut t = TextResources::new();
        let g = t.shape("abc", 16.0, 20.0);
        assert_eq!(g.len(), 3);
        // Each glyph has a distinct cache key (different glyph ids).
        let mut keys: Vec<_> = g.iter().map(|s| s.cache_key).collect();
        keys.sort_by_key(|k| k.glyph_id);
        keys.dedup();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn rasterize_roundtrip_produces_mask_bytes() {
        let mut t = TextResources::new();
        let g = t.shape("A", 32.0, 40.0);
        assert_eq!(g.len(), 1);
        let r = t.rasterize(g[0].cache_key).expect("rasterize A");
        assert!(r.width > 0 && r.height > 0, "empty bitmap");
        assert_eq!(
            r.data.len(),
            (r.width * r.height) as usize,
            "row-major R8 invariant"
        );
        assert!(
            r.data.iter().any(|&b| b > 0),
            "glyph bitmap has no non-zero coverage"
        );
    }

    #[test]
    fn measure_reports_nonzero_box() {
        let mut t = TextResources::new();
        let m = t.measure("hello", 16.0, 20.0);
        assert!(m.width > 0.0);
        assert!(m.height >= 20.0);
    }

    #[test]
    fn shape_cache_hits_on_repeat() {
        let mut t = TextResources::new();
        let _ = t.shape("repeat", 16.0, 20.0);
        assert_eq!(t.shape_cache.len(), 1);
        let _ = t.shape("repeat", 16.0, 20.0);
        assert_eq!(t.shape_cache.len(), 1, "second call should hit cache");
        // Different size → fresh entry.
        let _ = t.shape("repeat", 18.0, 22.0);
        assert_eq!(t.shape_cache.len(), 2);
    }

    #[test]
    fn measure_cache_hits_on_repeat() {
        let mut t = TextResources::new();
        let m1 = t.measure("hello", 16.0, 20.0);
        let m2 = t.measure("hello", 16.0, 20.0);
        assert_eq!(m1.width, m2.width);
        assert_eq!(t.measure_cache.len(), 1);
    }

    #[test]
    fn truncated_text_returns_original_when_fits() {
        let mut t = TextResources::new();
        // Full string fits in 1000 px at 16pt easily.
        let s = t.truncated_text("hello", 16.0, 20.0, 1000.0);
        assert_eq!(s, "hello");
    }

    #[test]
    fn truncated_text_appends_ellipsis_when_too_wide() {
        let mut t = TextResources::new();
        let full_w = t.measure("hello world", 16.0, 20.0).width;
        // Cap at half — must truncate.
        let s = t.truncated_text("hello world", 16.0, 20.0, full_w * 0.5);
        assert!(s.ends_with('\u{2026}'), "expected ellipsis suffix: {s:?}");
        assert!(s.chars().count() < "hello world".chars().count() + 1);
        let new_w = t.measure(&s, 16.0, 20.0).width;
        assert!(new_w <= full_w * 0.5 + 0.5, "truncated width {} exceeds cap {}", new_w, full_w * 0.5);
    }

    #[test]
    fn measure_constrained_matches_truncated_form() {
        let mut t = TextResources::new();
        let full_w = t.measure("a long string", 16.0, 20.0).width;
        let cap = full_w * 0.3;
        let constrained = t.measure_constrained("a long string", 16.0, 20.0, cap);
        assert!(constrained.width <= cap + 0.5);
    }

    #[test]
    fn shape_constrained_emits_fewer_glyphs_than_full() {
        let mut t = TextResources::new();
        let full = t.shape("the quick brown fox", 16.0, 20.0);
        let full_w = t.measure("the quick brown fox", 16.0, 20.0).width;
        let trunc = t.shape_constrained("the quick brown fox", 16.0, 20.0, full_w * 0.4);
        assert!(trunc.len() < full.len());
    }

    #[test]
    fn clear_shape_cache_drops_entries() {
        let mut t = TextResources::new();
        let _ = t.shape("hello", 16.0, 20.0);
        let _ = t.measure("hello", 16.0, 20.0);
        t.clear_shape_cache();
        assert!(t.shape_cache.is_empty());
        assert!(t.measure_cache.is_empty());
    }
}

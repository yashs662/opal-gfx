//! SVG rasterization via resvg + tiny-skia.
//!
//! Produces RGBA8 bytes in the same layout as [`crate::App::stage_image_rgba`]
//! / [`crate::App::upload_image_rgba`] expect. The typical pattern is:
//!
//! ```ignore
//! let bytes = include_bytes!("../assets/icon.svg");
//! let handle = app.stage_image_svg(bytes, 64);  // 64 × 64 physical px
//! scene.image((), handle).w_px(20.0).h_px(20.0).color([1.0, 1.0, 1.0, 1.0]);
//! ```
//!
//! Rasterization size is in **physical** px. Over-sample relative to your
//! biggest display size (e.g. 64 covers 16-32 px icons at 1×-2× DPI) so
//! the image-atlas bilinear filter downsamples cleanly without aliasing.

use resvg::usvg;
use tiny_skia::{Pixmap, Transform};

/// Rasterize an SVG byte slice to a square `px × px` RGBA8 buffer.
///
/// Channel layout matches [`crate::App::stage_image_rgba`]:
/// `[R, G, B, A]` per pixel, row-major, top-left origin. Internally
/// tiny-skia produces BGRA so we byte-swap channels 0 and 2 before
/// returning.
///
/// The SVG is scaled uniformly to fit within `px × px` — the smaller
/// of `px / svg.width` and `px / svg.height` is used so the icon's
/// aspect ratio is preserved.
///
/// Panics on invalid SVG bytes or allocation failure. SVGs in this
/// library context are typically `include_bytes!`-d static assets, so a
/// panic on bad input is the right failure mode — bad bytes mean the
/// build is broken.
pub fn rasterize_svg(bytes: &[u8], px: u32) -> Vec<u8> {
    rasterize_svg_to(bytes, px, px)
}

/// Rasterize to an arbitrary `w × h` buffer (use this when the source
/// SVG isn't square or you want letterboxed output). Otherwise prefer
/// [`rasterize_svg`].
pub fn rasterize_svg_to(bytes: &[u8], w: u32, h: u32) -> Vec<u8> {
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(bytes, &opt).expect("invalid SVG bytes");
    let mut pixmap = Pixmap::new(w, h).expect("alloc pixmap");
    let svg_size = tree.size();
    let scale_x = w as f32 / svg_size.width();
    let scale_y = h as f32 / svg_size.height();
    let scale = scale_x.min(scale_y);
    let xform = Transform::from_scale(scale, scale);
    resvg::render(&tree, xform, &mut pixmap.as_mut());
    // tiny_skia stores native BGRA; the image atlas expects RGBA.
    let mut data = pixmap.take();
    for px in data.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: &[u8] = br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="white"/></svg>"#;

    #[test]
    fn rasterize_produces_rgba_buffer_of_correct_size() {
        let rgba = rasterize_svg(SQUARE, 16);
        assert_eq!(rgba.len(), 16 * 16 * 4);
        // The white rect should produce fully-opaque white pixels at
        // the centre (after BGRA → RGBA swap, all four channels = 255).
        let mid = (8 * 16 + 8) * 4;
        assert_eq!(rgba[mid], 255);
        assert_eq!(rgba[mid + 1], 255);
        assert_eq!(rgba[mid + 2], 255);
        assert_eq!(rgba[mid + 3], 255);
    }

    #[test]
    fn rasterize_handles_non_square_target() {
        let rgba = rasterize_svg_to(SQUARE, 32, 8);
        assert_eq!(rgba.len(), 32 * 8 * 4);
    }
}

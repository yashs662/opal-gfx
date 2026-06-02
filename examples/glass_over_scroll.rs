//! P4 generic-glass correctness: a glass panel over a **promoted scroll
//! layer**. The scroll layer rasters content-local into a tall texture
//! and composites a windowed slice; the glass above must blur that
//! composited window — and when the list scrolls (composite-only, the
//! window's `src_origin` moves, no content re-raster), the glass backdrop
//! must follow.
//!
//! Exercises the path `glass_grid` (single-root) and `glass_over_layer`
//! (plain layer) don't: a `window`-bearing layer feeding the backdrop.
//!
//!     cargo run --example glass_over_scroll

use frostify_gfx::{App, Len, Scene};

const W: u32 = 600;
const H: u32 = 400;
const ROWS: u32 = 60;
const ROW_H: f32 = 40.0;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        // Dark backdrop so the (bright) scrolled rows are the only colour
        // the glass can pick up.
        p.rect("bg")
            .abs(0.0, 0.0)
            .size_px(W as f32, H as f32)
            .rgba(0.06, 0.06, 0.08, 1.0);

        // Promoted scroll list filling the window — its own scroll layer.
        // Rows alternate dark / bright-green so a vertical scroll changes
        // which colour band sits under the glass.
        p.col("list")
            .abs(0.0, 0.0)
            .size_px(W as f32, H as f32)
            .scroll_y()
            .layer()
            .child(|c| {
                for i in 0..ROWS {
                    let bright = i % 2 == 0;
                    let (r, g, b) = if bright { (0.1, 0.9, 0.3) } else { (0.06, 0.06, 0.08) };
                    c.rect(format!("row{i}"))
                        .w(Len::Fill)
                        .h_px(ROW_H)
                        .rgba(r, g, b, 1.0);
                }
            });

        // Glass panel over the centre. Blurs whatever rows are composited
        // beneath it right now.
        // Low blur (8 px ≪ ROW_H) so the bright/dark banding survives the
        // blur and a scroll visibly shifts the average under the glass.
        p.glass("glass")
            .abs(150.0, 120.0)
            .size_px(300.0, 160.0)
            .radius(20.0)
            .blur(8.0)
            .rgba(1.0, 1.0, 1.0, 0.06);
    });
}

fn avg_under_glass(rgba: &[u8], w: u32, h: u32) -> (f32, f32, f32) {
    let (cx, cy) = (300u32 * (w / W), 200u32 * (h / H));
    let half = 24u32;
    let (mut r, mut g, mut b, mut n) = (0f64, 0f64, 0f64, 0f64);
    for y in cy.saturating_sub(half)..(cy + half).min(h) {
        for x in cx.saturating_sub(half)..(cx + half).min(w) {
            let o = ((y * w + x) * 4) as usize;
            r += rgba[o] as f64;
            g += rgba[o + 1] as f64;
            b += rgba[o + 2] as f64;
            n += 1.0;
        }
    }
    if n == 0.0 { return (0.0, 0.0, 0.0); }
    ((r / n) as f32, (g / n) as f32, (b / n) as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("frostify-gfx — glass over scroll", W, H)
        .scene(build_scene)
        .headless(|h| {
            let list = h.ctx.node("list").expect("list node");
            // The scroll layer must exist + carry a window (it's promoted +
            // scrollable) and be below the glass.
            let has_scroll_layer = h
                .layer_tree
                .layers()
                .iter()
                .any(|l| l.root == Some(list) && l.window.is_some());
            assert!(has_scroll_layer, "expected a windowed scroll layer for `list`");

            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let g0 = avg_under_glass(&rgba, cw, ch).1;
            log::info!("scroll 0    → under-glass green = {g0:.0}");

            // Scroll one full row so the bright/dark banding under the
            // glass inverts. This re-flattens (content-local bytes are
            // unchanged → the scroll layer's raster is skipped; the window
            // src_origin moves). The glass backdrop must re-blur the new
            // window — its average green should change materially.
            h.ctx.tree.set_scroll_target(list, [0.0, ROW_H]);
            for _ in 0..240 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let g1 = avg_under_glass(&rgba, cw, ch).1;
            log::info!("scroll 1 row → under-glass green = {g1:.0}");

            assert!(
                (g1 - g0).abs() > 8.0,
                "glass backdrop must track the scrolled composite (green {g0:.0} → {g1:.0})"
            );
            log::info!("PASS: glass over a scroll layer blurs the scrolled composite");
            h.render();
            h.capture();
        });
    app.run()
}

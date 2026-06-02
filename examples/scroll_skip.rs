//! Compositor scroll-as-composite proof (P3 2a). A `.layer()`-promoted
//! `.scroll_y()` container over a tall content column. Its children
//! raster **content-local** into a tall texture once; scrolling moves the
//! composite sample window, so the scroll layer's content raster is
//! *skipped* (its instance bytes don't change) — only the root layer
//! (container chrome + scrollbar thumb) re-rasters.
//!
//! Headless: flush + render once (initial raster of every layer), scroll,
//! flush + render again, then assert the scroll layer skipped its raster.
//!
//!     cargo run --example scroll_skip

use frostify_gfx::{App, Len, Scene};

const W: u32 = 500;
const H: u32 = 600;
const ROWS: usize = 60;
const ROW_H: f32 = 44.0;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        // Promoted scroll container: clips + scrolls its tall content.
        p.col("list")
            .w(Len::Fill)
            .h(Len::Fill)
            .scroll_y()
            .layer()
            // Always-visible bar so the thumb-layer promotion is exercised
            // headlessly (auto-hide would keep bar_alpha=0 with no input).
            .scrollbar(|sb| sb.always_visible(true))
            .child(|c| {
                for i in 0..ROWS {
                    let t = (i as f32) / (ROWS as f32);
                    // Clickable so each row emits a HitEntry — exercises
                    // hit-space (screen coords) vs visual (texture coords).
                    c.rect(format!("row{i}"))
                        .w(Len::Fill)
                        .h_px(ROW_H)
                        .rgba(0.12 + t * 0.4, 0.2, 0.5 - t * 0.3, 1.0)
                        .on_click(|_| {});
                }
            });
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("frostify-gfx — scroll skip", W, H)
        .scene(build_scene)
        .headless(|h| {
            // `resumed` already flushed + rendered once (every layer
            // rastered). Find the promoted scroll layer.
            let list = h.ctx.node("list").expect("list node");
            let layers = h.layer_tree.layers().len();
            let has_scroll_layer = h
                .layer_tree
                .layers()
                .iter()
                .any(|l| l.root == Some(list) && l.window.is_some());
            assert!(
                has_scroll_layer,
                "expected a promoted scroll layer for `list`, got {layers} layers"
            );

            // Scroll down a few rows. This re-flattens (scrollbar thumb
            // moves in the root layer), but the scroll layer's children
            // are content-local so their bytes are unchanged → its raster
            // is skipped.
            h.ctx.tree.set_scroll_target(list, [0.0, 300.0]);
            // Settle the spring so `current` reaches the target before we
            // snapshot (otherwise the window is mid-animation, still valid
            // but harder to assert on).
            for _ in 0..240 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            let after = h.last_render_stats.expect("stats");
            log::info!(
                "after scroll: layers={} raster={} composite={}",
                after.layer_count,
                after.raster_count,
                after.composite_count
            );

            // The scroll layer must NOT have re-rastered (content-local
            // bytes unchanged). raster_count counts layers that rastered;
            // it must be strictly less than the layer count (the scroll
            // layer was skipped). Root may re-raster (scrollbar thumb).
            assert!(
                after.raster_count < after.layer_count,
                "scroll must skip the content layer's raster (raster={} layers={})",
                after.raster_count,
                after.layer_count
            );

            // Hit-space check: after scrolling 300 px (~6.8 rows), a click
            // at screen y=10 (top of the viewport) must land on the row now
            // visible there — NOT the texture-local row 0. The list scrolls
            // by whole-row snap; at scroll≈300 the top visible row is ~7.
            // `hits` were rebuilt by `flush`; pick the topmost at that point.
            let scroll_y = h.ctx.tree.get(list).and_then(|n| n.scroll.as_ref()).map(|s| s.current[1]).unwrap_or(0.0);
            let hit = frostify_gfx::InputState::hit_test(h.hits, 12.0, 10.0);
            // Recover the row index by matching the hit id against named rows.
            let hit_name =
                hit.and_then(|id| (0..ROWS).find(|i| h.ctx.node(&format!("row{i}")) == Some(id)));
            // Content y under the click = scroll + click_y (physical);
            // row = that / (row height × scale). Proves the hit landed on
            // the *scrolled* row, not texture-local row 0.
            let click_y = 10.0_f32;
            let expect_row = ((scroll_y + click_y) / (ROW_H * h.scale_factor)).floor() as usize;
            log::info!(
                "after scroll {scroll_y} (scale {}): top hit = row {hit_name:?}, expect {expect_row}",
                h.scale_factor
            );
            assert_eq!(
                hit_name,
                Some(expect_row),
                "hit-test must pick the scrolled row {expect_row}, not a texture-local row"
            );

            // Thumb-layer promotion: with an always-visible bar the moving
            // thumb is its own composite layer (window=Some), separate from
            // the content layer. So there are ≥2 windowed layers now.
            let windowed = h
                .layer_tree
                .layers()
                .iter()
                .filter(|l| l.window.is_some())
                .count();
            assert!(
                windowed >= 2,
                "expected a content layer + a promoted thumb layer (windowed={windowed})"
            );

            // Composite-only thumb move: scroll a tiny sub-row amount so no
            // new rows materialize. Both the content layer AND the thumb
            // layer keep stable instance bytes → EVERY raster is skipped;
            // only the composite windows move. This is the full win — a
            // pure scroll with raster 0 (the thumb no longer forces a root
            // re-raster).
            h.ctx.tree.set_scroll_target(list, [0.0, 304.0]);
            for _ in 0..240 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            let win2 = h.last_render_stats.expect("stats");
            log::info!(
                "sub-row re-scroll: layers={} raster={} composite={}",
                win2.layer_count, win2.raster_count, win2.composite_count
            );
            assert_eq!(
                win2.raster_count, 0,
                "a settled sub-row scroll must skip every raster (thumb is composited, not rastered)"
            );

            log::info!("PASS: scroll moved the composite window without re-rastering content; hit-test lands on the scrolled row; thumb is its own composite layer (pure scroll → raster 0)");

            // Visual check: the window must sample the scrolled content
            // (rows ~7+ at the top), clipped to the viewport — not blank,
            // not garbage. Capture for eyeballing.
            h.render();
            h.capture();
        });
    app.run()
}

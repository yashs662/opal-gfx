//! Compositor windowed-lazy scroll proof (P3 2b). A `.layer()`-promoted
//! **lazy list** of 5000 rows. Its texture is windowed to the materialized
//! rows (the full 5000×44 px virtual height would be ~hundreds of MB — non-
//! viable), so:
//!   - a **sub-window scroll** (stays within the materialized rows) moves
//!     only the composite sample origin → content raster skipped;
//!   - a **large scroll** crosses into fresh rows → materialize swaps the
//!     window's children → the content layer re-rasters (correct: new
//!     pixels), but still only the windowed texture, not 5000 rows.
//!
//!     cargo run --example lazy_scroll_skip

use frostify_gfx::{App, Len, Scene};

const W: u32 = 500;
const H: u32 = 600;
const ROWS: u32 = 5000;
const ROW_H: f32 = 44.0;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        p.lazy_list("list", ROWS, ROW_H, |row, i| {
            let t = (i % 32) as f32 / 32.0;
            row.rect(())
                .w(Len::Fill)
                .h_px(ROW_H - 4.0)
                .rgba(0.12 + t * 0.4, 0.2, 0.5 - t * 0.3, 1.0);
        })
        .w(Len::Fill)
        .h(Len::Fill)
        .layer();
    });
}

fn scroll_layer_present(h: &frostify_gfx::HeadlessHelper, list: frostify_gfx::NodeId) -> bool {
    h.layer_tree
        .layers()
        .iter()
        .any(|l| l.root == Some(list) && l.window.is_some())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("frostify-gfx — lazy scroll skip", W, H)
        .scene(build_scene)
        .headless(|h| {
            let list = h.ctx.node("list").expect("list node");
            // Materialize the initial window (resumed already flushed once,
            // but make the window explicit then re-flush so the lazy layer
            // exists with content).
            h.flush();
            h.render();
            assert!(
                scroll_layer_present(h, list),
                "expected a windowed lazy scroll layer for `list`"
            );
            let tex_h_rows = h
                .layer_tree
                .layers()
                .iter()
                .find(|l| l.root == Some(list))
                .and_then(|l| l.window)
                .map(|w| w.content[1])
                .unwrap_or(0.0);
            log::info!(
                "lazy layer texture height = {tex_h_rows} px (NOT {} px full virtual)",
                ROWS as f32 * ROW_H
            );
            assert!(
                tex_h_rows < ROWS as f32 * ROW_H * 0.2,
                "texture must be windowed, not full virtual height"
            );

            // (1) Sub-window scroll: 20 px (< one row). Stays within the
            // materialized window → no row materialize → content raster
            // skipped (only the composite sample origin moves).
            h.ctx.tree.set_scroll_target(list, [0.0, 20.0]);
            for _ in 0..240 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            let sub = h.last_render_stats.expect("stats");
            log::info!(
                "sub-window scroll: layers={} raster={} composite={}",
                sub.layer_count, sub.raster_count, sub.composite_count
            );
            assert!(
                sub.raster_count < sub.layer_count,
                "sub-window scroll must skip the content layer raster (raster={} layers={})",
                sub.raster_count, sub.layer_count
            );

            // (2) Large scroll: jump 2000 px (~45 rows). Crosses into fresh
            // rows → materialize swaps children → content layer re-rasters
            // (new pixels) — but only the windowed texture.
            h.ctx.tree.set_scroll_target(list, [0.0, 2000.0]);
            for _ in 0..240 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            let big = h.last_render_stats.expect("stats");
            log::info!(
                "cross-row scroll: layers={} raster={} composite={}",
                big.layer_count, big.raster_count, big.composite_count
            );
            assert!(
                big.raster_count >= 1,
                "crossing into fresh rows must re-raster the windowed content"
            );

            h.render();
            h.capture();
            log::info!("PASS: windowed lazy layer — sub-window scroll skips raster, row-cross re-rasters the window");
        });
    app.run()
}

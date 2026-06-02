//! Compositor raster-skip proof (P3). A static background plus a small
//! `.layer()`-promoted box. Frame 1 rasterizes every layer; then we move
//! the promoted layer via its **composite offset only** and re-render —
//! no re-flatten, no re-raster. The example asserts `raster_count` drops
//! to 0 on that composite-only frame while `composite_count` stays > 0.
//!
//! Run headless (renders two frames, checks stats, exits):
//!     cargo run --example layer_skip

use frostify_gfx::gpu::LayerDraw;
use frostify_gfx::{App, Len, Scene};

const W: u32 = 600;
const H: u32 = 400;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        // Static background: a grid of rects (the root layer).
        for i in 0..20 {
            let x = (i % 5) as f32 * 110.0 + 10.0;
            let y = (i / 5) as f32 * 90.0 + 10.0;
            p.rect(format!("bg{i}"))
                .abs(x, y)
                .size_px(100.0, 80.0)
                .rgba(0.15, 0.15 + (i as f32) * 0.02, 0.25, 1.0);
        }
        // Promoted moving box — its own compositor layer.
        p.rect("mover")
            .layer()
            .abs(40.0, 40.0)
            .size_px(120.0, 120.0)
            .rgba(0.9, 0.3, 0.2, 1.0);
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    // The shell's `resumed` already did the initial flush + render (which
    // rasterizes every layer once). The headless closure then probes the
    // damage split directly.
    let app = App::new("frostify-gfx — layer skip", W, H)
        .scene(build_scene)
        .headless(|h| {
            let rebuild_draws = |h: &mut frostify_gfx::HeadlessHelper| {
                let draws: Vec<LayerDraw> = h
                    .layer_tree
                    .layers()
                    .iter()
                    .map(|l| LayerDraw {
                        instances: (l.instances.start as u32)..(l.instances.end as u32),
                        offset: l.offset,
                        scale: l.scale,
                        opacity: l.opacity,
                        z: l.z,
                        window: None,
                    })
                    .collect();
                h.gpu.set_layers(&draws);
            };

            let mover = h
                .layer_tree
                .layers()
                .iter()
                .find_map(|l| l.root)
                .expect("a promoted layer");
            let layers = h.layer_tree.layers().len() as u32;
            assert!(layers >= 2, "expected root + promoted layer, got {layers}");

            // (1) Composite-only move: shift the promoted layer's offset
            // (no tree mutation, no flatten) and recomposite. Content is
            // unchanged, so EVERY raster is skipped — the GPU win.
            assert!(h.layer_tree.set_offset(mover, [80.0, 0.0]));
            rebuild_draws(h);
            h.render();
            let win = h.last_render_stats.expect("stats");
            log::info!(
                "composite-only move: layers={} raster={} composite={}",
                win.layer_count,
                win.raster_count,
                win.composite_count
            );
            assert_eq!(win.raster_count, 0, "composite-only move must skip ALL rasters");
            assert_eq!(win.composite_count, layers, "every layer still composites");

            // (2) Content change: recolor the promoted node, flush, render.
            // Its bytes change → its layer (only) re-rasterizes, proving the
            // skip is content-aware, not "never raster".
            h.ctx.tree.set_color(mover, [0.2, 0.8, 0.4, 1.0]);
            h.flush();
            h.render();
            let content = h.last_render_stats.expect("stats");
            log::info!(
                "content change: layers={} raster={} composite={}",
                content.layer_count,
                content.raster_count,
                content.composite_count
            );
            assert!(content.raster_count >= 1, "a content change must re-raster");

            log::info!("PASS: composite-only move skipped rasters; content change re-rastered");
        });
    app.run()
}

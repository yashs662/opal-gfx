//! P4 generic-glass proof: a glass panel blurs the **composite of a
//! promoted layer below it**, not a raw instance prefix.
//!
//! Scene: a full-window background, a `.layer()`-promoted coloured box
//! (its own compositor layer, painted *below* the glass), and a glass
//! panel on top. Moving the box via `set_layer_offset` is a
//! **composite-only** change (no re-raster of the box or the glass's own
//! content) — yet the glass must show the box **at its moved position**,
//! because the backdrop is re-sourced from the live composite of the
//! layers below the glass.
//!
//! This is what the single-root `glass_grid` parity gate can't cover: it
//! has no layer below its glass. Run headless:
//!     cargo run --example glass_over_layer

use frostify_gfx::gpu::LayerDraw;
use frostify_gfx::{App, Len, Scene};

const W: u32 = 600;
const H: u32 = 400;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        // Full-window dark background (root layer, behind everything).
        p.rect("bg")
            .abs(0.0, 0.0)
            .size_px(W as f32, H as f32)
            .rgba(0.08, 0.08, 0.10, 1.0);

        // Promoted bright box — its own compositor layer, painted below
        // the glass. Starts at the left; we slide it under the glass.
        p.rect("box")
            .layer()
            .abs(40.0, 150.0)
            .size_px(120.0, 100.0)
            .rgba(0.95, 0.30, 0.20, 1.0);

        // Glass panel covering the centre. Heavy blur so the box behind
        // it reads as a soft colour wash — and that wash must track the
        // box's composited position.
        p.glass("glass")
            .abs(200.0, 80.0)
            .size_px(200.0, 240.0)
            .radius(20.0)
            .blur(40.0)
            .rgba(1.0, 1.0, 1.0, 0.08);
    });
}

/// Average colour under the glass centre (a small window), read from a
/// captured RGBA frame. Used to detect whether the box's colour bled
/// into the blurred backdrop.
fn avg_under_glass(rgba: &[u8], w: u32, h: u32) -> (f32, f32, f32) {
    // Glass spans x∈[200,400], y∈[80,320] logical. At scale 2 the capture
    // is 2w×2h; sample a central patch in physical px.
    let (cx, cy) = (300u32 * (w / W), 200u32 * (h / H));
    let half = 20u32;
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
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    ((r / n) as f32, (g / n) as f32, (b / n) as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("frostify-gfx — glass over layer", W, H)
        .scene(build_scene)
        .headless(|h| {
            let boxid = h
                .layer_tree
                .layers()
                .iter()
                .find_map(|l| l.root)
                .expect("a promoted box layer");
            let layers = h.layer_tree.layers().len();
            assert!(layers >= 2, "expected root + promoted box layer, got {layers}");

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
                        external: None,
                        corner_radius: 0.0,
                        edge_fade: [0.0; 4],
                        edge_fade_falloff: 1.0,
                    })
                    .collect();
                h.gpu.set_layers(&draws);
            };

            // (1) Box at the left (x=40), not under the glass (x∈[200,400]).
            // The glass samples mostly the dark background.
            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let (r0, g0, b0) = avg_under_glass(&rgba, cw, ch);
            log::info!("box LEFT  → under-glass avg = ({r0:.0}, {g0:.0}, {b0:.0})");

            // (2) Composite-only move: slide the box right so it sits under
            // the glass (x≈300). No re-raster of the box layer or the glass
            // layer — only the box's composite offset changes. The glass
            // backdrop must pick up the box's new position.
            h.layer_tree.set_offset(boxid, [220.0, 0.0]); // 40+220=260, under glass
            rebuild_draws(h);
            // The backdrop must re-blur now that a below-glass layer moved.
            h.gpu.mark_backdrop_dirty();
            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let (r1, g1, b1) = avg_under_glass(&rgba, cw, ch);
            log::info!("box UNDER → under-glass avg = ({r1:.0}, {g1:.0}, {b1:.0})");

            // The box is bright red. Once it slides under the glass, the
            // blurred backdrop the glass samples must gain red + lose the
            // dark-blue background → red channel rises clearly.
            assert!(
                r1 > r0 + 20.0,
                "glass backdrop must follow the composited box (red {r0:.0} → {r1:.0})"
            );
            log::info!("PASS: glass blurs the composite of the layer below it (composite-only move tracked)");
            h.render();
            h.capture();
        });
    app.run()
}

//! Headless smoke test for the scrollbar overlay. Drives a small
//! scrolled target onto an outer + inner scroll container, settles
//! the spring, and captures one frame so the bars are visible.
//!
//! Run with:
//!     cargo run --example scrollbar_check

use opal_gfx::{App, Justify, Len, Scene};

const W: u32 = 540;
const H: u32 = 540;

fn build(s: &mut Scene) {
    s.col("root").fill().rgba(0.06, 0.07, 0.09, 1.0).child(|root| {
        root.col("list")
            .w(Len::Fill)
            .h(Len::Fill)
            .pad(12.0)
            .gap(8.0)
            .scroll_y()
            .child(|list| {
                for i in 0..30u32 {
                    list.row(format!("row{i}"))
                        .w(Len::Fill)
                        .h_px(36.0)
                        .pad_xy(12.0, 6.0)
                        .justify(Justify::Start)
                        .rgba(
                            0.3 + (i as f32 * 0.02) % 0.4,
                            0.4,
                            0.55,
                            1.0,
                        )
                        .radius(6.0);
                }
            });
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("scrollbar check", W, H, ())
        .scene(|s, _| build(s))
        .capture(1, "debug_captures/scrollbar")
        .headless(|h| {
            // Push the outer list partway down then settle so the bar
            // is at full alpha and the thumb is mid-track.
            let id = h
                .ctx
                .names
                .get("list")
                .copied()
                .expect("list node missing");
            h.ctx.tree.set_scroll_target(id, [0.0, 200.0]);
            // Snap the spring so capture shows the settled state.
            for _ in 0..30 {
                h.ctx.tree.tick_scrolls(1.0 / 60.0);
            }
            h.flush();
            h.render();
            h.capture();
        });
    app.run()
}

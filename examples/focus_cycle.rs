//! Tab-focus cycling smoke test. Five focusable tiles in a
//! column, each opted into the Tab order via `.focus_order(n)` and
//! wired to an `on_focus` signal that drives a reactive highlight.
//!
//! - **Tab** moves focus to the next tile (ascending `focus_order`).
//! - **Shift+Tab** moves to the previous tile.
//! - Focus wraps around both ends.
//! - The tiles are authored out of order on purpose (focus_order
//!   3,1,4,2,5) to show the cycle follows `focus_order`, not creation
//!   order. The middle tile is a real text field — Tab leaves it.
//!
//! Run with:
//!     cargo run --example focus_cycle

use frostify_gfx::{App, Computed, Len, Scene, Signal, deps};

const W: u32 = 520;
const H: u32 = 460;

const DIM: [f32; 4] = [0.16, 0.17, 0.21, 1.0];
const LIT: [f32; 4] = [0.20, 0.55, 0.95, 1.0];

fn tile(s: &mut Scene, name: &str, label: &str, order: u32) {
    let focused = Signal::new(false);
    let bg = Computed::new(deps!(focused), |(f,)| if f { LIT } else { DIM });
    s.row(name)
        .w(Len::Fill)
        .h_px(56.0)
        .radius(8.0)
        .color(bg)
        .on_focus(focused)
        .focus_order(order)
        .pad_xy(16.0, 0.0)
        .child(|p| {
            p.text((), label, 16.0).color([1.0, 1.0, 1.0, 0.9]);
        });
}

fn build(s: &mut Scene) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(24.0)
        .gap(12.0)
        .child(|p| {
            p.text("title", "Tab / Shift+Tab to cycle focus", 18.0)
                .color([1.0, 1.0, 1.0, 0.85]);
            // Authored top-to-bottom but with shuffled focus_order so
            // the visited sequence is row-b, row-d, row-a, field, row-e.
            tile(p, "row_a", "Tile A  (focus_order 3)", 3);
            tile(p, "row_b", "Tile B  (focus_order 1)", 1);
            tile(p, "row_d", "Tile D  (focus_order 4)", 4);

            // A text field in the cycle: focus_order 2 means Tab visits
            // it after Tile B; pressing Tab again leaves it for Tile A.
            let focused = Signal::new(false);
            let bg = Computed::new(deps!(focused), |(f,)| if f { LIT } else { DIM });
            p.text_field("field", "", 14.0)
                .placeholder("focus_order 2 — type, then Tab away…")
                .w(Len::Fill)
                .h_px(56.0)
                .pad_xy(16.0, 8.0)
                .radius(8.0)
                .color(bg)
                .on_focus(focused)
                .focus_order(2);

            tile(p, "row_e", "Tile E  (focus_order 5)", 5);
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("focus cycle", W, H);
    let app = app.scene(build);
    app.run()
}

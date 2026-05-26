//! 10,000-row virtualized list. Demonstrates that scroll cost stays
//! flat regardless of total item count — only the visible ~25 rows
//! are ever materialized as real tree nodes.
//!
//! Wheel scroll, drag the thumb, or use Page/Home/End / arrow keys
//! to navigate. The label at the top shows the currently visible
//! row range; watch it update at each cross of a row boundary.
//!
//! Run with:
//!     cargo run --example lazy_list

mod common;

use std::rc::Rc;

use frostify_gfx::{App, BarSide, Justify, Len, Scene};

use common::image::hsv;

const W: u32 = 540;
const H: u32 = 700;
const ROWS: u32 = 100_000;
const ROW_H: f32 = 36.0;
const GAP: f32 = 0.0;

fn build(s: &mut Scene) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(12.0)
        .gap(8.0)
        .child(|p| {
            p.text(
                "header",
                format!("{ROWS} rows — only the visible window is materialized"),
                14.0,
            )
            .color([1.0, 1.0, 1.0, 0.85]);

            // The lazy list itself is a scroll container — chain
            // builder methods on it directly. The render closure runs
            // once per visible row at flush time.
            let items: Rc<Vec<String>> = Rc::new(
                (0..ROWS).map(|i| format!("Track #{i:05}")).collect(),
            );
            let items_for_render = items.clone();
            p.lazy_list("playlist", ROWS, ROW_H, move |row, i| {
                let label = &items_for_render[i as usize];
                let c = hsv((i as f32 * 0.037).fract(), 0.45, 0.85);
                row.row(format!("row{i}"))
                    .w(Len::Fill)
                    .h_px(ROW_H)
                    .pad_xy(12.0, 6.0)
                    .gap(10.0)
                    .justify(Justify::Start)
                    .color(c)
                    .radius(4.0)
                    .child(|r| {
                        r.text(format!("row_lbl_{i}"), label, 13.0)
                            .color([0.0, 0.0, 0.0, 0.85]);
                    });
            })
            .w(Len::Fill)
            .h(Len::Fill)
            .scrollbar(|sb| {
                sb.thickness(10.0)
                    .min_thumb(30.0)
                    .margin(4.0)
                    .radius(5.0)
                    .y_side(BarSide::End)
                    .always_visible(true)
                    .track_color([0.10, 0.12, 0.18, 0.40])
                    .thumb_color([0.40, 0.65, 1.00, 0.55])
                    .thumb_hover_color([0.55, 0.80, 1.00, 0.85])
                    .thumb_active_color([0.85, 0.95, 1.00, 1.00])
            });

            // No GAP between rows (already counted in ROW_H). Touch
            // the const so the unused-import lint doesn't fire if the
            // file evolves later.
            let _ = GAP;
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("lazy_list", W, H).scene(build);
    app.run()
}

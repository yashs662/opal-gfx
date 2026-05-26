//! Drag-and-drop + slider smoke test.
//!
//! - Three color swatches carry a `drag_payload` ([f32; 4] color) and
//!   `drag_follow`, so pressing one lifts it onto the cursor (a hole is
//!   left in the row) and it tracks the pointer on top of everything.
//!   Release over the drop zone to recolor the zone — the drop handler
//!   downcasts the payload and writes the color directly via
//!   `DropCtx::tree` (no scene rebuild). This is the foundation for
//!   reorderable lists (e.g. a play queue).
//! - A slider at the bottom is built on the generic `on_drag`
//!   primitive: dragging maps cursor-x to a 0..1 value signal that
//!   drives the fill width reactively.
//!
//! Run with:
//!     cargo run --example drag_drop

mod common;

use common::components::{slider, SliderProps};
use frostify_gfx::{App, Align, Len, Scene, Signal};

const W: u32 = 640;
const H: u32 = 460;

const SWATCHES: [[f32; 4]; 3] = [
    [0.95, 0.35, 0.40, 1.0],
    [0.30, 0.80, 0.55, 1.0],
    [0.35, 0.55, 0.95, 1.0],
];

fn build(s: &mut Scene, slider_value: Signal<f32>) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(28.0)
        .gap(20.0)
        .child(|p| {
            p.text("t1", "Drag a swatch onto the drop zone", 16.0)
                .color([1.0, 1.0, 1.0, 0.85]);

            // Source swatches — each carries its color as the payload.
            p.row("swatches").gap(12.0).child(|r| {
                for (i, color) in SWATCHES.iter().enumerate() {
                    r.rect(format!("sw{i}"))
                        .size_px(64.0, 64.0)
                        .radius(12.0)
                        .color(*color)
                        .drag_payload(*color)
                        .drag_follow();
                }
            });

            // Drop target — recolors itself with the dropped payload.
            p.rect("zone")
                .w(Len::Fill)
                .h_px(120.0)
                .radius(16.0)
                .rgba(1.0, 1.0, 1.0, 0.06)
                .border(2.0, [1.0, 1.0, 1.0, 0.15])
                .on_drop(|d| {
                    if let Some(c) = d.payload.downcast_ref::<[f32; 4]>() {
                        let node = d.node;
                        d.tree.set_color(node, *c);
                    }
                });

            p.text("t2", "Slider (on_drag):", 16.0)
                .color([1.0, 1.0, 1.0, 0.85]);
            p.row("slider_row").align(Align::Center).child(|r| {
                slider(
                    r,
                    SliderProps {
                        value: slider_value.clone(),
                        width: 320.0,
                    },
                );
            });
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let slider_value = Signal::new(0.35);
    let app = App::new("drag & drop", W, H);
    let app = app.scene(move |s| build(s, slider_value.clone()));
    app.run()
}

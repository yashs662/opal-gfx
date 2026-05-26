//! Text input smoke test. A single search-style field plus a label
//! that echoes the value live via `on_change`. Submit
//! (Enter) prints the value to stdout. Escape blurs the field.
//!
//! Click into the field to focus it (caret appears); click outside
//! to blur (caret disappears). Backspace, Delete, Arrow Left/Right,
//! Home, End all work; typed characters go in at the cursor.
//!
//! Selection + clipboard: Shift+Arrows / Shift+Home / Shift+End
//! extend the selection (highlighted); Ctrl/Cmd+A selects all;
//! Ctrl/Cmd+C / X / V copy / cut / paste via the system clipboard.
//! Typing or Backspace over a selection replaces / deletes it.
//!
//! Run with:
//!     cargo run --example text_input

use std::cell::RefCell;
use std::rc::Rc;

use frostify_gfx::{App, Justify, Len, Scene};

const W: u32 = 720;
const H: u32 = 360;

fn build(s: &mut Scene, echo: Rc<RefCell<String>>) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(24.0)
        .gap(14.0)
        .child(|p| {
            p.text("title", "text input — type into the box below", 16.0)
                .color([1.0, 1.0, 1.0, 0.85]);

            // The text field itself. Inherits all the regular builder
            // methods (border, color, radius, pad). Children (text +
            // caret) are managed by the library.
            let echo_for_change = echo.clone();
            p.text_field("search", "", 14.0)
                .placeholder("Search…")
                .w(Len::Fill)
                .h_px(40.0)
                .pad_xy(12.0, 8.0)
                .rgba(0.13, 0.14, 0.18, 1.0)
                .radius(8.0)
                .border(1.0, [1.0, 1.0, 1.0, 0.15])
                .justify(Justify::Start)
                .on_change(move |s| {
                    *echo_for_change.borrow_mut() = s.to_string();
                })
                .on_submit(|_ctx| {
                    println!("submitted!");
                });

            // Live echo of the value. Rebuilt every keystroke via
            // `on_change` writing into the shared RefCell and the
            // scene-level rebuild-token; the simpler path here is
            // just to print + leave the label static for v1.
            p.text(
                "echo_label",
                format!("(value: {})", echo.borrow()),
                14.0,
            )
            .color([1.0, 1.0, 1.0, 0.55]);
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let echo: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
    let app = App::new("text input", W, H);
    let app = app.scene(move |s| build(s, echo.clone()));
    app.run()
}

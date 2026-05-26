//! Modal dialog smoke test. A button opens a centered dialog over
//! a dimmed scrim; clicking the scrim (anywhere outside the panel) or
//! pressing Escape-on-nothing dismisses it. Clicks *inside* the panel
//! are absorbed and do not dismiss.
//!
//! Demonstrates the `App::on_unhandled_press` hook + the
//! `dismiss_transparent` scrim flag. Visibility is a caller-owned
//! `Signal<bool>`; show/hide drives a scene rebuild via the rebuild
//! token (no library magic — grep `rebuild.set(true)`).
//!
//! Run with:
//!     cargo run --example modal

mod common;

use common::components::{modal, ModalProps};
use frostify_gfx::{App, Align, Scene, Signal};

const W: u32 = 640;
const H: u32 = 480;

fn build(s: &mut Scene, visible: Signal<bool>, rebuild: std::rc::Rc<std::cell::Cell<bool>>) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(28.0)
        .gap(16.0)
        .align(Align::Start)
        .child(|p| {
            p.text("title", "Modal demo — click the button", 18.0)
                .color([1.0, 1.0, 1.0, 0.85]);

            let open_vis = visible.clone();
            let open_rebuild = rebuild.clone();
            p.row("open_btn")
                .w_px(160.0)
                .h_px(44.0)
                .center()
                .rgba(0.20, 0.55, 0.95, 1.0)
                .radius(10.0)
                .hover_color([0.30, 0.62, 1.0, 1.0])
                .on_click(move |_| {
                    open_vis.set(true);
                    open_rebuild.set(true);
                })
                .child(|b| {
                    b.text((), "Open dialog", 15.0).color([1.0, 1.0, 1.0, 0.95]);
                });
        });

    // Gated render: the overlay only exists in the tree while visible.
    if visible.get() {
        modal(
            s,
            ModalProps {
                title: "Delete playlist?".into(),
                body: "This can't be undone. Click outside or Esc to cancel."
                    .into(),
            },
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let visible = Signal::new(false);
    let app = App::new("modal", W, H);
    let rebuild = app.rebuild_token();

    // Outside-press dismiss: flip visibility off + request a rebuild.
    let dismiss_vis = visible.clone();
    let dismiss_rebuild = rebuild.clone();

    let scene_vis = visible.clone();
    let scene_rebuild = rebuild.clone();
    let app = app
        .scene(move |s| build(s, scene_vis.clone(), scene_rebuild.clone()))
        .on_unhandled_press(move |_| {
            if dismiss_vis.get() {
                dismiss_vis.set(false);
                dismiss_rebuild.set(true);
            }
        });
    app.run()
}

//! View-routing smoke test. Two views (Library / Search) swap via
//! `App::rebuild_scene`. Demonstrates:
//!
//!   - `Signal<View>` held outside the scene closure
//!   - Nav buttons whose `on_click` mutates the signal and flips the
//!     rebuild token returned by `App::rebuild_token()`
//!   - The shell consumes the token at the top of `about_to_wait` and
//!     re-runs the stored scene closure on an empty `SceneCtx`. Bind
//!     slots, named ids, and active tweens from the prior view all
//!     drop cleanly via the bind-cleanup pipeline.
//!
//! No magic: the rebuild fires because the user *explicitly* flipped
//! the token. Grep `rebuild.set(true)` to find every site that
//! triggers a rebuild — there is no implicit signal-watcher.
//!
//! Run with:
//!     cargo run --example routes

use std::cell::Cell;
use std::rc::Rc;

use frostify_gfx::{Align, App, Justify, Len, Scene, Signal, WindowAction};

const W: u32 = 720;
const H: u32 = 460;

#[derive(Copy, Clone, PartialEq, Eq)]
enum View {
    Library,
    Search,
}

const PINK: [f32; 4] = [0.95, 0.25, 0.55, 1.0];
const BLUE: [f32; 4] = [0.30, 0.55, 0.95, 1.0];
const DIM: [f32; 4] = [0.18, 0.20, 0.24, 1.0];
const PINK_HOVER: [f32; 4] = [1.0, 0.40, 0.70, 1.0];
const BLUE_HOVER: [f32; 4] = [0.50, 0.75, 1.0, 1.0];

fn build(s: &mut Scene, view: View, on_library: Rc<dyn Fn()>, on_search: Rc<dyn Fn()>) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(20.0)
        .gap(14.0)
        .child(|p| {
            // Title bar (drag-move only — no dots, this is a smoke
            // test, not the demo).
            p.row("title")
                .w(Len::Fill)
                .h_px(40.0)
                .pad(12.0)
                .align(Align::Center)
                .rgba(0.13, 0.14, 0.18, 1.0)
                .radius(10.0)
                .window_action(WindowAction::DragMove)
                .child(|t| {
                    t.text("title_lbl", "routes — click a nav button", 14.0)
                        .color([1.0, 1.0, 1.0, 0.9]);
                });

            // Nav row: two buttons. The active view paints in its
            // accent color; the inactive in dim.
            let (library_color, search_color) = match view {
                View::Library => (PINK, DIM),
                View::Search => (DIM, BLUE),
            };
            p.row("nav")
                .w(Len::Fill)
                .h_px(48.0)
                .gap(10.0)
                .justify(Justify::Start)
                .child(|n| {
                    let cb = on_library.clone();
                    n.rect("nav_library")
                        .w_px(140.0)
                        .h_px(40.0)
                        .color(library_color)
                        .hover_color(PINK_HOVER)
                        .radius(8.0)
                        .on_click(move |_| cb());
                    let cb = on_search.clone();
                    n.rect("nav_search")
                        .w_px(140.0)
                        .h_px(40.0)
                        .color(search_color)
                        .hover_color(BLUE_HOVER)
                        .radius(8.0)
                        .on_click(move |_| cb());
                    n.text("nav_lib_lbl", "Library", 14.0)
                        .abs(40.0, 12.0)
                        .color([1.0, 1.0, 1.0, 1.0]);
                    n.text("nav_search_lbl", "Search", 14.0)
                        .abs(190.0, 12.0)
                        .color([1.0, 1.0, 1.0, 1.0]);
                });

            // View body: branched per `View`. Both branches are simple
            // colored cards with a label so the swap is obvious.
            p.col("body")
                .w(Len::Fill)
                .h(Len::Fill)
                .pad(16.0)
                .gap(12.0)
                .rgba(0.10, 0.11, 0.14, 1.0)
                .radius(12.0)
                .child(|b| match view {
                    View::Library => {
                        b.rect("lib_card_1")
                            .w(Len::Fill)
                            .h_px(60.0)
                            .color(PINK)
                            .radius(8.0);
                        b.text("lib_lbl", "Library — your saved playlists.", 18.0)
                            .color([1.0, 1.0, 1.0, 0.95]);
                        b.rect("lib_card_2")
                            .w(Len::Fill)
                            .h_px(60.0)
                            .rgba(PINK[0] * 0.6, PINK[1] * 0.6, PINK[2] * 0.6, 1.0)
                            .radius(8.0);
                    }
                    View::Search => {
                        b.rect("search_box")
                            .w(Len::Fill)
                            .h_px(40.0)
                            .rgba(0.18, 0.20, 0.24, 1.0)
                            .radius(20.0)
                            .border(1.0, [1.0, 1.0, 1.0, 0.10]);
                        b.text("search_lbl", "Search — try the text_input demo for editing.", 18.0)
                            .color([1.0, 1.0, 1.0, 0.95]);
                        b.rect("search_result_1")
                            .w(Len::Fill)
                            .h_px(60.0)
                            .color(BLUE)
                            .radius(8.0);
                    }
                });
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    // Caller-owned state.
    let view = Signal::new(View::Library);

    let app = App::new("routes", W, H);

    // Token wakes the shell up for a rebuild. Clones into each
    // closure that wants to trigger a view swap.
    let rebuild = app.rebuild_token();

    // Per-button click callbacks: type-erased to `dyn Fn()` so the
    // scene closure (which re-runs on every rebuild) can capture them
    // by Rc-clone and re-install fresh `on_click`s each pass.
    let on_library: Rc<dyn Fn()> = {
        let view = view.clone();
        let rebuild = rebuild.clone();
        Rc::new(move || {
            if view.get() != View::Library {
                view.set(View::Library);
                rebuild.set(true);
            }
        })
    };
    let on_search: Rc<dyn Fn()> = {
        let view = view.clone();
        let rebuild = rebuild.clone();
        Rc::new(move || {
            if view.get() != View::Search {
                view.set(View::Search);
                rebuild.set(true);
            }
        })
    };

    let app = app.scene(move |s| build(s, view.get(), on_library.clone(), on_search.clone()));

    // First-run rebuild count, just so the cell variable isn't
    // optimized into nothing on early returns.
    let _ = Cell::new(0_u32);

    app.run()
}

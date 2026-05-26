//! frostify-gfx demo — declarative flex scene + reactive bindings,
//! with a z-order playground showing painter's-order layering across
//! all node kinds (rect / glass / text / image).
//!
//! Layout engine is custom, not taffy. Root is a padded column:
//! title bar + hero card + stage row. Stage holds a sidebar plus
//! a layered canvas where each child's declared order = paint order.
//!
//! The canvas exercises every interesting overlap:
//!   - rect / image / text declared *before* glass → blurred
//!   - rect / image / text declared *after* glass → crisp
//!   - two glass panes intersecting (each independently blurs what
//!     was declared above it; neither blurs the other — glass is
//!     skipped from the backdrop pass)
//!   - an animated cyan blob that crosses every layer; arrow keys
//!     prove its z-order stays put while moving.
//!
//! Reusable components live in `common::components`. This file owns
//! state (`Sigs`), wires the top-level layout + canvas (the canvas is
//! one-off demo art, intentionally not componentized), and runs the
//! event loop.
//!
//! Controls:
//!   Mouse            Hover the hero rect to brighten it; click to
//!                    toggle the "lit" base color (same as Space).
//!                    Drag the title bar to move the window. Click
//!                    red/yellow/green dots = close/minimize/maximize.
//!                    Drag any window edge / corner to resize.
//!   Space            Toggle the hero "lit" base color.
//!   Arrow Left/Right Tween the cyan blob horizontally (Bind&lt;Position&gt;).
//!   B                Toggle blob size (Bind&lt;Size&gt;).
//!   F1               Toggle HUD + stats log.
//!   F2               Screenshot to `debug_captures/`.
//!   F4               Overdraw heatmap.
//!   F5               Force full rebuild + redraw.
//!   Esc              Exit.

mod common;

use std::env;
use std::time::{Duration, Instant};

use frostify_gfx::{App, HeadlessHelper, ImageHandle, Len, Scene, Signal, WindowAction};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

use common::components::{
    blob, hero, sidebar, title_bar, BlobProps, DotProps, HeroProps, SidebarProps, TitleBarProps,
};
use common::image::make_demo_image;

const W: u32 = 1280;
const H: u32 = 820;

const DOTS: [[f32; 4]; 3] = [
    [0.95, 0.30, 0.30, 1.0],
    [0.95, 0.75, 0.20, 1.0],
    [0.30, 0.85, 0.40, 1.0],
];

const BLOB_X0: f32 = 60.0;
const BLOB_Y: f32 = 90.0;
const BLOB_SIZE_SMALL: [f32; 2] = [140.0, 100.0];
const BLOB_SIZE_LARGE: [f32; 2] = [220.0, 160.0];

#[derive(Clone)]
struct Sigs {
    lit: Signal<bool>,
    hover: Signal<bool>,
    pressed: Signal<bool>,
    focused: Signal<bool>,
    blob_pos: Signal<[f32; 2]>,
    blob_size: Signal<[f32; 2]>,
}

impl Sigs {
    fn new() -> Self {
        Self {
            lit: Signal::new(false),
            hover: Signal::new(false),
            pressed: Signal::new(false),
            focused: Signal::new(false),
            blob_pos: Signal::new([BLOB_X0, BLOB_Y]),
            blob_size: Signal::new(BLOB_SIZE_SMALL),
        }
    }
}

fn build_scene(s: &mut Scene, sigs: &Sigs, art: ImageHandle) {
    s.col("root")
        .fill()
        .pad(24.0)
        .gap(16.0)
        .rgba(0.0, 0.0, 0.0, 0.5)
        .radius(28.0)
        .border(1.5, [1.0, 1.0, 1.0, 0.10])
        .shadow([0.0, 16.0], 40.0, [0.0, 0.0, 0.0, 1.0], 0.55)
        .child(|p| {
            title_bar(
                p,
                TitleBarProps {
                    title: "frostify-gfx demo".into(),
                    dots: vec![
                        DotProps {
                            color: DOTS[0],
                            action: WindowAction::Close,
                        },
                        DotProps {
                            color: DOTS[1],
                            action: WindowAction::Minimize,
                        },
                        DotProps {
                            color: DOTS[2],
                            action: WindowAction::ToggleMaximize,
                        },
                    ],
                },
            );

            hero(
                p,
                HeroProps {
                    lit: sigs.lit.clone(),
                    hover: sigs.hover.clone(),
                    pressed: sigs.pressed.clone(),
                    focused: sigs.focused.clone(),
                },
            );

            p.row("stage")
                .w(Len::Fill)
                .h(Len::Fill)
                .gap(20.0)
                .child(|r| {
                    sidebar(r, SidebarProps { art });
                    r.col("canvas")
                        .w(Len::Fill)
                        .h(Len::Fill)
                        .pad(0.0)
                        .rgba(0.10, 0.11, 0.14, 1.0)
                        .radius(14.0)
                        .border(1.0, [1.0, 1.0, 1.0, 0.05])
                        .child(|c| {
                            // === BAND 1 — blurred-through-glass ===
                            // Three nodes declared *before* glass A.
                            c.rect("b1_back")
                                .abs(20.0, 20.0)
                                .size_px(360.0, 200.0)
                                .rgba(0.95, 0.20, 0.55, 1.0)
                                .radius(20.0);
                            c.image("b1_img", art)
                                .abs(60.0, 60.0)
                                .size_px(96.0, 96.0)
                                .radius(12.0);
                            c.text("b1_text", "BEHIND", 26.0)
                                .abs(180.0, 80.0)
                                .color([1.0, 1.0, 1.0, 0.95]);
                            c.glass("glass_a")
                                .abs(20.0, 130.0)
                                .size_px(440.0, 90.0)
                                .radius(18.0)
                                .blur(20.0)
                                .refraction(8.0)
                                .rgba(1.0, 1.0, 1.0, 0.10);
                            // Three nodes declared *after* glass A.
                            c.rect("b1_chip")
                                .abs(40.0, 156.0)
                                .size_px(60.0, 38.0)
                                .rgba(0.20, 0.95, 0.55, 1.0)
                                .radius(10.0)
                                .border(1.0, [0.0, 0.0, 0.0, 0.4]);
                            c.image("b1_img2", art)
                                .abs(116.0, 156.0)
                                .size_px(38.0, 38.0)
                                .radius(8.0);
                            c.text("b1_label", "in front of glass A", 18.0)
                                .abs(174.0, 162.0)
                                .color([1.0, 1.0, 1.0, 1.0]);
                            // === BAND 2 — vertical glass crossing ===
                            c.rect("b2_a")
                                .abs(500.0, 20.0)
                                .size_px(220.0, 90.0)
                                .rgba(0.30, 0.55, 0.95, 1.0)
                                .radius(14.0);
                            c.rect("b2_b")
                                .abs(500.0, 120.0)
                                .size_px(220.0, 90.0)
                                .rgba(0.95, 0.75, 0.20, 1.0)
                                .radius(14.0);
                            c.text("b2_lbl", "stacked rects", 16.0)
                                .abs(520.0, 50.0)
                                .color([1.0, 1.0, 1.0, 0.9]);
                            c.glass("glass_b")
                                .abs(420.0, 30.0)
                                .size_px(120.0, 230.0)
                                .radius(20.0)
                                .blur(28.0)
                                .refraction(10.0)
                                .rgba(1.0, 1.0, 1.0, 0.08);
                            c.text("b2_front", "GLASS B", 22.0)
                                .abs(440.0, 130.0)
                                .color([1.0, 1.0, 1.0, 1.0]);
                            // === BAND 3 — animated blob, declared last so
                            // it paints on top regardless of where it sits.
                            blob(
                                c,
                                BlobProps {
                                    pos: sigs.blob_pos.clone(),
                                    size: sigs.blob_size.clone(),
                                },
                            );
                        });
                });
        });
}

fn run_headless(h: &mut HeadlessHelper, sigs: Sigs) {
    if env::var_os("FROSTIFY_AUTOCAPTURE_HIT").is_some() {
        // Hit the hero rect after layout places it.
        let (cx, cy) = match h.ctx.node("hero").and_then(|id| h.ctx.tree.get(id)) {
            Some(n) => (n.rect[0] + n.rect[2] * 0.5, n.rect[1] + n.rect[3] * 0.5),
            None => return,
        };
        let _ = h.input.on_cursor_moved(cx, cy, h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();

        let _ = h.input.on_left_pressed(h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();

        let _ = h.input.on_left_released(h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_GLASS").is_some() {
        let cur = sigs.blob_pos.get();
        sigs.blob_pos.set([cur[0] + 120.0, cur[1]]);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_ANIM").is_some() {
        sigs.hover.set(true);
        h.react(Instant::now());
        let t0 = Instant::now();
        for step in 1..=6u32 {
            let sim = t0 + Duration::from_millis(step as u64 * 40);
            let res = h.timeline.tick(sim);
            if res.updated {
                h.react(Instant::now());
            }
            h.flush();
            h.render();
            h.capture();
        }
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_TOGGLE").is_some() {
        sigs.lit.set(true);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_OVERDRAW").is_some() {
        h.gpu.set_overdraw(true);
        h.flush();
        h.render();
        h.capture();
        h.gpu.set_overdraw(false);
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_HUD").is_some() {
        h.show_hud();
        h.render();
        h.capture();
        h.hide_hud();
    }
}

fn parse_capture_cli() -> Option<(u32, std::path::PathBuf)> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg != "--capture" {
            continue;
        }
        let mut frames: u32 = 1;
        let mut out = std::path::PathBuf::from("debug_captures");
        for kv in args.by_ref() {
            if kv.starts_with("--") {
                break;
            }
            if let Some(v) = kv.strip_prefix("frames=") {
                frames = v.parse().unwrap_or(1).max(1);
            } else if let Some(v) = kv.strip_prefix("out=") {
                out = std::path::PathBuf::from(v);
            }
        }
        return Some((frames, out));
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let sigs = Sigs::new();

    let scene_sigs = sigs.clone();
    let key_sigs = sigs.clone();

    let mut app = App::new("frostify-gfx", W, H);
    let (img_w, img_h, img_bytes) = make_demo_image();
    let art = app.stage_image_rgba(img_w, img_h, img_bytes);
    let mut app = app
        .scene(move |scene| build_scene(scene, &scene_sigs, art))
        .capture_from_env();
    if let Some((frames, dir)) = parse_capture_cli() {
        app = app.capture(frames, dir);
    }
    let mut app = app.on_key(move |code, state, _ctx| {
        if state != ElementState::Pressed {
            return;
        }
        match code {
            KeyCode::Space => {
                key_sigs.lit.set(!key_sigs.lit.get());
            }
            KeyCode::ArrowLeft | KeyCode::ArrowRight => {
                let delta = if code == KeyCode::ArrowLeft { -40.0 } else { 40.0 };
                let cur = key_sigs.blob_pos.get();
                key_sigs.blob_pos.set([cur[0] + delta, cur[1]]);
            }
            KeyCode::KeyB => {
                let cur = key_sigs.blob_size.get();
                let next = if cur == BLOB_SIZE_SMALL {
                    BLOB_SIZE_LARGE
                } else {
                    BLOB_SIZE_SMALL
                };
                key_sigs.blob_size.set(next);
            }
            _ => {}
        }
    });

    if env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
        let hsigs = sigs.clone();
        app = app.headless(move |h| run_headless(h, hsigs));
    }

    app.run()
}

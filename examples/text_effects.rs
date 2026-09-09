//! The per-glyph text effects, running on a stand-in lyric line — one row
//! per registered effect, so a shader edit can be eyeballed (and
//! screenshot-diffed) without the app.
//!
//! Each row runs the params the lyrics page feeds it,
//! `[active, start, duration, rate, frozen]`, on a line that restarts
//! every [`LINE_SECS`] — so the sweep plays over and over.
//!
//! `OPAL_EFFECT_TIME=<secs>` pins the shader clock, which is what makes a
//! capture of a given instant reproducible.
//!
//! Run with:
//!     cargo run --example text_effects

use opal_gfx::{Align, App, Len, Scene, TEXT_EFFECTS, TextFxId};

const W: u32 = 1280;
const H: u32 = 300;

/// A saturated album accent — what a lit line takes its colour from.
const ACCENT: [f32; 4] = [0.655, 0.545, 1.0, 1.0];
const BG: [f32; 4] = [0.05, 0.05, 0.07, 1.0];
const LINE: &str = "and the walls kept tumbling down";
/// How long a demo line lasts before it starts over.
const LINE_SECS: f32 = 6.0;

/// Node name of the `i`-th row's text — the frame hook re-stamps its
/// params to loop the line.
fn line_node(i: usize) -> String {
    format!("line:{i}")
}

/// `[active, start, duration, rate, frozen]` for a line that began at
/// `start` on the shader clock.
fn params(start: f32) -> [f32; 5] {
    [1.0, start, LINE_SECS, 1.0, 0.0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // State = the clock the current pass began at.
    App::new("opal-gfx — text effects", W, H, 0.0f32)
        .scene(|s: &mut Scene, start: &f32| {
            let start = *start;
            s.col("root")
                .w(Len::Fill)
                .h(Len::Fill)
                .color(BG)
                .gap(24.0)
                .pad_xy(28.0, 28.0)
                .align(Align::Center)
                .child(|c| {
                    for (i, e) in TEXT_EFFECTS.iter().enumerate() {
                        lyric_line(c, i, TextFxId(i as u32), e.name, start);
                    }
                });
        })
        // Loop the line: one param push per pass, nothing per frame —
        // the same shape the app's frame loop uses.
        .on_frame(|ctx, _tl, _now, start: &mut f32| {
            if ctx.effect_time - *start < LINE_SECS {
                return;
            }
            *start = ctx.effect_time;
            for i in 0..TEXT_EFFECTS.len() {
                if let Some(node) = ctx.node(&line_node(i)) {
                    ctx.tree.set_effect_data(node, params(*start).to_vec());
                }
            }
        })
        // `OPAL_AUTOCAPTURE=1` writes one frame to `debug_captures/` and
        // exits; without it the example runs interactively.
        .capture_from_env()
        .run()
}

/// One demo row: the effect's name, then a lyric running that effect.
fn lyric_line(c: &mut Scene, i: usize, id: TextFxId, label: &str, start: f32) {
    c.text((), label, 13.0).color([0.6, 0.6, 0.66, 1.0]);
    c.text(line_node(i), LINE, 40.0)
        .color(ACCENT)
        .text_fx(id)
        .effect_data(&params(start))
        .effect_animated();
}

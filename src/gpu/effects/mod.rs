//! Procedural **effects** — the shader-authored shapes.
//!
//! An effect is one fragment shader that paints a node's quad. Each lives
//! in its own `.wgsl` file next to this module, exposes a single entry
//! point with a fixed signature, and is listed once in [`EFFECTS`]. The
//! shape pipeline stitches those files into `shape.wgsl` at build time and
//! generates the dispatch, so adding an effect is *one file plus one line*
//! — no shared switch to edit and no way for an id to drift from its
//! branch.
//!
//! # Writing one
//!
//! ```wgsl
//! // my_effect.wgsl
//! fn fx_my_effect(c: EffectCtx) -> vec4<f32> {
//!     // c.rel  — 0..1 across the quad (x right, y down)
//!     // c.px   — physical px within the quad; c.size — its size
//!     // c.color — the node's colour; fx_param(c, i) — its i-th param
//!     // globals.time — seconds; fx_band(i) — audio energy 0..1
//!     return vec4<f32>(c.color.rgb, 1.0) * c.color.a;   // premultiplied
//! }
//! ```
//!
//! Return **premultiplied** rgba in linear space (use `srgb_to_linear` on
//! authored colours), and stay inside the quad — the caller has already
//! clipped, rounded and applied the node's opacity.
//!
//! # Inputs
//!
//! - **Params**: a per-node `&[f32]` of any length ([`crate::scene::Scene`]
//!   `effect_data` / `effect_data_bind`), uploaded into one shared arena
//!   and read back with `fx_param`. Unlike the old packed-vec4 scheme this
//!   leaves the node's border, shadow and corner radius free to be used
//!   normally.
//! - **Globals**: `globals.time` (seconds, engine-owned) plus the audio
//!   levels the host publishes each frame. A time-driven effect therefore
//!   costs *nothing* per frame on the CPU — no bind pumping, no re-flatten
//!   — as long as the node is marked
//!   [`animated`](crate::scene::NodeBuilderRef::effect_animated), which is
//!   what keeps the loop rendering while it's on screen.

/// The per-frame inputs every effect can read: the engine's clock and
/// whatever audio the host is publishing. Uploaded once per frame in the
/// frame uniform, so an effect that animates on `time` (or reacts to the
/// music) costs nothing per frame on the CPU.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EffectGlobals {
    /// Seconds since the app started, wrapped by the engine to keep
    /// float precision over a long session.
    pub time: f32,
    /// Broadband level, 0..1.
    pub audio_level: f32,
    /// Per-band energy (low → high), 0..1. All zero when no audio source
    /// is publishing; effects then run on `time` alone.
    pub audio_bands: [f32; BANDS],
}

/// Audio bands carried to the shaders. Matches `fx_band`'s bound.
pub const BANDS: usize = 8;

/// Which effect a [`crate::gpu::SHAPE_KIND_EFFECT`] node runs. Ids are
/// assigned by position in [`EFFECTS`] — the generated dispatch and these
/// constants come from the same table, so they cannot disagree.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EffectId(pub u32);

/// One registered effect: its entry-point name and the WGSL that defines
/// it. `name` is for labels and error messages.
pub struct EffectDef {
    pub name: &'static str,
    pub entry: &'static str,
    pub wgsl: &'static str,
}

/// Every effect the shape shader can run, in id order.
pub const EFFECTS: &[EffectDef] = &[
    EffectDef {
        name: "curve",
        entry: "fx_curve",
        wgsl: include_str!("../shaders/effects/curve.wgsl"),
    },
    EffectDef {
        name: "now_playing",
        entry: "fx_now_playing",
        wgsl: include_str!("../shaders/effects/now_playing.wgsl"),
    },
];

/// Filled, anti-aliased area under a Catmull-Rom curve through the param
/// points (y in 0..1 from the bottom) — the equaliser response graph.
pub const EFFECT_CURVE: EffectId = EffectId(0);
/// Flowing wave field for the playing track: layered sine bands tinted
/// from the node's colour, drifting on `globals.time` and swelling with
/// the audio bands. Params: `[phase_seed, intensity]`.
pub const EFFECT_NOW_PLAYING: EffectId = EffectId(1);

/// Markers in `shape.wgsl` replaced with the registered effect bodies and
/// the generated dispatch.
const SNIPPETS_MARKER: &str = "//__EFFECT_SNIPPETS__";
const DISPATCH_MARKER: &str = "//__EFFECT_DISPATCH__";

/// Stitch every registered effect into `shape.wgsl`: the bodies where the
/// snippet marker sits, and an `if` chain over [`EffectId`] where the
/// dispatch marker sits.
pub fn compose_shader(shape_wgsl: &str) -> String {
    let mut snippets = String::new();
    let mut dispatch = String::new();
    for (i, e) in EFFECTS.iter().enumerate() {
        snippets.push_str("// --- effect: ");
        snippets.push_str(e.name);
        snippets.push_str(" ---\n");
        snippets.push_str(e.wgsl);
        snippets.push('\n');
        dispatch.push_str(&format!(
            "    if (id == {i}u) {{ return {}(c); }}\n",
            e.entry
        ));
    }
    shape_wgsl
        .replace(SNIPPETS_MARKER, &snippets)
        .replace(DISPATCH_MARKER, &dispatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_defines_the_entry_point_it_registers() {
        for e in EFFECTS {
            assert!(
                e.wgsl.contains(&format!("fn {}(", e.entry)),
                "{} does not define {}",
                e.name,
                e.entry
            );
        }
    }

    #[test]
    fn ids_match_table_order() {
        assert_eq!(EFFECTS[EFFECT_CURVE.0 as usize].entry, "fx_curve");
        assert_eq!(
            EFFECTS[EFFECT_NOW_PLAYING.0 as usize].entry,
            "fx_now_playing"
        );
    }

    #[test]
    fn composition_fills_both_markers() {
        let out = compose_shader("a\n//__EFFECT_SNIPPETS__\nb\n//__EFFECT_DISPATCH__\nc\n");
        assert!(!out.contains(SNIPPETS_MARKER));
        assert!(!out.contains(DISPATCH_MARKER));
        for e in EFFECTS {
            assert!(out.contains(e.entry));
        }
        // Dispatch reaches every id.
        for i in 0..EFFECTS.len() {
            assert!(out.contains(&format!("if (id == {i}u)")));
        }
    }
}

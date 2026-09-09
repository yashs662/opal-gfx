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
//! # Text effects
//!
//! [`TEXT_EFFECTS`] is the second registry: one function per effect, run
//! for **every glyph** of a text node that asks for it (see
//! [`text_fx`](crate::scene::NodeBuilderRef::text_fx)) — in the vertex
//! stage (where the glyph goes and how big it is) and in the fragment
//! stage (what colour it takes). It reads the glyph's place in its run
//! plus the same params and globals as below, so a whole line can sweep,
//! bounce or shimmer with nothing pumped per frame.
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

/// The shared prelude every effect body can call — palettes, envelopes,
/// edge fades. Stitched in ahead of the bodies so an effect never carries
/// its own copy of maths another one already needs.
const PRELUDE: &str = include_str!("../shaders/effects/shared.wgsl");

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

/// Which **text effect** a node's glyphs run. Like [`EffectId`], ids come
/// from position in [`TEXT_EFFECTS`], so the constants below and the
/// generated dispatch cannot disagree.
///
/// A text effect is one function evaluated per glyph, in both the vertex
/// and the fragment stage, returning where that glyph sits, how big it is
/// and what colour it takes. It reads where the glyph falls in its run
/// (`along`/`index`/`count`), the node's params, and the frame globals —
/// so a whole line can sweep, bounce or shimmer with **nothing** driven
/// per frame from the CPU.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextFxId(pub u32);

/// Every text effect the shape shader can run, in id order.
pub const TEXT_EFFECTS: &[EffectDef] = &[EffectDef {
    name: "karaoke",
    entry: "tfx_karaoke",
    wgsl: include_str!("../shaders/text_effects/karaoke.wgsl"),
}];

/// Sung-line sweep: the words light left to right in time with the line,
/// with a bloom riding the leading edge.
/// Params: `[active, start, duration, rate, frozen]` — see the file.
pub const TEXT_FX_KARAOKE: TextFxId = TextFxId(0);

/// Markers in `shape.wgsl` replaced with the registered effect bodies and
/// the generated dispatch.
const SNIPPETS_MARKER: &str = "//__EFFECT_SNIPPETS__";
const DISPATCH_MARKER: &str = "//__EFFECT_DISPATCH__";
/// The same pair for the per-glyph text effects.
const TEXT_SNIPPETS_MARKER: &str = "//__TEXTFX_SNIPPETS__";
const TEXT_DISPATCH_MARKER: &str = "//__TEXTFX_DISPATCH__";

/// Stitch the shared [`PRELUDE`] and both registries into `shape.wgsl`:
/// the prelude + bodies where each snippet marker sits, and an `if` chain
/// over the id where each dispatch marker sits.
pub fn compose_shader(shape_wgsl: &str) -> String {
    let (snippets, dispatch) = stitch(EFFECTS, "effect", PRELUDE);
    let (text_snippets, text_dispatch) = stitch(TEXT_EFFECTS, "text effect", "");
    shape_wgsl
        .replace(SNIPPETS_MARKER, &snippets)
        .replace(DISPATCH_MARKER, &dispatch)
        .replace(TEXT_SNIPPETS_MARKER, &text_snippets)
        .replace(TEXT_DISPATCH_MARKER, &text_dispatch)
}

/// The bodies of one registry, plus the `if` chain that reaches each by
/// id. Both registries dispatch on a single `c` argument, so one routine
/// generates both.
fn stitch(defs: &[EffectDef], label: &str, prelude: &str) -> (String, String) {
    let mut snippets = String::from(prelude);
    if !prelude.is_empty() {
        snippets.push('\n');
    }
    let mut dispatch = String::new();
    for (i, e) in defs.iter().enumerate() {
        snippets.push_str(&format!("// --- {label}: {} ---\n", e.name));
        snippets.push_str(e.wgsl);
        snippets.push('\n');
        dispatch.push_str(&format!(
            "    if (id == {i}u) {{ return {}(c); }}\n",
            e.entry
        ));
    }
    (snippets, dispatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_defines_the_entry_point_it_registers() {
        for e in EFFECTS.iter().chain(TEXT_EFFECTS) {
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
        for (id, entry) in [
            (EFFECT_CURVE, "fx_curve"),
            (EFFECT_NOW_PLAYING, "fx_now_playing"),
        ] {
            assert_eq!(EFFECTS[id.0 as usize].entry, entry);
        }
        assert_eq!(
            TEXT_EFFECTS[TEXT_FX_KARAOKE.0 as usize].entry,
            "tfx_karaoke"
        );
    }

    /// The composed shader is only compiled when a GPU device is created,
    /// so a typo in an effect body would otherwise surface as a black
    /// window on someone else's machine. Parse + validate it here instead:
    /// the same front end wgpu runs, at test speed.
    #[test]
    fn the_composed_shader_is_valid_wgsl() {
        let src = compose_shader(include_str!("../shaders/shape.wgsl"));
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("{}", e.emit_to_string(&src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("{}", e.emit_to_string(&src));
        }
    }

    #[test]
    fn composition_fills_every_marker() {
        let out = compose_shader(
            "a\n//__EFFECT_SNIPPETS__\nb\n//__EFFECT_DISPATCH__\nc\n\
             //__TEXTFX_SNIPPETS__\nd\n//__TEXTFX_DISPATCH__\n",
        );
        for marker in [
            SNIPPETS_MARKER,
            DISPATCH_MARKER,
            TEXT_SNIPPETS_MARKER,
            TEXT_DISPATCH_MARKER,
        ] {
            assert!(!out.contains(marker), "{marker} left unfilled");
        }
        for e in EFFECTS.iter().chain(TEXT_EFFECTS) {
            assert!(out.contains(e.entry));
        }
        // Dispatch reaches every id in both registries.
        for i in 0..EFFECTS.len().max(TEXT_EFFECTS.len()) {
            assert!(out.contains(&format!("if (id == {i}u)")));
        }
    }
}

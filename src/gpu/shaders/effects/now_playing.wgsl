// Now-playing wave field: layered bands drifting across the quad, tinted
// around the node's colour, swelling with the audio.
//
// Meant to sit *behind* a row of content — wide and short — so the waves
// are shaped in the quad's own 0..1 space rather than an aspect-corrected
// one: a handful of crests across the width whatever the row's size, and
// a soft falloff at every edge so the text on top stays readable.
//
// Params: `[phase_seed, intensity, fade_start, fade_dir]`.
//   phase_seed — decorrelates instances so two rows never move in lockstep.
//   intensity  — 0..1 master amount.
//   fade_start — `globals.time` the envelope began at.
//   fade_dir   — +1 fades in from there, -1 fades out. The envelope runs
//                on the clock, so the row losing the field can fade out
//                without anything driving it from the CPU.

const FX_NP_LAYERS: i32 = 5;
/// Crests across the width. Low, because a row is a letterbox: more than
/// this and the bands alias into a flat smear.
const FX_NP_FREQ: f32 = 1.7;
/// Seconds the field takes to swell in (or sink out).
const FX_NP_FADE: f32 = 0.45;
/// Saturation boost applied to every band. The field paints at a low
/// alpha so the row's text stays first — pushing the colour away from its
/// own grey keeps the ribbons reading as *colour* at that alpha instead
/// of washing toward the surface underneath.
const FX_NP_SAT: f32 = 1.75;

/// How far the band colours wander from the node's own hue.
const FX_NP_SPREAD: f32 = 0.5;
/// Soft dissolve at every edge, so the field ends in the row rather than
/// on a rectangle.
const FX_NP_INSET: f32 = 0.18;

fn fx_now_playing(c: EffectCtx) -> vec4<f32> {
    let seed = fx_param(c, 0u);
    let intensity = select(1.0, fx_param(c, 1u), c.params_len > 1u);
    // Envelope: 0 → 1 from `fade_start`, or 1 → 0 when fading out.
    let env = fx_env(fx_param(c, 2u), fx_param(c, 3u), FX_NP_FADE);
    if (env <= 0.001) {
        return vec4<f32>(0.0);
    }
    // x across the row, y centred: -1 at the top edge, +1 at the bottom.
    let x = c.rel.x;
    let y = c.rel.y * 2.0 - 1.0;
    let t = globals.time + seed;
    // Low bands drive the swell, the broadband level the brightness —
    // silence (or no audio source) leaves a slow, steady drift.
    let bass = fx_bass();
    let energy = globals.audio_level;
    let base = srgb_to_linear(c.color.rgb);

    var acc = vec3<f32>(0.0);
    for (var i = 0; i < FX_NP_LAYERS; i = i + 1) {
        let layer = f32(i) / f32(FX_NP_LAYERS);
        // Each layer rides its own speed and swell, which is what makes
        // the bundle braid instead of sliding as one.
        let amp = (0.62 + 0.45 * bass) * (0.55 + 0.45 * env)
            * (0.55 + 0.45 * sin(t * 0.6 + layer * 5.0))
            * (1.0 - 0.30 * layer);
        let phase = t * (0.5 + 0.7 * (1.0 - layer)) + layer * 3.0;
        let wave = amp * sin(FX_NP_FREQ * 6.28318 * x - phase);
        // Soft, wide band: at this height a thin line reads as a scratch,
        // a fat gaussian reads as light.
        let thick = 0.19 + 0.10 * energy;
        let d = (y - wave) / thick;
        let tint = fx_palette(base, 0.5 * x + layer - 0.08 * t, FX_NP_SPREAD);
        acc = acc + fx_vivid(tint, FX_NP_SAT) * exp(-d * d);
    }
    acc = acc / 4.0;

    let fade = fx_edge_fade(c.rel, FX_NP_INSET);
    let a = clamp(dot(acc, vec3<f32>(0.5)), 0.0, 1.0) * fade * intensity * env * c.color.a;
    return fx_premul(acc, a);
}

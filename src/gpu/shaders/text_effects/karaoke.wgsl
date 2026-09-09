// Karaoke sweep: the sung line lights left to right in time with itself,
// with a bloom riding the leading edge — the words are *being sung*, not
// merely highlighted.
//
// The sweep is stamped, not pumped: the host writes when the line started
// and how long it lasts, and the shader clock does the rest. A pause
// writes `rate` 0 and the progress to hold at, so the head freezes mid-word
// instead of running on without the music.
//
// Params: `[active, start, duration, rate, frozen]`.
//   active   — 1 on the line at the playhead, 0 on every other (which
//              returns the glyph untouched, so an idle line costs nothing).
//   start    — `globals.time` the line began at.
//   duration — seconds until the next line.
//   rate     — 1 while playing, 0 while frozen.
//   frozen   — progress (0..1) to hold at while `rate` is 0.

/// Half-width of the lit/unlit crossover, in fractions of the line. Wide
/// enough that the edge lands *between* letters rather than slicing one,
/// narrow enough to still read as an edge.
const TFX_KARAOKE_EDGE: f32 = 0.035;
/// Width of the bloom that rides the head. Roughly a word at typical line
/// lengths — a narrower one strobes letter by letter.
const TFX_KARAOKE_HEAD: f32 = 0.07;
/// How much the head grows and lifts. The lift is a fraction of the
/// glyph's own height, so it reads the same at any type size.
const TFX_KARAOKE_GROW: f32 = 0.18;
const TFX_KARAOKE_RISE: f32 = 0.22;
/// Alpha of the words still to come. They stay readable — the sweep is
/// the emphasis, not a reveal.
const TFX_KARAOKE_AHEAD: f32 = 0.45;
/// How far the head washes toward white at its brightest.
const TFX_KARAOKE_FLARE: f32 = 0.55;
/// Peak tilt at the head, in radians (~3.5°).
const TFX_KARAOKE_TILT: f32 = 0.06;
/// How much the low end adds to the head's growth. Zero audio leaves the
/// sweep running on the clock alone.
const TFX_KARAOKE_BASS: f32 = 0.6;

fn tfx_karaoke(c: TextFxCtx) -> TextFxOut {
    if (tfx_param(c, 0u) < 0.5) {
        return tfx_identity(c);
    }
    let start = tfx_param(c, 1u);
    let dur = max(tfx_param(c, 2u), 0.05);
    let running = (globals.time - start) / dur;
    let progress = clamp(select(tfx_param(c, 4u), running, tfx_param(c, 3u) > 0.5), 0.0, 1.0);

    // Signed distance from the head to this glyph, along the line.
    let d = progress - c.along;
    let lit = smoothstep(-TFX_KARAOKE_EDGE, TFX_KARAOKE_EDGE, d);
    let h = d / TFX_KARAOKE_HEAD;
    let bloom = exp(-h * h);
    // The sweep starts and ends *off* the line, so the first and last
    // glyphs get the same treatment as the ones in the middle.
    let swell = bloom * (1.0 + TFX_KARAOKE_BASS * fx_bass());

    let rgb = mix(c.color.rgb, vec3<f32>(1.0), TFX_KARAOKE_FLARE * bloom);
    let alpha = c.color.a * mix(TFX_KARAOKE_AHEAD, 1.0, lit);
    // A touch of tilt on the head, alternating with the glyph index, so
    // the sweep looks like it is *knocking* letters rather than sliding a
    // mask across them. Tiny — this is reading text.
    let tilt = TFX_KARAOKE_TILT * swell * select(-1.0, 1.0, (c.index % 2.0) < 0.5);
    return TextFxOut(
        vec2<f32>(0.0, -TFX_KARAOKE_RISE * c.size.y * swell),
        1.0 + TFX_KARAOKE_GROW * swell,
        vec4<f32>(rgb, alpha),
        tilt,
    );
}

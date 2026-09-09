// Shared prelude for the procedural effects — stitched in ahead of every
// effect body, so an effect never has to carry its own copy of the
// palette, envelope or falloff maths. Anything more than one effect needs
// belongs here; anything only one effect needs stays in that effect's
// file.
//
// Everything here is pure: the only state is `globals` (clock + audio),
// which is what keeps a time-driven effect free on the CPU.

/// Falloff to 0 within `inset` (in 0..1 units) of every edge of the quad,
/// so a field dissolves into its container instead of ending on a
/// rectangle.
fn fx_edge_fade(rel: vec2<f32>, inset: f32) -> f32 {
    let x = smoothstep(0.0, inset, rel.x) * smoothstep(0.0, inset, 1.0 - rel.x);
    let y = smoothstep(0.0, inset, rel.y) * smoothstep(0.0, inset, 1.0 - rel.y);
    return x * y;
}

/// Push a colour away from its own grey by `sat` (1 = unchanged). A field
/// painted at low alpha washes toward the surface behind it; this is what
/// keeps it reading as colour rather than a lighter grey.
fn fx_vivid(c: vec3<f32>, sat: f32) -> vec3<f32> {
    let grey = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return max(mix(vec3<f32>(grey), c, sat), vec3<f32>(0.0));
}

/// Cosine palette (Inigo Quilez) around `base`: neighbours of the album
/// accent rather than a fixed rainbow, so an effect always belongs to the
/// artwork it sits under. `spread` (0..1) is how far it wanders.
fn fx_palette(base: vec3<f32>, t: f32, spread: f32) -> vec3<f32> {
    let tint = 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.0, 0.33, 0.67)));
    return mix(base, base * 0.4 + tint * 0.9, spread);
}

/// Smoothstepped 0 → 1 fade that began at `start` on the shader clock and
/// takes `dur` seconds, or 1 → 0 when `dir` is negative. The envelope runs
/// on the clock, so a node can fade *out* with nothing driving it from the
/// CPU — and a rebuild mid-fade resumes it instead of restarting it.
fn fx_env(start: f32, dir: f32, dur: f32) -> f32 {
    let raw = clamp((globals.time - start) / max(dur, 1e-3), 0.0, 1.0);
    let linear = select(raw, 1.0 - raw, dir < 0.0);
    return linear * linear * (3.0 - 2.0 * linear);
}

/// Low-end energy, 0..1 — the swell an effect rides. Zero when the host
/// publishes no audio, which leaves the effect on its clock alone.
fn fx_bass() -> f32 {
    return fx_band(0u) * 0.6 + fx_band(1u) * 0.4;
}

/// Premultiply for the blend the shape pipeline expects.
fn fx_premul(rgb: vec3<f32>, a: f32) -> vec4<f32> {
    let c = clamp(a, 0.0, 1.0);
    return vec4<f32>(rgb * c, c);
}

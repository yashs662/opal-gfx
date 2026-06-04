// Composite pass (compositor P2 + P3 scroll window).
//
// Blits one cached layer texture onto the surface. The quad covers the
// screen rect `dst_origin .. dst_origin + dst_size`; the texture is
// sampled over the window `src_origin .. src_origin + src_extent`
// (in `tex_size` px); fragments outside `clip_rect` are discarded.
// Layers draw back-to-front so the premultiplied-over blend
// (One, OneMinusSrcAlpha) stacks them correctly.
//
// Full-surface identity — `dst_origin = offset`, `dst_size =
// surface*scale`, `src_origin = 0`, `src_extent = tex_size = surface`,
// `clip = NO_CLIP` — gives `uv = corner` with no discard: a byte-exact
// passthrough of a surface-sized layer texture (root-layer parity) and
// the P3 offset/scale/opacity composite-move. A scroll layer instead
// places the quad at the container rect and samples a 1:1-px window of a
// tall texture at the scroll offset.
//
// The layer texture is the surface format (sRGB): the hardware decodes
// on sample and re-encodes on write, so an identity passthrough
// round-trips losslessly. Premultiplied throughout.

struct Composite {
    // Screen-space top-left of the quad (physical px).
    dst_origin: vec2<f32>,
    // Screen-space size of the quad (physical px).
    dst_size: vec2<f32>,
    // Texture sample window origin (physical px) — the scroll offset.
    src_origin: vec2<f32>,
    // Texture sample window extent (physical px).
    src_extent: vec2<f32>,
    // Layer texture dimensions (physical px).
    tex_size: vec2<f32>,
    // Physical surface size, for the px → NDC map.
    surface_size: vec2<f32>,
    // Screen-space scissor (min_x, min_y, max_x, max_y). NO_CLIP = ±1e30.
    clip_rect: vec4<f32>,
    // Composite-time opacity multiplier (premultiplied: scales rgb+a).
    opacity: f32,
    // Rounds the clip rect's corners (physical px). 0 = plain box scissor.
    corner_radius: f32,
    // Falloff exponent for the edge fade (1 = linear).
    edge_fade_falloff: f32,
    _p2: f32,
    // Per-edge fade fractions [top, right, bottom, left] (0..1 of the dst
    // rect extent). Fade alpha to 0 near each edge. All-0 = no fade.
    edge_fade: vec4<f32>,
}

@group(0) @binding(0) var<uniform> c: Composite;
@group(0) @binding(1) var layer_tex: texture_2d<f32>;
@group(0) @binding(2) var layer_samp: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) scr: vec2<f32>,
}

// Unit quad as two triangles, corners in [0, 1].
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    let corner = CORNERS[vi];
    let screen_px = c.dst_origin + corner * c.dst_size;
    let ndc = vec2<f32>(
        screen_px.x / c.surface_size.x * 2.0 - 1.0,
        1.0 - screen_px.y / c.surface_size.y * 2.0,
    );
    var out: VSOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = (c.src_origin + corner * c.src_extent) / c.tex_size;
    out.scr = screen_px;
    return out;
}

// Per-edge alpha ramp: 1 away from the edge, ramping to 0 at it over
// `width` (fraction of the rect). `d` is the fragment's distance from the
// edge as a fraction (0 at the edge, 1 at the far side).
fn edge_ramp(d: f32, width: f32, falloff: f32) -> f32 {
    if width <= 0.0 {
        return 1.0;
    }
    return pow(clamp(d / width, 0.0, 1.0), falloff);
}

// Combined edge-fade alpha for a fragment, from all four edges.
fn edge_fade_alpha(scr: vec2<f32>) -> f32 {
    let f = c.edge_fade;
    if f.x <= 0.0 && f.y <= 0.0 && f.z <= 0.0 && f.w <= 0.0 {
        return 1.0;
    }
    // Position within the dst rect, 0..1 per axis.
    let p = (scr - c.dst_origin) / max(c.dst_size, vec2<f32>(1.0));
    let fall = max(c.edge_fade_falloff, 1e-4);
    let top = edge_ramp(p.y, f.x, fall);
    let right = edge_ramp(1.0 - p.x, f.y, fall);
    let bottom = edge_ramp(1.0 - p.y, f.z, fall);
    let left = edge_ramp(p.x, f.w, fall);
    return top * right * bottom * left;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let lo = c.clip_rect.xy;
    let hi = c.clip_rect.zw;
    let fade = edge_fade_alpha(in.scr);
    // Rounded-rect clip (external video layers honouring a rounded
    // container). The SDF subsumes the box scissor, with a 1px feather.
    if c.corner_radius > 0.0 {
        let center = (lo + hi) * 0.5;
        let half = (hi - lo) * 0.5;
        let p = in.scr - center;
        let q = abs(p) - (half - vec2<f32>(c.corner_radius));
        let dist = length(max(q, vec2<f32>(0.0)))
            + min(max(q.x, q.y), 0.0) - c.corner_radius;
        if dist > 0.5 {
            discard;
        }
        let aa = clamp(0.5 - dist, 0.0, 1.0);
        let s = textureSample(layer_tex, layer_samp, in.uv);
        return s * c.opacity * aa * fade;
    }
    // Screen-space box scissor (NO_CLIP sentinel never triggers).
    if in.scr.x < lo.x || in.scr.y < lo.y || in.scr.x > hi.x || in.scr.y > hi.y {
        discard;
    }
    // Premultiplied sample; opacity scales the whole premultiplied
    // tuple so it stays premultiplied for the over-blend.
    let s = textureSample(layer_tex, layer_samp, in.uv);
    return s * c.opacity * fade;
}

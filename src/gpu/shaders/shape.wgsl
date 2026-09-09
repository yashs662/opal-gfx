// SDF pipeline: rounded rectangles with per-corner radii, borders, drop
// shadows, plus frosted-glass shapes that sample a backdrop mipmap
// pyramid. Outputs PREMULTIPLIED alpha — pair with blend state One /
// OneMinusSrcAlpha and a PreMultiplied surface composite mode.
//
// Two fragment entry points:
//   - `fs_opaque`: no @group(1) usage. Used by the backdrop pass (which
//     writes mip 0 of the backdrop texture and must not sample the
//     texture it is writing).
//   - `fs_main`: includes the glass branch that samples `backdrop_tex`
//     via dual-filter upsample at a per-glass LOD. Used by the final
//     surface pass.

const SHAPE_KIND_RECT: u32  = 0u;
const SHAPE_KIND_GLASS: u32 = 1u;
const SHAPE_KIND_GLYPH: u32 = 2u;
const SHAPE_KIND_IMAGE: u32 = 3u;
// Procedural effect: the quad is painted by one of the registered effect
// shaders (`gpu::effects`), selected by `effect_id` and fed params from
// the shared arena. Bodies + dispatch are stitched in at pipeline build.
const SHAPE_KIND_EFFECT: u32 = 4u;
const SHAPE_KIND_MASK: u32  = 0xFFu;
// Border-side bits stored in `shape_kind >> 8`. 0b1111 = all (default).
const BORDER_SIDE_TOP: u32    = 0x1u;
const BORDER_SIDE_RIGHT: u32  = 0x2u;
const BORDER_SIDE_BOTTOM: u32 = 0x4u;
const BORDER_SIDE_LEFT: u32   = 0x8u;
const BORDER_SIDES_ALL: u32   = 0xFu;

struct ShapeInstance {
    color: vec4<f32>,
    border_color: vec4<f32>,
    shadow_color: vec4<f32>,
    border_radius: vec4<f32>,   // TL, TR, BL, BR (pixels)
    backdrop_uv_rect: vec4<f32>,
    /// Scissor rect in physical px: (min_x, min_y, max_x, max_y).
    /// Fragment is discarded if outside. Sentinel = ±1e30.
    clip_rect: vec4<f32>,
    /// Per-corner radii [TL, TR, BL, BR] (px) of the clip rect — rounds
    /// the scissor (rounded overflow clipping). All-0 = square clip.
    clip_radius: vec4<f32>,
    position: vec2<f32>,        // top-left in pixels
    size: vec2<f32>,            // pixels
    shadow_offset: vec2<f32>,
    shape_kind: u32,
    /// Glass-only frosted-texture variation. Per-fragment hash is
    /// scaled by this and offsets the backdrop sample UV by that many
    /// physical px at the chosen mip. 0 = mirror; 1 = subtle frost;
    /// 3+ = pebbled glass. Ignored for non-glass kinds.
    roughness: f32,
    border_width: f32,
    shadow_blur: f32,
    shadow_opacity: f32,
    opacity: f32,
    /// Per-shape visual scale around the rect centre. `(1.0, 1.0)` is
    /// identity. Layout + hit-test see the pre-scale geometry (this is
    /// purely a vertex/fragment-side transform), so hover-grow effects
    /// don't shift click boxes.
    scale: vec2<f32>,
    effect_id: u32,
    /// SHAPE_KIND_EFFECT only: where this instance's params live in the
    /// shared `effect_params` arena, and how many it has.
    effect_param_offset: u32,
    effect_param_len: u32,
    /// SHAPE_KIND_GLYPH + a text effect only: where this glyph sits in
    /// its run — `glyph_along` is its centre as a fraction of the run's
    /// advance width. Three scalars rather than a `vec3` on purpose: a
    /// `vec3` aligns to 16 and would pad the struct past its stride.
    glyph_along: f32,
    glyph_index: f32,
    glyph_count: f32,
}

// Signed distance to a rounded rect with min `lo`, max `hi`, per-corner
// radii `r` = [TL, TR, BL, BR]. <0 inside, >0 outside. Used for rounded
// scissor clipping; delegates to the shared per-corner SDF (which also
// clamps each radius to the half extents).
fn rounded_clip_sd(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>, r: vec4<f32>) -> f32 {
    let half = (hi - lo) * 0.5;
    return sd_rounded_rect(p - (lo + half), half, r);
}

struct Frame {
    screen_size: vec2<f32>,
    /// Maximum usable LOD for the backdrop mip pyramid. The glass
    /// branch clamps `log2(blur_px)` to this so very large authored
    /// radii don't punch through the bottom of the pyramid (a 1×1
    /// mip is just an average of the whole frame).
    max_backdrop_lod: f32,
    /// Window-corner clip radius in physical px. `0` disables the
    /// clip; otherwise the final-pass fragment shader masks every
    /// fragment by a rounded-rect SDF covering the whole surface so
    /// the corner pixels stay transparent.
    window_corner_radius: f32,
    /// Seconds since the app started, wrapped to keep float precision.
    /// The only clock effects need — a time-driven effect costs nothing
    /// per frame on the CPU.
    time: f32,
    /// Broadband level of the audio the host is publishing, 0..1.
    audio_level: f32,
    _pad0: f32,
    _pad1: f32,
    /// Per-band audio energy, 0..1, low to high. Read with `fx_band`;
    /// packed as vec4s because uniform arrays of scalars stride to 16.
    audio_bands: array<vec4<f32>, 2>,
}

@group(0) @binding(0) var<uniform> globals: Frame;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;
// Flat arena of every effect node's params, indexed per instance by
// `effect_param_offset`/`effect_param_len`.
@group(0) @binding(2) var<storage, read> effect_params: array<f32>;

// Final-pass-only bindings. The opaque-pass pipeline layout omits group 1
// so only fs_main references these.
//
// `backdrop_tex` is the opaque-pass output (rgba8unorm, linear premul)
// with a full mip pyramid filled by `cs_downsample` (dual-filter down).
// The glass branch picks a per-instance fractional LOD = log2(blur_px)
// and gathers via dual-filter upsample for smooth Gaussian-like blur
// at any radius.
@group(1) @binding(0) var backdrop_tex: texture_2d<f32>;
@group(1) @binding(1) var backdrop_samp: sampler;

// Glyph atlas (R8Unorm). Final-pass only — opaque pass discards glyph
// instances, so its layout omits group 2.
@group(2) @binding(0) var glyph_tex: texture_2d<f32>;
@group(2) @binding(1) var glyph_samp: sampler;

// Image atlas (Rgba8UnormSrgb — texture sampler returns linear). Final
// pass only; opaque discards. Authored colors round-trip through the
// sRGB texture decode automatically, so no manual srgb_to_linear here.
@group(3) @binding(0) var image_tex: texture_2d<f32>;
@group(3) @binding(1) var image_samp: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) inst_idx: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VSOut {
    let inst = instances[ii];

    // Expand quad to cover dropped shadow so fragment shader sees enough area.
    let shadow_pad = vec2<f32>(
        abs(inst.shadow_offset.x) + max(inst.shadow_blur, 0.0) * 2.0,
        abs(inst.shadow_offset.y) + max(inst.shadow_blur, 0.0) * 2.0,
    );
    // Per-shape scale is applied around the rect centre. Use max(1.0)
    // as the cover scale so scales < 1.0 still get the original quad
    // area (saves nothing to shrink coverage; the FS clips via SDF).
    var center_px = inst.position + inst.size * 0.5;
    var quad_size = inst.size;
    // A glyph running a text effect draws at the rect that effect puts it
    // at — the fragment stage recomputes the same rect, so the quad and
    // the atlas mapping cannot drift apart.
    if (has_text_fx(inst)) {
        let fx = text_fx_eval(inst.effect_id, text_fx_ctx(inst));
        let rect = text_fx_rect(inst, fx);
        // Cover the rotated glyph's bounds, not just its own size.
        quad_size = 2.0 * text_fx_bounds(rect.zw, fx.rotation);
        center_px = rect.xy + rect.zw * 0.5;
    }
    let cover_scale = max(inst.scale, vec2<f32>(1.0));
    let half_scaled = quad_size * 0.5 * cover_scale + shadow_pad;
    let min_px = center_px - half_scaled;
    let max_px = center_px + half_scaled;

    // Two triangles, six verts. vertex_index -> unit-square corner.
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let world_pos = mix(min_px, max_px, corner);

    // Pixel -> clip-space. Y flipped so (0,0) is top-left.
    let ndc = vec2<f32>(
        world_pos.x / globals.screen_size.x * 2.0 - 1.0,
        1.0 - world_pos.y / globals.screen_size.y * 2.0,
    );

    var out: VSOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.world_pos = world_pos;
    out.inst_idx = ii;
    return out;
}

// Decode an sRGB-authored color component to linear. ShapeInstance colors
// are authored in perceptual sRGB space (the numbers a designer types), so
// the shader converts once at fragment entry and does all math/blending in
// linear space. The swapchain is Bgra8UnormSrgb which auto-encodes the
// linear output back to sRGB bytes on store.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(lo, hi, c > cutoff);
}

// Inverse of srgb_to_linear. Used by the glass branch so the tint
// alpha-blend happens in perceptual (sRGB) space — author writes
// rgba(1,1,1,0.08) expecting an 8%-perceived white tint, not an
// 8%-linear-light tint (which would gamma-amplify to ~32% grey).
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(lo, hi, c > cutoff);
}

// Cheap hash → 2D value in [0, 1). Used for the glass roughness
// scatter so each fragment picks a slightly different backdrop tap
// without paying for true blue-noise. Reference: Hugo Elias / Dave_H
// fract-mul hash, vec2 variant.
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract(vec2<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

fn sd_rectangle(p: vec2<f32>, xy: vec2<f32>) -> f32 {
    let d = abs(p) - max(xy, vec2<f32>(0.0));
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sd_rounded_rect(p: vec2<f32>, xy: vec2<f32>, r: vec4<f32>) -> f32 {
    // `r` is [TL, TR, BL, BR] (matching the `.radii(tl,tr,bl,br)` API).
    // Fragment coords are y-DOWN (origin top-left), so `p.y < 0` is the TOP
    // half and `p.y > 0` the BOTTOM half — select the bottom cases for
    // `p.y > 0`. (A previous `p.y < 0` here swapped top/bottom radii, which
    // only shows up with per-corner radii — uniform corners are symmetric.)
    let qx = select(0u, 1u, p.x > 0.0);
    let qy = select(0u, 2u, p.y > 0.0);
    let idx = qx + qy;
    var s: f32;
    switch idx {
        case 0u: { s = r.x; }      // top-left
        case 1u: { s = r.y; }      // top-right
        case 2u: { s = r.z; }      // bottom-left
        default: { s = r.w; }      // bottom-right
    }
    s = min(s, min(xy.x, xy.y));
    return sd_rectangle(p, xy - vec2<f32>(s)) - s;
}

// Fill coverage for a rounded rect that keeps STRAIGHT edges fully covered
// (hard, up to the geometric edge) and anti-aliases only the rounded CORNERS.
// A translucent panel/glass whose edge lands mid-pixel otherwise feathers its
// axis-aligned edge to ~50%, and over content/backdrop that reads as a 1px
// brighter seam (e.g. a sticky header's top edge). Axis-aligned edges don't
// need AA, so cutting them hard removes the seam while corners still round.
// `half` = half-size, `r` = [TL,TR,BL,BR] (as `sd_rounded_rect`), `aa` = AA
// half-width.
fn rounded_coverage(p: vec2<f32>, half: vec2<f32>, r: vec4<f32>, aa: f32) -> f32 {
    let ap = abs(p);
    if (ap.x > half.x || ap.y > half.y) {
        return 0.0;
    }
    let qx = select(0u, 1u, p.x > 0.0);
    let qy = select(0u, 2u, p.y > 0.0);
    let idx = qx + qy;
    var s: f32;
    switch idx {
        case 0u: { s = r.x; }
        case 1u: { s = r.y; }
        case 2u: { s = r.z; }
        default: { s = r.w; }
    }
    s = min(s, min(half.x, half.y));
    let corner = half - vec2<f32>(s);
    if (s > 0.0 && ap.x > corner.x && ap.y > corner.y) {
        let d = length(ap - corner) - s;
        return smoothstep(aa, -aa, d);
    }
    return 1.0;
}

struct Body { rgb: vec3<f32>, a: f32 };

// Rect / border fill — shared between both fragment entry points.
// Color channels are sRGB-decoded to linear before any blending math.
//
// Per-side border (e.g. Spotify bottom-tab) is detected via the upper-
// bit mask in `shape_kind`. When the mask is anything other than
// "all sides" the shader takes an asymmetric inner-AABB path that
// **forces square corners** — `border_radius` is ignored. Use the
// default `BorderSides::ALL` for radius-aware borders.
fn compute_fill(inst: ShapeInstance, p: vec2<f32>, outer_half: vec2<f32>,
                comp_alpha: f32, aa: f32) -> Body {
    var body: Body;
    let color_lin = srgb_to_linear(inst.color.rgb);
    if (inst.border_width > 0.0) {
        let border_lin = srgb_to_linear(inst.border_color.rgb);
        let sides = (inst.shape_kind >> 8u) & 0xFu;
        // sides == 0 means an older path without the mask set — treat
        // as ALL so existing call sites keep working.
        let mask = select(sides, BORDER_SIDES_ALL, sides == 0u);
        // Scale border_width alongside outer_half so a scaled-up shape
        // has a proportionally scaled border (and not a 1px sliver lost
        // inside a 1.5×-scaled outer). Uniform scale assumed for the
        // border thickness — non-uniform `scale_xy` axis-distorts the
        // border in width vs height, which is the documented behavior.
        let bw = inst.border_width * inst.scale.x;
        let radii_s = inst.border_radius * inst.scale.x;

        var inner_dist: f32;
        if (mask == BORDER_SIDES_ALL) {
            let inner_half = max(outer_half - vec2<f32>(bw), vec2<f32>(0.0));
            let inner_radii = max(radii_s - vec4<f32>(bw), vec4<f32>(0.0));
            inner_dist = sd_rounded_rect(p, inner_half, inner_radii);
        } else {
            // Asymmetric inner AABB, per-side mask. p is in centered
            // coords (origin at rect center). A masked-off side leaves
            // the inner edge at the outer edge — i.e. no border there.
            let inset_t = select(0.0, bw, (mask & BORDER_SIDE_TOP) != 0u);
            let inset_r = select(0.0, bw, (mask & BORDER_SIDE_RIGHT) != 0u);
            let inset_b = select(0.0, bw, (mask & BORDER_SIDE_BOTTOM) != 0u);
            let inset_l = select(0.0, bw, (mask & BORDER_SIDE_LEFT) != 0u);
            let inner_min = vec2<f32>(-outer_half.x + inset_l, -outer_half.y + inset_t);
            let inner_max = vec2<f32>( outer_half.x - inset_r,  outer_half.y - inset_b);
            let inner_center = (inner_min + inner_max) * 0.5;
            let inner_half = max((inner_max - inner_min) * 0.5, vec2<f32>(0.0));
            inner_dist = sd_rectangle(p - inner_center, inner_half);
        }
        let inner_alpha = smoothstep(aa, -aa, inner_dist);
        let border_alpha = clamp(comp_alpha - inner_alpha, 0.0, 1.0);
        let fill_a = inst.color.a * inner_alpha;
        let fill_rgb = color_lin * fill_a;
        let bord_a = inst.border_color.a * border_alpha;
        let bord_rgb = border_lin * bord_a;
        body.a = bord_a + fill_a * (1.0 - bord_a);
        body.rgb = bord_rgb + fill_rgb * (1.0 - bord_a);
    } else {
        body.a = inst.color.a * comp_alpha;
        body.rgb = color_lin * body.a;
    }
    return body;
}

// Shadow contribution — shared between both entry points.
fn compute_shadow(inst: ShapeInstance, px: vec2<f32>, center: vec2<f32>,
                  outer_half: vec2<f32>) -> Body {
    var s: Body;
    s.rgb = vec3<f32>(0.0);
    s.a = 0.0;
    if (inst.shadow_opacity > 0.0) {
        let radii_s = inst.border_radius * inst.scale.x;
        let sd = sd_rounded_rect(px - (center + inst.shadow_offset),
                                 outer_half, radii_s);
        let s_edge = max(inst.shadow_blur, 1.0);
        let s_i = smoothstep(s_edge, -s_edge, sd);
        s.a = inst.shadow_color.a * s_i * inst.shadow_opacity;
        s.rgb = srgb_to_linear(inst.shadow_color.rgb) * s.a;
    }
    return s;
}

// --- Procedural effects (SHAPE_KIND_EFFECT) -----------------------------
// One fragment shader per effect, each in its own file under
// `shaders/effects/` and registered in `gpu::effects::EFFECTS`. The bodies
// and the dispatch below are stitched in at pipeline build; see that
// module for the authoring contract.

/// Everything an effect is handed. Built once per fragment.
struct EffectCtx {
    /// Position within the quad: 0..1, x right, y down.
    rel: vec2<f32>,
    /// Position within the quad in physical px, and the quad's size.
    px: vec2<f32>,
    size: vec2<f32>,
    /// The node's authored colour (sRGB; `srgb_to_linear` it to shade).
    color: vec4<f32>,
    /// This effect's slice of the shared param arena.
    params_base: u32,
    params_len: u32,
}

/// The effect's `i`-th param, or 0 past the end.
fn fx_param(c: EffectCtx, i: u32) -> f32 {
    if (i >= c.params_len) { return 0.0; }
    return effect_params[c.params_base + i];
}

/// Audio energy in band `i` (0 = lowest), 0..1. Zero when the host
/// publishes no audio, so effects degrade to time-only motion.
fn fx_band(i: u32) -> f32 {
    if (i >= 8u) { return 0.0; }
    return globals.audio_bands[i / 4u][i % 4u];
}

//__EFFECT_SNIPPETS__

// --- Per-glyph text effects (SHAPE_KIND_GLYPH) --------------------------
//
// One function per effect, evaluated for every glyph in BOTH stages: the
// vertex stage places the (possibly moved and scaled) quad, the fragment
// stage maps that same rect back into the atlas and shades it. Same
// inputs, same result, so the stages agree without passing anything
// between them — and a whole line can sweep, bounce or shimmer with
// nothing driven per frame from the CPU.

struct TextFxCtx {
    /// This glyph's centre along its run, 0..1 — the axis a sweep runs on.
    along: f32,
    /// Its index in the run, and how many glyphs the run has: the handle
    /// for staggering one glyph against the next.
    index: f32,
    count: f32,
    /// The glyph quad's size in physical px.
    size: vec2<f32>,
    /// The text node's authored colour (sRGB) — what an effect modulates.
    color: vec4<f32>,
    params_base: u32,
    params_len: u32,
}

/// What an effect does to one glyph: move it, resize and rotate it about
/// its own centre, and give it a colour. A zero offset and rotation with
/// `scale` 1 leaves the glyph exactly where the layout put it.
struct TextFxOut {
    offset: vec2<f32>,
    scale: f32,
    color: vec4<f32>,
    /// Radians, clockwise on screen (y grows downward). Small angles read
    /// as type being *pushed* by something; large ones as tumbling.
    rotation: f32,
}

/// The effect's `i`-th param, or 0 past the end.
fn tfx_param(c: TextFxCtx, i: u32) -> f32 {
    if (i >= c.params_len) { return 0.0; }
    return effect_params[c.params_base + i];
}

/// The untouched glyph — what an effect returns when it has nothing to
/// say about this one (an inactive line, a finished animation).
fn tfx_identity(c: TextFxCtx) -> TextFxOut {
    return TextFxOut(vec2<f32>(0.0), 1.0, c.color, 0.0);
}

/// An effect's result with no rotation — the common case, and what keeps
/// an effect that only moves and tints from having to name the field.
fn tfx_out(offset: vec2<f32>, scale: f32, color: vec4<f32>) -> TextFxOut {
    return TextFxOut(offset, scale, color, 0.0);
}

//__TEXTFX_SNIPPETS__

fn text_fx_eval(id: u32, c: TextFxCtx) -> TextFxOut {
//__TEXTFX_DISPATCH__
    return tfx_identity(c);
}

/// Whether this instance is a glyph carrying a text effect.
fn has_text_fx(inst: ShapeInstance) -> bool {
    return (inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLYPH
        && inst.effect_param_len > 0u;
}

fn text_fx_ctx(inst: ShapeInstance) -> TextFxCtx {
    return TextFxCtx(
        inst.glyph_along,
        inst.glyph_index,
        inst.glyph_count,
        inst.size,
        inst.color,
        inst.effect_param_offset,
        inst.effect_param_len,
    );
}

/// The glyph's rect after its effect has moved and resized it, as
/// `(position, size)`. Rotation is *not* folded in here: it happens about
/// the rect's own centre, so both stages apply it separately — the vertex
/// stage by growing the quad to the rotated bounds, the fragment stage by
/// turning the sample point back.
fn text_fx_rect(inst: ShapeInstance, fx: TextFxOut) -> vec4<f32> {
    let size = inst.size * max(fx.scale, 0.0);
    let center = inst.position + inst.size * 0.5 + fx.offset;
    return vec4<f32>(center - size * 0.5, size);
}

/// Half-extent of the axis-aligned box a rotated `size` needs. The quad
/// has to cover it or a tilted glyph would be clipped by its own corners.
fn text_fx_bounds(size: vec2<f32>, rotation: f32) -> vec2<f32> {
    let c = abs(cos(rotation));
    let s = abs(sin(rotation));
    return 0.5 * vec2<f32>(size.x * c + size.y * s, size.x * s + size.y * c);
}

/// Turn a point back into the glyph's own (unrotated) frame.
fn text_fx_unrotate(p: vec2<f32>, rotation: f32) -> vec2<f32> {
    let c = cos(rotation);
    let s = sin(rotation);
    return vec2<f32>(p.x * c + p.y * s, -p.x * s + p.y * c);
}

/// One glyph, shaded: sample the R8 atlas over the glyph's (possibly
/// effect-moved) rect and tint it. Shared by both passes so text can
/// enter the glass backdrop looking exactly as it does in front of it.
fn glyph_frag(inst: ShapeInstance, px: vec2<f32>) -> vec4<f32> {
    let fx = text_fx_eval(inst.effect_id, text_fx_ctx(inst));
    let fxd = has_text_fx(inst);
    // `backdrop_uv_rect` = (u0, v0, w, h) into the R8 glyph atlas.
    let rect = select(vec4<f32>(inst.position, inst.size), text_fx_rect(inst, fx), fxd);
    let color = select(inst.color, fx.color, fxd);
    let size = max(rect.zw, vec2<f32>(1.0));
    // Sample in the glyph's own frame: turn the point back about the
    // rect's centre, then map it to the atlas as usual.
    let centered = text_fx_unrotate(px - (rect.xy + size * 0.5), select(0.0, fx.rotation, fxd));
    let rel = (centered + size * 0.5) / size;
    // Fragments outside [0,1] (from any quad oversize) drop out.
    if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
        return vec4<f32>(-1.0);
    }
    let uv = inst.backdrop_uv_rect.xy + rel * inst.backdrop_uv_rect.zw;
    let cov = textureSampleLevel(glyph_tex, glyph_samp, uv, 0.0).r;
    let rgb_lin = srgb_to_linear(color.rgb);
    let a = color.a * cov * inst.opacity;
    return vec4<f32>(rgb_lin * a, a);
}


fn effect_frag(inst: ShapeInstance, px: vec2<f32>) -> vec4<f32> {
    let size = max(inst.size, vec2<f32>(1.0));
    let rel = (px - inst.position) / size;
    if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
        return vec4<f32>(0.0);
    }
    let c = EffectCtx(
        rel,
        px - inst.position,
        size,
        inst.color,
        inst.effect_param_offset,
        inst.effect_param_len,
    );
    let id = inst.effect_id;
//__EFFECT_DISPATCH__
    return vec4<f32>(0.0);
}

// Fragment entry for the backdrop (opaque) pass. Group 1 (the
// blurred backdrop sampler) is omitted — glass would feedback-loop on
// the texture being written. Glyph + image atlases (groups 2 and 3)
// ARE bound so text and images can enter the backdrop and appear
// blurred behind glass panels in painter's order.
@fragment
fn fs_opaque(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLASS) {
        // Glass samples the backdrop it would otherwise write into —
        // skip it here so the blur input contains everything *behind*
        // each glass panel without the glass itself contaminating it.
        discard;
    }
    let px = in.world_pos;
    if (any(inst.clip_radius > vec4<f32>(0.0))) {
        if (rounded_clip_sd(px, inst.clip_rect.xy, inst.clip_rect.zw, inst.clip_radius) > 0.5) {
            discard;
        }
    } else if (px.x < inst.clip_rect.x || px.y < inst.clip_rect.y ||
        px.x > inst.clip_rect.z || px.y > inst.clip_rect.w) {
        discard;
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLYPH) {
        let glyph = glyph_frag(inst, px);
        if (glyph.a < 0.0) {
            discard;
        }
        return glyph;
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_IMAGE) {
        let size = max(inst.size * inst.scale, vec2<f32>(1.0));
        let center_img = inst.position + inst.size * 0.5;
        let rel = (px - (center_img - size * 0.5)) / size;
        if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
            discard;
        }
        let uv = inst.backdrop_uv_rect.xy + rel * inst.backdrop_uv_rect.zw;
        let sample = textureSampleLevel(image_tex, image_samp, uv, 0.0);
        let p = px - center_img;
        let outer_half = inst.size * 0.5 * inst.scale;
        let radii_s = inst.border_radius * inst.scale.x;
        let comp_dist = sd_rounded_rect(p, outer_half, radii_s);
        let aa = max(fwidth(comp_dist), 0.5);
        // Hard straight edges + rounded corners — no 1px lighter border where
        // an album cover's edge would otherwise feather over the placeholder.
        let comp_alpha = rounded_coverage(p, outer_half, radii_s, aa);
        let tint_lin = srgb_to_linear(inst.color.rgb);
        let a = sample.a * inst.color.a * inst.opacity * comp_alpha;
        let rgb = sample.rgb * tint_lin * a;
        return vec4<f32>(rgb, a);
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_EFFECT) {
        return effect_frag(inst, px) * inst.opacity;
    }

    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5 * inst.scale;
    let radii_scaled = inst.border_radius * inst.scale.x;
    let comp_dist = sd_rounded_rect(p, outer_half, radii_scaled);
    let aa = max(fwidth(comp_dist), 0.5);
    // Hard straight edges + rounded corners (no 1px edge seam over backdrop).
    let comp_alpha = rounded_coverage(p, outer_half, radii_scaled, aa);

    let body = compute_fill(inst, p, outer_half, comp_alpha, aa);
    let shadow = compute_shadow(inst, px, center, outer_half);

    let out_a = body.a + shadow.a * (1.0 - body.a);
    let out_rgb = body.rgb + shadow.rgb * (1.0 - body.a);
    return vec4<f32>(out_rgb, out_a) * inst.opacity;
}

// Fragment entry for the final surface pass. Handles glass by sampling
// the pre-blurred backdrop texture.
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;
    // Scissor clip — rounded (overflow container with a radius) or square.
    var clip_cov = 1.0;
    if (any(inst.clip_radius > vec4<f32>(0.0))) {
        let sd = rounded_clip_sd(px, inst.clip_rect.xy, inst.clip_rect.zw, inst.clip_radius);
        if (sd > 0.5) {
            discard;
        }
        clip_cov = clamp(0.5 - sd, 0.0, 1.0); // 1px feather
    } else if (px.x < inst.clip_rect.x || px.y < inst.clip_rect.y ||
        px.x > inst.clip_rect.z || px.y > inst.clip_rect.w) {
        discard;
    }
    // The window corner is rounded *once* on the composited frame by the
    // final `window_round` pass (see `shaders/window_round.wgsl`), not per
    // instance here — so stacked translucent layers can't leak the back one
    // through a per-layer corner. Each branch still multiplies by the rounded
    // overflow-scissor coverage (`clip_cov`).
    let win_cov = clip_cov;

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLYPH) {
        let glyph = glyph_frag(inst, px);
        // Negative alpha = outside the glyph's rect (see `glyph_frag`).
        if (glyph.a < 0.0) {
            discard;
        }
        return glyph * win_cov;
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_IMAGE) {
        // `backdrop_uv_rect` = (u0, v0, w, h) into the Rgba8UnormSrgb image atlas.
        // Tint = `inst.color` (white = unmodified). Per-corner radii clip via SDF.
        let size_s = max(inst.size * inst.scale, vec2<f32>(1.0));
        let center_img = inst.position + inst.size * 0.5;
        let rel = (px - (center_img - size_s * 0.5)) / size_s;
        if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
            discard;
        }
        let uv = inst.backdrop_uv_rect.xy + rel * inst.backdrop_uv_rect.zw;
        let sample = textureSampleLevel(image_tex, image_samp, uv, 0.0);
        let p = px - center_img;
        let outer_half = inst.size * 0.5 * inst.scale;
        let radii_s = inst.border_radius * inst.scale.x;
        let comp_dist = sd_rounded_rect(p, outer_half, radii_s);
        let aa = max(fwidth(comp_dist), 0.5);
        // Hard straight edges + rounded corners — no 1px lighter border where
        // an album cover's edge would otherwise feather over the placeholder.
        let comp_alpha = rounded_coverage(p, outer_half, radii_s, aa);
        let tint_lin = srgb_to_linear(inst.color.rgb);
        let a = sample.a * inst.color.a * inst.opacity * comp_alpha;
        let rgb = sample.rgb * tint_lin * a;
        return vec4<f32>(rgb, a) * win_cov;
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_EFFECT) {
        return effect_frag(inst, px) * inst.opacity * win_cov;
    }

    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5 * inst.scale;
    let radii_scaled = inst.border_radius * inst.scale.x;
    let comp_dist = sd_rounded_rect(p, outer_half, radii_scaled);
    let aa = max(fwidth(comp_dist), 0.5);
    // Hard straight edges + rounded corners — no 1px translucent-edge seam
    // over the content/backdrop. `comp_dist` still drives refraction + shadow.
    let comp_alpha = rounded_coverage(p, outer_half, radii_scaled, aa);

    var body: Body;
    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLASS) {
        // Per-glass blur + refraction parameters. `backdrop_uv_rect` is
        // repurposed for glass: x = blur radius (px), y = refraction
        // strength (px). Both authored in physical px (CPU scaled them
        // from logical).
        let blur_px = max(inst.backdrop_uv_rect.x, 0.0);
        let refraction_px = max(inst.backdrop_uv_rect.y, 0.0);

        // SDF-driven refraction. The gradient of `comp_dist` gives the
        // outward normal of the rounded-rect surface; near the rim the
        // backdrop is sampled along that normal as if light bent through
        // a curved edge. Strength fades from 1 at the rim to 0 at the
        // centre over `refraction_px` of inset distance.
        var sample_px = px;
        if (refraction_px > 0.0) {
            let eps = 1.0;
            let nx = sd_rounded_rect(p + vec2<f32>(eps, 0.0), outer_half, inst.border_radius)
                   - sd_rounded_rect(p - vec2<f32>(eps, 0.0), outer_half, inst.border_radius);
            let ny = sd_rounded_rect(p + vec2<f32>(0.0, eps), outer_half, inst.border_radius)
                   - sd_rounded_rect(p - vec2<f32>(0.0, eps), outer_half, inst.border_radius);
            let normal = normalize(vec2<f32>(nx, ny) + vec2<f32>(1.0e-6));
            let dist_inside = max(-comp_dist, 0.0);
            // Quadratic falloff from rim → centre.
            let edge = clamp(1.0 - dist_inside / refraction_px, 0.0, 1.0);
            sample_px = px + normal * refraction_px * edge * edge;
        }

        // Dual-filter upsample (Bjørge / ARM). 8 trilinear taps weighted
        // (4 cardinals + 2× 4 diagonals) / 12. Paired with the dual-down
        // pyramid kernel this approximates a true Gaussian over the
        // chosen mip depth — no visible box-pixel haloing at high blur.
        // hp scales with selected LOD so offsets cover ~one pixel of
        // the source mip at that level.
        let inv_screen = 1.0 / globals.screen_size;
        var center_uv = sample_px * inv_screen;
        let lod = clamp(log2(max(blur_px, 1.0)), 0.0, globals.max_backdrop_lod);
        let hp = 0.5 * exp2(lod) * inv_screen;
        // Per-fragment scatter in physical px → frosted-glass texture.
        // Scale stays mip-independent (author asks for "N px of jitter"
        // regardless of blur radius) — small N reads as fine frost,
        // large N as pebbled glass.
        if (inst.roughness > 0.0) {
            let r2 = (hash22(in.pos.xy) - vec2<f32>(0.5)) * 2.0;
            center_uv = center_uv + r2 * inst.roughness * inv_screen;
        }
        // Confine the kernel to the glass's own rect. A glass only frosts
        // what's behind *its* pixels — but the mip chain pre-mixes
        // neighbourhoods and the taps reach ±2·hp, so an edge-flush
        // fragment would otherwise drag in content that never sits under
        // the glass (a bright hover row touching a glass bar's top edge
        // read as a "reflection" inside the bar). Inset by the kernel's
        // reach at this LOD and clamp; edge fragments reuse interior
        // blurred values (edge-extend), which reads as frost, not leak.
        let reach = min(vec2<f32>(3.0 * exp2(lod)), outer_half);
        let lo_uv = (center - outer_half + reach) * inv_screen;
        let hi_uv = (center + outer_half - reach) * inv_screen;
        center_uv = clamp(center_uv, lo_uv, hi_uv);
        let s_l  = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>(-2.0 * hp.x, 0.0),         lod);
        let s_r  = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>( 2.0 * hp.x, 0.0),         lod);
        let s_t  = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>(0.0,        -2.0 * hp.y),  lod);
        let s_b  = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>(0.0,         2.0 * hp.y),  lod);
        let s_tl = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>(-hp.x, -hp.y),             lod);
        let s_tr = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>( hp.x, -hp.y),             lod);
        let s_bl = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>(-hp.x,  hp.y),             lod);
        let s_br = textureSampleLevel(backdrop_tex, backdrop_samp, center_uv + vec2<f32>( hp.x,  hp.y),             lod);
        var bg = (s_l + s_r + s_t + s_b + 2.0 * (s_tl + s_tr + s_bl + s_br)) * (1.0 / 12.0);

        // sRGB-space tint blend (avoids gamma-amplifying small white
        // tints into mid-grey).
        let tint_a = inst.color.a;
        let bg_a = max(bg.a, 1.0e-4);
        let bg_lin = bg.rgb / bg_a;
        let bg_srgb = linear_to_srgb(bg_lin);
        let blend_srgb = inst.color.rgb * tint_a + bg_srgb * (1.0 - tint_a);
        let blend_lin = srgb_to_linear(blend_srgb);
        let fill_a = tint_a + bg.a * (1.0 - tint_a);
        let fill_rgb_p = blend_lin * fill_a;
        body.rgb = fill_rgb_p * comp_alpha;
        body.a = fill_a * comp_alpha;
    } else {
        body = compute_fill(inst, p, outer_half, comp_alpha, aa);
    }

    let shadow = compute_shadow(inst, px, center, outer_half);
    let out_a = body.a + shadow.a * (1.0 - body.a);
    let out_rgb = body.rgb + shadow.rgb * (1.0 - body.a);
    return vec4<f32>(out_rgb, out_a) * inst.opacity * win_cov;
}

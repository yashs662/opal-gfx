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
    /// Corner radius (px) of the clip rect — rounds the scissor (rounded
    /// overflow clipping). 0 = square clip.
    clip_radius: f32,
    _pad1: f32,
}

// Signed distance to a rounded rect with min `lo`, max `hi`, corner
// radius `r`. <0 inside, >0 outside. Used for rounded scissor clipping.
fn rounded_clip_sd(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>, r: f32) -> f32 {
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let rr = min(r, min(half.x, half.y));
    let q = abs(p - center) - (half - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
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
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;

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
    let center_px = inst.position + inst.size * 0.5;
    let cover_scale = max(inst.scale, vec2<f32>(1.0));
    let half_scaled = inst.size * 0.5 * cover_scale + shadow_pad;
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
        world_pos.x / frame.screen_size.x * 2.0 - 1.0,
        1.0 - world_pos.y / frame.screen_size.y * 2.0,
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
    let qx = select(0u, 1u, p.x > 0.0);
    let qy = select(0u, 2u, p.y < 0.0);
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
    if (inst.clip_radius > 0.0) {
        if (rounded_clip_sd(px, inst.clip_rect.xy, inst.clip_rect.zw, inst.clip_radius) > 0.5) {
            discard;
        }
    } else if (px.x < inst.clip_rect.x || px.y < inst.clip_rect.y ||
        px.x > inst.clip_rect.z || px.y > inst.clip_rect.w) {
        discard;
    }

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLYPH) {
        let size = max(inst.size, vec2<f32>(1.0));
        let rel = (px - inst.position) / size;
        if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
            discard;
        }
        let uv = inst.backdrop_uv_rect.xy + rel * inst.backdrop_uv_rect.zw;
        let cov = textureSampleLevel(glyph_tex, glyph_samp, uv, 0.0).r;
        let rgb_lin = srgb_to_linear(inst.color.rgb);
        let a = inst.color.a * cov * inst.opacity;
        return vec4<f32>(rgb_lin * a, a);
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
        let comp_alpha = smoothstep(aa, -aa, comp_dist);
        let tint_lin = srgb_to_linear(inst.color.rgb);
        let a = sample.a * inst.color.a * inst.opacity * comp_alpha;
        let rgb = sample.rgb * tint_lin * a;
        return vec4<f32>(rgb, a);
    }

    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5 * inst.scale;
    let radii_scaled = inst.border_radius * inst.scale.x;
    let comp_dist = sd_rounded_rect(p, outer_half, radii_scaled);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);

    let body = compute_fill(inst, p, outer_half, comp_alpha, aa);
    let shadow = compute_shadow(inst, px, center, outer_half);

    let out_a = body.a + shadow.a * (1.0 - body.a);
    let out_rgb = body.rgb + shadow.rgb * (1.0 - body.a);
    return vec4<f32>(out_rgb, out_a) * inst.opacity;
}

// Coverage of the window-corner SDF at `px`. `1.0` inside the rounded
// boundary, `0.0` outside, with sub-pixel anti-aliasing at the rim.
// `1.0` when the radius is disabled — caller can multiply unconditionally.
fn window_corner_coverage(px: vec2<f32>) -> f32 {
    if (frame.window_corner_radius <= 0.0) {
        return 1.0;
    }
    let win_half = frame.screen_size * 0.5;
    let win_p = px - win_half;
    let win_r = vec4<f32>(frame.window_corner_radius);
    let win_d = sd_rounded_rect(win_p, win_half, win_r);
    let win_aa = max(fwidth(win_d), 0.5);
    return smoothstep(win_aa, -win_aa, win_d);
}

// Fragment entry for the final surface pass. Handles glass by sampling
// the pre-blurred backdrop texture.
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;
    // Scissor clip — rounded (overflow container with a radius) or square.
    var clip_cov = 1.0;
    if (inst.clip_radius > 0.0) {
        let sd = rounded_clip_sd(px, inst.clip_rect.xy, inst.clip_rect.zw, inst.clip_radius);
        if (sd > 0.5) {
            discard;
        }
        clip_cov = clamp(0.5 - sd, 0.0, 1.0); // 1px feather
    } else if (px.x < inst.clip_rect.x || px.y < inst.clip_rect.y ||
        px.x > inst.clip_rect.z || px.y > inst.clip_rect.w) {
        discard;
    }
    // Window-corner clip applies to every shape kind. Each branch
    // multiplies its final output by `win_cov` so glyphs, images, and
    // glass all respect the rounded boundary — the original
    // implementation only clipped the generic rect path, which left
    // the album-art image visible in the corners. Fold the rounded-clip
    // coverage in here so it rides the same multiply for free.
    let win_cov = window_corner_coverage(px) * clip_cov;

    if ((inst.shape_kind & SHAPE_KIND_MASK) == SHAPE_KIND_GLYPH) {
        // `backdrop_uv_rect` = (u0, v0, w, h) into the R8 glyph atlas.
        let size = max(inst.size, vec2<f32>(1.0));
        let rel = (px - inst.position) / size;
        // Fragments outside [0,1] (from any quad oversize) drop out.
        if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
            discard;
        }
        let uv = inst.backdrop_uv_rect.xy + rel * inst.backdrop_uv_rect.zw;
        let cov = textureSampleLevel(glyph_tex, glyph_samp, uv, 0.0).r;
        let rgb_lin = srgb_to_linear(inst.color.rgb);
        let a = inst.color.a * cov * inst.opacity;
        return vec4<f32>(rgb_lin * a, a) * win_cov;
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
        let comp_alpha = smoothstep(aa, -aa, comp_dist);
        let tint_lin = srgb_to_linear(inst.color.rgb);
        let a = sample.a * inst.color.a * inst.opacity * comp_alpha;
        let rgb = sample.rgb * tint_lin * a;
        return vec4<f32>(rgb, a) * win_cov;
    }

    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5 * inst.scale;
    let radii_scaled = inst.border_radius * inst.scale.x;
    let comp_dist = sd_rounded_rect(p, outer_half, radii_scaled);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);

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
        let inv_screen = 1.0 / frame.screen_size;
        var center_uv = sample_px * inv_screen;
        let lod = clamp(log2(max(blur_px, 1.0)), 0.0, frame.max_backdrop_lod);
        let hp = 0.5 * exp2(lod) * inv_screen;
        // Per-fragment scatter in physical px → frosted-glass texture.
        // Scale stays mip-independent (author asks for "N px of jitter"
        // regardless of blur radius) — small N reads as fine frost,
        // large N as pebbled glass.
        if (inst.roughness > 0.0) {
            let r2 = (hash22(in.pos.xy) - vec2<f32>(0.5)) * 2.0;
            center_uv = center_uv + r2 * inst.roughness * inv_screen;
        }
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

// Single SDF pipeline for all M1 shapes: rounded rectangles with per-corner
// radii, borders, and drop shadows. Outputs PREMULTIPLIED alpha — pair with
// a blend state of One / OneMinusSrcAlpha and a PreMultiplied surface
// composite mode.

struct ShapeInstance {
    color: vec4<f32>,
    border_color: vec4<f32>,
    shadow_color: vec4<f32>,
    border_radius: vec4<f32>,   // TL, TR, BL, BR (pixels)
    position: vec2<f32>,        // top-left in pixels
    size: vec2<f32>,            // pixels
    shadow_offset: vec2<f32>,
    _pad0: vec2<f32>,
    border_width: f32,
    shadow_blur: f32,
    shadow_opacity: f32,
    opacity: f32,
}

struct Frame {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;

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
    let min_px = inst.position - shadow_pad;
    let max_px = inst.position + inst.size + shadow_pad;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;

    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5;

    let comp_dist = sd_rounded_rect(p, outer_half, inst.border_radius);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);

    // Fill + border in a single formula. Border is an inside-inset ring.
    var body_rgb: vec3<f32>;
    var body_a: f32;
    if (inst.border_width > 0.0) {
        let inner_half = max(outer_half - vec2<f32>(inst.border_width), vec2<f32>(0.0));
        let inner_radii = max(inst.border_radius - vec4<f32>(inst.border_width), vec4<f32>(0.0));
        let inner_dist = sd_rounded_rect(p, inner_half, inner_radii);
        let inner_alpha = smoothstep(aa, -aa, inner_dist);
        let border_alpha = clamp(comp_alpha - inner_alpha, 0.0, 1.0);

        let fill_a = inst.color.a * inner_alpha;
        let fill_rgb = inst.color.rgb * fill_a;
        let bord_a = inst.border_color.a * border_alpha;
        let bord_rgb = inst.border_color.rgb * bord_a;

        // Border on top of fill.
        body_a = bord_a + fill_a * (1.0 - bord_a);
        body_rgb = bord_rgb + fill_rgb * (1.0 - bord_a);
    } else {
        body_a = inst.color.a * comp_alpha;
        body_rgb = inst.color.rgb * body_a;
    }

    // Shadow under body.
    var shadow_rgb = vec3<f32>(0.0);
    var shadow_a = 0.0;
    if (inst.shadow_opacity > 0.0) {
        let sd = sd_rounded_rect(px - (center + inst.shadow_offset), outer_half, inst.border_radius);
        let s_edge = max(inst.shadow_blur, 1.0);
        let s_i = smoothstep(s_edge, -s_edge, sd);
        shadow_a = inst.shadow_color.a * s_i * inst.shadow_opacity;
        shadow_rgb = inst.shadow_color.rgb * shadow_a;
    }

    let out_a = body_a + shadow_a * (1.0 - body_a);
    let out_rgb = body_rgb + shadow_rgb * (1.0 - body_a);

    // Global opacity: scale premultiplied color.
    return vec4<f32>(out_rgb, out_a) * inst.opacity;
}

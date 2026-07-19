// Overdraw heatmap pipeline — counter pass.
//
// Counts per-pixel shape coverage by drawing each shape with additive
// blending into an Rgba16Float accumulator (R channel = sum of SDF
// coverage). The compose pass lives in `overdraw_compose.wgsl`.
//
// Reuses the same vertex shader and shape buffer as `shape.wgsl`, so the
// shape coverage exactly matches what the regular pipeline draws.

// Must mirror `shape.wgsl`'s ShapeInstance FIELD-FOR-FIELD: both shaders
// bind the same instance SSBO, so a truncated mirror changes the array
// stride and skews every per-instance read past index 0.
struct ShapeInstance {
    color: vec4<f32>,
    border_color: vec4<f32>,
    shadow_color: vec4<f32>,
    border_radius: vec4<f32>,
    backdrop_uv_rect: vec4<f32>,
    clip_rect: vec4<f32>,
    clip_radius: vec4<f32>,
    position: vec2<f32>,
    size: vec2<f32>,
    shadow_offset: vec2<f32>,
    shape_kind: u32,
    roughness: f32,
    border_width: f32,
    shadow_blur: f32,
    shadow_opacity: f32,
    opacity: f32,
    scale: vec2<f32>,
    effect_id: f32,
    _pad: f32,
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
    let shadow_pad = vec2<f32>(
        abs(inst.shadow_offset.x) + max(inst.shadow_blur, 0.0) * 2.0,
        abs(inst.shadow_offset.y) + max(inst.shadow_blur, 0.0) * 2.0,
    );
    let min_px = inst.position - shadow_pad;
    let max_px = inst.position + inst.size + shadow_pad;

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
        case 0u: { s = r.x; }
        case 1u: { s = r.y; }
        case 2u: { s = r.z; }
        default: { s = r.w; }
    }
    s = min(s, min(xy.x, xy.y));
    return sd_rectangle(p, xy - vec2<f32>(s)) - s;
}

@fragment
fn fs_count(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;
    if (px.x < inst.clip_rect.x || px.y < inst.clip_rect.y ||
        px.x > inst.clip_rect.z || px.y > inst.clip_rect.w) {
        discard;
    }
    let center = inst.position + inst.size * 0.5;
    let p = in.world_pos - center;
    let outer_half = inst.size * 0.5;
    let comp_dist = sd_rounded_rect(p, outer_half, inst.border_radius);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);
    return vec4<f32>(comp_alpha, 0.0, 0.0, 0.0);
}

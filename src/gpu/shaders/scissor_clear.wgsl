// Scissored clear: one oversized triangle written with REPLACE blending,
// so the bound scissor rect — and nothing else — is reset to transparent.
//
// A render pass's `LoadOp::Clear` covers the whole attachment, which is
// exactly what a damage-rect raster must NOT do (it keeps the rest of the
// cached layer texture). This is the scissor-shaped equivalent.

@vertex
fn vs_clear(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle in clip space; the scissor does the bounding.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_clear() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

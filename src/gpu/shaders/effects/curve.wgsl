// Equaliser response curve: a filled, anti-aliased area under a
// Catmull-Rom spline through the param points.
//
// Params: `n` control-point heights in 0..1 measured from the bottom of
// the quad, sitting at x = (i + 0.5)/n. The ends hold flat.

fn fx_curve_catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * (2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

/// Curve height (0..1 from the bottom) at `x` in 0..1.
fn fx_curve_height(c: EffectCtx, x: f32) -> f32 {
    let n = c.params_len;
    if (n == 0u) { return 0.0; }
    if (n == 1u) { return fx_param(c, 0u); }
    let fx = x * f32(n) - 0.5; // fractional control-point index
    if (fx <= 0.0) { return fx_param(c, 0u); }
    if (fx >= f32(n - 1u)) { return fx_param(c, n - 1u); }
    let i = u32(floor(fx));
    let t = fx - f32(i);
    let i0 = select(i - 1u, 0u, i == 0u);
    let i3 = min(i + 2u, n - 1u);
    return clamp(fx_curve_catmull_rom(
        fx_param(c, i0),
        fx_param(c, i),
        fx_param(c, i + 1u),
        fx_param(c, i3),
        t,
    ), 0.0, 1.0);
}

fn fx_curve(c: EffectCtx) -> vec4<f32> {
    let h = fx_curve_height(c, c.rel.x);
    // Work in pixels within the quad so the stroke has a real px width.
    let edge_y_px = (1.0 - h) * c.size.y; // curve position from the top
    let d_vert = c.px.y - edge_y_px;      // + below the curve
    // Perpendicular distance = vertical / sqrt(1 + slope²), so a steep
    // part of the curve keeps the *same* line width instead of thinning.
    let slope = dpdx(edge_y_px);
    let inv_len = inverseSqrt(1.0 + slope * slope);
    let d_perp = d_vert * inv_len;
    // Fill below the curve (AA the boundary perpendicular to it).
    let fill_cov = clamp(d_perp + 0.5, 0.0, 1.0);
    // Even-width stroke centred on the curve (~1.5 px half-width + 1 px AA).
    let stroke = 1.0 - smoothstep(1.5, 2.5, abs(d_perp));
    let rgb_lin = srgb_to_linear(c.color.rgb);
    let a = max(c.color.a * fill_cov, stroke);
    return vec4<f32>(rgb_lin * a, a);
}

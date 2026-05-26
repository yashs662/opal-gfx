//! Procedurally-generated test images. Pure functions — no I/O, no
//! asset deps — so every example can construct its art inline and stay
//! deterministic for headless captures.

#![allow(dead_code)]

/// 64×64 RGBA8 gradient with a soft 8×8 checker overlay. The cheapest
/// "non-trivial" image we can hand the `Image` node type — enough
/// frequency content to make filtering / mipping visible without
/// shipping a binary asset.
pub fn make_demo_image() -> (u32, u32, Vec<u8>) {
    const W: u32 = 64;
    const H: u32 = 64;
    let mut bytes = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let fx = x as f32 / (W - 1) as f32;
            let fy = y as f32 / (H - 1) as f32;
            let cell = ((x / 8) + (y / 8)) % 2 == 0;
            let r = (fx * 255.0) as u8;
            let g = ((1.0 - fy) * 255.0) as u8;
            let b = if cell { 220 } else { 90 };
            bytes.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (W, H, bytes)
}

/// HSV → RGBA in `[0, 1]`. Used by every demo that wants a rainbow of
/// saturated colors without hand-listing them.
pub fn hsv(h: f32, s: f32, v: f32) -> [f32; 4] {
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

/// 256² loud HSV checker. Even cells are saturated HSV-rotated; odd
/// cells are near-black. High-frequency content makes blur amounts
/// visually obvious — used by `glass_grid`.
pub fn make_checker_image() -> (u32, u32, Vec<u8>) {
    const N: u32 = 256;
    const CELLS: u32 = 8;
    let cell = N / CELLS;
    let mut bytes = vec![0u8; (N * N * 4) as usize];
    for y in 0..N {
        for x in 0..N {
            let cx = x / cell;
            let cy = y / cell;
            let parity = (cx + cy) % 2;
            let h = ((cx as f32) / CELLS as f32 + (cy as f32) / (CELLS as f32 * 2.0)) % 1.0;
            let (r, g, b) = if parity == 0 {
                let rgba = hsv(h, 0.85, 1.0);
                (
                    (rgba[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (rgba[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (rgba[2] * 255.0).clamp(0.0, 255.0) as u8,
                )
            } else {
                (28, 28, 36)
            };
            let off = ((y * N + x) * 4) as usize;
            bytes[off] = r;
            bytes[off + 1] = g;
            bytes[off + 2] = b;
            bytes[off + 3] = 255;
        }
    }
    (N, N, bytes)
}

//! Glass parameter sweep. A colorful checkerboard background sits
//! behind a 4×4 grid of glass panels: blur varies along columns,
//! refraction varies along rows. Useful for eyeballing what each
//! authored value actually looks like.
//!
//! Run with:
//!     cargo run --example glass_grid

use frostify_gfx::{App, ImageHandle, Len, Scene};

const W: u32 = 1000;
const H: u32 = 720;

const BLURS: [f32; 4] = [4.0, 12.0, 24.0, 40.0];
const REFRACTIONS: [f32; 4] = [0.0, 6.0, 14.0, 24.0];

const PANEL_W: f32 = 220.0;
const PANEL_H: f32 = 140.0;
const GAP: f32 = 14.0;
const OFF_X: f32 = 24.0;
const OFF_Y: f32 = 60.0;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
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
    (
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

/// 256² colorful checker. Even cells are HSV-rotated saturated colors,
/// odd cells are near-black. Loud high-frequency content makes the
/// blur amount visually obvious — gentle dark cells let the rainbow
/// edges bleed through clearly when smeared.
fn make_checker_image() -> (u32, u32, Vec<u8>) {
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
                hsv_to_rgb(h, 0.85, 1.0)
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

fn build_scene(s: &mut Scene, art: ImageHandle) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        // Layer 1: full-window checker. Declared first so it lands in
        // the backdrop pass and every glass panel blurs it.
        p.image("bg", art)
            .abs(0.0, 0.0)
            .size_px(W as f32, H as f32);

        // Header label.
        p.text("hdr", "blur →   refraction ↓", 16.0)
            .abs(OFF_X, 22.0)
            .color([1.0, 1.0, 1.0, 1.0]);

        // 4×4 sweep. Constant tint (10% white) so the only variables
        // are .blur() and .refraction(). Same border radius, same size.
        for (row, &refraction) in REFRACTIONS.iter().enumerate() {
            for (col, &blur) in BLURS.iter().enumerate() {
                let x = OFF_X + (PANEL_W + GAP) * col as f32;
                let y = OFF_Y + (PANEL_H + GAP) * row as f32;
                let id = format!("g_{}_{}", col, row);
                let lbl_id = format!("lbl_{}_{}", col, row);
                let label = format!("blur {}  refr {}", blur as u32, refraction as u32);

                p.glass(&id)
                    .abs(x, y)
                    .size_px(PANEL_W, PANEL_H)
                    .radius(18.0)
                    .blur(blur)
                    .refraction(refraction)
                    .rgba(1.0, 1.0, 1.0, 0.10);

                p.text(&lbl_id, &label, 15.0)
                    .abs(x + 14.0, y + 14.0)
                    .color([1.0, 1.0, 1.0, 1.0]);
            }
        }
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let mut app = App::new("frostify-gfx — glass grid", W, H);
    let (iw, ih, bytes) = make_checker_image();
    let art = app.stage_image_rgba(iw, ih, bytes);
    // `FROSTIFY_AUTOCAPTURE=1` writes one frame to `debug_captures/` and
    // exits — the deterministic root-layer-parity capture gate. Without
    // it the example runs interactively for eyeballing glass params.
    let app = app.scene(move |s| build_scene(s, art)).capture_from_env();
    app.run()
}

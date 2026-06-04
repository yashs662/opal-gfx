//! P6 external-texture layer proof. An `.external()` node owns no
//! instances; its pixels come from a caller-supplied `wgpu::Texture`
//! (the shape a video / Spotify Canvas decoder takes). Registering a new
//! texture + re-rendering is **composite-only** — no raster, no flatten —
//! the analog of a decoder swapping frames.
//!
//! The example registers a solid-red external texture, captures, asserts
//! the node's rect reads red; then swaps to green (no flush) and asserts
//! it reads green — proving the composite samples the live external view.
//!
//!     cargo run --example external_layer

use frostify_gfx::{App, Len, Scene};

const W: u32 = 400;
const H: u32 = 300;
// External node rect (logical px); at scale 2 → physical 2×.
const EX_X: f32 = 100.0;
const EX_Y: f32 = 80.0;
const EX_W: f32 = 200.0;
const EX_H: f32 = 140.0;

fn build_scene(s: &mut Scene) {
    s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
        p.rect("bg")
            .abs(0.0, 0.0)
            .size_px(W as f32, H as f32)
            .rgba(0.1, 0.1, 0.12, 1.0);
        // External-texture layer: no instances of its own; the compositor
        // blits the registered texture into this rect.
        p.rect("video")
            .external()
            .abs(EX_X, EX_Y)
            .size_px(EX_W, EX_H);
    });
}

/// Upload a `w×h` solid-colour texture in the surface format and return a
/// view (what a decoder would hand the compositor each frame). Colour is
/// premultiplied straight bytes (opaque → rgb as-is).
fn solid_view(
    gpu: &frostify_gfx::gpu::GpuContext,
    w: u32,
    h: u32,
    rgba: [u8; 4],
) -> wgpu::TextureView {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("external solid"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.surface_config.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Surface is BGRA — swap R/B for the upload bytes.
    let px = [rgba[2], rgba[1], rgba[0], rgba[3]];
    let data: Vec<u8> = std::iter::repeat_n(px, (w * h) as usize).flatten().collect();
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Average colour (straight sRGB bytes) of the external node's centre.
fn avg_center(rgba: &[u8], cw: u32, ch: u32) -> (f32, f32, f32) {
    let sx = cw / W;
    let sy = ch / H;
    let cx = ((EX_X + EX_W / 2.0) as u32) * sx;
    let cy = ((EX_Y + EX_H / 2.0) as u32) * sy;
    let half = 16u32;
    let (mut r, mut g, mut b, mut n) = (0f64, 0f64, 0f64, 0f64);
    for y in cy - half..cy + half {
        for x in cx - half..cx + half {
            let o = ((y * cw + x) * 4) as usize;
            r += rgba[o] as f64;
            g += rgba[o + 1] as f64;
            b += rgba[o + 2] as f64;
            n += 1.0;
        }
    }
    ((r / n) as f32, (g / n) as f32, (b / n) as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("frostify-gfx — external layer", W, H)
        .scene(build_scene)
        .headless(|h| {
            let video = h.ctx.node("video").expect("video node");
            // The external node must be a promoted layer carrying an
            // external rect (no instances).
            let has_ext = h.layer_tree.layers().iter().any(|l| {
                l.root == Some(video) && l.external.is_some()
            });
            assert!(has_ext, "expected an external-texture layer for `video`");

            let (tw, th) = (
                (EX_W * h.scale_factor) as u32,
                (EX_H * h.scale_factor) as u32,
            );

            // Frame 1: red. Register + recomposite (no flush/raster).
            let red = solid_view(h.gpu, tw, th, [220, 30, 30, 255]);
            h.gpu.set_external_texture(video, red);
            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let (r1, g1, b1) = avg_center(&rgba, cw, ch);
            log::info!("frame red   → center = ({r1:.0}, {g1:.0}, {b1:.0})");
            assert!(r1 > 150.0 && g1 < 90.0 && b1 < 90.0, "external red not composited");

            // Frame 2: green. Swap the texture only — composite-only, the
            // scene tree is untouched (no flush).
            let green = solid_view(h.gpu, tw, th, [30, 200, 60, 255]);
            h.gpu.set_external_texture(video, green);
            h.render();
            let (rgba, cw, ch) = h.gpu.capture_rgba();
            let (r2, g2, b2) = avg_center(&rgba, cw, ch);
            log::info!("frame green → center = ({r2:.0}, {g2:.0}, {b2:.0})");
            assert!(g2 > 130.0 && r2 < 110.0, "external green frame not composited");

            log::info!("PASS: external-texture layer composites caller frames (composite-only swap)");
        });
    app.run()
}

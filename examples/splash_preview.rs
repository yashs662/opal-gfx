//! Visual check for the CPU startup splash. Builds the logo + wordmark
//! bitmap (the cross-platform part) and dumps it to a PNG so the
//! compositing can be eyeballed without a GPU. On Windows it also pops the
//! real layered splash window for a few seconds.
//!
//! Run: `cargo run -p opal-gfx --example splash_preview`

use std::fs::File;
use std::io::BufWriter;

use opal_gfx::TextResources;
use opal_gfx::splash::{SplashBitmap, SplashConfig};

fn main() {
    let cfg = SplashConfig {
        logo_svg: include_bytes!("../../opal/assets/logo/geometric-opal.svg").to_vec(),
        wordmark: "Opal".to_string(),
        logo_px: 64.0,
        wordmark_px: 36.0,
        gap_px: 16.0,
        wordmark_color: [0.95, 0.95, 0.96, 1.0],
        bg_color: [0.06, 0.06, 0.07, 1.0],
    };
    let mut text = TextResources::new();
    let scale = 2.0; // exercise the DPI path
    let bmp = SplashBitmap::build(&cfg, &mut text, scale).expect("build splash bitmap");
    println!("splash bitmap: {}x{} px @ {scale}x", bmp.w, bmp.h);

    std::fs::create_dir_all("debug_captures").ok();
    let path = "debug_captures/splash_preview.png";
    let file = File::create(path).expect("create png");
    let mut enc = png::Encoder::new(BufWriter::new(file), bmp.w, bmp.h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&bmp.rgba)
        .expect("png data");
    println!("wrote {path}");

    #[cfg(windows)]
    {
        let s = opal_gfx::splash::Splash::show(&bmp, 700, 400).expect("show splash");
        println!("showing layered splash for 3s...");
        std::thread::sleep(std::time::Duration::from_secs(3));
        s.close();
    }
}

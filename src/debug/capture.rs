use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::gpu::FrameStats;

static FRAME_COUNTER: AtomicU32 = AtomicU32::new(0);

pub fn screenshot_path(dir: &Path) -> PathBuf {
    let n = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("frame_{n:04}.png"))
}

pub fn save_png(
    path: &Path,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // `Compression::Fast` (≈ zlib level 1) over `Default` (≈ 6) cuts
    // F2 encode time from ~1 s+ per frame to under 100 ms for a
    // 1280×780 capture. The file gets larger (~2 MB vs ~1 MB) but
    // debug captures aren't shipped, so the trade is one-sided.
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Encode + persist a screenshot **off the UI thread**. The hot path
/// (F2 mid-app) used to block winit's event handler for the full
/// duration of `capture_rgba` + PNG encode + disk write — multiple
/// seconds on debug builds. We split it: the synchronous part stays
/// inline (the GPU readback can't escape `&mut Gpu`), but the encode +
/// write moves to a spawned thread so the loop resumes immediately.
/// The closure logs on completion / failure since the caller can't
/// `await` the result.
pub fn save_png_async(
    path: PathBuf,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("opal.png_encode".into())
        .spawn(move || match save_png(&path, &rgba, width, height) {
            Ok(()) => log::info!("screenshot saved: {}", path.display()),
            Err(e) => log::error!("screenshot failed: {e}"),
        })
        .expect("spawn png_encode thread")
}

/// Write a `frame_NNNN.stats.json` next to the screenshot. Hand-rolled
/// JSON so the crate stays serde-free for stage 1.
pub fn write_stats_sidecar(png_path: &Path, stats: &FrameStats) {
    let path = sidecar_path(png_path);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = format!(
        "{{\"cpu_ms\":{:.4},\"gpu_ms\":{:.4},\"opaque_ms\":{:.4},\"final_ms\":{:.4},\"overdraw_ms\":{:.4},\"instance_count\":{},\"opaque_count\":{},\"glass_count\":{},\"drawcalls\":{},\"dirty_mask\":{}}}\n",
        stats.cpu_ms,
        stats.gpu_ms,
        stats.opaque_ms,
        stats.final_ms,
        stats.overdraw_ms,
        stats.instance_count,
        stats.opaque_count,
        stats.glass_count,
        stats.drawcalls,
        stats.dirty_mask,
    );
    match File::create(&path).and_then(|f| BufWriter::new(f).write_all(json.as_bytes())) {
        Ok(()) => log::info!("stats sidecar saved: {}", path.display()),
        Err(e) => log::error!("stats sidecar failed: {e}"),
    }
}

fn sidecar_path(png_path: &Path) -> PathBuf {
    let stem = png_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("frame");
    let dir = png_path.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!("{stem}.stats.json"))
}

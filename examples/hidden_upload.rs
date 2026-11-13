//! Hidden-window upload soak. A decoder-shaped thread streams clip after
//! clip into an `.external()` node's resident frame set while the script
//! hides the window (as another Space does on macOS), then shows it again.
//!
//! Every upload has to reach the GPU even though nothing presents:
//! process memory must stay flat for the whole hidden stretch. Watch it
//! from outside (Task Manager / Activity Monitor) or read the engine's own
//! `gpu memory` lines.
//!
//!     cargo run --release --example hidden_upload --features automation
//!
//! `HIDDEN_SECS` (default 20) sets how long the window stays hidden.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use opal_gfx::automation::{Script, Step};
use opal_gfx::{App, Len, NodeId, Scene};

const W: u32 = 400;
const H: u32 = 300;
/// One uploaded frame: 256 KiB of RGBA.
const FRAME: u32 = 256;
/// Frames per clip — a new epoch (a "track change") starts after each.
const CLIP_FRAMES: u64 = 60;
const FRAME_INTERVAL: Duration = Duration::from_millis(5);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let hidden = std::env::var("HIDDEN_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut steps = vec![Step::Wait(Duration::from_secs(3)), Step::MemReport];
    steps.push(Step::Occlude(true));
    for _ in 0..hidden {
        steps.push(Step::Wait(Duration::from_secs(1)));
    }
    steps.extend([
        Step::MemReport,
        Step::Occlude(false),
        Step::Wait(Duration::from_secs(3)),
        Step::MemReport,
    ]);

    let video: Arc<OnceLock<NodeId>> = Arc::default();
    let scene_video = video.clone();
    let app = App::new("opal-gfx — hidden upload soak", W, H, ())
        .scene(move |s: &mut Scene, _| {
            s.col("root").w(Len::Fill).h(Len::Fill).child(|p| {
                let id = p
                    .rect("video")
                    .external()
                    .abs(0.0, 0.0)
                    .size_px(W as f32, H as f32)
                    .id();
                let _ = scene_video.set(id);
            });
        })
        .automation(Script::new(steps));

    let sink = app.frame_sink();
    std::thread::spawn(move || {
        let node = loop {
            match video.get() {
                Some(&n) => break n,
                None => std::thread::sleep(FRAME_INTERVAL),
            }
        };
        for n in 0..=u64::MAX {
            let shade = (n % 255) as u8;
            let rgba = [shade, 80, 255 - shade, 255].repeat((FRAME * FRAME) as usize);
            sink.push_frame(node, n / CLIP_FRAMES, FRAME, FRAME, rgba);
            std::thread::sleep(FRAME_INTERVAL);
        }
    });
    app.run()
}

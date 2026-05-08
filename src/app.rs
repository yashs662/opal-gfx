//! Application shell — wraps winit + wgpu + scene + binds + input + anim.
//!
//! Use it like:
//!
//! ```ignore
//! frostify_gfx::app::App::new("demo", 1100, 750)
//!     .scene(|s| build_demo(s))
//!     .on_key(|code, state, ctx| handle_key(code, state, ctx))
//!     .run()
//!     .unwrap();
//! ```
//!
//! The app owns:
//! - a `SceneCtx` (tree + name index + bind registry) populated by the
//!   user-supplied `scene` closure;
//! - a `winit` window + `GpuContext` lazily created in `resumed`;
//! - an `InputState` driven by the window's pointer events;
//! - a `Timeline` whose tweens target the per-bind "displayed" signals
//!   that the scene allocated when an `animated(...)` color was set.
//!
//! The shell wires these together through a small set of helpers — the
//! one place to look when changing the reactive flow is
//! [`App::process_binds`] (snap or retarget) plus
//! [`App::pump_animated_displays`] (push interpolated values into the
//! tree on each anim tick). Everything else is winit boilerplate.
//!
//! Built-in keys: `Esc` exits, `F2` writes a screenshot under the
//! configured `capture_dir`, `F5` forces a full tree rebuild + redraw.
//! Pass `.on_key(...)` for additional bindings; pass `.headless(...)`
//! for scripted offscreen captures (used by CI/self-verify).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowId};

use crate::anim::Timeline;
use crate::debug;
use crate::gpu::{FrameStats, GpuContext, ImageHandle, ShapeInstance};
use crate::input::InputState;
use crate::node::{FlatEvent, HitEntry, ScrollAxis, ScrollHit, ScrollbarHit, WindowAction};
use crate::scene::{ColorBindSlot, PositionBindSlot, Scene, SceneCtx, SizeBindSlot};

/// Pixel band on each window edge that triggers a system resize-drag
/// instead of a normal click. 6 px matches Windows' frameless feel.
const RESIZE_GUTTER: f32 = 6.0;

/// Tween-key namespaces reserved for the bind registry. One key per
/// (kind, slot index) pair. User-chosen keys should stay below
/// `0xC000_0000`. Each kind gets a 16M-slot window — way more than any
/// realistic scene will ever need.
const BIND_TWEEN_KEY_COLOR: u32 = 0xC000_0000;
const BIND_TWEEN_KEY_POSITION: u32 = 0xC100_0000;
const BIND_TWEEN_KEY_SIZE: u32 = 0xC200_0000;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub decorations: bool,
    pub transparent: bool,
    pub blur: bool,
    pub capture_dir: PathBuf,
}

impl AppConfig {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            decorations: false,
            transparent: true,
            blur: true,
            capture_dir: PathBuf::from("debug_captures"),
        }
    }
}

enum StagedImage {
    Png(Vec<u8>),
    Rgba { w: u32, h: u32, bytes: Vec<u8> },
}

/// Active thumb-drag bookkeeping. While the user holds the mouse on a
/// thumb the pointer is the authoritative position — we map cursor
/// motion 1:1 onto scroll offset and write through `set_scroll_immediate`
/// so the spring stays at rest.
#[derive(Copy, Clone, Debug)]
struct BarDrag {
    node_id: crate::node::NodeId,
    axis: ScrollAxis,
    /// Cursor position when the press landed (physical px).
    pointer_origin: f32,
    /// Scroll offset (`current`) at the press instant.
    scroll_origin: f32,
    /// `track_len - thumb_len` — pixel range the thumb travels over.
    /// Drag delta on the cursor maps to scroll delta as
    /// `cursor_dy / track_travel * max_offset`.
    track_travel: f32,
    /// Maximum scroll offset on the dragged axis (`content - rect`).
    max_offset: f32,
}

/// Optional headless script. Called once from `resumed` after the
/// window is ready and the first frame has been captured. Use it to
/// drive synthetic input, mutate the tree, advance the timeline by
/// hand, and capture additional frames. After the closure returns,
/// the event loop exits.
pub type HeadlessFn = Box<dyn FnOnce(&mut HeadlessHelper)>;

/// Optional per-keypress hook. Called on `Pressed` events for any
/// `KeyCode` the shell hasn't already handled (Esc/F2/F5 are
/// reserved). After the closure returns, the shell processes binds,
/// flushes the tree, and requests a redraw if anything changed.
pub type KeyFn = Box<dyn FnMut(KeyCode, ElementState, &mut SceneCtx)>;

pub struct App {
    config: AppConfig,
    ctx: SceneCtx,
    instances: Vec<ShapeInstance>,
    glass_count: u32,
    /// Flatten output buffers reused across frames. Each `flush_tree`
    /// `clear()`s and re-fills them so the heap allocations made by
    /// the first flatten amortize over every subsequent one.
    flat_events: Vec<FlatEvent>,
    hits: Vec<HitEntry>,
    scroll_hits: Vec<ScrollHit>,
    scroll_bars: Vec<ScrollbarHit>,
    /// Per-axis active drag bookkeeping. While `Some`, pointer-move
    /// updates the captured axis's scroll position 1:1.
    bar_drag: Option<BarDrag>,
    /// Cursor position pending application during a thumb drag.
    /// CursorMoved events fire at the OS rate (often 500+ Hz on
    /// Windows raw mouse), and applying every one re-flattens + re-
    /// uploads the instance buffer per pixel of motion. We buffer the
    /// latest pointer here and let `about_to_wait` apply it at frame
    /// rate via `set_scroll_immediate`.
    pending_drag_cursor: Option<[f32; 2]>,
    /// Latched modifier state from `WindowEvent::ModifiersChanged`.
    /// Consulted by wheel routing for shift-axis swap.
    modifiers: winit::event::Modifiers,
    /// Scroll keys currently physically held. Tracked separately from
    /// OS auto-repeat events because the auto-repeat initial delay
    /// (Windows default ~250 ms) is longer than the scroll-input
    /// quiescence gate; without this set the gate would lapse during
    /// the gap between initial press and first repeat, fire a bounce,
    /// then have the bounce cancelled by the first repeat — the user
    /// sees that as a stretch / ease-back / stretch flicker. While
    /// any element is in this set, `about_to_wait` calls
    /// `tree.poke_scroll_input_recency()` so settle stays gated.
    held_scroll_keys: std::collections::HashSet<KeyCode>,
    input: InputState,
    timeline: Timeline,
    on_key: Option<KeyFn>,
    headless: Option<HeadlessFn>,
    /// Number of still frames to capture and exit after, or `None` for
    /// a normal interactive run. `Some(n)` triggers headless capture
    /// mode in `resumed`.
    capture_frames: Option<u32>,
    /// Last dirty bitmask consumed by `flush_tree`. Cleared on read by
    /// `take_dirty`, so we cache it for later stat dumps.
    last_dirty_mask: u32,
    /// Wall-clock CPU time of the most recent `render_once` call.
    last_cpu_ms: f32,
    /// Stats snapshot taken at the end of the most recent `render_once`.
    /// `save_screenshot` reads from here so the sidecar reflects the
    /// live render, not the re-encode that `capture_rgba` performs
    /// (which would clear `backdrop_dirty` and lose drawcall counts).
    last_render_stats: Option<FrameStats>,
    /// Stat-dump cadence: continuously log on every render when set.
    /// Toggled by `FROSTIFY_STATS=1` env var or by F1 in interactive mode.
    stats_log: bool,
    /// Bar-gauge HUD overlay: enabled by F1 (along with stats logging).
    /// Stage-1 has no text renderer, so the HUD is rect-only.
    hud_enabled: bool,
    /// Last cursor icon we set on the window — avoid spamming
    /// `set_cursor` when the cursor stays in the same gutter.
    last_cursor_icon: CursorIcon,
    /// Display DPI scale factor. Inputs to layout (`Len::Px`,
    /// padding, gap, `abs`, font sizes) and to ShapeInstance
    /// (border_radius, border_width, shadow_*) are in *logical* px
    /// and multiplied by this on the way to the GPU. Initialised
    /// from `window.scale_factor()` and refreshed on
    /// `WindowEvent::ScaleFactorChanged` (drag between monitors).
    scale_factor: f32,
    /// Logical inner size in winit-units. Tracked separately so we
    /// can survive rapid `ScaleFactorChanged` events that arrive
    /// before the matching `Resized` updates the GPU surface (e.g.
    /// dragging across a monitor boundary repeatedly).
    logical_size: [u32; 2],
    /// Timestamp of the most recent `ScaleFactorChanged`. Any `Resized`
    /// within a short window after this is treated as DPI-driven (do
    /// not derive logical from physical). Single-shot flag was too
    /// fragile: Win11 24H2 (winit#4041) leaks extra Resized events
    /// from earlier crossings, each of which would otherwise corrupt
    /// `logical_size` and compound on the next cross.
    last_dpi_change: Option<Instant>,
    /// Window outer position captured *before* the most recent
    /// ScaleFactorChanged. winit's WM_DPICHANGED handler preserves
    /// the cursor's **logical** offset within the window — when the
    /// physical width halves (4K→1080p), the window shifts hundreds
    /// of pixels on screen even though the user is dragging in the
    /// opposite direction. Browsers preserve the **physical** offset
    /// (window stays under the hand). We restore this position in
    /// the next DPI Resized to match that behavior.
    pre_dpi_outer: Option<winit::dpi::PhysicalPosition<i32>>,
    /// Wall-clock instant of the most recent `about_to_wait` scroll
    /// tick. `None` while the loop is parked on `Wait`. Drives the
    /// scroll-spring `dt`. Reset to `None` whenever both timeline +
    /// scrolls go idle, so the first tick after wake-up doesn't see a
    /// stale dt spanning the idle gap.
    last_scroll_tick: Option<Instant>,
    /// Last wall-clock instant the HUD overlay was rebuilt. Throttles
    /// `refresh_hud_overlay` to ~10 Hz so a fast-rendering scene
    /// doesn't re-shape 6 stat lines every frame just to display
    /// sub-ms jitter.
    last_hud_refresh: Option<Instant>,
    /// Images staged for upload at GPU init. Index in this vec equals
    /// the [`ImageHandle`] returned to the caller — the image atlas
    /// allocates handles sequentially from 0, so the two stay in
    /// lock-step as long as we drain in order.
    staged_images: Vec<StagedImage>,
    // Lazy:
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
}

impl App {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            config: AppConfig::new(title, width, height),
            ctx: SceneCtx::new(),
            instances: Vec::new(),
            glass_count: 0,
            flat_events: Vec::new(),
            hits: Vec::new(),
            scroll_hits: Vec::new(),
            scroll_bars: Vec::new(),
            bar_drag: None,
            pending_drag_cursor: None,
            modifiers: winit::event::Modifiers::default(),
            held_scroll_keys: std::collections::HashSet::new(),
            input: InputState::new(),
            timeline: Timeline::new(),
            on_key: None,
            headless: None,
            capture_frames: None,
            last_dirty_mask: 0,
            last_cpu_ms: 0.0,
            last_render_stats: None,
            stats_log: std::env::var_os("FROSTIFY_STATS").is_some(),
            hud_enabled: std::env::var_os("FROSTIFY_HUD").is_some(),
            last_cursor_icon: CursorIcon::Default,
            scale_factor: 1.0,
            logical_size: [width, height],
            last_dpi_change: None,
            pre_dpi_outer: None,
            last_scroll_tick: None,
            last_hud_refresh: None,
            staged_images: Vec::new(),
            window: None,
            gpu: None,
        }
    }

    /// Stage a PNG byte slice for upload to the image atlas. Returns
    /// a virtual handle that the scene closure can use immediately;
    /// the decode + upload runs at GPU init. `include_bytes!(...)` is
    /// the typical source. Reuse the same handle in multiple
    /// `Scene::image(...)` calls.
    pub fn stage_image_png(&mut self, bytes: impl Into<Vec<u8>>) -> ImageHandle {
        let id = self.staged_images.len() as u32;
        self.staged_images.push(StagedImage::Png(bytes.into()));
        ImageHandle(id)
    }

    /// Stage pre-decoded `Rgba8UnormSrgb` pixels (`w*h*4` bytes,
    /// row-major, top-left origin). Same scheduling as
    /// [`Self::stage_image_png`].
    pub fn stage_image_rgba(&mut self, w: u32, h: u32, bytes: Vec<u8>) -> ImageHandle {
        let id = self.staged_images.len() as u32;
        self.staged_images
            .push(StagedImage::Rgba { w, h, bytes });
        ImageHandle(id)
    }

    /// Run the user-supplied scene builder. Mutates the inner
    /// `SceneCtx` immediately — by the time `run` is called the tree
    /// and the bind registry are fully populated.
    pub fn scene<F: FnOnce(&mut Scene)>(mut self, f: F) -> Self {
        let mut scene = Scene::root(&mut self.ctx);
        f(&mut scene);
        self
    }

    /// Provide a closure invoked on every key-down event for keys the
    /// shell hasn't already consumed. The closure can mutate signals,
    /// the tree, or the scene context directly.
    pub fn on_key<F: FnMut(KeyCode, ElementState, &mut SceneCtx) + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_key = Some(Box::new(f));
        self
    }

    /// Configure the directory used by F2 screenshots and headless
    /// captures.
    pub fn capture_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.capture_dir = dir.into();
        self
    }


    /// Capture `frames` still snapshots of the initial scene under
    /// `capture_dir` and exit. For scripted scenarios that mutate
    /// state between frames use [`App::headless`] instead. `frames`
    /// must be ≥ 1; the first frame is always written before any
    /// scripted callback runs.
    pub fn capture(mut self, frames: u32, dir: impl Into<PathBuf>) -> Self {
        self.capture_frames = Some(frames.max(1));
        self.config.capture_dir = dir.into();
        self
    }

    /// Convenience: capture one frame to the default `capture_dir`
    /// and exit. Equivalent to `capture(1, "debug_captures")`.
    pub fn capture_once(self) -> Self {
        self.capture(1, "debug_captures")
    }

    /// Env-var shim for the legacy `FROSTIFY_AUTOCAPTURE` flag.
    /// Returns `self` unchanged when the variable is not set, so the
    /// call is harmless in normal interactive runs. CI/self-verify
    /// keeps working without code changes; scripted multi-frame flows
    /// still use [`App::headless`] separately.
    pub fn capture_from_env(self) -> Self {
        if std::env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
            self.capture_once()
        } else {
            self
        }
    }

    /// Provide a one-shot scripted headless callback. The closure
    /// receives a `HeadlessHelper` with mutable access to every
    /// piece of shell state, runs whatever sequence it likes, and
    /// returns. The shell then exits the event loop. Implies
    /// `capture_once()` so the initial frame is always saved before
    /// the script runs.
    pub fn headless<F: FnOnce(&mut HeadlessHelper) + 'static>(mut self, f: F) -> Self {
        self.headless = Some(Box::new(f));
        if self.capture_frames.is_none() {
            self.capture_frames = Some(1);
        }
        self
    }

    /// Window decoration flag. Default `false` (frameless).
    pub fn decorations(mut self, on: bool) -> Self {
        self.config.decorations = on;
        self
    }

    /// Get a read-only view of the scene context — useful for tests
    /// that build the scene then assert on it without ever calling
    /// `run`.
    pub fn ctx(&self) -> &SceneCtx {
        &self.ctx
    }

    /// Take the scene context out of the app. Intended as the
    /// escape hatch when a consumer wants to drive winit + wgpu
    /// themselves but still wants the declarative scene builder.
    pub fn into_ctx(self) -> SceneCtx {
        self.ctx
    }

    /// Run the event loop. Blocks until the window closes or the
    /// headless script exits.
    pub fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    // ---- Internal helpers ----------------------------------------------

    /// Re-flatten + upload the tree if any dirty flag is set.
    fn flush_tree(&mut self) -> bool {
        let mask = self.ctx.tree.take_dirty();
        if mask == 0 {
            return false;
        }
        self.last_dirty_mask = mask;
        if mask & (crate::node::dirty::TREE | crate::node::dirty::TRANSFORM) != 0 {
            let viewport = self.viewport();
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                viewport,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        self.ctx.tree.flatten_into_buffers(
            self.scale_factor,
            &mut self.flat_events,
            &mut self.hits,
            &mut self.scroll_hits,
            &mut self.scroll_bars,
        );
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        if let Some(gpu) = self.gpu.as_mut() {
            self.instances.clear();
            self.glass_count = expand_events_into(
                &self.flat_events,
                &mut self.instances,
                gpu,
                &mut self.ctx.text,
                self.scale_factor,
            );
            gpu.set_instances(&self.instances, self.glass_count, backdrop_hint);
        }
        true
    }

    fn viewport(&self) -> [f32; 2] {
        if let Some(g) = self.gpu.as_ref() {
            [
                g.surface_config.width as f32,
                g.surface_config.height as f32,
            ]
        } else {
            [self.config.width as f32, self.config.height as f32]
        }
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn render_once(&mut self) {
        if self.hud_enabled {
            self.refresh_hud_overlay();
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let t0 = Instant::now();
        gpu.render_frame();
        self.last_cpu_ms = t0.elapsed().as_secs_f32() * 1_000.0;
        let snapshot = self.current_stats();
        self.last_render_stats = Some(snapshot);
        if self.stats_log {
            log::info!("frame stats: {snapshot:?}");
        }
    }

    /// Build a numeric-label HUD from the most recent frame stats and
    /// upload it as overlay instances. Cleared when `hud_enabled` flips
    /// off. Throttled to ~10 Hz: cpu/gpu_ms jitter would otherwise
    /// invalidate any equality check and re-shape 6 lines per render
    /// frame for sub-ms wiggle a human can't read anyway.
    fn refresh_hud_overlay(&mut self) {
        let now = Instant::now();
        let due = self
            .last_hud_refresh
            .map(|t| now.duration_since(t) >= Duration::from_millis(100))
            .unwrap_or(true);
        if !due {
            return;
        }
        self.last_hud_refresh = Some(now);
        let stats = self.current_stats();
        let scale = self.scale_factor;
        if let Some(gpu) = self.gpu.as_mut() {
            let instances = build_hud_instances(&stats, gpu, &mut self.ctx.text, scale);
            gpu.set_overlay_instances(&instances);
        }
    }

    fn clear_hud_overlay(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_overlay_instances(&[]);
        }
        self.last_hud_refresh = None;
    }

    /// Forward the GPU memory-allocation snapshot from the renderer.
    /// Returns `None` if the GPU isn't initialised yet.
    pub fn memory_report(&self) -> Option<crate::gpu::MemoryReport> {
        self.gpu.as_ref().map(|g| g.memory_report())
    }

    /// Combine the renderer's GPU stats with the app-side CPU + dirty mask.
    pub fn current_stats(&self) -> FrameStats {
        let mut s = self
            .gpu
            .as_ref()
            .map(|g| g.last_frame_stats())
            .unwrap_or_default();
        s.cpu_ms = self.last_cpu_ms;
        s.dirty_mask = self.last_dirty_mask;
        s
    }

    /// Walk every reactive bind. For each slot whose source version
    /// has advanced since last seen: snap or retarget.
    fn process_binds(&mut self, now: Instant) {
        process_color_binds(
            &mut self.ctx.binds.color,
            &mut self.ctx.tree,
            &mut self.timeline,
            now,
        );
        process_position_binds(
            &mut self.ctx.binds.position,
            &mut self.ctx.tree,
            &mut self.timeline,
            now,
        );
        process_size_binds(
            &mut self.ctx.binds.size,
            &mut self.ctx.tree,
            &mut self.timeline,
            now,
        );
    }

    /// For animated binds, copy the current `displayed` signal value
    /// (driven by the timeline) into the tree. Called after every
    /// timeline tick.
    fn pump_animated_displays(&mut self) {
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.position {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.size {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_size_px(slot.node_id, disp.get());
            }
        }
    }

    /// Common reaction sequence after any input or key event:
    /// re-target binds, push displayed values, flush, redraw.
    fn react(&mut self) {
        self.process_binds(Instant::now());
        self.pump_animated_displays();
        if self.flush_tree() {
            self.request_redraw();
        }
    }

    /// Resize gutter hit-test. Returns the direction whose gutter
    /// contains `(x, y)`, or `None` if the cursor is in the interior.
    /// Skipped entirely when the window is maximised — Windows refuses
    /// to start a resize drag in that state anyway.
    fn edge_at(&self, x: f32, y: f32) -> Option<ResizeDirection> {
        let win = self.window.as_ref()?;
        if win.is_maximized() {
            return None;
        }
        let g = self.gpu.as_ref()?;
        let w = g.surface_config.width as f32;
        let h = g.surface_config.height as f32;
        let l = x < RESIZE_GUTTER;
        let r = x > w - RESIZE_GUTTER;
        let t = y < RESIZE_GUTTER;
        let b = y > h - RESIZE_GUTTER;
        match (t, b, l, r) {
            (true, _, true, _) => Some(ResizeDirection::NorthWest),
            (true, _, _, true) => Some(ResizeDirection::NorthEast),
            (_, true, true, _) => Some(ResizeDirection::SouthWest),
            (_, true, _, true) => Some(ResizeDirection::SouthEast),
            (true, _, _, _) => Some(ResizeDirection::North),
            (_, true, _, _) => Some(ResizeDirection::South),
            (_, _, true, _) => Some(ResizeDirection::West),
            (_, _, _, true) => Some(ResizeDirection::East),
            _ => None,
        }
    }

    /// Look up the [`WindowAction`] tagged on the currently hovered
    /// node, if any.
    fn hovered_window_action(&self) -> Option<WindowAction> {
        let id = self.input.hovered?;
        self.ctx.tree.get(id).and_then(|n| n.window_action)
    }

    /// Pick the right cursor icon for `(x, y)` and push it to the
    /// window if it changed. Edge gutters win over node hover so the
    /// title bar's drag cursor doesn't fight the corner-resize cursor.
    fn refresh_cursor(&mut self, x: f32, y: f32) {
        let icon = if let Some(dir) = self.edge_at(x, y) {
            CursorIcon::from(dir)
        } else if let Some(action) = self.hovered_window_action() {
            match action {
                WindowAction::DragMove => CursorIcon::Move,
                WindowAction::Close
                | WindowAction::Minimize
                | WindowAction::ToggleMaximize => CursorIcon::Pointer,
            }
        } else {
            CursorIcon::Default
        };
        if icon == self.last_cursor_icon {
            return;
        }
        self.last_cursor_icon = icon;
        if let Some(w) = &self.window {
            w.set_cursor(icon);
        }
    }

    /// Dispatch a window action to winit. Returns true if the action
    /// was a window-drag (so the caller should also skip the normal
    /// press → on_left_pressed path).
    fn dispatch_window_action(
        &mut self,
        action: WindowAction,
        event_loop: &ActiveEventLoop,
    ) -> bool {
        let Some(win) = &self.window else { return false };
        match action {
            WindowAction::DragMove => {
                let _ = win.drag_window();
                true
            }
            WindowAction::Close => {
                event_loop.exit();
                true
            }
            WindowAction::Minimize => {
                win.set_minimized(true);
                true
            }
            WindowAction::ToggleMaximize => {
                win.set_maximized(!win.is_maximized());
                true
            }
        }
    }

    fn save_screenshot(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let (rgba, w, h) = gpu.capture_rgba();
        let path = debug::screenshot_path(&self.config.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("screenshot saved: {}", path.display()),
            Err(e) => log::error!("screenshot failed: {e}"),
        }
        let stats = self
            .last_render_stats
            .unwrap_or_else(|| self.current_stats());
        debug::write_stats_sidecar(&path, &stats);
    }
}

/// Walk an ordered event list, expanding Text/Image events into GPU
/// instances at their declared position. Preserves painter's order
/// across all kinds so glass, text, images and rects can be layered
/// in any order. Returns the count of `SHAPE_KIND_GLASS` instances
/// produced (used by `gpu.set_instances` for stats and the
/// backdrop-pass gate).
fn expand_events_into(
    events: &[FlatEvent],
    out: &mut Vec<ShapeInstance>,
    gpu: &mut crate::gpu::GpuContext,
    text: &mut crate::text::TextResources,
    scale: f32,
) -> u32 {
    use crate::gpu::SHAPE_KIND_GLASS;
    let mut glass_count: u32 = 0;
    for event in events {
        match event {
            FlatEvent::Shape(s) => {
                let mut s = *s;
                if s.shape_kind == SHAPE_KIND_GLASS {
                    glass_count += 1;
                }
                // border_radius/border_width/shadow_offset/shadow_blur
                // are stored in logical px on the style. Position +
                // size already came out of layout in physical px.
                s.border_radius = [
                    s.border_radius[0] * scale,
                    s.border_radius[1] * scale,
                    s.border_radius[2] * scale,
                    s.border_radius[3] * scale,
                ];
                s.border_width *= scale;
                s.shadow_offset = [s.shadow_offset[0] * scale, s.shadow_offset[1] * scale];
                s.shadow_blur *= scale;
                if s.shape_kind == SHAPE_KIND_GLASS {
                    // For glass: x = blur_amount (px), y = refraction (px).
                    // Both authored in logical px, scaled to physical.
                    s.backdrop_uv_rect[0] *= scale;
                    s.backdrop_uv_rect[1] *= scale;
                    s.roughness *= scale;
                }
                out.push(s);
            }
            FlatEvent::Image(r) => {
                let mut r = r.clone();
                r.border_radius = [
                    r.border_radius[0] * scale,
                    r.border_radius[1] * scale,
                    r.border_radius[2] * scale,
                    r.border_radius[3] * scale,
                ];
                let resolved = gpu.build_image_instances(std::slice::from_ref(&r));
                out.extend(resolved);
            }
            FlatEvent::Text(r) => {
                // font_size + line_height in TextRef are logical;
                // shaping needs physical px so the glyph atlas
                // rasterizes at on-screen resolution.
                let mut r = r.clone();
                r.font_size *= scale;
                r.line_height *= scale;
                let glyphs = gpu.build_glyph_instances(text, std::slice::from_ref(&r));
                out.extend(glyphs);
            }
        }
    }
    glass_count
}

/// Numeric-label HUD. One row per metric: "<label> <value>". Uses the
/// glyph atlas on `gpu` so shaping + atlas upload happen here.
fn build_hud_instances(
    stats: &FrameStats,
    gpu: &mut crate::gpu::GpuContext,
    text: &mut crate::text::TextResources,
    scale: f32,
) -> Vec<ShapeInstance> {
    use crate::node::TextRef;

    // All values declared in logical px (matches the scene-graph
    // convention) and multiplied by the display scale before they
    // hit the GPU. Keeps the HUD on-screen footprint identical
    // across 100% / 200% monitors.
    let origin_x = 12.0 * scale;
    let origin_y = 12.0 * scale;
    let pad = 8.0 * scale;
    let line_h = 14.0 * scale;
    let font_size = 12.0 * scale;
    let panel_w = 160.0 * scale;
    let radius = 6.0 * scale;

    let lines: [String; 6] = [
        format!("cpu  {:>5.2} ms", stats.cpu_ms),
        format!("gpu  {:>5.2} ms", stats.gpu_ms),
        format!("opq  {:>5.2} ms", stats.opaque_ms),
        format!("fnl  {:>5.2} ms", stats.final_ms),
        format!("inst {:>5}", stats.instance_count),
        format!("draw {:>5}", stats.drawcalls),
    ];

    let panel_h = pad * 2.0 + line_h * lines.len() as f32;
    let mut out = Vec::with_capacity(1 + lines.len() * 12);

    out.push(ShapeInstance {
        color: [0.0, 0.0, 0.0, 0.6],
        position: [origin_x, origin_y],
        size: [panel_w, panel_h],
        border_radius: [radius; 4],
        ..Default::default()
    });

    let refs: Vec<TextRef> = lines
        .iter()
        .enumerate()
        .map(|(i, s)| TextRef {
            position: [origin_x + pad, origin_y + pad + i as f32 * line_h],
            color: [1.0, 1.0, 1.0, 1.0],
            opacity: 1.0,
            content: s.clone(),
            // HUD font_size + line_height already in physical px so
            // build_glyph_instances ships them straight to the
            // shaper — match that calling convention.
            font_size,
            line_height: line_h,
            clip_rect: crate::gpu::NO_CLIP,
        })
        .collect();
    out.extend(gpu.build_glyph_instances(text, &refs));
    out
}

/// Walk the color bind list. For each slot whose underlying source
/// has bumped its version: read the new target, advance
/// `last_version`, and either snap (`tree.set_color`) or start a
/// tween on the slot's `displayed` signal.
fn process_color_binds(
    slots: &mut [ColorBindSlot],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, slot) in slots.iter_mut().enumerate() {
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation())
        {
            let key = BIND_TWEEN_KEY_COLOR + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_color(slot.node_id, target);
        }
    }
}

fn process_position_binds(
    slots: &mut [PositionBindSlot],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, slot) in slots.iter_mut().enumerate() {
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation())
        {
            let key = BIND_TWEEN_KEY_POSITION + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_layout_pos_abs(slot.node_id, target);
        }
    }
}

fn process_size_binds(
    slots: &mut [SizeBindSlot],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, slot) in slots.iter_mut().enumerate() {
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation())
        {
            let key = BIND_TWEEN_KEY_SIZE + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_layout_size_px(slot.node_id, target);
        }
    }
}

/// Helper passed to a headless script. Bundles every piece of shell
/// state the script might need so it can drive synthetic events,
/// mutate the tree, advance the timeline, and capture frames without
/// reaching back into private fields.
pub struct HeadlessHelper<'a> {
    pub ctx: &'a mut SceneCtx,
    pub gpu: &'a mut GpuContext,
    pub input: &'a mut InputState,
    pub timeline: &'a mut Timeline,
    pub instances: &'a mut Vec<ShapeInstance>,
    pub glass_count: &'a mut u32,
    pub flat_events: &'a mut Vec<FlatEvent>,
    pub hits: &'a mut Vec<HitEntry>,
    pub scroll_hits: &'a mut Vec<crate::node::ScrollHit>,
    pub scroll_bars: &'a mut Vec<ScrollbarHit>,
    pub capture_dir: &'a Path,
    /// Stats snapshot taken at end of `render()`. `capture()` reads
    /// from here so the sidecar describes the live render frame, not
    /// the re-encode that `capture_rgba` performs.
    pub last_render_stats: &'a mut Option<FrameStats>,
    /// Display DPI scale factor — applied to layout + text the same
    /// way as `App` does. Defaults to `1.0` for headless captures
    /// (no real window). Override before calling `flush` if a
    /// scripted scenario needs to test scaled rendering.
    pub scale_factor: f32,
}

impl<'a> HeadlessHelper<'a> {
    /// Re-flatten + upload if the tree is dirty. Returns true if it
    /// actually uploaded (matches `App::flush_tree`).
    pub fn flush(&mut self) -> bool {
        let mask = self.ctx.tree.take_dirty();
        if mask == 0 {
            return false;
        }
        if mask & (crate::node::dirty::TREE | crate::node::dirty::TRANSFORM) != 0 {
            let vp = [
                self.gpu.surface_config.width as f32,
                self.gpu.surface_config.height as f32,
            ];
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                vp,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        self.ctx.tree.flatten_into_buffers(
            self.scale_factor,
            self.flat_events,
            self.hits,
            self.scroll_hits,
            self.scroll_bars,
        );
        self.instances.clear();
        *self.glass_count = expand_events_into(
            self.flat_events,
            self.instances,
            self.gpu,
            &mut self.ctx.text,
            self.scale_factor,
        );
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        self.gpu
            .set_instances(self.instances, *self.glass_count, backdrop_hint);
        true
    }

    pub fn render(&mut self) {
        self.gpu.render_frame();
        // Snapshot the render-frame stats before any subsequent
        // `capture()` re-encodes and clears `backdrop_dirty`.
        let mut snap = self.gpu.last_frame_stats();
        snap.dirty_mask = 0;
        *self.last_render_stats = Some(snap);
    }

    pub fn capture(&mut self) {
        let (rgba, w, h) = self.gpu.capture_rgba();
        let path = debug::screenshot_path(self.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("auto-capture saved: {}", path.display()),
            Err(e) => log::error!("auto-capture failed: {e}"),
        }
        let stats = self.last_render_stats.unwrap_or_else(|| {
            let mut s = self.gpu.last_frame_stats();
            s.dirty_mask = 0;
            s
        });
        debug::write_stats_sidecar(&path, &stats);
    }

    /// Fast-forward every active tween to its target, push the
    /// settled values through the bind registry, and clear the
    /// timeline. Useful after a scripted state change so the next
    /// capture reflects the destination value rather than the
    /// pre-tween value.
    pub fn settle(&mut self) {
        if !self.timeline.active() {
            return;
        }
        // One tick well past any plausible duration snaps every
        // tween's signal to its `to` target.
        let _ = self
            .timeline
            .tick(Instant::now() + std::time::Duration::from_secs(10));
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.position {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.size {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_size_px(slot.node_id, disp.get());
            }
        }
    }

    /// Build + upload the numeric-label HUD overlay from the
    /// renderer's most recent frame stats.
    pub fn show_hud(&mut self) {
        let stats = self.gpu.last_frame_stats();
        let inst = build_hud_instances(&stats, self.gpu, &mut self.ctx.text, self.scale_factor);
        self.gpu.set_overlay_instances(&inst);
    }

    /// Clear any active HUD overlay.
    pub fn hide_hud(&mut self) {
        self.gpu.set_overlay_instances(&[]);
    }

    /// Run reactive bind processing + animated display pump on the
    /// shared registry. Mirrors `App::react` for scripted captures.
    pub fn react(&mut self, now: Instant) {
        process_color_binds(
            &mut self.ctx.binds.color,
            &mut self.ctx.tree,
            self.timeline,
            now,
        );
        process_position_binds(
            &mut self.ctx.binds.position,
            &mut self.ctx.tree,
            self.timeline,
            now,
        );
        process_size_binds(
            &mut self.ctx.binds.size,
            &mut self.ctx.tree,
            self.timeline,
            now,
        );
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.position {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in &self.ctx.binds.size {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_size_px(slot.node_id, disp.get());
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Wait);

        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_transparent(self.config.transparent)
            .with_decorations(self.config.decorations)
            .with_resizable(true)
            .with_blur(self.config.blur)
            .with_visible(false)
            // Logical size — winit converts to physical based on the
            // monitor's scale factor, so the same `AppConfig::new(w, h)`
            // produces the same on-screen footprint on a 100% 1080p
            // display and a 200% 4k display.
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width as f64,
                self.config.height as f64,
            ));
        let window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_arc: Arc<Window> = Arc::new(window);
        self.scale_factor = window_arc.scale_factor() as f32;
        // Mirror the scale onto the tree so scroll setters can convert
        // logical-px configuration (snap_step, overscroll limit) on the
        // fly without threading the factor through every public method.
        self.ctx.tree.set_scale(self.scale_factor);
        // Seed logical_size from the actual window — winit may have
        // adjusted to fit the monitor or the minimum chrome size.
        let phys = window_arc.inner_size();
        let s = self.scale_factor.max(f32::EPSILON);
        self.logical_size = [
            (phys.width as f32 / s).round() as u32,
            (phys.height as f32 / s).round() as u32,
        ];
        self.gpu = Some(GpuContext::new(Arc::clone(&window_arc)));
        self.window = Some(window_arc);
        log::info!("display scale: {}", self.scale_factor);

        // Drain staged PNGs into the image atlas. The atlas allocates
        // handles sequentially from 0, matching the indices we handed
        // back from `stage_image_png`.
        if !self.staged_images.is_empty() {
            let gpu = self.gpu.as_mut().expect("gpu just initialized");
            let staged = std::mem::take(&mut self.staged_images);
            for (idx, image) in staged.into_iter().enumerate() {
                let ok = match image {
                    StagedImage::Png(bytes) => gpu.upload_image_png(&bytes).is_some(),
                    StagedImage::Rgba { w, h, bytes } => {
                        gpu.image_atlas.upload_rgba(&gpu.queue, w, h, &bytes).is_some()
                    }
                };
                if !ok {
                    log::warn!("staged image #{idx}: decode/upload failed");
                }
            }
        }

        if let Some(mem) = self.memory_report() {
            log::info!(
                "gpu memory: total={} KiB (instance={} overlay={} blur={} overdraw={} glyph_atlas={} image_atlas={} timing={} prev_cpu={})",
                mem.total() / 1024,
                mem.instance_buffer,
                mem.overlay_buffer,
                mem.blur_textures,
                mem.overdraw_textures,
                mem.glyph_atlas,
                mem.image_atlas,
                mem.timing,
                mem.prev_instances_cpu,
            );
        }

        // First flush + render so the visible window already shows
        // the scene by the time the user sees it.
        self.flush_tree();
        self.render_once();
        if let Some(w) = &self.window {
            w.set_visible(true);
            w.request_redraw();
        }

        if let Some(n) = self.capture_frames {
            // First frame is always written before the script (if any).
            self.save_screenshot();
            // Additional N-1 still frames for plain `.capture(N, ...)`
            // mode. Skipped when a headless script is also attached
            // because the script is responsible for its own captures.
            if self.headless.is_none() {
                for _ in 1..n {
                    self.render_once();
                    self.save_screenshot();
                }
            }
        }

        if let Some(script) = self.headless.take() {
            let mut helper = HeadlessHelper {
                ctx: &mut self.ctx,
                gpu: self.gpu.as_mut().expect("gpu"),
                input: &mut self.input,
                timeline: &mut self.timeline,
                instances: &mut self.instances,
                glass_count: &mut self.glass_count,
                flat_events: &mut self.flat_events,
                hits: &mut self.hits,
                scroll_hits: &mut self.scroll_hits,
                scroll_bars: &mut self.scroll_bars,
                capture_dir: &self.config.capture_dir,
                last_render_stats: &mut self.last_render_stats,
                scale_factor: self.scale_factor,
            };
            script(&mut helper);
            event_loop.exit();
            return;
        }

        if self.capture_frames.is_some() {
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(_gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(size.width, size.height);
                }
                let in_dpi_window = self
                    .last_dpi_change
                    .map(|t| t.elapsed() < Duration::from_millis(500))
                    .unwrap_or(false);
                let want_w = (self.logical_size[0] as f32 * self.scale_factor)
                    .round() as u32;
                let want_h = (self.logical_size[1] as f32 * self.scale_factor)
                    .round() as u32;
                if in_dpi_window {
                    // DPI-triggered resize. Keep logical_size; Win11
                    // 24H2 may deliver stale physical here.
                    let mismatch = size.width != want_w || size.height != want_h;
                    if mismatch && let Some(w) = &self.window {
                        log::warn!(
                            "DPI Resized {}x{} ≠ logical*scale {}x{}, forcing back (winit#4041)",
                            size.width,
                            size.height,
                            want_w,
                            want_h
                        );
                        let _ = w.request_inner_size(winit::dpi::PhysicalSize::new(
                            want_w, want_h,
                        ));
                    }
                    // Cancel winit's cursor-anchored reposition —
                    // restore pre-SF outer so the window stays under
                    // the cursor physically (matches browser
                    // behavior).
                    if let Some(p) = self.pre_dpi_outer.take()
                        && let Some(w) = &self.window
                        && w.outer_position().ok() != Some(p)
                    {
                        w.set_outer_position(p);
                    }
                } else if size.width != want_w || size.height != want_h {
                    // User-driven resize (drag handle / snap / max).
                    let s = self.scale_factor.max(f32::EPSILON);
                    self.logical_size = [
                        (size.width as f32 / s).round() as u32,
                        (size.height as f32 / s).round() as u32,
                    ];
                }
                // Viewport size feeds layout (Fill/Pct/Fr + glass
                // backdrop region). Reflow + re-upload at the new size.
                self.ctx.tree.mark_all_dirty();
                self.flush_tree();
                self.request_redraw();
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer: _,
            } => {
                let new_scale = scale_factor as f32;
                if (new_scale - self.scale_factor).abs() > f32::EPSILON {
                    log::debug!(
                        "scale factor: {} → {}",
                        self.scale_factor,
                        new_scale
                    );
                    if let Some(w) = &self.window
                        && let Ok(p) = w.outer_position()
                    {
                        self.pre_dpi_outer = Some(p);
                    }
                    self.last_dpi_change = Some(Instant::now());
                    self.scale_factor = new_scale;
                    self.ctx.tree.set_scale(new_scale);
                    if let Some(g) = self.gpu.as_mut() {
                        g.reset_glyph_atlas();
                    }
                    // Shape cache keys include physical font_size; old
                    // entries would never match again under the new
                    // scale so drop them to keep the table small.
                    self.ctx.text.clear_shape_cache();
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_once();
            }
            WindowEvent::Occluded(occluded) => {
                if !occluded {
                    self.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                self.input.cursor = Some([x, y]);
                if self.bar_drag.is_some() {
                    // Coalesce: don't apply per OS event. Stash the
                    // latest cursor and let `about_to_wait` push it
                    // through at frame rate. Skips hover refresh + the
                    // hit-test path during drag (cursor is captured by
                    // the thumb anyway). Ensure the loop wakes up so
                    // the pending cursor gets applied promptly.
                    self.pending_drag_cursor = Some([x, y]);
                    self.request_redraw();
                    return;
                }
                let bar_changed = crate::input::update_scrollbar_hover(
                    Some([x, y]),
                    &self.scroll_bars,
                    &mut self.ctx.tree,
                );
                let change =
                    self.input
                        .on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
                self.refresh_cursor(x, y);
                if change.any() || bar_changed {
                    self.react();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                let bar_changed = crate::input::update_scrollbar_hover(
                    None,
                    &self.scroll_bars,
                    &mut self.ctx.tree,
                );
                let change = self.input.on_cursor_left(&self.hits, &self.ctx.tree);
                if self.last_cursor_icon != CursorIcon::Default {
                    self.last_cursor_icon = CursorIcon::Default;
                    if let Some(w) = &self.window {
                        w.set_cursor(CursorIcon::Default);
                    }
                }
                if change.any() || bar_changed {
                    self.react();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods;
            }
            WindowEvent::Focused(false) => {
                // Window lost focus — winit doesn't deliver Released
                // events for keys still down, so the held set would
                // leak and suppress settle forever. Clear it now.
                self.held_scroll_keys.clear();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let Some([cx, cy]) = self.input.cursor else {
                    return;
                };
                // Convert winit's delta to pixel scroll-target delta.
                // Wheel forward (positive y) = scroll content *up* =
                // target.y decreases, hence the sign flip on both axes.
                // LineDelta uses 50 logical-px-per-line (typical desktop
                // line height) scaled by the display factor.
                let px = match delta {
                    winit::event::MouseScrollDelta::LineDelta(lx, ly) => {
                        let line = 50.0 * self.scale_factor;
                        [-lx * line, -ly * line]
                    }
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        [-p.x as f32, -p.y as f32]
                    }
                };
                let shift = self.modifiers.state().shift_key();
                if crate::input::on_wheel(
                    [cx, cy],
                    px,
                    &self.scroll_hits,
                    &mut self.ctx.tree,
                    shift,
                ) {
                    self.react();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    if let Some([cx, cy]) = self.input.cursor
                        && let Some(dir) = self.edge_at(cx, cy) {
                            if let Some(w) = &self.window {
                                let _ = w.drag_resize_window(dir);
                            }
                            return;
                        }
                    // Scrollbar layer wins over normal hit-test if the
                    // cursor is on a visible bar.
                    if let Some([cx, cy]) = self.input.cursor {
                        match crate::input::press_scrollbar(
                            [cx, cy],
                            &self.scroll_bars,
                            &mut self.ctx.tree,
                        ) {
                            crate::input::ScrollbarPress::StartDrag {
                                node_id,
                                axis,
                                pointer_origin,
                                scroll_origin,
                                track_travel,
                                max_offset,
                            } => {
                                self.bar_drag = Some(BarDrag {
                                    node_id,
                                    axis,
                                    pointer_origin,
                                    scroll_origin,
                                    track_travel,
                                    max_offset,
                                });
                                self.react();
                                return;
                            }
                            crate::input::ScrollbarPress::JumpedToPosition => {
                                self.react();
                                return;
                            }
                            crate::input::ScrollbarPress::Miss => {}
                        }
                    }
                    if let Some(action) = self.hovered_window_action()
                        && self.dispatch_window_action(action, event_loop) {
                            return;
                        }
                }
                let change = match state {
                    ElementState::Pressed => {
                        self.input.on_left_pressed(&self.hits, &self.ctx.tree)
                    }
                    ElementState::Released => {
                        // End any in-flight thumb drag. Apply the last
                        // pending cursor so the final pixel lands.
                        if let Some(d) = self.bar_drag.take() {
                            if let Some(c) = self.pending_drag_cursor.take() {
                                let _ = crate::input::drag_to(
                                    c,
                                    d.node_id,
                                    d.axis,
                                    d.pointer_origin,
                                    d.scroll_origin,
                                    d.track_travel,
                                    d.max_offset,
                                    &mut self.ctx.tree,
                                );
                            }
                            crate::input::end_drag(d.node_id, d.axis, &mut self.ctx.tree);
                        }
                        self.input.on_left_released(&self.hits, &self.ctx.tree)
                    }
                };
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                if state == ElementState::Pressed {
                    // Hotkeys fire on initial press only — holding F2
                    // shouldn't spam screenshots, holding Esc is just
                    // weird. Auto-repeat events are filtered here.
                    if !repeat {
                        match code {
                            KeyCode::Escape => {
                                event_loop.exit();
                                return;
                            }
                            KeyCode::F1 => {
                                self.hud_enabled = !self.hud_enabled;
                                self.stats_log = self.hud_enabled;
                                log::info!(
                                    "hud/stats: {} | last frame: {:?}",
                                    if self.hud_enabled { "on" } else { "off" },
                                    self.current_stats()
                                );
                                if !self.hud_enabled {
                                    self.clear_hud_overlay();
                                }
                                self.request_redraw();
                                return;
                            }
                            KeyCode::F2 => {
                                self.save_screenshot();
                                return;
                            }
                            KeyCode::F4 => {
                                if let Some(g) = self.gpu.as_mut() {
                                    let on = !g.overdraw_mode();
                                    g.set_overdraw(on);
                                    log::info!(
                                        "overdraw heatmap: {}",
                                        if on { "on" } else { "off" }
                                    );
                                }
                                self.request_redraw();
                                return;
                            }
                            KeyCode::F5 => {
                                self.ctx.tree.mark_all_dirty();
                                if self.flush_tree() {
                                    self.request_redraw();
                                }
                                return;
                            }
                            _ => {}
                        }
                    }
                    match code {
                        KeyCode::ArrowUp
                        | KeyCode::ArrowDown
                        | KeyCode::ArrowLeft
                        | KeyCode::ArrowRight => {
                            // Arrow keys: ignore OS auto-repeat and
                            // drive continuous scroll from `about_to_wait`
                            // via `pump_held_scroll`. OS auto-repeat has
                            // a ~250 ms initial-delay gap; relying on it
                            // makes the spring visibly settle at the
                            // first event's target then jump to the next
                            // when repeat finally fires. Per-tick pump
                            // gives smooth target progression.
                            if !repeat {
                                self.held_scroll_keys.insert(code);
                                let viewport = self.viewport();
                                if crate::input::on_scroll_key(
                                    code,
                                    self.input.cursor,
                                    viewport,
                                    self.scale_factor,
                                    &self.scroll_hits,
                                    &mut self.ctx.tree,
                                ) {
                                    self.react();
                                }
                            }
                            return;
                        }
                        KeyCode::PageUp
                        | KeyCode::PageDown
                        | KeyCode::Home
                        | KeyCode::End => {
                            // Page / Home / End honour OS auto-repeat —
                            // they're jump keys and `set_scroll_target`
                            // is idempotent at the clamps, so holding is
                            // harmless. Track the held set so the input
                            // quiescence gate stays suppressed.
                            if !repeat {
                                self.held_scroll_keys.insert(code);
                            }
                            let viewport = self.viewport();
                            if crate::input::on_scroll_key(
                                code,
                                self.input.cursor,
                                viewport,
                                self.scale_factor,
                                &self.scroll_hits,
                                &mut self.ctx.tree,
                            ) {
                                self.react();
                                return;
                            }
                        }
                        _ => {}
                    }
                } else if state == ElementState::Released {
                    self.held_scroll_keys.remove(&code);
                }
                // User on_key hook: fire on initial press only so
                // application code doesn't have to special-case repeat.
                if !repeat && let Some(handler) = self.on_key.as_mut() {
                    handler(code, state, &mut self.ctx);
                    self.react();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Animation + scroll pump. If both are idle, park on `Wait` so
        // the loop is 0% CPU. Otherwise: advance both, push interpolated
        // values through the bind registry, flush, redraw, and schedule
        // the next deadline.
        let now = Instant::now();
        // Drain any coalesced thumb-drag cursor: a single
        // set_scroll_immediate per frame, regardless of how many
        // CursorMoved events fired since the last tick.
        let drag_moved = if let (Some(d), Some(c)) = (self.bar_drag, self.pending_drag_cursor) {
            self.pending_drag_cursor = None;
            crate::input::drag_to(
                c,
                d.node_id,
                d.axis,
                d.pointer_origin,
                d.scroll_origin,
                d.track_travel,
                d.max_offset,
                &mut self.ctx.tree,
            )
        } else {
            false
        };
        // Hold-key poke: while any scroll key is physically held, keep
        // the input-quiescence gate suppressed so settle doesn't fire
        // during the OS auto-repeat initial-delay window.
        if !self.held_scroll_keys.is_empty() {
            self.ctx.tree.poke_scroll_input_recency();
        }
        let timeline_active = self.timeline.active();
        let mut scroll_active = self.ctx.tree.has_active_scrolls();
        // Hold-arrow continuous pump: replaces OS auto-repeat for the
        // arrow keys. Need a non-zero `dt` to compute the per-tick
        // delta — gate on a meaningful tick interval.
        let hold_dt = match self.last_scroll_tick {
            Some(prev) => (now - prev).as_secs_f32().min(0.05),
            None => 1.0 / 60.0,
        };
        let pumped = crate::input::pump_held_scroll(
            &self.held_scroll_keys,
            self.input.cursor,
            &self.scroll_hits,
            &mut self.ctx.tree,
            self.scale_factor,
            hold_dt,
        );
        if pumped {
            scroll_active = true;
        }
        if !timeline_active && !scroll_active && !drag_moved {
            self.last_scroll_tick = None;
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let dt = match self.last_scroll_tick {
            Some(prev) => (now - prev).as_secs_f32().min(0.05),
            None => 0.0,
        };
        self.last_scroll_tick = Some(now);
        let scroll_moved = self.ctx.tree.tick_scrolls(dt);
        let res = self.timeline.tick(now);
        if res.updated || scroll_moved || drag_moved {
            if res.updated {
                self.pump_animated_displays();
            }
            if self.flush_tree() {
                self.request_redraw();
            }
        }
        let next_scroll_deadline = if self.ctx.tree.has_active_scrolls() || self.bar_drag.is_some()
        {
            Some(now + std::time::Duration::from_millis(16))
        } else {
            None
        };
        let combined = match (res.next_deadline, next_scroll_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        match combined {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

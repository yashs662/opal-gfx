//! Application shell — wraps winit + wgpu + scene + binds + input + anim.
//!
//! Use it like:
//!
//! ```ignore
//! opal_gfx::app::App::new("demo", 1100, 750)
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
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowId};

use crate::anim::Timeline;
use crate::debug;
use crate::gpu::{FrameStats, GpuContext, ImageHandle, ShapeInstance};
use crate::input::InputState;
use crate::node::{FlatEvent, HitEntry, ScrollAxis, ScrollHit, ScrollbarHit, WindowAction};
use crate::scene::{
    ColorBindSlot, HeightPxBindSlot, ImageBindSlot, OpacityBindSlot, PositionBindSlot, Scene,
    SceneCtx, SizeBindSlot, TextBindSlot, WidthPctBindSlot, WidthPxBindSlot,
};

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
const BIND_TWEEN_KEY_OPACITY: u32 = 0xC300_0000;
/// Dedicated tween key for the drag-follow snap-back animation. Lives
/// above the bind-tween windows so it can never alias a slot tween.
const DRAG_RETURN_TWEEN_KEY: u32 = 0xD000_0000;
/// Snap-back duration when a dragged node returns to its resting slot.
const DRAG_RETURN_MS: u64 = 160;
/// Hover delay before a `hover_hint` tooltip starts fading in (the dwell
/// window when a node has a hint but no explicit `on_hover_dwell` duration).
/// Short + paired with a fade so it reveals quickly without popping.
const HINT_DWELL: Duration = Duration::from_millis(110);
/// Fade in / out duration for a `hover_hint` tooltip.
const HINT_FADE: Duration = Duration::from_millis(120);
/// Debounce window for re-deriving hover when the scene changes under a
/// stationary cursor (scroll, a slider morphing content, an async update).
/// Picked so a continuous change re-checks hover ~10×/s (no per-frame thrash)
/// and the last-armed check lands shortly after movement stops.
const HOVER_RECHECK: Duration = Duration::from_millis(90);

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub decorations: bool,
    pub transparent: bool,
    pub blur: bool,
    pub capture_dir: PathBuf,
    /// Window-corner radius in logical px applied by the final shader.
    /// `0.0` (default) = square corners; non-zero rounds the surface
    /// against the transparent winit window. Automatically suppressed
    /// when the window is maximised / fullscreen since edge rounding
    /// against the work-area boundary just clips usable pixels.
    pub window_corner_radius: f32,
    /// Taskbar / alt-tab window icon as `(width, height, rgba8)`. `None`
    /// (default) leaves winit to fall back to the embedded executable
    /// icon. Set via [`App::window_icon_rgba`].
    pub window_icon: Option<(u32, u32, Vec<u8>)>,
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
            window_corner_radius: 0.0,
            window_icon: None,
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

/// Optional per-frame hook. Fires once per `about_to_wait` iteration
/// — i.e. on every wake-up from input, animation tick, or scheduled
/// wait. Use for driving custom animations that read from `Signal`s
/// or mutate the tree on a schedule (e.g. interpolating a lazy-list
/// row height between click-to-expand and click-to-collapse). Gets
/// access to the timeline so it can poll active tweens or start new
/// ones.
pub type FrameFn = Box<dyn FnMut(&mut SceneCtx, &mut crate::anim::Timeline, Instant)>;

/// Optional outside-press hook. Fires on a left-button
/// **press** that lands on no interactive node — or on a node tagged
/// [`crate::node::Node::dismiss_transparent`] (a modal / context-menu
/// scrim that blocks click-through yet should still dismiss). The
/// canonical dismiss path for floating layers: the handler flips the
/// layer's visibility `Signal` (+ rebuild token) and the press never
/// reaches whatever is behind. After it returns the shell reacts.
pub type UnhandledPressFn = Box<dyn FnMut(&mut SceneCtx)>;

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
    /// Force-promoted (`.layer()`) subtree spans from the last flatten,
    /// in painter order (event-index ranges). Mapped to instance ranges
    /// and fed to the `LayerTree` in `flush_tree`.
    flat_spans: Vec<crate::node::LayerSpan>,
    /// Per-event instance-offset prefix from the last `expand_events_into`
    /// (`event_inst_start`). Reused across frames; maps `flat_spans`
    /// event ranges to instance ranges.
    flat_event_inst: Vec<u32>,
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
    /// Cursor pending an `on_hover_move` dispatch, coalesced to frame rate
    /// for the same reason as [`Self::pending_drag_cursor`]: a hover-move
    /// handler typically sets a Signal (e.g. the seek-bar timestamp preview)
    /// that dirties the tree, and firing per OS CursorMoved (500+ Hz) would
    /// re-flatten the whole instance buffer many times per displayed frame.
    /// Only set when the hovered node actually has a hover-move handler.
    pending_hover_cursor: Option<[f32; 2]>,
    /// Generic-drag bookkeeping. `drag_origin` is the
    /// cursor at press on a draggable node; `drag_last` is the cursor at
    /// the previous `on_drag` fire (so `DragCtx::delta` is per-event).
    /// Both clear on release / capture loss.
    drag_origin: Option<[f32; 2]>,
    drag_last: Option<[f32; 2]>,
    /// The node whose `on_drag` captured the current drag, remembered so
    /// `on_drag_end` can fire for it on release (capture is cleared first).
    drag_node: Option<crate::node::NodeId>,
    /// In-flight drag-and-drop payload. Latched from the pressed node's
    /// `drag_payload`; delivered to a drop target on release.
    drag_payload: Option<std::rc::Rc<dyn std::any::Any>>,
    /// Node currently animating back to its resting slot after a
    /// `drag_follow` release. The timeline tweens `drag_return_offset`
    /// toward `[0,0]`; `tick_drag_return` mirrors it into the tree's
    /// drag-follow offset each frame and clears the lift when it lands.
    drag_return: Option<crate::node::NodeId>,
    drag_return_offset: crate::signal::Signal<[f32; 2]>,
    /// System clipboard handle, lazily created on first copy / cut /
    /// paste. `None` if arboard init failed (headless / no display) — the
    /// editor still works, clipboard ops just no-op.
    clipboard: Option<arboard::Clipboard>,
    /// Caret-blink phase anchor for the currently-focused text field. Reset
    /// on focus + on every edit so the caret shows solid immediately after
    /// interaction, then blinks. Only consulted while `input.focused` is a
    /// text field.
    caret_blink_anchor: Instant,
    /// Last-applied caret-on state, so the blink only writes opacity (and
    /// flushes) on the half-second it actually toggles.
    caret_blink_on: bool,
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
    on_frame: Option<FrameFn>,
    on_unhandled_press: Option<UnhandledPressFn>,
    /// One-shot hook fired from `exiting()` (winit's last chance before
    /// the event loop tears down). Use for save-on-quit work — prefs
    /// flush, session disconnect, etc. `Option` so it can be `take`n
    /// (`FnOnce`-style) on first fire.
    on_exit: Option<Box<dyn FnOnce()>>,
    headless: Option<HeadlessFn>,
    /// Number of still frames to capture and exit after, or `None` for
    /// a normal interactive run. `Some(n)` triggers headless capture
    /// mode in `resumed`.
    capture_frames: Option<u32>,
    /// Optional delay before the autocapture fires. `None` = capture on
    /// the first rendered frame (legacy behaviour). `Some(d)` lets the
    /// event loop run normally for `d`, then captures + exits. Useful
    /// when the UI needs async data (worker fetches, image uploads) to
    /// land before the snapshot makes sense.
    capture_delay: Option<Duration>,
    /// Set in `resumed` when `capture_delay` is `Some`. `about_to_wait`
    /// polls against this; once `now >= deadline` it triggers the
    /// pending capture + exits the loop.
    capture_deadline: Option<Instant>,
    /// Background thread that owns the PNG encode + disk write from the
    /// most recent screenshot. Held so exit paths can join before the
    /// process tears down — otherwise an `event_loop.exit()` triggered
    /// shortly after F2 / autocapture would interrupt the encoder
    /// mid-write and leave a truncated file behind.
    pending_screenshot: Option<std::thread::JoinHandle<()>>,
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
    /// Retained layer tree (compositor). Rebuilt beside the node tree in
    /// `flush_tree`; partitions the instance stream into root segments +
    /// `.layer()`-promoted layers and owns their persistent composite
    /// transform/opacity. The GPU reports actual raster/composite/VRAM
    /// counts into `FrameStats`.
    layer_tree: crate::layer::LayerTree,
    /// Layer composite-opacity bindings: `(promoted node, source signal)`.
    /// Collected each `flush_tree` from `.layer_opacity(sig)` nodes and
    /// pumped each awake frame after the timeline tick — the signal value
    /// is pushed into the layer's composite opacity via
    /// [`Self::set_layer_opacity`] (composite-only, no re-raster). The
    /// bridge from a `Signal<f32>` (e.g. a crossfade tween) to a layer's
    /// composite opacity, mirroring how reactive binds drive node props.
    layer_opacity_binds: Vec<(crate::node::NodeId, crate::signal::Signal<f32>)>,
    /// Same bridge as [`Self::layer_opacity_binds`] but for a layer's
    /// composite **X offset** (logical px) — collected from `.layer_offset_x`
    /// nodes each flush and pumped each awake frame via
    /// [`Self::set_layer_offset`]. Drives cursor-following overlays without
    /// touching the layout tree.
    layer_offset_x_binds: Vec<(crate::node::NodeId, crate::signal::Signal<f32>)>,
    /// Reused scratch for [`Self::push_layer_draws`] so composite-only
    /// updates don't allocate a fresh draw-list Vec each frame.
    layer_draw_scratch: Vec<crate::gpu::LayerDraw>,
    /// Stat-dump cadence: continuously log on every render when set.
    /// Toggled by `OPAL_STATS=1` env var or by F1 in interactive mode.
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
    /// Stored scene builder. `App::scene` runs the closure once
    /// immediately and stashes it here so [`App::rebuild_scene`] can
    /// re-invoke it after clearing the tree — the canonical way to
    /// swap views (Library ⇄ Search) without leaking bind slots or
    /// timeline tweens. `FnMut` rather than `FnOnce` because rebuilds
    /// may fire many times over the app's lifetime; captured state
    /// must be re-usable.
    scene_builder: Option<Box<dyn FnMut(&mut Scene)>>,
    /// Cooperative rebuild signal. Closures (e.g. `on_click` handlers)
    /// can't reach `&mut App` to call `rebuild_scene` directly — they
    /// own a clone of this `Rc<Cell<bool>>` via [`App::rebuild_token`]
    /// and flip it to `true`. The shell consumes the flag at the top
    /// of `about_to_wait` and triggers the rebuild before the next
    /// frame. Cleared atomically via `replace(false)` so a handler
    /// firing during the rebuild itself queues a fresh one.
    rebuild_request: std::rc::Rc<std::cell::Cell<bool>>,
    /// Hover-dwell tracker. At most one node is hovered (input
    /// pin), so a single slot is enough. `node` is the currently-armed
    /// dwell target, `deadline` is when its handler should fire,
    /// `fired` flips true once the handler has run so we don't refire
    /// from cursor jitter inside the same node.
    dwell: Option<DwellTracker>,
    /// Node whose `hover_hint` tooltip is currently shown (rendered into the
    /// overlay buffer). `None` when no hint is up. Set when a hint node's
    /// dwell fires, cleared on hover-leave / rebuild.
    active_hint: Option<crate::node::NodeId>,
    /// A cursor move happened while a hint is shown — repaint it at the new
    /// position once this frame (coalesced; OS-cursor moves fire at 500+ Hz).
    pending_hint_repaint: bool,
    /// Tooltip opacity (0..1), tweened on the timeline: fades in when a hint
    /// appears, fades out when it leaves. Read by `paint_hint`.
    hint_fade: crate::signal::Signal<f32>,
    /// Whether the active hint is fading *in* (`true`, pointer on its node)
    /// or *out* (`false`, pointer left — finalised once `hint_fade` hits 0).
    hint_visible: bool,
    /// Debounced deadline to re-derive hover after the scene changed under a
    /// stationary cursor. Armed (no-reset) by content-changing flushes; fires
    /// in `about_to_wait`. See `arm_hover_recheck` / `tick_hover_recheck`.
    next_hover_recheck: Option<Instant>,
    /// External-wake plumbing for the case where the event loop is
    /// parked on `Wait` (no animation / scroll / dwell) and another
    /// thread (e.g. a worker delivering an async result) needs the UI
    /// to tick `on_frame` so it can drain a response channel. The
    /// `flag` is set by [`WakeHandle::wake`]; `about_to_wait` reads
    /// (and clears) it at the top of every iteration. The proxy wakes
    /// winit so we *get* to the next `about_to_wait` instead of
    /// sleeping until input.
    wake: Arc<WakeHandle>,
    /// Cross-thread image upload queue. Drained on `about_to_wait`.
    /// Shares the `wake` handle so an enqueue from a worker thread also
    /// wakes the parked event loop.
    uploader: Arc<crate::uploader::Uploader>,
    /// Cross-thread external-frame (video) submission queue. Drained on
    /// `about_to_wait`. Shares the `wake` handle so a decoder thread's
    /// frame submission also wakes the parked event loop.
    frame_sink: Arc<crate::frame_sink::FrameSink>,
    // Lazy:
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    /// Debug-only scripted-input driver (REMOVABLE — `automation` feature).
    #[cfg(feature = "automation")]
    automation: Option<crate::automation::AutomationState>,
}

/// External wake-up token. Clone into any thread that needs to nudge
/// the UI event loop out of `ControlFlow::Wait` — e.g. a worker that
/// just delivered a response on an `mpsc::Receiver` the UI thread
/// polls in [`App::on_frame`]. Calling [`WakeHandle::wake`] guarantees
/// the next `about_to_wait` tick fires the on-frame hook.
pub struct WakeHandle {
    flag: std::sync::atomic::AtomicBool,
    /// Set by `App::run` after the event loop is constructed. `OnceLock`
    /// keeps the no-proxy-yet window safe (e.g. if a worker fires a
    /// wake before `run` was called — the flag still flips, the next
    /// `about_to_wait` after run starts will observe it).
    proxy: std::sync::OnceLock<EventLoopProxy<()>>,
}

impl WakeHandle {
    fn new() -> Self {
        Self {
            flag: std::sync::atomic::AtomicBool::new(false),
            proxy: std::sync::OnceLock::new(),
        }
    }
    /// Flip the wake flag and (if `App::run` has begun) nudge winit so
    /// the loop returns from `Wait` to fire `about_to_wait`. Safe to
    /// call from any thread. No-op when the proxy isn't installed yet —
    /// the flag still latches so the next tick observes it.
    pub fn wake(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Release);
        if let Some(p) = self.proxy.get() {
            // Send a no-op user event; the empty payload is enough to
            // wake the loop. If the receiver is gone (window closing),
            // the error is harmless.
            let _ = p.send_event(());
        }
    }
    /// Atomically read + clear the flag.
    fn take(&self) -> bool {
        self.flag.swap(false, std::sync::atomic::Ordering::AcqRel)
    }
}

#[derive(Debug, Clone, Copy)]
struct DwellTracker {
    node: crate::node::NodeId,
    deadline: Instant,
    fired: bool,
}

impl App {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        let wake = Arc::new(WakeHandle::new());
        Self {
            config: AppConfig::new(title, width, height),
            ctx: SceneCtx::new(),
            instances: Vec::new(),
            glass_count: 0,
            flat_events: Vec::new(),
            hits: Vec::new(),
            scroll_hits: Vec::new(),
            scroll_bars: Vec::new(),
            flat_spans: Vec::new(),
            flat_event_inst: Vec::new(),
            bar_drag: None,
            pending_drag_cursor: None,
            pending_hover_cursor: None,
            drag_origin: None,
            drag_last: None,
            drag_node: None,
            drag_payload: None,
            drag_return: None,
            drag_return_offset: crate::signal::Signal::new([0.0, 0.0]),
            clipboard: None,
            caret_blink_anchor: Instant::now(),
            caret_blink_on: true,
            modifiers: winit::event::Modifiers::default(),
            held_scroll_keys: std::collections::HashSet::new(),
            input: InputState::new(),
            timeline: Timeline::new(),
            on_key: None,
            on_frame: None,
            on_unhandled_press: None,
            on_exit: None,
            headless: None,
            capture_frames: None,
            capture_delay: None,
            capture_deadline: None,
            pending_screenshot: None,
            last_dirty_mask: 0,
            layer_tree: crate::layer::LayerTree::single_root(),
            layer_opacity_binds: Vec::new(),
            layer_offset_x_binds: Vec::new(),
            layer_draw_scratch: Vec::new(),
            last_cpu_ms: 0.0,
            last_render_stats: None,
            stats_log: std::env::var_os("OPAL_STATS").is_some(),
            hud_enabled: std::env::var_os("OPAL_HUD").is_some(),
            last_cursor_icon: CursorIcon::Default,
            scale_factor: 1.0,
            logical_size: [width, height],
            last_dpi_change: None,
            pre_dpi_outer: None,
            last_scroll_tick: None,
            last_hud_refresh: None,
            staged_images: Vec::new(),
            scene_builder: None,
            rebuild_request: std::rc::Rc::new(std::cell::Cell::new(false)),
            dwell: None,
            active_hint: None,
            pending_hint_repaint: false,
            hint_fade: crate::signal::Signal::new(0.0),
            hint_visible: false,
            next_hover_recheck: None,
            wake: wake.clone(),
            uploader: crate::uploader::Uploader::new(wake.clone()),
            frame_sink: crate::frame_sink::FrameSink::new(wake),
            window: None,
            gpu: None,
            #[cfg(feature = "automation")]
            automation: None,
        }
    }

    /// Get a clone of the rebuild-request token. Pass into closures
    /// that want to trigger a scene rebuild — calling `.set(true)` on
    /// the returned `Rc<Cell<bool>>` schedules a rebuild at the top of
    /// the next event loop iteration.
    ///
    /// ```ignore
    /// let rebuild = app.rebuild_token();
    /// let view = view_sig.clone();
    /// some_button.on_click(move |_| {
    ///     view.set(View::Search);
    ///     rebuild.set(true);
    /// });
    /// ```
    pub fn rebuild_token(&self) -> std::rc::Rc<std::cell::Cell<bool>> {
        self.rebuild_request.clone()
    }

    /// Get a clone of the cross-thread wake handle. Pass into worker
    /// threads so they can nudge the UI loop out of `ControlFlow::Wait`
    /// after delivering a response on an `mpsc` channel the UI polls
    /// in [`Self::on_frame`]. Without this, a `Wait`-parked loop never
    /// sees the response until the user moves the mouse.
    pub fn wake_handle(&self) -> Arc<WakeHandle> {
        self.wake.clone()
    }

    /// Get a clone of the cross-thread image upload token. Pass to any
    /// worker that needs to ship decoded RGBA bytes onto the GPU at
    /// runtime — e.g. album art or remote avatars. See
    /// [`crate::Uploader`] for the threading model.
    pub fn uploader(&self) -> Arc<crate::uploader::Uploader> {
        self.uploader.clone()
    }

    /// Get a clone of the cross-thread external-frame (video) submission
    /// token. Pass to a decoder thread that produces frames for an
    /// [`.external()`](crate::node::NodeBuilder::external) node; each
    /// [`FrameSink::submit`](crate::frame_sink::FrameSink::submit) ships a
    /// frame onto that node's external texture (latest-wins). See
    /// [`crate::FrameSink`] for the threading model.
    pub fn frame_sink(&self) -> Arc<crate::frame_sink::FrameSink> {
        self.frame_sink.clone()
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

    /// Stage an SVG byte slice. Rasterized at `px × px` physical pixels
    /// via [`crate::svg::rasterize_svg`] and queued for upload alongside
    /// PNG / RGBA stages. Pick `px` to over-sample your largest expected
    /// display size — the image atlas bilinear-downsamples for smaller
    /// draws without re-rasterizing.
    ///
    /// `include_bytes!(...)` is the typical source.
    pub fn stage_image_svg(&mut self, bytes: &[u8], px: u32) -> ImageHandle {
        let rgba = crate::svg::rasterize_svg(bytes, px);
        self.stage_image_rgba(px, px, rgba)
    }

    /// Stage pre-decoded `Rgba8UnormSrgb` pixels (`w*h*4` bytes,
    /// row-major, top-left origin). Same scheduling as
    /// [`Self::stage_image_png`].
    pub fn stage_image_rgba(&mut self, w: u32, h: u32, bytes: Vec<u8>) -> ImageHandle {
        let id = self.staged_images.len() as u32;
        self.staged_images.push(StagedImage::Rgba { w, h, bytes });
        ImageHandle(id)
    }

    /// Install the scene builder. Runs the closure once immediately
    /// against the empty `SceneCtx` (preserving existing one-shot
    /// behavior) and stashes it for future [`App::rebuild_scene`]
    /// calls. The closure must be `FnMut` since rebuilds may fire
    /// repeatedly — capture signals + handles by clone, not by move.
    pub fn scene<F: FnMut(&mut Scene) + 'static>(mut self, mut f: F) -> Self {
        {
            let mut scene = Scene::root(&mut self.ctx);
            f(&mut scene);
        }
        self.scene_builder = Some(Box::new(f));
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

    /// Provide a closure invoked once per `about_to_wait` iteration —
    /// the per-frame hook for custom animations that need to interpolate
    /// state over time (e.g. lazy-list row heights). The shell calls
    /// it after consuming the rebuild token and before the timeline
    /// tick, so any tree mutations the closure makes ride this frame.
    /// `now` is the wall-clock at the start of the iteration.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut SceneCtx, &mut crate::anim::Timeline, Instant) + 'static,
    {
        self.on_frame = Some(Box::new(f));
        self
    }

    /// Hook fired once when the event loop is exiting — winit's
    /// `exiting()` callback. Use for save-on-quit work (persist prefs,
    /// disconnect remote sessions, flush logs). Fires whether the exit
    /// was triggered by the close button, `event_loop.exit()`, or a
    /// kill signal that lets winit unwind cleanly.
    pub fn on_exit<F: FnOnce() + 'static>(mut self, f: F) -> Self {
        self.on_exit = Some(Box::new(f));
        self
    }

    /// Register the outside-press hook. Fires on a
    /// left-button press that lands on no interactive node, or on a
    /// node tagged [`crate::scene::NodeBuilderRef::dismiss_transparent`].
    /// The canonical dismiss driver for modals + context menus: flip the
    /// layer's visibility signal in the closure and request a rebuild.
    pub fn on_unhandled_press<F: FnMut(&mut SceneCtx) + 'static>(mut self, f: F) -> Self {
        self.on_unhandled_press = Some(Box::new(f));
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

    /// Delay before the autocapture fires, in addition to whatever
    /// `capture_frames` requested. Lets the loop tick normally for `d`
    /// (worker responses, image uploads, animations all run) and then
    /// snaps the final frame. Cumulative with both `capture_once` and
    /// `capture(N, dir)`.
    pub fn capture_delay(mut self, d: Duration) -> Self {
        self.capture_delay = Some(d);
        self
    }

    /// Env-var shim for the legacy `OPAL_AUTOCAPTURE` flag.
    /// Returns `self` unchanged when the variable is not set, so the
    /// call is harmless in normal interactive runs. CI/self-verify
    /// keeps working without code changes; scripted multi-frame flows
    /// still use [`App::headless`] separately.
    ///
    /// Also honours `OPAL_AUTOCAPTURE_DELAY_MS` (integer ms) — if
    /// set, the autocapture is deferred by that long so async UI state
    /// has time to populate before the snapshot.
    pub fn capture_from_env(mut self) -> Self {
        if std::env::var_os("OPAL_AUTOCAPTURE").is_some() {
            self = self.capture_once();
        }
        if let Some(raw) = std::env::var_os("OPAL_AUTOCAPTURE_DELAY_MS")
            && let Ok(s) = raw.into_string()
            && let Ok(ms) = s.trim().parse::<u64>()
        {
            self = self.capture_delay(Duration::from_millis(ms));
        }
        self
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

    /// Round the window's four corners by `r` logical px. Implemented
    /// as an SDF clip in the final fragment shader, so it's free
    /// (~one extra ALU op per fragment) and works against the existing
    /// transparent winit surface. Automatically zeroed out while the
    /// window is maximised / fullscreen — rounding flush against the
    /// work-area boundary clips usable pixels for no visual gain.
    pub fn window_corner_radius(mut self, r: f32) -> Self {
        self.config.window_corner_radius = r.max(0.0);
        self
    }

    /// Set the taskbar / alt-tab window icon from raw RGBA8 (`w * h * 4`
    /// bytes). On Windows this complements the embedded executable icon —
    /// the exe icon covers file-explorer / pinned shortcuts, this covers
    /// the live window. Silently ignored if the byte length doesn't match
    /// `width * height * 4` (handled at window creation).
    pub fn window_icon_rgba(mut self, width: u32, height: u32, rgba: Vec<u8>) -> Self {
        self.config.window_icon = Some((width, height, rgba));
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
        // Install the wake proxy now that the loop exists. Any
        // `WakeHandle::wake` calls from worker threads from this point
        // forward will deliver an empty UserEvent that breaks the
        // loop out of `Wait`.
        let _ = self.wake.proxy.set(event_loop.create_proxy());
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    // ---- Internal helpers ----------------------------------------------

    /// Re-flatten + upload the tree if any dirty flag is set.
    fn flush_tree(&mut self) -> bool {
        // Refresh text-field display strings (value ↔ placeholder
        // swap on focus change / value clear) **before** `take_dirty`
        // so any text mutations made here ride this flush rather than
        // queuing a follow-up. `set_text` is idempotent — no-op when
        // content is unchanged — so this is cheap on idle frames.
        self.refresh_text_fields();
        let mask = self.ctx.tree.take_dirty();
        if mask == 0 {
            return false;
        }
        self.last_dirty_mask = mask;
        // Rebuild prefix-sum tables for any variable-height lazy
        // lists whose heights have moved. compute_layout's
        // `content_size` override + materialize's `visible_window`
        // both read from `total_height_logical()` / `prefix`, so
        // those need to be current before either runs.
        self.ensure_lazy_list_prefixes();
        // CPU sub-phase timing (logged under OPAL_STATS): pinpoints
        // where a scroll frame's CPU goes — layout vs flatten vs expand +
        // upload — so the scroll-path optimization targets the real hotspot.
        let t_layout = Instant::now();
        let needs_initial_layout =
            mask & (crate::node::dirty::TREE | crate::node::dirty::TRANSFORM) != 0;
        let viewport = self.viewport();
        if needs_initial_layout {
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                viewport,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        // Lazy-list materialization runs whenever the tree is touched
        // — including SCROLL-only flushes, since scrolling can cross a
        // row boundary and bring new rows into view. Internally gated
        // on `(new_range, version)` so idle frames cost nothing.
        if self.materialize_lazy_lists() {
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                viewport,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        if needs_initial_layout {
            // Post-layout caret + visibility sync. Editor caret rects
            // depend on the text-child's post-layout rect, so this
            // MUST run after `compute_layout` and BEFORE
            // `flatten_into_buffers` (which snapshots `rect` +
            // `visible` into instances).
            self.reposition_carets();
        }
        let layout_ms = t_layout.elapsed().as_secs_f32() * 1000.0;
        let t_flatten = Instant::now();
        self.ctx.tree.flatten_into_buffers(
            self.scale_factor,
            &mut self.flat_events,
            &mut self.hits,
            &mut self.scroll_hits,
            &mut self.scroll_bars,
            &mut self.flat_spans,
        );
        let flatten_ms = t_flatten.elapsed().as_secs_f32() * 1000.0;
        let t_expand = Instant::now();
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        if let Some(gpu) = self.gpu.as_mut() {
            self.instances.clear();
            self.glass_count = expand_events_into(
                &self.flat_events,
                &mut self.instances,
                gpu,
                &mut self.ctx.text,
                self.scale_factor,
                &mut self.flat_event_inst,
            );
            gpu.set_instances(&self.instances, self.glass_count, backdrop_hint);
        }
        let expand_ms = t_expand.elapsed().as_secs_f32() * 1000.0;
        if self.stats_log {
            log::info!(
                "flush cpu: layout={layout_ms:.2}ms flatten={flatten_ms:.2}ms expand+upload={expand_ms:.2}ms (mask={mask:#x} instances={})",
                self.instances.len(),
            );
        }
        // Rebuild the layer tree: map the promoted (`.layer()`) subtree
        // event spans to instance ranges, then partition the stream into
        // root segments + promoted layers. No promotions → single root
        // layer (P2 parity). Composite transform/opacity for surviving
        // promotions persists across this re-flatten.
        let damage = crate::layer::Damage::classify(mask);
        let promoted = spans_to_instance_ranges(&self.flat_spans, &self.flat_event_inst);
        self.layer_tree.rebuild(
            &promoted,
            self.instances.len(),
            [viewport[0] as u32, viewport[1] as u32],
            damage,
        );
        // A full flatten supersedes any pending composite-only change.
        self.layer_tree.take_composite_dirty();
        // Re-collect declarative layer-opacity bindings from the freshly
        // promoted nodes (`.layer_opacity(sig)`). Rebuilt each flush so it
        // tracks the current tree (promotions can come + go); the per-frame
        // pump then drives each layer's composite opacity from its signal.
        self.layer_opacity_binds.clear();
        self.layer_offset_x_binds.clear();
        for p in &promoted {
            if let Some(n) = self.ctx.tree.get(p.node) {
                if let Some(sig) = n.layer_opacity.clone() {
                    self.layer_opacity_binds.push((p.node, sig));
                }
                if let Some(sig) = n.layer_offset_x.clone() {
                    self.layer_offset_x_binds.push((p.node, sig));
                }
            }
        }
        self.pump_layer_opacity_binds();
        self.pump_layer_offset_binds();
        self.push_layer_draws();
        true
    }

    /// Build the GPU layer draw list from the current `LayerTree` and
    /// hand it to the renderer. Cheap; called after a flatten and on
    /// composite-only frames (where the instance stream is untouched but
    /// a layer's composite transform/opacity changed).
    fn push_layer_draws(&mut self) {
        // Build into a reused scratch buffer (cleared, not reallocated) so a
        // composite-only update — scroll, crossfade, a cursor-following
        // overlay offset — doesn't allocate a fresh Vec each frame.
        self.layer_draw_scratch.clear();
        let scale = self.scale_factor;
        for layer in self.layer_tree.layers() {
            let mut draw = layer_to_draw(layer);
            // Composite-time corner rounding for a promoted raster `.layer()`
            // (`root` set, not external/scroll) whose node carries a radius:
            // round the layer's *composited* result once at its rect, instead
            // of rounding each child. Lets a grouped card (e.g. an album-cover
            // crossfade) draw its content square and get a single clean
            // anti-aliased corner — stacked content can't leak through it.
            if draw.external.is_none()
                && draw.window.is_none()
                && let Some(id) = layer.root
                && let Some(n) = self.ctx.tree.get(id)
            {
                let r = n.style.border_radius[0];
                if r > 0.0 {
                    let rect = n.rect; // physical [x, y, w, h]
                    draw.corner_radius = r * scale;
                    draw.round_rect =
                        Some([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]);
                }
            }
            self.layer_draw_scratch.push(draw);
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_layers(&self.layer_draw_scratch);
        }
    }

    /// Set a `.layer()`-promoted subtree's composite offset (logical px;
    /// scaled to physical here). This is a **composite-only** change: no
    /// re-flatten, no re-raster — the cached layer texture is recomposited
    /// at the new offset. Drives slide transitions / scroll cheaply.
    /// Requests a redraw when the value actually changed.
    pub fn set_layer_offset(&mut self, node: crate::node::NodeId, offset: [f32; 2]) {
        let phys = [offset[0] * self.scale_factor, offset[1] * self.scale_factor];
        if self.layer_tree.set_offset(node, phys) {
            self.push_layer_draws();
            self.request_redraw();
        }
    }

    /// Set a promoted subtree's composite opacity. Composite-only — see
    /// [`Self::set_layer_offset`].
    pub fn set_layer_opacity(&mut self, node: crate::node::NodeId, opacity: f32) {
        if self.layer_tree.set_opacity(node, opacity) {
            self.push_layer_draws();
            self.request_redraw();
        }
    }

    /// Push every layer-opacity binding's current signal value into its
    /// layer's composite opacity. Called each awake frame after the
    /// timeline tick. No-op when no bindings are registered.
    fn pump_layer_opacity_binds(&mut self) {
        // Take/restore (no alloc) to iterate while calling `&mut self` —
        // see [`Self::pump_layer_offset_binds`].
        let binds = std::mem::take(&mut self.layer_opacity_binds);
        for (node, sig) in &binds {
            self.set_layer_opacity(*node, sig.get());
        }
        self.layer_opacity_binds = binds;
    }

    /// Push every layer-offset-x binding's current signal value into its
    /// layer's composite X offset (logical px). Composite-only — no
    /// re-flatten. Called each awake frame after the timeline tick.
    fn pump_layer_offset_binds(&mut self) {
        // Take the bind list out (swap with an empty Vec — no alloc) so the
        // `&mut self` calls below don't conflict with borrowing it, then put
        // it back. `set_layer_offset` never touches this field. Keeps a
        // cursor-following overlay's per-move cost alloc-free.
        let binds = std::mem::take(&mut self.layer_offset_x_binds);
        for (node, sig) in &binds {
            self.set_layer_offset(*node, [sig.get(), 0.0]);
        }
        self.layer_offset_x_binds = binds;
    }

    /// Register/replace the **external texture** for an `.external()` node
    /// (P6) — its layer composites this view (a video / Spotify Canvas
    /// decoder frame) instead of a rastered texture. Call each time the
    /// decoder produces a new frame, then `request_redraw` (or the next
    /// awake frame) recomposites. No-op until the GPU is initialised.
    pub fn set_external_texture(&mut self, node: crate::node::NodeId, view: wgpu::TextureView) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_external_texture(node, view);
            self.request_redraw();
        }
    }

    /// Upload a decoder frame (tightly-packed `width * height * 4` RGBA8,
    /// sRGB-encoded) as `node`'s external texture and recomposite. The
    /// engine owns + reuses the backing texture across frames, so calling
    /// this every video tick is cheap. Prefer this over building a
    /// `wgpu::Texture` yourself + [`Self::set_external_texture`] unless you
    /// already have a GPU texture in hand. No-op until the GPU is up.
    pub fn set_external_frame(
        &mut self,
        node: crate::node::NodeId,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.upload_external_frame(node, rgba, width, height);
            self.request_redraw();
        }
    }

    /// Drop a node's external texture. See [`Self::set_external_texture`].
    pub fn clear_external_texture(&mut self, node: crate::node::NodeId) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_external_texture(node);
        }
    }

    /// Set a promoted subtree's composite scale. Composite-only — see
    /// [`Self::set_layer_offset`].
    pub fn set_layer_scale(&mut self, node: crate::node::NodeId, scale: [f32; 2]) {
        if self.layer_tree.set_scale(node, scale) {
            self.push_layer_draws();
            self.request_redraw();
        }
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
        // A live hover-hint tooltip owns the overlay buffer; don't clobber it
        // with the debug HUD (the user-facing hint wins).
        if self.active_hint.is_some() {
            return;
        }
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

    /// Resolve a node's tooltip text: the reactive `hover_hint_text` signal
    /// takes priority (so it can update without a rebuild), falling back to the
    /// static `hover_hint`. Returns `None` only when the node has neither.
    fn hint_text(node: &crate::node::Node) -> Option<String> {
        node.hover_hint_text
            .as_ref()
            .map(|s| s.get().to_string())
            .or_else(|| node.hover_hint.clone())
    }

    /// Start showing `node`'s `hover_hint` tooltip (its dwell just fired):
    /// fade it in. Painted at the cursor and re-painted as the pointer moves
    /// + as the fade advances (see [`Self::paint_hint`]).
    fn show_hint(&mut self, node: crate::node::NodeId) {
        self.active_hint = Some(node);
        self.hint_visible = true;
        let fade = self.hint_fade.clone();
        self.timeline
            .animate(&fade, 1.0, crate::anim::Curve::EaseInOut, HINT_FADE, Instant::now());
        self.paint_hint();
    }

    /// Begin fading the active hint *out* (the pointer left its node but is
    /// still in the window, so the cursor anchor is valid through the fade).
    /// The pump finalises it once the fade reaches 0.
    fn hide_hint(&mut self) {
        if self.active_hint.is_none() || !self.hint_visible {
            return;
        }
        self.hint_visible = false;
        let fade = self.hint_fade.clone();
        self.timeline
            .animate(&fade, 0.0, crate::anim::Curve::EaseInOut, HINT_FADE, Instant::now());
    }

    /// (Re)render the active hint tooltip into the overlay buffer at the
    /// current cursor position, at the current fade opacity. Cheap — one pill
    /// + a short label (glyphs are atlas-cached).
    fn paint_hint(&mut self) {
        let Some(node) = self.active_hint else {
            return;
        };
        let Some(text) = self.ctx.tree.get(node).and_then(Self::hint_text) else {
            // Node gone (rebuild) — drop the hint immediately.
            self.clear_hint();
            return;
        };
        if text.is_empty() {
            // Reactive hint resolved to nothing (e.g. track not in any
            // playlist) — keep the overlay empty rather than a bare pill.
            self.clear_hint();
            return;
        }
        let cursor = self.input.cursor.unwrap_or([0.0, 0.0]);
        let fade = self.hint_fade.get().clamp(0.0, 1.0);
        let scale = self.scale_factor;
        // Surface size in physical px for edge-clamping the tooltip.
        let surface = [
            self.logical_size[0] as f32 * scale,
            self.logical_size[1] as f32 * scale,
        ];
        if let Some(gpu) = self.gpu.as_mut() {
            let instances =
                build_hint_instances(&text, cursor, surface, fade, gpu, &mut self.ctx.text, scale);
            gpu.set_overlay_instances(&instances);
        }
        self.request_redraw();
    }

    /// Immediately remove the hint (no fade): cursor left the window, or the
    /// scene rebuilt out from under it. Hands the overlay back to the HUD (if
    /// on) or clears it. No-op when no hint is shown.
    fn clear_hint(&mut self) {
        let was = self.active_hint.take().is_some();
        self.hint_visible = false;
        let fade = self.hint_fade.clone();
        self.timeline.stop_for(&fade);
        self.hint_fade.set(0.0);
        if !was {
            return;
        }
        if self.hud_enabled {
            // Force a rebuild past the throttle so the HUD reappears now.
            self.last_hud_refresh = None;
            self.refresh_hud_overlay();
        } else if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_overlay_instances(&[]);
        }
        self.request_redraw();
    }

    /// Forward the GPU memory-allocation snapshot from the renderer.
    /// Returns `None` if the GPU isn't initialised yet.
    pub fn memory_report(&self) -> Option<crate::gpu::MemoryReport> {
        self.gpu.as_ref().map(|g| g.memory_report())
    }

    /// Log the current memory breakdown. `image_atlas`/`*_textures` are
    /// VRAM; `*_cpu` are system RAM. Fired at startup (baseline) and on F1
    /// so the live working set (after covers stream + a canvas plays) can be
    /// read, not just the empty-scene baseline.
    pub fn log_memory_report(&self) {
        if let Some(mem) = self.memory_report() {
            log::info!(
                "gpu memory: total={} KiB (instance={} overlay={} blur={} overdraw={} glyph_atlas={} image_atlas={} canvas_frames={} img_src_cpu={} timing={} prev_cpu={})",
                mem.total() / 1024,
                mem.instance_buffer,
                mem.overlay_buffer,
                mem.blur_textures,
                mem.overdraw_textures,
                mem.glyph_atlas,
                mem.image_atlas,
                mem.external_frames,
                mem.image_sources_cpu,
                mem.timing,
                mem.prev_instances_cpu,
            );
        }
    }

    /// Drain the `Uploader` queue: perform each pending RGBA upload on
    /// the UI thread (where the GPU lives) and fire the caller's
    /// completion callback with the resolved [`ImageHandle`]. No-op when
    /// the queue is empty. Called from `about_to_wait`.
    fn drain_image_uploads(&mut self) {
        let pending = self.uploader.drain();
        if pending.is_empty() {
            return;
        }
        for p in pending {
            let handle = self.upload_image_rgba(p.w, p.h, &p.bytes);
            (p.cb)(handle);
        }
    }

    /// Drain queued external (video) frames, uploading each onto its
    /// node's external texture. Latest-wins per node — see
    /// [`crate::FrameSink`]. Triggers a recomposite when any frame lands.
    /// Drain queued external (video) frames onto their nodes' textures.
    /// Returns `true` if anything was drained — the caller renders directly
    /// rather than via `request_redraw`, because a redraw requested while
    /// the loop is about to park on `Wait` isn't reliably honoured (the
    /// video would freeze the moment no timeline keeps the loop awake).
    /// Uploads straight to the GPU here (no per-frame `request_redraw`).
    fn drain_external_frames(&mut self) -> bool {
        let pending = self.frame_sink.drain();
        if pending.is_empty() {
            return false;
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return false;
        };
        for cmd in pending {
            match cmd {
                crate::frame_sink::FrameCmd::Frame { node, frame } => {
                    gpu.upload_external_frame(node, &frame.rgba, frame.width, frame.height);
                }
                crate::frame_sink::FrameCmd::Push { node, epoch, frame } => {
                    gpu.push_external_frame(node, epoch, &frame.rgba, frame.width, frame.height);
                }
                crate::frame_sink::FrameCmd::Select { node, epoch, index } => {
                    gpu.select_external_frame(node, epoch, index);
                }
                crate::frame_sink::FrameCmd::Migrate { old, new } => {
                    gpu.migrate_external_frames(old, new);
                }
                crate::frame_sink::FrameCmd::Clear { node } => {
                    gpu.clear_external_texture(node);
                }
            }
        }
        true
    }

    /// Runtime image upload (post-GPU-init). Unlike [`stage_image_*`]
    /// — which queues bytes until first frame — this uploads
    /// immediately into the live atlas. Returns `None` if the GPU
    /// isn't initialised yet, the input dims are invalid, or the
    /// image is larger than the atlas itself.
    ///
    /// When the atlas is full, walks the tree for live image handles,
    /// drops sources for any handle no longer referenced, repacks the
    /// survivors into a fresh atlas layout, then retries the upload
    /// once. The repack updates every surviving handle's UVs in place
    /// — flatten + instance rebuild on the next frame picks them up
    /// automatically (TREE/TRANSFORM dirty flags are set by this
    /// method to force one).
    pub fn upload_image_rgba(
        &mut self,
        w: u32,
        h: u32,
        bytes: &[u8],
    ) -> Option<crate::gpu::ImageHandle> {
        // The live-handle set is only needed on the rare evict path, and
        // computing it walks the whole tree — so pass it lazily (borrow
        // `ctx.tree`, disjoint from the `gpu` borrow) instead of paying the
        // walk on every cover that lands during a fast scroll.
        let tree = &self.ctx.tree;
        let gpu = self.gpu.as_mut()?;
        // Cap atlas growth at the adapter's max 2D texture dimension —
        // 8192² with wgpu default limits, up to 16384² on most desktop
        // adapters if higher limits are requested. The atlas is allocated
        // large up front (see `ImageAtlas::new` in `context.rs`) so growth
        // — and its eviction fallback — rarely if ever fires; this keeps
        // every uploaded handle resident (no eviction → no covers that
        // "never load" from a dangling handle).
        let max = gpu.device.limits().max_texture_dimension_2d;
        let outcome = gpu.image_atlas.upload_rgba_growing(
            &gpu.device,
            &gpu.queue,
            w,
            h,
            bytes,
            || collect_live_image_handles(tree),
            max,
        )?;
        // Only force a full re-flatten when the atlas **repacked** (grow /
        // evict moved every existing handle's UV). On the common fast path
        // (the atlas had room → no UV moved) this is skipped: the node(s)
        // bound to the new handle are repainted by the reactive image-bind
        // (`set_image` dirties just those nodes), so a full
        // `mark_all_dirty` here would needlessly re-flatten + re-raster the
        // **whole scene** on every cover that lands — which is exactly what
        // made playlist scrolling lag while hundreds of covers streamed in
        // (it defeated the per-layer raster-skip). A redraw is enough.
        if outcome.layout_changed {
            self.ctx.tree.mark_all_dirty();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        Some(outcome.handle)
    }

    /// Runtime SVG upload. Rasterizes `bytes` to `px × px` and forwards
    /// to [`Self::upload_image_rgba`]. Same eviction semantics — over a
    /// full atlas, walks the tree for live handles and repacks survivors.
    pub fn upload_image_svg(&mut self, bytes: &[u8], px: u32) -> Option<crate::gpu::ImageHandle> {
        let rgba = crate::svg::rasterize_svg(bytes, px);
        self.upload_image_rgba(px, px, &rgba)
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
        // layer_count / raster_count / composite_count / layer_vram are
        // filled by the GPU (`last_frame_stats`) — it knows what it
        // actually rastered, composited, and allocated.
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
        process_image_binds(&mut self.ctx.binds.image, &mut self.ctx.tree);
        process_text_binds(&mut self.ctx.binds.text, &mut self.ctx.tree);
        process_width_pct_binds(&mut self.ctx.binds.width_pct, &mut self.ctx.tree);
        process_width_px_binds(&mut self.ctx.binds.width_px, &mut self.ctx.tree);
        process_height_px_binds(&mut self.ctx.binds.height_px, &mut self.ctx.tree);
        process_opacity_binds(
            &mut self.ctx.binds.opacity,
            &mut self.ctx.tree,
            &mut self.timeline,
            now,
        );
    }

    /// For animated binds, copy the current `displayed` signal value
    /// (driven by the timeline) into the tree. Called after every
    /// timeline tick.
    fn pump_animated_displays(&mut self) {
        for slot in self.ctx.binds.color.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.position.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.size.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_size_px(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.opacity.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                let o = disp.get();
                self.ctx.tree.set_opacity(slot.node_id, o);
                // Mirror the snap path's visibility gating against the live
                // tweened value, so a fading overlay drops out of flatten /
                // hit-testing exactly when it crosses transparent.
                self.ctx.tree.set_visible(slot.node_id, o > 0.001);
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

    /// Note that the scene re-flattened (the hit cache changed) so hover may
    /// be stale under a stationary cursor. Arms a debounced re-check — no
    /// reset while one is already pending, which both throttles continuous
    /// change (≈10 Hz) and lets the last-armed check fire once after it
    /// stops. Cheap no-op when the pointer isn't in the window.
    fn arm_hover_recheck(&mut self) {
        if self.input.cursor.is_some() && self.next_hover_recheck.is_none() {
            self.next_hover_recheck = Some(Instant::now() + HOVER_RECHECK);
        }
    }

    /// Fire a due hover re-check: re-run the hit-test at the current cursor
    /// against the freshly-flattened hits, so a row that scrolled / morphed
    /// under a still pointer lights up (or un-lights) correctly. Deferred
    /// while a drag owns the pointer (hover is suppressed then; it fires once
    /// the drag releases). Returns `true` if hover changed (it reacted).
    fn tick_hover_recheck(&mut self, now: Instant) -> bool {
        let Some(deadline) = self.next_hover_recheck else {
            return false;
        };
        if now < deadline {
            return false;
        }
        // A drag/capture owns the pointer — hover is moot until it releases.
        if self.input.captured.is_some()
            || self.bar_drag.is_some()
            || self.drag_origin.is_some()
        {
            self.next_hover_recheck = Some(now + HOVER_RECHECK);
            return false;
        }
        self.next_hover_recheck = None;
        let Some([x, y]) = self.input.cursor else {
            return false;
        };
        let _ = crate::input::update_scrollbar_hover(
            Some([x, y]),
            &self.scroll_bars,
            &mut self.ctx.tree,
        );
        let change = self.input.on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
        if change.hovered_changed {
            self.refresh_dwell(now);
        }
        if change.any() {
            self.react();
            return true;
        }
        false
    }

    /// Re-sync the hover-dwell tracker against the current hovered
    /// node. Called after any input event that may have changed
    /// `input.hovered`. If the hovered node owns a dwell handler and
    /// either no tracker is armed or the tracker points at a different
    /// node, arm a fresh deadline. If the hovered node has no dwell
    /// handler (or no node is hovered), clear the tracker. Idempotent
    /// while hover stays on the same dwell-handler-bearing node.
    fn refresh_dwell(&mut self, now: Instant) {
        // A node is a dwell target if it has an explicit dwell handler OR a
        // `hover_hint` tooltip (which uses the same timer + a default delay).
        let hovered_dwell = self.input.hovered.and_then(|id| {
            self.ctx.tree.get(id).and_then(|n| {
                n.on_hover_dwell
                    .as_ref()
                    .map(|(d, _)| *d)
                    .or_else(|| {
                        (n.hover_hint.is_some() || n.hover_hint_text.is_some())
                            .then_some(HINT_DWELL)
                    })
                    .map(|d| (id, d))
            })
        });
        // Fade a shown hint out the moment hover leaves its node (so it
        // doesn't linger when the cursor moves to a sibling icon). The cursor
        // is still in the window, so the fade stays anchored to it.
        if let Some(hint_node) = self.active_hint
            && self.input.hovered != Some(hint_node)
        {
            self.hide_hint();
        }
        match (hovered_dwell, self.dwell) {
            (None, _) => self.dwell = None,
            (Some((id, duration)), Some(tracker)) if tracker.node == id => {
                // Same node — keep existing armed deadline (no reset
                // on micro cursor moves over the same node).
                let _ = duration;
            }
            (Some((id, duration)), _) => {
                self.dwell = Some(DwellTracker {
                    node: id,
                    deadline: now + duration,
                    fired: false,
                });
            }
        }
    }

    /// If a dwell deadline has elapsed and not yet fired, invoke the
    /// node's handler and mark fired. Returns true if a handler ran
    /// (caller should `react()`).
    fn tick_dwell(&mut self, now: Instant) -> bool {
        let Some(tracker) = self.dwell else {
            return false;
        };
        if tracker.fired || now < tracker.deadline {
            return false;
        }
        // Confirm hover is still on the tracker's node.
        if self.input.hovered != Some(tracker.node) {
            self.dwell = None;
            return false;
        }
        let node = tracker.node;
        let (handler, has_hint) = self
            .ctx
            .tree
            .get(node)
            .map(|n| {
                (
                    n.on_hover_dwell.as_ref().map(|(_, h)| h.clone()),
                    n.hover_hint.is_some() || n.hover_hint_text.is_some(),
                )
            })
            .unwrap_or((None, false));
        if handler.is_none() && !has_hint {
            self.dwell = None;
            return false;
        }
        // Mark fired *before* invoking so a handler that mutates the
        // tree (and triggers a flush re-evaluating hover) doesn't
        // re-enter and double-fire.
        if let Some(t) = self.dwell.as_mut() {
            t.fired = true;
        }
        // Show the tooltip (if any) for this node.
        if has_hint {
            self.show_hint(node);
        }
        if let Some(h) = handler {
            let mut ectx = crate::event::EventCtx {
                tree: &mut self.ctx.tree,
                timeline: &mut self.timeline,
                node,
                now,
                cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
            };
            h(&mut ectx);
        }
        true
    }

    /// Drop a node and every descendant, cleaning up bind slots, named
    /// references, and any in-flight tweens that targeted them. The
    /// scene rebuilds its instance stream on the next flush. Use this
    /// for dynamic UIs — track-list refresh, view-routing, modal
    /// dismiss. Returns the same [`crate::scene::SubtreeRemoval`] as
    /// `SceneCtx::remove_subtree`; most callers ignore it.
    pub fn remove_subtree(&mut self, id: crate::node::NodeId) -> crate::scene::SubtreeRemoval {
        let removal = self.ctx.remove_subtree(id);
        stop_tweens_for_removal(&mut self.timeline, &removal);
        if self.flush_tree() {
            self.request_redraw();
        }
        removal
    }

    /// Apply an [`crate::editor::EditOp`] to the editor on `node_id`.
    /// Handles the full ripple: edit state mutation → text-child
    /// `set_text` → `on_change` fire → caret reposition (deferred to
    /// the next `flush_tree`) → `on_submit` fire → react.
    ///
    /// No-op if `node_id` has no editor.
    fn apply_edit(&mut self, node_id: crate::node::NodeId, op: crate::editor::EditOp) {
        // Step 1: apply op + capture every piece of post-edit state we
        // need below. Done in a scoped block so the &mut editor borrow
        // releases before subsequent tree mutations.
        let (mut outcome, new_value, on_change, on_submit, text_node) = {
            let Some(node) = self.ctx.tree.get_mut_raw(node_id) else {
                return;
            };
            let Some(ed) = node.editor.as_mut() else {
                return;
            };
            let outcome = crate::editor::apply(op, ed);
            (
                outcome,
                ed.value.clone(),
                ed.on_change.clone(),
                ed.on_submit.clone(),
                ed.text_node,
            )
        };

        // Clipboard write (Copy / Cut). Best-effort — a failed arboard
        // init just drops the write; the edit (Cut's delete) still ran.
        if let Some(text) = outcome.clipboard_write.take() {
            self.write_clipboard(text);
        }

        // Step 2: refresh the text child node's content. Marks
        // VISUAL | TRANSFORM so the layout pass re-measures + re-shapes
        // on the next flush.
        if outcome.value_changed {
            self.ctx.tree.set_text(text_node, new_value.as_str());
        }

        // Step 3: fire user callbacks. `on_change` runs with all
        // exclusive borrows released; `on_submit` re-borrows the tree
        // (EventCtx) so the user can drive sibling state in response
        // to Enter.
        if outcome.value_changed
            && let Some(cb) = on_change
        {
            cb(&new_value);
        }
        if outcome.submitted
            && let Some(h) = on_submit
        {
            let mut ectx = crate::event::EventCtx {
                tree: &mut self.ctx.tree,
                timeline: &mut self.timeline,
                node: node_id,
                now: Instant::now(),
                cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
            };
            h(&mut ectx);
        }

        // Step 4: react — runs bind processing + pumps animated
        // displays + flushes the tree (which calls reposition_carets
        // post-layout via flush_tree).
        if outcome.any() {
            // Cursor moves + selection changes without a value change
            // still want a re-render so the caret / highlight reposition.
            // The TRANSFORM flag isn't auto-set by those — set it here.
            if (outcome.cursor_moved || outcome.selection_changed) && !outcome.value_changed {
                self.ctx.tree.mark_transform_dirty();
            }
            // Keep the caret solid while actively editing — restart the
            // blink phase (and force it visible) so it doesn't wink out
            // mid-keystroke.
            self.reset_caret_blink(node_id);
            self.react();
        }
    }

    /// Advance the caret blink for the focused text field. Writes the caret
    /// node's opacity only on the half-second it toggles, and returns the
    /// next toggle deadline so the loop can wake for it. `None` when no text
    /// field is focused (caret blink idle → loop can park).
    fn tick_caret_blink(&mut self, now: Instant) -> (bool, Option<Instant>) {
        const HALF: std::time::Duration = std::time::Duration::from_millis(530);
        let Some(field) = self.input.focused else {
            self.caret_blink_on = true;
            return (false, None);
        };
        let Some(caret_node) = self
            .ctx
            .tree
            .get(field)
            .and_then(|n| n.editor.as_ref())
            .map(|ed| ed.caret_node)
        else {
            return (false, None);
        };
        let elapsed_ms = now.saturating_duration_since(self.caret_blink_anchor).as_millis();
        let on = (elapsed_ms / HALF.as_millis()).is_multiple_of(2);
        let changed = on != self.caret_blink_on;
        if changed {
            self.caret_blink_on = on;
            self.ctx
                .tree
                .set_opacity(caret_node, if on { 1.0 } else { 0.0 });
        }
        let into_phase = (elapsed_ms % HALF.as_millis()) as u64;
        let next = now + (HALF - std::time::Duration::from_millis(into_phase));
        (changed, Some(next))
    }

    /// Byte index of the char boundary in text field `node_id` nearest the
    /// given x (physical px). Walks the value's char boundaries, measuring
    /// each prefix against the text child's painted origin. `None` on a
    /// non-editor node.
    fn caret_index_at_x(&mut self, node_id: crate::node::NodeId, x: f32) -> Option<usize> {
        let (value, font_size, text_node) = {
            let ed = self.ctx.tree.get(node_id).and_then(|n| n.editor.as_ref())?;
            (ed.value.clone(), ed.font_size, ed.text_node)
        };
        let origin_x = self.ctx.tree.get(text_node).map(|n| n.rect[0]).unwrap_or(0.0);
        let scale = self.scale_factor;
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        // Every char boundary, including 0 and value.len() (one-past-end).
        for idx in (0..=value.len()).filter(|&i| value.is_char_boundary(i)) {
            let w = self
                .ctx
                .text
                .measure(&value[..idx], font_size * scale, font_size * scale * 1.25)
                .width;
            let dist = (origin_x + w - x).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    /// Restart the caret blink solid-on (after focus / click / edit). Resets
    /// the phase anchor AND forces the caret visible immediately — without
    /// the explicit `set_opacity`, a click landing mid-blink-off wouldn't
    /// repaint the caret until the next full cycle (it'd look unresponsive).
    fn reset_caret_blink(&mut self, field: crate::node::NodeId) {
        self.caret_blink_anchor = Instant::now();
        self.caret_blink_on = true;
        if let Some(cn) = self
            .ctx
            .tree
            .get(field)
            .and_then(|n| n.editor.as_ref())
            .map(|ed| ed.caret_node)
        {
            self.ctx.tree.set_opacity(cn, 1.0);
        }
    }

    /// Place the caret at the click x (physical px) — standard
    /// click-to-position. Sets a selection anchor at the same spot so a
    /// subsequent drag selects from here. No-op on a non-editor node.
    fn place_caret_from_click(&mut self, node_id: crate::node::NodeId, click_x: f32) {
        let Some(idx) = self.caret_index_at_x(node_id, click_x) else {
            return;
        };
        if let Some(ed) = self.ctx.tree.get_mut_raw(node_id).and_then(|n| n.editor.as_mut()) {
            ed.cursor = idx;
            // Anchor here; a plain click leaves an empty (invisible)
            // selection, a drag extends it.
            ed.selection_anchor = Some(idx);
        }
        self.reset_caret_blink(node_id);
        self.ctx.tree.mark_transform_dirty();
        self.react();
    }

    /// Extend the selection in text field `node_id` to the drag x: keeps the
    /// press-time anchor, moves the caret to the boundary nearest x. No-op if
    /// nothing changed or on a non-editor node.
    fn drag_select_to(&mut self, node_id: crate::node::NodeId, x: f32) -> bool {
        let Some(idx) = self.caret_index_at_x(node_id, x) else {
            return false;
        };
        let changed = match self.ctx.tree.get_mut_raw(node_id).and_then(|n| n.editor.as_mut()) {
            Some(ed) if ed.cursor != idx => {
                // First drag motion seeds an anchor if the press didn't.
                ed.selection_anchor.get_or_insert(ed.cursor);
                ed.cursor = idx;
                true
            }
            _ => false,
        };
        if changed {
            self.reset_caret_blink(node_id);
            self.ctx.tree.mark_transform_dirty();
            self.react();
        }
        changed
    }

    /// Lazily init the system clipboard and write `text`. Best-effort:
    /// silently drops the write if arboard can't init (headless / no
    /// display) or the set call fails.
    fn write_clipboard(&mut self, text: String) {
        if self.clipboard.is_none() {
            self.clipboard = arboard::Clipboard::new().ok();
        }
        if let Some(cb) = self.clipboard.as_mut() {
            let _ = cb.set_text(text);
        }
    }

    /// Lazily init the system clipboard and read its text. Returns an
    /// empty string when unavailable.
    fn read_clipboard(&mut self) -> String {
        if self.clipboard.is_none() {
            self.clipboard = arboard::Clipboard::new().ok();
        }
        self.clipboard
            .as_mut()
            .and_then(|cb| cb.get_text().ok())
            .unwrap_or_default()
    }

    /// Rebuild every lazy-list's prefix-sum table if its heights have
    /// changed since last seen. Runs before `compute_layout` so the
    /// layout pass can read `total_height_logical()` for its
    /// `content_size` override and `visible_window` for the materialize
    /// pass below. Each list's `ensure_prefix_fresh` is internally
    /// gated on `heights_version` — idle frames cost nothing.
    fn ensure_lazy_list_prefixes(&mut self) {
        ensure_lazy_list_prefixes(&mut self.ctx);
    }

    /// Walk lazy-list nodes and reconcile their materialized children
    /// with the current visible window. Removes children that have
    /// scrolled off; invokes the render closure once per row newly
    /// in view (or when the list's version was bumped). Returns true
    /// if any list mutated its children — caller (`flush_tree`) uses
    /// this to decide whether a second layout pass is needed.
    fn materialize_lazy_lists(&mut self) -> bool {
        materialize_lazy_lists(&mut self.ctx, &mut self.timeline, self.scale_factor)
    }

    /// Walk every text-field node and write the right display string
    /// into its `text_node`: `placeholder` when value is empty and
    /// the field is not focused, otherwise `value`. Idempotent — does
    /// nothing on idle frames because `tree.set_text` short-circuits
    /// when content matches.
    fn refresh_text_fields(&mut self) {
        let editor_ids: Vec<crate::node::NodeId> = self
            .ctx
            .tree
            .iter_ids()
            .filter(|id| {
                self.ctx
                    .tree
                    .get(*id)
                    .map(|n| n.editor.is_some())
                    .unwrap_or(false)
            })
            .collect();
        for id in editor_ids {
            let (text_node, display, color) = {
                let Some(n) = self.ctx.tree.get(id) else {
                    continue;
                };
                let Some(ed) = n.editor.as_ref() else {
                    continue;
                };
                let focused = n
                    .interact
                    .focused
                    .as_ref()
                    .map(|s| s.get())
                    .unwrap_or(false);
                let show_placeholder =
                    ed.value.is_empty() && !focused && !ed.placeholder.is_empty();
                let (display, color) = if show_placeholder {
                    (ed.placeholder.clone(), ed.placeholder_color)
                } else {
                    (ed.value.clone(), ed.text_color)
                };
                (ed.text_node, display, color)
            };
            self.ctx.tree.set_text(text_node, display);
            self.ctx.tree.set_color(text_node, color);
        }
    }

    /// Walk every text-field node and update its caret child's `rect`
    /// + `visible` based on current value/cursor/focus state. Called
    /// from `flush_tree` after `compute_layout` so the text child's
    /// post-layout bounds are fresh. Caret rect is written **directly**
    /// (bypassing the layout pass) so we don't pay a second layout
    /// cycle per edit.
    fn reposition_carets(&mut self) {
        // Collect node ids first to keep the walk + the mutating
        // updates from aliasing the same &mut tree.
        let editor_ids: Vec<crate::node::NodeId> = self
            .ctx
            .tree
            .iter_ids()
            .filter(|id| {
                self.ctx
                    .tree
                    .get(*id)
                    .map(|n| n.editor.is_some())
                    .unwrap_or(false)
            })
            .collect();
        let scale = self.scale_factor;
        for id in editor_ids {
            // Snapshot every needed read from the editor + sibling
            // nodes before re-borrowing for the write.
            let (
                cursor,
                value,
                font_size,
                text_node,
                caret_node,
                selection_node,
                sel_range,
                focused,
            ) = {
                let Some(n) = self.ctx.tree.get(id) else {
                    continue;
                };
                let Some(ed) = n.editor.as_ref() else {
                    continue;
                };
                let focused = n
                    .interact
                    .focused
                    .as_ref()
                    .map(|s| s.get())
                    .unwrap_or(false);
                (
                    ed.cursor,
                    ed.value.clone(),
                    ed.font_size,
                    ed.text_node,
                    ed.caret_node,
                    ed.selection_node,
                    crate::editor::selection_range(ed),
                    focused,
                )
            };
            // Measure helper in *physical* px (scaled font size + line
            // height), matching how the layout pass shaped the text.
            let measure_w = |text: &mut crate::text::TextResources, upto: usize| {
                text.measure(
                    &value[..upto.min(value.len())],
                    font_size * scale,
                    font_size * scale * 1.25,
                )
                .width
            };
            let prefix_w = measure_w(&mut self.ctx.text, cursor);
            let sel_px = sel_range.map(|(lo, hi)| {
                (
                    measure_w(&mut self.ctx.text, lo),
                    measure_w(&mut self.ctx.text, hi),
                )
            });
            let text_rect = self
                .ctx
                .tree
                .get(text_node)
                .map(|n| n.rect)
                .unwrap_or([0.0; 4]);
            if let Some(c) = self.ctx.tree.get_mut_raw(caret_node) {
                // Caret rect is absolute (screen-space, physical px).
                // x = text origin + prefix width; y/h = match text.
                c.rect = [
                    text_rect[0] + prefix_w,
                    text_rect[1],
                    2.0 * scale,
                    text_rect[3],
                ];
                c.visible = focused;
            }
            // Selection highlight: span from start-width to end-width,
            // hidden when there's no selection.
            if let Some(s) = self.ctx.tree.get_mut_raw(selection_node) {
                match sel_px {
                    Some((lo_w, hi_w)) => {
                        s.rect = [
                            text_rect[0] + lo_w,
                            text_rect[1],
                            (hi_w - lo_w).max(0.0),
                            text_rect[3],
                        ];
                        s.visible = true;
                    }
                    None => s.visible = false,
                }
            }
        }
    }

    /// Defensively end any in-flight scrollbar drag. Mirrors the
    /// release-path cleanup: applies the last pending cursor (so the
    /// final pixel of scroll lands) and calls `end_drag` to clear the
    /// `bar_active` flag and retarget for snap/bounce. Invoked when
    /// the OS can no longer guarantee we'll see the matching
    /// MouseInput::Released event — window loses focus, cursor exits
    /// the client area without an active capture, etc.
    fn end_bar_drag_if_active(&mut self) {
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
            self.react();
        }
    }

    /// Blur a focused text field: flip its focused signal to false and
    /// clear `input.focused` so subsequent keys don't route. Triggers
    /// a react so the caret repaints invisible.
    fn blur_text_field(&mut self, node_id: crate::node::NodeId) {
        if let Some(sig) = self
            .ctx
            .tree
            .get(node_id)
            .and_then(|n| n.interact.focused.clone())
        {
            sig.set(false);
        }
        if self.input.focused == Some(node_id) {
            self.input.focused = None;
        }
        // Caret visibility depends on the focused signal — mark
        // transform dirty so reposition_carets runs and hides it.
        self.ctx.tree.mark_transform_dirty();
        self.react();
    }

    /// Move focus to `target` (or clear it with `None`). Flips the old
    /// node's `interact.focused` signal off and the new one's on,
    /// updates `input.focused`, and reacts. Invariant preserved: at most
    /// one node's focused signal is true at a time. No-op if focus is
    /// already where requested.
    fn set_focus(&mut self, target: Option<crate::node::NodeId>) {
        if self.input.focused == target {
            return;
        }
        if let Some(prev) = self.input.focused
            && let Some(sig) = self
                .ctx
                .tree
                .get(prev)
                .and_then(|n| n.interact.focused.clone())
        {
            sig.set(false);
        }
        if let Some(next) = target
            && let Some(sig) = self
                .ctx
                .tree
                .get(next)
                .and_then(|n| n.interact.focused.clone())
        {
            sig.set(true);
        }
        self.input.focused = target;
        // Restart the blink solid-on so the caret is visible the instant the
        // field gains focus (and any prior field's faded-out caret resets).
        if let Some(next) = target {
            self.reset_caret_blink(next);
        }
        // A focused text field repositions/show its caret in
        // reposition_carets, which gates on TRANSFORM.
        self.ctx.tree.mark_transform_dirty();
        self.react();
    }

    /// Fire [`App::on_unhandled_press`] when the just-captured left-press
    /// target is "outside" any floating layer — i.e. there is no captured
    /// node (press hit empty space) or the captured node is tagged
    /// [`crate::node::Node::dismiss_transparent`] (a scrim). Take-then-
    /// restore so the hook can re-enter `&mut self.ctx`. No-op without a
    /// registered hook.
    fn maybe_fire_unhandled_press(&mut self) {
        let unhandled = match self.input.captured {
            None => true,
            Some(id) => self
                .ctx
                .tree
                .get(id)
                .map(|n| n.dismiss_transparent)
                .unwrap_or(false),
        };
        if unhandled && let Some(mut hook) = self.on_unhandled_press.take() {
            hook(&mut self.ctx);
            self.on_unhandled_press = Some(hook);
            self.react();
        }
    }

    /// On a fresh left-press, latch drag state if the captured node is
    /// draggable (has `on_drag` and/or a `drag_payload`). Records the
    /// press origin + last-fire cursor and latches a clone of any
    /// payload as the in-flight drag. No-op for non-draggable targets.
    fn begin_drag_if_draggable(&mut self) {
        let Some(cap) = self.input.captured else {
            return;
        };
        let Some(origin) = self.input.cursor else {
            return;
        };
        let (has_drag, payload, follow) = self
            .ctx
            .tree
            .get(cap)
            .map(|n| (n.on_drag.is_some(), n.drag_payload.clone(), n.drag_follow))
            .unwrap_or((false, None, false));
        if has_drag || payload.is_some() || follow {
            // A fresh grab cancels any in-flight snap-back so the new
            // drag owns the follow state.
            if self.drag_return.is_some() {
                self.timeline.stop(DRAG_RETURN_TWEEN_KEY);
                self.drag_return = None;
            }
            self.drag_origin = Some(origin);
            self.drag_last = Some(origin);
            self.drag_node = if has_drag { Some(cap) } else { None };
            self.drag_payload = payload;
            // Lift the node immediately (zero offset) so it's already on
            // top before the first move.
            if follow {
                self.ctx.tree.set_drag_follow(Some((cap, [0.0, 0.0])));
            }
            // Click-to-set: fire `on_drag` once at the press position so a
            // plain click (no drag) jumps the value. `delta` is zero here
            // (drag_last == origin); absolute handlers use `current`,
            // delta handlers no-op. Drives slider/scrubber click-to-seek.
            if has_drag {
                self.fire_drag(origin[0], origin[1]);
                self.react();
            }
        }
    }

    /// Update the drag-follow offset for the captured node (if it opted
    /// into `drag_follow`) to track the cursor. Returns true if a follow
    /// is active (caller should react so the ghost re-flattens).
    fn update_drag_follow(&mut self, x: f32, y: f32) -> bool {
        let Some(cap) = self.input.captured else {
            return false;
        };
        let Some(origin) = self.drag_origin else {
            return false;
        };
        let follows = self
            .ctx
            .tree
            .get(cap)
            .map(|n| n.drag_follow)
            .unwrap_or(false);
        if follows {
            self.ctx
                .tree
                .set_drag_follow(Some((cap, [x - origin[0], y - origin[1]])));
        }
        follows
    }

    /// Fire the captured node's `on_drag` (if any) for a cursor move to
    /// `[x, y]`. `DragCtx::delta` is relative to the previous fire.
    /// Returns true if a handler ran (caller should react).
    fn fire_drag(&mut self, x: f32, y: f32) -> bool {
        let Some(cap) = self.input.captured else {
            return false;
        };
        let Some(start) = self.drag_origin else {
            return false;
        };
        let Some(h) = self.ctx.tree.get(cap).and_then(|n| n.on_drag.clone()) else {
            return false;
        };
        let last = self.drag_last.unwrap_or(start);
        let rect = self
            .hits
            .iter()
            .find(|e| e.node_id == cap)
            .map(|e| {
                let b = e.bounds;
                [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            })
            .unwrap_or([0.0; 4]);
        let scale = self.scale_factor;
        let mut dctx = crate::event::DragCtx {
            tree: &mut self.ctx.tree,
            node: cap,
            start,
            current: [x, y],
            delta: [x - last[0], y - last[1]],
            rect,
            scale,
        };
        h(&mut dctx);
        self.drag_last = Some([x, y]);
        true
    }

    /// Fire the hovered (un-captured) node's `on_hover_move` for a cursor
    /// move to `[x, y]`. No-op while a press is captured (that's the drag
    /// path). Returns true if a handler ran (caller should react).
    fn fire_hover_move(&mut self, x: f32, y: f32) -> bool {
        if self.input.captured.is_some() {
            return false;
        }
        let Some(hovered) = self.input.hovered else {
            return false;
        };
        let Some(h) = self
            .ctx
            .tree
            .get(hovered)
            .and_then(|n| n.on_hover_move.clone())
        else {
            return false;
        };
        let rect = self
            .hits
            .iter()
            .find(|e| e.node_id == hovered)
            .map(|e| {
                let b = e.bounds;
                [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            })
            .unwrap_or([0.0; 4]);
        let scale = self.scale_factor;
        let mut hctx = crate::event::HoverCtx {
            tree: &mut self.ctx.tree,
            node: hovered,
            pos: [x, y],
            rect,
            scale,
        };
        h(&mut hctx);
        true
    }

    /// Fire the topmost `on_wheel` handler under the cursor, consuming the
    /// wheel tick (no scroll routing). `delta` is in wheel lines, positive
    /// y = wheel forward/up. Returns true if a handler ran. Entries
    /// without a wheel handler are skipped (a hoverable row above a
    /// slider doesn't swallow the tick), so only nodes that opted in
    /// intercept wheel input.
    fn fire_wheel(&mut self, x: f32, y: f32, delta: [f32; 2]) -> bool {
        let Some((node, rect, h)) = self.hits.iter().find_map(|e| {
            if !e.contains(x, y) {
                return None;
            }
            let h = self.ctx.tree.get(e.node_id).and_then(|n| n.on_wheel.clone())?;
            let b = e.bounds;
            Some((e.node_id, [b[0], b[1], b[2] - b[0], b[3] - b[1]], h))
        }) else {
            return false;
        };
        let scale = self.scale_factor;
        let mut wctx = crate::event::WheelCtx {
            tree: &mut self.ctx.tree,
            timeline: &mut self.timeline,
            now: std::time::Instant::now(),
            node,
            delta,
            rect,
            scale,
        };
        h(&mut wctx);
        true
    }

    /// On left-release, deliver any in-flight drag payload to the topmost
    /// drop target under the cursor, then animate a lifted node back to
    /// its resting slot. No-op when nothing was being dragged.
    fn finish_drag_on_release(&mut self) {
        // Fire the drag node's `on_drag_end` (commit-on-release sliders).
        // Capture was already cleared by `on_left_released`, so use the
        // remembered drag node.
        if let Some(dn) = self.drag_node.take()
            && let Some(h) = self.ctx.tree.get(dn).and_then(|n| n.on_drag_end.clone())
        {
            let mut ectx = crate::event::EventCtx {
                tree: &mut self.ctx.tree,
                timeline: &mut self.timeline,
                node: dn,
                now: Instant::now(),
                cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
            };
            h(&mut ectx);
        }
        let was_dragging = self.drag_payload.is_some() || self.drag_origin.is_some();
        // 1. Deliver the payload to a drop target under the cursor.
        if let Some(payload) = self.drag_payload.take()
            && let Some([x, y]) = self.input.cursor
        {
            let tree = &self.ctx.tree;
            let target = self
                .hits
                .iter()
                .find(|h| {
                    h.contains(x, y)
                        && tree
                            .get(h.node_id)
                            .map(|n| n.on_drop.is_some())
                            .unwrap_or(false)
                })
                .map(|h| h.node_id);
            if let Some(t) = target
                && let Some(h) = self.ctx.tree.get(t).and_then(|n| n.on_drop.clone())
            {
                let mut dctx = crate::event::DropCtx {
                    tree: &mut self.ctx.tree,
                    node: t,
                    payload,
                };
                h(&mut dctx);
            }
        }
        // 2. Snap-back: if a node was lifted (drag_follow), animate its
        //    offset from where it was dropped back to [0,0] rather than
        //    popping it into place. The per-frame `tick_drag_return`
        //    mirrors the tween into the tree and clears the lift when it
        //    lands. Without a follow node there's nothing to animate.
        let now = Instant::now();
        if let (Some(node), Some(origin), Some([x, y])) = (
            self.ctx.tree.drag_follow_target(),
            self.drag_origin,
            self.input.cursor,
        ) {
            let from = [x - origin[0], y - origin[1]];
            if from[0].abs() < 0.5 && from[1].abs() < 0.5 {
                // Barely moved (e.g. a plain click) — nothing to animate;
                // a from==to tween would be a no-op and leave the lift
                // stuck, so drop it back immediately.
                self.ctx.tree.set_drag_follow(None);
            } else {
                self.drag_return_offset.set(from);
                self.timeline.start(
                    DRAG_RETURN_TWEEN_KEY,
                    self.drag_return_offset.clone(),
                    [0.0, 0.0],
                    crate::anim::Curve::EaseInOut,
                    std::time::Duration::from_millis(DRAG_RETURN_MS),
                    now,
                );
                self.drag_return = Some(node);
                // Hold the ghost at its drop position for this frame; the
                // loop animates it home from here.
                self.ctx.tree.set_drag_follow(Some((node, from)));
            }
        }
        self.drag_origin = None;
        self.drag_last = None;
        self.drag_payload = None;
        if was_dragging {
            self.react();
        }
    }

    /// Per-frame step of the drag snap-back. Mirrors the tweened
    /// `drag_return_offset` into the tree's drag-follow offset; once it
    /// lands at the slot (≈ zero), drops the lift entirely. Called from
    /// `about_to_wait` while the return tween is active.
    fn tick_drag_return(&mut self) {
        if let Some(node) = self.drag_return {
            let off = self.drag_return_offset.get();
            if off[0].abs() < 0.5 && off[1].abs() < 0.5 {
                self.ctx.tree.set_drag_follow(None);
                self.drag_return = None;
            } else {
                self.ctx.tree.set_drag_follow(Some((node, off)));
            }
        }
    }

    /// Drop drag bookkeeping without delivering (capture loss — cursor
    /// left, window blur). The payload is discarded.
    fn clear_drag_state(&mut self) {
        self.drag_origin = None;
        self.drag_last = None;
        self.drag_node = None;
        self.drag_payload = None;
        self.timeline.stop(DRAG_RETURN_TWEEN_KEY);
        self.drag_return = None;
        self.ctx.tree.set_drag_follow(None);
    }

    /// Fully reset pointer interaction on capture loss — the cursor left
    /// the window or focus moved, so the button's eventual release will
    /// never reach us. Fires the active drag's `on_drag_end` (so app-side
    /// drag state, e.g. a seek-bar `seeking` signal, resets instead of
    /// latching on), settles scrollbar drag, drops generic-drag
    /// bookkeeping, and clears capture + every pressed/hover signal.
    /// Returns true if anything changed (caller should react).
    fn cancel_pointer_interaction(&mut self) -> bool {
        // Fire on_drag_end so the app can commit/reset a held scrub even
        // though we won't see the button release.
        if let Some(dn) = self.drag_node.take()
            && let Some(h) = self.ctx.tree.get(dn).and_then(|n| n.on_drag_end.clone())
        {
            let mut ectx = crate::event::EventCtx {
                tree: &mut self.ctx.tree,
                timeline: &mut self.timeline,
                node: dn,
                now: Instant::now(),
                cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
            };
            h(&mut ectx);
        }
        self.end_bar_drag_if_active();
        self.clear_drag_state();
        let bar_changed =
            crate::input::update_scrollbar_hover(None, &self.scroll_bars, &mut self.ctx.tree);
        let change = self.input.cancel(&self.hits, &self.ctx.tree);
        self.dwell = None;
        self.clear_hint();
        if self.last_cursor_icon != CursorIcon::Default {
            self.last_cursor_icon = CursorIcon::Default;
            if let Some(w) = &self.window {
                w.set_cursor(CursorIcon::Default);
            }
        }
        change.any() || bar_changed
    }

    /// Advance keyboard focus to the next (`forward = true`, Tab) or
    /// previous (`forward = false`, Shift+Tab) focusable node. Focusable
    /// = `layout.focus_order != 0`, `visible`, and the node's rect
    /// intersects the viewport. Visited in ascending `focus_order`, ties
    /// broken by creation order (stable). Wraps around the ends; when
    /// nothing is currently focused, Tab lands on the first candidate and
    /// Shift+Tab on the last. No-op when no node is focusable.
    fn focus_next(&mut self, forward: bool) {
        let [vw, vh] = self.viewport();
        let mut cands: Vec<(u32, crate::node::NodeId)> = Vec::new();
        for id in self.ctx.tree.iter_ids() {
            let Some(n) = self.ctx.tree.get(id) else {
                continue;
            };
            let order = n.layout.focus_order;
            if order == 0 || !n.visible {
                continue;
            }
            // Viewport intersection: rect is [x, y, w, h] in physical px.
            let r = n.rect;
            let onscreen = r[0] < vw && r[1] < vh && r[0] + r[2] > 0.0 && r[1] + r[3] > 0.0;
            if !onscreen {
                continue;
            }
            cands.push((order, id));
        }
        if cands.is_empty() {
            return;
        }
        // Stable sort keeps iter_ids (creation) order for equal orders.
        cands.sort_by_key(|(order, _)| *order);
        let pos = self
            .input
            .focused
            .and_then(|cur| cands.iter().position(|(_, id)| *id == cur));
        let len = cands.len();
        let next = match pos {
            Some(i) if forward => (i + 1) % len,
            Some(i) => (i + len - 1) % len,
            None if forward => 0,
            None => len - 1,
        };
        self.set_focus(Some(cands[next].1));
    }

    /// Drop every root + descendant and re-run the stored scene
    /// builder against the cleared `SceneCtx`. The canonical way to
    /// swap views — caller flips whatever signal the closure reads,
    /// then calls this method to materialize the new tree.
    ///
    /// **Side effects beyond the tree:** clears `InputState`
    /// (hovered/captured/focused — old `NodeId`s no longer exist) and
    /// stops every active tween whose slot got tombstoned. Staged
    /// images, GPU state, and the timeline's *idle* state (resting
    /// signals not under an active tween) are preserved.
    ///
    /// No-op if [`App::scene`] was never called.
    pub fn rebuild_scene(&mut self) {
        // 0. Snapshot EVERY scroller's offset (named or anonymous). A
        //    rebuild is often additive UI (a modal opening, a hot-patch)
        //    — teleporting any list back to the top would lose the
        //    user's place. Restored (clamped against the fresh layout)
        //    after the new tree flushes. Identity is [`scroll_identity`]:
        //    the node's name when named, else its structural path — so
        //    preservation is the DEFAULT, and a page that must reset on
        //    navigation opts out by scoping its name to its content
        //    (e.g. `detail_scroll:<id>` — new content ⇒ new identity ⇒
        //    fresh scroll).
        let saved_scroll: Vec<(String, [f32; 2])> = self
            .ctx
            .tree
            .scrollables()
            .to_vec()
            .into_iter()
            .filter_map(|id| {
                let off = self.ctx.tree.scroll_offset(id);
                (off != [0.0, 0.0])
                    .then(|| scroll_identity(&self.ctx, id).map(|key| (key, off)))
                    .flatten()
            })
            .collect();
        // 0b. Snapshot every `.external()` node's identity → current id. An
        //    external node (Canvas video) gets a fresh NodeId each rebuild;
        //    its GPU resident frame set is keyed by that id, so without
        //    carrying it across the new (empty) node paints one blank frame —
        //    a visible canvas flicker on every view change / modal toggle.
        //    Same identity scheme as the scroll snapshot above; restored by
        //    migrating the frame set onto the reincarnated id before flush.
        let saved_external: Vec<(String, crate::node::NodeId)> = self
            .ctx
            .tree
            .iter_ids()
            .filter(|&id| self.ctx.tree.get(id).map(|n| n.external).unwrap_or(false))
            .filter_map(|id| scroll_identity(&self.ctx, id).map(|key| (key, id)))
            .collect();
        // 1. Snapshot the root list, then drop each subtree. Iterating
        //    `tree.roots()` directly while removing would alias the
        //    same Vec.
        let roots: Vec<crate::node::NodeId> = self.ctx.tree.roots().to_vec();
        for r in roots {
            let removal = self.ctx.remove_subtree(r);
            stop_tweens_for_removal(&mut self.timeline, &removal);
        }
        // 2. Reset pointer state — old NodeIds in `captured` / `hovered` /
        //    `focused` would be stale generation matches at best,
        //    silent miss-fires at worst. Cursor *position* is the only
        //    field worth preserving: re-derive hover against the fresh
        //    hit cache below so the new tree's hover signals settle on
        //    the same node the user is still pointing at. Without this,
        //    every rebuild (e.g. a 5 Hz progress tick) would briefly
        //    drop hover and let any subsequent spurious CursorMoved
        //    flip it back on — visible as a hover-flicker on stationary
        //    buttons.
        let preserved_cursor = self.input.cursor;
        self.input = InputState::new();
        // Stale dwell tracker may point at a destroyed node id.
        self.dwell = None;
        // A shown hint references the old node id/rect — drop it. If the
        // cursor still rests on a hint node, the re-derived hover re-arms the
        // dwell and it reappears after the delay.
        self.clear_hint();
        // Drag bookkeeping references destroyed node ids — drop it.
        self.clear_drag_state();
        // Layer-opacity / layer-offset bindings hold node ids from the old
        // tree — drop them; the rebuilt scene re-collects them on next flush.
        self.layer_opacity_binds.clear();
        self.layer_offset_x_binds.clear();
        // 3. Re-invoke the stored builder on the now-empty ctx. Take
        //    + put back so the closure can call back into &mut self
        //    via captures without re-borrowing the builder slot.
        let Some(mut builder) = self.scene_builder.take() else {
            return;
        };
        {
            let mut scene = Scene::root(&mut self.ctx);
            builder(&mut scene);
        }
        self.scene_builder = Some(builder);
        // 3b. Carry resident external-frame sets (Canvas video) onto their
        //     reincarnated node ids, synchronously here — BEFORE flush/render
        //     so the rebuilt external node samples its video on the very
        //     first paint instead of flashing empty. A later, stale `migrate`
        //     from the decode thread (it still holds the old id until the
        //     next `sync_node`) lands as a harmless no-op (set already moved).
        if !saved_external.is_empty() {
            let saved: std::collections::HashMap<String, crate::node::NodeId> =
                saved_external.into_iter().collect();
            let pairs: Vec<(crate::node::NodeId, crate::node::NodeId)> = self
                .ctx
                .tree
                .iter_ids()
                .filter(|&id| self.ctx.tree.get(id).map(|n| n.external).unwrap_or(false))
                .filter_map(|id| {
                    let key = scroll_identity(&self.ctx, id)?;
                    saved.get(&key).map(|&old| (old, id))
                })
                .collect();
            if let Some(gpu) = self.gpu.as_mut() {
                for (old, new) in pairs {
                    gpu.migrate_external_frames(old, new);
                }
            }
        }
        // 4. Flush + redraw. flush_tree returns true if anything
        //    changed; rebuild always changes everything.
        if self.flush_tree() {
            self.request_redraw();
        }
        // 4b. Restore the scroll snapshot now that layout ran (clamping
        //     needs the fresh content sizes), then flush again so lazy
        //     windows re-materialize at the restored offset before the
        //     first paint instead of flashing the top of the list.
        //     Identities are recomputed against the NEW tree and matched
        //     to the snapshot — keys are computed first (shared borrow),
        //     then applied (mutable borrow).
        if !saved_scroll.is_empty() {
            let saved: std::collections::HashMap<String, [f32; 2]> =
                saved_scroll.into_iter().collect();
            let matches: Vec<(crate::node::NodeId, [f32; 2])> = self
                .ctx
                .tree
                .scrollables()
                .to_vec()
                .into_iter()
                .filter_map(|id| {
                    let key = scroll_identity(&self.ctx, id)?;
                    saved.get(&key).map(|&off| (id, off))
                })
                .collect();
            let restored = !matches.is_empty();
            for (id, off) in matches {
                self.ctx.tree.restore_scroll(id, off);
            }
            if restored && self.flush_tree() {
                self.request_redraw();
            }
        }
        // 5. Re-derive hover from the preserved cursor against the
        //    freshly-flattened hit cache. Mirrors the WindowEvent::
        //    CursorMoved path minus the cursor-icon refresh (no new
        //    OS-level cursor info to honor).
        if let Some([x, y]) = preserved_cursor {
            let _ = crate::input::update_scrollbar_hover(
                Some([x, y]),
                &self.scroll_bars,
                &mut self.ctx.tree,
            );
            let change = self.input.on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
            // Re-deriving hover set the hovered node's signal, dirtying the
            // tree AFTER the flush above — so the just-flattened instances
            // still show that node un-hovered. Re-flush now so the restored
            // hover lands in the instance buffer before the next paint.
            // Without this, a 60 fps external (Canvas) `render_once` paints
            // the one un-hovered frame, reading as a hover flicker on every
            // rebuild (the canvas makes the otherwise-invisible gap visible).
            if change.any() && self.flush_tree() {
                self.request_redraw();
            }
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

    /// Cursor for the topmost interactive node under `(x, y)`: an explicit
    /// [`crate::scene::NodeBuilderRef::cursor`] if set, otherwise
    /// [`CursorIcon::Pointer`] for any node carrying an `on_click` — so
    /// every clickable element presents the clickable affordance without
    /// each call site having to tag a cursor by hand. Hits are stored
    /// topmost-first (after the flatten reverse), so the first match wins;
    /// an explicit cursor on a node above wins over a click below it.
    fn hovered_node_cursor(&self, x: f32, y: f32) -> Option<CursorIcon> {
        for entry in &self.hits {
            if !entry.contains(x, y) {
                continue;
            }
            let Some(n) = self.ctx.tree.get(entry.node_id) else {
                continue;
            };
            if let Some(c) = n.cursor {
                return Some(c);
            }
            if n.on_click.is_some() {
                return Some(CursorIcon::Pointer);
            }
        }
        None
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
                WindowAction::Close | WindowAction::Minimize | WindowAction::ToggleMaximize => {
                    CursorIcon::Pointer
                }
            }
        } else if let Some(c) = self.hovered_node_cursor(x, y) {
            c
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
        let Some(win) = &self.window else {
            return false;
        };
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
        if self.gpu.is_none() {
            return;
        }
        // Wait for any previous F2 capture to finish before starting a
        // new one. Bounded by encode-rate (~100 ms with Fast compression)
        // so spam-pressing F2 stays responsive without overlapping
        // threads racing on the file system.
        self.flush_pending_screenshot();
        let gpu = self.gpu.as_mut().expect("gpu just checked");
        let (rgba, w, h) = gpu.capture_rgba();
        let path = debug::screenshot_path(&self.config.capture_dir);
        let stats = self
            .last_render_stats
            .unwrap_or_else(|| self.current_stats());
        debug::write_stats_sidecar(&path, &stats);
        // Encode + write off-thread so F2 in the middle of an
        // interactive session doesn't freeze winit for 1–5 s. Sidecar
        // is small and inline; the PNG is the expensive part.
        self.pending_screenshot = Some(debug::save_png_async(path, rgba, w, h));
    }

    /// Block until the most recent async screenshot finishes encoding
    /// + writing. Called from exit paths so a fast-exit-after-capture
    /// (autocapture deadline, exiting()) doesn't kill the encoder
    /// mid-write and leave a truncated PNG.
    fn flush_pending_screenshot(&mut self) {
        if let Some(handle) = self.pending_screenshot.take() {
            let _ = handle.join();
        }
    }
}

/// Walk the tree for every node referencing an `ImageHandle` (i.e.
/// `ShapeKind::Image` with a `Some(handle)`). Returned set is used as
/// the `live` argument to [`crate::gpu::ImageAtlas::rebuild_keeping`]
/// so eviction drops sources for handles no longer in any node.
pub fn collect_live_image_handles(
    tree: &crate::node::NodeTree,
) -> std::collections::HashSet<crate::gpu::ImageHandle> {
    let mut out = std::collections::HashSet::new();
    for id in tree.iter_ids() {
        if let Some(n) = tree.get(id)
            && let Some(h) = n.image
        {
            out.insert(h);
        }
    }
    out
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
    event_inst_start: &mut Vec<u32>,
) -> u32 {
    use crate::gpu::{SHAPE_KIND_GLASS, SHAPE_KIND_MASK};
    let mut glass_count: u32 = 0;
    // Prefix map event index → first instance index it produces, so a
    // promoted subtree's event range maps to an instance range (events
    // expand 1:N for text/images). `event_inst_start[events.len()]` is
    // the total instance count.
    event_inst_start.clear();
    event_inst_start.reserve(events.len() + 1);
    for event in events {
        event_inst_start.push(out.len() as u32);
        match event {
            FlatEvent::Shape(s) => {
                let mut s = *s;
                let kind = s.shape_kind & SHAPE_KIND_MASK;
                if kind == SHAPE_KIND_GLASS {
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
                if kind == SHAPE_KIND_GLASS {
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
                // rasterizes at on-screen resolution. max_width is
                // logical too — scale it alongside so the constrained
                // shape runs at physical px against a physical limit.
                let mut r = r.clone();
                r.font_size *= scale;
                r.line_height *= scale;
                r.max_width = r.max_width.map(|w| w * scale);
                let glyphs = gpu.build_glyph_instances(text, std::slice::from_ref(&r));
                out.extend(glyphs);
            }
        }
    }
    event_inst_start.push(out.len() as u32);
    glass_count
}

/// Convert a CPU `Layer` into a GPU `LayerDraw`. A scroll layer (`window`
/// = `Some`) maps to a [`crate::gpu::ScrollWindow`] (quad at the viewport,
/// sampling the tall texture at the scroll offset, scissor-clipped to the
/// viewport); a plain layer uses the full-surface identity path with its
/// composite offset/scale/opacity.
fn layer_to_draw(l: &crate::layer::Layer) -> crate::gpu::LayerDraw {
    let window = l.window.map(|s| crate::gpu::ScrollWindow {
        dst_origin: s.viewport_origin,
        dst_size: s.viewport,
        // Sample origin in *texture* space: scroll offset minus the
        // texture's content-top. `0` for a bounded scroller (tex_origin
        // = 0); for a windowed lazy list this lands the visible window at
        // the right texel since the texture only holds the materialized
        // rows starting at `tex_origin`.
        src_origin: [s.scroll[0] - s.tex_origin[0], s.scroll[1] - s.tex_origin[1]],
        tex_size: s.content,
        clip_rect: [
            s.viewport_origin[0],
            s.viewport_origin[1],
            s.viewport_origin[0] + s.viewport[0],
            s.viewport_origin[1] + s.viewport[1],
        ],
    });
    // External-texture layer: composite the caller's texture into the
    // node's screen rect (a full 1:1 blit, no clip). `window` carries the
    // rect; `external` carries the registry key (`root` node).
    let (window, external, corner_radius) = match l.external {
        Some(ext) => {
            let w = crate::gpu::ScrollWindow {
                dst_origin: ext.origin,
                dst_size: ext.size,
                src_origin: [0.0, 0.0],
                tex_size: ext.size,
                clip_rect: ext.clip,
            };
            (Some(w), l.root, ext.radius)
        }
        None => (window, None, 0.0),
    };
    crate::gpu::LayerDraw {
        instances: (l.instances.start as u32)..(l.instances.end as u32),
        offset: l.offset,
        scale: l.scale,
        opacity: l.opacity,
        z: l.z,
        window,
        external,
        corner_radius,
        round_rect: None,
        edge_fade: l.edge_fade,
        edge_fade_falloff: if l.edge_fade_falloff > 0.0 {
            l.edge_fade_falloff
        } else {
            1.0
        },
    }
}

/// Map promoted-subtree event spans to instance ranges using the
/// `event_inst_start` prefix produced by [`expand_events_into`]. Carries
/// the optional scroll-window geometry through unchanged.
fn spans_to_instance_ranges(
    spans: &[crate::node::LayerSpan],
    event_inst_start: &[u32],
) -> Vec<crate::layer::PromotedRange> {
    let mut ranges: Vec<crate::layer::PromotedRange> = spans
        .iter()
        .filter_map(|s| {
            let start = *event_inst_start.get(s.events.start)?;
            // External layers own no instances: a zero-width range at the
            // paint cursor `start` fixes their z without consuming the
            // stream. (`events` is empty, so the `end > start` guard below
            // would otherwise drop them.)
            if let Some(ext) = s.external {
                return Some(crate::layer::PromotedRange {
                    node: s.node,
                    instances: start..start,
                    scroll: None,
                    external: Some(ext),
                    edge_fade: s.edge_fade,
                    edge_fade_falloff: s.edge_fade_falloff,
                });
            }
            let end = *event_inst_start.get(s.events.end)?;
            (end > start).then_some(crate::layer::PromotedRange {
                node: s.node,
                instances: start..end,
                scroll: s.scroll,
                external: None,
                edge_fade: s.edge_fade,
                edge_fade_falloff: s.edge_fade_falloff,
            })
        })
        .collect();
    // `LayerTree::rebuild` walks a forward cursor, so promoted ranges must
    // be in ascending instance order. Flatten emits a scroll container's
    // promoted-thumb span *before* its content span (the thumb's quad is
    // emitted after the children in the stream, but `emit_scrollbars` runs
    // before the content-span close), so the two arrive out of order —
    // sort to restore painter order. Ranges are non-overlapping, so a
    // start-key sort is total.
    ranges.sort_by_key(|r| r.instances.start);
    ranges
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

    let lines: [String; 9] = [
        format!("cpu  {:>5.2} ms", stats.cpu_ms),
        format!("gpu  {:>5.2} ms", stats.gpu_ms),
        format!("opq  {:>5.2} ms", stats.opaque_ms),
        format!("fnl  {:>5.2} ms", stats.final_ms),
        format!("inst {:>5}", stats.instance_count),
        format!("draw {:>5}", stats.drawcalls),
        format!("lyr  {:>5}", stats.layer_count),
        format!(
            "rast {:>3} cmp {:>1}",
            stats.raster_count, stats.composite_count
        ),
        format!("vram {:>4} KiB", stats.layer_vram / 1024),
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
            max_width: None,
            clip_rect: crate::gpu::NO_CLIP,
        })
        .collect();
    out.extend(gpu.build_glyph_instances(text, &refs));
    out
}

/// Build a `hover_hint` tooltip (a rounded dark pill + its label) near the
/// `cursor` (physical px): centred horizontally on the pointer and sitting
/// just above it (top-middle). Near a side edge it slides in to stay fully
/// on-screen (so it ends up top-left / top-right there), and it drops below
/// the cursor only when there isn't room above.
fn build_hint_instances(
    text: &str,
    cursor: [f32; 2],
    surface: [f32; 2],
    fade: f32,
    gpu: &mut crate::gpu::GpuContext,
    text_res: &mut crate::text::TextResources,
    scale: f32,
) -> Vec<ShapeInstance> {
    use crate::node::TextRef;

    let font_size = 12.5 * scale;
    let line_h = 17.0 * scale;
    let pad_x = 9.0 * scale;
    let pad_y = 5.0 * scale;
    let radius = 6.0 * scale;
    // Small vertical gap between the pill and the pointer (kept tight so the
    // flipped-below pill doesn't sit far from the cursor).
    let gap_y = 6.0 * scale;

    let m = text_res.measure(text, font_size, line_h);
    let box_w = m.width + pad_x * 2.0;
    let box_h = line_h + pad_y * 2.0;

    // Centred on the cursor; the clamp below slides it in near a side edge so
    // it stays fully visible (becoming top-left / top-right only there).
    let mut x = cursor[0] - box_w / 2.0;
    // Above the cursor by `gap_y`; drop just below it only when there isn't
    // room above (so it never floats far overhead).
    let mut y = cursor[1] - gap_y - box_h;
    if y < 0.0 {
        y = cursor[1] + gap_y;
    }
    x = x.clamp(0.0, (surface[0] - box_w).max(0.0));
    y = y.clamp(0.0, (surface[1] - box_h).max(0.0));

    let fade = fade.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(1 + text.len());
    out.push(ShapeInstance {
        color: [0.04, 0.04, 0.05, 0.96 * fade],
        border_color: [1.0, 1.0, 1.0, 0.10 * fade],
        border_width: 1.0 * scale,
        position: [x, y],
        size: [box_w, box_h],
        border_radius: [radius; 4],
        ..Default::default()
    });
    let refs = [TextRef {
        position: [x + pad_x, y + pad_y],
        color: [0.96, 0.96, 0.97, 1.0],
        opacity: fade,
        content: text.to_string(),
        font_size,
        line_height: line_h,
        max_width: None,
        clip_rect: crate::gpu::NO_CLIP,
    }];
    out.extend(gpu.build_glyph_instances(text_res, &refs));
    out
}

/// Stable identity of a scroll node across scene rebuilds — what lets
/// [`App::rebuild_scene`] hand a destroyed scroller's offset to its
/// reincarnation in the new tree.
///
/// A named node IS its name (precise, and the app's opt-out lever: a
/// content-scoped name like `detail_scroll:<id>` changes identity with
/// the content, resetting scroll on navigation). An anonymous node is
/// its nearest named ancestor plus the child-index path down from it —
/// structural identity, so anonymous scrollers (a sidebar, a settings
/// list) survive rebuilds by default. Best-effort by construction: a
/// conditional sibling appearing *before* an anonymous scroller shifts
/// its path and the offset resets (never mis-corrupts — name the
/// scroller when its position isn't structurally stable).
fn scroll_identity(ctx: &SceneCtx, id: crate::node::NodeId) -> Option<String> {
    // Reverse name lookup. Built per call — rebuilds are rare, one-shot
    // events and the maps are small; precise beats cached-and-stale.
    let named: std::collections::HashMap<crate::node::NodeId, &str> =
        ctx.names.iter().map(|(k, &v)| (v, k.as_str())).collect();
    ctx.tree.get(id)?;
    let mut segments: Vec<String> = Vec::new();
    let mut cur = id;
    loop {
        if let Some(name) = named.get(&cur) {
            segments.push((*name).to_string());
            break;
        }
        match ctx.tree.parent(cur) {
            Some(p) => {
                let idx = ctx
                    .tree
                    .get(p)?
                    .children
                    .iter()
                    .position(|&c| c == cur)?;
                segments.push(idx.to_string());
                cur = p;
            }
            None => {
                let r = ctx.tree.roots().iter().position(|&x| x == cur)?;
                segments.push(format!("#root{r}"));
                break;
            }
        }
    }
    segments.reverse();
    Some(segments.join("/"))
}

/// Walk the color bind list. For each slot whose underlying source
/// has bumped its version: read the new target, advance
/// `last_version`, and either snap (`tree.set_color`) or start a
/// tween on the slot's `displayed` signal.
fn process_color_binds(
    slots: &mut [Option<ColorBindSlot>],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, opt) in slots.iter_mut().enumerate() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation()) {
            let key = BIND_TWEEN_KEY_COLOR + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
            log::debug!("[binds] color slot {idx}: animated tween started target={target:?}");
        } else {
            tree.set_color(slot.node_id, target);
        }
    }
}

/// Image-handle binds snap (no interpolation): when the source version
/// moves, write the new handle straight into the node. `set_image`
/// flags `BACKDROP` when glass exists, so a backdrop cover swap re-runs
/// the blur just like a colour change.
fn process_image_binds(slots: &mut [Option<ImageBindSlot>], tree: &mut crate::node::NodeTree) {
    for opt in slots.iter_mut() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        tree.set_image(slot.node_id, slot.bind.read());
        log::debug!("[binds] image slot updated -> {:?}", slot.bind.read());
    }
}

/// Text-content binds snap (no interpolation): on a version bump, write
/// the new string. `set_text` relayouts (text width may change), so a
/// reactive label resizes correctly without a scene rebuild.
fn process_text_binds(slots: &mut [Option<TextBindSlot>], tree: &mut crate::node::NodeTree) {
    for opt in slots.iter_mut() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        tree.set_text(slot.node_id, slot.bind.read().to_string());
        log::debug!("[binds] text slot updated -> {:?}", slot.bind.read());
    }
}

/// Percentage-width binds snap: on a version bump, set the node's width
/// to `Len::Pct(value)`. Drives responsive fills (progress bar) without
/// a rebuild.
fn process_width_pct_binds(
    slots: &mut [Option<WidthPctBindSlot>],
    tree: &mut crate::node::NodeTree,
) {
    for opt in slots.iter_mut() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        // `set_layout_width` flags BACKDROP only if the node is a
        // blur_source — a progress fill isn't, so a per-frame width
        // animation stays cheap (no full-window blur re-run).
        tree.set_layout_width(slot.node_id, crate::layout::Len::Pct(slot.bind.read()));
    }
}

fn process_width_px_binds(slots: &mut [Option<WidthPxBindSlot>], tree: &mut crate::node::NodeTree) {
    for opt in slots.iter_mut() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        tree.set_layout_width(slot.node_id, crate::layout::Len::Px(slot.bind.read()));
    }
}

fn process_height_px_binds(
    slots: &mut [Option<HeightPxBindSlot>],
    tree: &mut crate::node::NodeTree,
) {
    for opt in slots.iter_mut() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        tree.set_layout_height(slot.node_id, crate::layout::Len::Px(slot.bind.read()));
    }
}

fn process_opacity_binds(
    slots: &mut [Option<OpacityBindSlot>],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, opt) in slots.iter_mut().enumerate() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation()) {
            // Animated: tween `displayed` toward the new value; the pump
            // pushes it into the node opacity (+ visibility) each frame.
            let key = BIND_TWEEN_KEY_OPACITY + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_opacity(slot.node_id, target);
            // Fully transparent → drop the node (and its subtree) from
            // flatten so an invisible overlay can't eat input. Restored the
            // instant opacity rises above the threshold again.
            tree.set_visible(slot.node_id, target > 0.001);
        }
    }
}

fn process_position_binds(
    slots: &mut [Option<PositionBindSlot>],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, opt) in slots.iter_mut().enumerate() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation()) {
            let key = BIND_TWEEN_KEY_POSITION + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_layout_pos_abs(slot.node_id, target);
        }
    }
}

fn process_size_binds(
    slots: &mut [Option<SizeBindSlot>],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, opt) in slots.iter_mut().enumerate() {
        let Some(slot) = opt.as_mut() else { continue };
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation()) {
            let key = BIND_TWEEN_KEY_SIZE + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_layout_size_px(slot.node_id, target);
        }
    }
}

/// Resolve a winit `KeyEvent`'s `logical_key` + `text` into an
/// [`EditOp`]. Returns `None` for keys the editor doesn't handle (so
/// the caller can fall through to other shell-level handlers).
///
/// `text` is winit's pre-resolved "what character did this key
/// produce" string (handles shift, dead keys, etc. for us). Control
/// chars (\r, \n, escape, etc.) are filtered out — those come through
/// as named keys instead.
///
/// `mods` routes Shift (selection-extend on arrows / Home / End) and the
/// accelerator (Ctrl on Win/Linux, Cmd on macOS): accel+A = SelectAll,
/// +C = Copy, +X = Cut. Paste (+V) is **not** produced here — it needs a
/// clipboard read, so the caller handles it before calling this fn.
fn resolve_edit_op(
    logical: &Key,
    text: Option<&str>,
    mods: winit::keyboard::ModifiersState,
) -> Option<crate::editor::EditOp> {
    use crate::editor::EditOp;
    let shift = mods.shift_key();
    let accel = mods.control_key() || mods.super_key();
    if let Key::Named(named) = logical {
        match named {
            NamedKey::Backspace => return Some(EditOp::DeleteBack),
            NamedKey::Delete => return Some(EditOp::DeleteForward),
            NamedKey::ArrowLeft => {
                return Some(if shift {
                    EditOp::SelectLeft
                } else {
                    EditOp::MoveLeft
                });
            }
            NamedKey::ArrowRight => {
                return Some(if shift {
                    EditOp::SelectRight
                } else {
                    EditOp::MoveRight
                });
            }
            NamedKey::Home => {
                return Some(if shift {
                    EditOp::SelectHome
                } else {
                    EditOp::Home
                });
            }
            NamedKey::End => {
                return Some(if shift {
                    EditOp::SelectEnd
                } else {
                    EditOp::End
                });
            }
            NamedKey::Enter => return Some(EditOp::Submit),
            _ => {}
        }
    }
    // Accelerator combos on character keys (Ctrl/Cmd + A/C/X). Suppress
    // text insertion while accel is held — return None for anything else
    // so non-edit shortcuts (Ctrl+S, etc.) fall through to the shell.
    if accel {
        if let Key::Character(s) = logical {
            if s.as_str().eq_ignore_ascii_case("a") {
                return Some(EditOp::SelectAll);
            }
            if s.as_str().eq_ignore_ascii_case("c") {
                return Some(EditOp::Copy);
            }
            if s.as_str().eq_ignore_ascii_case("x") {
                return Some(EditOp::Cut);
            }
        }
        return None;
    }
    if let Some(s) = text
        && !s.is_empty()
        && !s.chars().any(|c| c.is_control())
    {
        return Some(EditOp::Insert(s.to_string()));
    }
    None
}

/// Translate a [`SubtreeRemoval`] into `Timeline::stop` calls for every
/// bind-slot index that was tombstoned. App-shell wrappers around
/// `SceneCtx::remove_subtree` call this immediately after the removal
/// so any in-flight tween targeting a freed slot drops.
fn stop_tweens_for_removal(timeline: &mut Timeline, r: &crate::scene::SubtreeRemoval) {
    for &idx in &r.dropped_color_slots {
        timeline.stop(BIND_TWEEN_KEY_COLOR + idx);
    }
    for &idx in &r.dropped_position_slots {
        timeline.stop(BIND_TWEEN_KEY_POSITION + idx);
    }
    for &idx in &r.dropped_size_slots {
        timeline.stop(BIND_TWEEN_KEY_SIZE + idx);
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
    pub layer_tree: &'a mut crate::layer::LayerTree,
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
        let vp = [
            self.gpu.surface_config.width as f32,
            self.gpu.surface_config.height as f32,
        ];
        // Mirror `App::flush_tree`: refresh lazy prefixes + initial layout,
        // then materialize the visible window (a scroll can cross a row
        // boundary), re-laying out if rows changed. Without this a headless
        // scroll never re-windows a lazy list (the live app does both).
        ensure_lazy_list_prefixes(self.ctx);
        if mask & (crate::node::dirty::TREE | crate::node::dirty::TRANSFORM) != 0 {
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                vp,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        if materialize_lazy_lists(self.ctx, self.timeline, self.scale_factor) {
            crate::layout::compute_layout(
                &mut self.ctx.tree,
                vp,
                &mut self.ctx.text,
                self.scale_factor,
            );
        }
        let mut spans = Vec::new();
        self.ctx.tree.flatten_into_buffers(
            self.scale_factor,
            self.flat_events,
            self.hits,
            self.scroll_hits,
            self.scroll_bars,
            &mut spans,
        );
        self.instances.clear();
        let mut event_inst = Vec::new();
        *self.glass_count = expand_events_into(
            self.flat_events,
            self.instances,
            self.gpu,
            &mut self.ctx.text,
            self.scale_factor,
            &mut event_inst,
        );
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        self.gpu
            .set_instances(self.instances, *self.glass_count, backdrop_hint);
        // Drive the renderer's layer tree from the promoted spans so
        // headless captures exercise the same raster/composite path as
        // the live app. No promotions → single root layer.
        let promoted = spans_to_instance_ranges(&spans, &event_inst);
        self.layer_tree.rebuild(
            &promoted,
            self.instances.len(),
            [
                self.gpu.surface_config.width,
                self.gpu.surface_config.height,
            ],
            crate::layer::Damage::classify(mask),
        );
        self.layer_tree.take_composite_dirty();
        let draws: Vec<crate::gpu::LayerDraw> =
            self.layer_tree.layers().iter().map(layer_to_draw).collect();
        self.gpu.set_layers(&draws);
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
        for slot in self.ctx.binds.color.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.position.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.size.iter().flatten() {
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
        for slot in self.ctx.binds.color.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.position.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_pos_abs(slot.node_id, disp.get());
            }
        }
        for slot in self.ctx.binds.size.iter().flatten() {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_layout_size_px(slot.node_id, disp.get());
            }
        }
    }
}

impl ApplicationHandler for App {
    /// Empty user-event handler. The only sender is
    /// [`WakeHandle::wake`]; the payload is unused, the side effect
    /// (winit returning from `Wait` into the next `about_to_wait`)
    /// is what we want. The flag itself is consumed in
    /// `about_to_wait`.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _: ()) {}

    /// Last call before the loop tears down. Forward to the consumer's
    /// `on_exit` hook (registered via [`App::on_exit`]) so save-on-quit
    /// work runs deterministically — covers the close-button path, an
    /// `event_loop.exit()` triggered by a callback, and any clean signal.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(hook) = self.on_exit.take() {
            hook();
        }
        // Wait for any in-flight PNG encode to finish writing — the
        // autocapture + exit path otherwise truncates the file when
        // the process tears down mid-write.
        self.flush_pending_screenshot();
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Wait);

        let mut attrs = Window::default_attributes()
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
        // Configured window icon (taskbar / alt-tab). Fail-soft: a
        // mismatched buffer just leaves winit on the exe-icon fallback.
        if let Some((w, h, rgba)) = self.config.window_icon.clone() {
            match winit::window::Icon::from_rgba(rgba, w, h) {
                Ok(icon) => attrs = attrs.with_window_icon(Some(icon)),
                Err(e) => log::warn!("window icon ignored: {e}"),
            }
        }
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
        // Apply the configured window-corner radius — suppressed while
        // maximised so the radius doesn't clip against the work-area
        // edge. Toggle is re-evaluated on every resize below.
        let initial_r = if window_arc.is_maximized() {
            0.0
        } else {
            self.config.window_corner_radius
        };
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_window_corner_radius(initial_r, self.scale_factor);
        }
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
                    StagedImage::Rgba { w, h, bytes } => gpu
                        .image_atlas
                        .upload_rgba(&gpu.queue, w, h, &bytes)
                        .is_some(),
                };
                if !ok {
                    log::warn!("staged image #{idx}: decode/upload failed");
                }
            }
        }

        self.log_memory_report();

        // First flush + render so the visible window already shows
        // the scene by the time the user sees it.
        self.flush_tree();
        self.render_once();
        if let Some(w) = &self.window {
            w.set_visible(true);
            w.request_redraw();
        }

        // Deferred autocapture: arm a deadline + let the loop run
        // normally. `about_to_wait` picks the deadline up + drives the
        // capture once the wall-clock elapses, so worker fetches,
        // image uploads, and animations all settle before the snap.
        if self.capture_frames.is_some()
            && let Some(d) = self.capture_delay
        {
            self.capture_deadline = Some(Instant::now() + d);
            return;
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
                layer_tree: &mut self.layer_tree,
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(_gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(size.width, size.height);
                }
                // Re-evaluate the window corner radius — Resized fires
                // on maximize/restore transitions on Windows + Linux,
                // so this is the canonical hook to switch between
                // "rounded floating window" and "square maximised".
                if self.config.window_corner_radius > 0.0
                    && let (Some(w), Some(g)) = (self.window.as_ref(), self.gpu.as_mut())
                {
                    let r = if w.is_maximized() {
                        0.0
                    } else {
                        self.config.window_corner_radius
                    };
                    g.set_window_corner_radius(r, self.scale_factor);
                }
                let in_dpi_window = self
                    .last_dpi_change
                    .map(|t| t.elapsed() < Duration::from_millis(500))
                    .unwrap_or(false);
                let want_w = (self.logical_size[0] as f32 * self.scale_factor).round() as u32;
                let want_h = (self.logical_size[1] as f32 * self.scale_factor).round() as u32;
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
                        let _ = w.request_inner_size(winit::dpi::PhysicalSize::new(want_w, want_h));
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
                    log::debug!("scale factor: {} → {}", self.scale_factor, new_scale);
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
                    // Re-shape + re-rasterize everything at the new physical
                    // size *now* so text stays sharp. We just emptied the
                    // glyph atlas + shape cache; without forcing a re-flatten
                    // here, sharpness would depend on a `Resized` happening to
                    // follow (not guaranteed for a pure scale change), and any
                    // redraw in between would sample a stale/empty atlas.
                    // Mirrors the `Resized` tail.
                    self.ctx.tree.mark_all_dirty();
                    self.flush_tree();
                    self.request_redraw();
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
                // A press captured on a text field turns cursor moves into a
                // selection drag (handled in `about_to_wait`, coalesced).
                let editor_captured = self
                    .input
                    .captured
                    .and_then(|c| self.ctx.tree.get(c))
                    .map(|n| n.editor.is_some())
                    .unwrap_or(false);
                if self.bar_drag.is_some() || self.drag_origin.is_some() || editor_captured {
                    // Coalesce: don't apply per OS event. Stash the
                    // latest cursor and let `about_to_wait` push it
                    // through at frame rate. OS cursor fires at 500+ Hz
                    // and a generic on_drag handler (e.g. a splitter
                    // mutating a width signal) would otherwise relayout
                    // 8× per displayed frame for no visual gain. Skips
                    // hover refresh + the hit-test path during drag
                    // (cursor is captured anyway). Ensure the loop wakes
                    // up so the pending cursor gets applied promptly.
                    self.pending_drag_cursor = Some([x, y]);
                    self.request_redraw();
                    return;
                }
                let bar_changed = crate::input::update_scrollbar_hover(
                    Some([x, y]),
                    &self.scroll_bars,
                    &mut self.ctx.tree,
                );
                let change = self.input.on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
                self.refresh_cursor(x, y);
                // Generic on_drag: fire while a press is captured on a
                // draggable node (slider/scrubber). Independent of the
                // scrollbar-thumb drag handled above.
                let dragged = self.fire_drag(x, y);
                // Hover-move: coalesce to frame rate. Firing per OS
                // CursorMoved (500+ Hz) would re-flatten the whole tree each
                // event, since the handler sets a Signal (seek-bar preview)
                // that dirties layout. Stash the cursor + wake; `about_to_wait`
                // fires it once per frame. Only arm when the hovered node has
                // a hover-move handler, so a plain hover stays at 0% CPU.
                let wants_hover_move = self
                    .input
                    .hovered
                    .and_then(|id| self.ctx.tree.get(id))
                    .map(|n| n.on_hover_move.is_some())
                    .unwrap_or(false);
                if wants_hover_move {
                    self.pending_hover_cursor = Some([x, y]);
                    self.request_redraw();
                }
                // Drag-follow: track the cursor for a lifted node.
                let following = self.update_drag_follow(x, y);
                if change.hovered_changed {
                    self.refresh_dwell(Instant::now());
                }
                // A shown hint follows the pointer — coalesce the move and
                // repaint it once this frame (`refresh_dwell` above already
                // cleared it if hover left its node).
                if self.active_hint.is_some() {
                    self.pending_hint_repaint = true;
                    self.request_redraw();
                }
                if change.any() || bar_changed || dragged || following {
                    self.react();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                // The cursor left the client area. Receiving this mid-drag
                // means OS capture wasn't held (or was lost) — the button's
                // release will never reach us, so fully reset pointer
                // interaction (capture, drag, pressed + hover signals)
                // rather than leave a node stuck "down"/"hovered" or a
                // seek-bar stuck "seeking".
                if self.cancel_pointer_interaction() {
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
                // Same for in-flight pointer interaction: a button held at
                // focus-loss (alt-tab, click-through to another window) will
                // release where we can't see it, so fully reset capture +
                // drag + pressed/hover signals (fires on_drag_end so a held
                // scrub commits/resets rather than sticking).
                if self.cancel_pointer_interaction() {
                    self.react();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let Some([cx, cy]) = self.input.cursor else {
                    return;
                };
                // A node-level wheel handler under the cursor (slider)
                // consumes the tick before scroll-container routing.
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(lx, ly) => [lx, ly],
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        let line = 50.0 * self.scale_factor;
                        [p.x as f32 / line, p.y as f32 / line]
                    }
                };
                if self.fire_wheel(cx, cy, lines) {
                    self.react();
                    return;
                }
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
                    winit::event::MouseScrollDelta::PixelDelta(p) => [-p.x as f32, -p.y as f32],
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
                        && let Some(dir) = self.edge_at(cx, cy)
                    {
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
                        && self.dispatch_window_action(action, event_loop)
                    {
                        return;
                    }
                }
                let change = match state {
                    ElementState::Pressed => self.input.on_left_pressed(&self.hits, &self.ctx.tree),
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
                // Outside-press dismissal. On a *press* that hit no
                // interactive node — or hit a dismiss-transparent scrim —
                // fire the hook so floating layers (modals, context menus)
                // can close. `input.captured` was just set to the press
                // target by `on_left_pressed`.
                if state == ElementState::Pressed {
                    self.begin_drag_if_draggable();
                    self.maybe_fire_unhandled_press();
                    // Click-to-position-caret: a press inside a text field
                    // moves the caret to the nearest char boundary.
                    if let Some(cap) = self.input.captured
                        && let Some([cx, _]) = self.input.cursor
                        && self
                            .ctx
                            .tree
                            .get(cap)
                            .map(|n| n.editor.is_some())
                            .unwrap_or(false)
                    {
                        self.place_caret_from_click(cap, cx);
                    }
                } else {
                    // Release: deliver any in-flight drag payload to a
                    // drop target under the cursor, then clear drag state.
                    self.finish_drag_on_release();
                }
                if change.hovered_changed {
                    self.refresh_dwell(Instant::now());
                }
                // Fire the click handler (if any) for the released node.
                // Clone the `Rc<dyn Fn>` out first so the immutable
                // borrow of the node drops before we re-borrow `&mut
                // tree` for the EventCtx.
                if let Some(target) = change.click_target {
                    let handler = self.ctx.tree.get(target).and_then(|n| n.on_click.clone());
                    if let Some(h) = handler {
                        let mut ectx = crate::event::EventCtx {
                            tree: &mut self.ctx.tree,
                            timeline: &mut self.timeline,
                            node: target,
                            now: Instant::now(),
                            cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
                        };
                        h(&mut ectx);
                    }
                }
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                let change = match state {
                    ElementState::Pressed => {
                        self.input.on_right_pressed(&self.hits, &self.ctx.tree)
                    }
                    ElementState::Released => {
                        self.input.on_right_released(&self.hits, &self.ctx.tree)
                    }
                };
                if let Some(target) = change.right_click_target {
                    let handler = self
                        .ctx
                        .tree
                        .get(target)
                        .and_then(|n| n.on_right_click.clone());
                    if let Some(h) = handler {
                        let mut ectx = crate::event::EventCtx {
                            tree: &mut self.ctx.tree,
                            timeline: &mut self.timeline,
                            node: target,
                            now: Instant::now(),
                            cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
                        };
                        h(&mut ectx);
                    }
                }
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        logical_key,
                        text,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                if state == ElementState::Pressed {
                    // Text-field routing fires *before* the hotkey path.
                    // While a TextField is focused, typeable keys + nav
                    // keys (Backspace, Arrow*, Home, End, Enter) feed the
                    // editor. Escape blurs. F-keys + other unmapped keys
                    // fall through to the hotkey + scroll handlers so
                    // F2 screenshot etc. still work mid-edit.
                    if let Some(focused) = self.input.focused {
                        let has_editor = self
                            .ctx
                            .tree
                            .get(focused)
                            .map(|n| n.editor.is_some())
                            .unwrap_or(false);
                        if has_editor {
                            let mods = self.modifiers.state();
                            if matches!(logical_key, Key::Named(NamedKey::Escape)) {
                                self.blur_text_field(focused);
                                return;
                            }
                            // Paste needs a clipboard read first, so it's
                            // handled here rather than in resolve_edit_op.
                            let accel = mods.control_key() || mods.super_key();
                            let is_v = matches!(
                                &logical_key,
                                Key::Character(s) if s.as_str().eq_ignore_ascii_case("v")
                            );
                            if accel && is_v {
                                let pasted = self.read_clipboard();
                                if !pasted.is_empty() {
                                    self.apply_edit(focused, crate::editor::EditOp::Paste(pasted));
                                }
                                return;
                            }
                            if let Some(op) = resolve_edit_op(&logical_key, text.as_deref(), mods) {
                                self.apply_edit(focused, op);
                                return;
                            }
                        }
                    }
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
                                // Dump the live memory split (VRAM vs system
                                // RAM) for the current working set.
                                self.log_memory_report();
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
                            KeyCode::Tab => {
                                // Tab / Shift+Tab cycle keyboard focus.
                                // Fires even with a text field focused —
                                // standard "Tab leaves the field" UX. The
                                // editor never claims Tab (resolve_edit_op
                                // has no mapping for it), so it reaches
                                // here untouched.
                                let forward = !self.modifiers.state().shift_key();
                                self.focus_next(forward);
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
                        KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End => {
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
        // Keep the scene ctx's display scale in sync so per-frame hooks
        // (`on_frame`) can convert physical reads (scroll offsets) to
        // logical layout units. Cheap; the value rarely changes.
        self.ctx.scale = self.scale_factor;
        // Flush any clipboard write an event handler requested this frame
        // (e.g. a "click to copy" button) using the shell's persistent
        // clipboard handle.
        if let Some(text) = self.ctx.tree.take_clipboard() {
            self.write_clipboard(text);
        }
        // Programmatic focus request (e.g. re-focus a field after a failed
        // submit) — route through the same path as a click.
        if let Some(id) = self.ctx.tree.take_focus_request() {
            self.set_focus(Some(id));
        }
        // Deferred autocapture: once the deadline elapses, render a
        // settled frame, snap, and exit. Loop has been running normally
        // up to this point so worker state, image uploads, and any
        // animations have all had a chance to land.
        if let Some(deadline) = self.capture_deadline
            && Instant::now() >= deadline
        {
            self.capture_deadline = None;
            self.render_once();
            self.save_screenshot();
            event_loop.exit();
            return;
        }
        // Consume any rebuild request first — if a click handler
        // flipped the token mid-event, the old tree is stale and
        // every subsequent tick (animation, scroll, etc.) would
        // operate on stale node ids. Rebuild before anything else.
        if self.rebuild_request.replace(false) {
            self.rebuild_scene();
        }
        // Drain queued cross-thread image uploads. Done before the
        // on-frame wake check so a worker doing `upload_rgba(...)` +
        // `wake()` in the same instant resolves the handle before the
        // on-frame hook fires and reads it.
        self.drain_image_uploads();
        // Drain queued external (video) frames + render them immediately.
        // Rendering directly (not via request_redraw) is required: the
        // decode thread's wake lands when the loop is otherwise idle and
        // about to park on `Wait`, where a requested redraw isn't honoured.
        if self.drain_external_frames() {
            self.render_once();
        }
        let now = Instant::now();
        // External wake (worker delivered a response on a channel the
        // `on_frame` hook drains, etc.). Fire `on_frame` even when no
        // timeline / scroll work is pending so the channel gets drained
        // — without this the worker's wake would land but the loop
        // would still park since `timeline_active == false`.
        if self.wake.take()
            && let Some(mut hook) = self.on_frame.take()
        {
            log::debug!(
                "[loop] wake-driven on_frame fired (timeline_tweens={})",
                self.timeline.len()
            );
            hook(&mut self.ctx, &mut self.timeline, now);
            self.on_frame = Some(hook);
            log::debug!(
                "[loop] after on_frame: timeline_tweens={} rebuild_requested={}",
                self.timeline.len(),
                self.rebuild_request.get()
            );
            // Drain any rebuild flag the hook just set so the rebuild
            // rides this tick rather than waiting for the next wake.
            if self.rebuild_request.replace(false) {
                self.rebuild_scene();
                log::debug!(
                    "[loop] rebuilt scene; timeline_tweens={}",
                    self.timeline.len()
                );
            } else {
                // No rebuild — but the hook may have set source signals
                // (worker-delivered state, etc.). Walk the bind registry
                // so those changes propagate to the tree (snap for
                // non-animated reactive binds; start a tween for
                // animated ones). Without this, `Signal::set` from any
                // non-input path would be lost between rebuilds.
                self.process_binds(now);
            }
            if self.flush_tree() {
                self.request_redraw();
                // Worker-delivered content (a list filled in, a row changed)
                // may have shifted what's under a still cursor.
                self.arm_hover_recheck();
            }
        }
        // Debug-only scripted-input driver (REMOVABLE — `automation`
        // feature). Executes any due step (synthetic move/click/scroll +
        // screenshots) and returns its next scheduled wake so the loop
        // stays alive + doesn't park past it.
        #[cfg(feature = "automation")]
        let auto_deadline = self.automation_tick(event_loop, now);
        #[cfg(not(feature = "automation"))]
        let auto_deadline: Option<Instant> = None;
        // Animation + scroll pump. If both are idle, park on `Wait` so
        // the loop is 0% CPU. Otherwise: advance both, push interpolated
        // values through the bind registry, flush, redraw, and schedule
        // the next deadline.
        // Drain any coalesced thumb-drag cursor: a single
        // set_scroll_immediate per frame, regardless of how many
        // CursorMoved events fired since the last tick.
        let drag_moved = if let Some(c) = self.pending_drag_cursor.take() {
            // Two drag mechanisms, both fed from the same coalesced
            // cursor: scrollbar thumb (bar_drag) and generic on_drag
            // (e.g. splitter). Either can be active independently.
            let bar = if let Some(d) = self.bar_drag {
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
            let generic = self.fire_drag(c[0], c[1]);
            let following = self.update_drag_follow(c[0], c[1]);
            // Text-field selection drag: a press captured on an editor
            // extends its selection toward the cursor x.
            let editor_sel = match self.input.captured {
                Some(cap)
                    if self
                        .ctx
                        .tree
                        .get(cap)
                        .map(|n| n.editor.is_some())
                        .unwrap_or(false) =>
                {
                    self.drag_select_to(cap, c[0])
                }
                _ => false,
            };
            bar || generic || following || editor_sel
        } else {
            false
        };
        // Drain the coalesced hover-move cursor: one `on_hover_move` dispatch
        // per frame regardless of how many CursorMoved events fired.
        let hover_moved = if let Some(c) = self.pending_hover_cursor.take() {
            self.fire_hover_move(c[0], c[1])
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
        // Dwell deadline is the third source of "stay awake" alongside
        // timeline + scroll. Fire here if elapsed; either way, propagate
        // the deadline below so the loop schedules a wake.
        let dwell_fired = self.tick_dwell(now);
        let dwell_pending = self.dwell.as_ref().map(|t| !t.fired).unwrap_or(false);
        // Re-derive hover if a debounced re-check is due (content changed
        // under a stationary cursor). Returns true if hover flipped.
        let hover_rechecked = self.tick_hover_recheck(now);
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
        // Caret blink for the focused text field. Applies the caret opacity
        // toggle (if due) and yields the next toggle deadline so the loop
        // wakes for it; a focused field therefore keeps the loop alive.
        let (caret_blinked, caret_deadline) = self.tick_caret_blink(now);
        if !timeline_active
            && !scroll_active
            && !drag_moved
            && !hover_moved
            && !dwell_pending
            && !dwell_fired
            && !hover_rechecked
            && self.next_hover_recheck.is_none()
            && !self.pending_hint_repaint
            && auto_deadline.is_none()
            && caret_deadline.is_none()
        {
            self.last_scroll_tick = None;
            event_loop.set_control_flow(ControlFlow::Wait);
            log::debug!("[loop] parking on Wait (idle)");
            return;
        }
        log::debug!(
            "[loop] active tick: timeline_tweens={} scroll={} drag={} dwell={}/{}",
            self.timeline.len(),
            scroll_active,
            drag_moved,
            dwell_pending,
            dwell_fired,
        );
        let dt = match self.last_scroll_tick {
            Some(prev) => (now - prev).as_secs_f32().min(0.05),
            None => 0.0,
        };
        self.last_scroll_tick = Some(now);
        let scroll_moved = self.ctx.tree.tick_scrolls(dt);
        let res = self.timeline.tick(now);
        // Hover-hint: repaint while the fade is animating or the cursor moved
        // (now that the fade value is current), and finalise once a fade-out
        // reaches zero.
        if self.active_hint.is_some() {
            let fade = self.hint_fade.get();
            let animating = if self.hint_visible {
                fade < 0.999
            } else {
                fade > 0.001
            };
            if self.pending_hint_repaint || animating {
                self.pending_hint_repaint = false;
                self.paint_hint();
            }
            if !self.hint_visible && fade <= 0.01 {
                self.clear_hint();
            }
        }
        // Mirror the snap-back tween into the tree's drag-follow offset
        // (and clear the lift when it lands). The tween keeps the loop
        // awake via `timeline_active`, so this runs every frame of the
        // return until it completes.
        self.tick_drag_return();
        // Per-frame hook fires after the timeline tick so it observes
        // the just-interpolated signal values — pushing those into
        // non-bind targets (e.g. lazy-list row heights) lands with
        // zero lag.
        let hook_active = self.on_frame.is_some();
        if let Some(mut hook) = self.on_frame.take() {
            hook(&mut self.ctx, &mut self.timeline, now);
            self.on_frame = Some(hook);
        }
        // Walk the bind registry every tick the loop is awake. This is
        // the bridge from "Signal was set somewhere" to "tree picks up
        // the new value":
        //  - Timeline tweens that target user-owned source Signals
        //    (e.g. a non-animated reactive bind whose Signal the caller
        //    is tweening directly) bump the source version every tick
        //    — process_binds snaps the new value into the tree.
        //  - Computed binds whose deps just bumped (because a tween
        //    above advanced one of their source signals) re-read here.
        //  - on_frame Signal::set calls land before flush via the same
        //    path.
        // Animated binds whose source was set go through the timeline-
        // start path inside process_binds; their displayed signals are
        // then pumped below.
        if res.updated || hook_active {
            self.process_binds(now);
        }
        // Push layer-opacity bindings (e.g. a crossfade tween driving a
        // promoted layer's composite opacity) every frame the timeline
        // advanced. Composite-only — recomposites without re-rastering the
        // layer; `set_layer_opacity` requests its own redraw on change.
        if res.updated {
            self.pump_layer_opacity_binds();
        }
        // Push coalesced hover-move / drag offsets into cursor-following
        // overlay layers (e.g. the seek tooltip) — composite-only, so the
        // overlay tracks the cursor with no re-flatten of the scene.
        if hover_moved || drag_moved {
            self.pump_layer_offset_binds();
        }
        if res.updated
            || scroll_moved
            || drag_moved
            || hover_moved
            || hook_active
            || dwell_fired
            || caret_blinked
        {
            if res.updated {
                self.pump_animated_displays();
            }
            if self.flush_tree() {
                self.request_redraw();
                // Content moved/morphed (scroll, animation, layout) — hover
                // may now point at the wrong node under a still cursor.
                self.arm_hover_recheck();
            }
        }
        let next_scroll_deadline = if self.ctx.tree.has_active_scrolls() || self.bar_drag.is_some()
        {
            Some(now + std::time::Duration::from_millis(16))
        } else {
            None
        };
        // Wake up at the dwell deadline so the tooltip fires on time
        // even when nothing else is animating.
        let dwell_deadline = self.dwell.as_ref().filter(|t| !t.fired).map(|t| t.deadline);
        // process_binds above may have *started* new tweens (when an
        // animated bind's source was just set). Those tweens didn't
        // exist when timeline.tick computed `res.next_deadline`, so
        // honour `timeline.active()` as the authoritative "still has
        // work" check.
        let timeline_deadline = if self.timeline.active() {
            Some(now + self.timeline.tick_interval())
        } else {
            res.next_deadline
        };
        let combined = match (timeline_deadline, next_scroll_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let combined = match (combined, dwell_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        // Wake by the hover-recheck deadline (content changed under a still
        // cursor) so the debounced re-derive lands even when otherwise idle.
        let combined = match (combined, self.next_hover_recheck) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        // Make sure the loop wakes by the deferred-capture deadline so
        // we don't park indefinitely while waiting to snap.
        let combined = match (combined, self.capture_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        // Merge the scripted-input driver's next wake (REMOVABLE).
        let combined = match (combined, auto_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        // Wake for the next caret-blink toggle while a field is focused.
        let combined = match (combined, caret_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        match combined {
            Some(deadline) => {
                let dt = deadline.saturating_duration_since(Instant::now());
                log::debug!("[loop] WaitUntil({:?} from now)", dt);
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline))
            }
            None => {
                log::debug!("[loop] no deadline; parking on Wait");
                event_loop.set_control_flow(ControlFlow::Wait)
            }
        }
    }
}

// ============================================================================
// Debug-only scripted-input + screenshot harness (REMOVABLE).
// Everything below is gated behind the `automation` feature. Deleting the
// feature + this block + `src/automation.rs` fully removes it. Synthetic
// input is routed through the same `input::*` handlers + `on_click` firing
// as real winit events, so the scripted path is the real path.
// ============================================================================
#[cfg(feature = "automation")]
impl App {
    /// Attach a scripted-input run. The driver ticks from `about_to_wait`,
    /// injecting synthetic input + capturing screenshots, then exits the
    /// app when the script ends.
    pub fn automation(mut self, script: crate::automation::Script) -> Self {
        self.automation = Some(crate::automation::AutomationState::new(script));
        self
    }

    /// Run the scripted-input driver: execute any due step, return the
    /// next scheduled wake (so the loop doesn't park past it). Exits the
    /// app once the script is exhausted.
    fn automation_tick(&mut self, event_loop: &ActiveEventLoop, now: Instant) -> Option<Instant> {
        let Some(mut state) = self.automation.take() else {
            return None;
        };
        if state.finished() {
            self.flush_pending_screenshot();
            log::info!("[automation] script finished — exiting");
            event_loop.exit();
            return None;
        }
        if state.due(now)
            && let Some(step) = state.current()
        {
            use crate::automation::Step;
            log::info!("[automation] step {:?}", step);
            match step {
                Step::Wait(d) => state.advance_after(now, d),
                Step::Hover(p, d) => {
                    self.inject_cursor_moved(p[0], p[1]);
                    state.advance_after(now, d);
                }
                Step::MoveMouse(p) => {
                    self.inject_cursor_moved(p[0], p[1]);
                    state.advance_now(now);
                }
                Step::Click(p) => {
                    self.inject_cursor_moved(p[0], p[1]);
                    self.inject_left(true);
                    self.inject_left(false);
                    state.advance_now(now);
                }
                Step::RightClick(p) => {
                    self.inject_cursor_moved(p[0], p[1]);
                    self.inject_right(true);
                    self.inject_right(false);
                    state.advance_now(now);
                }
                Step::Scroll(p, d) => {
                    self.inject_cursor_moved(p[0], p[1]);
                    self.inject_wheel(d[0], d[1]);
                    state.advance_now(now);
                }
                Step::Drag(a, b) => {
                    self.inject_cursor_moved(a[0], a[1]);
                    self.inject_left(true);
                    self.inject_cursor_moved(b[0], b[1]);
                    self.inject_left(false);
                    state.advance_now(now);
                }
                Step::Screenshot(path) => {
                    self.render_once();
                    self.capture_to(path);
                    state.advance_now(now);
                }
            }
        }
        let next = state.next_at();
        self.automation = Some(state);
        next
    }

    /// Mirror of the `WindowEvent::CursorMoved` non-drag path.
    fn inject_cursor_moved(&mut self, x: f32, y: f32) {
        self.input.cursor = Some([x, y]);
        let bar_changed = crate::input::update_scrollbar_hover(
            Some([x, y]),
            &self.scroll_bars,
            &mut self.ctx.tree,
        );
        let change = self.input.on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
        self.refresh_cursor(x, y);
        let hovered_moved = self.fire_hover_move(x, y);
        if change.hovered_changed {
            self.refresh_dwell(Instant::now());
        }
        if change.any() || bar_changed || hovered_moved {
            self.react();
        }
    }

    /// Mirror of the left `MouseInput` arm's core: press/release through
    /// `input`, fire `on_click` on the released target, drain any rebuild
    /// the handler requested. Skips edge-resize + window-action paths
    /// (irrelevant to scripted button targets).
    fn inject_left(&mut self, pressed: bool) {
        let change = if pressed {
            let c = self.input.on_left_pressed(&self.hits, &self.ctx.tree);
            self.begin_drag_if_draggable();
            self.maybe_fire_unhandled_press();
            c
        } else {
            let c = self.input.on_left_released(&self.hits, &self.ctx.tree);
            self.finish_drag_on_release();
            c
        };
        if change.hovered_changed {
            self.refresh_dwell(Instant::now());
        }
        if let Some(target) = change.click_target {
            let handler = self.ctx.tree.get(target).and_then(|n| n.on_click.clone());
            if let Some(h) = handler {
                let mut ectx = crate::event::EventCtx {
                    tree: &mut self.ctx.tree,
                    timeline: &mut self.timeline,
                    node: target,
                    now: Instant::now(),
                    cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
                };
                h(&mut ectx);
            }
        }
        // A nav/click handler may have flipped the rebuild token — apply
        // it now so the new view is live before the next step/screenshot.
        if self.rebuild_request.replace(false) {
            self.rebuild_scene();
        }
        if change.any() {
            self.react();
        }
    }

    /// Scripted right-click — mirror of the `MouseButton::Right`
    /// WindowEvent path: fire `on_right_click` on the target + apply any
    /// rebuild it requested (so a context menu is live for the next step).
    fn inject_right(&mut self, pressed: bool) {
        let change = if pressed {
            self.input.on_right_pressed(&self.hits, &self.ctx.tree)
        } else {
            self.input.on_right_released(&self.hits, &self.ctx.tree)
        };
        if let Some(target) = change.right_click_target {
            let handler = self.ctx.tree.get(target).and_then(|n| n.on_right_click.clone());
            if let Some(h) = handler {
                let mut ectx = crate::event::EventCtx {
                    tree: &mut self.ctx.tree,
                    timeline: &mut self.timeline,
                    node: target,
                    now: Instant::now(),
                    cursor: self.input.cursor.unwrap_or([0.0, 0.0]),
                };
                h(&mut ectx);
            }
        }
        if self.rebuild_request.replace(false) {
            self.rebuild_scene();
        }
        if change.any() {
            self.react();
        }
    }

    /// Mirror of the `WindowEvent::MouseWheel` arm (line deltas).
    fn inject_wheel(&mut self, dx: f32, dy: f32) {
        let Some([cx, cy]) = self.input.cursor else {
            return;
        };
        let line = 50.0 * self.scale_factor;
        let px = [-dx * line, -dy * line];
        let shift = self.modifiers.state().shift_key();
        if crate::input::on_wheel([cx, cy], px, &self.scroll_hits, &mut self.ctx.tree, shift) {
            self.react();
        }
    }

    /// Render + write a PNG to an explicit path (the scripted analog of
    /// `save_screenshot`, which auto-names into the capture dir).
    fn capture_to(&mut self, path: std::path::PathBuf) {
        self.flush_pending_screenshot();
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let (rgba, w, h) = gpu.capture_rgba();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        log::info!("[automation] screenshot → {}", path.display());
        self.pending_screenshot = Some(debug::save_png_async(path, rgba, w, h));
    }
}

/// Reconcile every lazy-list's materialized children with its current
/// visible window. Free fn so both `App::flush_tree` and
/// `HeadlessHelper::flush` drive the same path. Returns true if any
/// list mutated its children (caller may need a second layout pass).
fn materialize_lazy_lists(ctx: &mut SceneCtx, timeline: &mut Timeline, scale: f32) -> bool {
    let lazy_ids: Vec<crate::node::NodeId> = ctx
        .tree
        .iter_ids()
        .filter(|id| {
            ctx.tree
                .get(*id)
                .map(|n| n.lazy_list.is_some())
                .unwrap_or(false)
        })
        .collect();
    if lazy_ids.is_empty() {
        return false;
    }
    let mut any_changed = false;
    for list_id in lazy_ids {
        // Snapshot per-list state. `render` is an `Rc<dyn Fn>`;
        // the clone is cheap. Drop the &mut tree borrow before
        // invoking it (the closure spawns children, re-borrowing
        // the tree).
        let (
            render,
            prev_range,
            prev_materialized,
            current_version,
            last_seen_version,
            current_heights_version,
            last_seen_heights_version,
        ) = {
            let Some(n) = ctx.tree.get(list_id) else {
                continue;
            };
            let Some(ll) = n.lazy_list.as_ref() else {
                continue;
            };
            (
                ll.render.clone(),
                ll.range,
                ll.materialized.clone(),
                ll.version,
                ll.last_seen_version,
                ll.heights_version,
                ll.last_heights_version,
            )
        };

        // The list is its own scroll container; read viewport
        // size + scroll offset from its rect + ScrollState.
        let (scroll_top, viewport_h) = {
            let Some(n) = ctx.tree.get(list_id) else {
                continue;
            };
            let scroll_top = n.scroll.as_ref().map(|s| s.current[1]).unwrap_or(0.0);
            let viewport_h = n.rect[3];
            (scroll_top, viewport_h)
        };

        let new_range = {
            let Some(n) = ctx.tree.get(list_id) else {
                continue;
            };
            let ll = n.lazy_list.as_ref().unwrap();
            ll.visible_window(scroll_top, viewport_h, scale)
        };

        // Detect "needs re-position only" (heights changed but
        // window still covers the same rows). When this is the
        // case, skip the remove+render cycle and just rewrite
        // the abs offsets of the existing materialized children.
        let window_unchanged = new_range == prev_range;
        let version_changed = current_version != last_seen_version;
        let heights_changed = current_heights_version != last_seen_heights_version;
        let _ = last_seen_heights_version;
        if window_unchanged && !version_changed && !heights_changed {
            continue;
        }
        if window_unchanged && !version_changed && heights_changed {
            // Reposition existing rows in-place from the fresh
            // prefix table — much cheaper than full re-render.
            let mut any_moved = false;
            for (k, child_id) in prev_materialized.iter().enumerate() {
                let i = new_range[0] + k as u32;
                let top = ctx
                    .tree
                    .get(list_id)
                    .and_then(|n| n.lazy_list.as_ref())
                    .map(|ll| ll.row_top_logical(i))
                    .unwrap_or(0.0);
                if let Some(c) = ctx.tree.get_mut_raw(*child_id)
                    && c.layout.abs != Some([0.0, top])
                {
                    c.layout.abs = Some([0.0, top]);
                    any_moved = true;
                }
            }
            if let Some(n) = ctx.tree.get_mut_raw(list_id)
                && let Some(ll) = n.lazy_list.as_mut()
            {
                ll.last_heights_version = current_heights_version;
            }
            if any_moved {
                any_changed = true;
            }
            continue;
        }

        // Remove the previous materialized set. Each removal
        // walks descendant children, tombstones binds, and stops
        // matching tweens.
        for child_id in &prev_materialized {
            let removal = ctx.remove_subtree(*child_id);
            stop_tweens_for_removal(timeline, &removal);
        }

        // Materialize the new window. The render closure runs
        // under a Scene scope rooted at the list node. Snapshot
        // children-count before + after each call to identify the
        // single new child emitted for row `i`; write its
        // `layout.abs` so layout positions it at the right offset.
        let mut new_materialized = Vec::with_capacity((new_range[1] - new_range[0]) as usize);
        for i in new_range[0]..new_range[1] {
            let prev_count = ctx.tree.get(list_id).map(|n| n.children.len()).unwrap_or(0);
            {
                let mut scene = crate::scene::Scene::with_parent(ctx, list_id);
                render(&mut scene, i);
            }
            let next_count = ctx.tree.get(list_id).map(|n| n.children.len()).unwrap_or(0);
            if next_count > prev_count {
                let new_child = ctx.tree.get(list_id).unwrap().children[prev_count];
                let top = ctx
                    .tree
                    .get(list_id)
                    .and_then(|n| n.lazy_list.as_ref())
                    .map(|ll| ll.row_top_logical(i))
                    .unwrap_or(0.0);
                if let Some(c) = ctx.tree.get_mut_raw(new_child) {
                    c.layout.abs = Some([0.0, top]);
                }
                new_materialized.push(new_child);
            }
        }

        // Commit the new state.
        if let Some(n) = ctx.tree.get_mut_raw(list_id)
            && let Some(ll) = n.lazy_list.as_mut()
        {
            ll.materialized = new_materialized;
            ll.range = new_range;
            ll.last_seen_version = current_version;
            ll.last_heights_version = current_heights_version;
        }

        any_changed = true;
    }

    any_changed
}

/// Refresh every lazy-list prefix-sum table (variable-height mode).
/// Free fn shared by `App` + `HeadlessHelper`.
fn ensure_lazy_list_prefixes(ctx: &mut SceneCtx) {
    let lazy_ids: Vec<crate::node::NodeId> = ctx
        .tree
        .iter_ids()
        .filter(|id| {
            ctx.tree
                .get(*id)
                .map(|n| n.lazy_list.is_some())
                .unwrap_or(false)
        })
        .collect();
    for id in lazy_ids {
        if let Some(n) = ctx.tree.get_mut_raw(id)
            && let Some(ll) = n.lazy_list.as_mut()
        {
            ll.ensure_prefix_fresh();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Len;
    use crate::signal::Signal;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Named scrollers are identified by name; anonymous ones by their
    /// nearest named ancestor + child-index path. Both must be stable so
    /// `rebuild_scene` can hand offsets to the rebuilt tree.
    #[test]
    fn scroll_identity_names_and_structural_paths() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.col("root").child(|p| {
                p.col("named_scroll").h(Len::Px(50.0)).scroll_y();
                p.col(()).child(|q| {
                    q.rect(());
                    q.col(()).h(Len::Px(50.0)).scroll_y();
                });
            });
        }
        let named_id = ctx.node("named_scroll").unwrap();
        assert_eq!(scroll_identity(&ctx, named_id).as_deref(), Some("named_scroll"));
        let anon = ctx
            .tree
            .scrollables()
            .iter()
            .copied()
            .find(|&i| i != named_id)
            .expect("anonymous scroller registered");
        // Anchored at the named root: child 1 of root, child 1 within.
        assert_eq!(scroll_identity(&ctx, anon).as_deref(), Some("root/1/1"));
    }

    #[test]
    fn collect_live_image_handles_finds_every_image_node() {
        use crate::gpu::ImageHandle;
        use crate::node::Node;

        let mut tree = crate::node::NodeTree::new();
        let h1 = ImageHandle(42);
        let h2 = ImageHandle(99);
        let h3 = ImageHandle(7);
        tree.add_root(Node::image(h1).build());
        tree.add_root(Node::image(h2).build());
        // Non-image node — must not appear in live set.
        tree.add_root(Node::rect().build());
        let root3 = tree.add_root(Node::rect().build());
        // Image node nested as child — must still appear.
        tree.add_child(root3, Node::image(h3).build());

        let live = collect_live_image_handles(&tree);
        assert_eq!(live.len(), 3);
        assert!(live.contains(&h1));
        assert!(live.contains(&h2));
        assert!(live.contains(&h3));
    }

    #[test]
    fn rebuild_scene_swaps_named_nodes() {
        let toggle = Rc::new(Cell::new(false));
        let toggle_for_scene = toggle.clone();
        let app = App::new("test", 100, 100).scene(move |s| {
            if toggle_for_scene.get() {
                s.rect("view_b").size_px(20.0, 20.0);
            } else {
                s.rect("view_a").size_px(10.0, 10.0);
            }
        });
        // Initial: view_a is live.
        assert!(app.ctx().node("view_a").is_some());
        assert!(app.ctx().node("view_b").is_none());

        // Flip toggle and rebuild.
        toggle.set(true);
        let mut app = app;
        app.rebuild_scene();

        // After rebuild: view_b is live, view_a is gone.
        assert!(app.ctx().node("view_a").is_none());
        assert!(app.ctx().node("view_b").is_some());
    }

    #[test]
    fn rebuild_scene_clears_bind_registry_live_count() {
        let s1 = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let s2 = Signal::new([0.0_f32, 1.0, 0.0, 1.0]);
        let s1_for_scene = s1.clone();
        let s2_for_scene = s2.clone();
        let mut app = App::new("test", 100, 100).scene(move |scene| {
            scene.rect("a").color(s1_for_scene.clone());
            scene.rect("b").color(s2_for_scene.clone());
        });
        let live_before = app.ctx().binds.color.iter().filter(|s| s.is_some()).count();
        assert_eq!(live_before, 2);

        app.rebuild_scene();

        // After rebuild the prior slots are tombstoned, then the
        // free-list reuses them for the new scene's binds. Live count is
        // restored AND the vector stays bounded — no monotonic growth
        // across rebuilds (the whole point of the free-list).
        let live_after = app.ctx().binds.color.iter().filter(|s| s.is_some()).count();
        assert_eq!(live_after, 2);
        assert_eq!(
            app.ctx().binds.color.len(),
            2,
            "tombstoned slots reused — length must not grow across rebuilds"
        );
    }

    #[test]
    fn rebuild_scene_with_no_builder_is_noop() {
        let mut app = App::new("test", 100, 100);
        // No `.scene(...)` call — no stored builder.
        app.rebuild_scene();
        assert_eq!(app.ctx().tree.len(), 0);
    }

    // --- Tab focus cycling ---

    /// Build a 4-node Col scene where focus_order can differ from
    /// creation order. Returns (app, [ids in creation order], [signals]).
    fn focus_scene(orders: [u32; 4]) -> (App, Vec<crate::node::NodeId>, Vec<Signal<bool>>) {
        let sigs: Vec<Signal<bool>> = (0..4).map(|_| Signal::new(false)).collect();
        let sigs_for_scene = sigs.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            for i in 0..4 {
                let mut b = s.rect(format!("n{i}"));
                b.size_px(100.0, 40.0).on_focus(sigs_for_scene[i].clone());
                if orders[i] != 0 {
                    b.focus_order(orders[i]);
                }
            }
        });
        let _ = app.flush_tree();
        let ids: Vec<_> = (0..4)
            .map(|i| app.ctx().node(&format!("n{i}")).unwrap())
            .collect();
        (app, ids, sigs)
    }

    #[test]
    fn tab_cycles_focus_forward_and_wraps() {
        let (mut app, ids, _) = focus_scene([1, 2, 3, 4]);
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[0]));
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[1]));
        app.focus_next(true);
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[3]));
        // Wrap back to the first.
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[0]));
    }

    #[test]
    fn shift_tab_cycles_backward_and_wraps() {
        let (mut app, ids, _) = focus_scene([1, 2, 3, 4]);
        // From nothing, Shift+Tab lands on the last candidate.
        app.focus_next(false);
        assert_eq!(app.input.focused, Some(ids[3]));
        app.focus_next(false);
        assert_eq!(app.input.focused, Some(ids[2]));
        // Wrap forward off the front.
        app.focus_next(false);
        app.focus_next(false);
        assert_eq!(app.input.focused, Some(ids[0]));
        app.focus_next(false);
        assert_eq!(app.input.focused, Some(ids[3]));
    }

    #[test]
    fn tab_visits_in_focus_order_not_creation_order() {
        // Creation order n0,n1,n2,n3 but focus_order 3,1,4,2 → visit
        // sequence by ascending order is n1(1), n3(2), n0(3), n2(4).
        let (mut app, ids, _) = focus_scene([3, 1, 4, 2]);
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[1]));
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[3]));
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[0]));
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[2]));
    }

    #[test]
    fn focus_order_zero_is_excluded() {
        // n1 has focus_order 0 → skipped entirely.
        let (mut app, ids, _) = focus_scene([1, 0, 2, 3]);
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[0]));
        app.focus_next(true);
        assert_eq!(
            app.input.focused,
            Some(ids[2]),
            "n1 (order 0) must be skipped"
        );
    }

    #[test]
    fn focus_toggles_signals_exclusively() {
        let (mut app, _ids, sigs) = focus_scene([1, 2, 3, 4]);
        app.focus_next(true);
        assert!(sigs[0].get());
        app.focus_next(true);
        assert!(!sigs[0].get(), "old focus signal flips off");
        assert!(sigs[1].get(), "new focus signal flips on");
    }

    #[test]
    fn focus_skips_invisible_nodes() {
        let (mut app, ids, _) = focus_scene([1, 2, 3, 4]);
        app.ctx.tree.set_visible(ids[1], false);
        let _ = app.flush_tree();
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[0]));
        app.focus_next(true);
        assert_eq!(app.input.focused, Some(ids[2]), "invisible n1 skipped");
    }

    #[test]
    fn focus_next_noop_when_nothing_focusable() {
        let (mut app, _, _) = focus_scene([0, 0, 0, 0]);
        app.focus_next(true);
        assert_eq!(app.input.focused, None);
    }

    // --- outside-press dismissal ---

    #[test]
    fn unhandled_press_fires_on_empty_hit() {
        let count = Rc::new(Cell::new(0u32));
        let count_for_hook = count.clone();
        let mut app = App::new("test", 200, 200)
            .on_unhandled_press(move |_ctx| {
                count_for_hook.set(count_for_hook.get() + 1);
            })
            .scene(|s| {
                s.rect("btn").size_px(50.0, 50.0).on_click(|_| {});
            });
        let _ = app.flush_tree();
        // Press landed on empty space → captured is None.
        app.input.captured = None;
        app.maybe_fire_unhandled_press();
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn unhandled_press_fires_on_dismiss_transparent_scrim() {
        let count = Rc::new(Cell::new(0u32));
        let count_for_hook = count.clone();
        let mut app = App::new("test", 200, 200)
            .on_unhandled_press(move |_ctx| {
                count_for_hook.set(count_for_hook.get() + 1);
            })
            .scene(|s| {
                s.rect("scrim").fill().dismiss_transparent();
            });
        let _ = app.flush_tree();
        let scrim = app.ctx().node("scrim").unwrap();
        app.input.captured = Some(scrim);
        app.maybe_fire_unhandled_press();
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn unhandled_press_skips_regular_interactive_node() {
        let count = Rc::new(Cell::new(0u32));
        let count_for_hook = count.clone();
        let mut app = App::new("test", 200, 200)
            .on_unhandled_press(move |_ctx| {
                count_for_hook.set(count_for_hook.get() + 1);
            })
            .scene(|s| {
                s.rect("btn").size_px(50.0, 50.0).on_click(|_| {});
            });
        let _ = app.flush_tree();
        let btn = app.ctx().node("btn").unwrap();
        // Press hit a normal button — not an outside click.
        app.input.captured = Some(btn);
        app.maybe_fire_unhandled_press();
        assert_eq!(count.get(), 0, "press on a real button must not dismiss");
    }

    #[test]
    fn unhandled_press_noop_without_hook() {
        let mut app = App::new("test", 200, 200).scene(|s| {
            s.rect("btn").size_px(50.0, 50.0).on_click(|_| {});
        });
        let _ = app.flush_tree();
        app.input.captured = None;
        // No hook registered — must not panic.
        app.maybe_fire_unhandled_press();
    }

    #[test]
    fn dismiss_transparent_node_is_a_hit_target() {
        // A scrim with no other interactivity must still flatten into
        // the hit cache so it blocks click-through.
        let mut app = App::new("test", 200, 200).scene(|s| {
            s.rect("scrim").fill().dismiss_transparent();
        });
        let _ = app.flush_tree();
        let scrim = app.ctx().node("scrim").unwrap();
        assert!(
            app.hits.iter().any(|h| h.node_id == scrim),
            "dismiss_transparent node should be a hit target"
        );
    }

    // --- on_drag + drag-and-drop ---

    fn center_of(app: &App, id: crate::node::NodeId) -> [f32; 2] {
        let r = app.ctx().tree.get(id).unwrap().rect;
        [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
    }

    #[test]
    fn on_drag_fires_with_per_event_delta() {
        let last = Rc::new(Cell::new([0.0_f32, 0.0]));
        let start_seen = Rc::new(Cell::new([0.0_f32, 0.0]));
        let last_c = last.clone();
        let start_c = start_seen.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            let last_c = last_c.clone();
            let start_c = start_c.clone();
            s.rect("knob").size_px(40.0, 40.0).on_drag(move |d| {
                last_c.set(d.delta);
                start_c.set(d.start);
            });
        });
        let _ = app.flush_tree();
        let knob = app.ctx().node("knob").unwrap();
        app.input.captured = Some(knob);
        app.input.cursor = Some([10.0, 10.0]);
        app.begin_drag_if_draggable();
        assert!(app.fire_drag(15.0, 10.0));
        assert_eq!(
            last.get(),
            [5.0, 0.0],
            "delta is from press origin on first move"
        );
        assert_eq!(start_seen.get(), [10.0, 10.0]);
        // Second move: delta is relative to the previous fire, not start.
        assert!(app.fire_drag(22.0, 10.0));
        assert_eq!(last.get(), [7.0, 0.0]);
    }

    #[test]
    fn on_drag_not_fired_without_capture() {
        let fired = Rc::new(Cell::new(false));
        let fired_c = fired.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            let fired_c = fired_c.clone();
            s.rect("knob")
                .size_px(40.0, 40.0)
                .on_drag(move |_| fired_c.set(true));
        });
        let _ = app.flush_tree();
        // No press captured → no drag fires.
        app.input.captured = None;
        app.input.cursor = Some([10.0, 10.0]);
        assert!(!app.fire_drag(20.0, 10.0));
        assert!(!fired.get());
    }

    #[test]
    fn drag_payload_delivered_to_drop_target() {
        let got = Rc::new(Cell::new(None::<u32>));
        let got_c = got.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            let got_c = got_c.clone();
            s.col("root").fill().child(move |c| {
                c.rect("src").size_px(100.0, 40.0).drag_payload(42_u32);
                c.rect("dst").size_px(100.0, 40.0).on_drop(move |d| {
                    if let Some(v) = d.payload.downcast_ref::<u32>() {
                        got_c.set(Some(*v));
                    }
                });
            });
        });
        let _ = app.flush_tree();
        let src = app.ctx().node("src").unwrap();
        let dst = app.ctx().node("dst").unwrap();
        // Press on the source: latches the payload.
        app.input.captured = Some(src);
        app.input.cursor = Some(center_of(&app, src));
        app.begin_drag_if_draggable();
        assert!(app.drag_payload.is_some(), "payload latched on press");
        // Release over the drop target.
        app.input.cursor = Some(center_of(&app, dst));
        app.finish_drag_on_release();
        assert_eq!(got.get(), Some(42));
        assert!(app.drag_payload.is_none(), "payload cleared after drop");
    }

    #[test]
    fn drop_not_fired_when_released_off_target() {
        let got = Rc::new(Cell::new(None::<u32>));
        let got_c = got.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            let got_c = got_c.clone();
            s.col("root").fill().child(move |c| {
                c.rect("src").size_px(100.0, 40.0).drag_payload(7_u32);
                c.rect("dst").size_px(100.0, 40.0).on_drop(move |d| {
                    if let Some(v) = d.payload.downcast_ref::<u32>() {
                        got_c.set(Some(*v));
                    }
                });
            });
        });
        let _ = app.flush_tree();
        let src = app.ctx().node("src").unwrap();
        app.input.captured = Some(src);
        app.input.cursor = Some(center_of(&app, src));
        app.begin_drag_if_draggable();
        // Release way off in empty space — no drop target there.
        app.input.cursor = Some([350.0, 350.0]);
        app.finish_drag_on_release();
        assert_eq!(got.get(), None, "drop must not fire off-target");
        assert!(app.drag_payload.is_none(), "payload still cleared");
    }

    #[test]
    fn cursor_override_picked_from_hovered_node() {
        let mut app = App::new("test", 200, 200).scene(|s| {
            s.rect("handle")
                .size_px(20.0, 100.0)
                .cursor(CursorIcon::EwResize);
        });
        let _ = app.flush_tree();
        // Inside the handle's rect (0,0..20,100): returns the override.
        assert_eq!(
            app.hovered_node_cursor(10.0, 50.0),
            Some(CursorIcon::EwResize)
        );
        // Outside: no override → falls back to default in refresh_cursor.
        assert_eq!(app.hovered_node_cursor(150.0, 50.0), None);
    }

    #[test]
    fn width_px_bind_updates_layout_on_signal_change() {
        let w = Signal::new(200.0_f32);
        let w_for_scene = w.clone();
        let mut app = App::new("test", 800, 400).scene(move |s| {
            s.rect("panel")
                .width_px_bind(w_for_scene.clone())
                .h_px(40.0);
        });
        let _ = app.flush_tree();
        let id = app.ctx().node("panel").unwrap();
        let initial = app.ctx().tree.get(id).unwrap().rect[2];
        assert!((initial - 200.0).abs() < 0.5, "initial width = signal");
        // Mutate the signal → bind processing snaps the new value into
        // the tree's layout width → next flush re-runs layout.
        w.set(360.0);
        app.process_binds(Instant::now());
        let _ = app.flush_tree();
        let resized = app.ctx().tree.get(id).unwrap().rect[2];
        assert!(
            (resized - 360.0).abs() < 0.5,
            "width follows signal: got {resized}"
        );
    }

    #[test]
    fn height_px_bind_updates_layout_on_signal_change() {
        let h = Signal::new(60.0_f32);
        let h_for_scene = h.clone();
        let mut app = App::new("test", 400, 600).scene(move |s| {
            s.rect("panel")
                .w_px(120.0)
                .height_px_bind(h_for_scene.clone());
        });
        let _ = app.flush_tree();
        let id = app.ctx().node("panel").unwrap();
        let initial = app.ctx().tree.get(id).unwrap().rect[3];
        assert!((initial - 60.0).abs() < 0.5);
        h.set(180.0);
        app.process_binds(Instant::now());
        let _ = app.flush_tree();
        let resized = app.ctx().tree.get(id).unwrap().rect[3];
        assert!((resized - 180.0).abs() < 0.5, "got {resized}");
    }

    #[test]
    fn press_fires_initial_on_drag_for_click_to_set() {
        // A plain click (press, no move) must fire on_drag once at the
        // press position so sliders/scrubbers jump on click.
        let seen = Rc::new(Cell::new([f32::NAN, f32::NAN]));
        let seen_c = seen.clone();
        let mut app = App::new("test", 400, 400).scene(move |s| {
            let seen_c = seen_c.clone();
            s.rect("track")
                .size_px(200.0, 8.0)
                .on_drag(move |d| seen_c.set(d.current));
        });
        let _ = app.flush_tree();
        let track = app.ctx().node("track").unwrap();
        app.input.captured = Some(track);
        app.input.cursor = Some([120.0, 4.0]);
        app.begin_drag_if_draggable();
        assert_eq!(seen.get(), [120.0, 4.0], "on_drag fires at press position");
    }

    #[test]
    fn drag_follow_tracks_and_clears() {
        let mut app = App::new("test", 400, 400).scene(|s| {
            s.col("root").fill().child(|c| {
                c.rect("item")
                    .size_px(80.0, 40.0)
                    .drag_payload(1_u32)
                    .drag_follow();
            });
        });
        let _ = app.flush_tree();
        let item = app.ctx().node("item").unwrap();
        app.input.captured = Some(item);
        app.input.cursor = Some([10.0, 10.0]);
        app.begin_drag_if_draggable();
        assert_eq!(
            app.ctx().tree.drag_follow_target(),
            Some(item),
            "lifts on press"
        );
        app.input.cursor = Some([30.0, 25.0]);
        assert!(app.update_drag_follow(30.0, 25.0));
        assert_eq!(app.ctx().tree.drag_follow_target(), Some(item));
        // Release starts a snap-back animation — the lift is still active
        // (animating home), not cleared instantly.
        app.finish_drag_on_release();
        assert!(app.drag_return.is_some(), "release starts a snap-back");
        assert_eq!(
            app.ctx().tree.drag_follow_target(),
            Some(item),
            "still lifted mid-return"
        );
        // Simulate the tween reaching the slot, then tick the return.
        app.drag_return_offset.set([0.0, 0.0]);
        app.tick_drag_return();
        assert_eq!(
            app.ctx().tree.drag_follow_target(),
            None,
            "lands → lift cleared"
        );
        assert!(app.drag_return.is_none());
    }

    #[test]
    fn drag_follow_click_without_move_clears_immediately() {
        let mut app = App::new("test", 400, 400).scene(|s| {
            s.col("root").fill().child(|c| {
                c.rect("item")
                    .size_px(80.0, 40.0)
                    .drag_payload(1_u32)
                    .drag_follow();
            });
        });
        let _ = app.flush_tree();
        let item = app.ctx().node("item").unwrap();
        app.input.captured = Some(item);
        app.input.cursor = Some([10.0, 10.0]);
        app.begin_drag_if_draggable();
        assert_eq!(app.ctx().tree.drag_follow_target(), Some(item));
        // Release at the same spot — no snap-back needed.
        app.finish_drag_on_release();
        assert!(
            app.drag_return.is_none(),
            "no animation for a zero-move click"
        );
        assert_eq!(
            app.ctx().tree.drag_follow_target(),
            None,
            "lift cleared at once"
        );
    }

    #[test]
    fn drag_without_follow_flag_does_not_lift() {
        let mut app = App::new("test", 400, 400).scene(|s| {
            s.col("root").fill().child(|c| {
                c.rect("item").size_px(80.0, 40.0).drag_payload(1_u32);
            });
        });
        let _ = app.flush_tree();
        let item = app.ctx().node("item").unwrap();
        app.input.captured = Some(item);
        app.input.cursor = Some([10.0, 10.0]);
        app.begin_drag_if_draggable();
        assert_eq!(app.ctx().tree.drag_follow_target(), None);
        assert!(!app.update_drag_follow(30.0, 25.0));
    }

    #[test]
    fn begin_drag_ignores_non_draggable_node() {
        let mut app = App::new("test", 400, 400).scene(|s| {
            s.rect("btn").size_px(40.0, 40.0).on_click(|_| {});
        });
        let _ = app.flush_tree();
        let btn = app.ctx().node("btn").unwrap();
        app.input.captured = Some(btn);
        app.input.cursor = Some([10.0, 10.0]);
        app.begin_drag_if_draggable();
        assert!(app.drag_origin.is_none());
        assert!(app.drag_payload.is_none());
    }

    // --- resolve_edit_op modifier routing ---

    #[test]
    fn resolve_shift_arrow_extends_selection() {
        use crate::editor::EditOp;
        use winit::keyboard::{Key, ModifiersState, NamedKey};
        assert_eq!(
            resolve_edit_op(
                &Key::Named(NamedKey::ArrowRight),
                None,
                ModifiersState::SHIFT
            ),
            Some(EditOp::SelectRight)
        );
        assert_eq!(
            resolve_edit_op(
                &Key::Named(NamedKey::ArrowLeft),
                None,
                ModifiersState::SHIFT
            ),
            Some(EditOp::SelectLeft)
        );
        // Without shift, the same keys plain-move.
        assert_eq!(
            resolve_edit_op(
                &Key::Named(NamedKey::ArrowRight),
                None,
                ModifiersState::empty()
            ),
            Some(EditOp::MoveRight)
        );
    }

    #[test]
    fn resolve_shift_home_end_select_to_edges() {
        use crate::editor::EditOp;
        use winit::keyboard::{Key, ModifiersState, NamedKey};
        assert_eq!(
            resolve_edit_op(&Key::Named(NamedKey::Home), None, ModifiersState::SHIFT),
            Some(EditOp::SelectHome)
        );
        assert_eq!(
            resolve_edit_op(&Key::Named(NamedKey::End), None, ModifiersState::SHIFT),
            Some(EditOp::SelectEnd)
        );
    }

    #[test]
    fn resolve_accel_combos() {
        use crate::editor::EditOp;
        use winit::keyboard::{Key, ModifiersState};
        let ctrl = ModifiersState::CONTROL;
        assert_eq!(
            resolve_edit_op(&Key::Character("a".into()), None, ctrl),
            Some(EditOp::SelectAll)
        );
        assert_eq!(
            resolve_edit_op(&Key::Character("c".into()), None, ctrl),
            Some(EditOp::Copy)
        );
        assert_eq!(
            resolve_edit_op(&Key::Character("x".into()), None, ctrl),
            Some(EditOp::Cut)
        );
        // Unmapped accel combo (Ctrl+S) falls through, and the
        // character isn't inserted while accel is held.
        assert_eq!(
            resolve_edit_op(&Key::Character("s".into()), Some("s"), ctrl),
            None
        );
    }

    #[test]
    fn resolve_plain_char_inserts() {
        use crate::editor::EditOp;
        use winit::keyboard::{Key, ModifiersState};
        assert_eq!(
            resolve_edit_op(
                &Key::Character("q".into()),
                Some("q"),
                ModifiersState::empty()
            ),
            Some(EditOp::Insert("q".to_string()))
        );
    }

    #[test]
    fn lazy_list_materializes_visible_window_only() {
        // 10K-row list, item_height 40 logical px. With viewport
        // height 200, ~5 rows visible. Buffer 2 each side → ~9 rows
        // materialized.
        let app = App::new("test", 400, 200);
        let app = app.scene(move |s| {
            s.lazy_list("list", 10_000, 40.0, |row, i| {
                row.rect(format!("row{i}")).w(Len::Fill).h_px(40.0);
            })
            .fill();
        });
        let mut app = app;
        // First flush — runs layout, materializes initial window.
        let _ = app.flush_tree();
        let list_id = app.ctx().node("list").unwrap();
        let list = app.ctx().tree.get(list_id).unwrap();
        let ll = list.lazy_list.as_ref().unwrap();
        assert!(
            ll.materialized.len() < 50,
            "should not materialize all 10K rows: got {}",
            ll.materialized.len()
        );
        assert!(
            ll.materialized.len() >= 5,
            "should materialize at least the visible rows: got {}",
            ll.materialized.len()
        );
        // Content size must reflect total (10K * 40 = 400K logical
        // px → 400K physical px at scale 1.0), driving scroll bounds.
        assert!(
            (list.content_size[1] - 400_000.0).abs() < 1.0,
            "content_size {} should equal 10000 * 40",
            list.content_size[1]
        );
    }

    #[test]
    fn lazy_list_scroll_re_materializes() {
        let app = App::new("test", 400, 200);
        let app = app.scene(move |s| {
            s.lazy_list("list", 10_000, 40.0, |row, i| {
                row.rect(format!("row{i}")).w(Len::Fill).h_px(40.0);
            })
            .fill();
        });
        let mut app = app;
        let _ = app.flush_tree();
        let list_id = app.ctx().node("list").unwrap();
        let range_before = app
            .ctx()
            .tree
            .get(list_id)
            .unwrap()
            .lazy_list
            .as_ref()
            .unwrap()
            .range;
        // Scroll down 4000 logical px (100 rows).
        app.ctx.tree.set_scroll_target(list_id, [0.0, 4000.0]);
        // Set current immediately to skip the spring (test harness
        // doesn't run the timeline tick).
        app.ctx
            .tree
            .set_scroll_immediate(list_id, crate::node::ScrollAxis::Y, 4000.0);
        let _ = app.flush_tree();
        let range_after = app
            .ctx()
            .tree
            .get(list_id)
            .unwrap()
            .lazy_list
            .as_ref()
            .unwrap()
            .range;
        assert_ne!(range_before, range_after, "scrolling should shift window");
        assert!(
            range_after[0] >= 90,
            "should materialize rows around index 100"
        );
    }

    #[test]
    fn rebuild_scene_resets_input_state() {
        let mut app = App::new("test", 100, 100).scene(move |s| {
            s.rect("a").size_px(10.0, 10.0);
        });
        // Pretend something captured a node id.
        app.input.captured = app.ctx().node("a");
        app.input.hovered = app.ctx().node("a");
        assert!(app.input.captured.is_some());

        app.rebuild_scene();

        assert!(app.input.captured.is_none(), "captured must reset");
        assert!(app.input.hovered.is_none(), "hovered must reset");
    }

    #[test]
    fn rebuild_scene_preserves_hover_under_stationary_cursor() {
        // Regression: rebuild_scene used to wipe `input.cursor` along
        // with `hovered`, so a 5 Hz background rebuild (progress tick
        // etc.) would clear hover signals on stationary buttons and let
        // the next CursorMoved flip them back on — visible as flicker.
        // Now cursor is preserved and hover is re-derived against the
        // fresh hit cache so the signal stays on across rebuilds.
        let hover_sig = crate::signal::Signal::new(false);
        let hover_for_scene = hover_sig.clone();
        let mut app = App::new("test", 100, 100).scene(move |s| {
            s.rect("btn")
                .size_px(40.0, 40.0)
                .on_hover(hover_for_scene.clone());
        });
        let _ = app.flush_tree();
        let id = app.ctx().node("btn").unwrap();
        // Simulate the cursor sitting over the button.
        let change = app
            .input
            .on_cursor_moved(20.0, 20.0, &app.hits, &app.ctx.tree);
        assert!(change.hovered_changed);
        assert_eq!(app.input.hovered, Some(id));
        assert!(hover_sig.get(), "hover should latch on initial enter");

        // Rebuild fires (e.g. periodic redraw). Cursor hasn't moved.
        app.rebuild_scene();

        let new_id = app.ctx().node("btn").unwrap();
        assert_eq!(
            app.input.hovered,
            Some(new_id),
            "hover must re-derive against the rebuilt tree"
        );
        assert!(
            hover_sig.get(),
            "hover signal must stay on across a no-op rebuild"
        );
    }

    #[test]
    fn dwell_arms_on_hover_into_dwell_node() {
        let fired = Rc::new(Cell::new(0_u32));
        let fired2 = fired.clone();
        let mut app = App::new("test", 100, 100).scene(move |s| {
            let fired_inner = fired2.clone();
            s.rect("btn").size_px(40.0, 40.0).on_hover_dwell(
                std::time::Duration::from_millis(200),
                move |_| {
                    fired_inner.set(fired_inner.get() + 1);
                },
            );
        });
        let id = app.ctx().node("btn").unwrap();
        let now = Instant::now();
        // No hover → refresh_dwell leaves tracker None.
        app.refresh_dwell(now);
        assert!(app.dwell.is_none());
        // Simulate hover-enter.
        app.input.hovered = Some(id);
        app.refresh_dwell(now);
        assert!(app.dwell.is_some(), "dwell should arm on hover-enter");
        assert!(!app.dwell.unwrap().fired);
        // Tick before deadline → no fire.
        assert!(!app.tick_dwell(now + std::time::Duration::from_millis(100)));
        assert_eq!(fired.get(), 0);
        // Tick at/after deadline → fires once, marks fired.
        assert!(app.tick_dwell(now + std::time::Duration::from_millis(250)));
        assert_eq!(fired.get(), 1);
        assert!(app.dwell.unwrap().fired);
        // Tick again → no double-fire while still hovered.
        assert!(!app.tick_dwell(now + std::time::Duration::from_millis(500)));
        assert_eq!(fired.get(), 1);
    }

    #[test]
    fn dwell_resets_when_hover_leaves() {
        let fired = Rc::new(Cell::new(0_u32));
        let fired2 = fired.clone();
        let mut app = App::new("test", 100, 100).scene(move |s| {
            let fired_inner = fired2.clone();
            s.rect("btn").size_px(40.0, 40.0).on_hover_dwell(
                std::time::Duration::from_millis(200),
                move |_| {
                    fired_inner.set(fired_inner.get() + 1);
                },
            );
        });
        let id = app.ctx().node("btn").unwrap();
        let now = Instant::now();
        app.input.hovered = Some(id);
        app.refresh_dwell(now);
        // Leave before deadline.
        app.input.hovered = None;
        app.refresh_dwell(now + std::time::Duration::from_millis(50));
        assert!(app.dwell.is_none(), "dwell should clear on hover-leave");
        // Re-enter → fresh deadline; previous 50ms doesn't count.
        app.input.hovered = Some(id);
        let now2 = now + std::time::Duration::from_millis(60);
        app.refresh_dwell(now2);
        assert!(!app.tick_dwell(now2 + std::time::Duration::from_millis(150)));
        assert_eq!(fired.get(), 0);
        assert!(app.tick_dwell(now2 + std::time::Duration::from_millis(210)));
        assert_eq!(fired.get(), 1);
    }

    #[test]
    fn dwell_handler_can_mutate_tree_via_ctx() {
        let mut app = App::new("test", 100, 100).scene(move |s| {
            // Tooltip child starts hidden — handler shows it on dwell.
            s.rect("btn").size_px(40.0, 40.0).on_hover_dwell(
                std::time::Duration::from_millis(100),
                move |ctx| {
                    if let Some(n) = ctx.tree.get_mut_raw(ctx.node) {
                        n.style.color = [1.0, 0.0, 0.0, 1.0];
                    }
                },
            );
        });
        let id = app.ctx().node("btn").unwrap();
        let now = Instant::now();
        app.input.hovered = Some(id);
        app.refresh_dwell(now);
        app.tick_dwell(now + std::time::Duration::from_millis(150));
        let n = app.ctx().tree.get(id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }
}

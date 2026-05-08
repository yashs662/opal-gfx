//! Retained node tree.
//!
//! Generational-index arena. Nodes carry a [`LayoutStyle`] declaring
//! their sizing/alignment intent; the [`crate::layout::compute_layout`]
//! pass resolves them into absolute [`Node::rect`]s before each flush.
//! `NodeId`s are stable across mutations of *other* nodes — they only
//! invalidate when the specific slot they refer to is reused.

use crate::gpu::{ImageHandle, NO_CLIP, ShapeInstance, SHAPE_KIND_GLASS, SHAPE_KIND_IMAGE, SHAPE_KIND_RECT};
use crate::layout::{Align, Axis, Justify, Len, LayoutStyle};
use crate::signal::Signal;

/// Tree-level dirty flags.
pub mod dirty {
    pub const NONE: u32 = 0;
    /// Color, opacity, border or shadow style changed.
    pub const VISUAL: u32 = 1 << 0;
    /// Layout style (size, position, padding, gap, justify, align, abs)
    /// changed — requires a layout-pass re-run.
    pub const TRANSFORM: u32 = 1 << 1;
    /// Tree topology changed (add, remove, visibility flip).
    pub const TREE: u32 = 1 << 2;
    /// Glass region or the opaque content under it changed → re-run blur.
    pub const BACKDROP: u32 = 1 << 3;
    /// Scroll offset or scrollbar interaction state changed. Triggers a
    /// re-flatten (offset propagates to child positions, bar
    /// hover/active flip thumb color) but **does not** need
    /// `compute_layout` to re-run — node `rect`s are still valid. Kept
    /// separate from `TRANSFORM` so a fast drag doesn't re-shape text
    /// + re-measure flex on every cursor-move event.
    pub const SCROLL: u32 = 1 << 4;
    pub const ANY: u32 = VISUAL | TRANSFORM | TREE | BACKDROP | SCROLL;
}

/// One text node discovered during flatten. Carries the post-layout
/// absolute position so the caller (GpuContext) can shape + rasterize
/// + append glyph instances without re-walking the tree.
#[derive(Clone, Debug)]
pub struct TextRef {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub opacity: f32,
    pub content: String,
    pub font_size: f32,
    pub line_height: f32,
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    /// `crate::gpu::NO_CLIP` when none. Stamped onto every glyph
    /// instance built from this ref.
    pub clip_rect: [f32; 4],
}

/// One image node discovered during flatten. The atlas lookup happens
/// caller-side (`gpu.build_image_instances`) so the tree stays
/// gpu-free.
#[derive(Clone, Debug)]
pub struct ImageRef {
    pub position: [f32; 2],
    pub size: [f32; 2],
    /// Tint multiplier; `[1,1,1,1]` leaves the image unmodified.
    pub color: [f32; 4],
    pub opacity: f32,
    pub border_radius: [f32; 4],
    pub handle: ImageHandle,
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    pub clip_rect: [f32; 4],
}

/// One interactive rect in the hit-test cache. Produced by
/// `NodeTree::flatten_with_hits` in **topmost-first** order (last-painted
/// first) so hit-test can walk linearly and stop at the first containing
/// rect.
#[derive(Clone, Debug)]
pub struct HitEntry {
    pub node_id: NodeId,
    /// Absolute pixel AABB: `[min_x, min_y, max_x, max_y]`.
    /// Already includes any ancestor scroll offset — screen-space.
    pub bounds: [f32; 4],
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    /// Cursor outside this rect must miss this entry even if `bounds`
    /// would contain it. `crate::gpu::NO_CLIP` when no ancestor clips.
    pub clip_rect: [f32; 4],
}

impl HitEntry {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if x < self.bounds[0] || x >= self.bounds[2] || y < self.bounds[1] || y >= self.bounds[3] {
            return false;
        }
        x >= self.clip_rect[0]
            && x < self.clip_rect[2]
            && y >= self.clip_rect[1]
            && y < self.clip_rect[3]
    }
}

/// One scrollable container discovered during flatten. Wheel input
/// finds the topmost ScrollHit under the cursor and walks
/// `ancestor_chain` for edge-bubble (innermost-first; self is at index
/// 0). Built only for nodes with `layout.scrolls()`.
#[derive(Clone, Debug)]
pub struct ScrollHit {
    pub node_id: NodeId,
    /// Absolute, post-offset bounds — same convention as `HitEntry`.
    pub bounds: [f32; 4],
    pub clip_rect: [f32; 4],
    /// Innermost-first chain including self at `[0]`. Wheel bubble
    /// walks this on edge consumption: self first, then each scroll
    /// ancestor outward.
    pub ancestor_chain: Vec<NodeId>,
}

impl ScrollHit {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if x < self.bounds[0] || x >= self.bounds[2] || y < self.bounds[1] || y >= self.bounds[3] {
            return false;
        }
        x >= self.clip_rect[0]
            && x < self.clip_rect[2]
            && y >= self.clip_rect[1]
            && y < self.clip_rect[3]
    }
}

/// Which axis a scrollbar belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScrollAxis {
    X,
    Y,
}

impl ScrollAxis {
    pub fn index(self) -> usize {
        match self {
            ScrollAxis::X => 0,
            ScrollAxis::Y => 1,
        }
    }
}

/// Edge a scrollbar attaches to. `End` is the conventional side
/// (right for Y, bottom for X); `Start` flips it (left / top).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BarSide {
    Start,
    End,
}

/// Per-scrollbar visual + behavior config. Lives on `ScrollState`. All
/// pixel fields are in **logical** units — emit scales them to physical
/// at flatten time.
#[derive(Copy, Clone, Debug)]
pub struct ScrollbarStyle {
    pub track_color: [f32; 4],
    pub thumb_color: [f32; 4],
    pub thumb_hover_color: [f32; 4],
    pub thumb_active_color: [f32; 4],
    pub thickness: f32,
    pub min_thumb: f32,
    pub margin: f32,
    pub radius: f32,
    pub y_side: BarSide,
    pub x_side: BarSide,
    pub fade_seconds: f32,
    /// Don't pop the bar when scroll input arrives — only show it when
    /// the pointer enters the bar AABB or while a drag is in flight.
    /// Default false.
    pub auto_hide: bool,
    /// Pin `bar_alpha` to 1 (never fade). Useful for desktop apps
    /// where always-on bars are expected.
    pub always_visible: bool,
}

impl Default for ScrollbarStyle {
    fn default() -> Self {
        Self {
            track_color: [1.0, 1.0, 1.0, 0.10],
            thumb_color: [1.0, 1.0, 1.0, 0.45],
            thumb_hover_color: [1.0, 1.0, 1.0, 0.65],
            thumb_active_color: [1.0, 1.0, 1.0, 0.85],
            thickness: 4.0,
            min_thumb: 24.0,
            margin: 4.0,
            radius: 2.0,
            y_side: BarSide::End,
            x_side: BarSide::End,
            fade_seconds: 0.8,
            auto_hide: false,
            always_visible: false,
        }
    }
}

impl ScrollbarStyle {
    pub fn track_color(mut self, c: [f32; 4]) -> Self { self.track_color = c; self }
    pub fn thumb_color(mut self, c: [f32; 4]) -> Self { self.thumb_color = c; self }
    pub fn thumb_hover_color(mut self, c: [f32; 4]) -> Self { self.thumb_hover_color = c; self }
    pub fn thumb_active_color(mut self, c: [f32; 4]) -> Self { self.thumb_active_color = c; self }
    pub fn thickness(mut self, px: f32) -> Self { self.thickness = px; self }
    pub fn min_thumb(mut self, px: f32) -> Self { self.min_thumb = px; self }
    pub fn margin(mut self, px: f32) -> Self { self.margin = px; self }
    pub fn radius(mut self, px: f32) -> Self { self.radius = px; self }
    pub fn y_side(mut self, side: BarSide) -> Self { self.y_side = side; self }
    pub fn x_side(mut self, side: BarSide) -> Self { self.x_side = side; self }
    pub fn fade(mut self, seconds: f32) -> Self { self.fade_seconds = seconds.max(0.0); self }
    pub fn auto_hide(mut self, on: bool) -> Self { self.auto_hide = on; self }
    pub fn always_visible(mut self, on: bool) -> Self { self.always_visible = on; self }
}

/// One scrollbar AABB pair surfaced from flatten. Drives pointer
/// hover/click/drag routing in `input.rs`. Emitted for every active
/// axis on every visible scroll container, regardless of whether the
/// bar is currently rendered (`bar_alpha == 0`) — input still wants
/// to detect hover-enter on the bar region to bring it back.
#[derive(Clone, Debug)]
pub struct ScrollbarHit {
    pub node_id: NodeId,
    pub axis: ScrollAxis,
    /// Track AABB `[min_x, min_y, max_x, max_y]` in screen space.
    pub track: [f32; 4],
    /// Thumb AABB inside the track at the current scroll position.
    pub thumb: [f32; 4],
    pub clip_rect: [f32; 4],
    /// Maximum scroll offset in logical *physical* px on this axis
    /// (`content - rect`). Cached so input can map track-clicks
    /// directly without a tree lookup.
    pub max_offset: f32,
    /// Track travel = `track_len - thumb_len`. The pixel range a thumb
    /// drag covers; cached for the same reason as `max_offset`.
    pub track_travel: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Rect,
    Glass,
    Text,
    Image,
}

impl ShapeKind {
    pub fn as_u32(self) -> u32 {
        match self {
            ShapeKind::Rect => SHAPE_KIND_RECT,
            ShapeKind::Glass => SHAPE_KIND_GLASS,
            ShapeKind::Text => SHAPE_KIND_RECT,
            ShapeKind::Image => SHAPE_KIND_IMAGE,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

#[derive(Clone, Debug)]
pub struct ShapeStyle {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub border_width: f32,
    pub border_radius: [f32; 4],
    pub shadow_color: [f32; 4],
    pub shadow_offset: [f32; 2],
    pub shadow_blur: f32,
    pub shadow_opacity: f32,
    pub opacity: f32,
    pub kind: ShapeKind,
    /// Glass-only. Backdrop blur radius in logical px. 0 = sharp pass-
    /// through; ~16 = soft frosted look. Scaled to physical px before
    /// reaching the GPU.
    pub blur_amount: f32,
    /// Glass-only. Edge refraction strength in logical px. The SDF
    /// gradient bends backdrop sample UVs outward at the panel rim,
    /// mimicking how a thick glass slab refracts light. 0 disables.
    pub refraction: f32,
    /// Glass-only. Frosted-texture variation in logical px. Per-fragment
    /// hash scatters the backdrop sample UV by `roughness * pixel_of_mip`
    /// so the surface looks pebbled rather than mirror-smooth. 0
    /// disables; ~1 = subtle frost, ~3 = pronounced.
    pub roughness: f32,
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            // Transparent by default: a container node with no explicit
            // color should not render a filled rect. Callers opt in via
            // `.rgba(...)` / `.color(...)`.
            color: [0.0; 4],
            border_color: [0.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            border_radius: [0.0; 4],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
            kind: ShapeKind::Rect,
            blur_amount: 12.0,
            refraction: 0.0,
            roughness: 0.0,
        }
    }
}

/// System window action bound to a node. When the user left-presses a
/// node tagged with one of these the app shell calls into winit
/// directly (drag the window, exit, minimize, toggle maximize) instead
/// of running normal hit-test press bookkeeping. The node's
/// `NodeInteract` signals (if any) still receive hover updates so the
/// visual can react.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowAction {
    /// Initiate a system window drag (frameless title-bar behaviour).
    DragMove,
    /// Exit the event loop.
    Close,
    /// Minimise the window.
    Minimize,
    /// Toggle the window's maximised state.
    ToggleMaximize,
}

#[derive(Clone, Debug, Default)]
pub struct NodeInteract {
    pub hover: Option<Signal<bool>>,
    pub pressed: Option<Signal<bool>>,
    pub focused: Option<Signal<bool>>,
}

impl NodeInteract {
    pub fn is_any(&self) -> bool {
        self.hover.is_some() || self.pressed.is_some() || self.focused.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct NodeText {
    pub content: String,
    pub font_size: f32,
    pub line_height: f32,
}

/// Per-node scroll state. Allocated only on containers whose layout has
/// `overflow_x == Scroll || overflow_y == Scroll`. `current` is what the
/// flatten pass reads; `target` is what wheel input pushes. Each tick
/// `current` exponentially eases toward `target`.
#[derive(Copy, Clone, Debug)]
pub struct ScrollState {
    pub current: [f32; 2],
    pub target: [f32; 2],
    /// Exponential ease rate for the **forward** chase (in-range
    /// scroll toward target). Higher = snappier. Default 24 ≈ 190 ms
    /// time-to-rest. Bounce-back from past-edge uses the
    /// `bounce_stiffness` + `bounce_damping` damped-harmonic spring
    /// instead, since exponential ease has no overshoot.
    pub stiffness: f32,
    /// Damped-harmonic spring stiffness (`k`) for bounce-back from
    /// overscroll. Combined with `bounce_damping`, controls how the
    /// view snaps back into range after the rubber-band releases.
    /// Default 800 → ω₀ ≈ 28.3 rad/s, period ≈ 310 ms when `zeta < 1`.
    pub bounce_stiffness: f32,
    /// Damped-harmonic spring damping (`c`) for bounce-back. Default
    /// 42 with `bounce_stiffness=800` gives ζ ≈ 0.74 — minimal
    /// overshoot (~3 %) and a graceful settle. Lower values
    /// (toward 0.5·`2√k`) make the bounce more elastic; higher values
    /// (toward `2√k` = critical damping) give a smoother no-overshoot
    /// landing.
    pub bounce_damping: f32,
    /// When true, `target` is allowed past the content edge with
    /// rubber-band damping (each delta past the edge contributes less);
    /// once the spring quiesces past the edge it retargets back into
    /// `[0, max_off]` so the spring pulls the view to a rest position.
    /// When false (default), `target` is hard-clamped on every write.
    pub overscroll: bool,
    /// Per-axis snap step in **logical** px. When `> 0`, the target is
    /// retargeted to the nearest multiple of `snap_step` once the spring
    /// quiesces, and the spring then chases the snapped value. `0` =
    /// no snap (continuous scroll). Indexed by `ScrollAxis::index`.
    pub snap_step: [f32; 2],
    /// Seconds since the last user-driven input on this scroll
    /// (`add_scroll_delta`, `set_scroll_target`, or `set_scroll_immediate`).
    /// Used to gate the on-quiesce settle path: while input is still
    /// arriving (held arrow key, fast wheel burst, in-progress drag),
    /// settle must NOT clamp a saturated past-edge target back into
    /// range — otherwise the next repeat event would re-saturate it
    /// and the resulting target oscillation reads as a jerk. Initialized
    /// to a large value so a freshly-built scrollable settles
    /// immediately if invariants demand it.
    pub time_since_input: f32,
    /// Per-axis bounce-spring elapsed time in seconds. `< 0` means the
    /// axis is not bouncing. While bouncing, `tick_scrolls` advances
    /// this and samples the closed-form spring response to drive
    /// `current[axis]` from `bounce_from[axis]` toward `target[axis]`.
    pub bounce_elapsed: [f32; 2],
    /// Position at the moment the bounce on this axis started.
    pub bounce_from: [f32; 2],
    /// Target snapshot at the moment the bounce on this axis started.
    /// If `target[axis]` changes while bouncing (e.g. user wheels mid-
    /// bounce), the bounce restarts from the current position so the
    /// spring tracks the new target without an apparent jump.
    pub bounce_target: [f32; 2],
    /// Scrollbar fade alpha in `[0, 1]`. Pinned to 1 while the spring is
    /// chasing, while pointer is over the bar, or while a thumb is being
    /// dragged; decays over `style.fade_seconds` once idle. flatten emits
    /// the bars at `inst.color.a *= bar_alpha` so they fade in/out
    /// without a separate timeline.
    pub bar_alpha: f32,
    /// Visual + behavior config for both bars.
    pub style: ScrollbarStyle,
    /// Per-axis pointer hover state: `[x, y]`. Set by the input layer
    /// when the cursor enters the bar's track AABB; read by emit to
    /// pick the thumb color and pin `bar_alpha`. `[X, Y]` indexed by
    /// `ScrollAxis::index`.
    pub bar_hover: [bool; 2],
    /// Per-axis active (mouse-down on thumb) state. While true the
    /// thumb paints at `style.thumb_active_color` and the bar can't
    /// fade out.
    pub bar_active: [bool; 2],
}

/// Seconds of input quiescence required before `settle_target` will
/// clamp a saturated past-edge target back into range and start the
/// bounce. Anything below the OS auto-repeat period (~33 ms on Windows)
/// would let settle fire between repeat events and produce target
/// oscillation. 100 ms is comfortably above that and still feels
/// instant on release.
pub const SCROLL_INPUT_QUIESCE_SECONDS: f32 = 0.1;

/// Maximum logical-px the rubber-band model lets `target` travel past
/// either edge before saturating. The asymptote in `rubber_band_target`
/// caps cooked over at exactly this value, so a long burst of wheel
/// events can't push `target` to infinity — the further the user pushes,
/// the less effect each delta has. Smaller values shorten the bounce-
/// back animation; 60 logical px feels firm without disabling the
/// effect.
pub const OVERSCROLL_LIMIT_LOGICAL: f32 = 60.0;


impl Default for ScrollState {
    fn default() -> Self {
        Self {
            current: [0.0; 2],
            target: [0.0; 2],
            // 24 ≈ 190 ms to-rest. Lower values (the original 12)
            // perceptibly lag a fast wheel input — the list keeps
            // drifting after the user has stopped, which reads as
            // sluggish. iOS lists run closer to ~30; 24 hits a balance
            // of "snappy but still cushioned".
            stiffness: 24.0,
            // Bounce-back spring tuned for elegance over speed:
            //   ω₀ = √800 ≈ 28.3 rad/s → period ~310 ms
            //   ζ  = 42 / (2·28.3) ≈ 0.74 → ~3 % overshoot
            // Settles cleanly in ~280 ms with a single soft dip.
            // Earlier values (k=3500, c=50) were stiff enough to read
            // as a "snap" — the spring's restoring force was so strong
            // the recovery felt aggressive. Halving frequency and
            // pulling damping closer to critical (ζ→1) gives a
            // graceful pull rather than a slam.
            bounce_stiffness: 800.0,
            bounce_damping: 42.0,
            overscroll: false,
            snap_step: [0.0; 2],
            time_since_input: 1.0,
            bounce_elapsed: [-1.0; 2],
            bounce_from: [0.0; 2],
            bounce_target: [0.0; 2],
            bar_alpha: 0.0,
            style: ScrollbarStyle::default(),
            bar_hover: [false; 2],
            bar_active: [false; 2],
        }
    }
}

impl ScrollState {
    /// True while at least one axis is being dragged. Used to gate
    /// "pointer-down on track to jump" — clicks during a drag should
    /// not retarget.
    pub fn dragging(&self) -> bool {
        self.bar_active[0] || self.bar_active[1]
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub style: ShapeStyle,
    pub layout: LayoutStyle,
    /// Post-layout absolute rect `[x, y, w, h]`. Written by
    /// [`crate::layout::compute_layout`]; read by `flatten_with_text`.
    pub rect: [f32; 4],
    /// Bounding extent of all children, in physical px relative to
    /// `rect.xy`. Populated by `compute_layout` for every container;
    /// used by scroll math (`max_offset = content_size - rect_size`).
    pub content_size: [f32; 2],
    /// Present iff `layout.scrolls()`. See [`ScrollState`].
    pub scroll: Option<ScrollState>,
    pub visible: bool,
    pub children: Vec<NodeId>,
    pub interact: NodeInteract,
    pub text: Option<NodeText>,
    pub image: Option<ImageHandle>,
    pub window_action: Option<WindowAction>,
}

impl Node {
    pub fn rect() -> NodeBuilder {
        NodeBuilder {
            node: Node {
                style: ShapeStyle::default(),
                layout: LayoutStyle::default(),
                rect: [0.0; 4],
                content_size: [0.0; 2],
                scroll: None,
                visible: true,
                children: Vec::new(),
                interact: NodeInteract::default(),
                text: None,
                image: None,
                window_action: None,
            },
        }
    }

    /// Frosted glass rect. Samples the blurred backdrop behind it.
    pub fn glass() -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Glass;
        b.node.style.color = [1.0, 1.0, 1.0, 0.08];
        b
    }

    /// Text node. Content defaults to `Len::Auto` sizing; layout pass
    /// measures shaped width + `line_height` via the app's measurer.
    pub fn text(content: impl Into<String>, font_size: f32) -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Text;
        b.node.text = Some(NodeText {
            content: content.into(),
            font_size,
            line_height: font_size * 1.25,
        });
        b
    }

    /// Image node sourced from a previously-uploaded atlas handle. Tint
    /// via [`NodeBuilder::color`] / `.rgba()` (default `[1,1,1,1]` =
    /// unmodified). Sized like any other node — `.size_px(w,h)` for
    /// fixed pixels, layout drives Fill/Auto.
    pub fn image(handle: ImageHandle) -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Image;
        b.node.style.color = [1.0, 1.0, 1.0, 1.0];
        b.node.image = Some(handle);
        b
    }
}

pub struct NodeBuilder {
    node: Node,
}

impl NodeBuilder {
    // --- layout ---
    pub fn layout_axis(mut self, a: Axis) -> Self {
        self.node.layout.axis = a;
        self
    }
    pub fn layout_width(mut self, w: Len) -> Self {
        self.node.layout.width = w;
        self
    }
    pub fn layout_height(mut self, h: Len) -> Self {
        self.node.layout.height = h;
        self
    }
    pub fn layout_size(mut self, w: Len, h: Len) -> Self {
        self.node.layout.width = w;
        self.node.layout.height = h;
        self
    }
    pub fn layout_padding(mut self, p: [f32; 4]) -> Self {
        self.node.layout.padding = p;
        self
    }
    pub fn layout_gap(mut self, g: f32) -> Self {
        self.node.layout.gap = g;
        self
    }
    pub fn layout_justify(mut self, j: Justify) -> Self {
        self.node.layout.justify = j;
        self
    }
    pub fn layout_align(mut self, a: Align) -> Self {
        self.node.layout.align = a;
        self
    }
    pub fn layout_abs(mut self, x: f32, y: f32) -> Self {
        self.node.layout.abs = Some([x, y]);
        self
    }

    pub fn overflow(mut self, ox: crate::layout::Overflow, oy: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_x = ox;
        self.node.layout.overflow_y = oy;
        self
    }

    pub fn overflow_x(mut self, o: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_x = o;
        self
    }

    pub fn overflow_y(mut self, o: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_y = o;
        self
    }

    pub fn scroll(self) -> Self {
        self.overflow(crate::layout::Overflow::Scroll, crate::layout::Overflow::Scroll)
    }

    pub fn scroll_x(self) -> Self {
        self.overflow_x(crate::layout::Overflow::Scroll)
    }

    pub fn scroll_y(self) -> Self {
        self.overflow_y(crate::layout::Overflow::Scroll)
    }

    pub fn clip(self) -> Self {
        self.overflow(crate::layout::Overflow::Hidden, crate::layout::Overflow::Hidden)
    }

    /// Spring stiffness for scroll smoothing. Stored on the node's
    /// pre-allocated `ScrollState`; only takes effect once the node is
    /// also marked scrollable on at least one axis (otherwise insert
    /// drops `scroll` to `None`).
    pub fn scroll_smoothness(mut self, k: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.stiffness = k.max(0.0);
        self
    }

    pub fn overscroll(mut self, on: bool) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.overscroll = on;
        self
    }

    /// Configure the bounce-back spring used when releasing overscroll.
    /// `stiffness` (`k`) controls oscillation frequency
    /// (ω₀ = √k rad/s); `damping` (`c`) controls overshoot —
    /// `c < 2√k` is underdamped (visible bounce), `c = 2√k` is
    /// critically damped (no overshoot, smoothest landing), and
    /// `c > 2√k` is overdamped (slower, no overshoot). Defaults
    /// `(800, 42)` give ζ ≈ 0.74 — small overshoot, graceful settle
    /// in ~280 ms.
    pub fn bounce_spring(mut self, stiffness: f32, damping: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.bounce_stiffness = stiffness.max(0.0);
        s.bounce_damping = damping.max(0.0);
        self
    }

    /// Per-axis scroll snap step in **logical** px. Spring quiesce
    /// retargets to the nearest multiple. `0` on an axis disables snap
    /// there. Allocates `ScrollState` so the value sticks even if the
    /// node isn't yet scrollable; insert reconciles.
    pub fn snap_step(mut self, x: f32, y: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.snap_step = [x.max(0.0), y.max(0.0)];
        self
    }

    /// Y-axis snap step, leaves X unchanged. See [`Self::snap_step`].
    pub fn snap_step_y(mut self, px: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.snap_step[1] = px.max(0.0);
        self
    }

    /// X-axis snap step, leaves Y unchanged. See [`Self::snap_step`].
    pub fn snap_step_x(mut self, px: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.snap_step[0] = px.max(0.0);
        self
    }

    /// Replace the entire scrollbar style on this node. Allocates a
    /// `ScrollState` so the style sticks even if the node isn't yet
    /// scrollable; insert reconciles the `scrollable_ids` index.
    pub fn scrollbar_style(mut self, style: ScrollbarStyle) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.style = style;
        self
    }

    /// Mutate the scrollbar style with a closure: e.g.
    /// `.scrollbar(|s| s.thickness(8.0).thumb_color([1,1,1,0.7]))`.
    pub fn scrollbar<F: FnOnce(ScrollbarStyle) -> ScrollbarStyle>(mut self, f: F) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.style = f(s.style);
        self
    }

    // --- style ---
    pub fn color(mut self, rgba: [f32; 4]) -> Self {
        self.node.style.color = rgba;
        self
    }
    pub fn rgb(self, r: f32, g: f32, b: f32) -> Self {
        self.color([r, g, b, 1.0])
    }
    pub fn rgba(self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color([r, g, b, a])
    }
    pub fn radius(mut self, r: f32) -> Self {
        self.node.style.border_radius = [r; 4];
        self
    }
    pub fn radii(mut self, tl: f32, tr: f32, bl: f32, br: f32) -> Self {
        self.node.style.border_radius = [tl, tr, bl, br];
        self
    }
    pub fn border(mut self, width: f32, color: [f32; 4]) -> Self {
        self.node.style.border_width = width;
        self.node.style.border_color = color;
        self
    }
    pub fn shadow(mut self, offset: [f32; 2], blur: f32, color: [f32; 4], opacity: f32) -> Self {
        self.node.style.shadow_offset = offset;
        self.node.style.shadow_blur = blur;
        self.node.style.shadow_color = color;
        self.node.style.shadow_opacity = opacity;
        self
    }
    pub fn opacity(mut self, o: f32) -> Self {
        self.node.style.opacity = o;
        self
    }
    pub fn hidden(mut self) -> Self {
        self.node.visible = false;
        self
    }
    pub fn kind(mut self, kind: ShapeKind) -> Self {
        self.node.style.kind = kind;
        self
    }
    /// Per-glass backdrop blur radius (logical px). Typical UI values
    /// 8..32. 0 = no blur (sharp see-through).
    pub fn blur(mut self, px: f32) -> Self {
        self.node.style.blur_amount = px;
        self
    }
    /// Per-glass edge refraction strength (logical px). The backdrop
    /// sample UV is pushed outward by the SDF normal, falling off from
    /// rim to centre. Typical values 4..20. 0 disables.
    pub fn refraction(mut self, px: f32) -> Self {
        self.node.style.refraction = px;
        self
    }
    /// Per-glass frosted-texture variation (logical px). Per-fragment
    /// hash scatters the backdrop sample by this many pixels at the
    /// chosen mip. 0 = mirror-smooth; ~1 = subtle frost; ~3 = pebbled.
    pub fn roughness(mut self, px: f32) -> Self {
        self.node.style.roughness = px;
        self
    }
    pub fn line_height(mut self, h: f32) -> Self {
        if let Some(t) = self.node.text.as_mut() {
            t.line_height = h;
        }
        self
    }
    pub fn on_hover(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.hover = Some(signal);
        self
    }
    pub fn on_press(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.pressed = Some(signal);
        self
    }
    pub fn on_focus(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.focused = Some(signal);
        self
    }
    pub fn window_action(mut self, action: WindowAction) -> Self {
        self.node.window_action = Some(action);
        self
    }
    pub fn build(self) -> Node {
        self.node
    }
}

struct Slot {
    generation: u32,
    payload: Option<Node>,
}

pub struct NodeTree {
    slots: Vec<Slot>,
    free: Vec<u32>,
    roots: Vec<NodeId>,
    dirty: u32,
    /// Count of currently-inserted Glass-kind nodes. Used by mutators
    /// to skip the BACKDROP dirty flag when the tree has no glass —
    /// nothing samples the blurred backdrop in that case, so re-running
    /// the blur pass would be wasted work.
    glass_count: u32,
    /// Every node that currently owns a `ScrollState` (overflow set to
    /// Scroll on at least one axis). Used by `tick_scrolls` so the
    /// frame loop doesn't have to re-walk the tree every tick.
    /// Maintained by `set_layout_overflow` / `remove`.
    scrollable_ids: Vec<NodeId>,
    /// Current display scale factor. Mirror of `App.scale_factor` —
    /// kept on the tree so scroll setters can convert logical-px
    /// configuration (`snap_step`, overscroll limit) to physical at
    /// the point of use without threading the scale through every
    /// public method. Updated by the app at init + on
    /// `ScaleFactorChanged`. Defaults to `1.0` for headless tests.
    current_scale: f32,
}

impl Default for NodeTree {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            roots: Vec::new(),
            dirty: 0,
            glass_count: 0,
            scrollable_ids: Vec::new(),
            current_scale: 1.0,
        }
    }
}

impl NodeTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset `time_since_input` on every scrollable to 0, so the
    /// on-quiesce settle path stays gated. Called from the app shell
    /// each tick while a scroll key is physically held — the OS
    /// auto-repeat initial delay (Windows default ~250 ms) is longer
    /// than `SCROLL_INPUT_QUIESCE_SECONDS` (100 ms), so without this
    /// poke the gate would lapse during the no-event window between
    /// the initial keypress and the first auto-repeat. Settle would
    /// fire, start a bounce, and the next repeat would re-stretch the
    /// rubber-band — visible as a stretch / bounce / stretch flicker.
    pub fn poke_scroll_input_recency(&mut self) {
        for &id in self.scrollable_ids.clone().iter() {
            if let Some(slot) = self.slots.get_mut(id.index as usize)
                && slot.generation == id.generation
                && let Some(n) = slot.payload.as_mut()
                && let Some(s) = n.scroll.as_mut()
            {
                s.time_since_input = 0.0;
            }
        }
    }

    /// Update the cached display scale used by scroll math. Call from
    /// the app shell on init and after `WindowEvent::ScaleFactorChanged`.
    /// No-op below `f32::EPSILON`.
    pub fn set_scale(&mut self, scale: f32) {
        if scale > f32::EPSILON {
            self.current_scale = scale;
        }
    }

    /// Read the cached display scale. Mainly for tests.
    pub fn scale(&self) -> f32 {
        self.current_scale
    }

    fn insert(&mut self, mut node: Node) -> NodeId {
        let is_glass = matches!(node.style.kind, ShapeKind::Glass);
        // Reconcile scroll state with layout overflow declared on the
        // builder side: if either axis is Scroll, ensure ScrollState
        // exists so `scrollable_ids` and the wheel/tick paths see it.
        let needs_scroll = node.layout.scrolls();
        if needs_scroll && node.scroll.is_none() {
            node.scroll = Some(ScrollState::default());
        } else if !needs_scroll && node.scroll.is_some() {
            node.scroll = None;
        }
        let has_scroll = node.scroll.is_some();
        let id = if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.payload = Some(node);
            NodeId {
                index: idx,
                generation: slot.generation,
            }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                payload: Some(node),
            });
            NodeId {
                index: idx,
                generation: 0,
            }
        };
        if is_glass {
            self.glass_count += 1;
        }
        if has_scroll {
            self.scrollable_ids.push(id);
        }
        id
    }

    pub fn add_root(&mut self, node: Node) -> NodeId {
        let id = self.insert(node);
        self.roots.push(id);
        self.dirty |= dirty::TREE;
        id
    }

    pub fn add_child(&mut self, parent: NodeId, node: Node) -> NodeId {
        let id = self.insert(node);
        if let Some(p) = self.get_mut_raw(parent) {
            p.children.push(id);
        }
        self.dirty |= dirty::TREE;
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        let Some(slot) = self.slots.get_mut(id.index as usize) else {
            return;
        };
        if slot.generation != id.generation {
            return;
        }
        let payload = slot.payload.as_ref();
        let was_glass = payload
            .map(|n| matches!(n.style.kind, ShapeKind::Glass))
            .unwrap_or(false);
        let was_scrollable = payload.map(|n| n.scroll.is_some()).unwrap_or(false);
        slot.generation = slot.generation.wrapping_add(1);
        slot.payload = None;
        self.free.push(id.index);
        self.roots.retain(|r| *r != id);
        if was_scrollable {
            self.scrollable_ids.retain(|sid| *sid != id);
        }
        self.dirty |= dirty::TREE;
        if was_glass {
            self.glass_count = self.glass_count.saturating_sub(1);
            // Removed glass → backdrop no longer needed for it but
            // any remaining glass still samples the same texture; safe
            // to skip BACKDROP. TREE flag drives a full re-flatten which
            // already triggers a re-blur via set_instances if needed.
        }
    }

    // --- layout-mutating setters (flag TRANSFORM + BACKDROP conservatively) ---

    pub fn set_layout_width(&mut self, id: NodeId, w: Len) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.width != w {
                n.layout.width = w;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_height(&mut self, id: NodeId, h: Len) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.height != h {
                n.layout.height = h;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_abs(&mut self, id: NodeId, pos: Option<[f32; 2]>) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.abs != pos {
                n.layout.abs = pos;
                self.dirty |= mask;
            }
    }

    /// Convenience for animated position binds: forces `layout.abs =
    /// Some([x,y])`. Skips the dirty flag bump if the value didn't move.
    pub fn set_layout_pos_abs(&mut self, id: NodeId, pos: [f32; 2]) {
        self.set_layout_abs(id, Some(pos));
    }

    /// Convenience for animated size binds: forces both axes to `Px`.
    pub fn set_layout_size_px(&mut self, id: NodeId, size: [f32; 2]) {
        let w = Len::Px(size[0]);
        let h = Len::Px(size[1]);
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id) {
            let changed = n.layout.width != w || n.layout.height != h;
            if changed {
                n.layout.width = w;
                n.layout.height = h;
                self.dirty |= mask;
            }
        }
    }

    pub fn set_layout_padding(&mut self, id: NodeId, padding: [f32; 4]) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.padding != padding {
                n.layout.padding = padding;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_gap(&mut self, id: NodeId, gap: f32) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.gap != gap {
                n.layout.gap = gap;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_justify(&mut self, id: NodeId, j: Justify) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.justify != j {
                n.layout.justify = j;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_align(&mut self, id: NodeId, a: Align) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.align != a {
                n.layout.align = a;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_axis(&mut self, id: NodeId, ax: Axis) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.axis != ax {
                n.layout.axis = ax;
                self.dirty |= mask;
            }
    }

    /// Set per-axis overflow. Allocates `ScrollState` on the node when
    /// either axis becomes Scroll; clears it when both axes drop back
    /// to Visible/Hidden. Maintains `scrollable_ids` so the frame
    /// loop's scroll tick has an O(1) iteration list.
    pub fn set_layout_overflow(&mut self, id: NodeId, ox: crate::layout::Overflow,
                               oy: crate::layout::Overflow) {
        use crate::layout::Overflow;
        let mask = self.transform_mask();
        let mut allocated = false;
        let mut cleared = false;
        if let Some(n) = self.get_mut_raw(id) {
            let changed = n.layout.overflow_x != ox || n.layout.overflow_y != oy;
            if !changed {
                return;
            }
            n.layout.overflow_x = ox;
            n.layout.overflow_y = oy;
            let needs_scroll = matches!(ox, Overflow::Scroll) || matches!(oy, Overflow::Scroll);
            match (needs_scroll, &n.scroll) {
                (true, None) => {
                    n.scroll = Some(ScrollState::default());
                    allocated = true;
                }
                (false, Some(_)) => {
                    n.scroll = None;
                    cleared = true;
                }
                _ => {}
            }
            self.dirty |= mask;
        }
        if allocated {
            self.scrollable_ids.push(id);
        }
        if cleared {
            self.scrollable_ids.retain(|sid| *sid != id);
        }
    }

    /// Advance every active scroll spring by `dt` seconds. Spring is a
    /// single-pole exponential ease toward `target`: `current += (target
    /// - current) * (1 - exp(-stiffness * dt))`. Snaps when within
    /// 0.5 px so the loop can park on `Wait`. Returns true when at
    /// least one node moved — caller flags the dirty mask + flushes.
    /// Sets `TRANSFORM` (and `BACKDROP` if glass exists) so flatten
    /// + the blur pass re-run with the new offsets.
    pub fn tick_scrolls(&mut self, dt: f32) -> bool {
        if self.scrollable_ids.is_empty() || dt <= 0.0 {
            return false;
        }
        let mut moved = false;
        let mut bar_changed = false;
        let scale = self.current_scale;
        for i in 0..self.scrollable_ids.len() {
            let id = self.scrollable_ids[i];
            let Some(slot) = self.slots.get_mut(id.index as usize) else {
                continue;
            };
            if slot.generation != id.generation {
                continue;
            }
            let Some(n) = slot.payload.as_mut() else {
                continue;
            };
            let rect = n.rect;
            let content = n.content_size;
            let Some(s) = n.scroll.as_mut() else {
                continue;
            };
            let max_off = [
                (content[0] - rect[2]).max(0.0),
                (content[1] - rect[3]).max(0.0),
            ];
            s.time_since_input += dt;
            // Per-axis chase. Two paths:
            //   (a) Bounce-back via underdamped spring (closed-form,
            //       overshoots target slightly then settles) — engages
            //       when `current` is past edge AND `target` is in
            //       range. Models a real rubber-band release.
            //   (b) Forward chase via exponential ease — engages
            //       otherwise. Monotonic, no overshoot, snappy. Right
            //       feel for ordinary scrolling.
            for axis in 0..2 {
                let cur = s.current[axis];
                let tgt = s.target[axis];
                let max = max_off[axis];
                let already_bouncing = s.bounce_elapsed[axis] >= 0.0;
                let target_in_range = tgt >= 0.0 && tgt <= max;
                let cur_oor = cur < 0.0 || cur > max;
                // Trigger: not yet bouncing, current is past edge,
                // target sits inside range (settle has clamped it).
                // Continue: already bouncing AND target is still in
                // range (user hasn't wheeled past edge again, which
                // would re-engage rubber-band and cancel the bounce).
                // The continue branch runs even when `current`
                // momentarily crosses back into range during overshoot
                // — the spring's natural oscillation is what produces
                // the "alive" feel; gating on cur_oor would freeze it
                // mid-cycle.
                let bouncing =
                    target_in_range && (already_bouncing || cur_oor);
                if bouncing {
                    let target_shifted = already_bouncing
                        && (s.bounce_target[axis] - tgt).abs() > 0.5;
                    if !already_bouncing || target_shifted {
                        s.bounce_from[axis] = cur;
                        s.bounce_target[axis] = tgt;
                        s.bounce_elapsed[axis] = 0.0;
                    }
                    s.bounce_elapsed[axis] += dt;
                    let (x, v) = crate::anim::spring_eval(
                        s.bounce_stiffness,
                        s.bounce_damping,
                        s.bounce_elapsed[axis],
                    );
                    let new_pos = s.bounce_from[axis] * (1.0 - x) + tgt * x;
                    let settled = (x - 1.0).abs() < 1e-3 && v.abs() < 1e-3;
                    let new = if settled { tgt } else { new_pos };
                    if (new - cur).abs() > f32::EPSILON {
                        s.current[axis] = new;
                        moved = true;
                    }
                    if settled {
                        s.bounce_elapsed[axis] = -1.0;
                    }
                    s.bar_alpha = 1.0;
                } else {
                    // Not bouncing — clear bounce state and run normal
                    // exponential chase if target hasn't been reached.
                    s.bounce_elapsed[axis] = -1.0;
                    if cur != tgt {
                        let alpha = 1.0 - (-s.stiffness * dt).exp();
                        let mut new = cur + (tgt - cur) * alpha;
                        if (tgt - new).abs() < 0.5 {
                            new = tgt;
                        }
                        if new != cur {
                            s.current[axis] = new;
                            moved = true;
                        }
                        s.bar_alpha = 1.0;
                    }
                }
            }
            // Spring just quiesced (or was already at rest): apply
            // overscroll release + snap-to-step. Suppressed while:
            //   - a thumb drag is in flight (`s.dragging()` — drag
            //     target tracks the cursor),
            //   - input is still arriving (`time_since_input` below
            //     gate — held arrow / wheel burst would otherwise
            //     thrash target between settle-clamp and the next
            //     repeat's rubber-band push).
            // Otherwise idempotent: re-firing on idle ticks is a no-op.
            if !s.dragging()
                && s.time_since_input >= SCROLL_INPUT_QUIESCE_SECONDS
                && s.current == s.target
                && settle_target(s, rect, content, scale)
            {
                moved = true;
                if !s.style.auto_hide {
                    s.bar_alpha = 1.0;
                }
            }
            // Hold visible whenever the user is interacting with the
            // bar or the style demands always-on. Otherwise drain.
            let hold = s.style.always_visible
                || s.bar_hover[0]
                || s.bar_hover[1]
                || s.bar_active[0]
                || s.bar_active[1]
                || s.current != s.target;
            if hold {
                if s.bar_alpha < 1.0 {
                    s.bar_alpha = 1.0;
                    bar_changed = true;
                }
            } else if s.bar_alpha > 0.0 {
                let step = if s.style.fade_seconds > 0.0 {
                    dt / s.style.fade_seconds
                } else {
                    1.0
                };
                let new_alpha = (s.bar_alpha - step).max(0.0);
                if new_alpha != s.bar_alpha {
                    s.bar_alpha = new_alpha;
                    bar_changed = true;
                }
            }
        }
        if moved || bar_changed {
            self.dirty |= self.scroll_mask();
        }
        moved || bar_changed
    }

    /// True when at least one scrollable node still needs another tick:
    /// either the spring is chasing (`current != target`) or the bar is
    /// mid-fade (`bar_alpha > 0` while idle). Drives the loop's
    /// `WaitUntil` scheduling so the bar fades cleanly to 0 before the
    /// loop parks on `Wait`.
    pub fn has_active_scrolls(&self) -> bool {
        self.scrollable_ids.iter().any(|&id| {
            self.get(id)
                .and_then(|n| n.scroll.as_ref())
                .map(|s| {
                    s.current != s.target
                        || s.bounce_elapsed[0] >= 0.0
                        || s.bounce_elapsed[1] >= 0.0
                        || s.time_since_input < SCROLL_INPUT_QUIESCE_SECONDS
                        || s.bar_alpha > 0.0
                        || s.style.always_visible
                        || s.bar_hover[0]
                        || s.bar_hover[1]
                        || s.bar_active[0]
                        || s.bar_active[1]
                })
                .unwrap_or(false)
        })
    }

    /// Set the scroll target (where the spring is easing toward) on a
    /// scrollable node. Clamped to `[0, content_size - rect_size]`
    /// unless `ScrollState.overscroll == true`. Per-axis overflow
    /// gates the write — non-scroll axes ignore the input. Bumps
    /// TRANSFORM when the target moves so the next flush ticks the
    /// spring.
    pub fn set_scroll_target(&mut self, id: NodeId, target: [f32; 2]) {
        let (rect, content, sx, sy) = match self.get(id) {
            Some(n) => (
                n.rect,
                n.content_size,
                n.layout.overflow_x.scrolls(),
                n.layout.overflow_y.scrolls(),
            ),
            None => return,
        };
        let scale = self.current_scale;
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let max_off_x = (content[0] - rect[2]).max(0.0);
            let max_off_y = (content[1] - rect[3]).max(0.0);
            let want_x = if sx { target[0] } else { s.target[0] };
            let want_y = if sy { target[1] } else { s.target[1] };
            // Absolute set always hard-clamps (rubber-band is a
            // gestural / wheel-incremental concept) and applies snap so
            // programmatic targets land on a row boundary just like
            // wheel input does.
            let clamped_x = want_x.clamp(0.0, max_off_x);
            let clamped_y = want_y.clamp(0.0, max_off_y);
            let new_target = [
                snap_to_step(clamped_x, s.snap_step[0], scale, max_off_x),
                snap_to_step(clamped_y, s.snap_step[1], scale, max_off_y),
            ];
            // Reset input quiescence so the on-quiesce settle path
            // waits the full gate before retargeting. Prevents jerk
            // when continuous input (held arrow, wheel burst) has
            // pushed target past edge.
            s.time_since_input = 0.0;
            if s.target != new_target {
                s.target = new_target;
                if !s.style.auto_hide {
                    s.bar_alpha = 1.0;
                }
                self.dirty |= mask;
            }
        }
    }

    /// Add to the scroll target. Convenience for wheel input — caller
    /// passes raw delta and clamping happens here. Per-axis overflow
    /// gates the write: a Scroll-x-only container ignores y delta even
    /// if its `content_size.y > rect.h`. Returns the actual delta
    /// applied (may be less than requested at edges or zero on a non-
    /// scroll axis) so a wheel dispatcher can bubble the remainder to
    /// a parent scroll ancestor.
    /// Variant of [`Self::add_scroll_delta`] that **skips snap-on-input**.
    /// Used by the per-tick hold-to-scroll pump where the per-frame
    /// delta is small relative to `snap_step` — applying snap each
    /// tick would round the delta to zero and stall the scroll. Settle
    /// on quiesce (the post-input snap path) handles row alignment
    /// when the user releases the key.
    pub fn add_scroll_delta_continuous(&mut self, id: NodeId, delta: [f32; 2]) -> [f32; 2] {
        self.add_scroll_delta_inner(id, delta, false)
    }

    pub fn add_scroll_delta(&mut self, id: NodeId, delta: [f32; 2]) -> [f32; 2] {
        self.add_scroll_delta_inner(id, delta, true)
    }

    fn add_scroll_delta_inner(&mut self, id: NodeId, delta: [f32; 2], snap: bool) -> [f32; 2] {
        let (rect, content, sx, sy) = match self.get(id) {
            Some(n) => (
                n.rect,
                n.content_size,
                n.layout.overflow_x.scrolls(),
                n.layout.overflow_y.scrolls(),
            ),
            None => return [0.0; 2],
        };
        let scale = self.current_scale;
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let max_off_x = (content[0] - rect[2]).max(0.0);
            let max_off_y = (content[1] - rect[3]).max(0.0);
            let raw_dx = if sx { delta[0] } else { 0.0 };
            let raw_dy = if sy { delta[1] } else { 0.0 };
            let limit = OVERSCROLL_LIMIT_LOGICAL * scale;
            // Overscroll engages in two stages so wheel input doesn't
            // feel "forced" past the edge:
            //   1. First crossing: target lands exactly at the edge
            //      (clamp). The over-portion is dropped — no rubber-band
            //      yet. Mirrors iOS scroll: momentum stops at the edge.
            //   2. Subsequent push while target is *already* at the edge:
            //      `rubber_band_target` asymptote applies, so each
            //      additional wheel event stretches the rubber-band
            //      progressively until limit.
            // Direction symmetric for the top edge. Non-overscroll: hard
            // clamp on every input.
            let mut new_target = if s.overscroll {
                [
                    add_with_edge_gate(s.target[0], raw_dx, 0.0, max_off_x, limit),
                    add_with_edge_gate(s.target[1], raw_dy, 0.0, max_off_y, limit),
                ]
            } else {
                [
                    (s.target[0] + raw_dx).clamp(0.0, max_off_x),
                    (s.target[1] + raw_dy).clamp(0.0, max_off_y),
                ]
            };
            // Snap-on-input: when the post-clamp target lands inside the
            // valid range, immediately retarget to the nearest snap
            // multiple. Spring then eases to a row-aligned position from
            // the very first tick — no visible "settle then jump" pause.
            // Past-edge (rubber-band) targets keep their cooked value;
            // settle-on-quiesce snaps once the spring lands in range.
            // Skipped in continuous mode — see `add_scroll_delta_continuous`.
            if snap {
                if new_target[0] >= 0.0 && new_target[0] <= max_off_x {
                    new_target[0] = snap_to_step(new_target[0], s.snap_step[0], scale, max_off_x);
                }
                if new_target[1] >= 0.0 && new_target[1] <= max_off_y {
                    new_target[1] = snap_to_step(new_target[1], s.snap_step[1], scale, max_off_y);
                }
            }
            // Overscroll consumes the requested delta on its enabled
            // axes even when rubber-band shrinks the target movement —
            // bubbling the remainder up to an outer scroller while
            // rubber-banding the inner would feel wrong (the user
            // clearly wants to scroll the inner). Non-overscroll axes
            // still report the actual clamped movement so wheel routing
            // can bubble the leftover.
            let applied_x = if s.overscroll && sx {
                raw_dx
            } else {
                new_target[0] - s.target[0]
            };
            let applied_y = if s.overscroll && sy {
                raw_dy
            } else {
                new_target[1] - s.target[1]
            };
            let applied = [applied_x, applied_y];
            // Same input-quiescence reset as set_scroll_target — must
            // happen on every input even when target doesn't change
            // (e.g. user holds arrow against the rubber-band cap, raw
            // delta saturates so target stays the same).
            s.time_since_input = 0.0;
            if new_target != s.target {
                s.target = new_target;
                if !s.style.auto_hide {
                    s.bar_alpha = 1.0;
                }
                self.dirty |= mask;
            }
            return applied;
        }
        [0.0; 2]
    }

    /// Read the displayed scroll offset (current, not target). Returns
    /// `[0, 0]` for non-scrollable nodes.
    pub fn scroll_offset(&self, id: NodeId) -> [f32; 2] {
        self.get(id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.current)
            .unwrap_or([0.0; 2])
    }

    /// Read content size (bounding extent of children, includes
    /// trailing padding). Returns the node's own `rect` size for
    /// non-container leaves.
    pub fn scrollable_size(&self, id: NodeId) -> [f32; 2] {
        self.get(id).map(|n| n.content_size).unwrap_or([0.0; 2])
    }

    /// Set the spring stiffness (ease rate). No-op on non-scrollable.
    pub fn set_scroll_stiffness(&mut self, id: NodeId, k: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.stiffness = k.max(0.0);
        }
    }

    pub fn set_scroll_overscroll(&mut self, id: NodeId, on: bool) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.overscroll = on;
        }
    }

    /// Update the bounce-back spring for a scrollable node. See
    /// [`NodeBuilder::bounce_spring`] for the param semantics.
    pub fn set_scroll_bounce_spring(&mut self, id: NodeId, stiffness: f32, damping: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.bounce_stiffness = stiffness.max(0.0);
            s.bounce_damping = damping.max(0.0);
        }
    }

    /// Per-axis scroll snap step in **logical** px. `0` on an axis
    /// disables snap there (continuous scroll). When non-zero, the
    /// spring quiesce path retargets to the nearest multiple, so
    /// every settle lands on a row boundary regardless of how the
    /// user got there (wheel, drag-end, scrollbar click, arrow-key).
    pub fn set_scroll_snap_step(&mut self, id: NodeId, step: [f32; 2]) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let step = [step[0].max(0.0), step[1].max(0.0)];
            if s.snap_step != step {
                s.snap_step = step;
                // Snap activation is applied next time the spring
                // quiesces; flag SCROLL so the loop ticks again.
                self.dirty |= dirty::SCROLL;
            }
        }
    }

    /// Replace the entire scrollbar style on `id`. Allocates a
    /// `ScrollState` if one isn't already present so style changes can
    /// be authored before `.scroll()` is called.
    pub fn set_scrollbar_style(&mut self, id: NodeId, style: ScrollbarStyle) {
        let mut allocated = false;
        if let Some(n) = self.get_mut_raw(id) {
            if n.scroll.is_none() {
                n.scroll = Some(ScrollState::default());
                allocated = true;
            }
            if let Some(s) = n.scroll.as_mut() {
                s.style = style;
            }
        }
        if allocated {
            // Only push to scrollable_ids if the node already declared
            // an overflow that scrolls — otherwise insert/remove
            // already manages it. We allocate eagerly so styles can be
            // set before .scroll(); insert reconciles on add.
            let scrolls = self.get(id).map(|n| n.layout.scrolls()).unwrap_or(false);
            if scrolls && !self.scrollable_ids.contains(&id) {
                self.scrollable_ids.push(id);
            }
        }
    }

    /// Mutate the existing scrollbar style in place. Same allocation
    /// rules as [`Self::set_scrollbar_style`].
    pub fn with_scrollbar_style<F: FnOnce(&mut ScrollbarStyle)>(&mut self, id: NodeId, f: F) {
        let mut allocated = false;
        if let Some(n) = self.get_mut_raw(id) {
            if n.scroll.is_none() {
                n.scroll = Some(ScrollState::default());
                allocated = true;
            }
            if let Some(s) = n.scroll.as_mut() {
                f(&mut s.style);
            }
        }
        if allocated {
            let scrolls = self.get(id).map(|n| n.layout.scrolls()).unwrap_or(false);
            if scrolls && !self.scrollable_ids.contains(&id) {
                self.scrollable_ids.push(id);
            }
        }
    }

    /// Set per-axis pointer-hover flags on a scrollable node. Returns
    /// true if anything changed (caller can use this to gate redraw).
    /// `[X, Y]` indexed by `ScrollAxis::index`.
    pub fn set_bar_hover(&mut self, id: NodeId, hover: [bool; 2]) -> bool {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
            && s.bar_hover != hover
        {
            s.bar_hover = hover;
            // Hovering pops the bar to full alpha immediately so the
            // user gets feedback without waiting on the next tick.
            if hover[0] || hover[1] {
                s.bar_alpha = 1.0;
            }
            // SCROLL only — bar color change re-flattens but doesn't
            // touch layout or the opaque backdrop.
            self.dirty |= dirty::SCROLL;
            return true;
        }
        false
    }

    /// Set per-axis active (mouse-down on thumb) flags. Returns true
    /// on change.
    pub fn set_bar_active(&mut self, id: NodeId, active: [bool; 2]) -> bool {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
            && s.bar_active != active
        {
            s.bar_active = active;
            if active[0] || active[1] {
                s.bar_alpha = 1.0;
            }
            self.dirty |= dirty::SCROLL;
            return true;
        }
        false
    }

    /// Snap scroll on one axis to `pos` immediately (no spring chase).
    /// Intended for thumb-drag — the pointer is the authoritative
    /// position so easing toward it would just lag behind. Writes
    /// both `current` and `target` so the spring stays at rest.
    ///
    /// When `overscroll == true`, drag past the track end is allowed
    /// but rubber-banded via the same asymptote as wheel input — the
    /// effective position saturates at `max_off + OVERSCROLL_LIMIT *
    /// scale`. This produces the visual stretch users expect when they
    /// pull the thumb past either end. The drag-end handler is
    /// responsible for retargeting to a clamped position so the spring
    /// bounces back. Without overscroll, hard-clamps to `[0, max_off]`.
    pub fn set_scroll_immediate(&mut self, id: NodeId, axis: ScrollAxis, pos: f32) {
        let (rect, content) = match self.get(id) {
            Some(n) => (n.rect, n.content_size),
            None => return,
        };
        let scale = self.current_scale;
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let i = axis.index();
            let max_off = (content[i] - rect[2 + i]).max(0.0);
            let new_pos = if s.overscroll {
                let limit = OVERSCROLL_LIMIT_LOGICAL * scale;
                rubber_band_target(0.0, pos, 0.0, max_off, limit)
            } else {
                pos.clamp(0.0, max_off)
            };
            // Drag fires every frame from `drag_to` while the user
            // moves the thumb — keep the settle gate suppressed for
            // the full drag, then end_drag's set_scroll_target reset
            // starts the post-release timer.
            s.time_since_input = 0.0;
            if (s.current[i] - new_pos).abs() > f32::EPSILON
                || (s.target[i] - new_pos).abs() > f32::EPSILON
            {
                s.current[i] = new_pos;
                s.target[i] = new_pos;
                s.bar_alpha = 1.0;
                self.dirty |= mask;
            }
        }
    }

    pub fn set_color(&mut self, id: NodeId, color: [f32; 4]) {
        let has_glass = self.has_glass();
        if let Some(n) = self.get_mut_raw(id)
            && n.style.color != color {
                // Glass + Image render in the final pass only, so they
                // don't enter the blurred backdrop. And without any
                // glass node sampling it, the blur pass is skipped
                // anyway — no need to flag BACKDROP.
                let is_opaque_change =
                    !matches!(n.style.kind, ShapeKind::Glass | ShapeKind::Image);
                n.style.color = color;
                self.dirty |= dirty::VISUAL;
                if is_opaque_change && has_glass {
                    self.dirty |= dirty::BACKDROP;
                }
            }
    }

    pub fn set_opacity(&mut self, id: NodeId, opacity: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && n.style.opacity != opacity {
                n.style.opacity = opacity;
                self.dirty |= dirty::VISUAL;
            }
    }

    pub fn set_text(&mut self, id: NodeId, content: impl Into<String>) {
        let content = content.into();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(t) = n.text.as_mut()
                && t.content != content {
                    t.content = content;
                    // Text width changes → relayout (Auto-sized text).
                    self.dirty |= dirty::VISUAL | dirty::TRANSFORM;
                }
    }

    pub fn set_font_size(&mut self, id: NodeId, font_size: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(t) = n.text.as_mut()
                && t.font_size != font_size {
                    let old_ratio = t.line_height / t.font_size.max(0.0001);
                    t.font_size = font_size;
                    t.line_height = font_size * old_ratio;
                    self.dirty |= dirty::VISUAL | dirty::TRANSFORM;
                }
    }

    pub fn set_visible(&mut self, id: NodeId, visible: bool) {
        if let Some(n) = self.get_mut_raw(id)
            && n.visible != visible {
                n.visible = visible;
                self.dirty |= dirty::TREE;
            }
    }

    pub fn dirty(&self) -> u32 {
        self.dirty
    }

    /// True when at least one Glass-kind node lives in the tree. Used
    /// to gate the BACKDROP dirty flag on layout/visual mutations —
    /// without glass, nothing samples the blurred backdrop so re-running
    /// the blur is wasted work.
    pub fn has_glass(&self) -> bool {
        self.glass_count > 0
    }

    /// Mask to OR into `self.dirty` for any layout-mutating setter.
    /// Drops BACKDROP when the tree has no glass.
    fn transform_mask(&self) -> u32 {
        if self.has_glass() {
            dirty::TRANSFORM | dirty::BACKDROP
        } else {
            dirty::TRANSFORM
        }
    }

    /// Mask for scroll-offset writes. Layout doesn't need to re-run
    /// (rects are unchanged), only flatten — so `SCROLL` instead of
    /// `TRANSFORM`. Backdrop still re-blurs when glass exists because
    /// opaque content under glass moved.
    fn scroll_mask(&self) -> u32 {
        if self.has_glass() {
            dirty::SCROLL | dirty::BACKDROP
        } else {
            dirty::SCROLL
        }
    }

    pub fn take_dirty(&mut self) -> u32 {
        let d = self.dirty;
        self.dirty = dirty::NONE;
        d
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty |= dirty::ANY;
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.payload.as_ref()
    }

    pub fn get_mut_raw(&mut self, id: NodeId) -> Option<&mut Node> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.payload.as_mut()
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.payload.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    /// Every node currently owning a `ScrollState`, in insertion order
    /// (preserved across removes). Exposed so input routing can fall
    /// back to a default scroll target when the cursor isn't over any
    /// scroll container — typical use is `tree.scrollables().first()`.
    pub fn scrollables(&self) -> &[NodeId] {
        &self.scrollable_ids
    }

    /// DFS preorder flatten into a single ordered event stream,
    /// reading post-layout `Node.rect`. Parent opacity multiplies
    /// down. Painter's order across all kinds — caller resolves
    /// Text/Image events into GPU instances at their event index so
    /// layering is preserved. Hit cache is topmost-first.
    ///
    /// Clip + scroll offset propagate down the tree. Each node receives
    /// the intersection of its ancestors' clipping rects and the sum of
    /// its ancestors' scroll offsets — the recursive walk maintains the
    /// stack implicitly so emitted instances/hits are already in screen
    /// space.
    pub fn flatten(
        &self,
        scale: f32,
    ) -> (Vec<FlatEvent>, Vec<HitEntry>, Vec<ScrollHit>, Vec<ScrollbarHit>) {
        let mut events = Vec::with_capacity(self.len());
        let mut hits = Vec::new();
        let mut scroll_hits = Vec::new();
        let mut scroll_bars = Vec::new();
        self.flatten_into_buffers(
            scale,
            &mut events,
            &mut hits,
            &mut scroll_hits,
            &mut scroll_bars,
        );
        (events, hits, scroll_hits, scroll_bars)
    }

    /// Same as [`Self::flatten`] but reuses caller-owned buffers
    /// instead of allocating fresh `Vec`s. Each buffer is `clear()`ed
    /// before population so callers can amortize allocation across
    /// frames (a steady-state scene reuses the same heap blocks every
    /// flatten — saves ~5–20µs of allocator churn per frame). Hits are
    /// reversed at the end so the cache reads topmost-first as usual.
    pub fn flatten_into_buffers(
        &self,
        scale: f32,
        events: &mut Vec<FlatEvent>,
        hits: &mut Vec<HitEntry>,
        scroll_hits: &mut Vec<ScrollHit>,
        scroll_bars: &mut Vec<ScrollbarHit>,
    ) {
        events.clear();
        hits.clear();
        scroll_hits.clear();
        scroll_bars.clear();
        let mut scroll_stack: Vec<NodeId> = Vec::new();
        for root in &self.roots {
            self.flatten_into(
                *root,
                1.0,
                NO_CLIP,
                [0.0; 2],
                &mut scroll_stack,
                events,
                hits,
                scroll_hits,
                scroll_bars,
                scale,
            );
        }
        hits.reverse();
    }

    #[cfg(test)]
    fn dirty_for_test(&self) -> u32 {
        self.dirty
    }

    #[allow(clippy::too_many_arguments)]
    fn flatten_into(
        &self,
        id: NodeId,
        parent_opacity: f32,
        clip: [f32; 4],
        offset: [f32; 2],
        scroll_stack: &mut Vec<NodeId>,
        events: &mut Vec<FlatEvent>,
        hits: &mut Vec<HitEntry>,
        scroll_hits: &mut Vec<ScrollHit>,
        scroll_bars: &mut Vec<ScrollbarHit>,
        scale: f32,
    ) {
        let Some(node) = self.get(id) else { return };
        if !node.visible {
            return;
        }
        let rect = node.rect;
        let abs = [rect[0] - offset[0], rect[1] - offset[1]];
        let size = [rect[2], rect[3]];
        let opacity = parent_opacity * node.style.opacity;
        match node.style.kind {
            ShapeKind::Rect | ShapeKind::Glass => {
                // For glass, repurpose backdrop_uv_rect.xy to carry bevel
                // params (the field is ignored by the glass branch's UV
                // sampling since glass uses screen-space UVs).
                let extras = if matches!(node.style.kind, ShapeKind::Glass) {
                    [
                        node.style.blur_amount,
                        node.style.refraction,
                        0.0,
                        0.0,
                    ]
                } else {
                    [0.0; 4]
                };
                events.push(FlatEvent::Shape(ShapeInstance {
                    color: node.style.color,
                    border_color: node.style.border_color,
                    shadow_color: node.style.shadow_color,
                    border_radius: node.style.border_radius,
                    backdrop_uv_rect: extras,
                    clip_rect: clip,
                    position: abs,
                    size,
                    shadow_offset: node.style.shadow_offset,
                    shape_kind: node.style.kind.as_u32(),
                    roughness: node.style.roughness,
                    border_width: node.style.border_width,
                    shadow_blur: node.style.shadow_blur,
                    shadow_opacity: node.style.shadow_opacity,
                    opacity,
                }));
            }
            ShapeKind::Text => {
                if let Some(t) = node.text.as_ref() {
                    events.push(FlatEvent::Text(TextRef {
                        position: abs,
                        color: node.style.color,
                        opacity,
                        content: t.content.clone(),
                        font_size: t.font_size,
                        line_height: t.line_height,
                        clip_rect: clip,
                    }));
                }
            }
            ShapeKind::Image => {
                if let Some(handle) = node.image {
                    events.push(FlatEvent::Image(ImageRef {
                        position: abs,
                        size,
                        color: node.style.color,
                        opacity,
                        border_radius: node.style.border_radius,
                        handle,
                        clip_rect: clip,
                    }));
                }
            }
        }
        if node.interact.is_any() || node.window_action.is_some() {
            hits.push(HitEntry {
                node_id: id,
                bounds: [abs[0], abs[1], abs[0] + size[0], abs[1] + size[1]],
                clip_rect: clip,
            });
        }
        // Emit a ScrollHit for any container whose layout scrolls. The
        // ancestor chain is innermost-first: this node first, then each
        // scroll ancestor outward. Wheel routing pops from the front
        // when bubbling at edges.
        let pushed_scroll = if node.scroll.is_some() && node.layout.scrolls() {
            let mut chain = Vec::with_capacity(scroll_stack.len() + 1);
            chain.push(id);
            chain.extend(scroll_stack.iter().rev().copied());
            scroll_hits.push(ScrollHit {
                node_id: id,
                bounds: [abs[0], abs[1], abs[0] + size[0], abs[1] + size[1]],
                clip_rect: clip,
                ancestor_chain: chain,
            });
            scroll_stack.push(id);
            true
        } else {
            false
        };
        // Children: intersect parent clip with this node's self-clip
        // (axis-aware — only narrow the axes that clip), then add this
        // node's scroll offset to the running offset.
        let child_clip = if node.layout.clips() {
            let self_clip = [
                if node.layout.overflow_x.clips() { abs[0] } else { -1.0e30 },
                if node.layout.overflow_y.clips() { abs[1] } else { -1.0e30 },
                if node.layout.overflow_x.clips() { abs[0] + size[0] } else { 1.0e30 },
                if node.layout.overflow_y.clips() { abs[1] + size[1] } else { 1.0e30 },
            ];
            intersect_clip(clip, self_clip)
        } else {
            clip
        };
        let child_offset = if let Some(s) = node.scroll.as_ref() {
            [offset[0] + s.current[0], offset[1] + s.current[1]]
        } else {
            offset
        };
        for &child in &node.children {
            self.flatten_into(
                child,
                opacity,
                child_clip,
                child_offset,
                scroll_stack,
                events,
                hits,
                scroll_hits,
                scroll_bars,
                scale,
            );
        }
        // Emit scrollbar geometry last so visible bars paint over
        // children. The bar lives at the container's *unscrolled*
        // position (uses `abs`, not `child_offset`) and inherits the
        // parent's clip. Hits are populated regardless of `bar_alpha`
        // so input can detect hover-enter on a faded-out bar's region.
        if let Some(s) = node.scroll.as_ref() {
            emit_scrollbars(
                id,
                node,
                s,
                abs,
                size,
                opacity,
                clip,
                scale,
                events,
                scroll_bars,
            );
        }
        if pushed_scroll {
            scroll_stack.pop();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_scrollbars(
    node_id: NodeId,
    node: &Node,
    s: &ScrollState,
    abs: [f32; 2],
    size: [f32; 2],
    opacity: f32,
    clip: [f32; 4],
    scale: f32,
    events: &mut Vec<FlatEvent>,
    scroll_bars: &mut Vec<ScrollbarHit>,
) {
    let style = &s.style;
    let bar_w = style.thickness * scale;
    let bar_margin = style.margin * scale;
    let min_thumb = style.min_thumb * scale;
    let bar_alpha = if style.always_visible { 1.0 } else { s.bar_alpha };
    let visual = bar_alpha * opacity;

    let mut emit_quad = |position: [f32; 2], box_size: [f32; 2], rgba: [f32; 4]| {
        if rgba[3] <= 0.001 || box_size[0] <= 0.0 || box_size[1] <= 0.0 {
            return;
        }
        events.push(FlatEvent::Shape(ShapeInstance {
            color: rgba,
            border_color: [0.0; 4],
            shadow_color: [0.0; 4],
            // Logical px — `expand_events_into` re-scales it.
            border_radius: [style.radius; 4],
            backdrop_uv_rect: [0.0; 4],
            clip_rect: clip,
            position,
            size: box_size,
            shadow_offset: [0.0; 2],
            shape_kind: SHAPE_KIND_RECT,
            roughness: 0.0,
            border_width: 0.0,
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
        }));
    };

    // Y bar.
    if node.layout.overflow_y.scrolls() {
        let max_off = (node.content_size[1] - size[1]).max(0.0);
        if max_off > 0.0 {
            let track_x = match style.y_side {
                BarSide::End => abs[0] + size[0] - bar_w - bar_margin,
                BarSide::Start => abs[0] + bar_margin,
            };
            let track_y = abs[1] + bar_margin;
            let track_h = size[1] - bar_margin * 2.0;
            if track_h > 0.0 {
                let visible_ratio = (size[1] / node.content_size[1]).clamp(0.0, 1.0);
                let thumb_h = (track_h * visible_ratio).max(min_thumb).min(track_h);
                let frac = (s.current[1] / max_off).clamp(0.0, 1.0);
                let thumb_y = track_y + frac * (track_h - thumb_h);
                let thumb_color = pick_thumb_color(style, s.bar_active[1], s.bar_hover[1]);
                let track_rgba = scale_alpha(style.track_color, visual);
                let thumb_rgba = scale_alpha(thumb_color, visual);
                emit_quad([track_x, track_y], [bar_w, track_h], track_rgba);
                emit_quad([track_x, thumb_y], [bar_w, thumb_h], thumb_rgba);
                scroll_bars.push(ScrollbarHit {
                    node_id,
                    axis: ScrollAxis::Y,
                    track: [track_x, track_y, track_x + bar_w, track_y + track_h],
                    thumb: [track_x, thumb_y, track_x + bar_w, thumb_y + thumb_h],
                    clip_rect: clip,
                    max_offset: max_off,
                    track_travel: (track_h - thumb_h).max(0.0),
                });
            }
        }
    }
    // X bar.
    if node.layout.overflow_x.scrolls() {
        let max_off = (node.content_size[0] - size[0]).max(0.0);
        if max_off > 0.0 {
            let track_x = abs[0] + bar_margin;
            let track_y = match style.x_side {
                BarSide::End => abs[1] + size[1] - bar_w - bar_margin,
                BarSide::Start => abs[1] + bar_margin,
            };
            let track_w = size[0] - bar_margin * 2.0;
            // If both axes scroll, leave space for the y-bar on its
            // chosen side so the two tracks don't visually overlap.
            let reserved = if node.layout.overflow_y.scrolls() {
                bar_w + bar_margin
            } else {
                0.0
            };
            let (track_x, track_w) = match (node.layout.overflow_y.scrolls(), style.y_side) {
                (true, BarSide::Start) => (track_x + reserved, track_w - reserved),
                (true, BarSide::End) => (track_x, track_w - reserved),
                _ => (track_x, track_w),
            };
            if track_w > 0.0 {
                let visible_ratio = (size[0] / node.content_size[0]).clamp(0.0, 1.0);
                let thumb_w = (track_w * visible_ratio).max(min_thumb).min(track_w);
                let frac = (s.current[0] / max_off).clamp(0.0, 1.0);
                let thumb_x = track_x + frac * (track_w - thumb_w);
                let thumb_color = pick_thumb_color(style, s.bar_active[0], s.bar_hover[0]);
                let track_rgba = scale_alpha(style.track_color, visual);
                let thumb_rgba = scale_alpha(thumb_color, visual);
                emit_quad([track_x, track_y], [track_w, bar_w], track_rgba);
                emit_quad([thumb_x, track_y], [thumb_w, bar_w], thumb_rgba);
                scroll_bars.push(ScrollbarHit {
                    node_id,
                    axis: ScrollAxis::X,
                    track: [track_x, track_y, track_x + track_w, track_y + bar_w],
                    thumb: [thumb_x, track_y, thumb_x + thumb_w, track_y + bar_w],
                    clip_rect: clip,
                    max_offset: max_off,
                    track_travel: (track_w - thumb_w).max(0.0),
                });
            }
        }
    }
}

fn pick_thumb_color(style: &ScrollbarStyle, active: bool, hover: bool) -> [f32; 4] {
    if active {
        style.thumb_active_color
    } else if hover {
        style.thumb_hover_color
    } else {
        style.thumb_color
    }
}

fn scale_alpha(c: [f32; 4], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], c[3] * a]
}

/// Snap `value` to the nearest multiple of `step_logical * scale`,
/// treating `max_off` as a virtual snap point so the bottom/right edge
/// at end-of-list is always fully visible. `value` is assumed to be
/// pre-clamped to `[0, max_off]`. Returns `value` unchanged when
/// `step_logical <= 0` (snap disabled) or `max_off <= 0`.
fn snap_to_step(value: f32, step_logical: f32, scale: f32, max_off: f32) -> f32 {
    if step_logical <= 0.0 || max_off <= 0.0 {
        return value;
    }
    let step = step_logical * scale;
    if step <= f32::EPSILON {
        return value;
    }
    let mult = ((value / step).round() * step).clamp(0.0, max_off);
    if (max_off - value).abs() < (mult - value).abs() {
        max_off
    } else {
        mult
    }
}

/// Apply overscroll release + snap-to-step to `s.target` once the
/// spring quiesces. Snap on input handles the in-range case; this
/// runs after a wheel burst pushed `target` past edge with rubber-
/// band so the spring lands in-range and on a multiple. Returns true
/// if the target moved.
fn settle_target(
    s: &mut ScrollState,
    rect: [f32; 4],
    content: [f32; 2],
    scale: f32,
) -> bool {
    let max_off = [
        (content[0] - rect[2]).max(0.0),
        (content[1] - rect[3]).max(0.0),
    ];
    let mut new_target = [
        s.target[0].clamp(0.0, max_off[0]),
        s.target[1].clamp(0.0, max_off[1]),
    ];
    new_target[0] = snap_to_step(new_target[0], s.snap_step[0], scale, max_off[0]);
    new_target[1] = snap_to_step(new_target[1], s.snap_step[1], scale, max_off[1]);
    if new_target != s.target {
        s.target = new_target;
        true
    } else {
        false
    }
}

/// Compose `delta` onto `target` for an overscroll-enabled axis, with
/// the iOS-style "stop-at-edge-first" gate: a wheel push that first
/// crosses the edge clamps to the edge instead of immediately rubber-
/// banding. Only when `target` already sits at the edge (or past it
/// from a prior push) does the asymptote engage. Pull-back from past
/// the edge applies the raw delta directly. This avoids the "forced"
/// feel of rubber-band on a single big wheel event.
fn add_with_edge_gate(target: f32, delta: f32, min: f32, max: f32, limit: f32) -> f32 {
    if delta == 0.0 {
        return target;
    }
    let raw = target + delta;
    if raw > max {
        if delta > 0.0 && target < max {
            // First crossing: clamp at edge, drop the over-portion.
            max
        } else {
            // Already at-or-past edge with further push, OR pulling
            // back from past edge. Use the rubber-band asymptote
            // (which returns raw delta on pull-back).
            rubber_band_target(target, delta, min, max, limit)
        }
    } else if raw < min {
        if delta < 0.0 && target > min {
            min
        } else {
            rubber_band_target(target, delta, min, max, limit)
        }
    } else {
        raw
    }
}

/// Compute the new scroll target after applying `delta`, with iOS-style
/// rubber-band resistance when the result would land past `[min, max]`.
/// Returns `target + delta` clamped/cooked to `[min - limit, max + limit]`.
///
/// Asymptote: `cooked_over = limit * raw_over / (raw_over + limit)`. As
/// the user pushes further, additional input contributes diminishing
/// target movement, capping at exactly `max + limit` (or `min - limit`).
///
/// **Inverse-aware**: when `target` is already past the edge (i.e.
/// already a *cooked* value from a prior call), the function recovers
/// the underlying raw position via the inverse asymptote
/// (`raw = limit * cooked / (limit - cooked)`) before composing the
/// delta. Without this, repeated calls with `target = prev_cooked`
/// would treat the cooked value as raw — small positive deltas could
/// make cooked *decrease*, and long runs of small deltas would saturate
/// the wrong way. Matters most for per-tick continuous pumps where
/// each delta is small relative to `limit`.
///
/// Pull-back toward `[min, max]` is applied via the same composed
/// raw, so releasing the band converges smoothly back into range.
///
/// `limit` is in physical px (caller multiplies a logical constant by
/// the display scale).
fn rubber_band_target(target: f32, delta: f32, min: f32, max: f32, limit: f32) -> f32 {
    if limit <= f32::EPSILON {
        return (target + delta).clamp(min, max);
    }
    let raw = target + delta;
    // In-range outcome: no asymptote, just composed value.
    if raw >= min && raw <= max {
        return raw;
    }
    // Past max:
    //  - delta < 0 → pulling back into range; apply 1:1 so releasing
    //    the band feels free (the band un-stretches at full speed).
    //  - delta > 0 → pushing further past max; inverse-aware asymptote
    //    (`prev_cooked → prev_raw → +delta → new_cooked`) so repeated
    //    small deltas compose monotonically.
    // Symmetric for past min.
    if raw > max {
        if delta < 0.0 {
            return raw;
        }
        let prev_raw = if target > max {
            let co = target - max;
            if co >= limit - f32::EPSILON {
                f32::MAX / 4.0
            } else {
                max + (limit * co / (limit - co))
            }
        } else {
            target
        };
        let new_raw = (prev_raw + delta).max(max);
        let ro = new_raw - max;
        max + (limit * ro / (ro + limit))
    } else {
        // raw < min
        if delta > 0.0 {
            return raw;
        }
        let prev_raw = if target < min {
            let co = min - target;
            if co >= limit - f32::EPSILON {
                f32::MIN / 4.0
            } else {
                min - (limit * co / (limit - co))
            }
        } else {
            target
        };
        let new_raw = (prev_raw + delta).min(min);
        let ro = min - new_raw;
        min - (limit * ro / (ro + limit))
    }
}

fn intersect_clip(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// A single node's contribution to the rendered frame, in declared
/// order. Text/Image still need atlas resolution before they become
/// GPU instances; the caller walks the vec in order so layering is
/// preserved across all kinds.
#[derive(Clone, Debug)]
pub enum FlatEvent {
    Shape(ShapeInstance),
    Text(TextRef),
    Image(ImageRef),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Len;

    #[test]
    fn glass_count_tracks_inserts_and_removes() {
        let mut t = NodeTree::new();
        assert!(!t.has_glass());
        let a = t.add_root(Node::rect().build());
        assert!(!t.has_glass());
        let g = t.add_root(Node::glass().build());
        assert!(t.has_glass());
        let g2 = t.add_root(Node::glass().build());
        assert!(t.has_glass());
        t.remove(g);
        assert!(t.has_glass());
        t.remove(g2);
        assert!(!t.has_glass());
        t.remove(a);
        assert!(!t.has_glass());
    }

    #[test]
    fn layout_setter_skips_backdrop_without_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        t.take_dirty();
        t.set_layout_width(id, Len::Px(50.0));
        let d = t.dirty_for_test();
        assert!(d & dirty::TRANSFORM != 0);
        assert!(d & dirty::BACKDROP == 0, "no glass → no BACKDROP flag");
    }

    #[test]
    fn layout_setter_flags_backdrop_with_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        let _g = t.add_root(Node::glass().build());
        t.take_dirty();
        t.set_layout_width(id, Len::Px(50.0));
        let d = t.dirty_for_test();
        assert!(d & dirty::TRANSFORM != 0);
        assert!(d & dirty::BACKDROP != 0);
    }

    #[test]
    fn set_color_skips_backdrop_without_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        t.take_dirty();
        t.set_color(id, [0.5, 0.5, 0.5, 1.0]);
        let d = t.dirty_for_test();
        assert!(d & dirty::VISUAL != 0);
        assert!(d & dirty::BACKDROP == 0);
    }

    #[test]
    fn scroll_state_allocates_on_overflow_scroll() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        assert!(t.get(id).unwrap().scroll.is_none());
        t.set_layout_overflow(id, crate::layout::Overflow::Scroll, crate::layout::Overflow::Visible);
        assert!(t.get(id).unwrap().scroll.is_some());
    }

    #[test]
    fn add_scroll_delta_clamps_and_reports_remainder() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        // content > rect → 100 px scroll budget on y.
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 100.0];
            n.content_size = [200.0, 200.0];
        }
        let applied = t.add_scroll_delta(id, [0.0, 200.0]);
        assert!((applied[1] - 100.0).abs() < 0.01, "clamped applied = {applied:?}");
        // Already at edge — next push should report zero applied so
        // wheel routing can bubble.
        let again = t.add_scroll_delta(id, [0.0, 50.0]);
        assert_eq!(again, [0.0, 0.0]);
    }

    #[test]
    fn add_scroll_delta_ignores_non_scroll_axis() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_x().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [400.0, 400.0];
        }
        // y has plenty of content but isn't a scroll axis — should be 0.
        let applied = t.add_scroll_delta(id, [0.0, 50.0]);
        assert_eq!(applied, [0.0, 0.0]);
    }

    #[test]
    fn tick_scrolls_eases_toward_target_and_snaps() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 100.0];
            n.content_size = [200.0, 1100.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 1000.0]);
        // ~2 sec of 60 Hz ticks at default stiffness 12 → spring snaps
        // within the first half-second, then bar_alpha (default 0.8 s
        // fade) drains to 0 — has_active_scrolls returns false only
        // after both are settled.
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current, s.target, "should have snapped");
        assert_eq!(s.bar_alpha, 0.0, "bar fade should have drained");
        assert!(!t.has_active_scrolls());
    }

    #[test]
    fn set_color_on_glass_never_flags_backdrop() {
        let mut t = NodeTree::new();
        let g = t.add_root(Node::glass().build());
        t.take_dirty();
        t.set_color(g, [1.0, 0.0, 0.0, 0.5]);
        let d = t.dirty_for_test();
        assert!(d & dirty::VISUAL != 0);
        assert!(
            d & dirty::BACKDROP == 0,
            "glass color change doesn't enter the backdrop"
        );
    }

    #[test]
    fn flatten_emits_scrollbar_hits() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 200.0];
            n.content_size = [800.0, 1000.0];
            // Force-show the bars regardless of fade — we want geometry.
            if let Some(s) = n.scroll.as_mut() {
                s.bar_alpha = 1.0;
            }
        }
        let (_events, _hits, _scroll_hits, bars) = t.flatten(1.0);
        assert_eq!(bars.len(), 2, "two bars (X + Y) expected");
        let x = bars.iter().find(|b| b.axis == ScrollAxis::X).unwrap();
        let y = bars.iter().find(|b| b.axis == ScrollAxis::Y).unwrap();
        assert!(x.track_travel > 0.0);
        assert!(y.track_travel > 0.0);
        assert_eq!(x.max_offset, 800.0 - 200.0);
        assert_eq!(y.max_offset, 1000.0 - 200.0);
    }

    #[test]
    fn always_visible_keeps_alpha_pinned() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .scrollbar(|s| s.always_visible(true))
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        // No movement at all — but the tick should still pin alpha.
        for _ in 0..30 {
            t.tick_scrolls(1.0 / 60.0);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_alpha, 1.0, "always_visible must hold alpha at 1");
        assert!(t.has_active_scrolls(), "always_visible keeps loop ticking");
    }

    #[test]
    fn auto_hide_skips_pop_on_movement() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .scrollbar(|s| s.auto_hide(true))
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 100.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        // auto_hide: target moved but bar should still be invisible.
        assert_eq!(s.bar_alpha, 0.0, "auto_hide must not pop on scroll");
    }

    #[test]
    fn bar_hover_pops_alpha_to_one() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        assert_eq!(t.get(id).unwrap().scroll.unwrap().bar_alpha, 0.0);
        let changed = t.set_bar_hover(id, [false, true]);
        assert!(changed);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_alpha, 1.0);
        assert_eq!(s.bar_hover, [false, true]);
    }

    #[test]
    fn snap_step_retargets_to_nearest_multiple_after_settle() {
        let mut t = NodeTree::new();
        let id = t
            .add_root(Node::rect().scroll_y().snap_step_y(50.0).build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0];
        }
        // Wheel pushes 38 px; spring settles at 38; settle snaps to 50.
        let _ = t.add_scroll_delta(id, [0.0, 38.0]);
        let dt = 1.0 / 60.0;
        // Run enough ticks for spring to settle, snap, settle again,
        // and bar fade to drain so has_active_scrolls = false.
        for _ in 0..240 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 50.0, "target should snap to 50");
        assert_eq!(s.current[1], 50.0, "spring should chase to snapped");
    }

    #[test]
    fn snap_step_lands_on_max_off_when_near_bottom() {
        // max_off=130 isn't a clean multiple of step=50. When the user
        // scrolls past the last clean multiple (100), settle should
        // land on max_off itself (treats the bottom edge as a virtual
        // snap point) — the alternative (100) would clip the bottom of
        // the list with 30 px of empty space below the last "row".
        let mut t = NodeTree::new();
        let id = t
            .add_root(Node::rect().scroll_y().snap_step_y(50.0).build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 230.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 1000.0]);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 130.0, "expected max_off, got {}", s.target[1]);
    }

    #[test]
    fn snap_step_lands_on_multiple_when_far_from_edge() {
        // Mid-list scroll: nearest multiple is closer than max_off.
        let mut t = NodeTree::new();
        let id = t
            .add_root(Node::rect().scroll_y().snap_step_y(50.0).build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0]; // max_off = 900
        }
        // Push to ~280 — nearest multiple is 300, max_off (900) is far.
        let _ = t.add_scroll_delta(id, [0.0, 280.0]);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 300.0);
    }

    #[test]
    fn drag_rubber_bands_with_overscroll_capped_at_limit() {
        // set_scroll_immediate (used by thumb drag) honours overscroll
        // mode but caps via the rubber-band asymptote at
        // `max_off + OVERSCROLL_LIMIT`. Slow drag past the track end
        // produces an asymptotic stretch — never a runaway "scroll out
        // of view" — so the visual matches native scrollbar behaviour.
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0]; // max_off = 400, limit = 60
        }
        t.set_scroll_immediate(id, ScrollAxis::Y, 9999.0);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(
            s.current[1] > 400.0 && s.current[1] < 400.0 + 60.0,
            "drag should rubber-band into [max, max+limit), got {}",
            s.current[1]
        );
    }

    #[test]
    fn drag_without_overscroll_hard_clamps() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        t.set_scroll_immediate(id, ScrollAxis::Y, 9999.0);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current[1], 400.0);
    }

    #[test]
    fn end_drag_retargets_to_snapped_in_range() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .snap_step_y(50.0)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0]; // max_off=400, snap=50
        }
        // Simulate drag past edge: current/target ~440 (rubber-banded).
        t.set_scroll_immediate(id, ScrollAxis::Y, 1000.0);
        // End drag retargets to clamped + snapped position. Spring
        // should now chase from past-edge current toward in-range
        // multiple. Target should land at 400 (max_off, treated as
        // virtual snap point).
        crate::input::end_drag(id, ScrollAxis::Y, &mut t);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 400.0, "target should retarget to max_off");
        assert!(s.current[1] > 400.0, "current still past edge ready to bounce");
    }

    #[test]
    fn add_scroll_delta_snaps_immediately_in_range() {
        // Snap-on-input: target lands on the nearest multiple right
        // away. No "settle then jump" pause once the spring catches up.
        let mut t = NodeTree::new();
        let id = t
            .add_root(Node::rect().scroll_y().snap_step_y(50.0).build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 38.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 50.0, "target should snap on input, not on settle");
        // current still 0 — spring will chase 50.
        assert_eq!(s.current[1], 0.0);
    }

    #[test]
    fn bounce_spring_settles_within_one_second() {
        // Closed-form damped harmonic oscillator (k=3500, c=50, ζ≈0.42)
        // overshoots target slightly then settles. Generous 60-frame
        // budget covers the full damped oscillation including any
        // small undershoot — well past the perceptual settle (~250ms).
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0];
        }
        if let Some(n) = t.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.current[1] = 180.0;
            s.target[1] = 100.0;
        }
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current[1], 100.0, "bounce should settle within 1s");
        assert_eq!(s.bounce_elapsed[1], -1.0, "bounce flag must reset on settle");
    }

    #[test]
    fn settle_waits_for_input_quiescence() {
        // Held arrow / wheel burst: while input keeps arriving inside
        // the quiescence window, settle must NOT clamp a saturated
        // past-edge target back into range. Otherwise the next input
        // event would re-saturate it and the resulting target
        // oscillation reads as a jerk.
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0]; // max_off = 100
        }
        // Saturate past edge.
        for _ in 0..20 {
            let _ = t.add_scroll_delta(id, [0.0, 40.0]);
        }
        // Force spring to quiesce at saturated target.
        if let Some(n) = t.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.current[1] = s.target[1];
        }
        let saturated_target = t.get(id).unwrap().scroll.unwrap().target[1];
        assert!(saturated_target > 100.0, "setup must be past edge");
        // Simulate "still receiving input": fire a delta every tick at
        // a 33 ms cadence (OS auto-repeat). Settle must stay gated and
        // target must remain past edge across the burst.
        let dt = 1.0 / 60.0;
        for _ in 0..15 {
            t.tick_scrolls(dt);
            let _ = t.add_scroll_delta(id, [0.0, 40.0]);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(
            s.target[1] > 100.0,
            "settle must not fire while input is still arriving, target={}",
            s.target[1]
        );
        // Now stop sending input. After the gate elapses the next
        // tick should clamp target and start the bounce.
        for _ in 0..20 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(
            s.target[1] <= 100.0,
            "settle should fire after input quiescence, target={}",
            s.target[1]
        );
    }

    #[test]
    fn bounce_overshoots_target_slightly() {
        // Underdamped spring (default ζ ≈ 0.42) must dip past target
        // at least once — that's what gives the "alive" feel vs the
        // monotonic exponential ease used for forward chase.
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0]; // max_off = 100
        }
        if let Some(n) = t.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.current[1] = 150.0;
            s.target[1] = 100.0;
        }
        let dt = 1.0 / 240.0; // higher rate for finer overshoot capture
        let mut min_seen = f32::INFINITY;
        for _ in 0..480 {
            t.tick_scrolls(dt);
            let cur = t.get(id).unwrap().scroll.unwrap().current[1];
            if cur < min_seen {
                min_seen = cur;
            }
        }
        assert!(
            min_seen < 100.0 - 0.5,
            "underdamped bounce should overshoot below target, min={min_seen}"
        );
    }

    #[test]
    fn rubber_band_engages_only_after_first_edge_stop() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0]; // max_off = 100
        }
        // First crossing in a single big delta lands at edge; over-
        // portion dropped (the user hasn't asked for rubber-band yet).
        let _ = t.add_scroll_delta(id, [0.0, 150.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 100.0, "first crossing should stop at edge");
        // Now sitting at edge — additional push engages rubber-band.
        let _ = t.add_scroll_delta(id, [0.0, 50.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(
            s.target[1] > 100.0 && s.target[1] < 150.0,
            "subsequent push should asymptote past edge, got {}",
            s.target[1]
        );
    }

    #[test]
    fn first_crossing_clamps_at_edge_then_subsequent_pushes_rubber_band() {
        // Confirms the two-stage gate: a single big delta that crosses
        // from in-range to past-edge stops at the edge. Only when the
        // user pushes again *after* sitting at the edge does the
        // asymptote engage. Prevents the "forced" feel of rubber-band
        // on a single fast wheel event.
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 1000.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 100.0, "first crossing must stop at edge");
        let _ = t.add_scroll_delta(id, [0.0, 30.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(s.target[1] > 100.0, "second push should engage rubber-band");
    }

    #[test]
    fn rubber_band_caps_at_max_plus_limit() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0]; // max_off = 100, limit = 100
        }
        // Spam huge deltas — target asymptotically approaches 200 but
        // never exceeds it.
        for _ in 0..50 {
            let _ = t.add_scroll_delta(id, [0.0, 1000.0]);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(s.target[1] < 200.0, "must cap below limit, got {}", s.target[1]);
    }

    #[test]
    fn overscroll_target_settles_back_into_range() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0]; // max_off = 100
        }
        // Push hard past the edge. With rubber-band on, target lands
        // somewhere in [100, 100+limit] but spring will pull it back.
        for _ in 0..10 {
            let _ = t.add_scroll_delta(id, [0.0, 100.0]);
        }
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 100.0, "settle should clamp target");
        assert_eq!(s.current[1], 100.0, "spring should chase clamped");
    }

    #[test]
    fn rubber_band_unwinds_freely_toward_range() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .overscroll(true)
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 200.0];
        }
        // Two pushes to engage rubber-band past edge: first lands at
        // 100 (clamp), second asymptotes.
        let _ = t.add_scroll_delta(id, [0.0, 200.0]);
        let _ = t.add_scroll_delta(id, [0.0, 100.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        let past = s.target[1];
        assert!(past > 100.0, "must be past edge for setup, got {past}");
        // Pull back 30 px. Pull-back uses the raw delta directly so
        // target should drop by at least 30.
        let _ = t.add_scroll_delta(id, [0.0, -30.0]);
        let s2 = t.get(id).unwrap().scroll.unwrap();
        let moved = past - s2.target[1];
        assert!(
            moved >= 30.0 - 0.01,
            "back-toward-range should apply at least the requested delta, got {moved}"
        );
    }

    #[test]
    fn set_scroll_snap_step_takes_effect_after_settle() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 38.0]);
        let dt = 1.0 / 60.0;
        // Settle at 38 first.
        for _ in 0..30 {
            t.tick_scrolls(dt);
        }
        // Now configure snap; next tick re-settles to 50.
        t.set_scroll_snap_step(id, [0.0, 50.0]);
        for _ in 0..240 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 50.0);
    }

    #[test]
    fn set_scroll_immediate_writes_both_current_and_target() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        t.set_scroll_immediate(id, ScrollAxis::Y, 200.0);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current[1], 200.0);
        assert_eq!(s.target[1], 200.0, "drag must keep spring at rest");
    }
}

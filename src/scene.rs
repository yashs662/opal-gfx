//! Nested scene builder.
//!
//! `Scene` is a thin wrapper around [`NodeTree`] that hides parent-id
//! threading: nested scopes are introduced via `child(|p| { … })`
//! closures, and node handles are returned implicitly through the
//! builder chain (or looked up by name via [`SceneCtx::node`]).
//!
//! Sizing is length-based (see [`crate::layout::Len`]). Containers
//! declare `axis`/`justify`/`align`/`padding`/`gap`; leaves declare
//! `width`/`height` and (optionally) an absolute escape offset via
//! `abs(x, y)`.

use std::collections::HashMap;

use crate::gpu::ImageHandle;

/// Accepted as the `name` argument by every scene builder method.
/// Pass `&str`/`String` for a registered name (lookupable via
/// [`SceneCtx::node`]); pass `()` for an anonymous node that skips
/// name-registry insertion. Empty strings also map to anonymous so
/// `s.rect("")` continues to work without a registered entry.
///
/// Use `()` when spawning many ephemeral nodes (loop bodies, lazy-list
/// row renders, decorative leaves) — saves both the allocation and the
/// `format!()` boilerplate that would otherwise generate per-iteration
/// unique names.
pub trait IntoNodeName {
    fn into_node_name(self) -> Option<String>;
}

impl IntoNodeName for &str {
    fn into_node_name(self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(self.to_string())
        }
    }
}

impl IntoNodeName for String {
    fn into_node_name(self) -> Option<String> {
        if self.is_empty() { None } else { Some(self) }
    }
}

impl IntoNodeName for &String {
    fn into_node_name(self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(self.clone())
        }
    }
}

impl IntoNodeName for () {
    fn into_node_name(self) -> Option<String> {
        None
    }
}

impl<T: IntoNodeName> IntoNodeName for Option<T> {
    fn into_node_name(self) -> Option<String> {
        self.and_then(|v| v.into_node_name())
    }
}
use crate::layout::{Align, Axis, Justify, Len, Overflow};
use crate::node::{Node, NodeId, NodeInteract, NodeTree, WindowAction};
use crate::reactive::{Bind, ImageBind, TextBind};
use crate::signal::Signal;
use crate::text::TextResources;

/// Shared mutable state passed through every scene builder call.
/// Owned by the [`crate::app::App`] shell.
pub struct SceneCtx {
    pub tree: NodeTree,
    pub names: HashMap<String, NodeId>,
    pub binds: BindRegistry,
    /// Text shaping + rasterization cache. Shared with the layout pass
    /// (measure) and the GPU glyph-instance builder (shape + atlas
    /// upload).
    pub text: TextResources,
    /// Display scale (physical px per logical px), kept in sync by the app
    /// shell each frame. Lets per-frame hooks (`on_frame`) convert between
    /// physical reads like [`NodeTree::scroll_offset`] and logical layout
    /// units. Mirrors [`crate::event::EventCtx::scale`] for the frame path.
    pub scale: f32,
}

impl Default for SceneCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCtx {
    pub fn new() -> Self {
        Self {
            tree: NodeTree::new(),
            names: HashMap::new(),
            binds: BindRegistry::default(),
            text: TextResources::new(),
            scale: 1.0,
        }
    }

    /// Look up a previously-named node.
    pub fn node(&self, name: &str) -> Option<NodeId> {
        self.names.get(name).copied()
    }

    /// Remove a node and every descendant, then tombstone any
    /// `BindRegistry` slots pointing at the dropped nodes and prune
    /// matching entries from `names`. Returns [`SubtreeRemoval`]
    /// describing what was freed — callers (the app shell) walk the
    /// `dropped_color_slots`/`_position_slots`/`_size_slots` indices
    /// to stop any active timeline tweens keyed by those slot
    /// positions (`BIND_TWEEN_KEY_* + idx`).
    ///
    /// **Why return the indices instead of stopping tweens here?**
    /// `SceneCtx` doesn't own the `Timeline` — keeping the boundary
    /// clean prevents this method from growing a parameter list that
    /// reaches into half the App's internals.
    pub fn remove_subtree(&mut self, id: NodeId) -> SubtreeRemoval {
        let dropped = self.tree.remove_subtree(id);
        if dropped.is_empty() {
            return SubtreeRemoval::default();
        }
        let dropped_set: std::collections::HashSet<NodeId> = dropped.iter().copied().collect();

        let dropped_color_slots =
            tombstone_matching(&mut self.binds.color, &mut self.binds.color_free, |s| {
                dropped_set.contains(&s.node_id)
            });
        let dropped_position_slots = tombstone_matching(
            &mut self.binds.position,
            &mut self.binds.position_free,
            |s| dropped_set.contains(&s.node_id),
        );
        let dropped_size_slots =
            tombstone_matching(&mut self.binds.size, &mut self.binds.size_free, |s| {
                dropped_set.contains(&s.node_id)
            });
        // Image binds have no timeline tweens, so their dropped indices
        // aren't surfaced in `SubtreeRemoval` (nothing to stop) — just
        // tombstone + free-list them for reuse.
        let _ = tombstone_matching(&mut self.binds.image, &mut self.binds.image_free, |s| {
            dropped_set.contains(&s.node_id)
        });
        let _ = tombstone_matching(&mut self.binds.text, &mut self.binds.text_free, |s| {
            dropped_set.contains(&s.node_id)
        });
        let _ = tombstone_matching(
            &mut self.binds.width_pct,
            &mut self.binds.width_pct_free,
            |s| dropped_set.contains(&s.node_id),
        );
        let _ = tombstone_matching(
            &mut self.binds.width_px,
            &mut self.binds.width_px_free,
            |s| dropped_set.contains(&s.node_id),
        );
        let _ = tombstone_matching(
            &mut self.binds.height_px,
            &mut self.binds.height_px_free,
            |s| dropped_set.contains(&s.node_id),
        );
        let _ = tombstone_matching(&mut self.binds.opacity, &mut self.binds.opacity_free, |s| {
            dropped_set.contains(&s.node_id)
        });

        // Prune any named-node entries pointing at dropped ids.
        self.names.retain(|_, v| !dropped_set.contains(v));

        SubtreeRemoval {
            dropped,
            dropped_color_slots,
            dropped_position_slots,
            dropped_size_slots,
        }
    }
}

/// Summary of a `SceneCtx::remove_subtree` call.
///
/// `dropped` is the pre-order list of node ids freed from the arena.
/// The `dropped_*_slots` indices are positions in the respective
/// `BindRegistry` vectors whose slot was tombstoned (`None`-d in
/// place). App-shell code translates each index `i` into a timeline
/// tween key via `BIND_TWEEN_KEY_{COLOR|POSITION|SIZE} + i as u32`
/// and calls `timeline.stop(key)`.
#[derive(Default, Debug)]
pub struct SubtreeRemoval {
    pub dropped: Vec<NodeId>,
    pub dropped_color_slots: Vec<u32>,
    pub dropped_position_slots: Vec<u32>,
    pub dropped_size_slots: Vec<u32>,
}

fn tombstone_matching<T>(
    slots: &mut [Option<T>],
    free: &mut Vec<u32>,
    pred: impl Fn(&T) -> bool,
) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, slot) in slots.iter_mut().enumerate() {
        let take = matches!(slot, Some(s) if pred(s));
        if take {
            *slot = None;
            out.push(i as u32);
        }
    }
    // Returned indices feed the caller's `stop_tweens_for_removal`; the
    // same indices become reusable storage.
    free.extend(out.iter().copied());
    out
}

/// Per-prop reactive bind storage.
///
/// **Tombstoned vectors.** Each slot is `Option<_>`; `None` is a freed
/// slot left in place so subsequent indices stay stable. This matters
/// because tween keys in the timeline are derived from the slot's
/// position (`BIND_TWEEN_KEY_COLOR + idx`) — `Vec::retain` would
/// re-key every active tween. `remove_subtree` writes `None` over
/// dropped slots and stops their tweens; new pushes append (don't
/// re-use tombstones) — historically slot count grew monotonically with
/// every reactive prop ever declared, which is *not* bounded under a
/// per-track-change rebuild cadence (Frostify rebuilds the Home scene
/// several times per song). Each prop kind now keeps a free-list of
/// tombstoned indices; [`alloc_slot`] reuses one before appending, so
/// the vectors plateau at the high-water mark of *concurrently-live*
/// binds. Safe despite tween keys being index-derived: tweens target
/// cloned `displayed` signals (not the index), and `Timeline::start`
/// with a reused key evicts any lingering tween, so a reused slot never
/// inherits stale animation.
#[derive(Default)]
pub struct BindRegistry {
    pub color: Vec<Option<ColorBindSlot>>,
    /// Tombstoned `color` indices available for reuse.
    pub color_free: Vec<u32>,
    /// Absolute position binds. Each slot drives `layout.abs = Some([x,y])`
    /// — node must already declare `.abs(...)` (or have it set later) to
    /// participate in layout, but a position bind on a flow child still
    /// flips it into absolute mode on first apply.
    pub position: Vec<Option<PositionBindSlot>>,
    /// Tombstoned `position` indices available for reuse.
    pub position_free: Vec<u32>,
    /// Size binds. Each slot drives `layout.width/height = Px(_)`.
    pub size: Vec<Option<SizeBindSlot>>,
    /// Tombstoned `size` indices available for reuse.
    pub size_free: Vec<u32>,
    /// Image-handle binds. Each slot drives `node.image`. No tween (a
    /// handle swap is discrete), so unlike the others these carry no
    /// `displayed` signal and no timeline key.
    pub image: Vec<Option<ImageBindSlot>>,
    /// Tombstoned `image` indices available for reuse.
    pub image_free: Vec<u32>,
    /// Text-content binds. Each slot drives `node.text.content` via
    /// `set_text` (which relayouts — text width may change). No tween.
    pub text: Vec<Option<TextBindSlot>>,
    /// Tombstoned `text` indices available for reuse.
    pub text_free: Vec<u32>,
    /// Percentage-width binds. Each slot drives `layout.width =
    /// Len::Pct(_)` from an `f32` source — for responsive fills like a
    /// progress bar (a fixed-Px size bind can't express "% of parent").
    /// Snaps (no tween); animate the source signal for smooth motion.
    pub width_pct: Vec<Option<WidthPctBindSlot>>,
    /// Tombstoned `width_pct` indices available for reuse.
    pub width_pct_free: Vec<u32>,
    /// Absolute-width binds (`layout.width = Len::Px(_)`). Like
    /// `width_pct` but in physical-px-equivalent units — used for
    /// user-resizable panels driven by a drag handle. Snaps (no tween);
    /// animate the source for smooth motion.
    pub width_px: Vec<Option<WidthPxBindSlot>>,
    /// Tombstoned `width_px` indices available for reuse.
    pub width_px_free: Vec<u32>,
    /// Absolute-height binds (`layout.height = Len::Px(_)`). Mirror of
    /// `width_px` for the cross axis.
    pub height_px: Vec<Option<HeightPxBindSlot>>,
    /// Tombstoned `height_px` indices available for reuse.
    pub height_px_free: Vec<u32>,
    /// Node-opacity binds. Drive `style.opacity`, which the flatten pass
    /// multiplies down the subtree (group opacity) — so binding this on a
    /// container fades the whole subtree together (modal fade-in/out).
    /// Snaps per tick; animate the source signal for smooth motion.
    pub opacity: Vec<Option<OpacityBindSlot>>,
    /// Tombstoned `opacity` indices available for reuse.
    pub opacity_free: Vec<u32>,
}

/// Store `slot`, reusing a tombstoned index from `free` if one exists
/// (keeps the vector from growing without bound across rebuilds) and
/// otherwise appending. See [`BindRegistry`] for why index reuse is
/// safe w.r.t. timeline tween keys.
fn alloc_slot<T>(slots: &mut Vec<Option<T>>, free: &mut Vec<u32>, slot: T) {
    if let Some(idx) = free.pop() {
        slots[idx as usize] = Some(slot);
    } else {
        slots.push(Some(slot));
    }
}

pub struct ColorBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<[f32; 4]>,
    pub last_version: u64,
    pub displayed: Option<Signal<[f32; 4]>>,
}

pub struct PositionBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<[f32; 2]>,
    pub last_version: u64,
    pub displayed: Option<Signal<[f32; 2]>>,
}

pub struct SizeBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<[f32; 2]>,
    pub last_version: u64,
    pub displayed: Option<Signal<[f32; 2]>>,
}

pub struct ImageBindSlot {
    pub node_id: NodeId,
    pub bind: ImageBind,
    pub last_version: u64,
}

pub struct TextBindSlot {
    pub node_id: NodeId,
    pub bind: TextBind,
    pub last_version: u64,
}

pub struct WidthPctBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<f32>,
    pub last_version: u64,
}

pub struct WidthPxBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<f32>,
    pub last_version: u64,
}

pub struct HeightPxBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<f32>,
    pub last_version: u64,
}

pub struct OpacityBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<f32>,
    pub last_version: u64,
    /// `Some` for an animated opacity bind — the timeline tweens this toward
    /// the source value; `pump_animated_displays` pushes it into the node's
    /// opacity each frame. `None` = snap directly (the original behaviour).
    pub displayed: Option<Signal<f32>>,
}

/// A scoped scene cursor. Holds an implicit `parent` so nested
/// `child` closures don't need to thread `NodeId` by hand.
pub struct Scene<'a> {
    ctx: &'a mut SceneCtx,
    parent: Option<NodeId>,
}

impl<'a> Scene<'a> {
    pub fn root(ctx: &'a mut SceneCtx) -> Self {
        Self { ctx, parent: None }
    }

    /// Open a scene scope rooted at an existing node — typically used
    /// by mid-frame materialization passes (e.g. virtualized lists)
    /// that need to spawn children under a known parent without
    /// holding a [`NodeBuilderRef`].
    pub fn with_parent(ctx: &'a mut SceneCtx, parent: NodeId) -> Self {
        Self {
            ctx,
            parent: Some(parent),
        }
    }

    pub fn ctx(&self) -> &SceneCtx {
        self.ctx
    }

    pub fn ctx_mut(&mut self) -> &mut SceneCtx {
        self.ctx
    }

    /// Leaf rect (no children expected). Defaults to `Axis::Col` —
    /// matters only if you later nest children under it. Pass `()` for
    /// an anonymous node when you don't need name-lookup.
    pub fn rect(&mut self, name: impl IntoNodeName) -> NodeBuilderRef<'_> {
        self.spawn(name.into_node_name(), SpawnKind::Rect, Axis::Col)
    }

    /// Row container: flow children horizontally. Pass `()` for anon.
    pub fn row(&mut self, name: impl IntoNodeName) -> NodeBuilderRef<'_> {
        self.spawn(name.into_node_name(), SpawnKind::Rect, Axis::Row)
    }

    /// Column container: flow children vertically (the default). Pass `()` for anon.
    pub fn col(&mut self, name: impl IntoNodeName) -> NodeBuilderRef<'_> {
        self.spawn(name.into_node_name(), SpawnKind::Rect, Axis::Col)
    }

    /// Frosted glass rect. Pass `()` for anon.
    pub fn glass(&mut self, name: impl IntoNodeName) -> NodeBuilderRef<'_> {
        self.spawn(name.into_node_name(), SpawnKind::Glass, Axis::Col)
    }

    /// Text child. Default size is `Len::Auto` on both axes — the
    /// layout pass resolves them from the shaped bounding box. Pass
    /// `()` for the name when no lookup is needed.
    pub fn text(
        &mut self,
        name: impl IntoNodeName,
        content: impl Into<String>,
        font_size: f32,
    ) -> NodeBuilderRef<'_> {
        self.spawn(
            name.into_node_name(),
            SpawnKind::Text(content.into(), font_size),
            Axis::Col,
        )
    }

    /// Image child sourced from a previously-uploaded atlas handle.
    /// Default tint is `[1,1,1,1]` — chain `.color()`/`.rgba()` to tint.
    /// Default size is `Len::Auto`; chain `.size_px(w,h)` to fix size.
    /// Pass `()` for anon.
    pub fn image(&mut self, name: impl IntoNodeName, handle: ImageHandle) -> NodeBuilderRef<'_> {
        self.spawn(name.into_node_name(), SpawnKind::Image(handle), Axis::Col)
    }

    /// Image node whose texture handle tracks a reactive [`ImageBind`]
    /// (a `Signal`/`Computed` of `Option<ImageHandle>`, or a literal).
    /// The shell pumps it via the bind registry, so the rendered cover
    /// can change without a scene rebuild — used for the album-art
    /// backdrop/panel crossfade. A `None` value renders nothing until a
    /// handle resolves. Chain `.color()/.radius()/...` as usual.
    pub fn image_bound(
        &mut self,
        name: impl IntoNodeName,
        src: impl Into<ImageBind>,
    ) -> NodeBuilderRef<'_> {
        let bind = src.into();
        let initial = bind.read();
        let initial_version = bind.version();
        // Spawn an image node; SpawnKind::Image needs a concrete handle,
        // so seed with the initial (or a placeholder we immediately clear
        // to None — no flatten happens before then, so it never renders).
        let r = self.spawn(
            name.into_node_name(),
            SpawnKind::Image(initial.unwrap_or(ImageHandle(0))),
            Axis::Col,
        );
        let id = r.id;
        if initial.is_none()
            && let Some(n) = self.ctx.tree.get_mut_raw(id)
        {
            n.image = None;
        }
        if bind.is_reactive() {
            alloc_slot(
                &mut self.ctx.binds.image,
                &mut self.ctx.binds.image_free,
                ImageBindSlot {
                    node_id: id,
                    bind,
                    last_version: initial_version,
                },
            );
        }
        NodeBuilderRef { ctx: self.ctx, id }
    }

    /// Text node whose content tracks a reactive [`TextBind`] (a
    /// [`crate::signal::TextSignal`] or a literal `&str`/`String`). The
    /// shell pumps it via the bind registry + `set_text`, so the label
    /// updates without a scene rebuild — used for the now-playing
    /// title/artist. Chain `.color()/.max_width_px()/...` as usual.
    pub fn text_bound(
        &mut self,
        name: impl IntoNodeName,
        src: impl Into<TextBind>,
        font_size: f32,
    ) -> NodeBuilderRef<'_> {
        let bind = src.into();
        let initial = bind.read();
        let initial_version = bind.version();
        let r = self.spawn(
            name.into_node_name(),
            SpawnKind::Text(initial.to_string(), font_size),
            Axis::Col,
        );
        let id = r.id;
        if bind.is_reactive() {
            alloc_slot(
                &mut self.ctx.binds.text,
                &mut self.ctx.binds.text_free,
                TextBindSlot {
                    node_id: id,
                    bind,
                    last_version: initial_version,
                },
            );
        }
        NodeBuilderRef { ctx: self.ctx, id }
    }

    /// Virtualized fixed-height list. Acts as a scroll container; only
    /// the rows currently in (or near) the visible viewport are
    /// materialized as real tree children. `render` is invoked once
    /// per visible row at materialization time and must spawn exactly
    /// one child node (typically a row container) per call — the
    /// library positions that child via `layout.abs` at
    /// `i * item_height`.
    ///
    /// `item_count` and `item_height` (logical px) together drive the
    /// scroll container's `content_size`; scrolling the list past the
    /// edges follows the usual rubber-band / snap rules. To refresh
    /// the visible rows after mutating the data they read from, call
    /// `NodeTree::invalidate_lazy_list(id)` — bumps an internal
    /// version counter that forces a re-render even when the window
    /// is unchanged.
    pub fn lazy_list<F>(
        &mut self,
        name: impl IntoNodeName,
        item_count: u32,
        item_height: f32,
        render: F,
    ) -> NodeBuilderRef<'_>
    where
        F: Fn(&mut Scene, u32) + 'static,
    {
        // Spawn the host node — a Col container that will get
        // overflow_y = Scroll so the existing scroll machinery
        // (ScrollState allocation, wheel routing, scroll bar) picks
        // it up.
        let root_id = self
            .spawn(name.into_node_name(), SpawnKind::Rect, Axis::Col)
            .id();

        // Force scroll-y on. Tree setter reconciles ScrollState +
        // scrollable_ids in one call.
        self.ctx
            .tree
            .set_layout_overflow(root_id, Overflow::Visible, Overflow::Scroll);

        // Attach state.
        if let Some(n) = self.ctx.tree.get_mut_raw(root_id) {
            n.lazy_list = Some(Box::new(crate::lazy_list::LazyListState::new(
                item_count,
                item_height,
                render,
            )));
        }

        NodeBuilderRef {
            ctx: self.ctx,
            id: root_id,
        }
    }

    /// Editable single-line text field. Returns a builder pointing at
    /// the **parent** Rect node — chain `.rgba()`, `.border()`,
    /// `.size_px()` etc. on it like any other node. Under the hood
    /// two children are spawned: a `Text` node for the value, and a
    /// thin Rect for the caret (whose visibility tracks the parent's
    /// focused signal).
    ///
    /// `initial` seeds the value (cursor lands at end). `font_size` is
    /// in **logical** px and propagates to the inner Text node.
    ///
    /// To listen for value changes, chain `.on_change(|s| ...)`. For
    /// Enter, chain `.on_submit(|ctx| ...)`. To pre-fill placeholder
    /// text, chain `.placeholder(...)`.
    pub fn text_field(
        &mut self,
        name: impl IntoNodeName,
        initial: impl Into<String>,
        font_size: f32,
    ) -> NodeBuilderRef<'_> {
        let name = name.into_node_name();
        let initial = initial.into();
        let cursor = initial.len();

        // 1. Spawn the parent (a Row so the text + caret lay out left
        //    to right; padding/colors are caller-driven).
        let root_id = self.spawn(name, SpawnKind::Rect, Axis::Row).id();

        // 2a. Spawn the selection-highlight rect *first* so it paints
        //     behind the text (DFS child order = painter order). Abs-
        //     positioned + hidden; geometry set each layout pass from
        //     the selection span.
        let selection_node = self.ctx.tree.add_child(
            root_id,
            Node::rect()
                .color([0.20, 0.55, 0.95, 0.35])
                .hidden()
                .layout_abs(0.0, 0.0)
                .layout_size(
                    crate::layout::Len::Px(0.0),
                    crate::layout::Len::Px(font_size),
                )
                .build(),
        );

        // 2b. Spawn the text child. Content = value when non-empty.
        //    Empty value renders an empty Text node — placeholder
        //    handling is layered on later via `.placeholder()`.
        let text_node = self
            .ctx
            .tree
            .add_child(root_id, Node::text(initial.clone(), font_size).build());

        // 3. Spawn the caret. 2 logical-px wide rect, abs-positioned
        //    inside the parent. Initially hidden — visibility tracks
        //    the focused signal (driven via App-side sync each react).
        let caret_node = self.ctx.tree.add_child(
            root_id,
            Node::rect()
                .color([1.0, 1.0, 1.0, 0.9])
                .hidden()
                .layout_abs(0.0, 0.0)
                .layout_size(
                    crate::layout::Len::Px(2.0),
                    crate::layout::Len::Px(font_size),
                )
                .build(),
        );

        // 4. Wire focused signal + EditorState on the parent. Hover +
        //    pressed signals are *not* auto-allocated — caller can
        //    add them explicitly via `.on_hover(...)` etc. if they
        //    want hover styling.
        let focused = crate::signal::Signal::new(false);
        if let Some(n) = self.ctx.tree.get_mut_raw(root_id) {
            n.interact.focused = Some(focused);
            n.editor = Some(Box::new(crate::editor::EditorState {
                value: initial,
                cursor,
                selection_anchor: None,
                placeholder: String::new(),
                font_size,
                text_node,
                caret_node,
                selection_node,
                on_change: None,
                on_submit: None,
            }));
        }

        NodeBuilderRef {
            ctx: self.ctx,
            id: root_id,
        }
    }

    fn spawn(&mut self, name: Option<String>, kind: SpawnKind, axis: Axis) -> NodeBuilderRef<'_> {
        let mut node = match kind {
            SpawnKind::Rect => Node::rect().build(),
            SpawnKind::Glass => Node::glass().build(),
            SpawnKind::Text(content, size) => Node::text(content, size).build(),
            SpawnKind::Image(handle) => Node::image(handle).build(),
        };
        node.layout.axis = axis;
        let id = match self.parent {
            Some(p) => self.ctx.tree.add_child(p, node),
            None => self.ctx.tree.add_root(node),
        };
        if let Some(n) = name {
            self.ctx.names.insert(n, id);
        }
        NodeBuilderRef { ctx: self.ctx, id }
    }
}

enum SpawnKind {
    Rect,
    Glass,
    Text(String, f32),
    Image(ImageHandle),
}

pub struct NodeBuilderRef<'a> {
    ctx: &'a mut SceneCtx,
    id: NodeId,
}

impl<'a> NodeBuilderRef<'a> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    // --- layout ---

    pub fn w(&mut self, len: Len) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.width = len;
        }
        self
    }

    pub fn h(&mut self, len: Len) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.height = len;
        }
        self
    }

    pub fn size(&mut self, w: Len, h: Len) -> &mut Self {
        self.w(w).h(h)
    }

    pub fn w_px(&mut self, px: f32) -> &mut Self {
        self.w(Len::Px(px))
    }

    pub fn h_px(&mut self, px: f32) -> &mut Self {
        self.h(Len::Px(px))
    }

    pub fn size_px(&mut self, w: f32, h: f32) -> &mut Self {
        self.size(Len::Px(w), Len::Px(h))
    }

    pub fn fill(&mut self) -> &mut Self {
        self.size(Len::Fill, Len::Fill)
    }

    pub fn pad(&mut self, all: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.padding = [all; 4];
        }
        self
    }

    pub fn pad_xy(&mut self, x: f32, y: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.padding = [x, y, x, y];
        }
        self
    }

    pub fn pad_ltrb(&mut self, l: f32, t: f32, r: f32, b: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.padding = [l, t, r, b];
        }
        self
    }

    pub fn gap(&mut self, g: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.gap = g;
        }
        self
    }

    pub fn justify(&mut self, j: Justify) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.justify = j;
        }
        self
    }

    pub fn align(&mut self, a: Align) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.align = a;
        }
        self
    }

    /// Mark as absolutely positioned at `[x, y]` relative to the
    /// parent's content-box origin. Escapes flow layout.
    pub fn abs(&mut self, x: f32, y: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.abs = Some([x, y]);
        }
        self
    }

    /// Shortcut for `justify(Justify::Center).align(Align::Center)`.
    /// Centers all flow children on both axes within this container —
    /// the common case for "icon centered in a button" where the
    /// icon's intrinsic size is known to the layout pass (text via the
    /// measurer, fixed-size rects via `.size_px(...)`).
    pub fn center(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.justify = Justify::Center;
            n.layout.align = Align::Center;
        }
        self
    }

    /// Width-to-height ratio constraint. Width remains the driver
    /// (set via `w(Fill)` / `w_px` / `w_pct` / `width_px_bind`); the
    /// layout pass overrides height to `width / ratio` after the
    /// parent's flex pass resolves the width. Any explicit `h_*` set on
    /// the same node is ignored. Common values:
    /// - `1.0` — square (use [`Self::square`] for clarity)
    /// - `16.0 / 9.0` — widescreen tile
    /// - `4.0 / 3.0` — classic photo
    pub fn aspect_ratio(&mut self, ratio: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.aspect_ratio = Some(ratio.max(f32::EPSILON));
        }
        self
    }

    /// Shortcut for [`Self::aspect_ratio`]`(1.0)` — height tracks width.
    /// Pair with `w(Fill)` to get an album-cover-style square that
    /// resizes with its container.
    pub fn square(&mut self) -> &mut Self {
        self.aspect_ratio(1.0)
    }

    /// Constrain the text content to a maximum logical-px width;
    /// when the unconstrained shape exceeds it, the layout pass and
    /// glyph builder both substitute `prefix + "…"` truncation. No-op
    /// on non-text nodes.
    pub fn max_width_px(&mut self, px: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id)
            && let Some(t) = n.text.as_mut()
        {
            t.max_width = Some(px);
        }
        self
    }

    /// Push this flow child (and every sibling after it) to the end of
    /// the parent's main axis — equivalent to CSS `margin-left: auto`
    /// on the first such child. Only effective when the parent's
    /// `justify` is `Justify::Start` (the default). Useful for
    /// title-bar layouts where some buttons hug the left and others
    /// hug the right.
    pub fn push_end(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.push_end = true;
        }
        self
    }

    /// Opt this node into Tab-focus cycling at the given order. `0`
    /// (the default) excludes it. Nodes are visited in ascending order,
    /// ties broken by creation order. Pair with [`Self::on_focus`] to
    /// drive a focus-ring signal. See [`crate::App`] Tab handling.
    pub fn focus_order(&mut self, order: u32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.focus_order = order;
        }
        self
    }

    /// Reactive absolute position. Accepts a literal `[x, y]`, a
    /// `Signal<[f32; 2]>`, a `Computed<[f32; 2]>` or
    /// `animated(...)`. Forces the node into absolute layout mode
    /// (sets `layout.abs = Some(_)`).
    pub fn pos(&mut self, pos: impl Into<Bind<[f32; 2]>>) -> &mut Self {
        let bind = pos.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.abs = Some(initial);
        }
        let is_reactive = !matches!(bind, Bind::Value(_));
        let is_animated = bind.animation().is_some();
        if is_reactive || is_animated {
            let displayed = if is_animated {
                Some(Signal::new(initial))
            } else {
                None
            };
            alloc_slot(
                &mut self.ctx.binds.position,
                &mut self.ctx.binds.position_free,
                PositionBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                    displayed,
                },
            );
        }
        self
    }

    /// Reactive size in pixels. Both axes follow the bind value as
    /// `Len::Px`. Accepts the same shapes as [`Self::pos`].
    pub fn size_bind(&mut self, size: impl Into<Bind<[f32; 2]>>) -> &mut Self {
        let bind = size.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.width = Len::Px(initial[0]);
            n.layout.height = Len::Px(initial[1]);
        }
        let is_reactive = !matches!(bind, Bind::Value(_));
        let is_animated = bind.animation().is_some();
        if is_reactive || is_animated {
            let displayed = if is_animated {
                Some(Signal::new(initial))
            } else {
                None
            };
            alloc_slot(
                &mut self.ctx.binds.size,
                &mut self.ctx.binds.size_free,
                SizeBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                    displayed,
                },
            );
        }
        self
    }

    /// Reactive width as a percentage (`0.0..=1.0`) of the parent's
    /// content width, driven by an `f32` bind — the responsive companion
    /// to [`Self::size_bind`] (which is fixed Px). Snaps on change (no
    /// tween); for smooth motion animate the source signal. Used for the
    /// player progress bar so it tracks playback without a rebuild.
    pub fn width_pct(&mut self, pct: impl Into<Bind<f32>>) -> &mut Self {
        let bind = pct.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.width = Len::Pct(initial);
        }
        if !matches!(bind, Bind::Value(_)) {
            alloc_slot(
                &mut self.ctx.binds.width_pct,
                &mut self.ctx.binds.width_pct_free,
                WidthPctBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                },
            );
        }
        self
    }

    /// Reactive width in **logical pixels**, driven by an `f32` bind —
    /// the absolute-size companion to [`Self::width_pct`]. Lets a
    /// caller-owned `Signal<f32>` resize a panel without a scene
    /// rebuild (e.g. a draggable splitter mutating the signal on every
    /// cursor move). Snaps (no tween); animate the source for smooth
    /// motion.
    pub fn width_px_bind(&mut self, w: impl Into<Bind<f32>>) -> &mut Self {
        let bind = w.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.width = Len::Px(initial);
        }
        if !matches!(bind, Bind::Value(_)) {
            alloc_slot(
                &mut self.ctx.binds.width_px,
                &mut self.ctx.binds.width_px_free,
                WidthPxBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                },
            );
        }
        self
    }

    /// Reactive height in **logical pixels**. Mirror of
    /// [`Self::width_px_bind`] for the cross axis.
    pub fn height_px_bind(&mut self, h: impl Into<Bind<f32>>) -> &mut Self {
        let bind = h.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.height = Len::Px(initial);
        }
        if !matches!(bind, Bind::Value(_)) {
            alloc_slot(
                &mut self.ctx.binds.height_px,
                &mut self.ctx.binds.height_px_free,
                HeightPxBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                },
            );
        }
        self
    }

    /// Reactive node opacity (0..=1). Multiplies into the subtree's
    /// effective alpha (group opacity — see the flatten pass), so binding
    /// it on a container fades the whole subtree together. Ideal for a
    /// modal/popup fade-in-out: tween a `Signal<f32>` and bind it here.
    /// Snaps per tick; animate the source for smooth motion.
    ///
    /// A fully-transparent value (≤ 0.001) also marks the node
    /// **invisible**, so the flatten pass skips it and its subtree
    /// entirely — no render *and no hit-testing*. This is what stops a
    /// faded-out overlay (e.g. a closed modal's full-window scrim) from
    /// silently eating input while it's invisible.
    pub fn opacity_bind(&mut self, o: impl Into<Bind<f32>>) -> &mut Self {
        let bind = o.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.opacity = initial;
            n.visible = initial > 0.001;
        }
        if !matches!(bind, Bind::Value(_)) {
            // Animated binds tween a `displayed` signal (seeded to the
            // current value); non-animated reactive binds snap.
            let displayed = if bind.animation().is_some() {
                Some(Signal::new(initial))
            } else {
                None
            };
            alloc_slot(
                &mut self.ctx.binds.opacity,
                &mut self.ctx.binds.opacity_free,
                OpacityBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                    displayed,
                },
            );
        }
        self
    }

    pub fn axis(&mut self, a: Axis) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.axis = a;
        }
        self
    }

    /// Mark this node as backdrop content (behind a glass overlay) so its
    /// colour/image/layout changes re-run the blur. Without it, changes
    /// are treated as front-of-glass and skip the (expensive) blur pass.
    /// See [`crate::node::Node::blur_source`].
    pub fn blur_source(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.blur_source = true;
        }
        self
    }

    /// Force-promote this subtree to its own compositor layer. See
    /// [`crate::node::Node::layer`].
    pub fn layer(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layer = true;
        }
        self
    }

    /// Promote to a layer (implies [`Self::layer`]) and drive its composite
    /// opacity from `signal` each frame — composite-only, no re-raster.
    /// See [`crate::node::Node::layer_opacity`].
    pub fn layer_opacity(&mut self, signal: crate::signal::Signal<f32>) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layer = true;
            n.layer_opacity = Some(signal);
        }
        self
    }

    /// Promote to a layer (implies [`Self::layer`]) and drive its composite
    /// **X offset** (logical px) from `signal` each frame — composite-only,
    /// no re-flatten and no relayout. Lets an overlay (e.g. the seek-bar
    /// timestamp tooltip) follow the cursor without dirtying the layout tree.
    /// See [`crate::node::Node::layer_offset_x`].
    pub fn layer_offset_x(&mut self, signal: crate::signal::Signal<f32>) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layer = true;
            n.layer_offset_x = Some(signal);
        }
        self
    }

    /// Mark this node as an **external-texture layer** (P6). See
    /// [`crate::node::Node::external`].
    pub fn external(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.external = true;
        }
        self
    }

    /// Image nodes: scale the texture to **cover** the rect (preserve
    /// aspect, crop) instead of stretching. See
    /// [`crate::node::NodeBuilder::image_cover`].
    pub fn image_cover(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.image_cover = true;
        }
        self
    }

    /// Composite-time **edge fade** per side `[top, right, bottom, left]`
    /// (0..1). Applies to any promoted layer. See
    /// [`crate::node::NodeBuilder::fade_edges`].
    pub fn fade_edges(&mut self, top: f32, right: f32, bottom: f32, left: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.edge_fade = [
                top.clamp(0.0, 1.0),
                right.clamp(0.0, 1.0),
                bottom.clamp(0.0, 1.0),
                left.clamp(0.0, 1.0),
            ];
        }
        self
    }
    /// Falloff exponent for the edge fade (1 = linear). See
    /// [`crate::node::NodeBuilder::fade_falloff`].
    pub fn fade_falloff(&mut self, exp: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.edge_fade_falloff = exp.max(0.0);
        }
        self
    }
    /// Convenience: fade only the bottom edge over `frac` of the rect.
    pub fn fade_bottom(&mut self, frac: f32) -> &mut Self {
        self.fade_edges(0.0, 0.0, frac, 0.0)
    }

    // --- overflow / scroll ---

    /// Set per-axis overflow. Goes through [`NodeTree::set_layout_overflow`]
    /// so `ScrollState` allocation and the `scrollable_ids` index stay
    /// in sync.
    pub fn overflow(&mut self, ox: Overflow, oy: Overflow) -> &mut Self {
        self.ctx.tree.set_layout_overflow(self.id, ox, oy);
        self
    }

    pub fn overflow_x(&mut self, o: Overflow) -> &mut Self {
        let oy = self
            .ctx
            .tree
            .get(self.id)
            .map(|n| n.layout.overflow_y)
            .unwrap_or(Overflow::Visible);
        self.ctx.tree.set_layout_overflow(self.id, o, oy);
        self
    }

    pub fn overflow_y(&mut self, o: Overflow) -> &mut Self {
        let ox = self
            .ctx
            .tree
            .get(self.id)
            .map(|n| n.layout.overflow_x)
            .unwrap_or(Overflow::Visible);
        self.ctx.tree.set_layout_overflow(self.id, ox, o);
        self
    }

    /// Both axes scroll.
    pub fn scroll(&mut self) -> &mut Self {
        self.overflow(Overflow::Scroll, Overflow::Scroll)
    }

    pub fn scroll_x(&mut self) -> &mut Self {
        self.overflow_x(Overflow::Scroll)
    }

    pub fn scroll_y(&mut self) -> &mut Self {
        self.overflow_y(Overflow::Scroll)
    }

    /// Both axes Hidden — clip without accepting wheel input.
    pub fn clip(&mut self) -> &mut Self {
        self.overflow(Overflow::Hidden, Overflow::Hidden)
    }

    /// Spring stiffness controlling scroll smoothness. Higher = snappier.
    /// No-op on non-scrollable nodes (overflow not set to Scroll on
    /// either axis). Default 12 ≈ 100 ms time-to-converge.
    pub fn scroll_smoothness(&mut self, k: f32) -> &mut Self {
        self.ctx.tree.set_scroll_stiffness(self.id, k);
        self
    }

    /// Allow the scroll target to push past the content edge with
    /// rubber-band damping; the spring snaps back into range when the
    /// user lets go. Default false.
    pub fn overscroll(&mut self, on: bool) -> &mut Self {
        self.ctx.tree.set_scroll_overscroll(self.id, on);
        self
    }

    /// Tune the bounce-back spring for overscroll release. See
    /// [`crate::node::NodeBuilder::bounce_spring`] for param semantics.
    /// Defaults `(800, 42)` give a graceful pull-back in ~280 ms with
    /// minimal overshoot.
    pub fn bounce_spring(&mut self, stiffness: f32, damping: f32) -> &mut Self {
        self.ctx
            .tree
            .set_scroll_bounce_spring(self.id, stiffness, damping);
        self
    }

    /// Per-axis snap step in **logical** px. After every scroll input
    /// the spring eases to the nearest multiple — useful for row-based
    /// lists, paged carousels, etc. `0` on an axis disables snap.
    pub fn snap_step(&mut self, x: f32, y: f32) -> &mut Self {
        self.ctx.tree.set_scroll_snap_step(self.id, [x, y]);
        self
    }

    /// Y-only snap step. See [`Self::snap_step`].
    pub fn snap_step_y(&mut self, px: f32) -> &mut Self {
        let cur = self
            .ctx
            .tree
            .get(self.id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.snap_step[0])
            .unwrap_or(0.0);
        self.ctx.tree.set_scroll_snap_step(self.id, [cur, px]);
        self
    }

    /// X-only snap step. See [`Self::snap_step`].
    pub fn snap_step_x(&mut self, px: f32) -> &mut Self {
        let cur = self
            .ctx
            .tree
            .get(self.id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.snap_step[1])
            .unwrap_or(0.0);
        self.ctx.tree.set_scroll_snap_step(self.id, [px, cur]);
        self
    }

    /// Replace this node's scrollbar style outright. Allocates a
    /// `ScrollState` if the node isn't already scrollable so style
    /// can be authored before `.scroll*()`.
    pub fn scrollbar_style(&mut self, style: crate::node::ScrollbarStyle) -> &mut Self {
        self.ctx.tree.set_scrollbar_style(self.id, style);
        self
    }

    /// Mutate the scrollbar style with a closure: e.g.
    /// `.scrollbar(|s| s.thickness(8.0).thumb_color([1,1,1,0.7]))`.
    pub fn scrollbar<F: FnOnce(crate::node::ScrollbarStyle) -> crate::node::ScrollbarStyle>(
        &mut self,
        f: F,
    ) -> &mut Self {
        self.ctx.tree.with_scrollbar_style(self.id, |s| *s = f(*s));
        self
    }

    /// Override a single lazy-list row's logical height (switches the list
    /// into variable-height mode). Used to give a list a tall first "hero"
    /// row that scrolls away with the content while the rest stay uniform.
    /// No-op on a non-lazy-list node.
    pub fn lazy_list_row_height(&mut self, row: u32, height: f32) -> &mut Self {
        self.ctx.tree.set_lazy_list_row_height(self.id, row, height);
        self
    }

    // --- style ---

    pub fn color(&mut self, color: impl Into<Bind<[f32; 4]>>) -> &mut Self {
        let bind = color.into();
        let initial = bind.read();
        // Remember the colour bind as the node's reactive base and, if
        // interaction sugar is already wired, refresh its base so it keeps
        // following the live source (call order no longer matters).
        let has_sugar = if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.color = initial;
            n.color_bind = Some(bind.clone());
            if let Some(ic) = n.interact_colors.as_mut() {
                ic.base = bind.clone();
                true
            } else {
                false
            }
        } else {
            false
        };
        // With sugar present the colour flows through its `Computed`;
        // rebuild that against the new base rather than registering a
        // competing slot (which would race the sugar slot).
        if has_sugar {
            self.rebuild_color_sugar();
        } else {
            self.register_color_slot(bind);
        }
        self
    }

    /// Register a colour bind slot directly (no sugar interaction). Used
    /// by [`Self::color`] for the plain case and by
    /// [`Self::rebuild_color_sugar`] to install the sugar `Computed`.
    fn register_color_slot(&mut self, bind: Bind<[f32; 4]>) {
        let initial = bind.read();
        let initial_version = bind.version();
        let is_reactive = !matches!(bind, Bind::Value(_));
        let is_animated = bind.animation().is_some();
        if is_reactive || is_animated {
            let displayed = if is_animated {
                Some(Signal::new(initial))
            } else {
                None
            };
            alloc_slot(
                &mut self.ctx.binds.color,
                &mut self.ctx.binds.color_free,
                ColorBindSlot {
                    node_id: self.id,
                    bind,
                    last_version: initial_version,
                    displayed,
                },
            );
        }
    }

    pub fn rgb(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color([r, g, b, 1.0])
    }

    pub fn rgba(&mut self, r: f32, g: f32, b: f32, a: f32) -> &mut Self {
        self.color([r, g, b, a])
    }

    pub fn radius(&mut self, r: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_radius = [r; 4];
        }
        self
    }

    pub fn radii(&mut self, tl: f32, tr: f32, bl: f32, br: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_radius = [tl, tr, bl, br];
        }
        self
    }

    pub fn border(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_width = width;
            n.style.border_color = color;
            n.style.border_sides = crate::node::BorderSides::ALL;
        }
        self
    }
    /// Border on a specific mask of sides — see
    /// [`crate::node::NodeBuilder::border_sides`].
    pub fn border_sides(
        &mut self,
        sides: crate::node::BorderSides,
        width: f32,
        color: [f32; 4],
    ) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_width = width;
            n.style.border_color = color;
            n.style.border_sides = sides;
        }
        self
    }
    pub fn border_bottom(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        self.border_sides(crate::node::BorderSides::BOTTOM, width, color)
    }
    pub fn border_top(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        self.border_sides(crate::node::BorderSides::TOP, width, color)
    }
    pub fn border_left(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        self.border_sides(crate::node::BorderSides::LEFT, width, color)
    }
    pub fn border_right(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        self.border_sides(crate::node::BorderSides::RIGHT, width, color)
    }

    pub fn shadow(
        &mut self,
        offset: [f32; 2],
        blur: f32,
        color: [f32; 4],
        opacity: f32,
    ) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.shadow_offset = offset;
            n.style.shadow_blur = blur;
            n.style.shadow_color = color;
            n.style.shadow_opacity = opacity;
        }
        self
    }

    pub fn opacity(&mut self, o: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.opacity = o;
        }
        self
    }

    /// Per-shape visual scale around the rect centre. Affects render
    /// only — layout + hit-test see the pre-scale geometry, so a
    /// hover-grow effect doesn't shift click boxes.
    pub fn scale(&mut self, s: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.scale = [s, s];
        }
        self
    }
    pub fn scale_xy(&mut self, sx: f32, sy: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.scale = [sx, sy];
        }
        self
    }

    pub fn hidden(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.visible = false;
        }
        self
    }

    /// Per-glass backdrop blur radius (logical px). 0 = sharp.
    pub fn blur(&mut self, px: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.blur_amount = px;
        }
        self
    }

    /// Per-glass edge refraction strength (logical px). 0 disables.
    pub fn refraction(&mut self, px: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.refraction = px;
        }
        self
    }

    /// Per-glass frosted-texture variation (logical px). Per-fragment
    /// hash scatters the backdrop sample by this many pixels at the
    /// chosen mip. 0 = mirror; ~1 = subtle frost; ~3 = pebbled.
    pub fn roughness(&mut self, px: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.roughness = px;
        }
        self
    }

    pub fn line_height(&mut self, h: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id)
            && let Some(t) = n.text.as_mut()
        {
            t.line_height = h;
        }
        self
    }

    pub fn on_hover(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.hover = Some(signal));
        self
    }

    // --- text-field sugar (no-op on non-text_field nodes) ---

    /// Set the placeholder text shown when the field's value is empty
    /// and the field is not focused. No-op on nodes that aren't text
    /// fields.
    pub fn placeholder(&mut self, text: impl Into<String>) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id)
            && let Some(ed) = n.editor.as_mut()
        {
            ed.placeholder = text.into();
        }
        self
    }

    /// Fire `f` after every value mutation. Receives the new value.
    /// No-op on non-text_field nodes.
    pub fn on_change<F: Fn(&str) + 'static>(&mut self, f: F) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id)
            && let Some(ed) = n.editor.as_mut()
        {
            ed.on_change = Some(std::rc::Rc::new(f));
        }
        self
    }

    /// Fire `f` when Enter is pressed while focused. Same shape as
    /// `Node::on_click` — receives `EventCtx`. No-op on non-text_field
    /// nodes.
    pub fn on_submit<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::EventCtx<'h>) + 'static,
    {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id)
            && let Some(ed) = n.editor.as_mut()
        {
            ed.on_submit = Some(std::rc::Rc::new(f));
        }
        self
    }

    /// Swap the node's color to `c` while the cursor hovers it. Snapshots
    /// the current `style.color` as the unhovered "base" — author your
    /// base via `.color(...)` / `.rgba(...)` **before** this call. Reuses
    /// the hover `Signal<bool>` from any prior `.on_hover(...)`; allocates
    /// one otherwise.
    ///
    /// Chains naturally with [`Self::press_color`] — the resulting
    /// [`crate::Computed`] dispatches `pressed ? press_c : hovered ? hover_c
    /// : base`. The previous color bind slot for this node is replaced;
    /// authoring a reactive `Signal<Color>` base via `.color(my_signal)`
    /// then layering sugar drops the signal (the sugar can't follow a
    /// live source — build a hand-rolled `Computed` for that case).
    pub fn hover_color(&mut self, c: [f32; 4]) -> &mut Self {
        self.update_interact_colors(|ic| ic.hover = Some(crate::node::ColorMod::Fixed(c)));
        self.rebuild_color_sugar();
        self
    }

    /// Swap the node's color to `c` while the left mouse button is
    /// pressed on it (and the cursor still inside — drag-off un-presses,
    /// matching OS button feel). See [`Self::hover_color`] for chaining
    /// semantics and base-color caveats.
    pub fn press_color(&mut self, c: [f32; 4]) -> &mut Self {
        self.update_interact_colors(|ic| ic.press = Some(crate::node::ColorMod::Fixed(c)));
        self.rebuild_color_sugar();
        self
    }

    /// Multiply the node's alpha by `factor` (typically `<1.0`) while
    /// the cursor hovers it — the common "subtle dim on hover" effect.
    /// Implemented by stamping `(base.rgb, base.a * factor)` as the
    /// hover color and routing through the same Computed pipeline as
    /// [`Self::hover_color`]; chains naturally with `press_color`.
    ///
    /// **Limitation:** this drives the node's own fill alpha, not
    /// `style.opacity` — children are unaffected. For a true
    /// fade-the-whole-subtree effect, animate `style.opacity` via a
    /// hand-rolled tween in an `on_hover` handler.
    pub fn hover_opacity(&mut self, factor: f32) -> &mut Self {
        self.update_interact_colors(|ic| {
            ic.hover = Some(crate::node::ColorMod::AlphaScale(factor))
        });
        self.rebuild_color_sugar();
        self
    }

    /// Press-state companion to [`Self::hover_opacity`]. Multiplies the
    /// node's alpha by `factor` while pressed. Same scope (own fill,
    /// not subtree) and same chaining semantics as `press_color`.
    pub fn press_opacity(&mut self, factor: f32) -> &mut Self {
        self.update_interact_colors(|ic| {
            ic.press = Some(crate::node::ColorMod::AlphaScale(factor))
        });
        self.rebuild_color_sugar();
        self
    }

    /// Get-or-create the `InteractColors` struct on this node, snapshot-
    /// ting the current `style.color` as the base on first allocation.
    fn update_interact_colors(&mut self, f: impl FnOnce(&mut crate::node::InteractColors)) {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            // Live base: the node's last `.color(...)` bind if any,
            // otherwise a constant snapshot of the resting fill.
            let base = n.color_bind.clone().unwrap_or(Bind::Value(n.style.color));
            let ic = n
                .interact_colors
                .get_or_insert(crate::node::InteractColors {
                    base,
                    hover: None,
                    press: None,
                });
            f(ic);
        }
    }

    /// Get-or-create the hover `Signal<bool>` on this node's
    /// `NodeInteract`. Returns a clone of the live signal.
    fn ensure_hover_signal(&mut self) -> Signal<bool> {
        if let Some(s) = self
            .ctx
            .tree
            .get(self.id)
            .and_then(|n| n.interact.hover.clone())
        {
            return s;
        }
        let s = Signal::new(false);
        self.with_interact(|i| i.hover = Some(s.clone()));
        s
    }

    /// Get-or-create the pressed `Signal<bool>` on this node's
    /// `NodeInteract`.
    fn ensure_press_signal(&mut self) -> Signal<bool> {
        if let Some(s) = self
            .ctx
            .tree
            .get(self.id)
            .and_then(|n| n.interact.pressed.clone())
        {
            return s;
        }
        let s = Signal::new(false);
        self.with_interact(|i| i.pressed = Some(s.clone()));
        s
    }

    /// After any sugar call mutates `InteractColors`, build (or rebuild)
    /// a `Computed` covering whichever variants are present and route it
    /// through the normal `.color()` path. Removes any prior color bind
    /// slot for this node first so we end up with exactly one slot
    /// regardless of how many sugar calls fire.
    fn rebuild_color_sugar(&mut self) {
        let ic = match self
            .ctx
            .tree
            .get(self.id)
            .and_then(|n| n.interact_colors.clone())
        {
            Some(ic) => ic,
            None => return,
        };
        // Tombstone the prior slot for this node (sugar always wins over
        // earlier sugar — last-call semantics). Tombstone (not retain)
        // keeps subsequent indices stable so timeline tween keys remain
        // valid.
        let id = self.id;
        for i in 0..self.ctx.binds.color.len() {
            if matches!(&self.ctx.binds.color[i], Some(s) if s.node_id == id) {
                self.ctx.binds.color[i] = None;
                self.ctx.binds.color_free.push(i as u32);
            }
        }

        // The base is a live bind (added as a dep), so the sugar `Computed`
        // re-evaluates whenever the resting colour moves — a reactive accent
        // now flows through hover/press states instead of being frozen at
        // build time. `ColorMod::apply` resolves each state against the
        // current base value (so `*_opacity` scales the live alpha).
        let base = ic.base;
        let computed = match (ic.hover, ic.press) {
            (Some(hm), Some(pm)) => {
                let h = self.ensure_hover_signal();
                let p = self.ensure_press_signal();
                crate::reactive::Computed::new(crate::deps!(base, h, p), move |(b, h, p)| {
                    if p {
                        pm.apply(b)
                    } else if h {
                        hm.apply(b)
                    } else {
                        b
                    }
                })
            }
            (Some(hm), None) => {
                let h = self.ensure_hover_signal();
                crate::reactive::Computed::new(crate::deps!(base, h), move |(b, h)| {
                    if h { hm.apply(b) } else { b }
                })
            }
            (None, Some(pm)) => {
                let p = self.ensure_press_signal();
                crate::reactive::Computed::new(crate::deps!(base, p), move |(b, p)| {
                    if p { pm.apply(b) } else { b }
                })
            }
            (None, None) => return,
        };
        self.register_color_slot(computed.into());
    }

    /// Install a click callback. Fires on left-button release when the
    /// release lands on the same node that captured the press — OS-button
    /// semantics, so drag-off-and-release is *not* a click. The handler
    /// runs after the input layer has updated hover/pressed/focused
    /// signals, with mutable access to the node tree via [`EventCtx`].
    ///
    /// Captures must be `'static` (own your signals or [`Rc`] them into
    /// the closure). See `examples/hello_window.rs` for the canonical
    /// "toggle a signal on click" pattern.
    pub fn on_click<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::EventCtx<'h>) + 'static,
    {
        let handler: crate::event::EventHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_click = Some(handler);
        }
        self
    }

    /// Install a right-click callback. Fires on right-button
    /// release-inside-captured — same OS-button semantics as
    /// [`Self::on_click`]. Used for context menus.
    pub fn on_right_click<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::EventCtx<'h>) + 'static,
    {
        let handler: crate::event::EventHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_right_click = Some(handler);
        }
        self
    }

    /// Install a hover-dwell callback. Fires once after the cursor has
    /// been continuously hovering this node for at least `duration`.
    /// Re-arms each time hover leaves and re-enters. The shell
    /// auto-allocates a hover [`Signal<bool>`] if one isn't already
    /// wired (mirrors the `hover_color` sugar) so the dwell can detect
    /// hover transitions. Typical use: icon tooltips.
    pub fn on_hover_dwell<F>(&mut self, duration: std::time::Duration, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::EventCtx<'h>) + 'static,
    {
        let handler: crate::event::EventHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            if n.interact.hover.is_none() {
                n.interact.hover = Some(Signal::new(false));
            }
            n.on_hover_dwell = Some((duration, handler));
        }
        self
    }

    pub fn on_press(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.pressed = Some(signal));
        self
    }

    pub fn on_focus(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.focused = Some(signal));
        self
    }

    /// Tag this node with a system [`WindowAction`]. The app shell
    /// intercepts left-presses on the node and calls into winit
    /// directly (drag the window, exit, minimize, toggle maximize)
    /// instead of running the node's own press signal.
    pub fn window_action(&mut self, action: WindowAction) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.window_action = Some(action);
        }
        self
    }

    /// Mark this node as a modal / context-menu scrim. It becomes a hit
    /// target — blocking click-through to whatever sits behind it — and a
    /// left-press on it fires [`crate::app::App::on_unhandled_press`] so
    /// the floating layer can dismiss. Typical use: a full-window
    /// absolute rect under the modal panel, tinted `rgba(0,0,0,~0.5)`.
    pub fn dismiss_transparent(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.dismiss_transparent = true;
        }
        self
    }

    /// Continuous-drag callback. Fires on every cursor move while a
    /// left-press is captured on this node, with start / current /
    /// per-event delta in physical px ([`crate::event::DragCtx`]). The
    /// primitive behind sliders + scrubbers.
    pub fn on_drag<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::DragCtx<'h>) + 'static,
    {
        let handler: crate::event::DragHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_drag = Some(handler);
        }
        self
    }

    /// Hover-move callback — fires on every cursor move while the
    /// un-pressed cursor is over this node (hover analogue of
    /// [`Self::on_drag`]). See [`crate::event::HoverCtx`].
    pub fn on_hover_move<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::HoverCtx<'h>) + 'static,
    {
        let handler: crate::event::HoverHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_hover_move = Some(handler);
        }
        self
    }

    /// Drag-end callback — fires once when a press captured on this node
    /// releases, regardless of cursor position. Pairs with
    /// [`Self::on_drag`] for commit-on-release sliders.
    pub fn on_drag_end<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::EventCtx<'h>) + 'static,
    {
        let handler: crate::event::EventHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_drag_end = Some(handler);
        }
        self
    }

    /// Attach a drag-and-drop payload. When a left-press starts on this
    /// node the lib latches a clone of `payload` as the in-flight drag;
    /// releasing over a node with [`Self::on_drop`] delivers it.
    /// Type-erased to `Rc<dyn Any>` at the boundary — the drop handler
    /// downcasts.
    pub fn drag_payload<P: 'static>(&mut self, payload: P) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.drag_payload = Some(std::rc::Rc::new(payload));
        }
        self
    }

    /// Drop-target callback. Fires when a left-press release lands on
    /// this node while a drag payload is in flight. The handler receives
    /// the type-erased payload ([`crate::event::DropCtx`]).
    pub fn on_drop<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'h> Fn(&mut crate::event::DropCtx<'h>) + 'static,
    {
        let handler: crate::event::DropHandler = std::rc::Rc::new(f);
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.on_drop = Some(handler);
        }
        self
    }

    /// Override the OS cursor while pointing at this node. Topmost
    /// hit wins. Use for resize handles (`CursorIcon::EwResize` /
    /// `NsResize`) or link affordances (`CursorIcon::Pointer`).
    pub fn cursor(&mut self, icon: winit::window::CursorIcon) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.cursor = Some(icon);
        }
        self
    }

    /// Make this node follow the cursor 1:1 while it's being dragged.
    /// During the drag the node lifts out of layout flow (a hole remains
    /// at its resting slot) and paints on top of everything, tracking the
    /// pointer. Pair with [`Self::drag_payload`] + [`Self::on_drop`] for
    /// reorderable lists / drag-into targets.
    pub fn drag_follow(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.drag_follow = true;
        }
        self
    }

    fn with_interact(&mut self, f: impl FnOnce(&mut NodeInteract)) {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            f(&mut n.interact);
        }
    }

    /// Open a nested scope rooted at this node.
    pub fn child<F: FnOnce(&mut Scene)>(&mut self, f: F) -> &mut Self {
        let mut sub = Scene {
            ctx: &mut *self.ctx,
            parent: Some(self.id),
        };
        f(&mut sub);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Curve;
    use crate::reactive::{Computed, animated};
    use std::time::Duration;

    #[test]
    fn hover_opacity_derives_hover_color_with_alpha_scaled() {
        let mut ctx = SceneCtx::new();
        const BASE: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
        let id = {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .hover_opacity(0.4)
                .id()
        };
        let n = ctx.tree.get(id).unwrap();
        let ic = n
            .interact_colors
            .as_ref()
            .expect("interact_colors not allocated");
        assert_eq!(ic.base.read(), BASE);
        let hover = ic.hover.expect("hover color not set").apply(BASE);
        assert_eq!(hover[0..3], BASE[0..3]);
        assert!((hover[3] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn anon_name_skips_registry() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.col(()).child(|p| {
                p.rect(()).size_px(10.0, 10.0);
                p.text((), "hi", 12.0);
                p.rect("").size_px(10.0, 10.0); // empty string also anon
            });
        }
        assert!(ctx.names.is_empty(), "anonymous names must not register");
        assert_eq!(ctx.tree.len(), 4, "tree still contains all 4 nodes");
    }

    #[test]
    fn named_and_anon_coexist() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.col("root").child(|p| {
                p.rect("named");
                p.rect(());
            });
        }
        assert!(ctx.node("root").is_some());
        assert!(ctx.node("named").is_some());
        assert_eq!(ctx.names.len(), 2);
    }

    #[test]
    fn nested_children_register_under_parent() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .col("root")
                .size_px(100.0, 100.0)
                .rgba(1.0, 0.0, 0.0, 1.0)
                .child(|p| {
                    p.rect("a").size_px(20.0, 20.0);
                    p.rect("b").size_px(20.0, 20.0);
                });
        }
        let root_id = ctx.node("root").unwrap();
        let a_id = ctx.node("a").unwrap();
        let b_id = ctx.node("b").unwrap();
        let root = ctx.tree.get(root_id).unwrap();
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0], a_id);
        assert_eq!(root.children[1], b_id);
        assert_eq!(ctx.tree.len(), 3);
    }

    #[test]
    fn raw_color_does_not_register_bind() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).rgba(0.5, 0.5, 0.5, 1.0);
        }
        assert!(ctx.binds.color.is_empty());
    }

    #[test]
    fn signal_color_registers_bind() {
        let s = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).color(s.clone());
        }
        assert_eq!(ctx.binds.color.len(), 1);
        let slot = ctx.binds.color[0].as_ref().unwrap();
        assert!(slot.displayed.is_none());
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn computed_color_registers_bind_with_initial_value() {
        let lit = Signal::new(false);
        let c = Computed::new((lit.clone(),), |(l,)| {
            if l {
                [0.0, 1.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0, 1.0]
            }
        });
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).color(c);
        }
        assert_eq!(ctx.binds.color.len(), 1);
        let slot = ctx.binds.color[0].as_ref().unwrap();
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn animated_color_allocates_displayed_signal() {
        let s = Signal::new([0.0_f32, 0.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).color(animated(
                s.clone(),
                Curve::EaseInOut,
                Duration::from_millis(220),
            ));
        }
        let slot = ctx.binds.color[0].as_ref().unwrap();
        assert!(slot.displayed.is_some());
        assert_eq!(slot.displayed.as_ref().unwrap().get(), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn signal_pos_registers_bind_and_seeds_layout_abs() {
        let s = Signal::new([10.0_f32, 20.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).pos(s.clone());
        }
        assert_eq!(ctx.binds.position.len(), 1);
        let slot = ctx.binds.position[0].as_ref().unwrap();
        assert!(slot.displayed.is_none());
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.layout.abs, Some([10.0, 20.0]));
    }

    #[test]
    fn animated_pos_allocates_displayed_signal() {
        let s = Signal::new([0.0_f32, 0.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).pos(animated(
                s.clone(),
                Curve::Linear,
                Duration::from_millis(100),
            ));
        }
        let slot = ctx.binds.position[0].as_ref().unwrap();
        assert!(slot.displayed.is_some());
        assert_eq!(slot.displayed.as_ref().unwrap().get(), [0.0, 0.0]);
    }

    #[test]
    fn signal_size_bind_seeds_layout_px() {
        let s = Signal::new([80.0_f32, 40.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_bind(s.clone());
        }
        assert_eq!(ctx.binds.size.len(), 1);
        let slot = ctx.binds.size[0].as_ref().unwrap();
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.layout.width, Len::Px(80.0));
        assert_eq!(n.layout.height, Len::Px(40.0));
    }

    #[test]
    fn raw_pos_does_not_register_bind() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).pos([5.0_f32, 6.0]);
        }
        assert!(ctx.binds.position.is_empty());
        let n = ctx.tree.get(ctx.node("a").unwrap()).unwrap();
        assert_eq!(n.layout.abs, Some([5.0, 6.0]));
    }

    // --- color sugar (hover_color / press_color) ---

    const BASE: [f32; 4] = [0.3, 0.5, 0.9, 1.0];
    const HOVER: [f32; 4] = [0.5, 0.7, 1.0, 1.0];
    const PRESS: [f32; 4] = [0.2, 0.4, 0.7, 1.0];

    #[test]
    fn hover_color_allocates_hover_signal_and_registers_slot() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .hover_color(HOVER);
        }
        let id = ctx.node("a").unwrap();
        let n = ctx.tree.get(id).unwrap();
        assert!(
            n.interact.hover.is_some(),
            "hover signal should be allocated"
        );
        assert!(n.interact_colors.is_some());
        assert_eq!(n.interact_colors.as_ref().unwrap().base.read(), BASE);
        assert_eq!(ctx.binds.color.len(), 1);
        // Initial value (hover=false) = base.
        assert_eq!(n.style.color, BASE);
    }

    #[test]
    fn flipping_hover_signal_reads_hover_color() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .hover_color(HOVER);
        }
        let id = ctx.node("a").unwrap();
        let hover_sig = ctx.tree.get(id).unwrap().interact.hover.clone().unwrap();
        let slot = ctx.binds.color[0].as_ref().unwrap();
        assert_eq!(slot.bind.read(), BASE);
        hover_sig.set(true);
        assert_eq!(slot.bind.read(), HOVER);
        hover_sig.set(false);
        assert_eq!(slot.bind.read(), BASE);
    }

    #[test]
    fn press_color_alone_uses_press_signal() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .press_color(PRESS);
        }
        let id = ctx.node("a").unwrap();
        let n = ctx.tree.get(id).unwrap();
        assert!(n.interact.pressed.is_some());
        // Hover signal should NOT be allocated if only press_color was called.
        assert!(n.interact.hover.is_none());
        let press_sig = n.interact.pressed.clone().unwrap();
        let slot = ctx.binds.color[0].as_ref().unwrap();
        assert_eq!(slot.bind.read(), BASE);
        press_sig.set(true);
        assert_eq!(slot.bind.read(), PRESS);
    }

    #[test]
    fn hover_then_press_precedence_press_wins() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .hover_color(HOVER)
                .press_color(PRESS);
        }
        let id = ctx.node("a").unwrap();
        let n = ctx.tree.get(id).unwrap();
        // Both signals should be live, exactly one *live* slot for this
        // node (tombstone left in place by the chained rebuild).
        assert!(n.interact.hover.is_some());
        assert!(n.interact.pressed.is_some());
        let live = ctx
            .binds
            .color
            .iter()
            .filter(|s| matches!(s, Some(s) if s.node_id == id))
            .count();
        assert_eq!(live, 1, "chained sugar must leave exactly one live slot");

        let hover_sig = n.interact.hover.clone().unwrap();
        let press_sig = n.interact.pressed.clone().unwrap();
        // Pick the live slot (skip tombstone left by the first sugar call).
        let slot = ctx
            .binds
            .color
            .iter()
            .find_map(|s| s.as_ref().filter(|s| s.node_id == id))
            .unwrap();
        // base: neither set
        assert_eq!(slot.bind.read(), BASE);
        // hover only
        hover_sig.set(true);
        assert_eq!(slot.bind.read(), HOVER);
        // press wins over hover
        press_sig.set(true);
        assert_eq!(slot.bind.read(), PRESS);
        // release press, still hovered
        press_sig.set(false);
        assert_eq!(slot.bind.read(), HOVER);
    }

    // --- subtree removal + bind cleanup ---

    #[test]
    fn remove_subtree_tombstones_color_bind() {
        let s = Signal::new([0.5_f32, 0.5, 0.5, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").color(s.clone());
        }
        assert_eq!(
            ctx.binds.color.iter().filter(|s| s.is_some()).count(),
            1,
            "bind registered"
        );
        let id = ctx.node("a").unwrap();
        let removal = ctx.remove_subtree(id);
        assert_eq!(removal.dropped, vec![id]);
        assert_eq!(removal.dropped_color_slots, vec![0]);
        // Slot is now tombstoned, not removed — vector length unchanged.
        assert_eq!(ctx.binds.color.len(), 1);
        assert!(ctx.binds.color[0].is_none());
    }

    #[test]
    fn remove_subtree_tombstones_position_and_size_binds() {
        let pos = Signal::new([10.0_f32, 20.0]);
        let size = Signal::new([100.0_f32, 50.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").pos(pos.clone()).size_bind(size.clone());
        }
        let id = ctx.node("a").unwrap();
        let removal = ctx.remove_subtree(id);
        assert_eq!(removal.dropped_position_slots, vec![0]);
        assert_eq!(removal.dropped_size_slots, vec![0]);
        assert!(ctx.binds.position[0].is_none());
        assert!(ctx.binds.size[0].is_none());
    }

    #[test]
    fn remove_subtree_cleans_descendant_binds() {
        let pa = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let ca = Signal::new([0.0_f32, 1.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.col("parent").color(pa.clone()).child(|p| {
                p.rect("child").color(ca.clone());
            });
        }
        // Two slots registered.
        assert_eq!(ctx.binds.color.iter().filter(|s| s.is_some()).count(), 2);
        let parent_id = ctx.node("parent").unwrap();
        let removal = ctx.remove_subtree(parent_id);
        // Both slots now tombstoned.
        assert_eq!(removal.dropped.len(), 2);
        assert_eq!(removal.dropped_color_slots.len(), 2);
        assert!(ctx.binds.color.iter().all(|s| s.is_none()));
    }

    #[test]
    fn remove_subtree_prunes_named_ids() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.col("parent").child(|p| {
                p.rect("child");
            });
        }
        assert!(ctx.node("parent").is_some());
        assert!(ctx.node("child").is_some());
        let parent_id = ctx.node("parent").unwrap();
        let _ = ctx.remove_subtree(parent_id);
        assert!(ctx.node("parent").is_none());
        assert!(ctx.node("child").is_none());
    }

    #[test]
    fn tombstone_indices_stay_stable_after_remove() {
        let s1 = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let s2 = Signal::new([0.0_f32, 1.0, 0.0, 1.0]);
        let s3 = Signal::new([0.0_f32, 0.0, 1.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").color(s1.clone());
            scene.rect("b").color(s2.clone());
            scene.rect("c").color(s3.clone());
        }
        // Slots laid out at indices 0, 1, 2.
        let b_id = ctx.node("b").unwrap();
        let _ = ctx.remove_subtree(b_id);
        // Slot at index 1 (b's) is None, others untouched. Index
        // stability is the whole point of tombstoning.
        assert!(ctx.binds.color[0].is_some());
        assert!(ctx.binds.color[1].is_none());
        assert!(ctx.binds.color[2].is_some());
        // c is still at index 2 — drive its signal and assert read.
        s3.set([0.5, 0.5, 0.5, 1.0]);
        let c_slot = ctx.binds.color[2].as_ref().unwrap();
        assert_eq!(c_slot.bind.read(), [0.5, 0.5, 0.5, 1.0]);
    }

    // --- text_field builder ---

    #[test]
    fn text_field_spawns_four_nodes() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.text_field("search", "hi", 14.0);
        }
        // Parent + Selection + Text + Caret children = 4 nodes.
        assert_eq!(ctx.tree.len(), 4);
        let id = ctx.node("search").unwrap();
        let parent = ctx.tree.get(id).unwrap();
        assert_eq!(parent.children.len(), 3);
        assert!(parent.editor.is_some());
        let ed = parent.editor.as_ref().unwrap();
        assert_eq!(ed.value, "hi");
        assert_eq!(ed.cursor, 2);
        assert_eq!(ed.font_size, 14.0);
        // Selection highlight starts hidden.
        assert!(!ctx.tree.get(ed.selection_node).unwrap().visible);
    }

    #[test]
    fn text_field_auto_allocates_focused_signal() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.text_field("box", "", 14.0);
        }
        let id = ctx.node("box").unwrap();
        let n = ctx.tree.get(id).unwrap();
        assert!(n.interact.focused.is_some());
    }

    #[test]
    fn text_field_caret_starts_hidden() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.text_field("box", "", 14.0);
        }
        let id = ctx.node("box").unwrap();
        let ed = ctx.tree.get(id).unwrap().editor.as_ref().unwrap();
        let caret = ctx.tree.get(ed.caret_node).unwrap();
        assert!(!caret.visible);
    }

    #[test]
    fn placeholder_writes_to_editor_state() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.text_field("box", "", 14.0).placeholder("Search…");
        }
        let id = ctx.node("box").unwrap();
        let ed = ctx.tree.get(id).unwrap().editor.as_ref().unwrap();
        assert_eq!(ed.placeholder, "Search…");
    }

    #[test]
    fn on_change_callback_stored() {
        use std::cell::Cell;
        use std::rc::Rc;
        let calls = Rc::new(Cell::new(0_u32));
        let mut ctx = SceneCtx::new();
        {
            let calls = calls.clone();
            let mut scene = Scene::root(&mut ctx);
            scene
                .text_field("box", "", 14.0)
                .on_change(move |_| calls.set(calls.get() + 1));
        }
        let id = ctx.node("box").unwrap();
        let ed = ctx.tree.get(id).unwrap().editor.as_ref().unwrap();
        // Smoke: call the stored fn directly to confirm it's wired.
        if let Some(cb) = ed.on_change.as_ref() {
            cb("hello");
        }
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn editor_apply_threads_through_state() {
        // End-to-end on the editor module's apply fn — verify
        // SceneCtx-built EditorState mutates the same way.
        use crate::editor::{EditOp, apply};
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.text_field("box", "abc", 14.0);
        }
        let id = ctx.node("box").unwrap();
        {
            let n = ctx.tree.get_mut_raw(id).unwrap();
            let ed = n.editor.as_mut().unwrap();
            apply(EditOp::Insert("X".into()), ed);
            assert_eq!(ed.value, "abcX");
            assert_eq!(ed.cursor, 4);
        }
    }

    #[test]
    fn remove_subtree_idempotent_on_stale_id() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a");
        }
        let id = ctx.node("a").unwrap();
        let first = ctx.remove_subtree(id);
        assert_eq!(first.dropped, vec![id]);
        let second = ctx.remove_subtree(id);
        assert!(second.dropped.is_empty());
    }

    #[test]
    fn sugar_reuses_existing_on_hover_signal() {
        let user_sig = Signal::new(false);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .rgba(BASE[0], BASE[1], BASE[2], BASE[3])
                .on_hover(user_sig.clone())
                .hover_color(HOVER);
        }
        // Driving the user's signal must move the sugar bind — proves
        // the sugar reused the user-supplied signal instead of allocating
        // a fresh one (which would never get poked by the input layer).
        let slot = ctx.binds.color[0].as_ref().unwrap();
        assert_eq!(slot.bind.read(), BASE);
        user_sig.set(true);
        assert_eq!(slot.bind.read(), HOVER);
    }
}

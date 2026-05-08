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
use crate::layout::{Align, Axis, Justify, Len, Overflow};
use crate::node::{Node, NodeId, NodeInteract, NodeTree, WindowAction};
use crate::reactive::Bind;
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
        }
    }

    /// Look up a previously-named node.
    pub fn node(&self, name: &str) -> Option<NodeId> {
        self.names.get(name).copied()
    }
}

/// Per-prop reactive bind storage.
#[derive(Default)]
pub struct BindRegistry {
    pub color: Vec<ColorBindSlot>,
    /// Absolute position binds. Each slot drives `layout.abs = Some([x,y])`
    /// — node must already declare `.abs(...)` (or have it set later) to
    /// participate in layout, but a position bind on a flow child still
    /// flips it into absolute mode on first apply.
    pub position: Vec<PositionBindSlot>,
    /// Size binds. Each slot drives `layout.width/height = Px(_)`.
    pub size: Vec<SizeBindSlot>,
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

    pub fn ctx(&self) -> &SceneCtx {
        self.ctx
    }

    pub fn ctx_mut(&mut self) -> &mut SceneCtx {
        self.ctx
    }

    /// Leaf rect (no children expected). Defaults to `Axis::Col` —
    /// matters only if you later nest children under it.
    pub fn rect(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), SpawnKind::Rect, Axis::Col)
    }

    /// Row container: flow children horizontally.
    pub fn row(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), SpawnKind::Rect, Axis::Row)
    }

    /// Column container: flow children vertically (the default).
    pub fn col(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), SpawnKind::Rect, Axis::Col)
    }

    /// Frosted glass rect.
    pub fn glass(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), SpawnKind::Glass, Axis::Col)
    }

    /// Text child. Default size is `Len::Auto` on both axes — the
    /// layout pass resolves them from the shaped bounding box.
    pub fn text(
        &mut self,
        name: impl Into<String>,
        content: impl Into<String>,
        font_size: f32,
    ) -> NodeBuilderRef<'_> {
        self.spawn(
            name.into(),
            SpawnKind::Text(content.into(), font_size),
            Axis::Col,
        )
    }

    /// Image child sourced from a previously-uploaded atlas handle.
    /// Default tint is `[1,1,1,1]` — chain `.color()`/`.rgba()` to tint.
    /// Default size is `Len::Auto`; chain `.size_px(w,h)` to fix size.
    pub fn image(
        &mut self,
        name: impl Into<String>,
        handle: ImageHandle,
    ) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), SpawnKind::Image(handle), Axis::Col)
    }

    fn spawn(
        &mut self,
        name: String,
        kind: SpawnKind,
        axis: Axis,
    ) -> NodeBuilderRef<'_> {
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
        if !name.is_empty() {
            self.ctx.names.insert(name, id);
        }
        NodeBuilderRef {
            ctx: self.ctx,
            id,
        }
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
            self.ctx.binds.position.push(PositionBindSlot {
                node_id: self.id,
                bind,
                last_version: initial_version,
                displayed,
            });
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
            self.ctx.binds.size.push(SizeBindSlot {
                node_id: self.id,
                bind,
                last_version: initial_version,
                displayed,
            });
        }
        self
    }

    pub fn axis(&mut self, a: Axis) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.layout.axis = a;
        }
        self
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
        self.ctx
            .tree
            .with_scrollbar_style(self.id, |s| *s = f(*s));
        self
    }

    // --- style ---

    pub fn color(&mut self, color: impl Into<Bind<[f32; 4]>>) -> &mut Self {
        let bind = color.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.color = initial;
        }
        let is_reactive = !matches!(bind, Bind::Value(_));
        let is_animated = bind.animation().is_some();
        if is_reactive || is_animated {
            let displayed = if is_animated {
                Some(Signal::new(initial))
            } else {
                None
            };
            self.ctx.binds.color.push(ColorBindSlot {
                node_id: self.id,
                bind,
                last_version: initial_version,
                displayed,
            });
        }
        self
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
        }
        self
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
            && let Some(t) = n.text.as_mut() {
                t.line_height = h;
            }
        self
    }

    pub fn on_hover(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.hover = Some(signal));
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
    use crate::reactive::{animated, Computed};
    use crate::Curve;
    use std::time::Duration;

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
        let slot = &ctx.binds.color[0];
        assert!(slot.displayed.is_none());
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn computed_color_registers_bind_with_initial_value() {
        let lit = Signal::new(false);
        let c = Computed::new((lit.clone(),), |(l,)| {
            if l { [0.0, 1.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0, 1.0] }
        });
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size_px(10.0, 10.0).color(c);
        }
        assert_eq!(ctx.binds.color.len(), 1);
        let slot = &ctx.binds.color[0];
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn animated_color_allocates_displayed_signal() {
        let s = Signal::new([0.0_f32, 0.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .size_px(10.0, 10.0)
                .color(animated(s.clone(), Curve::EaseInOut, Duration::from_millis(220)));
        }
        let slot = &ctx.binds.color[0];
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
        let slot = &ctx.binds.position[0];
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
            scene
                .rect("a")
                .size_px(10.0, 10.0)
                .pos(animated(s.clone(), Curve::Linear, Duration::from_millis(100)));
        }
        let slot = &ctx.binds.position[0];
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
        let slot = &ctx.binds.size[0];
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
}

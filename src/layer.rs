//! Retained layer tree — compositor scaffold (P1).
//!
//! Built *beside* [`crate::node::NodeTree`]. A [`Layer`] caches the
//! rasterized pixels of one node subtree so the compositor can
//! recomposite cheaply instead of re-rasterizing the whole scene on
//! every dirty frame.
//!
//! **P1 is scaffold + measurement only — no behavior change.** There is
//! exactly one root layer covering the whole instance stream; no
//! offscreen textures are allocated yet (that lands in P2) and rendering
//! still flows through the monolithic path in `gpu/context.rs`. What P1
//! establishes:
//!   - the two-level **damage model** ([`Damage`]): content-dirty
//!     (re-raster) vs composite-dirty (recomposite-only);
//!   - a per-layer [`Layer::content_epoch`] bumped on content-dirty, so a
//!     consumer can cheaply answer "did this layer re-raster?";
//!   - the `FrameStats` layer / raster / composite / VRAM counters, fed
//!     from [`LayerTree::take_frame_stats`].
//!
//! A single root layer reproduces today's output exactly. The GPU win
//! (composite-only frames for transform/opacity/scroll) lands in P3 once
//! the damage split actually gates rasterization.

use std::collections::HashMap;
use std::ops::Range;

use crate::node::{dirty, NodeId, ScrollSpan};

/// Two-level damage classification derived from a tree dirty mask — the
/// core compositor idea. P3 acts on it to skip rasterization; P1 only
/// records it for the metrics baseline.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Damage {
    /// A layer's pixel content changed (color / text / image / topology
    /// / backdrop) → that layer must re-rasterize its texture.
    pub content: bool,
    /// Only composite-time properties changed (transform / scroll) →
    /// no re-raster; recomposite the cached texture with the new value.
    pub composite: bool,
}

impl Damage {
    /// Classify a [`crate::node::dirty`] bitmask into the two-level
    /// model.
    ///
    /// - `VISUAL | TREE | BACKDROP` ⇒ content changed (re-raster).
    /// - `TRANSFORM | SCROLL`       ⇒ composite-only (recomposite).
    ///
    /// `TRANSFORM` today *also* forces a `compute_layout` re-run; that
    /// stays true in P1. This classification only records that the
    /// *pixels* of a subtree don't change when its position does — the
    /// raster work that fact lets us skip is gated in P3.
    pub fn classify(mask: u32) -> Self {
        const CONTENT: u32 = dirty::VISUAL | dirty::TREE | dirty::BACKDROP;
        const COMPOSITE: u32 = dirty::TRANSFORM | dirty::SCROLL;
        Damage {
            content: mask & CONTENT != 0,
            composite: mask & COMPOSITE != 0,
        }
    }

    /// True when anything changed.
    pub fn any(&self) -> bool {
        self.content || self.composite
    }
}

/// One cached subtree. P1 holds only CPU-side bookkeeping; the
/// `wgpu::Texture` + composite-time transform / opacity / z / effect
/// land in P2+.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Root node of the subtree this layer caches, or `None` for the
    /// implicit scene root (the single root layer spanning all node
    /// roots). Force-promoted layers in P2+ carry `Some(id)`.
    pub root: Option<NodeId>,
    /// Painter-order slice of the flat instance stream owned by this
    /// layer. The single root layer owns the whole stream.
    pub instances: Range<usize>,
    /// Composite z-order; lower paints first.
    pub z: i32,
    /// Composite-time screen offset (physical px). Identity = `[0, 0]`.
    /// P2 keeps it identity; P3 animates it for composite-only motion.
    pub offset: [f32; 2],
    /// Composite-time scale about the layer's top-left. Identity = `1`.
    pub scale: [f32; 2],
    /// Composite-time opacity multiplier. Identity = `1`.
    pub opacity: f32,
    /// Bumped every time this layer's pixels change (content-dirty).
    pub content_epoch: u64,
    /// Physical-pixel size the layer's texture occupies.
    pub size_px: [u32; 2],
    /// `Some` → this is a **scroll layer**: its instances were emitted
    /// content-local into a tall texture; the composite samples a window
    /// of it at the scroll offset, clipped to the viewport. `None` →
    /// full-surface identity / composite-move layer (offset/scale/opacity).
    pub window: Option<ScrollSpan>,
}

/// Persistent composite state for one promoted layer, keyed by its
/// node. Survives [`LayerTree::rebuild`] (which reconstructs the layer
/// list every flush) so an animated offset/opacity isn't reset when the
/// tree re-flattens.
#[derive(Copy, Clone, Debug)]
struct PromotedState {
    offset: [f32; 2],
    scale: [f32; 2],
    opacity: f32,
    content_epoch: u64,
}

impl Default for PromotedState {
    fn default() -> Self {
        PromotedState {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            opacity: 1.0,
            content_epoch: 0,
        }
    }
}

fn root_segment(instances: std::ops::Range<usize>, viewport: [u32; 2], z: i32, epoch: u64) -> Layer {
    Layer {
        root: None,
        instances,
        z,
        offset: [0.0, 0.0],
        scale: [1.0, 1.0],
        opacity: 1.0,
        content_epoch: epoch,
        size_px: viewport,
        window: None,
    }
}

/// One promoted subtree fed to [`LayerTree::rebuild`]: which node, its
/// instance range in the flat stream, and optional scroll-window geometry
/// (`Some` for a scroll layer — content-local instances + a composite
/// window; `None` for a plain `.layer()` promotion).
#[derive(Clone, Debug, PartialEq)]
pub struct PromotedRange {
    pub node: NodeId,
    pub instances: Range<u32>,
    pub scroll: Option<ScrollSpan>,
}

/// Layer tree built beside the node tree. With **no** `.layer()`
/// promotions it is a single root layer (P2 parity). Each promoted
/// subtree splits the painter-order instance stream into root segments
/// + promoted layers, z-ordered by paint order. Composite transform /
/// opacity per promotion persists across re-flattens so animations are
/// composite-only (no re-raster). Auto cost-based promotion is P7.
pub struct LayerTree {
    layers: Vec<Layer>,
    /// Persistent per-promotion composite state (offset/opacity/epoch).
    state: HashMap<NodeId, PromotedState>,
    /// Monotonic epoch for root-segment layers, bumped on any
    /// content-dirty frame.
    root_epoch: u64,
    /// A composite-only change (set_offset/opacity/scale) occurred since
    /// the last frame: the app must redraw + recomposite **without**
    /// re-flattening. Cleared by [`LayerTree::take_composite_dirty`].
    composite_dirty: bool,
}

impl Default for LayerTree {
    fn default() -> Self {
        Self::single_root()
    }
}

impl LayerTree {
    /// One root layer covering the whole scene; no promotions.
    pub fn single_root() -> Self {
        LayerTree {
            layers: vec![root_segment(0..0, [0, 0], 0, 0)],
            state: HashMap::new(),
            root_epoch: 0,
            composite_dirty: false,
        }
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The first (z-lowest) layer. Always present.
    pub fn root_layer(&self) -> &Layer {
        &self.layers[0]
    }

    /// Set a promoted layer's composite offset (physical px). Patches the
    /// live layer in place too, so a composite-only redraw picks it up
    /// without a rebuild. Returns true if it changed (→ composite-dirty).
    pub fn set_offset(&mut self, node: NodeId, offset: [f32; 2]) -> bool {
        let st = self.state.entry(node).or_default();
        if st.offset == offset {
            return false;
        }
        st.offset = offset;
        self.composite_dirty = true;
        for l in &mut self.layers {
            if l.root == Some(node) {
                l.offset = offset;
            }
        }
        true
    }

    /// Set a promoted layer's composite opacity. See [`Self::set_offset`].
    pub fn set_opacity(&mut self, node: NodeId, opacity: f32) -> bool {
        let st = self.state.entry(node).or_default();
        if st.opacity == opacity {
            return false;
        }
        st.opacity = opacity;
        self.composite_dirty = true;
        for l in &mut self.layers {
            if l.root == Some(node) {
                l.opacity = opacity;
            }
        }
        true
    }

    /// Set a promoted layer's composite scale. See [`Self::set_offset`].
    pub fn set_scale(&mut self, node: NodeId, scale: [f32; 2]) -> bool {
        let st = self.state.entry(node).or_default();
        if st.scale == scale {
            return false;
        }
        st.scale = scale;
        self.composite_dirty = true;
        for l in &mut self.layers {
            if l.root == Some(node) {
                l.scale = scale;
            }
        }
        true
    }

    /// Consume the composite-only-change flag (set by the `set_*`
    /// methods). Drives the app's recomposite-without-reflatten path.
    pub fn take_composite_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.composite_dirty, false)
    }

    /// Rebuild the layer list from the promoted subtree instance ranges
    /// (painter order, non-overlapping). Partitions `0..total` into root
    /// segments interleaved with promoted layers, z-ordered by paint
    /// order. Composite state for surviving promotions is preserved;
    /// state for vanished promotions is dropped.
    pub fn rebuild(
        &mut self,
        promoted: &[PromotedRange],
        total: usize,
        viewport: [u32; 2],
        damage: Damage,
    ) {
        if damage.content {
            self.root_epoch += 1;
        }
        // Drop state for promotions that no longer exist.
        let live: std::collections::HashSet<NodeId> = promoted.iter().map(|p| p.node).collect();
        self.state.retain(|k, _| live.contains(k));
        if damage.content {
            for p in promoted {
                self.state.entry(p.node).or_default().content_epoch += 1;
            }
        }

        self.layers.clear();
        let mut z = 0i32;
        let mut cursor = 0usize;
        for p in promoted {
            let r = (p.instances.start as usize)..(p.instances.end as usize);
            // Root content painted before this promotion.
            if r.start > cursor {
                self.layers
                    .push(root_segment(cursor..r.start, viewport, z, self.root_epoch));
                z += 1;
            }
            let st = *self.state.entry(p.node).or_default();
            // A scroll layer's texture is content-sized (viewport-wide ×
            // content-tall); a plain promotion is viewport-sized.
            let size_px = match p.scroll {
                Some(s) => [s.content[0].ceil() as u32, s.content[1].ceil() as u32],
                None => viewport,
            };
            self.layers.push(Layer {
                root: Some(p.node),
                instances: r.clone(),
                z,
                offset: st.offset,
                scale: st.scale,
                opacity: st.opacity,
                content_epoch: st.content_epoch,
                size_px,
                window: p.scroll,
            });
            z += 1;
            cursor = r.end;
        }
        // Trailing root segment (also the sole layer when no promotions).
        if cursor < total || self.layers.is_empty() {
            self.layers
                .push(root_segment(cursor..total, viewport, z, self.root_epoch));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::dirty;

    #[test]
    fn classify_content_vs_composite() {
        assert_eq!(
            Damage::classify(dirty::VISUAL),
            Damage { content: true, composite: false }
        );
        assert_eq!(
            Damage::classify(dirty::BACKDROP),
            Damage { content: true, composite: false }
        );
        assert_eq!(
            Damage::classify(dirty::TREE),
            Damage { content: true, composite: false }
        );
        assert_eq!(
            Damage::classify(dirty::TRANSFORM),
            Damage { content: false, composite: true }
        );
        assert_eq!(
            Damage::classify(dirty::SCROLL),
            Damage { content: false, composite: true }
        );
        // Mixed mask flags both axes.
        assert_eq!(
            Damage::classify(dirty::VISUAL | dirty::SCROLL),
            Damage { content: true, composite: true }
        );
        assert_eq!(Damage::classify(dirty::NONE), Damage::default());
        assert!(!Damage::classify(dirty::NONE).any());
    }

    /// One plain (non-scroll) promoted range — the shape `rebuild`
    /// tests use.
    fn pr(node: NodeId, instances: Range<u32>) -> Vec<PromotedRange> {
        vec![PromotedRange { node, instances, scroll: None }]
    }

    fn node(i: u32) -> NodeId {
        // Test-only NodeId construction via the public SENTINEL shape is
        // not exposed; use distinct ids through add_root in integration.
        // Here we fabricate via transmute-free path: NodeId fields are
        // private, so route through a small tree.
        let mut t = crate::node::NodeTree::new();
        let mut id = NodeId::SENTINEL;
        for _ in 0..=i {
            id = t.add_root(crate::node::Node::rect().build());
        }
        id
    }

    #[test]
    fn no_promotions_is_single_root() {
        let mut lt = LayerTree::single_root();
        lt.rebuild(&[], 42, [800, 600], Damage::classify(dirty::TREE));
        assert_eq!(lt.layers().len(), 1);
        let root = lt.root_layer();
        assert_eq!(root.root, None);
        assert_eq!(root.instances, 0..42);
        assert_eq!(root.size_px, [800, 600]);
    }

    #[test]
    fn mid_stream_promotion_splits_into_three() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        // Promote [30,60) out of a 100-instance stream → before / promoted
        // / after, z-ordered 0,1,2.
        lt.rebuild(&pr(p, 30..60), 100, [10, 10], Damage::classify(dirty::TREE));
        let ls = lt.layers();
        assert_eq!(ls.len(), 3);
        assert_eq!((ls[0].root, ls[0].instances.clone(), ls[0].z), (None, 0..30, 0));
        assert_eq!((ls[1].root, ls[1].instances.clone(), ls[1].z), (Some(p), 30..60, 1));
        assert_eq!((ls[2].root, ls[2].instances.clone(), ls[2].z), (None, 60..100, 2));
    }

    #[test]
    fn promotion_at_head_has_no_leading_segment() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        let ls = lt.layers();
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].root, Some(p));
        assert_eq!(ls[1].root, None);
        assert_eq!(ls[1].instances, 40..100);
    }

    #[test]
    fn promotion_at_tail_has_no_trailing_segment() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 60..100), 100, [10, 10], Damage::classify(dirty::TREE));
        let ls = lt.layers();
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].root, None);
        assert_eq!(ls[0].instances, 0..60);
        assert_eq!(ls[1].root, Some(p));
    }

    #[test]
    fn composite_state_persists_across_rebuild() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        assert!(lt.set_offset(p, [5.0, 7.0]));
        assert!(lt.set_opacity(p, 0.5));
        assert!(lt.take_composite_dirty());
        assert!(!lt.take_composite_dirty()); // consumed
        // A re-flatten (content change) must NOT reset the animated offset.
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::VISUAL));
        let promoted = lt.layers().iter().find(|l| l.root == Some(p)).unwrap();
        assert_eq!(promoted.offset, [5.0, 7.0]);
        assert_eq!(promoted.opacity, 0.5);
    }

    #[test]
    fn set_offset_is_idempotent() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        assert!(lt.set_offset(p, [3.0, 0.0]));
        // Same value → no change, no composite-dirty.
        assert!(!lt.set_offset(p, [3.0, 0.0]));
        assert!(lt.take_composite_dirty());
    }

    #[test]
    fn content_epoch_bumps_only_on_content() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        let e0 = lt.layers().iter().find(|l| l.root == Some(p)).unwrap().content_epoch;
        // Composite-only frame → epoch unchanged.
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TRANSFORM));
        let e1 = lt.layers().iter().find(|l| l.root == Some(p)).unwrap().content_epoch;
        assert_eq!(e0, e1);
        // Content frame → epoch bumps.
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::VISUAL));
        let e2 = lt.layers().iter().find(|l| l.root == Some(p)).unwrap().content_epoch;
        assert_eq!(e2, e1 + 1);
    }

    #[test]
    fn vanished_promotion_state_dropped() {
        let mut lt = LayerTree::single_root();
        let p = node(0);
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        lt.set_offset(p, [9.0, 9.0]);
        // p no longer promoted → its state is pruned; re-promoting starts
        // fresh at identity.
        lt.rebuild(&[], 100, [10, 10], Damage::classify(dirty::TREE));
        lt.rebuild(&pr(p, 0..40), 100, [10, 10], Damage::classify(dirty::TREE));
        let promoted = lt.layers().iter().find(|l| l.root == Some(p)).unwrap();
        assert_eq!(promoted.offset, [0.0, 0.0]);
    }
}

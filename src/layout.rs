//! Custom layout engine.
//!
//! Stage-1 scope: row/col flex containers, Px/Fr/Auto/Pct lengths,
//! padding, gap, justify (main axis), align (cross axis), absolute
//! escape hatch. No wrap, no grid, no min/max constraints.
//!
//! The pass walks the tree once per dirty flush: resolve each node's
//! rect from its parent's content box, distribute flex children,
//! recurse. Intrinsic (Auto) main size for a container is
//! `sum(children_main) + gap*(n-1)`; for text it's `measurer.measure(...)`.
//! For a plain rect Auto reduces to 0 unless children expand it.
//!
//! Absolute children are placed relative to the parent's content-box
//! origin and do not influence flow-child sizing.

use crate::node::{NodeId, NodeTree};

/// Length primitive. Values are **logical pixels** — `compute_layout`
/// multiplies them by the display scale factor on the way out, so
/// the same `Len::Px(100.0)` produces a 100×100 on-screen footprint
/// on a 100% 1080p monitor and 200×200 physical pixels on a 200% 4K
/// monitor.
#[derive(Copy, Clone, Debug, PartialEq)]
#[derive(Default)]
pub enum Len {
    /// Fixed length in px.
    Px(f32),
    /// Flex fraction of remaining main-axis space.
    Fr(f32),
    /// Intrinsic size — shaped text width for [`NodeText`], 0 for
    /// rects with no children, or sum-of-children along main axis
    /// for containers.
    #[default]
    Auto,
    /// Percent of the parent's content-box size on the same axis
    /// (0..1).
    Pct(f32),
    /// Alias for `Fr(1.0)`. Takes all remaining space on the main
    /// axis, or stretches to full content box on the cross axis when
    /// the parent's cross `align` is `Stretch`.
    Fill,
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Axis {
    Row,
    #[default]
    Col,
}

/// Main-axis distribution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Justify {
    #[default]
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

/// Cross-axis alignment of each child within the line.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Start,
    End,
    Center,
    Stretch,
}

/// How a container handles flow children that exceed its content box on
/// a given axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Children paint past container bounds (current behavior).
    #[default]
    Visible,
    /// Children are clipped to the container; scroll offset is applied.
    /// Wheel input over the container updates the offset.
    Scroll,
    /// Clip but no scroll input.
    Hidden,
}

impl Overflow {
    pub fn clips(self) -> bool {
        !matches!(self, Overflow::Visible)
    }
    pub fn scrolls(self) -> bool {
        matches!(self, Overflow::Scroll)
    }
}

#[derive(Clone, Debug)]
pub struct LayoutStyle {
    pub axis: Axis,
    pub width: Len,
    pub height: Len,
    /// Inner padding `[left, top, right, bottom]`.
    pub padding: [f32; 4],
    /// Spacing between flow children along the main axis.
    pub gap: f32,
    pub justify: Justify,
    pub align: Align,
    /// Width-to-height ratio. When `Some`, the layout pass overrides
    /// height = width / ratio after the child's width is resolved by
    /// the parent flex pass. Width remains the driver — pair with
    /// `w_px`, `w_pct`, `w(Fill)`, etc.; any explicit `h_*` is ignored
    /// on this node. `aspect_ratio = 1.0` produces a square; `16.0/9.0`
    /// a video tile; `2.0` a wide banner.
    pub aspect_ratio: Option<f32>,
    /// When `Some`, the node is absolutely positioned at `[x, y]`
    /// relative to its parent's content-box origin and is skipped by
    /// flow layout. Root nodes treat `Some` as viewport-relative and
    /// `None` as origin `(0, 0)`.
    pub abs: Option<[f32; 2]>,
    /// Per-axis overflow behavior. Default Visible on both axes.
    pub overflow_x: Overflow,
    pub overflow_y: Overflow,
    /// When true on a flow child, absorbs all remaining slack on the
    /// main axis *before* this child — equivalent to CSS
    /// `margin-left: auto` on the first such child. Subsequent
    /// children pack normally after it. Has no effect on absolute
    /// children, on the first flow child (no slack to absorb), or
    /// when the parent's `justify` is anything other than
    /// `Justify::Start` (other justify variants already distribute
    /// slack, so push_end would double-shift).
    pub push_end: bool,
    /// Tab-focus order. `0` (default) means the node is **excluded**
    /// from keyboard focus cycling. A non-zero value opts the node into
    /// the Tab order; nodes are visited in ascending `focus_order`,
    /// ties broken by document (creation) order. Mirrors HTML
    /// `tabindex` except that `0` here means "skip" rather than
    /// "natural order". See [`crate::App`] Tab / Shift+Tab handling.
    pub focus_order: u32,
}

impl Default for LayoutStyle {
    fn default() -> Self {
        Self {
            axis: Axis::Col,
            width: Len::Auto,
            height: Len::Auto,
            padding: [0.0; 4],
            gap: 0.0,
            justify: Justify::Start,
            align: Align::Start,
            aspect_ratio: None,
            abs: None,
            overflow_x: Overflow::Visible,
            overflow_y: Overflow::Visible,
            push_end: false,
            focus_order: 0,
        }
    }
}

impl LayoutStyle {
    pub fn clips(&self) -> bool {
        self.overflow_x.clips() || self.overflow_y.clips()
    }
    pub fn scrolls(&self) -> bool {
        self.overflow_x.scrolls() || self.overflow_y.scrolls()
    }
}

/// Trait implemented by whatever text system the app uses. The layout
/// pass calls it to resolve intrinsic text size for `Auto` text
/// widths/heights.
pub trait Measurer {
    /// Return `[width, height]` in px for the given shaped text.
    fn measure_text(&mut self, content: &str, font_size: f32, line_height: f32) -> [f32; 2];

    /// Measure `content` word-wrapped at `max_width` (physical px) —
    /// the multi-line paragraph form: width of the longest wrapped
    /// line, height covering every line. Default impl falls back to
    /// the truncating [`Self::measure_text_constrained`].
    fn measure_text_wrapped(
        &mut self,
        content: &str,
        font_size: f32,
        line_height: f32,
        max_width: f32,
    ) -> [f32; 2] {
        self.measure_text_constrained(content, font_size, line_height, max_width)
    }

    /// Measure `content` constrained to `max_width` (physical px). When
    /// the unconstrained measurement would exceed the constraint, the
    /// returned dimensions describe the truncated `prefix + "…"` form.
    /// Default impl falls back to [`Self::measure_text`] — sufficient
    /// for measurers without a truncation pass (e.g. `NullMeasurer`).
    fn measure_text_constrained(
        &mut self,
        content: &str,
        font_size: f32,
        line_height: f32,
        _max_width: f32,
    ) -> [f32; 2] {
        self.measure_text(content, font_size, line_height)
    }
}

/// A text node's measurement inputs, snapshotted (content + physical-px
/// font metrics + the width constraint and whether it wraps into a
/// paragraph or truncates to one line at that width). Built once per
/// node per measure site so the constraint semantics live in one place.
pub(crate) struct TextProbe {
    content: String,
    size: f32,
    lh: f32,
    max_w: Option<f32>,
    wrap: bool,
    /// The node's width is parent-derived (Fill/Pct/Fr) rather than sized
    /// to its own content (Auto) — so a wrapped paragraph without an
    /// explicit cap flows to the available width. Auto-width text keeps
    /// its content-driven intrinsic sizing.
    fill_width: bool,
}

impl TextProbe {
    fn from_node(t: &crate::node::NodeText, width: &Len, scale: f32) -> Self {
        Self {
            content: t.content.clone(),
            size: t.font_size * scale,
            lh: t.line_height * scale,
            max_w: t.max_width.map(|w| w * scale),
            wrap: t.wrap,
            fill_width: matches!(width, Len::Fill | Len::Pct(_) | Len::Fr(_)),
        }
    }

    /// Measure the text. `wrap_avail` is the layout-available render width,
    /// supplied only when the caller measures along the block (height)
    /// axis — where the cross-available value *is* where the paragraph
    /// will wrap. A wrapped, fill-width node with no explicit cap flows to
    /// that width, so `.w(Fill).wrap()` fills its box and re-wraps as the
    /// container resizes. An explicit `max_width_px` (any width) keeps its
    /// fixed wrap width; Auto-width text keeps its content-driven size.
    fn measure<M: Measurer>(&self, measurer: &mut M, wrap_avail: Option<f32>) -> [f32; 2] {
        // Only fill-width nodes wrap to the available width; Auto-width
        // text sizes to content (its own intrinsic/max cap).
        let avail = wrap_avail.filter(|_| self.fill_width);
        let wrap_w = if self.wrap {
            match (self.max_w, avail) {
                // Explicit cap wins and is width-agnostic (unchanged).
                (Some(mx), _) => Some(mx),
                // Fill-width, no cap → flow to the available width (new).
                (None, Some(av)) => Some(av),
                // No cap and no fill width → nothing to wrap at.
                (None, None) => None,
            }
        } else {
            self.max_w
        };
        match (wrap_w, self.wrap) {
            (Some(w), true) => measurer.measure_text_wrapped(&self.content, self.size, self.lh, w),
            (Some(w), false) => {
                measurer.measure_text_constrained(&self.content, self.size, self.lh, w)
            }
            (None, _) => measurer.measure_text(&self.content, self.size, self.lh),
        }
    }
}

/// Null measurer — use when the tree contains no text nodes (e.g.
/// unit tests) or when you want explicit sizes everywhere.
pub struct NullMeasurer;

impl Measurer for NullMeasurer {
    fn measure_text(&mut self, content: &str, font_size: f32, _line_height: f32) -> [f32; 2] {
        // Coarse fallback: assumes 0.6 advance ratio. Not pretty, but
        // deterministic for tests.
        [content.chars().count() as f32 * font_size * 0.6, font_size]
    }
}

/// Walk every root in the tree and resolve `Node.rect` absolutely.
/// Viewport is the root constraint (in physical pixels); roots
/// without an explicit `abs` anchor at `(0, 0)`. `scale` is applied
/// to every logical-pixel input (`Len::Px`, padding, gap, `abs`
/// offsets, text font size) so the resolved rects come out in
/// physical pixels.
pub fn compute_layout<M: Measurer>(
    tree: &mut NodeTree,
    viewport: [f32; 2],
    measurer: &mut M,
    scale: f32,
) {
    let roots: Vec<NodeId> = tree.roots().to_vec();
    for root in roots {
        layout_root(tree, root, viewport, measurer, scale);
    }
}

fn layout_root<M: Measurer>(
    tree: &mut NodeTree,
    id: NodeId,
    viewport: [f32; 2],
    measurer: &mut M,
    scale: f32,
) {
    let Some(node) = tree.get(id) else { return };
    if !node.visible {
        return;
    }
    let style = node.layout.clone();
    let kind_text = node
        .text
        .as_ref()
        .map(|t| TextProbe::from_node(t, &node.layout.width, scale));

    let w = resolve_main_size(
        &style.width, viewport[0], measurer, tree, id, &kind_text, true, scale,
    );
    let h = resolve_main_size(
        &style.height, viewport[1], measurer, tree, id, &kind_text, false, scale,
    );
    let origin = style.abs.map(|a| [a[0] * scale, a[1] * scale]).unwrap_or([0.0, 0.0]);

    if let Some(n) = tree.get_mut_raw(id) {
        n.rect = [origin[0], origin[1], w, h];
    }
    layout_children(tree, id, measurer, scale);
}

fn layout_children<M: Measurer>(
    tree: &mut NodeTree,
    parent: NodeId,
    measurer: &mut M,
    scale: f32,
) {
    let Some(p) = tree.get(parent) else { return };
    let parent_rect = p.rect;
    let style = p.layout.clone();
    let children = p.children.clone();

    let pad_l = style.padding[0] * scale;
    let pad_t = style.padding[1] * scale;
    let pad_r = style.padding[2] * scale;
    let pad_b = style.padding[3] * scale;
    let gap = style.gap * scale;
    let content_x = parent_rect[0] + pad_l;
    let content_y = parent_rect[1] + pad_t;
    let content_w = (parent_rect[2] - pad_l - pad_r).max(0.0);
    let content_h = (parent_rect[3] - pad_t - pad_b).max(0.0);

    let (main_size, cross_size) = match style.axis {
        Axis::Row => (content_w, content_h),
        Axis::Col => (content_h, content_w),
    };

    // Split abs and flow children.
    let mut flow: Vec<NodeId> = Vec::new();
    let mut abs: Vec<NodeId> = Vec::new();
    for c in &children {
        let n = match tree.get(*c) {
            Some(n) if n.visible => n,
            _ => continue,
        };
        if n.layout.abs.is_some() {
            abs.push(*c);
        } else {
            flow.push(*c);
        }
    }

    // Resolve flow children main sizes.
    let gap_total = if flow.len() > 1 {
        gap * (flow.len() - 1) as f32
    } else {
        0.0
    };
    let mut child_main: Vec<f32> = Vec::with_capacity(flow.len());
    let mut fr_total: f32 = 0.0;
    let mut fixed_main: f32 = 0.0;
    for &c in &flow {
        let (len, cross_len, aspect, text) = {
            let n = tree.get(c).expect("flow child");
            let (len, cross_len) = match style.axis {
                Axis::Row => (n.layout.width, n.layout.height),
                Axis::Col => (n.layout.height, n.layout.width),
            };
            let text = n.text.as_ref().map(|t| TextProbe::from_node(t, &n.layout.width, scale));
            (len, cross_len, n.layout.aspect_ratio, text)
        };
        // In a Col parent an aspect-ratio child's height derives from its
        // width, which resolves on the cross axis independently of the
        // main-axis flex distribution — so it's knowable now. Resolve it
        // here so justify slack, Fr shares, and content_size all see the
        // real height; deferring to the placement override alone would
        // centre/space children by a stale (usually zero) main size.
        // (Row parents need nothing here: aspect drives their *cross*
        // axis, and placement already recomputes it.)
        if matches!(style.axis, Axis::Col)
            && let Some(ratio) = aspect
        {
            let w = resolve_cross_size(
                tree,
                c,
                &cross_len,
                cross_size,
                style.align,
                &text,
                measurer,
                style.axis,
                scale,
            );
            let v = w / ratio.max(f32::EPSILON);
            fixed_main += v;
            child_main.push(v);
            continue;
        }
        match len {
            Len::Fill => {
                fr_total += 1.0;
                child_main.push(f32::NEG_INFINITY); // resolved in second pass
            }
            Len::Fr(f) => {
                let f = f.max(0.0);
                fr_total += f;
                // Encode Fr share as negative to distinguish from fixed.
                // Use `-f - 1.0` so Fr(0) still flags as unresolved.
                child_main.push(-(f + 1.0));
            }
            Len::Px(p) => {
                let v = p * scale;
                fixed_main += v;
                child_main.push(v);
            }
            Len::Pct(p) => {
                let v = p * main_size;
                fixed_main += v;
                child_main.push(v);
            }
            Len::Auto => {
                let v = intrinsic_main(tree, c, style.axis, cross_size, &text, measurer, scale);
                fixed_main += v;
                child_main.push(v);
            }
        }
    }
    let remaining = (main_size - fixed_main - gap_total).max(0.0);
    if fr_total > 0.0 {
        for slot in child_main.iter_mut() {
            if *slot == f32::NEG_INFINITY {
                *slot = remaining * (1.0 / fr_total);
            } else if *slot < 0.0 {
                let fr = -*slot - 1.0;
                *slot = remaining * (fr / fr_total);
            }
        }
    } else {
        // No Fr/Fill: clamp unresolved to 0 just in case.
        for slot in child_main.iter_mut() {
            if *slot < 0.0 {
                *slot = 0.0;
            }
        }
    }
    let sum_main: f32 = child_main.iter().copied().sum::<f32>() + gap_total;

    // Justify offsets.
    let n = flow.len() as f32;
    let slack = (main_size - sum_main).max(0.0);
    let (leading, between_extra) = match style.justify {
        Justify::Start => (0.0, 0.0),
        Justify::End => (slack, 0.0),
        Justify::Center => (slack * 0.5, 0.0),
        Justify::SpaceBetween if n > 1.0 => (0.0, slack / (n - 1.0)),
        Justify::SpaceBetween => (0.0, 0.0),
        Justify::SpaceAround if n > 0.0 => {
            let unit = slack / n;
            (unit * 0.5, unit)
        }
        Justify::SpaceAround => (0.0, 0.0),
        Justify::SpaceEvenly if n > 0.0 => {
            let unit = slack / (n + 1.0);
            (unit, unit)
        }
        Justify::SpaceEvenly => (0.0, 0.0),
    };

    let (main_start, cross_start) = match style.axis {
        Axis::Row => (content_x, content_y),
        Axis::Col => (content_y, content_x),
    };

    // Track bounding extent for content_size (scroll math).
    let mut max_x: f32 = content_x;
    let mut max_y: f32 = content_y;

    // push_end resolution: find the first flow child (after index 0)
    // with push_end=true. When the parent justify is Start, inject
    // `slack` into the cursor immediately before that child — CSS
    // `margin-left: auto`. Other justify variants already distribute
    // slack; bail to avoid double-shift.
    let push_end_index = if matches!(style.justify, Justify::Start) {
        flow.iter().enumerate().skip(1).find_map(|(i, c)| {
            let n = tree.get(*c)?;
            if n.layout.push_end { Some(i) } else { None }
        })
    } else {
        None
    };

    let mut cursor = main_start + leading;
    for (i, &c) in flow.iter().enumerate() {
        if Some(i) == push_end_index {
            cursor += slack;
        }
        let m = child_main[i];
        // Cross-axis size.
        let (cross_len, text) = {
            let n = tree.get(c).expect("flow child");
            let len = match style.axis {
                Axis::Row => n.layout.height,
                Axis::Col => n.layout.width,
            };
            let text = n.text.as_ref().map(|t| TextProbe::from_node(t, &n.layout.width, scale));
            (len, text)
        };
        let cross = resolve_cross_size(
            tree,
            c,
            &cross_len,
            cross_size,
            style.align,
            &text,
            measurer,
            style.axis,
            scale,
        );
        // Align offset.
        let align_offset = match style.align {
            Align::Start | Align::Stretch => 0.0,
            Align::End => (cross_size - cross).max(0.0),
            Align::Center => ((cross_size - cross) * 0.5).max(0.0),
        };
        let (x, y, w, mut h) = match style.axis {
            Axis::Row => (cursor, cross_start + align_offset, m, cross),
            Axis::Col => (cross_start + align_offset, cursor, cross, m),
        };
        // Aspect-ratio constraint: width drives height. Pulled from the
        // child's layout so a `.square()` / `.aspect_ratio(16.0/9.0)`
        // call on the child takes effect *after* the parent's flex pass
        // resolved its width. Width remains whatever flex computed
        // (Fill / Px / Pct / Auto); height is rewritten and any
        // explicit `h_*` on the child is discarded for this node.
        let aspect = tree.get(c).and_then(|n| n.layout.aspect_ratio);
        if let Some(ratio) = aspect {
            h = w / ratio.max(f32::EPSILON);
        }
        if let Some(n) = tree.get_mut_raw(c) {
            n.rect = [x, y, w, h];
        }
        max_x = max_x.max(x + w);
        max_y = max_y.max(y + h);
        layout_children(tree, c, measurer, scale);
        // Advance by the *actual* main-axis size, which for a Col
        // parent is the (possibly aspect-ratio-overridden) height.
        // Without this, siblings stack as if the override never
        // happened — they sit at `cursor += original_m` and visually
        // overlap the resized child below.
        let main_consumed = match style.axis {
            Axis::Row => w,
            Axis::Col => h,
        };
        cursor += main_consumed + gap + between_extra;
    }

    // Absolute children: resolve against content box.
    for c in abs {
        let (width_len, height_len, abs_off, text) = {
            let n = tree.get(c).expect("abs child");
            let off = n.layout.abs.unwrap_or([0.0, 0.0]);
            let text = n.text.as_ref().map(|t| TextProbe::from_node(t, &n.layout.width, scale));
            (n.layout.width, n.layout.height, off, text)
        };
        let w = match width_len {
            Len::Px(p) => p * scale,
            Len::Pct(p) => p * content_w,
            Len::Fr(_) | Len::Fill => content_w,
            Len::Auto => intrinsic_main(tree, c, Axis::Row, content_h, &text, measurer, scale),
        };
        // Aspect ratio drives height from the resolved width for abs
        // children too, mirroring the flow path above — without this an
        // `.abs(...).w(Fill).aspect_ratio(r)` node silently lays out at
        // height 0 (Auto intrinsic of abs-only content).
        let aspect = tree.get(c).and_then(|n| n.layout.aspect_ratio);
        let h = if let Some(ratio) = aspect {
            w / ratio.max(f32::EPSILON)
        } else {
            match height_len {
                Len::Px(p) => p * scale,
                Len::Pct(p) => p * content_h,
                Len::Fr(_) | Len::Fill => content_h,
                Len::Auto => intrinsic_main(tree, c, Axis::Col, content_w, &text, measurer, scale),
            }
        };
        let x = content_x + abs_off[0] * scale;
        let y = content_y + abs_off[1] * scale;
        if let Some(n) = tree.get_mut_raw(c) {
            n.rect = [x, y, w, h];
        }
        max_x = max_x.max(x + w);
        max_y = max_y.max(y + h);
        layout_children(tree, c, measurer, scale);
    }

    // Persist content_size relative to parent rect (includes right/bottom
    // padding so a scrolled-to-end view shows the trailing pad).
    let content_extent_w = (max_x - parent_rect[0]) + pad_r;
    let content_extent_h = (max_y - parent_rect[1]) + pad_b;
    if let Some(p) = tree.get_mut_raw(parent) {
        let mut cs = [content_extent_w, content_extent_h];
        // A lazy list only materializes a small window of rows, so the
        // measured child extent is NOT the scrollable height. Override the
        // main axis with the full virtual height (item_count * item_height)
        // so the scrollbar + bounds reflect the whole list, not the window.
        // The container's own vertical padding stays part of the extent —
        // rows sit inside the content box, so a declared bottom inset must
        // let the last row scroll clear of the edge (the raw override used
        // to drop it, leaving the final row flush against the clip).
        if let Some(ll) = p.lazy_list.as_ref() {
            let pad_v = (p.layout.padding[1] + p.layout.padding[3]) * scale;
            cs[1] = ll.total_height_logical() * scale + pad_v;
        }
        p.content_size = cs;
        // Re-clamp scroll target if not in overscroll mode and content
        // shrank below current target.
        if let Some(s) = p.scroll.as_mut()
            && !s.overscroll
        {
            let max_off_x = (cs[0] - parent_rect[2]).max(0.0);
            let max_off_y = (cs[1] - parent_rect[3]).max(0.0);
            s.target[0] = s.target[0].clamp(0.0, max_off_x);
            s.target[1] = s.target[1].clamp(0.0, max_off_y);
            s.current[0] = s.current[0].clamp(0.0, max_off_x);
            s.current[1] = s.current[1].clamp(0.0, max_off_y);
        }
    }
}

/// Resolve the main-axis size of a single node against a parent-axis
/// extent. Used both for roots (against viewport) and for flow
/// children when the parent is laid out.
fn resolve_main_size<M: Measurer>(
    len: &Len,
    parent_extent: f32,
    measurer: &mut M,
    tree: &NodeTree,
    id: NodeId,
    text: &Option<TextProbe>,
    is_width_axis: bool,
    scale: f32,
) -> f32 {
    match len {
        Len::Px(p) => *p * scale,
        Len::Pct(p) => p * parent_extent,
        Len::Fr(_) | Len::Fill => parent_extent,
        Len::Auto => {
            // Use the node's own declared axis for intrinsic — a col
            // container's Auto width sums children widths (cross
            // axis = Row), and vice versa.
            let axis = tree.get(id).map(|n| n.layout.axis).unwrap_or(Axis::Col);
            let probe_axis = if is_width_axis { Axis::Row } else { Axis::Col };
            intrinsic_main_from_node(
                tree, id, axis, probe_axis, parent_extent, text, measurer, scale,
            )
        }
    }
}

/// Intrinsic size along `probe_axis`. Called recursively.
fn intrinsic_main<M: Measurer>(
    tree: &NodeTree,
    id: NodeId,
    probe_axis: Axis,
    cross_avail: f32,
    text: &Option<TextProbe>,
    measurer: &mut M,
    scale: f32,
) -> f32 {
    let Some(n) = tree.get(id) else { return 0.0 };
    if !n.visible {
        return 0.0;
    }
    if let Some(probe) = text {
        // Measuring height (Col): the cross-available value is the render
        // width, so a wrapped paragraph flows to it. Measuring width (Row):
        // no known render width yet — keep the intrinsic/max-capped shape.
        let wrap_avail = matches!(probe_axis, Axis::Col).then_some(cross_avail);
        let m = probe.measure(measurer, wrap_avail);
        return match probe_axis {
            Axis::Row => m[0],
            Axis::Col => m[1],
        };
    }
    // Container intrinsic.
    intrinsic_main_from_node(
        tree, id, n.layout.axis, probe_axis, cross_avail, text, measurer, scale,
    )
}

fn intrinsic_main_from_node<M: Measurer>(
    tree: &NodeTree,
    id: NodeId,
    self_axis: Axis,
    probe_axis: Axis,
    cross_avail: f32,
    self_text: &Option<TextProbe>,
    measurer: &mut M,
    scale: f32,
) -> f32 {
    let Some(n) = tree.get(id) else { return 0.0 };
    if let Some(probe) = self_text {
        let wrap_avail = matches!(probe_axis, Axis::Col).then_some(cross_avail);
        let m = probe.measure(measurer, wrap_avail);
        return match probe_axis {
            Axis::Row => m[0],
            Axis::Col => m[1],
        };
    }
    let pad_main = match probe_axis {
        Axis::Row => (n.layout.padding[0] + n.layout.padding[2]) * scale,
        Axis::Col => (n.layout.padding[1] + n.layout.padding[3]) * scale,
    };
    if n.children.is_empty() {
        return pad_main;
    }
    // What the children can actually span on the non-probe axis: the
    // incoming avail minus this node's own padding there. Without this,
    // a padded Auto container over-measures Pct/aspect children by its
    // padding (e.g. a padded col with a `w: Fill` square child reserves
    // the *unpadded* width as height — a phantom gap below the child).
    let pad_cross = match probe_axis {
        Axis::Row => (n.layout.padding[1] + n.layout.padding[3]) * scale,
        Axis::Col => (n.layout.padding[0] + n.layout.padding[2]) * scale,
    };
    let child_avail = (cross_avail - pad_cross).max(0.0);
    // Collect flow children intrinsic sizes along probe_axis.
    let flow: Vec<NodeId> = n
        .children
        .iter()
        .filter_map(|c| {
            let cn = tree.get(*c)?;
            if !cn.visible || cn.layout.abs.is_some() {
                None
            } else {
                Some(*c)
            }
        })
        .collect();
    if flow.is_empty() {
        return pad_main;
    }
    let gap_total = if flow.len() > 1 {
        n.layout.gap * scale * (flow.len() - 1) as f32
    } else {
        0.0
    };
    // If self axis == probe axis: sum children main sizes.
    // If they differ: max over children cross sizes.
    if self_axis == probe_axis {
        let mut sum = 0.0f32;
        for c in &flow {
            sum += intrinsic_child_size(tree, *c, probe_axis, child_avail, measurer, scale);
        }
        sum + gap_total + pad_main
    } else {
        let mut mx = 0.0f32;
        for c in &flow {
            let v = intrinsic_child_size(tree, *c, probe_axis, child_avail, measurer, scale);
            if v > mx {
                mx = v;
            }
        }
        mx + pad_main
    }
}

/// One flow child's intrinsic contribution along `probe_axis`, for
/// [`intrinsic_main`]. Honours `aspect_ratio`: an aspect child's height
/// derives from its resolved width (the height `Len` is ignored by
/// layout for such nodes), so probing the vertical axis measures the
/// width and divides — without this, a `w: Fill` aspect child would
/// contribute 0 to an Auto-height parent.
fn intrinsic_child_size<M: Measurer>(
    tree: &NodeTree,
    c: NodeId,
    probe_axis: Axis,
    cross_avail: f32,
    measurer: &mut M,
    scale: f32,
) -> f32 {
    let Some(cn) = tree.get(c) else { return 0.0 };
    let text = cn.text.as_ref().map(|t| TextProbe::from_node(t, &cn.layout.width, scale));
    if matches!(probe_axis, Axis::Col)
        && let Some(ratio) = cn.layout.aspect_ratio
    {
        let w = match cn.layout.width {
            Len::Px(p) => p * scale,
            Len::Pct(p) => p * cross_avail,
            Len::Fr(_) | Len::Fill => cross_avail,
            Len::Auto => intrinsic_main(tree, c, Axis::Row, cross_avail, &text, measurer, scale),
        };
        return w / ratio.max(f32::EPSILON);
    }
    let len = match probe_axis {
        Axis::Row => cn.layout.width,
        Axis::Col => cn.layout.height,
    };
    match len {
        Len::Px(p) => p * scale,
        Len::Pct(p) => p * cross_avail,
        Len::Fr(_) | Len::Fill => 0.0,
        Len::Auto => intrinsic_main(tree, c, probe_axis, cross_avail, &text, measurer, scale),
    }
}

/// Cross-axis size for a flow child. Honors explicit Px/Pct/Fr, respects
/// align=Stretch as a fallback for Auto/Fr, and measures text for Auto.
fn resolve_cross_size<M: Measurer>(
    tree: &NodeTree,
    id: NodeId,
    len: &Len,
    cross_avail: f32,
    align: Align,
    text: &Option<TextProbe>,
    measurer: &mut M,
    parent_axis: Axis,
    scale: f32,
) -> f32 {
    match len {
        Len::Px(p) => *p * scale,
        Len::Pct(p) => p * cross_avail,
        Len::Fr(_) | Len::Fill => cross_avail,
        Len::Auto => {
            if align == Align::Stretch {
                cross_avail
            } else {
                // Intrinsic along the cross axis.
                let cross_axis = match parent_axis {
                    Axis::Row => Axis::Col,
                    Axis::Col => Axis::Row,
                };
                intrinsic_main(tree, id, cross_axis, cross_avail, text, measurer, scale)
            }
        }
    }
}

// Silence `len` unused-binding in the Fill/Fr branch.
#[allow(dead_code)]
fn _suppress() {
    let _ = Len::Fr(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    /// Deterministic wrapping measurer: every char is `size*0.5` wide, a
    /// paragraph breaks into `ceil(natural / max_width)` lines.
    struct WrapMeasurer;
    impl Measurer for WrapMeasurer {
        fn measure_text(&mut self, content: &str, font_size: f32, line_height: f32) -> [f32; 2] {
            [content.chars().count() as f32 * font_size * 0.5, line_height]
        }
        fn measure_text_wrapped(
            &mut self,
            content: &str,
            font_size: f32,
            line_height: f32,
            max_width: f32,
        ) -> [f32; 2] {
            let natural = content.chars().count() as f32 * font_size * 0.5;
            let lines = (natural / max_width).ceil().max(1.0);
            [natural.min(max_width), lines * line_height]
        }
    }

    /// A `.w(Fill).wrap()` paragraph with no explicit cap flows to the
    /// container width: narrower box ⇒ more lines ⇒ taller, and the text
    /// box spans the full available width. This is the dynamic-wrap path
    /// (fixed-width panes would leave the old fixed `max_width` stranded).
    #[test]
    fn fill_width_wrap_flows_to_container() {
        let build = |container_w: f32| {
            let mut tree = NodeTree::new();
            // Auto-height col so the root height reflects the wrapped text.
            let root = tree.add_root(
                Node::rect()
                    .layout_axis(Axis::Col)
                    .layout_width(Len::Px(container_w))
                    .layout_height(Len::Auto)
                    .build(),
            );
            let mut text = Node::text("aaaa aaaa aaaa aaaa", 10.0)
                .layout_width(Len::Fill)
                .build();
            text.text.as_mut().unwrap().wrap = true;
            let tid = tree.add_child(root, text);
            compute_layout(&mut tree, [1000.0, 1000.0], &mut WrapMeasurer, 1.0);
            let r = tree.get(tid).unwrap().rect;
            (r[2], r[3]) // (width, height)
        };
        // natural width = 19 chars * 10 * 0.5 = 95.
        let (wide_w, wide_h) = build(100.0); // fits on 1 line
        let (narrow_w, narrow_h) = build(40.0); // wraps to 3 lines
        // Text box spans the container width in both cases.
        assert_eq!(wide_w, 100.0);
        assert_eq!(narrow_w, 40.0);
        // Narrower container ⇒ more wrapped lines ⇒ taller.
        assert!(narrow_h > wide_h, "narrow {narrow_h} should exceed wide {wide_h}");
    }

    #[test]
    fn root_fills_viewport() {
        let mut tree = NodeTree::new();
        tree.add_root(
            Node::rect()
                .layout_width(Len::Fill)
                .layout_height(Len::Fill)
                .build(),
        );
        compute_layout(&mut tree, [800.0, 600.0], &mut NullMeasurer, 1.0);
        let r = tree.get(tree.roots()[0]).unwrap().rect;
        assert_eq!(r, [0.0, 0.0, 800.0, 600.0]);
    }

    #[test]
    fn row_distributes_fr_remaining_space() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(300.0))
                .layout_height(Len::Px(100.0))
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Px(100.0))
                .layout_height(Len::Fill)
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Fr(1.0))
                .layout_height(Len::Fill)
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Fr(3.0))
                .layout_height(Len::Fill)
                .build(),
        );
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let kids: Vec<_> = tree
            .get(root)
            .unwrap()
            .children
            .iter()
            .map(|c| tree.get(*c).unwrap().rect)
            .collect();
        assert_eq!(kids[0], [0.0, 0.0, 100.0, 100.0]);
        assert_eq!(kids[1], [100.0, 0.0, 50.0, 100.0]);
        assert_eq!(kids[2], [150.0, 0.0, 150.0, 100.0]);
    }

    #[test]
    fn col_with_gap_and_padding() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Col)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(200.0))
                .layout_padding([10.0; 4])
                .layout_gap(8.0)
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Fill)
                .layout_height(Len::Px(50.0))
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Fill)
                .layout_height(Len::Px(50.0))
                .build(),
        );
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let kids: Vec<_> = tree
            .get(root)
            .unwrap()
            .children
            .iter()
            .map(|c| tree.get(*c).unwrap().rect)
            .collect();
        assert_eq!(kids[0], [10.0, 10.0, 180.0, 50.0]);
        assert_eq!(kids[1], [10.0, 68.0, 180.0, 50.0]);
    }

    #[test]
    fn absolute_positioning_escapes_flow() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Col)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(200.0))
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Fill)
                .layout_height(Len::Px(30.0))
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_abs(50.0, 60.0)
                .layout_width(Len::Px(40.0))
                .layout_height(Len::Px(40.0))
                .build(),
        );
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let kids: Vec<_> = tree
            .get(root)
            .unwrap()
            .children
            .iter()
            .map(|c| tree.get(*c).unwrap().rect)
            .collect();
        assert_eq!(kids[0], [0.0, 0.0, 200.0, 30.0]);
        assert_eq!(kids[1], [50.0, 60.0, 40.0, 40.0]);
    }

    #[test]
    fn push_end_absorbs_slack_before_marked_child() {
        // Row with 3 children of 20px each in a 200px container.
        // Middle child has push_end → slack (200 - 60 = 140) before it.
        // Expected x positions: 0, 160, 180.
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(40.0))
                .build(),
        );
        for i in 0..3 {
            let mut b = Node::rect()
                .layout_width(Len::Px(20.0))
                .layout_height(Len::Px(40.0));
            if i == 1 { b = b.push_end(); }
            tree.add_child(root, b.build());
        }
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let kids = tree.get(root).unwrap().children.clone();
        assert_eq!(tree.get(kids[0]).unwrap().rect[0], 0.0);
        assert_eq!(tree.get(kids[1]).unwrap().rect[0], 160.0);
        assert_eq!(tree.get(kids[2]).unwrap().rect[0], 180.0);
    }

    #[test]
    fn push_end_noop_on_first_child() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(40.0))
                .build(),
        );
        let c0 = tree.add_child(root, Node::rect().layout_width(Len::Px(20.0)).layout_height(Len::Px(40.0)).push_end().build());
        let c1 = tree.add_child(root, Node::rect().layout_width(Len::Px(20.0)).layout_height(Len::Px(40.0)).build());
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        assert_eq!(tree.get(c0).unwrap().rect[0], 0.0);
        assert_eq!(tree.get(c1).unwrap().rect[0], 20.0);
    }

    #[test]
    fn push_end_ignored_when_justify_is_not_start() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(40.0))
                .layout_justify(Justify::Center)
                .build(),
        );
        let c0 = tree.add_child(root, Node::rect().layout_width(Len::Px(20.0)).layout_height(Len::Px(40.0)).build());
        let c1 = tree.add_child(root, Node::rect().layout_width(Len::Px(20.0)).layout_height(Len::Px(40.0)).push_end().build());
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        // Center: slack 160, leading 80, children at 80 and 100.
        assert_eq!(tree.get(c0).unwrap().rect[0], 80.0);
        assert_eq!(tree.get(c1).unwrap().rect[0], 100.0);
    }

    #[test]
    fn center_shortcut_centers_text_child_both_axes() {
        // NullMeasurer returns [chars * font_size * 0.6, font_size]
        // for text → "ABC" at 10px = [18, 10]. Parent 100x40, expect
        // text at x=(100-18)/2=41, y=(40-10)/2=15.
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(100.0))
                .layout_height(Len::Px(40.0))
                .center()
                .build(),
        );
        let child = tree.add_child(root, Node::text("ABC", 10.0).build());
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let r = tree.get(child).unwrap().rect;
        assert_eq!(r[0], 41.0);
        assert_eq!(r[1], 15.0);
    }

    #[test]
    fn justify_center_and_align_center() {
        let mut tree = NodeTree::new();
        let root = tree.add_root(
            Node::rect()
                .layout_axis(Axis::Row)
                .layout_width(Len::Px(200.0))
                .layout_height(Len::Px(100.0))
                .layout_justify(Justify::Center)
                .layout_align(Align::Center)
                .build(),
        );
        tree.add_child(
            root,
            Node::rect()
                .layout_width(Len::Px(60.0))
                .layout_height(Len::Px(40.0))
                .build(),
        );
        compute_layout(&mut tree, [1000.0, 1000.0], &mut NullMeasurer, 1.0);
        let c = tree.get(tree.get(root).unwrap().children[0]).unwrap().rect;
        // x: (200-60)/2 = 70, y: (100-40)/2 = 30
        assert_eq!(c, [70.0, 30.0, 60.0, 40.0]);
    }
}

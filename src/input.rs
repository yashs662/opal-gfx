//! Reusable pointer-input bookkeeping.
//!
//! `InputState` owns the cursor/hover/capture/focus state and knows how
//! to sync it into the `Signal<bool>` slots on each interactive node.
//! It does **not** touch winit directly — callers translate their event
//! source into the four entry points (`on_cursor_moved`, `on_cursor_left`,
//! `on_left_pressed`, `on_left_released`) and pass the hit-test cache +
//! node tree alongside.
//!
//! The hit-test cache must be produced by `NodeTree::flatten` (topmost
//! first) and rebuilt whenever `TRANSFORM` or `TREE` dirty bits fire.

use crate::node::{HitEntry, NodeId, NodeTree, ScrollAxis, ScrollHit, ScrollbarHit};
use winit::keyboard::KeyCode;

/// Transient result returned by each event method so the caller can
/// decide whether to re-flush the tree + request a redraw.
#[derive(Default, Debug, Clone, Copy)]
pub struct InputChange {
    pub hovered_changed: bool,
    pub pressed_changed: bool,
    pub focused_changed: bool,
}

impl InputChange {
    pub fn any(&self) -> bool {
        self.hovered_changed || self.pressed_changed || self.focused_changed
    }
}

#[derive(Default, Debug, Clone)]
pub struct InputState {
    /// Last known cursor position in physical pixels.
    pub cursor: Option<[f32; 2]>,
    /// Topmost interactive node under the cursor right now. While
    /// `captured` is set, this is pinned to the captured node regardless
    /// of where the cursor actually is (matches native button feel).
    pub hovered: Option<NodeId>,
    /// Node that received a press and owns pointer capture until release.
    pub captured: Option<NodeId>,
    /// Most recently focused node (last clicked one that wanted focus).
    pub focused: Option<NodeId>,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk the hit cache top-down and return the first interactive node
    /// whose AABB contains the point. O(n) over interactive nodes only —
    /// `hits` is already filtered during flatten.
    pub fn hit_test(hits: &[HitEntry], x: f32, y: f32) -> Option<NodeId> {
        hits.iter().find(|h| h.contains(x, y)).map(|h| h.node_id)
    }

    /// Process a cursor move. Updates `hovered` (or keeps it pinned when
    /// captured), and while captured also updates `pressed` signals so
    /// dragging off the captured node visually un-presses it.
    pub fn on_cursor_moved(
        &mut self,
        x: f32,
        y: f32,
        hits: &[HitEntry],
        tree: &NodeTree,
    ) -> InputChange {
        self.cursor = Some([x, y]);

        let new_hover = if self.captured.is_some() {
            self.captured
        } else {
            Self::hit_test(hits, x, y)
        };

        let mut change = InputChange::default();
        if new_hover != self.hovered {
            self.hovered = new_hover;
            change.hovered_changed = sync_bool_signals(hits, tree, self.hovered, |n| {
                &n.interact.hover
            });
        }

        if let Some(cap) = self.captured {
            let over = hits
                .iter()
                .find(|h| h.node_id == cap)
                .map(|h| h.contains(x, y))
                .unwrap_or(false);
            let pressed_target = if over { Some(cap) } else { None };
            change.pressed_changed =
                sync_bool_signals(hits, tree, pressed_target, |n| &n.interact.pressed);
        }

        change
    }

    /// Clear hover on cursor leave.
    pub fn on_cursor_left(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        self.cursor = None;
        let mut change = InputChange::default();
        if self.hovered.is_some() {
            self.hovered = None;
            change.hovered_changed =
                sync_bool_signals(hits, tree, None, |n| &n.interact.hover);
        }
        change
    }

    /// Left-button press: capture currently hovered, set pressed + focused.
    pub fn on_left_pressed(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        let target = self.hovered;
        self.captured = target;
        let mut change = InputChange::default();
        change.pressed_changed =
            sync_bool_signals(hits, tree, target, |n| &n.interact.pressed);
        if self.focused != target {
            self.focused = target;
            change.focused_changed =
                sync_bool_signals(hits, tree, target, |n| &n.interact.focused);
        }
        change
    }

    /// Left-button release: clear pressed state and re-evaluate hover at
    /// the current cursor position.
    pub fn on_left_released(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        self.captured = None;
        let mut change = InputChange::default();
        change.pressed_changed = sync_bool_signals(hits, tree, None, |n| &n.interact.pressed);
        if let Some([x, y]) = self.cursor {
            let new_hover = Self::hit_test(hits, x, y);
            if new_hover != self.hovered {
                self.hovered = new_hover;
                change.hovered_changed =
                    sync_bool_signals(hits, tree, self.hovered, |n| &n.interact.hover);
            }
        }
        change
    }
}

/// Outcome of a left-press against the scrollbar layer. Built to be
/// dispatched by `app.rs` so the App can either hand the press to the
/// regular hit-test path (`Miss`) or own the drag bookkeeping
/// (`StartDrag`). `JumpedToPosition` already mutated the tree.
#[derive(Debug, Clone)]
pub enum ScrollbarPress {
    /// Cursor wasn't over any visible bar — proceed with normal hit-test.
    Miss,
    /// Cursor was on a thumb. Caller should latch a drag and update
    /// scroll on subsequent moves via [`drag_to`].
    StartDrag {
        node_id: NodeId,
        axis: ScrollAxis,
        pointer_origin: f32,
        scroll_origin: f32,
        track_travel: f32,
        max_offset: f32,
    },
    /// Cursor was on a track but off the thumb — the tree was already
    /// retargeted to the clicked position via spring smooth-scroll.
    JumpedToPosition,
}

/// Test a left-press against the visible scrollbar set. Topmost bar
/// wins (last in `bars` since flatten emits in DFS order — innermost
/// containers come last). Returns [`ScrollbarPress::StartDrag`] when
/// the press lands on a thumb, [`ScrollbarPress::JumpedToPosition`]
/// when it lands elsewhere on a track (spring chases to that pos), or
/// [`ScrollbarPress::Miss`] otherwise.
pub fn press_scrollbar(
    cursor: [f32; 2],
    bars: &[ScrollbarHit],
    tree: &mut NodeTree,
) -> ScrollbarPress {
    let [x, y] = cursor;
    // Reverse to get topmost (last DFS push) first.
    let Some(bar) = bars
        .iter()
        .rev()
        .find(|b| rect_contains(b.clip_rect, x, y) && rect_contains(b.track, x, y))
    else {
        return ScrollbarPress::Miss;
    };
    let i = bar.axis.index();
    let pointer_axis = if i == 0 { x } else { y };
    if rect_contains(bar.thumb, x, y) {
        let scroll_origin = tree
            .get(bar.node_id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.current[i])
            .unwrap_or(0.0);
        // Mark active so the thumb paints in the active color and
        // bar_alpha is held high.
        let mut active = tree
            .get(bar.node_id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.bar_active)
            .unwrap_or([false; 2]);
        active[i] = true;
        tree.set_bar_active(bar.node_id, active);
        return ScrollbarPress::StartDrag {
            node_id: bar.node_id,
            axis: bar.axis,
            pointer_origin: pointer_axis,
            scroll_origin,
            track_travel: bar.track_travel,
            max_offset: bar.max_offset,
        };
    }
    // Track click off the thumb: jump scroll target to the cursor's
    // fractional position. Spring chases — this gives the smooth-scroll
    // feel asked for.
    if bar.track_travel > 0.0 {
        let track_min = if i == 0 { bar.track[0] } else { bar.track[1] };
        let thumb_len = if i == 0 {
            bar.thumb[2] - bar.thumb[0]
        } else {
            bar.thumb[3] - bar.thumb[1]
        };
        // Centre the thumb on the click — `pointer - track_min - thumb/2`
        // gives the target thumb-top, then convert to scroll offset.
        let raw = pointer_axis - track_min - thumb_len * 0.5;
        let frac = (raw / bar.track_travel).clamp(0.0, 1.0);
        let target_off = frac * bar.max_offset;
        let mut target = tree
            .get(bar.node_id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.target)
            .unwrap_or([0.0; 2]);
        target[i] = target_off;
        tree.set_scroll_target(bar.node_id, target);
    }
    ScrollbarPress::JumpedToPosition
}

/// Continue an in-flight thumb drag. Maps the cursor delta along the
/// drag axis to a scroll offset using the cached track travel + max
/// offset, then writes through `set_scroll_immediate` (no spring) so
/// the thumb stays glued to the pointer. Returns true if the scroll
/// position changed.
pub fn drag_to(
    cursor: [f32; 2],
    node_id: NodeId,
    axis: ScrollAxis,
    pointer_origin: f32,
    scroll_origin: f32,
    track_travel: f32,
    max_offset: f32,
    tree: &mut NodeTree,
) -> bool {
    if track_travel <= 0.0 || max_offset <= 0.0 {
        return false;
    }
    let i = axis.index();
    let cursor_axis = if i == 0 { cursor[0] } else { cursor[1] };
    let cursor_delta = cursor_axis - pointer_origin;
    let scroll_delta = cursor_delta * (max_offset / track_travel);
    let new_pos = scroll_origin + scroll_delta;
    let prev = tree
        .get(node_id)
        .and_then(|n| n.scroll.as_ref())
        .map(|s| s.current[i])
        .unwrap_or(0.0);
    tree.set_scroll_immediate(node_id, axis, new_pos);
    let after = tree
        .get(node_id)
        .and_then(|n| n.scroll.as_ref())
        .map(|s| s.current[i])
        .unwrap_or(prev);
    (after - prev).abs() > f32::EPSILON
}

/// Clear the active-axis flag at the end of a drag and retarget so the
/// spring lands on a snap multiple (and bounces back from any rubber-
/// band overscroll). The retarget reads `current` and writes `target`
/// via `set_scroll_target`, which hard-clamps + snaps internally —
/// `current` is left alone so the bounce-back stiffness in
/// `tick_scrolls` engages when current is past the edge.
pub fn end_drag(node_id: NodeId, axis: ScrollAxis, tree: &mut NodeTree) {
    let mut active = tree
        .get(node_id)
        .and_then(|n| n.scroll.as_ref())
        .map(|s| s.bar_active)
        .unwrap_or([false; 2]);
    active[axis.index()] = false;
    tree.set_bar_active(node_id, active);
    let current = tree
        .get(node_id)
        .and_then(|n| n.scroll.as_ref())
        .map(|s| s.current)
        .unwrap_or([0.0; 2]);
    tree.set_scroll_target(node_id, current);
}

/// AABB containment for a `[min_x, min_y, max_x, max_y]` rect.
fn rect_contains(r: [f32; 4], x: f32, y: f32) -> bool {
    x >= r[0] && x < r[2] && y >= r[1] && y < r[3]
}

/// Walk every visible scrollbar AABB and update each scrollable node's
/// per-axis hover flag to match whether the cursor sits in that bar's
/// track. Returns true if any flag flipped — caller should re-flush so
/// the bar can re-color and `bar_alpha` pop. Bars whose containing
/// `clip_rect` excludes the cursor are skipped (they're not actually
/// visible).
///
/// Multiple bars may be reported per node (X + Y) but each axis is
/// tracked independently. Bars not in `bars` (e.g. the node has no
/// scroll on that axis or content fits) leave the hover flag at false
/// because the per-frame `next_hover[node][axis]` defaults to false.
pub fn update_scrollbar_hover(
    cursor: Option<[f32; 2]>,
    bars: &[ScrollbarHit],
    tree: &mut NodeTree,
) -> bool {
    use std::collections::HashMap;
    let mut next: HashMap<NodeId, [bool; 2]> = HashMap::new();
    if let Some([x, y]) = cursor {
        for b in bars {
            if !rect_contains(b.clip_rect, x, y) {
                continue;
            }
            if rect_contains(b.track, x, y) {
                let entry = next.entry(b.node_id).or_insert([false; 2]);
                entry[b.axis.index()] = true;
            }
        }
    }
    let mut changed = false;
    // Apply: anything in `next` gets that flag pair; anything in
    // `bars` but absent from `next` falls back to [false, false] for
    // its node — but only on the axes that bar represents.
    let mut touched: HashMap<NodeId, [bool; 2]> = HashMap::new();
    for b in bars {
        let cur = next.get(&b.node_id).copied().unwrap_or([false; 2]);
        let entry = touched.entry(b.node_id).or_insert([false; 2]);
        entry[b.axis.index()] = cur[b.axis.index()];
    }
    for (id, hover) in touched {
        if tree.set_bar_hover(id, hover) {
            changed = true;
        }
    }
    changed
}

/// Route a wheel delta to the topmost scroll container under the
/// cursor, then bubble any unconsumed remainder up its scroll-ancestor
/// chain. Caller passes deltas in **pixels** (already scaled), with
/// the convention `delta_y` *positive = scroll content downward* (the
/// content moves up under the viewport — matches CSS scrollTop / wheel
/// forward).
///
/// Returns true if any container's target moved. Callers should
/// `react()` / request redraw on true.
///
/// `shift` swaps axes — a vertical wheel becomes horizontal scroll.
/// Standard browser/native pattern.
pub fn on_wheel(
    cursor: [f32; 2],
    mut delta: [f32; 2],
    scroll_hits: &[ScrollHit],
    tree: &mut NodeTree,
    shift: bool,
) -> bool {
    if shift {
        delta = [delta[0] + delta[1], 0.0];
    }
    if delta[0].abs() < f32::EPSILON && delta[1].abs() < f32::EPSILON {
        return false;
    }
    // Topmost = innermost = last DFS push that contains the cursor.
    let Some(hit) = scroll_hits
        .iter()
        .rev()
        .find(|h| h.contains(cursor[0], cursor[1]))
    else {
        return false;
    };
    let mut request = delta;
    let mut moved = false;
    for &id in &hit.ancestor_chain {
        let applied = tree.add_scroll_delta(id, request);
        if applied != [0.0; 2] {
            moved = true;
        }
        request = [request[0] - applied[0], request[1] - applied[1]];
        if request[0].abs() < f32::EPSILON && request[1].abs() < f32::EPSILON {
            break;
        }
    }
    moved
}

/// Default arrow-key step size in **logical** px. Caller multiplies by
/// the current display scale to convert to physical-px deltas.
pub const ARROW_KEY_STEP_LOGICAL: f32 = 40.0;

/// Hold-to-scroll velocity in **logical** px/sec. Drives the per-tick
/// continuous pump that runs while a scroll arrow is physically held.
/// 600 logical px/sec ≈ 13–14 rows/sec on a 44-px-stride list — fast
/// enough that holding feels purposeful, slow enough that a quick tap
/// doesn't overshoot. Replaces OS auto-repeat for arrow keys
/// specifically because OS repeat has a ~250 ms initial-delay gap that
/// makes the spring visibly settle at one position then jump to the
/// next when the first repeat arrives.
pub const HOLD_SCROLL_VELOCITY_LOGICAL: f32 = 600.0;

/// Resolve a keyboard event to a scroll delta + (optionally) an
/// absolute target, route it through the same wheel-bubble path as
/// [`on_wheel`]. Returns true if any scroll target moved.
///
/// Routing prefers the topmost scroll container under the cursor; falls
/// back to the first entry in `tree.scrollables()` so keyboard
/// navigation works even with the pointer parked elsewhere. Handles:
/// - `ArrowUp/Down/Left/Right`: 1 line (`ARROW_KEY_STEP_LOGICAL` * scale)
/// - `PageUp/PageDown`: viewport height − 1 line
/// - `Home/End`: jump to 0 / `max_off` (bypasses incremental delta — uses
///   `set_scroll_target` so overscroll mode hard-clamps).
///
/// Unhandled keys return false unchanged. Caller (e.g. `App::window_event`)
/// only invokes this for recognised navigation keys; the user's `on_key`
/// hook still runs afterwards via the regular dispatch chain.
pub fn on_scroll_key(
    code: KeyCode,
    cursor: Option<[f32; 2]>,
    viewport: [f32; 2],
    scale: f32,
    scroll_hits: &[ScrollHit],
    tree: &mut NodeTree,
) -> bool {
    let line = ARROW_KEY_STEP_LOGICAL * scale;
    let page_y = (viewport[1] - line).max(line);
    let (delta, jump): (Option<[f32; 2]>, Option<JumpEdge>) = match code {
        KeyCode::ArrowUp => (Some([0.0, -line]), None),
        KeyCode::ArrowDown => (Some([0.0, line]), None),
        KeyCode::ArrowLeft => (Some([-line, 0.0]), None),
        KeyCode::ArrowRight => (Some([line, 0.0]), None),
        KeyCode::PageUp => (Some([0.0, -page_y]), None),
        KeyCode::PageDown => (Some([0.0, page_y]), None),
        KeyCode::Home => (None, Some(JumpEdge::Start)),
        KeyCode::End => (None, Some(JumpEdge::End)),
        _ => return false,
    };
    // Resolve the target chain: cursor-over → its ancestor_chain;
    // else → first scrollable.
    let chain: Vec<NodeId> = match cursor
        .and_then(|[x, y]| scroll_hits.iter().rev().find(|h| h.contains(x, y)))
    {
        Some(hit) => hit.ancestor_chain.clone(),
        None => match tree.scrollables().first().copied() {
            Some(id) => vec![id],
            None => return false,
        },
    };
    if let Some(d) = delta {
        let mut request = d;
        let mut moved = false;
        for id in &chain {
            let applied = tree.add_scroll_delta(*id, request);
            if applied != [0.0; 2] {
                moved = true;
            }
            request = [request[0] - applied[0], request[1] - applied[1]];
            if request[0].abs() < f32::EPSILON && request[1].abs() < f32::EPSILON {
                break;
            }
        }
        return moved;
    }
    if let Some(edge) = jump {
        let id = chain[0];
        let (rect, content, sx, sy) = match tree.get(id) {
            Some(n) => (
                n.rect,
                n.content_size,
                n.layout.overflow_x.scrolls(),
                n.layout.overflow_y.scrolls(),
            ),
            None => return false,
        };
        let max_off = [
            (content[0] - rect[2]).max(0.0),
            (content[1] - rect[3]).max(0.0),
        ];
        // Convention: Home / End drive the dominant scroll axis. If only
        // one axis scrolls, target that. If both, prefer Y (lists are
        // the common case). Caller can wire bespoke navigation in
        // `on_key` for grid-style scrollers.
        let (target_x, target_y) = match (sx, sy) {
            (true, false) => (
                if matches!(edge, JumpEdge::End) { max_off[0] } else { 0.0 },
                0.0,
            ),
            (_, true) => (
                0.0,
                if matches!(edge, JumpEdge::End) { max_off[1] } else { 0.0 },
            ),
            (false, false) => return false,
        };
        tree.set_scroll_target(id, [target_x, target_y]);
        return true;
    }
    false
}

#[derive(Copy, Clone)]
enum JumpEdge {
    Start,
    End,
}

/// Per-tick continuous scroll pump for held arrow keys. Routes a small
/// `HOLD_SCROLL_VELOCITY_LOGICAL * scale * dt` delta to the scroll
/// container under the cursor (or the first scrollable on fallback) on
/// every frame an arrow is in `held`. Bypasses snap-on-input so the
/// view scrolls smoothly across rows; settle-on-quiesce snaps to a
/// row boundary once the user releases.
///
/// Returns true if any scroll target moved (caller should `react()`).
/// Replaces OS auto-repeat for the arrow keys — the OS repeat path has
/// a ~250 ms initial-delay gap that makes the spring visibly settle at
/// one position then jump to the next.
pub fn pump_held_scroll(
    held: &std::collections::HashSet<KeyCode>,
    cursor: Option<[f32; 2]>,
    scroll_hits: &[ScrollHit],
    tree: &mut NodeTree,
    scale: f32,
    dt: f32,
) -> bool {
    if held.is_empty() || dt <= 0.0 {
        return false;
    }
    let step = HOLD_SCROLL_VELOCITY_LOGICAL * scale * dt;
    let mut delta = [0.0f32; 2];
    if held.contains(&KeyCode::ArrowUp) {
        delta[1] -= step;
    }
    if held.contains(&KeyCode::ArrowDown) {
        delta[1] += step;
    }
    if held.contains(&KeyCode::ArrowLeft) {
        delta[0] -= step;
    }
    if held.contains(&KeyCode::ArrowRight) {
        delta[0] += step;
    }
    if delta[0].abs() < f32::EPSILON && delta[1].abs() < f32::EPSILON {
        return false;
    }
    let chain: Vec<NodeId> = match cursor
        .and_then(|[x, y]| scroll_hits.iter().rev().find(|h| h.contains(x, y)))
    {
        Some(hit) => hit.ancestor_chain.clone(),
        None => match tree.scrollables().first().copied() {
            Some(id) => vec![id],
            None => return false,
        },
    };
    let mut request = delta;
    let mut moved = false;
    for id in &chain {
        let applied = tree.add_scroll_delta_continuous(*id, request);
        if applied != [0.0; 2] {
            moved = true;
        }
        request = [request[0] - applied[0], request[1] - applied[1]];
        if request[0].abs() < f32::EPSILON && request[1].abs() < f32::EPSILON {
            break;
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, ScrollHit};

    fn scrollable(t: &mut NodeTree, x: f32, y: f32, w: f32, h: f32, content: [f32; 2],
                  axis: char) -> NodeId {
        let b = match axis {
            'y' => Node::rect().scroll_y(),
            'x' => Node::rect().scroll_x(),
            _ => Node::rect().scroll(),
        };
        let id = t.add_root(b.build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [x, y, w, h];
            n.content_size = content;
        }
        id
    }

    #[test]
    fn wheel_bubbles_remainder_to_outer() {
        let mut t = NodeTree::new();
        let outer = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1000.0], 'y');
        let inner = scrollable(&mut t, 10.0, 10.0, 180.0, 100.0, [180.0, 150.0], 'y');
        // Inner already past its edge — wheel up over inner should
        // forward the unconsumed delta to outer.
        let _ = t.add_scroll_delta(inner, [0.0, 50.0]); // inner now at 50/50
        let scroll_hits = vec![
            ScrollHit {
                node_id: outer,
                bounds: [0.0, 0.0, 200.0, 200.0],
                clip_rect: crate::gpu::NO_CLIP,
                ancestor_chain: vec![outer],
            },
            ScrollHit {
                node_id: inner,
                bounds: [10.0, 10.0, 190.0, 110.0],
                clip_rect: crate::gpu::NO_CLIP,
                ancestor_chain: vec![inner, outer],
            },
        ];
        let moved = on_wheel([100.0, 60.0], [0.0, 100.0], &scroll_hits, &mut t, false);
        assert!(moved);
        // Inner stays clamped at its max (50). Outer absorbs the
        // remaining 100 of the requested 100 (since inner had 0 budget
        // left — it was already at edge).
        let outer_off = t.get(outer).unwrap().scroll.unwrap().target;
        assert!((outer_off[1] - 100.0).abs() < 0.01, "outer={outer_off:?}");
    }

    #[test]
    fn shift_swaps_wheel_axis() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 100.0, [800.0, 100.0], 'x');
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 200.0, 100.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        // Vertical wheel + shift → horizontal scroll.
        let moved = on_wheel([100.0, 50.0], [0.0, 80.0], &scroll_hits, &mut t, true);
        assert!(moved);
        let off = t.get(id).unwrap().scroll.unwrap().target;
        assert_eq!(off[1], 0.0, "y must remain 0 after shift swap");
        assert!((off[0] - 80.0).abs() < 0.01, "x should absorb the 80");
    }

    #[test]
    fn wheel_outside_any_scroll_hit_is_noop() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 500.0], 'y');
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 100.0, 100.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        let moved = on_wheel([500.0, 500.0], [0.0, 50.0], &scroll_hits, &mut t, false);
        assert!(!moved);
    }

    fn make_y_bar(node_id: NodeId, current_frac: f32) -> ScrollbarHit {
        // Track 200 px tall, thumb 50 px → travel 150 px.
        let track = [190.0, 0.0, 200.0, 200.0];
        let thumb_y = current_frac * 150.0;
        let thumb = [190.0, thumb_y, 200.0, thumb_y + 50.0];
        ScrollbarHit {
            node_id,
            axis: ScrollAxis::Y,
            track,
            thumb,
            clip_rect: crate::gpu::NO_CLIP,
            max_offset: 800.0,
            track_travel: 150.0,
        }
    }

    #[test]
    fn hover_in_track_pops_alpha() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 500.0], 'y');
        let bars = vec![make_y_bar(id, 0.0)];
        let changed = update_scrollbar_hover(Some([195.0, 100.0]), &bars, &mut t);
        assert!(changed);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_hover, [false, true]);
        assert_eq!(s.bar_alpha, 1.0);
    }

    #[test]
    fn hover_outside_clears_flag() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 500.0], 'y');
        // Pre-hover so the change has somewhere to fall from.
        t.set_bar_hover(id, [false, true]);
        let bars = vec![make_y_bar(id, 0.0)];
        let changed = update_scrollbar_hover(Some([10.0, 10.0]), &bars, &mut t);
        assert!(changed);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_hover, [false, false]);
    }

    #[test]
    fn press_on_thumb_starts_drag() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 900.0], 'y');
        let bars = vec![make_y_bar(id, 0.0)];
        let outcome = press_scrollbar([195.0, 25.0], &bars, &mut t);
        match outcome {
            ScrollbarPress::StartDrag {
                node_id,
                axis,
                track_travel,
                max_offset,
                ..
            } => {
                assert_eq!(node_id, id);
                assert_eq!(axis, ScrollAxis::Y);
                assert_eq!(track_travel, 150.0);
                assert_eq!(max_offset, 800.0);
            }
            other => panic!("expected StartDrag, got {other:?}"),
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_active, [false, true]);
    }

    #[test]
    fn press_on_track_off_thumb_jumps_target() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 900.0], 'y');
        let bars = vec![make_y_bar(id, 0.0)];
        // Click near the bottom of the track — far below the thumb at top.
        let outcome = press_scrollbar([195.0, 195.0], &bars, &mut t);
        assert!(matches!(outcome, ScrollbarPress::JumpedToPosition));
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(s.target[1] > 400.0, "target should jump near bottom: {:?}", s.target);
        assert_eq!(s.current[1], 0.0, "current shouldn't snap — spring chases");
    }

    #[test]
    fn drag_maps_pointer_delta_to_scroll() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 900.0], 'y');
        // Drag started at y=25 with scroll_origin=0; track 150 px maps
        // to 800 max_off → moving cursor to y=100 (delta 75) should set
        // current to 75 * (800/150) = 400.
        let moved = drag_to(
            [195.0, 100.0],
            id,
            ScrollAxis::Y,
            25.0,
            0.0,
            150.0,
            800.0,
            &mut t,
        );
        assert!(moved);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!((s.current[1] - 400.0).abs() < 0.01, "current={:?}", s.current);
        assert_eq!(s.target[1], s.current[1], "drag keeps target glued to current");
    }

    #[test]
    fn arrow_down_pushes_target_via_cursor_scroll() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1000.0], 'y');
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 200.0, 200.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        let moved = on_scroll_key(
            KeyCode::ArrowDown,
            Some([100.0, 100.0]),
            [200.0, 200.0],
            1.0,
            &scroll_hits,
            &mut t,
        );
        assert!(moved);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!((s.target[1] - 40.0).abs() < 0.01, "expected 40, got {}", s.target[1]);
    }

    #[test]
    fn arrow_falls_back_to_first_scrollable_without_cursor() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1000.0], 'y');
        let moved = on_scroll_key(
            KeyCode::ArrowDown,
            None,
            [200.0, 200.0],
            1.0,
            &[],
            &mut t,
        );
        assert!(moved);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert!(s.target[1] > 0.0);
    }

    #[test]
    fn page_down_jumps_by_viewport_minus_line() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 5000.0], 'y');
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 200.0, 200.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        on_scroll_key(
            KeyCode::PageDown,
            Some([100.0, 100.0]),
            [200.0, 200.0],
            1.0,
            &scroll_hits,
            &mut t,
        );
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 200.0 - 40.0);
    }

    #[test]
    fn end_jumps_to_max_offset() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1500.0], 'y');
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 200.0, 200.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        on_scroll_key(
            KeyCode::End,
            Some([100.0, 100.0]),
            [200.0, 200.0],
            1.0,
            &scroll_hits,
            &mut t,
        );
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 1300.0); // 1500 - 200
    }

    #[test]
    fn home_returns_to_zero() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1500.0], 'y');
        let _ = t.add_scroll_delta(id, [0.0, 500.0]);
        let scroll_hits = vec![ScrollHit {
            node_id: id,
            bounds: [0.0, 0.0, 200.0, 200.0],
            clip_rect: crate::gpu::NO_CLIP,
            ancestor_chain: vec![id],
        }];
        on_scroll_key(
            KeyCode::Home,
            Some([100.0, 100.0]),
            [200.0, 200.0],
            1.0,
            &scroll_hits,
            &mut t,
        );
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.target[1], 0.0);
    }

    #[test]
    fn unhandled_key_returns_false() {
        let mut t = NodeTree::new();
        let _id = scrollable(&mut t, 0.0, 0.0, 200.0, 200.0, [200.0, 1500.0], 'y');
        let moved = on_scroll_key(
            KeyCode::KeyA,
            None,
            [200.0, 200.0],
            1.0,
            &[],
            &mut t,
        );
        assert!(!moved);
    }

    #[test]
    fn end_drag_clears_active_flag() {
        let mut t = NodeTree::new();
        let id = scrollable(&mut t, 0.0, 0.0, 100.0, 100.0, [100.0, 900.0], 'y');
        t.set_bar_active(id, [false, true]);
        end_drag(id, ScrollAxis::Y, &mut t);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_active, [false, false]);
    }
}

/// Iterate the hit cache once and write `target == node_id` into the
/// signal returned by `select` for each interactive node. Returns true
/// if any signal flipped. `Signal::set` is a no-op write if unchanged.
fn sync_bool_signals(
    hits: &[HitEntry],
    tree: &NodeTree,
    target: Option<NodeId>,
    select: impl Fn(&crate::node::Node) -> &Option<crate::signal::Signal<bool>>,
) -> bool {
    let mut changed = false;
    for entry in hits {
        if let Some(n) = tree.get(entry.node_id)
            && let Some(sig) = select(n).as_ref() {
                let on = Some(entry.node_id) == target;
                if sig.set(on) {
                    changed = true;
                }
            }
    }
    changed
}

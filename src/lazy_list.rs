//! Virtualized list.
//!
//! [`LazyListState`] lives on a node that acts as the scroll container
//! for a large virtual list. Only the rows currently in (or near) the
//! visible viewport are materialized as actual tree children; the rest
//! exist only as a row count + per-row height.
//!
//! Performance contract: time + memory per frame are O(visible rows),
//! not O(item_count) — except when a row's height is mutated, which
//! recomputes the prefix-sum cache from the changed row onward.
//!
//! ## Fixed vs. variable row heights
//!
//! - **Fixed**: leave `row_heights` as `None`. Every row uses
//!   `item_height`. Window math runs in O(1) per lookup.
//! - **Variable**: populate `row_heights` (or mutate it via
//!   [`crate::node::NodeTree::set_lazy_list_row_height`]). The prefix
//!   sum table maps offset → row in O(log N) via binary search;
//!   total content height = `prefix[item_count]`.
//!
//! Animating a single row's height (e.g. expand-on-click) writes a
//! tweened height each frame; the partial-recompute path keeps cost at
//! O(item_count - changed_row).

use std::rc::Rc;

use crate::node::NodeId;
use crate::scene::Scene;

#[derive(Clone)]
pub struct LazyListState {
    /// Total logical row count.
    pub item_count: u32,
    /// Default row height in logical px. Used when `row_heights` is
    /// `None`, or as the fallback if a `row_heights` entry is missing
    /// (e.g. the vec is shorter than item_count for any reason).
    pub item_height: f32,
    /// Number of extra rows materialized above + below the visible
    /// window. Smooths fast-scroll feel; default 2.
    pub buffer: u32,
    /// Render closure: spawns exactly one child node per call, parented
    /// to the list. Multi-element rows go inside a row container the
    /// closure emits as its single output.
    pub render: Rc<dyn Fn(&mut Scene, u32) + 'static>,
    /// Bumps to invalidate the current materialized set without
    /// changing the visible window (e.g. when items list content
    /// mutated but length stayed the same).
    pub version: u64,
    /// Last `version` the materialization pass observed.
    pub last_seen_version: u64,
    /// Current materialized children in row-index order.
    pub materialized: Vec<NodeId>,
    /// Currently materialized index range `[start, end_exclusive)`.
    pub range: [u32; 2],

    /// Per-row heights in logical px. `None` enables the uniform fast
    /// path. When `Some`, the prefix table is the authoritative top
    /// offset for each row.
    pub row_heights: Option<Vec<f32>>,
    /// Cumulative top offsets in logical px. Length = item_count + 1.
    /// `prefix[i]` = top of row i; `prefix[item_count]` = total height.
    /// Empty when `row_heights` is `None`.
    pub prefix: Vec<f32>,
    /// Bumps on any mutation of `row_heights` (or initial population).
    pub heights_version: u64,
    /// Last `heights_version` the materialize pass rebuilt the prefix
    /// for. When this trails, the prefix table is stale.
    pub last_heights_version: u64,
    /// Smallest row index whose height changed since the prefix was
    /// last rebuilt. `recompute_prefix_from(first_dirty_row)` is then
    /// O(item_count - first_dirty_row) instead of O(item_count).
    /// `u32::MAX` means "clean".
    pub first_dirty_row: u32,
}

impl LazyListState {
    pub fn new<F>(item_count: u32, item_height: f32, render: F) -> Self
    where
        F: Fn(&mut Scene, u32) + 'static,
    {
        Self {
            item_count,
            item_height,
            buffer: 2,
            render: Rc::new(render),
            version: 0,
            last_seen_version: 0,
            materialized: Vec::new(),
            range: [0, 0],
            row_heights: None,
            prefix: Vec::new(),
            heights_version: 0,
            last_heights_version: 0,
            first_dirty_row: u32::MAX,
        }
    }

    /// Switch the list into variable-height mode. Initializes
    /// `row_heights` with one entry per row at `item_height`. Cheap
    /// no-op when already in variable mode.
    pub fn enable_variable_heights(&mut self) {
        if self.row_heights.is_some() {
            return;
        }
        self.row_heights = Some(vec![self.item_height; self.item_count as usize]);
        self.heights_version = self.heights_version.wrapping_add(1);
        self.first_dirty_row = 0;
    }

    /// Set row `i`'s logical height. Auto-enters variable mode on
    /// first call. Returns `true` if the height actually changed.
    pub fn set_row_height(&mut self, i: u32, h: f32) -> bool {
        if i >= self.item_count {
            return false;
        }
        if self.row_heights.is_none() {
            self.enable_variable_heights();
        }
        let heights = self.row_heights.as_mut().unwrap();
        if heights.len() != self.item_count as usize {
            heights.resize(self.item_count as usize, self.item_height);
        }
        let idx = i as usize;
        if (heights[idx] - h).abs() < f32::EPSILON {
            return false;
        }
        heights[idx] = h;
        self.first_dirty_row = self.first_dirty_row.min(i);
        self.heights_version = self.heights_version.wrapping_add(1);
        true
    }

    /// Replace the entire heights vec. Bumps versions and dirties
    /// every row.
    pub fn set_row_heights(&mut self, heights: Vec<f32>) {
        let n = self.item_count as usize;
        let mut v = heights;
        v.resize(n, self.item_height);
        self.row_heights = Some(v);
        self.heights_version = self.heights_version.wrapping_add(1);
        self.first_dirty_row = 0;
    }

    /// Lookup the logical top of row `i`. Returns `i * item_height`
    /// in the uniform path; reads from the prefix table when in
    /// variable mode. Caller is responsible for ensuring the prefix
    /// table is fresh (call from inside materialize, post
    /// `ensure_prefix_fresh`).
    pub fn row_top_logical(&self, i: u32) -> f32 {
        if self.prefix.is_empty() {
            return i as f32 * self.item_height;
        }
        let idx = (i as usize).min(self.prefix.len().saturating_sub(1));
        self.prefix[idx]
    }

    /// Total content height in logical px.
    pub fn total_height_logical(&self) -> f32 {
        if self.prefix.is_empty() {
            self.item_count as f32 * self.item_height
        } else {
            *self.prefix.last().unwrap_or(&0.0)
        }
    }

    /// Rebuild the prefix table from `first_dirty_row` onward. No-op
    /// when `last_heights_version` matches `heights_version` and the
    /// dirty marker is clear. Always grows `prefix` to length
    /// `item_count + 1`.
    pub fn ensure_prefix_fresh(&mut self) {
        // Uniform mode: clear the prefix vec so the fast paths kick in.
        if self.row_heights.is_none() {
            if !self.prefix.is_empty() {
                self.prefix.clear();
            }
            self.last_heights_version = self.heights_version;
            self.first_dirty_row = u32::MAX;
            return;
        }
        let target_len = self.item_count as usize + 1;
        if self.prefix.len() != target_len {
            self.prefix.resize(target_len, 0.0);
            self.first_dirty_row = 0;
        }
        let clean = self.last_heights_version == self.heights_version
            && self.first_dirty_row == u32::MAX;
        if clean {
            return;
        }
        let start = (self.first_dirty_row as usize).min(self.item_count as usize);
        let heights = self.row_heights.as_ref().unwrap();
        let count = self.item_count as usize;
        let fallback = self.item_height;
        // prefix[start] is unchanged from before (top of the first
        // dirty row equals sum of heights of rows above, which are
        // clean). Recompute prefix[start + 1 ..= count].
        let mut acc = self.prefix.get(start).copied().unwrap_or(0.0);
        for i in start..count {
            let h = heights.get(i).copied().unwrap_or(fallback);
            acc += h;
            self.prefix[i + 1] = acc;
        }
        self.last_heights_version = self.heights_version;
        self.first_dirty_row = u32::MAX;
    }

    /// Compute the visible row window from the list's own rect height
    /// + current scroll offset (physical px). Caller MUST have called
    /// `ensure_prefix_fresh` first when in variable mode.
    pub fn visible_window(&self, scroll_top: f32, viewport_h: f32, scale: f32) -> [u32; 2] {
        if self.item_count == 0 || viewport_h <= 0.0 {
            return [0, 0];
        }
        let count = self.item_count;
        let buf = self.buffer;

        // Uniform fast path.
        if self.prefix.is_empty() {
            let ih = self.item_height * scale;
            if ih <= 0.0 {
                return [0, 0];
            }
            let first = (scroll_top / ih).floor().max(0.0) as u32;
            let last = ((scroll_top + viewport_h) / ih).ceil().max(0.0) as u32;
            let start = first.saturating_sub(buf);
            let end = (last + buf).min(count);
            return [start, end.max(start)];
        }

        // Variable path: prefix in logical, scroll/viewport in physical.
        let scroll_top_l = (scroll_top / scale).max(0.0);
        let viewport_h_l = (viewport_h / scale).max(0.0);
        let bottom_l = scroll_top_l + viewport_h_l;

        // first row containing scroll_top_l:
        //   largest i with prefix[i] <= scroll_top_l
        let upper = self
            .prefix
            .partition_point(|&v| v <= scroll_top_l);
        let first = (upper.saturating_sub(1)) as u32;

        // last row visible: smallest i with prefix[i] >= bottom_l, then
        // include up to that row.
        let upper_b = self
            .prefix
            .partition_point(|&v| v < bottom_l);
        let last_inclusive = (upper_b.saturating_sub(1)) as u32;
        // partition_point returns 0..=len, but row indices are 0..count.
        let last_inclusive = last_inclusive.min(count.saturating_sub(1));

        let start = first.saturating_sub(buf);
        let end = (last_inclusive + 1 + buf).min(count);
        [start, end.max(start)]
    }
}

impl std::fmt::Debug for LazyListState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyListState")
            .field("item_count", &self.item_count)
            .field("item_height", &self.item_height)
            .field("buffer", &self.buffer)
            .field("version", &self.version)
            .field("range", &self.range)
            .field("materialized_count", &self.materialized.len())
            .field(
                "variable_heights",
                &self.row_heights.as_ref().map(|v| v.len()),
            )
            .field("heights_version", &self.heights_version)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(count: u32, h: f32) -> LazyListState {
        LazyListState::new(count, h, |_, _| {})
    }

    // --- uniform path ---

    #[test]
    fn window_covers_visible_rows_plus_buffer() {
        let ll = make(1000, 40.0);
        let w = ll.visible_window(200.0, 200.0, 1.0);
        assert_eq!(w, [3, 12]);
    }

    #[test]
    fn window_at_top_clamps_start_to_zero() {
        let ll = make(1000, 40.0);
        let w = ll.visible_window(0.0, 200.0, 1.0);
        assert_eq!(w[0], 0);
    }

    #[test]
    fn window_at_bottom_clamps_end_to_count() {
        let ll = make(20, 40.0);
        let w = ll.visible_window(800.0, 200.0, 1.0);
        assert_eq!(w[1], 20);
    }

    #[test]
    fn empty_list_yields_empty_window() {
        let ll = make(0, 40.0);
        let w = ll.visible_window(0.0, 200.0, 1.0);
        assert_eq!(w, [0, 0]);
    }

    #[test]
    fn scale_factor_applied() {
        let ll = make(100, 40.0);
        let w = ll.visible_window(0.0, 200.0, 2.0);
        assert_eq!(w, [0, 5]);
    }

    #[test]
    fn fewer_items_than_buffer_still_clamps() {
        let ll = make(3, 40.0);
        let w = ll.visible_window(0.0, 500.0, 1.0);
        assert_eq!(w, [0, 3]);
    }

    // --- variable path ---

    #[test]
    fn enable_variable_heights_initializes_to_item_height() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        ll.ensure_prefix_fresh();
        assert_eq!(ll.prefix, vec![0.0, 40.0, 80.0, 120.0, 160.0, 200.0]);
        assert_eq!(ll.total_height_logical(), 200.0);
    }

    #[test]
    fn set_row_height_bumps_versions_and_dirties_only_from_changed_row() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        ll.ensure_prefix_fresh();
        let v0 = ll.heights_version;
        let changed = ll.set_row_height(2, 100.0);
        assert!(changed);
        assert!(ll.heights_version > v0);
        assert_eq!(ll.first_dirty_row, 2);
        ll.ensure_prefix_fresh();
        // Rows 0,1 untouched; row 2 now 100 px tall; rows 3,4 still 40.
        assert_eq!(ll.prefix, vec![0.0, 40.0, 80.0, 180.0, 220.0, 260.0]);
    }

    #[test]
    fn no_op_set_row_height_returns_false() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        assert!(!ll.set_row_height(2, 40.0)); // already at item_height
    }

    #[test]
    fn variable_window_uses_prefix() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        // Row 0: 40, row 1: 200, rows 2..5: 40.
        ll.set_row_height(1, 200.0);
        ll.ensure_prefix_fresh();
        // prefix = [0, 40, 240, 280, 320, 360].
        // Scroll 0, viewport 100 → contains rows 0,1 (row 1 starts at 40).
        let w = ll.visible_window(0.0, 100.0, 1.0);
        // buffer 2 — [max(0-2,0), 1+1+2] = [0, 4] (clamped).
        assert_eq!(w, [0, 4]);
    }

    #[test]
    fn variable_window_jumps_past_tall_row() {
        let mut ll = make(10, 40.0);
        ll.enable_variable_heights();
        // Row 0 is 1000 px tall.
        ll.set_row_height(0, 1000.0);
        ll.ensure_prefix_fresh();
        // Scroll to mid-row-0 (500 px). Only row 0 in viewport.
        let w = ll.visible_window(500.0, 100.0, 1.0);
        // first = row containing 500 → row 0 (prefix[0]=0, prefix[1]=1000).
        // last_inclusive = row containing 600 → row 0.
        // buffer 2 → [0, 1+2] = [0, 3].
        assert_eq!(w, [0, 3]);
    }

    #[test]
    fn variable_path_drops_back_to_uniform_when_heights_cleared() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        ll.set_row_height(0, 100.0);
        ll.ensure_prefix_fresh();
        assert!(!ll.prefix.is_empty());
        // Drop back to uniform.
        ll.row_heights = None;
        ll.heights_version = ll.heights_version.wrapping_add(1);
        ll.ensure_prefix_fresh();
        assert!(ll.prefix.is_empty());
        assert_eq!(ll.total_height_logical(), 200.0);
    }

    #[test]
    fn variable_path_respects_scale_for_total() {
        let mut ll = make(3, 40.0);
        ll.enable_variable_heights();
        ll.set_row_height(1, 80.0);
        ll.ensure_prefix_fresh();
        // Total logical 40 + 80 + 40 = 160. Scale comes in elsewhere
        // — total_height_logical is logical-only.
        assert_eq!(ll.total_height_logical(), 160.0);
    }

    #[test]
    fn item_count_change_resizes_prefix() {
        let mut ll = make(5, 40.0);
        ll.enable_variable_heights();
        ll.ensure_prefix_fresh();
        assert_eq!(ll.prefix.len(), 6);
        // Pretend item_count grew.
        ll.item_count = 8;
        if let Some(h) = ll.row_heights.as_mut() {
            h.resize(8, 40.0);
        }
        ll.heights_version = ll.heights_version.wrapping_add(1);
        ll.first_dirty_row = 0;
        ll.ensure_prefix_fresh();
        assert_eq!(ll.prefix.len(), 9);
        assert_eq!(ll.total_height_logical(), 320.0);
    }
}

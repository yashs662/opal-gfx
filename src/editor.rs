//! Text-field state.
//!
//! `EditorState` is the per-node bag of state that turns a regular
//! Rect container into an editable text field. It owns the value
//! string, the cursor byte offset, the placeholder + font size, and
//! the `NodeId`s of the two render-time helper nodes:
//!
//!   - `text_node` — a `ShapeKind::Text` child that renders whatever
//!     should be visible right now (value, or placeholder when value
//!     is empty and the field isn't focused).
//!   - `caret_node` — a thin `ShapeKind::Rect` child whose `visible`
//!     flag follows the parent's focused signal. Position is
//!     re-computed after every layout pass from
//!     `TextResources::measure(value[..cursor])`.
//!
//! Mutations flow one way: the App's keyboard handler resolves
//! winit's `KeyEvent` to an [`EditOp`] and calls
//! `apply(op, &mut EditorState)`. The fn returns a [`EditOutcome`]
//! that the App uses to fire `on_change` / `on_submit` and to mark
//! the right dirty bits.
//!
//! Single-line, ASCII-friendly. Selection + clipboard are supported;
//! IME / multi-line remain out of scope here. The actual
//! clipboard IO (arboard) lives in the App layer — `apply` stays pure
//! and surfaces a `clipboard_write` request in its [`EditOutcome`].

use std::rc::Rc;

use crate::event::EventHandler;
use crate::node::NodeId;

/// Per-node text-field state. Lives on `Node.editor` as a
/// `Option<Box<EditorState>>`.
#[derive(Clone)]
pub struct EditorState {
    /// Current value. Edits mutate this in-place. Byte indices into
    /// the value are valid `cursor` positions, including `value.len()`
    /// (one-past-end). Non-grapheme cursor movement in v1 — Arrow keys
    /// step by *bytes*, which works for ASCII and breaks for combining
    /// marks. Documented limitation.
    pub value: String,
    /// Cursor byte offset into `value`. Always satisfies
    /// `cursor <= value.len()` and lands on a UTF-8 char boundary.
    pub cursor: usize,
    /// Selection anchor. `None` = no selection (a bare caret at
    /// `cursor`). When `Some(a)`, the selected span is
    /// `[min(a, cursor), max(a, cursor)]`; an empty span (`a == cursor`)
    /// renders as no highlight. Always on a UTF-8 boundary.
    pub selection_anchor: Option<usize>,
    pub placeholder: String,
    /// Logical px. Forwarded to the text child unchanged.
    pub font_size: f32,
    /// Colour of the typed value text. Applied to `text_node` whenever the
    /// value (not the placeholder) is shown. Defaults to near-white so a
    /// freshly-spawned field is legible without extra wiring.
    pub text_color: [f32; 4],
    /// Colour of the placeholder text (shown when empty + unfocused).
    pub placeholder_color: [f32; 4],
    /// Text-display child node id (a `ShapeKind::Text` Node).
    pub text_node: NodeId,
    /// Caret rect child node id (a `ShapeKind::Rect` Node, 2 logical
    /// px wide). `visible` tracks the parent's focused signal.
    pub caret_node: NodeId,
    /// Selection-highlight rect child node id. Painted *behind*
    /// the text; `visible` + geometry are recomputed each layout pass
    /// from the selection span.
    pub selection_node: NodeId,
    /// Fired after every value mutation. Receives the post-edit value.
    pub on_change: Option<Rc<dyn Fn(&str) + 'static>>,
    /// Fired when Enter is pressed while the field is focused. Same
    /// shape as `Node.on_click` — receives `EventCtx`.
    pub on_submit: Option<EventHandler>,
}

impl std::fmt::Debug for EditorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorState")
            .field("value", &self.value)
            .field("cursor", &self.cursor)
            .field("selection_anchor", &self.selection_anchor)
            .field("placeholder", &self.placeholder)
            .field("font_size", &self.font_size)
            .field("text_node", &self.text_node)
            .field("caret_node", &self.caret_node)
            .field("selection_node", &self.selection_node)
            .field("on_change", &self.on_change.as_ref().map(|_| "<fn>"))
            .field("on_submit", &self.on_submit.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

/// Resolved keyboard action — what the keyboard handler decided this
/// event should do. Decoupled from winit's `KeyEvent` so the editor
/// logic stays testable without spinning up a winit window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditOp {
    /// Insert this string at the cursor. Typically one character (a
    /// SmolStr from `KeyEvent.text`), but can be a multi-byte string
    /// from IME composition.
    Insert(String),
    /// Delete the byte/grapheme immediately before the cursor
    /// (Backspace).
    DeleteBack,
    /// Delete the byte/grapheme immediately after the cursor (Delete).
    DeleteForward,
    /// Move cursor one step to the left.
    MoveLeft,
    /// Move cursor one step to the right.
    MoveRight,
    /// Jump cursor to byte 0.
    Home,
    /// Jump cursor to `value.len()`.
    End,
    /// Submit (Enter pressed). Triggers `on_submit`.
    Submit,
    /// Extend selection one step left (Shift+Left).
    SelectLeft,
    /// Extend selection one step right (Shift+Right).
    SelectRight,
    /// Extend selection to byte 0 (Shift+Home).
    SelectHome,
    /// Extend selection to `value.len()` (Shift+End).
    SelectEnd,
    /// Select the whole value (Ctrl/Cmd+A).
    SelectAll,
    /// Copy the selection to the clipboard (Ctrl/Cmd+C). No mutation;
    /// surfaces the text via [`EditOutcome::clipboard_write`].
    Copy,
    /// Cut the selection (Ctrl/Cmd+X) — copies then deletes it.
    Cut,
    /// Paste a string at the cursor (Ctrl/Cmd+V), replacing any
    /// selection. The App reads the clipboard and supplies the text.
    Paste(String),
}

/// Result of an `apply` call — tells the caller what changed so it
/// can decide which dirty bits to set and whether to fire callbacks.
/// Not `Copy` — it carries an owned `clipboard_write` string.
#[derive(Default, Debug, Clone)]
pub struct EditOutcome {
    /// Value string changed (caller should fire `on_change` and
    /// refresh the text child node's content).
    pub value_changed: bool,
    /// Cursor moved (caller should reposition the caret child).
    pub cursor_moved: bool,
    /// Enter was pressed (caller should fire `on_submit`).
    pub submitted: bool,
    /// Selection span changed (caller should reposition the highlight).
    pub selection_changed: bool,
    /// Copy/Cut produced text the App should write to the system
    /// clipboard (arboard). `None` otherwise.
    pub clipboard_write: Option<String>,
}

impl EditOutcome {
    pub fn any(&self) -> bool {
        self.value_changed
            || self.cursor_moved
            || self.submitted
            || self.selection_changed
            || self.clipboard_write.is_some()
    }
}

/// Ordered selection span `[start, end)` in byte offsets, or `None`
/// when there is no (or an empty) selection.
pub fn selection_range(st: &EditorState) -> Option<(usize, usize)> {
    let a = st.selection_anchor?;
    let (lo, hi) = (a.min(st.cursor), a.max(st.cursor));
    (lo != hi).then_some((lo, hi))
}

/// The currently selected substring, or `None` when nothing is
/// selected.
pub fn selected_text(st: &EditorState) -> Option<String> {
    selection_range(st).map(|(lo, hi)| st.value[lo..hi].to_string())
}

/// Delete the active selection (if any) in place, leaving the cursor at
/// its start and clearing the anchor. Returns true if anything was
/// removed.
fn delete_selection(st: &mut EditorState) -> bool {
    if let Some((lo, hi)) = selection_range(st) {
        st.value.drain(lo..hi);
        st.cursor = lo;
        st.selection_anchor = None;
        true
    } else {
        st.selection_anchor = None;
        false
    }
}

/// Apply an [`EditOp`] to the editor state. Returns an outcome
/// describing what mutated. Caller wires the rest (text node refresh,
/// caret reposition, callback firing).
///
/// All cursor positions are kept on UTF-8 char boundaries via
/// `floor_char_boundary` / `ceil_char_boundary`. Inserting an
/// arbitrary `&str` at any cursor position cannot break this
/// invariant.
pub fn apply(op: EditOp, st: &mut EditorState) -> EditOutcome {
    let mut out = EditOutcome::default();
    let had_selection = selection_range(st).is_some();
    match op {
        EditOp::Insert(s) => {
            if s.is_empty() {
                return out;
            }
            // Typing over a selection replaces it.
            if delete_selection(st) {
                out.selection_changed = true;
            }
            st.value.insert_str(st.cursor, &s);
            st.cursor += s.len();
            out.value_changed = true;
            out.cursor_moved = true;
        }
        EditOp::Paste(s) => {
            if delete_selection(st) {
                out.selection_changed = true;
                out.value_changed = true;
            }
            if !s.is_empty() {
                st.value.insert_str(st.cursor, &s);
                st.cursor += s.len();
                out.value_changed = true;
                out.cursor_moved = true;
            }
        }
        EditOp::DeleteBack => {
            if delete_selection(st) {
                out.value_changed = true;
                out.cursor_moved = true;
                out.selection_changed = true;
            } else if st.cursor != 0 {
                // Walk back to the previous char boundary.
                let prev = prev_char_boundary(&st.value, st.cursor);
                st.value.drain(prev..st.cursor);
                st.cursor = prev;
                out.value_changed = true;
                out.cursor_moved = true;
            }
        }
        EditOp::DeleteForward => {
            if delete_selection(st) {
                out.value_changed = true;
                out.cursor_moved = true;
                out.selection_changed = true;
            } else if st.cursor < st.value.len() {
                let next = next_char_boundary(&st.value, st.cursor);
                st.value.drain(st.cursor..next);
                out.value_changed = true;
                // cursor doesn't move on delete-forward
            }
        }
        EditOp::MoveLeft => {
            // Collapse an active selection to its left edge.
            if let Some((lo, _)) = selection_range(st) {
                st.cursor = lo;
                st.selection_anchor = None;
                out.cursor_moved = true;
                out.selection_changed = true;
            } else if st.cursor != 0 {
                st.selection_anchor = None;
                st.cursor = prev_char_boundary(&st.value, st.cursor);
                out.cursor_moved = true;
            }
        }
        EditOp::MoveRight => {
            if let Some((_, hi)) = selection_range(st) {
                st.cursor = hi;
                st.selection_anchor = None;
                out.cursor_moved = true;
                out.selection_changed = true;
            } else if st.cursor < st.value.len() {
                st.selection_anchor = None;
                st.cursor = next_char_boundary(&st.value, st.cursor);
                out.cursor_moved = true;
            }
        }
        EditOp::Home => {
            if had_selection {
                out.selection_changed = true;
            }
            st.selection_anchor = None;
            if st.cursor != 0 {
                st.cursor = 0;
                out.cursor_moved = true;
            }
        }
        EditOp::End => {
            if had_selection {
                out.selection_changed = true;
            }
            st.selection_anchor = None;
            let end = st.value.len();
            if st.cursor != end {
                st.cursor = end;
                out.cursor_moved = true;
            }
        }
        EditOp::SelectLeft => {
            if st.cursor != 0 {
                st.selection_anchor.get_or_insert(st.cursor);
                st.cursor = prev_char_boundary(&st.value, st.cursor);
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::SelectRight => {
            if st.cursor < st.value.len() {
                st.selection_anchor.get_or_insert(st.cursor);
                st.cursor = next_char_boundary(&st.value, st.cursor);
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::SelectHome => {
            if st.cursor != 0 {
                st.selection_anchor.get_or_insert(st.cursor);
                st.cursor = 0;
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::SelectEnd => {
            let end = st.value.len();
            if st.cursor != end {
                st.selection_anchor.get_or_insert(st.cursor);
                st.cursor = end;
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::SelectAll => {
            if !st.value.is_empty() {
                st.selection_anchor = Some(0);
                st.cursor = st.value.len();
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::Copy => {
            out.clipboard_write = selected_text(st);
        }
        EditOp::Cut => {
            if let Some(text) = selected_text(st) {
                out.clipboard_write = Some(text);
                delete_selection(st);
                out.value_changed = true;
                out.cursor_moved = true;
                out.selection_changed = true;
            }
        }
        EditOp::Submit => {
            out.submitted = true;
        }
    }
    out
}

/// Walk `idx` back until it lands on a UTF-8 boundary. `idx` itself
/// must already be `≤ s.len()`. Returns the byte index of the char
/// *before* the original cursor — i.e. the boundary cursor moves to
/// on Backspace / MoveLeft.
fn prev_char_boundary(s: &str, mut idx: usize) -> usize {
    debug_assert!(idx <= s.len());
    if idx == 0 {
        return 0;
    }
    idx -= 1;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Walk `idx` forward until just past the next UTF-8 boundary.
fn next_char_boundary(s: &str, mut idx: usize) -> usize {
    debug_assert!(idx <= s.len());
    let len = s.len();
    if idx >= len {
        return len;
    }
    idx += 1;
    while idx < len && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeId;

    fn make(value: &str, cursor: usize) -> EditorState {
        // NodeIds are sentinels here — these tests don't touch a tree.
        let dummy = NodeId::SENTINEL;
        EditorState {
            value: value.to_string(),
            cursor,
            selection_anchor: None,
            placeholder: String::new(),
            font_size: 14.0,
            text_color: [1.0, 1.0, 1.0, 1.0],
            placeholder_color: [1.0, 1.0, 1.0, 0.45],
            text_node: dummy,
            caret_node: dummy,
            selection_node: dummy,
            on_change: None,
            on_submit: None,
        }
    }

    #[test]
    fn insert_at_end() {
        let mut st = make("foo", 3);
        let out = apply(EditOp::Insert("bar".to_string()), &mut st);
        assert_eq!(st.value, "foobar");
        assert_eq!(st.cursor, 6);
        assert!(out.value_changed && out.cursor_moved);
    }

    #[test]
    fn insert_in_middle() {
        let mut st = make("foo", 1);
        apply(EditOp::Insert("X".to_string()), &mut st);
        assert_eq!(st.value, "fXoo");
        assert_eq!(st.cursor, 2);
    }

    #[test]
    fn backspace_removes_prior_char() {
        let mut st = make("foo", 3);
        apply(EditOp::DeleteBack, &mut st);
        assert_eq!(st.value, "fo");
        assert_eq!(st.cursor, 2);
    }

    #[test]
    fn backspace_at_start_noop() {
        let mut st = make("foo", 0);
        let out = apply(EditOp::DeleteBack, &mut st);
        assert_eq!(st.value, "foo");
        assert!(!out.any());
    }

    #[test]
    fn delete_forward_keeps_cursor() {
        let mut st = make("foo", 1);
        apply(EditOp::DeleteForward, &mut st);
        assert_eq!(st.value, "fo");
        assert_eq!(st.cursor, 1);
    }

    #[test]
    fn move_left_and_right() {
        let mut st = make("ab", 2);
        apply(EditOp::MoveLeft, &mut st);
        assert_eq!(st.cursor, 1);
        apply(EditOp::MoveLeft, &mut st);
        assert_eq!(st.cursor, 0);
        let out = apply(EditOp::MoveLeft, &mut st);
        assert!(!out.cursor_moved, "no-op at start");
        apply(EditOp::MoveRight, &mut st);
        assert_eq!(st.cursor, 1);
    }

    #[test]
    fn home_end() {
        let mut st = make("hello", 2);
        apply(EditOp::Home, &mut st);
        assert_eq!(st.cursor, 0);
        apply(EditOp::End, &mut st);
        assert_eq!(st.cursor, 5);
    }

    #[test]
    fn utf8_boundaries_respected() {
        // "héllo" — `é` is 2 bytes (0xC3 0xA9).
        let mut st = make("héllo", 3); // cursor after `é`
        apply(EditOp::MoveLeft, &mut st);
        assert_eq!(st.cursor, 1); // walked past the 2-byte é
        apply(EditOp::DeleteForward, &mut st);
        assert_eq!(st.value, "hllo"); // dropped the full é
    }

    #[test]
    fn submit_sets_flag_only() {
        let mut st = make("foo", 3);
        let out = apply(EditOp::Submit, &mut st);
        assert!(out.submitted);
        assert!(!out.value_changed);
        assert_eq!(st.value, "foo");
    }

    // --- selection + clipboard ---

    #[test]
    fn shift_right_extends_selection() {
        let mut st = make("hello", 0);
        apply(EditOp::SelectRight, &mut st);
        apply(EditOp::SelectRight, &mut st);
        assert_eq!(st.selection_anchor, Some(0));
        assert_eq!(st.cursor, 2);
        assert_eq!(selection_range(&st), Some((0, 2)));
        assert_eq!(selected_text(&st).as_deref(), Some("he"));
    }

    #[test]
    fn shift_left_extends_then_shrinks() {
        let mut st = make("hello", 5);
        apply(EditOp::SelectLeft, &mut st); // anchor 5, cursor 4
        apply(EditOp::SelectLeft, &mut st); // cursor 3
        assert_eq!(selection_range(&st), Some((3, 5)));
        // Shrinking back the other way reduces the span.
        apply(EditOp::SelectRight, &mut st); // cursor 4
        assert_eq!(selection_range(&st), Some((4, 5)));
    }

    #[test]
    fn select_all_then_collapse_with_arrow() {
        let mut st = make("hello", 2);
        apply(EditOp::SelectAll, &mut st);
        assert_eq!(selection_range(&st), Some((0, 5)));
        // Left collapses to the start; Right would collapse to the end.
        let out = apply(EditOp::MoveLeft, &mut st);
        assert_eq!(st.cursor, 0);
        assert_eq!(st.selection_anchor, None);
        assert!(out.selection_changed);
    }

    #[test]
    fn typing_replaces_selection() {
        let mut st = make("hello", 0);
        apply(EditOp::SelectAll, &mut st);
        apply(EditOp::Insert("Z".to_string()), &mut st);
        assert_eq!(st.value, "Z");
        assert_eq!(st.cursor, 1);
        assert_eq!(st.selection_anchor, None);
    }

    #[test]
    fn delete_back_removes_whole_selection() {
        let mut st = make("hello", 1);
        apply(EditOp::SelectRight, &mut st); // select "e"
        apply(EditOp::SelectRight, &mut st); // select "el"
        apply(EditOp::DeleteBack, &mut st);
        assert_eq!(st.value, "hlo");
        assert_eq!(st.cursor, 1);
        assert_eq!(st.selection_anchor, None);
    }

    #[test]
    fn copy_reports_selection_without_mutating() {
        let mut st = make("hello", 0);
        apply(EditOp::SelectRight, &mut st);
        apply(EditOp::SelectRight, &mut st);
        let out = apply(EditOp::Copy, &mut st);
        assert_eq!(out.clipboard_write.as_deref(), Some("he"));
        assert!(!out.value_changed);
        assert_eq!(st.value, "hello");
        // Selection survives a copy.
        assert_eq!(selection_range(&st), Some((0, 2)));
    }

    #[test]
    fn cut_copies_then_deletes() {
        let mut st = make("hello", 0);
        apply(EditOp::SelectRight, &mut st);
        apply(EditOp::SelectRight, &mut st);
        let out = apply(EditOp::Cut, &mut st);
        assert_eq!(out.clipboard_write.as_deref(), Some("he"));
        assert_eq!(st.value, "llo");
        assert_eq!(st.cursor, 0);
        assert!(st.selection_anchor.is_none());
    }

    #[test]
    fn paste_replaces_selection() {
        let mut st = make("hello", 0);
        apply(EditOp::SelectRight, &mut st);
        apply(EditOp::SelectRight, &mut st); // select "he"
        apply(EditOp::Paste("XY".to_string()), &mut st);
        assert_eq!(st.value, "XYllo");
        assert_eq!(st.cursor, 2);
    }

    #[test]
    fn paste_at_caret_without_selection() {
        let mut st = make("ac", 1);
        apply(EditOp::Paste("b".to_string()), &mut st);
        assert_eq!(st.value, "abc");
        assert_eq!(st.cursor, 2);
    }

    #[test]
    fn copy_with_no_selection_writes_nothing() {
        let mut st = make("hello", 2);
        let out = apply(EditOp::Copy, &mut st);
        assert!(out.clipboard_write.is_none());
    }
}

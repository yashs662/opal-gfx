//! Event callbacks.
//!
//! A node's `on_click` is stored as an `EventHandler`: a refcounted,
//! `'static`-bound closure invoked by the app shell when a left-button
//! release lands on the same node that captured the press. Same OS-button
//! semantics: press-down, drag-off, release-anywhere-else = no click.
//!
//! The handler is given an `EventCtx` with mutable access to the node
//! tree so it can drive signals or push immediate tree mutations. The
//! caller (the app shell in `app.rs`) clones the `Rc<dyn Fn>` out of the
//! node first, drops every borrow into the tree, and *then* invokes the
//! handler — guarantees the handler can re-borrow `&mut NodeTree` from
//! `EventCtx` without aliasing the original `&Node` it came from.

use std::rc::Rc;
use std::time::Instant;

use crate::node::{NodeId, NodeTree};

/// Context handed to an [`EventHandler`] invocation. Holds an exclusive
/// borrow of the node tree so handlers can mutate live state directly
/// (e.g. flip a sibling's visibility) alongside the conventional path of
/// poking a captured `Signal<_>`. `timeline` is exposed so handlers can
/// kick off tweens that the shell ticks each frame.
pub struct EventCtx<'a> {
    pub tree: &'a mut NodeTree,
    pub timeline: &'a mut crate::anim::Timeline,
    pub node: NodeId,
    pub now: Instant,
}

/// A node-bound event callback. Refcounted so the same closure can be
/// shared across multiple nodes (e.g. a row builder that wires the same
/// "select" handler to every row in a list) without re-allocating per
/// site. The `'static` bound is the price of storing the closure on the
/// node — captures must be owned (`Signal<T>` clones are the common
/// pattern: `Rc` internally, cheap to clone into the closure).
pub type EventHandler = Rc<dyn for<'a> Fn(&mut EventCtx<'a>) + 'static>;

/// Convenience constructor — equivalent to `Rc::new(f)` but typed so the
/// inference path is unambiguous at call sites.
pub fn handler<F>(f: F) -> EventHandler
where
    F: for<'a> Fn(&mut EventCtx<'a>) + 'static,
{
    Rc::new(f)
}

/// Context handed to a [`DragHandler`] (`on_drag`). Fires on every
/// cursor move while a left-press is captured on the node. `start` is the
/// cursor position (physical px) at press; `current` is now; `delta` is
/// `current - last_fire` (per-event step, **not** total from start — sum
/// it yourself if you need the cumulative offset). Drives sliders +
/// scrubbers.
pub struct DragCtx<'a> {
    pub tree: &'a mut NodeTree,
    pub node: NodeId,
    pub start: [f32; 2],
    pub current: [f32; 2],
    pub delta: [f32; 2],
}

/// A node-bound drag callback. See [`DragCtx`].
pub type DragHandler = Rc<dyn for<'a> Fn(&mut DragCtx<'a>) + 'static>;

/// Context handed to a [`DropHandler`]. Fires when a left-press release
/// lands over this drop-target node while a drag payload is in flight.
/// `payload` is the type-erased value the drag source carried — the
/// receiver downcasts via `payload.downcast_ref::<T>()`.
pub struct DropCtx<'a> {
    pub tree: &'a mut NodeTree,
    pub node: NodeId,
    pub payload: Rc<dyn std::any::Any>,
}

/// A node-bound drop callback. See [`DropCtx`].
pub type DropHandler = Rc<dyn for<'a> Fn(&mut DropCtx<'a>) + 'static>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeTree};
    use std::cell::Cell;

    #[test]
    fn handler_fires_with_eventctx() {
        let mut tree = NodeTree::new();
        let mut timeline = crate::anim::Timeline::new();
        let counter = Rc::new(Cell::new(0_u32));
        let c2 = counter.clone();
        let h: EventHandler = handler(move |_ctx| {
            c2.set(c2.get() + 1);
        });
        let id = tree.add_root(Node::rect().build());
        tree.get_mut_raw(id).unwrap().on_click = Some(h);

        let cloned = tree.get(id).unwrap().on_click.clone();
        assert!(cloned.is_some());
        let h2 = cloned.unwrap();
        let mut ectx = EventCtx {
            tree: &mut tree,
            timeline: &mut timeline,
            node: id,
            now: std::time::Instant::now(),
        };
        h2(&mut ectx);
        h2(&mut ectx);
        assert_eq!(counter.get(), 2);
    }

    #[test]
    fn handler_can_mutate_tree_through_ctx() {
        let mut tree = NodeTree::new();
        let mut timeline = crate::anim::Timeline::new();
        let id = tree.add_root(Node::rect().build());
        let target = id;
        let h: EventHandler = handler(move |ctx| {
            if let Some(n) = ctx.tree.get_mut_raw(target) {
                n.style.color = [1.0, 0.0, 0.0, 1.0];
            }
        });
        let mut ectx = EventCtx {
            tree: &mut tree,
            timeline: &mut timeline,
            node: id,
            now: std::time::Instant::now(),
        };
        h(&mut ectx);
        assert_eq!(tree.get(id).unwrap().style.color, [1.0, 0.0, 0.0, 1.0]);
    }
}

//! Retained node tree.
//!
//! Generational-index arena. Stage-1 nodes are absolute-positioned via the
//! builder API; a parent's position is added to each descendant's local
//! offset (no rotation/scale propagation, no layout engine yet). Opacity
//! multiplies down the tree.
//!
//! `NodeId`s are stable across mutations of *other* nodes — they only
//! invalidate when the specific slot they refer to is reused.

use crate::gpu::ShapeInstance;

/// Tree-level dirty flags. M3 collapses everything to a single
/// re-flatten + full re-upload when any bit is set; M9 may switch to
/// per-slot tracking once instance counts justify it.
pub mod dirty {
    pub const NONE: u32 = 0;
    /// Color, opacity, border or shadow style changed.
    pub const VISUAL: u32 = 1 << 0;
    /// Position or size changed.
    pub const TRANSFORM: u32 = 1 << 1;
    /// Tree topology changed (add, remove, visibility flip).
    pub const TREE: u32 = 1 << 2;
    pub const ANY: u32 = VISUAL | TRANSFORM | TREE;
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
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            color: [1.0; 4],
            border_color: [0.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            border_radius: [0.0; 4],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub style: ShapeStyle,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub visible: bool,
    pub children: Vec<NodeId>,
}

impl Node {
    pub fn rect() -> NodeBuilder {
        NodeBuilder {
            node: Node {
                style: ShapeStyle::default(),
                position: [0.0, 0.0],
                size: [0.0, 0.0],
                visible: true,
                children: Vec::new(),
            },
        }
    }
}

pub struct NodeBuilder {
    node: Node,
}

impl NodeBuilder {
    pub fn pos(mut self, x: f32, y: f32) -> Self {
        self.node.position = [x, y];
        self
    }
    pub fn size(mut self, w: f32, h: f32) -> Self {
        self.node.size = [w, h];
        self
    }
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
    pub fn build(self) -> Node {
        self.node
    }
}

struct Slot {
    generation: u32,
    payload: Option<Node>,
}

#[derive(Default)]
pub struct NodeTree {
    slots: Vec<Slot>,
    free: Vec<u32>,
    roots: Vec<NodeId>,
    dirty: u32,
}

impl NodeTree {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, node: Node) -> NodeId {
        if let Some(idx) = self.free.pop() {
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
        }
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
        slot.generation = slot.generation.wrapping_add(1);
        slot.payload = None;
        self.free.push(id.index);
        self.roots.retain(|r| *r != id);
        self.dirty |= dirty::TREE;
    }

    pub fn set_position(&mut self, id: NodeId, position: [f32; 2]) {
        if let Some(n) = self.get_mut_raw(id) {
            if n.position != position {
                n.position = position;
                self.dirty |= dirty::TRANSFORM;
            }
        }
    }

    pub fn set_size(&mut self, id: NodeId, size: [f32; 2]) {
        if let Some(n) = self.get_mut_raw(id) {
            if n.size != size {
                n.size = size;
                self.dirty |= dirty::TRANSFORM;
            }
        }
    }

    pub fn set_color(&mut self, id: NodeId, color: [f32; 4]) {
        if let Some(n) = self.get_mut_raw(id) {
            if n.style.color != color {
                n.style.color = color;
                self.dirty |= dirty::VISUAL;
            }
        }
    }

    pub fn set_opacity(&mut self, id: NodeId, opacity: f32) {
        if let Some(n) = self.get_mut_raw(id) {
            if n.style.opacity != opacity {
                n.style.opacity = opacity;
                self.dirty |= dirty::VISUAL;
            }
        }
    }

    pub fn set_visible(&mut self, id: NodeId, visible: bool) {
        if let Some(n) = self.get_mut_raw(id) {
            if n.visible != visible {
                n.visible = visible;
                self.dirty |= dirty::TREE;
            }
        }
    }

    pub fn dirty(&self) -> u32 {
        self.dirty
    }

    /// Read-and-clear the dirty mask. App calls this each event tick to
    /// decide whether to re-flatten + re-upload + redraw.
    pub fn take_dirty(&mut self) -> u32 {
        let d = self.dirty;
        self.dirty = dirty::NONE;
        d
    }

    /// Force a full rebuild on the next tick. Used by F5 force-redraw.
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

    /// Untracked mutable accessor — does **not** flag dirty. Prefer the
    /// typed setters (`set_position`, `set_color`, …); reach for this only
    /// when you intentionally batch a multi-field edit and call
    /// `mark_all_dirty` yourself.
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

    /// DFS preorder flatten into GPU instances. Parent position adds to
    /// each descendant. Parent opacity multiplies down. Hidden nodes (and
    /// their subtrees) are skipped.
    pub fn flatten(&self) -> Vec<ShapeInstance> {
        let mut out = Vec::with_capacity(self.len());
        for root in &self.roots {
            self.flatten_into(*root, [0.0, 0.0], 1.0, &mut out);
        }
        out
    }

    fn flatten_into(
        &self,
        id: NodeId,
        parent_offset: [f32; 2],
        parent_opacity: f32,
        out: &mut Vec<ShapeInstance>,
    ) {
        let Some(node) = self.get(id) else { return };
        if !node.visible {
            return;
        }
        let abs = [
            parent_offset[0] + node.position[0],
            parent_offset[1] + node.position[1],
        ];
        let opacity = parent_opacity * node.style.opacity;
        out.push(ShapeInstance {
            color: node.style.color,
            border_color: node.style.border_color,
            shadow_color: node.style.shadow_color,
            border_radius: node.style.border_radius,
            position: abs,
            size: node.size,
            shadow_offset: node.style.shadow_offset,
            _pad0: [0.0; 2],
            border_width: node.style.border_width,
            shadow_blur: node.style.shadow_blur,
            shadow_opacity: node.style.shadow_opacity,
            opacity,
        });
        for &child in &node.children {
            self.flatten_into(child, abs, opacity, out);
        }
    }
}

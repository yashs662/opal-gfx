//! frostify-gfx — reactive GPU UI rendering library.
//!
//! Transparent window, SDF shapes with solid colors and frosted glass
//! (per-instance blur + edge refraction), text via cosmic-text, image
//! atlas blits, and a flex-style layout engine.
//!
//! The crate is a **library** — it does not own a window or an event
//! loop. The public surface is:
//!
//! - [`GpuContext`] — wgpu setup, multi-pass renderer, headless capture.
//! - [`NodeTree`], [`Node`], [`NodeId`], [`HitEntry`] — retained scene graph.
//! - [`Signal`] — reactive value primitive used by interactive nodes.
//! - [`InputState`] — cursor/hover/press bookkeeping that consumers plug
//!   into whichever event source they use (e.g. winit).
//! - [`debug`] — PNG screenshot helper for manual + headless verification.
//!
//! See `examples/hello_window.rs` for a full integration that builds a
//! winit event loop, a demo scene, and env-var-driven headless captures.

// Intentional for a graphics/UI library, not bugs:
//   - layout/vertex/transform helpers legitimately take many scalar
//     params (factoring into structs just adds indirection);
//   - retained-scene fields hold boxed callbacks (`Box<dyn FnMut(...)>`),
//     which clippy reads as "complex types";
//   - several doc comments use prose with leading `+`/`-` that the lint
//     mistakes for unindented markdown list continuations.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_lazy_continuation
)]

pub mod anim;
pub mod app;
pub mod debug;
pub mod editor;
pub mod event;
pub mod gpu;
pub mod input;
pub mod layout;
pub mod lazy_list;
pub mod node;
pub mod reactive;
pub mod scene;
pub mod signal;
pub mod svg;
pub mod text;
pub mod uploader;

/// Build a tuple of `.clone()`'d reactive sources for [`Computed::new`].
/// Drops the per-call `.clone()` boilerplate without changing the
/// dependency-graph mechanism — the expansion is grep-able and identical
/// to writing the tuple by hand.
///
/// ```ignore
/// let c = Computed::new(deps!(lit, hover, pressed), |(l, h, p)| { … });
/// // expands to
/// let c = Computed::new((lit.clone(), hover.clone(), pressed.clone()), |(l, h, p)| { … });
/// ```
#[macro_export]
macro_rules! deps {
    ($($src:expr),+ $(,)?) => {
        ( $(::std::clone::Clone::clone(&$src),)+ )
    };
}

pub use anim::{Curve, Lerp, TickResult, Timeline, Tween};
pub use app::{App, AppConfig, HeadlessHelper, WakeHandle};
pub use editor::{EditOp, EditOutcome, EditorState};
pub use lazy_list::LazyListState;
pub use event::{
    handler, DragCtx, DragHandler, DropCtx, DropHandler, EventCtx, EventHandler,
};
pub use gpu::{
    FrameStats, FrameTiming, FrameUniform, GpuContext, ImageAtlas, ImageEntry, ImageHandle,
    MemoryReport, ShapeInstance,
};
pub use input::{InputChange, InputState};
pub use layout::{Align, Axis, Justify, LayoutStyle, Len, Measurer, NullMeasurer, Overflow};
pub use node::{
    dirty, BarSide, BorderSides, HitEntry, ImageRef, Node, NodeBuilder, NodeId, NodeTree,
    ScrollAxis, ScrollHit, ScrollState, ScrollbarHit, ScrollbarStyle, ShapeKind, ShapeStyle,
    TextRef, WindowAction,
};
pub use reactive::{animated, AnimatedBind, Bind, Computed, DepTuple, ImageBind, Source, TextBind};
pub use scene::{
    BindRegistry, ColorBindSlot, ImageBindSlot, IntoNodeName, NodeBuilderRef, PositionBindSlot,
    Scene, SceneCtx, SizeBindSlot, SubtreeRemoval, TextBindSlot, WidthPctBindSlot,
};
pub use signal::{Signal, TextSignal};
pub use svg::{rasterize_svg, rasterize_svg_to};
pub use text::{RasterizedGlyph, ShapedGlyph, TextMetrics, TextResources};
pub use uploader::Uploader;

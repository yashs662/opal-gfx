//! frostify-gfx — reactive GPU UI rendering library.
//!
//! Stage 1 scope: transparent window, SDF shapes with solid colors and
//! glass/roughness. No text, no images, no layout engine. Absolute pixel
//! coordinates for debug layouts are available.

pub mod app;
pub mod debug;
pub mod gpu;
pub mod node;
pub mod signal;

pub use app::{App, AppConfig};
pub use gpu::{FrameUniform, GpuContext, ShapeInstance};
pub use node::{dirty, Node, NodeBuilder, NodeId, NodeTree, ShapeStyle};
pub use signal::Signal;

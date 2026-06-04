//! Cross-thread external-frame (video) submission handle.
//!
//! [`FrameSink`] is the video-side cousin of [`crate::uploader::Uploader`]:
//! a cheaply-clonable token a decoder thread calls into each time it
//! produces a frame for an [`.external()`](crate::node::NodeBuilder::external)
//! node. The shell drains it inside `about_to_wait`, uploads the pixels
//! on the UI thread (where the GPU lives) via
//! [`GpuContext::upload_external_frame`](crate::gpu::GpuContext::upload_external_frame),
//! and recomposites.
//!
//! Unlike the image [`Uploader`](crate::uploader::Uploader), submissions
//! are **latest-wins per node**: video only ever needs the most recent
//! frame, so a UI thread that falls behind the decoder drops the
//! intermediate frames instead of accumulating a backlog. A new frame (or
//! a clear) for a node simply replaces any not-yet-drained one.
//!
//! ## Threading
//! - **Submit side** (decoder thread): non-blocking insert + `wake()`.
//! - **Drain side** (UI thread in `about_to_wait`): pops via [`FrameSink::drain`],
//!   uploads / clears each, recomposites.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::WakeHandle;
use crate::node::NodeId;

/// A decoded frame awaiting upload: tightly-packed `width * height * 4`
/// RGBA8 (sRGB-encoded) bytes.
pub(crate) struct PendingFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One queued action for a node, latest-wins.
pub(crate) enum FrameMsg {
    /// Upload these pixels as the node's external texture.
    Frame(PendingFrame),
    /// Drop the node's external texture (back to compositing empty), e.g.
    /// the video stopped or the track has no Canvas.
    Clear,
}

/// Clone-anywhere external-frame submission token. Get one via
/// [`App::frame_sink`](crate::App::frame_sink).
pub struct FrameSink {
    /// Newest undrained action per node. Latest-wins — see module docs.
    pending: Mutex<HashMap<NodeId, FrameMsg>>,
    wake: Arc<WakeHandle>,
}

impl FrameSink {
    pub(crate) fn new(wake: Arc<WakeHandle>) -> Arc<Self> {
        Arc::new(Self {
            pending: Mutex::new(HashMap::new()),
            wake,
        })
    }

    /// Submit `node`'s latest frame: tightly-packed `width * height * 4`
    /// RGBA8 (sRGB) pixels. Replaces any pending action for `node`.
    /// Returns immediately (non-blocking) and wakes the event loop so the
    /// shell drains + recomposites even if otherwise parked.
    pub fn submit(&self, node: NodeId, width: u32, height: u32, rgba: Vec<u8>) {
        self.pending.lock().unwrap().insert(
            node,
            FrameMsg::Frame(PendingFrame {
                width,
                height,
                rgba,
            }),
        );
        self.wake.wake();
    }

    /// Drop `node`'s external texture on the next drain (the compositor
    /// falls back to whatever is behind the external layer). Replaces any
    /// pending frame for `node`. Wakes the loop so the clear is applied.
    pub fn clear(&self, node: NodeId) {
        self.pending.lock().unwrap().insert(node, FrameMsg::Clear);
        self.wake.wake();
    }

    /// Drain all pending actions. Called from `about_to_wait`.
    pub(crate) fn drain(&self) -> Vec<(NodeId, FrameMsg)> {
        std::mem::take(&mut *self.pending.lock().unwrap())
            .into_iter()
            .collect()
    }
}

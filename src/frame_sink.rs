//! Cross-thread external-frame (video) submission handle.
//!
//! [`FrameSink`] is the video-side cousin of [`crate::uploader::Uploader`]:
//! a cheaply-clonable token a decoder thread calls into. The shell drains it
//! inside `about_to_wait`, applies each command on the UI thread (where the
//! GPU lives), and recomposites.
//!
//! ## Two playback shapes
//! - **Resident loop** (default for a looping Canvas): the decoder uploads
//!   each frame of the first pass **once** via [`FrameSink::push_frame`],
//!   building a per-node set of GPU textures. Thereafter it loops by calling
//!   [`FrameSink::select`] with a frame index — a cheap view re-bind, **no
//!   pixel transfer across the bus**. A scene rebuild changes the node id, so
//!   the decoder issues [`FrameSink::migrate`] to move the resident set to
//!   the new id without re-uploading.
//! - **Streaming** (fallback): [`FrameSink::submit`] uploads a single frame
//!   into one reused texture every tick (a CPU→GPU copy per frame).
//!
//! ## Threading
//! - **Submit side** (decoder thread): non-blocking push + `wake()`.
//! - **Drain side** (UI thread in `about_to_wait`): [`FrameSink::drain`]
//!   takes the command queue in order, applies each, recomposites.
//!
//! Commands are a FIFO (not latest-wins): ordering matters — a `Push` or
//! `Migrate` must not be dropped by a later `Select`. `Select`/`Push` carry
//! no or one frame's pixels and are paced at the clip's fps, so the queue
//! stays short even if the UI briefly lags.

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

/// One queued action, applied in order on the UI thread.
pub(crate) enum FrameCmd {
    /// Streaming: upload `frame` into the node's single reused texture.
    Frame { node: NodeId, frame: PendingFrame },
    /// Resident loop: append `frame` to the node's GPU frame set and show
    /// it (the first-pass live build).
    Push { node: NodeId, frame: PendingFrame },
    /// Resident loop: bind the node's shown texture to `index` of its set.
    Select { node: NodeId, index: usize },
    /// Move a node's resident frame set to a new node id (scene rebuild
    /// reassigned the `.external()` node) without re-uploading.
    Migrate { old: NodeId, new: NodeId },
    /// Drop the node's external texture + any resident frame set.
    Clear { node: NodeId },
}

/// Clone-anywhere external-frame submission token. Get one via
/// [`App::frame_sink`](crate::App::frame_sink).
pub struct FrameSink {
    /// Pending commands in submission order. See module docs.
    queue: Mutex<Vec<FrameCmd>>,
    wake: Arc<WakeHandle>,
}

impl FrameSink {
    pub(crate) fn new(wake: Arc<WakeHandle>) -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(Vec::new()),
            wake,
        })
    }

    fn push_cmd(&self, cmd: FrameCmd) {
        self.queue.lock().unwrap().push(cmd);
        self.wake.wake();
    }

    /// Streaming upload of `node`'s latest frame into one reused texture
    /// (a CPU→GPU copy per call). Prefer [`push_frame`](Self::push_frame) +
    /// [`select`](Self::select) for a looping clip to avoid per-frame copies.
    pub fn submit(&self, node: NodeId, width: u32, height: u32, rgba: Vec<u8>) {
        self.push_cmd(FrameCmd::Frame {
            node,
            frame: PendingFrame { width, height, rgba },
        });
    }

    /// Append one frame to `node`'s resident GPU frame set (uploaded once)
    /// and show it. Used for each frame of the first decode pass.
    pub fn push_frame(&self, node: NodeId, width: u32, height: u32, rgba: Vec<u8>) {
        self.push_cmd(FrameCmd::Push {
            node,
            frame: PendingFrame { width, height, rgba },
        });
    }

    /// Show frame `index` of `node`'s resident set. Cheap — re-binds a
    /// texture view, no pixel transfer. Out-of-range indices are ignored.
    pub fn select(&self, node: NodeId, index: usize) {
        self.push_cmd(FrameCmd::Select { node, index });
    }

    /// Move `old`'s resident frame set to `new` (a rebuild reassigned the
    /// node id). No re-upload — the textures stay in VRAM.
    pub fn migrate(&self, old: NodeId, new: NodeId) {
        self.push_cmd(FrameCmd::Migrate { old, new });
    }

    /// Drop `node`'s external texture + resident set on the next drain.
    pub fn clear(&self, node: NodeId) {
        self.push_cmd(FrameCmd::Clear { node });
    }

    /// Drain all pending commands in order. Called from `about_to_wait`.
    pub(crate) fn drain(&self) -> Vec<FrameCmd> {
        std::mem::take(&mut *self.queue.lock().unwrap())
    }
}

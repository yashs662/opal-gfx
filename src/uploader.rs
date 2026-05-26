//! Cross-thread image upload handle.
//!
//! [`stage_image_*`] takes `&mut App` so it only works pre-`run`. Once
//! `App::run` consumes the App, there's no path back in for runtime
//! uploads — except via this handle.
//!
//! `Uploader` is the image-side cousin of [`WakeHandle`]: a cheaply-
//! clonable token that any thread (typically a worker fetching album
//! art) can call into. Each `upload_rgba(...)` enqueues bytes + a
//! completion callback and wakes the event loop. The shell drains the
//! queue inside `about_to_wait`, performs the actual atlas upload on
//! the UI thread (where the GPU lives), and fires the callback with
//! the resulting [`ImageHandle`].
//!
//! ## Threading
//!
//! - **Enqueue side** (any thread): non-blocking push + `wake()`.
//! - **Drain side** (UI thread inside `about_to_wait`): pops via
//!   [`Uploader::drain`], performs uploads, invokes callbacks. The
//!   callbacks run on the UI thread — keep them cheap (typical use is
//!   shipping the resolved handle over a channel back to a worker).
//!
//! ## When to use it
//!
//! Anything that arrives after first frame: album art, profile pics,
//! decoded SVGs from network responses, screenshots-of-self for
//! self-render, etc. For one-shot startup assets, prefer
//! [`App::stage_image_*`] — staged uploads skip the queue + callback
//! plumbing.

use std::sync::{Arc, Mutex};

use crate::WakeHandle;
use crate::gpu::ImageHandle;

/// Completion callback type. Receives `Some(handle)` on a successful
/// atlas upload, or `None` if the GPU isn't initialised yet, the
/// input dimensions are invalid, or the image is larger than the
/// atlas (after a rebuild-keeping-live retry).
pub type CompletionCb = Box<dyn FnOnce(Option<ImageHandle>) + Send + 'static>;

pub(crate) struct PendingUpload {
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
    pub cb: CompletionCb,
}

/// Clone-anywhere upload token. Get one via [`App::uploader`](crate::App::uploader).
pub struct Uploader {
    queue: Mutex<Vec<PendingUpload>>,
    wake: Arc<WakeHandle>,
}

impl Uploader {
    pub(crate) fn new(wake: Arc<WakeHandle>) -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(Vec::new()),
            wake,
        })
    }

    /// Queue raw `Rgba8UnormSrgb` pixels for upload on the next frame.
    /// `cb` is invoked from the UI thread with `Some(handle)` on
    /// success or `None` on failure.
    ///
    /// Returns immediately — does not block. Calls [`WakeHandle::wake`]
    /// so the shell drains the queue even if otherwise parked.
    pub fn upload_rgba<F>(&self, w: u32, h: u32, bytes: Vec<u8>, cb: F)
    where
        F: FnOnce(Option<ImageHandle>) + Send + 'static,
    {
        self.queue.lock().unwrap().push(PendingUpload {
            w,
            h,
            bytes,
            cb: Box::new(cb),
        });
        self.wake.wake();
    }

    /// Drain all pending uploads. Called from `about_to_wait`.
    pub(crate) fn drain(&self) -> Vec<PendingUpload> {
        std::mem::take(&mut self.queue.lock().unwrap())
    }
}

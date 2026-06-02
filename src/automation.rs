//! Scripted-input + screenshot harness — **DEBUG ONLY, REMOVABLE**.
//!
//! Gated behind the `automation` cargo feature (off by default, never in a
//! ship build). Lets a config-driven script drive the app like a user —
//! move the cursor, click, hover, scroll — and capture screenshots at
//! deterministic points, so issues can be reproduced and fixes verified
//! without a human at the mouse.
//!
//! Synthetic input is fed through the *same* internal handlers as real
//! winit events (see `App::inject_*` in `app.rs`), so hover signals,
//! dwell timers, hit-testing, and `on_click` all behave exactly as they
//! would for a real user. The driver is ticked from `App::about_to_wait`.
//!
//! ## Removing this harness before ship
//! 1. delete this file (`src/automation.rs`);
//! 2. delete the `automation = []` feature in `Cargo.toml`;
//! 3. delete every `#[cfg(feature = "automation")]` block in `src/app.rs`
//!    (search for `automation`) and the `pub mod automation` /
//!    re-export in `src/lib.rs`;
//! 4. on the consumer side, drop the `--config` script branch.
//! Nothing else references it.

use std::path::PathBuf;
use std::time::{Duration, Instant};

/// One scripted action. Coordinates are **physical** pixels (top-left
/// origin), matching the surface the screenshots capture.
#[derive(Clone, Debug)]
pub enum Step {
    /// Idle for `Duration` — lets async work (art loads, worker
    /// responses, animations) land before the next step.
    Wait(Duration),
    /// Render the current frame and write a PNG to `PathBuf`.
    Screenshot(PathBuf),
    /// Move the cursor to `[x, y]` (updates hover, fires dwell arming).
    MoveMouse([f32; 2]),
    /// Move to `[x, y]`, then press + release left button (fires
    /// `on_click` on the node under the cursor).
    Click([f32; 2]),
    /// Move to `[x, y]` and dwell there for `Duration` (so hover tints +
    /// hover-dwell tooltips fire). Scheduling-only; the dwell pump runs
    /// during the wait.
    Hover([f32; 2], Duration),
    /// Move to `[x, y]` and scroll by `[dx, dy]` wheel lines.
    Scroll([f32; 2], [f32; 2]),
    /// Press at `from`, move to `to`, release (drag a slider/splitter).
    Drag([f32; 2], [f32; 2]),
}

/// An ordered list of [`Step`]s. Built by the consumer (e.g. from a TOML
/// config) and handed to `App::automation(...)`.
#[derive(Clone, Debug, Default)]
pub struct Script {
    pub steps: Vec<Step>,
}

impl Script {
    pub fn new(steps: Vec<Step>) -> Self {
        Self { steps }
    }
}

/// Driver state held on the `App` while a script runs. Advances one step
/// per "due" tick; instant steps reschedule immediately, `Wait`/`Hover`
/// reschedule after their delay. When the steps run out the driver asks
/// the app to exit.
pub struct AutomationState {
    steps: Vec<Step>,
    idx: usize,
    /// When the current step becomes due. `None` → run immediately.
    next_at: Option<Instant>,
}

impl AutomationState {
    pub fn new(script: Script) -> Self {
        Self {
            steps: script.steps,
            idx: 0,
            next_at: None,
        }
    }

    pub fn finished(&self) -> bool {
        self.idx >= self.steps.len()
    }

    pub fn due(&self, now: Instant) -> bool {
        self.next_at.map(|t| now >= t).unwrap_or(true)
    }

    /// The current step (cloned so the borrow on `self` drops before the
    /// driver mutates `App`).
    pub fn current(&self) -> Option<Step> {
        self.steps.get(self.idx).cloned()
    }

    /// Advance past the current step; the next is due immediately.
    pub fn advance_now(&mut self, now: Instant) {
        self.idx += 1;
        self.next_at = Some(now);
    }

    /// Advance past the current step; the next is due after `delay`.
    pub fn advance_after(&mut self, now: Instant, delay: Duration) {
        self.idx += 1;
        self.next_at = Some(now + delay);
    }

    /// The scheduled wake time (so the event loop doesn't park past it).
    pub fn next_at(&self) -> Option<Instant> {
        self.next_at
    }
}

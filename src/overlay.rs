//! Modal overlay (popup / dialog) — a self-contained, reusable widget.
//!
//! Why this exists: a modal is more than "a node positioned on top". It
//! has to (1) render above everything, (2) **block all input** to the
//! content beneath it — hover, click, scroll — not just capture its own
//! clicks, (3) animate in and out, and (4) cost nothing when closed. Hand
//! -rolling that per app is the "tape and glue" trap: you end up wiring a
//! scrim node, a stable opacity `Signal`, a `Timeline` tween key, and the
//! open/close/dismiss plumbing by hand at every call site, and it's easy
//! to get the input-blocking or the animation lifecycle subtly wrong.
//!
//! [`Overlay`] owns all of that. You create one (once, stored in your app
//! state), call [`Overlay::render`] each frame with a closure that builds
//! the panel interior, and drive it with [`Overlay::open`] /
//! [`Overlay::close`] / [`Overlay::toggle`] from button handlers. The
//! tween key and opacity signal are private — you never see them.
//!
//! ## Multiple overlays / stacking
//! Each `Overlay` is independent (its own opacity signal + auto-allocated
//! tween key), so any number can be open at once. They **stack by render
//! order**: render them in the order you want them layered (last = on
//! top). Each one's full-window scrim blocks input to everything rendered
//! before it, so the topmost overlay owns input until dismissed, then the
//! one below takes over — standard modal-stack behaviour, no extra
//! bookkeeping. (Relies on topmost-wins hit-testing; see `input::hit_test`.)
//!
//! ## Cost
//! Closed (opacity 0) → the scrim's [`crate::scene::NodeBuilderRef::opacity_bind`]
//! marks the subtree invisible, so flatten skips it entirely: no draw, no
//! hit entries, no per-frame work. Open + idle → one full-window scrim
//! quad + the panel; the loop parks (0% CPU) until the next input. Only
//! the ~160 ms fade animates.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use winit::window::CursorIcon;

use crate::anim::{Curve, Timeline};
use crate::layout::{Align, Justify, Len};
use crate::scene::Scene;
use crate::signal::Signal;

/// Base of the tween-key range handed out to engine widgets. Kept far
/// above the small hand-picked keys apps use for their own tweens (those
/// are conventionally in the low range, e.g. `0x0001_xxxx`) so the two
/// can never collide.
const WIDGET_TWEEN_BASE: u32 = 0xF000_0000;
static NEXT_WIDGET_KEY: AtomicU32 = AtomicU32::new(WIDGET_TWEEN_BASE);

/// Default morph spring — a quick, decisive move into the new height with a
/// damping ratio just under critical (ζ≈0.77) for a barely-there bounce at
/// the tail. Snappy, not springy.
const MORPH_SPRING: Curve = Curve::Spring {
    stiffness: 380.0,
    damping: 30.0,
};
const MORPH_MS: u64 = 450;

/// Optional "morph" behaviour: the overlay owns a panel-height signal and
/// springs it between `collapsed` and a caller-set target on open / close /
/// dismiss, so a modal grows open and shrinks closed with no per-app
/// plumbing. Bound by the panel via [`crate::scene::NodeBuilderRef::height_px_bind`].
#[derive(Clone)]
struct Morph {
    /// Panel height (logical px). Springs collapsed ↔ target.
    height: Signal<f32>,
    /// The shrunk height the panel opens *out from* and closes back *to*.
    collapsed: f32,
    /// Own timeline key (distinct from the opacity fade) so re-targets
    /// interrupt smoothly.
    key: u32,
    curve: Curve,
    duration: Duration,
}

/// A modal overlay. Cheap to clone (`Rc`-backed signals + a few `Copy`
/// fields) — clone it into button handlers freely.
#[derive(Clone)]
pub struct Overlay {
    /// Group opacity 0..=1, tweened on open/close and bound to the scrim
    /// root (cascades to the whole subtree). Private: the only way to move
    /// it is open/close/toggle/reset.
    opacity: Signal<f32>,
    /// Stable, process-unique timeline key for this overlay's fade. Auto-
    /// allocated in [`Overlay::new`] so callers never pick magic numbers
    /// or risk collisions.
    tween_key: u32,
    fade: Duration,
    curve: Curve,
    /// Vertical placement of the panel within the scrim. `Center` (default)
    /// is the classic dialog; `Start` top-anchors it (e.g. a search modal
    /// that drops from the top bar). Horizontal is always centred.
    anchor: Justify,
    /// Height-morph state, if [`Self::with_morph`] enabled it. The overlay
    /// springs this on open/close/dismiss so the panel grows/shrinks; `None`
    /// = a plain fade modal.
    morph: Option<Morph>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    /// Create a closed overlay with a 160 ms ease-in-out fade.
    pub fn new() -> Self {
        Self {
            opacity: Signal::new(0.0),
            tween_key: NEXT_WIDGET_KEY.fetch_add(1, Ordering::Relaxed),
            fade: Duration::from_millis(160),
            curve: Curve::EaseInOut,
            anchor: Justify::Center,
            morph: None,
        }
    }

    /// Override the fade duration (applies to both directions).
    pub fn with_fade(mut self, fade: Duration) -> Self {
        self.fade = fade;
        self
    }

    /// Anchor the panel vertically — `Justify::Start` top-anchors it
    /// (default is `Center`). Horizontal centring is unchanged.
    pub fn with_anchor(mut self, anchor: Justify) -> Self {
        self.anchor = anchor;
        self
    }

    /// Override the fade easing curve.
    pub fn with_curve(mut self, curve: Curve) -> Self {
        self.curve = curve;
        self
    }

    /// Enable height-morph: the overlay owns a panel-height signal that
    /// springs between `collapsed` and a target on open / close / dismiss, so
    /// the modal grows open and shrinks closed by itself. Bind the panel's
    /// height to [`Self::morph_height`] and drive it with [`Self::morph_open`]
    /// / [`Self::morph_close`] / [`Self::morph_to`]; the built-in scrim
    /// dismiss shrinks it back with no extra wiring. The height starts at
    /// `collapsed` (never seen while closed).
    pub fn with_morph(mut self, collapsed: f32) -> Self {
        self.morph = Some(Morph {
            height: Signal::new(collapsed),
            collapsed,
            key: NEXT_WIDGET_KEY.fetch_add(1, Ordering::Relaxed),
            curve: MORPH_SPRING,
            duration: Duration::from_millis(MORPH_MS),
        });
        self
    }

    /// The panel-height signal to bind (`height_px_bind`). Only meaningful
    /// when [`Self::with_morph`] was used; otherwise a throwaway constant.
    pub fn morph_height(&self) -> Signal<f32> {
        match &self.morph {
            Some(m) => m.height.clone(),
            None => Signal::new(0.0),
        }
    }

    /// Fade in **and** spring the panel height up to `target` — the open
    /// morph. Interruptible: reopening mid-close reverses smoothly.
    pub fn morph_open(&self, timeline: &mut Timeline, now: Instant, target: f32) {
        self.open(timeline, now);
        self.spring_height(timeline, now, target);
    }

    /// Fade out **and** spring the panel height back to `collapsed` — the
    /// close morph, mirroring the open. Same as a scrim dismiss.
    pub fn morph_close(&self, timeline: &mut Timeline, now: Instant) {
        self.close(timeline, now);
        if let Some(m) = &self.morph {
            self.spring_height(timeline, now, m.collapsed);
        }
    }

    /// Re-spring the panel height to a new `target` while open (e.g. content
    /// grew/shrank). No opacity change.
    pub fn morph_to(&self, timeline: &mut Timeline, now: Instant, target: f32) {
        self.spring_height(timeline, now, target);
    }

    /// Start the height spring toward `target` on the morph's own key.
    fn spring_height(&self, timeline: &mut Timeline, now: Instant, target: f32) {
        if let Some(m) = &self.morph {
            timeline.start(m.key, m.height.clone(), target, m.curve, m.duration, now);
        }
    }

    /// Whether the overlay is logically open (more than half faded in).
    /// Used by [`Self::toggle`] and for "is a modal up?" checks.
    pub fn is_open(&self) -> bool {
        self.opacity.get() > 0.5
    }

    /// Fade in. Idempotent + interruptible: re-targets the same tween, so
    /// opening mid-close smoothly reverses from wherever it is.
    pub fn open(&self, timeline: &mut Timeline, now: Instant) {
        timeline.start(
            self.tween_key,
            self.opacity.clone(),
            1.0,
            self.curve,
            self.fade,
            now,
        );
    }

    /// Fade out. Same interruptible semantics as [`Self::open`].
    pub fn close(&self, timeline: &mut Timeline, now: Instant) {
        timeline.start(
            self.tween_key,
            self.opacity.clone(),
            0.0,
            self.curve,
            self.fade,
            now,
        );
    }

    /// Open if closed, close if open.
    pub fn toggle(&self, timeline: &mut Timeline, now: Instant) {
        if self.is_open() {
            self.close(timeline, now);
        } else {
            self.open(timeline, now);
        }
    }

    /// Snap shut with no animation. For when the surrounding context is
    /// being torn down anyway (e.g. signing out / changing views) and a
    /// fade would be pointless or run against an unmounted tree.
    pub fn reset(&self) {
        self.opacity.set(0.0);
    }

    /// Snap fully open with no animation — e.g. restoring a modal that
    /// was open when the app last closed, where a fade-in would look like
    /// a glitch on launch.
    pub fn open_instant(&self) {
        self.opacity.set(1.0);
    }

    /// Render the modal. Builds a full-window scrim (dimmed by
    /// `scrim_color`, click-to-dismiss, input-blocking) with the panel
    /// `content` centred on top; the whole thing fades via the owned
    /// opacity. `content` builds the panel interior — style it however you
    /// like; the overlay handles the scrim, dismissal, centring and the
    /// click-absorbing host so panel clicks never leak to the dismiss
    /// handler.
    ///
    /// **Call this last** in your scene (after all normal content) so it
    /// layers on top. For stacked modals, render them back-to-front.
    /// Always safe to call every frame: when closed it's skipped by the
    /// opacity-visibility gate.
    pub fn render<F: FnOnce(&mut Scene)>(&self, s: &mut Scene, scrim_color: [f32; 4], content: F) {
        let opacity = self.opacity.clone();
        let key = self.tween_key;
        let fade = self.fade;
        let curve = self.curve;
        // Scrim dismiss shrinks the height back too (if morphing) — captured
        // so the click handler is self-contained (no app callback needed).
        let morph = self.morph.clone();
        s.col(())
            .abs(0.0, 0.0)
            .w(Len::Fill)
            .h(Len::Fill)
            .rgba(
                scrim_color[0],
                scrim_color[1],
                scrim_color[2],
                scrim_color[3],
            )
            // Fades the whole subtree; at ~0 it also drops the scrim from
            // flatten, so a closed modal neither draws nor eats input.
            .opacity_bind(opacity.clone())
            .align(Align::Center)
            .justify(self.anchor)
            // Click outside the panel dismisses. (Topmost-wins hit-testing
            // means this only fires when the cursor isn't over the panel.)
            .on_click(move |ctx| {
                ctx.timeline
                    .start(key, opacity.clone(), 0.0, curve, fade, ctx.now);
                // Inverse-morph the height back to collapsed alongside the
                // fade — self-contained, so a click-outside close animates
                // with no app-side plumbing.
                if let Some(m) = &morph {
                    ctx.timeline.start(
                        m.key,
                        m.height.clone(),
                        m.collapsed,
                        m.curve,
                        m.duration,
                        ctx.now,
                    );
                }
            })
            // The dim area is click-to-dismiss but shouldn't read as a
            // button — keep the normal arrow over it (overrides the
            // auto-pointer the `on_click` would otherwise trigger).
            .cursor(CursorIcon::Default)
            .child(|scrim| {
                // Absorbing host: a click anywhere on the panel (including
                // its empty padding, which isn't itself a hit target)
                // lands here instead of falling through to the scrim's
                // dismiss handler. Same arrow-cursor reasoning as the
                // scrim — only the real buttons inside should show pointer.
                scrim
                    .col(())
                    .on_click(|_| {})
                    .cursor(CursorIcon::Default)
                    .child(content);
            });
    }
}

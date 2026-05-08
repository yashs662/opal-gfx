//! Animation primitives — generic over `Lerp`.
//!
//! Stage-1 scope: tween any `Signal<T: Lerp>` over a timeline. Curves
//! cover the four shapes a real UI app needs — linear, ease-in-out,
//! cubic-bezier (css-style timing), and a 1-DOF damped spring. The
//! `Lerp` impls cover scalars (`f32`), 2-vectors (`[f32; 2]`) and rgba
//! colors (`[f32; 4]`); add more as the type set grows.
//!
//! Vector tweens go through the same code path as scalar tweens — the
//! curve produces a normalized 0..1 progress and `Lerp::lerp` does the
//! per-component interpolation. The timeline stores tweens behind a
//! trait object (`Box<dyn TweenDyn>`) so the same `Vec` can hold mixed
//! types; tween counts are small (low dozens), so the indirection
//! is cheap. Upgrade to a `SmallVec` is a stage-2 concern.
//!
//! The timeline does not touch winit or the tree. It exposes
//! `tick(now)` which returns a `TickResult` containing `updated` (any
//! signal value actually changed) and `next_deadline` (the recommended
//! `ControlFlow::WaitUntil` target). Caller wiring:
//!
//! ```ignore
//! let res = timeline.tick(Instant::now());
//! if res.updated {
//!     refresh_derived_values();
//!     flush_tree();
//!     request_redraw();
//! }
//! match res.next_deadline {
//!     Some(deadline) => event_loop.set_control_flow(WaitUntil(deadline)),
//!     None => event_loop.set_control_flow(Wait),
//! }
//! ```

use std::time::{Duration, Instant};

use crate::signal::Signal;

/// Linear interpolation trait. All tweenable types must impl this.
///
/// `lerp(self, to, t)` returns `self * (1 - t) + to * t`. `t` is
/// expected in `[0, 1]` but implementations should not panic outside
/// that range — extrapolation is fine.
pub trait Lerp: Copy + PartialEq + 'static {
    fn lerp(self, to: Self, t: f32) -> Self;
}

impl Lerp for f32 {
    fn lerp(self, to: Self, t: f32) -> Self {
        self + (to - self) * t
    }
}

impl Lerp for [f32; 2] {
    fn lerp(self, to: Self, t: f32) -> Self {
        [self[0].lerp(to[0], t), self[1].lerp(to[1], t)]
    }
}

impl Lerp for [f32; 4] {
    fn lerp(self, to: Self, t: f32) -> Self {
        [
            self[0].lerp(to[0], t),
            self[1].lerp(to[1], t),
            self[2].lerp(to[2], t),
            self[3].lerp(to[3], t),
        ]
    }
}

/// Easing curves. All stored by value — `Copy`.
///
/// `Spring` ignores the tween duration except as a hard safety cap;
/// the motion is driven by elapsed seconds through a closed-form
/// damped harmonic oscillator with unit mass. Use stiffness ~150 and
/// damping ~18 for a snappy UI spring, or ~80/12 for a softer one.
#[derive(Copy, Clone, Debug)]
pub enum Curve {
    Linear,
    EaseInOut,
    /// CSS-style cubic bezier timing: `[x1, y1, x2, y2]`. Control
    /// points (0,0) and (1,1) are implicit.
    CubicBezier([f32; 4]),
    Spring {
        stiffness: f32,
        damping: f32,
    },
}

impl Curve {
    /// Evaluate at normalized `t ∈ [0,1]`. Panics for `Spring`, which
    /// is evaluated by elapsed seconds inside `Tween::sample` instead.
    pub fn eval(&self, t: f32) -> f32 {
        match *self {
            Curve::Linear => t.clamp(0.0, 1.0),
            Curve::EaseInOut => {
                let t = t.clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            Curve::CubicBezier([x1, y1, x2, y2]) => {
                let t = t.clamp(0.0, 1.0);
                cubic_bezier_y_at_x(x1, y1, x2, y2, t)
            }
            Curve::Spring { .. } => unreachable!("spring sampled via elapsed seconds"),
        }
    }
}

/// Solve `Bx(t) = x` for `t` via Newton's method, then evaluate `By(t)`.
///
/// Control points are `(0,0)`, `(x1,y1)`, `(x2,y2)`, `(1,1)` — i.e.
/// standard CSS timing function form. 8 iterations is plenty: the
/// function is smooth and almost always converges in 3–4.
fn cubic_bezier_y_at_x(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    let bx = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * t * x1 + 3.0 * it * t * t * x2 + t * t * t
    };
    let by = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * t * y1 + 3.0 * it * t * t * y2 + t * t * t
    };
    let dbx = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * x1 + 6.0 * it * t * (x2 - x1) + 3.0 * t * t * (1.0 - x2)
    };
    let mut t = x;
    for _ in 0..8 {
        let err = bx(t) - x;
        if err.abs() < 1e-4 {
            break;
        }
        let slope = dbx(t);
        if slope.abs() < 1e-6 {
            break;
        }
        t = (t - err / slope).clamp(0.0, 1.0);
    }
    by(t)
}

/// Closed-form unit-step response of a 1-DOF mass-spring-damper with
/// unit mass, initial position 0, initial velocity 0, target 1.
/// Returns `(position, velocity)` at elapsed time `t`. `pub(crate)` so
/// the scroll path can drive bounce-back through the same spring math
/// the timeline uses (instead of inventing a separate exponential).
pub(crate) fn spring_eval(k: f32, c: f32, t: f32) -> (f32, f32) {
    let w0 = k.max(1e-6).sqrt();
    let zeta = c / (2.0 * w0);
    if (zeta - 1.0).abs() < 1e-4 {
        // Critically damped.
        let e = (-w0 * t).exp();
        let x = 1.0 - e * (1.0 + w0 * t);
        let v = e * (w0 * w0 * t);
        (x, v)
    } else if zeta < 1.0 {
        // Underdamped — the UI default.
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let a = zeta * w0;
        let e = (-a * t).exp();
        let cos = (wd * t).cos();
        let sin = (wd * t).sin();
        let x = 1.0 - e * (cos + (a / wd) * sin);
        let v = (w0 * w0 / wd) * e * sin;
        (x, v)
    } else {
        // Overdamped.
        let disc = (zeta * zeta - 1.0).sqrt();
        let r1 = -w0 * (zeta - disc);
        let r2 = -w0 * (zeta + disc);
        let a1 = -r2 / (r1 - r2);
        let a2 = r1 / (r1 - r2);
        let x = 1.0 - a1 * (r1 * t).exp() - a2 * (r2 * t).exp();
        let v = -a1 * r1 * (r1 * t).exp() - a2 * r2 * (r2 * t).exp();
        (x, v)
    }
}

/// Evaluate a curve at the elapsed time of a tween. Returns
/// `(progress, done)` where `progress` is normalized 0..1 (or settled
/// at 1.0 for spring). Centralized so scalar and vector tweens share
/// the same path.
fn eval_progress(curve: Curve, start: Instant, duration: Duration, now: Instant) -> (f32, bool) {
    let elapsed = now.saturating_duration_since(start);
    match curve {
        Curve::Spring { stiffness, damping } => {
            let t = elapsed.as_secs_f32();
            let (x, vel) = spring_eval(stiffness, damping, t);
            let settled = (x - 1.0).abs() < 1e-3 && vel.abs() < 1e-3;
            (x, settled || elapsed >= duration)
        }
        _ => {
            let total = duration.as_secs_f32().max(1e-6);
            let t_norm = elapsed.as_secs_f32() / total;
            let done = t_norm >= 1.0;
            let eased = curve.eval(t_norm.clamp(0.0, 1.0));
            (eased, done)
        }
    }
}

/// One in-flight tween. Generic over the value type so colors,
/// positions, scalars, and future quaternions all share the same
/// machinery.
#[derive(Clone, Debug)]
pub struct Tween<T: Lerp> {
    /// User-facing tag. `Timeline::start` replaces any existing tween
    /// with the same key, which is how hover-on/hover-off interrupt
    /// each other smoothly.
    pub key: u32,
    pub signal: Signal<T>,
    pub from: T,
    pub to: T,
    pub curve: Curve,
    pub start: Instant,
    pub duration: Duration,
}

impl<T: Lerp> Tween<T> {
    /// Returns `(value, done)` at `now`. The caller snaps to `to` when
    /// `done` is true.
    fn sample(&self, now: Instant) -> (T, bool) {
        let (progress, done) = eval_progress(self.curve, self.start, self.duration, now);
        (self.from.lerp(self.to, progress), done)
    }
}

/// Type-erased tween interface used by `Timeline` so it can hold
/// tweens of mixed value types in a single `Vec`.
trait TweenDyn {
    fn key(&self) -> u32;
    /// Advance to `now`. Returns `(updated, done)` — `updated` is true
    /// if the underlying signal's value actually changed.
    fn step(&mut self, now: Instant) -> (bool, bool);
}

impl<T: Lerp> TweenDyn for Tween<T> {
    fn key(&self) -> u32 {
        self.key
    }
    fn step(&mut self, now: Instant) -> (bool, bool) {
        let (val, done) = self.sample(now);
        let final_val = if done { self.to } else { val };
        let updated = self.signal.set(final_val);
        (updated, done)
    }
}

/// Result of one `Timeline::tick`.
#[derive(Copy, Clone, Debug, Default)]
pub struct TickResult {
    /// True if any tween's signal actually changed value.
    pub updated: bool,
    /// Recommended wake-up time. `None` when the timeline is idle —
    /// the caller should fall back to `ControlFlow::Wait`.
    pub next_deadline: Option<Instant>,
}

/// Collection of running tweens. The tick cadence is fixed at ~16 ms
/// (≈60 Hz). Adaptive vsync-driven cadence is a future option.
pub struct Timeline {
    tweens: Vec<Box<dyn TweenDyn>>,
    tick_interval: Duration,
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            tweens: Vec::new(),
            tick_interval: Duration::from_millis(16),
        }
    }

    pub fn active(&self) -> bool {
        !self.tweens.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tweens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tweens.is_empty()
    }

    pub fn tick_interval(&self) -> Duration {
        self.tick_interval
    }

    pub fn set_tick_interval(&mut self, dt: Duration) {
        self.tick_interval = dt;
    }

    /// Start (or restart) a tween for `key`. `from` is read from the
    /// signal's current value so mid-flight interrupts land smoothly.
    /// No-op if `from == to`.
    pub fn start<T: Lerp>(
        &mut self,
        key: u32,
        signal: Signal<T>,
        to: T,
        curve: Curve,
        duration: Duration,
        now: Instant,
    ) {
        self.tweens.retain(|t| t.key() != key);
        let from = signal.get();
        if from == to {
            return;
        }
        self.tweens.push(Box::new(Tween {
            key,
            signal,
            from,
            to,
            curve,
            start: now,
            duration,
        }));
    }

    pub fn stop(&mut self, key: u32) {
        self.tweens.retain(|t| t.key() != key);
    }

    pub fn clear(&mut self) {
        self.tweens.clear();
    }

    /// Advance every tween to `now`. Writes to each tween's signal
    /// via `Signal::set` (which deduplicates no-op writes). Completed
    /// tweens snap to `to` and are removed.
    pub fn tick(&mut self, now: Instant) -> TickResult {
        let mut updated = false;
        self.tweens.retain_mut(|tw| {
            let (changed, done) = tw.step(now);
            if changed {
                updated = true;
            }
            !done
        });
        let next_deadline = if self.tweens.is_empty() {
            None
        } else {
            Some(now + self.tick_interval)
        };
        TickResult {
            updated,
            next_deadline,
        }
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_end_to_end() {
        let s = Signal::new(0.0f32);
        let mut tl = Timeline::new();
        let now = Instant::now();
        tl.start(
            0,
            s.clone(),
            1.0,
            Curve::Linear,
            Duration::from_millis(100),
            now,
        );
        assert!(tl.active());
        let mid = tl.tick(now + Duration::from_millis(50));
        assert!(mid.updated);
        assert!((s.get() - 0.5).abs() < 1e-3);
        let end = tl.tick(now + Duration::from_millis(120));
        assert!(end.updated);
        assert!(!tl.active());
        assert_eq!(s.get(), 1.0);
    }

    #[test]
    fn ease_in_out_monotonic() {
        let c = Curve::EaseInOut;
        let mut prev = -1.0;
        for i in 0..=10 {
            let y = c.eval(i as f32 / 10.0);
            assert!(y >= prev);
            prev = y;
        }
        assert!((c.eval(0.0)).abs() < 1e-4);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cubic_bezier_endpoints() {
        let c = Curve::CubicBezier([0.42, 0.0, 0.58, 1.0]);
        assert!(c.eval(0.0).abs() < 1e-3);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn spring_settles_near_target() {
        // Underdamped spring should hit ~1.0 within a few seconds.
        let (x, _v) = spring_eval(180.0, 20.0, 2.0);
        assert!((x - 1.0).abs() < 1e-2);
    }

    #[test]
    fn restart_picks_up_current_value() {
        let s = Signal::new(0.0f32);
        let mut tl = Timeline::new();
        let now = Instant::now();
        tl.start(
            0,
            s.clone(),
            1.0,
            Curve::Linear,
            Duration::from_millis(100),
            now,
        );
        tl.tick(now + Duration::from_millis(40));
        let mid_val = s.get();
        assert!(mid_val > 0.3 && mid_val < 0.5);
        // Reverse mid-flight. New tween must read current signal value
        // as `from` so the reversal is smooth (no jump back to 1.0).
        tl.start(
            0,
            s.clone(),
            0.0,
            Curve::Linear,
            Duration::from_millis(100),
            now + Duration::from_millis(40),
        );
        assert_eq!(tl.len(), 1);
        // After half the new tween's duration we should be roughly halfway
        // between mid_val and 0 — proves the new `from` was `mid_val`.
        tl.tick(now + Duration::from_millis(90));
        let after = s.get();
        let expected = mid_val * 0.5;
        assert!((after - expected).abs() < 0.05, "after={after} expected~{expected}");
    }

    #[test]
    fn vector_tween_lerps_components() {
        let s = Signal::new([0.0_f32, 0.0, 0.0, 1.0]);
        let mut tl = Timeline::new();
        let now = Instant::now();
        tl.start(
            0,
            s.clone(),
            [1.0, 0.5, 0.25, 1.0],
            Curve::Linear,
            Duration::from_millis(100),
            now,
        );
        tl.tick(now + Duration::from_millis(50));
        let v = s.get();
        assert!((v[0] - 0.5).abs() < 1e-3);
        assert!((v[1] - 0.25).abs() < 1e-3);
        assert!((v[2] - 0.125).abs() < 1e-3);
        assert_eq!(v[3], 1.0);
        let end = tl.tick(now + Duration::from_millis(120));
        assert!(end.updated);
        assert_eq!(s.get(), [1.0, 0.5, 0.25, 1.0]);
    }
}

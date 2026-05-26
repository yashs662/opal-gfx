//! Reactive prop bindings: `Source`, `Computed<T>`, `Bind<T>`,
//! `AnimatedBind<T>`.
//!
//! The reactive layer sits on top of the bare `Signal<T>` primitive
//! and powers the declarative builder API. Three concepts:
//!
//! 1. **Source** — anything with a value and a monotonic version. Both
//!    [`Signal`] and [`Computed`] implement it, so a computed can
//!    depend on another computed.
//! 2. **Computed** — cached derivation with explicitly declared
//!    dependencies. Recomputes lazily on the next `read`/`version`
//!    call when *any* dep's version has bumped. No global graph; a
//!    `Computed` only inspects the deps you handed it at construction.
//! 3. **Bind / AnimatedBind** — prop wrappers (in [`crate::bind`])
//!    that the builder API accepts.
//!
//! Why explicit deps? The thread-local "tracked read" approach used
//! by some signal libraries hides surprising recomputation costs and
//! breaks once you drop into `if`/`match` arms. Explicit tuples keep
//! the dependency graph local and grep-able. Macro-generated impls
//! cover tuples of size 1..=8, which is more than enough for stage 1.
//!
//! ```ignore
//! let lit     = Signal::new(false);
//! let hover   = Signal::new(false);
//! let pressed = Signal::new(false);
//! let color = Computed::new((lit.clone(), hover.clone(), pressed.clone()),
//!     |(l, h, p)| {
//!         let base = if l { GREEN } else { PINK };
//!         if p { darken(base) } else if h { brighten(base) } else { base }
//!     });
//! assert_eq!(color.read(), PINK);
//! hover.set(true);
//! assert_eq!(color.read(), brighten(PINK));
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use crate::anim::{Curve, Lerp};
use crate::gpu::ImageHandle;
use crate::signal::{Signal, TextSignal};

/// Anything with a current value and a monotonic version counter.
/// `Signal<T>` and `Computed<T>` both implement it so they compose
/// uniformly inside `Computed::new`'s dep tuple.
pub trait Source: Clone {
    type Value: Copy + PartialEq + 'static;
    fn version(&self) -> u64;
    fn read(&self) -> Self::Value;
}

impl<T: Copy + PartialEq + 'static> Source for Signal<T> {
    type Value = T;
    fn version(&self) -> u64 {
        Signal::version(self)
    }
    fn read(&self) -> T {
        Signal::get(self)
    }
}

impl<T: Copy + PartialEq + 'static> Source for Computed<T> {
    type Value = T;
    fn version(&self) -> u64 {
        Computed::version(self)
    }
    fn read(&self) -> T {
        Computed::read(self)
    }
}

/// Tuple of dependencies passed to `Computed::new`. Implemented for
/// tuples of arity 1..=8 by the macro below. `versions` returns one
/// `u64` per dep, in tuple order.
pub trait DepTuple: Clone + 'static {
    type Values;
    fn versions(&self) -> Vec<u64>;
    fn read(&self) -> Self::Values;
}

macro_rules! impl_dep_tuple {
    ($($T:ident => $idx:tt),+) => {
        impl<$($T: Source + 'static),+> DepTuple for ($($T,)+) {
            type Values = ($($T::Value,)+);
            fn versions(&self) -> Vec<u64> {
                vec![$(self.$idx.version()),+]
            }
            fn read(&self) -> Self::Values {
                ($(self.$idx.read(),)+)
            }
        }
    };
}

impl_dep_tuple!(A => 0);
impl_dep_tuple!(A => 0, B => 1);
impl_dep_tuple!(A => 0, B => 1, C => 2);
impl_dep_tuple!(A => 0, B => 1, C => 2, D => 3);
impl_dep_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4);
impl_dep_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5);
impl_dep_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5, G => 6);
impl_dep_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5, G => 6, H => 7);

/// Cached, lazily-recomputed derivation of a value from a tuple of
/// `Source` deps. Cheap to clone (refcounted).
///
/// Recomputation policy: each `read()` (or `version()`) inspects the
/// current dep versions. If any have changed since the last successful
/// recompute, the user closure runs once and the cached value is
/// updated. The computed's *own* version only advances when the new
/// value differs from the previous one — this keeps downstream
/// computeds quiet across no-op recomputes.
pub struct Computed<T: Copy + PartialEq + 'static> {
    inner: Rc<ComputedInner<T>>,
}

struct ComputedInner<T> {
    cached: Cell<T>,
    last_versions: RefCell<Vec<u64>>,
    self_version: Cell<u64>,
    /// Cheap: one `version()` call per dep, no user code.
    dep_versions: Box<dyn Fn() -> Vec<u64>>,
    /// Expensive: reads deps and runs the user closure.
    recompute: Box<dyn Fn() -> T>,
}

impl<T: Copy + PartialEq + 'static> Computed<T> {
    /// Build a new computed. `f` is invoked once eagerly to seed the
    /// cache so the first `read()` is free.
    pub fn new<D, F>(deps: D, f: F) -> Self
    where
        D: DepTuple,
        F: Fn(D::Values) -> T + 'static,
    {
        let deps = Rc::new(deps);
        let f = Rc::new(f);

        let initial_versions = deps.versions();
        let initial_value = f(deps.read());

        let deps_v = Rc::clone(&deps);
        let dep_versions: Box<dyn Fn() -> Vec<u64>> = Box::new(move || deps_v.versions());

        let deps_r = Rc::clone(&deps);
        let f_r = Rc::clone(&f);
        let recompute: Box<dyn Fn() -> T> = Box::new(move || f_r(deps_r.read()));

        Self {
            inner: Rc::new(ComputedInner {
                cached: Cell::new(initial_value),
                last_versions: RefCell::new(initial_versions),
                self_version: Cell::new(0),
                dep_versions,
                recompute,
            }),
        }
    }

    /// Read the current value, recomputing lazily if any dep moved.
    pub fn read(&self) -> T {
        self.refresh();
        self.inner.cached.get()
    }

    /// Returns the version counter, refreshing first so callers
    /// observing this number see a coherent (value, version) pair.
    pub fn version(&self) -> u64 {
        self.refresh();
        self.inner.self_version.get()
    }

    fn refresh(&self) {
        let current = (self.inner.dep_versions)();
        let stale = {
            let last = self.inner.last_versions.borrow();
            *last != current
        };
        if !stale {
            return;
        }
        let new_val = (self.inner.recompute)();
        *self.inner.last_versions.borrow_mut() = current;
        if self.inner.cached.get() != new_val {
            self.inner.cached.set(new_val);
            self.inner
                .self_version
                .set(self.inner.self_version.get().wrapping_add(1));
        }
    }
}

impl<T: Copy + PartialEq + 'static> Clone for Computed<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T: Copy + PartialEq + std::fmt::Debug + 'static> std::fmt::Debug for Computed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Computed")
            .field("value", &self.read())
            .field("version", &self.version())
            .finish()
    }
}

/// Prop wrapper accepted by builder methods. The variant stores the
/// active reactive flavor; the builder erases all of them to `Bind<T>`
/// at the call site so a single method signature like
/// `fn color(self, c: impl Into<Bind<[f32; 4]>>) -> Self` accepts a
/// raw color, a `Signal<[f32; 4]>`, a `Computed<[f32; 4]>` or an
/// `AnimatedBind<[f32; 4]>` interchangeably.
///
/// `T: Lerp` is required so that the `Animated` variant can exist;
/// non-animatable types still work — they just won't accept the
/// animated variant.
pub enum Bind<T: Lerp> {
    Value(T),
    Signal(Signal<T>),
    Computed(Computed<T>),
    Animated(AnimatedBind<T>),
}

impl<T: Lerp> Bind<T> {
    /// Current target value. For `Animated` this is the *destination*
    /// of the implicit tween — the app shell handles tween-time
    /// interpolation by starting a `Tween<T>` against an internal
    /// signal whenever this target moves.
    pub fn read(&self) -> T {
        match self {
            Bind::Value(v) => *v,
            Bind::Signal(s) => s.get(),
            Bind::Computed(c) => c.read(),
            Bind::Animated(a) => a.source.read(),
        }
    }

    /// Monotonic change counter. Static for `Value`, delegated for
    /// reactive variants. The shell stores the last seen version per
    /// bind site and re-evaluates when it advances.
    pub fn version(&self) -> u64 {
        match self {
            Bind::Value(_) => 0,
            Bind::Signal(s) => s.version(),
            Bind::Computed(c) => c.version(),
            Bind::Animated(a) => a.source.version(),
        }
    }

    /// `Some((curve, duration))` if this prop should be tweened on
    /// change; `None` if it should snap.
    pub fn animation(&self) -> Option<(Curve, Duration)> {
        match self {
            Bind::Animated(a) => Some((a.curve, a.duration)),
            _ => None,
        }
    }
}

impl<T: Lerp + std::fmt::Debug> std::fmt::Debug for Bind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bind::Value(v) => f.debug_tuple("Value").field(v).finish(),
            Bind::Signal(s) => f.debug_tuple("Signal").field(s).finish(),
            Bind::Computed(c) => f.debug_tuple("Computed").field(c).finish(),
            Bind::Animated(a) => f.debug_tuple("Animated").field(a).finish(),
        }
    }
}

impl<T: Lerp> From<T> for Bind<T> {
    fn from(value: T) -> Self {
        Bind::Value(value)
    }
}

impl<T: Lerp> From<Signal<T>> for Bind<T> {
    fn from(s: Signal<T>) -> Self {
        Bind::Signal(s)
    }
}

impl<T: Lerp> From<Computed<T>> for Bind<T> {
    fn from(c: Computed<T>) -> Self {
        Bind::Computed(c)
    }
}

impl<T: Lerp> From<AnimatedBind<T>> for Bind<T> {
    fn from(a: AnimatedBind<T>) -> Self {
        Bind::Animated(a)
    }
}

/// Reactive image-handle prop, accepted by [`crate::scene::Scene::image_bound`].
///
/// Distinct from [`Bind`] because an `ImageHandle` is a discrete swap —
/// there's nothing to interpolate, so it carries no animation policy and
/// doesn't need [`Lerp`]. `Option` so a node can render nothing until its
/// cover resolves (e.g. an album backdrop before the art lands). Accepts
/// a literal handle, an `Option<ImageHandle>`, a `Signal`, or a
/// `Computed` interchangeably via `From`.
pub enum ImageBind {
    Value(Option<ImageHandle>),
    Signal(Signal<Option<ImageHandle>>),
    Computed(Computed<Option<ImageHandle>>),
}

impl ImageBind {
    pub fn read(&self) -> Option<ImageHandle> {
        match self {
            ImageBind::Value(v) => *v,
            ImageBind::Signal(s) => s.get(),
            ImageBind::Computed(c) => c.read(),
        }
    }

    pub fn version(&self) -> u64 {
        match self {
            ImageBind::Value(_) => 0,
            ImageBind::Signal(s) => s.version(),
            ImageBind::Computed(c) => c.version(),
        }
    }

    /// `false` for a static literal — no slot needs registering.
    pub fn is_reactive(&self) -> bool {
        !matches!(self, ImageBind::Value(_))
    }
}

impl From<Option<ImageHandle>> for ImageBind {
    fn from(v: Option<ImageHandle>) -> Self {
        ImageBind::Value(v)
    }
}

impl From<ImageHandle> for ImageBind {
    fn from(v: ImageHandle) -> Self {
        ImageBind::Value(Some(v))
    }
}

impl From<Signal<Option<ImageHandle>>> for ImageBind {
    fn from(s: Signal<Option<ImageHandle>>) -> Self {
        ImageBind::Signal(s)
    }
}

impl From<Computed<Option<ImageHandle>>> for ImageBind {
    fn from(c: Computed<Option<ImageHandle>>) -> Self {
        ImageBind::Computed(c)
    }
}

/// Reactive text-content prop, accepted by [`crate::scene::Scene::text_bound`].
///
/// Like [`ImageBind`], it's separate from [`Bind`] — a string can't be
/// interpolated and `String`/`Rc<str>` aren't `Copy`, so it can't ride
/// the `Lerp`/`Source` machinery. No `Computed` variant: the `Source`
/// trait is `Copy`-bound, so a computed string isn't expressible; pass a
/// [`TextSignal`] (or update it from a closure) for derived text.
pub enum TextBind {
    Value(std::rc::Rc<str>),
    Signal(TextSignal),
}

impl TextBind {
    pub fn read(&self) -> std::rc::Rc<str> {
        match self {
            TextBind::Value(v) => v.clone(),
            TextBind::Signal(s) => s.get(),
        }
    }

    pub fn version(&self) -> u64 {
        match self {
            TextBind::Value(_) => 0,
            TextBind::Signal(s) => s.version(),
        }
    }

    pub fn is_reactive(&self) -> bool {
        matches!(self, TextBind::Signal(_))
    }
}

impl From<TextSignal> for TextBind {
    fn from(s: TextSignal) -> Self {
        TextBind::Signal(s)
    }
}

impl From<&str> for TextBind {
    fn from(s: &str) -> Self {
        TextBind::Value(s.into())
    }
}

impl From<String> for TextBind {
    fn from(s: String) -> Self {
        TextBind::Value(s.into())
    }
}

impl From<std::rc::Rc<str>> for TextBind {
    fn from(s: std::rc::Rc<str>) -> Self {
        TextBind::Value(s)
    }
}

/// A bind plus an animation policy. Created via [`animated`]. When
/// the shell sees the source change, it auto-starts a `Tween<T>` from
/// the current displayed value to the new target.
pub struct AnimatedBind<T: Lerp> {
    pub source: Box<Bind<T>>,
    pub curve: Curve,
    pub duration: Duration,
}

impl<T: Lerp + std::fmt::Debug> std::fmt::Debug for AnimatedBind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimatedBind")
            .field("source", &self.source)
            .field("curve", &self.curve)
            .field("duration", &self.duration)
            .finish()
    }
}

/// Wrap any bind-able source in an animation policy. The shell will
/// tween it on change.
///
/// ```ignore
/// .color(animated(hero_color, Curve::EaseInOut, Duration::from_millis(220)))
/// ```
pub fn animated<T: Lerp, S: Into<Bind<T>>>(
    source: S,
    curve: Curve,
    duration: Duration,
) -> AnimatedBind<T> {
    AnimatedBind {
        source: Box::new(source.into()),
        curve,
        duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_dep_recomputes_on_change() {
        let s = Signal::new(2_i32);
        let c = Computed::new((s.clone(),), |(v,)| v * 10);
        assert_eq!(c.read(), 20);
        assert_eq!(c.version(), 0);
        s.set(3);
        assert_eq!(c.read(), 30);
        assert_eq!(c.version(), 1);
    }

    #[test]
    fn multi_dep_recomputes_when_any_changes() {
        let a = Signal::new(1_i32);
        let b = Signal::new(2_i32);
        let c = Signal::new(3_i32);
        let comp = Computed::new((a.clone(), b.clone(), c.clone()), |(a, b, c)| a + b + c);
        assert_eq!(comp.read(), 6);
        b.set(20);
        assert_eq!(comp.read(), 24);
        c.set(30);
        assert_eq!(comp.read(), 51);
    }

    #[test]
    fn noop_recompute_does_not_bump_version() {
        // Dep version moves but the derived value is unchanged → the
        // computed's own version should NOT advance, so downstream
        // observers stay quiet.
        let s = Signal::new(0_i32);
        let c = Computed::new((s.clone(),), |(v,)| v / 100);
        assert_eq!(c.read(), 0);
        let v0 = c.version();
        s.set(1); // 1/100 == 0
        assert_eq!(c.read(), 0);
        assert_eq!(c.version(), v0);
        s.set(101);
        assert_eq!(c.read(), 1);
        assert_eq!(c.version(), v0 + 1);
    }

    #[test]
    fn computed_chain() {
        let s = Signal::new(2_i32);
        let c1 = Computed::new((s.clone(),), |(v,)| v + 1);
        let c2 = Computed::new((c1.clone(),), |(v,)| v * v);
        assert_eq!(c2.read(), 9);
        s.set(4);
        // c1 should see new dep version, recompute to 5, then c2's
        // dep version (c1's self_version) bumps and c2 recomputes.
        assert_eq!(c2.read(), 25);
    }

    #[test]
    fn bind_value_variants_resolve() {
        let raw: Bind<f32> = 0.5.into();
        assert_eq!(raw.read(), 0.5);
        assert_eq!(raw.version(), 0);
        assert!(raw.animation().is_none());

        let s = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let bind: Bind<[f32; 4]> = s.clone().into();
        assert_eq!(bind.read(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(bind.version(), 0);
        s.set([0.0, 1.0, 0.0, 1.0]);
        assert_eq!(bind.read(), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(bind.version(), 1);
    }

    #[test]
    fn bind_computed_tracks_chain() {
        let lit = Signal::new(false);
        let color = Computed::new((lit.clone(),), |(l,)| {
            if l { [0.0, 1.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0, 1.0] }
        });
        let bind: Bind<[f32; 4]> = color.clone().into();
        assert_eq!(bind.read(), [1.0, 0.0, 0.0, 1.0]);
        let v0 = bind.version();
        lit.set(true);
        assert_eq!(bind.read(), [0.0, 1.0, 0.0, 1.0]);
        assert!(bind.version() > v0);
    }

    #[test]
    fn animated_bind_carries_curve_and_delegates_read() {
        let s = Signal::new([0.0_f32, 0.0, 0.0, 1.0]);
        let ab = animated(s.clone(), Curve::EaseInOut, Duration::from_millis(220));
        let bind: Bind<[f32; 4]> = ab.into();
        assert_eq!(bind.read(), [0.0, 0.0, 0.0, 1.0]);
        let (curve, dur) = bind.animation().expect("animated");
        assert!(matches!(curve, Curve::EaseInOut));
        assert_eq!(dur, Duration::from_millis(220));
        s.set([1.0, 1.0, 1.0, 1.0]);
        assert_eq!(bind.read(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn lazy_only_runs_when_read() {
        // Use a refcell counter inside the closure to assert how many
        // times it ran. Setting deps without reading should NOT run f.
        let s = Signal::new(0_i32);
        let counter = Rc::new(Cell::new(0));
        let counter_for_f = Rc::clone(&counter);
        let c = Computed::new((s.clone(),), move |(v,)| {
            counter_for_f.set(counter_for_f.get() + 1);
            v * 2
        });
        // initial eager seed ran it once
        assert_eq!(counter.get(), 1);
        // set without reading
        s.set(1);
        s.set(2);
        s.set(3);
        assert_eq!(counter.get(), 1);
        // single read coalesces into one recompute
        assert_eq!(c.read(), 6);
        assert_eq!(counter.get(), 2);
        // reading again with no changes is free
        c.read();
        c.read();
        assert_eq!(counter.get(), 2);
    }
}

//! Reactive value primitive.
//!
//! Shared cell + monotonic version counter. `set` skips no-op writes
//! (value unchanged) so callers can drive signals from input events
//! without flooding dirty.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide monotonic source for [`Signal::id`]. Each `Signal::new`
/// takes the next value; clones share it (it lives on the Rc'd inner).
/// Used to derive a stable timeline tween key from a signal's identity
/// (see `Timeline::animate`) so callers don't hand-author tween keys.
static NEXT_SIGNAL_ID: AtomicU64 = AtomicU64::new(1);

pub struct Signal<T: Copy + PartialEq> {
    inner: Rc<SignalInner<T>>,
}

struct SignalInner<T> {
    value: Cell<T>,
    version: Cell<u64>,
    id: u64,
}

impl<T: Copy + PartialEq> Signal<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Rc::new(SignalInner {
                value: Cell::new(value),
                version: Cell::new(0),
                id: NEXT_SIGNAL_ID.fetch_add(1, Ordering::Relaxed),
            }),
        }
    }

    pub fn get(&self) -> T {
        self.inner.value.get()
    }

    /// Stable, process-unique identity (shared across clones). Drives
    /// identity-keyed tweens — see `Timeline::animate`.
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// Returns true if the value actually changed.
    pub fn set(&self, value: T) -> bool {
        if self.inner.value.get() == value {
            return false;
        }
        self.inner.value.set(value);
        self.inner
            .version
            .set(self.inner.version.get().wrapping_add(1));
        true
    }

    pub fn version(&self) -> u64 {
        self.inner.version.get()
    }
}

impl<T: Copy + PartialEq> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T: Copy + PartialEq + std::fmt::Debug> std::fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signal")
            .field("value", &self.get())
            .field("version", &self.version())
            .finish()
    }
}

/// Reactive string. [`Signal`] is `Cell`-backed (Copy-only), which
/// excludes `String`/`Rc<str>`; this is the non-`Copy` sibling for text
/// content. `RefCell<Rc<str>>` so `get` is a cheap refcount bump and the
/// value can be shared without cloning the bytes. `set` dedups on
/// content (str equality), same as `Signal`.
#[derive(Clone)]
pub struct TextSignal {
    inner: Rc<TextSignalInner>,
}

struct TextSignalInner {
    value: RefCell<Rc<str>>,
    version: Cell<u64>,
}

impl TextSignal {
    pub fn new(value: impl Into<Rc<str>>) -> Self {
        Self {
            inner: Rc::new(TextSignalInner {
                value: RefCell::new(value.into()),
                version: Cell::new(0),
            }),
        }
    }

    pub fn get(&self) -> Rc<str> {
        self.inner.value.borrow().clone()
    }

    /// Returns true if the content actually changed.
    pub fn set(&self, value: impl Into<Rc<str>>) -> bool {
        let value = value.into();
        if *self.inner.value.borrow() == value {
            return false;
        }
        *self.inner.value.borrow_mut() = value;
        self.inner
            .version
            .set(self.inner.version.get().wrapping_add(1));
        true
    }

    pub fn version(&self) -> u64 {
        self.inner.version.get()
    }
}

impl std::fmt::Debug for TextSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSignal")
            .field("value", &self.get())
            .field("version", &self.version())
            .finish()
    }
}

//! Reactive value primitive.
//!
//! Stage-1 minimal: a shared cell + monotonic version counter. `set` skips
//! no-op writes (value unchanged) so callers can drive signals from input
//! events without flooding dirty. Subscriber callbacks land in M5/M6 when
//! input + animation need them.

use std::cell::Cell;
use std::rc::Rc;

pub struct Signal<T: Copy + PartialEq> {
    inner: Rc<SignalInner<T>>,
}

struct SignalInner<T> {
    value: Cell<T>,
    version: Cell<u64>,
}

impl<T: Copy + PartialEq> Signal<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Rc::new(SignalInner {
                value: Cell::new(value),
                version: Cell::new(0),
            }),
        }
    }

    pub fn get(&self) -> T {
        self.inner.value.get()
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

//! Allocation-free registration shared by native completion adapters.
use atomic_waker::AtomicWaker;
use core::{
    sync::atomic::{AtomicBool, Ordering},
    task::Waker,
};

/// A sticky notification, independent of the authoritative native event state.
/// Registration and notification may race; either order wakes the registered
/// observer. It owns no native handles or buffers. Reset requires exclusive
/// access and, for foreign callbacks, proof that the previous callback retired.
#[derive(Debug, Default)]
pub struct CompletionNotification {
    notified: AtomicBool,
    waker: AtomicWaker,
}
impl CompletionNotification {
    /// Publish a hint and wake the current observer, if any.
    pub fn notify(&self) {
        self.notified.store(true, Ordering::Release);
        self.waker.wake();
    }
    /// Register or replace an observer. Already-published hints wake immediately.
    pub fn register(&self, waker: &Waker) {
        self.waker.register(waker);
        if self.is_notified() {
            self.waker.wake();
        }
    }
    /// Whether a producer published a hint for this generation.
    pub fn is_notified(&self) -> bool {
        self.notified.load(Ordering::Acquire)
    }
    /// Start a new generation only after every old producer has retired.
    pub fn reset(&mut self) {
        self.waker.take();
        *self.notified.get_mut() = false;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::sync::{Arc, atomic::AtomicUsize};
    use std::task::Wake;
    struct Counter(AtomicUsize);
    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn registration_order_replacement_and_reset_preserve_notifications() {
        let first = Arc::new(Counter(AtomicUsize::new(0)));
        let second = Arc::new(Counter(AtomicUsize::new(0)));
        let mut n = CompletionNotification::default();
        n.register(&Waker::from(first.clone()));
        n.register(&Waker::from(second.clone()));
        n.notify();
        assert_eq!(first.0.load(Ordering::Relaxed), 0);
        assert_eq!(second.0.load(Ordering::Relaxed), 1);
        n.reset();
        assert!(!n.is_notified());
        n.notify();
        n.register(&Waker::from(first.clone()));
        assert_eq!(first.0.load(Ordering::Relaxed), 1);
        n.reset();
        n.register(&Waker::from(second.clone()));
        assert_eq!(second.0.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn concurrent_registration_cannot_lose_a_completed_notification() {
        for _ in 0..100 {
            let n = Arc::new(CompletionNotification::default());
            let observer = Arc::new(Counter(AtomicUsize::new(0)));
            let producer = n.clone();
            let t = std::thread::spawn(move || producer.notify());
            n.register(&Waker::from(observer.clone()));
            t.join().unwrap();
            assert!(observer.0.load(Ordering::Relaxed) > 0);
        }
    }
}

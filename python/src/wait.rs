//! Hosted address-wait semantics without platform FFI or a newer macOS baseline.
//!
//! The atomic fast path avoids parking after a value change. A short mutex gate
//! closes the check-to-sleep race; the standard library supplies native timed
//! condition-variable parking on Linux, macOS and Windows. This is process-local.
//! It does not turn a polling-only accelerator API into a completion notifier.
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct WaitWord {
    value: AtomicU32,
    gate: Mutex<()>,
    changed: Condvar,
}

impl WaitWord {
    pub fn load(&self) -> u32 {
        self.value.load(Ordering::Acquire)
    }

    /// Publish a new value and wake waiters. Callers must recheck their predicate.
    pub fn store_and_wake(&self, value: u32) {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.value.store(value, Ordering::Release);
        self.changed.notify_all();
    }

    /// Return true if the word changed; false if the bounded wait expired.
    /// Publication and registration use the same gate, so a wake cannot be lost
    /// between the second predicate check and atomically unlocking to sleep.
    pub fn wait(&self, expected: u32, timeout: Duration) -> bool {
        if self.load() != expected {
            return true;
        }
        let guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = self
            .changed
            .wait_timeout_while(guard, timeout, |_| self.load() == expected)
            .unwrap_or_else(|e| e.into_inner());
        self.load() != expected
    }
}

/// At most four immediate retries within 50 us, then 50–1000 us timed parking.
/// Long-running work keeps the previous 1 ms polling ceiling; short work avoids
/// paying that delay on its first pending result. Reset for each submission.
#[derive(Default)]
pub(crate) struct AdaptivePoll {
    attempts: u32,
    parks: u32,
    started: Option<Instant>,
    word: WaitWord,
}

impl AdaptivePoll {
    fn delay(&mut self, elapsed: Duration) -> Duration {
        let attempt = self.attempts;
        self.attempts = self.attempts.saturating_add(1);
        if attempt < 4 && elapsed < Duration::from_micros(50) {
            Duration::ZERO
        } else {
            let delay = Duration::from_micros((50_u64 << self.parks.min(5)).min(1000));
            self.parks = self.parks.saturating_add(1);
            delay
        }
    }

    pub(crate) fn wait(&mut self, remaining: Duration) {
        let elapsed = self.started.get_or_insert_with(Instant::now).elapsed();
        let delay = self.delay(elapsed).min(remaining);
        if delay.is_zero() {
            std::hint::spin_loop();
        } else {
            // Baseline backends have no completion callback. Timed parking is
            // intentional; there is no helper thread polling to manufacture wakes.
            self.word.wait(0, delay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn notification_before_or_during_registration_is_not_lost() {
        for _ in 0..100 {
            let word = Arc::new(WaitWord::default());
            let ready = Arc::new(Barrier::new(2));
            let thread = {
                let word = word.clone();
                let ready = ready.clone();
                std::thread::spawn(move || {
                    ready.wait();
                    word.wait(0, Duration::from_secs(2))
                })
            };
            ready.wait();
            word.store_and_wake(1);
            assert!(thread.join().unwrap());
            assert!(word.wait(0, Duration::ZERO));
            assert!(!word.wait(1, Duration::from_micros(100)));
        }
    }

    #[test]
    fn retries_are_bounded_and_backoff_saturates() {
        let mut poll = AdaptivePoll::default();
        let delays: Vec<_> = (0..12)
            .map(|_| poll.delay(Duration::ZERO).as_micros())
            .collect();
        assert_eq!(
            delays,
            [0, 0, 0, 0, 50, 100, 200, 400, 800, 1000, 1000, 1000]
        );
        poll.attempts = u32::MAX;
        assert_eq!(poll.delay(Duration::ZERO), Duration::from_millis(1));
        poll.wait(Duration::ZERO);
        let mut slow_poll = AdaptivePoll::default();
        assert_eq!(
            slow_poll.delay(Duration::from_micros(50)),
            Duration::from_micros(50)
        );
    }
}

use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroUsize;

use crate::ValueId;

const SHAPE_TAG: u64 = 0x5348_4150_4500_0001;
const CTC_TAG: u64 = 0x4354_4300_0000_0002;

/// Failure while constructing a canonical specialization key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecializationError {
    ValuesOutOfOrder,
    TooManyWords,
    AllocationFailed,
}

impl fmt::Display for SpecializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Exact, collision-safe key for one set of dynamic shapes and CTC bytes.
///
/// The fingerprint accelerates rejection; equality always compares the canonical words as well.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecializationKey {
    words: Vec<u64>,
    fingerprint: [u64; 2],
}

impl SpecializationKey {
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub const fn fingerprint(&self) -> [u64; 2] {
        self.fingerprint
    }
}

/// Bounded builder for provider cache keys.
///
/// Values must be appended in increasing [`ValueId`] order, making equivalent submissions produce
/// identical keys without a sorting allocation. Byte payloads are packed little-endian.
#[derive(Debug)]
pub struct SpecializationKeyBuilder {
    words: Vec<u64>,
    max_words: usize,
    last_value: Option<ValueId>,
}

impl SpecializationKeyBuilder {
    pub fn new(max_words: usize) -> Self {
        Self {
            words: Vec::new(),
            max_words,
            last_value: None,
        }
    }

    pub fn push_shape(
        &mut self,
        value: ValueId,
        dimensions: &[u64],
    ) -> Result<(), SpecializationError> {
        self.begin_value(value, SHAPE_TAG, dimensions.len(), dimensions.len())?;
        self.words.extend_from_slice(dimensions);
        Ok(())
    }

    pub fn push_ctc(&mut self, value: ValueId, bytes: &[u8]) -> Result<(), SpecializationError> {
        let payload_words = bytes.len().div_ceil(8);
        self.begin_value(value, CTC_TAG, bytes.len(), payload_words)?;
        for chunk in bytes.chunks(8) {
            let mut packed = [0_u8; 8];
            packed[..chunk.len()].copy_from_slice(chunk);
            self.words.push(u64::from_le_bytes(packed));
        }
        Ok(())
    }

    pub fn finish(self) -> SpecializationKey {
        SpecializationKey {
            fingerprint: fingerprint(&self.words),
            words: self.words,
        }
    }

    fn begin_value(
        &mut self,
        value: ValueId,
        tag: u64,
        payload_len: usize,
        payload_words: usize,
    ) -> Result<(), SpecializationError> {
        if self.last_value.is_some_and(|prior| prior >= value) {
            return Err(SpecializationError::ValuesOutOfOrder);
        }
        let additional = 3_usize
            .checked_add(payload_words)
            .ok_or(SpecializationError::TooManyWords)?;
        // Reserve the complete record before mutating either the words or ordering state. A caller
        // can therefore recover from a limit/allocation error and retry with a smaller value.
        self.reserve(additional)?;
        self.words.push(tag);
        self.words.push(u64::from(value.get()));
        self.words
            .push(u64::try_from(payload_len).map_err(|_| SpecializationError::TooManyWords)?);
        self.last_value = Some(value);
        Ok(())
    }

    fn reserve(&mut self, additional: usize) -> Result<(), SpecializationError> {
        if self
            .words
            .len()
            .checked_add(additional)
            .is_none_or(|required| required > self.max_words)
        {
            return Err(SpecializationError::TooManyWords);
        }
        self.words
            .try_reserve_exact(additional)
            .map_err(|_| SpecializationError::AllocationFailed)
    }
}

fn fingerprint(words: &[u64]) -> [u64; 2] {
    let mut first = 0xcbf2_9ce4_8422_2325_u64;
    let mut second = 0x9e37_79b9_7f4a_7c15_u64;
    for &word in words {
        first ^= word;
        first = first.wrapping_mul(0x0000_0100_0000_01b3);
        second ^= word.wrapping_add(first.rotate_left(17));
        second = second.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    [first, second ^ words.len() as u64]
}

#[derive(Debug)]
struct CacheEntry<V> {
    key: SpecializationKey,
    value: V,
    last_used: u64,
}

/// An insertion that could not reserve its one bounded cache entry.
#[derive(Debug)]
pub struct CacheInsertError<V> {
    pub key: SpecializationKey,
    pub value: V,
}

/// Small exact-key LRU for compiled shape specializations.
///
/// This portable cache is deliberately synchronization-free. A concurrent provider should put a
/// lock or single-flight compilation state around it at its existing program/queue owner boundary.
#[derive(Debug)]
pub struct SpecializationCache<V> {
    capacity: NonZeroUsize,
    clock: u64,
    entries: Vec<CacheEntry<V>>,
}

impl<V> SpecializationCache<V> {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            clock: 0,
            entries: Vec::new(),
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity.get()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&mut self, key: &SpecializationKey) -> Option<&V> {
        let index = self.position(key)?;
        let stamp = self.tick();
        self.entries[index].last_used = stamp;
        Some(&self.entries[index].value)
    }

    pub fn get_mut(&mut self, key: &SpecializationKey) -> Option<&mut V> {
        let index = self.position(key)?;
        let stamp = self.tick();
        self.entries[index].last_used = stamp;
        Some(&mut self.entries[index].value)
    }

    /// Insert or replace a specialization, returning the replaced or evicted compiled value.
    pub fn insert(
        &mut self,
        key: SpecializationKey,
        value: V,
    ) -> Result<Option<V>, CacheInsertError<V>> {
        let stamp = self.tick();
        if let Some(index) = self.position(&key) {
            self.entries[index].last_used = stamp;
            return Ok(Some(core::mem::replace(
                &mut self.entries[index].value,
                value,
            )));
        }
        if self.entries.len() == self.capacity.get() {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("nonzero full cache");
            let evicted = self.entries.swap_remove(index).value;
            self.entries.push(CacheEntry {
                key,
                value,
                last_used: stamp,
            });
            return Ok(Some(evicted));
        }
        if self.entries.try_reserve_exact(1).is_err() {
            return Err(CacheInsertError { key, value });
        }
        self.entries.push(CacheEntry {
            key,
            value,
            last_used: stamp,
        });
        Ok(None)
    }

    fn position(&self, key: &SpecializationKey) -> Option<usize> {
        self.entries.iter().position(|entry| {
            entry.key.fingerprint == key.fingerprint && entry.key.words == key.words
        })
    }

    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            self.entries.sort_unstable_by_key(|entry| entry.last_used);
            for (index, entry) in self.entries.iter_mut().enumerate() {
                entry.last_used = index as u64;
            }
            self.clock = self.entries.len() as u64;
        }
        let stamp = self.clock;
        self.clock += 1;
        stamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(raw: u32) -> ValueId {
        ValueId::from_raw(raw)
    }

    fn key(raw: u32) -> SpecializationKey {
        let mut builder = SpecializationKeyBuilder::new(16);
        builder
            .push_shape(value(raw), &[1, u64::from(raw)])
            .unwrap();
        builder.finish()
    }

    #[test]
    fn keys_are_canonical_bounded_and_collision_safe() {
        let mut first = SpecializationKeyBuilder::new(16);
        first.push_shape(value(1), &[1, 2, 3]).unwrap();
        first
            .push_ctc(value(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        let first = first.finish();

        let mut second = SpecializationKeyBuilder::new(16);
        second.push_shape(value(1), &[1, 2, 3]).unwrap();
        second
            .push_ctc(value(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        assert_eq!(first, second.finish());

        let mut invalid = SpecializationKeyBuilder::new(8);
        invalid.push_shape(value(2), &[1]).unwrap();
        assert_eq!(
            invalid.push_shape(value(1), &[1]),
            Err(SpecializationError::ValuesOutOfOrder)
        );

        let mut retry = SpecializationKeyBuilder::new(4);
        assert_eq!(
            retry.push_shape(value(3), &[1, 2]),
            Err(SpecializationError::TooManyWords)
        );
        retry.push_shape(value(3), &[1]).unwrap();
        assert_eq!(retry.finish().words().len(), 4);
    }

    #[test]
    fn cache_replaces_and_evicts_the_least_recently_used_value() {
        let mut cache = SpecializationCache::new(NonZeroUsize::new(2).unwrap());
        let one = key(1);
        let two = key(2);
        let three = key(3);
        assert_eq!(cache.insert(one.clone(), 10).unwrap(), None);
        assert_eq!(cache.insert(two.clone(), 20).unwrap(), None);
        assert_eq!(cache.get(&one), Some(&10));
        assert_eq!(cache.insert(three.clone(), 30).unwrap(), Some(20));
        assert_eq!(cache.get(&two), None);
        assert_eq!(cache.get(&three), Some(&30));
        assert_eq!(cache.insert(one.clone(), 11).unwrap(), Some(10));
        assert_eq!(cache.get(&one), Some(&11));
    }
}

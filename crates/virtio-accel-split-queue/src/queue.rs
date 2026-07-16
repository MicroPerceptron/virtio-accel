use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::{IntoIter, Vec};
use core::sync::atomic::{AtomicU64, Ordering};

use virtio_accel_transport::{
    ChainId, DeviceChain, DeviceQueue, DriverQueue, MalformedChain, NotificationHint,
    NotificationRecheck, PublishError, PublishErrorKind, PublishedChain, QueueConfigError,
    QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize, QueueState, ReclaimedChain,
    UsedChain, UsedLength,
};

use crate::{DriverChain, SplitDeviceChain};

/// Failure while constructing a split queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitQueueInitError {
    /// The chain limit must be at least two and no larger than the maximum queue size.
    DescriptorLimit,
}

/// Concrete failure in the in-memory split-ring model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitQueueError {
    /// Allocation of configured ring storage failed.
    AllocationFailed,
    /// Reconfiguration would discard chains still owned by the queue.
    Busy,
    /// The configured queue is smaller than the queue's descriptor-chain limit.
    ChainLimitExceedsQueue,
    /// A normal driver publication supplied a malformed chain.
    MalformedDriverChain(MalformedChain),
    /// A deterministic test hook supplied an invalid counter or descriptor index.
    InvalidTestState,
    /// Internal ring ownership does not match the published indices.
    CorruptState,
}

/// Observable split-ring counters for deterministic wraparound tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingCounters {
    /// Driver-published available index.
    pub available: u16,
    /// Device-consumed available index.
    pub consumed_available: u16,
    /// Device-published used index.
    pub used: u16,
    /// Driver-consumed used index.
    pub consumed_used: u16,
}

/// Single-owner, bounded split-virtqueue reference model.
///
/// Configuration preallocates descriptor, available, used, and ownership tables. Publication,
/// consumption, completion, notification control, and reset perform no allocation or payload copy.
#[derive(Debug)]
pub struct SplitQueue {
    state: QueueState,
    max_chain_descriptors: u16,
    current_epoch: Rc<AtomicU64>,
    storage: Option<RingStorage>,
    available_notifications: bool,
    used_notifications: bool,
}

impl SplitQueue {
    /// Construct an unconfigured queue.
    pub fn new(
        max_size: QueueSize,
        max_chain_descriptors: u16,
    ) -> Result<Self, SplitQueueInitError> {
        if max_chain_descriptors < 2 || max_chain_descriptors > max_size.get() {
            return Err(SplitQueueInitError::DescriptorLimit);
        }
        let epoch = QueueEpoch::INITIAL;
        Ok(Self {
            state: QueueState::unconfigured(max_size, epoch),
            max_chain_descriptors,
            current_epoch: Rc::new(AtomicU64::new(epoch.get())),
            storage: None,
            available_notifications: true,
            used_notifications: true,
        })
    }

    /// Maximum flattened descriptor count accepted by normal publication.
    pub const fn max_chain_descriptors(&self) -> u16 {
        self.max_chain_descriptors
    }

    /// Inject a raw chain into the available ring, retaining malformed topology for device tests.
    ///
    /// This differs from [`DriverQueue::publish`] only by bypassing profile validation. Descriptor
    /// table and available-ring capacity checks still apply.
    pub fn inject_available(
        &mut self,
        chain: DriverChain,
    ) -> Result<PublishedChain, PublishError<DriverChain, SplitQueueError>> {
        self.publish_inner(chain, true)
    }

    /// Set all four ring indices while the configured queue owns no chains.
    pub fn set_empty_ring_index(&mut self, index: u16) -> Result<(), QueueError<SplitQueueError>> {
        let storage = self.storage_mut()?;
        if storage.active_chains != 0 {
            return Err(QueueError::Transport(SplitQueueError::Busy));
        }
        storage.available_index = index;
        storage.consumed_available = index;
        storage.used_index = index;
        storage.consumed_used = index;
        storage.available_count = 0;
        storage.used_count = 0;
        Ok(())
    }

    /// Select the descriptor index at which the next bounded allocation scan begins.
    pub fn set_next_descriptor(&mut self, index: u16) -> Result<(), QueueError<SplitQueueError>> {
        let storage = self.storage_mut()?;
        if index >= storage.size.get() {
            return Err(QueueError::Transport(SplitQueueError::InvalidTestState));
        }
        storage.next_descriptor = index;
        Ok(())
    }

    /// Return ring counters when storage is configured.
    pub fn ring_counters(&self) -> Option<RingCounters> {
        self.storage.as_ref().map(RingStorage::counters)
    }

    /// Return the number of currently free descriptor-table entries.
    pub fn free_descriptors(&self) -> Option<u16> {
        self.storage
            .as_ref()
            .map(|storage| storage.free_descriptors)
    }

    fn publish_inner(
        &mut self,
        mut chain: DriverChain,
        inject_malformed: bool,
    ) -> Result<PublishedChain, PublishError<DriverChain, SplitQueueError>> {
        if !self.state.ready() {
            return Err(PublishError::new(chain, PublishErrorKind::NotReady));
        }
        if !inject_malformed {
            if let Err(error) = chain.validation() {
                return Err(PublishError::new(
                    chain,
                    PublishErrorKind::Transport(SplitQueueError::MalformedDriverChain(error)),
                ));
            }
            if chain.descriptor_count() > self.max_chain_descriptors {
                return Err(PublishError::new(
                    chain,
                    PublishErrorKind::InsufficientDescriptors,
                ));
            }
        }

        let epoch = self.state.epoch();
        let notifications = self.available_notifications;
        let storage = match self.storage.as_mut() {
            Some(storage) => storage,
            None => {
                return Err(PublishError::new(
                    chain,
                    PublishErrorKind::Transport(SplitQueueError::CorruptState),
                ));
            }
        };
        if storage.available_count == storage.size.get() {
            return Err(PublishError::new(chain, PublishErrorKind::QueueFull));
        }
        if chain.descriptor_count() > storage.free_descriptors {
            return Err(PublishError::new(
                chain,
                PublishErrorKind::InsufficientDescriptors,
            ));
        }

        let head = match storage.allocate_descriptors(&mut chain) {
            Ok(head) => head,
            Err(error) => {
                return Err(PublishError::new(chain, PublishErrorKind::Transport(error)));
            }
        };
        let id = ChainId::new(epoch, u64::from(head));
        if storage.records[usize::from(head)].is_some() {
            storage.release_descriptors(&chain);
            return Err(PublishError::new(
                chain,
                PublishErrorKind::Transport(SplitQueueError::CorruptState),
            ));
        }

        storage.records[usize::from(head)] = Some(ChainRecord {
            id,
            state: ChainState::Available,
            chain,
        });
        let ring_slot = storage.ring_slot(storage.available_index);
        storage.available[ring_slot] = head;
        storage.available_index = storage.available_index.wrapping_add(1);
        storage.available_count += 1;
        storage.active_chains += 1;

        Ok(PublishedChain::new(
            id,
            if notifications {
                NotificationHint::Notify
            } else {
                NotificationHint::Suppressed
            },
        ))
    }

    fn storage_mut(&mut self) -> Result<&mut RingStorage, QueueError<SplitQueueError>> {
        self.storage.as_mut().ok_or(QueueError::NotReady)
    }

    fn advance_epoch(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Option<RingStorage>, QueueError<SplitQueueError>> {
        if next_epoch <= self.state.epoch() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::NonIncreasingEpoch,
            ));
        }

        self.current_epoch
            .store(next_epoch.get(), Ordering::Release);
        self.state = QueueState::unconfigured(self.state.max_size(), next_epoch);
        self.available_notifications = true;
        self.used_notifications = true;
        Ok(self.storage.take())
    }
}

impl QueuePort for SplitQueue {
    fn state(&self) -> QueueState {
        self.state
    }
}

impl QueueControl for SplitQueue {
    type Error = SplitQueueError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        if self.state.ready() {
            return Err(QueueError::Transport(SplitQueueError::Busy));
        }
        if size > self.state.max_size() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::SizeExceedsMaximum,
            ));
        }
        if size.get() < self.max_chain_descriptors {
            return Err(QueueError::Transport(
                SplitQueueError::ChainLimitExceedsQueue,
            ));
        }
        if self
            .storage
            .as_ref()
            .is_some_and(|storage| storage.active_chains != 0)
        {
            return Err(QueueError::Transport(SplitQueueError::Busy));
        }

        let storage = RingStorage::new(size).map_err(QueueError::Transport)?;
        self.state = QueueState::new(self.state.max_size(), Some(size), false, self.state.epoch())
            .map_err(QueueError::InvalidConfiguration)?;
        self.storage = Some(storage);
        Ok(())
    }

    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        self.state = QueueState::new(
            self.state.max_size(),
            self.state.size(),
            ready,
            self.state.epoch(),
        )
        .map_err(QueueError::InvalidConfiguration)?;
        Ok(())
    }
}

impl DriverQueue for SplitQueue {
    type Chain = DriverChain;
    type Reclaimed = ReclaimedChains;
    type Error = SplitQueueError;

    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
        self.publish_inner(chain, false)
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        let storage = self.storage_mut()?;
        if storage.used_count == 0 {
            return Ok(None);
        }

        let ring_slot = storage.ring_slot(storage.consumed_used);
        let used = storage.used[ring_slot]
            .take()
            .ok_or(QueueError::Transport(SplitQueueError::CorruptState))?;
        let Some(record) = storage.records[usize::from(used.head)].take() else {
            storage.used[ring_slot] = Some(used);
            return Err(QueueError::Transport(SplitQueueError::CorruptState));
        };
        if record.id != used.id || record.state != ChainState::Used {
            storage.records[usize::from(used.head)] = Some(record);
            storage.used[ring_slot] = Some(used);
            return Err(QueueError::Transport(SplitQueueError::CorruptState));
        }

        storage.release_descriptors(&record.chain);
        storage.consumed_used = storage.consumed_used.wrapping_add(1);
        storage.used_count -= 1;
        storage.active_chains -= 1;
        Ok(Some(UsedChain::new(record.id, used.used, record.chain)))
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        self.used_notifications = false;
        Ok(())
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        self.used_notifications = true;
        Ok(if self.storage_mut()?.used_count == 0 {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        Ok(ReclaimedChains::new(self.advance_epoch(next_epoch)?))
    }
}

impl DeviceQueue for SplitQueue {
    type Chain = SplitDeviceChain;
    type Error = SplitQueueError;

    fn pop_available(&mut self) -> Result<Option<Self::Chain>, QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        let epoch = self.state.epoch();
        let max_descriptors = self.max_chain_descriptors;
        let current_epoch = Rc::clone(&self.current_epoch);
        let storage = self.storage_mut()?;
        if storage.available_count == 0 {
            return Ok(None);
        }

        let ring_slot = storage.ring_slot(storage.consumed_available);
        let head = storage.available[ring_slot];
        let record = storage.records[usize::from(head)]
            .as_mut()
            .ok_or(QueueError::Transport(SplitQueueError::CorruptState))?;
        if record.id.epoch() != epoch || record.state != ChainState::Available {
            return Err(QueueError::Transport(SplitQueueError::CorruptState));
        }
        record.state = ChainState::InFlight;
        let chain = SplitDeviceChain::new(
            record.id,
            record.chain.data(),
            max_descriptors,
            current_epoch,
        );
        storage.consumed_available = storage.consumed_available.wrapping_add(1);
        storage.available_count -= 1;
        Ok(Some(chain))
    }

    fn complete(
        &mut self,
        chain: Self::Chain,
        used: UsedLength,
    ) -> Result<NotificationHint, QueueError<Self::Error>> {
        let id = chain.id();
        let current = self.state.epoch();
        if id.epoch() != current {
            return Err(QueueError::ResetRace {
                operation: id.epoch(),
                current,
            });
        }
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        let capacity = chain.writable_capacity();
        if u64::from(used.get()) > capacity {
            return Err(QueueError::UsedLengthExceeded { used, capacity });
        }
        let head = u16::try_from(id.token())
            .map_err(|_| QueueError::Transport(SplitQueueError::CorruptState))?;
        let notifications = self.used_notifications;
        let storage = self.storage_mut()?;
        let record = storage
            .records
            .get_mut(usize::from(head))
            .and_then(Option::as_mut)
            .ok_or(QueueError::Transport(SplitQueueError::CorruptState))?;
        if record.id != id || record.state != ChainState::InFlight {
            return Err(QueueError::Transport(SplitQueueError::CorruptState));
        }
        if storage.used_count == storage.size.get() {
            return Err(QueueError::Transport(SplitQueueError::CorruptState));
        }

        record.state = ChainState::Used;
        let ring_slot = storage.ring_slot(storage.used_index);
        storage.used[ring_slot] = Some(UsedElement { head, id, used });
        storage.used_index = storage.used_index.wrapping_add(1);
        storage.used_count += 1;
        Ok(if notifications {
            NotificationHint::Notify
        } else {
            NotificationHint::Suppressed
        })
    }

    fn disable_available_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        self.available_notifications = false;
        Ok(())
    }

    fn enable_available_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        self.available_notifications = true;
        Ok(if self.storage_mut()?.available_count == 0 {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<Self::Error>> {
        drop(self.advance_epoch(next_epoch)?);
        Ok(())
    }
}

#[derive(Debug)]
struct RingStorage {
    size: QueueSize,
    descriptors: Box<[Option<DescriptorOwner>]>,
    records: Box<[Option<ChainRecord>]>,
    available: Box<[u16]>,
    used: Box<[Option<UsedElement>]>,
    available_index: u16,
    consumed_available: u16,
    used_index: u16,
    consumed_used: u16,
    available_count: u16,
    used_count: u16,
    active_chains: u16,
    free_descriptors: u16,
    next_descriptor: u16,
}

impl RingStorage {
    fn new(size: QueueSize) -> Result<Self, SplitQueueError> {
        let len = usize::from(size.get());
        Ok(Self {
            size,
            descriptors: boxed_with(len, |_| None)?,
            records: boxed_with(len, |_| None)?,
            available: boxed_with(len, |_| 0)?,
            used: boxed_with(len, |_| None)?,
            available_index: 0,
            consumed_available: 0,
            used_index: 0,
            consumed_used: 0,
            available_count: 0,
            used_count: 0,
            active_chains: 0,
            free_descriptors: size.get(),
            next_descriptor: 0,
        })
    }

    fn ring_slot(&self, index: u16) -> usize {
        usize::from(index & (self.size.get() - 1))
    }

    const fn counters(&self) -> RingCounters {
        RingCounters {
            available: self.available_index,
            consumed_available: self.consumed_available,
            used: self.used_index,
            consumed_used: self.consumed_used,
        }
    }

    fn allocate_descriptors(&mut self, chain: &mut DriverChain) -> Result<u16, SplitQueueError> {
        let count = chain.descriptor_count();
        if count > self.free_descriptors {
            return Err(SplitQueueError::CorruptState);
        }

        let mut cursor = self.next_descriptor;
        let mut allocated = 0_usize;
        while allocated < usize::from(count) {
            let mut scanned = 0_u16;
            while self.descriptors[usize::from(cursor)].is_some() {
                cursor = self.wrapping_descriptor_add(cursor, 1);
                scanned += 1;
                if scanned == self.size.get() {
                    self.rollback_allocation(chain, allocated);
                    return Err(SplitQueueError::CorruptState);
                }
            }
            chain.slots_mut()[allocated] = cursor;
            self.descriptors[usize::from(cursor)] = Some(DescriptorOwner {
                head: 0,
                local: allocated as u16,
            });
            allocated += 1;
            cursor = self.wrapping_descriptor_add(cursor, 1);
        }

        let head = chain.queue_head_slot();
        for slot in chain.slots() {
            let owner = self.descriptors[usize::from(*slot)]
                .as_mut()
                .ok_or(SplitQueueError::CorruptState)?;
            owner.head = head;
        }
        self.next_descriptor = cursor;
        self.free_descriptors -= count;
        Ok(head)
    }

    fn release_descriptors(&mut self, chain: &DriverChain) {
        for slot in chain.slots() {
            if let Some(owner) = self.descriptors[usize::from(*slot)].take() {
                debug_assert_eq!(owner.head, chain.queue_head_slot());
                debug_assert!(usize::from(owner.local) < chain.slots().len());
                self.free_descriptors += 1;
            }
        }
    }

    fn rollback_allocation(&mut self, chain: &DriverChain, allocated: usize) {
        for slot in &chain.slots()[..allocated] {
            self.descriptors[usize::from(*slot)] = None;
        }
    }

    const fn wrapping_descriptor_add(&self, index: u16, amount: u16) -> u16 {
        index.wrapping_add(amount) & (self.size.get() - 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DescriptorOwner {
    head: u16,
    local: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainState {
    Available,
    InFlight,
    Used,
}

#[derive(Debug)]
struct ChainRecord {
    id: ChainId,
    state: ChainState,
    chain: DriverChain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UsedElement {
    head: u16,
    id: ChainId,
    used: UsedLength,
}

/// Allocation-free iterator over chains reclaimed by a driver reset.
#[derive(Debug)]
pub struct ReclaimedChains {
    records: IntoIter<Option<ChainRecord>>,
}

impl ReclaimedChains {
    fn new(storage: Option<RingStorage>) -> Self {
        let records = storage
            .map(|storage| storage.records.into_vec())
            .unwrap_or_default()
            .into_iter();
        Self { records }
    }
}

impl Iterator for ReclaimedChains {
    type Item = ReclaimedChain<DriverChain>;

    fn next(&mut self) -> Option<Self::Item> {
        self.records
            .by_ref()
            .find_map(|record| record.map(|record| ReclaimedChain::new(record.id, record.chain)))
    }
}

fn boxed_with<T>(
    len: usize,
    mut value: impl FnMut(usize) -> T,
) -> Result<Box<[T]>, SplitQueueError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| SplitQueueError::AllocationFailed)?;
    for index in 0..len {
        values.push(value(index));
    }
    Ok(values.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use virtio_accel_transport::{
        ByteAccessError, ChainError, DeviceChain, ReadableBytes, WritableBytes,
    };

    use super::*;
    use crate::{Descriptor, VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE};

    fn ready_queue(size: u16, max_chain_descriptors: u16) -> SplitQueue {
        let size = QueueSize::new(size).unwrap();
        let mut queue = SplitQueue::new(size, max_chain_descriptors).unwrap();
        QueueControl::configure(&mut queue, size).unwrap();
        QueueControl::set_ready(&mut queue, true).unwrap();
        queue
    }

    fn direct_chain(requests: &[&[u8]], responses: &[usize]) -> DriverChain {
        let mut descriptors = Vec::new();
        for request in requests {
            descriptors.push(Descriptor::readable(request.to_vec()));
        }
        for response in responses {
            descriptors.push(Descriptor::writable(vec![0; *response]));
        }
        DriverChain::direct(descriptors).unwrap()
    }

    fn assert_malformed(queue: &mut SplitQueue, chain: DriverChain, expected: MalformedChain) {
        queue.inject_available(chain).unwrap();
        let mut chain = DeviceQueue::pop_available(queue).unwrap().unwrap();
        match chain.io() {
            Err(ChainError::Malformed(actual)) => assert_eq!(actual, expected),
            result => panic!("unexpected chain result: {result:?}"),
        }
        DeviceQueue::complete(queue, chain, UsedLength::new(0)).unwrap();
        DriverQueue::pop_used(queue).unwrap().unwrap();
    }

    #[test]
    fn vq_001_vq_003_segmented_chain_round_trip_does_not_coalesce_payloads() {
        let mut queue = ready_queue(8, 4);
        let published =
            DriverQueue::publish(&mut queue, direct_chain(&[b"ab", b"cd"], &[2, 2])).unwrap();
        assert_eq!(published.notification(), NotificationHint::Notify);

        let mut device_chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        {
            let (_, request, response) = device_chain.io().unwrap().into_parts();
            let mut bytes = [0; 3];
            request.read_at(1, &mut bytes).unwrap();
            assert_eq!(&bytes, b"bcd");
            response.write_at(0, b"wxyz").unwrap();
        }
        DeviceQueue::complete(&mut queue, device_chain, UsedLength::new(4)).unwrap();

        let used = DriverQueue::pop_used(&mut queue).unwrap().unwrap();
        assert_eq!(used.id(), published.id());
        assert_eq!(used.used(), UsedLength::new(4));
        let (_, _, chain) = used.into_parts();
        let mut first = [0; 2];
        let mut second = [0; 2];
        chain.read_descriptor(2, 0, &mut first).unwrap();
        chain.read_descriptor(3, 0, &mut second).unwrap();
        assert_eq!(&first, b"wx");
        assert_eq!(&second, b"yz");
        assert_eq!(queue.free_descriptors(), Some(8));
    }

    #[test]
    fn split_ring_and_descriptor_indices_wrap_naturally() {
        let mut queue = ready_queue(8, 4);
        queue.set_empty_ring_index(u16::MAX).unwrap();
        queue.set_next_descriptor(7).unwrap();

        let published = DriverQueue::publish(&mut queue, direct_chain(&[b"r"], &[1])).unwrap();
        assert_eq!(published.id().token(), 7);
        assert_eq!(
            queue.storage.as_ref().unwrap().descriptors[0],
            Some(DescriptorOwner { head: 7, local: 1 })
        );
        let chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        DeviceQueue::complete(&mut queue, chain, UsedLength::new(1)).unwrap();
        DriverQueue::pop_used(&mut queue).unwrap().unwrap();

        assert_eq!(
            queue.ring_counters(),
            Some(RingCounters {
                available: 0,
                consumed_available: 0,
                used: 0,
                consumed_used: 0,
            })
        );
    }

    #[test]
    fn vq_016_descriptor_exhaustion_returns_the_unpublished_chain() {
        let mut queue = ready_queue(4, 4);
        DriverQueue::publish(&mut queue, direct_chain(&[b"a"], &[1])).unwrap();
        DriverQueue::publish(&mut queue, direct_chain(&[b"b"], &[1])).unwrap();
        assert_eq!(queue.free_descriptors(), Some(0));

        let error = DriverQueue::publish(&mut queue, direct_chain(&[b"c"], &[1])).unwrap_err();
        assert_eq!(error.kind(), &PublishErrorKind::InsufficientDescriptors);
        let (chain, _) = error.into_parts();
        assert_eq!(chain.descriptor_count(), 2);
        assert_eq!(queue.free_descriptors(), Some(0));
    }

    #[test]
    fn vq_004_vq_005_vq_006_malformed_topology_is_classified_boundedly() {
        let mut queue = ready_queue(8, 4);

        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::raw(vec![0], VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, 0),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::DescriptorLoop,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], VIRTQ_DESC_F_NEXT, 7),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::DescriptorIndex,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], 8, 0),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::DescriptorFlags,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], VIRTQ_DESC_F_INDIRECT, 0),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::IndirectUnsupported,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(Vec::new(), VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::ZeroLength,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![0], VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::readable(vec![1]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::Direction,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::writable(vec![0]),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::DescriptorCount,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![0], VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::Direction,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::raw(vec![1], VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::readable(vec![1]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::Direction,
        );
        assert_malformed(
            &mut queue,
            DriverChain::raw(
                vec![
                    Descriptor::unmapped(u64::MAX, VIRTQ_DESC_F_NEXT, 1),
                    Descriptor::writable(vec![0]),
                ],
                0,
            )
            .unwrap(),
            MalformedChain::Address,
        );
    }

    #[test]
    fn vq_006_normal_publication_rejects_malformed_and_oversized_chains() {
        let mut queue = ready_queue(8, 2);
        let malformed = DriverChain::raw(
            vec![
                Descriptor::raw(Vec::new(), VIRTQ_DESC_F_NEXT, 1),
                Descriptor::writable(vec![0]),
            ],
            0,
        )
        .unwrap();
        let error = DriverQueue::publish(&mut queue, malformed).unwrap_err();
        assert_eq!(
            error.kind(),
            &PublishErrorKind::Transport(SplitQueueError::MalformedDriverChain(
                MalformedChain::ZeroLength
            ))
        );

        let oversized = direct_chain(&[b"a", b"b"], &[1]);
        let error = DriverQueue::publish(&mut queue, oversized).unwrap_err();
        assert_eq!(error.kind(), &PublishErrorKind::InsufficientDescriptors);

        let oversized = direct_chain(&[b"a", b"b"], &[1]);
        queue.inject_available(oversized).unwrap();
        let mut chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        assert!(matches!(
            chain.io(),
            Err(ChainError::Malformed(MalformedChain::DescriptorCount))
        ));
    }

    #[test]
    fn vq_014_completion_order_drives_used_ring_order() {
        let mut queue = ready_queue(8, 4);
        let first = DriverQueue::publish(&mut queue, direct_chain(&[b"a"], &[2]))
            .unwrap()
            .id();
        let second = DriverQueue::publish(&mut queue, direct_chain(&[b"b"], &[2]))
            .unwrap()
            .id();
        let first_chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        let second_chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();

        DeviceQueue::complete(&mut queue, second_chain, UsedLength::new(2)).unwrap();
        DeviceQueue::complete(&mut queue, first_chain, UsedLength::new(1)).unwrap();
        assert_eq!(
            DriverQueue::pop_used(&mut queue).unwrap().unwrap().id(),
            second
        );
        assert_eq!(
            DriverQueue::pop_used(&mut queue).unwrap().unwrap().id(),
            first
        );
    }

    #[test]
    fn vq_017_notification_enablement_rechecks_for_missed_work() {
        let mut queue = ready_queue(8, 4);
        DeviceQueue::disable_available_notifications(&mut queue).unwrap();
        let published = DriverQueue::publish(&mut queue, direct_chain(&[b"a"], &[1])).unwrap();
        assert_eq!(published.notification(), NotificationHint::Suppressed);
        assert_eq!(
            DeviceQueue::enable_available_notifications(&mut queue).unwrap(),
            NotificationRecheck::WorkPending
        );

        let chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        assert_eq!(
            DeviceQueue::enable_available_notifications(&mut queue).unwrap(),
            NotificationRecheck::Idle
        );
        DriverQueue::disable_used_notifications(&mut queue).unwrap();
        assert_eq!(
            DeviceQueue::complete(&mut queue, chain, UsedLength::new(1)).unwrap(),
            NotificationHint::Suppressed
        );
        assert_eq!(
            DriverQueue::enable_used_notifications(&mut queue).unwrap(),
            NotificationRecheck::WorkPending
        );
        DriverQueue::pop_used(&mut queue).unwrap().unwrap();
        assert_eq!(
            DriverQueue::enable_used_notifications(&mut queue).unwrap(),
            NotificationRecheck::Idle
        );
    }

    #[test]
    fn vq_018_reset_invalidates_ports_before_reclaiming_driver_ownership() {
        let mut queue = ready_queue(8, 4);
        let published = DriverQueue::publish(&mut queue, direct_chain(&[b"a"], &[2])).unwrap();
        let mut device_chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        let next_epoch = queue.state().epoch().checked_next().unwrap();

        let mut reclaimed = {
            let (_, _, response) = device_chain.io().unwrap().into_parts();
            let reclaimed = DriverQueue::reset(&mut queue, next_epoch).unwrap();
            assert_eq!(response.write_at(0, b"x"), Err(ByteAccessError::Reset));
            reclaimed
        };

        let reclaimed_chain = reclaimed.next().unwrap();
        assert_eq!(reclaimed_chain.id(), published.id());
        assert!(reclaimed.next().is_none());
        assert!(matches!(
            device_chain.io(),
            Err(ChainError::ResetRace { chain, current })
                if chain == published.id().epoch() && current == next_epoch
        ));
        assert_eq!(
            DeviceQueue::complete(&mut queue, device_chain, UsedLength::new(0)),
            Err(QueueError::ResetRace {
                operation: published.id().epoch(),
                current: next_epoch,
            })
        );
        assert_eq!(
            queue.state(),
            QueueState::unconfigured(QueueSize::new(8).unwrap(), next_epoch)
        );
    }

    #[test]
    fn vq_013_excessive_used_length_publishes_no_used_entry() {
        let mut queue = ready_queue(8, 4);
        DriverQueue::publish(&mut queue, direct_chain(&[b"a"], &[2])).unwrap();
        let chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        assert_eq!(
            DeviceQueue::complete(&mut queue, chain, UsedLength::new(3)),
            Err(QueueError::UsedLengthExceeded {
                used: UsedLength::new(3),
                capacity: 2,
            })
        );
        assert_eq!(queue.ring_counters().unwrap().used, 0);
        assert!(DriverQueue::pop_used(&mut queue).unwrap().is_none());
    }
}

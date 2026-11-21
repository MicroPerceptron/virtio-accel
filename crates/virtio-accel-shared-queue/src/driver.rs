//! The driver's half: chains published, and taken back used.

use core::fmt;
use core::sync::atomic::{Ordering, fence};

use virtio_accel_transport::{
    ChainId, ChainLayoutError, ChainRegion, DriverChainBuffer, DriverQueue, NotificationHint,
    NotificationRecheck, PublishError, PublishErrorKind, PublishedChain, QueueConfigError,
    QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize, QueueState, ReclaimedChain,
    UsedChain, UsedLength, validate_chain_layout,
};

use crate::layout::{
    AVAILABLE_ENTRY, DESCRIPTOR_BYTES, RING_FLAGS, RING_HEADER, RING_INDEX, RingAddresses,
    USED_ENTRY, VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE,
    VIRTQ_USED_F_NO_NOTIFY, descriptor, each_span,
};
use crate::memory::{Access, MemoryFault, SharedMemory};

/// One run of bytes a descriptor names: a device address and a length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    /// The device address of the first byte.
    pub address: u64,
    /// Bytes from there. Never zero in a chain.
    pub len: u32,
}

/// Why a set of regions is not a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainShapeError {
    /// More regions than the chain type holds.
    Capacity,
    /// The protocol's own chain rule refused the shape: fewer than one
    /// region in either direction, or a region of no bytes.
    Layout(ChainLayoutError),
    /// A region runs past the end of the address space.
    AddressOverflow,
}

/// A chain a driver owns: the regions the device reads, then the regions it
/// writes, one descriptor each, at most `R` in all.
///
/// The value is a description. It owns no memory and holds no descriptor
/// until [`DriverRing::publish`] gives it some; the bytes it names are the
/// driver's, in memory `M` reaches by the same device addresses the device
/// will be given. Nothing is copied into a queue: a region may be a buffer
/// that already holds a payload.
#[derive(Debug)]
pub struct DriverChain<M, const R: usize> {
    memory: M,
    regions: [Region; R],
    /// Of the first `count` regions, the first `readable` are the device's
    /// to read.
    readable: u16,
    count: u16,
    readable_bytes: u64,
    writable_bytes: u64,
}

impl<M, const R: usize> DriverChain<M, R> {
    /// A chain of `readable` regions followed by `writable` ones.
    ///
    /// The shape is held to the protocol's own rule here
    /// (`validate_chain_layout`), so a `DriverChain` that exists is one a
    /// conforming device accepts.
    pub fn new(
        memory: M,
        readable: &[Region],
        writable: &[Region],
    ) -> Result<Self, ChainShapeError> {
        let count = readable.len() + writable.len();
        if count > R || count > usize::from(u16::MAX) {
            return Err(ChainShapeError::Capacity);
        }
        let mut regions = [Region::default(); R];
        let mut described = [ChainRegion::readable(0); R];
        for (index, (region, writes)) in readable
            .iter()
            .map(|region| (region, false))
            .chain(writable.iter().map(|region| (region, true)))
            .enumerate()
        {
            if region.address.checked_add(u64::from(region.len)).is_none() {
                return Err(ChainShapeError::AddressOverflow);
            }
            regions[index] = *region;
            described[index] = if writes {
                ChainRegion::writable(u64::from(region.len))
            } else {
                ChainRegion::readable(u64::from(region.len))
            };
        }
        // `count <= u16::MAX` was checked, and the rule's own bound is the
        // type's capacity: the queue applies its own, smaller one.
        let layout = validate_chain_layout(&described[..count], count as u16)
            .map_err(ChainShapeError::Layout)?;
        Ok(Self {
            memory,
            regions,
            readable: layout.readable_descriptors(),
            count: count as u16,
            readable_bytes: layout.readable_bytes(),
            writable_bytes: layout.writable_bytes(),
        })
    }

    /// The regions the device reads.
    pub fn readable(&self) -> &[Region] {
        &self.regions[..usize::from(self.readable)]
    }

    /// The regions the device writes.
    pub fn writable(&self) -> &[Region] {
        &self.regions[usize::from(self.readable)..usize::from(self.count)]
    }

    /// Descriptors the chain takes: one a region.
    pub const fn descriptors(&self) -> u16 {
        self.count
    }
}

impl<M: SharedMemory + fmt::Debug, const R: usize> DriverChainBuffer for DriverChain<M, R> {
    type Error = MemoryFault;

    fn device_readable_len(&self) -> u64 {
        self.readable_bytes
    }

    fn device_writable_len(&self) -> u64 {
        self.writable_bytes
    }

    fn write_device_readable(&mut self, offset: u64, source: &[u8]) -> Result<(), MemoryFault> {
        within(offset, source.len(), self.readable_bytes)?;
        let spans = self
            .readable()
            .iter()
            .map(|region| (region.address, region.len));
        each_span(spans, offset, source.len(), |at, from, count| {
            self.memory.write(at, &source[from..from + count])
        })
    }

    fn read_device_writable(&self, offset: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        within(offset, target.len(), self.writable_bytes)?;
        let spans = self
            .writable()
            .iter()
            .map(|region| (region.address, region.len));
        each_span(spans, offset, target.len(), |at, from, count| {
            self.memory.read(at, &mut target[from..from + count])
        })
    }
}

fn within(offset: u64, len: usize, room: u64) -> Result<(), MemoryFault> {
    match offset.checked_add(len as u64) {
        Some(end) if end <= room => Ok(()),
        _ => Err(MemoryFault::Range),
    }
}

/// Why the driver's half could not go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverRingError {
    /// A part of the queue is not where the layout allows: misaligned,
    /// overlapping another, or outside memory the driver may write.
    Layout,
    /// The queue cannot be reconfigured while it is ready or holds chains.
    Busy,
    /// The queue was sized smaller than the longest chain it must hold.
    ChainLimitExceedsQueue,
    /// The device published more used entries than it holds chains.
    UsedIndex,
    /// A used entry names a chain the device does not hold, or says it wrote
    /// more than the chain gave it room for.
    UsedEntry,
    /// The memory stopped answering for a ring the queue validated.
    Memory(MemoryFault),
}

impl From<MemoryFault> for QueueError<DriverRingError> {
    fn from(fault: MemoryFault) -> Self {
        Self::Transport(DriverRingError::Memory(fault))
    }
}

/// The driver's half of one split virtqueue in shared memory.
///
/// `N` bounds the queue's size and `R` a chain's regions; both are storage
/// the value carries inline, so the queue allocates nothing. One owner:
/// every operation takes `&mut self`, and there is no lock.
///
/// # What the driver trusts
///
/// Nothing the device wrote, and nothing it can write. The descriptor table
/// is never read back: the free list, each chain's links and its room are
/// kept here, so a device that rewrites a descriptor changes nothing the
/// driver does. A used entry is checked before it is believed — the head
/// must be a chain the device holds, the length no more than the chain's
/// writable bytes, and the used index no further ahead than chains are
/// outstanding. A device that fails any of them has stopped being a device:
/// the queue refuses every operation from then on
/// ([`DriverRingError::UsedIndex`], [`DriverRingError::UsedEntry`]) until it
/// is [`reset`](DriverQueue::reset), which is also what the driver owes the
/// device.
///
/// The profile is the protocol's: direct descriptors only, no indirect
/// tables, no event indices, and the two suppression flags.
#[derive(Debug)]
pub struct DriverRing<M, const N: usize, const R: usize> {
    memory: M,
    state: QueueState,
    max_chain_descriptors: u16,
    addresses: RingAddresses,
    /// The free list's links and each published chain's, by descriptor.
    next: [u16; N],
    free_head: u16,
    free: u16,
    /// The chains the device has, by head descriptor.
    held: [Option<DriverChain<M, R>>; N],
    outstanding: u16,
    /// Available entries published, and used entries taken back.
    published: u16,
    taken: u16,
    /// Set once the device has lied; cleared by a reset.
    failed: Option<DriverRingError>,
    /// The rings have not been cleared since the queue was sized or placed.
    fresh: bool,
}

impl<M: SharedMemory + Clone + fmt::Debug, const N: usize, const R: usize> DriverRing<M, N, R> {
    /// A queue in `memory` of at most `max_size` descriptors whose chains
    /// are at most `max_chain_descriptors` long. Unconfigured: the driver
    /// [`configure`](QueueControl::configure)s a size, says where the parts
    /// are ([`set_addresses`](Self::set_addresses)) and makes it ready.
    ///
    /// `None` when the limits do not fit: a queue no larger than `N`, and a
    /// chain of two descriptors at least, no longer than `R` or the queue.
    pub fn new(memory: M, max_size: QueueSize, max_chain_descriptors: u16) -> Option<Self> {
        let fits = usize::from(max_size.get()) <= N
            && max_chain_descriptors >= 2
            && usize::from(max_chain_descriptors) <= R
            && max_chain_descriptors <= max_size.get();
        fits.then(|| Self {
            memory,
            state: QueueState::unconfigured(max_size, QueueEpoch::INITIAL),
            max_chain_descriptors,
            addresses: RingAddresses::default(),
            next: [0; N],
            free_head: 0,
            free: 0,
            held: [const { None }; N],
            outstanding: 0,
            published: 0,
            taken: 0,
            failed: None,
            fresh: true,
        })
    }

    /// Where the queue's parts are. Taken while the queue is not ready and
    /// the device holds nothing, and checked when the queue is made ready.
    pub fn set_addresses(
        &mut self,
        addresses: RingAddresses,
    ) -> Result<(), QueueError<DriverRingError>> {
        if self.state.ready() || self.outstanding != 0 {
            return Err(QueueError::Transport(DriverRingError::Busy));
        }
        self.addresses = addresses;
        self.fresh = true;
        Ok(())
    }

    /// The addresses the queue was given: what the transport is told.
    pub const fn addresses(&self) -> RingAddresses {
        self.addresses
    }

    /// Chains the device holds: published and not yet taken back.
    pub const fn outstanding(&self) -> u16 {
        self.outstanding
    }

    /// Descriptors no chain holds.
    pub const fn free_descriptors(&self) -> u16 {
        self.free
    }

    /// The configured size of a ready queue that has not failed.
    fn live(&self) -> Result<QueueSize, QueueError<DriverRingError>> {
        if let Some(failure) = self.failed {
            return Err(QueueError::Transport(failure));
        }
        match (self.state.ready(), self.state.size()) {
            (true, Some(size)) => Ok(size),
            _ => Err(QueueError::NotReady),
        }
    }

    /// Record that the device lied, and say so.
    fn fail<T>(&mut self, failure: DriverRingError) -> Result<T, QueueError<DriverRingError>> {
        self.failed = Some(failure);
        Err(QueueError::Transport(failure))
    }

    /// Write `chain`'s descriptors into the table on descriptors taken from
    /// the free list, add its head to the available ring and publish the
    /// index. On a fault nothing is published and the free list is as it was.
    fn expose(
        &mut self,
        chain: &DriverChain<M, R>,
        size: u16,
    ) -> Result<(u16, NotificationHint), MemoryFault> {
        let head = self.free_head;
        let last = chain.count - 1;
        let mut index = head;
        for (position, region) in chain.regions[..usize::from(chain.count)].iter().enumerate() {
            let position = position as u16;
            let following = self.next[usize::from(index)];
            let mut flags = if position < chain.readable {
                0
            } else {
                VIRTQ_DESC_F_WRITE
            };
            if position < last {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            let next = if position < last { following } else { 0 };
            self.memory.write(
                self.addresses.descriptors + u64::from(index) * DESCRIPTOR_BYTES,
                &descriptor(region.address, region.len, flags, next),
            )?;
            index = following;
        }
        let slot = u64::from(self.published % size);
        self.memory.write(
            self.addresses.available + RING_HEADER + slot * AVAILABLE_ENTRY,
            &head.to_le_bytes(),
        )?;
        let published = self.published.wrapping_add(1);
        // Release: the request bytes, the descriptors and the entry are
        // visible to a device that has loaded this index.
        self.memory
            .store_u16(self.addresses.available + RING_INDEX, published)?;
        // Committed: the descriptors are the device's from here.
        self.published = published;
        self.free_head = index;
        self.free -= chain.count;
        // The index is published before the device's suppression flag is
        // read: a device that clears the flag and rechecks the index cannot
        // miss this chain. The chain is the device's by now, so a flag that
        // cannot be read is a device that is told, not a failed publication.
        fence(Ordering::SeqCst);
        let suppressed = matches!(
            self.memory.load_u16(self.addresses.used + RING_FLAGS),
            Ok(flags) if flags & VIRTQ_USED_F_NO_NOTIFY != 0
        );
        Ok((
            head,
            if suppressed {
                NotificationHint::Suppressed
            } else {
                NotificationHint::Notify
            },
        ))
    }

    /// Return the `count` descriptors chained from `head` to the free list.
    fn release(&mut self, head: u16, count: u16) {
        let mut tail = head;
        for _ in 1..count {
            tail = self.next[usize::from(tail)];
        }
        self.next[usize::from(tail)] = self.free_head;
        self.free_head = head;
        self.free += count;
    }
}

impl<M, const N: usize, const R: usize> QueuePort for DriverRing<M, N, R> {
    fn state(&self) -> QueueState {
        self.state
    }
}

impl<M: SharedMemory + Clone + fmt::Debug, const N: usize, const R: usize> QueueControl
    for DriverRing<M, N, R>
{
    type Error = DriverRingError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        if self.state.ready() || self.outstanding != 0 {
            return Err(QueueError::Transport(DriverRingError::Busy));
        }
        if size > self.state.max_size() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::SizeExceedsMaximum,
            ));
        }
        if size.get() < self.max_chain_descriptors {
            return Err(QueueError::Transport(
                DriverRingError::ChainLimitExceedsQueue,
            ));
        }
        self.state = QueueState::new(self.state.max_size(), Some(size), false, self.state.epoch())
            .map_err(QueueError::InvalidConfiguration)?;
        // Every descriptor is free, each linked to the one after it.
        for index in 0..size.get() {
            self.next[usize::from(index)] = index + 1;
        }
        self.free_head = 0;
        self.free = size.get();
        self.fresh = true;
        Ok(())
    }

    /// Making the queue ready checks the layout. The first time after it was
    /// sized or placed it also clears both rings' headers, because the memory
    /// may have held a queue before: do that before the transport is told the
    /// queue is live, while the memory is still only the driver's. A queue
    /// made not ready and ready again is a pause, and resumes where it was.
    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        if ready && !self.state.ready() {
            let Some(size) = self.state.size() else {
                return Err(QueueError::InvalidConfiguration(
                    QueueConfigError::ReadyWithoutSize,
                ));
            };
            if !self.addresses.hold(&self.memory, size, [Access::Write; 3]) {
                return Err(QueueError::Transport(DriverRingError::Layout));
            }
            if self.fresh {
                for ring in [self.addresses.available, self.addresses.used] {
                    self.memory.store_u16(ring + RING_FLAGS, 0)?;
                    self.memory.store_u16(ring + RING_INDEX, 0)?;
                }
                self.published = 0;
                self.taken = 0;
                self.fresh = false;
            }
        }
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

impl<M: SharedMemory + Clone + fmt::Debug, const N: usize, const R: usize> DriverQueue
    for DriverRing<M, N, R>
{
    type Chain = DriverChain<M, R>;
    type Reclaimed = ReclaimedChains<M, N, R>;
    type Error = DriverRingError;

    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
        let size = match self.live() {
            Ok(size) => size.get(),
            Err(QueueError::Transport(failure)) => {
                return Err(PublishError::new(
                    chain,
                    PublishErrorKind::Transport(failure),
                ));
            }
            Err(_) => return Err(PublishError::new(chain, PublishErrorKind::NotReady)),
        };
        // The reference model's classification, kept exactly: the ring is
        // full only when it has no slot for another head, which chains of
        // two descriptors at least never reach; a chain that does not fit
        // the limit, or the descriptors now free, lacks descriptors.
        if self.outstanding == size {
            return Err(PublishError::new(chain, PublishErrorKind::QueueFull));
        }
        if chain.count > self.max_chain_descriptors || chain.count > self.free {
            return Err(PublishError::new(
                chain,
                PublishErrorKind::InsufficientDescriptors,
            ));
        }
        match self.expose(&chain, size) {
            Ok((head, hint)) => {
                let id = ChainId::new(self.state.epoch(), u64::from(head));
                self.held[usize::from(head)] = Some(chain);
                self.outstanding += 1;
                Ok(PublishedChain::new(id, hint))
            }
            Err(fault) => Err(PublishError::new(
                chain,
                PublishErrorKind::Transport(DriverRingError::Memory(fault)),
            )),
        }
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
        let size = self.live()?.get();
        // Acquire: the entry and the response bytes the device wrote before
        // it published this index are visible to the reads below.
        let published = self.memory.load_u16(self.addresses.used + RING_INDEX)?;
        let pending = published.wrapping_sub(self.taken);
        if pending == 0 {
            return Ok(None);
        }
        if pending > self.outstanding {
            return self.fail(DriverRingError::UsedIndex);
        }
        let slot = u64::from(self.taken % size);
        // One snapshot: the entry is judged and acted on from the same bytes.
        let mut entry = [0u8; USED_ENTRY as usize];
        self.memory.read(
            self.addresses.used + RING_HEADER + slot * USED_ENTRY,
            &mut entry,
        )?;
        let head = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let written = u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]);
        let held = if head < u32::from(size) {
            self.held[head as usize].take()
        } else {
            None
        };
        let Some(chain) = held else {
            return self.fail(DriverRingError::UsedEntry);
        };
        if u64::from(written) > chain.writable_bytes {
            // Kept, so a reset still returns it to its owner.
            self.held[head as usize] = Some(chain);
            return self.fail(DriverRingError::UsedEntry);
        }
        let head = head as u16;
        self.taken = self.taken.wrapping_add(1);
        self.outstanding -= 1;
        self.release(head, chain.count);
        Ok(Some(UsedChain::new(
            ChainId::new(self.state.epoch(), u64::from(head)),
            UsedLength::new(written),
            chain,
        )))
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.live()?;
        self.memory.store_u16(
            self.addresses.available + RING_FLAGS,
            VIRTQ_AVAIL_F_NO_INTERRUPT,
        )?;
        Ok(())
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        self.live()?;
        self.memory
            .store_u16(self.addresses.available + RING_FLAGS, 0)?;
        // The flag is cleared before the index is read: a device that used a
        // chain while interrupts were off is seen here, and one that uses a
        // chain after sees the flag clear and interrupts.
        fence(Ordering::SeqCst);
        let published = self.memory.load_u16(self.addresses.used + RING_INDEX)?;
        Ok(if published == self.taken {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        let epoch = self.state.epoch();
        if next_epoch <= epoch {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::NonIncreasingEpoch,
            ));
        }
        let held = core::mem::replace(&mut self.held, [const { None }; N]);
        // Unconfigured: the queue a driver uses after a reset is one it has
        // sized, placed and made ready again.
        self.state = QueueState::unconfigured(self.state.max_size(), next_epoch);
        self.addresses = RingAddresses::default();
        self.free_head = 0;
        self.free = 0;
        self.outstanding = 0;
        self.published = 0;
        self.taken = 0;
        self.failed = None;
        self.fresh = true;
        Ok(ReclaimedChains {
            held,
            epoch,
            next: 0,
        })
    }
}

/// The chains a reset took back from the device, each with the identity it
/// was published under.
#[derive(Debug)]
pub struct ReclaimedChains<M, const N: usize, const R: usize> {
    held: [Option<DriverChain<M, R>>; N],
    epoch: QueueEpoch,
    next: usize,
}

impl<M, const N: usize, const R: usize> Iterator for ReclaimedChains<M, N, R> {
    type Item = ReclaimedChain<DriverChain<M, R>>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < N {
            let head = self.next;
            self.next += 1;
            if let Some(chain) = self.held[head].take() {
                return Some(ReclaimedChain::new(
                    ChainId::new(self.epoch, head as u64),
                    chain,
                ));
            }
        }
        None
    }
}

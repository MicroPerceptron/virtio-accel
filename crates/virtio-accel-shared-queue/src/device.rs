//! The device's half: chains taken from the available ring, and used.

use core::borrow::Borrow;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering, fence};

use virtio_accel_transport::{
    ByteAccessError, ChainError, ChainId, ChainIo, ChainIoResult, ChainRegion, DeviceChain,
    DeviceQueue, MalformedChain, NotificationHint, NotificationRecheck, QueueConfigError,
    QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize, QueueState, ReadableBytes,
    UsedLength, WritableBytes, validate_chain_layout,
};

use crate::layout::{
    AVAILABLE_ENTRY, DESCRIPTOR_BYTES, RING_FLAGS, RING_HEADER, RING_INDEX, RingAddresses,
    USED_ENTRY, VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT,
    VIRTQ_DESC_F_WRITE, VIRTQ_USED_F_NO_NOTIFY, each_span, fields,
};
use crate::memory::{Access, MemoryFault, SharedMemory};

/// Why the device's half could not go on. Each is the driver's doing, and
/// none is one chain's: a device that sees one stops serving the queue and
/// asks for a reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRingError {
    /// A part of the queue is not where the layout allows: misaligned,
    /// overlapping another, or outside memory the device may reach.
    Layout,
    /// The queue was made ready smaller than the longest chain it must hold.
    ChainLimitExceedsQueue,
    /// The queue cannot be reconfigured while it is ready.
    Busy,
    /// The available index is further ahead than the queue is long.
    AvailableIndex,
    /// A head was published that the device already holds.
    HeadInFlight,
    /// The memory stopped answering for a ring the queue validated.
    Memory(MemoryFault),
}

impl From<MemoryFault> for QueueError<DeviceRingError> {
    fn from(fault: MemoryFault) -> Self {
        Self::Transport(DeviceRingError::Memory(fault))
    }
}

/// One descriptor's bytes, snapshotted when its chain was taken.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    at: u64,
    len: u32,
}

/// A chain's spans in one direction, as one run of bytes.
#[derive(Clone, Debug)]
struct Port<M, E, const R: usize> {
    memory: M,
    spans: [Span; R],
    count: u16,
    len: u64,
    epoch: QueueEpoch,
    current: E,
}

impl<M: SharedMemory, E: Borrow<AtomicU64>, const R: usize> Port<M, E, R> {
    fn each(
        &self,
        offset: u64,
        len: usize,
        visit: impl FnMut(u64, usize, usize) -> Result<(), MemoryFault>,
    ) -> Result<(), ByteAccessError> {
        if self.current.borrow().load(Ordering::Acquire) != self.epoch.get() {
            return Err(ByteAccessError::Reset);
        }
        match offset.checked_add(len as u64) {
            Some(end) if end <= self.len => {}
            _ => return Err(ByteAccessError::OutOfBounds),
        }
        let spans = self.spans[..usize::from(self.count)]
            .iter()
            .map(|span| (span.at, span.len));
        each_span(spans, offset, len, visit).map_err(|_| ByteAccessError::Access)
    }

    fn mapped(&self, access: Access) -> bool {
        self.spans[..usize::from(self.count)]
            .iter()
            .all(|span| self.memory.admits(span.at, u64::from(span.len), access))
    }
}

/// The bytes a chain gives the device to read.
#[derive(Debug)]
pub struct RingSource<M, E, const R: usize>(Port<M, E, R>);

impl<M, E, const R: usize> ReadableBytes for RingSource<M, E, R>
where
    M: SharedMemory + fmt::Debug,
    E: Borrow<AtomicU64> + fmt::Debug,
{
    fn len(&self) -> u64 {
        self.0.len
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        let memory = &self.0.memory;
        let len = target.len();
        self.0.each(offset, len, |at, from, count| {
            memory.read(at, &mut target[from..from + count])
        })
    }
}

/// The bytes a chain gives the device to write.
#[derive(Debug)]
pub struct RingSink<M, E, const R: usize>(Port<M, E, R>);

impl<M, E, const R: usize> WritableBytes for RingSink<M, E, R>
where
    M: SharedMemory + fmt::Debug,
    E: Borrow<AtomicU64> + fmt::Debug,
{
    fn len(&self) -> u64 {
        self.0.len
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        let memory = &self.0.memory;
        self.0.each(offset, source.len(), |at, from, count| {
            memory.write(at, &source[from..from + count])
        })
    }
}

/// One chain the device took from the available ring and has not yet used.
/// Not `Clone`: [`DeviceQueue::complete`] consumes it, so a chain is used
/// once.
#[derive(Debug)]
pub struct DeviceRingChain<M, E, const R: usize> {
    id: ChainId,
    head: u16,
    regions: [ChainRegion; R],
    count: u16,
    malformed: Option<MalformedChain>,
    request: RingSource<M, E, R>,
    response: RingSink<M, E, R>,
}

impl<M, E, const R: usize> DeviceChain for DeviceRingChain<M, E, R>
where
    M: SharedMemory + fmt::Debug,
    E: Borrow<AtomicU64> + fmt::Debug,
{
    type Request = RingSource<M, E, R>;
    type Response = RingSink<M, E, R>;
    type Error = DeviceRingError;

    fn id(&self) -> ChainId {
        self.id
    }

    fn io(&mut self) -> ChainIoResult<'_, Self::Request, Self::Response, Self::Error> {
        let current = self.request.0.current.borrow().load(Ordering::Acquire);
        if current != self.id.epoch().get() {
            return Err(ChainError::ResetRace {
                chain: self.id.epoch(),
                // An epoch is never zero: it starts at one and only grows.
                current: QueueEpoch::new(current).unwrap_or(QueueEpoch::INITIAL),
            });
        }
        if let Some(malformed) = self.malformed {
            return Err(ChainError::Malformed(malformed));
        }
        Ok(ChainIo::new(
            &self.regions[..usize::from(self.count)],
            &self.request,
            &mut self.response,
        ))
    }
}

/// The device's half of one split virtqueue in shared memory.
///
/// `N` bounds the queue's size and `R` a chain's descriptors; both are
/// storage carried inline, so the queue allocates nothing. `E` is the cell
/// the queue and every chain it hands out share, so that a reset reaches a
/// chain the device still holds: an `Rc<AtomicU64>` where there is a heap, a
/// `&'static AtomicU64` where there is not. The queue sets it; its value
/// before [`DeviceRing::new`] does not matter.
///
/// # What the device trusts
///
/// Nothing the driver wrote. Every index is checked against the queue's
/// size, a chain is walked at most `size` descriptors with a visited set so
/// a loop ends, every span must lie in memory the device may reach for its
/// direction, and a chain is read **once**: the walk snapshots each
/// descriptor, and the ports then use the snapshot, so a driver that
/// rewrites a descriptor after publishing it changes nothing the device
/// does. A chain that is malformed is not an error of the queue. It is
/// handed up like any other, its [`io`](DeviceChain::io) says why, and the
/// device completes it with a used length of zero, as the protocol's
/// command-chain rules require.
///
/// The profile is the protocol's: direct descriptors only, no indirect
/// tables, no event indices, and the two suppression flags.
#[derive(Debug)]
pub struct DeviceRing<M, E, const N: usize, const R: usize> {
    memory: M,
    state: QueueState,
    max_chain_descriptors: u16,
    addresses: RingAddresses,
    current: E,
    /// Available entries taken, and used entries published.
    taken: u16,
    used: u16,
    /// Heads the device holds.
    in_flight: [bool; N],
    held: u16,
    /// The walk's visited set: a descriptor was met on walk `pass` when its
    /// mark is `pass`. Clearing the set is then one increment.
    mark: [u16; N],
    pass: u16,
}

impl<M, E, const N: usize, const R: usize> DeviceRing<M, E, N, R>
where
    M: SharedMemory + Clone + fmt::Debug,
    E: Borrow<AtomicU64> + Clone + fmt::Debug,
{
    /// A queue in `memory` of at most `max_size` descriptors, whose chains
    /// are at most `max_chain_descriptors` long. Unconfigured: the driver
    /// sets its size and addresses and makes it ready.
    ///
    /// `None` when the limits do not fit: a queue no larger than `N`, and a
    /// chain of two descriptors at least, no longer than `R` or the queue.
    pub fn new(
        memory: M,
        current: E,
        max_size: QueueSize,
        max_chain_descriptors: u16,
    ) -> Option<Self> {
        let fits = usize::from(max_size.get()) <= N
            && max_chain_descriptors >= 2
            && usize::from(max_chain_descriptors) <= R
            && max_chain_descriptors <= max_size.get();
        fits.then(|| {
            let epoch = QueueEpoch::INITIAL;
            current.borrow().store(epoch.get(), Ordering::Release);
            Self {
                memory,
                state: QueueState::unconfigured(max_size, epoch),
                max_chain_descriptors,
                addresses: RingAddresses::default(),
                current,
                taken: 0,
                used: 0,
                in_flight: [false; N],
                held: 0,
                mark: [0; N],
                pass: 0,
            }
        })
    }

    /// The memory the queue is in, given back: the queue is over.
    pub fn into_memory(self) -> M {
        self.memory
    }

    /// Where the driver says the queue's parts are. Taken while the queue is
    /// not ready, and checked when it is made so.
    pub fn set_addresses(
        &mut self,
        addresses: RingAddresses,
    ) -> Result<(), QueueError<DeviceRingError>> {
        if self.state.ready() {
            return Err(QueueError::Transport(DeviceRingError::Busy));
        }
        self.addresses = addresses;
        Ok(())
    }

    /// The addresses the queue was given.
    pub const fn addresses(&self) -> RingAddresses {
        self.addresses
    }

    /// Chains the device holds: taken and not yet used.
    pub const fn in_flight(&self) -> u16 {
        self.held
    }

    fn size(&self) -> Result<u16, QueueError<DeviceRingError>> {
        match (self.state.ready(), self.state.size()) {
            (true, Some(size)) => Ok(size.get()),
            _ => Err(QueueError::NotReady),
        }
    }

    /// Walk the chain at `head`, snapshotting each descriptor once.
    fn take(
        &mut self,
        head: u16,
        size: u16,
    ) -> Result<DeviceRingChain<M, E, R>, QueueError<DeviceRingError>> {
        let epoch = self.state.epoch();
        let port = || Port {
            memory: self.memory.clone(),
            spans: [Span::default(); R],
            count: 0,
            len: 0,
            epoch,
            current: self.current.clone(),
        };
        let mut chain = DeviceRingChain {
            id: ChainId::new(epoch, u64::from(head)),
            head,
            regions: [ChainRegion::readable(0); R],
            count: 0,
            malformed: None,
            request: RingSource(port()),
            response: RingSink(port()),
        };
        chain.malformed = self.walk(&mut chain, head, size)?.err();
        Ok(chain)
    }

    /// A fresh visited set.
    fn begin_pass(&mut self) -> u16 {
        self.pass = self.pass.wrapping_add(1);
        if self.pass == 0 {
            // Every mark a wrapped counter could collide with is forgotten.
            self.mark = [0; N];
            self.pass = 1;
        }
        self.pass
    }

    /// The walk proper. The outer error is the queue's own memory failing;
    /// the inner one is this chain being malformed, which is the chain's
    /// business and not the queue's.
    fn walk(
        &mut self,
        chain: &mut DeviceRingChain<M, E, R>,
        head: u16,
        size: u16,
    ) -> Result<Result<(), MalformedChain>, QueueError<DeviceRingError>> {
        let pass = self.begin_pass();
        let limit = self.max_chain_descriptors;
        let (mut next, mut met) = (head, 0u16);
        // At most `size` descriptors, a hostile loop included: the visited
        // set ends it on the first index met twice.
        for _ in 0..size {
            if next >= size {
                return Ok(Err(MalformedChain::DescriptorIndex));
            }
            if self.mark[usize::from(next)] == pass {
                return Ok(Err(MalformedChain::DescriptorLoop));
            }
            self.mark[usize::from(next)] = pass;
            let mut bytes = [0u8; DESCRIPTOR_BYTES as usize];
            self.memory.read(
                self.addresses.descriptors + u64::from(next) * DESCRIPTOR_BYTES,
                &mut bytes,
            )?;
            let (at, len, flags, following) = fields(&bytes);
            next = following;
            if flags & VIRTQ_DESC_F_INDIRECT != 0 {
                return Ok(Err(MalformedChain::IndirectUnsupported));
            }
            if flags & !(VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE) != 0 {
                return Ok(Err(MalformedChain::DescriptorFlags));
            }
            met += 1;
            // A chain past the limit is still walked to its end, so that it
            // is judged for what is wrong with it; only its first `limit`
            // descriptors are kept, and it is refused for its length below.
            if met <= limit {
                let writable = flags & VIRTQ_DESC_F_WRITE != 0;
                chain.regions[usize::from(chain.count)] = if writable {
                    ChainRegion::writable(u64::from(len))
                } else {
                    ChainRegion::readable(u64::from(len))
                };
                chain.count += 1;
                let port = if writable {
                    &mut chain.response.0
                } else {
                    &mut chain.request.0
                };
                port.spans[usize::from(port.count)] = Span { at, len };
                port.count += 1;
                port.len += u64::from(len);
            }
            if flags & VIRTQ_DESC_F_NEXT == 0 {
                if met > limit {
                    return Ok(Err(MalformedChain::DescriptorCount));
                }
                // Count, zero lengths, direction order and totals are the
                // protocol's own rule, by its own function.
                if let Err(error) =
                    validate_chain_layout(&chain.regions[..usize::from(chain.count)], limit)
                {
                    return Ok(Err(error.into()));
                }
                // Every byte the engine may touch is reachable, for the
                // direction it will be touched in, before the engine runs.
                if !chain.request.0.mapped(Access::Read) || !chain.response.0.mapped(Access::Write)
                {
                    return Ok(Err(MalformedChain::Address));
                }
                return Ok(Ok(()));
            }
        }
        // `size` descriptors and still a `next`: longer than any chain.
        Ok(Err(MalformedChain::DescriptorCount))
    }
}

impl<M, E, const N: usize, const R: usize> QueuePort for DeviceRing<M, E, N, R> {
    fn state(&self) -> QueueState {
        self.state
    }
}

impl<M, E, const N: usize, const R: usize> QueueControl for DeviceRing<M, E, N, R>
where
    M: SharedMemory + Clone + fmt::Debug,
    E: Borrow<AtomicU64> + Clone + fmt::Debug,
{
    type Error = DeviceRingError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        if self.state.ready() {
            return Err(QueueError::Transport(DeviceRingError::Busy));
        }
        if size > self.state.max_size() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::SizeExceedsMaximum,
            ));
        }
        if size.get() < self.max_chain_descriptors {
            return Err(QueueError::Transport(
                DeviceRingError::ChainLimitExceedsQueue,
            ));
        }
        self.state = QueueState::new(self.state.max_size(), Some(size), false, self.state.epoch())
            .map_err(QueueError::InvalidConfiguration)?;
        Ok(())
    }

    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        if ready {
            let Some(size) = self.state.size() else {
                return Err(QueueError::InvalidConfiguration(
                    QueueConfigError::ReadyWithoutSize,
                ));
            };
            // The device reads the descriptor table and the available ring,
            // and writes the used ring.
            let access = [Access::Read, Access::Read, Access::Write];
            if !self.addresses.hold(&self.memory, size, access) {
                return Err(QueueError::Transport(DeviceRingError::Layout));
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

impl<M, E, const N: usize, const R: usize> DeviceQueue for DeviceRing<M, E, N, R>
where
    M: SharedMemory + Clone + fmt::Debug,
    E: Borrow<AtomicU64> + Clone + fmt::Debug,
{
    type Chain = DeviceRingChain<M, E, R>;
    type Error = DeviceRingError;

    fn pop_available(&mut self) -> Result<Option<Self::Chain>, QueueError<Self::Error>> {
        let size = self.size()?;
        // Acquire: the entry and the descriptors the driver wrote before it
        // published this index are visible to the reads below.
        let published = self
            .memory
            .load_u16(self.addresses.available + RING_INDEX)?;
        let pending = published.wrapping_sub(self.taken);
        if pending == 0 {
            return Ok(None);
        }
        if pending > size {
            return Err(QueueError::Transport(DeviceRingError::AvailableIndex));
        }
        let slot = u64::from(self.taken % size);
        let mut entry = [0u8; AVAILABLE_ENTRY as usize];
        self.memory.read(
            self.addresses.available + RING_HEADER + slot * AVAILABLE_ENTRY,
            &mut entry,
        )?;
        let head = u16::from_le_bytes(entry);
        // A head the device already holds is the driver reusing descriptors
        // it does not own. Judged before the entry is taken, so the queue is
        // left exactly as it was.
        if head < size && self.in_flight[usize::from(head)] {
            return Err(QueueError::Transport(DeviceRingError::HeadInFlight));
        }
        let chain = self.take(head, size)?;
        self.taken = self.taken.wrapping_add(1);
        if head < size {
            self.in_flight[usize::from(head)] = true;
            self.held += 1;
        }
        Ok(Some(chain))
    }

    fn complete(
        &mut self,
        chain: Self::Chain,
        used: UsedLength,
    ) -> Result<NotificationHint, QueueError<Self::Error>> {
        let current = self.state.epoch();
        if chain.id.epoch() != current {
            return Err(QueueError::ResetRace {
                operation: chain.id.epoch(),
                current,
            });
        }
        let size = self.size()?;
        let capacity = chain.response.0.len;
        if u64::from(used.get()) > capacity {
            return Err(QueueError::UsedLengthExceeded { used, capacity });
        }
        let slot = u64::from(self.used % size);
        let mut entry = [0u8; USED_ENTRY as usize];
        entry[..4].copy_from_slice(&u32::from(chain.head).to_le_bytes());
        entry[4..].copy_from_slice(&used.get().to_le_bytes());
        self.memory.write(
            self.addresses.used + RING_HEADER + slot * USED_ENTRY,
            &entry,
        )?;
        let published = self.used.wrapping_add(1);
        // Release: the response bytes and the entry above are visible to a
        // driver that has loaded this index.
        self.memory
            .store_u16(self.addresses.used + RING_INDEX, published)?;
        self.used = published;
        if chain.head < size && self.in_flight[usize::from(chain.head)] {
            self.in_flight[usize::from(chain.head)] = false;
            self.held -= 1;
        }
        // The index is published before the driver's suppression flag is
        // read: a driver that clears the flag and then rechecks the index
        // cannot miss this entry, and one that set it is not interrupted. A
        // flag that cannot be read is a driver that is told.
        fence(Ordering::SeqCst);
        let suppressed = matches!(
            self.memory.load_u16(self.addresses.available + RING_FLAGS),
            Ok(flags) if flags & VIRTQ_AVAIL_F_NO_INTERRUPT != 0
        );
        Ok(if suppressed {
            NotificationHint::Suppressed
        } else {
            NotificationHint::Notify
        })
    }

    fn disable_available_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.size()?;
        self.memory
            .store_u16(self.addresses.used + RING_FLAGS, VIRTQ_USED_F_NO_NOTIFY)?;
        Ok(())
    }

    fn enable_available_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        self.size()?;
        self.memory.store_u16(self.addresses.used + RING_FLAGS, 0)?;
        // The flag is cleared before the index is read: a driver that
        // published while notifications were off is seen here, and one that
        // publishes after sees the flag clear and notifies.
        fence(Ordering::SeqCst);
        let published = self
            .memory
            .load_u16(self.addresses.available + RING_INDEX)?;
        Ok(if published == self.taken {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<Self::Error>> {
        if next_epoch <= self.state.epoch() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::NonIncreasingEpoch,
            ));
        }
        // Published first: a chain the device still holds refuses its bytes
        // and its completion from here on, so nothing of the old epoch is
        // written to memory the driver may now reuse.
        self.current
            .borrow()
            .store(next_epoch.get(), Ordering::Release);
        self.state = QueueState::unconfigured(self.state.max_size(), next_epoch);
        self.addresses = RingAddresses::default();
        self.taken = 0;
        self.used = 0;
        self.in_flight = [false; N];
        self.held = 0;
        Ok(())
    }
}

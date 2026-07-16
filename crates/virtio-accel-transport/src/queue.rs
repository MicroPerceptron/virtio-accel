//! Ownership-safe virtqueue lifecycle and notification ports.

use core::num::{NonZeroU16, NonZeroU64};

use crate::{ChainLayoutError, ChainRegion};

/// Maximum queue size permitted by the split-ring index representation.
pub const MAX_SPLIT_QUEUE_SIZE: u16 = 32_768;

/// Valid nonzero, power-of-two split-virtqueue size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct QueueSize(NonZeroU16);

impl QueueSize {
    /// Validate a split-virtqueue size.
    pub const fn new(value: u16) -> Option<Self> {
        if value.is_power_of_two() && value <= MAX_SPLIT_QUEUE_SIZE {
            match NonZeroU16::new(value) {
                Some(value) => Some(Self(value)),
                None => None,
            }
        } else {
            None
        }
    }

    /// Return the validated queue size.
    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Monotonic queue-reset epoch used to reject stale chain operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct QueueEpoch(NonZeroU64);

impl QueueEpoch {
    /// Initial epoch for a newly constructed queue.
    pub const INITIAL: Self = match NonZeroU64::new(1) {
        Some(value) => Self(value),
        None => panic!("one is nonzero"),
    };

    /// Construct a nonzero epoch.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Return the raw epoch value.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Return the next epoch, or `None` if the epoch space is exhausted.
    pub const fn checked_next(self) -> Option<Self> {
        match self.get().checked_add(1) {
            Some(value) => Self::new(value),
            None => None,
        }
    }
}

/// Opaque queue-chain identity scoped to one reset epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChainId {
    epoch: QueueEpoch,
    token: u64,
}

impl ChainId {
    /// Construct an identity from an epoch and transport-owned token.
    pub const fn new(epoch: QueueEpoch, token: u64) -> Self {
        Self { epoch, token }
    }

    /// Epoch in which this chain was published.
    pub const fn epoch(self) -> QueueEpoch {
        self.epoch
    }

    /// Opaque transport token, such as a split-ring descriptor head.
    pub const fn token(self) -> u64 {
        self.token
    }
}

/// Exact number of device-written bytes published with a used chain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct UsedLength(u32);

impl UsedLength {
    /// Construct an exact used length.
    pub const fn new(bytes: u32) -> Self {
        Self(bytes)
    }

    /// Return the used byte count.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Validated queue configuration and lifecycle snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueState {
    max_size: QueueSize,
    size: Option<QueueSize>,
    ready: bool,
    epoch: QueueEpoch,
}

impl QueueState {
    /// Construct an unconfigured queue state.
    pub const fn unconfigured(max_size: QueueSize, epoch: QueueEpoch) -> Self {
        Self {
            max_size,
            size: None,
            ready: false,
            epoch,
        }
    }

    /// Construct and validate a queue-state snapshot.
    pub const fn new(
        max_size: QueueSize,
        size: Option<QueueSize>,
        ready: bool,
        epoch: QueueEpoch,
    ) -> Result<Self, QueueConfigError> {
        if let Some(size) = size {
            if size.get() > max_size.get() {
                return Err(QueueConfigError::SizeExceedsMaximum);
            }
        }
        if ready && size.is_none() {
            return Err(QueueConfigError::ReadyWithoutSize);
        }
        Ok(Self {
            max_size,
            size,
            ready,
            epoch,
        })
    }

    /// Maximum queue size supported by the implementation.
    pub const fn max_size(self) -> QueueSize {
        self.max_size
    }

    /// Configured queue size, if any.
    pub const fn size(self) -> Option<QueueSize> {
        self.size
    }

    /// Whether descriptor publication and consumption are enabled.
    pub const fn ready(self) -> bool {
        self.ready
    }

    /// Current reset epoch.
    pub const fn epoch(self) -> QueueEpoch {
        self.epoch
    }
}

/// Invalid queue configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueConfigError {
    /// The configured queue size exceeds the implementation maximum.
    SizeExceedsMaximum,
    /// A queue cannot become ready before a size is configured.
    ReadyWithoutSize,
    /// A reset attempted to reuse or move backwards from the current epoch.
    NonIncreasingEpoch,
}

/// Portable queue operation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueError<E> {
    /// The queue is not configured and ready for the requested operation.
    NotReady,
    /// The queue configuration is invalid.
    InvalidConfiguration(QueueConfigError),
    /// An operation belongs to a different reset epoch.
    ResetRace {
        /// Epoch carried by the operation.
        operation: QueueEpoch,
        /// Current queue epoch.
        current: QueueEpoch,
    },
    /// Completion used length exceeds writable chain capacity.
    UsedLengthExceeded {
        /// Attempted used length.
        used: UsedLength,
        /// Writable capacity of the consumed chain.
        capacity: u64,
    },
    /// Concrete transport failure.
    Transport(E),
}

/// Structurally malformed descriptor-chain classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MalformedChain {
    /// A descriptor chain contains a loop.
    DescriptorLoop,
    /// A descriptor index is outside the configured queue.
    DescriptorIndex,
    /// A descriptor contains flags outside the negotiated split-ring profile.
    DescriptorFlags,
    /// The flattened descriptor count is invalid.
    DescriptorCount,
    /// A descriptor has zero length.
    ZeroLength,
    /// Readable and writable descriptor ordering is invalid.
    Direction,
    /// Descriptor byte totals overflow.
    LengthOverflow,
    /// Mapped byte ports do not match descriptor byte totals.
    PortLengthMismatch,
    /// An indirect descriptor was offered without negotiation.
    IndirectUnsupported,
    /// An indirect descriptor table is structurally invalid.
    IndirectMalformed,
    /// A descriptor range cannot be mapped for the required access.
    Address,
}

impl From<ChainLayoutError> for MalformedChain {
    fn from(error: ChainLayoutError) -> Self {
        match error {
            ChainLayoutError::DescriptorCount => Self::DescriptorCount,
            ChainLayoutError::ZeroLength => Self::ZeroLength,
            ChainLayoutError::Direction => Self::Direction,
            ChainLayoutError::LengthOverflow => Self::LengthOverflow,
            ChainLayoutError::PortLengthMismatch => Self::PortLengthMismatch,
        }
    }
}

/// Failure to expose one consumed chain as validated byte ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainError<E> {
    /// The chain is malformed and should be completed with used length zero when recoverable.
    Malformed(MalformedChain),
    /// Reset invalidated this chain before byte access.
    ResetRace {
        /// Epoch carried by the chain.
        chain: QueueEpoch,
        /// Current queue epoch.
        current: QueueEpoch,
    },
    /// Concrete transport or memory-access failure.
    Transport(E),
}

/// Borrowed regions and byte ports for one consumed device chain.
///
/// The source and sink types remain generic so this crate does not depend on command semantics or a
/// particular guest-memory library.
#[derive(Debug)]
pub struct ChainIo<'a, R: ?Sized, W: ?Sized> {
    regions: &'a [ChainRegion],
    request: &'a R,
    response: &'a mut W,
}

impl<'a, R: ?Sized, W: ?Sized> ChainIo<'a, R, W> {
    /// Construct a borrowed chain view after topology and mapping validation.
    pub fn new(regions: &'a [ChainRegion], request: &'a R, response: &'a mut W) -> Self {
        Self {
            regions,
            request,
            response,
        }
    }

    /// Consume the view into its region metadata, readable port, and writable port.
    pub fn into_parts(self) -> (&'a [ChainRegion], &'a R, &'a mut W) {
        (self.regions, self.request, self.response)
    }
}

/// Result of exposing one consumed chain as mapped byte ports.
pub type ChainIoResult<'a, R, W, E> = Result<ChainIo<'a, R, W>, ChainError<E>>;

/// Consumed device-side descriptor chain.
///
/// Implementations must keep the mapped bytes alive until this value is consumed by
/// [`DeviceQueue::complete`] or dropped during reset. The value must not be `Copy`; consuming it at
/// completion makes double completion impossible in safe implementations.
pub trait DeviceChain {
    /// Device-readable byte-port type.
    type Request: ?Sized;
    /// Device-writable byte-port type.
    type Response: ?Sized;
    /// Concrete transport or memory-access error.
    type Error;

    /// Return this chain's reset-scoped identity.
    ///
    /// This operation must not block or allocate.
    fn id(&self) -> ChainId;

    /// Expose mapped request and response ports.
    ///
    /// This operation must not block or allocate. A reset-race result must expose no guest bytes.
    fn io(&mut self) -> ChainIoResult<'_, Self::Request, Self::Response, Self::Error>;
}

/// Whether the peer should be notified after a publication operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotificationHint {
    /// Notification is suppressed by current queue state.
    Suppressed,
    /// The peer should be notified after the publication barrier.
    Notify,
}

/// Result of atomically enabling notifications and rechecking queue state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotificationRecheck {
    /// No work appeared before notification enablement became visible.
    Idle,
    /// Work is already pending and must be processed without sleeping.
    WorkPending,
}

/// Read-only state shared by driver, device, and configuration ports.
pub trait QueuePort {
    /// Return a coherent queue-state snapshot.
    ///
    /// This operation must not block or allocate. Implementations using shared state must use
    /// acquire ordering sufficient to observe the corresponding configuration and reset writes.
    fn state(&self) -> QueueState;
}

/// Queue configuration control plane.
pub trait QueueControl: QueuePort {
    /// Concrete transport configuration error.
    type Error;

    /// Configure queue size while the queue is not ready.
    ///
    /// This operation must not block. It may allocate storage bounded by `size`; implementations
    /// must report allocation failure through `Error` without changing the prior configuration.
    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>>;

    /// Enable or disable queue descriptor processing.
    ///
    /// This operation must not block or allocate. Making the queue ready is a release boundary for
    /// the configured queue state.
    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>>;
}

/// Successful driver publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublishedChain {
    id: ChainId,
    notification: NotificationHint,
}

impl PublishedChain {
    /// Construct a publication result.
    pub const fn new(id: ChainId, notification: NotificationHint) -> Self {
        Self { id, notification }
    }

    /// Published chain identity.
    pub const fn id(self) -> ChainId {
        self.id
    }

    /// Notification decision made after publishing the available index.
    pub const fn notification(self) -> NotificationHint {
        self.notification
    }
}

/// Pre-publication failure classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishErrorKind<E> {
    /// The queue is not configured and ready.
    NotReady,
    /// No free descriptor head or available-ring slot exists.
    QueueFull,
    /// The supplied command lacks enough descriptors or writable capacity.
    InsufficientDescriptors,
    /// Reset invalidated publication before ownership could transfer.
    ResetRace {
        /// Epoch in which publication began.
        operation: QueueEpoch,
        /// Current queue epoch.
        current: QueueEpoch,
    },
    /// Concrete transport failure before ownership transfer.
    Transport(E),
}

/// Failed driver publication that returns the unpublished chain for retry or reclamation.
#[derive(Debug, PartialEq, Eq)]
pub struct PublishError<C, E> {
    chain: C,
    kind: PublishErrorKind<E>,
}

impl<C, E> PublishError<C, E> {
    /// Construct a publication failure that retains caller ownership.
    pub const fn new(chain: C, kind: PublishErrorKind<E>) -> Self {
        Self { chain, kind }
    }

    /// Borrow the unpublished chain.
    pub const fn chain(&self) -> &C {
        &self.chain
    }

    /// Borrow the failure classification.
    pub const fn kind(&self) -> &PublishErrorKind<E> {
        &self.kind
    }

    /// Recover the unpublished chain and failure classification.
    pub fn into_parts(self) -> (C, PublishErrorKind<E>) {
        (self.chain, self.kind)
    }
}

/// Driver-owned chain returned by used-ring consumption.
#[derive(Debug, PartialEq, Eq)]
pub struct UsedChain<C> {
    id: ChainId,
    used: UsedLength,
    chain: C,
}

impl<C> UsedChain<C> {
    /// Construct a used-chain result.
    pub const fn new(id: ChainId, used: UsedLength, chain: C) -> Self {
        Self { id, used, chain }
    }

    /// Returned chain identity.
    pub const fn id(&self) -> ChainId {
        self.id
    }

    /// Exact number of bytes written by the device.
    pub const fn used(&self) -> UsedLength {
        self.used
    }

    /// Borrow the returned driver chain.
    pub const fn chain(&self) -> &C {
        &self.chain
    }

    /// Recover the identity, used length, and driver chain.
    pub fn into_parts(self) -> (ChainId, UsedLength, C) {
        (self.id, self.used, self.chain)
    }
}

/// Published driver chain reclaimed during reset without a used entry.
#[derive(Debug, PartialEq, Eq)]
pub struct ReclaimedChain<C> {
    id: ChainId,
    chain: C,
}

impl<C> ReclaimedChain<C> {
    /// Construct a reset-reclamation result.
    pub const fn new(id: ChainId, chain: C) -> Self {
        Self { id, chain }
    }

    /// Reclaimed chain identity.
    pub const fn id(&self) -> ChainId {
        self.id
    }

    /// Recover the identity and driver chain.
    pub fn into_parts(self) -> (ChainId, C) {
        (self.id, self.chain)
    }
}

/// Driver-side descriptor publication and used-ring consumption.
///
/// A successful [`DriverQueue::publish`] transfers exclusive chain ownership to the queue until
/// [`DriverQueue::pop_used`] or [`DriverQueue::reset`] returns it. Implementations must publish all
/// descriptor contents and the available index before returning a notification decision. A
/// successful used pop is an acquire boundary for response bytes written before device completion.
pub trait DriverQueue: QueuePort {
    /// Driver-owned command and descriptor resources.
    type Chain;
    /// Owned collection of chains reclaimed by reset.
    type Reclaimed: IntoIterator<Item = ReclaimedChain<Self::Chain>>;
    /// Concrete transport failure.
    type Error;

    /// Publish a complete chain or return it unchanged on pre-publication backpressure.
    ///
    /// This operation must not block or allocate. Success is a release boundary for descriptor and
    /// request writes. Failure must not transfer ownership or expose a partial chain to the device.
    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>>;

    /// Consume one used entry and recover the corresponding driver chain.
    ///
    /// This operation must not block or allocate. It is an acquire boundary for the used entry and
    /// response bytes. Independent completions may be returned out of publication order.
    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>>;

    /// Disable used-ring notifications before draining completions.
    ///
    /// This operation must not block or allocate.
    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>>;

    /// Enable used-ring notifications and atomically recheck for missed completions.
    ///
    /// This operation must not block or allocate. `WorkPending` requires the caller to continue
    /// draining rather than sleep.
    fn enable_used_notifications(&mut self)
    -> Result<NotificationRecheck, QueueError<Self::Error>>;

    /// Invalidate the old epoch and recover every chain still owned by the queue.
    ///
    /// This operation must not block or allocate. It may move existing queue-owned storage into the
    /// returned collection. `next_epoch` must be greater than the current epoch. No old used entry
    /// or response write may become visible after success.
    fn reset(&mut self, next_epoch: QueueEpoch)
    -> Result<Self::Reclaimed, QueueError<Self::Error>>;
}

/// Device-side available-ring consumption and completion publication.
///
/// A successful [`DeviceQueue::pop_available`] acquires request and descriptor writes made before
/// driver publication. [`DeviceQueue::complete`] consumes the chain, publishes response bytes and
/// the used index with release ordering, and only then returns a notification decision.
pub trait DeviceQueue: QueuePort {
    /// Owned consumed-chain type.
    type Chain: DeviceChain;
    /// Concrete transport failure.
    type Error;

    /// Consume one available descriptor chain.
    ///
    /// This operation must not block or allocate. Chains are returned in available-ring order.
    fn pop_available(&mut self) -> Result<Option<Self::Chain>, QueueError<Self::Error>>;

    /// Publish one chain completion and consume its ownership token.
    ///
    /// This operation must not block or allocate. It must reject a stale epoch without writing
    /// guest bytes, a used entry, or a notification. `used` must not exceed writable capacity.
    fn complete(
        &mut self,
        chain: Self::Chain,
        used: UsedLength,
    ) -> Result<NotificationHint, QueueError<Self::Error>>;

    /// Disable available-ring notifications before draining work.
    ///
    /// This operation must not block or allocate.
    fn disable_available_notifications(&mut self) -> Result<(), QueueError<Self::Error>>;

    /// Enable available-ring notifications and atomically recheck for missed work.
    ///
    /// This operation must not block or allocate. `WorkPending` requires the caller to continue
    /// draining rather than sleep.
    fn enable_available_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>>;

    /// Invalidate all consumed chains from the old epoch and return the queue to its base state.
    ///
    /// This operation must not block or allocate. `next_epoch` must be greater than the current
    /// epoch. No later completion from the old epoch may write guest memory or publish a used entry.
    fn reset(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<Self::Error>>;
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::vec::Vec;

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct TestPayload {
        request: [u8; 4],
        response: [u8; 4],
    }

    #[derive(Debug)]
    struct TestDeviceChain {
        id: ChainId,
        current_epoch: Arc<AtomicU64>,
        payload: TestPayload,
    }

    impl DeviceChain for TestDeviceChain {
        type Request = [u8; 4];
        type Response = [u8; 4];
        type Error = ();

        fn id(&self) -> ChainId {
            self.id
        }

        fn io(
            &mut self,
        ) -> Result<ChainIo<'_, Self::Request, Self::Response>, ChainError<Self::Error>> {
            let current = QueueEpoch::new(self.current_epoch.load(Ordering::Acquire)).unwrap();
            if current != self.id.epoch() {
                return Err(ChainError::ResetRace {
                    chain: self.id.epoch(),
                    current,
                });
            }
            static REGIONS: [ChainRegion; 2] = [ChainRegion::readable(4), ChainRegion::writable(4)];
            Ok(ChainIo::new(
                &REGIONS,
                &self.payload.request,
                &mut self.payload.response,
            ))
        }
    }

    struct TestQueue {
        state: QueueState,
        current_epoch: Arc<AtomicU64>,
        next_token: u64,
        available: VecDeque<(ChainId, TestPayload)>,
        used: VecDeque<UsedChain<TestPayload>>,
    }

    impl TestQueue {
        fn new() -> Self {
            let epoch = QueueEpoch::INITIAL;
            Self {
                state: QueueState::unconfigured(QueueSize::new(8).unwrap(), epoch),
                current_epoch: Arc::new(AtomicU64::new(epoch.get())),
                next_token: 0,
                available: VecDeque::new(),
                used: VecDeque::new(),
            }
        }

        fn check_ready(&self) -> Result<(), QueueError<()>> {
            if self.state.ready() {
                Ok(())
            } else {
                Err(QueueError::NotReady)
            }
        }

        fn advance_epoch(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<()>> {
            if next_epoch <= self.state.epoch() {
                return Err(QueueError::InvalidConfiguration(
                    QueueConfigError::NonIncreasingEpoch,
                ));
            }
            self.state = QueueState::unconfigured(self.state.max_size(), next_epoch);
            self.current_epoch
                .store(next_epoch.get(), Ordering::Release);
            Ok(())
        }
    }

    impl QueuePort for TestQueue {
        fn state(&self) -> QueueState {
            self.state
        }
    }

    impl QueueControl for TestQueue {
        type Error = ();

        fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
            if self.state.ready() || size > self.state.max_size() {
                return Err(QueueError::InvalidConfiguration(
                    QueueConfigError::SizeExceedsMaximum,
                ));
            }
            self.state =
                QueueState::new(self.state.max_size(), Some(size), false, self.state.epoch())
                    .unwrap();
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

    impl DriverQueue for TestQueue {
        type Chain = TestPayload;
        type Reclaimed = Vec<ReclaimedChain<Self::Chain>>;
        type Error = ();

        fn publish(
            &mut self,
            chain: Self::Chain,
        ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
            if !self.state.ready() {
                return Err(PublishError::new(chain, PublishErrorKind::NotReady));
            }
            if self.available.len() >= usize::from(self.state.size().unwrap().get()) {
                return Err(PublishError::new(chain, PublishErrorKind::QueueFull));
            }
            let id = ChainId::new(self.state.epoch(), self.next_token);
            self.next_token += 1;
            self.available.push_back((id, chain));
            Ok(PublishedChain::new(id, NotificationHint::Notify))
        }

        fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
            self.check_ready()?;
            Ok(self.used.pop_front())
        }

        fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
            self.check_ready()
        }

        fn enable_used_notifications(
            &mut self,
        ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
            self.check_ready()?;
            Ok(if self.used.is_empty() {
                NotificationRecheck::Idle
            } else {
                NotificationRecheck::WorkPending
            })
        }

        fn reset(
            &mut self,
            next_epoch: QueueEpoch,
        ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
            self.advance_epoch(next_epoch)?;
            let reclaimed = self
                .available
                .drain(..)
                .map(|(id, chain)| ReclaimedChain::new(id, chain))
                .collect();
            self.used.clear();
            Ok(reclaimed)
        }
    }

    impl DeviceQueue for TestQueue {
        type Chain = TestDeviceChain;
        type Error = ();

        fn pop_available(&mut self) -> Result<Option<Self::Chain>, QueueError<Self::Error>> {
            self.check_ready()?;
            Ok(self
                .available
                .pop_front()
                .map(|(id, payload)| TestDeviceChain {
                    id,
                    current_epoch: Arc::clone(&self.current_epoch),
                    payload,
                }))
        }

        fn complete(
            &mut self,
            chain: Self::Chain,
            used: UsedLength,
        ) -> Result<NotificationHint, QueueError<Self::Error>> {
            if chain.id.epoch() != self.state.epoch() {
                return Err(QueueError::ResetRace {
                    operation: chain.id.epoch(),
                    current: self.state.epoch(),
                });
            }
            if u64::from(used.get()) > chain.payload.response.len() as u64 {
                return Err(QueueError::UsedLengthExceeded {
                    used,
                    capacity: chain.payload.response.len() as u64,
                });
            }
            self.used
                .push_back(UsedChain::new(chain.id, used, chain.payload));
            Ok(NotificationHint::Notify)
        }

        fn disable_available_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
            self.check_ready()
        }

        fn enable_available_notifications(
            &mut self,
        ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
            self.check_ready()?;
            Ok(if self.available.is_empty() {
                NotificationRecheck::Idle
            } else {
                NotificationRecheck::WorkPending
            })
        }

        fn reset(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<Self::Error>> {
            self.advance_epoch(next_epoch)?;
            self.available.clear();
            self.used.clear();
            Ok(())
        }
    }

    #[test]
    fn queue_sizes_and_states_are_validated() {
        assert_eq!(QueueSize::new(0), None);
        assert_eq!(QueueSize::new(3), None);
        assert_eq!(QueueSize::new(MAX_SPLIT_QUEUE_SIZE + 1), None);

        let max = QueueSize::new(8).unwrap();
        let too_large = QueueSize::new(16).unwrap();
        assert_eq!(
            QueueState::new(max, Some(too_large), false, QueueEpoch::INITIAL),
            Err(QueueConfigError::SizeExceedsMaximum)
        );
        assert_eq!(
            QueueState::new(max, None, true, QueueEpoch::INITIAL),
            Err(QueueConfigError::ReadyWithoutSize)
        );
    }

    #[test]
    fn publication_transfers_ownership_until_used() {
        let mut queue = TestQueue::new();
        QueueControl::configure(&mut queue, QueueSize::new(2).unwrap()).unwrap();
        QueueControl::set_ready(&mut queue, true).unwrap();

        let published = DriverQueue::publish(
            &mut queue,
            TestPayload {
                request: *b"ping",
                response: [0; 4],
            },
        )
        .unwrap();
        assert_eq!(published.notification(), NotificationHint::Notify);

        let mut chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        assert_eq!(chain.id(), published.id());
        let (regions, request, response) = chain.io().unwrap().into_parts();
        assert_eq!(regions.len(), 2);
        assert_eq!(request, b"ping");
        response.copy_from_slice(b"pong");
        DeviceQueue::complete(&mut queue, chain, UsedLength::new(4)).unwrap();

        let used = DriverQueue::pop_used(&mut queue).unwrap().unwrap();
        assert_eq!(used.id(), published.id());
        assert_eq!(used.used(), UsedLength::new(4));
        assert_eq!(used.chain().response, *b"pong");
    }

    #[test]
    fn publication_backpressure_returns_the_chain() {
        let mut queue = TestQueue::new();
        let payload = TestPayload {
            request: *b"ping",
            response: [0; 4],
        };
        let error = DriverQueue::publish(&mut queue, payload).unwrap_err();
        let (payload, kind) = error.into_parts();
        assert_eq!(kind, PublishErrorKind::NotReady);
        assert_eq!(payload.request, *b"ping");
    }

    #[test]
    fn reset_epoch_rejects_late_completion() {
        let mut queue = TestQueue::new();
        QueueControl::configure(&mut queue, QueueSize::new(2).unwrap()).unwrap();
        QueueControl::set_ready(&mut queue, true).unwrap();
        DriverQueue::publish(
            &mut queue,
            TestPayload {
                request: *b"ping",
                response: [0; 4],
            },
        )
        .unwrap();
        let mut chain = DeviceQueue::pop_available(&mut queue).unwrap().unwrap();
        let old_epoch = chain.id().epoch();
        let next_epoch = old_epoch.checked_next().unwrap();

        DeviceQueue::reset(&mut queue, next_epoch).unwrap();
        assert!(matches!(
            chain.io(),
            Err(ChainError::ResetRace { chain, current })
                if chain == old_epoch && current == next_epoch
        ));
        assert_eq!(
            DeviceQueue::complete(&mut queue, chain, UsedLength::new(0)),
            Err(QueueError::ResetRace {
                operation: old_epoch,
                current: next_epoch,
            })
        );
        assert!(queue.used.is_empty());
    }
}

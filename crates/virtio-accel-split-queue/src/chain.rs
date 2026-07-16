use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::min;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use virtio_accel_transport::{
    ByteAccessError, ChainError, ChainId, ChainIo, ChainIoResult, ChainLayout, ChainRegion,
    DeviceChain, MAX_SPLIT_QUEUE_SIZE, MalformedChain, QueueEpoch, ReadableBytes, WritableBytes,
    validate_chain_layout,
};

/// Descriptor continues through its `next` field.
pub const VIRTQ_DESC_F_NEXT: u16 = 1;
/// Descriptor bytes are device-writable rather than device-readable.
pub const VIRTQ_DESC_F_WRITE: u16 = 2;
/// Descriptor points at an indirect descriptor table.
pub const VIRTQ_DESC_F_INDIRECT: u16 = 4;

const KNOWN_DESCRIPTOR_FLAGS: u16 = VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_INDIRECT;

#[derive(Debug)]
enum BufferStorage {
    Mapped(Rc<RefCell<Box<[u8]>>>),
    Unmapped(u64),
}

impl BufferStorage {
    fn mapped(bytes: Vec<u8>) -> Self {
        Self::Mapped(Rc::new(RefCell::new(bytes.into_boxed_slice())))
    }

    fn len(&self) -> u64 {
        match self {
            Self::Mapped(bytes) => bytes.borrow().len() as u64,
            Self::Unmapped(bytes) => *bytes,
        }
    }

    const fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        let Self::Mapped(bytes) = self else {
            return Err(ByteAccessError::Access);
        };
        let bytes = bytes.try_borrow().map_err(|_| ByteAccessError::Busy)?;
        let range = checked_range(offset, target.len(), bytes.len())?;
        target.copy_from_slice(&bytes[range]);
        Ok(())
    }

    fn write_at(&self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        let Self::Mapped(bytes) = self else {
            return Err(ByteAccessError::Access);
        };
        let mut bytes = bytes.try_borrow_mut().map_err(|_| ByteAccessError::Busy)?;
        let range = checked_range(offset, source.len(), bytes.len())?;
        bytes[range].copy_from_slice(source);
        Ok(())
    }
}

/// One address-free split-ring descriptor used by the in-memory model.
#[derive(Debug)]
pub struct Descriptor {
    buffer: BufferStorage,
    flags: u16,
    next: u16,
}

impl Descriptor {
    /// Construct one mapped device-readable descriptor.
    pub fn readable(bytes: Vec<u8>) -> Self {
        Self::raw(bytes, 0, 0)
    }

    /// Construct one mapped device-writable descriptor.
    pub fn writable(bytes: Vec<u8>) -> Self {
        Self::raw(bytes, VIRTQ_DESC_F_WRITE, 0)
    }

    /// Construct a mapped descriptor with raw split-ring flags and a local next index.
    pub fn raw(bytes: Vec<u8>, flags: u16, next: u16) -> Self {
        Self {
            buffer: BufferStorage::mapped(bytes),
            flags,
            next,
        }
    }

    /// Construct an unmapped descriptor for deterministic addressability tests.
    pub const fn unmapped(bytes: u64, flags: u16, next: u16) -> Self {
        Self {
            buffer: BufferStorage::Unmapped(bytes),
            flags,
            next,
        }
    }

    /// Descriptor length.
    pub fn len(&self) -> u64 {
        self.buffer.len()
    }

    /// Whether this descriptor has zero length.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw split-ring flags.
    pub const fn flags(&self) -> u16 {
        self.flags
    }

    /// Local next index used by [`DriverChain::raw`].
    pub const fn next(&self) -> u16 {
        self.next
    }

    const fn is_writable(&self) -> bool {
        self.flags & VIRTQ_DESC_F_WRITE != 0
    }
}

/// Failure while constructing driver-owned chain storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainBuildError {
    /// A chain must contain at least one descriptor.
    Empty,
    /// The supplied descriptor vector exceeds the maximum split queue size.
    DescriptorCount,
    /// Allocation of bounded chain metadata failed.
    AllocationFailed,
    /// A driver-created direct chain is malformed.
    Malformed(MalformedChain),
}

#[derive(Debug)]
enum ChainAnalysis {
    Valid {
        layout: ChainLayout,
        order: Box<[u16]>,
        regions: Box<[ChainRegion]>,
    },
    Invalid(MalformedChain),
}

impl ChainAnalysis {
    const fn validation(&self) -> Result<ChainLayout, MalformedChain> {
        match self {
            Self::Valid { layout, .. } => Ok(*layout),
            Self::Invalid(error) => Err(*error),
        }
    }

    fn order(&self) -> &[u16] {
        match self {
            Self::Valid { order, .. } => order,
            Self::Invalid(_) => &[],
        }
    }

    fn regions(&self) -> &[ChainRegion] {
        match self {
            Self::Valid { regions, .. } => regions,
            Self::Invalid(_) => &[],
        }
    }
}

#[derive(Debug)]
pub(crate) struct ChainData {
    descriptors: Box<[Descriptor]>,
    analysis: ChainAnalysis,
}

/// Driver-owned descriptor chain and buffers.
///
/// Construction allocates metadata proportional only to the caller-provided descriptor vector.
/// Queue publication, completion, and reset do not allocate or copy payload bytes.
#[derive(Debug)]
pub struct DriverChain {
    data: Rc<ChainData>,
    slots: Box<[u16]>,
    head: u16,
}

impl DriverChain {
    /// Construct a valid direct chain linked in vector order.
    pub fn direct(mut descriptors: Vec<Descriptor>) -> Result<Self, ChainBuildError> {
        let descriptor_count = descriptors.len();
        for (index, descriptor) in descriptors.iter_mut().enumerate() {
            descriptor.flags &= VIRTQ_DESC_F_WRITE;
            if index + 1 < descriptor_count {
                descriptor.flags |= VIRTQ_DESC_F_NEXT;
                descriptor.next = (index + 1) as u16;
            } else {
                descriptor.next = 0;
            }
        }
        let chain = Self::raw(descriptors, 0)?;
        if let Err(error) = chain.validation() {
            return Err(ChainBuildError::Malformed(error));
        }
        Ok(chain)
    }

    /// Construct a raw descriptor table for deterministic malformed-chain injection.
    ///
    /// `next` values are local indices in `descriptors`. Structural errors are retained and later
    /// exposed by [`SplitDeviceChain::io`] when the chain is injected into a queue.
    pub fn raw(descriptors: Vec<Descriptor>, head: u16) -> Result<Self, ChainBuildError> {
        if descriptors.is_empty() {
            return Err(ChainBuildError::Empty);
        }
        if descriptors.len() > usize::from(MAX_SPLIT_QUEUE_SIZE) {
            return Err(ChainBuildError::DescriptorCount);
        }

        let mut slots = zeroed_u16_box(descriptors.len())?;
        let analysis = analyze_chain(&descriptors, head, &mut slots)?;
        Ok(Self {
            data: Rc::new(ChainData {
                descriptors: descriptors.into_boxed_slice(),
                analysis,
            }),
            slots,
            head,
        })
    }

    /// Number of descriptor-table entries owned by this chain.
    pub fn descriptor_count(&self) -> u16 {
        self.data.descriptors.len() as u16
    }

    /// Validate descriptor topology independently of a queue's configured chain limit.
    pub fn validation(&self) -> Result<ChainLayout, MalformedChain> {
        self.data.analysis.validation()
    }

    /// Read bytes from one local descriptor after the queue returns ownership.
    pub fn read_descriptor(
        &self,
        index: u16,
        offset: u64,
        target: &mut [u8],
    ) -> Result<(), ByteAccessError> {
        self.descriptor(index)?.buffer.read_at(offset, target)
    }

    /// Write bytes into one local descriptor while the driver owns the chain.
    pub fn write_descriptor(
        &self,
        index: u16,
        offset: u64,
        source: &[u8],
    ) -> Result<(), ByteAccessError> {
        self.descriptor(index)?.buffer.write_at(offset, source)
    }

    fn descriptor(&self, index: u16) -> Result<&Descriptor, ByteAccessError> {
        self.data
            .descriptors
            .get(usize::from(index))
            .ok_or(ByteAccessError::OutOfBounds)
    }

    pub(crate) fn data(&self) -> Rc<ChainData> {
        Rc::clone(&self.data)
    }

    pub(crate) fn slots(&self) -> &[u16] {
        &self.slots
    }

    pub(crate) fn slots_mut(&mut self) -> &mut [u16] {
        &mut self.slots
    }

    pub(crate) fn queue_head_slot(&self) -> u16 {
        self.slots
            .get(usize::from(self.head))
            .copied()
            .unwrap_or(self.slots[0])
    }
}

/// Device-readable concatenation of a valid chain's readable descriptors.
pub struct SplitSource {
    data: Rc<ChainData>,
    epoch: QueueEpoch,
    current_epoch: Rc<AtomicU64>,
}

impl fmt::Debug for SplitSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SplitSource")
            .field("len", &self.len())
            .field("epoch", &self.epoch)
            .finish()
    }
}

impl ReadableBytes for SplitSource {
    fn len(&self) -> u64 {
        self.data
            .analysis
            .validation()
            .map_or(0, ChainLayout::readable_bytes)
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        self.check_epoch()?;
        checked_logical_range(offset, target.len(), self.len())?;
        copy_from_descriptors(&self.data, false, offset, target)
    }
}

impl SplitSource {
    fn check_epoch(&self) -> Result<(), ByteAccessError> {
        if self.current_epoch.load(Ordering::Acquire) == self.epoch.get() {
            Ok(())
        } else {
            Err(ByteAccessError::Reset)
        }
    }
}

/// Device-writable concatenation of a valid chain's writable descriptors.
pub struct SplitSink {
    data: Rc<ChainData>,
    epoch: QueueEpoch,
    current_epoch: Rc<AtomicU64>,
}

impl fmt::Debug for SplitSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SplitSink")
            .field("len", &self.len())
            .field("epoch", &self.epoch)
            .finish()
    }
}

impl WritableBytes for SplitSink {
    fn len(&self) -> u64 {
        self.data
            .analysis
            .validation()
            .map_or(0, ChainLayout::writable_bytes)
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        self.check_epoch()?;
        checked_logical_range(offset, source.len(), self.len())?;
        copy_to_descriptors(&self.data, offset, source)
    }
}

impl SplitSink {
    fn check_epoch(&self) -> Result<(), ByteAccessError> {
        if self.current_epoch.load(Ordering::Acquire) == self.epoch.get() {
            Ok(())
        } else {
            Err(ByteAccessError::Reset)
        }
    }
}

/// Non-copyable device ownership token for one available descriptor chain.
#[derive(Debug)]
pub struct SplitDeviceChain {
    id: ChainId,
    data: Rc<ChainData>,
    max_descriptors: u16,
    source: SplitSource,
    sink: SplitSink,
}

impl SplitDeviceChain {
    pub(crate) fn new(
        id: ChainId,
        data: Rc<ChainData>,
        max_descriptors: u16,
        current_epoch: Rc<AtomicU64>,
    ) -> Self {
        let source = SplitSource {
            data: Rc::clone(&data),
            epoch: id.epoch(),
            current_epoch: Rc::clone(&current_epoch),
        };
        let sink = SplitSink {
            data: Rc::clone(&data),
            epoch: id.epoch(),
            current_epoch,
        };
        Self {
            id,
            data,
            max_descriptors,
            source,
            sink,
        }
    }

    pub(crate) fn writable_capacity(&self) -> u64 {
        self.data
            .analysis
            .validation()
            .map_or(0, ChainLayout::writable_bytes)
    }

    fn check_epoch(&self) -> Result<(), ChainError<ByteAccessError>> {
        let current = QueueEpoch::new(self.sink.current_epoch.load(Ordering::Acquire))
            .expect("queue epochs are always nonzero");
        if current == self.id.epoch() {
            Ok(())
        } else {
            Err(ChainError::ResetRace {
                chain: self.id.epoch(),
                current,
            })
        }
    }
}

impl DeviceChain for SplitDeviceChain {
    type Request = SplitSource;
    type Response = SplitSink;
    type Error = ByteAccessError;

    fn id(&self) -> ChainId {
        self.id
    }

    fn io(&mut self) -> ChainIoResult<'_, Self::Request, Self::Response, Self::Error> {
        self.check_epoch()?;
        let layout = self
            .data
            .analysis
            .validation()
            .map_err(ChainError::Malformed)?;
        if layout.descriptor_count() > self.max_descriptors {
            return Err(ChainError::Malformed(MalformedChain::DescriptorCount));
        }
        Ok(ChainIo::new(
            self.data.analysis.regions(),
            &self.source,
            &mut self.sink,
        ))
    }
}

fn analyze_chain(
    descriptors: &[Descriptor],
    head: u16,
    visited: &mut [u16],
) -> Result<ChainAnalysis, ChainBuildError> {
    let mut order = Vec::new();
    order
        .try_reserve_exact(descriptors.len())
        .map_err(|_| ChainBuildError::AllocationFailed)?;
    let mut regions = Vec::new();
    regions
        .try_reserve_exact(descriptors.len())
        .map_err(|_| ChainBuildError::AllocationFailed)?;

    let mut current = head;
    loop {
        let Some(descriptor) = descriptors.get(usize::from(current)) else {
            return Ok(ChainAnalysis::Invalid(MalformedChain::DescriptorIndex));
        };
        if visited[usize::from(current)] != 0 {
            return Ok(ChainAnalysis::Invalid(MalformedChain::DescriptorLoop));
        }
        visited[usize::from(current)] = 1;
        if descriptor.flags & !KNOWN_DESCRIPTOR_FLAGS != 0 {
            return Ok(ChainAnalysis::Invalid(MalformedChain::DescriptorFlags));
        }
        if descriptor.flags & VIRTQ_DESC_F_INDIRECT != 0 {
            return Ok(ChainAnalysis::Invalid(MalformedChain::IndirectUnsupported));
        }
        if !descriptor.buffer.is_mapped() {
            return Ok(ChainAnalysis::Invalid(MalformedChain::Address));
        }

        order.push(current);
        regions.push(if descriptor.is_writable() {
            ChainRegion::writable(descriptor.len())
        } else {
            ChainRegion::readable(descriptor.len())
        });

        if descriptor.flags & VIRTQ_DESC_F_NEXT == 0 {
            break;
        }
        current = descriptor.next;
    }

    if order.len() != descriptors.len() {
        return Ok(ChainAnalysis::Invalid(MalformedChain::DescriptorCount));
    }

    let layout = match validate_chain_layout(&regions, u16::MAX) {
        Ok(layout) => layout,
        Err(error) => return Ok(ChainAnalysis::Invalid(error.into())),
    };
    Ok(ChainAnalysis::Valid {
        layout,
        order: order.into_boxed_slice(),
        regions: regions.into_boxed_slice(),
    })
}

fn copy_from_descriptors(
    data: &ChainData,
    writable: bool,
    offset: u64,
    target: &mut [u8],
) -> Result<(), ByteAccessError> {
    if target.is_empty() {
        return Ok(());
    }
    let mut skip = offset;
    let mut copied = 0;
    for index in data.analysis.order() {
        let descriptor = &data.descriptors[usize::from(*index)];
        if descriptor.is_writable() != writable {
            continue;
        }
        if skip >= descriptor.len() {
            skip -= descriptor.len();
            continue;
        }
        let available = usize::try_from(descriptor.len() - skip).unwrap_or(usize::MAX);
        let count = min(available, target.len() - copied);
        descriptor
            .buffer
            .read_at(skip, &mut target[copied..copied + count])?;
        copied += count;
        skip = 0;
        if copied == target.len() {
            return Ok(());
        }
    }
    Err(ByteAccessError::OutOfBounds)
}

fn copy_to_descriptors(
    data: &ChainData,
    offset: u64,
    source: &[u8],
) -> Result<(), ByteAccessError> {
    if source.is_empty() {
        return Ok(());
    }
    let mut skip = offset;
    let mut copied = 0;
    for index in data.analysis.order() {
        let descriptor = &data.descriptors[usize::from(*index)];
        if !descriptor.is_writable() {
            continue;
        }
        if skip >= descriptor.len() {
            skip -= descriptor.len();
            continue;
        }
        let available = usize::try_from(descriptor.len() - skip).unwrap_or(usize::MAX);
        let count = min(available, source.len() - copied);
        descriptor
            .buffer
            .write_at(skip, &source[copied..copied + count])?;
        copied += count;
        skip = 0;
        if copied == source.len() {
            return Ok(());
        }
    }
    Err(ByteAccessError::OutOfBounds)
}

fn zeroed_u16_box(len: usize) -> Result<Box<[u16]>, ChainBuildError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ChainBuildError::AllocationFailed)?;
    values.resize(len, 0);
    Ok(values.into_boxed_slice())
}

fn checked_logical_range(offset: u64, bytes: usize, len: u64) -> Result<(), ByteAccessError> {
    let bytes = u64::try_from(bytes).map_err(|_| ByteAccessError::OutOfBounds)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(ByteAccessError::OutOfBounds)?;
    if end > len {
        return Err(ByteAccessError::OutOfBounds);
    }
    Ok(())
}

fn checked_range(
    offset: u64,
    bytes: usize,
    len: usize,
) -> Result<core::ops::Range<usize>, ByteAccessError> {
    let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
    let end = start
        .checked_add(bytes)
        .filter(|end| *end <= len)
        .ok_or(ByteAccessError::OutOfBounds)?;
    Ok(start..end)
}

//! In-memory reference implementation of the portable accelerator lifecycle.
//!
//! This backend is intentionally synchronous internally, but submissions remain pending until
//! [`MockAccelerator::complete`] is called so transports can test in-flight ownership.

#![forbid(unsafe_code)]

pub mod fault;
pub mod reference;

use core::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AllocatedBuffer, ArtifactFormat, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, ByteSink, ByteSource, Capabilities,
    ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain, QueueDesc,
    ReleaseFailure, SubmitFailure, TargetIdentity, Timeout, validate_bindings,
};

use reference::Operation;

const EVENT_PENDING: u8 = 0;
const EVENT_EXECUTING: u8 = 1;
const EVENT_COMPLETE: u8 = 2;
const EVENT_CANCELLED: u8 = 3;
const EVENT_DEVICE_LOST: u8 = 4;
const TRANSFER_CHUNK_BYTES: usize = 256;

#[derive(Clone, Debug)]
pub struct MockContext {
    id: u64,
}

#[derive(Debug)]
pub struct MockBuffer {
    context_id: u64,
    desc: BufferDesc,
    data: Arc<[AtomicU8]>,
}

#[derive(Clone, Debug)]
pub struct MockProgram {
    context_id: u64,
    format: ArtifactFormat,
    target: TargetIdentity,
    payload_bytes: usize,
    operation: Operation,
}

impl MockProgram {
    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }

    pub const fn target(&self) -> TargetIdentity {
        self.target
    }

    pub const fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
}

#[derive(Clone, Debug)]
pub struct MockQueue {
    context_id: u64,
}

#[derive(Clone, Debug)]
pub struct MockEvent {
    inner: Arc<MockEventInner>,
}

#[derive(Debug)]
struct MockEventInner {
    state: AtomicU8,
    invocation: MockInvocation,
}

#[derive(Clone, Debug)]
struct BufferSlice {
    data: Arc<[AtomicU8]>,
    range: Range<usize>,
}

#[derive(Debug)]
enum MockInvocation {
    Barrier,
    Copy {
        source: BufferSlice,
        target: BufferSlice,
    },
    Fill {
        target: BufferSlice,
        value: u8,
    },
    Xor {
        target: BufferSlice,
        value: u8,
    },
}

impl MockInvocation {
    fn execute(&self) {
        match self {
            Self::Barrier => {}
            Self::Copy { source, target } => {
                let reverse = Arc::ptr_eq(&source.data, &target.data)
                    && target.range.start > source.range.start
                    && target.range.start < source.range.end;
                if reverse {
                    for index in (0..source.range.len()).rev() {
                        let byte = source.data[source.range.start + index].load(Ordering::Relaxed);
                        target.data[target.range.start + index].store(byte, Ordering::Relaxed);
                    }
                } else {
                    for index in 0..source.range.len() {
                        let byte = source.data[source.range.start + index].load(Ordering::Relaxed);
                        target.data[target.range.start + index].store(byte, Ordering::Relaxed);
                    }
                }
            }
            Self::Fill { target, value } => {
                for byte in &target.data[target.range.clone()] {
                    byte.store(*value, Ordering::Relaxed);
                }
            }
            Self::Xor { target, value } => {
                for byte in &target.data[target.range.clone()] {
                    byte.fetch_xor(*value, Ordering::Relaxed);
                }
            }
        }
    }
}

pub struct MockAccelerator {
    next_id: AtomicU64,
    direct_binding_admissions: AtomicU64,
    info: DeviceInfo,
}

impl Default for MockAccelerator {
    fn default() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            direct_binding_admissions: AtomicU64::new(0),
            info: DeviceInfo {
                identity: DeviceIdentity {
                    uuid: *b"virtio-accelmock",
                    class: AcceleratorClass::NPU,
                    vendor_id: 0,
                    device_id: 0,
                },
                capabilities: Capabilities::HOST_VISIBLE_MEMORY
                    | Capabilities::DEVICE_LOCAL_MEMORY
                    | Capabilities::SHARED_MEMORY
                    | Capabilities::EVENT_CANCELLATION,
                limits: DeviceLimits {
                    max_contexts: 64,
                    max_buffers_per_context: 1_024,
                    max_programs_per_context: 256,
                    max_queues_per_context: 16,
                    max_events_per_context: 4_096,
                    max_bindings_per_submission: 256,
                    max_buffer_bytes: 1 << 30,
                    max_artifact_bytes: 1 << 30,
                },
            },
        }
    }
}

impl MockAccelerator {
    /// Cumulative provider-owned bindings admitted without hidden submission staging.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.load(Ordering::Relaxed)
    }

    pub fn complete(&self, event: &MockEvent) -> Result<(), BackendError> {
        event
            .inner
            .state
            .compare_exchange(
                EVENT_PENDING,
                EVENT_EXECUTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| BackendError::Busy)?;
        event.inner.invocation.execute();
        event.inner.state.store(EVENT_COMPLETE, Ordering::Release);
        Ok(())
    }

    /// Fail a pending event before execution, simulating harness-controlled device loss.
    pub fn fail_device_lost(&self, event: &MockEvent) -> Result<(), BackendError> {
        event
            .inner
            .state
            .compare_exchange(
                EVENT_PENDING,
                EVENT_DEVICE_LOST,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| BackendError::Busy)
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| BackendError::ResourceLimit)
    }

    fn checked_range(
        total: usize,
        offset: u64,
        bytes: usize,
    ) -> Result<core::ops::Range<usize>, BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(bytes)
            .filter(|end| *end <= total)
            .ok_or(BackendError::OutOfBounds)?;
        Ok(start..end)
    }

    fn binding_for_slot<'slice, 'buffer>(
        bindings: &'slice [BindingRef<'buffer, MockBuffer>],
        slot: u32,
    ) -> Option<&'slice BindingRef<'buffer, MockBuffer>> {
        bindings.iter().find(|binding| binding.slot == slot)
    }

    fn buffer_slice(binding: &BindingRef<'_, MockBuffer>) -> Result<BufferSlice, BackendError> {
        let bytes =
            usize::try_from(binding.range.bytes()).map_err(|_| BackendError::OutOfBounds)?;
        Ok(BufferSlice {
            data: Arc::clone(&binding.buffer.data),
            range: Self::checked_range(binding.buffer.data.len(), binding.range.offset, bytes)?,
        })
    }

    fn prepare_invocation(
        operation: Operation,
        bindings: &[BindingRef<'_, MockBuffer>],
    ) -> Result<MockInvocation, BackendError> {
        let incompatible = || BackendError::Incompatible;
        match operation {
            Operation::Barrier { slot } => {
                if bindings.len() != 1 || Self::binding_for_slot(bindings, slot).is_none() {
                    return Err(incompatible());
                }
                Ok(MockInvocation::Barrier)
            }
            Operation::Copy {
                source_slot,
                target_slot,
            } => {
                if bindings.len() != 2 {
                    return Err(incompatible());
                }
                let source = Self::binding_for_slot(bindings, source_slot)
                    .filter(|binding| binding.access == virtio_accel_core::AccessMode::Read)
                    .ok_or_else(incompatible)?;
                let target = Self::binding_for_slot(bindings, target_slot)
                    .filter(|binding| binding.access == virtio_accel_core::AccessMode::Write)
                    .ok_or_else(incompatible)?;
                if source.range.bytes() != target.range.bytes() {
                    return Err(incompatible());
                }
                Ok(MockInvocation::Copy {
                    source: Self::buffer_slice(source)?,
                    target: Self::buffer_slice(target)?,
                })
            }
            Operation::Fill { target_slot, value } => {
                if bindings.len() != 1 {
                    return Err(incompatible());
                }
                let target = Self::binding_for_slot(bindings, target_slot)
                    .filter(|binding| binding.access == virtio_accel_core::AccessMode::Write)
                    .ok_or_else(incompatible)?;
                Ok(MockInvocation::Fill {
                    target: Self::buffer_slice(target)?,
                    value,
                })
            }
            Operation::Xor { target_slot, value } => {
                if bindings.len() != 1 {
                    return Err(incompatible());
                }
                let target = Self::binding_for_slot(bindings, target_slot)
                    .filter(|binding| binding.access == virtio_accel_core::AccessMode::ReadWrite)
                    .ok_or_else(incompatible)?;
                Ok(MockInvocation::Xor {
                    target: Self::buffer_slice(target)?,
                    value,
                })
            }
        }
    }
}

impl Accelerator for MockAccelerator {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        Ok(MockContext {
            id: self.next_id()?,
        })
    }

    fn destroy_context(
        &self,
        _context: Self::Context,
    ) -> Result<(), ReleaseFailure<Self::Context>> {
        Ok(())
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.info.validate_buffer_desc(desc)?;
        let properties = match desc.domain {
            MemoryDomain::Host => BufferProperties::HOST_VISIBLE,
            MemoryDomain::Device => BufferProperties::DEVICE_LOCAL,
            MemoryDomain::Shared => BufferProperties::HOST_VISIBLE,
        } | if desc.is_program_visible() || desc.domain == MemoryDomain::Shared {
            BufferProperties::DIRECT_BINDING
        } else {
            BufferProperties::empty()
        };
        let info = BufferInfo::new(desc, desc.bytes(), desc.alignment(), properties)?;
        let bytes = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfMemory)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .map_err(|_| BackendError::OutOfMemory)?;
        data.resize_with(bytes, || AtomicU8::new(0));
        Ok(AllocatedBuffer::new(
            MockBuffer {
                context_id: context.id,
                desc,
                data: Arc::from(data.into_boxed_slice()),
            },
            info,
        ))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(virtio_accel_core::BufferUsage::TRANSFER_DESTINATION)
        {
            return Err(BackendError::PermissionDenied);
        }
        let bytes = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let range = Self::checked_range(buffer.data.len(), offset, bytes)?;
        if let Some(source) = data.as_contiguous() {
            if source.len() != bytes {
                return Err(BackendError::InvalidArgument);
            }
            for (target, source) in buffer.data[range].iter().zip(source) {
                target.store(*source, Ordering::Relaxed);
            }
        } else {
            let mut scratch = [0; TRANSFER_CHUNK_BYTES];
            let mut copied = 0;
            while copied < bytes {
                let chunk_bytes = (bytes - copied).min(TRANSFER_CHUNK_BYTES);
                data.read_at(copied as u64, &mut scratch[..chunk_bytes])?;
                for (target, source) in buffer.data[range.start + copied..][..chunk_bytes]
                    .iter()
                    .zip(&scratch[..chunk_bytes])
                {
                    target.store(*source, Ordering::Relaxed);
                }
                copied += chunk_bytes;
            }
        }
        Ok(())
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(virtio_accel_core::BufferUsage::TRANSFER_SOURCE)
        {
            return Err(BackendError::PermissionDenied);
        }
        let bytes = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let range = Self::checked_range(buffer.data.len(), offset, bytes)?;
        if let Some(target) = data.as_contiguous_mut() {
            if target.len() != bytes {
                return Err(BackendError::InvalidArgument);
            }
            for (target, source) in target.iter_mut().zip(&buffer.data[range]) {
                *target = source.load(Ordering::Relaxed);
            }
        } else {
            let mut scratch = [0; TRANSFER_CHUNK_BYTES];
            let mut copied = 0;
            while copied < bytes {
                let chunk_bytes = (bytes - copied).min(TRANSFER_CHUNK_BYTES);
                for (target, source) in scratch[..chunk_bytes]
                    .iter_mut()
                    .zip(&buffer.data[range.start + copied..][..chunk_bytes])
                {
                    *target = source.load(Ordering::Relaxed);
                }
                data.write_at(copied as u64, &scratch[..chunk_bytes])?;
                copied += chunk_bytes;
            }
        }
        Ok(())
    }

    fn free_buffer(&self, _buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        Ok(())
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if artifact.payload.len() > self.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.format != reference::ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        if artifact.target != reference::TARGET_IDENTITY
            || artifact.resident_bytes != reference::RESIDENT_BYTES
        {
            return Err(BackendError::Incompatible);
        }
        let operation = reference::decode(artifact.payload)?;
        let payload_bytes =
            usize::try_from(artifact.payload.len()).map_err(|_| BackendError::ResourceLimit)?;
        Ok(MockProgram {
            context_id: context.id,
            format: artifact.format,
            target: artifact.target,
            payload_bytes,
            operation,
        })
    }

    fn unload_program(&self, _program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        Ok(())
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.info.validate_queue_desc(desc)?;
        Ok(MockQueue {
            context_id: context.id,
        })
    }

    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        Ok(())
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        _timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        validate_bindings(bindings, self.info.limits.max_bindings_per_submission)
            .map_err(SubmitFailure::Rejected)?;
        if queue.context_id != program.context_id
            || bindings
                .iter()
                .any(|binding| binding.buffer.context_id != queue.context_id)
        {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }
        for binding in bindings {
            if binding.range.end() > binding.buffer.desc.bytes() {
                return Err(SubmitFailure::Rejected(BackendError::OutOfBounds));
            }
            if !binding.buffer.desc.allows_access(binding.access) {
                return Err(SubmitFailure::Rejected(BackendError::PermissionDenied));
            }
        }
        let invocation = Self::prepare_invocation(program.operation, bindings)
            .map_err(SubmitFailure::Rejected)?;
        self.direct_binding_admissions
            .fetch_add(bindings.len() as u64, Ordering::Relaxed);
        Ok(MockEvent {
            inner: Arc::new(MockEventInner {
                state: AtomicU8::new(EVENT_PENDING),
                invocation,
            }),
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match event.inner.state.load(Ordering::Acquire) {
            EVENT_PENDING | EVENT_EXECUTING => Ok(EventState::Pending),
            EVENT_COMPLETE => Ok(EventState::Complete),
            EVENT_CANCELLED => Ok(EventState::Cancelled),
            EVENT_DEVICE_LOST => Ok(EventState::Failed(BackendError::DeviceLost)),
            _ => Err(BackendError::DeviceLost),
        }
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        event
            .inner
            .state
            .compare_exchange(
                EVENT_PENDING,
                EVENT_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| BackendError::Busy)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        let state = self
            .poll_event(&event)
            .map_err(|error| ReleaseFailure::Indeterminate { error })?;
        match state {
            EventState::Pending => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            }),
            EventState::Complete | EventState::Failed(_) | EventState::Cancelled => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_core::{
        AccessMode, BindingRef, BufferRange, BufferUsage, ContextFlags, QueueFlags,
    };

    fn load_reference(
        backend: &MockAccelerator,
        context: &MockContext,
        artifact: &reference::ReferenceArtifact,
    ) -> MockProgram {
        backend
            .load_program(
                context,
                ArtifactRef {
                    format: reference::ARTIFACT_FORMAT,
                    target: reference::TARGET_IDENTITY,
                    payload: artifact.as_bytes(),
                    resident_bytes: reference::RESIDENT_BYTES,
                },
            )
            .unwrap()
    }

    #[derive(Debug)]
    struct SplitSource<'a> {
        first: &'a [u8],
        second: &'a [u8],
    }

    impl ByteSource for SplitSource<'_> {
        fn len(&self) -> u64 {
            (self.first.len() + self.second.len()) as u64
        }

        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
            let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
            let end = start
                .checked_add(target.len())
                .filter(|end| *end <= self.first.len() + self.second.len())
                .ok_or(BackendError::OutOfBounds)?;
            for (segment_start, segment) in [(0, self.first), (self.first.len(), self.second)] {
                let overlap_start = start.max(segment_start);
                let overlap_end = end.min(segment_start + segment.len());
                if overlap_start < overlap_end {
                    target[overlap_start - start..overlap_end - start].copy_from_slice(
                        &segment[overlap_start - segment_start..overlap_end - segment_start],
                    );
                }
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct SplitSink<'a> {
        first: &'a mut [u8],
        second: &'a mut [u8],
    }

    impl ByteSink for SplitSink<'_> {
        fn len(&self) -> u64 {
            (self.first.len() + self.second.len()) as u64
        }

        fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
            let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
            let end = start
                .checked_add(source.len())
                .filter(|end| *end <= self.first.len() + self.second.len())
                .ok_or(BackendError::OutOfBounds)?;
            let first_len = self.first.len();
            for (segment_start, segment) in [(0, &mut *self.first), (first_len, &mut *self.second)]
            {
                let overlap_start = start.max(segment_start);
                let overlap_end = end.min(segment_start + segment.len());
                if overlap_start < overlap_end {
                    segment[overlap_start - segment_start..overlap_end - segment_start]
                        .copy_from_slice(&source[overlap_start - start..overlap_end - start]);
                }
            }
            Ok(())
        }
    }

    #[test]
    fn reference_backend_exercises_the_complete_lifecycle() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let desc = BufferDesc::new(
            16,
            8,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap();
        let allocation = backend.allocate_buffer(&context, desc).unwrap();
        assert_eq!(allocation.info().desc(), desc);
        assert_eq!(allocation.info().allocation_bytes(), 16);
        assert_eq!(allocation.info().alignment(), 8);
        assert!(
            allocation
                .info()
                .properties()
                .contains(BufferProperties::DIRECT_BINDING)
        );
        let (mut buffer, _) = allocation.into_parts();
        backend.write_buffer(&mut buffer, 4, &[1, 2, 3, 4]).unwrap();
        let mut output = [0; 4];
        backend.read_buffer(&buffer, 4, &mut output).unwrap();
        assert_eq!(output, [1, 2, 3, 4]);

        let artifact = reference::ReferenceArtifact::barrier(0);
        let program = load_reference(&backend, &context, &artifact);
        assert_eq!(program.format(), reference::ARTIFACT_FORMAT);
        assert_eq!(program.target(), reference::TARGET_IDENTITY);
        assert_eq!(program.payload_bytes(), reference::ARTIFACT_BYTES);
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range: BufferRange::new(0, 16).unwrap(),
            access: AccessMode::Read,
        }];
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        assert_eq!(backend.poll_event(&event), Ok(EventState::Pending));
        assert!(matches!(
            backend.destroy_event(event.clone()),
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                ..
            })
        ));
        backend.complete(&event).unwrap();
        assert_eq!(backend.poll_event(&event), Ok(EventState::Complete));

        backend.destroy_event(event).unwrap();
        backend.destroy_queue(queue).unwrap();
        backend.unload_program(program).unwrap();
        backend.free_buffer(buffer).unwrap();
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn explicit_transfers_enforce_declared_direction() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let allocation = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(4, 1, MemoryDomain::Host, BufferUsage::TRANSFER_SOURCE).unwrap(),
            )
            .unwrap();
        let (mut buffer, _) = allocation.into_parts();
        assert_eq!(
            backend.write_buffer(&mut buffer, 0, &[1]),
            Err(BackendError::PermissionDenied)
        );

        backend.free_buffer(buffer).unwrap();
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn reserved_creation_flags_are_rejected_without_resources() {
        let backend = MockAccelerator::default();
        assert!(matches!(
            backend.create_context(ContextDesc {
                flags: ContextFlags::SECURE,
            }),
            Err(BackendError::Unsupported)
        ));

        let context = backend.create_context(ContextDesc::default()).unwrap();
        assert!(matches!(
            backend.create_queue(
                &context,
                QueueDesc {
                    flags: QueueFlags::IN_ORDER,
                },
            ),
            Err(BackendError::Unsupported)
        ));
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn segmented_transfers_do_not_require_coalescing() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let allocation = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    4,
                    1,
                    MemoryDomain::Host,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION,
                )
                .unwrap(),
            )
            .unwrap();
        let (mut buffer, _) = allocation.into_parts();

        let source = SplitSource {
            first: &[1, 2],
            second: &[3, 4],
        };
        backend.write_buffer(&mut buffer, 0, &source).unwrap();

        let mut first = [0; 1];
        let mut second = [0; 3];
        let mut sink = SplitSink {
            first: &mut first,
            second: &mut second,
        };
        backend.read_buffer(&buffer, 0, &mut sink).unwrap();
        assert_eq!(first, [1]);
        assert_eq!(second, [2, 3, 4]);

        backend.free_buffer(buffer).unwrap();
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn cross_context_submission_is_rejected_before_acceptance() {
        let backend = MockAccelerator::default();
        let context_a = backend.create_context(ContextDesc::default()).unwrap();
        let context_b = backend.create_context(ContextDesc::default()).unwrap();
        let allocation = backend
            .allocate_buffer(
                &context_b,
                BufferDesc::new(1, 1, MemoryDomain::Host, BufferUsage::PROGRAM_INPUT).unwrap(),
            )
            .unwrap();
        let (buffer, _) = allocation.into_parts();
        let artifact = reference::ReferenceArtifact::barrier(0);
        let program = load_reference(&backend, &context_a, &artifact);
        let queue = backend
            .create_queue(&context_a, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range: BufferRange::new(0, 1).unwrap(),
            access: AccessMode::Read,
        }];
        assert!(matches!(
            backend.submit(&queue, &program, &bindings, Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
        ));
    }

    #[test]
    fn copy_produces_verifiable_output_only_after_completion() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let source = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    8,
                    1,
                    MemoryDomain::Host,
                    BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
                )
                .unwrap(),
            )
            .unwrap();
        let target = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    8,
                    1,
                    MemoryDomain::Host,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
                )
                .unwrap(),
            )
            .unwrap();
        let (mut source, _) = source.into_parts();
        let (target, _) = target.into_parts();
        backend.write_buffer(&mut source, 0, b"copy me").unwrap();

        let artifact = reference::ReferenceArtifact::copy(3, 7).unwrap();
        let program = load_reference(&backend, &context, &artifact);
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [
            BindingRef {
                slot: 7,
                buffer: &target,
                range: BufferRange::new(0, 7).unwrap(),
                access: AccessMode::Write,
            },
            BindingRef {
                slot: 3,
                buffer: &source,
                range: BufferRange::new(0, 7).unwrap(),
                access: AccessMode::Read,
            },
        ];
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();

        let mut output = [0; 7];
        backend.read_buffer(&target, 0, &mut output).unwrap();
        assert_eq!(output, [0; 7]);
        backend.complete(&event).unwrap();
        backend.read_buffer(&target, 0, &mut output).unwrap();
        assert_eq!(&output, b"copy me");
    }

    #[test]
    fn pending_operations_complete_in_harness_selected_order() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let allocation = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    4,
                    1,
                    MemoryDomain::Shared,
                    BufferUsage::TRANSFER_SOURCE
                        | BufferUsage::PROGRAM_OUTPUT
                        | BufferUsage::MUTABLE_STATE,
                )
                .unwrap(),
            )
            .unwrap();
        let (buffer, _) = allocation.into_parts();
        let fill = reference::ReferenceArtifact::fill(0, 0xa5);
        let xor = reference::ReferenceArtifact::xor(0, 0xff);
        let fill_program = load_reference(&backend, &context, &fill);
        let xor_program = load_reference(&backend, &context, &xor);
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let range = BufferRange::new(0, 4).unwrap();
        let fill_bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range,
            access: AccessMode::Write,
        }];
        let xor_bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range,
            access: AccessMode::ReadWrite,
        }];
        let fill_event = backend
            .submit(&queue, &fill_program, &fill_bindings, Timeout::Infinite)
            .unwrap();
        let xor_event = backend
            .submit(&queue, &xor_program, &xor_bindings, Timeout::Infinite)
            .unwrap();

        backend.complete(&xor_event).unwrap();
        let mut output = [0; 4];
        backend.read_buffer(&buffer, 0, &mut output).unwrap();
        assert_eq!(output, [0xff; 4]);
        assert_eq!(backend.poll_event(&fill_event), Ok(EventState::Pending));

        backend.complete(&fill_event).unwrap();
        backend.read_buffer(&buffer, 0, &mut output).unwrap();
        assert_eq!(output, [0xa5; 4]);
    }

    #[test]
    fn cancellation_and_device_loss_prevent_execution() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let allocation = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    4,
                    1,
                    MemoryDomain::Host,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
                )
                .unwrap(),
            )
            .unwrap();
        let (buffer, _) = allocation.into_parts();
        let artifact = reference::ReferenceArtifact::fill(0, 0x5a);
        let program = load_reference(&backend, &context, &artifact);
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range: BufferRange::new(0, 4).unwrap(),
            access: AccessMode::Write,
        }];

        let cancelled = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        backend.cancel_event(&cancelled).unwrap();
        assert_eq!(backend.complete(&cancelled), Err(BackendError::Busy));

        let lost = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        backend.fail_device_lost(&lost).unwrap();
        assert_eq!(
            backend.poll_event(&lost),
            Ok(EventState::Failed(BackendError::DeviceLost))
        );
        assert_eq!(backend.complete(&lost), Err(BackendError::Busy));

        let mut output = [0; 4];
        backend.read_buffer(&buffer, 0, &mut output).unwrap();
        assert_eq!(output, [0; 4]);
    }

    #[test]
    fn artifact_and_binding_incompatibility_fail_before_admission() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let artifact = reference::ReferenceArtifact::fill(2, 1);

        assert!(matches!(
            backend.load_program(
                &context,
                ArtifactRef {
                    format: reference::ARTIFACT_FORMAT,
                    target: TargetIdentity([0; 12]),
                    payload: artifact.as_bytes(),
                    resident_bytes: reference::RESIDENT_BYTES,
                },
            ),
            Err(BackendError::Incompatible)
        ));

        let mut malformed = *artifact.as_bytes();
        malformed[17] = 1;
        assert!(matches!(
            backend.load_program(
                &context,
                ArtifactRef {
                    format: reference::ARTIFACT_FORMAT,
                    target: reference::TARGET_IDENTITY,
                    payload: &malformed,
                    resident_bytes: reference::RESIDENT_BYTES,
                },
            ),
            Err(BackendError::InvalidArgument)
        ));
        let program = load_reference(&backend, &context, &artifact);

        let allocation = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(4, 1, MemoryDomain::Host, BufferUsage::MUTABLE_STATE).unwrap(),
            )
            .unwrap();
        let (buffer, _) = allocation.into_parts();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 2,
            buffer: &buffer,
            range: BufferRange::new(0, 4).unwrap(),
            access: AccessMode::ReadWrite,
        }];
        assert!(matches!(
            backend.submit(&queue, &program, &bindings, Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::Incompatible))
        ));
    }
}

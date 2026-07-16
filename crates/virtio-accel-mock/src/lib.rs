//! In-memory reference implementation of the portable accelerator lifecycle.
//!
//! This backend is intentionally synchronous internally, but submissions remain pending until
//! [`MockAccelerator::complete`] is called so transports can test in-flight ownership.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AllocatedBuffer, ArtifactFormat, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, ByteSink, ByteSource, Capabilities,
    ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain, QueueDesc,
    ReleaseFailure, SubmitFailure, TargetIdentity, Timeout, validate_bindings,
};

const EVENT_PENDING: u8 = 0;
const EVENT_COMPLETE: u8 = 1;
const EVENT_CANCELLED: u8 = 2;

#[derive(Clone, Debug)]
pub struct MockContext {
    id: u64,
}

#[derive(Debug)]
pub struct MockBuffer {
    context_id: u64,
    desc: BufferDesc,
    data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct MockProgram {
    context_id: u64,
    format: ArtifactFormat,
    target: TargetIdentity,
    payload_bytes: usize,
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
    state: Arc<AtomicU8>,
}

pub struct MockAccelerator {
    next_id: AtomicU64,
    info: DeviceInfo,
}

impl Default for MockAccelerator {
    fn default() -> Self {
        Self {
            next_id: AtomicU64::new(1),
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
    pub fn complete(&self, event: &MockEvent) -> Result<(), BackendError> {
        event
            .state
            .compare_exchange(
                EVENT_PENDING,
                EVENT_COMPLETE,
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
        data.resize(bytes, 0);
        Ok(AllocatedBuffer::new(
            MockBuffer {
                context_id: context.id,
                desc,
                data,
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
            buffer.data[range].copy_from_slice(source);
        } else {
            data.read_at(0, &mut buffer.data[range])?;
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
            target.copy_from_slice(&buffer.data[range]);
        } else {
            data.write_at(0, &buffer.data[range])?;
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
        let payload_bytes =
            usize::try_from(artifact.payload.len()).map_err(|_| BackendError::ResourceLimit)?;
        Ok(MockProgram {
            context_id: context.id,
            format: artifact.format,
            target: artifact.target,
            payload_bytes,
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
        Ok(MockEvent {
            state: Arc::new(AtomicU8::new(EVENT_PENDING)),
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match event.state.load(Ordering::Acquire) {
            EVENT_PENDING => Ok(EventState::Pending),
            EVENT_COMPLETE => Ok(EventState::Complete),
            EVENT_CANCELLED => Ok(EventState::Cancelled),
            _ => Err(BackendError::DeviceLost),
        }
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        event
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

        let program = backend
            .load_program(
                &context,
                ArtifactRef {
                    format: ArtifactFormat::new(1).unwrap(),
                    target: TargetIdentity([0; 12]),
                    payload: &[0xaa],
                    resident_bytes: 16,
                },
            )
            .unwrap();
        assert_eq!(program.format().get(), 1);
        assert_eq!(program.target(), TargetIdentity([0; 12]));
        assert_eq!(program.payload_bytes(), 1);
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
        let program = backend
            .load_program(
                &context_a,
                ArtifactRef {
                    format: ArtifactFormat::new(1).unwrap(),
                    target: TargetIdentity([0; 12]),
                    payload: &[1],
                    resident_bytes: 1,
                },
            )
            .unwrap();
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
}

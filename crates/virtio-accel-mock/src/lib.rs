//! In-memory reference implementation of the portable accelerator lifecycle.
//!
//! This backend is intentionally synchronous internally, but submissions remain pending until
//! [`MockAccelerator::complete`] is called so transports can test in-flight ownership.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, ArtifactFormat, ArtifactRef, BackendError, BindingRef,
    BufferDesc, Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState,
    QueueDesc, ReleaseFailure, SubmitFailure, TargetIdentity, Timeout, validate_bindings,
};

const EVENT_PENDING: u8 = 0;
const EVENT_COMPLETE: u8 = 1;
const EVENT_CANCELLED: u8 = 2;

#[derive(Clone, Debug)]
pub struct MockContext {
    id: u64,
}

#[derive(Clone, Debug)]
pub struct MockBuffer {
    context_id: u64,
    desc: BufferDesc,
    data: Arc<Mutex<Vec<u8>>>,
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

    fn create_context(&self, _desc: ContextDesc) -> Result<Self::Context, BackendError> {
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
    ) -> Result<Self::Buffer, BackendError> {
        if desc.bytes() > self.info.limits.max_buffer_bytes {
            return Err(BackendError::ResourceLimit);
        }
        let bytes = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfMemory)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .map_err(|_| BackendError::OutOfMemory)?;
        data.resize(bytes, 0);
        Ok(MockBuffer {
            context_id: context.id,
            desc,
            data: Arc::new(Mutex::new(data)),
        })
    }

    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), BackendError> {
        let mut target = buffer.data.lock().map_err(|_| BackendError::DeviceLost)?;
        let range = Self::checked_range(target.len(), offset, data.len())?;
        target[range].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), BackendError> {
        let source = buffer.data.lock().map_err(|_| BackendError::DeviceLost)?;
        let range = Self::checked_range(source.len(), offset, data.len())?;
        data.copy_from_slice(&source[range]);
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
        if artifact.payload.len() as u64 > self.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        Ok(MockProgram {
            context_id: context.id,
            format: artifact.format,
            target: artifact.target,
            payload_bytes: artifact.payload.len(),
        })
    }

    fn unload_program(&self, _program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        Ok(())
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        _desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
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
    use virtio_accel_core::{AccessMode, BindingRef, BufferRange, BufferUsage, MemoryDomain};

    #[test]
    fn reference_backend_exercises_the_complete_lifecycle() {
        let backend = MockAccelerator::default();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let buffer = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    16,
                    8,
                    MemoryDomain::Shared,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_INPUT,
                )
                .unwrap(),
            )
            .unwrap();
        backend.write_buffer(&buffer, 4, &[1, 2, 3, 4]).unwrap();
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
    fn cross_context_submission_is_rejected_before_acceptance() {
        let backend = MockAccelerator::default();
        let context_a = backend.create_context(ContextDesc::default()).unwrap();
        let context_b = backend.create_context(ContextDesc::default()).unwrap();
        let buffer = backend
            .allocate_buffer(
                &context_b,
                BufferDesc::new(1, 1, MemoryDomain::Host, BufferUsage::PROGRAM_INPUT).unwrap(),
            )
            .unwrap();
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

//! vAccel integration boundary for portable `virtio-accel` contracts.
//!
//! This crate owns the adapter seam for host-side integrations that need to project an
//! external execution model into the [`virtio_accel_core::Accelerator`] contract.
//!
//! The adapter is intentionally conservative:
//! - It never adds host/VMM dependencies to portable crates.
//! - It forwards lifecycle, allocation, submission, and release calls to the wrapped backend.
//! - It exposes best-effort copy-path diagnostics (`direct_binding_admissions` and
//!   `explicit_transfer_bytes`) to keep integration evidence visible in conformance harnesses.

#![forbid(unsafe_code)]
#![no_std]

use core::sync::atomic::{AtomicU64, Ordering};
use core::{convert::TryFrom, result::Result};

use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, ByteSink,
    ByteSource, ContextDesc, DeviceInfo, EventState, QueueDesc, ReleaseFailure, SubmitFailure,
    Timeout,
};

#[derive(Debug, Default)]
struct SubmissionPathCounters {
    direct_bindings: AtomicU64,
    explicit_transfer_bytes: AtomicU64,
}

impl SubmissionPathCounters {
    fn direct_binding_admissions(&self) -> u64 {
        self.direct_bindings.load(Ordering::Relaxed)
    }

    fn explicit_transfer_bytes(&self) -> u64 {
        self.explicit_transfer_bytes.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.direct_bindings.store(0, Ordering::Relaxed);
        self.explicit_transfer_bytes.store(0, Ordering::Relaxed);
    }

    fn record_submission(&self, bindings: usize) {
        self.direct_bindings.fetch_add(
            u64::try_from(bindings).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn record_explicit_transfer(&self, bytes: u64) {
        self.explicit_transfer_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Adapter seam from a concrete backend to the portable accelerator contract.
#[derive(Debug)]
pub struct VAccelAdapter<A> {
    inner: A,
    counters: SubmissionPathCounters,
}

impl<A> VAccelAdapter<A> {
    /// Create a new adapter around a concrete backend.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            counters: SubmissionPathCounters::default(),
        }
    }

    /// Borrow the wrapped backend to reach provider-specific metrics.
    pub fn backend(&self) -> &A {
        &self.inner
    }

    /// Borrow the wrapped backend mutably.
    pub fn backend_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    /// Consume the adapter and return the underlying backend.
    pub fn into_inner(self) -> A {
        self.inner
    }

    /// Number of observed accepted binding admissions.
    ///
    /// This counter is a first-order signal for backends that do not expose their native
    /// direct-bindings counters yet.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.counters.direct_binding_admissions()
    }

    /// Total bytes passed through explicit host transfers.
    ///
    /// This includes bytes requested by explicit `write_buffer` and `read_buffer` calls.
    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.counters.explicit_transfer_bytes()
    }

    /// Reset adapter-level metrics collected during the current integration window.
    pub fn reset_submission_path_metrics(&self) {
        self.counters.reset();
    }
}

impl<A: Accelerator> Accelerator for VAccelAdapter<A> {
    type Context = A::Context;
    type Buffer = A::Buffer;
    type Program = A::Program;
    type Queue = A::Queue;
    type Event = A::Event;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        self.inner.device_info()
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.inner.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.inner.destroy_context(context)
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.inner.allocate_buffer(context, desc)
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        let result = self.inner.write_buffer(buffer, offset, data);
        if result.is_ok() {
            self.counters.record_explicit_transfer(data.len());
        }
        result
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        self.counters.record_explicit_transfer(data.len());
        self.inner.read_buffer(buffer, offset, data)
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.inner.free_buffer(buffer)
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        self.inner.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.inner.unload_program(program)
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.inner.create_queue(context, desc)
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.inner.destroy_queue(queue)
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let result = self.inner.submit(queue, program, bindings, timeout);
        match &result {
            Ok(_) => self.counters.record_submission(bindings.len()),
            Err(SubmitFailure::Indeterminate { .. }) => {
                self.counters.record_submission(bindings.len())
            }
            Err(SubmitFailure::Rejected(_)) => {}
        }
        result
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        self.inner.poll_event(event)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.inner.cancel_event(event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.inner.destroy_event(event)
    }
}

mod support;

use std::cell::Cell;
use support::{target, target_in};
use virtio_accel_conformance::{
    CaseStatus, ConformanceHooks, SkipReason, SubmissionPathDiagnostics, run,
};
use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferInfo,
    BufferUsage, ByteSink, ByteSource, Capabilities, ContextDesc, DeviceInfo, EventState,
    MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defect {
    UnstableMetadata,
    DishonestAllocation,
    TransferValidation,
    ArtifactBounds,
    BindingValidation,
    ContextIsolation,
    AdmissionBoundary,
    TerminalRegression,
    PendingRelease,
    TimeoutClassification,
    MissingCancellation,
    CapabilitySubset,
    HiddenSubmissionStaging,
}

struct BrokenBackend {
    inner: MockAccelerator,
    defect: Defect,
    info_calls: Cell<u32>,
    terminal_polls: Cell<u32>,
    direct_bindings: Cell<u64>,
    staged_direct_bindings: Cell<u64>,
    staged_direct_bytes: Cell<u64>,
}

impl BrokenBackend {
    fn new(defect: Defect) -> Self {
        Self {
            inner: MockAccelerator::default(),
            defect,
            info_calls: Cell::new(0),
            terminal_polls: Cell::new(0),
            direct_bindings: Cell::new(0),
            staged_direct_bindings: Cell::new(0),
            staged_direct_bytes: Cell::new(0),
        }
    }
}

impl Accelerator for BrokenBackend {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        let mut info = self.inner.device_info()?;
        let call = self.info_calls.get();
        self.info_calls.set(call + 1);
        if self.defect == Defect::UnstableMetadata && call > 0 {
            info.identity.vendor_id ^= 1;
        }
        if self.defect == Defect::CapabilitySubset {
            info.capabilities = Capabilities::HOST_VISIBLE_MEMORY;
        }
        Ok(info)
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
        let allocation = self.inner.allocate_buffer(context, desc)?;
        if self.defect != Defect::DishonestAllocation {
            return Ok(allocation);
        }
        let actual = allocation.info();
        let (buffer, _) = allocation.into_parts();
        let dishonest_usage = if desc.usage == BufferUsage::TRANSFER_SOURCE {
            BufferUsage::TRANSFER_DESTINATION
        } else {
            BufferUsage::TRANSFER_SOURCE
        };
        let dishonest_desc =
            BufferDesc::new(desc.bytes(), desc.alignment(), desc.domain, dishonest_usage)?;
        let dishonest = BufferInfo::new(
            dishonest_desc,
            actual.allocation_bytes(),
            actual.alignment(),
            actual.properties(),
        )?;
        Ok(AllocatedBuffer::new(buffer, dishonest))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        match self.inner.write_buffer(buffer, offset, data) {
            Err(BackendError::OutOfBounds | BackendError::PermissionDenied)
                if self.defect == Defect::TransferValidation =>
            {
                Ok(())
            }
            result => result,
        }
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        match self.inner.read_buffer(buffer, offset, data) {
            Err(BackendError::OutOfBounds | BackendError::PermissionDenied)
                if self.defect == Defect::TransferValidation =>
            {
                Ok(())
            }
            result => result,
        }
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.inner.free_buffer(buffer)
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        match self.inner.load_program(context, artifact) {
            Err(BackendError::ResourceLimit) if self.defect == Defect::ArtifactBounds => {
                Err(BackendError::Unsupported)
            }
            result => result,
        }
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
        if self.defect == Defect::TimeoutClassification && matches!(timeout, Timeout::AfterNs(_)) {
            return Err(SubmitFailure::Rejected(BackendError::Busy));
        }
        let result = if self.defect == Defect::BindingValidation && bindings.len() > 1 {
            let first = &bindings[0];
            let valid = [BindingRef {
                slot: first.slot,
                buffer: first.buffer,
                range: first.range,
                access: first.access,
            }];
            self.inner.submit(queue, program, &valid, timeout)
        } else {
            self.inner.submit(queue, program, bindings, timeout)
        };
        match result {
            Ok(_event) if self.defect == Defect::AdmissionBoundary => {
                Err(SubmitFailure::Rejected(BackendError::Busy))
            }
            Ok(event) if self.defect == Defect::HiddenSubmissionStaging => {
                self.staged_direct_bindings
                    .set(self.staged_direct_bindings.get() + bindings.len() as u64);
                let staged_bytes = bindings
                    .iter()
                    .map(|binding| binding.range.bytes())
                    .sum::<u64>();
                self.staged_direct_bytes
                    .set(self.staged_direct_bytes.get() + staged_bytes);
                Ok(event)
            }
            Ok(event) => {
                self.direct_bindings
                    .set(self.direct_bindings.get() + bindings.len() as u64);
                Ok(event)
            }
            Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
                if self.defect == Defect::ContextIsolation =>
            {
                Err(SubmitFailure::Rejected(BackendError::Busy))
            }
            result => result,
        }
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        let state = self.inner.poll_event(event)?;
        if self.defect == Defect::TerminalRegression && state == EventState::Complete {
            let observation = self.terminal_polls.get();
            self.terminal_polls.set(observation + 1);
            if observation % 2 == 1 {
                return Ok(EventState::Pending);
            }
        }
        Ok(state)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        if self.defect == Defect::MissingCancellation {
            Err(BackendError::Unsupported)
        } else {
            self.inner.cancel_event(event)
        }
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        match self.inner.destroy_event(event) {
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                ..
            }) if self.defect == Defect::PendingRelease => Ok(()),
            result => result,
        }
    }
}

struct BrokenHooks;

impl ConformanceHooks<BrokenBackend> for BrokenHooks {
    fn complete_event(
        &self,
        backend: &BrokenBackend,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.inner.complete(event)
    }

    fn submission_path_diagnostics(
        &self,
        backend: &BrokenBackend,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_bindings.get(),
            staged_direct_bindings: backend.staged_direct_bindings.get(),
            staged_direct_bytes: backend.staged_direct_bytes.get(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

#[test]
fn major_backend_defects_are_detected_by_named_cases() {
    for (defect, case_id) in [
        (Defect::UnstableMetadata, "metadata.stable-valid"),
        (Defect::DishonestAllocation, "memory.host"),
        (
            Defect::TransferValidation,
            "buffer.segmented-transfer-bounds",
        ),
        (Defect::ArtifactBounds, "program.segmented-artifact-bounds"),
        (Defect::BindingValidation, "submission.binding-validation"),
        (Defect::ContextIsolation, "submission.context-isolation"),
        (Defect::AdmissionBoundary, "submission.binding-validation"),
        (
            Defect::TerminalRegression,
            "event.pending-release-terminal-stability",
        ),
        (
            Defect::PendingRelease,
            "event.pending-release-terminal-stability",
        ),
        (Defect::TimeoutClassification, "timeout.finite-admission"),
        (Defect::MissingCancellation, "event.cancellation-races"),
        (
            Defect::HiddenSubmissionStaging,
            "submission.copy-path-diagnostics",
        ),
    ] {
        let report = run(|| BrokenBackend::new(defect), &target(), &BrokenHooks);
        let case = report.case(case_id).unwrap();
        assert!(
            matches!(case.status, CaseStatus::Failed(_)),
            "{defect:?} was not detected by {case_id}: {report}"
        );
    }
}

#[test]
fn capability_skips_are_explicit_and_mandatory_cases_still_run() {
    let report = run(
        || BrokenBackend::new(Defect::CapabilitySubset),
        &target_in(MemoryDomain::Host),
        &BrokenHooks,
    );
    report.assert_conformant();
    for (case_id, capability) in [
        ("memory.device", Capabilities::DEVICE_LOCAL_MEMORY),
        ("memory.shared", Capabilities::SHARED_MEMORY),
        ("event.cancellation-races", Capabilities::EVENT_CANCELLATION),
    ] {
        assert_eq!(
            report.case(case_id).unwrap().status,
            CaseStatus::Skipped(SkipReason::CapabilityNotAdvertised(capability))
        );
    }
    assert!(report.cases().iter().all(|case| {
        !matches!(
            case.requirement,
            virtio_accel_conformance::CaseRequirement::Mandatory
        ) || case.status == CaseStatus::Passed
    }));
}

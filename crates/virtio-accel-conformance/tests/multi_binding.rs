//! Coverage for multi-binding target descriptions: programs with disjoint input and output
//! slots, read-only input fixtures, and the generalized negative submission subcases.

use virtio_accel_conformance::{
    BindingFixture, CaseStatus, ConformanceHooks, ProgramFixture, ResourceCounts,
    SubmissionPathDiagnostics, TargetDescription, TargetDescriptionError, run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc,
    ByteSink, ByteSource, ContextDesc, DeviceInfo, EventState, MemoryDomain, QueueDesc,
    ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_mock::fault::{FaultAccelerator, FaultEvent, FaultScript, ResourceState};
use virtio_accel_mock::reference::{
    ARTIFACT_FORMAT, RESIDENT_BYTES, ReferenceArtifact, TARGET_IDENTITY,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue,
};

const INPUT: [u8; 8] = [0x10, 0x22, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];

/// A copy program reading slot 0 and writing slot 1, described with one fixture per slot.
fn copy_target() -> TargetDescription {
    TargetDescription::with_bindings(
        ProgramFixture::new(
            ARTIFACT_FORMAT,
            TARGET_IDENTITY,
            ReferenceArtifact::copy(0, 1).unwrap().as_bytes(),
            RESIDENT_BYTES,
        )
        .unwrap(),
        vec![
            BindingFixture::read_only(0, MemoryDomain::Shared, 16, INPUT).unwrap(),
            BindingFixture::new(
                1,
                AccessMode::Write,
                MemoryDomain::Shared,
                16,
                [0u8; 8],
                INPUT,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

struct MockHooks;

impl ConformanceHooks<MockAccelerator> for MockHooks {
    fn complete_event(
        &self,
        backend: &MockAccelerator,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.complete(event)
    }

    fn submission_path_diagnostics(
        &self,
        backend: &MockAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

struct TrackedMockHooks;

impl ConformanceHooks<FaultAccelerator<MockAccelerator>> for TrackedMockHooks {
    fn complete_event(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
        event: &FaultEvent<MockEvent>,
    ) -> Result<(), BackendError> {
        backend.inner().complete(event.inner())
    }

    fn resource_counts(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
    ) -> Option<ResourceCounts> {
        let snapshot = backend.control().snapshot();
        let live = snapshot.resources_in(ResourceState::Live);
        let unknown = snapshot.resources_in(ResourceState::Indeterminate);
        Some(ResourceCounts {
            contexts: u64::from(live.contexts + unknown.contexts),
            buffers: u64::from(live.buffers + unknown.buffers),
            programs: u64::from(live.programs + unknown.programs),
            queues: u64::from(live.queues + unknown.queues),
            events: u64::from(live.events + unknown.events),
        })
    }

    fn submission_path_diagnostics(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.inner().direct_binding_admissions(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

#[test]
fn mock_copy_program_passes_the_generalized_suite() {
    let report = run(MockAccelerator::default, &copy_target(), &MockHooks);
    report.assert_conformant();
    assert_eq!(report.cases().len(), 15);
    assert!(matches!(
        report.case("accounting.resource-lifecycle").unwrap().status,
        CaseStatus::Skipped(_)
    ));
    assert!(
        report
            .cases()
            .iter()
            .filter(|case| case.id != "accounting.resource-lifecycle")
            .all(|case| case.status == CaseStatus::Passed)
    );
}

#[test]
fn ownership_accounting_is_balanced_with_multi_binding_fixtures() {
    let report = run(
        || FaultAccelerator::new(MockAccelerator::default(), FaultScript::default()),
        &copy_target(),
        &TrackedMockHooks,
    );
    report.assert_conformant();
    assert!(
        report
            .cases()
            .iter()
            .all(|case| case.status == CaseStatus::Passed)
    );
}

/// Delegating wrapper whose `read_buffer` corrupts any read-back that matches the known
/// read-only input bytes, simulating a program that clobbers its inputs.
struct CorruptingReads {
    inner: MockAccelerator,
}

#[derive(Debug)]
struct SliceSink<'a>(&'a mut [u8]);

impl ByteSink for SliceSink<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0, offset, source)
    }
}

impl Accelerator for CorruptingReads {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

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
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        let bytes = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let mut staged = vec![0; bytes];
        self.inner
            .read_buffer(buffer, offset, &mut SliceSink(&mut staged))?;
        if staged == INPUT {
            staged[0] ^= 0xff;
        }
        data.write_at(0, &staged)
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
        self.inner.submit(queue, program, bindings, timeout)
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

struct CorruptingHooks;

impl ConformanceHooks<CorruptingReads> for CorruptingHooks {
    fn complete_event(
        &self,
        backend: &CorruptingReads,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.inner.complete(event)
    }
}

#[test]
fn read_only_input_corruption_is_detected() {
    let report = run(
        || CorruptingReads {
            inner: MockAccelerator::default(),
        },
        &copy_target(),
        &CorruptingHooks,
    );
    assert!(!report.passed());
    let case = report.case("submission.binding-validation").unwrap();
    let CaseStatus::Failed(message) = &case.status else {
        panic!("input corruption was not detected: {:?}", case.status);
    };
    assert!(
        message.contains("slot 0"),
        "failure does not name the read-only slot: {message}"
    );
}

#[test]
fn read_only_fixtures_reject_invalid_shapes() {
    assert_eq!(
        BindingFixture::read_only(0, MemoryDomain::Shared, 16, Vec::new()).unwrap_err(),
        TargetDescriptionError::EmptyBinding
    );
    assert_eq!(
        BindingFixture::read_only(0, MemoryDomain::Shared, 3, [1u8]).unwrap_err(),
        TargetDescriptionError::InvalidAlignment
    );
    let fixture = BindingFixture::read_only(4, MemoryDomain::Host, 16, INPUT).unwrap();
    assert_eq!(fixture.access(), AccessMode::Read);
    assert_eq!(fixture.initial(), fixture.expected());
    assert!(!fixture.is_writable());
}

#[test]
fn writable_fixtures_still_reject_read_access_and_unobservable_outputs() {
    assert_eq!(
        BindingFixture::new(0, AccessMode::Read, MemoryDomain::Shared, 16, [1u8], [2u8])
            .unwrap_err(),
        TargetDescriptionError::ReadOnlyBinding
    );
    assert_eq!(
        BindingFixture::new(0, AccessMode::Write, MemoryDomain::Shared, 16, [1u8], [1u8])
            .unwrap_err(),
        TargetDescriptionError::UnobservableOutput
    );
}

fn program() -> ProgramFixture {
    ProgramFixture::new(
        ARTIFACT_FORMAT,
        TARGET_IDENTITY,
        ReferenceArtifact::xor(7, 0x5a).as_bytes(),
        RESIDENT_BYTES,
    )
    .unwrap()
}

fn writable(slot: u32) -> BindingFixture {
    BindingFixture::new(
        slot,
        AccessMode::Write,
        MemoryDomain::Shared,
        16,
        [0u8; 4],
        [1u8; 4],
    )
    .unwrap()
}

fn read_only(slot: u32) -> BindingFixture {
    BindingFixture::read_only(slot, MemoryDomain::Shared, 16, [7u8; 4]).unwrap()
}

#[test]
fn multi_binding_descriptions_validate_their_fixture_set() {
    assert_eq!(
        TargetDescription::with_bindings(program(), Vec::new()).unwrap_err(),
        TargetDescriptionError::NoBindings
    );
    assert_eq!(
        TargetDescription::with_bindings(program(), vec![writable(3), read_only(3)]).unwrap_err(),
        TargetDescriptionError::DuplicateBindingSlot
    );
    assert_eq!(
        TargetDescription::with_bindings(program(), vec![read_only(0), read_only(1)]).unwrap_err(),
        TargetDescriptionError::NoObservableBinding
    );

    let target =
        TargetDescription::with_bindings(program(), vec![read_only(0), writable(1)]).unwrap();
    assert_eq!(target.primary_index(), 1);
    assert_eq!(target.binding().slot(), 1);
    assert_eq!(target.bindings().len(), 2);
}

#[test]
fn single_binding_descriptions_are_unchanged() {
    let target = TargetDescription::new(program(), writable(7));
    assert_eq!(target.bindings().len(), 1);
    assert_eq!(target.primary_index(), 0);
    assert_eq!(target.binding().slot(), 7);
}

#[test]
#[should_panic(expected = "single-binding target must be writable")]
fn single_binding_descriptions_require_a_writable_fixture() {
    let _ = TargetDescription::new(program(), read_only(7));
}

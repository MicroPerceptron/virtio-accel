use virtio_accel::core::BackendError;
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, ResourceCounts, SubmissionPathDiagnostics,
    TargetDescription, run,
};
use virtio_accel_mock::fault::{FaultAccelerator, FaultEvent, FaultScript, ResourceState};
use virtio_accel_mock::{MockAccelerator, MockEvent, reference};

struct Hooks;

impl ConformanceHooks<FaultAccelerator<MockAccelerator>> for Hooks {
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

fn target() -> TargetDescription {
    let initial = vec![0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
    let expected = initial.iter().map(|byte| byte ^ 0x5a).collect::<Vec<_>>();
    TargetDescription::new(
        ProgramFixture::new(
            reference::ARTIFACT_FORMAT,
            reference::TARGET_IDENTITY,
            reference::ReferenceArtifact::xor(7, 0x5a).as_bytes(),
            reference::RESIDENT_BYTES,
        )
        .unwrap(),
        BindingFixture::new(
            7,
            virtio_accel::core::AccessMode::ReadWrite,
            virtio_accel::core::MemoryDomain::Shared,
            16,
            initial,
            expected,
        )
        .unwrap(),
    )
}

fn main() {
    let report = run(
        || FaultAccelerator::new(MockAccelerator::default(), FaultScript::default()),
        &target(),
        &Hooks,
    );
    report.assert_conformant();
    for case in report.cases() {
        println!("{}: {:?}", case.id, case.status);
    }
}

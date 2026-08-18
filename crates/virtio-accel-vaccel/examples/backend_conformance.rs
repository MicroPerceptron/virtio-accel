use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics, TargetDescription,
    run,
};
use virtio_accel_core::BackendError;
use virtio_accel_core::{AccessMode, MemoryDomain};
use virtio_accel_mock::{MockAccelerator, MockEvent, reference};
use virtio_accel_vaccel::VAccelAdapter;

struct Hooks;

impl ConformanceHooks<VAccelAdapter<MockAccelerator>> for Hooks {
    fn complete_event(
        &self,
        backend: &VAccelAdapter<MockAccelerator>,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.backend().complete(event)
    }

    fn submission_path_diagnostics(
        &self,
        backend: &VAccelAdapter<MockAccelerator>,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            shared_imported_bindings: 0,
            staged_direct_bindings: 0,
            staged_direct_bytes: 0,
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
        })
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
            AccessMode::ReadWrite,
            MemoryDomain::Shared,
            16,
            initial,
            expected,
        )
        .unwrap(),
    )
}

fn main() {
    let report = run(
        || VAccelAdapter::new(MockAccelerator::default()),
        &target(),
        &Hooks,
    );
    report.assert_conformant();
    for case in report.cases() {
        println!("{}: {:?}", case.id, case.status);
    }
}

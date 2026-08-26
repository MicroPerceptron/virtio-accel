//! The XDNA backend against the shared semantic conformance suite.
//!
//! This runs `virtio-accel-conformance`'s standard structural suite on a live NPU, including the
//! `submission.copy-path-diagnostics` case that proves each caller-owned allocation is bound
//! directly with no submission-time staging copy (issue #90). It compiles only in a `va_xdna` build
//! and needs both an accessible NPU and the pinned compiler toolchain (the suite's program is a
//! BF16 IDENTITY that `load_program` compiles); it skips cleanly when either is absent.
#![cfg(va_xdna)]

use std::time::Duration;

mod common;
use common::{bf16_identity_tosa, poll_to_terminal};

use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, ResourceCounts, SubmissionPathDiagnostics,
    TargetDescription, run,
};
use virtio_accel_core::{AccessMode, BackendError, EventState, MemoryDomain};
use virtio_accel_tosa::ARTIFACT_FORMAT;
use virtio_accel_xdna::{
    InitError, REQUIRED_RESIDENT_BYTES, XDNA_TOSA_TARGET, XdnaAccelerator, XdnaEvent,
};

const BUFFER_ALIGNMENT: u64 = 4096;
/// One IDENTITY DMA line (the smallest admitted count).
const ELEMENTS: usize = 1024;

fn toolchain_present() -> bool {
    std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_some()
}

/// Whether an NPU is accessible (a fresh backend constructs), so the suite can be skipped honestly.
fn device_present() -> bool {
    match XdnaAccelerator::new() {
        Ok(_) => true,
        Err(InitError::DeviceUnavailable) => false,
        Err(error) => panic!("unexpected initialization failure: {error}"),
    }
}

struct Hooks;

impl ConformanceHooks<XdnaAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &XdnaAccelerator,
        event: &XdnaEvent,
    ) -> Result<(), BackendError> {
        // The dispatch worker drives completion; poll the latched state to a terminal.
        match poll_to_terminal(backend, event, Duration::from_secs(30))? {
            EventState::Complete => Ok(()),
            EventState::Failed(error) => Err(error),
            EventState::Cancelled => Err(BackendError::Busy),
            EventState::Pending => unreachable!("poll_to_terminal never returns Pending"),
        }
    }

    fn resource_counts(&self, backend: &XdnaAccelerator) -> Option<ResourceCounts> {
        let counts = backend.resource_counts();
        Some(ResourceCounts {
            contexts: counts.contexts,
            buffers: counts.buffers,
            programs: counts.programs,
            queues: counts.queues,
            events: counts.events,
        })
    }

    fn submission_path_diagnostics(
        &self,
        backend: &XdnaAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

/// A BF16 IDENTITY program with a read input and a write output. The output binding's initial bytes
/// differ from the expected (post-copy) bytes so the harness can observe the device having run.
fn conformance_target() -> TargetDescription {
    let bytes = ELEMENTS * 2; // bf16
    // A nonzero pattern, distinct from the zeroed output initial state.
    let input: Vec<u8> = (0..bytes).map(|i| (i % 251 + 1) as u8).collect();
    let program = ProgramFixture::new(
        ARTIFACT_FORMAT,
        XDNA_TOSA_TARGET.to_identity(),
        bf16_identity_tosa(ELEMENTS),
        REQUIRED_RESIDENT_BYTES,
    )
    .unwrap();
    TargetDescription::with_bindings(
        program,
        vec![
            BindingFixture::read_only(0, MemoryDomain::Shared, BUFFER_ALIGNMENT, input.clone())
                .unwrap(),
            BindingFixture::new(
                1,
                AccessMode::Write,
                MemoryDomain::Shared,
                BUFFER_ALIGNMENT,
                vec![0u8; bytes],
                input,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn xdna_backend_passes_the_standard_semantic_suite() {
    if !device_present() {
        eprintln!("no XDNA NPU device accessible; skipping conformance suite");
        return;
    }
    if !toolchain_present() {
        eprintln!("no XDNA toolchain configured; skipping conformance suite");
        return;
    }
    let target = conformance_target();
    let report = run(
        || XdnaAccelerator::new().expect("construct XDNA backend"),
        &target,
        &Hooks,
    );
    // The direct-binding diagnostics case must have run (not skipped) and passed.
    let diagnostics = report
        .case("submission.copy-path-diagnostics")
        .expect("diagnostics case present");
    assert!(
        matches!(
            diagnostics.status,
            virtio_accel_conformance::CaseStatus::Passed
        ),
        "direct-binding diagnostics did not pass: {diagnostics:?}"
    );
    let accounting = report
        .case("accounting.resource-lifecycle")
        .expect("accounting case present");
    assert!(
        matches!(
            accounting.status,
            virtio_accel_conformance::CaseStatus::Passed
        ),
        "resource accounting did not pass: {accounting:?}"
    );
    let cancellation = report
        .case("event.cancellation-races")
        .expect("cancellation case present");
    assert!(matches!(
        cancellation.status,
        virtio_accel_conformance::CaseStatus::Skipped(
            virtio_accel_conformance::SkipReason::CapabilityNotAdvertised(capability)
        ) if capability == virtio_accel_core::Capabilities::EVENT_CANCELLATION
    ));
    let device_memory = report
        .case("memory.device")
        .expect("device-memory case present");
    assert!(matches!(
        device_memory.status,
        virtio_accel_conformance::CaseStatus::Skipped(
            virtio_accel_conformance::SkipReason::CapabilityNotAdvertised(capability)
        ) if capability == virtio_accel_core::Capabilities::DEVICE_LOCAL_MEMORY
    ));
    for case in report.cases() {
        if matches!(
            case.status,
            virtio_accel_conformance::CaseStatus::Skipped(_)
        ) {
            assert!(
                matches!(case.id, "memory.device" | "event.cancellation-races"),
                "unexpected conformance skip: {case:?}"
            );
        }
    }
    report.assert_conformant();
}

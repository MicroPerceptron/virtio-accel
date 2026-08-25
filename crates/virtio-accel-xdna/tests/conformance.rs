//! The XDNA backend against the shared semantic conformance suite.
//!
//! This runs `virtio-accel-conformance`'s standard structural suite on a live NPU, including the
//! `submission.copy-path-diagnostics` case that proves each caller-owned allocation is bound
//! directly with no submission-time staging copy (issue #90). It compiles only in a `va_xdna` build
//! and needs both an accessible NPU and the pinned compiler toolchain (the suite's program is a
//! BF16 IDENTITY that `load_program` compiles); it skips cleanly when either is absent.
#![cfg(va_xdna)]

use std::time::{Duration, Instant};

use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics, TargetDescription,
    run,
};
use virtio_accel_core::{Accelerator, AccessMode, BackendError, EventState, MemoryDomain};
use virtio_accel_tosa::{ARTIFACT_FORMAT, DType};
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
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

/// A BF16 IDENTITY TOSA artifact of `ELEMENTS` for the advertised target.
fn bf16_identity_tosa() -> Vec<u8> {
    let shape = vec![1, 1, ELEMENTS as i32];
    let mut graph = OwnedGraph::new("main");
    graph.push_tensor(OwnedTensor::new("x", shape.clone(), DType::BF16));
    graph.push_tensor(OwnedTensor::new("y", shape, DType::BF16));
    graph.push_operator(OwnedOperator::new(
        OperatorKind::Identity,
        vec!["x".into()],
        vec!["y".into()],
    ));
    graph.push_input("x");
    graph.push_output("y");
    graph.build(XDNA_TOSA_TARGET).expect("build bf16 identity")
}

struct Hooks;

impl ConformanceHooks<XdnaAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &XdnaAccelerator,
        event: &XdnaEvent,
    ) -> Result<(), BackendError> {
        // The dispatch worker drives completion; poll the latched state to a terminal.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match backend.poll_event(event)? {
                EventState::Pending => {
                    if Instant::now() >= deadline {
                        return Err(BackendError::DeadlineExpired);
                    }
                    std::thread::yield_now();
                }
                EventState::Complete => return Ok(()),
                EventState::Failed(error) => return Err(error),
                EventState::Cancelled => return Err(BackendError::Busy),
            }
        }
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
        bf16_identity_tosa(),
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
    report.assert_conformant();
}

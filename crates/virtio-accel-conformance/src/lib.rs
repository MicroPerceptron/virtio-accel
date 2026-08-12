//! Reusable semantic conformance tests for [`virtio_accel_core::Accelerator`] backends.
//!
//! The suite depends only on the transport-independent backend contract. An implementer supplies a
//! fresh-backend factory, one executable target fixture, and a small hook that advances an event to
//! completion. Optional accounting lets the same cases detect leaked provider resources. The
//! [`numerics`] module adds stable, device-neutral TOSA graphs and numerical oracles that host
//! backends can execute without substituting provider-specific fixtures.

#![forbid(unsafe_code)]

mod cases;
pub mod numerics;

use std::fmt;
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactFormat, BackendError, Capabilities, MemoryDomain,
    TargetIdentity,
};

/// Provider artifact accepted by the backend under test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramFixture {
    format: ArtifactFormat,
    target: TargetIdentity,
    payload: Vec<u8>,
    resident_bytes: u64,
}

impl ProgramFixture {
    /// Build one owned provider artifact fixture.
    pub fn new(
        format: ArtifactFormat,
        target: TargetIdentity,
        payload: impl Into<Vec<u8>>,
        resident_bytes: u64,
    ) -> Result<Self, TargetDescriptionError> {
        let payload = payload.into();
        if payload.is_empty() {
            return Err(TargetDescriptionError::EmptyArtifact);
        }
        if resident_bytes == 0 {
            return Err(TargetDescriptionError::ZeroResidentBytes);
        }
        Ok(Self {
            format,
            target,
            payload,
            resident_bytes,
        })
    }

    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }

    pub const fn target(&self) -> TargetIdentity {
        self.target
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// One observable program binding used by the standard execution cases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingFixture {
    slot: u32,
    access: AccessMode,
    domain: MemoryDomain,
    alignment: u64,
    initial: Vec<u8>,
    expected: Vec<u8>,
}

impl BindingFixture {
    /// Describe one nonempty binding and its expected bytes after successful completion.
    pub fn new(
        slot: u32,
        access: AccessMode,
        domain: MemoryDomain,
        alignment: u64,
        initial: impl Into<Vec<u8>>,
        expected: impl Into<Vec<u8>>,
    ) -> Result<Self, TargetDescriptionError> {
        let initial = initial.into();
        let expected = expected.into();
        if initial.is_empty() {
            return Err(TargetDescriptionError::EmptyBinding);
        }
        if initial.len() != expected.len() {
            return Err(TargetDescriptionError::OutputLengthMismatch);
        }
        if initial == expected {
            return Err(TargetDescriptionError::UnobservableOutput);
        }
        if access == AccessMode::Read {
            return Err(TargetDescriptionError::ReadOnlyBinding);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(TargetDescriptionError::InvalidAlignment);
        }
        Ok(Self {
            slot,
            access,
            domain,
            alignment,
            initial,
            expected,
        })
    }

    pub const fn slot(&self) -> u32 {
        self.slot
    }

    pub const fn access(&self) -> AccessMode {
        self.access
    }

    pub const fn domain(&self) -> MemoryDomain {
        self.domain
    }

    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn initial(&self) -> &[u8] {
        &self.initial
    }

    pub fn expected(&self) -> &[u8] {
        &self.expected
    }

    pub fn bytes(&self) -> u64 {
        self.initial.len() as u64
    }
}

/// Owned provider-specific data required by the transport-free standard suite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetDescription {
    program: ProgramFixture,
    binding: BindingFixture,
}

impl TargetDescription {
    pub const fn new(program: ProgramFixture, binding: BindingFixture) -> Self {
        Self { program, binding }
    }

    pub const fn program(&self) -> &ProgramFixture {
        &self.program
    }

    pub const fn binding(&self) -> &BindingFixture {
        &self.binding
    }
}

/// Invalid provider fixture supplied to the conformance suite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetDescriptionError {
    EmptyArtifact,
    ZeroResidentBytes,
    EmptyBinding,
    OutputLengthMismatch,
    UnobservableOutput,
    ReadOnlyBinding,
    InvalidAlignment,
}

impl fmt::Display for TargetDescriptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TargetDescriptionError {}

/// Provider-owned resource totals sampled by an optional accounting hook.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceCounts {
    pub contexts: u64,
    pub buffers: u64,
    pub programs: u64,
    pub queues: u64,
    pub events: u64,
}

impl ResourceCounts {
    pub const fn is_empty(self) -> bool {
        self.contexts == 0
            && self.buffers == 0
            && self.programs == 0
            && self.queues == 0
            && self.events == 0
    }
}

/// Provider-reported submission path counters used to separate real accelerator work from staging.
///
/// A backend that exposes these diagnostics should report cumulative counts for one backend
/// instance. Direct bindings are provider-owned buffers submitted without a bounce allocation.
/// Shared/imported bindings are reserved for future external-memory paths. Staged direct bindings
/// are buffers that should have been directly bound but were instead copied through a temporary
/// submission buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmissionPathDiagnostics {
    pub direct_bindings: u64,
    pub shared_imported_bindings: u64,
    pub staged_direct_bindings: u64,
    pub staged_direct_bytes: u64,
    pub explicit_transfer_bytes: u64,
}

impl SubmissionPathDiagnostics {
    pub const fn has_hidden_direct_staging(self) -> bool {
        self.staged_direct_bindings != 0 || self.staged_direct_bytes != 0
    }

    pub const fn saturating_delta(self, before: Self) -> Self {
        Self {
            direct_bindings: self.direct_bindings.saturating_sub(before.direct_bindings),
            shared_imported_bindings: self
                .shared_imported_bindings
                .saturating_sub(before.shared_imported_bindings),
            staged_direct_bindings: self
                .staged_direct_bindings
                .saturating_sub(before.staged_direct_bindings),
            staged_direct_bytes: self
                .staged_direct_bytes
                .saturating_sub(before.staged_direct_bytes),
            explicit_transfer_bytes: self
                .explicit_transfer_bytes
                .saturating_sub(before.explicit_transfer_bytes),
        }
    }
}

/// Provider-specific control needed by tests without weakening the [`Accelerator`] contract.
pub trait ConformanceHooks<A: Accelerator> {
    /// Advance one pending event to its successful terminal state.
    fn complete_event(&self, backend: &A, event: &A::Event) -> Result<(), BackendError>;

    /// Return live or indeterminate provider resource totals when the implementation exposes them.
    fn resource_counts(&self, _backend: &A) -> Option<ResourceCounts> {
        None
    }

    /// Return cumulative provider copy-path diagnostics when the implementation exposes them.
    fn submission_path_diagnostics(&self, _backend: &A) -> Option<SubmissionPathDiagnostics> {
        None
    }
}

/// Why a conformance case is required or conditional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseRequirement {
    Mandatory,
    Capability(Capabilities),
    AccountingHook,
    DiagnosticsHook,
}

/// Explicit reason that one conditional case did not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    CapabilityNotAdvertised(Capabilities),
    AccountingUnavailable,
    DiagnosticsUnavailable,
}

/// Outcome of one stable conformance case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaseStatus {
    Passed,
    Skipped(SkipReason),
    Failed(String),
}

/// Result for one independently constructed backend instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseResult {
    pub id: &'static str,
    pub name: &'static str,
    pub requirement: CaseRequirement,
    pub status: CaseStatus,
}

/// Aggregate result from the standard semantic backend suite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConformanceReport {
    cases: Vec<CaseResult>,
}

impl ConformanceReport {
    pub fn cases(&self) -> &[CaseResult] {
        &self.cases
    }

    pub fn case(&self, id: &str) -> Option<&CaseResult> {
        self.cases.iter().find(|case| case.id == id)
    }

    pub fn failures(&self) -> impl Iterator<Item = &CaseResult> {
        self.cases
            .iter()
            .filter(|case| matches!(case.status, CaseStatus::Failed(_)))
    }

    pub fn passed(&self) -> bool {
        self.failures().next().is_none()
    }

    /// Panic with every failed case, suitable for a backend crate's ordinary test target.
    #[track_caller]
    pub fn assert_conformant(&self) {
        assert!(self.passed(), "{self}");
    }
}

impl fmt::Display for ConformanceReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut failures = self.failures().peekable();
        if failures.peek().is_none() {
            return formatter.write_str("backend conformance passed");
        }
        formatter.write_str("backend conformance failed:")?;
        for case in failures {
            let CaseStatus::Failed(message) = &case.status else {
                continue;
            };
            write!(formatter, "\n- {}: {}", case.id, message)?;
        }
        Ok(())
    }
}

/// Run every mandatory and advertised capability-conditional semantic case.
///
/// `factory` is called once per case so an indeterminate result or accounting contradiction can
/// never contaminate a later case. The returned report preserves skips and all failures instead of
/// stopping at the first defect.
pub fn run<A, F, H>(factory: F, target: &TargetDescription, hooks: &H) -> ConformanceReport
where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
{
    ConformanceReport {
        cases: cases::run_all(&factory, target, hooks),
    }
}

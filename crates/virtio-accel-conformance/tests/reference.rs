mod support;

use support::target;
use virtio_accel_conformance::{CaseStatus, ConformanceHooks, ResourceCounts, run};
use virtio_accel_core::BackendError;
use virtio_accel_mock::fault::{FaultAccelerator, FaultEvent, FaultScript, ResourceState};
use virtio_accel_mock::{MockAccelerator, MockEvent};

struct MockHooks;

impl ConformanceHooks<MockAccelerator> for MockHooks {
    fn complete_event(
        &self,
        backend: &MockAccelerator,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.complete(event)
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
}

#[test]
fn reference_backend_passes_every_mandatory_and_advertised_case() {
    let report = run(MockAccelerator::default, &target(), &MockHooks);
    report.assert_conformant();
    assert_eq!(report.cases().len(), 14);
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
fn ownership_accounting_is_balanced_after_every_reference_case() {
    let report = run(
        || FaultAccelerator::new(MockAccelerator::default(), FaultScript::default()),
        &target(),
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

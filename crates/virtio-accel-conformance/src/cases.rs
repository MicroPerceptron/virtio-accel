use crate::{
    CaseRequirement, CaseResult, CaseStatus, ConformanceHooks, SkipReason, TargetDescription,
};
use core::num::NonZeroU64;
use std::string::String;
use std::vec;
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, Capabilities, ContextDesc, ContextFlags, DeviceInfo,
    EventState, MemoryDomain, QueueDesc, QueueFlags, ReleaseFailure, SubmitFailure, Timeout,
};

type CaseCheck = Result<(), String>;

pub(crate) fn run_all<A, F, H>(
    factory: &F,
    target: &TargetDescription,
    hooks: &H,
) -> Vec<CaseResult>
where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
{
    let capabilities = factory().device_info().ok().map(|info| info.capabilities);
    let mut results = Vec::new();
    run_case(
        &mut results,
        factory,
        hooks,
        "metadata.stable-valid",
        "stable valid device metadata",
        CaseRequirement::Mandatory,
        || true,
        |backend| metadata_is_stable_and_valid(backend),
    );
    run_case(
        &mut results,
        factory,
        hooks,
        "intent.reserved-flags",
        "reserved creation intent is rejected",
        CaseRequirement::Mandatory,
        || true,
        |backend| reserved_creation_intent_is_rejected(backend),
    );
    for (id, name, capability, domain) in [
        (
            "memory.host",
            "host-visible allocation contract",
            Capabilities::HOST_VISIBLE_MEMORY,
            MemoryDomain::Host,
        ),
        (
            "memory.device",
            "device-local allocation contract",
            Capabilities::DEVICE_LOCAL_MEMORY,
            MemoryDomain::Device,
        ),
        (
            "memory.shared",
            "shared allocation contract",
            Capabilities::SHARED_MEMORY,
            MemoryDomain::Shared,
        ),
    ] {
        run_capability_case(
            &mut results,
            factory,
            target,
            hooks,
            capabilities,
            id,
            name,
            capability,
            move |backend, target, _| allocation_contract(backend, target, domain),
        );
    }
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "buffer.segmented-transfer-bounds",
        "segmented transfers and range bounds",
        CaseRequirement::Mandatory,
        |backend, target, _| segmented_transfers_and_bounds(backend, target),
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "buffer.transfer-permissions",
        "explicit transfer permissions",
        CaseRequirement::Mandatory,
        |backend, target, _| transfer_permissions(backend, target),
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "program.segmented-artifact-bounds",
        "segmented artifact and advertised bounds",
        CaseRequirement::Mandatory,
        |backend, target, _| segmented_artifact_and_bounds(backend, target),
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "submission.binding-validation",
        "binding count, uniqueness, range, and access",
        CaseRequirement::Mandatory,
        submission_binding_validation,
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "submission.context-isolation",
        "cross-context admission rejection",
        CaseRequirement::Mandatory,
        context_isolation,
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "event.pending-release-terminal-stability",
        "pending release retry and stable completion",
        CaseRequirement::Mandatory,
        pending_release_and_terminal_stability,
    );
    run_target_case(
        &mut results,
        factory,
        target,
        hooks,
        "timeout.finite-admission",
        "finite timeout preserves the admission boundary",
        CaseRequirement::Mandatory,
        finite_timeout_preserves_admission,
    );
    run_capability_case(
        &mut results,
        factory,
        target,
        hooks,
        capabilities,
        "event.cancellation-races",
        "cancellation wins and loses deterministically",
        Capabilities::EVENT_CANCELLATION,
        cancellation_races,
    );
    run_accounting_case(&mut results, factory, hooks);
    results
}

#[allow(clippy::too_many_arguments)]
fn run_case<A, F, H, E, C>(
    results: &mut Vec<CaseResult>,
    factory: &F,
    hooks: &H,
    id: &'static str,
    name: &'static str,
    requirement: CaseRequirement,
    enabled: E,
    check: C,
) where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
    E: FnOnce() -> bool,
    C: FnOnce(&A) -> CaseCheck,
{
    if !enabled() {
        let CaseRequirement::Capability(capability) = requirement else {
            unreachable!("only capability cases may be disabled")
        };
        results.push(CaseResult {
            id,
            name,
            requirement,
            status: CaseStatus::Skipped(SkipReason::CapabilityNotAdvertised(capability)),
        });
        return;
    }

    let backend = factory();
    let mut failure = hooks
        .resource_counts(&backend)
        .filter(|counts| !counts.is_empty())
        .map(|counts| format!("factory returned live resources: {counts:?}"));
    if failure.is_none() {
        failure = check(&backend).err();
    }
    if let Some(counts) = hooks
        .resource_counts(&backend)
        .filter(|counts| !counts.is_empty())
    {
        append_failure(
            &mut failure,
            format!("provider resources remain after the case: {counts:?}"),
        );
    }
    results.push(CaseResult {
        id,
        name,
        requirement,
        status: match failure {
            Some(message) => CaseStatus::Failed(message),
            None => CaseStatus::Passed,
        },
    });
}

#[allow(clippy::too_many_arguments)]
fn run_target_case<A, F, H, C>(
    results: &mut Vec<CaseResult>,
    factory: &F,
    target: &TargetDescription,
    hooks: &H,
    id: &'static str,
    name: &'static str,
    requirement: CaseRequirement,
    check: C,
) where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
    C: FnOnce(&A, &TargetDescription, &H) -> CaseCheck,
{
    run_case(
        results,
        factory,
        hooks,
        id,
        name,
        requirement,
        || true,
        |backend| check(backend, target, hooks),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_capability_case<A, F, H, C>(
    results: &mut Vec<CaseResult>,
    factory: &F,
    target: &TargetDescription,
    hooks: &H,
    advertised: Option<Capabilities>,
    id: &'static str,
    name: &'static str,
    capability: Capabilities,
    check: C,
) where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
    C: FnOnce(&A, &TargetDescription, &H) -> CaseCheck,
{
    let enabled = advertised
        .map(|capabilities| capabilities.contains(capability))
        .unwrap_or(true);
    run_case(
        results,
        factory,
        hooks,
        id,
        name,
        CaseRequirement::Capability(capability),
        || enabled,
        |backend| check(backend, target, hooks),
    );
}

fn run_accounting_case<A, F, H>(results: &mut Vec<CaseResult>, factory: &F, hooks: &H)
where
    A: Accelerator,
    F: Fn() -> A,
    H: ConformanceHooks<A>,
{
    let backend = factory();
    let status = match hooks.resource_counts(&backend) {
        Some(counts) if counts.is_empty() => CaseStatus::Passed,
        Some(counts) => CaseStatus::Failed(format!(
            "fresh backend reports provider resources: {counts:?}"
        )),
        None => CaseStatus::Skipped(SkipReason::AccountingUnavailable),
    };
    results.push(CaseResult {
        id: "accounting.resource-lifecycle",
        name: "optional resource accounting is balanced",
        requirement: CaseRequirement::AccountingHook,
        status,
    });
}

fn append_failure(failure: &mut Option<String>, message: String) {
    match failure {
        Some(existing) => {
            existing.push_str("; ");
            existing.push_str(&message);
        }
        None => *failure = Some(message),
    }
}

fn metadata_is_stable_and_valid<A: Accelerator>(backend: &A) -> CaseCheck {
    let first = backend
        .device_info()
        .map_err(|error| format!("initial discovery failed: {error:?}"))?;
    first
        .validate()
        .map_err(|error| format!("invalid device metadata: {error:?}"))?;
    let second = backend
        .device_info()
        .map_err(|error| format!("repeat discovery failed: {error:?}"))?;
    if first != second {
        return Err(format!(
            "device metadata changed within one backend instance: {first:?} != {second:?}"
        ));
    }
    Ok(())
}

fn reserved_creation_intent_is_rejected<A: Accelerator>(backend: &A) -> CaseCheck {
    match backend.create_context(ContextDesc {
        flags: ContextFlags::SECURE,
    }) {
        Err(BackendError::Unsupported) => {}
        Err(error) => return Err(format!("reserved context flag returned {error:?}")),
        Ok(context) => {
            let _ = release_context(backend, context);
            return Err("reserved context flag created a context".into());
        }
    }

    let context = create_context(backend)?;
    let result = match backend.create_queue(
        &context,
        QueueDesc {
            flags: QueueFlags::IN_ORDER,
        },
    ) {
        Err(BackendError::Unsupported) => Ok(()),
        Err(error) => Err(format!("reserved execution-queue flag returned {error:?}")),
        Ok(queue) => {
            let _ = release_queue(backend, queue);
            Err("reserved execution-queue flag created a queue".into())
        }
    };
    merge(result, release_context(backend, context))
}

fn allocation_contract<A: Accelerator>(
    backend: &A,
    target: &TargetDescription,
    domain: MemoryDomain,
) -> CaseCheck {
    let info = device_info(backend)?;
    let context = create_context(backend)?;
    let desc = BufferDesc::new(
        target.binding().bytes(),
        target.binding().alignment(),
        domain,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION,
    )
    .map_err(|error| format!("invalid memory-domain fixture: {error:?}"))?;
    let result = match backend.allocate_buffer(&context, desc) {
        Ok(allocation) => {
            let actual = allocation.info();
            let (buffer, _) = allocation.into_parts();
            let check = info
                .validate_buffer_info(desc, actual)
                .map_err(|error| format!("allocation metadata is dishonest: {error:?}"));
            merge(check, release_buffer(backend, buffer))
        }
        Err(error) => Err(format!(
            "advertised {domain:?} allocation failed: {error:?}"
        )),
    };
    merge(result, release_context(backend, context))
}

fn segmented_transfers_and_bounds<A: Accelerator>(
    backend: &A,
    target: &TargetDescription,
) -> CaseCheck {
    ensure_target_fits(backend, target)?;
    let context = create_context(backend)?;
    let desc = target_buffer_desc(target, target.binding().domain())?;
    let allocation = backend
        .allocate_buffer(&context, desc)
        .map_err(|error| format!("target buffer allocation failed: {error:?}"))?;
    let (mut buffer, info) = allocation.into_parts();
    let operation = (|| {
        device_info(backend)?
            .validate_buffer_info(desc, info)
            .map_err(|error| format!("target allocation metadata is dishonest: {error:?}"))?;
        let midpoint = target.binding().initial().len() / 2;
        let source = SplitSource {
            first: &target.binding().initial()[..midpoint],
            second: &target.binding().initial()[midpoint..],
        };
        backend
            .write_buffer(&mut buffer, 0, &source)
            .map_err(|error| format!("segmented write failed: {error:?}"))?;

        let mut output = vec![0; target.binding().initial().len()];
        let midpoint = output.len() / 2;
        let (first, second) = output.split_at_mut(midpoint);
        let mut sink = SplitSink { first, second };
        backend
            .read_buffer(&buffer, 0, &mut sink)
            .map_err(|error| format!("segmented read failed: {error:?}"))?;
        if output != target.binding().initial() {
            return Err(format!(
                "explicit transfer round trip changed bytes: {output:?}"
            ));
        }

        expect_backend_error(
            backend.write_buffer(&mut buffer, desc.bytes(), &[0xa5]),
            BackendError::OutOfBounds,
            "out-of-bounds write",
        )?;
        let mut byte = [0];
        expect_backend_error(
            backend.read_buffer(&buffer, desc.bytes(), &mut byte),
            BackendError::OutOfBounds,
            "out-of-bounds read",
        )
    })();
    let released = release_buffer(backend, buffer);
    merge(
        merge(operation, released),
        release_context(backend, context),
    )
}

fn transfer_permissions<A: Accelerator>(backend: &A, target: &TargetDescription) -> CaseCheck {
    ensure_target_fits(backend, target)?;
    let context = create_context(backend)?;
    let source_desc = BufferDesc::new(
        target.binding().bytes(),
        target.binding().alignment(),
        target.binding().domain(),
        BufferUsage::TRANSFER_SOURCE,
    )
    .map_err(|error| format!("invalid source-only fixture: {error:?}"))?;
    let destination_desc = BufferDesc::new(
        target.binding().bytes(),
        target.binding().alignment(),
        target.binding().domain(),
        BufferUsage::TRANSFER_DESTINATION,
    )
    .map_err(|error| format!("invalid destination-only fixture: {error:?}"))?;
    let (mut source_only, _) = backend
        .allocate_buffer(&context, source_desc)
        .map_err(|error| format!("source-only allocation failed: {error:?}"))?
        .into_parts();
    let (destination_only, _) = backend
        .allocate_buffer(&context, destination_desc)
        .map_err(|error| format!("destination-only allocation failed: {error:?}"))?
        .into_parts();
    let operation = (|| {
        expect_backend_error(
            backend.write_buffer(&mut source_only, 0, &[0]),
            BackendError::PermissionDenied,
            "write without transfer-destination usage",
        )?;
        let mut byte = [0];
        expect_backend_error(
            backend.read_buffer(&destination_only, 0, &mut byte),
            BackendError::PermissionDenied,
            "read without transfer-source usage",
        )
    })();
    let release_source = release_buffer(backend, source_only);
    let release_destination = release_buffer(backend, destination_only);
    merge(
        merge(merge(operation, release_source), release_destination),
        release_context(backend, context),
    )
}

fn segmented_artifact_and_bounds<A: Accelerator>(
    backend: &A,
    target: &TargetDescription,
) -> CaseCheck {
    ensure_target_fits(backend, target)?;
    let info = device_info(backend)?;
    let context = create_context(backend)?;
    let payload = target.program().payload();
    let midpoint = payload.len() / 2;
    let segmented = SplitSource {
        first: &payload[..midpoint],
        second: &payload[midpoint..],
    };
    let artifact = ArtifactRef {
        format: target.program().format(),
        target: target.program().target(),
        payload: &segmented,
        resident_bytes: target.program().resident_bytes(),
    };
    let program = backend
        .load_program(&context, artifact)
        .map_err(|error| format!("segmented target artifact failed to load: {error:?}"))?;
    let mut result = release_program(backend, program);
    if let Some(oversized) = info.limits.max_artifact_bytes.checked_add(1) {
        let source = LengthOnlySource(oversized);
        let artifact = ArtifactRef {
            format: target.program().format(),
            target: target.program().target(),
            payload: &source,
            resident_bytes: target.program().resident_bytes(),
        };
        let check = expect_backend_error(
            backend.load_program(&context, artifact),
            BackendError::ResourceLimit,
            "artifact larger than the advertised limit",
        );
        result = merge(result, check);
    }
    merge(result, release_context(backend, context))
}

fn submission_binding_validation<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
) -> CaseCheck {
    let resources = StandardResources::create(backend, target)?;
    let operation = (|| {
        expect_submit_rejected(
            backend.submit(&resources.queue, &resources.program, &[], Timeout::Infinite),
            BackendError::ResourceLimit,
            "empty binding list",
        )?;

        let binding = valid_binding(target, &resources.buffer)?;
        if device_info(backend)?.limits.max_bindings_per_submission >= 2 {
            let duplicate = [
                BindingRef {
                    slot: binding.slot,
                    buffer: binding.buffer,
                    range: binding.range,
                    access: binding.access,
                },
                binding,
            ];
            expect_submit_rejected(
                backend.submit(
                    &resources.queue,
                    &resources.program,
                    &duplicate,
                    Timeout::Infinite,
                ),
                BackendError::InvalidArgument,
                "duplicate binding slots",
            )?;
        }

        let out_of_bounds = [BindingRef {
            slot: target.binding().slot(),
            buffer: &resources.buffer,
            range: BufferRange::new(target.binding().bytes(), 1).unwrap(),
            access: target.binding().access(),
        }];
        expect_submit_rejected(
            backend.submit(
                &resources.queue,
                &resources.program,
                &out_of_bounds,
                Timeout::Infinite,
            ),
            BackendError::OutOfBounds,
            "binding range outside the buffer",
        )?;

        let wrong_access = [BindingRef {
            slot: target.binding().slot(),
            buffer: &resources.buffer,
            range: BufferRange::new(0, target.binding().bytes()).unwrap(),
            access: different_access(target.binding().access()),
        }];
        expect_submit_rejected(
            backend.submit(
                &resources.queue,
                &resources.program,
                &wrong_access,
                Timeout::Infinite,
            ),
            BackendError::Incompatible,
            "program-incompatible binding access",
        )?;

        let valid = [valid_binding(target, &resources.buffer)?];
        let event = backend
            .submit(
                &resources.queue,
                &resources.program,
                &valid,
                Timeout::Infinite,
            )
            .map_err(|failure| {
                format!(
                    "valid submission was rejected: {:?}",
                    failure_error(&failure)
                )
            })?;
        complete_event(backend, target, hooks, &resources.buffer, event)
    })();
    merge(operation, resources.release(backend))
}

fn context_isolation<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    _hooks: &H,
) -> CaseCheck {
    ensure_target_fits(backend, target)?;
    let first = create_context(backend)?;
    let second = create_context(backend)?;
    let desc = target_buffer_desc(target, target.binding().domain())?;
    let (mut buffer, _) = backend
        .allocate_buffer(&second, desc)
        .map_err(|error| format!("second-context buffer allocation failed: {error:?}"))?
        .into_parts();
    let initial = SliceSource(target.binding().initial());
    backend
        .write_buffer(&mut buffer, 0, &initial)
        .map_err(|error| format!("second-context initialization failed: {error:?}"))?;
    let program = load_target(backend, &first, target)?;
    let queue = backend
        .create_queue(&first, QueueDesc::default())
        .map_err(|error| format!("first-context queue creation failed: {error:?}"))?;
    let bindings = [valid_binding(target, &buffer)?];
    let operation = expect_submit_rejected(
        backend.submit(&queue, &program, &bindings, Timeout::Infinite),
        BackendError::InvalidArgument,
        "cross-context buffer admission",
    );
    let released = merge(
        merge(
            release_queue(backend, queue),
            release_program(backend, program),
        ),
        release_buffer(backend, buffer),
    );
    merge(
        merge(operation, released),
        merge(
            release_context(backend, first),
            release_context(backend, second),
        ),
    )
}

fn pending_release_and_terminal_stability<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
) -> CaseCheck {
    let resources = StandardResources::create(backend, target)?;
    let bindings = [valid_binding(target, &resources.buffer)?];
    let event = backend
        .submit(
            &resources.queue,
            &resources.program,
            &bindings,
            Timeout::Infinite,
        )
        .map_err(|failure| format!("pending submission failed: {:?}", failure_error(&failure)))?;
    let event = match backend.poll_event(&event) {
        Ok(EventState::Pending) => match backend.destroy_event(event) {
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource,
            }) => resource,
            Err(ReleaseFailure::Rejected { error, .. }) => {
                return Err(format!("pending event release returned {error:?}"));
            }
            Err(ReleaseFailure::Indeterminate { error }) => {
                return Err(format!(
                    "pending event release became indeterminate: {error:?}"
                ));
            }
            Ok(()) => return Err("pending event release reported success".into()),
        },
        Ok(state) => {
            return Err(format!(
                "target did not expose a controllable pending event before the progress hook: {state:?}"
            ));
        }
        Err(error) => return Err(format!("initial event poll failed: {error:?}")),
    };
    let operation = complete_event(backend, target, hooks, &resources.buffer, event);
    merge(operation, resources.release(backend))
}

fn finite_timeout_preserves_admission<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
) -> CaseCheck {
    let resources = StandardResources::create(backend, target)?;
    let bindings = [valid_binding(target, &resources.buffer)?];
    let timeout = Timeout::AfterNs(NonZeroU64::new(1).unwrap());
    let operation = match backend.submit(&resources.queue, &resources.program, &bindings, timeout) {
        Ok(event) => settle_timed_event(backend, target, hooks, &resources.buffer, event),
        Err(SubmitFailure::Rejected(BackendError::DeadlineExpired)) => Ok(()),
        Err(SubmitFailure::Rejected(error)) => {
            Err(format!("finite timeout was rejected as {error:?}"))
        }
        Err(SubmitFailure::Indeterminate { error, event }) => {
            let truth = if error == BackendError::DeadlineExpired {
                Ok(())
            } else {
                Err(format!("indeterminate timeout returned {error:?}"))
            };
            merge(
                truth,
                settle_timed_event(backend, target, hooks, &resources.buffer, event),
            )
        }
    };
    merge(operation, resources.release(backend))
}

fn cancellation_races<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
) -> CaseCheck {
    let resources = StandardResources::create(backend, target)?;
    let first_bindings = [valid_binding(target, &resources.buffer)?];
    let first = backend
        .submit(
            &resources.queue,
            &resources.program,
            &first_bindings,
            Timeout::Infinite,
        )
        .map_err(|failure| {
            format!(
                "cancel-first submission failed: {:?}",
                failure_error(&failure)
            )
        })?;
    let operation = (|| {
        if backend.poll_event(&first) != Ok(EventState::Pending) {
            return Err("cancel-first event was not pending".into());
        }
        backend
            .cancel_event(&first)
            .map_err(|error| format!("advertised cancellation failed: {error:?}"))?;
        expect_stable_state(backend, &first, EventState::Cancelled)?;
        expect_backend_error(
            backend.cancel_event(&first),
            BackendError::Busy,
            "repeat cancellation after cancellation won",
        )?;
        release_event(backend, first)?;

        let second_bindings = [valid_binding(target, &resources.buffer)?];
        let second = backend
            .submit(
                &resources.queue,
                &resources.program,
                &second_bindings,
                Timeout::Infinite,
            )
            .map_err(|failure| {
                format!(
                    "completion-first submission failed: {:?}",
                    failure_error(&failure)
                )
            })?;
        hooks
            .complete_event(backend, &second)
            .map_err(|error| format!("completion hook failed: {error:?}"))?;
        expect_stable_state(backend, &second, EventState::Complete)?;
        expect_backend_error(
            backend.cancel_event(&second),
            BackendError::Busy,
            "cancellation after completion won",
        )?;
        release_event(backend, second)
    })();
    merge(operation, resources.release(backend))
}

struct StandardResources<A: Accelerator> {
    context: A::Context,
    buffer: A::Buffer,
    program: A::Program,
    queue: A::Queue,
}

impl<A: Accelerator> StandardResources<A> {
    fn create(backend: &A, target: &TargetDescription) -> Result<Self, String> {
        ensure_target_fits(backend, target)?;
        let context = create_context(backend)?;
        let desc = target_buffer_desc(target, target.binding().domain())?;
        let allocation = backend
            .allocate_buffer(&context, desc)
            .map_err(|error| format!("target buffer allocation failed: {error:?}"))?;
        device_info(backend)?
            .validate_buffer_info(desc, allocation.info())
            .map_err(|error| format!("target allocation metadata is dishonest: {error:?}"))?;
        let (mut buffer, _) = allocation.into_parts();
        let initial = SliceSource(target.binding().initial());
        backend
            .write_buffer(&mut buffer, 0, &initial)
            .map_err(|error| format!("target buffer initialization failed: {error:?}"))?;
        let program = load_target(backend, &context, target)?;
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .map_err(|error| format!("target queue creation failed: {error:?}"))?;
        Ok(Self {
            context,
            buffer,
            program,
            queue,
        })
    }

    fn release(self, backend: &A) -> CaseCheck {
        let mut result = release_queue(backend, self.queue);
        result = merge(result, release_program(backend, self.program));
        result = merge(result, release_buffer(backend, self.buffer));
        merge(result, release_context(backend, self.context))
    }
}

fn complete_event<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
    buffer: &A::Buffer,
    event: A::Event,
) -> CaseCheck {
    hooks
        .complete_event(backend, &event)
        .map_err(|error| format!("completion hook failed: {error:?}"))?;
    expect_stable_state(backend, &event, EventState::Complete)?;
    let mut output = vec![0; target.binding().expected().len()];
    let mut sink = SliceSink(output.as_mut_slice());
    backend
        .read_buffer(buffer, 0, &mut sink)
        .map_err(|error| format!("completed output read failed: {error:?}"))?;
    if output != target.binding().expected() {
        return Err(format!(
            "completed output mismatch: expected {:?}, got {output:?}",
            target.binding().expected()
        ));
    }
    release_event(backend, event)
}

fn settle_timed_event<A: Accelerator, H: ConformanceHooks<A>>(
    backend: &A,
    target: &TargetDescription,
    hooks: &H,
    buffer: &A::Buffer,
    event: A::Event,
) -> CaseCheck {
    let initial = backend
        .poll_event(&event)
        .map_err(|error| format!("timed event poll failed: {error:?}"))?;
    if initial == EventState::Pending {
        match hooks.complete_event(backend, &event) {
            Ok(()) | Err(BackendError::Busy | BackendError::DeadlineExpired) => {}
            Err(error) => return Err(format!("timed-event progress hook failed: {error:?}")),
        }
    }
    let terminal = backend
        .poll_event(&event)
        .map_err(|error| format!("timed terminal poll failed: {error:?}"))?;
    let observed = match terminal {
        EventState::Complete => {
            expect_stable_state(backend, &event, EventState::Complete)?;
            let mut output = vec![0; target.binding().expected().len()];
            let mut sink = SliceSink(output.as_mut_slice());
            backend
                .read_buffer(buffer, 0, &mut sink)
                .map_err(|error| format!("timed output read failed: {error:?}"))?;
            if output != target.binding().expected() {
                return Err(format!(
                    "timed completed output mismatch: expected {:?}, got {output:?}",
                    target.binding().expected()
                ));
            }
            Ok(())
        }
        EventState::Failed(BackendError::DeadlineExpired) => expect_stable_state(
            backend,
            &event,
            EventState::Failed(BackendError::DeadlineExpired),
        ),
        state => Err(format!("timed event reached unexpected state {state:?}")),
    };
    merge(observed, release_event(backend, event))
}

fn expect_stable_state<A: Accelerator>(
    backend: &A,
    event: &A::Event,
    expected: EventState,
) -> CaseCheck {
    for observation in 1..=2 {
        match backend.poll_event(event) {
            Ok(actual) if actual == expected => {}
            Ok(actual) => {
                return Err(format!(
                    "terminal observation {observation} changed from {expected:?} to {actual:?}"
                ));
            }
            Err(error) => {
                return Err(format!(
                    "terminal observation {observation} failed: {error:?}"
                ));
            }
        }
    }
    Ok(())
}

fn ensure_target_fits<A: Accelerator>(backend: &A, target: &TargetDescription) -> CaseCheck {
    let info = device_info(backend)?;
    if !info
        .capabilities
        .supports_memory_domain(target.binding().domain())
    {
        return Err(format!(
            "target requires unadvertised {:?} memory",
            target.binding().domain()
        ));
    }
    if target.binding().bytes() > info.limits.max_buffer_bytes {
        return Err("target binding exceeds max_buffer_bytes".into());
    }
    if target.program().payload().len() as u64 > info.limits.max_artifact_bytes {
        return Err("target artifact exceeds max_artifact_bytes".into());
    }
    Ok(())
}

fn device_info<A: Accelerator>(backend: &A) -> Result<DeviceInfo, String> {
    backend
        .device_info()
        .map_err(|error| format!("device discovery failed: {error:?}"))
}

fn create_context<A: Accelerator>(backend: &A) -> Result<A::Context, String> {
    backend
        .create_context(ContextDesc::default())
        .map_err(|error| format!("context creation failed: {error:?}"))
}

fn target_buffer_desc(
    target: &TargetDescription,
    domain: MemoryDomain,
) -> Result<BufferDesc, String> {
    BufferDesc::new(
        target.binding().bytes(),
        target.binding().alignment(),
        domain,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT
            | BufferUsage::MUTABLE_STATE,
    )
    .map_err(|error| format!("invalid target buffer descriptor: {error:?}"))
}

fn load_target<A: Accelerator>(
    backend: &A,
    context: &A::Context,
    target: &TargetDescription,
) -> Result<A::Program, String> {
    let payload = SliceSource(target.program().payload());
    let artifact = ArtifactRef {
        format: target.program().format(),
        target: target.program().target(),
        payload: &payload,
        resident_bytes: target.program().resident_bytes(),
    };
    backend
        .load_program(context, artifact)
        .map_err(|error| format!("target program load failed: {error:?}"))
}

fn valid_binding<'a, B>(
    target: &TargetDescription,
    buffer: &'a B,
) -> Result<BindingRef<'a, B>, String> {
    Ok(BindingRef {
        slot: target.binding().slot(),
        buffer,
        range: BufferRange::new(0, target.binding().bytes())
            .map_err(|error| format!("invalid target binding range: {error:?}"))?,
        access: target.binding().access(),
    })
}

const fn different_access(access: AccessMode) -> AccessMode {
    match access {
        AccessMode::Read => AccessMode::Write,
        AccessMode::Write => AccessMode::Read,
        AccessMode::ReadWrite => AccessMode::Read,
    }
}

fn expect_backend_error<T>(
    result: Result<T, BackendError>,
    expected: BackendError,
    operation: &str,
) -> CaseCheck {
    match result {
        Err(actual) if actual == expected => Ok(()),
        Err(actual) => Err(format!(
            "{operation} returned {actual:?}, expected {expected:?}"
        )),
        Ok(_) => Err(format!("{operation} unexpectedly succeeded")),
    }
}

fn expect_submit_rejected<E>(
    result: Result<E, SubmitFailure<E>>,
    expected: BackendError,
    operation: &str,
) -> CaseCheck {
    match result {
        Err(SubmitFailure::Rejected(actual)) if actual == expected => Ok(()),
        Err(SubmitFailure::Rejected(actual)) => Err(format!(
            "{operation} returned {actual:?}, expected {expected:?}"
        )),
        Err(SubmitFailure::Indeterminate { error, .. }) => Err(format!(
            "{operation} became indeterminate instead of rejected: {error:?}"
        )),
        Ok(_) => Err(format!("{operation} was accepted")),
    }
}

const fn failure_error<E>(failure: &SubmitFailure<E>) -> BackendError {
    match failure {
        SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => *error,
    }
}

fn release_context<A: Accelerator>(backend: &A, context: A::Context) -> CaseCheck {
    retry_release("context", context, |resource| {
        backend.destroy_context(resource)
    })
}

fn release_buffer<A: Accelerator>(backend: &A, buffer: A::Buffer) -> CaseCheck {
    retry_release("buffer", buffer, |resource| backend.free_buffer(resource))
}

fn release_program<A: Accelerator>(backend: &A, program: A::Program) -> CaseCheck {
    retry_release("program", program, |resource| {
        backend.unload_program(resource)
    })
}

fn release_queue<A: Accelerator>(backend: &A, queue: A::Queue) -> CaseCheck {
    retry_release("execution queue", queue, |resource| {
        backend.destroy_queue(resource)
    })
}

fn release_event<A: Accelerator>(backend: &A, event: A::Event) -> CaseCheck {
    retry_release("event", event, |resource| backend.destroy_event(resource))
}

fn retry_release<R>(
    kind: &str,
    resource: R,
    mut release: impl FnMut(R) -> Result<(), ReleaseFailure<R>>,
) -> CaseCheck {
    match release(resource) {
        Ok(()) => Ok(()),
        Err(ReleaseFailure::Rejected { resource, .. }) => match release(resource) {
            Ok(()) => Ok(()),
            Err(ReleaseFailure::Rejected { error, .. }) => {
                Err(format!("{kind} release remained rejected: {error:?}"))
            }
            Err(ReleaseFailure::Indeterminate { error }) => Err(format!(
                "{kind} release became indeterminate on retry: {error:?}"
            )),
        },
        Err(ReleaseFailure::Indeterminate { error }) => {
            Err(format!("{kind} release became indeterminate: {error:?}"))
        }
    }
}

fn merge(first: CaseCheck, second: CaseCheck) -> CaseCheck {
    match (first, second) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(first), Err(second)) => Err(format!("{first}; {second}")),
    }
}

#[derive(Debug)]
struct LengthOnlySource(u64);

impl ByteSource for LengthOnlySource {
    fn len(&self) -> u64 {
        self.0
    }

    fn read_at(&self, _offset: u64, _target: &mut [u8]) -> Result<(), BackendError> {
        Err(BackendError::OutOfBounds)
    }
}

#[derive(Debug)]
struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ByteSource::read_at(self.0, offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
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

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct SplitSource<'a> {
    first: &'a [u8],
    second: &'a [u8],
}

impl ByteSource for SplitSource<'_> {
    fn len(&self) -> u64 {
        (self.first.len() + self.second.len()) as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        copy_from_segments(self.first, self.second, offset, target)
    }
}

#[derive(Debug)]
struct SplitSink<'a> {
    first: &'a mut [u8],
    second: &'a mut [u8],
}

impl ByteSink for SplitSink<'_> {
    fn len(&self) -> u64 {
        (self.first.len() + self.second.len()) as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        copy_into_segments(self.first, self.second, offset, source)
    }
}

fn copy_from_segments(
    first: &[u8],
    second: &[u8],
    offset: u64,
    target: &mut [u8],
) -> Result<(), BackendError> {
    let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
    let end = start
        .checked_add(target.len())
        .filter(|end| *end <= first.len() + second.len())
        .ok_or(BackendError::OutOfBounds)?;
    let first_end = end.min(first.len());
    let first_start = start.min(first.len());
    let first_bytes = first_end.saturating_sub(first_start);
    target[..first_bytes].copy_from_slice(&first[first_start..first_end]);
    if first_bytes < target.len() {
        let second_start = start.saturating_sub(first.len());
        let second_end = second_start + target.len() - first_bytes;
        target[first_bytes..].copy_from_slice(&second[second_start..second_end]);
    }
    Ok(())
}

fn copy_into_segments(
    first: &mut [u8],
    second: &mut [u8],
    offset: u64,
    source: &[u8],
) -> Result<(), BackendError> {
    let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
    let end = start
        .checked_add(source.len())
        .filter(|end| *end <= first.len() + second.len())
        .ok_or(BackendError::OutOfBounds)?;
    let first_end = end.min(first.len());
    let first_start = start.min(first.len());
    let first_bytes = first_end.saturating_sub(first_start);
    first[first_start..first_end].copy_from_slice(&source[..first_bytes]);
    if first_bytes < source.len() {
        let second_start = start.saturating_sub(first.len());
        let second_end = second_start + source.len() - first_bytes;
        second[second_start..second_end].copy_from_slice(&source[first_bytes..]);
    }
    Ok(())
}

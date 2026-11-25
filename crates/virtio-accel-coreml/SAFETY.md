# Unsafe-code audit

This crate is the single host-native exception to the portable workspace's `forbid(unsafe_code)`
rule. Unsafe Rust is confined to `src/macos.rs` and has three responsibilities:

1. declaring and calling the audited C ABI in `native/coreml_bridge.h`;
2. owning page-aligned allocations made through `std::alloc`; and
3. exposing those allocations as slices only while the `Accelerator` contract supplies the
   required exclusive or terminal-event access.

The Objective-C bridge uses ARC for Core ML/Foundation objects. Native model pointers cross the ABI
with one retained reference. Native event pointers use an atomic two-reference scheme: one reference
belongs to Rust and one to either Core ML's completion block or the bridge's asynchronous exact-copy
block, so dropping a Rust event early cannot free memory still used by either callback. The submitted
backing guards are boxed and owned by that native block through a C callback, independently of the
Rust event handle; the callback drops them exactly once before publishing a terminal state.
Completion publishes buffer writes before its release-store to the event status; polling uses an
acquire-load before Rust reads output bytes.

`AlignedAllocation` owns exactly one `std::alloc::Layout`. Its pointer is non-null, is deallocated
with the same layout, and is kept alive by completion-owned `Arc` clones. `CoreMlBuffer` is
deliberately neither `Send` nor `Sync`. Its atomic in-flight state admits multiple native readers or
one native writer, rejects host transfers while either mode is active, and rejects every conflicting
submission. Multiple bindings from one event to the same allocation are collapsed to the strongest
required access before acquiring a guard. The completion callback drops its backing guards before
publishing the terminal event state, so a successful acquire-poll makes the output bytes available
to the next host transfer. `CoreMlBuffer::with_bytes` lends a slice only inside a
synchronous callback after checking transfer usage, completion, and the logical
byte range. An `EventBacking` read guard retains the allocation and excludes
reentrant native writers for the entire callback, including unwinding. The
callback's result cannot borrow the temporary slice. No raw pointer or unguarded
slice is returned. A
direct trait user can drop handles early without causing a dangling native pointer because the
completion-owned `Arc`s outlive prediction.

Core ML receives `MLMultiArray` objects with a no-op deallocator because Rust owns the backing. The
bridge validates scalar alignment before exposing a bound pointer and verifies the output data
pointer, element type, shape, and strides before reporting completion. A different Objective-C
wrapper over the same storage is safe; declining the exact proposed output storage is an execution
failure.

The model-loading ABI also takes a scalar `va_coreml_compute_units` tag. Rust's closed
`CoreMlComputeUnits` enum maps CPU/ANE, CPU/GPU and All to explicit bridge values.
The private CPUOnly value is used for FP32 NeuralNetwork graphs under every policy.
GPU and ANE may narrow untyped NeuralNetwork intermediates. This preserves FP32 arithmetic without expanding the
selected policy. Discovery checks the required compute-device classes (both for All).
The policy is immutable per backend and mapped at each TOSA model load, including
in-memory compilation. It changes placement permissions only, not pointer ownership or
event lifetimes. The legacy host-path constructor retains its ANE target identity and
policy. The bridge rejects invalid tags and disables low-precision GPU accumulation.

## Compressed constants

Constant INT4/INT8 and palette lowering uses safe Rust and the existing owned ML Program
byte buffer. Packed MIL types never become boundary tensors. No new FFI function, buffer
ownership rule or submission-time conversion is introduced. The optional compute-plan
diagnostic is a standalone Objective-C program outside the adapter's FFI boundary.

## Completion notification

Prediction and exact-copy submissions transfer one boxed `CompletionContext` to
the native completion block. Its first callback clears buffer backing guards;
the bridge then release-publishes terminal status. A second callback consumes the
box, signals its `Arc<CompletionNotification>`, and catches waker unwinding before
returning across the ABI. The public event owns another Arc, so registration can
race either callback or occur after completion. Dropping the public event early
does not invalidate callback state. Failed admission transfers no context and
Rust reclaims the box. Notification never substitutes for authoritative polling.

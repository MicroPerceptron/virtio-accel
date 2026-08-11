# Unsafe-code audit

This crate is the single host-native exception to the portable workspace's `forbid(unsafe_code)`
rule. Unsafe Rust is confined to `src/macos.rs` and has three responsibilities:

1. declaring and calling the audited C ABI in `native/coreml_bridge.h`;
2. owning page-aligned allocations made through `std::alloc`; and
3. exposing those allocations as slices only while the `Accelerator` contract supplies the
   required exclusive or terminal-event access.

The Objective-C bridge uses ARC for Core ML/Foundation objects. Native model pointers cross the ABI
with one retained reference. Native event pointers use an atomic two-reference scheme: one reference
belongs to Rust and one to Core ML's completion block, so dropping a Rust event early cannot free
memory still used by the callback. The submitted backing guards are boxed and owned by that native
completion block through a C callback, independently of the Rust event handle; the callback drops
them exactly once before publishing a terminal state. Completion publishes buffer writes before its
release-store to the event status; polling uses an acquire-load before Rust reads output bytes.

`AlignedAllocation` owns exactly one `std::alloc::Layout`. Its pointer is non-null, is deallocated
with the same layout, and is kept alive by completion-owned `Arc` clones. `CoreMlBuffer` is
deliberately neither `Send` nor `Sync`, and an atomic in-flight gate rejects host transfers and a
second submission until Core ML has stopped accessing the allocation. The completion callback drops
its backing guards before publishing the terminal event state, so a successful acquire-poll makes
the output bytes available to the next host transfer. The backend never creates public slices or
raw-pointer accessors. A direct trait user can drop handles early without causing a dangling native
pointer because the completion-owned `Arc`s outlive prediction.

Core ML receives `MLMultiArray` objects with a no-op deallocator because Rust owns the backing. The
bridge verifies output object identity before reporting completion, making it an execution failure
if Core ML declines the proposed direct output backing.

# 17. Immutable shared OpenVINO inputs and borrowed host output

Status: implemented, host-only API addition. No accepted/emitted protocol bytes,
wire capabilities, resource charges or wire ownership rules change.

The OpenVINO backend already wraps page-aligned host allocations directly in
native tensors. Local heterogeneous callers nevertheless had to upload a separate
copy of identical inputs into each backend context and copy every output back.

Add `SharedInput::initialize` to fill owned aligned storage once before publishing
it as immutable, cloneable input. `import_shared_input` creates context-local
buffer handles over the same Arc, with fixed read-only usage. The existing event
guards retain the allocation until native work is quiescent. Caller handles may
be dropped, retained for CPU readers, or sent to another device-owning thread.

Add `OpenVinoBuffer::with_bytes` for closure-scoped host reads, holding an existing
shared backing guard so reentrant native writes cannot invalidate the borrow.
This avoids a readback copy without exposing a mutable or raw-pointer interface.
The safety audit covers initialization, aliasing, reentrancy, panic and release.

This does not import arbitrary external pointers, mutable cross-device memory,
DMA-BUFs or strided output views. It does not claim that a vendor plugin never
stages host data internally. The explicit transfer counter measures copies through
write_buffer/read_buffer; application initialization/packing remains separate.
Shared physical RAM alone is not proof of device-level zero-copy execution.

# Unsafe-code audit

This crate will become a host-native exception to the portable workspace's `forbid(unsafe_code)`
rule, on the same terms as the other host backend adapters — with one structural difference from
the hand-written FFI precedents: raw Vulkan declarations come from the pinned `ash` crate
(ADR 0002), so the audit pins the `ash` version and the entry points actually used rather than
re-declaring a subset in-tree.

**Scaffold state.** The native modules do not exist yet. This build compiles **no `unsafe` at
all** — only the always-compiled admission constants (`src/lower.rs`) and a placeholder — so the
crate root still forbids unsafe outright. The full audit is authored with the FFI and lifecycle
ticket (ticket 8 of the
[Vulkan wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/154)), and the
audited exception is registered in `ci/check-release-policy.py` in the same change. Its planned
structure:

1. **Boundary statement** — safe admission constants (`lower.rs`) and safe residency accounting;
   unsafe confined to the lifecycle/native modules built over `ash`'s generated declarations.
2. **`ash` pin and used entry points** — the pinned `ash` version, the `Vk*` functions the crate
   calls, and the invariants the crate owns around them (one owner plus `Drop` per handle;
   `VkResult` checked before out-parameters are trusted).
3. **Loader and handle lifetime** — one `ash` `Entry` per process; instance, device, queue,
   buffer, memory, pipeline, command buffer, fence, and descriptor handles each with one Rust
   owner and exactly-once destruction; teardown ordered events → programs → buffers → queues →
   context → device.
4. **Buffers and mappings** — persistent mapped access bounded by the allocation, flush and
   invalidate discipline for non-coherent memory types, and no submission-time staging: binding at
   an offset copies nothing by construction.
5. **Device loss** — `VK_ERROR_DEVICE_LOST` maps to instance poisoning and whole-instance discard;
   after poisoning, entry points are never re-entered except destruction, and destruction errors
   are latched identically.
6. **Concurrency** — the bounded preallocated pool member owned by the event; `vkGetFenceStatus`
   is a read-only status query and needs no worker thread (to be proven in ticket 6, not assumed).
7. **Audited unsafe operations** — every `unsafe` block carries a local `SAFETY:` comment; the
   native test inventory is listed here.

# Unsafe-code audit

This crate is a host-native exception to the portable workspace's `forbid(unsafe_code)` rule, on
the same terms as the other host backend adapters, with one structural difference: the raw Vulkan
declarations come from the pinned `ash` crate (ADR 0002), so this audit pins the `ash` version and
the entry points actually used rather than re-declaring a subset in-tree. Unsafe Rust is confined to
the `va_vulkan` build configuration — `src/native.rs` — and the crate root carries
`cfg_attr(not(va_vulkan), forbid(unsafe_code))`. Builds forced to the placeholder compile no
`unsafe` at all: only admission (`src/lower.rs`), the SPIR-V assembler (`src/shader.rs`), and a
placeholder.

Scope: loader and instance lifetime, device and queue creation, dedicated buffer allocations with
persistent mappings, per-program arena allocations, pipeline creation from the crate-authored
shaders, the per-context submission ring, nonblocking fence polling, and blocking staging copies
for device-local memory and arena constants.

## `ash` pin and entry points used

`ash 0.38.0` (`+1.3.281`), features `loaded` and `std` only: the platform Vulkan loader is opened at
run time through `libloading` (ISC); no Vulkan-Headers, SDK, or link-time dependency exists. Every
raw call is an `unsafe fn` on `ash::Entry`, `ash::Instance`, or `ash::Device`, and every call site
carries a local `SAFETY:` comment. The entry points this crate calls, and nothing else:

| Area | Entry points |
| --- | --- |
| Loader and instance | `vkEnumerateInstanceVersion`, `vkCreateInstance`, `vkDestroyInstance`, `vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties2`, `vkGetPhysicalDeviceFeatures2`, `vkGetPhysicalDeviceQueueFamilyProperties`, `vkGetPhysicalDeviceMemoryProperties` |
| Device | `vkCreateDevice`, `vkDestroyDevice`, `vkGetDeviceQueue`, `vkDeviceWaitIdle`, `vkCreateDescriptorSetLayout`, `vkDestroyDescriptorSetLayout`, `vkCreatePipelineLayout`, `vkDestroyPipelineLayout` |
| Buffers | `vkCreateBuffer`, `vkDestroyBuffer`, `vkGetBufferMemoryRequirements`, `vkAllocateMemory`, `vkFreeMemory`, `vkBindBufferMemory`, `vkMapMemory`, `vkUnmapMemory`, `vkGetBufferDeviceAddress` |
| Programs | `vkCreateShaderModule`, `vkDestroyShaderModule`, `vkCreateComputePipelines`, `vkDestroyPipeline`, `vkCreatePipelineCache`, `vkDestroyPipelineCache` |
| Contexts | `vkCreateCommandPool`, `vkDestroyCommandPool`, `vkAllocateCommandBuffers`, `vkCreateDescriptorPool`, `vkDestroyDescriptorPool`, `vkAllocateDescriptorSets`, `vkCreateFence`, `vkDestroyFence` |
| Submission | `vkUpdateDescriptorSets`, `vkResetFences`, `vkBeginCommandBuffer`, `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdDispatch`, `vkCmdCopyBuffer`, `vkCmdPipelineBarrier2`, `vkEndCommandBuffer`, `vkQueueSubmit2` |
| Completion | `vkGetFenceStatus`, `vkWaitForFences` |

All are Vulkan 1.0–1.3 core; no extension is enabled (ADR 0005). Two optional features are enabled
at device creation when the probe reports them: `synchronization2` (mandatory; a device without it
is not enumerated) and `bufferDeviceAddress` (used only to measure allocation alignment).

Every `VkResult` is checked before an out-value is trusted: `ash` returns `Result<T, vk::Result>`,
and the crate never reads a handle or pointer from an `Err`. Unmapped result codes surface as
`BackendError::External` in the stable `"VULK"` domain (`0x5655_4c4b`).

## Loader, instance, and handle lifetime

- The loader is opened once per process in a `OnceLock` (`entry`); `ash::Entry` is reference
  counted, so every backend instance shares the one library handle.
- Each `VulkanAccelerator` owns one `Shared`: one `VkInstance`, one `VkDevice`, the device's compute
  queue, and the backend-wide descriptor-set and pipeline layouts. `Shared::drop` waits for the
  device to idle, destroys the layouts and device, then the instance field destroys the instance.
  Every child handle holds an `Rc<Shared>` (through its context), so `Shared` cannot drop before
  any object created from its device.
- Each handle type has exactly one Rust owner and one `Drop`, destroying exactly once:
  `ContextInner` (command pool, descriptor pool, ring fences, transfer fence),
  `VulkanBuffer` (mapping, buffer, memory), `VulkanProgram` (pipeline). Command buffers and
  descriptor sets are freed with their pools. `VulkanQueue` owns no Vulkan object.
- Parent-before-child destruction is refused, not tolerated: `destroy_context` returns
  `Rejected(Busy)` while any child holds the context's `Rc`; `free_buffer` and `unload_program`
  return `Rejected(Busy)` while an event references them (in-flight gate, program in-flight
  count). If a caller violates the contract and drops such a handle anyway, its `Drop` first waits
  for the device to idle (`vkDeviceWaitIdle`) so no memory a submission may address is freed.
- Handles are `!Send` and `!Sync` (`Rc` inside). `VkQueue` and `VkCommandPool` are externally
  synchronized objects; thread affinity discharges that requirement without locks.

## Buffers and mappings

Every buffer is one `VkBuffer` bound at offset 0 of one dedicated `VkDeviceMemory`
(`RawAllocation`), created with `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST` usage and a size
rounded up to a whole 32-bit word: byte-storage (`BOOL`) tensors are read and atomically written by
word, so the word containing a tensor's last byte must exist inside the allocation even when the
logical size (`desc.bytes()`, which every transfer and range check still uses) is not a multiple of
four. The memory
type comes from the ADR 0005 memory-domain map chosen at device open against the `memoryTypeBits`
of a probe buffer with the same usage (ANV exposes types buffers may not use): `Host` and `Shared`
require `HOST_COHERENT`, so no flush or invalidate is ever needed and none is issued; `Device`
allocations are never mapped. Host-visible allocations are mapped once (`vkMapMemory`, whole size)
and stay mapped for the buffer's lifetime: a persistent mapping. Explicit transfers copy through
that mapping only after the in-flight gate proves no submission references the buffer, and the
mapped range is validated against the logical buffer size before any pointer arithmetic. Alignment
is measured, not assumed: the mapped pointer's alignment and, when `bufferDeviceAddress` is
available, the buffer's device address alignment; a request the measurement cannot satisfy is
released and rejected as `ResourceLimit`.

Device-local transfers stage through a bounded (4 MiB) host-coherent `Staging` allocation and a
per-context transfer command buffer and fence: record `vkCmdCopyBuffer`, submit, wait for the fence
(30 s bound, after which the device is treated as lost), then copy through the staging mapping.
Every copy is followed by a memory barrier naming its consumers, because submissions carry no
implicit memory dependency between one another: a read into staging adds `COPY → HOST/HOST_READ`;
a write into a device-local buffer adds `COPY/TRANSFER_WRITE → COMPUTE_SHADER|COPY` with storage
and transfer read/write access so later dispatches and staging copies observe it. The host's own
writes into the staging mapping are ordered by the implicit host-write guarantee at
`vkQueueSubmit2`. The staging allocation is destroyed before the call returns.

## Program arenas

A program whose graph carries `CONST` tensors or intermediates owns one `Arena`: a dedicated,
never-mapped `RawAllocation` in device-local memory when the device has any (else the host type),
sized by the lowering's lifetime-packed layout and bounded by `maxStorageBufferRange`. Constants
are uploaded once at `load_program` through the same staging path as device-local transfers, with
the same `COPY → COMPUTE_SHADER|COPY` barrier so later submissions read them. The arena is destroyed with its `VulkanProgram` (after `vkDeviceWaitIdle`
if the contract was violated and submissions are still in flight), so no dispatch can address freed
memory. One arena per program is charged against the assumed `maxMemoryAllocationCount` alongside
the guest buffers: `16 × (190 + 64) + 1 < 4096`, enforced by a compile-time assertion.

## Submission ring and completion

Each context preallocates `RING_DEPTH` (`DeviceLimits.max_events_per_context`) triples of
(command buffer, fence, descriptor set) plus the transfer pair (ADR 0006). The descriptor set
layout is one binding: an array of storage buffers sized per device (bound slots plus the arena).
`submit` validates every binding against the program plan — exact tensor bytes, a start aligned to
`max(4, minStorageBufferOffsetAlignment)`, the plan's access mode, no aliasing with a written slot —
claims one free slot, acquires the buffer in-flight gates (shared for read-only bindings, exclusive
for writes), writes the whole descriptor array (slots in slot order, the arena, then a valid filler
for every element the program never addresses; byte-storage ranges rounded up to the containing
word), resets the fence, records (bind set; per dispatch an optional
`COMPUTE_SHADER/SHADER_STORAGE_WRITE → COMPUTE_SHADER/SHADER_STORAGE_READ|WRITE` barrier, bind
pipeline, dispatch; a final `COMPUTE_SHADER/SHADER_STORAGE_WRITE → HOST|COPY` barrier with host-read
and transfer read/write access, so the host mapping and any later staging copy of a device-local
output observe the results), and
calls `vkQueueSubmit2` with the slot's fence. Which dispatches need a barrier is decided by the
lowering from the plan's read/write sets, not at record time. `vkQueueSubmit2` success is the
admission boundary: any failure before it releases the slot and gates and rejects; an
out-of-memory result from `vkQueueSubmit2` itself is specified to leave every resource untouched
and is also a rejection; `VK_ERROR_DEVICE_LOST` returns `Indeterminate` with an event already
latched `Failed(DeviceLost)`.

`poll_event` is one `vkGetFenceStatus` call: a read-only status query, so no worker thread exists.
On the first terminal observation the gates are released and the program's in-flight count
decremented strictly before the latched state becomes observable, so a caller that sees a terminal
state can transfer buffer bytes immediately. `destroy_event` returns the slot to the ring only for
a terminal event; a pending event is returned `Rejected(Busy)`. Dropping a pending event outside
`destroy_event` (a contract violation) blocks on `vkWaitForFences` before the slot is returned, so
a recorded command buffer is never re-recorded while it may still execute.

## Device loss

`VK_ERROR_DEVICE_LOST` from any entry point, and a transfer wait that times out, set the sticky
`poisoned` flag on `Shared`. After poisoning, creation, transfer, and submission entry points return
`DeviceLost` without re-entering the driver; polling latches `Failed(DeviceLost)`; destruction still
runs (its errors are latched identically) so the instance can be discarded whole (ADR 0006).

## Shaders

The only SPIR-V the driver ever receives is assembled by this crate (`src/shader.rs`): guest bytes
never reach the shader compiler. A TOSA artifact selects kernel variants (`KernelKey`) and supplies
validated geometry — operand array indices and word offsets, element counts, dims and strides,
reduction extents, pooling windows, clamp bounds — through specialization constants; the modules
themselves are fixed templates whose only parameters are device properties chosen when the device
is opened (workgroup size, MATMUL tile, descriptor-array length). Modules are assembled once per
instance and cached; the lowering re-derives every index computation from the declared tensor
shapes and rejects any disagreement before a plan exists, because the kernels rely on that
geometry rather than on robust buffer access. Byte-storage tensors are written only with
`OpAtomicAnd`/`OpAtomicOr` on their containing word, so no kernel modifies a byte outside its
tensor. Every float arithmetic result is `NoContraction`. The crate's unit tests walk every
variant for structural well-formedness, and every variant executes on lavapipe in CI.

## Evidence

`tests/vulkan.rs` runs on every enumerated device: full lifecycle in every advertised memory
domain, every case of the shared FP32 operator corpus in every advertised memory domain, the
three-operator arena graph as one submission, byte-storage outputs bound inside a larger buffer
with every neighbouring byte untouched, transcendental kernels within one ulp of binary64
references across the full finite range, tiled MATMUL bit-identical to the sequential reference at
ragged sizes, rank-4 broadcasting, the shared `IDENTITY_EDGES_FP32` corpus (bit-exact, including
NaN payloads and the subnormal), offset bindings inside larger buffers with untouched neighbors,
segmented staging
transfers to device-local memory, binding validation and finite-timeout rejection, overlapping
read-only bindings across sixteen in-flight submissions with `Busy` on the shared input, ring
exhaustion as `ResourceLimit`, parent-release refusal, and the standard conformance suite
(`virtio-accel-conformance::run`) with the accounting and copy-path diagnostics hooks in every
advertised domain. On 2026-09-03 the IDENTITY + MATMUL suite passed on an Intel Arc 140V (Lunar
Lake, Mesa 26.0.8 ANV, Vulkan 1.4.335) and on the same host's llvmpipe; on 2026-09-06 the full
suite for the broadened FP32 tier (ADR 0007) passed on the same Arc 140V and its llvmpipe (LLVM
21.1.8), and on Mesa lavapipe (25.2.8, LLVM 20.1.2) in CI.

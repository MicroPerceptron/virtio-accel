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
persistent mappings, pipeline creation from the checked-in shaders, the per-context submission
ring, nonblocking fence polling, and blocking staging copies for device-local memory.

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
| Programs | `vkCreateShaderModule`, `vkDestroyShaderModule`, `vkCreateComputePipelines`, `vkDestroyPipeline` |
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
(`RawAllocation`), created with `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST` usage. The memory
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
Reads add a `COPY → HOST` memory barrier; writes rely on the implicit host-write ordering guarantee
at `vkQueueSubmit2`. The staging allocation is destroyed before the call returns.

## Submission ring and completion

Each context preallocates `RING_DEPTH` (`DeviceLimits.max_events_per_context`) triples of
(command buffer, fence, descriptor set) plus the transfer pair (ADR 0006). `submit` validates every
binding against the program plan, claims one free slot, acquires the buffer in-flight gates (shared
for read-only bindings, exclusive for writes), updates the slot's descriptor set, resets the fence,
records (bind pipeline, bind set, dispatch, `COMPUTE_SHADER/SHADER_STORAGE_WRITE → HOST/HOST_READ`
barrier), and calls `vkQueueSubmit2` with the slot's fence. `vkQueueSubmit2` success is the
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
never reach the shader compiler. A TOSA artifact selects a kernel and supplies its validated element
count through a specialization constant; the module itself is fixed. Each kernel's disassembly is
listed beside its assembler, and the crate's tests verify the module's structure; the module was
additionally validated with naga's SPIR-V front end and executes bit-exactly on ANV and lavapipe.

## Evidence

`tests/vulkan.rs` runs on every enumerated device: full lifecycle in every advertised memory
domain, the shared `IDENTITY_EDGES_FP32` corpus (bit-exact, including NaN payloads and the
subnormal), offset bindings inside larger buffers with untouched neighbors, segmented staging
transfers to device-local memory, binding validation and finite-timeout rejection, overlapping
read-only bindings across sixteen in-flight submissions with `Busy` on the shared input, ring
exhaustion as `ResourceLimit`, parent-release refusal, and the standard conformance suite
(`virtio-accel-conformance::run`) with the accounting and copy-path diagnostics hooks in every
advertised domain. On 2026-09-03 the suite passed on an Intel Arc 140V (Lunar Lake, Mesa 26.0.8
ANV, Vulkan 1.4.335) and on the same host's llvmpipe.

# 13. Imported host memory and host gates: a host-side zero-copy path for local callers

- Status: accepted (implemented; measured on Intel Arc (Lunar Lake, Mesa ANV, Linux 7.0) on
  2026-09-19, the device suite passing on ANV and llvmpipe)
- Extends: ADR 0005 (memory domains; this adds a way in that allocates nothing), ADR 0006 (the
  submission ring; a submission may now also wait on a semaphore), ADR 0012 (the copies it left)
- Does not change: the wire protocol. External memory import/export and timeline fences stay
  deferred protocol features (CONTRIBUTING.md); nothing here is advertised or negotiated.

## Context

After ADR 0012 a caller's bytes reach the device through one copy: `write_buffer` into memory
the backend allocated. For a local caller that already owns the memory the bytes should live in,
that copy is the only thing left between storage and the kernel. The motivating caller is an
inference host whose weights stream from an NVMe drive into a pool of huge pages (kore's model:
devices share one DRAM, storage writes straight into a chunk every device maps, and completion
wakes the consumer). There, the drive's DMA can already land an expert in the pool with no page
cache and no copy; what was missing was the GPU addressing those same pages, and a way to start
the GPU's work the moment a read completes without a round trip through the thread that owns the
backend.

On the platforms measured, two parts of kore's model are not reachable from a Vulkan backend:
an NVMe completion is an MSI-X write the IOMMU remaps to a CPU interrupt vector, handled by the
kernel's driver, so a CPU always observes it first; and core Vulkan has no command that waits on a
memory value written by another device. The nearest equivalent is a host thread (the one that saw
the read complete) signalling a semaphore the queued work waits on.

## Decision

1. **`VulkanAccelerator::import_host_buffer` (unsafe).** Wraps a caller's page-aligned range of
   live host memory (anonymous or huge-page mappings) as a buffer through
   `VK_EXT_external_memory_host`: a `VkBuffer` created for host-allocation handles, bound to
   memory imported from the pointer in a host-coherent type the pointer admits (device-local
   first). The result is an ordinary `VulkanBuffer`: bound by submissions, gated while in flight,
   usable by the explicit transfers, released by `free_buffer`, which frees the import and never the
   caller's pages. `mapped` is the caller's pointer, not a `vkMapMemory` mapping, so nothing is
   unmapped. Only the `Host` and `Shared` domains may be imported: their contract is host-visible
   memory. Pointer and length must be multiples of `minImportedHostPointerAlignment`
   (`host_import_alignment`). The caller vouches that the memory outlives the buffer and every
   submission that bound it.

2. **The extension is enabled when offered, and only for import.** `VK_EXT_external_memory_host`
   is the first device extension this backend enables. It is probed per device and enabled when
   reported; nothing else depends on it, admission and kernels are unchanged, and a device without
   it behaves exactly as before (`import_host_buffer` returns `Unsupported`).

3. **Host gates.** `host_gate` creates a `VulkanHostGate`: a timeline semaphore (Vulkan 1.2
   `timelineSemaphore`, enabled when reported). `submit_after` is `submit` with the batch's first
   dispatch waiting for the gate to reach a value. `VulkanHostGate::signal` hands out a
   `VulkanGateSignal` that any thread may use to `raise` the gate; raises are serialized and
   monotonic (a value at or below the current one changes nothing), and a raise after the gate is
   dropped is a reported no-op. Dropping the gate raises it to the highest value any submission
   awaited, waits for the device to idle, then destroys the semaphore, so no queued work can wait
   forever and no raise can reach a destroyed semaphore.

4. **What a gate permits.** A gated submission holds its bindings from `submit_after`, as
   `submit` does, with one difference the gate exists for: the bytes of its input bindings may
   still be written until the gate is raised to its value, by host stores into an imported buffer
   or by a device's DMA the host has seen complete. A host semaphore signal orders every host
   operation before it ahead of the device's wait, so those bytes are visible to the first
   dispatch. Outputs, and inputs after the raise, follow `submit`'s rules.

## Evidence

Lunar Lake (Intel Arc 130V/140V), Mesa ANV, Linux 7.0, 1 GiB huge pages reserved at boot.
`streamed_experts` in the consuming project imports a ring of four 6 MiB slots carved from one
1 GiB page, queues an FP8 projection per expert behind a gate, and has an I/O thread read each
expert (`[1856, 2688]` E4M3, 5.0 MB) with `O_DIRECT` into its slot and raise the gate when the
read returns. 128 experts:

| Queued ahead | Experts/s | Weights through the GPU | Read per expert | Raise → done |
|---|---:|---:|---:|---:|
| 1 | 265 | 1.3 GB/s | 1.42 ms | 1.71 ms |
| 4 | 634 | 3.2 GB/s | 1.45 ms | 1.63 ms |

`explicit_transfer_bytes` stays 0 throughout: no byte is copied between the drive and the kernel.
Every eighth result matches the decoded weight on the CPU within 4.4e-4 of the peak. With four
submissions queued ahead, reads and projections overlap and the single reader (one read in
flight, about 3.5 GB/s) sets the pace. The projection itself — a device `CAST` of the FP8 weight
to FP16, then a `[1856, 2688] × [2688, 1]` `MATMUL` — takes about 1.6 ms, which is the next
limit once reads are issued in parallel (see Consequences).

## Consequences

- A local caller can place bytes where the device reads them with no copy: huge pages, a pool
  shared with other engines, a slot an NVMe read targets. The protocol path is unchanged; a guest
  still transfers.
- The gate removes the owner thread from the completion path, not the CPU: the drive's interrupt
  and the thread that raises the gate still run. Parity with a design where storage completion
  wakes the device directly needs a device that can wait on memory, or a drive whose queues the
  host process owns (a user-space driver with completion queues in shared memory); neither is a
  Vulkan-backend decision.
- The measured projection is the orientation checkpoints store (`[out, in]` weights times an
  activation column) with FP8 weights against FP16 activations. It pays a separate `CAST` pass and
  a MATMUL geometry sized for wide outputs; a skinny-`N` geometry and a widening-in-kernel mixed
  FP8 × FP16 MATMUL would each remove a pass. Both are kernel work, not decided here.
- Imported memory is host memory: on a discrete GPU the device reads it across the bus. Nothing
  here changes which domain a caller should choose for data the device reads repeatedly.

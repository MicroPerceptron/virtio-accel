# 12. Copies eliminated: mapped `Device` memory on unified devices, and boundary views

- Status: accepted (implemented; measured on Intel Arc (Panther Lake, Mesa 26.0.8 ANV, Vulkan
  1.4.335) on 2026-09-18, the device suite passing on ANV and llvmpipe and clean under the
  Khronos validation layer)
- Extends: ADR 0005 (the memory-domain map this refines for single-heap devices), ADR 0007
  (whose arena views this generalizes)
- Resolves: the two redundant copies left in the data path after ADRs 0010 and 0011

## Context

With the kernels at the memory system's rate, the question became what bytes still move that
need not. Two copies remained, both at the edge of the device, both invisible to the kernel
benchmark until it timed transfers and dispatch counts as well as kernels.

**`Device`-domain transfers on unified memory.** ADR 0005 mapped `Device` to a device-local
memory type reached only through staging: a host copy into a transient buffer, a GPU copy into
the allocation, a submission and a wait — sound on a discrete GPU, where device-local memory is
a different memory. Panther Lake has one heap. Every one of its memory types is `DEVICE_LOCAL`,
four of the five are also `HOST_VISIBLE | HOST_COHERENT`, and the plan scored the one that is
not host-visible highest for `Device`. Kernels read all of them at the same rate — 109.5 GB/s
copy, 0.34 ms FP8 GEMV and 0.86 ms FP16 1024³ MATMUL whether the buffers were `Device`,
`Shared` or `Host`. The staging bought nothing and cost a 64 MiB `write_buffer` 32.6 ms
(2 GB/s) against 2.6 ms (26 GB/s) for the same bytes into `Shared`; `read_buffer` 25.1 ms
against 2.5 ms. Program constants took the same staged path into the arena.

**`RESHAPE` and `IDENTITY` at the program boundary.** ADR 0007 made them views when both ends
live in the arena and a copy dispatch otherwise. A graph's inputs and outputs live in bound
slots, so the common shapes — an input reshaped for a `MATMUL`, a `MATMUL` result reshaped into
the output — each cost a full-tensor copy dispatch and, for the input side, an arena region the
size of the input.

## Decision

1. **A single-heap device maps its `Device` domain.** When the physical device reports exactly
   one memory heap, there is no second memory for `Device` to be local to, and `MemoryPlan`
   prefers a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type (host-cached when offered) for
   the domain. Allocations in it are persistently mapped like `Host` and `Shared` ones, so
   `write_buffer` and `read_buffer` are one copy, program arenas are written by a plain copy
   at `load_program`, and buffers report `HOST_VISIBLE` alongside `DEVICE_LOCAL`. Devices with
   more than one heap — every discrete GPU — keep ADR 0005's plan unchanged; the choice is made
   from the heap count, never from a vendor or device-type table.

2. **The staged plan stays available and tested.** `VulkanOptions::map_unified_device_memory`
   (default `true`) is the only option `VulkanAccelerator::with_device_options` takes today;
   `false` selects the discrete-GPU plan on any device. The staging path's test opens the
   backend that way, so a CI lane on lavapipe — one heap — still runs the code a discrete GPU
   depends on, and a second test records which plan the default took on each device.

3. **Views on both sides of the boundary.** A `RESHAPE` or `IDENTITY` whose result is an
   intermediate is a view over its source wherever the source lives: an arena region, whose
   lifetime the view extends, or a bound input slot, which is read-only for the whole
   submission. A program output produced by `RESHAPE` or `IDENTITY` from an intermediate that
   nothing else reads is that intermediate under another shape, so lowering aliases the
   intermediate to the output slot before lowering begins and the producer writes the output
   directly; the copy operator then finds source and destination identical and emits nothing.
   Inputs, outputs and constants are never aliased (their bytes are already where they must
   be), nor is an intermediate with a second consumer. The copy dispatch remains for the one
   case that needs it: an input reshaped straight into an output, which are different buffers.

## Evidence

Intel Arc (Panther Lake), 64 MiB, median of 10:

| Transfer | Staged (`Device`, before) | Mapped (`Device`, after) | `Shared` |
|---|---:|---:|---:|
| `write_buffer` | 32.6 ms, 2.1 GB/s | 2.59 ms, 25.9 GB/s | 2.59 ms, 26.0 GB/s |
| `read_buffer` | 25.1 ms, 2.7 GB/s | 2.47 ms, 27.1 GB/s | 2.48 ms, 27.1 GB/s |

Kernel throughput in the `Device` domain is unchanged (FP8 GEMV 0.34 ms). The graph
`[6] → RESHAPE [1,2,3] → MATMUL → [1,2,2] → RESHAPE [4]` loads as one dispatch with no arena and
computes the plain matmul's result on both devices; the shared movement corpus passes unchanged;
the staged-transfer test passes on both devices under the opt-in plan and both report the mapped
plan by default.

## Consequences

- On integrated GPUs, Apple silicon and software ICDs, a guest that chooses `Device` for its
  weights pays one copy to load them and none per submission, the same as `Shared`. The
  domain's contract — device-local, direct binding — holds; what changed is that the backend no
  longer pretends a second memory exists.
- Discrete GPUs are unaffected. Whether a host-visible device-local type (resizable BAR) should
  be preferred there is a separate question with a real cost (BAR reads are slow) and is not
  decided here.
- The bench times `write_buffer`/`read_buffer` per domain (`VIRTIO_ACCEL_VULKAN_BENCH_DOMAIN`)
  so a regression in either path is a number, not an inference.
- Graphs with boundary reshapes — most graphs authored for a fixed input layout — lose a
  dispatch and an arena region per reshape. The remaining data-movement dispatches are the
  ones that move data: `TRANSPOSE`, `REVERSE`, `CONCAT`, and a reshape from an input directly
  into an output.

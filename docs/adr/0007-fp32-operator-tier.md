# 7. FP32 operator tier: descriptor-array kernels, program arenas, and crate-owned numerics

- Status: accepted (implemented; verified on Mesa lavapipe in CI and, on 2026-09-06, on
  Intel Arc 140V (Lunar Lake, Mesa 26.0.8 ANV, Vulkan 1.4.335) — see `docs/performance.md`)
- Extends: ADR 0003 (checked-in shaders specialized by constants), ADR 0004 (FP32 base tier),
  ADR 0006 (execution model)
- Resolves: the operator-coverage half of wayfinder map #154 ticket 9 — the FP32 tier now admits
  the 42 operators the Core ML and OpenVINO backends share

## Context

ADR 0003 committed the Vulkan backend to crate-authored SPIR-V specialized at `load_program`, with
a falsification trigger: if an advertised operator needed structure that specialization constants
cannot express, the design would be reopened before operator work proceeded. Growing from two
operators to the shared FP32 set raised exactly that question three times over:

1. **Bindings.** A module fixes the `(set, binding)` of every buffer it touches. `CONCAT` takes
   any number of inputs, a multi-operator graph has intermediates no caller binds, and constants
   arrive inside the artifact. One module per binding layout would multiply variants without
   bound.
2. **Graphs.** Core ML and OpenVINO admit whole graphs. A single-operator admission would leave
   "supported" meaning something weaker for Vulkan than for the backends it is compared with.
3. **Numerics.** Vulkan bounds `sin`/`cos` only to an absolute 2⁻¹¹ inside an unspecified range,
   does not specify `tanh` at all, permits fused multiply-add unless a result is decorated, and
   leaves `FMax`/`FMin` undefined on NaN. The shared oracles compare across devices; a tier that
   inherits per-driver transcendental behaviour cannot honestly claim cross-device equivalence.

## Decision

1. **One descriptor, an array of storage buffers.** Set 0, binding 0 is an array of
   `{ uint words[]; }` blocks sized per device (`min(maxPerStageDescriptorStorageBuffers,
   maxDescriptorSetStorageBuffers, 17)`). Elements `0..slots` are the submission's bound slots in
   slot order; the next element is the program's arena; the rest are filled with a valid buffer.
   Each kernel operand is a pair of specialization constants — array index and base offset in
   words — so one module per kernel variant serves every binding layout, arena tensor, and
   `CONCAT` arity. Indexing a resource array by a specialization constant is a constant index in
   SPIR-V's terms, so no dynamic-indexing feature is required. `DeviceLimits.
   max_bindings_per_submission` is derived from the array length (elements minus the arena).
   Because a single array cannot be `NonWritable` per element, that read-only hint is forgone;
   the compiler still sees each specialized index.
2. **Whole-graph lowering into one command buffer.** The analysis' execution order becomes a
   list of dispatches. `CONST` tensors and intermediates are placed in one per-program arena
   allocation (device-local when the device has such memory) by a first-fit allocator over
   execution-position lifetimes, 256-byte aligned; constants are uploaded once at `load_program`
   through the staging path with a `TRANSFER → COMPUTE|COPY` barrier. `RESHAPE` and `IDENTITY` whose
   input already lives in the arena become views (no dispatch, lifetimes merged). Operators the
   analysis marks `DEAD` are not dispatched. A `COMPUTE_SHADER` storage-write → storage-read/write
   memory barrier precedes any dispatch that reads bytes an earlier dispatch wrote, or writes
   bytes an earlier dispatch touched — hazards are tracked by arena byte range, so a region
   whose bytes the packer hands to a later tensor still orders that tensor's writer after the
   region's last reader; `CONCAT` segments writing disjoint regions of one output
   need none. The lowering re-derives every shape, axis, permutation, and pooling window from the
   declared tensors and rejects disagreement, because the kernels address storage with that
   geometry and no robust-buffer-access mode is relied upon.
3. **Byte-storage tensors by word.** `BOOL` tensors (one byte per element) are read with word
   loads and written with `OpAtomicAnd`/`OpAtomicOr` on the containing word, so a kernel never
   modifies a byte it does not own — including the three bytes past an unaligned tail, which may
   belong to the caller. Every `VkBuffer` is sized to a whole word above its logical size so that
   word always exists, byte-storage descriptor ranges are rounded up to it, and bindings must
   start word-aligned (`max(4, minStorageBufferOffsetAlignment)`). Any nonzero input byte reads
   as true; outputs are canonical `0`/`1`.
4. **Crate-owned numerics.** Every floating-point arithmetic result carries `NoContraction`.
   `SIN`/`COS` use the Cephes three-part π/4 reduction below |x| = 8192 and a Payne–Hanek
   reduction (128-bit window of 2/π, 24×128-bit integer multiply built from 16-bit products,
   renormalized fraction) above it, with the Cephes minimax polynomials; `TANH` is the Cephes
   odd polynomial below 0.625 and `1 − 2/(e^{2|x|}+1)` above; `ERF` is the Maclaurin series
   through `x²¹` below |x| = 1 and `1 − erfc` from the Chebyshev-fitted rational form above.
   `EXP`, `LOG`, `POW`, and `RSQRT` use the `GLSL.std.450` built-ins, whose relative error
   Vulkan does bound; `POW` adds the IEEE special cases the built-in leaves undefined. NaN modes
   are explicit `OpIsNan` selects following the TOSA `apply_max_s`/`apply_min_s` pseudocode
   (`PROPAGATE` and `IGNORE` both implemented, so no `PROPAGATING_NAN` constraint is advertised);
   `ARGMAX` under `PROPAGATE` returns the index of the first NaN. `MAX_POOL2D` skips padded taps
   rather than substituting a value, so nonzero padding is admitted.
5. **Tiled `MATMUL`.** Square workgroup tiles (16, or 8 on a device whose
   `maxComputeWorkGroupInvocations` is only the specification minimum) staged through shared
   memory, accumulating in ascending `k` with the inner bound `min(tile, k − k₀)` so no padded
   zero product is added — bit-identical to the sequential loop, signed zeros included.
6. **Device tuning, not device code paths.** Workgroup size (256, or 128 at the specification
   minimum), MATMUL tile, and descriptor-array length are fixed per device from its limits when
   the backend opens it; they parameterize module assembly and are cached per instance alongside
   one `VkPipelineCache`. Grid-stride loops keep every 1-D dispatch inside
   `maxComputeWorkGroupCount` whatever the element count.
7. **Limits.** With one arena per program, the assumed `maxMemoryAllocationCount` of 4096
   (ADR 0005) is shared: `16 × (190 buffers + 64 programs) + 1 staging = 4065`. The arena is
   bounded by `maxStorageBufferRange` (128 MiB on lavapipe); a program needing more is rejected
   with `ResourceLimit`, as is one whose slots exceed the descriptor array.

## Consequences

- ADR 0003's falsification trigger did not fire: every admitted operator is expressed by
  specialization constants over a fixed set of crate-authored templates. The template parameters
  that do change instructions (workgroup size, tile, array length) are device properties, never
  guest data, so the security invariant — guest bytes never reach the driver's shader compiler —
  holds unchanged.
- The tier is validated by the shared FP32 operator corpus (35 new fixtures plus the three
  existing ones, oracles evaluated in binary64) and by kernel-level tests: transcendental ulp
  sweeps against binary64 references (worst case 1 ulp for sin/cos/tanh and 2 ulp for erf on
  both ANV and llvmpipe, including arguments up to `f32::MAX`), tiled-MATMUL bit identity at
  ragged sizes and batches, rank-4 broadcasting, and
  byte-tensor neighbour safety.
- Per-element atomics make predicate outputs slower than word stores; a packed fast path for
  contiguous `BOOL` kernels is the obvious follow-on. `SIN`/`COS` evaluate both reductions and
  select, trading throughput for a branch-free kernel.
- Real-GPU evidence: the full suite passed on Intel Arc 140V (Mesa 26.0.8 ANV) on 2026-09-06
  with the same worst-case transcendental errors as llvmpipe (1 ulp sin/cos/tanh, 2 ulp erf);
  MoltenVK is still owed. `docs/performance.md` carries the commands and the no-timing-claims
  note.

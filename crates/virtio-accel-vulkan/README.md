# virtio-accel-vulkan

A vendor-neutral Vulkan compute host backend for `virtio-accel`, executing device-neutral TOSA
1.0 programs on any conformant Vulkan 1.3 implementation (RADV, ANV, NVIDIA, Mali, or a software
ICD such as lavapipe) without leaking Vulkan types into the portable crates.

**Portability tier:** `host-native` — the pinned `ash` crate loads the platform's Vulkan loader
dynamically at run time on the enumerated host targets (Linux, Android, Windows, macOS); a
compile-only placeholder elsewhere. Unlike the SDK-probing backends, there is nothing to detect at
build time (ADR 0002 in `docs/adr/`).

## What executes today

- **The FP32 operator tier** (`VULKAN_TOSA_TARGET`, `VULKAN_TOSA_CAPABILITY`): static
  single-block TOSA 1.0 graphs over the 42 operators the Core ML and OpenVINO backends share —
  `ABS`, `CEIL`, `FLOOR`, `NEGATE`, `RECIPROCAL`, `RSQRT`, `EXP`, `LOG`, `SIN`, `COS`, `TANH`,
  `SIGMOID`, `ERF`, `CLAMP`, `ADD`, `SUB`, `MUL`, `POW`, `MAXIMUM`, `MINIMUM`, `EQUAL`,
  `GREATER`, `GREATER_EQUAL`, `LOGICAL_AND`/`OR`/`XOR`/`NOT`, `SELECT`, `REDUCE_SUM`/`MAX`/
  `MIN`/`PRODUCT`, `ARGMAX`, `MATMUL`, `MAX_POOL2D`, `CONCAT`, `RESHAPE`, `REVERSE`, `TRANSPOSE`,
  `IDENTITY`, `CONST`, and `CONST_SHAPE` — over FP32 tensors with `BOOL` (predicates, logic,
  selection) and `INT32` (`ARGMAX` results, data movement) auxiliaries, rank up to 6 with TOSA
  broadcasting. `MATMUL` and `NEGATE` admit zero zero-points only, `MUL` a zero shift, and
  `RESHAPE` a constant shape. Every shape, axis, permutation, and pooling window is re-derived at
   admission and checked against the declared tensors before any kernel is dispatched.
- **The FP16 operator tier** (`VULKAN_TOSA_FP16_CAPABILITY`, ADR 0008): the same 42 operators
  and graph envelope over binary16 tensors, advertised on every device the backend opens — the
  tier needs no device feature. Packed binary16 tensors are unpacked and widened to binary32 by
  crate-owned integer code, the float lanes evaluate in binary32 — the implementation choice
  TOSA 1.0 §1.10.3 names explicitly, and the correctly rounded binary16 result for
  `ADD`/`SUB`/`MUL` (their exact results fit the binary32 significand) — and results narrow back
  through crate-owned round-to-nearest-even code that produces subnormals on every device.
  `NEGATE`/`ABS` are integer sign masks on the packed lane; data movement copies the 16-bit
  lanes as integers, so `IDENTITY_EDGES_FP16` — NaN payloads, subnormals, signed zeros — moves
  bit-exactly; MATMUL and reductions accumulate in binary32 (the accumulator width TOSA assigns
  FP16); stores repack with the same neighbour-safe atomics `BOOL` uses. Numerics are
  bit-identical across devices by construction.
- **Whole-graph execution** (ADR 0007): the graph's execution order becomes one command buffer of
  compute dispatches with `COMPUTE → COMPUTE` memory barriers between dependent dispatches.
  `CONST` tensors and intermediates live in one per-program arena allocation (lifetime-packed;
  `RESHAPE`/`IDENTITY` of arena tensors are views, not copies; operators the analysis proves dead
  are never dispatched). Kernels address tensors through one descriptor — an array of storage
  buffers holding the bound slots plus the arena — selected by specialization constants, so one
  crate-authored module per kernel serves every binding layout and `CONCAT` with any input count.
  Guest bytes never reach the driver's shader compiler (ADR 0003).
- **Kernel numerics** (ADR 0007): every float operation is `NoContraction`; `SIN`, `COS`,
  `TANH`, and `ERF` are crate-authored range reductions and polynomials (Cephes below
  |x| = 8192, Payne–Hanek above it, within one ulp of binary64 references in the lavapipe
  tests) instead of the driver's built-ins, whose precision Vulkan specifies loosely or not at
  all; NaN modes (`PROPAGATE`/`IGNORE`) follow the TOSA pseudocode literally; `MATMUL` is a
  shared-memory tiled kernel bit-identical to the sequential ascending-k sum. `BOOL` tensors are
  read by word and written with `OpAtomicAnd`/`OpAtomicOr`, so a predicate output never modifies
  a neighbouring byte, even at an unaligned tail.
- **Memory domains** (ADR 0005): `Host` and `Shared` are persistently mapped host-coherent
  allocations; `Device` is device-local memory reached only through bounded staging inside
  `write_buffer`/`read_buffer`. `Shared` and `Device` are advertised only when the device exposes
  a matching memory type. Every buffer is a dedicated allocation bound directly as a storage
  buffer; alignment is measured, never assumed.
- **Execution** (ADR 0006): a bounded per-context ring of (command buffer, fence, descriptor set)
  triples; `vkQueueSubmit2` success is the admission boundary; `poll_event` is one
  `vkGetFenceStatus` read with no worker thread; finite timeouts are rejected before admission;
  `VK_ERROR_DEVICE_LOST` poisons the instance. Pipelines are created against one per-instance
  `VkPipelineCache`.
- **Diagnostics:** `direct_binding_admissions`, `explicit_transfer_bytes`, and `live_resources`
  feed the conformance suite's copy-path and accounting hooks; `VulkanProgram::dispatch_count`
  and `arena_bytes` expose a loaded program's shape.

The provisional integer target (`VULKAN_TOSA_INTEGER_TARGET`) is declared but not advertised;
its per-device gating remains open under wayfinder ticket 5 (ADR 0004).

## Build-time gate

`VIRTIO_ACCEL_VULKAN=1` makes an unsupported target a loud build failure, `=0` forces the
placeholder, and unset is auto. The supported set is enumerated in `build.rs`; runtime presence of
a Vulkan 1.3 loader and a compute-capable device is discovered when the backend initializes and
reported as `InitError`.

## Running

```sh
cargo run -p virtio-accel-vulkan --example tosa_vulkan
cargo test -p virtio-accel-vulkan
```

The example executes the FP32 identity artifact and then the three-operator `tanh(x · w + bias)`
graph on the preferred device (discrete, integrated, virtual, then CPU) and exits successfully, or
reports that no device is available. The native tests run against every enumerated device and
skip without one; `VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1` turns absence into a failure, and
`VK_DRIVER_FILES` pins the ICD (the CI lane pins lavapipe).

On macOS the Vulkan loader comes from MoltenVK (Homebrew `molten-vk` plus `vulkan-loader`, or the
LunarG SDK). A Homebrew loader lives in `/opt/homebrew/lib`, which is not on the default `dlopen`
search path, so test and example runs need it exported:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/lib cargo test -p virtio-accel-vulkan
```

## Verified driver stacks

The full backend suite — admission, lifecycle, the conformance suite, every case of the shared
FP32 operator corpus in every advertised memory domain, and the kernel-level tests (transcendental
ulp sweeps, tiled-MATMUL bit identity, rank-4 broadcasting, byte-tensor neighbour safety) — passes
on Mesa lavapipe in the `vulkan-lavapipe-test` CI lane and, on 2026-09-06, on
Intel Arc 140V (Lunar Lake, Mesa 26.0.8 ANV, Vulkan 1.4.335) together with the same host's llvmpipe (LLVM 21.1.8), in
`Host`, `Shared`, and `Device` domains; the transcendental kernels measured 1 ulp (sin, cos, tanh)
and 2 ulp (erf) worst case against binary64 on both devices. On 2026-09-17 the same suite passed
on Apple M4 via MoltenVK 1.4.2 (local validation only, not a CI lane). One crate, no per-driver
code paths.

**FP16 tier.** The tier is advertised on every device the backend opens, lavapipe and MoltenVK
included, so the CI lane covers it continuously. On 2026-09-17 the full FP16 corpus passed on
Apple M4 via MoltenVK 1.4.2: the ten bit-exact cases, the ulp-tolerated unary/comparison/
logical/reduction/movement groups, the fully strict 65536-pattern `NEGATE` round trip, the
eight higher-precision lanes within 1 ulp of the correctly rounded binary64 references over the
whole finite binary16 domain, and the subnormal-arithmetic probe (add/sub/mul/compare/max/min/
reciprocal/abs over subnormal operands, exact IEEE results). Intel Arc LNL (Mesa ANV) and AMD
Radeon 860M (RADV) passed the corpus against the same kernels the same day; their confirmation
runs of the final probe are owed. Host-side, the binary16 conversions are verified exhaustively
against an independent reference and every kernel variant passes
`spirv-val --target-env vulkan1.3`.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

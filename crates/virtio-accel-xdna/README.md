# virtio-accel-xdna

An AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`, targeting XDNA2/Strix-class parts
through the HRX runtime (`libhrx`) with no XRT userspace dependency. It executes device-neutral
TOSA 1.0 programs on the NPU, compiling admitted graphs with the pinned aiecc toolchain as a
bounded subprocess (never a Cargo dependency, never in-process Python).

**Portability tier:** `host-native` — the amdxdna-native HRX runtime (`libhrx`) when detected at
build time; a compile-only unsupported-runtime placeholder elsewhere.

In a `va_xdna` build it runs the full `Accelerator`
lifecycle — device/stream owner, `hrx_buffer` primitives (persistent mapping, range
flush/invalidate, release), and a serialized dispatch worker bridging
`hrx_stream_dispatch`/`synchronize` to a latched nonblocking `poll_event`. `load_program` accepts
the crate-local precompiled artifact format directly, and a TOSA artifact by admitting it and
compiling it with the bounded aiecc helper subprocess (`compiler/xdna_compile.py`, run under the
pinned toolchain venv in a cleared environment, content-addressed in a cache). The compilable TOSA
subset today is BF16 IDENTITY (a DMA copy), BF16 → FP32 MATMUL (the spec-mandated FP32-accumulator
shape, batch 1, at multiples of the tested compute tile), BF16 NHWC MAX_POOL2D, and explicit
FP8E4M3/FP8E5M2 → BF16 CAST, plus exact INT8 IDENTITY, zero-point-aware INT8 → INT32 MATMUL,
and signed per-tensor INT32 → INT8 RESCALE.
FP8 is a storage tier, not an arithmetic tier: the guest keeps the
conversion visible in its graph, the NPU expands each value exactly, and subsequent programs use
the existing BF16 compute kernels. MAX_POOL2D is
deliberately bounded to batch 1, zero padding, propagating NaNs, kernel and stride dimensions no
larger than 8, and at most 8,192 input-plus-output elements so both tensors fit in the worker's
local-memory budget. All of this is exercised by `tests/hardware.rs` (a DMA passthrough, compiled
TOSA IDENTITY, bit-exact non-square MATMUL, the shared MAX_POOL2D oracle, and both shared
bit-exact FP8 → BF16 CAST oracles, plus the shared exact INT8 identity, nonzero-zero-point
MATMUL, and INT32 → INT8 RESCALE oracles) and by
`tests/conformance.rs` (the shared semantic suite, including the direct-binding copy-path
diagnostics, on the device). Hosts without HRX build the portable admission surface plus a
placeholder, compile no `unsafe`, and still unit-test admission and the artifact codec. The
[AMD XDNA wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/78) records the
implementation lineage; its design decisions cover crate layout (#83), FFI/buffers (#87),
execution model (#85), compiler helper (#84), and the advertised numerical tier (#82). Optional
throughput, parallelism, broader integer, and experimental block-scaled work remains in separately
scoped follow-up issues and does not expand this support claim.

The compiler is never a Cargo dependency and never runs in-process. `compile_artifact` exposes the
admit-then-compile path without a device (the offline / catalog-population use), so a build host can
produce precompiled artifacts that a device-less serving host later loads. It is available on any
unix build — with or without HRX — since the offline build host is exactly a machine without
libhrx; only the pinned toolchain (`VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`) is needed at run time.

## Build-time probe

The boundary is a build-environment probe rather than a target operating system. HRX exposes a
plain C ABI, so there is no C++ bridge and no `cc`/CMake step. The build script resolves an HRX
install prefix from `VIRTIO_ACCEL_HRX_DIR` (highest priority) or `HRX_DIR` (the variable the pinned
toolchain `env.sh` and the fork's tooling export), and enables the native modules only when that
prefix carries `include/hrx/hrx_runtime.h`, `include/hrx/hrx_amdxdna.h` (declaring
`hrx_amdxdna_executable_create` — its absence marks an older, incompatible libhrx generation), and
`lib/libhrx.so`.

`VIRTIO_ACCEL_XDNA=1` makes a missing or incomplete runtime a loud build failure, `=0` forces the
placeholder, and `VIRTIO_ACCEL_HRX_LIB_DIR` links a bare lib directory that ships no headers. No
standard locations are scanned: silently discovering an unpinned libhrx would defeat the version
pin. Builds without the runtime still compile and unit-test the portable admission surface.

## Advertised numerical tier

Per the numerical-tier decision (#82), the crate defines a BF16 floating-point target (TOSA
`EXT-BF16`), a separate FP8 storage target (`EXT-BF16 | EXT-FP8E4M3 | EXT-FP8E5M2`), and an exact
integer target. It rejects FP32/FP16 compute at admission rather than silently executing it as
BF16. Native builds expose the implemented BF16 IDENTITY, MATMUL, MAX_POOL2D, and explicit
FP8-to-BF16 CAST surfaces plus INT8 IDENTITY, MATMUL, and RESCALE through
`TosaCapabilityProvider`;
placeholder builds expose an empty capability list. The FP8 capability advertises FP8 only for
graph inputs and BF16 only for graph
outputs, so it cannot be mistaken for native FP8 arithmetic. MAX_POOL2D advertises the same
propagating-NaN and zero-padding semantic constraints as the OpenVINO reference backend, while
admission applies the narrower XDNA2 shape and local-memory envelope described above.

The integer capability deliberately preserves OpenVINO's `CONST`/`IDENTITY`/`MATMUL`, INT8, and
restricted INT32 semantic baseline, but does not claim the complete TOSA Integer profile. It adds
`RESCALE` as an intentional operator-surface divergence: released TOSA quantized inference needs
an explicit conversion from an INT32 accumulator back to INT8, and XDNA executes that arithmetic
exactly on the NPU rather than falling back to the host. The admitted form is signed INT32 → INT8,
per-tensor, `scale32=true`, `SINGLE_ROUND`, with one compile-time multiplier, shift, and output zero
point. Per-channel, unsigned, double-round, dynamic-shape, and invalid-shift forms are rejected.

Admission is narrower for an XDNA-specific reason: batch is one, dimensions are at most 512, and
each admitted one-core program's padded tensor footprint is at most 16 KiB so depth-two FIFOs fit
one AIE2P core.
Non-word-sized INT8 inputs are rounded up to a four-byte binding slot because AIE DMA descriptors
cannot transfer a shorter granularity; that padding is explicit in the artifact's binding plan and
ignored by the kernel, rather than copied into a hidden bounce buffer.
The same granularity bounds the other end of a binding: a submitted range must start on a four-byte
boundary as well as cover its slot's exact length, and a mid-word offset is rejected as
`Incompatible` before the device sees it. RESCALE similarly rounds
the output slot to a DMA word and clears its at-most-three-byte tail on the NPU. Zero points remain
compile-time values: MATMUL subtracts signed INT8 zero points before exact INT32 accumulation, and
RESCALE applies its signed INT8 output zero point only after exact 64-bit multiply-and-round math.

## Completion and fault model

One worker serializes each instance's accepted submissions. Finite timeouts are rejected before
admission because HRX exposes no cancellation primitive. A definite dispatch/synchronize failure
becomes a stable terminal `Failed` event and poisons that backend instance (device-loss tier 1);
the event and its buffers can still be released normally. A 120-second userspace watchdog, longer
than the kernel's 60-second NPU TDR, detects a synchronize call that never returns (tier 2). In that
case `poll_event` reports `DeviceLost`, the event remains pending and cannot be released, and the
host must discard the backend instance. The detached worker retains the stream, executable, and
buffer allocations so discarding cannot free native memory that HRX might still touch.

HRX's native release functions are infallible `void` calls, and this backend rejects submissions
before admission whenever ownership remains with the caller. Therefore the current XDNA runtime
produces no `SubmitFailure::Indeterminate` or `ReleaseFailure::Indeterminate` path; those protocol
states remain reserved for a future runtime operation whose ownership outcome is genuinely
unknown.

## Validated baseline

The hardware claim is pinned to the configuration proven on August 26, 2026:

- Fedora 44 with Linux `7.1.8-200.fc44.x86_64` and the in-tree `amdxdna` driver;
- PCI `1022:17f0` revision `0x20` (Strix/Krackan/Strix Halo XDNA2), exposed as
  `/dev/accel/accel0`;
- firmware `1.1.2.64` from `amdnpu/17f0_20/npu.sbin`;
- toolchain prefix `amdxdna-hrx-v2026.08`: Python 3.12.14, `amd-npu-compiler` commit
  `c9554426`, `mlir_aie==1.4.1`, Peano/`llvm-aie==21.0.0.2026080301+c9c5ecb7`, and
  `hrx-xclbinutil` commit `3940dd23`;
- HRX tag `flm-hrx-amdxdna-v2026.07.30`, source commit `eb0b39f4`, C ABI/library version
  `0.1.0`, release SHA-256 `661ed94051cc6ad04f53739b2df7a791aecb658bc435bd5a6ff3c46716696345`;
  and
- unfolded DDR addressing (`NPU_RUNTIME=hrx`, `fold_ddr_addr_offset=false`).

This is an evidence boundary, not a claim that every XDNA generation or runtime revision works.
The compiler/Python toolchain runs only while populating the content-addressed artifact catalog or
on a load path configured to compile; a serving host loading precompiled XDNP artifacts does not
need Python or the compiler. The currently validated native runtime still uses the Linux `amdxdna`
driver; a future kore-native runtime adapter is outside this support claim.

## Running

```sh
cargo run -p virtio-accel-xdna --example tosa_xdna
cargo test -p virtio-accel-xdna
```

Without HRX the example reports the placeholder state; in a `va_xdna` build it initializes the
device and stream. The portable tests exercise the advertised targets on every host; the
`va_xdna` `tests/hardware.rs` suite exercises the buffer primitives against a live NPU.

The exact manual replacement for the unavailable public hardware CI lane is:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
export VIRTIO_ACCEL_XDNA=1
export VIRTIO_ACCEL_XDNA_REQUIRE_HARDWARE=1
export VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08
cargo test -p virtio-accel-xdna --test conformance -- --test-threads=1
cargo test -p virtio-accel-xdna --features test-control --test hardware -- --test-threads=1
```

`test-control` only adds the hidden one-shot tier-1/tier-2 constructors used by the hardware test;
ordinary `XdnaAccelerator::new` retains production behavior even in an all-features build.
`VIRTIO_ACCEL_XDNA=1` makes an incomplete HRX install fail at build time, while the test-only
`VIRTIO_ACCEL_XDNA_REQUIRE_HARDWARE=1` turns an inaccessible NPU into a test failure instead of the
ordinary developer-machine skip. Together they prevent the manual hardware lane from passing
without exercising the native backend.

## Performance evidence

The FP8 storage conversion streams 1,024-element tiles through one AIE2P worker. That is an
intentional first-tier implementation boundary, not a claim that one worker is the final throughput
configuration. Program compilation happens at load time and is content-addressed; warm submission
binds the caller's FP8 and BF16 buffers directly and performs no explicit transfer or hidden bounce
copy.

An ignored release-mode benchmark mirrors the OpenVINO evidence structure: 20 warmups, 200 measured
submissions, separate admission and submit-to-complete percentiles, and direct-binding diagnostics.
It measures 1,024 through 1,048,576 elements and validates every output against the exact FP8 oracle
after timing:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
export VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08
cargo test --release -p virtio-accel-xdna --test hardware \
  measures_fp8_cast_scaling_on_one_aie_worker -- --ignored --nocapture --test-threads=1
```

On August 25, 2026, the reference `1022:17f0` XDNA2 NPU and v2026.08 toolchain produced stable
linear scaling: submit-to-complete p50 was 86.4 microseconds at 1,024 elements, 485.9 microseconds at
16,384, 6.743 milliseconds at 262,144, and 26.656 milliseconds at 1,048,576. Effective combined
input/output traffic reached 0.110 GiB/s at the largest size. Every size reported 440 direct
bindings and zero explicit transfer bytes across 20 warmups plus 200 measured submissions. These
numbers establish the one-worker baseline; increasing AIE-worker parallelism remains an optional
throughput optimization rather than a correctness requirement for this storage tier.

Wall-clock results are manual release evidence from the named NPU/toolchain, not a portable CI gate.
Deterministic CI continues to enforce exact numerics, direct bindings, and zero submission-time
transfer bytes.

The INT8 MATMUL benchmark follows the same 20-warmup/200-sample structure. Shapes divisible by the
AIE2P 4x4x8 INT16 matrix tile widen and subtract both zero points exactly on the NPU, then use the
native matrix unit; small shapes that cannot fill one tile use a scalar on-NPU kernel. Both paths
remain direct-bound and bit-exact. Run it with:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
export VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08
cargo test --release -p virtio-accel-xdna --test hardware \
  measures_exact_int8_matmul_latency -- --ignored --nocapture --test-threads=1
```

On August 26, 2026, the reference `1022:17f0` XDNA2 NPU and v2026.08 toolchain measured a
64x64x32 specialization at 661 ns admission p50 and 334.065 microseconds submit-to-complete p50
(343.723 microseconds p95), or 0.785 effective GOPS at this dispatch-sized workload. All 660
bindings across 220 submissions were direct and submission performed zero explicit transfer
bytes. This is evidence for the initial one-core tier, not a peak-throughput claim.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

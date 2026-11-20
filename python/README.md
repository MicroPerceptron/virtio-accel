# Experimental Python / NumPy frontend

For execution inside QEMU, use the [Linux guest driver](../guest/linux/README.md)
and `devices(provider="virtio")`. Vendor runtimes stay on the host. This provider
requires a wheel built from the guest-driver change; older releases lack it.

`virtio_accel.numpy` provides immutable FP16/BF16/FP32 arrays, static tracing, validated
TOSA export, and synchronous native execution through Core ML, Vulkan, OpenVINO,
Hexagon and XDNA. NumPy remains an explicit CPU reference. No speedup is claimed.

## Install from PyPI

```sh
python -m pip install virtio-accel==0.1.0a1
```

Requires Python 3.10 or newer. See the [generated API reference](https://virtio-accel.org/python/api/)
for public classes, functions and signatures. The online docs follow `main` and
identify the SDK version; they may describe changes not yet released on PyPI.

## Build from this repository

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
# On macOS, build/tag for the Core ML backend's minimum supported OS:
# export MACOSX_DEPLOYMENT_TARGET=14.0
python -m pip install .
```

Building requires Rust 1.85+ and access to Cargo dependencies. The Python extension is
an isolated Cargo workspace: PyO3 does not become a dependency of any portable crate.
The package is published on [PyPI](https://pypi.org/project/virtio-accel/).
The release workflow builds CPython
3.10+ stable-ABI wheels for Linux (glibc 2.28+) and macOS 14+ on x86-64/ARM64,
and Windows x86-64, plus a standalone source archive. Windows ARM64 wheels and
Qualcomm hardware CI are deferred, tracked in [#209](https://github.com/MicroPerceptron/virtio-accel/issues/209); Hexagon remains available through QAIRT-enabled
source builds. See [first publication](https://github.com/MicroPerceptron/virtio-accel/blob/main/python/releasing.md).
Prebuilt wheels do not require Rust. Source builds do; macOS source builds also require
a macOS 26 SDK, and Windows ARM64 Hexagon builds require QAIRT development headers.

## Familiar functions, explicit execution

```python
import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp

@va.jit
def project(x, weight, bias):
    return vnp.mean(vnp.maximum(x @ weight + bias, 0), axis=-1)

rng = np.random.default_rng(42)
x = rng.normal(size=(4, 8)).astype(np.float32)
weight = rng.normal(size=(8, 3)).astype(np.float32)
bias = np.zeros((3,), np.float32)

lowered = project.lower(x, weight, bias)
artifact = lowered.to_tosa()  # bytes, validated by the native Rust extension
with open("project.tosa", "wb") as f:
    f.write(artifact)

reference = lowered.reference(x, weight, bias).to_numpy()
np.testing.assert_allclose(
    reference, np.maximum(x @ weight + bias, 0).mean(axis=-1),
    rtol=2e-6, atol=2e-6,
)
```

The exported artifact is static TOSA 1.0 Floating Point, Level 8K, with no optional
extensions for FP16/FP32; BF16 graphs explicitly declare the BF16 extension. Validation is not a claim of admission by every backend. Backend dtype,
operator, shape, and runtime requirements still apply. Unsupported graphs fail admission;
there is no automatic provider, dtype, or CPU fallback.

Eager `vnp` operations run on NumPy as a CPU reference. Calling a decorated function
requires explicit `device="npu"` / `device="gpu"` / `device="cpu"` or `backend="reference"`;
a plain `@va.jit` supports `.lower()` and refuses execution. The reference evaluator
executes the normalized graph, not the original Python function.

`lower()` specializes on input shapes and dtypes, caches up to 16 traces by default,
and does not capture input values as constants. Functions must be pure: Python executes
once per new specialization. Captured arrays and Python literals are snapshotted when
traced; changing a closure or global requires `clear_cache()`. Fixed Python loops and
shape-dependent branches execute while tracing. Array-dependent control flow, nested
`jit`, variadic function signatures, and static non-tensor arguments are unsupported.
Functions may return one array or a nonempty tuple of arrays.

## Native execution

```python
print(va.devices())                 # immutable Device records, including provider/placement
print(va.device_diagnostics())      # missing runtimes and excluded endpoints

# Use the lowered FP32 project graph from above. Select an available class explicitly.
with lowered.compile("npu", timeout=30.0) as executable:
    result = executable(x, weight, bias).to_numpy()
    print(executable.device)       # Core ML reports cpu_participation=True

@va.jit(device="gpu", cache_size=4, timeout=30.0, strict=True)
def gpu_project(x, weight):
    return x @ weight

# gpu_project(x, weight) compiles once per shape/dtype and reuses native buffers.
# gpu_project.clear_cache() releases cached programs/buffers and retraces on next use.
# gpu_project.close() releases resources and permanently closes the function.
# JIT functions also support `with va.jit(...) as function:` for scoped ownership.
```

`cpu`, `npu` and `gpu` alias index zero of their class. Indices describe the current
discovery snapshot, not persistent hardware identities. `backend="reference"` is an
explicit NumPy oracle and never appears as a native device. Mock and vAccel wrapper
crates are testing/integration infrastructure, not additional physical providers.

Filtering with `devices(provider="vulkan")` probes only Vulkan. Pass the returned
`Device` object to preserve that filtered snapshot's indices; a bare string selector
uses the full discovery snapshot. Virtio caches metadata without keeping the device
open; executable creation validates its identity again. `device_diagnostics()` forces
a current probe, including permission and exclusive-ownership checks.

`cache_size` bounds traced graphs. `resident_cache_size` separately bounds programs
and buffers: it defaults to one for virtio and to `cache_size` for native providers.
Programs are retired before compiling a replacement, while retained graphs avoid
rerunning Python tracing. Independent JIT functions still share the host's resource
budget; close unused functions before loading another on a single-program host.

Completion polling uses a bounded fast phase (four retries or 50 µs), then timed
parking from 50 µs up to 1 ms. Native and virtio execution share the same Rust
utility. Its address-wait semantics use the standard library's cross-platform
atomic/condition-variable primitives, without extra platform FFI or a newer macOS
requirement. There is no completion callback in the baseline backend API or wire
protocol: timed polling remains necessary, and a timeout never releases live native
work prematurely. Scheduling precision is platform dependent.

| Provider | Endpoints | Build/runtime requirements | Python admission boundary |
| --- | --- | --- | --- |
| Core ML | NPU, GPU, hybrid CPU/GPU/ANE | macOS 14+ | FP16/FP32; CPU participation; no CAST |
| Vulkan | GPU, macOS/Linux/Windows | Vulkan loader and GPU driver (MoltenVK on macOS) | FP16/FP32; software CPU devices excluded |
| OpenVINO | CPU/GPU/NPU | OpenVINO 2026.3 C runtime and device plugins/drivers at execution time | FP16/FP32, subject to plugin admission; no floating-tier CAST; exact named device, no AUTO/HETERO |
| Hexagon | NPU, Windows ARM64 | Source/local wheel builds require the complete QAIRT/QNN SDK and HTP runtime; no prebuilt release wheel yet | FP16; FP32 rejected without downconversion |
| XDNA | NPU, Linux | Pinned HRX 0.1.0, AMD driver and aiecc compiler toolchain at execution time | BF16 identity and batch-1 BF16→FP32 MATMUL; backend shape restrictions apply |

All five adapters are default Cargo features, enabled on their supported platforms.
Release wheels contain the adapters, not vendor SDKs, drivers, or runtime libraries.
Missing runtimes appear in `va.device_diagnostics()`; importing, tracing, and TOSA
export still work. Installing a compatible runtime enables its compiled adapter
without rebuilding the wheel. A custom build that disables an adapter still needs a rebuild.

OpenVINO and XDNA use runtime loading in Python wheels; their Rust crates retain
build-time linking by default. OpenVINO can use the optional Python dependency:

```sh
# From checkout, before the first PyPI release:
python -m pip install '.[openvino]'
# After publication:
# python -m pip install --pre 'virtio-accel[openvino]'
```

The SDK locates the C library in the installed `openvino` package without importing it.
Explicit environment settings take precedence. Configure them before discovery:

| Runtime | Runtime library selection (highest priority first) |
| --- | --- |
| OpenVINO | `VIRTIO_ACCEL_OPENVINO_LIBRARY` full path; `VIRTIO_ACCEL_OPENVINO_LIB_DIR`; installed Python `openvino` package; OS loader |
| Hexagon | `VIRTIO_ACCEL_QNN_LIBRARY` full path; `VIRTIO_ACCEL_QNN_LIB_DIR`; `VIRTIO_ACCEL_QNN_SDK_ROOT` / `QNN_SDK_ROOT`; OS DLL search |
| XDNA | `VIRTIO_ACCEL_HRX_LIBRARY` full path; `VIRTIO_ACCEL_HRX_LIB_DIR`; `VIRTIO_ACCEL_HRX_DIR` / `HRX_DIR` plus `/lib` |

Libraries and native handles are retained for the process lifetime. Restart Python to
switch an already loaded runtime. XDNA also requires the pinned compiler environment
and `VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`; Hexagon needs its HTP/DSP support libraries and
`ADSP_LIBRARY_PATH` as described in the backend guides. GPU/NPU drivers remain host
prerequisites. See [OpenVINO](https://github.com/MicroPerceptron/virtio-accel/blob/main/crates/virtio-accel-openvino/README.md),
[Hexagon](https://github.com/MicroPerceptron/virtio-accel/blob/main/crates/virtio-accel-hexagon/README.md), and
[XDNA](https://github.com/MicroPerceptron/virtio-accel/blob/main/crates/virtio-accel-xdna/README.md).

Providers can expose the same physical accelerator. Discovery reports provider endpoints,
not a deduplicated physical-device count. To choose a provider explicitly, filter discovery
and pass its immutable record to `jit` or `compile`:

```python
cpu = va.devices("cpu", provider="openvino")[0]
with lowered.compile(cpu, strict=True) as executable:
    result = executable(x, weight, bias)
```

A stale or changed record fails resolution instead of silently selecting a different endpoint.
`strict=True` accepts an explicitly selected CPU; it rejects possible CPU participation in an
accelerator request. Hexagon and XDNA expose one backend endpoint each; multi-device
selection is not claimed for these providers.

For XDNA, BF16 is explicit (provided by `ml-dtypes`); neither FP16 nor FP32 inputs are
silently narrowed. Use rank-3 batch-1 MATMUL to avoid inserting reshape operations outside
XDNA's advertised tier. The current backend requires M and N multiples of 32,
K a multiple of 64, and each dimension at most 512:

```python
xdna = va.devices("npu", provider="xdna")[0]
a = np.ones((1, 32, 64), dtype=vnp.bfloat16)
b = np.ones((1, 64, 32), dtype=vnp.bfloat16)
with va.jit(lambda a, b: a @ b).lower(a, b).compile(xdna, timeout=900) as executable:
    result = executable(a, b)  # FP32 result; compile can invoke aiecc
```

BF16 identity requires a positive multiple of 1024 elements on the current XDNA backend.
Other BF16 operations can trace/export only when TOSA permits them; backend admission
still rejects unsupported graphs. FP8 and integer frontend APIs remain outside this change.

Core ML exposes three explicit policies: `npu` uses **CPUAndNeuralEngine**, `gpu`
uses **CPUAndGPU**, and `hybrid` uses **All** (CPU/GPU/ANE). Core ML chooses placement;
`All` permits both accelerators but does not guarantee simultaneous use. All three
report CPU participation and are rejected by `strict=True`. To preserve precision,
FP32 NeuralNetwork graphs exclude ANE: the NPU policy uses CPUOnly, while hybrid uses
CPUAndGPU. FP16 graphs retain the full selected policy. GPU low-precision accumulation
is disabled. The host-path Rust API retains its original CPU/ANE policy.

```python
combined = va.devices("hybrid", provider="coreml")[0]
with lowered.compile(combined) as executable:
    result = executable(x, weight)
```

Vulkan selects a GPU endpoint and excludes software CPU devices. Core ML, OpenVINO,
and Vulkan's base floating tiers have no `CAST`: mixed FP16/FP32 arithmetic, FP16
reductions (which widen to FP32), and explicit casts can export TOSA but are rejected
by these tiers. FP32 nearest-centroid and FP16/FP32 MLP/FIR are admitted.

Native handles remain on one worker thread per compiled specialization. Calls serialize,
release the GIL while waiting, explicitly copy inputs into reusable native allocations,
and return independent host arrays. These arrays are not resident device tensors.
Use the executable context manager or `close()`, and `clear_cache()` for decorated
functions, to observe cleanup errors. Garbage collection disconnects the worker for
best-effort cleanup; it is not a replacement for explicit close.

`timeout` bounds each compilation/execution/close **wait**, not discovery or native
cancellation. On an execution timeout the executable closes and cannot be reused; its
worker retains ownership until native completion, then cleans up. Indeterminate submission
is never retried. Unknown completion or release quarantines the affected resources and
reports an error; there is no device-reset API yet. A hung native operation can retain a
worker indefinitely, and quarantined resources may require a process restart. Applications
requiring hard cancellation should isolate device execution in their own process.

## Compatibility boundary

| Surface | Implemented |
| --- | --- |
| Creation | `asarray`, `array`, `zeros`, `ones`, `full` |
| Arithmetic | `add`, `subtract`, `multiply`, `negative`, operators `+ - * @` |
| Elementwise | `abs`, `maximum`, `minimum`, `exp`, `log`, `tanh`, `sin`, `cos` |
| Matrix product | 2D or matching-batch 3D `matmul`; BF16 returns FP32 |
| Reductions | `sum`, `mean`; static axis/axis tuple, `keepdims`, FP32 accumulation |
| Shapes | `reshape` with one inferred dimension, `transpose`, `.T` |
| Inspection | `shape`, `dtype`, `ndim`, `size`; explicit `to_numpy()` |

FP16, BF16 and FP32 are implemented, including `astype` and `Array.astype`. Python literals default to FP32; in binary operations with an array they adopt its dtype. Mixed floating dtypes promote to FP32 through explicit graph casts. Existing NumPy arrays and
scalars preserve their declared dtype: unsupported dtypes raise, unless the caller
explicitly requests `dtype=vnp.float16` or `dtype=vnp.float32`. There is no implicit FP64 narrowing. Noncontiguous
host inputs are copied into owned contiguous storage. Host reads return independent
copies. Creation inside a traced function currently creates host constants which are
embedded when used in the trace; device-native creation is future work.

Arrays are immutable. Mutation, `out=`, comparison/Boolean arrays, division, slicing,
dynamic shapes, zero-sized dimensions, arbitrary strides on device, FP8,
integer arithmetic, autograd, DLPack, and standard NumPy dispatch are not implemented.
Unsupported functions and keywords fail rather than forwarding to NumPy. In particular,
`np.asarray(array)` is refused; use `array.to_numpy()` explicitly.

FP16 operations round their results to FP16. Matrix products and reductions accumulate
in FP32. Reductions narrow only after all axes (and mean scaling) are complete;
`dtype=` selects the result dtype. The reference graph uses sequential axis reductions and mean via FP32 reciprocal
multiplication. Its rounding can differ from NumPy's direct multi-axis reduction or
division. The hardware smoke suite compares admitted workloads to this reference; full hardware
numerical conformance and bitwise equivalence are not claimed. NaN bytes in constants are preserved, and extrema request TOSA NaN propagation.

Limits are rank <= 6, positive dimensions fitting int32, 256 MiB per logical tensor,
4096 frontend nodes, and 16 MiB of diagnostic graph JSON. The JSON is an internal
bridge format, not a stable external ABI. TOSA export imposes the production parser's
additional bounds. Reference FP32 temporaries obey the same per-tensor bound. Aggregate execution-memory budgeting is future runtime work.

## Validation

With the package installed:

```sh
python -m unittest discover -s python/tests -v
cargo test --manifest-path python/Cargo.toml
cargo clippy --manifest-path python/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path python/Cargo.toml -- --check
```

For frontend-only tests without building the extension:

```sh
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

The native integration tests skip in that configuration. Wheel CI asserts the extension
is importable before running the suite, so missing native integration cannot silently pass.

## Next iterations

1. Extend per-provider hardware evidence and add FP8/integer frontend APIs.
2. Add Core ML CAST support with numerical evidence, provider-level reset/recovery, and aggregate
   memory budgets before exposing asynchronous or resident-array APIs.
3. Validate source distributions, produce release wheels, and benchmark cold compilation,
   warm execution, and explicit transfers separately.

The wire protocol and existing portable Rust public APIs are unchanged.


## Examples and device placement

See [examples](https://github.com/MicroPerceptron/virtio-accel/blob/main/python/examples/README.md) for an MLP classifier, nearest-centroid classification,
and FIR filtering, with independent numerical checks and optional TOSA export.
See [device placement](https://github.com/MicroPerceptron/virtio-accel/blob/main/python/device-placement.md) for selector semantics and provider constraints.

Run the opt-in hardware suite against an installed wheel:

```sh
VIRTIO_ACCEL_PYTHON_TEST_DEVICE=npu python -m unittest discover -s python/tests -v
# Select a provider when more than one exposes the same device class:
VIRTIO_ACCEL_PYTHON_TEST_DEVICE=hybrid VIRTIO_ACCEL_PYTHON_TEST_PROVIDER=coreml \
  python -m unittest discover -s python/tests -v
```

A requested but unavailable device fails these tests; it is never silently skipped.
Wheel CI checks installed wheels on Linux, macOS and Windows. An additional Linux lane builds
without a vendor SDK, installs the OpenVINO Python runtime afterward, and executes the wheel's FP16/FP32 tests and
examples on the CPU plugin; missing runtime/device is a failure. SDK-free Linux Rust tests
also run the frontend's BF16 fixtures through XDNA's real admission code. These checks do
not themselves prove Intel NPU/GPU, Qualcomm HTP, or AMD NPU execution; the opt-in hardware suite
contains provider-specific Hexagon and XDNA cases for those machines.


### Hardware validation

Installed-wheel results on 2026-11-19, using the mandatory provider/device selectors
above. Each invocation runs 45 tests; skips reflect another provider's tier or examples
outside the selected NPU tier, not a missing requested device.

| Hardware and runtime | Endpoint | Result |
| --- | --- | --- |
| Apple M4, macOS 27.0.1 | Core ML CPU/GPU, CPU/ANE, All (each) | 43 passed, 2 skipped |
| Intel Lunar Lake, Linux 7.0.0-44, OpenVINO 2026.3.0 | CPU and GPU (each) | 43 passed, 2 skipped |
| Same Intel host/runtime | NPU | 42 passed, 3 skipped; FP16 runtime workloads |
| Intel Lunar Lake, Mesa ANV 26.0.8 | Vulkan GPU | 43 passed, 2 skipped |
| AMD Radeon 860M, Linux 7.3.3-200.fc44, Mesa RADV 26.2.6 | Vulkan GPU | 43 passed, 2 skipped |
| AMD XDNA2 1022:17f0 rev 0x20, firmware 1.1.2.64; pinned amdxdna-hrx-v2026.08 toolchain | XDNA NPU | 40 passed, 5 skipped; BF16 identity and BF16-to-FP32 MATMUL |
| Qualcomm HTP v73, Windows ARM64 build 26300, QAIRT 2.49.0.260730, Python 3.14.10 | Hexagon NPU | 42 passed, 3 skipped; FP16 execution and explicit FP32 rejection |

Core ML results validate the selected policy and numerical outputs, not per-operation
engine attribution or simultaneous GPU/ANE activity. FP32 graphs follow the precision
policy described above.

The user-relayed Qualcomm run tested commit `41999d5d7175c29490c7aae10aaa09725c2ce3b3`
on 2026-11-19. Its installed ARM64 wheel discovered a real Hexagon endpoint and passed
FP16 projection/repeated execution, tuple outputs and threaded reuse, and explicit FP32
rejection. The three Python skips cover the generic examples, another provider's CAST
boundary, and XDNA-specific tests. The native Hexagon Rust suite also passed 15 unit and
13 integration tests, with one ignored test. The supplied `hexagon-results.zip` has
SHA-256 `8c6d6cdedfae71eb1efb31a1677003aff03486cff50481c9d2aa0a8bd164268a`.
This verifies the tested SDK/host combination, not installation portability of a published wheel.

### Windows Hexagon runner

Download [test-hexagon.ps1](https://github.com/MicroPerceptron/virtio-accel/blob/main/python/scripts/test-hexagon.ps1) and run it in PowerShell on the
Snapdragon machine (adjust the SDK path):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\test-hexagon.ps1 `
  -QnnSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.49.0.260730'
```

It requires Git, Rust 1.85+, the `aarch64-pc-windows-msvc` Rust target, native ARM64
Python 3.10+, Visual Studio ARM64 C++ build tools, and the complete QAIRT SDK. Pass
`-Python 'C:\path\to\ARM64\python.exe'` if `python` selects an x64 interpreter.
Existing `VIRTIO_ACCEL_QNN_SDK_ROOT` / `QNN_SDK_ROOT` and `ADSP_LIBRARY_PATH` settings
are accepted; `-AdspLibraryPath` overrides the DSP path, otherwise v73 is the default.

The runner clones the current PR 208 head into a fresh temporary directory, records
the exact revision, builds and installs a Hexagon-only wheel in an isolated venv,
requires real Hexagon discovery, and runs both the native Rust backend and Python
tests. `-Revision` can pin a commit; `-WorkDirectory` must name a new directory.
It does not install drivers or modify the SDK. The printed `hexagon-results.zip`
contains only logs and `report.json`, including failed preflight/build steps; relay
that report and the relevant failure logs. Windows CI validates PowerShell 5.1 syntax
and the wrong-architecture diagnostic path, not HTP execution.

## Distribution portability validation

The initial `0.1.0a1` runtime-loading build was exercised with 47 Python tests per
endpoint. Mac M4 Core ML GPU/NPU/hybrid passed 45 with 2 skips each. The portable
`manylinux_2_28_x86_64` wheel built on Gigatron passed OpenVINO CPU/GPU (45 passed,
2 skipped), NPU (44 passed, 3 skipped), and Intel Vulkan (45 passed, 2 skipped).
That same wheel, without rebuilding on AMD, passed XDNA (42 passed, 5 skipped) and
AMD Vulkan (45 passed, 2 skipped). Skips reflect provider/operator/dtype admission.
The new tests also prove import and TOSA export survive missing vendor libraries.

The Mac source archive was extracted outside the checkout and rebuilt successfully;
wheel metadata/license checks and installed-adapter checks pass on Mac and Linux.
These are packaging-portability checks, separate from the earlier Qualcomm execution
results above. The changed Windows runtime-path loader still needs manual validation
on a QAIRT-configured machine before restoring the Windows ARM64 release wheel.
The first release defers that wheel and its hardware CI; hosted CI covers the other
five wheel targets. No PyPI upload is claimed here.

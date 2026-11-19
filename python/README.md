# Experimental Python / NumPy frontend

`virtio_accel.numpy` provides immutable FP16/BF16/FP32 arrays, static tracing, validated
TOSA export, and synchronous native execution through Core ML, Vulkan, OpenVINO,
Hexagon and XDNA. NumPy remains an explicit CPU reference. No speedup is claimed.

## Build from this repository

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
# On macOS, build/tag for the Core ML backend's minimum supported OS:
# export MACOSX_DEPLOYMENT_TARGET=14.0
python -m pip install ./python
```

Building requires Rust 1.85+ and access to Cargo dependencies. The Python extension is
an isolated Cargo workspace: PyO3 does not become a dependency of any portable crate.
The package has not been published to PyPI. Prebuilt distribution wheels and source
distribution validation remain release work; CI builds and tests wheels from checkout.

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
```

`cpu`, `npu` and `gpu` alias index zero of their class. Indices describe the current
discovery snapshot, not persistent hardware identities. `backend="reference"` is an
explicit NumPy oracle and never appears as a native device. Mock and vAccel wrapper
crates are testing/integration infrastructure, not additional physical providers.

| Provider | Endpoints | Build/runtime requirements | Python admission boundary |
| --- | --- | --- | --- |
| Core ML | NPU, GPU, hybrid CPU/GPU/ANE | macOS 14+ | FP16/FP32; CPU participation; no CAST |
| Vulkan | GPU, macOS/Linux/Windows | Vulkan loader and GPU driver (MoltenVK on macOS) | FP16/FP32; software CPU devices excluded |
| OpenVINO | CPU/GPU/NPU | OpenVINO C runtime present **when building the wheel**, plus device plugins/drivers | FP16/FP32, subject to plugin admission; no floating-tier CAST; exact named device, no AUTO/HETERO |
| Hexagon | NPU, Windows ARM64 | Complete QAIRT/QNN SDK at build time and HTP runtime | FP16; FP32 rejected without downconversion |
| XDNA | NPU, Linux | Pinned HRX at build time, AMD driver and aiecc compiler toolchain | BF16 identity and batch-1 BF16→FP32 MATMUL; backend shape restrictions apply |

All five adapters are default Cargo features; SDK-free builds retain explicit unavailable
placeholders. Installing a vendor SDK after building a wheel does not enable its native
backend: rebuild with the backend's documented environment. See the build instructions
for [OpenVINO](../crates/virtio-accel-openvino/README.md),
[Hexagon](../crates/virtio-accel-hexagon/README.md) and
[XDNA](../crates/virtio-accel-xdna/README.md). Force-on variables
`VIRTIO_ACCEL_OPENVINO=1`, `VIRTIO_ACCEL_HEXAGON=1`, and `VIRTIO_ACCEL_XDNA=1`
make missing build prerequisites fail loudly. The OpenVINO CI lane uses a native Linux wheel
linked to its installed runtime, without bundling vendor libraries. For an equivalent local build:

```sh
maturin build --release --manifest-path python/Cargo.toml --out dist --compatibility linux --auditwheel skip
python -m pip install dist/*.whl
```

This is a machine-local wheel, not a portable manylinux distribution. Vendor-library
redistribution and release-wheel packaging remain separate release work.

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

See [examples](examples/README.md) for an MLP classifier, nearest-centroid classification,
and FIR filtering, with independent numerical checks and optional TOSA export.
See [device placement](device-placement.md) for selector semantics and provider constraints.

Run the opt-in hardware suite against an installed wheel:

```sh
VIRTIO_ACCEL_PYTHON_TEST_DEVICE=npu python -m unittest discover -s python/tests -v
# Select a provider when more than one exposes the same device class:
VIRTIO_ACCEL_PYTHON_TEST_DEVICE=hybrid VIRTIO_ACCEL_PYTHON_TEST_PROVIDER=coreml \
  python -m unittest discover -s python/tests -v
```

A requested but unavailable device fails these tests; it is never silently skipped.
Wheel CI checks installed wheels on Linux, macOS and Windows. An additional Linux lane builds
against the pinned OpenVINO runtime and executes the installed wheel's FP16/FP32 tests and
examples on the CPU plugin; missing runtime/device is a failure. SDK-free Linux Rust tests
also run the frontend's BF16 fixtures through XDNA's real admission code. These checks do
not themselves prove Intel NPU/GPU, Qualcomm HTP, or AMD NPU execution; the opt-in hardware suite
contains provider-specific Hexagon and XDNA cases for those machines.

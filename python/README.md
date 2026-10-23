# Experimental Python / NumPy frontend

`virtio_accel.numpy` provides immutable FP16/FP32 arrays, static tracing, validated
TOSA export, and synchronous native execution through Core ML (NPU-capable) and
Vulkan (GPU). NumPy remains an explicit CPU reference. No speedup is claimed.

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
extensions. Validation is not a claim of admission by every backend. Backend dtype,
operator, shape, and runtime requirements still apply. Unsupported graphs fail admission;
there is no automatic provider, dtype, or CPU fallback.

Eager `vnp` operations run on NumPy as a CPU reference. Calling a decorated function
requires explicit `device="npu"` / `device="gpu"` or `backend="reference"`;
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

`npu` and `gpu` alias `npu:0` and `gpu:0`. Indices describe the current discovery
snapshot; they are not persistent hardware identities. `cpu` is reserved but has no
native provider in this iteration. `backend="reference"` does not appear as a device.
The Python build includes Core ML on macOS and dynamically loaded Vulkan on macOS,
Linux and Windows. OpenVINO, Hexagon and XDNA bindings are subsequent work.

Core ML uses **CPUAndNeuralEngine**. This selects an NPU-capable provider, not proven
ANE-only execution; `strict=True` rejects it. Vulkan selects a GPU endpoint and excludes
software CPU devices. Core ML's existing tier has no `CAST`: mixed FP16/FP32 arithmetic,
FP16 reductions (which widen to FP32), and explicit casts can export TOSA but are rejected
by this provider. FP32 nearest-centroid, and FP16/FP32 MLP/FIR examples run through Core ML.

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
| Matrix product | 2D `matmul`, lowered to rank-3 TOSA MATMUL |
| Reductions | `sum`, `mean`; static axis/axis tuple, `keepdims`, FP32 accumulation |
| Shapes | `reshape` with one inferred dimension, `transpose`, `.T` |
| Inspection | `shape`, `dtype`, `ndim`, `size`; explicit `to_numpy()` |

FP16 and FP32 are implemented, including `astype` and `Array.astype`. Python literals default to FP32; in binary operations with an array they adopt its dtype. Mixed FP16/FP32 arrays promote to FP32 through explicit graph casts. Existing NumPy arrays and
scalars preserve their declared dtype: unsupported dtypes raise, unless the caller
explicitly requests `dtype=vnp.float16` or `dtype=vnp.float32`. There is no implicit FP64 narrowing. Noncontiguous
host inputs are copied into owned contiguous storage. Host reads return independent
copies. Creation inside a traced function currently creates host constants which are
embedded when used in the trace; device-native creation is future work.

Arrays are immutable. Mutation, `out=`, comparison/Boolean arrays, division, slicing,
dynamic shapes, zero-sized dimensions, arbitrary strides on device, FP8/BF16,
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

1. Add OpenVINO/Hexagon/XDNA discovery and execution adapters with device-specific admission tests.
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
# Or VIRTIO_ACCEL_PYTHON_TEST_DEVICE=gpu on a Vulkan GPU host.
```

A requested but unavailable device fails these tests; it is never silently skipped.
Wheel CI checks discovery and explicit unavailable-device failures without requiring hardware.

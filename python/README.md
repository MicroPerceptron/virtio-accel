# Experimental Python / NumPy frontend

This first iteration provides `virtio_accel.numpy`, immutable FP32 reference arrays,
static function tracing, and native export through `virtio-accel-tosa-build` and the
production TOSA validator. It is a compiler frontend prototype, not a hardware
execution binding. No device execution, automatic CPU fallback, or speedup is claimed.

## Build from this repository

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
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
operator, shape, and runtime requirements still apply. This iteration does not load
the artifact, allocate device buffers, submit work, or manage completion events.

Eager `vnp` operations run on NumPy as a CPU reference. Calling a decorated function
requires explicit `@va.jit(backend="reference")`; a plain `@va.jit` supports `.lower()`
and refuses execution. This prevents a future device API from silently executing on
the CPU today. The reference evaluator executes the normalized graph, not the original
Python function. `jit` is a forward-looking API name here; it is not a performance claim.

`lower()` specializes on input shapes and dtypes, caches up to 16 traces by default,
and does not capture input values as constants. Functions must be pure: Python executes
once per new specialization. Captured arrays and Python literals are snapshotted when
traced; changing a closure or global requires `clear_cache()`. Fixed Python loops and
shape-dependent branches execute while tracing. Array-dependent control flow, nested
`jit`, variadic function signatures, and static non-tensor arguments are unsupported.
Functions may return one array or a nonempty tuple of arrays.

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

Only FP32 is implemented. Python literals default to FP32. Existing NumPy arrays and
scalars preserve their declared dtype: unsupported dtypes raise, unless the caller
explicitly requests `dtype=vnp.float32`. There is no implicit FP64 narrowing. Noncontiguous
host inputs are copied into owned contiguous storage. Host reads return independent
copies. Creation inside a traced function currently creates host constants which are
embedded when used in the trace; device-native creation is future work.

Arrays are immutable. Mutation, `out=`, comparison/Boolean arrays, division, slicing,
dynamic shapes, zero-sized dimensions, arbitrary strides on device, FP16/FP8/BF16,
integer arithmetic, autograd, DLPack, and standard NumPy dispatch are not implemented.
Unsupported functions and keywords fail rather than forwarding to NumPy. In particular,
`np.asarray(array)` is refused; use `array.to_numpy()` explicitly.

The reference graph uses sequential axis reductions and mean via FP32 reciprocal
multiplication. Its rounding can differ from NumPy's direct multi-axis reduction or
division. Hardware numerical conformance is future work; no bitwise equivalence is
claimed. NaN bytes in constants are preserved, and extrema request TOSA NaN propagation.

Limits are rank <= 6, positive dimensions fitting int32, 256 MiB per logical tensor,
4096 frontend nodes, and 16 MiB of diagnostic graph JSON. The JSON is an internal
bridge format, not a stable external ABI. TOSA export imposes the production parser's
additional bounds. Aggregate execution-memory budgeting is future runtime work.

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

1. Bind one real backend with explicit context/program/buffer/event ownership, including
   indeterminate submissions, timeouts, release failures, and device loss.
2. Reuse resident programs and device buffers; expose async submission only after lifetime
   tests pass. Report actual backend placement rather than assuming all work runs on an NPU.
3. Add dtype/operation coverage against backend-specific numerical tests and publish the
   admitted subset. Add standard NumPy dispatch only as another frontend to these primitives.
4. Validate source distributions, produce release wheels, and benchmark cold compilation,
   warm execution, and explicit transfers separately.

The wire protocol and existing portable Rust public APIs are unchanged.


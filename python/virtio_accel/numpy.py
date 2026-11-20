"""A deliberately small NumPy-style tensor namespace.

Eager calls are an explicit CPU reference implementation. Traced calls construct
TOSA-compatible primitives. Unsupported functions are not forwarded to NumPy.
"""

import math
import operator
import numpy as _np
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2

from ._array import FLOAT_DTYPES, Array, cast, checked_dtype, checked_shape, host_array, primitive, promote

float16 = _np.float16
float32 = _np.float32
int8 = _np.int8
int32 = _np.int32
__all__ = ["float8_e4m3fn", "float8_e5m2", "int8", "int32", "float16", "bfloat16", "float32", "asarray", "array", "zeros", "ones", "full", "add",
           "subtract", "multiply", "maximum", "minimum", "negative", "abs",
           "exp", "log", "tanh", "sin", "cos", "matmul", "reshape",
           "transpose", "astype", "sum", "mean"]


def asarray(a, dtype=None):
    """Create an immutable array or reuse an existing compatible Array.

    FP16, BF16, FP32, FP8 E4M3FN/E5M2, INT8 and INT32 are supported. Host inputs are copied; symbolic
    values retain their trace. Use ``dtype`` to request an explicit conversion."""
    return host_array(a, dtype)


def array(a, dtype=None):
    """Create an immutable array; equivalent to :func:`asarray`."""
    return host_array(a, dtype)


def astype(a, dtype):
    """Explicitly cast an array; integer casts are limited to INT8 widening."""
    return cast(asarray(a), dtype)


def full(shape, fill_value, dtype=float32):
    """Create a constant with a static shape and scalar fill value.

    Dimensions must be positive and rank at most six; default dtype is FP32."""
    dtype = checked_dtype(dtype)
    shape = checked_shape((shape,) if isinstance(shape, int) else shape, dtype)
    if isinstance(fill_value, Array):
        raise TypeError("full requires a static scalar fill value")
    return asarray(_np.full(shape, fill_value, dtype=dtype))


def zeros(shape, dtype=float32):
    """Create a static array filled with zeros; default dtype is FP32."""
    return full(shape, 0, dtype)


def ones(shape, dtype=float32):
    """Create a static array filled with ones; default dtype is FP32."""
    return full(shape, 1, dtype)


def _binary(op, a, b):
    a, b = promote(a, b)
    _require_float(a)
    shape = checked_shape(_np.broadcast_shapes(a.shape, b.shape), a.dtype)
    # TOSA binary operands have equal rank; NumPy permits leading dimensions.
    if a.ndim < len(shape):
        a = reshape(a, (1,) * (len(shape) - a.ndim) + a.shape)
    if b.ndim < len(shape):
        b = reshape(b, (1,) * (len(shape) - b.ndim) + b.shape)
    return primitive(op, (a, b), shape)


def add(a, b):
    """Add arrays elementwise with broadcasting and frontend dtype promotion."""
    return _binary("add", a, b)


def subtract(a, b):
    """Subtract arrays elementwise with broadcasting and frontend dtype promotion."""
    return _binary("sub", a, b)


def multiply(a, b):
    """Multiply arrays elementwise with broadcasting and frontend dtype promotion."""
    return _binary("mul", a, b)


def maximum(a, b):
    """Select elementwise maxima with broadcasting and dtype promotion."""
    return _binary("maximum", a, b)


def minimum(a, b):
    """Select elementwise minima with broadcasting and dtype promotion."""
    return _binary("minimum", a, b)


def _require_float(a):
    if a.dtype not in FLOAT_DTYPES:
        raise TypeError("arithmetic requires float16/bfloat16/float32; explicitly widen FP8 or integer storage")

def _unary(op, a):
    a = asarray(a)
    _require_float(a)
    return primitive(op, (a,), a.shape)


def negative(a):
    """Negate each element, preserving shape and dtype."""
    return _unary("negate", a)


def abs(a):
    """Compute absolute values elementwise, preserving shape and dtype."""
    return _unary("abs", a)


def exp(a):
    """Compute the exponential elementwise, preserving shape and dtype."""
    return _unary("exp", a)


def log(a):
    """Compute the natural logarithm elementwise, preserving shape and dtype."""
    return _unary("log", a)


def tanh(a):
    """Compute the hyperbolic tangent elementwise, preserving shape and dtype."""
    return _unary("tanh", a)


def sin(a):
    """Compute sine elementwise in radians, preserving shape and dtype."""
    return _unary("sin", a)


def cos(a):
    """Compute cosine elementwise in radians, preserving shape and dtype."""
    return _unary("cos", a)


def reshape(a, shape):
    """Change shape without changing the element count.

    One dimension may be ``-1`` for inference; all resolved dimensions must
    be positive. The resulting rank must not exceed six."""
    a = asarray(a)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    shape = tuple(operator.index(n) for n in shape)
    if shape.count(-1) > 1:
        raise ValueError("at most one inferred dimension is allowed")
    if -1 in shape:
        known = math.prod(n for n in shape if n != -1)
        if known <= 0 or a.size % known:
            raise ValueError("cannot infer reshape dimension")
        shape = tuple(a.size // known if n == -1 else n for n in shape)
    shape = checked_shape(shape, a.dtype)
    if math.prod(shape) != a.size:
        raise ValueError("reshape must preserve the number of elements")
    return primitive("reshape", (a,), shape)


def transpose(a, axes=None):
    """Permute dimensions; reverse all dimensions when axes is None.

    Explicit axes must be a complete permutation; negative axes are accepted."""
    a = asarray(a)
    if axes is None:
        axes = tuple(reversed(range(a.ndim)))
    else:
        axes = tuple(operator.index(i) for i in axes)
        if any(i < -a.ndim or i >= a.ndim for i in axes):
            raise ValueError("transpose axis out of range")
        axes = tuple(i + a.ndim if i < 0 else i for i in axes)
    if sorted(axes) != list(range(a.ndim)):
        raise ValueError("axes must be a permutation of all dimensions")
    return primitive("transpose", (a,), tuple(a.shape[i] for i in axes), axes=list(axes))


def matmul(a, b):
    """Multiply two matrices or two batches with matching batch counts.

    Both operands must have the same rank, either two or three, with matching
    inner dimensions. Batch broadcasting and vector inputs are unsupported.
    INT8 operands produce exact INT32 output (zero points are zero).
    BF16 operands produce FP32 accumulator output; other output dtypes follow
    frontend promotion."""
    a, b = promote(a, b)
    if a.dtype != _np.dtype(int8):
        _require_float(a)
    if a.ndim != b.ndim or a.ndim not in (2, 3):
        raise NotImplementedError("matmul requires two 2D or matching-batch 3D arrays")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("matmul inner dimensions do not match")
    dtype = int32 if a.dtype == _np.dtype(int8) else float32 if a.dtype == _np.dtype(bfloat16) else a.dtype
    if a.ndim == 3:
        if a.shape[0] != b.shape[0]:
            raise ValueError("matmul batch dimensions must match")
        return primitive("matmul", (a, b), (a.shape[0], a.shape[1], b.shape[2]), dtype=dtype)
    m, k = a.shape
    _, n = b.shape
    value = primitive("matmul", (reshape(a, (1, m, k)), reshape(b, (1, k, n))),
                      (1, m, n), dtype=dtype)
    return reshape(value, (m, n))


def _axes(a, axis):
    if axis is None:
        return tuple(range(a.ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(operator.index(i) for i in axes)
    if any(i < -a.ndim or i >= a.ndim for i in axes):
        raise ValueError("reduction axis out of range")
    axes = tuple(i + a.ndim if i < 0 else i for i in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("duplicate reduction axis")
    return tuple(sorted(axes))


def _reduction(a, axis, dtype, out, keepdims, average):
    _require_float(a)
    if out is not None:
        raise NotImplementedError("out= and mutation are not supported")
    result_dtype = a.dtype if dtype is None else checked_dtype(dtype)
    if not isinstance(keepdims, bool):
        raise TypeError("keepdims must be a static bool")
    axes = _axes(a, axis)
    # Widen before reducing any axis. Narrow only after all reductions and,
    # for mean, scaling; an FP16 intermediate sum could overflow spuriously.
    value = cast(a, float32)
    for ax in axes:
        shape = list(value.shape)
        shape[ax] = 1
        value = primitive("reduce_sum", (value,), tuple(shape), axis=ax)
    if not keepdims:
        value = reshape(value, tuple(n for i, n in enumerate(a.shape) if i not in axes))
    if average:
        count = math.prod(a.shape[i] for i in axes)
        value = multiply(value, _np.float32(1.0 / count))
    return cast(value, result_dtype)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Sum over all dimensions or the specified integer/tuple of axes.

    Negative axes are accepted; duplicates are rejected. ``keepdims`` retains
    reduced dimensions of size one. ``dtype`` explicitly selects the result
    dtype; ``out`` must be None because arrays are immutable."""
    return _reduction(asarray(a), axis, dtype, out, keepdims, False)


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Average over all dimensions or specified integer/tuple of axes.

    Uses the same axis, dtype and keepdims rules as :func:`sum`.
    ``out`` must be None. Backend admission depends on the selected dtype."""
    return _reduction(asarray(a), axis, dtype, out, keepdims, True)


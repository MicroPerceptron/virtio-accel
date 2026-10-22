"""A deliberately small NumPy-style FP32 namespace.

Eager calls are an explicit CPU reference implementation. Traced calls construct
TOSA-compatible primitives. Unsupported functions are not forwarded to NumPy.
"""

import math
import operator
import numpy as _np

from ._array import Array, checked_shape, host_array, primitive

float32 = _np.float32
__all__ = ["float32", "asarray", "array", "zeros", "ones", "full", "add",
           "subtract", "multiply", "maximum", "minimum", "negative", "abs",
           "exp", "log", "tanh", "sin", "cos", "matmul", "reshape",
           "transpose", "sum", "mean"]


def asarray(a, dtype=None):
    return host_array(a, dtype)


def array(a, dtype=None):
    return host_array(a, dtype)


def full(shape, fill_value, dtype=float32):
    shape = checked_shape((shape,) if isinstance(shape, int) else shape)
    if _np.dtype(dtype) != _np.dtype("float32"):
        raise TypeError("only float32 is supported")
    if isinstance(fill_value, Array):
        raise TypeError("full requires a static scalar fill value")
    return asarray(_np.full(shape, fill_value, dtype=dtype))


def zeros(shape, dtype=float32):
    return full(shape, 0, dtype)


def ones(shape, dtype=float32):
    return full(shape, 1, dtype)


def _binary(op, a, b):
    a, b = asarray(a), asarray(b)
    shape = checked_shape(_np.broadcast_shapes(a.shape, b.shape))
    # TOSA binary operands have equal rank; NumPy permits leading dimensions.
    if a.ndim < len(shape):
        a = reshape(a, (1,) * (len(shape) - a.ndim) + a.shape)
    if b.ndim < len(shape):
        b = reshape(b, (1,) * (len(shape) - b.ndim) + b.shape)
    return primitive(op, (a, b), shape)


def add(a, b):
    return _binary("add", a, b)


def subtract(a, b):
    return _binary("sub", a, b)


def multiply(a, b):
    return _binary("mul", a, b)


def maximum(a, b):
    return _binary("maximum", a, b)


def minimum(a, b):
    return _binary("minimum", a, b)


def _unary(op, a):
    a = asarray(a)
    return primitive(op, (a,), a.shape)


def negative(a):
    return _unary("negate", a)


def abs(a):
    return _unary("abs", a)


def exp(a):
    return _unary("exp", a)


def log(a):
    return _unary("log", a)


def tanh(a):
    return _unary("tanh", a)


def sin(a):
    return _unary("sin", a)


def cos(a):
    return _unary("cos", a)


def reshape(a, shape):
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
    shape = checked_shape(shape)
    if math.prod(shape) != a.size:
        raise ValueError("reshape must preserve the number of elements")
    return primitive("reshape", (a,), shape)


def transpose(a, axes=None):
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
    a, b = asarray(a), asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise NotImplementedError("this iteration supports 2D matmul only")
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError("matmul inner dimensions do not match")
    value = primitive("matmul", (reshape(a, (1, m, k)), reshape(b, (1, k, n))),
                      (1, m, n))
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


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    a = asarray(a)
    if out is not None:
        raise NotImplementedError("out= and mutation are not supported")
    if dtype is not None and _np.dtype(dtype) != a.dtype:
        raise TypeError("only float32 accumulation is supported")
    if not isinstance(keepdims, bool):
        raise TypeError("keepdims must be a static bool")
    axes = _axes(a, axis)
    value = a
    for ax in axes:
        shape = list(value.shape)
        shape[ax] = 1
        value = primitive("reduce_sum", (value,), tuple(shape), axis=ax)
    if not keepdims:
        value = reshape(value, tuple(n for i, n in enumerate(a.shape) if i not in axes))
    return value


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    a = asarray(a)
    count = math.prod(a.shape[i] for i in _axes(a, axis))
    return multiply(sum(a, axis, dtype, out, keepdims), _np.float32(1.0 / count))


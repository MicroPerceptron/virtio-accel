"""Immutable reference arrays and symbolic values share the same primitives."""

from dataclasses import dataclass
import math
import operator

import numpy as np

MAX_RANK = 6
MAX_BYTES = 256 * 1024 * 1024
MAX_NODES = 4096


def checked_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise TypeError("only float16 and float32 are supported; request an explicit cast")
    return dtype


def checked_shape(shape, dtype=np.float32):
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise TypeError("shape must contain integers") from exc
    if len(shape) > MAX_RANK or any(n <= 0 or n > 2**31 - 1 for n in shape):
        raise ValueError("this frontend requires positive dimensions and rank <= 6")
    if math.prod(shape) * checked_dtype(dtype).itemsize > MAX_BYTES:
        raise ValueError("tensor exceeds the frontend's 256 MiB limit")
    return shape


@dataclass(frozen=True, eq=False)
class Array:
    """FP16/FP32 value. Eager values use NumPy; symbolic values belong to one trace.

    Use ``to_numpy()`` for an explicit, independent host copy. No device allocation
    or acceleration is implied by the reference implementation.
    """

    _shape: tuple
    _dtype: object = np.dtype("float32")
    _data: object = None
    _graph: object = None
    _id: int = -1
    __array_priority__ = 1000

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def T(self):
        from .numpy import transpose
        return transpose(self)

    def reshape(self, *shape):
        from .numpy import reshape
        return reshape(self, shape[0] if len(shape) == 1 else shape)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        from .numpy import sum
        return sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        from .numpy import mean
        return mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def astype(self, dtype):
        return cast(self, dtype)

    def to_numpy(self):
        if self._graph is not None:
            raise TypeError("a traced value cannot be read on the host")
        return self._data.copy()

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit NumPy conversion is disabled; use .to_numpy()")

    def __bool__(self):
        raise TypeError("array-dependent Python control flow is unsupported")

    def __eq__(self, other):
        raise NotImplementedError("array comparisons are not implemented in this floating-point subset")

    def __ne__(self, other):
        raise NotImplementedError("array comparisons are not implemented in this floating-point subset")

    def __iadd__(self, other):
        raise TypeError("arrays are immutable; use a = a + b")

    def __isub__(self, other):
        raise TypeError("arrays are immutable; use a = a - b")

    def __imul__(self, other):
        raise TypeError("arrays are immutable; use a = a * b")

    def __imatmul__(self, other):
        raise TypeError("arrays are immutable; use a = a @ b")

    def __repr__(self):
        kind = "traced" if self._graph is not None else "host"
        return f"Array(shape={self.shape}, dtype={self.dtype.name}, execution={kind!r})"

    def __add__(self, other):
        from .numpy import add
        return add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        from .numpy import subtract
        return subtract(self, other)

    def __rsub__(self, other):
        from .numpy import subtract
        return subtract(other, self)

    def __mul__(self, other):
        from .numpy import multiply
        return multiply(self, other)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        from .numpy import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .numpy import matmul
        return matmul(other, self)

    def __neg__(self):
        from .numpy import negative
        return negative(self)


def host_array(value, dtype=None):
    if isinstance(value, Array):
        return value if dtype is None else cast(value, dtype)
    # Preserve ndarray/scalar dtype unless the caller explicitly requests a cast.
    # Python literals use FP32 by default, matching this frontend's numeric tier.
    if dtype is None:
        dtype = value.dtype if isinstance(value, (np.ndarray, np.generic)) else np.float32
    dtype = checked_dtype(dtype)
    if isinstance(value, (np.ndarray, np.generic)):
        checked_shape(value.shape, dtype)  # Check destination size before copying/casting.
    data = np.array(value, dtype=dtype, copy=True, order="C")
    shape = checked_shape(data.shape, dtype)
    data.flags.writeable = False
    return Array(shape, _dtype=dtype, _data=data)


def cast(value, dtype):
    dtype = checked_dtype(dtype)
    if value.dtype == dtype:
        return value
    return primitive("cast", (value,), value.shape, dtype=dtype)


def promote(a, b):
    # Python scalar literals are weakly typed: adding 1 to an FP16 array stays
    # FP16. NumPy scalars/arrays and Array values carry an explicit strong dtype.
    weak = (bool, int, float)
    if type(a) in weak and isinstance(b, (Array, np.ndarray, np.generic)):
        b = host_array(b)
        a = host_array(a, b.dtype)
    elif type(b) in weak and isinstance(a, (Array, np.ndarray, np.generic)):
        a = host_array(a)
        b = host_array(b, a.dtype)
    else:
        a, b = host_array(a), host_array(b)
    dtype = checked_dtype(np.promote_types(a.dtype, b.dtype))
    return cast(a, dtype), cast(b, dtype)


class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []

    def node(self, op, shape, inputs=(), *, dtype=np.float32, **attrs):
        dtype = checked_dtype(dtype)
        shape = checked_shape(shape, dtype)
        if len(self.nodes) >= MAX_NODES:
            raise ValueError("trace exceeds the 4096-node limit")
        index = len(self.nodes)
        self.nodes.append(dict(op=op, shape=list(shape), dtype=dtype.name,
                               inputs=list(inputs), attrs=attrs))
        return Array(shape, _dtype=dtype, _graph=self, _id=index)

    def input(self, shape, dtype):
        value = self.node("input", shape, dtype=dtype)
        self.inputs.append(value._id)
        return value

    def lift(self, value):
        if value._graph is self:
            return value
        if value._graph is not None:
            raise ValueError("cannot mix values from different traces")
        return self.node("const", value.shape, dtype=value.dtype,
                         data=value._data.astype(value.dtype.newbyteorder("<"), copy=False).tobytes().hex())

    def document(self):
        return dict(version=2, nodes=self.nodes, inputs=self.inputs, outputs=self.outputs)


def primitive(op, values, shape, *, dtype=None, **attrs):
    values = tuple(host_array(v) for v in values)
    dtype = checked_dtype(values[0].dtype if dtype is None else dtype)
    shape = checked_shape(shape, dtype)  # Bound eager outputs before allocation.
    graphs = {id(v._graph): v._graph for v in values if v._graph is not None}
    if len(graphs) > 1:
        raise ValueError("cannot mix values from different traces")
    if graphs:
        graph = next(iter(graphs.values()))
        ids = [graph.lift(v)._id for v in values]
        return graph.node(op, shape, ids, dtype=dtype, **attrs)
    result = evaluate(op, [v._data for v in values], attrs, shape, dtype)
    return host_array(result)


def evaluate(op, args, attrs, shape, dtype):
    """Explicit NumPy reference evaluator, never used as a device fallback."""
    functions = dict(add=np.add, sub=np.subtract, mul=np.multiply,
                     maximum=np.maximum, minimum=np.minimum, negate=np.negative,
                     abs=np.abs, exp=np.exp, log=np.log, tanh=np.tanh,
                     sin=np.sin, cos=np.cos)
    dtype = checked_dtype(dtype)
    if op == "cast":
        return args[0].astype(dtype)
    if op in functions:
        # Sign/data operations preserve FP16 bit patterns. Arithmetic evaluates
        # in FP32 and rounds once to storage precision, as in the FP16 tier.
        if op not in ("negate", "abs", "maximum", "minimum"):
            for a in args:
                checked_shape(a.shape, np.float32)
            checked_shape(shape, np.float32)
            args = [a.astype(np.float32, copy=False) for a in args]
        return np.asarray(functions[op](*args), dtype=dtype)
    if op == "matmul":
        for a in args:
            checked_shape(a.shape, np.float32)
        checked_shape(shape, np.float32)
        return np.matmul(*(a.astype(np.float32, copy=False) for a in args)).astype(dtype)
    if op == "identity":
        return args[0]
    if op == "reshape":
        return np.reshape(args[0], shape)
    if op == "transpose":
        return np.transpose(args[0], attrs["axes"])
    if op == "reduce_sum":
        return np.sum(args[0], axis=attrs["axis"], keepdims=True, dtype=np.float32).astype(dtype)
    raise NotImplementedError(f"reference evaluator does not implement {op!r}")


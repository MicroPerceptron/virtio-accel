"""Immutable reference arrays and symbolic values share the same primitives."""

from dataclasses import dataclass
import math
import operator

import numpy as np

MAX_RANK = 6
MAX_BYTES = 256 * 1024 * 1024
MAX_NODES = 4096


def checked_shape(shape):
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise TypeError("shape must contain integers") from exc
    if len(shape) > MAX_RANK or any(n <= 0 or n > 2**31 - 1 for n in shape):
        raise ValueError("this frontend requires positive dimensions and rank <= 6")
    if math.prod(shape) * 4 > MAX_BYTES:
        raise ValueError("tensor exceeds the frontend's 256 MiB limit")
    return shape


@dataclass(frozen=True, eq=False)
class Array:
    """FP32 value. Eager values use NumPy; symbolic values belong to one trace.

    Use ``to_numpy()`` for an explicit, independent host copy. No device allocation
    or acceleration is implied by the reference implementation.
    """

    _shape: tuple
    _data: object = None
    _graph: object = None
    _id: int = -1
    __array_priority__ = 1000

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype("float32")

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

    def to_numpy(self):
        if self._graph is not None:
            raise TypeError("a traced value cannot be read on the host")
        return self._data.copy()

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit NumPy conversion is disabled; use .to_numpy()")

    def __bool__(self):
        raise TypeError("array-dependent Python control flow is unsupported")

    def __eq__(self, other):
        raise NotImplementedError("array comparisons are not implemented in this FP32 subset")

    def __ne__(self, other):
        raise NotImplementedError("array comparisons are not implemented in this FP32 subset")

    def __iadd__(self, other):
        raise TypeError("arrays are immutable; use a = a + b")

    def __isub__(self, other):
        raise TypeError("arrays are immutable; use a = a - b")

    def __imul__(self, other):
        raise TypeError("arrays are immutable; use a = a * b")

    def __imatmul__(self, other):
        raise TypeError("arrays are immutable; use a = a @ b")

    def __repr__(self):
        kind = "traced" if self._graph is not None else "reference:cpu"
        return f"Array(shape={self.shape}, dtype=float32, execution={kind!r})"

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
        if dtype is not None and np.dtype(dtype) != value.dtype:
            raise TypeError("only float32 is supported in this iteration")
        return value
    if dtype is not None and np.dtype(dtype) != np.dtype("float32"):
        raise TypeError("only float32 is supported in this iteration")
    if isinstance(value, (np.ndarray, np.generic)):
        checked_shape(value.shape)  # Reject oversized views before allocating a copy.
    # Preserve ndarray/scalar dtype unless the caller explicitly requests a cast.
    # Python literals use FP32 by default, matching this frontend's numeric tier.
    if dtype is None and isinstance(value, (np.ndarray, np.generic)):
        if value.dtype != np.dtype("float32"):
            raise TypeError("input dtype is unsupported; explicitly request dtype=float32")
    data = np.array(value, dtype=np.float32, copy=True, order="C")
    shape = checked_shape(data.shape)
    data.flags.writeable = False
    return Array(shape, _data=data)


class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []

    def node(self, op, shape, inputs=(), **attrs):
        shape = checked_shape(shape)
        if len(self.nodes) >= MAX_NODES:
            raise ValueError("trace exceeds the 4096-node limit")
        index = len(self.nodes)
        self.nodes.append(dict(op=op, shape=list(shape), inputs=list(inputs), attrs=attrs))
        return Array(shape, _graph=self, _id=index)

    def input(self, shape):
        value = self.node("input", shape)
        self.inputs.append(value._id)
        return value

    def lift(self, value):
        if value._graph is self:
            return value
        if value._graph is not None:
            raise ValueError("cannot mix values from different traces")
        return self.node("const", value.shape,
                         data=value._data.astype("<f4", copy=False).tobytes().hex())

    def document(self):
        return dict(version=1, nodes=self.nodes, inputs=self.inputs, outputs=self.outputs)


def primitive(op, values, shape, **attrs):
    shape = checked_shape(shape)  # Bound eager outputs before NumPy allocates them.
    values = tuple(host_array(v) for v in values)
    graphs = {id(v._graph): v._graph for v in values if v._graph is not None}
    if len(graphs) > 1:
        raise ValueError("cannot mix values from different traces")
    if graphs:
        graph = next(iter(graphs.values()))
        ids = [graph.lift(v)._id for v in values]
        return graph.node(op, shape, ids, **attrs)
    result = evaluate(op, [v._data for v in values], attrs, shape)
    return host_array(result, dtype=np.float32)


def evaluate(op, args, attrs, shape):
    """Explicit NumPy reference evaluator, never used as a device fallback."""
    functions = dict(add=np.add, sub=np.subtract, mul=np.multiply,
                     maximum=np.maximum, minimum=np.minimum, negate=np.negative,
                     abs=np.abs, exp=np.exp, log=np.log, tanh=np.tanh,
                     sin=np.sin, cos=np.cos, matmul=np.matmul)
    if op in functions:
        return functions[op](*args)
    if op == "identity":
        return args[0]
    if op == "reshape":
        return np.reshape(args[0], shape)
    if op == "transpose":
        return np.transpose(args[0], attrs["axes"])
    if op == "reduce_sum":
        return np.sum(args[0], axis=attrs["axis"], keepdims=True, dtype=np.float32)
    raise NotImplementedError(f"reference evaluator does not implement {op!r}")


"""Static tracing, bounded specialization cache, and explicit reference execution."""

from collections import OrderedDict
from functools import update_wrapper
import inspect
import json
from threading import RLock

import numpy as np

from ._array import Array, Graph, evaluate, host_array, primitive


class Lowered:
    """A frozen graph. Export validates it with the Rust TOSA implementation."""

    def __init__(self, graph, signature, specs, multiple):
        self._json = json.dumps(graph.document(), separators=(",", ":"), allow_nan=False)
        if len(self._json) > 16 * 1024 * 1024:
            raise ValueError("frontend graph exceeds the 16 MiB JSON limit")
        self._signature = signature
        self._specs = tuple(specs)
        self._multiple = multiple

    def to_dict(self):
        """Return an independent diagnostic graph, not a stable serialized ABI."""
        return json.loads(self._json)

    def to_tosa(self):
        """Return validated static TOSA 1.0 FP bytes; requires the native extension."""
        try:
            from ._native import build_tosa
        except ImportError as exc:
            raise RuntimeError("TOSA export requires the native extension; build with maturin") from exc
        return build_tosa(self._json)

    def reference(self, *args, **kwargs):
        """Evaluate with NumPy on the CPU. This is an oracle, not acceleration."""
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        supplied = tuple(host_array(v) for v in bound.arguments.values())
        if tuple((a.shape, a.dtype.str) for a in supplied) != self._specs:
            raise ValueError("inputs do not match this graph's shapes and dtypes")
        doc = self.to_dict()
        inputs = dict(zip(doc["inputs"], supplied))
        values = []
        for index, node in enumerate(doc["nodes"]):
            op, shape, attrs = node["op"], tuple(node["shape"]), node["attrs"]
            if op == "input":
                value = inputs[index].to_numpy()
            elif op == "const":
                value = np.frombuffer(bytes.fromhex(attrs["data"]), dtype="<f4").reshape(shape)
            else:
                value = evaluate(op, [values[i] for i in node["inputs"]], attrs, shape)
            values.append(value)
        outputs = tuple(host_array(values[i], dtype=np.float32) for i in doc["outputs"])
        return outputs if self._multiple else outputs[0]


class CompiledFunction:
    """Trace a pure function, specializing on input shapes and dtypes.

    The initial implementation exports TOSA and provides an opt-in CPU oracle.
    No hardware executor is selected implicitly.
    """

    def __init__(self, function, *, backend=None, cache_size=16):
        if backend not in (None, "reference"):
            raise NotImplementedError("device execution is not implemented; only backend='reference' exists")
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self._function = function
        self._signature = inspect.signature(function)
        for parameter in self._signature.parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("variadic functions are not supported")
        self._backend = backend
        self._cache_size = cache_size
        self._cache = OrderedDict()
        self._lock = RLock()
        update_wrapper(self, function)

    def lower(self, *args, **kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = tuple(host_array(v) for v in bound.arguments.values())
        if any(a._graph is not None for a in arrays):
            raise TypeError("nested jit calls are not supported in this iteration")
        specs = tuple((a.shape, a.dtype.str) for a in arrays)
        with self._lock:
            if specs in self._cache:
                self._cache.move_to_end(specs)
                return self._cache[specs]
            graph = Graph()
            for name, array in zip(bound.arguments, arrays):
                bound.arguments[name] = graph.input(array.shape)
            result = self._function(*bound.args, **bound.kwargs)
            multiple = isinstance(result, tuple)
            outputs = result if multiple else (result,)
            if not outputs:
                raise TypeError("a traced function must return at least one array")
            for output in outputs:
                output = graph.lift(host_array(output))
                # Give every graph output a distinct producer, including input passthrough.
                output = primitive("identity", (output,), output.shape)
                graph.outputs.append(output._id)
            lowered = Lowered(graph, self._signature, specs, multiple)
            self._cache[specs] = lowered
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return lowered

    def clear_cache(self):
        """Retrace on next use, including any changed captured constants."""
        with self._lock:
            self._cache.clear()

    def __call__(self, *args, **kwargs):
        if self._backend is None:
            raise RuntimeError("device execution is not implemented; use .lower(...).to_tosa() "
                               "or explicitly select jit(backend='reference')")
        return self.lower(*args, **kwargs).reference(*args, **kwargs)


def jit(function=None, *, backend=None, cache_size=16):
    """Trace/export a function; ``backend='reference'`` explicitly enables CPU evaluation."""
    def wrap(fn):
        return CompiledFunction(fn, backend=backend, cache_size=cache_size)
    return wrap if function is None else wrap(function)


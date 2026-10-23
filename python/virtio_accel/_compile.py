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

    def compile(self, device, *, timeout=30.0, strict=False):
        """Compile for a native device; use the returned executable as a context manager."""
        from ._runtime import Executable
        return Executable(self, device, timeout=timeout, strict=strict)

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
                dtype = np.dtype(node["dtype"]).newbyteorder("<")
                value = np.frombuffer(bytes.fromhex(attrs["data"]), dtype=dtype).reshape(shape)
            else:
                value = evaluate(op, [values[i] for i in node["inputs"]], attrs, shape, node["dtype"])
            values.append(value)
        outputs = tuple(host_array(values[i]) for i in doc["outputs"])
        return outputs if self._multiple else outputs[0]


class CompiledFunction:
    """Trace a pure function, specializing on input shapes and dtypes.

    Select device="npu" or "gpu" for native execution, or backend="reference"
    for the CPU oracle. Programs and buffers are cached with each specialization.
    """

    def __init__(self, function, *, backend=None, device=None, cache_size=16, timeout=30.0, strict=False):
        if backend not in (None, "reference"):
            raise NotImplementedError("backend accepts only 'reference'; use device='npu' or device='gpu' for native execution")
        from ._runtime import _selector, _timeout
        if device is not None and backend is not None:
            raise ValueError('device and backend are mutually exclusive')
        if not isinstance(strict, bool):
            raise TypeError('strict must be a bool')
        self._device = _selector(device) if device is not None else None
        self._timeout = _timeout(timeout)
        self._strict = strict
        self._executables = {}
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
                bound.arguments[name] = graph.input(array.shape, array.dtype)
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
                evicted, _ = self._cache.popitem(last=False)
                executable = self._executables.pop(evicted, None)
                if executable is not None:
                    executable.close()
            return lowered

    def clear_cache(self):
        """Retrace on next use, including any changed captured constants."""
        with self._lock:
            self._cache.clear()
            executables, self._executables = self._executables, {}
            errors = []
            for executable in executables.values():
                try:
                    executable.close()
                except RuntimeError as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))

    def __call__(self, *args, **kwargs):
        if self._device is not None:
            with self._lock:
                lowered = self.lower(*args, **kwargs)
                key = lowered._specs
                if key not in self._executables:
                    self._executables[key] = lowered.compile(self._device, timeout=self._timeout, strict=self._strict)
                return self._executables[key](*args, **kwargs)
        if self._backend is None:
            raise RuntimeError("device execution requires device=; use .lower(...).to_tosa() "
                               "or explicitly select jit(backend='reference')")
        return self.lower(*args, **kwargs).reference(*args, **kwargs)


def jit(function=None, *, backend=None, device=None, cache_size=16, timeout=30.0, strict=False):
    """Trace/export, execute with device=, or explicitly select the CPU reference."""
    def wrap(fn):
        return CompiledFunction(fn, backend=backend, device=device, cache_size=cache_size, timeout=timeout, strict=strict)
    return wrap if function is None else wrap(function)


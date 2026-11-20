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
        if tuple((a.shape, a.dtype.name) for a in supplied) != self._specs:
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

    Select device="cpu", "npu", "gpu", or a discovered Device for native execution, or backend="reference"
    for the CPU oracle. Programs and buffers are cached with each specialization.
    """

    def __init__(self, function, *, backend=None, device=None, cache_size=16, resident_cache_size=None, timeout=30.0, strict=False):
        if backend not in (None, "reference"):
            raise NotImplementedError("backend accepts only 'reference'; use device= for native execution")
        from ._runtime import _selector, _timeout
        if device is not None and backend is not None:
            raise ValueError('device and backend are mutually exclusive')
        if not isinstance(strict, bool):
            raise TypeError('strict must be a bool')
        self._device = _selector(device) if device is not None else None
        self._timeout = _timeout(timeout)
        self._strict = strict
        self._executables = OrderedDict()
        self._closed = False
        if resident_cache_size is not None and (
            isinstance(resident_cache_size, bool) or not isinstance(resident_cache_size, int) or resident_cache_size < 1
        ):
            raise ValueError("resident_cache_size must be a positive integer or None")
        self._resident_cache_size = resident_cache_size
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
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
        """Trace or reuse a graph specialized to the input shapes and dtypes.

        The Python function runs only when a specialization is first traced.
        Captured values are snapshotted; clear the cache to capture changes."""
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = tuple(host_array(v) for v in bound.arguments.values())
        if any(a._graph is not None for a in arrays):
            raise TypeError("nested jit calls are not supported in this iteration")
        specs = tuple((a.shape, a.dtype.name) for a in arrays)
        with self._lock:
            if self._closed:
                raise RuntimeError("compiled function is closed")
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
                # Inputs and duplicate outputs need distinct producers. Existing operation
                # outputs are already distinct; avoid imposing redundant backend operators.
                if graph.nodes[output._id]['op'] == 'input' or output._id in graph.outputs:
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
            executables, self._executables = self._executables, OrderedDict()
            errors = []
            for executable in executables.values():
                try:
                    executable.close()
                except (RuntimeError, OSError) as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))

    def close(self):
        """Release every resident program and permanently close this function."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.clear_cache()

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("compiled function is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, *args, **kwargs):
        """Run using the explicitly selected device or CPU reference backend."""
        with self._lock:
            if self._closed:
                raise RuntimeError("compiled function is closed")
            if self._device is None and self._backend is None:
                raise RuntimeError("device execution requires device=; use .lower(...).to_tosa() "
                                   "or explicitly select jit(backend='reference')")
            # Snapshot once. lower() and Executable accept these immutable Arrays
            # without copying the original NumPy inputs a second time.
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for name, value in bound.arguments.items():
                bound.arguments[name] = host_array(value)
            lowered = self.lower(*bound.args, **bound.kwargs)
            if self._device is not None:
                key = lowered._specs
                if key not in self._executables:
                    provider = getattr(self._device, "provider", None)
                    if self._executables:
                        resident = next(iter(self._executables.values()))
                        provider = getattr(getattr(resident, "device", None), "provider", provider)
                    limit = self._resident_cache_size
                    if limit is None:
                        limit = 1 if provider == "virtio" else self._cache_size
                    # Retire before compiling: the current native vhost budget
                    # admits one program. Traced graphs remain in their own LRU.
                    while len(self._executables) >= limit:
                        _, executable = self._executables.popitem(last=False)
                        executable.close()
                    self._executables[key] = lowered.compile(self._device, timeout=self._timeout, strict=self._strict)
                self._executables.move_to_end(key)
                return self._executables[key](*bound.args, **bound.kwargs)
            return lowered.reference(*bound.args, **bound.kwargs)


def jit(function=None, *, backend=None, device=None, cache_size=16, resident_cache_size=None, timeout=30.0, strict=False):
    """Decorate a pure function for static tracing and explicit execution.

    Args:
        function: Function to wrap, or None when used as a decorator factory.
        backend: Set to ``"reference"`` for the NumPy CPU oracle. Mutually
            exclusive with ``device``. There is no automatic CPU fallback.
        device: A discovered Device or selector such as ``"gpu"``, ``"npu"``,
            ``"cpu"`` or ``"hybrid"``. Bare kinds select index zero.
        cache_size: Maximum cached shape/dtype specializations (default 16).
        resident_cache_size: Maximum resident executables. By default, one for
            virtio endpoints and ``cache_size`` for native endpoints. Eviction
            releases programs before compilation; traced graphs stay cached.
        timeout: Native-operation timeout in seconds, greater than zero and
            no more than 86400. A timeout does not cancel native work.
        strict: Reject accelerator endpoints that permit CPU participation.

    Returns:
        A CompiledFunction, or a decorator when function is omitted.

    With neither device nor backend, only lowering/export is enabled. Functions
    must be pure and return one Array or a nonempty tuple of Arrays. Python
    control flow may depend on static shapes, but never on tensor values.
    """
    def wrap(fn):
        return CompiledFunction(fn, backend=backend, device=device, cache_size=cache_size, resident_cache_size=resident_cache_size, timeout=timeout, strict=strict)
    return wrap if function is None else wrap(function)


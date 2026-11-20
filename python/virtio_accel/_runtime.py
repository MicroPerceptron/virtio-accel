"""Portable device selection and synchronous native execution with explicit host copies."""
from dataclasses import asdict, dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import math
import re
import sys
from threading import RLock

import numpy as np

from ._array import host_array


@dataclass(frozen=True)
class Device:
    """Immutable endpoint discovered by :func:`virtio_accel.devices`.

    IDs are local to a discovery snapshot. Multiple providers can expose the
    same hardware. CPU participation is a placement allowance, not a claim
    about which processors executed a particular graph.
    """
    #: Snapshot-local selector, for example ``gpu:0``.
    id: str
    #: ``cpu``, ``gpu``, ``npu`` or ``hybrid``.
    kind: str
    #: ``coreml``, ``vulkan``, ``openvino``, ``hexagon``, ``xdna`` or ``virtio``.
    provider: str
    #: Human-readable endpoint name.
    name: str
    #: Whether CPU participation is permitted or cannot be excluded by the transport.
    cpu_participation: bool
    # Retain the discovery scope so filtered IDs can be revalidated unchanged.
    _identity: tuple | None = field(default=None, repr=False)
    _discovery_provider: str | None = field(default=None, repr=False, compare=False)


def _native():
    try:
        from . import _native as native
    except ImportError as exc:
        raise RuntimeError('device execution requires the native extension; build with maturin') from exc
    return native


def _selector(value):
    if isinstance(value, Device):
        return value
    if not isinstance(value, str) or re.fullmatch(r'(cpu|gpu|npu|hybrid)(?::(?:0|[1-9][0-9]*))?', value) is None:
        raise ValueError('device must be cpu, gpu, npu, hybrid, or an indexed form such as gpu:0')
    return value if ':' in value else value + ':0'


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 86400:
        raise ValueError('timeout must be finite and in (0, 86400] seconds')
    return float(value)


def _openvino_library():
    # Respect explicit process configuration. Do not mutate environment variables
    # while native worker threads may be reading them.
    if any(name in os.environ for name in (
        'VIRTIO_ACCEL_OPENVINO_LIBRARY', 'VIRTIO_ACCEL_OPENVINO_LIB_DIR',
    )):
        return None
    try:
        spec = importlib.util.find_spec('openvino')
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    candidates = set()
    for location in spec.submodule_search_locations:
        directory = Path(location) / 'libs'
        for pattern in ('libopenvino_c.so*', 'libopenvino_c*.dylib', 'openvino_c.dll'):
            candidates.update(path.resolve() for path in directory.glob(pattern) if path.is_file())
    # Refuse ambiguous package contents instead of choosing a version arbitrarily.
    return str(next(iter(candidates))) if len(candidates) == 1 else None


def _discovery(provider=None, *, refresh=False):
    result = (dict(devices=[], diagnostics=[]) if provider == "virtio" else
              json.loads(_native().discover_devices(
                  _openvino_library() if provider in (None, "openvino") else None, provider)))
    if sys.platform == 'linux' and provider in (None, "virtio"):
        from ._virtio import discover
        discover(result, refresh=refresh)
    return result


def devices(kind=None, *, provider=None):
    """Discover endpoints. Indices belong to this provider-filtered snapshot.

    Filter by kind or provider, and pass a returned Device to jit/compile to select
    that endpoint. Multiple providers may expose the same physical accelerator.
    Virtio metadata is cached until its node changes; execution revalidates identity.
    """
    if kind not in (None, 'cpu', 'gpu', 'npu', 'hybrid'):
        raise ValueError('kind must be cpu, gpu, npu, hybrid, or None')
    if provider not in (None, 'coreml', 'vulkan', 'openvino', 'hexagon', 'xdna', 'virtio'):
        raise ValueError('unknown device provider')
    return tuple(Device(**d, _discovery_provider=provider) for d in _discovery(provider)['devices']
                 if (kind is None or d['kind'] == kind)
                 and (provider is None or d['provider'] == provider))


def device_diagnostics():
    """Explain missing runtimes, failed probes and excluded native endpoints."""
    return tuple(_discovery(refresh=True)['diagnostics'])


def _resolve(selector, strict):
    identifier = _selector(selector)
    scope = identifier._discovery_provider if isinstance(identifier, Device) else None
    discovery = _discovery(scope)
    for entry in discovery['devices']:
        device = Device(**entry, _discovery_provider=scope)
        if (device == identifier if isinstance(identifier, Device) else device.id == identifier):
            if strict and device.kind != 'cpu' and device.cpu_participation:
                raise RuntimeError(f'{identifier} via {device.provider} cannot guarantee exclusive {device.kind} placement')
            return device
    reasons = '; '.join(discovery['diagnostics'])
    raise RuntimeError(f'device {identifier} is unavailable; no fallback was selected' + (f' ({reasons})' if reasons else ''))


class Executable:
    """Resident program and reusable native buffers; results are independent host arrays.

    Close explicitly or use a context manager to observe release failures. A wait
    timeout closes this executable without cancelling or retrying native work.
    """
    def __init__(self, lowered, device, *, timeout=30.0, strict=False):
        if not isinstance(strict, bool):
            raise TypeError('strict must be a bool')
        self._timeout = _timeout(timeout)
        self.device = _resolve(device, strict)
        self._lowered = lowered
        doc = lowered.to_dict()
        self._outputs = tuple((np.dtype(doc["nodes"][i]["dtype"]).newbyteorder("<"),
                               tuple(doc["nodes"][i]["shape"])) for i in doc["outputs"])
        self._lock = RLock()
        self._closed = False
        if self.device.provider == 'virtio':
            from ._virtio import VirtioExecutable
            self._native = VirtioExecutable(self.device, lowered, self._timeout)
        else:
            self._native = _native().NativeExecutable(json.dumps({k: v for k, v in asdict(self.device).items() if not k.startswith("_")}), lowered._json, self._timeout)

    def __call__(self, *args, **kwargs):
        """Execute with the compiled shapes and dtypes, returning independent host arrays.

        Calls are serialized per executable. Results are an Array or a tuple
        matching the traced return structure. Use ``to_numpy()`` for host copies."""
        with self._lock:
            if self._closed:
                raise RuntimeError('executable is closed')
            bound = self._lowered._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arrays = tuple(host_array(v) for v in bound.arguments.values())
            if tuple((a.shape, a.dtype.name) for a in arrays) != self._lowered._specs:
                raise ValueError("inputs do not match this graph's shapes and dtypes")
            payloads = [a._data.astype(a.dtype.newbyteorder('<'), copy=False).tobytes() for a in arrays]
            outputs = self._native.run(payloads, self._timeout)
            if len(outputs) != len(self._outputs):
                raise RuntimeError('native output count does not match compiled graph')
            result = []
            for data, (dtype, shape) in zip(outputs, self._outputs):
                array = np.frombuffer(data, dtype=dtype).reshape(shape)
                result.append(host_array(array))
            return tuple(result) if self._lowered._multiple else result[0]

    def close(self):
        """Release native programs and buffers; repeated calls have no effect.

        Use a context manager or call explicitly to observe release errors."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._native.close(self._timeout)

    def __enter__(self):
        if self._closed:
            raise RuntimeError('executable is closed')
        return self

    def __exit__(self, *exc):
        self.close()

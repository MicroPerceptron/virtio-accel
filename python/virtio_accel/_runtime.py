"""Portable device selection and synchronous native execution with explicit host copies."""
from dataclasses import asdict, dataclass
import json
import math
import re
from threading import RLock

import numpy as np

from ._array import host_array


@dataclass(frozen=True)
class Device:
    id: str
    kind: str
    provider: str
    name: str
    cpu_participation: bool


def _native():
    try:
        from . import _native as native
    except ImportError as exc:
        raise RuntimeError('device execution requires the native extension; build with maturin') from exc
    return native


def _selector(value):
    if not isinstance(value, str) or re.fullmatch(r'(cpu|gpu|npu)(?::(?:0|[1-9][0-9]*))?', value) is None:
        raise ValueError('device must be cpu, gpu, npu, or an indexed form such as gpu:0')
    return value if ':' in value else value + ':0'


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 86400:
        raise ValueError('timeout must be finite and in (0, 86400] seconds')
    return float(value)


def _discovery():
    return json.loads(_native().discover_devices())


def devices(kind=None):
    """Discover usable native endpoints. Indices belong to this discovery snapshot.

    Core ML supplies npu endpoints with possible CPU participation. Vulkan supplies
    gpu endpoints, excluding software CPU devices and ambiguous duplicate names.
    """
    if kind not in (None, 'cpu', 'gpu', 'npu'):
        raise ValueError('kind must be cpu, gpu, npu, or None')
    return tuple(Device(**d) for d in _discovery()['devices'] if kind is None or d['kind'] == kind)


def device_diagnostics():
    """Explain missing runtimes, failed probes and excluded native endpoints."""
    return tuple(_discovery()['diagnostics'])


def _resolve(selector, strict):
    identifier = _selector(selector)
    discovery = _discovery()
    for entry in discovery['devices']:
        if entry['id'] == identifier:
            device = Device(**entry)
            if strict and device.cpu_participation:
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
        self._lock = RLock()
        self._closed = False
        self._native = _native().NativeExecutable(json.dumps(asdict(self.device)), lowered._json, self._timeout)

    def __call__(self, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError('executable is closed')
            bound = self._lowered._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arrays = tuple(host_array(v) for v in bound.arguments.values())
            if tuple((a.shape, a.dtype.str) for a in arrays) != self._lowered._specs:
                raise ValueError("inputs do not match this graph's shapes and dtypes")
            payloads = [a.to_numpy().astype(a.dtype.newbyteorder('<'), copy=False).tobytes() for a in arrays]
            outputs = self._native.run(payloads, self._timeout)
            doc = self._lowered.to_dict()
            if len(outputs) != len(doc['outputs']):
                raise RuntimeError('native output count does not match compiled graph')
            result = []
            for data, index in zip(outputs, doc['outputs']):
                node = doc['nodes'][index]
                array = np.frombuffer(data, dtype=np.dtype(node['dtype']).newbyteorder('<')).reshape(node['shape'])
                result.append(host_array(array))
            return tuple(result) if self._lowered._multiple else result[0]

    def close(self):
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

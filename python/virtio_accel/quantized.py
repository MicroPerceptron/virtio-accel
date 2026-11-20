"""Explicit low-bit projections, composable inside :func:`virtio_accel.jit`.

INT4, INT8 and palette weights are immutable captured constants. NVFP4 weights
and block scales can be runtime INT8 byte carriers. None of these helpers
quantizes floating weights or selects a device automatically.
"""
import operator
import numpy as np
from ml_dtypes import float8_e4m3fn

from . import numpy as vnp
from ._array import Array, checked_shape, host_array, primitive

__all__ = ["int4_matmul", "int8_matmul", "palette4_matmul", "nvfp4_matmul"]


def _activation(x, *, nvfp4=False):
    x = host_array(x)
    if x.dtype not in ((np.dtype('float16'), np.dtype('float32')) if nvfp4 else (np.dtype('float16'),)):
        raise TypeError('activations must be FP16' + (' or FP32' if nvfp4 else ''))
    if x.ndim not in (2, 3) or (x.ndim == 3 and x.shape[0] != 1):
        raise ValueError('activations must have shape [M,K] or [1,M,K]')
    return x


def _constant(value, dtype, name):
    value = host_array(value)
    if value._graph is not None:
        raise TypeError(f'{name} must be a captured constant, not a runtime tensor')
    if value.dtype != np.dtype(dtype):
        raise TypeError(f'{name} must have dtype {np.dtype(dtype).name}')
    return value.to_numpy()


def _geometry(x, n, group):
    group = operator.index(group)
    k = x.shape[-1]
    if group <= 0 or k % group:
        raise ValueError('group_size must be positive and divide K')
    checked_shape((n, k), np.int32)  # Bound unpacked index/weight temporaries too.
    checked_shape((1, x.shape[-2], n), x.dtype)
    return k, group


def _unpack(packed, count):
    if count % 2:
        raise ValueError('N*K must be even')
    if isinstance(packed, (bytes, bytearray, memoryview)):
        if memoryview(packed).nbytes != count // 2:
            raise ValueError('packed byte count must equal N*K/2')
        raw = np.frombuffer(bytes(packed), dtype=np.uint8)
    elif isinstance(packed, np.ndarray) and packed.dtype == np.uint8:
        if packed.size != count // 2:
            raise ValueError('packed byte count must equal N*K/2')
        raw = packed.reshape(-1)
    else:
        raise TypeError('packed values must be bytes or a uint8 ndarray')
    if raw.size != count // 2:
        raise ValueError('packed byte count must equal N*K/2')
    values = np.empty(count, dtype=np.int32)
    values[::2], values[1::2] = raw & 15, raw >> 4
    return values


def _lift(value, *anchors):
    value = host_array(value)
    for anchor in anchors:
        if anchor._graph is not None:
            return anchor._graph.lift(value)
    return value


def _project(x, weights, n):
    k = x.shape[-1]
    rhs = vnp.transpose(vnp.reshape(weights, (1, n, k)), (0, 2, 1))
    a = vnp.reshape(x, (1, x.shape[-2], k)) if x.ndim == 2 else x
    result = vnp.matmul(a, rhs)
    return vnp.reshape(result, (x.shape[-2], n)) if x.ndim == 2 else result


def _integer(x, weights, scales, group):
    n, k = weights.shape
    weights = _lift(weights.reshape(n, k // group, group), x)
    scales = _lift(scales.reshape(n, k // group, 1), x)
    return _project(x, vnp.multiply(weights.astype(np.float16), scales), n)


def int4_matmul(x, packed_weights, scales, *, group_size):
    """FP16 projection with captured signed INT4 weights, low nibble first.

    Weights are row-major [N,K], two's complement [-8,7]. ``scales`` must be a
    positive finite FP16 ndarray [N,K/group_size]. Output is [M,N] or [1,M,N].
    Each decoded weight rounds to FP16 before the FP16 MATMUL. Core ML and
    OpenVINO may repack these constants at load time; placement is policy dependent.
    """
    x = _activation(x)
    scales = _constant(scales, np.float16, 'scales')
    if scales.ndim != 2:
        raise ValueError('scales must have shape [N,K/group_size]')
    n = scales.shape[0]
    k, group = _geometry(x, n, group_size)
    _scales(scales, n, k, group)
    values = _unpack(packed_weights, n * k)
    weights = np.where(values >= 8, values - 16, values).astype(np.int8).reshape(n, k)
    return _integer(x, weights, scales, group)


def _scales(scales, n, k, group):
    if scales.shape != (n, k // group):
        raise ValueError('scales must have shape [N,K/group_size]')
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError('scales must be positive and finite')


def int8_matmul(x, weights, scales, *, group_size):
    """FP16 projection with captured INT8 weights [N,K] and FP16 group scales.

    Uses the same scale geometry and rounding as :func:`int4_matmul`.
    For integer activations and INT32 output use ``numpy.matmul`` instead.
    """
    x = _activation(x)
    weights = _constant(weights, np.int8, 'weights')
    scales = _constant(scales, np.float16, 'scales')
    if weights.ndim != 2 or weights.shape[1] != x.shape[-1]:
        raise ValueError('weights must have shape [N,K]')
    n = weights.shape[0]
    k, group = _geometry(x, n, group_size)
    _scales(scales, n, k, group)
    return _integer(x, weights, scales, group)


def palette4_matmul(x, packed_indices, palettes, *, group_size):
    """FP16 projection with captured four-bit indices and arbitrary FP16 palettes.

    Indices are row-major [N,K], low nibble first. ``palettes`` is a finite FP16
    ndarray [N,K/group_size,16]. Values include signed zero; no scale or
    clustering is applied. Core ML recognizes this constant lookup pattern.
    """
    x = _activation(x)
    palettes = _constant(palettes, np.float16, 'palettes')
    if palettes.ndim != 3:
        raise ValueError('palettes must have shape [N,K/group_size,16]')
    n = palettes.shape[0]
    k, group = _geometry(x, n, group_size)
    if palettes.shape != (n, k // group, 16) or not np.all(np.isfinite(palettes)):
        raise ValueError('palettes must be finite with shape [N,K/group_size,16]')
    count = n * k // group
    indices = _lift(_unpack(packed_indices, n * k).reshape(count, group), x)
    table = _lift(palettes.reshape(count, 16, 1), x)
    decoded = primitive('gather', (table, indices), (count, group, 1))
    return _project(x, decoded, n)


def nvfp4_matmul(x, packed_weights, block_scales, *, tensor_scale=1.0):
    """Project runtime NVFP4 weights with FP16 or FP32 activations/output.

    K must be divisible by 16. ``packed_weights`` is an INT8 byte carrier
    [1,N*K/2], and ``block_scales`` an INT8 carrier [1,N*K/16] holding raw
    E4M3FN bits. Use ``uint8_array.view(np.int8)`` to preserve byte encodings.
    E2M1 weights are low nibble first in row-major [N,K]. Decode, apply block
    scales, MATMUL, then apply the separate positive finite tensor scale in the
    activation dtype. Byte inputs remain runtime tensors and can change per call.
    """
    x = _activation(x, nvfp4=True)
    weights, scales = host_array(packed_weights), host_array(block_scales)
    k = x.shape[-1]
    if k % 16 or weights.ndim != 2 or weights.shape[0] != 1 or weights.size * 2 % k:
        raise ValueError('K must divide the packed weight geometry and be a multiple of 16')
    n = weights.size * 2 // k
    _geometry(x, n, 16)
    if weights.dtype != np.dtype('int8') or scales.dtype != np.dtype('int8'):
        raise TypeError('NVFP4 weights and block scales must be INT8 byte carriers')
    if scales.shape != (1, n * k // 16):
        raise ValueError('block_scales must have shape [1,N*K/16]')
    if isinstance(tensor_scale, Array) or np.ndim(tensor_scale) != 0:
        raise TypeError('tensor_scale must be a static scalar')
    with np.errstate(over='ignore', invalid='ignore'):
        scale = np.asarray(tensor_scale, dtype=x.dtype)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('tensor_scale must be positive and finite in the activation dtype')
    anchors = (x, weights, scales)
    # Signed INT8 -> INT32 + 128 proves lookup bounds to backend admission.
    def lookup(value, table):
        indices = primitive('add', (value.astype(np.int32), _lift(np.array([[128]], np.int32), *anchors)), value.shape)
        table = _lift(np.asarray(table, dtype=x.dtype).reshape(1, 256, 1), *anchors)
        return primitive('gather', (table, indices), (1, value.size, 1))
    raw = np.arange(-128, 128, dtype=np.int16).astype(np.uint8)
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    lo, hi = lookup(weights, levels[raw & 15]), lookup(weights, levels[raw >> 4])
    decoded = primitive('concat', (lo, hi), (1, weights.size, 2), axis=2)
    decoded = vnp.reshape(decoded, (n, k // 16, 16))
    scales = vnp.reshape(lookup(scales, raw.view(float8_e4m3fn)), (n, k // 16, 1))
    product = _project(x, vnp.multiply(decoded, scales), n)
    return vnp.multiply(product, scale)

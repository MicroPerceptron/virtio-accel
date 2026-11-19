"""Resolve opt-in hardware tests without silently changing providers."""
import os
import virtio_accel as va
from virtio_accel import _runtime


def requested_device():
    selector = os.environ['VIRTIO_ACCEL_PYTHON_TEST_DEVICE']
    provider = os.environ.get('VIRTIO_ACCEL_PYTHON_TEST_PROVIDER')
    if provider and ':' not in selector:
        candidates = va.devices(selector, provider=provider)
        if not candidates:
            raise RuntimeError(f'requested {provider} {selector} is unavailable: {va.device_diagnostics()}')
        return candidates[0]
    record = _runtime._resolve(selector, False)
    if provider and record.provider != provider:
        raise RuntimeError(f'requested {provider}, but {selector} belongs to {record.provider}')
    return record

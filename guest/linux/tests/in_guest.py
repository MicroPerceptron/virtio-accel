"""Runs as root ONLY inside the disposable test VM, never on the host."""

import errno
import fcntl
import os
import struct
import time
from pathlib import Path

import numpy as np
import virtio_accel as va
from virtio_accel import _virtio as v


def expect_errno(number, action):
    try:
        action()
    except OSError as error:
        assert error.errno == number, error
    else:
        raise AssertionError(f"expected errno {number}")


def driver_contracts():
    path = "/dev/virtio-accel0"
    s = v.Session(path)
    try:
        expect_errno(errno.EBUSY, lambda: os.open(path, os.O_RDWR))
        expect_errno(errno.ENOTTY, lambda: fcntl.ioctl(s.fd, 0))
        expect_errno(errno.EFAULT, lambda: fcntl.ioctl(s.fd, v.EXCHANGE, 0))
        header = bytearray(v.IOCTL.pack(16, 16, 100, 1, 0, 0, 0, 0) + bytes(16))
        expect_errno(errno.EINVAL, lambda: fcntl.ioctl(s.fd, v.EXCHANGE, header, True))
        context = s.object(0x100, bytes(8), 10)
        # More than one page, then larger than a single ioctl payload.
        data = bytes(range(256)) * (s.max_request // 256 + 17)
        buffer = s.object(
            0x200, context + struct.pack("<QQB7xII", len(data), 1, 1, 3, 0), 10
        )
        chunk = s.max_request - 40
        for offset in range(0, len(data), chunk):
            tail = data[offset : offset + chunk]
            s.request(0x202, buffer + struct.pack("<QQ", offset, len(tail)) + tail)
        output = []
        chunk = s.max_response - 16
        for offset in range(0, len(data), chunk):
            count = min(chunk, len(data) - offset)
            output.append(
                s.request(0x203, buffer + struct.pack("<QQ", offset, count), size=count)
            )
        assert b"".join(output) == data
        print(
            "PASS: exclusive open, ioctl validation, paged SG and chunked buffer copies",
            flush=True,
        )
    finally:
        s.poison()  # leave live resources to verify session reset
    s = v.Session(path)
    try:
        try:
            s.request(0x101, context)
        except RuntimeError as error:
            assert "status 11" in str(error), error
        else:
            raise AssertionError("old context survived close/reset")
        replacement = s.object(0x100, bytes(8), 10)
        assert replacement != context
        s.request(0x101, replacement)
    finally:
        s.poison()
    print("PASS: close/reopen reset rejects stale IDs", flush=True)


def sdk_execution():
    devices = va.devices(provider="virtio")
    assert len(devices) == 1, va.device_diagnostics()
    device = devices[0]
    assert device.cpu_participation
    # Subsequent discovery must not open/close the idle device and consume a
    # reset namespace. Session creation still validates the cached identity.
    from unittest.mock import patch

    with patch.object(v, "acquire", side_effect=AssertionError("idle reprobe")):
        assert va.devices(provider="virtio") == devices
    x = np.arange(32, dtype=np.float32).reshape(1, 32) / 8
    for function, expected in ((lambda a: a, x), (lambda a: a + a, x + x)):
        compiled = va.jit(function, device=device)
        try:
            for _ in range(3):
                np.testing.assert_array_equal(compiled(x).to_numpy(), expected)
            assert (
                va.devices(provider="virtio") == devices
            )  # probe shares active session
        finally:
            compiled.clear_cache()
    # Multiple executables can share the fd without one close resetting its peer.
    lowered = va.jit(lambda a: a).lower(x)
    with lowered.compile(device) as executable:
        np.testing.assert_array_equal(executable(x).to_numpy(), x)
        try:
            lowered.compile(device, strict=True)
        except RuntimeError as error:
            assert "exclusive" in str(error), error
        else:
            raise AssertionError("unverifiable strict placement was accepted")
    traces = []

    def changing_shape(a):
        traces.append(a.shape)
        return a + a

    with va.jit(changing_shape, device=device) as compiled:
        for size in (32, 64, 32):
            values = np.arange(size, dtype=np.float32).reshape(1, size)
            np.testing.assert_array_equal(compiled(values).to_numpy(), values + values)
        assert traces == [(1, 32), (1, 64)]
    print(
        "PASS: shape changes retire resident programs while retaining traced graphs",
        flush=True,
    )
    print(
        f"PASS: Python SDK JIT identity/add, cached runs, discovery and close via {device}",
        flush=True,
    )


def polling_measurements():
    """Compare policies on the same resident graph; report, never gate on timing."""
    import statistics

    from virtio_accel import _native

    adaptive = _native._PollWait

    class FixedWait:
        def wait(self, remaining):
            time.sleep(min(0.001, remaining))

    (device,) = va.devices(provider="virtio")
    values = np.arange(32, dtype=np.float32).reshape(1, 32)
    try:
        with va.jit(lambda a: a + a, device=device) as compiled:
            compiled(values)
            for label, policy in (("fixed-1ms", FixedWait), ("adaptive", adaptive)):
                _native._PollWait = policy
                samples = []
                cpu = time.process_time()
                for _ in range(64):
                    started = time.perf_counter()
                    result = compiled(values)
                    samples.append((time.perf_counter() - started) * 1e6)
                    np.testing.assert_array_equal(result.to_numpy(), values + values)
                cpu = (time.process_time() - cpu) * 1e6 / len(samples)
                print(
                    f"POLL_BENCH {label}: median_us={statistics.median(samples):.1f} "
                    f"p95_us={sorted(samples)[60]:.1f} guest_cpu_us_per_call={cpu:.1f}",
                    flush=True,
                )
    finally:
        _native._PollWait = adaptive


def timeout_contracts():
    s = v.Session("/dev/virtio-accel0")
    print("VIRTIO_ACCEL_PAUSE_HOST", flush=True)
    time.sleep(0.2)  # harness pauses only this test daemon for two seconds
    expect_errno(errno.ETIMEDOUT, lambda: s.request(1, size=76, timeout=0.02))
    assert s.fd is None
    s = v.Session("/dev/virtio-accel0")
    s.poison()
    print(
        "PASS: timeout quiesces published buffers and permits clean reopen", flush=True
    )


def malformed_frame_rejection():
    s = v.Session("/dev/virtio-accel0")
    # Header declares a payload absent from the readable descriptors. The host
    # returns a structured protocol rejection, without losing the session.
    data = bytearray(
        v.IOCTL.pack(16, 16, 1000, 0, 0, 0, 0, 0) + v.HEADER.pack(1, 0, 1, 1)
    )
    fcntl.ioctl(s.fd, v.EXCHANGE, data, True)
    assert v.IOCTL.unpack_from(data)[1] == 16
    assert v.HEADER.unpack_from(data, 32) == (3, 0, 0, 1)
    assert len(s.request(1, size=76)) == 76
    s.poison()
    print("PASS: malformed frame rejected without losing the session", flush=True)


def remove_contracts():
    s = v.Session("/dev/virtio-accel0")
    device = next(
        p for p in Path("/sys/bus/virtio/drivers/virtio_accel").glob("virtio[0-9]*")
    )
    (device.parent / "unbind").write_text(device.name)
    expect_errno(errno.ENODEV, lambda: fcntl.ioctl(s.fd, v.INFO, bytearray(32), True))
    s.poison()
    (device.parent / "bind").write_text(device.name)
    s = v.Session("/dev/virtio-accel0")
    s.poison()
    print("PASS: driver unbind with open fd, safe removal and rebind", flush=True)


if __name__ == "__main__":
    driver_contracts()
    if os.environ.get("VA_TEST_BACKEND") != "mock":
        sdk_execution()
        polling_measurements()
    timeout_contracts()
    malformed_frame_rejection()
    remove_contracts()
    print("VIRTIO_ACCEL_GUEST_PASS", flush=True)

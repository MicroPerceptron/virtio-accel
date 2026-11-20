"""Linux ioctl adapter for frozen protocol 1.0 (explicit copies, no replay).

The module is imported only on Linux. OS details deliberately stay outside the
portable Rust crates. The native compiler owns TOSA serialization and targets.
"""

import fcntl
import math
import os
import stat
import struct
import time
from pathlib import Path
from threading import RLock

# asm-generic ioctl encoding used by the supported Linux x86/ARM wheels.
EXCHANGE = 0xC020B700
INFO = 0x8020B701
IOCTL = struct.Struct("=8I")
HEADER = struct.Struct("<HHIQ")
DEVICE_INFO = struct.Struct("<16sHHIIQ6IQQ")
_sessions = {}
_registry_lock = RLock()


def _after_fork():
    # Inherited executables fail their PID check before touching inherited locks.
    # New processes must exec before opening the same exclusive device.
    global _registry_lock, _sessions
    _registry_lock = RLock()
    _sessions = {}


os.register_at_fork(after_in_child=_after_fork)


class ProtocolError(RuntimeError):
    pass


class Session:
    def __init__(self, path):
        self.path = path
        self.pid = os.getpid()
        self.lock = RLock()
        self.references = 0
        self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        self.sequence = 0
        try:
            if not stat.S_ISCHR(os.fstat(self.fd).st_mode):
                raise RuntimeError(f"{path} is not a character device")
            info = bytearray(IOCTL.size)
            fcntl.ioctl(self.fd, INFO, info, True)
            abi, self.max_request, self.max_response, major, minor, *reserved = (
                IOCTL.unpack(info)
            )
            if (abi, major, minor) != (1, 1, 0) or any(reserved):
                raise ProtocolError("unsupported Linux driver ABI or device protocol")
            if not (
                97 <= self.max_request <= 1 << 20 and 92 <= self.max_response <= 1 << 20
            ):
                raise ProtocolError("invalid Linux driver transfer bounds")
            payload = self.request(1, size=DEVICE_INFO.size)
            (
                self.uuid,
                self.device_class,
                reserved,
                self.vendor,
                self.device,
                self.capabilities,
                *limits,
            ) = DEVICE_INFO.unpack(payload)
            if reserved or self.capabilities & 0x18 or not self.capabilities & 0x23:
                raise ProtocolError("invalid device capabilities")
            if not all(limits) or limits[5] > 4096:
                raise ProtocolError("invalid device limits")
            self.max_bindings, self.max_buffer, self.max_artifact = limits[5:]
        except BaseException:
            self.poison()
            raise

    def check(self):
        if self.pid != os.getpid():
            raise RuntimeError(
                "virtio sessions cannot be used after fork; exec a new process"
            )
        if self.fd is None:
            raise RuntimeError(
                "virtio session is lost; close its executables before reopening"
            )

    def poison(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)  # final descriptor close resets/quiesces the entire device

    def request(self, opcode, payload=b"", *, size=0, timeout=30.0):
        self.check()
        with self.lock:
            self.check()
            if (
                len(payload) + HEADER.size > self.max_request
                or size + HEADER.size > self.max_response
            ):
                raise ValueError(
                    "request exceeds the Linux driver's bounded transfer limit"
                )
            if self.sequence == (1 << 64) - 1:
                self.poison()
                raise RuntimeError("virtio request IDs exhausted")
            self.sequence += 1
            request = HEADER.pack(opcode, 0, len(payload), self.sequence) + payload
            capacity = HEADER.size + size
            data = bytearray(IOCTL.size + max(len(request), capacity))
            IOCTL.pack_into(
                data,
                0,
                len(request),
                capacity,
                max(1, min(60000, math.ceil(timeout * 1000))),
                0,
                0,
                0,
                0,
                0,
            )
            data[IOCTL.size : IOCTL.size + len(request)] = request
            try:
                fcntl.ioctl(self.fd, EXCHANGE, data, True)
                request_len, used, milliseconds, flags, *reserved = IOCTL.unpack_from(
                    data
                )
                if (
                    request_len != len(request)
                    or not 16 <= used <= capacity
                    or not milliseconds
                    or flags
                    or any(reserved)
                ):
                    raise ProtocolError("invalid ioctl completion")
                status, flags, length, sequence = HEADER.unpack_from(data, IOCTL.size)
                if flags or sequence != self.sequence or length != used - HEADER.size:
                    raise ProtocolError("invalid response header")
                result = bytes(data[IOCTL.size + HEADER.size : IOCTL.size + used])
                if status:
                    # SUBMIT/DEVICE_LOST can carry an indeterminate event. Never
                    # replay it, even when the provider reports a nonzero ID.
                    if status == 9 or status not in (*range(12), 65535):
                        raise ProtocolError(f"device lost or unknown status {status}")
                    if result:
                        raise ProtocolError("unexpected failure payload")
                    raise RuntimeError(
                        f"virtio opcode {opcode:#x}: status {status}; no fallback selected"
                    )
                if len(result) != size:
                    raise ProtocolError("unexpected response payload length")
                return result
            except (OSError, ProtocolError):
                self.poison()
                raise

    def object(self, opcode, payload, timeout):
        result = self.request(opcode, payload, size=8, timeout=timeout)
        (identifier,) = struct.unpack("<Q", result)
        if not identifier:
            self.poison()
            raise ProtocolError("zero object ID")
        return result


def acquire(path):
    with _registry_lock:
        session = _sessions.get(path)
        if session is None or session.fd is None:
            session = Session(path)
            _sessions[path] = session
        session.check()
        session.references += 1
        return session


def release(session):
    with _registry_lock:
        session.references -= 1
        if session.references == 0:
            with session.lock:
                session.poison()
            if _sessions.get(session.path) is session:
                del _sessions[session.path]


def discover(result):
    for path in sorted(Path("/dev").glob("virtio-accel[0-9]*")):
        session = None
        try:
            session = acquire(str(path))
            # OTHER is not evidence of CPU placement. Protocol 1.0 has no CPU
            # participation bit or provider-name string; report conservatively.
            kind = {1: "npu", 2: "gpu"}.get(session.device_class, "hybrid")
            index = sum(d["kind"] == kind for d in result["devices"])
            result["devices"].append(
                {
                    "id": f"{kind}:{index}",
                    "kind": kind,
                    "provider": "virtio",
                    "name": str(path),
                    "cpu_participation": True,
                }
            )
        except (OSError, RuntimeError) as error:
            result["diagnostics"].append(f"Virtio {path}: {error}")
        finally:
            if session is not None:
                release(session)


class VirtioExecutable:
    def __init__(self, device, lowered, timeout):
        from ._runtime import _native

        artifact, target = _native().build_tosa_program(lowered._json)
        doc = lowered.to_dict()
        self.inputs = len(doc["inputs"])
        self.sizes = [
            math.prod(doc["nodes"][i]["shape"])
            * (4 if doc["nodes"][i]["dtype"] == "float32" else 2)
            for i in doc["inputs"] + doc["outputs"]
        ]
        self.session = acquire(device.name)
        self.context = self.program = self.queue = None
        self.buffers = []
        self.closed = False
        s = self.session
        try:
            with s.lock:
                if len(self.sizes) > s.max_bindings or any(
                    n > s.max_buffer for n in self.sizes
                ):
                    raise ValueError("graph exceeds device resource limits")
                if len(artifact) > min(s.max_artifact, s.max_request - 96):
                    raise ValueError(
                        "TOSA artifact exceeds guest transfer limit (program loads cannot be chunked)"
                    )
                self.context = s.object(0x100, bytes(8), timeout)
                envelope = self.context + struct.pack(
                    "<II12IQQ", 0x544F5341, 0, *target, len(artifact), (1 << 64) - 1
                )
                self.program = s.object(0x300, envelope + artifact, timeout)
                self.queue = s.object(0x400, self.context + bytes(8), timeout)
                domain = 3 if s.capabilities & 32 else (1 if s.capabilities & 1 else 2)
                for i, count in enumerate(self.sizes):
                    desc = self.context + struct.pack(
                        "<QQB7xII", count, 1, domain, 6 if i < self.inputs else 9, 0
                    )
                    self.buffers.append(s.object(0x200, desc, timeout))
        except BaseException:
            try:
                self.close(timeout)
            except (RuntimeError, OSError):
                # Preserve the original admission/transport failure.
                pass
            raise

    def run(self, payloads, timeout):
        s = self.session
        s.check()
        with s.lock:
            if self.closed:
                raise RuntimeError("executable is closed")
            if (
                len(payloads) != self.inputs
                or [len(p) for p in payloads] != self.sizes[: self.inputs]
            ):
                raise ValueError("input sizes do not match the compiled graph")
            deadline = time.monotonic() + timeout

            def remaining():
                value = deadline - time.monotonic()
                if value <= 0:
                    raise TimeoutError(
                        "virtio execution timed out; session reset without replay"
                    )
                return value

            try:
                for buffer, data in zip(self.buffers, payloads):
                    chunk = s.max_request - 40
                    for offset in range(0, len(data), chunk):
                        tail = data[offset : offset + chunk]
                        s.request(
                            0x202,
                            buffer + struct.pack("<QQ", offset, len(tail)) + tail,
                            timeout=remaining(),
                        )
                bindings = b"".join(
                    buffer
                    + struct.pack("<QQIB3x", 0, count, i, 1 if i < self.inputs else 2)
                    for i, (buffer, count) in enumerate(zip(self.buffers, self.sizes))
                )
                event = s.object(
                    0x500,
                    self.queue
                    + self.program
                    + struct.pack("<IIQ", len(self.buffers), 0, 0)
                    + bindings,
                    remaining(),
                )
                while True:
                    state, error, reserved = struct.unpack(
                        "<HHI", s.request(0x501, event, size=8, timeout=remaining())
                    )
                    if (
                        reserved
                        or state > 3
                        or (state != 2 and error)
                        or (state == 2 and not error)
                    ):
                        raise ProtocolError("invalid event state")
                    if state:
                        break
                    time.sleep(min(0.001, remaining()))
                s.request(0x503, event, timeout=remaining())
                if state != 1:
                    raise RuntimeError(
                        f"virtio event failed: state={state}, error={error}"
                    )
                outputs = []
                for buffer, count in zip(
                    self.buffers[self.inputs :], self.sizes[self.inputs :]
                ):
                    parts = []
                    chunk = s.max_response - 16
                    for offset in range(0, count, chunk):
                        length = min(chunk, count - offset)
                        parts.append(
                            s.request(
                                0x203,
                                buffer + struct.pack("<QQ", offset, length),
                                size=length,
                                timeout=remaining(),
                            )
                        )
                    outputs.append(b"".join(parts))
                return outputs
            except BaseException:
                # A failed run can own an event or partially modified buffers.
                # Quarantine the shared session; never blindly replay a submit.
                s.poison()
                raise

    def close(self, timeout):
        if self.closed:
            return
        s = self.session
        if s.pid != os.getpid():
            raise RuntimeError(
                "virtio sessions cannot be closed after fork; exec a new process"
            )
        self.closed = True
        try:
            with s.lock:
                if s.fd is None:
                    return  # prior failure already reset the session
                for opcode, objects in (
                    (0x401, [self.queue]),
                    (0x301, [self.program]),
                    (0x201, self.buffers),
                    (0x101, [self.context]),
                ):
                    for obj in objects:
                        if obj is not None:
                            s.request(opcode, obj, timeout=timeout)
        except BaseException:
            s.poison()
            raise
        finally:
            release(s)

    def __del__(self):
        # Best effort only; explicit close/context manager observes failures.
        if hasattr(self, "closed") and not self.closed:
            try:
                self.close(1.0)
            except (RuntimeError, OSError):
                pass  # explicit close is the error-reporting path

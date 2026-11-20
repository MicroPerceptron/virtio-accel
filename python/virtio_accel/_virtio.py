"""Linux ioctl adapter for frozen protocol 1.0 (explicit copies, no replay).

The module is imported only on Linux. OS details deliberately stay outside the
portable Rust crates. The native compiler owns TOSA serialization and targets.
"""

import errno
import fcntl
import math
import os
import stat
import struct
import time
from pathlib import Path
from threading import RLock

from ._array import checked_dtype

# asm-generic ioctl encoding used by the supported Linux x86/ARM wheels.
EXCHANGE = 0xC020B700
INFO = 0x8020B701
IOCTL = struct.Struct("=8I")
HEADER = struct.Struct("<HHIQ")
DEVICE_INFO = struct.Struct("<16sHHIIQ6IQQ")
_sessions = {}
_device_cache = {}
_registry_lock = RLock()


def _after_fork():
    # Inherited executables fail their PID check before touching inherited locks.
    # New processes must exec before opening the same exclusive device.
    global _registry_lock, _sessions, _device_cache
    _registry_lock = RLock()
    _sessions = {}
    _device_cache = {}


os.register_at_fork(after_in_child=_after_fork)


OPERATIONS = {
    1: "GET_DEVICE_INFO",
    0x100: "CREATE_CONTEXT",
    0x101: "DESTROY_CONTEXT",
    0x200: "CREATE_BUFFER",
    0x201: "DESTROY_BUFFER",
    0x202: "WRITE_BUFFER",
    0x203: "READ_BUFFER",
    0x300: "LOAD_PROGRAM",
    0x301: "UNLOAD_PROGRAM",
    0x400: "CREATE_QUEUE",
    0x401: "DESTROY_QUEUE",
    0x500: "SUBMIT",
    0x501: "POLL_EVENT",
    0x503: "DESTROY_EVENT",
}
STATUSES = dict(
    enumerate(
        (
            "OK",
            "UNSUPPORTED",
            "INCOMPATIBLE",
            "INVALID_ARGUMENT",
            "OUT_OF_BOUNDS",
            "BUSY",
            "OUT_OF_MEMORY",
            "RESOURCE_LIMIT",
            "DEADLINE_EXPIRED",
            "DEVICE_LOST",
            "PERMISSION_DENIED",
            "STALE_OBJECT",
        )
    )
)
STATUSES[65535] = "INTERNAL_ERROR"


def _node_identity(info):
    return (info.st_dev, info.st_ino, info.st_rdev, info.st_ctime_ns)


class ProtocolError(RuntimeError):
    pass


class Session:
    def __init__(self, path):
        self.path = path
        self.pid = os.getpid()
        self.lock = RLock()
        self.references = 0
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            hints = {
                errno.EBUSY: "device is owned by another process; close its session first",
                errno.EACCES: "access denied; ask the guest administrator to configure the trusted virtio-accel group",
                errno.EPERM: "access denied by guest device policy",
            }
            if error.errno in hints:
                raise OSError(error.errno, hints[error.errno], path) from error
            raise
        self.sequence = 0
        try:
            info = os.fstat(self.fd)
            self.node_identity = _node_identity(info)
            if not stat.S_ISCHR(info.st_mode):
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
            self._transfer = bytearray()
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
            request_len = HEADER.size + len(payload)
            capacity = HEADER.size + size
            required = IOCTL.size + max(request_len, capacity)
            data = self._transfer
            if len(data) < required:
                # Geometric growth is bounded by the validated driver limits.
                capacity_bytes = min(
                    max(required, 2 * len(data)),
                    IOCTL.size + max(self.max_request, self.max_response),
                )
                data.extend(bytes(capacity_bytes - len(data)))
            IOCTL.pack_into(
                data,
                0,
                request_len,
                capacity,
                max(1, min(60000, math.ceil(timeout * 1000))),
                0,
                0,
                0,
                0,
                0,
            )
            HEADER.pack_into(data, IOCTL.size, opcode, 0, len(payload), self.sequence)
            data[IOCTL.size + HEADER.size : IOCTL.size + request_len] = payload
            try:
                fcntl.ioctl(self.fd, EXCHANGE, data, True)
                echoed_len, used, milliseconds, flags, *reserved = IOCTL.unpack_from(
                    data
                )
                if (
                    echoed_len != request_len
                    or not 16 <= used <= capacity
                    or not milliseconds
                    or flags
                    or any(reserved)
                ):
                    raise ProtocolError("invalid ioctl completion")
                status, flags, length, sequence = HEADER.unpack_from(data, IOCTL.size)
                if flags or sequence != self.sequence or length != used - HEADER.size:
                    raise ProtocolError("invalid response header")
                result = bytes(
                    memoryview(data)[IOCTL.size + HEADER.size : IOCTL.size + used]
                )
                if status:
                    # SUBMIT/DEVICE_LOST can carry an indeterminate event. Never
                    # replay it, even when the provider reports a nonzero ID.
                    if status == 9 or status not in (*range(12), 65535):
                        raise ProtocolError(f"device lost or unknown status {status}")
                    if result:
                        raise ProtocolError("unexpected failure payload")
                    hint = ""
                    if status == 7:
                        hint = (
                            "; close unused executables or reduce resident_cache_size; "
                            "current native hosts admit one resident program"
                        )
                    raise RuntimeError(
                        f"{OPERATIONS.get(opcode, hex(opcode))}: {STATUSES[status]} "
                        f"(status {status}){hint}; no fallback selected"
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


def discover(result, *, refresh=False):
    paths = sorted(Path("/dev").glob("virtio-accel[0-9]*"))
    with _registry_lock:
        for missing in _device_cache.keys() - {str(p) for p in paths}:
            del _device_cache[missing]
        for path in paths:
            session = None
            name = str(path)
            try:
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISCHR(info.st_mode):
                    raise RuntimeError(f"{path} is not a character device")
                identity = _node_identity(info)
                cached = _device_cache.get(name)
                if not refresh and cached and cached[0] == identity:
                    entry = cached[1].copy()
                else:
                    session = acquire(name)
                    # Cache metadata, never a spare fd: an idle discovery must
                    # not hold exclusive ownership or repeatedly reset the device.
                    kind = {1: "npu", 2: "gpu"}.get(session.device_class, "hybrid")
                    entry = dict(
                        kind=kind,
                        provider="virtio",
                        name=name,
                        cpu_participation=True,
                        _identity=(session.node_identity, session.uuid.hex()),
                    )
                    _device_cache[name] = (session.node_identity, entry.copy())
                index = sum(d["kind"] == entry["kind"] for d in result["devices"])
                entry["id"] = f"{entry['kind']}:{index}"
                result["devices"].append(entry)
            except (OSError, RuntimeError) as error:
                _device_cache.pop(name, None)
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
            * checked_dtype(doc["nodes"][i]["dtype"]).itemsize
            for i in doc["inputs"] + doc["outputs"]
        ]
        self.session = acquire(device.name)
        self.context = self.program = self.queue = None
        self.buffers = []
        self.closed = False
        s = self.session
        try:
            with s.lock:
                if device._identity != (s.node_identity, s.uuid.hex()):
                    raise RuntimeError(
                        "virtio device changed since discovery; discover it again"
                    )
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
                bindings = b"".join(
                    buffer
                    + struct.pack("<QQIB3x", 0, count, i, 1 if i < self.inputs else 2)
                    for i, (buffer, count) in enumerate(zip(self.buffers, self.sizes))
                )
                self.submission = (
                    self.queue
                    + self.program
                    + struct.pack("<IIQ", len(self.buffers), 0, 0)
                    + bindings
                )
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
                event = s.object(0x500, self.submission, remaining())
                poll = None
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
                    if poll is None:
                        # Already-complete work needs no timer or PyO3 wait object.
                        from ._native import _PollWait

                        poll = _PollWait()
                    poll.wait(remaining())
                s.request(0x503, event, timeout=remaining())
                if state != 1:
                    raise RuntimeError(
                        f"virtio event failed: state={state}, error={error}"
                    )
                outputs = []
                for buffer, count in zip(
                    self.buffers[self.inputs :], self.sizes[self.inputs :]
                ):
                    output = bytearray(count)
                    chunk = s.max_response - 16
                    for offset in range(0, count, chunk):
                        length = min(chunk, count - offset)
                        output[offset : offset + length] = s.request(
                            0x203,
                            buffer + struct.pack("<QQ", offset, length),
                            size=length,
                            timeout=remaining(),
                        )
                    outputs.append(output)
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

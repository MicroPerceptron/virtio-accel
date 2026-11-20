import errno
import json
import os
import struct
import sys
import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import patch

if sys.platform == "linux":
    from virtio_accel import _virtio as v


@unittest.skipUnless(sys.platform == "linux", "Linux ioctl transport")
class ResponseTests(unittest.TestCase):
    def test_quantized_binding_sizes_follow_storage_dtype(self):
        import numpy as np
        import virtio_accel as va
        import virtio_accel.numpy as vnp
        from virtio_accel import _runtime
        from types import SimpleNamespace
        for dtype, size in [(np.int8, 1), (np.int32, 4), (vnp.float8_e4m3fn, 1), (vnp.float8_e5m2, 1), (np.float16, 2), (np.float32, 4)]:
            lowered = va.jit(lambda x: x).lower(np.ones(3, dtype))
            executable = object.__new__(v.VirtioExecutable)
            with patch.object(_runtime, '_native') as native, patch.object(v, 'acquire', side_effect=RuntimeError('stop before allocation')):
                native.return_value.build_tosa_program.return_value = (b'model', ())
                with self.assertRaisesRegex(RuntimeError, 'stop before allocation'):
                    executable.__init__(SimpleNamespace(name='/dev/test'), lowered, 1)
            self.assertEqual(executable.sizes, [3 * size, 3 * size])

    def test_reused_transfer_storage_returns_independent_payloads(self):
        session = self.session()
        allocations = []

        def reply(fd, op, data, mutate):
            allocations.append(id(data))
            sequence = v.HEADER.unpack_from(data, 32)[3]
            self.response(data, sequence=sequence, length=8, used=24)
            data[48:56] = bytes([sequence]) * 8

        with patch.object(v.fcntl, "ioctl", side_effect=reply):
            first = session.request(1, size=8)
            second = session.request(1, size=8)
        self.assertEqual(allocations[0], allocations[1])
        self.assertEqual(first, bytes([1]) * 8)
        self.assertEqual(second, bytes([2]) * 8)
        self.assertLessEqual(len(session._transfer), v.IOCTL.size + session.max_request)

    def test_cached_discovery_skips_native_and_reprobes_recreated_nodes(self):
        from types import SimpleNamespace
        from virtio_accel import _runtime

        node = Path("/dev/virtio-accel0")
        info = SimpleNamespace(
            st_mode=0o20600, st_dev=1, st_ino=2, st_rdev=3, st_ctime_ns=4
        )
        session = SimpleNamespace(
            node_identity=(1, 2, 3, 4), uuid=bytes(16), device_class=2
        )
        with (
            patch.object(v, "_device_cache", {}),
            patch.object(v.Path, "glob", return_value=[node]),
            patch.object(v.Path, "stat", return_value=info),
            patch.object(v, "acquire", return_value=session) as acquire,
            patch.object(v, "release"),
            patch.object(_runtime, "_native") as native,
        ):
            (first,) = _runtime.devices(provider="virtio")
            self.assertEqual(_runtime.devices(provider="virtio"), (first,))
            acquire.assert_called_once()
            native.assert_not_called()
            info.st_ino = 9
            session.node_identity = (1, 9, 3, 4)
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                _runtime._resolve(first, False)
            self.assertEqual(acquire.call_count, 2)
            _runtime._discovery("virtio", refresh=True)
            self.assertEqual(acquire.call_count, 3)

    def session(self):
        session = object.__new__(v.Session)
        session.fd = 123
        session.pid = os.getpid()
        session.sequence = 0
        session.max_request = session.max_response = 4096
        session.lock = RLock()
        session._transfer = bytearray()
        return session

    def response(self, data, *, status=0, flags=0, length=0, sequence=1, used=16):
        fields = list(v.IOCTL.unpack_from(data))
        fields[1] = used
        v.IOCTL.pack_into(data, 0, *fields)
        v.HEADER.pack_into(data, 32, status, flags, length, sequence)

    def test_untrusted_completions_poison_without_replay(self):
        for changes in (
            {"used": 0},
            {"used": 17},
            {"flags": 1},
            {"sequence": 2},
            {"length": 4},
            {"status": 9},
            {"status": 12},
        ):
            with self.subTest(changes=changes):
                session = self.session()
                with (
                    patch.object(
                        v.fcntl,
                        "ioctl",
                        side_effect=lambda fd, op, data, mutate, changes=changes: (
                            self.response(data, **changes)
                        ),
                    ) as ioctl,
                    patch.object(v.os, "close") as close,
                ):
                    with self.assertRaises(v.ProtocolError):
                        session.request(0x101, bytes(8))
                    self.assertIsNone(session.fd)
                    ioctl.assert_called_once()
                    close.assert_called_once_with(123)
                    with self.assertRaisesRegex(RuntimeError, "lost"):
                        session.request(1)
                    ioctl.assert_called_once()

    def test_transport_failure_never_replays_mutation(self):
        for code in (errno.ETIMEDOUT, errno.ENODEV, errno.EIO, errno.EFAULT):
            session = self.session()
            with (
                patch.object(
                    v.fcntl, "ioctl", side_effect=OSError(code, "test")
                ) as ioctl,
                patch.object(v.os, "close"),
            ):
                with self.assertRaises(OSError):
                    session.request(0x500, bytes(32), size=8)
                self.assertIsNone(session.fd)
                ioctl.assert_called_once()

    def test_defined_rejection_preserves_session(self):
        session = self.session()
        with (
            patch.object(
                v.fcntl,
                "ioctl",
                side_effect=lambda fd, op, data, mutate: self.response(data, status=7),
            ),
            self.assertRaisesRegex(RuntimeError, "status 7"),
        ):
            session.request(0x100, bytes(8), size=8)
        self.assertEqual(session.fd, 123)

    def test_zero_object_and_nonempty_error_are_invalid(self):
        for status in (0, 7):
            session = self.session()

            def reply(fd, op, data, mutate, status=status):
                self.response(data, status=status, length=8, used=24)
                data[48:56] = bytes(8)

            with (
                patch.object(v.fcntl, "ioctl", side_effect=reply),
                patch.object(v.os, "close"),
            ):
                with self.assertRaises(v.ProtocolError):
                    session.object(0x100, bytes(8), 1)
                self.assertIsNone(session.fd)

    def test_bounds_and_fork_fail_before_submission(self):
        session = self.session()
        with patch.object(v.fcntl, "ioctl") as ioctl:
            with self.assertRaises(ValueError):
                session.request(1, bytes(4096))
            with self.assertRaises(ValueError):
                session.request(1, size=4096)
            session.pid = -1
            with self.assertRaisesRegex(RuntimeError, "fork"):
                session.request(1)
            ioctl.assert_not_called()

    def test_framing_matches_frozen_vectors(self):
        path = Path(__file__).resolve().parents[2] / "conformance/v1.0/vectors.json"
        frames = json.loads(path.read_text())["frames"]
        for frame in frames:
            if frame["kind"] != "request":
                continue
            expected = bytes.fromhex(frame["hex"])
            opcode, flags, length, sequence = struct.unpack("<HHIQ", expected[:16])
            self.assertEqual(flags, 0)
            self.assertEqual(length, len(expected) - 16)
            session = self.session()
            session.sequence = sequence - 1

            def reply(fd, op, data, mutate, expected=expected, sequence=sequence):
                count = v.IOCTL.unpack_from(data)[0]
                self.assertEqual(bytes(data[32 : 32 + count]), expected)
                self.response(data, sequence=sequence)

            with patch.object(v.fcntl, "ioctl", side_effect=reply):
                self.assertEqual(session.request(opcode, expected[16:]), b"")

    def test_shared_session_lifetime(self):
        session = self.session()
        session.path = "/test-device"
        session.references = 0
        with (
            patch.object(v, "_sessions", {}),
            patch.object(v, "Session", return_value=session) as ctor,
            patch.object(v.os, "close") as close,
        ):
            first = v.acquire(session.path)
            second = v.acquire(session.path)
            self.assertIs(first, second)
            v.release(first)
            close.assert_not_called()
            v.release(second)
            close.assert_called_once_with(123)
            self.assertFalse(v._sessions)
            ctor.assert_called_once()


if __name__ == "__main__":
    unittest.main()

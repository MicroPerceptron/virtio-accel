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
    def session(self):
        session = object.__new__(v.Session)
        session.fd = 123
        session.pid = os.getpid()
        session.sequence = 0
        session.max_request = session.max_response = 4096
        session.lock = RLock()
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

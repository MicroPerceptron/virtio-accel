#!/usr/bin/env python3
"""Unit tests for the irreversible publication driver's pure policy logic."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

from publication import PublicationPolicyError, validate_release_tag
from publish import (
    PublishError,
    archive_checksum,
    parse_index_checksum,
    require_matching_checksum,
    sparse_index_path,
)


class PublicationTests(unittest.TestCase):
    def test_release_tag_must_match_workspace_version_exactly(self) -> None:
        validate_release_tag("v1.2.3", "1.2.3")
        for invalid in ("1.2.3", "v1.2.4", "release-v1.2.3", "v1.2.3 "):
            with self.subTest(invalid=invalid), self.assertRaises(PublicationPolicyError):
                validate_release_tag(invalid, "1.2.3")

    def test_archive_checksum_is_streamed_sha256(self) -> None:
        payload = b"release archive bytes"
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / "crate"
            archive.write_bytes(payload)
            self.assertEqual(archive_checksum(archive), hashlib.sha256(payload).hexdigest())

    def test_sparse_index_paths_follow_registry_layout(self) -> None:
        self.assertEqual(sparse_index_path("a"), "1/a")
        self.assertEqual(sparse_index_path("ab"), "2/ab")
        self.assertEqual(sparse_index_path("abc"), "3/a/abc")
        self.assertEqual(sparse_index_path("Serde"), "se/rd/serde")
        self.assertEqual(
            sparse_index_path("virtio-accel"), "vi/rt/virtio-accel"
        )

    def test_registry_response_is_bound_to_name_and_version(self) -> None:
        checksum = "a" * 64
        payload = json.dumps(
            {"name": "virtio-accel", "vers": "1.2.2", "cksum": "b" * 64}
        ).encode() + b"\n" + json.dumps(
            {"name": "virtio-accel", "vers": "1.2.3", "cksum": checksum}
        ).encode()
        self.assertEqual(
            parse_index_checksum(payload, "virtio-accel", "1.2.3"), checksum
        )
        self.assertIsNone(parse_index_checksum(payload, "virtio-accel", "1.2.4"))
        with self.assertRaises(PublishError):
            parse_index_checksum(payload, "another-crate", "1.2.3")

    def test_existing_version_requires_exact_archive_checksum(self) -> None:
        require_matching_checksum("virtio-accel", "1.2.3", "a" * 64, "a" * 64)
        with self.assertRaises(PublishError):
            require_matching_checksum("virtio-accel", "1.2.3", "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main()

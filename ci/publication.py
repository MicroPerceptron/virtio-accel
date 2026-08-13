#!/usr/bin/env python3
"""Shared crates.io publication policy.

The order is deliberately a single source of truth used by the package dry run,
the irreversible publisher, and the release-policy checker.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Topologically sorted with development dependencies included: a published crate's versioned
# dev-dependencies must resolve from the registry too, or `cargo test` on the packaged source
# cannot run. Keep the documentary dependency table in docs/release-policy.md in sync.
PUBLISH_ORDER = (
    "virtio-accel-transport",
    "virtio-accel-cleanroom",
    "virtio-accel-proto",
    "virtio-accel-core",
    "virtio-accel-tosa",
    "virtio-accel-split-queue",
    "virtio-accel-guest",
    "virtio-accel-mock",
    "virtio-accel-device",
    "virtio-accel-conformance",
    "virtio-accel-coreml",
    "virtio-accel-openvino",
    "virtio-accel",
)


class PublicationPolicyError(RuntimeError):
    """A release cannot safely proceed under the repository policy."""


def capture(argv: list[str], cwd: pathlib.Path = ROOT) -> str:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PublicationPolicyError(f"{' '.join(argv)} failed: {detail}")
    return result.stdout


def workspace_version() -> str:
    """Return the one lockstep version shared by every published package."""
    metadata = json.loads(
        capture(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    )
    versions = {
        package["name"]: package["version"]
        for package in metadata["packages"]
        if package["name"] in PUBLISH_ORDER
    }
    missing = set(PUBLISH_ORDER) - versions.keys()
    if missing:
        raise PublicationPolicyError(
            f"packages missing from the workspace: {sorted(missing)}"
        )
    distinct = set(versions.values())
    if len(distinct) != 1:
        raise PublicationPolicyError(
            f"expected one workspace version, found {sorted(distinct)}"
        )
    return distinct.pop()


def expected_release_tag(version: str) -> str:
    return f"v{version}"


def validate_release_tag(tag: str, version: str) -> None:
    expected = expected_release_tag(version)
    if tag != expected:
        raise PublicationPolicyError(
            f"release tag must be {expected!r} for workspace version {version}, found {tag!r}"
        )

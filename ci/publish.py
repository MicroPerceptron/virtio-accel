#!/usr/bin/env python3
"""Publish every workspace crate to crates.io in the reviewed dependency order.

This is the irreversible half of the release pipeline. The caller must run
`ci/publish-dry-run.py` first. Existing versions are skipped only when the
locally generated `.crate` archive has the exact checksum recorded by crates.io,
which makes a workflow rerun safe after a partial or ambiguously completed upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

from publication import (
    PUBLISH_ORDER,
    PublicationPolicyError,
    ROOT,
    validate_release_tag,
    workspace_version,
)

CRATES_IO_INDEX = "https://index.crates.io"
USER_AGENT = "virtio-accel-release/1.0 (https://github.com/MicroPerceptron/virtio-accel)"
POLL_INTERVAL_SECONDS = 10
PUBLISH_VISIBILITY_TIMEOUT_SECONDS = 10 * 60
FAILED_PUBLISH_VISIBILITY_TIMEOUT_SECONDS = 2 * 60


class PublishError(RuntimeError):
    pass


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=ROOT, text=True)


def capture(argv: list[str]) -> str:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PublishError(f"{' '.join(argv)} failed: {detail}")
    return result.stdout.strip()


def validate_release_checkout(tag: str, version: str) -> None:
    """Require the canonical version tag to resolve to the checked-out commit."""
    validate_release_tag(tag, version)
    head = capture(["git", "rev-parse", "HEAD"])
    tagged = capture(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"])
    if head != tagged:
        raise PublishError(
            f"release tag {tag!r} resolves to {tagged}, but the checked-out commit is {head}"
        )
    tracked_changes = capture(["git", "status", "--porcelain", "--untracked-files=no"])
    if tracked_changes:
        raise PublishError("the release checkout has modified tracked files")


def archive_checksum(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_index_path(name: str) -> str:
    """Map a canonical crate name to its crates.io sparse-index path."""
    lowered = name.lower()
    if len(lowered) == 1:
        return f"1/{lowered}"
    if len(lowered) == 2:
        return f"2/{lowered}"
    if len(lowered) == 3:
        return f"3/{lowered[0]}/{lowered}"
    return f"{lowered[:2]}/{lowered[2:4]}/{lowered}"


def parse_index_checksum(payload: bytes, name: str, version: str) -> str | None:
    """Read one version checksum from the newline-delimited registry index."""
    for raw_line in payload.splitlines():
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise PublishError(f"crates.io returned an invalid index entry for {name}") from error
        if entry.get("vers") != version:
            continue
        if entry.get("name") != name:
            raise PublishError(
                f"crates.io index entry for {name} {version} names {entry.get('name')!r}"
            )
        checksum = entry.get("cksum")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise PublishError(f"crates.io returned an invalid checksum for {name} {version}")
        return checksum
    return None


def remote_checksum(name: str, version: str) -> str | None:
    request = urllib.request.Request(
        f"{CRATES_IO_INDEX}/{sparse_index_path(name)}",
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return parse_index_checksum(response.read(), name, version)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise PublishError(
            f"crates.io lookup for {name} {version} failed with HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise PublishError(f"crates.io lookup for {name} {version} failed: {error}") from error


def require_matching_checksum(
    name: str, version: str, local_checksum: str, registry_checksum: str
) -> None:
    if registry_checksum != local_checksum:
        raise PublishError(
            f"{name} {version} already exists with checksum {registry_checksum}, but the "
            f"release archive has checksum {local_checksum}; crates.io versions are immutable, "
            "so bump the workspace version and create the matching release tag"
        )


def wait_until_visible(
    name: str, version: str, local_checksum: str, timeout_seconds: int
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        registry_checksum = remote_checksum(name, version)
        if registry_checksum is not None:
            require_matching_checksum(name, version, local_checksum, registry_checksum)
            return True
        if time.monotonic() >= deadline:
            return False
        print(f"  waiting for {name} {version} to become visible on crates.io", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)


def package(name: str, version: str) -> tuple[pathlib.Path, str]:
    archive = ROOT / "target" / "package" / f"{name}-{version}.crate"
    archive.unlink(missing_ok=True)
    result = run(["cargo", "package", "--locked", "--no-verify", "-p", name])
    if result.returncode != 0:
        raise PublishError(f"{name}: cargo package failed with exit code {result.returncode}")
    if not archive.is_file():
        raise PublishError(f"{name}: cargo did not create {archive}")
    return archive, archive_checksum(archive)


def publish_one(name: str, version: str) -> None:
    _archive, local_checksum = package(name, version)
    registry_checksum = remote_checksum(name, version)
    if registry_checksum is not None:
        require_matching_checksum(name, version, local_checksum, registry_checksum)
        print(f"  {name} {version} already exists with the exact release checksum; skipping")
        return

    # The tagged archives have already been built, tested, and documented without the token by
    # publish-dry-run.py. Do not execute package build scripts after the token enters the step, and
    # do not permit repository Cargo configuration to redirect the upload to another registry.
    result = run(
        [
            "cargo",
            "publish",
            "--locked",
            "--no-verify",
            "--registry",
            "crates-io",
            "-p",
            name,
        ]
    )
    timeout = (
        PUBLISH_VISIBILITY_TIMEOUT_SECONDS
        if result.returncode == 0
        else FAILED_PUBLISH_VISIBILITY_TIMEOUT_SECONDS
    )
    if wait_until_visible(name, version, local_checksum, timeout):
        if result.returncode != 0:
            print(
                f"  cargo publish exited {result.returncode}, but crates.io records the exact "
                "archive; treating the upload as complete"
            )
        else:
            print(f"  {name} {version} is visible on crates.io with the expected checksum")
        return
    raise PublishError(
        f"{name} {version} did not become visible on crates.io within {timeout} seconds "
        f"after cargo publish exited {result.returncode}; stop and inspect the registry before "
        "rerunning"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="GitHub release tag")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the version, tag, commit, and clean checkout without publishing",
    )
    args = parser.parse_args()

    try:
        version = workspace_version()
        validate_release_checkout(args.tag, version)
        print(
            f"release checkout validated: {args.tag} contains {len(PUBLISH_ORDER)} "
            f"packages at version {version}"
        )
        if args.check_only:
            return 0
        if not os.environ.get("CARGO_REGISTRY_TOKEN"):
            raise PublishError("CARGO_REGISTRY_TOKEN is empty")

        for position, name in enumerate(PUBLISH_ORDER, start=1):
            print(f"\n[{position}/{len(PUBLISH_ORDER)}] {name} {version}", flush=True)
            publish_one(name, version)
        print(f"\npublished all {len(PUBLISH_ORDER)} crates for {args.tag}", flush=True)
        return 0
    except (PublicationPolicyError, PublishError) as error:
        print(f"publication refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ordered crates.io publication dry run against an isolated local registry.

`cargo publish --dry-run` cannot verify this workspace: crate N declares its first-party
dependencies as `version = "0.1.0"`, and while crates 1..N-1 are unpublished there is nothing on
crates.io for them to resolve against. `cargo package`'s own verify step is not a substitute either,
because it only builds the library target -- that is precisely the gap that let four cross-package
`include_str!` sites reach outside their package directories unnoticed.

So this script stands up a local registry and walks the documented publication order:

  1. vendor every third-party dependency into a Cargo *directory source*;
  2. for each crate, in order:
     a. `cargo package --no-verify` to produce the real `.crate` tarball,
     b. extract it somewhere with no path dependencies and no parent workspace,
     c. run `cargo build`, `cargo test`, and `cargo doc` inside that extracted source,
     d. only then add the crate to the registry.

Because a crate is added to the registry in step (d) -- after it is verified in step (c) -- it can
only ever resolve its predecessors. A wrong publication order fails with an unresolvable
dependency rather than passing quietly.

Usage:
    python3 ci/publish-dry-run.py [--work-dir DIR] [--keep] [--jobs N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile

from publication import PUBLISH_ORDER, PublicationPolicyError, workspace_version

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Examples live in the facade and are part of its published surface.
EXAMPLES = {"virtio-accel": ("backend_conformance", "reference_execution")}

# Nothing below may appear in any published tarball.
FORBIDDEN_PREFIXES = (".github/", "ci/", "fuzz/", "target/")
FORBIDDEN_PATHS = (".gitignore", "deny.toml", "rustfmt.toml", "clippy.toml")

REQUIRED_PATHS = ("LICENSE-MIT", "LICENSE-APACHE", "README.md")


class Failure(Exception):
    pass


def run(argv: list[str], cwd: pathlib.Path, what: str) -> None:
    print(f"    $ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=cwd)
    if result.returncode != 0:
        raise Failure(f"{what} failed with exit code {result.returncode}")


def capture(argv: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Failure(f"{' '.join(argv)} failed:\n{result.stderr}")
    return result.stdout


def vendor_third_party(registry: pathlib.Path, version: str) -> None:
    """Populate the registry with every third-party dependency.

    `cargo vendor` deliberately skips workspace members, so this leaves the registry containing
    exactly the crates that are already on crates.io. The registry must start out with none of our
    own crates in it, or the ordering below would prove nothing.
    """
    print("[1/2] vendoring third-party dependencies", flush=True)
    registry.mkdir(parents=True, exist_ok=True)
    capture(["cargo", "vendor", "--versioned-dirs", "--locked", "--quiet", str(registry)], ROOT)
    vendored = sorted(p.name for p in registry.iterdir() if p.is_dir())
    seeded = sorted(set(vendored) & {f"{name}-{version}" for name in PUBLISH_ORDER})
    if seeded:
        raise Failure(f"cargo vendor unexpectedly vendored workspace members: {seeded}")
    print(f"      {len(vendored)} third-party crates: {', '.join(vendored)}", flush=True)


def check_contents(crate_file: pathlib.Path, name: str, version: str) -> None:
    """Assert the tarball ships license and README material and no repository plumbing."""
    prefix = f"{name}-{version}/"
    with tarfile.open(crate_file) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
        contents = {m.removeprefix(prefix) for m in members}

        for entry in sorted(contents):
            if entry.startswith(FORBIDDEN_PREFIXES) or entry in FORBIDDEN_PATHS:
                raise Failure(f"{name}: tarball must not contain {entry}")

        for required in REQUIRED_PATHS:
            if required not in contents:
                raise Failure(f"{name}: tarball is missing {required}")

        if name == "virtio-accel" and "include/virtio_accel.h" not in contents:
            raise Failure("virtio-accel: tarball is missing include/virtio_accel.h")

        for license_name in ("LICENSE-MIT", "LICENSE-APACHE"):
            packaged = tar.extractfile(prefix + license_name)
            assert packaged is not None
            if packaged.read() != (ROOT / license_name).read_bytes():
                raise Failure(f"{name}: packaged {license_name} differs from the root file")

    print(f"      contents ok ({len(contents)} files)", flush=True)


def extract(crate_file: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """Unpack a `.crate` into `destination` and return its single top-level directory."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(crate_file) as tar:
        roots = {pathlib.PurePosixPath(m.name).parts[0] for m in tar.getmembers()}
        if len(roots) != 1:
            raise Failure(f"{crate_file.name}: expected one top-level directory, found {roots}")
        tar.extractall(destination, filter="data")
    return destination / roots.pop()


def source_args(registry: pathlib.Path) -> list[str]:
    """Point crates.io at the isolated local registry for one cargo invocation.

    Passing this inline rather than writing a `.cargo/config.toml` keeps the checkout and the
    extracted sources untouched, so nothing can leak between steps.
    """
    return [
        "--config",
        'source.crates-io.replace-with="local-registry"',
        "--config",
        f"source.local-registry.directory='{registry.as_posix()}'",
    ]


def add_to_registry(crate_file: pathlib.Path, registry: pathlib.Path) -> None:
    """Publish a packaged crate into the directory source."""
    digest = hashlib.sha256(crate_file.read_bytes()).hexdigest()
    entry = extract(crate_file, registry)
    # A directory source needs a checksum manifest. An empty `files` map disables per-file
    # verification, which is what every offline-vendoring workflow does; `package` still pins the
    # tarball this entry came from.
    (entry / ".cargo-checksum.json").write_text(
        json.dumps({"files": {}, "package": digest}) + "\n"
    )


def verify_packaged_source(
    name: str, crate_dir: pathlib.Path, registry: pathlib.Path, jobs: list[str]
) -> None:
    """Build, test, and document the packaged source as a consumer would receive it."""
    # A published library's lock file is informational; a consumer resolves fresh. Removing it
    # forces resolution against the registry rather than replaying workspace path sources.
    (crate_dir / "Cargo.lock").unlink(missing_ok=True)

    common = [*source_args(registry), *jobs]
    steps: list[tuple[str, list[str]]] = [
        ("cargo build", ["build", "--all-features"]),
        ("cargo test", ["test", "--all-targets", "--all-features"]),
        ("cargo test --doc", ["test", "--doc", "--all-features"]),
        ("cargo doc", ["doc", "--no-deps", "--all-features"]),
    ]
    steps += [
        (f"cargo run --example {example}", ["run", "--example", example])
        for example in EXAMPLES.get(name, ())
    ]
    for what, argv in steps:
        run(["cargo", *argv, *common], crate_dir, f"{name}: {what}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--keep", action="store_true", help="keep the work directory for inspection"
    )
    parser.add_argument("--jobs", type=int, help="value for cargo --jobs")
    args = parser.parse_args()

    jobs = ["--jobs", str(args.jobs)] if args.jobs else []

    work_dir = args.work_dir or pathlib.Path(tempfile.mkdtemp(prefix="virtio-accel-dry-run-"))
    work_dir = work_dir.resolve()
    if args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)

    registry = work_dir / "registry"
    verify_root = work_dir / "verify"
    package_dir = ROOT / "target" / "package"

    print(f"work directory: {work_dir}", flush=True)
    try:
        version = workspace_version()
        vendor_third_party(registry, version)

        print(f"[2/2] publishing {len(PUBLISH_ORDER)} crates in order", flush=True)
        for position, name in enumerate(PUBLISH_ORDER, start=1):
            print(f"\n  ({position}/{len(PUBLISH_ORDER)}) {name} {version}", flush=True)

            crate_file = package_dir / f"{name}-{version}.crate"
            crate_file.unlink(missing_ok=True)
            # --no-verify because cargo's verify step only builds the library target.
            # Verification happens below, against the local registry, and covers much more.
            # Packaging still resolves the generated manifest, so this is already the first
            # place a wrong publication order would fail.
            run(
                [
                    "cargo",
                    "package",
                    "--no-verify",
                    "--allow-dirty",
                    "-p",
                    name,
                    *source_args(registry),
                ],
                ROOT,
                f"{name}: cargo package",
            )
            if not crate_file.exists():
                raise Failure(f"{name}: expected {crate_file} to exist")

            check_contents(crate_file, name, version)
            crate_dir = extract(crate_file, verify_root / name)
            verify_packaged_source(name, crate_dir, registry, jobs)
            add_to_registry(crate_file, registry)
            print(f"      {name} verified and added to the local registry", flush=True)

        print(
            f"\nordered publication dry run passed for all {len(PUBLISH_ORDER)} crates",
            flush=True,
        )
        return 0
    except (Failure, PublicationPolicyError) as error:
        print(f"\ndry run failed: {error}", file=sys.stderr)
        return 1
    finally:
        if args.keep:
            print(f"work directory kept at {work_dir}", flush=True)
        elif not args.work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

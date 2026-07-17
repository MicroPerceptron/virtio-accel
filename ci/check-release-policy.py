#!/usr/bin/env python3
"""Verify release-policy invariants that should not drift silently."""

from __future__ import annotations

import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

PACKAGE_MANIFESTS = (
    ROOT / "Cargo.toml",
    ROOT / "conformance" / "rust-clean-room" / "Cargo.toml",
    *(ROOT / "crates").glob("*/Cargo.toml"),
)

CRATE_ROOTS = (
    ROOT / "src" / "lib.rs",
    ROOT / "conformance" / "rust-clean-room" / "src" / "lib.rs",
    ROOT / "fuzz" / "src" / "lib.rs",
    *((manifest.parent / "src" / "lib.rs") for manifest in (ROOT / "crates").glob("*/Cargo.toml")),
)

REQUIRED_LINKS = {
    ROOT / "README.md": (
        "docs/release-policy.md",
        "docs/releases/v1.0.md",
        "conformance/v1.0/freeze-audit.md",
    ),
    ROOT / "docs" / "releases" / "v1.0.md": (
        "../../conformance/v1.0/freeze-audit.md",
        "../release-policy.md",
    ),
    ROOT / "conformance" / "v1.0" / "README.md": ("freeze-audit.md",),
}


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def check_manifest(path: pathlib.Path) -> None:
    data = tomllib.loads(path.read_text())
    package = data.get("package")
    if not isinstance(package, dict):
        fail(f"{rel(path)} has no [package] table")

    if path == ROOT / "Cargo.toml":
        workspace_package = data.get("workspace", {}).get("package", {})
        if workspace_package.get("license") != "MIT OR Apache-2.0":
            fail("workspace package license must remain MIT OR Apache-2.0")
        if workspace_package.get("rust-version") != "1.85":
            fail("workspace rust-version must remain 1.85 until release policy review updates it")

    for key in ("license", "rust-version"):
        value = package.get(key)
        if value != {"workspace": True}:
            fail(f"{rel(path)} package.{key} must inherit from workspace")

    description = package.get("description")
    if not isinstance(description, str) or not description.strip():
        fail(f"{rel(path)} package.description must be present")

    if package.get("publish") is not False:
        fail(f"{rel(path)} package.publish must remain false until publish metadata review")


def check_crate_root(path: pathlib.Path) -> None:
    if not path.exists():
        fail(f"missing crate root {rel(path)}")
    text = path.read_text()
    if "#![forbid(unsafe_code)]" not in text:
        fail(f"{rel(path)} must keep #![forbid(unsafe_code)] or document an audited exception")


def check_links() -> None:
    for path, required in REQUIRED_LINKS.items():
        text = path.read_text()
        for needle in required:
            if needle not in text:
                fail(f"{rel(path)} must link {needle}")


def main() -> int:
    for manifest in sorted(PACKAGE_MANIFESTS):
        check_manifest(manifest)
    for crate_root in sorted(CRATE_ROOTS):
        check_crate_root(crate_root)
    check_links()
    print("release policy invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

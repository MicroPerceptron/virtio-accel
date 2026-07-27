#!/usr/bin/env python3
"""Verify release-policy invariants that should not drift silently."""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

PACKAGE_MANIFESTS = (
    ROOT / "Cargo.toml",
    ROOT / "conformance" / "rust-clean-room" / "Cargo.toml",
    *(ROOT / "crates").glob("*/Cargo.toml"),
)

# The complete set of packages published to crates.io, keyed by manifest directory relative to the
# repository root. Every manifest in PACKAGE_MANIFESTS must appear here: adding a package without
# deciding whether it is public is exactly the drift this check exists to prevent. The publication
# order is documented in docs/release-policy.md and enforced by ci/publish-dry-run.py.
PUBLISHED_PACKAGES = {
    ".": "virtio-accel",
    "conformance/rust-clean-room": "virtio-accel-cleanroom",
    "crates/virtio-accel-conformance": "virtio-accel-conformance",
    "crates/virtio-accel-core": "virtio-accel-core",
    "crates/virtio-accel-device": "virtio-accel-device",
    "crates/virtio-accel-guest": "virtio-accel-guest",
    "crates/virtio-accel-mock": "virtio-accel-mock",
    "crates/virtio-accel-proto": "virtio-accel-proto",
    "crates/virtio-accel-split-queue": "virtio-accel-split-queue",
    "crates/virtio-accel-transport": "virtio-accel-transport",
}

# Publish metadata that must be inherited from [workspace.package] so ten manifests cannot drift
# apart, mapped to the value the workspace is required to declare. `None` means "any value, but it
# must be present and inherited".
INHERITED_PACKAGE_KEYS = {
    "license": "MIT OR Apache-2.0",
    "rust-version": "1.85",
    "repository": "https://github.com/MicroPerceptron/virtio-accel",
    "homepage": "https://github.com/MicroPerceptron/virtio-accel",
    "keywords": None,
    "categories": None,
}

# crates.io rejects these outright, and neither `cargo package` nor `cargo publish --dry-run`
# checks them before the upload is attempted.
MAX_KEYWORDS = 5
MAX_KEYWORD_LENGTH = 20
MAX_CATEGORIES = 5

# Verified against the live crates.io category list. An unrecognized slug is a hard publish
# failure, so this stays an explicit allowlist rather than a free-form string.
VALID_CATEGORY_SLUGS = frozenset(
    {"no-std", "hardware-support", "virtualization", "embedded", "os", "api-bindings"}
)

LICENSE_FILES = ("LICENSE-MIT", "LICENSE-APACHE")

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
        "SECURITY.md",
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


def check_workspace_metadata() -> None:
    """The workspace is the single source of truth for every inherited publish field."""
    data = tomllib.loads((ROOT / "Cargo.toml").read_text())
    workspace_package = data.get("workspace", {}).get("package", {})

    for key, expected in INHERITED_PACKAGE_KEYS.items():
        value = workspace_package.get(key)
        if value is None:
            fail(f"workspace package.{key} must be declared so every package can inherit it")
        if expected is not None and value != expected:
            fail(f"workspace package.{key} must remain {expected!r}, found {value!r}")

    keywords = workspace_package["keywords"]
    if not isinstance(keywords, list) or not keywords:
        fail("workspace package.keywords must be a non-empty list")
    if len(keywords) > MAX_KEYWORDS:
        fail(f"crates.io allows at most {MAX_KEYWORDS} keywords, found {len(keywords)}")
    for keyword in keywords:
        if not isinstance(keyword, str) or len(keyword) > MAX_KEYWORD_LENGTH:
            fail(f"keyword {keyword!r} exceeds the {MAX_KEYWORD_LENGTH}-character crates.io limit")

    categories = workspace_package["categories"]
    if not isinstance(categories, list) or not categories:
        fail("workspace package.categories must be a non-empty list")
    if len(categories) > MAX_CATEGORIES:
        fail(f"crates.io allows at most {MAX_CATEGORIES} categories, found {len(categories)}")
    for category in categories:
        if category not in VALID_CATEGORY_SLUGS:
            fail(
                f"category {category!r} is not a known crates.io slug; an unrecognized slug is a "
                "hard publish failure that cargo does not catch locally"
            )


def check_manifest(path: pathlib.Path) -> None:
    data = tomllib.loads(path.read_text())
    package = data.get("package")
    if not isinstance(package, dict):
        fail(f"{rel(path)} has no [package] table")

    directory = rel(path.parent) if path.parent != ROOT else "."
    if directory not in PUBLISHED_PACKAGES:
        fail(
            f"{rel(path)} is not in the published-package allowlist in {rel(pathlib.Path(__file__))}"
            "; decide whether it is public and record the decision there"
        )
    if package.get("name") != PUBLISHED_PACKAGES[directory]:
        fail(f"{rel(path)} package.name must be {PUBLISHED_PACKAGES[directory]!r}")

    for key in INHERITED_PACKAGE_KEYS:
        if package.get(key) != {"workspace": True}:
            fail(f"{rel(path)} package.{key} must inherit from workspace")

    description = package.get("description")
    if not isinstance(description, str) or not description.strip():
        fail(f"{rel(path)} package.description must be present")

    # These ten are published. `publish = false` on any of them is a regression, and an explicit
    # allowlist beats a blanket rule the next new crate would silently join.
    publish = package.get("publish")
    if publish not in (None, True):
        fail(
            f"{rel(path)} is a published package, so package.publish must be absent or true, "
            f"found {publish!r}"
        )

    readme = package.get("readme")
    if readme != "README.md":
        fail(f'{rel(path)} package.readme must be "README.md", found {readme!r}')
    if not (path.parent / "README.md").is_file():
        fail(f"{directory}/README.md is missing; the crates.io landing page would be blank")

    # Every package directory carries its own license text: cargo only packages files inside the
    # package directory, so the root LICENSE-* files do not reach the sub-crate tarballs. Copies,
    # not symlinks -- CI runs windows-latest, where git symlinks need core.symlinks and developer
    # mode. Byte-identity is asserted here so the copies cannot drift.
    for license_name in LICENSE_FILES:
        packaged = path.parent / license_name
        if not packaged.is_file():
            fail(f"{directory}/{license_name} is missing; every published tarball must carry it")
        if packaged.read_bytes() != (ROOT / license_name).read_bytes():
            fail(f"{directory}/{license_name} is not byte-identical to the root {license_name}")


def check_published_set() -> None:
    """Every allowlisted package must exist, and nothing may be missing from the allowlist."""
    found = {rel(path.parent) if path.parent != ROOT else "." for path in PACKAGE_MANIFESTS}
    missing = PUBLISHED_PACKAGES.keys() - found
    if missing:
        fail(f"published packages have no manifest: {sorted(missing)}")


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


def check_publication_order_agrees() -> None:
    """The publication order and the published-package allowlist must describe the same ten crates.

    They live in different files for good reasons -- one is an order, the other is a policy -- but a
    crate added to only one of them would either never be published or never be verified.
    """
    source = ROOT / "ci" / "publish-dry-run.py"
    tree = ast.parse(source.read_text())
    order = next(
        (
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PUBLISH_ORDER"
                for target in node.targets
            )
        ),
        None,
    )
    if order is None:
        fail(f"{rel(source)} does not define PUBLISH_ORDER")
    order = list(order)
    if len(order) != len(set(order)):
        fail("ci/publish-dry-run.py PUBLISH_ORDER contains a duplicate")
    if set(order) != set(PUBLISHED_PACKAGES.values()):
        only_order = sorted(set(order) - set(PUBLISHED_PACKAGES.values()))
        only_allowlist = sorted(set(PUBLISHED_PACKAGES.values()) - set(order))
        fail(
            "ci/publish-dry-run.py PUBLISH_ORDER and the published-package allowlist disagree; "
            f"only in the order: {only_order}; only in the allowlist: {only_allowlist}"
        )


def check_fuzz_stays_unpublished() -> None:
    """The fuzz harness is a separate workspace and is deliberately not one of the ten."""
    manifest = ROOT / "fuzz" / "Cargo.toml"
    package = tomllib.loads(manifest.read_text()).get("package", {})
    if package.get("publish") is not False:
        fail(f"{rel(manifest)} package.publish must remain false")


def main() -> int:
    check_workspace_metadata()
    check_published_set()
    check_publication_order_agrees()
    for manifest in sorted(PACKAGE_MANIFESTS):
        check_manifest(manifest)
    check_fuzz_stays_unpublished()
    for crate_root in sorted(CRATE_ROOTS):
        check_crate_root(crate_root)
    check_links()
    print(f"release policy invariants hold for {len(PUBLISHED_PACKAGES)} published packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

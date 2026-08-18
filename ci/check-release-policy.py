#!/usr/bin/env python3
"""Verify release-policy invariants that should not drift silently."""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

from publication import PUBLISH_ORDER

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
    "crates/virtio-accel-coreml": "virtio-accel-coreml",
    "crates/virtio-accel-device": "virtio-accel-device",
    "crates/virtio-accel-guest": "virtio-accel-guest",
    "crates/virtio-accel-hexagon": "virtio-accel-hexagon",
    "crates/virtio-accel-mock": "virtio-accel-mock",
    "crates/virtio-accel-openvino": "virtio-accel-openvino",
    "crates/virtio-accel-proto": "virtio-accel-proto",
    "crates/virtio-accel-vaccel": "virtio-accel-vaccel",
    "crates/virtio-accel-split-queue": "virtio-accel-split-queue",
    "crates/virtio-accel-tosa": "virtio-accel-tosa",
    "crates/virtio-accel-transport": "virtio-accel-transport",
}

# Publish metadata that must be inherited from [workspace.package] so manifests cannot drift
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

UNSAFE_AUDITS = {
    ROOT / "crates" / "virtio-accel-coreml" / "src" / "lib.rs": (
        ROOT / "crates" / "virtio-accel-coreml" / "SAFETY.md",
        'cfg_attr(not(target_os = "macos"), forbid(unsafe_code))',
        ("Objective-C bridge", "AlignedAllocation", "atomic two-reference"),
    ),
    ROOT / "crates" / "virtio-accel-openvino" / "src" / "lib.rs": (
        ROOT / "crates" / "virtio-accel-openvino" / "SAFETY.md",
        "cfg_attr(not(va_openvino), forbid(unsafe_code))",
        ("OpenVINO C API", "AlignedAllocation", "poll-latch"),
    ),
    ROOT / "crates" / "virtio-accel-hexagon" / "src" / "lib.rs": (
        ROOT / "crates" / "virtio-accel-hexagon" / "SAFETY.md",
        "cfg_attr(not(va_hexagon), forbid(unsafe_code))",
        ("public QNN C interface", "AlignedAllocation", "`poll_event`"),
    ),
    ROOT / "crates" / "virtio-accel-tosa" / "src" / "lib.rs": (
        ROOT / "crates" / "virtio-accel-tosa" / "SAFETY.md",
        "#![deny(unsafe_code)]",
        (
            "generated module is private",
            "root_as_tosa_graph_with_opts",
            "Regeneration is a security-sensitive source change",
        ),
    ),
}

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

PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def rel(path: pathlib.Path) -> str:
    # Policy keys and package paths use Cargo's portable forward-slash spelling. Normalizing here
    # also lets contributors run the release gate directly on Windows.
    return path.relative_to(ROOT).as_posix()


def check_workspace_metadata() -> None:
    """The workspace is the single source of truth for every inherited publish field."""
    data = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
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
    data = tomllib.loads(path.read_text(encoding="utf-8"))
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

    # These packages are published. `publish = false` on any of them is a regression, and an explicit
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
    text = path.read_text(encoding="utf-8")
    if "#![forbid(unsafe_code)]" in text:
        return
    audit_spec = UNSAFE_AUDITS.get(path)
    if audit_spec is None:
        fail(f"{rel(path)} must keep #![forbid(unsafe_code)] or document an audited exception")
    audit, root_marker, required_markers = audit_spec
    if not audit.is_file():
        fail(f"{rel(path)} unsafe exception is missing {rel(audit)}")
    if root_marker not in text:
        fail(f"{rel(path)} unsafe exception must keep its confinement marker {root_marker!r}")
    audit_text = audit.read_text(encoding="utf-8")
    for required in required_markers:
        if required not in audit_text:
            fail(f"{rel(audit)} must document the {required!r} unsafe invariant")


def check_links() -> None:
    for path, required in REQUIRED_LINKS.items():
        text = path.read_text(encoding="utf-8")
        for needle in required:
            if needle not in text:
                fail(f"{rel(path)} must link {needle}")


def check_publication_order_agrees() -> None:
    """The publication order and the published-package allowlist must describe the same crates.

    They live in different files for good reasons -- one is an order, the other is a policy -- but a
    crate added to only one of them would either never be published or never be verified.
    """
    order = list(PUBLISH_ORDER)
    if len(order) != len(set(order)):
        fail("ci/publication.py PUBLISH_ORDER contains a duplicate")
    if set(order) != set(PUBLISHED_PACKAGES.values()):
        only_order = sorted(set(order) - set(PUBLISHED_PACKAGES.values()))
        only_allowlist = sorted(set(PUBLISHED_PACKAGES.values()) - set(order))
        fail(
            "ci/publication.py PUBLISH_ORDER and the published-package allowlist disagree; "
            f"only in the order: {only_order}; only in the allowlist: {only_allowlist}"
        )


def check_publish_workflow() -> None:
    """Keep the token-bearing workflow narrow and coupled to the repository gates."""
    if not PUBLISH_WORKFLOW.is_file():
        fail(f"missing {rel(PUBLISH_WORKFLOW)}")
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    required = (
        '"on":\n  release:\n    types:\n      - published',
        "permissions:\n  contents: read",
        "group: crates-io-release",
        "cancel-in-progress: false",
        "persist-credentials: false",
        "ref: ${{ github.event.release.tag_name }}",
        "DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}",
        'git merge-base --is-ancestor HEAD "origin/$DEFAULT_BRANCH"',
        "python3 ci/check-release-policy.py",
        "python3 ci/publish.py --tag \"$GITHUB_REF_NAME\" --check-only",
        "python3 ci/publish-dry-run.py",
        "CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_KEY }}",
        "python3 ci/publish.py --tag \"$GITHUB_REF_NAME\"",
    )
    for fragment in required:
        if fragment not in text:
            fail(f"{rel(PUBLISH_WORKFLOW)} is missing required release guard {fragment!r}")

    for forbidden_trigger in ("\n  pull_request:", "\n  push:", "\n  workflow_dispatch:"):
        if forbidden_trigger in text:
            fail(
                f"{rel(PUBLISH_WORKFLOW)} must run only for published GitHub releases; found "
                f"{forbidden_trigger.strip()!r}"
            )
    if text.count("secrets.CRATES_IO_KEY") != 1:
        fail(f"{rel(PUBLISH_WORKFLOW)} must expose CRATES_IO_KEY to exactly one step")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not re.search(r"@[0-9a-f]{40}(?:\s|$)", stripped):
            fail(f"{rel(PUBLISH_WORKFLOW)} action is not pinned to a full commit: {stripped}")

    publisher = (ROOT / "ci" / "publish.py").read_text(encoding="utf-8")
    for fragment in ('"--no-verify"', '"--registry",\n            "crates-io"'):
        if fragment not in publisher:
            fail(
                "ci/publish.py must use the token only after token-free verification and must "
                f"pin crates.io explicitly; missing {fragment!r}"
            )


def check_fuzz_stays_unpublished() -> None:
    """The fuzz harness is a separate workspace and is deliberately not published."""
    manifest = ROOT / "fuzz" / "Cargo.toml"
    package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
    if package.get("publish") is not False:
        fail(f"{rel(manifest)} package.publish must remain false")


def main() -> int:
    check_workspace_metadata()
    check_published_set()
    check_publication_order_agrees()
    check_publish_workflow()
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

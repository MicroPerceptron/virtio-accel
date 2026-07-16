#!/usr/bin/env python3
"""Generate or verify the protocol's normative-requirement ledger."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
from typing import Final

ROOT: Final = pathlib.Path(__file__).resolve().parents[1]
LEDGER: Final = ROOT / "conformance" / "v1.0" / "requirements.json"
SOURCES: Final = (
    ("SPEC", "docs/specification.md"),
    ("WIRE", "docs/wire-abi.md"),
    ("VIRTQ", "docs/virtqueue.md"),
)
KEYWORD: Final = re.compile(
    r"\*\*(MUST NOT|SHOULD NOT|MUST|SHOULD|MAY|REQUIRED|OPTIONAL)\*\*"
)
SECTION: Final = re.compile(r"^##\s+(\d+)(?:\.|\s)")


@dataclasses.dataclass(frozen=True)
class Coverage:
    status: str
    evidence: tuple[str, ...]
    issues: tuple[str, ...]
    rationale: str


def coverage(
    status: str,
    evidence: tuple[str, ...] = (),
    issues: tuple[str, ...] = (),
    rationale: str = "",
) -> Coverage:
    return Coverage(status, evidence, issues, rationale)


COVERAGE: Final[dict[tuple[str, int], Coverage]] = {
    ("SPEC", 1): coverage(
        "definition",
        rationale="Defines RFC 2119 and RFC 8174 terminology; it imposes no implementation behavior.",
    ),
    ("SPEC", 2): coverage(
        "mixed",
        (
            "ci/check-portable-dependencies.sh",
            "crates/virtio-accel-transport/src/lib.rs",
            "docs/architecture.md",
        ),
        ("#20", "#21"),
        "Portable queue and dependency boundaries are enforced now; concrete end-to-end and provider adapters remain tracked.",
    ),
    ("SPEC", 3): coverage(
        "mixed",
        (
            "ci/check-portable-dependencies.sh",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/architecture.md",
        ),
        ("#20", "#21"),
        "Layer and ownership boundaries are enforceable now; runtime provider integration remains tracked.",
    ),
    ("SPEC", 4): coverage(
        "mixed",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/src/object_table.rs",
            "crates/virtio-accel-mock/src/lib.rs",
        ),
        ("#20", "#21", "#25"),
        "Core ownership types and lifecycle tests exist; command-engine enforcement, complete backend policy, and retained-byte accounting remain tracked.",
    ),
    ("SPEC", 5): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/src/lib.rs",
        ),
        ("#20", "#32"),
        "Namespaces and reserved values are executable; runtime negotiation and future evolution remain tracked.",
    ),
    ("SPEC", 6): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "conformance/v1.0/layout.json",
            "conformance/v1.0/vectors.json",
        ),
        ("#32", "#33"),
        "Version and exact-length behavior is executable; release governance and the final freeze remain tracked.",
    ),
    ("SPEC", 7): coverage(
        "mixed",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/src/object_table.rs",
            "crates/virtio-accel-mock/src/lib.rs",
        ),
        ("#20", "#21"),
        "Handle and generational-ID invariants are tested; end-to-end release recovery remains tracked.",
    ),
    ("SPEC", 8): coverage(
        "mixed",
        ("crates/virtio-accel-core/src/lib.rs",),
        ("#20", "#21"),
        "Relative timeout semantics are tested; bounded polling and admission behavior require the command engine.",
    ),
    ("SPEC", 9): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-core/src/lib.rs",
        ),
        ("#20", "#21"),
        "Wire error shapes and ownership-aware failures are executable; backend-to-device recovery is tracked.",
    ),
    ("SPEC", 10): coverage(
        "executable",
        (
            ".github/workflows/ci.yml",
            "ci/check-portable-dependencies.sh",
            "docs/portability.md",
        ),
        rationale="The target matrix and dependency guards directly enforce the portable-layer restrictions.",
    ),
    ("SPEC", 11): coverage(
        "mixed",
        ("ci/check-normative-requirements.py",),
        ("#21", "#25", "#32", "#33"),
        "The ledger prevents silent requirements; the section's named completion work remains explicitly tracked.",
    ),
    ("WIRE", 1): coverage(
        "executable",
        (
            "conformance/rust-clean-room/src/lib.rs",
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/src/lib.rs",
        ),
        rationale="Both codecs use checked arithmetic and independently enforce the global limits.",
    ),
    ("WIRE", 2): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/src/lib.rs",
        ),
        rationale="Configuration limits, version compatibility, queue-size bounds, and features are tested.",
    ),
    ("WIRE", 3): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/src/lib.rs",
        ),
        rationale="Headers, exact lengths, flags, request correlation, and unknown numeric values are tested.",
    ),
    ("WIRE", 4): coverage(
        "executable",
        (
            "conformance/rust-clean-room/src/lib.rs",
            "conformance/rust-clean-room/tests/vectors.rs",
        ),
        rationale="The independent semantic codec validates object IDs, domains, usage, access, and event states.",
    ),
    ("WIRE", 5): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/tests/semantic_interop.rs",
        ),
        ("#20", "#25"),
        "Every payload layout and scalar invariant is cross-decoded; live-object and advertised-quota checks are tracked.",
    ),
    ("WIRE", 6): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-core/src/lib.rs",
        ),
        ("#20", "#21"),
        "Malformed input classifications and opaque statuses are executable; provider error mapping is tracked.",
    ),
    ("WIRE", 7): coverage(
        "mixed",
        (
            "conformance/v1.0/vectors.json",
            "crates/virtio-accel-split-queue/src/queue.rs",
        ),
        ("#20",),
        "Queue-level used accounting and preflight failures are executable; post-mutation command response atomicity requires full-path execution.",
    ),
    ("WIRE", 8): coverage(
        "executable",
        (
            "conformance/v1.0/layout.json",
            "conformance/v1.0/vectors.json",
            "crates/virtio-accel-proto/src/lib.rs",
        ),
        rationale="Checked-in artifacts are immutable test inputs and drift is detected by both codec suites.",
    ),
    ("WIRE", 9): coverage(
        "mixed",
        ("ci/check-normative-requirements.py",),
        ("#32", "#33"),
        "Coordinated documentation and evidence are machine-checked; final release policy and freeze remain tracked.",
    ),
    ("VIRTQ", 1): coverage(
        "executable",
        (
            "crates/virtio-accel-proto/src/lib.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
        ),
        rationale="The baseline features and direct split-ring profile are tested, including rejection of indirect and unknown descriptor flags.",
    ),
    ("VIRTQ", 2): coverage(
        "executable",
        (
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "crates/virtio-accel-transport/src/regions.rs",
        ),
        rationale="The split model maps direct descriptor tables into address-free segmented byte ports and reset-scoped completion tokens.",
    ),
    ("VIRTQ", 3): coverage(
        "executable",
        (
            "crates/virtio-accel-device/src/frame.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/regions.rs",
        ),
        rationale="Topology, direction, count, nonzero length, addressability, arithmetic, and command-frame limits are executable.",
    ),
    ("VIRTQ", 4): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-device/src/frame.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/regions.rs",
        ),
        ("#20",),
        "Descriptor-backed segmented presentation and frame exactness are executable; one-command full-path behavior remains tracked.",
    ),
    ("VIRTQ", 5): coverage(
        "tracked",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
        ),
        ("#20",),
        "Queue ordering and semantic validation classifications are executable separately; full failure atomicity requires their integration.",
    ),
    ("VIRTQ", 6): coverage(
        "tracked",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
        ),
        ("#20",),
        "Scatter writes and used-length accounting are executable; post-mutation command failure requires full-path execution.",
    ),
    ("VIRTQ", 7): coverage(
        "mixed",
        (
            "crates/virtio-accel-split-queue/src/queue.rs",
            "docs/virtqueue.md",
        ),
        ("#20",),
        "Available-order consumption and out-of-order used publication are executable; command/event distinction awaits the full path.",
    ),
    ("VIRTQ", 8): coverage(
        "mixed",
        (
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/virtqueue.md",
        ),
        ("#19", "#20"),
        "Publication ownership and retryable backpressure types are executable; guest behavior and full-path tests remain tracked.",
    ),
    ("VIRTQ", 9): coverage(
        "mixed",
        (
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/virtqueue.md",
        ),
        ("#19",),
        "Base split-ring suppression and mandatory rechecks are executable; guest-side notification behavior remains tracked.",
    ),
    ("VIRTQ", 10): coverage(
        "mixed",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
        ),
        ("#20",),
        "Malformed frames and descriptor chains are isolated separately; full command-chain continuation remains tracked.",
    ),
    ("VIRTQ", 11): coverage(
        "mixed",
        (
            "crates/virtio-accel-transport/src/queue.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-device/src/engine.rs",
            "docs/virtqueue.md",
        ),
        ("#19", "#20"),
        "Split-ring and engine reset are executable separately; guest reclamation and full-path reset remain tracked.",
    ),
    ("VIRTQ", 12): coverage(
        "tracked",
        ("crates/virtio-accel-core/src/lib.rs",),
        ("#20", "#25"),
        "Ownership-aware failure types exist; DEVICE_NEEDS_RESET transitions require the command engine and threat limits.",
    ),
}

RESET_ENGINE_COVERAGE: Final = coverage(
    "executable",
    (
        "crates/virtio-accel-device/src/engine.rs",
        "crates/virtio-accel-device/src/object_table.rs",
        "crates/virtio-accel-device/tests/command_processor.rs",
    ),
    rationale="The reset engine and tests enforce bounded child-before-parent teardown, sticky backend discard, explicit quarantine accounting, and fresh object namespaces.",
)
RESET_TRANSPORT_COVERAGE: Final = coverage(
    "mixed",
    (
        "crates/virtio-accel-transport/src/queue.rs",
        "crates/virtio-accel-split-queue/src/queue.rs",
        "crates/virtio-accel-device/src/engine.rs",
        "crates/virtio-accel-device/tests/command_processor.rs",
        "docs/virtqueue.md",
    ),
    ("#20",),
    "The split model stops fetch and publication while atomically invalidating old byte ports; full transport-to-engine reset remains tracked.",
)
RESET_ENGINE_MARKERS: Final = (
    "Teardown **MUST** be bounded",
    "The device **MAY** reuse a backend instance",
    "repeated reset attempts **MUST NOT** invoke that backend again",
    "Successful reinitialization **MUST** use a fresh nonzero object namespace",
)


def requirement_coverage(source_code: str, number: int, statement: str) -> Coverage | None:
    if source_code == "SPEC" and number == 4:
        if statement.startswith("The transport **MUST** stop fetching command chains"):
            return RESET_TRANSPORT_COVERAGE
        if any(marker in statement for marker in RESET_ENGINE_MARKERS):
            return RESET_ENGINE_COVERAGE
    return COVERAGE.get((source_code, number))


def section_number(heading: str) -> int:
    match = SECTION.match(heading)
    if match is None:
        raise ValueError(f"normative keyword occurs outside a numbered section: {heading!r}")
    return int(match.group(1))


def requirement_id(
    source_code: str,
    line_number: int,
    section: str,
    statement: str,
    keyword: str,
    ordinal: int,
) -> str:
    material = "\0".join(
        (source_code, str(line_number), section, statement, keyword, str(ordinal))
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:12].upper()
    return f"REQ-{source_code}-{digest}"


def generate() -> dict[str, object]:
    requirements: list[dict[str, object]] = []

    for source_code, relative_source in SOURCES:
        source = ROOT / relative_source
        heading = ""
        for line_number, raw_line in enumerate(source.read_text().splitlines(), start=1):
            if raw_line.startswith("## "):
                heading = raw_line
            statement = raw_line.strip()
            matches = tuple(KEYWORD.finditer(raw_line))
            if not matches:
                continue
            number = section_number(heading)
            assigned = requirement_coverage(source_code, number, statement)
            if assigned is None:
                raise ValueError(
                    f"no coverage mapping for {relative_source}:{line_number} ({heading})"
                )
            if not assigned.rationale:
                raise ValueError(f"coverage rationale is empty for {source_code} section {number}")
            if assigned.status not in {"definition", "executable", "mixed", "tracked"}:
                raise ValueError(f"invalid coverage status {assigned.status!r}")
            if assigned.status != "definition" and not (
                assigned.evidence or assigned.issues
            ):
                raise ValueError(
                    f"{source_code} section {number} has neither evidence nor issues"
                )
            for evidence in assigned.evidence:
                if not (ROOT / evidence).exists():
                    raise ValueError(
                        f"coverage evidence does not exist for {source_code} "
                        f"section {number}: {evidence}"
                    )
            for issue in assigned.issues:
                if re.fullmatch(r"#[1-9][0-9]*", issue) is None:
                    raise ValueError(
                        f"invalid tracked issue for {source_code} section {number}: {issue}"
                    )

            for ordinal, match in enumerate(matches, start=1):
                keyword = match.group(1)
                requirements.append(
                    {
                        "id": requirement_id(
                            source_code,
                            line_number,
                            heading,
                            statement,
                            keyword,
                            ordinal,
                        ),
                        "source": relative_source,
                        "line": line_number,
                        "section": heading.removeprefix("## "),
                        "keyword": keyword,
                        "statement": statement,
                        "coverage": {
                            "status": assigned.status,
                            "evidence": list(assigned.evidence),
                            "issues": list(assigned.issues),
                            "rationale": assigned.rationale,
                        },
                    }
                )

    identifiers = [entry["id"] for entry in requirements]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("requirement IDs are not unique")

    return {
        "schema": "virtio-accel-normative-requirements-1",
        "generated_by": "ci/check-normative-requirements.py",
        "requirement_count": len(requirements),
        "requirements": requirements,
    }


def render(ledger: dict[str, object]) -> str:
    return json.dumps(ledger, indent=2, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()

    try:
        expected = render(generate())
    except ValueError as error:
        print(f"normative requirement generation failed: {error}", file=sys.stderr)
        return 1

    if args.write:
        LEDGER.write_text(expected)
        print(f"wrote {LEDGER.relative_to(ROOT)}")
        return 0

    if not LEDGER.exists():
        print(f"missing {LEDGER.relative_to(ROOT)}; run with --write", file=sys.stderr)
        return 1
    actual = LEDGER.read_text()
    if actual != expected:
        try:
            actual_data = json.loads(actual)
            expected_data = json.loads(expected)
            actual_ids = {
                entry["id"] for entry in actual_data.get("requirements", [])
            }
            expected_ids = {
                entry["id"] for entry in expected_data.get("requirements", [])
            }
            added = sorted(expected_ids - actual_ids)
            removed = sorted(actual_ids - expected_ids)
            if added:
                print(f"uncatalogued normative requirements: {', '.join(added)}", file=sys.stderr)
            if removed:
                print(f"stale normative requirements: {', '.join(removed)}", file=sys.stderr)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        print(
            "normative requirement ledger is stale; review the change and run "
            "python3 ci/check-normative-requirements.py --write",
            file=sys.stderr,
        )
        return 1

    count = json.loads(actual)["requirement_count"]
    print(f"normative requirement ledger is complete ({count} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

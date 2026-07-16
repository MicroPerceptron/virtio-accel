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
        ("ci/check-portable-dependencies.sh", "docs/architecture.md"),
        ("#17", "#20", "#21"),
        "Portable dependency direction is enforced now; concrete transport and provider boundaries are tracked.",
    ),
    ("SPEC", 3): coverage(
        "mixed",
        ("ci/check-portable-dependencies.sh", "docs/architecture.md"),
        ("#17", "#20", "#21"),
        "Layer dependencies are enforceable now; runtime adapter boundaries require their implementations.",
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
        "tracked",
        ("conformance/v1.0/vectors.json",),
        ("#18", "#20"),
        "Error shapes are fixed, while writable preflight and post-mutation response atomicity require queue execution.",
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
        "tracked",
        ("docs/virtqueue.md",),
        ("#17", "#18"),
        "The negotiated reference profile is specified; transport feature enforcement requires the queue adapter.",
    ),
    ("VIRTQ", 2): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#17", "#18"),
        "The flattened region contract is specified and assigned to the region-port and split-ring implementations.",
    ),
    ("VIRTQ", 3): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#17", "#18"),
        "Topology and mapping rules have stable VQ cases but require executable descriptor-chain ports.",
    ),
    ("VIRTQ", 4): coverage(
        "tracked",
        ("conformance/rust-clean-room/tests/vectors.rs",),
        ("#17", "#18"),
        "Frame exactness is executable; cross-region presentation requires the region adapter and split-ring model.",
    ),
    ("VIRTQ", 5): coverage(
        "tracked",
        ("conformance/rust-clean-room/tests/vectors.rs",),
        ("#18", "#20"),
        "Semantic validation classifications are executable; ordering and failure atomicity require queue execution.",
    ),
    ("VIRTQ", 6): coverage(
        "tracked",
        ("conformance/rust-clean-room/tests/vectors.rs",),
        ("#18", "#20"),
        "Response bytes and lengths are defined; scatter writes and post-mutation failure require queue execution.",
    ),
    ("VIRTQ", 7): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#18", "#20"),
        "Stable VQ ordering cases exist; concurrent publication and completion behavior awaits the queue model.",
    ),
    ("VIRTQ", 8): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#19", "#20"),
        "Backpressure semantics are specified and assigned to the guest and full-path implementations.",
    ),
    ("VIRTQ", 9): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#18", "#19"),
        "Notification invariants are specified and assigned to split-ring and guest implementations.",
    ),
    ("VIRTQ", 10): coverage(
        "tracked",
        ("conformance/rust-clean-room/tests/vectors.rs",),
        ("#18", "#20"),
        "Malformed frame behavior is executable; malformed descriptor-chain isolation requires queue execution.",
    ),
    ("VIRTQ", 11): coverage(
        "tracked",
        ("docs/virtqueue.md",),
        ("#18", "#19", "#20"),
        "Reset cases are stable, while descriptor reclamation and late-completion exclusion require full implementations.",
    ),
    ("VIRTQ", 12): coverage(
        "tracked",
        ("crates/virtio-accel-core/src/lib.rs",),
        ("#20", "#25"),
        "Ownership-aware failure types exist; DEVICE_NEEDS_RESET transitions require the command engine and threat limits.",
    ),
}


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
            assigned = COVERAGE.get((source_code, number))
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

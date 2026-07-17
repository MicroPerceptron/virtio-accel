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
        "executable",
        (
            "ci/check-portable-dependencies.sh",
            "crates/virtio-accel-transport/src/lib.rs",
            "docs/architecture.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Portable dependency, provider, queue, and end-to-end boundaries are directly enforced; concrete platform adapters are outside portable v1.",
    ),
    ("SPEC", 3): coverage(
        "executable",
        (
            "ci/check-portable-dependencies.sh",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/architecture.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Layer direction, queue ownership, provider ownership, and the complete portable lifecycle are executable.",
    ),
    ("SPEC", 4): coverage(
        "executable",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/src/engine.rs",
            "crates/virtio-accel-device/src/state.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "crates/virtio-accel-device/src/object_table.rs",
            "crates/virtio-accel-mock/src/lib.rs",
            "docs/threat-model.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Provider and command-engine lifecycle, finite attacker dimensions, and aggregate retained-byte policy are specified and executable.",
    ),
    ("SPEC", 5): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/src/lib.rs",
            "ci/check-release-policy.py",
            "docs/release-policy.md",
        ),
        rationale="Namespaces, reserved values, runtime capability selection, negotiation, and release-policy guardrails are executable or policy-checked.",
    ),
    ("SPEC", 6): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "conformance/v1.0/layout.json",
            "conformance/v1.0/vectors.json",
            "conformance/v1.0/freeze-audit.md",
            "docs/release-policy.md",
        ),
        rationale="Version, exact-length behavior, release governance, and the final freeze audit are checked into the v1.0 evidence set.",
    ),
    ("SPEC", 7): coverage(
        "executable",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/src/object_table.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Handle ownership, generational IDs, rejected and indeterminate release, reset recovery, and transfer-failure publication are executable.",
    ),
    ("SPEC", 8): coverage(
        "executable",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Relative timeouts, bounded polling and cancellation, admission ownership, and end-to-end progress are executable.",
    ),
    ("SPEC", 9): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Wire error shapes, ownership-aware failures, deterministic provider mapping, and backend-to-device recovery are executable.",
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
        "executable",
        (
            "ci/check-normative-requirements.py",
            "ci/check-release-policy.py",
            "conformance/v1.0/freeze-audit.md",
            "docs/releases/v1.0.md",
            "docs/release-policy.md",
            "docs/threat-model.md",
        ),
        rationale="The ledger, release-policy check, release note, and freeze audit preserve the post-freeze audit trail.",
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
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-proto/tests/semantic_interop.rs",
            "crates/virtio-accel-device/src/state.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "docs/threat-model.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Every payload invariant is cross-decoded, and live-object, advertised-quota, and aggregate retained-byte bounds are executable.",
    ),
    ("WIRE", 6): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
        ),
        rationale="Malformed input classifications, opaque statuses, and provider error mapping are executable.",
    ),
    ("WIRE", 7): coverage(
        "executable",
        (
            "conformance/v1.0/vectors.json",
            "crates/virtio-accel-device/src/response.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Preflight, post-mutation response atomicity, exact used accounting, and short-completion rejection are executable through the full path.",
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
        "executable",
        (
            "ci/check-normative-requirements.py",
            "conformance/v1.0/freeze-audit.md",
            "docs/release-policy.md",
        ),
        rationale="Coordinated documentation, evidence, release classification, and freeze status are checked into versioned artifacts.",
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
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-device/src/frame.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/regions.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Descriptor-backed segmented presentation, frame exactness, and complete command behavior are executable through the full path.",
    ),
    ("VIRTQ", 5): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Queue ordering, semantic validation, and full-path failure atomicity are executable.",
    ),
    ("VIRTQ", 6): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Scatter writes, used-length accounting, and post-mutation command failure are executable through the full path.",
    ),
    ("VIRTQ", 7): coverage(
        "executable",
        (
            "crates/virtio-accel-split-queue/src/queue.rs",
            "docs/virtqueue.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Available-order consumption, out-of-order completion, and command/event distinction are executable through the full path.",
    ),
    ("VIRTQ", 8): coverage(
        "executable",
        (
            "crates/virtio-accel-guest/src/client.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/virtqueue.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Publication ownership, guest retryable backpressure, and full-path completion are executable.",
    ),
    ("VIRTQ", 9): coverage(
        "executable",
        (
            "crates/virtio-accel-guest/src/client.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "docs/virtqueue.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Split-ring suppression, guest notification decisions, mandatory rechecks, and lost-wakeup scenarios are executable.",
    ),
    ("VIRTQ", 10): coverage(
        "executable",
        (
            "conformance/rust-clean-room/tests/vectors.rs",
            "crates/virtio-accel-split-queue/src/chain.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Malformed frames and descriptor chains are isolated while later command chains continue through the full path.",
    ),
    ("VIRTQ", 11): coverage(
        "executable",
        (
            "crates/virtio-accel-guest/src/client.rs",
            "crates/virtio-accel-transport/src/queue.rs",
            "crates/virtio-accel-split-queue/src/queue.rs",
            "crates/virtio-accel-device/src/engine.rs",
            "docs/virtqueue.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Split-ring invalidation, guest reclamation, engine teardown, and full-path reset are executable.",
    ),
    ("VIRTQ", 12): coverage(
        "executable",
        (
            "crates/virtio-accel-core/src/lib.rs",
            "crates/virtio-accel-device/src/engine.rs",
            "crates/virtio-accel-device/src/state.rs",
            "crates/virtio-accel-device/tests/command_processor.rs",
            "docs/threat-model.md",
            "tests/portable_end_to_end.rs",
        ),
        rationale="Ownership-aware failures, DEVICE_NEEDS_RESET transitions, quarantine accounting, and adversarial resource policy are executable.",
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
PROVIDER_CAPABILITY_COVERAGE: Final = coverage(
    "executable",
    (
        "crates/virtio-accel-core/src/lib.rs",
        "crates/virtio-accel-device/src/decoder.rs",
        "crates/virtio-accel-device/src/engine.rs",
        "crates/virtio-accel-device/tests/command_processor.rs",
        "crates/virtio-accel-guest/src/client.rs",
        "crates/virtio-accel-mock/src/lib.rs",
    ),
    rationale="Construction-time metadata checks and operation preflight tests enforce stable limits, usable memory domains, reserved flags, and capability-to-method consistency.",
)
RESET_TRANSPORT_COVERAGE: Final = coverage(
    "executable",
    (
        "crates/virtio-accel-transport/src/queue.rs",
        "crates/virtio-accel-split-queue/src/queue.rs",
        "crates/virtio-accel-device/src/engine.rs",
        "crates/virtio-accel-device/tests/command_processor.rs",
        "docs/virtqueue.md",
        "tests/portable_end_to_end.rs",
    ),
    rationale="The full path stops fetch and publication, atomically invalidates old byte ports, tears down backend state, and reclaims guest ownership.",
)
RESET_ENGINE_MARKERS: Final = (
    "Teardown **MUST** be bounded",
    "The device **MAY** reuse a backend instance",
    "repeated reset attempts **MUST NOT** invoke that backend again",
    "Successful reinitialization **MUST** use a fresh nonzero object namespace",
)
PROVIDER_CAPABILITY_MARKERS: Final = (
    "If the backend does not advertise semantic event-cancellation capability",
    "The device **MUST** reject an allocation for an unsupported memory domain",
    "A baseline backend **MUST** advertise at least one assigned memory-domain capability",
    "resource-count, binding-count, and byte limit in `DeviceInfo`",
    "capabilities, and limits **MUST** remain stable",
    "If `EVENT_CANCELLATION` is absent",
    "invocation. If it is present, the backend",
    "**MUST** reject nonempty values before backend invocation",
)


def requirement_coverage(source_code: str, number: int, statement: str) -> Coverage | None:
    if source_code == "SPEC" and number == 4:
        if statement.startswith("The transport **MUST** stop fetching command chains"):
            return RESET_TRANSPORT_COVERAGE
        if any(marker in statement for marker in RESET_ENGINE_MARKERS):
            return RESET_ENGINE_COVERAGE
    if source_code == "SPEC" and number == 5:
        if any(marker in statement for marker in PROVIDER_CAPABILITY_MARKERS):
            return PROVIDER_CAPABILITY_COVERAGE
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

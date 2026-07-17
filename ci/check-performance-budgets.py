#!/usr/bin/env python3
"""Validate the v1 performance budget and baseline manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ROOT / "conformance" / "v1.0" / "performance-budgets.json"
BASELINE = ROOT / "conformance" / "v1.0" / "performance-baseline.json"

REQUIRED_OPERATIONS = {
    "wire.config_decode",
    "wire.request_decode_non_submit",
    "wire.submit_decode",
    "transport.segmented_region_access",
    "state.object_lookup",
    "device.command_dispatch",
    "device.submission_admission",
    "device.polling",
    "device.reset",
    "provider.explicit_transfer",
    "provider.copy_path_diagnostics",
}


def load(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate checked-in manifests")
    args = parser.parse_args()
    require(args.check, "only --check is supported")

    budgets = load(BUDGETS)
    baseline = load(BASELINE)
    require(budgets["schema"] == "virtio-accel-performance-budgets-1", "bad budget schema")
    require(budgets["protocol"] == "1.0", "bad budget protocol")
    require(baseline["schema"] == "virtio-accel-performance-baseline-1", "bad baseline schema")
    require(baseline["budgets"] == "performance-budgets.json", "baseline does not reference budgets")

    operations = budgets["operations"]
    seen = {entry["id"] for entry in operations}
    missing = REQUIRED_OPERATIONS - seen
    extra = seen - REQUIRED_OPERATIONS
    require(not missing, f"missing performance budget operations: {sorted(missing)}")
    require(not extra, f"unknown performance budget operations: {sorted(extra)}")

    for entry in operations:
        op_id = entry["id"]
        for field in [
            "path",
            "complexity",
            "allocation_profile",
            "copy_profile",
            "permitted_copy_boundary",
            "allocates_from_unvalidated_guest_count",
            "thresholds",
        ]:
            require(field in entry, f"{op_id} missing {field}")
        require(
            entry["allocates_from_unvalidated_guest_count"] is False,
            f"{op_id} allocates from an unvalidated guest count",
        )

    baseline_ops = {entry["operation"] for entry in baseline["deterministic_results"]}
    require(
        {"wire.request_decode_non_submit", "wire.submit_decode", "device.submission_admission"}
        <= baseline_ops,
        "baseline omits deterministic hot-path results",
    )
    print(f"performance budget manifests are complete ({len(operations)} operations)")


if __name__ == "__main__":
    main()

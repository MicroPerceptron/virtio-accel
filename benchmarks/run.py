#!/usr/bin/env python3
"""Capture Divan output and provenance without replacing framework statistics."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def capture(*args: str) -> str:
    r = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else f"unavailable: {r.stderr.strip()}"

def main() -> int:
    # Redirected Windows consoles otherwise use cp1252, which cannot encode Divan's tree.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["portable", "coreml", "openvino", "vulkan", "hexagon", "xdna"], required=True)
    p.add_argument("--device", default="")
    p.add_argument("--case", default=None)
    p.add_argument("--phase", choices=["load_unload", "upload", "download", "warm_execution", "round_trip", "pipeline_execution", "all"], default="warm_execution")
    p.add_argument("--in-flight", type=int, default=1, help="pipeline slots, bounded by provider resources")
    p.add_argument("--requests", type=int, help="completed requests per pipeline sample, default max(256, depth)")
    p.add_argument("--input-mode", choices=["shared", "per-slot"], default="shared")
    p.add_argument("--output", type=Path)
    p.add_argument("--runtime-version", default="unrecorded", help="operator-supplied vendor runtime version")
    p.add_argument("divan_args", nargs=argparse.REMAINDER, help="Divan flags after --")
    a = p.parse_args()
    if a.backend in {"coreml", "openvino", "vulkan"} and not a.device:
        p.error("--device is required; no default accelerator or CPU fallback")
    if a.backend == "coreml" and a.device not in {"cpu-ane", "cpu-gpu", "all"}:
        p.error("Core ML --device must be cpu-ane, cpu-gpu or all")
    if a.backend in {"xdna", "hexagon", "portable"} and a.device:
        p.error("this backend has no selectable device argument; actual identity is recorded")
    case = a.case or ({"xdna": "MatmulBf16Decode", "vulkan": "MatmulF32Decode"}.get(a.backend, "MatmulF16Decode"))
    if not re.fullmatch(r"[A-Za-z0-9_]+", case):
        p.error("--case must be an exact benchmark case name")
    if a.in_flight < 1 or (a.requests is not None and a.requests < a.in_flight):
        p.error("require 1 <= --in-flight <= --requests")
    if (a.backend == "portable" or a.phase not in {"pipeline_execution", "all"}) and (a.in_flight != 1 or a.requests is not None or a.input_mode != "shared"):
        p.error("pipeline settings require a hardware --phase pipeline_execution (or all)")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (a.output or ROOT / "benchmarks" / "results" / f"{stamp}-{a.backend}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(VA_BENCH_BACKEND=a.backend, VA_BENCH_DEVICE=a.device, VA_BENCH_METADATA=str(out / "cases.jsonl"))
    env.update(VA_BENCH_IN_FLIGHT=str(a.in_flight), VA_BENCH_REQUESTS=str(a.requests or max(256, a.in_flight)), VA_BENCH_INPUT_MODE=a.input_mode)
    cmd = ["cargo", "bench", "--locked", "--manifest-path", "benchmarks/Cargo.toml", "--bench", "portable" if a.backend == "portable" else "hardware"]
    if a.backend != "portable":
        cmd += ["--features", a.backend]
    cmd += ["--", "--color", "never"]
    if a.backend != "portable":
        phase = "[^:]+" if a.phase == "all" else a.phase
        cmd += [f"{phase}::{case}$"]
    cmd += a.divan_args[1:] if a.divan_args[:1] == ["--"] else a.divan_args
    lock = ROOT / "benchmarks" / "Cargo.lock"
    meta = {
        "schema": 1, "started_utc": stamp, "command": cmd,
        "git_commit": capture("git", "rev-parse", "HEAD"), "git_status": capture("git", "status", "--porcelain=v1"),
        "tracked_diff_sha256": hashlib.sha256(capture("git", "diff", "--binary", "HEAD").encode()).hexdigest(),
        "rustc": capture("rustc", "-Vv"), "cargo": capture("cargo", "-V"),
        "os": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
        "backend": a.backend, "requested_device": a.device,
        "runtime_version_operator_supplied": a.runtime_version,
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "environment": {k: env[k] for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "VA_BENCH_IN_FLIGHT", "VA_BENCH_REQUESTS", "VA_BENCH_INPUT_MODE", "VA_BENCH_WARMUPS", "VA_BENCH_WAIT_US", "VA_BENCH_TIMEOUT_SECS", "VIRTIO_ACCEL_OPENVINO_LIBRARY", "VIRTIO_ACCEL_HRX_LIBRARY", "VIRTIO_ACCEL_HRX_DIR", "VIRTIO_ACCEL_HRX_LIB_DIR", "HRX_DIR", "VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN", "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VIRTIO_ACCEL_QNN_SDK_ROOT", "VIRTIO_ACCEL_QNN_LIB_DIR", "QNN_SDK_ROOT"] if k in env},
        "timing": "Divan host wall-clock statistics; report.txt is authoritative; no custom estimator",
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Writing results to {out}", flush=True)
    with (out / "report.txt").open("w", encoding="utf-8") as report:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert child.stdout is not None
        for line in child.stdout:
            report.write(line)
            print(line, end="", flush=True)
        code = child.wait()
    if code == 0 and a.backend != "portable" and not (out / "cases.jsonl").is_file():
        print("No hardware case executed; check the case/phase filter", file=sys.stderr)
        code = 2
    meta.update(exit_code=code, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return code

if __name__ == "__main__":
    raise SystemExit(main())

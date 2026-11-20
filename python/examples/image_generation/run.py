"""Measure and capture real image generation through an explicitly selected SDK endpoint."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import virtio_accel as va

from model import decoder, latent_batch, load_checkpoint, reference


def summarize(samples, batch):
    values = np.asarray(samples, dtype=np.float64)
    if values.size < 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("need at least three positive finite timing samples")
    median = float(np.median(values))
    return dict(
        median_ms=median / 1e6,
        p95_ms=float(np.percentile(values, 95)) / 1e6,
        images_per_second=batch * 1e9 / median,
        samples_ns=[int(x) for x in samples],
    )


def validate(actual, expected):
    if (
        actual.shape != expected.shape
        or actual.dtype != np.float16
        or not np.isfinite(actual).all()
    ):
        raise ValueError("invalid output shape, dtype or finite values")
    delta = actual.astype(np.float32) - expected.astype(np.float32)
    error = dict(
        max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta**2)))
    )
    # Compare image probabilities rather than arbitrary internal accumulation bits.
    if (
        error["max_abs"] > 0.025
        or error["rmse"] > 0.003
        or actual.min() < 0
        or actual.max() > 1
    ):
        raise ValueError(f"decoder diverged from independent reference: {error}")
    return error


def capture(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--provider",
        choices=("coreml", "openvino", "vulkan", "hexagon", "xdna", "virtio"),
        required=True,
    )
    p.add_argument("--device", choices=("cpu", "gpu", "npu", "hybrid"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    p.add_argument("--samples", type=int, default=101)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument(
        "--duration", type=float, default=1.0, help="minimum measured seconds per batch"
    )
    p.add_argument("--warmup-seconds", type=float, default=1.0)
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="capture a latent interpolation separately from timing",
    )
    p.add_argument("--frame-batch", type=int, default=100)
    args = p.parse_args()
    if (
        args.output.exists()
        or not 3 <= args.samples <= 10000
        or not 1 <= args.warmups <= 100
        or not 0 <= args.frames <= 240
        or not 0 <= args.duration <= 60
        or not 0 <= args.warmup_seconds <= 60
    ):
        p.error(
            "output must be new; samples 3..10000, warmups 1..100, frames 0..240, durations 0..60 seconds"
        )
    if len(set(args.batch)) != len(args.batch):
        p.error("batch sizes must be unique")
    for batch in [*args.batch, args.frame_batch]:
        if not 1 <= batch <= 4096:
            p.error("batch sizes must be 1..4096")
    devices = va.devices(args.device, provider=args.provider)
    if len(devices) != 1:
        p.error(
            f"expected exactly one {args.provider} {args.device} endpoint, got {devices}; {va.device_diagnostics()}"
        )
    device = devices[0]
    weights, metadata, sha = load_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True)
    report = dict(
        schema=1,
        status="running",
        utc=datetime.now(timezone.utc).isoformat(),
        device={k: v for k, v in asdict(device).items() if not k.startswith("_")},
        model=metadata,
        checkpoint_sha256=sha,
        os=platform.platform(),
        machine=platform.machine(),
        logical_cpus=os.cpu_count(),
        processor=(
            capture(["sysctl", "-n", "machdep.cpu.brand_string"])
            if platform.system() == "Darwin"
            else capture(["lscpu"])
        ),
        source_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("model.py", "run.py")
        },
        python=platform.python_version(),
        numpy=np.__version__,
        sdk=importlib.metadata.version("virtio-accel"),
        runtime_version=(
            importlib.metadata.version("openvino")
            if args.provider == "openvino"
            else platform.mac_ver()[0] if args.provider == "coreml" else None
        ),
        git_commit=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--porcelain=v1"]),
        timing_boundary="synchronous SDK call, host input copy/upload, execution/wait, output download and host copy",
        minimum_warmups=args.warmups,
        minimum_warmup_seconds=args.warmup_seconds,
        minimum_samples=args.samples,
        minimum_sample_seconds=args.duration,
        in_flight=1,
        dtype="float16",
        cases=[],
    )
    path = args.output / "report.json"

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        with decoder(weights) as function:
            for batch in args.batch:
                inputs = [latent_batch(batch, phase=phase) for phase in (0.0, 0.5, 1.0)]
                expected = [reference(x, weights) for x in inputs]
                start = time.perf_counter_ns()
                lowered = function.lower(inputs[0])
                trace_ns = time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                executable = lowered.compile(device)
                compile_ns = time.perf_counter_ns() - start
                with executable:
                    errors = [
                        validate(executable(x).to_numpy(), ref)
                        for x, ref in zip(inputs, expected)
                    ]
                    warmups = 0
                    warmup_start = time.perf_counter()
                    while (
                        warmups < args.warmups
                        or time.perf_counter() - warmup_start < args.warmup_seconds
                    ):
                        executable(inputs[warmups % 3]).to_numpy()
                        warmups += 1
                    samples = []
                    sample_start = time.perf_counter()
                    while (
                        len(samples) < args.samples
                        or time.perf_counter() - sample_start < args.duration
                    ):
                        if len(samples) == 100000:
                            raise RuntimeError(
                                "sample cap reached before minimum duration"
                            )
                        start = time.perf_counter_ns()
                        result = executable(inputs[len(samples) % 3]).to_numpy()
                        samples.append(time.perf_counter_ns() - start)
                    errors.append(validate(result, expected[(len(samples) - 1) % 3]))
                case = dict(
                    batch=batch,
                    warmups=warmups,
                    trace_ms=trace_ns / 1e6,
                    compile_ms=compile_ns / 1e6,
                    input_sha256=[
                        hashlib.sha256(x.tobytes()).hexdigest() for x in inputs
                    ],
                    validation=errors,
                    **summarize(samples, batch),
                )
                report["cases"].append(case)
                save()
                print(
                    f'{device.provider}/{device.kind} batch {batch}: {case["median_ms"]:.3f} ms; {case["images_per_second"]:,.0f} images/s',
                    flush=True,
                )
            if args.frames:
                x = latent_batch(args.frame_batch)
                with function.lower(x).compile(device) as executable:
                    frames = []
                    for index in range(args.frames):
                        x = latent_batch(
                            args.frame_batch, phase=2 * np.pi * index / args.frames
                        )
                        result = executable(x).to_numpy()
                        if index in (0, args.frames - 1):
                            validate(result, reference(x, weights))
                        frames.append(
                            np.rint(result * 255)
                            .astype(np.uint8)
                            .reshape(args.frame_batch, 28, 28)
                        )
                    np.savez_compressed(
                        args.output / "frames.npz", pixels=np.stack(frames)
                    )
                report["capture"] = dict(
                    frames=args.frames,
                    batch=args.frame_batch,
                    timing="separate untimed inference capture; playback is illustrative, not real-time",
                )
        report["status"] = "passed"
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()

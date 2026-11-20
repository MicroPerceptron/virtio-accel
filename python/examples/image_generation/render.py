"""Render an actual Intel NPU capture and matched measurements into README media."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BACKGROUND = "#0b1220"
PANEL = "#121e30"
TEXT = "#eff5fc"
MUTED = "#a4b4c9"
ACCENT = "#7ce7ce"


def comparison(directory, batch):
    reports = {
        kind: json.loads((directory / f"{kind}.json").read_text())
        for kind in ("cpu", "gpu", "npu")
    }
    baseline = reports["npu"]
    cases = {}

    def cpu_identity(report):
        # lscpu also includes live frequency scaling; compare only CPU identity.
        return [
            line.strip()
            for line in report["processor"].splitlines()
            if line.strip().startswith(
                ("Architecture:", "Model name:", "Vendor ID:", "CPU(s):")
            )
        ]

    if not cpu_identity(baseline):
        raise ValueError("missing CPU identity")
    for kind, report in reports.items():
        if (
            report["status"] != "passed"
            or report["device"]["provider"] != "openvino"
            or report["device"]["kind"] != kind
        ):
            raise ValueError(
                "requires passed, explicitly selected OpenVINO CPU/GPU/NPU reports"
            )
        for key in (
            "checkpoint_sha256",
            "source_sha256",
            "os",
            "machine",
            "logical_cpus",
            "sdk",
            "runtime_version",
            "dtype",
            "timing_boundary",
            "minimum_warmups",
            "minimum_warmup_seconds",
            "minimum_samples",
            "minimum_sample_seconds",
            "in_flight",
        ):
            if report[key] != baseline[key]:
                raise ValueError(f"incomparable reports: {key}")
        if cpu_identity(report) != cpu_identity(baseline):
            raise ValueError("CPU identities differ")
        cases[kind] = next(case for case in report["cases"] if case["batch"] == batch)
        samples = np.asarray(cases[kind]["samples_ns"], dtype=np.float64)
        if samples.size < 3 or not np.isfinite(samples).all() or (samples <= 0).any():
            raise ValueError("invalid measured samples")
        if not np.isclose(
            cases[kind]["images_per_second"], batch * 1e9 / np.median(samples)
        ):
            raise ValueError("throughput does not match recorded samples")
    if any(
        case["input_sha256"] != cases["npu"]["input_sha256"] for case in cases.values()
    ):
        raise ValueError("input hashes differ")
    return reports, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=Path,
        required=True,
        help="cpu.json, gpu.json, npu.json and NPU frames.npz",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--font", type=Path, required=True)
    p.add_argument("--bold-font", type=Path, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.fps <= 60:
        p.error("output must be new and fps must be 1..60")
    reports, cases = comparison(args.results, args.batch)
    with np.load(args.results / "frames.npz", allow_pickle=False) as archive:
        pixels = archive["pixels"]
    capture = reports["npu"]["capture"]
    if (
        pixels.dtype != np.uint8
        or pixels.shape != (capture["frames"], 100, 28, 28)
        or capture["batch"] != 100
    ):
        raise ValueError("renderer needs the NPU 100-image capture")
    args.output.mkdir(parents=True)
    fonts = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(
                str(args.bold_font if bold else args.font), size
            )
        return fonts[key]

    frames = []
    peak = max(case["images_per_second"] for case in cases.values())
    for index, images in enumerate(pixels):
        canvas = Image.new("RGB", (1280, 800), BACKGROUND)
        draw = ImageDraw.Draw(canvas)

        def text(x, y, value, size=20, fill=TEXT, bold=False):
            draw.text((x, y), value, font=font(size, bold), fill=fill)

        text(44, 28, "virtio-accel  /  INFERENCE SHOWCASE", 19, ACCENT, True)
        text(44, 68, "From latent space to pixels", 46, bold=True)
        text(
            46,
            132,
            "Trained image decoder  /  FP16  /  Python SDK → OpenVINO → Intel NPU",
            22,
            MUTED,
        )
        draw.rounded_rectangle((44, 188, 548, 692), radius=18, fill=PANEL)
        for tile, image in enumerate(images):
            image = Image.fromarray(image).resize((44, 44), Image.Resampling.BICUBIC)
            image = ImageOps.colorize(image, "#0b1220", "#c3fff0")
            canvas.paste(image, (58 + tile % 10 * 48, 202 + tile // 10 * 48))
        draw.rounded_rectangle((576, 188, 1236, 692), radius=18, fill=PANEL)
        text(604, 210, "Intel NPU · generated pixels", 23, ACCENT, True)
        text(600, 246, f'{cases["npu"]["images_per_second"]:,.0f}', 64, bold=True)
        text(604, 321, f"28 × 28 images/s  ·  batch {args.batch}", 23, MUTED)
        text(
            604,
            363,
            f'Median {cases["npu"]["median_ms"]:.2f} ms   /   p95 {cases["npu"]["p95_ms"]:.2f} ms',
            20,
        )
        draw.line((604, 407, 1208, 407), fill="#2b3b51")
        text(604, 423, "SAME HOST · MODEL · INPUTS · BATCH", 17, MUTED, True)
        for row, kind in enumerate(("cpu", "gpu", "npu")):
            y = 462 + row * 60
            color = ACCENT if kind == "npu" else "#6a94be"
            text(604, y, kind.upper(), 20, color, True)
            draw.rounded_rectangle(
                (
                    664,
                    y + 3,
                    664 + max(2, 370 * cases[kind]["images_per_second"] / peak),
                    y + 23,
                ),
                radius=4,
                fill=color,
            )
            text(1050, y, f'{cases[kind]["images_per_second"]/1000:.1f}k/s', 20)
        text(604, 649, "SDK round trip · one invocation in flight", 18, MUTED)
        text(46, 712, "100 latent samples · conditioned on digits 0–9", 19, MUTED)
        text(
            46,
            748,
            f"Native output capture  /  {args.fps} fps illustrative playback  /  not real-time",
            18,
            MUTED,
        )
        text(838, 748, "686,864 learned parameters", 18, MUTED)
        # A subtle progress track makes the latent interpolation apparent.
        draw.rectangle(
            (44, 786, 44 + 1192 * (index + 1) / len(pixels), 789), fill=ACCENT
        )
        frames.append(canvas)
    frames[0].save(args.output / "poster.png", optimize=True)
    palette = frames[0].quantize(colors=128)
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    gif_frames[0].save(
        args.output / "showcase.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        for index, frame in enumerate(frames):
            frame.save(Path(tmp) / f"{index:03d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(Path(tmp) / "%03d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "showcase.mp4"),
            ],
            check=True,
        )
    manifest = dict(
        schema=1,
        capture_sha256=hashlib.sha256(
            (args.results / "frames.npz").read_bytes()
        ).hexdigest(),
        report_sha256={
            kind: hashlib.sha256(
                (args.results / f"{kind}.json").read_bytes()
            ).hexdigest()
            for kind in reports
        },
        frames=len(pixels),
        fps=args.fps,
        benchmark_batch=args.batch,
        font_sha256=[
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.font, args.bold_font)
        ],
        playback="illustrative, not real-time; measurements recorded separately",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

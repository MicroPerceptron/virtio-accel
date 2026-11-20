# Image-generation capture and measurements

Measured 2026-11-20 on Intel Core Ultra 5 226V / Lunar Lake, OpenVINO 2026.4.1.
These are single-run application measurements, not dedicated-lab hardware rankings.
CPU frequency/power state, temperature and background load were not controlled;
raw samples retain scheduling outliers. No energy-use claim is made.

| Batch | CPU median / p95 (ms) | GPU median / p95 (ms) | NPU median / p95 (ms) |
| --- | --- | --- | --- |
| 1 | 0.165 / 0.462 | 0.314 / 0.610 | 0.911 / 1.629 |
| 16 | 0.378 / 1.265 | 0.330 / 0.625 | 0.915 / 1.671 |
| 64 | 2.076 / 3.466 | 0.409 / 0.845 | 1.448 / 1.827 |
| 256 | 5.847 / 10.470 | 0.876 / 1.671 | 1.543 / 4.862 |

At batch 256 the median-derived throughput is CPU 43,783, GPU 292,127 and NPU
165,883 generated 28 × 28 images/second. The NPU is about 3.8× CPU throughput
for this case; GPU is faster still. At batch 1, CPU has lower latency than NPU.
A tiny decoder does not predict performance for a full diffusion model or language model.

Both Core ML CPU/ANE and CPU/GPU policies also passed all four batches on Apple M4.
The batch-256 sustained medians were 0.600 ms and 1.254 ms respectively. These
policies permit CPU participation; no per-operation placement was collected.
The two Mac reports are additional validation, not a cross-machine comparison.

Reports in [results/](results/) preserve all raw samples, numerical validation,
checkpoint/input/source hashes, timing boundaries, runtime/OS and CPU details.
The measured runner is commit `1d227f3`; source hashes identify its exact files.
Later edits add pre-allocation checkpoint-header validation and media/docs only;
the inference graph and measurement loop are unchanged.
Minimum warmup and sampling windows are each one second per batch, with at
least five warmups and 101 samples. One synchronous invocation is in flight.
Compilation, NumPy reference work and media rendering are outside those windows.

[frames.npz](results/frames.npz) is a separate capture from the explicitly selected
Intel NPU endpoint: 48 frames, each containing 100 generated images. Bicubic
upscaling and a fixed teal tint are presentation only. The animation uses an
illustrative playback rate, not the measured inference rate.

[manifest.json](manifest.json) links the rendered media to reports, raw capture
and rendering fonts by SHA-256. See the [example guide](../../python/examples/image_generation/README.md)
for model provenance, training, measurement and rendering commands.

The trained checkpoint and generated digit media use [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
MNIST attribution: Yann LeCun, Corinna Cortes and Christopher J. C. Burges.
The source MNIST dataset is not included. Rendering/training code follows the
repository's MIT OR Apache-2.0 license. Fonts are supplied locally, not redistributed.

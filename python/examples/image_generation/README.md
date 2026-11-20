# Learned image generation through the Python SDK

Generate 28 × 28 handwritten digits with a trained conditional VAE decoder.
The entire inference graph runs through `virtio-accel`: three dense layers,
ReLU activations and a sigmoid expressed with `tanh`. The host supplies a latent
sample and a digit label, then receives generated pixels. No training images
are read during inference. This is a compact generative model, not diffusion
or text-to-image generation.

## Run

From the repository root, with the Python SDK built from this branch installed:

```sh
python python/examples/image_generation/run.py \
  --checkpoint assets/inference/decoder.npz \
  --provider openvino --device npu --output /tmp/digits-npu \
  --batch 1 16 64 256 --samples 101 --frames 48
```

Select `--provider coreml --device npu` for the CPU/Neural Engine policy on
Apple Silicon, or `--device gpu` for CPU/GPU. Core ML chooses operation
placement; selecting the NPU policy does **not** prove ANE execution.
OpenVINO selects the named native CPU, GPU or NPU plugin explicitly. A missing
endpoint or unsupported graph fails; this example does not retry on CPU.
Other providers are accepted by the command but are not validated for this
model. All inference uses FP16; compressed weights are not exercised here.

An installed vendor runtime and compatible drivers are required. Inference
requires only the SDK's existing NumPy dependencies; PyTorch is used only by
the optional training command below. Output directories must be new.

## What is measured

`report.json` records raw timing samples, median and p95 latency, images per
second, separate tracing/compilation times, device/runtime versions, source
hashes, checkpoint hash, inputs and numerical error. Each batch compiles once,
validates three deterministic inputs against an independent NumPy reference,
warms up for at least one second and five calls, then measures at least one
second and 101 synchronous SDK calls with **one invocation in flight**.
Each sample includes input copying/upload, execution/wait and output download
plus the final NumPy copy. Training, reference computation, compilation and
frame rendering are excluded. Throughput is batch size divided by median
round-trip time, not an independently measured asynchronous saturation rate.

Compare CPU/GPU/NPU on the same machine with identical checkpoint, inputs,
batches and sampling settings. Close other accelerator workloads. These are
application measurements; use the repository's [Divan benchmark module](../../../benchmarks/README.md)
for calibrated primitive and in-flight pipeline experiments. A single run is
not a power-efficiency or general model-performance claim.

Validation requires FP16 output probabilities in [0, 1], maximum absolute
error <= 0.025 and RMSE <= 0.003 relative to explicit FP16 storage boundaries.
Those are image-probability tolerances, not a claim of bitwise equality or a
quality score. The saved animation is a separate, untimed sequence of native
inferences along a deterministic latent loop; its playback speed is illustrative.

## Model and reproduction

[`decoder.npz`](../../../assets/inference/decoder.npz) is a newly trained checkpoint, not random weights. It contains
686,864 FP16 parameters in a 42 → 512 → 512 → 784 decoder: 32 latent dimensions
and 10 one-hot class inputs. Its SHA-256 is
`3dca1f985621c1aa79f57333cd700bf706c09a7fd714a8db171ed18dc7aaa3ce`.
The archive contains only numeric arrays and JSON metadata; loading disables
pickle and checks member names, expanded size, shapes, dtypes and finite values.

The bundled checkpoint was trained with PyTorch 2.15.0 on Apple M4/MPS for
20 epochs, batch 256, seed 17, Adam at 0.001, using the 60,000 MNIST training
examples. The encoder is used only during training. Evaluation on the 10,000
held-out examples gave mean summed binary cross-entropy **74.1414** using
posterior means (before exporting the decoder to FP16). This reconstruction
metric does not measure sample realism or classification accuracy. Some generated
digits are ambiguous. Different training devices/library versions can produce
different weights despite the fixed seed.

```sh
python -m pip install 'torch==2.15.0'
python python/examples/image_generation/train.py \
  --device mps --epochs 20 --output /tmp/my-decoder.npz
```

Use `--device cpu` elsewhere. The script downloads only the four checksum-pinned
MNIST IDX archives from the [CVDF mirror](https://github.com/cvdfoundation/mnist),
and records their SHA-256 digests and training history in the checkpoint.

MNIST attribution: Yann LeCun, Corinna Cortes and Christopher J. C. Burges,
*The MNIST database of handwritten digits*. See the
[dataset description](https://www.tensorflow.org/datasets/catalog/mnist) and
[dataset license notice](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data).
Code follows the repository's MIT OR Apache-2.0 license. The bundled trained
checkpoint and generated digit media are distributed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), with this
attribution retained; the training dataset itself is not included.

## Recreate the media

Install Pillow (`python -m pip install 'Pillow>=12,<13'`) and `ffmpeg`. Put matched
reports at `cpu.json`, `gpu.json`, `npu.json` and the NPU's `frames.npz` in one
directory. Rendering verifies provider, checkpoint, source, CPU identity, inputs
and measurement settings before showing the comparison. With the bundled capture:

```sh
python python/examples/image_generation/render.py \
  --results assets/inference/results --output /tmp/digit-media \
  --font /path/to/regular.ttf --bold-font /path/to/bold.ttf
```

Use fonts installed on your machine (for example DejaVu Sans on Linux or Arial
on macOS/Windows). This writes a PNG poster, animated GIF, H.264 MP4 and a
provenance manifest. The bundled media used Pillow 12.3.0 and Arial. It displays
the batch-256 comparison, while the animated grid contains 100 separate latent
samples. See [all measured batches and raw results](../../../assets/inference/README.md).

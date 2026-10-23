"""Two-layer classifier inference with FP16 or FP32 weights and activations.

The demo uses a synthetic checkpoint to avoid model downloads. Replace the five
input arrays with a batch and trained weights to use the same inference graph.
"""

import argparse
from pathlib import Path

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


@va.jit
def logits(batch, hidden_weight, hidden_bias, output_weight, output_bias):
    hidden = vnp.maximum(batch @ hidden_weight + hidden_bias, 0)
    return hidden @ output_weight + output_bias


def demo(dtype="float16", export=None, device=None):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(7)
    shapes = ((16, 32), (32, 24), (24,), (24, 5), (5,))
    batch, w1, b1, w2, b2 = inputs = tuple(
        (rng.normal(size=shape) * scale).astype(dtype)
        for shape, scale in zip(shapes, (1.0, 0.25, 0.05, 0.25, 0.05))
    )
    lowered = logits.lower(*inputs)
    if device is None:
        scores = lowered.reference(*inputs).to_numpy()
    else:
        with lowered.compile(device) as executable:
            scores = executable(*inputs).to_numpy()
            print(f"Native target: {executable.device}")
    # Independent NumPy inference with storage rounding at each operator boundary.
    hidden = (batch.astype(np.float32) @ w1.astype(np.float32)).astype(dtype)
    hidden = np.maximum((hidden + b1).astype(dtype), 0)
    expected = (hidden.astype(np.float32) @ w2.astype(np.float32)).astype(dtype)
    expected = (expected + b2).astype(dtype)
    tolerance = 2e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(scores, expected, rtol=tolerance, atol=tolerance)
    if export is not None:
        Path(export).write_bytes(lowered.to_tosa())
    # Argmax is deliberately host postprocessing; it is outside the traced subset.
    classes = scores.argmax(axis=-1)
    print(f"{device or 'CPU reference'}, {dtype.name}: {len(classes)} classifications: {classes.tolist()}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--export", type=Path, help="write validated TOSA (requires native extension)")
    parser.add_argument("--device", help="native target, e.g. npu or gpu; default is CPU reference")
    args = parser.parse_args()
    demo(args.dtype, args.export, args.device)

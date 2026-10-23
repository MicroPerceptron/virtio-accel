"""FIR filtering of a noisy sampled signal using a batch of sliding windows.

Host code constructs windows; the compiled graph computes all window dot products.
This is valid convolution: no implicit boundary padding is applied.
"""

import argparse
from pathlib import Path

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


@va.jit
def filter_windows(windows, reversed_taps):
    return vnp.reshape(windows @ reversed_taps, (-1,))


def demo(dtype="float16", export=None, device=None):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(23)
    time = np.arange(512, dtype=np.float64) / 512
    signal = (np.sin(2 * np.pi * 5 * time) + rng.normal(0, 0.1, len(time))).astype(dtype)
    taps = np.hanning(17)
    taps = (taps / taps.sum()).astype(dtype)
    # This is a noncontiguous view. The frontend explicitly packs it into owned
    # storage; the example does not claim a zero-copy sliding-window device view.
    windows = np.lib.stride_tricks.sliding_window_view(signal, len(taps))
    reversed_taps = taps[::-1].reshape(-1, 1)
    lowered = filter_windows.lower(windows, reversed_taps)
    if device is None:
        filtered = lowered.reference(windows, reversed_taps).to_numpy()
    else:
        with lowered.compile(device) as executable:
            filtered = executable(windows, reversed_taps).to_numpy()
            print(f"Native target: {executable.device}")
    expected = np.convolve(signal.astype(np.float64), taps.astype(np.float64), mode="valid")
    tolerance = 1e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(filtered, expected, rtol=tolerance, atol=tolerance)
    if export is not None:
        Path(export).write_bytes(lowered.to_tosa())
    error = float(np.max(np.abs(filtered.astype(np.float64) - expected)))
    print(f"{device or 'CPU reference'}, {dtype.name}: {len(filtered)} filtered samples; "
          f"max error against float64 convolution {error:.3g}")
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--export", type=Path, help="write validated TOSA (requires native extension)")
    parser.add_argument("--device", help="native target, e.g. npu or gpu; default is CPU reference")
    args = parser.parse_args()
    demo(args.dtype, args.export, args.device)

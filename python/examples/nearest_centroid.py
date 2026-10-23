"""Pairwise squared distances for nearest-centroid classification or clustering.

Uses a matrix product instead of constructing the (points, centers, features)
temporary. FP16 inputs are explicitly widened for distance arithmetic.
"""

import argparse
from pathlib import Path

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


@va.jit
def squared_distances(points, centers):
    points = vnp.astype(points, vnp.float32)
    centers = vnp.astype(centers, vnp.float32)
    point_norms = vnp.sum(points * points, axis=1, keepdims=True)
    center_norms = vnp.sum(centers * centers, axis=1, keepdims=True).T
    return vnp.maximum(point_norms + center_norms - 2 * (points @ centers.T), 0)


def demo(dtype="float16", export=None):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(11)
    centers = rng.normal(size=(8, 12)).astype(dtype)
    labels = np.arange(64) % len(centers)
    points = (centers[labels].astype(np.float32) + rng.normal(0, 0.05, (64, 12))).astype(dtype)
    lowered = squared_distances.lower(points, centers)
    distances = lowered.reference(points, centers).to_numpy()
    # A different formulation and higher precision provide a numerical oracle.
    differences = points.astype(np.float64)[:, None, :] - centers.astype(np.float64)[None, :, :]
    expected = np.sum(differences * differences, axis=-1)
    np.testing.assert_allclose(distances, expected, rtol=2e-5, atol=2e-5)
    # The norm identity loses accuracy for nearly equal, large vectors; FP32 helps
    # but does not remove cancellation. Compare absolute error near zero.
    predicted = distances.argmin(axis=-1)
    np.testing.assert_array_equal(predicted, labels)
    if export is not None:
        Path(export).write_bytes(lowered.to_tosa())
    print(f"CPU reference, {dtype.name} inputs / float32 distances: "
          f"{len(points)} points assigned correctly to {len(centers)} centers")
    return distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--export", type=Path, help="write validated TOSA (requires native extension)")
    args = parser.parse_args()
    demo(args.dtype, args.export)

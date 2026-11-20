"""A trained conditional image decoder expressed entirely through the Python SDK."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp

SHAPES = {
    "w0": (42, 512),
    "b0": (512,),
    "w1": (512, 512),
    "b1": (512,),
    "w2": (512, 784),
    "b2": (784,),
}


def load_checkpoint(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("checkpoint exceeds 8 MiB limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (
            len(members) != len(SHAPES) + 1
            or {m.filename for m in members}
            != {k + ".npy" for k in (*SHAPES, "metadata")}
            or sum(m.file_size for m in members) > 8 * 1024 * 1024
        ):
            raise ValueError("invalid or oversized checkpoint archive")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SHAPES) | {"metadata"}:
            raise ValueError("unexpected checkpoint members")
        weights = {}
        for key, shape in SHAPES.items():
            value = archive[key]
            if (
                value.shape != shape
                or value.dtype != np.float16
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid tensor {key}")
            weights[key] = value.copy()
        metadata = json.loads(str(archive["metadata"]))
    if metadata.get("schema") != 1 or metadata.get("architecture") != [
        42,
        512,
        512,
        784,
    ]:
        raise ValueError("unsupported decoder architecture")
    return weights, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def decoder(weights):
    """Return a traceable decoder with immutable FP16 learned constants."""
    w = {k: vnp.asarray(value) for k, value in weights.items()}

    def forward(inputs):
        x = vnp.maximum(inputs @ w["w0"] + w["b0"], 0)
        x = vnp.maximum(x @ w["w1"] + w["b1"], 0)
        logits = x @ w["w2"] + w["b2"]
        # Stable sigmoid using an admitted primitive, including its final FP16 rounding.
        return vnp.tanh(logits * 0.5) * 0.5 + 0.5

    return va.jit(forward)


def reference(inputs, weights):
    """Independent NumPy reference with each FP16 storage boundary made explicit."""
    x = inputs
    for i in range(3):
        x = (x.astype(np.float32) @ weights[f"w{i}"].astype(np.float32)).astype(
            np.float16
        )
        x = (x + weights[f"b{i}"]).astype(np.float16)
        if i < 2:
            x = np.maximum(x, np.float16(0))
    x = (x * np.float16(0.5)).astype(np.float16)
    x = np.tanh(x.astype(np.float32)).astype(np.float16)
    return ((x * np.float16(0.5)).astype(np.float16) + np.float16(0.5)).astype(
        np.float16
    )


def latent_batch(batch, seed=23, phase=0.0):
    """Deterministic class-conditioned latent samples; phase traces a smooth loop."""
    if not 1 <= batch <= 4096:
        raise ValueError("batch must be between 1 and 4096")
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, batch, 32))
    z = (a * np.cos(phase) + b * np.sin(phase)).astype(np.float16)
    labels = np.eye(10, dtype=np.float16)[np.arange(batch) % 10]
    return np.concatenate((z, labels), axis=-1)

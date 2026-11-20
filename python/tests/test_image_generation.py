"""Portable checks for the real learned decoder; no training or downloads."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXAMPLE = Path(__file__).parents[1] / "examples" / "image_generation"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load("image_decoder", EXAMPLE / "model.py")
with patch.dict("sys.modules", {"model": model}):
    runner = load("image_runner", EXAMPLE / "run.py")


class ImageGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights, cls.metadata, cls.sha = model.load_checkpoint(
            EXAMPLE.parents[2] / "assets" / "inference" / "decoder.npz"
        )

    def test_learned_model_exports_and_matches_independent_oracle(self):
        self.assertEqual(
            self.sha, "3dca1f985621c1aa79f57333cd700bf706c09a7fd714a8db171ed18dc7aaa3ce"
        )
        self.assertEqual(self.metadata["held_out_examples"], 10000)
        with model.decoder(self.weights) as function:
            for batch in (1, 16):
                x = model.latent_batch(batch)
                lowered = function.lower(x)
                runner.validate(
                    lowered.reference(x).to_numpy(), model.reference(x, self.weights)
                )
                if importlib.util.find_spec("virtio_accel._native"):
                    self.assertEqual(lowered.to_tosa()[4:8], b"TOSA")
        # Class conditioning changes the output even for the same latent vector.
        x = model.latent_batch(2)
        x[1, :32] = x[0, :32]
        images = model.reference(x, self.weights)
        self.assertGreater(float(np.mean(np.abs(images[0] - images[1]))), 0.05)

    def test_latent_loop_repeats_and_preserves_class_labels(self):
        x = model.latent_batch(20)
        np.testing.assert_array_equal(x, model.latent_batch(20))
        np.testing.assert_allclose(
            x, model.latent_batch(20, phase=2 * np.pi), atol=1e-6
        )
        np.testing.assert_array_equal(x[:, 32:].argmax(axis=1), np.arange(20) % 10)
        for bad in (0, 4097):
            with self.assertRaises(ValueError):
                model.latent_batch(bad)

    def test_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            for replacement in (
                np.zeros((1,), dtype=np.float16),
                np.full((512,), np.nan, dtype=np.float16),
            ):
                np.savez_compressed(
                    path,
                    **{**self.weights, "b0": replacement},
                    metadata=np.array(json.dumps(self.metadata)),
                )
                with self.assertRaisesRegex(ValueError, "invalid tensor b0"):
                    model.load_checkpoint(path)
            np.savez_compressed(path, unexpected=np.zeros(1))
            with self.assertRaises(ValueError):
                model.load_checkpoint(path)

    def test_timing_statistics_keep_outlier_and_validation_rejects_bad_pixels(self):
        result = runner.summarize([1_000_000, 2_000_000, 100_000_000], 16)
        self.assertEqual(result["median_ms"], 2)
        self.assertEqual(result["images_per_second"], 8000)
        self.assertGreater(result["p95_ms"], 90)
        self.assertEqual(len(result["samples_ns"]), 3)
        expected = np.zeros((1, 784), dtype=np.float16)
        for bad in (
            np.ones_like(expected),
            np.full_like(expected, np.nan),
            expected.astype(np.float32),
        ):
            with self.assertRaises(ValueError):
                runner.validate(bad, expected)

    @unittest.skipUnless(
        os.environ.get("VIRTIO_ACCEL_PYTHON_TEST_DEVICE"),
        "hardware examples are opt-in",
    )
    def test_native_decoder(self):
        from hardware_support import requested_device

        device = requested_device()
        if device.provider not in ("coreml", "openvino"):
            self.skipTest("image decoder currently validated on Core ML and OpenVINO")
        x = model.latent_batch(16)
        with model.decoder(self.weights) as function:
            with function.lower(x).compile(device) as executable:
                runner.validate(
                    executable(x).to_numpy(), model.reference(x, self.weights)
                )

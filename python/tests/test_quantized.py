"""Quantized storage, portable graph validation and opt-in device regressions."""
import importlib.util
import os
import unittest

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp
import virtio_accel.quantized as q

NATIVE = importlib.util.find_spec('virtio_accel._native') is not None


class QuantizedTests(unittest.TestCase):
    def check_graph(self, function, args, expected, *, exact=True):
        lowered = va.jit(function).lower(*args)
        actual = lowered.reference(*args).to_numpy()
        if exact:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        if NATIVE:
            self.assertTrue(lowered.to_tosa())
        return lowered

    def test_fp8_storage_casts_and_cache_movement(self):
        for dtype in (vnp.float8_e4m3fn, vnp.float8_e5m2):
            raw = np.arange(256, dtype=np.uint8).view(dtype).reshape(16, 16)
            self.check_graph(lambda x: x.T.reshape(256).astype(vnp.float32), (raw,), raw.T.reshape(256).astype(np.float32))
            lowered = va.jit(lambda x: x.T).lower(raw)
            np.testing.assert_array_equal(lowered.reference(raw).to_numpy().view(np.uint8), raw.T.view(np.uint8))
            if NATIVE:
                self.assertTrue(lowered.to_tosa())
            x = np.array([-2, -.5, 0, .5, 2], np.float32)
            self.check_graph(lambda x: x.astype(dtype), (x,), x.astype(dtype))

    def test_integer_matmul_preserves_accumulator_precision(self):
        a = np.full((1, 1, 2048), 127, np.int8)
        b = np.full((1, 2048, 1), 127, np.int8)
        b[0, 0, 0] = 126  # An odd INT32 result above 2**24 must not go through FP32.
        expected = a.astype(np.int32) @ b.astype(np.int32)
        self.check_graph(lambda a, b: a @ b, (a, b), expected)

    def test_integer_weights_all_nibbles_and_int8_extremes(self):
        x = np.eye(16, dtype=np.float16)
        raw = np.arange(0x10, 0x100, 0x22, dtype=np.uint8)
        scales = np.array([[.5, 2]], np.float16)
        expected = (np.arange(-8, 8, dtype=np.int8)[(np.arange(16) + 8) % 16].astype(np.float16)
                    * np.repeat(scales[0], 8)).reshape(16, 1)
        f = lambda x: q.int4_matmul(x, raw, scales, group_size=8)
        lowered = self.check_graph(f, (x,), expected)
        self.assertIn('cast', [n['op'] for n in lowered.to_dict()['nodes']])
        raw[:] = 0  # The lowered snapshot is immutable.
        np.testing.assert_array_equal(lowered.reference(x).to_numpy(), expected)
        w = np.array([[-128, -8, 7, 127]], np.int8)
        self.check_graph(lambda x: q.int8_matmul(x, w, np.ones((1, 1), np.float16), group_size=4),
                         (np.eye(4, dtype=np.float16),), w.T.astype(np.float16))

    def test_palette_all_indices_and_groups(self):
        raw = np.arange(0x10, 0x100, 0x22, dtype=np.uint8)
        palettes = np.arange(32, dtype=np.float16).reshape(1, 2, 16) / np.float16(4)
        expected = np.r_[palettes[0, 0, :8], palettes[0, 1, 8:]].reshape(16, 1)
        self.check_graph(lambda x: q.palette4_matmul(x, raw, palettes, group_size=8),
                         (np.eye(16, dtype=np.float16),), expected)

    def test_nvfp4_runtime_bytes_and_scale_order(self):
        x = np.eye(16, dtype=np.float16)
        weights = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]], np.uint8).view(np.int8)
        scales = np.array([[.5]], vnp.float8_e4m3fn).view(np.int8)
        values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float16)
        self.check_graph(lambda x, w, s: q.nvfp4_matmul(x, w, s, tensor_scale=2),
                         (x, weights, scales), values.reshape(16, 1))

    def test_rejects_ambiguous_types_and_invalid_geometry(self):
        for dtype in (vnp.float8_e4m3fn, vnp.float8_e5m2, np.int8, np.int32):
            a = vnp.ones((2, 2), dtype=dtype)
            for f in (lambda: a + a, lambda: vnp.exp(a), lambda: a.sum()):
                with self.assertRaises(TypeError):
                    f()
            with self.assertRaises(TypeError):
                a + vnp.ones((2, 2))
        with self.assertRaises(TypeError):
            vnp.ones(2).astype(np.int8)
        x = np.ones((1, 16), np.float16)
        for scale in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                q.int4_matmul(x, bytes(8), np.array([[scale]], np.float16), group_size=16)
        for size in (0, 3, 32):
            with self.assertRaises(ValueError):
                q.int4_matmul(x, bytes(8), np.ones((1, 1), np.float16), group_size=size)
        with self.assertRaises(ValueError):
            q.int4_matmul(x, bytes(7), np.ones((1, 1), np.float16), group_size=16)
        with self.assertRaises(TypeError):
            va.jit(lambda x, w: q.int8_matmul(x, w, np.ones((1, 1), np.float16), group_size=16)).lower(x, np.ones((1, 16), np.int8))
        with self.assertRaises(TypeError):
            q.int4_matmul(x, bytes(8), np.ones((1, 1), np.float32), group_size=16)


@unittest.skipUnless(os.environ.get('VIRTIO_ACCEL_PYTHON_TEST_DEVICE'), 'opt-in hardware tests')
class QuantizedHardwareTests(unittest.TestCase):
    def setUp(self):
        from hardware_support import requested_device
        self.device = requested_device()
        if self.device.provider not in ('coreml', 'openvino'):
            self.skipTest('quantized execution currently targets Core ML and OpenVINO')

    def execute(self, fn, args, expected):
        with va.jit(fn, device=self.device) as f:
            for _ in range(2):
                np.testing.assert_array_equal(f(*args).to_numpy(), expected)

    def test_compressed_weights(self):
        x = np.eye(16, dtype=np.float16)
        scales = np.ones((1, 1), np.float16)
        self.execute(lambda x: q.int4_matmul(x, bytes([0x21] * 8), scales, group_size=16),
                     (x,), np.tile([1, 2], 8).astype(np.float16).reshape(16, 1))
        w = np.arange(-8, 8, dtype=np.int8).reshape(1, 16)
        self.execute(lambda x: q.int8_matmul(x, w, scales, group_size=16), (x,), w.T.astype(np.float16))
        if self.device.provider == 'coreml':
            table = (np.arange(16, dtype=np.float16) / np.float16(4)).reshape(1, 1, 16)
            self.execute(lambda x: q.palette4_matmul(x, bytes([0x21] * 8), table, group_size=16),
                         (x,), np.tile([.25, .5], 8).astype(np.float16).reshape(16, 1))

    def test_nvfp4_inputs_can_change_without_recompilation(self):
        x = np.ones((1, 16), np.float16)
        w = np.full((1, 8), 0x22, np.int8)
        s = np.array([[1]], vnp.float8_e4m3fn).view(np.int8)
        with va.jit(lambda x, w, s: q.nvfp4_matmul(x, w, s), device=self.device) as f:
            np.testing.assert_array_equal(f(x, w, s).to_numpy(), [[16]])
            s = np.array([[2]], vnp.float8_e4m3fn).view(np.int8)
            np.testing.assert_array_equal(f(x, w, s).to_numpy(), [[32]])
            self.assertEqual(len(f._executables), 1)

    def test_fp8_decode_and_integer_accumulator(self):
        if self.device.provider == 'openvino' and self.device.kind == 'npu':
            self.skipTest('OpenVINO NPU mixed FP16 tier only')
        for dtype in (vnp.float8_e4m3fn, vnp.float8_e5m2):
            raw = np.arange(256, dtype=np.uint8).view(dtype).reshape(1, 256)
            self.execute(lambda x: x.astype(np.float32), (raw,), raw.astype(np.float32))
            self.execute(lambda x: x.T, (raw,), raw.T)
            values = np.array([[-2, -.5, 0, .5, 2]], np.float32)
            self.execute(lambda x: x.astype(dtype), (values,), values.astype(dtype))
        a = np.full((1, 1, 2048), 127, np.int8)
        b = np.full((1, 2048, 1), 127, np.int8)
        b[0, 0, 0] = 126
        if self.device.provider == 'coreml':
            with self.assertRaises(RuntimeError):
                va.jit(lambda a, b: a @ b).lower(a, b).compile(self.device)
            a, b = a[:, :, :1024].copy(), b[:, :1024, :].copy()
        self.execute(lambda a, b: a @ b, (a, b), a.astype(np.int32) @ b.astype(np.int32))

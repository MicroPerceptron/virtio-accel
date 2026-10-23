import importlib.util
import json
import unittest

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


class Float16Tests(unittest.TestCase):
    def test_storage_casts_promotion_and_cache(self):
        x = np.ones(3, np.float16)
        a = vnp.asarray(x)
        self.assertEqual(a.dtype, np.float16)
        self.assertEqual((a + 1).dtype, np.float16)
        self.assertEqual((a + np.float32(1)).dtype, np.float32)
        self.assertEqual(a.astype(vnp.float32).dtype, np.float32)
        f = va.jit(lambda x, y: x + y)
        mixed = f.lower(x, x.astype(np.float32))
        self.assertIn('cast', [n['op'] for n in mixed.to_dict()['nodes']])
        self.assertEqual(mixed.reference(x, x.astype(np.float32)).dtype, np.float32)
        self.assertIsNot(f.lower(x, x), f.lower(x.astype(np.float32), x.astype(np.float32)))

    def test_fp32_accumulation_and_final_rounding(self):
        a = np.array([[400, 400]], np.float16)
        b = np.array([[400], [-400]], np.float16)
        f = va.jit(lambda a, b: a @ b, backend='reference')
        np.testing.assert_array_equal(f(a, b).to_numpy(), [[0]])
        x = np.full((3, 4), 40000, np.float16)
        mean = va.jit(lambda x: vnp.mean(x), backend='reference')(x)
        self.assertEqual(mean.dtype, np.float16)
        self.assertEqual(mean.to_numpy(), 40000)
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, 9)).astype(np.float16)
        for dtype in (np.float16, np.float32):
            f = va.jit(lambda x: vnp.sum(x, axis=(0, 1), dtype=dtype), backend='reference')
            expected = x.astype(np.float32).sum(axis=0, keepdims=True).sum(axis=1).reshape(()).astype(dtype)
            np.testing.assert_array_equal(f(x).to_numpy(), expected)

    def test_unary_reference_and_eager_rounding(self):
        x = np.array([.125, .5, 1, 2], np.float16)
        for name in ('exp', 'log', 'tanh', 'sin', 'cos', 'abs', 'negative'):
            with self.subTest(name=name):
                fn = getattr(vnp, name)
                expected = getattr(np, name)(x.astype(np.float32)).astype(np.float16)
                for result in (fn(x), va.jit(fn, backend='reference')(x)):
                    self.assertEqual(result.dtype, np.float16)
                    np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_constant_bits_and_bounds(self):
        bits = np.arange(65536, dtype=np.uint16)
        x = bits.view(np.float16)
        f = va.jit(lambda unused: vnp.asarray(x))
        lowered = f.lower(np.ones(1, np.float16))
        np.testing.assert_array_equal(lowered.reference(np.ones(1, np.float16)).to_numpy().view(np.uint16), bits)
        constant = next(n for n in lowered.to_dict()['nodes'] if n['op'] == 'const')
        self.assertEqual(bytes.fromhex(constant['attrs']['data']), x.astype('<f2').tobytes())
        from virtio_accel._array import checked_shape
        self.assertEqual(checked_shape((2**27,), np.float16), (2**27,))
        with self.assertRaisesRegex(ValueError, '256 MiB'):
            checked_shape((2**27,), np.float32)
        with self.assertRaisesRegex(ValueError, '256 MiB'):
            vnp.asarray(np.broadcast_to(np.float16(1), (2**28,)))

    @unittest.skipUnless(importlib.util.find_spec('virtio_accel._native'), 'native extension not built')
    def test_native_fp16_operators_and_casts(self):
        x = np.ones((2, 2), np.float16)
        functions = [lambda x: x @ x, lambda x: x * x, lambda x: x + x,
                     lambda x: x - x, lambda x: vnp.minimum(x, 1),
                     lambda x: vnp.maximum(x, 1), lambda x: vnp.mean(x),
                     lambda x: vnp.sum(x), lambda x: x.T,
                     lambda x: x.astype(vnp.float32).astype(vnp.float16)]
        functions += [getattr(vnp, n) for n in ('exp', 'log', 'tanh', 'sin', 'cos', 'abs', 'negative')]
        for fn in functions:
            self.assertEqual(va.jit(fn).lower(x).to_tosa()[4:8], b'TOSA')
        from virtio_accel._native import build_tosa
        doc = va.jit(lambda x: x).lower(x).to_dict()
        doc['nodes'][0]['dtype'] = 'float64'
        with self.assertRaises(ValueError):
            build_tosa(json.dumps(doc))

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


class BFloat16Tests(unittest.TestCase):
    def test_exact_storage_constants_and_promotion(self):
        bits = np.arange(65536, dtype=np.uint16)
        x = bits.view(vnp.bfloat16)
        lowered = va.jit(lambda unused: vnp.asarray(x)).lower(np.ones(1, np.float32))
        actual = lowered.reference(np.ones(1, np.float32)).to_numpy()
        np.testing.assert_array_equal(actual.view(np.uint16), bits)
        a = vnp.ones((4,), dtype=vnp.bfloat16)
        self.assertEqual((a + 1).dtype, np.dtype(vnp.bfloat16))
        self.assertEqual((a + np.float16(1)).dtype, np.dtype(np.float32))
        f = va.jit(lambda x: x)
        self.assertIsNot(f.lower(np.ones(4, np.float16)), f.lower(a))

    def test_bf16_matmul_accumulates_and_returns_fp32(self):
        a = np.array([[[1, 2], [3, 4]]], dtype=vnp.bfloat16)
        b = np.array([[[2, 0], [0, 3]]], dtype=vnp.bfloat16)
        lowered = va.jit(lambda a, b: a @ b).lower(a, b)
        expected = a.astype(np.float32) @ b.astype(np.float32)
        actual = lowered.reference(a, b)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(actual.to_numpy(), expected)
        self.assertEqual([n['op'] for n in lowered.to_dict()['nodes']], ['input', 'input', 'matmul'])
        with self.assertRaises(ValueError):
            vnp.matmul(a, np.ones((2, 2, 2), dtype=vnp.bfloat16))

    def test_shared_xdna_fixtures(self):
        for name, fn, args in (
            ('bf16_identity', lambda x: x, (np.ones(1024, vnp.bfloat16),)),
            ('bf16_matmul', lambda a, b: a @ b, (np.ones((1, 32, 64), vnp.bfloat16), np.ones((1, 64, 32), vnp.bfloat16))),
        ):
            fixture = json.loads((Path(__file__).parent / 'data' / (name + '.json')).read_text())
            self.assertEqual(va.jit(fn).lower(*args).to_dict(), fixture)

    @unittest.skipUnless(importlib.util.find_spec('virtio_accel._native'), 'native extension not built')
    def test_native_bf16_export(self):
        for fn, args in (
            (lambda x: x, (np.ones(1024, vnp.bfloat16),)),
            (lambda a, b: a @ b, (np.ones((1, 32, 64), vnp.bfloat16), np.ones((1, 64, 32), vnp.bfloat16))),
        ):
            self.assertEqual(va.jit(fn).lower(*args).to_tosa()[4:8], b'TOSA')

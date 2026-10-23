import importlib.util
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp
from virtio_accel import _runtime


NPU = dict(id='npu:0', kind='npu', provider='coreml', name='test ANE', cpu_participation=True)
GPU = dict(id='gpu:0', kind='gpu', provider='vulkan', name='test GPU', cpu_participation=False)


class DeviceTests(unittest.TestCase):
    def test_selector_alias_filtering_and_strict_placement(self):
        with patch.object(_runtime, '_discovery', return_value=dict(devices=[NPU, GPU], diagnostics=[])):
            self.assertEqual([d.id for d in va.devices('gpu')], ['gpu:0'])
            self.assertEqual(_runtime._resolve('npu', False).id, 'npu:0')
            self.assertEqual(_runtime._resolve('gpu:0', True).provider, 'vulkan')
            with self.assertRaisesRegex(RuntimeError, 'exclusive'):
                _runtime._resolve('npu', True)
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                _runtime._resolve('gpu:1', False)
            self.assertEqual(va.devices('cpu'), ())

    def test_unavailable_device_reports_probe_failure(self):
        with patch.object(_runtime, '_discovery', return_value=dict(devices=[], diagnostics=['loader absent'])):
            self.assertEqual(va.device_diagnostics(), ('loader absent',))
            with self.assertRaisesRegex(RuntimeError, 'loader absent'):
                _runtime._resolve('gpu', False)

    def test_invalid_options_fail_before_discovery(self):
        for device in ('coreml', 'gpu:-1', 'npu:01', 'gpu:abc', '', 0):
            with self.subTest(device=device), self.assertRaises(ValueError):
                va.jit(lambda x: x, device=device)
        for timeout in (0, -1, float('nan'), float('inf'), 86401, True):
            with self.assertRaises(ValueError):
                va.jit(lambda x: x, device='npu', timeout=timeout)
        with self.assertRaises(ValueError):
            va.jit(lambda x: x, device='npu', backend='reference')
        with self.assertRaises(TypeError):
            va.jit(lambda x: x, device='gpu', strict='yes')

    def test_executable_cache_reuses_and_releases_on_eviction(self):
        instances = []
        class FakeExecutable:
            def __init__(self, lowered):
                self.lowered = lowered
                self.closed = False
                instances.append(self)
            def __call__(self, *args):
                return self.lowered.reference(*args)
            def close(self):
                self.closed = True
        with patch.object(va.Lowered, 'compile', lambda lowered, *a, **k: FakeExecutable(lowered)):
            f = va.jit(lambda x: x + 1, device='gpu', cache_size=1)
            x = np.ones(2, np.float16)
            f(x); f(x * 2)
            self.assertEqual(len(instances), 1)
            f(np.ones(3, np.float32))
            self.assertTrue(instances[0].closed)
            self.assertFalse(instances[1].closed)
            f.clear_cache()
            self.assertTrue(instances[1].closed)
            f(x)
            self.assertEqual(len(instances), 3)
            f.clear_cache()

    @unittest.skipUnless(importlib.util.find_spec('virtio_accel._native'), 'native extension not built')
    def test_native_discovery_and_unavailable_index(self):
        for device in va.devices():
            self.assertIn(device.provider, ('coreml', 'vulkan'))
            self.assertIn(device.kind, ('npu', 'gpu'))
        lowered = va.jit(lambda x: x + 1).lower(np.ones(2, np.float32))
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            lowered.compile('gpu:999999')


@unittest.skipUnless(os.environ.get('VIRTIO_ACCEL_PYTHON_TEST_DEVICE'), 'set VIRTIO_ACCEL_PYTHON_TEST_DEVICE for hardware tests')
class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ['VIRTIO_ACCEL_PYTHON_TEST_DEVICE']
        # Requested hardware is mandatory: discovery/admission failures must fail this suite.
        self.record = _runtime._resolve(self.device, False)

    def test_projection_both_precisions_and_repeated_calls(self):
        rng = np.random.default_rng(2)
        for dtype in (np.float16, np.float32):
            x = rng.normal(size=(4, 8)).astype(dtype)
            w = rng.normal(size=(8, 3)).astype(dtype)
            b = rng.normal(size=3).astype(dtype)
            f = va.jit(lambda x, w, b: vnp.maximum(x @ w + b, 0), device=self.device)
            try:
                for scale in (1, 2, .5):
                    args = (x * dtype(scale), w, b)
                    actual = f(*args)
                    expected = f.lower(*args).reference(*args)
                    self.assertEqual(actual.dtype, np.dtype(dtype))
                    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-3 if dtype == np.float16 else 2e-5, atol=2e-3 if dtype == np.float16 else 2e-5)
            finally:
                f.clear_cache()

    def test_tuple_noncontiguous_inputs_and_threaded_reuse(self):
        x = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::-1]
        lowered = va.jit(lambda x: (x + 1, x * 2)).lower(x)
        with lowered.compile(self.device) as executable:
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(executable, [x, x + 1, x + 2]))
            for index, (a, b) in enumerate(results):
                np.testing.assert_allclose(a.to_numpy(), x + index + 1)
                np.testing.assert_allclose(b.to_numpy(), (x + index) * 2)
            with self.assertRaises(ValueError):
                executable(np.ones(3, np.float32))
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            executable(x)
        executable.close()

    def test_coreml_unsupported_cast_is_explicit_and_recoverable(self):
        if self.record.provider != 'coreml':
            self.skipTest('Core ML-specific admission boundary')
        x = np.ones((2, 2), np.float16)
        lowered = va.jit(lambda x: x.astype(vnp.float32)).lower(x)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'CAST'):
                lowered.compile(self.device)
        with self.assertRaisesRegex(RuntimeError, 'exclusive'):
            va.jit(lambda x: x).lower(x).compile(self.device, strict=True)
        with va.jit(lambda x: x + 1).lower(x).compile(self.device) as executable:
            np.testing.assert_allclose(executable(x).to_numpy(), x + 1)

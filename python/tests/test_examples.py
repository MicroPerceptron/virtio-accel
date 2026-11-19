import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


class ExampleTests(unittest.TestCase):
    def test_examples_against_independent_oracles(self):
        native = importlib.util.find_spec('virtio_accel._native') is not None
        for name in ('mlp_classifier', 'nearest_centroid', 'fir_filter'):
            spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'examples' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for dtype in ('float16', 'float32'):
                with self.subTest(example=name, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / 'graph.tosa' if native else None
                    module.demo(dtype=dtype, export=output)
                    if native:
                        self.assertEqual(output.read_bytes()[4:8], b'TOSA')

    @unittest.skipUnless(os.environ.get('VIRTIO_ACCEL_PYTHON_TEST_DEVICE'), 'hardware examples are opt-in')
    def test_native_examples(self):
        from hardware_support import requested_device
        device = requested_device()
        record = device
        provider = record.provider
        if provider in ('hexagon', 'xdna') or (provider == 'openvino' and record.kind == 'npu'):
            self.skipTest('provider-specific numerical tiers are exercised in test_runtime')
        for name in ('mlp_classifier', 'nearest_centroid', 'fir_filter'):
            spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'examples' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for dtype in ('float16', 'float32'):
                with self.subTest(example=name, dtype=dtype):
                    if provider in ('coreml', 'openvino', 'vulkan') and name == 'nearest_centroid' and dtype == 'float16':
                        with self.assertRaisesRegex(RuntimeError, 'CAST'):
                            module.demo(dtype=dtype, device=device)
                    else:
                        module.demo(dtype=dtype, device=device)

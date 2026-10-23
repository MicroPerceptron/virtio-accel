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
        from virtio_accel._runtime import _resolve
        device = os.environ['VIRTIO_ACCEL_PYTHON_TEST_DEVICE']
        provider = _resolve(device, False).provider
        for name in ('mlp_classifier', 'nearest_centroid', 'fir_filter'):
            spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'examples' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for dtype in ('float16', 'float32'):
                with self.subTest(example=name, dtype=dtype):
                    if provider == 'coreml' and name == 'nearest_centroid' and dtype == 'float16':
                        with self.assertRaisesRegex(RuntimeError, 'CAST'):
                            module.demo(dtype=dtype, device=device)
                    else:
                        module.demo(dtype=dtype, device=device)

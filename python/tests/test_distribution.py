"""Installed-wheel portability checks; vendor SDKs are not required to import."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from virtio_accel import _runtime


class RuntimeDiscoveryTests(unittest.TestCase):
    def test_openvino_package_library_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / 'libs').mkdir()
            library = package / 'libs' / 'libopenvino_c.so.2630'
            library.touch()
            spec = SimpleNamespace(submodule_search_locations=[str(package)])
            with patch.object(_runtime.os, 'environ', {}), patch.object(
                _runtime.importlib.util, 'find_spec', return_value=spec
            ):
                self.assertEqual(_runtime._openvino_library(), str(library.resolve()))
                with patch.object(_runtime.os, 'environ', {'VIRTIO_ACCEL_OPENVINO_LIBRARY': '/explicit/runtime'}):
                    self.assertIsNone(_runtime._openvino_library())
                (package / 'libs' / 'libopenvino_c.so.9999').touch()
                self.assertIsNone(_runtime._openvino_library())

    @unittest.skipUnless(importlib.util.find_spec('virtio_accel._native'), 'native extension not built')
    def test_missing_vendor_runtimes_leave_sdk_importable(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'does-not-exist')
            environment = dict(os.environ, VIRTIO_ACCEL_OPENVINO_LIBRARY=missing,
                               VIRTIO_ACCEL_HRX_LIBRARY=missing, VIRTIO_ACCEL_QNN_LIBRARY=missing)
            result = subprocess.run([
                sys.executable, '-c',
                "import json, numpy as np, virtio_accel as va; "
                "print(json.dumps({'providers':[d.provider for d in va.devices()], "
                "'diagnostics':va.device_diagnostics(), "
                "'tosa':va.jit(lambda x:x+1).lower(np.ones(2,np.float32)).to_tosa()[4:8].decode()}))",
            ], env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse({'openvino', 'xdna', 'hexagon'} & set(report['providers']))
            self.assertEqual(report['tosa'], 'TOSA')
            self.assertTrue(report['diagnostics'])

"""Regression checks for the generated public-API coverage gate."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check


class PythonApiCoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.book = Path(self.directory.name)
        (self.book / 'python/api').mkdir(parents=True)

    def validate(self, value, html):
        module = SimpleNamespace(__all__=['exported'], exported=value)
        for page in ('runtime', 'numpy', 'quantized'):
            (self.book / 'python/api' / (page + '.html')).write_text(html)
        with patch.object(check, 'BOOK', self.book), patch.object(
                check.importlib, 'import_module', return_value=module):
            return check.check_python_api()

    def test_new_export_must_have_generated_anchor(self):
        def exported():
            """A documented API."""
        problems = self.validate(exported, '')
        self.assertEqual(len(problems), 3)
        self.assertTrue(all('missing public export' in item for item in problems))

    def test_generated_signature_without_prose_is_rejected(self):
        def exported():
            pass
        problems = self.validate(exported, '<a id="virtio_accel.exported"></a>'
                                 '<a id="virtio_accel.numpy.exported"></a>'
                                 '<a id="virtio_accel.quantized.exported"></a>')
        self.assertEqual(len(problems), 3)
        self.assertTrue(all('missing docstring' in item for item in problems))

    def test_public_methods_and_properties_must_be_documented(self):
        class Exported:
            """Public class."""
            __module__ = 'virtio_accel'

            def method(self):
                """Public method."""

            @property
            def value(self):
                """Public property."""
                return 1
        html = '<a id="virtio_accel.exported"></a><a id="virtio_accel.numpy.exported"></a><a id="virtio_accel.quantized.exported"></a>'
        problems = self.validate(Exported, html)
        self.assertEqual(len(problems), 6)
        self.assertTrue(all('missing member' in item for item in problems))
        for module in ('virtio_accel', 'virtio_accel.numpy', 'virtio_accel.quantized'):
            for member in ('method', 'value'):
                html += f'<a id="{module}.exported.{member}"></a>'
        self.assertEqual(self.validate(Exported, html), [])


if __name__ == '__main__':
    unittest.main()

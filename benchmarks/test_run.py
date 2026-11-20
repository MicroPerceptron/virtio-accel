"""Guard against recording successful runs that did not measure any hardware."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run


class RunnerTests(unittest.TestCase):
    def execute(self, child_exit, emit_case, console=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"

            class Child:
                stdout = io.StringIO("╰─ framework output\n")

                def wait(self):
                    if emit_case:
                        (output / "cases.jsonl").write_text('{"case":"example"}\n')
                    return child_exit

            args = ["run.py", "--backend", "openvino", "--device", "NPU", "--output", str(output)]
            with patch.object(sys, "argv", args), patch.object(run, "capture", return_value="test"), patch.object(run.platform, "platform", return_value="test-os"), patch.object(run.subprocess, "Popen", return_value=Child()), contextlib.redirect_stdout(console or io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = run.main()
            meta = json.loads((output / "run.json").read_text())
            self.assertEqual(code, meta["exit_code"])
            self.assertEqual((output / "report.txt").read_text(encoding="utf-8"), "╰─ framework output\n")
            return code

    def test_empty_hardware_filter_is_failure(self):
        self.assertEqual(self.execute(0, False), 2)

    def test_native_failure_is_preserved_even_with_metadata(self):
        self.assertEqual(self.execute(101, True), 101)

    def test_success_requires_executed_case(self):
        self.assertEqual(self.execute(0, True), 0)

    def test_redirected_windows_console_preserves_unicode_report(self):
        with io.TextIOWrapper(io.BytesIO(), encoding="cp1252") as console:
            self.assertEqual(self.execute(0, True, console), 0)
            console.flush()
            self.assertIn("╰─ framework output", console.buffer.getvalue().decode("utf-8"))

    def test_device_is_required_before_building(self):
        with patch.object(sys, "argv", ["run.py", "--backend", "openvino"]), patch.object(run.subprocess, "Popen") as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run.main()
            self.assertEqual(error.exception.code, 2)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()

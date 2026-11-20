"""Exercise launcher process/socket ownership without requiring an accelerator."""

import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "launch.py"
STUB = """import os, socket, sys, time
from pathlib import Path
if '-device' in sys.argv and sys.argv[sys.argv.index('-device') + 1].endswith(',help'):
    sys.exit(0)
with open(os.environ['VA_LAUNCH_PIDS'], 'a') as f:
    f.write(str(os.getpid()) + '\\n')
if '--socket' in sys.argv:
    server = socket.socket(socket.AF_UNIX)
    server.bind(sys.argv[sys.argv.index('--socket') + 1])
    server.listen(1)
    connection, _ = server.accept()
    connection.recv(1)
else:
    address = sys.argv[sys.argv.index('-chardev') + 1].split('path=', 1)[1]
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(address)
    Path(os.environ['VA_LAUNCH_READY']).touch()
    if '--stay' in sys.argv:
        time.sleep(3600)
"""


class LauncherTests(unittest.TestCase):
    def test_exit_and_termination_reap_children_and_remove_private_socket(self):
        for terminate in (False, True):
            with (
                self.subTest(terminate=terminate),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                stub = root / "stub"
                stub.write_text(f"#!{sys.executable}\n" + STUB)
                stub.chmod(0o755)
                pids, ready = root / "pids", root / "ready"
                command = [
                    sys.executable,
                    str(LAUNCHER),
                    "--qemu",
                    str(stub),
                    "--daemon",
                    str(stub),
                    "--kernel",
                    str(stub),
                    "--backend",
                    "mock",
                ]
                if terminate:
                    command += ["--", "--stay"]
                with (root / "log").open("w+") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=log,
                        env=dict(
                            os.environ,
                            VA_LAUNCH_PIDS=str(pids),
                            VA_LAUNCH_READY=str(ready),
                        ),
                    )
                    try:
                        if terminate:
                            deadline = time.monotonic() + 10
                            while not ready.exists() and process.poll() is None:
                                self.assertLess(time.monotonic(), deadline)
                                time.sleep(0.01)
                            process.send_signal(signal.SIGTERM)
                        result = process.wait(timeout=15)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                    log.seek(0)
                    output = log.read()
                self.assertEqual(result, 130 if terminate else 0, output)
                children = [int(pid) for pid in pids.read_text().splitlines()]
                self.assertEqual(len(children), 2)
                for pid in children:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                host = shlex.split(output.splitlines()[0])
                self.assertFalse(Path(host[host.index("--socket") + 1]).parent.exists())


if __name__ == "__main__":
    unittest.main()

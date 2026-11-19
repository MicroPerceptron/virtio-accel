"""Exercise publication blockers without requiring registry access or hardware."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import check_release as release


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.version = release.release_version()

    def wheel(self, platform='manylinux_2_28_x86_64', *, abi='abi3', extra=None):
        path = self.root / f'virtio_accel-{self.version}-cp310-{abi}-{platform}.whl'
        info = f'virtio_accel-{self.version}.dist-info'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(info + '/METADATA', f'Name: virtio-accel\nVersion: {self.version}\n')
            archive.writestr(info + '/licenses/LICENSE-MIT', 'test license')
            archive.writestr(info + '/licenses/LICENSE-APACHE', 'test license')
            archive.writestr('virtio_accel/_native.abi3.so', b'test extension')
            if extra:
                archive.writestr(extra, b'unexpected payload')
        return path

    def test_accepts_platform_wheel_but_rejects_incomplete_release(self):
        self.wheel()
        with contextlib.redirect_stdout(io.StringIO()):
            release.check_artifacts(self.root, False)
        with self.assertRaisesRegex(ValueError, 'Incomplete release'):
            release.check_artifacts(self.root, True)

    def test_complete_release_requires_five_hosted_wheels_and_source(self):
        platforms = ('manylinux_2_28_x86_64', 'manylinux_2_28_aarch64',
                     'macosx_14_0_arm64', 'macosx_14_0_x86_64', 'win_amd64')
        wheels = [self.wheel(platform) for platform in platforms]
        archive = self.root / f'virtio_accel-{self.version}.tar.gz'
        archive.touch()
        # Source contents are independently checked by the real sdist rebuild in CI.
        with patch.object(release, 'check_sdist'), contextlib.redirect_stdout(io.StringIO()):
            release.check_artifacts(self.root, True)
            for wheel in wheels:
                held = wheel.with_suffix('.held')
                wheel.rename(held)
                with self.subTest(wheel=wheel.name), self.assertRaisesRegex(ValueError, 'missing platforms'):
                    release.check_artifacts(self.root, True)
                held.rename(wheel)
            archive.unlink()
            with self.assertRaisesRegex(ValueError, 'source archives=0'):
                release.check_artifacts(self.root, True)

    def test_manual_arm64_wheel_is_valid_but_not_part_of_automated_release(self):
        wheel = self.wheel('win_arm64')
        self.assertEqual(release.check_wheel(wheel, self.version), {'win_arm64'})
        with self.assertRaisesRegex(ValueError, "unexpected platforms \\['win_arm64'\\]"):
            release.check_artifacts(self.root, True)

    def test_rejects_nonportable_tag_and_wrong_abi(self):
        for options in ({'platform': 'linux_x86_64'}, {'abi': 'cp310'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                release.check_wheel(self.wheel(**options), self.version)

    def test_rejects_vendor_redistribution(self):
        for name in ('virtio_accel/QnnHtp.dll', 'virtio_accel/libopenvino_c.so.2630',
                     'virtio_accel/libhrx.so.0.1.0'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'vendor payload'):
                release.check_wheel(self.wheel(extra=name), self.version)

    def test_production_requires_matching_python_tag(self):
        for ref, success in (('refs/heads/main', False), ('refs/tags/v0.4.1', False),
                             ('refs/tags/python-v' + self.version, True)):
            result = subprocess.run([sys.executable, str(Path(release.__file__)), 'ref',
                                     '--destination', 'pypi', '--ref', ref], capture_output=True)
            self.assertEqual(result.returncode == 0, success, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()

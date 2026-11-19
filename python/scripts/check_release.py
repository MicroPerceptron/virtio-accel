"""Validate Python release metadata, complete artifacts, and installed adapters.

No registry credentials or upload code belong in this script. Publication is a
separate GitHub Actions job using PyPI Trusted Publishing.
"""
import argparse
from email.parser import BytesParser
from importlib.metadata import version as installed_version
import json
from pathlib import Path
import re
import sys
import sysconfig
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = {
    'manylinux_2_28_x86_64', 'manylinux_2_28_aarch64',
    'macosx_14_0_arm64', 'macosx_14_0_x86_64', 'win_amd64', 'win_arm64',
}


def release_version():
    # Also works on Python 3.10, which is part of the wheel compatibility matrix.
    python = re.search(r'^version = "([^"]+)"$', (ROOT / 'pyproject.toml').read_text(), re.M).group(1)
    rust = re.search(r'^version = "([^"]+)"$', (ROOT / 'python/Cargo.toml').read_text(), re.M).group(1)
    normalized = re.sub(r'-(alpha|beta|rc)\.(\d+)$', lambda m: {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}[m[1]] + m[2], rust)
    if normalized != python:
        raise ValueError(f'Python/Cargo versions disagree: {python} vs {rust}')
    return python


def check_wheel(path, expected):
    parts = path.stem.split('-')
    if len(parts) != 5 or parts[:4] != ['virtio_accel', expected, 'cp310', 'abi3']:
        raise ValueError(f'Unexpected wheel name/ABI: {path.name}')
    platforms = set(parts[4].split('.'))
    if not platforms or not platforms <= PLATFORMS:
        raise ValueError(f'Unexpected platform tag: {path.name}')
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            raise ValueError(f'Expected one METADATA file in {path.name}')
        metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
        if metadata['Name'].replace('_', '-') != 'virtio-accel' or metadata['Version'] != expected:
            raise ValueError(f'Incorrect wheel metadata in {path.name}')
        for license_name in ('LICENSE-MIT', 'LICENSE-APACHE'):
            if not any(name.endswith('/' + license_name) for name in names):
                raise ValueError(f'Missing {license_name} in {path.name}')
        if not any(name.startswith('virtio_accel/_native.') and name.endswith(('.so', '.pyd')) for name in names):
            raise ValueError(f'Missing native extension in {path.name}')
        # SDK files stay on build/serving hosts; only our compiled adapter ships.
        for name in names:
            base = Path(name).name.lower()
            if base.startswith(('qnn', 'libhrx', 'libopenvino', 'openvino_c')) and any(
                suffix in base for suffix in ('.dll', '.lib', '.so', '.dylib', '.h')
            ):
                raise ValueError(f'Unexpected vendor payload in {path.name}: {name}')
    return platforms


def check_sdist(path, expected):
    prefix = f'virtio_accel-{expected}/'
    if path.name != prefix[:-1] + '.tar.gz':
        raise ValueError(f'Unexpected source archive: {path.name}')
    with tarfile.open(path) as archive:
        names = set(archive.getnames())
        required = {'pyproject.toml', 'python/Cargo.toml', 'python/Cargo.lock',
                    'python/src/lib.rs', 'python/virtio_accel/__init__.py', 'src/lib.rs'}
        for backend in ('core', 'tosa', 'tosa-build', 'transport', 'coreml', 'vulkan', 'openvino', 'hexagon', 'xdna'):
            required.add(f'crates/virtio-accel-{backend}/Cargo.toml')
            required.add(f'crates/virtio-accel-{backend}/src/lib.rs')
        missing = {prefix + name for name in required} - names
        if missing:
            raise ValueError(f'Incomplete source archive: {sorted(missing)}')


def check_artifacts(directory, complete):
    expected = release_version()
    platforms = set()
    archives = []
    wheels = list(directory.glob('*.whl'))
    for path in wheels:
        found = check_wheel(path, expected)
        if platforms & found:
            raise ValueError(f'Duplicate wheel platform: {path.name}')
        platforms.update(found)
    for path in directory.glob('*.tar.gz'):
        check_sdist(path, expected)
        archives.append(path)
    if not wheels and not archives:
        raise ValueError('No distribution artifacts found')
    if complete and (platforms != PLATFORMS or len(archives) != 1):
        raise ValueError(f'Incomplete release: missing platforms {sorted(PLATFORMS - platforms)}, source archives={len(archives)}')
    print(json.dumps({'version': expected, 'platforms': sorted(platforms), 'source_archives': len(archives)}))


def check_installed():
    import numpy as np
    import virtio_accel as va
    from virtio_accel import _native

    platform = sysconfig.get_platform()
    expected = {'vulkan', 'openvino'}
    if sys.platform == 'darwin':
        expected.add('coreml')
    elif sys.platform.startswith('linux'):
        expected.add('xdna')
    elif platform == 'win-arm64':
        expected.add('hexagon')
    built = set(_native.compiled_backends())
    if built != expected:
        raise ValueError(f'Wrong compiled adapters for {platform}: expected {sorted(expected)}, got {sorted(built)}')
    actual = installed_version('virtio-accel')
    if actual != release_version():
        raise ValueError(f'Installed version mismatch: {actual}')
    package_root = Path(sysconfig.get_path('platlib')).resolve()
    if not Path(_native.__file__).resolve().is_relative_to(package_root):
        raise ValueError(f'Test is not using the installed wheel: {_native.__file__}')
    assert va.jit(lambda x: x + 1).lower(np.ones(2, np.float32)).to_tosa()[4:8] == b'TOSA'
    print(json.dumps({'version': actual, 'platform': platform, 'compiled_backends': sorted(built)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    artifacts = sub.add_parser('artifacts')
    artifacts.add_argument('directory', type=Path)
    artifacts.add_argument('--complete', action='store_true')
    sub.add_parser('installed')
    ref = sub.add_parser('ref')
    ref.add_argument('--destination', choices=('artifacts', 'testpypi', 'pypi'), required=True)
    ref.add_argument('--ref', required=True)
    args = parser.parse_args()
    if args.command == 'artifacts':
        check_artifacts(args.directory, args.complete)
    elif args.command == 'installed':
        check_installed()
    else:
        version = release_version()
        expected_ref = 'refs/tags/python-v' + version
        if args.ref.startswith('refs/tags/') and args.ref != expected_ref:
            raise ValueError(f'Tag must match package version: {expected_ref}')
        if args.destination == 'pypi' and args.ref != expected_ref:
            raise ValueError(f'Production publishing requires {expected_ref}')
        print(version)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build a disposable Linux test initramfs from this Python environment.

Requires an installed guest SDK wheel, numpy, ml-dtypes, busybox (static), ldd,
cpio and gzip. Copies only explicitly selected userspace, never host credentials.
Use the same architecture and compatible libc as the kernel/QEMU guest.
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def build(args, root):
    def copy(source, destination=None):
        source = Path(source)
        target = root / str(destination or source).lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)

    copy(shutil.which("busybox"), "/bin/busybox")
    copy(sys.executable, "/usr/bin/python3")
    stdlib = Path(sysconfig.get_path("stdlib"))
    libdest = (
        root / "usr/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    shutil.copytree(
        stdlib,
        libdest,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "site-packages",
            "dist-packages",
            "test",
            "tests",
            "idlelib",
            "tkinter",
        ),
    )
    sources = [Path(sys.executable).resolve(), *stdlib.rglob("*.so")]
    for name in ("virtio_accel", "numpy", "ml_dtypes"):
        spec = importlib.util.find_spec(name)
        source = Path(next(iter(spec.submodule_search_locations)))
        shutil.copytree(
            source,
            root / "opt/site" / name,
            ignore=shutil.ignore_patterns("__pycache__", "tests"),
        )
        sources.extend(source.rglob("*.so"))
        extra = source.parent / f"{name}.libs"
        if extra.exists():
            shutil.copytree(extra, root / "opt/site" / extra.name)
            sources.extend(extra.rglob("*.so*"))
    # ldd recursively resolves DT_NEEDED; preserve absolute loader/library paths.
    for source in sources:
        output = subprocess.run(
            ["ldd", str(source)], check=False, text=True, capture_output=True
        ).stdout
        for library in re.findall(r"(?:=>\s+)?(/[^\s()]+)", output):
            if Path(library).is_file():
                copy(library)
    copy(args.module.resolve(), "/virtio_accel.ko")
    copy(Path(__file__).with_name("in_guest.py"), "/test.py")
    for directory in ("dev", "proc", "sys", "tmp"):
        (root / directory).mkdir(exist_ok=True)
    (root / "bin/sh").symlink_to("busybox")
    init = root / "init"
    init.write_text(
        '''#!/bin/sh
export PATH=/bin:/usr/bin
/bin/busybox --install -s /bin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
export PYTHONHOME=/usr PYTHONPATH=/opt/site OPENBLAS_NUM_THREADS=1
export VA_TEST_BACKEND="'''
        + args.backend
        + """"
if insmod /virtio_accel.ko; then
    python3 /test.py
    result=$?
else
    result=1
fi
if [ "$result" != 0 ]; then echo VIRTIO_ACCEL_GUEST_FAIL; fi
dmesg | tail -30
echo VIRTIO_ACCEL_GUEST_DONE
while :; do sleep 3600; done
"""
    )
    init.chmod(0o755)
    # Filenames originate from trusted build inputs; NUL-delimit them for cpio.
    names = b"".join(
        os.fsencode(str(p.relative_to(root))) + b"\0" for p in sorted(root.rglob("*"))
    )
    with args.output.open("wb") as archive:
        pack = subprocess.Popen(
            ["cpio", "--null", "-o", "--format=newc", "--owner=0:0"],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        compress = subprocess.Popen(["gzip", "-1"], stdin=pack.stdout, stdout=archive)
        pack.stdout.close()
        pack.communicate(names)
        if pack.returncode or compress.wait():
            raise RuntimeError("initramfs creation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["mock", "openvino", "vulkan", "xdna"], required=True
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="va-initramfs-") as directory:
        build(args, Path(directory))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Boot Linux and run the installed Python SDK through the real guest driver."""

import argparse
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qemu", required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--initramfs", type=Path, required=True)
    parser.add_argument("--daemon", required=True)
    parser.add_argument(
        "--backend", choices=["mock", "openvino", "vulkan", "xdna"], required=True
    )
    parser.add_argument("--device")
    parser.add_argument("--accel", choices=["kvm", "tcg"], default="tcg")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="va-linux-") as directory:
        root = Path(directory)
        command = [
            args.daemon,
            "--socket",
            str(root / "vhost.sock"),
            "--backend",
            args.backend,
        ]
        if args.backend != "mock":
            command += ["--program-budget", str((1 << 64) - 1)]
        if args.device:
            command += ["--device", args.device]
        with (root / "daemon.log").open("w+") as hostlog:
            daemon = subprocess.Popen(command, stdout=hostlog, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while not (root / "vhost.sock").exists():
                    if daemon.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("host daemon did not start")
                    time.sleep(0.01)
                command = [
                    args.qemu,
                    "-machine",
                    "microvm,acpi=off,auto-kernel-cmdline=on,memory-backend=ram",
                    "-accel",
                    args.accel,
                    "-cpu",
                    "host,kvm-asyncpf=off,kvm-asyncpf-int=off"
                    if args.accel == "kvm"
                    else "max",
                    "-m",
                    "1536M",
                    "-object",
                    "memory-backend-memfd,id=ram,size=1536M,share=on",
                    "-nodefaults",
                    "-display",
                    "none",
                    "-serial",
                    "stdio",
                    "-no-reboot",
                    "-kernel",
                    str(args.kernel),
                    "-initrd",
                    str(args.initramfs),
                    "-append",
                    "console=ttyS0 earlyprintk=serial rdinit=/init panic=0 oops=panic",
                    "-global",
                    "virtio-mmio.force-legacy=false",
                    "-chardev",
                    f"socket,id=accel,path={root / 'vhost.sock'}",
                    "-device",
                    (
                        "vhost-user-test-device,bus=virtio-mmio-bus.0,chardev=accel,"
                        "virtio-id=65535,num_vqs=1,vq_size=1024,config_size=16"
                    ),
                ]
                with args.log.open("w") as log:
                    guest = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                    deadline = time.monotonic() + 240
                    paused = False
                    resume = None
                    finished = False
                    try:
                        while guest.poll() is None:
                            now = time.monotonic()
                            output = args.log.read_text()
                            if "VIRTIO_ACCEL_GUEST_DONE" in output:
                                finished = True
                                guest.terminate()
                                guest.wait(timeout=10)
                                break
                            if now > deadline:
                                raise TimeoutError(f"guest stalled; see {args.log}")
                            if not paused and "VIRTIO_ACCEL_PAUSE_HOST" in output:
                                daemon.send_signal(signal.SIGSTOP)
                                paused, resume = True, now + 2
                            if resume is not None and now >= resume:
                                daemon.send_signal(signal.SIGCONT)
                                resume = None
                            time.sleep(0.01)
                    finally:
                        if resume is not None:
                            daemon.send_signal(signal.SIGCONT)
                        if guest.poll() is None:
                            guest.kill()
                        guest.wait()
                text = args.log.read_text()
                print(text[-12000:])
                if (
                    not finished
                    or "VIRTIO_ACCEL_GUEST_PASS" not in text
                    or "VIRTIO_ACCEL_GUEST_FAIL" in text
                    or "Kernel panic" in text
                    or "BUG:" in text
                    or "WARNING: CPU:" in text
                ):
                    raise RuntimeError(f"guest failed; see {args.log}")
            finally:
                daemon.terminate()
                try:
                    daemon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait()
                hostlog.seek(0)
                print(hostlog.read()[-6000:])


if __name__ == "__main__":
    main()

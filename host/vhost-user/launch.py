#!/usr/bin/env python3
"""Launch a Linux VM with private vhost-user transport and explicit host policy."""

import argparse
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path


def executable(value):
    found = shutil.which(value)
    if not found:
        raise argparse.ArgumentTypeError(f"executable not found: {value}")
    return found


def readable(value):
    path = Path(value).resolve()
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return str(path)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qemu", type=executable, required=True)
    parser.add_argument("--daemon", type=executable, required=True)
    parser.add_argument(
        "--backend", choices=("mock", "openvino", "vulkan", "xdna"), required=True
    )
    parser.add_argument("--device")
    parser.add_argument("--allow-unknown-program-residency", action="store_true")
    parser.add_argument("--buffer-budget", type=int, default=512 << 20)
    parser.add_argument("--target", choices=("x86_64", "aarch64"), default="x86_64")
    parser.add_argument("--accel", choices=("tcg", "kvm"), default="tcg")
    parser.add_argument("--kernel", type=readable, required=True)
    parser.add_argument("--initrd", type=readable)
    parser.add_argument("--append", default="console=ttyS0")
    parser.add_argument("--memory-mib", type=int, default=1536)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--iommu",
        action="store_true",
        help="put the device behind the machine's IOMMU: virtio-pci, "
        "VIRTIO_F_ACCESS_PLATFORM required, the host maps only what the "
        "guest's domain grants (aarch64 virt with SMMUv3 only)",
    )
    parser.add_argument(
        "qemu_args", nargs=argparse.REMAINDER, help="additional QEMU options after --"
    )
    args = parser.parse_args()
    if args.memory_mib <= 0 or args.buffer_budget <= 0:
        parser.error("memory and buffer budget must be positive")
    if args.backend != "mock" and not args.allow_unknown_program_residency:
        parser.error(
            "native backends require --allow-unknown-program-residency (one resident program; not a physical-memory bound)"
        )
    if args.iommu and args.target != "aarch64":
        # Virtio MMIO has no route to an IOMMU in QEMU, and microvm has no
        # PCI. An x86 machine with one (q35, intel-iommu) has not been run.
        parser.error("--iommu is validated on --target aarch64 only")
    if args.accel == "kvm" and not args.dry_run:
        try:
            with open("/dev/kvm", "rb+"):
                pass
        except OSError as error:
            parser.error(f"KVM is unavailable: {error}; use --accel tcg")
    with tempfile.TemporaryDirectory(prefix="virtio-accel-") as directory:
        socket = Path(directory) / "accel.sock"
        if "," in str(socket):
            parser.error("TMPDIR cannot contain commas in QEMU socket options")
        host_command = [
            args.daemon,
            "--socket",
            str(socket),
            "--backend",
            args.backend,
            "--buffer-budget",
            str(args.buffer_budget),
        ]
        if args.device:
            host_command += ["--device", args.device]
        if args.allow_unknown_program_residency:
            host_command += ["--allow-unknown-program-residency"]
        identity = "chardev=accel,virtio-id=56,num_vqs=1,vq_size=1024,config_size=16"
        if args.iommu:
            # Both halves or neither: the daemon refuses to run unless the
            # frontend translates, and the frontend refuses a daemon that
            # cannot follow the translation.
            host_command += ["--iommu"]
            frontend = f"vhost-user-test-device-pci,{identity},iommu_platform=on"
        else:
            frontend = f"vhost-user-test-device,bus=virtio-mmio-bus.0,{identity}"
        machine = (
            "microvm,acpi=off,auto-kernel-cmdline=on"
            if args.target == "x86_64"
            else "virt,iommu=smmuv3"
            if args.iommu
            else "virt"
        )
        cpu = "max" if args.accel == "tcg" else "host"
        if args.target == "x86_64" and args.accel == "kvm":
            cpu += ",kvm-asyncpf=off,kvm-asyncpf-int=off"
        command = [
            args.qemu,
            "-machine",
            machine + ",memory-backend=ram",
            "-accel",
            args.accel,
            "-cpu",
            cpu,
            "-m",
            f"{args.memory_mib}M",
            "-object",
            f"memory-backend-memfd,id=ram,size={args.memory_mib}M,share=on",
            "-nodefaults",
            "-display",
            "none",
            "-serial",
            "stdio",
            "-no-reboot",
            "-kernel",
            args.kernel,
            "-append",
            args.append,
            "-global",
            "virtio-mmio.force-legacy=false",
            "-chardev",
            f"socket,id=accel,path={socket}",
            "-device",
            frontend,
        ]
        if args.initrd:
            command += ["-initrd", args.initrd]
        extra = args.qemu_args
        command += extra[1:] if extra[:1] == ["--"] else extra
        print(shlex.join(host_command), flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        # Detect a missing frontend before starting the host process. The caller
        # must still supply the reviewed patched build, not an upstream binary.
        subprocess.run(
            [args.qemu, "-device", frontend.split(",")[0] + ",help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        host = guest = None
        try:
            host = subprocess.Popen(host_command)
            deadline = time.monotonic() + 30
            while not socket.exists():
                if host.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        "host did not become ready; see its diagnostics above"
                    )
                time.sleep(0.01)
            if not stat.S_ISSOCK(socket.stat().st_mode):
                raise RuntimeError("host did not create a Unix socket")
            guest = subprocess.Popen(command)
            result = guest.wait()
            host_result = host.wait(timeout=10)
            return result or host_result
        finally:
            # QEMU must stop referencing shared memory before host teardown.
            stop(guest)
            stop(host)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None

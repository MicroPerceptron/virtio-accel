#!/usr/bin/env python3
"""Real QEMU MMIO/vhost-user integration, driven by qtest without a guest OS.

Requires qemu-system-aarch64 with vhost-user-test-device. No KVM, root, guest
image, third-party Python packages, or hardware accelerator is needed for mock.
"""

import argparse
import json
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path


def connect(path, processes):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"process exited early: {process.args}")
        sock = socket.socket(socket.AF_UNIX)
        sock.settimeout(10)
        try:
            sock.connect(str(path))
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            time.sleep(0.01)
    raise TimeoutError(path)


class QTest:
    def __init__(self, sock):
        self.stream = sock.makefile("rwb", buffering=0)

    def command(self, text):
        self.stream.write(text.encode() + b"\n")
        while True:
            line = self.stream.readline().decode().strip()
            if line.startswith("IRQ "):
                continue
            if not line.startswith("OK"):
                raise RuntimeError(f"{text}: {line}")
            return line.split()[1:]

    def read(self, address, count):
        return bytes.fromhex(self.command(f"read {address:#x} {count:#x}")[0][2:])

    def write(self, address, data):
        self.command(f"write {address:#x} {len(data):#x} 0x{data.hex()}")

    def readl(self, address):
        return int(self.command(f"readl {address:#x}")[0], 0)

    def writel(self, address, value):
        self.command(f"writel {address:#x} {value:#x}")


class Device:
    DESC, AVAIL, USED = 0x41000000, 0x41002000, 0x41003000
    REQUEST, RESPONSE = 0x41004000, 0x41008000
    SIZE = 8

    def __init__(self, qtest):
        self.q = qtest
        candidates = range(0x0A000000, 0x0A004000, 0x200)
        self.base = next(addr for addr in candidates if self.q.readl(addr + 8) == 56)
        assert self.q.readl(self.base) == 0x74726976
        assert self.q.readl(self.base + 4) == 2, "modern Virtio MMIO required"
        self.request_id = 0
        self.start()

    def start(self):
        q, base = self.q, self.base
        q.writel(base + 0x70, 0)
        assert q.readl(base + 0x70) == 0
        q.writel(base + 0x70, 3)
        q.writel(base + 0x14, 0)
        low = q.readl(base + 0x10)
        assert low & 0x1F == 0, "no device-specific extensions"
        assert low & ((1 << 28) | (1 << 29)) == 0, "no indirect/event-index"
        q.writel(base + 0x14, 1)
        assert q.readl(base + 0x10) & 1
        q.writel(base + 0x24, 0)
        q.writel(base + 0x20, 0)
        q.writel(base + 0x24, 1)
        q.writel(base + 0x20, 1)  # VERSION_1
        q.writel(base + 0x70, 11)
        assert q.readl(base + 0x70) == 11
        config = b"".join(
            struct.pack("<I", q.readl(base + 0x100 + i)) for i in range(0, 16, 4)
        )
        assert struct.unpack("<HHHHII", config) == (1, 0, 1, 256, 1 << 20, 1 << 20)
        for address in (self.DESC, self.AVAIL, self.USED):
            q.write(address, bytes(256))
        q.writel(base + 0x30, 0)
        assert q.readl(base + 0x34) >= self.SIZE
        q.writel(base + 0x38, self.SIZE)
        for register, address in (
            (0x80, self.DESC),
            (0x90, self.AVAIL),
            (0xA0, self.USED),
        ):
            q.writel(base + register, address)
            q.writel(base + register + 4, 0)
        q.writel(base + 0x44, 1)
        q.writel(base + 0x70, 15)
        assert q.readl(base + 0x70) == 15
        self.index = 0

    def transfer(self, request, response_size=256, malformed=None, segmented=False):
        q = self.q
        q.write(self.REQUEST, request)
        q.write(self.RESPONSE, bytes([0xA5]) * response_size)
        if segmented:
            descs = [
                (self.REQUEST, 7, 1, 1),
                (self.REQUEST + 7, len(request) - 7, 1, 2),
                (self.RESPONSE, 9, 3, 3),
                (self.RESPONSE + 9, response_size - 9, 2, 0),
            ]
        else:
            descs = [
                (self.REQUEST, len(request), 1, 1),
                (self.RESPONSE, response_size, 2, 0),
            ]
        if malformed == "loop":
            descs[0] = (self.REQUEST, len(request), 1, 0)
        elif malformed == "indirect":
            descs[0] = (self.REQUEST, len(request), 4, 0)
        elif malformed == "unmapped":
            descs[0] = (0xDEAD000000, len(request), 1, 1)
        elif malformed == "overlap":
            descs[1] = (self.REQUEST, response_size, 2, 0)
        elif malformed == "ring-alias":
            descs[1] = (self.USED, response_size, 2, 0)
        elif malformed == "direction":
            descs[0] = (self.REQUEST, len(request), 3, 1)
            descs[1] = (self.RESPONSE, response_size, 0, 0)
        q.write(self.DESC, b"".join(struct.pack("<QIHH", *d) for d in descs))
        q.write(self.AVAIL + 4 + (self.index % self.SIZE) * 2, struct.pack("<H", 0))
        expected = (self.index + 1) & 0xFFFF
        q.write(self.AVAIL + 2, struct.pack("<H", expected))
        q.writel(self.base + 0x50, 0)
        deadline = time.monotonic() + 10
        while struct.unpack("<H", q.read(self.USED + 2, 2))[0] != expected:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"used ring stalled on request {self.request_id}")
            time.sleep(0.001)
        head, length = struct.unpack(
            "<II", q.read(self.USED + 4 + (self.index % self.SIZE) * 8, 8)
        )
        assert head == 0 and length <= response_size
        self.index = expected
        # Used publication precedes eventfd interrupt delivery; allow that short interval.
        while not q.readl(self.base + 0x60) & 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("missing used-ring interrupt")
            time.sleep(0.001)
        q.writel(self.base + 0x64, 1)
        response = q.read(self.RESPONSE, response_size)
        assert response[length:] == bytes([0xA5]) * (response_size - length), (
            "wrote beyond used length"
        )
        return response[:length]

    def request(self, opcode, payload=b"", status=0, **kwargs):
        self.request_id += 1
        frame = struct.pack("<HHIQ", opcode, 0, len(payload), self.request_id) + payload
        response = self.transfer(frame, **kwargs)
        actual, flags, length, request_id = struct.unpack("<HHIQ", response[:16])
        assert (actual, flags, length, request_id) == (
            status,
            0,
            len(response) - 16,
            self.request_id,
        ), response.hex()
        return response[16:]


def execute_program(device, context_id, backend):
    def allocate(usage):
        response = device.request(
            0x200, struct.pack("<QQQB7xII", context_id, 4, 64, 1, usage, 0)
        )
        return struct.unpack("<Q", response)[0]

    output = allocate(9)  # TRANSFER_SOURCE | PROGRAM_OUTPUT
    if backend == "mock":
        artifact = (
            b"VAMK"
            + bytes([1, 0, 2, 1])
            + struct.pack("<II", 0, 0xFFFFFFFF)
            + bytes([42])
            + bytes(7)
        )
        target = [0x56414D4B, 0x10000, 0x54455354, 0x4F4E4C59] + [0] * 8
        format_id, resident = 0x56414D4B, 64
        bindings = [(output, 0, 4, 0, 2)]
        expected = bytes([42]) * 4
        inputs = []
    else:
        source = allocate(6)  # TRANSFER_DESTINATION | PROGRAM_INPUT
        expected = struct.pack("<f", 3.25)
        device.request(0x202, struct.pack("<QQQ", source, 0, 4) + expected)
        root = Path(__file__).resolve().parents[3]
        artifact = (
            root / "crates/virtio-accel-vulkan/tests/data/identity-fp32-v1.0.0.tosa"
        ).read_bytes()
        target = [0x544F5341, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0]
        format_id, resident = 0x544F5341, (1 << 64) - 1
        bindings = [(source, 0, 4, 0, 1), (output, 0, 4, 1, 2)]
        inputs = [source]
    (program,) = struct.unpack(
        "<Q",
        device.request(
            0x300,
            struct.pack(
                "<QII12IQQ", context_id, format_id, 0, *target, len(artifact), resident
            )
            + artifact,
            segmented=True,
        ),
    )
    (queue,) = struct.unpack(
        "<Q", device.request(0x400, struct.pack("<QII", context_id, 0, 0))
    )
    submission = struct.pack("<QQIIQ", queue, program, len(bindings), 0, 0)
    submission += b"".join(struct.pack("<QQQIB3x", *binding) for binding in bindings)
    event = device.request(0x500, submission, segmented=True)
    deadline = time.monotonic() + 30
    while True:
        state, error, reserved = struct.unpack("<HHI", device.request(0x501, event))
        assert not error and not reserved, (state, error, reserved)
        if state == 1:
            break
        assert state == 0 and time.monotonic() < deadline, state
        time.sleep(0.005)
    # Retirement releases bound buffers for readback under the protocol's ownership rules.
    device.request(0x503, event)
    assert device.request(0x203, struct.pack("<QQQ", output, 0, 4)) == expected
    device.request(0x401, struct.pack("<Q", queue))
    device.request(0x301, struct.pack("<Q", program))
    for buffer in [*inputs, output]:
        device.request(0x201, struct.pack("<Q", buffer))
    print(
        f"PASS: {backend} program load, submit, poll, retire and exact output {expected.hex()}",
        flush=True,
    )


def exercise(device, backend, pause_resume):
    info = device.request(1, segmented=True)
    assert len(info) == 76
    context = device.request(0x100, bytes(8))
    (context_id,) = struct.unpack("<Q", context)
    # Host-visible buffer, transfer source + destination usage.
    allocation = struct.pack("<QQQB7xII", context_id, 64, 64, 1, 3, 0)
    buffer = device.request(0x200, allocation)
    (buffer_id,) = struct.unpack("<Q", buffer)
    data = bytes(range(64))
    device.request(
        0x202, struct.pack("<QQQ", buffer_id, 0, len(data)) + data, segmented=True
    )
    assert (
        device.request(
            0x203, struct.pack("<QQQ", buffer_id, 0, len(data)), segmented=True
        )
        == data
    )
    pause_resume()
    assert device.request(0x203, struct.pack("<QQQ", buffer_id, 0, len(data))) == data
    execute_program(device, context_id, backend)
    device.request(0x203, struct.pack("<QQQ", buffer_id, 63, 2), status=4)
    device.request(0xFFFF, status=1)
    for bad in ("loop", "indirect", "unmapped", "overlap", "ring-alias", "direction"):
        assert device.transfer(struct.pack("<HHIQ", 1, 0, 0, 42), malformed=bad) == b""
        assert device.request(1) == info, f"queue did not recover after {bad}"
    # Insufficient response storage must not create a context behind the driver's back.
    assert (
        device.transfer(
            struct.pack("<HHIQ", 0x100, 0, 8, 43) + bytes(8), response_size=8
        )
        == b""
    )
    for _ in range(20):
        assert device.request(1) == info  # cross ring wrap repeatedly
    device.request(0x201, buffer)
    device.request(0x201, buffer, status=11)
    # Reset with a live context, then prove old IDs cannot reach the new object graph.
    device.start()
    device.request(0x101, context, status=11)
    replacement = device.request(0x100, bytes(8))
    assert replacement != context
    device.request(0x101, replacement)
    print(
        "PASS: QEMU negotiation, segmented requests/responses, buffer roundtrip, interrupts, malformed chains, ring wrap, reset and stale IDs",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daemon",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "target/debug/virtio-accel-vhost-user",
    )
    parser.add_argument("--qemu", default="qemu-system-aarch64")
    parser.add_argument(
        "--backend", choices=["mock", "openvino", "vulkan", "xdna"], default="mock"
    )
    parser.add_argument("--device", help="explicit OpenVINO endpoint, e.g. NPU or GPU")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="va-qemu-") as directory:
        root = Path(directory)
        processes = []
        logs = []
        try:
            for name in ("daemon", "qemu"):
                logs.append((root / f"{name}.log").open("w+"))
            daemon_args = [
                str(args.daemon.resolve()),
                "--socket",
                str(root / "vhost.sock"),
                "--backend",
                args.backend,
            ]
            if args.backend != "mock":
                # Native compiler residency has no finite declared upper bound.
                daemon_args += ["--program-budget", str((1 << 64) - 1)]
            if args.device:
                daemon_args += ["--device", args.device]
            daemon = subprocess.Popen(
                daemon_args, stdout=logs[0], stderr=subprocess.STDOUT
            )
            processes.append(daemon)
            deadline = time.monotonic() + 10
            while not (root / "vhost.sock").exists():
                if daemon.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("daemon did not start")
                time.sleep(0.01)
            qemu = subprocess.Popen(
                [
                    args.qemu,
                    "-machine",
                    "virt,memory-backend=ram",
                    "-cpu",
                    "cortex-a57",
                    "-accel",
                    "tcg",
                    "-m",
                    "64M",
                    "-object",
                    "memory-backend-memfd,id=ram,size=64M,share=on",
                    "-display",
                    "none",
                    "-nodefaults",
                    "-S",
                    "-global",
                    "virtio-mmio.force-legacy=false",
                    "-chardev",
                    f"socket,id=accel,path={root / 'vhost.sock'}",
                    "-device",
                    "vhost-user-test-device,bus=virtio-mmio-bus.0,chardev=accel,virtio-id=56,num_vqs=1,vq_size=1024,config_size=16",
                    "-qtest-log",
                    "/dev/null",
                    "-qtest",
                    f"unix:{root / 'qtest.sock'},server=on,wait=off",
                    "-qmp",
                    f"unix:{root / 'qmp.sock'},server=on,wait=off",
                ],
                stdout=logs[1],
                stderr=subprocess.STDOUT,
            )
            processes.append(qemu)
            with (
                connect(root / "qtest.sock", processes) as qs,
                connect(root / "qmp.sock", processes) as ms,
            ):
                qmp = ms.makefile("rwb", buffering=0)
                assert "QMP" in json.loads(qmp.readline())

                def monitor(command):
                    qmp.write(json.dumps({"execute": command}).encode() + b"\n")
                    while True:
                        response = json.loads(qmp.readline())
                        if "return" in response:
                            return
                        assert "error" not in response, response

                monitor("qmp_capabilities")
                monitor("cont")
                exercise(
                    Device(QTest(qs)),
                    args.backend,
                    lambda: (monitor("stop"), monitor("cont")),
                )
                logs[0].flush()
                logs[0].seek(0)
                print(logs[0].read(), end="", flush=True)
        except BaseException:
            for log in logs:
                log.flush()
                log.seek(0)
                print(log.read()[-12000:], flush=True)
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()

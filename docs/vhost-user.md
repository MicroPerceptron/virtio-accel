# Experimental vhost-user host and QEMU binding

The Linux host adapter in [`host/vhost-user`](../host/vhost-user/README.md) exposes the existing
protocol 1.0 command processor through real split virtqueues. QEMU owns guest-visible Virtio MMIO
registers and interrupts; a separate process owns the native accelerator backend.

```text
Guest Virtio MMIO driver (or the qtest test driver)
    -> QEMU vhost-user frontend
    -> shared split virtqueue + kick/call eventfds
    -> checked guest-memory byte ports
    -> CommandProcessor<Accelerator>
    -> selected native backend
```

The socket carries vhost-user control messages and file descriptors. Request/response payloads
travel through shared guest RAM. Accelerator buffers remain provider-owned: protocol WRITE_BUFFER
and READ_BUFFER perform explicit transfers. Shared guest RAM does not enable external-memory import.

## Status and identity

This is a development binding, **not an assigned Virtio device type or a production QEMU device**.
The test configuration uses experimental device ID **65535 over modern Virtio MMIO**. Do not use
that ID with the modern PCI device-ID arithmetic or distribute a PCI device impersonating an
assigned type. Official allocation and a production frontend remain tracked by
[issue #103](https://github.com/MicroPerceptron/virtio-accel/issues/103).

The current QEMU 10.2.1 generic test frontend needs two reviewed local patches, shipped in
[`host/vhost-user/qemu`](../host/vhost-user/qemu/):

1. Accept experimental ID 65535 in the diagnostic device-name lookup. Unpatched QEMU asserts.
2. Forward whole-device reset to the backend and wait for its acknowledgement. Without this,
   stopping the queues leaves the backend's old context and buffer handles alive.

The build script pins upstream commit `2d3df8abca265c9bcc9e438d691d561592060998` and applies
these patches. They have not been upstreamed. QEMU itself describes `vhost-user-test-device` as
[development-only](https://www.qemu.org/docs/master/system/devices/virtio/vhost-user.html).

## Build and test without a guest image

On Linux, with Rust 1.85+, Python 3, Git, a C compiler, Ninja, pkg-config, and GLib development
headers installed:

```sh
cargo build --manifest-path host/vhost-user/Cargo.toml --locked
host/vhost-user/qemu/build.sh /tmp/virtio-accel-qemu
python3 host/vhost-user/tests/qemu-smoke.py \
  --qemu /tmp/virtio-accel-qemu/build/qemu-system-aarch64
```

On Ubuntu the additional build packages are `ninja-build libglib2.0-dev python3-venv`.
`JOBS=6` can increase QEMU build concurrency. The script refuses another source revision and
accepts an already-applied patch without discarding other changes.

The test uses QEMU's ARM `virt` machine under TCG and drives its real MMIO registers and guest
RAM through qtest. It requires no KVM, elevated privileges, guest image, or accelerator hardware
for the default `mock` backend. It exercises:

- feature negotiation and the 16-byte configuration;
- segmented request/response chains and initialized used lengths;
- context/buffer lifecycle, explicit transfers, program load, submit, poll, event retirement,
  and exact output readback;
- interrupt delivery, descriptor-ring wrap, malformed chains and recovery;
- VM pause/resume retaining live resources;
- whole-device reset and rejection of stale handles.

Rust tests additionally cover invalid producer-index jumps, 16-bit index wrap, interrupt
suppression, memory-map replacement validation, address-space translation, alias rejection,
and namespace exhaustion. CI builds the pinned QEMU frontend and runs the mock test on Linux.
These tests validate the **host/QEMU binding**, not a Linux or Kore guest driver or boot sequence.

## Native backends

Build the optional Linux adapters explicitly:

```sh
cargo build --manifest-path host/vhost-user/Cargo.toml --locked \
  --features openvino,vulkan,xdna
```

Use `--backend openvino`, `--backend vulkan`, or `--backend xdna`. `--device NPU` or `--device GPU`
selects an OpenVINO endpoint explicitly; without it OpenVINO prefers NPU, then GPU, then CPU.
Vulkan prints its selected device. Backend initialization failure is an error, not a silent
substitution of the mock backend. Existing backend runtime-discovery environment variables apply,
including `VIRTIO_ACCEL_OPENVINO_LIBRARY` for the OpenVINO C runtime.

The same QEMU test can perform an exact FP32 identity computation on hardware:

```sh
python3 host/vhost-user/tests/qemu-smoke.py \
  --qemu /tmp/virtio-accel-qemu/build/qemu-system-aarch64 \
  --backend openvino --device NPU
python3 host/vhost-user/tests/qemu-smoke.py \
  --qemu /tmp/virtio-accel-qemu/build/qemu-system-aarch64 \
  --backend vulkan
```

The reference backend executes its deterministic test artifact on the first event poll. Its
existing default/manual-completion mode remains unchanged for lifecycle tests.

Core ML/macOS and Hexagon/Windows hosting are not implemented by this Linux adapter. Selecting
vhost-user does not make native SDKs or rust-vmm's Linux eventfd implementation cross-platform.

## Start a daemon

Use a private directory accessible only to the QEMU operator. The socket grants access to the
accelerator, and the trusted frontend supplies shared-memory mappings of guest RAM.

```sh
run_dir=$(mktemp -d)
host/vhost-user/target/debug/virtio-accel-vhost-user \
  --socket "$run_dir/accel.sock" --backend mock
```

Connect the patched QEMU frontend with `virtio-id=65535,num_vqs=1,vq_size=1024,config_size=16`,
modern Virtio MMIO (`-global virtio-mmio.force-legacy=false`), a `chardev socket`, and a shared
`memory-backend-memfd` object. The complete runnable invocation is in the smoke test. A real guest
still needs a driver implementing this device's command protocol.

The daemon serves one connection and exits on disconnect or an unrecoverable transport/backend
fault. It does not replace an existing socket, reconnect an old device, or remove the caller's
socket path. Remove the private directory after its processes have stopped.

### Resource policy

`--buffer-budget BYTES` defaults to 512 MiB of actual retained buffer backing.
`--program-budget BYTES` defaults to 256 MiB of declared retained program charge.
Both must be nonzero. These budgets are host policy and cannot be enlarged by guest commands.

Current native compilers declare `u64::MAX` program residency because they expose no finite upper
bound. To admit one such program, the operator must explicitly use:

```sh
--program-budget 18446744073709551615
```

The native-backend smoke test sets this option. It is **not a finite physical-memory guarantee**;
use process/cgroup limits when compiler memory or CPU consumption must be bounded. The charge
allows one maximal-residency program at a time. The adapter does not silently reduce the provider's
charge to fit a convenient budget.

## Binding and ownership rules

- Protocol major/minor: 1.0, unchanged command and response bytes. No device-specific feature bits.
- One command queue, direct split descriptors only; up to 1024 queue entries and 256 descriptors
  per chain. The 1024 queue bound matches QEMU's MMIO queue maximum. Request and response limits
  are each 1 MiB, as reported in `WireConfig`.
- Base `VIRTIO_F_VERSION_1` is separate from the backend-only vhost-user protocol-features bit.
  Only CONFIG, REPLY_ACK, and RESET_DEVICE protocol features are offered. No packed/indirect rings,
  event-index, logging, migration, external memory, or queue-reset feature is advertised.
- Vring configuration addresses are QEMU userspace addresses and are translated through the
  memory table. Descriptor payload addresses are guest physical addresses. Overflow, holes,
  malformed direction/order, loops, indirect descriptors, and payload/ring aliasing are rejected.
- Descriptor metadata is copied into bounded stack storage before execution. Guest bytes use
  `vm-memory` volatile access, never ordinary borrowed Rust slices. Payload ports cannot escape
  a command, and native backends never receive guest addresses.
- One calling thread owns all native handles. The lower-level `vhost` request handler is used
  instead of imposing `vhost-user-backend`'s `Send + Sync` requirements on every accelerator.
  Its uncontended mutex only bridges the control-handler API; no worker executes concurrently.
- Processing yields to control messages after at most one maximum queue of commands. Metadata
  traversal is bounded; alias checks are quadratic in the capped descriptor count. Byte-port
  access and event-loop polling allocate no per-command transport storage. Native calls may
  block; this is not a host watchdog or hard scheduling deadline.
- Whole reset stops queue access and retires notification descriptors before resetting the
  processor with a fresh namespace. Pause/stop alone preserves semantic resources. Reset failure
  or namespace exhaustion closes the connection rather than reusing an ambiguous object graph.
- Recoverable malformed chains complete with length zero. A response-write failure after possible
  mutation, invalid ring state, or backend loss ends the connection. The generic QEMU frontend
  does not yet provide complete guest-visible DEVICE_NEEDS_RESET propagation; a production
  frontend must provide that and a defined disconnect policy before deployment.
- The command processor isolates device epochs and contexts, but guest process identities are
  not on this wire. Future guest drivers/services must mediate which process owns each object.

## Change classification

This adds a host binding and an additive mock-backend constructor. It changes no accepted or
emitted protocol 1.0 bytes, assigned numbers, wire ownership rules, or negotiated extensions.
It is not a protocol minor/major version change and does not regenerate the frozen conformance
artifacts. Platform dependencies live in an unpublished separate workspace, following the Python
binding's separation from the portable publication graph.

## Initial validation

On 2026-11-19, the patched QEMU 10.2.1 path passed the lifecycle/reset suite on Gigatron with:

- the mock reference backend, including exact fill output;
- Vulkan on Intel Lunar Lake graphics, exact FP32 `3.25` identity readback;
- OpenVINO explicitly selecting the Intel NPU, the same exact FP32 readback.

XDNA is build-enabled but not yet hardware-validated through this QEMU path. These results do not
claim guest-driver coverage, migration, production deployment readiness, or macOS/Windows hosting.

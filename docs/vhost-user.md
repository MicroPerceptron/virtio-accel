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
The test configuration uses provisional device ID **56**, over modern Virtio MMIO or modern
Virtio PCI (`1af4:1078`). The number is not assigned: it is unallocated in the Virtio
specification today, clear of the IDs other VMMs use unofficially (61 to 63), and small enough for
the PCI device-ID arithmetic, which the earlier 65535 was not. It is one constant and will change
when an ID is allocated. Do not distribute a device that impersonates an assigned type. Official
allocation and a production frontend remain tracked by
[issue #103](https://github.com/MicroPerceptron/virtio-accel/issues/103).

The current QEMU 10.2.1 generic test frontend needs four local patches, shipped in
[`host/vhost-user/qemu`](../host/vhost-user/qemu/):

1. Accept provisional ID 56 in the diagnostic device-name lookup. Unpatched QEMU asserts.
2. Forward whole-device reset to the backend and wait for its acknowledgement. Without this,
   stopping the queues leaves the backend's old context and buffer handles alive.
3. Let the transport, not the backend, offer `VIRTIO_F_ACCESS_PLATFORM`. Without this, a backend
   that can translate makes the guest see the feature even with `iommu_platform=off`, where
   nothing translates: a device that claims confinement and has none.
4. Grant each IOTLB miss the access it asked for. Without this, every translation is handed to
   the backend read-write, and a page the guest mapped read-only can be written once read.

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
Set `QEMU_TARGET=x86_64` for Linux microvm guests (TCG or KVM). CI caches the
complete source/build tree, including firmware, separately for each target.
The key covers the pinned revision, patches, build script, runner architecture,
compiler and runtime dependencies. Every restored tree is checked and rebuilt
incrementally before the real QEMU tests run; there are no partial cache restores.
The script configures out of tree and fingerprints its options, toolchain and GLib
configuration, retaining object files on repeated invocations. Set
`QEMU_RECONFIGURE=1` to force configuration after local dependency changes.

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
The separate [Linux guest driver and SDK boot test](../guest/linux/README.md)
exercise Python inside a real Linux VM using this same host binding.

## Behind an IOMMU

By default the device takes guest-physical addresses and the daemon maps all of guest RAM: a
driver that names any guest memory in a descriptor has the device read or write it. A guest whose
drivers are not trusted with that needs the device confined, and `--iommu` confines it:

- the daemon offers **and requires** `VIRTIO_F_ACCESS_PLATFORM`, and never starts a queue for a
  frontend that would hand it untranslated addresses;
- it keeps the memory table's files **without mapping them**, maps each range when the guest's
  IOMMU domain grants it (vhost-user `IOTLB_MSG`), in the direction granted, and unmaps on
  invalidation;
- rings and every buffer of a chain are resolved through the domain before the command is
  dispatched. An address the domain does not map, or does not map for that direction, makes the
  chain unusable (used length 0), exactly like any other unreachable buffer.

The frontend must be Virtio PCI with `iommu_platform=on` behind the machine's IOMMU; QEMU does not
route a Virtio MMIO transport through one. `launch.py --iommu` assembles both halves (aarch64
`virt` with SMMUv3; an x86 machine has not been run). QEMU patches 3 and 4 above are required for
the confinement to be real, and `vendor/vhost` must be the `vhost` 0.16.0 crate with IOTLB support
added: no released version has the backend side.

Cost: the frontend answers a miss with one page, so a buffer's first use costs a socket round trip
and a mapping per page. Grants persist until invalidated, so memory a driver keeps mapped pays once.

The translation is enforced by host software — QEMU's IOMMU model decides and this daemon obeys.
The daemon still holds the guest's memory files.

## Native backends

Build the optional Linux adapters explicitly:

```sh
cargo build --manifest-path host/vhost-user/Cargo.toml --locked \
  --features openvino,vulkan,xdna
```

Use `--backend openvino`, `--backend vulkan`, or `--backend xdna`. `--device NPU` or `--device GPU`
selects an OpenVINO endpoint explicitly; without it OpenVINO prefers NPU, then GPU, then CPU.
`--backend vulkan --device 'exact device name'` selects a Vulkan endpoint. Discover
names with `--backend openvino --list-devices` or `--backend vulkan --list-devices`
(no socket needed). Vulkan prints its selected device. Backend initialization failure is an error, not a silent
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

Connect the patched QEMU frontend with `virtio-id=56,num_vqs=1,vq_size=1024,config_size=16`,
modern Virtio MMIO (`-global virtio-mmio.force-legacy=false`), a `chardev socket`, and a shared
`memory-backend-memfd` object. The complete runnable invocation is in the smoke test. A real guest
still needs a driver implementing this device's command protocol.

The daemon serves one connection and exits on disconnect or an unrecoverable transport/backend
fault. It does not replace an existing socket, reconnect an old device, or remove the caller's
socket path. Remove the private directory after its processes have stopped.

For a Linux kernel/initramfs, `python3 host/vhost-user/launch.py --help` provides
a launcher which checks prerequisites, creates a private socket directory, prints
both complete commands, and stops its children before removing that directory.
Use `--dry-run` to inspect the commands without starting either process. For example:

```sh
python3 host/vhost-user/launch.py \
  --qemu /tmp/va-qemu/build/qemu-system-x86_64 \
  --daemon host/vhost-user/target/debug/virtio-accel-vhost-user \
  --backend openvino --device NPU --allow-unknown-program-residency \
  --kernel /path/to/vmlinuz --initrd /path/to/initramfs \
  --append 'console=ttyS0 rdinit=/init'
```

TCG is the default. `--accel kvm` requires accessible `/dev/kvm`; the x86 launcher
disables asynchronous page-fault injection for Linux/microvm compatibility.
For ARM use `--target aarch64` with a matching QEMU binary/kernel and
`--append 'console=ttyAMA0 ...'`. Additional disk/network arguments can follow `--`;
do not duplicate the launcher's machine, RAM, serial or accelerator definitions.
The guest image must contain a matching driver and SDK; the launcher does not
install software into it. The direct daemon leaves socket cleanup to its caller.

### Resource policy

`--buffer-budget BYTES` defaults to 512 MiB of actual retained buffer backing.
`--program-budget BYTES` defaults to 256 MiB of declared retained program charge.
Both must be nonzero. These budgets are host policy and cannot be enlarged by guest commands.

Current native compilers declare `u64::MAX` program residency because they expose no finite upper
bound. To admit one such program, the operator must explicitly use:

```sh
--program-budget 18446744073709551615
```

The equivalent, readable option is `--allow-unknown-program-residency`. Starting
a native host without that admission policy fails immediately with guidance,
instead of waiting for the first guest program to fail with RESOURCE_LIMIT.

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
  not on this wire. The Linux guest driver grants one exclusive whole-device
  session per open file description and resets it on final close. A future
  multi-tenant guest service would need finer object ownership mediation.

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

# Experimental Linux guest driver

This out-of-tree module connects the Python SDK inside a Linux VM to the
[vhost-user host](../../docs/vhost-user.md). It registers `/dev/virtio-accelN`
for provisional Virtio device ID **56**. This is not an allocated Virtio
ID, an upstream Linux driver, or a production QEMU device. The frozen device
protocol remains 1.0; Linux APIs stay outside the portable crates.

## Use inside a guest

Build against the running guest kernel's headers. The kernel needs
`CONFIG_VIRTIO`, `CONFIG_VIRTIO_MMIO` and `CONFIG_MODULES`.

```sh
make -C guest/linux
sudo insmod guest/linux/virtio_accel.ko
ls -l /dev/virtio-accel*
```

The device defaults to `0600`. Grant access to a trusted guest user/group through
your device-management policy. Access grants ownership of this virtual device
and its configured host resource budgets.

For a persistent installation, `sudo make -C guest/linux install` installs for
`KDIR`'s kernel, then `sudo modprobe virtio_accel` loads it. `uninstall` removes
that installed module and refreshes dependencies; unload it first with
`sudo modprobe -r virtio_accel`. Secure Boot guests may require module signing.
`DESTDIR=/path/to/staging` stages installation without modifying the running OS.

An optional udev rule grants access to a trusted guest group:

```sh
sudo groupadd -f virtio-accel
sudo usermod -aG virtio-accel "$USER"
sudo make -C guest/linux install-udev
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=misc --sysname-match='virtio-accel*'
```

Log in again for group membership to take effect. `uninstall-udev` removes the
rule; reload rules and recreate the device to restore its default permissions.

Install a wheel built **from this change**; earlier PyPI releases lack the guest
provider. The regular wheel includes the transport. To build a smaller guest-only
wheel on a Linux build machine of the same architecture:

```sh
python3 -m venv .venv
.venv/bin/pip install 'maturin>=1.16,<2'
.venv/bin/maturin build --manifest-path python/Cargo.toml --locked \
  --no-default-features --features python --out dist
```

Copy the wheel to the guest and install with `pip install /path/to/virtio_accel-*.whl`
in a guest virtual environment. Vendor SDKs/runtimes remain exclusively on the host.

```python
import numpy as np
import virtio_accel as va

device, = va.devices(provider="virtio")
with va.jit(lambda x: x + x, device=device) as double:
    x = np.arange(32, dtype=np.float32).reshape(1, 32)
    np.testing.assert_array_equal(double(x).to_numpy(), x + x)
```

`va.device_diagnostics()` explains permission, busy-device and protocol errors.
Provider-filtered discovery probes only that provider. Device indices belong to
that filtered snapshot; pass the `Device` object to preserve its selection scope.
Virtio discovery caches validated metadata until the device node changes, without
retaining an idle exclusive fd. Executable creation rechecks the node identity and
device UUID. Diagnostics explicitly refresh the probe, so they can report current
permission/busy errors (an idle refresh opens/closes and resets the device).
The device name identifies its guest node. Its class comes from the host: GPU,
NPU, or conservatively `hybrid` for OTHER/DSP. Protocol 1.0 does not expose the
host provider's name or CPU participation. Guest endpoints conservatively set
`cpu_participation=True` and reject `strict=True`. The existing Rust compiler
provides TOSA bytes and target words; the host admits or rejects each graph.
There is no fallback or numerical-contract change.

## Ownership and limits

- One open file description owns the entire virtual device. Other opens return
  `EBUSY`; duplicated descriptors share ownership. SDK executables in one process
  share a serialized session. Do not use executables after `fork`; exec a fresh
  process. Close executables explicitly to observe release errors.
- Final fd close resets the device and invalidates old object IDs. A malformed
  completion, transport failure or execution timeout loses the shared session,
  including other executables using it. Accepted submissions are never replayed.
- Each ioctl publishes one direct chain of kernel-owned pages. There is no
  userspace mapping or external-memory import. Tensor transfers use the frozen
  `WRITE_BUFFER`/`READ_BUFFER` copy operations.
- Capacity derives from descriptor count, ring size, page size and protocol
  limits, capped at 1 MiB per direction. The current host with 4 KiB pages gives
  **512 KiB** per direction. Buffer copies are chunked. A TOSA artifact must fit
  one request: currently **512 KiB minus 96 bytes**. Oversized artifacts are
  rejected before object creation.
- Each ioctl waits up to 60 seconds for completion. Reset must finish before
  reclaiming pages; a hung VMM/host control plane can delay reset itself. Python's
  execution deadline also covers polling and copies.
- Native host backends currently require a maximal declared program-residency
  charge, allowing one resident native program per host instance. Clear unused
  SDK executables before loading another independent program. JIT keeps up to 16
  traced graphs but defaults to **one resident executable** for virtio, retiring
  the old program before compiling a different shape. `resident_cache_size` is
  separate from `cache_size`; larger resident caches require a compatible host
  budget. `clear_cache()` permits reuse; `close()` and context-manager exit
  permanently close a JIT function and report cleanup errors.
- Completion polling retries at most four times within 50 µs, then parks for 50–1000 µs
  with a bounded exponential backoff. The same Rust wait utility serves native
  workers and guest Python, releasing the GIL during waits. Protocol 1.0 still
  requires polling; there is no event queue or synthesized completion notifier.
- Suspend/hibernation is rejected. Migration, reconnect, packed queues,
  multi-tenant scheduling and an allocated device ID remain outside this prototype.

The provisional Linux ioctl ABI is in
[`include/uapi/virtio_accel.h`](include/uapi/virtio_accel.h). Fixed-width native-endian
headers and an inline byte tail have identical compat layouts. This ABI is
separate from the little-endian device protocol. Read the [memory/lifetime audit](SAFETY.md).

## Reproduce the boot test

The harness uses x86-64 QEMU microvm. Install the guest kernel and matching headers,
static BusyBox, cpio, gzip, C compiler, Python/venv, ninja and GLib development
headers on the build host. `KDIR` must identify the **guest** kernel headers.
The harness never loads the module into the host kernel.

```sh
make -C guest/linux KDIR=/lib/modules/GUEST_KERNEL_VERSION/build W=1
QEMU_TARGET=x86_64 host/vhost-user/qemu/build.sh /tmp/va-qemu
cargo build --manifest-path host/vhost-user/Cargo.toml --locked --features openvino

.venv/bin/pip install dist/*.whl
.venv/bin/python guest/linux/tests/build_initramfs.py \
  --module guest/linux/virtio_accel.ko --output /tmp/va-initramfs.gz --backend openvino

# Install OpenVINO in a separate HOST environment; supply its C runtime path.
export VIRTIO_ACCEL_OPENVINO_LIBRARY=/path/to/host/openvino/libs/libopenvino_c.so.VERSION
python3 guest/linux/tests/qemu_guest.py \
  --qemu /tmp/va-qemu/build/qemu-system-x86_64 \
  --kernel /path/to/readable/vmlinuz --initramfs /tmp/va-initramfs.gz \
  --daemon host/vhost-user/target/debug/virtio-accel-vhost-user \
  --backend openvino --device CPU --log /tmp/va-linux-guest.log
```

Reading a distribution kernel in `/boot` may require making a readable copy.
The guest kernel also needs `CONFIG_VIRTIO_MMIO_CMDLINE_DEVICES` for microvm.
Use `--device NPU` with an installed OpenVINO NPU runtime, or build the host with
`--features vulkan` and use `--backend vulkan` without `--device`. `--accel kvm`
is optional; CI uses TCG without nested virtualization. The harness disables KVM
asynchronous page-fault injection after an observed Linux 7.0/microvm early-boot
fault before module loading. It stops the VM after its final serial marker.

Tests cover ioctl validation, exclusive open, paged/chunked copies, stale IDs,
cached Python identity/add graphs, injected timeout recovery, and unbind/rebind
with an open fd. The timeout test pauses only its own host daemon for two seconds.
Guest root is confined to the disposable VM. CPU results prove the SDK/transport
path; physical GPU/NPU runs provide separate hardware evidence. ARM64 driver
execution has not yet been validated.

API references: [Linux Virtio drivers](https://docs.kernel.org/driver-api/virtio/writing_virtio_drivers.html),
[ioctl guidance](https://docs.kernel.org/driver-api/ioctl.html).

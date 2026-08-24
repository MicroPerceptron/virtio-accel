# AMD virtio-npu vs virtio-accel: a positioning comparison

Review notes on [amd/virtio-npu](https://github.com/amd/virtio-npu) (read at commit
`bb835b6`, 2026-08-23), AMD's own NPU-virtualization code, compared against this project's
protocol. Reviewed by Cass and by this session independently; conclusions agreed. Useful as
positioning evidence for the eventual standardization story.

## What AMD built

Not a new virtio device class and not a portable protocol: a **DRM native context riding on
virtio-gpu**. The repo is two QEMU patches (~1,000 lines) plus a test client; the substance
lives in `libvxdna.so` inside `amd/xdna-driver`. Mechanics:

- The QEMU device subclasses `TYPE_VIRTIO_GPU_PCI_BASE` and advertises
  `VIRTIO_GPU_CAPSET_DRM` ("Since NPU accelerator is GPU DRM device, we use
  VIRTIO_GPU_CAPSET_DRM"), with BLOB/DMABUF/CONTEXT_INIT flags enabled.
- The guest runs AMD's **entire native stack** (XRT, shim, real driver semantics). Vendor
  commands (`AMDXDNA_CCMD_CREATE_CTX/CREATE_BO/…`, `amdxdna_proto.h`) are marshaled through
  `DRM_IOCTL_VIRTGPU_EXECBUFFER` and replayed by `libvxdna.so` against the host's
  `/dev/accel/accel0`.
- Buffers are virtio-gpu blob resources over shared memfd VM memory (zero-copy);
  completion reuses virtio-gpu's fence queue (fence fd + poll in the test client).
- Host configuration: `-device virtio-accel-pci,accel-node=accel0`. Validation: `xrt-smi`
  inside the VM.
- Naming collision: their device type and trace namespace are literally `virtio-accel-pci` /
  `virtio_accel*` — identical to this project's name. Resolve early if standardization is
  pursued.

## Comparison

Different games, not just different scores:

| Axis | AMD virtio-npu | virtio-accel |
|---|---|---|
| Abstraction | Vendor ioctl tunnel (native context) | Device-neutral TOSA contract |
| Guest requirements | Full AMD userspace, version-locked to host driver | One portable guest driver, any vendor backend |
| Attack surface | Entire amdxdna ioctl + XRT command surface exposed to guest-controlled input, no semantic validation between guest and kernel driver | Validated TOSA admission, per-op legality, reject-don't-fallback; small auditable TCB |
| Numerics | None (transport only) | Advertised tiers + conformance oracles (machine-checked honesty) |
| Feature breadth | Everything XRT does, day one (arbitrary xclbins, custom kernels, profiling) | Exactly the advertised tier |
| Performance | Near-native (command-buffer replay, zero-copy blobs) | Admission + backend translation on the path |
| Migration/versioning | Native-context compat matrix (guest XRT ↔ host driver ABI drift) | Frozen wire ABI + versioned conformance directories |

Verdict: for this project's goals — portability, minimal TCB, honest numerics, a
capability-based host (kore) — virtio-accel isn't merely better, it's answering a question
virtio-npu doesn't ask. Conversely, as an *ecosystem tunnel* theirs is effective and cheap.
Their existence is also evidence that **the standardization slot for a portable NPU virtio
class remains open**: the vendor's own answer is "tunnel our stack," not "define a neutral
device."

## Ideas adopted or noted

1. **Composition with kore's doctrine** (subsume-by-virtualization): a future kore host can
   run both — virtio-accel as the portable seam for normal guests, and a virtio-npu-style
   vendor tunnel into one sacrificial Linux guest hosting the AMD compiler/XRT tooling.
   That guest doubles as the offline compile box for the #84 catalog mode.
2. **Raw-artifact escape hatch validated**: `amdxdna_proto.h` is the demand-proof for the
   vendor-native artifact `format` sketched in the #84 outlook — ours would be an opt-in
   registered format beside validated TOSA, never the only door.
3. **Zero-copy blob mapping over shared VM memory**: the mature reference for what
   `DIRECT_BINDING`/host-shared buffer properties should become at real VMM integration.
4. **Completion signaling**: fence-fd/poll → map our event completions onto
   irqfd/eventfd-style signals at the device layer when a VMM integration lands.
5. **Ergonomics**: host-node-by-name device configuration (`accel-node=accel0`).

No settled decision changes; items 2–4 are future device-layer/protocol work, item 1 is
kore-planning context.

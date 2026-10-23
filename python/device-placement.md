# Portable device selection

Status: `npu`/`npu:0` through Core ML and `gpu`/`gpu:N` through Vulkan are implemented.
`cpu` is reserved and currently unavailable. OpenVINO, Hexagon and XDNA remain unbound.
The remaining design work is described below; [README](README.md) covers execution and lifetimes.

The public execution API selects a device class, with an optional instance index:

```python
@va.jit(device="npu")       # default NPU instance; equivalent to npu:0
def infer(x, weight):
    return x @ weight

devices = va.devices("npu") # runtime discovery, not a platform-name heuristic
```

Recognize `cpu`, `gpu`, `npu`, and indexed forms such as `gpu:1` and `npu:0`. An index is
relative to the current discovery snapshot, not a persistent hardware identity. Keep
the indexed form even on single-NPU laptops: discrete accelerators, passthrough devices,
and remote/virtual adapters can make multiple instances useful.

Provider selection belongs below this API. Users should not need to choose `coreml`,
`openvino`, `hexagon`, `xdna`, or `vulkan` for ordinary code. Keep provider names visible
in diagnostic/device records and optionally available as an advanced override later.

## Resolution and numerical contracts

1. Enumerate available runtime adapters and real devices, excluding test mocks.
2. Normalize the requested class/index and choose one device deterministically.
3. Check the graph's dtype, operator, shape and target against that device's advertised
   tier; native program compilation remains the final admission check.
4. Report the selected provider, native device name, supported tier, and placement
   guarantees. Reject an unavailable class/index or unsupported program explicitly.

An unsupported FP16 program on an XDNA device must not silently become BF16. A missing
NPU must not silently select a CPU or GPU. Any future fallback policy must be explicit.

`backend="reference"` remains a test/oracle execution mode and must not appear in real
device discovery as `cpu:0` or `npu:0`.

## Structural frictions in the current implementation

| Area | Current behavior | Adapter work needed |
| --- | --- | --- |
| Core ML | `CPUAndNeuralEngine` allows mixed per-operator placement; the backend reports a fixed NPU identity | Distinguish an NPU-capable runtime from proven exclusive NPU execution |
| OpenVINO | Enumerates devices and supports `with_device`; default preference is NPU, GPU, CPU | Use explicit resolved device names so a requested NPU never takes the generic default fallback path |
| Vulkan | Enumerates devices by name; a physical GPU can also be visible through OpenVINO | Deduplicate using verified physical identity, not provider names alone |
| XDNA | Current HRX device index is fixed to zero; advertised BF16/FP8/INT8 tiers do not include FP16 arithmetic | Expose discovery/index selection before claiming multiple instances; report FP16 admission failures |
| Hexagon | Runtime/build availability is platform-gated; current identity is not a complete discovery inventory | Probe actual runtime availability and preserve hardware evidence limits |
| Portable identity | No assigned CPU class; several UUID/vendor fields are synthetic or fixed | Keep richer host-side discovery metadata; do not assign new protocol values implicitly |

In particular, `device="npu"` cannot truthfully mean "every operation runs only on the
NPU" across all current backends. A practical default can mean selecting the NPU-capable
provider, with its possible CPU participation reported in placement metadata. The implemented
`strict=True` option rejects Core ML because exclusive placement cannot be guaranteed.
The frontend does not infer exclusive placement from `AcceleratorClass::NPU` alone.

Discovery also needs stable ordering and a policy for physical devices exposed through
multiple providers. The existing UUIDs are insufficient for universal deduplication.
Prefer platform/PCI identities where available, and report ambiguity rather than claiming
that two provider endpoints definitely represent two separate physical accelerators.

These changes belong in host-side adapters and Python runtime metadata. Selecting an
existing device through a portable name does not require a wire-protocol change.

Sources in this repository: [device support matrix](../docs/device-support-matrix.md),
[Core ML](../crates/virtio-accel-coreml/src/lib.rs),
[OpenVINO](../crates/virtio-accel-openvino/src/lib.rs),
and [Vulkan](../crates/virtio-accel-vulkan/src/lib.rs).

## Current discovery limits

Vulkan endpoints are sorted by name, opened to verify their reported class, and CPU
software devices are excluded. The Rust selection API currently opens by name, so
all duplicate names are excluded with a diagnostic; the Python adapter does not invent
an index that would silently reopen the first matching device. A future Rust selection
API must expose physical-device identity to support multiple identically named GPUs.
Discovery diagnostics include missing loaders and initialization failures. No installation
or environment mutation is performed automatically.

Core ML currently lacks CAST in its advertised floating-point tier. FP16 graph support
does not imply that graphs mixing FP16 and FP32 will be admitted. This is a provider
limitation, separate from the portable selector syntax.

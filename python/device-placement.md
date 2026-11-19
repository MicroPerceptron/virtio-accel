# Portable device selection

Status: Core ML, Vulkan, OpenVINO, Hexagon and XDNA adapters are implemented.
`cpu` selects native OpenVINO CPU execution. Vendor SDK/runtime detection and the
selected provider's advertised tier determine availability; see [README](README.md).

The public execution API selects a device class, with an optional instance index:

```python
@va.jit(device="npu")       # default NPU instance; equivalent to npu:0
def infer(x, weight):
    return x @ weight

devices = va.devices("npu") # runtime discovery, not a platform-name heuristic
```

Recognize `cpu`, `gpu`, `npu`, `hybrid`, and indexed forms such as `gpu:1` and `npu:0`. An index is
relative to the current discovery snapshot, not a persistent hardware identity. Keep
the indexed form even on single-NPU laptops: discrete accelerators, passthrough devices,
and remote/virtual adapters can make multiple instances useful.

Provider selection belongs below this API. Users should not need to choose `coreml`,
`openvino`, `hexagon`, `xdna`, or `vulkan` for ordinary code. Keep provider names visible
in diagnostic/device records and available through `va.devices(provider=...)`; pass the returned `Device` to `jit` or `compile`.

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

## Provider boundaries

| Area | Current behavior | Boundary preserved by Python |
| --- | --- | --- |
| Core ML | CPU/ANE, CPU/GPU, and All policies expose NPU, GPU, and hybrid endpoints | Reports CPU participation and rejects strict placement; hybrid is not an exclusive device |
| OpenVINO | Enumerates devices and supports `with_device`; default preference is NPU, GPU, CPU | Opens exact enumerated names; rejects AUTO/HETERO/MULTI endpoints |
| Vulkan | Enumerates devices by name; a physical GPU can also be visible through OpenVINO | Reports distinct provider endpoints; makes no physical deduplication claim |
| XDNA | Current HRX device index is fixed to zero; advertised BF16/FP8/INT8 tiers do not include FP16 arithmetic | Exposes one HRX endpoint; supports explicit BF16 graphs and rejects FP16/FP32 inputs |
| Hexagon | Runtime/build availability is platform-gated; current identity is not a complete discovery inventory | Probes QNN HTP; exposes one endpoint and keeps its FP16 precision boundary |
| Portable identity | No assigned CPU class; several UUID/vendor fields are synthetic or fixed | Keep richer host-side discovery metadata; do not assign new protocol values implicitly |

In particular, `device="npu"` cannot truthfully mean "every operation runs only on the
NPU" across all current backends. A practical default can mean selecting the NPU-capable
provider, with its possible CPU participation reported in placement metadata. The implemented
`strict=True` option rejects Core ML because exclusive placement cannot be guaranteed.
The frontend does not infer exclusive placement from `AcceleratorClass::NPU` alone.

Discovery orders Core ML NPU first, then Vulkan GPU, Core ML GPU/hybrid, OpenVINO, Hexagon and XDNA, with
sorted native names. Indices are allocated per class across providers. The existing UUIDs are insufficient for universal deduplication.
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

Core ML, OpenVINO and the Vulkan base tier currently lack CAST in their advertised floating-point tiers. FP16 graph support
does not imply that graphs mixing FP16 and FP32 will be admitted. This is a provider
limitation, separate from the portable selector syntax.

Core ML's `hybrid` endpoint permits CPU, GPU and ANE in one model without promising
simultaneous execution. It requires both accelerators at discovery. FP32 NeuralNetwork
graphs narrow the permitted set to CPUOnly for the NPU policy or CPUAndGPU for hybrid,
since ANE may otherwise reduce intermediate precision. FP16 retains the selected policy.
These are host-side choices; hybrid maps to the existing OTHER class without new wire values.

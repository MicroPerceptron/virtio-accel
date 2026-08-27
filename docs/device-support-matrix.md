# Device support matrix

The [backend support table](../README.md#backend-support) answers "which dtypes and programs does
each backend admit". This document answers the different question underneath it: **which physical
devices actually execute the work, and what exactly stops the rest from doing so.**

The distinction matters because two of the four host backends are heterogeneous by construction.
Core ML and OpenVINO are runtimes that dispatch across a machine's whole inference estate, not NPU
drivers, so parts of this project already run on CPUs and can run on GPUs. The other two, XDNA and
Hexagon, are single-device by design. Naming that split explicitly is more useful than an "NPU"
label that is true of the intent and only partly true of the code.

## How to read this

Every row carries one of four statuses. They are claims about *this repository*, not about the
hardware.

| Status | Meaning |
| ------ | ------- |
| **Validated** | Executed on that device, with an evidence pin in-repo naming the host, driver, and runtime versions. |
| **Reachable** | The selection and dispatch path drives the device today with no code change, but no in-repo evidence pins that part. |
| **One change away** | Not reachable today. Each row names the single gate — a constant, a path, or a build condition — and what it would take. |
| **Out of scope** | No path, and none implied by the current design. |

"Reachable" is deliberately weaker than "supported". It means the code will select the device and
try; it does not promise the program admits, the numerics match, or the performance is sane.

## The matrix

| Device | Class | Backend | Host OS / arch | Status |
| ------ | ----- | ------- | -------------- | ------ |
| **Apple Neural Engine**, Apple silicon (M-series) | NPU | `coreml` | macOS 14+ | **Validated** — Apple M4, macOS 26.5.2 ([performance.md](performance.md#core-ml-provider-evidence)) |
| **Apple CPU**, as Core ML per-operator placement | CPU | `coreml` | macOS 14+ | **Reachable** — see [Core ML](#apple-core-ml-virtio-accel-coreml) |
| **Apple GPU** | GPU | `coreml` | macOS 14+ | **One change away** — compute units are pinned |
| **Intel Mac** (CPU, AMD/Intel GPU) | CPU / GPU | `coreml` | macOS 14+ | **One change away** — ANE gate refuses construction |
| **Intel NPU**, Core Ultra (Meteor Lake / Lunar Lake / Arrow Lake) | NPU | `openvino` | Any host with the runtime | **Reachable** — first device preference |
| **Intel GPU**, integrated Xe/UHD and discrete Arc | GPU | `openvino` | Any host with the runtime | **Reachable** — including indexed `GPU.1` |
| **x86-64 CPU**, Intel | CPU | `openvino` | Any host with the runtime | **Validated** — `openvino-host-test` CI lane, OpenVINO 2026.3.0 |
| **x86-64 CPU**, AMD | CPU | `openvino` | Any host with the runtime | **Reachable** — misreports vendor, see [OpenVINO](#intel-openvino-virtio-accel-openvino) |
| **ARM64 CPU** (Apple silicon, Ampere, Raspberry Pi) | CPU | `openvino` | Any host with an ARM CPU-plugin build | **Reachable** — enumerates as `CPU`, misreports vendor |
| **OpenVINO virtual devices** (`AUTO`, `MULTI`, `HETERO`, `BATCH`) | — | `openvino` | — | **One change away** — resolution requires enumeration |
| **AMD XDNA2 NPU**, Strix Point / Strix Halo / Krackan Point | NPU | `xdna` | Linux, `amdxdna` driver | **Validated** — PCI `1022:17f0` rev `0x20`, Fedora 44 ([baseline](../crates/virtio-accel-xdna/README.md#validated-baseline)) |
| **AMD XDNA1 NPU**, Phoenix / Hawk Point | NPU | `xdna` | Linux, `amdxdna` driver | **Reachable** — ungated but unvalidated, see [XDNA](#amd-xdna-virtio-accel-xdna) |
| **Second and later XDNA NPUs** in one host | NPU | `xdna` | Linux | **One change away** — device index is fixed at 0 |
| **Qualcomm Hexagon HTP v73**, Snapdragon X | NPU | `hexagon` | Windows 11 ARM64 | **Validated** — Snapdragon X126100, QAIRT 2.49 ([baseline](../crates/virtio-accel-hexagon/README.md#validated-baseline)) |
| **Qualcomm Hexagon HTP v75+**, newer Snapdragon | NPU | `hexagon` | Windows 11 ARM64 | **Reachable** — ungated, misreports v73 |
| **Qualcomm Adreno GPU / Kryo CPU** via QNN | GPU / CPU | `hexagon` | Windows 11 ARM64 | **One change away** — backend library path is fixed, and deliberately so |
| **Snapdragon on Linux or Android** | NPU | `hexagon` | — | **One change away** — build target gate |
| **No device**, executed in software | — | `mock` | Any | Deterministic in-memory reference; outside this vocabulary |
| **Whatever a wrapped provider drives** | — | `vaccel` | Any (`std`) | Pass-through; the wrapped backend decides |
| **Vulkan-capable GPUs** | GPU | — | — | **Out of scope** — planned, not started |

## Per-backend detail

### Apple Core ML (`virtio-accel-coreml`)

**What selects the device.** Nothing does, explicitly. The backend hands Core ML a fixed compute
budget and Core ML places each operator itself:

```objc
configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
```

— [`coreml_bridge.m:277`](../crates/virtio-accel-coreml/native/coreml_bridge.m#L277)

That constant is the whole device policy. It grants the ANE and the CPU, and withholds the GPU.

**The CPU is already in play.** Because `MLComputeUnitsCPUAndNeuralEngine` includes the CPU, a
model containing operators the ANE declines runs those operators on the CPU — silently, inside a
submission that `device_info()` reports as `AcceleratorClass::NPU`. This is not a defect; it is
Core ML's documented placement model, and the crate README states it. It does mean **the project's
first non-NPU execution path already exists and already ships**, and that per-submission device
attribution is not observable through the `Accelerator` contract.

**What excludes everything else.** Construction refuses any host without an ANE:

```rust
if unsafe { va_coreml_has_neural_engine() } == 0 {
    return Err(InitError::NeuralEngineUnavailable);
}
```

— [`macos.rs:383`](../crates/virtio-accel-coreml/src/macos.rs#L383), backed by a
`MLNeuralEngineComputeDevice` scan at
[`coreml_bridge.m:121`](../crates/virtio-accel-coreml/native/coreml_bridge.m#L121)

This gate, not the framework, is what excludes Intel Macs — Core ML runs there, it simply has no
ANE. It also excludes ANE-less VMs, which is why the CI example on `macos-latest` skips rather than
fails.

**The two changes.**

- *Apple GPU:* switch the constant to `MLComputeUnitsAll`. One line. It widens placement to the
  GPU without adding a device to select, so nothing above the bridge changes.
- *Intel Macs and ANE-less hosts:* soften the ANE gate to a capability probe. Larger than it looks
  — `identity.uuid` (`apple-coreml-ane`) and `identity.class` (`NPU`) are compile-time constants at
  [`macos.rs:393`](../crates/virtio-accel-coreml/src/macos.rs#L393) and would have to become
  runtime-derived to stay truthful.

### Intel OpenVINO (`virtio-accel-openvino`)

**The genuinely heterogeneous backend.** Device selection is explicit, ordered, and already covers
three classes:

```rust
let device = ["NPU", "GPU", "CPU"]
    .into_iter()
    .find_map(|preferred| { /* first enumerated match */ })
```

— [`native.rs:842`](../crates/virtio-accel-openvino/src/native.rs#L842)

`with_device` ([`native.rs:855`](../crates/virtio-accel-openvino/src/native.rs#L855)) pins one
device by enumerated name or class prefix, so `"GPU.1"` selects the second GPU and `"NPU"` selects
whatever NPU instance exists. `matches_device`
([`native.rs:634`](../crates/virtio-accel-openvino/src/native.rs#L634)) makes prefix matching
strict: `"GPU"` matches `GPU` and `GPU.1`, never `GPUX`.

**Not an Intel-only backend.** The build gate is a pkg-config probe for the runtime, not a target
OS or vendor check ([`build.rs:35`](../crates/virtio-accel-openvino/build.rs#L35)). Consequently the
CPU plugin admits any x86-64 host including AMD, and an ARM CPU-plugin build enumerates `CPU` on
Apple silicon or Ampere just as readily. **This is the only backend in the repository whose device
execution is proven by CI** — `openvino-host-test` installs OpenVINO 2026.3.0 on an x86-64 Ubuntu
runner and executes the real path against the CPU plugin.

**Two truthfulness gaps.** `device_info_for`
([`native.rs:642`](../crates/virtio-accel-openvino/src/native.rs#L642)) hardcodes
`vendor_id: 0x8086`, so an AMD or ARM CPU host reports itself as Intel. And a CPU device falls
through to `AcceleratorClass::OTHER`, because [the class enum](../crates/virtio-accel-core/src/lib.rs#L44)
defines `OTHER`, `NPU`, `GPU`, and `DSP` but no `CPU`. A guest cannot currently distinguish "a CPU"
from "a device this backend does not recognize". Adding `AcceleratorClass::CPU` is additive — the
type is a `#[repr(transparent)] u16` designed for exactly this.

**The one change.** Virtual devices (`AUTO`, `MULTI`, `HETERO`, `BATCH`) are unreachable because
both constructors resolve requests *against the enumerated device list*, and the standard plugin
set does not enumerate them. `with_device("AUTO")` therefore returns `DeviceUnavailable`. A
pass-through for a known set of virtual names would unlock OpenVINO's own scheduling — worth
weighing against this project's preference for one submission mapping to one identified device.

### AMD XDNA (`virtio-accel-xdna`)

**Single device, index zero.** The process-wide owner takes the first device HRX reports and never
enumerates further:

```rust
check(ffi::hrx_gpu_device_get(0, &mut device))
```

— [`native.rs:132`](../crates/virtio-accel-xdna/src/native.rs#L132)

Multi-NPU hosts are therefore reachable only at index 0. The fix is mechanical — plumb an index
through `shared_device` — but the HRX fork's model is one process-wide device, so it is a design
question rather than a parameter change.

**No generation gate.** Nothing checks the PCI ID. An XDNA1 part (Phoenix / Hawk Point) would be
opened and driven, while `device_info` reports it as XDNA2 regardless:

```rust
uuid: *b"amd.xdna.npu\0\0\0\0",
vendor_id: 0x1022,
device_id: 0x17f0,
```

— [`native.rs:691`](../crates/virtio-accel-xdna/src/native.rs#L691)

Program admission also applies the XDNA2 shape and local-memory envelope, so an XDNA1 host would
most likely fail during compilation rather than produce wrong numerics — but it would fail
confusingly, and it would misidentify itself first.

**Linux only, by runtime.** The build script requires `libhrx.so` and the amdxdna-native headers
([`build.rs:58`](../crates/virtio-accel-xdna/build.rs#L58)), and the validated stack is the in-tree
`amdxdna` driver exposing `/dev/accel/accel0`. Windows XDNA uses a different driver stack entirely
and is not addressed.

### Qualcomm Hexagon (`virtio-accel-hexagon`)

**HTP only, by construction.** The QNN backend library is a fixed path:

```rust
let path = root.join("lib/aarch64-windows-msvc/QnnHtp.dll");
```

— [`native.rs:81`](../crates/virtio-accel-hexagon/src/native.rs#L81)

QAIRT also ships `QnnCpu`, `QnnGpu`, and `QnnDsp` libraries, and parameterizing this path would
reach the Adreno GPU and Kryo CPU. **This exclusion is deliberate**, not an oversight: the crate
documents that SDK-free builds fail explicitly rather than fall back, so that a host silently
executing on the CPU can never be mistaken for NPU evidence. Treat it as a policy to revisit
consciously, not a gap to close.

**No SoC gate.** The build pins the *target*, `windows` + `aarch64`
([`build.rs:31`](../crates/virtio-accel-hexagon/build.rs#L31)), but nothing pins the SoC. A newer
Snapdragon with HTP v75 or v79 would load and run, reporting itself as v73 the whole time
(`uuid: *b"qualcomm-htp-v73"`, `device_id: 73` at
[`native.rs:457`](../crates/virtio-accel-hexagon/src/native.rs#L457)). In practice such a host also
needs `ADSP_LIBRARY_PATH` pointed at its own DSP libraries — an environment concern the crate
README covers — and the FP32/FP8 rejections recorded for v73 may not describe it correctly.

**Linux and Android are a build gate, not a port.** QAIRT ships aarch64 Linux and Android
libraries. The `windows`/`aarch64` assertion and the hardcoded `lib/aarch64-windows-msvc` path are
the only two things naming the OS.

## What each backend reports

Useful when reading `DeviceInfo` in a trace. Only the OpenVINO row varies at runtime, and only from
the enumerated device *name* — not from the hardware. Every other value here is a compile-time
constant, which is why a v75 Snapdragon still reports `qualcomm-htp-v73`.

| Backend | `uuid` | `class` | `vendor_id` | `device_id` |
| ------- | ------ | ------- | ----------- | ----------- |
| `coreml` | `apple-coreml-ane` | `NPU` | `0x106b` (Apple) | `0` |
| `openvino` | `intel-ov-<device>` | `NPU` / `GPU` / `OTHER` | `0x8086` (always) | `0` |
| `xdna` | `amd.xdna.npu` | `NPU` | `0x1022` (AMD) | `0x17f0` (always) |
| `hexagon` | `qualcomm-htp-v73` | `NPU` | `0x17cb` (Qualcomm) | `73` (always) |
| `mock` | `virtio-accelmock` | `NPU` | `0` | `0` |

`class` comes from [`AcceleratorClass`](../crates/virtio-accel-core/src/lib.rs#L44), an extensible
`u16` newtype: `OTHER = 0`, `NPU = 1`, `GPU = 2`, `DSP = 3`. Unknown values stay representable
across implementations, so new classes are additive.

## Out of scope

- **Vulkan compute.** Listed as planned in the README backend table; no crate exists.
- **NVIDIA, via any path.** No CUDA, TensorRT, or cuDNN backend, and none implied.
- **Guest-side device passthrough.** The project claims no Virtio device ID yet; guests reach
  hardware through a host backend behind the [vAccel adapter](../crates/virtio-accel-vaccel/README.md).
- **Non-Apple ANE-class fixed-function blocks** with no runtime this project speaks to.

## Keeping this current

This document tracks device *reachability*, which changes for different reasons than dtype
coverage. Revisit it when:

- a device selection constant moves — the compute-unit budget, the QNN library path, the HRX device
  index, or the OpenVINO preference order;
- a `build.rs` target or runtime gate changes, which is what most often converts "one change away"
  into "reachable";
- a hardware evidence pin lands in a crate README or [performance.md](performance.md), which is what
  converts "reachable" into "validated"; or
- a `DeviceIdentity` field stops being a constant and starts being probed.

Adding a dtype or operator to an existing backend does not require a change here.

# Direct HTP runtime

This directory contains the provider-local FastRPC protocol used by
`DirectHexagonAccelerator`. It is deliberately separate from the QNN/TOSA
provider:

- QNN remains the conformant FP16/INT8 graph path.
- This runtime loads a signed, architecture-specific HTP skel and executes
  coarse kernels from one directly mapped `rpcmem` arena.
- V73 floating-point vector arithmetic uses QFloat32 (`Q6_Vqf32_*` followed
  by `Q6_Vsf_equals_Vqf32`). It is not advertised as strict IEEE/TOSA FP32.

`build.ps1` builds the V73 skel, generates a Windows catalog, and signs it.
The PFX must already be trusted by the machine and test-signed HTP modules
must be permitted by the installed Qualcomm driver. The script never changes
Secure Boot or boot policy.

The build requires Hexagon SDK 6.6 (compiler 19.0.07), CMake, Ninja, a trusted
PFX in `HEXAGON_HTP_CERT`, and Windows SDK tools. `Inf2Cat.exe` and the ARM64
`signtool.exe` are discovered independently because Windows SDK installations
can place them in different version directories:

```powershell
powershell -ExecutionPolicy Bypass -File .\crates\virtio-accel-hexagon\native\direct_htp\build.ps1
```

The signed local package is written to
`crates\virtio-accel-hexagon\target\direct-htp-package`. It is deliberately
ignored by Git: generated catalogs, signed skels, SDK binaries, certificates,
and private keys must not be published with the crate.

FastRPC reads `ADSP_LIBRARY_PATH` when its driver DLL is loaded. Set both paths
in the parent shell before starting a Rust executable; setting the variable
after the process has loaded FastRPC is too late:

```powershell
$package = (Resolve-Path '.\crates\virtio-accel-hexagon\target\direct-htp-package').Path
$env:ADSP_LIBRARY_PATH = $package
$env:VIRTIO_ACCEL_HTP_MODULE_DIR = $package
cargo test -p virtio-accel-hexagon direct_fp32_capability_spike_covers_required_operations_and_edges -- --ignored --nocapture
```

The provider-local artifact ABI currently admits identity, add, multiply,
reciprocal, reciprocal square root, matrix multiplication, fused Dneg
wormhole tracing, fused Kerr tracing, the compact diagnostic Kerr frame, and
resident reference-scene Kerr shading. Every
request is one FastRPC call
to this skel; there is no QNN graph, CPU, or GPU fallback. The fused tracing
kernels process 32 FP32 lanes per HVX vector through V73 QFloat32 arithmetic.
Four dedicated QuRT workers execute disjoint lane ranges on the four HVX
units. Each worker receives a private part of the 8 MiB VTCM reservation and
internally tiles larger host submissions through that scratchpad. Wormhole
`atan` and `log` use the SDK's HVX vector-polynomial routines.

The fused frame takes only a four-byte control binding; its camera, metric,
integrator, and render parameters are resident in the provider-local artifact.
HTP generates rays, runs the unchanged centered-difference midpoint
integrator, classifies events, executes the Axiom sky/disk shader, and returns
a 32-byte diagnostic header plus packed RGBA. Its workers use two VTCM slots
and overlap packed output with the next tile through V73 user DMA. The header
records event totals, the worker mask, peak concurrent workers, applied
compute/HVX/DCVS/core/bus votes, and DMA use.

For one-step frames, the common no-event path stays in HVX registers and uses
a low-stack event probe. A packet with any terminal lane is rerun through the
full trace before shading, preserving event state and pixels. The host can
consume the coherent mapped framebuffer through a scoped zero-copy view or
request an owned copy.

The accepted reciprocal and reciprocal-square-root implementation uses three
Newton refinements for unseeded values and two for nearby seeded values. This
configuration makes the restored static 320-step Kerr event and refinement
maps match the CPU oracle exactly. More aggressive refinement reductions and
an extra seeded refinement were rejected after worsening the hardware oracles.

The resident reference shader supports B8/E4M3, UF8 E4M4, and UF8 E5M3
storage; bilinear surface and trilinear volume sampling; emission/extinction
marching; spectral-transfer and blackbody lookup/interpolation; boundary
supersampling; animated plasma/sky time; and ACES/sRGB packed output. Axiom
uploads the immutable scene once and changes only a four-byte time word for
each animated frame.

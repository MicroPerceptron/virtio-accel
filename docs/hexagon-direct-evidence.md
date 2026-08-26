# Hexagon V73 direct-HTP evidence

Recorded on 2026-08-25 for issue #128 and Axiom issue #49.

## Change classification

This is a provider-local artifact/runtime addition. It does not change an
accepted or emitted virtio-accel 1.0 frame, opcode, payload, feature bit,
ownership rule, or transport behavior. Under `docs/wire-abi.md` section 9 it is
therefore **not a wire-protocol change**. The target and artifact identifiers
are private to `virtio-accel-hexagon` and require explicit caller selection.

## Hardware and software identity

- Device: `Snapdragon(R) X - X126100 - Qualcomm(R) Hexagon(TM) NPU`
- HTP architecture: V73; four HVX units; 8 MiB VTCM
- Windows driver: `30.0.222.0` (`oem131.inf`)
- QAIRT: `2.49.0.260730`
- Hexagon SDK: `6.6.0.0`
- Compiler: `QuIC LLVM Hexagon Clang version 19.0.07`
- Signed skel SHA-256: `c1da44916b212c0f2daebf6e6d5525644e083e4b4a32efa5101894fd95ee0b86`
- Development signing certificate: `CN=GGML.HTP.v1`; the catalog passed
  `signtool verify /pa`. Neither the certificate nor signed output is tracked.

The skel reports V73 from inside the FastRPC process. Direct artifacts can
only execute by invoking that signed skel; the implementation contains no QNN,
CPU, or GPU fallback.

## FP32 capability spike

Run with:

```powershell
cargo test -p virtio-accel-hexagon direct_fp32_capability_spike_covers_required_operations_and_edges -- --ignored --nocapture
```

Representative raw results were:

| Probe | Result bits |
|---|---|
| Identity | `3f800000 00000001 80000000 4788b800 c788b800 7f800000 ff800000 7fc12345` |
| ADD | `3f800008 4788b880 00000002 00000000 7f800000 ffffffff ffffffff 00000000` |
| MUL | `3f800008 00000002 80000000 c788b800 80000000 ffffffff ff800000 ffffffff` |
| Reciprocal | `3eaaaaab ff800000 7f800000 00000000 80000000 ffffffff 376facad 7f800000` |
| Reciprocal square root | `3f000000 3f3504f3 7f800000 ff800000 00000000 ffffffff ffffffff 64b504f3` |
| 2x3 by 3x2 MATMUL | `42000000 40900000 4888b740 c788b840` |

This proves preservation of FP32 subnormals, values beyond FP16 range, signed
zero in identity, and arithmetic differences below FP16 resolution. It also
proves the target is not generally IEEE-754 conformant: invalid arithmetic is
canonicalized to `0xffffffff` NaNs and some signed-zero results differ from
strict IEEE expectations. The capability remains named QFloat32 and strict
TOSA FP32 remains rejected.

## Axiom correctness and rendering

`axnn-vaccel/tests/hexagon_direct.rs` compares fused HTP Kerr and Dneg results
with Axiom's independent scalar integrators. On this machine both tests passed:
terminal event classifications matched and finite state values remained within
the declared relative/absolute tolerances. Each test uses 128 lanes, forcing
four 32-lane HVX partitions through the worker pool and VTCM path. A third
hardware test covers the fused Kerr frame and asserts worker mask `0xf`, peak
concurrency four, all compute/HVX/DCVS/core/bus votes, user-DMA use, compact
output length, event totals, and packed unresolved color.

Deterministic render artifacts produced by the examples:

| Scene/profile | SHA-256 |
|---|---|
| Kerr 160x90, 320 steps, fused HTP frame | `3eaebbd07b01e6f558e6e91cd2b6eca59682681a63693dd966a3fd4533623a2d` |
| Wormhole 160x90, 256 steps, four-worker HVX | `333e5ab1106d072212e73b76f471aefe487bfa49c226cdf18ed9d0ad8165033b` |
| Kerr 1080x760, one-step lower bound | `886bec4ba2d24073a520ffc5acd131f6c0b7f524d4ba91e01d52dbb8d57ff9f4` |

The generated PPMs live under Axiom's ignored `target` directory. Commands to
reproduce them are in Axiom's README.

The Kerr terminal counts match exactly (549 capture, 6,084 disk, 7,698 sky,
69 unresolved). Against CPU reference SHA-256
`dc3100646e7b264b37e3e40a0350e07368c642e034339406262d8f5bf930fbb8`, the
fused frame has mean absolute channel error 0.001227, maximum channel error 1,
53 differing channels, and no channel above error 8. With
wormhole sky rotation disabled, the vector HTP frame versus CPU reference has
mean absolute channel error 0.706, 41 of 14,400 pixels (0.285%) with any
channel error above 8, and maximum channel error 252.
The latter are trajectory-boundary divergences rather than broad image drift;
the CPU wormhole reference SHA-256 is
`f1957e2241a64c4557b1eeeff62ba764ce89f77dc9a6735fcfc7d24124ab3214`.

## Fused implementation and performance result

The kernels use explicit 32-lane V73 QFloat32 operations, four dedicated QuRT
workers, and one private VTCM slice per worker. The complete Kerr-frame opcode
moves unchanged camera-ray generation and Axiom shading/packing onto HTP. It
replaces nine uploaded FP32 planes with a four-byte control binding and twelve
returned FP32 planes with a 32-byte header plus one packed RGBA word per pixel.
Large submissions are internally tiled through ping-pong VTCM slots; V73 user
DMA overlaps packed readback staging with computation of the next tile.

The backend applies compute-class, HVX, performance-mode DCVS, turbo core, and
turbo bus votes. Hardware diagnostics prove all four workers overlap. The
one-step frame path keeps camera rays and the common no-event trace in HVX
registers, hoists camera-position geometry, reuses nearby-geometry Newton
seeds, and reruns the full trace only when a packet needs terminal state for
shading. Packet-wide predicate checks skip masked outward/disk work and shared
denominators reuse reciprocal results. Two refinements remain on the base and
discriminant paths; the nearby radius path uses one seeded refinement that
passes both scalar oracles. More aggressive one-refinement variants were
rejected after exceeding the declared oracle tolerances. The simulation still
uses its original centered finite-difference gradient and midpoint step.

The packed output remains in the provider's coherent host-mapped allocation.
Axiom's scoped frame view consumes it in place, eliminating three redundant
host vectors/copies from the measured render path while retaining an owned
result API for callers that require one.

| Profile | Total | Rate | FastRPC request | Other measured stages |
|---|---:|---:|---:|---:|
| Kerr 160x90, 320 steps, fused frame | 529.2 ms | 1.890 FPS | 529.033 ms | 0.179 ms |
| Wormhole 160x90, 256 steps | 29.1 ms | 494.8k rays/s | 24.182 ms | 4.918 ms |
| Kerr 1080x760, **one-step diagnostic**, warm 20-frame p50 | 49.748 ms | 20.102 FPS | 49.939 ms (last) | zero-copy mapped view; all rays unresolved |

The exact run used five warm-up frames and twenty measured frames. P95 was
50.063 ms, two of 20 frames exceeded 50 ms, the last request recorded 947,502
device qtimer ticks, and every frame reported worker mask `0xf`, peak
concurrency four, and flags `0x11f`. The deterministic image hash remained
unchanged. A preceding independent run measured p50 49.935 ms (20.026 FPS), so
the one-step pipeline diagnostic passes on repeat runs. It does **not** satisfy
the rendering acceptance criterion because all rays remain unresolved.

The outstanding target is the complete Kerr scene at exactly 1080x760 using
the unchanged reference 320-step integrator, with a moving camera or another
visible temporal change, sustained at 20 FPS or better. The current complete
160x90/320-step live renderer measures about 1.93 FPS, so issues #128 and Axiom
#49 remain open.

The passing runs used Windows Balanced mode with successful HTP performance,
turbo-core, and turbo-bus votes. Later diagnostic runs after the machine
reached 9% battery measured 13.82 FPS with the identical signed skel, showing
that Windows/firmware low-battery throttling can override successful HTP votes;
that power-limited state is not the acceptance configuration.

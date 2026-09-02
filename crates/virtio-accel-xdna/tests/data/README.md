# Test fixtures

## `passthrough-dmas-npu2.xdnp`

A precompiled XDNA artifact (crate-local `XDNP` container format; see
`src/artifact.rs`) wrapping the fork's DMA-only passthrough example for the npu2 device. Used by
`tests/hardware.rs::precompiled_passthrough_runs_the_full_lifecycle` to prove the full
`Accelerator` lifecycle on hardware before TOSA compilation exists.

The design (`programming_examples/basic/passthrough_dmas`) is pure shim-DMA loopback — no compute
kernel. It declares three runtime buffers: `a_in`, an unused second input `_b_unused`, and `c_out`
(n = 4096 `int32` = 16 KiB each). The DMA copies the first input to the output; the container
(format version 2) records `inputs = 2, outputs = 1`, per-slot byte sizes `[16384, 16384, 16384]`,
entry point `MLIR_AIE`.

### Reproducing it

Built with the pinned toolchain (`~/toolchains/amdxdna-hrx-v2026.08`, see
`docs/research/amdxdna-toolchain-provisioning.md`), which sets `NPU_RUNTIME=hrx` so the JIT emits
the **unfolded** HRX DDR ABI that `hrx_amdxdna_executable_create` expects:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
cd ~/toolchains/amdxdna-hrx-v2026.08/amd-npu-compiler/programming_examples/basic/passthrough_dmas
# Run once through the HRX path so the JIT caches an HRX-ABI artifact (self-verifies "PASS!").
NPU_RUNTIME=hrx python3 passthrough_dmas.py -d npu2
```

Then package the cached `final.xclbin` + `insts.bin` with `virtio_accel_xdna::artifact::encode`
(entry `MLIR_AIE`, input sizes `[16384, 16384]`, output sizes `[16384]`). The artifact is device-
and toolchain-specific; regenerate it when the pinned toolchain, the target device, or the `XDNP`
format version changes.

## Checksums

Every committed binary in this directory is pinned by its BLAKE3 hash in `BLAKE3SUMS`, verified
on every host (CI included) by `tests/fixtures.rs`. After any deliberate rebuild, regenerate
with `b3sum *.xdnp *.xbfp > BLAKE3SUMS` and let the hash change show up in review beside the
binary it covers.

## `xbfp-mxint8-matmul-8x512x8-v1.xbfp`

The flavor-1 `XBFP` experiment container (see `src/bfp_experiment.rs`): the two-input K = 512
block-scaled MATMUL design built by the issue #146 probe pipeline
(`research/bfp16ebs8/probe_compile.py xbfp`, pinned v2026.08 toolchain) and wrapped by
`bfp_experiment::encode(8, 512, 8, ...)`. Used by `tests/bfp_experiment.rs` for the on-metal
vendor-experiment suite.

#!/usr/bin/env python3
"""virtio-accel-xdna compiler helper.

Invoked as a bounded subprocess by the backend's ``load_program`` (never imported in-process,
never a Cargo dependency). It compiles one admitted TOSA operator specialization to the amdxdna
artifacts HRX consumes (``final.xclbin`` + unfolded ``insts.bin``), using the pinned IRON toolchain.

Two modes:

* ``compile <workdir>`` — read ``<workdir>/spec.json``, compile, and write ``final.xclbin``,
  ``insts.bin`` and ``result.json`` into the workdir.
* ``identity`` — print the installed toolchain identity as JSON (for the cache key).

``spec.json`` (all fields validated integers or closed enums; no guest bytes reach this process):

    {"op": "IDENTITY", "dtype": "bf16" | "i8", "elements": <n>, "device": "npu2",
     "fold_ddr_addr_offset": false}
    {"op": "CAST", "in_dtype": "fp8e4m3" | "fp8e5m2", "out_dtype": "bf16",
     "elements": <n>, "device": "npu2", "fold_ddr_addr_offset": false}
    {"op": "MATMUL", "in_dtype": "bf16", "out_dtype": "f32", "m": <M>, "k": <K>, "n": <N>,
     "device": "npu2", "fold_ddr_addr_offset": false}
    {"op": "MATMUL", "in_dtype": "fp8e4m3" | "fp8e5m2", "out_dtype": "f32", "m": <M>, "k": <K>,
     "n": <N>, "tile_m": <TM>, "tile_k": <TK>, "tile_n": <TN>, "max_dim": <D>, "device": "npu2",
     "fold_ddr_addr_offset": false}   # fused: FP8 in, BF16 promotion in L1, FP32 out
    {"op": "MATMUL", "in_dtype": "i8", "out_dtype": "i32", "m": <M>, "k": <K>, "n": <N>,
     "left_zero_point": <i8>, "right_zero_point": <i8>, "device": "npu2",
     "fold_ddr_addr_offset": false}
    {"op": "RESCALE", "in_dtype": "i32", "out_dtype": "i8", "elements": <n>,
     "multiplier": <i32>, "shift": <2..62>, "input_zero_point": 0,
     "output_zero_point": <i8>, "rounding_mode": "SINGLE_ROUND", "per_channel": false,
     "device": "npu2", "fold_ddr_addr_offset": false}
    {"op": "MAX_POOL2D", "dtype": "bf16", "layout": "NHWC", "batch": 1,
     "input_h": <H>, "input_w": <W>, "channels": <C>, "output_h": <OH>, "output_w": <OW>,
     "kernel_h": <KH>, "kernel_w": <KW>, "stride_h": <SH>, "stride_w": <SW>,
     "pad": [0, 0, 0, 0], "nan_mode": "PROPAGATE", ...}

``result.json``: ``{"schema": 2, "ok": true, "stage": "...", "entry": "MLIR_AIE",
"input_bytes": [..], "output_bytes": [..]}`` or ``{"schema": 2, "ok": false, "stage": "...",
"message": "..."}``. The byte arrays are the per-slot binding plan (exact tensor sizes the compiled
transaction stream transfers). IDENTITY binds one input and one output; MATMUL binds two inputs
(A, B) and one output (C) — the zero-points are compile-time constants, not runtime bindings;
MAX_POOL2D binds one NHWC input and one NHWC output.
CAST binds one FP8 input directly and one BF16 output directly.

The environment is pinned by the caller (cleared, with PEANO_INSTALL_DIR / AIE_XCLBINUTIL / PATH
and a private HOME/TMPDIR/NPU_CACHE_HOME); this script sets no ambient state and never dispatches.
"""

import json
import sys
import traceback
from pathlib import Path

def _fail(workdir: Path, stage: str, message: str) -> int:
    (workdir / "result.json").write_text(
        json.dumps({"schema": 2, "ok": False, "stage": stage, "message": message})
    )
    print(f"xdna_compile: {stage}: {message}", file=sys.stderr)
    return 1


def _identity() -> int:
    import importlib.metadata as md

    def version(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "unknown"

    print(
        json.dumps(
            {
                "schema": 1,
                "helper": 1,
                "mlir_aie": version("mlir_aie"),
                "llvm_aie": version("llvm-aie"),
            }
        )
    )
    return 0


def _configure_toolchain_env() -> None:
    """Pin the aiecc toolchain env from the prefix, reconstructing what `utils/env_setup.sh` sets.

    The caller runs us under a cleared environment with only the prefix and a private workdir. We
    resolve tool paths from the venv's `site-packages` (via `sysconfig`) and the prefix, add the
    IRON python package to this process's path, and set the variables aiecc's subprocesses need.
    `NPU_RUNTIME=hrx` selects the unfolded DDR ABI HRX requires; the tensor factory locates
    `libhrx.so` through `HRX_DIR`.
    """
    import os
    import sys
    import sysconfig

    site_packages = Path(sysconfig.get_paths()["purelib"])
    mlir_dir = site_packages / "mlir_aie"
    peano = site_packages / "llvm-aie"

    # `aie` lives under `mlir_aie/python`; make it importable in this process and its children.
    aie_python = mlir_dir / "python"
    sys.path.insert(0, str(aie_python))

    os.environ["MLIR_AIE_INSTALL_DIR"] = str(mlir_dir)
    if peano.is_dir():
        os.environ["PEANO_INSTALL_DIR"] = str(peano)
    os.environ["NPU_RUNTIME"] = "hrx"
    os.environ["NPU2"] = "1"

    prefix = os.environ.get("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN")
    if prefix:
        xclbinutil = Path(prefix) / "amd-npu-compiler/bld-xclbinutil/tools/hrx-xclbinutil"
        if xclbinutil.is_file():
            os.environ["AIE_XCLBINUTIL"] = str(xclbinutil)
        hrx_dir = _find_hrx_dir(Path(prefix))
        if hrx_dir is not None:
            os.environ["HRX_DIR"] = str(hrx_dir)

    def prepend(var: str, value: str) -> None:
        current = os.environ.get(var, "")
        os.environ[var] = f"{value}{':' + current if current else ''}"

    prepend("PYTHONPATH", str(aie_python))
    prepend("PATH", f"{mlir_dir / 'bin'}:{peano / 'bin'}")
    prepend("LD_LIBRARY_PATH", str(mlir_dir / "lib"))


def _find_hrx_dir(prefix: Path) -> Path | None:
    """Locate the extracted HRX release prefix (the dir holding `lib/libhrx.so`) under the prefix."""
    root = prefix / "amd-npu-compiler/third_party/.hrx-release"
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir()):
        if (candidate / "lib/libhrx.so").is_file():
            return candidate
    return None


_FP8_DECODERS = r"""
#include <cstdint>

static inline uint16_t fp8e4m3_to_bf16(uint8_t bits) {
    const uint16_t sign = static_cast<uint16_t>(bits & 0x80u) << 8;
    const uint16_t exponent = (bits >> 3) & 0x0fu;
    const uint16_t fraction = bits & 0x07u;
    if (exponent == 0) {
        static const uint16_t subnormal[8] = {
            0x0000, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60
        };
        return sign | subnormal[fraction];
    }
    if (exponent == 0x0f && fraction == 0x07) {
        return sign | 0x7fc0; // quiet BF16 NaN; TOSA permits payload canonicalization
    }
    return sign | static_cast<uint16_t>((exponent + 120) << 7) | (fraction << 4);
}

static inline uint16_t fp8e5m2_to_bf16(uint8_t bits) {
    const uint16_t sign = static_cast<uint16_t>(bits & 0x80u) << 8;
    const uint16_t exponent = (bits >> 2) & 0x1fu;
    const uint16_t fraction = bits & 0x03u;
    if (exponent == 0) {
        static const uint16_t subnormal[4] = {0x0000, 0x3780, 0x3800, 0x3840};
        return sign | subnormal[fraction];
    }
    if (exponent == 0x1f) {
        return sign | (fraction == 0 ? 0x7f80 : 0x7fc0);
    }
    return sign | static_cast<uint16_t>((exponent + 112) << 7) | (fraction << 5);
}

"""

# The standalone CAST tier's two fixed 1,024-element entry points, unchanged.
_FP8_CAST_KERNEL_SOURCE = _FP8_DECODERS + r"""extern "C" void cast_fp8e4m3_to_bf16(
    const uint8_t *__restrict input, uint16_t *__restrict output) {
#pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < 1024; ++i) {
        output[i] = fp8e4m3_to_bf16(input[i]);
    }
}

extern "C" void cast_fp8e5m2_to_bf16(
    const uint8_t *__restrict input, uint16_t *__restrict output) {
#pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < 1024; ++i) {
        output[i] = fp8e5m2_to_bf16(input[i]);
    }
}
"""


def _fp8_cast_kernel_source(symbol: str, fp8_dtype: str, length: int) -> str:
    """One sized FP8-to-BF16 entry point over the shared exact decoders.

    The fused MATMUL widens whole L1 tiles, whose length is the tile geometry rather than the
    CAST tier's transport line, so the loop bound is emitted rather than fixed.
    """
    decoder = "fp8e4m3_to_bf16" if fp8_dtype == "fp8e4m3" else "fp8e5m2_to_bf16"
    return _FP8_DECODERS + f'''
extern "C" void {symbol}(
    const uint8_t *__restrict input, uint16_t *__restrict output) {{
#pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < {length}; ++i) {{
        output[i] = {decoder}(input[i]);
    }}
}}
'''



def _build_identity(line_size: int, dtype: str):
    """Return the @iron.jit direct-DMA IDENTITY design (shim -> memtile -> shim)."""
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
    from aie.iron.device import AnyShimTile

    element_type = ml_dtypes.bfloat16 if dtype == "bf16" else np.int8

    @iron.jit
    def identity(x_in: In, y_out: Out, *, n: CompileTime[int]):
        vector_ty = np.ndarray[(n,), np.dtype[element_type]]
        line_ty = np.ndarray[(line_size,), np.dtype[element_type]]
        of_in = ObjectFifo(line_ty, name="in")
        of_out = of_in.cons().forward()

        def sequence(x, y, in_h, out_h):
            in_h.fill(x)
            out_h.drain(y, wait=True)

        rt = Runtime(
            sequence,
            [
                vector_ty,
                vector_ty,
                of_in.prod(tile=AnyShimTile),
                of_out.cons(tile=AnyShimTile),
            ],
        )
        return Program(iron.get_current_device(), rt).resolve_program()

    return identity


def _build_fp8_to_bf16(line_size: int, in_dtype: str):
    """Return an explicit FP8 storage-to-BF16 conversion design.

    The runtime DMA binds the caller's FP8 bytes directly. One AIE2P worker expands fixed 1,024
    element tiles to exact BF16 storage bits; there is no host-side or submission-time staging.
    The external kernel is compiled by Peano because it targets the AIE core ISA (a host compiler,
    including zig cc, cannot produce this device object).
    """
    import aie.iron as iron
    import numpy as np
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction

    symbol = f"cast_{in_dtype}_to_bf16"

    @iron.jit
    def fp8_to_bf16(x_in: In, y_out: Out, *, n: CompileTime[int]):
        input_ty = np.ndarray[(n,), np.dtype[np.uint8]]
        # The worker writes BF16 storage bits directly. IRON uses uint16 here only as the exact
        # two-byte transport type; no integer-to-float conversion or host-side copy is inserted.
        output_ty = np.ndarray[(n,), np.dtype[np.uint16]]
        input_line_ty = np.ndarray[(line_size,), np.dtype[np.uint8]]
        output_line_ty = np.ndarray[(line_size,), np.dtype[np.uint16]]
        converter = ExternalFunction(
            symbol,
            source_string=_FP8_CAST_KERNEL_SOURCE,
            arg_types=[input_line_ty, output_line_ty],
        )
        of_in = ObjectFifo(input_line_ty, name="fp8_in")
        of_out = ObjectFifo(output_line_ty, name="bf16_out")

        def core_fn(in_fifo, out_fifo, kernel):
            for _ in range_(n // line_size):
                source = in_fifo.acquire(1)
                destination = out_fifo.acquire(1)
                kernel(source, destination)
                in_fifo.release(1)
                out_fifo.release(1)

        worker = Worker(core_fn, [of_in.cons(), of_out.prod(), converter])

        def sequence(source, destination, in_prod, out_cons):
            in_prod.fill(source)
            out_cons.drain(destination, wait=True)

        rt = Runtime(sequence, [input_ty, output_ty, of_in.prod(), of_out.cons()])
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return fp8_to_bf16



def _build_fp8_matmul(tile_m: int, tile_k: int, tile_n: int, fp8_dtype: str):
    """Return the fused FP8 -> BF16 -> FP32 MATMUL design (`C[M,N] = A[M,K] . B[K,N]`).

    Structurally `_build_matmul`, with one difference: A and B arrive from DDR as FP8 storage
    bytes, and each L1 tile is widened to BF16 by the same exact decoders the standalone CAST tier
    uses. The BF16 operands exist only as core-local scratch, so the caller never allocates a BF16
    tensor and the promotion costs no DDR round trip.

    Numerically this is CAST-then-MATMUL with the intermediate never materialized: FP8 -> BF16 is
    exact for every encoding, and the multiply is the identical `kernels.mm` bf16 -> f32 kernel, so
    the result is bit-identical to running the two admitted tiers back to back. The graph still
    states the promotion explicitly (two TOSA CAST operators); fusing is a placement choice, not a
    relabeling of the arithmetic.

    The L2 -> L1 layout transform is expressed in elements, so it is dtype-agnostic; widening
    afterwards is elementwise and preserves the micro-tile ordering the matmul kernel expects.
    """
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
    from aie.iron import (
        Buffer,
        CompileTime,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        TaskGroup,
        Worker,
        kernels,
    )
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction

    in_ty = ml_dtypes.bfloat16
    out_ty = np.float32
    tm, tk, tn = tile_m, tile_k, tile_n

    @iron.jit
    def matmul_fp8_f32(
        input0: In,
        input1: In,
        output: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
        N: CompileTime[int],
    ):
        matmul_kernel = kernels.mm(
            dim_m=tm, dim_k=tk, dim_n=tn,
            input_dtype=in_ty, output_dtype=out_ty, vectorized=True,
        )
        r, s, t = matmul_kernel.mac_dims

        A_ty = np.ndarray[(M, K), np.dtype[np.uint8]]
        B_ty = np.ndarray[(K, N), np.dtype[np.uint8]]
        C_ty = np.ndarray[(M, N), np.dtype[out_ty]]
        a_storage_ty = np.ndarray[(tm * tk,), np.dtype[np.uint8]]
        b_storage_ty = np.ndarray[(tk * tn,), np.dtype[np.uint8]]
        a_ty = np.ndarray[(tm * tk,), np.dtype[in_ty]]
        b_ty = np.ndarray[(tk * tn,), np.dtype[in_ty]]
        c_ty = np.ndarray[(tm * tn,), np.dtype[out_ty]]

        widen_a_symbol = f"widen_a_{fp8_dtype}"
        widen_b_symbol = f"widen_b_{fp8_dtype}"
        widen_a = ExternalFunction(
            widen_a_symbol,
            source_string=_fp8_cast_kernel_source(widen_a_symbol, fp8_dtype, tm * tk),
            arg_types=[a_storage_ty, a_ty],
        )
        widen_b = ExternalFunction(
            widen_b_symbol,
            source_string=_fp8_cast_kernel_source(widen_b_symbol, fp8_dtype, tk * tn),
            arg_types=[b_storage_ty, b_ty],
        )

        fifo_a_l3l2 = ObjectFifo(a_storage_ty, name="A_L3L2")
        tap_a = TensorTiler2D.group_tiler((tm, tk), (r, s), (tm // r, tk // s))[0]
        fifo_a_l2l1 = fifo_a_l3l2.cons().forward(dims_to_stream=tap_a.transformation_dims, name="A_L2L1")

        fifo_b_l3l2 = ObjectFifo(b_storage_ty, name="B_L3L2")
        tap_b = TensorTiler2D.group_tiler((tk, tn), (s, t), (tk // s, tn // t))[0]
        fifo_b_l2l1 = fifo_b_l3l2.cons().forward(dims_to_stream=tap_b.transformation_dims, name="B_L2L1")

        fifo_c_l1l2 = ObjectFifo(c_ty, name="C_L1L2")
        tap_c = TensorAccessPattern(
            tensor_dims=(tm, tn), offset=0,
            sizes=[tm // r, r, tn // t, t], strides=[r * tn, t, r * t, 1],
        )
        fifo_c_l2l3 = fifo_c_l1l2.cons().forward(dims_to_stream=list(tap_c.transformation_dims), name="C_L2L3")

        # The promoted operands: core-local only, never a runtime binding.
        a_scratch = Buffer(a_ty, name="A_bf16_scratch")
        b_scratch = Buffer(b_ty, name="B_bf16_scratch")

        def core_fn(of_a, of_b, of_c, a_bf16, b_bf16, widen_lhs, widen_rhs, matmul):
            for _ in range_(M // tm * N // tn):
                elem_out = of_c.acquire(1)
                for i in range_(tm * tn):
                    elem_out[i] = 0
                for _ in range_(K // tk):
                    elem_in_a = of_a.acquire(1)
                    elem_in_b = of_b.acquire(1)
                    widen_lhs(elem_in_a, a_bf16)
                    widen_rhs(elem_in_b, b_bf16)
                    matmul(a_bf16, b_bf16, elem_out)
                    of_a.release(1)
                    of_b.release(1)
                of_c.release(1)

        worker = Worker(
            core_fn,
            [
                fifo_a_l2l1.cons(), fifo_b_l2l1.cons(), fifo_c_l1l2.prod(),
                a_scratch, b_scratch, widen_a, widen_b, matmul_kernel,
            ],
        )

        a_taps = TensorTiler2D.group_tiler((M, K), (tm, tk), (1, K // tk), pattern_repeat=(N // tn))
        b_tap = TensorTiler2D.group_tiler((K, N), (tk, tn), (K // tk, N // tn), tile_group_col_major=True)[0]
        c_taps = TensorTiler2D.group_tiler((M, N), (tm, tn), (1, N // tn))

        def sequence(a_src, b_src, c_dst, a_prod, b_prod, c_cons):
            for tile_row in range(M // tm):
                task_group = TaskGroup()
                a_prod.fill(a_src, tap=a_taps[tile_row], group=task_group)
                b_prod.fill(b_src, tap=b_tap, group=task_group)
                c_cons.drain(c_dst, tap=c_taps[tile_row], group=task_group, wait=True)
                task_group.finish()

        rt = Runtime(
            sequence,
            [A_ty, B_ty, C_ty, fifo_a_l3l2.prod(), fifo_b_l3l2.prod(), fifo_c_l2l3.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return matmul_fp8_f32


def _build_matmul(tile_m: int, tile_k: int, tile_n: int):
    """Return the @iron.jit BF16 -> FP32 single-core MATMUL design (`C[M,N] = A[M,K] . B[K,N]`).

    Structurally the fork's ``matrix_multiplication_single_core`` design, specialized to a bf16
    input / fp32 output kernel (the TOSA-mandated accumulator). The FP32 output is the accumulator
    type, so the compute tile is (32, 64, 32) — a multiple of the (4, 8, 8) micro-tile that keeps
    the double-buffered tiles inside the compute core's L1.
    """
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels
    from aie.iron.controlflow import range_

    in_ty = ml_dtypes.bfloat16
    out_ty = np.float32
    tm, tk, tn = tile_m, tile_k, tile_n

    @iron.jit
    def matmul_bf16_f32(
        input0: In,
        input1: In,
        output: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
        N: CompileTime[int],
    ):
        matmul_kernel = kernels.mm(
            dim_m=tm, dim_k=tk, dim_n=tn,
            input_dtype=in_ty, output_dtype=out_ty, vectorized=True,
        )
        # The DMA layout transforms are driven by the kernel's own micro-tile geometry.
        r, s, t = matmul_kernel.mac_dims

        A_ty = np.ndarray[(M, K), np.dtype[in_ty]]
        B_ty = np.ndarray[(K, N), np.dtype[in_ty]]
        C_ty = np.ndarray[(M, N), np.dtype[out_ty]]
        a_ty = np.ndarray[(tm * tk,), np.dtype[in_ty]]
        b_ty = np.ndarray[(tk * tn,), np.dtype[in_ty]]
        c_ty = np.ndarray[(tm * tn,), np.dtype[out_ty]]

        fifo_a_l3l2 = ObjectFifo(a_ty, name="A_L3L2")
        tap_a = TensorTiler2D.group_tiler((tm, tk), (r, s), (tm // r, tk // s))[0]
        fifo_a_l2l1 = fifo_a_l3l2.cons().forward(dims_to_stream=tap_a.transformation_dims, name="A_L2L1")

        fifo_b_l3l2 = ObjectFifo(b_ty, name="B_L3L2")
        tap_b = TensorTiler2D.group_tiler((tk, tn), (s, t), (tk // s, tn // t))[0]
        fifo_b_l2l1 = fifo_b_l3l2.cons().forward(dims_to_stream=tap_b.transformation_dims, name="B_L2L1")

        fifo_c_l1l2 = ObjectFifo(c_ty, name="C_L1L2")
        tap_c = TensorAccessPattern(
            tensor_dims=(tm, tn), offset=0,
            sizes=[tm // r, r, tn // t, t], strides=[r * tn, t, r * t, 1],
        )
        fifo_c_l2l3 = fifo_c_l1l2.cons().forward(dims_to_stream=list(tap_c.transformation_dims), name="C_L2L3")

        def core_fn(of_a, of_b, of_c, matmul):
            for _ in range_(M // tm * N // tn):
                elem_out = of_c.acquire(1)
                for i in range_(tm * tn):
                    elem_out[i] = 0
                for _ in range_(K // tk):
                    elem_in_a = of_a.acquire(1)
                    elem_in_b = of_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    of_a.release(1)
                    of_b.release(1)
                of_c.release(1)

        worker = Worker(core_fn, [fifo_a_l2l1.cons(), fifo_b_l2l1.cons(), fifo_c_l1l2.prod(), matmul_kernel])

        a_taps = TensorTiler2D.group_tiler((M, K), (tm, tk), (1, K // tk), pattern_repeat=(N // tn))
        b_tap = TensorTiler2D.group_tiler((K, N), (tk, tn), (K // tk, N // tn), tile_group_col_major=True)[0]
        c_taps = TensorTiler2D.group_tiler((M, N), (tm, tn), (1, N // tn))

        def sequence(a_src, b_src, c_dst, a_prod, b_prod, c_cons):
            for tile_row in range(M // tm):
                task_group = TaskGroup()
                a_prod.fill(a_src, tap=a_taps[tile_row], group=task_group)
                b_prod.fill(b_src, tap=b_tap, group=task_group)
                c_cons.drain(c_dst, tap=c_taps[tile_row], group=task_group, wait=True)
                task_group.finish()

        rt = Runtime(
            sequence,
            [A_ty, B_ty, C_ty, fifo_a_l3l2.prod(), fifo_b_l3l2.prod(), fifo_c_l2l3.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return matmul_bf16_f32


def _int8_matmul_kernel_source(
    m: int, k: int, n: int, left_zero_point: int, right_zero_point: int
) -> str:
    """Generate the exact scalar fallback for shapes off the native MMUL tiling.

    OpenVINO expresses zero-point handling as INT32 widen/subtract nodes before MatMul. IRON has no
    equivalent provider graph, so XDNA specializes the same arithmetic into the AIE core kernel.
    The caller has already bounded K so every exact dot product fits INT32. Shapes on the native
    tiling never reach this template; they take the DMA-tiled design in
    `_build_int8_matmul_tiled`, whose zero-point terms are corrected exactly after a raw MMUL pass.
    """
    # Arbitrary small shapes (including the shared 2x3x2 corpus case) cannot use a complete MMUL
    # tile. Keep this fallback scalar and explicitly disable Peano's unsupported generic vector
    # legalization; no host fallback is involved.
    return f"""
#include <cstdint>

extern "C" void matmul_i8_i32_zp(
    const int8_t *__restrict lhs,
    const int8_t *__restrict rhs,
    int32_t *__restrict output) {{
    constexpr unsigned M = {m};
    constexpr unsigned K = {k};
    constexpr unsigned N = {n};
    constexpr int32_t LEFT_ZERO_POINT = {left_zero_point};
    constexpr int32_t RIGHT_ZERO_POINT = {right_zero_point};
    for (unsigned row = 0; row < M; ++row) {{
        for (unsigned column = 0; column < N; ++column) {{
            int32_t accumulator = 0;
#pragma clang loop vectorize(disable) interleave(disable)
            for (unsigned inner = 0; inner < K; ++inner) {{
                const int32_t left = static_cast<int32_t>(lhs[row * K + inner]);
                const int32_t right = static_cast<int32_t>(rhs[inner * N + column]);
                accumulator +=
                    (left - LEFT_ZERO_POINT) * (right - RIGHT_ZERO_POINT);
            }}
            output[row * N + column] = accumulator;
        }}
    }}
}}
"""


def _int8_zp_correction_kernel_source(
    m: int, k: int, n: int, left_zero_point: int, right_zero_point: int
) -> str:
    """Generate the exact zero-point correction pass for the DMA-tiled INT8 MATMUL.

    The main pass computes the raw product `R = A . B` on the native INT8 MMUL with no widening.
    TOSA's contract expands exactly as

        C[i][j] = R[i][j] - zb * rowsum(A)[i] - za * colsum(B)[j] + K * za * zb

    so this pass derives both sums from the same tiled L1 buffers (via MMUL against a constant
    ones tile - the matrix unit is the cheapest reducer available) and applies the correction to
    the raw tile-major output in place.

    Exactness/overflow proof, for K <= 512 (`max_dim`) and INT8 values/zero points:
    |R| <= K*128*128 < 2^23; |rowsum|,|colsum| <= K*128 = 2^16; |zb*rowsum|,|za*colsum| <= 2^23;
    |K*za*zb| <= 512*2^14 = 2^23. |C| <= 2^25 < 2^31: every term and total is exact in INT32,
    and the MMUL accumulates in acc32 with the same bound. This is the same arithmetic the
    widening formulation computed, term-for-term rearranged; no rounding exists anywhere.
    """
    ones_b = ", ".join(["1"] * 64)
    ones_a = ", ".join(["1"] * 64)
    return f"""
#include <cstdint>
#include <aie_api/aie.hpp>

extern "C" void zp_correct_i8_i32(
    const int8_t *__restrict lhs_tiled,
    const int8_t *__restrict rhs_tiled,
    int32_t *__restrict output_tiled) {{
    constexpr unsigned M = {m};
    constexpr unsigned K = {k};
    constexpr unsigned N = {n};
    constexpr int32_t LEFT_ZERO_POINT = {left_zero_point};
    constexpr int32_t RIGHT_ZERO_POINT = {right_zero_point};
    // The DMA delivers A, B, and the raw output in (8, 8, 8) micro-tile-major order; these
    // constants must match the mm kernel's mac_dims on npu2 (asserted at design build time).
    constexpr unsigned R = 8, S = 8, T = 8;
    using MMUL = aie::mmul<R, S, T, int8, int8, acc32>;
    alignas(aie::vector_decl_align) static constexpr int8 ONES_B[S * T] = {{{ones_b}}};
    alignas(aie::vector_decl_align) static constexpr int8 ONES_A[R * S] = {{{ones_a}}};
    // Static, not stack: the AIE core stack is small and M, N reach 512 (4 KiB total here).
    alignas(aie::vector_decl_align) static int32_t row_sums[M];
    alignas(aie::vector_decl_align) static int32_t column_sums[N];
    alignas(aie::vector_decl_align) int32_t scratch[R * T];

    const aie::vector<int8, S * T> ones_b = aie::load_v<S * T>(ONES_B);
    const aie::vector<int8, R * S> ones_a = aie::load_v<R * S>(ONES_A);

    // rowsum(A): A_tile(4x8) . ONES(8x8) accumulated over K/S leaves rowsums in every output
    // column; read column 0.
    // The accumulators are constructed explicitly zeroed rather than default-constructed. A
    // default-constructed aie::mmul carries a "zero on first mac" flag, and this pinned Peano
    // release mis-rotates that flag's config register in the software-pipelined loop epilogue at
    // trip count exactly 2 (K/S == 2): the final mac re-zeroes the accumulator and the sum
    // collapses to the last tile. An explicit zero accumulator makes every mac's config uniform,
    // which sidesteps the rotation entirely and is correct at every trip count. Proven on metal
    // by `tosa_int8_matmul_tiled_path_matches_the_exact_oracle_on_the_npu` (16x16x16 hits the
    // trip-count-2 case).
    for (unsigned i = 0; i < M / R; ++i) {{
        MMUL accumulator(aie::zeros<acc32, MMUL::size_C>());
        for (unsigned kk = 0; kk < K / S; ++kk) {{
            accumulator.mac(
                aie::load_v<MMUL::size_A>(lhs_tiled + (i * (K / S) + kk) * R * S), ones_b);
        }}
        aie::store_v(scratch, accumulator.template to_vector<int32>());
        for (unsigned row = 0; row < R; ++row) {{
            row_sums[i * R + row] = scratch[row * T];
        }}
    }}
    // colsum(B): ONES(4x8) . B_tile(8x8) accumulated over K/S leaves colsums in every output
    // row; read row 0.
    for (unsigned j = 0; j < N / T; ++j) {{
        MMUL accumulator(aie::zeros<acc32, MMUL::size_C>());
        for (unsigned kk = 0; kk < K / S; ++kk) {{
            accumulator.mac(
                ones_a, aie::load_v<MMUL::size_B>(rhs_tiled + (kk * (N / T) + j) * S * T));
        }}
        aie::store_v(scratch, accumulator.template to_vector<int32>());
        for (unsigned column = 0; column < T; ++column) {{
            column_sums[j * T + column] = scratch[column];
        }}
    }}

    constexpr int32_t BASE = static_cast<int32_t>(K) * LEFT_ZERO_POINT * RIGHT_ZERO_POINT;
    for (unsigned i = 0; i < M / R; ++i) {{
        for (unsigned j = 0; j < N / T; ++j) {{
            int32_t *tile = output_tiled + (i * (N / T) + j) * R * T;
            for (unsigned row = 0; row < R; ++row) {{
                const int32_t row_term =
                    BASE - RIGHT_ZERO_POINT * row_sums[i * R + row];
#pragma clang loop vectorize(disable) interleave(disable)
                for (unsigned column = 0; column < T; ++column) {{
                    tile[row * T + column] +=
                        row_term - LEFT_ZERO_POINT * column_sums[j * T + column];
                }}
            }}
        }}
    }}
}}
"""


def _build_int8_matmul_tiled(
    m: int, k: int, n: int, left_zero_point: int, right_zero_point: int
):
    """Return the DMA-tiled exact one-core INT8 -> INT32 MATMUL design.

    Structurally the BF16 `_build_matmul` (itself the fork's single-core design): the DMA layout
    transforms deliver A, B, and C in the MMUL's micro-tile order, so no core cycle is spent on
    packing or widening. The raw product runs on the fork's vectorized i8 -> i32 `mm` kernel
    against the caller's raw INT8 bytes, and one correction pass applies the zero-point terms
    exactly (see `_int8_zp_correction_kernel_source` for the identity and the overflow proof).
    The admitted envelope keeps the whole tensor set inside one core's L1, so the compute tile is
    the whole tensor and each operand streams exactly once.
    """
    import aie.iron as iron
    import numpy as np
    from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels
    from aie.iron.kernel import ExternalFunction

    @iron.jit
    def matmul_int8_int32(lhs_in: In, rhs_in: In, output_out: Out):
        matmul_kernel = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n,
            input_dtype=np.int8, output_dtype=np.int32, vectorized=True,
        )
        zero_kernel = matmul_kernel.zero
        r, s, t = matmul_kernel.mac_dims
        # The correction kernel hardcodes npu2's (8, 8, 8) INT8 micro-tile; a toolchain that
        # changes the mm kernel's geometry must fail the compile, not corrupt the layout contract.
        if (r, s, t) != (8, 8, 8):
            raise RuntimeError(f"unexpected i8 mm micro-tile: {(r, s, t)}")
        correction_kernel = ExternalFunction(
            "zp_correct_i8_i32",
            source_string=_int8_zp_correction_kernel_source(
                m, k, n, left_zero_point, right_zero_point
            ),
            arg_types=[
                np.ndarray[(m * k,), np.dtype[np.int8]],
                np.ndarray[(k * n,), np.dtype[np.int8]],
                np.ndarray[(m * n,), np.dtype[np.int32]],
            ],
        )

        A_ty = np.ndarray[(m, k), np.dtype[np.int8]]
        B_ty = np.ndarray[(k, n), np.dtype[np.int8]]
        C_ty = np.ndarray[(m, n), np.dtype[np.int32]]
        a_ty = np.ndarray[(m * k,), np.dtype[np.int8]]
        b_ty = np.ndarray[(k * n,), np.dtype[np.int8]]
        c_ty = np.ndarray[(m * n,), np.dtype[np.int32]]

        fifo_a_l3l2 = ObjectFifo(a_ty, name="A_L3L2")
        tap_a = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
        fifo_a_l2l1 = fifo_a_l3l2.cons().forward(
            dims_to_stream=tap_a.transformation_dims, name="A_L2L1"
        )

        fifo_b_l3l2 = ObjectFifo(b_ty, name="B_L3L2")
        tap_b = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
        fifo_b_l2l1 = fifo_b_l3l2.cons().forward(
            dims_to_stream=tap_b.transformation_dims, name="B_L2L1"
        )

        fifo_c_l1l2 = ObjectFifo(c_ty, name="C_L1L2")
        tap_c = TensorAccessPattern(
            tensor_dims=(m, n), offset=0,
            sizes=[m // r, r, n // t, t], strides=[r * n, t, r * t, 1],
        )
        fifo_c_l2l3 = fifo_c_l1l2.cons().forward(
            dims_to_stream=list(tap_c.transformation_dims), name="C_L2L3"
        )

        def core_fn(of_a, of_b, of_c, zero, matmul, correct):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)
            zero(elem_c)
            matmul(elem_a, elem_b, elem_c)
            correct(elem_a, elem_b, elem_c)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

        worker = Worker(
            core_fn,
            [
                fifo_a_l2l1.cons(),
                fifo_b_l2l1.cons(),
                fifo_c_l1l2.prod(),
                zero_kernel,
                matmul_kernel,
                correction_kernel,
            ],
        )

        a_tap = TensorTiler2D.group_tiler((m, k), (m, k), (1, 1))[0]
        b_tap = TensorTiler2D.group_tiler((k, n), (k, n), (1, 1))[0]
        c_tap = TensorTiler2D.group_tiler((m, n), (m, n), (1, 1))[0]

        def sequence(lhs, rhs, output, a_prod, b_prod, c_cons):
            task_group = TaskGroup()
            a_prod.fill(lhs, tap=a_tap, group=task_group)
            b_prod.fill(rhs, tap=b_tap, group=task_group)
            c_cons.drain(output, tap=c_tap, group=task_group, wait=True)
            task_group.finish()

        rt = Runtime(
            sequence,
            [A_ty, B_ty, C_ty, fifo_a_l3l2.prod(), fifo_b_l3l2.prod(), fifo_c_l2l3.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return matmul_int8_int32


def _build_int8_matmul(
    m: int, k: int, n: int, left_zero_point: int, right_zero_point: int
):
    """Return the exact one-core scalar INT8 -> INT32 MATMUL design (off-tiling shapes).

    Complete bounded tensors are direct-DMA'd into depth-two object FIFOs. The worker invokes one
    Peano-compiled AIE kernel; no host staging, host arithmetic, or floating-point conversion occurs.
    Shapes on the native (4, 8, 8) MMUL tiling take `_build_int8_matmul_tiled` instead.
    """
    import aie.iron as iron
    import numpy as np
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction

    lhs_transport_elements = (m * k + 3) & ~3
    rhs_transport_elements = (k * n + 3) & ~3

    @iron.jit
    def matmul_int8_int32(lhs_in: In, rhs_in: In, output_out: Out):
        # AIE DMA lengths are 32-bit granular. Padding is part of the declared direct-binding ABI,
        # and the kernel's M/K/N loops never inspect it.
        lhs_ty = np.ndarray[(lhs_transport_elements,), np.dtype[np.int8]]
        rhs_ty = np.ndarray[(rhs_transport_elements,), np.dtype[np.int8]]
        output_ty = np.ndarray[(m * n,), np.dtype[np.int32]]
        kernel = ExternalFunction(
            "matmul_i8_i32_zp",
            source_string=_int8_matmul_kernel_source(
                m, k, n, left_zero_point, right_zero_point
            ),
            arg_types=[lhs_ty, rhs_ty, output_ty],
        )
        lhs_fifo = ObjectFifo(lhs_ty, name="int8_lhs")
        rhs_fifo = ObjectFifo(rhs_ty, name="int8_rhs")
        output_fifo = ObjectFifo(output_ty, name="int32_output")

        def core_fn(lhs_source, rhs_source, output_destination, matmul):
            lhs = lhs_source.acquire(1)
            rhs = rhs_source.acquire(1)
            output = output_destination.acquire(1)
            matmul(lhs, rhs, output)
            lhs_source.release(1)
            rhs_source.release(1)
            output_destination.release(1)

        worker = Worker(
            core_fn,
            [lhs_fifo.cons(), rhs_fifo.cons(), output_fifo.prod(), kernel],
        )

        def sequence(lhs, rhs, output, lhs_prod, rhs_prod, output_cons):
            lhs_prod.fill(lhs)
            rhs_prod.fill(rhs)
            output_cons.drain(output, wait=True)

        rt = Runtime(
            sequence,
            [
                lhs_ty,
                rhs_ty,
                output_ty,
                lhs_fifo.prod(),
                rhs_fifo.prod(),
                output_fifo.cons(),
            ],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return matmul_int8_int32


def _rescale_kernel_source(
    elements: int, multiplier: int, shift: int, output_zero_point: int
) -> str:
    """Generate exact TOSA scale32 SINGLE_ROUND arithmetic for one fixed tensor."""
    return f"""
#include <cstdint>

extern "C" void rescale_i32_i8(
    const int32_t *__restrict input,
    int8_t *__restrict output) {{
    constexpr unsigned ELEMENTS = {elements};
    constexpr unsigned OUTPUT_TRANSPORT_ELEMENTS = (ELEMENTS + 3) & ~3U;
    constexpr int64_t MULTIPLIER = {multiplier};
    constexpr unsigned SHIFT = {shift};
    constexpr int64_t OUTPUT_ZERO_POINT = {output_zero_point};
    constexpr int64_t ROUND = int64_t{{1}} << (SHIFT - 1);
    constexpr int64_t DENOMINATOR = int64_t{{1}} << SHIFT;

#pragma clang loop vectorize(disable) interleave(disable)
    for (unsigned index = 0; index < ELEMENTS; ++index) {{
        const int64_t rounded =
            static_cast<int64_t>(input[index]) * MULTIPLIER + ROUND;
        // Spell out arithmetic right shift so negative rounding does not depend on a C++
        // implementation choice. floor(rounded / 2^SHIFT) equals -ceil(-rounded / 2^SHIFT).
        const int64_t scaled = rounded >= 0
            ? rounded >> SHIFT
            : -((-rounded + DENOMINATOR - 1) >> SHIFT);
        const int64_t shifted = scaled + OUTPUT_ZERO_POINT;
        output[index] = static_cast<int8_t>(
            shifted < -128 ? -128 : (shifted > 127 ? 127 : shifted));
    }}
    // The direct-binding slot is word-rounded for DMA. Clear its non-tensor tail so no prior local
    // memory contents become observable through the explicit padding bytes.
    for (unsigned index = ELEMENTS; index < OUTPUT_TRANSPORT_ELEMENTS; ++index) {{
        output[index] = 0;
    }}
}}
"""


def _build_int32_to_int8_rescale(
    elements: int, multiplier: int, shift: int, output_zero_point: int
):
    """Return one direct-bound exact INT32-to-INT8 RESCALE worker."""
    import aie.iron as iron
    import numpy as np
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction

    output_transport_elements = (elements + 3) & ~3

    @iron.jit
    def rescale_int32_int8(input_in: In, output_out: Out):
        input_ty = np.ndarray[(elements,), np.dtype[np.int32]]
        # AIE DMA is word-granular. The slot exposes padding explicitly; the kernel writes the
        # graph-visible prefix and clears the at-most-three-byte transport tail.
        output_ty = np.ndarray[(output_transport_elements,), np.dtype[np.int8]]
        kernel = ExternalFunction(
            "rescale_i32_i8",
            source_string=_rescale_kernel_source(
                elements, multiplier, shift, output_zero_point
            ),
            arg_types=[input_ty, output_ty],
        )
        input_fifo = ObjectFifo(input_ty, name="rescale_input")
        output_fifo = ObjectFifo(output_ty, name="rescale_output")

        def core_fn(input_source, output_destination, rescale):
            input_value = input_source.acquire(1)
            output_value = output_destination.acquire(1)
            rescale(input_value, output_value)
            input_source.release(1)
            output_destination.release(1)

        worker = Worker(core_fn, [input_fifo.cons(), output_fifo.prod(), kernel])

        def sequence(input_value, output_value, input_prod, output_cons):
            input_prod.fill(input_value)
            output_cons.drain(output_value, wait=True)

        rt = Runtime(
            sequence,
            [input_ty, output_ty, input_fifo.prod(), output_fifo.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return rescale_int32_int8


def _build_max_pool2d():
    """Return a batch-1 BF16 NHWC MAX_POOL2D design with propagating NaNs.

    The complete bounded tensors are double-buffered in one compute core. The worker walks output
    positions and channels with compact SCF loops, while the at-most-8x8 window is statically
    unrolled. ``arith.maximumf`` implements the TOSA propagating-NaN maximum and preserves the
    required floating-point max behavior without converting BF16 storage to another dtype.
    """
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.extras.dialects import arith
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_

    @iron.jit
    def max_pool2d_bf16(
        input_tensor: In,
        output_tensor: Out,
        *,
        input_h: CompileTime[int],
        input_w: CompileTime[int],
        channels: CompileTime[int],
        output_h: CompileTime[int],
        output_w: CompileTime[int],
        kernel_h: CompileTime[int],
        kernel_w: CompileTime[int],
        stride_h: CompileTime[int],
        stride_w: CompileTime[int],
    ):
        input_elements = input_h * input_w * channels
        output_elements = output_h * output_w * channels
        input_ty = np.ndarray[(input_elements,), np.dtype[ml_dtypes.bfloat16]]
        output_ty = np.ndarray[(output_elements,), np.dtype[ml_dtypes.bfloat16]]
        of_in = ObjectFifo(input_ty, name="pool_in")
        of_out = ObjectFifo(output_ty, name="pool_out")

        def core_fn(in_fifo, out_fifo):
            source = in_fifo.acquire(1)
            destination = out_fifo.acquire(1)
            for oy in range_(output_h):
                for ox in range_(output_w):
                    for channel in range_(channels):
                        input_y = oy * stride_h
                        input_x = ox * stride_w
                        first = (input_y * input_w + input_x) * channels + channel
                        maximum = source[first]
                        for ky in range(kernel_h):
                            for kx in range(kernel_w):
                                if ky == 0 and kx == 0:
                                    continue
                                index = (
                                    ((input_y + ky) * input_w + input_x + kx) * channels
                                    + channel
                                )
                                maximum = arith.maximumf(maximum, source[index])
                        output_index = (oy * output_w + ox) * channels + channel
                        destination[output_index] = maximum
            in_fifo.release(1)
            out_fifo.release(1)

        worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

        def sequence(source, destination, in_prod, out_cons):
            in_prod.fill(source)
            out_cons.drain(destination, wait=True)

        rt = Runtime(sequence, [input_ty, output_ty, of_in.prod(), of_out.cons()])
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return max_pool2d_bf16


def _compile(workdir: Path) -> int:
    try:
        spec = json.loads((workdir / "spec.json").read_text())
    except (OSError, ValueError) as error:
        return _fail(workdir, "spec-rejected", f"unreadable spec.json: {error}")

    op = spec.get("op")
    device = spec.get("device")
    if device != "npu2":
        return _fail(workdir, "spec-rejected", f"unsupported device: {device}")

    # Re-validate the spec here (defence in depth: `admit` already checked it) and select the
    # design plus its runtime binding counts. `build` returns a specialized, ready-to-compile design.
    if op == "IDENTITY":
        dtype = spec.get("dtype")
        elements = spec.get("elements")
        line_size = spec.get("line_size")
        if dtype not in ("bf16", "i8"):
            return _fail(workdir, "spec-rejected", f"unsupported dtype: {dtype}")
        if not isinstance(line_size, int) or line_size <= 0:
            return _fail(workdir, "spec-rejected", f"invalid line_size: {line_size}")
        if not isinstance(elements, int) or elements <= 0 or elements % line_size != 0:
            return _fail(
                workdir, "spec-rejected", f"elements must be a positive multiple of {line_size}"
            )
        element_bytes = 2 if dtype == "bf16" else 1
        if dtype == "i8":
            max_line_size = spec.get("max_line_size")
            if max_line_size != 1024 or line_size > max_line_size:
                return _fail(workdir, "spec-rejected", "unsupported INT8 IDENTITY line size")
        input_bytes, output_bytes = [elements * element_bytes], [elements * element_bytes]

        def build():
            return _build_identity(line_size, dtype).specialize(n=elements)
    elif op == "CAST":
        in_dtype = spec.get("in_dtype")
        out_dtype = spec.get("out_dtype")
        elements = spec.get("elements")
        line_size = spec.get("line_size")
        if in_dtype not in ("fp8e4m3", "fp8e5m2") or out_dtype != "bf16":
            return _fail(
                workdir, "spec-rejected", f"unsupported dtype pair: {in_dtype}->{out_dtype}"
            )
        if line_size != 1024:
            return _fail(workdir, "spec-rejected", f"unsupported CAST line_size: {line_size}")
        if not isinstance(elements, int) or elements <= 0 or elements % line_size != 0:
            return _fail(
                workdir, "spec-rejected", f"elements must be a positive multiple of {line_size}"
            )
        input_bytes, output_bytes = [elements], [elements * 2]

        def build():
            return _build_fp8_to_bf16(line_size, in_dtype).specialize(n=elements)
    elif op == "MATMUL":
        in_dtype = spec.get("in_dtype")
        out_dtype = spec.get("out_dtype")
        m, k, n = spec.get("m"), spec.get("k"), spec.get("n")
        max_dim = spec.get("max_dim")
        if (in_dtype, out_dtype) not in (
            ("bf16", "f32"),
            ("i8", "i32"),
            ("fp8e4m3", "f32"),
            ("fp8e5m2", "f32"),
        ):
            return _fail(
                workdir, "spec-rejected", f"unsupported dtype pair: {in_dtype}->{out_dtype}"
            )
        if not isinstance(max_dim, int) or max_dim <= 0:
            return _fail(workdir, "spec-rejected", f"invalid max_dim: {max_dim}")
        for name, dim in (("m", m), ("k", k), ("n", n)):
            if not isinstance(dim, int) or dim <= 0 or dim > max_dim:
                return _fail(
                    workdir, "spec-rejected", f"{name}={dim} must be positive and <= {max_dim}"
                )
        if in_dtype in ("fp8e4m3", "fp8e5m2"):
            tile_m, tile_k, tile_n = (
                spec.get("tile_m"),
                spec.get("tile_k"),
                spec.get("tile_n"),
            )
            for name, dim, tile in (("m", m, tile_m), ("k", k, tile_k), ("n", n, tile_n)):
                if not isinstance(tile, int) or tile <= 0:
                    return _fail(workdir, "spec-rejected", f"invalid tile_{name}: {tile}")
                if dim % tile != 0:
                    return _fail(
                        workdir,
                        "spec-rejected",
                        f"{name}={dim} must be a multiple of {tile}",
                    )
            # FP8 operands bind one byte per element; only the FP32 result reaches DDR at full
            # width. The promoted BF16 operands never leave the compute core.
            input_bytes, output_bytes = [m * k, k * n], [m * n * 4]

            def build():
                return _build_fp8_matmul(tile_m, tile_k, tile_n, in_dtype).specialize(M=m, K=k, N=n)

        elif in_dtype == "bf16":
            tile_m, tile_k, tile_n = (
                spec.get("tile_m"),
                spec.get("tile_k"),
                spec.get("tile_n"),
            )
            for name, dim, tile in (("m", m, tile_m), ("k", k, tile_k), ("n", n, tile_n)):
                if not isinstance(tile, int) or tile <= 0:
                    return _fail(workdir, "spec-rejected", f"invalid tile_{name}: {tile}")
                if dim % tile != 0:
                    return _fail(
                        workdir,
                        "spec-rejected",
                        f"{name}={dim} must be a multiple of {tile}",
                    )
            input_bytes, output_bytes = [m * k * 2, k * n * 2], [m * n * 4]

            def build():
                return _build_matmul(tile_m, tile_k, tile_n).specialize(M=m, K=k, N=n)
        else:
            left_zero_point = spec.get("left_zero_point")
            right_zero_point = spec.get("right_zero_point")
            max_total_bytes = spec.get("max_total_bytes")
            if (
                not isinstance(left_zero_point, int)
                or not -128 <= left_zero_point <= 127
                or not isinstance(right_zero_point, int)
                or not -128 <= right_zero_point <= 127
            ):
                return _fail(workdir, "spec-rejected", "INT8 zero point is out of range")
            if not isinstance(max_total_bytes, int) or max_total_bytes <= 0:
                return _fail(workdir, "spec-rejected", "invalid INT8 local-memory bound")
            lhs_bytes = (m * k + 3) & ~3
            rhs_bytes = (k * n + 3) & ~3
            total_bytes = lhs_bytes + rhs_bytes + m * n * 4
            if total_bytes > max_total_bytes:
                return _fail(workdir, "spec-rejected", "INT8 MATMUL local-memory bound exceeded")
            input_bytes, output_bytes = [lhs_bytes, rhs_bytes], [m * n * 4]

            # The mm kernel's native INT8 micro-tile on npu2 is (8, 8, 8), and its vectorized
            # variant additionally unrolls 2x2 micro-tiles (mm.cc asserts m % 2r and n % 2t).
            # K must also span at least two tiles: a single-tile K dimension degenerates the DMA
            # layout transform into a zero step size, which aie.dma_bd rejects. Shapes on that
            # grid take the DMA-tiled raw-MMUL design with an exact zero-point correction pass.
            # Everything else stays on the scalar whole-tensor design (including the shared 2x3x2
            # corpus case). Both compute the identical TOSA integer contract.
            if m % 16 == 0 and k % 8 == 0 and k >= 16 and n % 16 == 0:

                def build():
                    return _build_int8_matmul_tiled(
                        m, k, n, left_zero_point, right_zero_point
                    ).specialize()

            else:

                def build():
                    return _build_int8_matmul(
                        m, k, n, left_zero_point, right_zero_point
                    ).specialize()
    elif op == "RESCALE":
        in_dtype = spec.get("in_dtype")
        out_dtype = spec.get("out_dtype")
        elements = spec.get("elements")
        multiplier = spec.get("multiplier")
        shift = spec.get("shift")
        input_zero_point = spec.get("input_zero_point")
        output_zero_point = spec.get("output_zero_point")
        max_total_bytes = spec.get("max_total_bytes")
        if (
            in_dtype != "i32"
            or out_dtype != "i8"
            or spec.get("rounding_mode") != "SINGLE_ROUND"
            or spec.get("per_channel") is not False
            or input_zero_point != 0
        ):
            return _fail(workdir, "spec-rejected", "unsupported RESCALE semantic envelope")
        if not isinstance(elements, int) or elements <= 0:
            return _fail(workdir, "spec-rejected", "invalid RESCALE element count")
        if not isinstance(multiplier, int) or not 0 <= multiplier <= 0x7FFFFFFF:
            return _fail(workdir, "spec-rejected", "invalid RESCALE multiplier")
        if not isinstance(shift, int) or not 2 <= shift <= 62:
            return _fail(workdir, "spec-rejected", "invalid RESCALE shift")
        if not isinstance(output_zero_point, int) or not -128 <= output_zero_point <= 127:
            return _fail(workdir, "spec-rejected", "invalid RESCALE output zero point")
        if not isinstance(max_total_bytes, int) or max_total_bytes <= 0:
            return _fail(workdir, "spec-rejected", "invalid RESCALE memory bound")
        output_bytes = (elements + 3) & ~3
        if elements * 4 + output_bytes > max_total_bytes:
            return _fail(workdir, "spec-rejected", "RESCALE local-memory bound exceeded")
        input_bytes = [elements * 4]
        output_bytes = [output_bytes]

        def build():
            return _build_int32_to_int8_rescale(
                elements, multiplier, shift, output_zero_point
            ).specialize()
    elif op == "MAX_POOL2D":
        dtype = spec.get("dtype")
        layout = spec.get("layout")
        batch = spec.get("batch")
        input_h, input_w = spec.get("input_h"), spec.get("input_w")
        channels = spec.get("channels")
        output_h, output_w = spec.get("output_h"), spec.get("output_w")
        kernel_h, kernel_w = spec.get("kernel_h"), spec.get("kernel_w")
        stride_h, stride_w = spec.get("stride_h"), spec.get("stride_w")
        max_kernel = spec.get("max_kernel")
        max_stride = spec.get("max_stride")
        max_total_elements = spec.get("max_total_elements")
        if (
            dtype != "bf16"
            or layout != "NHWC"
            or batch != 1
            or spec.get("pad") != [0, 0, 0, 0]
            or spec.get("nan_mode") != "PROPAGATE"
        ):
            return _fail(workdir, "spec-rejected", "unsupported MAX_POOL2D semantic envelope")
        dimensions = {
            "input_h": input_h,
            "input_w": input_w,
            "channels": channels,
            "output_h": output_h,
            "output_w": output_w,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or value <= 0:
                return _fail(workdir, "spec-rejected", f"invalid {name}: {value}")
        for name, value in (
            ("max_kernel", max_kernel),
            ("max_stride", max_stride),
            ("max_total_elements", max_total_elements),
        ):
            if not isinstance(value, int) or value <= 0:
                return _fail(workdir, "spec-rejected", f"invalid {name}: {value}")
        if kernel_h > max_kernel or kernel_w > max_kernel:
            return _fail(workdir, "spec-rejected", "MAX_POOL2D kernel exceeds admission bound")
        if stride_h > max_stride or stride_w > max_stride:
            return _fail(workdir, "spec-rejected", "MAX_POOL2D stride exceeds admission bound")
        if kernel_h > input_h or kernel_w > input_w:
            return _fail(workdir, "spec-rejected", "MAX_POOL2D kernel exceeds input")
        expected_h = (input_h - kernel_h) // stride_h + 1
        expected_w = (input_w - kernel_w) // stride_w + 1
        if output_h != expected_h or output_w != expected_w:
            return _fail(workdir, "spec-rejected", "MAX_POOL2D output shape is inconsistent")
        input_elements = input_h * input_w * channels
        output_elements = output_h * output_w * channels
        if input_elements + output_elements > max_total_elements:
            return _fail(workdir, "spec-rejected", "MAX_POOL2D local-memory bound exceeded")
        input_bytes, output_bytes = [input_elements * 2], [output_elements * 2]

        def build():
            return _build_max_pool2d().specialize(
                input_h=input_h,
                input_w=input_w,
                channels=channels,
                output_h=output_h,
                output_w=output_w,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride_h,
                stride_w=stride_w,
            )
    else:
        return _fail(workdir, "spec-rejected", f"unsupported op: {op}")

    try:
        _configure_toolchain_env()
        import aie.iron as iron
        from aie.iron.device import from_name

        iron.set_current_device(from_name("npu2"))
    except Exception as error:  # noqa: BLE001 - report any toolchain import/setup failure
        return _fail(workdir, "template", f"toolchain setup failed: {error}")

    try:
        build().compile(
            xclbin_path=str(workdir / "final.xclbin"),
            inst_path=str(workdir / "insts.bin"),
        )
    except Exception as error:  # noqa: BLE001 - aiecc/Peano failures are reported as a stage
        return _fail(workdir, "compile", f"{error}\n{traceback.format_exc()}")

    if not (workdir / "final.xclbin").is_file() or not (workdir / "insts.bin").is_file():
        return _fail(workdir, "package", "compiler produced no artifacts")

    (workdir / "result.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "ok": True,
                "stage": "done",
                "entry": "MLIR_AIE",
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
            }
        )
    )
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 1 and args[0] == "identity":
        return _identity()
    if len(args) == 2 and args[0] == "compile":
        return _compile(Path(args[1]))
    print("usage: xdna_compile.py (identity | compile <workdir>)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

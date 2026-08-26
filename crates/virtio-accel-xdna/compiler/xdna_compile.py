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

    {"op": "IDENTITY", "dtype": "bf16", "elements": <n>, "device": "npu2",
     "fold_ddr_addr_offset": false}
    {"op": "CAST", "in_dtype": "fp8e4m3" | "fp8e5m2", "out_dtype": "bf16",
     "elements": <n>, "device": "npu2", "fold_ddr_addr_offset": false}
    {"op": "MATMUL", "in_dtype": "bf16", "out_dtype": "f32", "m": <M>, "k": <K>, "n": <N>,
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


_FP8_CAST_KERNEL_SOURCE = r"""
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

extern "C" void cast_fp8e4m3_to_bf16(
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


def _build_identity(line_size: int):
    """Return the @iron.jit IDENTITY design (a bf16 DMA copy: shim -> memtile -> shim)."""
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
    from aie.iron.device import AnyShimTile

    @iron.jit
    def identity_bf16(x_in: In, y_out: Out, *, n: CompileTime[int]):
        vector_ty = np.ndarray[(n,), np.dtype[ml_dtypes.bfloat16]]
        line_ty = np.ndarray[(line_size,), np.dtype[ml_dtypes.bfloat16]]
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

    return identity_bf16


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
        if dtype != "bf16":
            return _fail(workdir, "spec-rejected", f"unsupported dtype: {dtype}")
        if not isinstance(line_size, int) or line_size <= 0:
            return _fail(workdir, "spec-rejected", f"invalid line_size: {line_size}")
        if not isinstance(elements, int) or elements <= 0 or elements % line_size != 0:
            return _fail(
                workdir, "spec-rejected", f"elements must be a positive multiple of {line_size}"
            )
        # The binding plan: exact per-slot byte sizes (bf16 = 2 B/element).
        input_bytes, output_bytes = [elements * 2], [elements * 2]

        def build():
            return _build_identity(line_size).specialize(n=elements)
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
        tile_m, tile_k, tile_n = spec.get("tile_m"), spec.get("tile_k"), spec.get("tile_n")
        max_dim = spec.get("max_dim")
        if in_dtype != "bf16" or out_dtype != "f32":
            return _fail(
                workdir, "spec-rejected", f"unsupported dtype pair: {in_dtype}->{out_dtype}"
            )
        if not isinstance(max_dim, int) or max_dim <= 0:
            return _fail(workdir, "spec-rejected", f"invalid max_dim: {max_dim}")
        for name, dim, tile in (("m", m, tile_m), ("k", k, tile_k), ("n", n, tile_n)):
            if not isinstance(tile, int) or tile <= 0:
                return _fail(workdir, "spec-rejected", f"invalid tile_{name}: {tile}")
            if not isinstance(dim, int) or dim <= 0 or dim % tile != 0 or dim > max_dim:
                return _fail(
                    workdir, "spec-rejected", f"{name}={dim} must be a positive multiple of {tile} <= {max_dim}"
                )
        # The binding plan: A[M,K] and B[K,N] are bf16 (2 B), C[M,N] is the fp32 accumulator (4 B).
        input_bytes, output_bytes = [m * k * 2, k * n * 2], [m * n * 4]

        def build():
            return _build_matmul(tile_m, tile_k, tile_n).specialize(M=m, K=k, N=n)
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

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
    {"op": "MATMUL", "in_dtype": "bf16", "out_dtype": "f32", "m": <M>, "k": <K>, "n": <N>,
     "device": "npu2", "fold_ddr_addr_offset": false}

``result.json``: ``{"schema": 1, "ok": true, "stage": "...", "entry": "MLIR_AIE",
"inputs": <i>, "outputs": <o>}`` or ``{"schema": 1, "ok": false, "stage": "...", "message": "..."}``.
IDENTITY binds one input and one output; MATMUL binds two inputs (A, B) and one output (C) — the
zero-points are compile-time constants, not runtime bindings.

The environment is pinned by the caller (cleared, with PEANO_INSTALL_DIR / AIE_XCLBINUTIL / PATH
and a private HOME/TMPDIR/NPU_CACHE_HOME); this script sets no ambient state and never dispatches.
"""

import json
import sys
import traceback
from pathlib import Path

LINE_SIZE = 1024

# The one tested MATMUL compute tile (m, k, n), matching `MATMUL_TILE_{M,K,N}` in `src/lower.rs`.
# Every admitted dimension is a positive multiple of the matching tile. The FP32 output tile is
# 4 B/element, so the tile is kept small enough that the double-buffered C tile plus the A/B tiles
# fit the AIE2P compute core's ~64 KiB L1.
MATMUL_TILE_M, MATMUL_TILE_K, MATMUL_TILE_N = 32, 64, 32
MATMUL_MAX_DIM = 512


def _fail(workdir: Path, stage: str, message: str) -> int:
    (workdir / "result.json").write_text(
        json.dumps({"schema": 1, "ok": False, "stage": stage, "message": message})
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


def _build_identity(elements: int):
    """Return the @iron.jit IDENTITY design (a bf16 DMA copy: shim -> memtile -> shim)."""
    import aie.iron as iron
    import ml_dtypes
    import numpy as np
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
    from aie.iron.device import AnyShimTile

    @iron.jit
    def identity_bf16(x_in: In, y_out: Out, *, n: CompileTime[int] = elements):
        vector_ty = np.ndarray[(n,), np.dtype[ml_dtypes.bfloat16]]
        line_ty = np.ndarray[(LINE_SIZE,), np.dtype[ml_dtypes.bfloat16]]
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


def _build_matmul(m: int, k: int, n: int):
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
    tm, tk, tn = MATMUL_TILE_M, MATMUL_TILE_K, MATMUL_TILE_N

    @iron.jit
    def matmul_bf16_f32(
        input0: In,
        input1: In,
        output: Out,
        *,
        M: CompileTime[int] = m,
        K: CompileTime[int] = k,
        N: CompileTime[int] = n,
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
        if dtype != "bf16":
            return _fail(workdir, "spec-rejected", f"unsupported dtype: {dtype}")
        if not isinstance(elements, int) or elements <= 0 or elements % LINE_SIZE != 0:
            return _fail(
                workdir, "spec-rejected", f"elements must be a positive multiple of {LINE_SIZE}"
            )
        inputs, outputs = 1, 1

        def build():
            return _build_identity(elements).specialize(n=elements)
    elif op == "MATMUL":
        in_dtype = spec.get("in_dtype")
        out_dtype = spec.get("out_dtype")
        m, k, n = spec.get("m"), spec.get("k"), spec.get("n")
        if in_dtype != "bf16" or out_dtype != "f32":
            return _fail(
                workdir, "spec-rejected", f"unsupported dtype pair: {in_dtype}->{out_dtype}"
            )
        for name, dim, tile in (("m", m, MATMUL_TILE_M), ("k", k, MATMUL_TILE_K), ("n", n, MATMUL_TILE_N)):
            if not isinstance(dim, int) or dim <= 0 or dim % tile != 0 or dim > MATMUL_MAX_DIM:
                return _fail(
                    workdir, "spec-rejected", f"{name}={dim} must be a positive multiple of {tile} <= {MATMUL_MAX_DIM}"
                )
        inputs, outputs = 2, 1

        def build():
            return _build_matmul(m, k, n).specialize(M=m, K=k, N=n)
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
                "schema": 1,
                "ok": True,
                "stage": "done",
                "entry": "MLIR_AIE",
                "inputs": inputs,
                "outputs": outputs,
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

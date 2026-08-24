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

``result.json``: ``{"schema": 1, "ok": true, "stage": "...", "entry": "MLIR_AIE",
"inputs": 1, "outputs": 1}`` or ``{"schema": 1, "ok": false, "stage": "...", "message": "..."}``.

The environment is pinned by the caller (cleared, with PEANO_INSTALL_DIR / AIE_XCLBINUTIL / PATH
and a private HOME/TMPDIR/NPU_CACHE_HOME); this script sets no ambient state and never dispatches.
"""

import json
import sys
import traceback
from pathlib import Path

LINE_SIZE = 1024


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


def _compile(workdir: Path) -> int:
    try:
        spec = json.loads((workdir / "spec.json").read_text())
    except (OSError, ValueError) as error:
        return _fail(workdir, "spec-rejected", f"unreadable spec.json: {error}")

    op = spec.get("op")
    dtype = spec.get("dtype")
    elements = spec.get("elements")
    device = spec.get("device")
    if op != "IDENTITY" or dtype != "bf16":
        return _fail(workdir, "spec-rejected", f"unsupported op/dtype: {op}/{dtype}")
    if not isinstance(elements, int) or elements <= 0 or elements % LINE_SIZE != 0:
        return _fail(
            workdir, "spec-rejected", f"elements must be a positive multiple of {LINE_SIZE}"
        )
    if device != "npu2":
        return _fail(workdir, "spec-rejected", f"unsupported device: {device}")

    try:
        _configure_toolchain_env()
        import aie.iron as iron
        from aie.iron.device import from_name

        iron.set_current_device(from_name("npu2"))
    except Exception as error:  # noqa: BLE001 - report any toolchain import/setup failure
        return _fail(workdir, "template", f"toolchain setup failed: {error}")

    try:
        design = _build_identity(elements)
        design.specialize(n=elements).compile(
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
                "inputs": 1,
                "outputs": 1,
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

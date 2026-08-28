#!/usr/bin/env python3
"""Standalone IRON driver for the bfp16ebs8 characterization probes (issue #146).

Deliberately separate from crates/virtio-accel-xdna/compiler/xdna_compile.py: that helper is
the serving path's compiler; these probes are backend-local research artifacts. The toolchain
env setup mirrors the helper's `_configure_toolchain_env` against the same pinned prefix.

Usage:
    VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08 \
        python3 probe_compile.py p0 <output-dir>

Writes <output-dir>/final.xclbin and <output-dir>/insts.bin.
"""

import os
import sys
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent

P0_INPUT_FLOATS = 64
P0_OUTPUT_WORDS = 37  # 64 mantissa + 8 exponent + 72 native-struct bytes + 1 rnd word
P1_OUTPUT_WORDS = 181  # 10 modes x 72 plane bytes + 1 rnd word
P4_INPUT_BYTES = 576  # A (4 x 72) then B (4 x 72) raw bfp16ebs8 planes
P4_OUTPUT_WORDS = 64  # 64 FP32 accumulator lanes


def configure_toolchain_env() -> None:
    import sysconfig

    site_packages = Path(sysconfig.get_paths()["purelib"])
    mlir_dir = site_packages / "mlir_aie"
    peano = site_packages / "llvm-aie"

    aie_python = mlir_dir / "python"
    sys.path.insert(0, str(aie_python))

    os.environ["MLIR_AIE_INSTALL_DIR"] = str(mlir_dir)
    if peano.is_dir():
        os.environ["PEANO_INSTALL_DIR"] = str(peano)
    os.environ["NPU_RUNTIME"] = "hrx"
    os.environ["NPU2"] = "1"

    prefix = os.environ.get("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN")
    if not prefix:
        raise SystemExit("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not set")
    xclbinutil = Path(prefix) / "amd-npu-compiler/bld-xclbinutil/tools/hrx-xclbinutil"
    if xclbinutil.is_file():
        os.environ["AIE_XCLBINUTIL"] = str(xclbinutil)
    hrx_root = Path(prefix) / "amd-npu-compiler/third_party/.hrx-release"
    if hrx_root.is_dir():
        for candidate in sorted(hrx_root.iterdir()):
            if (candidate / "lib/libhrx.so").is_file():
                os.environ["HRX_DIR"] = str(candidate)
                break

    def prepend(var: str, value: str) -> None:
        current = os.environ.get(var, "")
        os.environ[var] = f"{value}{':' + current if current else ''}"

    prepend("PYTHONPATH", str(aie_python))
    prepend("PATH", f"{mlir_dir / 'bin'}:{peano / 'bin'}")
    prepend("LD_LIBRARY_PATH", str(mlir_dir / "lib"))


def build_probe(name: str, output_words: int):
    """One worker: one input line in, one raw little-endian word dump out."""
    import aie.iron as iron
    import numpy as np
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction

    kernel_source = (PROBE_DIR / f"kernel_{name}.cc").read_text()

    @iron.jit
    def probe(x_in: In, y_out: Out):
        if name == "p4":
            input_ty = np.ndarray[(P4_INPUT_BYTES,), np.dtype[np.uint8]]
            output_ty = np.ndarray[(output_words,), np.dtype[np.float32]]
        else:
            input_ty = np.ndarray[(P0_INPUT_FLOATS,), np.dtype[np.float32]]
            output_ty = np.ndarray[(output_words,), np.dtype[np.uint32]]
        kernel = ExternalFunction(
            f"probe_{name}",
            source_string=kernel_source,
            arg_types=[input_ty, output_ty],
        )
        of_in = ObjectFifo(input_ty, name="probe_in")
        of_out = ObjectFifo(output_ty, name="probe_out")

        def core_fn(in_fifo, out_fifo, probe):
            source = in_fifo.acquire(1)
            destination = out_fifo.acquire(1)
            probe(source, destination)
            in_fifo.release(1)
            out_fifo.release(1)

        worker = Worker(core_fn, [of_in.cons(), of_out.prod(), kernel])

        def sequence(source, destination, in_prod, out_cons):
            in_prod.fill(source)
            out_cons.drain(destination, wait=True)

        rt = Runtime(sequence, [input_ty, output_ty, of_in.prod(), of_out.cons()])
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    return probe


def main() -> int:
    probes = {"p0": P0_OUTPUT_WORDS, "p1": P1_OUTPUT_WORDS, "p4": P4_OUTPUT_WORDS}
    if len(sys.argv) != 3 or sys.argv[1] not in probes:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_toolchain_env()
    import aie.iron as iron
    from aie.iron.device import from_name

    iron.set_current_device(from_name("npu2"))

    design = build_probe(sys.argv[1], probes[sys.argv[1]])
    design.compile(
        xclbin_path=str(out_dir / "final.xclbin"),
        inst_path=str(out_dir / "insts.bin"),
    )
    print(f"wrote {out_dir}/final.xclbin and {out_dir}/insts.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

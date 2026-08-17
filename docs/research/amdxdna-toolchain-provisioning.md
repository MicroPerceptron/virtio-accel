# AMD XDNA toolchain provisioning and first hardware proof

Resolution record for [issue #81](https://github.com/MicroPerceptron/virtio-accel/issues/81),
part of the wayfinder map for the `virtio-accel-amdxdna` backend
([issue #75](https://github.com/MicroPerceptron/virtio-accel/issues/75), map
[issue #78](https://github.com/MicroPerceptron/virtio-accel/issues/78)). Executes the install
route designed by the HRX research ticket
([issue #79](https://github.com/MicroPerceptron/virtio-accel/issues/79),
`docs/research/hrx-runtime-contract.md` §4.2) on the reference machine, records every exact
pin, and proves the stack below Rust: driver + firmware + HRX + Krackan silicon executing
compiled designs correctly, with **no XRT userspace installed at any point**.

**Provisioned and proven: 2026-08-17.** Both proof runs `PASS!` (§6).

---

## 1. Prefix path and layout

Everything lives in one self-contained prefix; nothing was installed system-wide, and no
system configuration (udev, limits, packages) was changed. Deleting the directory removes the
toolchain completely.

```
~/toolchains/amdxdna-hrx-v2026.08/
├── env.sh                      # single entry point: source this
├── MANIFEST.md                 # machine-local copy of the pins + proof results
├── pip-freeze.txt              # exact venv package snapshot
├── python/                     # standalone CPython 3.12.14 (not the system interpreter)
├── ironenv/                    # Python 3.12 venv with all pinned wheels
├── npu-cache/                  # IRON JIT artifact cache (NPU_CACHE_HOME)
├── downloads/                  # sha256-verified source tarballs
└── amd-npu-compiler/           # fork checkout at the pinned commit
    ├── third_party/hrx-xclbinutil/    # pinned submodule
    ├── third_party/.hrx-release/      # extracted pinned libhrx release
    └── bld-xclbinutil/tools/hrx-xclbinutil   # XRT-free xclbinutil (built standalone)
```

The system interpreter is Python 3.14.6 and `python3.12` is not installed via dnf (the
machine has no passwordless sudo, and the ticket forbids system-wide installs anyway). The
pinned Python 3.12 requirement is satisfied by a standalone relocatable CPython inside the
prefix — a deliberate improvement over §4.2's `dnf install python3.12` step.

## 2. Exact pins (all verified)

| Component | Pin | Integrity |
|---|---|---|
| CPython 3.12 | `cpython-3.12.14+20260814-x86_64-unknown-linux-gnu-install_only.tar.gz` (astral-sh/python-build-standalone release `20260814`) | sha256 `3297691a…0766c0` matches the GitHub asset digest |
| Fork | `MicroPerceptron/amd-npu-compiler` @ `c95544269f0c074d6d3e213ee43cc34dc4100801` | commit checked out exactly; `git log -1` = `c9554426 Add AIECC_PATH environment variable override…` |
| `mlir_aie` | `1.4.1`, cp312 manylinux_2_35_x86_64 wheel from `https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.1` (valid because fork ≡ upstream; contains `hrxruntime`) | pip-resolved, recorded in `pip-freeze.txt` |
| Peano | `llvm-aie==21.0.0.2026080301+c9c5ecb7` (from `utils/peano-requirements.txt`) | pip-resolved |
| HRX | `jtuyls/hrx` tag `flm-hrx-amdxdna-v2026.07.30`, asset `hrx-amdxdna-2026.07.30-amdxdna-hal-native-rel-eb0b39f-linux-x86_64.tar.zst` → `libhrx.so.0.1.0` (executable-record ABI v0) | sha256 `661ed940…696345` verified by `utils/fetch-hrx-release.sh` and re-verified manually |
| `hrx-xclbinutil` | submodule @ `3940dd23bda7a941df9f2d10015fd66c9410b7af`, built Release with system gcc 16.1.1 | gitlink matches `.gitmodules` pin |
| eudsl | `eudsl-python-extras==0.1.0.20260801.905+68a0d7a` | pip-resolved (sha256 in `pip-freeze.txt`) |
| numpy | `2.5.2` (satisfies `>=2.5.1,<3.0`) | pip-resolved |

Anonymous (unauthenticated) downloadability of the HRX asset remains unproven — the fetch
ran under the machine's existing `gh` auth, closing #79's open question only for
authenticated access.

## 3. Host state (recorded, not modified)

| Fact | Value |
|---|---|
| OS / kernel | Fedora 44, `7.1.8-200.fc44.x86_64` |
| NPU | `04:00.1 Signal processing controller [1180]: AMD Strix/Krackan/Strix Halo NPU [1022:17f0] (rev 20)` — npu6 / Krackan Point |
| Driver | in-tree `amdxdna` (`drivers/accel/amdxdna`, drm-reported version 0.8.0), loaded at boot |
| Firmware loaded | `amdnpu/17f0_10/npu_7.sbin` → `npu.sbin.1.1.2.64` (protocol-7 kernel path, exactly as #79 predicted; `npu.sbin` → 1.0.0.63 also shipped) |
| Device node | `/dev/accel/accel0`, mode `0666` — Fedora 44's defaults already grant access; **the udev rule from #79's caveat list was not needed** |
| memlock | `ulimit -l` = 8192 KB, unchanged — **sufficient for the proof workloads**; revisit only if larger buffers hit pinning failures |
| Host toolchain used | gcc/g++ 16.1.1, cmake 4.3.0 (system) + cmake 4.4.2/ninja 1.13 (venv), zstd 1.5.7, git 2.55.0 — all already present; nothing installed |

## 4. Reproducible command sequence

As executed (deviations from `hrx-runtime-contract.md` §4.2 are marked ▲ and explained in §5):

```bash
PREFIX=~/toolchains/amdxdna-hrx-v2026.08
mkdir -p $PREFIX/downloads && cd $PREFIX/downloads

# 1. ▲ Standalone CPython 3.12 into the prefix (instead of dnf python3.12)
gh release download 20260814 --repo astral-sh/python-build-standalone \
  --pattern 'cpython-3.12.14+20260814-x86_64-unknown-linux-gnu-install_only.tar.gz'
# verify sha256 against the release asset digest, then:
cd $PREFIX && tar xzf downloads/cpython-3.12.14+*.tar.gz    # extracts to ./python
$PREFIX/python/bin/python3.12 -m venv $PREFIX/ironenv
source $PREFIX/ironenv/bin/activate
pip install --upgrade pip && pip install "cmake>=3.30" ninja

# 2. Fork at the pinned commit + xclbinutil submodule
git clone --filter=blob:none https://github.com/MicroPerceptron/amd-npu-compiler \
  $PREFIX/amd-npu-compiler
cd $PREFIX/amd-npu-compiler
git checkout c95544269f0c074d6d3e213ee43cc34dc4100801
git submodule update --init third_party/hrx-xclbinutil

# 3. Pinned wheels
pip install mlir_aie==1.4.1 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.1
pip install -r utils/peano-requirements.txt
pip install -r python/requirements.txt

# 4. Pinned libhrx (downloads, sha256-verifies, extracts, synthesizes env.sh)
./utils/fetch-hrx-release.sh

# 5. XRT-free xclbinutil, standalone from the submodule
cmake -G Ninja -B bld-xclbinutil -S third_party/hrx-xclbinutil -DCMAKE_BUILD_TYPE=Release
cmake --build bld-xclbinutil --target hrx-xclbinutil

# 6. Write $PREFIX/env.sh (§4.1 below), then verify:
source $PREFIX/env.sh
python3 -c "import aie.utils as u; print(u.has_hrx)"                    # -> True
python3 -c "import aie.utils as u; print(u.DEFAULT_TENSOR_CLASS.__name__)"  # -> HRXTensor
```

### 4.1 Environment (`$PREFIX/env.sh`)

Mirrors the first half of the fork's `utils/env_setup.sh`, skipping its `xrt-smi` probe
(the documented workaround for an XRT-free box):

```bash
source "$_prefix/ironenv/bin/activate"
export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | awk '/^Location:/ {print $2 "/mlir_aie"}')"
export PEANO_INSTALL_DIR="$(pip show llvm-aie | awk '/^Location:/ {print $2 "/llvm-aie"}')"
export PATH="${MLIR_AIE_INSTALL_DIR}/bin:${PEANO_INSTALL_DIR}/bin:${PATH}"   # aiecc, llvm-objcopy
export PYTHONPATH="${MLIR_AIE_INSTALL_DIR}/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${MLIR_AIE_INSTALL_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$_prefix/amd-npu-compiler/third_party/.hrx-release/hrx-amdxdna-…-linux-x86_64/env.sh"
                                    # sets HRX_DIR, LD_LIBRARY_PATH, CMAKE_PREFIX_PATH
export AIE_XCLBINUTIL="$_prefix/amd-npu-compiler/bld-xclbinutil/tools/hrx-xclbinutil"
export NPU2=1                       # XDNA2-class device
export NPU_RUNTIME=hrx              # select HRX in both the IRON/Python and C++ make flows
export NPU_CACHE_HOME="$_prefix/npu-cache"   # keep the JIT cache inside the prefix
```

`llvm-objcopy` needs no separate install: the `mlir_aie` wheel ships one in its `bin/`
(#79's caveat about GNU objcopy rejecting `EM_AIE` is satisfied by the PATH line above).

### 4.2 The proof runs

```bash
source $PREFIX/env.sh
cd $PREFIX/amd-npu-compiler/programming_examples/basic/vector_scalar_add

# ▲ Compile xclbin + insts.bin only (the Makefile's `all` also wants insts.elf → aiebu-asm)
python3 vector_scalar_add.py -d npu2 --xclbin-path=build/final.xclbin --insts-path=build/insts.bin

# ▲ Build the native HRX chain testbench with the system compiler (default is gcc-13)
mkdir -p _build_runlist_hrx && cd _build_runlist_hrx
cmake .. -DTARGET_NAME=vector_scalar_add_runlist_hrx -DTEST_SRC=test_runlist_hrx.cpp \
  -DUSE_HRX=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build . --config Release
cd ..

# Run on /dev/accel/accel0
./_build_runlist_hrx/vector_scalar_add_runlist_hrx -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

# Second, independent proof: the pure-Python IRON JIT path
cd ../vector_vector_add && python3 vector_vector_add.py
```

## 5. Deviations from the §4.2 plan (all Fedora-44 findings)

1. **Standalone CPython instead of `dnf python3.12`.** No sudo required, strictly more
   self-contained, same pinned interpreter version (3.12.14). The venv route is otherwise
   identical.
2. **`make all` fails on an XRT-free box — expected, and not needed.** The example Makefiles
   build `insts.elf` alongside the xclbin/insts pair via a single grouped rule, and the ELF
   leg shells out to `aiebu-asm` (an XRT-ecosystem tool, deliberately absent). HRX consumes
   only `final.xclbin` + `insts.bin`, so drive the design's `compile()` directly with
   `--xclbin-path/--insts-path` (what the `jit_xclbin` make template does). Backend tickets
   generating artifacts (`--aot-dir` later) never need the ELF leg.
3. **Host testbench CMake defaults to `gcc-13`/`g++-13`** (`mlir_aie_init.cmake` sets them
   only when undefined). Pass `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`; gcc 16.1.1
   compiles the C++23 testbench and links `hrx::hrx` cleanly.
4. **No udev rule, no memlock raise, no `NPU2=1` detection override needed.** Fedora 44
   ships `/dev/accel/accel0` as 0666; the 8 MB default memlock survived every proof run; the
   sysfs probe correctly identified `0x17f0` → `npu2` without `IRON_HRX_DEVICE`.

## 6. Results

Both proofs ran against `/dev/accel/accel0` (Krackan, `1022:17f0` rev 0x20) on 2026-08-17:

- **Native HRX chain testbench** (`vector_scalar_add`, two chained dispatches through
  `hrx_amdxdna_executable_create` → `hrx_stream_dispatch` ×2 → one `hrx_stream_synchronize`,
  lowered to a single `ERT_CMD_CHAIN`):

  ```
  Checking run 0 (out0 == in + 1)
  Checking run 1 (out1 == out0 + 1)
  Chain NPU time: 376us.
  PASS!
  ```

  Exit 0. This exercises the exact C ABI sequence the Rust backend will bind (#79 §5.2),
  including correct patch-table derivation inside libhrx (all-zero output would have
  signalled the known no-patch-table failure mode).

- **Pure-Python IRON JIT** (`vector_vector_add.py` under `NPU_RUNTIME=hrx`): compiled
  through aiecc + Peano + `hrx-xclbinutil` into the prefix-local cache and executed on the
  NPU — `PASS!`, exit 0.

Together these retire #79's two hardware unknowns for this machine: the §4.2 sequence **is**
valid on Fedora 44 (with the §5 adjustments), and Krackan rev 0x20 silicon executes
HRX-dispatched designs correctly despite not being CI-exercised by the fork.

## 7. What later tickets inherit

- Enter the toolchain with `source ~/toolchains/amdxdna-hrx-v2026.08/env.sh`; everything
  (compiler, Peano, libhrx, xclbinutil, JIT cache) resolves from the prefix.
- Artifact generation for the backend: the vector-add design's `--aot-dir` mode under this
  environment produces the HRX-ABI (unfolded) `final.xclbin`/`insts.bin` pair — the input
  for the precompiled-passthrough lifecycle ticket.
- The Rust backend's runtime needs exactly `$HRX_DIR/lib/libhrx.so.0.1.0` and the two
  headers in `$HRX_DIR/include/hrx/`; the Python environment is build-time only.
- The compiler-helper subprocess contract ticket should assume `aiecc` invoked from this
  venv. `$PREFIX/ironenv/bin/aiecc` launches in a clean environment (verified with
  `env -i`), so the helper spec should pass `AIE_XCLBINUTIL` and `PEANO_INSTALL_DIR`
  explicitly rather than inherit ambient environment; proving a full clean-environment
  compile is that ticket's job.

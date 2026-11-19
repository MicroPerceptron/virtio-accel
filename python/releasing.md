# Publishing the Python SDK

The distribution is named **virtio-accel** on PyPI; Python imports it as
`virtio_accel`. PyPI is the registry, `pip` is the installer, and a wheel (`.whl`)
is the prebuilt package. The source archive (`.tar.gz`) is the fallback for users
building with Rust. Python releases have their own version, independently of the
Rust crates and the protocol. Nothing has been uploaded by adding this workflow.

## One-time account setup

1. Create accounts on [PyPI](https://pypi.org/account/register/) and, separately,
   [TestPyPI](https://test.pypi.org/account/register/). Verify your email and enable
   two-factor authentication; save your recovery codes.
2. In each account's **Publishing** settings, add a **pending publisher** for a new
   project. Use these values (the environment differs between registries):

   | Field | Value |
   | --- | --- |
   | PyPI project name | `virtio-accel` |
   | GitHub owner | `MicroPerceptron` |
   | Repository | `virtio-accel` |
   | Workflow filename | `python-release.yml` |
   | Environment | `pypi` on PyPI; `testpypi` on TestPyPI |

   If you already own that project, add the publisher in its project settings
   instead. A pending publisher does not reserve a name; resolve any ownership
   conflict before releasing. See PyPI's [new-project Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
3. Create GitHub repository environments named `pypi` and `testpypi` under
   **Settings → Environments**. Configure their release approval/branch rules as
   desired. The workflow uses GitHub's short-lived identity to publish; no PyPI
   API token or repository publishing secret is required.

## Deferred Windows ARM64 distribution

Follow-up: [#209 — Windows ARM64 wheels and Qualcomm release validation](https://github.com/MicroPerceptron/virtio-accel/issues/209).

The first automated release contains five wheels: Linux x86-64/ARM64, macOS
x86-64/ARM64, and Windows x86-64. Windows ARM64 wheels and Qualcomm hardware
checks are deferred. No self-hosted runner is required or used by this workflow.

Hexagon remains implemented in the SDK and included in the source archive. Native
Windows ARM64 builds still require the complete QAIRT SDK, ARM64 Python and Rust,
and Visual Studio ARM64 C++ tools. Use [the manual test script](scripts/test-hexagon.ps1)
on the Snapdragon machine to build a local wheel and relay its test results.
It validates all Windows adapters and runtime DLL selection without registering
that machine as a GitHub Actions runner. These local wheels are not automatically
included in the PyPI release. On Windows ARM64, force a native Hexagon source build
with `VIRTIO_ACCEL_HEXAGON=1` and the documented QAIRT setup; installing the source
archive without the SDK cannot enable Hexagon.

Restoring a prebuilt Windows ARM64 wheel requires a suitable QAIRT build path and
validation of the current runtime loader. The earlier successful Qualcomm hardware
tests remain recorded in the README; they do not validate the subsequent loader changes.

## First rehearsal and publication

1. Merge the reviewed packaging work after PR CI passes.
2. Open **Actions → Release Python SDK → Run workflow** on `main`, choosing
   **artifacts**. This builds and tests all five wheels and the source archive,
   without uploading to a package registry. Download the `python-dist-*` artifacts
   for any additional hardware checks.
3. Run the same workflow on `main` with **testpypi**. TestPyPI is a separate registry;
   installing from it does not test your production publisher configuration.
   In a fresh environment, install dependencies from PyPI first, then the SDK:

   ```sh
   python -m pip install 'numpy>=2,<3' 'ml-dtypes>=0.5,<1'
   python -m pip install --no-deps --only-binary=:all: \
     --index-url https://test.pypi.org/simple/ virtio-accel==0.1.0a1
   python -c "import virtio_accel as va; print(va.devices()); print(va.device_diagnostics())"
   ```

4. For production, create a GitHub release from the reviewed `main` commit with
   tag **`python-v0.1.0a1`**. Mark this initial alpha as a prerelease. Publishing
   that GitHub release triggers the workflow. It publishes only after all builds,
   installed-wheel tests, source rebuild, and artifact
   completeness checks pass. The crates.io workflow ignores `python-v` releases.
5. Confirm the workflow succeeds and the files appear on the PyPI project page.
   In a fresh environment, test `python -m pip install --pre virtio-accel`.
   Optional OpenVINO runtime installation is `pip install --pre 'virtio-accel[openvino]'`
   on platforms/Python versions supported by Intel's package. Drivers and other
   vendor runtimes remain host prerequisites.

The five wheel targets are Linux x86-64/ARM64 (glibc 2.28+), macOS x86-64/ARM64
(macOS 14+), and Windows x86-64. They use CPython's `cp310-abi3` ABI for
ordinary CPython 3.10 and newer. This does not promise PyPy, free-threaded CPython,
musl Linux, or accelerator hardware on every architecture. Each wheel includes
only the providers supported on its operating system; discovery still requires
compatible hardware and runtime libraries.

## Subsequent releases

Update both `pyproject.toml` and `python/Cargo.toml`, then regenerate
`python/Cargo.lock`. For example, Python `0.1.0a2` corresponds to Rust
`0.1.0-alpha.2`; a stable release uses `0.1.0` in both. The tag must match the
Python version exactly. PyPI files cannot be overwritten: use a new version
for changed artifacts. If an upload is interrupted, inspect which files reached
the registry before retrying; the workflow deliberately does not skip existing
files silently.

Locally, the key packaging checks are:

```sh
python -m pip install 'maturin[zig]>=1.16,<2' 'twine>=6,<7'
python -m unittest discover -s python/scripts -p 'test_*.py' -v
maturin sdist --manifest-path python/Cargo.toml --out dist
python -m pip wheel --no-deps dist/*.tar.gz --wheel-dir rebuilt
python -m twine check --strict dist/*
```

Build Linux release wheels with `maturin build --release --locked
--manifest-path python/Cargo.toml --out dist --zig --compatibility manylinux_2_28
--auditwheel check --strip`. On macOS and Windows omit the Linux-specific flags.
`python python/scripts/check_release.py artifacts dist --complete` requires the
whole five-wheel release plus its source archive. `installed` verifies that the
active environment imports the installed wheel and contains the expected native
adapters. Full hardware regression tests remain necessary when native code changes.

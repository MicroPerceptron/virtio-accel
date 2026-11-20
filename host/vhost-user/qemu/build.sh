#!/usr/bin/env bash
# Build the reviewed development frontend; requires git, C compiler, Python,
# ninja, pkg-config, and GLib development headers. Patches retain upstream file licenses.
set -euo pipefail
patches=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=${1:?Usage: build.sh SOURCE_DIRECTORY}
commit=2d3df8abca265c9bcc9e438d691d561592060998
target=${QEMU_TARGET:-aarch64}
extra=()
case "$target" in
    aarch64) ;;
    x86_64) extra+=(--enable-kvm) ;;
    *) echo "QEMU_TARGET must be aarch64 or x86_64" >&2; exit 1 ;;
esac
if [[ ! -e "$source_dir" ]]; then
    git clone --depth 1 --branch v10.2.1 https://github.com/qemu/qemu.git "$source_dir"
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$commit" ]]; then
    echo "Expected QEMU v10.2.1 commit $commit; refusing to alter another checkout" >&2
    exit 1
fi
for patch in "$patches"/*.patch; do
    if git -C "$source_dir" apply --check "$patch" 2>/dev/null; then
        git -C "$source_dir" apply "$patch"
    else
        # Permit an already-applied patch, but do not repair unrelated changes.
        git -C "$source_dir" apply --reverse --check "$patch"
    fi
done
cd "$source_dir"
source_dir=$(pwd)
fingerprint=$(python3 - "$commit" "$target" "$patches" "$source_dir" <<'PY'
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

digest = hashlib.sha256()
def add(value):
    digest.update(value)
    digest.update(b'\0')
for value in (*sys.argv[1:], sys.executable, sys.version, platform.system(), platform.machine()):
    add(value.encode())
patches = Path(sys.argv[3])
for path in [patches / 'build.sh', *sorted(patches.glob('*.patch'))]:
    add(path.read_bytes())
for name in ('CC', 'CXX', 'AR', 'LD', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS',
             'PKG_CONFIG_PATH', 'PKG_CONFIG_SYSROOT_DIR', 'LD_LIBRARY_PATH'):
    add(f'{name}={os.environ.get(name, "")}'.encode())
for command in (shlex.split(os.environ.get('CC', 'cc')) + ['--version'],
                ['ninja', '--version'], ['pkg-config', '--modversion', 'glib-2.0'],
                ['pkg-config', '--cflags', '--libs', 'glib-2.0']):
    add(subprocess.check_output(command))
print(digest.hexdigest())
PY
)
mkdir -p build
cd build
if [[ ${QEMU_RECONFIGURE:-0} == 1 || ! -f build.ninja || \
      ! -f .virtio-accel-config || $(cat .virtio-accel-config) != "$fingerprint" ]]; then
    # Invoking configure from the source root deletes its auto-created build
    # directory. Configure out of tree so restored object files survive.
    ../configure --target-list="$target-softmmu" --disable-docs --disable-tools \
        --without-default-features --enable-tcg --enable-vhost-user --disable-werror "${extra[@]}"
    printf '%s\n' "$fingerprint" > .virtio-accel-config
else
    echo "Reusing matching QEMU configuration"
fi
ninja -j "${JOBS:-2}" "qemu-system-$target"
printf 'QEMU executable: %s/build/qemu-system-%s\n' "$source_dir" "$target"

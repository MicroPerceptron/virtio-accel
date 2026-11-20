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
./configure --target-list="$target-softmmu" --disable-docs --disable-tools \
    --without-default-features --enable-tcg --enable-vhost-user --disable-werror "${extra[@]}"
ninja -C build -j "${JOBS:-2}" "qemu-system-$target"
printf 'QEMU executable: %s/build/qemu-system-%s\n' "$(pwd)" "$target"

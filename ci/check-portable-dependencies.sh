#!/usr/bin/env bash
set -euo pipefail

readonly packages=(
  virtio-accel-cleanroom
  virtio-accel-proto
  virtio-accel-transport
  virtio-accel-core
)

for package in "${packages[@]}"; do
  tree="$(cargo tree -p "$package" -e normal,build,features --prefix none)"
  if grep -Eq 'feature "(std|alloc)"' <<<"$tree"; then
    echo "$package unexpectedly enables a std or alloc dependency feature:" >&2
    echo "$tree" >&2
    exit 1
  fi
done

cleanroom_tree="$(cargo tree -p virtio-accel-cleanroom -e normal,build --depth 1 --prefix none)"
cleanroom_dependencies="$(sed '1d' <<<"$cleanroom_tree")"
if [[ -n "$cleanroom_dependencies" ]]; then
  echo "virtio-accel-cleanroom must remain independent of every normal/build dependency:" >&2
  echo "$cleanroom_tree" >&2
  exit 1
fi

transport_tree="$(cargo tree -p virtio-accel-transport -e normal,build --depth 1 --prefix none)"
transport_dependencies="$(sed '1d' <<<"$transport_tree")"
if [[ -n "$transport_dependencies" ]]; then
  echo "virtio-accel-transport must remain independent of every normal/build dependency:" >&2
  echo "$transport_tree" >&2
  exit 1
fi

echo "portable dependency features are clean"

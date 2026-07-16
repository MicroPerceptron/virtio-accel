#!/usr/bin/env bash
set -euo pipefail

readonly packages=(
  virtio-accel-proto
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

echo "portable dependency features are clean"

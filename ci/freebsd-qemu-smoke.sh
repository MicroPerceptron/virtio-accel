#!/usr/bin/env bash

set -euo pipefail

WORKDIR="${FREEBSD_QEMU_WORKDIR:-${RUNNER_TEMP:-/tmp}/virtio-accel-freebsd-qemu}"
IMAGE_RELEASE="${FREEBSD_QEMU_IMAGE_RELEASE:-14.4-RELEASE}"
IMAGE_NAME="${FREEBSD_QEMU_IMAGE_NAME:-}"
SSH_PORT="${FREEBSD_QEMU_SSH_PORT:-2222}"
SSH_TIMEOUT_SECONDS="${FREEBSD_QEMU_SSH_TIMEOUT_SECONDS:-600}"
GUEST_COMMAND_TIMEOUT_SECONDS="${FREEBSD_QEMU_COMMAND_TIMEOUT_SECONDS:-1800}"
GUEST_CHECKOUT="${GITHUB_SHA:-${GITHUB_REF_NAME:-main}}"
GUEST_REPO="${FREEBSD_QEMU_GITHUB_REPO:-https://github.com/MicroPerceptron/virtio-accel.git}"

mkdir -p "${WORKDIR}/images"

IMAGE_SOURCE="${WORKDIR}/images/${IMAGE_NAME}"
IMAGE_FILE="${WORKDIR}/images/freebsd-ci.qcow2"
VM_DISK="${WORKDIR}/vm.qcow2"
QEMU_LOG="${WORKDIR}/qemu.log"
SERIAL_LOG="${WORKDIR}/serial.log"
FAILED_BASES=()

if [[ -z "${IMAGE_NAME}" ]]; then
  IMAGE_BASE=""
  for release in "${IMAGE_RELEASE}" "14.4-RELEASE" "14.3-RELEASE" "14.2-RELEASE"; do
    for base in \
      "https://download.freebsd.org/releases/VM-IMAGES/${release}/amd64/Latest" \
      "https://download.freebsd.org/releases/VM-IMAGES/${release}/amd64"; do
      base_index="${base}"
      base_html="$(curl -fsSL "${base_index}/" 2>/dev/null | tr -d '\r' || true)"
      FAILED_BASES+=("${base_index}/")
      if [[ -n "${base_html}" ]]; then
        parsed="$(printf '%s' "${base_html}" | sed -n 's/.*href="\([^"]*amd64[^"]*qcow2[^"]*\.xz\)".*/\1/p' | head -n 1)"
        if [[ -n "${parsed}" ]]; then
          IMAGE_NAME="${parsed##*/}"
          IMAGE_BASE="${base_index}"
          break 2
        fi
      fi
      for candidate in \
        "FreeBSD-${release}-amd64-BASIC-CLOUDINIT-ufs.qcow2.xz" \
        "FreeBSD-${release}-amd64-BASIC.qcow2.xz" \
        "FreeBSD-${release}-amd64-ufs.qcow2.xz" \
        "FreeBSD-${release}-amd64-zfs.qcow2.xz" \
        "FreeBSD-${release}-amd64-BASIC-CLOUDINIT-zfs.qcow2.xz"; do
        if curl -fsSLI "${base_index}/${candidate}" >/dev/null 2>&1; then
          IMAGE_NAME="${candidate}"
          IMAGE_BASE="${base_index}"
          break 3
        fi
        FAILED_BASES+=("${base_index}/${candidate}")
      done
    done
  done
fi

if [[ -n "${IMAGE_NAME}" ]]; then
  IMAGE_SOURCE="${WORKDIR}/images/${IMAGE_NAME}"
fi

if [[ -z "${IMAGE_NAME}" ]]; then
  echo "Could not resolve a FreeBSD QCOW2 image URL. Attempted:"
  for failed in "${FAILED_BASES[@]:-}"; do
    echo "  - ${failed}"
  done
  exit 1
fi

if [[ ! -s "${IMAGE_SOURCE}" ]]; then
  echo "Downloading ${IMAGE_NAME}"
  curl -fSL "${IMAGE_BASE}/${IMAGE_NAME}" -o "${IMAGE_SOURCE}"
fi

if [[ ! -s "${IMAGE_FILE}" ]]; then
  if [[ "${IMAGE_NAME}" == *.xz ]]; then
    unxz -c "${IMAGE_SOURCE}" > "${IMAGE_FILE}"
  else
    cp "${IMAGE_SOURCE}" "${IMAGE_FILE}"
  fi
fi

cp "${IMAGE_FILE}" "${VM_DISK}"

cleanup() {
  if [[ -n "${QEMU_PID:-}" ]] && kill -0 "${QEMU_PID}" >/dev/null 2>&1; then
    kill "${QEMU_PID}" >/dev/null 2>&1 || true
    wait "${QEMU_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

qemu-system-x86_64 \
  -accel tcg \
  -machine q35 \
  -cpu max \
  -smp 2 \
  -m 2048 \
  -drive file="${VM_DISK}",if=virtio,format=qcow2 \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
  -display none \
  -serial file:"${SERIAL_LOG}" \
  -monitor none \
  -no-reboot \
  > "${QEMU_LOG}" 2>&1 &

QEMU_PID=$!

wait_for_ssh() {
  local user="$1"
  local password="$2"
  local deadline=$((SECONDS + SSH_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if sshpass -p "${password}" ssh -p "${SSH_PORT}" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=5 \
      "${user}@127.0.0.1" true >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
  return 1
}

FREEBSD_USER=
FREEBSD_PASS=

if wait_for_ssh freebsd freebsd; then
  FREEBSD_USER="freebsd"
  FREEBSD_PASS="freebsd"
elif wait_for_ssh root root; then
  FREEBSD_USER="root"
  FREEBSD_PASS="root"
else
  echo "Timed out waiting for SSH on port ${SSH_PORT}"
  tail -n 120 "${SERIAL_LOG}" || true
  exit 1
fi

REMOTE_REPO="${GUEST_REPO}"
REMOTE_REF="${GUEST_CHECKOUT}"

timeout "${GUEST_COMMAND_TIMEOUT_SECONDS}" sshpass -p "${FREEBSD_PASS}" ssh \
  -p "${SSH_PORT}" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${FREEBSD_USER}@127.0.0.1" \
  "GITHUB_SHA='${REMOTE_REF}' REPO_URL='${REMOTE_REPO}' bash -s" <<'EOF'
set -euo pipefail

if ! command -v pkg >/dev/null 2>&1; then
  echo "pkg is not available in the FreeBSD image"
  exit 1
fi

pkg update -y
if ! command -v git >/dev/null 2>&1; then
  pkg install -y git
fi
if ! command -v cargo >/dev/null 2>&1; then
  pkg install -y rust
fi

rm -rf /tmp/virtio-accel
git clone --depth 1 --filter=blob:none "${REPO_URL}" /tmp/virtio-accel
cd /tmp/virtio-accel
git fetch --depth 1 origin "${GITHUB_SHA}"
git checkout "${GITHUB_SHA}"

cargo run --example backend_conformance
EOF

echo "FreeBSD QEMU smoke test passed. Logs:"
tail -n 80 "${SERIAL_LOG}" || true

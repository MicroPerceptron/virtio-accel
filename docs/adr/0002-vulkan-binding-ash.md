# 2. Vulkan binding strategy: `ash` with runtime loader discovery

- Status: accepted
- Resolves: wayfinder map #154, ticket 2 (grilling: ratify the Vulkan binding strategy) and the
  crate layout convention question carried by ticket 7 (scaffold)

## Context

The workspace convention for host-native FFI is a small hand-written `ffi.rs` declaring only the
validated subset (OpenVINO's is 171 lines; XDNA's declares only the hardware-validated HRX
subset). Vulkan's surface is far larger than any prior target's validated subset. The map's
scouting table (recorded 2026-08-27) evaluated `ash`, `vulkanalia`, `vulkano`, `erupt`, `wgpu`,
`gpu-allocator`, `vk-mem`, and `rspirv`; this decision ratifies its `ash` lean and settles the
build-time gate that the scaffold (`virtio-accel-vulkan`) is built on.

## Decision

- Bind Vulkan through `ash` when the FFI lands (ticket 8), pinned to an exact version. The
  `loaded` feature only: the platform Vulkan loader is dynamically loaded at runtime via
  `libloading` — no Vulkan-Headers, no SDK at build time, no link-time dependency.
- The audit posture moves with the declarations: `SAFETY.md` pins the `ash` version, the entry
  points actually used, and the invariants this crate owns around them (one owner plus `Drop` per
  handle; status checked before out-parameters are trusted). The release-policy
  discussion-and-evidence requirement for the new unsafe exception is discharged alongside the FFI
  landing (tickets 2/8); the scaffold compiles no `unsafe` at all.
- `gpu-allocator` is deferred to evidence: it is adopted only if allocation probing (tickets 3 and
  8) shows `maxMemoryAllocationCount` pressure; `vk-mem` stays rejected; `vulkanalia` remains the
  named fallback if `ash` vetting fails at ticket 8.

## Convention question resolved

With `ash` under `loaded` there is nothing to detect at build time, so the `va_vulkan` cfg cannot
probe an SDK the way `va_openvino`, `va_hexagon`, or `va_xdna` do. The three-state env control and
loud force-on failure semantics are preserved without a file probe:

- `VIRTIO_ACCEL_VULKAN=0` forces the placeholder everywhere;
- `VIRTIO_ACCEL_VULKAN=1` forces the native path, and the build fails loudly when the target OS is
  outside the enumerated supported host set;
- unset is auto: the native path compiles on enumerated host targets, the placeholder elsewhere.

The supported set is enumerated in `build.rs` (Linux, Android, Windows, and macOS). Runtime
presence is discovered by loader lookup at run time and reported as
`InitError::RuntimeUnavailable`; it is never a compile-time property. This differs from the
SDK-probing backends only in what "detection" means, and `docs/portability.md` carries the
Vulkan-specific note that the `host-native` tier wording demands.

## Rationale

- `ash` is the ecosystem-standard raw binding (wgpu sits on it; 33M+ downloads), thin and
  dependency-light enough to survive `deny.toml`'s `multiple-versions = "deny"` gate, and MIT OR
  Apache-2.0 licensed. A hand-written subset would put a much larger unsafe surface under in-tree
  audit with no validated subset worth the name; `ash` moves the raw declarations out of tree and
  shrinks `SAFETY.md` to the entry points actually used.
- Runtime discovery over build-time probe: the issue's own scope (see #126) requires that the
  workspace builds and tests on hosts with no Vulkan SDK; a loader lookup at run time makes the
  GPU-less CI lane (lavapipe) and the real-GPU lane identical code paths.

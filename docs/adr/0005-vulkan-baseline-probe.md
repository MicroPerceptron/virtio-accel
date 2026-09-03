# 5. Vulkan baseline and capability probe: 1.3 floor with per-feature gating

- Status: accepted (research; verified against Mesa sources and gpuinfo as of Sep 2026)
- Resolves: wayfinder map #154, ticket 3 / GitHub issue #176 (research: minimum Vulkan baseline
  and honest capability probe)

## Findings (evidence recorded 2026-09-03)

| Question | Finding | Source |
| --- | --- | --- |
| lavapipe API floor | Vulkan 1.3 since Mesa 23.1 (May 2023); 1.4 in Mesa 25.x; current Mesa source pins `LVP_API_VERSION` to 1.4 with a dead 1.3 fallback branch | `src/gallium/frontends/lavapipe/lvp_device.c` |
| synchronization2 | Core in Vulkan 1.3 (KHR-promoted): lavapipe exposes both the extension and the core feature | Khronos registry; `lvp_device.c` |
| `VK_KHR_shader_float16_int8` / `VK_KHR_shader_integer_dot_product` | lavapipe, RADV, ANV all expose both; both core (feature-gated) in 1.3 | `lvp_device.c`, `src/amd/vulkan/radv_physical_device.c`, `src/intel/vulkan/anv_physical_device.c` |
| `VK_KHR_shader_float_controls` (+ `float_controls2`) | Exposed by lavapipe, RADV, ANV | same sources |
| `maxMemoryAllocationCount` | Spec minimum 4096; RADV/ANV/lavapipe report UINT32_MAX; NVIDIA (WDDM on Windows) commonly 4096 | spec limits; gpuinfo; NVIDIA forum |
| `minStorageBufferOffsetAlignment` | RADV 4, ANV 4, lavapipe 16 | same sources |
| `nonCoherentAtomSize` | RADV 64, ANV 64, lavapipe 64; gpuinfo shows 64/128 | same sources |

## Decision

1. **Vulkan 1.3 is the minimum API baseline.** Lavapipe has been ≥ 1.3 since Mesa 23.1; every
   plausible 2026 CI image satisfies it. A 1.2 floor buys compatibility with nothing relevant to
   the lavapipe lane. The `synchronization2`, `dynamicRendering`, `maintenance4`, and
   `shaderIntegerDotProduct` features become core-extension-free; they must still be *enabled* at
   device creation (core removes extension dependency, not the feature enable).
2. **Required-feature set (baseline).** `synchronization2` and `shaderIntegerDotProduct` enabled
   unconditionally. All remaining items are *queries*, not requirements: FP16 tier advertises only
   when `shaderFloat16` and `vkGetPhysicalDeviceFeatures2` float-controls prove per-device (ADR 0004);
   INT8 operator tiers advertise only when `shaderInt8` is present (MATMUL additionally needs
   `shaderIntegerDotProduct`, which is baseline here).
3. **Memory-domain map.** `Host` → `HOST_VISIBLE|HOST_COHERENT`, persistently mapped. `Device` →
   `DEVICE_LOCAL` with staging confined to `write_buffer`/`read_buffer` (this backend is the first
   to *advertise* `DEVICE_LOCAL_MEMORY`; the mock is the only prior). `Shared` →
   `DEVICE_LOCAL|HOST_VISIBLE` (ReBAR/UMA), advertised **only when the type actually exists** —
   never assumed.
4. **Allocation count.** Assume `maxMemoryAllocationCount = 4096` (NVIDIA/WDDM binds this).
   `DeviceLimits` stays aggregation-safe: dedicated allocations with honest low limits;
   suballocation is permitted and remains `DIRECT_BINDING`-legal because binding a VkBuffer derived
   from the same allocation at an offset copies nothing.
5. **Alignment.** Do not assume RADV's (4/64). Report the actual
   `minStorageBufferOffsetAlignment` and `nonCoherentAtomSize` per device; the flush/invalidate
   granularity uses a safe superset of observed values (64 vs 128 across vendors).

## Consequences

- Ticket 8 (FFI) uses 1.3-only entry points; extension probing collapses to *feature queries*.
- Ticket 5 (FP16) relies on `VK_KHR_shader_float_controls` exposure of the three known drivers;
   gating remains per-device with per-lane verification in ticket 10.

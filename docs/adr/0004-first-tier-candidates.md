# 4. First advertised numerical tier: FP32 base, INT8 candidate, FP16 deferred

- Status: provisional (declares the scaffold's `Target` constants; wayfinder map #154 ticket 5
  closes on per-ICD float-controls evidence before operator work in ticket 9)
- Resolves: nothing final; records the candidate set the scaffold compiles

## Decision

- **FP32 base tier** (`VULKAN_TOSA_TARGET`): TOSA 1.0, floating-point profile, no extensions.
  Universal across Vulkan 1.x; the floor every advertised tier list contains.
- **INT8 tier candidate** (`VULKAN_TOSA_INTEGER_TARGET`): TOSA 1.0, integer profile, no
  extensions; gated per device on `shaderInt8` (plus `VK_KHR_shader_integer_dot_product` for
  MATMUL). Declared as a scaffold candidate pending the operator subset table ticket 5 must
  produce.
- **FP16 deferred**: no FP16 target constant is declared. A device may offer it only when
  `shaderFloat16` (`VK_KHR_shader_float16_int8`) and `VK_KHR_shader_float_controls` probing prove
  the shared corpus's non-finite, subnormal, and signed-zero edges (`IDENTITY_EDGES_*`). That is
  per-tier evidence, not an assumption, and it is checked per lane in the CI ticket.
- **FP8**: rejected loudly at admission; no portable Vulkan path exists.

## Consequences

- The scaffold exports exactly the two constants above from `crates/virtio-accel-vulkan/src/lower.rs`,
  so hardware-free golden lowering tests have a stable footing before the capability descriptors
  and operator table arrive with ticket 5's close.

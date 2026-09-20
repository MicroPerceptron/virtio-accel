# 3. TOSA→SPIR-V lowering: checked-in shaders specialized by constants

- Status: accepted (with falsification trigger)
- Resolves: wayfinder map #154, ticket 4 (grilling: program representation and lowering mechanism)

## Context

XDNA requires a bounded out-of-process compiler because `aiecc` is Python; Hexagon similarly
bridges QAIRT. Vulkan compute needs neither: per-operator SPIR-V compute shaders can be authored
once, precompiled, and checked into the crate; shapes and tiling are then bound at compile-free
specialization. TOSA-only artifacts stay the v1 admission format (#126 decision 2); a
SPIR-V-direct format is a deliberate follow-on, never smuggled in.

## Decision

- Program compilation: per-operator SPIR-V compute shaders checked into the crate, specialized at
  `load_program` via Vulkan specialization constants. No toolchain on the serving host, no
  subprocess, no Python.
- `VkShaderModule` and `VkComputePipeline` creation happens at `load_program`, never during
  `submit`. Retained pipelines, command buffers, and descriptor pools are charged against
  `ArtifactRef::resident_bytes`. The `VkPipelineCache` policy defaults to none unless ticket 8
  measures a need, since pipeline creation cost is paid once per program at load.
- Security invariant recorded: guest bytes never reach the driver's shader compiler. The driver
  consumes only crate-authored SPIR-V parameterized by validated shapes — the threat-model's
  transient-compile-budget clause (`docs/threat-model.md`).
- `spirv-tools`/`naga` are dev-dependencies at most, for CI validation of the checked-in SPIR-V.
  Runtime `rspirv` emission at load remains the documented fallback.

## Falsification trigger

If any advertised operator needs structure that specialization constants cannot express (for
example data-dependent control flow), the fallback to `rspirv` load-time emission is reopened as
a design note before operator work (ticket 9) proceeds. Until that trigger fires, the checked-in
shader approach holds.

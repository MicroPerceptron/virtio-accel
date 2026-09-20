# 1. Data-plane posture for GPU-class hardware: the graph shape is permanent

- Status: accepted (ratifies the lean seeded in wayfinder map #154)
- Resolves: wayfinder map #154, ticket 1 (grilling: data-plane posture for GPU-class hardware)

## Context

Protocol 1.0's data plane is graph-shaped: one submission is one opaque program artifact plus flat
slot-to-range buffer bindings plus a relative timeout, yielding exactly one completion event. No
command buffers, descriptor sets, barriers, fences, dispatch geometry, or cross-submission
dependency edges are guest-visible. The first GPU-class decision is whether that shape is an
NPU-era provisional to be reshaped toward GPU-native concepts, or the spec's permanent shape that
GPU backends fit through like every other provider.

## Decision

The graph shape is ratified as the permanent shape. GPUs operate NPU-like for this protocol's
purposes: whole-program artifacts, flat slot bindings, and one event per submission, implemented
in the most performant way Vulkan allows. Guest-visible command primitives are never required for
steady-state efficiency — pre-recorded command buffers, specialization constants, pooled
descriptors, and fence polling are all provider-internal.

## Rationale

- Reshaping the submission payload is a category-3 breaking wire change (new protocol major;
  `docs/wire-abi.md` §9) — deliberately expensive, classified before any code moves.
- The GPU-flavored primitives already have deliberately reserved, unassigned feature bits
  (`MULTI_QUEUE`, `EVENT_QUEUE`, `EXTERNAL_MEMORY`, `TIMELINE_FENCES`). If reshaping is ever
  warranted, the sanctioned path is a protocol-change proposal through those bits (#113 is the
  precedent: it designed the external-memory handoff and chose completion-gating over fences,
  leaving values unassigned).
- Hexagon (#77, #95, #96) is the existence proof that unlike hardware operates graph-shaped with
  zero protocol change.

## Falsifier

The bet stays honest through tickets 6 (execution/event model) and 8 (identity end-to-end): if
steady-state submission provably cannot meet performance budgets without guest-visible batching
or dependency edges, that evidence feeds a protocol-change proposal through the reserved bits —
never a backend workaround that leaks GPU concepts into the graph-shaped contract.

## Consequences

- The Vulkan backend stays strictly inside #126's non-goals: no wire-ABI or portable-contract
  change.
- Any Protocol 1.0 document wording this resolution touches is erratum-class (category 1: no
  accepted or emitted bytes change).

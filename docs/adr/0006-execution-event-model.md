# 6. Execution and event model: fence poll, bounded pools, no cancellation

- Status: accepted (design; proven-in-ticket-8 clause recorded below)
- Resolves: wayfinder map #154, ticket 6 (grilling: execution and event model)

## Decision

1. **Nonblocking completion without a worker thread.** `vkGetFenceStatus` is a genuine
   nonblocking status read; `poll_event` consults it directly. Unlike the Hexagon and XDNA
   backends, no dispatch/synchronization worker is serialized. Ticket 8 proves this — the clause
   in SAFETY.md says "no worker thread (proven in ticket 6, not assumed)" in proof form: if ticket
   8's I/O reaches indeterminate under blocking-poll, this design reopens.
2. **Bounded preallocated pools sized against `DeviceLimits`.** Each context preallocates a
   fixed ring of (command buffer, fence, descriptor set) triples sized at
   `DeviceLimits.max_events_per_context`. `submit` claims one slot, records commands, and
   `vkQueueSubmit` success is the admission boundary — rejected before it, indeterminate only on
   ambiguous failure after it. Destroyed events return slots to the ring, which keeps the host
   queue depth bounded even under guest control.
3. **Finite timeouts rejected pre-admission (Hexagon precedent).** Since Vulkan has no cancel
   primitive, `Timeout::AfterNs` is rejected at `submit` with `BackendError::DeadlineExpired` — the
   same admission posture Hexagon and XDNA hold. `Timeout::Infinite` is the only accepted timeout.
   OpenVINO's deadline-latched poll would work here too, but its latching keeps resources retained
   until the fence actually signals, which would complicate the bounded-ring accounting without
   improving semantics.
4. **Device loss.** `VK_ERROR_DEVICE_LOST` maps to an event latch of `Failed(DeviceLost)`, a
   sticky poisoned instance flag, and whole-instance discard as prescribed by #126's non-goal
   wording. After poisoning, entry points are never re-entered except destruction, whose errors are
   latched identically.
5. **No `EVENT_CANCELLATION`.** Capability is not advertised; `cancel_event` returns
   `Unsupported` by default.

## Rationale

- `Timeout::AfterNs` pre-admission rejection keeps event/slot ownership unambiguous — a fence is
  never orphaned without a slot, and reclaim only depends on fence signal.
- The bounded ring is exactly the single "pool member owned by the event" language SAFETY.md
  already requires of the audit ticket.

## Falsification guard

If ticket 8 benchmarks show `vkGetFenceStatus` polling dominates steady-state (e.g. under
lavapipe), the fallback to a worker-thread poll sits behind the same slot ring unchanged; the
AST gate in ticket 8 measures before re-factoring.

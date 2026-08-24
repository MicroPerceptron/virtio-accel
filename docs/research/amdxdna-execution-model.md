# virtio-accel-amdxdna: execution-model mapping and event-bridge spec

Design record for [issue #85](https://github.com/MicroPerceptron/virtio-accel/issues/85)
(wayfinder map [#78](https://github.com/MicroPerceptron/virtio-accel/issues/78)). Implemented
by the scaffold/FFI/hardware tickets (#86–#88 onward). Grounding: the HRX ABI research (#79),
the #83 crate design, #75's decided constraints (serialized dispatch worker; bounded
preallocated submission ring/event pool; no `EVENT_CANCELLATION` absent an honest primitive),
and **new primary-source research into the in-tree `amdxdna` kernel driver at v7.1** (§5,
read directly from `drivers/accel/amdxdna/aie2_ctx.c`).

Precedent stance (Intel-alignment principle): every choice below matches
`virtio-accel-openvino` where structure is comparable (the process-wide shared runtime owner,
logical contexts/queues, the acceptance boundary), matches `virtio-accel-hexagon` where the
*hardware situation* is comparable (no cancellation → finite deadlines rejected), and exceeds
both only in §5, where our kernel driver offers verified recovery behavior neither precedent
had available.

## 1. Object mapping

| Trait object | HRX representation |
|---|---|
| process-wide | One audited shared owner (`OnceLock`-held, refcounted): `hrx_gpu_initialize(0)` + `hrx_gpu_device_get(0)`. Never shut down (`hrx_gpu_shutdown` unused — fork precedent). The OpenVINO `shared_core()` pattern. |
| `AmdXdnaAccelerator` instance | One `hrx_stream_t` + one long-lived dispatch worker thread + the submission ring/event pool. The serialization unit and the device-loss discard unit are the same object. |
| `Context` | Logical record (id + accounting). Owns no HRX state. |
| `Queue` | Logical record. All queues funnel into the instance's single worker lane (the HRX stream is not safe for concurrent dispatch); cross-queue ordering is global admission order — documented. |
| `Program` | Retained `hrx_executable_t` + export ordinal + slot plan validated at load via `hrx_executable_export_info`. |
| `Buffer` | HRX buffer + persistent mapping + an in-flight guard (generation/refcount). `write_buffer` flushes (`flush_range`) immediately after writing; transfers reject in-flight allocations. |
| `Event` | Preallocated pool slot holding one latched atomic terminal state + guards on the invocation's buffers/program until terminal + destroyed. |

## 2. The worker lane and the bounded ring

- `submit` performs validation only (slot plan, access modes, ranges, in-flight conflicts,
  poisoning check), then enqueues onto the **preallocated submission ring**. Ring full →
  `SubmitFailure::Rejected(Busy)`; submit never blocks and never allocates.
- **Ring depth defaults to 1** — deliberately matching the Hexagon backend's
  one-admitted-request semantics so schedulers observe identical behavior across backends —
  and is configurable at construction; the event pool is sized with it. Raising the default
  is a data-driven decision for after first hardware experience.
- Worker loop per entry: `hrx_stream_dispatch` → `hrx_stream_synchronize` →
  `invalidate_range` on each output buffer → latch the event terminal state exactly once →
  release the entry's guards.
- **Threading note (decided against thread-per-submission):** the HRX stream requires
  caller-side serialization, so per-submission threads would serialize on a lock while adding
  a spawn (allocation + latency) per submission — a queue wearing thread costumes. At depth 1
  the persistent worker is observably identical to Hexagon's thread-per-request; unlike it,
  the same machinery scales by changing one number. On a future kore host the worker is one
  long-lived parked task — the cheapest citizen a scheduler can host. Dynamic placement
  parallelism is delivered by multiple instances/partitions (§7), not by multiplying blocked
  waiter threads.

## 3. No cross-submission batching

HRX can lower several recorded dispatches into one `ERT_CMD_CHAIN` (measured chain overhead
~376 µs on the reference machine, #81). v1 still executes **one submission per
dispatch+synchronize round-trip**: a chained failure cannot be attributed to a single
submission, and exact failure attribution outranks amortized launch cost. Chaining is
reserved for future multi-dispatch *programs* (one event, attribution intact). Neither
precedent backend batches.

## 4. Timeout semantics

`Timeout::AfterNs` is **rejected before admission as `DeadlineExpired`** — the Hexagon
precedent, applied for the same reason: cancellation does not exist for this hardware at any
layer (§5.1), so a finite deadline is a promise the backend cannot keep once work reaches
hardware. `Timeout::Infinite` is the supported form. OpenVINO accepts finite deadlines only
because `ov_infer_request_cancel` exists; matching Intel's *honesty rule* here produces
Qualcomm's behavior. Deadline management belongs to the scheduler above (placement-time
decisions), which — unlike the device — can decline to place time-critical work on a
non-preemptible engine. Queue-only deadline enforcement (safe abandonment of not-yet-
dispatched entries) is a documented possible v2, not v1.

`EVENT_CANCELLATION` is not advertised; `EventState::Cancelled` is unreachable.

## 5. Device loss: two tiers, grounded in the kernel driver

**Research finding (primary source, `torvalds/linux` v7.1,
`drivers/accel/amdxdna/aie2_ctx.c`):**

1. There is **no user-facing cancel/abort** for a submitted command: the driver's ioctl
   surface is create/destroy/config hwctx, submit, wait. HRX exposes none either (#79).
   Firmware preemption features (`AIE2_PREEMPT`, frame-boundary preempt, per-context QoS
   priority) schedule *between* hardware contexts; they cancel nothing.
2. The driver arms a **60 s per-job watchdog** (`HWCTX_MAX_TIMEOUT = 60000`,
   aie2_ctx.c:30). On expiry, `aie2_sched_job_timedout` captures a firmware health report
   (fatal type, exception PC, failing sub-command), marks the command
   `ERT_CMD_STATE_TIMEOUT`, **destroys and recreates the hardware context**
   (`aie2_hwctx_stop` → `aie2_destroy_context` → restart), and resumes the scheduler. The
   userspace wait *returns with an error*, and the kernel has quiesced device access to the
   job's buffers before anyone unblocks.

**Tier 1 — expected hang path.** `hrx_stream_synchronize` returns an error (typically within
~60 s via kernel TDR): the worker latches the event **`Failed(External{AMDXDNA})` — a normal
terminal state**; buffers are releasable because the kernel quiesced the context first.
Diagnostics (HRX status string; kernel health report via host logs) stay host-side. Because
libhrx is **not validated** to remain healthy after an under-the-hood context recreate (the
fork's own runbook prescribes a driver reload), the instance is nonetheless **poisoned**:
every subsequent operation returns `DeviceLost`. Trusting the stream post-TDR is a possible
upgrade gated on hardware evidence.

**Tier 2 — true wedge (rare).** The sync never returns (kernel TDR itself failed). The
worker-side watchdog — **default 120 s, derived: strictly greater than the kernel's 60 s so
the kernel always acts first**; configurable — latches the instance poisoned. The in-flight
event is deliberately **never made terminal** (a terminal state licenses the caller to free
memory an out-of-control device may still write); `poll_event` on it returns
`Err(DeviceLost)`; `destroy_event` returns `Rejected` (pending events cannot be destroyed).
The invocation's guards intentionally leak until process exit — stated in SAFETY.md §4.

**Recovery (both tiers):** per the implementer guide, no reset method exists — discard the
instance, apply the documented driver-reload procedure when required, construct a fresh
instance. The map's device-loss-testing fog item sharpens to: induce a hang on hardware,
observe tier 1 (kernel TDR → terminal `Failed` + poisoning), exercise instance discard;
tier 2 is exercised with a fault injector.

## 6. Rejected vs indeterminate ownership

Ring entry is the acceptance boundary. Before it: every failure is
`SubmitFailure::Rejected` — nothing retained, retry lawful. After it: the event owns the
invocation; worker-side HRX errors latch `Failed(...)` (normal terminal, buffers released on
guard drop). `Indeterminate` is reserved for genuine unknowns; HRX status reporting is
definite, so v1's paths produce none. Releases: `destroy_event` on pending → `Rejected`
(returned live); parents with live children → `Rejected`; terminal-event release frees the
pool slot and cannot fail. Both precedent backends draw the identical line.

## 7. Parallelism outlook (follow-on ticket, outside map #78)

The silicon supports real spatial parallelism: the 8-column array partitions
(`npu2_1col`…`npu2_7col`), the driver multiplexes a finite pool of hardware contexts, and
firmware (version-gated) adds per-context priority with frame-boundary preemption between
contexts. HRX exposes one stream and default QoS today, so v1 models the NPU as one serial
lane. The follow-on ticket covers: validating multiple HRX streams/instances on hardware,
column-partition targeting, QoS priority plumbing, and exposing multiple lanes as
schedulable resources to a placement scheduler (kore's manifold). Nothing in this spec
forecloses it: instances are self-contained lanes by construction.

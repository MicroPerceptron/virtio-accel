# Linux adapter memory and lifetime audit

The C module is an OS adapter outside portable Rust packages. Python uses standard
`fcntl`; no Rust unsafe exception is introduced. Module source is MIT-licensed
and declares `Dual MIT/GPL` for kernel export compatibility.

## Address validation

Copy the ioctl header once. Check sizes against immutable per-device bounds and
reject nonzero flags/padding before copying its inline byte tail. There are no
embedded user pointers. All user access uses fault-handling copy helpers. A
post-submit copy fault resets the session because work may already be accepted.

The device sees only `vzalloc`-owned pages through Linux scatterlists and the
Virtio transport's DMA mapping. Arenas are page-aligned; SG arrays are sized for
`ceil(capacity / PAGE_SIZE)`. Both maximum arenas together fit the advertised
chain bound and queue capacity. Buffers never resize while published. Responses
are zeroed before reuse. Invalid used lengths lose the session; even a lying
length cannot reveal a previous response.

## Synchronization and ownership

One mutex serializes queue submission/reclaim, close and removal. Only one
request is published at a time. The IRQ callback only signals completion; it
never mutates the queue. Completion is reinitialized before rechecking the
atomic removal flag, preventing a lost removal wake-up.

Registration and each successful open hold a `kref`. `misc_deregister`
synchronizes opening paths before registration ownership drops. Removal sets
its flag and wakes a blocked ioctl before waiting for the submission mutex.
Reset, unused-buffer detach and `del_vqs` precede freeing published storage;
the transport synchronizes device access and callbacks. Removed fds return
`ENODEV`. Their final close frees arenas without dereferencing a removed Virtio
device. `file_operations.owner` pins the module while fds remain open.

Reset guarantees depend on the reviewed QEMU reset-forwarding patch and host
reset contract. An indefinitely hung VMM is not a bounded reset service.

## Resource scope and failure

The node grants one whole-device session; it does not pretend to mediate raw
object IDs between competing processes. Final close resets all objects. The host
enforces protocol ownership and resource budgets. Python reference-counts one
fd per device and serializes execution. A failed/uncertain run closes that fd
and invalidates all sharing executables without replay. Event retirement always
precedes output reads. Queue, program, buffers and context are released in order;
a failed release poisons the session and resets it.

The completion wait is bounded and uninterruptible to prevent libc/CPython EINTR
retries of accepted mutations. Its limit bounds waiting for completion, not
subsequent synchronous reset. Process exit still waits for safe quiescence.
Python exceptions cause session quarantine. Inherited executables check their
PID before taking inherited locks; forked children must exec before device use.
Suspend, migration and reconnect are unsupported. The provisional device/ioctl
numbers are not upstream allocations.

## Evidence

The boot test covers exclusive open, bad ioctl arguments, page-backed SG,
large transfers, stale IDs, injected timeout and removal with an open fd.
Python unit tests inject malformed completions, transport failures and object
errors and assert no replay. CI builds a guest-only wheel and executes TOSA
inside QEMU TCG against an explicit OpenVINO CPU host. Hardware validation is
separate evidence, not a replacement for these gates.

//! Native XDNA backend over the HRX runtime (`libhrx`).
//!
//! This module owns every call across the `ffi` boundary. `SAFETY.md` is the audit of record; each
//! `unsafe` block below carries a local `SAFETY:` note. It implements the execution-model spec
//! (issue #85): one audited process-wide device owner, a per-instance stream driven by one
//! serialized dispatch worker, a bounded preallocated submission ring and event-slot pool, and the
//! `hrx_buffer` primitives (persistent host mapping, range flush/invalidate, release).
//!
//! `load_program` accepts the crate-local precompiled artifact format directly, and a TOSA
//! artifact by admitting it (`lower`) and compiling it with the bounded helper subprocess
//! (`compiler`); it builds an `hrx_amdxdna_executable` and dispatches it through the worker.
//! Finite timeouts are rejected before admission (no cancellation exists); a synchronize error
//! latches the event `Failed` and poisons the instance (device-loss tier 1). A 120-second watchdog
//! detects a synchronize call that outlives the kernel's 60-second TDR (tier 2), poisons the
//! instance, and quarantines the still-pending job until the backend is discarded.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::collections::VecDeque;
use std::ffi::CString;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};

use crate::artifact::{self, PrecompiledArtifact};
use crate::compiler::Compiler;
use crate::ffi;
use crate::lower;
use crate::{InitError, XDNA_ERROR_DOMAIN};

/// Upper bound advertised for a loaded artifact (mirrors the OpenVINO backend).
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// Default submission-ring depth: up to four requests outstanding per instance. Issue #85 started
/// at one (matching the Hexagon backend); pipelined dispatch (#151/#149) raised it so consecutive
/// submissions overlap on the stream and the per-submission host cost amortizes.
const DEFAULT_RING_DEPTH: usize = 4;

/// Completion waits poll the stream timeline in bounded slices so the worker observes a watchdog
/// wedge verdict promptly and no HRX call blocks unboundedly.
const WAIT_SLICE_NS: u64 = 100_000_000;

/// AIE DMA descriptors transfer whole four-byte words, so a bound range must begin on a word
/// boundary as well as cover its slot's exact byte length. The base mapping is page-aligned, so the
/// caller's offset alone decides alignment (the reference backend rejects the same way, against its
/// per-slot scalar size).
const DMA_WORD_BYTES: u64 = 4;

/// The kernel's NPU TDR is 60 seconds. A longer userspace watchdog distinguishes a returned
/// device-loss error (tier 1) from an HRX synchronize call that never returns (tier 2).
const DEFAULT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(120);

/// Internal state-machine failure: a live event handle observed a free/unknown slot state.
const INVALID_EVENT_STATE_CODE: i64 = 200;

/// Consume an HRX status and map it to a `BackendError`.
///
/// A non-NULL `hrx_status_t` is owned by the caller and must be consumed exactly once. This reads
/// its code, then ignores it (freeing it); on error it also renders and frees the message so the
/// message allocation is released (it is not logged — no logging facility exists at this layer).
/// `NULL` is success.
fn check(status: ffi::hrx_status_t) -> Result<(), BackendError> {
    if status.is_null() {
        return Ok(());
    }
    // SAFETY: `status` is a non-NULL status just returned by an HRX call; reading its code and
    // ignoring it are the documented consume operations and are valid exactly once.
    let code = unsafe { ffi::hrx_status_code(status) };
    // SAFETY: out-pointers are valid locals; `hrx_status_to_string` returns its own status which we
    // ignore, and any produced message is freed with `hrx_status_free_message`.
    unsafe {
        let mut message: *mut core::ffi::c_char = ptr::null_mut();
        let mut length: usize = 0;
        let string_status = ffi::hrx_status_to_string(status, &mut message, &mut length);
        if !message.is_null() {
            ffi::hrx_status_free_message(message);
        }
        ffi::hrx_status_ignore(string_status);
        ffi::hrx_status_ignore(status);
    }
    Err(backend_error_from_code(code))
}

/// The two `hrx_status_code_t` values the retire loop branches on by value.
mod code {
    pub(super) const OK: super::ffi::hrx_status_code_t = 0;
    pub(super) const DEADLINE_EXCEEDED: super::ffi::hrx_status_code_t = 4;
}

/// Map an HRX status code (`hrx_status_code_t`, mirroring IREE) to a `BackendError`.
fn backend_error_from_code(code: ffi::hrx_status_code_t) -> BackendError {
    match code {
        3 => BackendError::InvalidArgument,  // INVALID_ARGUMENT
        4 => BackendError::DeadlineExpired,  // DEADLINE_EXCEEDED
        7 => BackendError::PermissionDenied, // PERMISSION_DENIED
        8 => BackendError::OutOfMemory,      // OUT_OF_MEMORY (RESOURCE_EXHAUSTED)
        11 => BackendError::OutOfBounds,     // OUT_OF_RANGE
        12 => BackendError::Unsupported,     // UNIMPLEMENTED
        14 => BackendError::DeviceLost,      // UNAVAILABLE
        other => BackendError::External {
            domain: XDNA_ERROR_DOMAIN,
            code: other as i64,
        },
    }
}

/// The process-wide HRX device, initialized once and never shut down (the fork's model: one
/// device per process, `hrx_gpu_shutdown` is never called).
struct SharedDevice(NonNull<ffi::hrx_device_s>);

// SAFETY: the device handle is refcounted and, in HRX's amdxdna HAL, is a process-wide singleton
// reachable from any thread. This module uses it only to create per-instance streams and
// executables (read-only factory use), never mutating it; sharing the pointer across threads for
// that use is sound. Concurrency hazards live on the stream, which is serialized (see `Lane`).
unsafe impl Send for SharedDevice {}
unsafe impl Sync for SharedDevice {}

fn shared_device() -> Result<&'static SharedDevice, InitError> {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Result<SharedDevice, InitError>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            // SAFETY: HRX GPU initialization is process-wide and idempotent under the OnceLock;
            // the out-pointers are valid locals and every returned status is consumed by `check`.
            unsafe {
                check(ffi::hrx_gpu_initialize(0)).map_err(|_| InitError::Initialization)?;
                let mut count: core::ffi::c_int = 0;
                check(ffi::hrx_gpu_device_count(&mut count))
                    .map_err(|_| InitError::Initialization)?;
                if count < 1 {
                    return Err(InitError::DeviceUnavailable);
                }
                let mut device: ffi::hrx_device_t = ptr::null_mut();
                check(ffi::hrx_gpu_device_get(0, &mut device))
                    .map_err(|_| InitError::Initialization)?;
                NonNull::new(device)
                    .map(SharedDevice)
                    .ok_or(InitError::Initialization)
            }
        })
        .as_ref()
        .map_err(|error| *error)
}

/// An owned HRX stream. Access is serialized by the `Lane` mutex.
struct Stream(NonNull<ffi::hrx_stream_s>);

// SAFETY: the stream pointer is only ever dereferenced while the `Lane` stream mutex is held, so it
// is never used concurrently; moving it between the constructing thread and the worker is sound.
unsafe impl Send for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the lane owns exactly one stream reference. A normal teardown joins the worker;
        // a detached tier-2 worker itself holds the last lane Arc, so this Drop still cannot run
        // until its HRX call has returned and no dispatch is in progress.
        unsafe { ffi::hrx_stream_release(self.0.as_ptr()) };
    }
}

/// Terminal-state codes latched into an [`EventSlot`].
mod slot_state {
    pub const FREE: u8 = 0;
    pub const PENDING: u8 = 1;
    pub const COMPLETE: u8 = 2;
    pub const FAILED: u8 = 3;
}

/// One preallocated event slot: a latched terminal state plus the failure detail.
#[derive(Debug)]
struct EventSlot {
    state: AtomicU8,
    error: Mutex<Option<BackendError>>,
}

/// Provider-owned resources that are currently live for one backend instance.
///
/// This is intentionally backend-local rather than part of the `Accelerator` ABI. The shared
/// conformance hook translates it to its own `ResourceCounts` type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XdnaResourceCounts {
    pub contexts: u64,
    pub buffers: u64,
    pub programs: u64,
    pub queues: u64,
    pub events: u64,
}

#[derive(Debug, Default)]
struct ResourceTracker {
    contexts: AtomicU64,
    buffers: AtomicU64,
    programs: AtomicU64,
    queues: AtomicU64,
    events: AtomicU64,
}

impl ResourceTracker {
    fn snapshot(&self) -> XdnaResourceCounts {
        XdnaResourceCounts {
            contexts: self.contexts.load(Ordering::Acquire),
            buffers: self.buffers.load(Ordering::Acquire),
            programs: self.programs.load(Ordering::Acquire),
            queues: self.queues.load(Ordering::Acquire),
            events: self.events.load(Ordering::Acquire),
        }
    }

    fn decrement(counter: &AtomicU64) {
        let previous = counter.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "XDNA resource counter underflow");
    }
}

impl EventSlot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(slot_state::FREE),
            error: Mutex::new(None),
        })
    }

    fn latch(&self, result: Result<(), BackendError>) {
        match result {
            Ok(()) => self.state.store(slot_state::COMPLETE, Ordering::Release),
            Err(error) => {
                *self.error.lock().expect("event error mutex") = Some(error);
                self.state.store(slot_state::FAILED, Ordering::Release);
            }
        }
    }
}

/// One queued dispatch handed to the worker. The raw pointers are HRX handles — heap objects HRX
/// owns, whose addresses are independent of where the caller's `XdnaBuffer`/`XdnaProgram` values
/// live. The accompanying Arcs keep those handles referenced for the full worker operation, even
/// when a poisoned backend is discarded, and the worker touches them only under the stream mutex.
/// The in-flight gates live in the same Arcs, not in caller-movable structs.
struct Job {
    program: Arc<ProgramInner>,
    bindings: Vec<ffi::hrx_buffer_ref_t>,
    outputs: Vec<(ffi::hrx_buffer_t, usize, usize)>,
    /// Retains each distinct HRX allocation and supplies its address-stable in-flight gate.
    buffers: Vec<Arc<BufferInner>>,
    slot: Arc<EventSlot>,
}

// SAFETY: the raw pointers in a `Job` are HRX buffer/executable handles: heap objects owned by the
// HRX runtime whose addresses do not change when the caller moves its Rust-side handle values.
// Their accompanying Arcs keep them referenced until the worker has stopped touching them, and
// they are dereferenced only on the worker thread while the stream mutex is held. Moving the job to
// the worker transfers that exclusive access. Everything else in the job (`Arc`s) is `Send`.
unsafe impl Send for Job {}

/// The bounded submission ring. `queue` holds accepted jobs the worker has not yet dispatched;
/// `dispatched` counts jobs the worker has moved onto the device but not yet retired. Their sum is
/// bounded by the lane depth, so `submit` rejects with `Busy` exactly when `depth` submissions are
/// outstanding in any mix of queued and in-flight.
struct Ring {
    queue: VecDeque<Job>,
    dispatched: usize,
    stopping: bool,
}

#[derive(Debug)]
struct WatchdogState {
    generation: u64,
    active: Option<(u64, Instant)>,
    stopping: bool,
}

#[cfg(feature = "test-control")]
#[derive(Clone, Copy, Debug)]
enum InjectedFault {
    Tier1,
    Tier2 { stall: Duration },
    HoldDispatch { hold: Duration },
}

/// One-shot fault used by the on-metal fault-path tests.
#[cfg(feature = "test-control")]
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub enum XdnaTestFault {
    /// Return a definite device-loss error from the worker before dispatching to HRX.
    Tier1,
    /// Hold the worker beyond the watchdog deadline without touching HRX.
    Tier2 { stall: Duration },
    /// Delay the worker before one normal dispatch (no fault): a deterministic pending window for
    /// tests that assert in-flight semantics, which are otherwise a race against real completion.
    HoldDispatch { hold: Duration },
}

/// Test-only construction parameters. This API is absent unless `test-control` is enabled.
#[cfg(feature = "test-control")]
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct XdnaTestConfig {
    pub watchdog_timeout: Duration,
    pub fault: XdnaTestFault,
}

/// One instance's serialized dispatch lane: the stream, the ring, and the poison flag.
struct Lane {
    stream: Mutex<Stream>,
    ring: Mutex<Ring>,
    signal: Condvar,
    poisoned: AtomicBool,
    wedged: AtomicBool,
    watchdog: Mutex<WatchdogState>,
    watchdog_signal: Condvar,
    watchdog_timeout: Duration,
    #[cfg(feature = "test-control")]
    injected_fault: Mutex<Option<InjectedFault>>,
    depth: usize,
}

impl Lane {
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    fn arm_watchdog(&self) -> u64 {
        let mut watchdog = self.watchdog.lock().expect("watchdog mutex");
        watchdog.generation = watchdog.generation.wrapping_add(1);
        let generation = watchdog.generation;
        watchdog.active = Some((generation, Instant::now() + self.watchdog_timeout));
        self.watchdog_signal.notify_one();
        generation
    }

    /// Disarm one dispatch. `false` means the watchdog already declared this lane wedged.
    fn disarm_watchdog(&self, generation: u64) -> bool {
        let mut watchdog = self.watchdog.lock().expect("watchdog mutex");
        if matches!(watchdog.active, Some((active, _)) if active == generation) {
            watchdog.active = None;
        }
        self.watchdog_signal.notify_one();
        !self.wedged.load(Ordering::Acquire)
    }

    fn run_watchdog(self: &Arc<Self>) {
        let mut watchdog = self.watchdog.lock().expect("watchdog mutex");
        loop {
            if watchdog.stopping {
                return;
            }
            let Some((generation, deadline)) = watchdog.active else {
                watchdog = self
                    .watchdog_signal
                    .wait(watchdog)
                    .expect("watchdog condvar");
                continue;
            };
            let now = Instant::now();
            if now < deadline {
                let (next, _) = self
                    .watchdog_signal
                    .wait_timeout(watchdog, deadline - now)
                    .expect("watchdog condvar");
                watchdog = next;
                continue;
            }
            if matches!(watchdog.active, Some((active, _)) if active == generation) {
                watchdog.active = None;
                self.wedged.store(true, Ordering::Release);
                self.poisoned.store(true, Ordering::Release);
                return;
            }
        }
    }

    /// Run the worker loop: keep up to `depth` jobs in flight on the stream, retiring the oldest
    /// while later submissions are already recorded and executing. Dispatch (record + flush +
    /// timeline position) holds the stream mutex briefly; the completion wait blocks on the
    /// stream's timeline semaphore with no lock held, so `allocate_buffer` never queues behind a
    /// running dispatch. In-flight jobs are worker-owned: on a tier-2 wedge the worker parks
    /// forever holding them (and the `Arc<Lane>` stream), forming the quarantine.
    fn run_worker(self: &Arc<Self>) {
        let mut in_flight: VecDeque<(Job, ffi::hrx_timeline_point_t)> = VecDeque::new();
        // The timeline target of the most recently dispatched job. The stream is instance-private
        // and dispatches are serialized under the stream mutex, so each job's target is exactly
        // the first timeline value observed past its predecessor's (see `dispatch_job`).
        let mut last_target: u64 = 0;
        loop {
            // Fill: move queued jobs onto the device up to the lane depth.
            loop {
                let job = {
                    let mut ring = self.ring.lock().expect("ring mutex");
                    if in_flight.len() >= self.depth {
                        None
                    } else if let Some(job) = ring.queue.pop_front() {
                        ring.dispatched += 1;
                        Some(job)
                    } else {
                        None
                    }
                };
                let Some(job) = job else { break };
                match self.dispatch_job(job, &mut last_target) {
                    DispatchOutcome::InFlight(entry) => in_flight.push_back(entry),
                    // Terminal at dispatch (error, poisoned short-circuit, or an injected
                    // fault). `finish` released the ring capacity where a terminal state was
                    // latched; the deliberately-pending outcomes (untrusted boundary, injected
                    // tier 2) keep their slot and capacity, as quarantined work must.
                    DispatchOutcome::Retired => {}
                }
            }
            // Retire the oldest in-flight job, or sleep until there is work.
            if let Some((job, position)) = in_flight.pop_front() {
                if !self.retire(job, position) {
                    // Tier-2 wedge: no trustworthy completion boundary exists for this job or
                    // anything recorded behind it. Park forever holding every in-flight job's
                    // retained resources and the stream; `Drop` detaches this thread.
                    loop {
                        std::thread::sleep(Duration::from_secs(3600));
                    }
                }
                continue;
            }
            let mut ring = self.ring.lock().expect("ring mutex");
            loop {
                if !ring.queue.is_empty() {
                    break;
                }
                if ring.stopping {
                    return;
                }
                ring = self.signal.wait(ring).expect("ring condvar");
            }
        }
    }

    /// Dispatch one job: record it on the stream, flush, and capture its timeline position. On a
    /// definite error the job is latched `Failed` here and the instance poisons (device-loss
    /// tier 1); on a poisoned lane the job short-circuits to `Failed(DeviceLost)` without touching
    /// the stream. Injected test faults are consumed here, before any HRX call.
    fn dispatch_job(&self, job: Job, last_target: &mut u64) -> DispatchOutcome {
        #[cfg(feature = "test-control")]
        {
            let injected = self
                .injected_fault
                .lock()
                .expect("fault-injector mutex")
                .take();
            match injected {
                Some(InjectedFault::Tier2 { stall }) => {
                    let watchdog_generation = self.arm_watchdog();
                    std::thread::sleep(stall);
                    let _ = self.disarm_watchdog(watchdog_generation);
                    // Tier 2 has no trustworthy completion boundary: leave the event pending and
                    // every gate armed. Discarding the backend quarantines this job's native
                    // resources. (No HRX call was made, so dropping the job itself is safe.)
                    return DispatchOutcome::Retired;
                }
                Some(InjectedFault::Tier1) => {
                    let generation = self.arm_watchdog();
                    let trusted = self.disarm_watchdog(generation);
                    if trusted {
                        self.finish(job, Err(BackendError::DeviceLost));
                    }
                    return DispatchOutcome::Retired;
                }
                Some(InjectedFault::HoldDispatch { hold }) => {
                    // A deterministic pending window: the event and every in-flight gate stay
                    // armed for at least `hold`, then the job proceeds normally.
                    std::thread::sleep(hold);
                }
                None => {}
            }
        }
        // A poisoned lane refuses the stream: jobs accepted before the poison latch terminally.
        if self.is_poisoned() {
            self.finish(job, Err(BackendError::DeviceLost));
            return DispatchOutcome::Retired;
        }

        let config = ffi::hrx_dispatch_config_t {
            workgroup_count: [1, 1, 1],
            workgroup_size: [1, 1, 1],
            subgroup_size: 0,
        };
        let program = job.program.as_ref();
        let stream = self.stream.lock().expect("stream mutex");
        // The watchdog covers the record/flush ioctls too: a driver hang here is the same
        // ownership-boundary loss as one during the completion wait.
        let watchdog_generation = self.arm_watchdog();
        // SAFETY: the stream, executable, and every bound buffer are retained by the lane/job for
        // this call; the bindings slice is valid for `binding_count`; the config is a valid local;
        // constants are unused on this path.
        let dispatch = unsafe {
            ffi::hrx_stream_dispatch(
                stream.0.as_ptr(),
                program.executable.as_ptr(),
                program.ordinal,
                &config,
                ptr::null(),
                0,
                job.bindings.as_ptr(),
                job.bindings.len(),
                0,
            )
        };
        let result = check(dispatch).and_then(|()| {
            // SAFETY: the stream is live; flush submits the recorded work without waiting.
            check(unsafe { ffi::hrx_stream_flush(stream.0.as_ptr()) })?;
            // The flushed batch's timeline value is assigned asynchronously: a position read
            // before the dispatch (or immediately after the flush, on an unlucky schedule) still
            // reports the previous batch's target, and waiting on that retires this job early --
            // observed on metal as stale outputs. The stream is instance-private and dispatches
            // are serialized under this mutex, so only this flush can advance the timeline past
            // the previous job's target: spin until it does, and that value is exactly this
            // job's completion tick. The watchdog is armed, so a driver that never assigns is
            // declared wedged rather than spun on forever.
            loop {
                let mut position = ffi::hrx_timeline_point_t {
                    semaphore: ptr::null_mut(),
                    value: 0,
                };
                // SAFETY: the stream is live; the out-pointer is a valid local.
                check(unsafe {
                    ffi::hrx_stream_get_timeline_position(stream.0.as_ptr(), &mut position)
                })?;
                if position.value > *last_target {
                    *last_target = position.value;
                    break Ok(position);
                }
                if self.wedged.load(Ordering::Acquire) {
                    break Err(BackendError::DeviceLost);
                }
                std::thread::yield_now();
            }
        });
        drop(stream);
        let trusted = self.disarm_watchdog(watchdog_generation);
        if !trusted {
            // The ioctl returned only after the watchdog declared the boundary lost: keep the
            // event pending, keep the gates armed; the caller's teardown quarantines.
            return DispatchOutcome::Retired;
        }
        match result {
            Ok(position) => DispatchOutcome::InFlight((job, position)),
            Err(error) => {
                self.finish(job, Err(error));
                DispatchOutcome::Retired
            }
        }
    }

    /// Wait for one in-flight job's timeline position and latch its terminal state. Returns
    /// `false` when the watchdog declared the lane wedged while waiting (tier 2): the caller must
    /// quarantine, because the device may still write through this and every later in-flight
    /// job's bindings.
    fn retire(&self, job: Job, position: ffi::hrx_timeline_point_t) -> bool {
        let watchdog_generation = self.arm_watchdog();
        let result = loop {
            // Bounded slices keep this wait off the stream mutex and let the watchdog's verdict
            // surface promptly; the semaphore is the stream's own timeline object, valid while
            // the lane retains the stream, and safe to wait on while another thread holds the
            // stream mutex (it is a standalone synchronization object).
            // SAFETY: `position.semaphore` is the live stream timeline semaphore (see above); the
            // status is consumed.
            let status = unsafe {
                ffi::hrx_semaphore_wait(position.semaphore, position.value, WAIT_SLICE_NS)
            };
            let code = unsafe { ffi::hrx_status_code(status) };
            unsafe { ffi::hrx_status_ignore(status) };
            match code {
                code::OK => break Ok(()),
                code::DEADLINE_EXCEEDED => {
                    if self.wedged.load(Ordering::Acquire) {
                        // Watchdog verdict while we sliced: same as disarm returning untrusted.
                        return false;
                    }
                }
                other => break Err(backend_error_from_code(other)),
            }
        };
        let trusted = self.disarm_watchdog(watchdog_generation);
        if !trusted {
            return false;
        }
        let result = result.and_then(|()| {
            for &(buffer, offset, len) in &job.outputs {
                // SAFETY: each output buffer is live and persistently mapped; the range was
                // validated at submit; invalidate makes device writes host-visible.
                check(unsafe { ffi::hrx_buffer_invalidate_range(buffer, offset, len) })?;
            }
            Ok(())
        });
        self.finish(job, result);
        true
    }

    /// Publish a job's terminal state: clear the in-flight gates, latch the event, and poison the
    /// instance on failure (device-loss tier 1).
    fn finish(&self, job: Job, result: Result<(), BackendError>) {
        let failed = result.is_err();
        // Release this job's ring capacity before the terminal state becomes observable: a caller
        // that polls `Complete`, destroys the event, and immediately resubmits must never bounce
        // off a stale `dispatched` count.
        {
            let mut ring = self.ring.lock().expect("ring mutex");
            ring.dispatched -= 1;
        }
        // Clear the in-flight gates before publishing the terminal state, so a caller that observes
        // completion may immediately read or free the buffers. The retained `Arc`s remain valid
        // regardless of where (or whether) the caller's buffer values still live.
        for buffer in &job.buffers {
            buffer.in_flight.store(0, Ordering::Release);
        }
        // Latch the terminal state before poisoning, so this event reports its real error rather
        // than the instance-level `DeviceLost`.
        job.slot.latch(result);
        if failed {
            // Device-loss tier 1: the kernel TDR (or a definite HRX error) has returned; the
            // instance refuses further work (libhrx is not validated to remain healthy after an
            // under-the-hood context recreate).
            self.poisoned.store(true, Ordering::Release);
        }
    }
}

/// What became of one job handed to `dispatch_job`.
enum DispatchOutcome {
    /// Recorded and flushed; retire it at this timeline position.
    InFlight((Job, ffi::hrx_timeline_point_t)),
    /// Terminal at dispatch: latched (error paths) or deliberately left pending (untrusted
    /// boundary and the injected tier-2 fault).
    Retired,
}

/// One HRX backend instance: a serialized dispatch lane over the shared device.
pub struct XdnaAccelerator {
    lane: Arc<Lane>,
    worker: Option<JoinHandle<()>>,
    watchdog: Option<JoinHandle<()>>,
    slots: Vec<Arc<EventSlot>>,
    compiler: CompilerState,
    info: DeviceInfo,
    next_id: AtomicU64,
    resources: Arc<ResourceTracker>,
    /// Cumulative count of buffers admitted as direct bindings (no submission-time staging copy).
    direct_binding_admissions: AtomicU64,
    /// Cumulative bytes moved by explicit `write_buffer`/`read_buffer` transfers.
    explicit_transfer_bytes: AtomicU64,
    _not_send_sync: PhantomData<*mut u8>,
}

enum CompilerState {
    Ready(Compiler),
    Unconfigured,
    Failed(BackendError),
}

impl XdnaAccelerator {
    /// Initialize the shared device (once per process), create this instance's stream, and start
    /// its dispatch worker.
    pub fn new() -> Result<Self, InitError> {
        #[cfg(feature = "test-control")]
        return Self::initialize(DEFAULT_WATCHDOG_TIMEOUT, None);
        #[cfg(not(feature = "test-control"))]
        Self::initialize(DEFAULT_WATCHDOG_TIMEOUT)
    }

    /// Construct a backend with one deterministic fault and a shortened watchdog.
    #[cfg(feature = "test-control")]
    #[doc(hidden)]
    pub fn new_for_testing(config: XdnaTestConfig) -> Result<Self, InitError> {
        if config.watchdog_timeout.is_zero() {
            return Err(InitError::Initialization);
        }
        let fault = match config.fault {
            XdnaTestFault::Tier1 => InjectedFault::Tier1,
            XdnaTestFault::Tier2 { stall } if stall > config.watchdog_timeout => {
                InjectedFault::Tier2 { stall }
            }
            XdnaTestFault::Tier2 { .. } => return Err(InitError::Initialization),
            // The hold must stay well inside the watchdog deadline: it delays a healthy
            // dispatch, it does not simulate a hang.
            XdnaTestFault::HoldDispatch { hold }
                if !hold.is_zero() && hold * 2 <= config.watchdog_timeout =>
            {
                InjectedFault::HoldDispatch { hold }
            }
            XdnaTestFault::HoldDispatch { .. } => return Err(InitError::Initialization),
        };
        Self::initialize(config.watchdog_timeout, Some(fault))
    }

    fn initialize(
        watchdog_timeout: Duration,
        #[cfg(feature = "test-control")] fault: Option<InjectedFault>,
    ) -> Result<Self, InitError> {
        let device = shared_device()?;
        let mut stream: ffi::hrx_stream_t = ptr::null_mut();
        // SAFETY: `device.0` is the live process-wide device; the out-pointer is a valid local and
        // the returned status is consumed by `check`.
        let status = unsafe { ffi::hrx_stream_create(device.0.as_ptr(), 0, &mut stream) };
        check(status).map_err(|_| InitError::Initialization)?;
        let stream = NonNull::new(stream).ok_or(InitError::Initialization)?;

        let depth = DEFAULT_RING_DEPTH;
        let lane = Arc::new(Lane {
            stream: Mutex::new(Stream(stream)),
            ring: Mutex::new(Ring {
                queue: VecDeque::with_capacity(depth),
                dispatched: 0,
                stopping: false,
            }),
            signal: Condvar::new(),
            poisoned: AtomicBool::new(false),
            wedged: AtomicBool::new(false),
            watchdog: Mutex::new(WatchdogState {
                generation: 0,
                active: None,
                stopping: false,
            }),
            watchdog_signal: Condvar::new(),
            watchdog_timeout,
            #[cfg(feature = "test-control")]
            injected_fault: Mutex::new(fault),
            depth,
        });
        let slots = (0..depth).map(|_| EventSlot::new()).collect();

        let watchdog_lane = Arc::clone(&lane);
        let watchdog = std::thread::Builder::new()
            .name("xdna-watchdog".into())
            .spawn(move || watchdog_lane.run_watchdog())
            .map_err(|_| InitError::Initialization)?;

        let worker_lane = Arc::clone(&lane);
        let worker = match std::thread::Builder::new()
            .name("xdna-dispatch".into())
            .spawn(move || worker_lane.run_worker())
        {
            Ok(worker) => worker,
            Err(_) => {
                {
                    let mut state = lane.watchdog.lock().expect("watchdog mutex");
                    state.stopping = true;
                }
                lane.watchdog_signal.notify_all();
                let _ = watchdog.join();
                return Err(InitError::Initialization);
            }
        };

        let compiler = match Compiler::from_env() {
            Ok(compiler) => CompilerState::Ready(compiler),
            Err(_error) if std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_none() => {
                CompilerState::Unconfigured
            }
            Err(error) => CompilerState::Failed(error),
        };

        Ok(Self {
            lane,
            worker: Some(worker),
            watchdog: Some(watchdog),
            slots,
            compiler,
            info: device_info(),
            next_id: AtomicU64::new(1),
            resources: Arc::new(ResourceTracker::default()),
            direct_binding_admissions: AtomicU64::new(0),
            explicit_transfer_bytes: AtomicU64::new(0),
            _not_send_sync: PhantomData,
        })
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Cumulative count of buffers admitted as direct bindings (no submission-time staging copy).
    ///
    /// Every XDNA buffer is a persistently mapped, device-visible allocation bound straight into
    /// the dispatch with no bounce buffer, so a submission admits `bindings.len()` direct bindings
    /// and stages none. Surfaced to the conformance transfer/staging diagnostics hook (issue #90).
    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.load(Ordering::Relaxed)
    }

    /// Cumulative bytes moved by explicit `write_buffer`/`read_buffer` transfers.
    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.explicit_transfer_bytes.load(Ordering::Relaxed)
    }

    /// Snapshot provider-owned resources that have been admitted but not successfully released.
    pub fn resource_counts(&self) -> XdnaResourceCounts {
        self.resources.snapshot()
    }

    /// Claim a free event slot, transitioning it to `PENDING`.
    fn claim_slot(&self) -> Option<Arc<EventSlot>> {
        self.slots.iter().find_map(|slot| {
            slot.state
                .compare_exchange(
                    slot_state::FREE,
                    slot_state::PENDING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
                .then(|| Arc::clone(slot))
        })
    }
}

impl Drop for XdnaAccelerator {
    fn drop(&mut self) {
        // Signal both threads to stop. A normally quiescent worker is joined before the stream is
        // released. If an accepted event is still pending (including a tier-2 wedge), detach the
        // worker instead: its `Arc<Lane>` and the job's retained native resources form the
        // quarantine and make discarding the poisoned instance nonblocking and memory-safe.
        {
            let mut ring = self.lane.ring.lock().expect("ring mutex");
            ring.stopping = true;
        }
        self.lane.signal.notify_all();
        {
            let mut watchdog = self.lane.watchdog.lock().expect("watchdog mutex");
            watchdog.stopping = true;
        }
        self.lane.watchdog_signal.notify_all();
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
        let has_pending = self
            .slots
            .iter()
            .any(|slot| slot.state.load(Ordering::Acquire) == slot_state::PENDING);
        if let Some(worker) = self.worker.take() {
            if !has_pending && !self.lane.wedged.load(Ordering::Acquire) {
                let _ = worker.join();
            }
        }
    }
}

/// A device-neutral description of the XDNA NPU.
fn device_info() -> DeviceInfo {
    DeviceInfo {
        identity: DeviceIdentity {
            uuid: *b"amd.xdna.npu\0\0\0\0",
            class: AcceleratorClass::NPU,
            vendor_id: 0x1022,
            device_id: 0x17f0,
        },
        capabilities: Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY,
        // Finite aggregate bounds: the device-state layer checked-multiplies `max_contexts` by each
        // per-context limit, so `u32::MAX` would overflow and reject the backend. These mirror the
        // OpenVINO backend's advertised limits.
        limits: DeviceLimits {
            max_contexts: 64,
            max_buffers_per_context: 1_024,
            max_programs_per_context: 64,
            max_queues_per_context: 64,
            max_events_per_context: 4_096,
            max_bindings_per_submission: 256,
            max_buffer_bytes: 16 * 1024 * 1024 * 1024,
            max_artifact_bytes: MAX_TOSA_ARTIFACT_BYTES,
        },
    }
}

/// Create and load an amdxdna executable from a parsed precompiled artifact.
fn build_executable(
    device: &SharedDevice,
    parsed: PrecompiledArtifact<'_>,
    context_id: u64,
    resources: &Arc<ResourceTracker>,
) -> Result<XdnaProgram, BackendError> {
    // Build the borrowed create-params (valid only for the duration of the create call).
    let xclbin_span = ffi::hrx_const_byte_span_t {
        data: parsed.xclbin.as_ptr().cast(),
        data_length: parsed.xclbin.len(),
    };
    let run = ffi::hrx_amdxdna_executable_run_t {
        record_length: size_of::<ffi::hrx_amdxdna_executable_run_t>() as u32,
        abi_version: ffi::HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0,
        transaction: ffi::hrx_const_byte_span_t {
            data: parsed.insts.as_ptr().cast(),
            data_length: parsed.insts.len(),
        },
        data_payload: ffi::hrx_const_byte_span_t {
            data: ptr::null(),
            data_length: 0,
        },
    };
    let entry = ffi::hrx_amdxdna_executable_entry_point_t {
        record_length: size_of::<ffi::hrx_amdxdna_executable_entry_point_t>() as u32,
        abi_version: ffi::HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0,
        name: ffi::hrx_string_view_t {
            data: parsed.entry.as_ptr().cast(),
            size: parsed.entry.len(),
        },
        context_mode: ffi::HRX_AMDXDNA_CONTEXT_MODE_CREATE,
        xclbin_ordinal: 0,
        pdi_ordinal: 0,
        source_line: 0,
        source_file: ffi::hrx_string_view_t {
            data: ptr::null(),
            size: 0,
        },
        runs: &run,
        run_count: 1,
    };
    let params = ffi::hrx_amdxdna_executable_create_params_t {
        record_length: size_of::<ffi::hrx_amdxdna_executable_create_params_t>() as u32,
        abi_version: ffi::HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0,
        flags: 0,
        reserved: 0,
        xclbins: &xclbin_span,
        xclbin_count: 1,
        entry_points: &entry,
        entry_point_count: 1,
    };

    let mut executable: ffi::hrx_executable_t = ptr::null_mut();
    // SAFETY: the device is live; every span/record referenced by `params` is borrowed from
    // `parsed`, which outlives this call; the out-pointer is a valid local; status consumed. HRX
    // copies what it needs and the borrows may be released after return.
    let status =
        unsafe { ffi::hrx_amdxdna_executable_create(device.0.as_ptr(), &params, &mut executable) };
    check(status)?;
    let executable = NonNull::new(executable).ok_or(BackendError::External {
        domain: XDNA_ERROR_DOMAIN,
        code: 0,
    })?;

    // Resolve the export ordinal by entry-point name.
    let ordinal = CString::new(parsed.entry)
        .map_err(|_| BackendError::InvalidArgument)
        .and_then(|entry_c| {
            let mut ordinal: u32 = 0;
            // SAFETY: the executable is live; `entry_c` is a valid NUL-terminated string; the
            // out-pointer is a valid local; status consumed.
            let status = unsafe {
                ffi::hrx_executable_lookup_export_by_name(
                    executable.as_ptr(),
                    entry_c.as_ptr(),
                    &mut ordinal,
                )
            };
            check(status).map(|()| ordinal)
        });
    let ordinal = match ordinal {
        Ok(ordinal) => ordinal,
        Err(error) => {
            // SAFETY: the executable is live and owned here; release it before returning.
            unsafe { ffi::hrx_executable_release(executable.as_ptr()) };
            return Err(error);
        }
    };

    resources.programs.fetch_add(1, Ordering::AcqRel);
    Ok(XdnaProgram {
        inner: Arc::new(ProgramInner {
            executable,
            ordinal,
            inputs: parsed.inputs,
            outputs: parsed.outputs,
            slot_bytes: parsed.slot_bytes,
            resources: Arc::clone(resources),
        }),
        context_id,
    })
}

/// A logical execution context. HRX holds no per-context state; the id gives resources a home so
/// cross-context admission can be rejected (queues, programs, and buffers carry their context id).
#[derive(Debug)]
pub struct XdnaContext {
    id: u64,
    resources: Arc<ResourceTracker>,
}

/// A logical execution queue funnelling into the instance's single stream lane.
#[derive(Debug)]
pub struct XdnaQueue {
    context_id: u64,
    resources: Arc<ResourceTracker>,
}

/// A loaded, refcounted amdxdna executable and its dispatch plan.
#[derive(Debug)]
struct ProgramInner {
    executable: NonNull<ffi::hrx_executable_s>,
    ordinal: u32,
    inputs: usize,
    outputs: usize,
    /// Exact per-slot byte sizes (inputs `0..inputs`, then outputs). The compiled TXN stream DMAs
    /// these extents regardless of the bound range length, so submit must enforce an exact match.
    slot_bytes: Vec<u64>,
    resources: Arc<ResourceTracker>,
}

// SAFETY: the executable is refcounted and used only by the worker under the stream mutex, plus
// released once on drop. Sharing the handle across threads (submit clones the `Arc` into a job) is
// sound because access is serialized on the worker.
unsafe impl Send for ProgramInner {}
unsafe impl Sync for ProgramInner {}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of the executable reference (the caller unloaded it and no
        // in-flight job still holds a clone); release drops exactly one reference.
        unsafe { ffi::hrx_executable_release(self.executable.as_ptr()) };
        ResourceTracker::decrement(&self.resources.programs);
    }
}

/// A loaded program: a cheap handle over the shared [`ProgramInner`], tagged with its context.
#[derive(Debug)]
pub struct XdnaProgram {
    inner: Arc<ProgramInner>,
    context_id: u64,
}

/// A submission event: a handle over one preallocated [`EventSlot`].
#[derive(Debug)]
pub struct XdnaEvent {
    slot: Arc<EventSlot>,
    resources: Arc<ResourceTracker>,
}

/// One owned HRX allocation and its persistent mapping.
#[derive(Debug)]
struct BufferInner {
    buffer: NonNull<ffi::hrx_buffer_s>,
    mapped: NonNull<u8>,
    len: usize,
    /// Set while the device may be reading or writing this buffer; guards host access and release.
    in_flight: AtomicU64,
    resources: Arc<ResourceTracker>,
}

// SAFETY: the HRX allocation and persistent mapping are only accessed through XDNA's host-access
// API or by the serialized worker. The in-flight gate excludes those paths from overlapping, and
// the `Arc` keeps the allocation live until the worker has stopped touching it.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: this is the last `Arc` owning one buffer reference. The persistent mapping is
        // released together with the buffer (the fork never unmaps first).
        unsafe { ffi::hrx_buffer_release(self.buffer.as_ptr()) };
        ResourceTracker::decrement(&self.resources.buffers);
    }
}

/// An HRX buffer handle over a refcounted native allocation.
#[derive(Debug)]
pub struct XdnaBuffer {
    inner: Arc<BufferInner>,
    desc: BufferDesc,
    /// The context that allocated this buffer; a submission may only bind buffers from its own.
    context_id: u64,
    _not_send_sync: PhantomData<*mut u8>,
}

impl XdnaBuffer {
    /// Byte range `[offset, offset+len)` checked against the mapping, returning `(start, end)`.
    fn checked_range(&self, offset: u64, len: usize) -> Result<(usize, usize), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(len)
            .filter(|end| *end <= self.inner.len)
            .ok_or(BackendError::OutOfBounds)?;
        Ok((start, end))
    }
}

impl Accelerator for XdnaAccelerator {
    type Context = XdnaContext;
    type Buffer = XdnaBuffer;
    type Program = XdnaProgram;
    type Queue = XdnaQueue;
    type Event = XdnaEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        self.resources.contexts.fetch_add(1, Ordering::AcqRel);
        Ok(XdnaContext {
            id: self.next_id(),
            resources: Arc::clone(&self.resources),
        })
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        ResourceTracker::decrement(&context.resources.contexts);
        Ok(())
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.info.validate_buffer_desc(desc)?;
        if self.lane.is_poisoned() {
            return Err(BackendError::DeviceLost);
        }
        if desc.domain == MemoryDomain::Device {
            return Err(BackendError::Unsupported);
        }
        let size = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfMemory)?;

        let mut buffer: ffi::hrx_buffer_t = ptr::null_mut();
        // SAFETY: the stream is live (held under the lane mutex); the out-pointer is a valid local;
        // the status is consumed. The lock serializes allocation against the worker's dispatch.
        let status = {
            let stream = self.lane.stream.lock().expect("stream mutex");
            unsafe {
                ffi::hrx_buffer_allocate(
                    stream.0.as_ptr(),
                    size,
                    ffi::HRX_MEMORY_TYPE_HOST_LOCAL | ffi::HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                    ffi::HRX_BUFFER_USAGE_DEFAULT | ffi::HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                    &mut buffer,
                )
            }
        };
        check(status)?;
        let buffer = NonNull::new(buffer).ok_or(BackendError::External {
            domain: XDNA_ERROR_DOMAIN,
            code: 0,
        })?;

        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: `buffer` is the just-allocated live buffer; the out-pointer is a valid local; the
        // status is consumed. Mapping does not touch the stream.
        let status = unsafe {
            ffi::hrx_buffer_map_with_mode(
                buffer.as_ptr(),
                ffi::HRX_MAPPING_MODE_PERSISTENT,
                ffi::HRX_MAP_READ | ffi::HRX_MAP_WRITE,
                0,
                size,
                &mut mapped,
            )
        };
        let mapped = match check(status).and_then(|()| {
            NonNull::new(mapped.cast::<u8>()).ok_or(BackendError::External {
                domain: XDNA_ERROR_DOMAIN,
                code: 0,
            })
        }) {
            Ok(mapped) => mapped,
            Err(error) => {
                // SAFETY: `buffer` is live and owned here; release its only reference on the error
                // path before the handle is discarded.
                unsafe { ffi::hrx_buffer_release(buffer.as_ptr()) };
                return Err(error);
            }
        };

        let alignment = 1u64 << (mapped.as_ptr() as usize).trailing_zeros().min(63);
        let info = BufferInfo::new(
            desc,
            size as u64,
            alignment,
            BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
        )
        .inspect_err(|_| {
            // SAFETY: the buffer is live and owned; release its only reference before returning.
            unsafe { ffi::hrx_buffer_release(buffer.as_ptr()) };
        })?;

        self.resources.buffers.fetch_add(1, Ordering::AcqRel);
        Ok(AllocatedBuffer::new(
            XdnaBuffer {
                inner: Arc::new(BufferInner {
                    buffer,
                    mapped,
                    len: size,
                    in_flight: AtomicU64::new(0),
                    resources: Arc::clone(&self.resources),
                }),
                desc,
                context_id: context.id,
                _not_send_sync: PhantomData,
            },
            info,
        ))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(BufferUsage::TRANSFER_DESTINATION)
        {
            return Err(BackendError::PermissionDenied);
        }
        if buffer.inner.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let (start, end) = buffer.checked_range(offset, len)?;
        // SAFETY: the mapping is valid for `buffer.len` bytes and `[start,end)` is within it; the
        // exclusive borrow and the in-flight gate rule out concurrent device or host access.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(buffer.inner.mapped.as_ptr().add(start), len)
        };
        data.read_at(0, dst)?;
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe {
            ffi::hrx_buffer_flush_range(buffer.inner.buffer.as_ptr(), start, end - start)
        })?;
        // Count only completed transfers, so the diagnostics never include a failed write.
        self.explicit_transfer_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        if !buffer.desc.usage.contains(BufferUsage::TRANSFER_SOURCE) {
            return Err(BackendError::PermissionDenied);
        }
        if buffer.inner.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let (start, end) = buffer.checked_range(offset, len)?;
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe {
            ffi::hrx_buffer_invalidate_range(buffer.inner.buffer.as_ptr(), start, end - start)
        })?;
        // SAFETY: the mapping is valid for `buffer.len` bytes and `[start,end)` is within it; the
        // shared borrow with the in-flight gate rules out concurrent device writes.
        let src =
            unsafe { core::slice::from_raw_parts(buffer.inner.mapped.as_ptr().add(start), len) };
        data.write_at(0, src)?;
        self.explicit_transfer_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        if buffer.inner.in_flight.load(Ordering::Acquire) != 0 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: buffer,
            });
        }
        // The buffer's `Drop` releases the HRX handle (and its persistent mapping).
        drop(buffer);
        Ok(())
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if self.lane.is_poisoned() {
            return Err(BackendError::DeviceLost);
        }
        if artifact.payload.len() > self.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        // HRX publishes no finite residency bound for a loaded executable, so only the maximal
        // charge keeps the caller's program-residency accounting truthful (mirrors OpenVINO).
        if artifact.resident_bytes != crate::REQUIRED_RESIDENT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        let mut owned = Vec::new();
        let bytes = match artifact.payload.as_contiguous() {
            Some(bytes) => bytes,
            None => {
                let len = usize::try_from(artifact.payload.len())
                    .map_err(|_| BackendError::ResourceLimit)?;
                owned
                    .try_reserve_exact(len)
                    .map_err(|_| BackendError::OutOfMemory)?;
                owned.resize(len, 0);
                artifact.payload.read_at(0, &mut owned)?;
                &owned
            }
        };

        // The precompiled format loads directly; a TOSA artifact is admitted and compiled to one.
        let container = if artifact.format == artifact::XDNA_PRECOMPILED_FORMAT {
            std::borrow::Cow::Borrowed(bytes)
        } else if artifact.format == virtio_accel_tosa::ARTIFACT_FORMAT {
            let target = virtio_accel_tosa::Target::from_identity(artifact.target)
                .map_err(|_| BackendError::Incompatible)?;
            let spec = lower::admit(bytes, target)?;
            let compiler = match &self.compiler {
                CompilerState::Ready(compiler) => compiler,
                CompilerState::Unconfigured => return Err(BackendError::Unsupported),
                CompilerState::Failed(error) => return Err(*error),
            };
            std::borrow::Cow::Owned(compiler.compile(spec)?)
        } else {
            return Err(BackendError::Unsupported);
        };

        let parsed = PrecompiledArtifact::parse(container.as_ref())?;
        let device = shared_device().map_err(|_| BackendError::DeviceLost)?;
        build_executable(device, parsed, context.id, &self.resources)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        // Dropping the caller's `Arc` releases the executable once no in-flight job holds a clone.
        drop(program);
        Ok(())
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.info.validate_queue_desc(desc)?;
        self.resources.queues.fetch_add(1, Ordering::AcqRel);
        Ok(XdnaQueue {
            context_id: context.id,
            resources: Arc::clone(&self.resources),
        })
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        ResourceTracker::decrement(&queue.resources.queues);
        Ok(())
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let reject = |error| Err(SubmitFailure::Rejected(error));

        if self.lane.is_poisoned() {
            return reject(BackendError::DeviceLost);
        }
        // No cancellation exists on this hardware, so a finite deadline cannot be honored once work
        // reaches the device; reject it before admission (issue #85).
        if matches!(timeout, Timeout::AfterNs(_)) {
            return reject(BackendError::DeadlineExpired);
        }
        // An empty or over-limit binding list is a resource-limit violation (mirrors OpenVINO).
        if bindings.is_empty()
            || bindings.len() > self.info.limits.max_bindings_per_submission as usize
        {
            return reject(BackendError::ResourceLimit);
        }
        // Cross-context admission is rejected: the queue, the program, and every bound buffer must
        // belong to one context.
        if queue.context_id != program.context_id {
            return reject(BackendError::InvalidArgument);
        }

        let inputs = program.inner.inputs;
        let outputs = program.inner.outputs;
        let total = inputs + outputs;
        if bindings.len() != total {
            return reject(BackendError::InvalidArgument);
        }

        // One pass over the caller's bindings, writing each into its slot-indexed position:
        // inputs occupy slots `0..inputs`, outputs `inputs..total`. A 256-bit occupancy mask
        // rejects duplicate slots; with `bindings.len() == total` that also proves full coverage.
        let mut refs: Vec<ffi::hrx_buffer_ref_t> = (0..total)
            .map(|_| ffi::hrx_buffer_ref_t {
                buffer: ptr::null_mut(),
                offset: 0,
                length: 0,
            })
            .collect();
        let mut output_ranges: Vec<(ffi::hrx_buffer_t, usize, usize)> = Vec::with_capacity(outputs);
        let mut retained: Vec<Arc<BufferInner>> = Vec::with_capacity(total);
        // (buffer handle, bound-for-write) pairs for the aliasing rule below.
        let mut bound: Vec<(ffi::hrx_buffer_t, bool)> = Vec::with_capacity(total);
        let mut occupied = [0u64; 4]; // total <= max_bindings_per_submission = 256
        for binding in bindings {
            let slot = binding.slot as usize;
            if slot >= total {
                return reject(BackendError::InvalidArgument);
            }
            let bit = 1u64 << (slot % 64);
            if occupied[slot / 64] & bit != 0 {
                return reject(BackendError::InvalidArgument);
            }
            occupied[slot / 64] |= bit;
            let is_output = slot >= inputs;
            let expected = if is_output {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            if binding.access != expected {
                // The access is incompatible with the program's slot plan (mirrors OpenVINO).
                return reject(BackendError::Incompatible);
            }
            let buffer = binding.buffer;
            if buffer.context_id != queue.context_id {
                return reject(BackendError::InvalidArgument);
            }
            // The host is required to enforce usage-vs-access before submit (Wire ABI 4.4); repeat
            // it as defense in depth for in-process consumers, exactly as OpenVINO does.
            if !buffer.desc.allows_access(binding.access) {
                return reject(BackendError::PermissionDenied);
            }
            let offset = usize::try_from(binding.range.offset).unwrap_or(usize::MAX);
            let length = usize::try_from(binding.range.bytes()).unwrap_or(usize::MAX);
            if buffer.checked_range(binding.range.offset, length).is_err() {
                return reject(BackendError::OutOfBounds);
            }
            // The compiled TXN stream DMAs the slot's exact tensor extent regardless of the bound
            // length, so anything but an exact match would read or write past the caller's declared
            // range (mirrors OpenVINO's per-slot byte_len check).
            if binding.range.bytes() != program.inner.slot_bytes[slot] {
                return reject(BackendError::Incompatible);
            }
            // The descriptor starts at `offset`, so an unaligned start would ask the DMA for a
            // sub-word transfer it cannot express.
            if binding.range.offset % DMA_WORD_BYTES != 0 {
                return reject(BackendError::Incompatible);
            }
            // Aliasing rule: binding one buffer to several read slots is sound (the kernel only
            // loads from it; OpenVINO admits the same). Any alias involving a write slot has
            // kernel-order-dependent results and is rejected.
            let handle = buffer.inner.buffer.as_ptr();
            let mut already_retained = false;
            for &(other, other_writes) in &bound {
                if core::ptr::eq(other, handle) {
                    if is_output || other_writes {
                        return reject(BackendError::InvalidArgument);
                    }
                    already_retained = true;
                }
            }
            bound.push((handle, is_output));
            refs[slot] = ffi::hrx_buffer_ref_t {
                buffer: handle,
                offset,
                length,
            };
            if !already_retained {
                retained.push(Arc::clone(&buffer.inner));
            }
            if is_output {
                output_ranges.push((handle, offset, length));
            }
        }

        // Acceptance boundary: claim a ring entry and an event slot, then arm the in-flight gates.
        let slot = {
            let mut ring = self.lane.ring.lock().expect("ring mutex");
            if ring.queue.len() + ring.dispatched >= self.lane.depth {
                return reject(BackendError::Busy);
            }
            let Some(slot) = self.claim_slot() else {
                return reject(BackendError::Busy);
            };
            for (armed, buffer) in retained.iter().enumerate() {
                if buffer
                    .in_flight
                    .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    for prior in &retained[..armed] {
                        prior.in_flight.store(0, Ordering::Release);
                    }
                    slot.state.store(slot_state::FREE, Ordering::Release);
                    return reject(BackendError::Busy);
                }
            }
            self.resources.events.fetch_add(1, Ordering::AcqRel);
            ring.queue.push_back(Job {
                program: Arc::clone(&program.inner),
                bindings: refs,
                outputs: output_ranges,
                buffers: retained,
                slot: Arc::clone(&slot),
            });
            slot
        };
        self.lane.signal.notify_one();
        // Every binding was bound directly (persistent device-visible mapping, no bounce buffer).
        self.direct_binding_admissions
            .fetch_add(total as u64, Ordering::Relaxed);
        Ok(XdnaEvent {
            slot,
            resources: Arc::clone(&self.resources),
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match event.slot.state.load(Ordering::Acquire) {
            slot_state::PENDING => {
                if self.lane.is_poisoned() {
                    // A tier-2 wedge has no trustworthy terminal boundary. Keep the event pending
                    // internally while reporting device loss at the polling API.
                    Err(BackendError::DeviceLost)
                } else {
                    Ok(EventState::Pending)
                }
            }
            slot_state::COMPLETE => Ok(EventState::Complete),
            slot_state::FAILED => {
                let error = event
                    .slot
                    .error
                    .lock()
                    .expect("event error mutex")
                    .unwrap_or(BackendError::External {
                        domain: XDNA_ERROR_DOMAIN,
                        code: 0,
                    });
                Ok(EventState::Failed(error))
            }
            _ => Err(BackendError::External {
                domain: XDNA_ERROR_DOMAIN,
                code: INVALID_EVENT_STATE_CODE,
            }),
        }
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        // A pending event cannot be destroyed (the worker still writes its slot and gates).
        if event.slot.state.load(Ordering::Acquire) == slot_state::PENDING {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            });
        }
        // Dropping reclaims the slot, so this is the single reclaim path: an event that is merely
        // dropped instead of destroyed cannot strand its ring entry.
        drop(event);
        Ok(())
    }
}

impl Drop for XdnaEvent {
    /// Reclaim the ring slot. `destroy_event` is the intended release path, but a
    /// dropped-instead-of-destroyed event would strand its slot; enough of them exhaust the ring
    /// and fail every later submission with `Busy` for the life of the instance.
    ///
    /// Only a terminal slot is reclaimed. A `PENDING` slot still belongs to the dispatch worker,
    /// which will latch it; freeing it here would hand a live slot to the next submission. Such a
    /// slot stays charged to `resource_counts`, which is exactly where a wedged lane's event
    /// belongs. The error detail is cleared before `FREE` is published, so a later claimer of this
    /// slot can never observe the previous submission's failure.
    fn drop(&mut self) {
        let state = self.slot.state.load(Ordering::Acquire);
        if state != slot_state::COMPLETE && state != slot_state::FAILED {
            return;
        }
        *self.slot.error.lock().expect("event error mutex") = None;
        self.slot.state.store(slot_state::FREE, Ordering::Release);
        ResourceTracker::decrement(&self.resources.events);
    }
}

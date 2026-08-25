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
//! Finite timeouts are rejected
//! before admission (no cancellation exists); a synchronize error latches the event `Failed` and
//! poisons the instance (device-loss tier 1). The tier-2 wedge watchdog is the fault-paths ticket.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::collections::VecDeque;
use std::ffi::CString;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};

use crate::InitError;
use crate::artifact::{self, PrecompiledArtifact};
use crate::compiler::Compiler;
use crate::ffi;
use crate::lower;

/// `BackendError::External` domain tag for HRX failures ("XDNA" in ASCII).
pub const XDNA_ERROR_DOMAIN: u32 = 0x5844_4e41;

/// Upper bound advertised for a loaded artifact (mirrors the OpenVINO backend).
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// Default submission-ring depth (issue #85: one admitted request, matching the Hexagon backend).
const DEFAULT_RING_DEPTH: usize = 1;

/// Consume an HRX status and map it to a `BackendError`.
///
/// A non-NULL `hrx_status_t` is owned by the caller and must be consumed exactly once. This reads
/// its code, then ignores it (freeing it); on error it also renders and frees the message for
/// host-side context before discarding it. `NULL` is success.
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
        // SAFETY: the lane owns exactly one reference to the stream and is dropped once, after the
        // worker has been joined (no dispatch can be in progress).
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

/// One queued dispatch handed to the worker. All raw pointers reference resources the caller keeps
/// alive until the event is terminal and destroyed (the `Accelerator` contract), and are touched
/// only by the worker while the stream mutex is held.
struct Job {
    program: XdnaProgram,
    bindings: Vec<ffi::hrx_buffer_ref_t>,
    outputs: Vec<(ffi::hrx_buffer_t, usize, usize)>,
    in_flight: Vec<*const AtomicU64>,
    slot: Arc<EventSlot>,
}

// SAFETY: the raw pointers in a `Job` (buffer handles, in-flight gates) reference caller-owned
// resources kept live until the event is terminal and destroyed, and are dereferenced only on the
// worker thread while the stream mutex is held. Moving the job to the worker transfers that
// exclusive access.
unsafe impl Send for Job {}

/// The bounded submission ring.
struct Ring {
    queue: VecDeque<Job>,
    stopping: bool,
}

/// One instance's serialized dispatch lane: the stream, the ring, and the poison flag.
struct Lane {
    stream: Mutex<Stream>,
    ring: Mutex<Ring>,
    signal: Condvar,
    poisoned: AtomicBool,
    depth: usize,
}

impl Lane {
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Run the worker loop: drain the ring one job at a time, dispatching and synchronizing under
    /// the stream mutex, until stopped and empty.
    fn run_worker(self: &Arc<Self>) {
        loop {
            let job = {
                let mut ring = self.ring.lock().expect("ring mutex");
                loop {
                    if let Some(job) = ring.queue.pop_front() {
                        break job;
                    }
                    if ring.stopping {
                        return;
                    }
                    ring = self.signal.wait(ring).expect("ring condvar");
                }
            };
            self.execute(job);
        }
    }

    /// Dispatch one job on the stream, block on synchronize, make outputs host-visible, clear the
    /// in-flight gates, and latch the event. A synchronize error latches `Failed` and poisons the
    /// instance (device-loss tier 1).
    fn execute(&self, job: Job) {
        let config = ffi::hrx_dispatch_config_t {
            workgroup_count: [1, 1, 1],
            workgroup_size: [1, 1, 1],
            subgroup_size: 0,
        };
        let program = job.program.inner.as_ref();
        let result = {
            let stream = self.stream.lock().expect("stream mutex");
            // SAFETY: the stream, executable, and every bound buffer are live for this call
            // (the caller holds them until the event is terminal); the bindings slice is valid for
            // `binding_count`; the config is a valid local; constants are unused on this path.
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
            check(dispatch).and_then(|()| {
                // SAFETY: the stream is live; synchronize flushes and blocks until completion.
                check(unsafe { ffi::hrx_stream_synchronize(stream.0.as_ptr()) })
            })
        };

        let result = result.and_then(|()| {
            for &(buffer, offset, len) in &job.outputs {
                // SAFETY: each output buffer is live and persistently mapped; the range was
                // validated at submit; invalidate makes device writes host-visible.
                check(unsafe { ffi::hrx_buffer_invalidate_range(buffer, offset, len) })?;
            }
            Ok(())
        });

        let failed = result.is_err();
        // Clear the in-flight gates before publishing the terminal state, so a caller that observes
        // completion may immediately read or free the buffers.
        for &gate in &job.in_flight {
            // SAFETY: each gate points into a live buffer's `in_flight` atomic (kept alive by the
            // caller until the event is terminal and destroyed).
            unsafe { (*gate).store(0, Ordering::Release) };
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

/// One HRX backend instance: a serialized dispatch lane over the shared device.
pub struct XdnaAccelerator {
    lane: Arc<Lane>,
    worker: Option<JoinHandle<()>>,
    slots: Vec<Arc<EventSlot>>,
    compiler: Option<Compiler>,
    info: DeviceInfo,
    next_id: AtomicU64,
    /// Cumulative count of buffers admitted as direct bindings (no submission-time staging copy).
    direct_binding_admissions: AtomicU64,
    /// Cumulative bytes moved by explicit `write_buffer`/`read_buffer` transfers.
    explicit_transfer_bytes: AtomicU64,
    _not_send_sync: PhantomData<*mut u8>,
}

impl XdnaAccelerator {
    /// Initialize the shared device (once per process), create this instance's stream, and start
    /// its dispatch worker.
    pub fn new() -> Result<Self, InitError> {
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
                stopping: false,
            }),
            signal: Condvar::new(),
            poisoned: AtomicBool::new(false),
            depth,
        });
        let slots = (0..depth).map(|_| EventSlot::new()).collect();

        let worker_lane = Arc::clone(&lane);
        let worker = std::thread::Builder::new()
            .name("xdna-dispatch".into())
            .spawn(move || worker_lane.run_worker())
            .map_err(|_| InitError::Initialization)?;

        Ok(Self {
            lane,
            worker: Some(worker),
            slots,
            compiler: Compiler::from_env().ok(),
            info: device_info(),
            next_id: AtomicU64::new(1),
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
        // Signal the worker to stop, then join it before the lane (and its stream) is released.
        {
            let mut ring = self.lane.ring.lock().expect("ring mutex");
            ring.stopping = true;
        }
        self.lane.signal.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        // The worker is joined, so its `Arc<Lane>` is dropped; this instance's is now the last
        // reference, and dropping it releases the stream in `Stream::drop`.
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

    Ok(XdnaProgram {
        inner: Arc::new(ProgramInner {
            executable,
            ordinal,
            inputs: parsed.inputs,
            outputs: parsed.outputs,
        }),
        context_id,
    })
}

/// A logical execution context. HRX holds no per-context state; the id gives resources a home so
/// cross-context admission can be rejected (queues, programs, and buffers carry their context id).
#[derive(Debug)]
pub struct XdnaContext {
    id: u64,
}

/// A logical execution queue funnelling into the instance's single stream lane.
#[derive(Debug)]
pub struct XdnaQueue {
    context_id: u64,
}

/// A loaded, refcounted amdxdna executable and its dispatch plan.
#[derive(Debug)]
struct ProgramInner {
    executable: NonNull<ffi::hrx_executable_s>,
    ordinal: u32,
    inputs: usize,
    outputs: usize,
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
    }
}

/// A loaded program: a cheap handle over the shared [`ProgramInner`], tagged with its context.
#[derive(Clone, Debug)]
pub struct XdnaProgram {
    inner: Arc<ProgramInner>,
    context_id: u64,
}

/// A submission event: a handle over one preallocated [`EventSlot`].
#[derive(Debug)]
pub struct XdnaEvent {
    slot: Arc<EventSlot>,
}

/// An HRX buffer with its persistent host mapping.
#[derive(Debug)]
pub struct XdnaBuffer {
    buffer: NonNull<ffi::hrx_buffer_s>,
    mapped: NonNull<u8>,
    len: usize,
    desc: BufferDesc,
    /// The context that allocated this buffer; a submission may only bind buffers from its own.
    context_id: u64,
    /// Set while the device may be reading or writing this buffer; guards host access and release.
    in_flight: AtomicU64,
    _not_send_sync: PhantomData<*mut u8>,
}

impl XdnaBuffer {
    /// Byte range `[offset, offset+len)` checked against the mapping, returning `(start, end)`.
    fn checked_range(&self, offset: u64, len: usize) -> Result<(usize, usize), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(len)
            .filter(|end| *end <= self.len)
            .ok_or(BackendError::OutOfBounds)?;
        Ok((start, end))
    }
}

impl Drop for XdnaBuffer {
    fn drop(&mut self) {
        // SAFETY: this handle owns one buffer reference and is dropped once. The persistent mapping
        // is released together with the buffer (the fork never unmaps first).
        unsafe { ffi::hrx_buffer_release(self.buffer.as_ptr()) };
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
        Ok(XdnaContext { id: self.next_id() })
    }

    fn destroy_context(
        &self,
        _context: Self::Context,
    ) -> Result<(), ReleaseFailure<Self::Context>> {
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

        Ok(AllocatedBuffer::new(
            XdnaBuffer {
                buffer,
                mapped,
                len: size,
                desc,
                context_id: context.id,
                in_flight: AtomicU64::new(0),
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
        if buffer.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let (start, end) = buffer.checked_range(offset, len)?;
        // SAFETY: the mapping is valid for `buffer.len` bytes and `[start,end)` is within it; the
        // exclusive borrow and the in-flight gate rule out concurrent device or host access.
        let dst =
            unsafe { core::slice::from_raw_parts_mut(buffer.mapped.as_ptr().add(start), len) };
        data.read_at(0, dst)?;
        self.explicit_transfer_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe { ffi::hrx_buffer_flush_range(buffer.buffer.as_ptr(), start, end - start) })
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
        if buffer.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let (start, end) = buffer.checked_range(offset, len)?;
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe {
            ffi::hrx_buffer_invalidate_range(buffer.buffer.as_ptr(), start, end - start)
        })?;
        // SAFETY: the mapping is valid for `buffer.len` bytes and `[start,end)` is within it; the
        // shared borrow with the in-flight gate rules out concurrent device writes.
        let src = unsafe { core::slice::from_raw_parts(buffer.mapped.as_ptr().add(start), len) };
        data.write_at(0, src)?;
        self.explicit_transfer_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        if buffer.in_flight.load(Ordering::Acquire) != 0 {
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
        let len =
            usize::try_from(artifact.payload.len()).map_err(|_| BackendError::ResourceLimit)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| BackendError::OutOfMemory)?;
        bytes.resize(len, 0);
        artifact.payload.read_at(0, &mut bytes)?;

        // The precompiled format loads directly; a TOSA artifact is admitted and compiled to one.
        let container = if artifact.format == artifact::XDNA_PRECOMPILED_FORMAT {
            bytes
        } else if artifact.format == virtio_accel_tosa::ARTIFACT_FORMAT {
            let target = virtio_accel_tosa::Target::from_identity(artifact.target)
                .map_err(|_| BackendError::Incompatible)?;
            let spec = lower::admit(&bytes, target).map_err(|error| match error {
                lower::AdmitError::Parse => BackendError::InvalidArgument,
                lower::AdmitError::Analysis => BackendError::Incompatible,
                lower::AdmitError::Unsupported => BackendError::Unsupported,
            })?;
            self.compiler
                .as_ref()
                .ok_or(BackendError::Unsupported)?
                .compile(spec)?
        } else {
            return Err(BackendError::Unsupported);
        };

        let parsed = PrecompiledArtifact::parse(&container)?;
        let device = shared_device().map_err(|_| BackendError::DeviceLost)?;
        build_executable(device, parsed, context.id)
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
        if !desc.flags.is_empty() {
            return Err(BackendError::Unsupported);
        }
        Ok(XdnaQueue {
            context_id: context.id,
        })
    }

    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
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

        // Order the bindings by slot: inputs occupy slots `0..inputs`, outputs `inputs..total`.
        let mut refs: Vec<ffi::hrx_buffer_ref_t> = Vec::with_capacity(total);
        let mut output_ranges: Vec<(ffi::hrx_buffer_t, usize, usize)> = Vec::with_capacity(outputs);
        let mut gates: Vec<*const AtomicU64> = Vec::with_capacity(total);
        for slot in 0..total {
            let Some(binding) = bindings.iter().find(|b| b.slot as usize == slot) else {
                return reject(BackendError::InvalidArgument);
            };
            let expected = if slot < inputs {
                AccessMode::Read
            } else {
                AccessMode::Write
            };
            if binding.access != expected {
                // The access is incompatible with the program's slot plan (mirrors OpenVINO).
                return reject(BackendError::Incompatible);
            }
            let buffer = binding.buffer;
            if buffer.context_id != queue.context_id {
                return reject(BackendError::InvalidArgument);
            }
            let offset = usize::try_from(binding.range.offset).unwrap_or(usize::MAX);
            let length = usize::try_from(binding.range.bytes()).unwrap_or(usize::MAX);
            if buffer.checked_range(binding.range.offset, length).is_err() {
                return reject(BackendError::OutOfBounds);
            }
            refs.push(ffi::hrx_buffer_ref_t {
                buffer: buffer.buffer.as_ptr(),
                offset,
                length,
            });
            gates.push(&buffer.in_flight as *const AtomicU64);
            if slot >= inputs {
                output_ranges.push((buffer.buffer.as_ptr(), offset, length));
            }
        }

        // Reject an aliased binding conflict (any buffer bound more than once).
        for (index, gate) in gates.iter().enumerate() {
            if gates[..index].contains(gate) {
                return reject(BackendError::InvalidArgument);
            }
        }

        // Acceptance boundary: claim a ring entry and an event slot, then arm the in-flight gates.
        let slot = {
            let mut ring = self.lane.ring.lock().expect("ring mutex");
            if ring.queue.len() >= self.lane.depth {
                return reject(BackendError::Busy);
            }
            let Some(slot) = self.claim_slot() else {
                return reject(BackendError::Busy);
            };
            for &gate in &gates {
                // SAFETY: each gate points into a live bound buffer's `in_flight` atomic.
                unsafe { (*gate).store(1, Ordering::Release) };
            }
            ring.queue.push_back(Job {
                program: program.clone(),
                bindings: refs,
                outputs: output_ranges,
                in_flight: gates,
                slot: Arc::clone(&slot),
            });
            slot
        };
        self.lane.signal.notify_one();
        // Every binding was bound directly (persistent device-visible mapping, no bounce buffer).
        self.direct_binding_admissions
            .fetch_add(total as u64, Ordering::Relaxed);
        Ok(XdnaEvent { slot })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match event.slot.state.load(Ordering::Acquire) {
            slot_state::PENDING => {
                if self.lane.is_poisoned() {
                    // Tier-2 wedge (a never-terminal event) is the fault-paths ticket; a poisoned
                    // instance with a still-pending event reports device loss.
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
            _ => Ok(EventState::Pending),
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
        *event.slot.error.lock().expect("event error mutex") = None;
        event.slot.state.store(slot_state::FREE, Ordering::Release);
        Ok(())
    }
}

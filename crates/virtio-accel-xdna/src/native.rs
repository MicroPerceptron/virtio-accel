//! Native XDNA backend over the HRX runtime (`libhrx`).
//!
//! This module owns every call across the `ffi` boundary. `SAFETY.md` is the audit of record; each
//! `unsafe` block below carries a local `SAFETY:` note. Scope for this ticket: one audited
//! process-wide device owner, a per-instance stream, and `hrx_buffer` primitives (allocate with a
//! persistent host mapping, range flush/invalidate, release). Program loading and dispatch return
//! `Unsupported`; they land with the execution path in a later ticket, at which point the
//! uninhabited `XdnaProgram`/`XdnaEvent` become real types.

use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicU64, Ordering};
use std::ffi::c_void;
use std::sync::OnceLock;

use virtio_accel_core::{
    Accelerator, AcceleratorClass, AllocatedBuffer, ArtifactRef, BackendError, BindingRef,
    BufferDesc, BufferInfo, BufferProperties, ByteSink, ByteSource, Capabilities, ContextDesc,
    DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain, QueueDesc, ReleaseFailure,
    SubmitFailure, Timeout,
};

use crate::InitError;
use crate::ffi;

/// `BackendError::External` domain tag for HRX failures ("XDNA" in ASCII).
pub const XDNA_ERROR_DOMAIN: u32 = 0x5844_4e41;

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
    // Render the message for host-side logging, then free both the message and the status.
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
// reachable from any thread. This module uses it only to create per-instance streams
// (`hrx_stream_create`), never mutating it; sharing the pointer across threads for that read-only
// factory use is sound. Concurrency hazards live on the stream (not concurrency-safe), which is
// per-instance and not shared here.
unsafe impl Send for SharedDevice {}
unsafe impl Sync for SharedDevice {}

fn shared_device() -> Result<&'static SharedDevice, InitError> {
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

/// One HRX backend instance: a serialized dispatch lane (one stream) over the shared device.
pub struct XdnaAccelerator {
    stream: NonNull<ffi::hrx_stream_s>,
    info: DeviceInfo,
    next_id: AtomicU64,
    // One stream is not safe for concurrent dispatch; the instance stays single-threaded until the
    // execution path introduces the serialized worker.
    _not_send_sync: PhantomData<*mut u8>,
}

impl XdnaAccelerator {
    /// Initialize the shared device (once per process) and create this instance's stream.
    pub fn new() -> Result<Self, InitError> {
        let device = shared_device()?;
        let mut stream: ffi::hrx_stream_t = ptr::null_mut();
        // SAFETY: `device.0` is the live process-wide device; the out-pointer is a valid local and
        // the returned status is consumed by `check`.
        let status = unsafe { ffi::hrx_stream_create(device.0.as_ptr(), 0, &mut stream) };
        check(status).map_err(|_| InitError::Initialization)?;
        let stream = NonNull::new(stream).ok_or(InitError::Initialization)?;
        Ok(Self {
            stream,
            info: device_info(),
            next_id: AtomicU64::new(1),
            _not_send_sync: PhantomData,
        })
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Drop for XdnaAccelerator {
    fn drop(&mut self) {
        // SAFETY: this instance owns exactly one reference to the stream and is dropped once.
        unsafe { ffi::hrx_stream_release(self.stream.as_ptr()) };
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
        limits: DeviceLimits {
            max_contexts: u32::MAX,
            max_buffers_per_context: u32::MAX,
            max_programs_per_context: u32::MAX,
            max_queues_per_context: u32::MAX,
            max_events_per_context: u32::MAX,
            max_bindings_per_submission: 256,
            max_buffer_bytes: u64::MAX,
            max_artifact_bytes: u64::MAX,
        },
    }
}

/// A logical execution context. HRX holds no per-context state; the id supports accounting.
#[derive(Debug)]
pub struct XdnaContext {
    _id: u64,
}

/// A logical execution queue funnelling into the instance's single stream lane.
#[derive(Debug)]
pub struct XdnaQueue {
    _id: u64,
}

/// Program loading lands with the execution path; no program is constructed in this ticket.
pub enum XdnaProgram {}

/// Dispatch lands with the execution path; no event is constructed in this ticket.
pub enum XdnaEvent {}

/// An HRX buffer with its persistent host mapping.
#[derive(Debug)]
pub struct XdnaBuffer {
    buffer: NonNull<ffi::hrx_buffer_s>,
    mapped: NonNull<u8>,
    len: usize,
    desc: BufferDesc,
    /// Set while the device may be reading or writing this buffer; guards release and host access.
    /// Always zero until the dispatch path sets it.
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
        Ok(XdnaContext {
            _id: self.next_id(),
        })
    }

    fn destroy_context(
        &self,
        _context: Self::Context,
    ) -> Result<(), ReleaseFailure<Self::Context>> {
        Ok(())
    }

    fn allocate_buffer(
        &self,
        _context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.info.validate_buffer_desc(desc)?;
        // HRX host-mapped buffers are device-visible shared memory; device-local-only is not a
        // separate path here.
        if desc.domain == MemoryDomain::Device {
            return Err(BackendError::Unsupported);
        }
        let size = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfMemory)?;
        // HRX rejects zero-size allocations; `BufferDesc` already guarantees a nonzero size.

        let mut buffer: ffi::hrx_buffer_t = ptr::null_mut();
        // SAFETY: the stream is live; the out-pointer is a valid local; the status is consumed.
        let status = unsafe {
            ffi::hrx_buffer_allocate(
                self.stream.as_ptr(),
                size,
                ffi::HRX_MEMORY_TYPE_HOST_LOCAL | ffi::HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                ffi::HRX_BUFFER_USAGE_DEFAULT | ffi::HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                &mut buffer,
            )
        };
        check(status)?;
        let buffer = NonNull::new(buffer).ok_or(BackendError::External {
            domain: XDNA_ERROR_DOMAIN,
            code: 0,
        })?;

        // Establish the persistent read/write mapping for the whole buffer. On any failure the
        // buffer is released before returning so no handle leaks.
        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: `buffer` is the just-allocated live buffer; the out-pointer is a valid local; the
        // status is consumed.
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
                // SAFETY: `buffer` is live and owned here; releasing it on the error path drops the
                // only reference before the handle is discarded.
                unsafe { ffi::hrx_buffer_release(buffer.as_ptr()) };
                return Err(error);
            }
        };

        // Report the mapping's actual alignment (the largest power of two dividing its address),
        // which must satisfy the requested alignment.
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
            .contains(virtio_accel_core::BufferUsage::TRANSFER_DESTINATION)
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
        // Push the host writes device-ward.
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe { ffi::hrx_buffer_flush_range(buffer.buffer.as_ptr(), start, end - start) })
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        if buffer.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let (start, end) = buffer.checked_range(offset, len)?;
        // Make device writes visible to the host before reading.
        // SAFETY: the buffer is live and currently mapped; the range is validated; status consumed.
        check(unsafe {
            ffi::hrx_buffer_invalidate_range(buffer.buffer.as_ptr(), start, end - start)
        })?;
        // SAFETY: the mapping is valid for `buffer.len` bytes and `[start,end)` is within it; the
        // shared borrow with the in-flight gate rules out concurrent device writes.
        let src = unsafe { core::slice::from_raw_parts(buffer.mapped.as_ptr().add(start), len) };
        data.write_at(0, src)
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
        _context: &Self::Context,
        _artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        // Program compilation and loading land with the execution path.
        Err(BackendError::Unsupported)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        match program {}
    }

    fn create_queue(
        &self,
        _context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        if !desc.flags.is_empty() {
            return Err(BackendError::Unsupported);
        }
        Ok(XdnaQueue {
            _id: self.next_id(),
        })
    }

    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        Ok(())
    }

    fn submit(
        &self,
        _queue: &Self::Queue,
        _program: &Self::Program,
        _bindings: &[BindingRef<'_, Self::Buffer>],
        _timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        // Dispatch lands with the execution path.
        Err(SubmitFailure::Rejected(BackendError::Unsupported))
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match *event {}
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        match event {}
    }
}

//! Audited macOS implementation. See `SAFETY.md`.

use crate::artifact::{DecodeError, FeatureRole, MAX_ARTIFACT_BYTES, decode};
use crate::{ARTIFACT_FORMAT, InitError, REQUIRED_RESIDENT_BYTES, TARGET_IDENTITY};
use core::ffi::c_void;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::BTreeSet;
use std::fmt;
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AllocatedBuffer, ArtifactRef, BackendError, BindingRef,
    BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource, Capabilities,
    ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain, QueueDesc,
    ReleaseFailure, SubmitFailure, Timeout, validate_bindings,
};

const COREML_MIN_ALIGNMENT: usize = 16 * 1024;
const TRANSFER_CHUNK_BYTES: usize = 16 * 1024;
const COREML_EXTERNAL_DOMAIN: u32 = 0x434d_4c45;

const ERROR_UNSUPPORTED: u32 = 1;
const ERROR_INCOMPATIBLE: u32 = 2;
const ERROR_INVALID_ARGUMENT: u32 = 3;
const ERROR_OUT_OF_BOUNDS: u32 = 4;
const ERROR_OUT_OF_MEMORY: u32 = 5;
const ERROR_RESOURCE_LIMIT: u32 = 6;
const ERROR_DEVICE_LOST: u32 = 7;
const ERROR_EXTERNAL: u32 = 8;

const EVENT_PENDING: u32 = 0;
const EVENT_COMPLETE: u32 = 1;
const EVENT_FAILED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeError {
    kind: u32,
    domain: u32,
    code: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeFeatureMapping {
    slot: u32,
    role: u8,
    name: *const u8,
    name_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeBinding {
    slot: u32,
    access: u8,
    data: *mut c_void,
    bytes: u64,
}

type NativeReleaseContext = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" {
    fn va_coreml_has_neural_engine() -> std::ffi::c_int;
    fn va_coreml_model_load(
        path: *const u8,
        path_len: usize,
        mappings: *const NativeFeatureMapping,
        mapping_count: usize,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn va_coreml_model_release(model: *mut c_void);
    fn va_coreml_submit(
        model: *mut c_void,
        bindings: *const NativeBinding,
        binding_count: usize,
        context: *mut c_void,
        release_context: NativeReleaseContext,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn va_coreml_event_poll(event: *mut c_void, error: *mut NativeError) -> u32;
    fn va_coreml_event_release(event: *mut c_void);
}

#[derive(Debug)]
struct AlignedAllocation {
    pointer: NonNull<u8>,
    layout: Layout,
    in_flight: AtomicU64,
}

impl AlignedAllocation {
    fn new(bytes: u64, requested_alignment: u64) -> Result<Self, BackendError> {
        let bytes = usize::try_from(bytes).map_err(|_| BackendError::OutOfMemory)?;
        let requested_alignment =
            usize::try_from(requested_alignment).map_err(|_| BackendError::OutOfMemory)?;
        let alignment = requested_alignment.max(COREML_MIN_ALIGNMENT);
        let allocation_bytes = bytes
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or(BackendError::OutOfMemory)?;
        let layout = Layout::from_size_align(allocation_bytes, alignment)
            .map_err(|_| BackendError::OutOfMemory)?;
        // SAFETY: `layout` is valid and nonzero. The returned pointer is owned by this value and is
        // released with the identical layout in `Drop`.
        let pointer =
            NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(BackendError::OutOfMemory)?;
        Ok(Self {
            pointer,
            layout,
            in_flight: AtomicU64::new(0),
        })
    }

    fn allocation_bytes(&self) -> u64 {
        self.layout.size() as u64
    }

    fn alignment(&self) -> u64 {
        self.layout.align() as u64
    }

    fn pointer_at(&self, offset: usize) -> *mut u8 {
        // SAFETY: every caller first validates `offset` against the allocation's logical buffer
        // range, which is no larger than this layout.
        unsafe { self.pointer.as_ptr().add(offset) }
    }
}

#[derive(Debug)]
struct InFlightBacking(Arc<AlignedAllocation>);

impl InFlightBacking {
    fn acquire(allocation: Arc<AlignedAllocation>) -> Result<Self, BackendError> {
        allocation
            .in_flight
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| BackendError::Busy)?;
        Ok(Self(allocation))
    }
}

impl Drop for InFlightBacking {
    fn drop(&mut self) {
        let prior = self.0.in_flight.fetch_sub(1, Ordering::Release);
        debug_assert_eq!(prior, 1);
    }
}

// SAFETY: allocation access is available only through the `Accelerator` methods. The buffer handle
// is neither Send nor Sync, and its atomic in-flight gate excludes host transfers while native
// prediction owns a backing guard. Completion synchronizes through the native release/acquire event
// status. See `SAFETY.md`.
unsafe impl Send for AlignedAllocation {}
// SAFETY: same invariant as the `Send` implementation.
unsafe impl Sync for AlignedAllocation {}

impl Drop for AlignedAllocation {
    fn drop(&mut self) {
        // SAFETY: `pointer` was returned by `alloc_zeroed(self.layout)` and has not been freed.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// Core ML context handle.
#[derive(Clone, Debug)]
pub struct CoreMlContext {
    id: u64,
}

/// Page-aligned buffer directly wrapped by Core ML `MLMultiArray` objects.
#[derive(Debug)]
pub struct CoreMlBuffer {
    context_id: u64,
    desc: BufferDesc,
    allocation: Arc<AlignedAllocation>,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Resident Core ML model handle.
pub struct CoreMlProgram {
    context_id: u64,
    native: NonNull<c_void>,
}

impl fmt::Debug for CoreMlProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CoreMlProgram")
            .field("context_id", &self.context_id)
            .finish_non_exhaustive()
    }
}

impl Drop for CoreMlProgram {
    fn drop(&mut self) {
        // SAFETY: `native` owns one bridge retain and is released exactly once here.
        unsafe { va_coreml_model_release(self.native.as_ptr()) };
    }
}

/// Core ML execution queue handle.
#[derive(Clone, Debug)]
pub struct CoreMlQueue {
    context_id: u64,
}

/// Asynchronous Core ML prediction event.
pub struct CoreMlEvent {
    native: NonNull<c_void>,
}

impl fmt::Debug for CoreMlEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CoreMlEvent")
            .finish_non_exhaustive()
    }
}

impl Drop for CoreMlEvent {
    fn drop(&mut self) {
        // SAFETY: `native` owns the Rust-side event reference. The bridge's completion callback
        // owns a separate reference, so this is safe even for an early Rust drop.
        unsafe { va_coreml_event_release(self.native.as_ptr()) };
    }
}

/// One Core ML backend instance rooted at a host-controlled model directory.
#[derive(Debug)]
pub struct CoreMlAccelerator {
    model_root: PathBuf,
    next_id: AtomicU64,
    direct_binding_admissions: AtomicU64,
    info: DeviceInfo,
}

impl CoreMlAccelerator {
    /// Construct a backend and require an accessible Apple Neural Engine.
    pub fn new(model_root: impl AsRef<Path>) -> Result<Self, InitError> {
        // SAFETY: the function has no pointer arguments and returns a scalar availability result.
        if unsafe { va_coreml_has_neural_engine() } == 0 {
            return Err(InitError::NeuralEngineUnavailable);
        }
        let model_root = model_root
            .as_ref()
            .canonicalize()
            .map_err(|_| InitError::InvalidModelRoot)?;
        if !model_root.is_dir() || model_root.to_str().is_none() {
            return Err(InitError::InvalidModelRoot);
        }
        Ok(Self {
            model_root,
            next_id: AtomicU64::new(1),
            direct_binding_admissions: AtomicU64::new(0),
            info: DeviceInfo {
                identity: DeviceIdentity {
                    uuid: *b"apple-coreml-ane",
                    class: AcceleratorClass::NPU,
                    vendor_id: 0x106b,
                    device_id: 0,
                },
                capabilities: Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY,
                limits: DeviceLimits {
                    max_contexts: 64,
                    max_buffers_per_context: 1_024,
                    max_programs_per_context: 64,
                    max_queues_per_context: 64,
                    max_events_per_context: 4_096,
                    max_bindings_per_submission: 256,
                    max_buffer_bytes: 16 * 1024 * 1024 * 1024,
                    max_artifact_bytes: MAX_ARTIFACT_BYTES,
                },
            },
        })
    }

    /// Number of exact provider allocations admitted as Core ML bindings.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.load(Ordering::Relaxed)
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| BackendError::ResourceLimit)
    }

    fn resolve_model(&self, relative: &str) -> Result<PathBuf, BackendError> {
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(BackendError::PermissionDenied);
        }
        let resolved = self
            .model_root
            .join(path)
            .canonicalize()
            .map_err(|_| BackendError::InvalidArgument)?;
        if !resolved.starts_with(&self.model_root) || resolved.as_os_str().to_str().is_none() {
            return Err(BackendError::PermissionDenied);
        }
        Ok(resolved)
    }

    fn checked_range(
        buffer: &CoreMlBuffer,
        offset: u64,
        bytes: u64,
    ) -> Result<(usize, usize), BackendError> {
        if bytes == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= buffer.desc.bytes())
            .ok_or(BackendError::OutOfBounds)?;
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = usize::try_from(end).map_err(|_| BackendError::OutOfBounds)?;
        Ok((start, end))
    }

    fn native_error(error: NativeError) -> BackendError {
        match error.kind {
            ERROR_UNSUPPORTED => BackendError::Unsupported,
            ERROR_INCOMPATIBLE => BackendError::Incompatible,
            ERROR_INVALID_ARGUMENT => BackendError::InvalidArgument,
            ERROR_OUT_OF_BOUNDS => BackendError::OutOfBounds,
            ERROR_OUT_OF_MEMORY => BackendError::OutOfMemory,
            ERROR_RESOURCE_LIMIT => BackendError::ResourceLimit,
            ERROR_DEVICE_LOST => BackendError::DeviceLost,
            ERROR_EXTERNAL => BackendError::External {
                domain: if error.domain == 0 {
                    COREML_EXTERNAL_DOMAIN
                } else {
                    error.domain
                },
                code: error.code,
            },
            _ => BackendError::External {
                domain: COREML_EXTERNAL_DOMAIN,
                code: error.code,
            },
        }
    }
}

unsafe extern "C" fn release_event_backings(context: *mut c_void) {
    // SAFETY: successful submission transfers exactly one pointer produced by `Box::into_raw` to
    // the bridge, and its completion block calls this function exactly once.
    drop(unsafe { Box::from_raw(context.cast::<Vec<InFlightBacking>>()) });
}

impl Accelerator for CoreMlAccelerator {
    type Context = CoreMlContext;
    type Buffer = CoreMlBuffer;
    type Program = CoreMlProgram;
    type Queue = CoreMlQueue;
    type Event = CoreMlEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        Ok(CoreMlContext {
            id: self.next_id()?,
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
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.info.validate_buffer_desc(desc)?;
        let allocation = Arc::new(AlignedAllocation::new(desc.bytes(), desc.alignment())?);
        let properties = BufferProperties::HOST_VISIBLE
            | if desc.is_program_visible() || desc.domain == MemoryDomain::Shared {
                BufferProperties::DIRECT_BINDING
            } else {
                BufferProperties::empty()
            };
        let info = BufferInfo::new(
            desc,
            allocation.allocation_bytes(),
            allocation.alignment(),
            properties,
        )?;
        Ok(AllocatedBuffer::new(
            CoreMlBuffer {
                context_id: context.id,
                desc,
                allocation,
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
        if buffer.allocation.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let target_len = end - start;
        if let Some(source) = data.as_contiguous() {
            if source.len() != target_len {
                return Err(BackendError::InvalidArgument);
            }
            // SAFETY: the validated range lies within the owned allocation, `buffer` is exclusively
            // borrowed, and the source is a distinct borrowed byte region for this call.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    buffer.allocation.pointer_at(start),
                    target_len,
                )
            };
            return Ok(());
        }

        let mut scratch = [0; TRANSFER_CHUNK_BYTES];
        let mut copied = 0usize;
        while copied < target_len {
            let chunk = (target_len - copied).min(scratch.as_slice().len());
            data.read_at(copied as u64, &mut scratch[..chunk])?;
            // SAFETY: `start + copied .. + chunk` stays inside the validated exclusive range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    buffer.allocation.pointer_at(start + copied),
                    chunk,
                )
            };
            copied += chunk;
        }
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
        if buffer.allocation.in_flight.load(Ordering::Acquire) != 0 {
            return Err(BackendError::Busy);
        }
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let source_len = end - start;
        if let Some(target) = data.as_contiguous_mut() {
            if target.len() != source_len {
                return Err(BackendError::InvalidArgument);
            }
            // SAFETY: the source range is validated and the in-flight gate proved that no native
            // event still mutates this buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.allocation.pointer_at(start),
                    target.as_mut_ptr(),
                    source_len,
                )
            };
            return Ok(());
        }

        let mut scratch = [0; TRANSFER_CHUNK_BYTES];
        let mut copied = 0usize;
        while copied < source_len {
            let chunk = (source_len - copied).min(scratch.as_slice().len());
            // SAFETY: `start + copied .. + chunk` remains in the validated source range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.allocation.pointer_at(start + copied),
                    scratch.as_mut_ptr(),
                    chunk,
                )
            };
            data.write_at(copied as u64, &scratch[..chunk])?;
            copied += chunk;
        }
        Ok(())
    }

    fn free_buffer(&self, _buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        Ok(())
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if artifact.format != ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        if artifact.target != TARGET_IDENTITY {
            return Err(BackendError::Incompatible);
        }
        if artifact.payload.len() > self.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.resident_bytes != REQUIRED_RESIDENT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        let decoded = decode(artifact.payload).map_err(|error| match error {
            DecodeError::Invalid => BackendError::InvalidArgument,
            DecodeError::OutOfBounds => BackendError::OutOfBounds,
            DecodeError::ResourceLimit => BackendError::ResourceLimit,
        })?;
        let model_path = self.resolve_model(&decoded.model_path)?;
        let path = model_path.as_os_str().as_bytes();
        let mappings = decoded
            .mappings
            .iter()
            .map(|mapping| NativeFeatureMapping {
                slot: mapping.slot,
                role: match mapping.role {
                    FeatureRole::Input => 1,
                    FeatureRole::Output => 2,
                },
                name: mapping.name.as_ptr(),
                name_len: mapping.name.len(),
            })
            .collect::<Vec<_>>();
        let mut error = NativeError::default();
        // SAFETY: every byte/name pointer remains valid for this synchronous call. The bridge copies
        // all retained strings and returns one owned model reference on success.
        let native = unsafe {
            va_coreml_model_load(
                path.as_ptr(),
                path.len(),
                mappings.as_ptr(),
                mappings.len(),
                &mut error,
            )
        };
        let native = NonNull::new(native).ok_or_else(|| Self::native_error(error))?;
        Ok(CoreMlProgram {
            context_id: context.id,
            native,
        })
    }

    fn unload_program(&self, _program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        Ok(())
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.info.validate_queue_desc(desc)?;
        Ok(CoreMlQueue {
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
        _timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        validate_bindings(bindings, self.info.limits.max_bindings_per_submission)
            .map_err(SubmitFailure::Rejected)?;
        if queue.context_id != program.context_id {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }

        let mut native_bindings = Vec::new();
        let mut allocations = Vec::new();
        native_bindings
            .try_reserve_exact(bindings.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        allocations
            .try_reserve_exact(bindings.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        let mut contexts = BTreeSet::new();
        for binding in bindings {
            contexts.insert(binding.buffer.context_id);
            if binding.buffer.context_id != queue.context_id {
                return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
            }
            if !binding.buffer.desc.allows_access(binding.access) {
                return Err(SubmitFailure::Rejected(BackendError::PermissionDenied));
            }
            let (start, _) =
                Self::checked_range(binding.buffer, binding.range.offset, binding.range.bytes())
                    .map_err(SubmitFailure::Rejected)?;
            native_bindings.push(NativeBinding {
                slot: binding.slot,
                access: binding.access as u8,
                data: binding.buffer.allocation.pointer_at(start).cast(),
                bytes: binding.range.bytes(),
            });
            if !allocations
                .iter()
                .any(|prior| Arc::ptr_eq(prior, &binding.buffer.allocation))
            {
                allocations.push(Arc::clone(&binding.buffer.allocation));
            }
        }
        if contexts.len() != 1 {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }

        let backings = allocations
            .into_iter()
            .map(InFlightBacking::acquire)
            .collect::<Result<Vec<_>, _>>()
            .map_err(SubmitFailure::Rejected)?;
        let backings = Box::new(backings);
        let backing_context = Box::into_raw(backings).cast::<c_void>();
        let mut error = NativeError::default();
        // SAFETY: model and binding pointers remain valid through this admission call. The Core ML
        // arrays created by the bridge borrow the allocations. On success the completion block
        // owns `backing_context`; on rejection ownership remains here and is reconstructed below.
        let native = unsafe {
            va_coreml_submit(
                program.native.as_ptr(),
                native_bindings.as_ptr(),
                native_bindings.len(),
                backing_context,
                release_event_backings,
                &mut error,
            )
        };
        let native = match NonNull::new(native) {
            Some(native) => native,
            None => {
                // SAFETY: the bridge contract retains the context only when it returns an event.
                drop(unsafe { Box::from_raw(backing_context.cast::<Vec<InFlightBacking>>()) });
                return Err(SubmitFailure::Rejected(Self::native_error(error)));
            }
        };
        self.direct_binding_admissions
            .fetch_add(bindings.len() as u64, Ordering::Relaxed);
        Ok(CoreMlEvent { native })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        let mut error = NativeError::default();
        // SAFETY: `event.native` owns a live native event reference for this borrow.
        let status = unsafe { va_coreml_event_poll(event.native.as_ptr(), &mut error) };
        match status {
            EVENT_PENDING => Ok(EventState::Pending),
            EVENT_COMPLETE => Ok(EventState::Complete),
            EVENT_FAILED => Ok(EventState::Failed(Self::native_error(error))),
            _ => Err(BackendError::External {
                domain: COREML_EXTERNAL_DOMAIN,
                code: i64::from(status),
            }),
        }
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        match self.poll_event(&event) {
            Ok(EventState::Pending) => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            }),
            Ok(EventState::Complete | EventState::Failed(_) | EventState::Cancelled) => Ok(()),
            Err(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: event,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_backing_guard_excludes_a_second_user() {
        let allocation = Arc::new(AlignedAllocation::new(32, 16).unwrap());
        let guard = InFlightBacking::acquire(Arc::clone(&allocation)).unwrap();
        assert_eq!(
            InFlightBacking::acquire(Arc::clone(&allocation)).unwrap_err(),
            BackendError::Busy
        );
        drop(guard);
        assert!(InFlightBacking::acquire(allocation).is_ok());
    }
}

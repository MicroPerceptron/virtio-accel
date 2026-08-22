//! Audited macOS implementation. See `SAFETY.md`.

use crate::artifact::{DecodeError, FeatureRole, MAX_ARTIFACT_BYTES, decode};
use crate::lower::{LoweredFeature, LoweredFeatureRole, LoweringError, lower_tosa};
use crate::{ARTIFACT_FORMAT, InitError, REQUIRED_RESIDENT_BYTES, TARGET_IDENTITY};
use core::ffi::c_void;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_tosa::{CapabilityDescriptor, TosaCapabilityProvider};

const COREML_MIN_ALIGNMENT: usize = 16 * 1024;
const TRANSFER_CHUNK_BYTES: usize = 16 * 1024;
const COREML_EXTERNAL_DOMAIN: u32 = 0x434d_4c45;
const EXCLUSIVE_NATIVE_ACCESS: u64 = 1 << 63;
const MAX_SHARED_NATIVE_USERS: u64 = EXCLUSIVE_NATIVE_ACCESS - 1;
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

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

impl NativeBinding {
    const EMPTY: Self = Self {
        slot: 0,
        access: 0,
        data: core::ptr::null_mut(),
        bytes: 0,
    };
}

#[derive(Clone, Copy, Debug)]
struct SlotPlan {
    slot: u32,
    access: u8,
}

type NativeReleaseContext = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" {
    fn va_coreml_has_neural_engine() -> std::ffi::c_int;
    fn va_coreml_supports_int8() -> std::ffi::c_int;
    fn va_coreml_model_load(
        path: *const u8,
        path_len: usize,
        mappings: *const NativeFeatureMapping,
        mapping_count: usize,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn va_coreml_model_load_memory(
        bytes: *const u8,
        bytes_len: usize,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackingAccess {
    Shared,
    Exclusive,
}

impl BackingAccess {
    const fn for_binding(access: AccessMode) -> Self {
        match access {
            AccessMode::Read => Self::Shared,
            AccessMode::Write | AccessMode::ReadWrite => Self::Exclusive,
        }
    }

    const fn combine(self, other: Self) -> Self {
        if matches!(self, Self::Exclusive) || matches!(other, Self::Exclusive) {
            Self::Exclusive
        } else {
            Self::Shared
        }
    }
}

#[derive(Debug)]
struct EventBacking {
    key: usize,
    allocation: Arc<AlignedAllocation>,
    access: BackingAccess,
    acquired: bool,
}

impl EventBacking {
    fn new(allocation: Arc<AlignedAllocation>, access: AccessMode) -> Self {
        Self {
            key: Arc::as_ptr(&allocation) as usize,
            allocation,
            access: BackingAccess::for_binding(access),
            acquired: false,
        }
    }

    fn acquire(&mut self) -> Result<(), BackendError> {
        match self.access {
            BackingAccess::Shared => {
                self.allocation
                    .in_flight
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                        (current < MAX_SHARED_NATIVE_USERS).then_some(current + 1)
                    })
                    .map_err(|_| BackendError::Busy)?;
            }
            BackingAccess::Exclusive => {
                self.allocation
                    .in_flight
                    .compare_exchange(
                        0,
                        EXCLUSIVE_NATIVE_ACCESS,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .map_err(|_| BackendError::Busy)?;
            }
        }
        self.acquired = true;
        Ok(())
    }
}

fn prepare_event_backings(backings: &mut Vec<EventBacking>) -> Result<(), BackendError> {
    backings.sort_unstable_by_key(|backing| backing.key);
    backings.dedup_by(|current, previous| {
        if current.key != previous.key {
            return false;
        }
        previous.access = previous.access.combine(current.access);
        true
    });
    for backing in backings {
        backing.acquire()?;
    }
    Ok(())
}

impl Drop for EventBacking {
    fn drop(&mut self) {
        if !self.acquired {
            return;
        }
        match self.access {
            BackingAccess::Shared => {
                let prior = self.allocation.in_flight.fetch_sub(1, Ordering::Release);
                debug_assert!((1..=MAX_SHARED_NATIVE_USERS).contains(&prior));
            }
            BackingAccess::Exclusive => {
                let result = self.allocation.in_flight.compare_exchange(
                    EXCLUSIVE_NATIVE_ACCESS,
                    0,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
                debug_assert_eq!(result, Ok(EXCLUSIVE_NATIVE_ACCESS));
            }
        }
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
#[derive(Debug)]
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
    slots: Vec<SlotPlan>,
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
#[derive(Debug)]
pub struct CoreMlQueue {
    context_id: u64,
    native_bindings: RefCell<Vec<NativeBinding>>,
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

/// One Core ML backend instance.
#[derive(Debug)]
pub struct CoreMlAccelerator {
    model_root: Option<PathBuf>,
    next_id: AtomicU64,
    direct_binding_admissions: AtomicU64,
    explicit_transfer_bytes: AtomicU64,
    info: DeviceInfo,
}

impl CoreMlAccelerator {
    /// Construct a backend with the production TOSA-to-Core ML path and require an accessible ANE.
    pub fn new_tosa() -> Result<Self, InitError> {
        Self::with_model_root(None)
    }

    /// Construct a backend with TOSA lowering plus the legacy host-path artifact escape hatch.
    ///
    /// New device integrations should use [`Self::new_tosa`]. The model root exists only for
    /// explicitly host-owned Core ML artifacts and is never encoded into portable TOSA payloads.
    pub fn new(model_root: impl AsRef<Path>) -> Result<Self, InitError> {
        let model_root = model_root
            .as_ref()
            .canonicalize()
            .map_err(|_| InitError::InvalidModelRoot)?;
        if !model_root.is_dir() || model_root.to_str().is_none() {
            return Err(InitError::InvalidModelRoot);
        }
        Self::with_model_root(Some(model_root))
    }

    fn with_model_root(model_root: Option<PathBuf>) -> Result<Self, InitError> {
        // SAFETY: the function has no pointer arguments and returns a scalar availability result.
        if unsafe { va_coreml_has_neural_engine() } == 0 {
            return Err(InitError::NeuralEngineUnavailable);
        }
        Ok(Self {
            model_root,
            next_id: AtomicU64::new(1),
            direct_binding_admissions: AtomicU64::new(0),
            explicit_transfer_bytes: AtomicU64::new(0),
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
                    max_artifact_bytes: MAX_TOSA_ARTIFACT_BYTES,
                },
            },
        })
    }

    /// Number of exact provider allocations admitted as Core ML bindings.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.load(Ordering::Relaxed)
    }

    /// Bytes copied through the two explicit transfer methods.
    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.explicit_transfer_bytes.load(Ordering::Relaxed)
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| BackendError::ResourceLimit)
    }

    fn resolve_model(&self, relative: &str) -> Result<PathBuf, BackendError> {
        let model_root = self.model_root.as_ref().ok_or(BackendError::Unsupported)?;
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(BackendError::PermissionDenied);
        }
        let resolved = model_root
            .join(path)
            .canonicalize()
            .map_err(|_| BackendError::InvalidArgument)?;
        if !resolved.starts_with(model_root) || resolved.as_os_str().to_str().is_none() {
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

    fn lowering_error(error: LoweringError) -> BackendError {
        match error {
            LoweringError::Parse(_)
            | LoweringError::Analysis(_)
            | LoweringError::InvalidConstant => BackendError::InvalidArgument,
            LoweringError::UnsupportedGraph
            | LoweringError::UnsupportedType(_)
            | LoweringError::UnsupportedOperator(_) => BackendError::Unsupported,
            LoweringError::ResourceLimit => BackendError::ResourceLimit,
        }
    }
}

impl TosaCapabilityProvider for CoreMlAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        // SAFETY: this scalar query has no pointer arguments and reports whether the host OS can
        // load the separately described integer ML Program tier.
        if unsafe { va_coreml_supports_int8() } == 1 {
            crate::ALL_CAPABILITIES
        } else {
            crate::FLOAT_CAPABILITIES
        }
    }
}

fn build_slot_plan(
    mappings: &[crate::artifact::FeatureMapping],
) -> Result<Vec<SlotPlan>, BackendError> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(mappings.len())
        .map_err(|_| BackendError::OutOfMemory)?;
    for mapping in mappings {
        slots.push(SlotPlan {
            slot: mapping.slot,
            access: match mapping.role {
                FeatureRole::Input => AccessMode::Read as u8,
                FeatureRole::Output => AccessMode::Write as u8,
            },
        });
    }
    slots.sort_unstable_by_key(|slot| slot.slot);
    slots.dedup_by(|current, previous| {
        if current.slot != previous.slot {
            return false;
        }
        previous.access |= current.access;
        true
    });
    Ok(slots)
}

fn build_lowered_slot_plan(mappings: &[LoweredFeature]) -> Vec<SlotPlan> {
    mappings
        .iter()
        .map(|mapping| SlotPlan {
            slot: mapping.slot,
            access: match mapping.role {
                LoweredFeatureRole::Input => AccessMode::Read as u8,
                LoweredFeatureRole::Output => AccessMode::Write as u8,
            },
        })
        .collect()
}

unsafe extern "C" fn release_event_backings(context: *mut c_void) {
    // SAFETY: successful submission transfers exactly one pointer produced by `Box::into_raw` to
    // the bridge, and its completion block calls this function exactly once.
    drop(unsafe { Box::from_raw(context.cast::<Vec<EventBacking>>()) });
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
            self.explicit_transfer_bytes
                .fetch_add(target_len as u64, Ordering::Relaxed);
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
            self.explicit_transfer_bytes
                .fetch_add(chunk as u64, Ordering::Relaxed);
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
            self.explicit_transfer_bytes
                .fetch_add(source_len as u64, Ordering::Relaxed);
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
            self.explicit_transfer_bytes
                .fetch_add(chunk as u64, Ordering::Relaxed);
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
        if artifact.payload.len() > self.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.resident_bytes != REQUIRED_RESIDENT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.format == virtio_accel_tosa::ARTIFACT_FORMAT {
            let target = virtio_accel_tosa::Target::from_identity(artifact.target)
                .map_err(|_| BackendError::Incompatible)?;
            if target == crate::COREML_TOSA_INTEGER_TARGET {
                // SAFETY: this query takes no pointers and reports the Foundation runtime's
                // operating-system version check. INT8 model boundaries are unavailable before
                // macOS 26 even though this binary weak-links the newer SDK declarations.
                if unsafe { va_coreml_supports_int8() } != 1 {
                    return Err(BackendError::Unsupported);
                }
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
            let lowered = lower_tosa(bytes, target).map_err(Self::lowering_error)?;
            let slots = build_lowered_slot_plan(&lowered.features);
            let mappings = lowered
                .features
                .iter()
                .map(|mapping| NativeFeatureMapping {
                    slot: mapping.slot,
                    role: match mapping.role {
                        LoweredFeatureRole::Input => 1,
                        LoweredFeatureRole::Output => 2,
                    },
                    name: mapping.name.as_ptr(),
                    name_len: mapping.name.len(),
                })
                .collect::<Vec<_>>();
            let mut error = NativeError::default();
            // SAFETY: model and feature-name bytes remain valid for this synchronous call. The
            // bridge writes the model to a unique temporary source, compiles and loads it, removes
            // the source, copies retained strings, and returns one owned model reference.
            let native = unsafe {
                va_coreml_model_load_memory(
                    lowered.bytes.as_ptr(),
                    lowered.bytes.len(),
                    mappings.as_ptr(),
                    mappings.len(),
                    &mut error,
                )
            };
            let native = NonNull::new(native).ok_or_else(|| Self::native_error(error))?;
            return Ok(CoreMlProgram {
                context_id: context.id,
                native,
                slots,
            });
        }

        if artifact.format != ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        if artifact.target != TARGET_IDENTITY {
            return Err(BackendError::Incompatible);
        }
        if artifact.payload.len() > MAX_ARTIFACT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        let decoded = decode(artifact.payload).map_err(|error| match error {
            DecodeError::Invalid => BackendError::InvalidArgument,
            DecodeError::OutOfBounds => BackendError::OutOfBounds,
            DecodeError::ResourceLimit => BackendError::ResourceLimit,
        })?;
        let model_path = self.resolve_model(&decoded.model_path)?;
        let slots = build_slot_plan(&decoded.mappings)?;
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
            slots,
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
            native_bindings: RefCell::new(Vec::new()),
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
        if bindings.is_empty()
            || bindings.len() > self.info.limits.max_bindings_per_submission as usize
        {
            return Err(SubmitFailure::Rejected(BackendError::ResourceLimit));
        }
        if queue.context_id != program.context_id {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }
        let mut native_bindings = queue.native_bindings.borrow_mut();
        native_bindings.clear();
        native_bindings
            .try_reserve(program.slots.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        native_bindings.resize(program.slots.len(), NativeBinding::EMPTY);
        let mut event_backings = Vec::new();
        event_backings
            .try_reserve_exact(bindings.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        let mut seen = [0_u64; 4];
        for binding in bindings {
            if binding.buffer.context_id != queue.context_id {
                return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
            }
            if !binding.buffer.desc.allows_access(binding.access) {
                return Err(SubmitFailure::Rejected(BackendError::PermissionDenied));
            }
            let (start, _) =
                Self::checked_range(binding.buffer, binding.range.offset, binding.range.bytes())
                    .map_err(SubmitFailure::Rejected)?;
            let index = program
                .slots
                .binary_search_by_key(&binding.slot, |slot| slot.slot)
                .map_err(|_| SubmitFailure::Rejected(BackendError::Incompatible))?;
            let bit = 1_u64 << (index % 64);
            if seen[index / 64] & bit != 0 {
                return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
            }
            seen[index / 64] |= bit;
            if binding.access as u8 != program.slots[index].access {
                return Err(SubmitFailure::Rejected(BackendError::Incompatible));
            }
            native_bindings[index] = NativeBinding {
                slot: binding.slot,
                access: binding.access as u8,
                data: binding.buffer.allocation.pointer_at(start).cast(),
                bytes: binding.range.bytes(),
            };
            event_backings.push(EventBacking::new(
                Arc::clone(&binding.buffer.allocation),
                binding.access,
            ));
        }
        if bindings.len() != program.slots.len() {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }

        prepare_event_backings(&mut event_backings).map_err(SubmitFailure::Rejected)?;
        let backings = Box::new(event_backings);
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
                drop(unsafe { Box::from_raw(backing_context.cast::<Vec<EventBacking>>()) });
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
    fn native_backing_guard_allows_shared_users_and_excludes_writers() {
        let allocation = Arc::new(AlignedAllocation::new(32, 16).unwrap());
        let mut first = EventBacking::new(Arc::clone(&allocation), AccessMode::Read);
        first.acquire().unwrap();
        let mut second = EventBacking::new(Arc::clone(&allocation), AccessMode::Read);
        second.acquire().unwrap();
        let mut blocked_writer = EventBacking::new(Arc::clone(&allocation), AccessMode::Write);
        assert_eq!(blocked_writer.acquire().unwrap_err(), BackendError::Busy);
        drop(first);
        drop(second);
        let mut writer = EventBacking::new(Arc::clone(&allocation), AccessMode::ReadWrite);
        writer.acquire().unwrap();
        let mut blocked_reader = EventBacking::new(allocation, AccessMode::Read);
        assert_eq!(blocked_reader.acquire().unwrap_err(), BackendError::Busy);
        drop(writer);
    }

    #[test]
    fn duplicate_allocation_access_collapses_to_exclusive() {
        let allocation = Arc::new(AlignedAllocation::new(32, 16).unwrap());
        let mut pending = vec![
            EventBacking::new(Arc::clone(&allocation), AccessMode::Read),
            EventBacking::new(allocation, AccessMode::Write),
        ];
        prepare_event_backings(&mut pending).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].access, BackingAccess::Exclusive);
    }

    #[test]
    fn slot_plan_is_sorted_and_combines_input_output_access() {
        let mappings = vec![
            crate::artifact::FeatureMapping {
                slot: 8,
                role: FeatureRole::Output,
                name: "y".into(),
            },
            crate::artifact::FeatureMapping {
                slot: 7,
                role: FeatureRole::Input,
                name: "x".into(),
            },
            crate::artifact::FeatureMapping {
                slot: 8,
                role: FeatureRole::Input,
                name: "state".into(),
            },
        ];
        let plan = build_slot_plan(&mappings).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].slot, 7);
        assert_eq!(plan[0].access, AccessMode::Read as u8);
        assert_eq!(plan[1].slot, 8);
        assert_eq!(plan[1].access, AccessMode::ReadWrite as u8);
    }
}

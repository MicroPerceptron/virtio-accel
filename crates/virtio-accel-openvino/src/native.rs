//! Native OpenVINO runtime integration: the audited `Accelerator` implementation.
//!
//! Every `unsafe` block carries a `SAFETY:` comment referencing the invariants audited in
//! `SAFETY.md`. Each OpenVINO handle has exactly one Rust owner with a `Drop` implementation;
//! the shared `ov_core_t` is process-wide. A request-owned callback only publishes
//! a notification; the owning thread observes native status and releases storage.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, RefCell, RefMut};
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, CompletionNotification, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits,
    EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_tosa::{CapabilityDescriptor, TosaCapabilityProvider};

use crate::lower::{LoweredFeature, LoweredFeatureRole, OvElement, lower_tosa_with_weights};
use crate::{InitError, REQUIRED_RESIDENT_BYTES, ffi};

/// Maximal TOSA artifact bytes admitted before parsing.
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// Minimum host allocation alignment. Individual plugins may still stage data internally.
const OPENVINO_MIN_ALIGNMENT: usize = 4096;

/// Stack scratch size for segmented explicit transfers.
const TRANSFER_CHUNK_BYTES: usize = 16 * 1024;

/// Stable provider-owned error namespace for unmapped OpenVINO statuses (`"OVNO"`).
const OPENVINO_EXTERNAL_DOMAIN: u32 = 0x4f56_4e4f;

const EXCLUSIVE_NATIVE_ACCESS: u64 = 1 << 63;
const MAX_SHARED_NATIVE_USERS: u64 = EXCLUSIVE_NATIVE_ACCESS - 1;

/// Owned `ov_core_t`; in practice held by [`shared_core`] for the process lifetime.
struct CoreHandle {
    core: NonNull<ffi::ov_core_t>,
}

// SAFETY: `ov::Core` is documented thread-safe by OpenVINO and the C wrapper adds no thread
// affinity; the handle is freed exactly once by `Drop`.
unsafe impl Send for CoreHandle {}
// SAFETY: shared `&CoreHandle` use only reaches thread-safe `ov_core_*` entry points.
unsafe impl Sync for CoreHandle {}

/// The process-wide OpenVINO core.
///
/// Exactly one `ov_core_t` is created per process and lives until process exit. Re-initializing
/// the runtime is not crash-safe: a second `ov_core_create` re-creates plugin engines, and the
/// Intel NPU plugin's second `zeInitDrivers` call segfaults inside the Level Zero loader
/// (observed with ze_loader 1.28 and OpenVINO 2026.3 on hosts without a vendor driver). Every
/// accelerator instance shares this core; contexts, buffers, programs, queues, and events remain
/// per-instance state.
fn shared_core() -> Result<Arc<CoreHandle>, InitError> {
    ffi::ensure_loaded()?;
    static CORE: OnceLock<Result<Arc<CoreHandle>, InitError>> = OnceLock::new();
    CORE.get_or_init(|| CoreHandle::create().map(Arc::new))
        .clone()
}

impl CoreHandle {
    fn create() -> Result<Self, InitError> {
        let mut core = ptr::null_mut();
        // SAFETY: `core` is a valid out-pointer; the runtime initializes it only on OK.
        let status = unsafe { (ffi::api().ov_core_create)(&mut core) };
        if status != ffi::OV_STATUS_OK {
            return Err(InitError::CoreCreationFailed);
        }
        NonNull::new(core)
            .map(|core| Self { core })
            .ok_or(InitError::CoreCreationFailed)
    }

    const fn as_const_ptr(&self) -> *const ffi::ov_core_t {
        self.core.as_ptr()
    }

    /// Enumerate inference device names through this core.
    fn available_devices(&self) -> Result<Vec<String>, InitError> {
        let mut devices = ffi::ov_available_devices_t {
            devices: ptr::null_mut(),
            size: 0,
        };
        // SAFETY: the core is live and `devices` is a valid out-structure the runtime fills on
        // OK; on failure it stays in its zeroed state and must not be freed.
        let status = unsafe {
            (ffi::api().ov_core_get_available_devices)(self.as_const_ptr(), &mut devices)
        };
        if status != ffi::OV_STATUS_OK {
            return Err(InitError::DeviceEnumerationFailed);
        }
        let mut names = Vec::with_capacity(devices.size);
        // SAFETY: on OK the runtime owns `devices.size` NUL-terminated names; they are copied
        // before the single required `ov_available_devices_free` call releases them.
        unsafe {
            for index in 0..devices.size {
                let name = *devices.devices.add(index);
                if !name.is_null() {
                    names.push(CStr::from_ptr(name).to_string_lossy().into_owned());
                }
            }
            (ffi::api().ov_available_devices_free)(&mut devices);
        }
        Ok(names)
    }
}

impl Drop for CoreHandle {
    fn drop(&mut self) {
        // SAFETY: the handle owns exactly one core reference and is dropped exactly once.
        unsafe { (ffi::api().ov_core_free)(self.core.as_ptr()) }
    }
}

fn backend_error_from_status(status: ffi::ov_status_e) -> BackendError {
    match status {
        ffi::OV_STATUS_NOT_IMPLEMENTED | ffi::OV_STATUS_NOT_IMPLEMENT_C_METHOD => {
            BackendError::Unsupported
        }
        ffi::OV_STATUS_PARAMETER_MISMATCH => BackendError::Incompatible,
        ffi::OV_STATUS_NOT_FOUND | ffi::OV_STATUS_INVALID_C_PARAM => BackendError::InvalidArgument,
        ffi::OV_STATUS_OUT_OF_BOUNDS => BackendError::OutOfBounds,
        ffi::OV_STATUS_REQUEST_BUSY => BackendError::Busy,
        ffi::OV_STATUS_NOT_ALLOCATED => BackendError::OutOfMemory,
        other => BackendError::External {
            domain: OPENVINO_EXTERNAL_DOMAIN,
            code: i64::from(other),
        },
    }
}

fn check_status(status: ffi::ov_status_e) -> Result<(), BackendError> {
    if status == ffi::OV_STATUS_OK {
        Ok(())
    } else {
        Err(backend_error_from_status(status))
    }
}

/// Page-aligned buffer backing directly wrapped by OpenVINO tensors.
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
        let alignment = requested_alignment.max(OPENVINO_MIN_ALIGNMENT);
        let allocation_bytes = bytes
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or(BackendError::OutOfMemory)?;
        let layout = Layout::from_size_align(allocation_bytes, alignment)
            .map_err(|_| BackendError::OutOfMemory)?;
        // SAFETY: `layout` is valid and nonzero. The returned pointer is owned by this value and
        // is released with the identical layout in `Drop`.
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

// SAFETY: mutable storage is accessed only through guarded backend methods; frozen
// SharedInput storage has no writer. Buffer handles remain !Send/!Sync. Host views
// and native events share the atomic access gate. See `SAFETY.md`.
unsafe impl Send for AlignedAllocation {}
// SAFETY: same invariant as the `Send` implementation.
unsafe impl Sync for AlignedAllocation {}

impl Drop for AlignedAllocation {
    fn drop(&mut self) {
        // SAFETY: `pointer` was returned by `alloc_zeroed(self.layout)` and has not been freed.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// Immutable, page-aligned host input that can be imported by several OpenVINO
/// contexts/devices without duplicating its bytes. Cloning shares ownership.
/// Initialization happens before publication; no mutable access is exposed later.
/// This is a local host API, not a wire-protocol external-memory capability.
#[derive(Clone, Debug)]
pub struct SharedInput {
    allocation: Arc<AlignedAllocation>,
    bytes: usize,
}

impl SharedInput {
    /// Allocate zero-initialized storage and fill it in place before freezing it.
    /// The callback cannot retain a borrow of the allocation. No runtime is needed.
    pub fn initialize(
        bytes: u64,
        alignment: u64,
        fill: impl FnOnce(&mut [u8]) -> Result<(), BackendError>,
    ) -> Result<Self, BackendError> {
        BufferDesc::new(
            bytes,
            alignment,
            MemoryDomain::Shared,
            BufferUsage::PROGRAM_INPUT,
        )?;
        let allocation = AlignedAllocation::new(bytes, alignment)?;
        let bytes = usize::try_from(bytes).map_err(|_| BackendError::OutOfMemory)?;
        // SAFETY: newly owned initialized allocation, validated logical length,
        // and no native/host aliases exist before this exclusive callback ends.
        fill(unsafe { std::slice::from_raw_parts_mut(allocation.pointer.as_ptr(), bytes) })?;
        Ok(Self {
            allocation: Arc::new(allocation),
            bytes,
        })
    }

    /// Borrow the frozen input. Imported descriptors forbid every write access.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: immutable allocation retained by self. Every import is read-only,
        // so neither safe host APIs nor native bindings can mutate these bytes.
        unsafe { std::slice::from_raw_parts(self.allocation.pointer.as_ptr(), self.bytes) }
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

/// In-flight guard for one allocation bound to one submission.
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

    // `try_update` is not available on our Rust 1.85 MSRV. Keep the equivalent
    // `fetch_update` operation until the minimum compiler supports its new name.
    #[allow(deprecated)]
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

/// Collapse duplicate allocations to their strongest access, then acquire every guard.
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

/// Owned `ov_shape_t`, created once per program slot and reused across submissions.
struct OwnedShape {
    shape: ffi::ov_shape_t,
}

impl OwnedShape {
    fn new(dims: &[i64]) -> Result<Self, BackendError> {
        let mut shape = ffi::ov_shape_t {
            rank: 0,
            dims: ptr::null_mut(),
        };
        // SAFETY: `dims` is a live slice and `shape` a valid out-structure; the runtime copies
        // the dims into its own allocation, released exactly once in `Drop`.
        check_status(unsafe {
            (ffi::api().ov_shape_create)(dims.len() as i64, dims.as_ptr(), &mut shape)
        })?;
        Ok(Self { shape })
    }

    /// By-value view for C calls that copy the shape (tensor creation).
    const fn as_value(&self) -> ffi::ov_shape_t {
        self.shape
    }
}

impl Drop for OwnedShape {
    fn drop(&mut self) {
        // SAFETY: `shape` was initialized by `ov_shape_create` and is freed exactly once.
        let _ = unsafe { (ffi::api().ov_shape_free)(&mut self.shape) };
    }
}

/// Remote context owned by the same resident program as its idle requests.
struct NpuContext {
    context: NonNull<ffi::ov_remote_context_t>,
    api: &'static ffi::RemoteApi,
}

impl NpuContext {
    fn for_compiled(compiled: &CompiledModelHandle, device: &str) -> Option<Self> {
        if !cfg!(target_os = "linux") || device.split('.').next() != Some("NPU") {
            return None;
        }
        let api = ffi::api().remote.as_ref()?;
        let mut context = ptr::null_mut();
        // SAFETY: compiled is live and the runtime initializes the out-pointer on success.
        check_status(unsafe {
            (api.ov_compiled_model_get_context)(compiled.as_const_ptr(), &mut context)
        })
        .ok()?;
        NonNull::new(context).map(|context| Self { context, api })
    }

    fn import(&self, plan: &SlotPlan, data: *mut c_void) -> Option<TensorHandle> {
        if plan.role != LoweredFeatureRole::Input
            || data as usize % OPENVINO_MIN_ALIGNMENT != 0
            || plan.byte_len % OPENVINO_MIN_ALIGNMENT as u64 != 0
        {
            return None;
        }
        let mut tensor = ptr::null_mut();
        // SAFETY: submit validated the whole range. BoundTensor retains these immutable
        // pages until the last request using this import is destroyed. The C API takes
        // six variadic arguments: MEM_HANDLE is a void pointer, the rest are C strings.
        check_status(unsafe {
            (self.api.ov_remote_context_create_tensor)(
                self.context.as_ptr(),
                plan.element,
                plan.shape.as_value(),
                6,
                &mut tensor,
                c"MEM_TYPE".as_ptr(),
                c"CPU_VA".as_ptr(),
                c"MEM_HANDLE".as_ptr(),
                data,
                c"TENSOR_TYPE".as_ptr(),
                c"INPUT".as_ptr(),
            )
        })
        .ok()?;
        NonNull::new(tensor).map(|tensor| TensorHandle { tensor })
    }
}

impl Drop for NpuContext {
    fn drop(&mut self) {
        // SAFETY: one owned remote context, released after the idle request pool.
        unsafe { (self.api.ov_remote_context_free)(self.context.as_ptr()) }
    }
}

/// Owned `ov_tensor_t`.
struct TensorHandle {
    tensor: NonNull<ffi::ov_tensor_t>,
}

impl TensorHandle {
    /// Wrap caller-owned memory; the tensor never frees `data`, whose region must stay live and
    /// unmoved for this handle's lifetime.
    fn from_host_ptr(
        element: ffi::ov_element_type_e,
        shape: ffi::ov_shape_t,
        data: *mut c_void,
    ) -> Result<Self, BackendError> {
        let mut tensor = ptr::null_mut();
        // SAFETY: the shape is initialized, `data` addresses a live region large enough for the
        // shape/element pair (validated by the caller), and the runtime wraps it without
        // ownership.
        check_status(unsafe {
            (ffi::api().ov_tensor_create_from_host_ptr)(element, shape, data, &mut tensor)
        })?;
        NonNull::new(tensor)
            .map(|tensor| Self { tensor })
            .ok_or(BackendError::External {
                domain: OPENVINO_EXTERNAL_DOMAIN,
                code: 0,
            })
    }

    const fn as_const_ptr(&self) -> *const ffi::ov_tensor_t {
        self.tensor.as_ptr()
    }
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        // SAFETY: this handle owns one tensor reference and is dropped exactly once.
        unsafe { (ffi::api().ov_tensor_free)(self.tensor.as_ptr()) }
    }
}

/// One event-owned tensor plus the information needed to verify direct output backing.
///
/// Inputs use a null `expected_output`; output buffer pointers are always non-null because they
/// originate in a live [`AlignedAllocation`]. Keeping the verification metadata beside the
/// tensor removes a second per-event allocation.
struct BoundTensor {
    imported: bool,
    _tensor: TensorHandle,
    // Idle requests retain storage without retaining an access guard.
    _allocation: Arc<AlignedAllocation>,
    data: *mut c_void,
    output_io_index: usize,
    expected_output: *mut c_void,
}

#[derive(Default)]
struct EventMetadata {
    tensors: Vec<BoundTensor>,
    backings: Vec<EventBacking>,
}

/// Queue-owned warm-path metadata capacity.
///
/// Only one empty vector of each kind is retained. Returned vectors are cleared before entering
/// the spare slots, so this object never owns an OpenVINO handle, a buffer backing guard, or a
/// guest-memory pointer. Events hold a weak reference so outstanding work cannot keep a queue
/// alive after its owner destroys it.
#[derive(Default)]
struct QueueScratch {
    pointer_slots: RefCell<Vec<*mut c_void>>,
    metadata_spare: RefCell<Option<EventMetadata>>,
}

impl QueueScratch {
    fn take_pointer_slots(&self, count: usize) -> Result<PointerSlots<'_>, BackendError> {
        let mut slots = self.pointer_slots.borrow_mut();
        slots.clear();
        slots
            .try_reserve(count)
            .map_err(|_| BackendError::OutOfMemory)?;
        slots.resize(count, ptr::null_mut());
        Ok(PointerSlots { slots })
    }

    fn take_metadata(
        &self,
        backing_count: usize,
        tensor_count: usize,
    ) -> Result<EventMetadata, BackendError> {
        let mut metadata = self.metadata_spare.borrow_mut().take().unwrap_or_default();
        debug_assert!(metadata.backings.is_empty());
        debug_assert!(metadata.tensors.is_empty());
        metadata
            .backings
            .try_reserve_exact(backing_count)
            .map_err(|_| BackendError::OutOfMemory)?;
        metadata
            .tensors
            .try_reserve_exact(tensor_count)
            .map_err(|_| BackendError::OutOfMemory)?;
        Ok(metadata)
    }

    fn recycle_metadata(&self, mut returned: EventMetadata) {
        returned.tensors.clear();
        returned.backings.clear();
        let mut spare = self.metadata_spare.borrow_mut();
        let returned_capacity = returned
            .tensors
            .capacity()
            .saturating_add(returned.backings.capacity());
        if spare.as_ref().is_none_or(|current| {
            current
                .tensors
                .capacity()
                .saturating_add(current.backings.capacity())
                < returned_capacity
        }) {
            *spare = Some(returned);
        }
    }
}

impl std::fmt::Debug for QueueScratch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QueueScratch")
            .field("pointer_capacity", &self.pointer_slots.borrow().capacity())
            .field(
                "backing_capacity",
                &self
                    .metadata_spare
                    .borrow()
                    .as_ref()
                    .map_or(0, |metadata| metadata.backings.capacity()),
            )
            .field(
                "tensor_capacity",
                &self
                    .metadata_spare
                    .borrow()
                    .as_ref()
                    .map_or(0, |metadata| metadata.tensors.capacity()),
            )
            .finish()
    }
}

/// Borrowed queue pointer slots that are scrubbed on every exit path.
///
/// Submission validation needs random slot lookup, but the queue must not retain guest pointers
/// after `submit` returns, including on rejection. The guard makes that property independent of
/// every `?` in the admission and native-binding phases.
struct PointerSlots<'a> {
    slots: RefMut<'a, Vec<*mut c_void>>,
}

impl Deref for PointerSlots<'_> {
    type Target = Vec<*mut c_void>;

    fn deref(&self) -> &Self::Target {
        &self.slots
    }
}

impl DerefMut for PointerSlots<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slots
    }
}

impl Drop for PointerSlots<'_> {
    fn drop(&mut self) {
        self.slots.clear();
    }
}

/// Owned `ov_model_t`; a short-lived reader artifact freed right after compilation.
struct ModelHandle {
    model: NonNull<ffi::ov_model_t>,
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        // SAFETY: this handle owns one model reference and is dropped exactly once.
        unsafe { (ffi::api().ov_model_free)(self.model.as_ptr()) }
    }
}

/// Owned `ov_compiled_model_t`.
struct CompiledModelHandle {
    compiled: NonNull<ffi::ov_compiled_model_t>,
}

impl CompiledModelHandle {
    const fn as_const_ptr(&self) -> *const ffi::ov_compiled_model_t {
        self.compiled.as_ptr()
    }
}

impl Drop for CompiledModelHandle {
    fn drop(&mut self) {
        // SAFETY: this handle owns one compiled-model reference and is dropped exactly once.
        unsafe { (ffi::api().ov_compiled_model_free)(self.compiled.as_ptr()) }
    }
}

/// Read an IR document into a model, sharing the caller's weights memory when there is any.
fn read_model(
    core: &CoreHandle,
    xml: &[u8],
    weights: Option<&TensorHandle>,
) -> Result<ModelHandle, BackendError> {
    let mut model = ptr::null_mut();
    // SAFETY: core, document bytes, and the (possibly null) weights tensor are live for this
    // synchronous call; the created model shares the weights memory, which the caller keeps
    // resident for as long as the model lives.
    let status = unsafe {
        (ffi::api().ov_core_read_model_from_memory_buffer)(
            core.as_const_ptr(),
            xml.as_ptr().cast(),
            xml.len(),
            weights.map_or(ptr::null(), TensorHandle::as_const_ptr),
            &mut model,
        )
    };
    check_status(status)?;
    NonNull::new(model)
        .map(|model| ModelHandle { model })
        .ok_or(BackendError::External {
            domain: OPENVINO_EXTERNAL_DOMAIN,
            code: 0,
        })
}

/// Whether this device's compiler accepts an FP8 graph, asked by compiling the smallest one.
///
/// Advertised capability has to be true of the device in hand, not of the backend in general.
/// Intel NPU arch 5010 (Panther Lake) compiles FP8; arch 40XX (Lunar Lake) refuses even an FP8
/// `IDENTITY`, and the FP8 handling in the driver-side compiler's own tree is arch-gated. There
/// is no property to read for this -- `OPTIMIZATION_CAPABILITIES` omits FP8 on 5010, where it
/// works -- so the tier is probed rather than branded, and a driver that gains FP8 later starts
/// advertising it here with no allowlist to edit.
fn device_accepts_fp8(core: &CoreHandle, device: &CStr) -> bool {
    let probe = crate::lower::fp8_probe_document();
    let Ok(model) = read_model(core, &probe.xml, None) else {
        return false;
    };
    let accepted = compile_with_accuracy(core, &model, device).is_ok();
    drop(model);
    accepted
}

fn device_accepts_quantized_fp32(core: &CoreHandle, device: &CStr) -> bool {
    let probe = crate::lower::fp32_probe_document();
    let Ok(model) = read_model(core, &probe.xml, None) else {
        return false;
    };
    compile_with_precision(core, &model, device, true).is_ok()
}

/// Compile existing native tiers with the ACCURACY hint. Quantized FP32 paths use
/// `compile_with_precision` because the hint alone does not guarantee FP32 on every plugin.
fn compile_with_accuracy(
    core: &CoreHandle,
    model: &ModelHandle,
    device: &CStr,
) -> Result<CompiledModelHandle, BackendError> {
    compile_with_precision(core, model, device, false)
}

fn compile_with_precision(
    core: &CoreHandle,
    model: &ModelHandle,
    device: &CStr,
    fp32: bool,
) -> Result<CompiledModelHandle, BackendError> {
    let mut compiled = ptr::null_mut();
    let disable_dynamic_quantization =
        device.to_bytes().starts_with(b"CPU") || device.to_bytes().starts_with(b"GPU");
    // SAFETY: core, model, and device are live; `property_args_size` counts the variadic
    // arguments (one key/value pair contributes two) per the documented convention pinned by a
    // unit test. All keys/values are static C strings (or the retained runtime key).
    // CPU/GPU otherwise insert lossy activation quantization for compressed weights,
    // even with ACCURACY. NPU does not expose this property. Optional pairs form a
    // contiguous prefix, with the count matching exactly; all pointers outlive the call.
    let status = unsafe {
        (ffi::api().ov_core_compile_model)(
            core.as_const_ptr(),
            model.model.as_ptr(),
            device.as_ptr(),
            2 + 2 * usize::from(disable_dynamic_quantization) + 2 * usize::from(fp32),
            &mut compiled,
            ffi::execution_mode_property(),
            c"ACCURACY".as_ptr(),
            if disable_dynamic_quantization {
                c"DYNAMIC_QUANTIZATION_GROUP_SIZE".as_ptr()
            } else {
                c"INFERENCE_PRECISION_HINT".as_ptr()
            },
            if disable_dynamic_quantization {
                c"0".as_ptr()
            } else {
                c"f32".as_ptr()
            },
            c"INFERENCE_PRECISION_HINT".as_ptr(),
            c"f32".as_ptr(),
        )
    };
    check_status(status)?;
    NonNull::new(compiled)
        .map(|compiled| CompiledModelHandle { compiled })
        .ok_or(BackendError::External {
            domain: OPENVINO_EXTERNAL_DOMAIN,
            code: 0,
        })
}

const fn element_code(element: OvElement) -> ffi::ov_element_type_e {
    match element {
        OvElement::F32 => ffi::ELEMENT_F32,
        OvElement::F16 => ffi::ELEMENT_F16,
        OvElement::F8E4M3 => ffi::ELEMENT_F8E4M3,
        OvElement::F8E5M2 => ffi::ELEMENT_F8E5M2,
        OvElement::I8 => ffi::ELEMENT_I8,
        OvElement::I4 => panic!("i4 is internal constant storage, never a boundary tensor"),
        OvElement::I32 => ffi::ELEMENT_I32,
        OvElement::I64 => ffi::ELEMENT_I64,
        OvElement::Bool => ffi::ELEMENT_BOOLEAN,
    }
}

/// One binding slot of a loaded program.
struct SlotPlan {
    slot: u32,
    access: u8,
    role: LoweredFeatureRole,
    io_index: usize,
    element: ffi::ov_element_type_e,
    scalar_bytes: u64,
    byte_len: u64,
    shape: OwnedShape,
}

fn build_slot_plan(features: &[LoweredFeature]) -> Result<Vec<SlotPlan>, BackendError> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(features.len())
        .map_err(|_| BackendError::OutOfMemory)?;
    for feature in features {
        slots.push(SlotPlan {
            slot: feature.slot,
            access: match feature.role {
                LoweredFeatureRole::Input => AccessMode::Read as u8,
                LoweredFeatureRole::Output => AccessMode::Write as u8,
            },
            role: feature.role,
            io_index: feature.io_index as usize,
            element: element_code(feature.element),
            scalar_bytes: u64::from(feature.element.scalar_bytes()),
            byte_len: feature.byte_len,
            shape: OwnedShape::new(&feature.dims)?,
        });
    }
    debug_assert!(slots.is_sorted_by_key(|plan| plan.slot));
    Ok(slots)
}

/// Whether an enumerated device name satisfies a requested name or class prefix.
///
/// `"GPU"` matches `"GPU"` and indexed instances such as `"GPU.1"`, but never `"GPUX"`.
fn matches_device(available: &str, requested: &str) -> bool {
    match available.as_bytes().get(requested.len()) {
        None => available == requested,
        Some(b'.') => available.starts_with(requested),
        Some(_) => false,
    }
}

fn device_info_for(device: &str) -> DeviceInfo {
    let class = if device.starts_with("NPU") {
        AcceleratorClass::NPU
    } else if device.starts_with("GPU") {
        AcceleratorClass::GPU
    } else {
        AcceleratorClass::OTHER
    };
    let mut uuid = *b"intel-ov-\0\0\0\0\0\0\0";
    for (slot, byte) in uuid[9..].iter_mut().zip(device.bytes()) {
        *slot = byte.to_ascii_lowercase();
    }
    DeviceInfo {
        identity: DeviceIdentity {
            uuid,
            class,
            vendor_id: 0x8086,
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
    }
}

/// OpenVINO context handle.
#[derive(Debug)]
pub struct OpenVinoContext {
    id: u64,
}

/// Page-aligned buffer directly wrapped by OpenVINO tensors.
#[derive(Debug)]
pub struct OpenVinoBuffer {
    immutable_input: bool,
    context_id: u64,
    desc: BufferDesc,
    allocation: Arc<AlignedAllocation>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl OpenVinoBuffer {
    /// Inspect host-visible bytes without a readback copy. A read guard excludes
    /// native writers for the entire callback, including reentrant submissions.
    /// The borrowed slice cannot escape the callback; panic releases the guard.
    ///
    /// ```compile_fail
    /// fn escape(buffer: &virtio_accel_openvino::OpenVinoBuffer) -> &[u8] {
    ///     buffer.with_bytes(|bytes| bytes).unwrap()
    /// }
    /// ```
    pub fn with_bytes<R>(&self, inspect: impl FnOnce(&[u8]) -> R) -> Result<R, BackendError> {
        if !self.desc.usage.contains(BufferUsage::TRANSFER_SOURCE) {
            return Err(BackendError::PermissionDenied);
        }
        let mut guard = EventBacking::new(self.allocation.clone(), AccessMode::Read);
        guard.acquire()?;
        // SAFETY: self owns the allocation, logical bytes fit its layout, and the
        // live shared guard prevents native/host writes until inspect returns.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.allocation.pointer.as_ptr(), self.desc.bytes() as usize)
        };
        Ok(inspect(bytes))
    }
}

/// Resident compiled OpenVINO model.
pub struct OpenVinoProgram {
    context_id: u64,
    resident: Rc<ProgramResident>,
    slots: Vec<SlotPlan>,
}

// Field order releases requests before compiled models and constant storage.
struct ProgramResident {
    idle: RefCell<Vec<CachedRequest>>,
    npu: Option<NpuContext>,
    compiled: CompiledModelHandle,
    /// The reader shares the weights memory; both stay resident for the program lifetime
    /// because plugin retention of shared constants is unspecified.
    _weights_tensor: Option<TensorHandle>,
    _weights: Vec<u8>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for OpenVinoProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenVinoProgram")
            .field("context_id", &self.context_id)
            .finish_non_exhaustive()
    }
}

// Only completed requests enter this bounded per-program cache. Their tensor
// wrappers retain allocations, but access guards are reacquired on every start.
const MAX_IDLE_REQUESTS: usize = 8;
// The C runtime captures the ADDRESS of ov_callback_t, not a copy. Both the
// ABI record and its argument therefore live in one request-owned stable Box.
struct CallbackState {
    notification: CompletionNotification,
    record: ffi::ov_callback_t,
}
impl CallbackState {
    fn new() -> Box<Self> {
        let mut state = Box::new(Self {
            notification: CompletionNotification::default(),
            record: ffi::ov_callback_t {
                callback_func: completion_callback,
                args: ptr::null_mut(),
            },
        });
        state.record.args = (&state.notification as *const CompletionNotification)
            .cast_mut()
            .cast();
        state
    }
}
unsafe extern "C" fn completion_callback(context: *mut c_void) {
    // SAFETY: the event/cache owns the stable Box until request destruction has
    // joined callbacks. Reset/reuse happens only after native terminal observation.
    let notification = unsafe { &*context.cast::<CompletionNotification>() };
    // A user-supplied waker must never unwind across the foreign ABI.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| notification.notify()));
}

struct CachedRequest {
    callback: Option<Box<CallbackState>>,
    request: NonNull<ffi::ov_infer_request_t>,
    tensors: Vec<BoundTensor>,
}
impl Drop for CachedRequest {
    fn drop(&mut self) {
        // SAFETY: this request is quiescent. Free before releasing bound storage.
        unsafe { (ffi::api().ov_infer_request_free)(self.request.as_ptr()) };
    }
}

/// OpenVINO execution queue handle with reusable per-submission scratch.
#[derive(Debug)]
pub struct OpenVinoQueue {
    context_id: u64,
    scratch: Rc<QueueScratch>,
}

/// Asynchronous OpenVINO inference event.
///
/// Owns the infer request, the bound tensors, and the buffer in-flight guards. The first
/// observed terminal state is latched; the guards are dropped strictly before the latch is
/// published.
pub struct OpenVinoEvent {
    callback: Option<Box<CallbackState>>,
    resident: Rc<ProgramResident>,
    request: NonNull<ffi::ov_infer_request_t>,
    metadata: RefCell<EventMetadata>,
    scratch: Weak<QueueScratch>,
    latched: Cell<Option<EventState>>,
    deadline: Option<Instant>,
    cancel_issued: Cell<bool>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for OpenVinoEvent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenVinoEvent")
            .field("latched", &self.latched.get())
            .finish_non_exhaustive()
    }
}

impl OpenVinoEvent {
    /// Publish the first observed terminal state; guard release precedes latch publication so a
    /// caller observing a terminal state can immediately transfer buffer bytes.
    fn latch(&self, state: EventState) -> EventState {
        self.metadata.borrow_mut().backings.clear();
        self.latched.set(Some(state));
        state
    }

    fn recycle_metadata(&self) {
        let mut metadata = self.metadata.borrow_mut();
        if metadata.tensors.capacity() == 0 && metadata.backings.capacity() == 0 {
            return;
        }
        metadata.tensors.clear();
        metadata.backings.clear();
        let returned = mem::take(&mut *metadata);
        drop(metadata);
        if let Some(scratch) = self.scratch.upgrade() {
            scratch.recycle_metadata(returned);
        }
    }

    /// Verify the runtime executed into the caller's own output allocations.
    fn outputs_are_directly_backed(&self) -> Result<bool, BackendError> {
        let metadata = self.metadata.borrow();
        for bound in &metadata.tensors {
            if bound.expected_output.is_null() {
                continue;
            }
            let mut tensor = ptr::null_mut();
            // SAFETY: the request is live; the runtime returns one owned tensor reference.
            check_status(unsafe {
                (ffi::api().ov_infer_request_get_output_tensor_by_index)(
                    self.request.as_ptr(),
                    bound.output_io_index,
                    &mut tensor,
                )
            })?;
            let tensor = NonNull::new(tensor).ok_or(BackendError::External {
                domain: OPENVINO_EXTERNAL_DOMAIN,
                code: 0,
            })?;
            let probe = TensorHandle { tensor };
            let mut data = ptr::null_mut();
            // SAFETY: the probe tensor is live and `data` is a valid out-pointer.
            check_status(unsafe { (ffi::api().ov_tensor_data)(probe.as_const_ptr(), &mut data) })?;
            if data != bound.expected_output {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Drop for OpenVinoEvent {
    fn drop(&mut self) {
        if self.latched.get() == Some(EventState::Complete)
            && self.resident.idle.borrow().len() < MAX_IDLE_REQUESTS
        {
            let tensors = mem::take(&mut self.metadata.borrow_mut().tensors);
            self.resident.idle.borrow_mut().push(CachedRequest {
                callback: self.callback.take(),
                request: self.request,
                tensors,
            });
            self.recycle_metadata();
            return;
        }
        if self.latched.get().is_none() {
            // The inference may still be running and writing through the bound tensors into
            // allocations this drop is about to release: cancel best-effort, then block until
            // the request is terminal before freeing anything.
            // SAFETY: the request is live; cancel and wait are ordinary request entry points and
            // their statuses are deliberately ignored (any terminal outcome suffices).
            unsafe {
                let _ = (ffi::api().ov_infer_request_cancel)(self.request.as_ptr());
                let _ = (ffi::api().ov_infer_request_wait)(self.request.as_ptr());
            }
        }
        // SAFETY: this handle owns the request reference and frees it exactly once; the bound
        // tensors are dropped only after this call, then the already-terminal backing guards are
        // released. This ordering also applies to a pre-start rejection.
        unsafe { (ffi::api().ov_infer_request_free)(self.request.as_ptr()) };
        self.recycle_metadata();
    }
}

/// Intel OpenVINO backend instance bound to one inference device.
pub struct OpenVinoAccelerator {
    core: Arc<CoreHandle>,
    device: CString,
    next_id: AtomicU64,
    direct_binding_admissions: AtomicU64,
    npu_page_imports: AtomicU64,
    native_requests_created: AtomicU64,
    native_requests_reused: AtomicU64,
    explicit_transfer_bytes: AtomicU64,
    info: DeviceInfo,
    /// Whether this device's compiler accepts FP8, probed once at open.
    fp8: bool,
    quantized_fp32: bool,
}

impl std::fmt::Debug for OpenVinoAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenVinoAccelerator")
            .field("device", &self.device_name())
            .finish_non_exhaustive()
    }
}

impl OpenVinoAccelerator {
    /// Import immutable host input into this context. The fixed descriptor cannot
    /// be used as an output or transfer destination. Events retain the backing;
    /// dropping every caller handle cannot free pages still in use by a device.
    pub fn import_shared_input(
        &self,
        context: &OpenVinoContext,
        input: &SharedInput,
    ) -> Result<AllocatedBuffer<OpenVinoBuffer>, BackendError> {
        let desc = BufferDesc::new(
            input.bytes as u64,
            input.allocation.alignment(),
            MemoryDomain::Shared,
            BufferUsage::PROGRAM_INPUT | BufferUsage::TRANSFER_SOURCE,
        )?;
        self.info.validate_buffer_desc(desc)?;
        let info = BufferInfo::new(
            desc,
            input.allocation.allocation_bytes(),
            input.allocation.alignment(),
            BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
        )?;
        Ok(AllocatedBuffer::new(
            OpenVinoBuffer {
                context_id: context.id,
                desc,
                immutable_input: true,
                allocation: input.allocation.clone(),
                _not_send_sync: PhantomData,
            },
            info,
        ))
    }
    /// Initialize runtime loading from an installed library (for example a Python
    /// OpenVINO wheel). The first successful load is retained for the process
    /// lifetime. Call before discovery; this never replaces an active runtime.
    #[cfg(feature = "runtime-loading")]
    pub fn load_runtime_from(path: &std::path::Path) -> Result<(), InitError> {
        ffi::load_from(path)
    }

    /// Enumerate concrete runtime endpoints without choosing a default device.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        shared_core()?.available_devices()
    }

    /// Open the preferred available inference device: NPU, then GPU, then CPU.
    pub fn new() -> Result<Self, InitError> {
        let core = shared_core()?;
        let devices = core.available_devices()?;
        let device = ["NPU", "GPU", "CPU"]
            .into_iter()
            .find_map(|preferred| {
                devices
                    .iter()
                    .find(|available| matches_device(available, preferred))
            })
            .cloned()
            .ok_or(InitError::DeviceUnavailable)?;
        Self::with_selected(core, device)
    }

    /// Open one specific device, by enumerated name (`"GPU.1"`) or class prefix (`"NPU"`).
    pub fn with_device(device: &str) -> Result<Self, InitError> {
        let core = shared_core()?;
        let devices = core.available_devices()?;
        // An enumerated endpoint must win over a class-prefix match.
        let resolved = devices
            .iter()
            .find(|available| available.as_str() == device)
            .or_else(|| {
                devices
                    .iter()
                    .find(|available| matches_device(available, device))
            })
            .cloned()
            .ok_or(InitError::DeviceUnavailable)?;
        Self::with_selected(core, resolved)
    }

    fn with_selected(core: Arc<CoreHandle>, device: String) -> Result<Self, InitError> {
        let info = device_info_for(&device);
        let device = CString::new(device).map_err(|_| InitError::DeviceUnavailable)?;
        let fp8 = device_accepts_fp8(&core, &device);
        let quantized_fp32 = device_accepts_quantized_fp32(&core, &device);
        Ok(Self {
            core,
            device,
            next_id: AtomicU64::new(0),
            direct_binding_admissions: AtomicU64::new(0),
            npu_page_imports: AtomicU64::new(0),
            native_requests_created: AtomicU64::new(0),
            native_requests_reused: AtomicU64::new(0),
            explicit_transfer_bytes: AtomicU64::new(0),
            info,
            fp8,
            quantized_fp32,
        })
    }

    /// Whether this instance advertises the FP8 tier, which depends on the device it opened.
    pub fn advertises_fp8(&self) -> bool {
        self.fp8
    }

    /// The enumerated name of the device this instance executes on.
    /// The runtime's build number (`ov_get_openvino_version`), the version a captured compiled
    /// blob must be bound to.
    pub fn runtime_version() -> Result<String, InitError> {
        ffi::ensure_loaded()?;
        let mut version = ffi::ov_version_t {
            build_number: ptr::null(),
            description: ptr::null(),
        };
        // SAFETY: `version` is a valid out-structure; on success the runtime owns both strings
        // until `ov_version_free`, and they are copied before that release.
        unsafe {
            if (ffi::api().ov_get_openvino_version)(&mut version) != ffi::OV_STATUS_OK
                || version.build_number.is_null()
            {
                return Err(InitError::RuntimeUnavailable);
            }
            let build = CStr::from_ptr(version.build_number)
                .to_string_lossy()
                .into_owned();
            (ffi::api().ov_version_free)(&mut version);
            Ok(build)
        }
    }

    /// Export the plugin-compiled form of `program` to `path` through
    /// `ov_compiled_model_export_model`: the NPU plugin writes its VPUX ELF, the GPU plugin its
    /// serialized graph with the OpenCL program binaries (Intel GT zebin ELFs) embedded. The
    /// bytes are the runtime's own export format for this device and OpenVINO version; a
    /// consumer that keeps them must bind them to the artifact and version that produced them.
    /// The program stays loaded and usable afterwards.
    pub fn export_program(
        &self,
        program: &OpenVinoProgram,
        path: &std::path::Path,
    ) -> Result<(), BackendError> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| BackendError::InvalidArgument)?;
        // SAFETY: the compiled model is live for the program's lifetime and `path` is a valid
        // NUL-terminated string for the duration of the call.
        unsafe {
            check_status((ffi::api().ov_compiled_model_export_model)(
                program.resident.compiled.as_const_ptr(),
                path.as_ptr(),
            ))
        }
    }

    pub fn device_name(&self) -> &str {
        self.device.to_str().unwrap_or_default()
    }

    /// Re-enumerate inference devices through this instance's runtime core.
    pub fn runtime_devices(&self) -> Result<Vec<String>, InitError> {
        self.core.available_devices()
    }

    /// Immutable page imports bound to NPU infer requests, excluding cached replays.
    pub fn npu_page_imports(&self) -> u64 {
        self.npu_page_imports.load(Ordering::Relaxed)
    }

    /// Cumulative native infer requests created, including rejected submissions.
    pub fn native_requests_created(&self) -> u64 {
        self.native_requests_created.load(Ordering::Relaxed)
    }
    /// Successfully completed native requests reused on subsequent submissions.
    pub fn native_requests_reused(&self) -> u64 {
        self.native_requests_reused.load(Ordering::Relaxed)
    }

    /// Cumulative count of buffers admitted as direct bindings.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.load(Ordering::Relaxed)
    }

    /// Cumulative bytes moved by explicit `write_buffer`/`read_buffer` transfers.
    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.explicit_transfer_bytes.load(Ordering::Relaxed)
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(BackendError::ResourceLimit);
        }
        Ok(id)
    }

    fn checked_range(
        buffer: &OpenVinoBuffer,
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

    /// Owning-worker completion observation. After a callback hint, explicitly
    /// join its retirement to obtain the C API's authoritative success/error.
    /// OpenVINO publishes that result AFTER its callback returns. This may block
    /// for the callback tail; never call it from submit, poll_event or a callback.
    /// Without a hint it performs the ordinary nonblocking readiness probe.
    pub fn observe_notified_event(
        &self,
        event: &OpenVinoEvent,
    ) -> Result<EventState, BackendError> {
        if let Some(state) = event.latched.get() {
            return Ok(state);
        }
        if event
            .callback
            .as_ref()
            .expect("event callback")
            .notification
            .is_notified()
        {
            // SAFETY: the event retains a live request and all storage; its
            // callback only notifies and does not depend on the owning worker.
            let status = unsafe { (ffi::api().ov_infer_request_wait)(event.request.as_ptr()) };
            return Ok(self.latch_request_status(event, status));
        }
        Ok(self.poll_request(event))
    }

    fn lowering_error(error: crate::LoweringError) -> BackendError {
        match error {
            crate::LoweringError::Parse(_)
            | crate::LoweringError::Analysis(_)
            | crate::LoweringError::InvalidConstant => BackendError::InvalidArgument,
            crate::LoweringError::UnsupportedGraph
            | crate::LoweringError::UnsupportedType(_)
            | crate::LoweringError::UnsupportedOperator(_) => BackendError::Unsupported,
            crate::LoweringError::ResourceLimit => BackendError::ResourceLimit,
        }
    }

    /// Probe a live request without blocking and latch any terminal state.
    fn poll_request(&self, event: &OpenVinoEvent) -> EventState {
        if let Some(state) = event.latched.get() {
            return state;
        }
        // SAFETY: the request is live; a zero timeout is a non-blocking readiness probe.
        let mut status =
            unsafe { (ffi::api().ov_infer_request_wait_for)(event.request.as_ptr(), 0) };
        if status == ffi::OV_STATUS_RESULT_NOT_READY {
            let expired = event
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline);
            if !expired || event.cancel_issued.get() {
                return EventState::Pending;
            }
            event.cancel_issued.set(true);
            // SAFETY: cancel on a live request; the follow-up probe observes the outcome.
            unsafe {
                let _ = (ffi::api().ov_infer_request_cancel)(event.request.as_ptr());
                status = (ffi::api().ov_infer_request_wait_for)(event.request.as_ptr(), 0);
            }
            if status == ffi::OV_STATUS_RESULT_NOT_READY {
                return EventState::Pending;
            }
        }
        self.latch_request_status(event, status)
    }

    fn latch_request_status(&self, event: &OpenVinoEvent, status: ffi::ov_status_e) -> EventState {
        let state = match status {
            ffi::OV_STATUS_OK => match event.outputs_are_directly_backed() {
                Ok(true) => EventState::Complete,
                // A provider-side output reallocation is an execution failure, never a hidden
                // copy back into the caller's binding.
                Ok(false) => EventState::Failed(BackendError::Incompatible),
                Err(error) => EventState::Failed(error),
            },
            ffi::OV_STATUS_INFER_CANCELLED if event.cancel_issued.get() => {
                EventState::Failed(BackendError::DeadlineExpired)
            }
            other => EventState::Failed(backend_error_from_status(other)),
        };
        event.latch(state)
    }
}

impl TosaCapabilityProvider for OpenVinoAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        match (self.fp8, self.quantized_fp32) {
            (true, true) => crate::TOSA_CAPABILITIES_WITH_FP8,
            (false, true) => crate::TOSA_CAPABILITIES,
            (true, false) => crate::TOSA_CAPABILITIES_FP16_WITH_FP8,
            (false, false) => crate::TOSA_CAPABILITIES_FP16,
        }
    }
}

impl Accelerator for OpenVinoAccelerator {
    type Context = OpenVinoContext;
    type Buffer = OpenVinoBuffer;
    type Program = OpenVinoProgram;
    type Queue = OpenVinoQueue;
    type Event = OpenVinoEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        Ok(OpenVinoContext {
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
            OpenVinoBuffer {
                context_id: context.id,
                desc,
                immutable_input: false,
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
            // SAFETY: the validated range lies within the owned allocation, `buffer` is
            // exclusively borrowed, and the source is a distinct borrowed byte region.
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
            // inference still mutates this buffer.
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
        if artifact.format != virtio_accel_tosa::ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        let target = virtio_accel_tosa::Target::from_identity(artifact.target)
            .map_err(|_| BackendError::Incompatible)?;
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
        // GPU's compressed kernel preserves the explicit FP16 dequantization.
        // CPU avoids per-inference expansion; NPU needs a materialized rounded
        // constant to prevent scale fusion from changing FP16 intermediates.
        let expand_weights = self.device.to_bytes().starts_with(b"CPU")
            || self.device.to_bytes().starts_with(b"NPU");
        let lowered = lower_tosa_with_weights(
            bytes,
            target,
            expand_weights,
            self.device.to_bytes().starts_with(b"GPU"),
        )
        .map_err(Self::lowering_error)?;
        let require_fp32 = (target == crate::OPENVINO_TOSA_QUANTIZED_TARGET
            || target == crate::OPENVINO_TOSA_FP8_TARGET)
            && lowered.requires_fp32;
        if require_fp32 && !self.quantized_fp32 {
            return Err(BackendError::Unsupported);
        }
        let slots = build_slot_plan(&lowered.features)?;
        let inputs = lowered
            .features
            .iter()
            .filter(|feature| feature.role == LoweredFeatureRole::Input)
            .count();
        let outputs = lowered.features.len() - inputs;

        let weights_tensor = if lowered.weights.is_empty() {
            None
        } else {
            let shape = OwnedShape::new(&[lowered.weights.len() as i64])?;
            Some(TensorHandle::from_host_ptr(
                ffi::ELEMENT_U8,
                shape.as_value(),
                lowered.weights.as_ptr().cast_mut().cast(),
            )?)
        };

        let model = read_model(&self.core, &lowered.xml, weights_tensor.as_ref())?;
        let compiled = compile_with_precision(&self.core, &model, &self.device, require_fp32)?;
        drop(model);

        // Emission order defined the model I/O indices; verify the compiled counts agree before
        // trusting any `*_by_index` call.
        let (mut model_inputs, mut model_outputs) = (0usize, 0usize);
        // SAFETY: the compiled model is live and the out-pointers are valid.
        unsafe {
            check_status((ffi::api().ov_compiled_model_inputs_size)(
                compiled.as_const_ptr(),
                &mut model_inputs,
            ))?;
            check_status((ffi::api().ov_compiled_model_outputs_size)(
                compiled.as_const_ptr(),
                &mut model_outputs,
            ))?;
        }
        if (model_inputs, model_outputs) != (inputs, outputs) {
            return Err(BackendError::Incompatible);
        }

        Ok(OpenVinoProgram {
            context_id: context.id,
            slots,
            resident: Rc::new(ProgramResident {
                idle: RefCell::new(Vec::with_capacity(MAX_IDLE_REQUESTS)),
                npu: NpuContext::for_compiled(&compiled, self.device_name()),
                compiled,
                _weights_tensor: weights_tensor,
                _weights: lowered.weights,
                _not_send_sync: PhantomData,
            }),
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
        Ok(OpenVinoQueue {
            context_id: context.id,
            scratch: Rc::new(QueueScratch::default()),
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
        if bindings.is_empty()
            || bindings.len() > self.info.limits.max_bindings_per_submission as usize
        {
            return Err(SubmitFailure::Rejected(BackendError::ResourceLimit));
        }
        if queue.context_id != program.context_id {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }
        let mut pointers = queue
            .scratch
            .take_pointer_slots(program.slots.len())
            .map_err(SubmitFailure::Rejected)?;
        let mut event_metadata = queue
            .scratch
            .take_metadata(bindings.len(), program.slots.len())
            .map_err(SubmitFailure::Rejected)?;
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
            let plan = &program.slots[index];
            if binding.access as u8 != plan.access {
                return Err(SubmitFailure::Rejected(BackendError::Incompatible));
            }
            // The tensor wraps the range directly: it must cover the exact tensor bytes and be
            // scalar-aligned (the base allocation is page-aligned, so the offset decides).
            if binding.range.bytes() != plan.byte_len || start as u64 % plan.scalar_bytes != 0 {
                return Err(SubmitFailure::Rejected(BackendError::Incompatible));
            }
            pointers[index] = binding.buffer.allocation.pointer_at(start).cast();
            event_metadata.backings.push(EventBacking::new(
                Arc::clone(&binding.buffer.allocation),
                binding.access,
            ));
        }
        if bindings.len() != program.slots.len() {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }

        prepare_event_backings(&mut event_metadata.backings).map_err(SubmitFailure::Rejected)?;

        // Native phase. Failures below free every created handle and reject: nothing observable
        // was started until `start_async` succeeds, so `SubmitFailure::Indeterminate` is never
        // produced.
        let (request, mut callback) = if let Some(mut cached) =
            program.resident.idle.borrow_mut().pop().filter(|cached| {
                // The driver requires imported pages to outlive every request that used them.
                cached
                    .tensors
                    .iter()
                    .zip(&*pointers)
                    .all(|(tensor, data)| !tensor.imported || tensor.data == *data)
            }) {
            event_metadata.tensors = mem::take(&mut cached.tensors);
            let request = cached.request;
            let callback = cached.callback.take().expect("cached callback");
            // Transfer request ownership into the event without freeing it.
            mem::forget(cached);
            self.native_requests_reused.fetch_add(1, Ordering::Relaxed);
            (request, callback)
        } else {
            let mut request = ptr::null_mut();
            // SAFETY: compiled model is live and request is a valid out-pointer.
            check_status(unsafe {
                (ffi::api().ov_compiled_model_create_infer_request)(
                    program.resident.compiled.as_const_ptr(),
                    &mut request,
                )
            })
            .map_err(SubmitFailure::Rejected)?;
            let request =
                NonNull::new(request).ok_or(SubmitFailure::Rejected(BackendError::External {
                    domain: OPENVINO_EXTERNAL_DOMAIN,
                    code: 0,
                }))?;
            let callback = CallbackState::new();
            // SAFETY: both request and callback record are stable and remain owned
            // until after native request destruction. Nothing has started yet.
            if let Err(error) = check_status(unsafe {
                (ffi::api().ov_infer_request_set_callback)(request.as_ptr(), &callback.record)
            }) {
                // SAFETY: no inference was started; free before the callback Box.
                unsafe { (ffi::api().ov_infer_request_free)(request.as_ptr()) };
                return Err(SubmitFailure::Rejected(error));
            }
            self.native_requests_created.fetch_add(1, Ordering::Relaxed);
            (request, callback)
        };
        callback.notification.reset();
        let event = OpenVinoEvent {
            callback: Some(callback),
            resident: program.resident.clone(),
            request,
            metadata: RefCell::new(event_metadata),
            scratch: Rc::downgrade(&queue.scratch),
            // A pre-latched event never re-enters the runtime: on rejection below, `Drop` frees
            // the never-started request and tensors immediately.
            latched: Cell::new(Some(EventState::Failed(BackendError::InvalidArgument))),
            deadline: match timeout {
                Timeout::Infinite => None,
                Timeout::AfterNs(nanos) => {
                    Instant::now().checked_add(Duration::from_nanos(nanos.get()))
                }
            },
            cancel_issued: Cell::new(false),
            _not_send_sync: PhantomData,
        };

        let mut metadata = event.metadata.borrow_mut();
        let old_len = metadata.tensors.len();
        let same_bindings = old_len == pointers.len()
            && metadata
                .tensors
                .iter()
                .zip(pointers.iter())
                .all(|(t, p)| t.data == *p);
        if !same_bindings {
            // Reserve before binding so each successful native bind is immediately guarded.
            metadata
                .tensors
                .try_reserve(program.slots.len())
                .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
            for (index, (plan, data)) in program.slots.iter().zip(pointers.iter()).enumerate() {
                if metadata.tensors.get(index).is_some_and(|t| t.data == *data) {
                    continue;
                }
                let buffer = bindings
                    .iter()
                    .find(|b| b.slot == plan.slot)
                    .expect("validated slot")
                    .buffer;
                let imported = program
                    .resident
                    .npu
                    .as_ref()
                    .filter(|_| buffer.immutable_input)
                    .and_then(|context| context.import(plan, *data));
                let mut is_imported = imported.is_some();
                let mut tensor = match imported {
                    Some(tensor) => tensor,
                    None => TensorHandle::from_host_ptr(plan.element, plan.shape.as_value(), *data)
                        .map_err(SubmitFailure::Rejected)?,
                };
                // SAFETY: request and tensor are live; the runtime retains its own tensor reference.
                let bind = |tensor: &TensorHandle| unsafe {
                    match plan.role {
                        LoweredFeatureRole::Input => (ffi::api()
                            .ov_infer_request_set_input_tensor_by_index)(
                            event.request.as_ptr(),
                            plan.io_index,
                            tensor.as_const_ptr(),
                        ),
                        LoweredFeatureRole::Output => (ffi::api()
                            .ov_infer_request_set_output_tensor_by_index)(
                            event.request.as_ptr(),
                            plan.io_index,
                            tensor.as_const_ptr(),
                        ),
                    }
                };
                let mut status = bind(&tensor);
                if is_imported && status != ffi::OV_STATUS_OK {
                    tensor =
                        TensorHandle::from_host_ptr(plan.element, plan.shape.as_value(), *data)
                            .map_err(SubmitFailure::Rejected)?;
                    status = bind(&tensor);
                    is_imported = false;
                }
                check_status(status).map_err(SubmitFailure::Rejected)?;
                if is_imported {
                    self.npu_page_imports.fetch_add(1, Ordering::Relaxed);
                }
                let bound = BoundTensor {
                    imported: is_imported,
                    _tensor: tensor,
                    _allocation: buffer.allocation.clone(),
                    data: *data,
                    output_io_index: plan.io_index,
                    expected_output: if plan.role == LoweredFeatureRole::Output {
                        *data
                    } else {
                        ptr::null_mut()
                    },
                };
                if index < old_len {
                    // Keep the displaced ordinary binding alive through partial replacement.
                    let previous = mem::replace(&mut metadata.tensors[index], bound);
                    metadata.tensors.push(previous);
                } else {
                    metadata.tensors.push(bound);
                }
            }
            metadata.tensors.truncate(program.slots.len());
        }
        drop(metadata);

        // SAFETY: the request is live with every boundary tensor bound.
        let status = unsafe { (ffi::api().ov_infer_request_start_async)(event.request.as_ptr()) };
        if status != ffi::OV_STATUS_OK {
            // Defense in depth: even though a failed start is specified not to have started,
            // join the request before the event drop releases the guards and tensors.
            // SAFETY: cancel and wait on a live request; statuses are deliberately ignored.
            unsafe {
                let _ = (ffi::api().ov_infer_request_cancel)(event.request.as_ptr());
                let _ = (ffi::api().ov_infer_request_wait)(event.request.as_ptr());
            }
            return Err(SubmitFailure::Rejected(backend_error_from_status(status)));
        }
        event.latched.set(None);
        self.direct_binding_admissions
            .fetch_add(bindings.len() as u64, Ordering::Relaxed);
        Ok(event)
    }

    fn register_event_waker(
        &self,
        event: &Self::Event,
        waker: &core::task::Waker,
    ) -> Result<bool, BackendError> {
        // Deadline cancellation still requires the caller's timer/poll path.
        if event.deadline.is_some() {
            return Ok(false);
        }
        event
            .callback
            .as_ref()
            .expect("event callback")
            .notification
            .register(waker);
        Ok(true)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        Ok(self.poll_request(event))
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        match self.poll_request(&event) {
            EventState::Pending => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            }),
            EventState::Complete | EventState::Failed(_) | EventState::Cancelled => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_input_is_immutable_aligned_and_shared_across_threads() {
        let input = SharedInput::initialize(17, 16384, |bytes| {
            bytes.copy_from_slice(&[7; 17]);
            Ok(())
        })
        .unwrap();
        assert_eq!(input.as_bytes().as_ptr() as usize % 16384, 0);
        let clone = input.clone();
        assert!(Arc::ptr_eq(&input.allocation, &clone.allocation));
        drop(input);
        std::thread::spawn(move || assert_eq!(clone.as_bytes(), &[7; 17]))
            .join()
            .unwrap();
        for (bytes, alignment) in [(0, 16), (4, 0), (4, 3), (u64::MAX, 16)] {
            assert!(SharedInput::initialize(bytes, alignment, |_| Ok(())).is_err());
        }
        assert!(SharedInput::initialize(4, 16, |_| Err(BackendError::InvalidArgument)).is_err());
    }

    #[test]
    fn borrowed_host_view_excludes_writers_and_releases_on_panic() {
        let allocation = Arc::new(AlignedAllocation::new(16, 16).unwrap());
        let buffer = OpenVinoBuffer {
            immutable_input: false,
            context_id: 0,
            desc: BufferDesc::new(
                16,
                16,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
            allocation: allocation.clone(),
            _not_send_sync: PhantomData,
        };
        let mut writer = EventBacking::new(allocation.clone(), AccessMode::Write);
        writer.acquire().unwrap();
        assert_eq!(buffer.with_bytes(|_| ()), Err(BackendError::Busy));
        drop(writer);
        buffer
            .with_bytes(|bytes| {
                assert_eq!(bytes, &[0; 16]);
                let mut writer = EventBacking::new(allocation.clone(), AccessMode::Write);
                assert_eq!(writer.acquire(), Err(BackendError::Busy));
            })
            .unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = buffer.with_bytes(|_| panic!("inspection failed"));
            }))
            .is_err()
        );
        let mut writer = EventBacking::new(allocation, AccessMode::Write);
        writer.acquire().unwrap();
    }

    #[test]
    fn imported_shared_input_is_read_only_and_retains_identical_backing() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let input = SharedInput::initialize(16, 16384, |bytes| {
            bytes.fill(3);
            Ok(())
        })
        .unwrap();
        let (mut first, _) = backend
            .import_shared_input(&context, &input)
            .unwrap()
            .into_parts();
        let context2 = backend.create_context(ContextDesc::default()).unwrap();
        let (second, _) = backend
            .import_shared_input(&context2, &input)
            .unwrap()
            .into_parts();
        assert!(Arc::ptr_eq(&first.allocation, &second.allocation));
        assert!(Arc::ptr_eq(&first.allocation, &input.allocation));
        assert!(!first.desc.allows_access(AccessMode::Write));
        assert!(!first.desc.allows_access(AccessMode::ReadWrite));
        assert_eq!(
            backend.write_buffer(&mut first, 0, &SliceSource(&[0; 16])),
            Err(BackendError::PermissionDenied)
        );
        drop(input);
        drop(first);
        assert_eq!(
            second.with_bytes(|bytes| bytes.iter().sum::<u8>()).unwrap(),
            48
        );
        assert_eq!(backend.explicit_transfer_bytes(), 0);
    }
    use virtio_accel_conformance::numerics::{
        IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3, IDENTITY_FP8E5M2, MATMUL_FP16, MATMUL_FP32,
        MAX_POOL2D_FP32,
    };

    const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    fn backend() -> Option<OpenVinoAccelerator> {
        match OpenVinoAccelerator::new() {
            Ok(backend) => Some(backend),
            Err(InitError::DeviceUnavailable | InitError::RuntimeUnavailable)
                if std::env::var("VIRTIO_ACCEL_OPENVINO_REQUIRE_HARDWARE").as_deref()
                    != Ok("1")
                    && std::env::var("VIRTIO_ACCEL_OPENVINO").as_deref() != Ok("1") =>
            {
                None
            }
            Err(error) => panic!("backend initialization failed: {error}"),
        }
    }

    #[derive(Debug)]
    struct SliceSource<'a>(&'a [u8]);

    impl ByteSource for SliceSource<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
            ByteSource::read_at(self.0, offset, target)
        }

        fn as_contiguous(&self) -> Option<&[u8]> {
            Some(self.0)
        }
    }

    #[derive(Debug)]
    struct SliceSink<'a>(&'a mut [u8]);

    impl ByteSink for SliceSink<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
            ByteSink::write_at(self.0, offset, source)
        }

        fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
            Some(self.0)
        }
    }

    fn tosa_artifact<'a>(payload: &'a SliceSource<'a>) -> ArtifactRef<'a> {
        tosa_artifact_for(payload, crate::OPENVINO_TOSA_TARGET)
    }

    fn tosa_artifact_for<'a>(
        payload: &'a SliceSource<'a>,
        target: virtio_accel_tosa::Target,
    ) -> ArtifactRef<'a> {
        ArtifactRef {
            format: virtio_accel_tosa::ARTIFACT_FORMAT,
            target: target.to_identity(),
            payload,
            resident_bytes: REQUIRED_RESIDENT_BYTES,
        }
    }

    struct CompletionWake {
        ready: std::sync::atomic::AtomicBool,
        thread: std::thread::Thread,
    }
    impl std::task::Wake for CompletionWake {
        fn wake(self: std::sync::Arc<Self>) {
            self.ready.store(true, std::sync::atomic::Ordering::Release);
            self.thread.unpark();
        }
    }
    fn wait_for_terminal(backend: &OpenVinoAccelerator, event: &OpenVinoEvent) -> EventState {
        let observer = std::sync::Arc::new(CompletionWake {
            ready: std::sync::atomic::AtomicBool::new(false),
            thread: std::thread::current(),
        });
        let waker = core::task::Waker::from(observer.clone());
        let notified = backend.register_event_waker(event, &waker).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        if notified {
            while !observer.ready.load(std::sync::atomic::Ordering::Acquire) {
                let left = deadline
                    .checked_duration_since(Instant::now())
                    .expect("completion notification lost");
                std::thread::park_timeout(left);
            }
            let terminal = backend.observe_notified_event(event).unwrap();
            assert_ne!(terminal, EventState::Pending);
            observer
                .ready
                .store(false, std::sync::atomic::Ordering::Release);
            assert!(backend.register_event_waker(event, &waker).unwrap());
            assert!(
                observer.ready.load(std::sync::atomic::Ordering::Acquire),
                "late registration lost"
            );
            return terminal;
        }
        loop {
            match backend.poll_event(event).unwrap() {
                EventState::Pending => {
                    assert!(Instant::now() < deadline, "inference never completed");
                    std::thread::sleep(Duration::from_millis(1));
                }
                terminal => return terminal,
            }
        }
    }

    #[test]
    fn device_matching_requires_exact_names_or_indexed_instances() {
        assert!(matches_device("NPU", "NPU"));
        assert!(matches_device("GPU.1", "GPU"));
        assert!(matches_device("GPU.1", "GPU.1"));
        assert!(!matches_device("GPUX", "GPU"));
        assert!(!matches_device("GPU", "GPU.1"));
        assert!(!matches_device("CPU", "GPU"));
    }

    #[test]
    fn selects_a_device_and_reports_valid_stable_metadata() {
        let Some(backend) = backend() else { return };
        backend.device_info().unwrap().validate().unwrap();
        assert_eq!(backend.device_info(), backend.device_info());
        let name = backend.device_name().to_owned();
        assert!(!name.is_empty());
        assert!(backend.runtime_devices().unwrap().contains(&name));
    }

    #[test]
    fn explicit_selection_resolves_every_enumerated_device() {
        let Some(backend) = backend() else { return };
        for device in backend.runtime_devices().unwrap() {
            let explicit = OpenVinoAccelerator::with_device(&device).unwrap();
            assert_eq!(explicit.device_name(), device);
            explicit.device_info().unwrap().validate().unwrap();
        }
        assert_eq!(
            OpenVinoAccelerator::with_device("no-such-device").unwrap_err(),
            InitError::DeviceUnavailable
        );
    }

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
    fn queue_scratch_reuses_only_cleared_metadata_capacity() {
        let scratch = QueueScratch::default();
        let allocation = Arc::new(AlignedAllocation::new(32, 16).unwrap());

        let mut metadata = scratch.take_metadata(2, 2).unwrap();
        let backing_storage = metadata.backings.as_ptr();
        let tensor_storage = metadata.tensors.as_ptr();
        metadata
            .backings
            .push(EventBacking::new(Arc::clone(&allocation), AccessMode::Read));
        assert_eq!(Arc::strong_count(&allocation), 2);
        scratch.recycle_metadata(metadata);
        assert_eq!(Arc::strong_count(&allocation), 1);

        let reused = scratch.take_metadata(2, 2).unwrap();
        assert_eq!(reused.backings.as_ptr(), backing_storage);
        assert_eq!(reused.tensors.as_ptr(), tensor_storage);
        assert!(reused.backings.is_empty());
        assert!(reused.tensors.is_empty());
    }

    #[test]
    fn pointer_scratch_is_scrubbed_on_every_exit() {
        let scratch = QueueScratch::default();
        {
            let mut slots = scratch.take_pointer_slots(2).unwrap();
            slots[0] = NonNull::<u8>::dangling().as_ptr().cast();
            slots[1] = NonNull::<u16>::dangling().as_ptr().cast();
        }
        assert!(scratch.pointer_slots.borrow().is_empty());
    }

    /// Pins the runtime facts the lowering design depends on: in-memory IR v11 acceptance with a
    /// null weights tensor, a direct parameter-to-result edge, the variadic property convention
    /// of `ov_core_compile_model`, and index-stable I/O sizes — via the production load path.
    #[test]
    fn runtime_accepts_lowered_corpus_documents() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        for (name, artifact) in [
            ("identity-fp32", IDENTITY_EDGES_FP32.artifact),
            ("matmul-fp32", MATMUL_FP32.artifact),
            ("max-pool2d-fp32", MAX_POOL2D_FP32.artifact),
            ("matmul-fp16", MATMUL_FP16.artifact),
        ] {
            let program = backend
                .load_program(&context, tosa_artifact(&SliceSource(artifact)))
                .unwrap_or_else(|error| panic!("{name}: load failed: {error:?}"));
            backend.unload_program(program).unwrap();
        }
        backend.destroy_context(context).unwrap();
    }

    /// The milestone gate: a device-neutral TOSA identity executes end-to-end on the selected
    /// device with direct bindings only.
    #[test]
    fn fp8_tier_graphs_compile_on_the_device() {
        let Some(backend) = backend() else { return };
        eprintln!(
            "FP8 tier device: {} (FP8 advertised: {})",
            backend.device_name(),
            backend.advertises_fp8()
        );
        if !backend.advertises_fp8() {
            return;
        }
        let context = backend.create_context(ContextDesc::default()).unwrap();
        for dtype in [
            virtio_accel_tosa::DType::FP8E4M3,
            virtio_accel_tosa::DType::FP8E5M2,
        ] {
            for (name, artifact) in [
                ("matmul", crate::lower::fp8_matmul_graph(dtype)),
                ("max_pool2d", crate::lower::fp8_pool_graph(dtype)),
                ("transpose", crate::lower::fp8_transpose_graph(dtype)),
            ] {
                let program = backend
                    .load_program(
                        &context,
                        tosa_artifact_for(&SliceSource(&artifact), crate::OPENVINO_TOSA_FP8_TARGET),
                    )
                    .unwrap_or_else(|error| {
                        panic!("{dtype:?} {name}: device compilation failed: {error:?}")
                    });
                backend.unload_program(program).unwrap();
            }
        }
        backend.destroy_context(context).unwrap();
    }

    /// Advertisement must match what the device will actually do, in both directions. A device
    /// that advertises FP8 loads an FP8 graph; a device that does not must refuse it at load
    /// rather than accept the program and fail later. Intel NPU arch 5010 takes the first branch
    /// and arch 40XX the second, so this is the one FP8 test that is meaningful on every device.
    #[test]
    fn fp8_advertisement_matches_what_the_device_accepts() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let artifact = crate::lower::fp8_matmul_graph(virtio_accel_tosa::DType::FP8E4M3);
        let outcome = backend.load_program(
            &context,
            tosa_artifact_for(&SliceSource(&artifact), crate::OPENVINO_TOSA_FP8_TARGET),
        );
        match (backend.advertises_fp8(), outcome) {
            (true, Ok(program)) => backend.unload_program(program).unwrap(),
            (true, Err(error)) => panic!(
                "{}: advertises FP8 but refused an FP8 graph: {error:?}",
                backend.device_name()
            ),
            (false, Err(_)) => {}
            (false, Ok(_)) => panic!(
                "{}: does not advertise FP8 but accepted an FP8 graph",
                backend.device_name()
            ),
        }
        backend.destroy_context(context).unwrap();
    }

    /// The lowering tests assert the emitted document's shape; this asserts the real compiler
    /// builds a blob from it. The distinction is the whole point of the widening: a raw FP8
    /// MatMul reaches MLIR's verifier and is rejected, because the NPU compiler's IE dialect
    /// declares MatMul operands without the FP8 types.
    #[test]
    fn fp8_identity_moves_every_byte_pattern_on_the_device() {
        let Some(backend) = backend() else { return };
        // Name the device: FP8 movement works on several plugins, so a pass says nothing about
        // which one ran it. The evidence pin needs the device, not just the result.
        eprintln!(
            "FP8 device: {} (FP8 advertised: {})",
            backend.device_name(),
            backend.advertises_fp8()
        );
        if !backend.advertises_fp8() {
            return;
        }
        for case in [IDENTITY_FP8E4M3, IDENTITY_FP8E5M2] {
            let context = backend.create_context(ContextDesc::default()).unwrap();
            let program = backend
                .load_program(
                    &context,
                    tosa_artifact_for(&SliceSource(case.artifact), crate::OPENVINO_TOSA_FP8_TARGET),
                )
                .unwrap_or_else(|error| panic!("{}: load rejected: {error:?}", case.name));
            let bytes = program.slots[0].byte_len;
            let desc = BufferDesc::new(
                bytes,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE
                    | BufferUsage::TRANSFER_DESTINATION
                    | BufferUsage::PROGRAM_INPUT
                    | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap();
            let (mut input, _) = backend
                .allocate_buffer(&context, desc)
                .unwrap()
                .into_parts();
            let (output, _) = backend
                .allocate_buffer(&context, desc)
                .unwrap()
                .into_parts();
            // Walk the encoding space: one byte per element, so NaNs, both zeros, the
            // subnormals and the maxima are all covered rather than sampled.
            let payload = (0..bytes as usize)
                .map(|index| (index % 256) as u8)
                .collect::<Vec<_>>();
            backend
                .write_buffer(&mut input, 0, &SliceSource(&payload))
                .unwrap();
            let queue = backend
                .create_queue(&context, QueueDesc::default())
                .unwrap();
            let bindings = [
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: virtio_accel_core::BufferRange::new(0, bytes).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: virtio_accel_core::BufferRange::new(0, bytes).unwrap(),
                    access: AccessMode::Write,
                },
            ];
            let event = backend
                .submit(&queue, &program, &bindings, Timeout::Infinite)
                .unwrap_or_else(|_| panic!("{}: submission failed", case.name));
            assert_eq!(
                wait_for_terminal(&backend, &event),
                EventState::Complete,
                "{}",
                case.name
            );
            let mut result = vec![0u8; bytes as usize];
            backend
                .read_buffer(&output, 0, &mut SliceSink(&mut result))
                .unwrap();
            assert_eq!(
                result, payload,
                "{}: FP8 movement is not bit-exact",
                case.name
            );
            backend.destroy_event(event).unwrap();
        }
    }

    #[test]
    fn executes_device_neutral_tosa_identity_end_to_end() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = backend
            .load_program(&context, tosa_artifact(&SliceSource(IDENTITY_FP32_LOCAL)))
            .unwrap();
        let bytes = program.slots[0].byte_len;
        let desc = BufferDesc::new(
            bytes,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT
                | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let (mut input, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (output, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let payload = (1..=bytes as u32 / 4)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        backend
            .write_buffer(&mut input, 0, &SliceSource(&payload))
            .unwrap();

        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: virtio_accel_core::BufferRange::new(0, bytes).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: virtio_accel_core::BufferRange::new(0, bytes).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|failure| match failure {
                SubmitFailure::Rejected(error) => panic!("submission rejected: {error:?}"),
                SubmitFailure::Indeterminate { error, .. } => {
                    panic!("submission indeterminate: {error:?}")
                }
            });
        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
        assert!(event.metadata.borrow().backings.is_empty());
        assert!(queue.scratch.metadata_spare.borrow().is_none());

        let mut result = vec![0u8; bytes as usize];
        backend
            .read_buffer(&output, 0, &mut SliceSink(&mut result))
            .unwrap();
        assert_eq!(result, payload);
        assert_eq!(backend.direct_binding_admissions(), 2);

        backend.destroy_event(event).unwrap();
        assert_eq!(program.resident.idle.borrow().len(), 1);
        assert!(
            queue
                .scratch
                .metadata_spare
                .borrow()
                .as_ref()
                .is_some_and(|m| m.backings.is_empty() && m.tensors.is_empty())
        );
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
        backend.destroy_event(event).unwrap();
        assert_eq!(backend.native_requests_created(), 1);
        assert_eq!(backend.native_requests_reused(), 1);
        // Even identical artifacts have independent pools and compiled owners.
        let other = backend
            .load_program(&context, tosa_artifact(&SliceSource(IDENTITY_FP32_LOCAL)))
            .unwrap();
        let separate = backend
            .submit(&queue, &other, &bindings, Timeout::Infinite)
            .unwrap();
        assert_eq!(wait_for_terminal(&backend, &separate), EventState::Complete);
        backend.destroy_event(separate).unwrap();
        assert_eq!(backend.native_requests_created(), 2);
        assert_eq!(backend.native_requests_reused(), 1);
        backend.unload_program(other).unwrap();
        // Rebind an idle request to different allocations, then drop every public
        // handle including the program before waiting. The event retains all data.
        let (mut replacement, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let doubled: Vec<_> = (1..=bytes as u32 / 4)
            .flat_map(|v| (2.0 * v as f32).to_le_bytes())
            .collect();
        backend
            .write_buffer(&mut replacement, 0, &SliceSource(&doubled))
            .unwrap();
        let rebound = [
            BindingRef {
                buffer: &replacement,
                ..bindings[0]
            },
            BindingRef { ..bindings[1] },
        ];
        let event = backend
            .submit(&queue, &program, &rebound, Timeout::Infinite)
            .unwrap();
        backend.free_buffer(replacement).unwrap();
        backend.unload_program(program).unwrap();
        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
        backend
            .read_buffer(&output, 0, &mut SliceSink(&mut result))
            .unwrap();
        assert_eq!(result, doubled);
        backend.destroy_event(event).unwrap();
        assert_eq!(backend.native_requests_created(), 2);
        assert_eq!(backend.native_requests_reused(), 2);
        backend.destroy_queue(queue).unwrap();
        backend.free_buffer(input).unwrap();
        backend.free_buffer(output).unwrap();
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn centered_normalization_handles_zero_variance_and_nearly_constant_rows() {
        let Some(discovery) = backend() else { return };
        let artifact = crate::lower::language_model_tests::normalization_graph(0.25, 1e-5, false);
        for device in discovery.runtime_devices().unwrap() {
            let backend = OpenVinoAccelerator::with_device(&device).unwrap();
            let context = backend.create_context(ContextDesc::default()).unwrap();
            let queue = backend
                .create_queue(&context, QueueDesc::default())
                .unwrap();
            let program = backend
                .load_program(&context, tosa_artifact(&SliceSource(&artifact)))
                .unwrap();
            let mut buffers: Vec<_> = program
                .slots
                .iter()
                .map(|slot| {
                    backend
                        .allocate_buffer(
                            &context,
                            BufferDesc::new(
                                slot.byte_len,
                                4096,
                                MemoryDomain::Shared,
                                BufferUsage::TRANSFER_SOURCE
                                    | BufferUsage::TRANSFER_DESTINATION
                                    | BufferUsage::PROGRAM_INPUT
                                    | BufferUsage::PROGRAM_OUTPUT,
                            )
                            .unwrap(),
                        )
                        .unwrap()
                        .into_parts()
                        .0
                })
                .collect();
            let mut cases = vec![
                [-4.0f32, -1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                [16384.0; 8],
                [-100.0, 100.0, -100.0, 100.0, -2.0, -1.0, 1.0, 2.0],
            ];
            if !device.starts_with("NPU") {
                cases.push([1.0, 1.0001, 1.0002, 1.0003, -1.0, -1.0001, -1.0002, -1.0003]);
            }
            for input in cases {
                let raw: Vec<_> = input.iter().flat_map(|x| x.to_ne_bytes()).collect();
                backend
                    .write_buffer(&mut buffers[0], 0, &SliceSource(&raw))
                    .unwrap();
                let bindings: Vec<_> = buffers
                    .iter()
                    .zip(&program.slots)
                    .enumerate()
                    .map(|(i, (buffer, slot))| BindingRef {
                        slot: i as u32,
                        buffer,
                        range: virtio_accel_core::BufferRange::new(0, slot.byte_len).unwrap(),
                        access: if i == 0 {
                            AccessMode::Read
                        } else {
                            AccessMode::Write
                        },
                    })
                    .collect();
                let event = backend
                    .submit(&queue, &program, &bindings, Timeout::Infinite)
                    .unwrap();
                assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
                buffers[1]
                    .with_bytes(|bytes| {
                        for (i, raw) in bytes.chunks_exact(4).enumerate() {
                            let row = &input[i / 4 * 4..i / 4 * 4 + 4];
                            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 4.0;
                            let variance = row
                                .iter()
                                .map(|&x| (f64::from(x) - mean).powi(2))
                                .sum::<f64>()
                                / 4.0;
                            let expected = (f64::from(input[i]) - mean) / (variance + 1e-5).sqrt();
                            let actual = f32::from_ne_bytes(raw.try_into().unwrap());
                            let tolerance = if device.starts_with("NPU") {
                                0.03
                            } else {
                                0.005
                            };
                            assert!(
                                actual.is_finite()
                                    && (f64::from(actual) - expected).abs() < tolerance,
                                "{device} input={input:?} element={i}: {actual} != {expected}"
                            );
                        }
                    })
                    .unwrap();
                backend.destroy_event(event).unwrap();
            }
            backend.unload_program(program).unwrap();
            for buffer in buffers {
                backend.free_buffer(buffer).unwrap();
            }
            backend.destroy_queue(queue).unwrap();
            backend.destroy_context(context).unwrap();
        }
    }

    #[test]
    fn external_residual_is_read_on_every_reused_request() {
        let Some(discovery) = backend() else { return };
        for device in discovery.runtime_devices().unwrap() {
            let backend = OpenVinoAccelerator::with_device(&device).unwrap();
            let context = backend.create_context(ContextDesc::default()).unwrap();
            let queue = backend
                .create_queue(&context, QueueDesc::default())
                .unwrap();
            for (rows, columns) in [(1, 4), (2, 7), (16, 4)] {
                for swapped in [false, true] {
                    let artifact =
                        crate::lower::residual_tests::residual_graph(rows, columns, swapped);
                    let program = backend
                        .load_program(&context, tosa_artifact(&SliceSource(&artifact)))
                        .unwrap();
                    let mut buffers: Vec<_> = program
                        .slots
                        .iter()
                        .map(|slot| {
                            let desc = BufferDesc::new(
                                slot.byte_len,
                                4096,
                                MemoryDomain::Shared,
                                BufferUsage::TRANSFER_SOURCE
                                    | BufferUsage::TRANSFER_DESTINATION
                                    | BufferUsage::PROGRAM_INPUT
                                    | BufferUsage::PROGRAM_OUTPUT,
                            )
                            .unwrap();
                            backend
                                .allocate_buffer(&context, desc)
                                .unwrap()
                                .into_parts()
                                .0
                        })
                        .collect();
                    let created = backend.native_requests_created();
                    let reused = backend.native_requests_reused();
                    for iteration in 0..8 {
                        let input: Vec<f32> = (0..rows * 3)
                            .map(|i| (i - iteration) as f32 / 16.0)
                            .collect();
                        let residual: Vec<f32> = (0..rows * columns)
                            .map(|i| (i + iteration * 3 - 5) as f32 / 32.0)
                            .collect();
                        for (buffer, values) in buffers.iter_mut().zip([&input, &residual]) {
                            let bytes: Vec<_> =
                                values.iter().flat_map(|x| x.to_ne_bytes()).collect();
                            backend
                                .write_buffer(buffer, 0, &SliceSource(&bytes))
                                .unwrap();
                        }
                        let bindings: Vec<_> = buffers
                            .iter()
                            .zip(&program.slots)
                            .enumerate()
                            .map(|(i, (buffer, slot))| BindingRef {
                                slot: i as u32,
                                buffer,
                                range: virtio_accel_core::BufferRange::new(0, slot.byte_len)
                                    .unwrap(),
                                access: if i == 2 {
                                    AccessMode::Write
                                } else {
                                    AccessMode::Read
                                },
                            })
                            .collect();
                        let event = backend
                            .submit(&queue, &program, &bindings, Timeout::Infinite)
                            .unwrap();
                        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
                        buffers[2].with_bytes(|bytes| {
                            for (i, b) in bytes.chunks_exact(4).enumerate() {
                                let row = i / columns as usize;
                                let col = i % columns as usize;
                                let dot: f32 = (0..3).map(|k| input[row*3+k] * (((k*columns as usize+col)%7) as f32-3.0)/8.0).sum();
                                let expected = (dot + residual[i])*0.5;
                                let actual = f32::from_ne_bytes(b.try_into().unwrap());
                                assert!((actual-expected).abs() < 0.005,
                                    "{device} rows={rows} columns={columns} swapped={swapped} iteration={iteration} element={i}: {actual} != {expected}");
                            }
                        }).unwrap();
                        backend.destroy_event(event).unwrap();
                    }
                    assert_eq!(backend.native_requests_created() - created, 1);
                    assert_eq!(backend.native_requests_reused() - reused, 7);
                    backend.unload_program(program).unwrap();
                    for buffer in buffers {
                        backend.free_buffer(buffer).unwrap();
                    }
                }
            }
            backend.destroy_queue(queue).unwrap();
            backend.destroy_context(context).unwrap();
        }
    }

    #[test]
    fn host_transfers_are_busy_while_an_inference_holds_the_backing() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let desc = BufferDesc::new(
            32,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap();
        let (mut buffer, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let mut guard = EventBacking::new(Arc::clone(&buffer.allocation), AccessMode::Write);
        guard.acquire().unwrap();
        assert_eq!(
            backend.write_buffer(&mut buffer, 0, &SliceSource(&[0u8; 32])),
            Err(BackendError::Busy)
        );
        let mut sink = [0u8; 32];
        assert_eq!(
            backend.read_buffer(&buffer, 0, &mut SliceSink(&mut sink)),
            Err(BackendError::Busy)
        );
        drop(guard);
        backend
            .write_buffer(&mut buffer, 0, &SliceSource(&[1u8; 32]))
            .unwrap();
        backend.free_buffer(buffer).unwrap();
        backend.destroy_context(context).unwrap();
    }

    #[test]
    fn misaligned_and_wrong_length_bindings_are_rejected_before_admission() {
        let Some(backend) = backend() else { return };
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = backend
            .load_program(&context, tosa_artifact(&SliceSource(IDENTITY_FP32_LOCAL)))
            .unwrap();
        let bytes = program.slots[0].byte_len;
        let desc = BufferDesc::new(
            bytes * 2,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT
                | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let (input, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (output, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();

        let submit = |input_range: (u64, u64)| {
            let bindings = [
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: virtio_accel_core::BufferRange::new(input_range.0, input_range.1)
                        .unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: virtio_accel_core::BufferRange::new(0, bytes).unwrap(),
                    access: AccessMode::Write,
                },
            ];
            match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
                Err(SubmitFailure::Rejected(error)) => error,
                Err(SubmitFailure::Indeterminate { error, .. }) => {
                    panic!("indeterminate: {error:?}")
                }
                Ok(_) => panic!("invalid binding was admitted"),
            }
        };
        // One byte into the buffer: scalar-misaligned for f32.
        assert_eq!(submit((1, bytes)), BackendError::Incompatible);
        // Wrong length for the slot's tensor.
        assert_eq!(submit((0, bytes * 2)), BackendError::Incompatible);

        backend.destroy_queue(queue).unwrap();
        backend.unload_program(program).unwrap();
        backend.free_buffer(input).unwrap();
        backend.free_buffer(output).unwrap();
        backend.destroy_context(context).unwrap();
    }
}

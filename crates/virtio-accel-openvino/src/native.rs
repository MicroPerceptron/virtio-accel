//! Native OpenVINO runtime integration: the audited `Accelerator` implementation.
//!
//! Every `unsafe` block carries a `SAFETY:` comment referencing the invariants audited in
//! `SAFETY.md`. Each OpenVINO handle has exactly one Rust owner with a `Drop` implementation;
//! the shared `ov_core_t` is process-wide. Completion is poll-latched: no foreign callback ever
//! owns Rust memory.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};

use crate::lower::{LoweredFeature, LoweredFeatureRole, OvElement, lower_tosa};
use crate::{InitError, REQUIRED_RESIDENT_BYTES, ffi};

/// Maximal TOSA artifact bytes admitted before parsing.
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// Minimum allocation alignment: one page, satisfying every plugin's zero-copy import rules.
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
    static CORE: OnceLock<Result<Arc<CoreHandle>, InitError>> = OnceLock::new();
    CORE.get_or_init(|| CoreHandle::create().map(Arc::new))
        .clone()
}

impl CoreHandle {
    fn create() -> Result<Self, InitError> {
        let mut core = ptr::null_mut();
        // SAFETY: `core` is a valid out-pointer; the runtime initializes it only on OK.
        let status = unsafe { ffi::ov_core_create(&mut core) };
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
        let status =
            unsafe { ffi::ov_core_get_available_devices(self.as_const_ptr(), &mut devices) };
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
            ffi::ov_available_devices_free(&mut devices);
        }
        Ok(names)
    }
}

impl Drop for CoreHandle {
    fn drop(&mut self) {
        // SAFETY: the handle owns exactly one core reference and is dropped exactly once.
        unsafe { ffi::ov_core_free(self.core.as_ptr()) }
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

// SAFETY: allocation access is available only through the `Accelerator` methods. The buffer
// handle is neither Send nor Sync, and its atomic in-flight gate excludes host transfers while a
// native inference owns a backing guard. Guards are dropped strictly before an event's terminal
// state becomes observable. See `SAFETY.md`.
unsafe impl Send for AlignedAllocation {}
// SAFETY: same invariant as the `Send` implementation.
unsafe impl Sync for AlignedAllocation {}

impl Drop for AlignedAllocation {
    fn drop(&mut self) {
        // SAFETY: `pointer` was returned by `alloc_zeroed(self.layout)` and has not been freed.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
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
            ffi::ov_shape_create(dims.len() as i64, dims.as_ptr(), &mut shape)
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
        let _ = unsafe { ffi::ov_shape_free(&mut self.shape) };
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
            ffi::ov_tensor_create_from_host_ptr(element, shape, data, &mut tensor)
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
        unsafe { ffi::ov_tensor_free(self.tensor.as_ptr()) }
    }
}

/// Owned `ov_model_t`; a short-lived reader artifact freed right after compilation.
struct ModelHandle {
    model: NonNull<ffi::ov_model_t>,
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        // SAFETY: this handle owns one model reference and is dropped exactly once.
        unsafe { ffi::ov_model_free(self.model.as_ptr()) }
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
        unsafe { ffi::ov_compiled_model_free(self.compiled.as_ptr()) }
    }
}

/// The only C-variadic call site: compile with the `ACCURACY` execution-mode hint so plugins may
/// not silently run a declared-FP32 model at reduced precision.
fn compile_with_accuracy(
    core: &CoreHandle,
    model: &ModelHandle,
    device: &CStr,
) -> Result<CompiledModelHandle, BackendError> {
    let mut compiled = ptr::null_mut();
    // SAFETY: core, model, and device are live; `property_args_size` counts the variadic
    // arguments (one key/value pair contributes two) per the documented convention pinned by a
    // unit test; the key symbol and value stay valid for the duration of the call.
    let status = unsafe {
        ffi::ov_core_compile_model(
            core.as_const_ptr(),
            model.model.as_ptr(),
            device.as_ptr(),
            2,
            &mut compiled,
            ffi::ov_property_key_hint_execution_mode,
            c"ACCURACY".as_ptr(),
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
    context_id: u64,
    desc: BufferDesc,
    allocation: Arc<AlignedAllocation>,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Resident compiled OpenVINO model.
pub struct OpenVinoProgram {
    context_id: u64,
    compiled: CompiledModelHandle,
    slots: Vec<SlotPlan>,
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

/// OpenVINO execution queue handle with reusable per-submission scratch.
#[derive(Debug)]
pub struct OpenVinoQueue {
    context_id: u64,
    scratch: RefCell<Vec<*mut c_void>>,
}

/// Asynchronous OpenVINO inference event.
///
/// Owns the infer request, the bound tensors, and the buffer in-flight guards. The first
/// observed terminal state is latched; the guards are dropped strictly before the latch is
/// published.
pub struct OpenVinoEvent {
    request: NonNull<ffi::ov_infer_request_t>,
    tensors: Vec<TensorHandle>,
    output_checks: Vec<(usize, *mut c_void)>,
    backings: RefCell<Vec<EventBacking>>,
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
        self.backings.borrow_mut().clear();
        self.latched.set(Some(state));
        state
    }

    /// Verify the runtime executed into the caller's own output allocations.
    fn outputs_are_directly_backed(&self) -> Result<bool, BackendError> {
        for (io_index, expected) in &self.output_checks {
            let mut tensor = ptr::null_mut();
            // SAFETY: the request is live; the runtime returns one owned tensor reference.
            check_status(unsafe {
                ffi::ov_infer_request_get_output_tensor_by_index(
                    self.request.as_ptr(),
                    *io_index,
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
            check_status(unsafe { ffi::ov_tensor_data(probe.as_const_ptr(), &mut data) })?;
            if data != *expected {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Drop for OpenVinoEvent {
    fn drop(&mut self) {
        if self.latched.get().is_none() {
            // The inference may still be running and writing through the bound tensors into
            // allocations this drop is about to release: cancel best-effort, then block until
            // the request is terminal before freeing anything.
            // SAFETY: the request is live; cancel and wait are ordinary request entry points and
            // their statuses are deliberately ignored (any terminal outcome suffices).
            unsafe {
                let _ = ffi::ov_infer_request_cancel(self.request.as_ptr());
                let _ = ffi::ov_infer_request_wait(self.request.as_ptr());
            }
        }
        // SAFETY: this handle owns the request reference and frees it exactly once; the bound
        // tensor handles and backing guards drop afterwards in field order.
        unsafe { ffi::ov_infer_request_free(self.request.as_ptr()) };
    }
}

/// Intel OpenVINO backend instance bound to one inference device.
pub struct OpenVinoAccelerator {
    core: Arc<CoreHandle>,
    device: CString,
    next_id: AtomicU64,
    direct_binding_admissions: AtomicU64,
    explicit_transfer_bytes: AtomicU64,
    info: DeviceInfo,
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
        let resolved = core
            .available_devices()?
            .iter()
            .find(|available| matches_device(available, device))
            .cloned()
            .ok_or(InitError::DeviceUnavailable)?;
        Self::with_selected(core, resolved)
    }

    fn with_selected(core: Arc<CoreHandle>, device: String) -> Result<Self, InitError> {
        let info = device_info_for(&device);
        let device = CString::new(device).map_err(|_| InitError::DeviceUnavailable)?;
        Ok(Self {
            core,
            device,
            next_id: AtomicU64::new(0),
            direct_binding_admissions: AtomicU64::new(0),
            explicit_transfer_bytes: AtomicU64::new(0),
            info,
        })
    }

    /// The enumerated name of the device this instance executes on.
    pub fn device_name(&self) -> &str {
        self.device.to_str().unwrap_or_default()
    }

    /// Re-enumerate inference devices through this instance's runtime core.
    pub fn runtime_devices(&self) -> Result<Vec<String>, InitError> {
        self.core.available_devices()
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
        let mut status = unsafe { ffi::ov_infer_request_wait_for(event.request.as_ptr(), 0) };
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
                let _ = ffi::ov_infer_request_cancel(event.request.as_ptr());
                status = ffi::ov_infer_request_wait_for(event.request.as_ptr(), 0);
            }
            if status == ffi::OV_STATUS_RESULT_NOT_READY {
                return EventState::Pending;
            }
        }
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
        let lowered = lower_tosa(bytes, target).map_err(Self::lowering_error)?;
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

        let mut model = ptr::null_mut();
        // SAFETY: core, document bytes, and the (possibly null) weights tensor are live for this
        // synchronous call; the created model shares the weights memory, which the returned
        // program keeps resident.
        let status = unsafe {
            ffi::ov_core_read_model_from_memory_buffer(
                self.core.as_const_ptr(),
                lowered.xml.as_ptr().cast(),
                lowered.xml.len(),
                weights_tensor
                    .as_ref()
                    .map_or(ptr::null(), TensorHandle::as_const_ptr),
                &mut model,
            )
        };
        check_status(status)?;
        let model = NonNull::new(model)
            .map(|model| ModelHandle { model })
            .ok_or(BackendError::External {
                domain: OPENVINO_EXTERNAL_DOMAIN,
                code: 0,
            })?;
        let compiled = compile_with_accuracy(&self.core, &model, &self.device)?;
        drop(model);

        // Emission order defined the model I/O indices; verify the compiled counts agree before
        // trusting any `*_by_index` call.
        let (mut model_inputs, mut model_outputs) = (0usize, 0usize);
        // SAFETY: the compiled model is live and the out-pointers are valid.
        unsafe {
            check_status(ffi::ov_compiled_model_inputs_size(
                compiled.as_const_ptr(),
                &mut model_inputs,
            ))?;
            check_status(ffi::ov_compiled_model_outputs_size(
                compiled.as_const_ptr(),
                &mut model_outputs,
            ))?;
        }
        if (model_inputs, model_outputs) != (inputs, outputs) {
            return Err(BackendError::Incompatible);
        }

        Ok(OpenVinoProgram {
            context_id: context.id,
            compiled,
            slots,
            _weights_tensor: weights_tensor,
            _weights: lowered.weights,
            _not_send_sync: PhantomData,
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
            scratch: RefCell::new(Vec::new()),
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
        let mut pointers = queue.scratch.borrow_mut();
        pointers.clear();
        pointers
            .try_reserve(program.slots.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        pointers.resize(program.slots.len(), ptr::null_mut());
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
            event_backings.push(EventBacking::new(
                Arc::clone(&binding.buffer.allocation),
                binding.access,
            ));
        }
        if bindings.len() != program.slots.len() {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }

        prepare_event_backings(&mut event_backings).map_err(SubmitFailure::Rejected)?;

        // Native phase. Failures below free every created handle and reject: nothing observable
        // was started until `start_async` succeeds, so `SubmitFailure::Indeterminate` is never
        // produced.
        let mut request = ptr::null_mut();
        // SAFETY: the compiled model is live and `request` is a valid out-pointer.
        let status = unsafe {
            ffi::ov_compiled_model_create_infer_request(
                program.compiled.as_const_ptr(),
                &mut request,
            )
        };
        check_status(status).map_err(SubmitFailure::Rejected)?;
        let Some(request) = NonNull::new(request) else {
            return Err(SubmitFailure::Rejected(BackendError::External {
                domain: OPENVINO_EXTERNAL_DOMAIN,
                code: 0,
            }));
        };
        let mut event = OpenVinoEvent {
            request,
            tensors: Vec::new(),
            output_checks: Vec::new(),
            backings: RefCell::new(event_backings),
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
        event
            .tensors
            .try_reserve_exact(program.slots.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;
        event
            .output_checks
            .try_reserve_exact(program.slots.len())
            .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfMemory))?;

        for (plan, data) in program.slots.iter().zip(pointers.iter()) {
            let tensor = TensorHandle::from_host_ptr(plan.element, plan.shape.as_value(), *data)
                .map_err(SubmitFailure::Rejected)?;
            // SAFETY: request and tensor are live; the runtime retains its own tensor reference.
            let status = unsafe {
                match plan.role {
                    LoweredFeatureRole::Input => ffi::ov_infer_request_set_input_tensor_by_index(
                        event.request.as_ptr(),
                        plan.io_index,
                        tensor.as_const_ptr(),
                    ),
                    LoweredFeatureRole::Output => ffi::ov_infer_request_set_output_tensor_by_index(
                        event.request.as_ptr(),
                        plan.io_index,
                        tensor.as_const_ptr(),
                    ),
                }
            };
            check_status(status).map_err(SubmitFailure::Rejected)?;
            if plan.role == LoweredFeatureRole::Output {
                event.output_checks.push((plan.io_index, *data));
            }
            event.tensors.push(tensor);
        }

        // SAFETY: the request is live with every boundary tensor bound.
        let status = unsafe { ffi::ov_infer_request_start_async(event.request.as_ptr()) };
        if status != ffi::OV_STATUS_OK {
            // Defense in depth: even though a failed start is specified not to have started,
            // join the request before the event drop releases the guards and tensors.
            // SAFETY: cancel and wait on a live request; statuses are deliberately ignored.
            unsafe {
                let _ = ffi::ov_infer_request_cancel(event.request.as_ptr());
                let _ = ffi::ov_infer_request_wait(event.request.as_ptr());
            }
            return Err(SubmitFailure::Rejected(backend_error_from_status(status)));
        }
        event.latched.set(None);
        self.direct_binding_admissions
            .fetch_add(bindings.len() as u64, Ordering::Relaxed);
        Ok(event)
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
    use virtio_accel_conformance::numerics::{
        IDENTITY_EDGES_FP32, MATMUL_FP16, MATMUL_FP32, MAX_POOL2D_FP32,
    };

    const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    fn backend() -> Option<OpenVinoAccelerator> {
        match OpenVinoAccelerator::new() {
            Ok(backend) => Some(backend),
            Err(InitError::DeviceUnavailable) => None,
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
        ArtifactRef {
            format: virtio_accel_tosa::ARTIFACT_FORMAT,
            target: crate::OPENVINO_TOSA_TARGET.to_identity(),
            payload,
            resident_bytes: REQUIRED_RESIDENT_BYTES,
        }
    }

    fn wait_for_terminal(backend: &OpenVinoAccelerator, event: &OpenVinoEvent) -> EventState {
        let deadline = Instant::now() + Duration::from_secs(15);
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

        let mut result = vec![0u8; bytes as usize];
        backend
            .read_buffer(&output, 0, &mut SliceSink(&mut result))
            .unwrap();
        assert_eq!(result, payload);
        assert_eq!(backend.direct_binding_admissions(), 2);

        backend.destroy_event(event).unwrap();
        backend.destroy_queue(queue).unwrap();
        backend.unload_program(program).unwrap();
        backend.free_buffer(input).unwrap();
        backend.free_buffer(output).unwrap();
        backend.destroy_context(context).unwrap();
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

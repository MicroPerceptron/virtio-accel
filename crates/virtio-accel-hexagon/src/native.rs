//! Native QNN HTP backend compiled against the detected QAIRT headers.

use crate::ffi;
use crate::lower::{FeatureRole, LoweredNode, lower_tosa};
use crate::{InitError, REQUIRED_RESIDENT_BYTES};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};

const QNN_EXTERNAL_DOMAIN: u32 = 0x0051_4e4e;
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
const MIN_ALIGNMENT: usize = 4096;
const TRANSFER_CHUNK_BYTES: usize = 64 * 1024;
const EXCLUSIVE_NATIVE_ACCESS: u64 = u64::MAX;
const MAX_SHARED_NATIVE_USERS: u64 = u64::MAX - 1;
const MESSAGE_BYTES: usize = 512;

fn status_error(status: u64) -> BackendError {
    match status {
        ffi::ERROR_INCOMPATIBLE => BackendError::Incompatible,
        ffi::ERROR_BUSY => BackendError::Busy,
        ffi::ERROR_INVALID_ARGUMENT => BackendError::InvalidArgument,
        ffi::ERROR_OUT_OF_MEMORY => BackendError::OutOfMemory,
        ffi::ERROR_INTERNAL => BackendError::DeviceLost,
        other => BackendError::External {
            domain: QNN_EXTERNAL_DOMAIN,
            code: other as i64,
        },
    }
}

fn check_status(status: u64) -> Result<(), BackendError> {
    if status == ffi::SUCCESS {
        Ok(())
    } else {
        Err(status_error(status))
    }
}

fn c_array(array: &[core::ffi::c_char]) -> String {
    // SAFETY: bridge-owned fixed arrays are always initialized and explicitly NUL-terminated.
    unsafe { CStr::from_ptr(array.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn qnn_library_path() -> Result<CString, InitError> {
    let root = Path::new(env!("VIRTIO_ACCEL_QNN_SDK_ROOT"));
    let path = root.join("lib/aarch64-windows-msvc/QnnHtp.dll");
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| InitError::RuntimeUnavailable)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QnnRuntimeInfo {
    pub provider_name: String,
    pub build_id: String,
    pub core_version: [u32; 3],
    pub backend_version: [u32; 3],
}

struct RuntimeHandle {
    raw: NonNull<ffi::Runtime>,
    info: QnnRuntimeInfo,
}

impl RuntimeHandle {
    fn new() -> Result<Self, InitError> {
        let library = qnn_library_path()?;
        let mut raw = ptr::null_mut();
        let mut native_info = ffi::RuntimeInfo::default();
        let mut message = [0; MESSAGE_BYTES];
        // SAFETY: every pointer is valid for this synchronous call. The bridge copies the DLL
        // path and initializes either a complete owned runtime or a null output.
        let status = unsafe {
            ffi::va_qnn_runtime_create(
                library.as_ptr(),
                &mut raw,
                &mut native_info,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        if status != ffi::SUCCESS {
            return Err(if status == ffi::ERROR_INCOMPATIBLE {
                InitError::IncompatibleRuntime
            } else {
                InitError::DeviceUnavailable
            });
        }
        let raw = NonNull::new(raw).ok_or(InitError::DeviceUnavailable)?;
        Ok(Self {
            raw,
            info: QnnRuntimeInfo {
                provider_name: c_array(&native_info.provider_name),
                build_id: c_array(&native_info.build_id),
                core_version: [
                    native_info.core_major,
                    native_info.core_minor,
                    native_info.core_patch,
                ],
                backend_version: [
                    native_info.backend_major,
                    native_info.backend_minor,
                    native_info.backend_patch,
                ],
            },
        })
    }
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        // SAFETY: this value owns the runtime and is the final Rc owner. Child graph/event values
        // retain their own Rc, so none can still reference the native runtime here.
        let status = unsafe { ffi::va_qnn_runtime_free(self.raw.as_ptr()) };
        debug_assert_eq!(status, ffi::SUCCESS);
    }
}

struct GraphHandle {
    raw: Cell<Option<NonNull<ffi::Graph>>>,
    _runtime: Rc<RuntimeHandle>,
}

impl GraphHandle {
    fn release(&self) -> Result<(), BackendError> {
        let Some(raw) = self.raw.get() else {
            return Ok(());
        };
        // SAFETY: raw is one live bridge graph owned by this value. A successful call consumes it.
        let status = unsafe { ffi::va_qnn_graph_free(raw.as_ptr()) };
        check_status(status)?;
        self.raw.set(None);
        Ok(())
    }
}

impl Drop for GraphHandle {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.get() {
            // SAFETY: best-effort cleanup of the graph owned by this value. A busy result leaks the
            // native graph rather than freeing descriptors that an execution may still use.
            let _ = unsafe { ffi::va_qnn_graph_free(raw.as_ptr()) };
        }
    }
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
        let alignment = requested_alignment.max(MIN_ALIGNMENT);
        let allocation_bytes = bytes
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or(BackendError::OutOfMemory)?;
        let layout = Layout::from_size_align(allocation_bytes, alignment)
            .map_err(|_| BackendError::OutOfMemory)?;
        // SAFETY: layout is valid and nonzero; Drop uses the identical layout exactly once.
        let pointer =
            NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(BackendError::OutOfMemory)?;
        Ok(Self {
            pointer,
            layout,
            in_flight: AtomicU64::new(0),
        })
    }

    fn pointer_at(&self, offset: usize) -> *mut u8 {
        // SAFETY: callers validate offset and range against the logical allocation first.
        unsafe { self.pointer.as_ptr().add(offset) }
    }
}

impl Drop for AlignedAllocation {
    fn drop(&mut self) {
        // SAFETY: pointer came from alloc_zeroed with this exact layout and has not been freed.
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

#[derive(Debug)]
struct EventBacking {
    key: usize,
    allocation: Rc<AlignedAllocation>,
    access: BackingAccess,
    acquired: bool,
}

impl EventBacking {
    fn new(allocation: Rc<AlignedAllocation>, access: AccessMode) -> Self {
        Self {
            key: Rc::as_ptr(&allocation) as usize,
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

#[derive(Clone, Copy, Debug)]
struct SlotPlan {
    slot: u32,
    role: FeatureRole,
    io_index: usize,
    byte_len: u64,
}

#[derive(Debug)]
pub struct HexagonContext {
    id: u64,
}

#[derive(Debug)]
pub struct HexagonBuffer {
    context_id: u64,
    desc: BufferDesc,
    allocation: Rc<AlignedAllocation>,
    _not_send_sync: PhantomData<Rc<()>>,
}

pub struct HexagonProgram {
    context_id: u64,
    graph: Rc<GraphHandle>,
    slots: Vec<SlotPlan>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for HexagonProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HexagonProgram")
            .field("context_id", &self.context_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct HexagonQueue {
    context_id: u64,
}

pub struct HexagonEvent {
    raw: Cell<Option<NonNull<ffi::Event>>>,
    backings: RefCell<Vec<EventBacking>>,
    latched: Cell<Option<EventState>>,
    _graph: Rc<GraphHandle>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for HexagonEvent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HexagonEvent")
            .field("latched", &self.latched.get())
            .finish_non_exhaustive()
    }
}

impl HexagonEvent {
    fn poll_native(&self) -> EventState {
        if let Some(state) = self.latched.get() {
            return state;
        }
        let Some(raw) = self.raw.get() else {
            return EventState::Failed(BackendError::DeviceLost);
        };
        let mut error = ffi::SUCCESS;
        // SAFETY: raw stays owned by this event until terminal release.
        let state = unsafe { ffi::va_qnn_event_poll(raw.as_ptr(), &mut error) };
        let terminal = match state {
            ffi::EVENT_PENDING => return EventState::Pending,
            ffi::EVENT_COMPLETE => EventState::Complete,
            ffi::EVENT_FAILED => EventState::Failed(status_error(error)),
            _ => EventState::Failed(BackendError::DeviceLost),
        };
        self.backings.borrow_mut().clear();
        self.latched.set(Some(terminal));
        terminal
    }

    fn release_native(&self) -> Result<(), BackendError> {
        let Some(raw) = self.raw.get() else {
            return Ok(());
        };
        // SAFETY: terminal-state checks happen in the bridge and success consumes the event.
        check_status(unsafe { ffi::va_qnn_event_free(raw.as_ptr()) })?;
        self.raw.set(None);
        Ok(())
    }
}

impl Drop for HexagonEvent {
    fn drop(&mut self) {
        while self.poll_native() == EventState::Pending {
            std::thread::sleep(Duration::from_millis(1));
        }
        let _ = self.release_native();
    }
}

pub struct HexagonAccelerator {
    runtime: Rc<RuntimeHandle>,
    next_id: AtomicU64,
    info: DeviceInfo,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for HexagonAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HexagonAccelerator")
            .field("runtime", &self.runtime.info)
            .finish_non_exhaustive()
    }
}

impl HexagonAccelerator {
    pub fn new() -> Result<Self, InitError> {
        let runtime = Rc::new(RuntimeHandle::new()?);
        Ok(Self {
            runtime,
            next_id: AtomicU64::new(0),
            info: DeviceInfo {
                identity: DeviceIdentity {
                    uuid: *b"qualcomm-htp-v73",
                    class: AcceleratorClass::NPU,
                    vendor_id: 0x17cb,
                    device_id: 73,
                },
                capabilities: Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY,
                limits: DeviceLimits {
                    max_contexts: 64,
                    max_buffers_per_context: 1_024,
                    max_programs_per_context: 64,
                    max_queues_per_context: 64,
                    max_events_per_context: 1,
                    max_bindings_per_submission: 256,
                    max_buffer_bytes: u64::from(u32::MAX),
                    max_artifact_bytes: MAX_TOSA_ARTIFACT_BYTES,
                },
            },
            _not_send_sync: PhantomData,
        })
    }

    pub fn available_devices() -> Result<Vec<String>, InitError> {
        let runtime = RuntimeHandle::new()?;
        Ok(vec![format!(
            "QNN HTP {} (core {}.{}.{}, backend {}.{}.{})",
            runtime.info.provider_name,
            runtime.info.core_version[0],
            runtime.info.core_version[1],
            runtime.info.core_version[2],
            runtime.info.backend_version[0],
            runtime.info.backend_version[1],
            runtime.info.backend_version[2]
        )])
    }

    pub fn runtime_info(&self) -> &QnnRuntimeInfo {
        &self.runtime.info
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map(|value| value + 1)
            .map_err(|_| BackendError::ResourceLimit)
    }

    fn checked_range(
        buffer: &HexagonBuffer,
        offset: u64,
        bytes: u64,
    ) -> Result<(usize, usize), BackendError> {
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= buffer.desc.bytes())
            .ok_or(BackendError::OutOfBounds)?;
        Ok((
            usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?,
            usize::try_from(end).map_err(|_| BackendError::OutOfBounds)?,
        ))
    }

    fn lowering_error(error: crate::LoweringError) -> BackendError {
        match error {
            crate::LoweringError::Parse(_) | crate::LoweringError::Analysis(_) => {
                BackendError::InvalidArgument
            }
            crate::LoweringError::ResourceLimit => BackendError::ResourceLimit,
            crate::LoweringError::UnsupportedGraph
            | crate::LoweringError::UnsupportedType(_)
            | crate::LoweringError::UnsupportedOperator(_)
            | crate::LoweringError::InvalidConstant => BackendError::Unsupported,
        }
    }
}

impl Accelerator for HexagonAccelerator {
    type Context = HexagonContext;
    type Buffer = HexagonBuffer;
    type Program = HexagonProgram;
    type Queue = HexagonQueue;
    type Event = HexagonEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        Ok(HexagonContext {
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
        if desc.domain == MemoryDomain::Device {
            return Err(BackendError::Unsupported);
        }
        let allocation = Rc::new(AlignedAllocation::new(desc.bytes(), desc.alignment())?);
        let properties = BufferProperties::HOST_VISIBLE
            | if desc.is_program_visible() || desc.domain == MemoryDomain::Shared {
                BufferProperties::DIRECT_BINDING
            } else {
                BufferProperties::empty()
            };
        let info = BufferInfo::new(
            desc,
            allocation.layout.size() as u64,
            allocation.layout.align() as u64,
            properties,
        )?;
        Ok(AllocatedBuffer::new(
            HexagonBuffer {
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
        let length = end - start;
        if let Some(source) = data.as_contiguous() {
            if source.len() != length {
                return Err(BackendError::InvalidArgument);
            }
            // SAFETY: the exclusive buffer borrow and in-flight gate protect the validated range.
            unsafe {
                ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    buffer.allocation.pointer_at(start),
                    length,
                )
            };
            return Ok(());
        }
        let mut scratch = [0; TRANSFER_CHUNK_BYTES];
        let mut copied = 0;
        while copied < length {
            let chunk = (length - copied).min(scratch.as_slice().len());
            data.read_at(copied as u64, &mut scratch[..chunk])?;
            // SAFETY: each chunk remains inside the validated exclusive range.
            unsafe {
                ptr::copy_nonoverlapping(
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
        let length = end - start;
        if let Some(target) = data.as_contiguous_mut() {
            if target.len() != length {
                return Err(BackendError::InvalidArgument);
            }
            // SAFETY: the in-flight gate proves no QNN execution can mutate this range.
            unsafe {
                ptr::copy_nonoverlapping(
                    buffer.allocation.pointer_at(start),
                    target.as_mut_ptr(),
                    length,
                )
            };
            return Ok(());
        }
        let mut scratch = [0; TRANSFER_CHUNK_BYTES];
        let mut copied = 0;
        while copied < length {
            let chunk = (length - copied).min(scratch.as_slice().len());
            // SAFETY: each chunk remains inside the validated source range.
            unsafe {
                ptr::copy_nonoverlapping(
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

        let slots = lowered
            .features
            .iter()
            .map(|feature| SlotPlan {
                slot: feature.slot,
                role: feature.role,
                io_index: feature.io_index as usize,
                byte_len: feature.byte_len,
            })
            .collect::<Vec<_>>();
        let tensor_descriptions = lowered
            .tensors
            .iter()
            .map(|tensor| {
                let (role, io_index) = lowered.boundary(tensor.value).map_or(
                    (ffi::TENSOR_NATIVE, ffi::NO_IO_INDEX),
                    |(feature_role, io_index)| {
                        (
                            match feature_role {
                                FeatureRole::Input => ffi::TENSOR_INPUT,
                                FeatureRole::Output => ffi::TENSOR_OUTPUT,
                            },
                            io_index,
                        )
                    },
                );
                ffi::TensorDesc {
                    value: tensor.value,
                    role,
                    io_index,
                    dimensions: tensor.dims.as_ptr(),
                    rank: tensor.dims.len() as u32,
                }
            })
            .collect::<Vec<_>>();
        let node_descriptions = lowered
            .nodes
            .iter()
            .map(|node| match node {
                LoweredNode::Identity { input, output } => ffi::NodeDesc {
                    kind: ffi::NODE_RESHAPE,
                    input0: *input,
                    output: *output,
                    ..ffi::NodeDesc::default()
                },
                LoweredNode::MatMul {
                    left,
                    right,
                    output,
                } => ffi::NodeDesc {
                    kind: ffi::NODE_MATMUL,
                    input0: *left,
                    input1: *right,
                    output: *output,
                    ..ffi::NodeDesc::default()
                },
                LoweredNode::MaxPool2d {
                    input,
                    output,
                    kernel,
                    stride,
                } => ffi::NodeDesc {
                    kind: ffi::NODE_MAX_POOL_2D,
                    input0: *input,
                    output: *output,
                    kernel: *kernel,
                    stride: *stride,
                    ..ffi::NodeDesc::default()
                },
            })
            .collect::<Vec<_>>();
        let mut graph = ptr::null_mut();
        let mut message = [0; MESSAGE_BYTES];
        // SAFETY: descriptor slices and their dimension storage remain live for the synchronous
        // call; the bridge copies and retains its own descriptors before returning.
        check_status(unsafe {
            ffi::va_qnn_graph_create(
                self.runtime.raw.as_ptr(),
                tensor_descriptions.as_ptr(),
                tensor_descriptions.len() as u32,
                node_descriptions.as_ptr(),
                node_descriptions.len() as u32,
                &mut graph,
                message.as_mut_ptr(),
                message.len(),
            )
        })?;
        let graph = NonNull::new(graph).ok_or(BackendError::DeviceLost)?;
        Ok(HexagonProgram {
            context_id: context.id,
            graph: Rc::new(GraphHandle {
                raw: Cell::new(Some(graph)),
                _runtime: Rc::clone(&self.runtime),
            }),
            slots,
            _not_send_sync: PhantomData,
        })
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        if Rc::strong_count(&program.graph) != 1 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: program,
            });
        }
        match program.graph.release() {
            Ok(()) => Ok(()),
            Err(BackendError::Busy) => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: program,
            }),
            Err(error) => Err(ReleaseFailure::Indeterminate { error }),
        }
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.info.validate_queue_desc(desc)?;
        Ok(HexagonQueue {
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
        if !matches!(timeout, Timeout::Infinite) {
            return Err(SubmitFailure::Rejected(BackendError::Unsupported));
        }
        if queue.context_id != program.context_id {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        }
        if bindings.len() != program.slots.len()
            || bindings.len() > self.info.limits.max_bindings_per_submission as usize
        {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }
        let mut pointers = vec![ptr::null_mut(); program.slots.len()];
        let mut backings = Vec::with_capacity(bindings.len());
        let mut seen = vec![false; program.slots.len()];
        for binding in bindings {
            if binding.buffer.context_id != queue.context_id {
                return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
            }
            if !binding.buffer.desc.allows_access(binding.access) {
                return Err(SubmitFailure::Rejected(BackendError::PermissionDenied));
            }
            let index = program
                .slots
                .binary_search_by_key(&binding.slot, |slot| slot.slot)
                .map_err(|_| SubmitFailure::Rejected(BackendError::Incompatible))?;
            if seen[index] {
                return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
            }
            seen[index] = true;
            let plan = program.slots[index];
            let expected_access = match plan.role {
                FeatureRole::Input => AccessMode::Read,
                FeatureRole::Output => AccessMode::Write,
            };
            if binding.access != expected_access || binding.range.bytes() != plan.byte_len {
                return Err(SubmitFailure::Rejected(BackendError::Incompatible));
            }
            let (start, _) =
                Self::checked_range(binding.buffer, binding.range.offset, binding.range.bytes())
                    .map_err(SubmitFailure::Rejected)?;
            pointers[index] = binding.buffer.allocation.pointer_at(start).cast();
            backings.push(EventBacking::new(
                Rc::clone(&binding.buffer.allocation),
                binding.access,
            ));
        }
        prepare_event_backings(&mut backings).map_err(SubmitFailure::Rejected)?;

        let input_count = program
            .slots
            .iter()
            .filter(|slot| slot.role == FeatureRole::Input)
            .count();
        let output_count = program.slots.len() - input_count;
        let mut inputs = vec![
            ffi::Binding {
                data: ptr::null_mut(),
                size: 0,
            };
            input_count
        ];
        let mut outputs = vec![
            ffi::Binding {
                data: ptr::null_mut(),
                size: 0,
            };
            output_count
        ];
        for (index, plan) in program.slots.iter().enumerate() {
            let binding = ffi::Binding {
                data: pointers[index],
                size: plan.byte_len,
            };
            match plan.role {
                FeatureRole::Input => inputs[plan.io_index] = binding,
                FeatureRole::Output => outputs[plan.io_index] = binding,
            }
        }
        let Some(graph) = program.graph.raw.get() else {
            return Err(SubmitFailure::Rejected(BackendError::InvalidArgument));
        };
        let mut event = ptr::null_mut();
        let mut message = [0; MESSAGE_BYTES];
        // SAFETY: the graph is live; bindings point into guarded allocations retained by the Rust
        // event. The bridge copies descriptors and owns them until its worker reaches completion.
        let status = unsafe {
            ffi::va_qnn_graph_execute_async(
                graph.as_ptr(),
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_ptr(),
                outputs.len() as u32,
                &mut event,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check_status(status).map_err(SubmitFailure::Rejected)?;
        let event =
            NonNull::new(event).ok_or_else(|| SubmitFailure::Rejected(BackendError::DeviceLost))?;
        Ok(HexagonEvent {
            raw: Cell::new(Some(event)),
            backings: RefCell::new(backings),
            latched: Cell::new(None),
            _graph: Rc::clone(&program.graph),
            _not_send_sync: PhantomData,
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        Ok(event.poll_native())
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        if event.poll_native() == EventState::Pending {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            });
        }
        match event.release_native() {
            Ok(()) => Ok(()),
            Err(BackendError::Busy) => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            }),
            Err(error) => Err(ReleaseFailure::Indeterminate { error }),
        }
    }
}

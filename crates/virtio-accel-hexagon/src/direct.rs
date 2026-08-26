//! Provider-local V73 FastRPC runtime.
//!
//! This is intentionally not a TOSA capability provider. Its artifact target
//! names QFloat32 semantics explicitly and therefore cannot be selected by a
//! conformant FP32/TOSA scheduler by accident.

use crate::DirectInitError;
use core::ffi::{c_char, c_void};
use std::cell::Cell;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactFormat, ArtifactRef,
    BackendError, BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink,
    ByteSource, Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState,
    MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, TargetIdentity, Timeout,
};

const EXTERNAL_DOMAIN: u32 = 0x4854_5037;
const DEFAULT_ARENA_BYTES: u32 = 512 * 1024 * 1024;
const MIN_ALIGNMENT: u64 = 128;
const MAX_ARTIFACT_BYTES: u64 = 64 * 1024;
const ARTIFACT_MAGIC: u32 = u32::from_le_bytes(*b"VAHD");
const ARTIFACT_ABI: u32 = 2;
const TARGET_MAGIC: u32 = u32::from_le_bytes(*b"V73Q");
const TARGET_ABI: u32 = 1;
const MESSAGE_BYTES: usize = 512;

pub const DIRECT_HTP_ARTIFACT_FORMAT: ArtifactFormat =
    match ArtifactFormat::new(u32::from_le_bytes(*b"VQ32")) {
        Some(value) => value,
        None => panic!("nonzero direct HTP artifact format"),
    };

/// V73 QFloat32 target. Word 2 is the architecture, word 3 is the arithmetic ABI.
pub const DIRECT_HTP_V73_TARGET: TargetIdentity = TargetIdentity([
    TARGET_MAGIC,
    TARGET_ABI,
    73,
    1, // QFloat32 vector result conversion semantics.
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DirectHtpOperation {
    Identity = 1,
    Add = 2,
    Multiply = 3,
    Reciprocal = 4,
    Rsqrt = 5,
    MatMul = 6,
    WormholeTrace = 16,
    KerrTrace = 17,
    KerrFrame = 18,
    KerrShade = 19,
}

impl DirectHtpOperation {
    const fn binding_count(self) -> usize {
        match self {
            Self::Identity
            | Self::Reciprocal
            | Self::Rsqrt
            | Self::WormholeTrace
            | Self::KerrTrace
            | Self::KerrFrame
            | Self::KerrShade => 2,
            Self::Add | Self::Multiply | Self::MatMul => 3,
        }
    }

    const fn binding_bytes(self, lanes: u32, slot: usize) -> Option<u64> {
        let plane = lanes as u64 * 4;
        match self {
            Self::Identity | Self::Add | Self::Multiply | Self::Reciprocal | Self::Rsqrt => {
                Some(plane)
            }
            Self::MatMul => None,
            Self::WormholeTrace => match slot {
                0 => Some(plane * 5),
                1 => Some(plane * 4),
                _ => None,
            },
            Self::KerrTrace => match slot {
                0 => Some(plane * 9),
                1 => Some(plane * 12),
                _ => None,
            },
            Self::KerrFrame => match slot {
                0 => Some(4),
                1 => plane.checked_add(32),
                _ => None,
            },
            Self::KerrShade => None,
        }
    }
}

/// Stable provider-local artifact for one coarse direct-HTP dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectHtpArtifact(Vec<u8>);

impl DirectHtpArtifact {
    pub fn new(operation: DirectHtpOperation, lanes: u32) -> Option<Self> {
        if lanes == 0 {
            return None;
        }
        let mut bytes = [0; 16];
        let magic = ARTIFACT_MAGIC.to_le_bytes();
        let abi = ARTIFACT_ABI.to_le_bytes();
        let opcode = (operation as u32).to_le_bytes();
        let lanes = lanes.to_le_bytes();
        let mut index = 0;
        while index < 4 {
            bytes[index] = magic[index];
            bytes[4 + index] = abi[index];
            bytes[8 + index] = opcode[index];
            bytes[12 + index] = lanes[index];
            index += 1;
        }
        Some(Self(bytes.to_vec()))
    }

    pub fn wormhole_trace(lanes: u32, parameters: WormholeTraceParameters) -> Option<Self> {
        let mut artifact = Self::new(DirectHtpOperation::WormholeTrace, lanes)?;
        for word in [
            parameters.rho.to_bits(),
            parameters.a.to_bits(),
            parameters.m.to_bits(),
            parameters.step_size.to_bits(),
            parameters.max_steps,
            parameters.escape_ell.to_bits(),
        ] {
            artifact.0.extend_from_slice(&word.to_le_bytes());
        }
        Some(artifact)
    }

    pub fn kerr_trace(lanes: u32, parameters: KerrTraceParameters) -> Option<Self> {
        if !parameters.is_valid() {
            return None;
        }
        let mut artifact = Self::new(DirectHtpOperation::KerrTrace, lanes)?;
        for word in [
            parameters.mass.to_bits(),
            parameters.spin.to_bits(),
            parameters.step_size.to_bits(),
            parameters.max_steps,
            parameters.gradient_epsilon.to_bits(),
            parameters.escape_radius.to_bits(),
            parameters.disk_inner_radius.to_bits(),
            parameters.disk_outer_radius.to_bits(),
            parameters.termination_radius.to_bits(),
        ] {
            artifact.0.extend_from_slice(&word.to_le_bytes());
        }
        Some(artifact)
    }

    /// One coarse Kerr render: generate rays, trace, shade, and pack RGBA on HTP.
    pub fn kerr_frame(lanes: u32, parameters: KerrFrameParameters) -> Option<Self> {
        if parameters.width.checked_mul(parameters.height)? != lanes
            || parameters.samples_per_pixel != 1
            || !parameters.is_valid()
        {
            return None;
        }
        let mut artifact = Self::new(DirectHtpOperation::KerrFrame, lanes)?;
        for word in [
            parameters.trace.mass.to_bits(),
            parameters.trace.spin.to_bits(),
            parameters.trace.step_size.to_bits(),
            parameters.trace.max_steps,
            parameters.trace.gradient_epsilon.to_bits(),
            parameters.trace.escape_radius.to_bits(),
            parameters.trace.disk_inner_radius.to_bits(),
            parameters.trace.disk_outer_radius.to_bits(),
            parameters.trace.termination_radius.to_bits(),
            parameters.width,
            parameters.height,
            parameters.samples_per_pixel,
            parameters.tan_half_fov.to_bits(),
        ]
        .into_iter()
        .chain(parameters.camera_position.into_iter().map(f32::to_bits))
        .chain(parameters.camera_time.into_iter().map(f32::to_bits))
        .chain(parameters.camera_right.into_iter().map(f32::to_bits))
        .chain(parameters.camera_up.into_iter().map(f32::to_bits))
        .chain(parameters.camera_forward.into_iter().map(f32::to_bits))
        {
            artifact.0.extend_from_slice(&word.to_le_bytes());
        }
        Some(artifact)
    }

    /// Shade a packed reference Kerr boundary/plasma scene resident in shared
    /// memory. The scene ABI is provider-local and versioned by its own header.
    pub fn kerr_shade(lanes: u32, scene_bytes: u32) -> Option<Self> {
        if scene_bytes < 160 {
            return None;
        }
        let mut artifact = Self::new(DirectHtpOperation::KerrShade, lanes)?;
        artifact.0.extend_from_slice(&scene_bytes.to_le_bytes());
        Some(artifact)
    }

    pub fn matmul(rows: u32, inner: u32, columns: u32) -> Option<Self> {
        let lanes = rows.checked_mul(columns)?;
        if inner == 0 {
            return None;
        }
        let mut artifact = Self::new(DirectHtpOperation::MatMul, lanes)?;
        for word in [rows, inner, columns] {
            artifact.0.extend_from_slice(&word.to_le_bytes());
        }
        Some(artifact)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    fn parse(bytes: &[u8]) -> Result<(DirectHtpOperation, u32, Vec<u8>), BackendError> {
        if bytes.len() < 16 {
            return Err(BackendError::InvalidArgument);
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if word(0) != ARTIFACT_MAGIC || word(4) != ARTIFACT_ABI {
            return Err(BackendError::Incompatible);
        }
        let operation = match word(8) {
            1 => DirectHtpOperation::Identity,
            2 => DirectHtpOperation::Add,
            3 => DirectHtpOperation::Multiply,
            4 => DirectHtpOperation::Reciprocal,
            5 => DirectHtpOperation::Rsqrt,
            6 => DirectHtpOperation::MatMul,
            16 => DirectHtpOperation::WormholeTrace,
            17 => DirectHtpOperation::KerrTrace,
            18 => DirectHtpOperation::KerrFrame,
            19 => DirectHtpOperation::KerrShade,
            _ => return Err(BackendError::Unsupported),
        };
        let lanes = word(12);
        if lanes == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let expected_parameters = match operation {
            DirectHtpOperation::Identity
            | DirectHtpOperation::Add
            | DirectHtpOperation::Multiply
            | DirectHtpOperation::Reciprocal
            | DirectHtpOperation::Rsqrt => 0,
            DirectHtpOperation::MatMul => 12,
            DirectHtpOperation::WormholeTrace => 24,
            DirectHtpOperation::KerrTrace => 36,
            DirectHtpOperation::KerrFrame => 128,
            DirectHtpOperation::KerrShade => 4,
        };
        if bytes.len() != 16 + expected_parameters {
            return Err(BackendError::InvalidArgument);
        }
        Ok((operation, lanes, bytes[16..].to_vec()))
    }
}

impl ByteSource for DirectHtpArtifact {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        self.0.as_slice().read_at(offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WormholeTraceParameters {
    pub rho: f32,
    pub a: f32,
    pub m: f32,
    pub step_size: f32,
    pub max_steps: u32,
    pub escape_ell: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrTraceParameters {
    pub mass: f32,
    pub spin: f32,
    pub step_size: f32,
    pub max_steps: u32,
    pub gradient_epsilon: f32,
    pub escape_radius: f32,
    pub disk_inner_radius: f32,
    pub disk_outer_radius: f32,
    /// Positive capture/termination surface, at or inside the outer horizon.
    pub termination_radius: f32,
}

impl KerrTraceParameters {
    fn is_valid(self) -> bool {
        let horizon_discriminant = self.mass * self.mass - self.spin * self.spin;
        if !self.mass.is_finite()
            || !self.spin.is_finite()
            || !self.step_size.is_finite()
            || !self.gradient_epsilon.is_finite()
            || !self.escape_radius.is_finite()
            || !self.disk_inner_radius.is_finite()
            || !self.disk_outer_radius.is_finite()
            || !self.termination_radius.is_finite()
            || self.mass <= 0.0
            || horizon_discriminant < 0.0
        {
            return false;
        }
        let outer_horizon = self.mass + horizon_discriminant.sqrt();
        self.step_size > 0.0
            && self.max_steps > 0
            && self.gradient_epsilon > 0.0
            && self.termination_radius > 0.0
            && self.termination_radius <= outer_horizon
            && self.escape_radius > outer_horizon
            && self.disk_inner_radius >= outer_horizon
            && self.disk_outer_radius >= self.disk_inner_radius
    }
}

/// Frozen Axiom Kerr camera/render state consumed by the fused HTP frame program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrFrameParameters {
    pub trace: KerrTraceParameters,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub tan_half_fov: f32,
    pub camera_position: [f32; 3],
    pub camera_time: [f32; 4],
    pub camera_right: [f32; 4],
    pub camera_up: [f32; 4],
    pub camera_forward: [f32; 4],
}

impl KerrFrameParameters {
    fn is_valid(self) -> bool {
        self.trace.is_valid()
            && self.width > 0
            && self.height > 0
            && self.samples_per_pixel == 1
            && self.tan_half_fov.is_finite()
            && self.tan_half_fov > 0.0
            && self
                .camera_position
                .into_iter()
                .chain(self.camera_time)
                .chain(self.camera_right)
                .chain(self.camera_up)
                .chain(self.camera_forward)
                .all(f32::is_finite)
    }
}

#[repr(C)]
struct NativeRuntime {
    _private: [u8; 0],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct NativeRuntimeInfo {
    arch: u32,
    hvx_units: u32,
    vtcm_bytes: u32,
    arena_bytes: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct NativeBinding {
    address: *mut c_void,
    bytes: u32,
    slot: u32,
    access: u32,
}

const SUCCESS: u64 = 0;
const ERROR_UNAVAILABLE: u64 = 1;
const ERROR_INCOMPATIBLE: u64 = 2;
const ERROR_INVALID_ARGUMENT: u64 = 3;
const ERROR_OUT_OF_MEMORY: u64 = 4;
const ERROR_DEVICE_LOST: u64 = 5;
const ERROR_BUSY: u64 = 6;

unsafe extern "C" {
    fn va_htp_runtime_create(
        module_directory: *const c_char,
        arena_bytes: u32,
        runtime: *mut *mut NativeRuntime,
        info: *mut NativeRuntimeInfo,
        message: *mut c_char,
        message_bytes: usize,
    ) -> u64;
    fn va_htp_runtime_free(runtime: *mut NativeRuntime) -> u64;
    fn va_htp_buffer_alloc(
        runtime: *mut NativeRuntime,
        bytes: u32,
        alignment: u32,
        offset: *mut u32,
        address: *mut *mut c_void,
    ) -> u64;
    fn va_htp_buffer_free(runtime: *mut NativeRuntime, address: *mut c_void, bytes: u32) -> u64;
    fn va_htp_execute_direct(
        runtime: *mut NativeRuntime,
        opcode: u32,
        lanes: u32,
        parameters: *const c_void,
        parameter_bytes: u32,
        bindings: *const NativeBinding,
        binding_count: u32,
        elapsed_cycles: *mut u64,
    ) -> u64;
}

fn backend_error(status: u64) -> BackendError {
    match status {
        ERROR_INCOMPATIBLE => BackendError::Incompatible,
        ERROR_INVALID_ARGUMENT => BackendError::InvalidArgument,
        ERROR_OUT_OF_MEMORY => BackendError::OutOfMemory,
        ERROR_DEVICE_LOST => BackendError::DeviceLost,
        ERROR_BUSY => BackendError::Busy,
        other => BackendError::External {
            domain: EXTERNAL_DOMAIN,
            code: other as i64,
        },
    }
}

fn check(status: u64) -> Result<(), BackendError> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(backend_error(status))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectHtpRuntimeInfo {
    pub arch: u32,
    pub hvx_units: u32,
    pub vtcm_bytes: u32,
    pub arena_bytes: u32,
}

#[derive(Debug)]
struct Runtime {
    raw: NonNull<NativeRuntime>,
    info: DirectHtpRuntimeInfo,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: this Rc owner holds the live native runtime and releases it once.
        let result = unsafe { va_htp_runtime_free(self.raw.as_ptr()) };
        debug_assert_eq!(result, SUCCESS);
    }
}

#[derive(Debug)]
pub struct DirectHexagonContext {
    id: u64,
}

#[derive(Debug)]
pub struct DirectHexagonBuffer {
    context_id: u64,
    desc: BufferDesc,
    bytes: u32,
    address: NonNull<u8>,
    runtime: Rc<Runtime>,
}

impl DirectHexagonBuffer {
    /// Returns the mapped shared-memory contents without an intermediate copy.
    ///
    /// # Safety
    ///
    /// The caller must not retain the returned slice across a submission that
    /// can write this buffer, free the buffer, or destroy its context.
    pub unsafe fn mapped_bytes(&self) -> &[u8] {
        // SAFETY: allocation establishes a live arena subrange of `bytes`
        // bytes at `address`; the caller upholds the synchronization contract.
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.bytes as usize) }
    }
}

#[derive(Debug)]
pub struct DirectHexagonProgram {
    context_id: u64,
    operation: DirectHtpOperation,
    lanes: u32,
    parameters: Vec<u8>,
}

impl DirectHexagonProgram {
    fn binding_bytes(&self, slot: usize) -> Option<u64> {
        if !matches!(
            self.operation,
            DirectHtpOperation::MatMul | DirectHtpOperation::KerrShade
        ) {
            return self.operation.binding_bytes(self.lanes, slot);
        }
        let word = |offset| {
            self.parameters
                .get(offset..offset + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
        };
        if self.operation == DirectHtpOperation::KerrShade {
            return match slot {
                0 => Some(u64::from(word(0)?)),
                1 => u64::from(self.lanes).checked_mul(4),
                _ => None,
            };
        }
        let rows = u64::from(word(0)?);
        let inner = u64::from(word(4)?);
        let columns = u64::from(word(8)?);
        match slot {
            0 => rows.checked_mul(inner)?.checked_mul(4),
            1 => inner.checked_mul(columns)?.checked_mul(4),
            2 => rows.checked_mul(columns)?.checked_mul(4),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct DirectHexagonQueue {
    context_id: u64,
}

#[derive(Debug)]
pub struct DirectHexagonEvent {
    state: EventState,
    elapsed_cycles: u64,
}

impl DirectHexagonEvent {
    pub const fn elapsed_cycles(&self) -> u64 {
        self.elapsed_cycles
    }
}

pub struct DirectHexagonAccelerator {
    runtime: Rc<Runtime>,
    next_id: Cell<u64>,
    direct_binding_admissions: Cell<u64>,
    explicit_transfer_bytes: Cell<u64>,
    info: DeviceInfo,
}

impl std::fmt::Debug for DirectHexagonAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DirectHexagonAccelerator")
            .field("runtime", &self.runtime.info)
            .finish_non_exhaustive()
    }
}

impl DirectHexagonAccelerator {
    pub fn new() -> Result<Self, DirectInitError> {
        let directory = module_directory()?;
        let directory = CString::new(directory.to_string_lossy().as_bytes())
            .map_err(|_| DirectInitError::ModuleUnavailable)?;
        let arena_bytes = std::env::var("VIRTIO_ACCEL_HTP_ARENA_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_ARENA_BYTES);
        let mut raw = ptr::null_mut();
        let mut native_info = NativeRuntimeInfo::default();
        let mut message = [0; MESSAGE_BYTES];
        // SAFETY: all pointers are valid for this synchronous constructor call.
        let status = unsafe {
            va_htp_runtime_create(
                directory.as_ptr(),
                arena_bytes,
                &mut raw,
                &mut native_info,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        if status != SUCCESS {
            let details = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy();
            eprintln!("direct HTP initialization failed: {details}");
            return Err(match status {
                ERROR_INCOMPATIBLE => DirectInitError::IncompatibleHardware,
                ERROR_UNAVAILABLE => DirectInitError::ModuleUnavailable,
                _ => DirectInitError::RuntimeUnavailable,
            });
        }
        let raw = NonNull::new(raw).ok_or(DirectInitError::RuntimeUnavailable)?;
        let runtime_info = DirectHtpRuntimeInfo {
            arch: native_info.arch,
            hvx_units: native_info.hvx_units,
            vtcm_bytes: native_info.vtcm_bytes,
            arena_bytes: native_info.arena_bytes,
        };
        let runtime = Rc::new(Runtime {
            raw,
            info: runtime_info,
        });
        Ok(Self {
            runtime,
            next_id: Cell::new(0),
            direct_binding_admissions: Cell::new(0),
            explicit_transfer_bytes: Cell::new(0),
            info: DeviceInfo {
                identity: DeviceIdentity {
                    uuid: *b"qcom-htp-v73-qf3",
                    class: AcceleratorClass::NPU,
                    vendor_id: 0x17cb,
                    device_id: 73,
                },
                capabilities: Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY,
                limits: DeviceLimits {
                    max_contexts: 1,
                    max_buffers_per_context: 1_024,
                    max_programs_per_context: 64,
                    max_queues_per_context: 64,
                    max_events_per_context: 1,
                    max_bindings_per_submission: 32,
                    max_buffer_bytes: u64::from(native_info.arena_bytes),
                    max_artifact_bytes: MAX_ARTIFACT_BYTES,
                },
            },
        })
    }

    pub fn runtime_info(&self) -> DirectHtpRuntimeInfo {
        self.runtime.info
    }

    pub fn direct_binding_admissions(&self) -> u64 {
        self.direct_binding_admissions.get()
    }

    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.explicit_transfer_bytes.get()
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        let next = self
            .next_id
            .get()
            .checked_add(1)
            .ok_or(BackendError::ResourceLimit)?;
        self.next_id.set(next);
        Ok(next)
    }

    fn checked_range(
        buffer: &DirectHexagonBuffer,
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
}

fn module_directory() -> Result<PathBuf, DirectInitError> {
    if let Some(path) = std::env::var_os("VIRTIO_ACCEL_HTP_MODULE_DIR") {
        return Ok(PathBuf::from(path));
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or(DirectInitError::ModuleUnavailable)
}

impl Accelerator for DirectHexagonAccelerator {
    type Context = DirectHexagonContext;
    type Buffer = DirectHexagonBuffer;
    type Program = DirectHexagonProgram;
    type Queue = DirectHexagonQueue;
    type Event = DirectHexagonEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.info.validate_context_desc(desc)?;
        Ok(DirectHexagonContext {
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
        let bytes = u32::try_from(desc.bytes()).map_err(|_| BackendError::ResourceLimit)?;
        let alignment = u32::try_from(desc.alignment().max(MIN_ALIGNMENT))
            .map_err(|_| BackendError::ResourceLimit)?;
        let mut offset = 0;
        let mut address = ptr::null_mut();
        // SAFETY: outputs are valid and the runtime outlives the returned buffer.
        check(unsafe {
            va_htp_buffer_alloc(
                self.runtime.raw.as_ptr(),
                bytes,
                alignment,
                &mut offset,
                &mut address,
            )
        })?;
        let address = NonNull::new(address.cast()).ok_or(BackendError::DeviceLost)?;
        let properties = BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING;
        let info = BufferInfo::new(desc, bytes.into(), alignment.into(), properties)?;
        Ok(AllocatedBuffer::new(
            DirectHexagonBuffer {
                context_id: context.id,
                desc,
                bytes,
                address,
                runtime: Rc::clone(&self.runtime),
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
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let target = unsafe {
            std::slice::from_raw_parts_mut(buffer.address.as_ptr().add(start), end - start)
        };
        data.read_at(0, target)?;
        self.explicit_transfer_bytes.set(
            self.explicit_transfer_bytes
                .get()
                .saturating_add(data.len()),
        );
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
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let source =
            unsafe { std::slice::from_raw_parts(buffer.address.as_ptr().add(start), end - start) };
        data.write_at(0, source)?;
        self.explicit_transfer_bytes.set(
            self.explicit_transfer_bytes
                .get()
                .saturating_add(data.len()),
        );
        Ok(())
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        // SAFETY: consuming the Rust buffer proves its arena subrange is no longer bindable.
        match check(unsafe {
            va_htp_buffer_free(
                buffer.runtime.raw.as_ptr(),
                buffer.address.as_ptr().cast(),
                buffer.bytes,
            )
        }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ReleaseFailure::Indeterminate { error }),
        }
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if artifact.format != DIRECT_HTP_ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        if artifact.target != DIRECT_HTP_V73_TARGET {
            return Err(BackendError::Incompatible);
        }
        if artifact.payload.len() > MAX_ARTIFACT_BYTES
            || artifact.resident_bytes < artifact.payload.len()
        {
            return Err(BackendError::ResourceLimit);
        }
        let payload_len =
            usize::try_from(artifact.payload.len()).map_err(|_| BackendError::ResourceLimit)?;
        let mut bytes = vec![0; payload_len];
        artifact.payload.read_at(0, &mut bytes)?;
        let (operation, lanes, parameters) = DirectHtpArtifact::parse(&bytes)?;
        Ok(DirectHexagonProgram {
            context_id: context.id,
            operation,
            lanes,
            parameters,
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
        Ok(DirectHexagonQueue {
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
            return Err(SubmitFailure::Rejected(BackendError::DeadlineExpired));
        }
        if queue.context_id != program.context_id
            || bindings.len() != program.operation.binding_count()
        {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }
        let mut native = Vec::with_capacity(bindings.len());
        for expected_slot in 0..bindings.len() {
            let binding = bindings
                .iter()
                .find(|binding| binding.slot == expected_slot as u32)
                .ok_or(SubmitFailure::Rejected(BackendError::Incompatible))?;
            let expected_access = if expected_slot + 1 == bindings.len() {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            if binding.access != expected_access
                || !binding.buffer.desc.allows_access(binding.access)
            {
                return Err(SubmitFailure::Rejected(BackendError::PermissionDenied));
            }
            let required_bytes = program
                .binding_bytes(expected_slot)
                .ok_or(SubmitFailure::Rejected(BackendError::Incompatible))?;
            if binding.buffer.context_id != queue.context_id
                || binding.range.bytes() != required_bytes
            {
                return Err(SubmitFailure::Rejected(BackendError::Incompatible));
            }
            Self::checked_range(binding.buffer, binding.range.offset, binding.range.bytes())
                .map_err(SubmitFailure::Rejected)?;
            let byte_offset = usize::try_from(binding.range.offset)
                .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfBounds))?;
            native.push(NativeBinding {
                address: unsafe { binding.buffer.address.as_ptr().add(byte_offset).cast() },
                bytes: u32::try_from(binding.range.bytes())
                    .map_err(|_| SubmitFailure::Rejected(BackendError::OutOfBounds))?,
                slot: binding.slot,
                access: binding.access as u32,
            });
        }
        let mut elapsed_cycles = 0;
        // SAFETY: every native binding names a validated live arena range. Execution is
        // synchronous, so no pointer or binding metadata is retained after the call.
        check(unsafe {
            va_htp_execute_direct(
                self.runtime.raw.as_ptr(),
                program.operation as u32,
                program.lanes,
                program.parameters.as_ptr().cast(),
                program.parameters.len() as u32,
                native.as_ptr(),
                native.len() as u32,
                &mut elapsed_cycles,
            )
        })
        .map_err(SubmitFailure::Rejected)?;
        self.direct_binding_admissions.set(
            self.direct_binding_admissions
                .get()
                .saturating_add(bindings.len() as u64),
        );
        Ok(DirectHexagonEvent {
            state: EventState::Complete,
            elapsed_cycles,
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        Ok(event.state)
    }

    fn destroy_event(&self, _event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_core::{BufferRange, ContextDesc, QueueDesc};

    fn desc(bytes: u64, input: bool) -> BufferDesc {
        BufferDesc::new(
            bytes,
            128,
            MemoryDomain::Shared,
            if input {
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT
            } else {
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT
            },
        )
        .unwrap()
    }

    #[test]
    fn fused_kerr_frame_artifact_has_compact_exact_bindings() {
        let trace = KerrTraceParameters {
            mass: 1.0,
            spin: 0.6,
            step_size: 0.18,
            max_steps: 320,
            gradient_epsilon: 0.01,
            escape_radius: 32.0,
            disk_inner_radius: 3.829_069,
            disk_outer_radius: 15.0,
            termination_radius: 1.8,
        };
        let parameters = KerrFrameParameters {
            trace,
            width: 160,
            height: 90,
            samples_per_pixel: 1,
            tan_half_fov: 0.58,
            camera_position: [0.0, -22.0, 7.5],
            camera_time: [1.0, 0.0, 0.0, 0.0],
            camera_right: [0.0, 1.0, 0.0, 0.0],
            camera_up: [0.0, 0.0, 0.0, 1.0],
            camera_forward: [0.0, 0.0, 1.0, 0.0],
        };
        let artifact = DirectHtpArtifact::kerr_frame(14_400, parameters).unwrap();
        let (operation, lanes, encoded) = DirectHtpArtifact::parse(artifact.bytes()).unwrap();
        assert_eq!(operation, DirectHtpOperation::KerrFrame);
        assert_eq!(lanes, 14_400);
        assert_eq!(encoded.len(), 128);
        assert_eq!(operation.binding_bytes(lanes, 0), Some(4));
        assert_eq!(operation.binding_bytes(lanes, 1), Some(57_632));
        assert!(DirectHtpArtifact::kerr_frame(14_399, parameters).is_none());
        assert!(
            DirectHtpArtifact::kerr_frame(
                14_400,
                KerrFrameParameters {
                    samples_per_pixel: 2,
                    ..parameters
                }
            )
            .is_none()
        );
        assert!(
            DirectHtpArtifact::kerr_frame(
                14_400,
                KerrFrameParameters {
                    tan_half_fov: f32::NAN,
                    ..parameters
                }
            )
            .is_none()
        );
        assert!(
            DirectHtpArtifact::kerr_frame(
                14_400,
                KerrFrameParameters {
                    camera_forward: [0.0, 0.0, f32::INFINITY, 0.0],
                    ..parameters
                }
            )
            .is_none()
        );
        assert!(
            DirectHtpArtifact::kerr_frame(
                14_400,
                KerrFrameParameters {
                    trace: KerrTraceParameters {
                        termination_radius: 1.81,
                        ..trace
                    },
                    ..parameters
                }
            )
            .is_none()
        );
    }

    #[derive(Debug)]
    struct SliceBytes<'a>(&'a [u8]);

    impl ByteSource for SliceBytes<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
            let start = offset as usize;
            let end = start
                .checked_add(target.len())
                .ok_or(BackendError::OutOfBounds)?;
            target.copy_from_slice(self.0.get(start..end).ok_or(BackendError::OutOfBounds)?);
            Ok(())
        }
        fn as_contiguous(&self) -> Option<&[u8]> {
            Some(self.0)
        }
    }

    #[derive(Debug)]
    struct SliceBytesMut<'a>(&'a mut [u8]);

    impl ByteSink for SliceBytesMut<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
            let start = offset as usize;
            let end = start
                .checked_add(source.len())
                .ok_or(BackendError::OutOfBounds)?;
            self.0
                .get_mut(start..end)
                .ok_or(BackendError::OutOfBounds)?
                .copy_from_slice(source);
            Ok(())
        }
        fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
            Some(self.0)
        }
    }

    fn execute_probe(
        accelerator: &DirectHexagonAccelerator,
        context: &DirectHexagonContext,
        queue: &DirectHexagonQueue,
        artifact: &DirectHtpArtifact,
        inputs: &[&[f32]],
        output_values: usize,
    ) -> Vec<f32> {
        let program = accelerator
            .load_program(
                context,
                ArtifactRef {
                    format: DIRECT_HTP_ARTIFACT_FORMAT,
                    target: DIRECT_HTP_V73_TARGET,
                    payload: artifact,
                    resident_bytes: artifact.bytes().len() as u64,
                },
            )
            .unwrap();
        let mut input_buffers = Vec::with_capacity(inputs.len());
        for values in inputs {
            let bytes = (values.len() * 4) as u64;
            let (mut buffer, _) = accelerator
                .allocate_buffer(context, desc(bytes, true))
                .unwrap()
                .into_parts();
            let encoded = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            accelerator
                .write_buffer(&mut buffer, 0, &SliceBytes(encoded.as_slice()))
                .unwrap();
            input_buffers.push(buffer);
        }
        let output_bytes = (output_values * 4) as u64;
        let (output, _) = accelerator
            .allocate_buffer(context, desc(output_bytes, false))
            .unwrap()
            .into_parts();
        let mut bindings = input_buffers
            .iter()
            .enumerate()
            .map(|(slot, buffer)| BindingRef {
                slot: slot as u32,
                buffer,
                range: BufferRange::new(0, u64::from(buffer.bytes)).unwrap(),
                access: AccessMode::Read,
            })
            .collect::<Vec<_>>();
        bindings.push(BindingRef {
            slot: inputs.len() as u32,
            buffer: &output,
            range: BufferRange::new(0, output_bytes).unwrap(),
            access: AccessMode::Write,
        });
        let event = accelerator
            .submit(queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        assert!(event.elapsed_cycles() > 0);
        accelerator.destroy_event(event).unwrap();
        let mut encoded = vec![0; output_bytes as usize];
        accelerator
            .read_buffer(&output, 0, &mut SliceBytesMut(&mut encoded))
            .unwrap();
        let values = encoded
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        accelerator.unload_program(program).unwrap();
        for buffer in input_buffers {
            accelerator.free_buffer(buffer).unwrap();
        }
        accelerator.free_buffer(output).unwrap();
        values
    }

    fn report_probe(name: &str, output: &[f32]) {
        let bits = output
            .iter()
            .map(|value| format!("0x{:08x}", value.to_bits()))
            .collect::<Vec<_>>();
        eprintln!("direct HTP {name}: {bits:?}");
    }

    #[test]
    #[ignore = "requires the signed V73 skel and Qualcomm FastRPC driver"]
    fn direct_fp32_capability_spike_covers_required_operations_and_edges() {
        let accelerator = DirectHexagonAccelerator::new().expect("direct V73 runtime");
        assert_eq!(accelerator.runtime_info().arch, 73);
        assert!(accelerator.runtime_info().hvx_units > 0);
        let context = accelerator.create_context(ContextDesc::default()).unwrap();
        let queue = accelerator
            .create_queue(&context, QueueDesc::default())
            .unwrap();

        let edge = [
            1.0_f32,
            f32::from_bits(1),
            -0.0,
            70_000.0,
            -70_000.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc1_2345),
        ];
        let identity = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::new(DirectHtpOperation::Identity, edge.len() as u32).unwrap(),
            &[&edge],
            edge.len(),
        );
        report_probe("identity", &identity);
        assert_eq!(
            identity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            edge.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let add_lhs = [
            1.0,
            70_000.0,
            f32::from_bits(1),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            3.0,
        ];
        let add_rhs = [
            2.0_f32.powi(-20),
            1.0,
            f32::from_bits(1),
            0.0,
            1.0,
            f32::INFINITY,
            1.0,
            -3.0,
        ];
        let add = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::new(DirectHtpOperation::Add, add_lhs.len() as u32).unwrap(),
            &[&add_lhs, &add_rhs],
            add_lhs.len(),
        );
        report_probe("add", &add);
        assert_ne!(add[0].to_bits(), 1.0_f32.to_bits());
        assert!(add[1].is_finite() && add[1] > 65_504.0);
        assert_eq!(add[3].to_bits(), 0.0_f32.to_bits());
        assert!(add[4].is_infinite() && add[4].is_sign_positive());
        assert!(add[5].is_nan() && add[6].is_nan());

        let multiply_rhs = [1.0 + 2.0_f32.powi(-20), 2.0, 2.0, -1.0, 0.0, 0.0, 2.0, -1.0];
        let multiply = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::new(DirectHtpOperation::Multiply, edge.len() as u32).unwrap(),
            &[&edge, &multiply_rhs],
            edge.len(),
        );
        report_probe("multiply", &multiply);
        assert_ne!(multiply[0].to_bits(), 1.0_f32.to_bits());
        assert!(multiply[3].is_finite() && multiply[3] < -65_504.0);
        assert_eq!(multiply[2].to_bits(), (-0.0_f32).to_bits());
        assert!(multiply[5].is_nan());
        assert!(multiply[6].is_infinite() && multiply[6].is_sign_negative());
        assert!(multiply[7].is_nan());

        let reciprocal_input = [
            3.0,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            70_000.0,
            f32::from_bits(1),
        ];
        let reciprocal = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::new(
                DirectHtpOperation::Reciprocal,
                reciprocal_input.len() as u32,
            )
            .unwrap(),
            &[&reciprocal_input],
            reciprocal_input.len(),
        );
        report_probe("reciprocal", &reciprocal);
        assert!(reciprocal[1].is_infinite() && reciprocal[1].is_sign_negative());
        assert!(reciprocal[2].is_infinite() && reciprocal[2].is_sign_positive());
        assert_eq!(reciprocal[3].to_bits(), 0.0_f32.to_bits());
        assert_eq!(reciprocal[4].to_bits(), (-0.0_f32).to_bits());
        assert!(reciprocal[5].is_nan());

        let rsqrt_input = [
            4.0,
            2.0,
            0.0,
            -0.0,
            f32::INFINITY,
            -1.0,
            f32::NAN,
            f32::from_bits(1),
        ];
        let rsqrt = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::new(DirectHtpOperation::Rsqrt, rsqrt_input.len() as u32).unwrap(),
            &[&rsqrt_input],
            rsqrt_input.len(),
        );
        report_probe("rsqrt", &rsqrt);
        assert_eq!(rsqrt[0], 0.5);
        assert!(rsqrt[2].is_infinite() && rsqrt[2].is_sign_positive());
        assert!(rsqrt[3].is_infinite() && rsqrt[3].is_sign_negative());
        assert_eq!(rsqrt[4].to_bits(), 0.0_f32.to_bits());
        assert!(rsqrt[5].is_nan() && rsqrt[6].is_nan());

        let mat_lhs = [1.0, 2.0, 3.0, 70_000.0, 2.0_f32.powi(-20), -1.0];
        let mat_rhs = [4.0, -1.0, 5.0, 2.0, 6.0, 0.5];
        let matmul = execute_probe(
            &accelerator,
            &context,
            &queue,
            &DirectHtpArtifact::matmul(2, 3, 2).unwrap(),
            &[&mat_lhs, &mat_rhs],
            4,
        );
        report_probe("matmul", &matmul);
        for (actual, expected) in matmul.iter().zip([32.0, 4.5, 279_994.0, -70_000.0]) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-5);
        }

        assert_eq!(accelerator.direct_binding_admissions(), 15);
        accelerator.destroy_queue(queue).unwrap();
        accelerator.destroy_context(context).unwrap();
    }

    #[test]
    #[ignore = "requires the signed V73 skel and Qualcomm FastRPC driver"]
    fn direct_qfloat32_add_uses_full_accelerator_lifecycle() {
        let accelerator = DirectHexagonAccelerator::new().expect("direct V73 runtime");
        assert_eq!(accelerator.runtime_info().arch, 73);
        assert!(accelerator.runtime_info().hvx_units > 0);
        let context = accelerator.create_context(ContextDesc::default()).unwrap();
        let queue = accelerator
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let values = 32_u32;
        let bytes = u64::from(values) * 4;
        let (mut lhs, _) = accelerator
            .allocate_buffer(&context, desc(bytes, true))
            .unwrap()
            .into_parts();
        let (mut rhs, _) = accelerator
            .allocate_buffer(&context, desc(bytes, true))
            .unwrap()
            .into_parts();
        let (out, _) = accelerator
            .allocate_buffer(&context, desc(bytes, false))
            .unwrap()
            .into_parts();

        let lhs_values = (0..values).map(|i| 1.0_f32 + i as f32).collect::<Vec<_>>();
        let rhs_values = (0..values)
            .map(|i| if i == 0 { 2.0_f32.powi(-20) } else { 0.25 })
            .collect::<Vec<_>>();
        let bytes_of = |values: &[f32]| {
            let mut bytes = [0_u8; 128];
            for (target, value) in bytes.chunks_exact_mut(4).zip(values) {
                target.copy_from_slice(&value.to_le_bytes());
            }
            bytes
        };
        let lhs_bytes = bytes_of(&lhs_values);
        let rhs_bytes = bytes_of(&rhs_values);
        accelerator.write_buffer(&mut lhs, 0, &lhs_bytes).unwrap();
        accelerator.write_buffer(&mut rhs, 0, &rhs_bytes).unwrap();

        let artifact = DirectHtpArtifact::new(DirectHtpOperation::Add, values).unwrap();
        let program = accelerator
            .load_program(
                &context,
                ArtifactRef {
                    format: DIRECT_HTP_ARTIFACT_FORMAT,
                    target: DIRECT_HTP_V73_TARGET,
                    payload: &artifact,
                    resident_bytes: 16,
                },
            )
            .unwrap();
        let range = BufferRange::new(0, bytes).unwrap();
        let event = accelerator
            .submit(
                &queue,
                &program,
                &[
                    BindingRef {
                        slot: 0,
                        buffer: &lhs,
                        range,
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 1,
                        buffer: &rhs,
                        range,
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 2,
                        buffer: &out,
                        range,
                        access: AccessMode::Write,
                    },
                ],
                Timeout::Infinite,
            )
            .unwrap();
        assert_eq!(
            accelerator.poll_event(&event).unwrap(),
            EventState::Complete
        );
        assert!(event.elapsed_cycles() > 0);
        accelerator.destroy_event(event).unwrap();
        let mut output = [0_u8; 128];
        accelerator.read_buffer(&out, 0, &mut output).unwrap();
        let output = output
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        // V73 QFloat32 keeps information below FP16 resolution. Different
        // compiler revisions have produced the nearest IEEE result and a
        // result one FP32 ULP above it, so admission does not claim IEEE.
        assert!(matches!(output[0].to_bits(), 0x3f80_0008 | 0x3f80_0009));
        assert_ne!(output[0].to_bits(), 1.0_f32.to_bits());
        for i in 1..values as usize {
            assert!((output[i] - (lhs_values[i] + rhs_values[i])).abs() <= 4.0e-6);
        }
        assert_eq!(accelerator.direct_binding_admissions(), 3);
        assert_eq!(accelerator.explicit_transfer_bytes(), bytes * 3);

        accelerator.unload_program(program).unwrap();
        accelerator.free_buffer(lhs).unwrap();
        accelerator.free_buffer(rhs).unwrap();
        accelerator.free_buffer(out).unwrap();
        accelerator.destroy_queue(queue).unwrap();
        accelerator.destroy_context(context).unwrap();
    }

    #[test]
    #[ignore = "requires the signed V73 skel and Qualcomm FastRPC driver"]
    #[allow(clippy::needless_range_loop)] // SoA planes are indexed by logical lane and component.
    fn fused_wormhole_trace_runs_on_direct_htp() {
        const LANES: usize = 32;
        fn radius(p: WormholeTraceParameters, ell: f32) -> f32 {
            let exterior = ell.abs() - p.a;
            if exterior <= 0.0 {
                return p.rho;
            }
            let x = (2.0 / std::f32::consts::PI) * exterior / p.m;
            p.rho + p.m * (x * x.atan() - 0.5 * (x * x).ln_1p())
        }
        fn force(p: WormholeTraceParameters, ell: f32, impact: f32) -> f32 {
            let exterior = ell.abs() - p.a;
            let r = radius(p, ell);
            let derivative = if exterior <= 0.0 {
                0.0
            } else {
                let x = (2.0 / std::f32::consts::PI) * exterior / p.m;
                (2.0 / std::f32::consts::PI) * x.atan() * if ell < 0.0 { -1.0 } else { 1.0 }
            };
            impact * impact * derivative / (r * r * r)
        }
        fn trace(p: WormholeTraceParameters, mut state: [f32; 4]) -> ([f32; 3], f32) {
            let terminal = |ell: f32, p_ell: f32| {
                (ell <= -p.escape_ell && p_ell < 0.0) || (ell >= p.escape_ell && p_ell > 0.0)
            };
            let mut active = 1.0;
            for _ in 0..p.max_steps {
                if terminal(state[0], state[2]) {
                    active = 0.0;
                    break;
                }
                let old_r = radius(p, state[0]);
                let half = 0.5 * p.step_size;
                state[2] += half * force(p, state[0], state[3]);
                state[0] += p.step_size * state[2];
                let new_r = radius(p, state[0]);
                state[2] += half * force(p, state[0], state[3]);
                state[1] += half * state[3] * (1.0 / (old_r * old_r) + 1.0 / (new_r * new_r));
                if terminal(state[0], state[2]) {
                    active = 0.0;
                    break;
                }
            }
            ([state[0], state[1], state[2]], active)
        }

        let parameters = WormholeTraceParameters {
            rho: 1.0,
            a: 2.0,
            m: 0.1,
            step_size: 0.05,
            max_steps: 160,
            escape_ell: 10.0,
        };
        let mut planes = [[0.0_f32; LANES]; 5];
        for lane in 0..LANES {
            planes[0][lane] = 8.0;
            planes[1][lane] = 0.0;
            planes[3][lane] = lane as f32 / LANES as f32 * 1.4;
            let r = radius(parameters, planes[0][lane]);
            planes[2][lane] = -(1.0 - planes[3][lane].powi(2) / r.powi(2)).sqrt();
            planes[4][lane] = 1.0;
        }
        let mut input_bytes = Vec::with_capacity(LANES * 5 * 4);
        for plane in &planes {
            for value in plane {
                input_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut output_bytes = vec![0_u8; LANES * 4 * 4];

        let accelerator = DirectHexagonAccelerator::new().unwrap();
        let context = accelerator.create_context(ContextDesc::default()).unwrap();
        let queue = accelerator
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let (mut input, _) = accelerator
            .allocate_buffer(&context, desc(input_bytes.len() as u64, true))
            .unwrap()
            .into_parts();
        let (output, _) = accelerator
            .allocate_buffer(&context, desc(output_bytes.len() as u64, false))
            .unwrap()
            .into_parts();
        accelerator
            .write_buffer(&mut input, 0, &SliceBytes(&input_bytes))
            .unwrap();
        let artifact = DirectHtpArtifact::wormhole_trace(LANES as u32, parameters).unwrap();
        let program = accelerator
            .load_program(
                &context,
                ArtifactRef {
                    format: DIRECT_HTP_ARTIFACT_FORMAT,
                    target: DIRECT_HTP_V73_TARGET,
                    payload: &artifact,
                    resident_bytes: artifact.bytes().len() as u64,
                },
            )
            .unwrap();
        let event = accelerator
            .submit(
                &queue,
                &program,
                &[
                    BindingRef {
                        slot: 0,
                        buffer: &input,
                        range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 1,
                        buffer: &output,
                        range: BufferRange::new(0, output_bytes.len() as u64).unwrap(),
                        access: AccessMode::Write,
                    },
                ],
                Timeout::Infinite,
            )
            .unwrap();
        assert_eq!(
            accelerator.poll_event(&event).unwrap(),
            EventState::Complete
        );
        accelerator.destroy_event(event).unwrap();
        accelerator
            .read_buffer(&output, 0, &mut SliceBytesMut(&mut output_bytes))
            .unwrap();
        let value = |plane: usize, lane: usize| {
            let start = (plane * LANES + lane) * 4;
            f32::from_le_bytes(output_bytes[start..start + 4].try_into().unwrap())
        };
        for lane in 0..LANES {
            let (expected, active) = trace(
                parameters,
                [
                    planes[0][lane],
                    planes[1][lane],
                    planes[2][lane],
                    planes[3][lane],
                ],
            );
            for plane in 0..3 {
                // The V73 HVX atan/log polynomial path is a relaxed QFloat32 tier.
                let tolerance = 1.0e-2 * expected[plane].abs().max(1.0);
                assert!(
                    (value(plane, lane) - expected[plane]).abs() <= tolerance,
                    "lane {lane} plane {plane}: {} vs {}",
                    value(plane, lane),
                    expected[plane]
                );
            }
            assert_eq!(value(3, lane), active);
        }
        accelerator.unload_program(program).unwrap();
        accelerator.free_buffer(input).unwrap();
        accelerator.free_buffer(output).unwrap();
        accelerator.destroy_queue(queue).unwrap();
        accelerator.destroy_context(context).unwrap();
    }
}

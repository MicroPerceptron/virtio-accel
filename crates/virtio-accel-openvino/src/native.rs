//! Native OpenVINO runtime integration: core lifecycle and device discovery.
//!
//! Every `unsafe` block carries a `SAFETY:` comment referencing the invariants audited in
//! `SAFETY.md`. Handle ownership is single-owner with `Drop`; the shared `ov_core_t` is
//! reference-counted so dependent handles keep the runtime alive.

use std::ffi::{CStr, CString};
use std::ptr::{self, NonNull};
use std::sync::{Arc, OnceLock};

use virtio_accel_core::{AcceleratorClass, Capabilities, DeviceIdentity, DeviceInfo, DeviceLimits};

use crate::InitError;
use crate::ffi;

/// Maximal TOSA artifact bytes admitted before parsing.
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

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

/// Intel OpenVINO backend instance bound to one inference device.
pub struct OpenVinoAccelerator {
    core: Arc<CoreHandle>,
    device: CString,
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
        Ok(Self { core, device, info })
    }

    /// The enumerated name of the device this instance executes on.
    pub fn device_name(&self) -> &str {
        self.device.to_str().unwrap_or_default()
    }

    /// Re-enumerate inference devices through this instance's runtime core.
    pub fn runtime_devices(&self) -> Result<Vec<String>, InitError> {
        self.core.available_devices()
    }

    /// Stable device metadata advertised by this instance.
    pub fn device_info(&self) -> DeviceInfo {
        self.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::c_char;

    fn backend() -> Option<OpenVinoAccelerator> {
        match OpenVinoAccelerator::new() {
            Ok(backend) => Some(backend),
            Err(InitError::DeviceUnavailable) => None,
            Err(error) => panic!("backend initialization failed: {error}"),
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
        backend.device_info().validate().unwrap();
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
            explicit.device_info().validate().unwrap();
        }
        assert_eq!(
            OpenVinoAccelerator::with_device("no-such-device").unwrap_err(),
            InitError::DeviceUnavailable
        );
    }

    /// Test-local declarations for the model-compilation surface that production code adopts in
    /// the lowering and submission milestones.
    mod pin_ffi {
        use core::ffi::{c_char, c_void};

        use crate::ffi::{ov_core_t, ov_status_e};

        unsafe extern "C" {
            pub(super) fn ov_core_read_model_from_memory_buffer(
                core: *const ov_core_t,
                model_str: *const c_char,
                str_len: usize,
                weights: *const c_void,
                model: *mut *mut c_void,
            ) -> ov_status_e;
            pub(super) fn ov_core_compile_model(
                core: *const ov_core_t,
                model: *const c_void,
                device_name: *const c_char,
                property_args_size: usize,
                compiled_model: *mut *mut c_void,
                ...
            ) -> ov_status_e;
            pub(super) fn ov_model_free(model: *mut c_void);
            pub(super) fn ov_compiled_model_free(compiled_model: *mut c_void);
            pub(super) fn ov_compiled_model_inputs_size(
                compiled_model: *const c_void,
                size: *mut usize,
            ) -> ov_status_e;
            pub(super) fn ov_compiled_model_outputs_size(
                compiled_model: *const c_void,
                size: *mut usize,
            ) -> ov_status_e;
            pub(super) static ov_property_key_hint_execution_mode: *const c_char;
        }
    }

    /// Minimal hand-written IR v11 document: one f32 tensor forwarded from parameter to result.
    const IDENTITY_IR: &str = r#"<?xml version="1.0"?>
<net name="pin" version="11">
    <layers>
        <layer id="0" name="input_0" type="Parameter" version="opset1">
            <data shape="8" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>8</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
"#;

    /// Pins the runtime facts the lowering design depends on: in-memory IR v11 acceptance with a
    /// null weights tensor, a direct parameter-to-result edge, the variadic property convention
    /// of `ov_core_compile_model` (argument count, not pair count), and index-stable I/O sizes.
    #[test]
    fn runtime_accepts_ir_v11_with_a_direct_parameter_result_edge() {
        let Some(backend) = backend() else { return };
        let mut model = ptr::null_mut();
        // SAFETY: the core is live, the IR bytes are valid for the call, and a null weights
        // tensor declares a constant-free model.
        let status = unsafe {
            pin_ffi::ov_core_read_model_from_memory_buffer(
                backend.core.as_const_ptr(),
                IDENTITY_IR.as_ptr().cast::<c_char>(),
                IDENTITY_IR.len(),
                ptr::null(),
                &mut model,
            )
        };
        assert_eq!(status, ffi::OV_STATUS_OK, "IR v11 model was rejected");
        assert!(!model.is_null());

        let mut compiled = ptr::null_mut();
        // SAFETY: core/model/device are live; `property_args_size` counts the variadic
        // arguments (one key/value pair = 2 arguments) per the documented convention; the key
        // symbol and value stay valid for the duration of the call.
        let status = unsafe {
            pin_ffi::ov_core_compile_model(
                backend.core.as_const_ptr(),
                model,
                backend.device.as_ptr(),
                2,
                &mut compiled,
                pin_ffi::ov_property_key_hint_execution_mode,
                c"ACCURACY".as_ptr(),
            )
        };
        assert_eq!(
            status,
            ffi::OV_STATUS_OK,
            "compilation with the ACCURACY execution-mode hint failed"
        );
        assert!(!compiled.is_null());

        let (mut inputs, mut outputs) = (0usize, 0usize);
        // SAFETY: the compiled model is live and the out-pointers are valid.
        unsafe {
            assert_eq!(
                pin_ffi::ov_compiled_model_inputs_size(compiled, &mut inputs),
                ffi::OV_STATUS_OK
            );
            assert_eq!(
                pin_ffi::ov_compiled_model_outputs_size(compiled, &mut outputs),
                ffi::OV_STATUS_OK
            );
        }
        assert_eq!((inputs, outputs), (1, 1));

        // SAFETY: both handles are live, owned here, and freed exactly once.
        unsafe {
            pin_ffi::ov_compiled_model_free(compiled);
            pin_ffi::ov_model_free(model);
        }
    }
}

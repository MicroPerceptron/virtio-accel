//! Hand-written declarations for the OpenVINO C API (`libopenvino_c`).
//!
//! This module is the crate's only foreign ABI boundary. Every declaration mirrors the stable C
//! headers shipped with OpenVINO (`openvino/c/*.h`) and is audited in `SAFETY.md`. It contains
//! types and a process-lifetime symbol table; handle ownership and calls live in `native.rs`.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_uint, c_void};
use core::marker::{PhantomData, PhantomPinned};

/// Status codes returned by every fallible OpenVINO C API function.
///
/// Mirrors `ov_status_e` in `openvino/c/ov_common.h`; negative values map C++ exceptions.
pub(crate) type ov_status_e = core::ffi::c_int;

pub(crate) const OV_STATUS_OK: ov_status_e = 0;
pub(crate) const OV_STATUS_NOT_IMPLEMENTED: ov_status_e = -2;
pub(crate) const OV_STATUS_PARAMETER_MISMATCH: ov_status_e = -4;
pub(crate) const OV_STATUS_NOT_FOUND: ov_status_e = -5;
pub(crate) const OV_STATUS_OUT_OF_BOUNDS: ov_status_e = -6;
pub(crate) const OV_STATUS_REQUEST_BUSY: ov_status_e = -8;
pub(crate) const OV_STATUS_RESULT_NOT_READY: ov_status_e = -9;
pub(crate) const OV_STATUS_NOT_ALLOCATED: ov_status_e = -10;
pub(crate) const OV_STATUS_INFER_CANCELLED: ov_status_e = -13;
pub(crate) const OV_STATUS_INVALID_C_PARAM: ov_status_e = -14;
pub(crate) const OV_STATUS_NOT_IMPLEMENT_C_METHOD: ov_status_e = -16;

/// Element type codes, aligned with `ov_element_type_e` in `openvino/c/ov_common.h`.
pub(crate) type ov_element_type_e = c_uint;

pub(crate) const ELEMENT_BOOLEAN: ov_element_type_e = 1;
pub(crate) const ELEMENT_F16: ov_element_type_e = 3;
pub(crate) const ELEMENT_F32: ov_element_type_e = 4;
pub(crate) const ELEMENT_I8: ov_element_type_e = 7;
pub(crate) const ELEMENT_I32: ov_element_type_e = 9;
pub(crate) const ELEMENT_I64: ov_element_type_e = 10;
pub(crate) const ELEMENT_U8: ov_element_type_e = 16;
pub(crate) const ELEMENT_F8E4M3: ov_element_type_e = 21;
/// `f8e5m2`. Upstream spells the enumerator `F8E5M3` -- a typo in `ov_common.h`, whose own
/// doc comment on the same line reads "f8e5m2 element type". The code, 22, is what matters.
pub(crate) const ELEMENT_F8E5M2: ov_element_type_e = 22;

macro_rules! opaque_handle {
    ($(#[$doc:meta] $name:ident),* $(,)?) => {
        $(
            #[$doc]
            #[repr(C)]
            pub(crate) struct $name {
                _unconstructable: [u8; 0],
                _not_send_sync: PhantomData<(*mut u8, PhantomPinned)>,
            }
        )*
    };
}

opaque_handle! {
    /// Opaque OpenVINO core handle (`ov_core_t`).
    ov_core_t,
    /// Opaque source model handle (`ov_model_t`).
    ov_model_t,
    /// Opaque compiled model handle (`ov_compiled_model_t`).
    ov_compiled_model_t,
    /// Remote context for device tensor import.
    ov_remote_context_t,
    /// Opaque inference request handle (`ov_infer_request_t`).
    ov_infer_request_t,
    /// Opaque tensor handle (`ov_tensor_t`).
    ov_tensor_t,
}

/// OpenVINO retains a pointer to this record; it must stay at a stable address.
#[repr(C)]
pub(crate) struct ov_callback_t {
    pub callback_func: unsafe extern "C" fn(*mut c_void),
    pub args: *mut c_void,
}

/// Tensor shape (`ov_shape_t` in `openvino/c/ov_shape.h`); passed by value where the C API does.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ov_shape_t {
    pub rank: i64,
    pub dims: *mut i64,
}

/// Device enumeration result (`ov_available_devices_t` in `openvino/c/ov_core.h`).
#[repr(C)]
pub(crate) struct ov_available_devices_t {
    pub devices: *mut *mut c_char,
    pub size: usize,
}

/// `ov_version_t`: two runtime-owned C strings released by `ov_version_free`.
#[repr(C)]
pub(crate) struct ov_version_t {
    pub build_number: *const c_char,
    pub description: *const c_char,
}

// Keep one typed declaration list for linked and runtime-loaded builds. The latter
// resolves every symbol before publishing the table, and retains the library for
// the process lifetime so handles and asynchronous work never outlive its code.
macro_rules! native_api {
    ($( $(#[$attr:meta])* pub(crate) fn $name:ident($($args:tt)*) $(-> $ret:ty)?; )*
     optional_remote { $(pub(crate) fn $remote:ident($($params:tt)*) $(-> $result:ty)?;)* }) => {
        #[cfg(not(feature = "runtime-loading"))]
        unsafe extern "C" {
            $( $(#[$attr])* pub(crate) fn $name($($args)*) $(-> $ret)?; )*
            $(pub(crate) fn $remote($($params)*) $(-> $result)?;)*
        }

        pub(crate) struct RemoteApi {
            $(pub(crate) $remote: unsafe extern "C" fn($($params)*) $(-> $result)?,)*
        }
        pub(crate) struct Api {
            pub(crate) remote: Option<RemoteApi>,
            $(pub(crate) $name: unsafe extern "C" fn($($args)*) $(-> $ret)?,)*
            #[cfg(feature = "runtime-loading")]
            _library: libloading::Library,
            #[cfg(feature = "runtime-loading")]
            execution_mode: std::ffi::CString,
        }

        #[cfg(not(feature = "runtime-loading"))]
        pub(crate) fn api() -> &'static Api {
            static API: Api = Api { $($name,)* remote: Some(RemoteApi { $($remote,)* }), };
            &API
        }

        #[cfg(feature = "runtime-loading")]
        static API: std::sync::OnceLock<Api> = std::sync::OnceLock::new();

        #[cfg(feature = "runtime-loading")]
        pub(crate) fn api() -> &'static Api {
            API.get().expect("native runtime must be initialized before use")
        }

        #[cfg(feature = "runtime-loading")]
        fn load(path: &std::path::Path) -> Result<Api, Box<dyn std::error::Error>> {
            // SAFETY: loading executes the selected vendor library's initialization.
            // Candidates are an explicitly configured runtime or the OS loader's
            // search name. The ABI is the same audited C ABI as the linked build.
            #[cfg(not(target_os = "windows"))]
            let library = unsafe { libloading::Library::new(path) }?;
            #[cfg(target_os = "windows")]
            let library: libloading::Library = if path.components().count() > 1 {
                let absolute = std::path::absolute(path)?;
                // SAFETY: use the selected DLL's directory for its dependencies,
                // without changing the process-wide DLL search configuration.
                unsafe { libloading::os::windows::Library::load_with_flags(
                    absolute, libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
                ) }?.into()
            } else {
                // SAFETY: an unqualified name intentionally uses the OS search path.
                unsafe { libloading::Library::new(path) }?
            };
            // SAFETY: this exported variable has type `const char *`. Symbol
            // lookup returns its address; the pointer value refers to an immutable
            // NUL-terminated property key. Copy it before retaining the library.
            let execution_mode = unsafe {
                let symbol = library.get::<*const *const c_char>(b"ov_property_key_hint_execution_mode\0")?;
                let key = **symbol;
                if key.is_null() { return Err("null OpenVINO property key".into()); }
                std::ffi::CStr::from_ptr(key).to_owned()
            };
            // SAFETY: identical C ABI and lifetime to the required table. Older runtimes
            // may omit the remote API and still support ordinary host tensors.
            let remote = (|| Some(RemoteApi {
                $($remote: unsafe { *library.get(concat!(stringify!($remote), "\0").as_bytes()).ok()? },)*
            }))();
            Ok(Api {
                remote,
                execution_mode,
                // SAFETY: signatures are transcribed from the pinned C headers.
                // All required symbols must resolve; the retained library outlives
                // every copied function pointer and all native resources.
                $($name: unsafe { *library.get::<unsafe extern "C" fn($($args)*) $(-> $ret)?>(
                    concat!(stringify!($name), "\0").as_bytes()
                )? },)*
                _library: library,
            })
        }
    };
}

native_api! {

    // Core lifecycle and discovery.
    pub(crate) fn ov_get_openvino_version(version: *mut ov_version_t) -> ov_status_e;
    pub(crate) fn ov_version_free(version: *mut ov_version_t);
    pub(crate) fn ov_core_create(core: *mut *mut ov_core_t) -> ov_status_e;
    pub(crate) fn ov_core_free(core: *mut ov_core_t);
    pub(crate) fn ov_core_get_available_devices(
        core: *const ov_core_t,
        devices: *mut ov_available_devices_t,
    ) -> ov_status_e;
    pub(crate) fn ov_available_devices_free(devices: *mut ov_available_devices_t);

    // Model reading and compilation.
    /// Serialize a compiled model (the plugin's device-specific form) to a file.
    pub(crate) fn ov_compiled_model_export_model(
        compiled_model: *const ov_compiled_model_t,
        export_model_path: *const c_char,
    ) -> ov_status_e;
    pub(crate) fn ov_core_read_model_from_memory_buffer(
        core: *const ov_core_t,
        model_str: *const c_char,
        str_len: usize,
        weights: *const ov_tensor_t,
        model: *mut *mut ov_model_t,
    ) -> ov_status_e;
    /// C-variadic: `property_args_size` counts the variadic arguments, two per key/value pair.
    pub(crate) fn ov_core_compile_model(
        core: *const ov_core_t,
        model: *const ov_model_t,
        device_name: *const c_char,
        property_args_size: usize,
        compiled_model: *mut *mut ov_compiled_model_t,
        ...
    ) -> ov_status_e;
    pub(crate) fn ov_model_free(model: *mut ov_model_t);

    // Compiled models.
    pub(crate) fn ov_compiled_model_create_infer_request(
        compiled_model: *const ov_compiled_model_t,
        infer_request: *mut *mut ov_infer_request_t,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_inputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_outputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_free(compiled_model: *mut ov_compiled_model_t);

    // Shapes and tensors.
    pub(crate) fn ov_shape_create(
        rank: i64,
        dims: *const i64,
        shape: *mut ov_shape_t,
    ) -> ov_status_e;
    pub(crate) fn ov_shape_free(shape: *mut ov_shape_t) -> ov_status_e;
    pub(crate) fn ov_tensor_create_from_host_ptr(
        ty: ov_element_type_e,
        shape: ov_shape_t,
        host_ptr: *mut c_void,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_tensor_data(tensor: *const ov_tensor_t, data: *mut *mut c_void)
    -> ov_status_e;
    pub(crate) fn ov_tensor_free(tensor: *mut ov_tensor_t);

    // Inference requests.
    pub(crate) fn ov_infer_request_set_input_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        idx: usize,
        tensor: *const ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_set_output_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        idx: usize,
        tensor: *const ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_get_output_tensor_by_index(
        infer_request: *const ov_infer_request_t,
        idx: usize,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_set_callback(
        infer_request: *mut ov_infer_request_t,
        callback: *const ov_callback_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_start_async(
        infer_request: *mut ov_infer_request_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_wait(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub(crate) fn ov_infer_request_wait_for(
        infer_request: *mut ov_infer_request_t,
        timeout_ms: i64,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_cancel(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub(crate) fn ov_infer_request_free(infer_request: *mut ov_infer_request_t);

    optional_remote {
        pub(crate) fn ov_compiled_model_get_context(
            compiled: *const ov_compiled_model_t, context: *mut *mut ov_remote_context_t,
        ) -> ov_status_e;
        pub(crate) fn ov_remote_context_create_tensor(
            context: *const ov_remote_context_t, ty: ov_element_type_e, shape: ov_shape_t,
            property_args_size: usize, tensor: *mut *mut ov_tensor_t, ...
        ) -> ov_status_e;
        pub(crate) fn ov_remote_context_free(context: *mut ov_remote_context_t);
    }
}

/// Resolve the complete ABI before any native initialization. Failed loads are not
/// cached: callers may configure a runtime and retry. A successful load is immutable.
pub(crate) fn ensure_loaded() -> Result<(), crate::InitError> {
    #[cfg(feature = "runtime-loading")]
    if API.get().is_none() {
        for path in library_candidates() {
            if load_from(&path).is_ok() {
                return Ok(());
            }
        }
        return Err(crate::InitError::RuntimeUnavailable);
    }
    Ok(())
}

#[cfg(feature = "runtime-loading")]
pub(crate) fn load_from(path: &std::path::Path) -> Result<(), crate::InitError> {
    if API.get().is_none() {
        let api = load(path).map_err(|_| crate::InitError::RuntimeUnavailable)?;
        // A concurrent successful initialization wins; dropping this unused
        // library reference cannot unload the retained table's library.
        let _ = API.set(api);
    }
    Ok(())
}

#[cfg(feature = "runtime-loading")]
fn library_candidates() -> Vec<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("VIRTIO_ACCEL_OPENVINO_LIBRARY") {
        return vec![path.into()];
    }
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["openvino_c.dll"]
    } else if cfg!(target_os = "macos") {
        &["libopenvino_c.dylib", "libopenvino_c.2630.dylib"]
    } else {
        &["libopenvino_c.so", "libopenvino_c.so.2630"]
    };
    let directory = std::env::var_os("VIRTIO_ACCEL_OPENVINO_LIB_DIR");
    names
        .iter()
        .map(|name| match &directory {
            Some(directory) => std::path::Path::new(directory).join(name),
            None => (*name).into(),
        })
        .collect()
}

#[cfg(not(feature = "runtime-loading"))]
unsafe extern "C" {
    static ov_property_key_hint_execution_mode: *const c_char;
}

pub(crate) fn execution_mode_property() -> *const c_char {
    #[cfg(feature = "runtime-loading")]
    {
        api().execution_mode.as_ptr()
    }
    #[cfg(not(feature = "runtime-loading"))]
    // SAFETY: the linked runtime owns this immutable property string for the process lifetime.
    unsafe {
        ov_property_key_hint_execution_mode
    }
}

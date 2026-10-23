//! Thread-affine execution. Only owned byte vectors cross the worker boundary.
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    Capabilities, ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure,
    Timeout,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub kind: String,
    pub provider: String,
    pub name: String,
    pub cpu_participation: bool,
}

#[derive(Default, Serialize)]
pub struct Discovery {
    pub devices: Vec<Device>,
    pub diagnostics: Vec<String>,
}

pub fn discover() -> Discovery {
    let mut result = Discovery::default();
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    match virtio_accel_coreml::CoreMlAccelerator::new_tosa() {
        Ok(_) => result.devices.push(Device {
            id: "npu:0".into(),
            kind: "npu".into(),
            provider: "coreml".into(),
            name: "Apple Neural Engine (Core ML CPUAndNeuralEngine)".into(),
            cpu_participation: true,
        }),
        Err(error) => result.diagnostics.push(format!("Core ML: {error}")),
    }
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    match virtio_accel_vulkan::VulkanAccelerator::available_devices() {
        Ok(mut names) => {
            names.sort();
            for name in &names {
                // The current Rust API selects by name. Refuse ambiguous names instead of
                // accidentally opening the first of two identically named physical devices.
                if names.iter().filter(|n| *n == name).count() != 1 {
                    result
                        .diagnostics
                        .push(format!("Vulkan: ambiguous device name {name:?}"));
                    continue;
                }
                match virtio_accel_vulkan::VulkanAccelerator::with_device(name) {
                    Ok(backend) => match backend.device_info() {
                        Ok(info)
                            if info.identity.class == virtio_accel_core::AcceleratorClass::GPU =>
                        {
                            let index = result.devices.iter().filter(|d| d.kind == "gpu").count();
                            result.devices.push(Device {
                                id: format!("gpu:{index}"),
                                kind: "gpu".into(),
                                provider: "vulkan".into(),
                                name: name.clone(),
                                cpu_participation: false,
                            });
                        }
                        Ok(_) => result
                            .diagnostics
                            .push(format!("Vulkan: excluded non-GPU {name:?}")),
                        Err(error) => result
                            .diagnostics
                            .push(format!("Vulkan {name:?}: {error:?}")),
                    },
                    Err(error) => result.diagnostics.push(format!("Vulkan {name:?}: {error}")),
                }
            }
        }
        Err(error) => result.diagnostics.push(format!("Vulkan: {error}")),
    }
    result
}

fn error(stage: &str, error: impl std::fmt::Debug) -> String {
    format!("{stage}: {error:?}")
}

fn backend_error(
    stage: &str,
    e: BackendError,
    poisoned: &mut bool,
    quarantined: &mut bool,
) -> String {
    if e == BackendError::DeviceLost {
        *poisoned = true;
        *quarantined = true;
    }
    error(stage, e)
}

struct Session<A: Accelerator> {
    backend: A,
    context: Option<A::Context>,
    program: Option<A::Program>,
    queue: Option<A::Queue>,
    buffers: Vec<A::Buffer>,
    event: Option<A::Event>,
    sizes: Vec<usize>,
    inputs: usize,
    poisoned: bool,
    quarantined: bool,
}

impl<A: Accelerator + virtio_accel_tosa::TosaCapabilityProvider> Session<A> {
    fn new(backend: A, json: &str) -> Result<Self, String> {
        let mut session = Self {
            backend,
            context: None,
            program: None,
            queue: None,
            buffers: Vec::new(),
            event: None,
            sizes: Vec::new(),
            inputs: 0,
            poisoned: false,
            quarantined: false,
        };
        if let Err(error) = session.initialize(json) {
            let cleanup = session.close();
            return Err(format!(
                "{error}{}",
                cleanup.err().map(|e| format!("; {e}")).unwrap_or_default()
            ));
        }
        Ok(session)
    }

    fn initialize(&mut self, json: &str) -> Result<(), String> {
        let bytes = super::compile_json(json)?;
        let doc: super::Document = serde_json::from_str(json).map_err(|e| e.to_string())?;
        self.inputs = doc.inputs.len();
        self.sizes = doc
            .inputs
            .iter()
            .chain(&doc.outputs)
            .map(|i| {
                let n = &doc.nodes[*i];
                n.shape
                    .iter()
                    .fold(n.dtype.bytes(), |size, d| size * *d as usize)
            })
            .collect();
        let info = self.backend.device_info().map_err(|e| {
            backend_error("device info", e, &mut self.poisoned, &mut self.quarantined)
        })?;
        if self.sizes.len() > info.limits.max_bindings_per_submission as usize {
            return Err("graph exceeds device binding limit".into());
        }
        let domain = if info.capabilities.contains(Capabilities::SHARED_MEMORY) {
            MemoryDomain::Shared
        } else {
            MemoryDomain::Host
        };
        self.context = Some(
            self.backend
                .create_context(ContextDesc::default())
                .map_err(|e| {
                    backend_error(
                        "create context",
                        e,
                        &mut self.poisoned,
                        &mut self.quarantined,
                    )
                })?,
        );
        let context = self.context.as_ref().unwrap();
        let model = virtio_accel_tosa::parse(&bytes).map_err(|e| error("parse", e))?;
        let target = virtio_accel_tosa::Target::new(
            virtio_accel_tosa::Version::TOSA_1_0,
            virtio_accel_tosa::ProfileSet::FLOATING_POINT,
            virtio_accel_tosa::Level::Level8K,
            virtio_accel_tosa::ExtensionSet::NONE,
        );
        let analysis = model.analyze_for(target).map_err(|e| error("analyze", e))?;
        let tier = self
            .backend
            .tosa_capabilities()
            .iter()
            .find(|c| c.target == target)
            .ok_or("device does not advertise the requested TOSA target")?;
        for operator in analysis.operators() {
            if !tier.supports_operator(operator.op()) {
                return Err(format!(
                    "device advertised tier does not support {:?}; no fallback was selected",
                    operator.op()
                ));
            }
        }
        self.program = Some(
            self.backend
                .load_program(
                    context,
                    model
                        .artifact_ref(target, u64::MAX)
                        .map_err(|e| error("artifact", e))?,
                )
                .map_err(|e| {
                    backend_error(
                        "load program (graph may be outside the device's advertised tier)",
                        e,
                        &mut self.poisoned,
                        &mut self.quarantined,
                    )
                })?,
        );
        self.queue = Some(
            self.backend
                .create_queue(context, QueueDesc::default())
                .map_err(|e| {
                    backend_error("create queue", e, &mut self.poisoned, &mut self.quarantined)
                })?,
        );
        for (index, size) in self.sizes.iter().enumerate() {
            let usage = if index < self.inputs {
                BufferUsage::PROGRAM_INPUT | BufferUsage::TRANSFER_DESTINATION
            } else {
                BufferUsage::PROGRAM_OUTPUT | BufferUsage::TRANSFER_SOURCE
            };
            let desc = BufferDesc::new(*size as u64, 16 * 1024, domain, usage)
                .map_err(|e| error("buffer description", e))?;
            let (buffer, _) = self
                .backend
                .allocate_buffer(context, desc)
                .map_err(|e| {
                    backend_error(
                        "allocate buffer",
                        e,
                        &mut self.poisoned,
                        &mut self.quarantined,
                    )
                })?
                .into_parts();
            self.buffers.push(buffer);
        }
        Ok(())
    }

    fn run(&mut self, inputs: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, String> {
        if self.poisoned {
            return Err("executable is poisoned; close it and create a new one".into());
        }
        if inputs.len() != self.inputs
            || inputs
                .iter()
                .zip(&self.sizes)
                .any(|(a, size)| a.len() != *size)
        {
            return Err("input byte lengths do not match compiled graph".into());
        }
        for (buffer, data) in self.buffers.iter_mut().zip(&inputs) {
            self.backend
                .write_buffer(
                    buffer,
                    0,
                    &virtio_accel_core::TransportByteSource::new(data.as_slice()),
                )
                .map_err(|e| {
                    backend_error("write input", e, &mut self.poisoned, &mut self.quarantined)
                })?;
        }
        let bindings: Vec<_> = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindingRef {
                slot: i as u32,
                buffer,
                range: BufferRange::new(0, self.sizes[i] as u64).unwrap(),
                access: if i < self.inputs {
                    AccessMode::Read
                } else {
                    AccessMode::Write
                },
            })
            .collect();
        let admission_error = match self.backend.submit(
            self.queue.as_ref().unwrap(),
            self.program.as_ref().unwrap(),
            &bindings,
            Timeout::Infinite,
        ) {
            Ok(event) => {
                self.event = Some(event);
                None
            }
            Err(SubmitFailure::Rejected(e)) => {
                self.poisoned = e == BackendError::DeviceLost;
                self.quarantined = self.poisoned;
                return Err(error("submission rejected", e));
            }
            Err(SubmitFailure::Indeterminate { error: e, event }) => {
                self.event = Some(event);
                self.poisoned = true;
                Some(error(
                    "submission acceptance indeterminate (not retried)",
                    e,
                ))
            }
        };
        let completion = loop {
            match self.backend.poll_event(self.event.as_ref().unwrap()) {
                Ok(EventState::Pending) => std::thread::sleep(Duration::from_millis(1)),
                Ok(EventState::Complete) => break Ok(()),
                Ok(EventState::Failed(e)) => {
                    self.poisoned = true;
                    break Err(error("execution failed", e));
                }
                Ok(EventState::Cancelled) => {
                    self.poisoned = true;
                    break Err("execution cancelled".into());
                }
                Err(e) => {
                    // A poll error is not proof of termination. close() quarantines the
                    // event and all dependencies rather than releasing live resources.
                    self.poisoned = true;
                    self.quarantined = true;
                    return Err(error("completion unknown; resources quarantined", e));
                }
            }
        };
        if let Err(failure) = self.backend.destroy_event(self.event.take().unwrap()) {
            self.poisoned = true;
            self.quarantined = true;
            let message = error("release event; resources quarantined", failure.error());
            if let ReleaseFailure::Rejected { resource, .. } = failure {
                self.event = Some(resource);
            }
            return Err(message);
        }
        if let Some(error) = admission_error {
            return Err(error);
        }
        completion?;
        let mut outputs = Vec::new();
        for (buffer, size) in self.buffers[self.inputs..]
            .iter()
            .zip(&self.sizes[self.inputs..])
        {
            let mut output = vec![0; *size];
            self.backend
                .read_buffer(
                    buffer,
                    0,
                    &mut virtio_accel_core::TransportByteSink::new(output.as_mut_slice()),
                )
                .map_err(|e| {
                    backend_error("read output", e, &mut self.poisoned, &mut self.quarantined)
                })?;
            outputs.push(output);
        }
        Ok(outputs)
    }

    fn close(mut self) -> Result<(), String> {
        if self.event.is_some() || self.quarantined {
            // No reset API exists at this layer. Conservatively retain the entire owner
            // on ambiguous completion/release/device loss, including backend parents.
            std::mem::forget(self);
            return Err("native resources quarantined after uncertain completion or device loss; process restart may be needed".into());
        }
        macro_rules! release {
            ($value:expr, $method:ident) => {
                if let Some(resource) = $value {
                    if let Err(failure) = self.backend.$method(resource) {
                        let message = error(stringify!($method), failure.error());
                        if let ReleaseFailure::Rejected { resource, .. } = failure {
                            std::mem::forget(resource);
                        }
                        std::mem::forget(self);
                        return Err(format!("{message}; remaining resources quarantined"));
                    }
                }
            };
        }
        release!(self.queue.take(), destroy_queue);
        release!(self.program.take(), unload_program);
        while let Some(buffer) = self.buffers.pop() {
            release!(Some(buffer), free_buffer);
        }
        release!(self.context.take(), destroy_context);
        Ok(())
    }
}

type Response = Sender<Result<Vec<Vec<u8>>, String>>;
enum Job {
    Run(Vec<Vec<u8>>, Response),
    Close(Sender<Result<(), String>>),
}

fn serve<A: Accelerator + virtio_accel_tosa::TosaCapabilityProvider>(
    backend: A,
    json: String,
    receiver: Receiver<Job>,
    ready: Sender<Result<(), String>>,
) {
    let mut session = match Session::new(backend, &json) {
        Ok(session) => session,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    if ready.send(Ok(())).is_ok() {
        while let Ok(job) = receiver.recv() {
            match job {
                Job::Run(inputs, response) => {
                    let _ = response.send(session.run(inputs));
                }
                Job::Close(response) => {
                    let _ = response.send(session.close());
                    return;
                }
            }
        }
    }
    if let Err(error) = session.close() {
        eprintln!("virtio-accel: {error}");
    }
}

pub struct Worker {
    sender: Option<Sender<Job>>,
}
impl Worker {
    pub fn new(device: Device, json: String, timeout: Duration) -> Result<Self, String> {
        let (sender, receiver) = mpsc::channel();
        let (ready, result) = mpsc::channel();
        std::thread::Builder::new()
            .name("virtio-accel-executor".into())
            .spawn(move || {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                if device.provider == "coreml" && device.kind == "npu" && device.id == "npu:0" {
                    match virtio_accel_coreml::CoreMlAccelerator::new_tosa() {
                        Ok(backend) => serve(backend, json, receiver, ready),
                        Err(e) => {
                            let _ = ready.send(Err(error("Core ML initialization", e)));
                        }
                    }
                    return;
                }
                #[cfg(all(
                    feature = "vulkan",
                    any(target_os = "macos", target_os = "linux", target_os = "windows")
                ))]
                if device.provider == "vulkan" && device.kind == "gpu" {
                    match virtio_accel_vulkan::VulkanAccelerator::with_device(&device.name) {
                        Ok(backend) => {
                            if backend.device_info().is_ok_and(|i| {
                                i.identity.class == virtio_accel_core::AcceleratorClass::GPU
                            }) {
                                serve(backend, json, receiver, ready);
                            } else {
                                let _ =
                                    ready.send(Err("selected Vulkan endpoint is not a GPU".into()));
                            }
                        }
                        Err(e) => {
                            let _ = ready.send(Err(error("Vulkan initialization", e)));
                        }
                    }
                    return;
                }
                let _ = ready.send(Err(format!(
                    "unavailable device provider: {}",
                    device.provider
                )));
            })
            .map_err(|e| error("start executor", e))?;
        result.recv_timeout(timeout).map_err(|e| {
            error(
                "compile wait failed; worker retains ownership until compilation finishes",
                e,
            )
        })??;
        Ok(Self {
            sender: Some(sender),
        })
    }

    pub fn run(&mut self, inputs: Vec<Vec<u8>>, timeout: Duration) -> Result<Vec<Vec<u8>>, String> {
        let sender = self
            .sender
            .as_ref()
            .ok_or("executable is closed or timed out")?;
        let (response, receiver) = mpsc::channel();
        sender
            .send(Job::Run(inputs, response))
            .map_err(|e| error("executor unavailable", e))?;
        match receiver.recv_timeout(timeout) {
            Ok(result) => result,
            Err(e) => {
                // Disconnect rather than retry or cancel. Worker retains every handle until
                // completion and then closes; a hung device can retain one worker indefinitely.
                self.sender.take();
                Err(error(
                    "execution wait failed; executable closed, native work may still be running",
                    e,
                ))
            }
        }
    }

    pub fn close(&mut self, timeout: Duration) -> Result<(), String> {
        if let Some(sender) = self.sender.take() {
            let (response, receiver) = mpsc::channel();
            sender
                .send(Job::Close(response))
                .map_err(|e| error("executor unavailable", e))?;
            receiver
                .recv_timeout(timeout)
                .map_err(|e| error("close wait failed; worker still owns resources", e))?
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;
    use std::sync::Mutex;

    fn duration(seconds: f64) -> PyResult<Duration> {
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 86400.0 {
            return Err(PyValueError::new_err(
                "timeout must be finite and in (0, 86400] seconds",
            ));
        }
        Ok(Duration::from_secs_f64(seconds))
    }

    #[pyfunction]
    fn discover_devices(py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&discover()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyclass]
    struct NativeExecutable {
        worker: Mutex<Worker>,
    }

    #[pymethods]
    impl NativeExecutable {
        #[new]
        fn new(
            py: Python<'_>,
            device_json: String,
            graph_json: String,
            timeout: f64,
        ) -> PyResult<Self> {
            let device: Device = serde_json::from_str(&device_json)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let timeout = duration(timeout)?;
            let worker = py
                .detach(move || Worker::new(device, graph_json, timeout))
                .map_err(PyRuntimeError::new_err)?;
            Ok(Self {
                worker: Mutex::new(worker),
            })
        }

        fn run<'py>(
            &self,
            py: Python<'py>,
            inputs: Vec<Vec<u8>>,
            timeout: f64,
        ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
            let timeout = duration(timeout)?;
            let outputs = py
                .detach(|| {
                    self.worker
                        .lock()
                        .map_err(|_| "executor lock poisoned".to_string())?
                        .run(inputs, timeout)
                })
                .map_err(PyRuntimeError::new_err)?;
            Ok(outputs.iter().map(|b| PyBytes::new(py, b)).collect())
        }

        fn close(&self, py: Python<'_>, timeout: f64) -> PyResult<()> {
            let timeout = duration(timeout)?;
            py.detach(|| {
                self.worker
                    .lock()
                    .map_err(|_| "executor lock poisoned".to_string())?
                    .close(timeout)
            })
            .map_err(PyRuntimeError::new_err)
        }
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(discover_devices, module)?)?;
        module.add_class::<NativeExecutable>()?;
        Ok(())
    }
}
#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use virtio_accel_core::{AllocatedBuffer, ArtifactRef, ByteSink, ByteSource, DeviceInfo};

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Complete,
        Rejected,
        Indeterminate,
        PollError,
        ReleaseRejected,
        ReleaseUnknown,
        Failed,
        Pending,
        WriteLost,
    }
    struct State {
        mode: Mode,
        log: Vec<&'static str>,
    }
    struct Fake(Arc<Mutex<State>>);
    impl Fake {
        fn record(&self, message: &'static str) {
            self.0.lock().unwrap().log.push(message);
        }
        fn mode(&self) -> Mode {
            self.0.lock().unwrap().mode
        }
    }
    impl virtio_accel_tosa::TosaCapabilityProvider for Fake {
        fn tosa_capabilities(&self) -> &'static [virtio_accel_tosa::CapabilityDescriptor] {
            &[]
        }
    }
    impl Accelerator for Fake {
        type Context = ();
        type Program = ();
        type Queue = ();
        type Buffer = Vec<u8>;
        type Event = ();
        fn device_info(&self) -> Result<DeviceInfo, BackendError> {
            Err(BackendError::Unsupported)
        }
        fn create_context(&self, _: ContextDesc) -> Result<(), BackendError> {
            Ok(())
        }
        fn destroy_context(&self, _: ()) -> Result<(), ReleaseFailure<()>> {
            self.record("context");
            Ok(())
        }
        fn allocate_buffer(
            &self,
            _: &(),
            _: BufferDesc,
        ) -> Result<AllocatedBuffer<Vec<u8>>, BackendError> {
            Err(BackendError::Unsupported)
        }
        fn write_buffer(
            &self,
            buffer: &mut Vec<u8>,
            _: u64,
            source: &dyn ByteSource,
        ) -> Result<(), BackendError> {
            self.record("write");
            if self.mode() == Mode::WriteLost {
                return Err(BackendError::DeviceLost);
            }
            source.read_at(0, buffer)
        }
        fn read_buffer(
            &self,
            buffer: &Vec<u8>,
            _: u64,
            sink: &mut dyn ByteSink,
        ) -> Result<(), BackendError> {
            self.record("read");
            sink.write_at(0, buffer)
        }
        fn free_buffer(&self, _: Vec<u8>) -> Result<(), ReleaseFailure<Vec<u8>>> {
            self.record("buffer");
            Ok(())
        }
        fn load_program(&self, _: &(), _: ArtifactRef<'_>) -> Result<(), BackendError> {
            Ok(())
        }
        fn unload_program(&self, _: ()) -> Result<(), ReleaseFailure<()>> {
            self.record("program");
            Ok(())
        }
        fn create_queue(&self, _: &(), _: QueueDesc) -> Result<(), BackendError> {
            Ok(())
        }
        fn destroy_queue(&self, _: ()) -> Result<(), ReleaseFailure<()>> {
            self.record("queue");
            Ok(())
        }
        fn submit(
            &self,
            _: &(),
            _: &(),
            _: &[BindingRef<'_, Vec<u8>>],
            _: Timeout,
        ) -> Result<(), SubmitFailure<()>> {
            self.record("submit");
            match self.mode() {
                Mode::Rejected => Err(SubmitFailure::Rejected(BackendError::Busy)),
                Mode::Indeterminate => Err(SubmitFailure::Indeterminate {
                    error: BackendError::Incompatible,
                    event: (),
                }),
                _ => Ok(()),
            }
        }
        fn poll_event(&self, _: &()) -> Result<EventState, BackendError> {
            match self.mode() {
                Mode::Pending => Ok(EventState::Pending),
                Mode::PollError => Err(BackendError::DeviceLost),
                Mode::Failed => Ok(EventState::Failed(BackendError::Incompatible)),
                _ => Ok(EventState::Complete),
            }
        }
        fn destroy_event(&self, _: ()) -> Result<(), ReleaseFailure<()>> {
            self.record("event");
            match self.mode() {
                Mode::ReleaseRejected => Err(ReleaseFailure::Rejected {
                    error: BackendError::Busy,
                    resource: (),
                }),
                Mode::ReleaseUnknown => Err(ReleaseFailure::Indeterminate {
                    error: BackendError::DeviceLost,
                }),
                _ => Ok(()),
            }
        }
    }
    fn session(state: Arc<Mutex<State>>) -> Session<Fake> {
        Session {
            backend: Fake(state),
            context: Some(()),
            program: Some(()),
            queue: Some(()),
            buffers: vec![vec![0; 4], vec![0; 4]],
            event: None,
            sizes: vec![4, 4],
            inputs: 1,
            poisoned: false,
            quarantined: false,
        }
    }
    fn state(mode: Mode) -> Arc<Mutex<State>> {
        Arc::new(Mutex::new(State {
            mode,
            log: Vec::new(),
        }))
    }

    #[test]
    fn reuse_then_release_event_before_its_dependencies() {
        let state = state(Mode::Complete);
        let mut session = session(state.clone());
        for _ in 0..2 {
            assert!(session.run(vec![vec![1; 4]]).is_ok());
        }
        session.close().unwrap();
        assert_eq!(
            state.lock().unwrap().log,
            [
                "write", "submit", "event", "read", "write", "submit", "event", "read", "queue",
                "program", "buffer", "buffer", "context"
            ]
        );
    }

    #[test]
    fn rejected_submit_is_retryable_without_an_event() {
        let state = state(Mode::Rejected);
        let mut session = session(state.clone());
        assert!(
            session
                .run(vec![vec![1; 4]])
                .unwrap_err()
                .contains("rejected")
        );
        state.lock().unwrap().mode = Mode::Complete;
        assert!(session.run(vec![vec![1; 4]]).is_ok());
        session.close().unwrap();
        assert_eq!(
            state
                .lock()
                .unwrap()
                .log
                .iter()
                .filter(|s| **s == "event")
                .count(),
            1
        );
    }

    #[test]
    fn indeterminate_admission_is_drained_but_never_retried() {
        let state = state(Mode::Indeterminate);
        let mut session = session(state.clone());
        assert!(
            session
                .run(vec![vec![1; 4]])
                .unwrap_err()
                .contains("indeterminate")
        );
        assert!(
            session
                .run(vec![vec![1; 4]])
                .unwrap_err()
                .contains("poisoned")
        );
        session.close().unwrap();
        let log = &state.lock().unwrap().log;
        assert_eq!(log.iter().filter(|s| **s == "submit").count(), 1);
        assert!(log.contains(&"event"));
        assert!(!log.contains(&"read"));
        assert_eq!(log.last(), Some(&"context"));
    }

    #[test]
    fn uncertain_completion_and_release_never_destroy_dependencies() {
        for mode in [Mode::PollError, Mode::ReleaseRejected, Mode::ReleaseUnknown] {
            let state = state(mode);
            let mut session = session(state.clone());
            assert!(session.run(vec![vec![1; 4]]).is_err());
            assert!(session.close().is_err());
            let log = &state.lock().unwrap().log;
            assert!(!log.contains(&"read"));
            assert!(!log.contains(&"queue"));
            assert!(!log.contains(&"program"));
            assert!(!log.contains(&"buffer"));
            assert!(!log.contains(&"context"));
        }
    }

    #[test]
    fn terminal_failure_can_release_but_cannot_publish_output() {
        let state = state(Mode::Failed);
        let mut session = session(state.clone());
        assert!(session.run(vec![vec![1; 4]]).is_err());
        session.close().unwrap();
        assert!(!state.lock().unwrap().log.contains(&"read"));
        assert_eq!(state.lock().unwrap().log.last(), Some(&"context"));
    }

    #[test]
    fn transfer_device_loss_prevents_reentry() {
        let state = state(Mode::WriteLost);
        let mut session = session(state.clone());
        assert!(
            session
                .run(vec![vec![1; 4]])
                .unwrap_err()
                .contains("DeviceLost")
        );
        assert!(
            session
                .run(vec![vec![1; 4]])
                .unwrap_err()
                .contains("poisoned")
        );
        assert!(session.close().is_err());
        assert_eq!(state.lock().unwrap().log, ["write"]);
    }

    #[test]
    fn timeout_disconnects_without_releasing_pending_resources() {
        let state = state(Mode::Pending);
        let worker_state = state.clone();
        let (sender, receiver) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut session = session(worker_state);
            if let Job::Run(inputs, response) = receiver.recv().unwrap() {
                let result = session.run(inputs);
                let _ = response.send(result);
            } else {
                panic!("expected run");
            }
            assert!(receiver.recv().is_err());
            session.close().unwrap();
        });
        let mut worker = Worker {
            sender: Some(sender),
        };
        assert!(
            worker
                .run(vec![vec![1; 4]], Duration::from_millis(5))
                .is_err()
        );
        assert!(
            worker
                .run(vec![vec![1; 4]], Duration::from_secs(1))
                .unwrap_err()
                .contains("closed")
        );
        assert!(!state.lock().unwrap().log.contains(&"buffer"));
        state.lock().unwrap().mode = Mode::Complete;
        thread.join().unwrap();
        assert_eq!(state.lock().unwrap().log.last(), Some(&"context"));
    }
}

//! Run the shared exact INT8 quantized linear classifier on any host NPU backend that admits it.
//!
//! The program is backend-neutral: two INT8 feature samples times a direct-bound 3x2 weight
//! matrix, producing exact INT32 logits with a host-side integer argmax. Every attempted
//! backend reports its result — the first one available wins, and only exact quantized integer
//! execution ever runs; nothing dequantizes through floating point anywhere in the example.

use std::time::{Duration, Instant};
use virtio_accel::core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_conformance::numerics::{QUANTIZED_CLASSIFIER_INT8, TosaInt8MatmulCase};
use virtio_accel_tosa::{Model, Target, parse};

fn classify<A: Accelerator>(
    backend: &A,
    case: TosaInt8MatmulCase,
    model: &Model,
    target: Target,
    resident_bytes: u64,
) -> Result<Vec<Vec<i32>>, BackendError> {
    fn release<T>(result: Result<(), ReleaseFailure<T>>) -> Result<(), BackendError> {
        match result {
            Ok(()) => Ok(()),
            Err(
                ReleaseFailure::Rejected { error, .. } | ReleaseFailure::Indeterminate { error },
            ) => Err(error),
        }
    }

    let context = backend.create_context(ContextDesc::default())?;
    let mut buffers = Vec::new();
    for (slot, tensor) in case.inputs.iter().enumerate() {
        let desc = BufferDesc::new(
            tensor.bytes.len() as u64,
            16,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )?;
        let (mut buffer, _) = backend.allocate_buffer(&context, desc)?.into_parts();
        let bytes: [u8; 6] = tensor
            .bytes
            .try_into()
            .expect("fixture tensor lanes are six packed bytes");
        backend.write_buffer(&mut buffer, 0, &bytes)?;
        buffers.push((slot as u32, tensor, buffer));
    }
    let output_bytes = (case.outputs[0].values.len() * 4) as u64;
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_bytes,
                16,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )?,
        )?
        .into_parts();

    let program = backend.load_program(
        &context,
        model
            .artifact_ref(target, resident_bytes)
            .map_err(|_| BackendError::Incompatible)?,
    )?;
    let queue = backend.create_queue(&context, QueueDesc::default())?;
    let mut bindings: Vec<BindingRef<_>> = Vec::new();
    for (slot, tensor, buffer) in &buffers {
        bindings.push(BindingRef {
            slot: *slot,
            buffer,
            range: BufferRange::new(0, tensor.bytes.len() as u64)?,
            access: AccessMode::Read,
        });
    }
    bindings.push(BindingRef {
        slot: case.inputs.len() as u32,
        buffer: &output,
        range: BufferRange::new(0, output_bytes)?,
        access: AccessMode::Write,
    });
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .map_err(|failure| match failure {
            SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => error,
        })?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(&event)? {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => return Err(BackendError::DeadlineExpired),
            EventState::Complete => break,
            EventState::Failed(error) => return Err(error),
            EventState::Cancelled => return Err(BackendError::DeviceLost),
        }
    }

    let classes = case.inputs[1].shape[2];
    let samples = case.outputs[0].values.len() / classes;
    let mut raw = [0u8; 16];
    assert_eq!(
        raw.len(),
        classes * samples * 4,
        "logit lane count must equal samples times the class dimension"
    );
    backend.read_buffer(&output, 0, &mut raw)?;
    let logits: Vec<i32> = raw
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("packed i32 lane")))
        .collect();
    let rows = (0..samples)
        .map(|sample| logits[sample * classes..(sample + 1) * classes].to_vec())
        .collect();

    release(backend.destroy_event(event))?;
    release(backend.destroy_queue(queue))?;
    release(backend.unload_program(program))?;
    release(backend.free_buffer(output))?;
    for (_, _, buffer) in buffers {
        release(backend.free_buffer(buffer))?;
    }
    release(backend.destroy_context(context))?;
    Ok(rows)
}

fn via_coreml(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
    #[cfg(target_os = "macos")]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        use virtio_accel_coreml::{
            COREML_TOSA_INTEGER_TARGET, CoreMlAccelerator, REQUIRED_RESIDENT_BYTES,
        };
        let backend = CoreMlAccelerator::new_tosa().map_err(|error| format!("{error:?}"))?;
        classify(
            &backend,
            case,
            model,
            COREML_TOSA_INTEGER_TARGET,
            REQUIRED_RESIDENT_BYTES,
        )
        .map_err(|error| format!("{error:?}"))
    }
    #[cfg(not(target_os = "macos"))]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        let _ = (case, model);
        Err("Apple Core ML requires macOS 26+ and an Apple Neural Engine".to_owned())
    }
    run(case, model)
}

fn via_openvino(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
    #[cfg(va_openvino)]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        use virtio_accel_openvino::{
            OPENVINO_TOSA_INTEGER_TARGET, OpenVinoAccelerator, REQUIRED_RESIDENT_BYTES,
        };
        let backend = OpenVinoAccelerator::new().map_err(|error| format!("{error:?}"))?;
        classify(
            &backend,
            case,
            model,
            OPENVINO_TOSA_INTEGER_TARGET,
            REQUIRED_RESIDENT_BYTES,
        )
        .map_err(|error| format!("{error:?}"))
    }
    #[cfg(not(va_openvino))]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        let _ = (case, model);
        Err("the OpenVINO runtime was not detected at build time".to_owned())
    }
    run(case, model)
}

fn via_hexagon(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
    #[cfg(va_hexagon)]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        use virtio_accel_hexagon::{
            HEXAGON_TOSA_INTEGER_TARGET, HexagonAccelerator, REQUIRED_RESIDENT_BYTES,
        };
        let backend = HexagonAccelerator::new().map_err(|error| format!("{error:?}"))?;
        classify(
            &backend,
            case,
            model,
            HEXAGON_TOSA_INTEGER_TARGET,
            REQUIRED_RESIDENT_BYTES,
        )
        .map_err(|error| format!("{error:?}"))
    }
    #[cfg(not(va_hexagon))]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        let _ = (case, model);
        Err("the QAIRT/QNN SDK was not detected at build time".to_owned())
    }
    run(case, model)
}

fn via_xdna(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
    #[cfg(va_xdna)]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        use virtio_accel_xdna::{
            REQUIRED_RESIDENT_BYTES, XDNA_TOSA_INTEGER_TARGET, XdnaAccelerator,
        };
        let backend = XdnaAccelerator::new().map_err(|error| format!("{error:?}"))?;
        classify(
            &backend,
            case,
            model,
            XDNA_TOSA_INTEGER_TARGET,
            REQUIRED_RESIDENT_BYTES,
        )
        .map_err(|error| format!("{error:?}"))
    }
    #[cfg(not(va_xdna))]
    fn run(case: TosaInt8MatmulCase, model: &Model) -> Result<Vec<Vec<i32>>, String> {
        let _ = (case, model);
        Err("the HRX runtime was not detected at build time".to_owned())
    }
    run(case, model)
}

type Runner = fn(TosaInt8MatmulCase, &Model) -> Result<Vec<Vec<i32>>, String>;

/// NPUs first, in the order this project's device matrix ranks honesty of device selection.
fn runners() -> &'static [(&'static str, Runner)] {
    &[
        ("amd-xdna", via_xdna),
        ("qualcomm-hexagon", via_hexagon),
        ("apple-coreml", via_coreml),
        ("intel-openvino", via_openvino),
    ]
}

fn main() -> std::process::ExitCode {
    let case = QUANTIZED_CLASSIFIER_INT8;
    let model = match parse(case.artifact) {
        Ok(model) => model,
        Err(_) => {
            eprintln!("the shared INT8 classifier artifact must always parse");
            return std::process::ExitCode::FAILURE;
        }
    };
    for (name, run) in runners() {
        match run(case, &model) {
            Ok(rows) => {
                println!("executed on {name}");
                for (sample, logits) in rows.iter().enumerate() {
                    let winner = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.cmp(b))
                        .map(|(index, _)| index)
                        .expect("nonempty logits");
                    println!("sample {sample}: logits {logits:?} -> class {winner}");
                }
                return std::process::ExitCode::SUCCESS;
            }
            Err(error) => println!("skipping {name}: {error}"),
        }
    }
    eprintln!("no backend admitted the quantized INT8 classifier on this host");
    std::process::ExitCode::FAILURE
}

//! P0 probe runner (issue #146): drive the encoding-dump kernel through the released
//! `Accelerator` lifecycle and decode the raw `bfp16ebs8` planes against hypothesis H1.
//!
//! Usage: `bfp16ebs8-probe-runner <dir-with-final.xclbin-and-insts.bin>`

use std::time::{Duration, Instant};

use virtio_accel_core::{
    AccessMode, Accelerator, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc,
    TargetIdentity, Timeout,
};
use virtio_accel_xdna::{XDNA_PRECOMPILED_FORMAT, XdnaAccelerator, artifact};

#[derive(Debug)]
struct Slice<'a>(&'a [u8]);

impl ByteSource for Slice<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(dst.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        dst.copy_from_slice(&self.0[start..end]);
        Ok(())
    }
}

#[derive(Debug)]
struct SliceMut<'a>(&'a mut [u8]);

impl ByteSink for SliceMut<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, src: &[u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(src.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        self.0[start..end].copy_from_slice(src);
        Ok(())
    }
}

const INPUT_BYTES: u64 = 64 * 4;
const OUTPUT_BYTES: u64 = 148;

/// H1: value = mantissa * 2^(exponent - 127 - 6), two's-complement int8 mantissa.
fn h1_decode(mantissa: i8, exponent: u8) -> f64 {
    f64::from(mantissa) * (f64::from(exponent) - 133.0).exp2()
}

fn run_case<A: Accelerator>(
    backend: &A,
    queue: &A::Queue,
    program: &A::Program,
    input: &mut A::Buffer,
    output: &A::Buffer,
    name: &str,
    values: &[f32; 64],
) {
    let mut payload = Vec::with_capacity(INPUT_BYTES as usize);
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    backend
        .write_buffer(input, 0, &Slice(&payload))
        .expect("write input");

    let bindings = [
        BindingRef {
            slot: 0,
            buffer: input,
            range: BufferRange::new(0, INPUT_BYTES).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: output,
            range: BufferRange::new(0, OUTPUT_BYTES).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = match backend.submit(queue, program, &bindings, Timeout::Infinite) {
        Ok(event) => event,
        Err(virtio_accel_core::SubmitFailure::Rejected(error)) => {
            panic!("submit rejected: {error:?}")
        }
        Err(_) => panic!("submit indeterminate"),
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => panic!("probe did not complete in 30s"),
            EventState::Failed(error) => panic!("probe dispatch failed: {error:?}"),
            _ => break,
        }
    }
    if backend.destroy_event(event).is_err() {
        panic!("destroy event failed");
    }

    let mut raw = vec![0u8; OUTPUT_BYTES as usize];
    backend
        .read_buffer(output, 0, &mut SliceMut(&mut raw))
        .expect("read output");

    let mantissa = &raw[0..64];
    let exponent = &raw[64..72];
    let native = &raw[72..144];
    let rnd = u32::from_le_bytes(raw[144..148].try_into().unwrap());

    println!("== case {name} (rnd mode at entry: {rnd})");
    println!("   exponent plane: {exponent:02x?}");
    for block in 0..8 {
        let e = exponent[block];
        print!("   block {block} (e={e:3}): ");
        for lane in 0..8 {
            let idx = block * 8 + lane;
            let m = mantissa[idx] as i8;
            let input_value = values[idx];
            let decoded = h1_decode(m, e);
            let mark = if (decoded - f64::from(input_value)).abs() < 1e-9 {
                ' '
            } else {
                '!'
            };
            print!("{input_value}->m={m}{mark} ");
        }
        println!();
    }
    // Native-layout comparison: where do the register planes land in the stored struct?
    let plane_order = native[0..64] == raw[0..64] && native[64..72] == raw[64..72];
    let exp_first = native[0..8] == raw[64..72] && native[8..72] == raw[0..64];
    println!(
        "   native struct layout: mantissa-then-exponent={plane_order} exponent-then-mantissa={exp_first}"
    );
    if !plane_order && !exp_first {
        println!("   native bytes: {native:02x?}");
    }
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: runner <probe-dir>");
    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");
    let container = artifact::encode(
        "MLIR_AIE",
        &[INPUT_BYTES],
        &[OUTPUT_BYTES],
        &xclbin,
        &insts,
    );

    let backend = XdnaAccelerator::new().expect("construct XDNA backend (needs the NPU + HRX)");
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: XDNA_PRECOMPILED_FORMAT,
                target: TargetIdentity([0; 12]),
                payload: &Slice(&container),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load probe artifact");

    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                INPUT_BYTES,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                OUTPUT_BYTES,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();

    // Case A: each block holds one constant power of two -> distinct exponent per block.
    let mut a = [0f32; 64];
    for block in 0..8 {
        for lane in 0..8 {
            a[block * 8 + lane] = (2f32).powi(block as i32 - 3);
        }
    }
    // Case B: one block probing sign and range; other blocks zero.
    let mut b = [0f32; 64];
    b[..8].copy_from_slice(&[1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 127.0 / 64.0, -2.0]);
    // Case C: mixed magnitudes in one block -> normalization loss.
    let mut c = [0f32; 64];
    c[..8].copy_from_slice(&[
        1.0,
        1.0 / 64.0,
        1.0 / 128.0,
        1.5 / 64.0,
        -1.0 / 64.0,
        0.0,
        1.0 + 1.0 / 64.0,
        -1.0 - 1.0 / 64.0,
    ]);
    // Case D: all zeros.
    let d = [0f32; 64];

    for (name, values) in [("A", &a), ("B", &b), ("C", &c), ("D", &d)] {
        run_case(&backend, &queue, &program, &mut input, &output, name, values);
    }

    assert!(backend.free_buffer(input).is_ok(), "free input");
    assert!(backend.free_buffer(output).is_ok(), "free output");
    assert!(backend.unload_program(program).is_ok(), "unload");
    assert!(backend.destroy_queue(queue).is_ok(), "destroy queue");
    assert!(backend.destroy_context(context).is_ok(), "destroy context");
    println!("P0 complete");
}

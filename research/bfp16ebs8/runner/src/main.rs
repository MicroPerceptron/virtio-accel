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
const P0_OUTPUT_BYTES: u64 = 148;
const P1_OUTPUT_BYTES: u64 = 724;
const P1_MODES: [(u32, &str); 10] = [
    (0, "floor"),
    (1, "ceil"),
    (2, "sym_floor"),
    (3, "sym_ceil"),
    (8, "neg_inf"),
    (9, "pos_inf"),
    (10, "sym_zero"),
    (11, "sym_inf"),
    (12, "conv_even"),
    (13, "conv_odd"),
];

/// Reference rounding of an exact mantissa-unit value under a crrnd mode.
fn round_reference(mode: u32, x: f64) -> f64 {
    let half = x.abs().fract() == 0.5;
    match mode {
        0 => x.floor(),
        1 => x.ceil(),
        2 => x.trunc(),
        3 => x.abs().ceil().copysign(x),
        8 => {
            if half {
                x.floor()
            } else {
                x.round()
            }
        }
        9 => {
            if half {
                x.ceil()
            } else {
                x.round()
            }
        }
        10 => {
            if half {
                x.trunc()
            } else {
                x.round()
            }
        }
        11 => x.round(), // f64::round is ties-away-from-zero
        12 => {
            // ties to even
            if half {
                let down = x.floor();
                if (down as i64) % 2 == 0 { down } else { x.ceil() }
            } else {
                x.round()
            }
        }
        13 => {
            // ties to odd
            if half {
                let down = x.floor();
                if (down as i64) % 2 != 0 { down } else { x.ceil() }
            } else {
                x.round()
            }
        }
        _ => unreachable!(),
    }
}

/// H1: value = mantissa * 2^(exponent - 127 - 6), two's-complement int8 mantissa.
fn h1_decode(mantissa: i8, exponent: u8) -> f64 {
    f64::from(mantissa) * (f64::from(exponent) - 133.0).exp2()
}

fn submit_case<A: Accelerator>(
    backend: &A,
    queue: &A::Queue,
    program: &A::Program,
    input: &mut A::Buffer,
    output: &A::Buffer,
    out_bytes: u64,
    values: &[f32; 64],
) -> Vec<u8> {
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
            range: BufferRange::new(0, out_bytes).unwrap(),
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

    let mut raw = vec![0u8; out_bytes as usize];
    backend
        .read_buffer(output, 0, &mut SliceMut(&mut raw))
        .expect("read output");
    raw
}

fn decode_p0(name: &str, values: &[f32; 64], raw: &[u8]) {
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

fn decode_p1(name: &str, values: &[f32; 64], raw: &[u8]) {
    let entry_rnd = u32::from_le_bytes(raw[720..724].try_into().unwrap());
    println!("== case {name} (rnd mode at entry: {entry_rnd})");
    for (slot, (mode, mode_name)) in P1_MODES.iter().enumerate() {
        let plane = &raw[slot * 72..slot * 72 + 72];
        let mantissa = &plane[0..64];
        let exponent = &plane[64..72];
        let mut mismatches = Vec::new();
        for block in 0..8 {
            let e = exponent[block];
            for lane in 0..8 {
                let idx = block * 8 + lane;
                let v = f64::from(values[idx]);
                if v == 0.0 && e == 0 {
                    continue;
                }
                let exact = v * (133.0 - f64::from(e)).exp2();
                let expected = round_reference(*mode, exact);
                let got = f64::from(mantissa[idx] as i8);
                if got != expected {
                    mismatches.push(format!(
                        "idx {idx} v={v} e={e} exact={exact} expected m={expected} got m={got}"
                    ));
                }
            }
        }
        if mismatches.is_empty() {
            println!(
                "   mode {mode:2} {mode_name:>9}: PASS (all 64 lanes match reference)  e[0..4]={:?}",
                &exponent[0..4]
            );
        } else {
            println!("   mode {mode:2} {mode_name:>9}: {} MISMATCHES", mismatches.len());
            for m in &mismatches {
                println!("      {m}");
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let probe = args.next().expect("usage: runner <p0|p1> <probe-dir>");
    let dir = args.next().expect("usage: runner <p0|p1> <probe-dir>");
    let out_bytes = match probe.as_str() {
        "p0" | "p2" | "p3" => P0_OUTPUT_BYTES,
        "p1" => P1_OUTPUT_BYTES,
        other => panic!("unknown probe {other}"),
    };
    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");
    let container = artifact::encode("MLIR_AIE", &[INPUT_BYTES], &[out_bytes], &xclbin, &insts);

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
                out_bytes,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();

    match probe.as_str() {
        "p0" => {
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
                let raw =
                    submit_case(&backend, &queue, &program, &mut input, &output, out_bytes, values);
                decode_p0(name, values, &raw);
            }
        }
        "p1" => {
            // All blocks anchored by max member ±1.0 (IEEE exp 0 -> e = 127), so one mantissa
            // unit is exactly 1/64 and the chosen fractions produce exact binary ties.
            let mut t = [0f32; 64];
            // b0: half-unit ties, positive and negative.
            t[..8].copy_from_slice(&[
                1.0,
                0.5 / 64.0,
                1.5 / 64.0,
                2.5 / 64.0,
                -0.5 / 64.0,
                -1.5 / 64.0,
                -2.5 / 64.0,
                -1.0,
            ]);
            // b1: quarter-unit non-ties (round direction without tie ambiguity).
            t[8..16].copy_from_slice(&[
                1.0,
                0.25 / 64.0,
                0.75 / 64.0,
                1.25 / 64.0,
                -0.25 / 64.0,
                -0.75 / 64.0,
                -1.25 / 64.0,
                -1.0,
            ]);
            // b2: large-mantissa ties and the saturation corner (exact x: 127, -127, 64.5,
            // -64.5, 65.5, -65.5, 32, -32 at e = 127).
            t[16..24].copy_from_slice(&[
                127.0 / 64.0,
                -127.0 / 64.0,
                64.5 / 64.0,
                -64.5 / 64.0,
                65.5 / 64.0,
                -65.5 / 64.0,
                0.5,
                -0.5,
            ]);
            let raw = submit_case(&backend, &queue, &program, &mut input, &output, out_bytes, &t);
            decode_p1("ties", &t, &raw);

            // Saturation-by-rounding: exact x = +-127.5 at e = 127. A mode that rounds the
            // magnitude up cannot produce m = 128; the exponent plane shows whether the
            // hardware bumps e to 128 (m = 64) or saturates m at 127.
            let mut sat = [0f32; 64];
            sat[..8].copy_from_slice(&[
                127.5 / 64.0,
                -127.5 / 64.0,
                1.0,
                -1.0,
                127.0 / 64.0,
                -127.0 / 64.0,
                0.5,
                -0.5,
            ]);
            let raw =
                submit_case(&backend, &queue, &program, &mut input, &output, out_bytes, &sat);
            decode_p1("sat", &sat, &raw);
        }
        "p2" => {
            // Normalization corners through the P0 kernel (default rounding = floor).
            // N1: exponent spread beyond the mantissa (tiny values vanish at the max's e).
            let mut n1 = [0f32; 64];
            n1[..8].copy_from_slice(&[
                1.0,
                (2f32).powi(-7),
                (2f32).powi(-13),
                (2f32).powi(-14),
                -(2f32).powi(-13),
                (2f32).powi(-20),
                0.0,
                1.0,
            ]);
            // N2: negative-only block (max magnitude negative).
            let mut n2 = [0f32; 64];
            n2[..8].copy_from_slice(&[-1.0, -0.5, -0.25, -0.75, -1.25, -1.75, -1.984375, -0.125]);
            // N3: FP32 subnormal-only block (IEEE exponent field 0).
            let mut n3 = [0f32; 64];
            n3[..8].copy_from_slice(&[
                f32::from_bits(0x0000_0001),
                f32::from_bits(0x007f_ffff),
                f32::from_bits(0x0040_0000),
                -f32::from_bits(0x0040_0000),
                0.0,
                0.0,
                0.0,
                0.0,
            ]);
            // N4: near the top of the FP32 exponent range.
            let mut n4 = [0f32; 64];
            n4[..8].copy_from_slice(&[
                f32::MAX,
                f32::MAX / 2.0,
                1.0,
                -1.0,
                (2f32).powi(120),
                -(2f32).powi(120),
                0.0,
                0.0,
            ]);
            for (name, values) in [("N1", &n1), ("N2", &n2), ("N3", &n3), ("N4", &n4)] {
                let raw =
                    submit_case(&backend, &queue, &program, &mut input, &output, out_bytes, values);
                decode_p0(name, values, &raw);
            }
        }
        "p3" => {
            // Exceptional values through the P0 kernel.
            let mut x1 = [0f32; 64];
            x1[..8].copy_from_slice(&[
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                -f32::NAN,
                1.0,
                -1.0,
                0.0,
                -0.0,
            ]);
            // Inf/NaN isolated per block (do they poison only their own block's exponent?).
            let mut x2 = [0f32; 64];
            x2[0] = f32::INFINITY;
            x2[8] = f32::NAN;
            x2[16] = 1.0;
            x2[17] = f32::from_bits(0x7f7f_ffff); // f32::MAX
            x2[24] = -0.0;
            for (name, values) in [("X1", &x1), ("X2", &x2)] {
                let raw =
                    submit_case(&backend, &queue, &program, &mut input, &output, out_bytes, values);
                decode_p0(name, values, &raw);
            }
        }
        _ => unreachable!(),
    }

    assert!(backend.free_buffer(input).is_ok(), "free input");
    assert!(backend.free_buffer(output).is_ok(), "free output");
    assert!(backend.unload_program(program).is_ok(), "unload");
    assert!(backend.destroy_queue(queue).is_ok(), "destroy queue");
    assert!(backend.destroy_context(context).is_ok(), "destroy context");
    println!("{probe} complete");
}

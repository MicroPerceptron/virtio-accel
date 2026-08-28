//! P0 probe runner (issue #146): drive the encoding-dump kernel through the released
//! `Accelerator` lifecycle and decode the raw `bfp16ebs8` planes against hypothesis H1.
//!
//! Usage: `bfp16ebs8-probe-runner <dir-with-final.xclbin-and-insts.bin>`

use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc,
    TargetIdentity, Timeout,
};
use virtio_accel_xdna::{XDNA_PRECOMPILED_FORMAT, XdnaAccelerator, artifact};

#[allow(dead_code)] // The model is the citable #146 deliverable; probes use what each needs.
mod model;

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

const F32_INPUT_BYTES: u64 = 64 * 4;
const P0_OUTPUT_BYTES: u64 = 148;
const P1_OUTPUT_BYTES: u64 = 724;
const P4_INPUT_BYTES: u64 = 576;
const P4_OUTPUT_BYTES: u64 = 256;
const XBFP_K: usize = 512;
const XBFP_INPUT_BYTES: u64 = 2 * (XBFP_K as u64 / 8) * 72;
const XBFP_OUTPUT_BYTES: u64 = 256;
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
                if (down as i64) % 2 == 0 {
                    down
                } else {
                    x.ceil()
                }
            } else {
                x.round()
            }
        }
        13 => {
            // ties to odd
            if half {
                let down = x.floor();
                if (down as i64) % 2 != 0 {
                    down
                } else {
                    x.ceil()
                }
            } else {
                x.round()
            }
        }
        _ => unreachable!(),
    }
}

fn f32_payload(values: &[f32; 64]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(F32_INPUT_BYTES as usize);
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
}

/// One raw v64bfp16ebs8: 64 mantissa bytes then 8 exponent bytes (layout pinned by P0).
fn craft(mantissa: &[i8; 64], exponent: &[u8; 8]) -> [u8; 72] {
    let mut unit = [0u8; 72];
    for (i, m) in mantissa.iter().enumerate() {
        unit[i] = *m as u8;
    }
    unit[64..72].copy_from_slice(exponent);
    unit
}

/// Reference MMUL under the transposed-B hypothesis: A is 8x32 (chunk k holds columns
/// 8k..8k+8, row-major i*8+c within the chunk), B is the 8x32 view of the transposed
/// 32x8 operand (same chunk/lane layout), C lane i*8+j = sum_k A[i,k] * B[j,k].
/// All products are integers scaled by powers of two, so f64 accumulation is exact and
/// FP32 comparison is order-independent for the magnitudes the cases use.
fn p4_reference(a_planes: &[[u8; 72]; 4], b_planes: &[[u8; 72]; 4]) -> [f32; 64] {
    let mut c = [0f64; 64];
    for chunk in 0..4 {
        let a = &a_planes[chunk];
        let b = &b_planes[chunk];
        for i in 0..8 {
            for j in 0..8 {
                for lane in 0..8 {
                    let ea = a[64 + i.min(7)]; // per-block exponent: block = row index
                    let eb = b[64 + j.min(7)];
                    let ma = f64::from(a[i * 8 + lane] as i8);
                    let mb = f64::from(b[j * 8 + lane] as i8);
                    if ma != 0.0 && mb != 0.0 {
                        c[i * 8 + j] += ma * mb * (f64::from(ea) + f64::from(eb) - 266.0).exp2();
                    }
                }
            }
        }
    }
    let mut out = [0f32; 64];
    for (o, v) in out.iter_mut().zip(c) {
        *o = v as f32;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn mm_run<A: Accelerator>(
    backend: &A,
    queue: &A::Queue,
    program: &A::Program,
    input: &mut A::Buffer,
    output: &A::Buffer,
    a: &[[u8; 72]; 4],
    b: &[[u8; 72]; 4],
) -> [f32; 64] {
    let mut payload = Vec::with_capacity(576);
    for chunk in a {
        payload.extend_from_slice(chunk);
    }
    for chunk in b {
        payload.extend_from_slice(chunk);
    }
    let raw = submit_case(
        backend,
        queue,
        program,
        input,
        output,
        P4_INPUT_BYTES,
        P4_OUTPUT_BYTES,
        &payload,
    );
    let mut lanes = [0f32; 64];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    }
    lanes
}

#[allow(clippy::too_many_arguments)]
fn mm_check<A: Accelerator>(
    backend: &A,
    queue: &A::Queue,
    program: &A::Program,
    input: &mut A::Buffer,
    output: &A::Buffer,
    name: &str,
    a: &[[u8; 72]; 4],
    b: &[[u8; 72]; 4],
) {
    let got = mm_run(backend, queue, program, input, output, a, b);
    let expected = p4_reference(a, b);
    let mismatches: Vec<usize> = (0..64)
        .filter(|&i| got[i].to_bits() != expected[i].to_bits())
        .collect();
    if mismatches.is_empty() {
        println!("   {name}: PASS (all 64 lanes bit-exact vs reference)");
    } else {
        println!("   {name}: {} MISMATCHES", mismatches.len());
        for i in mismatches.iter().take(12) {
            println!(
                "      lane {i} (i={},j={}): got {} ({:#010x}), expected {} ({:#010x})",
                i / 8,
                i % 8,
                got[*i],
                got[*i].to_bits(),
                expected[*i],
                expected[*i].to_bits()
            );
        }
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
    in_bytes: u64,
    out_bytes: u64,
    payload: &[u8],
) -> Vec<u8> {
    assert_eq!(payload.len() as u64, in_bytes);
    backend
        .write_buffer(input, 0, &Slice(payload))
        .expect("write input");

    let bindings = [
        BindingRef {
            slot: 0,
            buffer: input,
            range: BufferRange::new(0, in_bytes).unwrap(),
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
            print!(" {input_value}->m={m}{mark}");
        }
        println!();
    }
    // Model cross-validation: the silicon planes must equal the reference model bit for bit.
    let (model_m, model_e) = model::encode_v64(values, model::HARDWARE_DEFAULT_MODE);
    let silicon_m: Vec<i8> = mantissa.iter().map(|&b| b as i8).collect();
    if silicon_m == model_m && exponent == model_e {
        println!("   model check: PASS (planes bit-identical to model::encode_v64)");
    } else {
        println!("   model check: MISMATCH");
        println!("      model e: {model_e:?}");
        println!("      model m: {model_m:?}");
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
            println!(
                "   mode {mode:2} {mode_name:>9}: {} MISMATCHES",
                mismatches.len()
            );
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
    let (in_bytes, out_bytes) = match probe.as_str() {
        "p0" | "p2" | "p3" => (F32_INPUT_BYTES, P0_OUTPUT_BYTES),
        "p1" => (F32_INPUT_BYTES, P1_OUTPUT_BYTES),
        "p4" | "p5" => (P4_INPUT_BYTES, P4_OUTPUT_BYTES),
        "xbfp" => (XBFP_INPUT_BYTES, XBFP_OUTPUT_BYTES),
        other => panic!("unknown probe {other}"),
    };
    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");
    let container = artifact::encode("MLIR_AIE", &[in_bytes], &[out_bytes], &xclbin, &insts);

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
                in_bytes,
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
            // Case E: pseudorandom finite values — silicon planes must equal the model exactly.
            let mut e_case = [0f32; 64];
            let mut state = 0x00c0_ffeeu32;
            for v in e_case.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                // Finite, mixed sign and magnitude across ~2^-8..2^8.
                let mag = ((state >> 8) & 0xffff) as f32 / 256.0;
                *v = if state & 1 == 0 { mag } else { -mag };
            }
            for (name, values) in [("A", &a), ("B", &b), ("C", &c), ("D", &d), ("E", &e_case)] {
                let raw = submit_case(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    in_bytes,
                    out_bytes,
                    &f32_payload(values),
                );
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
            let raw = submit_case(
                &backend,
                &queue,
                &program,
                &mut input,
                &output,
                in_bytes,
                out_bytes,
                &f32_payload(&t),
            );
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
            let raw = submit_case(
                &backend,
                &queue,
                &program,
                &mut input,
                &output,
                in_bytes,
                out_bytes,
                &f32_payload(&sat),
            );
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
                let raw = submit_case(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    in_bytes,
                    out_bytes,
                    &f32_payload(values),
                );
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
                let raw = submit_case(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    in_bytes,
                    out_bytes,
                    &f32_payload(values),
                );
                decode_p0(name, values, &raw);
            }
        }
        "p4" | "p5" => {
            let zero = craft(&[0i8; 64], &[0u8; 8]);

            if probe == "p4" {
                println!("== P4 layout inference (single-entry A, graded B, chunk 0 only)");
                // B lane q holds mantissa q+1 at e=127 in every block.
                let mut mb = [0i8; 64];
                for (q, m) in mb.iter_mut().enumerate() {
                    *m = (q + 1) as i8;
                }
                let b0 = craft(&mb, &[127u8; 8]);
                let b = [b0, zero, zero, zero];
                for p in [0usize, 1, 8, 9] {
                    let mut ma = [0i8; 64];
                    ma[p] = 64;
                    let a = [craft(&ma, &[127u8; 8]), zero, zero, zero];
                    let lanes = mm_run(&backend, &queue, &program, &mut input, &output, &a, &b);
                    let nonzero: Vec<String> = (0..64)
                        .filter(|&i| lanes[i] != 0.0)
                        .map(|i| format!("C[{},{}]={}", i / 8, i % 8, lanes[i]))
                        .collect();
                    println!("   A[{p}]=1.0: {}", nonzero.join(" "));
                }

                println!("== P4 exactness under the inferred layout");
                // Deterministic pseudo-random mantissas including -128 and +/-127 corners.
                let mut ma = [0i8; 64];
                let mut mb2 = [0i8; 64];
                let mut state = 0x12345678u32;
                let mut next = move || {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8 as i8
                };
                for i in 0..64 {
                    ma[i] = next();
                    mb2[i] = next();
                }
                ma[0] = -128;
                ma[1] = 127;
                mb2[0] = 127;
                mb2[1] = -128;
                let a = [
                    craft(&ma, &[127u8; 8]),
                    craft(&mb2, &[127u8; 8]),
                    craft(&ma, &[126u8; 8]),
                    craft(&mb2, &[125u8; 8]),
                ];
                let b = [
                    craft(&mb2, &[127u8; 8]),
                    craft(&ma, &[127u8; 8]),
                    craft(&mb2, &[128u8; 8]),
                    craft(&ma, &[124u8; 8]),
                ];
                mm_check(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    "uniform + per-chunk exponents (incl. m = -128)",
                    &a,
                    &b,
                );

                println!("== P4 per-block exponent disagreement inside one chunk");
                let ea: [u8; 8] = [120, 121, 122, 123, 124, 125, 126, 127];
                let eb: [u8; 8] = [127, 126, 125, 124, 123, 122, 121, 120];
                let a = [craft(&ma, &ea), zero, zero, zero];
                let b = [craft(&mb2, &eb), zero, zero, zero];
                mm_check(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    "disagreeing block exponents",
                    &a,
                    &b,
                );
            } else {
                println!("== P5 MXINT8 block-32 decomposition (H6)");
                // Two MXINT8-32 operand rows: 32 int8 elements, ONE shared E8M0 scale each.
                // Decomposed to four block-8 groups with EQUAL exponent bytes = the MX scale.
                let mut state = 0x5eed_cafeu32;
                let mut next = move || {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8 as i8
                };
                // A: rows i share mantissa pattern; every chunk gets the SAME exponent (MX-32).
                let mut ma = [[0i8; 64]; 4];
                let mut mb = [[0i8; 64]; 4];
                for chunk in 0..4 {
                    for lane in 0..64 {
                        ma[chunk][lane] = next();
                        mb[chunk][lane] = next();
                    }
                }
                // MX scales: A row scale e=127, B row scale e=125 (both uniform across chunks).
                let a = [
                    craft(&ma[0], &[127u8; 8]),
                    craft(&ma[1], &[127u8; 8]),
                    craft(&ma[2], &[127u8; 8]),
                    craft(&ma[3], &[127u8; 8]),
                ];
                let b = [
                    craft(&mb[0], &[125u8; 8]),
                    craft(&mb[1], &[125u8; 8]),
                    craft(&mb[2], &[125u8; 8]),
                    craft(&mb[3], &[125u8; 8]),
                ];
                mm_check(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    "MXINT8 32-dot as four equal-exponent block-8 groups",
                    &a,
                    &b,
                );

                // The mapping-direction evidence: the same numeric values expressed with
                // per-sub-block exponents (block-8 native form, more precision available).
                // Halving mantissas at e+1 must reproduce identical products.
                let mut ma_half = [[0i8; 64]; 4];
                for chunk in 0..4 {
                    for lane in 0..64 {
                        // keep only even mantissas so m/2 at e+1 is exact
                        ma_half[chunk][lane] = ma[chunk][lane] & !1;
                    }
                }
                let a_even = [
                    craft(&ma_half[0], &[127u8; 8]),
                    craft(&ma_half[1], &[127u8; 8]),
                    craft(&ma_half[2], &[127u8; 8]),
                    craft(&ma_half[3], &[127u8; 8]),
                ];
                let mut ma_shift = [[0i8; 64]; 4];
                for chunk in 0..4 {
                    for lane in 0..64 {
                        ma_shift[chunk][lane] = ma_half[chunk][lane] / 2;
                    }
                }
                let a_shift = [
                    craft(&ma_shift[0], &[128u8; 8]),
                    craft(&ma_shift[1], &[128u8; 8]),
                    craft(&ma_shift[2], &[128u8; 8]),
                    craft(&ma_shift[3], &[128u8; 8]),
                ];
                let c_even = mm_run(&backend, &queue, &program, &mut input, &output, &a_even, &b);
                let c_shift = mm_run(
                    &backend, &queue, &program, &mut input, &output, &a_shift, &b,
                );
                let same = (0..64).all(|i| c_even[i].to_bits() == c_shift[i].to_bits());
                println!(
                    "   same values via (m, e) vs (m/2, e+1): {}",
                    if same { "BIT-IDENTICAL" } else { "DIFFER" }
                );
                mm_check(
                    &backend,
                    &queue,
                    &program,
                    &mut input,
                    &output,
                    "equal-exponent form vs reference",
                    &a_even,
                    &b,
                );
            }
        }
        "xbfp" => {
            println!("== XBFP flavor-1 accumulation-order contract (K = {XBFP_K})");
            // Per-row planes: CHUNKS units for A (row set 0..8) and B. Mixed per-32-group
            // exponents with a large spread so FP32 fold order genuinely matters.
            let chunks = XBFP_K / 8;
            let mut state = 0x0148_1481u32;
            let mut next = move || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8 as i8
            };
            // Mantissa planes per chunk (64 lanes = 8 rows x 8 k-lanes).
            let mut a_m = vec![[0i8; 64]; chunks];
            let mut b_m = vec![[0i8; 64]; chunks];
            for chunk in 0..chunks {
                for lane in 0..64 {
                    a_m[chunk][lane] = next();
                    b_m[chunk][lane] = next();
                }
            }
            // Exponents: per MXINT8 semantics the four chunks of one 32-group share a byte.
            // Group 0 is large; every later group's chunk values sit just below the running
            // accumulator's FP32 half-ULP, so an FP32 fold drops each one individually while
            // an f64 sum accumulates them past the rounding boundary — the discriminating
            // case for the documented accumulation order.
            let group_e = |group: usize| -> u8 { if group == 0 { 130 } else { 118 } };
            let mut a_units: Vec<[u8; 72]> = Vec::with_capacity(chunks);
            let mut b_units: Vec<[u8; 72]> = Vec::with_capacity(chunks);
            for chunk in 0..chunks {
                let e = group_e(chunk / 4);
                a_units.push(craft(&a_m[chunk], &[e; 8]));
                b_units.push(craft(&b_m[chunk], &[e; 8]));
            }
            let mut payload = Vec::with_capacity(XBFP_INPUT_BYTES as usize);
            for unit in &a_units {
                payload.extend_from_slice(unit);
            }
            for unit in &b_units {
                payload.extend_from_slice(unit);
            }
            let raw = submit_case(
                &backend, &queue, &program, &mut input, &output, in_bytes, out_bytes, &payload,
            );

            // Oracle per output lane (i, j): fold row i of A against row j of B in chunk order.
            let mut fold_matches = 0usize;
            let mut naive_differs = 0usize;
            for i in 0..8 {
                for j in 0..8 {
                    let mut am = Vec::with_capacity(XBFP_K);
                    let mut bm = Vec::with_capacity(XBFP_K);
                    let mut ae = Vec::with_capacity(chunks);
                    let mut be = Vec::with_capacity(chunks);
                    for chunk in 0..chunks {
                        for lane in 0..8 {
                            am.push(a_m[chunk][i * 8 + lane]);
                            bm.push(b_m[chunk][j * 8 + lane]);
                        }
                        let e = group_e(chunk / 4);
                        ae.push(e);
                        be.push(e);
                    }
                    let expected = model::dot_fold_f32(&am, &ae, &bm, &be);
                    let naive = model::dot_reference(&am, &ae, &bm, &be, 8) as f32;
                    let got = f32::from_le_bytes(
                        raw[(i * 8 + j) * 4..(i * 8 + j) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    if got.to_bits() == expected.to_bits() {
                        fold_matches += 1;
                    } else {
                        println!(
                            "   lane ({i},{j}): got {got} ({:#010x}) expected {expected} ({:#010x})",
                            got.to_bits(),
                            expected.to_bits()
                        );
                    }
                    if expected.to_bits() != naive.to_bits() {
                        naive_differs += 1;
                    }
                }
            }
            println!("   fold-order oracle: {fold_matches}/64 lanes bit-exact");
            println!(
                "   order sensitivity: fold differs from single-rounded f64 sum on {naive_differs}/64 lanes"
            );
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

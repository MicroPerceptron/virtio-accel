//! On-metal suite for the AMD `bfp16ebs8` vendor experiment (issue #148).
//!
//! Gated on a detected HRX runtime like `hardware.rs`; the committed fixture is precompiled,
//! so this suite needs the NPU but not the compiler toolchain. The numerical oracle is the
//! vendored #146 reference model (`bfp_model.rs`): every comparison is bit-exact against the
//! documented FP32 ascending-`k` fold, including a case constructed so that the fold order is
//! the only explanation of the observed bits.
#![cfg(va_xdna)]

#[path = "bfp_model.rs"]
mod bfp_model;

use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, Timeout,
};
use virtio_accel_xdna::bfp_experiment::{
    UNIT_BYTES, XDNA_BFP_EXPERIMENT_FORMAT, XDNA_BFP_EXPERIMENT_TARGET_IDENTITY,
};
use virtio_accel_xdna::{REQUIRED_RESIDENT_BYTES, XDNA_TOSA_TARGET, XdnaAccelerator};

/// Precompiled flavor-1 design at the envelope ceiling (built by the #146 probe pipeline from
/// the pinned v2026.08 toolchain; regeneration: `research/bfp16ebs8/probe_compile.py xbfp`).
const FIXTURE: &[u8] = include_bytes!("data/xbfp-mxint8-matmul-8x512x8-v1.xbfp");
const K: usize = 512;
const CHUNKS: usize = K / 8;
const GROUPS: usize = K / 32;
const OPERAND_BYTES: u64 = (CHUNKS as u64) * UNIT_BYTES;
const OUTPUT_BYTES: u64 = 256;

const REQUIRE_HARDWARE_ENV: &str = "VIRTIO_ACCEL_XDNA_REQUIRE_HARDWARE";

fn hardware_required() -> bool {
    match std::env::var(REQUIRE_HARDWARE_ENV) {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => panic!("{REQUIRE_HARDWARE_ENV} must be \"0\" or \"1\", not {value:?}"),
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => panic!("invalid {REQUIRE_HARDWARE_ENV}: {error}"),
    }
}

fn backend() -> Option<XdnaAccelerator> {
    match XdnaAccelerator::new() {
        Ok(backend) => Some(backend),
        Err(error) => {
            assert!(
                !hardware_required(),
                "{REQUIRE_HARDWARE_ENV}=1 but the XDNA runtime is unusable: {error}"
            );
            eprintln!("XDNA runtime unavailable ({error}); skipping experiment test");
            None
        }
    }
}

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

/// One operand: 8 rows, each `GROUPS` MXINT8 groups (mantissas plus one scale per group),
/// serialized into the documented `CHUNKS x 72-byte` unit stream.
struct Operand {
    mantissa: [[i8; K]; 8],
    scale: [[u8; GROUPS]; 8],
}

impl Operand {
    fn to_units(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(OPERAND_BYTES as usize);
        for chunk in 0..CHUNKS {
            for row in 0..8 {
                for lane in 0..8 {
                    bytes.push(self.mantissa[row][chunk * 8 + lane] as u8);
                }
            }
            for row in 0..8 {
                bytes.push(self.scale[row][chunk / 4]);
            }
        }
        bytes
    }

    fn fold_row(&self, row: usize) -> (Vec<i8>, Vec<u8>) {
        let mantissa = self.mantissa[row].to_vec();
        let exponent: Vec<u8> = (0..CHUNKS)
            .map(|chunk| self.scale[row][chunk / 4])
            .collect();
        (mantissa, exponent)
    }
}

fn run_and_check(name: &str, a: &Operand, b: &Operand) {
    let Some(backend) = backend() else { return };
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
                format: XDNA_BFP_EXPERIMENT_FORMAT,
                target: XDNA_BFP_EXPERIMENT_TARGET_IDENTITY,
                payload: &Slice(FIXTURE),
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        )
        .expect("load XBFP fixture");

    let operand_desc = BufferDesc::new(
        OPERAND_BYTES,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let (mut a_buffer, _) = backend
        .allocate_buffer(&context, operand_desc)
        .expect("A buffer")
        .into_parts();
    let (mut b_buffer, _) = backend
        .allocate_buffer(&context, operand_desc)
        .expect("B buffer")
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
        .expect("C buffer")
        .into_parts();

    backend
        .write_buffer(&mut a_buffer, 0, &Slice(&a.to_units()))
        .expect("write A");
    backend
        .write_buffer(&mut b_buffer, 0, &Slice(&b.to_units()))
        .expect("write B");

    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &a_buffer,
            range: BufferRange::new(0, OPERAND_BYTES).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &b_buffer,
            range: BufferRange::new(0, OPERAND_BYTES).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 2,
            buffer: &output,
            range: BufferRange::new(0, OUTPUT_BYTES).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
        Ok(event) => event,
        Err(failure) => panic!("{name}: submit rejected: {failure:?}"),
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => panic!("{name}: no completion in 30s"),
            EventState::Failed(error) => panic!("{name}: dispatch failed: {error:?}"),
            _ => break,
        }
    }
    assert!(backend.destroy_event(event).is_ok(), "destroy event");

    let mut raw = vec![0u8; OUTPUT_BYTES as usize];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut raw))
        .expect("read C");

    for i in 0..8 {
        let (a_m, a_e) = a.fold_row(i);
        for j in 0..8 {
            let (b_m, b_e) = b.fold_row(j);
            let expected = bfp_model::dot_fold_f32(&a_m, &a_e, &b_m, &b_e);
            let got = f32::from_le_bytes(
                raw[(i * 8 + j) * 4..(i * 8 + j) * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{name}: lane ({i},{j}): got {got}, expected {expected}"
            );
        }
    }

    assert!(backend.free_buffer(a_buffer).is_ok(), "free A");
    assert!(backend.free_buffer(b_buffer).is_ok(), "free B");
    assert!(backend.free_buffer(output).is_ok(), "free C");
    assert!(backend.unload_program(program).is_ok(), "unload");
    assert!(backend.destroy_queue(queue).is_ok(), "destroy queue");
    assert!(backend.destroy_context(context).is_ok(), "destroy context");
}

/// Guest-side quantization through the vendored OCP MXINT8 model, per row and 32-group.
fn quantized_operand(seed: u32) -> Operand {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 16) as i16 as f32) / 16384.0
    };
    let mut operand = Operand {
        mantissa: [[0i8; K]; 8],
        scale: [[0u8; GROUPS]; 8],
    };
    for row in 0..8 {
        for group in 0..GROUPS {
            let mut values = [0f32; 32];
            for v in values.iter_mut() {
                *v = next();
            }
            let (m, e) = bfp_model::mxint8_quantize_block::<32>(&values);
            operand.mantissa[row][group * 32..group * 32 + 32].copy_from_slice(&m);
            operand.scale[row][group] = e;
        }
    }
    operand
}

/// The full lifecycle on guest-quantized MXINT8 data: bit-exact against the fold oracle.
#[test]
fn mxint8_matmul_matches_the_fold_oracle_on_the_npu() {
    let a = quantized_operand(0x0148_0001);
    let b = quantized_operand(0x0148_0002);
    run_and_check("quantized", &a, &b);
}

/// The accumulation-order case: group 0 large, every later group's chunk contribution just
/// below the running accumulator's FP32 half-ULP.
///
/// IGNORED pending probe P6: the chain matches a per-step FP32 RNE fold on 63 of 64 lanes,
/// but on tie-adjacent steps the accumulator's rounding is provably NOT round-to-nearest-even
/// — and not floor, to-odd, half-away, half-toward-zero, or any wider-precision model either
/// (each is refuted by at least one on-metal observation; crafted ties break toward zero
/// while an organic mid-chain tie broke away, pointing at guard/sticky accumulator state).
/// Until P6 pins the exact rule, the tier's oracle cannot claim tie-adjacent bit-exactness,
/// and this test would encode a falsified model.
#[test]
#[ignore = "accumulator tie rounding not yet characterized (P6)"]
fn accumulation_order_contract_holds_on_the_npu() {
    let mut state = 0x0148_1481u32;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8 as i8
    };
    let mut make = |_: ()| {
        let mut operand = Operand {
            mantissa: [[0i8; K]; 8],
            scale: [[0u8; GROUPS]; 8],
        };
        for row in 0..8 {
            for lane in 0..K {
                operand.mantissa[row][lane] = next();
            }
            for group in 0..GROUPS {
                operand.scale[row][group] = if group == 0 { 130 } else { 118 };
            }
        }
        operand
    };
    let a = make(());
    let b = make(());
    // The case must actually discriminate: the fold differs from a single-rounded f64 sum.
    let (a_m, a_e) = a.fold_row(0);
    let (b_m, b_e) = b.fold_row(0);
    let fold = bfp_model::dot_fold_f32(&a_m, &a_e, &b_m, &b_e);
    let single = bfp_model::dot_reference(&a_m, &a_e, &b_m, &b_e, 8) as f32;
    assert_ne!(
        fold.to_bits(),
        single.to_bits(),
        "schedule failed to make accumulation order observable"
    );
    run_and_check("order-contract", &a, &b);
}

/// The experiment's label is immutable: a load under any other identity is relabeling and is
/// rejected before native resource creation.
#[test]
fn load_rejects_a_foreign_target_identity() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let result = backend.load_program(
        &context,
        ArtifactRef {
            format: XDNA_BFP_EXPERIMENT_FORMAT,
            target: XDNA_TOSA_TARGET.to_identity(),
            payload: &Slice(FIXTURE),
            resident_bytes: REQUIRED_RESIDENT_BYTES,
        },
    );
    assert!(matches!(result, Err(BackendError::Incompatible)));
    backend.destroy_context(context).expect("destroy context");
}

/// The committed fixture stays parseable and inside the flavor-1 envelope.
#[test]
fn fixture_parses_and_derives_the_documented_slot_plan() {
    let parsed = virtio_accel_xdna::bfp_experiment::BfpExperimentArtifact::parse(FIXTURE)
        .expect("fixture parses");
    assert_eq!((parsed.m, parsed.k as usize, parsed.n), (8, K, 8));
    let (inputs, outputs) = parsed.slot_bytes();
    assert_eq!(inputs, [OPERAND_BYTES, OPERAND_BYTES]);
    assert_eq!(outputs, [OUTPUT_BYTES]);
}

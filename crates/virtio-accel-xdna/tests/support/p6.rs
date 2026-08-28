//! Shared scratch harness for P6 exploration.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, Timeout,
};
use virtio_accel_xdna::bfp_experiment::{
    XDNA_BFP_EXPERIMENT_FORMAT, XDNA_BFP_EXPERIMENT_TARGET_IDENTITY,
};
use virtio_accel_xdna::{REQUIRED_RESIDENT_BYTES, XdnaAccelerator};

const FIXTURE: &[u8] = include_bytes!("../data/xbfp-mxint8-matmul-8x512x8-v1.xbfp");
pub const K: usize = 512;
pub const CHUNKS: usize = K / 8;
const OPERAND_BYTES: u64 = (CHUNKS as u64) * 72;
const OUTPUT_BYTES: u64 = 256;

#[derive(Debug)]
struct Slice<'a>(&'a [u8]);
impl ByteSource for Slice<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        dst.copy_from_slice(&self.0[start..start + dst.len()]);
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
        self.0[start..start + src.len()].copy_from_slice(src);
        Ok(())
    }
}

pub struct Planes {
    pub units: Vec<[u8; 72]>,
}

impl Planes {
    pub fn zero() -> Self {
        Self {
            units: vec![[0u8; 72]; CHUNKS],
        }
    }
    /// Set row-0 lane `lane` of chunk `chunk` to mantissa `m` with block exponent `e`.
    pub fn set(&mut self, chunk: usize, lane: usize, m: i8, e: u8) {
        self.units[chunk][lane] = m as u8;
        self.units[chunk][64] = e; // row 0 exponent byte
    }
    pub fn negate_chunk(&mut self, _chunk: usize) {}
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(OPERAND_BYTES as usize);
        for unit in &self.units {
            out.extend_from_slice(unit);
        }
        out
    }
}

pub struct Harness {
    backend: XdnaAccelerator,
}

impl Harness {
    pub fn new() -> Option<Self> {
        match XdnaAccelerator::new() {
            Ok(backend) => Some(Self { backend }),
            Err(error) => {
                eprintln!("XDNA runtime unavailable ({error}); skipping");
                None
            }
        }
    }

    pub fn run_lane00(&self, a: &Planes, b: &Planes) -> f32 {
        let backend = &self.backend;
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
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
            .unwrap();
        let desc = BufferDesc::new(
            OPERAND_BYTES,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap();
        let (mut ab, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (mut bb, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (out, _) = backend
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
            .unwrap()
            .into_parts();
        backend
            .write_buffer(&mut ab, 0, &Slice(&a.bytes()))
            .unwrap();
        backend
            .write_buffer(&mut bb, 0, &Slice(&b.bytes()))
            .unwrap();
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &ab,
                range: BufferRange::new(0, OPERAND_BYTES).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &bb,
                range: BufferRange::new(0, OPERAND_BYTES).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 2,
                buffer: &out,
                range: BufferRange::new(0, OUTPUT_BYTES).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let event = match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
            Ok(event) => event,
            Err(failure) => panic!("submit rejected: {failure:?}"),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match backend.poll_event(&event).unwrap() {
                EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
                EventState::Pending => panic!("timeout"),
                EventState::Failed(error) => panic!("failed: {error:?}"),
                _ => break,
            }
        }
        assert!(backend.destroy_event(event).is_ok());
        let mut raw = vec![0u8; OUTPUT_BYTES as usize];
        backend
            .read_buffer(&out, 0, &mut SliceMut(&mut raw))
            .unwrap();
        let c00 = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert!(backend.free_buffer(ab).is_ok());
        assert!(backend.free_buffer(bb).is_ok());
        assert!(backend.free_buffer(out).is_ok());
        assert!(backend.unload_program(program).is_ok());
        assert!(backend.destroy_queue(queue).is_ok());
        assert!(backend.destroy_context(context).is_ok());
        c00
    }
}

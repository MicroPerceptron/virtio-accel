//! Provider-native NVFP4 matrix products.
//!
//! TOSA has no FP4 tensor type. Encoding packed weights as an integer TOSA
//! graph would hide the operation from provider admission and explode it into
//! scalar nodes, so Vulkan owns a small artifact format for the exact storage
//! contract used by NVFP4 checkpoints. This is an artifact-format extension,
//! not a virtio-accel wire change.

use virtio_accel_core::{ArtifactFormat, ArtifactRef, TargetIdentity};

use crate::lower::{
    DispatchPlan, KernelSpec, LoweringError, ProgramPlan, SlotPlan, SlotRole, Work,
};
use crate::shader::{Operand, Storage, nvfp4_matmul_spec};

pub const VULKAN_NVFP4_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x564e_4634) {
    Some(format) => format,
    None => panic!("nonzero artifact format"),
};

pub const VULKAN_NVFP4_TARGET: TargetIdentity =
    TargetIdentity([0x564b_4e46, 0x5034_0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

const MAGIC: [u8; 8] = *b"VKNVFP4\0";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;

/// Fused projection epilogue. Its integer representation is part of artifact v1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Nvfp4Activation {
    #[default]
    None = 0,
    Silu = 1,
    Sigmoid = 2,
}

/// A validated F32 × NVFP4 projection artifact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nvfp4Artifact {
    bytes: [u8; HEADER_BYTES],
}

impl Nvfp4Artifact {
    /// Describe `[m,k] F32 × [n,k] NVFP4 -> [m,n] F32`.
    ///
    /// The tensor scale is a one-element F32 input rather than artifact metadata, so one loaded
    /// program serves every checkpoint tensor with this geometry.
    pub fn new(m: u32, n: u32, k: u32, activation: Nvfp4Activation) -> Result<Self, LoweringError> {
        if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(16) {
            return Err(LoweringError::UnsupportedGraph);
        }
        checked_lengths(m, n, k)?;
        let mut bytes = [0; HEADER_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        for (offset, value) in [VERSION, m, n, k, activation as u32, 0]
            .into_iter()
            .enumerate()
        {
            bytes[8 + offset * 4..12 + offset * 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_ref(&self) -> ArtifactRef<'_> {
        ArtifactRef {
            format: VULKAN_NVFP4_FORMAT,
            target: VULKAN_NVFP4_TARGET,
            payload: &self.bytes,
            resident_bytes: crate::REQUIRED_RESIDENT_BYTES,
        }
    }
}

fn checked_lengths(m: u32, n: u32, k: u32) -> Result<[u64; 5], LoweringError> {
    let product = |a: u32, b: u32, bytes: u64| {
        u64::from(a)
            .checked_mul(u64::from(b))
            .and_then(|v| v.checked_mul(bytes))
            .ok_or(LoweringError::ResourceLimit)
    };
    Ok([
        product(m, k, 4)?,
        product(n, k, 1)? / 2,
        product(n, k, 1)? / 16,
        4,
        product(m, n, 4)?,
    ])
}

fn word(bytes: &[u8], offset: usize) -> Result<u32, LoweringError> {
    bytes
        .get(offset..offset + 4)
        .and_then(|v| v.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or(LoweringError::UnsupportedGraph)
}

pub(crate) fn lower_nvfp4(bytes: &[u8]) -> Result<ProgramPlan, LoweringError> {
    if bytes.len() != HEADER_BYTES || bytes[..8] != MAGIC {
        return Err(LoweringError::UnsupportedGraph);
    }
    let version = word(bytes, 8)?;
    let m = word(bytes, 12)?;
    let n = word(bytes, 16)?;
    let k = word(bytes, 20)?;
    let activation = word(bytes, 24)?;
    let reserved = word(bytes, 28)?;
    if version != VERSION
        || activation > Nvfp4Activation::Sigmoid as u32
        || reserved != 0
        || m == 0
        || n == 0
        || k == 0
        || !k.is_multiple_of(16)
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let lengths = checked_lengths(m, n, k)?;
    let operand = |slot| Operand {
        buffer: slot,
        base: 0,
    };
    Ok(ProgramPlan {
        slots: vec![
            SlotPlan {
                slot: 0,
                role: SlotRole::Input,
                byte_len: lengths[0],
                storage: Storage::Word,
            },
            SlotPlan {
                slot: 1,
                role: SlotRole::Input,
                byte_len: lengths[1],
                storage: Storage::Byte,
            },
            SlotPlan {
                slot: 2,
                role: SlotRole::Input,
                byte_len: lengths[2],
                storage: Storage::Byte,
            },
            SlotPlan {
                slot: 3,
                role: SlotRole::Input,
                byte_len: lengths[3],
                storage: Storage::Word,
            },
            SlotPlan {
                slot: 4,
                role: SlotRole::Output,
                byte_len: lengths[4],
                storage: Storage::Word,
            },
        ],
        arena_bytes: 0,
        constants: Vec::new(),
        dispatches: vec![DispatchPlan {
            kernel: KernelSpec::Nvfp4Matmul,
            spec: nvfp4_matmul_spec(
                operand(0),
                operand(1),
                operand(2),
                operand(3),
                operand(4),
                m,
                n,
                k,
                activation,
            ),
            work: Work::Nvfp4Matmul { m, n },
            barrier_before: false,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_round_trips_into_exact_buffer_contract() {
        let artifact = Nvfp4Artifact::new(3, 17, 2688, Nvfp4Activation::Silu).unwrap();
        let plan = lower_nvfp4(artifact.as_bytes()).unwrap();
        assert_eq!(
            plan.slots
                .iter()
                .map(|slot| slot.byte_len)
                .collect::<Vec<_>>(),
            vec![3 * 2688 * 4, 17 * 2688 / 2, 17 * 2688 / 16, 4, 3 * 17 * 4,]
        );
        assert_eq!(plan.dispatches[0].work, Work::Nvfp4Matmul { m: 3, n: 17 });
    }

    #[test]
    fn malformed_geometry_is_rejected() {
        assert!(Nvfp4Artifact::new(1, 1, 15, Nvfp4Activation::None).is_err());
    }
}

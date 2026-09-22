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
    /// `max(x, 0)^2` after the projection has applied `tensor_scale` once.
    /// A fused routed-expert caller folds the omitted second scale squared
    /// into the following projection's tensor scale. This keeps the private
    /// intermediate normal without changing the composed result.
    SquaredRelu = 3,
}

const MOE_MAGIC: [u8; 8] = *b"VKNVMOE\0";
const MOE_HEADER_BYTES: usize = 160;

/// Two native NVFP4 expert projections with a device-local squared-ReLU
/// intermediate. Each expert occupies one externally bound cache-slot span;
/// the four offsets identify its up/down packed and block-scale planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nvfp4MoeArtifact {
    bytes: [u8; MOE_HEADER_BYTES],
}

impl Nvfp4MoeArtifact {
    pub fn new(
        batch: u32,
        width: u32,
        inner: u32,
        expert_bytes: u32,
        offsets: &[[u32; 4]],
    ) -> Result<Self, LoweringError> {
        if batch == 0
            || batch > 6
            || offsets.len() != batch as usize
            || !width.is_multiple_of(16)
            || !inner.is_multiple_of(16)
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        let fits = |offset: u32, bytes: u64| {
            u64::from(offset)
                .checked_add(bytes)
                .is_some_and(|last| last <= u64::from(expert_bytes))
        };
        for &[up_packed, up_scales, down_packed, down_scales] in offsets {
            if [up_packed, up_scales, down_packed, down_scales]
                .into_iter()
                .any(|offset| !offset.is_multiple_of(4))
                || !fits(up_packed, u64::from(inner) * u64::from(width) / 2)
                || !fits(up_scales, u64::from(inner) * u64::from(width) / 16)
                || !fits(down_packed, u64::from(width) * u64::from(inner) / 2)
                || !fits(down_scales, u64::from(width) * u64::from(inner) / 16)
            {
                return Err(LoweringError::ResourceLimit);
            }
        }
        let mut bytes = [0; MOE_HEADER_BYTES];
        bytes[..8].copy_from_slice(&MOE_MAGIC);
        for (index, value) in [VERSION, batch, width, inner, expert_bytes]
            .into_iter()
            .enumerate()
        {
            bytes[8 + index * 4..12 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        for (index, value) in offsets.iter().flatten().copied().enumerate() {
            bytes[28 + index * 4..32 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self { bytes })
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
        Self::build(m, n, k, activation, 0)
    }

    /// Describe `batch` independent `[k] F32 × [n,k] NVFP4 products.
    ///
    /// Activations, packed weights, block scales and tensor scales all carry
    /// the leading batch dimension. This is the natural execution unit for a
    /// routed MoE layer and amortizes submission over all selected experts.
    pub fn new_batched(
        batch: u32,
        n: u32,
        k: u32,
        activation: Nvfp4Activation,
    ) -> Result<Self, LoweringError> {
        Self::build(batch, n, k, activation, 1)
    }

    /// Describe an expert batch whose packed weights and block scales are
    /// separate bindings. This retains zero-copy imports for independently
    /// cached experts while still issuing one dispatch. The Vulkan descriptor
    /// budget permits up to six experts.
    pub fn new_expert_batch(
        batch: u32,
        n: u32,
        k: u32,
        activation: Nvfp4Activation,
    ) -> Result<Self, LoweringError> {
        Self::build(batch, n, k, activation, 2)
    }

    fn build(
        m: u32,
        n: u32,
        k: u32,
        activation: Nvfp4Activation,
        weight_mode: u32,
    ) -> Result<Self, LoweringError> {
        if m == 0
            || n == 0
            || k == 0
            || !k.is_multiple_of(16)
            || weight_mode > 2
            || (weight_mode == 2 && m > 6)
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        checked_lengths(m, n, k, weight_mode != 0)?;
        let mut bytes = [0; HEADER_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        for (offset, value) in [VERSION, m, n, k, activation as u32, weight_mode]
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

fn checked_lengths(
    m: u32,
    n: u32,
    k: u32,
    batched_weights: bool,
) -> Result<[u64; 5], LoweringError> {
    let product = |a: u32, b: u32, bytes: u64| {
        u64::from(a)
            .checked_mul(u64::from(b))
            .and_then(|v| v.checked_mul(bytes))
            .ok_or(LoweringError::ResourceLimit)
    };
    let batches = if batched_weights { m } else { 1 };
    Ok([
        product(m, k, 4)?,
        product(n, k, u64::from(batches))? / 2,
        product(n, k, u64::from(batches))? / 16,
        u64::from(batches) * 4,
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
    if bytes.len() == MOE_HEADER_BYTES && bytes[..8] == MOE_MAGIC {
        return lower_nvfp4_moe(bytes);
    }
    if bytes.len() != HEADER_BYTES || bytes[..8] != MAGIC {
        return Err(LoweringError::UnsupportedGraph);
    }
    let version = word(bytes, 8)?;
    let m = word(bytes, 12)?;
    let n = word(bytes, 16)?;
    let k = word(bytes, 20)?;
    let activation = word(bytes, 24)?;
    let weight_mode = word(bytes, 28)?;
    if version != VERSION
        || activation > Nvfp4Activation::SquaredRelu as u32
        || weight_mode > 2
        || (weight_mode == 2 && m > 6)
        || m == 0
        || n == 0
        || k == 0
        || !k.is_multiple_of(16)
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let lengths = checked_lengths(m, n, k, weight_mode != 0)?;
    let operand = |slot| Operand {
        buffer: slot,
        base: 0,
    };
    if weight_mode == 2 {
        let packed_bytes = u64::from(n) * u64::from(k) / 2;
        let scale_bytes = u64::from(n) * u64::from(k) / 16;
        let mut slots = vec![SlotPlan {
            slot: 0,
            role: SlotRole::Input,
            byte_len: lengths[0],
            storage: Storage::Word,
        }];
        let mut packed = Vec::with_capacity(m as usize);
        let mut scales = Vec::with_capacity(m as usize);
        for expert in 0..m {
            let packed_slot = 1 + expert * 2;
            let scale_slot = packed_slot + 1;
            slots.push(SlotPlan {
                slot: packed_slot,
                role: SlotRole::Input,
                byte_len: packed_bytes,
                storage: Storage::Byte,
            });
            slots.push(SlotPlan {
                slot: scale_slot,
                role: SlotRole::Input,
                byte_len: scale_bytes,
                storage: Storage::Byte,
            });
            packed.push(operand(packed_slot));
            scales.push(operand(scale_slot));
        }
        let tensor_scale_slot = 1 + m * 2;
        let output_slot = tensor_scale_slot + 1;
        slots.push(SlotPlan {
            slot: tensor_scale_slot,
            role: SlotRole::Input,
            byte_len: u64::from(m) * 4,
            storage: Storage::Word,
        });
        slots.push(SlotPlan {
            slot: output_slot,
            role: SlotRole::Output,
            byte_len: lengths[4],
            storage: Storage::Word,
        });
        return Ok(ProgramPlan {
            slots,
            arena_bytes: 0,
            constants: Vec::new(),
            dispatches: vec![DispatchPlan {
                kernel: KernelSpec::Nvfp4Matmul { cooperative: false },
                spec: nvfp4_matmul_spec(
                    operand(0),
                    &packed,
                    &scales,
                    operand(tensor_scale_slot),
                    operand(output_slot),
                    m,
                    n,
                    k,
                    activation,
                    weight_mode,
                ),
                work: Work::Nvfp4Matmul { m, n },
                barrier_before: false,
            }],
        });
    }
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
            kernel: KernelSpec::Nvfp4Matmul {
                cooperative: weight_mode == 0 && m >= 8,
            },
            spec: nvfp4_matmul_spec(
                operand(0),
                &[operand(1)],
                &[operand(2)],
                operand(3),
                operand(4),
                m,
                n,
                k,
                activation,
                weight_mode,
            ),
            work: Work::Nvfp4Matmul { m, n },
            barrier_before: false,
        }],
    })
}

fn lower_nvfp4_moe(bytes: &[u8]) -> Result<ProgramPlan, LoweringError> {
    let values: Vec<u32> = (0..5)
        .map(|i| word(bytes, 8 + i * 4))
        .collect::<Result<_, _>>()?;
    let [version, batch, width, inner, expert_bytes] = values.as_slice() else {
        return Err(LoweringError::UnsupportedGraph);
    };
    if *version != VERSION || *batch == 0 || *batch > 6 || *expert_bytes == 0 {
        return Err(LoweringError::UnsupportedGraph);
    }
    let offsets = (0..*batch as usize)
        .map(|expert| {
            Ok([
                word(bytes, 28 + (expert * 4) * 4)?,
                word(bytes, 28 + (expert * 4 + 1) * 4)?,
                word(bytes, 28 + (expert * 4 + 2) * 4)?,
                word(bytes, 28 + (expert * 4 + 3) * 4)?,
            ])
        })
        .collect::<Result<Vec<[u32; 4]>, LoweringError>>()?;
    let artifact = Nvfp4MoeArtifact::new(*batch, *width, *inner, *expert_bytes, &offsets)?;
    let _ = artifact;
    let activation_bytes = u64::from(*batch) * u64::from(*width) * 4;
    let intermediate_bytes = u64::from(*batch) * u64::from(*inner) * 4;
    let output_bytes = activation_bytes;
    let mut slots = vec![SlotPlan {
        slot: 0,
        role: SlotRole::Input,
        byte_len: activation_bytes,
        storage: Storage::Word,
    }];
    for expert in 0..*batch {
        slots.push(SlotPlan {
            slot: 1 + expert,
            role: SlotRole::Input,
            byte_len: u64::from(*expert_bytes),
            storage: Storage::Byte,
        });
    }
    let up_tensor_slot = 1 + *batch;
    let down_tensor_slot = up_tensor_slot + 1;
    let output_slot = down_tensor_slot + 1;
    for slot in [up_tensor_slot, down_tensor_slot] {
        slots.push(SlotPlan {
            slot,
            role: SlotRole::Input,
            byte_len: u64::from(*batch) * 4,
            storage: Storage::Word,
        });
    }
    slots.push(SlotPlan {
        slot: output_slot,
        role: SlotRole::Output,
        byte_len: output_bytes,
        storage: Storage::Word,
    });
    let arena_slot = slots.len() as u32;
    let operands = |plane: usize| {
        (0..*batch)
            .map(|expert| Operand {
                buffer: 1 + expert,
                // SPIR-V storage operands index 32-bit words; the artifact
                // exposes byte offsets to match host tensor views.
                base: offsets[expert as usize][plane] / 4,
            })
            .collect::<Vec<_>>()
    };
    Ok(ProgramPlan {
        slots,
        arena_bytes: intermediate_bytes,
        constants: Vec::new(),
        dispatches: vec![
            DispatchPlan {
                kernel: KernelSpec::Nvfp4Matmul { cooperative: false },
                spec: nvfp4_matmul_spec(
                    Operand { buffer: 0, base: 0 },
                    &operands(0),
                    &operands(1),
                    Operand {
                        buffer: up_tensor_slot,
                        base: 0,
                    },
                    Operand {
                        buffer: arena_slot,
                        base: 0,
                    },
                    *batch,
                    *inner,
                    *width,
                    Nvfp4Activation::SquaredRelu as u32,
                    2,
                ),
                work: Work::Nvfp4Matmul {
                    m: *batch,
                    n: *inner,
                },
                barrier_before: false,
            },
            DispatchPlan {
                kernel: KernelSpec::Nvfp4Matmul { cooperative: false },
                spec: nvfp4_matmul_spec(
                    Operand {
                        buffer: arena_slot,
                        base: 0,
                    },
                    &operands(2),
                    &operands(3),
                    Operand {
                        buffer: down_tensor_slot,
                        base: 0,
                    },
                    Operand {
                        buffer: output_slot,
                        base: 0,
                    },
                    *batch,
                    *width,
                    *inner,
                    Nvfp4Activation::None as u32,
                    2,
                ),
                work: Work::Nvfp4Matmul {
                    m: *batch,
                    n: *width,
                },
                barrier_before: true,
            },
        ],
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
    fn batched_artifact_batches_weights_scales_and_tensor_scales() {
        let artifact = Nvfp4Artifact::new_batched(6, 1856, 2688, Nvfp4Activation::None).unwrap();
        let plan = lower_nvfp4(artifact.as_bytes()).unwrap();
        assert_eq!(
            plan.slots
                .iter()
                .map(|slot| slot.byte_len)
                .collect::<Vec<_>>(),
            vec![
                6 * 2688 * 4,
                6 * 1856 * 2688 / 2,
                6 * 1856 * 2688 / 16,
                6 * 4,
                6 * 1856 * 4,
            ]
        );
    }

    #[test]
    fn expert_batch_keeps_each_weight_pair_in_its_own_binding() {
        let artifact =
            Nvfp4Artifact::new_expert_batch(6, 1856, 2688, Nvfp4Activation::None).unwrap();
        let plan = lower_nvfp4(artifact.as_bytes()).unwrap();
        assert_eq!(plan.slots.len(), 15);
        assert_eq!(plan.slots[0].byte_len, 6 * 2688 * 4);
        for expert in 0..6 {
            assert_eq!(plan.slots[1 + expert * 2].byte_len, 1856 * 2688 / 2);
            assert_eq!(plan.slots[2 + expert * 2].byte_len, 1856 * 2688 / 16);
        }
        assert_eq!(plan.slots[13].byte_len, 6 * 4);
        assert_eq!(plan.slots[14].byte_len, 6 * 1856 * 4);
    }

    #[test]
    fn fused_moe_uses_one_expert_binding_and_a_private_intermediate() {
        let width = 2688;
        let inner = 1856;
        let up_bytes = inner * width / 2;
        let down_bytes = width * inner / 2;
        let scale_bytes = inner * width / 16;
        let artifact = Nvfp4MoeArtifact::new(
            6,
            width,
            inner,
            6 << 20,
            &[[0, up_bytes, 3 << 20, (3 << 20) + down_bytes]; 6],
        )
        .unwrap();
        assert!(scale_bytes < 1 << 20);
        let plan = lower_nvfp4(&artifact.bytes).unwrap();
        assert_eq!(plan.slots.len(), 10);
        assert_eq!(plan.arena_bytes, 6 * u64::from(inner) * 4);
        assert_eq!(plan.dispatches.len(), 2);
        assert!(!plan.dispatches[0].barrier_before);
        assert!(plan.dispatches[1].barrier_before);
        // First packed operand follows the activation operand in the
        // specialization payload and names byte offset zero in words.
        assert_eq!(plan.dispatches[0].spec[2..4], [1, 0]);
        // The first down packed operand begins at 3 MiB, encoded as words.
        assert_eq!(plan.dispatches[1].spec[2..4], [1, (3 << 20) / 4]);
    }

    #[test]
    fn malformed_geometry_is_rejected() {
        assert!(Nvfp4Artifact::new(1, 1, 15, Nvfp4Activation::None).is_err());
    }
}

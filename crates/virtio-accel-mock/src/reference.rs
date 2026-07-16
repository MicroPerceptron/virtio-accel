//! Test-only artifact format understood by [`crate::MockAccelerator`].
//!
//! This format exists solely to make the reference backend's execution observable and
//! deterministic. It is not part of the virtio-accel ABI, and production transports must continue
//! to treat provider artifacts as opaque bytes.

use virtio_accel_core::{ArtifactFormat, BackendError, ByteSource, TargetIdentity};

const MAGIC: [u8; 4] = *b"VAMK";
const VERSION_MAJOR: u8 = 1;
const VERSION_MINOR: u8 = 0;
const BINDING_ABI_VERSION: u8 = 1;
const UNUSED_SLOT: u32 = u32::MAX;

const OP_BARRIER: u8 = 0;
const OP_COPY: u8 = 1;
const OP_FILL: u8 = 2;
const OP_XOR: u8 = 3;

/// Provider-owned format ID reserved by the reference backend for its test artifact.
pub const ARTIFACT_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x5641_4d4b) {
    Some(format) => format,
    None => panic!("reference artifact format must be nonzero"),
};

/// Provider-owned target identity for the reference execution model.
pub const TARGET_IDENTITY: TargetIdentity = TargetIdentity([
    0x5641_4d4b,
    0x0001_0000,
    0x5445_5354,
    0x4f4e_4c59,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]);

/// Exact encoded payload size for every reference artifact.
pub const ARTIFACT_BYTES: usize = 24;

/// Exact resident charge required by the reference backend.
pub const RESIDENT_BYTES: u64 = 64;

/// A validated reference operation embedded in a test artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operation {
    Barrier { slot: u32 },
    Copy { source_slot: u32, target_slot: u32 },
    Fill { target_slot: u32, value: u8 },
    Xor { target_slot: u32, value: u8 },
}

/// Fixed-size builder for the mock backend's test-only executable format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceArtifact {
    bytes: [u8; ARTIFACT_BYTES],
}

impl ReferenceArtifact {
    /// Build a no-op submission used to exercise lifecycle and synchronization behavior.
    ///
    /// The operation requires exactly one binding at `slot`, with any otherwise-valid access.
    pub fn barrier(slot: u32) -> Self {
        Self::new(OP_BARRIER, slot, UNUSED_SLOT, 0)
    }

    /// Build a byte-for-byte copy from a `Read` binding to an equal-length `Write` binding.
    pub fn copy(source_slot: u32, target_slot: u32) -> Result<Self, BackendError> {
        if source_slot == target_slot {
            return Err(BackendError::InvalidArgument);
        }
        Ok(Self::new(OP_COPY, source_slot, target_slot, 0))
    }

    /// Build a fill of the complete `Write` range bound at `target_slot`.
    pub fn fill(target_slot: u32, value: u8) -> Self {
        Self::new(OP_FILL, target_slot, UNUSED_SLOT, value)
    }

    /// Build an in-place XOR of the complete `ReadWrite` range bound at `target_slot`.
    pub fn xor(target_slot: u32, value: u8) -> Self {
        Self::new(OP_XOR, target_slot, UNUSED_SLOT, value)
    }

    pub const fn as_bytes(&self) -> &[u8; ARTIFACT_BYTES] {
        &self.bytes
    }

    fn new(opcode: u8, first_slot: u32, second_slot: u32, operand: u8) -> Self {
        let mut bytes = [0; ARTIFACT_BYTES];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = VERSION_MAJOR;
        bytes[5] = VERSION_MINOR;
        bytes[6] = opcode;
        bytes[7] = BINDING_ABI_VERSION;
        bytes[8..12].copy_from_slice(&first_slot.to_le_bytes());
        bytes[12..16].copy_from_slice(&second_slot.to_le_bytes());
        bytes[16] = operand;
        Self { bytes }
    }
}

impl AsRef<[u8]> for ReferenceArtifact {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

pub(crate) fn decode(payload: &dyn ByteSource) -> Result<Operation, BackendError> {
    if payload.len() != ARTIFACT_BYTES as u64 {
        return Err(BackendError::InvalidArgument);
    }

    let mut bytes = [0; ARTIFACT_BYTES];
    payload.read_at(0, &mut bytes)?;
    if bytes[0..4] != MAGIC {
        return Err(BackendError::InvalidArgument);
    }
    if bytes[4] != VERSION_MAJOR || bytes[5] != VERSION_MINOR || bytes[7] != BINDING_ABI_VERSION {
        return Err(BackendError::Incompatible);
    }
    if bytes[17..].iter().any(|byte| *byte != 0) {
        return Err(BackendError::InvalidArgument);
    }

    let first_slot = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let second_slot = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let operand = bytes[16];
    match bytes[6] {
        OP_BARRIER if second_slot == UNUSED_SLOT && operand == 0 => {
            Ok(Operation::Barrier { slot: first_slot })
        }
        OP_COPY if first_slot != second_slot && operand == 0 => Ok(Operation::Copy {
            source_slot: first_slot,
            target_slot: second_slot,
        }),
        OP_FILL if second_slot == UNUSED_SLOT => Ok(Operation::Fill {
            target_slot: first_slot,
            value: operand,
        }),
        OP_XOR if second_slot == UNUSED_SLOT => Ok(Operation::Xor {
            target_slot: first_slot,
            value: operand,
        }),
        OP_BARRIER | OP_COPY | OP_FILL | OP_XOR => Err(BackendError::InvalidArgument),
        _ => Err(BackendError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_round_trip_through_the_fixed_envelope() {
        for (artifact, expected) in [
            (
                ReferenceArtifact::barrier(9),
                Operation::Barrier { slot: 9 },
            ),
            (
                ReferenceArtifact::copy(2, 7).unwrap(),
                Operation::Copy {
                    source_slot: 2,
                    target_slot: 7,
                },
            ),
            (
                ReferenceArtifact::fill(3, 0xa5),
                Operation::Fill {
                    target_slot: 3,
                    value: 0xa5,
                },
            ),
            (
                ReferenceArtifact::xor(4, 0x5a),
                Operation::Xor {
                    target_slot: 4,
                    value: 0x5a,
                },
            ),
        ] {
            assert_eq!(decode(artifact.as_bytes()), Ok(expected));
        }
        assert_eq!(
            ReferenceArtifact::copy(2, 2),
            Err(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn malformed_and_incompatible_envelopes_are_distinguished() {
        let mut malformed = ReferenceArtifact::barrier(0).bytes;
        malformed[17] = 1;
        assert_eq!(decode(&malformed), Err(BackendError::InvalidArgument));

        let mut incompatible = ReferenceArtifact::barrier(0).bytes;
        incompatible[4] += 1;
        assert_eq!(decode(&incompatible), Err(BackendError::Incompatible));

        let mut unsupported = ReferenceArtifact::barrier(0).bytes;
        unsupported[6] = 0xff;
        assert_eq!(decode(&unsupported), Err(BackendError::Unsupported));
    }
}

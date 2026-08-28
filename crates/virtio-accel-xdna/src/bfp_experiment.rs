//! AMD `bfp16ebs8` vendor experiment (block-8): the `XBFP` artifact container.
//!
//! This is a backend-local **experiment**, not a TOSA capability: it is never advertised
//! through `TosaCapabilityProvider`, never accepted under the stable TOSA artifact format, and
//! creates no protocol value. Design and boundaries:
//! `docs/plans/issue-148-bfp16ebs8-vendor-tier.md`; the silicon-characterized numerical
//! contract it exposes: `docs/research/amdxdna-bfp16ebs8-characterization.md` (issue #146).
//!
//! Flavor 1 is a block-scaled MATMUL with MXINT8 semantics executed on the proven block-8
//! decomposition: `C[8,N=8] (FP32) = A[8,K] · B[8,K]ᵀ`, `K ∈ {32, 64, …, 512}`. Operand
//! planes are streams of 72-byte `v64bfp16ebs8` units (64 two's-complement int8 mantissas,
//! then 8 exponent bytes); MXINT8 semantics require the four exponent bytes of each 32-group
//! to be equal, and `e = 255` (the hardware's structural Inf/NaN space) is outside the
//! contract. Accumulation is FP32 in ascending-`k` chain order — the guest-visible oracle is
//! an FP32 fold in exactly that order, proven bit-exact on the reference NPU.

use virtio_accel_core::{ArtifactFormat, BackendError, TargetIdentity};

/// The experiment's artifact format word (`"XBFP"` read as a big-endian u32). Deliberately
/// distinct from [`crate::XDNA_PRECOMPILED_FORMAT`] and from released TOSA.
pub const XDNA_BFP_EXPERIMENT_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x5842_4650) {
    Some(format) => format,
    None => unreachable!(),
};

/// The experiment's own target identity. A load must present exactly this identity: the
/// numerical label is immutable, and no TOSA target may alias it.
pub const XDNA_BFP_EXPERIMENT_TARGET_IDENTITY: TargetIdentity = TargetIdentity([
    u32::from_le_bytes(*b"XBFP"),
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]);

const MAGIC: [u8; 4] = *b"XBFP";
const VERSION: u32 = 1;
const FLAVOR_MXINT8_MATMUL: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8;

/// One 72-byte `v64bfp16ebs8` unit: 64 mantissa bytes then 8 exponent bytes.
pub const UNIT_BYTES: u64 = 72;

/// A parsed, envelope-validated `XBFP` container.
#[derive(Debug)]
pub struct BfpExperimentArtifact<'a> {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub xclbin: &'a [u8],
    pub insts: &'a [u8],
}

impl<'a> BfpExperimentArtifact<'a> {
    /// Parse and validate the container framing and the flavor-1 shape envelope. Everything is
    /// rejected before any native resource exists; malformed framing is the guest's mistake
    /// (`InvalidArgument`), a well-formed container outside the envelope is `Unsupported`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BackendError> {
        if bytes.len() < HEADER_LEN || bytes[0..4] != MAGIC {
            return Err(BackendError::InvalidArgument);
        }
        let word =
            |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("header word"));
        if word(4) != VERSION {
            return Err(BackendError::Incompatible);
        }
        if word(8) != FLAVOR_MXINT8_MATMUL {
            return Err(BackendError::Unsupported);
        }
        let (m, k, n) = (word(12), word(16), word(20));
        let xclbin_len = u64::from_le_bytes(bytes[24..32].try_into().expect("header word"));
        let insts_len = u64::from_le_bytes(bytes[32..40].try_into().expect("header word"));

        let xclbin_end = (HEADER_LEN as u64)
            .checked_add(xclbin_len)
            .ok_or(BackendError::InvalidArgument)?;
        let total = xclbin_end
            .checked_add(insts_len)
            .ok_or(BackendError::InvalidArgument)?;
        if total != bytes.len() as u64 || insts_len % 4 != 0 || xclbin_len == 0 || insts_len == 0 {
            return Err(BackendError::InvalidArgument);
        }

        // Flavor-1 envelope: the silicon-proven one-worker shape (see the plan's envelope
        // section). Anything else is rejected instead of approximated.
        if m != 8 || n != 8 || !(32..=512).contains(&k) || k % 32 != 0 {
            return Err(BackendError::Unsupported);
        }

        let xclbin_end = usize::try_from(xclbin_end).map_err(|_| BackendError::InvalidArgument)?;
        Ok(Self {
            m,
            k,
            n,
            xclbin: &bytes[HEADER_LEN..xclbin_end],
            insts: &bytes[xclbin_end..],
        })
    }

    /// Derived slot plan — never self-declared by the artifact. Slot 0 is A (`k/8` units),
    /// slot 1 is B (`k/8` units), slot 2 is C (`m·n` FP32 lanes).
    pub fn slot_bytes(&self) -> ([u64; 2], [u64; 1]) {
        let operand = u64::from(self.k) / 8 * UNIT_BYTES;
        let output = u64::from(self.m) * u64::from(self.n) * 4;
        ([operand, operand], [output])
    }

    /// Translate into the crate's internal precompiled container, so loading reuses the one
    /// audited executable-construction path (`artifact::parse` + `build_executable`).
    pub fn to_precompiled_container(&self) -> Vec<u8> {
        let (inputs, outputs) = self.slot_bytes();
        crate::artifact::encode("MLIR_AIE", &inputs, &outputs, self.xclbin, self.insts)
    }
}

/// Build a flavor-1 `XBFP` container. Offline tooling and tests only; the serving path never
/// encodes.
pub fn encode(m: u32, k: u32, n: u32, xclbin: &[u8], insts: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + xclbin.len() + insts.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&FLAVOR_MXINT8_MATMUL.to_le_bytes());
    out.extend_from_slice(&m.to_le_bytes());
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&(xclbin.len() as u64).to_le_bytes());
    out.extend_from_slice(&(insts.len() as u64).to_le_bytes());
    out.extend_from_slice(xclbin);
    out.extend_from_slice(insts);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(k: u32) -> Vec<u8> {
        encode(8, k, 8, &[0xAA; 16], &[0xBB; 8])
    }

    #[test]
    fn round_trips_and_derives_the_slot_plan() {
        let bytes = sample(512);
        let parsed = BfpExperimentArtifact::parse(&bytes).expect("valid container");
        assert_eq!((parsed.m, parsed.k, parsed.n), (8, 512, 8));
        assert_eq!(parsed.xclbin, &[0xAA; 16]);
        assert_eq!(parsed.insts, &[0xBB; 8]);
        let (inputs, outputs) = parsed.slot_bytes();
        assert_eq!(inputs, [4608, 4608]);
        assert_eq!(outputs, [256]);

        let container = parsed.to_precompiled_container();
        let inner =
            crate::artifact::PrecompiledArtifact::parse(&container).expect("valid translation");
        assert_eq!(inner.entry, "MLIR_AIE");
        assert_eq!(inner.slot_bytes, [4608, 4608, 256]);
        assert_eq!((inner.inputs, inner.outputs), (2, 1));
    }

    #[test]
    fn rejects_bad_magic_version_flavor_and_framing() {
        let mut bytes = sample(64);
        bytes[0] = b'Y';
        assert_eq!(
            BfpExperimentArtifact::parse(&bytes).unwrap_err(),
            BackendError::InvalidArgument
        );
        let mut bytes = sample(64);
        bytes[4] = 9;
        assert_eq!(
            BfpExperimentArtifact::parse(&bytes).unwrap_err(),
            BackendError::Incompatible
        );
        let mut bytes = sample(64);
        bytes[8] = 2;
        assert_eq!(
            BfpExperimentArtifact::parse(&bytes).unwrap_err(),
            BackendError::Unsupported
        );
        let bytes = sample(64);
        assert_eq!(
            BfpExperimentArtifact::parse(&bytes[..bytes.len() - 1]).unwrap_err(),
            BackendError::InvalidArgument
        );
        assert_eq!(
            BfpExperimentArtifact::parse(&bytes[..HEADER_LEN - 1]).unwrap_err(),
            BackendError::InvalidArgument
        );
    }

    #[test]
    fn rejects_shapes_outside_the_proven_envelope() {
        for (m, k, n) in [
            (8, 24, 8),
            (8, 544, 8),
            (8, 33, 8),
            (16, 64, 8),
            (8, 64, 16),
        ] {
            let bytes = encode(m, k, n, &[1; 4], &[2; 4]);
            assert_eq!(
                BfpExperimentArtifact::parse(&bytes).unwrap_err(),
                BackendError::Unsupported,
                "m={m} k={k} n={n}"
            );
        }
    }

    #[test]
    fn format_and_identity_collide_with_nothing_released() {
        assert_ne!(
            XDNA_BFP_EXPERIMENT_FORMAT,
            crate::artifact::XDNA_PRECOMPILED_FORMAT
        );
        assert_ne!(
            XDNA_BFP_EXPERIMENT_FORMAT.get(),
            virtio_accel_tosa::ARTIFACT_FORMAT.get()
        );
        for target in [
            crate::XDNA_TOSA_TARGET,
            crate::XDNA_TOSA_INTEGER_TARGET,
            crate::XDNA_TOSA_FP8_TARGET,
        ] {
            assert_ne!(target.to_identity(), XDNA_BFP_EXPERIMENT_TARGET_IDENTITY);
        }
    }
}

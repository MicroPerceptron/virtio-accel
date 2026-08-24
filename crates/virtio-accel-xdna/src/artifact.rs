//! The crate-local precompiled XDNA artifact format.
//!
//! Until TOSA compilation exists, the backend accepts a self-describing container carrying a
//! prebuilt `final.xclbin` and its unfolded `insts.bin` (the XAie transaction stream), the entry
//! point name, and the input/output binding counts. This format is portable and `unsafe`-free: a
//! host or tool builds it with [`encode`] and the backend parses it with [`PrecompiledArtifact::parse`].
//!
//! Layout (little-endian), header then payloads:
//!
//! | field | bytes |
//! |---|---|
//! | magic `b"XDNP"` | 4 |
//! | version (`= 1`) | 4 |
//! | input count | 4 |
//! | output count | 4 |
//! | entry-name length | 4 |
//! | xclbin length | 8 |
//! | insts length | 8 |
//! | entry-name bytes | entry-name length |
//! | xclbin bytes | xclbin length |
//! | insts bytes | insts length |

use virtio_accel_core::{ArtifactFormat, BackendError};

/// Artifact-format tag for a precompiled XDNA container ("XDNP" in ASCII).
pub const XDNA_PRECOMPILED_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x5844_4e50) {
    Some(format) => format,
    None => unreachable!(),
};

const MAGIC: [u8; 4] = *b"XDNP";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 4 + 4 + 4 + 8 + 8;

/// A parsed precompiled artifact: borrowed views into the container bytes.
#[derive(Clone, Copy, Debug)]
pub struct PrecompiledArtifact<'a> {
    pub xclbin: &'a [u8],
    pub insts: &'a [u8],
    pub entry: &'a str,
    pub inputs: usize,
    pub outputs: usize,
}

impl<'a> PrecompiledArtifact<'a> {
    /// Parse and validate a container. Malformed input is `InvalidArgument`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BackendError> {
        if bytes.len() < HEADER_LEN || bytes[0..4] != MAGIC {
            return Err(BackendError::InvalidArgument);
        }
        let u32_at = |offset: usize| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
        };
        let u64_at = |offset: usize| {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("8 bytes"))
        };
        if u32_at(4) != VERSION {
            return Err(BackendError::Incompatible);
        }
        let inputs = u32_at(8) as usize;
        let outputs = u32_at(12) as usize;
        let entry_len = u32_at(16) as usize;
        let xclbin_len = usize::try_from(u64_at(20)).map_err(|_| BackendError::InvalidArgument)?;
        let insts_len = usize::try_from(u64_at(28)).map_err(|_| BackendError::InvalidArgument)?;

        // Every section must lie within the container, and the counts must be nonzero and bounded.
        if inputs == 0 || inputs + outputs == 0 {
            return Err(BackendError::InvalidArgument);
        }
        if inputs.checked_add(outputs).is_none_or(|total| total > 256) {
            return Err(BackendError::InvalidArgument);
        }
        // HRX consumes the TXN stream as little-endian u32 words.
        if insts_len % 4 != 0 || insts_len == 0 || xclbin_len == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let entry_end = HEADER_LEN
            .checked_add(entry_len)
            .ok_or(BackendError::InvalidArgument)?;
        let xclbin_end = entry_end
            .checked_add(xclbin_len)
            .ok_or(BackendError::InvalidArgument)?;
        let insts_end = xclbin_end
            .checked_add(insts_len)
            .ok_or(BackendError::InvalidArgument)?;
        if insts_end != bytes.len() {
            return Err(BackendError::InvalidArgument);
        }
        let entry = core::str::from_utf8(&bytes[HEADER_LEN..entry_end])
            .map_err(|_| BackendError::InvalidArgument)?;
        if entry.is_empty() || entry.contains('\0') {
            return Err(BackendError::InvalidArgument);
        }
        Ok(Self {
            xclbin: &bytes[entry_end..xclbin_end],
            insts: &bytes[xclbin_end..insts_end],
            entry,
            inputs,
            outputs,
        })
    }
}

/// Build a container from its parts (for hosts and tools producing precompiled artifacts).
pub fn encode(entry: &str, inputs: u32, outputs: u32, xclbin: &[u8], insts: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + entry.len() + xclbin.len() + insts.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&inputs.to_le_bytes());
    out.extend_from_slice(&outputs.to_le_bytes());
    out.extend_from_slice(&(entry.len() as u32).to_le_bytes());
    out.extend_from_slice(&(xclbin.len() as u64).to_le_bytes());
    out.extend_from_slice(&(insts.len() as u64).to_le_bytes());
    out.extend_from_slice(entry.as_bytes());
    out.extend_from_slice(xclbin);
    out.extend_from_slice(insts);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_encode_and_parse() {
        let bytes = encode("MLIR_AIE", 1, 1, &[0xAB; 40], &[0x11; 8]);
        let parsed = PrecompiledArtifact::parse(&bytes).expect("valid container");
        assert_eq!(parsed.entry, "MLIR_AIE");
        assert_eq!((parsed.inputs, parsed.outputs), (1, 1));
        assert_eq!(parsed.xclbin, &[0xAB; 40]);
        assert_eq!(parsed.insts, &[0x11; 8]);
    }

    #[test]
    fn rejects_bad_magic_version_and_alignment() {
        assert!(matches!(
            PrecompiledArtifact::parse(b"nope"),
            Err(BackendError::InvalidArgument)
        ));
        let mut bytes = encode("MLIR_AIE", 1, 1, &[0u8; 4], &[0u8; 4]);
        bytes[4] = 2; // version
        assert!(matches!(
            PrecompiledArtifact::parse(&bytes),
            Err(BackendError::Incompatible)
        ));
        // insts length not a multiple of 4.
        let bad = encode("MLIR_AIE", 1, 1, &[0u8; 4], &[0u8; 6]);
        assert!(matches!(
            PrecompiledArtifact::parse(&bad),
            Err(BackendError::InvalidArgument)
        ));
    }

    #[test]
    fn rejects_truncated_container() {
        let bytes = encode("MLIR_AIE", 1, 1, &[0xAB; 40], &[0x11; 8]);
        assert!(matches!(
            PrecompiledArtifact::parse(&bytes[..bytes.len() - 4]),
            Err(BackendError::InvalidArgument)
        ));
    }
}

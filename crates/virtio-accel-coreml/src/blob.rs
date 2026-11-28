//! Core ML's binary weight file (MIL "blob storage", version 2).
//!
//! A constant written inline in an ML Program is turned into text and parsed
//! again by Core ML's compiler, which dominates load time once a graph holds
//! more than a few thousand weights. A large constant is stored here instead,
//! and the program refers to it by file and offset.
//!
//! Layout, little-endian, every entry starting on a 64-byte boundary:
//! a 64-byte header (`count: u32`, `version: u32 = 2`, zeros), then for each
//! entry a 64-byte description (`sentinel: u32 = 0xDEADBEEF`, `type: u32`,
//! `bytes: u64`, `data offset: u64`, `unused bits: u64`, zeros) followed by its
//! data. A reference names the offset of the description.

use crate::lower::LoweringError;

/// The name a program uses for the file; `@model_path` is the package's folder.
pub(crate) const FILE_NAME: &str = "@model_path/weights/weight.bin";
/// Constants smaller than this stay inline.
pub(crate) const INLINE_BELOW: usize = 4096;
const ALIGNMENT: usize = 64;
const SENTINEL: u32 = 0xDEAD_BEEF;
const VERSION: u32 = 2;

/// An element type the file can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Float16,
    Float32,
    Int8,
    Int4,
    UInt4,
    Int32,
}

impl Kind {
    const fn code(self) -> u32 {
        match self {
            Self::Float16 => 1,
            Self::Float32 => 2,
            Self::Int8 => 4,
            Self::Int4 => 8,
            Self::UInt4 => 11,
            Self::Int32 => 14,
        }
    }

    const fn bits(self) -> u64 {
        match self {
            Self::Int4 | Self::UInt4 => 4,
            Self::Int8 => 8,
            Self::Float16 => 16,
            Self::Float32 | Self::Int32 => 32,
        }
    }
}

/// A weight file being written.
#[derive(Default)]
pub(crate) struct Blobs {
    bytes: Vec<u8>,
    count: u32,
}

impl Blobs {
    /// Append `data`, `elements` values of `kind` packed low bits first, and
    /// return the offset a program refers to it by.
    pub(crate) fn push(
        &mut self,
        kind: Kind,
        elements: u64,
        data: &[u8],
    ) -> Result<u64, LoweringError> {
        let bits = elements
            .checked_mul(kind.bits())
            .ok_or(LoweringError::ResourceLimit)?;
        if bits.div_ceil(8) != data.len() as u64 {
            return Err(LoweringError::InvalidConstant);
        }
        if self.bytes.is_empty() {
            self.bytes.resize(ALIGNMENT, 0);
            self.bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        }
        let offset = self.bytes.len();
        let padded = data
            .len()
            .checked_next_multiple_of(ALIGNMENT)
            .ok_or(LoweringError::ResourceLimit)?;
        self.bytes
            .try_reserve(ALIGNMENT + padded)
            .map_err(|_| LoweringError::ResourceLimit)?;
        let mut entry = [0u8; ALIGNMENT];
        entry[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        entry[4..8].copy_from_slice(&kind.code().to_le_bytes());
        entry[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        entry[16..24].copy_from_slice(&((offset + ALIGNMENT) as u64).to_le_bytes());
        entry[24..32].copy_from_slice(&(data.len() as u64 * 8 - bits).to_le_bytes());
        self.bytes.extend_from_slice(&entry);
        self.bytes.extend_from_slice(data);
        self.bytes.resize(offset + ALIGNMENT + padded, 0);
        self.count = self
            .count
            .checked_add(1)
            .ok_or(LoweringError::ResourceLimit)?;
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        Ok(offset as u64)
    }

    /// The file, empty when nothing was stored in it.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    fn half(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn entries_are_aligned_described_and_counted() {
        let mut blobs = Blobs::default();
        let first = blobs.push(Kind::Float16, 3, &[1, 2, 3, 4, 5, 6]).unwrap();
        // Three four-bit values fill a byte and a half.
        let second = blobs.push(Kind::Int4, 3, &[0x21, 0x03]).unwrap();
        let file = blobs.finish();
        assert_eq!((first, second, file.len()), (64, 192, 320));
        assert_eq!((half(&file, 0), half(&file, 4)), (2, 2));
        assert_eq!(half(&file, 64), SENTINEL);
        assert_eq!(half(&file, 68), 1);
        assert_eq!(
            (word(&file, 72), word(&file, 80), word(&file, 88)),
            (6, 128, 0)
        );
        assert_eq!(&file[128..134], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(half(&file, 196), 8);
        assert_eq!(
            (word(&file, 200), word(&file, 208), word(&file, 216)),
            (2, 256, 4)
        );
        assert_eq!(&file[256..258], &[0x21, 0x03]);
        assert!(file[258..].iter().all(|b| *b == 0));
    }

    #[test]
    fn a_length_that_does_not_match_is_refused() {
        let mut blobs = Blobs::default();
        assert!(blobs.push(Kind::Float32, 2, &[0; 7]).is_err());
        assert!(blobs.push(Kind::Int4, 4, &[0; 3]).is_err());
        assert!(blobs.push(Kind::Int8, u64::MAX, &[0]).is_err());
        assert!(blobs.finish().is_empty());
    }
}

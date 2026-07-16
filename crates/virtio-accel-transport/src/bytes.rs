//! Transport-owned byte access that is independent of backend error semantics.

use core::fmt;

/// Failure while accessing a validated transport byte region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteAccessError {
    /// The requested logical range is outside the mapped region.
    OutOfBounds,
    /// Another live access currently owns the same bytes.
    Busy,
    /// Queue reset invalidated the region before access.
    Reset,
    /// The concrete transport can no longer access the mapped memory.
    Access,
}

/// Bounded readable bytes exposed by a transport implementation.
///
/// Implementations may be segmented. Every operation is nonblocking and allocation-free.
pub trait ReadableBytes: fmt::Debug {
    /// Stable logical byte length.
    fn len(&self) -> u64;

    /// Whether this source contains no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `target` from the exact logical range beginning at `offset`.
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError>;

    /// Borrow the complete source when the concrete mapping is contiguous and stable.
    fn as_contiguous(&self) -> Option<&[u8]> {
        None
    }
}

/// Bounded writable bytes exposed by a transport implementation.
///
/// Implementations may be segmented. Every operation is nonblocking and allocation-free.
pub trait WritableBytes: fmt::Debug {
    /// Stable logical byte length.
    fn len(&self) -> u64;

    /// Whether this destination contains no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write `source` to the exact logical range beginning at `offset`.
    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError>;

    /// Borrow the complete destination when the concrete mapping is contiguous and stable.
    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        None
    }
}

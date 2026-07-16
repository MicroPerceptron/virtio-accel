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

/// Driver-owned byte access for one unpublished or reclaimed descriptor chain.
///
/// The readable side is written by the driver and later read by the device. The writable side is
/// written by the device and may be read by the driver only after used-ring reclamation. Methods
/// are nonblocking and allocation-free; queue ownership prevents calls while a chain is published.
pub trait DriverChainBuffer: fmt::Debug {
    /// Concrete transport byte-access failure.
    type Error;

    /// Total bytes readable by the device.
    fn device_readable_len(&self) -> u64;

    /// Total bytes writable by the device.
    fn device_writable_len(&self) -> u64;

    /// Write an exact logical range into the device-readable side.
    fn write_device_readable(&mut self, offset: u64, source: &[u8]) -> Result<(), Self::Error>;

    /// Read an exact logical range from the device-writable side after completion.
    fn read_device_writable(&self, offset: u64, target: &mut [u8]) -> Result<(), Self::Error>;
}

impl ReadableBytes for [u8] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
        let end = start
            .checked_add(target.len())
            .ok_or(ByteAccessError::OutOfBounds)?;
        let source = self.get(start..end).ok_or(ByteAccessError::OutOfBounds)?;
        target.copy_from_slice(source);
        Ok(())
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self)
    }
}

impl WritableBytes for [u8] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
        let end = start
            .checked_add(source.len())
            .ok_or(ByteAccessError::OutOfBounds)?;
        let target = self
            .get_mut(start..end)
            .ok_or(ByteAccessError::OutOfBounds)?;
        target.copy_from_slice(source);
        Ok(())
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self)
    }
}

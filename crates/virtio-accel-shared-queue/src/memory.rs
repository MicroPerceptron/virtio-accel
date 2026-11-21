//! This side's access to the memory the two sides share.

/// Why an address did not name memory this side may touch that way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFault {
    /// Some byte of the range is not shared, or not for the access asked.
    Range,
    /// A word access named an address that is not a multiple of its size.
    Alignment,
}

/// What an access does to the bytes it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// The bytes are read.
    Read,
    /// The bytes are written.
    Write,
}

/// The memory a driver and a device share, as one side reaches it.
///
/// Addresses are **device addresses**: the numbers a driver writes into a
/// descriptor and into the three ring-address registers. Each side maps them
/// its own way — an offset into a window the driver lent, a guest-physical
/// address, an I/O virtual address resolved through the driver's own mapping
/// of its DMA memory — and that mapping, with whatever `unsafe` it needs, is
/// the implementation's. Nothing in this crate forms a reference into the
/// memory or holds an address across a call.
///
/// # What an implementation promises
///
/// - A range it does not [`admit`](Self::admits) is refused by every access,
///   with nothing read and nothing written.
/// - The other side may write any byte at any moment. [`read`](Self::read)
///   returns a snapshot; it need not be coherent across bytes, and it must
///   not be undefined behaviour for it to race a write.
/// - [`load_u16`](Self::load_u16) and [`store_u16`](Self::store_u16) are the
///   queue's publication points. A store has release ordering: every write
///   this side made before it is visible to the other side once it has loaded
///   the word. A load has acquire ordering. Words are little-endian whatever
///   the host is.
pub trait SharedMemory {
    /// Whether `len` bytes at `at` may be accessed as `access` asks. A range
    /// that wraps the address space is never admitted; an empty one is.
    fn admits(&self, at: u64, len: u64, access: Access) -> bool;

    /// Snapshot `target.len()` bytes at `at`.
    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault>;

    /// Store `source` at `at`.
    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault>;

    /// Load the little-endian `u16` at `at`, acquire.
    fn load_u16(&self, at: u64) -> Result<u16, MemoryFault>;

    /// Store a little-endian `u16` at `at`, release.
    fn store_u16(&self, at: u64, value: u16) -> Result<(), MemoryFault>;
}

impl<T: SharedMemory + ?Sized> SharedMemory for &T {
    fn admits(&self, at: u64, len: u64, access: Access) -> bool {
        (**self).admits(at, len, access)
    }

    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        (**self).read(at, target)
    }

    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault> {
        (**self).write(at, source)
    }

    fn load_u16(&self, at: u64) -> Result<u16, MemoryFault> {
        (**self).load_u16(at)
    }

    fn store_u16(&self, at: u64, value: u16) -> Result<(), MemoryFault> {
        (**self).store_u16(at, value)
    }
}

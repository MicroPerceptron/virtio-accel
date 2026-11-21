//! Where a split virtqueue's three parts are, and what is in them
//! (Virtio 1.2 §2.7).

use virtio_accel_transport::QueueSize;

use crate::memory::{Access, SharedMemory};

/// One descriptor: `addr: u64, len: u32, flags: u16, next: u16` (§2.7.5).
pub(crate) const DESCRIPTOR_BYTES: u64 = 16;
pub(crate) const DESCRIPTOR_LEN: usize = 8;
pub(crate) const DESCRIPTOR_FLAGS: usize = 12;
pub(crate) const DESCRIPTOR_NEXT: usize = 14;

/// The chain continues at `next`.
pub const VIRTQ_DESC_F_NEXT: u16 = 1 << 0;
/// The device writes this descriptor's bytes; otherwise it reads them.
pub const VIRTQ_DESC_F_WRITE: u16 = 1 << 1;
/// The descriptor names an indirect table: not in this profile.
pub const VIRTQ_DESC_F_INDIRECT: u16 = 1 << 2;

/// Both rings begin `flags: u16, idx: u16` (§2.7.6, §2.7.8).
pub(crate) const RING_FLAGS: u64 = 0;
pub(crate) const RING_INDEX: u64 = 2;
pub(crate) const RING_HEADER: u64 = 4;
/// An available-ring entry is a descriptor index.
pub(crate) const AVAILABLE_ENTRY: u64 = 2;
/// A used-ring entry is `id: u32, len: u32`.
pub(crate) const USED_ENTRY: u64 = 8;

/// The driver asks not to be interrupted when the device uses a chain.
pub const VIRTQ_AVAIL_F_NO_INTERRUPT: u16 = 1 << 0;
/// The device asks not to be notified when the driver publishes a chain.
pub const VIRTQ_USED_F_NO_NOTIFY: u16 = 1 << 0;

/// Alignments §2.7 requires of each part.
const DESCRIPTOR_ALIGN: u64 = 16;
const AVAILABLE_ALIGN: u64 = 2;
const USED_ALIGN: u64 = 4;

/// Where a driver put a queue's three parts, as device addresses: what it
/// writes to the transport's descriptor, driver and device area registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingAddresses {
    /// The descriptor table.
    pub descriptors: u64,
    /// The available ring, which the driver writes.
    pub available: u64,
    /// The used ring, which the device writes.
    pub used: u64,
}

impl RingAddresses {
    /// Bytes of descriptor table a queue of `size` has.
    pub const fn descriptor_table_bytes(size: QueueSize) -> u64 {
        size.get() as u64 * DESCRIPTOR_BYTES
    }

    /// Bytes of available ring a queue of `size` has. The trailing
    /// `used_event` word exists only with event indices, which this profile
    /// does not negotiate.
    pub const fn available_ring_bytes(size: QueueSize) -> u64 {
        RING_HEADER + size.get() as u64 * AVAILABLE_ENTRY
    }

    /// Bytes of used ring a queue of `size` has.
    pub const fn used_ring_bytes(size: QueueSize) -> u64 {
        RING_HEADER + size.get() as u64 * USED_ENTRY
    }

    /// The three parts one after another from `base`, each aligned as §2.7
    /// requires, and the bytes the whole takes. `None` when `base` is not
    /// aligned for a descriptor table or the queue would wrap the address
    /// space.
    pub const fn contiguous(base: u64, size: QueueSize) -> Option<(Self, u64)> {
        if base % DESCRIPTOR_ALIGN != 0 {
            return None;
        }
        // A descriptor table's length is a multiple of the available ring's
        // alignment, so only the used ring needs rounding up to its own.
        let Some(available) = base.checked_add(Self::descriptor_table_bytes(size)) else {
            return None;
        };
        let Some(end) = available.checked_add(Self::available_ring_bytes(size)) else {
            return None;
        };
        let Some(used) = end.checked_next_multiple_of(USED_ALIGN) else {
            return None;
        };
        let Some(end) = used.checked_add(Self::used_ring_bytes(size)) else {
            return None;
        };
        Some((
            Self {
                descriptors: base,
                available,
                used,
            },
            end - base,
        ))
    }

    /// Each part with its length, for a queue of `size`.
    const fn parts(&self, size: QueueSize) -> [(u64, u64); 3] {
        [
            (self.descriptors, Self::descriptor_table_bytes(size)),
            (self.available, Self::available_ring_bytes(size)),
            (self.used, Self::used_ring_bytes(size)),
        ]
    }

    /// Whether the three parts are aligned for their entries, do not wrap,
    /// do not overlap one another, and lie in memory that admits each as
    /// `access` says (descriptor table, available ring, used ring).
    pub(crate) fn hold<M: SharedMemory>(
        &self,
        memory: &M,
        size: QueueSize,
        access: [Access; 3],
    ) -> bool {
        let parts = self.parts(size);
        let aligned = self.descriptors % DESCRIPTOR_ALIGN == 0
            && self.available % AVAILABLE_ALIGN == 0
            && self.used % USED_ALIGN == 0;
        let disjoint =
            |a: (u64, u64), b: (u64, u64)| match (a.0.checked_add(a.1), b.0.checked_add(b.1)) {
                (Some(a_end), Some(b_end)) => a_end <= b.0 || b_end <= a.0,
                _ => false,
            };
        aligned
            && disjoint(parts[0], parts[1])
            && disjoint(parts[0], parts[2])
            && disjoint(parts[1], parts[2])
            && parts
                .iter()
                .zip(access)
                .all(|((at, len), access)| memory.admits(*at, *len, access))
    }
}

/// One descriptor's sixteen bytes.
pub(crate) fn descriptor(
    address: u64,
    len: u32,
    flags: u16,
    next: u16,
) -> [u8; DESCRIPTOR_BYTES as usize] {
    let mut bytes = [0u8; DESCRIPTOR_BYTES as usize];
    bytes[..DESCRIPTOR_LEN].copy_from_slice(&address.to_le_bytes());
    bytes[DESCRIPTOR_LEN..DESCRIPTOR_FLAGS].copy_from_slice(&len.to_le_bytes());
    bytes[DESCRIPTOR_FLAGS..DESCRIPTOR_NEXT].copy_from_slice(&flags.to_le_bytes());
    bytes[DESCRIPTOR_NEXT..].copy_from_slice(&next.to_le_bytes());
    bytes
}

/// The four fields of one descriptor's sixteen bytes.
pub(crate) fn fields(bytes: &[u8; DESCRIPTOR_BYTES as usize]) -> (u64, u32, u16, u16) {
    let word = |at: usize, len: usize| {
        let mut wide = [0u8; 8];
        wide[..len].copy_from_slice(&bytes[at..at + len]);
        u64::from_le_bytes(wide)
    };
    (
        word(0, 8),
        word(DESCRIPTOR_LEN, 4) as u32,
        word(DESCRIPTOR_FLAGS, 2) as u16,
        word(DESCRIPTOR_NEXT, 2) as u16,
    )
}

/// Visit the spans `len` bytes at `offset` fall in, out of `spans` taken as
/// one run of bytes: `(address, from, count)` with `from` the offset into the
/// caller's buffer. The caller has checked that the run holds them.
pub(crate) fn each_span<E>(
    spans: impl Iterator<Item = (u64, u32)>,
    offset: u64,
    len: usize,
    mut visit: impl FnMut(u64, usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    let (mut skip, mut done) = (offset, 0usize);
    for (at, span) in spans {
        let span = u64::from(span);
        if skip >= span {
            skip -= span;
            continue;
        }
        if done == len {
            break;
        }
        let count = ((span - skip) as usize).min(len - done);
        visit(at + skip, done, count)?;
        done += count;
        skip = 0;
    }
    Ok(())
}

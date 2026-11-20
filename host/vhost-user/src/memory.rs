use virtio_accel_transport::{ByteAccessError, ReadableBytes, WritableBytes};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Segment {
    pub address: u64,
    pub len: u64,
}

/// Metadata is snapshotted before execution. Shared guest RAM is accessed only through
/// vm-memory's volatile copy APIs, never exposed as an ordinary borrowed Rust slice.
#[derive(Debug)]
pub(crate) struct Port<'a> {
    pub memory: &'a GuestMemoryMmap,
    pub segments: &'a [Segment],
    pub len: u64,
}

impl Port<'_> {
    fn visit(
        &self,
        mut offset: u64,
        bytes: usize,
        mut copy: impl FnMut(GuestAddress, usize, usize) -> Result<(), ByteAccessError>,
    ) -> Result<(), ByteAccessError> {
        if offset
            .checked_add(bytes as u64)
            .filter(|end| *end <= self.len)
            .is_none()
        {
            return Err(ByteAccessError::OutOfBounds);
        }
        let mut done = 0;
        for segment in self.segments {
            if offset >= segment.len {
                offset -= segment.len;
                continue;
            }
            let count = (segment.len - offset).min((bytes - done) as u64) as usize;
            copy(GuestAddress(segment.address + offset), done, count)?;
            done += count;
            offset = 0;
            if done == bytes {
                return Ok(());
            }
        }
        if done == bytes {
            Ok(())
        } else {
            Err(ByteAccessError::OutOfBounds)
        }
    }
}

impl ReadableBytes for Port<'_> {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        self.visit(offset, target.len(), |addr, done, count| {
            self.memory
                .read_slice(&mut target[done..done + count], addr)
                .map_err(|_| ByteAccessError::Access)
        })
    }
}
impl WritableBytes for Port<'_> {
    fn len(&self) -> u64 {
        self.len
    }
    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        self.visit(offset, source.len(), |addr, done, count| {
            self.memory
                .write_slice(&source[done..done + count], addr)
                .map_err(|_| ByteAccessError::Access)
        })
    }
}

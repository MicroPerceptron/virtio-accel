//! Segmented byte ports over transport-neutral descriptor metadata.
//!
//! Each segmented port access scans at most the segment list and copies only the requested range;
//! constructors reject empty, zero-length, and overflowing segment collections.

use core::cmp::min;

use virtio_accel_core::{BackendError, ByteSink, ByteSource};
pub use virtio_accel_transport::{
    ChainLayout, ChainLayoutError, ChainRegion, RegionDirection, validate_chain_layout,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentedRegionError {
    Empty,
    ZeroLength,
    LengthOverflow,
}

/// Borrowed segmented source used by unit tests, fuzz targets, and simple transport adapters.
#[derive(Debug)]
pub struct SegmentedSource<'segments, 'bytes> {
    segments: &'segments [&'bytes [u8]],
    len: u64,
}

impl<'segments, 'bytes> SegmentedSource<'segments, 'bytes> {
    pub fn new(segments: &'segments [&'bytes [u8]]) -> Result<Self, SegmentedRegionError> {
        let len = checked_segment_len(segments.iter().map(|segment| segment.len()))?;
        Ok(Self { segments, len })
    }
}

impl ByteSource for SegmentedSource<'_, '_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        checked_range(offset, target.len(), self.len)?;
        if target.is_empty() {
            return Ok(());
        }

        let mut skip = offset;
        let mut written = 0;
        for segment in self.segments {
            let segment_len = segment.len() as u64;
            if skip >= segment_len {
                skip -= segment_len;
                continue;
            }

            let start = skip as usize;
            let count = min(segment.len() - start, target.len() - written);
            target[written..written + count].copy_from_slice(&segment[start..start + count]);
            written += count;
            skip = 0;
            if written == target.len() {
                return Ok(());
            }
        }

        Err(BackendError::OutOfBounds)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        (self.segments.len() == 1).then_some(self.segments[0])
    }
}

/// Borrowed segmented sink used by unit tests, fuzz targets, and simple transport adapters.
#[derive(Debug)]
pub struct SegmentedSink<'segments, 'bytes> {
    segments: &'segments mut [&'bytes mut [u8]],
    len: u64,
}

impl<'segments, 'bytes> SegmentedSink<'segments, 'bytes> {
    pub fn new(segments: &'segments mut [&'bytes mut [u8]]) -> Result<Self, SegmentedRegionError> {
        let len = checked_segment_len(segments.iter().map(|segment| segment.len()))?;
        Ok(Self { segments, len })
    }
}

impl ByteSink for SegmentedSink<'_, '_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        checked_range(offset, source.len(), self.len)?;
        if source.is_empty() {
            return Ok(());
        }

        let mut skip = offset;
        let mut read = 0;
        for segment in self.segments.iter_mut() {
            let segment = &mut **segment;
            let segment_len = segment.len() as u64;
            if skip >= segment_len {
                skip -= segment_len;
                continue;
            }

            let start = skip as usize;
            let count = min(segment.len() - start, source.len() - read);
            segment[start..start + count].copy_from_slice(&source[read..read + count]);
            read += count;
            skip = 0;
            if read == source.len() {
                return Ok(());
            }
        }

        Err(BackendError::OutOfBounds)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        if self.segments.len() == 1 {
            Some(&mut *self.segments[0])
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReadableRegion<'a> {
    source: &'a dyn ByteSource,
    offset: u64,
    len: u64,
}

impl<'a> ReadableRegion<'a> {
    pub fn new(source: &'a dyn ByteSource, offset: u64, len: u64) -> Result<Self, BackendError> {
        checked_range_u64(offset, len, source.len())?;
        Ok(Self {
            source,
            offset,
            len,
        })
    }
}

impl ByteSource for ReadableRegion<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        checked_range(offset, target.len(), self.len)?;
        let source_offset = self
            .offset
            .checked_add(offset)
            .ok_or(BackendError::OutOfBounds)?;
        self.source.read_at(source_offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        let start = usize::try_from(self.offset).ok()?;
        let len = usize::try_from(self.len).ok()?;
        let end = start.checked_add(len)?;
        self.source.as_contiguous()?.get(start..end)
    }
}

#[derive(Debug)]
pub struct WritableRegion<'a> {
    sink: &'a mut dyn ByteSink,
    offset: u64,
    len: u64,
}

impl<'a> WritableRegion<'a> {
    pub fn new(sink: &'a mut dyn ByteSink, offset: u64, len: u64) -> Result<Self, BackendError> {
        checked_range_u64(offset, len, sink.len())?;
        Ok(Self { sink, offset, len })
    }
}

impl ByteSink for WritableRegion<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        checked_range(offset, source.len(), self.len)?;
        let sink_offset = self
            .offset
            .checked_add(offset)
            .ok_or(BackendError::OutOfBounds)?;
        self.sink.write_at(sink_offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        let start = usize::try_from(self.offset).ok()?;
        let len = usize::try_from(self.len).ok()?;
        let end = start.checked_add(len)?;
        self.sink.as_contiguous_mut()?.get_mut(start..end)
    }
}

fn checked_segment_len(
    lengths: impl IntoIterator<Item = usize>,
) -> Result<u64, SegmentedRegionError> {
    let mut count = 0_usize;
    let mut total = 0_u64;
    for len in lengths {
        count += 1;
        if len == 0 {
            return Err(SegmentedRegionError::ZeroLength);
        }
        total = total
            .checked_add(len as u64)
            .ok_or(SegmentedRegionError::LengthOverflow)?;
    }
    if count == 0 {
        return Err(SegmentedRegionError::Empty);
    }
    Ok(total)
}

fn checked_range(offset: u64, bytes: usize, len: u64) -> Result<(), BackendError> {
    let bytes = u64::try_from(bytes).map_err(|_| BackendError::OutOfBounds)?;
    checked_range_u64(offset, bytes, len)
}

fn checked_range_u64(offset: u64, bytes: u64, len: u64) -> Result<(), BackendError> {
    let end = offset.checked_add(bytes).ok_or(BackendError::OutOfBounds)?;
    if end > len {
        return Err(BackendError::OutOfBounds);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmented_ports_cross_every_boundary() {
        let bytes = *b"segmented";
        for split in 1..bytes.as_slice().len() {
            let source_segments = [&bytes[..split], &bytes[split..]];
            let source = SegmentedSource::new(&source_segments).unwrap();
            let mut decoded = [0_u8; 9];
            source.read_at(0, &mut decoded).unwrap();
            assert_eq!(decoded, bytes);

            let mut first = [0_u8; 9];
            let (left, right) = first.split_at_mut(split);
            let mut sink_segments: [&mut [u8]; 2] = [left, right];
            let mut sink = SegmentedSink::new(&mut sink_segments).unwrap();
            sink.write_at(0, &bytes).unwrap();
            assert_eq!(first, bytes);
        }
    }

    #[test]
    fn subregions_preserve_bounds_and_contiguous_fast_paths() {
        let bytes = *b"01234567";
        let region = ReadableRegion::new(&bytes, 2, 4).unwrap();
        assert_eq!(region.as_contiguous(), Some(&b"2345"[..]));

        let mut output = [0_u8; 8];
        {
            let mut region = WritableRegion::new(&mut output, 2, 4).unwrap();
            assert_eq!(region.as_contiguous_mut().unwrap().len(), 4);
            region.write_at(0, b"abcd").unwrap();
        }
        assert_eq!(&output, b"\0\0abcd\0\0");
    }
}

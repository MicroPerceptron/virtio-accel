//! Bounded response framing over a potentially segmented destination.
//!
//! Callers can expose an exact payload subregion directly to a backend, then commit the 16-byte
//! response header only after the payload is initialized. The convenience streaming path uses one
//! fixed 4 KiB stack scratch buffer and never allocates or writes beyond the preflighted frame.

use virtio_accel_core::{ByteSink, ByteSource};
use virtio_accel_proto::{ResponseHeader, StatusCode};
use zerocopy::IntoBytes;

const RESPONSE_HEADER_BYTES: u64 = core::mem::size_of::<virtio_accel_proto::ResponseHeader>() as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseWriteError {
    FrameTooLarge,
    InsufficientCapacity,
    SourceAccess,
    SinkAccess,
}

/// Preflighted writer for one response frame.
pub struct ResponseWriter<'a> {
    sink: &'a mut dyn ByteSink,
    max_response_bytes: u32,
}

impl core::fmt::Debug for ResponseWriter<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResponseWriter")
            .field("capacity", &self.sink.len())
            .field("max_response_bytes", &self.max_response_bytes)
            .finish()
    }
}

impl<'a> ResponseWriter<'a> {
    pub fn new(sink: &'a mut dyn ByteSink, max_response_bytes: u32) -> Self {
        Self {
            sink,
            max_response_bytes,
        }
    }

    pub const fn max_response_bytes(&self) -> u32 {
        self.max_response_bytes
    }

    /// Borrow the exact success-payload destination.
    ///
    /// The returned guard commits the same payload length that was preflighted here, preventing a
    /// caller from initializing one range and advertising another in the response header.
    pub fn payload(
        &mut self,
        payload_bytes: u64,
    ) -> Result<ResponsePayload<'_>, ResponseWriteError> {
        let used = self.preflight(payload_bytes)?;
        Ok(ResponsePayload {
            sink: self.sink,
            payload_bytes: u32::try_from(payload_bytes)
                .map_err(|_| ResponseWriteError::FrameTooLarge)?,
            used,
        })
    }

    /// Write a bounded payload first, then atomically expose it by committing the header.
    pub fn write_response(
        &mut self,
        status: StatusCode,
        request_id: u64,
        payload: &dyn ByteSource,
    ) -> Result<u32, ResponseWriteError> {
        let payload_bytes = payload.len();
        let mut destination = self.payload(payload_bytes)?;

        if let Some(contiguous) = payload.as_contiguous() {
            destination
                .write_at(0, contiguous)
                .map_err(|_| ResponseWriteError::SinkAccess)?;
        } else {
            let mut scratch = [0_u8; 4096];
            let mut offset = 0_u64;
            while offset < payload_bytes {
                let remaining = payload_bytes - offset;
                let count = usize::try_from(remaining.min(scratch.as_slice().len() as u64))
                    .map_err(|_| ResponseWriteError::FrameTooLarge)?;
                payload
                    .read_at(offset, &mut scratch[..count])
                    .map_err(|_| ResponseWriteError::SourceAccess)?;
                destination
                    .write_at(offset, &scratch[..count])
                    .map_err(|_| ResponseWriteError::SinkAccess)?;
                offset += count as u64;
            }
        }

        destination.commit(status, request_id)
    }

    pub fn write_empty(
        &mut self,
        status: StatusCode,
        request_id: u64,
    ) -> Result<u32, ResponseWriteError> {
        let used = self.preflight(0)?;
        write_header(self.sink, status, request_id, 0)?;
        Ok(used)
    }

    fn preflight(&self, payload_bytes: u64) -> Result<u32, ResponseWriteError> {
        let total = RESPONSE_HEADER_BYTES
            .checked_add(payload_bytes)
            .ok_or(ResponseWriteError::FrameTooLarge)?;
        if total > u64::from(self.max_response_bytes) || total > u64::from(u32::MAX) {
            return Err(ResponseWriteError::FrameTooLarge);
        }
        if total > self.sink.len() {
            return Err(ResponseWriteError::InsufficientCapacity);
        }
        Ok(total as u32)
    }
}

/// Exact payload destination for one preflighted response.
pub struct ResponsePayload<'a> {
    sink: &'a mut dyn ByteSink,
    payload_bytes: u32,
    used: u32,
}

impl core::fmt::Debug for ResponsePayload<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResponsePayload")
            .field("payload_bytes", &self.payload_bytes)
            .field("used", &self.used)
            .finish()
    }
}

impl ResponsePayload<'_> {
    /// Commit the response header after the complete payload has been initialized.
    pub fn commit(self, status: StatusCode, request_id: u64) -> Result<u32, ResponseWriteError> {
        write_header(self.sink, status, request_id, self.payload_bytes)?;
        Ok(self.used)
    }
}

impl ByteSink for ResponsePayload<'_> {
    fn len(&self) -> u64 {
        u64::from(self.payload_bytes)
    }

    fn write_at(
        &mut self,
        offset: u64,
        source: &[u8],
    ) -> Result<(), virtio_accel_core::BackendError> {
        let bytes = u64::try_from(source.len())
            .map_err(|_| virtio_accel_core::BackendError::OutOfBounds)?;
        let end = offset
            .checked_add(bytes)
            .ok_or(virtio_accel_core::BackendError::OutOfBounds)?;
        if end > self.len() {
            return Err(virtio_accel_core::BackendError::OutOfBounds);
        }
        self.sink.write_at(RESPONSE_HEADER_BYTES + offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        let sink = self.sink.as_contiguous_mut()?;
        let end = RESPONSE_HEADER_BYTES
            .checked_add(u64::from(self.payload_bytes))
            .and_then(|end| usize::try_from(end).ok())?;
        sink.get_mut(RESPONSE_HEADER_BYTES as usize..end)
    }
}

fn write_header(
    sink: &mut dyn ByteSink,
    status: StatusCode,
    request_id: u64,
    payload_bytes: u32,
) -> Result<(), ResponseWriteError> {
    let header = ResponseHeader::new(status, payload_bytes, request_id);
    sink.write_at(0, header.as_bytes())
        .map_err(|_| ResponseWriteError::SinkAccess)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SegmentedSink, SegmentedSource};

    #[test]
    fn response_header_and_payload_cross_every_writable_split() {
        let payload = *b"response";
        let payload_segments = [&payload[..3], &payload[3..]];
        let payload_source = SegmentedSource::new(&payload_segments).unwrap();
        let expected_len = 16 + payload.as_slice().len();
        for split in 1..expected_len {
            let mut output = [0_u8; 24];
            let (left, right) = output.split_at_mut(split);
            let mut segments: [&mut [u8]; 2] = [left, right];
            let mut sink = SegmentedSink::new(&mut segments).unwrap();
            let mut writer = ResponseWriter::new(&mut sink, 1024);
            assert_eq!(
                writer
                    .write_response(StatusCode::OK, 0x0102_0304_0506_0708, &payload_source)
                    .unwrap(),
                expected_len as u32
            );

            assert_eq!(&output[0..2], &StatusCode::OK.0.to_le_bytes());
            assert_eq!(
                &output[4..8],
                &(payload.as_slice().len() as u32).to_le_bytes()
            );
            assert_eq!(&output[16..], &payload);
        }
    }

    #[test]
    fn direct_payload_region_does_not_touch_excess_capacity() {
        let mut output = [0xaa_u8; 32];
        let mut writer = ResponseWriter::new(&mut output, 32);
        let mut payload = writer.payload(4).unwrap();
        payload.write_at(0, b"data").unwrap();
        assert_eq!(payload.commit(StatusCode::OK, 7), Ok(20));
        assert_eq!(&output[16..20], b"data");
        assert_eq!(&output[20..], &[0xaa; 12]);
    }

    #[test]
    fn response_length_overflow_is_rejected_before_writing() {
        let mut output = [0xaa_u8; 16];
        let mut writer = ResponseWriter::new(&mut output, u32::MAX);
        assert_eq!(
            writer.payload(u64::MAX).unwrap_err(),
            ResponseWriteError::FrameTooLarge
        );
        assert_eq!(output, [0xaa; 16]);
    }

    #[test]
    fn payload_guard_commits_the_preflighted_length() {
        let mut output = [0xaa_u8; 24];
        let mut writer = ResponseWriter::new(&mut output, 24);
        let mut payload = writer.payload(8).unwrap();
        assert_eq!(payload.len(), 8);
        assert!(payload.write_at(7, &[1, 2]).is_err());
        payload.write_at(0, &[0; 8]).unwrap();
        assert_eq!(payload.commit(StatusCode::OK, 9), Ok(24));
        assert_eq!(&output[4..8], &8_u32.to_le_bytes());
    }
}

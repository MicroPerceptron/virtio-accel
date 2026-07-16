//! Bounded response framing over a potentially segmented destination.
//!
//! Callers can expose an exact payload subregion directly to a backend, then commit the 16-byte
//! response header only after the payload is initialized. The convenience streaming path uses one
//! fixed 4 KiB stack scratch buffer and never allocates or writes beyond the preflighted frame.

use virtio_accel_core::{ByteSink, ByteSource};
use virtio_accel_proto::StatusCode;

use crate::WritableRegion;

const RESPONSE_HEADER_BYTES: u64 = 16;

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

    pub fn payload_region(
        &mut self,
        payload_bytes: u64,
    ) -> Result<WritableRegion<'_>, ResponseWriteError> {
        self.preflight(payload_bytes)?;
        WritableRegion::new(self.sink, RESPONSE_HEADER_BYTES, payload_bytes)
            .map_err(|_| ResponseWriteError::InsufficientCapacity)
    }

    /// Commit the response header after its payload has been initialized.
    pub fn commit(
        &mut self,
        status: StatusCode,
        request_id: u64,
        payload_bytes: u32,
    ) -> Result<u32, ResponseWriteError> {
        let used = self.preflight(u64::from(payload_bytes))?;
        let mut header = [0_u8; RESPONSE_HEADER_BYTES as usize];
        header[0..2].copy_from_slice(&status.0.to_le_bytes());
        header[2..4].copy_from_slice(&0_u16.to_le_bytes());
        header[4..8].copy_from_slice(&payload_bytes.to_le_bytes());
        header[8..16].copy_from_slice(&request_id.to_le_bytes());
        self.sink
            .write_at(0, &header)
            .map_err(|_| ResponseWriteError::SinkAccess)?;
        Ok(used)
    }

    /// Write a bounded payload first, then atomically expose it by committing the header.
    pub fn write_response(
        &mut self,
        status: StatusCode,
        request_id: u64,
        payload: &dyn ByteSource,
    ) -> Result<u32, ResponseWriteError> {
        let payload_bytes = payload.len();
        self.preflight(payload_bytes)?;

        if let Some(contiguous) = payload.as_contiguous() {
            self.sink
                .write_at(RESPONSE_HEADER_BYTES, contiguous)
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
                self.sink
                    .write_at(RESPONSE_HEADER_BYTES + offset, &scratch[..count])
                    .map_err(|_| ResponseWriteError::SinkAccess)?;
                offset += count as u64;
            }
        }

        self.commit(
            status,
            request_id,
            u32::try_from(payload_bytes).map_err(|_| ResponseWriteError::FrameTooLarge)?,
        )
    }

    pub fn write_empty(
        &mut self,
        status: StatusCode,
        request_id: u64,
    ) -> Result<u32, ResponseWriteError> {
        self.commit(status, request_id, 0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SegmentedSink;

    #[test]
    fn response_header_and_payload_cross_every_writable_split() {
        let payload = *b"response";
        let expected_len = 16 + payload.as_slice().len();
        for split in 1..expected_len {
            let mut output = [0_u8; 24];
            let (left, right) = output.split_at_mut(split);
            let mut segments: [&mut [u8]; 2] = [left, right];
            let mut sink = SegmentedSink::new(&mut segments).unwrap();
            let mut writer = ResponseWriter::new(&mut sink, 1024);
            assert_eq!(
                writer
                    .write_response(StatusCode::OK, 0x0102_0304_0506_0708, &payload)
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
        {
            let mut payload = writer.payload_region(4).unwrap();
            payload.write_at(0, b"data").unwrap();
        }
        assert_eq!(writer.commit(StatusCode::OK, 7, 4), Ok(20));
        assert_eq!(&output[16..20], b"data");
        assert_eq!(&output[20..], &[0xaa; 12]);
    }

    #[test]
    fn response_length_overflow_is_rejected_before_writing() {
        let mut output = [0xaa_u8; 16];
        let mut writer = ResponseWriter::new(&mut output, u32::MAX);
        assert_eq!(
            writer.payload_region(u64::MAX).unwrap_err(),
            ResponseWriteError::FrameTooLarge
        );
        assert_eq!(output, [0xaa; 16]);
    }
}

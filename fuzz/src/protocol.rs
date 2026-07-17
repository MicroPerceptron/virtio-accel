use virtio_accel_cleanroom::decode_request as cleanroom_decode;
use virtio_accel_core::{Accelerator, TransportByteSource};
use virtio_accel_device::{DecoderLimits, FrameDecodeError, FrameDecoder, SegmentedSource};
use virtio_accel_mock::MockAccelerator;
use virtio_accel_proto::{
    BASELINE_COMMAND_QUEUES, Le16, Le32, PROTOCOL_MAJOR, PROTOCOL_MINOR, WireConfig,
};

use crate::{Input, MAX_FRAME_BYTES};

const CONTROL_BYTES: usize = 8;
const RESPONSE_LIMIT: u64 = MAX_FRAME_BYTES as u64;

pub fn fuzz_protocol_decode(data: &[u8]) {
    let data = &data[..data.len().min(MAX_FRAME_BYTES + CONTROL_BYTES)];
    let control_len = data.len().min(CONTROL_BYTES);
    let (controls, frame) = data.split_at(control_len);
    let mut input = Input::new(controls);
    let response_capacity = u64::from(input.u16()) % (RESPONSE_LIMIT + 1);
    let segments = segment(frame, &mut input);

    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(32),
        max_request_bytes: Le32::new(MAX_FRAME_BYTES as u32),
        max_response_bytes: Le32::new(RESPONSE_LIMIT as u32),
    };
    let info = MockAccelerator::default().device_info().unwrap();
    let decoder = FrameDecoder::new(DecoderLimits::new(&config, info).unwrap());

    let full = observe_decode(&decoder, frame, &segments, RESPONSE_LIMIT);
    let contiguous_full = observe_decode(&decoder, frame, &[], RESPONSE_LIMIT);
    assert_eq!(
        full, contiguous_full,
        "segmented and contiguous decode disagreed"
    );
    let clean = cleanroom_decode(frame);

    if let Ok(decoded) = &full {
        let clean = clean
            .as_ref()
            .expect("primary accepted a frame rejected by clean-room codec");
        assert_eq!(decoded.request_id, clean.request_id);
        assert_eq!(decoded.opcode, clean.body.opcode() as u16);

        let mut encoded = vec![0_u8; clean.encoded_len().unwrap()];
        let written = clean.encode(&mut encoded).unwrap();
        assert_eq!(written, frame.len());
        assert_eq!(encoded, frame);
    }

    let limited = observe_decode(&decoder, frame, &segments, response_capacity);
    let contiguous_limited = observe_decode(&decoder, frame, &[], response_capacity);
    assert_eq!(
        limited, contiguous_limited,
        "segmented and contiguous capacity decode disagreed"
    );
    if let Ok(decoded) = full {
        match limited {
            Ok(limited) => {
                assert!(response_capacity >= decoded.required_response_bytes);
                assert_eq!(limited.request_id, decoded.request_id);
                assert_eq!(
                    limited.required_response_bytes,
                    decoded.required_response_bytes
                );
            }
            Err(FrameDecodeError::Unrecoverable(_)) => assert!(response_capacity < 16),
            Err(FrameDecodeError::InsufficientResponse {
                request_id,
                required,
                available,
            }) => {
                assert_eq!(request_id, decoded.request_id);
                assert_eq!(required, decoded.required_response_bytes);
                assert_eq!(available, response_capacity);
                assert!((16..required).contains(&response_capacity));
            }
            Err(error) => panic!("capacity changed semantic decoding: {error:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DecodeObservation {
    request_id: u64,
    opcode: u16,
    required_response_bytes: u64,
}

fn observe_decode(
    decoder: &FrameDecoder,
    frame: &[u8],
    segments: &[&[u8]],
    response_capacity: u64,
) -> Result<DecodeObservation, FrameDecodeError> {
    if segments.is_empty() {
        let source = TransportByteSource::new(frame);
        let decoded = decoder.decode(&source, response_capacity)?;
        Ok(DecodeObservation {
            request_id: decoded.request_id(),
            opcode: decoded.body().opcode() as u16,
            required_response_bytes: u64::from(decoded.required_response_bytes()),
        })
    } else {
        let source = SegmentedSource::new(segments).unwrap();
        let decoded = decoder.decode(&source, response_capacity)?;
        Ok(DecodeObservation {
            request_id: decoded.request_id(),
            opcode: decoded.body().opcode() as u16,
            required_response_bytes: u64::from(decoded.required_response_bytes()),
        })
    }
}

fn segment<'a>(bytes: &'a [u8], input: &mut Input<'_>) -> Vec<&'a [u8]> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::with_capacity(CONTROL_BYTES + 1);
    let mut offset = 0;
    while offset < bytes.len() {
        let requested = usize::from(input.byte()).saturating_add(1);
        let end = offset.saturating_add(requested).min(bytes.len());
        segments.push(&bytes[offset..end]);
        offset = end;
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_proto::{KnownOpcode, RequestFlags, RequestHeader};
    use zerocopy::IntoBytes;

    #[test]
    fn canonical_frame_survives_arbitrary_segmentation() {
        let header = RequestHeader::new(KnownOpcode::GetDeviceInfo, RequestFlags::empty(), 0, 1);
        let mut input = vec![0, 0, 1, 2, 3, 4, 5, 6];
        input.extend_from_slice(header.as_bytes());
        fuzz_protocol_decode(&input);
    }
}

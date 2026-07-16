//! Atomic preflight for one transport-neutral command frame.
//!
//! This is the only byte-oriented entry point needed by the command engine. It validates the
//! flattened chain shape, confirms that the byte ports match the advertised totals, decodes the
//! complete request, and writes recoverable protocol errors before returning. Only
//! [`FramePreflight::Ready`] contains a request that semantic dispatch may act on.

use virtio_accel_core::{ByteSink, ByteSource};
use virtio_accel_proto::StatusCode;

use crate::{
    ChainLayoutError, ChainRegion, DecodedRequest, FrameDecodeError, FrameDecoder,
    ResponseWriteError, ResponseWriter, UnrecoverableDecodeError, validate_chain_layout,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnusableFrame {
    ChainLayout(ChainLayoutError),
    Request(UnrecoverableDecodeError),
    InsufficientResponse {
        request_id: u64,
        required: u64,
        available: u64,
    },
}

#[derive(Debug)]
pub enum FramePreflight<'a> {
    /// The complete frame is validated and may proceed to semantic dispatch.
    Ready(DecodedRequest<'a>),
    /// A recoverable protocol error was written to the response port.
    Rejected {
        request_id: u64,
        status: StatusCode,
        used: u32,
    },
    /// No response bytes were written and the chain must complete with used length zero.
    Unusable(UnusableFrame),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePreflightError {
    ResponseWrite(ResponseWriteError),
}

/// Validate one complete chain before semantic state or a backend can be reached.
///
/// The function is allocation-free except for the bounded `SUBMIT` binding allocation performed
/// by [`FrameDecoder`]. `Ready` leaves the response untouched. `Rejected` initializes exactly one
/// 16-byte error header. `Unusable` leaves the response untouched.
pub fn preflight_command_frame<'a>(
    decoder: &FrameDecoder,
    regions: &[ChainRegion],
    request: &'a dyn ByteSource,
    response: &mut dyn ByteSink,
) -> Result<FramePreflight<'a>, FramePreflightError> {
    let layout = match validate_chain_layout(regions, decoder.limits().max_chain_descriptors()) {
        Ok(layout) => layout,
        Err(error) => {
            return Ok(FramePreflight::Unusable(UnusableFrame::ChainLayout(error)));
        }
    };
    if let Err(error) = layout.validate_port_lengths(request.len(), response.len()) {
        return Ok(FramePreflight::Unusable(UnusableFrame::ChainLayout(error)));
    }

    match decoder.decode(request, response.len()) {
        Ok(request) => Ok(FramePreflight::Ready(request)),
        Err(FrameDecodeError::Protocol { request_id, status }) => {
            let used = ResponseWriter::new(response, decoder.limits().max_response_bytes())
                .write_empty(status, request_id)
                .map_err(FramePreflightError::ResponseWrite)?;
            Ok(FramePreflight::Rejected {
                request_id,
                status,
                used,
            })
        }
        Err(FrameDecodeError::Unrecoverable(error)) => {
            Ok(FramePreflight::Unusable(UnusableFrame::Request(error)))
        }
        Err(FrameDecodeError::InsufficientResponse {
            request_id,
            required,
            available,
        }) => Ok(FramePreflight::Unusable(
            UnusableFrame::InsufficientResponse {
                request_id,
                required,
                available,
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::{DecoderLimits, SegmentedSink, SegmentedSource};
    use serde_json::Value;
    use std::vec::Vec;
    use virtio_accel_core::{
        AcceleratorClass, Capabilities, DeviceIdentity, DeviceInfo, DeviceLimits,
    };
    use virtio_accel_proto::{
        BASELINE_COMMAND_QUEUES, HARD_MAX_BINDINGS, HARD_MAX_REQUEST_BYTES,
        HARD_MAX_RESPONSE_BYTES, Le16, Le32, PROTOCOL_MAJOR, PROTOCOL_MINOR, WireConfig,
    };

    const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

    fn corpus() -> Value {
        serde_json::from_str(include_str!("../../../conformance/v1.0/vectors.json")).unwrap()
    }

    fn vector(section: &str, name: &str) -> Vec<u8> {
        let corpus = corpus();
        let entry = corpus[section]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"] == name)
            .unwrap();
        decode_hex(entry["hex"].as_str().unwrap())
    }

    fn decode_hex(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as char).to_digit(16).unwrap() as u8;
                let low = (pair[1] as char).to_digit(16).unwrap() as u8;
                (high << 4) | low
            })
            .collect()
    }

    fn decoder() -> FrameDecoder {
        let config = WireConfig {
            protocol_major: Le16::new(PROTOCOL_MAJOR),
            protocol_minor: Le16::new(PROTOCOL_MINOR),
            command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
            max_chain_descriptors: Le16::new(16),
            max_request_bytes: Le32::new(HARD_MAX_REQUEST_BYTES),
            max_response_bytes: Le32::new(HARD_MAX_RESPONSE_BYTES),
        };
        let info = DeviceInfo {
            identity: DeviceIdentity {
                uuid: [0; 16],
                class: AcceleratorClass::OTHER,
                vendor_id: 0,
                device_id: 0,
            },
            capabilities: Capabilities::HOST_VISIBLE_MEMORY
                | Capabilities::DEVICE_LOCAL_MEMORY
                | Capabilities::SHARED_MEMORY,
            limits: DeviceLimits {
                max_contexts: 64,
                max_buffers_per_context: 1024,
                max_programs_per_context: 256,
                max_queues_per_context: 16,
                max_events_per_context: 4096,
                max_bindings_per_submission: HARD_MAX_BINDINGS,
                max_buffer_bytes: 1 << 30,
                max_artifact_bytes: 1 << 28,
            },
        };
        FrameDecoder::new(DecoderLimits::new(&config, info).unwrap())
    }

    #[test]
    fn valid_frame_is_dispatchable_without_touching_response() {
        let request = vector("frames", "request_get_device_info");
        let request_segments = [&request[..7], &request[7..]];
        let request_source = SegmentedSource::new(&request_segments).unwrap();
        let mut response = [0xaa_u8; 92];
        let (left, right) = response.split_at_mut(31);
        let mut response_segments: [&mut [u8]; 2] = [left, right];
        let mut response_sink = SegmentedSink::new(&mut response_segments).unwrap();
        let regions = [
            ChainRegion::readable(7),
            ChainRegion::readable((request.len() - 7) as u64),
            ChainRegion::writable(31),
            ChainRegion::writable(61),
        ];

        let outcome =
            preflight_command_frame(&decoder(), &regions, &request_source, &mut response_sink)
                .unwrap();
        let FramePreflight::Ready(decoded) = outcome else {
            panic!("valid frame was not dispatchable");
        };
        assert_eq!(decoded.request_id(), REQUEST_ID);
        assert_eq!(response, [0xaa; 92]);
    }

    #[test]
    fn recoverable_malformed_frame_writes_exact_error_header() {
        let request = vector("edge_cases", "request_reserved_flag");
        let request_segments = [request.as_slice()];
        let request_source = SegmentedSource::new(&request_segments).unwrap();
        let mut response = [0xaa_u8; 24];
        let (left, right) = response.split_at_mut(5);
        let mut response_segments: [&mut [u8]; 2] = [left, right];
        let mut response_sink = SegmentedSink::new(&mut response_segments).unwrap();
        let regions = [
            ChainRegion::readable(request.len() as u64),
            ChainRegion::writable(5),
            ChainRegion::writable(19),
        ];

        assert!(matches!(
            preflight_command_frame(&decoder(), &regions, &request_source, &mut response_sink,),
            Ok(FramePreflight::Rejected {
                request_id: REQUEST_ID,
                status: StatusCode::UNSUPPORTED,
                used: 16,
            })
        ));
        assert_eq!(&response[0..2], &StatusCode::UNSUPPORTED.0.to_le_bytes());
        assert_eq!(&response[4..8], &0_u32.to_le_bytes());
        assert_eq!(&response[8..16], &REQUEST_ID.to_le_bytes());
        assert_eq!(&response[16..], &[0xaa; 8]);
    }

    #[test]
    fn unusable_frames_write_nothing_and_never_become_dispatchable() {
        let request = vector("frames", "request_read_buffer");
        let request_segments = [request.as_slice()];
        let request_source = SegmentedSource::new(&request_segments).unwrap();
        let mut response = [0xaa_u8; 19];
        let mut response_segments: [&mut [u8]; 1] = [&mut response];
        let mut response_sink = SegmentedSink::new(&mut response_segments).unwrap();
        let regions = [
            ChainRegion::readable(request.len() as u64),
            ChainRegion::writable(19),
        ];

        assert!(matches!(
            preflight_command_frame(&decoder(), &regions, &request_source, &mut response_sink,),
            Ok(FramePreflight::Unusable(
                UnusableFrame::InsufficientResponse { .. }
            ))
        ));
        assert_eq!(response, [0xaa; 19]);

        let mut response = [0xaa_u8; 20];
        let mut response_segments: [&mut [u8]; 1] = [&mut response];
        let mut response_sink = SegmentedSink::new(&mut response_segments).unwrap();
        let invalid_regions = [
            ChainRegion::writable(20),
            ChainRegion::readable(request.len() as u64),
        ];
        assert!(matches!(
            preflight_command_frame(
                &decoder(),
                &invalid_regions,
                &request_source,
                &mut response_sink,
            ),
            Ok(FramePreflight::Unusable(UnusableFrame::ChainLayout(
                ChainLayoutError::Direction
            )))
        ));
        assert_eq!(response, [0xaa; 20]);
    }
}

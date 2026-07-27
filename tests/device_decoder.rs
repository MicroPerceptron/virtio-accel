//! Frame-decoder assertions against the frozen `conformance/v1.0` golden vectors.
//!
//! These assertions live in the facade rather than in `virtio-accel-device` because the golden
//! vectors are shipped by this package. A crate cannot `include_str!` data from outside its own
//! package directory and still have that data present in its published `.crate`.

use serde_json::Value;
use virtio_accel::core::*;
use virtio_accel::device::*;
use virtio_accel::proto::*;

const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

fn corpus() -> Value {
    serde_json::from_str(include_str!("../conformance/v1.0/vectors.json")).unwrap()
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
        max_chain_descriptors: Le16::new(HARD_MAX_CHAIN_DESCRIPTORS),
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

fn opcode(name: &str) -> KnownOpcode {
    match name {
        "GET_DEVICE_INFO" => KnownOpcode::GetDeviceInfo,
        "CREATE_CONTEXT" => KnownOpcode::CreateContext,
        "DESTROY_CONTEXT" => KnownOpcode::DestroyContext,
        "ALLOCATE_BUFFER" => KnownOpcode::AllocateBuffer,
        "FREE_BUFFER" => KnownOpcode::FreeBuffer,
        "WRITE_BUFFER" => KnownOpcode::WriteBuffer,
        "READ_BUFFER" => KnownOpcode::ReadBuffer,
        "LOAD_PROGRAM" => KnownOpcode::LoadProgram,
        "UNLOAD_PROGRAM" => KnownOpcode::UnloadProgram,
        "CREATE_QUEUE" => KnownOpcode::CreateQueue,
        "DESTROY_QUEUE" => KnownOpcode::DestroyQueue,
        "SUBMIT" => KnownOpcode::Submit,
        "POLL_EVENT" => KnownOpcode::PollEvent,
        "CANCEL_EVENT" => KnownOpcode::CancelEvent,
        "DESTROY_EVENT" => KnownOpcode::DestroyEvent,
        other => panic!("unknown opcode name {other}"),
    }
}

#[test]
fn every_canonical_request_decodes_across_every_byte_split() {
    let decoder = decoder();
    let corpus = corpus();
    for frame in corpus["frames"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|frame| frame["kind"] == "request")
    {
        let bytes = decode_hex(frame["hex"].as_str().unwrap());
        let expected_opcode = opcode(frame["opcode"].as_str().unwrap());

        let contiguous_segments = [bytes.as_slice()];
        let contiguous = SegmentedSource::new(&contiguous_segments).unwrap();
        let decoded = decoder
            .decode(&contiguous, u64::from(HARD_MAX_RESPONSE_BYTES))
            .unwrap();
        assert_eq!(decoded.request_id(), REQUEST_ID);
        assert_eq!(decoded.body().opcode(), expected_opcode);

        for split in 1..bytes.len() {
            let segments = [&bytes[..split], &bytes[split..]];
            let segmented = SegmentedSource::new(&segments).unwrap();
            let decoded = decoder
                .decode(&segmented, u64::from(HARD_MAX_RESPONSE_BYTES))
                .unwrap_or_else(|error| {
                    panic!(
                        "{} failed at split {split}: {error:?}",
                        frame["name"].as_str().unwrap()
                    )
                });
            assert_eq!(decoded.request_id(), REQUEST_ID);
            assert_eq!(decoded.body().opcode(), expected_opcode);
        }
    }
}

#[test]
fn transfer_and_artifact_tails_remain_borrowed_segmented_sources() {
    let decoder = decoder();

    let write = vector("frames", "request_write_buffer");
    let write_segments = [&write[..41], &write[41..43], &write[43..]];
    let write_source = SegmentedSource::new(&write_segments).unwrap();
    let decoded = decoder
        .decode(&write_source, u64::from(HARD_MAX_RESPONSE_BYTES))
        .unwrap();
    let DecodedRequestBody::WriteBuffer { data, range, .. } = decoded.body() else {
        panic!("write request decoded as wrong body");
    };
    assert_eq!(range.bytes(), 4);
    assert!(data.as_contiguous().is_none());
    let mut write_payload = [0_u8; 4];
    data.read_at(0, &mut write_payload).unwrap();
    assert_eq!(write_payload, [0xde, 0xad, 0xbe, 0xef]);

    let load = vector("frames", "request_load_program");
    let load_segments = [&load[..97], &load[97..99], &load[99..]];
    let load_source = SegmentedSource::new(&load_segments).unwrap();
    let decoded = decoder
        .decode(&load_source, u64::from(HARD_MAX_RESPONSE_BYTES))
        .unwrap();
    let DecodedRequestBody::LoadProgram {
        payload,
        resident_bytes,
        ..
    } = decoded.body()
    else {
        panic!("load request decoded as wrong body");
    };
    assert_eq!(*resident_bytes, 4096);
    let mut artifact = [0_u8; 4];
    payload.read_at(0, &mut artifact).unwrap();
    assert_eq!(artifact, [0xa1, 0xb2, 0xc3, 0xd4]);
}

#[test]
fn malformed_conformance_vectors_map_to_defined_failures() {
    let decoder = decoder();
    let cases = [
        (
            "request_unknown_opcode",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::UNSUPPORTED,
            },
        ),
        (
            "request_reserved_flag",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::UNSUPPORTED,
            },
        ),
        (
            "request_trailing_byte",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::INVALID_ARGUMENT,
            },
        ),
        (
            "request_reserved_context_flag",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::UNSUPPORTED,
            },
        ),
        (
            "request_unknown_memory_domain",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::INVALID_ARGUMENT,
            },
        ),
        (
            "request_binding_count_overflow",
            FrameDecodeError::Protocol {
                request_id: REQUEST_ID,
                status: StatusCode::RESOURCE_LIMIT,
            },
        ),
    ];

    for (name, expected) in cases {
        let bytes = vector("edge_cases", name);
        let segments = [bytes.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        match decoder.decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)) {
            Err(actual) => assert_eq!(actual, expected, "{name}"),
            Ok(_) => panic!("{name} unexpectedly decoded"),
        }
    }

    let truncated = vector("edge_cases", "request_header_truncated");
    let segments = [truncated.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    assert!(matches!(
        decoder.decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
        Err(FrameDecodeError::Unrecoverable(
            UnrecoverableDecodeError::RequestHeader
        ))
    ));
}

#[test]
fn duplicate_submission_slots_are_rejected_before_dispatch() {
    let mut request = vector("frames", "request_submit");
    let binding = request[48..80].to_vec();
    request[4..8].copy_from_slice(&96_u32.to_le_bytes());
    request[32..36].copy_from_slice(&2_u32.to_le_bytes());
    request.extend_from_slice(&binding);

    let segments = [request.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    assert!(matches!(
        decoder().decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
        Err(FrameDecodeError::Protocol {
            status: StatusCode::INVALID_ARGUMENT,
            ..
        })
    ));
}

#[test]
fn transfer_range_overflow_is_a_protocol_error() {
    let mut request = vector("frames", "request_read_buffer");
    request[24..32].copy_from_slice(&u64::MAX.to_le_bytes());

    let segments = [request.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    assert!(matches!(
        decoder().decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
        Err(FrameDecodeError::Protocol {
            status: StatusCode::INVALID_ARGUMENT,
            ..
        })
    ));
}

#[test]
fn short_success_capacity_writes_nothing() {
    let request = vector("frames", "request_read_buffer");
    let segments = [request.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    match decoder().decode(&source, 19) {
        Err(actual) => assert_eq!(
            actual,
            FrameDecodeError::InsufficientResponse {
                request_id: REQUEST_ID,
                required: 20,
                available: 19,
            }
        ),
        Ok(_) => panic!("short response capacity unexpectedly decoded"),
    }
}

#[test]
fn malformed_request_needs_only_error_header_capacity() {
    let request = vector("edge_cases", "request_reserved_context_flag");
    let segments = [request.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    assert!(matches!(
        decoder().decode(&source, 16),
        Err(FrameDecodeError::Protocol {
            status: StatusCode::UNSUPPORTED,
            ..
        })
    ));
}

#[test]
fn unsupported_memory_domain_capability_is_rejected_before_backend_use() {
    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(8),
        max_request_bytes: Le32::new(1024),
        max_response_bytes: Le32::new(1024),
    };
    let info = DeviceInfo {
        identity: DeviceIdentity {
            uuid: [0; 16],
            class: AcceleratorClass::OTHER,
            vendor_id: 0,
            device_id: 0,
        },
        capabilities: Capabilities::HOST_VISIBLE_MEMORY,
        limits: DeviceLimits {
            max_contexts: 1,
            max_buffers_per_context: 1,
            max_programs_per_context: 1,
            max_queues_per_context: 1,
            max_events_per_context: 1,
            max_bindings_per_submission: 1,
            max_buffer_bytes: 1 << 20,
            max_artifact_bytes: 1 << 20,
        },
    };
    let decoder = FrameDecoder::new(DecoderLimits::new(&config, info).unwrap());
    let request = vector("frames", "request_allocate_buffer");
    let segments = [request.as_slice()];
    let source = SegmentedSource::new(&segments).unwrap();
    assert!(matches!(
        decoder.decode(&source, 1024),
        Err(FrameDecodeError::Protocol {
            status: StatusCode::UNSUPPORTED,
            ..
        })
    ));
}

//! Independent clean-room codec assertions against the frozen `conformance/v1.0` golden vectors.
//!
//! These assertions live in the facade rather than in `virtio-accel-cleanroom` because the golden
//! vectors are shipped by this package. A crate cannot `include_str!` data from outside its own
//! package directory and still have that data present in its published `.crate`. Keeping them here
//! also leaves `virtio-accel-cleanroom` with no dependencies of any kind, normal or development.

use serde_json::Value;
use std::collections::BTreeSet;
use virtio_accel_cleanroom::{
    Binding, Bindings, Error, EventKind, EventState, Opcode, Request, RequestBody, Response,
    ResponseBody, ResponseContext, Submit, TransferBuffer, decode_config, decode_request,
    decode_response, validate_features,
};

const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

fn corpus() -> Value {
    serde_json::from_str(include_str!("../conformance/v1.0/vectors.json")).unwrap()
}

fn decode_hex(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    (0..hex.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
        .collect()
}

fn vector(group: &str, name: &str) -> Vec<u8> {
    let corpus = corpus();
    let entry = corpus[group]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("missing vector {name}"));
    decode_hex(entry["hex"].as_str().unwrap())
}

fn response_context(name: &str) -> ResponseContext {
    let opcode = match name {
        "response_get_device_info" => Some(Opcode::GetDeviceInfo),
        "response_create_context" => Some(Opcode::CreateContext),
        "response_destroy_context" => Some(Opcode::DestroyContext),
        "response_allocate_buffer" => Some(Opcode::AllocateBuffer),
        "response_free_buffer" => Some(Opcode::FreeBuffer),
        "response_write_buffer" => Some(Opcode::WriteBuffer),
        "response_read_buffer" => Some(Opcode::ReadBuffer),
        "response_load_program" => Some(Opcode::LoadProgram),
        "response_unload_program" => Some(Opcode::UnloadProgram),
        "response_create_queue" => Some(Opcode::CreateQueue),
        "response_destroy_queue" => Some(Opcode::DestroyQueue),
        "response_submit_accepted" | "response_submit_indeterminate" => Some(Opcode::Submit),
        name if name.starts_with("response_poll_event_") => Some(Opcode::PollEvent),
        "response_cancel_event" => Some(Opcode::CancelEvent),
        "response_destroy_event" => Some(Opcode::DestroyEvent),
        "response_unknown_opcode" => None,
        other => panic!("no response context for {other}"),
    };
    ResponseContext {
        opcode,
        request_id: REQUEST_ID,
        read_bytes: (opcode == Some(Opcode::ReadBuffer)).then_some(4),
    }
}

#[test]
fn every_canonical_frame_round_trips_through_the_independent_codec() {
    let corpus = corpus();
    let mut request_opcodes = BTreeSet::new();
    let mut response_names = BTreeSet::new();
    let mut config_count = 0;

    for frame in corpus["frames"].as_array().unwrap() {
        let name = frame["name"].as_str().unwrap();
        let bytes = decode_hex(frame["hex"].as_str().unwrap());
        let mut output = vec![0xa5; bytes.len()];
        let written = match frame["kind"].as_str().unwrap() {
            "config" => {
                config_count += 1;
                decode_config(&bytes, 256)
                    .unwrap()
                    .encode(256, &mut output)
                    .unwrap()
            }
            "request" => {
                let request = decode_request(&bytes).unwrap();
                request_opcodes.insert(request.body.opcode() as u16);
                request.encode(&mut output).unwrap()
            }
            "response" => {
                response_names.insert(name);
                let context = response_context(name);
                decode_response(&bytes, context)
                    .unwrap()
                    .encode(context, &mut output)
                    .unwrap()
            }
            other => panic!("unknown frame kind {other}"),
        };
        assert_eq!(written, bytes.len(), "{name}");
        assert_eq!(output, bytes, "{name}");
    }

    assert_eq!(config_count, 2);
    assert_eq!(request_opcodes.len(), 15);
    assert_eq!(response_names.len(), 20);
}

#[test]
fn independent_semantics_match_the_reviewed_values() {
    let submit_bytes = vector("frames", "request_submit");
    let submit = decode_request(&submit_bytes).unwrap();
    let RequestBody::Submit(submit) = submit.body else {
        panic!("SUBMIT vector decoded as the wrong command");
    };
    assert_eq!(submit.queue_id, 0x4142_4344_4546_4748);
    assert_eq!(submit.program_id, 0x3132_3334_3536_3738);
    assert_eq!(submit.timeout_ns, 1_000_000);
    assert_eq!(submit.bindings.count(), Ok(1));
    assert_eq!(
        submit.bindings.get(0),
        Ok(Binding {
            buffer_id: 0x2122_2324_2526_2728,
            offset: 0,
            bytes: 4_096,
            slot: 7,
            access: 3,
        })
    );

    let info_context = response_context("response_get_device_info");
    let info_bytes = vector("frames", "response_get_device_info");
    let info = decode_response(&info_bytes, info_context).unwrap();
    let ResponseBody::DeviceInfo(info) = info.body else {
        panic!("device-info response decoded as the wrong payload");
    };
    assert_eq!(&info.uuid, b"virtio-accelmock");
    assert_eq!(info.class, 1);
    assert_eq!(info.vendor_id, 0x1234_5678);
    assert_eq!(info.device_id, 0x9abc_def0);
    assert_eq!(info.max_bindings_per_submission, 256);

    let failed_context = response_context("response_poll_event_failed");
    let failed_bytes = vector("frames", "response_poll_event_failed");
    let failed = decode_response(&failed_bytes, failed_context).unwrap();
    assert_eq!(
        failed.body,
        ResponseBody::EventState(EventState {
            kind: EventKind::Failed,
            error: 9,
        })
    );
}

#[test]
fn reviewed_edge_vectors_receive_the_normative_classification() {
    assert_eq!(
        decode_config(
            &vector("edge_cases", "config_descriptor_limit_above_hard_max"),
            256,
        ),
        Err(Error::ChainDescriptorLimit)
    );
    assert_eq!(
        decode_config(&vector("frames", "config_maximum"), 128),
        Err(Error::ChainDescriptorLimit)
    );

    let future =
        decode_config(&vector("edge_cases", "config_future_minor_compatible"), 256).unwrap();
    assert!(future.protocol_minor > virtio_accel_cleanroom::PROTOCOL_MINOR);

    assert_eq!(
        decode_config(&vector("edge_cases", "config_unknown_major"), 256),
        Err(Error::Version)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_header_truncated")),
        Err(Error::Size)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_unknown_opcode")),
        Err(Error::Unsupported)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_reserved_flag")),
        Err(Error::Unsupported)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_trailing_byte")),
        Err(Error::InvalidArgument)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_reserved_context_flag")),
        Err(Error::Unsupported)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_unknown_memory_domain")),
        Err(Error::InvalidArgument)
    );
    assert_eq!(
        decode_request(&vector("edge_cases", "request_binding_count_overflow")),
        Err(Error::ResourceLimit)
    );

    let unknown_status = vector("edge_cases", "response_unknown_status");
    let context = ResponseContext {
        opcode: None,
        request_id: REQUEST_ID,
        read_bytes: None,
    };
    let response = decode_response(&unknown_status, context).unwrap();
    assert_eq!(response.status, 0x1234);
    assert_eq!(response.body, ResponseBody::Empty);

    let unknown_event = vector("edge_cases", "response_unknown_event_state");
    let context = ResponseContext {
        opcode: Some(Opcode::PollEvent),
        request_id: REQUEST_ID,
        read_bytes: None,
    };
    assert_eq!(
        decode_response(&unknown_event, context),
        Err(Error::RecoveryRequired)
    );
}

#[test]
fn encoder_rejects_values_that_cannot_cross_the_wire() {
    assert_eq!(validate_features(1), Err(Error::Unsupported));

    let zero_object = Request {
        request_id: REQUEST_ID,
        body: RequestBody::DestroyContext { context_id: 0 },
    };
    assert_eq!(zero_object.encoded_len(), Err(Error::InvalidArgument));

    let overflowing_transfer = Request {
        request_id: REQUEST_ID,
        body: RequestBody::ReadBuffer(TransferBuffer {
            buffer_id: 1,
            offset: u64::MAX,
            bytes: 1,
        }),
    };
    assert_eq!(
        overflowing_transfer.encoded_len(),
        Err(Error::InvalidArgument)
    );

    let duplicate_bindings = [
        Binding {
            buffer_id: 1,
            offset: 0,
            bytes: 4,
            slot: 7,
            access: 1,
        },
        Binding {
            buffer_id: 2,
            offset: 0,
            bytes: 4,
            slot: 7,
            access: 2,
        },
    ];
    let duplicate_submit = Request {
        request_id: REQUEST_ID,
        body: RequestBody::Submit(Submit {
            queue_id: 3,
            program_id: 4,
            timeout_ns: 0,
            bindings: Bindings::Values(&duplicate_bindings),
        }),
    };
    assert_eq!(duplicate_submit.encoded_len(), Err(Error::InvalidArgument));

    let invalid_event = Response {
        status: 0,
        request_id: REQUEST_ID,
        body: ResponseBody::EventState(EventState {
            kind: EventKind::Failed,
            error: 0,
        }),
    };
    let event_context = ResponseContext {
        opcode: Some(Opcode::PollEvent),
        request_id: REQUEST_ID,
        read_bytes: None,
    };
    assert_eq!(
        invalid_event.encoded_len(event_context),
        Err(Error::InvalidArgument)
    );

    let mut mismatched = vector("frames", "response_create_context");
    mismatched[8] ^= 1;
    assert_eq!(
        decode_response(
            &mismatched,
            ResponseContext {
                opcode: Some(Opcode::CreateContext),
                request_id: REQUEST_ID,
                read_bytes: None,
            },
        ),
        Err(Error::InvalidArgument)
    );
}
